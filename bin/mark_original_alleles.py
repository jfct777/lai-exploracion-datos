#!/usr/bin/env python3
"""Annotate allele multiplicity in the supplied, pre-normalization VCF/BCF.

Records sharing CHROM/POS are grouped before this pipeline splits their ALT
alleles. The union of their REF/ALT strings defines ORIG_NALLELES. This also
recognizes separately represented ALT alleles already present in the input;
it cannot recover alleles removed before that input was produced. Neither
annotation certifies the earlier variant-calling history or modifies genotypes.

Production execution belongs to the M01 Nextflow process. Memory is bounded
by one coordinate group, plus the set of previously encountered chromosomes.
The optional CPU budget parallelizes BGZF I/O only; grouping and annotation
remain ordered and serial. No variant or chromosome is partitioned.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pysam


MARKER_KEY = "dnabr_original_alleles"
MARKER_VERSION = "v1"
INFO_ALLELE_COUNT = "ORIG_NALLELES"
INFO_SITE = "ORIG_SITE"
HEADER_DEFINITION_KINDS = ("contigs", "info", "formats", "filters")


def output_mode(path: Path) -> str:
    """Select an explicit pysam mode; compressed VCF is BGZF, not plain gzip."""
    if path.name.endswith(".bcf"):
        return "wb"
    if path.name.endswith(".vcf.gz"):
        return "wz"
    if path.name.endswith(".vcf"):
        return "w"
    raise ValueError("Output must end in .bcf, .vcf.gz or .vcf")


def io_thread_allocation(total_threads: int, compressed_output: bool) -> tuple[int, int]:
    """Return pysam (reader, writer) thread arguments within a total budget.

    In the pinned pysam 0.23.3, an argument of one is serial. An argument
    of n > 1 creates n - 1 BGZF workers plus one htslib I/O thread. Count
    those I/O threads as well as the single Python annotation thread:
    total = 1 + sum(n for n in (reader, writer) if n > 1).

    Budgets 1 and 2 therefore remain serial. For compressed output and
    budgets 3 or 4, prioritize compression. At >= 5 split the remaining
    budget between both streams, giving an odd extra worker to writing.
    Plain VCF output needs no compressor, so its spare budget goes to
    the reader. An uncompressed input may leave reader capacity unused.
    This is an upper bound, not a promise that every CPU will stay busy.
    """
    if type(total_threads) is not int or total_threads < 1:
        raise ValueError("threads must be an integer >= 1 (total CPU budget)")
    if total_threads < 3:
        return 1, 1
    if not compressed_output:
        return total_threads - 1, 1
    if total_threads < 5:
        return 1, total_threads - 1
    reader_threads = (total_threads - 1) // 2
    return reader_threads, total_threads - 1 - reader_threads


def positive_threads(value: str) -> int:
    """Validate the CLI CPU budget before opening either genomic file."""
    try:
        threads = int(value)
        io_thread_allocation(threads, compressed_output=True)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--threads must be an integer >= 1") from error
    return threads


def iter_site_groups(
    records: Iterable[pysam.VariantRecord],
) -> Iterator[list[pysam.VariantRecord]]:
    """Group adjacent coordinates, rejecting descending POS or chromosome reuse."""
    seen_chromosomes: set[str] = set()
    previous: tuple[str, int] | None = None
    group: list[pysam.VariantRecord] = []
    for record in records:
        key = (record.contig, record.pos)
        if not key[0] or key[1] < 1:
            raise ValueError("Each record must have a chromosome and positive POS")
        if previous is None or key[0] != previous[0]:
            if key[0] in seen_chromosomes:
                raise ValueError(f"Chromosome is revisited: {key[0]}")
            seen_chromosomes.add(key[0])
        elif key[1] < previous[1]:
            raise ValueError(f"Positions are not sorted within chromosome {key[0]}")
        if group and key != previous:
            yield group
            group = []
        group.append(record)
        previous = key
    if group:
        yield group


def site_allele_count(records: Sequence[pysam.VariantRecord]) -> int:
    """Count distinct allele strings, retaining symbolic alleles and '*' as alleles."""
    if not records:
        raise ValueError("Cannot annotate an empty coordinate group")
    first = records[0]
    coordinate = (first.contig, first.pos)
    if not first.ref or first.ref == ".":
        raise ValueError(f"Missing REF at {first.contig}:{first.pos}")
    alleles = {first.ref}
    for record in records:
        if (record.contig, record.pos) != coordinate:
            raise ValueError("A coordinate group contains multiple CHROM/POS values")
        if record.ref != first.ref:
            raise ValueError(f"Inconsistent REF at {first.contig}:{first.pos}")
        if not record.alts or any(not alt or alt == "." for alt in record.alts):
            raise ValueError(f"Missing ALT at {first.contig}:{first.pos}")
        alleles.update(record.alts)
    return len(alleles)


def annotated_header(header: pysam.VariantHeader) -> pysam.VariantHeader:
    """Copy the header and refuse to recertify pre-existing provenance."""
    if any(record.key == MARKER_KEY for record in header.records):
        raise ValueError(f"Input already declares {MARKER_KEY}")
    for tag in (INFO_ALLELE_COUNT, INFO_SITE):
        if tag in header.info:
            raise ValueError(f"Input already declares INFO/{tag}")
    result = header.copy()
    result.info.add(
        INFO_ALLELE_COUNT,
        1,
        "Integer",
        "Distinct REF/ALT strings at CHROM/POS in the supplied input before "
        "pipeline normalization; does not certify earlier calling history",
    )
    result.info.add(
        INFO_SITE,
        1,
        "String",
        "CHROM|POS in the supplied input before pipeline normalization",
    )
    result.add_meta(MARKER_KEY, value=MARKER_VERSION)
    return result


def validate_header_definitions(
    header: pysam.VariantHeader, declared: dict[str, frozenset[str]]
) -> None:
    """Reject undefined fields that htslib adds while parsing malformed VCF.

    In particular, END is exposed as record.stop rather than a regular INFO
    key, so checking record.info alone would miss an undeclared END. Comparing
    the header dictionaries also catches undeclared contigs, FORMAT and FILTER.
    Their lengths normally stay fixed; sets are built only upon a mismatch.
    """
    for kind, original in declared.items():
        current = getattr(header, kind)
        if len(current) != len(original):
            unknown = sorted(set(current) - original)
            raise ValueError(f"Input contains undeclared {kind}: {', '.join(unknown)}")


def annotate(input_path: Path, output_path: Path, threads: int = 1) -> tuple[int, int]:
    """Write annotated records without an index; return (record count, site count)."""
    import pysam

    mode = output_mode(output_path)
    reader_threads, writer_threads = io_thread_allocation(threads, mode != "w")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    n_records = n_sites = 0
    with pysam.VariantFile(str(input_path), threads=reader_threads) as reader:
        declared = {kind: frozenset(getattr(reader.header, kind))
                    for kind in HEADER_DEFINITION_KINDS}
        header = annotated_header(reader.header)
        # Exclusive creation also protects against an output appearing after
        # the check above. A failed run retains no misleading partial output.
        with output_path.open("xb") as output_handle:
            try:
                with pysam.VariantFile(output_handle, mode, header=header,
                                       threads=writer_threads) as writer:
                    for group in iter_site_groups(reader):
                        validate_header_definitions(reader.header, declared)
                        allele_count = site_allele_count(group)
                        site = f"{group[0].contig}|{group[0].pos}"
                        for record in group:
                            if any(tag in record.info for tag in (INFO_ALLELE_COUNT, INFO_SITE)):
                                raise ValueError("Input already contains original-allele annotations")
                            # Each iterator result owns its record buffer and
                            # is consumed exactly once. Translate that record
                            # to the copied output header instead of duplicating
                            # all sample fields. This does not mutate the input
                            # file or reader.header; translate remains required
                            # because the new INFO definitions live in header.
                            record.translate(header)
                            record.info[INFO_ALLELE_COUNT] = allele_count
                            record.info[INFO_SITE] = site
                            writer.write(record)
                            n_records += 1
                        n_sites += 1
            except BaseException:
                output_path.unlink()
                raise
    return n_records, n_sites


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--threads", type=positive_threads, default=1,
        help="Total CPU budget including annotation and BGZF I/O threads (default: 1)",
    )
    args = parser.parse_args()
    records, sites = annotate(args.input, args.output, threads=args.threads)
    print(f"Annotated {records} records at {sites} supplied-input sites")


if __name__ == "__main__":
    main()
