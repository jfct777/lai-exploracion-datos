#!/usr/bin/env python3
"""Annotate allele multiplicity in the supplied, pre-normalization VCF/BCF.

Records sharing CHROM/POS are grouped before this pipeline splits their ALT
alleles. The union of their REF/ALT strings defines ORIG_NALLELES. This also
recognizes separately represented ALT alleles already present in the input;
it cannot recover alleles removed before that input was produced. Neither
annotation certifies the earlier variant-calling history or modifies genotypes.

Production execution belongs to the M01 Nextflow process. Memory is bounded
by one coordinate group, plus the set of previously encountered chromosomes.
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


def annotate(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Write annotated records without an index; return (record count, site count)."""
    import pysam

    mode = output_mode(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    n_records = n_sites = 0
    with pysam.VariantFile(str(input_path)) as reader:
        declared = {kind: frozenset(getattr(reader.header, kind))
                    for kind in HEADER_DEFINITION_KINDS}
        header = annotated_header(reader.header)
        # Exclusive creation also protects against an output appearing after
        # the check above. A failed run retains no misleading partial output.
        with output_path.open("xb") as output_handle:
            try:
                with pysam.VariantFile(output_handle, mode, header=header) as writer:
                    for group in iter_site_groups(reader):
                        validate_header_definitions(reader.header, declared)
                        allele_count = site_allele_count(group)
                        site = f"{group[0].contig}|{group[0].pos}"
                        for record in group:
                            if any(tag in record.info for tag in (INFO_ALLELE_COUNT, INFO_SITE)):
                                raise ValueError("Input already contains original-allele annotations")
                            annotated = record.copy()
                            annotated.translate(header)
                            annotated.info[INFO_ALLELE_COUNT] = allele_count
                            annotated.info[INFO_SITE] = site
                            writer.write(annotated)
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
    args = parser.parse_args()
    records, sites = annotate(args.input, args.output)
    print(f"Annotated {records} records at {sites} supplied-input sites")


if __name__ == "__main__":
    main()
