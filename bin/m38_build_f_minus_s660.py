#!/usr/bin/env python3
"""Build exact F-minus-S660 FLARE inputs from authenticated M34 VCFs.

The transformation is deliberately narrow.  It reads only the M34 reference
and target VCFs plus ``m34_selected_loci.npz``; it has no interface for local
ancestry labels, predictions, or scores.  Variants are identified exclusively
by the explicit ``CHROM/POS/REF/ALT`` tuple.  Every retained VCF row is copied
byte-for-byte at the text level and both outputs are reopened before the
append-safe directory is promoted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, TextIO

import numpy as np

from _experiment_invariants import validate_exact_locus_partition


SELECTED_SCHEMA = {"chrom", "pos", "ref", "alt", "cM", "locus_id"}
OUTPUT_NAMES = {
    "reference": "m38_f_minus_s660_reference.chr22.vcf",
    "target": "m38_f_minus_s660_target.chr22.vcf",
    "receipt": "m38_f_minus_s660_filter.receipt.json",
}
HEX_DIGITS = frozenset("0123456789abcdef")


class FMinusS660ContractError(ValueError):
    """Raised when an input or an exact-partition invariant differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FMinusS660ContractError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"input is not a regular file: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256(value: str, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS),
        f"{label} expected SHA-256 is malformed",
    )
    return value


def canonical_chromosome(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("ascii")
    else:
        text = str(value)
    return text.removeprefix("chr")


def allele_text(value: object, label: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("ascii")
    else:
        text = str(value)
    require(text in {"A", "C", "G", "T"}, f"{label} is not an uppercase SNV allele")
    return text


VariantKey = tuple[str, int, str, str]


def variant_key(
    chrom_value: object,
    position_value: object,
    ref_value: object,
    alt_value: object,
    *,
    expected_chromosome: str,
    label: str,
) -> VariantKey:
    chrom = canonical_chromosome(chrom_value)
    require(chrom == expected_chromosome, f"{label} chromosome differs: {chrom}")
    if isinstance(position_value, (bytes, np.bytes_)):
        try:
            position_value = position_value.decode("ascii")
        except UnicodeDecodeError:
            raise FMinusS660ContractError(f"{label} position is invalid") from None
    if isinstance(position_value, (int, np.integer)) and not isinstance(position_value, bool):
        position = int(position_value)
    elif isinstance(position_value, str) and position_value.isascii() \
            and position_value.isdecimal():
        position = int(position_value)
    else:
        raise FMinusS660ContractError(f"{label} position is invalid")
    require(position > 0, f"{label} position is invalid")
    ref = allele_text(ref_value, f"{label} REF")
    alt = allele_text(alt_value, f"{label} ALT")
    require(ref != alt, f"{label} REF and ALT are identical")
    return chrom, position, ref, alt


def axis_sha256(keys: Sequence[VariantKey], domain: bytes) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for chrom, position, ref, alt in keys:
        digest.update(f"{chrom}\t{position}\t{ref}\t{alt}\n".encode("ascii"))
    return digest.hexdigest()


def sample_axis_sha256(samples: Sequence[str], domain: bytes) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for sample in samples:
        digest.update(sample.encode("utf-8") + b"\n")
    return digest.hexdigest()


def lines_sha256(lines: Sequence[str], domain: bytes) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for line in lines:
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def open_vcf_text(path: Path) -> Iterator[TextIO]:
    """Open a plain or gzip/BGZF VCF without normalizing line endings."""
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle


@dataclass(frozen=True)
class VcfHeader:
    lines: tuple[str, ...]
    samples: tuple[str, ...]


@dataclass(frozen=True)
class VcfRow:
    key: VariantKey
    raw_axis: tuple[str, str, str, str]
    line: str


def read_vcf_header(handle: TextIO, label: str) -> VcfHeader:
    lines: list[str] = []
    for line_number, line in enumerate(handle, 1):
        require(line.endswith("\n"), f"{label} header line {line_number} lacks a newline")
        if line.startswith("##"):
            lines.append(line)
            continue
        if line.startswith("#CHROM"):
            fields = line.rstrip("\r\n").split("\t")
            require(
                len(fields) >= 10
                and fields[:9]
                == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"],
                f"{label} #CHROM header differs",
            )
            samples = tuple(fields[9:])
            require(
                samples
                and all(sample and sample == sample.strip() for sample in samples)
                and len(samples) == len(set(samples)),
                f"{label} sample axis is empty, padded, or duplicated",
            )
            lines.append(line)
            return VcfHeader(tuple(lines), samples)
        raise FMinusS660ContractError(f"{label} malformed header at line {line_number}")
    raise FMinusS660ContractError(f"{label} lacks a #CHROM header")


def require_m34_roles(
    reference_header: VcfHeader,
    target_header: VcfHeader,
    *,
    split: str,
) -> dict[str, str]:
    """Authenticate the biological roles embedded by the M34 bridge."""
    donor_role = "SOURCE_VALID" if split == "FIT" else "SOURCE_TEST"
    expected = {
        "reference": "##m34_bridge_vcf_role=REFERENCE_REF_TRAIN\n",
        "target": f"##m34_bridge_vcf_role=TARGET_{donor_role}_MOSAICS\n",
        "reference_role": "##m34_reference_and_frequency_role=REF_TRAIN\n",
        "donor_role": f"##m34_mosaic_donor_role_upstream={donor_role}\n",
    }
    require(reference_header.lines.count(expected["reference"]) == 1,
            "reference M34 role header differs")
    require(target_header.lines.count(expected["target"]) == 1,
            "target M34 role header differs")
    for label, header in (("reference", reference_header), ("target", target_header)):
        require(header.lines.count(expected["reference_role"]) == 1,
                f"{label} M34 reference/frequency role header differs")
        require(header.lines.count(expected["donor_role"]) == 1,
                f"{label} M34 upstream donor role header differs")
    return {
        "reference_and_frequency_role": "REF_TRAIN",
        "target_partition": split,
        "upstream_mosaic_donor_role": donor_role,
    }


def iter_vcf_rows(
    handle: TextIO,
    header: VcfHeader,
    *,
    label: str,
    expected_chromosome: str,
) -> Iterator[VcfRow]:
    previous_position = -1
    seen: set[VariantKey] = set()
    line_number = len(header.lines) + 1
    for line in handle:
        require(line.endswith("\n"), f"{label} row {line_number} lacks a newline")
        require(line.strip() and not line.startswith("#"), f"{label} row {line_number} is invalid")
        fields = line.rstrip("\r\n").split("\t")
        require(
            len(fields) == 9 + len(header.samples),
            f"{label} row {line_number} width differs from its sample axis",
        )
        key = variant_key(
            fields[0], fields[1], fields[3], fields[4],
            expected_chromosome=expected_chromosome,
            label=f"{label} row {line_number}",
        )
        require(key not in seen, f"{label} contains duplicate variant {key}")
        require(
            key[1] >= previous_position,
            f"{label} variant positions are not ordered",
        )
        seen.add(key)
        previous_position = key[1]
        yield VcfRow(key, (fields[0], fields[1], fields[3], fields[4]), line)
        line_number += 1


def load_selected_axis(
    path: Path,
    *,
    expected_chromosome: str,
    expected_count: int,
) -> tuple[VariantKey, ...]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == SELECTED_SCHEMA, "selected-loci NPZ schema differs")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, FMinusS660ContractError):
            raise
        raise FMinusS660ContractError("selected-loci NPZ cannot be read safely") from exc

    require(expected_count > 0, "expected selected-loci count must be positive")
    for name, array in arrays.items():
        require(array.ndim == 1, f"selected-loci {name} axis is not one-dimensional")
        require(array.shape[0] == expected_count, f"selected-loci {name} count differs")

    keys = tuple(
        variant_key(
            arrays["chrom"][index], arrays["pos"][index],
            arrays["ref"][index], arrays["alt"][index],
            expected_chromosome=expected_chromosome,
            label=f"selected-loci row {index}",
        )
        for index in range(expected_count)
    )
    require(len(keys) == len(set(keys)), "selected-loci contains duplicate variants")
    require(
        all(left[1] <= right[1] for left, right in zip(keys, keys[1:])),
        "selected-loci axis is not ordered",
    )
    return keys


def _hash_rows(rows: Sequence[str]) -> str:
    digest = hashlib.sha256(b"M38_VCF_RECORD_PAYLOAD_V1\0")
    for row in rows:
        digest.update(row.encode("utf-8"))
    return digest.hexdigest()


def audit_output_vcf(
    path: Path,
    *,
    label: str,
    expected_chromosome: str,
    expected_header: VcfHeader,
    expected_axis: Sequence[VariantKey],
    expected_record_sha256: str,
) -> dict[str, object]:
    with open_vcf_text(path) as handle:
        header = read_vcf_header(handle, label)
        require(header.lines == expected_header.lines, f"{label} header was not preserved exactly")
        require(header.samples == expected_header.samples, f"{label} sample axis drifted")
        rows = tuple(
            iter_vcf_rows(
                handle, header, label=label,
                expected_chromosome=expected_chromosome,
            )
        )
    observed_axis = tuple(row.key for row in rows)
    require(observed_axis == tuple(expected_axis), f"{label} F-minus-S660 locus axis drifted")
    observed_record_sha256 = _hash_rows(tuple(row.line for row in rows))
    require(
        observed_record_sha256 == expected_record_sha256,
        f"{label} retained VCF rows changed",
    )
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "header_sha256": lines_sha256(header.lines, b"M38_VCF_HEADER_V1"),
        "sample_axis_sha256": sample_axis_sha256(header.samples, b"M38_SAMPLE_AXIS_V1"),
        "record_payload_sha256": observed_record_sha256,
    }


def build_f_minus_s660(
    *,
    split: str,
    reference_vcf: Path,
    target_vcf: Path,
    selected_loci: Path,
    expected_reference_sha256: str,
    expected_target_sha256: str,
    expected_selected_sha256: str,
    expected_chromosome: str,
    expected_full_count: int,
    expected_selected_count: int,
    expected_reference_samples: int,
    expected_target_samples: int,
    outdir: Path,
) -> dict[str, object]:
    """Filter one authenticated REF/TARGET pair and return its receipt."""
    require(split in {"FIT", "VALID"}, "split must be FIT or VALID")
    expected_chromosome = canonical_chromosome(expected_chromosome)
    require(expected_chromosome == "22", "M38 F-minus-S660 pilot is frozen to chromosome 22")
    require(expected_full_count > expected_selected_count > 0, "expected locus counts are invalid")
    require(expected_reference_samples > 0 and expected_target_samples > 0,
            "expected sample counts must be positive")
    inputs = {
        "reference_vcf": reference_vcf,
        "target_vcf": target_vcf,
        "selected_loci": selected_loci,
    }
    require(len({path.resolve() for path in inputs.values()}) == 3,
            "input paths must be distinct")
    expected_hashes = {
        "reference_vcf": validate_sha256(expected_reference_sha256, "reference VCF"),
        "target_vcf": validate_sha256(expected_target_sha256, "target VCF"),
        "selected_loci": validate_sha256(expected_selected_sha256, "selected loci"),
    }
    observed_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    for name in inputs:
        require(observed_hashes[name] == expected_hashes[name], f"SHA-256 mismatch for {name}")
    require(not outdir.exists(), "refusing to overwrite the F-minus-S660 output directory")
    outdir.parent.mkdir(parents=True, exist_ok=True)

    selected_axis = load_selected_axis(
        selected_loci,
        expected_chromosome=expected_chromosome,
        expected_count=expected_selected_count,
    )
    selected_set = set(selected_axis)
    stage = Path(tempfile.mkdtemp(prefix=f".{outdir.name}.stage.", dir=outdir.parent))
    try:
        reference_output = stage / OUTPUT_NAMES["reference"]
        target_output = stage / OUTPUT_NAMES["target"]
        full_axis: list[VariantKey] = []
        fminus_axis: list[VariantKey] = []
        removed_axis: list[VariantKey] = []
        reference_fminus_rows: list[str] = []
        target_fminus_rows: list[str] = []
        source_contig_label: str | None = None

        with open_vcf_text(reference_vcf) as reference_handle, \
                open_vcf_text(target_vcf) as target_handle:
            reference_header = read_vcf_header(reference_handle, "reference")
            target_header = read_vcf_header(target_handle, "target")
            roles = require_m34_roles(reference_header, target_header, split=split)
            require(len(reference_header.samples) == expected_reference_samples,
                    "reference sample count drifted")
            require(len(target_header.samples) == expected_target_samples,
                    "target sample count drifted")
            require(set(reference_header.samples).isdisjoint(target_header.samples),
                    "reference and target sample axes overlap")

            with reference_output.open("x", encoding="utf-8", newline="") as reference_out, \
                    target_output.open("x", encoding="utf-8", newline="") as target_out:
                reference_out.writelines(reference_header.lines)
                target_out.writelines(target_header.lines)
                reference_rows = iter_vcf_rows(
                    reference_handle, reference_header, label="reference",
                    expected_chromosome=expected_chromosome,
                )
                target_rows = iter_vcf_rows(
                    target_handle, target_header, label="target",
                    expected_chromosome=expected_chromosome,
                )
                for record_number, pair in enumerate(
                    itertools.zip_longest(reference_rows, target_rows), 1
                ):
                    reference_row, target_row = pair
                    require(reference_row is not None and target_row is not None,
                            "reference/target VCF record counts differ")
                    require(reference_row.raw_axis == target_row.raw_axis,
                            f"reference/target raw CHROM/POS/REF/ALT axes differ at record {record_number}")
                    require(reference_row.key == target_row.key,
                            f"reference/target canonical axes differ at record {record_number}")
                    key = reference_row.key
                    if source_contig_label is None:
                        source_contig_label = reference_row.raw_axis[0]
                    require(reference_row.raw_axis[0] == source_contig_label,
                            "source VCF uses more than one raw chromosome label")
                    full_axis.append(key)
                    if key in selected_set:
                        removed_axis.append(key)
                        continue
                    fminus_axis.append(key)
                    reference_out.write(reference_row.line)
                    target_out.write(target_row.line)
                    reference_fminus_rows.append(reference_row.line)
                    target_fminus_rows.append(target_row.line)
                reference_out.flush()
                target_out.flush()
                os.fsync(reference_out.fileno())
                os.fsync(target_out.fileno())

        require(len(full_axis) == expected_full_count, "full FLARE locus count drifted")
        require(source_contig_label is not None, "full FLARE locus axis is empty")
        require(set(removed_axis) == selected_set,
                "one or more selected rare variants are absent from the full FLARE axis")
        require(len(removed_axis) == expected_selected_count,
                "the number of removed S660 variants differs")
        require(tuple(removed_axis) == selected_axis,
                "selected rare axis order differs from the full FLARE axis")
        raw_partition = validate_exact_locus_partition(full_axis, fminus_axis, selected_axis)
        partition = {
            "full_loci": raw_partition["counts"]["F_full"],
            "f_minus_s660_loci": raw_partition["counts"]["F_minus_selected"],
            "selected_s660_loci": raw_partition["counts"]["selected"],
            "overlap_loci": raw_partition["counts"]["overlap"],
        }
        require(partition["overlap_loci"] == 0, "F-minus-S660 and S660 axes overlap")
        require(partition["f_minus_s660_loci"] == expected_full_count - expected_selected_count,
                "F-minus-S660 locus count differs")

        reference_record_sha = _hash_rows(reference_fminus_rows)
        target_record_sha = _hash_rows(target_fminus_rows)
        reference_audit = audit_output_vcf(
            reference_output,
            label="filtered reference",
            expected_chromosome=expected_chromosome,
            expected_header=reference_header,
            expected_axis=fminus_axis,
            expected_record_sha256=reference_record_sha,
        )
        target_audit = audit_output_vcf(
            target_output,
            label="filtered target",
            expected_chromosome=expected_chromosome,
            expected_header=target_header,
            expected_axis=fminus_axis,
            expected_record_sha256=target_record_sha,
        )
        os.chmod(reference_output, 0o400)
        os.chmod(target_output, 0o400)

        receipt: dict[str, object] = {
            "schema_version": "m38_f_minus_s660_filter_receipt_v1",
            "stage": "M38_EXACT_F_MINUS_S660_BASELINE_FILTER",
            "status": "PASS_F_FULL_EQUALS_F_MINUS_S660_DISJOINT_UNION_S660",
            "split": split,
            "roles": roles,
            "scope": {
                "chromosome": expected_chromosome,
                "inputs_read": ["M34_REFERENCE_VCF", "M34_TARGET_VCF", "M34_SELECTED_LOCI"],
                "local_ancestry_labels_read": False,
                "model_predictions_read": False,
                "model_scores_read": False,
                "model_executed": False,
            },
            "identity": {
                "variant_key": ["CHROM", "POS", "REF", "ALT"],
                "source_contig_label": source_contig_label,
                "locus_id_used": False,
                "position_only_matching_used": False,
            },
            "inputs": {
                name: {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "expected_sha256": expected_hashes[name],
                    "observed_sha256": observed_hashes[name],
                }
                for name, path in inputs.items()
            },
            "counts": {
                "full_loci": len(full_axis),
                "selected_rare_loci": len(selected_axis),
                "removed_from_reference": len(removed_axis),
                "removed_from_target": len(removed_axis),
                "f_minus_s660_loci": len(fminus_axis),
                "reference_samples": len(reference_header.samples),
                "target_samples": len(target_header.samples),
            },
            "axis_sha256": {
                "full_CHROM_POS_REF_ALT": axis_sha256(full_axis, b"M38_FULL_AXIS_V1"),
                "selected_rare_CHROM_POS_REF_ALT": axis_sha256(selected_axis, b"M38_RARE_AXIS_V1"),
                "f_minus_s660_CHROM_POS_REF_ALT": axis_sha256(
                    fminus_axis, b"M38_F_MINUS_S660_AXIS_V1"
                ),
                "removed_in_full_order_CHROM_POS_REF_ALT": axis_sha256(
                    removed_axis, b"M38_RARE_AXIS_V1"
                ),
                "reference_samples": sample_axis_sha256(
                    reference_header.samples, b"M38_SAMPLE_AXIS_V1"
                ),
                "target_samples": sample_axis_sha256(
                    target_header.samples, b"M38_SAMPLE_AXIS_V1"
                ),
            },
            "partition": {
                **partition,
                "F_full_equals_disjoint_union_F_minus_S660_and_S660": True,
                "reference_target_full_axes_identical_and_ordered": True,
                "reference_target_F_minus_S660_axes_identical_and_ordered": True,
                "S660_absent_from_F_minus_S660": True,
            },
            "downstream_constraints": {
                "identical_scoring_grid_required": True,
                "raw_F_full_and_F_minus_S660_marker_grids_differ": True,
                "F_full_marker_count": len(full_axis),
                "F_minus_S660_marker_count": len(fminus_axis),
                "direct_metric_comparison_before_grid_alignment_forbidden": True,
            },
            "preservation": {
                "reference_header_exact": True,
                "target_header_exact": True,
                "reference_retained_rows_exact": True,
                "target_retained_rows_exact": True,
                "reference_record_payload_sha256": reference_record_sha,
                "target_record_payload_sha256": target_record_sha,
            },
            "outputs": {
                "reference_vcf": {"name": reference_output.name, **reference_audit},
                "target_vcf": {"name": target_output.name, **target_audit},
            },
        }
        receipt_path = stage / OUTPUT_NAMES["receipt"]
        with receipt_path.open("x", encoding="utf-8", newline="") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(receipt_path, 0o400)
        os.replace(stage, outdir)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("FIT", "VALID"), required=True)
    parser.add_argument("--reference-vcf", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--selected-loci", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--selected-sha256", required=True)
    parser.add_argument("--chromosome", default="22")
    parser.add_argument("--expected-full-loci", type=int, required=True)
    parser.add_argument("--expected-selected-loci", type=int, required=True)
    parser.add_argument("--expected-reference-samples", type=int, required=True)
    parser.add_argument("--expected-target-samples", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    receipt = build_f_minus_s660(
        split=args.split,
        reference_vcf=args.reference_vcf,
        target_vcf=args.target_vcf,
        selected_loci=args.selected_loci,
        expected_reference_sha256=args.reference_sha256,
        expected_target_sha256=args.target_sha256,
        expected_selected_sha256=args.selected_sha256,
        expected_chromosome=args.chromosome,
        expected_full_count=args.expected_full_loci,
        expected_selected_count=args.expected_selected_loci,
        expected_reference_samples=args.expected_reference_samples,
        expected_target_samples=args.expected_target_samples,
        outdir=args.outdir,
    )
    print(json.dumps({"split": receipt["split"], "status": receipt["status"],
                      "counts": receipt["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
