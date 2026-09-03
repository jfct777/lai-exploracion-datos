#!/usr/bin/env python3
"""Project M34 full FLARE and FIT truth onto the exact M38B marker grid.

The comparison grid is F-minus-S660.  Matching is exclusively by the ordered
``CHROM/POS/REF/ALT`` key; position-only projection is forbidden.  Truth is
opened only in this downstream alignment stage, never during FLARE execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import m33_safe_bridge_core as core
import m34_parse_flare_truth as m34
from _experiment_invariants import validate_exact_locus_partition
from m38_build_f_minus_s660 import load_selected_axis


F0_MEMBERS = m34.F0_MEMBERS
TRUTH_MEMBERS = m34.TRUTH_MEMBERS
HEX_DIGITS = frozenset("0123456789abcdef")
VariantKey = tuple[str, int, str, str]


class M38BBaselineAlignmentError(ValueError):
    """Raised when full, F-minus-S660, selected, or truth axes drift."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BBaselineAlignmentError(message)


def validate_sha256(value: str, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS),
        f"{label} SHA-256 is malformed",
    )
    return value


def sha256_file(path: Path) -> str:
    # Accept both copied and symlink-staged Nextflow inputs; every load-bearing
    # source is authenticated by SHA-256 before its contents are consumed.
    require(path.is_file(), f"invalid regular input: {path}")
    return m34.sha256_file(path)


def verify_hash(path: Path, expected: str, label: str) -> str:
    wanted = validate_sha256(expected, label)
    observed = sha256_file(path)
    require(observed == wanted, f"SHA-256 mismatch for {label}")
    return observed


def decode_allele(value: object, label: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        text = bytes(value).decode("ascii")
    else:
        text = str(value)
    require(text in {"A", "C", "G", "T"}, f"{label} is not an uppercase SNV allele")
    return text


def f0_axis(arrays: Mapping[str, np.ndarray], label: str) -> tuple[VariantKey, ...]:
    marker_count = len(arrays["marker_pos"])
    require(
        all(
            arrays[name].shape == (marker_count,)
            for name in ("marker_chrom", "marker_pos", "marker_ref", "marker_alt")
        ),
        f"{label} marker axes differ",
    )
    keys: list[VariantKey] = []
    for index in range(marker_count):
        chrom = str(int(arrays["marker_chrom"][index]))
        position = int(arrays["marker_pos"][index])
        ref = decode_allele(arrays["marker_ref"][index], f"{label} REF")
        alt = decode_allele(arrays["marker_alt"][index], f"{label} ALT")
        require(chrom == "22" and position > 0 and ref != alt,
                f"{label} marker {index} differs")
        keys.append((chrom, position, ref, alt))
    require(len(keys) == len(set(keys)), f"{label} marker axis is duplicated")
    require(
        all(left[1] < right[1] for left, right in zip(keys, keys[1:])),
        f"{label} marker axis is not strictly ordered",
    )
    return tuple(keys)


def load_f0(path: Path, label: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == F0_MEMBERS, f"{label} F0 members differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    sample_count = len(arrays["sample_key_sha256"])
    marker_count = len(arrays["marker_pos"])
    require(
        arrays["sample_key_sha256"].dtype == np.dtype("|S64")
        and arrays["F0"].dtype == np.dtype("<f4")
        and arrays["F0"].shape == (sample_count, 2, marker_count, 3)
        and len(set(arrays["sample_key_sha256"].tolist())) == sample_count,
        f"{label} F0 dimensions or dtypes differ",
    )
    for start in range(0, marker_count, 4096):
        block = arrays["F0"][:, :, start:start + 4096, :]
        require(
            np.all(np.isfinite(block))
            and np.all(block >= 0)
            and np.allclose(block.sum(axis=3), 1.0, rtol=0, atol=5e-6),
            f"{label} F0 is outside the probability simplex",
        )
    f0_axis(arrays, label)
    return arrays


def load_marker_cm(path: Path, expected_markers: int, label: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"marker_cM"}, f"{label} marker-cM members differ")
        values = np.ascontiguousarray(archive["marker_cM"])
    require(
        values.dtype == np.dtype("<f8")
        and values.shape == (expected_markers,)
        and np.all(np.isfinite(values))
        and np.all(values[:-1] <= values[1:]),
        f"{label} marker-cM axis differs",
    )
    return values


def load_truth(path: Path, expected_samples: int, expected_markers: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == TRUTH_MEMBERS, "full FIT truth members differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    require(
        arrays["sample_key_sha256"].dtype == np.dtype("|S64")
        and arrays["marker_pos"].dtype == np.dtype("<i8")
        and arrays["labels"].dtype == np.dtype("|i1")
        and arrays["sample_key_sha256"].shape == (expected_samples,)
        and arrays["marker_pos"].shape == (expected_markers,)
        and arrays["labels"].shape == (expected_samples, 2, expected_markers)
        and np.all((arrays["labels"] >= 0) & (arrays["labels"] < 3)),
        "full FIT truth dimensions, dtypes, or ancestry labels differ",
    )
    return arrays


def sample_axis_sha256(values: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256(b"M38B_SAMPLE_KEY_AXIS_V1\0")
    for value in values.tolist():
        digest.update(bytes(value) + b"\n")
    return digest.hexdigest()


def validate_parse_receipt(
    receipt_path: Path,
    fminus_f0: Path,
    fminus_marker_cm: Path,
    expected_samples: int,
    expected_markers: int,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == "M38B_PARSE_F_MINUS_S660_FLARE_F0"
        and receipt.get("decision") == "PASS_M38B_F_MINUS_S660_F0_TRUTH_BLIND"
        and receipt.get("sample_count") == expected_samples
        and receipt.get("marker_count") == expected_markers
        and receipt.get("truth_opened") is False,
        "M38B parsed F-minus-S660 receipt differs",
    )
    outputs = receipt.get("outputs", {})
    require(
        outputs.get(fminus_f0.name, {}).get("sha256") == sha256_file(fminus_f0)
        and outputs.get(fminus_marker_cm.name, {}).get("sha256")
        == sha256_file(fminus_marker_cm),
        "M38B parsed artifact hashes differ from their receipt",
    )
    return receipt


def align_baselines(
    *,
    experiment: Path,
    full_f0: Path,
    full_marker_cm: Path,
    full_truth: Path,
    selected_loci: Path,
    fminus_f0: Path,
    fminus_marker_cm: Path,
    fminus_receipt: Path,
    expected_full_f0_sha256: str,
    expected_full_marker_cm_sha256: str,
    expected_full_truth_sha256: str,
    expected_selected_loci_sha256: str,
    outdir: Path,
    expected_samples: int = 96,
    expected_full_markers: int = 42986,
    expected_selected_markers: int = 660,
) -> dict[str, Any]:
    expected_minus = expected_full_markers - expected_selected_markers
    require(expected_minus > 0, "expected F-minus-S660 count is invalid")
    require(not outdir.exists() or (outdir.is_dir() and not any(outdir.iterdir())),
            "output directory must be absent or empty")
    contract = json.loads(experiment.read_text(encoding="utf-8"))
    require(
        contract.get("experiment_id") == "M38B_S660_INCREMENTAL_LAI_CHR22_R0_FIT"
        and contract.get("claim_scope", {}).get("target_partition") == "FIT_ONLY"
        and contract.get("claim_scope", {}).get("valid_opened") is False
        and contract.get("claim_scope", {}).get("test_opened") is False,
        "M38B experiment is not FIT-only with VALID/TEST closed",
    )
    source_hashes = {
        "full_f0": verify_hash(full_f0, expected_full_f0_sha256, "full_f0"),
        "full_marker_cm": verify_hash(
            full_marker_cm, expected_full_marker_cm_sha256, "full_marker_cm"
        ),
        "full_truth": verify_hash(full_truth, expected_full_truth_sha256, "full_truth"),
        "selected_loci": verify_hash(
            selected_loci, expected_selected_loci_sha256, "selected_loci"
        ),
    }
    preregistered = contract.get("source_artifacts", {})
    require(
        preregistered.get("f_full_npz_sha256") == source_hashes["full_f0"]
        and preregistered.get("f_full_marker_cm_sha256") == source_hashes["full_marker_cm"]
        and preregistered.get("fit_truth_sha256") == source_hashes["full_truth"]
        and preregistered.get("s660_selected_sha256") == source_hashes["selected_loci"],
        "canonical source hashes differ from the M38B preregistration",
    )
    validate_parse_receipt(
        fminus_receipt,
        fminus_f0,
        fminus_marker_cm,
        expected_samples,
        expected_minus,
    )

    full = load_f0(full_f0, "F_full")
    minus = load_f0(fminus_f0, "F_minus_S660")
    require(full["F0"].shape == (expected_samples, 2, expected_full_markers, 3),
            "F_full shape differs")
    require(minus["F0"].shape == (expected_samples, 2, expected_minus, 3),
            "F_minus_S660 shape differs")
    require(np.array_equal(full["sample_key_sha256"], minus["sample_key_sha256"]),
            "F_full and F_minus_S660 sample axes differ")
    full_axis = f0_axis(full, "F_full")
    minus_axis = f0_axis(minus, "F_minus_S660")
    try:
        selected_axis = load_selected_axis(
            selected_loci,
            expected_chromosome="22",
            expected_count=expected_selected_markers,
        )
        partition = validate_exact_locus_partition(full_axis, minus_axis, selected_axis)
    except ValueError as exc:
        raise M38BBaselineAlignmentError(str(exc)) from exc
    require(
        partition["counts"]
        == {
            "F_full": expected_full_markers,
            "F_minus_selected": expected_minus,
            "selected": expected_selected_markers,
            "overlap": 0,
        },
        "exact locus partition counts differ",
    )
    index_by_key = {key: index for index, key in enumerate(full_axis)}
    projection = np.asarray([index_by_key[key] for key in minus_axis], dtype=np.int64)
    require(np.all(projection[:-1] < projection[1:]),
            "F-minus-S660 projection does not preserve F_full order")

    full_cm = load_marker_cm(full_marker_cm, expected_full_markers, "F_full")
    minus_cm = load_marker_cm(fminus_marker_cm, expected_minus, "F_minus_S660")
    require(np.array_equal(full_cm[projection], minus_cm),
            "F_full and F-minus-S660 genetic-coordinate axes differ after projection")
    truth = load_truth(full_truth, expected_samples, expected_full_markers)
    require(np.array_equal(truth["sample_key_sha256"], full["sample_key_sha256"]),
            "truth and F_full sample axes differ")
    require(np.array_equal(truth["marker_pos"], full["marker_pos"]),
            "truth and F_full marker-position axes differ")

    full_projected = {
        "sample_key_sha256": np.ascontiguousarray(full["sample_key_sha256"]),
        "marker_chrom": np.ascontiguousarray(full["marker_chrom"][projection]),
        "marker_pos": np.ascontiguousarray(full["marker_pos"][projection]),
        "marker_ref": np.ascontiguousarray(full["marker_ref"][projection]),
        "marker_alt": np.ascontiguousarray(full["marker_alt"][projection]),
        "F0": np.ascontiguousarray(full["F0"][:, :, projection, :]),
    }
    truth_projected = {
        "sample_key_sha256": np.ascontiguousarray(truth["sample_key_sha256"]),
        "marker_pos": np.ascontiguousarray(truth["marker_pos"][projection]),
        "labels": np.ascontiguousarray(truth["labels"][:, :, projection]),
    }
    common_cm = {"marker_cM": np.ascontiguousarray(minus_cm)}
    require(
        all(
            np.array_equal(full_projected[name], minus[name])
            for name in ("sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref", "marker_alt")
        ),
        "projected F_full and F-minus-S660 axes are not byte-identical",
    )
    require(np.array_equal(truth_projected["marker_pos"], minus["marker_pos"]),
            "projected truth and F-minus-S660 marker axes differ")

    outdir.mkdir(parents=True, exist_ok=True)
    full_output = outdir / "m38b_f_full_projected_to_f_minus_s660.npz"
    truth_output = outdir / "m38b_fit_truth_projected_to_f_minus_s660.npz"
    cm_output = outdir / "m38b_common_marker_cM.npz"
    m34.write_deterministic_npz(full_output, full_projected)
    m34.write_deterministic_npz(truth_output, truth_projected)
    m34.write_deterministic_npz(cm_output, common_cm)
    m34.reopen_npz(full_output, full_projected)
    m34.reopen_npz(truth_output, truth_projected)
    m34.reopen_npz(cm_output, common_cm)
    partition_axes = partition["axis_sha256"]
    common_axis_sha = partition_axes["F_minus_selected"]
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660",
        "decision": "PASS_EXACT_SHARED_F_MINUS_S660_SCORING_GRID",
        "scope": {
            "chromosome": "22",
            "mosaic_root": "R0",
            "target_partition": "FIT",
            "valid_opened": False,
            "test_opened": False,
            "claim_level": "exploratory",
        },
        "identity": {
            "variant_key": ["CHROM", "POS", "REF", "ALT"],
            "position_only_matching_used": False,
            "f_minus_is_common_only": False,
        },
        "counts": {
            "F_full": expected_full_markers,
            "F_minus_S660": expected_minus,
            "S660": expected_selected_markers,
            "partition_overlap": 0,
            "target_people": expected_samples,
            "target_haplotypes": 2 * expected_samples,
        },
        "partition": partition,
        "axis_sha256": {
            "hash_definition": "DNABR_LOCUS_AXIS_V1 canonical JSON",
            "F_full_CHROM_POS_REF_ALT": partition_axes["F_full"],
            "S660_CHROM_POS_REF_ALT": partition_axes["selected"],
            "common_F_minus_S660_CHROM_POS_REF_ALT": common_axis_sha,
            "F_full_projected_CHROM_POS_REF_ALT": common_axis_sha,
            "truth_projected_CHROM_POS_REF_ALT": common_axis_sha,
            "target_sample_keys": sample_axis_sha256(minus["sample_key_sha256"]),
        },
        "alignment": {
            "F_full_projected_equals_F_minus_S660_axis": True,
            "truth_projected_equals_F_minus_S660_axis": True,
            "truth_alleles_inherited_from_authenticated_F_full_axis": True,
            "marker_cM_projected_equals_F_minus_S660": True,
            "sample_axes_identical": True,
            "F_full_probability_values_reestimated": False,
            "truth_labels_reestimated": False,
        },
        "inputs": {
            **source_hashes,
            "fminus_f0": sha256_file(fminus_f0),
            "fminus_marker_cm": sha256_file(fminus_marker_cm),
            "fminus_receipt": sha256_file(fminus_receipt),
            "experiment": sha256_file(experiment),
        },
        "outputs": {
            full_output.name: m34.output_descriptor(full_output),
            truth_output.name: m34.output_descriptor(truth_output),
            cm_output.name: m34.output_descriptor(cm_output),
        },
    }
    receipt["semantic_sha256"] = m34.canonical_json_sha256(receipt)
    core.write_exclusive_json(outdir / "m38b_baseline_alignment.receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--full-f0", type=Path, required=True)
    parser.add_argument("--full-marker-cm", type=Path, required=True)
    parser.add_argument("--full-truth", type=Path, required=True)
    parser.add_argument("--selected-loci", type=Path, required=True)
    parser.add_argument("--fminus-f0", type=Path, required=True)
    parser.add_argument("--fminus-marker-cm", type=Path, required=True)
    parser.add_argument("--fminus-receipt", type=Path, required=True)
    parser.add_argument("--full-f0-sha256", required=True)
    parser.add_argument("--full-marker-cm-sha256", required=True)
    parser.add_argument("--full-truth-sha256", required=True)
    parser.add_argument("--selected-loci-sha256", required=True)
    parser.add_argument("--expected-samples", type=int, default=96)
    parser.add_argument("--expected-full-markers", type=int, default=42986)
    parser.add_argument("--expected-selected-markers", type=int, default=660)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = align_baselines(
        experiment=args.experiment,
        full_f0=args.full_f0,
        full_marker_cm=args.full_marker_cm,
        full_truth=args.full_truth,
        selected_loci=args.selected_loci,
        fminus_f0=args.fminus_f0,
        fminus_marker_cm=args.fminus_marker_cm,
        fminus_receipt=args.fminus_receipt,
        expected_full_f0_sha256=args.full_f0_sha256,
        expected_full_marker_cm_sha256=args.full_marker_cm_sha256,
        expected_full_truth_sha256=args.full_truth_sha256,
        expected_selected_loci_sha256=args.selected_loci_sha256,
        outdir=args.outdir,
        expected_samples=args.expected_samples,
        expected_full_markers=args.expected_full_markers,
        expected_selected_markers=args.expected_selected_markers,
    )
    print(json.dumps({"decision": receipt["decision"], "counts": receipt["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
