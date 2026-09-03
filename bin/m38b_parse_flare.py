#!/usr/bin/env python3
"""Parse the M38B F-minus-S660 FLARE probabilities without opening truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import m33_safe_bridge_core as core
import m34_generate_mosaics as mosaics
import m34_parse_flare_truth as m34


F0_MEMBERS = m34.F0_MEMBERS
RUN_STAGE = "M38B_F_MINUS_S660_FLARE"
RUN_STATUS = "PASS_TRUTH_BLIND_FLARE_F_MINUS_S660_FIT"


class M38BParseError(ValueError):
    """Raised when FLARE output or its marker axis differs from M38B."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BParseError(message)


def load_run_receipt(
    path: Path,
    flare_anc: Path,
    expected_samples: int = 96,
    expected_markers: int = 42326,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    require(
        receipt.get("stage") == RUN_STAGE
        and receipt.get("status") == RUN_STATUS
        and receipt.get("scope", {}).get("target_partition") == "FIT"
        and receipt.get("truth_accessed") is False
        and receipt.get("scoring_performed") is False,
        "M38B FLARE receipt differs or is not truth-blind FIT",
    )
    require(
        receipt.get("shape", {}).get("marker_count") == expected_markers
        and receipt.get("shape", {}).get("target_sample_count") == expected_samples,
        "M38B FLARE receipt shape differs",
    )
    ancestry_audit = receipt.get("ancestry_vcf_audit")
    require(
        isinstance(ancestry_audit, dict)
        and ancestry_audit.get("sha256") == m34.sha256_file(flare_anc),
        "M38B FLARE ancestry VCF hash differs from its receipt",
    )
    return receipt


def parse_f0(
    *,
    flare_anc: Path,
    flare_receipt: Path,
    genetic_map: Path,
    genetic_map_sha256: str,
    outdir: Path,
    expected_samples: int = 96,
    expected_markers: int = 42326,
) -> dict[str, Any]:
    require(not outdir.exists() or (outdir.is_dir() and not any(outdir.iterdir())),
            "output directory must be absent or empty")
    require(m34.sha256_file(genetic_map) == genetic_map_sha256,
            "genetic-map SHA-256 differs")
    load_run_receipt(flare_receipt, flare_anc, expected_samples, expected_markers)
    ancestry_order = m34.parse_ancestry_order("AFR,EUR,NAM")
    id_map = m34.parse_flare_id_map("0=AFR,1=EUR,2=NAM")
    axis = m34.scan_flare(flare_anc, id_map)
    require(len(axis.samples) == expected_samples, "M38B FIT sample count differs")
    require(len(axis.loci) == expected_markers, "M38B F-minus-S660 marker count differs")
    genetic = mosaics.read_genetic_map(genetic_map, "22")
    require(
        axis.loci[0][0] >= genetic.start_bp and axis.loci[-1][0] <= genetic.end_bp,
        "genetic map does not cover the complete M38B marker axis",
    )
    probabilities, probability_audit = m34.parse_flare_probabilities(
        flare_anc, axis, ancestry_order, id_map
    )
    positions = np.asarray([row[0] for row in axis.loci], dtype="<i8")
    marker_cm = np.asarray(
        [genetic.bp_to_cm(int(position)) for position in positions], dtype="<f8"
    )
    require(
        np.all(np.isfinite(marker_cm)) and np.all(marker_cm[:-1] <= marker_cm[1:]),
        "interpolated marker cM axis is invalid",
    )
    arrays = {
        "sample_key_sha256": np.asarray(
            [core.sample_key(sample) for sample in axis.samples], dtype="|S64"
        ),
        "marker_chrom": np.full(len(axis.loci), 22, dtype="|u1"),
        "marker_pos": positions,
        "marker_ref": np.asarray([row[1].encode("ascii") for row in axis.loci], dtype="|S1"),
        "marker_alt": np.asarray([row[2].encode("ascii") for row in axis.loci], dtype="|S1"),
        "F0": probabilities,
    }
    require(
        set(arrays) == F0_MEMBERS
        and arrays["F0"].shape == (expected_samples, 2, expected_markers, 3),
        "M38B F-minus-S660 F0 schema differs",
    )
    marker_arrays = {"marker_cM": marker_cm}
    outdir.mkdir(parents=True, exist_ok=True)
    f0_path = outdir / "m38b_f_minus_s660_f0.npz"
    marker_path = outdir / "m38b_f_minus_s660_marker_cM.npz"
    m34.write_deterministic_npz(f0_path, arrays)
    m34.write_deterministic_npz(marker_path, marker_arrays)
    m34.reopen_npz(f0_path, arrays)
    m34.reopen_npz(marker_path, marker_arrays)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "M38B_PARSE_F_MINUS_S660_FLARE_F0",
        "decision": "PASS_M38B_F_MINUS_S660_F0_TRUTH_BLIND",
        "scope": {
            "chromosome": "22",
            "mosaic_root": "R0",
            "target_partition": "FIT",
            "valid_opened": False,
            "test_opened": False,
        },
        "ancestry_order": list(ancestry_order),
        "flare_id_map": id_map,
        "sample_count": expected_samples,
        "marker_count": expected_markers,
        "haplotype_count": 2,
        **probability_audit,
        "truth_argument_available": False,
        "truth_opened": False,
        "contains_truth": False,
        "contains_raw_sample_ids": False,
        "inputs": {
            "flare_anc_sha256": m34.sha256_file(flare_anc),
            "flare_receipt_sha256": m34.sha256_file(flare_receipt),
            "genetic_map_sha256": m34.sha256_file(genetic_map),
        },
        "outputs": {
            f0_path.name: m34.output_descriptor(f0_path),
            marker_path.name: m34.output_descriptor(marker_path),
        },
    }
    receipt["semantic_sha256"] = m34.canonical_json_sha256(receipt)
    core.write_exclusive_json(outdir / "m38b_f_minus_s660_f0.receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flare-anc", type=Path, required=True)
    parser.add_argument("--flare-receipt", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--genetic-map-sha256", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=96)
    parser.add_argument("--expected-markers", type=int, default=42326)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = parse_f0(
        flare_anc=args.flare_anc,
        flare_receipt=args.flare_receipt,
        genetic_map=args.genetic_map,
        genetic_map_sha256=args.genetic_map_sha256,
        outdir=args.outdir,
        expected_samples=args.expected_samples,
        expected_markers=args.expected_markers,
    )
    print(json.dumps({"decision": receipt["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
