#!/usr/bin/env python3
"""Summarize the preassigned balanced FLARE2 minus direct-FLARE comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metric(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("status") == "PASS_SCORED" and
            value.get("truth_opened_only_by_scorer") is True,
            "M35B requires canonical truth-opened scorer output")
    return value


def summarize(direct_path: Path, flare2_path: Path, canonical_path: Path,
              inference_receipt_path: Path) -> dict[str, Any]:
    direct = load_metric(direct_path)
    flare2 = load_metric(flare2_path)
    canonical = load_metric(canonical_path)
    inference = json.loads(inference_receipt_path.read_text(encoding="utf-8"))
    require(inference.get("status") == "PASS_M35B_PAIRED_INFERENCE_READY_FOR_SEPARATE_SCORE" and
            inference.get("truth_input_present") is False,
            "M35B paired inference receipt differs")
    require(direct["input_sha256"]["truth"] == flare2["input_sha256"]["truth"] ==
            canonical["input_sha256"]["truth"],
            "M35B arms and contextual F0 used different truth")
    for key in ("sample_count", "haplotype_count", "marker_count", "ancestry_names", "cm_span"):
        require(direct[key] == flare2[key], f"M35B paired scoring {key} differs")
    require(direct["sample_count"] == 32 and direct["marker_count"] == 42986,
            "M35B paired score geometry differs from R0")
    require(direct["ancestry_names"] == ["AFR", "EUR", "NAM"],
            "M35B ancestry axis differs")

    ancestry = direct["ancestry_names"]
    tolerances = sorted(direct["boundary"], key=float)
    require(set(tolerances) == set(flare2["boundary"]), "M35B boundary tolerance axes differ")

    def delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, Any]:
        return {
            "macro_ancestry_dose_MAE": right["macro_ancestry_dose_MAE"] - left["macro_ancestry_dose_MAE"],
            "haplotype_Brier": right["haplotype_Brier"] - left["haplotype_Brier"],
            "NAM_truth_present_MAE": (
                None if left["NAM_truth_present_MAE"] is None else
                right["NAM_truth_present_MAE"] - left["NAM_truth_present_MAE"]
            ),
            "per_ancestry_MAE": {
                name: right["per_ancestry_MAE"][name] - left["per_ancestry_MAE"][name]
                for name in ancestry
            },
            "boundary": {
                tolerance: {
                    "f1": right["boundary"][tolerance]["f1"] - left["boundary"][tolerance]["f1"],
                    "false_transitions_per_cM": (
                        right["boundary"][tolerance]["false_transitions_per_cM"] -
                        left["boundary"][tolerance]["false_transitions_per_cM"]
                    ),
                    "matched": right["boundary"][tolerance]["matched"] - left["boundary"][tolerance]["matched"],
                    "predicted": right["boundary"][tolerance]["predicted"] - left["boundary"][tolerance]["predicted"],
                }
                for tolerance in tolerances
            },
        }

    result = {
        "schema_version": "1.0.0",
        "stage": "M35B_BALANCED_FLARE2_PAIRED_SCORE",
        "status": "PASS_M35B_EXPLORATORY_PAIRED_POINT_ESTIMATE",
        "claim_level": "exploratory_single_chromosome_single_target_root",
        "comparison": "FLARE2_BALANCED_MINUS_FLARE_0_6_BALANCED",
        "reference_sample_counts": {"AFR": 25, "EUR": 25, "NAM": 25},
        "preassigned_pair": {
            "selection_seed": inference["selection_seed"],
            "gmm_seed": inference["gmm_seed"],
            "granularity": inference["granularity"],
        },
        "shared_geometry": {key: direct[key] for key in
                            ("sample_count", "haplotype_count", "marker_count", "cm_span")},
        "metrics": {
            "FLARE_0_6_BALANCED": direct,
            "FLARE2_BALANCED": flare2,
            "M34_FLARE_0_6_FULL_REFERENCE_CONTEXT_ONLY": canonical,
        },
        "delta_FLARE2_balanced_minus_FLARE_0_6_balanced": delta(flare2, direct),
        "delta_FLARE_0_6_balanced_minus_M34_full_reference_context": delta(direct, canonical),
        "interpretation_guard": (
            "The paired estimand is FLARE2 versus direct FLARE on the same balanced subset. "
            "The M34 full-reference comparison is contextual and is not the paired algorithmic contrast."
        ),
        "input_sha256": {
            "direct_metrics": sha256_file(direct_path),
            "flare2_metrics": sha256_file(flare2_path),
            "canonical_full_reference_metrics": sha256_file(canonical_path),
            "inference_receipt": sha256_file(inference_receipt_path),
            "truth": direct["input_sha256"]["truth"],
        },
        "truth_opened_only_by_scorer": True,
        "post_hoc_seed_selection": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-metrics", type=Path, required=True)
    parser.add_argument("--flare2-metrics", type=Path, required=True)
    parser.add_argument("--canonical-f0-metrics", type=Path, required=True)
    parser.add_argument("--inference-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite M35B paired summary")
    args.output.write_text(json.dumps(summarize(
        args.direct_metrics, args.flare2_metrics, args.canonical_f0_metrics,
        args.inference_receipt,
    ), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
