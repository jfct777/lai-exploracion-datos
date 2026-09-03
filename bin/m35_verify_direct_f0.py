#!/usr/bin/env python3
"""Require the direct M35 FLARE 0.6 metrics to reproduce frozen M34 F0 metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _same(left: Any, right: Any, path: str = "metric", absolute_tolerance: float = 0.0) -> None:
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), f"{path} keys differ")
        for key in left:
            _same(left[key], right[key], f"{path}.{key}", absolute_tolerance)
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), f"{path} length differs")
        for index, value in enumerate(left):
            _same(value, right[index], f"{path}[{index}]", absolute_tolerance)
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        require(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=absolute_tolerance),
                f"{path} differs from canonical F0")
    else:
        require(left == right, f"{path} differs from canonical F0")


def verify(direct_path: Path, canonical_path: Path) -> dict[str, Any]:
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    require(direct.get("status") == canonical.get("status") == "PASS_SCORED",
            "direct and canonical F0 metrics must be canonical scorer outputs")
    require(direct["input_sha256"]["truth"] == canonical["input_sha256"]["truth"],
            "direct FLARE and canonical F0 did not use the same M34 truth")
    for key in ("sample_count", "haplotype_count", "marker_count", "ancestry_names"):
        _same(direct[key], canonical[key], key)
    _same(direct["cm_span"], canonical["cm_span"], "cm_span", 1e-4)
    for key in ("macro_ancestry_dose_MAE", "per_ancestry_MAE", "NAM_truth_present_MAE",
                "haplotype_Brier"):
        _same(direct[key], canonical[key], key, 1e-6)
    require(set(direct["boundary"]) == set(canonical["boundary"]), "boundary tolerances differ")
    boundary_delta: dict[str, Any] = {}
    for tolerance in direct["boundary"]:
        left, right = direct["boundary"][tolerance], canonical["boundary"][tolerance]
        for key in ("matched", "predicted", "truth", "f1"):
            _same(left[key], right[key], f"boundary.{tolerance}.{key}")
        _same(left["false_transitions_per_cM"], right["false_transitions_per_cM"],
              f"boundary.{tolerance}.false_transitions_per_cM", 1e-6)
        boundary_delta[tolerance] = {
            key: float(left[key]) - float(right[key])
            for key in ("f1", "false_transitions_per_cM")
        }
    return {
        "schema_version": "1.0.0", "stage": "M35_DIRECT_FLARE060_CANONICAL_F0_GATE",
        "status": "PASS_NUMERICALLY_EQUIVALENT_TO_M34_CANONICAL_F0", "claim_level": "exploratory",
        "direct_metrics_sha256": sha256_file(direct_path),
        "canonical_f0_metrics_sha256": sha256_file(canonical_path),
        "shared_truth_sha256": direct["input_sha256"]["truth"],
        "prediction_hashes_equal": (
            direct["input_sha256"].get("prediction") == canonical["input_sha256"].get("prediction")
        ),
        "tolerances": {"cm_span_cM": 1e-4, "continuous_metrics": 1e-6},
        "observed_deltas": {
            "cm_span_cM": direct["cm_span"] - canonical["cm_span"],
            "macro_ancestry_dose_MAE": (
                direct["macro_ancestry_dose_MAE"] - canonical["macro_ancestry_dose_MAE"]
            ),
            "haplotype_Brier": direct["haplotype_Brier"] - canonical["haplotype_Brier"],
            "NAM_truth_present_MAE": (
                direct["NAM_truth_present_MAE"] - canonical["NAM_truth_present_MAE"]
            ),
            "boundary": boundary_delta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-metrics", type=Path, required=True)
    parser.add_argument("--canonical-f0-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite F0 verification")
    args.output.write_text(json.dumps(verify(args.direct_metrics, args.canonical_f0_metrics), indent=2,
                                      sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
