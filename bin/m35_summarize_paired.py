#!/usr/bin/env python3
"""Summarize a truth-opened paired FLARE2 minus FLARE 0.6 comparison."""

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
            "M35 requires canonical M34 truth-opened score metrics")
    return value


def summarize(flare060_path: Path, flare2_path: Path, direct_gate_path: Path) -> dict[str, Any]:
    flare060, flare2 = load_metric(flare060_path), load_metric(flare2_path)
    direct_gate = json.loads(direct_gate_path.read_text(encoding="utf-8"))
    require(direct_gate.get("status") == "PASS_NUMERICALLY_EQUIVALENT_TO_M34_CANONICAL_F0" and
            direct_gate.get("direct_metrics_sha256") == sha256_file(flare060_path),
            "paired delta is blocked until direct FLARE reproduces canonical M34 F0")
    require(flare060["input_sha256"]["truth"] == flare2["input_sha256"]["truth"],
            "paired arms were not scored against the same truth")
    require(flare060["ancestry_names"] == flare2["ancestry_names"],
            "paired ancestry axes differ")
    require(flare060["marker_count"] == flare2["marker_count"] and
            flare060["sample_count"] == flare2["sample_count"] and
            flare060["cm_span"] == flare2["cm_span"], "paired scoring geometry differs")
    ancestry = flare060["ancestry_names"]
    return {
        "schema_version": "1.0.0",
        "stage": "M35_FLARE2_MINUS_FLARE060_PAIRED_SUMMARY",
        "status": "PASS_PAIRED_POINT_ESTIMATE",
        "claim_level": "exploratory",
        "comparison": "FLARE2_MINUS_FLARE_0_6",
        "shared_truth_sha256": flare060["input_sha256"]["truth"],
        "paired_geometry": {key: flare060[key] for key in ("sample_count", "marker_count", "cm_span")},
        "delta_flare2_minus_flare060": {
            "macro_ancestry_dose_MAE": flare2["macro_ancestry_dose_MAE"] - flare060["macro_ancestry_dose_MAE"],
            "haplotype_Brier": flare2["haplotype_Brier"] - flare060["haplotype_Brier"],
            "NAM_truth_present_MAE": (
                None if flare060["NAM_truth_present_MAE"] is None else
                flare2["NAM_truth_present_MAE"] - flare060["NAM_truth_present_MAE"]
            ),
            "per_ancestry_MAE": {
                name: flare2["per_ancestry_MAE"][name] - flare060["per_ancestry_MAE"][name]
                for name in ancestry
            },
            "boundary": {
                tolerance: {
                    "f1": flare2["boundary"][tolerance]["f1"] - flare060["boundary"][tolerance]["f1"],
                    "false_transitions_per_cM": (
                        flare2["boundary"][tolerance]["false_transitions_per_cM"] -
                        flare060["boundary"][tolerance]["false_transitions_per_cM"]
                    ),
                }
                for tolerance in flare060["boundary"]
            },
        },
        "input_sha256": {"flare_0_6_metrics": sha256_file(flare060_path),
                         "flare2_metrics": sha256_file(flare2_path),
                         "direct_f0_gate": sha256_file(direct_gate_path)},
        "truth_opened_only_by_scorer": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flare-0-6-metrics", type=Path, required=True)
    parser.add_argument("--flare2-metrics", type=Path, required=True)
    parser.add_argument("--direct-f0-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite paired summary")
    result = summarize(args.flare_0_6_metrics, args.flare2_metrics, args.direct_f0_gate)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
