#!/usr/bin/env python3
"""Summarize the frozen M33 R0 DEVELOPMENT screen without retuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


NAME = re.compile(
    r"^R(?P<rotation>\d+)\.(?P<family>.+)\.r(?P<radius>[0-9.]+)\.qq01\."
    r"(?P<arm>RD|RE)\.metrics\.json$"
)
EPSILON = 1e-12
ANCESTRIES = ("AFR", "EUR", "ASIA")
EXPECTED_PAIRS = {
    (family, radius)
    for family in ("local_linear", "small_residual_cnn_1d")
    for radius in (0.05, 0.1, 0.2, 0.5)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metrics(paths: list[Path]) -> dict[tuple[str, float, str], tuple[Path, dict]]:
    result: dict[tuple[str, float, str], tuple[Path, dict]] = {}
    for directory in paths:
        for path in sorted(directory.glob("*.metrics.json")):
            match = NAME.match(path.name)
            if match is None:
                continue
            key = (match.group("family"), float(match.group("radius")), match.group("arm"))
            if key in result:
                raise ValueError(f"duplicate candidate {key}: {path} and {result[key][0]}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("truth_opened_only_by_scorer") is not True:
                raise ValueError(f"truth barrier receipt missing in {path}")
            result[key] = (path, payload)
    return result


def finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite metric: {label}")
    return value


def summarize(args: argparse.Namespace) -> dict:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidates = load_metrics(args.metrics_dir)
    pairs = {(family, radius) for family, radius, _arm in candidates}
    observed_keys = set(candidates)
    expected_keys = {
        (family, radius, arm)
        for family, radius in EXPECTED_PAIRS
        for arm in ("RD", "RE")
    }
    if pairs != EXPECTED_PAIRS or observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"candidate grid differs from the frozen screen; missing={missing}, "
            f"unexpected={unexpected}"
        )

    f0_f1 = finite(baseline["boundary"]["0.2"]["f1"], "F0 F1")
    f0_false = finite(
        baseline["boundary"]["0.2"]["false_transitions_per_cM"], "F0 false transitions"
    )
    f0_mae = finite(baseline["macro_ancestry_dose_MAE"], "F0 macro MAE")
    rows = []
    for family, radius in sorted(pairs):
        rd_path, rd = candidates[(family, radius, "RD")]
        re_path, re_payload = candidates[(family, radius, "RE")]
        rd_f1 = finite(rd["boundary"]["0.2"]["f1"], f"{family} {radius} RD F1")
        re_f1 = finite(re_payload["boundary"]["0.2"]["f1"], f"{family} {radius} RE F1")
        rd_false = finite(rd["boundary"]["0.2"]["false_transitions_per_cM"], "RD false")
        re_false = finite(re_payload["boundary"]["0.2"]["false_transitions_per_cM"], "RE false")
        rd_mae = finite(rd["macro_ancestry_dose_MAE"], "RD macro MAE")
        re_mae = finite(re_payload["macro_ancestry_dose_MAE"], "RE macro MAE")
        ancestry_guardrails = {}
        for ancestry in ANCESTRIES:
            re_value = finite(re_payload["per_ancestry_truth_present_MAE"][ancestry], f"RE {ancestry}")
            ancestry_guardrails[ancestry] = {
                "RE": re_value,
                "RD": finite(rd["per_ancestry_truth_present_MAE"][ancestry], f"RD {ancestry}"),
                "F0": finite(baseline["per_ancestry_truth_present_MAE"][ancestry], f"F0 {ancestry}"),
            }
        checks = {
            "RE_F1_gt_RD": re_f1 - rd_f1 > EPSILON,
            "RE_F1_gt_F0": re_f1 - f0_f1 > EPSILON,
            "RE_false_le_RD": re_false - rd_false <= EPSILON,
            "RE_false_le_F0": re_false - f0_false <= EPSILON,
            "RE_macro_MAE_le_RD": re_mae - rd_mae <= EPSILON,
            "RE_macro_MAE_le_F0": re_mae - f0_mae <= EPSILON,
        }
        for ancestry, values in ancestry_guardrails.items():
            checks[f"RE_{ancestry}_MAE_le_RD"] = values["RE"] - values["RD"] <= EPSILON
            checks[f"RE_{ancestry}_MAE_le_F0"] = values["RE"] - values["F0"] <= EPSILON
        rows.append({
            "family": family,
            "radius_cM": radius,
            "F0_F1_0.2cM": f0_f1,
            "RD_F1_0.2cM": rd_f1,
            "RE_F1_0.2cM": re_f1,
            "RE_minus_RD_F1": re_f1 - rd_f1,
            "RE_minus_F0_F1": re_f1 - f0_f1,
            "F0_false_transitions_per_cM": f0_false,
            "RD_false_transitions_per_cM": rd_false,
            "RE_false_transitions_per_cM": re_false,
            "F0_macro_MAE": f0_mae,
            "RD_macro_MAE": rd_mae,
            "RE_macro_MAE": re_mae,
            "ancestry_truth_present_MAE": ancestry_guardrails,
            "checks": checks,
            "passes_screen_promotion": all(checks.values()),
            "input_sha256": {
                "RD_metrics": sha256_file(rd_path),
                "RE_metrics": sha256_file(re_path),
            },
        })

    promoted = [row for row in rows if row["passes_screen_promotion"]]
    family_status = {}
    for family in sorted({row["family"] for row in rows}):
        family_status[family] = (
            "CONTINUE" if any(row["passes_screen_promotion"] for row in rows if row["family"] == family)
            else "STOP_NO_RADIUS_PASSES"
        )
    output = {
        "schema_version": "1.0.0",
        "stage": "M33_DEVELOPMENT_SCREEN_SUMMARY",
        "status": "PASS_CANDIDATE_FOR_FULL_DEVELOPMENT" if promoted else "STOP_SCREEN_NO_CANDIDATE",
        "scope": "R0_seed1103_200_updates_DEVELOPMENT_only_no_EVAL_claim",
        "baseline_sha256": sha256_file(args.baseline),
        "candidate_count": len(rows),
        "promoted_candidate_count": len(promoted),
        "family_status": family_status,
        "candidates": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    header = [
        "family", "radius_cM", "F0_F1_0.2cM", "RD_F1_0.2cM", "RE_F1_0.2cM",
        "RE_minus_RD_F1", "RE_minus_F0_F1", "RD_false_transitions_per_cM",
        "RE_false_transitions_per_cM", "RD_macro_MAE", "RE_macro_MAE",
        "passes_screen_promotion",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row[name]) for name in header))
    args.output_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = summarize(parse_args())
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "promoted_candidate_count": result["promoted_candidate_count"],
    }, sort_keys=True))
