#!/usr/bin/env python3
"""Aggregate every preregistered M35B screen without selecting a favorable seed."""

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


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("experiment_id") == "M35B_FLARE2_BALANCED_SENSITIVITY_CHR22" and
            value.get("status") == "PREREGISTERED_EXPLORATORY_SCREEN",
            "M35B aggregate contract differs")
    require(value["cluster_screen"]["primary_gate"] ==
            "all_9_coarse_selection_by_gmm_combinations_must_pass",
            "M35B aggregate gate differs")
    return value


def aggregate(contract_path: Path, screen_dirs: list[Path], output: Path,
              go_token: Path) -> dict[str, Any]:
    require(not output.exists() and not go_token.exists(), "refusing to overwrite M35B aggregate")
    contract = load_contract(contract_path)
    selection_seeds = contract["reference_balance"]["selection_seeds"]
    gmm_seeds = contract["cluster_screen"]["gmm_seeds"]
    expected = {(selection, granularity, gmm)
                for selection in selection_seeds
                for granularity in ("coarse", "fine")
                for gmm in gmm_seeds}
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    for directory in screen_dirs:
        evidence_path = directory / "m35b.cluster_evidence.json"
        receipt_path = directory / "m35b.screen_receipt.json"
        require(evidence_path.is_file() and receipt_path.is_file(), "M35B screen output is incomplete")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        key = (evidence["selection_seed"], evidence["granularity"], evidence["gmm_seed"])
        require(key in expected and key not in rows, "M35B screen grid is unexpected or duplicated")
        require(evidence.get("target_truth_opened") is False and
                receipt.get("truth_input_present") is False and
                receipt.get("final_inference_performed") is False,
                "M35B truth was opened before the aggregate gate")
        require(evidence["balanced_macro_counts"] == {"AFR": 25, "EUR": 25, "NAM": 25},
                "M35B screen is not balanced")
        require(evidence["marker_axis_sha256"] == contract["scope"]["marker_axis_sha256"],
                "M35B screen marker axis differs")
        require(receipt["evidence_sha256"] == sha256_file(evidence_path),
                "M35B screen evidence hash differs from receipt")
        rows[key] = evidence
        evidence_hashes[f"{key[0]}:{key[1]}:{key[2]}"] = sha256_file(evidence_path)
    require(set(rows) == expected, "M35B screen grid is incomplete")

    coarse = [rows[(selection, "coarse", gmm)]
              for selection in selection_seeds for gmm in gmm_seeds]
    fine = [rows[(selection, "fine", gmm)]
            for selection in selection_seeds for gmm in gmm_seeds]
    coarse_passes = sum(row["status"] == "PASS_M35B_CLUSTER_SEPARATION" for row in coarse)
    fine_passes = sum(row["status"] == "PASS_M35B_CLUSTER_SEPARATION" for row in fine)
    primary_pass = coarse_passes == len(coarse)
    status = ("PASS_M35B_PRIMARY_9_OF_9_GO_PREASSIGNED_FINAL" if primary_pass else
              "NO_GO_M35B_PRIMARY_NOT_9_OF_9_STOP_BEFORE_TRUTH")

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        supports = [row["NAM_support"] for row in values]
        margins = [row["log_margin"] for row in values]
        return {
            "passed": sum(row["status"] == "PASS_M35B_CLUSTER_SEPARATION" for row in values),
            "total": len(values),
            "NAM_support_min": min(supports),
            "NAM_support_median": sorted(supports)[len(supports) // 2],
            "NAM_support_max": max(supports),
            "log_margin_min": min(margins),
            "log_margin_median": sorted(margins)[len(margins) // 2],
            "log_margin_max": max(margins),
        }

    result = {
        "schema_version": "1.0.0",
        "stage": "M35B_CLUSTER_SCREEN_AGGREGATE",
        "status": status,
        "claim_level": "exploratory",
        "primary": {"granularity": "coarse", **summarize(coarse)},
        "sensitivity": {"granularity": "fine", **summarize(fine)},
        "primary_rule": "all_9_coarse_combinations_must_pass",
        "truth_opened": False,
        "post_hoc_seed_selection": False,
        "preassigned_final_pair": contract["primary_final_pair"],
        "screen_rows": [
            {
                "selection_seed": key[0], "granularity": key[1], "gmm_seed": key[2],
                "status": rows[key]["status"], "NAM_support": rows[key]["NAM_support"],
                "log_margin": rows[key]["log_margin"],
            }
            for key in sorted(rows)
        ],
        "input_sha256": {
            "contract": sha256_file(contract_path),
            "screen_evidence": evidence_hashes,
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if primary_pass:
        go_token.write_text(json.dumps({
            "status": "GO_PREASSIGNED_FINAL_ONLY",
            "gate_sha256": sha256_file(output),
            "selection_seed": contract["primary_final_pair"]["selection_seed"],
            "gmm_seed": contract["primary_final_pair"]["gmm_seed"],
            "granularity": contract["primary_final_pair"]["granularity"],
        }, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--go-token", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.contract, args.screen_dir, args.output, args.go_token)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
