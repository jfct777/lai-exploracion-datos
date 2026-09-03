#!/usr/bin/env python3
"""Aggregate the complete blind M35C grid without favorable-seed selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("EXTERNAL_NAM", "NATWGS")


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
    require(value.get("experiment_id") == "M35C_NATWGS_SOURCE_SENSITIVITY_CHR22" and
            value.get("status") == "PREREGISTERED_EXPLORATORY_SOURCE_SCREEN",
            "M35C aggregate contract differs")
    require(value["cluster_screen"]["primary_gate"] ==
            "all_9_NATWGS_coarse_combinations_must_pass",
            "M35C aggregate gate differs")
    return value


def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    supports = sorted(row["NAM_support"] for row in values)
    margins = sorted(row["log_margin"] for row in values)
    return {
        "passed": sum(row["status"] == "PASS_M35C_CLUSTER_SEPARATION" for row in values),
        "total": len(values),
        "NAM_support_min": supports[0],
        "NAM_support_median": supports[len(supports) // 2],
        "NAM_support_max": supports[-1],
        "log_margin_min": margins[0],
        "log_margin_median": margins[len(margins) // 2],
        "log_margin_max": margins[-1],
    }


def aggregate(contract_path: Path, screen_dirs: list[Path], output: Path,
              go_token: Path) -> dict[str, Any]:
    require(not output.exists() and not go_token.exists(), "refusing to overwrite M35C aggregate")
    contract = load_contract(contract_path)
    selection_seeds = contract["reference_design"]["selection_seeds"]
    gmm_seeds = contract["cluster_screen"]["gmm_seeds"]
    expected = {
        (arm, selection, granularity, gmm)
        for arm in ARMS for selection in selection_seeds
        for granularity in ("coarse", "fine") for gmm in gmm_seeds
    }
    rows: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    preparation_by_seed: dict[int, str] = {}
    for directory in screen_dirs:
        evidence_path = directory / "m35c.cluster_evidence.json"
        receipt_path = directory / "m35c.screen_receipt.json"
        require(evidence_path.is_file() and receipt_path.is_file(),
                "M35C screen output is incomplete")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        key = (evidence["arm"], evidence["selection_seed"],
               evidence["granularity"], evidence["gmm_seed"])
        require(key in expected and key not in rows, "M35C screen grid is unexpected or duplicated")
        require(evidence.get("target_truth_opened") is False and
                receipt.get("truth_input_present") is False and
                receipt.get("final_inference_performed") is False,
                "M35C truth was opened before the aggregate gate")
        require(evidence["balanced_macro_counts"] == {"AFR": 23, "EUR": 23, "NAM": 23},
                "M35C screen is not 23/23/23")
        require(evidence["marker_axis_sha256"] == contract["scope"]["marker_axis_sha256"],
                "M35C screen marker axis differs")
        require(receipt["evidence_sha256"] == sha256_file(evidence_path),
                "M35C screen evidence hash differs from receipt")
        prior = preparation_by_seed.setdefault(evidence["selection_seed"],
                                               receipt["prepare_receipt_sha256"])
        require(prior == receipt["prepare_receipt_sha256"],
                "M35C source arms do not share one preparation receipt within seed")
        rows[key] = evidence
        evidence_hashes[":".join(map(str, key))] = sha256_file(evidence_path)
    require(set(rows) == expected, "M35C screen grid is incomplete")

    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        summaries[arm] = {}
        for granularity in ("coarse", "fine"):
            values = [rows[(arm, selection, granularity, gmm)]
                      for selection in selection_seeds for gmm in gmm_seeds]
            summaries[arm][granularity] = summarize(values)
    primary = summaries["NATWGS"]["coarse"]
    primary_pass = primary["passed"] == primary["total"] == 9
    status = ("PASS_M35C_NATWGS_PRIMARY_9_OF_9_GO_PREASSIGNED_POST_GATE" if primary_pass else
              "NO_GO_M35C_NATWGS_PRIMARY_NOT_9_OF_9_STOP_BEFORE_TRUTH")
    result = {
        "schema_version": "1.0.0",
        "stage": "M35C_SOURCE_CLUSTER_SCREEN_AGGREGATE",
        "status": status,
        "claim_level": "exploratory",
        "primary": {"arm": "NATWGS", "granularity": "coarse", **primary},
        "matched_comparator": {"arm": "EXTERNAL_NAM", "granularity": "coarse",
                               **summaries["EXTERNAL_NAM"]["coarse"]},
        "all_summaries": summaries,
        "primary_rule": "all_9_NATWGS_coarse_combinations_must_pass",
        "truth_opened": False,
        "post_hoc_seed_selection": False,
        "preassigned_post_gate_pair": contract["preassigned_post_gate_pair"],
        "historical_Brazilian_23_used": False,
        "screen_rows": [
            {
                "arm": key[0], "selection_seed": key[1], "granularity": key[2],
                "gmm_seed": key[3], "status": rows[key]["status"],
                "NAM_support": rows[key]["NAM_support"], "log_margin": rows[key]["log_margin"],
            }
            for key in sorted(rows)
        ],
        "input_sha256": {
            "contract": sha256_file(contract_path),
            "screen_evidence": evidence_hashes,
            "preparation_receipts_by_selection_seed": {
                str(seed): digest for seed, digest in sorted(preparation_by_seed.items())
            },
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if primary_pass:
        assigned = contract["preassigned_post_gate_pair"]
        go_token.write_text(json.dumps({
            "status": "GO_PREASSIGNED_POST_GATE_PAIR_ONLY",
            "gate_sha256": sha256_file(output),
            "arm": assigned["arm"],
            "selection_seed": assigned["selection_seed"],
            "gmm_seed": assigned["gmm_seed"],
            "granularity": assigned["granularity"],
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
