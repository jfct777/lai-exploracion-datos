#!/usr/bin/env python3
"""Write root-specific B0 markers without evaluating BR/BS geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from m28b_marker_capacity_audit import sha256, write_marker_manifest  # noqa: E402
from m28b_optimal_matching_audit import prepare_markers  # noqa: E402


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("stage") != "M29_ROOT_B0_PRODUCTION" or contract.get("status") != "PRE_FROZEN_BEFORE_B0_SELECTION":
        raise ValueError("M29 B0 production contract is not frozen")
    return contract


def run(args: argparse.Namespace) -> dict:
    contract = load_contract(args.production_contract)
    roots = {int(row["root_seed"]): row for row in contract["roots"]}
    if args.root_seed not in roots:
        raise ValueError("root seed is not preregistered")
    root = roots[args.root_seed]
    if sha256(args.m28b_contract) != contract["shared_inputs"]["m28b_v5_contract_sha256"]:
        raise ValueError("M28B-v5 contract hash mismatch")
    m28b = json.loads(args.m28b_contract.read_text(encoding="utf-8"))
    mode = root["m28b_mode"]
    prepared = prepare_markers(args, m28b, mode)
    prefix = mode
    root_observed = {
        "tree": prepared["hashes"]["tree_sequence"],
        "pools": prepared["hashes"]["pool_manifest"],
        "preflight_report": prepared["hashes"][f"{prefix}_preflight_report"],
        "preflight_manifest": prepared["hashes"][f"{prefix}_preflight_manifest"],
    }
    if any(root["sha256"][key] != value for key, value in root_observed.items()):
        raise ValueError("production contract and authenticated M28B root inputs disagree")
    shared_observed = {
        "m28_contract_sha256": prepared["hashes"]["m28_preregistration"],
        "reproducibility_receipt_sha256": prepared["hashes"]["m28_v2_reproducibility"],
        "genetic_map_sha256": prepared["hashes"]["genetic_map"],
    }
    if any(contract["shared_inputs"][key] != value for key, value in shared_observed.items()):
        raise ValueError("production contract and authenticated shared inputs disagree")
    if contract["shared_inputs"]["baseline_template"]["sha256"] != prepared["hashes"]["baseline_template"]:
        raise ValueError("production contract and authenticated baseline disagree")
    b0 = prepared["b0"]
    if len(b0) != contract["expected"]["b0_markers"]:
        raise ValueError("root-specific B0 cardinality mismatch")
    args.outdir.mkdir(parents=True, exist_ok=False)
    marker_path = args.outdir / "m29_b0_markers.tsv.gz"
    write_marker_manifest(marker_path, "B0", b0, include_carrier_individuals=True)
    report = {
        "stage": "M29_ROOT_B0_SELECTION",
        "scope": "root_specific_B0_only_no_BR_BS_no_TARGET_no_truth_no_training",
        "root_seed": args.root_seed,
        "m28b_mode": mode,
        "production_contract_sha256": sha256(args.production_contract),
        "m28b_contract_sha256": sha256(args.m28b_contract),
        "input_sha256": prepared["hashes"],
        "output_sha256": {marker_path.name: sha256(marker_path)},
        "counts": {"B0": len(b0), "common_universe": len(prepared["common"])},
        "selection_rule": "M28B-v5 prepare_markers: 0.05 cM bins, Hamilton allocation, fixed SHA-256 ordering",
        "BR_BS_geometry_evaluated": False,
        "target_or_mosaic_read": False,
        "truth_read": False,
        "model_training_performed": False,
        "decision": "GO_ROOT_B0_MATERIALIZATION",
    }
    (args.outdir / "m29_b0_selection.public.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--tree-sequence", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--preflight-manifest", required=True, type=Path)
    parser.add_argument("--preflight-reproducibility", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--baseline-template", required=True, type=Path)
    parser.add_argument("--m28-preregistration", required=True, type=Path)
    parser.add_argument("--m28b-contract", dest="preregistration", required=True, type=Path)
    parser.add_argument("--production-contract", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.m28b_contract = args.preregistration
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"decision": result["decision"], "root_seed": result["root_seed"]}, sort_keys=True))
