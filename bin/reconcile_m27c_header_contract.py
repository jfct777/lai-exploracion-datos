#!/usr/bin/env python3
"""Reconcile a corrected header audit with an immutable M27C full run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-summary", required=True, type=Path)
    parser.add_argument("--full-input-contract", required=True, type=Path)
    parser.add_argument("--header-contract", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(
    full_summary: dict[str, object],
    full_input: dict[str, object],
    header: dict[str, object],
) -> dict[str, object]:
    full_manifest_sha = str(full_input["gcs_input_manifest"]["sha256"])
    if str(header["input_manifest_sha256"]) != full_manifest_sha:
        raise ValueError("Header audit and full run use different input manifests")
    if int(header["expected_samples"]) != int(full_input["n_gvcf"]):
        raise ValueError("Header audit and full run use different sample counts")
    if not bool(header["header_contract_pass"]):
        raise ValueError("Corrected header audit did not pass")
    if any(full_summary["gates"].get(gate) != "PASS" for gate in ("C1", "C2", "C3")):
        raise ValueError("Only C0 can be reconciled by this program")

    gates = dict(full_summary["gates"])
    gates["C0"] = "PASS"
    robustness = str(full_summary["robustness_classification"])
    decision = (
        "REVIEW_THRESHOLD_SENSITIVITY"
        if robustness == "PASS_THRESHOLD_SENSITIVE"
        else "READY_FOR_RARE_DONOR_AUDIT_ONLY"
    )
    return {
        "stage": "M27C_HEADER_CONTRACT_RECONCILIATION",
        "decision": decision,
        "gates": gates,
        "primary_candidate_panel_ready_fraction": full_summary[
            "primary_candidate_panel_ready_fraction"
        ],
        "minimum_marker_fraction": full_summary["minimum_marker_fraction"],
        "robustness_classification": robustness,
        "input_manifest_sha256": full_manifest_sha,
        "c0_recomputed_from_headers_only": True,
        "c1_c2_c3_reused_without_recomputation": True,
        "final_donor_panel_certified": False,
        "pcrelate_executed": False,
        "gnomix_executed": False,
        "simulation_performed": False,
        "training_performed": False,
        "test_opened": False,
        "sample_ids_emitted": False,
    }


def main() -> int:
    args = parse_args()
    full_summary = json.loads(args.full_summary.read_text(encoding="utf-8"))
    full_input = json.loads(args.full_input_contract.read_text(encoding="utf-8"))
    header = json.loads(args.header_contract.read_text(encoding="utf-8"))
    result = reconcile(full_summary, full_input, header)
    result["git_commit"] = args.git_commit
    result["evidence_sha256"] = {
        "full_summary": sha256(args.full_summary),
        "full_input_contract": sha256(args.full_input_contract),
        "header_contract": sha256(args.header_contract),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
