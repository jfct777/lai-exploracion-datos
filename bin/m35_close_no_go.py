#!/usr/bin/env python3
"""Close an already-stopped M35 model-build directory with authenticated NO_GO evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import m35_flare2_paired as paired


def close(contract_path: Path, outdir: Path) -> dict:
    contract = paired.load_contract(contract_path)
    delta_path = outdir / "m35_paired.delta_manifest.json"
    resources_path = outdir / "m35_paired.resource_estimate.json"
    model_path = outdir / "m35.flare2.model.model"
    evidence_path = outdir / "m35.flare2.cluster_assignment.evidence.json"
    receipt_path = outdir / "m35_paired.receipt.json"
    paired.require(all(path.is_file() for path in (delta_path, resources_path, model_path)),
                   "M35 existing closure lacks model or authenticated preflight artifacts")
    paired.require(not evidence_path.exists() and not receipt_path.exists(),
                   "M35 existing closure refuses to overwrite evidence or receipt")
    delta = json.loads(delta_path.read_text(encoding="utf-8"))
    paired.require(delta.get("experiment_id") == contract["experiment_id"], "M35 closure contract identity differs")
    gate = contract["methods"]["flare2"]["cluster_assignment"]
    evidence = paired.cluster_assignment_evidence_from_model(
        model_path, contract["ancestry_order"], delta["shared_axes"]["panel_to_ancestry"],
        gate["min_probability"], gate["min_log_margin"],
    )
    paired.require(evidence["status"] == "NO_GO_TRUTH_BLIND_CLUSTER_ASSIGNMENT",
                   "M35 closure is only for a scientifically failed assignment gate")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "1.0.0", "experiment_id": contract["experiment_id"],
        "status": "NO_GO_CLUSTER_ASSIGNMENT_BEFORE_FINAL_FLARE2", "closure_mode": "POST_MODEL_GATE_AUDIT",
        "contract_sha256": paired.sha256_file(contract_path),
        "delta_manifest_sha256": paired.sha256_file(delta_path),
        "resource_estimate_sha256": paired.sha256_file(resources_path),
        "cluster_assignment_evidence_sha256": paired.sha256_file(evidence_path),
        "cluster_assignment_evidence": evidence, "final_flare2_launched": False,
        "scoring_performed": False, "label_input_present": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"status": close(args.contract, args.outdir)["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
