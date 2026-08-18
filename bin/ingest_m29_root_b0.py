#!/usr/bin/env python3
"""Compatibility adapter around the audited Gnomix ingest gate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_m28c_gnomix_ingest import audit, sha256  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    contract = json.loads(args.production_contract.read_text(encoding="utf-8"))
    if contract.get("stage") != "M29_ROOT_B0_PRODUCTION" or contract.get("status") != "PRE_FROZEN_BEFORE_B0_SELECTION":
        raise ValueError("M29 B0 production contract is not frozen")
    contract_hash = sha256(args.production_contract)
    if args.root_seed not in {int(row["root_seed"]) for row in contract["roots"]}:
        raise ValueError("root seed is not preregistered")
    materialization = json.loads(args.materialization_report.read_text(encoding="utf-8"))
    if materialization.get("m29_stage") != "M29_ROOT_B0_MATERIALIZATION" or int(materialization.get("root_seed", -1)) != args.root_seed:
        raise ValueError("materialization report belongs to another stage or root")
    if materialization.get("m29_production_contract_sha256") != contract_hash:
        raise ValueError("materialization cites another production contract")
    legacy = {
        "stage": "M28C_B0_GNOMIX_INGEST_AUDIT",
        "scope": "technical_ingest_only_no_training_no_truth_no_effect_estimation",
        "root_seed": args.root_seed,
        "seed_role": "M29_DEV_root_not_independent_validation",
        "expected": {"markers": 79791, "reference_samples": 90, "target_samples": 30, "ploidy": 2, "chromosome": "22", "ref": "A", "alt": "C"},
        "software": {"gnomix_commit": contract["software"]["gnomix_commit"]},
        "decision": {"pass": "GO_M29_ROOT_B0_READY_FOR_TRAINING", "fail": "STOP_M29_ROOT_B0_INGEST"},
    }
    with tempfile.TemporaryDirectory(prefix="m29-ingest-contract-") as temporary:
        adapter = Path(temporary) / "legacy_ingest_contract.json"
        adapter.write_text(json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
        report = audit(SimpleNamespace(reference_vcf=args.reference_vcf, target_vcf=args.target_vcf, materialization_report=args.materialization_report, preregistration=adapter, gnomix_root=args.gnomix_root, root_seed=args.root_seed, outdir=args.outdir))
    report.update({"m29_stage": "M29_ROOT_B0_GNOMIX_INGEST", "m29_production_contract_sha256": contract_hash, "dev_root_only": True})
    (args.outdir / "m28c_b0_gnomix_ingest.public.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--reference-vcf", required=True, type=Path)
    parser.add_argument("--target-vcf", required=True, type=Path)
    parser.add_argument("--materialization-report", required=True, type=Path)
    parser.add_argument("--production-contract", required=True, type=Path)
    parser.add_argument("--gnomix-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"decision": result["decision"], "root_seed": result["root_seed"]}, sort_keys=True))
