#!/usr/bin/env python3
"""Compatibility adapter around the audited M28C VCF materializer."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from materialize_m28c_b0_inputs import materialize, sha256  # noqa: E402


def run(args: argparse.Namespace) -> dict:
    contract = json.loads(args.production_contract.read_text(encoding="utf-8"))
    if contract.get("stage") != "M29_ROOT_B0_PRODUCTION" or contract.get("status") != "PRE_FROZEN_BEFORE_B0_SELECTION":
        raise ValueError("M29 B0 production contract is not frozen")
    contract_hash = sha256(args.production_contract)
    roots = {int(row["root_seed"]): row for row in contract["roots"]}
    root = roots.get(args.root_seed)
    if root is None:
        raise ValueError("root seed is not preregistered")
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    if selection.get("stage") != "M29_ROOT_B0_SELECTION" or selection.get("root_seed") != args.root_seed or selection.get("decision") != "GO_ROOT_B0_MATERIALIZATION":
        raise ValueError("invalid upstream B0 selection report")
    if selection.get("production_contract_sha256") != contract_hash:
        raise ValueError("B0 selection cites another production contract")
    if selection.get("output_sha256", {}).get(args.b0_markers.name) != sha256(args.b0_markers):
        raise ValueError("B0 marker hash differs from selection report")
    expected = root["sha256"]
    observed = {"tree": sha256(args.tree_sequence), "pools": sha256(args.pool_manifest), "mosaic_events": sha256(args.mosaic_events)}
    if any(observed[key] != expected[key] for key in observed):
        raise ValueError("root materialization input hash mismatch")
    legacy = {
        "stage": "M28C_B0_INPUT_PREFLIGHT",
        "scope": "technical_smoke_only_no_LAI_no_effect_estimation",
        "root_seed": args.root_seed,
        "seed_role": "M29_DEV_root_not_independent_validation",
        "chromosome": "22",
        "ancestries": ["AFR", "EUR", "ASIA"],
        "expected": {"b0_markers": 79791, "reference_haplotypes_per_ancestry": 60, "reference_pseudodiploids_per_ancestry": 30, "target_haplotypes": 60, "target_diploids": 30},
        "inputs": {"tree_sequence_sha256": observed["tree"], "pool_manifest_sha256": observed["pools"], "mosaic_events_sha256": observed["mosaic_events"], "b0_markers_sha256": sha256(args.b0_markers)},
        "coordinate_contract": {"tree_sequence_start_bp": 15287922, "tree_sequence_end_bp_exclusive": 50791378, "hg38_chr22_length_bp": 50818468, "vcf_contig": "22", "build": "hg38", "ref_allele": "A", "alt_allele": "C", "note": "A/C encode simulated binary states 0/1; they are not nucleotide identities."},
        "decision": {"pass": "GO_EXTERNAL_GNOMIX_INGEST_VALIDATION", "fail": "STOP_M29_ROOT_B0_MATERIALIZATION"},
    }
    with tempfile.TemporaryDirectory(prefix="m29-b0-contract-") as temporary:
        adapter = Path(temporary) / "legacy_materialization_contract.json"
        adapter.write_text(json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")
        report = materialize(SimpleNamespace(tree_sequence=args.tree_sequence, pool_manifest=args.pool_manifest, mosaic_events=args.mosaic_events, b0_markers=args.b0_markers, preregistration=adapter, outdir=args.outdir))
    report.update({"m29_stage": "M29_ROOT_B0_MATERIALIZATION", "m29_production_contract_sha256": contract_hash, "m29_selection_report_sha256": sha256(args.selection_report), "dev_root_only": True})
    (args.outdir / "m28c_b0_input_preflight.public.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--tree-sequence", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--mosaic-events", required=True, type=Path)
    parser.add_argument("--b0-markers", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--production-contract", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"decision": result["decision"], "root_seed": result["root_seed"]}, sort_keys=True))
