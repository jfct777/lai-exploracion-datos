#!/usr/bin/env python3
"""Materialize phased REF/TARGET VCFs for one M33 DEVELOPMENT root.

This compatibility adapter reuses the audited M28C projector.  It receives
generative mosaic events only to copy donor alleles into TARGET haplotypes; the
merged local-ancestry truth is not accepted as an input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from materialize_m28c_b0_inputs import materialize, sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(args: argparse.Namespace) -> dict:
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    require(args.root_seed in pre4["root_registry"]["DEVELOPMENT"],
            "root is not registered for DEVELOPMENT")
    require(not args.outdir.exists(), f"refusing to overwrite {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=False)
    data_dir = args.outdir / "data"
    contract = {
        "stage": "M28C_B0_INPUT_PREFLIGHT",
        "scope": "technical_smoke_only_no_LAI_no_effect_estimation",
        "root_seed": args.root_seed,
        "seed_role": "M33_DEVELOPMENT_prospective",
        "chromosome": "22",
        "ancestries": ["AFR", "EUR", "ASIA"],
        "expected": {
            "b0_markers": 79791,
            "reference_haplotypes_per_ancestry": 60,
            "reference_pseudodiploids_per_ancestry": 30,
            "target_haplotypes": 60,
            "target_diploids": 30,
        },
        "inputs": {
            "tree_sequence_sha256": sha256(args.tree_sequence),
            "pool_manifest_sha256": sha256(args.pool_manifest),
            "mosaic_events_sha256": sha256(args.mosaic_events),
            "b0_markers_sha256": sha256(args.flare_grid),
        },
        "coordinate_contract": {
            "tree_sequence_start_bp": 15287922,
            "tree_sequence_end_bp_exclusive": 50791378,
            "hg38_chr22_length_bp": 50818468,
            "vcf_contig": "22",
            "build": "hg38",
            "ref_allele": "A",
            "alt_allele": "C",
            "note": "Binary simulation states are encoded deterministically as A/C.",
        },
        "decision": {
            "pass": "GO_M33_TRUTH_BLIND_FLARE",
            "fail": "STOP_M33_FLARE_INPUTS",
        },
    }
    contract_path = args.outdir / "m33_flare_input_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = materialize(SimpleNamespace(
        tree_sequence=args.tree_sequence,
        pool_manifest=args.pool_manifest,
        mosaic_events=args.mosaic_events,
        b0_markers=args.flare_grid,
        preregistration=contract_path,
        outdir=data_dir,
    ))
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M33_DEVELOPMENT_FLARE_INPUTS",
        "status": "PASS_TRUTH_UNMOUNTED_PHASED_INPUTS",
        "root_seed": args.root_seed,
        "pre4_sha256": sha256(args.pre4),
        "contract_sha256": sha256(contract_path),
        "input_sha256": contract["inputs"],
        "output_sha256": {
            name: sha256(data_dir / name)
            for name in (
                "m28c_b0_reference.vcf.gz",
                "m28c_b0_target.vcf.gz",
                "m28c_b0_reference.sample_map.tsv",
                "m28c_b0_reference_pairs.private.tsv",
            )
        },
        "projection_gates": result["gates"],
        "truth_argument_available": False,
        "truth_accessed": False,
    }
    (args.outdir / "m33_flare_inputs.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--tree-sequence", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--mosaic-events", type=Path, required=True)
    parser.add_argument("--flare-grid", type=Path, required=True)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "root_seed": result["root_seed"]}, sort_keys=True))
