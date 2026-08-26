#!/usr/bin/env python3
"""Prepare the FREQ-only, minor-oriented rare channel for M33 DEVELOPMENT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m31_ordered_rare_preflight import (
    derive_freq_sites,
    known_answers,
    materialize_target,
    sha256_file,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(args: argparse.Namespace) -> dict:
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    require(args.root_seed in pre4["root_registry"]["DEVELOPMENT"],
            "root is not registered for DEVELOPMENT")
    require(not args.outdir.exists(), f"refusing to overwrite {args.outdir}")
    contract = {
        "chromosome_domain": {"chrom": "22", "start_bp": 15287922, "end_bp_exclusive": 50791378},
        "rare_universe": {
            "selector": "FREQ_only",
            "minimum_mac": 2,
            "maximum_maf_exclusive": 0.01,
            "minimum_carrier_individuals": 2,
            "prohibited_selectors": [
                "REF_LAI", "DONOR", "TARGET", "truth", "Gnomix_prediction", "FLARE_prediction"
            ],
        },
    }
    selected, freq_audit = derive_freq_sites(
        args.tree_sequence, args.pool_manifest, args.rare_catalog, contract
    )
    args.outdir.mkdir(parents=True, exist_ok=False)
    target_audit = materialize_target(
        args.rare_haplotypes, selected, args.root_seed, args.outdir
    )
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M33_DEVELOPMENT_RARE_CHANNEL",
        "status": "PASS_FREQ_ONLY_MINOR_ORIENTED_TWO_CARRIERS",
        "root_seed": args.root_seed,
        "pre4_sha256": sha256_file(args.pre4),
        "input_sha256": {
            "tree_sequence": sha256_file(args.tree_sequence),
            "pool_manifest": sha256_file(args.pool_manifest),
            "rare_catalog": sha256_file(args.rare_catalog),
            "rare_haplotypes": sha256_file(args.rare_haplotypes),
        },
        "known_answers": known_answers(),
        "freq_only_audit": freq_audit,
        "target_materialization_audit": target_audit,
        "output_sha256": {
            name: sha256_file(args.outdir / name)
            for name in (
                "m31_ordered_rare.sites.tsv.gz",
                "m31_ordered_rare.target.tsv.gz",
                "m31_ordered_rare.samples.tsv",
            )
        },
        "truth_argument_available": False,
        "truth_accessed": False,
    }
    (args.outdir / "m33_rare_channel.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--tree-sequence", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--rare-catalog", type=Path, required=True)
    parser.add_argument("--rare-haplotypes", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "root_seed": result["root_seed"]}, sort_keys=True))
