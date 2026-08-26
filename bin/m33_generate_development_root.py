#!/usr/bin/env python3
"""Generate one prospective M33 DEVELOPMENT root with the frozen M28-v2 engine.

This adapter changes only the root registry.  Every biological and simulation
parameter is inherited from the authenticated M28-v2 contract and checked
against the M33 PRE-4 preregistration before generation.  EVAL roots and the
previously consumed technical roots are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from m28_simulation_preflight import (
    allocate_pools,
    audit_pool_disjunction,
    audit_rare_exposure,
    derive_seeds,
    draw_mosaics,
    merge_truth,
    read_genetic_map,
    sha256,
    simulate_sources,
    source_diploid_counts,
    validate_segment_cover,
    write_pool_manifest,
    write_segments,
)


STAGE = "M33_DEVELOPMENT_ROOT_GENERATION"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_contracts(m28_path: Path, pre4_path: Path, root_seed: int) -> tuple[dict, dict]:
    m28 = json.loads(m28_path.read_text(encoding="utf-8"))
    pre4 = json.loads(pre4_path.read_text(encoding="utf-8"))
    require(m28.get("stage") == "M28_LAI_SIMULATION_PREFLIGHT", "wrong M28 contract")
    require(m28.get("version") == 2, "M33 requires the individual-safe M28-v2 engine")
    require(pre4.get("schema_version") == "2.0.0", "unsupported PRE-4 contract")

    registry = pre4["root_registry"]
    development = tuple(int(value) for value in registry["DEVELOPMENT"])
    consumed = set(map(int, registry["consumed_technical_only"]))
    eval_roots = set(map(int, registry["EVAL_reserved_not_generated"]))
    require(root_seed in development, "root is not a frozen DEVELOPMENT root")
    require(root_seed not in consumed, "consumed technical root is forbidden")
    require(root_seed not in eval_roots, "EVAL root must remain ungenerated")

    generator = pre4["simulation_contract"]
    require(generator["generator"] == "M28_v2_individual_safe", "simulation engine drift")
    require(generator["demographic_model"] == "stdpopsim_HomSap_AmericanAdmixture_4B18",
            "demographic model drift")
    require(generator["ancestries"] == ["AFR", "EUR", "ASIA"], "ancestry order drift")
    require(generator["ASIA_is_not_NAM"] is True, "ASIA scope drift")
    require(registry["target_diploid_people_per_root"] == m28["pools"]["target_diploids"],
            "TARGET count drift")
    require(m28["pools"]["frequency_diploids"]["total"] == 300, "FREQ count drift")
    require(3 * m28["pools"]["lai_reference_diploids_per_ancestry"] == 90,
            "REF count drift")
    require(3 * m28["pools"]["mosaic_donor_haplotypes_per_ancestry"] // 2 == 768,
            "DONOR count drift")
    require(m28["pools"]["allocation_unit"] == "diploid_individual", "individual disjunction drift")
    require(m28["rare_definition"]["minimum_mac"] == 2, "minimum MAC drift")
    require(m28["rare_definition"]["maximum_maf_exclusive"] == 0.01, "MAF drift")
    return m28, pre4


def run(args: argparse.Namespace) -> dict[str, Any]:
    m28, pre4 = load_contracts(args.m28_contract, args.pre4, args.root_seed)
    require(not args.outdir.exists(), f"refusing to overwrite {args.outdir}")
    genetic_map = read_genetic_map(args.genetic_map, m28)
    seeds = derive_seeds(args.root_seed)
    args.outdir.mkdir(parents=True, exist_ok=False)

    ts = simulate_sources(genetic_map, m28, seeds)
    pools = allocate_pools(ts, m28, seeds["pool"])
    disjunction = audit_pool_disjunction(ts, pools)
    require(disjunction["cross_role_individuals"] == 0, "source individual crosses roles")
    mosaics = draw_mosaics(genetic_map, pools["DONOR"], m28, seeds["mosaic"])
    truth = [merge_truth(segments) for segments in mosaics]
    for segments, merged in zip(mosaics, truth):
        validate_segment_cover(segments, genetic_map.length_bp)
        validate_segment_cover(merged, genetic_map.length_bp)

    tree_path = args.outdir / "m28_sources.trees"
    pool_path = args.outdir / "m28_pools.private.tsv"
    mosaic_path = args.outdir / "m28_mosaic_events.private.tsv.gz"
    truth_path = args.outdir / "m28_lai_truth.private.tsv.gz"
    ts.dump(tree_path)
    write_pool_manifest(pool_path, pools, ts)
    write_segments(mosaic_path, mosaics, genetic_map)
    write_segments(truth_path, truth, genetic_map)
    exposure = audit_rare_exposure(ts, pools, mosaics, genetic_map, m28, args.outdir)

    output_names = [
        "m28_sources.trees",
        "m28_pools.private.tsv",
        "m28_mosaic_events.private.tsv.gz",
        "m28_lai_truth.private.tsv.gz",
        "m28_rare_catalog.tsv.gz",
        "m28_rare_haplotypes.tsv.gz",
    ]
    report = {
        "schema_version": "1.0.0",
        "stage": STAGE,
        "status": "PASS_GENERATION_TRUTH_PRIVATE_NO_MODELING",
        "root_seed": args.root_seed,
        "root_role": "DEVELOPMENT",
        "m28_contract_sha256": sha256(args.m28_contract),
        "pre4_sha256": sha256(args.pre4),
        "genetic_map_sha256": sha256(args.genetic_map),
        "derived_seeds": seeds,
        "source_diploid_counts": source_diploid_counts(m28),
        "pool_disjunction": disjunction,
        "sequence": {
            "length": int(ts.sequence_length),
            "trees": int(ts.num_trees),
            "sites": int(ts.num_sites),
            "mutations": int(ts.num_mutations),
        },
        "exposure": exposure,
        "scientific_semantic_sha256": canonical_sha256({
            "root_seed": args.root_seed,
            "derived_seeds": seeds,
            "sequence_length": int(ts.sequence_length),
            "trees": int(ts.num_trees),
            "sites": int(ts.num_sites),
            "mutations": int(ts.num_mutations),
            "disjunction": disjunction,
            "exposure": exposure,
        }),
        "output_sha256": {name: sha256(args.outdir / name) for name in output_names},
        "truth_policy": {
            "generated": True,
            "private": True,
            "forbidden_to_materializer_and_trainer": True,
            "allowed_only_to_final_score_process": True,
        },
        "claims_excluded": pre4["claims_excluded"],
    }
    report_path = args.outdir / "m33_generation.receipt.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m28-contract", required=True, type=Path)
    parser.add_argument("--pre4", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "root_seed": result["root_seed"]}, sort_keys=True))
