#!/usr/bin/env python3
"""Select a root-specific common-marker FLARE grid for M33 DEVELOPMENT.

The selector reads only the simulated source tree, role manifest, genetic map
and the authenticated baseline geometry.  TARGET mosaics, truth and rare-target
states are deliberately absent from the command-line interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tskit

from m28_simulation_preflight import read_genetic_map, sha256
from m28b_joint_capacity_audit import read_baseline_template
from m28b_marker_capacity_audit import inventory_markers, load_allowed_pools, write_marker_manifest
from m28b_optimal_matching_audit import grouped, hamilton_quotas, stable_key


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(args: argparse.Namespace) -> dict:
    m28 = json.loads(args.m28_contract.read_text(encoding="utf-8"))
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    matching = json.loads(args.matching_contract.read_text(encoding="utf-8"))
    require(args.root_seed in pre4["root_registry"]["DEVELOPMENT"],
            "root is not registered for DEVELOPMENT")
    require(args.root_seed not in pre4["root_registry"]["EVAL_reserved_not_generated"],
            "EVAL root must remain sealed")
    expected_template = matching["shared_inputs"]["baseline_template"]
    require(sha256(args.baseline_template) == expected_template["sha256"],
            "baseline-template hash drift")
    require(sha256(args.genetic_map) == m28["region"]["map_sha256"], "genetic-map hash drift")
    require(matching["development"]["b0_target_count"] == 79791, "FLARE grid size drift")

    genetic_map = read_genetic_map(args.genetic_map, m28)
    baseline = read_baseline_template(args.baseline_template, genetic_map, expected_template)
    require(sum(row.ref_polymorphic for row in baseline) == 79791,
            "baseline informative-marker count drift")
    tree = tskit.load(str(args.tree_sequence))
    pools = load_allowed_pools(
        args.pool_manifest,
        m28["source_populations"]["labels"],
        tree_sequence=tree,
        require_individual_schema=True,
    )
    markers, inventory = inventory_markers(tree, pools, genetic_map, m28)
    first_bp = int(expected_template["first_position"])
    last_bp = int(expected_template["last_position"])
    interval = [marker for marker in markers if first_bp <= marker.bp <= last_bp]
    ref_haplotypes = int(inventory["ref_total_haplotypes"])
    common = [
        marker for marker in interval
        if marker.maf >= 0.01 and 0 < marker.ref_minor_total < ref_haplotypes
    ]
    width = float(matching["development"]["bin_width_cm"])
    origin_cm = baseline[0].cm
    bins = grouped(common, origin_cm, width)
    quotas = hamilton_quotas({key: len(values) for key, values in bins.items()}, 79791)
    salt = matching["development"]["fixed_hash_salt"]
    selected = [
        marker
        for key in sorted(bins)
        for marker in sorted(bins[key], key=lambda value: stable_key(value, salt, "B0"))[:quotas[key]]
    ]
    selected.sort(key=lambda marker: marker.bp)
    require(len(selected) == 79791, "could not allocate the complete FLARE grid")
    require(len({marker.site_id for marker in selected}) == 79791, "duplicate site ID")
    require(len({marker.bp for marker in selected}) == 79791, "duplicate genomic position")
    args.outdir.mkdir(parents=True, exist_ok=False)
    marker_path = args.outdir / "m33_flare_grid.tsv.gz"
    write_marker_manifest(marker_path, "F0", selected, include_carrier_individuals=True)
    report = {
        "schema_version": "1.0.0",
        "stage": "M33_DEVELOPMENT_FLARE_GRID",
        "status": "PASS_TRUTH_FREE_GRID_SELECTION",
        "root_seed": args.root_seed,
        "input_sha256": {
            "tree_sequence": sha256(args.tree_sequence),
            "pool_manifest": sha256(args.pool_manifest),
            "genetic_map": sha256(args.genetic_map),
            "baseline_template": sha256(args.baseline_template),
            "m28_contract": sha256(args.m28_contract),
            "pre4": sha256(args.pre4),
            "matching_contract": sha256(args.matching_contract),
        },
        "counts": {
            "all_tree_markers": len(markers),
            "interval_markers": len(interval),
            "common_ref_polymorphic_candidates": len(common),
            "selected_flare_markers": len(selected),
        },
        "selection": {
            "bin_width_cM": width,
            "bin_origin_cM": origin_cm,
            "allocation": "Hamilton quotas proportional to root-specific common supply",
            "within_bin_order": "fixed SHA-256 ordering",
        },
        "forbidden_inputs_absent": ["TARGET", "mosaic_events", "truth", "LAI_predictions"],
        "output_sha256": {"m33_flare_grid.tsv.gz": sha256(marker_path)},
    }
    (args.outdir / "m33_flare_grid.receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--tree-sequence", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--baseline-template", type=Path, required=True)
    parser.add_argument("--m28-contract", type=Path, required=True)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--matching-contract", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "root_seed": report["root_seed"]}, sort_keys=True))
