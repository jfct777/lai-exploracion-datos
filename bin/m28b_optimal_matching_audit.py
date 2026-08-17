#!/usr/bin/env python3
"""Develop and validate the final M28B-v4 geometric marker comparator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m28_simulation_preflight import load_contract as load_m28_contract, rare_under_contract, read_genetic_map  # noqa: E402
from m28b_generic_capacity_audit import (  # noqa: E402
    capacity_bin,
    grouped,
    hamilton_quotas,
    monte_carlo_rank,
    stable_key,
    write_null_table,
    write_pair_table,
)
from m28b_joint_capacity_audit import read_baseline_template  # noqa: E402
from m28b_marker_capacity_audit import (  # noqa: E402
    Marker,
    MarkerPair,
    bp_to_cm,
    inventory_markers,
    load_allowed_pools,
    load_json,
    pair_diagnostics,
    sha256,
    verify_hash,
    write_json,
    write_marker_manifest,
)


def optimal_subsequence_pairs(queries: list[Marker], candidates: list[Marker]) -> list[MarkerPair]:
    """Exact 1D minimum-L1 matching to an ordered unique candidate subsequence."""

    ordered_queries = sorted(queries, key=lambda marker: (marker.cm, marker.bp, marker.site_id))
    ordered_candidates = sorted(candidates, key=lambda marker: (marker.cm, marker.bp, marker.site_id))
    n_queries = len(ordered_queries)
    n_candidates = len(ordered_candidates)
    if n_candidates < n_queries:
        raise ValueError("Exact matching has fewer controls than queries")
    if not ordered_queries:
        return []

    infinity = float("inf")
    previous = [0.0] * (n_candidates + 1)
    take = [bytearray(n_candidates + 1) for _ in range(n_queries + 1)]
    for query_index in range(1, n_queries + 1):
        current = [infinity] * (n_candidates + 1)
        query = ordered_queries[query_index - 1]
        for candidate_index in range(1, n_candidates + 1):
            skip_cost = current[candidate_index - 1]
            match_cost = previous[candidate_index - 1] + abs(
                query.cm - ordered_candidates[candidate_index - 1].cm
            )
            # A tie keeps the earlier ordered candidate through skip_cost.
            if match_cost < skip_cost:
                current[candidate_index] = match_cost
                take[query_index][candidate_index] = 1
            else:
                current[candidate_index] = skip_cost
        previous = current

    pairs: list[MarkerPair] = []
    query_index = n_queries
    candidate_index = n_candidates
    while query_index:
        if candidate_index == 0:
            raise RuntimeError("Exact matching backtrack lost a required candidate")
        if take[query_index][candidate_index]:
            query = ordered_queries[query_index - 1]
            candidate = ordered_candidates[candidate_index - 1]
            pairs.append(MarkerPair(query.bp, query.cm, candidate))
            query_index -= 1
            candidate_index -= 1
        else:
            candidate_index -= 1
    pairs.reverse()
    return pairs


def allocate_exact_k(capacity: dict[int, int], target_k: int) -> dict[int, int]:
    positive = {key: value for key, value in capacity.items() if value > 0}
    if target_k > sum(positive.values()):
        raise ValueError("Frozen K exceeds validation capacity")
    return hamilton_quotas(positive, target_k)


def prepare_markers(args: argparse.Namespace, contract: dict, mode: str) -> dict:
    shared = contract["shared_inputs"]
    expected = contract["development_inputs"] if mode == "development" else contract["validation_inputs"]
    baseline_expected = shared["baseline_template"]
    hashes = {
        "tree_sequence": verify_hash(args.tree_sequence, expected["tree_sequence_sha256"], f"{mode} tree sequence"),
        "pool_manifest": verify_hash(args.pool_manifest, expected["pool_manifest_sha256"], f"{mode} pool manifest"),
        "genetic_map": verify_hash(args.genetic_map, shared["genetic_map_sha256"], "genetic map"),
        "m28_preregistration": verify_hash(args.m28_preregistration, shared["m28_preflight_contract_sha256"], "M28 preregistration"),
        "baseline_template": verify_hash(args.baseline_template, baseline_expected["sha256"], "baseline template"),
        "m28b_v4_preregistration": sha256(args.preregistration),
    }
    if mode == "validation":
        if args.preflight_manifest is None:
            raise ValueError("Validation requires --preflight-manifest")
        hashes["validation_preflight_manifest"] = verify_hash(
            args.preflight_manifest,
            expected["preflight_manifest_sha256"],
            "validation preflight manifest",
        )

    m28_contract = load_m28_contract(args.m28_preregistration)
    genetic_map = read_genetic_map(args.genetic_map, m28_contract)
    baseline = read_baseline_template(args.baseline_template, genetic_map, baseline_expected)
    baseline_poly = sum(row.ref_polymorphic for row in baseline)
    target_b0 = int(contract["development"]["b0_target_count"])
    if baseline_poly != target_b0:
        raise ValueError("Authenticated baseline informative count no longer matches B0")
    pools = load_allowed_pools(args.pool_manifest, m28_contract["source_populations"]["labels"])

    import tskit

    tree_sequence = tskit.load(str(args.tree_sequence))
    markers, inventory = inventory_markers(tree_sequence, pools, genetic_map, m28_contract)
    first_bp = int(baseline_expected["first_position"])
    last_bp = int(baseline_expected["last_position"])
    interval = [marker for marker in markers if first_bp <= marker.bp <= last_bp]
    ref_haplotypes = int(inventory["ref_total_haplotypes"])
    rare = [
        marker for marker in interval
        if rare_under_contract({"mac": marker.mac, "maf": marker.maf}, m28_contract)
        and marker.ref_minor_total >= 1
    ]
    common = [
        marker for marker in interval
        if marker.maf >= 0.01 and 0 < marker.ref_minor_total < ref_haplotypes
    ]
    width = float(contract["development"]["bin_width_cm"])
    origin_cm = bp_to_cm(genetic_map, first_bp)
    rare_bins = grouped(rare, origin_cm, width)
    common_bins = grouped(common, origin_cm, width)
    b0_quotas = hamilton_quotas(
        {key: len(values) for key, values in common_bins.items()}, target_b0
    )
    salt = contract["development"]["fixed_hash_salt"]
    b0_bins: dict[int, list[Marker]] = {}
    reserve_bins: dict[int, list[Marker]] = {}
    for key, values in common_bins.items():
        ordered = sorted(values, key=lambda marker: stable_key(marker, salt, "B0"))
        quota = b0_quotas.get(key, 0)
        b0_bins[key] = ordered[:quota]
        reserve_bins[key] = ordered[quota:]
    capacity = {
        key: min(
            len(rare_bins.get(key, [])),
            len(reserve_bins.get(key, [])),
            len(b0_bins.get(key, [])),
        )
        for key in sorted(set(rare_bins) | set(reserve_bins) | set(b0_bins))
    }
    return {
        "hashes": hashes,
        "inventory": inventory,
        "first_bp": first_bp,
        "last_bp": last_bp,
        "origin_cm": origin_cm,
        "width": width,
        "rare": rare,
        "common": common,
        "rare_bins": rare_bins,
        "b0_bins": b0_bins,
        "reserve_bins": reserve_bins,
        "capacity": capacity,
        "b0": [marker for key in sorted(b0_bins) for marker in b0_bins[key]],
    }


def evaluate_configuration(prepared: dict, k_by_bin: dict[int, int], contract: dict) -> dict:
    salt = contract["development"]["fixed_hash_salt"]
    null_replicates = int(contract["development"]["null_replicates"])
    selected_rare: list[Marker] = []
    selected_controls: list[Marker] = []
    observed_pairs: list[MarkerPair] = []
    per_bin: list[dict] = []

    for key in sorted(prepared["capacity"]):
        k_bin = int(k_by_bin.get(key, 0))
        rare_here = sorted(
            prepared["rare_bins"].get(key, []),
            key=lambda marker: stable_key(marker, salt, "BR"),
        )[:k_bin]
        reserve_here = prepared["reserve_bins"].get(key, [])
        pairs_here = optimal_subsequence_pairs(rare_here, reserve_here)
        selected_rare.extend(rare_here)
        selected_controls.extend(pair.control for pair in pairs_here)
        observed_pairs.extend(pairs_here)
        per_bin.append({
            "bin": key,
            "capacity": prepared["capacity"][key],
            "K": k_bin,
            "rare_eligible": len(prepared["rare_bins"].get(key, [])),
            "common_reserve": len(reserve_here),
            "B0": len(prepared["b0_bins"].get(key, [])),
        })

    observed = pair_diagnostics(observed_pairs)
    null_summaries: list[dict] = []
    for replicate in range(null_replicates):
        null_pairs: list[MarkerPair] = []
        for key in sorted(prepared["capacity"]):
            k_bin = int(k_by_bin.get(key, 0))
            if not k_bin:
                continue
            pseudo_queries = sorted(
                prepared["b0_bins"].get(key, []),
                key=lambda marker: stable_key(marker, salt, f"NULL_{replicate}"),
            )[:k_bin]
            null_pairs.extend(
                optimal_subsequence_pairs(pseudo_queries, prepared["reserve_bins"].get(key, []))
            )
        null_summaries.append({"replicate": replicate, **pair_diagnostics(null_pairs)})

    p95_null = [float(row["p95_absolute_delta_cm"]) for row in null_summaries]
    wasserstein_null = [float(row["wasserstein_distance_cm"]) for row in null_summaries]
    geometry_pass = (
        float(observed["p95_absolute_delta_cm"]) <= max(p95_null)
        and float(observed["wasserstein_distance_cm"]) <= max(wasserstein_null)
    )
    ancestry_support = {
        "AFR": sum(marker.ref_minor_afr > 0 for marker in selected_rare),
        "EUR": sum(marker.ref_minor_eur > 0 for marker in selected_rare),
        "ASIA": sum(marker.ref_minor_asia > 0 for marker in selected_rare),
    }
    b0_ids = {marker.site_id for marker in prepared["b0"]}
    rare_ids = {marker.site_id for marker in selected_rare}
    control_ids = {marker.site_id for marker in selected_controls}
    parity_pass = (
        len(selected_rare) == len(selected_controls) == sum(k_by_bin.values())
        and len(rare_ids) == len(selected_rare)
        and len(control_ids) == len(selected_controls)
        and not (b0_ids & rare_ids or b0_ids & control_ids or rare_ids & control_ids)
    )
    return {
        "K": len(selected_rare),
        "geometry_pass": geometry_pass,
        "parity_pass": parity_pass,
        "ancestry_pass": all(value > 0 for value in ancestry_support.values()),
        "ancestry_support": ancestry_support,
        "rare_common": observed,
        "common_common_null": {
            "replicates": len(null_summaries),
            "p95_absolute_delta_cm_min": min(p95_null),
            "p95_absolute_delta_cm_median": sorted(p95_null)[len(p95_null) // 2],
            "p95_absolute_delta_cm_max": max(p95_null),
            "p95_monte_carlo_rank": monte_carlo_rank(float(observed["p95_absolute_delta_cm"]), p95_null),
            "wasserstein_distance_cm_min": min(wasserstein_null),
            "wasserstein_distance_cm_median": sorted(wasserstein_null)[len(wasserstein_null) // 2],
            "wasserstein_distance_cm_max": max(wasserstein_null),
            "wasserstein_monte_carlo_rank": monte_carlo_rank(float(observed["wasserstein_distance_cm"]), wasserstein_null),
        },
        "per_bin": per_bin,
        "k_by_bin": k_by_bin,
        "rare_markers": selected_rare,
        "control_markers": selected_controls,
        "pairs": observed_pairs,
        "null_summaries": null_summaries,
    }


def public_evaluation(result: dict) -> dict:
    excluded = {"rare_markers", "control_markers", "pairs", "null_summaries"}
    return {key: value for key, value in result.items() if key not in excluded}


def write_screen_table(path: Path, screens: list[dict]) -> None:
    columns = [
        "fraction", "target_K", "pass", "parity_pass", "ancestry_pass",
        "geometry_pass", "rare_common_p95_delta_cm", "null_p95_max_cm",
        "rare_common_wasserstein_cm", "null_wasserstein_max_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in screens:
            result = row["evaluation"]
            writer.writerow({
                "fraction": row["fraction"],
                "target_K": result["K"],
                "pass": row["pass"],
                "parity_pass": result["parity_pass"],
                "ancestry_pass": result["ancestry_pass"],
                "geometry_pass": result["geometry_pass"],
                "rare_common_p95_delta_cm": result["rare_common"]["p95_absolute_delta_cm"],
                "null_p95_max_cm": result["common_common_null"]["p95_absolute_delta_cm_max"],
                "rare_common_wasserstein_cm": result["rare_common"]["wasserstein_distance_cm"],
                "null_wasserstein_max_cm": result["common_common_null"]["wasserstein_distance_cm_max"],
            })


def write_outputs(outdir: Path, prefix: str, prepared: dict, evaluation: dict | None) -> None:
    b0 = prepared["b0"] if evaluation is not None else []
    rare = evaluation["rare_markers"] if evaluation is not None else []
    controls = evaluation["control_markers"] if evaluation is not None else []
    pairs = evaluation["pairs"] if evaluation is not None else []
    nulls = evaluation["null_summaries"] if evaluation is not None else []
    write_marker_manifest(outdir / f"{prefix}_B0.tsv.gz", "B0", b0)
    write_marker_manifest(outdir / f"{prefix}_BR_additions.tsv.gz", "BR_addition", rare)
    write_marker_manifest(outdir / f"{prefix}_BS_additions.tsv.gz", "BS_addition", controls)
    write_pair_table(outdir / f"{prefix}_BR_BS_pairs.tsv.gz", pairs, rare)
    write_null_table(outdir / f"{prefix}_common_common_null.tsv", nulls)


def run_development(args: argparse.Namespace, contract: dict) -> dict:
    prepared = prepare_markers(args, contract, "development")
    total_capacity = sum(prepared["capacity"].values())
    screens: list[dict] = []
    for fraction in contract["development"]["capacity_fractions"]:
        target_k = math.floor(float(fraction) * total_capacity + 1e-12)
        k_by_bin = allocate_exact_k(prepared["capacity"], target_k)
        evaluation = evaluate_configuration(prepared, k_by_bin, contract)
        passed = evaluation["parity_pass"] and evaluation["ancestry_pass"] and evaluation["geometry_pass"]
        screens.append({"fraction": float(fraction), "pass": passed, "evaluation": evaluation})
    passing = [screen for screen in screens if screen["pass"]]
    selected = max(passing, key=lambda row: row["fraction"]) if passing else None
    decision = "DEV_CONFIGURATION_FROZEN" if selected else "STOP_DEV_GEOMETRY"

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_screen_table(args.outdir / "m28b_v4_dev_screens.tsv", screens)
    write_outputs(
        args.outdir,
        "m28b_v4_dev",
        prepared,
        None if selected is None else selected["evaluation"],
    )
    frozen = {
        "stage": contract["stage"],
        "decision": decision,
        "preregistration_sha256": sha256(args.preregistration),
        "development_tree_sha256": prepared["hashes"]["tree_sequence"],
        "bin_width_cm": prepared["width"],
        "selected_fraction": None if selected is None else selected["fraction"],
        "frozen_K": None if selected is None else selected["evaluation"]["K"],
        "fixed_hash_salt": contract["development"]["fixed_hash_salt"],
    }
    write_json(args.outdir / "m28b_v4_frozen_selection.json", frozen)
    report = {
        "stage": contract["stage"],
        "phase": "development",
        "scope": contract["scope"],
        "decision": decision,
        "hashes": prepared["hashes"],
        "inventory": {
            **prepared["inventory"],
            "rare_REF1_in_shared_interval": len(prepared["rare"]),
            "common_ref_polymorphic": len(prepared["common"]),
            "B0": len(prepared["b0"]),
            "maximum_K_capacity": total_capacity,
        },
        "screens": [
            {"fraction": row["fraction"], "pass": row["pass"], "evaluation": public_evaluation(row["evaluation"])}
            for row in screens
        ],
        "selected": None if selected is None else {
            "fraction": selected["fraction"],
            "evaluation": public_evaluation(selected["evaluation"]),
        },
        "gates": {
            "V4_0_INPUT_IDENTITY": True,
            "V4_1_ACCESS_BOUNDARY": True,
            "V4_2_DEV_SELECTION": selected is not None,
            "V4_3_B0_AND_PARITY": selected is not None and len(prepared["b0"]) == 79791 and selected["evaluation"]["parity_pass"],
            "V4_4_ANCESTRY_SCOPE": selected is not None and selected["evaluation"]["ancestry_pass"],
            "V4_7_SCOPE": True,
        },
        "interpretation": "Technical DEV selection only; no validation seed, TARGET, truth or LAI performance entered this phase.",
    }
    write_json(args.outdir / "m28b_v4_dev.public.json", report)
    return report


def run_validation(args: argparse.Namespace, contract: dict) -> dict:
    if args.frozen_selection is None:
        raise ValueError("Validation requires --frozen-selection")
    frozen = load_json(args.frozen_selection)
    if frozen["preregistration_sha256"] != sha256(args.preregistration):
        raise ValueError("Frozen DEV selection belongs to another preregistration")
    prepared = prepare_markers(args, contract, "validation")
    frozen_k = frozen["frozen_K"]
    if frozen_k is None:
        evaluation = None
        decision = "STOP_DEV_GEOMETRY"
    elif sum(prepared["capacity"].values()) < int(frozen_k):
        evaluation = None
        decision = "STOP_VALIDATION_CAPACITY"
    else:
        k_by_bin = allocate_exact_k(prepared["capacity"], int(frozen_k))
        evaluation = evaluate_configuration(prepared, k_by_bin, contract)
        passed = evaluation["parity_pass"] and evaluation["ancestry_pass"] and evaluation["geometry_pass"]
        decision = "GO_PREREGISTER_GENERIC_LAI_PILOT" if passed else "STOP_VALIDATION_GEOMETRY"

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.outdir, "m28b_v4_validation", prepared, evaluation)
    report = {
        "stage": contract["stage"],
        "phase": "single_untouched_validation",
        "scope": contract["scope"],
        "decision": decision,
        "hashes": {**prepared["hashes"], "frozen_selection": sha256(args.frozen_selection)},
        "frozen": frozen,
        "inventory": {
            **prepared["inventory"],
            "rare_REF1_in_shared_interval": len(prepared["rare"]),
            "common_ref_polymorphic": len(prepared["common"]),
            "B0": len(prepared["b0"]),
            "maximum_K_capacity": sum(prepared["capacity"].values()),
        },
        "evaluation": None if evaluation is None else public_evaluation(evaluation),
        "gates": {
            "V4_0_INPUT_IDENTITY": True,
            "V4_1_ACCESS_BOUNDARY": True,
            "V4_2_DEV_SELECTION": frozen_k is not None,
            "V4_3_B0_AND_PARITY": evaluation is not None and len(prepared["b0"]) == 79791 and evaluation["parity_pass"],
            "V4_4_ANCESTRY_SCOPE": evaluation is not None and evaluation["ancestry_pass"],
            "V4_5_VALIDATION_GEOMETRY": evaluation is not None and evaluation["geometry_pass"],
            "V4_6_REPRODUCIBILITY": None,
            "V4_7_SCOPE": True,
        },
        "interpretation": "Single technical validation of marker geometry only. Passing does not show LAI improvement or transfer to DNABR/NAM.",
    }
    write_json(args.outdir / "m28b_v4_validation.public.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("development", "validation"))
    parser.add_argument("--tree-sequence", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--baseline-template", required=True, type=Path)
    parser.add_argument("--m28-preregistration", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--frozen-selection", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_json(args.preregistration)
    report = run_development(args, contract) if args.phase == "development" else run_validation(args, contract)
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
