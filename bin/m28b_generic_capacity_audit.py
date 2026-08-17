#!/usr/bin/env python3
"""Build and audit generic simulated B0/BR/BS marker comparators for M28B-v3.

The authenticated real baseline contributes only its independently measured
count of REF-polymorphic markers and its chromosome span. Its positions are not
used as a template. Allocation reads FREQ and REF_LAI only and never accepts
TARGET, DONOR, mosaic truth, or LAI performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m28_simulation_preflight import load_contract as load_m28_contract, rare_under_contract, read_genetic_map  # noqa: E402
from m28b_joint_capacity_audit import read_baseline_template  # noqa: E402
from m28b_marker_capacity_audit import (  # noqa: E402
    Marker,
    MarkerPair,
    TemplatePosition,
    bp_to_cm,
    deterministic_gzip_text,
    inventory_markers,
    load_allowed_pools,
    load_json,
    nearest_monotonic_pairs,
    pair_diagnostics,
    sha256,
    verify_hash,
    write_json,
    write_marker_manifest,
)


def capacity_bin(cm: float, start_cm: float, width_cm: float) -> int:
    return int(math.floor((cm - start_cm) / width_cm + 1e-12))


def grouped(markers: list[Marker], start_cm: float, width_cm: float) -> dict[int, list[Marker]]:
    result: dict[int, list[Marker]] = {}
    for marker in markers:
        result.setdefault(capacity_bin(marker.cm, start_cm, width_cm), []).append(marker)
    return result


def stable_key(marker: Marker, salt: str, purpose: str) -> tuple[str, int, int]:
    digest = hashlib.sha256(
        f"{salt}:{purpose}:{marker.site_id}:{marker.bp}".encode()
    ).hexdigest()
    return digest, marker.bp, marker.site_id


def hamilton_quotas(counts: dict[int, int], target: int) -> dict[int, int]:
    """Allocate an exact target proportionally using largest remainders."""

    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f"B0 target {target} is outside common capacity {total}")
    raw = {key: target * count / total for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = target - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    if sum(quotas.values()) != target or any(quotas[key] > counts[key] for key in counts):
        raise RuntimeError("Hamilton allocation did not preserve the exact feasible target")
    return quotas


def monte_carlo_rank(observed: float, null_values: list[float]) -> float:
    """One-sided finite-sample rank; large distance is the adverse direction."""

    return (1 + sum(value >= observed for value in null_values)) / (len(null_values) + 1)


def pair_markers(queries: list[Marker], candidates: list[Marker]) -> list[MarkerPair]:
    pairs = nearest_monotonic_pairs(
        [TemplatePosition(marker.bp, marker.cm) for marker in queries], candidates
    )
    if pairs is None:
        raise RuntimeError("Pairing contradicted the previously counted bin capacity")
    return pairs


def assign_width(
    rare: list[Marker],
    rare_ref2: list[Marker],
    common: list[Marker],
    start_cm: float,
    width_cm: float,
    target_b0: int,
    salt: str,
    null_replicates: int,
) -> dict:
    rare_bins = grouped(rare, start_cm, width_cm)
    rare_ref2_bins = grouped(rare_ref2, start_cm, width_cm)
    common_bins = grouped(common, start_cm, width_cm)
    quotas = hamilton_quotas({key: len(value) for key, value in common_bins.items()}, target_b0)
    all_bins = sorted(set(common_bins) | set(rare_bins))

    selected_b0: list[Marker] = []
    selected_rare: list[Marker] = []
    selected_controls: list[Marker] = []
    observed_pairs: list[MarkerPair] = []
    bin_state: dict[int, dict[str, list[Marker] | int]] = {}
    per_bin: list[dict] = []
    sensitivity_ref2_k = 0
    null_capacity_pass = True

    for key in all_bins:
        ordered_common = sorted(
            common_bins.get(key, []),
            key=lambda marker: stable_key(marker, salt, "B0"),
        )
        quota = quotas.get(key, 0)
        b0_here = ordered_common[:quota]
        reserve_here = ordered_common[quota:]
        ordered_rare = sorted(
            rare_bins.get(key, []),
            key=lambda marker: stable_key(marker, salt, "BR"),
        )
        k_bin = min(len(ordered_rare), len(reserve_here))
        rare_here = ordered_rare[:k_bin]
        pairs_here = pair_markers(rare_here, reserve_here) if k_bin else []
        controls_here = [pair.control for pair in pairs_here]
        if len(b0_here) < k_bin:
            null_capacity_pass = False
        sensitivity_ref2_k += min(len(rare_ref2_bins.get(key, [])), len(reserve_here))
        selected_b0.extend(b0_here)
        selected_rare.extend(rare_here)
        selected_controls.extend(controls_here)
        observed_pairs.extend(pairs_here)
        bin_state[key] = {
            "b0": b0_here,
            "reserve": reserve_here,
            "k": k_bin,
        }
        per_bin.append({
            "bin": key,
            "common_candidates": len(ordered_common),
            "b0": len(b0_here),
            "common_reserve": len(reserve_here),
            "rare_eligible": len(ordered_rare),
            "K": k_bin,
        })

    observed = pair_diagnostics(observed_pairs)
    null_summaries: list[dict] = []
    if null_capacity_pass and observed_pairs:
        for replicate in range(null_replicates):
            null_pairs: list[MarkerPair] = []
            for key in all_bins:
                state = bin_state[key]
                k_bin = int(state["k"])
                if not k_bin:
                    continue
                b0_here = list(state["b0"])
                reserve_here = list(state["reserve"])
                pseudo_queries = sorted(
                    b0_here,
                    key=lambda marker: stable_key(marker, salt, f"NULL_{replicate}"),
                )[:k_bin]
                null_pairs.extend(pair_markers(pseudo_queries, reserve_here))
            null_summaries.append({"replicate": replicate, **pair_diagnostics(null_pairs)})

    p95_null = [float(row["p95_absolute_delta_cm"]) for row in null_summaries]
    wasserstein_null = [float(row["wasserstein_distance_cm"]) for row in null_summaries]
    geometry_pass = bool(null_summaries) and (
        float(observed["p95_absolute_delta_cm"]) <= max(p95_null)
        and float(observed["wasserstein_distance_cm"]) <= max(wasserstein_null)
    )
    ancestry_support = {
        "AFR": sum(marker.ref_minor_afr > 0 for marker in selected_rare),
        "EUR": sum(marker.ref_minor_eur > 0 for marker in selected_rare),
        "ASIA": sum(marker.ref_minor_asia > 0 for marker in selected_rare),
    }
    k_global = len(selected_rare)
    b0_pass = len(selected_b0) == target_b0 and len({marker.site_id for marker in selected_b0}) == target_b0
    parity_pass = (
        k_global > 0
        and len(selected_controls) == k_global
        and len({marker.site_id for marker in selected_controls}) == k_global
        and not ({marker.site_id for marker in selected_b0} & {marker.site_id for marker in selected_controls})
    )
    ancestry_pass = all(value > 0 for value in ancestry_support.values())
    passed = b0_pass and parity_pass and ancestry_pass and geometry_pass
    return {
        "bin_width_cm": width_cm,
        "pass": passed,
        "B0": len(selected_b0),
        "K": k_global,
        "sensitivity_REF2_K_capacity": sensitivity_ref2_k,
        "b0_pass": b0_pass,
        "parity_pass": parity_pass,
        "ancestry_pass": ancestry_pass,
        "null_capacity_pass": null_capacity_pass,
        "geometry_pass": geometry_pass,
        "ancestry_support": ancestry_support,
        "rare_common": observed,
        "common_common_null": {
            "replicates": len(null_summaries),
            "p95_absolute_delta_cm_min": min(p95_null) if p95_null else None,
            "p95_absolute_delta_cm_median": sorted(p95_null)[len(p95_null) // 2] if p95_null else None,
            "p95_absolute_delta_cm_max": max(p95_null) if p95_null else None,
            "p95_monte_carlo_rank": monte_carlo_rank(float(observed["p95_absolute_delta_cm"]), p95_null) if p95_null else None,
            "wasserstein_distance_cm_min": min(wasserstein_null) if wasserstein_null else None,
            "wasserstein_distance_cm_median": sorted(wasserstein_null)[len(wasserstein_null) // 2] if wasserstein_null else None,
            "wasserstein_distance_cm_max": max(wasserstein_null) if wasserstein_null else None,
            "wasserstein_monte_carlo_rank": monte_carlo_rank(float(observed["wasserstein_distance_cm"]), wasserstein_null) if wasserstein_null else None,
        },
        "per_bin": per_bin,
        "null_summaries": null_summaries,
        "b0_markers": selected_b0,
        "rare_markers": selected_rare,
        "control_markers": selected_controls,
        "pairs": observed_pairs,
    }


def public_screen(result: dict) -> dict:
    excluded = {"b0_markers", "rare_markers", "control_markers", "pairs", "null_summaries"}
    return {key: value for key, value in result.items() if key not in excluded}


def write_screen_table(path: Path, screens: list[dict]) -> None:
    columns = [
        "bin_width_cm", "pass", "B0", "K", "sensitivity_REF2_K_capacity",
        "b0_pass", "parity_pass", "ancestry_pass", "null_capacity_pass",
        "geometry_pass", "rare_common_p95_delta_cm", "null_p95_max_cm",
        "rare_common_wasserstein_cm", "null_wasserstein_max_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in screens:
            writer.writerow({
                "bin_width_cm": result["bin_width_cm"],
                "pass": result["pass"],
                "B0": result["B0"],
                "K": result["K"],
                "sensitivity_REF2_K_capacity": result["sensitivity_REF2_K_capacity"],
                "b0_pass": result["b0_pass"],
                "parity_pass": result["parity_pass"],
                "ancestry_pass": result["ancestry_pass"],
                "null_capacity_pass": result["null_capacity_pass"],
                "geometry_pass": result["geometry_pass"],
                "rare_common_p95_delta_cm": result["rare_common"]["p95_absolute_delta_cm"],
                "null_p95_max_cm": result["common_common_null"]["p95_absolute_delta_cm_max"],
                "rare_common_wasserstein_cm": result["rare_common"]["wasserstein_distance_cm"],
                "null_wasserstein_max_cm": result["common_common_null"]["wasserstein_distance_cm_max"],
            })


def write_pair_table(path: Path, pairs: list[MarkerPair], rare: list[Marker]) -> None:
    rare_by_position = {(marker.bp, marker.cm): marker for marker in rare}
    with deterministic_gzip_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "rare_site_id", "rare_position", "rare_cm", "control_site_id",
            "control_position", "control_cm", "absolute_delta_bp", "absolute_delta_cm",
        ))
        for pair in sorted(pairs, key=lambda value: (value.query_bp, value.control.bp)):
            marker = rare_by_position[(pair.query_bp, pair.query_cm)]
            writer.writerow((
                marker.site_id, marker.bp, f"{marker.cm:.12g}", pair.control.site_id,
                pair.control.bp, f"{pair.control.cm:.12g}", abs(marker.bp - pair.control.bp),
                f"{abs(marker.cm - pair.control.cm):.12g}",
            ))


def write_null_table(path: Path, summaries: list[dict]) -> None:
    columns = [
        "replicate", "pairs", "median_absolute_delta_cm", "p95_absolute_delta_cm",
        "maximum_absolute_delta_cm", "wasserstein_distance_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in columns})


def run(args: argparse.Namespace) -> dict:
    contract = load_json(args.preregistration)
    m28_contract = load_m28_contract(args.m28_preregistration)
    authenticated = contract["authenticated_inputs"]
    baseline_expected = authenticated["baseline_template"]
    hashes = {
        "tree_sequence": verify_hash(args.tree_sequence, authenticated["m28_tree_sequence_sha256"], "tree sequence"),
        "pool_manifest": verify_hash(args.pool_manifest, authenticated["m28_pool_manifest_sha256"], "pool manifest"),
        "genetic_map": verify_hash(args.genetic_map, authenticated["genetic_map_sha256"], "genetic map"),
        "m28_preregistration": verify_hash(args.m28_preregistration, authenticated["m28_preflight_contract_sha256"], "M28 preregistration"),
        "baseline_template": verify_hash(args.baseline_template, baseline_expected["sha256"], "baseline template"),
        "m28b_v3_preregistration": sha256(args.preregistration),
    }
    genetic_map = read_genetic_map(args.genetic_map, m28_contract)
    baseline = read_baseline_template(args.baseline_template, genetic_map, baseline_expected)
    baseline_poly = sum(row.ref_polymorphic for row in baseline)
    pools = load_allowed_pools(args.pool_manifest, m28_contract["source_populations"]["labels"])

    import tskit

    tree_sequence = tskit.load(str(args.tree_sequence))
    markers, inventory = inventory_markers(tree_sequence, pools, genetic_map, m28_contract)
    first_bp = int(baseline_expected["first_position"])
    last_bp = int(baseline_expected["last_position"])
    interval_markers = [marker for marker in markers if first_bp <= marker.bp <= last_bp]
    ref_haplotypes = int(inventory["ref_total_haplotypes"])
    rare = [
        marker for marker in interval_markers
        if rare_under_contract({"mac": marker.mac, "maf": marker.maf}, m28_contract)
        and marker.ref_minor_total >= 1
    ]
    rare_ref2 = [marker for marker in rare if marker.ref_minor_total >= 2]
    common = [
        marker for marker in interval_markers
        if marker.maf >= 0.01 and 0 < marker.ref_minor_total < ref_haplotypes
    ]
    allocation = contract["allocation"]
    target_b0 = int(allocation["b0_target_count"])
    if baseline_poly != target_b0:
        raise ValueError("Authenticated baseline informative count no longer matches B0 anchor")
    start_cm = bp_to_cm(genetic_map, first_bp)
    screens: list[dict] = []
    primary = None
    for width in allocation["bin_widths_cm"]:
        result = assign_width(
            rare, rare_ref2, common, start_cm, float(width), target_b0,
            allocation["fixed_hash_salt"], int(allocation["null_replicates"]),
        )
        screens.append(result)
        if primary is None and result["pass"]:
            primary = result

    b0 = [] if primary is None else primary["b0_markers"]
    br = [] if primary is None else primary["rare_markers"]
    bs = [] if primary is None else primary["control_markers"]
    pairs = [] if primary is None else primary["pairs"]
    null_summaries = [] if primary is None else primary["null_summaries"]
    b0_ids = {marker.site_id for marker in b0}
    br_ids = {marker.site_id for marker in br}
    bs_ids = {marker.site_id for marker in bs}
    interval_pass = all(first_bp <= marker.bp <= last_bp for marker in b0 + br + bs)
    b0_pass = bool(primary) and len(b0) == target_b0 and len(b0_ids) == target_b0
    parity_pass = (
        bool(primary)
        and len(br) == len(bs) > 0
        and len(br_ids) == len(br)
        and len(bs_ids) == len(bs)
        and not (b0_ids & br_ids or b0_ids & bs_ids or br_ids & bs_ids)
    )
    ancestry_pass = bool(primary) and primary["ancestry_pass"]
    geometry_pass = bool(primary) and primary["geometry_pass"]
    gates = {
        "V3_0_INPUT_IDENTITY": True,
        "V3_1_ACCESS_BOUNDARY": True,
        "V3_2_SHARED_INTERVAL": interval_pass,
        "V3_3_B0_ANCHOR": b0_pass,
        "V3_4_ARM_PARITY": parity_pass,
        "V3_5_RARE_SCOPE": ancestry_pass,
        "V3_6_GEOMETRY": geometry_pass,
        "V3_7_REPRODUCIBILITY": None,
        "V3_8_SCOPE": True,
    }
    if primary is None:
        if not any(screen["b0_pass"] and screen["parity_pass"] for screen in screens):
            decision = "STOP_CAPACITY"
        elif not any(screen["ancestry_pass"] for screen in screens):
            decision = "STOP_ANCESTRY_SCOPE"
        else:
            decision = "STOP_GEOMETRY"
    elif not interval_pass or not b0_pass or not parity_pass:
        decision = "STOP_PARITY"
    elif not ancestry_pass:
        decision = "STOP_ANCESTRY_SCOPE"
    elif not geometry_pass:
        decision = "STOP_GEOMETRY"
    else:
        decision = "AUDIT_COMPLETE_PENDING_REPRODUCIBILITY"

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_screen_table(args.outdir / "m28b_v3_capacity_screens.tsv", screens)
    write_marker_manifest(args.outdir / "m28b_v3_B0.tsv.gz", "B0", b0)
    write_marker_manifest(args.outdir / "m28b_v3_BR_additions.tsv.gz", "BR_addition", br)
    write_marker_manifest(args.outdir / "m28b_v3_BS_additions.tsv.gz", "BS_addition", bs)
    write_pair_table(args.outdir / "m28b_v3_BR_BS_pairs.tsv.gz", pairs, br)
    write_null_table(args.outdir / "m28b_v3_common_common_null.tsv", null_summaries)
    report = {
        "stage": contract["stage"],
        "scope": contract["scope"],
        "decision": decision,
        "hashes": hashes,
        "interval": {"first_bp": first_bp, "last_bp": last_bp, "origin_cm": start_cm},
        "inventory": {
            **inventory,
            "sites_in_shared_interval": len(interval_markers),
            "rare_REF1_in_shared_interval": len(rare),
            "rare_REF2_in_shared_interval": len(rare_ref2),
            "common_ref_polymorphic": len(common),
            "baseline_informative_anchor": baseline_poly,
        },
        "screens": [public_screen(result) for result in screens],
        "primary_construction": None if primary is None else public_screen(primary),
        "gates": gates,
        "interpretation": "Generic simulated marker capacity and geometry only. The real baseline contributes a count anchor and interval, not positions, LD, weights or performance. No LAI or truth metric was run.",
    }
    write_json(args.outdir / "m28b_v3_capacity.public.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-sequence", required=True, type=Path)
    parser.add_argument("--pool-manifest", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--baseline-template", required=True, type=Path)
    parser.add_argument("--m28-preregistration", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
