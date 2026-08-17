#!/usr/bin/env python3
"""Jointly allocate the M28B-v2 B0 baseline and matched BR/BS additions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m28_simulation_preflight import load_contract as load_m28_contract, rare_under_contract, read_genetic_map  # noqa: E402
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
    open_text,
    pair_diagnostics,
    sha256,
    verify_hash,
    write_json,
    write_marker_manifest,
)


@dataclass(frozen=True)
class BaselineTemplate:
    bp: int
    cm: float
    ref_polymorphic: bool


@dataclass(frozen=True)
class JointAssignment:
    query_kind: str
    query_site_id: int | None
    query_bp: int
    query_cm: float
    query_ref_polymorphic: bool
    candidate: Marker


def genotype_is_polymorphic(sample_fields: list[str]) -> bool:
    observed: set[int] = set()
    for field in sample_fields:
        gt = field.split(":", 1)[0].replace("|", "/")
        for allele in gt.split("/"):
            if allele != ".":
                observed.add(int(allele))
    if not observed:
        raise ValueError("Baseline-template row has no called reference genotype")
    return len(observed) > 1


def read_baseline_template(path: Path, genetic_map, expected: dict) -> list[BaselineTemplate]:
    rows: list[BaselineTemplate] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise ValueError("Baseline-template VCF has no reference genotypes")
            if fields[0] not in {"22", "chr22"}:
                raise ValueError(f"Unexpected baseline-template chromosome: {fields[0]}")
            bp = int(fields[1])
            rows.append(BaselineTemplate(
                bp=bp,
                cm=bp_to_cm(genetic_map, bp),
                ref_polymorphic=genotype_is_polymorphic(fields[9:]),
            ))
    if len(rows) != int(expected["records"]):
        raise ValueError("Baseline-template record count mismatch")
    positions = [row.bp for row in rows]
    if positions != sorted(positions) or len(set(positions)) != int(expected["unique_positions"]):
        raise ValueError("Baseline-template positions are not sorted and unique")
    if positions[0] != int(expected["first_position"]) or positions[-1] != int(expected["last_position"]):
        raise ValueError("Baseline-template bounds mismatch")
    polymorphic = sum(row.ref_polymorphic for row in rows)
    monomorphic = len(rows) - polymorphic
    if polymorphic != int(expected["known_answer_ref_polymorphic"]):
        raise ValueError(f"Baseline polymorphic count mismatch: {polymorphic}")
    if monomorphic != int(expected["known_answer_ref_monomorphic"]):
        raise ValueError(f"Baseline monomorphic count mismatch: {monomorphic}")
    return rows


def marker_is_ref_polymorphic(marker: Marker, ref_haplotypes: int) -> bool:
    return 0 < marker.ref_minor_total < ref_haplotypes


def capacity_bin(cm: float, start_cm: float, width_cm: float) -> int:
    import math

    return int(math.floor((cm - start_cm) / width_cm + 1e-12))


def grouped(values, start_cm: float, width_cm: float, cm_getter):
    result: dict[int, list] = {}
    for value in values:
        result.setdefault(capacity_bin(cm_getter(value), start_cm, width_cm), []).append(value)
    return result


def stable_rare_key(marker: Marker, salt: str) -> tuple[str, int, int]:
    digest = hashlib.sha256(f"{salt}:{marker.site_id}:{marker.bp}".encode()).hexdigest()
    return digest, marker.bp, marker.site_id


def nearest_with_reuse(queries: list[Marker], candidates: list[Marker]) -> list[MarkerPair]:
    import bisect

    ordered = sorted(candidates, key=lambda marker: (marker.cm, marker.bp, marker.site_id))
    cms = [marker.cm for marker in ordered]
    pairs: list[MarkerPair] = []
    for query in sorted(queries, key=lambda marker: (marker.cm, marker.bp, marker.site_id)):
        index = bisect.bisect_left(cms, query.cm)
        choices = []
        if index < len(ordered):
            choices.append(ordered[index])
        if index > 0:
            choices.append(ordered[index - 1])
        control = min(
            choices,
            key=lambda marker: (abs(marker.cm - query.cm), abs(marker.bp - query.bp), marker.site_id),
        )
        pairs.append(MarkerPair(query.bp, query.cm, control))
    return pairs


def assign_width(
    templates: list[BaselineTemplate],
    rare: list[Marker],
    nonrare_poly: list[Marker],
    nonrare_mono: list[Marker],
    start_cm: float,
    width_cm: float,
    salt: str,
    ref_haplotypes: int,
    sensitivity_rare: list[Marker],
) -> dict:
    template_bins = grouped(templates, start_cm, width_cm, lambda value: value.cm)
    rare_bins = grouped(rare, start_cm, width_cm, lambda value: value.cm)
    sensitivity_bins = grouped(sensitivity_rare, start_cm, width_cm, lambda value: value.cm)
    poly_bins = grouped(nonrare_poly, start_cm, width_cm, lambda value: value.cm)
    mono_bins = grouped(nonrare_mono, start_cm, width_cm, lambda value: value.cm)
    all_bins = sorted(set(template_bins) | set(rare_bins) | set(poly_bins) | set(mono_bins))
    deficits: list[dict] = []
    per_bin: list[dict] = []
    K = 0
    sensitivity_K = 0
    selected_rare: list[Marker] = []

    for key in all_bins:
        baseline = template_bins.get(key, [])
        poly_demand = sum(row.ref_polymorphic for row in baseline)
        mono_demand = len(baseline) - poly_demand
        poly_supply = len(poly_bins.get(key, []))
        mono_supply = len(mono_bins.get(key, []))
        if poly_supply < poly_demand or mono_supply < mono_demand:
            deficits.append({
                "bin": key,
                "poly_deficit": max(poly_demand - poly_supply, 0),
                "mono_deficit": max(mono_demand - mono_supply, 0),
            })
        residual_poly = max(poly_supply - poly_demand, 0)
        eligible = sorted(rare_bins.get(key, []), key=lambda marker: stable_rare_key(marker, salt))
        bin_K = min(len(eligible), residual_poly)
        sensitivity_bin_K = min(len(sensitivity_bins.get(key, [])), residual_poly)
        selected_rare.extend(eligible[:bin_K])
        K += bin_K
        sensitivity_K += sensitivity_bin_K
        per_bin.append({
            "bin": key,
            "baseline_poly_demand": poly_demand,
            "baseline_mono_demand": mono_demand,
            "nonrare_poly_supply": poly_supply,
            "nonrare_mono_supply": mono_supply,
            "rare_eligible": len(eligible),
            "K": bin_K,
        })

    result = {
        "bin_width_cm": width_cm,
        "baseline_bins_with_deficit": len(deficits),
        "maximum_poly_deficit": max((row["poly_deficit"] for row in deficits), default=0),
        "maximum_mono_deficit": max((row["mono_deficit"] for row in deficits), default=0),
        "K": K,
        "sensitivity_REF2_K_capacity": sensitivity_K,
        "per_bin": per_bin,
        "pass": not deficits and K > 0,
        "assignments": [],
        "selected_rare": [],
    }
    if deficits or K == 0:
        return result

    assignments: list[JointAssignment] = []
    for key in all_bins:
        baseline = template_bins.get(key, [])
        selected_here = [
            marker for marker in selected_rare
            if capacity_bin(marker.cm, start_cm, width_cm) == key
        ]
        joint_queries = [
            ("B0", None, row.bp, row.cm, True)
            for row in baseline if row.ref_polymorphic
        ] + [
            ("BS", marker.site_id, marker.bp, marker.cm, True)
            for marker in selected_here
        ]
        joint_queries.sort(key=lambda row: (row[3], row[2], row[0], row[1] or -1))
        joint_positions = [TemplatePosition(row[2], row[3]) for row in joint_queries]
        poly_pairs = nearest_monotonic_pairs(joint_positions, poly_bins.get(key, []))
        if poly_pairs is None:
            raise RuntimeError("Joint polymorphic allocation contradicted capacity counts")
        for query, pair in zip(joint_queries, poly_pairs):
            assignments.append(JointAssignment(*query, pair.control))

        mono_queries = sorted(
            [row for row in baseline if not row.ref_polymorphic],
            key=lambda row: (row.cm, row.bp),
        )
        mono_pairs = nearest_monotonic_pairs(
            [TemplatePosition(row.bp, row.cm) for row in mono_queries],
            mono_bins.get(key, []),
        )
        if mono_pairs is None:
            raise RuntimeError("Monomorphic B0 allocation contradicted capacity counts")
        for query, pair in zip(mono_queries, mono_pairs):
            assignments.append(JointAssignment(
                "B0", None, query.bp, query.cm, False, pair.control
            ))

    b0_assignments = [row for row in assignments if row.query_kind == "B0"]
    bs_assignments = [row for row in assignments if row.query_kind == "BS"]
    selected_by_id = {marker.site_id: marker for marker in selected_rare}
    b0_pairs = [MarkerPair(row.query_bp, row.query_cm, row.candidate) for row in b0_assignments]
    br_bs_pairs = [MarkerPair(row.query_bp, row.query_cm, row.candidate) for row in bs_assignments]
    selected_ordered = [selected_by_id[row.query_site_id] for row in bs_assignments]
    reuse_pairs = []
    for key in all_bins:
        queries = [
            marker for marker in selected_ordered
            if capacity_bin(marker.cm, start_cm, width_cm) == key
        ]
        if queries:
            reuse_pairs.extend(nearest_with_reuse(queries, poly_bins.get(key, [])))
    ancestry_support = {
        "AFR": sum(marker.ref_minor_afr > 0 for marker in selected_ordered),
        "EUR": sum(marker.ref_minor_eur > 0 for marker in selected_ordered),
        "ASIA": sum(marker.ref_minor_asia > 0 for marker in selected_ordered),
    }
    b0_markers = [row.candidate for row in b0_assignments]
    bs_markers = [row.candidate for row in bs_assignments]
    b0_poly = sum(marker_is_ref_polymorphic(marker, ref_haplotypes) for marker in b0_markers)
    result.update({
        "assignments": assignments,
        "selected_rare": selected_ordered,
        "b0_markers": b0_markers,
        "bs_markers": bs_markers,
        "b0_pairs": b0_pairs,
        "br_bs_pairs": br_bs_pairs,
        "B0_markers": len(b0_markers),
        "B0_ref_polymorphic": b0_poly,
        "B0_ref_monomorphic": len(b0_markers) - b0_poly,
        "ancestry_support": ancestry_support,
        "B0_mapping": pair_diagnostics(b0_pairs),
        "BR_BS_matching": pair_diagnostics(br_bs_pairs),
        "BR_nearest_with_reuse": pair_diagnostics(reuse_pairs),
    })
    return result


def serializable_screen(result: dict) -> dict:
    excluded = {
        "assignments", "selected_rare", "b0_markers", "bs_markers",
        "b0_pairs", "br_bs_pairs",
    }
    return {key: value for key, value in result.items() if key not in excluded}


def write_screen_table(path: Path, screens: list[dict]) -> None:
    columns = [
        "bin_width_cm", "pass", "baseline_bins_with_deficit", "maximum_poly_deficit",
        "maximum_mono_deficit", "K", "sensitivity_REF2_K_capacity", "B0_markers",
        "B0_ref_polymorphic", "B0_ref_monomorphic",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for result in screens:
            writer.writerow({column: result.get(column, "") for column in columns})


def write_b0_mapping(path: Path, assignments: list[JointAssignment]) -> None:
    with deterministic_gzip_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "template_position", "template_cm", "template_ref_class", "site_id",
            "simulated_position", "simulated_cm", "absolute_delta_bp", "absolute_delta_cm",
        ))
        for row in sorted((value for value in assignments if value.query_kind == "B0"), key=lambda value: value.query_bp):
            writer.writerow((
                row.query_bp, f"{row.query_cm:.12g}",
                "polymorphic" if row.query_ref_polymorphic else "monomorphic",
                row.candidate.site_id, row.candidate.bp, f"{row.candidate.cm:.12g}",
                abs(row.candidate.bp - row.query_bp),
                f"{abs(row.candidate.cm - row.query_cm):.12g}",
            ))


def write_br_bs_pairs(path: Path, assignments: list[JointAssignment], rare: list[Marker]) -> None:
    rare_by_id = {marker.site_id: marker for marker in rare}
    with deterministic_gzip_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "rare_site_id", "rare_position", "rare_cm", "control_site_id",
            "control_position", "control_cm", "absolute_delta_bp", "absolute_delta_cm",
        ))
        for row in sorted((value for value in assignments if value.query_kind == "BS"), key=lambda value: value.query_bp):
            marker = rare_by_id[row.query_site_id]
            writer.writerow((
                marker.site_id, marker.bp, f"{marker.cm:.12g}", row.candidate.site_id,
                row.candidate.bp, f"{row.candidate.cm:.12g}",
                abs(row.candidate.bp - marker.bp), f"{abs(row.candidate.cm - marker.cm):.12g}",
            ))


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
        "m28b_v2_preregistration": sha256(args.preregistration),
    }
    genetic_map = read_genetic_map(args.genetic_map, m28_contract)
    templates = read_baseline_template(args.baseline_template, genetic_map, baseline_expected)
    pools = load_allowed_pools(args.pool_manifest, m28_contract["source_populations"]["labels"])

    import tskit

    ts = tskit.load(str(args.tree_sequence))
    markers, inventory = inventory_markers(ts, pools, genetic_map, m28_contract)
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
    nonrare = [marker for marker in interval_markers if marker.maf >= 0.01]
    nonrare_poly = [marker for marker in nonrare if marker_is_ref_polymorphic(marker, ref_haplotypes)]
    nonrare_mono = [marker for marker in nonrare if not marker_is_ref_polymorphic(marker, ref_haplotypes)]

    screens = []
    primary = None
    allocation = contract["joint_allocation"]
    for width in allocation["bin_widths_cm"]:
        result = assign_width(
            templates, rare, nonrare_poly, nonrare_mono, float(genetic_map.cm[0]),
            float(width), allocation["fixed_hash_salt"], ref_haplotypes, rare_ref2,
        )
        screens.append(result)
        if primary is None and result["pass"]:
            primary = result

    if primary is None:
        b0_markers: list[Marker] = []
        selected_rare: list[Marker] = []
        bs_markers: list[Marker] = []
        assignments: list[JointAssignment] = []
    else:
        b0_markers = primary["b0_markers"]
        selected_rare = primary["selected_rare"]
        bs_markers = primary["bs_markers"]
        assignments = primary["assignments"]

    b0_ids = {marker.site_id for marker in b0_markers}
    rare_ids = {marker.site_id for marker in selected_rare}
    bs_ids = {marker.site_id for marker in bs_markers}
    interval_pass = all(first_bp <= marker.bp <= last_bp for marker in b0_markers + selected_rare + bs_markers)
    b0_fidelity = bool(primary) and (
        len(b0_markers) == int(allocation["b0_target_count"])
        and primary["B0_ref_polymorphic"] == int(allocation["b0_ref_polymorphic_target"])
        and primary["B0_ref_monomorphic"] == int(allocation["b0_ref_monomorphic_target"])
        and len(b0_ids) == len(b0_markers)
    )
    parity = bool(primary) and (
        len(selected_rare) == len(bs_markers) == int(primary["K"])
        and len(rare_ids) == len(selected_rare)
        and len(bs_ids) == len(bs_markers)
        and not (b0_ids & rare_ids or b0_ids & bs_ids or rare_ids & bs_ids)
    )
    ancestry_pass = bool(primary) and all(value > 0 for value in primary["ancestry_support"].values())
    gates = {
        "V2_0_INPUT_IDENTITY": True,
        "V2_1_ACCESS_BOUNDARY": True,
        "V2_2_SHARED_INTERVAL": interval_pass,
        "V2_3_B0_FIDELITY": b0_fidelity,
        "V2_4_ARM_PARITY": parity,
        "V2_5_RARE_SCOPE": ancestry_pass,
        "V2_6_REPRODUCIBILITY": None,
        "V2_7_SCOPE": True,
    }
    if not interval_pass:
        decision = "STOP_SHARED_INTERVAL"
    elif primary is None:
        decision = "STOP_B0_FIDELITY"
    elif not b0_fidelity:
        decision = "STOP_B0_FIDELITY"
    elif not parity or not ancestry_pass:
        decision = "STOP_PARITY"
    else:
        decision = "AUDIT_COMPLETE_PENDING_REPRODUCIBILITY"

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_screen_table(args.outdir / "m28b_v2_capacity_screens.tsv", screens)
    write_marker_manifest(args.outdir / "m28b_v2_B0.tsv.gz", "B0", b0_markers)
    write_marker_manifest(args.outdir / "m28b_v2_BR_additions.tsv.gz", "BR_addition", selected_rare)
    write_marker_manifest(args.outdir / "m28b_v2_BS_additions.tsv.gz", "BS_addition", bs_markers)
    write_b0_mapping(args.outdir / "m28b_v2_B0_mapping.tsv.gz", assignments)
    write_br_bs_pairs(args.outdir / "m28b_v2_BR_BS_pairs.tsv.gz", assignments, selected_rare)
    report = {
        "stage": contract["stage"],
        "scope": contract["scope"],
        "decision": decision,
        "hashes": hashes,
        "interval": {"first_bp": first_bp, "last_bp": last_bp},
        "inventory": {
            **inventory,
            "sites_in_shared_interval": len(interval_markers),
            "rare_REF1_in_shared_interval": len(rare),
            "rare_REF2_in_shared_interval": len(rare_ref2),
            "nonrare_in_shared_interval": len(nonrare),
            "nonrare_ref_polymorphic": len(nonrare_poly),
            "nonrare_ref_monomorphic": len(nonrare_mono),
            "baseline_ref_polymorphic": sum(row.ref_polymorphic for row in templates),
            "baseline_ref_monomorphic": sum(not row.ref_polymorphic for row in templates),
        },
        "screens": [serializable_screen(result) for result in screens],
        "primary_construction": None if primary is None else {
            "bin_width_cm": primary["bin_width_cm"],
            "K": primary["K"],
            "sensitivity_REF2_K_capacity": primary["sensitivity_REF2_K_capacity"],
            "B0_markers": primary["B0_markers"],
            "B0_ref_polymorphic": primary["B0_ref_polymorphic"],
            "B0_ref_monomorphic": primary["B0_ref_monomorphic"],
            "ancestry_support": primary["ancestry_support"],
            "B0_mapping": primary["B0_mapping"],
            "BR_BS_matching": primary["BR_BS_matching"],
            "BR_nearest_with_reuse": primary["BR_nearest_with_reuse"],
        },
        "gates": gates,
        "interpretation": "Capacity and geometry only. No LAI model or performance metric was run.",
    }
    write_json(args.outdir / "m28b_v2_capacity.public.json", report)
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
