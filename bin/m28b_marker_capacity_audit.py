#!/usr/bin/env python3
"""Audit marker capacity for a leakage-safe M28 B0/BR/BS comparison.

This stage never accepts target genotypes, mosaic events, LAI truth, or donor
genotypes. Marker statistics are recomputed from the M28 tree sequence using
only FREQ and REF_LAI nodes named in the authenticated pool manifest.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from m28_simulation_preflight import (  # noqa: E402
    GeneticMap,
    load_contract as load_m28_contract,
    minor_allele_stats,
    rare_under_contract,
    read_genetic_map,
)


@dataclass(frozen=True)
class Marker:
    site_id: int
    bp: int
    cm: float
    minor_code: int
    mac: int
    an: int
    maf: float
    ref_minor_total: int
    ref_minor_afr: int
    ref_minor_eur: int
    ref_minor_asia: int
    freq_minor_carrier_individuals: int = 0


@dataclass(frozen=True)
class TemplatePosition:
    bp: int
    cm: float


@dataclass(frozen=True)
class MarkerPair:
    query_bp: int
    query_cm: float
    control: Marker


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, observed {observed}")
    return observed


def count_minor_carrier_individuals(
    tree_sequence,
    nodes: Iterable[int],
    genotype_by_node: dict[int, int],
    minor_code: int,
) -> int:
    """Count distinct diploid individuals carrying the FREQ minor allele."""

    carriers = {
        int(tree_sequence.node(node).individual)
        for node in nodes
        if int(genotype_by_node[node]) == minor_code
    }
    if -1 in carriers:
        raise ValueError("A FREQ minor-allele carrier has no source individual")
    return len(carriers)


def bp_to_cm(genetic_map: GeneticMap, bp: int) -> float:
    if bp < genetic_map.bp[0] or bp > genetic_map.bp[-1]:
        raise ValueError(f"Position {bp} is outside the authenticated genetic map")
    right = bisect.bisect_right(genetic_map.bp, bp)
    if right == 0:
        return float(genetic_map.cm[0])
    if right == len(genetic_map.bp):
        return float(genetic_map.cm[-1])
    left = right - 1
    left_bp, right_bp = genetic_map.bp[left], genetic_map.bp[right]
    left_cm, right_cm = genetic_map.cm[left], genetic_map.cm[right]
    fraction = (bp - left_bp) / (right_bp - left_bp)
    return float(left_cm + fraction * (right_cm - left_cm))


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                yield text


def load_allowed_pools(
    path: Path,
    labels: Iterable[str],
    *,
    tree_sequence=None,
    require_individual_schema: bool = False,
) -> dict[str, dict[str, list[int]]]:
    allowed_roles = {"FREQ", "REF_LAI"}
    all_roles = allowed_roles | {"DONOR"}
    label_set = set(labels)
    pools = {role: {label: [] for label in label_set} for role in allowed_roles}
    individual_assignments: dict[int, tuple[str, str, list[int]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        legacy_columns = {"role", "ancestry", "node_id", "haplotype_sha256"}
        individual_columns = {
            "role", "ancestry", "individual_id", "node_id", "node_identity_sha256"
        }
        if columns not in (legacy_columns, individual_columns):
            raise ValueError("Unexpected pool-manifest columns")
        if require_individual_schema and columns != individual_columns:
            raise ValueError("Individual-safe audit requires the v2 pool-manifest schema")
        for row in reader:
            role = row["role"]
            if role not in all_roles:
                raise ValueError(f"Unexpected pool role: {role}")
            ancestry = row["ancestry"]
            if ancestry not in label_set:
                raise ValueError(f"Unexpected ancestry in pool manifest: {ancestry}")
            node = int(row["node_id"])
            if columns == individual_columns:
                expected_identity = hashlib.sha256(f"source-node:{node}".encode()).hexdigest()
                if row["node_identity_sha256"] != expected_identity:
                    raise ValueError(f"Node identity hash mismatch for node {node}")
                individual = int(row["individual_id"])
                if tree_sequence is not None:
                    observed_individual = int(tree_sequence.node(node).individual)
                    if observed_individual != individual:
                        raise ValueError(
                            f"Manifest individual {individual} does not match tree individual "
                            f"{observed_individual} for node {node}"
                        )
                assigned = individual_assignments.setdefault(
                    individual, (role, ancestry, [])
                )
                if assigned[:2] != (role, ancestry):
                    raise ValueError(
                        f"Individual {individual} crosses roles or ancestries"
                    )
                assigned[2].append(node)
            if role in allowed_roles:
                pools[role][ancestry].append(node)
    if individual_assignments:
        incomplete = {
            individual: nodes
            for individual, (_, _, nodes) in individual_assignments.items()
            if len(nodes) != 2 or len(set(nodes)) != 2
        }
        if incomplete:
            raise ValueError(
                f"Pool manifest does not keep exactly two unique nodes per individual: "
                f"{list(incomplete)[:5]}"
            )
    for role in allowed_roles:
        for ancestry in label_set:
            if not pools[role][ancestry]:
                raise ValueError(f"Empty {role}/{ancestry} pool")
    loaded = [node for role in pools.values() for nodes in role.values() for node in nodes]
    if len(loaded) != len(set(loaded)):
        raise ValueError("FREQ and REF_LAI nodes overlap")
    return pools


def read_template(path: Path, genetic_map: GeneticMap, expected: dict) -> list[TemplatePosition]:
    positions: list[int] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 3)
            if len(fields) < 2:
                raise ValueError("Malformed baseline-template VCF row")
            chrom, raw_bp = fields[0], fields[1]
            if chrom not in {"22", "chr22"}:
                raise ValueError(f"Unexpected baseline-template chromosome: {chrom}")
            positions.append(int(raw_bp))
    if len(positions) != int(expected["records"]):
        raise ValueError("Baseline-template record count mismatch")
    if positions != sorted(positions):
        raise ValueError("Baseline-template positions are not sorted")
    if len(set(positions)) != int(expected["unique_positions"]):
        raise ValueError("Baseline-template positions are not unique")
    if positions[0] != int(expected["first_position"]) or positions[-1] != int(expected["last_position"]):
        raise ValueError("Baseline-template bounds mismatch")
    return [TemplatePosition(bp=bp, cm=bp_to_cm(genetic_map, bp)) for bp in positions]


def inventory_markers(ts, pools, genetic_map: GeneticMap, m28_contract: dict) -> tuple[list[Marker], dict]:
    sample_index = {int(node): index for index, node in enumerate(ts.samples())}
    labels = tuple(m28_contract["source_populations"]["labels"])
    if int(m28_contract.get("version", 0)) >= 2:
        by_individual: dict[int, tuple[str, str, list[int]]] = {}
        for role in ("FREQ", "REF_LAI"):
            for label in labels:
                for node in pools[role][label]:
                    individual = int(ts.node(node).individual)
                    assigned = by_individual.setdefault(individual, (role, label, []))
                    if assigned[:2] != (role, label):
                        raise ValueError(
                            f"Tree individual {individual} crosses allowed roles or ancestries"
                        )
                    assigned[2].append(node)
        malformed = {
            individual: values[2]
            for individual, values in by_individual.items()
            if len(values[2]) != 2 or len(set(values[2])) != 2
        }
        if malformed:
            raise ValueError(
                "Tree and pool manifest do not preserve complete diploid individuals: "
                f"{list(malformed)[:5]}"
            )
    freq_nodes = [node for label in labels for node in pools["FREQ"][label]]
    ref_nodes = {label: pools["REF_LAI"][label] for label in labels}
    ref_total_haplotypes = sum(len(nodes) for nodes in ref_nodes.values())
    markers: list[Marker] = []
    mac_histogram: dict[int, int] = {}
    rare_count = 0
    ref_support_patterns: dict[str, int] = {}
    for variant in ts.variants():
        freq_genotypes = [int(variant.genotypes[sample_index[node]]) for node in freq_nodes]
        stats = minor_allele_stats(freq_genotypes)
        minor = int(stats["minor_code"])
        ref_counts = {
            label: sum(int(variant.genotypes[sample_index[node]]) == minor for node in ref_nodes[label])
            for label in labels
        }
        carrier_individual_count = count_minor_carrier_individuals(
            ts,
            freq_nodes,
            {node: int(variant.genotypes[sample_index[node]]) for node in freq_nodes},
            minor,
        )
        marker = Marker(
            site_id=int(variant.site.id),
            bp=genetic_map.start_bp + int(variant.site.position),
            cm=bp_to_cm(genetic_map, genetic_map.start_bp + int(variant.site.position)),
            minor_code=minor,
            mac=int(stats["mac"]),
            an=int(stats["an"]),
            maf=float(stats["maf"]),
            ref_minor_total=sum(ref_counts.values()),
            ref_minor_afr=ref_counts["AFR"],
            ref_minor_eur=ref_counts["EUR"],
            ref_minor_asia=ref_counts["ASIA"],
            freq_minor_carrier_individuals=carrier_individual_count,
        )
        markers.append(marker)
        if rare_under_contract(stats, m28_contract):
            rare_count += 1
            mac_histogram[marker.mac] = mac_histogram.get(marker.mac, 0) + 1
            pattern = "+".join(label for label in labels if ref_counts[label] > 0) or "NONE"
            ref_support_patterns[pattern] = ref_support_patterns.get(pattern, 0) + 1
    return markers, {
        "sites": len(markers),
        "rare_sites": rare_count,
        "rare_mac_histogram": {str(key): value for key, value in sorted(mac_histogram.items())},
        "rare_ref_support_patterns": dict(sorted(ref_support_patterns.items())),
        "ref_total_haplotypes": ref_total_haplotypes,
    }


def marker_sort_key(marker: Marker) -> tuple[float, int, int]:
    return marker.cm, marker.bp, marker.site_id


def nearest_monotonic_pairs(
    queries: list[TemplatePosition], candidates: list[Marker]
) -> list[MarkerPair] | None:
    """Map ordered queries to unique ordered candidates using a deterministic greedy rule."""

    if len(candidates) < len(queries):
        return None
    candidates = sorted(candidates, key=marker_sort_key)
    cms = [marker.cm for marker in candidates]
    pairs: list[MarkerPair] = []
    previous = -1
    for query_index, query in enumerate(sorted(queries, key=lambda value: (value.cm, value.bp))):
        lower = previous + 1
        remaining_queries = len(queries) - query_index - 1
        upper = len(candidates) - remaining_queries - 1
        if lower > upper:
            return None
        insertion = bisect.bisect_left(cms, query.cm, lo=lower, hi=upper + 1)
        choices = {min(max(insertion, lower), upper)}
        if insertion - 1 >= lower:
            choices.add(insertion - 1)
        chosen = min(
            choices,
            key=lambda index: (
                abs(candidates[index].cm - query.cm),
                abs(candidates[index].bp - query.bp),
                candidates[index].site_id,
            ),
        )
        marker = candidates[chosen]
        pairs.append(MarkerPair(query.bp, query.cm, marker))
        previous = chosen
    return pairs


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def pair_diagnostics(pairs: list[MarkerPair]) -> dict:
    delta_cm = [abs(pair.control.cm - pair.query_cm) for pair in pairs]
    delta_bp = [abs(pair.control.bp - pair.query_bp) for pair in pairs]
    query_cm = sorted(pair.query_cm for pair in pairs)
    control_cm = sorted(pair.control.cm for pair in pairs)
    wasserstein = statistics.fmean(abs(left - right) for left, right in zip(query_cm, control_cm)) if pairs else None
    return {
        "pairs": len(pairs),
        "median_absolute_delta_cm": statistics.median(delta_cm) if delta_cm else None,
        "p95_absolute_delta_cm": percentile(delta_cm, 0.95),
        "maximum_absolute_delta_cm": max(delta_cm) if delta_cm else None,
        "median_absolute_delta_bp": statistics.median(delta_bp) if delta_bp else None,
        "p95_absolute_delta_bp": percentile([float(value) for value in delta_bp], 0.95),
        "maximum_absolute_delta_bp": max(delta_bp) if delta_bp else None,
        "wasserstein_distance_cm": wasserstein,
    }


def bin_index(cm: float, start_cm: float, width_cm: float) -> int:
    return int(math.floor((cm - start_cm) / width_cm + 1e-12))


def match_controls_by_bin(
    rare: list[Marker], reserve: list[Marker], start_cm: float, width_cm: float
) -> tuple[list[MarkerPair] | None, dict]:
    rare_bins: dict[int, list[Marker]] = {}
    reserve_bins: dict[int, list[Marker]] = {}
    for marker in rare:
        rare_bins.setdefault(bin_index(marker.cm, start_cm, width_cm), []).append(marker)
    for marker in reserve:
        reserve_bins.setdefault(bin_index(marker.cm, start_cm, width_cm), []).append(marker)
    missing = {
        key: len(values) - len(reserve_bins.get(key, []))
        for key, values in rare_bins.items()
        if len(reserve_bins.get(key, [])) < len(values)
    }
    capacity = sum(min(len(values), len(reserve_bins.get(key, []))) for key, values in rare_bins.items())
    if missing:
        return None, {
            "eligible_rare": len(rare),
            "matched": capacity,
            "unmatched_rare_count": len(rare) - capacity,
            "bins_without_capacity": len(missing),
            "maximum_bin_deficit": max(missing.values()),
        }
    pairs: list[MarkerPair] = []
    for key in sorted(rare_bins):
        queries = [TemplatePosition(marker.bp, marker.cm) for marker in sorted(rare_bins[key], key=marker_sort_key)]
        selected = nearest_monotonic_pairs(queries, reserve_bins[key])
        if selected is None:
            raise RuntimeError("Internal capacity contradiction")
        pairs.extend(selected)
    return pairs, {
        "eligible_rare": len(rare),
        "matched": len(pairs),
        "unmatched_rare_count": 0,
        "bins_without_capacity": 0,
        "maximum_bin_deficit": 0,
        **pair_diagnostics(pairs),
    }


def rare_for_threshold(markers: list[Marker], m28_contract: dict, ref_threshold: int) -> list[Marker]:
    selected = []
    for marker in markers:
        stats = {"mac": marker.mac, "maf": marker.maf}
        if rare_under_contract(stats, m28_contract) and marker.ref_minor_total >= ref_threshold:
            selected.append(marker)
    return selected


def nonrare_candidates(markers: list[Marker], minimum_maf: float, ref_total_haplotypes: int) -> list[Marker]:
    return [
        marker
        for marker in markers
        if marker.maf >= minimum_maf
        and marker.ref_minor_total > 0
        and marker.ref_minor_total < ref_total_haplotypes
    ]


def write_screen_table(path: Path, screens: list[dict]) -> None:
    columns = [
        "nonrare_screen",
        "minimum_maf",
        "nonrare_candidates",
        "b0_mapped",
        "b0_pass",
        "reserve_candidates",
        "ref_minor_threshold",
        "rare_eligible",
        "bin_width_cm",
        "matched",
        "unmatched_rare_count",
        "bins_without_capacity",
        "capacity_pass",
        "median_absolute_delta_cm",
        "p95_absolute_delta_cm",
        "maximum_absolute_delta_cm",
        "wasserstein_distance_cm",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in screens:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_marker_manifest(
    path: Path,
    arm_component: str,
    markers: list[Marker],
    *,
    include_carrier_individuals: bool = False,
) -> None:
    with deterministic_gzip_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        columns = [
            "arm_component", "site_id", "chrom", "position", "cm", "minor_code",
            "mac_freq", "an_freq", "maf_freq", "ref_minor_total", "ref_minor_AFR",
            "ref_minor_EUR", "ref_minor_ASIA",
        ]
        if include_carrier_individuals:
            columns.append("freq_minor_carrier_individuals")
        writer.writerow(columns)
        for marker in sorted(markers, key=lambda value: (value.bp, value.site_id)):
            row = [
                arm_component, marker.site_id, "chr22", marker.bp, f"{marker.cm:.12g}",
                marker.minor_code, marker.mac, marker.an, f"{marker.maf:.12g}",
                marker.ref_minor_total, marker.ref_minor_afr, marker.ref_minor_eur,
                marker.ref_minor_asia,
            ]
            if include_carrier_individuals:
                row.append(marker.freq_minor_carrier_individuals)
            writer.writerow(row)


def write_b0_mapping(path: Path, pairs: list[MarkerPair]) -> None:
    with deterministic_gzip_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "template_position", "template_cm", "site_id", "simulated_position",
            "simulated_cm", "absolute_delta_bp", "absolute_delta_cm",
        ))
        for pair in pairs:
            writer.writerow((
                pair.query_bp, f"{pair.query_cm:.12g}", pair.control.site_id,
                pair.control.bp, f"{pair.control.cm:.12g}",
                abs(pair.control.bp - pair.query_bp),
                f"{abs(pair.control.cm - pair.query_cm):.12g}",
            ))


def write_br_bs_pairs(path: Path, rare: list[Marker], pairs: list[MarkerPair]) -> None:
    rare_by_position = {(marker.bp, marker.cm): marker for marker in rare}
    with deterministic_gzip_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow((
            "rare_site_id", "rare_position", "rare_cm", "control_site_id",
            "control_position", "control_cm", "absolute_delta_bp", "absolute_delta_cm",
        ))
        for pair in sorted(pairs, key=lambda value: (value.query_bp, value.control.bp)):
            rare_marker = rare_by_position[(pair.query_bp, pair.query_cm)]
            writer.writerow((
                rare_marker.site_id, rare_marker.bp, f"{rare_marker.cm:.12g}",
                pair.control.site_id, pair.control.bp, f"{pair.control.cm:.12g}",
                abs(pair.control.bp - rare_marker.bp),
                f"{abs(pair.control.cm - rare_marker.cm):.12g}",
            ))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    contract = load_json(args.preregistration)
    m28_contract = load_m28_contract(args.m28_preregistration)
    authenticated = contract["authenticated_inputs"]
    hashes = {
        "tree_sequence": verify_hash(args.tree_sequence, authenticated["m28_tree_sequence_sha256"], "tree sequence"),
        "pool_manifest": verify_hash(args.pool_manifest, authenticated["m28_pool_manifest_sha256"], "pool manifest"),
        "genetic_map": verify_hash(args.genetic_map, authenticated["genetic_map_sha256"], "genetic map"),
        "m28_preregistration": verify_hash(args.m28_preregistration, authenticated["m28_preflight_contract_sha256"], "M28 preregistration"),
        "baseline_template": verify_hash(args.baseline_template, authenticated["gnomix_chr22_template"]["sha256"], "baseline template"),
        "m28b_preregistration": sha256(args.preregistration),
    }
    genetic_map = read_genetic_map(args.genetic_map, m28_contract)
    template = read_template(args.baseline_template, genetic_map, authenticated["gnomix_chr22_template"])
    pools = load_allowed_pools(args.pool_manifest, m28_contract["source_populations"]["labels"])

    import tskit

    ts = tskit.load(str(args.tree_sequence))
    markers, inventory = inventory_markers(ts, pools, genetic_map, m28_contract)
    rare_contract = contract["rare_selector"]
    rare_known_answer = (
        inventory["rare_sites"] == int(rare_contract["known_answer_rare_sites"])
        and inventory["rare_mac_histogram"] == rare_contract["known_answer_mac_histogram"]
    )

    screen_rows: list[dict] = []
    screen_objects: dict[str, dict] = {}
    primary_band_id = contract["primary_nonrare_screen"]
    primary_ref_threshold = int(rare_contract["primary_reference_minor_copy_threshold"])
    primary_width = None
    primary_b0: list[Marker] = []
    primary_b0_pairs: list[MarkerPair] = []
    primary_rare: list[Marker] = []
    primary_controls: list[Marker] = []
    primary_control_pairs: list[MarkerPair] = []

    for band in contract["nonrare_screens"]:
        band_id = band["id"]
        minimum_maf = float(band["minimum_maf_inclusive"])
        candidates = nonrare_candidates(markers, minimum_maf, inventory["ref_total_haplotypes"])
        b0_pairs = nearest_monotonic_pairs(template, candidates)
        b0_diagnostics = pair_diagnostics(b0_pairs or [])
        b0_ids = {pair.control.site_id for pair in b0_pairs or []}
        b0_pass = bool(b0_pairs) and (
            len(b0_ids) == int(contract["b0_mapping"]["target_count"])
            and b0_diagnostics["p95_absolute_delta_cm"] <= float(contract["b0_mapping"]["p95_absolute_delta_cm_max"])
            and b0_diagnostics["maximum_absolute_delta_cm"] <= float(contract["b0_mapping"]["maximum_absolute_delta_cm_max"])
        )
        reserve = [marker for marker in candidates if marker.site_id not in b0_ids]
        band_result = {
            "minimum_maf": minimum_maf,
            "nonrare_candidates": len(candidates),
            "b0_mapped": len(b0_pairs or []),
            "b0_unique": len(b0_ids),
            "b0_pass": b0_pass,
            "b0_mapping": b0_diagnostics,
            "reserve_candidates": len(reserve),
            "matching_screens": [],
        }
        for ref_threshold in rare_contract["reference_minor_copy_thresholds"]:
            rare = rare_for_threshold(markers, m28_contract, int(ref_threshold))
            for width in contract["bs_matching"]["bin_widths_cm"]:
                pairs, diagnostics = match_controls_by_bin(
                    rare, reserve, float(genetic_map.cm[0]), float(width)
                )
                capacity_pass = bool(b0_pass and pairs is not None and len(pairs) == len(rare))
                row = {
                    "nonrare_screen": band_id,
                    "minimum_maf": minimum_maf,
                    "nonrare_candidates": len(candidates),
                    "b0_mapped": len(b0_pairs or []),
                    "b0_pass": b0_pass,
                    "reserve_candidates": len(reserve),
                    "ref_minor_threshold": int(ref_threshold),
                    "rare_eligible": len(rare),
                    "bin_width_cm": float(width),
                    "capacity_pass": capacity_pass,
                    **diagnostics,
                }
                screen_rows.append(row)
                band_result["matching_screens"].append(row)
                if (
                    band_id == primary_band_id
                    and int(ref_threshold) == primary_ref_threshold
                    and primary_width is None
                    and capacity_pass
                ):
                    primary_width = float(width)
                    primary_b0 = [pair.control for pair in b0_pairs or []]
                    primary_b0_pairs = list(b0_pairs or [])
                    primary_rare = rare
                    primary_controls = [pair.control for pair in pairs or []]
                    primary_control_pairs = list(pairs or [])
        screen_objects[band_id] = band_result

    known_primary_ref = len(rare_for_threshold(markers, m28_contract, primary_ref_threshold))
    c2 = rare_known_answer and known_primary_ref == int(rare_contract["known_answer_ref_observed_at_primary_threshold"])
    b0_ids = {marker.site_id for marker in primary_b0}
    rare_ids = {marker.site_id for marker in primary_rare}
    control_ids = {marker.site_id for marker in primary_controls}
    c4 = bool(primary_width is not None) and (
        len(primary_rare) == len(primary_controls)
        and not (b0_ids & rare_ids)
        and not (b0_ids & control_ids)
        and not (rare_ids & control_ids)
        and len(b0_ids) == len(primary_b0)
        and len(rare_ids) == len(primary_rare)
        and len(control_ids) == len(primary_controls)
    )
    ancestry_support = {
        "AFR": sum(marker.ref_minor_afr > 0 for marker in primary_rare),
        "EUR": sum(marker.ref_minor_eur > 0 for marker in primary_rare),
        "ASIA": sum(marker.ref_minor_asia > 0 for marker in primary_rare),
    }
    gates = {
        "C0_INPUT_IDENTITY": True,
        "C1_ACCESS_BOUNDARY": True,
        "C2_RARE_DEFINITION": c2,
        "C3_B0_MAPPING": bool(primary_b0) and screen_objects[primary_band_id]["b0_pass"],
        "C4_ARM_PARITY": c4,
        "C5_ANCESTRY_SUPPORT": bool(primary_rare) and all(value > 0 for value in ancestry_support.values()),
        "C6_REPRODUCIBILITY": None,
        "C7_SCOPE": True,
    }
    if not c2:
        decision = "STOP_RARE_DEFINITION"
    elif not gates["C3_B0_MAPPING"]:
        decision = "STOP_BASELINE_MAPPING"
    elif primary_width is None:
        decision = "STOP_CAPACITY"
    elif not c4:
        decision = "STOP_PARITY"
    else:
        decision = "AUDIT_COMPLETE_PENDING_REPRODUCIBILITY"

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_screen_table(args.outdir / "m28b_capacity_screens.tsv", screen_rows)
    write_marker_manifest(args.outdir / "m28b_B0.tsv.gz", "B0", primary_b0)
    write_marker_manifest(args.outdir / "m28b_BR_additions.tsv.gz", "BR_addition", primary_rare)
    write_marker_manifest(args.outdir / "m28b_BS_additions.tsv.gz", "BS_addition", primary_controls)
    write_b0_mapping(args.outdir / "m28b_B0_mapping.tsv.gz", primary_b0_pairs)
    write_br_bs_pairs(
        args.outdir / "m28b_BR_BS_pairs.tsv.gz", primary_rare, primary_control_pairs
    )
    report = {
        "stage": contract["stage"],
        "scope": contract["scope"],
        "decision": decision,
        "hashes": hashes,
        "inventory": inventory,
        "screens": screen_objects,
        "primary_construction": {
            "nonrare_screen": primary_band_id,
            "ref_minor_threshold": primary_ref_threshold,
            "bin_width_cm": primary_width,
            "B0_markers": len(primary_b0),
            "BR_additions": len(primary_rare),
            "BS_additions": len(primary_controls),
            "ancestry_reference_support": ancestry_support,
        },
        "gates": gates,
        "interpretation": "Technical marker-capacity audit only. No LAI result or evidence of improvement is produced.",
    }
    write_json(args.outdir / "m28b_capacity.public.json", report)
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
