#!/usr/bin/env python3
"""Deterministic comparison of historical ALT-coded and minor-oriented M14.

The comparison is paired: cohort, loci, coordinates and segmentation parameters
must already be identical.  It does not run community detection or inspect TEST.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from audit_rare_allele_orientation import interval_overlap_summary, window_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dir", required=True, type=Path)
    parser.add_argument("--minor-dir", required=True, type=Path)
    parser.add_argument("--chromosomes", default=",".join(map(str, range(1, 23))))
    parser.add_argument("--min-edge-bp", type=int, default=2_000_000)
    parser.add_argument("--min-max-segment-bp", type=int, default=500_000)
    parser.add_argument("--chr22-expected-variants", type=int, default=447_907)
    parser.add_argument("--chr22-expected-pairs", type=int, default=197)
    parser.add_argument("--chr22-expected-segments", type=int, default=203)
    parser.add_argument("--chr22-expected-pair-bp", type=int, default=265_460_384)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    return parser.parse_args()


def prefix(chrom: str) -> str:
    return f"dnabr.hg38.2723.chr{chrom}"


def load_summary(directory: Path, chrom: str) -> dict:
    path = directory / f"{prefix(chrom)}.sharing_scan.summary.json"
    if not path.exists():
        raise SystemExit(f"Missing summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_frame(directory: Path, chrom: str, suffix: str) -> pd.DataFrame:
    path = directory / f"{prefix(chrom)}.{suffix}.tsv.gz"
    if not path.exists():
        raise SystemExit(f"Missing M14 output: {path}")
    return pd.read_csv(path, sep="\t", compression="gzip")


def pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["sample_a"].astype(str), frame["sample_b"].astype(str)))


def jaccard(left: set, right: set) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def accumulate_pairs(frame: pd.DataFrame, accumulator: dict) -> None:
    for row in frame.itertuples(index=False):
        pair = (str(row.sample_a), str(row.sample_b))
        entry = accumulator[pair]
        entry["total_shared_bp"] += int(row.length_bp)
        entry["n_segments"] += 1
        entry["max_segment_bp"] = max(entry["max_segment_bp"], int(row.length_bp))


def filter_graph(pairs: dict, min_edge_bp: int, min_max_segment_bp: int) -> dict:
    return {
        pair: values
        for pair, values in pairs.items()
        if values["total_shared_bp"] >= min_edge_bp
        and values["max_segment_bp"] >= min_max_segment_bp
    }


def graph_structure(samples: list[str], edges: dict) -> dict:
    parent = {sample: sample for sample in samples}
    degree = {sample: 0 for sample in samples}
    strength = {sample: 0 for sample in samples}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for (left, right), values in edges.items():
        if left not in parent or right not in parent:
            raise SystemExit(f"Graph edge includes sample outside canonical cohort: {left}, {right}")
        degree[left] += 1
        degree[right] += 1
        strength[left] += int(values["total_shared_bp"])
        strength[right] += int(values["total_shared_bp"])
        union(left, right)

    components = defaultdict(list)
    for sample in samples:
        components[find(sample)].append(sample)
    sizes = sorted((len(nodes) for nodes in components.values()), reverse=True)
    return {
        "n_nodes": len(samples),
        "n_edges": len(edges),
        "n_isolated": sum(value == 0 for value in degree.values()),
        "n_components_including_isolates": len(components),
        "largest_component_size": sizes[0] if sizes else 0,
        "component_size_multiset": sizes,
        "degree": degree,
        "strength_bp": strength,
    }


def graph_comparison(samples: list[str], historical: dict, minor: dict) -> tuple[dict, pd.DataFrame]:
    historical_edges = set(historical)
    minor_edges = set(minor)
    hist_structure = graph_structure(samples, historical)
    minor_structure = graph_structure(samples, minor)
    individual_rows = []
    for sample in samples:
        individual_rows.append({
            "sample_id": sample,
            "historical_degree": hist_structure["degree"][sample],
            "minor_degree": minor_structure["degree"][sample],
            "delta_degree": minor_structure["degree"][sample] - hist_structure["degree"][sample],
            "historical_strength_bp": hist_structure["strength_bp"][sample],
            "minor_strength_bp": minor_structure["strength_bp"][sample],
            "delta_strength_bp": (
                minor_structure["strength_bp"][sample]
                - hist_structure["strength_bp"][sample]
            ),
        })
    for structure in (hist_structure, minor_structure):
        structure.pop("degree")
        structure.pop("strength_bp")
    comparison = {
        "historical": hist_structure,
        "minor": minor_structure,
        "edge_set_jaccard": jaccard(historical_edges, minor_edges),
        "edges_common": len(historical_edges & minor_edges),
        "edges_removed": len(historical_edges - minor_edges),
        "edges_added": len(minor_edges - historical_edges),
        "weights_identical_on_common_edges": all(
            historical[pair]["total_shared_bp"] == minor[pair]["total_shared_bp"]
            for pair in historical_edges & minor_edges
        ),
    }
    return comparison, pd.DataFrame(individual_rows)


def main() -> None:
    args = parse_args()
    chromosomes = [value.removeprefix("chr") for value in args.chromosomes.split(",") if value]
    if len(chromosomes) != len(set(chromosomes)):
        raise SystemExit("Chromosome list contains duplicates")
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_chromosome = []
    historical_pairs = defaultdict(lambda: {"total_shared_bp": 0, "n_segments": 0, "max_segment_bp": 0})
    minor_pairs = defaultdict(lambda: {"total_shared_bp": 0, "n_segments": 0, "max_segment_bp": 0})
    canonical_samples = None
    canonical_parameters = None

    for chrom in chromosomes:
        historical_summary = load_summary(args.historical_dir, chrom)
        minor_summary = load_summary(args.minor_dir, chrom)
        if minor_summary.get("carrier_allele_mode") != "minor_allele":
            raise SystemExit(f"chr{chrom} does not declare carrier_allele_mode=minor_allele")
        hist_samples = list(map(str, historical_summary.get("selected_samples", [])))
        minor_samples = list(map(str, minor_summary.get("selected_samples", [])))
        if hist_samples != minor_samples:
            raise SystemExit(f"Cohort identity/order differs on chr{chrom}")
        if canonical_samples is None:
            canonical_samples = hist_samples
            canonical_parameters = historical_summary.get("parameters_used", {})
        if hist_samples != canonical_samples:
            raise SystemExit(f"Historical cohort differs between chromosomes at chr{chrom}")
        if historical_summary.get("parameters_used", {}) != minor_summary.get("parameters_used", {}):
            raise SystemExit(f"Load-bearing parameters differ on chr{chrom}")
        if minor_summary.get("parameters_used", {}) != canonical_parameters:
            raise SystemExit(f"Parameters differ between chromosomes at chr{chrom}")

        historical_windows = load_frame(args.historical_dir, chrom, "sharing_windows")
        minor_windows = load_frame(args.minor_dir, chrom, "sharing_windows")
        historical_segments = load_frame(args.historical_dir, chrom, "pairwise_segments")
        minor_segments = load_frame(args.minor_dir, chrom, "pairwise_segments")
        historical_pair_set = pair_set(historical_segments)
        minor_pair_set = pair_set(minor_segments)
        overlap = interval_overlap_summary(historical_segments, minor_segments)
        windows = window_comparison(historical_windows, minor_windows)

        row = {
            "chrom": chrom,
            "historical_variants": int(historical_summary["n_shared_carrier_variants"]),
            "minor_variants": int(minor_summary["n_shared_carrier_variants"]),
            "historical_pairs": len(historical_pair_set),
            "minor_pairs": len(minor_pair_set),
            "historical_segments": int(len(historical_segments)),
            "minor_segments": int(len(minor_segments)),
            "historical_pair_bp": int(historical_segments["length_bp"].sum()),
            "minor_pair_bp": int(minor_segments["length_bp"].sum()),
            "pair_set_jaccard": jaccard(historical_pair_set, minor_pair_set),
            "pairs_removed": len(historical_pair_set - minor_pair_set),
            "pairs_added": len(minor_pair_set - historical_pair_set),
            **overlap,
            **windows,
        }
        per_chromosome.append(row)
        accumulate_pairs(historical_segments, historical_pairs)
        accumulate_pairs(minor_segments, minor_pairs)

    if canonical_samples is None:
        raise SystemExit("No chromosomes were compared")
    if len(canonical_samples) != 2619:
        raise SystemExit(f"Canonical sample count {len(canonical_samples)} != 2619")

    by_chromosome = pd.DataFrame(per_chromosome)
    by_chromosome.to_csv(
        args.outdir / "m14_orientation_by_chromosome.tsv", sep="\t", index=False
    )

    raw_comparison, raw_individuals = graph_comparison(
        canonical_samples, historical_pairs, minor_pairs
    )
    filtered_historical = filter_graph(
        historical_pairs, args.min_edge_bp, args.min_max_segment_bp
    )
    filtered_minor = filter_graph(minor_pairs, args.min_edge_bp, args.min_max_segment_bp)
    filtered_comparison, filtered_individuals = graph_comparison(
        canonical_samples, filtered_historical, filtered_minor
    )
    raw_individuals = raw_individuals.add_prefix("raw_").rename(
        columns={"raw_sample_id": "sample_id"}
    )
    filtered_individuals = filtered_individuals.add_prefix("filtered_").rename(
        columns={"filtered_sample_id": "sample_id"}
    )
    individual = raw_individuals.merge(filtered_individuals, on="sample_id", validate="one_to_one")
    individual.to_csv(
        args.outdir / "m14_orientation_individual_deltas.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )

    chr22 = next((row for row in per_chromosome if row["chrom"] == "22"), None)
    chr22_gate = None
    if chr22 is not None:
        expected = {
            "minor_variants": args.chr22_expected_variants,
            "minor_pairs": args.chr22_expected_pairs,
            "minor_segments": args.chr22_expected_segments,
            "minor_pair_bp": args.chr22_expected_pair_bp,
        }
        observed = {key: chr22[key] for key in expected}
        chr22_gate = {"expected": expected, "observed": observed, "pass": observed == expected}
        if not chr22_gate["pass"]:
            raise SystemExit(f"chr22 regression gate failed: {chr22_gate}")

    totals = {
        "historical_segments": int(by_chromosome["historical_segments"].sum()),
        "minor_segments": int(by_chromosome["minor_segments"].sum()),
        "historical_pair_bp": int(by_chromosome["historical_pair_bp"].sum()),
        "minor_pair_bp": int(by_chromosome["minor_pair_bp"].sum()),
        "historical_unique_pairs": len(historical_pairs),
        "minor_unique_pairs": len(minor_pairs),
        "windows_with_changed_pair_count": int(
            by_chromosome["windows_with_changed_pair_count"].sum()
        ),
    }
    report = {
        "status": "PASS",
        "scope": "paired_deterministic_m14_alt_vs_minor_no_clustering_no_test",
        "chromosomes": chromosomes,
        "n_samples": len(canonical_samples),
        "parameters": canonical_parameters,
        "totals": totals,
        "raw_m14_graph": raw_comparison,
        "m16_5_input_graph": {
            "filters": {
                "min_edge_bp": args.min_edge_bp,
                "min_max_segment_bp": args.min_max_segment_bp,
            },
            **filtered_comparison,
            "exactly_equivalent": (
                filtered_comparison["edges_removed"] == 0
                and filtered_comparison["edges_added"] == 0
                and filtered_comparison["weights_identical_on_common_edges"]
            ),
        },
        "chr22_regression_gate": chr22_gate,
        "interpretation_limits": [
            "M14 measures genotype-level co-carriage (IBS-like), not phased IBD or biological validation.",
            "The minor allele is defined inside the canonical 2619-sample cohort.",
            "No Leiden/NMF model was fitted and TEST was not used.",
        ],
    }
    (args.outdir / "m14_orientation_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
