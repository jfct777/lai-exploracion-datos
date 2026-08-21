#!/usr/bin/env python3
"""Truth-free context-occupancy screen on ordered chr22 genetic coordinates."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Sequence


def validate_positions(values: Sequence[float], label: str) -> list[float]:
    positions = [float(value) for value in values]
    if not positions:
        raise ValueError(f"{label} must be a non-empty vector")
    if not all(math.isfinite(value) for value in positions):
        raise ValueError(f"{label} contains non-finite cM")
    if any(right < left for left, right in zip(positions, positions[1:])):
        raise ValueError(f"{label} must be non-decreasing")
    return positions


def context_counts(grid_cm: Sequence[float], rare_cm: Sequence[float], radius_cm: float) -> list[int]:
    grid = validate_positions(grid_cm, "FLARE grid")
    rare = validate_positions(rare_cm, "rare loci")
    if not math.isfinite(radius_cm) or radius_cm <= 0:
        raise ValueError("radius_cm must be positive and finite")
    return [bisect.bisect_right(rare, marker + radius_cm) - bisect.bisect_left(rare, marker - radius_cm) for marker in grid]


def _nearest_quantile(sorted_counts: Sequence[int], probability: float) -> int:
    index = int(math.floor(probability * (len(sorted_counts) - 1) + 0.5))
    return int(sorted_counts[index])


def occupancy_report(grid_cm: Sequence[float], rare_cm: Sequence[float], radii_cm: Sequence[float]) -> dict:
    grid = validate_positions(grid_cm, "FLARE grid")
    rare = validate_positions(rare_cm, "rare loci")
    reports = []
    quantiles = (("q0", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95), ("q100", 1.0))
    for radius_value in radii_cm:
        radius = float(radius_value)
        counts = context_counts(grid, rare, radius)
        left_counts = [bisect.bisect_left(rare, marker) - bisect.bisect_left(rare, marker - radius) for marker in grid]
        right_counts = [bisect.bisect_right(rare, marker + radius) - bisect.bisect_left(rare, marker) for marker in grid]
        ordered = sorted(counts)
        reports.append({
            "radius_cm": radius,
            "total_width_cm": 2 * radius,
            "marker_count": len(grid),
            "rare_locus_count": len(rare),
            "count_quantiles": {name: _nearest_quantile(ordered, probability) for name, probability in quantiles},
            "empty_fraction": sum(value == 0 for value in counts) / len(counts),
            "maximum_total_loci": max(counts),
            "maximum_loci_left": max(left_counts),
            "maximum_loci_right_including_ties": max(right_counts),
            "context_width_cm": 2 * radius,
        })
    return {
        "chromosome": "chr22",
        "definition": "symmetric_radius_cm_around_each_flare_marker",
        "selects_radius": False,
        "grid_marker_count": len(grid),
        "rare_locus_count": len(rare),
        "contexts": reports,
    }


def read_positions(path: Path) -> list[float]:
    values: list[float] = []
    base_pairs: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"chrom", "bp", "cm"}.issubset(reader.fieldnames):
            raise ValueError("position table requires chrom, bp and cm columns")
        for row in reader:
            if row["chrom"] != "chr22":
                raise ValueError("position table is not restricted to chr22")
            base_pairs.append(int(row["bp"]))
            values.append(float(row["cm"]))
    if any(right <= left for left, right in zip(base_pairs, base_pairs[1:])):
        raise ValueError("position table bp must be unique and strictly increasing")
    return validate_positions(values, path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flare-grid", type=Path, required=True)
    parser.add_argument("--rare-loci", type=Path, required=True)
    parser.add_argument("--radii-cm", nargs="+", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = occupancy_report(read_positions(args.flare_grid), read_positions(args.rare_loci), args.radii_cm)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
