#!/usr/bin/env python3
"""Materialize truth-free chr22 coordinate tables from frozen M30/M31 inputs."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Sequence

from m32_locus_contract import sha256_file
from m32_locus_smoke import write_json_atomic


class GeneticMap:
    def __init__(self, positions: Sequence[int], cms: Sequence[float]):
        self.positions = list(positions)
        self.cms = list(cms)
        if len(self.positions) < 2 or len(self.positions) != len(self.cms):
            raise ValueError("genetic map requires at least two paired points")
        if any(right <= left for left, right in zip(self.positions, self.positions[1:])):
            raise ValueError("genetic-map bp are not strictly increasing")
        if any(not math.isfinite(value) for value in self.cms) or any(right < left for left, right in zip(self.cms, self.cms[1:])):
            raise ValueError("genetic-map cM are nonfinite or decreasing")

    def cm_at(self, position: int) -> float:
        if position < self.positions[0] or position > self.positions[-1]:
            raise ValueError(f"position {position} lies outside the genetic map")
        index = bisect.bisect_right(self.positions, position) - 1
        if index == len(self.positions) - 1 or position == self.positions[index]:
            return self.cms[index]
        x0, x1 = self.positions[index], self.positions[index + 1]
        y0, y1 = self.cms[index], self.cms[index + 1]
        return y0 + (position - x0) / (x1 - x0) * (y1 - y0)


def load_map(path: Path) -> GeneticMap:
    positions, cms = [], []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3 or fields[0].removeprefix("chr") != "22":
                raise ValueError(f"map:{line_number}: invalid chr22 row")
            positions.append(int(fields[1]))
            cms.append(float(fields[2]))
    return GeneticMap(positions, cms)


def load_contract(path: Path, root_label: str, root_seed: int) -> tuple[dict, dict]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("stage") != "M32_REAL_CHR22_OCCUPANCY_SCREEN" or contract.get("version") != 1:
        raise ValueError("unsupported real-occupancy contract")
    if contract.get("status") != "TRUTH_FREE_COORDINATE_SCREEN_ONLY_NOT_SCIENTIFIC_EVIDENCE":
        raise ValueError("real-occupancy contract is not truth-free smoke-only")
    if contract.get("select_radius") is not False or contract.get("scientific_run_authorized") is not False:
        raise ValueError("contract would authorize scientific selection")
    root = contract.get("roots", {}).get(root_label)
    if root is None or root.get("root_seed") != root_seed or root.get("role") != "consumed_technical_known_answer_only":
        raise ValueError("root is not an authenticated consumed technical root")
    return contract, root


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} sha256 mismatch: {observed} != {expected}")
    return observed


def validate_rows(rows: list[tuple[str, int, float, str]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    if any(chrom != "chr22" for chrom, _, _, _ in rows):
        raise ValueError(f"{label} chromosome mismatch")
    bp = [row[1] for row in rows]
    cm = [row[2] for row in rows]
    ids = [row[3] for row in rows]
    if any(right <= left for left, right in zip(bp, bp[1:])):
        raise ValueError(f"{label} bp are duplicated or not strictly increasing")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} identifiers are duplicated")
    if any(not math.isfinite(value) for value in cm) or any(right < left for left, right in zip(cm, cm[1:])):
        raise ValueError(f"{label} cM are nonfinite or decreasing")


def load_rare_sites(path: Path, root_seed: int, genetic_map: GeneticMap) -> list[tuple[str, int, float, str]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"root_seed", "locus_index", "chrom", "position", "minor_code", "mac", "an", "maf"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("rare-sites header is invalid")
        for expected_index, row in enumerate(reader):
            if int(row["root_seed"]) != root_seed or int(row["locus_index"]) != expected_index:
                raise ValueError("rare-sites root or locus order drifted")
            if row["chrom"].removeprefix("chr") != "22" or int(row["minor_code"]) not in (0, 1):
                raise ValueError("rare-sites chromosome or minor orientation is invalid")
            position = int(row["position"])
            rows.append(("chr22", position, genetic_map.cm_at(position), f"rare-{expected_index}"))
    validate_rows(rows, "rare sites")
    return rows


def load_flare_grid(path: Path, genetic_map: GeneticMap) -> list[tuple[str, int, float, str]]:
    rows = []
    header_seen = False
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header_seen = True
                continue
            fields = line.rstrip("\n").split("\t", 5)
            if len(fields) < 5 or fields[0].removeprefix("chr") != "22":
                raise ValueError(f"FLARE VCF:{line_number}: invalid coordinate row")
            position = int(fields[1])
            marker_id = fields[2] if fields[2] != "." else f"chr22:{position}:{fields[3]}:{fields[4]}"
            rows.append(("chr22", position, genetic_map.cm_at(position), marker_id))
    if not header_seen:
        raise ValueError("FLARE VCF lacks a header")
    validate_rows(rows, "FLARE grid")
    return rows


def write_coordinates(path: Path, rows: Sequence[tuple[str, int, float, str]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("chrom", "bp", "cm", "locus_id"))
        for chrom, bp, cm, locus_id in rows:
            writer.writerow((chrom, bp, format(cm, ".17g"), locus_id))


def coordinate_summary(path: Path, rows: Sequence[tuple[str, int, float, str]]) -> dict:
    ties = sum(right[2] == left[2] for left, right in zip(rows, rows[1:]))
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "count": len(rows),
        "first": {"bp": rows[0][1], "cm": rows[0][2], "locus_id": rows[0][3]},
        "last": {"bp": rows[-1][1], "cm": rows[-1][2], "locus_id": rows[-1][3]},
        "adjacent_cm_ties": ties,
        "bp_strictly_increasing": True,
        "cm_non_decreasing": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--root-label", required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--rare-sites", type=Path, required=True)
    parser.add_argument("--flare-vcf", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract, root = load_contract(args.preregistration, args.root_label, args.root_seed)
    input_hashes = {
        "genetic_map": require_hash(args.genetic_map, contract["genetic_map"]["sha256"], "genetic map"),
        "rare_sites": require_hash(args.rare_sites, root["rare_sites_sha256"], "rare sites"),
        "flare_grid": require_hash(args.flare_vcf, root["flare_grid_sha256"], "FLARE grid"),
    }
    args.outdir.mkdir(parents=True, exist_ok=False)
    genetic_map = load_map(args.genetic_map)
    rare = load_rare_sites(args.rare_sites, args.root_seed, genetic_map)
    grid = load_flare_grid(args.flare_vcf, genetic_map)
    rare_path = args.outdir / "rare_loci.coordinates.tsv"
    grid_path = args.outdir / "flare_grid.coordinates.tsv"
    write_coordinates(rare_path, rare)
    write_coordinates(grid_path, grid)
    write_json_atomic(args.outdir / "coordinate_materialization.json", {
        "stage": contract["stage"],
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "input_sha256": input_hashes,
        "map_interpolation": "piecewise_linear_identical_to_M28D_GeneticMap.cm_at",
        "rare_loci": coordinate_summary(rare_path, rare),
        "flare_grid": coordinate_summary(grid_path, grid),
        "truth_accessed": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
