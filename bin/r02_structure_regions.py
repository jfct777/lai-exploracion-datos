#!/usr/bin/env python3
"""Deterministic map-based targets and exclusion masks, without genotype access.

Geometry is a declared design choice, not an estimate of independent LD blocks.
The resulting manifest explicitly requires a separate FIT-only support audit.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "r02_structure_regions_v1"
BLOCK_COLUMNS = ["block_id", "chromosome", "start_bp", "end_bp",
                 "excluded_start_bp", "excluded_end_bp"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_points(points, chromosome, length_bp):
    require(str(chromosome) in {str(x) for x in range(1, 23)}, "Autosome required")
    require(type(length_bp) is int and length_bp > 0, "Positive chromosome length required")
    require(len(points) >= 2, "Map needs at least two points")
    previous_bp, previous_cm = 0, -math.inf
    for bp, cm in points:
        require(type(bp) is int and previous_bp < bp <= length_bp,
                "Map positions must increase inside chromosome bounds")
        require(math.isfinite(cm) and cm >= previous_cm, "Map must have finite nondecreasing cM")
        previous_bp, previous_cm = bp, cm
    require(points[-1][1] > points[0][1], "Map has no positive genetic span")


def read_map(path, chromosome, length_bp, format="shapeit"):
    require(format in {"shapeit", "shapeit3", "tsv_chrom_pos_cm"}, "Unsupported map format")
    opener = gzip.open if str(path).endswith(".gz") else open
    points = []
    with opener(path, "rt") as handle:
        header = handle.readline().split()
        known = (["pos", "chr", "cM"], ["Position(bp)", "Rate(cM/Mb)", "Map(cM)"],
                 ["position", "COMBINED_rate(cM/Mb)", "Genetic_Map(cM)"])
        require(header == ["chrom", "pos", "cm"] if format == "tsv_chrom_pos_cm" else header in known,
                "Map header differs from declared format")
        for line in handle:
            fields = line.split()
            require(len(fields) == 3, "Map requires three columns")
            if format == "tsv_chrom_pos_cm":
                c, bp, cm = fields
            elif header == ["pos", "chr", "cM"]:
                bp, c, cm = fields
            else:
                bp, _, cm = fields
                c = str(chromosome)  # Chromosome authenticated by source manifest.
            require(c in {str(chromosome), "chr" + str(chromosome)}, "Map chromosome mismatch")
            points.append((int(bp), float(cm)))
    validate_points(points, chromosome, length_bp)
    return points


def interpolate_cm(points, position):
    require(points[0][0] <= position <= points[-1][0], "No map extrapolation")
    i = bisect.bisect_left(points, (position, -math.inf))
    if points[i][0] == position:
        return points[i][1]
    a, b = points[i-1], points[i]
    return a[1] + (position-a[0]) * (b[1]-a[1]) / (b[0]-a[0])


def inverse_cm(points, genetic, side="left"):
    """Interpolate a cM boundary; on a plateau select its declared physical edge."""
    require(side in {"left", "right"}, "Unknown plateau edge")
    cms = [p[1] for p in points]
    require(cms[0] <= genetic <= cms[-1], "No inverse map extrapolation")
    i = bisect.bisect_left(cms, genetic)
    if cms[i] == genetic:
        j = i if side == "left" else bisect.bisect_right(cms, genetic)-1
        return float(points[j][0])
    a, b = points[i-1], points[i]
    return a[0] + (genetic-a[1]) * (b[0]-a[0]) / (b[1]-a[1])


def build_regions(map_points, chromosome, length_bp, parts=2, margin_cm=2.0):
    validate_points(map_points, chromosome, length_bp)
    require(type(parts) is int and 1 <= parts <= 100, "Invalid number of target parts")
    require(type(margin_cm) in (int, float) and math.isfinite(margin_cm) and margin_cm >= 0,
            "Margin must be nonnegative finite cM")
    first, last = map_points[0], map_points[-1]
    cuts = [first[0]-1]
    for k in range(1, parts):
        target_cm = first[1] + (last[1]-first[1])*k/parts
        cuts.append(math.floor(inverse_cm(map_points, target_cm, "left")))
    cuts.append(last[0])
    require(all(a < b for a, b in zip(cuts, cuts[1:])), "Empty targets after bp rounding")
    rows = []
    for k in range(parts):
        start, end = cuts[k]+1, cuts[k+1]
        lo_cm, hi_cm = interpolate_cm(map_points, start), interpolate_cm(map_points, end)
        # If the requested margin reaches a map boundary, mask the unmapped
        # chromosome tail too. This does not assign it an invented cM value.
        lo = 1 if lo_cm-margin_cm <= first[1] else math.floor(inverse_cm(map_points, lo_cm-margin_cm, "left"))
        hi = length_bp if hi_cm+margin_cm >= last[1] else math.ceil(inverse_cm(map_points, hi_cm+margin_cm, "right"))
        require(1 <= lo <= start <= end <= hi <= length_bp, "Invalid exclusion mask")
        rows.append(dict(block_id=f"chr{int(chromosome):02d}_part{k+1:02d}", chromosome=str(chromosome),
            start_bp=start, end_bp=end, excluded_start_bp=lo, excluded_end_bp=hi,
            start_cm=lo_cm, end_cm=hi_cm, geometry="EQUAL_MAP_DISTANCE_PARTS",
            margin_cm=margin_cm, target_map_supported=True,
            edge_mask_policy="EXCLUDE_UNMAPPED_TAIL_WHEN_MARGIN_REACHES_MAP_EDGE"))
    require(sum(r["end_bp"]-r["start_bp"]+1 for r in rows) == last[0]-first[0]+1,
            "Targets must partition the mapped physical domain without gaps")
    return rows


def write_table(path, rows, columns):
    with Path(path).open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(contract_path, output, maps_dir=None):
    contract = json.loads(Path(contract_path).read_text())
    require(contract.get("schema") == SCHEMA and contract.get("genome_build") == "GRCh38",
            "Explicit schema and GRCh38 contract required")
    maps = contract["maps"]
    require(maps and contract["chromosomes"], "Nonempty chromosome maps required")
    require(set(maps) == {str(c) for c in contract["chromosomes"]}, "Exact chromosome map set required")
    require(len(maps) == len(contract["chromosomes"]), "Duplicate chromosome")
    require(len({Path(d["path"]).name for d in maps.values()}) == len(maps), "Duplicate map basenames")
    require(contract["selection_basis"] == "MAP_ONLY_NO_OUTCOME_NO_RARE_DATA", "Invalid geometry selection basis")
    rows, audit = [], []
    for chrom in sorted(maps, key=int):
        desc = maps[chrom]
        path = Path(maps_dir)/Path(desc["path"]).name if maps_dir else Path(desc["path"])
        require(sha(path) == desc["sha256"], "Genetic map SHA256 mismatch")
        points = read_map(path, chrom, desc["length_bp"], desc["format"])
        rows.extend(build_regions(points, chrom, desc["length_bp"], contract["parts"], contract["margin_cm"]))
        audit.append(dict(chromosome=chrom, map_sha256=desc["sha256"], points=len(points),
            first_bp=points[0][0], last_bp=points[-1][0], first_cm=points[0][1], last_cm=points[-1][1],
            unmapped_bp=desc["length_bp"]-(points[-1][0]-points[0][0]+1)))
    out = Path(output)
    require(not out.exists(), "Output exists; use Nextflow cache, never overwrite")
    out.mkdir(parents=True, mode=0o700)
    write_table(out/"blocks.tsv", rows, BLOCK_COLUMNS)
    write_table(out/"geometry.tsv", rows, list(rows[0]))
    write_table(out/"map_support.tsv", audit, list(audit[0]))
    result = dict(schema=SCHEMA, stage="GEOMETRY_COMPLETE_FIT_SUPPORT_REQUIRED",
        contract_sha256=sha(contract_path), code_sha256=sha(__file__), n_blocks=len(rows),
        chromosomes=sorted(maps, key=int), parts=contract["parts"], margin_cm=contract["margin_cm"],
        blocks_sha256=sha(out/"blocks.tsv"), genotypes_read=False, rare_data_read=False,
        ld_independence_demonstrated=False, training_allowed=False, blocks=rows,
        sensitivity="WHOLE_CHROMOSOME_SAME_TARGETS_AND_PEOPLE",
        conclusion="Coordinate design only; FIT support and LD audit remain separate outputs")
    with (out/"manifest.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.contract, args.output, args.maps_dir), sort_keys=True))


if __name__ == "__main__":
    main()
