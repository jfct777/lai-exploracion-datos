#!/usr/bin/env python3
"""Benchmark indexed gVCF access without retaining genomic records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import resource
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvcfs", nargs="+", required=True, type=Path)
    parser.add_argument("--gnomix-reference-vcf", required=True, type=Path)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--reader-grid", default="1,4,8")
    parser.add_argument("--full-sample-count", type=int, default=128)
    parser.add_argument("--within-best-fraction", type=float, default=0.20)
    parser.add_argument("--compute-upper-usd-per-hour", type=float, default=0.40)
    parser.add_argument("--non_compute_overhead_upper_usd", type=float, default=0.80)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def write_regions(reference_vcf: Path, out: Path) -> int:
    positions: list[tuple[str, int]] = []
    with open_text(reference_vcf) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 3)
            positions.append((fields[0], int(fields[1])))
    with out.open("w", encoding="utf-8") as handle:
        for chrom, position in positions:
            label = chrom if chrom.startswith("chr") else f"chr{chrom}"
            handle.write(f"{label}\t{position - 1}\t{position}\n")
    return len(positions)


def read_one(bcftools: str, regions: Path, gvcf: Path) -> tuple[int, int]:
    process = subprocess.Popen(
        [bcftools, "view", "-R", str(regions), "-H", str(gvcf)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    n_lines = n_bytes = 0
    for line in process.stdout:
        n_lines += 1
        n_bytes += len(line)
    stderr = process.stderr.read() if process.stderr is not None else b""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"bcftools failed: {stderr[-2000:].decode('utf-8', errors='replace')}")
    return n_lines, n_bytes


def run_grid(args: argparse.Namespace, regions: Path, readers: int) -> dict[str, object]:
    started = time.monotonic()
    results: list[tuple[int, int] | None] = [None] * len(args.gvcfs)
    with ThreadPoolExecutor(max_workers=readers) as pool:
        futures = {pool.submit(read_one, args.bcftools, regions, path): index for index, path in enumerate(args.gvcfs)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    elapsed = time.monotonic() - started
    observed = [result for result in results if result is not None]
    return {
        "readers": readers,
        "n_smoke_samples": len(observed),
        "elapsed_seconds": elapsed,
        "records_total": sum(result[0] for result in observed),
        "uncompressed_bytes_total": sum(result[1] for result in observed),
        "records_per_sample_min": min(result[0] for result in observed),
        "records_per_sample_max": max(result[0] for result in observed),
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    reader_grid = sorted({int(value) for value in args.reader_grid.split(",")})
    if not reader_grid or reader_grid[0] < 1:
        raise SystemExit("Invalid reader grid")
    regions = args.outdir / "m27c_smoke_model_positions.bed"
    n_markers = write_regions(args.gnomix_reference_vcf, regions)
    rows = [run_grid(args, regions, readers) for readers in reader_grid]
    signatures = {(row["records_total"], row["uncompressed_bytes_total"]) for row in rows}
    if len(signatures) != 1:
        raise SystemExit("Reader configurations did not return identical data")
    best_elapsed = min(float(row["elapsed_seconds"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["elapsed_seconds"]) <= best_elapsed * (1.0 + args.within_best_fraction)
    ]
    selected = min(eligible, key=lambda row: int(row["readers"]))
    projected_seconds = (
        float(selected["elapsed_seconds"]) * args.full_sample_count / len(args.gvcfs)
    )
    projected_cost = (
        projected_seconds / 3600 * args.compute_upper_usd_per_hour
        + args.non_compute_overhead_upper_usd
    )
    summary = {
        "stage": "M27C_TARGETED_GVCF_RESOURCE_SMOKE",
        "n_model_markers": n_markers,
        "n_smoke_samples": len(args.gvcfs),
        "reader_grid": reader_grid,
        "selection_rule": f"smallest reader count within {args.within_best_fraction:.0%} of best elapsed time",
        "selected_readers": int(selected["readers"]),
        "projected_full_elapsed_seconds_conservative": projected_seconds,
        "projected_full_compute_and_fixed_overhead_usd_upper": projected_cost,
        "non_compute_overhead_upper_usd": args.non_compute_overhead_upper_usd,
        "non_compute_overhead_note": "Conservative allowance for Coldline retrieval, Class B reads, cross-region image and small US inputs; the gVCFs remain in-region.",
        "scientific_thresholds_tuned": False,
        "full_run_allowed_by_preregistered_cap": projected_seconds <= 3 * 3600 and projected_cost <= 2.0,
    }
    with (args.outdir / "m27c_resource_screen.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (args.outdir / "m27c_resource_screen.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
