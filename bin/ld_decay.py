#!/usr/bin/env python3

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="PLINK2 pairwise LD file (.ld/.r2/.ld.gz)")
    p.add_argument("--chr", required=True)
    p.add_argument("--bin_size_bp", type=int, default=1000)
    p.add_argument("--max_dist_bp", type=int, default=1000000)
    p.add_argument("--out_tsv", required=True)
    p.add_argument("--out_summary_json", required=True)
    return p.parse_args()


def detect_columns(header_cols):
    """Detect which columns to use for distance and r2 from header."""
    pos1_col = pos2_col = r2_col = None
    for col in header_cols:
        col_upper = col.upper().lstrip("#")
        if col_upper in ("BP_A", "POS_A", "POS1", "BP"):
            pos1_col = col
        elif col_upper in ("BP_B", "POS_B", "POS2", "BP2"):
            pos2_col = col
        elif col_upper in ("R2", "R^2", "UNPHASED_R2"):
            r2_col = col
    return pos1_col, pos2_col, r2_col


def main():
    args = parse_args()

    in_path = Path(args.pairs)
    if not in_path.exists():
        raise SystemExit(f"Missing LD pairs file: {in_path}")

    bin_size = args.bin_size_bp
    max_dist = args.max_dist_bp
    n_bins = (max_dist // bin_size) + 1

    # Accumulators for streaming aggregation (sum, sum_sq, count per bin)
    bin_sum = [0.0] * n_bins
    bin_count = [0] * n_bins
    bin_values = [[] for _ in range(n_bins)]  # for median (store values)

    total_pairs = 0
    pos1_col = pos2_col = r2_col = None
    pos1_idx = pos2_idx = r2_idx = None

    # Stream through file in chunks
    open_fn = gzip.open if str(in_path).endswith(".gz") else open
    with open_fn(in_path, "rt") as fh:
        header_line = fh.readline().strip()
        # Handle space or tab delimited (PLINK2 .vcor uses spaces)
        if "\t" in header_line:
            sep = "\t"
        else:
            sep = None  # split on whitespace
        header_cols = header_line.split(sep)
        
        pos1_col, pos2_col, r2_col = detect_columns(header_cols)
        if not all([pos1_col, pos2_col, r2_col]):
            raise SystemExit(f"Unsupported LD file columns: {header_cols}")
        
        pos1_idx = header_cols.index(pos1_col)
        pos2_idx = header_cols.index(pos2_col)
        r2_idx = header_cols.index(r2_col)

        for line in fh:
            parts = line.strip().split(sep)
            if len(parts) <= max(pos1_idx, pos2_idx, r2_idx):
                continue
            try:
                pos1 = int(parts[pos1_idx])
                pos2 = int(parts[pos2_idx])
                r2 = float(parts[r2_idx])
            except (ValueError, IndexError):
                continue

            dist = abs(pos2 - pos1)
            if dist < 0 or dist > max_dist:
                continue

            bin_idx = dist // bin_size
            if bin_idx >= n_bins:
                continue

            bin_sum[bin_idx] += r2
            bin_count[bin_idx] += 1
            bin_values[bin_idx].append(r2)
            total_pairs += 1

    if total_pairs == 0:
        raise SystemExit("No LD pairs within max_dist_bp")

    # Build output dataframe
    rows = []
    for i in range(n_bins):
        if bin_count[i] == 0:
            continue
        bin_start = i * bin_size
        mean_r2 = bin_sum[i] / bin_count[i]
        vals = sorted(bin_values[i])
        n = len(vals)
        median_r2 = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        rows.append({
            "chr": args.chr,
            "bin": bin_start,
            "n_pairs": bin_count[i],
            "mean_r2": mean_r2,
            "median_r2": median_r2,
        })

    agg = pd.DataFrame(rows)
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_tsv, sep="\t", index=False)

    summary = {
        "chr": args.chr,
        "n_pairs": total_pairs,
        "bin_size_bp": bin_size,
        "max_dist_bp": max_dist,
    }
    out_json = Path(args.out_summary_json)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
