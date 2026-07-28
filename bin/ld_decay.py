#!/usr/bin/env python3

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="PLINK2 pairwise LD file (.ld/.r2/.ld.gz)")
    p.add_argument("--chr", required=True)
    p.add_argument("--bin_size_bp", type=int, default=1000)
    p.add_argument("--max_dist_bp", type=int, default=1000000)
    p.add_argument("--out_tsv", required=True)
    p.add_argument("--out_summary_json", required=True)
    return p.parse_args()


def main():
    """Resume el decaimiento de LD por distancia."""
    args = parse_args()

    in_path = Path(args.pairs)
    if not in_path.exists():
        raise SystemExit(f"Missing LD pairs file: {in_path}")

    # Acepta las variantes de formato producidas por PLINK 1 y PLINK 2.
    if str(in_path).endswith(".gz"):
        df = pd.read_csv(in_path, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(in_path, sep="\t")

    # Identifica las columnas habituales de posición y r².
    col_dist = None
    col_r2 = None

    if "BP_A" in df.columns and "BP_B" in df.columns:
        col_dist = (df["BP_B"] - df["BP_A"]).abs()
    elif "BP" in df.columns and "BP2" in df.columns:
        col_dist = (df["BP2"] - df["BP"]).abs()
    elif "POS1" in df.columns and "POS2" in df.columns:
        col_dist = (df["POS2"] - df["POS1"]).abs()
    elif "POS_A" in df.columns and "POS_B" in df.columns:
        col_dist = (df["POS_B"] - df["POS_A"]).abs()

    if "R2" in df.columns:
        col_r2 = df["R2"]
    elif "R^2" in df.columns:
        col_r2 = df["R^2"]
    elif "UNPHASED_R2" in df.columns:
        col_r2 = df["UNPHASED_R2"]

    if col_dist is None or col_r2 is None:
        raise SystemExit(f"Unsupported LD file columns: {list(df.columns)}")

    df2 = pd.DataFrame({"dist_bp": col_dist.astype(int), "r2": col_r2.astype(float)})
    df2 = df2[(df2["dist_bp"] >= 0) & (df2["dist_bp"] <= args.max_dist_bp)].copy()

    if df2.empty:
        raise SystemExit("No LD pairs within max_dist_bp")

    df2["bin"] = (df2["dist_bp"] // args.bin_size_bp) * args.bin_size_bp

    agg = df2.groupby("bin", as_index=False).agg(n_pairs=("r2", "size"), mean_r2=("r2", "mean"), median_r2=("r2", "median"))
    agg.insert(0, "chr", args.chr)

    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_tsv, sep="\t", index=False)

    summary = {
        "chr": args.chr,
        "n_pairs": int(df2.shape[0]),
        "bin_size_bp": args.bin_size_bp,
        "max_dist_bp": args.max_dist_bp,
    }
    out_json = Path(args.out_summary_json)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
