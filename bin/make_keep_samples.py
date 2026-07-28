#!/usr/bin/env python3

import argparse
import sys

import pandas as pd


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser()
    p.add_argument("--flags", required=True, help="flags_per_sample.tsv")
    p.add_argument("--out", required=True, help="Output keep_samples.txt")
    p.add_argument("--flag_fail_col", default="flag_fail")
    p.add_argument("--sample_id_col", default="IID")
    return p.parse_args()


def main():
    """Genera la lista de muestras que superan los filtros de calidad."""
    args = parse_args()
    df = pd.read_csv(args.flags, sep="\t", dtype=str)

    if args.sample_id_col not in df.columns and args.sample_id_col == "IID" and "#IID" in df.columns:
        sample_col = "#IID"
    else:
        sample_col = args.sample_id_col

    if sample_col not in df.columns:
        raise SystemExit(f"Missing sample_id_col '{sample_col}' in {args.flags}. Columns: {list(df.columns)}")

    if args.flag_fail_col not in df.columns:
        raise SystemExit(f"Missing flag_fail_col '{args.flag_fail_col}' in {args.flags}. Columns: {list(df.columns)}")

    fail = df[args.flag_fail_col].astype(str).str.lower().isin(["true", "1", "yes"])
    keep_df = df.loc[~fail, [sample_col]].copy()

    keep_df[sample_col].to_csv(args.out, index=False, header=False)


if __name__ == "__main__":
    main()
