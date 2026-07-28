#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import List

import pandas as pd


def _chr_sort_key(c: str) -> tuple:
    """Sort chromosomes naturally: 1-22, then X, Y."""
    c = c.replace("chr", "")
    if c.isdigit():
        return (0, int(c))
    elif c == "X":
        return (1, 0)
    elif c == "Y":
        return (1, 1)
    else:
        return (2, c)


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["per_chr", "aggregate"])

    p.add_argument("--chr")
    p.add_argument("--imiss")
    p.add_argument("--het")
    p.add_argument("--out")

    p.add_argument("--qc", action="append", default=[])
    p.add_argument("--stats", action="append", default=[])
    p.add_argument("--counts", action="append", default=[])

    p.add_argument("--outdir", default=".")
    p.add_argument("--sample_missing_fail", type=float, required=False, default=0.05)
    p.add_argument("--het_sd_fail", type=float, required=False, default=3.0)

    return p.parse_args()


def read_plink_imiss(path: str) -> pd.DataFrame:
    """Carga la tabla de missingness de PLINK y normaliza sus columnas."""
    df = pd.read_csv(path, delim_whitespace=True)
    cols = set(df.columns.astype(str).tolist())
    if {"FID", "IID", "MISS_PHENO", "N_MISS", "N_GENO", "F_MISS"}.issubset(cols):
        out = df[["FID", "IID", "MISS_PHENO", "N_MISS", "N_GENO", "F_MISS"]].copy()
        out["IID"] = out["IID"].astype(str)
        return out

    if {"#IID", "MISSING_CT", "OBS_CT", "F_MISS"}.issubset(cols):
        out = df[["#IID", "MISSING_CT", "OBS_CT", "F_MISS"]].copy()
        out = out.rename(columns={"#IID": "IID", "MISSING_CT": "N_MISS", "OBS_CT": "N_GENO"})
        out["IID"] = out["IID"].astype(str)
        out.insert(0, "FID", out["IID"])
        out.insert(2, "MISS_PHENO", 0)
        return out[["FID", "IID", "MISS_PHENO", "N_MISS", "N_GENO", "F_MISS"]]

    raise KeyError(f"Unexpected columns in missingness file {path}: {sorted(cols)}")


def read_plink_het(path: str) -> pd.DataFrame:
    """Carga la tabla de heterocigosidad de PLINK y calcula las métricas usadas."""
    df = pd.read_csv(path, delim_whitespace=True)
    cols = set(df.columns.astype(str).tolist())
    if {"FID", "IID", "O(HOM)", "E(HOM)", "N(NM)", "F"}.issubset(cols):
        out = df[["FID", "IID", "O(HOM)", "E(HOM)", "N(NM)", "F"]].copy()
        out["IID"] = out["IID"].astype(str)
        return out

    if {"#IID", "O(HOM)", "E(HOM)", "OBS_CT", "F"}.issubset(cols):
        out = df[["#IID", "O(HOM)", "E(HOM)", "OBS_CT", "F"]].copy()
        out = out.rename(columns={"#IID": "IID", "OBS_CT": "N(NM)"})
        out["IID"] = out["IID"].astype(str)
        out.insert(0, "FID", out["IID"])
        return out[["FID", "IID", "O(HOM)", "E(HOM)", "N(NM)", "F"]]

    raise KeyError(f"Unexpected columns in het file {path}: {sorted(cols)}")


def per_chr(args):
    """Construye el resumen de control de calidad para un cromosoma."""
    if not args.chr or not args.imiss or not args.het or not args.out:
        raise SystemExit("per_chr requires --chr --imiss --het --out")

    imiss = read_plink_imiss(args.imiss)
    het = read_plink_het(args.het)

    merged = imiss.merge(het, on=["FID", "IID"], how="inner")

    out = pd.DataFrame(
        {
            "chr": args.chr,
            "IID": merged["IID"],
            "F_MISS": merged["F_MISS"],
            "N_MISS": merged["N_MISS"],
            "N_GENO": merged["N_GENO"],
            "O_HOM": merged["O(HOM)"],
            "E_HOM": merged["E(HOM)"],
            "F": merged["F"],
        }
    )

    out.to_csv(args.out, sep="\t", index=False)


def _read_many_tsv(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        dfs.append(pd.read_csv(p, sep="\t"))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def aggregate(args):
    """Combina resúmenes cromosómicos y genera el informe de cohorte."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    qc_df = _read_many_tsv(args.qc)
    stats_df = _read_many_tsv(args.stats)
    counts_df = _read_many_tsv(args.counts)

    chrs = sorted(set(qc_df["chr"].astype(str).tolist()), key=_chr_sort_key) if not qc_df.empty else []

    if qc_df.empty:
        merged_per_sample = pd.DataFrame(columns=["IID"]) 
    else:
        agg = qc_df.groupby("IID", as_index=False).agg(
            {
                "F_MISS": "mean",
                "N_MISS": "sum",
                "N_GENO": "sum",
                "O_HOM": "sum",
                "E_HOM": "sum",
                "F": "mean",
            }
        )
        merged_per_sample = agg

    if merged_per_sample.empty:
        f_mean = None
        f_sd = None
    else:
        f_mean = float(merged_per_sample["F"].mean())
        f_sd = float(merged_per_sample["F"].std(ddof=0))

    flags = merged_per_sample.copy()
    if not flags.empty:
        flags["flag_missing"] = flags["F_MISS"] > float(args.sample_missing_fail)
        if f_mean is None or f_sd is None:
            flags["flag_het"] = False
        else:
            lo = f_mean - float(args.het_sd_fail) * f_sd
            hi = f_mean + float(args.het_sd_fail) * f_sd
            flags["flag_het"] = (flags["F"] < lo) | (flags["F"] > hi)
        flags["flag_fail"] = flags["flag_missing"] | flags["flag_het"]

    merged_path = outdir / "merged_per_sample.tsv"
    flags_path = outdir / "flags_per_sample.tsv"
    summary_path = outdir / "summary.json"
    report_path = outdir / "report.html"

    merged_per_sample.to_csv(merged_path, sep="\t", index=False)
    flags.to_csv(flags_path, sep="\t", index=False)

    summary = {
        "qc_partial": True,
        "chromosomes_processed": chrs,
        "n_chromosomes_processed": len(chrs),
        "n_samples": int(merged_per_sample.shape[0]) if not merged_per_sample.empty else 0,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    html = [
        "<html><head><meta charset='utf-8'><title>DNABR QC Report</title></head><body>",
        "<h1>DNABR QC Report</h1>",
        "<p><b>QC parcial:</b> cromosomas procesados = " + ",".join(chrs) + "</p>",
        "<h2>Resumen</h2>",
        "<pre>" + json.dumps(summary, indent=2) + "</pre>",
        "</body></html>",
    ]
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))


def main():
    """Ejecuta el modo por cromosoma o el modo agregado."""
    args = parse_args()

    if args.mode == "per_chr":
        per_chr(args)
    elif args.mode == "aggregate":
        aggregate(args)
    else:
        raise SystemExit("Unknown mode")


if __name__ == "__main__":
    main()
