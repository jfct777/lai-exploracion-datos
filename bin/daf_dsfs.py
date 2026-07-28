#!/usr/bin/env python3

import argparse
import gzip
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args():
    """Define y devuelve los argumentos de línea de comandos."""
    p = argparse.ArgumentParser()
    p.add_argument("--vcf", required=True)
    p.add_argument("--chr", required=True)
    p.add_argument("--ancestral_tsv_gz", required=True)
    p.add_argument("--out_prefix", required=True)

    p.add_argument("--rare_tail_max_ac", type=int, default=20)
    p.add_argument("--sfs_bins_af", default="0,0.001,0.005,0.01,0.05,0.1,0.2,0.5,1.0")

    return p.parse_args()


def _run(cmd):
    r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.stdout


def main():
    """Calcula DAF y el espectro derivado a partir de la tabla ancestral."""
    args = parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    bins = [float(x) for x in args.sfs_bins_af.split(",")]
    if bins[0] != 0.0 or bins[-1] != 1.0:
        raise SystemExit("sfs_bins_af must start with 0 and end with 1")

    # Carga la tabla ancestral con las columnas CHROM, POS, REF, ALT, ANCESTRAL y STATUS.
    anc = pd.read_csv(args.ancestral_tsv_gz, sep="\t", dtype=str)
    required = {"CHROM", "POS", "REF", "ALT", "ANCESTRAL", "STATUS"}
    if not required.issubset(set(anc.columns)):
        raise SystemExit(f"ancestral_tsv missing columns. Found: {list(anc.columns)}")

    anc = anc[anc["STATUS"].isin(["ok_ref", "ok_alt"])].copy()
    if anc.empty:
        raise SystemExit("No polarizable sites (ok_ref/ok_alt) in ancestral table")

    anc["POS"] = anc["POS"].astype(int)

    # Consulta AC y AN directamente en el VCF.
    q = _run(["bcftools", "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\t%INFO/AC\t%INFO/AN\n", args.vcf])

    rows = []
    for line in q.splitlines():
        if not line:
            continue
        chrom, pos_s, ref, alt, ac_s, an_s = line.split("\t")
        if "," in alt:
            continue
        if len(ref) != 1 or len(alt) != 1:
            continue
        try:
            pos = int(pos_s)
            ac = int(ac_s.split(",")[0])
            an = int(an_s)
        except Exception:
            continue
        if an <= 0:
            continue
        rows.append((chrom, pos, ref, alt, ac, an))

    df = pd.DataFrame(rows, columns=["CHROM", "POS", "REF", "ALT", "AC", "AN"])
    if df.empty:
        raise SystemExit("No AC/AN records found in VCF")

    merged = df.merge(anc, on=["CHROM", "POS", "REF", "ALT"], how="inner")
    if merged.empty:
        raise SystemExit("No overlap between VCF and ancestral TSV")

    # Calcula el conteo y la frecuencia del alelo derivado.
    def _dac(row):
        if row["ANCESTRAL"] == row["REF"]:
            return int(row["AC"])
        if row["ANCESTRAL"] == row["ALT"]:
            return int(row["AN"]) - int(row["AC"])
        return None

    merged["DAC"] = merged.apply(_dac, axis=1)
    merged = merged.dropna(subset=["DAC"]).copy()
    merged["DAC"] = merged["DAC"].astype(int)
    merged["DAF"] = merged["DAC"] / merged["AN"].astype(float)

    # Agrupa el espectro derivado por intervalos de DAF.
    hist = [0 for _ in range(len(bins) - 1)]
    rare_tail = Counter()

    for dac, an, daf in zip(merged["DAC"].tolist(), merged["AN"].tolist(), merged["DAF"].tolist()):
        if 1 <= dac <= args.rare_tail_max_ac:
            rare_tail[int(dac)] += 1
        for i in range(len(bins) - 1):
            lo = bins[i]
            hi = bins[i + 1]
            if (daf >= lo and daf < hi) or (i == len(bins) - 2 and math.isclose(daf, 1.0)):
                hist[i] += 1
                break

    dsfs_rows = []
    for i in range(len(bins) - 1):
        dsfs_rows.append({"chr": args.chr, "daf_bin_lo": bins[i], "daf_bin_hi": bins[i + 1], "n_sites": hist[i]})

    rare_rows = []
    for ac in range(1, args.rare_tail_max_ac + 1):
        rare_rows.append({"chr": args.chr, "DAC": ac, "n_sites": int(rare_tail.get(ac, 0))})

    dsfs_path = str(out_prefix) + ".dsfs.tsv"
    rare_path = str(out_prefix) + ".dsfs_rare_tail_dac.tsv"
    per_site_path = str(out_prefix) + ".daf_per_site.tsv.gz"
    summary_path = str(out_prefix) + ".summary.json"

    pd.DataFrame(dsfs_rows).to_csv(dsfs_path, sep="\t", index=False)
    pd.DataFrame(rare_rows).to_csv(rare_path, sep="\t", index=False)

    with gzip.open(per_site_path, "wt", encoding="utf-8") as fh:
        merged[["CHROM", "POS", "REF", "ALT", "AN", "AC", "ANCESTRAL", "DAC", "DAF"]].to_csv(
            fh, sep="\t", index=False
        )

    summary = {
        "chr": args.chr,
        "n_sites_polarizable": int(merged.shape[0]),
        "rare_tail_max_ac": args.rare_tail_max_ac,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
