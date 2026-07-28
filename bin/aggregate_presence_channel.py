#!/usr/bin/env python3
"""Módulo 21 — agregado genome-wide del canal de presencia externa, para UN panel.

Junta los ``*.external_presence.summary.json`` por cromosoma de un mismo panel. En lugar de
promediar fracciones, suma numeradores y denominadores y vuelve a calcular las tasas y el
enriquecimiento de todo el genoma. Sigue el mismo patrón de agregado de M17.

Salidas:
  <prefix>.external_presence.genomewide.json   conteos sumados + fracciones recomputadas
  <prefix>.external_presence.per_chr.tsv       una fila por cromosoma (auditoría/figura)
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import logging
from pathlib import Path

LOG = logging.getLogger("aggregate_presence")

STATES = ("PRESENT_ALLELE", "PRESENT_POS_ONLY", "REF_MISMATCH", "ABSENT_FROM_PANEL")


def _chrom_sort_key(chrom: str):
    """Orden natural chr1..chr22, luego X/Y/MT."""
    c = chrom[3:] if chrom.startswith("chr") else chrom
    return (0, int(c)) if c.isdigit() else (1, c)


def main() -> int:
    """Agrega por panel los resultados cromosómicos del canal de presencia."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="*.external_presence.summary.json",
                   help="patrón de los summary.json por-cromosoma en el cwd")
    p.add_argument("--out-prefix", required=True, help="prefijo de salida (típicamente el panel_id)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    files = sorted(globmod.glob(args.glob))
    if not files:
        raise SystemExit(f"No se encontró ningún summary con patrón '{args.glob}' en el cwd")
    LOG.info("agregando %d summaries del panel", len(files))

    summaries = [json.loads(Path(f).read_text()) for f in files]
    panel_ids = {s["panel_id"] for s in summaries}
    if len(panel_ids) != 1:
        raise SystemExit(f"Los summaries mezclan paneles distintos: {sorted(panel_ids)} "
                         "(este agregado es por-panel; revisar el glob/wiring)")
    panel_id = panel_ids.pop()

    counts_full = {s: 0 for s in STATES}
    n_target = n_in_lcr = n_cpg = 0
    present_in_lcr = present_out_lcr = present_cpg = present_noncpg_ti = present_tv = 0
    pf_rows = pf_pass = pf_nonpass = 0
    has_drop = any(s.get("drop_sample_found") for s in summaries)
    ec57_present_drop = ec57_lost = 0
    per_chr_rows = []

    for s in summaries:
        for st in STATES:
            counts_full[st] += s["counts_full"].get(st, 0)
        n_target += s["n_target_rare_snv"]
        n_in_lcr += s["n_target_in_lcr_segdup"]
        n_cpg += s["n_target_cpg_ti"]
        strat = s["present_allele_stratification"]
        present_in_lcr += strat["in_lcr"]
        present_out_lcr += strat["out_lcr"]
        present_cpg += strat["cpg_ti"]
        present_noncpg_ti += strat["noncpg_ti"]
        present_tv += strat["transversion"]
        pf = s.get("panel_filter", {})
        pf_rows += pf.get("n_rows", 0)
        pf_pass += pf.get("n_pass", 0)
        pf_nonpass += pf.get("n_nonpass", 0)
        if s.get("ec57_sensitivity"):
            ec57_present_drop += s["ec57_sensitivity"]["present_allele_drop"]
            ec57_lost += s["ec57_sensitivity"]["lost_when_drop"]
        per_chr_rows.append({
            "chrom": s["chrom"],
            "n_target_rare_snv": s["n_target_rare_snv"],
            "present_allele": s["counts_full"].get("PRESENT_ALLELE", 0),
            "present_pos_only": s["counts_full"].get("PRESENT_POS_ONLY", 0),
            "ref_mismatch": s["counts_full"].get("REF_MISMATCH", 0),
            "absent": s["counts_full"].get("ABSENT_FROM_PANEL", 0),
            "fraction_present_allele": s["fraction_present_allele"],
            "lcr_enrichment_vs_target": strat["lcr_enrichment_vs_target"],
            "frac_nonpass": pf.get("frac_nonpass", 0.0),
        })

    n_present = counts_full["PRESENT_ALLELE"]
    frac_present_in_lcr = present_in_lcr / max(n_present, 1)
    frac_target_in_lcr = n_in_lcr / max(n_target, 1)
    genomewide = {
        "panel_id": panel_id,
        "panel_version": summaries[0]["panel_version"],
        "panel_n_samples": summaries[0]["panel_n_samples"],
        "n_chromosomes": len(summaries),
        "n_target_rare_snv": n_target,
        "counts_full": counts_full,
        "ref_mismatch": counts_full["REF_MISMATCH"],
        "fraction_present_allele": n_present / max(n_target, 1),
        "fraction_present_pos_only": counts_full["PRESENT_POS_ONLY"] / max(n_target, 1),
        "panel_filter": {
            "n_rows": pf_rows, "n_pass": pf_pass, "n_nonpass": pf_nonpass,
            "frac_nonpass": pf_nonpass / max(pf_rows, 1),
            "pass_only_applied": bool(summaries[0].get("panel_filter", {}).get("pass_only_applied")),
        },
        "present_allele_stratification": {
            "in_lcr": present_in_lcr, "out_lcr": present_out_lcr,
            "frac_in_lcr": frac_present_in_lcr, "frac_target_in_lcr": frac_target_in_lcr,
            "lcr_enrichment_vs_target": frac_present_in_lcr / max(frac_target_in_lcr, 1e-12),
            "cpg_ti": present_cpg, "noncpg_ti": present_noncpg_ti, "transversion": present_tv,
        },
        "ec57_sensitivity": {
            "present_allele_drop": ec57_present_drop,
            "lost_when_drop": ec57_lost,
            "frac_present_lost_when_drop": ec57_lost / max(n_present, 1),
        } if has_drop else None,
        "interpretation": ("Agregado genome-wide del canal de presencia externa. "
                           "PRESENT_ALLELE descarta privacidad; ABSENT_FROM_PANEL no confirma founder."),
    }
    Path(f"{args.out_prefix}.external_presence.genomewide.json").write_text(
        json.dumps(genomewide, indent=2))

    per_chr_rows.sort(key=lambda r: _chrom_sort_key(r["chrom"]))
    cols = ["chrom", "n_target_rare_snv", "present_allele", "present_pos_only", "ref_mismatch",
            "absent", "fraction_present_allele", "lcr_enrichment_vs_target", "frac_nonpass"]
    with open(f"{args.out_prefix}.external_presence.per_chr.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in per_chr_rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    LOG.info("[%s] genome-wide: PRESENT_ALLELE=%d / %d (%.3f%%); REF_MISMATCH=%d; frac_nonpass=%.4f",
             panel_id, n_present, n_target, 100 * n_present / max(n_target, 1),
             counts_full["REF_MISMATCH"], pf_nonpass / max(pf_rows, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
