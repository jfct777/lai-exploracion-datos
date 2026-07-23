#!/usr/bin/env python3
"""
Agrega los summary.json por-cromosoma de rare_variants_in_lai_tracts.py al
resultado genome-wide. Suma CONTEOS CRUDOS (no promedia fracciones) y recomputa
el enriquecimiento sobre el total acumulado.
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ANC = ["African", "European", "Native_American"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="Patrón de los *.summary.json por crom")
    p.add_argument("--out_prefix", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"Sin archivos para patrón {args.glob}")

    tot = {k: {a: 0.0 for a in ANC} for k in
           ("observed_exact", "observed_fractional_raw", "expected_copies_raw",
            "expected_copies_exact_raw")}
    n_events = n_copies = n_amb = n_unassign = n_skip = 0
    per_chr_rows = []

    for f in files:
        s = json.loads(Path(f).read_text())
        chrom = Path(f).stem.replace(".summary", "")
        for k in tot:
            # bugfix 2026-06-02: expected_copies_exact_raw es el baseline correcto para la
            # métrica exacta. Si falta (summary viejo pre-bugfix), cae a expected_copies_raw
            # (comportamiento antiguo) — re-correr los scans para que exista.
            src = s.get(k) if s.get(k) is not None else s["expected_copies_raw"]
            for a in ANC:
                tot[k][a] += src[a]
        n_events += s["n_carrier_events"]
        n_copies += s["n_rare_allele_copies"]
        n_amb += s["n_ambiguous_copies_phase_limited"]
        n_unassign += s["n_unassignable_outside_painting"]
        n_skip += s["n_skipped_not_in_2619"]
        e = s["enrichment_exact_obs_over_exp"]
        per_chr_rows.append({"chrom": chrom, **{f"enr_{a}": e[a] for a in ANC},
                             "n_copies": s["n_rare_allele_copies"]})

    def enr(observed, expected):
        obs_tot = sum(observed.values())
        exp_tot = sum(expected.values())
        out = {}
        for a in ANC:
            of = observed[a] / obs_tot if obs_tot else 0.0
            ef = expected[a] / exp_tot if exp_tot else 0.0
            out[a] = round(of / ef, 4) if ef > 0 else None
        return out

    gw = {
        "n_chromosomes": len(files),
        "n_carrier_events": n_events,
        "n_rare_allele_copies": n_copies,
        "pct_ambiguous_of_copies": round(100 * n_amb / n_copies, 2) if n_copies else None,
        "n_unassignable_outside_painting": n_unassign,
        "n_skipped_not_in_2619": n_skip,
        "observed_exact_copies": {a: tot["observed_exact"][a] for a in ANC},
        "expected_copies_raw": {a: round(tot["expected_copies_raw"][a], 1) for a in ANC},
        "expected_copies_exact_raw": {a: round(tot["expected_copies_exact_raw"][a], 1) for a in ANC},
        "enrichment_genomewide_exact": enr(tot["observed_exact"], tot["expected_copies_exact_raw"]),
        "enrichment_genomewide_fractional": enr(tot["observed_fractional_raw"], tot["expected_copies_raw"]),
        "CAVEAT": "Descriptivo. Baseline posicional NO remueve la tautología "
                  "burden-raras ∝ NAM a nivel-individuo (requiere residualización, Paso 2).",
    }
    Path(f"{args.out_prefix}.genomewide.json").write_text(json.dumps(gw, indent=2, ensure_ascii=False))
    pd.DataFrame(per_chr_rows).to_csv(f"{args.out_prefix}.per_chr_enrichment.tsv", sep="\t", index=False)
    print(json.dumps(gw, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
