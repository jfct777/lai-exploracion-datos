#!/usr/bin/env python3
"""Construye el feature store por-individuo del núcleo V.02-A (Q + Sigma-l + densidad).

Dos modos:
  define-cohort  -> fija la cohorte certificada N (rare ∩ metadata, dedup explícito) + assert.
  aggregate      -> une Q + Sigma-l + densidad + flags -> feature_store.{tsv,parquet} + manifest.

La densidad sale del bloque PSC de `bcftools stats -s -` (lo produce el módulo, por cromosoma):
sumando nNonRefHom/nHets/nMissing por individuo a lo largo de los autosomas. Tres lecturas, no una:
dosis ALT, sitios-portador y sitios-no-missing (denominador), para distinguir "menos raras
biológicas" de "menos datos interrogables".
"""

import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import sys

# Orden de las columnas Q en metadata, de 11 a 14, y su etiqueta de ancestría.
# El orden no es AFR/EUR/NAM/EAS; permutar Q produciría un error silencioso (contrato §A2).
Q_COLUMN_TO_LABEL = {
    "Autosomes_Indigenous_anc": "NAM",
    "Autosomes_European_anc": "EUR",
    "Autosomes_EastAsian_anc": "EAS",
    "Autosomes_African_anc": "AFR",
}
Q_OUTPUT_ORDER = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR"]


def _read_tsv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return reader.fieldnames, list(reader)


def _file_provenance(path):
    """Tamaño + mtime + sha256 (barato salvo VCFs; aquí solo TSV/JSON chicos)."""
    st = os.stat(path)
    prov = {"path": os.path.abspath(path), "size_bytes": st.st_size,
            "mtime_utc": _dt.datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z"}
    if st.st_size <= 256 * 1024 * 1024:  # sha256 solo si <=256 MB
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        prov["sha256"] = h.hexdigest()
    return prov


# ---------------------------------------------------------------------------
# Modo define-cohort
# ---------------------------------------------------------------------------
def define_cohort(args):
    """Define la cohorte canónica y escribe su orden y procedencia."""
    with open(args.rare_samples, "r", encoding="utf-8") as fh:
        rare = [ln.strip() for ln in fh if ln.strip()]
    rare_set = set(rare)
    if len(rare_set) != len(rare):
        sys.exit(f"ERROR: la lista de muestras del VCF de raras tiene duplicados ({len(rare) - len(rare_set)}).")

    header, rows = _read_tsv_rows(args.metadata)
    if "ID" not in header:
        sys.exit("ERROR: metadata sin columna 'ID'.")

    # Las filas duplicadas solo se colapsan si son idénticas; cualquier diferencia termina el proceso.
    by_id = {}
    for r in rows:
        by_id.setdefault(r["ID"], []).append(r)
    dup_report = []
    meta_ids = set()
    for mid, recs in by_id.items():
        if len(recs) == 1:
            meta_ids.add(mid)
            continue
        uniq = {tuple(sorted(rec.items())) for rec in recs}
        if len(uniq) == 1:
            dup_report.append({"id": mid, "n_rows": len(recs), "action": "colapsado (filas idénticas)"})
            meta_ids.add(mid)
        else:
            sys.exit(f"ERROR: ID duplicado con filas distintas: {mid} ({len(recs)} filas, "
                     f"{len(uniq)} distintas). Resolver a mano antes de fijar la cohorte.")

    cohort = sorted(rare_set & meta_ids)
    rare_not_meta = sorted(rare_set - meta_ids)
    meta_not_rare = sorted(meta_ids - rare_set)

    report = {
        "n_rare_samples": len(rare_set),
        "n_metadata_ids_raw": len(rows),
        "n_metadata_ids_dedup": len(meta_ids),
        "n_cohort_intersect": len(cohort),
        "expected_n": args.expected_n,
        "dedup_handled": dup_report,
        "n_rare_not_in_metadata": len(rare_not_meta),
        "n_metadata_not_in_rare": len(meta_not_rare),
        "rare_not_in_metadata_examples": rare_not_meta[:10],
        "metadata_not_in_rare_examples": meta_not_rare[:10],
    }
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    if len(cohort) != args.expected_n:
        sys.exit(f"ERROR: cohorte = {len(cohort)} ≠ esperado {args.expected_n}. "
                 f"Ver {args.report} (rare_not_meta={len(rare_not_meta)}, "
                 f"meta_not_rare={len(meta_not_rare)}).")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("sample_id\n")
        for s in cohort:
            fh.write(s + "\n")
    print(f"[define-cohort] cohorte certificada N={len(cohort)} -> {args.out}")


# ---------------------------------------------------------------------------
# Modo aggregate
# ---------------------------------------------------------------------------
def aggregate(args):
    """Une canales de variables y publica la tabla final y su manifiesto."""
    import pandas as pd

    cohort = pd.read_csv(args.cohort, sep="\t")["sample_id"].astype(str).tolist()
    df = pd.DataFrame({"sample_id": cohort})

    # Q y sexo se leen por nombre de columna, no por posición.
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str)
    meta = meta.drop_duplicates(subset=["ID"])  # dedup ya validado en define-cohort
    missing_q = [c for c in Q_COLUMN_TO_LABEL if c not in meta.columns]
    if missing_q:
        sys.exit(f"ERROR: metadata sin columnas de Q: {missing_q}")
    qcols = {col: f"Q_{lab}" for col, lab in Q_COLUMN_TO_LABEL.items()}
    keep = ["ID"] + list(Q_COLUMN_TO_LABEL)
    if "Sex" in meta.columns:
        keep.append("Sex")
    if "Cohort" in meta.columns:
        keep.append("Cohort")
    meta = meta[keep].rename(columns={"ID": "sample_id", **qcols, "Sex": "sex", "Cohort": "cohort"})
    for c in Q_OUTPUT_ORDER:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    df = df.merge(meta, on="sample_id", how="left")

    # --- Sigma-l (individual_sharing_summary); ausentes = aislados -> 0 ---
    sig = pd.read_csv(args.sigma_summary, sep="\t")
    required_sigma_cols = {"sample_id", "total_shared_bp", "n_sharing_partners"}
    missing_sigma = required_sigma_cols - set(sig.columns)
    if missing_sigma:
        sys.exit(f"ERROR: individual_sharing_summary.tsv sin columnas críticas de Σℓ: {missing_sigma}")
    sig_cols = ["sample_id", "n_sharing_partners", "n_segments_involved",
                "total_shared_bp", "n_chromosomes_with_sharing"]
    sig = sig[[c for c in sig_cols if c in sig.columns]]
    sig["sample_id"] = sig["sample_id"].astype(str)
    n_sigma_not_in_cohort = int((~sig["sample_id"].isin(cohort)).sum())
    df = df.merge(sig, on="sample_id", how="left")
    # flag_aislado indica ausencia del resumen de M14; se detecta por NaN antes
    # del fillna. Un total_shared_bp==0.0 presente en el resumen se conserva como dato válido.
    df["flag_aislado"] = df["total_shared_bp"].isna()
    for c in ["n_sharing_partners", "n_segments_involved", "total_shared_bp", "n_chromosomes_with_sharing"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # --- densidad: suma PSC por individuo a lo largo de los cromosomas ---
    psc_files = sorted(glob.glob(args.psc_glob))
    if not psc_files:
        sys.exit(f"ERROR: sin archivos PSC que matcheen {args.psc_glob}")
    psc = pd.concat([pd.read_csv(f, sep="\t") for f in psc_files], ignore_index=True)
    for c in ["nRefHom", "nNonRefHom", "nHets", "nMissing"]:
        psc[c] = pd.to_numeric(psc[c], errors="coerce").fillna(0)
    agg = psc.groupby("sample_id", as_index=False)[["nRefHom", "nNonRefHom", "nHets", "nMissing"]].sum()
    agg["rare_alt_dosage_sum"] = agg["nHets"] + 2 * agg["nNonRefHom"]
    agg["rare_carrier_site_count"] = agg["nHets"] + agg["nNonRefHom"]
    agg["rare_gt_nonmissing_sites"] = agg["nRefHom"] + agg["nHets"] + agg["nNonRefHom"]
    agg["rare_missing_sites"] = agg["nMissing"]
    agg["rare_density"] = agg["rare_alt_dosage_sum"] / agg["rare_gt_nonmissing_sites"].replace(0, pd.NA)
    agg["sample_id"] = agg["sample_id"].astype(str)
    df = df.merge(agg[["sample_id", "rare_alt_dosage_sum", "rare_carrier_site_count",
                       "rare_gt_nonmissing_sites", "rare_missing_sites", "rare_density"]],
                  on="sample_id", how="left")

    df["flag_missing_Q"] = df[Q_OUTPUT_ORDER].isna().any(axis=1)
    df["flag_missing_density"] = df["rare_gt_nonmissing_sites"].isna()

    # Orden de columnas estable
    front = ["sample_id"] + Q_OUTPUT_ORDER
    for extra in ["cohort", "sex"]:
        if extra in df.columns:
            front.append(extra)
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    if len(df) != args.expected_n:
        sys.exit(f"ERROR: feature_store tiene {len(df)} filas ≠ esperado {args.expected_n}.")

    tsv_path = f"{args.out_prefix}.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    parquet_written = False
    try:
        df.to_parquet(f"{args.out_prefix}.parquet", index=False)
        parquet_written = True
    except Exception as e:  # pyarrow ausente u otro -> no mata la corrida
        print(f"[aggregate] WARN: parquet no escrito ({e}); TSV + manifest sí.", file=sys.stderr)

    flag_counts = {c: int(df[c].sum()) for c in df.columns if c.startswith("flag_")}
    manifest = {
        "module": "M20_BUILD_FEATURE_STORE",
        "build_date": args.build_date,
        "n_rows": int(len(df)),
        "expected_n": args.expected_n,
        "n_sigma_not_in_cohort": n_sigma_not_in_cohort,
        "columns": list(df.columns),
        "q_order_mapping": {lab: col for col, lab in Q_COLUMN_TO_LABEL.items()},
        "q_output_order": Q_OUTPUT_ORDER,
        "flag_counts": flag_counts,
        "parquet_written": parquet_written,
        "inputs": {
            "cohort": _file_provenance(args.cohort),
            "sigma_summary": _file_provenance(args.sigma_summary),
            "metadata": _file_provenance(args.metadata),
            "psc_files": [_file_provenance(f) for f in psc_files],
        },
        "outputs": {"feature_store_tsv": _file_provenance(tsv_path)},
    }
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"[aggregate] feature_store N={len(df)} -> {tsv_path} (parquet={parquet_written})")


def main():
    """Ejecuta la etapa de cohorte o la etapa de agregación."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("define-cohort")
    pc.add_argument("--rare-samples", required=True)
    pc.add_argument("--metadata", required=True)
    pc.add_argument("--dedup-id", default=None, help="solo informativo; el dedup es general por fila idéntica")
    pc.add_argument("--expected-n", type=int, required=True)
    pc.add_argument("--out", required=True)
    pc.add_argument("--report", required=True)
    pc.set_defaults(func=define_cohort)

    pa = sub.add_parser("aggregate")
    pa.add_argument("--cohort", required=True)
    pa.add_argument("--sigma-summary", required=True)
    pa.add_argument("--metadata", required=True)
    pa.add_argument("--psc-glob", required=True)
    pa.add_argument("--expected-n", type=int, required=True)
    pa.add_argument("--build-date", required=True)
    pa.add_argument("--out-prefix", required=True)
    pa.add_argument("--manifest", required=True)
    pa.set_defaults(func=aggregate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
