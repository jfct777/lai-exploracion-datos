#!/usr/bin/env python3
"""Smoke técnico de la matriz rara de M23 sobre chr22. No ejecuta CV ni
selecciona modelos ni hiperparametros, no reporta metricas de acierto. Su unico proposito es validar
que la maquinaria tecnica del benchmark corre sobre las dimensiones reales:

  1. dimensiones      -> n_train, variantes de entrada/retenidas
  2. missingness      -> tasas por variante; la matriz no tiene NaN (faltantes imputados a 0)
  3. memoria          -> RSS pico del proceso durante el fit
  4. tiempo           -> wallclock del unico fit tecnico
  5. sparse-safety    -> la matriz permanece sparse en todo el pipeline (nunca se densifica)
  6. bloqueo fold 3   -> ninguna muestra de la matriz es TEST; n_filas == n_train

El único ajuste es técnico y usa una etiqueta permutada para destruir la señal: SAGA elastic-net, C fijo,
l1_ratio fijo. Solo se reportan tiempo, memoria, convergencia (n_iter_ < max_iter) e iteraciones.
La regla de retencion fold-fitted (MAC_train>=k, n_alt_carriers_train>=k, missingness, varianza) se
se aplica sobre todo TRAIN porque el smoke no hace CV; la versión por fold corresponde al
benchmark cientifico posterior, fuera de este smoke.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler


def _peak_rss_mb():
    # ru_maxrss en Linux esta en KiB -> MiB
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main():
    """Ejecuta el smoke técnico de la matriz rara y escribe sus métricas."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-npz", required=True, type=Path)
    ap.add_argument("--variants-tsv", required=True, type=Path)
    ap.add_argument("--samples-tsv", required=True, type=Path)
    ap.add_argument("--split-manifest", required=True, type=Path)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--sample-id-col", default="sample_id")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--label-col", default="y")
    ap.add_argument("--train-label", default="TRAIN")
    ap.add_argument("--test-label", default="TEST")
    ap.add_argument("--min-mac-train", type=int, default=2)
    ap.add_argument("--min-alt-carriers-train", type=int, default=2)
    ap.add_argument("--max-missing-train", type=float, default=0.10)
    ap.add_argument("--min-variance-train", type=float, default=0.0)
    ap.add_argument("--smoke-c", type=float, default=0.01)
    ap.add_argument("--smoke-l1-ratio", type=float, default=0.5)
    ap.add_argument("--smoke-max-iter", type=int, default=1000)
    ap.add_argument("--smoke-tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    checks = {}          # Cada nombre guarda un estado PASS o FAIL y sus detalles.
    report = {"chrom": args.chrom, "sklearn_version": sklearn.__version__, "seed": args.seed}

    X = sp.load_npz(args.matrix_npz).tocsc()
    variants = pd.read_csv(args.variants_tsv, sep="\t")
    samples = pd.read_csv(args.samples_tsv, sep="\t", dtype={args.sample_id_col: str})[args.sample_id_col].tolist()
    manifest = pd.read_csv(args.split_manifest, sep="\t", dtype={args.sample_id_col: str})
    split_of = dict(zip(manifest[args.sample_id_col], manifest[args.split_col]))
    label_of = dict(zip(manifest[args.sample_id_col], manifest[args.label_col]))

    n_rows, n_var_input = X.shape

    # --- (6) bloqueo del fold 3 -------------------------------------------------------------------
    test_hits = [s for s in samples if split_of.get(s) == args.test_label]
    non_train = [s for s in samples if split_of.get(s) != args.train_label]
    n_train_manifest = int((manifest[args.split_col] == args.train_label).sum())
    fold3_ok = (len(test_hits) == 0) and (len(non_train) == 0) and (n_rows == len(samples) == n_train_manifest)
    checks["fold3_block"] = {
        "status": "PASS" if fold3_ok else "FAIL",
        "test_samples_in_matrix": len(test_hits),
        "non_train_samples_in_matrix": len(non_train),
        "matrix_rows": n_rows, "n_train_manifest": n_train_manifest,
    }

    # Retención ajustada sobre todo TRAIN; el smoke no ejecuta CV.
    mac = variants["mac_train"].to_numpy()
    carriers = variants["n_alt_carriers_train"].to_numpy()
    missing_rate = variants["missing_rate_train"].to_numpy()
    keep_mac = mac >= args.min_mac_train
    keep_carriers = carriers >= args.min_alt_carriers_train
    keep_missing = missing_rate <= args.max_missing_train
    keep = keep_mac & keep_carriers & keep_missing
    Xk = X[:, keep].tocsc()

    # VarianceThreshold fit en train (sparse-safe); descarta columnas monomorficas tras el filtro.
    vt = VarianceThreshold(threshold=args.min_variance_train)
    Xk = vt.fit_transform(Xk)
    n_retained = Xk.shape[1]

    checks["dimensions"] = {
        "status": "PASS" if (n_retained > 0 and n_rows == n_train_manifest) else "FAIL",
        "n_train": n_rows, "n_variants_input": int(n_var_input),
        "dropped_by_mac_lt_%d" % args.min_mac_train: int((~keep_mac).sum()),
        "dropped_by_carriers_lt_%d" % args.min_alt_carriers_train: int((~keep_carriers).sum()),
        "dropped_by_missing_gt_%.3f" % args.max_missing_train: int((~keep_missing).sum()),
        "dropped_by_joint_retention": int((~keep).sum()),
        "dropped_by_variance": int(keep.sum() - n_retained),
        "n_variants_retained": int(n_retained),
    }

    # --- (2) missingness / sin NaN ---------------------------------------------------------------
    has_nan = bool(np.isnan(Xk.data).any()) if Xk.nnz else False
    retained_missing = missing_rate[keep]
    checks["missingness"] = {
        "status": "PASS" if not has_nan else "FAIL",
        "matrix_has_nan": has_nan,
        "retained_missing_rate_max": round(float(retained_missing.max()), 6) if retained_missing.size else None,
        "retained_missing_rate_mean": round(float(retained_missing.mean()), 6) if retained_missing.size else None,
        "note": "faltantes imputados a 0 (no-portador) en extraccion; matriz entera sin NaN",
    }

    # --- (5) sparse-safety: escalado sparse-safe, nunca densificar --------------------------------
    scaler = MaxAbsScaler(copy=False)
    Xs = scaler.fit_transform(Xk)
    Xs = Xs.tocsr()  # CSR para el fit (acceso por fila); sigue siendo sparse
    sparse_ok = sp.issparse(Xk) and sp.issparse(Xs)
    checks["sparse_safety"] = {
        "status": "PASS" if sparse_ok else "FAIL",
        "matrix_is_sparse": bool(sparse_ok),
        "format_at_fit": Xs.format,
        "densified": False,
        "note": "MaxAbsScaler(copy=False) sparse; .tocsr() mantiene sparse; nunca .toarray()",
    }

    # Ajuste técnico con SAGA elastic-net, etiqueta permutada y C/l1_ratio fijos.
    rng = np.random.RandomState(args.seed)
    y = np.array([int(label_of[s]) for s in samples])
    y_perm = rng.permutation(y)
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=args.smoke_l1_ratio, C=args.smoke_c,
        max_iter=args.smoke_max_iter, tol=args.smoke_tol, random_state=args.seed,
    )
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    clf.fit(Xs, y_perm)
    fit_seconds = round(time.perf_counter() - t0, 3)
    rss_after = _peak_rss_mb()
    n_iter = int(np.ravel(clf.n_iter_)[0])
    converged = n_iter < args.smoke_max_iter
    checks["technical_fit"] = {
        "status": "PASS" if converged else "FAIL",   # PASS indica que n_iter quedó por debajo de max_iter.
        "note": ("ajuste técnico con etiqueta permutada y sin señal; solo se reportan tiempo/memoria/"
                 "convergencia/iteraciones; ninguna metrica de acierto. PASS exige converged=true"),
        "solver": "saga", "penalty": "elasticnet",
        "C": args.smoke_c, "l1_ratio": args.smoke_l1_ratio,
        "max_iter": args.smoke_max_iter, "tol": args.smoke_tol,
        "fit_seconds": fit_seconds,
        "n_iter": n_iter,
        "converged": bool(converged),
        "peak_rss_mb": max(rss_before, rss_after),
    }

    overall = "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "FAIL"
    report.update({
        "overall_status": overall,
        "retention_rule": {
            "min_mac_train": args.min_mac_train,
            "min_alt_carriers_train": args.min_alt_carriers_train,
            "max_missing_train": args.max_missing_train,
            "min_variance_train": args.min_variance_train,
            "note": "aplicada sobre todo TRAIN en el smoke sin CV; por fold en el benchmark científico",
        },
        "checks": checks,
        "scope": ("Smoke técnico de chr22: sin CV, grilla ni métricas científicas; etiqueta permutada. "
                  "Fold 3 nunca cargado."),
    })
    (args.outdir / f"{args.chrom}.rare_bench_smoke_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if overall != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
