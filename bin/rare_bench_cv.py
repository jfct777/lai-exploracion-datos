#!/usr/bin/env python3
"""Benchmark M23 con validación cruzada anidada dentro del conjunto de entrenamiento.

Compara cinco representaciones:
  A = ancestría y sexo
  B = densidad de variantes raras y de portadores
  C = A + B
  D = matriz reducida de variantes raras
  E = C + matriz reducida

El contraste principal E-C mide el aporte multivariado de la matriz por encima
de la carga marginal. La métrica principal es balanced accuracy y AUROC se
informa como métrica secundaria.

La validación exterior deja fuera uno de los folds de entrenamiento {0,1,2,4}.
La validación interior usa StratifiedGroupKFold y agrupa por componente de
parentesco. El fold 3 corresponde a TEST y su presencia provoca la interrupción
del proceso.

La selección de variantes y el escalado se ajustan dentro de cada fold. La
matriz se mantiene en formato disperso durante todo el cálculo.

Los modos `preflight`, `abc`, `fold` y `aggregate` dividen el trabajo en tareas
reanudadables. El modo `monolithic` conserva la ejecución completa para
comprobar la equivalencia de resultados. Todas las tareas comparten las mismas
entradas y la misma huella de código, datos, configuración y entorno numérico.

Los modos `refit` y `refit_aggregate` son un control acotado de convergencia del
solver: reajustan el modelo final de un set y fold con los hiperparámetros ya
seleccionados en una corrida base y un techo de iteraciones más alto. No hay
grilla ni selección nueva; las métricas de los sets densos se reutilizan tal
cual desde la corrida base.
"""
import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import _common
_common.pin_env()  # env vars de 1-hilo ANTES de numpy/sklearn (import-time; standalone/nube)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

# loky lanza TerminatedWorkerError cuando uno de sus procesos termina de forma inesperada.
# El código distingue un SIGKILL por falta de memoria de otros errores del proceso.
from joblib.externals.loky.process_executor import TerminatedWorkerError

_THREAD_STATE = _common.pin_threadpools()  # El límite de hilos se aplica en runtime y queda en la huella.

# Los sets A y B siguen la definición de M22 en bin/model_primary_cv.py.
A_COLS = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR", "sex"]
B_DENSITY_COL = "rare_density"           # columna directa del modeling_master
B_CARRIER_NUM = "rare_carrier_site_count"  # solo para DERIVAR carrier_density (no es predictor crudo)
B_CARRIER_DEN = "rare_gt_nonmissing_sites"  # denominador de carrier_density (no es predictor crudo)
B_NAMES = ["rare_density", "carrier_density"]  # orden M22

# Nombres canonicos de los sets Elastic Net (unica fuente; el particionado los reusa).
SET_A, SET_B, SET_C = "A_Q_sex", "B_burden", "C_Q_sex_burden"
SET_D, SET_E = "D_matrix_elasticnet", "E_full_elasticnet"
ABC_SETS = [SET_A, SET_B, SET_C]        # densos, milisegundos -> tarea compacta
HEAVY_SETS = [SET_D, SET_E]            # matriz -> una tarea por (set,fold)


class RareRetention(BaseEstimator, TransformerMixin):
    """Retención fold-fitted de la matriz rara sparse. fit() calcula la máscara de columnas con el
    train del fold usando los límites de MAC, número de portadores y varianza. transform() selecciona
    esas columnas sin densificar la matriz. El resultado puede variar entre folds.
    """

    def __init__(self, min_mac=2, min_carriers=2, min_variance=0.0):
        self.min_mac = min_mac
        self.min_carriers = min_carriers
        self.min_variance = min_variance

    def fit(self, X, y=None):
        Xc = sp.csc_matrix(X)
        # MAC = suma de dosis alt por columna (train del fold); carriers = individuos con dosis>0.
        mac = np.asarray(Xc.sum(axis=0)).ravel()
        carriers = np.diff(Xc.indptr)  # nnz por columna == nro de portadores (ceros no almacenados)
        # varianza por columna sparse: E[x^2] - E[x]^2
        n = Xc.shape[0]
        sq = Xc.multiply(Xc)
        mean = mac / n
        var = np.asarray(sq.sum(axis=0)).ravel() / n - mean ** 2
        self.mask_ = (mac >= self.min_mac) & (carriers >= self.min_carriers) & (var > self.min_variance)
        self.n_in_ = Xc.shape[1]
        self.n_out_ = int(self.mask_.sum())
        return self

    def transform(self, X):
        Xc = sp.csc_matrix(X)
        return Xc[:, self.mask_].astype(np.float32)


def _elasticnet_clf(max_iter, tol, seed, class_weight):
    return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=max_iter, tol=tol,
                              random_state=seed, class_weight=class_weight)


def _pipeline_dense_en(max_iter, tol, seed, class_weight):
    # A / B / C: features densas pequenas -> estandarizar + elastic net.
    return Pipeline([("scale", StandardScaler()), ("clf", _elasticnet_clf(max_iter, tol, seed, class_weight))])


def _pipeline_matrix_en(ret_params, max_iter, tol, seed, class_weight):
    # D elastic net: retencion fold-fitted -> MaxAbs sparse -> elastic net saga.
    return Pipeline([("ret", RareRetention(**ret_params)),
                     ("scale", MaxAbsScaler()),
                     ("clf", _elasticnet_clf(max_iter, tol, seed, class_weight))])


def _pipeline_matrix_svd(ret_params, n_components, max_iter, tol, seed, class_weight):
    # D svd: retencion -> TruncatedSVD -> estandarizar -> logistica L2.
    return Pipeline([("ret", RareRetention(**ret_params)),
                     ("svd", TruncatedSVD(n_components=n_components, random_state=seed)),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                                                max_iter=max_iter, tol=tol, random_state=seed,
                                                class_weight=class_weight))])


def _combined_transformer_en(n_dense, ret_params):
    # E elastic net: [C denso (0:n_dense) | matriz sparse (n_dense:)] -> C estandarizado (sin centrar,
    # sparse-safe) ++ matriz retenida+MaxAbs -> elastic net. ColumnTransformer preserva sparsity.
    return ColumnTransformer([
        ("dense", StandardScaler(with_mean=False), slice(0, n_dense)),
        ("matrix", Pipeline([("ret", RareRetention(**ret_params)), ("scale", MaxAbsScaler())]),
         slice(n_dense, None)),
    ], sparse_threshold=1.0)


def _combined_transformer_svd(n_dense, ret_params, n_components, seed):
    return ColumnTransformer([
        ("dense", StandardScaler(with_mean=False), slice(0, n_dense)),
        ("matrix", Pipeline([("ret", RareRetention(**ret_params)),
                             ("svd", TruncatedSVD(n_components=n_components, random_state=seed)),
                             ("scale", StandardScaler())]), slice(n_dense, None)),
    ], sparse_threshold=0.0)


def _en_grid(c_grid, l1_ratios, prefix="clf"):
    return {f"{prefix}__C": c_grid, f"{prefix}__l1_ratio": l1_ratios}


def _split_for_held(X, y, outer_groups, held):
    """Reproduce el split del modo monolítico para el outer fold `held`.

    Itera LeaveOneGroupOut en el mismo orden y devuelve los índices de entrenamiento y validación del
    grupo dejado fuera. Así se conserva el mismo orden en el modo particionado.
    """
    logo = LeaveOneGroupOut()
    for h, (tr, va) in zip(sorted(np.unique(outer_groups)), logo.split(X, y, groups=outer_groups)):
        if int(h) == int(held):
            return tr, va
    raise SystemExit(f"[cv] outer-fold {held} no existe en {sorted(int(x) for x in np.unique(outer_groups))}")


def _solver_convergence(fitted):
    """Obtiene n_iter y converged del clasificador final.

    LogisticRegression expone `n_iter_` como un array de un elemento en el caso binario. Si el paso
    `clf` no ofrece este dato, devuelve (None, None).
    """
    clf = getattr(fitted, "named_steps", {}).get("clf")
    n_it = getattr(clf, "n_iter_", None)
    max_it = getattr(clf, "max_iter", None)
    if n_it is None or max_it is None:
        return None, None
    n_it = int(np.max(np.asarray(n_it)))
    return n_it, bool(n_it < int(max_it))


def _peak_rss_gb():
    """Devuelve el pico de RSS en GiB.

    En Linux, ru_maxrss se expresa en KiB. `self` no incluye los workers de loky y `children` conserva
    el máximo de un hijo, no la suma. La traza de Nextflow contiene el pico total para la grilla. En
    el modo refit, que no crea hijos, `self` corresponde al pico del proceso.
    """
    ru_s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (round(ru_s / (1024 ** 2), 2), round(ru_c / (1024 ** 2), 2))


def _fit_one_fold(estimator, grid, X, y, held, tr, va, inner_group, inner_splits, scorer, seed,
                  n_jobs, threshold, pre_dispatch):
    """Ajusta y evalúa un outer fold.

    Esta es la unidad de reanudación del modo particionado y comparte el mismo cálculo con el modo
    monolítico. Si `grid` está vacío o es None, se hace un ajuste con los parámetros ya fijados.
    """
    Xtr = X[tr]
    ytr = y[tr]
    gtr = inner_group[tr]
    Xva = X[va]
    yva = y[va]
    inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    t_fit = time.perf_counter()
    if grid:
        # pre_dispatch acota cuantas tareas (y copias/intermedios por worker) loky mantiene en vuelo a la
        # vez: es la palanca que evita el pico de RAM que OOM-mataba al set E (matriz completa) con 16 workers.
        gs = GridSearchCV(estimator, grid, scoring=scorer, cv=inner, n_jobs=n_jobs, refit=True,
                          pre_dispatch=pre_dispatch, error_score="raise")
        gs.fit(Xtr, ytr, groups=gtr)
        best = gs.best_estimator_
        chosen = {k: v for k, v in gs.best_params_.items()}
    else:
        best = estimator.fit(Xtr, ytr)
        chosen = {}
    fit_seconds = round(time.perf_counter() - t_fit, 1)
    n_iter, converged = _solver_convergence(best)
    peak_self_gb, peak_children_gb = _peak_rss_gb()
    proba = best.predict_proba(Xva)[:, 1]
    ypred = (proba >= threshold).astype(int)
    # nro de columnas retenidas en este fold (auditoria anti-leakage: debe variar entre folds).
    # D: RareRetention es un paso directo del Pipeline. E: vive dentro del ColumnTransformer
    # ("comb") -> named_transformers_["matrix"] (Pipeline) -> named_steps["ret"].
    n_ret = None
    steps = getattr(best, "named_steps", {})
    for name, step in steps.items():
        if isinstance(step, RareRetention):
            n_ret = int(step.n_out_)
    if n_ret is None and "comb" in steps and hasattr(steps["comb"], "named_transformers_"):
        mt = steps["comb"].named_transformers_.get("matrix")
        if mt is not None and hasattr(mt, "named_steps") and "ret" in mt.named_steps:
            n_ret = int(mt.named_steps["ret"].n_out_)
    return {
        "held_out_fold": int(held),
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "balanced_accuracy": round(float(balanced_accuracy_score(yva, ypred)), 6),
        "auroc": round(float(roc_auc_score(yva, proba)), 6),
        "recall_pos": round(float(recall_score(yva, ypred, pos_label=1, zero_division=0)), 6),
        "recall_neg": round(float(recall_score(yva, ypred, pos_label=0, zero_division=0)), 6),
        "chosen_params": chosen,
        "n_retained_rare_cols": n_ret,
        # Auditoria del solver (campos NUEVOS, opcionales en el esquema del agregador para poder
        # seguir leyendo artefactos de corridas anteriores que no los traen).
        "n_iter": n_iter,
        "converged": converged,
        "fit_seconds": fit_seconds,
        "peak_rss_self_gb": peak_self_gb,
        "peak_rss_children_gb": peak_children_gb,
    }


def _fit_score_nested(estimator, grid, X, y, outer_groups, inner_group, inner_splits, scorer, seed,
                      n_jobs, threshold, pre_dispatch, svd_prefix=None):
    """CV anidada completa de un set (los 4 outer-folds). Delega cada fold en _fit_one_fold para que
    monolítico y particionado compartan el mismo cálculo."""
    logo = LeaveOneGroupOut()
    per_fold = []
    for held, (tr, va) in zip(sorted(np.unique(outer_groups)),
                              logo.split(X, y, groups=outer_groups)):
        per_fold.append(_fit_one_fold(estimator, grid, X, y, held, tr, va, inner_group,
                                      inner_splits, scorer, seed, n_jobs, threshold, pre_dispatch))
    return per_fold


def _agg(per_fold, key):
    v = np.array([f[key] for f in per_fold], dtype=float)
    return {"mean": round(float(v.mean()), 6), "sd": round(float(v.std(ddof=1)), 6),
            "per_fold": [round(float(x), 6) for x in v]}


def _paired_delta(pf_hi, pf_lo, key="balanced_accuracy"):
    # El delta se calcula entre entradas con el mismo held_out_fold.
    lo = {f["held_out_fold"]: f[key] for f in pf_lo}
    d = np.array([f[key] - lo[f["held_out_fold"]] for f in pf_hi], dtype=float)
    return {"mean": round(float(d.mean()), 6), "sd": round(float(d.std(ddof=1)), 6),
            "per_fold": [round(float(x), 6) for x in d]}


# ---------------------------------------------------------------------------------------------------
# Carga y validación compartidas por los modos que usan la matriz.
# ---------------------------------------------------------------------------------------------------
def load_context(args):
    """Carga la matriz y los metadatos, y valida el fold 3 antes de cualquier cálculo.

    También construye A, B, C y E_input. Devuelve un diccionario reutilizado por los modos monolithic,
    abc y fold. Todas las rutas se reciben mediante argumentos.
    """
    class_weight = None if str(args.class_weight).lower() == "none" else args.class_weight
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]
    l1_ratios = [float(x) for x in args.l1_ratios.split(",") if x.strip()]
    ret_params = {"min_mac": args.min_mac_train, "min_carriers": args.min_alt_carriers_train,
                  "min_variance": args.min_variance_train}

    X = sp.load_npz(args.matrix_npz).tocsr()
    samples = pd.read_csv(args.samples_tsv, sep="\t", dtype={args.sample_id_col: str})[args.sample_id_col].tolist()
    man = pd.read_csv(args.split_manifest, sep="\t", dtype={args.sample_id_col: str})
    man_by_id = man.set_index(args.sample_id_col)

    # Toda muestra de la matriz debe pertenecer a TRAIN y ninguna puede estar en el fold de TEST.
    missing_ids = [s for s in samples if s not in man_by_id.index]
    if missing_ids:
        raise SystemExit(f"[rare_bench_cv] {len(missing_ids)} muestras de la matriz no estan en el split: {missing_ids[:5]}")
    split_of = man_by_id[args.split_col].to_dict()
    fold_of = man_by_id[args.fold_col].to_dict()
    non_train = [s for s in samples if split_of[s] != args.train_label]
    test_fold_hits = [s for s in samples if int(fold_of[s]) == args.test_fold]
    if non_train or test_fold_hits:
        raise SystemExit(f"[rare_bench_cv] FOLD 3 / no-TRAIN en la matriz: non_train={len(non_train)} "
                         f"test_fold={len(test_fold_hits)} -> abortado (TEST nunca entra)")

    y = np.array([int(man_by_id.loc[s, args.label_col]) for s in samples])
    folds = np.array([int(fold_of[s]) for s in samples])
    groups = np.array([str(man_by_id.loc[s, args.group_col]) for s in samples])
    if args.test_fold in np.unique(folds):
        raise SystemExit(f"[rare_bench_cv] el fold de TEST {args.test_fold} aparece en outer folds {np.unique(folds)}")

    mm = pd.read_csv(args.modeling_master, sep="\t", dtype={args.sample_id_col: str}).set_index(args.sample_id_col)
    for col in A_COLS + [B_DENSITY_COL, B_CARRIER_NUM, B_CARRIER_DEN]:
        if col not in mm.columns:
            raise SystemExit(f"[rare_bench_cv] columna '{col}' ausente en {args.modeling_master}")
    A = mm.loc[samples, A_COLS].to_numpy(dtype=float)
    # Burden sigue la definición de M22: [rare_density, carrier_density], con carrier_density derivado.
    # Los conteos crudos no se usan como predictores; solo permiten calcular carrier_density.
    # El denominador debe ser distinto de cero, como exige M22.
    density = mm.loc[samples, B_DENSITY_COL].to_numpy(dtype=float)
    carrier_num = mm.loc[samples, B_CARRIER_NUM].to_numpy(dtype=float)
    carrier_den = mm.loc[samples, B_CARRIER_DEN].to_numpy(dtype=float)
    if (carrier_den == 0).any():
        raise SystemExit(f"[rare_bench_cv] {int((carrier_den == 0).sum())} muestras con "
                         f"{B_CARRIER_DEN}==0 -> carrier_density seria NaN (baseline M22 lo prohibe)")
    B = np.column_stack([density, carrier_num / carrier_den])  # [rare_density, carrier_density] (orden M22)
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise SystemExit("[rare_bench_cv] NaN/inf en features A/B (Q/sexo/burden) tras alinear -> abortado")
    C = np.hstack([A, B])
    n_dense_C = C.shape[1]

    # Entradas combinadas para E (C denso ++ matriz sparse) como una sola matriz sparse.
    # A/B/C se pasan DENSOS (5-7 columnas, no ganan sparsity) para que StandardScaler pueda centrar
    # (with_mean=True); la version sparse rompe StandardScaler ("Cannot center sparse matrices").
    E_input = sp.hstack([sp.csr_matrix(C), X], format="csr")

    meta = {
        "n_train": int(X.shape[0]), "n_variants_input": int(X.shape[1]),
        "outer_folds": sorted(int(x) for x in np.unique(folds)),
        "prevalence_train": round(float(y.mean()), 6),
    }
    return {
        "X": X, "y": y, "folds": folds, "groups": groups,
        "A": A, "B": B, "C": C, "E_input": E_input, "n_dense_C": n_dense_C,
        "c_grid": c_grid, "l1_ratios": l1_ratios, "en_grid": _en_grid(c_grid, l1_ratios),
        "svd_c_grid": {"clf__C": c_grid}, "ret_params": ret_params, "class_weight": class_weight,
        "scorer": args.inner_scorer, "meta": meta,
        "common": dict(outer_groups=folds, inner_group=groups, inner_splits=args.inner_splits,
                       scorer=args.inner_scorer, seed=args.seed, n_jobs=args.n_jobs,
                       threshold=args.decision_threshold, pre_dispatch=args.pre_dispatch),
    }


def _set_spec(name, args, ctx):
    """Única fuente de la definición de cada set: (estimator, grid, Xin). Reusada por monolithic/abc/
    fold, por lo que el modo particionado debe coincidir con el monolítico."""
    en = ctx["en_grid"]
    if name == SET_A:
        return _pipeline_dense_en(args.max_iter, args.tol, args.seed, ctx["class_weight"]), en, ctx["A"]
    if name == SET_B:
        return _pipeline_dense_en(args.max_iter, args.tol, args.seed, ctx["class_weight"]), en, ctx["B"]
    if name == SET_C:
        return _pipeline_dense_en(args.max_iter, args.tol, args.seed, ctx["class_weight"]), en, ctx["C"]
    if name == SET_D:
        return _pipeline_matrix_en(ctx["ret_params"], args.max_iter, args.tol, args.seed, ctx["class_weight"]), en, ctx["X"]
    if name == SET_E:
        e_en = Pipeline([("comb", _combined_transformer_en(ctx["n_dense_C"], ctx["ret_params"])),
                         ("clf", _elasticnet_clf(args.max_iter, args.tol, args.seed, ctx["class_weight"]))])
        return e_en, en, ctx["E_input"]
    raise SystemExit(f"[rare_bench_cv] set desconocido para el modo particionado: {name}")


# ---------------------------------------------------------------------------------------------------
# Construcción del reporte final compartida por los modos monolithic y aggregate.
# ---------------------------------------------------------------------------------------------------
def build_report(results, args, meta, elapsed, partition_meta=None):
    """Arma el resumen, los contrastes y el plan de cómputo a partir de `results`.

    Los modos monolithic y aggregate usan el mismo esquema. Solo cambian los datos operativos de
    tiempo y la procedencia de la partición.
    """
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]
    l1_ratios = [float(x) for x in args.l1_ratios.split(",") if x.strip()]
    n_variants = meta["n_variants_input"]

    summary = {name: {"family": r["family"],
                      "balanced_accuracy": _agg(r["per_fold"], "balanced_accuracy"),
                      "auroc": _agg(r["per_fold"], "auroc"),
                      "recall_pos": _agg(r["per_fold"], "recall_pos"),
                      "recall_neg": _agg(r["per_fold"], "recall_neg"),
                      "retained_rare_cols_per_fold": [f["n_retained_rare_cols"] for f in r["per_fold"]],
                      "chosen_params_per_fold": [f["chosen_params"] for f in r["per_fold"]]}
               for name, r in results.items()}

    C_pf = results[SET_C]["per_fold"]
    A_pf = results[SET_A]["per_fold"]
    contrasts = {
        "PRIMARY_delta_balacc_E_minus_C": _paired_delta(results[SET_E]["per_fold"], C_pf),
        "PRIMARY_delta_auroc_E_minus_C": _paired_delta(results[SET_E]["per_fold"], C_pf, key="auroc"),
        "secondary_delta_balacc_E_minus_A": _paired_delta(results[SET_E]["per_fold"], A_pf),
        "secondary_delta_balacc_D_minus_A": _paired_delta(results[SET_D]["per_fold"], A_pf),
    }

    # auditoria anti-leakage: las columnas raras retenidas DEBEN diferir entre outer folds.
    d_ret = summary[SET_D]["retained_rare_cols_per_fold"]
    anti_leakage_ok = len(set(x for x in d_ret if x is not None)) > 1

    # Plan de cómputo y estimación de duración.
    n_outer = len(meta["outer_folds"])
    grid_en = len(c_grid) * len(l1_ratios)
    grid_svd = len(c_grid)
    fits_per_en_set = n_outer * (args.inner_splits * grid_en + 1)
    fits_per_svd_set = n_outer * (args.inner_splits * grid_svd + 1)
    n_en_sets, n_svd_sets = 5, (2 if args.run_svd else 0)
    n_fits_total = n_en_sets * fits_per_en_set + n_svd_sets * fits_per_svd_set
    SMOKE_FIT_S, SMOKE_COLS = 17.3, 394363
    per_heavy_fit_s = round(SMOKE_FIT_S * n_variants / SMOKE_COLS, 1)
    n_heavy_fits = 2 * fits_per_en_set
    serial_hours = round(n_heavy_fits * per_heavy_fit_s / 3600.0, 2)
    parallel_hours_lb = round(serial_hours / max(1, args.n_jobs), 2)
    compute_plan = {
        "n_model_fits_total": int(n_fits_total),
        "fits_per_elasticnet_set": int(fits_per_en_set),
        "elasticnet_sets": ["A_Q_sex", "B_burden", "C_Q_sex_burden", "D_matrix", "E_full"],
        "svd_enabled": bool(args.run_svd),
        "fits_per_svd_set": int(fits_per_svd_set) if args.run_svd else 0,
        "formula": "n_outer * (inner_splits * grid + 1) por set; grid_en=%d, grid_svd=%d, inner=%d, outer=%d"
                   % (grid_en, grid_svd, args.inner_splits, n_outer),
        "time_estimate": {
            "anchor": "smoke chr22: 1 fit SAGA = 17.3 s con 394363 columnas retenidas",
            "per_heavy_fit_seconds_est": per_heavy_fit_s,
            "n_heavy_fits": int(n_heavy_fits),
            "serial_hours_est": serial_hours,
            "parallel_hours_lower_bound_est": parallel_hours_lb,
            "note": "Estimación O(columnas), no medida. La cota inferior supone paralelismo perfecto "
                    "con n_jobs. Un C alto o una matriz menos dispersa pueden aumentar el tiempo; el "
                    "valor observado queda en elapsed_seconds.",
        },
    }

    report = {
        "module": "23_RARE_MATRIX_BENCHMARK", "stage": "cv (benchmark cientifico)",
        "sklearn_version": sklearn.__version__, "seed": args.seed,
        "n_train": meta["n_train"], "n_variants_input": n_variants,
        "outer_folds": meta["outer_folds"],
        "test_fold_excluded": args.test_fold,
        "prevalence_train": meta["prevalence_train"],
        "burden_definition_M22": {"features": B_NAMES,
                                  "carrier_density": f"{B_CARRIER_NUM} / {B_CARRIER_DEN} (derivado, no crudo)",
                                  "note": "identico a bin/model_primary_cv.py; conteos crudos no son predictores"},
        "class_weight": args.class_weight,
        "svd_run": bool(args.run_svd),
        "compute_plan": compute_plan,
        "cv_design": {
            "outer": "LeaveOneGroupOut sobre fold congelado {0,1,2,4}",
            "inner": f"StratifiedGroupKFold(n_splits={args.inner_splits}, shuffle=True, seed={args.seed}), "
                     f"estratifica por y, agrupa por {args.group_col}",
            "grid": {"C": c_grid, "l1_ratio": l1_ratios}, "inner_scorer": args.inner_scorer,
            "decision_threshold": args.decision_threshold,
            "svd_components": args.svd_components if args.run_svd else None,
        },
        "retention_rule_fold_fitted": {
            "min_mac_train": args.min_mac_train, "min_alt_carriers_train": args.min_alt_carriers_train,
            "min_variance_train": args.min_variance_train,
            "note": "Se vuelve a calcular dentro del conjunto de entrenamiento de cada fold "
                    "(RareRetention). El missingness usa un prefiltro global sobre TRAIN, casi inerte "
                    "e independiente de la etiqueta."},
        "anti_leakage_retained_cols_differ_across_folds": bool(anti_leakage_ok),
        "per_set_summary": summary,
        "contrasts": contrasts,
        "framing": ("Recuperabilidad técnica de la etiqueta interna de M14. Delta>0 indica concordancia, no "
                    "descubrimiento; por DPI-en-muestra-finita puede ser <=0. No controla fuga de "
                    "parentesco sub-0.0442, confound de batch ni colapso de la minoria NAM; "
                    "la lectura se limita a recuperabilidad, sin afirmar subestructura."),
        # Las métricas del modelo se mantienen separadas de los datos de ejecución.
        "operational": {
            "elapsed_seconds": elapsed,
            "mean_seconds_per_fit_actual": round(elapsed / n_fits_total, 3) if n_fits_total else None,
            "execution_mode": "aggregate" if partition_meta else "monolithic",
            "partition_provenance": partition_meta,
        },
    }
    return report


def _write_and_echo(args, report, extra_echo=None):
    _common.write_json(args.outdir / "rare_bench_cv_results.json", report)
    contrasts = report["contrasts"]
    summary = report["per_set_summary"]
    echo = {"PRIMARY_delta_balacc_E_minus_C": contrasts["PRIMARY_delta_balacc_E_minus_C"],
            "C_balacc": summary[SET_C]["balanced_accuracy"],
            "E_balacc": summary[SET_E]["balanced_accuracy"],
            "n_model_fits_total": report["compute_plan"]["n_model_fits_total"],
            "execution_mode": report["operational"]["execution_mode"],
            "anti_leakage_ok": report["anti_leakage_retained_cols_differ_across_folds"],
            "elapsed_seconds": report["operational"]["elapsed_seconds"]}
    if extra_echo:
        echo.update(extra_echo)
    print(json.dumps(echo, indent=2))


# ---------------------------------------------------------------------------------------------------
# Huella compartida por preflight y por las tareas.
# ---------------------------------------------------------------------------------------------------
def _science_config(args):
    """Devuelve la configuración científica usada en la huella.

    Los valores operativos como n_jobs, outdir, mode, set, fold y checkpoint quedan fuera. Un cambio
    en cualquiera de los valores incluidos invalida la reutilización de tareas.
    """
    return {
        "c_grid": args.c_grid, "l1_ratios": args.l1_ratios, "inner_splits": args.inner_splits,
        "inner_scorer": args.inner_scorer, "decision_threshold": args.decision_threshold,
        "class_weight": args.class_weight, "run_svd": bool(args.run_svd),
        "svd_components": args.svd_components, "max_iter": args.max_iter, "tol": args.tol,
        "seed": args.seed, "test_fold": args.test_fold, "train_label": args.train_label,
        "min_mac_train": args.min_mac_train, "min_alt_carriers_train": args.min_alt_carriers_train,
        "min_variance_train": args.min_variance_train,
        "sample_id_col": args.sample_id_col, "split_col": args.split_col,
        "label_col": args.label_col, "group_col": args.group_col, "fold_col": args.fold_col,
        "A_COLS": A_COLS, "B_NAMES": B_NAMES,
    }


_FP_CACHE = None


def _fingerprint(args, inherited_input_hashes=None):
    """Huella guardada en memoria durante el proceso. Las tareas abc y fold reciben
    `inherited_input_hashes` desde preflight para no volver a calcular la huella de la matriz. El
    preflight calcula una vez las huellas de las cuatro entradas. Cada proceso ejecuta un modo con
    argumentos inmutables, por lo que la caché es segura. El estado de los hilos también queda
    registrado en la huella.
    """
    global _FP_CACHE
    if _FP_CACHE is None:
        code_files = [Path(__file__).resolve(), Path(__file__).resolve().parent / "_common.py"]
        # entradas indexadas por ROL (no por nombre de archivo): estable ante renombres, y define el
        # contrato inmutable que preflight y las tareas consumen por los mismos canales.
        input_files = {"matrix": args.matrix_npz, "samples": args.samples_tsv,
                       "split_manifest": args.split_manifest, "modeling_master": args.modeling_master}
        _FP_CACHE = _common.compute_fingerprint(code_files, input_files, _science_config(args),
                                                args.container_sha256, thread_env=_THREAD_STATE,
                                                input_hashes=inherited_input_hashes)
    return _FP_CACHE


def _load_preflight_and_fingerprint(args):
    """Carga la huella de preflight para las tareas abc y fold.

    input_sha256 se hereda sin volver a leer la matriz para calcular su hash. El código, la
    configuración, el contenedor y los hilos se comprueban localmente antes de devolver la huella.
    """
    if args.preflight_json is None:
        raise SystemExit("[rare_bench_cv] --mode abc/fold requiere --preflight-json")
    pf = _common.read_json(args.preflight_json)
    ref = pf.get("fingerprint", pf)
    inherited = ref.get("input_sha256")
    if not inherited:
        raise SystemExit("[rare_bench_cv] preflight.json no contiene input_sha256")
    fp = _fingerprint(args, inherited_input_hashes=inherited)
    _common.assert_fingerprint_matches(ref, fp, context="(tarea vs preflight)")
    return fp


def _reject_svd_partitioned(args):
    """Rechaza SVD en el modo particionado, donde todavía no existen tareas para esos sets."""
    if args.run_svd:
        raise SystemExit("[rare_bench_cv] --run-svd no está disponible en modo particionado (abc/fold/"
                         "aggregate); el comparador SVD solo corre en --mode monolithic.")


# ---------------------------------------------------------------------------------------------------
# Modos de ejecucion
# ---------------------------------------------------------------------------------------------------
def run_preflight(args, ctx):
    """Valida cohortes y fold 3, y emite la huella usada por las tareas. El token
    preflight.json lo consume cada tarea como `path`."""
    fp = _fingerprint(args)
    token = {"stage": "RARE_BENCH_PREFLIGHT", "fingerprint": fp,
             "meta": ctx["meta"], "sets_expected": {"abc": ABC_SETS, "heavy": HEAVY_SETS},
             "outer_folds_required": ctx["meta"]["outer_folds"]}
    _common.write_json(args.outdir / "preflight.json", token)
    print(json.dumps({"preflight_ok": True, "fingerprint_id": fp["fingerprint_id"],
                      "meta": ctx["meta"]}, indent=2))


def run_abc(args, ctx):
    """Computa A,B,C (densos, ms) -> abc_results.json con per_fold por set."""
    _reject_svd_partitioned(args)
    fp = _load_preflight_and_fingerprint(args)
    sets = {}
    for name in ABC_SETS:
        estimator, grid, Xin = _set_spec(name, args, ctx)
        sets[name] = {"family": "elasticnet",
                      "per_fold": _fit_score_nested(estimator, grid, Xin, ctx["y"], **ctx["common"])}
    out = {"stage": "RARE_BENCH_CV_ABC", "sets": sets, "meta": ctx["meta"],
           "config": _science_config(args),
           "fingerprint_id": fp["fingerprint_id"]}
    _common.write_json(args.outdir / "abc_results.json", out)
    print(json.dumps({"abc_ok": True, "sets": list(sets)}, indent=2))


def run_fold(args, ctx):
    """Calcula un set y outer fold para D o E y escribe <set>.fold<K>.json."""
    _reject_svd_partitioned(args)
    fp = _load_preflight_and_fingerprint(args)
    if args.set_name not in HEAVY_SETS:
        raise SystemExit(f"[rare_bench_cv] --mode fold requiere --set in {HEAVY_SETS}; recibido {args.set_name!r}")
    if args.fold is None or int(args.fold) not in ctx["meta"]["outer_folds"]:
        raise SystemExit(f"[rare_bench_cv] --fold {args.fold} no esta en outer_folds {ctx['meta']['outer_folds']}")
    estimator, grid, Xin = _set_spec(args.set_name, args, ctx)
    tr, va = _split_for_held(Xin, ctx["y"], ctx["folds"], int(args.fold))
    entry = _fit_one_fold(estimator, grid, Xin, ctx["y"], int(args.fold), tr, va, ctx["groups"],
                          args.inner_splits, ctx["scorer"], args.seed, args.n_jobs,
                          args.decision_threshold, args.pre_dispatch)
    out = {"stage": "RARE_BENCH_CV_FOLD", "set": args.set_name, "held_out_fold": int(args.fold),
           "family": "elasticnet", "per_fold_entry": entry, "meta": ctx["meta"],
           "config": _science_config(args),
           "fingerprint_id": fp["fingerprint_id"]}
    _common.write_json(args.outdir / f"{args.set_name}.fold{int(args.fold)}.json", out)
    print(json.dumps({"fold_ok": True, "set": args.set_name, "fold": int(args.fold),
                      "balanced_accuracy": entry["balanced_accuracy"]}, indent=2))


_FOLD_ENTRY_FIELDS = ("held_out_fold", "n_train", "n_val", "balanced_accuracy", "auroc",
                      "recall_pos", "recall_neg", "chosen_params", "n_retained_rare_cols")


def _check(cond, msg):
    if not cond:
        raise SystemExit(f"[aggregate] esquema inválido: {msg}")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_fold_entry(e, where):
    _check(isinstance(e, dict), f"{where}: la entrada per-fold no es un objeto")
    for f in _FOLD_ENTRY_FIELDS:
        _check(f in e, f"{where}: falta el campo '{f}'")
    _check(_is_int(e["held_out_fold"]), f"{where}: held_out_fold no es int")
    for f in ("n_train", "n_val"):
        _check(_is_int(e[f]) and e[f] > 0, f"{where}: '{f}'={e[f]!r} no es int>0")
    for f in ("balanced_accuracy", "auroc", "recall_pos", "recall_neg"):
        v = e[f]
        _check(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               and 0.0 <= v <= 1.0, f"{where}: '{f}'={v!r} no es finito en [0,1]")
    _check(isinstance(e["chosen_params"], dict), f"{where}: chosen_params no es dict")
    _check(e["n_retained_rare_cols"] is None or _is_int(e["n_retained_rare_cols"]),
           f"{where}: n_retained_rare_cols no es int/None")


def _validate_aggregate_inputs(abc, fold_objs):
    """Valida la estructura de abc y de los folds antes de agregarlos."""
    _check(isinstance(abc, dict) and isinstance(abc.get("sets"), dict), "abc_results.json sin 'sets'")
    _check(abc.get("stage") == "RARE_BENCH_CV_ABC", f"abc con stage inesperado: {abc.get('stage')!r}")
    _check(set(abc["sets"]) == set(ABC_SETS),
           f"abc debe tener exactamente {ABC_SETS}; tiene {sorted(abc['sets'])}")
    _check(isinstance(abc.get("meta"), dict) and isinstance(abc["meta"].get("outer_folds"), list),
           "abc sin meta.outer_folds")
    _check("config" in abc and "fingerprint_id" in abc, "abc sin config/fingerprint_id")
    for name, r in abc["sets"].items():
        _check(isinstance(r, dict) and isinstance(r.get("per_fold"), list), f"abc[{name}] sin per_fold lista")
        _check(r.get("family") == "elasticnet", f"abc[{name}] family inesperada: {r.get('family')!r}")
        for e in r["per_fold"]:
            _validate_fold_entry(e, f"abc[{name}]")
    _check(len(fold_objs) > 0, "no se paso ningun --fold-json")
    for o in fold_objs:
        _check(isinstance(o, dict), "un --fold-json no es un objeto")
        _check(o.get("stage") == "RARE_BENCH_CV_FOLD", f"fold con stage inesperado: {o.get('stage')!r}")
        _check(o.get("set") in HEAVY_SETS, f"fold con set invalido: {o.get('set')!r} (esperado {HEAVY_SETS})")
        _check(o.get("family") == "elasticnet", f"fold {o.get('set')} family inesperada: {o.get('family')!r}")
        _check("per_fold_entry" in o, f"fold {o.get('set')} sin per_fold_entry")
        _check("config" in o and "fingerprint_id" in o, f"fold {o.get('set')} sin config/fingerprint_id")
        _check(isinstance(o.get("meta"), dict), f"fold {o.get('set')} sin meta")
        _check(_is_int(o.get("held_out_fold")), f"fold {o.get('set')}: held_out_fold no es int")
        _validate_fold_entry(o["per_fold_entry"], f"fold {o.get('set')}")
        _check(int(o["per_fold_entry"]["held_out_fold"]) == int(o["held_out_fold"]),
               f"fold {o.get('set')}: held_out_fold del envoltorio != del entry")
    # Los metadatos deben coincidir entre todas las tareas.
    for o in fold_objs:
        _check(o["meta"] == abc["meta"], f"fold {o.get('set')}: meta difiere del abc -> dataset distinto")
    # Cada set A/B/C debe cubrir los outer folds requeridos sin faltantes ni duplicados.
    required = list(abc["meta"]["outer_folds"])
    for name in ABC_SETS:
        held_abc = sorted(int(e["held_out_fold"]) for e in abc["sets"][name]["per_fold"])
        _check(held_abc == required, f"abc[{name}] folds {held_abc} != requeridos {required}")


def run_aggregate(args):
    """Valida y reúne abc y los folds en el reporte final sin cargar la matriz."""
    _reject_svd_partitioned(args)
    abc = _common.read_json(args.abc_json)
    fold_objs = [_common.read_json(p) for p in args.fold_json]
    _validate_aggregate_inputs(abc, fold_objs)

    # Todas las tareas deben compartir la misma huella de código, datos, configuración y entorno.
    fps = {abc.get("fingerprint_id")} | {o.get("fingerprint_id") for o in fold_objs}
    if len(fps) != 1 or None in fps:
        raise SystemExit(f"[aggregate] fingerprint_id inconsistente entre tareas: {fps}")
    # La configuración de aggregate debe coincidir con la de las tareas porque
    # build_report deriva la metadata cientifica (test_fold, cv_design, retencion, class_weight) de los
    # args de ESTA invocacion. Si difieren, el reporte mostraria metadata falsa aunque los numeros
    # per-fold (leidos de los JSON) sean correctos. El `config` guardado por cada tarea es la verdad.
    agg_config = _science_config(args)
    for obj, tag in [(abc, "abc")] + [(o, o.get("set")) for o in fold_objs]:
        if obj.get("config") != agg_config:
            raise SystemExit(f"[aggregate] la config de --mode aggregate difiere de la tarea {tag} -> "
                             "la metadata del reporte sería incorrecta.")
    meta = abc["meta"]
    required = list(meta["outer_folds"])

    results = {name: {"family": abc["sets"][name]["family"], "per_fold": abc["sets"][name]["per_fold"]}
               for name in ABC_SETS}
    for name in HEAVY_SETS:
        entries = [o for o in fold_objs if o["set"] == name]
        held = sorted(int(o["held_out_fold"]) for o in entries)
        if held != required:
            raise SystemExit(f"[aggregate] set {name}: folds {held} != requeridos {required}; "
                             "faltan o sobran tareas por fold")
        # reensamblado en orden sorted(held) -> secuencia identica al monolitico (bit-exactitud de _agg).
        by_held = {int(o["held_out_fold"]): o["per_fold_entry"] for o in entries}
        results[name] = {"family": entries[0]["family"], "per_fold": [by_held[h] for h in required]}

    loaded = {name: [f["held_out_fold"] for f in results[name]["per_fold"]] for name in results}
    partition_meta = {"execution": "partitioned_nextflow",
                      "fingerprint_id": fps.pop(),
                      "folds_per_set": loaded,
                      "note": "cada (set,fold) se computo como tarea Nextflow independiente; "
                              "reensamblado por sorted(held). Cientifico == monolitico; solo difieren tiempos."}
    report = build_report(results, args, meta, elapsed=0.0, partition_meta=partition_meta)
    _write_and_echo(args, report)


# ---------------------------------------------------------------------------------------------------
# Control acotado de convergencia del solver (modos refit y refit_aggregate)
# ---------------------------------------------------------------------------------------------------
# No repite el benchmark ni la búsqueda de hiperparámetros. Reajusta el modelo final de cada
# combinación de set y fold con los valores guardados en la corrida base y un techo de iteraciones
# más alto. Los sets densos A/B/C se leen de abc_results.json. El fold de test continúa excluido por
# load_context.
REFIT_STAGE = "RARE_BENCH_REFIT_FOLD"


def _science_config_delta(base_cfg, cur_cfg):
    """Devuelve las claves científicas que cambian entre las dos configuraciones."""
    keys = set(base_cfg) | set(cur_cfg)
    return sorted(k for k in keys if base_cfg.get(k) != cur_cfg.get(k))


def run_refit(args, ctx):
    """Reajusta un set y fold con los hiperparámetros de la base y un max_iter más alto."""
    _reject_svd_partitioned(args)
    fp = _load_preflight_and_fingerprint(args)
    if args.baseline_fold_json is None:
        raise SystemExit("[rare_bench_cv] --mode refit requiere --baseline-fold-json para leer los "
                         "hiperparámetros elegidos en la corrida base")
    if args.set_name not in HEAVY_SETS:
        raise SystemExit(f"[rare_bench_cv] --mode refit requiere --set in {HEAVY_SETS}; "
                         f"recibido {args.set_name!r}")
    if args.fold is None or int(args.fold) not in ctx["meta"]["outer_folds"]:
        raise SystemExit(f"[rare_bench_cv] --fold {args.fold} no esta en outer_folds "
                         f"{ctx['meta']['outer_folds']}")
    base = _common.read_json(args.baseline_fold_json)
    _check(base.get("stage") == "RARE_BENCH_CV_FOLD",
           f"baseline con stage inesperado: {base.get('stage')!r}")
    _check(base.get("set") == args.set_name,
           f"baseline es del set {base.get('set')!r}, se pidio {args.set_name!r}")
    _check(_is_int(base.get("held_out_fold")) and int(base["held_out_fold"]) == int(args.fold),
           f"baseline es del fold {base.get('held_out_fold')!r}, se pidio {args.fold!r}")
    _check(isinstance(base.get("config"), dict), "baseline sin config")
    _validate_fold_entry(base["per_fold_entry"], "baseline")

    chosen = base["per_fold_entry"]["chosen_params"]
    if not chosen:
        raise SystemExit("[rare_bench_cv] la corrida base tiene chosen_params vacío; no hay un modelo "
                         "final para reajustar")
    # También se compara el contenido de las entradas. _science_config() solo cubre valores como
    # columnas, umbrales y semilla, mientras que fingerprint_id cambia al aumentar max_iter.
    # Sin esta comprobación, un split_manifest regenerado con el mismo esquema podría comparar grupos
    # de entrenamiento y validación distintos.
    if args.baseline_preflight_json is None:
        raise SystemExit("[rare_bench_cv] --mode refit requiere --baseline-preflight-json para "
                         "comprobar las entradas usadas por la corrida base")
    base_pf = _common.read_json(args.baseline_preflight_json)
    base_inputs = base_pf.get("fingerprint", base_pf).get("input_sha256")
    if not base_inputs:
        raise SystemExit("[rare_bench_cv] el preflight de la corrida base no contiene input_sha256")
    cur_inputs = fp.get("input_sha256")
    if not cur_inputs:
        raise SystemExit("[rare_bench_cv] la huella de esta tarea no contiene input_sha256")
    if cur_inputs != base_inputs:
        changed = sorted(k for k in set(base_inputs) | set(cur_inputs)
                         if base_inputs.get(k) != cur_inputs.get(k))
        raise SystemExit(f"[rare_bench_cv] las entradas cambiaron frente a la corrida base en {changed}; "
                         "el reajuste no reproduciría el mismo split ni la misma retención")

    base_cfg = base["config"]
    base_max_iter = int(base_cfg["max_iter"])
    if int(args.max_iter) <= base_max_iter:
        raise SystemExit(f"[rare_bench_cv] --max-iter {args.max_iter} debe ser mayor que el valor de la "
                         f"corrida base ({base_max_iter})")
    # Solo puede cambiar max_iter. Cualquier otra diferencia impediría comparar el resultado con la
    # corrida base y con las métricas de C que se reutilizan.
    diff = [k for k in _science_config_delta(base_cfg, _science_config(args)) if k != "max_iter"]
    if diff:
        raise SystemExit(f"[rare_bench_cv] la configuración difiere de la base en {diff}; solo puede "
                         "cambiar max_iter")

    estimator, _grid, Xin = _set_spec(args.set_name, args, ctx)
    estimator.set_params(**chosen)  # Parámetros fijos: se hace un ajuste sin una nueva selección.
    tr, va = _split_for_held(Xin, ctx["y"], ctx["folds"], int(args.fold))
    entry = _fit_one_fold(estimator, None, Xin, ctx["y"], int(args.fold), tr, va, ctx["groups"],
                          args.inner_splits, ctx["scorer"], args.seed, args.n_jobs,
                          args.decision_threshold, args.pre_dispatch)
    entry["chosen_params"] = dict(chosen)  # Se conservan los valores elegidos en la corrida base.
    out = {"stage": REFIT_STAGE, "set": args.set_name, "held_out_fold": int(args.fold),
           "family": "elasticnet", "per_fold_entry": entry,
           "baseline": {"per_fold_entry": base["per_fold_entry"],
                        "max_iter": base_max_iter,
                        "fingerprint_id": base.get("fingerprint_id"),
                        "config": base_cfg,
                        "input_sha256": base_inputs},
           "input_sha256": cur_inputs,
           "max_iter": int(args.max_iter), "meta": ctx["meta"],
           "config": _science_config(args), "fingerprint_id": fp["fingerprint_id"]}
    _common.write_json(args.outdir / f"{args.set_name}.fold{int(args.fold)}.refit.json", out)
    print(json.dumps({"refit_ok": True, "set": args.set_name, "fold": int(args.fold),
                      "n_iter": entry["n_iter"], "converged": entry["converged"],
                      "max_iter": int(args.max_iter),
                      "balanced_accuracy": entry["balanced_accuracy"],
                      "balanced_accuracy_baseline": base["per_fold_entry"]["balanced_accuracy"],
                      "fit_seconds": entry["fit_seconds"]}, indent=2))


def _validate_refit_inputs(abc, refit_objs, args):
    """Valida las entradas de refit_aggregate.

    El abc que aporta las métricas de C debe compartir la configuración de la corrida base incluida
    en cada reajuste. Solo se permite que cambie max_iter para mantener comparable el contraste
    entre cada set y C.
    """
    _check(isinstance(abc, dict) and isinstance(abc.get("sets"), dict), "abc_results.json sin 'sets'")
    _check(abc.get("stage") == "RARE_BENCH_CV_ABC", f"abc con stage inesperado: {abc.get('stage')!r}")
    _check(SET_C in abc["sets"], f"abc sin el set de referencia {SET_C}")
    _check(isinstance(abc.get("meta"), dict) and isinstance(abc["meta"].get("outer_folds"), list),
           "abc sin meta.outer_folds")
    for e in abc["sets"][SET_C]["per_fold"]:
        _validate_fold_entry(e, f"abc[{SET_C}]")
    _check(len(refit_objs) > 0, "no se paso ningun --refit-json")
    cur_cfg = _science_config(args)
    for o in refit_objs:
        tag = f"{o.get('set')}.fold{o.get('held_out_fold')}"
        _check(o.get("stage") == REFIT_STAGE, f"{tag}: stage inesperado {o.get('stage')!r}")
        _check(o.get("set") in HEAVY_SETS, f"{tag}: set invalido {o.get('set')!r}")
        _check(_is_int(o.get("held_out_fold")), f"{tag}: held_out_fold no es int")
        _check(isinstance(o.get("baseline"), dict), f"{tag}: sin bloque baseline")
        _validate_fold_entry(o["per_fold_entry"], tag)
        _validate_fold_entry(o["baseline"]["per_fold_entry"], f"{tag}/baseline")
        _check(int(o["per_fold_entry"]["held_out_fold"]) == int(o["held_out_fold"]),
               f"{tag}: held_out_fold del envoltorio != del entry")
        _check(int(o["baseline"]["per_fold_entry"]["held_out_fold"]) == int(o["held_out_fold"]),
               f"{tag}: held_out_fold del baseline != del refit")
        _check(o["meta"] == abc["meta"], f"{tag}: meta difiere del abc -> dataset distinto")
        _check(o["baseline"]["config"] == abc.get("config"),
               f"{tag}: la config del baseline difiere de la del abc -> las metricas de C que se "
               "reutilizan no son comparables con este baseline")
        _check(o["baseline"]["fingerprint_id"] == abc.get("fingerprint_id"),
               f"{tag}: fingerprint del baseline != del abc -> no son la misma corrida base")
        _check(o.get("config") == cur_cfg,
               f"{tag}: la config de la tarea difiere de la de --mode refit_aggregate")
        only = [k for k in _science_config_delta(o["baseline"]["config"], cur_cfg) if k != "max_iter"]
        _check(not only, f"{tag}: el refit difiere de la base en {only} ademas de max_iter")
        # El techo debe aumentar para describir el resultado como un reajuste con más iteraciones.
        _check(int(o["max_iter"]) > int(o["baseline"]["max_iter"]),
               f"{tag}: max_iter del refit ({o['max_iter']}) <= de la base ({o['baseline']['max_iter']}) "
               "-> no es un control de convergencia")
        _check(int(o["config"]["max_iter"]) == int(o["max_iter"]),
               f"{tag}: config.max_iter ({o['config']['max_iter']}) != max_iter declarado ({o['max_iter']})")
        # Las entradas también deben coincidir; comparar solo la configuración podría mezclar
        # resultados calculados sobre datos distintos.
        _check(o.get("input_sha256") and o["input_sha256"] == o["baseline"].get("input_sha256"),
               f"{tag}: input_sha256 del reajuste no coincide con la corrida base")
    fps = {o.get("fingerprint_id") for o in refit_objs}
    _check(len(fps) == 1 and None not in fps, f"fingerprint_id inconsistente entre refits: {fps}")
    inputs = {json.dumps(o.get("input_sha256"), sort_keys=True) for o in refit_objs}
    _check(len(inputs) == 1, "los refits no comparten input_sha256 -> vienen de datos distintos")


def _refit_verdict(per_set, material_delta):
    """Aplica el criterio de cierre definido en la configuración.

    Primero se comprueba la convergencia. El signo y la magnitud del cambio solo se evalúan cuando
    todos los modelos convergieron.
    """
    flat = [f for s in per_set.values() for f in s["per_fold"]]
    not_conv = [f"{s}.fold{f['held_out_fold']}" for s, v in per_set.items()
                for f in v["per_fold"] if f["converged"] is False]
    unknown = [f"{s}.fold{f['held_out_fold']}" for s, v in per_set.items()
               for f in v["per_fold"] if f["converged"] is None]
    max_change = max((abs(f["delta_balacc_vs_baseline"]) for f in flat), default=0.0)
    all_neg = all(f["delta_balacc_vs_C"] < 0 for f in flat)
    if unknown:
        return {"verdict": "INDETERMINADO_SIN_n_iter",
                "action": "el solver no expuso n_iter en " + ", ".join(unknown) + " -> no se puede "
                          "juzgar convergencia; revisar antes de concluir",
                "not_converged": not_conv, "n_iter_unknown": unknown,
                "max_abs_delta_balacc_vs_baseline": round(max_change, 6), "all_delta_vs_C_negative": all_neg}
    if not_conv:
        return {"verdict": "SIGUE_SIN_CONVERGER",
                "action": "reportar la trayectoria y la estabilidad antes de subir el techo otra vez; "
                          "no cerrar con un modelo que sigue alcanzando max_iter",
                "not_converged": not_conv, "n_iter_unknown": [],
                "max_abs_delta_balacc_vs_baseline": round(max_change, 6), "all_delta_vs_C_negative": all_neg}
    if all_neg and max_change < material_delta:
        return {"verdict": "CIERRE_CON_CAVEAT",
                "action": f"el delta frente a C es negativo en todos los folds y el cambio máximo de "
                          f"balanced accuracy ({max_change:.6f}) es menor que el umbral material "
                          f"({material_delta}). La estabilidad del ajuste final no cambia el resultado: "
                          "la matriz rara cruda no muestra utilidad incremental sobre C en el pipeline "
                          "Elastic Net preespecificado de este benchmark. Esta conclusión se limita a "
                          "esa ruta y no se extiende a la familia Elastic Net ni a la representación "
                          "diádica.",
                "scope_of_closure": ("sin utilidad incremental bajo el pipeline Elastic Net "
                                     "preespecificado; no es un veredicto sobre Elastic Net como familia "
                                     "ni sobre la señal diádica de las raras"),
                "not_converged": [], "n_iter_unknown": [],
                "max_abs_delta_balacc_vs_baseline": round(max_change, 6), "all_delta_vs_C_negative": all_neg}
    return {"verdict": "DETENERSE_Y_REVISAR",
            "action": ("hay un cambio material o un cambio de signo; no cerrar ni repetir una selección "
                       "sin revisar primero el resultado"),
            "not_converged": [], "n_iter_unknown": [],
            "max_abs_delta_balacc_vs_baseline": round(max_change, 6), "all_delta_vs_C_negative": all_neg}


def run_refit_aggregate(args):
    """Reúne convergencia, métricas, tiempo y memoria por fold.

    También calcula el cambio frente a C usando las métricas de la corrida base y el cambio frente al
    resultado original. No vuelve a cargar la matriz.
    """
    _reject_svd_partitioned(args)
    abc = _common.read_json(args.abc_json)
    refit_objs = [_common.read_json(p) for p in args.refit_json]
    _validate_refit_inputs(abc, refit_objs, args)

    c_per_fold = abc["sets"][SET_C]["per_fold"]
    c_bal = {int(e["held_out_fold"]): e["balanced_accuracy"] for e in c_per_fold}
    required = list(abc["meta"]["outer_folds"])
    per_set = {}
    for name in sorted({o["set"] for o in refit_objs}):
        objs = sorted((o for o in refit_objs if o["set"] == name), key=lambda o: int(o["held_out_fold"]))
        held = [int(o["held_out_fold"]) for o in objs]
        if held != required:
            raise SystemExit(f"[refit_aggregate] set {name}: folds {held} != requeridos {required}; "
                             "faltan o sobran tareas por fold")
        rows = []
        for o in objs:
            h = int(o["held_out_fold"])
            r, b = o["per_fold_entry"], o["baseline"]["per_fold_entry"]
            if h not in c_bal:
                raise SystemExit(f"[refit_aggregate] {name}.fold{h}: el abc no trae {SET_C} para ese fold")
            rows.append({
                "held_out_fold": h,
                "chosen_params": r["chosen_params"],
                "max_iter_baseline": int(o["baseline"]["max_iter"]),
                "max_iter_refit": int(o["max_iter"]),
                "n_iter": r["n_iter"], "converged": r["converged"],
                "balanced_accuracy": r["balanced_accuracy"],
                "balanced_accuracy_baseline": b["balanced_accuracy"],
                "delta_balacc_vs_baseline": round(r["balanced_accuracy"] - b["balanced_accuracy"], 6),
                "auroc": r["auroc"], "auroc_baseline": b["auroc"],
                "delta_auroc_vs_baseline": round(r["auroc"] - b["auroc"], 6),
                "recall_pos": r["recall_pos"], "recall_neg": r["recall_neg"],
                "balanced_accuracy_C": c_bal[h],
                "delta_balacc_vs_C": round(r["balanced_accuracy"] - c_bal[h], 6),
                "delta_balacc_vs_C_baseline": round(b["balanced_accuracy"] - c_bal[h], 6),
                "n_retained_rare_cols": r["n_retained_rare_cols"],
                "fit_seconds": r["fit_seconds"],
                "peak_rss_self_gb": r["peak_rss_self_gb"],
            })
        c_entries = [e for e in c_per_fold if int(e["held_out_fold"]) in required]
        per_set[name] = {
            "per_fold": rows,
            "balanced_accuracy": _agg(rows, "balanced_accuracy"),
            "balanced_accuracy_baseline": _agg(rows, "balanced_accuracy_baseline"),
            "delta_balacc_vs_baseline": _agg(rows, "delta_balacc_vs_baseline"),
            "delta_balacc_vs_C": _paired_delta(rows, c_entries, "balanced_accuracy"),
            "delta_auroc_vs_C": _paired_delta(rows, c_entries, "auroc"),
            "fit_seconds": _agg(rows, "fit_seconds"),
        }

    verdict = _refit_verdict(per_set, float(args.refit_material_delta))
    report = {
        "module": "23_RARE_MATRIX_BENCHMARK",
        "stage": "refit (control acotado de convergencia)",
        "sklearn_version": sklearn.__version__,
        "seed": args.seed,
        "meta": abc["meta"],
        "test_fold_excluded": args.test_fold,
        "scope": ("reajuste del modelo final por (set,fold) con los hiperparámetros seleccionados en "
                  "la corrida base y un max_iter más alto. No se repite la grilla ni se hace una "
                  f"selección nueva, y el fold de test queda excluido. Las métricas de {SET_C} se "
                  "leen del abc_results.json de la base."),
        "max_iter": {"baseline": sorted({int(o["baseline"]["max_iter"]) for o in refit_objs}),
                     "refit": sorted({int(o["max_iter"]) for o in refit_objs})},
        "material_delta_balacc": float(args.refit_material_delta),
        "closure_criterion": ("Criterio definido antes de la ejecución. (1) Si algún modelo sigue sin "
                              "converger, se reporta antes de subir el techo. (2) Si el delta frente a C "
                              "sigue negativo en los cuatro folds y el cambio de balanced accuracy es "
                              "menor que el umbral material, se cierra con la salvedad y el alcance de "
                              "esta ruta. (3) En cualquier otro caso, se detiene para revisión."),
        "closure_scope_declared": ("un cierre de este control significa 'sin utilidad incremental bajo el "
                                   "pipeline Elastic Net preespecificado', no 'la familia Elastic Net "
                                   "queda refutada' ni 'las raras no tienen señal'. El estimador es "
                                   "marginal por variante y no representa la interacción diádica que "
                                   "motiva el encoder, por lo que no permite sacar conclusiones sobre "
                                   "ella. La evaluación que puede descartar la matriz queda fuera de "
                                   "M14 y de este benchmark."),
        "per_set": per_set,
        "reference_C": {"set": SET_C, "balanced_accuracy": _agg(c_per_fold, "balanced_accuracy"),
                        "source": "abc_results.json de la corrida base (reutilizado, no recomputado)"},
        "closure": verdict,
        "config": _science_config(args),
        "baseline_fingerprint_id": abc.get("fingerprint_id"),
        "refit_fingerprint_id": refit_objs[0].get("fingerprint_id"),
    }
    _common.write_json(args.outdir / "rare_bench_refit_results.json", report)
    print(json.dumps({"refit_aggregate_ok": True, "closure": verdict,
                      "per_set_delta_vs_C": {k: v["delta_balacc_vs_C"] for k, v in per_set.items()},
                      "per_set_delta_vs_baseline": {k: v["delta_balacc_vs_baseline"]
                                                    for k, v in per_set.items()}}, indent=2))


def run_monolithic(args, ctx):
    """Corrida de un tiron (REFERENCIA). Todos los sets/folds -> rare_bench_cv_results.json. No usa
    preflight (es la referencia standalone que hashea sus propias entradas al cargar)."""
    results = {}
    t0 = time.perf_counter()
    for name in ABC_SETS + HEAVY_SETS:
        estimator, grid, Xin = _set_spec(name, args, ctx)
        results[name] = {"family": "elasticnet",
                         "per_fold": _fit_score_nested(estimator, grid, Xin, ctx["y"], **ctx["common"])}
    # Comparador secundario TruncatedSVD con logística para D y E, activado mediante --run-svd.
    if args.run_svd:
        d_svd = _pipeline_matrix_svd(ctx["ret_params"], args.svd_components, args.max_iter, args.tol,
                                     args.seed, ctx["class_weight"])
        results["D_matrix_svd_logistic"] = {"family": "svd_logistic",
            "per_fold": _fit_score_nested(d_svd, ctx["svd_c_grid"], ctx["X"], ctx["y"], **ctx["common"])}
        e_svd = Pipeline([("comb", _combined_transformer_svd(ctx["n_dense_C"], ctx["ret_params"], args.svd_components, args.seed)),
                          ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=args.max_iter,
                                                     tol=args.tol, random_state=args.seed, class_weight=ctx["class_weight"]))])
        results["E_full_svd_logistic"] = {"family": "svd_logistic",
            "per_fold": _fit_score_nested(e_svd, ctx["svd_c_grid"], ctx["E_input"], ctx["y"], **ctx["common"])}
    elapsed = round(time.perf_counter() - t0, 1)
    report = build_report(results, args, ctx["meta"], elapsed, partition_meta=None)
    _write_and_echo(args, report)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-npz", type=Path)
    ap.add_argument("--samples-tsv", type=Path)
    ap.add_argument("--split-manifest", type=Path)
    ap.add_argument("--modeling-master", type=Path)
    ap.add_argument("--sample-id-col", default="sample_id")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--label-col", default="y")
    ap.add_argument("--fold-col", default="fold")
    ap.add_argument("--group-col", default="split_group_key")
    ap.add_argument("--train-label", default="TRAIN")
    ap.add_argument("--test-fold", type=int, default=3)
    ap.add_argument("--min-mac-train", type=int, default=2)
    ap.add_argument("--min-alt-carriers-train", type=int, default=2)
    ap.add_argument("--min-variance-train", type=float, default=0.0)
    ap.add_argument("--c-grid", default="1e-3,1e-2,1e-1,1")
    ap.add_argument("--l1-ratios", default="0.1,0.5,0.9")
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--inner-scorer", default="balanced_accuracy")
    ap.add_argument("--decision-threshold", type=float, default=0.5)
    ap.add_argument("--svd-components", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8,
                    help="workers de loky para GridSearchCV. Cada worker mantiene copias/intermedios de los "
                         "datos del fold; usar 8 reduce el pico de RAM del set E. Este parámetro no entra "
                         "en la huella científica (_science_config).")
    ap.add_argument("--pre-dispatch", default="8",
                    help="pre_dispatch de GridSearchCV: tareas (y copias/intermedios por worker) que loky "
                         "pre-despacha a la vez. Acota el pico de RAM del paralelismo (int o expr tipo "
                         "'2*n_jobs'). Es un valor operativo y no entra en la huella científica.")
    ap.add_argument("--class-weight", default="balanced",
                    help="class_weight de la logistica; 'balanced' (default, prev~0.25) o 'none'")
    ap.add_argument("--run-svd", action="store_true",
                    help="activa el comparador secundario TruncatedSVD+logistica (sets D,E). "
                         "Está desactivado por defecto.")
    ap.add_argument("--outdir", required=True, type=Path)
    # Modos de ejecución particionada.
    ap.add_argument("--mode", default="monolithic",
                    choices=["monolithic", "preflight", "abc", "fold", "aggregate",
                             "refit", "refit_aggregate"],
                    help="monolithic (referencia) | preflight | abc | fold (--set --fold) | aggregate | "
                         "refit (control de convergencia de un modelo final) | refit_aggregate")
    ap.add_argument("--set", dest="set_name", default=None, help="set para --mode fold (D/E)")
    ap.add_argument("--fold", type=int, default=None, help="outer-fold para --mode fold")
    ap.add_argument("--container-sha256", default=None,
                    help="sha256 del contenedor usado en la huella de preflight y las tareas")
    ap.add_argument("--preflight-json", type=Path, default=None,
                    help="token de preflight que cada tarea compara con su propia huella")
    ap.add_argument("--abc-json", type=Path, default=None, help="abc_results.json para --mode aggregate")
    ap.add_argument("--fold-json", action="append", default=[], type=Path,
                    help="cada <set>.fold<K>.json para --mode aggregate (repetible)")
    # Control acotado de convergencia.
    ap.add_argument("--baseline-fold-json", type=Path, default=None,
                    help="<set>.fold<K>.json de la corrida base para --mode refit. De ahí se leen los "
                         "hiperparametros ganadores (no se transcriben) y la config a comparar.")
    ap.add_argument("--baseline-preflight-json", type=Path, default=None,
                    help="preflight.json de la corrida base para --mode refit. Su input_sha256 comprueba "
                         "que se usaron las mismas entradas (matriz/split/modeling_master); sin "
                         "el, un archivo regenerado con el mismo esquema pasaria desapercibido.")
    ap.add_argument("--refit-json", action="append", default=[], type=Path,
                    help="cada <set>.fold<K>.refit.json para --mode refit_aggregate (repetible)")
    ap.add_argument("--refit-material-delta", type=float, default=0.01,
                    help="umbral de cambio material en balanced accuracy del criterio de cierre "
                         "predeclarado. Operacional-de-decision: no entra en la huella cientifica "
                         "porque no altera ningun ajuste, solo etiqueta el veredicto.")
    return ap


# Código de salida reservado para un worker de loky terminado por falta de memoria.
# La configuración de Nextflow lo reconoce como reintentable; otras señales no se remapean.
EXIT_OOM_WORKER = 42


def _is_oom_worker_kill(exc):
    """Indica si loky informa un SIGKILL asociado a falta de memoria.

    Otras señales se conservan como errores no reintentables.
    """
    return isinstance(exc, TerminatedWorkerError) and "SIGKILL(-9)" in str(exc)


def _run(args):
    if args.mode in ("aggregate", "refit_aggregate"):
        # No cargan la matriz: solo reensamblan JSONs. Barato y portable.
        if args.abc_json is None:
            raise SystemExit(f"[rare_bench_cv] --mode {args.mode} requiere --abc-json")
        if args.mode == "aggregate":
            run_aggregate(args)
        else:
            if not args.refit_json:
                raise SystemExit("[rare_bench_cv] --mode refit_aggregate requiere al menos un --refit-json")
            run_refit_aggregate(args)
        return

    # Todos los demas modos requieren la matriz + metadatos.
    for req in ("matrix_npz", "samples_tsv", "split_manifest", "modeling_master"):
        if getattr(args, req) is None:
            raise SystemExit(f"[rare_bench_cv] --mode {args.mode} requiere --{req.replace('_','-')}")
    ctx = load_context(args)  # El fold 3 se valida antes de cualquier cálculo.

    dispatch = {"preflight": run_preflight, "abc": run_abc, "fold": run_fold,
                "monolithic": run_monolithic, "refit": run_refit}
    dispatch[args.mode](args, ctx)


def main():
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    try:
        _run(args)
    except TerminatedWorkerError as exc:
        # loky solo mata un worker con SIGKILL(-9) cuando el kernel lo OOM-killea (o alguien lo mata a
        # mano) -> fallo de RECURSOS transitorio: remapear a un exit dedicado reintentable. Cualquier
        # otra terminacion del worker (SIGSEGV -11 = bug de codigo/datos, etc.) se re-lanza y sale
        # Los errores científicos terminan con código 1.
        if _is_oom_worker_kill(exc):
            print(f"[rare_bench_cv] worker loky OOM-killed (SIGKILL -9) -> exit {EXIT_OOM_WORKER} "
                  f"(reintentable; sube memoria en el 2o intento)", file=sys.stderr)
            sys.exit(EXIT_OOM_WORKER)
        raise


if __name__ == "__main__":
    main()
