#!/usr/bin/env python3
"""Evaluación única del fold de test del pipeline candidato de regresión logística.

Reglas verificadas mediante asserts: no se reajusta ni selecciona ningún parámetro con el fold de test. Los modelos,
features, preprocesamiento, hiperparámetros y umbral se importan congelados de `model_primary_cv.py`
(commit 491f097) no se vuelven a especificar aquí para evitar divergencias. El entrenamiento usa TRAIN
(2091) y se evalua TEST (fold 3, 522) exactamente una vez por feature-set.

Contraste principal: Delta balanced_accuracy (combinado C - comun A) con IC95% por bootstrap sobre
las filas de TEST. Los modelos permanecen fijos y la incertidumbre viene del muestreo de TEST. Se
reporta bootstrap a nivel individuo y a nivel grupo-de-parentesco (mas honesto ante correlacion).

Este análisis evalúa recuperabilidad técnica condicional a ancestría, no descubrimiento. Ver las
declared_limitations del JSON de validacion interna.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Importa el pipeline, los roles y el umbral congelados en 491f097.
from model_primary_cv import COMMON, RARE, make_pipeline, _sens_spec, SEED, THRESHOLD

N_BOOT = 2000


def _add_carrier_density(df):
    df = df.copy()
    df["carrier_density"] = df["rare_carrier_site_count"].astype(float) / df["rare_gt_nonmissing_sites"].astype(float)
    for c in COMMON + ["rare_density"]:
        df[c] = df[c].astype(float)
    return df


def _metrics(y, proba):
    pred = (proba >= THRESHOLD).astype(int)  # umbral congelado (importado)
    sens, spec = _sens_spec(y, pred)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "sensitivity": float(sens), "specificity": float(spec)}


def _boot_ci(delta_fn, n, rng, groups=None):
    vals = []
    if groups is None:
        for _ in range(N_BOOT):
            idx = rng.randint(0, n, size=n)
            vals.append(delta_fn(idx))
    else:
        uniq = np.unique(groups)
        gmembers = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(N_BOOT):
            gsel = uniq[rng.randint(0, len(uniq), size=len(uniq))]
            idx = np.concatenate([gmembers[g] for g in gsel])
            vals.append(delta_fn(idx))
    vals = np.array(vals)
    return {"mean": round(float(vals.mean()), 4),
            "ci95_low": round(float(np.percentile(vals, 2.5)), 4),
            "ci95_high": round(float(np.percentile(vals, 97.5)), 4)}


def main():
    """Entrena en TRAIN y realiza la evaluación congelada del fold de test."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--split-audit", required=True, type=Path,
                    help="split_manifest_audit.json -- para verificar las precondiciones congeladas")
    ap.add_argument("--cv-results", required=True, type=Path,
                    help="model_primary_cv_results.json -- ancla el candidato; prohibe re-seleccionar con TEST")
    ap.add_argument("--candidate-model", default="logreg")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="re-correr pese a existir resultados (deja rastro)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Evita repetir la evaluación si ya existe un resultado, salvo que se use --force.
    res_path = args.outdir / "evaluate_test_results.json"
    assert args.force or not res_path.exists(), (
        f"{res_path} ya existe; usa --force solo si hay una justificación para repetir la evaluación")

    mm = pd.read_csv(args.master, sep="\t", dtype={"sample_id": str})
    sp = pd.read_csv(args.split, sep="\t", dtype={"sample_id": str})
    audit = json.load(open(args.split_audit))
    cv = json.load(open(args.cv_results))

    # El modelo debe coincidir con el candidato declarado por la CV interna.
    assert args.candidate_model == cv["candidate"]["model"], (
        f"--candidate-model={args.candidate_model} != candidato pre-declarado en CV "
        f"({cv['candidate']['model']}) -- prohibido re-seleccionar modelo con TEST")

    # Comprueba las precondiciones congeladas antes de leer el fold de test.
    assert audit["n_total"] == 2619 and audit["n_excluded"] == 6, "manifiesto != 2619 filas / 6 EXCLUDE"
    assert audit["fold_selection"]["chosen_test_fold"] == 3, "el TEST congelado no es el fold 3"
    df = sp.merge(mm, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_mm"))
    train = _add_carrier_density(df[df["split"] == "TRAIN"])
    test = _add_carrier_density(df[df["split"] == "TEST"])
    assert len(train) == 2091 and len(test) == 522, f"TRAIN/TEST = {len(train)}/{len(test)} (esperado 2091/522)"
    assert train[COMMON + ["rare_density", "carrier_density"]].notna().all().all(), "NaN en features de TRAIN"
    assert test[COMMON + ["rare_density", "carrier_density"]].notna().all().all(), "NaN en features de TEST"

    feature_sets = {"A_common": COMMON, "B_rare": RARE, "C_combined": COMMON + RARE}
    y_tr = train["y"].astype(int).to_numpy()
    y_te = test["y"].astype(int).to_numpy()

    # Se entrena en TRAIN y se evalúa una vez por set sobre el fold de test.
    test_metrics, proba_te = {}, {}
    for sname, cols in feature_sets.items():
        pipe = make_pipeline(args.candidate_model)
        pipe.fit(train[cols].to_numpy(dtype=float), y_tr)   # El ajuste usa únicamente TRAIN.
        p = pipe.predict_proba(test[cols].to_numpy(dtype=float))[:, 1]
        proba_te[sname] = p
        test_metrics[sname] = _metrics(y_te, p)

    # Contraste principal: cambio de balanced accuracy de C frente a A con IC95% por bootstrap.
    pA, pC = proba_te["A_common"], proba_te["C_combined"]
    predA, predC = (pA >= THRESHOLD).astype(int), (pC >= THRESHOLD).astype(int)

    def delta(idx):
        """Calcula el cambio de balanced accuracy entre C y A para un remuestreo."""
        return balanced_accuracy_score(y_te[idx], predC[idx]) - balanced_accuracy_score(y_te[idx], predA[idx])

    delta_point = float(balanced_accuracy_score(y_te, predC) - balanced_accuracy_score(y_te, predA))
    rng_i = np.random.RandomState(SEED)
    rng_g = np.random.RandomState(SEED)
    groups = test["split_group_key"].to_numpy()
    ci_indiv = _boot_ci(delta, len(y_te), rng_i, groups=None)
    ci_group = _boot_ci(delta, len(y_te), rng_g, groups=groups)

    out = {
        "evaluation": "Evaluación autorizada del fold 3. No se reajustaron ni seleccionaron parámetros con test.",
        "estimand": "Recuperabilidad técnica condicionada a ancestría; no es descubrimiento biológico.",
        "sklearn_version": sklearn.__version__, "seed": SEED,
        "candidate_model": args.candidate_model,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "test_prevalence_y1": round(float(y_te.mean()), 4),
        "frozen_source": "features/modelos/preproc/hiperparametros/umbral importados de model_primary_cv.py (commit 491f097)",
        "test_metrics": test_metrics,
        "primary_contrast_delta_bal_acc_C_minus_A": {
            "point": round(delta_point, 4),
            "ci95_bootstrap_individual": ci_indiv,
            "ci95_bootstrap_by_kinship_group": ci_group,
            "n_boot": N_BOOT,
            "note": ("IC por bootstrap sobre TEST con modelos fijos entrenados solo en TRAIN. El "
                     "bootstrap por grupo de parentesco es el mas honesto ante correlacion intra-familia."),
        },
        "internal_validation_reference": {"delta_bal_acc_C_minus_A": 0.0509, "delta_sd": 0.0242,
                                          "source": "model_primary_cv_results.json (commit 491f097)"},
        "declared_limitations": [
            "Etiqueta interna a M14 (grado->etiqueta AUC~0.99): concordancia, no descubrimiento.",
            "El origen del incremento Delta C-A no está identificado con este diseño. No puede "
            "atribuirse a subestructura rara ni separarse de ancestría (rare_density corr "
            "~0.99 con Q_AFR) ni de artefacto tecnico. No atribuir a una causa concreta.",
            "Baseline comun = Q 4-D (no burden comun completo): 'rara>comun' sub-potenciado hacia rara.",
            "Señal fina real de raras es DIADICA (co-sharing), no el burden marginal 1-D usado aqui.",
            "La evaluación no se repite; un segundo pase sobre TEST invalidaría su carácter held-out.",
        ],
    }
    with open(res_path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ["test_metrics", "primary_contrast_delta_bal_acc_C_minus_A",
                                          "test_prevalence_y1"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
