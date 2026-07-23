#!/usr/bin/env python3
"""Fase primaria de modelado -- validacion interna sobre TRAIN (TEST cerrado). NO abre el fold de
TEST, NO corre sensibilidades de parentesco, NO usa modelos complejos.

Este análisis mide la recuperabilidad técnica de una etiqueta interna a M14
(comunidad Leiden), CONDICIONAL a la ancestria global Q. NO es descubrimiento biologico -- la
etiqueta deriva del grafo M14 (grado del grafo -> etiqueta, AUC~0.99 tautologico) y el burden
marginal de raras es termometro de ancestria (rare_density corr ~0.99 con Q_AFR). Un balanced-acc
alto es concordancia esperada, no señal de subestructura rara ortogonal.

Diseno:
  - Universo: TRAIN = split=='TRAIN' (2091). TEST (fold del manifiesto marcado TEST) NUNCA se carga
    para entrenar/seleccionar. Validacion interna = LeaveOneGroupOut sobre la columna `fold` (los 4
    folds congelados restantes); los folds ya respetan grupos de parentesco (0 cruzados).
  - Roles de columna (ver README/print): etiqueta y; features raras principales = burden
    density-normalizado; covariables comunes/control = Q + sex; prohibidas = features de grafo M14
    (circulares), flag_aislado (fuga de y), cohort/region/state (fuga de diseño), community_res_*/
    assignment_confidence (etiqueta), missingness/callability (proxy de batch).
  - Sets nested para el incremental "¿rara aporta sobre comun?": A=comun (Q+sex), B=rara,
    C=comun+rara.
  - Modelos: LogReg(l2, balanced, C=1), RF(100, balanced), KNN(5). Defaults fijos declarados.
  - Metricas por fold: balanced_accuracy (ARBITRO primario), roc_auc, sensibilidad, especificidad.
  - Seleccion (pre-declarada): mejor (modelo x set) por balanced_accuracy media; si dos caen dentro
    de 1 sd de la diferencia pareada por fold -> parsimonia (LogReg > KNN > RF).
  - Null ancestry-stratified para el candidato: permutar y dentro de cuartiles de Q_AFR.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

SEED = 42
THRESHOLD = 0.5  # umbral de decision congelado (fuente unica, importado por evaluate_test.py)
COMMON = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR", "sex"]
RARE = ["rare_density", "carrier_density"]  # burden density-normalizado (de-confundido de cobertura)
PARSIMONY = {"logreg": 0, "knn": 1, "rf": 2}  # menor = mas simple (desempate)


class VarianceSafeScaler(StandardScaler):
    """StandardScaler que no produce NaN si una feature es constante en un fold (var=0)."""
    def transform(self, X):
        Xt = super().transform(X)
        return np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)


def make_pipeline(name):
    if name == "logreg":
        # L2 es el default en sklearn 1.8 (penalty='l2' quedó deprecado); C=1.0 fija la regularización.
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
        return Pipeline([("scaler", VarianceSafeScaler()), ("clf", clf)])
    if name == "knn":
        return Pipeline([("scaler", VarianceSafeScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])
    if name == "rf":
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)
        return Pipeline([("clf", clf)])  # RF no necesita escalado
    raise ValueError(name)


def _sens_spec(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def cv_evaluate(X, y, folds, model_name):
    """LeaveOneGroupOut sobre `folds`. Devuelve metricas por fold (lista) para un modelo/feature-set."""
    logo = LeaveOneGroupOut()
    per_fold = []
    for tr, te in logo.split(X, y, groups=folds):
        pipe = make_pipeline(model_name)
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        sens, spec = _sens_spec(y[te], pred)
        per_fold.append({
            "fold": int(folds[te][0]),
            "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
            "roc_auc": float(roc_auc_score(y[te], proba)),
            "sensitivity": float(sens), "specificity": float(spec),
        })
    return per_fold


def summarize(per_fold):
    ba = np.array([r["balanced_accuracy"] for r in per_fold])
    return {
        "balanced_accuracy_mean": round(float(ba.mean()), 4),
        "balanced_accuracy_sd": round(float(ba.std(ddof=1)), 4),
        "roc_auc_mean": round(float(np.mean([r["roc_auc"] for r in per_fold])), 4),
        "sensitivity_mean": round(float(np.nanmean([r["sensitivity"] for r in per_fold])), 4),
        "specificity_mean": round(float(np.nanmean([r["specificity"] for r in per_fold])), 4),
        "per_fold": per_fold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, type=Path)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--n-perm", type=int, default=200, help="permutaciones del null ancestry-stratified")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    mm = pd.read_csv(args.master, sep="\t", dtype={"sample_id": str})
    sp = pd.read_csv(args.split, sep="\t", dtype={"sample_id": str})
    df = sp.merge(mm, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_mm"))

    train = df[df["split"] == "TRAIN"].copy()
    assert len(train) == 2091, f"TRAIN esperado 2091, {len(train)}"
    # re-confirmar cero grupos de parentesco cruzados entre folds dentro de TRAIN
    crossed = int((train.groupby("split_group_key")["fold"].nunique() > 1).sum())
    assert crossed == 0, f"{crossed} grupos de parentesco cruzan folds en TRAIN -- fuga"

    # GUARDIAS de completitud (transparencia: nada de imputacion silenciosa). El feature store marca
    # flag_missing_Q/density; si un Q_* o el denominador de burden viniera NaN, rompe el null
    # estratificado (qcut manda el NaN a un estrato que nunca se permuta) y VarianceSafeScaler lo
    # imputaria a 0 sin declararlo. Empiricamente TRAIN esta completo; el assert lo hace explicito.
    assert train[COMMON + ["rare_density", "rare_carrier_site_count", "rare_gt_nonmissing_sites"]].notna().all().all(), (
        "Q_*/sex/burden con NaN en TRAIN -- imputar o excluir explicitamente antes de modelar")
    assert (train["rare_gt_nonmissing_sites"].astype(float) > 0).all(), (
        "rare_gt_nonmissing_sites==0 en TRAIN -- carrier_density seria NaN")

    # features derivadas (density-normalizado de-confundido de cobertura)
    train["carrier_density"] = train["rare_carrier_site_count"].astype(float) / train["rare_gt_nonmissing_sites"].astype(float)
    for c in COMMON + ["rare_density"]:
        train[c] = train[c].astype(float)
    y = train["y"].astype(int).to_numpy()
    folds = train["fold"].astype(int).to_numpy()

    feature_sets = {"A_common": COMMON, "B_rare": RARE, "C_combined": COMMON + RARE}
    models = ["logreg", "rf", "knn"]

    results = {}
    for sname, cols in feature_sets.items():
        X = train[cols].to_numpy(dtype=float)
        for m in models:
            results[f"{sname}::{m}"] = {"feature_set": sname, "model": m, "features": cols,
                                       **summarize(cv_evaluate(X, y, folds, m))}

    # --- seleccion del candidato sobre el set C (rara+comun), regla pre-declarada ---
    cand_keys = [k for k in results if results[k]["feature_set"] == "C_combined"]
    best = max(cand_keys, key=lambda k: results[k]["balanced_accuracy_mean"])
    best_ba = results[best]["balanced_accuracy_mean"]
    # empate: dentro de 1 sd de la DIFERENCIA PAREADA por fold -> parsimonia
    def paired_sd(k1, k2):
        d = np.array([a["balanced_accuracy"] - b["balanced_accuracy"]
                      for a, b in zip(results[k1]["per_fold"], results[k2]["per_fold"])])
        return float(d.std(ddof=1))
    tied = [k for k in cand_keys
            if best_ba - results[k]["balanced_accuracy_mean"] <= paired_sd(best, k)]
    candidate = min(tied, key=lambda k: PARSIMONY[results[k]["model"]])
    cand_model = results[candidate]["model"]

    # --- incremental C vs A (¿rara aporta sobre comun?) para el modelo candidato, pareado por fold ---
    kC, kA, kB = f"C_combined::{cand_model}", f"A_common::{cand_model}", f"B_rare::{cand_model}"
    dCA = np.array([c["balanced_accuracy"] - a["balanced_accuracy"]
                    for c, a in zip(results[kC]["per_fold"], results[kA]["per_fold"])])
    incremental = {"delta_balanced_accuracy_C_minus_A_mean": round(float(dCA.mean()), 4),
                   "delta_sd": round(float(dCA.std(ddof=1)), 4),
                   "common_only_ba": results[kA]["balanced_accuracy_mean"],
                   "rare_only_ba": results[kB]["balanced_accuracy_mean"],
                   "combined_ba": results[kC]["balanced_accuracy_mean"]}

    # --- null ancestry-stratified para el candidato en set C: permutar y dentro de cuartiles Q_AFR ---
    rng = np.random.RandomState(SEED)
    qafr = train["Q_AFR"].to_numpy()
    strata = pd.qcut(qafr, 4, labels=False, duplicates="drop")
    Xc = train[feature_sets["C_combined"]].to_numpy(dtype=float)
    obs = np.mean([r["balanced_accuracy"] for r in cv_evaluate(Xc, y, folds, cand_model)])
    null_bas = []
    for _ in range(args.n_perm):
        yp = y.copy()
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            yp[idx] = rng.permutation(yp[idx])
        null_bas.append(np.mean([r["balanced_accuracy"] for r in cv_evaluate(Xc, yp, folds, cand_model)]))
    null_bas = np.array(null_bas)
    p_val = float((np.sum(null_bas >= obs) + 1) / (args.n_perm + 1))

    out = {
        "estimand": ("asociacion de burden raro (density-normalizado) con la pertenencia a comunidad "
                     "Leiden CONDICIONAL a la ancestria global Q. Etiqueta interna a M14 -> mide "
                     "concordancia/recuperabilidad tecnica, NO descubrimiento biologico."),
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "n_train": int(len(train)),
        "cv": "LeaveOneGroupOut sobre folds congelados %s" % sorted(set(folds.tolist())),
        "crossed_kinship_groups_in_train": crossed,
        "column_roles": {
            "label": "y (comunidad Leiden res 1)",
            "rare_principal": RARE + ["(carrier_density = rare_carrier_site_count/rare_gt_nonmissing_sites)"],
            "common_control": COMMON,
            "excluded_technical_batch_proxy": ["rare_missing_sites", "rare_gt_nonmissing_sites",
                                               "flag_missing_Q", "flag_missing_density"],
            "prohibited_circular_or_leak": ["n_sharing_partners", "n_segments_involved", "total_shared_bp",
                "n_chromosomes_with_sharing", "grado_M14", "grado_M14_ponderado", "flag_aislado",
                "cohort", "region", "state", "community_res_*", "assignment_confidence", "qc_*"],
            "identifiers": ["sample_id", "split_group_key", "kinship_group_id_*", "duplicate_or_MZ_group",
                            "fold", "split", "eligible", "exclusion_reason", "y0_subtype"],
        },
        "selection_rule": ("arbitro = balanced_accuracy media sobre set C_combined; empate dentro de "
                           "1 sd de la diferencia pareada por fold -> parsimonia LogReg>KNN>RF"),
        "candidate": {"key": candidate, "model": cand_model,
                      "balanced_accuracy_mean": results[candidate]["balanced_accuracy_mean"],
                      "roc_auc_mean": results[candidate]["roc_auc_mean"]},
        "incremental_rare_over_common": incremental,
        "ancestry_stratified_null": {"n_perm": args.n_perm, "observed_ba": round(float(obs), 4),
            "null_ba_mean": round(float(null_bas.mean()), 4),
            "null_ba_p95": round(float(np.percentile(null_bas, 95)), 4),
            "p_value": round(p_val, 4),
            "note": ("null permuta y DENTRO de cuartiles de Q_AFR (1-D). NO aisla el burden raro: al "
                     "randomizar y dentro de estratos de Q_AFR quedan intactos Q_EUR/NAM/EAS/sex como "
                     "predictores, y el baseline comun A (0.727) YA supera el null (~0.614). El p-valor "
                     "NO es evidencia de senal rara; el unico test que aisla la rara es el incremental "
                     "C-A (Delta 0.051).")},
        "results": results,
        "declared_limitations": [
            "Etiqueta interna a M14 (grado->etiqueta AUC~0.99): concordancia, no descubrimiento.",
            "rare_density corr ~0.99 con Q_AFR: un positivo es consistente con confound de ancestria.",
            "Baseline comun = Q 4-D (no burden comun completo): 'rara>comun' sub-potenciado hacia rara.",
            "Señal fina real de raras es DIADICA (co-sharing), no el burden marginal 1-D usado aqui.",
            "El null ancestry-stratified es 1-D (solo Q_AFR): NO aisla el burden raro de Q_EUR/NAM/EAS; "
            "su p-valor NO es evidencia de senal rara (el baseline comun A solo ya supera el null).",
            "El ORIGEN del incremento Delta bal-acc (C sobre A) NO esta identificado con este diseño: "
            "NO es atribuible a subestructura rara, y NO se puede separar de ancestria ni de artefacto "
            "tecnico. No atribuir a una causa concreta.",
            "TEST cerrado: la evaluacion unica en TEST requiere autorizacion separada.",
        ],
    }
    with open(args.outdir / "model_primary_cv_results.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ["candidate", "incremental_rare_over_common",
                                          "ancestry_stratified_null", "crossed_kinship_groups_in_train"]},
                     indent=2, ensure_ascii=False))
    # tabla compacta
    print("\n=== balanced_accuracy media (sd) por set x modelo ===")
    for sname in feature_sets:
        row = " | ".join(f"{m}:{results[f'{sname}::{m}']['balanced_accuracy_mean']:.3f}"
                         f"({results[f'{sname}::{m}']['balanced_accuracy_sd']:.3f})" for m in models)
        print(f"  {sname:12s} {row}")


if __name__ == "__main__":
    main()
