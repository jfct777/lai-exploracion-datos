#!/usr/bin/env python3
"""SPLIT_POLICY -- construye el split train/test PRIMARIO, reproducible y congelable, a partir de
`modeling_master.tsv` (P0_DATASET=COMPLETO). NO selecciona features ni entrena nada.

El diseño aplica las siguientes reglas:
  - Manifiesto con las 2619 filas. Las 6 QC-rojas: eligible=false, split=EXCLUDE, fold=NA,
    exclusion_reason=qc_red. Las 2613 elegibles se particionan.
  - Etiqueta y: y=1 si community_res_1 >= 0 (pertenece a comunidad Leiden); y=0 si
    community_res_1 == -1 (ruido de Leiden) O falta (aislado sin nodo en el grafo).
    y0_subtype distingue {community, leiden_noise, isolated} para trazabilidad.
  - Grupo (anti-fuga de familias): componentes conexas PC-Relate phi>=0.0442. La etiqueta de grupo
    se CANONICALIZA a min(sample_id) del componente ANTES de particionar -> invariante al orden de
    generacion de PC-Relate (sklearn ordena los grupos por su etiqueta para la asignacion greedy).
  - Particionado: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) genera 5 folds.
  - SELECCION del fold de TEST (regla DECLARADA de antemano, usa SOLO metadatos -- etiqueta,
    cohorte, region -- NUNCA features raras ni resultados de modelos): se elige el fold cuya
    distribucion es MAS REPRESENTATIVA del total elegible, minimizando el desbalance conjunto
      joint = |prev_fold - prev_overall| + TVD_cohorte + TVD_region
    (TVD = distancia de variacion total). En empate, el fold de menor numero. Los otros 4 = TRAIN.

Fuga ACEPTADA declarada (no es cero): el agrupamiento a phi>=0.0442 garantiza que ningun par con
phi>=0.0442 cruza el split, pero el 4.º grado (phi>=0.0221) y la co-descendencia founder-difusa
(que phi de SNPs comunes no ve) PUEDEN quedar a ambos lados. Limite explicito, no ausencia de fuga.

Modo --verify: re-deriva el split desde el mismo modeling_master.tsv y afirma que el sha256 del
manifiesto coincide con el congelado, columna por columna (re-derivabilidad bit a bit).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

COLS_OUT = ["sample_id", "eligible", "y", "y0_subtype", "split_group_key",
            "fold", "split", "exclusion_reason", "cohort", "region"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_y(community_res_1):
    """y=1 si pertenece a comunidad (community_res_1 >= 0); y=0 si -1 (ruido) o faltante (aislado)."""
    if community_res_1 == "" or pd.isna(community_res_1):
        return 0
    return 0 if float(community_res_1) < 0 else 1


def _y0_subtype(y, community_res_1):
    if y == 1:
        return "community"
    if community_res_1 == "" or pd.isna(community_res_1):
        return "isolated"
    return "leiden_noise"


def _tvd(p, q):
    """Distancia de variacion total entre dos distribuciones categoricas (dicts categoria->prop)."""
    cats = set(p) | set(q)
    return 0.5 * sum(abs(p.get(c, 0.0) - q.get(c, 0.0)) for c in cats)


def _dist(series):
    vc = series.value_counts(normalize=True)
    return {str(k): float(v) for k, v in vc.items()}


def build_split(master_path, red_samples, n_splits, seed, group_col, community_col):
    dtype = {"sample_id": str, group_col: str, community_col: str}
    master = pd.read_csv(master_path, sep="\t", dtype=dtype)
    master[community_col] = master[community_col].fillna("")
    red = set(s.strip() for s in red_samples.split(",")) if red_samples else set()

    # fuente UNICA de elegibilidad = columna qc_red del master; --red-samples es un GUARD.
    n_red_found = int(master["sample_id"].isin(red).sum())
    assert n_red_found == len(red), f"{len(red) - n_red_found} IDs de --red-samples no estan en el master"
    qc_red_ids = set(master.loc[master["qc_red"].astype(str) == "True", "sample_id"])
    assert qc_red_ids == red, (
        f"columna qc_red ({sorted(qc_red_ids)}) != --red-samples ({sorted(red)}) -- fuente divergente")

    master["y"] = master[community_col].map(_label_y).astype(int)
    master["y0_subtype"] = [_y0_subtype(y, c) for y, c in zip(master["y"], master[community_col])]

    elig = master[master["qc_red"].astype(str) != "True"].copy()
    assert len(elig) == len(master) - len(red), "elegibles != master - rojas"

    # clave de grupo CANONICA = min(sample_id) del componente phi0442, sobre el universo elegible.
    elig["split_group_key"] = elig.groupby(group_col)["sample_id"].transform("min")
    elig = elig.sort_values("sample_id", kind="mergesort").reset_index(drop=True)

    y = elig["y"].to_numpy()
    groups = elig["split_group_key"].to_numpy()
    X = np.zeros((len(elig), 1))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.full(len(elig), -1, dtype=int)
    for fold_idx, (_train, test) in enumerate(sgkf.split(X, y, groups)):
        fold_of[test] = fold_idx
    assert (fold_of >= 0).all(), "toda muestra elegible debe caer en un fold"
    elig["fold"] = fold_of
    return master, elig


def select_test_fold(elig, n_splits):
    """Regla DECLARADA: el fold de TEST es el mas representativo del total elegible -- minimiza
    joint = |prev_fold-prev_overall| + TVD_cohorte + TVD_region. Empate -> menor numero de fold.
    Usa SOLO etiqueta/cohorte/region (metadatos), nunca features raras ni resultados de modelos."""
    prev_all = float((elig["y"] == 1).mean())
    coh_all = _dist(elig["cohort"])
    reg_all = _dist(elig["region"])
    per_fold = []
    for f in range(n_splits):
        sub = elig[elig["fold"] == f]
        prev_f = float((sub["y"] == 1).mean())
        d_label = abs(prev_f - prev_all)
        d_cohort = _tvd(_dist(sub["cohort"]), coh_all)
        d_region = _tvd(_dist(sub["region"]), reg_all)
        per_fold.append({
            "fold": f, "n": int(len(sub)),
            "n_pos": int((sub["y"] == 1).sum()), "prevalence_y1": round(prev_f, 4),
            "d_label_prevalence": round(d_label, 4),
            "tvd_cohort": round(d_cohort, 4), "tvd_region": round(d_region, 4),
            "joint_imbalance": round(d_label + d_cohort + d_region, 4),
        })
    chosen = min(range(n_splits), key=lambda f: (per_fold[f]["joint_imbalance"], f))
    return chosen, per_fold, {"prevalence_y1": round(prev_all, 4)}


def assemble_manifest(master, elig, chosen_fold):
    elig = elig.copy()
    elig["eligible"] = True
    elig["exclusion_reason"] = ""
    elig["split"] = np.where(elig["fold"] == chosen_fold, "TEST", "TRAIN")
    elig["fold"] = elig["fold"].astype(int).astype(str)

    excl = master[master["qc_red"].astype(str) == "True"].copy()
    excl["eligible"] = False
    excl["exclusion_reason"] = "qc_red"
    excl["split"] = "EXCLUDE"
    excl["fold"] = ""                # NA: las excluidas no se particionan
    excl["split_group_key"] = ""     # NA: no participan del agrupamiento elegible

    full = pd.concat([elig[COLS_OUT], excl[COLS_OUT]], ignore_index=True)
    full = full.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return full


def audit(master, elig, full, chosen_fold, per_fold, overall, master_path,
          n_splits, seed, group_col, community_col):
    # etiqueta split final por fold (TEST/TRAIN) sobre las representaciones ya elegidas
    per_fold_out = []
    for r in per_fold:
        r = dict(r)
        r["role"] = "TEST" if r["fold"] == chosen_fold else "TRAIN"
        per_fold_out.append(r)
    g_folds = elig.groupby("split_group_key")["fold"].nunique()
    crossed = int((g_folds > 1).sum())
    y0 = elig[elig["y"] == 0]
    isolated_matches_flag = None
    if "flag_aislado" in master.columns:
        iso_ids = set(elig.loc[elig["y0_subtype"] == "isolated", "sample_id"])
        flag_ids = set(master.loc[master["flag_aislado"].astype(str).isin(["True", "1", "1.0"]), "sample_id"])
        isolated_matches_flag = bool(iso_ids == (flag_ids & set(elig["sample_id"])))
    chosen = per_fold_out[chosen_fold]
    # bloque familiar mas grande, derivado de los datos (NO hardcodear -- CERO invencion)
    grp_sizes = elig.groupby("split_group_key").size()
    big_gid = grp_sizes.idxmax()
    big = elig[elig["split_group_key"] == big_gid]
    big_n = int(len(big))
    big_cohort = str(big["cohort"].mode().iloc[0]) if "cohort" in big else "?"
    big_region = str(big["region"].mode().iloc[0]) if "region" in big else "?"
    big_fold = int(big["fold"].iloc[0]) if big["fold"].nunique() == 1 else -1
    rep = {
        "n_total": int(len(full)),
        "n_eligible": int(len(elig)),
        "n_excluded": int((full["eligible"] == False).sum()),
        "exclusion_reason_counts": {k: int(v) for k, v in full.loc[full["eligible"] == False, "exclusion_reason"].value_counts().items()},
        "n_pos_total": int((elig["y"] == 1).sum()),
        "n_neg_total": int((elig["y"] == 0).sum()),
        "prevalence_y1_overall": overall["prevalence_y1"],
        "n_groups": int(elig["split_group_key"].nunique()),
        "n_singleton_groups": int((elig.groupby("split_group_key").size() == 1).sum()),
        "largest_group_size": int(elig.groupby("split_group_key").size().max()),
        "crossed_groups_between_folds": crossed,
        "y0_subtype_counts": {k: int(v) for k, v in y0["y0_subtype"].value_counts().items()},
        "isolated_matches_flag_aislado": isolated_matches_flag,
        "fold_selection": {
            "declared_rule": ("TEST = fold que minimiza joint = |prev-prev_overall| + TVD_cohorte + "
                              "TVD_region vs el total elegible; empate -> menor numero de fold. Solo "
                              "metadatos (etiqueta/cohorte/region), nunca features raras ni modelos."),
            "chosen_test_fold": chosen_fold,
            "per_fold": per_fold_out,
            "largest_family_block": {"size": big_n, "modal_cohort": big_cohort,
                                     "modal_region": big_region, "in_fold": big_fold,
                                     "in_test": big_fold == chosen_fold},
            "limitation": (
                "El fold de TEST elegido tiene tvd_cohort=%.4f y tvd_region=%.4f; el residual proviene "
                "de bloques familiares indivisibles (el mayor, n=%d, modal cohort=%s / region=%s, cae "
                "entero en el fold %d por la garantia anti-fuga). No se corrige con un metodo nuevo: es "
                "el costo estructural de no partir familias." % (
                    chosen["tvd_cohort"], chosen["tvd_region"], big_n, big_cohort, big_region, big_fold)),
        },
        "per_fold": {str(r["fold"]): {"n": r["n"], "n_pos": r["n_pos"],
                                      "prevalence_y1": r["prevalence_y1"], "role": r["role"]}
                     for r in per_fold_out},
        "test_fold": chosen_fold,
        "cohort_by_split": {role: {k: int(v) for k, v in full[full["split"] == role]["cohort"].value_counts().items()}
                            for role in ("TRAIN", "TEST", "EXCLUDE")},
        "region_by_split": {role: {k: int(v) for k, v in full[full["split"] == role]["region"].value_counts().items()}
                            for role in ("TRAIN", "TEST", "EXCLUDE")},
        "freeze": {
            "n_splits": n_splits, "random_state": seed, "shuffle": True,
            "group_column": group_col,
            "group_label_canonicalized_as": "min(sample_id) del componente",
            "test_fold_selection": "representatividad declarada (metadatos), no fold fijo",
            "label_definition": "y=1 si %s>=0; y=0 si ==-1 o faltante (aislado)" % community_col,
            "input_row_order": "sort by sample_id (mergesort)",
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "modeling_master_sha256": _sha256(master_path),
        },
        "accepted_leakage_limit": (
            "phi>=0.0442 garantiza 0 pares cruzando; 4.º grado (phi>=0.0221) y co-descendencia "
            "founder-difusa PUEDEN cruzar -> fuga aceptada, no cero. Sensibilidad (reagrupar a "
            "phi>=0.0221, medir dAUC) diferida al modelado."),
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, type=Path)
    ap.add_argument("--red-samples", required=True, help="6 QC-rojas, separadas por comas")
    ap.add_argument("--group-col", default="kinship_group_id_phi0442")
    ap.add_argument("--community-col", default="community_res_1")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    master, elig = build_split(args.master, args.red_samples, args.n_splits, args.seed,
                               args.group_col, args.community_col)
    chosen_fold, per_fold, overall = select_test_fold(elig, args.n_splits)
    full = assemble_manifest(master, elig, chosen_fold)
    manifest_path = args.outdir / "split_manifest.tsv"
    audit_path = args.outdir / "split_manifest_audit.json"

    if args.verify:
        old_audit = json.load(open(audit_path))
        master_sha_now = _sha256(args.master)
        master_sha_frozen = old_audit["freeze"]["modeling_master_sha256"]
        tmp_path = args.outdir / "_verify_reds.tsv"
        full.to_csv(tmp_path, sep="\t", index=False)
        new_sha = _sha256(tmp_path)
        frozen = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
        rederived = pd.read_csv(tmp_path, sep="\t", dtype=str, keep_default_na=False)
        tmp_path.unlink()
        assert len(frozen) == len(rederived), f"cardinalidad: {len(frozen)} vs {len(rederived)}"
        merged = frozen.merge(rederived, on="sample_id", suffixes=("_frozen", "_new"), validate="one_to_one")
        assert len(merged) == len(frozen), "el merge perdio/agrego filas"
        col_mismatch = {c: int((merged[f"{c}_frozen"] != merged[f"{c}_new"]).sum())
                        for c in COLS_OUT if c != "sample_id"}
        old_sha = old_audit["frozen_sha256_split_manifest"]
        ok = (master_sha_now == master_sha_frozen) and (new_sha == old_sha) and \
             all(v == 0 for v in col_mismatch.values())
        print(json.dumps({"verify": "PASS" if ok else "FAIL",
                          "master_sha256_match": master_sha_now == master_sha_frozen,
                          "rederived_sha256_match": new_sha == old_sha,
                          "column_mismatches": col_mismatch}, indent=2))
        raise SystemExit(0 if ok else 1)

    full.to_csv(manifest_path, sep="\t", index=False)
    rep = audit(master, elig, full, chosen_fold, per_fold, overall, args.master,
                args.n_splits, args.seed, args.group_col, args.community_col)
    rep["frozen_sha256_split_manifest"] = _sha256(manifest_path)
    with open(audit_path, "w") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)

    label_md = """# Definicion de etiqueta y del split -- congelado

**Manifiesto:** las %(n_total)d filas canonicas. Las %(n_excl)d QC-rojas van con `eligible=false`,
`split=EXCLUDE`, `fold` vacio (NA), `exclusion_reason=qc_red`. Las %(n_elig)d elegibles se particionan.

**Etiqueta y (binaria):**
- `y=1` si `%(cc)s` >= 0 -> pertenece a una comunidad Leiden detectada.
- `y=0` si `%(cc)s` == -1 (ruido de Leiden) O faltante (aislado sin nodo en el grafo).

**`y0_subtype`** (trazabilidad de la clase negativa): `leiden_noise` (community == -1),
`isolated` (community faltante; coincide con `flag_aislado`), `community` (para y=1).

**Grupo anti-fuga:** componentes conexas PC-Relate phi>=0.0442 (`%(gc)s`), etiqueta canonicalizada
a `min(sample_id)` del componente -> split funcion determinista de la particion, no del orden de
generacion de PC-Relate.

**Particionado y seleccion del TEST:** `StratifiedGroupKFold(n_splits=%(ns)d, shuffle=True,
random_state=%(seed)d)` genera 5 folds. El fold de TEST se elige por una **regla declarada de
antemano** que usa SOLO metadatos (etiqueta, cohorte, region -- nunca features raras ni resultados
de modelos): el fold mas representativo del total elegible, que minimiza
`joint = |prev_fold - prev_overall| + TVD_cohorte + TVD_region`; en empate, el fold de menor numero.
El fold elegido y las metricas por fold estan en `split_manifest_audit.json` (`fold_selection`).

**Fuga ACEPTADA (limite explicito, NO cero):** phi>=0.0442 garantiza 0 pares cruzando; el 4.º grado
(phi>=0.0221) y la co-descendencia founder-difusa PUEDEN quedar a ambos lados. Sensibilidad diferida
al modelado. No afirmar "sin fuga".

**Limitacion declarada:** el desbalance residual de cohorte/region del TEST proviene de bloques
familiares indivisibles (el mayor, n=119, amazonico, cae entero en un fold por la garantia
anti-fuga). Se documenta, no se corrige con un metodo nuevo. Ver `fold_selection.limitation`.

**Re-derivabilidad:** correr este script con `--verify` en un container con las mismas versiones
(sklearn/numpy en el bloque `freeze`) -> debe dar `verify: PASS` (sha256 y todas las columnas).
""" % {"cc": args.community_col, "gc": args.group_col, "ns": args.n_splits, "seed": args.seed,
       "n_total": rep["n_total"], "n_excl": rep["n_excluded"], "n_elig": rep["n_eligible"]}
    (args.outdir / "label_definition.md").write_text(label_md)

    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
