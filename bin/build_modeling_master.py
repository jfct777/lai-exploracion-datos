#!/usr/bin/env python3
"""P0 -- ensambla modeling_master.tsv (cohorte canonica 2619) para el experimento train/test con
features de variantes raras definidas para el análisis.

Estado: P0_DATASET=COMPLETO, SPLIT_POLICY=PENDIENTE. El dataset de modelado esta completo; NO hace
split ni entrena nada, y NO fija el umbral de parentesco primario (esa decision se difiere al diseno
de split_manifest.tsv). Construye:
  - modeling_master.tsv        : una fila por individuo canonico
  - modeling_master_dict.md    : diccionario de columnas
  - join_audit.json            : duplicados / faltantes / conteos de cada join + estado (dataset/split)
  - qc_sample_flags.tsv        : artefacto reusable de las 6 rojas (25 grises = NA, sin lista)
  - kinship_components_report.json : reporte de percolacion por umbral PC-Relate (los 4 grados)
  - kinship_x_leiden_crosstab.tsv  : cuanto de cada comunidad cae en un solo bloque de parentesco
  - duplicate_or_mz_report.json : deteccion de duplicado tecnico O gemelo MZ (PC-Relate phi>=0.354)

Fuentes (todas ya existentes en disco, ninguna se recalcula):
  feature_store.tsv, leiden_assignments.tsv, graph_nodes.tsv, dnabr.pcrelate.kin.tsv,
  metadata_cleaned.txt (region/estado)
"""
import argparse
import json
from pathlib import Path

import pandas as pd


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _components(sub_pairs, ids):
    """Componentes conexas del subgrafo `sub_pairs` (columnas ID1/ID2 ya filtradas a `ids`).

    Devuelve id_to_group (sample_id -> representante) y group_size (representante -> tamano).
    Un individuo sin ninguna arista es su propio grupo singleton.
    """
    uf = UnionFind(ids)
    for id1, id2 in zip(sub_pairs["ID1"], sub_pairs["ID2"]):
        uf.union(id1, id2)
    groups = {}
    for i in ids:
        groups.setdefault(uf.find(i), []).append(i)
    id_to_group = {i: uf.find(i) for i in ids}
    group_size = {gid: len(members) for gid, members in groups.items()}
    return id_to_group, group_size


def kinship_components(pairs, ids, threshold):
    """Componentes conexas de PC-Relate restringidas a `ids`, a un umbral de phi.

    Reporta el criterio de Molloy-Reed (1995) para aparicion de componente gigante en un
    grafo con secuencia de grados ARBITRARIA (no asume Poisson): kappa = <k^2>/<k>, kappa>2
    predice percolacion. El grado medio <k> solo (criterio Erdos-Renyi) NO es sustituto de
    kappa cuando la distribucion de grado tiene cola pesada (familias grandes = hubs), que es
    exactamente el regimen esperado en un grafo de parentesco -- corregido tras revision de la
    Se usa el criterio de Molloy-Reed; no basta con comprobar que el grado medio sea mayor que uno.
    """
    sub = pairs[(pairs["kin"] >= threshold) & pairs["ID1"].isin(ids) & pairs["ID2"].isin(ids)]
    id_to_group, group_size = _components(sub, ids)
    sizes = sorted(group_size.values(), reverse=True)
    n_with_edge = len(set(sub["ID1"]) | set(sub["ID2"]))
    nontrivial = [s for s in sizes if s > 1]
    giant = sizes[0] if sizes else 0

    degree = {i: 0 for i in ids}
    for id1, id2 in zip(sub["ID1"], sub["ID2"]):
        degree[id1] += 1
        degree[id2] += 1
    degrees = list(degree.values())
    sum_k = sum(degrees)
    sum_k2 = sum(d * d for d in degrees)
    mean_k = sum_k / len(ids)
    kappa = (sum_k2 / sum_k) if sum_k > 0 else None
    max_degree = max(degrees) if degrees else 0

    report = {
        "threshold": threshold,
        "n_pairs_used": int(len(sub)),
        "n_individuals_with_edge": int(n_with_edge),
        "mean_degree_over_cohort": round(mean_k, 4),
        "mean_k_squared_over_cohort": round(sum_k2 / len(ids), 4),
        "molloy_reed_kappa": (round(kappa, 4) if kappa is not None else None),
        "molloy_reed_percolates": (bool(kappa > 2) if kappa is not None else None),
        "max_degree": int(max_degree),
        "n_components_total": len(group_size),
        "n_components_nontrivial": len(nontrivial),
        "giant_component_size": int(giant),
        "giant_component_fraction": round(giant / len(ids), 4),
        "median_nontrivial_size": (sorted(nontrivial)[len(nontrivial) // 2] if nontrivial else None),
    }
    return report, id_to_group, group_size


def duplicate_or_mz_detection(pairs, ids, threshold):
    """Duplicado tecnico O gemelo MZ via PC-Relate: banda phi>=threshold (Manichaikul et al. 2010,
    KING: dup/MZ si kinship > 1/2^(3/2) ~= 0.354, confirmado por k2 -> 1). PC-Relate NO distingue
    una replica tecnica (mismo ADN secuenciado dos veces) de un gemelo monocigotico: ambos dan
    kinship ~0.5 y k2 ~1. Por eso se etiqueta "duplicate_or_MZ", no "replica".

    Devuelve:
      id_to_group, group_size : componentes conexas INTRA-cohorte (parejas con AMBOS extremos en `ids`).
      external : lista de (id_canonico, id_externo, kin, k2) donde solo UNO de los extremos es canonico
                 -> el duplicado/MZ del individuo vivio fuera de la cohorte (dedup aguas arriba).
    """
    hi = pairs[pairs["kin"] >= threshold]
    ina = hi["ID1"].isin(ids)
    inb = hi["ID2"].isin(ids)
    intra = hi[ina & inb]
    id_to_group, group_size = _components(intra, ids)
    ext_rows = hi[(ina | inb) & ~(ina & inb)]
    external = []
    for _, r in ext_rows.iterrows():
        canonical, other = (r["ID1"], r["ID2"]) if r["ID1"] in ids else (r["ID2"], r["ID1"])
        external.append((canonical, other, float(r["kin"]), float(r["k2"])))
    return id_to_group, group_size, external


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-store", required=True, type=Path)
    ap.add_argument("--leiden", required=True, type=Path)
    ap.add_argument("--graph-nodes", required=True, type=Path)
    ap.add_argument("--pcrelate-kin", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path,
                     help="metadata_cleaned.txt -- fuente de region/estado (el feature store no las trae)")
    ap.add_argument("--red-samples", required=True, help="lista separada por comas de las 6 rojas")
    ap.add_argument("--kinship-thresholds", default="0.0221,0.0442,0.0884,0.1770",
                     help="umbrales de phi a reportar en el JSON (4to,3ro,2do,1er grado)")
    ap.add_argument("--kinship-group-columns", default="0.0442,0.0884",
                     help="umbrales que se materializan como columnas explicitas de grupo en la tabla. "
                          "NO hay columna primaria: la eleccion se difiere al split (subset de "
                          "--kinship-thresholds)")
    ap.add_argument("--replicate-threshold", type=float, default=0.354,
                     help="phi de duplicado/MZ (Manichaikul 2010, KING band: kinship>1/2^(3/2))")
    ap.add_argument("--leiden-col-for-crosstab", default="community_res_1")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # dtype=str en las columnas de ID es obligatorio: IDs numericos como "1001" se leen como
    # int64 en unos chunks y str en otros (pandas low_memory + columna mixta) si no se fuerza,
    # lo que rompe silenciosamente cualquier .isin()/merge (int(1001) != str("1001")). Verificado
    # en este turno: sin forzar, dnabr.pcrelate.kin.tsv reporta 3096 IDs unicos en vez de 2851 reales.
    fs = pd.read_csv(args.feature_store, sep="\t", dtype={"sample_id": str})
    leid = pd.read_csv(args.leiden, sep="\t", dtype={"sample_id": str})
    graph = pd.read_csv(args.graph_nodes, sep="\t", dtype={"sample_id": str})
    kin = pd.read_csv(args.pcrelate_kin, sep="\t", dtype={"ID1": str, "ID2": str})
    md = pd.read_csv(args.metadata, sep="\t", dtype=str)

    n_fs_raw = len(fs)
    dup_fs = fs["sample_id"].duplicated().sum()
    assert dup_fs == 0, f"feature_store tiene {dup_fs} sample_id duplicados -- no se asume dedup silencioso"
    canonical_ids = set(fs["sample_id"])
    assert len(canonical_ids) == 2619, f"cohorte canonica esperada 2619, encontrada {len(canonical_ids)}"

    # --- region/estado desde metadata_cleaned: dedup del unico ID duplicado (filas identicas) ---
    md_dup_ids = md.loc[md["ID"].duplicated(keep=False), "ID"].unique().tolist()
    for dup_id in md_dup_ids:
        block = md[md["ID"] == dup_id].drop(columns=["ID"])
        # comparacion robusta a NaN: nunique(dropna=True) contaria como iguales dos filas que
        # difieren solo en un faltante (una con valor, otra con NaN) -> falso "identicas". Se
        # rellena con un centinela y se compara toda la fila contra la primera.
        filled = block.fillna("\x00__NA__\x00")
        assert bool(filled.eq(filled.iloc[0]).all().all()), (
            f"ID {dup_id} duplicado en metadata con filas NO identicas -- no se dedup silenciosamente")
    md_dedup = md.drop_duplicates("ID", keep="first")
    geo = md_dedup[["ID", "Region", "State"]].rename(
        columns={"ID": "sample_id", "Region": "region", "State": "state"})
    n_canon_missing_geo = len(canonical_ids - set(geo["sample_id"]))
    assert n_canon_missing_geo == 0, (
        f"{n_canon_missing_geo} canonicos sin region/estado -- la cohorte se definio como "
        f"VCF ∩ metadata_cleaned, cobertura debe ser total")

    # --- QC flags: reusable, solo las 6 rojas identificadas; 25 grises = NA (no hay lista) ---
    red_ids = set(s.strip() for s in args.red_samples.split(","))
    missing_red = red_ids - canonical_ids
    qc = pd.DataFrame({"sample_id": sorted(canonical_ids)})
    qc["qc_red"] = qc["sample_id"].isin(red_ids)
    qc["qc_gray"] = pd.NA
    qc["qc_gray_source_available"] = False
    qc["qc_status"] = qc["qc_red"].map({True: "red", False: "unclassified"})
    qc.to_csv(args.outdir / "qc_sample_flags.tsv", sep="\t", index=False)

    # --- percolacion PC-Relate a varios umbrales, restringida a los 2619 canonicos ---
    thresholds = [float(t) for t in args.kinship_thresholds.split(",")]
    group_col_thrs = [float(t) for t in args.kinship_group_columns.split(",")]
    for t in group_col_thrs:
        assert t in thresholds, (
            f"--kinship-group-columns {t} debe estar en --kinship-thresholds {thresholds}")
    reports = []
    id_to_group_by_thr = {}
    group_size_by_thr = {}
    for thr in thresholds:
        rep, id_to_group, group_size = kinship_components(kin, canonical_ids, thr)
        reports.append(rep)
        id_to_group_by_thr[thr] = id_to_group
        group_size_by_thr[thr] = group_size
    with open(args.outdir / "kinship_components_report.json", "w") as fh:
        json.dump(reports, fh, indent=2, ensure_ascii=False)

    # --- cruce grupo-de-parentesco x comunidad Leiden, para cada umbral reportado ---
    # Diagnostico de CIRCULARIDAD (no de fuga train/test): dice si un bloque de parentesco cae
    # entero dentro de una comunidad; la interpretacion vive en el diccionario.
    leid_map = leid.set_index("sample_id")[args.leiden_col_for_crosstab].to_dict()
    crosstab_rows = []
    for thr in thresholds:
        id_to_group = id_to_group_by_thr[thr]
        group_size = group_size_by_thr[thr]
        by_group = {}
        for sid, gid in id_to_group.items():
            comm = leid_map.get(sid)
            if pd.isna(comm) or comm is None:
                continue
            by_group.setdefault(gid, set()).add(comm)
        for gid, comms in by_group.items():
            size = group_size[gid]
            if size < 2:
                continue
            crosstab_rows.append({
                "threshold": thr,
                "group_id": str(gid),
                "group_size": size,
                "n_leiden_communities_touched": len(comms),
                "single_community": len(comms) == 1,
                "communities": ";".join(str(c) for c in sorted(comms)),
            })
    # Orden canonico explicito: sin esto el orden de filas hereda el de insercion del dict
    # (id_to_group.items() -> by_group), reproducible solo bajo inputs+entorno fijos y fragil ante
    # drift del orden de entrada o de version de libreria. Sort estable por (threshold, group_id)
    # lo fija de forma deterministica y agnostica al entorno.
    crosstab_cols = ["threshold", "group_id", "group_size",
                     "n_leiden_communities_touched", "single_community", "communities"]
    if crosstab_rows:
        crosstab_df = pd.DataFrame(crosstab_rows).sort_values(
            ["threshold", "group_id"], kind="stable"
        ).reset_index(drop=True)
    else:
        # guard: sin grupos size>=2 en ningun umbral, sort_values sobre un DataFrame sin
        # columnas lanzaria KeyError; escribe cabecera vacia deterministica (no ocurre en el
        # cohorte real — ya hay componentes no triviales a phi>=0.0442 — pero blinda el script).
        crosstab_df = pd.DataFrame(columns=crosstab_cols)
    crosstab_df.to_csv(args.outdir / "kinship_x_leiden_crosstab.tsv", sep="\t", index=False)

    # --- deteccion de duplicado tecnico O gemelo MZ via PC-Relate (PC-Relate no los distingue) ---
    rep_id_to_group, rep_group_size, rep_external = duplicate_or_mz_detection(
        kin, canonical_ids, args.replicate_threshold)
    rep_nontrivial = [s for s in rep_group_size.values() if s > 1]
    external_canonical = sorted({c for c, _, _, _ in rep_external})
    hi = kin[kin["kin"] >= args.replicate_threshold]
    hi_ina = hi["ID1"].isin(canonical_ids)
    hi_inb = hi["ID2"].isin(canonical_ids)
    n_intra_pairs = int((hi_ina & hi_inb).sum())
    # La tercera categoría reúne pares cuyos dos extremos están fuera de la cohorte.
    # Se incluyen para que el informe represente todos los casos detectados.
    both_external = hi[~hi_ina & ~hi_inb]
    both_external_pairs = [
        {"id1": r["ID1"], "id2": r["ID2"], "kin": round(float(r["kin"]), 4), "k2": round(float(r["k2"]), 4)}
        for _, r in both_external.iterrows()
    ]
    dup_mz_report = {
        "threshold": args.replicate_threshold,
        "citation": "Manichaikul et al. 2010 (KING): dup/MZ si kinship > 1/2^(3/2) ~= 0.354",
        "note": ("PC-Relate NO distingue replica tecnica de gemelo MZ (ambos kinship ~0.5, k2 ~1); "
                 "por eso 'duplicate_or_MZ', no 'replica'. Distinto del duplicado de FILAS de "
                 "metadata (BB-COVL-397, dos filas identicas colapsadas en el join), que no es un "
                 "duplicado genomico sino un artefacto de la tabla de metadata."),
        "n_intra_cohort_pairs": n_intra_pairs,
        "n_nontrivial_groups": len(rep_nontrivial),
        "n_individuals_with_external_duplicate_or_MZ": len(external_canonical),
        "external_pairs": [
            {"canonical_id": c, "external_id": o, "kin": round(k, 4), "k2": round(k2, 4)}
            for c, o, k, k2 in sorted(rep_external)
        ],
        "n_pairs_both_external": len(both_external_pairs),
        "pairs_both_external": both_external_pairs,
        "interpretation": (
            "0 grupos dup/MZ no-triviales intra-cohorte -> el dedup ocurrio aguas arriba: cada "
            "individuo canonico con un duplicado/MZ genomico conserva UNA sola version; la copia "
            "gemela (sufijo -R, prefijo RHT/DRC, o version alterna) quedo FUERA de los 2619. "
            "La copia canónica la decidió el join lai_rare∩metadata y no un criterio de QC; queda "
            "pendiente comprobar que se retuvo la de mejor call-rate/het. Las features "
            "de co-sharing M14 se computaron sobre los 2619 (carriers recomputados en el subset), asi "
            "que la copia externa no aporta aristas. Los conteos rare_* son por-individuo (genotipos "
            "propios), de modo que la copia externa no suma a la cuenta de un individuo; pero OJO: el "
            "universo de SITIO 'raro' NO se recalculo dentro de 2619 (ver rare_* en el diccionario)."),
    }
    with open(args.outdir / "duplicate_or_mz_report.json", "w") as fh:
        json.dump(dup_mz_report, fh, indent=2, ensure_ascii=False)

    dup_mz_df = pd.DataFrame({
        "sample_id": list(rep_id_to_group.keys()),
        "duplicate_or_MZ_group": [str(g) for g in rep_id_to_group.values()],
    })
    rep_size_by_str = {str(k): v for k, v in rep_group_size.items()}
    dup_mz_df["duplicate_or_MZ_group_size"] = dup_mz_df["duplicate_or_MZ_group"].map(rep_size_by_str)
    dup_mz_df["has_external_duplicate_or_MZ"] = dup_mz_df["sample_id"].isin(external_canonical)

    # --- columnas explicitas de grupo de parentesco (SIN primaria; una por umbral pedido) ---
    kinship_df = pd.DataFrame({"sample_id": sorted(canonical_ids)})
    for thr in group_col_thrs:
        tag = f"phi{str(thr).split('.')[1]}"  # 0.0442 -> "phi0442"
        id_to_group = id_to_group_by_thr[thr]
        group_size = group_size_by_thr[thr]
        size_by_str = {str(k): v for k, v in group_size.items()}
        gid_col = f"kinship_group_id_{tag}"
        siz_col = f"kinship_group_size_{tag}"
        kinship_df[gid_col] = kinship_df["sample_id"].map({k: str(v) for k, v in id_to_group.items()})
        kinship_df[siz_col] = kinship_df[gid_col].map(size_by_str)

    # --- ensamblar modeling_master ---
    master = fs.merge(leid, on="sample_id", how="left", validate="one_to_one")
    n_after_leiden = len(master)
    master = master.merge(graph[["sample_id", "degree", "weighted_degree"]], on="sample_id",
                           how="left", validate="one_to_one")
    master = master.merge(qc, on="sample_id", how="left", validate="one_to_one")
    master = master.merge(geo, on="sample_id", how="left", validate="one_to_one")
    master = master.merge(kinship_df, on="sample_id", how="left", validate="one_to_one")
    master = master.merge(dup_mz_df, on="sample_id", how="left", validate="one_to_one")
    master = master.rename(columns={"degree": "grado_M14", "weighted_degree": "grado_M14_ponderado"})

    assert len(master) == 2619, f"modeling_master debe tener 2619 filas, tiene {len(master)}"
    n_no_community = master[args.leiden_col_for_crosstab].isna().sum()

    master.to_csv(args.outdir / "modeling_master.tsv", sep="\t", index=False)

    audit = {
        "p0_dataset_status": "COMPLETO",
        "split_policy_status": "PENDIENTE",
        "p0_status_reason": (
            "dataset de modelado COMPLETO (region/estado + dup/MZ + parentesco a doble umbral "
            "phi0442/phi0884, sin primaria); la POLITICA de split queda PENDIENTE: umbral primario "
            "y unidad de particion se deciden en el diseno del split, todavia no autorizado"),
        "n_feature_store_rows": n_fs_raw,
        "n_feature_store_duplicated_ids": int(dup_fs),
        "n_leiden_rows": len(leid),
        "n_after_join_leiden": n_after_leiden,
        "n_final_rows": len(master),
        "n_individuals_missing_leiden_community": int(n_no_community),
        "metadata_duplicated_ids_deduped": md_dup_ids,
        "n_canonical_missing_geo": n_canon_missing_geo,
        "kinship_group_columns_materialized": group_col_thrs,
        "n_red_samples_requested": len(red_ids),
        "n_red_samples_missing_from_canonical": len(missing_red),
        "red_samples_missing_from_canonical": sorted(missing_red),
        "duplicate_or_mz_threshold": args.replicate_threshold,
        "n_intra_cohort_dup_mz_groups_nontrivial": len(rep_nontrivial),
        "n_individuals_with_external_duplicate_or_MZ": len(external_canonical),
        "n_dup_mz_pairs_both_external": len(both_external_pairs),
        "pcrelate_universe_n_unique_ids": int(pd.concat([kin["ID1"], kin["ID2"]]).nunique()),
    }
    with open(args.outdir / "join_audit.json", "w") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    dict_md = """# Diccionario de columnas -- modeling_master.tsv  (P0_DATASET=COMPLETO, SPLIT_POLICY=PENDIENTE)

Cohorte: 2619 individuos (VCF lai_rare ∩ metadata_cleaned, sin KING). Una fila = un individuo.

**Estado: `P0_DATASET=COMPLETO`, `SPLIT_POLICY=PENDIENTE`.** El dataset de modelado esta completo
(feature store + etiqueta + parentesco + region/estado + dup/MZ + flags QC). No hay split, no hay
features seleccionadas, no hay modelo entrenado. **No se fija un umbral de parentesco primario**: la
tabla trae columnas de grupo para ambos umbrales candidatos y la politica del split (umbral primario
y unidad de particion) se difiere al diseno de `split_manifest.tsv`.

**Fuentes textuales exactas:**
- `Q_*`, `cohort`, `sex`, `n_sharing_partners`, `n_segments_involved`, `total_shared_bp`,
  `n_chromosomes_with_sharing`, `flag_aislado`, `rare_*`, `flag_missing_*`:
  `20_feature_store/20_feature_store/feature_store.tsv` (modulo M20).
  **Procedencia de `rare_*` (corregida 2026-07-15):** M20 corre `bcftools stats -s -` (bloque PSC)
  sobre TODAS las muestras del VCF raro (2723) y luego restringe las FILAS del feature store a los
  2619; **NO recalcula el universo MAC/MAF dentro de 2619**. Es decir: los conteos `rare_*` son
  por-individuo (genotipos propios de cada quien), pero la definicion de sitio 'raro' (MAF<1%)
  proviene del VCF construido sobre el panel completo, no de un recalculo sobre la cohorte de 2619.
- `community_res_*`, `assignment_confidence`: `leiden_assignments.tsv` (M16.5, resolucion Leiden
  sobre M14 MAC>=2 autosomas). NA para los 18 individuos aislados (sin nodo en el grafo).
- `grado_M14`, `grado_M14_ponderado`: `graph_nodes.tsv` (M16.5), grado / grado ponderado en el
  mismo grafo del que sale la etiqueta -- **riesgo de circularidad**: la etiqueta y el grado
  comparten sustrato (M14). No usar `grado_M14` como feature de entrenamiento sin declarar esto.
- `region`, `state`: `metadata_cleaned.txt` (columnas Region/State, Nunes et al. 2025). Cobertura
  total de los 2619 (0 NA); `Unknown` es una categoria explicita, no un faltante. El unico ID
  duplicado en metadata (`BB-COVL-397`, dos filas IDENTICAS) se dedup con verificacion previa de
  identidad de filas. **Esto es un duplicado de FILAS de la tabla de metadata, NO un duplicado
  genomico** -- es distinto de las columnas `duplicate_or_MZ_*` (que vienen de PC-Relate); no se
  mezclan.
- `qc_red`, `qc_gray`, `qc_gray_source_available`, `qc_status`: construidas a partir de
  la lista externa de seis muestras con mezcla de ADN. `qc_gray` queda en NA para las 2613 personas
  restantes porque **no existe artefacto con los IDs de las 25 grises** -- no se inventa la lista;
  `qc_status='unclassified'` para todo el que no sea de las 6 rojas, exactamente para no rotular
  a las 25 grises no identificadas como "limpias".
- `kinship_group_id_phi0442` / `kinship_group_size_phi0442` (3er grado, phi>=0.0442) y
  `kinship_group_id_phi0884` / `kinship_group_size_phi0884` (2do grado, phi>=0.0884): componentes
  conexas de PC-Relate (Conomos 2016, `qc_pcrelate/dnabr.pcrelate.kin.tsv`) restringidas
  a los 2619 canónicos, a cada umbral. **Se conservan ambos** para evaluar la sensibilidad del
  agrupamiento al umbral de parentesco. No hay columna `kinship_group_id` sin sufijo para evitar
  que el split quede ligado de forma implícita a un umbral.
- `duplicate_or_MZ_group` / `duplicate_or_MZ_group_size` / `has_external_duplicate_or_MZ`:
  duplicado tecnico O gemelo monocigotico via PC-Relate phi>=0.354 (Manichaikul et al. 2010, banda
  KING dup/MZ). **PC-Relate NO distingue replica tecnica de gemelo MZ** (ambos kinship ~0.5, k2 ~1);
  por eso el nombre `duplicate_or_MZ`, no `replica`. **0 grupos dup/MZ no-triviales intra-cohorte**
  -- todos son singleton. `has_external_duplicate_or_MZ=True` marca los ~20 individuos cuya copia
  gemela quedo FUERA de los 2619 (dedup aguas arriba: la version `-R`/RHT/DRC alterna no entro). La
  llamada se sostiene por **k2 (IBD2) ~= 1**, no por el kinship puntual: k2->1 es la firma inequivoca
  de dup/MZ y es practicamente inmune al sesgo por admixtura (compartir los DOS alelos IBD en todo el
  genoma no lo fabrica la estructura poblacional); PC-Relate (Conomos 2016) ademas es robusto a
  admixtura por construccion. Ver `duplicate_or_mz_report.json` (incluye la 3.a categoria: pares con
  ambos extremos externos, ni canonico).

**Por que `has_external_duplicate_or_MZ` es inerte para el split (verificado, no asumido):** las
features de co-sharing M14 (`n_sharing_partners`, `n_segments_involved`, `total_shared_bp`,
`grado_M14`) se computaron sobre EXACTAMENTE los 2619 (el painter recomputa carriers sobre las
muestras seleccionadas, MAC>=2 en el subset), asi que la copia externa no aporta aristas. Los
conteos `rare_*` son por-individuo (genotipos propios), de modo que la copia externa no suma a la
cuenta de un individuo; PERO (ver procedencia de `rare_*` arriba) el universo de SITIO 'raro' NO se
recalculo dentro de 2619. Deuda declarada (crítica): que copia de cada par dup/MZ quedo canonica
lo decidio el join lai_rare∩metadata, NO un criterio de QC -> auditar a futuro que se retuvo la de
mejor call-rate/het.

**phi agrupa pedigrí, NO toda la co-descendencia founder-difusa (matiz de dominio):** phi de
PC-Relate se estima con SNPs COMUNES y captura parentesco genealogico reciente. El co-sharing de
RARAS lo genera descendencia reciente PERO TAMBIEN deriva founder difusa; en las comunidades founder
amazonicas (BrazilA) muchos pares con phi<0.0442 comparten muchas raras por drift compartido. Por
eso un phi bajo NO garantiza independencia para el co-sharing de raras en esas comunidades: las
bandas agrupan bien la familia literal, pero no toda la co-descendencia.

**Riesgo de circularidad declarado (NO es fuga train/test):** `community_res_*` y `grado_M14`
derivan del mismo grafo M14. Dentro de una comunidad, features diadicas como Sigma-l, n_segments
y densidad son parecidas entre emparentados y no-emparentados por construccion del grafo. Esto es
una **limitacion de circularidad** de la etiqueta (deriva de M14), no necesariamente una fuga de
informacion entre train y test; por si sola NO obliga a un diseno community-holdout. Un modelo
entrenado con esas features y evaluado contra `community_res_*` mide recuperabilidad técnica fuera
de muestra, no validación biológica independiente.

**Condiciones necesarias del split:**
`split_manifest.tsv` NO se considera valido si no declara, en su header, las tres decisiones que
P0 dejo abiertas a proposito:
  1. **Umbral de agrupamiento** usado (`phi0442` vs `phi0884`) y por que; correr AMBOS como analisis
     de sensibilidad y reportar el delta de metrica. Sin esto, el TSV plano deja que downstream use
     la columna equivocada y se re-introduce el sesgo que P0 elimino.
  2. **Unidad de particion** (bloque de parentesco vs comunidad vs individuo). A phi>=0.0442 el
     `kappa` del modelo de configuracion (Molloy-Reed) es >2, MIENTRAS la mayor componente OBSERVADA
     es 119/2619 (~4,5%). kappa>2 es una propiedad del modelo aleatorio de configuracion, NO una
     percolacion observada; el grafo de parentesco real esta mas "en clusters". El diseno del split
     debe medir el tamano de bloque real antes de asumir que GroupKFold por bloque es viable.
  3. **Mascara de parentesco en la feature diadica**: si el split es por-individuo, la feature
     diadica DEBE enmascarar pares con phi>=umbral, o el ΔAUC sera parentesco disfrazado de senal
     (historico del proyecto: 72% del ΔAUC eran parientes sin mascara). El rechazo del
     community-holdout es correcto, pero NO exime de esta mascara.

**Deuda declarada (`SPLIT_POLICY=PENDIENTE`, no olvido):** `region`/`state` y `grupos de
duplicado/MZ` se agregaron en esta corrida (dataset completo). Quedan como deuda a decidir en el split: (a) las 3
precondiciones de arriba, (b) auditar la calidad de la copia canonizada de los ~20 pares
duplicados, (c) medir colinealidad `region` x `community_res_1` (Cramer's V) antes de usar `region`
como covariable de estratificacion -- si `region` casi-determina la comunidad, reintroduce la
etiqueta por la puerta de atras.
"""
    (args.outdir / "modeling_master_dict.md").write_text(dict_md)

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("dup_or_mz:", json.dumps(dup_mz_report, indent=2, ensure_ascii=False))
    print("componentes:", json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
