#!/usr/bin/env python3
"""M18: comparación de asIBD común con comunidades de co-sharing raro de M16.5.

Construye la estructura de comunidades de variantes comunes a igual ancestría local a partir del
asIBD de Nunes. Los archivos contienen Refined IBD sobre WGS refaseado y están estratificados por
ancestría local como anc1, anc2 y anc3. Se usa la misma configuración de Leiden de M16.5 y se
comparan las tres ancestrías desde dos perspectivas:

  - concordancia: una comunidad rara ya está contenida en la partición común de más de 2 cM;
  - complementariedad: la comunidad rara separa una estructura que asIBD colapsa y cuenta con
    respaldo de un eje ortogonal;
  - sin respaldo: la comunidad no pasa el criterio ortogonal y se excluye indicando el motivo.

El análisis no estima fechas. M14 usa IBS no faseado en bp y aquí solo se mide la resolución de las
particiones; los cM de asIBD provienen de Nunes y se usan como afinidad. Una comunidad rara se
considera complementaria cuando muestra enriquecimiento del haplogrupo mitocondrial frente al null
de permutaciones y no está contenida en la partición común de su propia ancestría. El mtDNA es el
eje ortogonal fijo. chrY, geografía y finestructure se informan como contexto, pero no intervienen
en ese criterio porque finestructure deriva del mismo sharing común.

La afinidad se normaliza por la oportunidad de ancestría local mediante
cM acumulados / (dosage_a * dosage_b). Así se evita considerar cercana una pareja solo porque
ambos individuos tienen una mayor proporción de esa ancestría. El informe conserva el vector
AFR/EUR/NAM/EAS completo de cada comunidad, sin asignar individuos por argmax.

La implementación reutiliza la configuración de Leiden de M16.5 importando
`ibd_community_enhanced`, que se prepara junto a este script.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ibd_community_enhanced import sparse_to_igraph, run_leiden_multiresolution  # noqa: E402

LOG = logging.getLogger("asibd_comparator")

ANC_COLS = ["Autosomes_African_anc", "Autosomes_European_anc",
            "Autosomes_Indigenous_anc", "Autosomes_EastAsian_anc"]
ANC_SHORT = {"Autosomes_African_anc": "AFR", "Autosomes_European_anc": "EUR",
             "Autosomes_Indigenous_anc": "NAM", "Autosomes_EastAsian_anc": "EAS"}
SHORT_TO_DOSE_COL = {v: k for k, v in ANC_SHORT.items()}
# El orden inicial de Gnomix es African=0, European=1 y Native=2. confirm_anc_mapping comprueba
# esta correspondencia durante la ejecución usando la dosis media de ancestría local, sin argmax.
DEFAULT_ANC_MAP = "anc1=AFR,anc2=EUR,anc3=NAM"
# Geografía y finestructure se informan como contexto, pero no forman parte del criterio ortogonal.
INFORMATIVE_AXES = ["Region", "State", "finestructure_clusters", "CHRY_MAIN_HAPLOGROUP"]
ORTHO_AXIS = "MTDNA_MAIN_HAPLOGROUP"  # Eje fijo de linaje materno, independiente del grafo autosómico.
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    """Define y devuelve los argumentos de línea de comandos."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leiden", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--asibd_dir", required=True)
    ap.add_argument("--asibd_glob", default="anc*.gapfilled_ibd")
    ap.add_argument("--anc_map", default=DEFAULT_ANC_MAP)
    ap.add_argument("--resolution_col", default="community_res_1")
    # Mantiene la configuración de Leiden usada en la corrida C de M16.5.
    ap.add_argument("--leiden_resolutions", default="0.5,0.8,1.0,1.2,1.5,2.0,3.0")
    ap.add_argument("--leiden_n_seeds", type=int, default=25)
    ap.add_argument("--leiden_min_community_size", type=int, default=3)
    ap.add_argument("--leiden_consensus_resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--affinity_weight", choices=["sum_cm", "n_seg", "max_cm"], default="sum_cm")
    ap.add_argument("--normalize_by_dosage", default="true",
                    help="divide la afinidad por dosage_a*dosage_b de la ancestría del archivo")
    ap.add_argument("--min_edge_weight", type=float, default=0.0)
    ap.add_argument("--arbiter_min_cm", type=float, default=2.0,
                    help="Umbral en cM del árbitro Refined-IBD por ancestría")
    ap.add_argument("--ortho_perm_n", type=int, default=2000,
                    help="Cantidad de permutaciones para el null de enriquecimiento de mtDNA")
    ap.add_argument("--ortho_perm_alpha", type=float, default=0.05)
    ap.add_argument("--arbiter_containment", type=float, default=0.80,
                    help="Fracción de una comunidad rara necesaria para considerarla contenida")
    ap.add_argument("--out_prefix", required=True)
    return ap.parse_args()


def load_rare(path: str, res_col: str) -> pd.DataFrame:
    """Carga las comunidades raras y normaliza los ID de muestra."""
    la = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
    if res_col not in la.columns:
        raise SystemExit(f"resolution_col {res_col!r} not in {list(la.columns)}")
    out = la[["sample_id", res_col]].rename(columns={res_col: "rare_comm"})
    out["rare_comm"] = pd.to_numeric(out["rare_comm"], errors="coerce")
    return out


def load_metadata(path: str, keep_ids: set[str]) -> pd.DataFrame:
    """Carga metadatos para las muestras incluidas en el análisis."""
    md = pd.read_csv(path, sep="\t", dtype=str).drop_duplicates("ID")  # dedup BB-COVL-397
    for c in ANC_COLS:
        md[c] = pd.to_numeric(md.get(c), errors="coerce")
    md = md[md["ID"].isin(keep_ids)].copy()
    md["asibd_id"] = md["Cohort"].astype(str) + "_" + md["ID"].astype(str)
    md = md.rename(columns={"ID": "sample_id"})  # leiden sample_id == metadata ID
    LOG.info("metadata: %d individuals joined to leiden", len(md))
    return md


def parse_asibd(path: Path, asibd_to_sample: dict[str, str], weight: str) -> pd.DataFrame:
    """Lee gapfilled_ibd por streaming y agrega segmentos por pareja no ordenada.

    Solo conserva parejas cuyos dos ID pertenecen a la cohorte canónica.
    """
    agg_cm: dict[tuple[str, str], float] = {}
    agg_n: dict[tuple[str, str], int] = {}
    agg_max: dict[tuple[str, str], float] = {}
    n_lines = n_kept = 0
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 8:
                continue
            n_lines += 1
            a = asibd_to_sample.get(f[0]); b = asibd_to_sample.get(f[2])
            if a is None or b is None or a == b:
                continue
            try:
                cm = float(f[7])
            except ValueError:
                continue
            key = (a, b) if a < b else (b, a)
            agg_cm[key] = agg_cm.get(key, 0.0) + cm
            agg_n[key] = agg_n.get(key, 0) + 1
            agg_max[key] = max(agg_max.get(key, 0.0), cm)
            n_kept += 1
    LOG.info("%s: %d segs, %d kept, %d pairs", path.name, n_lines, n_kept, len(agg_cm))
    if not agg_cm:
        return pd.DataFrame(columns=["sample_a", "sample_b", "raw_cm", "n_seg", "max_cm"])
    rows = [(a, b, agg_cm[(a, b)], agg_n[(a, b)], agg_max[(a, b)]) for (a, b) in agg_cm]
    df = pd.DataFrame(rows, columns=["sample_a", "sample_b", "raw_cm", "n_seg", "max_cm"])
    df["weight"] = {"sum_cm": df["raw_cm"], "n_seg": df["n_seg"].astype(float),
                    "max_cm": df["max_cm"]}[weight]
    return df


def build_affinity(pair_df: pd.DataFrame, samples: list[str], min_w: float,
                   dose: dict[str, float] | None) -> sp.csr_matrix:
    """Construye la afinidad simétrica y dispersa sobre `samples`.

    Cuando se proporciona `dose`, divide el peso por dosage_a*dosage_b para normalizar la
    oportunidad de ancestría local.
    """
    idx = {s: i for i, s in enumerate(samples)}
    n = len(samples)
    if pair_df.empty:
        return sp.csr_matrix((n, n), dtype=np.float64)
    d = pair_df.copy()
    w = d["weight"].to_numpy(dtype=np.float64)
    if dose is not None:
        da = d["sample_a"].map(dose).to_numpy(dtype=np.float64)
        db = d["sample_b"].map(dose).to_numpy(dtype=np.float64)
        w = w / np.maximum(EPS, da * db)
    keep = w > min_w
    i = d["sample_a"].map(idx).to_numpy()[keep]
    j = d["sample_b"].map(idx).to_numpy()[keep]
    S = sp.coo_matrix((w[keep], (i, j)), shape=(n, n)).tocsr()
    return (S + S.T).tocsr()


def common_partition(pair_df: pd.DataFrame, samples: list[str], args,
                     dose: dict[str, float] | None) -> np.ndarray:
    """Aplica Leiden de M16.5 a la afinidad común y alinea la partición con las muestras."""
    S = build_affinity(pair_df, samples, args.min_edge_weight, dose)
    g = sparse_to_igraph(S, samples)
    resolutions = [float(x) for x in args.leiden_resolutions.split(",")]
    assign_df, _m, _c, _b, _mem = run_leiden_multiresolution(
        g, resolutions=resolutions, n_seeds=args.leiden_n_seeds,
        min_community_size=args.leiden_min_community_size, base_seed=args.seed,
        consensus_resolution=args.leiden_consensus_resolution)
    col = f"community_res_{args.leiden_consensus_resolution:g}"
    if col not in assign_df.columns:
        col = [c for c in assign_df.columns if c.startswith("community_res_")][0]
    # assign_df no incluye sample_id; sus filas siguen el orden de los vértices del grafo. La serie se
    # construye sobre `samples` para conservar esa correspondencia.
    return pd.Series(assign_df[col].to_numpy(), index=samples).reindex(samples).fillna(-1).to_numpy().astype(np.int64)


def confirm_anc_mapping(per_anc_pairs: dict[str, pd.DataFrame], md: pd.DataFrame,
                        anc_map: dict[str, str]) -> dict[str, dict]:
    """Comprueba el mapa anc->ancestría mediante la dosis local media de los portadores.

    El archivo declarado como NAM debe mostrar la mayor media de Autosomes_Indigenous_anc.
    """
    md_i = md.set_index("sample_id")
    report = {}
    for anc, pdf in per_anc_pairs.items():
        ids = pd.unique(pd.concat([pdf["sample_a"], pdf["sample_b"]], ignore_index=True)) if not pdf.empty else []
        sub = md_i.reindex(ids)
        means = {ANC_SHORT[c]: round(float(sub[c].mean()), 4) for c in ANC_COLS} if len(ids) else {}
        observed = max(means, key=means.get) if means else None
        declared = anc_map.get(anc, anc)
        report[anc] = {"declared": declared, "n_carriers": int(len(ids)),
                       "mean_local_dosage": means, "observed_max_dosage": observed,
                       "match": (observed == declared)}
    return report


def auto_detect_anc_map(per_anc_pairs: dict[str, pd.DataFrame],
                        md: pd.DataFrame) -> tuple[dict[str, str], dict]:
    """Detecta la ancestría de cada archivo por enriquecimiento de dosis ponderado por cM.

    La media simple de portadores no sirve porque casi todos los individuos tienen IBD en todos los
    archivos. Por eso se pondera la dosis de cada pareja por los cM compartidos y se compara con la
    media de la cohorte. La asignación entre archivo y ancestría es uno a uno.
    """
    dose = {a: md.set_index("sample_id")[c].to_dict() for a, c in SHORT_TO_DOSE_COL.items()}
    gmean = {a: float(md[SHORT_TO_DOSE_COL[a]].mean()) for a in SHORT_TO_DOSE_COL}
    enr: dict[str, dict[str, float]] = {}
    for ak, pdf in per_anc_pairs.items():
        if pdf.empty:
            continue
        cm = pdf["raw_cm"].to_numpy(dtype=np.float64)
        den = cm.sum()
        if den <= 0:
            continue
        e = {}
        for a in dose:
            if gmean[a] <= 0:
                continue
            da = pdf["sample_a"].map(dose[a]).fillna(0.0).to_numpy(dtype=np.float64)
            db = pdf["sample_b"].map(dose[a]).fillna(0.0).to_numpy(dtype=np.float64)
            e[a] = float((cm * (da + db) / 2.0).sum() / den / gmean[a])
        enr[ak] = {a: round(v, 3) for a, v in e.items()}
    triples = sorted(((v, ak, a) for ak, d in enr.items() for a, v in d.items()), reverse=True)
    assigned, used = {}, set()
    for _v, ak, a in triples:
        if ak not in assigned and a not in used:
            assigned[ak] = a
            used.add(a)
    return assigned, enr


def _modal_fraction(labels: pd.Series) -> tuple[str, float]:
    vc = labels.dropna().value_counts(normalize=True)
    return (str(vc.index[0]), float(vc.iloc[0])) if len(vc) else ("NA", 0.0)


def _perm_pvalue(all_labels: np.ndarray, k: int, obs_frac: float, n_perm: int, seed: int) -> float:
    """One-sided p: prob. that a random size-k draw from the cohort's mtDNA labels reaches modal
    fraction >= obs_frac. Breaks the arbitrary 0.5 cutoff AND the disjunctive trap (fixed axis)."""
    valid = all_labels[~pd.isna(all_labels)]
    if k < 1 or len(valid) == 0 or k > len(valid):
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        s = rng.choice(valid, size=k, replace=False)
        _, c = np.unique(s, return_counts=True)
        if (c.max() / k) >= obs_frac:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def concordance_table(md: pd.DataFrame, common_by_anc, arbiter_by_anc,
                      samples: list[str], args) -> pd.DataFrame:
    """Resume concordancia y enriquecimiento entre particiones raras y comunes."""
    smap = {s: i for i, s in enumerate(samples)}
    md = md.set_index("sample_id")
    rare = md["rare_comm"]
    mtdna_all = md[ORTHO_AXIS].to_numpy() if ORTHO_AXIS in md.columns else np.array([np.nan] * len(md))
    rows = []
    for ci, c in enumerate(sorted(int(x) for x in rare.dropna().unique() if int(x) >= 0)):
        members = rare.index[rare == c]
        n = len(members)
        anc_vec = md.loc[members, ANC_COLS].mean()
        dom = ANC_SHORT[ANC_COLS[int(np.argmax(anc_vec.to_numpy()))]]
        # Criterio A: eje ortogonal fijo de mtDNA frente al null de permutaciones.
        mt_lab, mt_frac = _modal_fraction(md.loc[members, ORTHO_AXIS]) if ORTHO_AXIS in md.columns else ("NA", 0.0)
        p_perm = _perm_pvalue(mtdna_all, n, mt_frac, args.ortho_perm_n, args.seed + ci)
        predicts_ortho = p_perm < args.ortho_perm_alpha
        # Criterio B: contención en el árbitro de su propia ancestría dominante.
        arb = arbiter_by_anc.get(dom)
        if arb is not None:
            lab = pd.Series([arb[smap[s]] for s in members if s in smap])
            lab = lab[lab >= 0]
            _l, arb_frac = _modal_fraction(lab) if len(lab) else ("NA", 0.0)
        else:
            arb_frac = 0.0
        contained = arb_frac >= args.arbiter_containment
        if not predicts_ortho:
            category = "fail_no_orthogonal_support"
        elif not contained:
            category = "complement"      # A and B
        else:
            category = "reaffirm"        # supported, but common arbiter already contains it
        # Estos ejes se informan como contexto, pero no forman parte del criterio.
        info = {}
        for ax in INFORMATIVE_AXES:
            if ax in md.columns:
                lab, frac = _modal_fraction(md.loc[members, ax])
                info[f"{ax}_modal"] = lab
                info[f"{ax}_modal_frac"] = round(frac, 3)
        rows.append({
            "rare_community": c, "n": n, "dom_anc_plurality": dom,
            "anc_AFR": round(float(anc_vec[ANC_COLS[0]]), 3), "anc_EUR": round(float(anc_vec[ANC_COLS[1]]), 3),
            "anc_NAM": round(float(anc_vec[ANC_COLS[2]]), 3), "anc_EAS": round(float(anc_vec[ANC_COLS[3]]), 3),
            "mtDNA_modal": mt_lab, "mtDNA_modal_frac": round(mt_frac, 3), "mtDNA_perm_p": round(p_perm, 4),
            "predicts_orthogonal_A": predicts_ortho,
            "arbiter_containment_frac": round(arb_frac, 3), "not_contained_in_arbiter_B": not contained,
            "category": category, **info,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Ejecuta el comparador y escribe tablas y gráficos de salida."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    anc_map = dict(kv.split("=") for kv in args.anc_map.split(","))
    normalize = str(args.normalize_by_dosage).lower() in ("true", "1", "yes")

    rare = load_rare(args.leiden, args.resolution_col)
    keep_ids = set(rare["sample_id"])
    md = load_metadata(args.metadata, keep_ids).merge(rare, on="sample_id", how="inner")
    samples = md["sample_id"].tolist()
    asibd_to_sample = dict(zip(md["asibd_id"], md["sample_id"]))
    LOG.info("cohort for comparison: %d individuals", len(samples))

    per_anc_pairs = {}
    for f in sorted(Path(args.asibd_dir).glob(args.asibd_glob)):
        anc_key = f.stem.split(".")[0]
        per_anc_pairs[anc_key] = parse_asibd(f, asibd_to_sample, args.affinity_weight)

    detected_map, anc_enrichment = auto_detect_anc_map(per_anc_pairs, md)
    anc_audit = confirm_anc_mapping(per_anc_pairs, md, anc_map)  # Referencia secundaria basada en medias.
    LOG.info("anc map: declared=%s | detected(cM-weighted)=%s | enrichment=%s",
             anc_map, detected_map, json.dumps(anc_enrichment, ensure_ascii=False))
    declared_map = dict(anc_map)
    anc_map = detected_map  # Se usa el mapa detectado porque el orden declarado no estaba comprobado.

    rare_arr = md.set_index("sample_id")["rare_comm"].reindex(samples).to_numpy()
    md_i = md.set_index("sample_id")
    common_by_anc, arbiter_by_anc, ari_rows = {}, {}, []
    for anc_key, pdf in per_anc_pairs.items():
        anc = anc_map.get(anc_key, anc_key)
        dose = md_i[SHORT_TO_DOSE_COL[anc]].to_dict() if (normalize and anc in SHORT_TO_DOSE_COL) else None
        memb = common_partition(pdf, samples, args, dose)
        common_by_anc[anc] = memb
        # El árbitro por ancestría usa cM sin normalizar para medir la contención cruda.
        arb_pairs = pdf[pdf["raw_cm"] > args.arbiter_min_cm] if not pdf.empty else pdf
        arbiter_by_anc[anc] = common_partition(arb_pairs, samples, args, None)
        mask = (rare_arr >= 0) & (memb >= 0)
        ari = adjusted_rand_score(rare_arr[mask], memb[mask]) if mask.sum() > 1 else float("nan")
        ami = adjusted_mutual_info_score(rare_arr[mask], memb[mask]) if mask.sum() > 1 else float("nan")
        ari_rows.append({"ancestry": anc, "anc_file": anc_key, "ARI_rare_vs_common": round(ari, 4),
                         "AMI_rare_vs_common": round(ami, 4), "n_compared": int(mask.sum()),
                         "NOTE": "ARI es una referencia descriptiva entre escalas temporales distintas"})

    conc = concordance_table(md, common_by_anc, arbiter_by_anc, samples, args)

    out = Path(args.out_prefix); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ari_rows).to_csv(f"{args.out_prefix}.ari_by_ancestry.tsv", sep="\t", index=False)
    conc.to_csv(f"{args.out_prefix}.concordance_complementarity.tsv", sep="\t", index=False)
    cats = conc["category"].value_counts().to_dict() if not conc.empty else {}
    summary = {
        "n_individuals": len(samples),
        "affinity_weight": args.affinity_weight, "dosage_normalized": normalize,
        "anc_mapping_used": anc_map, "anc_mapping_declared_initial": declared_map,
        "anc_enrichment_cM_weighted": anc_enrichment, "anc_mapping_audit_weak_meanCarriers": anc_audit,
        "ari_by_ancestry": ari_rows, "category_counts": {k: int(v) for k, v in cats.items()},
        "arbiter_min_cm": args.arbiter_min_cm,
        "discriminant": (f"Discriminante conjunto: resolución complementaria = (A) enriquecimiento mtDNA "
                         f"supera el null de permutaciones (p<{args.ortho_perm_alpha}) y (B) no está "
                         f"contenida (>{args.arbiter_containment}) en el árbitro de "
                         f">{args.arbiter_min_cm} cM de su ancestría. Deben cumplirse ambas condiciones."),
        "CAVEAT": ("ARI se usa como referencia descriptiva y no como evidencia. mtDNA es el único eje "
                   "del criterio A; chrY, geografía y finestructure solo aportan contexto. La afinidad "
                   "se normaliza por dosis de ancestría local. No se realiza datación temporal porque "
                   "M14 contiene IBS no faseado en bp. Se informan las tres ancestrías, la concordancia, "
                   "la complementariedad y los casos sin respaldo."),
    }
    Path(f"{args.out_prefix}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOG.info("done: %s", cats)


if __name__ == "__main__":
    main()
