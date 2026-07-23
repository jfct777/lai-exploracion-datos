#!/usr/bin/env python3
"""M18 — Common-variant asIBD comparator vs rare-variant co-sharing communities (M16.5).

Builds the COMMON-variant community structure, *at equal local ancestry*, from Nunes' asIBD
(Refined IBD on the rephased WGS, stratified by local ancestry: anc1/anc2/anc3) using the SAME
Leiden as M16.5, and compares it against the rare-variant communities. Reports ALL three
ancestries and BOTH faces:

  - CONCORDANCE    : rare community already contained in the common >2cM arbiter -> reaffirms Nunes.
  - COMPLEMENTARITY: rare community split structure the common asIBD collapses, AND supported by an
                     orthogonal axis -> real resolution.
  - FAILS          : not supported by the orthogonal axis -> excluded, with reason.

Design constraints (project rules) and review fixes applied:
  * Report the 3 ancestries, not only NAM (feedback_all_communities_not_just_nam).
  * Report concordance as a finding, not only divergence (feedback_report_all_validated_findings).
  * NO temporal dating: M14 is unphased IBS in bp; this measures partition RESOLUTION, not Ne
    (decision 2026-06-02-no-convertir-bp-a-cm). asIBD cM here is Nunes', used only for affinity.
  * Pre-registered JOINT discriminant (decision 2026-06-03): a rare community = REAL resolution iff
    (A) it predicts a FIXED, a-priori orthogonal axis (maternal mtDNA haplogroup) ABOVE A PERMUTATION
    NULL  AND  (B) it is NOT contained in the >arbiter_min_cm common asIBD partition of its own
    ancestry. "A OR B" is forbidden (feedback_baseline_normalized_axes_test_power_confound).
    -- mtDNA is the fixed axis (the 9 validated founders are maternal); it is truly orthogonal to the
       autosomal sharing graph. chrY / geography / finestructure are reported as INFORMATIVE only,
       NEVER as criterion A (finestructure derives from the same common sharing as the arbiter).
  * Affinity is NORMALIZED by local-ancestry dosage opportunity (summed cM / (dosage_a*dosage_b)) so
    a pair is not "close" merely for carrying more of that ancestry's tract (review fix #3).
  * No argmax labelling of individuals; full AFR/EUR/NAM/EAS vector reported per community.

Reuses the M16.5 Leiden verbatim by importing from `ibd_community_enhanced` (sibling in bin/).
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
# Hypothesis from Gnomix code order (African=0/European=1/Native=2 -> anc1/2/3). CONFIRMED at runtime
# by `confirm_anc_mapping` using the MEAN LOCAL-ANCESTRY DOSAGE of the carriers (not argmax).
DEFAULT_ANC_MAP = "anc1=AFR,anc2=EUR,anc3=NAM"
# Geography / finestructure are reported but NOT used as the orthogonal criterion (see module docstring).
INFORMATIVE_AXES = ["Region", "State", "finestructure_clusters", "CHRY_MAIN_HAPLOGROUP"]
ORTHO_AXIS = "MTDNA_MAIN_HAPLOGROUP"  # fixed, a-priori, truly orthogonal (maternal lineage)
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leiden", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--asibd_dir", required=True)
    ap.add_argument("--asibd_glob", default="anc*.gapfilled_ibd")
    ap.add_argument("--anc_map", default=DEFAULT_ANC_MAP)
    ap.add_argument("--resolution_col", default="community_res_1")
    # Leiden config — MUST match the M16.5 winning run (corrida_C).
    ap.add_argument("--leiden_resolutions", default="0.5,0.8,1.0,1.2,1.5,2.0,3.0")
    ap.add_argument("--leiden_n_seeds", type=int, default=25)
    ap.add_argument("--leiden_min_community_size", type=int, default=3)
    ap.add_argument("--leiden_consensus_resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--affinity_weight", choices=["sum_cm", "n_seg", "max_cm"], default="sum_cm")
    ap.add_argument("--normalize_by_dosage", default="true",
                    help="divide pair affinity by dosage_a*dosage_b of the file's ancestry (review fix #3)")
    ap.add_argument("--min_edge_weight", type=float, default=0.0)
    ap.add_argument("--arbiter_min_cm", type=float, default=2.0,
                    help=">cM threshold for the per-ancestry Refined-IBD arbiter (Browning&Browning 2cM)")
    ap.add_argument("--ortho_perm_n", type=int, default=2000,
                    help="permutations for the mtDNA enrichment null (criterion A)")
    ap.add_argument("--ortho_perm_alpha", type=float, default=0.05)
    ap.add_argument("--arbiter_containment", type=float, default=0.80,
                    help="fraction of a rare community sitting in one arbiter community = 'contained' (B)")
    ap.add_argument("--out_prefix", required=True)
    return ap.parse_args()


def load_rare(path: str, res_col: str) -> pd.DataFrame:
    la = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
    if res_col not in la.columns:
        raise SystemExit(f"resolution_col {res_col!r} not in {list(la.columns)}")
    out = la[["sample_id", res_col]].rename(columns={res_col: "rare_comm"})
    out["rare_comm"] = pd.to_numeric(out["rare_comm"], errors="coerce")
    return out


def load_metadata(path: str, keep_ids: set[str]) -> pd.DataFrame:
    md = pd.read_csv(path, sep="\t", dtype=str).drop_duplicates("ID")  # dedup BB-COVL-397
    for c in ANC_COLS:
        md[c] = pd.to_numeric(md.get(c), errors="coerce")
    md = md[md["ID"].isin(keep_ids)].copy()
    md["asibd_id"] = md["Cohort"].astype(str) + "_" + md["ID"].astype(str)
    md = md.rename(columns={"ID": "sample_id"})  # leiden sample_id == metadata ID
    LOG.info("metadata: %d individuals joined to leiden", len(md))
    return md


def parse_asibd(path: Path, asibd_to_sample: dict[str, str], weight: str) -> pd.DataFrame:
    """Stream gapfilled_ibd (id1 hap1 id2 hap2 chr start end cM); aggregate per unordered pair,
    keeping only pairs whose BOTH ids map into the canonical cohort."""
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
    """Symmetric sparse affinity over `samples`. If `dose` given, divide each pair weight by
    dosage_a*dosage_b (local-ancestry opportunity normalization, review fix #3)."""
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
    """Same M16.5 Leiden on the common affinity; consensus-resolution membership aligned to samples."""
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
    # FIX (review): assign_df has NO sample_id column (only community_res_*); rows are aligned to the
    # graph vertex order == `samples`. Build the Series directly on `samples`.
    return pd.Series(assign_df[col].to_numpy(), index=samples).reindex(samples).fillna(-1).to_numpy().astype(np.int64)


def confirm_anc_mapping(per_anc_pairs: dict[str, pd.DataFrame], md: pd.DataFrame,
                        anc_map: dict[str, str]) -> dict[str, dict]:
    """Audit the anc->ancestry map by MEAN LOCAL-ANCESTRY DOSAGE of the carriers (not argmax):
    the file declared NAM should have the highest mean Autosomes_Indigenous_anc among its carriers."""
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
    """Detect anc-file -> ancestry from cM-WEIGHTED dosage enrichment (NOT mean-of-carriers,
    which fails because ~all individuals carry IBD in all files). For each anc file, weight each
    pair's global-ancestry dosage by its shared cM; the ancestry whose dosage is most enriched
    over the cohort mean identifies the file. Greedy 1-to-1 assignment. Returns (map, enrichment)."""
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
        # (A) FIXED orthogonal axis = mtDNA, vs permutation null
        mt_lab, mt_frac = _modal_fraction(md.loc[members, ORTHO_AXIS]) if ORTHO_AXIS in md.columns else ("NA", 0.0)
        p_perm = _perm_pvalue(mtdna_all, n, mt_frac, args.ortho_perm_n, args.seed + ci)
        predicts_ortho = p_perm < args.ortho_perm_alpha
        # (B) containment in the arbiter OF ITS OWN dominant ancestry
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
        # informative-only axes (reported, NOT criteria)
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
    anc_audit = confirm_anc_mapping(per_anc_pairs, md, anc_map)  # informative only (mean-of-carriers, weak)
    LOG.info("anc map: declared=%s | DETECTED(cM-weighted)=%s | enrichment=%s",
             anc_map, detected_map, json.dumps(anc_enrichment, ensure_ascii=False))
    declared_map = dict(anc_map)
    anc_map = detected_map  # USE detected — declared (Gnomix 0/1/2 order) was unverified and WRONG

    rare_arr = md.set_index("sample_id")["rare_comm"].reindex(samples).to_numpy()
    md_i = md.set_index("sample_id")
    common_by_anc, arbiter_by_anc, ari_rows = {}, {}, []
    for anc_key, pdf in per_anc_pairs.items():
        anc = anc_map.get(anc_key, anc_key)
        dose = md_i[SHORT_TO_DOSE_COL[anc]].to_dict() if (normalize and anc in SHORT_TO_DOSE_COL) else None
        memb = common_partition(pdf, samples, args, dose)
        common_by_anc[anc] = memb
        # per-ancestry arbiter (>arbiter_min_cm on raw cM, NOT dosage-normalized: contención cruda)
        arb_pairs = pdf[pdf["raw_cm"] > args.arbiter_min_cm] if not pdf.empty else pdf
        arbiter_by_anc[anc] = common_partition(arb_pairs, samples, args, None)
        mask = (rare_arr >= 0) & (memb >= 0)
        ari = adjusted_rand_score(rare_arr[mask], memb[mask]) if mask.sum() > 1 else float("nan")
        ami = adjusted_mutual_info_score(rare_arr[mask], memb[mask]) if mask.sum() > 1 else float("nan")
        ari_rows.append({"ancestry": anc, "anc_file": anc_key, "ARI_rare_vs_common": round(ari, 4),
                         "AMI_rare_vs_common": round(ami, 4), "n_compared": int(mask.sum()),
                         "NOTE": "ARI is the NULL expectation (distinct time-depths), descriptive only"})

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
        "discriminant": (f"JOINT pre-registered: REAL resolution (complement) = (A) mtDNA enrichment "
                         f"beats permutation null (p<{args.ortho_perm_alpha}, FIXED axis) AND (B) not "
                         f"contained (>{args.arbiter_containment}) in the >{args.arbiter_min_cm}cM "
                         f"per-ancestry arbiter. 'A OR B' forbidden."),
        "CAVEAT": ("ARI = null expectation, not evidence. mtDNA is the only criterion-A axis (truly "
                   "orthogonal); chrY/geo/finestructure are informative only (finestructure derives "
                   "from common sharing). Affinity dosage-normalized to avoid local-ancestry-dose "
                   "confound. No temporal dating (M14 = unphased IBS in bp). Report ALL ancestries + "
                   "concordance + complementarity + fails."),
    }
    Path(f"{args.out_prefix}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOG.info("done: %s", cats)


if __name__ == "__main__":
    main()
