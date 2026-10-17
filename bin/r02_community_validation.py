#!/usr/bin/env python3
"""Conditional associations of frozen communities, ancestry and reported IBD.

This consumer reuses saved pair unions and genotype-availability counts. It
does not infer IBD, biological absence, ancestry, populations or causal effects.
Dyadic standard errors condition on the saved partitions and on an explicitly
stated component-dependence model. They do not correct historical selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import re
from statistics import NormalDist
import sys
import time

import numpy as np

import r02_community_ibd as source

SCHEMA = "r02_community_validation_inputs_v1"
RESULT_SCHEMA = "r02_community_validation_v1"
STATUS = "COMPLETE_CONDITIONAL_REPORTED_ASSOCIATIONS_NOT_POPULATION_VALIDATION"
MODEL_SCOPE = "FIXED_COMMUNITIES_COMMON_SUPPORT_REPORTED_CALLS"
PARTITION_MODE = "K05_PARTITION_ASSIGNED"
PARTITION_SCOPE = "FIXED_COMMUNITIES_PARTITION_ASSIGNED_REPORTED_CALLS"
LIMITATIONS = [
    "The common assigned support is a selected subset, not the full cohort or Brazilian population.",
    "Community assignments were learned on these genomes and previously examined; inference conditions on them.",
    "Reported IBD length is a callset outcome; zero is no reported call, not biological IBD absence.",
    "Availability counts describe retained genotype catalogues, not IBD detection territory or DP/GQ.",
    "Collection is an observed sampling label, not sequencing batch; their effects may be confounded.",
    "Global ancestry is estimated; its estimation uncertainty is not propagated by these regressions.",
    "PC-Relate components are threshold-dependent operational units, not certified pedigrees or independent samples.",
    "Dyadic uncertainty assumes errors uncorrelated across component-disjoint pairs and sufficiently many units.",
    "Conditional asymptotic intervals do not account for partition selection or establish independent replication.",
    "Leave-one-component-out ranges are influence diagnostics, not confidence intervals.",
    "The ancestry model is a declared approximation; residual association does not identify fine structure.",
    "Positive-call conditioning changes the estimand and may introduce selection; it is only a sensitivity.",
    "Component-pair balancing changes the target to equally weighted observed component pairs.",
    "No universal winner, rare-increment claim, founder inference or complete K01-K07 validation is produced.",
]


def require(condition, message):
    source.require(condition, message)


def progress(stage, **fields):
    """Operational evidence without identifiers or individual genotypes."""
    print(json.dumps(dict(stage=stage, **fields), sort_keys=True), file=sys.stderr, flush=True)


def finite_array(value, name, ndim=None):
    a = np.asarray(value, dtype=np.float64)
    require((ndim is None or a.ndim == ndim) and np.isfinite(a).all(), "Nonfinite or malformed " + name)
    return a


def ancestry_design(q_a, q_b):
    """Symmetric second-order basis of all four estimated ancestry fractions.

    Means m=(Qa+Qb)/2 and differences d=(Qa-Qb)/2 give m, m*m and
    d*d cross-products. Swapping endpoints leaves every column unchanged.
    All four coordinates remain present; rank reduction handles compositional
    constraints instead of silently treating EAS as absent.
    """
    a, b = finite_array(q_a, "ancestry", 2), finite_array(q_b, "ancestry", 2)
    require(a.shape == b.shape and a.shape[1] == 4, "Four ancestry coordinates required")
    require(np.all((a >= 0) & (a <= 1)) and np.all((b >= 0) & (b <= 1)), "Ancestry outside [0,1]")
    mean, delta = (a+b)/2., (a-b)/2.
    columns, names = [mean[:, k] for k in range(4)], [f"Q_mean_{k}" for k in range(4)]
    for i in range(4):
        for j in range(i, 4):
            columns.extend((mean[:, i]*mean[:, j], delta[:, i]*delta[:, j]))
            names.extend((f"Q_mean_product_{i}_{j}", f"Q_half_difference_product_{i}_{j}"))
    return np.column_stack(columns), names


def collection_design(left, right):
    """Unordered collection-pair indicators; unknown is an explicit category."""
    require(len(left) == len(right), "Collection endpoint lengths differ")
    keys = [tuple(sorted((a if a is not None else "[UNKNOWN]", b if b is not None else "[UNKNOWN]")))
            for a, b in zip(left, right)]
    levels = sorted(set(keys))
    lookup = {key: index for index, key in enumerate(levels)}
    codes = np.fromiter((lookup[key] for key in keys), dtype=np.int64, count=len(keys))
    matrix = np.zeros((len(keys), max(0, len(levels)-1)))
    selected = codes > 0
    matrix[np.flatnonzero(selected), codes[selected]-1] = 1.
    return matrix, ["collection_pair:"+repr(v) for v in levels[1:]], levels


def ss_tolerance(energy, rank_rtol, rank, crossproducts=False):
    """Scale-relative numerical tolerance, not a biological effect threshold.

    Direct residual norms use the squared relative SVD tolerance. Downdated
    cross-products lose precision through subtraction, so additionally require
    signal above 64*eps*rank times the original energy. A failed deletion is
    reported as numerically unevaluable, never promoted to a large coefficient.
    """
    relative = max(rank_rtol**2, (64*np.finfo(float).eps*max(1, rank)) if crossproducts else 0.)
    return relative*np.maximum(np.asarray(energy, dtype=float), np.finfo(float).tiny)


def partial_r2(cross, exposure_ss, outcome_ss, rank):
    """Bound only floating-point excursions; material violations are missing."""
    value = float(cross*cross/(exposure_ss*outcome_ss))
    tolerance = 128*np.finfo(float).eps*max(1, rank)
    if not math.isfinite(value) or value < -tolerance or value > 1.+tolerance:
        return None
    return min(1., max(0., value))


def component_pair_codes(a, b):
    require(len(a) == len(b) and len(a) > 0, "Empty or incompatible component pairs")
    require(all(v is not None for v in a) and all(v is not None for v in b), "Unknown components require exclusion, not a fake cluster")
    levels = sorted(set(a) | set(b))
    lookup = {v: i for i, v in enumerate(levels)}
    ia, ib = np.asarray([lookup[v] for v in a]), np.asarray([lookup[v] for v in b])
    raw = np.minimum(ia, ib)*len(levels)+np.maximum(ia, ib)
    _, groups, counts = np.unique(raw, return_inverse=True, return_counts=True)
    return ia, ib, groups, counts, levels


def dyadic_variance(scores, component_a, component_b, pair_groups, bread):
    """Scalar sandwich with all pairs incident to either component dependent.

    Sum squared component scores minus squared scores for unordered *distinct*
    component pairs. This is the direct overlap covariance sum: a pair of
    observations sharing both endpoints must count once, not twice. Within-
    component observations are incident once and are not subtracted.
    """
    s = finite_array(scores, "score", 1)
    a, b, groups = map(np.asarray, (component_a, component_b, pair_groups))
    require(s.shape == a.shape == b.shape == groups.shape and len(s), "Invalid dyadic score dimensions")
    require(np.all(a >= 0) and np.all(b >= 0) and np.all(groups >= 0), "Negative component index")
    require(math.isfinite(bread) and bread > 0, "Positive residual information required")
    size = int(max(a.max(), b.max()))+1
    incident = np.bincount(a, weights=s, minlength=size)
    distinct = a != b
    incident += np.bincount(b[distinct], weights=s[distinct], minlength=size)
    repeated = np.bincount(groups[distinct], weights=s[distinct])
    meat = float(incident@incident - repeated@repeated)
    variance = meat/(bread*bread)
    return variance, meat, incident


def residual_basis(z, weights, rtol):
    """Weighted nuisance space with scale-normalized SVD and explicit rank."""
    z, w = finite_array(z, "nuisance", 2), finite_array(weights, "weights", 1)
    require(len(z) == len(w) and len(z) > 0 and np.all(w > 0), "Positive aligned weights required")
    require(0 < rtol < 1, "Invalid numerical SVD tolerance")
    root = np.sqrt(w)
    norm = np.sqrt(np.sum(w[:, None]*z*z, axis=0))
    usable = norm > 0
    require(usable.any(), "Nuisance basis has no intercept/information")
    scaled = root[:, None]*z[:, usable]/norm[usable]
    u, s, _ = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(s > s[0]*rtol))
    return u[:, :rank], root, dict(nuisance_columns=z.shape[1], nuisance_rank=rank,
                                  nuisance_zero_columns=int((~usable).sum()),
                                  nuisance_condition_retained=float(s[0]/s[rank-1]))


def linear_projection(y, exposures, z, weights, component_a, component_b,
                      confidence, rank_rtol, influence=True, guard=lambda: None, family_size=None):
    """Fit multiple one-exposure models with the same nuisance/eligible support.

    Exposures are alternatives, never included jointly. FWL residualization
    reuses one nuisance SVD. Influence refits remove every observation incident
    to a component using exact sufficient cross-products, not pair bootstrap.
    """
    y, d = finite_array(y, "outcome", 1), finite_array(exposures, "exposures", 2)
    z, w = finite_array(z, "nuisance", 2), finite_array(weights, "weights", 1)
    require(len(y) == len(d) == len(z) == len(w), "Regression dimensions differ")
    require(0 < confidence < 1, "Confidence outside (0,1)")
    family_size = d.shape[1] if family_size is None else family_size
    require(type(family_size) is int and family_size >= d.shape[1], "Multiplicity family smaller than alternatives")
    if len(y) < 2:
        return [dict(status="INSUFFICIENT_PAIRS", n_pairs=len(y), coefficient=None) for _ in range(d.shape[1])]
    a, b, groups, pair_counts, levels = component_pair_codes(component_a, component_b)
    u, root, rank_info = residual_basis(z, w, rank_rtol)
    yd, dd = y*root, d*root[:, None]
    uy, ud = u.T@yd, u.T@dd
    yr, dr = yd-u@uy, dd-u@ud
    yy, dx = float(yr@yr), np.sum(dr*dr, axis=0)
    xy = dr.T@yr
    total_y = float(np.sum(w*(y-np.average(y, weights=w))**2))
    energy_y, energy_d = float(yd@yd), np.sum(dd*dd, axis=0)
    tolerance_y = ss_tolerance(energy_y, rank_rtol, rank_info["nuisance_rank"])
    tolerance_d = ss_tolerance(energy_d, rank_rtol, rank_info["nuisance_rank"])
    valid = (dx > tolerance_d) & (yy > tolerance_y) & (len(y) > rank_info["nuisance_rank"]+1)
    coef = np.full(d.shape[1], np.nan)
    coef[valid] = xy[valid]/dx[valid]
    exposure_means = np.average(d, axis=0, weights=w)
    rows = []
    standard_q = NormalDist().inv_cdf((1+confidence)/2)
    family_q = NormalDist().inv_cdf(1-(1-confidence)/(2*max(1, family_size)))
    for k in range(d.shape[1]):
        row = dict(n_pairs=len(y), n_components=len(levels), n_component_pairs=len(pair_counts),
            n_same_component_pairs=int(np.sum(a == b)), sum_weights=float(w.sum()),
            effective_weight_count=float(w.sum()**2/(w@w)), **rank_info,
            outcome_weighted_mean=float(np.average(y, weights=w)), exposure_weighted_mean=float(exposure_means[k]),
            residual_exposure_ss=float(dx[k]), residual_outcome_ss=yy,
            exposure_ss_numerical_tolerance=float(tolerance_d[k]), outcome_ss_numerical_tolerance=float(tolerance_y),
            outcome_total_ss=total_y, coefficient=None, partial_r2=None, standard_error=None,
            interval_low=None, interval_high=None, family_interval_low=None, family_interval_high=None,
            family_size=family_size, family_scope="ALTERNATIVES_IN_THIS_EXACT_SUPPORT_MODEL_STRATUM_ONLY_NOT_OTHER_SENSITIVITIES_OR_SELECTION",
            variance_status="NOT_EVALUABLE", variance_meat=None,
            max_component_score_fraction=None, loo_valid=0, loo_invalid=0,
            loo_coefficient_min=None, loo_coefficient_max=None, loo_partial_r2_min=None,
            loo_partial_r2_max=None, loo_sign_changes=None,
            status="CONDITIONAL_ASSOCIATION" if valid[k] else
                "NO_RESIDUAL_DOF" if len(y) <= rank_info["nuisance_rank"]+1 else
                "CONSTANT_OR_EXPLAINED_EXPOSURE" if dx[k] <= tolerance_d[k] else "CONSTANT_OR_EXPLAINED_OUTCOME")
        if valid[k]:
            beta = float(coef[k])
            r2 = partial_r2(xy[k], dx[k], yy, rank_info["nuisance_rank"])
            if r2 is None:
                valid[k] = False
                row["status"] = "NUMERICAL_PROJECTION_INCONSISTENCY"
                rows.append(row)
                continue
            row.update(coefficient=beta, partial_r2=r2)
            error = yr-dr[:, k]*beta
            variance, meat, incident = dyadic_variance(dr[:, k]*error, a, b, groups, dx[k])
            contribution = incident**2
            row.update(variance_meat=meat, max_component_score_fraction=float(contribution.max()/contribution.sum())
                       if contribution.sum() else None)
            if len(levels) < 3:
                row["variance_status"] = "INSUFFICIENT_COMPONENTS_FOR_ASYMPTOTIC_INTERVAL"
            elif variance <= 0 or not math.isfinite(variance):
                row["variance_status"] = "NONPOSITIVE_DYADIC_VARIANCE_NO_INTERVAL"
            else:
                se = math.sqrt(variance)
                row.update(standard_error=se, interval_low=beta-standard_q*se, interval_high=beta+standard_q*se,
                    family_interval_low=beta-family_q*se, family_interval_high=beta+family_q*se,
                    variance_status="ASYMPTOTIC_CONDITIONAL_COMPONENT_MODEL_NOT_SELECTION_ADJUSTED")
        rows.append(row)
    guard()
    if influence and valid.any():
        original_xy = dd.T@yd
        original_xx = np.sum(dd*dd, axis=0)
        original_yy = float(yd@yd)
        loo_tolerance_d = ss_tolerance(original_xx, rank_rtol, u.shape[1], True)
        loo_tolerance_y = ss_tolerance(original_yy, rank_rtol, u.shape[1], True)
        for group in range(len(levels)):
            removed = (a == group) | (b == group)
            keep_n = len(y)-int(removed.sum())
            if keep_n < 2:
                for k in np.flatnonzero(valid): rows[k]["loo_invalid"] += 1
                continue
            ug, dg, yg = u[removed], dd[removed], yd[removed]
            gram = np.eye(u.shape[1])-ug.T@ug
            inverse = np.linalg.pinv(gram, rcond=rank_rtol, hermitian=True)
            remaining_rank = int(np.linalg.matrix_rank(gram, tol=rank_rtol))
            zx, zy = ud-ug.T@dg, uy-ug.T@yg
            residual_xx = original_xx-np.sum(dg*dg, axis=0)-np.sum(zx*(inverse@zx), axis=0)
            residual_xy = original_xy-dg.T@yg-zx.T@(inverse@zy)
            residual_yy = original_yy-float(yg@yg)-float(zy@inverse@zy)
            for k in np.flatnonzero(valid):
                row = rows[k]
                if (keep_n <= remaining_rank+1 or residual_xx[k] <= loo_tolerance_d[k] or
                        residual_yy <= loo_tolerance_y):
                    row["loo_invalid"] += 1
                    continue
                v = float(residual_xy[k]/residual_xx[k])
                r2 = partial_r2(residual_xy[k], residual_xx[k], residual_yy, remaining_rank)
                if r2 is None or not math.isfinite(v):
                    row["loo_invalid"] += 1
                    continue
                row["loo_valid"] += 1
                for key, value in (("coefficient", v), ("partial_r2", r2)):
                    for suffix, func in (("min", min), ("max", max)):
                        field = f"loo_{key}_{suffix}"
                        row[field] = value if row[field] is None else func(row[field], value)
                row["loo_sign_changes"] = (row["loo_sign_changes"] or 0)+int(v*coef[k] < 0)
            if group % 16 == 0: guard()
    return rows


def partition_requests(settings):
    """Explicit frozen hypotheses; IDs are also safe Nextflow task identifiers."""
    ids = settings.get("partition_ids")
    require(isinstance(ids, list) and ids and
            all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", v) for v in ids) and
            len(ids) == len(set(ids)),
            "Unique safe partition_ids required")
    return ids


def analysis_support(people, settings, expected_common, partition_id=None):
    """Change only support, retaining source intersection as a lineage check."""
    common = [i for i, q in enumerate(people["ancestry"]) if q is not None and
              all(people["labels"][p["partition_id"]][i] >= 0 for p in people["partitions"])]
    require(type(expected_common) is int and len(common) == expected_common and len(common) >= 2,
            "Shared assigned ancestry-complete support changed or insufficient")
    people["n_shared_assigned"] = len(common)
    mode = settings["analysis_mode"]
    if mode == PARTITION_MODE:
        ids = partition_requests(settings)
        by_id = {p["partition_id"]: p for p in people["partitions"]}
        require(set(ids) <= set(by_id) and partition_id in ids, "Unknown or unrequested partition ID")
        people["n_source_partitions"] = len(by_id)
        people["partitions"] = [by_id[partition_id]]
        common = [i for i, q in enumerate(people["ancestry"]) if q is not None and
                  people["labels"][partition_id][i] >= 0]
        require(len(common) >= 2, "Partition has insufficient assigned ancestry-complete people")
    else:
        require(partition_id is None, "Partition ID is only valid for K05_PARTITION_ASSIGNED")
        if mode == "K01_FULL_COHORT":
            common = [i for i, q in enumerate(people["ancestry"]) if q is not None]
    return common


def load_inputs(path, expected_sha256, guard, partition_id=None):
    path = Path(path).resolve(strict=True)
    ledger = source.Ledger(path.parent)
    ledger.add(dict(path=str(path), sha256=expected_sha256))
    c = source.strict_json(path)
    require(set(c) == {"schema", "scientific_scope", "source_ibd_contract", "ibd_parts",
            "pair_evidence_manifest", "pair_evidence_npz", "expected_common_samples", "settings"}, "Unexpected validation contract fields")
    settings = c["settings"]
    partition_mode = settings.get("analysis_mode") == PARTITION_MODE
    require(c["schema"] == SCHEMA and c["scientific_scope"] == (PARTITION_SCOPE if partition_mode else MODEL_SCOPE),
            "Unexpected scientific scope")
    required_settings = {"confidence", "rank_rtol", "collection_field", "kinship_thresholds", "leave_one_component_out", "analysis_mode"}
    require(set(settings) == required_settings | ({"partition_ids"} if partition_mode else set()),
            "Unspecified or unrecognized model settings")
    require(0 < settings["confidence"] < 1 and 0 < settings["rank_rtol"] < 1 and
            settings["leave_one_component_out"] is True, "Invalid interval/rank/influence contract")
    require(settings["analysis_mode"] in {"K01_K05_COMMON", "K01_FULL_COHORT", PARTITION_MODE}, "Unknown support/analysis mode")
    old_path = ledger.add(c["source_ibd_contract"])
    old, old_ledger = source.load_contract(old_path, c["source_ibd_contract"]["sha256"])
    people = source.load_people(old, old_ledger)
    require(settings["kinship_thresholds"] == people["phis"], "Component thresholds differ from existing source")
    old_ledger.recheck()
    ledger.records.update(old_ledger.records)
    cm = source.strict_json(old_ledger.add(old["community_evidence"]))
    meta_path = old_ledger.adjacent(old_ledger.add(old["community_evidence"]), cm, "individuals.private.tsv")
    metadata = list(source.table(meta_path))
    field = settings["collection_field"]
    require(field == "Cohort" and all(field in r for r in metadata), "Explicit collection label required; not batch")
    people["collection"] = [None if r[field] == "NA" else r[field] for r in metadata]
    common = analysis_support(people, settings, c["expected_common_samples"], partition_id)
    manifest = source.strict_json(ledger.add(c["pair_evidence_manifest"]))
    npz = ledger.add(c["pair_evidence_npz"])
    require(manifest["schema"] == "r02_pair_evidence_v1" and manifest["kind"] == "aggregate" and
            manifest["output_npz_sha256"] == c["pair_evidence_npz"]["sha256"] and
            manifest["samples_sha256"] == people["sample_hash"] and manifest["chromosomes"] == old["chromosomes"],
            "Genotype availability lineage differs")
    needed = ("Q", "common_N", "J", "individual_callable", "individual_carrier", "n_sites")
    arrays = {}
    with np.load(npz, allow_pickle=False) as data:
        require(data["samples"].tolist() == people["samples"] and str(data["sample_hash"]) == people["sample_hash"] and
                data["chromosomes"].tolist() == old["chromosomes"] and str(data["schema"]) == manifest["schema"] and
                str(data["kind"]) == "aggregate", "NPZ sample/chromosome/schema mismatch")
        n = len(people["samples"])
        for key in needed:
            a = data[key]
            if key in ("Q", "common_N", "J"):
                require(a.shape == (n, n), "Pair matrix dimensions differ")
                require(np.allclose(a, a.T, rtol=0, atol=0, equal_nan=True), "Asymmetric pair matrix")
                if key != "J": require(np.isfinite(a).all() and np.all(a >= 0), "Invalid availability counts")
                else: require(np.all(np.isnan(a) | ((a >= 0) & (a <= 1))), "Invalid rare similarity")
                arrays[key] = a[np.ix_(common, common)].copy()
            elif key == "n_sites":
                require(a.ndim == 0 and int(a) > 0, "Invalid rare site denominator")
                arrays[key] = int(a)
            else:
                require(a.shape == (n,) and np.isfinite(a).all() and np.all(a >= 0), "Invalid individual availability/count")
                arrays[key] = a[common].copy()
            del a
            guard.check()
    require(np.all(arrays["Q"] <= arrays["n_sites"]) and
            np.all(arrays["individual_carrier"] <= arrays["individual_callable"]) and
            np.all(arrays["individual_callable"] <= arrays["n_sites"]), "Rare count denominators inconsistent")
    require([r["chromosome"] for r in c["ibd_parts"]] == old["chromosomes"], "Missing/duplicate chromosome parts")
    local_order = {people["samples"][v]: k for k, v in enumerate(common)}
    lengths = np.zeros((len(common), len(common)), dtype=np.uint64)
    part_records = []
    for spec in c["ibd_parts"] if settings["analysis_mode"] != "K01_FULL_COHORT" else []:
        mp = ledger.add(spec["manifest"])
        m = source.strict_json(mp)
        require(m["schema"] == source.PART_SCHEMA and
                m["status"] == "COMPLETE_REPORTED_IBD_CHROMOSOME_DESCRIPTION" and
                m["source_contract_sha256"] == c["source_ibd_contract"]["sha256"] and
                m["samples_sha256"] == people["sample_hash"] and m["chromosome"] == spec["chromosome"] and
                m["genome_build"] == old["genome_build"], "IBD pair part lineage mismatch")
        pair_path = ledger.adjacent(mp, m, "pairs.private.tsv.gz")
        seen, total, n_pairs = set(), 0, 0
        for row in source.table(pair_path):
            require(row["chromosome"] == spec["chromosome"], "Pair chromosome mismatch")
            a, b = people["order"][row["sample_a"]], people["order"][row["sample_b"]]
            require(a < b and (a, b) not in seen, "Repeated/reversed reported pair within chromosome")
            seen.add((a, b))
            bp = int(row["reported_union_bp"])
            require(bp > 0, "Reported union must be positive")
            total += bp; n_pairs += 1
            if row["sample_a"] in local_order and row["sample_b"] in local_order:
                lengths[local_order[row["sample_a"]], local_order[row["sample_b"]]] += bp
        require(total == m["totals"]["reported_union_bp"] and n_pairs == m["totals"]["pairs_with_reported_call"],
                "Pair table totals differ from saved manifest")
        part_records.append(dict(chromosome=spec["chromosome"], n_reported_pairs=n_pairs, reported_union_bp=total))
        progress("VERIFIED_IBD_CHROMOSOME", chromosome=spec["chromosome"], reported_pairs=n_pairs)
        guard.check()
    arrays["reported_union_bp"] = lengths+lengths.T
    ledger.recheck()
    return c, ledger, people, np.asarray(common), arrays, part_records


def prepared_pairs(people, common, arrays, analysis_mode="K01_K05_COMMON"):
    ia, ib = np.triu_indices(len(common), 1)
    q = np.asarray([people["ancestry"][i] for i in common], dtype=float)
    anc, anc_names = ancestry_design(q[ia], q[ib]) if analysis_mode != "K01_FULL_COHORT" else (np.empty((len(ia), 0)), [])
    collection = [people["collection"][i] for i in common]
    cat, cat_names, cat_levels = collection_design([collection[i] for i in ia], [collection[i] for i in ib])
    called = arrays["individual_callable"]/arrays["n_sites"]
    meas = np.column_stack((arrays["Q"][ia, ib]/arrays["n_sites"], arrays["common_N"][ia, ib],
        (called[ia]+called[ib])/2., np.abs(called[ia]-called[ib])))
    meas_names = ["joint_rare_GT_fraction", "joint_common_GT_count", "mean_individual_rare_GT_fraction", "absolute_individual_rare_GT_fraction_difference"]
    count = arrays["individual_carrier"]
    counts = np.column_stack(((count[ia]+count[ib])/2., np.abs(count[ia]-count[ib])))
    y = arrays["reported_union_bp"][ia, ib].astype(float)/1e6
    eligible = (arrays["Q"][ia, ib] > 0) & (arrays["common_N"][ia, ib] > 0)
    return dict(a=ia, b=ib, ancestry=anc, ancestry_names=anc_names, collection=cat,
        collection_names=cat_names, collection_levels=cat_levels, measurement=meas,
        measurement_names=meas_names, rare_counts=counts, y_ibd=y,
        J=arrays["J"][ia, ib], ancestral_similarity=-np.sum((q[ia]-q[ib])**2, axis=1),
        eligible_measurement=eligible)


def evaluation_rows(c, people, common, pairs, guard):
    """All saved partitions, identical pair support inside each declared stratum."""
    settings = c["settings"]
    pids = [p["partition_id"] for p in people["partitions"]]
    intercept = np.ones((len(pairs["a"]), 1))
    partition_mode = settings["analysis_mode"] == PARTITION_MODE
    if settings["analysis_mode"] != "K01_FULL_COHORT":
        assignments = np.asarray([[people["labels"][pid][i] for pid in pids] for i in common])
        exposure = (assignments[pairs["a"]] == assignments[pairs["b"]]).astype(float)
        q_model = np.column_stack((intercept, pairs["ancestry"]))
        full_model = np.column_stack((q_model, pairs["collection"], pairs["measurement"]))
    z_k01 = np.column_stack((intercept, pairs["collection"], pairs["measurement"]))
    for phi in settings["kinship_thresholds"]:
        component = [people["components"][phi][i] for i in common]
        ca = np.asarray([component[i] for i in pairs["a"]], dtype=object)
        cb = np.asarray([component[i] for i in pairs["b"]], dtype=object)
        known = (ca != None) & (cb != None)  # noqa: E711; object None is an explicit missing ID.
        for relation in ("ALL_KNOWN_COMPONENT", "DIFFERENT_COMPONENT"):
            base = known & pairs["eligible_measurement"]
            if relation == "DIFFERENT_COMPONENT": base &= ca != cb
            for call_scope in (("COMPLETE_REPORTED_CALLSET", "REPORTED_ONLY_SENSITIVITY")
                               if settings["analysis_mode"] != "K01_FULL_COHORT" else []):
                selected = base.copy()
                if call_scope == "REPORTED_ONLY_SENSITIVITY": selected &= pairs["y_ibd"] > 0
                for model, z in (("CRUDE", intercept), ("ANCESTRY", q_model), ("ANCESTRY_COLLECTION_GT_AVAILABILITY", full_model)):
                    result = linear_projection(pairs["y_ibd"][selected], exposure[selected], z[selected],
                        np.ones(int(selected.sum())), ca[selected].tolist(), cb[selected].tolist(),
                        settings["confidence"], settings["rank_rtol"], True, guard.check,
                        family_size=len(settings["partition_ids"]) if partition_mode else None) if selected.sum() >= 2 else [
                            dict(status="INSUFFICIENT_PAIRS", n_pairs=int(selected.sum()), coefficient=None) for _ in pids]
                    progress("MODEL_COMPLETE", criterion="K05", phi=phi, relation=relation, call_scope=call_scope,
                             model=model, pairs=int(selected.sum()), alternatives=len(result))
                    for p, metrics in zip(people["partitions"], result):
                        if partition_mode:
                            metrics.update(family_size=len(settings["partition_ids"]),
                                family_scope="DECLARED_PARTITIONS_DISTINCT_ASSIGNED_SUPPORTS_PER_MODEL_STRATUM_NOT_RANKING_OR_SELECTION_ADJUSTED")
                        yield dict(criterion="K05", outcome="reported_union_Mb", predictor="same_frozen_community",
                            partition_id=p["partition_id"], graph_id=p["graph_id"], resolution=p["resolution"],
                            kinship_threshold=phi, relation=relation, call_scope=call_scope, model=model,
                            weighting="person_pair_uniform", support_people=len(common),
                            support="PARTITION_ASSIGNED_ANCESTRY_COMPLETE" if partition_mode else "SHARED_ASSIGNED_ANCESTRY_COMPLETE",
                            possible_common_pairs=len(base), excluded_unknown_component=int((~known).sum()),
                            excluded_unavailable_GT=int((known & ~pairs["eligible_measurement"]).sum()), **metrics)
            if partition_mode:
                continue
            # K01 does not repeat an identical J/ancestry analysis 57 times.
            selected = base & np.isfinite(pairs["J"])
            for weighting in ("person_pair_uniform", "component_pair_equal_mass"):
                weights = np.ones(int(selected.sum()))
                if weighting == "component_pair_equal_mass" and selected.sum():
                    _, _, groups, sizes, _ = component_pair_codes(ca[selected].tolist(), cb[selected].tolist())
                    weights = 1./sizes[groups]
                for model, z in (("COLLECTION_GT_AVAILABILITY", z_k01),
                    ("COLLECTION_GT_AVAILABILITY_RARE_QUANTITY", np.column_stack((z_k01, pairs["rare_counts"])))):
                    result = linear_projection(pairs["J"][selected], pairs["ancestral_similarity"][selected, None],
                        z[selected], weights, ca[selected].tolist(), cb[selected].tolist(), settings["confidence"],
                        settings["rank_rtol"], True, guard.check) if selected.sum() >= 2 else [
                            dict(status="INSUFFICIENT_PAIRS", n_pairs=int(selected.sum()), coefficient=None)]
                    progress("MODEL_COMPLETE", criterion="K01", phi=phi, relation=relation, model=model,
                             weighting=weighting, pairs=int(selected.sum()))
                    yield dict(criterion="K01", outcome="rare_Jaccard", predictor="negative_squared_four_Q_distance",
                        partition_id="SHARED_57_SUPPORT" if settings["analysis_mode"] == "K01_K05_COMMON" else "COHORT_ANCESTRY_COMPLETE",
                        graph_id="primary/R", resolution=None,
                        kinship_threshold=phi, relation=relation, call_scope="GENOTYPE_EVALUABLE_PAIRS",
                        model=model, weighting=weighting, support_people=len(common),
                        support="SHARED_ASSIGNED_ANCESTRY_COMPLETE" if settings["analysis_mode"] == "K01_K05_COMMON" else "COHORT_ANCESTRY_COMPLETE",
                        possible_support_pairs=len(base),
                        excluded_unknown_component=int((~known).sum()), excluded_unavailable_GT=int((known & ~pairs["eligible_measurement"]).sum()),
                        excluded_undefined_J=int((base & ~np.isfinite(pairs["J"])).sum()), **result[0])


def run(inputs, expected_sha256, output, guard, partition_id=None):
    started = time.monotonic()
    output = Path(output).absolute()
    require(not output.exists() and output.parent.is_dir(), "Output exists or parent absent")
    progress("VERIFY_INPUTS")
    c, ledger, people, common, arrays, parts = load_inputs(inputs, expected_sha256, guard, partition_id)
    mode = c["settings"]["analysis_mode"]
    pairs = prepared_pairs(people, common, arrays, mode)
    progress("INPUTS_READY", analysis_mode=mode, people=len(common), pairs=len(pairs["a"]))
    output.mkdir(mode=0o700)
    rows = list(evaluation_rows(c, people, common, pairs, guard))
    expected = (len(people["partitions"])*len(people["phis"])*2*2*3 if mode != "K01_FULL_COHORT" else 0)
    expected += 0 if mode == PARTITION_MODE else len(people["phis"])*2*2*2
    require(len(rows) == expected, "Incomplete criterion/model/partition grid")
    fields = sorted(set().union(*(r.keys() for r in rows)))
    source.write_table(output/"conditional_associations.tsv", [{f: r.get(f) for f in fields} for r in rows], fields)
    coverage = [dict(partition_id=p["partition_id"], graph_id=p["graph_id"], resolution=p["resolution"],
        n_cohort=len(people["samples"]), n_assigned=sum(v >= 0 for v in people["labels"][p["partition_id"]]),
        n_shared_assigned_ancestry_complete=people["n_shared_assigned"], comparison_support=mode)
        for p in people["partitions"]]
    source.write_table(output/"coverage.tsv", coverage)
    source.write_table(output/"support.private.tsv", [dict(sample_id=people["samples"][i], support=mode) for i in common])
    summary = dict(schema=RESULT_SCHEMA, status=STATUS, analysis_mode=mode, n_cohort=len(people["samples"]),
        n_analysis_people=len(common), n_shared_assigned_people=people["n_shared_assigned"],
        n_possible_common_pairs=len(pairs["a"]), n_partitions=len(people["partitions"]), table_rows=len(rows),
        model_status_counts=dict(Counter(r["status"] for r in rows)), chromosomes=parts,
        settings=c["settings"], measurement_columns=pairs["measurement_names"], ancestry_columns=pairs["ancestry_names"],
        collection_column=pairs["collection_names"], collection_levels=pairs["collection_levels"],
        no_new_clustering=True, no_new_ibd=True, no_new_genotypes=True, no_pvalues=True, no_winner_selected=True,
        scientific_validation=False, no_confirmatory_inference=True, interval_scope="CONDITIONAL_ASYMPTOTIC_COMPONENT_MODEL",
        confidence_family="57 fixed K05 alternatives per exact support/model stratum; not all endpoints or historical selection",
        effective_weight_count_semantics="Kish concentration of mathematical weights, not independent biological replicates",
        limitations=LIMITATIONS)
    if mode == PARTITION_MODE:
        summary.update(partition_id=partition_id, n_source_partitions=people["n_source_partitions"],
            requested_partition_ids=c["settings"]["partition_ids"],
            comparison_support="PARTITION_ASSIGNED_ANCESTRY_COMPLETE",
            confidence_family="Declared partitions per model/kinship/relation/call stratum, distinct assigned supports; not a ranking or selection correction",
            limitations=[v for v in LIMITATIONS if not v.startswith("The common assigned support")] + [
                "This partition's assigned ancestry-complete people define its estimand; unassigned people are excluded, never made a community.",
                "Effects across distinct partition-specific populations cannot be ranked as if their support were identical.",
                "The number of declared partition hypotheses, not the number of tasks in this invocation, defines family intervals."])
    source.save_json(output/"summary.json", summary)
    with (output/"INFORME.md").open("x") as handle:
        if mode == PARTITION_MODE:
            handle.write("# K05: soporte asignado de una partición congelada\n\n")
            handle.write(f"Partición {partition_id}: {len(common)} personas asignadas con ancestría completa, "
                f"{len(pairs['a'])} parejas candidatas y {len(rows)} contrastes K05. No se calculó K01. "
                f"La intersección fuente de {people['n_shared_assigned']} personas sólo verifica procedencia.\n\n")
            handle.write("Respuesta: unión de longitudes IBD informadas por pareja, sumada entre cromosomas, en Mb. "
                "Exposición: misma comunidad congelada; los no asignados se excluyen, no forman un grupo. "
                "Se conservan los ajustes crudo, ancestría cuadrática de cuatro componentes, y ancestría/colección/disponibilidad. "
                "Cada fila identifica corte de parentesco, relación y alcance de llamadas. Cero llamadas no prueba ausencia biológica.\n\n")
            handle.write(f"Los intervalos de familia usan {len(c['settings']['partition_ids'])} hipótesis declaradas "
                "por estrato, aunque esta tarea calcula una sola partición. No corrigen la selección histórica "
                "ni todas las sensibilidades. Varianza no positiva se conserva sin intervalo. "
                "La retirada de un componente completo mide influencia, no otro intervalo de confianza.\n\n")
            handle.write("Los efectos de particiones con soportes distintos responden poblaciones diferentes; "
                "no constituyen un ranking. No se demuestra población, causalidad, fundador ni incremento raro frente a comunes.\n\n")
            for item in summary['limitations']: handle.write("- "+item+"\n")
        else:
            handle.write("# Asociación condicionada: comunidades, ancestría e IBD informado\n\n")
            handle.write(f"El modo es {mode}. Se conservan {len(people['partitions'])} particiones guardadas, sin repetir M16.5; "
                "K01_FULL_COHORT sólo evalúa la relación continua J/ancestría, no vuelve a comparar comunidades. "
                f"El soporte analizado contiene {len(common)} de {len(people['samples'])} personas; "
                f"sus {len(pairs['a'])} parejas no son observaciones biológicas independientes.\n\n")
            handle.write("## K05: qué significa el coeficiente\n\n"
                "Y es la unión de las llamadas IBD por pareja, sumada sobre cromosomas distintos, en megabases. "
                "Cero significa que no se informó una llamada, no ausencia biológica. El indicador C vale uno si ambas personas "
                "pertenecen a la misma comunidad. Se estima Y=βC+Zθ+error: β es una asociación lineal condicionada, no un efecto causal. "
                "Se muestran modelos crudo, ajustado por las cuatro proporciones ancestrales y ampliado por colección y genotipos observables. "
                "Las proporciones entran con una base cuadrática simétrica; es una aproximación declarada, no un modelo demográfico. "
                "La colección no es el lote de secuenciación. La disponibilidad GT de los catálogos no es el territorio detectable de IBD. "
                "Se conservan ambos cortes de componentes PC-Relate, todas las parejas y la sensibilidad que excluye el mismo componente. "
                "El resultado limitado a parejas con llamada responde otra pregunta y se identifica como sensibilidad.\n\n"
                "## K01: semejanza rara y composición ancestral\n\n"
                "J es la intersección/ unión de portación rara en genotipos conjuntamente observados. V es menos la distancia cuadrática "
                "entre las cuatro proporciones ancestrales. El coeficiente de J=βV+Zθ+error y el R² parcial describen la asociación "
                "residual tras colección/disponibilidad; otro ajuste añade cantidad rara. El R² parcial es la correlación residual al cuadrado; "
                "su signo se lee en β. Los pesos uniformes y los que dan igual masa a cada par de componentes responden preguntas distintas.\n\n"
                "## Incertidumbre e influencia\n\n"
                "La matriz de varianza suma productos de puntuaciones de parejas que comparten al menos un componente, contando cada "
                "intersección una sola vez. Sigue la lógica diádica de Aronow, Samii y Assenova (2015), "
                "https://arxiv.org/abs/1312.3398, con componentes operativos como unidades y observaciones repetidas dentro de pares de componentes. "
                "Supone covarianza residual nula entre parejas de componentes disjuntas y suficientes unidades para la aproximación asintótica. "
                "Estas condiciones no están certificadas por el umbral PC-Relate. Los intervalos son condicionales a las particiones ya examinadas, "
                "no corrigen su selección. Los intervalos Bonferroni sólo abarcan las 57 alternativas dentro del mismo soporte/modelo, "
                "no todas las sensibilidades ni la historia del proyecto. Varianza no positiva o falta de rango produce NA, no un intervalo artificial. "
                "También se recalcula el estimando retirando cada componente completo con todas sus parejas incidentes: el rango resultante "
                "es influencia, no intervalo de confianza.\n\n"
                "## Precisión numérica\n\n"
                "Las columnas de control se escalan antes de la descomposición SVD. Un residuo explicado por los controles se considera "
                "no identificable cuando su suma de cuadrados no supera la tolerancia relativa al cuadrado multiplicada por su energía original. "
                "La retirada de componentes usa diferencias de productos cruzados: exige además superar 64 veces la precisión de máquina, "
                "multiplicada por el rango y la energía original. Son protecciones aritméticas, no umbrales de importancia biológica. "
                "Un R² fuera de [0,1] sólo se redondea si la desviación no excede 128 veces precisión de máquina por rango; "
                "una desviación mayor se registra como no evaluable.\n\n"
                "## Conclusiones permitidas\n\n"
                "Una asociación persistente aporta evidencia descriptiva condicionada sobre el callset y el soporte indicados. "
                "No confirma poblaciones, estructura fina, historia fundadora ni aporte raro frente a un modelo común suficiente. "
                "Una asociación imprecisa no demuestra ausencia universal de información. No se selecciona un ganador biológico.\n\n")
            for item in LIMITATIONS: handle.write("- "+item+"\n")
    ledger.recheck(); guard.check()
    manifest = dict(schema=RESULT_SCHEMA, status=STATUS, source_contract_sha256=expected_sha256,
        input_files=ledger.records, source_sha256={p.name: source.sha256(p) for p in
            (Path(__file__), Path(source.__file__), Path(source.__file__).with_name("r02_community_levels.py"))},
        analysis_mode=mode, n_cohort=len(people["samples"]), n_analysis_people=len(common),
        n_shared_assigned_people=people["n_shared_assigned"], n_partitions=len(people["partitions"]),
        table_rows=len(rows), seconds=time.monotonic()-started, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        contains_individual_identifiers=True, public_distribution_allowed=False, scientific_validation=False,
        outputs_sha256={p.name: source.sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    if mode == PARTITION_MODE:
        manifest.update(partition_id=partition_id, n_source_partitions=people["n_source_partitions"],
            requested_partition_ids=c["settings"]["partition_ids"], no_k01=True,
            comparison_support="PARTITION_ASSIGNED_ANCESTRY_COMPLETE")
    source.save_json(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    parser.add_argument("--max-memory-mb", required=True, type=int)
    parser.add_argument("--min-free-mib", required=True, type=int)
    parser.add_argument("--partition-id")
    args = parser.parse_args()
    guard = source.Guard(args.scratch_dir, args.max_memory_mb, 1, args.min_free_mib)
    result = run(args.inputs, args.expected_inputs_sha256, args.output_dir, guard, args.partition_id)
    print(result["status"])


if __name__ == "__main__":
    main()
