#!/usr/bin/env python3
"""Sensitivity audit for the M16.5 rare-allele co-sharing graph.

The module intentionally reuses the frozen M16.5 graph, Leiden and plotting
implementation, but runs a preregistered threshold grid as a separate
experiment.  Heavy scientific dependencies are imported only at run time so
that the contract helpers remain unit-testable in a minimal Python install.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


NOISE_LABEL = -1
ABSENT_LABEL = -2
SCIENTIFIC_ARTIFACT_FILENAMES = (
    "configuration_summary.tsv",
    "resolution_summary.tsv",
    "assignments.tsv.gz",
    "neighbor_comparisons.tsv",
    "pcrelate_community_concentration.tsv",
    "identity_control.json",
    "decision.json",
)


def _slug_bp(value: int) -> str:
    """Stable human-readable token for an integer number of base pairs."""
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}Mb"
    if value % 100_000 == 0:
        return f"{value / 1_000_000:g}Mb".replace(".", "p")
    return f"{value}bp"


def load_contract(path: Path | str) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "experiment_id", "status", "scope", "inputs",
        "cohort", "fixed_parameters", "configuration_design",
        "decision_rules", "outputs",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"preregistration missing keys: {missing}")
    fixed = contract["fixed_parameters"]
    for key in ("resolutions", "primary_resolution", "n_seeds", "base_seed",
                "minimum_community_size", "weight_transform", "umap"):
        if key not in fixed:
            raise ValueError(f"fixed_parameters missing '{key}'")
    if fixed["primary_resolution"] not in fixed["resolutions"]:
        raise ValueError("primary_resolution must occur in resolutions")
    scope = contract["scope"]
    if scope.get("runs_nmf") is not False:
        raise ValueError("this experiment must explicitly disable NMF")
    if scope.get("selects_configuration_by_finestructure") is not False:
        raise ValueError("fineSTRUCTURE cannot select a configuration")
    configs = build_configurations(contract)
    ids = [c["config_id"] for c in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("configuration ids must be unique")
    identity = contract["configuration_design"]["identity_control"]
    if identity["must_match"] not in ids:
        raise ValueError("identity control target is not a configuration")
    return contract


def build_configurations(contract: dict[str, Any]) -> list[dict[str, Any]]:
    design = contract["configuration_design"]
    grid = design["primary_grid"]
    configurations: list[dict[str, Any]] = []
    for edge_bp in grid["minimum_total_shared_bp"]:
        for segment_bp in grid["minimum_longest_segment_bp"]:
            config_id = f"edge{_slug_bp(int(edge_bp))}_seg{_slug_bp(int(segment_bp))}"
            role = "main_anchor" if (int(edge_bp), int(segment_bp)) == (
                5_000_000, 1_000_000) else "main_grid"
            configurations.append({
                "config_id": config_id,
                "role": role,
                "minimum_total_shared_bp": int(edge_bp),
                "minimum_longest_segment_bp": int(segment_bp),
            })
    for key, role in (("identity_control", "identity_control"),
                      ("stress_control", "stress_control")):
        item = design[key]
        configurations.append({
            "config_id": str(item["id"]),
            "role": role,
            "minimum_total_shared_bp": int(item["minimum_total_shared_bp"]),
            "minimum_longest_segment_bp": int(item["minimum_longest_segment_bp"]),
        })
    return configurations


def neighboring_grid_pairs(configurations: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Pairs differing by one adjacent value on one primary-grid axis."""
    main = [c for c in configurations if c["role"] in {"main_grid", "main_anchor"}]
    edges = sorted({c["minimum_total_shared_bp"] for c in main})
    segments = sorted({c["minimum_longest_segment_bp"] for c in main})
    lookup = {(c["minimum_total_shared_bp"], c["minimum_longest_segment_bp"]):
              c["config_id"] for c in main}
    pairs: list[tuple[str, str]] = []
    for edge in edges:
        for left, right in zip(segments, segments[1:]):
            pairs.append((lookup[(edge, left)], lookup[(edge, right)]))
    for segment in segments:
        for left, right in zip(edges, edges[1:]):
            pairs.append((lookup[(left, segment)], lookup[(right, segment)]))
    return pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_core(path: Path):
    spec = importlib.util.spec_from_file_location("m16_5_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import core module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mean_pairwise_ari(memberships, min_size: int, mask, metrics, np) -> float:
    filtered = [m16_relabel(m, min_size, np) for m in memberships]
    values = []
    for left, right in itertools.combinations(filtered, 2):
        if int(mask.sum()) < 2:
            continue
        values.append(metrics.adjusted_rand_score(left[mask], right[mask]))
    return float(np.mean(values)) if values else math.nan


def m16_relabel(membership, min_size: int, np):
    values = np.asarray(membership, dtype=np.int64).copy()
    labels, counts = np.unique(values, return_counts=True)
    small = labels[counts < min_size]
    values[np.isin(values, small)] = NOISE_LABEL
    keep = values != NOISE_LABEL
    if keep.any():
        remap = {int(old): new for new, old in enumerate(np.unique(values[keep]))}
        values[keep] = np.array([remap[int(x)] for x in values[keep]], dtype=np.int64)
    return values


def _safe_auc(y, score, metrics, np) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    if valid.sum() < 3 or np.unique(y[valid]).size != 2:
        return math.nan
    return float(metrics.roc_auc_score(y[valid], score[valid]))


def _safe_spearman(y, score, scipy_stats, np) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(score)
    if valid.sum() < 3 or np.unique(y[valid]).size < 2 or np.unique(score[valid]).size < 2:
        return math.nan
    return float(scipy_stats.spearmanr(y[valid], score[valid]).statistic)


def _oof_q_auc(q_matrix, assigned, diagnostic: dict[str, Any], np,
               metrics, model_selection, pipeline, preprocessing,
               linear_model) -> float:
    """Five-fold OOF diagnostic; no fitted prediction is retained."""
    x = np.asarray(q_matrix, dtype=float)
    y = np.asarray(assigned, dtype=np.int8)
    valid = np.isfinite(x).all(axis=1)
    x = x[valid]
    y = y[valid]
    counts = np.bincount(y, minlength=2)
    n_folds = int(diagnostic["n_folds"])
    if x.shape[0] < 2 * n_folds or int(counts.min()) < n_folds:
        raise ValueError(
            "five-fold autosomal-Q diagnostic is not estimable for this configuration")
    estimator = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            C=float(diagnostic["C"]),
            class_weight=str(diagnostic["class_weight"]),
            max_iter=int(diagnostic["max_iter"]),
            solver="lbfgs",
            random_state=int(diagnostic["random_seed"]),
        ),
    )
    splitter = model_selection.StratifiedKFold(
        n_splits=n_folds,
        shuffle=bool(diagnostic["shuffle"]),
        random_state=int(diagnostic["random_seed"]),
    )
    probabilities = model_selection.cross_val_predict(
        estimator, x, y, cv=splitter, method="predict_proba", n_jobs=1)[:, 1]
    return float(metrics.roc_auc_score(y, probabilities))


def _bias_corrected_cramers_v(left, right, pd, scipy_stats, np) -> float:
    table = pd.crosstab(pd.Series(left, dtype="object"),
                        pd.Series(right, dtype="object"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return math.nan
    n = int(table.to_numpy().sum())
    if n <= 1:
        return math.nan
    chi2 = float(scipy_stats.chi2_contingency(table, correction=False)[0])
    rows, cols = table.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    cols_corrected = cols - ((cols - 1) ** 2) / (n - 1)
    denominator = min(rows_corrected - 1, cols_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else math.nan


def _find(parent: dict[str, str], sample: str) -> str:
    root = sample
    while parent[root] != root:
        root = parent[root]
    while parent[sample] != sample:
        following = parent[sample]
        parent[sample] = root
        sample = following
    return root


def _union(parent: dict[str, str], size: dict[str, int], left: str, right: str) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left == root_right:
        return
    if size[root_left] < size[root_right]:
        root_left, root_right = root_right, root_left
    parent[root_right] = root_left
    size[root_left] += size[root_right]


def _load_pcrelate(path: Path, samples: list[str], contract: dict[str, Any], pd):
    """Stream PC-Relate and retain only thresholded pairs in the M14 cohort."""
    columns = contract["inputs"]["pcrelate_columns"]
    settings = contract["fixed_parameters"]["pcrelate"]
    threshold = float(settings["kinship_threshold"])
    sample_set = set(samples)
    parent = {sample: sample for sample in samples}
    size = {sample: 1 for sample in samples}
    related_pairs: set[tuple[str, str]] = set()
    rows_seen = 0
    reader = pd.read_csv(
        path, sep="\t", compression="infer",
        usecols=[columns["sample_a"], columns["sample_b"], columns["kinship"]],
        dtype={columns["sample_a"]: str, columns["sample_b"]: str},
        chunksize=int(settings["chunksize_rows"]),
    )
    for chunk in reader:
        rows_seen += len(chunk)
        kinship = pd.to_numeric(chunk[columns["kinship"]], errors="raise")
        keep = (
            (kinship >= threshold)
            & chunk[columns["sample_a"]].isin(sample_set)
            & chunk[columns["sample_b"]].isin(sample_set)
            & (chunk[columns["sample_a"]] != chunk[columns["sample_b"]])
        )
        for left, right in chunk.loc[
                keep, [columns["sample_a"], columns["sample_b"]]].itertuples(index=False):
            pair = tuple(sorted((str(left), str(right))))
            if pair in related_pairs:
                continue
            related_pairs.add(pair)
            _union(parent, size, pair[0], pair[1])
    if rows_seen == 0:
        raise ValueError("PC-Relate input is empty")
    if not related_pairs:
        raise ValueError("no in-cohort PC-Relate pair passes the preregistered threshold")
    if rows_seen != int(settings["expected_input_rows"]):
        raise ValueError("PC-Relate row count does not match preregistration")
    if len(related_pairs) != int(settings["expected_related_pairs_in_m14_observed_cohort"]):
        raise ValueError("thresholded in-cohort PC-Relate pair count changed")
    roots = {sample: _find(parent, sample) for sample in samples}
    component_sizes: dict[str, int] = {}
    for root in roots.values():
        component_sizes[root] = component_sizes.get(root, 0) + 1
    minimum_size = int(settings["family_component_minimum_size"])
    family_component = {
        sample: root for sample, root in roots.items()
        if component_sizes[root] >= minimum_size
    }
    return sorted(related_pairs), family_component, rows_seen


def _pcrelate_metrics(samples, membership, related_pairs, family_component,
                       config_id: str, minimum_community_size: int, np):
    index = {sample: idx for idx, sample in enumerate(samples)}
    comparable = []
    same = 0
    for left, right in related_pairs:
        left_label = int(membership[index[left]])
        right_label = int(membership[index[right]])
        if left_label < 0 or right_label < 0:
            continue
        comparable.append((left, right))
        same += int(left_label == right_label)
    assigned = membership >= 0
    rows = []
    maximum = 0.0
    maximum_sized = 0.0
    for community in sorted(np.unique(membership[assigned]).tolist()):
        members = [samples[i] for i in np.flatnonzero(membership == community)]
        counts: dict[str, int] = {}
        n_family_members = 0
        for sample in members:
            component = family_component.get(sample)
            if component is None:
                continue
            n_family_members += 1
            counts[component] = counts.get(component, 0) + 1
        largest = max(counts.values(), default=0)
        concentration = largest / len(members) if members else math.nan
        maximum = max(maximum, concentration if math.isfinite(concentration) else 0.0)
        if len(members) >= minimum_community_size:
            maximum_sized = max(
                maximum_sized,
                concentration if math.isfinite(concentration) else 0.0)
        rows.append({
            "config_id": config_id,
            "community_res_1": int(community),
            "community_size": len(members),
            "n_members_in_family_components": n_family_members,
            "largest_family_component_size": largest,
            "largest_family_component_fraction": concentration,
        })
    n_assigned = int(assigned.sum())
    n_assigned_family = sum(
        sample in family_component
        for sample, keep in zip(samples, assigned, strict=True) if keep)
    return {
        "n_pcrelate_pairs_in_m14_cohort": len(related_pairs),
        "n_pcrelate_pairs_both_assigned": len(comparable),
        "pcrelate_pairs_same_community_fraction": (
            same / len(comparable) if comparable else math.nan),
        "maximum_family_component_fraction_any_community": maximum,
        "maximum_family_component_fraction_community_ge_10": maximum_sized,
        "assigned_in_nontrivial_family_component_fraction": (
            n_assigned_family / n_assigned if n_assigned else math.nan),
    }, rows


def _comparison(left, right, np, metrics) -> dict[str, Any]:
    assigned_left = left >= 0
    assigned_right = right >= 0
    union = assigned_left | assigned_right
    intersection = assigned_left & assigned_right
    jaccard = float(intersection.sum() / union.sum()) if union.any() else math.nan
    if intersection.sum() >= 2:
        ari = float(metrics.adjusted_rand_score(left[intersection], right[intersection]))
        nmi = float(metrics.normalized_mutual_info_score(left[intersection], right[intersection]))
    else:
        ari = nmi = math.nan
    return {
        "assigned_set_jaccard": jaccard,
        "n_common_assigned": int(intersection.sum()),
        "ari_common_assigned": ari,
        "nmi_common_assigned": nmi,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
                    encoding="utf-8")


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats
    from sklearn import linear_model, metrics, model_selection, pipeline, preprocessing

    contract = load_contract(args.preregistration)
    core = _load_core(args.core_script)
    outdir = args.outdir.resolve()
    plot_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    pair_df = core.load_pair_summary(args.pair_summary)
    required_pair = {"sample_a", "sample_b", "total_shared_bp", "max_segment_bp"}
    missing_pair = required_pair - set(pair_df.columns)
    if missing_pair:
        raise ValueError(f"pair summary missing required columns: {sorted(missing_pair)}")
    if len(pair_df) != int(contract["cohort"]["expected_pair_rows"]):
        raise ValueError("pair-summary row count does not match preregistration")
    if float(pair_df["max_segment_bp"].min()) < float(
            contract["cohort"]["minimum_observed_longest_segment_bp"]):
        raise ValueError("pair summary violates the preregistered M14 segment floor")
    samples = core.load_individuals(args.individual_summary)
    global_summary = json.loads(args.global_summary.read_text(encoding="utf-8"))
    full_samples = [str(x) for x in global_summary.get("ordered_samples", [])]
    expected_total = int(contract["cohort"]["expected_total_samples"])
    expected_observed = int(contract["cohort"]["expected_m14_observed_samples"])
    if len(full_samples) != expected_total or len(set(full_samples)) != expected_total:
        raise ValueError("global ordered_samples does not match preregistered cohort")
    if len(samples) != expected_observed or not set(samples).issubset(full_samples):
        raise ValueError("M14 observed sample set does not match preregistration")

    input_cfg = contract["inputs"]
    burden = pd.read_csv(args.burden_table, sep="\t", compression="infer",
                         usecols=["sample_id", input_cfg["burden_column"]],
                         dtype={"sample_id": str})
    if burden["sample_id"].duplicated().any():
        raise ValueError("burden table contains duplicate sample_id values")
    burden_lookup = burden.set_index("sample_id")[input_cfg["burden_column"]].astype(float)
    burden_full = np.array([burden_lookup.get(s, np.nan) for s in full_samples], dtype=float)

    metadata_values, metadata_name, metadata_warnings = core.load_metadata_safe(
        args.metadata, input_cfg["metadata_comparison_column"], samples)
    if metadata_values is None:
        raise ValueError("fineSTRUCTURE metadata could not be aligned: " + "; ".join(metadata_warnings))
    metadata = pd.read_csv(args.metadata, sep="\t", dtype={input_cfg["metadata_id_column"]: str})
    metadata_required = {
        input_cfg["metadata_id_column"], input_cfg["metadata_cohort_column"],
        *input_cfg["autosomal_q_columns"],
    }
    missing_metadata = metadata_required - set(metadata.columns)
    if missing_metadata:
        raise ValueError(f"metadata missing required columns: {sorted(missing_metadata)}")
    metadata_rows_before = len(metadata)
    metadata = metadata.drop_duplicates().copy()
    exact_duplicate_rows_removed = metadata_rows_before - len(metadata)
    if exact_duplicate_rows_removed != int(
            input_cfg["expected_exact_duplicate_rows_removed"]):
        raise ValueError(
            "metadata exact-duplicate count differs from preregistration")
    if metadata[input_cfg["metadata_id_column"]].duplicated().any():
        raise ValueError(
            "metadata contains conflicting rows for one sample identifier")
    metadata = metadata.set_index(input_cfg["metadata_id_column"]).reindex(samples)
    if metadata.index.hasnans or metadata[input_cfg["metadata_cohort_column"]].isna().any():
        raise ValueError("cohort metadata is incomplete for M14 observed samples")
    q_observed = metadata[input_cfg["autosomal_q_columns"]].apply(
        pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(q_observed).all():
        raise ValueError("autosomal Q values are incomplete or non-finite")
    cohort_observed = metadata[input_cfg["metadata_cohort_column"]].astype(str).to_numpy()

    related_pairs, family_component, pcrelate_rows_seen = _load_pcrelate(
        args.pcrelate, samples, contract, pd)

    fixed = contract["fixed_parameters"]
    resolutions = [float(x) for x in fixed["resolutions"]]
    primary = float(fixed["primary_resolution"])
    primary_col = f"community_res_{primary:g}"
    configs = build_configurations(contract)
    full_index = {sample: idx for idx, sample in enumerate(full_samples)}
    observed_full_idx = np.array([full_index[s] for s in samples], dtype=np.int64)

    core._PALETTE_NAME = fixed["palette"]
    core._COMMUNITY_PALETTE = core._resolve_palette(fixed["palette"], 64)

    summary_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    assignment_frames = []
    family_rows: list[dict[str, Any]] = []
    primary_assignments: dict[str, Any] = {}
    matrices: dict[str, Any] = {}

    for config in configs:
        config_id = config["config_id"]
        pair_w = core.aggregate_pair_weights(
            pair_df, None, fixed["weight_transform"],
            config["minimum_longest_segment_bp"])
        S, _ = core.build_sparse_matrix(
            pair_w, samples, config["minimum_total_shared_bp"])
        graph = core.sparse_to_igraph(S, samples)
        assignments, modularity, _consensus, _best_mod, memberships = (
            core.run_leiden_multiresolution(
                graph, resolutions, int(fixed["n_seeds"]),
                int(fixed["minimum_community_size"]), int(fixed["base_seed"]),
                primary))
        membership = assignments[primary_col].to_numpy(dtype=np.int64)
        primary_assignments[config_id] = membership.copy()
        matrices[config_id] = S
        degree = np.diff(S.indptr).astype(np.int64)
        weighted_degree = np.asarray(S.sum(axis=1)).ravel()
        connected = degree > 0
        assigned = membership >= 0

        full_membership = np.full(expected_total, ABSENT_LABEL, dtype=np.int64)
        full_degree = np.zeros(expected_total, dtype=np.int64)
        full_weighted_degree = np.zeros(expected_total, dtype=float)
        full_membership[observed_full_idx] = membership
        full_degree[observed_full_idx] = degree
        full_weighted_degree[observed_full_idx] = weighted_degree
        full_assigned = full_membership >= 0

        fine_mask = assigned & (np.asarray(metadata_values, dtype=object) != "NA")
        if fine_mask.sum() >= 2:
            fine_ari = float(metrics.adjusted_rand_score(
                membership[fine_mask], np.asarray(metadata_values)[fine_mask]))
            fine_nmi = float(metrics.normalized_mutual_info_score(
                membership[fine_mask], np.asarray(metadata_values)[fine_mask]))
        else:
            fine_ari = fine_nmi = math.nan

        burden_observed = burden_full[observed_full_idx]
        ancestry_auc = _oof_q_auc(
            q_observed, assigned, fixed["ancestry_assignment_diagnostic"], np,
            metrics, model_selection, pipeline, preprocessing, linear_model)
        try:
            ancestry_auc_connected = _oof_q_auc(
                q_observed[connected], assigned[connected],
                fixed["ancestry_assignment_diagnostic"], np, metrics,
                model_selection, pipeline, preprocessing, linear_model)
        except ValueError:
            ancestry_auc_connected = math.nan
        assignment_cohort_nmi = float(metrics.normalized_mutual_info_score(
            assigned.astype(np.int8), cohort_observed))
        assignment_cohort_cramers_v = _bias_corrected_cramers_v(
            assigned.astype(np.int8), cohort_observed, pd, scipy_stats, np)
        assignment_cohort_nmi_connected = float(
            metrics.normalized_mutual_info_score(
                assigned[connected].astype(np.int8), cohort_observed[connected]))
        assignment_cohort_cramers_v_connected = _bias_corrected_cramers_v(
            assigned[connected].astype(np.int8), cohort_observed[connected],
            pd, scipy_stats, np)
        related_metrics, per_community_family = _pcrelate_metrics(
            samples, membership, related_pairs, family_component, config_id,
            int(fixed["pcrelate"]["family_concentration_minimum_community_size"]),
            np)
        family_rows.extend(per_community_family)
        summary_rows.append({
            **config,
            "n_total_samples": expected_total,
            "n_m14_observed_samples": len(samples),
            "n_absent_from_m14": expected_total - len(samples),
            "n_edges": int(graph.ecount()),
            "n_nonisolated": int(connected.sum()),
            "n_isolated": int((~connected).sum()),
            "n_assigned": int(assigned.sum()),
            "n_noise": int((membership == NOISE_LABEL).sum()),
            "n_communities": int(np.unique(membership[assigned]).size),
            "seed_ari_all_observed": _mean_pairwise_ari(
                memberships[primary], int(fixed["minimum_community_size"]),
                np.ones(len(samples), dtype=bool), metrics, np),
            "seed_ari_connected": _mean_pairwise_ari(
                memberships[primary], int(fixed["minimum_community_size"]),
                connected, metrics, np),
            "assignment_auc_degree_observed": _safe_auc(assigned, degree, metrics, np),
            "assignment_auc_degree_connected": _safe_auc(
                assigned[connected], degree[connected], metrics, np),
            "assignment_auc_degree_full": _safe_auc(full_assigned, full_degree, metrics, np),
            "assignment_auc_burden_observed": _safe_auc(assigned, burden_observed, metrics, np),
            "assignment_auc_burden_connected": _safe_auc(
                assigned[connected], burden_observed[connected], metrics, np),
            "assignment_auc_burden_full": _safe_auc(full_assigned, burden_full, metrics, np),
            "assignment_spearman_degree_observed": _safe_spearman(
                assigned, degree, scipy_stats, np),
            "assignment_spearman_degree_full": _safe_spearman(
                full_assigned, full_degree, scipy_stats, np),
            "assignment_spearman_burden_observed": _safe_spearman(
                assigned, burden_observed, scipy_stats, np),
            "assignment_spearman_burden_full": _safe_spearman(
                full_assigned, burden_full, scipy_stats, np),
            "assignment_oof_auc_four_autosomal_q": ancestry_auc,
            "assignment_oof_auc_four_autosomal_q_connected": ancestry_auc_connected,
            "assignment_vs_cohort_nmi": assignment_cohort_nmi,
            "assignment_vs_cohort_nmi_connected": assignment_cohort_nmi_connected,
            "assignment_vs_cohort_bias_corrected_cramers_v": assignment_cohort_cramers_v,
            "assignment_vs_cohort_bias_corrected_cramers_v_connected": (
                assignment_cohort_cramers_v_connected),
            **related_metrics,
            "finestructure_ari_assigned": fine_ari,
            "finestructure_nmi_assigned": fine_nmi,
            "n_finestructure_compared": int(fine_mask.sum()),
        })

        grouped_mod = modularity.groupby("resolution", sort=True)
        for resolution, frame in grouped_mod:
            column = f"community_res_{float(resolution):g}"
            labels = assignments[column].to_numpy(dtype=np.int64)
            raw = memberships[float(resolution)]
            resolution_rows.append({
                "config_id": config_id,
                "resolution": float(resolution),
                "modularity_best": float(frame["modularity"].max()),
                "modularity_mean": float(frame["modularity"].mean()),
                "modularity_sd": float(frame["modularity"].std(ddof=1)),
                "n_communities": int(np.unique(labels[labels >= 0]).size),
                "n_noise": int((labels == NOISE_LABEL).sum()),
                "seed_ari_all_observed": _mean_pairwise_ari(
                    raw, int(fixed["minimum_community_size"]),
                    np.ones(len(samples), dtype=bool), metrics, np),
                "seed_ari_connected": _mean_pairwise_ari(
                    raw, int(fixed["minimum_community_size"]),
                    connected, metrics, np),
            })

        full_resolution_assignments = {}
        for column in assignments.columns:
            values = np.full(expected_total, ABSENT_LABEL, dtype=np.int64)
            values[observed_full_idx] = assignments[column].to_numpy(dtype=np.int64)
            full_resolution_assignments[column] = values
        frame = pd.DataFrame({
            "sample_id": full_samples,
            "config_id": config_id,
            "role": config["role"],
            "status": np.where(full_membership == ABSENT_LABEL,
                               "absent_from_m14",
                               np.where(full_membership == NOISE_LABEL,
                                        "noise", "assigned")),
            "degree": full_degree,
            "weighted_degree": full_weighted_degree,
            input_cfg["burden_column"]: burden_full,
            **full_resolution_assignments,
        })
        assignment_frames.append(frame)

        plot_cfg = fixed["umap"]
        core.plot_network_umap(
            S, membership, plot_dir / f"network_umap_{config_id}.png",
            metadata_values=metadata_values, metadata_name=metadata_name,
            max_nodes=int(plot_cfg["max_nodes"]), dpi=int(plot_cfg["dpi"]),
            width_in=float(plot_cfg["width"]), height_in=float(plot_cfg["height"]),
            export_pdf=False, export_svg=False,
            layout_seed=int(plot_cfg["layout_seed"]),
            n_spectral=int(plot_cfg["n_spectral"]),
            label_min_size=int(plot_cfg["label_min_size"]),
            community_annotations=None, adjust_labels=False)

    identity_cfg = contract["configuration_design"]["identity_control"]
    identity_id = identity_cfg["id"]
    anchor_id = identity_cfg["must_match"]
    matrix_equal = bool((matrices[identity_id] != matrices[anchor_id]).nnz == 0)
    assignment_equal = bool(np.array_equal(
        primary_assignments[identity_id], primary_assignments[anchor_id]))
    identity_payload = {
        "identity_config": identity_id,
        "anchor_config": anchor_id,
        "matrix_exactly_equal": matrix_equal,
        "primary_assignment_exactly_equal": assignment_equal,
        "pass": matrix_equal and assignment_equal,
        "explanation": "M14 retained only segments >=1 Mb, so a 0.5 Mb longest-segment filter must be identical to 1 Mb on these inputs."
    }
    _write_json(outdir / "identity_control.json", identity_payload)

    neighbor_rows = []
    for left_id, right_id in neighboring_grid_pairs(configs):
        neighbor_rows.append({
            "comparison_type": "adjacent_primary_grid",
            "config_left": left_id,
            "config_right": right_id,
            **_comparison(primary_assignments[left_id], primary_assignments[right_id],
                          np, metrics),
        })
    neighbor_rows.append({
        "comparison_type": "identity_control",
        "config_left": identity_id,
        "config_right": anchor_id,
        **_comparison(primary_assignments[identity_id], primary_assignments[anchor_id],
                      np, metrics),
    })
    pd.DataFrame(summary_rows).to_csv(
        outdir / "configuration_summary.tsv", sep="\t", index=False)
    pd.DataFrame(resolution_rows).to_csv(
        outdir / "resolution_summary.tsv", sep="\t", index=False)
    pd.concat(assignment_frames, ignore_index=True).to_csv(
        outdir / "assignments.tsv.gz", sep="\t", index=False,
        compression="gzip")
    pd.DataFrame(neighbor_rows).to_csv(
        outdir / "neighbor_comparisons.tsv", sep="\t", index=False)
    pd.DataFrame(family_rows).to_csv(
        outdir / "pcrelate_community_concentration.tsv", sep="\t", index=False)

    decision = {
        "experiment_id": contract["experiment_id"],
        "status": ("PASS_DESCRIPTIVE_SENSITIVITY_READY" if identity_payload["pass"]
                   else "FAIL_IDENTITY_CONTROL"),
        "automatic_winner_selected": False,
        "canonical_m16_5_replaced": False,
        "nmf_run": False,
        "interpretation": (
            "All configurations are reported for joint scientific review; "
            "fineSTRUCTURE concordance is descriptive and was not optimized."
        ),
        "metadata_warnings": metadata_warnings,
        "identity_control": identity_payload,
        "pcrelate": {
            "method": "PC-Relate",
            "threshold": fixed["pcrelate"]["kinship_threshold"],
            "rows_streamed": pcrelate_rows_seen,
            "in_cohort_pairs_passing_threshold": len(related_pairs),
        },
        "metadata_exact_duplicate_rows_removed": exact_duplicate_rows_removed,
    }
    _write_json(outdir / "decision.json", _json_value(decision))

    inventory_path = outdir / "artifact_inventory.tsv"
    artifacts = [outdir / name for name in SCIENTIFIC_ARTIFACT_FILENAMES]
    artifacts.extend(sorted((outdir / "plots").glob("*.png")))
    missing_artifacts = [path for path in artifacts if not path.is_file()]
    if missing_artifacts:
        raise RuntimeError(
            f"scientific artifact inventory is incomplete: {missing_artifacts}")
    with inventory_path.open("w", encoding="utf-8") as handle:
        handle.write("relative_path\tsize_bytes\tsha256\n")
        for artifact in artifacts:
            handle.write(
                f"{artifact.relative_to(outdir)}\t{artifact.stat().st_size}\t"
                f"{sha256_file(artifact)}\n")
    if not identity_payload["pass"]:
        raise RuntimeError("identity control failed; results are not interpretable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-summary", type=Path, required=True)
    parser.add_argument("--individual-summary", type=Path, required=True)
    parser.add_argument("--global-summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--burden-table", type=Path, required=True)
    parser.add_argument("--pcrelate", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--core-script", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
