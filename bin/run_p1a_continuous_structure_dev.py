#!/usr/bin/env python3
"""Nested DEV-only test of continuous structure in the M14-minor graph.

The estimator is deliberately narrow.  It asks whether fold-specific spectral
coordinates add regional association within RHT/Q_EUR>0.5 after ancestry,
minor-allele burden, missingness and local graph connectivity are controlled.
Fold 3 is never an endpoint, anchor, tuning row or reported prediction.  M14
itself was built transductively on the cohort, so a PASS is not independent
biological validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import helmert
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler


DEV_FOLDS = (0, 1, 2, 4)
RESERVED_FOLD = 3
MODEL_LEVELS = ("B0", "B1", "A")


class ContractError(RuntimeError):
    """A fail-closed contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractError(f"{label}: missing columns {missing}")


def unique_rows(frame: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
        raise ContractError(f"{label}: blank {key}")
    frame = frame.copy()
    frame[key] = frame[key].astype(str).str.strip()
    duplicated = frame[frame.duplicated(key, keep=False)]
    for value, group in duplicated.groupby(key, sort=False):
        if len(group.fillna("__NA__").drop_duplicates()) != 1:
            raise ContractError(f"{label}: conflicting duplicate {value}")
    return frame.drop_duplicates(key, keep="first").copy()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != "P1A-continuous-structure-DEV":
        raise ContractError("wrong P1A contract")
    if contract.get("status") != "PRE_AMENDED_BEFORE_OUTCOMES":
        raise ContractError("P1A contract is not PRE_AMENDED_BEFORE_OUTCOMES")
    amendments = contract.get("amendments", [])
    if len(amendments) != 1:
        raise ContractError("P1A requires exactly one preregistration amendment")
    amendment = amendments[0]
    required_amendment = {
        "amendment_id": "P1A-PRE-A1-OUTER-PROJECTABILITY",
        "status": "PRE_AMENDED_BEFORE_OUTCOMES",
        "aborted_run_id": "p1a-continuous-dev-20260819a",
        "outcomes_observed": False,
        "rationale": "inner anchor regime not calibrated by outer 20% gate",
    }
    if any(amendment.get(key) != value for key, value in required_amendment.items()):
        raise ContractError("P1A preregistration amendment drifted")
    scope = contract["scope"]
    forbidden = ("uses_reserved_fold_3", "uses_novel_brazilian_variants", "uses_nam_target", "runs_graph_nulls")
    if any(bool(scope.get(key)) for key in forbidden):
        raise ContractError("P1A scope opened a forbidden branch")
    if tuple(contract["population"]["development_folds"]) != DEV_FOLDS:
        raise ContractError("development folds drifted")
    if int(contract["population"]["reserved_fold"]) != RESERVED_FOLD:
        raise ContractError("reserved fold drifted")
    return contract


def verify_hashes(paths: dict[str, Path], contract: dict[str, Any]) -> dict[str, str]:
    expected = contract["inputs"]["expected_sha256"]
    observed: dict[str, str] = {}
    for label, wanted in expected.items():
        if label not in paths:
            raise ContractError(f"hash contract references unknown input: {label}")
        got = sha256_file(paths[label])
        observed[label] = got
        if got != wanted:
            raise ContractError(f"{label}: sha256 {got} != frozen {wanted}")
    return observed


def validate_preflight(
    preflight: dict[str, Any], observed_hashes: dict[str, str], contract: dict[str, Any]
) -> None:
    """Fail closed unless the immutable preflight matches this DEV contract exactly."""
    expected = contract["preflight_contract"]
    if preflight.get("schema_version") != expected["schema_version"]:
        raise ContractError("preflight schema_version drifted")
    if preflight.get("decision") != expected["decision"]:
        raise ContractError("P1A preflight did not authorize DEV")
    observed_scope = preflight.get("scope")
    if not isinstance(observed_scope, dict):
        raise ContractError("preflight scope is missing")
    for key, value in expected["scope"].items():
        if observed_scope.get(key) != value:
            raise ContractError(f"preflight scope drifted at {key}")
    observed_counts = preflight.get("counts")
    if not isinstance(observed_counts, dict):
        raise ContractError("preflight counts are missing")
    for key, value in expected["expected_counts"].items():
        observed = observed_counts.get(key)
        if key == "primary_regions":
            observed = sorted(str(item) for item in (observed or []))
            value = sorted(str(item) for item in value)
        if observed != value:
            raise ContractError(f"preflight count drifted at {key}: {observed} != {value}")
    observed_gates = preflight.get("gates")
    if not isinstance(observed_gates, dict):
        raise ContractError("preflight gates are missing")
    for key in expected["required_true_gates"]:
        if observed_gates.get(key) is not True:
            raise ContractError(f"preflight required gate is absent or false: {key}")
    preflight_hashes = preflight.get("input_sha256")
    if not isinstance(preflight_hashes, dict):
        raise ContractError("preflight input_sha256 is missing")
    for label, digest in observed_hashes.items():
        if label not in preflight_hashes:
            raise ContractError(f"preflight lacks frozen hash for {label}")
        if preflight_hashes[label] != digest:
            raise ContractError(f"preflight hash mismatch for {label}")


def ilr_transform(q: np.ndarray, delta: float) -> np.ndarray:
    """Four-part ancestry composition to three orthonormal ILR coordinates."""
    values = np.asarray(q, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ContractError("Q must be an n x 4 matrix")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("Q contains invalid values")
    values = np.maximum(values, delta)
    values /= values.sum(axis=1, keepdims=True)
    return np.log(values) @ helmert(4, full=False).T


def canonicalize_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    """Resolve sign only; linear isotropic ridge remains rotation invariant."""
    result = np.asarray(vectors, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def macro_log_loss(y: np.ndarray, probabilities: np.ndarray, classes: list[str], clip: float) -> float:
    y = np.asarray(y, dtype=object)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y), len(classes)):
        raise ContractError("probability matrix shape mismatch")
    if not np.isfinite(probabilities).all():
        raise ContractError("non-finite probabilities")
    probabilities = np.clip(probabilities, clip, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    class_to_index = {label: index for index, label in enumerate(classes)}
    losses = []
    for label in classes:
        mask = y == label
        if not mask.any():
            raise ContractError(f"macro log-loss lacks class {label}")
        losses.append(float(-np.log(probabilities[mask, class_to_index[label]]).mean()))
    return float(np.mean(losses))


def true_class_loss(y: np.ndarray, probabilities: np.ndarray, classes: list[str], clip: float) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    indexes = np.asarray([lookup[str(label)] for label in y], dtype=int)
    p = np.clip(probabilities[np.arange(len(y)), indexes], clip, 1.0)
    return -np.log(p)


def balanced_accuracy(y: np.ndarray, probabilities: np.ndarray, classes: list[str]) -> float:
    """Mean recall across frozen classes, reported descriptively only."""
    y = np.asarray(y, dtype=object)
    predicted = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
    return float(np.mean([(predicted[y == label] == label).mean() for label in classes]))


def macro_brier(y: np.ndarray, probabilities: np.ndarray, classes: list[str]) -> float:
    """Class-balanced mean multiclass squared probability error; descriptive only."""
    y = np.asarray(y, dtype=object)
    one_hot = np.column_stack([y == label for label in classes]).astype(float)
    row_score = np.square(np.asarray(probabilities, dtype=float) - one_hot).sum(axis=1)
    return float(np.mean([row_score[y == label].mean() for label in classes]))


def align_probabilities(model: LogisticRegression, values: np.ndarray, classes: list[str]) -> np.ndarray:
    raw = model.predict_proba(values)
    aligned = np.zeros((len(values), len(classes)), dtype=float)
    lookup = {str(label): index for index, label in enumerate(model.classes_)}
    if set(lookup) != set(classes):
        raise ContractError("model did not fit every frozen class")
    for index, label in enumerate(classes):
        aligned[:, index] = raw[:, lookup[label]]
    return aligned


def _edge_weights(edges: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "binary":
        return np.ones(len(edges), dtype=float)
    if mode == "log_length":
        bp = pd.to_numeric(edges["total_shared_bp"], errors="raise").to_numpy(dtype=float)
        return np.log1p(bp / 1_000_000.0)
    raise ValueError(mode)


def spectral_fold_features(
    edges: pd.DataFrame,
    anchor_train_ids: set[str],
    target_train_ids: set[str],
    target_eval_ids: set[str],
    mode: str,
    dimensions: int,
    contract: dict[str, Any],
    reserved_ids: set[str] | None = None,
    *,
    stage: str,
    enforce_projectability_gate: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build TRAIN spectral coordinates and strict EVAL-to-TRAIN Nyström projection."""
    if stage not in {"inner", "outer"}:
        raise ContractError(f"unknown projectability stage: {stage}")
    if enforce_projectability_gate != (stage == "outer"):
        raise ContractError(
            "projectability gate contract requires enforce=false for inner and true for outer"
        )
    if target_train_ids & target_eval_ids:
        raise ContractError("TRAIN/EVAL target overlap")
    if not target_train_ids.issubset(anchor_train_ids):
        raise ContractError("target TRAIN is outside anchor TRAIN")
    anchors = sorted(anchor_train_ids)
    anchor_index = {sample: index for index, sample in enumerate(anchors)}
    tt = edges[edges["sample_a"].isin(anchor_train_ids) & edges["sample_b"].isin(anchor_train_ids)].copy()
    tt_weights = _edge_weights(tt, mode)
    rows = tt["sample_a"].map(anchor_index).to_numpy(dtype=int)
    cols = tt["sample_b"].map(anchor_index).to_numpy(dtype=int)
    adjacency = sparse.coo_matrix(
        (np.r_[tt_weights, tt_weights], (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(len(anchors), len(anchors)),
    ).tocsr()
    adjacency.sum_duplicates()
    binary = adjacency.copy()
    binary.data[:] = 1.0
    n_components, labels = csgraph.connected_components(binary, directed=False)
    component_sizes = np.bincount(labels, minlength=n_components)
    gcc_label = int(np.argmax(component_sizes))
    gcc_global = np.flatnonzero(labels == gcc_label)
    gcc_ids = [anchors[index] for index in gcc_global]
    gcc_set = set(gcc_ids)
    w_train = adjacency[gcc_global][:, gcc_global].tocsr()
    degree = np.asarray(w_train.sum(axis=1)).ravel()
    tau = float(degree.mean()) if len(degree) else 0.0
    if not math.isfinite(tau) or tau <= 0:
        raise ContractError("non-positive mean TRAIN degree")
    inverse = 1.0 / np.sqrt(degree + tau)
    normalized = sparse.diags(inverse) @ w_train @ sparse.diags(inverse)
    k = dimensions + 1
    if normalized.shape[0] <= k:
        raise ContractError("TRAIN GCC too small for spectral dimension")
    graph_cfg = contract["graph"]
    v0 = np.linspace(1.0, 2.0, normalized.shape[0], dtype=float)
    eigenvalues, eigenvectors = eigsh(
        normalized,
        k=k,
        which="LA",
        tol=float(graph_cfg["eigensolver_tolerance"]),
        maxiter=int(graph_cfg["eigensolver_maxiter"]),
        v0=v0,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][1 : dimensions + 1]
    eigenvectors = canonicalize_eigenvectors(eigenvectors[:, order][:, 1 : dimensions + 1])
    if (np.abs(eigenvalues) < float(graph_cfg["minimum_abs_eigenvalue"])).any():
        raise ContractError("Nyström eigenvalue below frozen tolerance")
    residuals = np.linalg.norm(normalized @ eigenvectors - eigenvectors * eigenvalues, axis=0)
    if (residuals > 1e-5).any():
        raise ContractError("spectral residual exceeds 1e-5")

    all_target = sorted(target_train_ids | target_eval_ids)
    features = pd.DataFrame({"sample_id": all_target})
    features["projectable"] = 0.0
    features["degree_to_train_total"] = 0.0
    features["weight_strength_to_train_total"] = 0.0
    features["bp_to_train_total"] = 0.0
    features["degree_to_gcc"] = 0.0
    features["weight_strength_to_gcc"] = 0.0
    features["bp_to_gcc"] = 0.0
    for dim in range(dimensions):
        features[f"z{dim + 1}"] = 0.0
    feature_index = {sample: index for index, sample in enumerate(all_target)}
    gcc_index = {sample: index for index, sample in enumerate(gcc_ids)}

    # TRAIN local summaries use TRAIN endpoints only.  Coordinates outside the
    # GCC remain zero with projectable=0.
    tt_bp = pd.to_numeric(tt["total_shared_bp"], errors="raise").to_numpy(dtype=float)
    train_degree: defaultdict[str, float] = defaultdict(float)
    train_strength: defaultdict[str, float] = defaultdict(float)
    train_bp: defaultdict[str, float] = defaultdict(float)
    train_gcc_degree: defaultdict[str, float] = defaultdict(float)
    train_gcc_strength: defaultdict[str, float] = defaultdict(float)
    train_gcc_bp: defaultdict[str, float] = defaultdict(float)
    for left, right, bp, weight in zip(
        tt["sample_a"], tt["sample_b"], tt_bp, tt_weights, strict=True
    ):
        train_degree[str(left)] += 1.0
        train_degree[str(right)] += 1.0
        train_strength[str(left)] += float(weight)
        train_strength[str(right)] += float(weight)
        train_bp[str(left)] += float(bp)
        train_bp[str(right)] += float(bp)
        if str(left) in gcc_set and str(right) in gcc_set:
            train_gcc_degree[str(left)] += 1.0
            train_gcc_degree[str(right)] += 1.0
            train_gcc_strength[str(left)] += float(weight)
            train_gcc_strength[str(right)] += float(weight)
            train_gcc_bp[str(left)] += float(bp)
            train_gcc_bp[str(right)] += float(bp)
    spectral_scale = math.sqrt(len(gcc_ids))
    for sample in target_train_ids:
        row = feature_index[sample]
        features.loc[row, "degree_to_train_total"] = train_degree[sample]
        features.loc[row, "weight_strength_to_train_total"] = train_strength[sample]
        features.loc[row, "bp_to_train_total"] = train_bp[sample]
        features.loc[row, "degree_to_gcc"] = train_gcc_degree[sample]
        features.loc[row, "weight_strength_to_gcc"] = train_gcc_strength[sample]
        features.loc[row, "bp_to_gcc"] = train_gcc_bp[sample]
        if sample in gcc_index:
            features.loc[row, "projectable"] = 1.0
            features.loc[row, [f"z{x + 1}" for x in range(dimensions)]] = (
                eigenvectors[gcc_index[sample]] * spectral_scale
            )

    # EVAL summaries and Nyström consume only EVAL-to-TRAIN edges.  W_EE is
    # neither constructed nor accepted by this function.
    et_left = edges["sample_a"].isin(target_eval_ids) & edges["sample_b"].isin(anchor_train_ids)
    et_right = edges["sample_b"].isin(target_eval_ids) & edges["sample_a"].isin(anchor_train_ids)
    et = pd.concat(
        [
            edges.loc[et_left].assign(eval_id=lambda x: x["sample_a"], train_id=lambda x: x["sample_b"]),
            edges.loc[et_right].assign(eval_id=lambda x: x["sample_b"], train_id=lambda x: x["sample_a"]),
        ],
        ignore_index=True,
    )
    et_bp = pd.to_numeric(et["total_shared_bp"], errors="raise").to_numpy(dtype=float)
    et_all_weights = _edge_weights(et, mode)
    eval_degree = et.groupby("eval_id", sort=False).size().to_dict()
    eval_strength = (
        et.assign(_weight=et_all_weights).groupby("eval_id", sort=False)["_weight"].sum().to_dict()
    )
    eval_bp = et.assign(_bp=et_bp).groupby("eval_id", sort=False)["_bp"].sum().to_dict()
    et_gcc = et[et["train_id"].isin(gcc_set)].copy()
    et_gcc_bp = pd.to_numeric(et_gcc["total_shared_bp"], errors="raise").to_numpy(dtype=float)
    et_gcc_weights = _edge_weights(et_gcc, mode)
    eval_gcc_degree = et_gcc.groupby("eval_id", sort=False).size().to_dict()
    eval_gcc_strength = (
        et_gcc.assign(_weight=et_gcc_weights)
        .groupby("eval_id", sort=False)["_weight"]
        .sum()
        .to_dict()
    )
    eval_gcc_bp = (
        et_gcc.assign(_bp=et_gcc_bp).groupby("eval_id", sort=False)["_bp"].sum().to_dict()
    )
    if not et_gcc.empty:
        eval_ids = sorted(set(et_gcc["eval_id"]))
        eval_index = {sample: index for index, sample in enumerate(eval_ids)}
        et_matrix = sparse.coo_matrix(
            (
                et_gcc_weights,
                (
                    et_gcc["eval_id"].map(eval_index).to_numpy(dtype=int),
                    et_gcc["train_id"].map(gcc_index).to_numpy(dtype=int),
                ),
            ),
            shape=(len(eval_ids), len(gcc_ids)),
        ).tocsr()
        eval_weight_degree = np.asarray(et_matrix.sum(axis=1)).ravel()
        normalized_et = sparse.diags(1.0 / np.sqrt(eval_weight_degree + tau)) @ et_matrix @ sparse.diags(inverse)
        projected = (normalized_et @ eigenvectors) / eigenvalues[np.newaxis, :]
        projected *= spectral_scale
        for sample in eval_ids:
            row = feature_index[sample]
            features.loc[row, "projectable"] = 1.0
            features.loc[row, [f"z{x + 1}" for x in range(dimensions)]] = projected[eval_index[sample]]
    for sample in target_eval_ids:
        row = feature_index[sample]
        features.loc[row, "degree_to_train_total"] = float(eval_degree.get(sample, 0.0))
        features.loc[row, "weight_strength_to_train_total"] = float(
            eval_strength.get(sample, 0.0)
        )
        features.loc[row, "bp_to_train_total"] = float(eval_bp.get(sample, 0.0))
        features.loc[row, "degree_to_gcc"] = float(eval_gcc_degree.get(sample, 0.0))
        features.loc[row, "weight_strength_to_gcc"] = float(
            eval_gcc_strength.get(sample, 0.0)
        )
        features.loc[row, "bp_to_gcc"] = float(eval_gcc_bp.get(sample, 0.0))

    train_missing = 1.0 - float(features[features["sample_id"].isin(target_train_ids)]["projectable"].mean())
    eval_missing = 1.0 - float(features[features["sample_id"].isin(target_eval_ids)]["projectable"].mean())
    maximum = float(graph_cfg["maximum_unprojectable_fraction"])
    within_projectability_threshold = train_missing <= maximum and eval_missing <= maximum
    if enforce_projectability_gate and not within_projectability_threshold:
        raise ContractError(
            f"GCC/projectability stop: train={train_missing:.6f}, eval={eval_missing:.6f}"
        )
    reserved = reserved_ids or set()
    reserved_fold_endpoints_used = int(
        tt["sample_a"].isin(reserved).sum()
        + tt["sample_b"].isin(reserved).sum()
        + et["eval_id"].isin(reserved).sum()
        + et["train_id"].isin(reserved).sum()
    )
    diagnostics = {
        "mode": mode,
        "projectability_stage": stage,
        "n_anchor_train": len(anchors),
        "n_train_edges": int(len(tt)),
        "n_train_components": int(n_components),
        "train_gcc_size": int(len(gcc_ids)),
        "train_gcc_fraction": float(len(gcc_ids) / len(anchors)),
        "target_train_unprojectable_fraction": train_missing,
        "target_eval_unprojectable_fraction": eval_missing,
        "projectability_gate_enforced": bool(enforce_projectability_gate),
        "projectability_within_20pct": bool(within_projectability_threshold),
        "n_eval_train_edges": int(len(et)),
        "n_eval_gcc_edges": int(len(et_gcc)),
        "tau": tau,
        "eigenvalues": ";".join(f"{value:.12g}" for value in eigenvalues),
        "maximum_residual": float(residuals.max()),
        "w_eval_eval_used": 0,
        "reserved_fold_endpoints_used": reserved_fold_endpoints_used,
    }
    return features, diagnostics


def prepare_tables(
    paths: dict[str, Path], contract: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], set[str]]:
    global_summary = json.loads(paths["global_summary"].read_text(encoding="utf-8"))
    inputs = contract["inputs"]
    if global_summary.get("carrier_allele_mode") != inputs["m14_carrier_allele_mode"]:
        raise ContractError("M14 is not minor-allele oriented")
    if int(global_summary.get("total_sharing_pairs", -1)) != int(inputs["expected_pair_rows"]):
        raise ContractError("M14 pair count drifted")
    if int(global_summary.get("n_samples", -1)) != int(inputs["expected_samples"]):
        raise ContractError("M14 sample count drifted")
    if int(global_summary.get("parameters_used", {}).get("min_segment_bp", -1)) != int(inputs["m14_min_segment_bp"]):
        raise ContractError("M14 minimum segment drifted")
    ordered_samples = [str(value).strip() for value in global_summary.get("ordered_samples", [])]
    expected_sample_ids = set(ordered_samples)
    if (
        len(ordered_samples) != int(inputs["expected_samples"])
        or len(expected_sample_ids) != int(inputs["expected_samples"])
        or any(not value for value in ordered_samples)
    ):
        raise ContractError("global ordered_samples is not the frozen unique sample universe")

    pairs = read_tsv(paths["pairs"])
    require_columns(pairs, ["sample_a", "sample_b", "n_shared_variants_total", "total_shared_bp"], "pairs")
    if len(pairs) != int(inputs["expected_pair_rows"]):
        raise ContractError("pair rows differ from frozen 54,522")
    pairs[["sample_a", "sample_b"]] = pairs[["sample_a", "sample_b"]].astype(str)
    canonical = pairs.apply(lambda row: tuple(sorted((row["sample_a"], row["sample_b"]))), axis=1)
    if canonical.duplicated().any() or (pairs["sample_a"] == pairs["sample_b"]).any():
        raise ContractError("M14 contains duplicate or self pairs")
    if not (set(pairs["sample_a"]) | set(pairs["sample_b"])).issubset(expected_sample_ids):
        raise ContractError("M14 pair endpoint is outside ordered_samples")
    if (pd.to_numeric(pairs["total_shared_bp"], errors="raise") <= 0).any():
        raise ContractError("M14 has non-positive pair length")

    split = unique_rows(read_tsv(paths["split_manifest"]), "sample_id", "split")
    require_columns(split, ["sample_id", "eligible", "fold", "split"], "split")
    if set(split["sample_id"]) != expected_sample_ids:
        raise ContractError("split sample universe differs from ordered_samples")
    split["fold"] = pd.to_numeric(split["fold"], errors="coerce")
    eligible = split["eligible"].str.lower().isin({"true", "1", "yes"})
    dev = split[eligible & split["split"].eq("TRAIN") & split["fold"].isin(DEV_FOLDS)].copy()
    dev["fold"] = dev["fold"].astype(int)
    reserved_ids = set(split.loc[split["fold"].eq(RESERVED_FOLD), "sample_id"])
    if RESERVED_FOLD in set(dev["fold"]):
        raise ContractError("reserved fold entered DEV")

    feature = unique_rows(read_tsv(paths["feature_store"]), "sample_id", "feature")
    require_columns(feature, ["sample_id", "Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR", "rare_missing_sites"], "feature")
    if set(feature["sample_id"]) != expected_sample_ids:
        raise ContractError("feature sample universe differs from ordered_samples")
    metadata = unique_rows(read_tsv(paths["metadata"]), "ID", "metadata").rename(
        columns={"ID": "sample_id", "Cohort": "metadata_cohort", "Region": "region", "State": "state"}
    )
    require_columns(metadata, ["sample_id", "metadata_cohort", "region", "state"], "metadata")
    if not expected_sample_ids.issubset(set(metadata["sample_id"])):
        raise ContractError("metadata does not cover ordered_samples")
    burden = unique_rows(read_tsv(paths["burden"]), "sample_id", "burden")
    burden_column = inputs["burden_column"]
    require_columns(burden, ["sample_id", burden_column], "burden")
    if set(burden["sample_id"]) != expected_sample_ids:
        raise ContractError("burden sample universe differs from ordered_samples")
    modeling = unique_rows(read_tsv(paths["modeling_master"]), "sample_id", "modeling")
    group_column = contract["population"]["kinship_group_column"]
    require_columns(modeling, ["sample_id", group_column], "modeling")
    if set(modeling["sample_id"]) != expected_sample_ids:
        raise ContractError("modeling sample universe differs from ordered_samples")

    samples = (
        dev[["sample_id", "fold"]]
        .merge(feature[["sample_id", "Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR", "rare_missing_sites"]], on="sample_id", validate="one_to_one")
        .merge(metadata[["sample_id", "metadata_cohort", "region", "state"]], on="sample_id", validate="one_to_one")
        .merge(burden[["sample_id", burden_column]], on="sample_id", validate="one_to_one")
        .merge(modeling[["sample_id", group_column]], on="sample_id", validate="one_to_one")
    )
    if len(samples) != len(dev):
        raise ContractError("DEV join changed cardinality")
    numeric = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR", "rare_missing_sites", burden_column]
    for column in numeric:
        samples[column] = pd.to_numeric(samples[column], errors="coerce")
    if samples[numeric + ["metadata_cohort", "region", group_column]].isna().any().any():
        raise ContractError("DEV join contains missing required values")
    samples = samples.rename(columns={burden_column: "burden", group_column: "kinship_group"})
    samples["region"] = samples["region"].astype(str).str.strip().str.upper()
    samples["state"] = samples["state"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    samples.loc[samples["state"].isin({"", "NAN"}), "state"] = "UNKNOWN"
    population = contract["population"]
    analysis_mask = (
        samples["metadata_cohort"].astype(str).str.upper().eq(population["cohort"])
        & samples["Q_EUR"].gt(float(population["q_eur_strictly_greater_than"]))
    )
    target = samples[
        analysis_mask
        & ~samples["region"].isin(set(population["excluded_region_labels"]))
    ].copy()
    counts = target["region"].value_counts()
    observed_primary = sorted(
        counts[counts >= int(population["minimum_people_per_primary_region"])].index
    )
    classes = sorted(str(value) for value in population["primary_regions"])
    if observed_primary != classes:
        raise ContractError(
            f"primary regions drifted: observed={observed_primary}, frozen={classes}"
        )
    target = target[target["region"].isin(classes)].copy()
    if len(target) != int(population["expected_primary_target_samples"]):
        raise ContractError(
            f"primary target count drifted: {len(target)} != "
            f"{population['expected_primary_target_samples']}"
        )
    observed_region_counts = {
        str(label): int(value) for label, value in target["region"].value_counts().items()
    }
    expected_region_counts = {
        str(label): int(value)
        for label, value in population["expected_primary_region_counts"].items()
    }
    if observed_region_counts != expected_region_counts:
        raise ContractError(
            f"primary region counts drifted: {observed_region_counts} != {expected_region_counts}"
        )
    observed_fold_counts = {
        str(int(label)): int(value) for label, value in target["fold"].value_counts().items()
    }
    expected_fold_counts = {
        str(label): int(value) for label, value in population["expected_primary_fold_counts"].items()
    }
    if observed_fold_counts != expected_fold_counts:
        raise ContractError(
            f"primary fold counts drifted: {observed_fold_counts} != {expected_fold_counts}"
        )
    observed_fold_region = {
        str(fold): {
            region: int(
                ((target["fold"].eq(int(fold))) & target["region"].eq(region)).sum()
            )
            for region in classes
        }
        for fold in DEV_FOLDS
    }
    expected_fold_region = {
        str(fold): {str(region): int(value) for region, value in counts.items()}
        for fold, counts in population["expected_primary_fold_region_counts"].items()
    }
    if observed_fold_region != expected_fold_region:
        raise ContractError(
            f"primary fold-region counts drifted: {observed_fold_region} != {expected_fold_region}"
        )
    if len(classes) < 3:
        raise ContractError("fewer than three primary regions")
    for fold in DEV_FOLDS:
        if set(target.loc[target["fold"].eq(fold), "region"]) != set(classes):
            raise ContractError(f"outer fold {fold} lacks a primary region")
        inner_train_folds = set(DEV_FOLDS) - {fold}
        for inner in inner_train_folds:
            training = target[target["fold"].isin(inner_train_folds - {inner})]
            if set(training["region"]) != set(classes):
                raise ContractError(f"nested TRAIN outer={fold} inner={inner} lacks a class")
    samples["is_target"] = samples["sample_id"].isin(set(target["sample_id"]))
    if int(samples["is_target"].sum()) != int(population["expected_primary_target_samples"]):
        raise ContractError("frozen target indicator cardinality drifted")
    return samples, pairs, classes, reserved_ids


def select_target_rows(samples: pd.DataFrame) -> pd.DataFrame:
    """Return only the explicitly frozen RHT/Q_EUR/primary-region target rows."""
    require_columns(samples, ["is_target"], "samples")
    if not samples["is_target"].isin([True, False]).all():
        raise ContractError("is_target is not boolean")
    return samples[samples["is_target"]].copy()


def build_design(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    level: str,
    dimension: int,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    model_cfg = contract["model"]
    q_columns = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR"]
    train_ilr = ilr_transform(train[q_columns].to_numpy(float), float(model_cfg["q_ilr_zero_replacement"]))
    eval_ilr = ilr_transform(evaluation[q_columns].to_numpy(float), float(model_cfg["q_ilr_zero_replacement"]))
    scalar_columns = ["burden", "log1p_missing"]
    if level in {"B1", "A"}:
        scalar_columns += ["log1p_degree_to_gcc", "log1p_weight_strength_to_gcc"]
    spline = SplineTransformer(
        degree=int(model_cfg["spline_degree"]),
        n_knots=int(model_cfg["spline_n_knots"]),
        include_bias=False,
    )
    train_spline = spline.fit_transform(train[scalar_columns].to_numpy(float))
    eval_spline = spline.transform(evaluation[scalar_columns].to_numpy(float))
    train_parts = [train_ilr, train_spline]
    eval_parts = [eval_ilr, eval_spline]
    if level in {"B1", "A"}:
        train_parts.append(train[["projectable"]].to_numpy(float))
        eval_parts.append(evaluation[["projectable"]].to_numpy(float))
    train_base = np.column_stack(train_parts)
    eval_base = np.column_stack(eval_parts)
    scaler = StandardScaler()
    train_base = scaler.fit_transform(train_base)
    eval_base = scaler.transform(eval_base)
    if level == "A":
        z_columns = [f"z{x + 1}" for x in range(dimension)]
        train_z = train[z_columns].to_numpy(float)
        eval_z = evaluation[z_columns].to_numpy(float)
        center = train_z.mean(axis=0, keepdims=True)
        train_base = np.column_stack([train_base, train_z - center])
        eval_base = np.column_stack([eval_base, eval_z - center])
    return train_base, eval_base


def fit_predict(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    level: str,
    dimension: int,
    C: float,
    classes: list[str],
    contract: dict[str, Any],
) -> np.ndarray:
    x_train, x_eval = build_design(train, evaluation, level, dimension, contract)
    cfg = contract["model"]
    model = LogisticRegression(
        C=float(C),
        l1_ratio=0.0,
        solver="lbfgs",
        class_weight=str(cfg["class_weight"]),
        max_iter=int(cfg["max_iter"]),
        tol=float(cfg["tolerance"]),
        random_state=0,
    )
    model.fit(x_train, train["region"].to_numpy(object))
    if int(np.max(model.n_iter_)) >= int(cfg["max_iter"]):
        raise ContractError(f"{level} did not converge at C={C}")
    return align_probabilities(model, x_eval, classes)


def attach_graph_features(base: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    result = base.merge(graph, on="sample_id", how="left", validate="one_to_one")
    if result.isna().any().any():
        raise ContractError("graph feature join contains missing values")
    result["log1p_missing"] = np.log1p(result["rare_missing_sites"].astype(float))
    result["log1p_degree_to_gcc"] = np.log1p(result["degree_to_gcc"].astype(float))
    result["log1p_weight_strength_to_gcc"] = np.log1p(
        result["weight_strength_to_gcc"].astype(float)
    )
    # Whole-TRAIN totals and raw bp remain audit/descriptive quantities. B1/A
    # use only the GCC-matched degree and exact graph-weight strength above.
    return result


def candidate_key(level: str, C: float, dimension: int = 0) -> tuple[str, float, int]:
    return level, float(C), int(dimension)


def select_one_standard_error(
    fold_scores: dict[tuple[str, float, int], list[float]],
    level: str,
) -> tuple[tuple[str, float, int], float]:
    """Choose strongest ridge/smallest dimension within one SE of the best mean."""
    candidates = [key for key in fold_scores if key[0] == level]
    if not candidates:
        raise ContractError(f"no inner candidates for {level}")
    summaries: dict[tuple[str, float, int], tuple[float, float]] = {}
    for key in candidates:
        values = np.asarray(fold_scores[key], dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ContractError(f"invalid inner-fold scores for {key}")
        summaries[key] = (float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values))))
    best = min(candidates, key=lambda key: (summaries[key][0], key[2], key[1]))
    threshold = summaries[best][0] + summaries[best][1]
    eligible = [key for key in candidates if summaries[key][0] <= threshold + 1e-15]
    selected = min(eligible, key=lambda key: (key[1], key[2]))
    return selected, threshold


def nested_outer(
    samples: pd.DataFrame,
    edges: pd.DataFrame,
    classes: list[str],
    outer_fold: int,
    mode: str,
    contract: dict[str, Any],
    reserved_ids: set[str],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[pd.DataFrame],
]:
    dimensions = [int(value) for value in contract["graph"]["dimensions"]]
    c_grid = [float(value) for value in contract["model"]["C_grid"]]
    dmax = max(dimensions)
    target = select_target_rows(samples)
    outer_train_folds = set(DEV_FOLDS) - {outer_fold}
    inner_predictions: defaultdict[tuple[str, float, int], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    graph_rows: list[dict[str, Any]] = []
    graph_feature_rows: list[pd.DataFrame] = []
    for inner_fold in sorted(outer_train_folds):
        anchor_folds = outer_train_folds - {inner_fold}
        anchor_mask = samples["fold"].isin(anchor_folds)
        anchor_ids = set(samples.loc[anchor_mask, "sample_id"])
        train_base = target[target["fold"].isin(anchor_folds)].copy()
        eval_base = target[target["fold"].eq(inner_fold)].copy()
        graph, diagnostic = spectral_fold_features(
            edges,
            anchor_ids,
            set(train_base["sample_id"]),
            set(eval_base["sample_id"]),
            mode,
            dmax,
            contract,
            reserved_ids,
            stage="inner",
            enforce_projectability_gate=False,
        )
        diagnostic.update({"stage": "inner", "outer_fold": outer_fold, "eval_fold": inner_fold})
        graph_rows.append(diagnostic)
        graph_audit = graph.copy()
        graph_audit["mode"] = mode
        graph_audit["stage"] = "inner"
        graph_audit["outer_fold"] = outer_fold
        graph_audit["eval_fold"] = inner_fold
        graph_audit["role"] = np.where(
            graph_audit["sample_id"].isin(set(train_base["sample_id"])), "TRAIN", "EVAL"
        )
        graph_feature_rows.append(graph_audit)
        train = attach_graph_features(train_base, graph)
        evaluation = attach_graph_features(eval_base, graph)
        for C in c_grid:
            for level in ("B0", "B1"):
                p = fit_predict(train, evaluation, level, 0, C, classes, contract)
                inner_predictions[candidate_key(level, C)].append((evaluation["region"].to_numpy(object), p))
            for dimension in dimensions:
                p = fit_predict(train, evaluation, "A", dimension, C, classes, contract)
                inner_predictions[candidate_key("A", C, dimension)].append((evaluation["region"].to_numpy(object), p))

    clip = float(contract["model"]["probability_clip"])
    selection_rows: list[dict[str, Any]] = []
    fold_scores: dict[tuple[str, float, int], list[float]] = {}
    pooled_scores: dict[tuple[str, float, int], float] = {}
    for key, chunks in inner_predictions.items():
        y = np.concatenate([chunk[0] for chunk in chunks])
        p = np.vstack([chunk[1] for chunk in chunks])
        values = [macro_log_loss(chunk_y, chunk_p, classes, clip) for chunk_y, chunk_p in chunks]
        fold_scores[key] = values
        pooled_scores[key] = macro_log_loss(y, p, classes, clip)
        selection_rows.append(
            {
                "mode": mode,
                "outer_fold": outer_fold,
                "model": key[0],
                "C": key[1],
                "dimension": key[2],
                "inner_mean_macro_log_loss": float(np.mean(values)),
                "inner_se_macro_log_loss": float(np.std(values, ddof=1) / math.sqrt(len(values))),
                "inner_pooled_macro_log_loss": pooled_scores[key],
            }
        )
    selected: dict[str, tuple[str, float, int]] = {}
    thresholds: dict[str, float] = {}
    for level in MODEL_LEVELS:
        selected[level], thresholds[level] = select_one_standard_error(fold_scores, level)
    for row in selection_rows:
        key = candidate_key(row["model"], row["C"], row["dimension"])
        row["one_se_threshold"] = thresholds[row["model"]]
        row["within_one_se"] = row["inner_mean_macro_log_loss"] <= thresholds[row["model"]] + 1e-15
        row["selected"] = key == selected[row["model"]]

    anchor_mask = samples["fold"].isin(outer_train_folds)
    anchor_ids = set(samples.loc[anchor_mask, "sample_id"])
    train_base = target[target["fold"].isin(outer_train_folds)].copy()
    eval_base = target[target["fold"].eq(outer_fold)].copy()
    graph, diagnostic = spectral_fold_features(
        edges,
        anchor_ids,
        set(train_base["sample_id"]),
        set(eval_base["sample_id"]),
        mode,
        dmax,
        contract,
        reserved_ids,
        stage="outer",
        enforce_projectability_gate=True,
    )
    diagnostic.update({"stage": "outer", "outer_fold": outer_fold, "eval_fold": outer_fold})
    graph_rows.append(diagnostic)
    graph_audit = graph.copy()
    graph_audit["mode"] = mode
    graph_audit["stage"] = "outer"
    graph_audit["outer_fold"] = outer_fold
    graph_audit["eval_fold"] = outer_fold
    graph_audit["role"] = np.where(
        graph_audit["sample_id"].isin(set(train_base["sample_id"])), "TRAIN", "EVAL"
    )
    graph_feature_rows.append(graph_audit)
    train = attach_graph_features(train_base, graph)
    evaluation = attach_graph_features(eval_base, graph)
    predictions: dict[str, np.ndarray] = {}
    for level in MODEL_LEVELS:
        _, C, dimension = selected[level]
        predictions[level] = fit_predict(train, evaluation, level, dimension, C, classes, contract)
    output = evaluation[
        [
            "sample_id",
            "fold",
            "region",
            "kinship_group",
            "projectable",
            "degree_to_train_total",
            "weight_strength_to_train_total",
            "bp_to_train_total",
            "degree_to_gcc",
            "weight_strength_to_gcc",
            "bp_to_gcc",
        ]
    ].copy()
    output["mode"] = mode
    for level in MODEL_LEVELS:
        output[f"selected_C_{level}"] = selected[level][1]
        output[f"selected_d_{level}"] = selected[level][2]
        output[f"loss_{level}"] = true_class_loss(
            output["region"].to_numpy(object), predictions[level], classes, clip
        )
        for index, label in enumerate(classes):
            output[f"p_{level}_{label}"] = predictions[level][:, index]
    metrics = []
    for level in MODEL_LEVELS:
        metrics.append(
            {
                "mode": mode,
                "outer_fold": outer_fold,
                "model": level,
                "macro_log_loss": macro_log_loss(
                    output["region"].to_numpy(object), predictions[level], classes, clip
                ),
                "balanced_accuracy": balanced_accuracy(
                    output["region"].to_numpy(object), predictions[level], classes
                ),
                "macro_brier": macro_brier(
                    output["region"].to_numpy(object), predictions[level], classes
                ),
                "selected_C": selected[level][1],
                "selected_dimension": selected[level][2],
                "n_eval": len(output),
            }
        )
    return output, selection_rows, metrics, graph_rows, graph_feature_rows


def has_all_classes(frame: pd.DataFrame, classes: list[str]) -> bool:
    return set(frame["region"].astype(str)) == set(classes)


def macro_delta_from_losses(frame: pd.DataFrame, classes: list[str]) -> float:
    if not has_all_classes(frame, classes):
        raise ContractError("macro delta lacks a frozen class")
    baseline = np.mean(
        [frame.loc[frame["region"].eq(label), "loss_B1"].mean() for label in classes]
    )
    augmented = np.mean(
        [frame.loc[frame["region"].eq(label), "loss_A"].mean() for label in classes]
    )
    return float(augmented - baseline)


def evaluation_metric_table(
    oof: pd.DataFrame, classes: list[str], clip: float
) -> pd.DataFrame:
    """Metrics by graph mode, outer fold/pooled and projectability estimand."""
    rows: list[dict[str, Any]] = []
    for mode in ("binary", "log_length"):
        mode_frame = oof[oof["mode"].eq(mode)]
        scopes: list[tuple[str | int, pd.DataFrame]] = [
            *[(fold, mode_frame[mode_frame["fold"].eq(fold)]) for fold in DEV_FOLDS],
            ("POOLED", mode_frame),
        ]
        for outer_fold, scope_frame in scopes:
            subsets = {
                "ALL": scope_frame,
                "PROJECTABLE_ONLY": scope_frame[scope_frame["projectable"].eq(1.0)],
                "NONPROJECTABLE": scope_frame[scope_frame["projectable"].eq(0.0)],
            }
            for subset, frame in subsets.items():
                if subset == "ALL" and not has_all_classes(frame, classes):
                    raise ContractError(
                        f"{mode}/{outer_fold}/{subset} lacks a frozen class"
                    )
                if (
                    subset == "PROJECTABLE_ONLY"
                    and outer_fold == "POOLED"
                    and not has_all_classes(frame, classes)
                ):
                    raise ContractError(f"{mode}/POOLED/PROJECTABLE_ONLY lacks a frozen class")
                if subset == "PROJECTABLE_ONLY" and not has_all_classes(frame, classes):
                    continue
                if subset == "NONPROJECTABLE" and not has_all_classes(frame, classes):
                    continue
                delta = macro_delta_from_losses(frame, classes)
                for level in MODEL_LEVELS:
                    probabilities = frame[
                        [f"p_{level}_{label}" for label in classes]
                    ].to_numpy(float)
                    row = {
                        "mode": mode,
                        "outer_fold": outer_fold,
                        "subset": subset,
                        "model": level,
                        "macro_log_loss": macro_log_loss(
                            frame["region"].to_numpy(object), probabilities, classes, clip
                        ),
                        "balanced_accuracy": balanced_accuracy(
                            frame["region"].to_numpy(object), probabilities, classes
                        ),
                        "macro_brier": macro_brier(
                            frame["region"].to_numpy(object), probabilities, classes
                        ),
                        "n_eval": int(len(frame)),
                        "delta_A_minus_B1": delta,
                    }
                    if outer_fold == "POOLED":
                        row.update({"selected_C": np.nan, "selected_dimension": np.nan})
                    else:
                        row.update(
                            {
                                "selected_C": float(frame[f"selected_C_{level}"].iloc[0]),
                                "selected_dimension": int(
                                    frame[f"selected_d_{level}"].iloc[0]
                                ),
                            }
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def region_metric_table(binary: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "ALL": binary,
        "PROJECTABLE_ONLY": binary[binary["projectable"].eq(1.0)],
        "NONPROJECTABLE": binary[binary["projectable"].eq(0.0)],
    }
    for subset, subset_frame in subsets.items():
        if subset in {"ALL", "PROJECTABLE_ONLY"} and not has_all_classes(subset_frame, classes):
            raise ContractError(f"binary/{subset} lacks a frozen class")
        if subset == "NONPROJECTABLE" and not has_all_classes(subset_frame, classes):
            continue
        for label in classes:
            frame = subset_frame[subset_frame["region"].eq(label)]
            baseline = float(frame["loss_B1"].mean())
            augmented = float(frame["loss_A"].mean())
            rows.append(
                {
                    "subset": subset,
                    "region": label,
                    "n": int(len(frame)),
                    "B1_log_loss": baseline,
                    "A_log_loss": augmented,
                    "delta_A_minus_B1": augmented - baseline,
                    "relative_change": (augmented - baseline) / baseline,
                }
            )
    return pd.DataFrame(rows)


def projectability_count_table(binary: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_label, frame in [
        *[(str(fold), binary[binary["fold"].eq(fold)]) for fold in DEV_FOLDS],
        ("POOLED", binary),
    ]:
        for region in classes:
            region_frame = frame[frame["region"].eq(region)]
            denominator = int(len(region_frame))
            for projectable in (0, 1):
                n = int(region_frame["projectable"].eq(float(projectable)).sum())
                rows.append(
                    {
                        "fold": fold_label,
                        "region": region,
                        "projectable": projectable,
                        "n": n,
                        "region_fold_denominator": denominator,
                        "fraction": float(n / denominator) if denominator else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def projectability_coverage_gates(
    counts: pd.DataFrame, classes: list[str]
) -> dict[str, bool]:
    outer = counts[
        counts["fold"].isin([str(fold) for fold in DEV_FOLDS])
        & counts["projectable"].eq(1)
    ]
    complete = len(outer) == len(DEV_FOLDS) * len(classes) and outer["n"].gt(0).all()
    southern = outer[outer["region"].eq("SOUTHERN")]
    return {
        "all_regions_projectable_in_each_outer_fold": bool(complete),
        "southern_projectable_in_each_outer_fold": bool(
            len(southern) == len(DEV_FOLDS) and southern["n"].gt(0).all()
        ),
    }


def cluster_bootstrap(
    oof: pd.DataFrame,
    classes: list[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired component bootstrap; macro loss is recomputed in every replicate."""
    groups = sorted(oof["kinship_group"].astype(str).unique())
    by_group = {group: oof[oof["kinship_group"].astype(str).eq(group)] for group in groups}
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        chunks = [by_group[group] for group in sampled]
        frame = pd.concat(chunks, ignore_index=True)
        if set(frame["region"]) != set(classes):
            continue
        baseline = float(np.mean([frame.loc[frame["region"].eq(label), "loss_B1"].mean() for label in classes]))
        augmented = float(np.mean([frame.loc[frame["region"].eq(label), "loss_A"].mean() for label in classes]))
        rows.append({"replicate": replicate, "delta_A_minus_B1": augmented - baseline})
    result = pd.DataFrame(rows)
    if len(result) < math.ceil(0.99 * replicates):
        raise ContractError("too many bootstrap replicates lost a class")
    return result


def write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, sep="\t", index=False, lineterminator="\n")


def run(args: argparse.Namespace) -> None:
    contract = load_contract(Path(args.preregistration))
    paths = {
        "pairs": Path(args.pairs),
        "global_summary": Path(args.global_summary),
        "metadata": Path(args.metadata),
        "burden": Path(args.burden),
        "feature_store": Path(args.feature_store),
        "modeling_master": Path(args.modeling_master),
        "split_manifest": Path(args.split_manifest),
    }
    for path in [*paths.values(), Path(args.preflight_report)]:
        if not path.is_file():
            raise ContractError(f"missing input: {path}")
    hashes = verify_hashes(paths, contract)
    preflight = json.loads(Path(args.preflight_report).read_text(encoding="utf-8"))
    validate_preflight(preflight, hashes, contract)

    samples, edges, classes, reserved_ids = prepare_tables(paths, contract)
    target_samples = select_target_rows(samples)
    region_state_counts = (
        target_samples.groupby(["region", "state"], dropna=False, sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    oof_rows: list[pd.DataFrame] = []
    inner_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    graph_feature_rows: list[pd.DataFrame] = []
    for mode in ("binary", "log_length"):
        for outer_fold in DEV_FOLDS:
            oof, selection, metrics, diagnostics, graph_features = nested_outer(
                samples, edges, classes, outer_fold, mode, contract, reserved_ids
            )
            oof_rows.append(oof)
            inner_rows.extend(selection)
            graph_rows.extend(diagnostics)
            graph_feature_rows.extend(graph_features)
    oof = pd.concat(oof_rows, ignore_index=True).sort_values(["mode", "fold", "sample_id"])
    clip = float(contract["model"]["probability_clip"])
    metrics_output = evaluation_metric_table(oof, classes, clip)
    binary = oof[oof["mode"].eq("binary")].copy()
    binary_projectable = binary[binary["projectable"].eq(1.0)].copy()
    region_metrics = region_metric_table(binary, classes)
    region_projectability_counts = projectability_count_table(binary, classes)
    projectability_gates = projectability_coverage_gates(
        region_projectability_counts, classes
    )
    eval_cfg = contract["evaluation"]
    bootstrap = cluster_bootstrap(
        binary,
        classes,
        int(eval_cfg["bootstrap_replicates"]),
        int(eval_cfg["bootstrap_seed"]),
    )
    alpha = 1.0 - float(eval_cfg["confidence_level"])
    ci_lower, ci_upper = np.quantile(
        bootstrap["delta_A_minus_B1"].to_numpy(float), [alpha / 2, 1 - alpha / 2]
    )
    binary_outer = metrics_output[
        metrics_output["mode"].eq("binary")
        & metrics_output["subset"].eq("ALL")
        & metrics_output["model"].eq("A")
        & ~metrics_output["outer_fold"].astype(str).eq("POOLED")
    ].copy()
    binary_projectable_outer = metrics_output[
        metrics_output["mode"].eq("binary")
        & metrics_output["subset"].eq("PROJECTABLE_ONLY")
        & metrics_output["model"].eq("A")
        & ~metrics_output["outer_fold"].astype(str).eq("POOLED")
    ].copy()
    binary_projectable_delta = macro_delta_from_losses(binary_projectable, classes)
    binary_all_delta = macro_delta_from_losses(binary, classes)
    weighted = oof[oof["mode"].eq("log_length")]
    weighted_delta = macro_delta_from_losses(weighted, classes)
    all_region_metrics = region_metrics[region_metrics["subset"].eq("ALL")]
    decision_cfg = contract["decision"]
    fold3_absent = bool(not oof["fold"].eq(RESERVED_FOLD).any())
    graph_diagnostics = pd.DataFrame(graph_rows)
    reserved_endpoints_used = int(graph_diagnostics["reserved_fold_endpoints_used"].sum())
    outer_graph_diagnostics = graph_diagnostics[graph_diagnostics["stage"].eq("outer")]
    inner_graph_diagnostics = graph_diagnostics[graph_diagnostics["stage"].eq("inner")]
    gates = {
        "binary_delta_negative_all_four_outer_folds": bool(
            len(binary_outer) == 4 and binary_outer["delta_A_minus_B1"].lt(0).all()
        ),
        "cluster_bootstrap_upper_below_zero": bool(ci_upper < 0),
        "no_region_more_than_5pct_worse": bool(
            all_region_metrics["relative_change"].le(
                float(decision_cfg["maximum_relative_worsening_in_any_region"])
            ).all()
        ),
        "log_length_sensitivity_no_sign_reversal": bool(weighted_delta < 0),
        "binary_projectable_pooled_same_negative_direction": bool(
            binary_projectable_delta < 0 and binary_all_delta < 0
        ),
        "binary_projectable_pooled_has_all_regions": has_all_classes(
            binary_projectable, classes
        ),
        "outer_projectability_at_most_20pct": bool(
            len(outer_graph_diagnostics) == 2 * len(DEV_FOLDS)
            and outer_graph_diagnostics["projectability_gate_enforced"].eq(True).all()
            and outer_graph_diagnostics["projectability_within_20pct"].eq(True).all()
        ),
        "fold3_absent": fold3_absent,
        "reserved_fold_endpoints_absent": reserved_endpoints_used == 0,
        "graph_nulls_not_run_in_phase1": True,
    }
    load_bearing = [
        "binary_delta_negative_all_four_outer_folds",
        "cluster_bootstrap_upper_below_zero",
        "no_region_more_than_5pct_worse",
        "log_length_sensitivity_no_sign_reversal",
        "binary_projectable_pooled_same_negative_direction",
        "binary_projectable_pooled_has_all_regions",
        "outer_projectability_at_most_20pct",
        "fold3_absent",
        "reserved_fold_endpoints_absent",
        "graph_nulls_not_run_in_phase1",
    ]
    passed = all(gates[key] for key in load_bearing)
    decision = decision_cfg["pass_label"] if passed else decision_cfg["stop_label"]
    report = {
        "schema_version": "p1a-continuous-structure-dev-v1",
        "decision": decision,
        "scope": "DEV transductive regional association; not independent biological validation",
        "preregistration_amendment": "P1A-PRE-A1-OUTER-PROJECTABILITY",
        "aborted_run_before_amendment": {
            "run_id": "p1a-continuous-dev-20260819a",
            "status": "technical_abort_before_model_fit",
            "outcomes_observed": False,
        },
        "code_commit": args.code_commit,
        "classes": classes,
        "n_target": int(len(binary)),
        "primary_estimands": {
            "binary_ALL_n": int(len(binary)),
            "binary_PROJECTABLE_ONLY_n": int(len(binary_projectable)),
            "binary_NONPROJECTABLE_n": int(len(binary) - len(binary_projectable)),
            "binary_ALL_delta_A_minus_B1": binary_all_delta,
            "binary_PROJECTABLE_ONLY_delta_A_minus_B1": binary_projectable_delta,
        },
        "region_state_counts": {
            str(region): {
                str(row.state): int(row.n)
                for row in region_state_counts[region_state_counts["region"].eq(region)].itertuples(index=False)
            }
            for region in classes
        },
        "n_fold3_rows": int(oof["fold"].eq(RESERVED_FOLD).sum()),
        "reserved_fold_endpoints_used_across_graph_builds": reserved_endpoints_used,
        "primary_graph": "binary all 54,522 M14-minor pairs",
        "sensitivity_graph": (
            "same all-DNABR TRAIN anchors, log1p(total_shared_bp/1Mb) weights only; "
            "cannot rescue binary failure"
        ),
        "binary_outer_delta_A_minus_B1": {
            str(int(row.outer_fold)): float(row.delta_A_minus_B1)
            for row in binary_outer.itertuples(index=False)
        },
        "binary_projectable_outer_delta_A_minus_B1_descriptive": {
            str(int(row.outer_fold)): float(row.delta_A_minus_B1)
            for row in binary_projectable_outer.itertuples(index=False)
        },
        "cluster_bootstrap": {
            "requested_replicates": int(eval_cfg["bootstrap_replicates"]),
            "valid_replicates": int(len(bootstrap)),
            "confidence_level": float(eval_cfg["confidence_level"]),
            "delta_ci_lower": float(ci_lower),
            "delta_ci_upper": float(ci_upper),
        },
        "log_length_pooled_delta_A_minus_B1": weighted_delta,
        "projectability_coverage_diagnostics_not_load_bearing": projectability_gates,
        "inner_projectability_diagnostic": {
            "n_splits": int(len(inner_graph_diagnostics)),
            "n_above_20pct": int(
                (~inner_graph_diagnostics["projectability_within_20pct"]).sum()
            ),
            "maximum_eval_unprojectable_fraction": float(
                inner_graph_diagnostics["target_eval_unprojectable_fraction"].max()
            ),
            "threshold_enforced": False,
        },
        "southern_projectability_by_fold": {
            str(fold): {
                "n_projectable": int(
                    region_projectability_counts.loc[
                        region_projectability_counts["fold"].eq(str(fold))
                        & region_projectability_counts["region"].eq("SOUTHERN")
                        & region_projectability_counts["projectable"].eq(1),
                        "n",
                    ].iloc[0]
                ),
                "denominator": int(
                    region_projectability_counts.loc[
                        region_projectability_counts["fold"].eq(str(fold))
                        & region_projectability_counts["region"].eq("SOUTHERN")
                        & region_projectability_counts["projectable"].eq(1),
                        "region_fold_denominator",
                    ].iloc[0]
                ),
            }
            for fold in DEV_FOLDS
        },
        "gates": gates,
        "primary_metric": "macro_log_loss",
        "secondary_metrics": {
            "names": ["balanced_accuracy", "macro_brier"],
            "role": "descriptive_only_not_used_in_decision_gates",
        },
        "input_sha256": hashes,
        "preflight_report_sha256": sha256_file(Path(args.preflight_report)),
        "preregistration_sha256": sha256_file(Path(args.preregistration)),
        "runner_sha256": sha256_file(Path(__file__)),
        "limitations": [
            "Region is a recruitment/geographic label, not independent biological truth.",
            "M14-minor was ascertained and oriented using the full cohort; this is transductive DEV.",
            "No degree/block-preserving graph null was run in this stage.",
            "Run p1a-continuous-dev-20260819a aborted on the pre-amendment inner gate before model fit and produced no outcome metrics.",
            "A PASS authorizes only a separate null-design PRE, not a biological claim or TEST opening.",
        ],
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "p1a_dev_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(inner_rows).sort_values(
        ["mode", "outer_fold", "model", "dimension", "C"]
    ).to_csv(output / "p1a_inner_selection.tsv", sep="\t", index=False)
    metrics_output.sort_values(
        ["mode", "outer_fold", "subset", "model"], key=lambda x: x.astype(str)
    ).to_csv(
        output / "p1a_outer_metrics.tsv", sep="\t", index=False
    )
    region_metrics.to_csv(output / "p1a_region_metrics.tsv", sep="\t", index=False)
    region_state_counts.to_csv(output / "p1a_region_state_counts.tsv", sep="\t", index=False)
    region_projectability_counts.to_csv(
        output / "p1a_region_projectability_counts.tsv", sep="\t", index=False
    )
    graph_diagnostics.sort_values(["mode", "stage", "outer_fold", "eval_fold"]).to_csv(
        output / "p1a_graph_diagnostics.tsv", sep="\t", index=False
    )
    graph_features = pd.concat(graph_feature_rows, ignore_index=True).sort_values(
        ["mode", "stage", "outer_fold", "eval_fold", "role", "sample_id"]
    )
    write_deterministic_gzip(graph_features, output / "p1a_graph_features.tsv.gz")
    write_deterministic_gzip(oof, output / "p1a_oof_predictions.tsv.gz")
    write_deterministic_gzip(bootstrap, output / "p1a_cluster_bootstrap.tsv.gz")
    missing_outputs = [name for name in contract["outputs"] if not (output / name).is_file()]
    if missing_outputs:
        raise ContractError(f"declared outputs were not written: {missing_outputs}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "pairs",
        "global_summary",
        "metadata",
        "burden",
        "feature_store",
        "modeling_master",
        "split_manifest",
        "preflight_report",
        "preregistration",
    ):
        result.add_argument("--" + name.replace("_", "-"), required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--code-commit", required=True)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
