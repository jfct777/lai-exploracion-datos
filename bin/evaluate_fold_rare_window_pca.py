#!/usr/bin/env python3
"""Evaluate fold-specific PCA of rare-minor-carrier window profiles.

PCA, window means and burden regressions are fitted only on each outer FIT
partition and projected to its held-out TRAIN fold.  Q, cohort and rare-site
callability are joined only after OOF predictions exist and are diagnostic,
never inputs to PCA or rank selection.  The rank grid is a capacity curve; this
stage deliberately does not choose the rank with minimum reconstruction error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

from build_rare_window_features import deterministic_gzip_writer, sha256_file, write_json


FOLD_PATTERN = re.compile(r"\.fold(\d+)\.sample_window_features\.tsv\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", required=True)
    parser.add_argument("--fold-qc", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--train-covariates-audit", required=True)
    parser.add_argument("--train-covariates-manifest", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--outer-folds", default="0,1,2,4")
    parser.add_argument("--ranks", default="1,2,4,8,16,32")
    parser.add_argument("--primary-rank", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="chr22")
    return parser.parse_args()


def parse_int_grid(raw: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw.split(",") if value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated integers") from exc
    if not values or any(value < 1 for value in values) or tuple(sorted(set(values))) != values:
        raise SystemExit(f"{name} must be unique, positive and increasing")
    return values


def paths_by_fold(paths: list[str], expected: tuple[int, ...], pattern: re.Pattern) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in paths:
        path = Path(raw)
        match = pattern.search(path.name)
        if not match:
            raise SystemExit(f"cannot extract fold from {path.name}")
        fold = int(match.group(1))
        if fold in result:
            raise SystemExit(f"duplicate input for fold {fold}")
        result[fold] = path
    if tuple(sorted(result)) != tuple(sorted(expected)):
        raise SystemExit(f"feature folds {sorted(result)} != expected {expected}")
    return result


def load_split(path: str | Path, expected_folds: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"sample_id": str, "split_group_key": str})
    required = {"sample_id", "split", "fold", "split_group_key", "cohort"}
    if not required.issubset(frame.columns) or frame["sample_id"].duplicated().any():
        raise SystemExit("invalid split manifest")
    train = frame.loc[frame["split"] == "TRAIN"].copy()
    train["fold"] = train["fold"].astype(int)
    if tuple(sorted(train["fold"].unique())) != tuple(sorted(expected_folds)):
        raise SystemExit("TRAIN fold set differs from contract")
    if len(train) != 2091 or int((frame["split"] == "TEST").sum()) != 522:
        raise SystemExit("canonical TRAIN/TEST cardinality gate failed")
    if train["split_group_key"].isna().any() or train["split_group_key"].str.strip().eq("").any():
        raise SystemExit("TRAIN contains a missing/empty kinship group")
    if int(train.groupby("split_group_key")["fold"].nunique().max()) != 1:
        raise SystemExit("a kinship group crosses outer folds")
    return train


def load_diagnostic_covariates(path: str | Path, train_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    columns = ["sample_id", "Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR", "cohort"]
    frame = pd.read_csv(path, sep="\t", usecols=columns, dtype={"sample_id": str})
    if frame["sample_id"].duplicated().any():
        raise SystemExit("feature store contains duplicate sample IDs")
    n_rows_total = len(frame)
    filtered = frame.loc[frame["sample_id"].isin(train_ids)].copy()
    if set(filtered["sample_id"]) != train_ids or filtered.isna().any().any():
        raise SystemExit("TRAIN diagnostic covariates are incomplete")
    return filtered.set_index("sample_id"), {
        "n_metadata_rows_seen": n_rows_total,
        "n_train_metadata_rows_used": len(filtered),
        "n_non_train_metadata_rows_used": 0,
        "fields_used": columns[1:],
        "rare_fields_from_historical_feature_store_used": [],
    }


def validate_extraction_manifest(
    manifest_path: str | Path,
    used_paths: list[Path],
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("stage") != "BUILD_FOLD_RARE_WINDOW_FEATURES":
        raise SystemExit("unexpected fold-feature manifest stage")
    container = manifest.get("container", {})
    if not container.get("path") or not container.get("sha256"):
        raise SystemExit("fold-feature manifest lacks pinned container provenance")
    expected = manifest.get("sha256", {})
    for path in used_paths:
        observed = sha256_file(path)
        if expected.get(path.name) != observed:
            raise SystemExit(f"fold-feature manifest hash mismatch for {path.name}")
    return manifest


def validate_train_covariate_provenance(
    covariates_path: str | Path,
    audit_path: str | Path,
    manifest_path: str | Path,
) -> dict:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if (
        audit.get("status") != "PASS"
        or audit.get("n_train_rows_emitted") != 2091
        or audit.get("n_test_rows_emitted") != 0
        or audit.get("n_test_values_used_in_pca_or_diagnostics") != 0
    ):
        raise SystemExit("TRAIN diagnostic covariate audit failed")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("stage") != "BUILD_TRAIN_DIAGNOSTIC_COVARIATES":
        raise SystemExit("unexpected TRAIN covariate manifest stage")
    expected = manifest.get("sha256", {})
    for path in (Path(covariates_path), Path(audit_path)):
        if expected.get(path.name) != sha256_file(path):
            raise SystemExit(f"TRAIN covariate manifest hash mismatch for {path.name}")
    return audit


def linear_predictions(
    x_fit: np.ndarray,
    x_validation: np.ndarray,
    burden_fit: np.ndarray,
    burden_validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unweighted mean/burden predictions; coefficients are FIT-only."""
    n_windows = x_fit.shape[1]
    mean = np.empty(n_windows, dtype=float)
    intercept = np.empty(n_windows, dtype=float)
    slope = np.empty(n_windows, dtype=float)
    design = np.column_stack([np.ones(len(burden_fit)), burden_fit])
    for window in range(n_windows):
        mean[window] = float(x_fit[:, window].mean())
        coef, *_ = np.linalg.lstsq(design, x_fit[:, window], rcond=None)
        intercept[window], slope[window] = coef
    mean_prediction = np.broadcast_to(mean, x_validation.shape).copy()
    burden_prediction = intercept[None, :] + burden_validation[:, None] * slope[None, :]
    return mean_prediction, burden_prediction, intercept, slope


def error_metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    if y.shape != prediction.shape or y.shape != weights.shape:
        raise ValueError("metric arrays must have identical shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metric arrays contain non-finite values")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("callability sensitivity weights must be finite and positive")
    residual = y - prediction
    squared = residual**2
    absolute = np.abs(residual)
    sse = float(np.sum(squared))
    callable_weighted_sse = float(np.sum(weights * squared))
    return {
        "sse": sse,
        "rmse": math.sqrt(sse / float(y.size)),
        "callable_weighted_sse_sensitivity": callable_weighted_sse,
        "callable_weighted_rmse_sensitivity": math.sqrt(
            callable_weighted_sse / float(np.sum(weights))
        ),
        "macro_rmse": float(np.mean(np.sqrt(np.mean(squared, axis=1)))),
        "macro_mae": float(np.mean(np.mean(absolute, axis=1))),
        "window_macro_rmse": float(np.mean(np.sqrt(np.mean(squared, axis=0)))),
        "window_macro_mae": float(np.mean(np.mean(absolute, axis=0))),
    }


def orthonormal_rows(loadings: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(loadings.T)
    return q.T


def subspace_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape[0] != right.shape[0]:
        raise ValueError("subspaces must have the same requested rank")
    rank = left.shape[0]
    if left.shape[1] < rank or right.shape[1] < rank:
        raise ValueError("common window support is smaller than rank")
    if np.linalg.matrix_rank(left) < rank or np.linalg.matrix_rank(right) < rank:
        raise ValueError("restricted loading matrix is rank deficient")
    left_q = orthonormal_rows(left)
    right_q = orthonormal_rows(right)
    singular = np.linalg.svd(left_q @ right_q.T, compute_uv=False)
    angles = subspace_angles(left_q.T, right_q.T)
    return {
        "overlap": float(np.sum(singular**2) / len(singular)),
        "mean_angle_degrees": float(np.degrees(angles).mean()),
        "max_angle_degrees": float(np.degrees(angles).max()),
    }


def group_bootstrap_skill(
    sample_table: pd.DataFrame,
    ranks: tuple[int, ...],
    replicates: int,
    seed: int,
) -> list[dict]:
    required = {
        "sample_id", "outer_fold", "split_group_key", "rank",
        "pca_sse", "burden_sse", "mean_sse",
    }
    if not required.issubset(sample_table.columns) or sample_table[list(required)].isna().any().any():
        raise ValueError("bootstrap table is incomplete")
    counts = sample_table.groupby(["rank", "sample_id"]).size()
    if not (counts == 1).all():
        raise ValueError("each OOF sample must occur once per rank")
    group_folds = sample_table.groupby("split_group_key")["outer_fold"].nunique()
    if int(group_folds.max()) != 1:
        raise ValueError("a kinship group crosses outer folds in bootstrap input")
    rng = np.random.default_rng(seed)
    grouped = (
        sample_table.groupby(["outer_fold", "split_group_key", "rank"], as_index=False)[
            ["pca_sse", "burden_sse", "mean_sse"]
        ].sum()
    )
    output: list[dict] = []
    for rank in ranks:
        rank_frame = grouped.loc[grouped["rank"] == rank]
        burden_total = float(rank_frame["burden_sse"].sum())
        if not math.isfinite(burden_total) or burden_total <= 0:
            raise ValueError("burden SSE must be finite and positive")
        observed = 1.0 - rank_frame["pca_sse"].sum() / burden_total
        values = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            pca_sse = 0.0
            burden_sse = 0.0
            for fold in sorted(rank_frame["outer_fold"].unique()):
                fold_frame = rank_frame.loc[rank_frame["outer_fold"] == fold]
                draw = rng.integers(0, len(fold_frame), size=len(fold_frame))
                sampled = fold_frame.iloc[draw]
                pca_sse += float(sampled["pca_sse"].sum())
                burden_sse += float(sampled["burden_sse"].sum())
            values[replicate] = 1.0 - pca_sse / burden_sse
        output.append(
            {
                "rank": rank,
                "observed_skill_vs_burden": float(observed),
                "bootstrap_replicates": replicates,
                "bootstrap_ci95_low": float(np.quantile(values, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(values, 0.975)),
            }
        )
    return output


def primary_performance_gate(
    rank: int,
    primary_rank: int,
    fold_skills: np.ndarray,
    bootstrap_ci95_low: float,
) -> bool:
    """Only the preregistered capacity-matched rank may drive branching."""
    return bool(
        rank == primary_rank
        and len(fold_skills) == 4
        and np.all(np.isfinite(fold_skills))
        and np.all(fold_skills > 0)
        and math.isfinite(bootstrap_ci95_low)
        and bootstrap_ci95_low > 0
    )


def diagnostic_r2(
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    train_covariates: pd.DataFrame,
    validation_covariates: pd.DataFrame,
    burden_train: np.ndarray,
    burden_validation: np.ndarray,
    callability_train: np.ndarray,
    callability_validation: np.ndarray,
    max_components: int = 8,
) -> list[dict]:
    results: list[dict] = []
    q_columns = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cohort_fit = encoder.fit_transform(train_covariates[["cohort"]])
    cohort_validation = encoder.transform(validation_covariates[["cohort"]])
    predictors = {
        "Q_4D": (train_covariates[q_columns].to_numpy(), validation_covariates[q_columns].to_numpy()),
        "cohort": (cohort_fit, cohort_validation),
        "global_burden": (burden_train[:, None], burden_validation[:, None]),
        "global_callability": (callability_train[:, None], callability_validation[:, None]),
    }
    for component in range(min(max_components, train_scores.shape[1])):
        for name, (fit_x, validation_x) in predictors.items():
            model = LinearRegression().fit(fit_x, train_scores[:, component])
            prediction = model.predict(validation_x)
            results.append(
                {
                    "component": component + 1,
                    "diagnostic": name,
                    "validation_r2": float(
                        r2_score(validation_scores[:, component], prediction)
                    ),
                }
            )
    return results


def main() -> None:
    args = parse_args()
    folds = parse_int_grid(args.outer_folds, "outer folds")
    ranks = parse_int_grid(args.ranks, "ranks")
    if args.primary_rank not in ranks:
        raise SystemExit("primary rank must be in ranks")
    if args.bootstrap_replicates < 100:
        raise SystemExit("insufficient bootstrap replicates")

    feature_paths = paths_by_fold(args.features, folds, FOLD_PATTERN)
    window_pattern = re.compile(r"\.fold(\d+)\.windows\.tsv$")
    window_paths = paths_by_fold(args.windows, folds, window_pattern)
    extraction_manifest = validate_extraction_manifest(
        args.fold_manifest,
        [Path(args.fold_qc), *feature_paths.values(), *window_paths.values()],
    )
    extraction_qc = json.loads(Path(args.fold_qc).read_text(encoding="utf-8"))
    if extraction_qc.get("status") != "PASS" or extraction_qc.get("n_test_genotypes_emitted") != 0:
        raise SystemExit("fold extraction QC is not PASS with zero TEST genotypes")
    split = load_split(args.split_manifest, folds)
    train_ids = set(split["sample_id"])
    train_covariate_audit = validate_train_covariate_provenance(
        args.feature_store,
        args.train_covariates_audit,
        args.train_covariates_manifest,
    )
    covariates, covariate_audit = load_diagnostic_covariates(args.feature_store, train_ids)
    q_columns = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR"]
    q_values = covariates[q_columns].to_numpy(dtype=float)
    if np.any(q_values < 0) or np.any(q_values > 1) or not np.allclose(
        q_values.sum(axis=1), 1.0, atol=5e-3
    ):
        raise SystemExit("TRAIN Q covariates violate range/sum invariants")
    split_cohort = split.set_index("sample_id").loc[covariates.index, "cohort"].astype(str)
    if not np.array_equal(split_cohort.to_numpy(), covariates["cohort"].astype(str).to_numpy()):
        raise SystemExit("cohort differs between split manifest and diagnostic covariates")

    fold_models: dict[int, dict] = {}
    fold_metric_rows: list[dict] = []
    sample_error_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    loadings_rows: list[list] = []
    score_rows: list[list] = []
    prediction_rows: list[list] = []
    baseline_rows: list[list] = []

    for fold in folds:
        frame = pd.read_csv(feature_paths[fold], sep="\t", dtype={"sample_id": str})
        windows = pd.read_csv(window_paths[fold], sep="\t")
        if frame.duplicated(["sample_id", "window_id"]).any():
            raise SystemExit(f"fold {fold}: duplicate sample-window rows")
        if set(frame["outer_fold"].astype(int).unique()) != {fold}:
            raise SystemExit(f"fold {fold}: outer_fold column mismatch")
        sample_order = split["sample_id"].tolist()
        window_order = windows.sort_values("start_0based")["window_id"].tolist()
        if len(window_order) != extraction_qc["n_windows"] or set(frame["sample_id"]) != train_ids:
            raise SystemExit(f"fold {fold}: matrix cardinality mismatch")

        def matrix(column: str) -> np.ndarray:
            pivot = frame.pivot(index="sample_id", columns="window_id", values=column)
            return pivot.reindex(index=sample_order, columns=window_order).to_numpy(dtype=float)

        x_all = matrix("minor_carrier_rate")
        callable_all = matrix("n_callable_rare_sites")
        carrier_all = matrix("minor_carrier_site_count")
        panel = windows.set_index("window_id").loc[window_order, "n_fold_train_rare_sites"].to_numpy()
        split_fold = split.set_index("sample_id").loc[sample_order, "fold"].to_numpy(dtype=int)
        fit_mask = split_fold != fold
        validation_mask = split_fold == fold
        expected_roles = np.where(validation_mask, "VALIDATION", "FIT")
        observed_roles = (
            frame[["sample_id", "fold_role"]]
            .drop_duplicates()
            .set_index("sample_id")
            .loc[sample_order, "fold_role"]
            .to_numpy()
        )
        if not np.array_equal(expected_roles, observed_roles):
            raise SystemExit(f"fold {fold}: fold_role differs from canonical split")
        if int(validation_mask.sum()) != extraction_qc["fold_counts"][str(fold)]:
            raise SystemExit(f"fold {fold}: validation membership mismatch")
        informative = (
            (panel > 0)
            & np.all(np.isfinite(x_all[fit_mask]), axis=0)
            & (np.ptp(x_all[fit_mask], axis=0) > 0)
        )
        if int(informative.sum()) != extraction_qc["folds"][str(fold)]["n_informative_windows"]:
            raise SystemExit(f"fold {fold}: informative mask differs from extraction QC")
        x_fit = x_all[fit_mask][:, informative]
        x_validation = x_all[validation_mask][:, informative]
        w_fit = callable_all[fit_mask][:, informative]
        w_validation = callable_all[validation_mask][:, informative]
        if not np.all(np.isfinite(x_fit)) or not np.all(np.isfinite(x_validation)):
            raise SystemExit(f"fold {fold}: non-finite value in an informative window")
        if ranks[-1] >= min(x_fit.shape):
            raise SystemExit(f"fold {fold}: maximum rank exceeds matrix dimensions")

        burden_all = np.divide(
            carrier_all[:, informative].sum(axis=1),
            callable_all[:, informative].sum(axis=1),
        )
        callability_all = np.divide(
            callable_all[:, informative].sum(axis=1),
            np.sum(panel[informative]),
        )
        mean_prediction, burden_prediction, burden_intercept, burden_slope = linear_predictions(
            x_fit, x_validation, burden_all[fit_mask], burden_all[validation_mask]
        )
        mean_metrics = error_metrics(x_validation, mean_prediction, w_validation)
        burden_metrics = error_metrics(x_validation, burden_prediction, w_validation)

        pca = PCA(n_components=ranks[-1], svd_solver="full").fit(x_fit)
        if not np.allclose(pca.components_ @ pca.components_.T, np.eye(ranks[-1]), atol=1e-10):
            raise SystemExit(f"fold {fold}: PCA components are not orthonormal")
        fit_scores = pca.transform(x_fit)
        validation_scores = pca.transform(x_validation)
        fit_ids = split.loc[fit_mask, "sample_id"].tolist()
        validation_ids = split.loc[validation_mask, "sample_id"].tolist()
        informative_ids = [window_order[index] for index in np.flatnonzero(informative)]

        for component in range(ranks[-1]):
            for sample_id, role, value in zip(fit_ids, ["FIT"] * len(fit_ids), fit_scores[:, component]):
                score_rows.append([sample_id, fold, role, component + 1, value])
            for sample_id, role, value in zip(
                validation_ids, ["VALIDATION"] * len(validation_ids), validation_scores[:, component]
            ):
                score_rows.append([sample_id, fold, role, component + 1, value])
            loading_map = dict(zip(informative_ids, pca.components_[component]))
            for _, window_row in windows.sort_values("start_0based").iterrows():
                window_id = window_row["window_id"]
                loadings_rows.append(
                    [
                        fold, component + 1, window_id, int(window_row["start_0based"]),
                        int(window_row["end_0based"]), int(window_id in loading_map),
                        loading_map.get(window_id, math.nan),
                    ]
                )

        validation_meta = split.loc[validation_mask, ["sample_id", "split_group_key"]].reset_index(drop=True)
        for sample_index, sample_id in enumerate(validation_ids):
            for window_index_value, window_id in enumerate(informative_ids):
                baseline_rows.append(
                    [
                        sample_id, fold, window_id, x_validation[sample_index, window_index_value],
                        w_validation[sample_index, window_index_value],
                        mean_prediction[sample_index, window_index_value],
                        burden_prediction[sample_index, window_index_value],
                    ]
                )

        for rank in ranks:
            reconstruction = (
                validation_scores[:, :rank] @ pca.components_[:rank] + pca.mean_
            )
            metrics = error_metrics(x_validation, reconstruction, w_validation)
            if burden_metrics["sse"] <= 0 or mean_metrics["sse"] <= 0:
                raise SystemExit(f"fold {fold}: baseline SSE is not positive")
            skill_burden = 1.0 - metrics["sse"] / burden_metrics["sse"]
            skill_mean = 1.0 - metrics["sse"] / mean_metrics["sse"]
            fold_metric_rows.append(
                {
                    "outer_fold": fold,
                    "rank": rank,
                    "n_fit": len(fit_ids),
                    "n_validation": len(validation_ids),
                    "n_informative_windows": len(informative_ids),
                    "explained_variance_ratio_cumulative": float(
                        pca.explained_variance_ratio_[:rank].sum()
                    ),
                    **metrics,
                    "mean_sse": mean_metrics["sse"],
                    "burden_sse": burden_metrics["sse"],
                    "skill_vs_mean": skill_mean,
                    "skill_vs_burden": skill_burden,
                    "mean_callable_weighted_sse_sensitivity": mean_metrics[
                        "callable_weighted_sse_sensitivity"
                    ],
                    "burden_callable_weighted_sse_sensitivity": burden_metrics[
                        "callable_weighted_sse_sensitivity"
                    ],
                    "skill_vs_burden_callable_weighted_sensitivity": (
                        1.0
                        - metrics["callable_weighted_sse_sensitivity"]
                        / burden_metrics["callable_weighted_sse_sensitivity"]
                    ),
                }
            )
            residual = x_validation - reconstruction
            for sample_index, sample_id in enumerate(validation_ids):
                weights = w_validation[sample_index]
                sample_error_rows.append(
                    {
                        "sample_id": sample_id,
                        "outer_fold": fold,
                        "split_group_key": validation_meta.loc[sample_index, "split_group_key"],
                        "rank": rank,
                        "pca_sse": float(np.sum(residual[sample_index] ** 2)),
                        "burden_sse": float(
                            np.sum((x_validation[sample_index] - burden_prediction[sample_index]) ** 2)
                        ),
                        "mean_sse": float(
                            np.sum((x_validation[sample_index] - mean_prediction[sample_index]) ** 2)
                        ),
                        "pca_callable_weighted_sse_sensitivity": float(
                            np.sum(weights * residual[sample_index] ** 2)
                        ),
                        "burden_callable_weighted_sse_sensitivity": float(
                            np.sum(
                                weights
                                * (x_validation[sample_index] - burden_prediction[sample_index]) ** 2
                            )
                        ),
                        "mean_callable_weighted_sse_sensitivity": float(
                            np.sum(
                                weights
                                * (x_validation[sample_index] - mean_prediction[sample_index]) ** 2
                            )
                        ),
                    }
                )
                for window_index_value, window_id in enumerate(informative_ids):
                    prediction_rows.append(
                        [
                            sample_id, fold, rank, window_id,
                            reconstruction[sample_index, window_index_value],
                            residual[sample_index, window_index_value],
                        ]
                    )

        ordered_covariates = covariates.loc[sample_order]
        fold_diagnostics = diagnostic_r2(
            fit_scores,
            validation_scores,
            ordered_covariates.iloc[np.flatnonzero(fit_mask)],
            ordered_covariates.iloc[np.flatnonzero(validation_mask)],
            burden_all[fit_mask],
            burden_all[validation_mask],
            callability_all[fit_mask],
            callability_all[validation_mask],
        )
        for row in fold_diagnostics:
            diagnostic_rows.append({"outer_fold": fold, **row})
        fold_models[fold] = {
            "windows": informative_ids,
            "components": pca.components_.copy(),
            "burden_intercept": burden_intercept,
            "burden_slope": burden_slope,
        }

    stability_rows: list[dict] = []
    for left, right in combinations(folds, 2):
        common = sorted(set(fold_models[left]["windows"]) & set(fold_models[right]["windows"]))
        left_index = [fold_models[left]["windows"].index(window) for window in common]
        right_index = [fold_models[right]["windows"].index(window) for window in common]
        for rank in ranks:
            metrics = subspace_metrics(
                fold_models[left]["components"][:rank, left_index],
                fold_models[right]["components"][:rank, right_index],
            )
            stability_rows.append(
                {
                    "source": "observed",
                    "replicate": -1,
                    "fold_a": left,
                    "fold_b": right,
                    "rank": rank,
                    "n_common_windows": len(common),
                    **metrics,
                }
            )

    metrics_frame = pd.DataFrame(fold_metric_rows)
    sample_errors = pd.DataFrame(sample_error_rows)
    bootstrap_rows = group_bootstrap_skill(
        sample_errors, ranks, args.bootstrap_replicates, args.seed
    )
    stability_frame = pd.DataFrame(stability_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    rank_decisions = []
    for rank in ranks:
        fold_skills = metrics_frame.loc[metrics_frame["rank"] == rank, "skill_vs_burden"]
        bootstrap = bootstrap_frame.loc[bootstrap_frame["rank"] == rank].iloc[0]
        observed_overlap = stability_frame.loc[
            (stability_frame["rank"] == rank) & (stability_frame["source"] == "observed"),
            "overlap",
        ]
        decision = {
            "rank": rank,
            "all_four_folds_skill_vs_burden_positive": bool((fold_skills > 0).all()),
            "bootstrap_ci95_low": float(bootstrap["bootstrap_ci95_low"]),
            "median_observed_subspace_overlap": float(observed_overlap.median()),
        }
        decision["passes_exploratory_gate"] = primary_performance_gate(
            rank,
            args.primary_rank,
            fold_skills.to_numpy(dtype=float),
            decision["bootstrap_ci95_low"],
        )
        rank_decisions.append(decision)

    output_prefix = Path(args.out_prefix)
    metrics_frame.to_csv(f"{output_prefix}.pca_fold_metrics.tsv", sep="\t", index=False)
    sample_errors.to_csv(f"{output_prefix}.pca_sample_errors.tsv", sep="\t", index=False)
    bootstrap_frame.to_csv(f"{output_prefix}.pca_bootstrap.tsv", sep="\t", index=False)
    stability_frame.to_csv(f"{output_prefix}.pca_subspace_stability.tsv", sep="\t", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        f"{output_prefix}.pca_diagnostics.tsv", sep="\t", index=False
    )

    with deterministic_gzip_writer(f"{output_prefix}.pca_scores.tsv.gz") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample_id", "outer_fold", "fold_role", "component", "score"])
        writer.writerows(score_rows)
    with deterministic_gzip_writer(f"{output_prefix}.pca_loadings.tsv.gz") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "outer_fold", "component", "window_id", "start_0based", "end_0based",
                "informative_mask", "loading",
            ]
        )
        writer.writerows(loadings_rows)
    with deterministic_gzip_writer(f"{output_prefix}.pca_oof_baselines.tsv.gz") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sample_id", "outer_fold", "window_id", "observed", "n_callable",
                "mean_prediction", "burden_prediction",
            ]
        )
        writer.writerows(baseline_rows)
    with deterministic_gzip_writer(f"{output_prefix}.pca_oof_reconstructions.tsv.gz") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample_id", "outer_fold", "rank", "window_id", "prediction", "residual"])
        writer.writerows(prediction_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for fold in folds:
        subset = metrics_frame.loc[metrics_frame["outer_fold"] == fold]
        axes[0].plot(subset["rank"], subset["skill_vs_burden"], marker="o", label=f"fold {fold}")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ranks, labels=[str(value) for value in ranks])
    axes[0].set_xlabel("PCA rank (capacity curve; not selected)")
    axes[0].set_ylabel("OOF skill vs burden")
    axes[0].legend(frameon=False, fontsize=8)
    observed_summary = (
        stability_frame.loc[stability_frame["source"] == "observed"]
        .groupby("rank")["overlap"].median()
    )
    axes[1].plot(ranks, observed_summary.loc[list(ranks)], marker="o", label="observed median")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ranks, labels=[str(value) for value in ranks])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("PCA rank")
    axes[1].set_ylabel("subspace overlap")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(f"{output_prefix}.pca_benchmark.png", dpi=180)
    fig.savefig(f"{output_prefix}.pca_benchmark.pdf")
    plt.close(fig)

    output_files = [
        f"{output_prefix}.pca_fold_metrics.tsv",
        f"{output_prefix}.pca_sample_errors.tsv",
        f"{output_prefix}.pca_bootstrap.tsv",
        f"{output_prefix}.pca_subspace_stability.tsv",
        f"{output_prefix}.pca_diagnostics.tsv",
        f"{output_prefix}.pca_scores.tsv.gz",
        f"{output_prefix}.pca_loadings.tsv.gz",
        f"{output_prefix}.pca_oof_baselines.tsv.gz",
        f"{output_prefix}.pca_oof_reconstructions.tsv.gz",
        f"{output_prefix}.pca_benchmark.png",
        f"{output_prefix}.pca_benchmark.pdf",
    ]
    summary = {
        "status": "PASS",
        "stage": "M25B_FOLD_PCA",
        "scope": "internal OOF reconstruction within canonical TRAIN; TEST remains blind",
        "estimand": "full-vector low-rank reconstructability/compressibility of minor-carrier-rate beyond a FIT-only global-burden profile",
        "rank_policy": {
            "grid": list(ranks),
            "primary_capacity_matched_rank": args.primary_rank,
            "selection": "none; ordinary PCA reconstruction error is monotone in rank",
        },
        "preprocessing": {
            "feature": "minor_carrier_rate",
            "centering": "FIT window means only",
            "variance_scaling": False,
            "fold_specific_site_selection_and_orientation": True,
            "master_grid_windows": extraction_qc["n_windows"],
        },
        "baselines": {
            "mean": "FIT arithmetic mean per window",
            "burden": "FIT ordinary least squares per window: carrier_rate = alpha + beta * global carrier burden",
        },
        "diagnostics_not_model_inputs": ["Q_4D", "cohort", "global_callability", "global_burden"],
        "covariate_audit": covariate_audit,
        "train_covariate_sealing_audit": train_covariate_audit,
        "extraction_manifest_container": extraction_manifest["container"],
        "n_test_genotypes_used": 0,
        "n_test_samples_scored": 0,
        "bootstrap": {
            "unit": "canonical split_group_key stratified by outer fold",
            "scope": "conditional OOF error uncertainty for the four fitted models; no model refit inside bootstrap",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "stability": {
            "role": "descriptive POST-review diagnostic; excluded from the automatic performance gate",
            "reason_no_null_gate": "outer FIT sets overlap, so an independently permuted null would be mismatched",
        },
        "rank_decisions": rank_decisions,
        "decision": (
            "PASS_PRIMARY_PERFORMANCE_GATE_REQUIRES_POST_REVIEW"
            if any(row["passes_exploratory_gate"] for row in rank_decisions)
            else "STOP_PRIMARY_PERFORMANCE_GATE"
        ),
        "interpretation_limit": "Passing the primary performance gate supports compressible chr22 covariance within the cohort-ascertained panel; it does not establish biological validation, fine substructure or LAI improvement.",
        "inputs_sha256": {
            "fold_qc": sha256_file(args.fold_qc),
            "fold_manifest": sha256_file(args.fold_manifest),
            "split_manifest": sha256_file(args.split_manifest),
            "feature_store": sha256_file(args.feature_store),
            "train_covariates_audit": sha256_file(args.train_covariates_audit),
            "train_covariates_manifest": sha256_file(args.train_covariates_manifest),
            **{path.name: sha256_file(path) for path in feature_paths.values()},
            **{path.name: sha256_file(path) for path in window_paths.values()},
        },
        "outputs_sha256": {Path(path).name: sha256_file(path) for path in output_files},
    }
    write_json(f"{output_prefix}.pca_summary.json", summary)
    print(json.dumps({"status": "PASS", "decision": summary["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
