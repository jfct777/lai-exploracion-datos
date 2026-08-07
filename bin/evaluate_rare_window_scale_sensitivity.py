#!/usr/bin/env python3
"""M25C: evaluate rare-carrier PCA robustness across genomic window grids.

The script consumes the fold-specific 250 kb counts produced by M25B.  It
never rereads genotypes, refits MAC/MAF/orientation, or accesses TEST.  Larger
windows are derived by summing integer numerators and denominators; rates are
always recomputed after aggregation.  Overlapping grids are represented by
separate non-overlapping phases so shared loci are not treated as additional
observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path.cwd()))

from build_rare_window_features import deterministic_gzip_writer, sha256_file, write_json
from evaluate_fold_rare_window_pca import (
    diagnostic_r2,
    error_metrics,
    linear_predictions,
    load_diagnostic_covariates,
    load_split,
    parse_int_grid,
    paths_by_fold,
    subspace_metrics,
    validate_extraction_manifest,
    validate_train_covariate_provenance,
)


FEATURE_PATTERN = re.compile(r"\.fold(\d+)\.sample_window_features\.tsv\.gz$")
WINDOW_PATTERN = re.compile(r"\.fold(\d+)\.windows\.tsv$")
ADDITIVE_COLUMNS = (
    "n_callable_rare_sites",
    "n_missing_rare_sites",
    "minor_carrier_site_count",
    "minor_dosage_sum",
    "het_minor_count",
    "hom_minor_count",
)
REFERENCE_SCHEME = "physical_250_o0"
NEW_CORE_SCHEMES = (
    "physical_500_o0",
    "physical_1000_o0",
    "equal_site_approx",
)
BOUNDARY_SCHEMES = ("physical_500_o250", "physical_1000_o500")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--windows", nargs="+", required=True)
    parser.add_argument("--fold-qc", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--diagnostic-covariates", required=True)
    parser.add_argument("--diagnostic-covariates-audit", required=True)
    parser.add_argument("--diagnostic-covariates-manifest", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--chrom", default="22")
    parser.add_argument("--outer-folds", default="0,1,2,4")
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--primary-rank", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="chr22")
    return parser.parse_args()


def validate_preregistration(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        data.get("stage") != "M25C_RARE_WINDOW_SCALE_SENSITIVITY"
        or data.get("status") != "PREREGISTERED_BEFORE_EXECUTION"
    ):
        raise SystemExit("invalid M25C preregistration")
    observed = [row["id"] for row in data.get("fixed_schemes", [])]
    expected = [
        REFERENCE_SCHEME,
        "physical_500_o0",
        "physical_500_o250",
        "physical_1000_o0",
        "physical_1000_o500",
    ]
    if observed != expected or data.get("adaptive_scheme", {}).get("id") != "equal_site_approx":
        raise SystemExit("preregistered M25C scheme list differs from executable contract")
    return data


def validate_base_windows(windows: pd.DataFrame, fold: int) -> pd.DataFrame:
    required = {
        "outer_fold",
        "chrom",
        "window_id",
        "start_0based",
        "end_0based",
        "window_bp",
        "n_input_cohort_rare_sites",
        "n_fold_train_rare_sites",
    }
    if not required.issubset(windows.columns):
        raise SystemExit(f"fold {fold}: base window catalogue is incomplete")
    ordered = windows.sort_values("start_0based").reset_index(drop=True).copy()
    if ordered["window_id"].duplicated().any() or set(ordered["outer_fold"].astype(int)) != {fold}:
        raise SystemExit(f"fold {fold}: invalid base window identifiers/fold")
    starts = ordered["start_0based"].to_numpy(dtype=int)
    ends = ordered["end_0based"].to_numpy(dtype=int)
    if starts[0] != 0 or not np.array_equal(starts[1:], ends[:-1]):
        raise SystemExit(f"fold {fold}: 250 kb atoms are not a contiguous tiling")
    if not np.all((ends - starts) == ordered["window_bp"].to_numpy(dtype=int)):
        raise SystemExit(f"fold {fold}: window length columns disagree")
    if not np.all((ends[:-1] - starts[:-1]) == 250_000):
        raise SystemExit(f"fold {fold}: non-terminal atoms are not 250 kb")
    return ordered


def _window_record(
    scheme: str,
    role: str,
    ordinal: int,
    source_indices: list[int],
    windows: pd.DataFrame,
    expected_width_bp: int | None,
    target_sites: float | None = None,
) -> dict:
    source = windows.iloc[source_indices]
    start = int(source["start_0based"].min())
    end = int(source["end_0based"].max())
    input_sites = int(source["n_input_cohort_rare_sites"].sum())
    panel_sites = int(source["n_fold_train_rare_sites"].sum())
    actual_width = end - start
    return {
        "scheme": scheme,
        "scheme_role": role,
        "derived_window_id": f"{scheme}_w{ordinal:04d}",
        "start_0based": start,
        "end_0based": end,
        "window_bp": actual_width,
        "expected_width_bp": expected_width_bp if expected_width_bp is not None else math.nan,
        "is_partial": int(expected_width_bp is not None and actual_width < expected_width_bp),
        "is_terminal": int(end == int(windows["end_0based"].max())),
        "is_structural_gap": int(input_sites == 0),
        "n_input_cohort_rare_sites": input_sites,
        "n_fold_train_rare_sites": panel_sites,
        "n_base_atoms": len(source_indices),
        "target_fold_train_sites": target_sites if target_sites is not None else math.nan,
        "source_indices": tuple(source_indices),
        "source_window_ids": tuple(source["window_id"].astype(str)),
    }


def fixed_window_map(
    windows: pd.DataFrame,
    width_bp: int,
    offset_bp: int,
    role: str,
) -> list[dict]:
    if width_bp % 250_000 or offset_bp % 250_000 or offset_bp >= width_bp:
        raise ValueError("fixed schemes must align to the 250 kb atomic grid")
    scheme = f"physical_{width_bp // 1000}_o{offset_bp // 1000}"
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in windows.iterrows():
        start = int(row["start_0based"])
        if start < offset_bp:
            continue
        groups[(start - offset_bp) // width_bp].append(int(index))
    return [
        _window_record(scheme, role, ordinal, groups[key], windows, width_bp)
        for ordinal, key in enumerate(sorted(groups))
    ]


def _contiguous_true_blocks(mask: np.ndarray) -> list[list[int]]:
    blocks: list[list[int]] = []
    current: list[int] = []
    for index, value in enumerate(mask):
        if value:
            current.append(index)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def equal_site_window_map(windows: pd.DataFrame) -> list[dict]:
    structural = windows["n_input_cohort_rare_sites"].to_numpy(dtype=int) > 0
    selected = windows["n_fold_train_rare_sites"].to_numpy(dtype=int)
    informative_atoms = structural & (selected > 0)
    if int(informative_atoms.sum()) == 0:
        raise ValueError("equal-site map has no informative atomic windows")
    target = float(selected[informative_atoms].sum() / informative_atoms.sum())
    records: list[dict] = []
    ordinal = 0

    for block in _contiguous_true_blocks(structural):
        groups: list[list[int]] = []
        current: list[int] = []
        current_sites = 0
        for index in block:
            next_sites = int(selected[index])
            if (
                current
                and current_sites < target
                and current_sites + next_sites > target
                and abs(target - current_sites) <= abs(target - current_sites - next_sites)
            ):
                groups.append(current)
                current = []
                current_sites = 0
            current.append(index)
            current_sites += next_sites
            if current_sites >= target:
                groups.append(current)
                current = []
                current_sites = 0
        if current:
            if groups and sum(selected[index] for index in current) < 0.5 * target:
                groups[-1].extend(current)
            else:
                groups.append(current)
        for group in groups:
            records.append(
                _window_record(
                    "equal_site_approx", "core", ordinal, group, windows, None, target
                )
            )
            ordinal += 1

    for block in _contiguous_true_blocks(~structural):
        records.append(
            _window_record(
                "equal_site_approx", "core", ordinal, block, windows, None, target
            )
        )
        ordinal += 1
    return sorted(records, key=lambda row: row["start_0based"])


def build_schemes(windows: pd.DataFrame) -> dict[str, list[dict]]:
    schemes = {
        REFERENCE_SCHEME: fixed_window_map(windows, 250_000, 0, "reference"),
        "physical_500_o0": fixed_window_map(windows, 500_000, 0, "core"),
        "physical_500_o250": fixed_window_map(windows, 500_000, 250_000, "boundary_phase"),
        "physical_1000_o0": fixed_window_map(windows, 1_000_000, 0, "core"),
        "physical_1000_o500": fixed_window_map(
            windows, 1_000_000, 500_000, "boundary_phase"
        ),
        "equal_site_approx": equal_site_window_map(windows),
    }
    if tuple(schemes) != (REFERENCE_SCHEME, *NEW_CORE_SCHEMES[:1], BOUNDARY_SCHEMES[0], NEW_CORE_SCHEMES[1], BOUNDARY_SCHEMES[1], NEW_CORE_SCHEMES[2]):
        raise AssertionError("internal scheme order drift")
    return schemes


def pivot_matrix(
    frame: pd.DataFrame,
    sample_order: list[str],
    window_order: list[str],
    column: str,
) -> np.ndarray:
    pivot = frame.pivot(index="sample_id", columns="window_id", values=column)
    return pivot.reindex(index=sample_order, columns=window_order).to_numpy(dtype=float)


def aggregate_columns(matrix: np.ndarray, records: list[dict]) -> np.ndarray:
    return np.column_stack(
        [matrix[:, np.asarray(row["source_indices"], dtype=int)].sum(axis=1) for row in records]
    )


def paired_group_bootstrap(
    sample_errors: pd.DataFrame,
    schemes: tuple[str, ...],
    ranks: tuple[int, ...],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    combos = [(scheme, rank) for scheme in schemes for rank in ranks]
    grouped = (
        sample_errors.groupby(
            ["outer_fold", "split_group_key", "scheme", "rank"], as_index=False
        )[
            [
                "pca_sse",
                "burden_sse",
                "pca_weighted_sse",
                "burden_weighted_sse",
            ]
        ].sum()
    )
    metric_names = ["pca_sse", "burden_sse", "pca_weighted_sse", "burden_weighted_sse"]
    per_fold: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for fold in sorted(grouped["outer_fold"].unique()):
        subset = grouped.loc[grouped["outer_fold"] == fold]
        groups = sorted(subset["split_group_key"].astype(str).unique())
        array = np.empty((len(groups), len(combos), len(metric_names)), dtype=float)
        indexed = subset.set_index(["split_group_key", "scheme", "rank"])
        for group_index, group in enumerate(groups):
            for combo_index, (scheme, rank) in enumerate(combos):
                try:
                    row = indexed.loc[(group, scheme, rank)]
                except KeyError as exc:
                    raise ValueError(f"bootstrap combination missing: {fold}/{group}/{scheme}/{rank}") from exc
                array[group_index, combo_index] = row[metric_names].to_numpy(dtype=float)
        per_fold[int(fold)] = array

    replicate_skill = np.empty((replicates, len(combos)), dtype=float)
    replicate_weighted = np.empty_like(replicate_skill)
    for replicate in range(replicates):
        totals = np.zeros((len(combos), len(metric_names)), dtype=float)
        for array in per_fold.values():
            draw = rng.integers(0, array.shape[0], size=array.shape[0])
            counts = np.bincount(draw, minlength=array.shape[0]).astype(float)
            totals += np.tensordot(counts, array, axes=(0, 0))
        replicate_skill[replicate] = 1.0 - totals[:, 0] / totals[:, 1]
        replicate_weighted[replicate] = 1.0 - totals[:, 2] / totals[:, 3]

    observed = (
        sample_errors.groupby(["scheme", "rank"])[metric_names].sum().reindex(combos)
    )
    rows: list[dict] = []
    for index, (scheme, rank) in enumerate(combos):
        values = observed.loc[(scheme, rank)].to_numpy(dtype=float)
        reference_index = combos.index((REFERENCE_SCHEME, rank))
        rows.append(
            {
                "scheme": scheme,
                "rank": rank,
                "bootstrap_replicates": replicates,
                "observed_skill_vs_burden": 1.0 - values[0] / values[1],
                "bootstrap_ci95_low": float(np.quantile(replicate_skill[:, index], 0.025)),
                "bootstrap_ci95_high": float(np.quantile(replicate_skill[:, index], 0.975)),
                "observed_weighted_skill_vs_burden": 1.0 - values[2] / values[3],
                "weighted_bootstrap_ci95_low": float(
                    np.quantile(replicate_weighted[:, index], 0.025)
                ),
                "weighted_bootstrap_ci95_high": float(
                    np.quantile(replicate_weighted[:, index], 0.975)
                ),
                "delta_skill_vs_reference": float(
                    (1.0 - values[0] / values[1])
                    - (1.0 - observed.loc[(REFERENCE_SCHEME, rank), "pca_sse"] / observed.loc[(REFERENCE_SCHEME, rank), "burden_sse"])
                ),
                "delta_skill_ci95_low": float(
                    np.quantile(
                        replicate_skill[:, index] - replicate_skill[:, reference_index], 0.025
                    )
                ),
                "delta_skill_ci95_high": float(
                    np.quantile(
                        replicate_skill[:, index] - replicate_skill[:, reference_index], 0.975
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_site_bias(contributions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (fold, scheme), subset in contributions.groupby(["outer_fold", "scheme"]):
        rho, p_value = spearmanr(
            subset["n_fold_train_rare_sites"].to_numpy(dtype=float),
            subset["delta_sse"].to_numpy(dtype=float),
        )
        ordered = subset.copy()
        ordered["site_quintile"] = pd.qcut(
            ordered["n_fold_train_rare_sites"].rank(method="first"), 5, labels=False
        )
        total = float(ordered["delta_sse"].sum())
        low = float(ordered.loc[ordered["site_quintile"] == 0, "delta_sse"].sum())
        rows.append(
            {
                "outer_fold": int(fold),
                "scheme": scheme,
                "spearman_delta_sse_vs_sites": float(rho),
                "spearman_p_value_descriptive": float(p_value),
                "net_gain": total,
                "lowest_site_quintile_gain": low,
                "lowest_site_quintile_fraction_of_net_gain": low / total if total != 0 else math.nan,
                "median_sites_lowest_quintile": float(
                    ordered.loc[ordered["site_quintile"] == 0, "n_fold_train_rare_sites"].median()
                ),
                "n_informative_windows": len(ordered),
            }
        )
    return pd.DataFrame(rows)


def plot_results(
    prefix: Path,
    fold_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    site_bias: pd.DataFrame,
    concordance: pd.DataFrame,
    schemes: tuple[str, ...],
    ranks: tuple[int, ...],
) -> list[str]:
    labels = {
        REFERENCE_SCHEME: "250/250",
        "physical_500_o0": "500/500",
        "physical_500_o250": "500/250 fase 2",
        "physical_1000_o0": "1000/1000",
        "physical_1000_o500": "1000/500 fase 2",
        "equal_site_approx": "sitios aprox.",
    }
    x = np.arange(len(schemes))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    rank1 = fold_metrics.loc[fold_metrics["rank"] == 1]
    boot1 = bootstrap.loc[bootstrap["rank"] == 1].set_index("scheme")
    for axis, column, pooled_column, title in (
        (axes[0], "skill_vs_burden", "observed_skill_vs_burden", "Peso uniforme por ventana"),
        (
            axes[1],
            "weighted_skill_vs_burden",
            "observed_weighted_skill_vs_burden",
            "Peso por sitios evaluables",
        ),
    ):
        for index, scheme in enumerate(schemes):
            values = rank1.loc[rank1["scheme"] == scheme, column].to_numpy(dtype=float)
            axis.scatter(np.full(len(values), index), values, color="#4c78a8", alpha=0.75, s=28)
            axis.scatter(index, boot1.loc[scheme, pooled_column], color="#e45756", marker="D", s=48)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [labels[value] for value in schemes], rotation=30, ha="right")
        axis.set_ylabel("Skill OOF frente a carga")
        axis.set_title(title)
    fig.suptitle("M25C: estabilidad de la reconstrucción según escala y bordes")
    skill_png = f"{prefix}.multiscale_skill.png"
    skill_pdf = f"{prefix}.multiscale_skill.pdf"
    fig.savefig(skill_png, dpi=200)
    fig.savefig(skill_pdf)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for scheme in schemes:
        subset = bootstrap.loc[bootstrap["scheme"] == scheme].set_index("rank")
        axes[0].plot(ranks, subset.loc[list(ranks), "observed_skill_vs_burden"], marker="o", label=labels[scheme])
        axes[1].plot(ranks, subset.loc[list(ranks), "observed_weighted_skill_vs_burden"], marker="o", label=labels[scheme])
    for axis, title in zip(axes, ("Peso uniforme", "Peso por sitios evaluables")):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(ranks)
        axis.set_xlabel("Rango PCA (2 y 4 son descriptivos)")
        axis.set_ylabel("Skill OOF frente a carga")
        axis.set_title(title)
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    rank_png = f"{prefix}.multiscale_rank_curve.png"
    rank_pdf = f"{prefix}.multiscale_rank_curve.pdf"
    fig.savefig(rank_png, dpi=200)
    fig.savefig(rank_pdf, bbox_inches="tight")
    plt.close(fig)

    bias_summary = site_bias.groupby("scheme").agg(
        median_rho=("spearman_delta_sse_vs_sites", "median"),
        median_low_fraction=("lowest_site_quintile_fraction_of_net_gain", "median"),
    ).reindex(schemes)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(x, bias_summary["median_rho"], color="#72b7b2")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Mediana de Spearman (ΔSSE vs sitios)")
    axes[0].set_title("Dependencia de la ganancia con el número de sitios")
    axes[1].bar(x, bias_summary["median_low_fraction"], color="#f58518")
    axes[1].axhline(1, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Fracción de ganancia del quintil con menos sitios")
    axes[1].set_title("Concentración en ventanas pobres en sitios")
    for axis in axes:
        axis.set_xticks(x, [labels[value] for value in schemes], rotation=30, ha="right")
    bias_png = f"{prefix}.multiscale_site_bias.png"
    bias_pdf = f"{prefix}.multiscale_site_bias.pdf"
    fig.savefig(bias_png, dpi=200)
    fig.savefig(bias_pdf)
    plt.close(fig)

    heat = (
        concordance.groupby(["scheme", "rank"])["score_subspace_overlap"]
        .median()
        .unstack("rank")
        .reindex(index=schemes, columns=ranks)
    )
    fig, axis = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
    image = axis.imshow(heat.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(np.arange(len(ranks)), [str(value) for value in ranks])
    axis.set_yticks(np.arange(len(schemes)), [labels[value] for value in schemes])
    axis.set_xlabel("Rango")
    axis.set_title("Concordancia de scores con la grilla 250/250")
    fig.colorbar(image, ax=axis, label="Solapamiento de subespacios")
    concordance_png = f"{prefix}.multiscale_score_concordance.png"
    concordance_pdf = f"{prefix}.multiscale_score_concordance.pdf"
    fig.savefig(concordance_png, dpi=200)
    fig.savefig(concordance_pdf)
    plt.close(fig)
    return [
        skill_png,
        skill_pdf,
        rank_png,
        rank_pdf,
        bias_png,
        bias_pdf,
        concordance_png,
        concordance_pdf,
    ]


def main() -> None:
    args = parse_args()
    folds = parse_int_grid(args.outer_folds, "outer folds", minimum=0)
    ranks = parse_int_grid(args.ranks, "ranks")
    if args.primary_rank != 1 or ranks != (1, 2, 4):
        raise SystemExit("M25C preregistration fixes ranks 1,2,4 with primary rank 1")
    if args.bootstrap_replicates < 1000:
        raise SystemExit("M25C requires at least 1000 paired group-bootstrap replicates")
    preregistration = validate_preregistration(args.preregistration)
    feature_paths = paths_by_fold(args.features, folds, FEATURE_PATTERN)
    window_paths = paths_by_fold(args.windows, folds, WINDOW_PATTERN)
    extraction_manifest = validate_extraction_manifest(
        args.fold_manifest,
        [Path(args.fold_qc), *feature_paths.values(), *window_paths.values()],
    )
    extraction_qc = json.loads(Path(args.fold_qc).read_text(encoding="utf-8"))
    if extraction_qc.get("status") != "PASS" or extraction_qc.get("n_test_genotypes_emitted") != 0:
        raise SystemExit("M25B input QC is not PASS with zero TEST genotypes")
    split = load_split(args.split_manifest, folds).reset_index(drop=True)
    train_ids = set(split["sample_id"])
    validate_train_covariate_provenance(
        args.diagnostic_covariates,
        args.diagnostic_covariates_audit,
        args.diagnostic_covariates_manifest,
    )
    covariates, covariate_audit = load_diagnostic_covariates(
        args.diagnostic_covariates, train_ids
    )
    sample_order = split["sample_id"].astype(str).tolist()
    ordered_covariates = covariates.loc[sample_order]
    fold_assignment = split.set_index("sample_id").loc[sample_order, "fold"].to_numpy(dtype=int)

    schemes = (
        REFERENCE_SCHEME,
        "physical_500_o0",
        "physical_500_o250",
        "physical_1000_o0",
        "physical_1000_o500",
        "equal_site_approx",
    )
    fold_metric_rows: list[dict] = []
    sample_error_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    contribution_rows: list[dict] = []
    map_rows: list[dict] = []
    score_rows: list[list] = []
    validation_scores_by_fold: dict[int, dict[str, np.ndarray]] = defaultdict(dict)

    for fold in folds:
        frame = pd.read_csv(feature_paths[fold], sep="\t", dtype={"sample_id": str})
        windows = validate_base_windows(pd.read_csv(window_paths[fold], sep="\t"), fold)
        if frame.duplicated(["sample_id", "window_id"]).any() or set(frame["sample_id"]) != train_ids:
            raise SystemExit(f"fold {fold}: invalid sample-window matrix")
        window_order = windows["window_id"].astype(str).tolist()
        matrices = {
            column: pivot_matrix(frame, sample_order, window_order, column)
            for column in ADDITIVE_COLUMNS
        }
        base_panel = windows["n_fold_train_rare_sites"].to_numpy(dtype=float)
        base_informative = base_panel > 0
        global_burden = np.divide(
            matrices["minor_carrier_site_count"][:, base_informative].sum(axis=1),
            matrices["n_callable_rare_sites"][:, base_informative].sum(axis=1),
        )
        global_callability = np.divide(
            matrices["n_callable_rare_sites"][:, base_informative].sum(axis=1),
            base_panel[base_informative].sum(),
        )
        fit_mask = fold_assignment != fold
        validation_mask = fold_assignment == fold
        validation_ids = split.loc[validation_mask, "sample_id"].astype(str).tolist()
        validation_meta = split.loc[
            validation_mask, ["sample_id", "split_group_key"]
        ].reset_index(drop=True)
        scheme_maps = build_schemes(windows)

        for scheme in schemes:
            records = scheme_maps[scheme]
            for record in records:
                for source_id in record["source_window_ids"]:
                    map_rows.append(
                        {
                            "outer_fold": fold,
                            **{
                                key: value
                                for key, value in record.items()
                                if key not in {"source_indices", "source_window_ids"}
                            },
                            "source_window_id": source_id,
                        }
                    )
            aggregated = {
                column: aggregate_columns(matrix, records) for column, matrix in matrices.items()
            }
            callable_all = aggregated["n_callable_rare_sites"]
            carrier_all = aggregated["minor_carrier_site_count"]
            x_all = np.divide(
                carrier_all,
                callable_all,
                out=np.full_like(carrier_all, np.nan, dtype=float),
                where=callable_all > 0,
            )
            panel = np.asarray([record["n_fold_train_rare_sites"] for record in records], dtype=float)
            informative = (
                (panel > 0)
                & np.all(np.isfinite(x_all[fit_mask]), axis=0)
                & np.all(np.isfinite(x_all[validation_mask]), axis=0)
                & (np.ptp(x_all[fit_mask], axis=0) > 0)
            )
            x_fit = x_all[fit_mask][:, informative]
            x_validation = x_all[validation_mask][:, informative]
            w_validation = callable_all[validation_mask][:, informative]
            if x_fit.shape[1] <= ranks[-1] or not np.all(w_validation > 0):
                raise SystemExit(f"fold {fold}/{scheme}: insufficient valid derived windows")
            informative_records = [record for record, keep in zip(records, informative) if keep]
            mean_prediction, burden_prediction, _, _ = linear_predictions(
                x_fit,
                x_validation,
                global_burden[fit_mask],
                global_burden[validation_mask],
            )
            mean_metrics = error_metrics(x_validation, mean_prediction, w_validation)
            burden_metrics = error_metrics(x_validation, burden_prediction, w_validation)
            pca = PCA(n_components=ranks[-1], svd_solver="full").fit(x_fit)
            fit_scores = pca.transform(x_fit)
            validation_scores = pca.transform(x_validation)
            validation_scores_by_fold[fold][scheme] = validation_scores.copy()
            for component in range(ranks[-1]):
                for sample_id, value in zip(validation_ids, validation_scores[:, component]):
                    score_rows.append([sample_id, fold, scheme, component + 1, value])
            fold_diagnostics = diagnostic_r2(
                fit_scores,
                validation_scores,
                ordered_covariates.iloc[np.flatnonzero(fit_mask)],
                ordered_covariates.iloc[np.flatnonzero(validation_mask)],
                global_burden[fit_mask],
                global_burden[validation_mask],
                global_callability[fit_mask],
                global_callability[validation_mask],
                max_components=ranks[-1],
            )
            for row in fold_diagnostics:
                diagnostic_rows.append({"outer_fold": fold, "scheme": scheme, **row})

            for rank in ranks:
                reconstruction = validation_scores[:, :rank] @ pca.components_[:rank] + pca.mean_
                metrics = error_metrics(x_validation, reconstruction, w_validation)
                weighted_skill = (
                    1.0
                    - metrics["callable_weighted_sse_sensitivity"]
                    / burden_metrics["callable_weighted_sse_sensitivity"]
                )
                partial_mask = np.asarray(
                    [not bool(record["is_partial"]) for record in informative_records], dtype=bool
                )
                if partial_mask.any():
                    partial_pca = error_metrics(
                        x_validation[:, partial_mask],
                        reconstruction[:, partial_mask],
                        w_validation[:, partial_mask],
                    )
                    partial_burden = error_metrics(
                        x_validation[:, partial_mask],
                        burden_prediction[:, partial_mask],
                        w_validation[:, partial_mask],
                    )
                    no_partial_skill = 1.0 - partial_pca["sse"] / partial_burden["sse"]
                    no_partial_weighted_skill = (
                        1.0
                        - partial_pca["callable_weighted_sse_sensitivity"]
                        / partial_burden["callable_weighted_sse_sensitivity"]
                    )
                else:
                    no_partial_skill = math.nan
                    no_partial_weighted_skill = math.nan
                fold_metric_rows.append(
                    {
                        "outer_fold": fold,
                        "scheme": scheme,
                        "scheme_role": informative_records[0]["scheme_role"],
                        "rank": rank,
                        "n_fit": int(fit_mask.sum()),
                        "n_validation": int(validation_mask.sum()),
                        "n_informative_windows": len(informative_records),
                        "median_window_bp": float(np.median([row["window_bp"] for row in informative_records])),
                        "median_fold_train_sites": float(
                            np.median([row["n_fold_train_rare_sites"] for row in informative_records])
                        ),
                        "explained_variance_ratio_cumulative": float(
                            pca.explained_variance_ratio_[:rank].sum()
                        ),
                        **metrics,
                        "mean_sse": mean_metrics["sse"],
                        "burden_sse": burden_metrics["sse"],
                        "skill_vs_mean": 1.0 - metrics["sse"] / mean_metrics["sse"],
                        "skill_vs_burden": 1.0 - metrics["sse"] / burden_metrics["sse"],
                        "burden_weighted_sse": burden_metrics[
                            "callable_weighted_sse_sensitivity"
                        ],
                        "weighted_skill_vs_burden": weighted_skill,
                        "skill_vs_burden_excluding_partial": no_partial_skill,
                        "weighted_skill_vs_burden_excluding_partial": no_partial_weighted_skill,
                    }
                )
                residual = x_validation - reconstruction
                burden_residual = x_validation - burden_prediction
                for sample_index, sample_id in enumerate(validation_ids):
                    weights = w_validation[sample_index]
                    sample_error_rows.append(
                        {
                            "sample_id": sample_id,
                            "outer_fold": fold,
                            "split_group_key": str(
                                validation_meta.loc[sample_index, "split_group_key"]
                            ),
                            "scheme": scheme,
                            "rank": rank,
                            "pca_sse": float(np.sum(residual[sample_index] ** 2)),
                            "burden_sse": float(np.sum(burden_residual[sample_index] ** 2)),
                            "pca_weighted_sse": float(
                                np.sum(weights * residual[sample_index] ** 2)
                            ),
                            "burden_weighted_sse": float(
                                np.sum(weights * burden_residual[sample_index] ** 2)
                            ),
                        }
                    )
                if rank == args.primary_rank:
                    for window_index, record in enumerate(informative_records):
                        weights = w_validation[:, window_index]
                        pca_window_sse = float(np.sum(residual[:, window_index] ** 2))
                        burden_window_sse = float(np.sum(burden_residual[:, window_index] ** 2))
                        contribution_rows.append(
                            {
                                "outer_fold": fold,
                                "scheme": scheme,
                                "derived_window_id": record["derived_window_id"],
                                "start_0based": record["start_0based"],
                                "end_0based": record["end_0based"],
                                "window_bp": record["window_bp"],
                                "is_partial": record["is_partial"],
                                "n_fold_train_rare_sites": record["n_fold_train_rare_sites"],
                                "pca_sse": pca_window_sse,
                                "burden_sse": burden_window_sse,
                                "delta_sse": burden_window_sse - pca_window_sse,
                                "weighted_delta_sse": float(
                                    np.sum(weights * burden_residual[:, window_index] ** 2)
                                    - np.sum(weights * residual[:, window_index] ** 2)
                                ),
                            }
                        )

    concordance_rows: list[dict] = []
    for fold in folds:
        reference = validation_scores_by_fold[fold][REFERENCE_SCHEME]
        for scheme in schemes:
            scores = validation_scores_by_fold[fold][scheme]
            for rank in ranks:
                metrics = subspace_metrics(reference[:, :rank].T, scores[:, :rank].T)
                concordance_rows.append(
                    {
                        "outer_fold": fold,
                        "scheme": scheme,
                        "rank": rank,
                        "score_subspace_overlap": metrics["overlap"],
                        "mean_angle_degrees": metrics["mean_angle_degrees"],
                        "max_angle_degrees": metrics["max_angle_degrees"],
                        "abs_rank1_score_correlation": (
                            float(abs(np.corrcoef(reference[:, 0], scores[:, 0])[0, 1]))
                            if rank == 1
                            else math.nan
                        ),
                    }
                )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    sample_errors = pd.DataFrame(sample_error_rows)
    contributions = pd.DataFrame(contribution_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    concordance = pd.DataFrame(concordance_rows)
    window_map = pd.DataFrame(map_rows)
    bootstrap = paired_group_bootstrap(
        sample_errors, schemes, ranks, args.bootstrap_replicates, args.seed
    )
    site_bias = summarize_site_bias(contributions)

    decisions: list[dict] = []
    bootstrap_rank1 = bootstrap.loc[bootstrap["rank"] == 1].set_index("scheme")
    for scheme in schemes:
        subset = fold_metrics.loc[(fold_metrics["scheme"] == scheme) & (fold_metrics["rank"] == 1)]
        row = bootstrap_rank1.loc[scheme]
        decisions.append(
            {
                "scheme": scheme,
                "all_four_folds_unweighted_positive": bool((subset["skill_vs_burden"] > 0).all()),
                "all_four_folds_weighted_positive": bool(
                    (subset["weighted_skill_vs_burden"] > 0).all()
                ),
                "unweighted_bootstrap_ci95_low_above_zero": bool(row["bootstrap_ci95_low"] > 0),
                "weighted_bootstrap_ci95_low_above_zero": bool(
                    row["weighted_bootstrap_ci95_low"] > 0
                ),
            }
        )
    decision_by_scheme = {row["scheme"]: row for row in decisions}
    new_core_pass = all(
        all(value for key, value in decision_by_scheme[scheme].items() if key != "scheme")
        for scheme in NEW_CORE_SCHEMES
    )
    boundary_consistent = all(
        decision_by_scheme[scheme]["all_four_folds_unweighted_positive"]
        and decision_by_scheme[scheme]["all_four_folds_weighted_positive"]
        for scheme in BOUNDARY_SCHEMES
    )
    final_decision = (
        "ELIGIBLE_FOR_INDEPENDENT_CONFIRMATION_NOT_NMF"
        if new_core_pass and boundary_consistent
        else "STOP_SCALE_ROBUSTNESS"
    )

    prefix = Path(args.out_prefix)
    fold_metrics.to_csv(f"{prefix}.multiscale_fold_metrics.tsv", sep="\t", index=False)
    sample_errors.to_csv(f"{prefix}.multiscale_sample_errors.tsv", sep="\t", index=False)
    bootstrap.to_csv(f"{prefix}.multiscale_bootstrap.tsv", sep="\t", index=False)
    diagnostics.to_csv(f"{prefix}.multiscale_diagnostics.tsv", sep="\t", index=False)
    concordance.to_csv(f"{prefix}.multiscale_score_concordance.tsv", sep="\t", index=False)
    site_bias.to_csv(f"{prefix}.multiscale_site_bias.tsv", sep="\t", index=False)
    window_map.to_csv(f"{prefix}.multiscale_window_map.tsv", sep="\t", index=False)
    with deterministic_gzip_writer(f"{prefix}.multiscale_window_contributions.tsv.gz") as handle:
        contributions.to_csv(handle, sep="\t", index=False, lineterminator="\n")
    with deterministic_gzip_writer(f"{prefix}.multiscale_oof_scores.tsv.gz") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample_id", "outer_fold", "scheme", "component", "score"])
        writer.writerows(score_rows)
    figures = plot_results(prefix, fold_metrics, bootstrap, site_bias, concordance, schemes, ranks)
    output_files = [
        f"{prefix}.multiscale_fold_metrics.tsv",
        f"{prefix}.multiscale_sample_errors.tsv",
        f"{prefix}.multiscale_bootstrap.tsv",
        f"{prefix}.multiscale_diagnostics.tsv",
        f"{prefix}.multiscale_score_concordance.tsv",
        f"{prefix}.multiscale_site_bias.tsv",
        f"{prefix}.multiscale_window_map.tsv",
        f"{prefix}.multiscale_window_contributions.tsv.gz",
        f"{prefix}.multiscale_oof_scores.tsv.gz",
        *figures,
    ]
    summary = {
        "status": "PASS",
        "stage": "M25C_RARE_WINDOW_SCALE_SENSITIVITY",
        "decision": final_decision,
        "scope": "Post-hoc chr22 scale/denominator sensitivity within canonical TRAIN; TEST remains blind",
        "schemes": list(schemes),
        "new_core_schemes": list(NEW_CORE_SCHEMES),
        "boundary_phase_schemes": list(BOUNDARY_SCHEMES),
        "ranks": list(ranks),
        "primary_rank": args.primary_rank,
        "scheme_decisions": decisions,
        "new_core_pass": new_core_pass,
        "boundary_phases_consistent": boundary_consistent,
        "n_test_genotypes_used": 0,
        "n_test_samples_scored": 0,
        "global_burden_policy": "Computed once per fold from original non-overlapping 250 kb atoms and reused in all schemes",
        "overlap_policy": "Half-step overlapping grids evaluated as separate non-overlapping phases; phases never concatenated",
        "bootstrap": {
            "unit": "split_group_key stratified by outer fold",
            "paired_across_all_schemes_and_ranks": True,
            "windows_resampled": False,
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "scope": "Conditional uncertainty for fitted chr22 models; no refit and no autosomal generalization",
        },
        "covariate_audit": covariate_audit,
        "interpretation_limit": "A pass supports internal robustness to discretization only. It does not validate biology, LAI improvement, NMF or AE.",
        "preregistration_sha256": sha256_file(args.preregistration),
        "input_manifest_container": extraction_manifest["container"],
        "inputs_sha256": {
            "fold_qc": sha256_file(args.fold_qc),
            "fold_manifest": sha256_file(args.fold_manifest),
            "split_manifest": sha256_file(args.split_manifest),
            "diagnostic_covariates": sha256_file(args.diagnostic_covariates),
            "diagnostic_covariates_audit": sha256_file(args.diagnostic_covariates_audit),
            "diagnostic_covariates_manifest": sha256_file(args.diagnostic_covariates_manifest),
            **{path.name: sha256_file(path) for path in feature_paths.values()},
            **{path.name: sha256_file(path) for path in window_paths.values()},
        },
        "outputs_sha256": {Path(path).name: sha256_file(path) for path in output_files},
        "preregistration": preregistration,
    }
    write_json(f"{prefix}.multiscale_summary.json", summary)
    print(json.dumps({"status": "PASS", "decision": final_decision}, sort_keys=True))


if __name__ == "__main__":
    main()
