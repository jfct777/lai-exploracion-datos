#!/usr/bin/env python3
"""Audit whether the fixed M16.5-minor target has usable grouped support.

The audit is descriptive and single-pass.  It never trains a model, changes the
M16.5 partition, reads genotype data, or evaluates the held-out TEST fold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


CONTINUOUS = (
    "Q_NAM",
    "Q_EUR",
    "Q_EAS",
    "Q_AFR",
    "rare_density",
    "alt_carrier_density_historical",
    "rare_site_callability",
)
CATEGORICAL = ("cohort", "region", "state", "dominant_global_ancestry")
COMPARATORS = ("historical_lost", "nonassigned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minor-assignments", required=True, type=Path)
    parser.add_argument("--modeling-master", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", dtype={"sample_id": str})


def load_preregistration(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stage") != "M26_M16_TARGET_VIABILITY_AUDIT":
        raise SystemExit("Invalid M26 preregistration stage")
    folds = tuple(data["population"]["allowed_outer_folds"])
    if folds != (0, 1, 2, 4) or data["population"]["forbidden_outer_fold"] != 3:
        raise SystemExit("M26 preregistration must keep fold 3 outside the audit")
    return data


def assert_unique(frame: pd.DataFrame, name: str) -> None:
    if "sample_id" not in frame or frame["sample_id"].isna().any():
        raise SystemExit(f"{name} lacks valid sample_id")
    if frame["sample_id"].duplicated().any():
        raise SystemExit(f"{name} has duplicate sample_id")


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False})


def prepare_analysis(
    minor: pd.DataFrame,
    master: pd.DataFrame,
    split: pd.DataFrame,
    prereg: dict,
) -> tuple[pd.DataFrame, dict]:
    for frame, name in ((minor, "minor assignments"), (master, "modeling master"), (split, "split")):
        assert_unique(frame, name)

    required_master = {
        "sample_id", "community_res_1", "Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR",
        "cohort", "region", "state", "rare_density", "rare_carrier_site_count",
        "rare_gt_nonmissing_sites", "rare_missing_sites", "qc_red",
    }
    required_split = {"sample_id", "eligible", "fold", "split", "split_group_key", "y"}
    required_minor = {"sample_id", "community_res_1"}
    for observed, required, name in (
        (set(master), required_master, "modeling master"),
        (set(split), required_split, "split"),
        (set(minor), required_minor, "minor assignments"),
    ):
        missing = required - observed
        if missing:
            raise SystemExit(f"{name} missing columns: {sorted(missing)}")

    if not set(minor["sample_id"]).issubset(set(master["sample_id"])):
        raise SystemExit("Minor assignments contain samples outside modeling master")
    if set(master["sample_id"]) != set(split["sample_id"]):
        raise SystemExit("Modeling master and split sample universes differ")

    split = split.copy()
    split["eligible"] = as_bool(split["eligible"])
    split["fold"] = pd.to_numeric(split["fold"], errors="coerce")
    allowed = set(prereg["population"]["allowed_outer_folds"])
    analytic_split = split.loc[
        split["eligible"].eq(True)
        & split["split"].eq("TRAIN")
        & split["fold"].isin(allowed)
    ].copy()
    expected_n = int(prereg["population"]["expected_train_validation_samples"])
    if len(analytic_split) != expected_n:
        raise SystemExit(f"Expected {expected_n} TRAIN/VALIDATION rows; observed {len(analytic_split)}")
    if analytic_split["fold"].astype(int).eq(3).any():
        raise SystemExit("Forbidden fold 3 entered the analytic cohort")

    selected = master.merge(
        analytic_split[["sample_id", "fold", "split_group_key", "y"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != expected_n:
        raise SystemExit("TRAIN/VALIDATION join changed cardinality")
    qc_red = as_bool(selected["qc_red"])
    if qc_red.isna().any() or qc_red.any():
        raise SystemExit("QC-red sample entered the analytic cohort")

    minor_label = minor[["sample_id", "community_res_1"]].rename(
        columns={"community_res_1": "minor_community"}
    )
    selected = selected.rename(columns={"community_res_1": "historical_community"}).merge(
        minor_label, on="sample_id", how="left", validate="one_to_one"
    )
    selected["historical_assigned"] = selected["historical_community"].ge(0)
    selected["minor_assigned"] = selected["minor_community"].ge(0)
    if not (selected["y"].astype(int).eq(selected["historical_assigned"].astype(int))).all():
        raise SystemExit("Frozen split target disagrees with historical M16.5 assignment")
    if (selected["minor_assigned"] & ~selected["historical_assigned"]).any():
        raise SystemExit("Minor target contains positives absent from historical target")

    selected["target_state"] = np.select(
        [
            selected["minor_assigned"],
            selected["historical_assigned"] & ~selected["minor_assigned"],
        ],
        ["minor_assigned", "historical_lost"],
        default="nonassigned",
    )
    denom = selected["rare_gt_nonmissing_sites"] + selected["rare_missing_sites"]
    selected["alt_carrier_density_historical"] = (
        selected["rare_carrier_site_count"] / selected["rare_gt_nonmissing_sites"]
    )
    selected["rare_site_callability"] = selected["rare_gt_nonmissing_sites"] / denom
    q_cols = ["Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR"]
    selected["dominant_global_ancestry"] = (
        selected[q_cols].astype(float).idxmax(axis=1).str.replace("Q_", "", regex=False)
    )
    selected["fold"] = selected["fold"].astype(int)
    # split_group_key is the canonicalized phi0442 component actually used by
    # StratifiedGroupKFold.  The older modeling-master component identifier is
    # order-dependent and must not replace the frozen split unit.
    selected["split_group_key"] = selected["split_group_key"].astype(str)

    integrity = {
        "n_master": int(len(master)),
        "n_split": int(len(split)),
        "n_minor_graph_nodes": int(len(minor)),
        "n_train_validation": int(len(selected)),
        "allowed_outer_folds_observed": sorted(map(int, selected["fold"].unique())),
        "n_forbidden_test_rows_analyzed": 0,
        "n_qc_red_analyzed": 0,
        "sample_ids_emitted": False,
        "pass": True,
    }
    return selected, integrity


def effective_group_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n_individuals": 0,
            "n_groups": 0,
            "effective_groups": 0.0,
            "max_group_share": None,
        }
    shares = frame["split_group_key"].value_counts(normalize=True)
    return {
        "n_individuals": int(len(frame)),
        "n_groups": int(len(shares)),
        "effective_groups": float(1.0 / np.square(shares.to_numpy(dtype=float)).sum()),
        "max_group_share": float(shares.max()),
    }


def support_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    community_rows = []
    folds = sorted(data["fold"].unique())
    communities = sorted(data.loc[data["minor_assigned"], "minor_community"].astype(int).unique())
    for fold in folds:
        for partition, mask in (
            ("assessment", data["fold"].eq(fold)),
            ("fit", data["fold"].ne(fold)),
        ):
            part = data.loc[mask]
            positives = part.loc[part["minor_assigned"]]
            fold_rows.append(
                {"outer_fold": int(fold), "partition": partition, **effective_group_metrics(positives)}
            )
            for community in communities:
                members = part.loc[part["minor_community"].eq(community)]
                community_rows.append(
                    {
                        "outer_fold": int(fold),
                        "partition": partition,
                        "minor_community": int(community),
                        **effective_group_metrics(members),
                    }
                )
    return pd.DataFrame(fold_rows), pd.DataFrame(community_rows)


def state_counts(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_label, frame in [("pooled", data)] + [
        (f"fold_{fold}", data.loc[data["fold"].eq(fold)]) for fold in sorted(data["fold"].unique())
    ]:
        counts = frame["target_state"].value_counts()
        for state in ("minor_assigned", "historical_lost", "nonassigned"):
            rows.append(
                {
                    "scope": fold_label,
                    "target_state": state,
                    "n_samples": int(counts.get(state, 0)),
                    "fraction": float(counts.get(state, 0) / len(frame)),
                }
            )
    return pd.DataFrame(rows)


def directional_auc(positive: pd.Series, negative: pd.Series) -> float | None:
    x = pd.to_numeric(positive, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(negative, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(x) or not len(y):
        return None
    pooled = np.concatenate([x, y])
    ranks = rankdata(pooled, method="average")[: len(x)]
    auc = (ranks.sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
    return float(max(auc, 1.0 - auc))


def standardized_mean_difference(left: pd.Series, right: pd.Series) -> float | None:
    x = pd.to_numeric(left, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(right, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return None
    pooled_sd = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return None if pooled_sd == 0 else float((np.mean(x) - np.mean(y)) / pooled_sd)


def continuous_effects(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("pooled", data)] + [
        (f"fold_{fold}", data.loc[data["fold"].eq(fold)]) for fold in sorted(data["fold"].unique())
    ]
    for scope, frame in scopes:
        positive = frame.loc[frame["target_state"].eq("minor_assigned")]
        for comparator in COMPARATORS:
            negative = frame.loc[frame["target_state"].eq(comparator)]
            for variable in CONTINUOUS:
                x = pd.to_numeric(positive[variable], errors="coerce").dropna()
                y = pd.to_numeric(negative[variable], errors="coerce").dropna()
                rows.append(
                    {
                        "scope": scope,
                        "comparator": comparator,
                        "variable": variable,
                        "n_positive": int(len(x)),
                        "n_comparator": int(len(y)),
                        "positive_median": float(x.median()) if len(x) else None,
                        "positive_q25": float(x.quantile(0.25)) if len(x) else None,
                        "positive_q75": float(x.quantile(0.75)) if len(x) else None,
                        "comparator_median": float(y.median()) if len(y) else None,
                        "comparator_q25": float(y.quantile(0.25)) if len(y) else None,
                        "comparator_q75": float(y.quantile(0.75)) if len(y) else None,
                        "standardized_mean_difference": standardized_mean_difference(x, y),
                        "directional_auc": directional_auc(x, y),
                    }
                )
    return pd.DataFrame(rows)


def categorical_composition(data: pd.DataFrame, suppress_n: int) -> pd.DataFrame:
    rows = []
    scopes = [("pooled", data)] + [
        (f"fold_{fold}", data.loc[data["fold"].eq(fold)]) for fold in sorted(data["fold"].unique())
    ]
    for scope, frame in scopes:
        for variable in CATEGORICAL:
            background = frame[variable].fillna("MISSING").astype(str).value_counts()
            for state in ("minor_assigned", "historical_lost", "nonassigned"):
                subset = frame.loc[frame["target_state"].eq(state), variable].fillna("MISSING").astype(str)
                counts = subset.value_counts()
                for category, count in counts.items():
                    rows.append(
                        {
                            "scope": scope,
                            "variable": variable,
                            "target_state": state,
                            "category": category if count >= suppress_n else "SUPPRESSED_LT_N",
                            "n_samples": int(count) if count >= suppress_n else None,
                            "state_share": float(count / len(subset)) if len(subset) else None,
                            "background_share": float(background.get(category, 0) / len(frame)),
                            "suppressed": bool(count < suppress_n),
                        }
                    )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.groupby(
            ["scope", "variable", "target_state", "category", "suppressed"],
            as_index=False,
            dropna=False,
        ).agg(
            n_samples=("n_samples", "sum"),
            state_share=("state_share", "sum"),
            background_share=("background_share", "sum"),
        )
    return result


def decide(
    data: pd.DataFrame,
    fold_support: pd.DataFrame,
    community_support: pd.DataFrame,
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    prereg: dict,
) -> tuple[str, list[str], dict]:
    thresholds = prereg["decision_thresholds"]
    reasons: list[str] = []
    assessment = fold_support.loc[fold_support["partition"].eq("assessment")]
    if (assessment["n_individuals"] < thresholds["min_positive_individuals_per_assessment_fold"]).any():
        reasons.append("INSUFFICIENT_POSITIVES_IN_ASSESSMENT_FOLD")
    if (assessment["effective_groups"] < thresholds["min_effective_kinship_groups_per_assessment_fold"]).any():
        reasons.append("INSUFFICIENT_EFFECTIVE_GROUPS_IN_ASSESSMENT_FOLD")
    if (assessment["max_group_share"] >= thresholds["max_single_kinship_group_share"]).fillna(True).any():
        reasons.append("KINSHIP_GROUP_DOMINATES_ASSESSMENT_FOLD")

    community_assessment = community_support.loc[community_support["partition"].eq("assessment")]
    if (community_assessment["n_individuals"] == 0).any():
        reasons.append("COMMUNITY_ABSENT_FROM_ASSESSMENT_FOLD")
    if (community_assessment["n_individuals"] < thresholds["min_community_individuals_per_assessment_fold"]).any():
        reasons.append("COMMUNITY_SUPPORT_BELOW_DECISION_FLOOR")
    if (community_assessment["effective_groups"] < thresholds["min_community_effective_groups_per_assessment_fold"]).any():
        reasons.append("COMMUNITY_EFFECTIVE_GROUPS_BELOW_DECISION_FLOOR")
    if (community_assessment["max_group_share"] >= thresholds["max_single_kinship_group_share"]).fillna(True).any():
        reasons.append("COMMUNITY_DOMINATED_BY_KINSHIP_GROUP")

    auc_cutoff = float(thresholds["near_deterministic_univariate_auc"])
    near_deterministic = []
    for variable in CONTINUOUS:
        for comparator in COMPARATORS:
            rows = continuous.loc[
                continuous["variable"].eq(variable)
                & continuous["comparator"].eq(comparator)
                & continuous["scope"].str.startswith("fold_")
            ]
            values = rows["directional_auc"].dropna()
            if len(values) == 4 and (values >= auc_cutoff).all():
                near_deterministic.append(f"{variable}:{comparator}")
    if near_deterministic:
        reasons.append("KNOWN_CONTINUOUS_CONFOUNDER_NEAR_DETERMINISTIC_IN_ALL_FOLDS")

    positive_share_cutoff = float(thresholds["categorical_alias_positive_share"])
    max_background = float(thresholds["categorical_alias_max_background_share"])
    min_folds = int(thresholds["categorical_alias_min_folds"])
    categorical_alias = []
    for variable in CATEGORICAL:
        folded = categorical.loc[
            categorical["variable"].eq(variable)
            & categorical["target_state"].eq("minor_assigned")
            & categorical["scope"].str.startswith("fold_")
            & ~categorical["suppressed"]
        ]
        qualifying = folded.loc[
            (folded["state_share"] >= positive_share_cutoff)
            & (folded["background_share"] < max_background)
        ]
        if qualifying["scope"].nunique() >= min_folds:
            categorical_alias.append(variable)
    if categorical_alias:
        reasons.append("KNOWN_CATEGORICAL_CONFOUNDER_ALIASES_TARGET")

    reasons = sorted(set(reasons))
    verdict = "USABLE_EXPLORATORY" if not reasons else "CLOSE_INTERNAL_TARGET"
    diagnostics = {
        "near_deterministic_continuous": near_deterministic,
        "categorical_aliases": categorical_alias,
        "n_minor_assigned_train_validation": int(data["minor_assigned"].sum()),
        "n_minor_communities_train_validation": int(data.loc[data["minor_assigned"], "minor_community"].nunique()),
    }
    return verdict, reasons, diagnostics


def main() -> None:
    args = parse_args()
    prereg = load_preregistration(args.preregistration)
    minor = read_tsv(args.minor_assignments)
    master = read_tsv(args.modeling_master)
    split = read_tsv(args.split_manifest)
    data, integrity = prepare_analysis(minor, master, split, prereg)

    fold_support, community_support = support_tables(data)
    counts = state_counts(data)
    continuous = continuous_effects(data)
    suppress_n = int(prereg["decision_thresholds"]["small_cell_suppression_n"])
    categorical = categorical_composition(data, suppress_n)
    verdict, reasons, diagnostics = decide(
        data, fold_support, community_support, continuous, categorical, prereg
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    tables = {
        "m16_target_state_counts.tsv": counts,
        "m16_target_fold_support.tsv": fold_support,
        "m16_target_community_fold_support.tsv": community_support,
        "m16_target_continuous_effects.tsv": continuous,
        "m16_target_categorical_composition.tsv": categorical,
    }
    for filename, table in tables.items():
        table.to_csv(args.outdir / filename, sep="\t", index=False)

    report = {
        "stage": "M26_M16_TARGET_VIABILITY_AUDIT",
        "status": "PASS",
        "verdict": verdict,
        "decision_reasons": reasons,
        "scope": "fixed_m16_5_minor_internal_target_train_validation_only",
        "integrity": integrity,
        "diagnostics": diagnostics,
        "decision_thresholds": prereg["decision_thresholds"],
        "feature_semantics": {
            "rare_density": "historical ALT dosage per nonmissing rare site",
            "alt_carrier_density_historical": "historical ALT-carrier sites per nonmissing rare site",
            "rare_site_callability": "nonmissing genotypes divided by nonmissing plus missing sites in the upstream rare-site universe",
            "minor_allele_burden": "not available and not inferred in this audit",
        },
        "interpretation_limits": [
            "M16.5-minor is internal and transductive because the graph was built on all 2619 individuals.",
            "The audit tests grouped support and obvious selection bias, not biological validity.",
            "No TEST rows, sample identifiers, genotype data, model fitting, reclustering or parameter tuning enter the reported analysis.",
            "A USABLE_EXPLORATORY verdict would require a new PRE before any downstream training.",
            "A CLOSE_INTERNAL_TARGET verdict closes this label, not rare variants, diadic sharing or LAI.",
        ],
    }
    (args.outdir / "m16_target_viability_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
