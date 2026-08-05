#!/usr/bin/env python3
"""Aggregate chromosome orientation inventories and quantify marginal impact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MODES = (
    "historical_alt",
    "minor_filter_cohort",
    "minor_m14_subset",
    "exclude_alt_major_filter_cohort",
    "exclude_alt_major_m14_subset",
)
EXPECTED_CHROMS = tuple(map(str, range(1, 23)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--expected-chr22-alt-major-m14", type=int, default=429)
    parser.add_argument("--expected-chr22-sites", type=int, default=470033)
    parser.add_argument("--outdir", type=Path, default=Path("."))
    return parser.parse_args()


def chrom_key(value: str) -> int:
    return int(str(value).removeprefix("chr"))


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return None
    return float(pd.Series(left).corr(pd.Series(right), method=method))


def comparison(historical: np.ndarray, current: np.ndarray) -> dict:
    delta = current.astype(np.float64) - historical.astype(np.float64)
    slope = intercept = None
    if len(np.unique(historical)) >= 2:
        slope, intercept = map(float, np.polyfit(historical.astype(float), current.astype(float), 1))
    rank_h = pd.Series(historical).rank(method="average").to_numpy()
    rank_c = pd.Series(current).rank(method="average").to_numpy()
    return {
        "pearson": safe_corr(historical, current, "pearson"),
        "spearman": safe_corr(historical, current, "spearman"),
        "linear_slope_current_on_historical": slope,
        "linear_intercept_current_on_historical": intercept,
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(math.sqrt(np.mean(delta ** 2))),
        "delta_current_minus_historical_quantiles": {
            str(q): float(np.quantile(delta, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
        "absolute_rank_shift_quantiles": {
            str(q): float(np.quantile(np.abs(rank_c - rank_h), q))
            for q in (0.5, 0.95, 1.0)
        },
        "n_individuals_changed": int(np.count_nonzero(delta)),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(
        args.input_dir.glob("chr*.orientation_inventory.json"),
        key=lambda path: chrom_key(path.name.split(".", 1)[0]),
    )
    burden_paths = sorted(
        args.input_dir.glob("chr*.sample_burden_by_mode.tsv.gz"),
        key=lambda path: chrom_key(path.name.split(".", 1)[0]),
    )
    manifest_paths = sorted(
        args.input_dir.glob("chr*.orientation_inventory.manifest.json"),
        key=lambda path: chrom_key(path.name.split(".", 1)[0]),
    )
    if len(summary_paths) != 22 or len(burden_paths) != 22 or len(manifest_paths) != 22:
        raise SystemExit(
            "Expected 22 summaries, burdens and manifests; found "
            f"{len(summary_paths)}, {len(burden_paths)}, {len(manifest_paths)}"
        )

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    observed_chroms = tuple(str(item["chrom"]) for item in summaries)
    if observed_chroms != EXPECTED_CHROMS:
        raise SystemExit(f"Chromosome set/order {observed_chroms} != {EXPECTED_CHROMS}")
    if any(item.get("status") != "PASS" for item in summaries):
        raise SystemExit("At least one chromosome inventory is not PASS")
    if any(item["reference_qc"]["mismatches"] != 0 for item in summaries):
        raise SystemExit("At least one chromosome has a reference mismatch")
    for universe in ("filter_cohort", "m14_subset", "train_subset"):
        hashes = {item["sample_order_sha256"][universe] for item in summaries}
        if len(hashes) != 1:
            raise SystemExit(f"Sample identity/order hash differs across chromosomes for {universe}")

    chr22 = summaries[-1]
    chr22_m14_counts = chr22["orientation_universes"]["m14_subset"]["counts"]
    if chr22_m14_counts.get("alt_major", 0) != args.expected_chr22_alt_major_m14:
        raise SystemExit(
            "chr22 regression gate failed: M14-subset ALT-major "
            f"{chr22_m14_counts.get('alt_major', 0)} != {args.expected_chr22_alt_major_m14}"
        )
    if chr22_m14_counts.get("total_sites", 0) != args.expected_chr22_sites:
        raise SystemExit(
            f"chr22 site gate failed: {chr22_m14_counts.get('total_sites', 0)} "
            f"!= {args.expected_chr22_sites}"
        )

    per_chrom_rows = []
    global_universe_counts = {name: {} for name in ("filter_cohort", "m14_subset")}
    global_burden_totals = {mode: {} for mode in MODES}
    for item in summaries:
        row = {
            "chrom": item["chrom"],
            "n_sites": item["expected_sites_from_m14_summary"],
            "wall_seconds": item["resource_usage"]["analysis_wall_seconds"],
            "self_max_rss_kib": item["resource_usage"]["self_max_rss_kib"],
            "child_max_rss_kib": item["resource_usage"]["largest_reaped_child_max_rss_kib"],
        }
        for universe in ("filter_cohort", "m14_subset"):
            universe_data = item["orientation_universes"][universe]
            row[f"{universe}_alt_major"] = universe_data["counts"].get("alt_major", 0)
            row[f"{universe}_alt_major_fraction_orientable"] = universe_data[
                "alt_major_fraction_of_orientable"
            ]
            for key, value in universe_data["counts"].items():
                global_universe_counts[universe][key] = (
                    global_universe_counts[universe].get(key, 0) + int(value)
                )
        for mode in MODES:
            for key, value in item["burden_totals"][mode].items():
                global_burden_totals[mode][key] = global_burden_totals[mode].get(key, 0) + int(value)
        per_chrom_rows.append(row)
    pd.DataFrame(per_chrom_rows).to_csv(
        args.outdir / "autosomes.orientation_by_chromosome.tsv", sep="\t", index=False
    )

    aggregate_burden = None
    for path in burden_paths:
        current = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
        if aggregate_burden is None:
            aggregate_burden = current.copy()
            continue
        if current["sample_id"].tolist() != aggregate_burden["sample_id"].tolist():
            raise SystemExit(f"Sample identity/order differs in {path.name}")
        if current["is_train"].tolist() != aggregate_burden["is_train"].tolist():
            raise SystemExit(f"TRAIN mask differs in {path.name}")
        numeric_columns = [column for column in current.columns if column not in ("sample_id", "is_train")]
        aggregate_burden[numeric_columns] = (
            aggregate_burden[numeric_columns].to_numpy(dtype=np.int64)
            + current[numeric_columns].to_numpy(dtype=np.int64)
        )
    assert aggregate_burden is not None

    for mode in MODES:
        dosage_column = f"{mode}_dosage_sum"
        carrier_column = f"{mode}_carrier_site_count"
        callable_column = f"{mode}_callable_sites"
        if int(aggregate_burden[dosage_column].sum()) != global_burden_totals[mode]["dosage_m14"]:
            raise SystemExit(f"Dosage reconciliation failed for {mode}")
        if int(aggregate_burden[carrier_column].sum()) != global_burden_totals[mode]["carrier_incidence_m14"]:
            raise SystemExit(f"Carrier reconciliation failed for {mode}")
        aggregate_burden[f"{mode}_dosage_per_callable_site"] = (
            aggregate_burden[dosage_column] / aggregate_burden[callable_column].replace(0, np.nan)
        )
        aggregate_burden[f"{mode}_carrier_rate"] = (
            aggregate_burden[carrier_column] / aggregate_burden[callable_column].replace(0, np.nan)
        )
    aggregate_burden.to_csv(
        args.outdir / "autosomes.sample_burden_by_mode.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )

    comparisons = {}
    historical_dosage = aggregate_burden["historical_alt_dosage_sum"].to_numpy(dtype=np.int64)
    historical_carriers = aggregate_burden["historical_alt_carrier_site_count"].to_numpy(dtype=np.int64)
    for mode in MODES[1:]:
        comparisons[mode] = {
            "dosage": comparison(
                historical_dosage,
                aggregate_burden[f"{mode}_dosage_sum"].to_numpy(dtype=np.int64),
            ),
            "carrier_sites": comparison(
                historical_carriers,
                aggregate_burden[f"{mode}_carrier_site_count"].to_numpy(dtype=np.int64),
            ),
        }

    universe_summaries = {}
    for universe, counts in global_universe_counts.items():
        orientable = counts.get("alt_minor", 0) + counts.get("alt_major", 0)
        total = counts["total_sites"]
        universe_summaries[universe] = {
            "counts": counts,
            "orientable_sites": orientable,
            "alt_major_fraction_of_orientable": counts.get("alt_major", 0) / orientable,
            "alt_major_fraction_of_total": counts.get("alt_major", 0) / total,
        }

    historical = global_burden_totals["historical_alt"]
    corrected = global_burden_totals["minor_filter_cohort"]
    historical_filter_orientable_dosage = historical[
        "dosage_m14_on_filter_cohort_orientable"
    ]
    historical_filter_orientable_carriers = historical[
        "carrier_incidence_m14_on_filter_cohort_orientable"
    ]
    impact = {
        "status": "PASS",
        "scope": "22_autosomes_orientation_and_marginal_impact",
        "n_chromosomes": 22,
        "n_m14_samples": int(len(aggregate_burden)),
        "n_train_samples": int(aggregate_burden["is_train"].sum()),
        "orientation_universes": universe_summaries,
        "burden_totals": global_burden_totals,
        "autosomal_sample_comparisons_vs_historical_alt": comparisons,
        "m20_impact": {
            "status": "quantified_for_marginal_alt_burden_fields",
            "historical_alt_dosage_sum": historical["dosage_m14"],
            "minor_filter_cohort_dosage_sum": corrected["dosage_m14"],
            "historical_alt_carrier_incidence": historical["carrier_incidence_m14"],
            "minor_filter_cohort_carrier_incidence": corrected["carrier_incidence_m14"],
            "historical_dosage_fraction_at_filter_cohort_alt_major": (
                historical["dosage_m14_at_filter_cohort_alt_major"]
                / historical_filter_orientable_dosage
            ),
            "historical_carrier_incidence_fraction_at_filter_cohort_alt_major": (
                historical["carrier_incidence_m14_at_filter_cohort_alt_major"]
                / historical_filter_orientable_carriers
            ),
            "unchanged_fields": ["rare_gt_nonmissing_sites", "rare_missing_sites"],
            "not_quantified_here": [
                "M14-derived n_sharing_partners", "n_segments_involved", "total_shared_bp",
                "n_chromosomes_with_sharing", "flag_aislado",
            ],
        },
        "m23_impact": {
            "status": "matrix_mass_quantified_performance_not_recomputed",
            "historical_train_nnz": historical["nnz_train"],
            "minor_filter_cohort_train_nnz": corrected["nnz_train"],
            "historical_train_dosage": historical["dosage_train"],
            "minor_filter_cohort_train_dosage": corrected["dosage_train"],
            "historical_columns_before_missingness_filter": historical["retained_sites"],
            "minor_filter_cohort_columns_before_missingness_filter": corrected["retained_sites"],
            "historical_columns_after_10pct_missingness_filter": (
                historical["retained_sites"] - historical["sites_with_missing_rate_train_gt_0_1"]
            ),
            "minor_filter_cohort_columns_after_10pct_missingness_filter": (
                corrected["retained_sites"] - corrected["sites_with_missing_rate_train_gt_0_1"]
            ),
            "sites_with_mac_train_ge_2_historical": historical["sites_with_mac_train_ge_2"],
            "sites_with_mac_train_ge_2_minor_filter_cohort": corrected["sites_with_mac_train_ge_2"],
            "not_quantified_here": [
                "Elastic Net coefficients", "convergence", "balanced accuracy", "AUROC",
                "effect of the corrected M14-derived target",
            ],
        },
        "m14_m16_5_m22_impact": {
            "status": "not_quantifiable_from_marginal_inventory",
            "known": "chr22 topology changed materially in the frozen M24 sensitivity run",
            "requires": "corrected M14 windows/segments on chr1-21 followed by M16.5 reconstruction",
            "do_not_infer_from": ["site fractions", "burden correlations", "marginal dosage changes"],
        },
        "regression_gates": {
            "chr22_m14_subset_alt_major": args.expected_chr22_alt_major_m14,
            "chr22_total_sites": args.expected_chr22_sites,
        },
        "interpretation_limits": [
            "Additive burden changes do not determine pairwise or spatial M14 topology.",
            "No model was trained or evaluated and TEST was not used as an outcome.",
            "The historical M22/M23 results remain reproducible results for ALT-coded inputs and an M14-derived historical target.",
        ],
    }
    (args.outdir / "autosomes.orientation_impact.json").write_text(
        json.dumps(impact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
