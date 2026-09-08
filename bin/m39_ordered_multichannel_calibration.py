#!/usr/bin/env python3
"""Probability-only calibration comparator for the multichannel anchor pilot.

Historical transforms and TRAIN-neighbor grids are reused without changing the
historical calibration. This new comparator minimizes Brier in TRAIN and selects
one TRAIN-fitted family in SELECT; historical calibration minimized log-loss.
No SCORE path is accepted. SELECT remains reused exploratory development data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

import m39_calibration as historical
from m33_safe_bridge_core import write_deterministic_npz, write_exclusive_json
from m39_anchor_screen import metrics, sha256
from m39_ordered_context import OrderedContextStore
from m39_ordered_training_data import bind_development, require

SCHEMA = "m39-multichannel-calibration-v1"
SCOPE = "exploratory_chr22_R0_development_anchors_only"
SOURCE_FILES = ("m39_ordered_multichannel_calibration.py", "m39_calibration.py",
                "m39_ordered_training_data.py", "m39_ordered_context.py",
                "m39_ordered_models.py", "m39_anchor_screen.py", "m39_carrier_models.py",
                "m33_safe_bridge_core.py", "m34_prepare_panel_factors.py", "m34_generate_mosaics.py")
SELECTION_RULE = "minimum_brier_then_log_loss_with_1e-12_numeric_ties_then_family_dimension_order_parameter_distance"


def choose_brier(records: list[dict], prefix: str, tolerance: float) -> dict:
    """Numerical ties only; never infer practical equivalence from this tolerance."""
    require(bool(records) and tolerance == 1e-12, "empty candidates or tie policy differs")
    for name in (f"{prefix}_brier", f"{prefix}_log_loss"):
        require(all(math.isfinite(record[name]) for record in records), "nonfinite candidate metric")
        minimum = min(record[name] for record in records)
        records = [record for record in records if record[name] <= minimum + tolerance]
    return min(records, key=lambda record: (historical.DIMENSIONS[record["family"]],
        historical.FAMILIES.index(record["family"]),
        abs(math.log(record["temperature"])) + record["alpha"],
        record["temperature"], record["alpha"]))


def fit_families_brier(probabilities: np.ndarray, labels: np.ndarray, plan: dict,
                      *, check_budget=lambda: None) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Fit five families from TRAIN tensors only; no SELECT argument exists."""
    historical.validate_plan(plan)
    prior = historical.train_prior(labels, plan["prior_pseudocount"])
    candidates, finalists = [], []
    for family in historical.FAMILIES:
        temperatures = (plan["temperature_grid"] if family in ("temperature", "temperature_prior_mix") else [1.])
        alphas = (plan["alpha_grid"] if family in ("uniform_mix", "prior_mix", "temperature_prior_mix") else [0.])
        records, seen = [], set()

        def search(ts, mixtures, stage):
            for temperature, alpha in itertools.product(ts, mixtures):
                if (temperature, alpha) in seen:
                    continue
                check_budget()
                seen.add((temperature, alpha))
                transformed = historical.transform(probabilities, family, temperature, alpha, prior)
                report = metrics(transformed, labels)
                records.append({"family": family, "temperature": temperature, "alpha": alpha,
                                "stage": stage, "train_brier": report["brier"],
                                "train_log_loss": report["log_loss"]})

        search(temperatures, alphas, "initial")
        initial = choose_brier(records, "train", plan["tie_tolerance"])
        local_t = (historical.neighbors(temperatures, initial["temperature"], True,
                                       plan["temperature_endpoints"]) if len(temperatures) > 1 else temperatures)
        local_a = (historical.neighbors(alphas, initial["alpha"], False) if len(alphas) > 1 else alphas)
        search(local_t, local_a, "train_neighbor_refinement")
        best = choose_brier(records, "train", plan["tie_tolerance"]).copy()
        for record in records:
            record["train_finalist"] = (record["temperature"], record["alpha"]) == (best["temperature"], best["alpha"])
        finalists.append(best)
        candidates.extend(records)
    return prior, candidates, finalists


def select_family(probabilities: np.ndarray, labels: np.ndarray, prior: np.ndarray,
                  finalists: list[dict], tolerance: float) -> tuple[dict, list[dict]]:
    """Only evaluate frozen TRAIN finalists; SELECT does not tune their values."""
    require(len(finalists) == len(historical.FAMILIES)
            and {record["family"] for record in finalists} == set(historical.FAMILIES),
            "exactly one TRAIN finalist per family required")
    evaluated = []
    for record in finalists:
        p = historical.transform(probabilities, record["family"], record["temperature"], record["alpha"], prior)
        report = metrics(p, labels)
        evaluated.append({**record, "select_brier": report["brier"], "select_log_loss": report["log_loss"]})
    return choose_brier(evaluated, "select", tolerance).copy(), evaluated


def _anchors(indices: np.ndarray | None, count: int) -> np.ndarray:
    if indices is None:
        return np.arange(count, dtype=np.int64)
    require(isinstance(indices, np.ndarray) and indices.ndim == 1 and indices.dtype.kind in "iu"
            and len(indices) > 0 and np.all(indices < count)
            and np.all(indices >= 0) and np.all(np.diff(indices.astype(np.int64)) > 0),
            "anchor indices must be unique, ordered, in-range integers")
    return indices.astype(np.int64, copy=True)


def run_calibration(train_path: Path, select_path: Path, development_path: Path,
                    historical_plan_path: Path, outdir: Path, *, train_manifest_sha256: str,
                    select_manifest_sha256: str, development_sha256: str,
                    historical_plan_sha256: str, anchor_indices: np.ndarray | None = None,
                    max_runtime_seconds: int = 60) -> dict:
    """Authenticate development, fit on TRAIN, select on SELECT, and save new outputs.

    Stores are opened for exact identity/REF/axis binding, but their genotypes or
    common windows are never predictors. Fminus and Ffull use identical bound
    person/locus axes. Output directories and historical artifacts are never reused.
    ``anchor_indices`` denotes positions in the already authenticated anchor axis,
    not a genotype- or outcome-driven selection operation.
    """
    started = time.monotonic()
    require(type(max_runtime_seconds) is int and 0 < max_runtime_seconds <= 60,
            "calibration runtime ceiling must be an integer at most 60 seconds")
    outdir = Path(outdir)
    require(not outdir.exists() and not outdir.is_symlink(), "calibration output already exists")
    source_hashes = {name: sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}

    def check_budget():
        require(time.monotonic() - started <= max_runtime_seconds, "calibration time ceiling exceeded")

    plan_path = historical.authenticated(Path(historical_plan_path), historical_plan_sha256)
    plan = json.loads(plan_path.read_text())
    historical.validate_plan(plan)
    require(plan["development_sha256"] == development_sha256,
            "historical grid plan/development identity differs")
    train = OrderedContextStore.open(Path(train_path), expected_manifest_sha256=train_manifest_sha256)
    select = OrderedContextStore.open(Path(select_path), expected_manifest_sha256=select_manifest_sha256)
    bound = bind_development(Path(development_path), development_sha256, train, select)
    anchors = _anchors(anchor_indices, train.shape[1])
    check_budget()
    prior, candidates, finalists = fit_families_brier(bound["train"]["baseline"][:, anchors],
        bound["train"]["truth_state"][:, anchors], plan, check_budget=check_budget)
    selected, finalists = select_family(bound["select"]["baseline"][:, anchors],
        bound["select"]["truth_state"][:, anchors], prior, finalists, plan["tie_tolerance"])
    calibration = {"schema_version": SCHEMA, "scope": SCOPE, "family": selected["family"],
        "temperature": selected["temperature"], "alpha": selected["alpha"], "prior": prior.tolist(),
        "state_names": list(historical.STATE_NAMES), "parameters_fitted_on": "TRAIN",
        "prior_fitted_on": "TRAIN", "family_selected_on": "SELECT", "selection_rule": SELECTION_RULE,
        "primary_metric": "brier", "historical_primary_metric": "log_loss",
        "policy_changed_before_new_results": True, "refit_train_select": False,
        "probability_floor_added_to_predictions": False, "anchor_indices": anchors.tolist()}
    report = {"schema_version": SCHEMA, "scope": SCOPE,
        "status": "EXPLORATORY_CALIBRATION_ONLY_NOT_VALIDATION", "calibration": calibration,
        "role_counts": {role: store.shape[0] for role, store in (("TRAIN", train), ("SELECT", select))},
        "anchors": len(anchors), "candidate_count": len(candidates), "finalists": finalists,
        "selection_rule": SELECTION_RULE,
        "parameter_complexity_order": list(historical.FAMILIES),
        "source_sha256": source_hashes,
        "input_sha256": {"development": development_sha256, "historical_grid_plan": historical_plan_sha256,
                         "train_manifest": train_manifest_sha256, "select_manifest": select_manifest_sha256},
        "historical_grid": {key: plan[key] for key in ("temperature_grid", "alpha_grid", "temperature_endpoints",
                                                     "prior_pseudocount", "floor", "tie_tolerance")},
        "score_read": False, "source_test_or_valid_opened": False, "genotypes_used_as_predictors": False,
        "ordered_stores_opened_for_exact_binding": True, "global_boundary_F1_evaluated": False,
        "role_metrics": {}, "zero_probability_diagnostics": {}, "outputs_sha256": {}}
    predictions = {key: train.arrays[key][anchors].copy() for key in
                   ("chrom", "pos", "ref", "alt", "locus_id", "anchor_indices")}
    predictions.update(coords=train.arrays["cM"][anchors].copy(),
                       state_names=np.asarray(historical.STATE_NAMES, dtype="S2"))
    for role in ("train", "select"):
        y = bound[role]["truth_state"][:, anchors]
        baseline = bound[role]["baseline"][:, anchors]
        full = bound[role]["full_baseline"][:, anchors]
        calibrated = historical.transform(baseline, selected["family"], selected["temperature"],
                                          selected["alpha"], prior)
        arrays = {"FMINUS": baseline, "FFULL": full, "CALIBRATION_ONLY": calibrated}
        report["role_metrics"][role.upper()] = {name: metrics(p, y) for name, p in arrays.items()}
        report["zero_probability_diagnostics"][role.upper()] = {
            name: {"entries": int(p.size), "zero_entries": int(np.count_nonzero(p == 0)),
                   "observations": int(y.size), "true_zero_count": int(np.count_nonzero(
                       np.take_along_axis(p, y.astype(np.int64)[..., None], -1) == 0))}
            for name, p in arrays.items()}
        predictions.update({f"{role}_{name}": p for name, p in arrays.items()})
        predictions[f"{role}_truth_state"] = y.copy()
        predictions[f"{role}_sample_key_sha256"] = bound[role]["sample_key_sha256"].copy()
    check_budget()
    require(all(sha256(Path(__file__).with_name(name)) == digest for name, digest in source_hashes.items()),
            "calibration source changed during execution")
    outdir.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_exclusive_json(outdir / "calibration.json", calibration)
    historical.write_csv(outdir / "candidates.csv", candidates)
    write_deterministic_npz(outdir / "predictions.npz", predictions)
    # Reopen exactly what was saved, with the same metric implementation.
    with np.load(outdir / "predictions.npz", allow_pickle=False) as saved:
        require(set(saved.files) == set(predictions), "calibration prediction inventory changed")
        for key, value in predictions.items():
            require(np.array_equal(saved[key], value), f"calibration prediction axis/value differs: {key}")
        for role in ("train", "select"):
            for name in ("FMINUS", "FFULL", "CALIBRATION_ONLY"):
                recalculated = metrics(saved[f"{role}_{name}"], saved[f"{role}_truth_state"])
                require(recalculated == report["role_metrics"][role.upper()][name],
                        "calibration metric reopening differs")
    report["outputs_sha256"] = {name: sha256(outdir / name) for name in
                                ("calibration.json", "candidates.csv", "predictions.npz")}
    report["elapsed_seconds"] = time.monotonic() - started
    report["reopened_predictions_exact"] = True
    write_exclusive_json(outdir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "select", "development", "historical-plan", "outdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("train-manifest", "select-manifest", "development", "historical-plan"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--anchor-indices", type=Path,
                        help="Optional JSON array of fixed ordered anchor indices; omitted means all anchors")
    parser.add_argument("--max-runtime-seconds", type=int, default=60)
    args = parser.parse_args()
    indices = None
    if args.anchor_indices is not None:
        raw = json.loads(args.anchor_indices.read_text())
        require(isinstance(raw, list) and all(type(index) is int for index in raw),
                "anchor JSON must be an integer list")
        indices = np.asarray(raw, dtype=np.int64)
    report = run_calibration(args.train, args.select, args.development, args.historical_plan, args.outdir,
        train_manifest_sha256=args.train_manifest_sha256, select_manifest_sha256=args.select_manifest_sha256,
        development_sha256=args.development_sha256, historical_plan_sha256=args.historical_plan_sha256,
        anchor_indices=indices, max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({"status": report["status"], "family": report["calibration"]["family"],
                      "anchors": report["anchors"], "role_counts": report["role_counts"],
                      "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
