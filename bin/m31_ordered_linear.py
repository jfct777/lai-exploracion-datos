#!/usr/bin/env python3
"""Numerical core and contract guard for the M31 ordered-linear experiment.

The module intentionally depends only on NumPy and the Python standard library.
It does not read validation/test data and it does not choose hyperparameters.
Those responsibilities belong to the caller's root-local nested-CV workflow.

Ring convention
---------------
For an anchor at ``g`` and a preregistered ring ``[inner, outer]``, membership
uses signed half-open intervals: left is ``[g-outer, g-inner)`` and right is
``[g+inner, g+outer)``.  Thus every exact boundary belongs to one ring only and
a site at the anchor belongs to the right side of the innermost ring.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

import numpy as np


EXPERIMENT_ID = "M31_ORDERED_LINEAR_DEV"
SCHEMA_VERSION = "1.1.0"
STATUS = "PREREGISTERED_NOT_RUN"
SIDES = ("left", "right")
EXPECTED_RINGS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0))
EXPECTED_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
EXPECTED_BOUNDARY_WEIGHTS = (1.0, 5.0, 20.0)
ANCESTRIES = ("AFR", "EUR", "ASIA")
ROOTS = ("root17", "root18")
SHAM_REPLICATES = 32
BOUNDARY_TOLERANCES_CM = (0.1, 0.2, 0.5)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 31002001
DIPLOID_CLASSES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
EXPECTED_RUN_INPUT_SHA256 = {
    "genetic_map": "33c7a94e0cbc0ce3cc3ff83cd3838119a881cb7e838825521a778e07a01ee6e9",
    "root17.sites": "f23686f123abec9bfe14fcf3d88e3954e32fa6206707d0cc2d2b68bb9d360c4e",
    "root17.target": "d9ad299ea7f5cc2b0a92b5397b743b03b1300fd8e59fa71b247b6959eb3837ec",
    "root17.tree": "566946ec93cbc28c7f36d27878e12d091f1c87ebdd6fb6423b940f4f778391f7",
    "root17.pools": "56f9891bfb7f77d63ec3328107fc8e89b919c34ad16fda2c0c72dc3fde24f0e4",
    "root17.truth": "bf059f94ba40b033a75bb2d4ea782ce2d8f6cb86098a77e4812ee019f07da303",
    "root17.flare_vcf": "85dfd76df2c14cb8fe0a753910f25c49c88d38edc5708ec6d641053d95cc74e8",
    "root17.flare_audit": "4273247ff352effc0ffb09797d3d634cde338519c1f858e2deab1e0adcbf8a5d",
    "root18.sites": "eec6a6a3cfbe0f832618c69451a6bf588b4d60a5b734e05fc8acc343849f1f9b",
    "root18.target": "c6622c0bbaa6657187f6488a6fcfd523ab90b7d628da4cac03adc9476a9ca6d5",
    "root18.tree": "780800c1f3d279746ace8631fa1b1ccd57782e9bf697a3dacc2cc983523f5f54",
    "root18.pools": "046c6e903ca4b3beea8ac468836d28bb953d32b4e731c13be86f42d7c110a4ff",
    "root18.truth": "e5e2895b8316673afb6796795a2724d167c1bf2e98b2122ef0f7645b985e296a",
    "root18.flare_vcf": "edc4bcdc62f5ce0ffe04bd27e9d6d6ee892e03282a1474639fc3082fbc3832c9",
    "root18.flare_audit": "ff328cab1fc77fd8b503f5782e9886e60b3c10aa8d52650b7fec5e209a9a7342",
}


class ContractError(ValueError):
    """Raised when the preregistered M31 contract has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _finite_number(value: Any, label: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             f"{label} must be a number")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def parse_rings(raw: Any) -> tuple[tuple[float, float], ...]:
    """Parse non-overlapping, contiguous, non-negative half-open rings."""
    _require(isinstance(raw, list) and raw, "signed_half_open_rings_cM must be a non-empty list")
    rings: list[tuple[float, float]] = []
    for index, pair in enumerate(raw):
        _require(isinstance(pair, list) and len(pair) == 2, f"ring {index} must have two endpoints")
        inner = _finite_number(pair[0], f"ring {index} inner")
        outer = _finite_number(pair[1], f"ring {index} outer")
        _require(0.0 <= inner < outer, f"ring {index} must satisfy 0 <= inner < outer")
        if rings:
            _require(inner == rings[-1][1], "rings must be exactly contiguous and non-overlapping")
        rings.append((inner, outer))
    _require(rings[0][0] == 0.0, "the first ring must start at zero")
    return tuple(rings)


@dataclass(frozen=True)
class RootDirection:
    name: str
    train_seed: int
    evaluation_seed: int


@dataclass(frozen=True)
class OrderedLinearContract:
    experiment_id: str
    directions: tuple[RootDirection, ...]
    rings_cm: tuple[tuple[float, float], ...]
    alphas: tuple[float, ...]
    boundary_weights: tuple[float, ...]
    boundary_definition_cm: float


def parse_contract(payload: Mapping[str, Any]) -> OrderedLinearContract:
    """Validate load-bearing preregistration fields and return typed values."""
    _require(isinstance(payload, Mapping), "contract must be a JSON object")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "unexpected experiment_id")
    _require(payload.get("status") == STATUS, "contract is not preregistered as NOT_RUN")
    _require(payload.get("preregistration_revision") == {
        "revision": "PRE_1",
        "date": "2026-08-20",
        "supersedes_schema_version": "1.0.0",
        "reason": "Before any fitted M31 run, freeze Voronoi-cM row weights, guarded primary-endpoint inner-CV selection, global-count scoring, and the additive H ceiling.",
    }, "PRE revision metadata drifted")
    _require(payload.get("scope") == "chr22_two_root_development_only_no_validation",
             "M31 scope drifted or permits validation")
    _require(payload.get("roots_are_not_independent_validation") is True,
             "the two directions must not be treated as independent validation")
    _require(payload.get("execution_post") == {
        "full_run": "BLOCKED_PENDING_SEPARATE_POST",
        "pilot_direction": "train_root17_predict_root18_only",
        "pilot_fitted_arms": ["C", "L", "D"],
        "pilot_excluded": ["H", "DSHAM", "HSHAM", "scientific_decision"],
        "process_barrier": "fit-predict_fsync_manifest_and_exit_before_score-pilot_accepts_root18_truth",
        "resume_unit": "hash_bound_complete_arm_checkpoint",
        "pilot_label": "NO_SCIENTIFIC_DECISION",
    }, "execution POST protocol drifted")

    roots = payload.get("roots")
    _require(isinstance(roots, Mapping) and len(roots) == 2, "contract must define exactly two directions")
    directions: list[RootDirection] = []
    for name, pair in roots.items():
        _require(isinstance(name, str) and name, "root direction name must be non-empty")
        _require(isinstance(pair, list) and len(pair) == 2, f"root direction {name} must contain two seeds")
        _require(all(isinstance(seed, int) and not isinstance(seed, bool) for seed in pair),
                 f"root direction {name} seeds must be integers")
        _require(pair[0] != pair[1], f"root direction {name} cannot train and evaluate on the same root")
        directions.append(RootDirection(name, pair[0], pair[1]))
    _require({(item.train_seed, item.evaluation_seed) for item in directions}
             == {(20260817, 20260818), (20260818, 20260817)},
             "root directions differ from the frozen reciprocal development roots")

    baseline = payload.get("baseline")
    _require(isinstance(baseline, Mapping), "baseline is missing")
    _require(baseline.get("primary") == "FLARE_v0.6.0_frozen_M30", "baseline primary drifted")
    _require(baseline.get("historical_guardrail") == "Gnomix_frozen_M29", "historical guardrail drifted")
    _require(baseline.get("marker_count") == 79791, "exact M30 marker count drifted")
    _require(baseline.get("prediction_support") == "the exact phased target marker grid and Voronoi genetic cells authenticated by M30", "baseline prediction support drifted")

    rare = payload.get("rare_universe")
    _require(isinstance(rare, Mapping), "rare_universe is missing")
    _require(rare.get("selector") == "FREQ_only", "rare-site selection must be FREQ-only")
    _require(rare.get("definition") == "minor allele with MAC>=2, MAF<0.01 and at least two FREQ carrier individuals",
             "rare-site definition drifted")
    _require(rare.get("minor_presence") == "I(observed_state == minor_code)",
             "minor-allele orientation contract drifted")
    _require(rare.get("unsupported_in_REF_LAI") == "retain the FREQ-selected site and expose an explicit no-support mask",
             "REF_LAI unsupported-site policy drifted")
    forbidden = rare.get("forbidden_selectors")
    _require(forbidden == ["REF_LAI", "DONOR", "TARGET", "truth", "FLARE_prediction", "Gnomix_prediction"],
             "forbidden selector set/order drifted")

    representation = payload.get("ordered_representation")
    _require(isinstance(representation, Mapping), "ordered_representation is missing")
    _require(representation.get("coordinate") == "genetic_cM", "ordered coordinate must be genetic cM")
    _require(representation.get("evaluation_grid") == "exact_M30_FLARE_markers_no_interpolated_grid",
             "evaluation grid must remain the exact M30 FLARE marker grid")
    rings = parse_rings(representation.get("signed_half_open_rings_cM"))
    _require(rings == EXPECTED_RINGS, "signed ring grid differs from preregistration")
    _require(representation.get("sides") == list(SIDES), "ring side order must be left, right")
    _require(representation.get("normalization") == ["genetic_length", "callable_or_observed_site_count"],
             "ring normalizations drifted")
    _require(representation.get("edge_masks") is True, "edge masks must be retained")
    _require(representation.get("frequency_weighting") == "none_no_inverse_MAF",
             "inverse-frequency weighting is forbidden")

    arms = payload.get("arms")
    expected_arms = {
        "F0": "frozen FLARE haplotype posterior without a fitted corrector",
        "C": "residual corrector using only FLARE probabilities and ordered common context",
        "L": "C plus local diploid rare load without reference ancestry labels",
        "D": "L plus ancestry-specific REF_LAI rare support; diploid target dose is shared across haplotypes",
        "H": "D plus phase-aware rare presence by target haplotype; simulation ceiling only",
        "DSHAM": "D with complete diploid REF_LAI ancestry labels permuted",
        "HSHAM": "H with complete diploid REF_LAI ancestry labels permuted",
    }
    _require(arms == expected_arms, "arm definitions drifted")

    model = payload.get("model")
    _require(isinstance(model, Mapping), "model is missing")
    _require(model.get("estimator") == "multivariate_L2_ridge_residual", "estimator drifted")
    _require(model.get("prediction") == "simplex_projection(FLARE_haplotype_posterior + X_beta)",
             "prediction rule drifted")
    _require(model.get("posthoc_smoothing") is False, "post-hoc smoothing is forbidden")
    alphas = tuple(_finite_number(value, "alpha") for value in model.get("alphas", []))
    weights = tuple(_finite_number(value, "boundary training weight")
                    for value in model.get("boundary_training_weights", []))
    _require(alphas == EXPECTED_ALPHAS and all(value > 0.0 for value in alphas), "ridge alpha grid drifted")
    _require(weights == EXPECTED_BOUNDARY_WEIGHTS and all(value > 0.0 for value in weights),
             "boundary training weights drifted")
    boundary_cm = _finite_number(model.get("boundary_training_definition_cM"),
                                 "boundary_training_definition_cM")
    _require(boundary_cm == 0.2, "boundary training definition drifted")
    _require(model.get("training_row_weight") == "Voronoi_genetic_cM * haplotype_specific_boundary_multiplier",
             "training row-weight definition drifted")
    _require(model.get("weight_normalization") == "sum_to_number_of_diploid_individuals",
             "weight normalization drifted")
    _require(model.get("feature_dtype_policy") == "materialize_float32_for_TRAIN_CV_EVAL_accumulate_and_solve_float64",
             "feature dtype policy drifted")
    _require(model.get("feature_standardization") == "fit_on_training_individuals_only",
             "feature standardization would permit leakage")
    _require(model.get("inner_cv") == "three deterministic folds grouped by complete diploid individual",
             "inner CV must be deterministic grouped three-fold")
    _require(model.get("hyperparameter_selection")
             == "performed independently inside the training root for every fitted real or sham arm",
             "hyperparameter selection scope drifted")
    _require(model.get("inner_cv_primary") == "OOF_boundary_F1_at_0.2_cM",
             "inner-CV primary endpoint drifted")
    _require(model.get("inner_cv_guardrails") == [
        "macro_ancestry_dose_MAE_not_worse_than_OOF_F0",
        "false_transitions_per_cM_at_0.2_cM_not_worse_than_OOF_F0",
        "NO_GUARDED_CONFIG_if_empty",
    ], "inner-CV guardrails drifted")
    _require(model.get("inner_cv_tie_break") == ["min_false_transitions_per_cM_at_0.2_cM", "min_macro_ancestry_dose_MAE", "min_haplotype_Brier", "min_boundary_training_weight", "max_alpha"],
             "inner-CV tie-break drifted")

    null = payload.get("null")
    _require(isinstance(null, Mapping) and null.get("replicates") == SHAM_REPLICATES, "sham replicate count drifted")
    _require(null.get("unit") == "complete_diploid_REF_LAI_individual", "sham unit is not diploid individual")
    _require(null.get("changed") == "mapping between REF_LAI individuals and ancestry labels", "sham changed quantity drifted")
    _require(null.get("fixed") == ["loci", "positions", "target_genotypes", "target_phase", "missingness", "rare_load", "LD", "ancestry_sample_sizes"], "sham invariants drifted")
    _require(null.get("formal_resolution") == "if the real statistic exceeds all 32 shams, exact exploratory p=1/33; do not report a percentile-95 significance claim", "sham formal resolution drifted")

    evaluation = payload.get("evaluation")
    _require(isinstance(evaluation, Mapping), "evaluation is missing")
    _require(evaluation.get("primary") == "boundary_F1_at_0.2_cM", "primary endpoint drifted")
    _require(evaluation.get("sensitivities_cM") == [0.1, 0.5], "boundary sensitivities drifted")
    _require(evaluation.get("secondary") == ["false_transitions_per_cM", "matched_boundary_median_cM", "matched_boundary_p90_cM", "macro_ancestry_dose_MAE", "MAE_by_ancestry", "haplotype_Brier", "diploid_macro_F1"], "secondary metrics drifted")
    _require(evaluation.get("unit_of_uncertainty") == "diploid_individual",
             "uncertainty unit must be the diploid individual")
    _require(evaluation.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES, "bootstrap replicate count drifted")
    _require(evaluation.get("point_estimate_aggregation") == "sum_global_sufficient_counts_then_compute_metrics",
             "point-estimate aggregation drifted")
    _require(evaluation.get("bootstrap_aggregation") == "resample_complete_diploid_individual_count_bundles_then_reconstruct_global_metrics",
             "bootstrap aggregation drifted")
    _require(evaluation.get("prediction_truth_boundary") == "hash_immutable_truth_blind_prediction_artifact_before_truth_scoring",
             "prediction/truth boundary drifted")
    _require(evaluation.get("directions_are_not_independent_replicates") is True,
             "root directions must not be pooled as independent replicates")

    expected_stop_rules = [
        "any input or code SHA-256 mismatch",
        "minor-allele known answers fail",
        "marker, sample, phase, locus-order or root identity mismatch",
        "any split separates haplotypes or loci from the same individual",
        "normalization uses the evaluation root",
        "any sham changes a quantity listed as fixed",
        "ridge solve or simplex projection is non-finite",
        "any pilot checkpoint, prediction or manifest SHA-256 mismatch",
        "fit-predict accepts or reads evaluation-root truth",
        "full reciprocal run requested before a separate runner POST",
        "any VALID or TEST input is accessed",
    ]
    _require(payload.get("stop_rules") == expected_stop_rules, "stop-rule set/order drifted")

    decision = payload.get("decision")
    expected_decision = {
        "NO_GUARDED_CONFIG": "at least one fitted real or sham arm has no hyperparameter configuration satisfying both OOF F0 guardrails; no GO label is permitted",
        "GO_NEW_ROOTS": "D improves boundary F1 over F0, C and L in both directions, exceeds every DSHAM improvement, does not increase macro MAE or false transitions, and no ancestry worsens in both directions",
        "LOAD_ONLY": "L improves but D does not improve over L",
        "PHASE_CEILING_ONLY": "H passes its sham and guardrails while D does not",
        "TRADEOFF": "boundary F1 improves but at least one prespecified guardrail fails",
        "STOP_LINEAR_ORDERED_RARE": "neither D nor H shows replicated incremental benefit",
        "next_step_after_GO": "generate prospectively frozen new simulation roots before CNN or any confirmatory claim",
    }
    _require(decision == expected_decision, "decision definitions drifted")
    _require(payload.get("claims_excluded") == ["confirmatory_validation", "DNABR_generalization", "Native_American_LAI", "Brazil_novel_variant_effect", "deep_learning_benefit"], "excluded claims drifted")

    return OrderedLinearContract(
        EXPERIMENT_ID,
        tuple(sorted(directions, key=lambda item: item.name)),
        rings,
        alphas,
        weights,
        boundary_cm,
    )


def load_contract(path: str | Path) -> OrderedLinearContract:
    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read M31 contract {contract_path}: {exc}") from exc
    return parse_contract(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def project_simplex(values: Any, axis: int = -1) -> np.ndarray:
    """Euclidean projection onto the unit probability simplex along ``axis``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("simplex input must have at least one dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError("simplex input must be finite")
    try:
        moved = np.moveaxis(array, axis, -1)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid simplex axis {axis}") from exc
    if moved.shape[-1] == 0:
        raise ValueError("simplex axis cannot be empty")
    flat = moved.reshape(-1, moved.shape[-1])
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    ranks = np.arange(1, flat.shape[1] + 1, dtype=np.float64)
    positive = ordered - cumulative / ranks > 0.0
    rho = positive.sum(axis=1) - 1
    theta = cumulative[np.arange(flat.shape[0]), rho] / (rho + 1.0)
    projected = np.maximum(flat - theta[:, None], 0.0)
    # Repair only roundoff in the affine constraint, without changing support.
    maxima = np.argmax(projected, axis=1)
    residual = 1.0 - projected.sum(axis=1)
    projected[np.arange(projected.shape[0]), maxima] += residual
    return np.moveaxis(projected.reshape(moved.shape), -1, axis)


@dataclass(frozen=True)
class RingAggregation:
    """Aggregates with axes ``marker, side, ring, *value_channels``."""

    sums: np.ndarray
    by_genetic_length: np.ndarray
    by_observed_site_count: np.ndarray
    observed_site_count: np.ndarray
    genetic_length_cm: np.ndarray
    edge_mask: np.ndarray
    rings_cm: tuple[tuple[float, float], ...]
    sides: tuple[str, str] = SIDES


def _ordered_coordinates(values: Any, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    if np.any(np.diff(result) < 0.0):
        raise ValueError(f"{label} must be nondecreasing")
    return result


def aggregate_signed_half_open_rings(
    marker_cm: Any,
    site_cm: Any,
    site_values: Any,
    rings_cm: Sequence[Sequence[float]] = EXPECTED_RINGS,
    *,
    observed: Any | None = None,
    domain_cm: tuple[float, float] | None = None,
) -> RingAggregation:
    """Aggregate site values around markers in signed half-open genetic rings.

    ``site_values`` has site as its first axis and may have arbitrary trailing
    sample/haplotype/channel axes. Missing cells can be represented by NaN or by
    an explicit boolean ``observed`` mask of identical shape.
    """
    markers = _ordered_coordinates(marker_cm, "marker_cm")
    sites = _ordered_coordinates(site_cm, "site_cm")
    values = np.asarray(site_values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != sites.size:
        raise ValueError("site_values first dimension must equal site_cm length")
    if observed is None:
        observed_array = np.isfinite(values)
    else:
        observed_array = np.asarray(observed)
        if observed_array.dtype != np.bool_ or observed_array.shape != values.shape:
            raise ValueError("observed must be a boolean array with the same shape as site_values")
        if np.any(observed_array & ~np.isfinite(values)):
            raise ValueError("observed site_values must be finite")
    safe_values = np.where(observed_array, values, 0.0)
    parsed_rings = parse_rings([[float(pair[0]), float(pair[1])] for pair in rings_cm])

    if domain_cm is None:
        domain_start = float(min(markers[0], sites[0]))
        domain_end = float(max(markers[-1], sites[-1]))
    else:
        if not isinstance(domain_cm, tuple) or len(domain_cm) != 2:
            raise ValueError("domain_cm must be a (start, end) tuple")
        domain_start = _finite_number(domain_cm[0], "domain start")
        domain_end = _finite_number(domain_cm[1], "domain end")
    if not domain_start < domain_end:
        raise ValueError("domain_cm must have positive length")
    if sites[0] < domain_start or sites[-1] > domain_end or markers[0] < domain_start or markers[-1] > domain_end:
        raise ValueError("domain_cm must contain all markers and sites")

    trailing = values.shape[1:]
    base_shape = (markers.size, len(SIDES), len(parsed_rings))
    output_shape = base_shape + trailing
    sums = np.empty(output_shape, dtype=np.float64)
    counts = np.empty(output_shape, dtype=np.int64)
    lengths = np.empty(base_shape, dtype=np.float64)
    edges = np.empty(base_shape, dtype=np.bool_)
    value_prefix = np.concatenate((np.zeros((1,) + trailing), np.cumsum(safe_values, axis=0)), axis=0)
    count_prefix = np.concatenate((np.zeros((1,) + trailing, dtype=np.int64),
                                   np.cumsum(observed_array, axis=0, dtype=np.int64)), axis=0)

    for marker_index, marker in enumerate(markers):
        for ring_index, (inner, outer) in enumerate(parsed_rings):
            intervals = ((marker - outer, marker - inner), (marker + inner, marker + outer))
            for side_index, (lower, upper) in enumerate(intervals):
                first = int(np.searchsorted(sites, lower, side="left"))
                last = int(np.searchsorted(sites, upper, side="left"))
                sums[(marker_index, side_index, ring_index)] = value_prefix[last] - value_prefix[first]
                counts[(marker_index, side_index, ring_index)] = count_prefix[last] - count_prefix[first]
                clipped = max(0.0, min(upper, domain_end) - max(lower, domain_start))
                width = outer - inner
                lengths[marker_index, side_index, ring_index] = clipped
                edges[marker_index, side_index, ring_index] = clipped < width - 1e-12

    length_denominator = lengths.reshape(base_shape + (1,) * len(trailing))
    by_length = np.divide(sums, length_denominator, out=np.zeros_like(sums), where=length_denominator > 0.0)
    by_count = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    for array in (sums, by_length, by_count, counts, lengths, edges):
        array.setflags(write=False)
    return RingAggregation(sums, by_length, by_count, counts, lengths, edges, parsed_rings)


# Short public alias for callers that already encode the half-open convention.
aggregate_signed_rings = aggregate_signed_half_open_rings


@dataclass(frozen=True)
class GroupedFold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    validation_samples: tuple[str, ...]


def _sample_vector(sample_ids: Sequence[str], expected_length: int | None = None) -> np.ndarray:
    samples = np.asarray(sample_ids, dtype=object)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("sample_ids must be a non-empty one-dimensional sequence")
    if expected_length is not None and samples.size != expected_length:
        raise ValueError("sample_ids length differs from observation count")
    if any(not isinstance(value, str) or not value for value in samples.tolist()):
        raise ValueError("sample_ids must contain non-empty strings")
    return samples


def grouped_three_fold_ids(sample_ids: Sequence[str], seed: int = 0) -> np.ndarray:
    """Return stable fold IDs, keeping every row of an individual together."""
    samples = _sample_vector(sample_ids)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    unique = sorted(set(samples.tolist()))
    if len(unique) < 3:
        raise ValueError("grouped three-fold CV requires at least three individuals")

    def key(sample: str) -> tuple[bytes, str]:
        token = f"{seed}\0{sample}".encode("utf-8")
        return hashlib.sha256(token).digest(), sample

    ordered = sorted(unique, key=key)
    assignment = {sample: index % 3 for index, sample in enumerate(ordered)}
    result = np.fromiter((assignment[sample] for sample in samples), dtype=np.int8, count=samples.size)
    return result


def grouped_three_fold_split(sample_ids: Sequence[str], seed: int = 0) -> tuple[GroupedFold, ...]:
    samples = _sample_vector(sample_ids)
    fold_ids = grouped_three_fold_ids(samples.tolist(), seed=seed)
    folds: list[GroupedFold] = []
    for fold in range(3):
        validation = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        validation_samples = tuple(sorted(set(samples[validation].tolist())))
        if not len(validation) or not len(train):
            raise AssertionError("grouped fold is empty")
        if set(samples[train].tolist()) & set(validation_samples):
            raise AssertionError("a diploid individual leaked across a grouped fold")
        folds.append(GroupedFold(fold, train, validation, validation_samples))
    return tuple(folds)


grouped_3fold_split = grouped_three_fold_split


def normalize_weights(
    weights: Any,
    sample_ids: Sequence[str],
    *,
    target_total: float | None = None,
) -> np.ndarray:
    """Normalize non-negative row weights to the diploid-individual count."""
    raw = np.asarray(weights, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    samples = _sample_vector(sample_ids, expected_length=raw.size)
    if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    expected = float(len(set(samples.tolist()))) if target_total is None else float(target_total)
    if not math.isfinite(expected) or expected <= 0.0:
        raise ValueError("target_total must be finite and positive")
    result = raw * (expected / total)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("normalized weights are non-finite")
    return result


@dataclass(frozen=True)
class WeightedStandardizedRidgeResidual:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_intercept: np.ndarray
    coefficients: np.ndarray
    alpha: float
    normalized_weight_sum: float
    n_individuals: int

    def _standardize(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_mean.size:
            raise ValueError("prediction features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        return (values - self.feature_mean) / self.feature_scale

    def predict_residual(self, features: Any) -> np.ndarray:
        result = self.residual_intercept + self._standardize(features) @ self.coefficients
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("ridge residual prediction is non-finite")
        return result

    def predict(self, features: Any, baseline_probabilities: Any) -> np.ndarray:
        baseline = np.asarray(baseline_probabilities, dtype=np.float64)
        residual = self.predict_residual(features)
        if baseline.shape != residual.shape or not np.all(np.isfinite(baseline)):
            raise ValueError("baseline probabilities must be finite and match ridge outputs")
        return project_simplex(baseline + residual, axis=1)


def fit_weighted_standardized_ridge_residual(
    features: Any,
    residual_targets: Any,
    sample_ids: Sequence[str],
    *,
    weights: Any | None = None,
    alpha: float = 1.0,
) -> WeightedStandardizedRidgeResidual:
    """Fit weighted multivariate ridge on training-only standardized features."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(residual_targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] == 0:
        raise ValueError("residual_targets must be a non-empty matrix with matching rows")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("training matrices must be finite")
    samples = _sample_vector(sample_ids, expected_length=x.shape[0])
    ridge_alpha = _finite_number(alpha, "alpha")
    if ridge_alpha <= 0.0:
        raise ValueError("alpha must be positive")
    raw_weights = np.ones(x.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    normalized = normalize_weights(raw_weights, samples.tolist())
    weight_sum = float(normalized.sum())

    feature_mean = np.sum(normalized[:, None] * x, axis=0) / weight_sum
    centered = x - feature_mean
    variance = np.sum(normalized[:, None] * centered * centered, axis=0) / weight_sum
    feature_scale = np.sqrt(np.maximum(variance, 0.0))
    feature_scale[feature_scale <= np.finfo(np.float64).eps] = 1.0
    standardized = centered / feature_scale
    residual_intercept = np.sum(normalized[:, None] * y, axis=0) / weight_sum
    centered_targets = y - residual_intercept
    gram = standardized.T @ (normalized[:, None] * standardized)
    gram.flat[:: gram.shape[0] + 1] += ridge_alpha
    rhs = standardized.T @ (normalized[:, None] * centered_targets)
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError("ridge solve failed") from exc
    if not all(np.all(np.isfinite(array)) for array in
               (feature_mean, feature_scale, residual_intercept, coefficients)):
        raise FloatingPointError("ridge fit is non-finite")
    for array in (feature_mean, feature_scale, residual_intercept, coefficients):
        array.setflags(write=False)
    return WeightedStandardizedRidgeResidual(
        feature_mean,
        feature_scale,
        residual_intercept,
        coefficients,
        ridge_alpha,
        weight_sum,
        len(set(samples.tolist())),
    )


weighted_standardized_multivariate_ridge_residual = fit_weighted_standardized_ridge_residual


def fit_ridge_corrector(
    features: Any,
    truth_probabilities: Any,
    baseline_probabilities: Any,
    sample_ids: Sequence[str],
    *,
    weights: Any | None = None,
    alpha: float = 1.0,
) -> WeightedStandardizedRidgeResidual:
    """Convenience wrapper that forms truth-minus-FLARE residual targets."""
    truth = np.asarray(truth_probabilities, dtype=np.float64)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != baseline.shape:
        raise ValueError("truth and baseline probabilities must be matching matrices")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(baseline)):
        raise ValueError("truth and baseline probabilities must be finite")
    if np.any(truth < 0.0) or np.any(baseline < 0.0):
        raise ValueError("probabilities cannot be negative")
    if not np.allclose(truth.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("truth probabilities must sum to one")
    if not np.allclose(baseline.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("baseline probabilities must sum to one")
    return fit_weighted_standardized_ridge_residual(
        features, truth - baseline, sample_ids, weights=weights, alpha=alpha
    )


def open_text(path: str | Path) -> TextIO:
    source = Path(path)
    return gzip.open(source, "rt", encoding="utf-8", newline="") if source.suffix == ".gz" else source.open("r", encoding="utf-8", newline="")


def _normalize_chrom(value: str) -> str:
    return value.removeprefix("chr")


@dataclass(frozen=True)
class GeneticMap:
    positions: np.ndarray
    cms: np.ndarray

    def cm_at(self, positions: Any) -> np.ndarray | float:
        query = np.asarray(positions, dtype=np.float64)
        if np.any(query < self.positions[0]) or np.any(query > self.positions[-1]):
            raise ValueError("physical position lies outside the authenticated genetic map")
        result = np.interp(query, self.positions, self.cms)
        return float(result) if query.ndim == 0 else result


def load_genetic_map(path: str | Path, chrom: str = "22") -> GeneticMap:
    points: list[tuple[int, float]] = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3 or _normalize_chrom(fields[0]) != _normalize_chrom(chrom):
                raise ValueError(f"genetic map row {line_number} has an unexpected chromosome/header")
            points.append((int(fields[1]), float(fields[2])))
    if len(points) < 2:
        raise ValueError("genetic map requires at least two points")
    positions = np.asarray([item[0] for item in points], dtype=np.int64)
    cms = np.asarray([item[1] for item in points], dtype=np.float64)
    if np.any(np.diff(positions) <= 0) or np.any(np.diff(cms) < 0) or not np.all(np.isfinite(cms)):
        raise ValueError("genetic map must be finite and ordered")
    return GeneticMap(positions, cms)


@dataclass(frozen=True)
class RareInput:
    root_seed: int
    positions: np.ndarray
    minor_codes: np.ndarray
    samples: tuple[str, ...]
    hap_presence: np.ndarray  # site x sample x hap, NaN means missing


def load_ordered_rare(sites_path: str | Path, target_path: str | Path, expected_seed: int) -> RareInput:
    site_rows: list[dict[str, str]] = []
    with open_text(sites_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"root_seed", "locus_index", "chrom", "position", "minor_code", "mac", "an", "maf", "freq_carrier_individuals"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("M31 sites table has an unexpected header")
        site_rows = list(reader)
    if not site_rows:
        raise ValueError("M31 sites table is empty")
    positions = np.asarray([int(row["position"]) for row in site_rows], dtype=np.int64)
    locus_indexes = [int(row["locus_index"]) for row in site_rows]
    seeds = {int(row["root_seed"]) for row in site_rows}
    minor = np.asarray([int(row["minor_code"]) for row in site_rows], dtype=np.int8)
    if seeds != {expected_seed} or locus_indexes != list(range(len(site_rows))):
        raise ValueError("M31 site root/locus identity is not exact")
    if np.any(np.diff(positions) <= 0) or np.any((minor != 0) & (minor != 1)):
        raise ValueError("M31 sites must be strictly ordered biallelic loci")
    for row in site_rows:
        mac, an, maf, carriers = int(row["mac"]), int(row["an"]), float(row["maf"]), int(row["freq_carrier_individuals"])
        if mac < 2 or carriers < 2 or not 0.0 < maf < 0.01 or not math.isclose(mac / an, maf, abs_tol=1e-12):
            raise ValueError("M31 sites table violates the frozen FREQ-only rare universe")

    rows: dict[tuple[int, str], tuple[float, float]] = {}
    samples: set[str] = set()
    with open_text(target_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"root_seed", "sample_id", "locus_index", "chrom", "position", "minor_code", "h0_minor_presence", "h1_minor_presence", "minor_dosage", "missing_haplotypes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("M31 target table has an unexpected header")
        previous_key: tuple[int, str] | None = None
        for row in reader:
            index, sample = int(row["locus_index"]), row["sample_id"]
            key = (index, sample)
            if not sample or key in rows or (previous_key is not None and key <= previous_key):
                raise ValueError("M31 target rows are duplicated or not site-major/sample-ordered")
            previous_key = key
            if int(row["root_seed"]) != expected_seed or not 0 <= index < len(site_rows):
                raise ValueError("M31 target root/locus identity mismatch")
            if int(row["position"]) != positions[index] or int(row["minor_code"]) != int(minor[index]):
                raise ValueError("M31 target position/minor_code differs from sites table")
            pair: list[float] = []
            for name in ("h0_minor_presence", "h1_minor_presence"):
                value = row[name]
                pair.append(float("nan") if value in {"", ".", "NA"} else float(int(value)))
            if any(np.isfinite(value) and value not in (0.0, 1.0) for value in pair):
                raise ValueError("M31 target presence is not binary/missing")
            missing = sum(not np.isfinite(value) for value in pair)
            if missing != int(row["missing_haplotypes"]):
                raise ValueError("M31 target missingness count mismatch")
            if missing == 0 and int(row["minor_dosage"]) != int(sum(pair)):
                raise ValueError("M31 target minor dosage mismatch")
            if missing and row["minor_dosage"] not in {"", ".", "NA"}:
                raise ValueError("partially missing diploid dosage must remain missing")
            rows[key] = (pair[0], pair[1])
            samples.add(sample)
    ordered_samples = tuple(sorted(samples))
    expected_keys = {(index, sample) for index in range(len(site_rows)) for sample in ordered_samples}
    if set(rows) != expected_keys:
        raise ValueError("M31 target table is not the complete sites-by-samples product")
    hap = np.asarray([[rows[(index, sample)] for sample in ordered_samples] for index in range(len(site_rows))], dtype=np.float64)
    return RareInput(expected_seed, positions, minor, ordered_samples, hap)


@dataclass(frozen=True)
class FlareInput:
    loci: tuple[tuple[str, int, str, str], ...]
    samples: tuple[str, ...]
    probabilities: np.ndarray  # marker x sample x hap x ancestry


def load_flare(path: str | Path) -> FlareInput:
    ancestry_codes: dict[str, str] = {}
    required_format_headers = {
        "AN1": '##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">',
        "AN2": '##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">',
        "ANP1": '##FORMAT=<ID=ANP1,Number=3,Type=Float,Description="Ancestry probabilities for first haplotype">',
        "ANP2": '##FORMAT=<ID=ANP2,Number=3,Type=Float,Description="Ancestry probabilities for second haplotype">',
    }
    seen_format_headers: dict[str, str] = {}
    samples: tuple[str, ...] | None = None
    loci: list[tuple[str, int, str, str]] = []
    rows: list[np.ndarray] = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##ANCESTRY=<"):
                for token in line.strip()[len("##ANCESTRY=<"):-1].split(","):
                    ancestry, code = token.split("=", 1)
                    ancestry_codes[code] = ancestry
                continue
            if line.startswith("##FORMAT=<ID="):
                identifier = line.split("ID=", 1)[1].split(",", 1)[0]
                if identifier in required_format_headers:
                    seen_format_headers[identifier] = line.rstrip("\n")
                continue
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                if seen_format_headers != required_format_headers:
                    raise ValueError("FLARE AN1/AN2/ANP1/ANP2 headers do not bind first/second haplotypes exactly")
                values = line.rstrip("\n").split("\t")[9:]
                samples = tuple(values)
                if not samples or len(samples) != len(set(samples)):
                    raise ValueError("FLARE sample IDs are missing or duplicated")
                continue
            if line.startswith("#"):
                continue
            if samples is None or ancestry_codes != {"0": "AFR", "1": "EUR", "2": "ASIA"}:
                raise ValueError("FLARE header/order must be AFR=0, EUR=1, ASIA=2")
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 + len(samples):
                raise ValueError(f"malformed FLARE VCF row {line_number}")
            locus = (_normalize_chrom(fields[0]), int(fields[1]), fields[3], fields[4])
            if locus[0] != "22" or "," in locus[3] or (loci and locus[1] <= loci[-1][1]):
                raise ValueError("FLARE loci are not strictly ordered chr22 biallelic markers")
            fmt = fields[8].split(":")
            if not all(name in fmt for name in ("AN1", "AN2", "ANP1", "ANP2")):
                raise ValueError("FLARE FORMAT lacks AN1/AN2/ANP1/ANP2")
            indexes = {name: fmt.index(name) for name in ("AN1", "AN2", "ANP1", "ANP2")}
            matrix = np.empty((len(samples), 2, 3), dtype=np.float64)
            for sample_index, raw_sample in enumerate(fields[9:]):
                values = raw_sample.split(":")
                for hap, (probability_name, hard_name) in enumerate((("ANP1", "AN1"), ("ANP2", "AN2"))):
                    probability = np.asarray([float(value) for value in values[indexes[probability_name]].split(",")])
                    if probability.shape != (3,) or np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
                        raise ValueError("invalid FLARE probability vector")
                    total = float(probability.sum())
                    if not 0.99 - 1e-12 <= total <= 1.01 + 1e-12:
                        raise ValueError("FLARE rounded probabilities lie outside [0.99,1.01]")
                    probability /= total
                    hard = int(values[indexes[hard_name]])
                    if hard not in (0, 1, 2) or probability[hard] < probability.max() - 1e-12:
                        raise ValueError("FLARE hard ancestry is not a probability maximum")
                    matrix[sample_index, hap] = probability
            loci.append(locus)
            rows.append(matrix)
    if samples is None or not rows:
        raise ValueError("FLARE VCF is empty")
    return FlareInput(tuple(loci), samples, np.asarray(rows, dtype=np.float64))


@dataclass(frozen=True)
class TruthSegment:
    start: int
    end: int
    ancestry: str


def load_truth(path: str | Path, samples: Sequence[str], start: int, end: int) -> dict[str, tuple[list[TruthSegment], list[TruthSegment]]]:
    grouped: dict[tuple[str, int], list[TruthSegment]] = {(sample, hap): [] for sample in samples for hap in (0, 1)}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"target_haplotype", "chrom", "start_bp", "end_bp_exclusive", "ancestry"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("truth table has an unexpected header")
        for row in reader:
            token = row["target_haplotype"]
            if "_h" not in token:
                raise ValueError("truth target_haplotype must end in _h0/_h1")
            sample, hap_text = token.rsplit("_h", 1)
            key = (sample, int(hap_text))
            if key not in grouped or _normalize_chrom(row["chrom"]) != "22" or row["ancestry"] not in ANCESTRIES:
                raise ValueError("truth contains an unexpected sample/haplotype/chromosome/ancestry")
            left, right = max(start, int(row["start_bp"])), min(end, int(row["end_bp_exclusive"]))
            if left < right:
                grouped[key].append(TruthSegment(left, right, row["ancestry"]))
    output: dict[str, tuple[list[TruthSegment], list[TruthSegment]]] = {}
    for sample in samples:
        pair: list[list[TruthSegment]] = []
        for hap in (0, 1):
            segments = sorted(grouped[(sample, hap)], key=lambda value: value.start)
            cursor = start
            for segment in segments:
                if segment.start != cursor or segment.end <= segment.start:
                    raise ValueError(f"truth {sample}_h{hap} has a gap/overlap")
                cursor = segment.end
            if cursor != end:
                raise ValueError(f"truth {sample}_h{hap} lacks complete marker-domain coverage")
            pair.append(segments)
        output[sample] = (pair[0], pair[1])
    return output


def truth_at_markers(truth: Mapping[str, tuple[list[TruthSegment], list[TruthSegment]]], samples: Sequence[str], positions: Sequence[int]) -> np.ndarray:
    result = np.zeros((len(positions), len(samples), 2, 3), dtype=np.float64)
    for sample_index, sample in enumerate(samples):
        for hap in (0, 1):
            segment_index = 0
            segments = truth[sample][hap]
            for marker_index, position in enumerate(positions):
                while segments[segment_index].end <= position:
                    segment_index += 1
                segment = segments[segment_index]
                if not segment.start <= position < segment.end:
                    raise ValueError("truth does not cover a FLARE marker")
                result[marker_index, sample_index, hap, ANCESTRIES.index(segment.ancestry)] = 1.0
    return result


def validate_phase_binding(
    rare: RareInput,
    flare: FlareInput,
    truth: Mapping[str, tuple[list[TruthSegment], list[TruthSegment]]],
) -> dict[str, str]:
    """Freeze the direct h0/h1 lineage shared by TARGET, FLARE and truth."""
    if rare.samples != flare.samples or tuple(truth) != flare.samples:
        raise ValueError("TARGET/FLARE/truth sample identity or order differs")
    if rare.hap_presence.shape[1:] != (len(flare.samples), 2):
        raise ValueError("TARGET does not expose exactly h0/h1 for every FLARE sample")
    if any(len(truth[sample]) != 2 for sample in flare.samples):
        raise ValueError("truth does not expose exactly h0/h1 for every FLARE sample")
    return {
        "target_h0": "truth_h0",
        "target_h1": "truth_h1",
        "FLARE_ANP1_AN1": "truth_h0",
        "FLARE_ANP2_AN2": "truth_h1",
        "post_truth_haplotype_swap": "forbidden",
    }


def load_ref_lai_people(path: str | Path) -> tuple[list[tuple[str, tuple[int, int]]], list[str]]:
    grouped: dict[str, list[int]] = {}
    labels: dict[str, str] = {}
    seen_nodes: set[int] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"role", "ancestry", "individual_id", "node_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("pool manifest has an unexpected header")
        for row in reader:
            node = int(row["node_id"])
            if node in seen_nodes:
                raise ValueError("pool manifest duplicates a node")
            seen_nodes.add(node)
            if row["role"] != "REF_LAI":
                continue
            ancestry, individual = row["ancestry"], row["individual_id"]
            if ancestry not in ANCESTRIES or not individual:
                raise ValueError("REF_LAI has an invalid ancestry/individual")
            if individual in labels and labels[individual] != ancestry:
                raise ValueError("a REF_LAI individual crosses ancestries")
            labels[individual] = ancestry
            grouped.setdefault(individual, []).append(node)
    people: list[tuple[str, tuple[int, int]]] = []
    ancestries: list[str] = []
    for individual in sorted(grouped):
        nodes = tuple(sorted(grouped[individual]))
        if len(nodes) != 2:
            raise ValueError("each REF_LAI individual must have two homologues")
        people.append((individual, (nodes[0], nodes[1])))
        ancestries.append(labels[individual])
    if not people:
        raise ValueError("pool manifest has no REF_LAI individuals")
    return people, ancestries


def load_ref_minor_dosage(
    tree_path: str | Path,
    pools_path: str | Path,
    rare: RareInput,
    genetic_map: GeneticMap,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Return site x REF_LAI diploid minor dosage and immutable individual labels."""
    import tskit

    people, labels = load_ref_lai_people(pools_path)
    ts = tskit.load(str(tree_path))
    sample_index = {int(node): index for index, node in enumerate(ts.samples())}
    try:
        indexes = np.asarray([sample_index[node] for _person, pair in people for node in pair], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"REF_LAI node is not a tree-sequence sample: {exc.args[0]}") from exc
    tree_positions = np.asarray([int(variant.site.position) for variant in ts.variants()], dtype=np.int64)
    candidates = sorted({0, int(genetic_map.positions[0]), int(rare.positions[0] - tree_positions[0])})
    selected_set = set(rare.positions.tolist())
    offsets = [offset for offset in candidates if selected_set.issubset(set((tree_positions + offset).tolist()))]
    if len(offsets) != 1:
        raise ValueError(f"cannot identify one exact tree-to-physical offset; candidates={offsets}")
    offset = offsets[0]
    by_position = {int(position): index for index, position in enumerate(rare.positions)}
    dosage = np.empty((len(rare.positions), len(people)), dtype=np.int8)
    seen: set[int] = set()
    for variant in ts.variants():
        absolute = offset + int(variant.site.position)
        rare_index = by_position.get(absolute)
        if rare_index is None:
            continue
        if len(variant.alleles) != 2:
            raise ValueError("a selected M31 site is not biallelic in the tree")
        states = np.asarray(variant.genotypes, dtype=np.int8)[indexes]
        if np.any((states != 0) & (states != 1)):
            raise ValueError("REF_LAI tree genotypes are non-binary/missing")
        dosage[rare_index] = (states == rare.minor_codes[rare_index]).reshape(len(people), 2).sum(axis=1)
        seen.add(absolute)
    if seen != selected_set:
        raise ValueError("tree does not contain every exact selected rare locus")
    return dosage, tuple(individual for individual, _pair in people), tuple(labels)


def stable_sham_seed(root_seed: int, replicate: int) -> int:
    if not 0 <= replicate < SHAM_REPLICATES:
        raise ValueError("M31 sham replicate must lie in [0,32)")
    return int.from_bytes(hashlib.sha256(f"M31|{root_seed}|{replicate}".encode()).digest()[:8], "big")


def permute_diploid_labels(labels: Sequence[str], root_seed: int, replicate: int) -> tuple[str, ...]:
    values = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(stable_sham_seed(root_seed, replicate))
    result = tuple(values[rng.permutation(values.size)].tolist())
    if Counter(result) != Counter(labels):
        raise AssertionError("M31 sham changed REF_LAI ancestry sample sizes")
    return result


def ancestry_support(ref_dosage: np.ndarray, labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    dosage = np.asarray(ref_dosage, dtype=np.float64)
    label_array = np.asarray(labels, dtype=object)
    if dosage.ndim != 2 or dosage.shape[1] != label_array.size or set(label_array.tolist()) - set(ANCESTRIES):
        raise ValueError("REF_LAI dosage/label dimensions are inconsistent")
    panel_sizes = np.asarray([(label_array == ancestry).sum() for ancestry in ANCESTRIES], dtype=np.float64)
    if np.any(panel_sizes == 0):
        raise ValueError("REF_LAI lacks at least one frozen ancestry")
    # Use within-ancestry allele frequency before cross-ancestry
    # normalization, so unequal panel sizes cannot manufacture support.
    frequencies = np.column_stack([
        dosage[:, label_array == ancestry].sum(axis=1) / (2.0 * panel_sizes[index])
        for index, ancestry in enumerate(ANCESTRIES)
    ])
    totals = frequencies.sum(axis=1)
    support = np.divide(frequencies, totals[:, None], out=np.zeros_like(frequencies), where=totals[:, None] > 0)
    no_support = totals == 0
    if not np.allclose(support.sum(axis=1), (~no_support).astype(float), atol=1e-12, rtol=0.0):
        raise AssertionError("REF_LAI support does not preserve the supported/no-support partition")
    return support, no_support


def _flatten_ring_features(prefix: str, aggregation: RingAggregation, channel_names: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Flatten both frozen normalizations and explicit geometry masks."""
    if aggregation.sums.ndim != 4 or aggregation.sums.shape[-1] != len(channel_names):
        raise ValueError("ring channel names do not match aggregation")
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for normalization, array in (("per_cM", aggregation.by_genetic_length), ("per_observed_site", aggregation.by_observed_site_count)):
        blocks.append(array.reshape(array.shape[0], -1))
        names.extend(
            f"{prefix}.{normalization}.{side}.r{ring_index}.{channel}"
            for side in SIDES for ring_index in range(len(aggregation.rings_cm)) for channel in channel_names
        )
    blocks.append(aggregation.edge_mask.reshape(aggregation.edge_mask.shape[0], -1).astype(np.float64))
    names.extend(f"{prefix}.edge.{side}.r{ring_index}" for side in SIDES for ring_index in range(len(aggregation.rings_cm)))
    return np.column_stack(blocks), tuple(names)


@dataclass(frozen=True)
class SampleFeatures:
    baseline: np.ndarray  # marker x hap x ancestry
    truth: np.ndarray
    arms: Mapping[str, np.ndarray]  # each marker x hap x feature
    feature_names: Mapping[str, tuple[str, ...]]


def materialize_sample_features(
    marker_cm: np.ndarray,
    rare_cm: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    target_hap_presence: np.ndarray,
    support: np.ndarray,
    no_support: np.ndarray,
    rings_cm: Sequence[Sequence[float]] = EXPECTED_RINGS,
    *,
    requested_arms: Sequence[str] | None = None,
) -> SampleFeatures:
    """Materialize exact F0/C/L/D/H features for one diploid target.

    C describes the same-haplotype FLARE posterior in signed common-marker
    rings. The exact anchor is deliberately present twice: explicitly as the
    central posterior and in the right innermost half-open context. This frozen
    redundancy lets ridge decide whether the local level or its context is
    useful without changing ring membership. L adds diploid rare load without reference labels. D adds diploid
    target dose apportioned by REF_LAI support, and H adds target-haplotype
    presence to D (simulation ceiling). Every rare channel exposes
    both frozen normalizations and the same left/right edge masks.
    """
    requested = tuple(requested_arms) if requested_arms is not None else ("C", "L", "D", "H")
    if not requested or len(set(requested)) != len(requested) or set(requested) - {"C", "L", "D", "H"}:
        raise ValueError("requested_arms must be a non-empty unique subset of C/L/D/H")
    baseline = np.asarray(baseline, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    target = np.asarray(target_hap_presence, dtype=np.float64)
    if baseline.shape != truth.shape or baseline.ndim != 3 or baseline.shape[1:] != (2, 3):
        raise ValueError("sample FLARE/truth arrays must be marker x 2 x 3")
    if target.shape != (len(rare_cm), 2) or support.shape != (len(rare_cm), 3) or no_support.shape != (len(rare_cm),):
        raise ValueError("sample rare/support arrays have inconsistent dimensions")
    domain = (float(marker_cm[0]), float(marker_cm[-1]))
    rare_in_domain = (rare_cm >= domain[0]) & (rare_cm <= domain[1])
    rare_cm = rare_cm[rare_in_domain]
    target = target[rare_in_domain]
    support = support[rare_in_domain]
    no_support = no_support[rare_in_domain]
    if rare_cm.size == 0:
        raise ValueError("no rare site lies inside the exact FLARE marker domain")
    common_arms: list[np.ndarray] = []
    common_names: tuple[str, ...] | None = None
    for hap in (0, 1):
        aggregate = aggregate_signed_rings(marker_cm, marker_cm, baseline[:, hap, :], rings_cm, domain_cm=domain)
        features, names = _flatten_ring_features("common", aggregate, ANCESTRIES)
        common_arms.append(np.column_stack([baseline[:, hap, :], features]))
        common_names = tuple(f"center.{ancestry}" for ancestry in ANCESTRIES) + names
    common = np.stack(common_arms, axis=1)

    if requested == ("C",):
        return SampleFeatures(baseline, truth, {"C": common}, {"C": common_names or ()})

    finite = np.isfinite(target)
    diploid_callable = finite.all(axis=1)
    diploid_load = np.where(diploid_callable, target.sum(axis=1) / 2.0, np.nan)
    missing = (~diploid_callable).astype(float)
    load_values = np.column_stack([diploid_load, missing])
    load_observed = np.column_stack([diploid_callable, np.ones(len(rare_cm), dtype=bool)])
    load_agg = aggregate_signed_rings(marker_cm, rare_cm, load_values, rings_cm, observed=load_observed, domain_cm=domain)
    load, load_names = _flatten_ring_features("rare_load", load_agg, ("diploid_minor_fraction", "missing_diploid"))
    load_for_haps = np.repeat(load[:, None, :], 2, axis=1)

    arms = {"C": common, "L": np.concatenate([common, load_for_haps], axis=2)}
    names_by_arm = {"C": common_names or (), "L": (common_names or ()) + load_names}
    if "D" in requested or "H" in requested:
        apportioned = np.column_stack([diploid_load[:, None] * support, diploid_load * no_support])
        apportioned_observed = np.repeat(diploid_callable[:, None], 4, axis=1)
        d_agg = aggregate_signed_rings(marker_cm, rare_cm, apportioned, rings_cm, observed=apportioned_observed, domain_cm=domain)
        d_features, d_names = _flatten_ring_features("diploid_support", d_agg, (*ANCESTRIES, "NO_REF_LAI_SUPPORT"))
        d_arm = np.concatenate([common, load_for_haps, np.repeat(d_features[:, None, :], 2, axis=1)], axis=2)
        d_arm_names = (common_names or ()) + load_names + d_names
        if "D" in requested:
            arms["D"] = d_arm
            names_by_arm["D"] = d_arm_names
    if "H" in requested:
        haps: list[np.ndarray] = []
        h_names: tuple[str, ...] | None = None
        for hap in (0, 1):
            callable_hap = finite[:, hap]
            phase_values = np.column_stack([target[:, hap, None] * support, target[:, hap] * no_support])
            phase_observed = np.repeat(callable_hap[:, None], 4, axis=1)
            h_agg = aggregate_signed_rings(marker_cm, rare_cm, phase_values, rings_cm, observed=phase_observed, domain_cm=domain)
            phase, h_names = _flatten_ring_features("haplotype_support", h_agg, (*ANCESTRIES, "NO_REF_LAI_SUPPORT"))
            haps.append(phase)
        arms["H"] = np.concatenate([d_arm, np.stack(haps, axis=1)], axis=2)
        names_by_arm["H"] = d_arm_names + (h_names or ())
    arms = {arm: arms[arm] for arm in requested}
    names_by_arm = {arm: names_by_arm[arm] for arm in requested}
    for arm, matrix in arms.items():
        if matrix.shape != (len(marker_cm), 2, len(names_by_arm[arm])) or not np.all(np.isfinite(matrix)):
            raise AssertionError(f"{arm} feature materialization is non-finite or dimensionally inconsistent")
    return SampleFeatures(baseline, truth, arms, names_by_arm)


def authenticate_frozen_run_inputs(genetic_map: str | Path, roots: Mapping[str, Mapping[str, str | Path]]) -> dict[str, str]:
    """Fail closed unless every real M31 input has the frozen exact digest."""
    observed = {"genetic_map": sha256_file(genetic_map)}
    for root in ROOTS:
        if root not in roots:
            raise ContractError(f"missing frozen input bundle {root}")
        for key in ("sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit"):
            if key not in roots[root]:
                raise ContractError(f"missing frozen input {root}.{key}")
            observed[f"{root}.{key}"] = sha256_file(roots[root][key])
    if observed != EXPECTED_RUN_INPUT_SHA256:
        mismatches = {
            key: {"expected": EXPECTED_RUN_INPUT_SHA256.get(key), "observed": value}
            for key, value in observed.items() if EXPECTED_RUN_INPUT_SHA256.get(key) != value
        }
        raise ContractError(f"M31 frozen input SHA-256 mismatch: {mismatches}")
    return observed


def validate_flare_audit(path: str | Path, root: str, flare_path: str | Path) -> dict[str, Any]:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_markers = 79791
    _require(audit.get("experiment_id") == "M30_FLARE_BASELINE", f"{root} FLARE audit experiment mismatch")
    _require(audit.get("stage") == "M30_FLARE_INFERENCE_AUDIT" and audit.get("status") == "PASS", f"{root} FLARE audit did not pass")
    _require(audit.get("root_label") == root, f"{root} FLARE audit root mismatch")
    _require(audit.get("truth_accessed") is False and audit.get("target_truth_accuracy_computed") is False, f"{root} FLARE inference was not truth-blind")
    _require(audit.get("model_audit", {}).get("ancestry_order") == list(ANCESTRIES), f"{root} FLARE ancestry order mismatch")
    _require(audit.get("ancestry_vcf_audit", {}).get("markers") == expected_markers, f"{root} FLARE marker count mismatch")
    _require(audit.get("output_sha256", {}).get("ancestry_vcf") == sha256_file(flare_path), f"{root} FLARE VCF is not bound by its audit")
    return audit


@dataclass(frozen=True)
class SelectedRidge:
    alpha: float
    boundary_weight: float
    cv_boundary_f1_0_2cm: float
    cv_false_transitions_per_cm_0_2cm: float
    cv_macro_ancestry_dose_mae: float
    cv_brier: float
    guarded: bool
    selection_status: str
    model: WeightedStandardizedRidgeResidual
    fold_samples: tuple[tuple[str, ...], ...]


def select_grouped_ridge_corrector(
    features: Any,
    truth_probabilities: Any,
    baseline_probabilities: Any,
    sample_ids: Sequence[str],
    boundary_rows: Any,
    *,
    marker_cm: Any,
    marker_weights_cm: Any,
    alphas: Sequence[float] = EXPECTED_ALPHAS,
    boundary_weights: Sequence[float] = EXPECTED_BOUNDARY_WEIGHTS,
    seed: int = 31,
) -> SelectedRidge:
    """Select alpha and boundary weight strictly inside grouped 3-fold CV."""
    x = np.asarray(features, dtype=np.float64)
    truth = np.asarray(truth_probabilities, dtype=np.float64)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    boundary = np.asarray(boundary_rows)
    samples = _sample_vector(sample_ids, expected_length=x.shape[0])
    if truth.shape != baseline.shape or truth.shape != (x.shape[0], 3):
        raise ValueError("CV truth/baseline rows must match features and have three ancestries")
    if boundary.dtype != np.bool_ or boundary.shape != (x.shape[0],):
        raise ValueError("boundary_rows must be a boolean vector matching feature rows")
    marker_coordinates = np.asarray(marker_cm, dtype=np.float64)
    marker_weights = np.asarray(marker_weights_cm, dtype=np.float64)
    if marker_coordinates.ndim != 1 or marker_weights.shape != marker_coordinates.shape or np.any(marker_weights < 0.0):
        raise ValueError("marker_cm/marker_weights_cm are inconsistent")
    unique_samples = sorted(set(samples.tolist()))
    rows_per_sample = 2 * len(marker_coordinates)
    if len(x) != len(unique_samples) * rows_per_sample:
        raise ValueError("grouped ridge rows must be sample-major marker x haplotype")
    base_weights = np.tile(np.repeat(marker_weights, 2), len(unique_samples))
    folds = grouped_three_fold_split(samples.tolist(), seed=seed)
    baseline_summary, _baseline_rows = evaluate_haplotype_predictions(
        baseline, truth, marker_coordinates, samples.tolist(), marker_weights_cm=marker_weights,
    )
    candidates: list[tuple[tuple[float, ...], float, float, dict[str, float]]] = []
    guarded_candidates: list[tuple[tuple[float, ...], float, float, dict[str, float]]] = []
    for boundary_weight in boundary_weights:
        raw_weights = base_weights * np.where(boundary, float(boundary_weight), 1.0)
        for alpha in alphas:
            oof_predicted: list[np.ndarray] = []
            oof_truth: list[np.ndarray] = []
            oof_samples: list[str] = []
            for fold in folds:
                model = fit_ridge_corrector(
                    x[fold.train_indices], truth[fold.train_indices], baseline[fold.train_indices],
                    samples[fold.train_indices].tolist(), weights=raw_weights[fold.train_indices], alpha=float(alpha),
                )
                predicted = model.predict(x[fold.validation_indices], baseline[fold.validation_indices])
                oof_predicted.append(predicted)
                oof_truth.append(truth[fold.validation_indices])
                oof_samples.extend(samples[fold.validation_indices].tolist())
            metrics, _rows = evaluate_haplotype_predictions(
                np.vstack(oof_predicted), np.vstack(oof_truth), marker_coordinates,
                oof_samples, marker_weights_cm=marker_weights,
            )
            selection_key = (
                -metrics["boundary_f1_0.2cM"],
                metrics["false_transitions_per_cM_0.2cM"],
                metrics["macro_ancestry_dose_mae"],
                metrics["haplotype_brier"],
                float(boundary_weight),
                -float(alpha),
            )
            candidate = (selection_key, float(alpha), float(boundary_weight), metrics)
            candidates.append(candidate)
            if (
                metrics["macro_ancestry_dose_mae"] <= baseline_summary["macro_ancestry_dose_mae"] + 1e-15
                and metrics["false_transitions_per_cM_0.2cM"]
                <= baseline_summary["false_transitions_per_cM_0.2cM"] + 1e-15
            ):
                guarded_candidates.append(candidate)
    guarded = bool(guarded_candidates)
    _key, alpha, boundary_weight, cv_metrics = min(
        guarded_candidates if guarded else candidates, key=lambda value: value[0],
    )
    final_weights = base_weights * np.where(boundary, boundary_weight, 1.0)
    model = fit_ridge_corrector(x, truth, baseline, samples.tolist(), weights=final_weights, alpha=alpha)
    return SelectedRidge(alpha, boundary_weight, cv_metrics["boundary_f1_0.2cM"],
                         cv_metrics["false_transitions_per_cM_0.2cM"],
                         cv_metrics["macro_ancestry_dose_mae"], cv_metrics["haplotype_brier"],
                         guarded, "GUARDED_CONFIG" if guarded else "NO_GUARDED_CONFIG",
                         model, tuple(fold.validation_samples for fold in folds))


@dataclass(frozen=True)
class Boundary:
    cm: float
    before: int
    after: int


def _hard_boundaries(probabilities: np.ndarray, marker_cm: np.ndarray) -> list[Boundary]:
    labels = np.argmax(probabilities, axis=1)
    return [
        Boundary(float(marker_cm[index]), int(labels[index - 1]), int(labels[index]))
        for index in range(1, len(labels)) if labels[index] != labels[index - 1]
    ]


def _better_boundary_match(
    left: tuple[int, float, tuple[tuple[int, int], ...]],
    right: tuple[int, float, tuple[tuple[int, int], ...]],
) -> tuple[int, float, tuple[tuple[int, int], ...]]:
    """Cardinality first, then total distance, then deterministic path."""
    left_key, right_key = (left[0], -left[1]), (right[0], -right[1])
    if left_key != right_key:
        return left if left_key > right_key else right
    return left if left[2] <= right[2] else right


def ordered_boundary_pairs(
    truth: Sequence[Boundary], prediction: Sequence[Boundary], tolerance_cm: float,
) -> list[tuple[int, int, float]]:
    """Exact ordered maximum-cardinality/minimum-distance label-aware match.

    This ports the frozen M28 dynamic program. Haplotype identity is direct:
    ANP1/AN1 is compared only with truth h0 and ANP2/AN2 only with truth h1.
    No post-truth direct/swap optimization is permitted for phase-aware M31 H.
    """
    if tolerance_cm < 0 or not math.isfinite(tolerance_cm):
        raise ValueError("boundary tolerance must be finite and nonnegative")
    if any(right.cm < left.cm for left, right in zip(truth, truth[1:])):
        raise ValueError("truth boundaries are not ordered")
    if any(right.cm < left.cm for left, right in zip(prediction, prediction[1:])):
        raise ValueError("prediction boundaries are not ordered")
    n, m = len(truth), len(prediction)
    dp: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]] = [
        [(0, 0.0, tuple()) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = _better_boundary_match(dp[i - 1][j], dp[i][j - 1])
            truth_boundary, predicted_boundary = truth[i - 1], prediction[j - 1]
            distance = abs(truth_boundary.cm - predicted_boundary.cm)
            if (
                truth_boundary.before == predicted_boundary.before
                and truth_boundary.after == predicted_boundary.after
                and distance <= tolerance_cm + 1e-12
            ):
                previous = dp[i - 1][j - 1]
                candidate = (previous[0] + 1, previous[1] + distance, previous[2] + ((i - 1, j - 1),))
                best = _better_boundary_match(best, candidate)
            dp[i][j] = best
    return [
        (truth_index, prediction_index, abs(truth[truth_index].cm - prediction[prediction_index].cm))
        for truth_index, prediction_index in dp[n][m][2]
    ]


def _marker_voronoi_weights(marker_cm: np.ndarray) -> np.ndarray:
    if marker_cm.ndim != 1 or marker_cm.size < 2 or np.any(np.diff(marker_cm) <= 0):
        raise ValueError("marker cM coordinates must be strictly increasing")
    edges = np.r_[marker_cm[0], (marker_cm[:-1] + marker_cm[1:]) / 2.0, marker_cm[-1]]
    weights = np.diff(edges)
    if np.any(weights < 0) or not np.isclose(weights.sum(), marker_cm[-1] - marker_cm[0]):
        raise AssertionError("marker Voronoi weights do not reconstruct the cM span")
    return weights


def _diploid_macro_f1(predicted: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    predicted_states = [tuple(sorted((int(a), int(b)))) for a, b in np.argmax(predicted, axis=2)]
    truth_states = [tuple(sorted((int(a), int(b)))) for a, b in np.argmax(truth, axis=2)]
    f1: list[float] = []
    for state in DIPLOID_CLASSES:
        pred_mask = np.asarray([value == state for value in predicted_states])
        truth_mask = np.asarray([value == state for value in truth_states])
        tp = float(weights[pred_mask & truth_mask].sum())
        fp = float(weights[pred_mask & ~truth_mask].sum())
        fn = float(weights[~pred_mask & truth_mask].sum())
        f1.append(2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 0.0)
    return float(np.mean(f1))


def evaluate_haplotype_predictions(
    predicted: np.ndarray,
    truth: np.ndarray,
    marker_cm: np.ndarray,
    sample_ids: Sequence[str],
    *,
    marker_weights_cm: Any | None = None,
    tolerances_cm: Sequence[float] = BOUNDARY_TOLERANCES_CM,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute all preregistered metrics at the diploid-individual unit.

    Row layout is sample-major, then marker, then h0/h1. Phase binding is
    direct and immutable; this scorer never selects a direct/swap permutation
    after seeing truth.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    samples = _sample_vector(sample_ids, expected_length=predicted.shape[0])
    if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError("prediction/truth matrices must be matching row x ancestry arrays")
    if not np.allclose(predicted.sum(axis=1), 1.0) or not np.allclose(truth.sum(axis=1), 1.0):
        raise ValueError("prediction/truth rows must lie on the simplex")
    unique = sorted(set(samples.tolist()))
    marker_count = len(marker_cm)
    marker_cm = np.asarray(marker_cm, dtype=np.float64)
    tolerances = tuple(float(value) for value in tolerances_cm)
    if tolerances != BOUNDARY_TOLERANCES_CM:
        raise ValueError("M31 boundary tolerances must remain exactly 0.1/0.2/0.5 cM")
    weights = _marker_voronoi_weights(marker_cm) if marker_weights_cm is None else np.asarray(marker_weights_cm, dtype=np.float64)
    if weights.shape != (marker_count,) or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("marker_weights_cm must be finite, nonnegative and match marker count")
    span_cm = float(weights.sum())
    if predicted.shape[0] != len(unique) * 2 * marker_count:
        raise ValueError("metric rows must contain every marker/haplotype for every individual")
    individual: list[dict[str, Any]] = []
    global_dose_numerator = np.zeros(3, dtype=np.float64)
    global_brier_numerator = 0.0
    global_confusion = np.zeros((len(DIPLOID_CLASSES), len(DIPLOID_CLASSES)), dtype=np.float64)
    diploid_index = {state: index for index, state in enumerate(DIPLOID_CLASSES)}
    global_boundaries: dict[float, list[Any]] = {
        tolerance: [0, 0, 0, []] for tolerance in tolerances
    }
    for sample in unique:
        indexes = np.flatnonzero(samples == sample)
        if indexes.size != 2 * marker_count:
            raise ValueError("each individual must have exactly marker_count x two rows")
        sample_pred = predicted[indexes].reshape(marker_count, 2, 3)
        sample_truth = truth[indexes].reshape(marker_count, 2, 3)
        truth_boundaries = [_hard_boundaries(sample_truth[:, hap], marker_cm) for hap in (0, 1)]
        pred_boundaries = [_hard_boundaries(sample_pred[:, hap], marker_cm) for hap in (0, 1)]
        predicted_dose = sample_pred.sum(axis=1)
        truth_dose = sample_truth.sum(axis=1)
        dose_error = np.abs(predicted_dose - truth_dose) / 2.0
        global_dose_numerator += (weights[:, None] * dose_error).sum(axis=0)
        ancestry_mae = {
            ancestry: float(np.average(dose_error[:, index], weights=weights))
            for index, ancestry in enumerate(ANCESTRIES)
        }
        haplotype_brier_by_marker = np.square(sample_pred - sample_truth).sum(axis=2).mean(axis=1) / 2.0
        global_brier_numerator += float(np.dot(weights, haplotype_brier_by_marker))
        predicted_states = [tuple(sorted(pair)) for pair in np.argmax(sample_pred, axis=2).astype(int).tolist()]
        truth_states = [tuple(sorted(pair)) for pair in np.argmax(sample_truth, axis=2).astype(int).tolist()]
        for marker_index, (observed, predicted_state) in enumerate(zip(truth_states, predicted_states)):
            global_confusion[diploid_index[observed], diploid_index[predicted_state]] += weights[marker_index]
        row: dict[str, Any] = {
            "sample_id": sample,
            "macro_ancestry_dose_mae": float(np.mean(list(ancestry_mae.values()))),
            **{f"ancestry_dose_mae_{ancestry}": value for ancestry, value in ancestry_mae.items()},
            "haplotype_brier": float(np.average(haplotype_brier_by_marker, weights=weights)),
            "diploid_macro_f1_fixed_six": _diploid_macro_f1(sample_pred, sample_truth, weights),
        }
        for tolerance in tolerances:
            pairs = [
                pair for hap in (0, 1)
                for pair in ordered_boundary_pairs(truth_boundaries[hap], pred_boundaries[hap], tolerance)
            ]
            distances = [pair[2] for pair in pairs]
            n_truth, n_pred, matched = sum(map(len, truth_boundaries)), sum(map(len, pred_boundaries)), len(pairs)
            accumulator = global_boundaries[tolerance]
            accumulator[0] += n_truth
            accumulator[1] += n_pred
            accumulator[2] += matched
            accumulator[3].extend(distances)
            precision = matched / n_pred if n_pred else (1.0 if n_truth == 0 else 0.0)
            recall = matched / n_truth if n_truth else (1.0 if n_pred == 0 else 0.0)
            suffix = f"{tolerance:.1f}cM"
            row[f"boundary_f1_{suffix}"] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            row[f"false_transitions_per_cM_{suffix}"] = (n_pred - matched) / (2.0 * span_cm)
            row[f"matched_boundary_median_{suffix}"] = float(np.median(distances)) if distances else float("nan")
            row[f"matched_boundary_p90_{suffix}"] = float(np.quantile(distances, 0.9)) if distances else float("nan")
        individual.append(row)
    total_span_cm = len(unique) * span_cm
    global_dose = global_dose_numerator / total_span_cm
    summary: dict[str, Any] = {
        "macro_ancestry_dose_mae": float(global_dose.mean()),
        **{f"ancestry_dose_mae_{ancestry}": float(global_dose[index]) for index, ancestry in enumerate(ANCESTRIES)},
        "haplotype_brier": global_brier_numerator / total_span_cm,
        "diploid_macro_f1_fixed_six": float(np.mean([
            (lambda tp, fp, fn: 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 0.0)(
                global_confusion[index, index],
                global_confusion[:, index].sum() - global_confusion[index, index],
                global_confusion[index, :].sum() - global_confusion[index, index],
            )
            for index in range(len(DIPLOID_CLASSES))
        ])),
    }
    for tolerance in tolerances:
        n_truth, n_pred, matched, distances = global_boundaries[tolerance]
        precision = matched / n_pred if n_pred else (1.0 if n_truth == 0 else 0.0)
        recall = matched / n_truth if n_truth else (1.0 if n_pred == 0 else 0.0)
        suffix = f"{tolerance:.1f}cM"
        summary[f"boundary_f1_{suffix}"] = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[f"false_transitions_per_cM_{suffix}"] = (n_pred - matched) / (2.0 * total_span_cm)
        summary[f"matched_boundary_median_{suffix}"] = float(np.median(distances)) if distances else float("nan")
        summary[f"matched_boundary_p90_{suffix}"] = float(np.quantile(distances, 0.9)) if distances else float("nan")
    # Stable aliases used by the earlier numerical known-answer contract.
    summary["macro_ancestry_mae"] = summary["macro_ancestry_dose_mae"]
    summary["boundary_f1_0.2cM"] = summary["boundary_f1_0.2cM"]
    summary["false_transitions_per_cM"] = summary["false_transitions_per_cM_0.2cM"]
    summary["matched_boundary_median_cM"] = summary["matched_boundary_median_0.2cM"]
    summary["matched_boundary_p90_cM"] = summary["matched_boundary_p90_0.2cM"]
    return summary, individual


def bootstrap_individual_metrics(
    individual_rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap over complete diploid individuals."""
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("M31 bootstrap must use exactly 10000 replicates")
    if not individual_rows or len({row["sample_id"] for row in individual_rows}) != len(individual_rows):
        raise ValueError("bootstrap requires one unique row per diploid individual")
    metric_keys = [key for key in individual_rows[0] if key != "sample_id"]
    matrix = np.asarray([[float(row[key]) for key in metric_keys] for row in individual_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(metric_keys)), dtype=np.float64)
    for replicate in range(replicates):
        selected = matrix[rng.integers(0, len(matrix), size=len(matrix))]
        for column in range(len(metric_keys)):
            finite = selected[:, column][np.isfinite(selected[:, column])]
            draws[replicate, column] = finite.mean() if finite.size else np.nan
    intervals = {}
    for column, key in enumerate(metric_keys):
        finite = draws[:, column][np.isfinite(draws[:, column])]
        intervals[key] = {
            "lower": float(np.quantile(finite, 0.025)) if finite.size else None,
            "upper": float(np.quantile(finite, 0.975)) if finite.size else None,
        }
    return {
        "unit": "complete_diploid_individual",
        "replicates": replicates,
        "seed": seed,
        "interval": "percentile_95",
        "metrics": intervals,
    }


def run_synthetic_end_to_end() -> dict[str, Any]:
    """Known-answer grouped-CV fit and held-individual scoring without real data."""
    marker_cm = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    marker_truth = np.eye(3)[[0, 0, 1, 1, 2]]

    def rows(samples: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
        features: list[np.ndarray] = []
        truth: list[np.ndarray] = []
        baseline: list[np.ndarray] = []
        ids: list[str] = []
        boundaries: list[bool] = []
        for sample_index, sample in enumerate(samples):
            for marker_index, target in enumerate(marker_truth):
                for hap in (0, 1):
                    # The nuisance coordinate varies by complete individual;
                    # the one-hot signal is stable across the held individuals.
                    features.append(np.r_[target, (sample_index + hap) / 20.0])
                    truth.append(target)
                    baseline.append(np.full(3, 1.0 / 3.0))
                    ids.append(sample)
                    boundaries.append(marker_index in (2, 4))
        return np.asarray(features), np.asarray(truth), np.asarray(baseline), ids, np.asarray(boundaries, dtype=bool)

    train_x, train_truth, train_baseline, train_ids, train_boundary = rows(tuple(f"TRAIN{i}" for i in range(6)))
    selected = select_grouped_ridge_corrector(
        train_x, train_truth, train_baseline, train_ids, train_boundary,
        marker_cm=marker_cm, marker_weights_cm=_marker_voronoi_weights(marker_cm),
        alphas=(0.0001, 0.01, 1.0), boundary_weights=(1.0, 5.0), seed=3101,
    )
    test_x, test_truth, test_baseline, test_ids, _ = rows(("EVAL0", "EVAL1"))
    predicted = selected.model.predict(test_x, test_baseline)
    corrected, individuals = evaluate_haplotype_predictions(predicted, test_truth, marker_cm, test_ids)
    baseline_metrics, _ = evaluate_haplotype_predictions(test_baseline, test_truth, marker_cm, test_ids)
    fold_sets = [set(items) for items in selected.fold_samples]
    if set.union(*fold_sets) != set(train_ids) or any(left & right for index, left in enumerate(fold_sets) for right in fold_sets[index + 1:]):
        raise AssertionError("synthetic grouped CV leaked an individual")
    if not corrected["haplotype_brier"] < baseline_metrics["haplotype_brier"]:
        raise AssertionError("synthetic corrector did not improve the held-individual Brier score")
    if corrected["boundary_f1_0.2cM"] != 1.0:
        raise AssertionError("synthetic corrector did not recover the exact held-individual boundaries")
    return {
        "status": "PASS",
        "train_individuals": 6,
        "evaluation_individuals": 2,
        "selected_alpha": selected.alpha,
        "selected_boundary_weight": selected.boundary_weight,
        "cv_brier": selected.cv_brier,
        "baseline_haplotype_brier": baseline_metrics["haplotype_brier"],
        "corrected_haplotype_brier": corrected["haplotype_brier"],
        "boundary_f1_0.2cM": corrected["boundary_f1_0.2cM"],
        "individual_rows": len(individuals),
        "split_unit": "complete_diploid_individual",
    }


def run_known_answer_selftest() -> dict[str, str]:
    """Run deterministic exact known answers for every numerical primitive."""
    projected = project_simplex(np.array([[0.25, 0.75], [-1.0, 2.0]]))
    expected_projection = np.array([[0.25, 0.75], [0.0, 1.0]])
    if not np.array_equal(projected, expected_projection):
        raise AssertionError(f"simplex known answer failed: {projected!r}")

    aggregation = aggregate_signed_half_open_rings(
        [1.0],
        [0.5, 0.75, 1.0, 1.25, 1.5],
        [1.0, 2.0, 4.0, 8.0, 16.0],
        ((0.0, 0.25), (0.25, 0.5)),
        domain_cm=(0.5, 1.5),
    )
    expected_sums = np.array([[[2.0, 1.0], [4.0, 8.0]]])
    if not np.array_equal(aggregation.sums, expected_sums):
        raise AssertionError(f"ring known answer failed: {aggregation.sums!r}")
    anchor = aggregate_signed_half_open_rings(
        [1.0], [1.0], [7.0], ((0.0, 0.1),), domain_cm=(0.9, 1.1)
    )
    if not np.array_equal(anchor.sums, np.array([[[0.0], [7.0]]])):
        raise AssertionError("an exact anchor site was not assigned to the right innermost ring")
    adversarial_truth = [Boundary(0.10, 0, 1), Boundary(0.20, 0, 1)]
    adversarial_prediction = [Boundary(0.00, 0, 1), Boundary(0.11, 0, 1)]
    adversarial_pairs = ordered_boundary_pairs(adversarial_truth, adversarial_prediction, 0.11)
    if len(adversarial_pairs) != 2 or not math.isclose(sum(pair[2] for pair in adversarial_pairs), 0.19):
        raise AssertionError("exact boundary matching failed the greedy-trap known answer")
    minimum_distance_pairs = ordered_boundary_pairs(
        [Boundary(0.10, 0, 1), Boundary(0.30, 0, 1)],
        [Boundary(0.00, 0, 1), Boundary(0.11, 0, 1), Boundary(0.31, 0, 1)],
        0.25,
    )
    if [(left, right) for left, right, _ in minimum_distance_pairs] != [(0, 1), (1, 2)]:
        raise AssertionError("exact boundary matching did not minimize total distance after cardinality")
    wrong_transition = ordered_boundary_pairs(
        [Boundary(0.1, 0, 1)], [Boundary(0.1, 1, 0)], 0.1
    )
    if wrong_transition:
        raise AssertionError("boundary matching ignored before-to-after labels")

    ids = ["A", "A", "B", "B", "C", "C", "D", "D"]
    fold_ids = grouped_three_fold_ids(ids, seed=31)
    if any(len(set(fold_ids[np.asarray(ids) == sample].tolist())) != 1 for sample in set(ids)):
        raise AssertionError("grouped-fold known answer separated an individual")

    normalized = normalize_weights([1.0, 1.0, 3.0, 3.0], ["A", "A", "B", "B"])
    if float(normalized.sum()) != 2.0:
        raise AssertionError("weight known answer did not sum to two individuals")

    model = fit_weighted_standardized_ridge_residual(
        [[-1.0], [1.0]],
        [[-2.0, 2.0], [2.0, -2.0]],
        ["A", "B"],
        alpha=2.0,
    )
    if not np.array_equal(model.coefficients, np.array([[1.0, -1.0]])):
        raise AssertionError(f"ridge known answer failed: {model.coefficients!r}")
    if not np.array_equal(model.predict_residual([[-1.0], [1.0]]),
                          np.array([[-1.0, 1.0], [1.0, -1.0]])):
        raise AssertionError("ridge prediction known answer failed")
    unequal_dosage = np.array([[2, 0, 1, 0], [0, 0, 0, 0]])
    unequal_labels = ("AFR", "EUR", "EUR", "ASIA")
    support, unsupported = ancestry_support(unequal_dosage, unequal_labels)
    if not np.allclose(support[0], [0.8, 0.2, 0.0]) or not unsupported[1]:
        raise AssertionError("ancestry support was biased by unequal panel sizes")
    sham = permute_diploid_labels(("AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"), 20260817, 0)
    if Counter(sham) != Counter(("AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA")):
        raise AssertionError("diploid sham changed ancestry sample sizes")
    synthetic = run_synthetic_end_to_end()
    if synthetic["status"] != "PASS":
        raise AssertionError("synthetic M31 end-to-end failed")
    return {
        "simplex_projection": "PASS",
        "signed_half_open_rings": "PASS",
        "exact_anchor_is_right": "PASS",
        "exact_label_aware_boundary_matching": "PASS",
        "grouped_three_fold": "PASS",
        "normalized_weights": "PASS",
        "weighted_multivariate_ridge": "PASS",
        "unequal_panel_frequency_support": "PASS",
        "diploid_sham_invariants": "PASS",
        "synthetic_end_to_end": "PASS",
    }


known_answer_selftest = run_known_answer_selftest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, help="M31 ordered-linear preregistration JSON")
    parser.add_argument("--selftest", action="store_true", help="run exact numerical known answers")
    parser.add_argument("--genetic-map", type=Path, help="frozen chr22 genetic map")
    for root in ROOTS:
        for key in ("sites", "target", "tree", "pools", "truth", "flare-vcf", "flare-audit"):
            parser.add_argument(
                f"--{root}-{key}",
                dest=f"{root}_{key.replace('-', '_')}",
                type=Path,
                help=f"frozen {root} {key.replace('-', ' ')} input",
            )
    parser.add_argument("--output", type=Path, help="write the JSON report here instead of stdout")
    return parser


def _frozen_inputs_from_args(args: argparse.Namespace) -> tuple[Path, dict[str, dict[str, Path]]] | None:
    """Return a complete frozen-input bundle, rejecting partial authentication."""
    values: list[Path | None] = [args.genetic_map]
    roots: dict[str, dict[str, Path]] = {}
    for root in ROOTS:
        roots[root] = {}
        for key in ("sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit"):
            value = getattr(args, f"{root}_{key}")
            values.append(value)
            if value is not None:
                roots[root][key] = value
    if not any(value is not None for value in values):
        return None
    if any(value is None for value in values):
        raise ContractError("frozen-input authentication requires the map and all seven files for both roots")
    assert args.genetic_map is not None
    return args.genetic_map, roots


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    frozen = _frozen_inputs_from_args(args)
    if args.contract is None and not args.selftest and frozen is None:
        parser.error("at least one of --contract, --selftest or a complete frozen-input bundle is required")
    report: dict[str, Any] = {"stage": EXPERIMENT_ID}
    if args.contract is not None:
        parsed = load_contract(args.contract)
        report["contract"] = {
            "status": "PASS",
            "sha256": sha256_file(args.contract),
            "directions": [item.name for item in parsed.directions],
            "rings_cm": [list(pair) for pair in parsed.rings_cm],
            "alphas": list(parsed.alphas),
            "boundary_weights": list(parsed.boundary_weights),
        }
    if args.selftest:
        report["known_answers"] = run_known_answer_selftest()
    if frozen is not None:
        genetic_map, roots = frozen
        hashes = authenticate_frozen_run_inputs(genetic_map, roots)
        for root in ROOTS:
            validate_flare_audit(roots[root]["flare_audit"], root, roots[root]["flare_vcf"])
        report["frozen_inputs"] = {
            "status": "PASS",
            "count": len(hashes),
            "sha256": hashes,
            "flare_audits": {root: "PASS" for root in ROOTS},
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
