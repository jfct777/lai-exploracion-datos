#!/usr/bin/env python3
"""One-temperature probability operations for the R02 LAI protocol.

``authenticate_split`` preserves the legacy inner-FIT admission control and
never invents donor partitions.  The weighted beta functions separately train
one scalar parameter on all of FIT_A for later model/temperature selection on
donor-separated SELECT_B; fitting on FIT_A is not independent validation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


BLOCKED = "BLOCKED_NO_DONOR_DISJOINT_INNER_FIT"
SPLIT_SCHEMA = "r02_lai_inner_fit_donor_split_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_probabilities(probabilities: np.ndarray,
                           evaluation_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(probabilities)
    require(values.ndim >= 2 and values.shape[-1] == 6,
            "six-state probability axis required")
    require(np.issubdtype(values.dtype, np.floating),
            "floating probabilities required")
    if evaluation_mask is None:
        active = values.reshape(-1, 6)
    else:
        mask = np.asarray(evaluation_mask)
        require(mask.dtype == bool and mask.shape == values.shape[:-1] and mask.any(),
                "probability evaluation mask axes differ or are empty")
        active = values[mask]
    require(active.shape[0] > 0, "probability array is empty")
    require(np.isfinite(active).all(), "nonfinite probability on evaluated support")
    require((active >= 0).all() and (active <= 1).all(),
            "probability outside [0,1]")
    require(np.max(np.abs(active.astype(np.float64).sum(-1) - 1.0)) <= 1e-6,
            "probabilities do not sum to one")
    return values


def temperature_scale(probabilities: np.ndarray, temperature: float,
                      evaluation_mask: np.ndarray | None = None) -> np.ndarray:
    """Apply q_s(T)=p_s^(1/T)/sum_j p_j^(1/T), preserving exact zeros.

    T=1 deliberately returns a byte-identical copy.  This makes the identity
    control independent of floating-point logarithm/exponentiation details.
    """
    values = validate_probabilities(probabilities, evaluation_mask)
    require(math.isfinite(temperature) and temperature > 0,
            "temperature must be finite and positive")
    if temperature == 1.0:
        return values.copy()
    work = values.astype(np.float64)
    mask = (np.ones(values.shape[:-1], dtype=bool) if evaluation_mask is None
            else np.asarray(evaluation_mask))
    active = work[mask]
    with np.errstate(divide="ignore"):
        logits = np.log(active) / temperature
    maximum = np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits - maximum)
    scaled_active = weights / weights.sum(-1, keepdims=True)
    scaled = work.copy()
    scaled[mask] = scaled_active
    require(np.array_equal(scaled_active == 0, active == 0),
            "temperature changed exact zero support")
    return scaled


def blocked_receipt(reason: str = BLOCKED) -> dict:
    require(reason == BLOCKED, "unrecognized calibration blocker")
    return {
        "schema": "r02_lai_probability_temperature_control_v1",
        "status": BLOCKED,
        "role": "FIT",
        "temperature_parameters_fitted": 0,
        "identity_temperature": 1.0,
        "identity_probability_operation": "EXACT_COPY",
        "zero_probability_support_preserved": True,
        "select_opened": False,
        "evaluate_opened": False,
    }


def authenticate_split(split_path: str | Path, expected_sha256: str,
                       expected_units: set[tuple[str, str]]) -> dict:
    """Authenticate a complete donor map for all scored case/person units."""
    path = Path(split_path)
    require(file_sha256(path) == expected_sha256, "split manifest SHA differs")
    split = json.loads(path.read_text())
    require(split.get("schema") == SPLIT_SCHEMA and split.get("role") == "FIT",
            "split is not an inner-FIT donor split")
    require(split.get("donor_sets_authenticated") is True,
            "donor sets are not authenticated")
    evidence = split.get("source_evidence")
    require(isinstance(evidence, dict) and set(evidence) == {"path", "sha256"}
            and file_sha256(evidence["path"]) == evidence["sha256"],
            "donor evidence is absent or unauthenticated")
    rows = split.get("units")
    require(isinstance(rows, list) and rows, "split has no scored units")
    seen: set[tuple[str, str]] = set()
    donors = {"CALIBRATION": set(), "EVALUATION": set()}
    partitions: dict[tuple[str, str], str] = {}
    for row in rows:
        require(set(row) == {"case_id", "person_id", "partition", "donor_ids"},
                "split unit fields differ")
        unit = (str(row["case_id"]), str(row["person_id"]))
        partition = row["partition"]
        donor_ids = row["donor_ids"]
        require(unit not in seen and partition in donors,
                "duplicate unit or invalid partition")
        require(isinstance(donor_ids, list) and donor_ids
                and len(set(map(str, donor_ids))) == len(donor_ids),
                "unit donor set is empty or duplicated")
        seen.add(unit)
        partitions[unit] = partition
        donors[partition].update(map(str, donor_ids))
    require(seen == expected_units, "split does not cover the exact scored units")
    require(all(donors.values()) and donors["CALIBRATION"].isdisjoint(donors["EVALUATION"]),
            "calibration and evaluation donors overlap")
    require(set(partitions.values()) == set(donors), "both partitions are required")
    return {
        "manifest": split,
        "partitions": partitions,
        "calibration_donors": sorted(donors["CALIBRATION"]),
        "evaluation_donors": sorted(donors["EVALUATION"]),
    }


def mean_log_loss(probabilities: np.ndarray, truth: np.ndarray,
                  observed: np.ndarray) -> float:
    values = np.asarray(probabilities)
    require(truth.shape == observed.shape == values.shape[:-1]
            and observed.dtype == bool and observed.any(),
            "truth/observed axes differ")
    values = validate_probabilities(values, observed).astype(np.float64)
    require(truth.dtype.kind in "iu" and np.isin(truth[observed], range(6)).all(),
            "true state must be an integer in [0,5]")
    safe = np.where(observed, truth, 0).astype(int)
    target = np.take_along_axis(values, safe[..., None], axis=-1)[..., 0]
    if (target[observed] == 0).any():
        return math.inf
    return float((-np.log(target[observed])).mean())


def fit_temperature(probabilities: np.ndarray, truth: np.ndarray,
                    observed: np.ndarray, lower: float = 0.05,
                    upper: float = 20.0, iterations: int = 80) -> float:
    """Deterministic golden-section minimization on log(T), after admission."""
    require(0 < lower < 1 < upper and iterations >= 20,
            "temperature search bounds/iterations differ")
    a, b = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - ratio * (b - a), a + ratio * (b - a)

    def objective(log_temperature: float) -> float:
        return mean_log_loss(
            temperature_scale(probabilities, math.exp(log_temperature), observed),
            truth, observed)

    fc, fd = objective(c), objective(d)
    for _ in range(iterations):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = objective(d)
    return math.exp((a + b) / 2.0)


def _prepare_weighted_beta(probabilities: np.ndarray, truth: np.ndarray,
                           weights: np.ndarray) -> dict:
    values = validate_probabilities(np.asarray(probabilities)).astype(np.float64)
    labels = np.asarray(truth)
    query_weights = np.asarray(weights, dtype=np.float64)
    require(values.ndim == 2 and labels.shape == query_weights.shape == values.shape[:1],
            "weighted temperature axes differ")
    require(labels.dtype.kind in "iu" and np.isin(labels, range(6)).all(),
            "true state must be an integer in [0,5]")
    require(np.isfinite(query_weights).all() and (query_weights > 0).all()
            and abs(float(query_weights.sum()) - 1.0) <= 1e-10,
            "query weights must be positive and sum to one")
    with np.errstate(divide="ignore"):
        logp = np.log(values)
    target_logp = logp[np.arange(len(labels)), labels.astype(int)]
    positive = np.isfinite(logp)
    row_maximum = np.max(logp, axis=1)
    finite_minimum = np.min(np.where(positive, logp, math.inf), axis=1)
    maximum_span = float(np.max(row_maximum - finite_minimum))
    return {"logp": logp, "target_logp": target_logp, "weights": query_weights,
            "positive": positive, "row_maximum": row_maximum,
            "maximum_span": maximum_span,
            "zero_true": ~np.isfinite(target_logp)}


def _weighted_beta_statistics(prepared: dict, beta: float) -> dict:
    require(math.isfinite(beta) and beta >= 0, "beta must be finite and nonnegative")
    logp = prepared["logp"]
    target_logp = prepared["target_logp"]
    query_weights = prepared["weights"]
    positive = prepared["positive"]
    row_maximum = prepared["row_maximum"]
    maximum_span = prepared["maximum_span"]
    zero_true = prepared["zero_true"]
    if zero_true.any():
        return {
            "nll": math.inf,
            "derivative": math.nan,
            "limit_derivative": math.inf,
            "zero_true_probability_weight": float(query_weights[zero_true].sum()),
            "max_log_probability_span": maximum_span,
        }
    if beta == 0:
        logits = np.where(positive, 0.0, -math.inf)
    else:
        logits = beta * logp
    maximum = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - maximum)
    normalizer = exp_shifted.sum(axis=1, keepdims=True)
    q = exp_shifted / normalizer
    log_normalizer = maximum[:, 0] + np.log(normalizer[:, 0])
    nll = log_normalizer - beta * target_logp
    safe_logp = np.where(positive, logp, 0.0)
    expected_logp = (q * safe_logp).sum(axis=1)
    derivative = expected_logp - target_logp
    return {
        "nll": float(np.dot(query_weights, nll)),
        "derivative": float(np.dot(query_weights, derivative)),
        "limit_derivative": float(np.dot(query_weights, row_maximum - target_logp)),
        "zero_true_probability_weight": 0.0,
        "max_log_probability_span": maximum_span,
    }


def weighted_beta_statistics(probabilities: np.ndarray, truth: np.ndarray,
                             weights: np.ndarray, beta: float) -> dict:
    """Weighted NLL and derivative for q_s(beta) proportional to p_s**beta.

    Exact zero probabilities remain outside the support, including at beta=0.
    ``weights`` are the complete query weights after the case/component/person
    reduction; they must be positive and sum to one.
    """
    return _weighted_beta_statistics(
        _prepare_weighted_beta(probabilities, truth, weights), beta)


def _solve_beta_from_statistics(statistics, at_zero: dict,
                                log_probability_tolerance: float,
                                maximum_bisections: int) -> dict:
    """Solve a convex one-dimensional objective from its statistics callback."""
    at_one = statistics(1.0)
    span = at_zero["max_log_probability_span"]
    numeric_tolerance = 128 * np.finfo(np.float64).eps * max(1.0, span)
    finite_or_none = lambda value: float(value) if math.isfinite(value) else None
    diagnostics = {
        "derivative_at_beta_zero": finite_or_none(at_zero["derivative"]),
        "derivative_at_beta_one": finite_or_none(at_one["derivative"]),
        "limit_derivative": finite_or_none(at_zero["limit_derivative"]),
        "numeric_derivative_tolerance": numeric_tolerance,
    }
    if at_zero["zero_true_probability_weight"] > 0:
        return {
            "status": "IMPOSSIBLE_TRUE_PROBABILITY_ZERO",
            "beta": None, "temperature": None, "nll": None,
            **{key: at_zero[key] for key in
               ("zero_true_probability_weight", "max_log_probability_span")},
            "diagnostics": diagnostics,
        }
    if span == 0.0:
        return {
            "status": "ALL_BETA_EQUIVALENT",
            "beta": 0.0, "temperature": None, "nll": at_zero["nll"],
            "zero_true_probability_weight": 0.0,
            "max_log_probability_span": at_zero["max_log_probability_span"],
            "log_probability_error_bound": 0.0, "bisections": 0,
            "diagnostics": diagnostics,
        }
    if abs(at_zero["derivative"]) <= numeric_tolerance:
        return {
            "status": "UNRESOLVED_NUMERIC_DERIVATIVE_AT_BETA_ZERO",
            "beta": None, "temperature": None, "nll": None,
            "zero_true_probability_weight": 0.0,
            "max_log_probability_span": span,
            "log_probability_error_bound": None, "bisections": 0,
            "diagnostics": diagnostics,
        }
    if at_zero["derivative"] > numeric_tolerance:
        return {
            "status": "BETA_ZERO_BOUNDARY", "beta": 0.0,
            "temperature": None, "nll": at_zero["nll"],
            "zero_true_probability_weight": 0.0,
            "max_log_probability_span": at_zero["max_log_probability_span"],
            "log_probability_error_bound": 0.0, "bisections": 0,
            "diagnostics": diagnostics,
        }
    if at_zero["limit_derivative"] == 0.0:
        return {
            "status": "BETA_INFINITY_BOUNDARY", "beta": None,
            "temperature": None, "temperature_limit": 0.0, "nll": None,
            "zero_true_probability_weight": 0.0,
            "max_log_probability_span": at_zero["max_log_probability_span"],
            "log_probability_error_bound": None, "bisections": 0,
            "diagnostics": diagnostics,
        }
    lower, upper = 0.0, 1.0
    upper_stats = statistics(upper)
    expansions = 0
    while upper_stats["derivative"] < 0 and expansions < maximum_bisections:
        lower, upper = upper, upper * 2.0
        upper_stats = statistics(upper)
        expansions += 1
    require(upper_stats["derivative"] >= 0,
            "failed to bracket the finite beta optimum")
    bisections = 0
    while span * (upper - lower) > log_probability_tolerance:
        require(bisections < maximum_bisections, "beta bisection did not converge")
        middle = (lower + upper) / 2.0
        middle_stats = statistics(middle)
        if middle_stats["derivative"] < 0:
            lower = middle
        else:
            upper = middle
        bisections += 1
    beta = (lower + upper) / 2.0
    fitted = statistics(beta)
    return {
        "status": "INTERIOR", "beta": beta, "temperature": 1.0 / beta,
        "nll": fitted["nll"], "zero_true_probability_weight": 0.0,
        "max_log_probability_span": span,
        "log_probability_error_bound": span * (upper - lower) / 2.0,
        "bisections": bisections, "bracket_expansions": expansions,
        "diagnostics": diagnostics,
    }


def fit_weighted_temperature(probabilities: np.ndarray, truth: np.ndarray,
                             weights: np.ndarray,
                             log_probability_tolerance: float = 1e-10,
                             maximum_bisections: int = 128) -> dict:
    """Fit beta=1/T by monotone-derivative bisection, without a grid.

    The stopping rule bounds the change in every fitted log probability by
    ``log_probability_tolerance`` natural-log units. Boundary optima and true
    states assigned exact probability zero are returned explicitly.
    """
    require(math.isfinite(log_probability_tolerance)
            and 0 < log_probability_tolerance <= 1e-4,
            "invalid log-probability tolerance")
    require(isinstance(maximum_bisections, int) and 16 <= maximum_bisections <= 512,
            "invalid bisection limit")
    prepared = _prepare_weighted_beta(probabilities, truth, weights)
    statistics = lambda beta: _weighted_beta_statistics(prepared, beta)
    at_zero = statistics(0.0)
    return _solve_beta_from_statistics(statistics, at_zero,
                                       log_probability_tolerance,
                                       maximum_bisections)


def fit_weighted_log_probabilities(log_probabilities: np.ndarray,
                                   truth: np.ndarray, weights: np.ndarray,
                                   log_probability_tolerance: float = 1e-8,
                                   maximum_bisections: int = 128,
                                   chunk_rows: int = 262_144) -> dict:
    """Fit beta from validated log probabilities using bounded-memory chunks.

    This entry point is intended for an on-disk memmap prepared while the
    authenticated source probabilities are read once.  It never materializes
    another full query-by-state matrix in memory.
    """
    logp = np.asarray(log_probabilities)
    labels = np.asarray(truth)
    query_weights = np.asarray(weights)
    require(logp.ndim == 2 and logp.shape[1] == 6
            and labels.shape == query_weights.shape == logp.shape[:1],
            "chunked weighted temperature axes differ")
    require(np.issubdtype(logp.dtype, np.floating)
            and np.issubdtype(query_weights.dtype, np.floating),
            "chunked log probabilities/weights must be floating")
    require(isinstance(chunk_rows, int) and 1024 <= chunk_rows <= 1_048_576,
            "invalid chunk row count")
    require(math.isfinite(log_probability_tolerance)
            and 0 < log_probability_tolerance <= 1e-4,
            "invalid log-probability tolerance")
    require(isinstance(maximum_bisections, int) and 16 <= maximum_bisections <= 512,
            "invalid bisection limit")
    weight_sum = 0.0
    maximum_span = 0.0
    zero_true_weight = 0.0
    limit_derivative = 0.0
    for start in range(0, len(labels), chunk_rows):
        stop = min(start + chunk_rows, len(labels))
        block = np.asarray(logp[start:stop], dtype=np.float64)
        block_labels = labels[start:stop]
        block_weights = np.asarray(query_weights[start:stop], dtype=np.float64)
        require(block_labels.dtype.kind in "iu" and np.isin(block_labels, range(6)).all()
                and np.isfinite(block_weights).all() and (block_weights > 0).all(),
                "invalid chunked truth or weight")
        require((np.isfinite(block) | np.isneginf(block)).all()
                and (block <= 1e-12).all() and np.isfinite(block).any(1).all(),
                "invalid log probability")
        target = block[np.arange(stop - start), block_labels.astype(int)]
        positive = np.isfinite(block)
        row_maximum = np.max(block, axis=1)
        finite_minimum = np.min(np.where(positive, block, math.inf), axis=1)
        maximum_span = max(maximum_span, float(np.max(row_maximum - finite_minimum)))
        zero = ~np.isfinite(target)
        zero_true_weight += float(block_weights[zero].sum())
        if not zero.any():
            limit_derivative += float(np.dot(block_weights, row_maximum - target))
        weight_sum += float(block_weights.sum())
    require(abs(weight_sum - 1.0) <= 1e-10,
            "chunked query weights must sum to one")

    def statistics(beta: float) -> dict:
        require(math.isfinite(beta) and beta >= 0,
                "beta must be finite and nonnegative")
        if zero_true_weight > 0:
            return {"nll": math.inf, "derivative": math.nan,
                    "limit_derivative": math.inf,
                    "zero_true_probability_weight": zero_true_weight,
                    "max_log_probability_span": maximum_span}
        total_nll = 0.0
        total_derivative = 0.0
        for start in range(0, len(labels), chunk_rows):
            stop = min(start + chunk_rows, len(labels))
            block = np.asarray(logp[start:stop], dtype=np.float64)
            target = block[np.arange(stop - start), labels[start:stop].astype(int)]
            block_weights = np.asarray(query_weights[start:stop], dtype=np.float64)
            positive = np.isfinite(block)
            logits = np.where(positive, 0.0, -math.inf) if beta == 0 else beta * block
            maximum = np.max(logits, axis=1, keepdims=True)
            shifted = np.exp(logits - maximum)
            denominator = shifted.sum(axis=1)
            nll = maximum[:, 0] + np.log(denominator) - beta * target
            safe_logp = np.where(positive, block, 0.0)
            expected = (shifted * safe_logp).sum(axis=1) / denominator
            total_nll += float(np.dot(block_weights, nll))
            total_derivative += float(np.dot(block_weights, expected - target))
        return {"nll": total_nll, "derivative": total_derivative,
                "limit_derivative": limit_derivative,
                "zero_true_probability_weight": 0.0,
                "max_log_probability_span": maximum_span}

    return _solve_beta_from_statistics(statistics, statistics(0.0),
                                       log_probability_tolerance,
                                       maximum_bisections)
