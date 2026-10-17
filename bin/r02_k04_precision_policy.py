"""Fail-closed K04 selection-policy amendments reviewed on 2026-10-11.

Shared policy for the family consumer. Historical frozen executions retain
their original sources; no old FP64 scalar is promoted to an exact certificate.
"""
import math


def inclusive_guard_candidate_ids(summaries, penalties, *, guard_decades=1., spacing_log10=.5):
    """Return every penalty candidate in either inclusive outer guard decade."""
    if not penalties:
        return set()
    if len(summaries) != len(penalties) + 1 or summaries[0].get("id") != "INTERCEPT_ONLY":
        raise ValueError("Intercept plus one aligned summary per penalty required")
    if (len({row.get("id") for row in summaries}) != len(summaries) or
            any(row.get("penalty") != penalty
                for row, penalty in zip(summaries[1:], penalties))):
        raise ValueError("Unique candidate IDs aligned with penalties required")
    if not (math.isfinite(guard_decades) and guard_decades > 0 and
            math.isfinite(spacing_log10) and spacing_log10 > 0 and
            all(math.isfinite(x) and x > 0 for x in penalties)):
        raise ValueError("Positive finite guard grid required")
    logs = [math.log10(x) for x in penalties]
    if any(right <= left for left, right in zip(logs, logs[1:])):
        raise ValueError("Strictly increasing penalty grid required")
    tolerance = 32 * math.ulp(max(1., max(abs(x) for x in logs)))
    low = logs[0] + guard_decades
    high = logs[-1] - guard_decades
    return {summaries[index + 1]["id"] for index, value in enumerate(logs)
            if value <= low + tolerance or value >= high - tolerance}


def fp64_intercept_candidate(intercept, n_coefficients):
    """Represent an unbounded FP64 intercept without an exactness claim.

    Until an outward numerical error bound is supplied, this candidate keeps
    the finite-family selection closed.  The fitted scalar remains available
    as an operational value and is not promoted to a certified result.
    """
    if not (math.isfinite(intercept) and type(n_coefficients) is int and n_coefficients >= 0):
        raise ValueError("Finite FP64 intercept and coefficient width required")
    import numpy as np
    return {"id": "INTERCEPT_ONLY", "penalty": None,
        "status": "INCOMPLETE_FP64_INTERCEPT_BOUND", "intercept": float(intercept),
        "coefficients": np.zeros(n_coefficients), "coefficient_error_bound": None,
        "iterations": 0, "elapsed_seconds": 0.,
        "diagnostics": {"numerical_scope": "FP64_CONDITIONAL_NOT_OUTWARD_BOUNDED"},
        "provenance": {"role": "FIT", "method": "FINITE_INTERCEPT_ONLY_FP64",
                       "exact_arithmetic_claimed": False}}


def selection_decision(summaries, penalties):
    """Use all inclusive guard candidates and reject missing numeric intervals."""
    edge = inclusive_guard_candidate_ids(summaries, penalties)
    complete = bool(summaries) and all(
        str(row.get("status", "")).startswith("CONVERGED") and
        all(isinstance(row.get(key), (int, float)) and
            math.isfinite(row[key]) for key in ("select_loss_lower", "select_loss_upper")) and
        0 <= row["select_loss_lower"] <= row["select_loss_upper"] <= 1
        for row in summaries)
    if not complete:
        return "INCOMPLETE_CANDIDATES", []
    best_upper = min(row["select_loss_upper"] for row in summaries)
    plausible = [row["id"] for row in summaries if row["select_loss_lower"] <= best_upper]
    status = ("NEEDS_GUARD_EXTENSION" if edge.intersection(plausible) else
              "SELECTED_UNIQUE" if len(plausible) == 1 else "NUMERICALLY_UNRESOLVED")
    return status, plausible
