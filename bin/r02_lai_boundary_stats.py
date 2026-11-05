#!/usr/bin/env python3
"""Aggregate M37 boundary matches with two dynamic-programming rows.

This module preserves the objective and eligibility rule of
``m37_trace_score.match_pairs`` but does not reconstruct matched index pairs.
"""
from __future__ import annotations

import math

import numpy as np

from m37_trace_core import require
from m37_trace_score import boundaries


_SCORE_DTYPE = np.dtype([("matched", np.int64), ("absolute_error_cm", np.float64)])


def _dp_payload_bytes(predicted_count: int) -> int:
    """Bytes in two DP rows; each cell stores one count and one error sum."""
    return 2 * (predicted_count + 1) * _SCORE_DTYPE.itemsize


def _ordered_match_stats(truth, predicted, tolerance, memory_bytes):
    """Return M37's optimal match count and error sum without its index path."""
    require(type(memory_bytes) is int and memory_bytes > 0,
            "Boundary matcher requires a positive integer memory bound")
    require(math.isfinite(tolerance) and tolerance >= 0,
            "Boundary tolerance must be finite and nonnegative")
    if not truth or not predicted:
        return 0, 0.0

    required = _dp_payload_bytes(len(predicted))
    require(required <= memory_bytes,
            "Two-row boundary matcher exceeds configured memory bound")
    score = np.empty((2, len(predicted) + 1), dtype=_SCORE_DTYPE)
    score["matched"].fill(0)
    score["absolute_error_cm"].fill(0.0)

    for truth_index, actual in enumerate(truth, start=1):
        previous = score[(truth_index - 1) % 2]
        current = score[truth_index % 2]
        current[0] = (0, 0.0)
        previous_count = previous["matched"]
        previous_error = previous["absolute_error_cm"]
        current_count = current["matched"]
        current_error = current["absolute_error_cm"]
        for predicted_index, predicted_boundary in enumerate(predicted, start=1):
            # M37 lists the upward option before the leftward option. Keeping it
            # on an exact tie reproduces that ordering, although the returned
            # aggregate is identical for either path.
            best_count = int(previous_count[predicted_index])
            best_error = float(previous_error[predicted_index])
            left_count = int(current_count[predicted_index - 1])
            left_error = float(current_error[predicted_index - 1])
            if left_count > best_count or (left_count == best_count and left_error < best_error):
                best_count, best_error = left_count, left_error

            distance = abs(actual[0] - predicted_boundary[0])
            if distance <= tolerance and actual[1:] == predicted_boundary[1:]:
                diagonal_count = int(previous_count[predicted_index - 1]) + 1
                diagonal_error = float(previous_error[predicted_index - 1]) + float(distance)
                if (diagonal_count > best_count or
                        (diagonal_count == best_count and diagonal_error < best_error)):
                    best_count, best_error = diagonal_count, diagonal_error
            current_count[predicted_index] = best_count
            current_error[predicted_index] = best_error

    final = score[len(truth) % 2, -1]
    return int(final["matched"]), float(final["absolute_error_cm"])


def boundary_report_stats(truth, hard, mask, cm, groups, memory_bytes):
    """Return ``boundary_report`` aggregates without storing matched paths.

    Missing positions split observed runs exactly as in ``r02_lai_evaluate``.
    ``memory_bytes`` bounds the two-row dynamic-programming array for each run.
    """
    # Resolve these historical primitives at call time so a future evaluator
    # can import this module without creating an import cycle.
    from r02_lai_evaluate import TOLERANCES, boundary_rates

    require(type(memory_bytes) is int and memory_bytes > 0,
            "Boundary matcher requires a positive integer memory bound")
    totals = {str(tolerance): dict(truth=0, predicted=0, matched=0,
                                   absolute_error_cm=0., observed_span_cm=0.)
              for tolerance in TOLERANCES}
    for _, left, right in groups:
        observed = np.asarray(mask[left:right], dtype=bool)
        edges = np.flatnonzero(np.diff(np.r_[False, observed, False]))
        for begin, end in zip(edges[::2], edges[1::2]):
            begin, end = int(begin) + left, int(end) + left
            actual = boundaries(truth[begin:end], cm[begin:end])
            predicted = boundaries(hard[begin:end], cm[begin:end])
            for tolerance in TOLERANCES:
                matched, error = _ordered_match_stats(
                    actual, predicted, tolerance, memory_bytes)
                row = totals[str(tolerance)]
                row["truth"] += len(actual)
                row["predicted"] += len(predicted)
                row["matched"] += matched
                row["absolute_error_cm"] += error
                row["observed_span_cm"] += float(cm[end - 1] - cm[begin])
    return {key: boundary_rates(value) for key, value in totals.items()}
