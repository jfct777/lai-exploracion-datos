#!/usr/bin/env python3
"""Synthetic equivalence and memory tests for two-row boundary statistics."""
from __future__ import annotations

import random
import unittest

import numpy as np

from m37_trace_score import match_pairs
from r02_lai_boundary_stats import (
    _dp_payload_bytes, _ordered_match_stats, boundary_report_stats,
)
from r02_lai_evaluate import TOLERANCES, boundary_report


def reference_stats(truth, predicted, tolerance):
    pairs = match_pairs(truth, predicted, tolerance)
    return len(pairs), sum(abs(truth[i][0] - predicted[j][0]) for i, j in pairs)


class BoundaryStatsTest(unittest.TestCase):
    def test_ordered_match_matches_original_for_random_small_cases(self):
        rng = random.Random(20261104)
        transitions = [(left, right) for left in range(6) for right in range(6) if left != right]
        for _ in range(300):
            truth = [(float(value) / 10, *rng.choice(transitions))
                     for value in sorted(rng.randrange(0, 31) for _ in range(rng.randrange(0, 8)))]
            predicted = [(float(value) / 10, *rng.choice(transitions))
                         for value in sorted(rng.randrange(0, 31) for _ in range(rng.randrange(0, 8)))]
            tolerance = rng.choice((0.0, 0.1, 0.2, 0.5, 1.0))
            expected = reference_stats(truth, predicted, tolerance)
            actual = _ordered_match_stats(truth, predicted, tolerance, 1_000_000)
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_ties_preserve_optimal_count_and_minimum_error(self):
        transition = (1, 4)
        cases = [
            ([(0.0, *transition), (2.0, *transition)], [(1.0, *transition)], 1.0),
            ([(0.0, *transition)], [(1.0, *transition), (1.0, *transition)], 1.0),
            ([(0.0, *transition), (1.0, *transition)],
             [(0.4, *transition), (0.6, *transition)], 1.0),
            ([(0.0, 1, 4)], [(0.0, 4, 1)], 0.0),
        ]
        for truth, predicted, tolerance in cases:
            self.assertEqual(
                _ordered_match_stats(truth, predicted, tolerance, 1_000_000),
                reference_stats(truth, predicted, tolerance),
            )

    def test_full_report_matches_original_with_missing_runs_and_groups(self):
        rng = np.random.default_rng(20261104)
        for length in range(1, 20):
            for _ in range(15):
                truth = rng.integers(0, 6, length, dtype=np.int8)
                hard = rng.integers(0, 6, length, dtype=np.int8)
                mask = rng.integers(0, 2, length, dtype=np.int8)
                cm = np.cumsum(rng.choice((0.0, 0.05, 0.1, 0.2), length)).astype(np.float64)
                split = max(1, length // 2)
                groups = [("1", 0, split)]
                if split < length:
                    groups.append(("2", split, length))
                expected = boundary_report(truth, hard, mask, cm, groups, 1_000_000_000)
                actual = boundary_report_stats(truth, hard, mask, cm, groups, 1_000_000)
                self.assertEqual(actual, expected)

    def test_missing_cell_does_not_create_cross_gap_boundary(self):
        truth = np.array([0, 0, 5, 5], dtype=np.int8)
        hard = np.array([0, 0, 1, 5], dtype=np.int8)
        mask = np.array([1, 1, 0, 1], dtype=bool)
        cm = np.array([0.0, 0.1, 0.2, 0.3])
        report = boundary_report_stats(truth, hard, mask, cm, [("1", 0, 4)], 1024)
        for tolerance in TOLERANCES:
            self.assertEqual(report[str(tolerance)]["truth"], 0)
            self.assertEqual(report[str(tolerance)]["predicted"], 0)

    def test_memory_bound_is_linear_in_predicted_boundaries(self):
        transition = (2, 5)
        truth = [(float(i), *transition) for i in range(12)]
        predicted = [(float(i) / 10, *transition) for i in range(4096)]
        required = _dp_payload_bytes(len(predicted))
        self.assertEqual(required, 2 * (len(predicted) + 1) * 16)
        with self.assertRaisesRegex(ValueError, "exceeds configured memory"):
            _ordered_match_stats(truth, predicted, 0.2, required - 1)
        self.assertEqual(
            _ordered_match_stats(truth, predicted, 0.2, required),
            reference_stats(truth, predicted, 0.2),
        )

    def test_invalid_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            boundary_report_stats(np.array([0]), np.array([0]), np.array([1]),
                                  np.array([0.0]), [("1", 0, 1)], 0)


if __name__ == "__main__":
    unittest.main()
