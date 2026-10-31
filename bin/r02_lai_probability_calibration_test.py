#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import r02_lai_probability_calibration as calibration


class TemperatureControlTest(unittest.TestCase):
    def test_identity_is_exact_and_preserves_zero(self):
        probabilities = np.array([[[0.0, 0.2, 0.3, 0.0, 0.1, 0.4]]], np.float32)
        scaled = calibration.temperature_scale(probabilities, 1.0)
        self.assertTrue(np.array_equal(probabilities, scaled))
        self.assertEqual(probabilities.dtype, scaled.dtype)
        self.assertIsNot(probabilities, scaled)

    def test_nonidentity_preserves_zero_support_and_normalization(self):
        probabilities = np.array([[0.0, 0.2, 0.3, 0.0, 0.1, 0.4]])
        scaled = calibration.temperature_scale(probabilities, 2.0)
        self.assertTrue(np.array_equal(probabilities == 0, scaled == 0))
        self.assertAlmostEqual(float(scaled.sum()), 1.0, places=14)

    def test_nan_outside_evaluation_mask_is_preserved(self):
        probabilities = np.array([
            [0.0, 0.2, 0.3, 0.0, 0.1, 0.4],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ], dtype=np.float32)
        mask = np.array([True, False])
        identity = calibration.temperature_scale(probabilities, 1.0, mask)
        self.assertEqual(identity.tobytes(), probabilities.tobytes())
        scaled = calibration.temperature_scale(probabilities, 2.0, mask)
        self.assertTrue(np.isnan(scaled[1]).all())
        self.assertAlmostEqual(float(scaled[0].sum()), 1.0, places=14)

    def test_nan_inside_evaluation_mask_is_rejected(self):
        probabilities = np.array([
            [0.0, 0.2, 0.3, 0.0, 0.1, 0.4],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ])
        with self.assertRaisesRegex(ValueError, "evaluated support"):
            calibration.temperature_scale(probabilities, 1.0,
                                          np.array([False, True]))

    def test_invalid_temperature_and_probability_rejected(self):
        probabilities = np.full((1, 6), 1 / 6)
        for value in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                calibration.temperature_scale(probabilities, value)
        with self.assertRaises(ValueError):
            calibration.temperature_scale(np.ones((1, 6)), 1.0)

    def test_missing_split_is_explicitly_blocked(self):
        receipt = calibration.blocked_receipt()
        self.assertEqual(receipt["status"], calibration.BLOCKED)
        self.assertEqual(receipt["temperature_parameters_fitted"], 0)
        self.assertFalse(receipt["select_opened"])

    def _write_split(self, directory: Path, overlap: bool = False):
        evidence = directory / "donor-evidence.json"
        evidence.write_text("{}")
        evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
        split = {
            "schema": calibration.SPLIT_SCHEMA,
            "role": "FIT",
            "donor_sets_authenticated": True,
            "source_evidence": {"path": str(evidence), "sha256": evidence_sha},
            "units": [
                {"case_id": "c1", "person_id": "p1", "partition": "CALIBRATION",
                 "donor_ids": ["d1", "d2"]},
                {"case_id": "c1", "person_id": "p2", "partition": "EVALUATION",
                 "donor_ids": ["d2" if overlap else "d3"]},
            ],
        }
        path = directory / "split.json"
        path.write_text(json.dumps(split, sort_keys=True))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_only_complete_donor_disjoint_inner_fit_is_admitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_split(Path(temporary))
            result = calibration.authenticate_split(
                path, digest, {("c1", "p1"), ("c1", "p2")})
            self.assertEqual(result["calibration_donors"], ["d1", "d2"])
            self.assertEqual(result["evaluation_donors"], ["d3"])

    def test_overlapping_donor_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_split(Path(temporary), overlap=True)
            with self.assertRaisesRegex(ValueError, "donors overlap"):
                calibration.authenticate_split(
                    path, digest, {("c1", "p1"), ("c1", "p2")})

    def test_temperature_fit_recovers_synthetic_overconfidence(self):
        probabilities = np.array([
            [0.98, 0.004, 0.004, 0.004, 0.004, 0.004],
            [0.98, 0.004, 0.004, 0.004, 0.004, 0.004],
            [0.004, 0.98, 0.004, 0.004, 0.004, 0.004],
            [0.004, 0.98, 0.004, 0.004, 0.004, 0.004],
        ])
        truth = np.array([0, 1, 1, 0])
        observed = np.ones(4, dtype=bool)
        fitted = calibration.fit_temperature(probabilities, truth, observed)
        before = calibration.mean_log_loss(probabilities, truth, observed)
        after = calibration.mean_log_loss(
            calibration.temperature_scale(probabilities, fitted), truth, observed)
        self.assertGreater(fitted, 1.0)
        self.assertLess(after, before)

    def test_weighted_beta_fit_uses_supplied_hierarchical_weights(self):
        probabilities = np.array([
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.1, 0.7, 0.05, 0.05, 0.05, 0.05],
        ])
        truth = np.array([0, 1, 1])
        # First person has two queries (1/4 each); second has one query (1/2).
        weights = np.array([0.25, 0.25, 0.5])
        result = calibration.fit_weighted_temperature(probabilities, truth, weights)
        self.assertEqual(result["status"], "INTERIOR")
        self.assertGreater(result["temperature"], 1.0)
        self.assertLessEqual(result["log_probability_error_bound"], 1e-10)

    def test_weighted_beta_fit_reports_unrepairable_true_zero(self):
        probabilities = np.array([[0.0, 0.4, 0.3, 0.1, 0.1, 0.1]])
        result = calibration.fit_weighted_temperature(
            probabilities, np.array([0]), np.array([1.0]))
        self.assertEqual(result["status"], "IMPOSSIBLE_TRUE_PROBABILITY_ZERO")
        self.assertEqual(result["zero_true_probability_weight"], 1.0)

    def test_weighted_beta_fit_reports_boundaries_without_inventing_T(self):
        uniform = np.full((2, 6), 1 / 6)
        flat = calibration.fit_weighted_temperature(
            uniform, np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertEqual(flat["status"], "ALL_BETA_EQUIVALENT")
        perfect = np.array([[0.8, 0.04, 0.04, 0.04, 0.04, 0.04]])
        boundary = calibration.fit_weighted_temperature(
            perfect, np.array([0]), np.array([1.0]))
        self.assertEqual(boundary["status"], "BETA_INFINITY_BOUNDARY")
        self.assertIsNone(boundary["beta"])

    def test_chunked_log_probability_solver_matches_array_solver(self):
        probabilities = np.array([
            [0.90, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0.05, 0.70, 0.05, 0.05, 0.10, 0.05],
        ] * 600)
        truth = np.tile(np.array([1, 1]), 600)
        weights = np.full(len(truth), 1 / len(truth))
        in_memory = calibration.fit_weighted_temperature(
            probabilities, truth, weights, log_probability_tolerance=1e-8)
        chunked = calibration.fit_weighted_log_probabilities(
            np.log(probabilities), truth, weights,
            log_probability_tolerance=1e-8, chunk_rows=1024)
        self.assertEqual(chunked["status"], "INTERIOR")
        self.assertAlmostEqual(chunked["beta"], in_memory["beta"], places=12)
        self.assertAlmostEqual(chunked["nll"], in_memory["nll"], places=12)

    def test_numeric_zero_derivative_is_reported_unresolved(self):
        probabilities = np.array([
            [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
        ])
        result = calibration.fit_weighted_temperature(
            probabilities, np.array([0, 1]), np.array([0.5, 0.5]))
        self.assertEqual(
            result["status"], "UNRESOLVED_NUMERIC_DERIVATIVE_AT_BETA_ZERO")
        self.assertIn("derivative_at_beta_zero", result["diagnostics"])
        self.assertIn("derivative_at_beta_one", result["diagnostics"])
        self.assertIn("limit_derivative", result["diagnostics"])

    def test_noninteger_truth_is_rejected(self):
        probabilities = np.full((1, 6), 1 / 6)
        with self.assertRaisesRegex(ValueError, "integer"):
            calibration.fit_weighted_temperature(
                probabilities, np.array([1.0]), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
