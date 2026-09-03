from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m38b_oof_core import (  # noqa: E402
    analytic_residual,
    build_outer_roles,
    per_person_log_loss,
    slice_features,
    smooth_evidence_triangular,
    voronoi_cm_weights,
)


class M38BOofCoreTest(unittest.TestCase):
    def test_rotation_has_exact_roles_and_one_oof_prediction_per_person(self) -> None:
        keys = np.asarray([f"sample-{index:03d}".encode() for index in range(96)], dtype="S64")
        first, seeds = build_outer_roles(keys, outer_seed=7, inner_seed_start=100)
        second, second_seeds = build_outer_roles(keys, outer_seed=7, inner_seed_start=100)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(seeds, second_seeds)
        self.assertTrue(np.all(np.sum(first == "SCORE", axis=0) == 1))
        self.assertEqual([(row == "TRAIN").sum() for row in first], [48, 48, 48])
        self.assertEqual([(row == "SELECT").sum() for row in first], [16, 16, 16])
        self.assertEqual([(row == "SCORE").sum() for row in first], [32, 32, 32])

    def test_slice_features_remaps_people_events_and_schedule(self) -> None:
        features = {
            "sample_key_sha256": np.asarray([b"a", b"b", b"c"]),
            "baseline_states": np.zeros((3, 2, 6), dtype=np.float32),
            "evidence_field": np.zeros((3, 2, 6), dtype=np.float32),
            "event_counts": np.zeros((3, 2, 1), dtype=np.float32),
            "event_sample": np.asarray([0, 1, 2, 2], dtype=np.uint32),
            "event_values": np.arange(8, dtype=np.float32).reshape(4, 2),
            "event_cM": np.arange(4, dtype=np.float64),
            "schedule_sample": np.asarray([0, 2], dtype=np.uint32),
            "schedule_marker": np.asarray([4, 5], dtype=np.uint32),
            "marker_cM": np.asarray([0.0, 1.0]),
        }
        observed = slice_features(features, [2, 0])
        np.testing.assert_array_equal(observed["sample_key_sha256"], np.asarray([b"c", b"a"]))
        np.testing.assert_array_equal(observed["event_sample"], np.asarray([1, 0, 0]))
        np.testing.assert_array_equal(observed["event_cM"], np.asarray([0.0, 2.0, 3.0]))
        np.testing.assert_array_equal(observed["schedule_sample"], np.asarray([1, 0]))

    def test_voronoi_weighting_is_not_uniform_on_irregular_map(self) -> None:
        marker = np.asarray([0.0, 1.0, 4.0])
        np.testing.assert_allclose(voronoi_cm_weights(marker), np.asarray([0.5, 2.0, 1.5]))
        probability = np.full((1, 3, 6), 0.02, dtype=np.float64)
        probability[:, :, 0] = np.asarray([0.9, 0.5, 0.1])
        probability /= probability.sum(axis=2, keepdims=True)
        truth = np.zeros((1, 3), dtype=np.uint8)
        uniform = per_person_log_loss(probability, truth, marker, weighted=False)
        weighted = per_person_log_loss(probability, truth, marker, weighted=True)
        self.assertFalse(np.isclose(uniform, weighted).all())

    def test_voronoi_plateau_is_shared_and_order_invariant(self) -> None:
        marker = np.asarray([0.0, 0.0, 1.0, 4.0])
        np.testing.assert_allclose(
            voronoi_cm_weights(marker), np.asarray([0.25, 0.25, 2.0, 1.5]),
        )
        truth = np.zeros((1, 4), dtype=np.uint8)
        first = np.full((1, 4, 6), 0.02, dtype=np.float64)
        first[:, :, 0] = np.asarray([0.9, 0.2, 0.7, 0.8])
        first /= first.sum(axis=2, keepdims=True)
        second = first.copy()
        second[:, [0, 1]] = second[:, [1, 0]]
        np.testing.assert_allclose(
            per_person_log_loss(first, truth, marker, weighted=True),
            per_person_log_loss(second, truth, marker, weighted=True),
        )

    def test_analytic_zero_strength_is_literal_baseline(self) -> None:
        rng = np.random.default_rng(4)
        baseline = rng.dirichlet(np.ones(6), size=(2, 4)).astype(np.float32)
        evidence = rng.normal(size=baseline.shape).astype(np.float32)
        observed = analytic_residual(baseline, evidence, 0.0)
        self.assertTrue(np.array_equal(observed, baseline))

    def test_triangular_evidence_uses_cm_radius_and_zero_outside(self) -> None:
        marker = np.asarray([0.0, 0.1, 0.2, 0.3, 0.5])
        field = np.zeros((1, 5, 6), dtype=np.float32)
        field[0, 2, 4] = 2.0
        observed = smooth_evidence_triangular(field, marker, 0.2)
        np.testing.assert_allclose(
            observed[0, :, 4], np.asarray([0.0, 1.0, 2.0, 1.0, 0.0]), atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
