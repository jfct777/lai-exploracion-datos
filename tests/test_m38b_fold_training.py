from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from m33_safe_bridge_core import write_deterministic_npz  # noqa: E402
from m38b_make_folds import build as build_folds  # noqa: E402
from m38b_partition_fold import partition_features, partition_truth  # noqa: E402
from m38b_train_fold import analytic_prediction  # noqa: E402
from m37_trace_train import deterministic_train_tune  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_features(people: int = 96, markers: int = 7) -> dict[str, np.ndarray]:
    keys = np.asarray([f"sample-{index:03d}".encode() for index in range(people)], dtype="S64")
    baseline = np.full((people, markers, 6), 1.0 / 6.0, dtype=np.float32)
    event_sample = np.arange(people, dtype=np.uint32)
    event_marker = np.full(people, markers // 2, dtype=np.uint32)
    evidence = np.zeros_like(baseline)
    evidence[:, markers // 2, 0] = 2.0
    return {
        "sample_key_sha256": keys,
        "marker_pos": np.arange(100, 100 + markers, dtype=np.int64),
        "baseline_states": baseline,
        "evidence_field": evidence,
        "event_counts": np.ones((people, markers, 1), dtype=np.float32),
        "calendar_marker": np.asarray([markers // 2], dtype=np.uint32),
        "schedule_sample": event_sample.copy(),
        "schedule_marker": event_marker.copy(),
        "event_sample": event_sample,
        "event_locus": np.zeros(people, dtype=np.uint32),
        "event_genotype": np.ones(people, dtype=np.uint8),
        "event_target_callable": np.ones(people, dtype=np.uint8),
        "event_reference_callable": np.ones(people, dtype=np.uint8),
        "event_loglik": np.zeros((people, 6), dtype=np.float32),
        "event_pooled_loglik": np.zeros((people, 6), dtype=np.float32),
        "event_uncertainty": np.zeros((people, 3), dtype=np.float32),
        "event_support": np.ones((people, 3), dtype=np.float32),
        "event_context_7mer": np.zeros(people, dtype=np.uint16),
        "event_carrier_support": np.zeros(people, dtype=np.float32),
        "event_origin_support": np.zeros(people, dtype=np.float32),
        "context_7mer_available": np.zeros(1, dtype=np.uint8),
        "carrier_support_available": np.zeros(1, dtype=np.uint8),
        "origin_support_available": np.zeros(1, dtype=np.uint8),
        "marker_cM": np.linspace(0.0, 0.6, markers),
        "marker_axis_sha256": np.asarray(["fixture-axis"]),
        "event_cM": np.full(people, 0.3, dtype=np.float64),
        "event_marker_left": event_marker.copy(),
        "event_marker_right": event_marker.copy(),
        "event_delta_left_cM": np.zeros(people, dtype=np.float32),
        "event_delta_right_cM": np.zeros(people, dtype=np.float32),
        "ancestry_names": np.asarray(["AFR", "EUR", "NAM"]),
        "state_names": np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"]),
        "reference_frequency_policy": np.asarray(["REF_TRAIN"]),
        "fold_ensemble_available": np.zeros(1, dtype=np.uint8),
        "baseline_method": np.asarray(["FLARE_MINUS"]),
        "baseline_source_sha256": np.asarray(["a" * 64]),
        "marker_axis_source_sha256": np.asarray(["b" * 64]),
        "event_values": np.zeros((people, 23), dtype=np.float32),
    }


class M38BFoldTrainingTest(unittest.TestCase):
    def test_feature_and_truth_partition_have_exact_disjoint_axes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "features.npz"
            truth = root / "truth.npz"
            folds = root / "folds.npz"
            write_deterministic_npz(source, fixture_features())
            np.savez(
                truth,
                sample_key_sha256=fixture_features()["sample_key_sha256"],
                marker_pos=fixture_features()["marker_pos"],
                state_labels=np.zeros((96, 7), dtype=np.uint8),
            )
            build_folds(source, folds, 7, 100)
            feature_args = Namespace(
                source=source, folds=folds, fold=0,
                source_receipt=root / "features.receipt.json",
                arm="RE",
                fit_output=root / "fit.features.npz",
                score_output=root / "score.features.npz",
            )
            feature_args.source_receipt.write_text(json.dumps({
                "stage": "M37_TRACE_MATERIALIZE", "arm": "RE",
                "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            feature_receipt = partition_features(feature_args)
            truth_args = Namespace(
                source=truth, folds=folds, fold=0,
                source_receipt=root / "alignment.receipt.json",
                fit_output=root / "fit.truth.npz",
                score_output=root / "score.truth.npz",
            )
            truth_args.source_receipt.write_text(json.dumps({
                "stage": "M38B_ALIGN_FULL_AND_TRUTH_TO_F_MINUS_S660",
                "decision": "PASS_EXACT_SHARED_F_MINUS_S660_SCORING_GRID",
                "counts": {"target_people": 96, "F_minus_S660": 42326},
                "outputs": {truth.name: {"sha256": digest(truth)}},
            }), encoding="utf-8")
            truth_receipt = partition_truth(truth_args)
            self.assertFalse(feature_receipt["truth_read"])
            self.assertFalse(truth_receipt["model_selection_performed"])
            with np.load(feature_args.fit_output, allow_pickle=False) as fit, \
                    np.load(feature_args.score_output, allow_pickle=False) as score, \
                    np.load(truth_args.fit_output, allow_pickle=False) as fit_truth, \
                    np.load(truth_args.score_output, allow_pickle=False) as score_truth:
                self.assertEqual(len(fit["sample_key_sha256"]), 64)
                self.assertEqual(len(score["sample_key_sha256"]), 32)
                self.assertEqual(len(np.intersect1d(
                    fit["sample_key_sha256"], score["sample_key_sha256"],
                )), 0)
                np.testing.assert_array_equal(
                    fit["sample_key_sha256"], fit_truth["sample_key_sha256"],
                )
                np.testing.assert_array_equal(
                    score["sample_key_sha256"], score_truth["sample_key_sha256"],
                )

    def test_analytic_selector_uses_select_and_zero_is_available(self) -> None:
        full = fixture_features(64)
        score = fixture_features(32)
        # Evidence favors AA, matching the SELECT truth.  A non-zero lambda
        # should therefore be selected from the frozen grid.
        truth = np.zeros((64, 7), dtype=np.uint8)
        inner_seed = next(
            seed for seed in range(100, 100_000)
            if len(deterministic_train_tune(full, seed, 0.25)[1]) == 16
        )
        prediction, diagnostics = analytic_prediction(
            full, score, truth, inner_seed=inner_seed, radius_cm=0.2,
            lambda_grid=(0.0, 0.25, 0.5, 1.0, 2.0),
        )
        self.assertGreater(diagnostics["selected_lambda"], 0.0)
        self.assertEqual(prediction.shape, (32, 7, 6))
        self.assertTrue(np.allclose(prediction.sum(axis=2), 1.0))

    def test_model_interface_has_no_score_truth_argument(self) -> None:
        # The production fitter receives FIT truth only.  SCORE truth is a
        # separate downstream artifact consumed by the scorer.
        parameters = set(inspect.signature(analytic_prediction).parameters)
        self.assertNotIn("score_truth", parameters)
        source = (ROOT / "bin/m38b_train_fold.py").read_text(encoding="utf-8")
        self.assertNotIn("--score-truth", source)

    def test_rd_off_is_never_fitted(self) -> None:
        source = (ROOT / "bin/m38b_train_fold.py").read_text(encoding="utf-8")
        self.assertIn('{"RE", "RD", "SHAM", "POSITIVE"}', source)
        self.assertIn("RD/OFF is exact F-minus-S660 and must not be fitted", source)


if __name__ == "__main__":
    unittest.main()
