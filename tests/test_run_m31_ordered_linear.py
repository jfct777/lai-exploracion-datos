#!/usr/bin/env python3
"""Load-bearing tests for the streaming M31 ordered-linear runner.

These tests stay synthetic and local.  They exercise numerical equivalence,
grouped inner CV, global-count scoring, reconstructive bootstrap, the additive
H ceiling, and the truth-blind prediction boundary before any real M31 run.
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "bin" / "run_m31_ordered_linear.py"
CORE_PATH = REPO / "bin" / "m31_ordered_linear.py"
PRE2_RECEIPT_PATH = REPO / "bin" / "m31_pre2_receipt.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load("m31_ordered_linear", CORE_PATH)
RUNNER = _load("run_m31_ordered_linear_test", RUNNER_PATH)
PRE2_RECEIPT = _load("m31_pre2_receipt_test", PRE2_RECEIPT_PATH)


def _boundary_counts(truth: int, predicted: int, matched: int):
    return {
        tolerance: (truth, predicted, matched, tuple(0.05 for _ in range(matched)))
        for tolerance in CORE.BOUNDARY_TOLERANCES_CM
    }


def _score_counts(
    sample: str,
    *,
    total_cm: float,
    dose_error: float,
    truth_boundaries: int = 1,
    predicted_boundaries: int = 1,
    matched_boundaries: int = 1,
):
    return RUNNER.ScoreCounts(
        sample_id=sample,
        dose_mae_numerator=np.full(3, dose_error, dtype=float),
        brier_numerator=dose_error,
        total_cm=total_cm,
        confusion=np.zeros((6, 6), dtype=float),
        boundary=_boundary_counts(truth_boundaries, predicted_boundaries, matched_boundaries),
    )


def _mock_threadpool_runtime():
    return {
        "controller": "threadpoolctl",
        "controller_version": "3.6.0",
        "status": "VERIFIED_ALL_POOLS_SINGLE_THREAD",
        "pools": [{
            "user_api": "blas", "internal_api": "openblas", "prefix": "libopenblas",
            "version": "0.3.test", "num_threads": 1, "threading_layer": "pthreads",
            "architecture": "fixture",
        }],
    }


def _valid_fitted(arm: str):
    f0 = {
        "boundary_f1_0.2cM": 0.5,
        "false_transitions_per_cM_0.2cM": 0.2,
        "macro_ancestry_dose_mae": 0.3,
        "haplotype_brier": 0.4,
    }
    metrics = {
        "boundary_f1_0.2cM": 0.7,
        "false_transitions_per_cM_0.2cM": 0.1,
        "macro_ancestry_dose_mae": 0.2,
        "haplotype_brier": 0.3,
    }
    selected_pair = (1.0, 10.0)
    candidates = tuple({
        "boundary_weight": float(weight),
        "alpha": float(alpha),
        **metrics,
        "guarded": True,
        "guard_failures": [],
        "selected": (float(weight), float(alpha)) == selected_pair,
    } for weight in CORE.EXPECTED_BOUNDARY_WEIGHTS for alpha in CORE.EXPECTED_ALPHAS)
    return SimpleNamespace(
        arm=arm, alpha=selected_pair[1], boundary_weight=selected_pair[0],
        cv_boundary_f1=metrics["boundary_f1_0.2cM"],
        cv_false_transitions_per_cm=metrics["false_transitions_per_cM_0.2cM"],
        cv_macro_ancestry_dose_mae=metrics["macro_ancestry_dose_mae"],
        cv_brier=metrics["haplotype_brier"], guarded=True, selection_status="GUARDED_CONFIG",
        feature_count=RUNNER.PILOT_FEATURE_COUNTS[arm], f0_cv_metrics=f0,
        candidate_table=candidates, guard_failures=(),
        sufficient_stats_sha256={str(int(weight)): "a" * 64 for weight in CORE.EXPECTED_BOUNDARY_WEIGHTS},
        fold_stats_sha256={
            f"fold{fold}:{int(weight)}": "b" * 64
            for fold in range(3) for weight in CORE.EXPECTED_BOUNDARY_WEIGHTS
        },
    )


class SufficientStatisticsEquivalenceTest(unittest.TestCase):
    def test_fit_from_stats_matches_core_fit_with_voronoi_boundary_weights(self):
        genetic_map = CORE.GeneticMap(
            np.array([100, 500], dtype=np.int64),
            np.array([0.0, 2.0], dtype=float),
        )
        _left, _right, marker_weights = RUNNER._physical_voronoi(
            np.array([100, 180, 340, 500], dtype=np.int64), genetic_map
        )
        row_voronoi = np.repeat(marker_weights, 2)
        rng = np.random.default_rng(3101)
        all_x = []
        all_y = []
        all_w = []
        all_ids = []
        combined = RUNNER.RawRidgeStats.zero(5)
        for index, sample in enumerate(("S0", "S1", "S2", "S3")):
            x = rng.normal(size=(len(row_voronoi), 5))
            residual = rng.normal(scale=0.1, size=(len(row_voronoi), 3))
            boundary = np.arange(len(row_voronoi)) % (index + 2) == 0
            weights = row_voronoi * np.where(boundary, 5.0, 1.0)
            combined.add(RUNNER.sample_stats(x, residual, weights))
            all_x.append(x)
            all_y.append(residual)
            all_w.append(weights)
            all_ids.extend([sample] * len(row_voronoi))

        streamed = RUNNER.fit_from_stats(combined, alpha=0.1)
        direct = CORE.fit_weighted_standardized_ridge_residual(
            np.vstack(all_x),
            np.vstack(all_y),
            all_ids,
            weights=np.concatenate(all_w),
            alpha=0.1,
        )
        np.testing.assert_allclose(streamed.feature_mean, direct.feature_mean, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(streamed.feature_scale, direct.feature_scale, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(streamed.residual_intercept, direct.residual_intercept, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(streamed.coefficients, direct.coefficients, rtol=2e-12, atol=2e-12)
        self.assertEqual(streamed.n_individuals, 4)
        self.assertAlmostEqual(streamed.normalized_weight_sum, 4.0)


class GroupedCvSelectionTest(unittest.TestCase):
    class _Root:
        name = "root17"
        seed = 20260817
        samples = tuple(f"S{index}" for index in range(6))
        marker_weights_cm = np.array([0.4, 0.6], dtype=float)
        boundary_rows = tuple(np.ones(4, dtype=bool) for _ in samples)
        truth_markers = np.zeros((2, 6, 2, 3), dtype=float)
        flare = SimpleNamespace(probabilities=np.zeros((2, 6, 2, 3), dtype=float))

        @staticmethod
        def features(sample_index: int, arm: str, replicate=None):
            return np.full((4, 2), float(sample_index + 1), dtype=float)

    class _Model:
        def __init__(self, code: float):
            self.code = code

        def predict(self, features, baseline):
            return np.full_like(np.asarray(baseline, dtype=float), self.code)

    def test_grouped_cv_has_no_individual_leakage_and_uses_guarded_tie_break(self):
        fit_calls = []

        def fake_fit(stats, alpha):
            # Each individual contributes two total Voronoi-cM before the
            # boundary multiplier.  This recovers the candidate weight.
            boundary_weight = stats.weight_sum / (2.0 * stats.individuals)
            fit_calls.append((stats.individuals, boundary_weight, float(alpha)))
            return self._Model(10.0 * boundary_weight + float(alpha))

        def fake_score(_root, _truth, _sample_index, predicted):
            return {}, float(np.asarray(predicted).flat[0])

        def fake_summary(tokens):
            code = round(float(tokens[0]), 7)
            if code == 0.0:  # frozen F0
                return {
                    "boundary_f1_0.2cM": 0.40,
                    "false_transitions_per_cM_0.2cM": 0.50,
                    "macro_ancestry_dose_mae": 0.50,
                    "haplotype_brier": 0.50,
                }
            if code == 51.0:  # best F1, but prespecified guardrail fails
                return {
                    "boundary_f1_0.2cM": 0.99,
                    "false_transitions_per_cM_0.2cM": 0.60,
                    "macro_ancestry_dose_mae": 0.40,
                    "haplotype_brier": 0.30,
                }
            if code == 10.1:  # lower primary endpoint
                f1 = 0.70
            else:  # equal guarded endpoint: tie-break must decide
                f1 = 0.80
            return {
                "boundary_f1_0.2cM": f1,
                "false_transitions_per_cM_0.2cM": 0.40,
                "macro_ancestry_dose_mae": 0.40,
                "haplotype_brier": 0.40,
            }

        with (
            mock.patch.object(RUNNER, "fit_from_stats", side_effect=fake_fit),
            mock.patch.object(RUNNER, "score_sample", side_effect=fake_score),
            mock.patch.object(RUNNER, "summarize_counts", side_effect=fake_summary),
        ):
            truth = RUNNER.TruthBundle(
                root_name="root17",
                segments={},
                markers=np.zeros((2, 6, 2, 3), dtype=float),
                boundary_rows=tuple(np.ones(4, dtype=bool) for _ in range(6)),
            )
            fitted = RUNNER.fit_arm_streaming(
                self._Root(), truth, "C", None,
                alphas=(0.1, 1.0), boundary_weights=(1.0, 5.0), cv_seed=71,
            )

        # Three grouped folds fit on four whole diploid individuals each;
        # only the final selected model sees all six.
        self.assertEqual([item[0] for item in fit_calls[:-1]], [4] * 12)
        self.assertEqual(fit_calls[-1][0], 6)
        self.assertEqual(fitted.boundary_weight, 1.0)
        self.assertEqual(fitted.alpha, 1.0)
        self.assertEqual(fitted.cv_boundary_f1, 0.80)


class ParallelIndividualEquivalenceTest(unittest.TestCase):
    @staticmethod
    def _fixture(fail_index=None):
        samples = tuple(f"S{index}" for index in range(6))
        positions = np.array([100, 200, 300, 400], dtype=np.int64)
        genetic_map = CORE.GeneticMap(
            np.array([100, 401], dtype=np.int64), np.array([0.0, 0.301], dtype=float),
        )
        marker_cm = np.asarray(genetic_map.cm_at(positions), dtype=float)
        left, right, marker_weights = RUNNER._physical_voronoi(positions, genetic_map)
        segments = {
            sample: (
                [CORE.TruthSegment(100, 250, "AFR"), CORE.TruthSegment(250, 401, "EUR")],
                [CORE.TruthSegment(100, 250, "AFR"), CORE.TruthSegment(250, 401, "EUR")],
            )
            for sample in samples
        }
        truth_markers = CORE.truth_at_markers(segments, samples, positions)
        probabilities = np.full((len(positions), len(samples), 2, 3), 1.0 / 3.0, dtype=float)
        boundary_rows = RUNNER._truth_boundary_rows(segments, samples, marker_cm, genetic_map, 0.2)

        class Root:
            @staticmethod
            def support(_arm, _replicate):
                return np.empty((0, 3)), np.empty(0, dtype=bool)

            @staticmethod
            def features(sample_index, _arm, _replicate=None):
                if sample_index == fail_index:
                    raise RuntimeError("synthetic worker failure")
                target = truth_markers[:, sample_index].reshape(-1, 3)
                marker = np.repeat(np.linspace(-1.0, 1.0, len(positions)), 2)[:, None]
                hap = np.tile([0.0, 1.0], len(positions))[:, None]
                base = np.column_stack([target, marker, hap])
                padding = np.zeros((len(base), RUNNER.PILOT_FEATURE_COUNTS["C"] - base.shape[1]))
                return np.asarray(np.column_stack([base, padding]), dtype=np.float32)

        root = Root()
        root.name = "root17"
        root.seed = 20260817
        root.flare = SimpleNamespace(probabilities=probabilities)
        root.marker_positions = positions
        root.marker_cm = marker_cm
        root.marker_weights_cm = marker_weights
        root.cell_left_bp = left
        root.cell_right_bp = right
        root.samples = samples
        root.genetic_map = genetic_map
        truth = RUNNER.TruthBundle("root17", segments, truth_markers, boundary_rows)
        return root, truth

    def test_workers1_and4_are_exact_for_stats_selection_metrics_model_and_predictions(self):
        root, truth = self._fixture()
        kwargs = dict(
            root=root, truth=truth, arm="C", replicate=None,
            alphas=CORE.EXPECTED_ALPHAS,
            boundary_weights=CORE.EXPECTED_BOUNDARY_WEIGHTS,
            cv_seed=71,
        )
        serial = RUNNER.fit_arm_streaming(**kwargs, workers=1)
        with mock.patch.dict(RUNNER.os.environ, {name: "1" for name in RUNNER.THREAD_LIMIT_ENV}), \
                mock.patch.object(RUNNER, "_threadpool_runtime", return_value=_mock_threadpool_runtime()):
            parallel = RUNNER.fit_arm_streaming(**kwargs, workers=4)

        self.assertEqual(serial.sufficient_stats_sha256, parallel.sufficient_stats_sha256)
        self.assertEqual(serial.fold_stats_sha256, parallel.fold_stats_sha256)
        self.assertEqual(serial.alpha, parallel.alpha)
        self.assertEqual(serial.boundary_weight, parallel.boundary_weight)
        self.assertEqual(serial.f0_cv_metrics, parallel.f0_cv_metrics)
        self.assertEqual(serial.candidate_table, parallel.candidate_table)
        self.assertEqual(serial.guard_failures, parallel.guard_failures)
        self.assertEqual(len(parallel.candidate_table), 18)
        self.assertEqual(sum(bool(row["selected"]) for row in parallel.candidate_table), 1)
        for row in parallel.candidate_table:
            self.assertIn("guard_failures", row)
            self.assertIn("guarded", row)
        for name in ("feature_mean", "feature_scale", "residual_intercept", "coefficients"):
            np.testing.assert_array_equal(getattr(serial.model, name), getattr(parallel.model, name))
        serial_artifact = RUNNER.prepare_predictions(root, serial, "C", None)
        parallel_artifact = RUNNER.prepare_predictions(root, parallel, "C", None)
        self.assertEqual(serial_artifact.sha256, parallel_artifact.sha256)
        for left_array, right_array in zip(serial_artifact.arrays, parallel_artifact.arrays):
            np.testing.assert_array_equal(left_array, right_array)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = RUNNER._write_prediction_checkpoint(
                Path(tmp), parallel_artifact, parallel, "c" * 64,
            )
            fit = checkpoint["fit"]
            self.assertEqual(len(fit["candidate_table"]), 18)
            self.assertEqual(fit["f0_cv_metrics"], parallel.f0_cv_metrics)
            self.assertEqual(fit["guard_failures"], list(parallel.guard_failures))
            self.assertEqual(fit["sufficient_stats_sha256"], parallel.sufficient_stats_sha256)
            self.assertEqual(fit["fold_stats_sha256"], parallel.fold_stats_sha256)

    def test_workers_validation_and_worker_failure_are_fail_closed(self):
        root, truth = self._fixture(fail_index=2)
        with self.assertRaisesRegex(RUNNER.RunnerError, ">=1"):
            RUNNER.fit_arm_streaming(
                root, truth, "C", None, (0.1,), (1.0,), cv_seed=71, workers=0,
            )
        with mock.patch.dict(RUNNER.os.environ, {}, clear=True), self.assertRaisesRegex(
            RUNNER.RunnerError, "environment limits"
        ):
            RUNNER.fit_arm_streaming(
                root, truth, "C", None, (0.1,), (1.0,), cv_seed=71, workers=4,
            )
        with mock.patch.dict(RUNNER.os.environ, {name: "1" for name in RUNNER.THREAD_LIMIT_ENV}), \
                mock.patch.object(RUNNER, "_threadpool_runtime", return_value=_mock_threadpool_runtime()), \
                self.assertRaisesRegex(RUNNER.RunnerError, "parallel individual worker failed.*worker failure"):
            RUNNER.fit_arm_streaming(
                root, truth, "C", None, (0.1,), (1.0,), cv_seed=71, workers=4,
            )

    def test_threadpoolctl_runtime_is_required_normalized_and_effectively_single_threaded(self):
        good = SimpleNamespace(
            __version__="3.6.0",
            threadpool_info=lambda: [
                {"user_api": "blas", "internal_api": "openblas", "prefix": "z", "version": "2",
                 "num_threads": 1, "threading_layer": "pthreads", "architecture": "zen",
                 "filepath": "/nonportable/z.so"},
                {"user_api": "blas", "internal_api": "mkl", "prefix": "a", "version": "1",
                 "num_threads": 1, "threading_layer": "intel", "architecture": "x86",
                 "filepath": "/nonportable/a.so"},
            ],
        )
        with mock.patch.object(RUNNER.importlib, "import_module", return_value=good):
            runtime = RUNNER._threadpool_runtime(4)
        self.assertEqual(runtime["status"], "VERIFIED_ALL_POOLS_SINGLE_THREAD")
        self.assertEqual(runtime["controller_version"], "3.6.0")
        self.assertEqual([pool["internal_api"] for pool in runtime["pools"]], ["mkl", "openblas"])
        self.assertTrue(all("filepath" not in pool for pool in runtime["pools"]))

        with mock.patch.object(RUNNER.importlib, "import_module", side_effect=ImportError("absent")), \
                self.assertRaisesRegex(RUNNER.RunnerError, "importable threadpoolctl"):
            RUNNER._threadpool_runtime(4)
        for pools, pattern in (([], "at least one"), ([{"num_threads": 2}], "num_threads=1")):
            fake = SimpleNamespace(__version__="3.6.0", threadpool_info=lambda pools=pools: pools)
            with self.subTest(pools=pools), \
                    mock.patch.object(RUNNER.importlib, "import_module", return_value=fake), \
                    self.assertRaisesRegex(RUNNER.RunnerError, pattern):
                RUNNER._threadpool_runtime(4)

    def test_fail_closed_cli_boundary_reports_worker_error_and_exit_two(self):
        stderr = io.StringIO()
        with mock.patch.object(RUNNER, "main", side_effect=RUNNER.RunnerError("worker exploded")), \
                mock.patch.object(RUNNER.sys, "stderr", stderr):
            self.assertEqual(RUNNER.fail_closed_main(["fit-predict"]), 2)
        self.assertEqual(stderr.getvalue(), "M31_FAIL_CLOSED: worker exploded\n")


class ScoringAndBootstrapTest(unittest.TestCase):
    def test_boundary_f1_uses_global_counts_not_mean_individual_f1(self):
        counts = [
            _score_counts("A", total_cm=1.0, dose_error=0.0,
                          truth_boundaries=100, predicted_boundaries=1, matched_boundaries=1),
            _score_counts("B", total_cm=1.0, dose_error=0.0,
                          truth_boundaries=1, predicted_boundaries=1, matched_boundaries=1),
        ]
        observed = RUNNER.summarize_counts(counts)["boundary_f1_0.2cM"]
        expected_global = 2.0 * (2.0 / 2.0) * (2.0 / 101.0) / ((2.0 / 2.0) + (2.0 / 101.0))
        individual_mean = (2.0 / 101.0 + 1.0) / 2.0
        self.assertAlmostEqual(observed, expected_global)
        self.assertNotAlmostEqual(observed, individual_mean)

    def test_bootstrap_is_deterministic_and_reconstructs_global_metrics(self):
        counts = [
            _score_counts("A", total_cm=1.0, dose_error=0.0),
            _score_counts("B", total_cm=10.0, dose_error=30.0),
            _score_counts("C", total_cm=2.0, dose_error=4.0),
        ]
        replicates = 41
        seed = 31002001
        with (
            mock.patch.object(RUNNER.core, "BOOTSTRAP_REPLICATES", replicates),
            mock.patch.object(RUNNER.core, "BOOTSTRAP_SEED", seed),
        ):
            first = RUNNER.bootstrap_counts(counts)
            second = RUNNER.bootstrap_counts(counts)
        self.assertEqual(first, second)

        rng = np.random.default_rng(seed)
        reconstructed = []
        for _ in range(replicates):
            selected = [counts[index] for index in rng.integers(0, len(counts), size=len(counts))]
            reconstructed.append(RUNNER.summarize_counts(selected)["macro_ancestry_dose_mae"])
        interval = first["metrics"]["macro_ancestry_dose_mae"]
        self.assertAlmostEqual(interval["lower"], float(np.quantile(reconstructed, 0.025)))
        self.assertAlmostEqual(interval["upper"], float(np.quantile(reconstructed, 0.975)))
        self.assertEqual(first["aggregation"], "resample_individual_sufficient_counts_then_reconstruct_global_metrics")


class FeatureHierarchyTest(unittest.TestCase):
    def test_H_has_243_features_and_contains_D_as_an_exact_prefix(self):
        marker_cm = np.linspace(0.0, 2.0, 11)
        rare_cm = np.array([0.05, 0.18, 0.44, 0.81, 1.17, 1.63, 1.95])
        baseline = np.full((len(marker_cm), 2, 3), 1.0 / 3.0)
        truth = np.zeros_like(baseline)
        target = np.array([[0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        support = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0],
            [0.2, 0.3, 0.5], [0, 1, 0], [1, 0, 0],
        ], dtype=float)
        features = CORE.materialize_sample_features(
            marker_cm, rare_cm, baseline, truth, target, support,
            np.zeros(len(rare_cm), dtype=bool), requested_arms=("D", "H"),
        )
        self.assertEqual(features.arms["D"].shape[-1], 171)
        self.assertEqual(features.arms["H"].shape[-1], 243)
        self.assertEqual(features.feature_names["H"][:171], features.feature_names["D"])
        np.testing.assert_array_equal(features.arms["H"][..., :171], features.arms["D"])


class DecisionTableTest(unittest.TestCase):
    DIRECTIONS = ("train_root17_test_root18", "train_root18_test_root17")

    @staticmethod
    def _row(direction, arm, replicate, f1, *, mae=0.4, false=0.4, guarded=True):
        return {
            "direction": direction,
            "arm": arm,
            "sham_replicate": "" if replicate is None else replicate,
            "boundary_f1_0.2cM": f1,
            "macro_ancestry_dose_mae": mae,
            "false_transitions_per_cM_0.2cM": false,
            "ancestry_dose_mae_AFR": mae,
            "ancestry_dose_mae_EUR": mae,
            "ancestry_dose_mae_ASIA": mae,
            "inner_cv_guarded": "" if arm == "F0" else guarded,
        }

    def _metrics(self, scenario):
        profiles = {
            "GO_NEW_ROOTS": {"F0": 0.50, "C": 0.51, "L": 0.55, "D": 0.80, "H": 0.56},
            # L also improves here: H must have priority over LOAD_ONLY.
            "PHASE_CEILING_ONLY": {"F0": 0.50, "C": 0.51, "L": 0.55, "D": 0.54, "H": 0.80},
            "LOAD_ONLY": {"F0": 0.50, "C": 0.51, "L": 0.70, "D": 0.60, "H": 0.60},
            "TRADEOFF": {"F0": 0.50, "C": 0.51, "L": 0.50, "D": 0.80, "H": 0.50},
            "STOP_LINEAR_ORDERED_RARE": {"F0": 0.50, "C": 0.51, "L": 0.50, "D": 0.50, "H": 0.50},
        }
        values = profiles[scenario]
        rows = []
        for direction in self.DIRECTIONS:
            for arm in ("F0", "C", "L", "D", "H"):
                mae = 0.6 if scenario == "TRADEOFF" and arm == "D" else 0.4
                rows.append(self._row(direction, arm, None, values[arm], mae=mae))
            for replicate in range(32):
                rows.append(self._row(direction, "DSHAM", replicate, values["L"] + 0.01))
                rows.append(self._row(direction, "HSHAM", replicate, values["L"] + 0.01))
        return rows

    def test_decision_table_is_exhaustive_and_H_precedes_load_only(self):
        for expected in (
            "GO_NEW_ROOTS", "PHASE_CEILING_ONLY", "LOAD_ONLY",
            "TRADEOFF", "STOP_LINEAR_ORDERED_RARE",
        ):
            with self.subTest(expected=expected):
                observed = RUNNER.decide(self._metrics(expected))
                self.assertEqual(observed["label"], expected)

    def test_any_unguarded_real_or_sham_blocks_go_fail_closed(self):
        metrics = self._metrics("GO_NEW_ROOTS")
        for row in metrics:
            if row["arm"] == "D" and row["sham_replicate"] == "":
                row["inner_cv_guarded"] = False
        self.assertEqual(RUNNER.decide(metrics)["label"], "NO_GUARDED_CONFIG")

        metrics = self._metrics("GO_NEW_ROOTS")
        for row in metrics:
            if row["arm"] == "DSHAM" and row["sham_replicate"] == 7:
                row["inner_cv_guarded"] = False
                row["boundary_f1_0.2cM"] = 0.90
        # The poor sham fit is retained, and its unguarded status also blocks GO.
        self.assertEqual(RUNNER.decide(metrics)["label"], "NO_GUARDED_CONFIG")

    def test_requires_exact_32_unique_shams_and_reports_exact_p_resolution(self):
        metrics = self._metrics("GO_NEW_ROOTS")
        decision = RUNNER.decide(metrics)
        for direction in self.DIRECTIONS:
            for arm in ("D", "H"):
                evidence = decision["null_evidence"][direction][arm]
                self.assertEqual(evidence["replicates"], 32)
                self.assertAlmostEqual(evidence["exploratory_p"], 1.0 / 33.0)
            self.assertTrue(decision["null_evidence"][direction]["D"]["exceeds_all_shams"])

        for mutation in ("missing", "duplicate"):
            with self.subTest(mutation=mutation):
                incomplete = self._metrics("GO_NEW_ROOTS")
                target = next(
                    index for index, row in enumerate(incomplete)
                    if row["direction"] == self.DIRECTIONS[0]
                    and row["arm"] == "DSHAM" and row["sham_replicate"] == 31
                )
                if mutation == "missing":
                    incomplete.pop(target)
                else:
                    incomplete[target]["sham_replicate"] = 30
                with self.assertRaisesRegex(RUNNER.RunnerError, "32|replicate|sham"):
                    RUNNER.decide(incomplete)


class PredictionBoundaryAndScopeTest(unittest.TestCase):
    def test_evaluation_freezes_and_hashes_predictions_before_scoring(self):
        probabilities = np.arange(36, dtype=float).reshape(3, 2, 2, 3)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        root = SimpleNamespace(
            name="root18",
            samples=("S0", "S1"),
            flare=SimpleNamespace(probabilities=probabilities),
        )
        truth = RUNNER.TruthBundle("root18", {}, np.empty((0,)), ())
        artifact = RUNNER.prepare_predictions(root, None, "F0", None)
        self.assertTrue(all(not array.flags.writeable for array in artifact.arrays))
        self.assertEqual(
            artifact.sha256,
            RUNNER._prediction_sha256(root.name, root.samples, artifact.arrays),
        )
        scoring_calls = []

        def scoring_spy(_root, _truth, sample_index, predicted):
            self.assertFalse(predicted.flags.writeable)
            with self.assertRaises(ValueError):
                predicted.flat[0] = 0.0
            scoring_calls.append(sample_index)
            return {"sample_id": root.samples[sample_index]}, object()

        with (
            mock.patch.object(RUNNER, "score_sample", side_effect=scoring_spy),
            mock.patch.object(RUNNER, "summarize_counts", return_value={"boundary_f1_0.2cM": 0.0}),
        ):
            _summary, rows, _counts = RUNNER.score_prediction_artifact(root, truth, artifact)
        self.assertEqual(scoring_calls, [0, 1])
        self.assertEqual([row["sample_id"] for row in rows], ["S0", "S1"])

        tampered_arrays = list(artifact.arrays)
        tampered_arrays[0] = np.array(tampered_arrays[0], copy=True)
        tampered_arrays[0].flat[0] += 1e-6
        tampered_arrays[0].setflags(write=False)
        tampered = replace(artifact, arrays=tuple(tampered_arrays))
        with (
            mock.patch.object(RUNNER, "score_sample") as forbidden_scorer,
            self.assertRaises(RUNNER.RunnerError),
        ):
            RUNNER.score_prediction_artifact(root, truth, tampered)
        forbidden_scorer.assert_not_called()

    def test_real_input_dataclass_and_runner_api_have_no_valid_or_test_bundle(self):
        self.assertEqual(
            set(RUNNER.RootPaths.__dataclass_fields__),
            {"sites", "target", "tree", "pools", "truth", "flare_vcf", "flare_audit"},
        )
        parameters = set(inspect.signature(RUNNER.run_two_root_dev).parameters)
        self.assertTrue({"root17", "root18", "contract"}.issubset(parameters))
        self.assertFalse({"valid", "validation", "test", "holdout"} & parameters)
        for forbidden in (Path("/data/VALID/root17.vcf.gz"), Path("/data/test/root17.vcf.gz")):
            with self.subTest(path=forbidden), self.assertRaises(RUNNER.RunnerError):
                RUNNER._reject_forbidden_partitions([forbidden])
        RUNNER._reject_forbidden_partitions([Path("/data/root18/flare.anc.vcf.gz")])

    def test_pilot_evaluation_feature_bundle_cannot_carry_truth(self):
        self.assertEqual(
            set(RUNNER.FeaturePaths.__dataclass_fields__),
            {"sites", "target", "tree", "pools", "flare_vcf", "flare_audit"},
        )
        self.assertNotIn("truth", RUNNER.FeaturePaths.__dataclass_fields__)
        self.assertIn("truth", RUNNER.TrainingPaths.__dataclass_fields__)

    @unittest.skipUnless(hasattr(RUNNER, "build_parser") and hasattr(RUNNER, "main"),
                         "runner CLI has not been exposed yet")
    def test_cli_exposes_only_authenticated_dry_run_and_known_answer_modes(self):
        parser = RUNNER.build_parser()
        choices = set()
        for action in parser._actions:
            if getattr(action, "choices", None):
                choices.update(action.choices)
        self.assertTrue({"dry-run", "known-answer"}.issubset(choices))
        self.assertFalse({"valid", "validation", "test", "holdout"} & choices)
        subparsers = next(
            action for action in parser._actions
            if isinstance(getattr(action, "choices", None), dict)
            and "fit-predict" in action.choices
        )
        fit_predict_parser = subparsers.choices["fit-predict"]
        fit_predict_actions = {action.dest: action for action in fit_predict_parser._actions}
        self.assertEqual(fit_predict_actions["workers"].default, 1)
        self.assertIs(fit_predict_actions["workers"].type, int)
        self.assertNotIn("eval_root18_truth", fit_predict_actions)

        contract = REPO / "conf" / "m31_ordered_linear_preregistration.json"
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "known"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                return_code = RUNNER.main([
                    "known-answer", "--contract", str(contract), "--outdir", str(outdir),
                ])
            self.assertEqual(return_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "PASS")
            summary = json.loads((outdir / RUNNER.OUTPUT_NAMES["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "KNOWN_ANSWER_NO_REAL_DATA")
            self.assertEqual(summary["scientific_decision"], "NOT_EVALUATED")
            self.assertTrue(summary["prediction_hash_before_truth_scoring"])

        dry_arguments = [
            "dry-run", "--contract", str(contract), "--genetic-map", "/data/root17/map.tsv",
            "--expected-contract-sha256", "a" * 64,
            "--expected-runner-sha256", "b" * 64,
            "--expected-core-sha256", "c" * 64,
        ]
        for root in CORE.ROOTS:
            for key in ("sites", "target", "tree", "pools", "truth", "flare-vcf", "flare-audit"):
                dry_arguments.extend([f"--{root}-{key}", f"/data/{root}/{key}"])
        stdout = io.StringIO()
        with (
            mock.patch.object(RUNNER, "dry_run_estimate", return_value={
                "status": "DRY_RUN_AUTHENTICATED_NO_FIT", "real_execution_started": False,
            }) as dry,
            redirect_stdout(stdout),
        ):
            self.assertEqual(RUNNER.main(dry_arguments), 0)
        dry.assert_called_once()
        self.assertFalse(json.loads(stdout.getvalue())["real_execution_started"])

    def test_cli_enforces_two_process_truth_boundary_and_blocks_full_run(self):
        parser = RUNNER.build_parser()
        subparsers = next(action for action in parser._actions if hasattr(action, "choices") and action.choices)

        def options(command):
            return {
                option
                for action in subparsers.choices[command]._actions
                for option in action.option_strings
            }

        for command in ("pilot", "fit-predict"):
            self.assertNotIn("--eval-root18-truth", options(command))
            self.assertIn("--train-root17-truth", options(command))
        self.assertIn("--eval-root18-truth", options("score-pilot"))
        self.assertIn("--prediction-manifest", options("score-pilot"))
        self.assertNotIn("--benchmark-truth", options("benchmark-sample"))
        with self.assertRaisesRegex(RUNNER.RunnerError, "POST_REQUIRED_FULL_RUN_BLOCKED"):
            RUNNER.main(["run"])


class DurableCheckpointAndProvenanceTest(unittest.TestCase):
    @staticmethod
    def _artifact():
        rng = np.random.default_rng(3199)
        arrays = []
        for _ in range(2):
            raw = rng.uniform(size=(4, 2, 3))
            raw /= raw.sum(axis=-1, keepdims=True)
            raw.setflags(write=False)
            arrays.append(raw)
        sample_ids = ("S0", "S1")
        arrays = tuple(arrays)
        return RUNNER.PredictionArtifact(
            "root18", sample_ids, "D", None, arrays,
            RUNNER._prediction_sha256("root18", sample_ids, arrays),
        )

    def test_checkpoint_is_fsynced_hash_bound_and_resumable_only_in_same_context(self):
        artifact = self._artifact()
        context = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            with mock.patch.object(RUNNER.os, "fsync", wraps=RUNNER.os.fsync) as fsync:
                checkpoint = RUNNER._write_prediction_checkpoint(outdir, artifact, _valid_fitted("D"), context)
            # File data and both containing-directory entries are flushed.
            self.assertGreaterEqual(fsync.call_count, 4)
            checkpoint_path = outdir / checkpoint["checkpoint_file"]
            prediction_path = outdir / checkpoint["prediction_file"]
            self.assertEqual(checkpoint["checkpoint_file_sha256"], CORE.sha256_file(checkpoint_path))
            self.assertEqual(checkpoint["prediction_file_sha256"], CORE.sha256_file(prediction_path))

            resumed, loaded = RUNNER._load_prediction_checkpoint(outdir, "D", context)
            self.assertEqual(resumed.sha256, artifact.sha256)
            self.assertEqual(loaded["context_sha256"], context)
            with self.assertRaisesRegex(RUNNER.RunnerError, "context|status"):
                RUNNER._load_prediction_checkpoint(outdir, "D", "d" * 64)

            stacked = np.load(prediction_path, allow_pickle=False)
            stacked[0, 0, 0, 0] += 1e-5
            with prediction_path.open("wb") as handle:
                np.save(handle, stacked, allow_pickle=False)
            with self.assertRaisesRegex(RUNNER.RunnerError, "SHA-256"):
                RUNNER._load_prediction_checkpoint(outdir, "D", context)

    def test_resume_rejects_manipulated_fit_audit_payloads(self):
        artifact = self._artifact()
        context = "c" * 64

        def mutate_duplicate_candidate(payload):
            payload["fit"]["candidate_table"][1]["boundary_weight"] = \
                payload["fit"]["candidate_table"][0]["boundary_weight"]
            payload["fit"]["candidate_table"][1]["alpha"] = payload["fit"]["candidate_table"][0]["alpha"]

        def mutate_wrong_selection(payload):
            for row in payload["fit"]["candidate_table"]:
                row["selected"] = False
            payload["fit"]["candidate_table"][0]["selected"] = True

        mutations = {
            "fit_null": lambda payload: payload.__setitem__("fit", None),
            "feature_count": lambda payload: payload["fit"].__setitem__("feature_count", 170),
            "candidate_count": lambda payload: payload["fit"]["candidate_table"].pop(),
            "duplicate_candidate": mutate_duplicate_candidate,
            "wrong_selection": mutate_wrong_selection,
            "nonfinite_metric": lambda payload: payload["fit"]["candidate_table"][0].__setitem__(
                "haplotype_brier", float("nan")
            ),
            "guard_mismatch": lambda payload: payload["fit"]["candidate_table"][0].__setitem__(
                "guarded", False
            ),
            "summary_guarded_mismatch": lambda payload: payload["fit"].__setitem__("guarded", False),
            "selection_status_mismatch": lambda payload: payload["fit"].__setitem__(
                "selection_status", "NO_GUARDED_CONFIG"
            ),
            "selected_metric_mismatch": lambda payload: payload["fit"].__setitem__("cv_brier", 0.31),
            "missing_f0_metric": lambda payload: payload["fit"]["f0_cv_metrics"].pop("haplotype_brier"),
            "bad_total_hash": lambda payload: payload["fit"]["sufficient_stats_sha256"].__setitem__(
                "5", "not-a-sha"
            ),
            "bad_fold_hash": lambda payload: payload["fit"]["fold_stats_sha256"].__setitem__(
                "fold2:20", "A" * 64
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                outdir = Path(tmp)
                checkpoint = RUNNER._write_prediction_checkpoint(
                    outdir, artifact, _valid_fitted("D"), context,
                )
                checkpoint_path = outdir / checkpoint["checkpoint_file"]
                payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                mutation(payload)
                checkpoint_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
                with self.assertRaisesRegex(RUNNER.RunnerError, "checkpoint"):
                    RUNNER._load_prediction_checkpoint(outdir, "D", context)

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            f0_artifact = replace(artifact, arm="F0")
            checkpoint = RUNNER._write_prediction_checkpoint(outdir, f0_artifact, None, context)
            checkpoint_path = outdir / checkpoint["checkpoint_file"]
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["fit"] = {"tampered": True}
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.RunnerError, "F0 fit must be null"):
                RUNNER._load_prediction_checkpoint(outdir, "F0", context)

    def test_provenance_never_labels_an_unverified_HEAD_as_verified(self):
        contract = REPO / "conf" / "m31_ordered_linear_preregistration.json"
        unverified = RUNNER._base_provenance("known-answer", contract)
        self.assertIsNone(unverified["git_commit"])
        self.assertEqual(unverified["git_commit_verification"], "NOT_DECLARED_UNVERIFIED")

        declared = "a" * 40
        verified = RUNNER._base_provenance(
            "known-answer", contract, verified_git_commit=declared,
            command=("python3", "runner.py", "known-answer"),
            container_digest="sha256:" + "b" * 64,
        )
        self.assertEqual(verified["git_commit"], declared)
        self.assertEqual(verified["git_commit_verification"], "HEAD_EXACT_RELEVANT_FILES_CLEAN_AND_TRACKED")
        self.assertEqual(verified["command"], ["python3", "runner.py", "known-answer"])
        self.assertEqual(verified["runtime"]["container_digest"], "sha256:" + "b" * 64)

        # A caller cannot authenticate an arbitrary but well-formed SHA as the
        # current repository state.
        with self.assertRaisesRegex(RUNNER.RunnerError, "HEAD .* differs"):
            RUNNER._verify_git_provenance("0" * 40, (RUNNER_PATH, CORE_PATH, contract))


class PilotProcessBoundaryTest(unittest.TestCase):
    @staticmethod
    def _artifact(arm):
        arrays = tuple(np.full((3, 2, 3), 1.0 / 3.0) for _ in range(2))
        for array in arrays:
            array.setflags(write=False)
        samples = ("S0", "S1")
        return RUNNER.PredictionArtifact(
            "root18", samples, arm, None, arrays,
            RUNNER._prediction_sha256("root18", samples, arrays),
        )

    @staticmethod
    def _fitted(arm):
        return _valid_fitted(arm)

    @staticmethod
    def _path_bundles(tmp):
        root = Path(tmp)
        train = RUNNER.TrainingPaths(*(
            root / name for name in (
                "r17.sites", "r17.target", "r17.tree", "r17.pools", "r17.truth",
                "r17.flare", "r17.audit",
            )
        ))
        evaluation = RUNNER.FeaturePaths(*(
            root / name for name in (
                "r18.sites", "r18.target", "r18.tree", "r18.pools", "r18.flare", "r18.audit",
            )
        ))
        return train, evaluation

    def test_fit_predict_pilot_is_one_direction_CLD_only_and_never_loads_eval_truth(self):
        contract_path = REPO / "conf" / "m31_ordered_linear_preregistration.json"
        contract = CORE.load_contract(contract_path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outdir = root / "producer"
            train_paths, evaluation_paths = self._path_bundles(tmp)
            args = SimpleNamespace(
                contract=contract_path, genetic_map=root / "map.tsv", outdir=outdir,
                resume=False, workers=1, container_digest="sha256:" + "a" * 64,
                invocation=("python3", "runner.py", "fit-predict"),
            )
            train_features = SimpleNamespace(name="root17", seed=20260817, samples=("S0", "S1"))
            evaluation_features = SimpleNamespace(name="root18", seed=20260818, samples=("S0", "S1"))
            loaded_truth_paths = []

            def fake_load_feature(name, *_args, **_kwargs):
                return train_features if name == "root17" else evaluation_features

            def fake_load_truth(paths, _features):
                loaded_truth_paths.append(paths)
                return object()

            def fake_prepare(_evaluation, _fitted, arm, _replicate):
                return self._artifact(arm)

            with (
                mock.patch.object(RUNNER, "_verify_code_contract_and_commit",
                                  return_value=(contract, "b" * 40,
                                                {"contract": "c", "runner": "r", "core": "k"})),
                mock.patch.object(RUNNER, "_pilot_paths_from_args",
                                  return_value=(train_paths, evaluation_paths)),
                mock.patch.object(RUNNER, "_pilot_input_hashes", return_value={"genetic_map": "g"}),
                mock.patch.object(RUNNER.core, "load_genetic_map", return_value=object()),
                mock.patch.object(RUNNER, "load_feature_root", side_effect=fake_load_feature),
                mock.patch.object(RUNNER, "load_truth_bundle", side_effect=fake_load_truth),
                mock.patch.object(RUNNER, "_threadpool_runtime", return_value=_mock_threadpool_runtime()),
                mock.patch.object(RUNNER, "fit_arm_streaming",
                                  side_effect=lambda _r, _t, arm, *_a, **_k: self._fitted(arm)) as fit,
                mock.patch.object(RUNNER, "prepare_predictions", side_effect=fake_prepare),
            ):
                result = RUNNER.fit_predict_pilot(args)

            self.assertEqual(result["status"], "COMPLETE_FSYNC_PROCESS_EXIT_REQUIRED")
            self.assertEqual(result["manifest_sha256"], CORE.sha256_file(Path(result["manifest"])))
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["label"], "NO_SCIENTIFIC_DECISION")
            self.assertEqual(manifest["context"]["train_root"], "root17")
            self.assertEqual(manifest["context"]["evaluation_root"], "root18")
            self.assertEqual(manifest["context"]["fitted_arms"], ["C", "L", "D"])
            self.assertEqual(manifest["context"]["workers"], 1)
            self.assertEqual(set(manifest["checkpoints"]), {"F0", "C", "L", "D"})
            self.assertNotIn("H", manifest["checkpoints"])
            self.assertEqual([call.args[2] for call in fit.call_args_list], ["C", "L", "D"])
            self.assertEqual([call.kwargs["workers"] for call in fit.call_args_list], [1, 1, 1])
            self.assertEqual(loaded_truth_paths, [train_paths])
            self.assertNotIn("truth", evaluation_paths.as_dict())
            provenance = json.loads((outdir / "pilot.fit_predict.provenance.json").read_text(encoding="utf-8"))
            self.assertFalse(provenance["evaluation_truth_was_accepted_or_read"])
            self.assertEqual(provenance["scientific_decision"], "NO_SCIENTIFIC_DECISION")
            self.assertEqual(provenance["workers"], 1)
            self.assertEqual(manifest["context"]["threadpool_runtime"], _mock_threadpool_runtime())
            self.assertEqual(provenance["threadpool_runtime"], _mock_threadpool_runtime())

    def test_fit_worker_failure_never_publishes_that_arm_checkpoint_or_manifest(self):
        contract_path = REPO / "conf" / "m31_ordered_linear_preregistration.json"
        contract = CORE.load_contract(contract_path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outdir = root / "producer"
            train_paths, evaluation_paths = self._path_bundles(tmp)
            args = SimpleNamespace(
                contract=contract_path, genetic_map=root / "map.tsv", outdir=outdir,
                resume=False, workers=4, container_digest="sha256:" + "a" * 64,
                invocation=("python3", "runner.py", "fit-predict"),
            )
            train_features = SimpleNamespace(name="root17", seed=20260817, samples=("S0", "S1"))
            evaluation_features = SimpleNamespace(name="root18", seed=20260818, samples=("S0", "S1"))

            def fake_load_feature(name, *_args, **_kwargs):
                return train_features if name == "root17" else evaluation_features

            with (
                mock.patch.dict(RUNNER.os.environ, {name: "1" for name in RUNNER.THREAD_LIMIT_ENV}),
                mock.patch.object(RUNNER, "_verify_code_contract_and_commit",
                                  return_value=(contract, "b" * 40,
                                                {"contract": "c", "runner": "r", "core": "k"})),
                mock.patch.object(RUNNER, "_pilot_paths_from_args",
                                  return_value=(train_paths, evaluation_paths)),
                mock.patch.object(RUNNER, "_pilot_input_hashes", return_value={"genetic_map": "g"}),
                mock.patch.object(RUNNER.core, "load_genetic_map", return_value=object()),
                mock.patch.object(RUNNER, "load_feature_root", side_effect=fake_load_feature),
                mock.patch.object(RUNNER, "load_truth_bundle", return_value=object()),
                mock.patch.object(RUNNER, "_threadpool_runtime", return_value=_mock_threadpool_runtime()),
                mock.patch.object(RUNNER, "fit_arm_streaming",
                                  side_effect=RUNNER.RunnerError("parallel individual worker failed: synthetic")),
                mock.patch.object(RUNNER, "prepare_predictions",
                                  side_effect=lambda *_args: self._artifact("F0")),
                self.assertRaisesRegex(RUNNER.RunnerError, "worker failed"),
            ):
                RUNNER.fit_predict_pilot(args)

            self.assertTrue((outdir / "pilot.F0.checkpoint.json").exists())
            self.assertFalse((outdir / "pilot.C.checkpoint.json").exists())
            self.assertFalse((outdir / "pilot.C.predictions.npy").exists())
            self.assertFalse((outdir / "pilot.fit_predict.manifest.json").exists())

    def _producer_fixture(self, directory, code_hashes, commit):
        context = {
            "code_sha256": code_hashes,
            "git_commit": commit,
            "input_sha256": {
                "genetic_map": "g",
                **{f"root18.{key}": key for key in RUNNER.FeaturePaths.__dataclass_fields__},
            },
        }
        context_hash = RUNNER._payload_sha256(context)
        checkpoints = {
            arm: RUNNER._write_prediction_checkpoint(
                directory, self._artifact(arm), None if arm == "F0" else self._fitted(arm), context_hash,
            )
            for arm in RUNNER.PILOT_OUTPUT_ARMS
        }
        manifest = {
            "status": "COMPLETE_FSYNC",
            "stage": "M31_PILOT_ROOT17_TO_ROOT18_FIT_PREDICT",
            "context": context,
            "context_sha256": context_hash,
            "checkpoints": checkpoints,
        }
        path = directory / "pilot.fit_predict.manifest.json"
        RUNNER._atomic_json_fsync(path, manifest)
        return path

    def test_score_pilot_accepts_truth_only_after_manifest_and_rejects_manifest_tamper(self):
        contract_path = REPO / "conf" / "m31_ordered_linear_preregistration.json"
        code_hashes = {"contract": "c", "runner": "r", "core": "k"}
        commit = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            producer_dir = root / "producer"
            producer_dir.mkdir()
            manifest_path = self._producer_fixture(producer_dir, code_hashes, commit)
            train_paths, evaluation_paths = self._path_bundles(tmp)
            del train_paths
            args = SimpleNamespace(
                contract=contract_path, genetic_map=root / "map.tsv",
                prediction_manifest=manifest_path,
                expected_prediction_manifest_sha256=CORE.sha256_file(manifest_path),
                eval_root18_truth=root / "r18.truth", outdir=root / "score",
                container_digest="sha256:" + "a" * 64,
                invocation=("python3", "runner.py", "score-pilot"),
                **{f"eval_root18_{key}": value for key, value in evaluation_paths.as_dict().items()},
            )
            score_hashes = {
                "genetic_map": "g", "root18.truth": "truth",
                **{f"root18.{key}": key for key in RUNNER.FeaturePaths.__dataclass_fields__},
            }
            features = SimpleNamespace(name="root18", samples=("S0", "S1"))
            truth = object()
            counts = [_score_counts("S0", total_cm=1, dose_error=0)]
            summary = RUNNER.summarize_counts(counts)

            common_patches = (
                mock.patch.object(RUNNER, "_verify_code_contract_and_commit",
                                  return_value=(object(), commit, code_hashes)),
                mock.patch.object(RUNNER, "_authenticate_exact_subset", return_value=score_hashes),
                mock.patch.object(RUNNER.core, "load_genetic_map", return_value=object()),
                mock.patch.object(RUNNER, "load_feature_root", return_value=features),
                mock.patch.object(RUNNER, "load_truth_bundle", return_value=truth),
                mock.patch.object(RUNNER, "score_prediction_artifact",
                                  return_value=(summary, [{"sample_id": "S0"}], counts)),
                mock.patch.object(RUNNER, "bootstrap_counts", return_value={"status": "TEST"}),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
                    common_patches[4] as truth_loader, common_patches[5], common_patches[6]:
                result = RUNNER.score_pilot(args)
            self.assertEqual(result["status"], "COMPLETE_NO_SCIENTIFIC_DECISION")
            truth_loader.assert_called_once()
            scored_summary = json.loads((args.outdir / RUNNER.OUTPUT_NAMES["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(scored_summary["label"], "NO_SCIENTIFIC_DECISION")
            self.assertEqual(scored_summary["arms"], ["F0", "C", "L", "D"])
            self.assertEqual(scored_summary["shams"], 0)
            self.assertFalse(scored_summary["H_included"])

            args.outdir = root / "bad-score"
            args.expected_prediction_manifest_sha256 = "0" * 64
            with (
                mock.patch.object(RUNNER, "_verify_code_contract_and_commit",
                                  return_value=(object(), commit, code_hashes)),
                mock.patch.object(RUNNER, "load_truth_bundle") as forbidden_truth,
                self.assertRaisesRegex(RUNNER.RunnerError, "manifest SHA-256"),
            ):
                RUNNER.score_pilot(args)
            forbidden_truth.assert_not_called()

            producer = json.loads(manifest_path.read_text(encoding="utf-8"))
            producer["checkpoints"]["D"]["checkpoint_file_sha256"] = "f" * 64
            RUNNER._atomic_json_fsync(manifest_path, producer)
            args.expected_prediction_manifest_sha256 = CORE.sha256_file(manifest_path)
            with (
                mock.patch.object(RUNNER, "_verify_code_contract_and_commit",
                                  return_value=(object(), commit, code_hashes)),
                mock.patch.object(RUNNER, "_authenticate_exact_subset", return_value=score_hashes),
                mock.patch.object(RUNNER.core, "load_genetic_map", return_value=object()),
                mock.patch.object(RUNNER, "load_feature_root", return_value=features),
                mock.patch.object(RUNNER, "load_truth_bundle", return_value=truth),
                mock.patch.object(RUNNER, "score_prediction_artifact",
                                  return_value=(summary, [{"sample_id": "S0"}], counts)),
                mock.patch.object(RUNNER, "bootstrap_counts", return_value={"status": "TEST"}),
                self.assertRaisesRegex(RUNNER.RunnerError, "manifest/checkpoint SHA-256"),
            ):
                RUNNER.score_pilot(args)

    def test_real_sample_benchmark_materializes_CLDH_without_truth_or_fit(self):
        contract_path = REPO / "conf" / "m31_ordered_linear_preregistration.json"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _train, feature_paths = self._path_bundles(tmp)
            args = SimpleNamespace(
                contract=contract_path, genetic_map=root / "map.tsv", root="root18",
                sample_id=None, output=root / "benchmark.json",
                container_digest="sha256:" + "a" * 64,
                invocation=("python3", "runner.py", "benchmark-sample"),
                **{f"benchmark_{key}": value for key, value in feature_paths.as_dict().items()},
            )
            dimensions = {"C": 59, "L": 99, "D": 171, "H": 243}

            class Features:
                samples = ("S0", "S1")

                @staticmethod
                def features(_sample_index, arm, _replicate):
                    return np.zeros((8, dimensions[arm]), dtype=np.float32)

            with (
                mock.patch.object(RUNNER, "_verify_code_contract_and_commit",
                                  return_value=(object(), "b" * 40,
                                                {"contract": "c", "runner": "r", "core": "k"})),
                mock.patch.object(RUNNER, "_authenticate_exact_subset", return_value={"genetic_map": "g"}),
                mock.patch.object(RUNNER.core, "load_genetic_map", return_value=object()),
                mock.patch.object(RUNNER, "load_feature_root", return_value=Features()),
                mock.patch.object(RUNNER, "load_truth_bundle") as forbidden_truth,
                mock.patch.object(RUNNER, "fit_arm_streaming") as forbidden_fit,
            ):
                report = RUNNER.benchmark_real_sample(args)
            forbidden_truth.assert_not_called()
            forbidden_fit.assert_not_called()
            self.assertFalse(report["truth_accessed"])
            self.assertFalse(report["fit_performed"])
            self.assertEqual([row["arm"] for row in report["arms"]], ["C", "L", "D", "H"])
            self.assertEqual(
                {row["arm"]: row["feature_count"] for row in report["arms"]}, dimensions,
            )
            self.assertEqual(json.loads(args.output.read_text(encoding="utf-8"))["status"],
                             "COMPLETE_TRUTH_BLIND_NO_FIT")
            forbidden_truth.assert_not_called()


class Pre2MetricsAndDecisionTest(unittest.TestCase):
    @staticmethod
    def _metrics(f1: float, error: float, ft: float = 0.10):
        metrics = {
            "boundary_f1_0.2cM": f1,
            "macro_ancestry_dose_mae": error,
            "false_transitions_per_cM_0.2cM": ft,
        }
        for ancestry in CORE.ANCESTRIES:
            metrics[f"ancestry_dose_mae_{ancestry}"] = error
            metrics[f"ancestry_dose_mae_truth_present_{ancestry}"] = error
        return metrics

    @staticmethod
    def _technical(**overrides):
        requirements = {name: True for name in RUNNER.PRE2_TECHNICAL_REQUIREMENTS}
        requirements.update(overrides)
        return requirements

    def test_truth_present_mae_conditions_on_diploid_truth_presence(self):
        genetic_map = CORE.GeneticMap(
            np.array([100, 200], dtype=np.int64), np.array([0.0, 1.0], dtype=float),
        )
        root = SimpleNamespace(
            name="root18", samples=("S0",), marker_positions=np.array([150]),
            cell_left_bp=np.array([100]), cell_right_bp=np.array([200]), genetic_map=genetic_map,
        )
        truth = RUNNER.TruthBundle(
            "root18",
            {"S0": (
                [CORE.TruthSegment(100, 201, "AFR")],
                [CORE.TruthSegment(100, 201, "EUR")],
            )},
            np.zeros((1, 1, 2, 3), dtype=float),
            (np.zeros(2, dtype=bool),),
        )
        predicted = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        summary, counts = RUNNER.score_sample(root, truth, 0, predicted)
        self.assertAlmostEqual(summary["ancestry_dose_mae_truth_present_AFR"], 0.5)
        self.assertAlmostEqual(summary["ancestry_dose_mae_truth_present_EUR"], 0.5)
        self.assertIsNone(summary["ancestry_dose_mae_truth_present_ASIA"])
        np.testing.assert_allclose(counts.truth_present_cm_denominator, [1.0, 1.0, 0.0])
        for ancestry in CORE.ANCESTRIES:
            self.assertIsNone(summary[f"boundary_f1_0.2cM_{ancestry}"])
            self.assertEqual(summary[f"boundary_truth_count_0.2cM_{ancestry}"], 0)

    def test_truth_present_counts_fail_closed_when_partial_or_invalid(self):
        base = _score_counts("S0", total_cm=1.0, dose_error=0.1)
        partial = replace(
            base, truth_present_mae_numerator=np.ones(3),
            truth_present_cm_denominator=None,
        )
        with self.assertRaisesRegex(RUNNER.RunnerError, "pairing differs"):
            RUNNER.summarize_counts([partial])
        negative = replace(
            base, truth_present_mae_numerator=np.ones(3),
            truth_present_cm_denominator=np.array([1.0, -1.0, 1.0]),
        )
        with self.assertRaisesRegex(RUNNER.RunnerError, "negative"):
            RUNNER.summarize_counts([negative])

    def test_ancestry_boundary_metrics_attribute_true_and_false_transitions(self):
        genetic_map = CORE.GeneticMap(
            np.array([100, 300], dtype=np.int64), np.array([0.0, 2.0], dtype=float),
        )
        root = SimpleNamespace(
            name="root18", samples=("S0",), marker_positions=np.array([150, 250]),
            cell_left_bp=np.array([100, 200]), cell_right_bp=np.array([200, 300]),
            genetic_map=genetic_map,
        )
        truth = RUNNER.TruthBundle(
            "root18",
            {"S0": (
                [CORE.TruthSegment(100, 200, "AFR"), CORE.TruthSegment(200, 301, "EUR")],
                [CORE.TruthSegment(100, 301, "AFR")],
            )},
            np.zeros((2, 1, 2, 3), dtype=float),
            (np.zeros(4, dtype=bool),),
        )
        predicted = np.array([
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ])
        summary, _counts = RUNNER.score_sample(root, truth, 0, predicted)
        self.assertAlmostEqual(summary["boundary_f1_0.2cM_AFR"], 2.0 / 3.0)
        self.assertEqual(summary["boundary_f1_0.2cM_EUR"], 1.0)
        self.assertEqual(summary["boundary_f1_0.2cM_ASIA"], 0.0)
        self.assertEqual(summary["boundary_truth_count_0.2cM_AFR"], 1)
        self.assertEqual(summary["boundary_prediction_count_0.2cM_AFR"], 2)
        self.assertEqual(summary["boundary_matched_count_0.2cM_AFR"], 1)
        self.assertAlmostEqual(summary["false_transitions_per_cM_0.2cM_AFR"], 1.0 / 3.0)
        self.assertIsNone(summary["false_transitions_per_cM_0.2cM_ASIA"])

    def test_root17_gate_uses_primary_delta_and_reports_sensitivities(self):
        metrics = {
            "F0": self._metrics(0.70, 0.20),
            "L": self._metrics(0.70, 0.21),
            "D": self._metrics(0.711, 0.19),
        }
        decision = RUNNER.evaluate_pre2_root17_gate(
            metrics, d_guarded=True, l_guarded=False,
            technical_requirements=self._technical(),
        )
        self.assertEqual(decision["status"], "OPEN_ROOT18")
        self.assertEqual(decision["candidate_scope"], "CANDIDATE_RARE_COMBINED_ONLY")
        self.assertEqual(
            decision["delta_f1_sensitivity_pass"],
            {"0.005": True, "0.010": True, "0.020": False},
        )

    def test_root17_gate_fails_closed_on_technical_or_ancestry_guard(self):
        metrics = {
            "F0": self._metrics(0.70, 0.20),
            "L": self._metrics(0.70, 0.21),
            "D": self._metrics(0.72, 0.19),
        }
        metrics["D"]["ancestry_dose_mae_truth_present_ASIA"] = 0.21
        decision = RUNNER.evaluate_pre2_root17_gate(
            metrics, d_guarded=True, l_guarded=False,
            technical_requirements=self._technical(truth_blind_prediction_manifest_fsynced=False),
        )
        self.assertEqual(decision["status"], "STOP_PRE2_BEFORE_ROOT18")
        self.assertFalse(all(item["pass"] for item in decision["technical_checks"]))
        self.assertFalse(all(item["pass"] for item in decision["scientific_checks"]))

        with self.assertRaisesRegex(RUNNER.RunnerError, "requirement set drifted"):
            RUNNER.evaluate_pre2_root17_gate(
                metrics, d_guarded=True, l_guarded=False,
                technical_requirements={"known_answers_pass": True},
            )

    def test_root17_rechecks_guarded_flag_and_frozen_parameters(self):
        metrics = {
            "F0": self._metrics(0.70, 0.20, 0.10),
            "L": self._metrics(0.70, 0.21, 0.10),
            "D": self._metrics(0.72, 0.21, 0.10),
        }
        decision = RUNNER.evaluate_pre2_root17_gate(
            metrics, d_guarded=True, l_guarded=False,
            technical_requirements=self._technical(),
        )
        self.assertEqual(decision["status"], "STOP_PRE2_BEFORE_ROOT18")
        for name, value in (("delta", 0.005), ("tau", 0.0)):
            kwargs = {name: value}
            with self.subTest(name=name), self.assertRaisesRegex(RUNNER.RunnerError, "drifted"):
                RUNNER.evaluate_pre2_root17_gate(
                    metrics, d_guarded=True, l_guarded=False,
                    technical_requirements=self._technical(), **kwargs,
                )

    def test_root17_cannot_hide_a_guarded_L(self):
        metrics = {
            "F0": self._metrics(0.70, 0.20, 0.10),
            "L": self._metrics(0.705, 0.19, 0.09),
            "D": self._metrics(0.72, 0.18, 0.08),
        }
        decision = RUNNER.evaluate_pre2_root17_gate(
            metrics, d_guarded=True, l_guarded=False,
            technical_requirements=self._technical(),
        )
        self.assertEqual(decision["status"], "STOP_PRE2_BEFORE_ROOT18")

        for value in (0, None):
            with self.subTest(value=value), self.assertRaisesRegex(
                RUNNER.RunnerError, "must be booleans"
            ):
                RUNNER.evaluate_pre2_root17_gate(
                    metrics, d_guarded=True, l_guarded=value,
                    technical_requirements=self._technical(),
                )

    def test_f1_comparison_uses_only_frozen_computational_tolerance(self):
        metrics = {
            "F0": self._metrics(0.70, 0.20),
            "D": self._metrics(0.71 - 0.5e-15, 0.19),
        }
        within_tau = RUNNER.evaluate_pre2_root18_decision(metrics, l_guarded=False)
        self.assertEqual(within_tau["status"], "CANDIDATE_RARE_COMBINED_ONLY_VS_F0")
        metrics["D"]["boundary_f1_0.2cM"] = 0.71 - 2.0e-15
        beyond_tau = RUNNER.evaluate_pre2_root18_decision(metrics, l_guarded=False)
        self.assertEqual(beyond_tau["status"], "STOP_PRE2_ON_THESE_ROOTS")

    def test_root18_requires_D_to_pass_against_guarded_L_and_F0(self):
        metrics = {
            "F0": self._metrics(0.70, 0.20, 0.12),
            "L": self._metrics(0.705, 0.19, 0.11),
            "D": self._metrics(0.716, 0.18, 0.10),
        }
        passed = RUNNER.evaluate_pre2_root18_decision(metrics, l_guarded=True)
        self.assertEqual(passed["status"], "CANDIDATE_D_FOR_NEW_PROSPECTIVE_ROOTS")
        metrics["D"]["macro_ancestry_dose_mae"] = 0.195
        failed = RUNNER.evaluate_pre2_root18_decision(metrics, l_guarded=True)
        self.assertEqual(failed["status"], "STOP_PRE2_ON_THESE_ROOTS")

    def test_root18_unguarded_L_is_not_a_comparator(self):
        metrics = {
            "F0": self._metrics(0.70, 0.20),
            "D": self._metrics(0.711, 0.19),
        }
        decision = RUNNER.evaluate_pre2_root18_decision(metrics, l_guarded=False)
        self.assertEqual(decision["applicable_comparators"], ["F0"])
        self.assertEqual(decision["status"], "CANDIDATE_RARE_COMBINED_ONLY_VS_F0")

    def test_gate_rejects_out_of_domain_metrics(self):
        for name, value in (
            ("boundary_f1_0.2cM", 1.01),
            ("macro_ancestry_dose_mae", -0.01),
            ("false_transitions_per_cM_0.2cM", -0.01),
        ):
            metrics = {
                "F0": self._metrics(0.70, 0.20),
                "D": self._metrics(0.72, 0.19),
            }
            metrics["D"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                RUNNER.RunnerError, "outside|negative"
            ):
                RUNNER.evaluate_pre2_root18_decision(metrics, l_guarded=False)


class Pre2GateReceiptTest(unittest.TestCase):
    @staticmethod
    def _metrics(f1: float, error: float, ft: float = 0.10):
        payload = {
            "boundary_f1_0.2cM": f1,
            "macro_ancestry_dose_mae": error,
            "false_transitions_per_cM_0.2cM": ft,
        }
        for ancestry in CORE.ANCESTRIES:
            payload[f"ancestry_dose_mae_{ancestry}"] = error
            payload[f"ancestry_dose_mae_truth_present_{ancestry}"] = error
        return payload

    @staticmethod
    def _binding():
        return {
            "contract_sha256": "a" * 64,
            "git_commit": "b" * 40,
            "runner_sha256": "c" * 64,
            "core_sha256": "d" * 64,
            "container_digest": "sha256:" + "e" * 64,
            "prediction_manifest_sha256": "f" * 64,
            "context_sha256": "1" * 64,
            "root17_metrics_sha256": "2" * 64,
            "technical_evidence_sha256": "3" * 64,
            "worker_screen_sha256": "9" * 64,
            "execution_authorization_sha256": "0" * 64,
            "contract_code_sha256": "9" * 64,
            "receipt_code_sha256": "4" * 64,
            "orchestrator_sha256": "5" * 64,
            "module_sha256": "6" * 64,
            "workflow_sha256": "7" * 64,
            "config_sha256": "8" * 64,
        }

    def _checkpoint_fits(self, root: Path):
        fits = {}
        for arm in ("L", "D"):
            artifact = RUNNER.PredictionArtifact(
                "root18", ("S0",), arm, None,
                (np.zeros((2, 2, 3), dtype=float),), "",
            )
            artifact = replace(
                artifact,
                sha256=RUNNER._prediction_sha256(
                    artifact.root_name, artifact.sample_ids, artifact.arrays,
                ),
            )
            fitted = _valid_fitted(arm)
            selected_f1 = 0.705 if arm == "L" else 0.72
            fitted.cv_boundary_f1 = selected_f1
            fitted.candidate_table = tuple(
                {**row, "boundary_f1_0.2cM": selected_f1}
                if row["selected"] else row
                for row in fitted.candidate_table
            )
            checkpoint = RUNNER._write_prediction_checkpoint(root, artifact, fitted, "2" * 64)
            fits[arm] = checkpoint["fit"]
        return fits

    @staticmethod
    def _claims():
        contract = json.loads(
            (REPO / "conf" / "m31_ordered_linear_pre2_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        return contract["claims_excluded"]

    def test_receipt_reconstructs_exactly_and_binds_open_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_fits = self._checkpoint_fits(Path(tmp))
            metrics = {
                "F0": self._metrics(0.50, 0.30, 0.20),
                "L": self._metrics(0.705, 0.20, 0.10),
                "D": self._metrics(0.72, 0.20, 0.10),
            }
            technical = {name: True for name in RUNNER.PRE2_TECHNICAL_REQUIREMENTS}
            claims = self._claims()
            receipt = PRE2_RECEIPT.build_root17_gate_receipt(
                metrics=metrics, checkpoint_fits=checkpoint_fits,
                technical_requirements=technical, binding=self._binding(),
                claims_excluded=claims,
            )
            self.assertEqual(receipt["decision"]["status"], "OPEN_ROOT18")
            with self.assertRaisesRegex(PRE2_RECEIPT.ReceiptError, "frozen contract"):
                PRE2_RECEIPT.build_root17_gate_receipt(
                    metrics=metrics, checkpoint_fits=checkpoint_fits,
                    technical_requirements=technical, binding=self._binding(),
                    claims_excluded=claims[:-1],
                )
            rebuilt = PRE2_RECEIPT.validate_root17_gate_receipt(
                receipt, expected_binding=self._binding(), checkpoint_fits=checkpoint_fits,
                expected_metrics=metrics, expected_technical_requirements=technical,
                expected_claims_excluded=claims,
            )
            self.assertEqual(rebuilt, receipt)

            tampered = json.loads(json.dumps(receipt))
            tampered["root17_metrics"]["D"]["boundary_f1_0.2cM"] = 0.99
            with self.assertRaisesRegex(PRE2_RECEIPT.ReceiptError, "semantic SHA-256 mismatch"):
                PRE2_RECEIPT.validate_root17_gate_receipt(
                    tampered, expected_binding=self._binding(), checkpoint_fits=checkpoint_fits,
                    expected_metrics=metrics, expected_technical_requirements=technical,
                    expected_claims_excluded=claims,
                )

            resigned_metrics = json.loads(json.dumps(metrics))
            resigned_metrics["D"]["ancestry_dose_mae_truth_present_AFR"] = 0.01
            resigned = PRE2_RECEIPT.build_root17_gate_receipt(
                metrics=resigned_metrics, checkpoint_fits=checkpoint_fits,
                technical_requirements=technical, binding=self._binding(),
                claims_excluded=claims,
            )
            with self.assertRaisesRegex(PRE2_RECEIPT.ReceiptError, "reconstruct exactly"):
                PRE2_RECEIPT.validate_root17_gate_receipt(
                    resigned, expected_binding=self._binding(), checkpoint_fits=checkpoint_fits,
                    expected_metrics=metrics, expected_technical_requirements=technical,
                    expected_claims_excluded=claims,
                )

    def test_stop_receipt_cannot_open_root18(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_fits = self._checkpoint_fits(Path(tmp))
            metrics = {
                "F0": self._metrics(0.50, 0.30, 0.20),
                "L": self._metrics(0.705, 0.20, 0.10),
                "D": self._metrics(0.72, 0.20, 0.10),
            }
            technical = {name: True for name in RUNNER.PRE2_TECHNICAL_REQUIREMENTS}
            technical["workers_1_4_8_exact_equality_pass"] = False
            claims = self._claims()
            receipt = PRE2_RECEIPT.build_root17_gate_receipt(
                metrics=metrics, checkpoint_fits=checkpoint_fits,
                technical_requirements=technical, binding=self._binding(),
                claims_excluded=claims,
            )
            self.assertEqual(receipt["decision"]["status"], "STOP_PRE2_BEFORE_ROOT18")
            with self.assertRaisesRegex(PRE2_RECEIPT.ReceiptError, "does not authorize"):
                PRE2_RECEIPT.validate_root17_gate_receipt(
                    receipt, expected_binding=self._binding(), checkpoint_fits=checkpoint_fits,
                    expected_metrics=metrics, expected_technical_requirements=technical,
                    expected_claims_excluded=claims,
                )


if __name__ == "__main__":
    unittest.main()
