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

    def test_real_D_or_H_must_be_guarded_but_unguarded_sham_stays_conservative(self):
        metrics = self._metrics("GO_NEW_ROOTS")
        for row in metrics:
            if row["arm"] == "D" and row["sham_replicate"] == "":
                row["inner_cv_guarded"] = False
        # An unguarded real D cannot claim GO; the already-improving L remains
        # the narrow descriptive outcome.
        self.assertEqual(RUNNER.decide(metrics)["label"], "LOAD_ONLY")

        metrics = self._metrics("GO_NEW_ROOTS")
        for row in metrics:
            if row["arm"] == "DSHAM" and row["sham_replicate"] == 7:
                row["inner_cv_guarded"] = False
                row["boundary_f1_0.2cM"] = 0.90
        # The poor sham fit is not dropped post hoc: its larger null gain blocks
        # D and therefore leaves only the load result.
        self.assertEqual(RUNNER.decide(metrics)["label"], "LOAD_ONLY")

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
                checkpoint = RUNNER._write_prediction_checkpoint(outdir, artifact, None, context)
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
        return SimpleNamespace(
            arm=arm, alpha=1.0, boundary_weight=5.0,
            cv_boundary_f1=0.7, cv_false_transitions_per_cm=0.1,
            cv_macro_ancestry_dose_mae=0.2, cv_brier=0.3,
            guarded=True, selection_status="GUARDED_CONFIG", feature_count=59,
        )

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
                resume=False, container_digest="sha256:" + "a" * 64,
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
            self.assertEqual(set(manifest["checkpoints"]), {"F0", "C", "L", "D"})
            self.assertNotIn("H", manifest["checkpoints"])
            self.assertEqual([call.args[2] for call in fit.call_args_list], ["C", "L", "D"])
            self.assertEqual(loaded_truth_paths, [train_paths])
            self.assertNotIn("truth", evaluation_paths.as_dict())
            provenance = json.loads((outdir / "pilot.fit_predict.provenance.json").read_text(encoding="utf-8"))
            self.assertFalse(provenance["evaluation_truth_was_accepted_or_read"])
            self.assertEqual(provenance["scientific_decision"], "NO_SCIENTIFIC_DECISION")

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


if __name__ == "__main__":
    unittest.main()
