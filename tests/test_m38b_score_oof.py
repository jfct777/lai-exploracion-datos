#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m38b_score_oof as subject


class M38BScoringTests(unittest.TestCase):
    @staticmethod
    def probabilities_from_truth(truth: np.ndarray, confidence: float = 0.9) -> np.ndarray:
        values = np.full(truth.shape + (6,), (1.0 - confidence) / 5.0, dtype=np.float64)
        rows, columns = np.indices(truth.shape)
        values[rows, columns, truth] = confidence
        return values

    @staticmethod
    def write_inputs(
        directory: Path,
        truth: np.ndarray,
        predictions: dict[str, np.ndarray],
        marker_cm: np.ndarray | None = None,
        dosage_truth: bool = False,
    ) -> tuple[Path, dict[str, Path]]:
        people, markers = truth.shape
        person_ids = np.asarray([f"person-{index}" for index in range(people)], dtype="S32")
        folds = np.asarray([index % 3 for index in range(people)], dtype=np.int8)
        cm = (np.arange(markers, dtype=np.float64) * 0.1
              if marker_cm is None else np.asarray(marker_cm, dtype=np.float64))
        positions = np.arange(10_000, 10_000 + markers, dtype=np.int64)
        truth_path = directory / "truth.npz"
        truth_payload = {
            "truth_dosage": subject.STATE_DOSAGE[truth].astype(np.int8),
        } if dosage_truth else {"state_labels": truth.astype(np.int8)}
        np.savez(
            truth_path, person_ids=person_ids, fold_ids=folds,
            marker_cM=cm, marker_pos=positions, **truth_payload,
        )
        paths: dict[str, Path] = {}
        for arm, probability in predictions.items():
            path = directory / f"{arm}.npz"
            np.savez(
                path, arm=np.asarray([arm]), person_ids=person_ids,
                fold_ids=folds, marker_cM=cm, marker_pos=positions,
                state_names=np.asarray(subject.STATE_NAMES),
                probabilities=probability,
            )
            paths[arm] = path
        return truth_path, paths

    def test_probability_floor_is_followed_by_renormalisation(self) -> None:
        probability = np.zeros((1, 2, 6), dtype=np.float64)
        probability[0, 0, 0] = 1.0
        probability[0, 1] = np.asarray([0.2, 0.3, 0.5, 0.0, 0.0, 0.0])
        observed = subject.normalise_probabilities(probability)
        self.assertTrue(np.all(observed > 0))
        self.assertTrue(np.allclose(observed.sum(axis=2), 1.0, atol=1e-15, rtol=0))
        self.assertLess(observed[0, 0, 0], 1.0)
        with self.assertRaisesRegex(ValueError, "simplex"):
            subject.normalise_probabilities(np.ones((1, 1, 6)))

    def test_voronoi_cm_weighting_differs_from_uniform_marker_weighting(self) -> None:
        cm = np.asarray([0.0, 1.0, 10.0])
        self.assertTrue(np.allclose(subject.normalised_voronoi_cm_weights(cm), [.05, .50, .45]))
        truth = np.zeros((1, 3), dtype=np.int8)
        probability = np.full((1, 3, 6), 0.02, dtype=np.float64)
        probability[:, :, 0] = 0.9
        probability[0, 2, 0] = 0.1
        probability[0, 2, 1:] = 0.18
        score = subject.score_arm(probability, truth, cm)
        self.assertGreater(score.summary["log_loss_cm"], score.summary["log_loss_uniform"])
        expected = float(np.dot(-np.log([0.9, 0.9, 0.1]), [.05, .50, .45]))
        self.assertAlmostEqual(score.summary["log_loss_cm"], expected, places=10)

    def test_boundary_matching_is_directed_one_to_one_and_exactly_point_two_cm(self) -> None:
        truth = [(0.1, 0, 1), (0.5, 1, 2)]
        predicted = [(0.3, 0, 1), (0.3, 0, 1), (0.5, 2, 1)]
        matched, cost = subject.directed_boundary_match(truth, predicted, 0.2)
        self.assertEqual(matched, 1)
        self.assertAlmostEqual(cost, 0.2)
        with self.assertRaisesRegex(ValueError, "exactly 0.2"):
            subject.directed_boundary_match(truth, predicted, 0.2001)
        # A nearest-first greedy matcher would consume 0.05 for the first
        # truth boundary and lose the second match.  The ordered optimum keeps
        # both pairs.
        optimal, _ = subject.directed_boundary_match(
            [(0.0, 0, 1), (0.18, 0, 1)],
            [(-0.15, 0, 1), (0.05, 0, 1)],
            0.2,
        )
        self.assertEqual(optimal, 2)

    def test_boundary_metrics_use_point_two_for_f1_error_and_false_transitions(self) -> None:
        cm = np.asarray([0.0, 0.2, 0.4, 0.6, 1.0])
        truth = np.asarray([[0, 1, 1, 1, 1]], dtype=np.int8)  # boundary at 0.1
        hard = np.asarray([[0, 0, 1, 2, 2]], dtype=np.int8)   # 0->1 at 0.3, extra 1->2
        probability = np.eye(6, dtype=np.float64)[hard]
        score = subject.score_arm(probability, truth, cm)
        boundary = score.summary["boundary_0_2cm"]
        self.assertEqual(boundary["matched_one_to_one_directed"], 1)
        self.assertAlmostEqual(boundary["mean_error_cm"], 0.2)
        self.assertEqual(boundary["predicted"], 2)
        self.assertGreater(boundary["false_transitions_per_morgan"], 0)
        self.assertAlmostEqual(boundary["f1_micro"], 2 / 3)

    def test_metrics_remain_separate_and_diploid_dosage_has_expected_scale(self) -> None:
        cm = np.asarray([0.0, 0.1, 0.4])
        truth = np.asarray([[0, 2, 5], [3, 4, 0]], dtype=np.int8)
        probability = self.probabilities_from_truth(truth, confidence=0.7)
        score = subject.score_arm(probability, truth, cm)
        self.assertEqual(set(subject.PER_PERSON_METRICS),
                         {name for name in score.per_person if not name.startswith("_")})
        self.assertNotEqual(score.summary["log_loss_cm"], score.summary["brier_cm"])
        self.assertNotEqual(score.summary["ancestry_proportion_mae_macro_cm"], score.summary["ancestry_proportion_mae_nam_cm"])
        self.assertGreaterEqual(score.summary["ancestry_proportion_mae_nam_cm"], 0.0)
        self.assertLessEqual(score.summary["ancestry_proportion_mae_nam_cm"], 1.0)

    def test_bootstrap_resamples_whole_people_within_fold_and_reuses_indices(self) -> None:
        folds = np.asarray([0, 0, 1, 1, 2, 2])
        first = subject.stratified_person_bootstrap_indices(folds, 200, 91)
        second = subject.stratified_person_bootstrap_indices(folds, 200, 91)
        self.assertTrue(np.array_equal(first, second))
        for row in first:
            self.assertEqual(sum(folds[index] == 0 for index in row), 2)
            self.assertEqual(sum(folds[index] == 1 for index in row), 2)
            self.assertEqual(sum(folds[index] == 2 for index in row), 2)
        deltas = np.column_stack((np.arange(6, dtype=float), 2 * np.arange(6, dtype=float)))
        inference, boot, indices = subject.bootstrap_primary_contrasts(deltas, folds, 200, 91)
        self.assertTrue(np.allclose(boot[:, 1], 2 * boot[:, 0]))
        observed = deltas.mean(axis=0)
        upper = observed + np.quantile(observed[None, :] - boot, 0.975, axis=0)
        self.assertTrue(np.allclose(
            inference["bonferroni_two_candidate_upper_97_5"], upper,
        ))
        self.assertEqual(indices.shape, (200, 6))

    def test_boundary_f1_bootstrap_sums_tp_fp_fn_instead_of_averaging_people(self) -> None:
        truth_count = np.asarray([100, 1, 100, 1, 100, 1])
        left_matched = np.asarray([100, 0, 100, 0, 100, 0])
        right_matched = np.asarray([90, 1, 90, 1, 90, 1])
        empty = {name: np.zeros(6) for name in subject.PER_PERSON_METRICS}
        scored = {
            "left": subject.ArmScore({}, empty, {
                "truth": truth_count, "predicted": truth_count,
                "matched": left_matched,
            }),
            "right": subject.ArmScore({}, empty, {
                "truth": truth_count, "predicted": truth_count,
                "matched": right_matched,
            }),
        }
        indices = np.tile(np.arange(6), (100, 1))
        observed, bootstrap = subject.bootstrap_boundary_f1_contrasts(
            scored, (("left-right", "left", "right"),), indices,
        )
        per_person_delta = np.mean(left_matched / truth_count - right_matched / truth_count)
        self.assertLess(per_person_delta, 0.0)
        self.assertGreater(observed[0], 0.0)
        self.assertTrue(np.allclose(bootstrap[:, 0], observed[0]))

    def test_joint_inference_uses_three_fold_direction_and_no_fold_ci(self) -> None:
        truth = np.tile(np.asarray([[0, 2, 1, 2]], dtype=np.int8), (6, 1))
        better = self.probabilities_from_truth(truth, 0.95)
        worse = self.probabilities_from_truth(truth, 0.60)
        with tempfile.TemporaryDirectory() as raw:
            truth_path, paths = self.write_inputs(
                Path(raw), truth,
                {"full": better, "minus": worse, "RE": better, "RD": worse, "SHAM": worse},
                dosage_truth=True,
            )
            result, arrays = subject.analyse_files(
                paths, truth_path,
                (("full-minus", "full", "minus"),
                 ("RE-RD", "RE", "RD"),
                 ("RE-SHAM", "RE", "SHAM"),
                 ("RE-full", "RE", "full")),
                bootstrap_replicates=200, bootstrap_seed=7,
            )
        self.assertTrue(result["contrasts"]["RE-RD"]["direction_3_of_3"])
        self.assertTrue(result["contrasts"]["RE-RD"]["candidate_contrast_gate"])
        self.assertIsNone(result["contrasts"]["RE-full"]["candidate_contrast_gate"])
        self.assertTrue(result["candidate_incremental_gate"]["pass"])
        self.assertFalse(result["bootstrap"]["per_fold_confidence_intervals_used"])
        self.assertEqual(result["truth_usage"],
                         "evaluation_only; no model selection or checkpoint choice")
        self.assertEqual(arrays["bootstrap_primary_deltas"].shape, (200, 4))
        self.assertEqual(arrays["bootstrap_boundary_f1_deltas"].shape, (200, 4))
        self.assertEqual(arrays["bootstrap_person_indices"].shape, (200, 6))
        self.assertIn("brier_cm",
                      result["contrasts"]["RE-RD"]["metric_deltas_left_minus_right"])

    def test_axes_arm_and_probability_contracts_fail_closed(self) -> None:
        truth = np.tile(np.asarray([[0, 2, 0]], dtype=np.int8), (6, 1))
        probability = self.probabilities_from_truth(truth)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth_path, paths = self.write_inputs(root, truth, {"full": probability, "minus": probability})
            with np.load(paths["minus"], allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            payload["marker_cM"] = payload["marker_cM"] + np.asarray([0.0, 0.0, 0.01])
            np.savez(paths["minus"], **payload)
            with self.assertRaisesRegex(ValueError, "marker_cM axis differs"):
                subject.analyse_files(
                    paths, truth_path, (("full-minus", "full", "minus"),),
                    bootstrap_replicates=100,
                )
            with self.assertRaisesRegex(ValueError, "expected 96 people"):
                subject.analyse_files(
                    {"full": paths["full"]}, truth_path,
                    (("self", "full", "full"),), bootstrap_replicates=100,
                    expected_person_count=96,
                )

    def test_truth_poisoning_is_an_evaluation_change_not_a_selection_path(self) -> None:
        truth = np.tile(np.asarray([[0, 2, 0]], dtype=np.int8), (6, 1))
        probability = self.probabilities_from_truth(truth, 0.9)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth_path, paths = self.write_inputs(root, truth, {"full": probability, "minus": probability})
            first, _ = subject.analyse_files(
                paths, truth_path, (("full-minus", "full", "minus"),),
                bootstrap_replicates=100, bootstrap_seed=3,
            )
            with np.load(truth_path, allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            payload["state_labels"] = np.full_like(payload["state_labels"], 5)
            np.savez(truth_path, **payload)
            second, _ = subject.analyse_files(
                paths, truth_path, (("full-minus", "full", "minus"),),
                bootstrap_replicates=100, bootstrap_seed=3,
            )
        self.assertEqual(first["inputs_sha256"]["predictions"],
                         second["inputs_sha256"]["predictions"])
        self.assertNotEqual(first["arm_metrics"]["full"]["log_loss_cm"],
                            second["arm_metrics"]["full"]["log_loss_cm"])
        self.assertEqual(second["truth_usage"],
                         "evaluation_only; no model selection or checkpoint choice")

    def test_cli_writes_bound_json_deterministic_npz_and_receipt(self) -> None:
        truth = np.tile(np.asarray([[0, 2, 1, 1]], dtype=np.int8), (6, 1))
        good = self.probabilities_from_truth(truth, 0.9)
        weak = self.probabilities_from_truth(truth, 0.6)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth_path, paths = self.write_inputs(
                root, truth,
                {"full": good, "minus": weak, "RE": good, "RD": weak, "SHAM": weak},
            )
            output = root / "metrics.json"
            provenance = {
                "model_contract_receipt_sha256": "c" * 64,
                "base_contract_sha256": "b" * 64,
                "amendment_sha256": "a" * 64,
                "amendment_2_sha256": "d" * 64,
                "folds_sha256": "f" * 64,
                "folds_receipt_sha256": "e" * 64,
            }
            baseline_receipt = root / "baselines.receipt.json"
            baseline_receipt.write_text(json.dumps({
                "stage": "M38B_PACK_TRUTH_BLIND_OOF_BASELINES",
                "status": "PASS_FULL_MINUS_AND_ANALYTIC_RD_PACKED",
                "truth_read": False, "people": 96, "markers": 42326,
                "RD_alias": "OFF_EXACT_F_MINUS_S660_NO_FIT",
                "outputs": {
                    paths[arm].name: {
                        "sha256": subject.sha256_file(paths[arm]),
                        "source": "F_full_projected" if arm == "full" else "F_minus_S660",
                    }
                    for arm in ("full", "minus", "RD")
                }, **provenance,
            }), encoding="utf-8")
            prediction_receipts = {}
            for arm in ("RE", "SHAM"):
                receipt = root / f"{arm}.receipt.json"
                receipt.write_text(json.dumps({
                    "stage": "M38B_COLLECT_TRUTH_BLIND_OOF",
                    "status": "PASS_EXACT_ONE_OOF_PREDICTION_PER_PERSON",
                    "family": "analytic", "arm": arm, "diagnostic_only": False,
                    "output_sha256": subject.sha256_file(paths[arm]), **provenance,
                }), encoding="utf-8")
                prediction_receipts[arm] = receipt
            truth_receipt = root / "truth.receipt.json"
            truth_receipt.write_text(json.dumps({
                "stage": "M38B_PACK_OOF_SCORE_TRUTH",
                "status": "PASS_TRUTH_SEPARATE_SCORING_BRANCH",
                "output_sha256": subject.sha256_file(truth_path), **provenance,
            }), encoding="utf-8")
            argv = ["m38b_score_oof.py", "--truth", str(truth_path),
                    "--truth-receipt", str(truth_receipt), "--family", "analytic",
                    "--bootstrap-replicates", "100",
                    "--expected-person-count", "6", "--expected-fold-size", "2",
                    "--output", str(output)]
            for arm, path in paths.items():
                argv.extend(("--prediction", f"{arm}={path}"))
                receipt = (baseline_receipt if arm in {"full", "minus", "RD"}
                           else prediction_receipts[arm])
                argv.extend(("--prediction-receipt", f"{arm}={receipt}"))
            with mock.patch.object(sys, "argv", argv):
                subject.main()
            per_person = output.with_suffix(".per_person.npz")
            receipt = output.with_suffix(".receipt.json")
            self.assertTrue(output.is_file() and per_person.is_file() and receipt.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            audit = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["per_person_output"]["sha256"],
                             subject.sha256_file(per_person))
            self.assertEqual(audit["output_sha256"], subject.sha256_file(output))
            with np.load(per_person, allow_pickle=False) as archive:
                self.assertEqual(archive["per_person_contrasts"].shape, (4, 6, 9))
                self.assertEqual(archive["bootstrap_primary_deltas"].shape, (100, 4))
                self.assertEqual(archive["bootstrap_boundary_f1_deltas"].shape, (100, 4))


if __name__ == "__main__":
    unittest.main()
