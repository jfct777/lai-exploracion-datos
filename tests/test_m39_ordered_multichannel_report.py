"""Report contract fixtures; real primary auditing is covered by the sweep suite."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import m39_ordered_multichannel_report as R
import m39_ordered_multichannel_sweep as S
import test_m39_ordered_multichannel_sweep as fixtures


class MultichannelReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MultichannelSweepTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        cfg = self.fixture.cfg
        cfg.update(anchor_count=2, steps=4, evaluate_every_steps=2, evaluate_initial=True)
        self.fixture.base.write_text(json.dumps(cfg))
        self.keys = np.asarray([b"a" * 64, b"b" * 64], dtype="S64")
        self.loci = np.asarray([11, 25, 39], dtype=np.int64)
        self.truth = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        self.base = .7 * np.eye(6)[self.truth] + .3 / 6
        # Middle locus deliberately has different loss so whole-axis substitution fails.
        self.base[:, 1] = 1 / 6
        self.full = .9 * np.eye(6)[self.truth] + .1 / 6
        self.calibration = self.root / "calibration"
        self.calibration.mkdir()
        self.write_calibration()
        self.stage_data = {}
        self.screen = self.make_stage("A", self.fixture.recipes)
        followup = [dict(id=f"{family}-seed{seed}", family=family, learning_rate=.0003,
                        seed=seed, pair_seed=seed + 100)
                    for family in ("cnn", "attention") for seed in (18, 19)]
        self.followup = self.make_stage("B", followup)

    def tearDown(self):
        self.fixture.tearDown()

    def write_calibration(self):
        cfg = self.fixture.cfg
        indices = np.asarray([0, 2])
        parameters = {"family": "identity", "temperature": 1., "alpha": 0., "prior": [1 / 6] * 6,
                      "anchor_indices": indices.tolist()}
        (self.calibration / "calibration.json").write_text(json.dumps(parameters))
        (self.calibration / "candidates.csv").write_text("family,brier\nidentity,0.1\n")
        np.savez(self.calibration / "predictions.npz", locus_id=self.loci[indices], anchor_indices=indices,
                 state_names=np.asarray(R.STATE_NAMES, dtype="S2"),
                 select_sample_key_sha256=self.keys, select_truth_state=self.truth[:, indices],
                 select_FMINUS=self.base[:, indices], select_FFULL=self.full[:, indices],
                 select_CALIBRATION_ONLY=self.base[:, indices])
        report = {"schema_version": "m39-multichannel-calibration-v1",
                  "status": "EXPLORATORY_CALIBRATION_ONLY_NOT_VALIDATION",
                  "score_read": False, "source_test_or_valid_opened": False,
                  "genotypes_used_as_predictors": False, "reopened_predictions_exact": True,
                  "calibration": parameters,
                  "input_sha256": {"development": cfg["development_sha256"],
                                   "train_manifest": cfg["train_manifest_sha256"],
                                   "select_manifest": cfg["select_manifest_sha256"]},
                  "role_metrics": {"SELECT": {name: R.metrics(value[:, indices], self.truth[:, indices]) for name, value in
                      (("FMINUS", self.base), ("FFULL", self.full), ("CALIBRATION_ONLY", self.base))}},
                  "outputs_sha256": {name: R.sha256(self.calibration / name) for name in
                      ("calibration.json", "predictions.npz", "candidates.csv")}}
        (self.calibration / "report.json").write_text(json.dumps(report))

    def make_stage(self, label, recipes):
        stage = "multichannel_screen" if label == "A" else "multichannel_followup"
        plan = S.freeze_plan(self.fixture.base, self.fixture.resources, recipes, stage, self.root / ("plan-" + label))
        outputs = self.root / ("outputs-" + label)
        outputs.mkdir()
        parsed = S.load_plan(plan)
        groups = []
        for group in parsed["groups"]:
            parent = outputs / ("training-" + group["id"])
            parent.mkdir()
            group_metrics = {}
            for spec in group["configs"]:
                cfg = R.load_config(plan.parent / spec["file"])
                # Reversed sample order and a strict locus subset exercise both joins.
                target_truth = self.truth[::-1][:, [0, 2]]
                error = { .0001: .3, .0003: .1, .001: .2 }[cfg["learning_rate"]]
                error += (list(R.ARMS).index(cfg["arm"]) + 1) * .005
                probability = (1 - error) * np.eye(6)[target_truth] + error / 6
                selected = R.metrics(probability, target_truth)
                group_metrics[cfg["arm"]] = selected
                directory = parent / cfg["arm"]
                directory.mkdir()
                np.savez(directory / "select.predictions.npz", sample_key_sha256=self.keys[::-1],
                         locus_id=self.loci[[0, 2]], anchor_indices=np.asarray([0, 2]),
                         truth_state=target_truth, probabilities=probability)
                (directory / "checkpoint.pt").write_bytes(b"synthetic-checkpoint-not-torch-loaded")
                curve = []
                for step in (0, 2, 4):
                    point = {"step": step, "checkpoint_eligible": step > 0,
                        "SELECT": selected, "TRAIN_probe": None, "train_cross_entropy_window": None if not step else .3,
                        "gradient_norm_last_step": None if not step else .2,
                        "train_observations_cumulative": step * 2,
                        "TRAIN_exposure": {"complete_passes_over_declared_pairs": step // 2},
                        "training_seconds_cumulative": step, "evaluation_seconds_cumulative": step / 2}
                    curve.append(point)
                    (directory / f"curve-step-{step:07d}.json").write_text(json.dumps(point))
                receipt = {"config": cfg, "selected_SELECT_metrics": selected, "selected_step": 2,
                    "anchor_indices": [0, 2],
                    "TRAIN_exposure": {"complete_passes_over_declared_pairs": 2},
                    "budget_diagnostics": {"best_checkpoint_at_budget_end": False}, "curve": curve,
                    "Fminus_SELECT": R.metrics(self.base[::-1][:, [0, 2]], target_truth),
                    "Ffull_SELECT": R.metrics(self.full[::-1][:, [0, 2]], target_truth)}
                (directory / "training.receipt.json").write_text(json.dumps(receipt))
            (parent / "group.completion.json").write_text(json.dumps({"synthetic_fixture": True}))
            groups.append({"id": group["id"], "family": cfg["model"]["family"],
                           "seed": cfg["seed"], "learning_rate": cfg["learning_rate"],
                           "NONE_metrics": group_metrics["none"], "BOTH_metrics": group_metrics["both"]})
        comparison = {"schema_version": S.PROTOCOL.comparison_schema, "stage": stage,
                      "plan_sha256": R.sha256(plan), "groups": groups,
                      "metric_recomputation": {"policy": "finite_float64_absolute_error_only_v1",
                                               "atol": 1e-12, "nonzero_differences": []},
                      "scope": {"SCORE_opened": False, "technical_e2e_only": False}}
        comparison_path = self.root / ("comparison-" + label + ".json")
        comparison_path.write_text(json.dumps(comparison))
        self.stage_data[str(plan)] = comparison
        return plan, outputs, comparison_path

    def run_report(self, **kwargs):
        values = {"screen_plan": self.screen[0], "screen_outputs": self.screen[1], "screen_comparison": self.screen[2],
                  "followup_plan": self.followup[0], "followup_outputs": self.followup[1],
                  "followup_comparison": self.followup[2], "calibration_dir": self.calibration,
                  "calibration_report_sha256": R.sha256(self.calibration / "report.json"),
                  "outdir": self.root / "report"}
        values.update(kwargs)
        # Do not pretend these minimal fixtures passed a real primary audit.
        # The report must call that boundary; its actual E2E is tested separately.
        with patch.object(S, "audit_results", side_effect=lambda plan, outputs: self.stage_data[str(plan)]):
            return R.build_report(**values)

    def test_complete_32_fit_report_rescores_exact_calibration_axis_and_units(self):
        report = self.run_report()
        self.assertEqual(report["counts"], {"screen_groups": 6, "screen_fits": 12,
            "followup_groups": 4, "followup_fits": 20, "SELECT_people": 2, "anchors": 2})
        self.assertEqual(len(report["all_neural_cases"]), 32)
        self.assertEqual(len(report["followup_table"]), 32)
        self.assertEqual(len(report["BOTH_minus_comparator"]), 28)
        self.assertEqual(len(report["learning_curves"]), 96)
        self.assertEqual(report["selected_learning_rates"]["cnn"]["learning_rate"], .0003)
        calibrated = next(row for row in report["followup_table"] if row["arm"] == "CALIBRATION_ONLY")
        expected = R.metrics(self.base[:, [0, 2]], self.truth[:, [0, 2]])
        self.assertAlmostEqual(calibrated["brier"], expected["brier"])
        self.assertNotAlmostEqual(calibrated["brier"], R.metrics(self.base, self.truth)["brier"])
        self.assertTrue(all(not row["subset_rescored"] for row in report["calibration_alignment"].values()))
        self.assertTrue(all(not row["selection_used_larger_anchor_universe"] for row in report["calibration_alignment"].values()))
        self.assertTrue(all(row["source_axis_sha256"] != row["evaluated_axis_sha256"]
                            for row in report["calibration_alignment"].values()))
        delta = next(row for row in report["BOTH_minus_comparator"] if row["comparator"] == "FMINUS")
        focal = next(row for row in report["followup_table"] if row["arm"] == "BOTH" and row["group"] == delta["group"])
        baseline = next(row for row in report["followup_table"] if row["arm"] == "FMINUS" and row["group"] == delta["group"])
        self.assertEqual(delta["delta_brier"], focal["brier"] - baseline["brier"])
        self.assertAlmostEqual(delta["delta_brier"], focal["brier"] - expected["brier"])
        self.assertEqual(report["units"]["dosage_mae"], "ancestral_copies_0_to_2")
        markdown = (self.root / "report/report.md").read_text()
        for text in ("12 ajustes", "20 ajustes", "copias 0–2", "no significación", "No se evaluaron LAI denso"):
            self.assertIn(text, markdown)
        self.assertEqual({p.name for p in (self.root / "report").iterdir()},
                         {"report.md", "report.json", "cases.csv", "followup-table.csv", "contrasts.csv", "learning-curves.csv"})

    def test_calibration_hash_and_artifact_modification_fail_before_output(self):
        with self.assertRaisesRegex(ValueError, "report hash"):
            self.run_report(calibration_report_sha256="0" * 64)
        path = self.calibration / "predictions.npz"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            self.run_report()
        self.assertFalse((self.root / "report").exists())

    def test_unverified_comparison_cannot_be_used(self):
        source = json.loads(self.screen[2].read_text())
        source["groups"][0]["BOTH_metrics"]["brier"] += .1
        self.screen[2].write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "verified comparison"):
            self.run_report()

    def test_axis_join_rejects_missing_duplicate_and_truth_mismatch(self):
        arrays = {"select_sample_key_sha256": self.keys, "locus_id": self.loci,
                  "select_truth_state": self.truth, **{"select_" + n: self.base for n in R.COMPARATORS}}
        neural = {"sample_key_sha256": self.keys[::-1], "locus_id": self.loci[[0, 2]],
                  "truth_state": self.truth[::-1][:, [0, 2]]}
        aligned, binding = R.align_calibration(arrays, neural)
        self.assertTrue(binding["subset_rescored"])
        self.assertTrue(binding["selection_used_larger_anchor_universe"])
        self.assertEqual(aligned["FMINUS"], R.metrics(self.base[::-1][:, [0, 2]], neural["truth_state"]))
        for changed in (dict(neural, locus_id=np.asarray([11, 99])),
                        dict(neural, locus_id=np.asarray([11, 11])),
                        dict(neural, sample_key_sha256=np.asarray([b"x" * 64, b"y" * 64], dtype="S64")),
                        dict(neural, truth_state=(neural["truth_state"] + 1) % 6)):
            with self.assertRaises(ValueError):
                R.align_calibration(arrays, changed)

    def test_partial_followup_is_not_reported_as_complete(self):
        plan = json.loads(self.followup[0].read_text())
        plan["groups"].pop()
        self.followup[0].chmod(0o600)
        self.followup[0].write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "both families"):
            self.run_report()
        self.assertFalse((self.root / "report").exists())

    def test_wrong_followup_learning_rate_and_geometry_rejected(self):
        path = self.followup[0]
        plan = json.loads(path.read_text())
        for spec in plan["groups"][0]["configs"]:
            config_path = path.parent / spec["file"]
            cfg = json.loads(config_path.read_text())
            cfg["learning_rate"] = .001
            config_path.chmod(0o600)
            config_path.write_text(json.dumps(cfg))
            spec["sha256"] = R.sha256(config_path)
        path.chmod(0o600)
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "learning rate differs"):
            self.run_report()

    def test_report_never_overwrites_existing_output(self):
        (self.root / "report").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_report()

    def test_comparison_numeric_policy_is_not_silently_ignored(self):
        comparison = json.loads(self.screen[2].read_text())
        comparison["metric_recomputation"]["atol"] = .1
        self.screen[2].write_text(json.dumps(comparison))
        with self.assertRaisesRegex(ValueError, "verified comparison"):
            self.run_report()

    def test_larger_calibration_fit_is_rejected_even_after_safe_subset_rescoring(self):
        original = R.align_calibration

        def larger_fit(*args):
            aligned, binding = original(*args)
            return aligned, dict(binding, selection_used_larger_anchor_universe=True)

        with patch.object(R, "align_calibration", side_effect=larger_fit):
            with self.assertRaisesRegex(ValueError, "selection anchor universe differs"):
                self.run_report()
        self.assertFalse((self.root / "report").exists())


if __name__ == "__main__":
    unittest.main()
