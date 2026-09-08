"""Brier calibration contracts on artificial TRAIN/SELECT inputs only."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import m39_calibration as H
import m39_ordered_multichannel_calibration as C
import test_m39_ordered_training as F


class MultichannelCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = F.OrderedTrainingTests("test_binding_joins_role_keys_and_exact_allele_axis")
        self.fixture.setUp()
        self.root = self.fixture.root
        (_, _, self.train, self.select, self.development, _, self.cfg) = self.fixture.setup_training()
        self.plan = {"schema_version": "m39-calibration-plan-v1", "scope": H.SCOPE,
            "development_sha256": self.cfg["development_sha256"], "score_sha256": "a" * 64,
            "comparators_manifest_sha256": "b" * 64, "temperature_grid": [.25, .5, 1., 2., 4., 8., 16.],
            "alpha_grid": [0, .0001, .001, .01, .03, .1, .2, .4, .6, .8, 1],
            "temperature_endpoints": [.125, 32.], "prior_pseudocount": 1,
            "floor": 1e-12, "tie_tolerance": 1e-12, "bootstrap_replicates": 10000,
            "bootstrap_seed": 39062027, "reliability_bins": [5, 10, 20]}
        self.plan_path = self.root / "historical-grid.json"
        self.plan_path.write_text(json.dumps(self.plan))

    def tearDown(self):
        self.fixture.tearDown()

    def run_calibration(self, **kwargs):
        params = {"train_manifest_sha256": self.cfg["train_manifest_sha256"],
                  "select_manifest_sha256": self.cfg["select_manifest_sha256"],
                  "development_sha256": self.cfg["development_sha256"],
                  "historical_plan_sha256": C.sha256(self.plan_path)}
        params.update(kwargs)
        return C.run_calibration(self.train, self.select, self.development, self.plan_path,
                                 self.root / "output", **params)

    def test_brier_then_log_loss_then_simplicity_order(self):
        records = [dict(family="identity", temperature=1., alpha=0., train_brier=.4, train_log_loss=.5),
                   dict(family="temperature", temperature=2., alpha=0., train_brier=.3, train_log_loss=.7)]
        self.assertEqual(C.choose_brier(records, "train", 1e-12)["family"], "temperature")
        records[0]["train_brier"] = .3
        self.assertEqual(C.choose_brier(records, "train", 1e-12)["family"], "identity")
        records[0]["train_log_loss"] = .7
        self.assertEqual(C.choose_brier(records[::-1], "train", 1e-12)["family"], "identity")

    def test_fit_cannot_accept_select_and_finalists_minimize_train_brier(self):
        self.assertNotIn("select", inspect.signature(C.fit_families_brier).parameters)
        p = np.asarray([[[.85, .05, .025, .025, .025, .025], [.05, .85, .025, .025, .025, .025]]])
        y = np.asarray([[0, 2]], dtype=np.uint8)
        prior, candidates, finalists = C.fit_families_brier(p, y, self.plan)
        self.assertEqual(len(finalists), 5)
        self.assertLess(len(candidates), 160)
        np.testing.assert_array_equal(prior, H.train_prior(y, 1))
        for final in finalists:
            family = [row for row in candidates if row["family"] == final["family"]]
            self.assertEqual(C.choose_brier(family, "train", 1e-12)["temperature"], final["temperature"])
            self.assertEqual(C.choose_brier(family, "train", 1e-12)["alpha"], final["alpha"])
            self.assertEqual(sum(row["train_finalist"] for row in family), 1)
            self.assertEqual(len({(row["temperature"], row["alpha"]) for row in family}), len(family))
        self.assertTrue(any(row["stage"] == "train_neighbor_refinement" for row in candidates))

    def test_select_only_evaluates_frozen_finalists_and_does_not_change_prior(self):
        p = np.full((1, 2, 6), 1 / 6)
        prior, _, finalists = C.fit_families_brier(p, np.asarray([[0, 0]]), self.plan)
        snapshot, frozen_prior = copy.deepcopy(finalists), prior.copy()
        with patch.object(H, "train_prior", side_effect=AssertionError("SELECT fitting prior")), \
                patch.object(H, "neighbors", side_effect=AssertionError("SELECT grid refinement")):
            _, evaluated = C.select_family(p, np.asarray([[5, 5]]), prior, finalists, 1e-12)
        self.assertEqual(finalists, snapshot)
        np.testing.assert_array_equal(prior, frozen_prior)
        self.assertEqual({(r["family"], r["temperature"], r["alpha"]) for r in evaluated},
                         {(r["family"], r["temperature"], r["alpha"]) for r in finalists})

    def test_identity_temperature_zeros_and_positive_mixture_policy_reused(self):
        p = np.eye(6, dtype=np.float32)[np.asarray([[0, 1]])]
        y = np.asarray([[0, 2]])
        prior = H.train_prior(y)
        np.testing.assert_array_equal(H.transform(p, "identity"), p)
        np.testing.assert_array_equal(H.transform(p, "temperature", 2), p)
        self.assertTrue(np.all(H.transform(p, "prior_mix", alpha=.1, prior=prior) > 0))
        before = p.copy()
        C.fit_families_brier(p, y, self.plan)
        np.testing.assert_array_equal(p, before)

    def test_end_to_end_outputs_exact_reopening_and_no_score(self):
        report = self.run_calibration()
        self.assertFalse(report["score_read"])
        self.assertFalse(report["source_test_or_valid_opened"])
        self.assertFalse(report["genotypes_used_as_predictors"])
        self.assertTrue(report["reopened_predictions_exact"])
        self.assertEqual(report["calibration"]["primary_metric"], "brier")
        self.assertEqual(report["calibration"]["historical_primary_metric"], "log_loss")
        self.assertEqual(report["role_counts"], {"TRAIN": 1, "SELECT": 1})
        self.assertEqual(set(report["outputs_sha256"]), {"calibration.json", "candidates.csv", "predictions.npz"})
        with np.load(self.root / "output/predictions.npz", allow_pickle=False) as z:
            self.assertEqual(z["train_CALIBRATION_ONLY"].shape, (1, 2, 6))
            np.testing.assert_array_equal(z["train_truth_state"], [[0, 5]])
            np.testing.assert_array_equal(z["select_truth_state"], [[1, 4]])
            for role in ("train", "select"):
                for name in ("FMINUS", "FFULL", "CALIBRATION_ONLY"):
                    self.assertEqual(C.metrics(z[f"{role}_{name}"], z[f"{role}_truth_state"]),
                                     report["role_metrics"][role.upper()][name])

    def test_fixed_anchor_subset_is_exact_and_invalid_indices_fail(self):
        report = self.run_calibration(anchor_indices=np.asarray([1], dtype=np.int64))
        self.assertEqual(report["anchors"], 1)
        self.assertEqual(report["calibration"]["anchor_indices"], [1])
        with np.load(self.root / "output/predictions.npz", allow_pickle=False) as z:
            self.assertEqual(z["train_CALIBRATION_ONLY"].shape, (1, 1, 6))
            np.testing.assert_array_equal(z["anchor_indices"], [1])
        for invalid in (np.asarray([]), np.asarray([True]), np.asarray([2]), np.asarray([1, 0]),
                        np.asarray([0, 0]), np.asarray([.5])):
            with self.assertRaises(ValueError):
                C._anchors(invalid, 2)

    def test_source_and_development_hashes_and_extra_score_field_fail(self):
        for key in ("development_sha256", "train_manifest_sha256", "historical_plan_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_calibration(**{key: "c" * 64})
        with np.load(self.development, allow_pickle=False) as z:
            altered = {key: z[key] for key in z.files}
        altered["score_indices"] = np.asarray([], dtype=np.int64)
        np.savez(self.development, **altered)
        changed_hash = C.sha256(self.development)
        self.plan["development_sha256"] = changed_hash
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "SCORE"):
            self.run_calibration(development_sha256=changed_hash)

    def test_budget_and_output_collision_fail_without_overwriting(self):
        with self.assertRaisesRegex(ValueError, "runtime ceiling"):
            self.run_calibration(max_runtime_seconds=61)
        with self.assertRaisesRegex(ValueError, "time ceiling"):
            C.fit_families_brier(np.full((1, 2, 6), 1 / 6), np.asarray([[0, 0]]), self.plan,
                check_budget=lambda: C.require(False, "calibration time ceiling exceeded"))
        (self.root / "output").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_calibration()


if __name__ == "__main__":
    unittest.main()
