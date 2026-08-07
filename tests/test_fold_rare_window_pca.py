#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module("build_rare_window_features", ROOT / "bin/build_rare_window_features.py")
FOLD = load_module("build_fold_rare_window_features", ROOT / "bin/build_fold_rare_window_features.py")
PCA = load_module("evaluate_fold_rare_window_pca", ROOT / "bin/evaluate_fold_rare_window_pca.py")


class FoldRareWindowPcaTest(unittest.TestCase):
    def test_weighted_baselines_use_fit_values_only(self):
        x_fit = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
        x_validation = np.array([[99.0, 99.0]])
        mean, burden, intercept, slope = PCA.linear_predictions(
            x_fit,
            x_validation,
            np.array([0.0, 1.0, 2.0]),
            np.array([3.0]),
        )
        np.testing.assert_allclose(mean, [[1.0, 3.0]])
        np.testing.assert_allclose(intercept, [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(slope, [1.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(burden, [[3.0, 7.0]], atol=1e-12)

    def test_subspace_overlap_is_rotation_invariant(self):
        left = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        right = rotation @ left
        metrics = PCA.subspace_metrics(left, right)
        self.assertAlmostEqual(metrics["overlap"], 1.0)
        self.assertAlmostEqual(metrics["max_angle_degrees"], 0.0)

    def test_nonprimary_rank_cannot_drive_automatic_gate(self):
        positive = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertTrue(PCA.primary_performance_gate(1, 1, positive, 0.01))
        self.assertFalse(PCA.primary_performance_gate(2, 1, positive, 0.01))
        self.assertFalse(PCA.primary_performance_gate(1, 1, positive, -0.01))

    def test_pairwise_site_stability_counts_orientation_flips(self):
        result = FOLD.pairwise_site_stability(
            {0: {"a": "ALT", "b": "REF"}, 1: {"b": "ALT", "c": "REF"}}
        )[0]
        self.assertEqual(result["intersection"], 1)
        self.assertEqual(result["orientation_flips_in_intersection"], 1)
        self.assertAlmostEqual(result["jaccard"], 1 / 3)

    def test_validation_genotypes_cannot_change_fit_minor_orientation(self):
        fit = np.array(
            [[ord("1"), ord("1")], [ord("1"), ord("1")],
             [ord("0"), ord("1")], [ord("0"), ord("1")]],
            dtype=np.uint8,
        )
        validation_a = np.array([[ord("0"), ord("0")]], dtype=np.uint8)
        validation_b = np.array([[ord("1"), ord("1")]], dtype=np.uint8)
        metrics_a = BASE.site_metrics(np.vstack([fit, validation_a])[:4])
        metrics_b = BASE.site_metrics(np.vstack([fit, validation_b])[:4])
        self.assertEqual(metrics_a["counted_allele"], "REF")
        self.assertEqual(metrics_a["minor_count"], 2)
        self.assertEqual(metrics_a["counted_allele"], metrics_b["counted_allele"])
        self.assertEqual(metrics_a["minor_count"], metrics_b["minor_count"])

    def test_metric_rejects_zero_callability_sensitivity_weight(self):
        with self.assertRaises(ValueError):
            PCA.error_metrics(
                np.ones((2, 2)), np.zeros((2, 2)), np.array([[1.0, 1.0], [1.0, 0.0]])
            )

    def test_subspace_rejects_rank_larger_than_common_support(self):
        with self.assertRaises(ValueError):
            PCA.subspace_metrics(np.ones((2, 1)), np.ones((2, 1)))

    def test_workflow_contract_keeps_test_blind_and_rank_descriptive(self):
        module = (ROOT / "modules/25B_RARE_WINDOW_PCA.nf").read_text(encoding="utf-8")
        subworkflow = (ROOT / "subworkflows/25B_RARE_WINDOW_PCA.nf").read_text(
            encoding="utf-8"
        )
        config = (ROOT / "conf/google_batch.config").read_text(encoding="utf-8")
        self.assertIn("expected-test-samples", module)
        self.assertIn("rank_selection", module)
        self.assertIn("no rank selected by reconstruction error", subworkflow)
        self.assertIn("resourceLabels = [team: 'frank']", config)
        self.assertIn("rare_window_pca_input_dir", config)
        self.assertIn("results_modtest_mac2/lai_rare", config)
        self.assertIn("container_path", subworkflow)
        self.assertIn("container_sha256", subworkflow)
        self.assertEqual(module.count("--run-provenance-ref ../run_provenance.json"), 3)
        self.assertIn("--fold-manifest", module)
        self.assertNotIn("null_replicates", module)
        for forbidden in ("NMF", "AUTOENCODER", "GNOMIX", "RFMIX"):
            self.assertNotIn(forbidden, module.upper())


if __name__ == "__main__":
    unittest.main()
