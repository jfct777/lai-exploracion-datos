#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "bin" / "build_rare_window_features.py"
SPEC = importlib.util.spec_from_file_location("build_rare_window_features", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RareWindowFeaturesTest(unittest.TestCase):
    def test_window_boundaries_are_zero_based_half_open(self):
        self.assertEqual(MODULE.window_index(1, 250_000), 0)
        self.assertEqual(MODULE.window_index(250_000, 250_000), 0)
        self.assertEqual(MODULE.window_index(250_001, 250_000), 1)

    def test_alt_minor_site_uses_called_alleles_for_ac_an(self):
        alleles = np.array(
            [[ord("0"), ord("0")], [ord("0"), ord("1")], [ord("."), ord("1")]],
            dtype=np.uint8,
        )
        metrics = MODULE.site_metrics(alleles)
        self.assertEqual((metrics["alt_count"], metrics["allele_number"]), (2, 5))
        self.assertEqual(metrics["counted_allele"], "ALT")
        self.assertEqual(int(metrics["complete"].sum()), 2)
        self.assertEqual(int(metrics["partial"].sum()), 1)

    def test_ref_minor_orientation_and_filter(self):
        alleles = np.array(
            [[ord("1"), ord("1")], [ord("1"), ord("1")], [ord("0"), ord("1")]],
            dtype=np.uint8,
        )
        metrics = MODULE.site_metrics(alleles)
        self.assertEqual(metrics["counted_allele"], "REF")
        self.assertEqual(metrics["minor_count"], 1)
        self.assertEqual(MODULE.exclusion_reason(metrics, 2, 0.5), "mac_below_min")

    def test_tie_has_no_unique_minor_allele(self):
        alleles = np.array(
            [[ord("0"), ord("1")], [ord("0"), ord("1")]], dtype=np.uint8
        )
        metrics = MODULE.site_metrics(alleles)
        self.assertIsNone(metrics["counted_allele"])
        self.assertEqual(MODULE.exclusion_reason(metrics, 1, 0.5), "frequency_tie")

    def test_aggregate_invariants_accept_consistent_counts(self):
        panel = np.array([2])
        callable_sites = np.array([[2, 1]], dtype=np.int32)
        carriers = np.array([[1, 1]], dtype=np.int32)
        dosage = np.array([[1, 2]], dtype=np.int32)
        het = np.array([[1, 0]], dtype=np.int32)
        hom = np.array([[0, 1]], dtype=np.int32)
        MODULE.validate_aggregates(panel, callable_sites, carriers, dosage, het, hom)

    def test_aggregate_invariants_fail_on_impossible_dosage(self):
        with self.assertRaises(SystemExit):
            MODULE.validate_aggregates(
                np.array([1]),
                np.array([[1]], dtype=np.int32),
                np.array([[0]], dtype=np.int32),
                np.array([[1]], dtype=np.int32),
                np.array([[0]], dtype=np.int32),
                np.array([[0]], dtype=np.int32),
            )


if __name__ == "__main__":
    unittest.main()
