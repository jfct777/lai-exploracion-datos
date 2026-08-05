#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "bin" / "rare_allele_orientation.py"
SPEC = importlib.util.spec_from_file_location("rare_allele_orientation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RareAlleleOrientationTest(unittest.TestCase):
    def test_alt_minor_keeps_historical_carriers(self):
        genotypes = [[0, 0, False], [0, 1, False], [0, 1, True], [-1, -1, False]]
        orientation = MODULE.summarize_orientation(genotypes)
        self.assertEqual((orientation.alt_count, orientation.allele_number), (2, 6))
        self.assertFalse(orientation.alt_is_major)
        expected = frozenset({1, 2})
        self.assertEqual(MODULE.carrier_indices(genotypes, orientation, "historical_alt"), expected)
        self.assertEqual(MODULE.carrier_indices(genotypes, orientation, "minor_allele"), expected)
        self.assertEqual(MODULE.carrier_indices(genotypes, orientation, "exclude_alt_major"), expected)

    def test_alt_major_is_reoriented_to_reference_minor(self):
        genotypes = [[1, 1, False], [1, 1, True], [0, 1, False], [-1, -1, False]]
        orientation = MODULE.summarize_orientation(genotypes)
        self.assertEqual((orientation.alt_count, orientation.allele_number), (5, 6))
        self.assertTrue(orientation.alt_is_major)
        self.assertEqual(
            MODULE.carrier_indices(genotypes, orientation, "historical_alt"),
            frozenset({0, 1, 2}),
        )
        self.assertEqual(
            MODULE.carrier_indices(genotypes, orientation, "minor_allele"),
            frozenset({2}),
        )
        self.assertIsNone(
            MODULE.carrier_indices(genotypes, orientation, "exclude_alt_major")
        )

    def test_minor_dosage_reverses_alt_dosage_at_alt_major_site(self):
        orientation = MODULE.SiteOrientation(alt_count=5, allele_number=6)
        self.assertEqual(MODULE.dosage_for_mode([1, 1, False], orientation, "minor_allele"), 0)
        self.assertEqual(MODULE.dosage_for_mode([0, 1, False], orientation, "minor_allele"), 1)
        self.assertEqual(MODULE.dosage_for_mode([0, 0, False], orientation, "minor_allele"), 2)
        self.assertIsNone(MODULE.dosage_for_mode([0, -1, False], orientation, "minor_allele"))

    def test_tie_keeps_historical_semantics_but_has_no_unique_minor(self):
        tie = MODULE.summarize_orientation([[0, 1, False], [0, 1, False]])
        self.assertTrue(tie.is_tie)
        genotypes = [[0, 1, False], [0, 1, False]]
        expected = frozenset({0, 1})
        self.assertEqual(MODULE.carrier_indices(genotypes, tie, "historical_alt"), expected)
        self.assertEqual(MODULE.carrier_indices(genotypes, tie, "exclude_alt_major"), expected)
        self.assertIsNone(MODULE.carrier_indices(genotypes, tie, "minor_allele"))
        self.assertEqual(MODULE.dosage_for_mode([0, 1, False], tie, "historical_alt"), 1)
        self.assertIsNone(MODULE.dosage_for_mode([0, 1, False], tie, "minor_allele"))

    def test_all_missing_is_excluded_from_every_mode(self):
        missing = MODULE.summarize_orientation([[-1, -1, False]])
        self.assertEqual(missing.allele_number, 0)
        self.assertIsNone(MODULE.dosage_for_mode([-1, -1, False], missing, "historical_alt"))

    def test_unknown_mode_fails_loudly(self):
        orientation = MODULE.SiteOrientation(alt_count=1, allele_number=4)
        with self.assertRaises(ValueError):
            MODULE.carrier_indices([[0, 1, False]], orientation, "mystery")


if __name__ == "__main__":
    unittest.main()
