#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(REPO / "bin"))
inventory = load_module("inventory_rare_allele_orientation", REPO / "bin" / "inventory_rare_allele_orientation.py")
aggregate = load_module(
    "aggregate_rare_allele_orientation_inventory",
    REPO / "bin" / "aggregate_rare_allele_orientation_inventory.py",
)
from rare_allele_orientation import SiteOrientation


class OrientationInventoryTest(unittest.TestCase):
    def test_filter_and_m14_universes_can_choose_different_minor_alleles(self):
        filter_orientation = SiteOrientation(alt_count=3, allele_number=10)
        subset_orientation = SiteOrientation(alt_count=5, allele_number=6)
        self.assertEqual(
            inventory.counted_allele(
                "minor_filter_cohort", filter_orientation, subset_orientation
            ),
            1,
        )
        self.assertEqual(
            inventory.counted_allele(
                "minor_m14_subset", filter_orientation, subset_orientation
            ),
            0,
        )

    def test_ties_have_no_unique_minor_allele(self):
        tie = SiteOrientation(alt_count=2, allele_number=4)
        minor = SiteOrientation(alt_count=1, allele_number=4)
        self.assertIsNone(inventory.counted_allele("minor_filter_cohort", tie, minor))
        self.assertEqual(inventory.counted_allele("historical_alt", tie, minor), 1)

    def test_comparison_does_not_reduce_impact_to_correlation(self):
        historical = np.array([100, 200, 300], dtype=np.int64)
        current = historical - 50
        result = aggregate.comparison(historical, current)
        self.assertAlmostEqual(result["pearson"], 1.0)
        self.assertEqual(result["mae"], 50.0)
        self.assertEqual(result["n_individuals_changed"], 3)


if __name__ == "__main__":
    unittest.main()
