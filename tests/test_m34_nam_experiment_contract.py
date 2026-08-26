#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "conf" / "m34_nam_experiment_contract.json"


class M34NamExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_scope_is_explicitly_exploratory_afr_eur_nam_chr22(self):
        self.assertEqual(self.contract["claim_level"], "exploratory")
        self.assertEqual(self.contract["chromosome"], "22")
        self.assertEqual(self.contract["ancestry_order"], ["AFR", "EUR", "NAM"])

    def test_roles_keep_genealogies_disjoint_and_forbid_king(self):
        roles = self.contract["roles"]
        self.assertEqual(
            roles["source_test"],
            "OPENED_ONLY_AS_EXPLORATORY_VALID_DONORS_AFTER_CONFIRMATORY_STATUS_WAS_LOST",
        )
        self.assertEqual(roles["mosaic_fit_donors"], "SOURCE_VALID")
        self.assertEqual(roles["mosaic_valid_donors"], "SOURCE_TEST")
        self.assertEqual(roles["relatedness_sources"], ["PC-Relate", "Refined IBD"])
        self.assertFalse(roles["king_used"])

    def test_rare_selection_is_minor_oriented_and_truth_blind(self):
        rare = self.contract["rare_definition"]
        self.assertEqual(rare["minimum_mac"], 2)
        self.assertEqual(rare["maximum_maf_exclusive"], 0.01)
        self.assertIn("minor_allele", rare["orientation"])
        self.assertFalse(rare["target_genotypes_used_for_selection"])
        self.assertFalse(rare["truth_or_baseline_used_for_selection"])

    def test_primary_brazilian_mixture_and_generation_screen_are_frozen(self):
        mosaics = self.contract["mosaics"]
        self.assertEqual(mosaics["primary_mixture_proportions"], {
            "AFR": 0.25, "EUR": 0.60, "NAM": 0.15,
        })
        self.assertAlmostEqual(sum(mosaics["primary_mixture_proportions"].values()), 1.0)
        self.assertEqual(mosaics["primary_admixture_generations"], 12)
        self.assertEqual(self.contract["baseline"]["parameters"]["generations"], 12.0)
        self.assertEqual(
            self.contract["rotation_semantics"],
            "independent_mosaic_realization_root_not_donor_unit_rotation",
        )
        self.assertEqual(mosaics["generation_sensitivity"], [9, 12, 17])

    def test_small_and_medium_sizes_align_with_materializer_shards(self):
        materializer = json.loads(
            (ROOT / self.contract["materializer_contract"]).read_text(encoding="utf-8")
        )
        shard = materializer["sample_shard_size"]
        sizes = self.contract["mosaics"]["target_sizes"]
        for row in sizes.values():
            self.assertEqual(row["people"], row["fit"] + row["valid"])
            self.assertEqual(row["fit"] % shard, 0)
            self.assertEqual(row["valid"] % shard, 0)

    def test_referenced_contracts_exist(self):
        for name in ("model_screen_contract", "materializer_contract", "asset_registry"):
            self.assertTrue((ROOT / self.contract[name]).is_file())


if __name__ == "__main__":
    unittest.main()
