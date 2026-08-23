#!/usr/bin/env python3
"""Adversarial tests for the M33 sanitized-F0 amendment boundary."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m33_m0_f0_sanitized_amendment", ROOT / "bin" / "m33_m0_f0_sanitized_amendment.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
AMENDMENT_PATH = ROOT / "conf" / "m33_m0_f0_sanitized_amendment_contract.json"
BASE_PATH = ROOT / "conf" / "m33_m0_materializer_contract.json"


class AmendmentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.amendment = MODULE.load_json(AMENDMENT_PATH)
        self.base = MODULE.load_json(BASE_PATH)

    def validate(self, amendment: dict | None = None, base: dict | None = None) -> None:
        MODULE.validate_contract(amendment or self.amendment, base or self.base, BASE_PATH)

    def test_frozen_amendment_passes_without_opening_execution(self) -> None:
        self.validate()
        authorization = self.amendment["execution_authorization"]
        self.assertTrue(authorization["contract_validation"])
        self.assertFalse(authorization["real_asset_read"])
        self.assertFalse(authorization["materialize"])
        self.assertFalse(authorization["training"])

    def test_materialize_sees_sanitized_f0_not_raw_flare(self) -> None:
        inputs = self.amendment["materialize_boundary"]["exact_input_logical_ids_after_amendment"]
        self.assertIn("flare_f0_sanitized", inputs)
        self.assertIn("safe_bridge_receipt", inputs)
        self.assertIn("safe_bridge_independent_verify_receipt", inputs)
        self.assertNotIn("a0_authenticated_flare_anc", inputs)
        self.assertNotIn("derived_flare_anc_tbi", inputs)

    def test_fit_normalization_and_genetic_map_remain_explicit(self) -> None:
        boundary = self.amendment["materialize_boundary"]
        inputs = boundary["exact_input_logical_ids_after_amendment"]
        self.assertIn("authenticated_fit_callable_normalization_manifest", inputs)
        self.assertIn("authenticated_genetic_map", inputs)
        self.assertIn("FIT_roots_only", boundary["fit_callable_normalization_manifest_role"])
        self.assertIn("geometry_only", boundary["genetic_map_role"])

    def test_sanitized_artifact_has_probabilities_without_genotypes_or_truth(self) -> None:
        sanitizer = self.amendment["sanitizer_boundary"]
        artifact = sanitizer["artifact"]
        self.assertEqual(artifact["arrays"]["F0"]["dtype"], "<f4")
        self.assertEqual(artifact["arrays"]["F0"]["axes"],
                         ["sample", "haplotype", "marker", "ancestry"])
        self.assertFalse(artifact["contains_raw_genotypes"])
        self.assertFalse(artifact["contains_hard_calls"])
        self.assertFalse(artifact["contains_truth"])
        self.assertFalse(sanitizer["truth_mounted"])

    def test_exact_three_development_roots_and_rotations_are_frozen(self) -> None:
        roots = self.amendment["development_roots"]
        self.assertEqual(roots["exact_seeds"], MODULE.EXPECTED_DEVELOPMENT_ROOTS)
        self.assertEqual(roots["rotations"], MODULE.EXPECTED_ROTATIONS)
        self.assertTrue(set(roots["exact_seeds"]).isdisjoint(roots["forbidden_technical_seeds"]))
        self.assertTrue(set(roots["exact_seeds"]).isdisjoint(roots["forbidden_eval_seeds"]))

    def test_base_contract_raw_or_semantic_drift_fails(self) -> None:
        for key in ("raw_sha256", "semantic_sha256"):
            changed = copy.deepcopy(self.amendment)
            changed["base_contract"][key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed)

    def test_raw_flare_truth_or_hard_calls_cannot_enter_materialize(self) -> None:
        for forbidden in ("a0_authenticated_flare_anc", "derived_flare_anc_tbi", "truth", "GT"):
            changed = copy.deepcopy(self.amendment)
            changed["materialize_boundary"]["exact_input_logical_ids_after_amendment"].append(forbidden)
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                self.validate(changed)

    def test_root_radius_channel_or_shard_changes_fail(self) -> None:
        cases = [
            ("development_roots", "exact_seeds", [1, 2, 3]),
            ("unchanged_scientific_contract", "radii_cM", [0.2]),
            ("unchanged_scientific_contract", "channel_count", 12),
            ("unchanged_scientific_contract", "maximum_valid_tokens_per_shard", 1),
        ]
        for section, field, value in cases:
            changed = copy.deepcopy(self.amendment)
            changed[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                self.validate(changed)

    def test_score_only_cannot_become_fit_normalizer(self) -> None:
        changed = copy.deepcopy(self.amendment)
        changed["materialize_boundary"]["fit_callable_normalization_manifest_role"] = "all_roots"
        with self.assertRaises(ValueError):
            self.validate(changed)

    def test_float32_simplex_tolerance_and_haplotype_axis_are_fixed(self) -> None:
        for field, value in (("float32_simplex_absolute_tolerance", 0.01),
                             ("haplotype_axis_preserved", False)):
            changed = copy.deepcopy(self.amendment)
            changed["sanitizer_boundary"]["probability_contract"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(changed)


if __name__ == "__main__":
    unittest.main()
