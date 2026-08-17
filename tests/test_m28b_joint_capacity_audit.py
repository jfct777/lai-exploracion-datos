"""Invariant tests for the joint M28B-v2 capacity allocation."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location(
    "m28b_joint_capacity_audit", BIN / "m28b_joint_capacity_audit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def marker(site_id: int, bp: int, cm: float, maf: float, ref: int, afr: int = 0, eur: int = 0, asia: int = 0):
    return MODULE.Marker(site_id, bp, cm, 1, 2, 600, maf, ref, afr, eur, asia)


class TestBaselineGenotypes(unittest.TestCase):
    def test_polymorphism_is_recomputed_from_gt_not_info(self):
        self.assertTrue(MODULE.genotype_is_polymorphic(["0|0", "0|1", "0|0"]))
        self.assertFalse(MODULE.genotype_is_polymorphic(["0|0", "0|0", "0|0"]))
        self.assertFalse(MODULE.genotype_is_polymorphic(["1|1", "1|1"]))


class TestJointAllocation(unittest.TestCase):
    def setUp(self):
        self.templates = [
            MODULE.BaselineTemplate(100, 0.010, True),
            MODULE.BaselineTemplate(200, 0.020, False),
        ]
        self.rare = [
            marker(1, 120, 0.012, 0.005, 1, afr=1),
            marker(2, 150, 0.015, 0.005, 1, eur=1),
        ]
        self.poly = [
            marker(10, 101, 0.0101, 0.1, 3),
            marker(11, 121, 0.0121, 0.1, 3),
            marker(12, 151, 0.0151, 0.1, 3),
        ]
        self.mono = [marker(20, 201, 0.0201, 0.1, 0)]

    def test_joint_assignment_preserves_b0_and_uses_residual_capacity(self):
        result = MODULE.assign_width(
            self.templates, self.rare, self.poly, self.mono, 0.0, 0.05,
            "fixed", 10, [],
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["K"], 2)
        self.assertEqual(result["B0_markers"], 2)
        self.assertEqual(result["B0_ref_polymorphic"], 1)
        self.assertEqual(result["B0_ref_monomorphic"], 1)
        assigned = [row.candidate.site_id for row in result["assignments"]]
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_category_deficit_fails_before_mapping(self):
        result = MODULE.assign_width(
            self.templates, self.rare, self.poly, [], 0.0, 0.05,
            "fixed", 10, [],
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["maximum_mono_deficit"], 1)

    def test_hash_ranking_is_order_independent(self):
        left = sorted(self.rare, key=lambda value: MODULE.stable_rare_key(value, "fixed"))
        right = sorted(reversed(self.rare), key=lambda value: MODULE.stable_rare_key(value, "fixed"))
        self.assertEqual(left, right)


class TestV2Contract(unittest.TestCase):
    def test_contract_preserves_baseline_and_ref1_universe(self):
        contract = json.loads(
            (REPO / "conf" / "m28b_lai_marker_capacity_preregistration.v2.json").read_text()
        )
        self.assertEqual(contract["joint_allocation"]["b0_target_count"], 110074)
        self.assertEqual(contract["joint_allocation"]["b0_ref_polymorphic_target"], 79791)
        self.assertEqual(contract["joint_allocation"]["b0_ref_monomorphic_target"], 30283)
        self.assertIn("at least one", contract["marker_definitions"]["rare_universe"])

    def test_cli_rejects_truth_input(self):
        from unittest import mock

        argv = [
            "audit.py", "--tree-sequence", "a", "--pool-manifest", "b",
            "--genetic-map", "c", "--baseline-template", "d",
            "--m28-preregistration", "e", "--preregistration", "f",
            "--outdir", "g", "--truth", "forbidden",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            MODULE.parse_args()


if __name__ == "__main__":
    unittest.main()
