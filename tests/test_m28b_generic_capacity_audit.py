"""Invariant tests for the generic M28B-v3 comparator allocation."""

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
    "m28b_generic_capacity_audit", BIN / "m28b_generic_capacity_audit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_m28b_reproducibility_v3", BIN / "verify_m28b_reproducibility.py"
)
VERIFY = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFY)


def marker(site_id: int, bp: int, cm: float, maf: float, ref: int, ancestry: str = "AFR"):
    kwargs = {"ref_minor_afr": 0, "ref_minor_eur": 0, "ref_minor_asia": 0}
    kwargs[f"ref_minor_{ancestry.lower()}"] = 1
    return MODULE.Marker(
        site_id=site_id,
        bp=bp,
        cm=cm,
        minor_code=1,
        mac=2,
        an=600,
        maf=maf,
        ref_minor_total=ref,
        **kwargs,
    )


class TestHamiltonAllocation(unittest.TestCase):
    def test_exact_target_and_deterministic_ties(self):
        quotas = MODULE.hamilton_quotas({0: 3, 1: 3, 2: 4}, 7)
        self.assertEqual(sum(quotas.values()), 7)
        self.assertEqual(quotas, {0: 2, 1: 2, 2: 3})

    def test_rejects_infeasible_target(self):
        with self.assertRaises(ValueError):
            MODULE.hamilton_quotas({0: 2}, 3)


class TestGenericAllocation(unittest.TestCase):
    def setUp(self):
        self.common = [
            marker(index, index * 10, index / 1000, 0.10, 3)
            for index in range(1, 13)
        ]
        self.rare = [
            marker(101, 25, 0.0025, 0.005, 1, "AFR"),
            marker(102, 55, 0.0055, 0.005, 1, "EUR"),
            marker(103, 85, 0.0085, 0.005, 1, "ASIA"),
        ]

    def test_exact_b0_and_arm_parity(self):
        result = MODULE.assign_width(
            self.rare, self.rare, self.common, 0.0, 0.05, 8, "fixed", 8
        )
        self.assertEqual(result["B0"], 8)
        self.assertEqual(result["K"], 3)
        self.assertTrue(result["b0_pass"])
        self.assertTrue(result["parity_pass"])
        self.assertTrue(result["ancestry_pass"])
        b0 = {value.site_id for value in result["b0_markers"]}
        controls = {value.site_id for value in result["control_markers"]}
        self.assertFalse(b0 & controls)

    def test_hash_selection_is_reproducible(self):
        left = MODULE.assign_width(
            self.rare, self.rare, self.common, 0.0, 0.05, 8, "fixed", 8
        )
        right = MODULE.assign_width(
            list(reversed(self.rare)), self.rare, list(reversed(self.common)),
            0.0, 0.05, 8, "fixed", 8,
        )
        self.assertEqual(
            [value.site_id for value in left["b0_markers"]],
            [value.site_id for value in right["b0_markers"]],
        )
        self.assertEqual(
            [value.site_id for value in left["rare_markers"]],
            [value.site_id for value in right["rare_markers"]],
        )

    def test_monte_carlo_rank_has_finite_sample_correction(self):
        self.assertEqual(MODULE.monte_carlo_rank(5.0, [1.0, 2.0, 6.0]), 0.5)


class TestV3Contract(unittest.TestCase):
    def test_scope_and_anchor_are_frozen(self):
        contract = json.loads(
            (REPO / "conf" / "m28b_lai_generic_capacity_preregistration.json").read_text()
        )
        self.assertEqual(contract["allocation"]["b0_target_count"], 79791)
        self.assertEqual(contract["allocation"]["null_replicates"], 32)
        self.assertIn("positions are not used", contract["analysis_interval"]["rationale"])
        self.assertIn("TARGET", contract["pool_access"]["forbidden"])

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

    def test_reproducibility_profile_covers_every_v3_scientific_output(self):
        self.assertEqual(len(VERIFY.PROFILE_FILES["v3"]), 7)
        self.assertIn("m28b_v3_common_common_null.tsv", VERIFY.PROFILE_FILES["v3"])


if __name__ == "__main__":
    unittest.main()
