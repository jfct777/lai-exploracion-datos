"""Invariant tests for M28B-v4 exact matching and frozen validation."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location(
    "m28b_optimal_matching_audit", BIN / "m28b_optimal_matching_audit.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_m28b_reproducibility_v4", BIN / "verify_m28b_reproducibility.py"
)
VERIFY = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFY)


def marker(site_id: int, cm: float, maf: float = 0.1, ancestry: str = "AFR"):
    kwargs = {"ref_minor_afr": 0, "ref_minor_eur": 0, "ref_minor_asia": 0}
    kwargs[f"ref_minor_{ancestry.lower()}"] = 1
    return MODULE.Marker(
        site_id=site_id,
        bp=site_id * 10,
        cm=cm,
        minor_code=1,
        mac=2,
        an=600,
        maf=maf,
        ref_minor_total=3,
        **kwargs,
    )


class TestExactMatching(unittest.TestCase):
    def test_selects_global_minimum_subsequence(self):
        queries = [marker(1, 4.0), marker(2, 5.0)]
        candidates = [marker(10, 0.0), marker(11, 4.0), marker(12, 100.0)]
        pairs = MODULE.optimal_subsequence_pairs(queries, candidates)
        self.assertEqual([pair.control.site_id for pair in pairs], [10, 11])
        self.assertEqual(sum(abs(pair.query_cm - pair.control.cm) for pair in pairs), 5.0)

    def test_equal_cost_keeps_earlier_ordered_candidate(self):
        pairs = MODULE.optimal_subsequence_pairs([marker(1, 5.0)], [marker(10, 4.0), marker(11, 6.0)])
        self.assertEqual(pairs[0].control.site_id, 10)

    def test_rejects_insufficient_control_capacity(self):
        with self.assertRaises(ValueError):
            MODULE.optimal_subsequence_pairs([marker(1, 1.0), marker(2, 2.0)], [marker(3, 1.0)])

    def test_dynamic_program_matches_brute_force_cost(self):
        queries = [marker(1, 1.5), marker(2, 4.5), marker(3, 9.0)]
        candidates = [
            marker(10, 0.0), marker(11, 2.0), marker(12, 3.0),
            marker(13, 8.0), marker(14, 10.0),
        ]
        pairs = MODULE.optimal_subsequence_pairs(queries, candidates)
        observed = sum(abs(pair.query_cm - pair.control.cm) for pair in pairs)
        expected = min(
            sum(abs(query.cm - control.cm) for query, control in zip(queries, subset))
            for subset in itertools.combinations(candidates, len(queries))
        )
        self.assertEqual(observed, expected)


class TestFrozenCapacity(unittest.TestCase):
    def test_exact_k_is_distributed_with_caps(self):
        quotas = MODULE.allocate_exact_k({0: 2, 1: 6, 2: 2}, 5)
        self.assertEqual(sum(quotas.values()), 5)
        self.assertTrue(all(quotas[key] <= value for key, value in {0: 2, 1: 6, 2: 2}.items()))

    def test_validation_capacity_cannot_reduce_frozen_k(self):
        with self.assertRaises(ValueError):
            MODULE.allocate_exact_k({0: 2, 1: 2}, 5)

    def test_evaluation_preserves_k_and_ancestry(self):
        b0 = [marker(index, index / 1000) for index in range(10, 16)]
        reserve = [marker(index, index / 1000) for index in range(20, 24)]
        rare = [
            marker(101, 0.0205, 0.005, "AFR"),
            marker(102, 0.0215, 0.005, "EUR"),
            marker(103, 0.0225, 0.005, "ASIA"),
        ]
        prepared = {
            "capacity": {0: 3},
            "rare_bins": {0: rare},
            "b0_bins": {0: b0},
            "reserve_bins": {0: reserve},
            "b0": b0,
        }
        contract = {"development": {"fixed_hash_salt": "fixed", "null_replicates": 4}}
        result = MODULE.evaluate_configuration(prepared, {0: 3}, contract)
        self.assertEqual(result["K"], 3)
        self.assertTrue(result["parity_pass"])
        self.assertTrue(result["ancestry_pass"])


class TestV4Contract(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (REPO / "conf" / "m28b_lai_optimal_matching_preregistration.json").read_text()
        )

    def test_dev_and_validation_are_independent_and_frozen(self):
        self.assertEqual(self.contract["development_inputs"]["root_seed"], 20260817)
        self.assertEqual(self.contract["validation_inputs"]["root_seed"], 20260818)
        self.assertNotEqual(
            self.contract["development_inputs"]["tree_sequence_sha256"],
            self.contract["validation_inputs"]["tree_sequence_sha256"],
        )
        self.assertEqual(self.contract["development"]["capacity_fractions"], [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(self.contract["development"]["bin_width_cm"], 0.05)

    def test_forbidden_inputs_are_not_cli_options(self):
        from unittest import mock

        argv = [
            "audit.py", "--phase", "development", "--tree-sequence", "a",
            "--pool-manifest", "b", "--genetic-map", "c",
            "--baseline-template", "d", "--m28-preregistration", "e",
            "--preregistration", "f", "--outdir", "g", "--truth", "forbidden",
        ]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            MODULE.parse_args()

    def test_reproducibility_profile_covers_dev_and_validation(self):
        files = VERIFY.PROFILE_FILES["v4"]
        self.assertEqual(len(files), 14)
        self.assertIn("m28b_v4_frozen_selection.json", files)
        self.assertIn("m28b_v4_validation.public.json", files)


if __name__ == "__main__":
    unittest.main()
