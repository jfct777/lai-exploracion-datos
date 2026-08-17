"""Invariant tests for M28B-v4 exact matching and frozen validation."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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


def marker(
    site_id: int,
    cm: float,
    maf: float = 0.1,
    ancestry: str = "AFR",
    carriers: int = 2,
):
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
        freq_minor_carrier_individuals=carriers,
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
        contract = {
            "version": 4,
            "stage": "M28B_V4_LAI_OPTIMAL_MATCHING_AUDIT",
            "development": {"fixed_hash_salt": "fixed", "null_replicates": 4},
        }
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


class TestV5CorrectionContract(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            (
                REPO
                / "conf"
                / "m28b_lai_optimal_matching_preregistration.v5.json"
            ).read_text()
        )

    def test_individual_carrier_rule_is_load_bearing(self):
        self.assertEqual(self.contract["version"], 5)
        self.assertEqual(
            self.contract["marker_definitions"][
                "minimum_freq_minor_carrier_individuals"
            ],
            2,
        )
        self.assertIn(
            "change_carrier_rule",
            self.contract["validation"]["prohibited_after_validation"],
        )

    def test_historical_k_is_not_inherited(self):
        self.assertEqual(self.contract["amendment"]["historical_k"], 8694)
        self.assertIn(
            "not inherited",
            self.contract["amendment"]["historical_k_policy"],
        )
        self.assertEqual(
            self.contract["development"]["capacity_fractions"],
            [0.25, 0.5, 0.75, 1.0],
        )
        self.assertNotIn("nested", self.contract["development"]["selection_rule"])

    def test_reproducibility_profile_covers_v5_dev_and_validation(self):
        files = VERIFY.PROFILE_FILES["v5"]
        self.assertEqual(len(files), 14)
        self.assertIn("m28b_v5_frozen_selection.json", files)
        self.assertIn("m28b_v5_validation.public.json", files)
        dev_files = VERIFY.PROFILE_FILES["v5-dev"]
        self.assertEqual(len(dev_files), 8)
        self.assertNotIn("m28b_v5_validation.public.json", dev_files)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run_a, run_b = root / "a", root / "b"
            run_a.mkdir()
            run_b.mkdir()
            for filename in dev_files:
                (run_a / filename).write_text("same", encoding="utf-8")
                (run_b / filename).write_text("same", encoding="utf-8")
            report = VERIFY.verify(
                run_a,
                run_b,
                dev_files,
                "DEV_REPRODUCIBILITY_CONFIRMED",
            )
            self.assertEqual(
                report["decision"], "DEV_REPRODUCIBILITY_CONFIRMED"
            )

    def test_one_homozygous_carrier_is_rejected_but_two_carriers_pass(self):
        m28_contract = {
            "rare_definition": {
                "minimum_mac": 2,
                "maximum_maf_exclusive": 0.01,
            }
        }
        one_carrier = marker(1, 0.1, 2 / 600, carriers=1)
        two_carriers = marker(2, 0.2, 2 / 600, carriers=2)
        self.assertFalse(MODULE.eligible_rare_marker(one_carrier, m28_contract, 2))
        self.assertTrue(MODULE.eligible_rare_marker(two_carriers, m28_contract, 2))

    def test_preflight_auth_requires_passing_reproducibility_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report = {
                "stage": "M28_LAI_SIMULATION_PREFLIGHT",
                "root_seed": 17,
                "pool_allocation_unit": "diploid_individual",
                "pool_disjunction": {"cross_role_individuals": 0},
                "decision": "GO_REPRODUCIBILITY_CHECK",
                "contract_sha256": "contract",
                "gates": {
                    "S0_MAP": True,
                    "S1_REPRODUCIBILITY": None,
                    "S2_DISJUNCTION": True,
                    "S3_PHASE_AND_TRUTH": True,
                    "S4_RARENESS": True,
                    "S5_EXPOSURE": True,
                    "S6_SCOPE": True,
                },
            }
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            manifest = {
                "stage": "M28_LAI_SIMULATION_PREFLIGHT",
                "params": {"root_seed": 17},
                "inputs": {
                    "m28_lai_simulation_preflight_preregistration.v2.json": "contract"
                },
                "sha256": {
                    "m28_sources.trees": "tree",
                    "m28_pools.private.tsv": "pools",
                    "m28_preflight.public.json": MODULE.sha256(report_path),
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            receipt = {
                "stage": "M28_LAI_SIMULATION_PREFLIGHT",
                "gate": "S1_REPRODUCIBILITY",
                "decision": "GO_PREFLIGHT_COMPLETE",
                "passed": True,
                "amendment_sha256": "contract",
                "tree_sequence_check": {"semantic_equality": True},
                "byte_checks": {"one": {"identical": True}},
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            args = SimpleNamespace(
                preflight_report=report_path,
                preflight_manifest=manifest_path,
                preflight_reproducibility=receipt_path,
            )
            expected = {
                "root_seed": 17,
                "tree_sequence_sha256": "tree",
                "pool_manifest_sha256": "pools",
                "preflight_report_sha256": MODULE.sha256(report_path),
                "preflight_manifest_sha256": MODULE.sha256(manifest_path),
            }
            shared = {
                "m28_preflight_contract_sha256": "contract",
                "m28_reproducibility_receipt_sha256": MODULE.sha256(receipt_path),
            }
            MODULE.authenticate_v5_preflight(args, expected, shared, {}, "development")
            receipt["passed"] = False
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            shared["m28_reproducibility_receipt_sha256"] = MODULE.sha256(receipt_path)
            with self.assertRaisesRegex(ValueError, "incomplete or failed"):
                MODULE.authenticate_v5_preflight(
                    args, expected, shared, {}, "development"
                )


if __name__ == "__main__":
    unittest.main()
