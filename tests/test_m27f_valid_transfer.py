#!/usr/bin/env python3
"""Contracts for the one-shot M27F SOURCE_VALID transfer gate."""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bin"))
import audit_m27f_valid_transfer as valid  # noqa: E402


class TestM27FValidTransfer(unittest.TestCase):
    def test_block_states_separate_absence_from_measurement_failure(self):
        self.assertEqual(valid.block_state(4, 1, 0), "PRESENT")
        self.assertEqual(valid.block_state(4, 0, 0), "ABSENT")
        self.assertEqual(valid.block_state(4, 0, 1), "UNEVALUABLE_PHASE")
        self.assertEqual(valid.block_state(0, 0, 0), "UNEVALUABLE_CALLABILITY")

    def test_decision_requires_two_distinct_valid_carrier_patterns(self):
        self.assertEqual(valid.transfer_decision({"a", "b"}, 0), "PASS_LOCAL_TRANSFER")
        self.assertEqual(valid.transfer_decision({"a"}, 1), "INCONCLUSIVE_TECHNICAL")
        self.assertEqual(valid.transfer_decision({"a"}, 0), "STOP_M27_LOCAL_TRANSFER")
        self.assertEqual(valid.transfer_decision(set(), 1), "STOP_M27_LOCAL_TRANSFER")

    def test_preregistration_freezes_actual_valid_split_and_gate(self):
        prereg = json.loads(
            (REPO / "conf/m27f_valid_transfer_preregistration.json").read_text(
                encoding="utf-8"
            )
        )
        upstream = prereg["upstream_contract"]
        transfer = prereg["transfer_contract"]
        self.assertEqual(upstream["expected_valid_native_american_atomic_units"], 3)
        self.assertEqual(upstream["expected_primary_historical_baseline_disjoint_sites"], 3)
        self.assertEqual(upstream["expected_primary_ref_patterns"], 3)
        self.assertFalse(transfer["valid_used_for_selection"])
        self.assertFalse(transfer["source_test_opened"])
        self.assertFalse(transfer["direction_or_enrichment_used_as_gate"])

    def test_projection_is_positive_allowlist_without_variant_filters(self):
        source = (REPO / "bin/project_m27f_valid_panel.py").read_text(encoding="utf-8")
        common = (REPO / "bin/project_m27f_ref_panel.py").read_text(encoding="utf-8")
        self.assertIn('row["role"] == "SOURCE_VALID"', source)
        self.assertIn('"--samples-file"', common)
        self.assertIn('"--no-update"', common)
        self.assertIn('"--no-version"', common)
        self.assertNotIn('"--include"', source + common)
        self.assertNotIn('"--exclude"', source + common)
        self.assertNotIn("KING", source + common)

    def test_nextflow_opens_valid_only_and_keeps_outputs_private(self):
        module = (REPO / "modules/27F_VALID_LOCAL_TRANSFER.nf").read_text(encoding="utf-8")
        workflow = (REPO / "workflows/m27f_valid_local_transfer.nf").read_text(encoding="utf-8")
        config = (REPO / "conf/m27f_valid_transfer.config").read_text(encoding="utf-8")
        self.assertIn("chmod 600 m27f_valid_site_support.private.tsv", module)
        self.assertIn("source_test_opened: false", workflow)
        self.assertIn('m27f_valid_cpus = 2', config)
        self.assertIn('m27f_valid_memory = "4 GB"', config)
        self.assertEqual(module.count("container params.m27f_valid_container_image"), 3)
        self.assertNotIn("KING", module + workflow + config)


if __name__ == "__main__":
    unittest.main()
