#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "m33_m0_factorized_lazy_amendment",
    ROOT / "bin" / "m33_m0_factorized_lazy_amendment.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FactorizedLazyAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.contract_path = ROOT / "conf" / "m33_m0_factorized_lazy_amendment_contract.json"
        self.materializer_path = ROOT / "conf" / "m33_m0_materializer_contract.json"
        self.sanitizer_path = ROOT / "conf" / "m33_m0_f0_sanitized_amendment_contract.json"
        self.pre4_path = ROOT / "conf" / "m33_pre4_preregistration.json"
        self.contract = MODULE.load_json(self.contract_path)
        self.materializer = MODULE.load_json(self.materializer_path)
        self.sanitizer = MODULE.load_json(self.sanitizer_path)
        self.pre4 = MODULE.load_json(self.pre4_path)

    def validate(self, contract=None):
        MODULE.validate_contract(contract or self.contract, self.materializer, self.sanitizer,
                                 self.pre4, self.materializer_path, self.sanitizer_path,
                                 self.pre4_path)

    def test_contract_passes_and_cli_receipt_stays_contract_only(self):
        self.validate()
        with tempfile.TemporaryDirectory() as raw:
            receipt = Path(raw) / "receipt.json"
            status = MODULE.main([
                "--contract", str(self.contract_path),
                "--materializer-contract", str(self.materializer_path),
                "--sanitized-f0-contract", str(self.sanitizer_path),
                "--pre4-preregistration", str(self.pre4_path),
                "--output", str(receipt),
            ])
            self.assertEqual(status, 0)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"],
                             "PASS_CONTRACT_AND_SYNTHETIC_EQUIVALENCE_GATE_ONLY")
            self.assertFalse(payload["real_root_execution"])
            self.assertFalse(payload["training"])

    def test_rejects_persistent_expanded_arrays(self):
        changed = copy.deepcopy(self.contract)
        changed["physical_amendment"]["persistent_packed_rare_context_shard"] = True
        with self.assertRaisesRegex(ValueError, "expanded persistence"):
            self.validate(changed)

    def test_rejects_projected_byte_drift(self):
        changed = copy.deepcopy(self.contract)
        changed["proportionality_gate"]["projected_minimum_token_array_bytes_all_copies"] += 1
        with self.assertRaisesRegex(ValueError, "storage estimate"):
            self.validate(changed)

    def test_rejects_output_to_lab_datalake(self):
        changed = copy.deepcopy(self.contract)
        changed["physical_amendment"]["output_namespace_template"] = "gs://projects-usp/dnaBr-lai/datalake/"
        with self.assertRaisesRegex(ValueError, "project bucket"):
            self.validate(changed)

    def test_rejects_score_only_in_fit(self):
        changed = copy.deepcopy(self.contract)
        changed["unchanged_scientific_contract"]["rotations"]["R0"]["fit_root_seeds"][0] = 386357765
        with self.assertRaisesRegex(ValueError, "scientific design"):
            self.validate(changed)

    def test_rejects_weakened_ref_label_sham(self):
        changed = copy.deepcopy(self.contract)
        changed["ref_label_sham_dependency"]["modeling_blocked_until_resolved"] = False
        with self.assertRaisesRegex(ValueError, "sham was weakened"):
            self.validate(changed)

    def test_rejects_order_and_stop_rule_drift(self):
        changed = copy.deepcopy(self.contract)
        changed["lazy_reconstruction"]["row_order"] = "marker_major"
        with self.assertRaisesRegex(ValueError, "ordering"):
            self.validate(changed)
        changed = copy.deepcopy(self.contract)
        changed["interval_artifact"]["ordering"] = "arbitrary"
        with self.assertRaisesRegex(ValueError, "interval contract"):
            self.validate(changed)
        changed = copy.deepcopy(self.contract)
        changed["stop_rules"] = []
        with self.assertRaisesRegex(ValueError, "stop rules"):
            self.validate(changed)

    def test_rejects_channel_and_publication_schema_drift(self):
        changed = copy.deepcopy(self.contract)
        changed["lazy_reconstruction"]["channel_semantics"] = "anything"
        with self.assertRaisesRegex(ValueError, "channel semantics"):
            self.validate(changed)
        mutations = (
            ("factorized_root_receipt_required_keys", "status", "root receipt schema"),
            ("lazy_context_recipe_required_keys", "role_in_rotation", "lazy recipe schema"),
            ("factorized_READY_required_keys", "status", "READY schema"),
        )
        for field, removed, message in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.contract)
                changed["publication_contract"][field].remove(removed)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(changed)


if __name__ == "__main__":
    unittest.main()
