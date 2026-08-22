#!/usr/bin/env python3

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "m33_infra_kat_contract.py"
AUTH = ROOT / "conf" / "m33_infra_kat_authorization.json"
STORAGE = ROOT / "conf" / "m33_storage_namespace_policy.json"
M0 = ROOT / "conf" / "m33_m0_materializer_contract.json"
SPEC = importlib.util.spec_from_file_location("m33_infra_kat_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class InfraKatContractTests(unittest.TestCase):
    def setUp(self):
        self.payload = MODULE.load_json(AUTH)

    def validate(self, payload=None, **kwargs):
        return MODULE.validate_authorization(
            self.payload if payload is None else payload,
            storage_policy=STORAGE,
            m0_contract=M0,
            **kwargs,
        )

    def test_current_authorization_has_published_immutable_digest(self):
        result = self.validate()
        self.assertFalse(result["authorization"]["lab_asset_read"])
        self.assertFalse(result["authorization"]["training"])
        self.validate(require_published_digest=True)
        self.assertRegex(result["runtime"]["oci_digest"], r"^sha256:[0-9a-f]{64}$")

    def test_real_assets_and_scientific_steps_remain_forbidden(self):
        for key in (
            "lab_asset_read", "root17_read", "root18_read", "eval_read",
            "derived_real_index_write", "safe_bridge", "materialization", "training",
        ):
            changed = copy.deepcopy(self.payload)
            changed["authorization"][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed)

    def test_identity_label_and_namespace_are_exact(self):
        mutations = [
            ("gcp", "runtime_service_account", "batch-genomics-worker@uspbr-242713.iam.gserviceaccount.com"),
            ("gcp", "controller_service_account", "653458115080-compute@developer.gserviceaccount.com"),
            ("gcp", "managed_folder", "gs://teams-usp/nathan/"),
            ("gcp", "resource_labels", {"team": "other"}),
            ("storage_permissions", "work_allows_delete", True),
            ("storage_permissions", "work_allows_update", True),
            ("storage_permissions", "append_only_precondition", "none"),
            ("storage_permissions", "ready_is_last", False),
        ]
        for section, key, value in mutations:
            changed = copy.deepcopy(self.payload)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                self.validate(changed)

    def test_versions_and_digest_are_fail_closed(self):
        for key, value in (
            ("tabix_version", "1.20"),
            ("nextflow_version", "latest"),
            ("nf_google_version", "latest"),
            ("oci_digest", "m33-tabix:latest"),
        ):
            changed = copy.deepcopy(self.payload)
            changed["runtime"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(changed)

    def test_parent_contract_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed_storage = Path(temporary) / "storage.json"
            changed_storage.write_text(STORAGE.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parent contract hash"):
                MODULE.validate_authorization(
                    self.payload, storage_policy=changed_storage, m0_contract=M0
                )

    def test_duplicate_json_keys_are_rejected(self):
        text = AUTH.read_text(encoding="utf-8").replace(
            '"schema_version": "1.0.0",',
            '"schema_version": "1.0.0", "schema_version": "1.0.0",',
            1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                MODULE.load_json(path)


if __name__ == "__main__":
    unittest.main()
