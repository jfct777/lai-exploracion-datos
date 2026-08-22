#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "m33_tabix_kat_controller.py"
SPEC = importlib.util.spec_from_file_location("m33_tabix_kat_controller", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ControllerTests(unittest.TestCase):
    def test_controller_identity_is_exact(self):
        self.assertEqual(
            MODULE.CONTROLLER,
            "dnabr-m33-controller-frank@uspbr-242713.iam.gserviceaccount.com",
        )

    def test_receipt_write_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            MODULE.write_exclusive(path, {"status": "PASS"})
            self.assertEqual(json.loads(path.read_text())["status"], "PASS")
            with self.assertRaises(ValueError):
                MODULE.write_exclusive(path, {"status": "PASS"})

    def test_metadata_identity_rejects_missing_google_header(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.headers.get.return_value = None
        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "metadata response"):
                MODULE.controller_email()

    def test_credential_overrides_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "credential override"):
            MODULE.reject_credential_overrides({"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/key.json"})

    def test_source_inventory_is_closed(self):
        self.assertIn("workflows/m33_tabix_kat.nf", MODULE.SOURCE_FILES)
        self.assertIn("conf/m33_infra_kat_authorization.json", MODULE.SOURCE_FILES)
        self.assertIn("bin/m33_batch_postflight.py", MODULE.SOURCE_FILES)
        self.assertIn("bin/m33_tabix_kat_cloud_runner.py", MODULE.SOURCE_FILES)
        self.assertIn("conf/gcp/m33_batch_submitter_role.yaml", MODULE.SOURCE_FILES)
        self.assertNotIn(".claude/handoff.md", MODULE.SOURCE_FILES)

    def test_real_runner_fixes_v1_parser_and_removes_credential_overrides(self):
        environment = MODULE.nextflow_environment(
            {"PATH": "/bin", "NXF_SYNTAX_PARSER": "v2", "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/key"},
            {"DNABR_RUN_ID": "m33-run"},
        )
        self.assertEqual(environment["NXF_SYNTAX_PARSER"], "v1")
        self.assertEqual(environment["NXF_OFFLINE"], "true")
        self.assertNotIn("GOOGLE_APPLICATION_CREDENTIALS", environment)

    def test_current_source_bundle_is_exactly_authorized(self):
        source_auth, digest = MODULE.validate_source_auth(ROOT)
        self.assertEqual(source_auth.name, "m33_tabix_kat_source_auth.json")
        self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
