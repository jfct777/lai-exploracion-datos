#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules" / "33_I0_FIXTURE.nf").read_text(encoding="utf-8")
WORKFLOW = (ROOT / "workflows" / "m33_i0_fixture.nf").read_text(encoding="utf-8")
CONFIG = (ROOT / "conf" / "m33_i0_fixture.config").read_text(encoding="utf-8")
AUTH = (ROOT / "conf" / "m33_i0_fixture_authorization.json").read_text(encoding="utf-8")


class M33I0NextflowTests(unittest.TestCase):
    def test_workflow_is_fixture_only_and_nextflow_first(self):
        self.assertEqual(len(re.findall(r"^process ", MODULE, flags=re.MULTILINE)), 2)
        self.assertIn("M33_I0_MAKE_FIXTURE.out", WORKFLOW)
        self.assertIn("I0_FIXTURE_PASS", MODULE)
        self.assertNotIn("READY", MODULE)
        self.assertIn("m33_i0_fixture_source_auth", WORKFLOW)

    def test_execution_is_local_serial_and_container_pinned(self):
        self.assertIn("executor = 'local'", CONFIG)
        self.assertIn("maxForks = 1", CONFIG)
        self.assertIn("maxRetries = 0", CONFIG)
        self.assertRegex(CONFIG, r"m33-tabix@sha256:[0-9a-f]{64}")
        self.assertIn("--network none", CONFIG)
        self.assertEqual(MODULE.count("--container-image '${task.container}'"), 2)

    def test_no_real_asset_or_cloud_sink_is_reachable(self):
        combined = "\n".join((MODULE, WORKFLOW, CONFIG, AUTH)).lower()
        for forbidden in (
            "gs://projects-usp", "frozen-data-br", "google-batch", "gsutil", "gcloud",
            "safe_bridge true", "materialize true", "training true", "path truth",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn('"real_asset_read": false', AUTH)
        self.assertIn('"global_ready_forbidden": true', AUTH)
        self.assertIn("m33_i0_fixture_enabled?.toString() != 'true'", WORKFLOW)

    def test_processes_disable_cache_for_independent_indexing(self):
        self.assertEqual(len(re.findall(r"cache false", MODULE)), 2)
        self.assertIn("chmod a-w", MODULE)


if __name__ == "__main__":
    unittest.main()
