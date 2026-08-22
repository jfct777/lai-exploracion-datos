import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules/33_A0_REAL_ADAPTER.nf").read_text(encoding="utf-8")
WORKFLOW = (ROOT / "workflows/m33_a0_real_adapter.nf").read_text(encoding="utf-8")
CONFIG = (ROOT / "conf/m33_a0_real_adapter.config").read_text(encoding="utf-8")


class M33A0NextflowTests(unittest.TestCase):
    def test_two_processes_and_source_auth_precedes_adapter(self):
        self.assertEqual(len(re.findall(r"^process ", MODULE, flags=re.MULTILINE)), 3)
        self.assertIn("M33_A0_AUTHENTICATE_SOURCES.out.auth", WORKFLOW)
        self.assertIn("M33_A0_VALIDATE_INDEXES.out.audit", WORKFLOW)
        self.assertIn("--source-auth ${source_auth}", MODULE)
        self.assertIn("path('m33_a0_index_audit.json')", MODULE)
        self.assertIn("overwrite:false", MODULE)

    def test_root17_only_and_no_scientific_actions(self):
        self.assertIn("root17", WORKFLOW)
        self.assertIn("20260817", WORKFLOW)
        combined = "\n".join((MODULE, WORKFLOW, CONFIG)).lower()
        for forbidden in ("root18/20260818", "truth", "gradient", "checkpoint", "optimizer", "train ", "gcloud", "gsutil"):
            self.assertNotIn(forbidden, combined)

    def test_fail_closed_resources_and_pinned_container(self):
        self.assertIn("executor = 'local'", CONFIG)
        self.assertIn("maxRetries = 0", CONFIG)
        self.assertIn("maxForks = 1", CONFIG)
        self.assertRegex(CONFIG, r"m28-lai-sim@sha256:[0-9a-f]{64}")
        self.assertIn("--network=none", CONFIG)
        self.assertIn("--read-only", CONFIG)
        self.assertIn("docker.enabled = true", CONFIG)


if __name__ == "__main__":
    unittest.main()
