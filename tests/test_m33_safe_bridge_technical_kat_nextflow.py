import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TechnicalKatNextflowTests(unittest.TestCase):
    def test_workflow_requires_both_roots_and_exact_commit(self):
        workflow = (ROOT / "workflows/m33_safe_bridge_technical_kat.nf").read_text()
        self.assertIn("tuple('root17', 20260817", workflow)
        self.assertIn("tuple('root18', 20260818", workflow)
        self.assertIn("/[0-9a-f]{40}/", workflow)

    def test_process_is_unprivileged_offline_and_read_only(self):
        module = (ROOT / "modules/33_SAFE_BRIDGE_TECHNICAL_KAT.nf").read_text()
        config = (ROOT / "conf/m33_safe_bridge_technical_kat.config").read_text()
        self.assertIn("setpriv --reuid=65534 --regid=65534", module)
        self.assertIn("chmod 0500 '${root_label}.technical_kat'", module)
        self.assertIn("env -u HOME -u GOOGLE_APPLICATION_CREDENTIALS -u CLOUDSDK_CONFIG", module)
        self.assertIn("--network=none", config)
        self.assertIn("--read-only", config)
        self.assertIn("stageInMode = 'copy'", config)
        self.assertIn("maxForks = 2", config)
        self.assertIn("cache false", module)
        self.assertIn("maxRetries = 0", config)

    def test_no_truth_materialize_ready_or_gcs_output(self):
        combined = "\n".join(
            (ROOT / path).read_text()
            for path in (
                "modules/33_SAFE_BRIDGE_TECHNICAL_KAT.nf",
                "workflows/m33_safe_bridge_technical_kat.nf",
                "conf/m33_safe_bridge_technical_kat.config",
            )
        ).lower()
        for forbidden in ("mosaic_events", "lai_truth", "materialize_tensor", "write_ready", "gs://projects-usp/"):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
