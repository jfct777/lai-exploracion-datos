#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "modules/33_SAFE_BRIDGE_KAT.nf").read_text(encoding="utf-8")
WORKFLOW = (ROOT / "workflows/m33_safe_bridge_kat.nf").read_text(encoding="utf-8")
CONFIG = (ROOT / "conf/m33_safe_bridge_kat.config").read_text(encoding="utf-8")


class NextflowBoundaryTests(unittest.TestCase):
    def test_process_is_small_fail_closed_and_non_consumable(self) -> None:
        for token in ("cpus 1", "memory '2 GB'", "time '10m'", "cache false",
                      "maxRetries 0", "errorStrategy 'terminate'", "test ! -e"):
            self.assertIn(token, MODULE)
        for forbidden in ("truth", "touch READY", "gs://projects-usp", "gs://teams-usp"):
            self.assertNotIn(forbidden, MODULE)

    def test_workflow_has_only_one_kat_process(self) -> None:
        self.assertEqual(WORKFLOW.count("SAFE_BRIDGE_KAT("), 1)
        self.assertIn("checkIfExists: true", WORKFLOW)

    def test_local_config_is_serial_and_copy_staged(self) -> None:
        for token in ("executor = 'local'", "maxForks = 1", "stageInMode = 'copy'",
                      "docker.enabled = true", "@sha256:",
                      "--network=none --read-only --user 1017:1020"):
            self.assertIn(token, CONFIG)


if __name__ == "__main__":
    unittest.main()
