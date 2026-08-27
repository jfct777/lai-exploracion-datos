#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "conf/m34_nam_inputs.config"
BATCH = ROOT / "conf/m34_nam_128_google_batch.config"


class M34Nam128GoogleBatchTests(unittest.TestCase):
    def test_own_bucket_labels_and_bounded_parallelism(self) -> None:
        text = BATCH.read_text(encoding="utf-8")
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", text)
        self.assertNotIn("gs://projects-usp", text)
        self.assertNotIn("gs://frozen-data-br", text)
        self.assertIn("resourceLabels = [team: 'frank']", text)
        self.assertIn("executor.queueSize = 2", text)
        self.assertIn("maxForks = 2", text)
        self.assertIn("spot = false", text)
        self.assertIn("maxRetries = 0", text)
        self.assertIn("docker.enabled = false", text)
        self.assertRegex(text, r"m33-t0a@sha256:[0-9a-f]{64}")

    @unittest.skipUnless(shutil.which("nextflow"), "nextflow is required")
    def test_combined_configuration_parses(self) -> None:
        environment = dict(os.environ)
        environment["DNABR_RUN_ID"] = "m34-nam-128-config-test"
        completed = subprocess.run(
            ["nextflow", "-C", f"{LOCAL},{BATCH}", "config", "-flat"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("process.executor = 'google-batch'", completed.stdout)
        self.assertIn("executor.queueSize = 2", completed.stdout)
        self.assertIn("process.resourceLabels.team = 'frank'", completed.stdout)


if __name__ == "__main__":
    unittest.main()
