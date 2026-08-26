#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M34ModelSmokeNextflowTests(unittest.TestCase):
    def test_workflow_is_generic_parallel_and_digest_pinned(self):
        workflow = (ROOT / "workflows/m34_model_smoke.nf").read_text(encoding="utf-8")
        module = (ROOT / "modules/34_MODEL_SMOKE.nf").read_text(encoding="utf-8")
        config = (ROOT / "conf/m34_model_smoke.config").read_text(encoding="utf-8")
        combined = workflow + module + config
        self.assertIn("contract.families.each", workflow)
        self.assertIn("maxForks params.m34_smoke_max_forks", module)
        self.assertIn("--arm both", module)
        self.assertIn("@sha256:", workflow)
        self.assertIn("--network none", config)
        self.assertIn("m34_smoke_container_user", config)
        self.assertNotIn("ASIA", combined)
        self.assertNotIn("sample_count = 30", combined)
        self.assertNotIn("2.619", combined)

    def test_nextflow_configuration_parses(self):
        command = [
            "nextflow", "-C", str(ROOT / "conf/m34_model_smoke.config"),
            "config", "-flat",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
