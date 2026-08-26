#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "conf" / "m34_nam_inputs.config"
BATCH_CONFIG = ROOT / "conf" / "m34_train_pending_google_batch.config"


class M34TrainPendingGoogleBatchTests(unittest.TestCase):
    def test_batch_config_uses_only_the_project_write_namespace(self):
        text = BATCH_CONFIG.read_text(encoding="utf-8")
        project_root = "gs://teams-usp/frank/lai-exploracion-datos"

        self.assertIn(f"def m34BatchRoot = '{project_root}'", text)
        self.assertIn('m34_inputs_results_dir = "${m34BatchRoot}/runs"', text)
        self.assertIn('workDir = "${m34BatchRoot}/work/nextflow/${m34BatchRunId}"', text)
        self.assertNotIn("gs://projects-usp", text)
        self.assertNotIn("gs://frozen-data-br", text)
        self.assertNotIn("/tmp/", text)

    def test_resources_concurrency_label_and_pinned_image_are_explicit(self):
        text = BATCH_CONFIG.read_text(encoding="utf-8")

        self.assertIn("id 'nf-google@1.27.3'", text)
        self.assertIn("executor = 'google-batch'", text)
        self.assertIn("withName: 'M34_NAM_TRAIN_FACTORIZED'", text)
        self.assertIn("withName: 'M34_NAM_TRAIN_TRANSFORMER_FACTORIZED'", text)
        self.assertIn("withName: 'M34_NAM_SCORE_VALID'", text)
        self.assertIn("m34_inputs_train_cpus = 4", text)
        self.assertIn("m34_inputs_train_memory = '8 GB'", text)
        self.assertIn("m34_inputs_score_cpus = 1", text)
        self.assertIn("m34_inputs_score_memory = '4 GB'", text)
        self.assertIn("maxForks = 4", text)
        self.assertIn("executor.queueSize = 4", text)
        self.assertIn("resourceLabels = [team: 'frank']", text)
        self.assertIn("spot = false", text)
        self.assertIn("docker.enabled = false", text)
        self.assertIn("m34_inputs_container_user = '0:0'", text)
        self.assertRegex(text, r"m33-t0a@sha256:[0-9a-f]{64}")

    def test_local_execution_remains_the_default(self):
        local = LOCAL_CONFIG.read_text(encoding="utf-8")
        self.assertIn("executor = 'local'", local)
        self.assertNotIn("m34_train_pending_google_batch.config", local)

    @unittest.skipUnless(shutil.which("nextflow"), "nextflow is required for config parsing")
    def test_combined_configuration_parses_without_launching(self):
        environment = dict(os.environ)
        environment.update({
            "DNABR_RUN_ID": "m34-nam-batch-config-test",
            "NXF_SYNTAX_PARSER": "v1",
        })
        completed = subprocess.run(
            [
                "nextflow", "-C", f"{LOCAL_CONFIG},{BATCH_CONFIG}",
                "config", "-flat",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        flattened = completed.stdout
        self.assertIn("process.executor = 'google-batch'", flattened)
        self.assertIn("executor.queueSize = 4", flattened)
        self.assertIn("process.resourceLabels = [team:'frank']", flattened)
        self.assertIn("params.m34_inputs_train_cpus = 4", flattened)
        self.assertIn("params.m34_inputs_train_memory = '8 GB'", flattened)
        self.assertIn("params.m34_inputs_score_cpus = 1", flattened)
        self.assertIn("params.m34_inputs_score_memory = '4 GB'", flattened)
        self.assertIn("docker.enabled = false", flattened)
        self.assertIn(
            "workDir = 'gs://teams-usp/frank/lai-exploracion-datos/"
            "work/nextflow/m34-nam-batch-config-test",
            flattened,
        )

    @unittest.skipUnless(shutil.which("nextflow"), "nextflow is required for config parsing")
    def test_invalid_or_missing_run_id_fails_closed(self):
        for run_id in (None, "../escape", "UPPERCASE"):
            environment = dict(os.environ)
            environment["NXF_SYNTAX_PARSER"] = "v1"
            if run_id is None:
                environment.pop("DNABR_RUN_ID", None)
            else:
                environment["DNABR_RUN_ID"] = run_id
            completed = subprocess.run(
                ["nextflow", "-C", str(BATCH_CONFIG), "config", "-flat"],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            with self.subTest(run_id=run_id):
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("DNABR_RUN_ID", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
