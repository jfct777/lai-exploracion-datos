#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "conf" / "m34_nam_inputs.config"
BATCH_CONFIG = ROOT / "conf" / "m34_train_pending_google_batch.config"
WORKFLOW = ROOT / "workflows" / "m34_train_pending.nf"


class M34TrainPendingGoogleBatchTests(unittest.TestCase):
    def test_batch_config_uses_only_the_project_write_namespace(self):
        text = BATCH_CONFIG.read_text(encoding="utf-8")
        project_root = "gs://teams-usp/frank/lai-exploracion-datos"

        self.assertIn(f"m34_batch_root = '{project_root}'", text)
        self.assertIn('m34_inputs_results_dir = "${params.m34_batch_root}/runs"', text)
        self.assertIn(
            'workDir = "${params.m34_batch_root}/work/nextflow/'
            '${params.m34_inputs_run_id}"',
            text,
        )
        self.assertNotIn("def m34Batch", text)
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
        environment["DNABR_RUN_ID"] = "m34-nam-batch-config-test"
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
        self.assertIn("process.resourceLabels.team = 'frank'", flattened)
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
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            overlay = temporary_path / "local-parser.config"
            overlay.write_text(
                "process.executor = 'local'\n"
                "docker.enabled = false\n"
                f"workDir = '{temporary_path / 'work'}'\n"
                "params.m34_inputs_results_dir = '"
                f"{temporary_path / 'results'}'\n",
                encoding="utf-8",
            )
            for run_id in (None, "../escape", "UPPERCASE"):
                environment = dict(os.environ)
                if run_id is None:
                    environment.pop("DNABR_RUN_ID", None)
                else:
                    environment["DNABR_RUN_ID"] = run_id
                completed = subprocess.run(
                    [
                        "nextflow", "-C",
                        f"{LOCAL_CONFIG},{BATCH_CONFIG},{overlay}",
                        "run", str(WORKFLOW),
                    ],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                with self.subTest(run_id=run_id):
                    output = completed.stderr + completed.stdout
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertNotIn("Config parsing failed", output)
                    self.assertIn(
                        "m34_inputs_run_id must be a valid explicit run identifier",
                        output,
                    )

    @unittest.skipUnless(shutil.which("nextflow"), "nextflow is required for config parsing")
    def test_v2_run_parser_reaches_the_workflow_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            overlay = temporary_path / "local-parser.config"
            overlay.write_text(
                "process.executor = 'local'\n"
                "docker.enabled = false\n"
                f"workDir = '{temporary_path / 'work'}'\n"
                "params.m34_inputs_results_dir = '"
                f"{temporary_path / 'results'}'\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["DNABR_RUN_ID"] = "m34-v2-run-parser-test"
            completed = subprocess.run(
                [
                    "nextflow", "-C",
                    f"{LOCAL_CONFIG},{BATCH_CONFIG},{overlay}",
                    "run", str(WORKFLOW),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = completed.stderr + completed.stdout
            self.assertNotEqual(completed.returncode, 0)
            self.assertNotIn("Config parsing failed", output)
            self.assertIn(
                "results, factor bundle, adaptive contract and pending plan are required",
                output,
            )


if __name__ == "__main__":
    unittest.main()
