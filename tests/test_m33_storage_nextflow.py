#!/usr/bin/env python3

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "conf" / "m33_storage_namespace.config"


class StorageNextflowConfigTests(unittest.TestCase):
    def test_config_has_only_personal_persistent_root(self):
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", text)
        self.assertNotIn("gs://projects-usp", text)
        self.assertNotIn("gs://frozen-data-br", text)
        self.assertIn("workDir = m33WorkRoot", text)
        self.assertIn("m33_batch_resource_labels = [team: 'frank']", text)
        self.assertIn("m33_real_run_authorized = false", text)
        self.assertNotIn("executor = 'google-batch'", text)

    def test_nextflow_resolves_all_sinks_under_personal_root(self):
        env = dict(os.environ)
        env["DNABR_RUN_ID"] = "m33-storage-fixture-20260822a"
        env["NXF_SYNTAX_PARSER"] = "v1"
        env["DNABR_M33_CONTRACT_INSPECTION"] = "1"
        completed = subprocess.run([
            "nextflow", "-C", str(CONFIG), "config", "-flat"
        ], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        output = completed.stdout
        root = "gs://teams-usp/frank/lai-exploracion-datos/"
        self.assertIn(f"workDir = '{root}work/nextflow/m33-storage-fixture-20260822a'", output)
        for key in ("m33_results_dir", "m33_work_dir", "m33_logs_dir", "m33_manifests_dir"):
            line = next(line for line in output.splitlines() if line.startswith(f"params.{key} ="))
            self.assertIn(root, line)
        self.assertIn("params.m33_batch_resource_labels = [team:'frank']", output)
        self.assertIn("params.m33_real_run_authorized = false", output)
        self.assertIn(
            f"params.m33_storage_policy = '{ROOT}/conf/m33_storage_namespace_policy.json'",
            output,
        )

    def test_invalid_run_id_stops_config_resolution(self):
        env = dict(os.environ)
        env["DNABR_RUN_ID"] = "../escape"
        env["NXF_SYNTAX_PARSER"] = "v1"
        env["DNABR_M33_CONTRACT_INSPECTION"] = "1"
        completed = subprocess.run([
            "nextflow", "-C", str(CONFIG), "config", "-flat"
        ], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("DNABR_RUN_ID no es válido", completed.stderr + completed.stdout)

    def test_config_is_blocked_without_contract_inspection_gate(self):
        env = dict(os.environ)
        env["DNABR_RUN_ID"] = "m33-storage-fixture-20260822a"
        env["NXF_SYNTAX_PARSER"] = "v1"
        env.pop("DNABR_M33_CONTRACT_INSPECTION", None)
        completed = subprocess.run([
            "nextflow", "-C", str(CONFIG), "config", "-flat"
        ], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("M33 permanece bloqueado", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
