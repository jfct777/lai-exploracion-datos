#!/usr/bin/env python3

import os
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "conf" / "m33_tabix_kat.config"
MODULE = ROOT / "modules" / "33_TABIX_KAT.nf"
AUTH = ROOT / "conf" / "m33_infra_kat_authorization.json"
SOURCE_AUTH = ROOT / "conf" / "m33_tabix_kat_source_auth.json"


class TabixKatNextflowTests(unittest.TestCase):
    @staticmethod
    def environment():
        environment = dict(os.environ)
        environment.update({
            "DNABR_M33_INFRA_KAT": "1",
            "DNABR_RUN_ID": "m33-tabix-kat-20260822a",
            "DNABR_M33_TABIX_IMAGE": (
                "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
                "m33-tabix@sha256:" + "a" * 64
            ),
            "NXF_SYNTAX_PARSER": "v1",
        })
        return environment

    def test_config_pins_identity_label_versions_and_personal_workdir(self):
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("id 'nf-google@1.27.3'", text)
        self.assertIn("serviceAccountEmail = 'dnabr-m33-frank@uspbr-242713.iam.gserviceaccount.com'", text)
        self.assertIn("resourceLabels = [team: 'frank']", text)
        self.assertIn("maxForks = 2", text)
        self.assertIn("gs://teams-usp/frank/lai-exploracion-datos", text)
        self.assertNotIn("gs://projects-usp", text)
        self.assertNotIn("gs://frozen-data-br", text)

    def test_direct_launch_is_blocked_without_controller_receipt(self):
        completed = subprocess.run(
            ["nextflow", "-C", str(CONFIG), "config", "-flat"],
            cwd=ROOT, env=self.environment(), capture_output=True, text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("runner controlador", completed.stderr + completed.stdout)

    def test_missing_gate_tag_or_invalid_run_id_stops(self):
        cases = [
            ("DNABR_M33_INFRA_KAT", None),
            ("DNABR_M33_TABIX_IMAGE", "m33-tabix:latest"),
            ("DNABR_RUN_ID", "../escape"),
        ]
        for key, value in cases:
            environment = self.environment()
            if value is None:
                environment.pop(key)
            else:
                environment[key] = value
            completed = subprocess.run(
                ["nextflow", "-C", str(CONFIG), "config", "-flat"],
                cwd=ROOT, env=environment, capture_output=True, text=True,
            )
            with self.subTest(key=key):
                self.assertNotEqual(completed.returncode, 0)

    def test_replicas_are_uncached_independent_tasks(self):
        text = MODULE.read_text(encoding="utf-8")
        self.assertIn("cache false", text)
        self.assertIn("--task-hash ${task.hash}", text)
        self.assertIn("--task-work-uri ${task.workDir}", text)
        self.assertNotIn("publishDir", text)
        self.assertIn("--expected-work-prefix", text)
        self.assertIn("--expected-runtime-service-account", text)
        self.assertIn("--require-cloud", text)

    def test_exact_controller_receipt_opens_config(self):
        environment = self.environment()
        runtime = json.loads(AUTH.read_text(encoding="utf-8"))["runtime"]
        runtime_image = f"{runtime['oci_repository']}@{runtime['oci_digest']}"
        environment["DNABR_M33_TABIX_IMAGE"] = runtime_image
        environment["DNABR_M33_AUTHORIZATION"] = str(AUTH)
        environment["DNABR_M33_SOURCE_AUTH"] = str(SOURCE_AUTH)
        run_id = environment["DNABR_RUN_ID"]
        payload = {
            "stage": "M33_TABIX_KAT_CONTROLLER",
            "status": "PASS_CONTROLLER_IDENTITY_AND_AUTHORIZATION",
            "controller_service_account": "dnabr-m33-controller-frank@uspbr-242713.iam.gserviceaccount.com",
            "authorization_sha256": hashlib.sha256(AUTH.read_bytes()).hexdigest(),
            "runtime_image": runtime_image,
            "nextflow_version": "26.04.6",
            "nf_google_version": "1.27.3",
            "run_id": run_id,
            "work_prefix": f"gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/{run_id}/",
            "source_auth_sha256": hashlib.sha256(SOURCE_AUTH.read_bytes()).hexdigest(),
            "real_asset_read": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "controller.json"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            environment["DNABR_M33_CONTROLLER_RECEIPT"] = str(receipt)
            completed = subprocess.run(
                ["nextflow", "-C", str(CONFIG), "config", "-flat"],
                cwd=ROOT, env=environment, capture_output=True, text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
