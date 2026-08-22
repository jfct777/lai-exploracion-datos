#!/usr/bin/env python3

import importlib.util
import unittest
import json
import tempfile
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_tabix_kat_cloud_runner", ROOT / "bin" / "m33_tabix_kat_cloud_runner.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CloudRunnerTests(unittest.TestCase):
    def test_controller_spec_is_pinned_labeled_and_bounded(self):
        labels = {"team": "frank", "pipeline": "m33-tabix-kat", "run": "0123456789abcdef"}
        runtime = "runtime@sha256:" + "a" * 64
        controller = "controller@sha256:" + "b" * 64
        spec = MODULE.controller_job_spec(
            run_id="run-id", runtime_image=runtime, controller_image=controller, labels=labels
        )
        self.assertEqual(spec["labels"], labels)
        self.assertEqual(spec["allocationPolicy"]["serviceAccount"]["email"], MODULE.CONTROLLER_SERVICE_ACCOUNT)
        task = spec["taskGroups"][0]["taskSpec"]
        self.assertEqual(task["maxRetryCount"], 0)
        self.assertEqual(task["maxRunDuration"], "1800s")
        self.assertEqual(task["runnables"][0]["container"]["imageUri"], controller)
        self.assertIn(runtime, task["runnables"][0]["container"]["commands"])

    def test_launch_authorization_requires_exact_published_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "launch.json"
            payload = json.loads((ROOT / "conf" / "m33_tabix_kat_launch_authorization.json").read_text())
            payload["status"] = "AUTHORIZED_EXACT_IMMUTABLE_IMAGES"
            payload["controller_image"] = "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-controller@sha256:" + "b" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(MODULE.load_launch_authorization(path, ROOT)["status"], "AUTHORIZED_EXACT_IMMUTABLE_IMAGES")
            payload["controller_image"] = "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-controller@sha256:" + "z" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "controller image"):
                MODULE.load_launch_authorization(path, ROOT)
            payload["controller_image"] = "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-controller@sha256:" + "b" * 64
            payload = json.loads(path.read_text()); payload["status"] = "BLOCKED_PENDING_PUBLISHED_CONTROLLER_DIGEST"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remains blocked"):
                MODULE.load_launch_authorization(path, ROOT)

    def test_team_delta_rejects_unexpected_job_without_run_label(self):
        before = {"old"}
        expected = {"controller", "child-a"}
        after = [
            {"name": "old", "labels": {"team": "frank"}},
            {"name": "controller", "labels": {"team": "frank", "run": "expected"}},
            {"name": "child-a", "labels": {"team": "frank", "run": "expected"}},
        ]
        self.assertEqual(MODULE.validate_team_delta(before, after, expected), expected)
        after.append({"name": "unexpected", "labels": {"team": "frank"}})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            MODULE.validate_team_delta(before, after, expected)

    def test_gcloud_batch_reads_pin_project(self):
        completed = mock.Mock(stdout="[]")
        with mock.patch.object(MODULE.subprocess, "run", return_value=completed) as run:
            self.assertEqual(MODULE.gcloud_json(["batch", "jobs", "list"]), [])
        self.assertIn("--project=uspbr-242713", run.call_args.args[0])

    def test_failure_drain_waits_for_three_stable_terminal_reads(self):
        class Postflight:
            @staticmethod
            def select_run_jobs(jobs, _run_id): return jobs
            @staticmethod
            def inventory_signature(jobs):
                return tuple((row["name"], row["status"]["state"]) for row in jobs)
        active = [{"name": "controller", "status": {"state": "RUNNING"}}]
        terminal = [{"name": "controller", "status": {"state": "FAILED"}}]
        with mock.patch.object(MODULE, "list_jobs", side_effect=[active, terminal, terminal, terminal]), \
             mock.patch.object(MODULE.time, "sleep"):
            observed = MODULE.drain_run_jobs(
                Postflight, run_id="run", controller_job_name="controller", timeout_seconds=10
            )
        self.assertEqual(observed, terminal)


if __name__ == "__main__":
    unittest.main()
