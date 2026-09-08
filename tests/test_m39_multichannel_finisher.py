"""Detached finalizer contracts; no real Docker, tmux, cloud or model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import m39_multichannel_finisher as F


class MultichannelFinisherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="m39-finisher-test-")
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        private = self.repo / ".claude/runs"
        self.run = private / "campaign"
        self.source = private / "frozen/bin"
        self.calibration = private / "calibration"
        self.stages = {"screen": private / "m39-gpu-fixture-a", "followup": private / "m39-gpu-fixture-b"}
        for directory in (self.run, self.source, self.calibration, *self.stages.values()):
            directory.mkdir(parents=True)
        for name in ("m39_multichannel_finisher.py", "m39_ordered_multichannel_report.py",
                     "m39_ordered_multichannel_sweep.py", "m39_ordered_multichannel_training.py"):
            (self.source / name).write_text("# frozen artificial source\n")
        F.write_json(self.calibration / "report.json", {"fixture": True})
        plan = self.run / "plan-a/plan.json"
        plan.parent.mkdir()
        F.write_json(plan, {})
        self.cfg = {"schema_version": "m39-multichannel-gpu-campaign-v1", "repository": str(self.repo),
            "run_dir": str(self.run), "source_commit": "a" * 40,
            "source_sha256": {"bin/m39_ordered_multichannel_training.py": F.sha256(self.source / "m39_ordered_multichannel_training.py")},
            "native_auth_dir": str(self.repo / "auth"), "service_account": "fixture@project.iam.gserviceaccount.com",
            "cpu_image": "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/cpu@sha256:" + "b" * 64,
            "gpu_image": "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/gpu@sha256:" + "c" * 64,
            "wall_timeout_seconds": 60,
            "screen": {"run_dir": str(self.stages["screen"]), "plan": str(plan)},
            "followup": {"run_dir": str(self.stages["followup"])}}
        self.path = self.run / "campaign.json"
        F.write_json(self.path, self.cfg)
        self.bound = F.configuration(self.path, self.source, self.calibration, F.sha256(self.calibration / "report.json"))
        self.args = argparse.Namespace(campaign=self.path)
        self.launch = {"campaign_sha256": F.sha256(self.path), "source_sha256": F.source_inventory(self.source),
            "calibration_report_sha256": F.sha256(self.calibration / "report.json"),
            "wait_seconds": 10, "poll_seconds": 1, "report_timeout_seconds": 30}
        F.write_json(self.run / "finisher.launch.json", self.launch)

    def terminal(self, status=F.SUCCESS):
        terminal = {"schema_version": "m39-ordered-campaign-completion-v1", "status": status,
            "source_commit": self.cfg["source_commit"], "campaign_sha256": F.sha256(self.path), "SCORE_opened": False}
        for short, long in (("a", "screen"), ("b", "followup")):
            comparison = self.run / f"cpu-audit-{short}/result/comparison.json"
            comparison.parent.mkdir(parents=True)
            F.write_json(comparison, {})
            terminal[f"comparison_{short}"] = str(comparison)
            (self.run / f"primary-{short}").mkdir()
            receipt = self.stages[long] / "gpu.launch.json"
            F.write_json(receipt, {"run_id": self.stages[long].name, "source_commit": self.cfg["source_commit"],
                "process_name": "M39_ORDERED_GPU_TRAINING", "SCORE_staged": False, "resources": {"max_jobs": 1}})
            F.write_json(self.run / f"event-001-stage-{short}-start.json", {"event": f"stage-{short}-start", "receipt": str(receipt)})
            F.write_json(self.run / f"event-002-stage-{short}-downloaded.json",
                         {"event": f"stage-{short}-downloaded", "outputs": str(self.run / f"primary-{short}")})
        followup_plan = self.run / "cpu-prepare-b/result/plan.json"
        followup_plan.parent.mkdir(parents=True)
        F.write_json(followup_plan, {})
        F.write_json(self.run / "campaign.completion.json", terminal)
        return terminal

    def fake_report(self, *args, **kwargs):
        output = self.run / "final-reading/report"
        output.mkdir()
        F.write_json(output / "report.json", {"schema_version": "m39-ordered-multichannel-report-v1",
            "status": "AUDITED_EXPLORATORY_SELECT_ONLY", "counts": {"screen_fits": 12, "followup_fits": 20},
            "generator_sha256": F.sha256(self.source / "m39_ordered_multichannel_report.py")})
        for name in F.REPORT_FILES[1:]:
            (output / name).write_text("fixture\n")

    def test_success_reports_both_audited_stages_then_checks_owned_workers(self):
        self.terminal()
        events = []
        def report(*args, **kwargs):
            events.append("report")
            self.fake_report()
        def workers(*args, **kwargs):
            events.append("workers")
            return {"status": "OWNED_JOBS_TERMINAL_WORKERS_RETIRED"}
        with patch.object(F.subprocess, "run", side_effect=report), patch.object(F, "owned_workers", side_effect=workers):
            result = F.watch(self.args, self.bound)
        self.assertEqual(events, ["report", "workers"])
        self.assertEqual(result["status"], "FINAL_REPORT_VERIFIED")
        self.assertEqual(set(result["report_files_sha256"]), set(F.REPORT_FILES))
        self.assertFalse(result["SCORE_opened"])
        self.assertEqual(F.read_json(self.run / "finisher.completion.json"), result)

    def test_failed_or_budget_stopped_campaign_does_not_report_or_launch_training(self):
        self.terminal("SCREEN_AUDITED_FOLLOWUP_STOPPED_BY_BUDGET")
        with patch.object(F.subprocess, "run") as launch, patch.object(F, "owned_workers", return_value={"status": "retired"}):
            result = F.watch(self.args, self.bound)
        launch.assert_not_called()
        self.assertEqual(result["status"], "CAMPAIGN_FAILED_OR_INCOMPLETE")
        self.assertEqual(result["report_status"], "NOT_RUN")

    def test_report_failure_persists_receipt_and_still_checks_workers(self):
        self.terminal()
        with patch.object(F.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["fixture"])), \
                patch.object(F, "owned_workers", return_value={"status": "retired"}) as workers:
            result = F.watch(self.args, self.bound)
        workers.assert_called_once()
        self.assertEqual(result["status"], "FAILED_PRESERVED_FOR_REVIEW")
        self.assertEqual(result["failure_type"], "CalledProcessError")

    def test_external_auth_failure_never_means_zero_workers(self):
        self.terminal()
        with patch.object(F.subprocess, "run", side_effect=self.fake_report), \
                patch.object(F, "owned_workers", side_effect=ValueError("authentication expired")):
            result = F.watch(self.args, self.bound)
        self.assertEqual(result["status"], "REPORT_VERIFIED_EXTERNAL_CHECK_PENDING")
        self.assertEqual(result["report_status"], "AUDITED_EXPLORATORY_SELECT_ONLY")
        self.assertEqual(result["owned_worker_check"]["status"], "UNVERIFIED_EXTERNAL_STATE")

    def test_wait_is_bounded_and_rejects_wrong_terminal_scope(self):
        with patch.object(F.time, "monotonic", side_effect=[0, 11]):
            with self.assertRaisesRegex(ValueError, "deadline"):
                F.wait_terminal(self.run / "missing.json", "a" * 64, "b" * 40, 10, 1)
        terminal = self.terminal()
        with self.assertRaisesRegex(ValueError, "binding"):
            F.wait_terminal(self.run / "campaign.completion.json", "f" * 64, terminal["source_commit"], 10, 1)
        for seconds, interval in ((True, 1), (47401, 1), (10, 61), (0, 1)):
            with self.assertRaises(ValueError):
                F.wait_terminal(self.run / "missing.json", "a" * 64, "b" * 40, seconds, interval)

    def test_source_drift_and_existing_output_fail_without_overwrite_or_cleanup(self):
        self.terminal()
        old = self.run / "final-reading"
        old.mkdir()
        (old / "report.cid").write_text("d" * 64)
        with patch.object(F.subprocess, "run") as launch, patch.object(F, "owned_workers", return_value={}):
            result = F.watch(self.args, self.bound)
        launch.assert_not_called()  # In particular no removal of an old report container.
        self.assertEqual(result["status"], "FAILED_PRESERVED_FOR_REVIEW")
        self.assertEqual((old / "report.cid").read_text(), "d" * 64)
        with self.assertRaisesRegex(ValueError, "already completed"):
            F.watch(self.args, self.bound)

    def test_command_uses_exact_events_readonly_mounts_and_no_development_truth(self):
        terminal = self.terminal()
        command = F.report_command(self.bound, terminal, self.run / "new-report")
        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("/code/m39_ordered_multichannel_report.py", command)
        self.assertNotIn("--development", command)
        self.assertEqual(command[command.index("--outdir") + 1], "/output/report")
        mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
        self.assertTrue(all(item.endswith(",readonly") for item in mounts[:-1]))
        self.assertIn("dst=/output", mounts[-1])
        F.write_json(self.run / "event-999-stage-a-downloaded.json", {"event": "stage-a-downloaded", "outputs": "wrong"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            F.report_command(self.bound, terminal, self.run / "new-report")

    def test_configuration_rejects_symlink_external_paths_and_changed_calibration(self):
        with self.assertRaisesRegex(ValueError, "calibration report hash"):
            F.configuration(self.path, self.source, self.calibration, "f" * 64)
        with self.assertRaisesRegex(ValueError, "private"):
            F.configuration(self.path, self.repo, self.calibration, F.sha256(self.calibration / "report.json"))
        (self.source / "bad.py").symlink_to(self.source / "m39_multichannel_finisher.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            F.source_inventory(self.source)

    def test_owned_resource_check_only_reads_exact_run_labels_and_native_uid(self):
        self.terminal()
        job = {"name": "projects/p/locations/l/jobs/expected-job", "uid": "expected-uid",
            "labels": {"team": "frank", "m39_run": self.stages["screen"].name}, "status": {"state": "SUCCEEDED"}}
        job_b = {**job, "labels": {"team": "frank", "m39_run": self.stages["followup"].name}}
        responses = [json.dumps(job), "[]", json.dumps(job_b), "[]"]
        with patch.object(F, "native_auth", return_value={"service_account": "fixture"}), \
                patch.object(F, "native_env_prefix", return_value=[]), \
                patch.object(F, "observed_job_ids", return_value={"expected-job": "expected-uid"}), \
                patch.object(F.subprocess, "check_output", side_effect=responses) as query:
            result = F.owned_workers(self.cfg, timeout_seconds=120, require_success=True)
        self.assertEqual(result["status"], "OWNED_JOBS_TERMINAL_WORKERS_RETIRED")
        for call in query.call_args_list:
            command = call.args[0]
            self.assertNotIn("delete", command)
            self.assertNotIn("cancel", command)
            if "batch" in command:
                self.assertIn("describe", command)
                self.assertIn("expected-job", command)
                self.assertNotIn("list", command)
            else:
                self.assertIn("list", command)
                self.assertTrue(any(word.startswith("--filter=labels.m39_run=m39-gpu-fixture-") for word in command))

    def test_worker_uid_mismatch_or_active_worker_fail_closed(self):
        self.terminal()
        job = {"name": "expected-job", "uid": "wrong-uid", "labels": {
            "team": "frank", "m39_run": self.stages["screen"].name}, "status": {"state": "SUCCEEDED"}}
        with patch.object(F, "native_auth", return_value={"service_account": "fixture"}), \
                patch.object(F, "native_env_prefix", return_value=[]), \
                patch.object(F, "observed_job_ids", return_value={"expected-job": "expected-uid"}), \
                patch.object(F.subprocess, "check_output", return_value=json.dumps(job)):
            with self.assertRaisesRegex(ValueError, "UID"):
                F.owned_workers(self.cfg, timeout_seconds=120)
        vm = {"name": "own-vm", "status": "RUNNING", "labels": {
              "team": "frank", "m39_run": self.stages["screen"].name}}
        with patch.object(F, "native_auth", return_value={"service_account": "fixture"}), \
                patch.object(F, "native_env_prefix", return_value=[]), \
                patch.object(F, "observed_job_ids", return_value={}), \
                patch.object(F.subprocess, "check_output", return_value=json.dumps([vm])):
            with self.assertRaisesRegex(ValueError, "not retired"):
                F.owned_workers(self.cfg, timeout_seconds=120)

    def test_failed_cleanup_404_is_explicit_and_still_checks_exact_compute_inventory(self):
        self.terminal("FAILED")
        missing = subprocess.CalledProcessError(1, ["gcloud"], output="NOT_FOUND: resource deleted")
        with patch.object(F, "native_auth", return_value={"service_account": "fixture"}), \
                patch.object(F, "native_env_prefix", return_value=[]), \
                patch.object(F, "observed_job_ids", return_value={"expected-job": "expected-uid"}), \
                patch.object(F.subprocess, "check_output", side_effect=[missing, "[]", missing, "[]"]):
            report = F.owned_workers(self.cfg, timeout_seconds=120)
        self.assertEqual(report["stages"]["screen"]["native_jobs_returned_NOT_FOUND"], ["expected-job"])
        self.assertEqual(report["stages"]["screen"]["nonretired_instances"], 0)

    def test_success_requires_every_declared_job_and_verified_success_state(self):
        self.terminal()
        with patch.object(F, "native_auth", return_value={"service_account": "fixture"}), \
                patch.object(F, "native_env_prefix", return_value=[]), \
                patch.object(F, "observed_job_ids", return_value={}):
            with self.assertRaisesRegex(ValueError, "complete native"):
                F.owned_workers(self.cfg, timeout_seconds=120, require_success=True)

    def test_pre_watch_source_failure_writes_launch_bound_failure_receipt(self):
        # Simulate mutation in the short interval between detach and bootstrap.
        launch = {**self.launch, "schema_version": F.SCHEMA, "argv": ["python", "--campaign", str(self.path)]}
        (self.run / "finisher.launch.json").write_text(json.dumps(launch))
        argv = ["finisher", "--watch", "--campaign", str(self.path), "--source-dir", str(self.source),
                "--calibration-dir", str(self.calibration), "--calibration-report-sha256", "f" * 64]
        with patch.object(sys, "argv", argv), patch.object(F, "watch") as watcher:
            with self.assertRaisesRegex(ValueError, "calibration report hash"):
                F.main()
        watcher.assert_not_called()
        report = F.read_json(self.run / "finisher.completion.json")
        self.assertEqual(report["status"], "BOOTSTRAP_FAILED_PRESERVED_FOR_REVIEW")
        self.assertEqual(report["report_status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
