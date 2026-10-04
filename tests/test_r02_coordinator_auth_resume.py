"""Small mocked operational recovery tests: no credentials, cloud or real jobs."""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"bin"))
import r02_coordinator_auth_resume as auth


class CoordinatorAuthRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base/"project/.claude/runs/run"
        self.previous = self.run/auth.PREVIOUS
        self.previous.mkdir(parents=True)
        self.config = self.base/"private_config"
        self.config.mkdir()
        self.account = "synthetic@example.invalid"
        self.uri = "gs://synthetic-bucket/worker/completion.json"
        for name in auth.COPIES:
            (self.previous/name).write_text("{}\n")
        (self.previous/"resume.py").write_bytes(Path(auth.history.__file__).read_bytes())
        self.current = dict(coordinator_sha256=auth.history.sha(self.previous/"coordinator.py"),
                            remote_chromosomes=[dict(completion_uri=self.uri)],
                            deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
        self.write(self.previous/"manifest.json", self.current)
        old = dict(resume_sha256=auth.history.sha(self.previous/"resume.py"),
                   coordinator_manifest_sha256=auth.history.sha(self.previous/"manifest.json"))
        self.write(self.previous/"resume.manifest.json", old)
        self.failure = dict(state="FAILED", resume_manifest_sha256=auth.history.sha(self.previous/"resume.manifest.json"),
                            error="Cannot authenticate cloud object: " + self.uri + ": ERROR: (gcloud.storage.objects.describe) You do not currently have an active account selected.\n")
        self.write(self.previous/"status.json", self.failure)
        self.write(self.previous/"resume_activation.json", {"old": True})
        self.write(self.previous/"coordinator_activation.json", {"old": True})
        self.write(self.run/"checkpoints/chr21_complete.json", {"original": True})
        self.write(self.run/"checkpoints/chr22_complete.json", {"original": True})
        self.write(self.run/"repairs/parallel-v1/imports/chr01_import.json", dict(completion=dict(uri=self.uri, generation="123"), outputs=[]))
        self.service = patch.object(auth, "inactive_service", return_value={"ActiveState": "failed", "MainPID": "0"})
        self.service.start()
        self.addCleanup(self.service.stop)
        self.env = patch.dict(os.environ, CLOUDSDK_CONFIG=str(self.config), CLOUDSDK_CORE_ACCOUNT=self.account)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.path = auth.prepare(self.run, "synthetic.service", self.config, self.account)
        self.spec = auth.history.read(self.path)
        self.coordinator = types.SimpleNamespace(GCS=Mock(), validate_metadata=Mock(), run_after_boundary=Mock(),
            timestamp=lambda value: datetime.fromisoformat(value).timestamp())
        self.recovery = types.SimpleNamespace(boundary_lock=Mock(return_value=nullcontext()), validate_mount=Mock())
        self.handoff = types.SimpleNamespace(write_json=self.write)
        self.pipeline = types.SimpleNamespace(verify_files=Mock())
        self.context = patch.object(auth.history, "validate_spec", return_value=(
            {}, self.current, self.coordinator, self.recovery, {}, self.handoff, self.pipeline, "historical404"))
        self.context.start()
        self.addCleanup(self.context.stop)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def verify(self):
        return auth.validate_spec(self.path, auth.history.sha(self.path))

    def test_exact_missing_account_failure_only(self):
        self.assertEqual(auth.authenticate_failure(self.failure, self.failure["resume_manifest_sha256"], self.current), self.uri)
        for message in ("not found: 404", "403 permission denied", "Reauthentication failed"):
            failure = dict(self.failure, error="Cannot authenticate cloud object: " + self.uri + ": " + message)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, "explicit missing-account"):
                auth.authenticate_failure(failure, self.failure["resume_manifest_sha256"], self.current)

    def test_wrong_failure_run_or_uri_rejected(self):
        with self.assertRaisesRegex(ValueError, "authenticated previous"):
            auth.authenticate_failure(self.failure, "bad", self.current)
        with self.assertRaisesRegex(ValueError, "unassigned"):
            auth.authenticate_failure(dict(self.failure, error="other"), self.failure["resume_manifest_sha256"], self.current)

    def test_verify_is_readonly_and_performs_real_probe_interface(self):
        before = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        result = auth.run(self.path, auth.history.sha(self.path))
        self.assertEqual(result["state"], "VERIFIED_AUTH_RECOVERY_NOT_EXECUTED")
        self.coordinator.GCS().metadata.assert_called_once_with(self.uri)
        self.coordinator.validate_metadata.assert_called_once()
        self.coordinator.run_after_boundary.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()})

    def test_changed_checkpoint_or_failure_rejected(self):
        for path in (self.run/"checkpoints/chr21_complete.json", self.previous/"status.json"):
            original = path.read_bytes()
            path.write_bytes(original+b" ")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "evidence changed"):
                self.verify()
            path.write_bytes(original)

    def test_missing_import_protection_rejected(self):
        self.spec["protected_sha256"].pop("repairs/parallel-v1/imports/chr01_import.json")
        self.write(self.path, self.spec)
        with self.assertRaisesRegex(ValueError, "Incomplete/changed protected"):
            self.verify()

    def test_changed_snapshot_rejected(self):
        path = self.path.parent/"previous_evidence/checkpoints/chr22_complete.json"
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.verify()

    def test_wrong_account_environment_rejected(self):
        with patch.dict(os.environ, CLOUDSDK_CORE_ACCOUNT="other@example.invalid"):
            with self.assertRaisesRegex(ValueError, "Authentication environment"):
                self.verify()

    def test_credentials_must_not_be_inside_project(self):
        self.spec["cloudsdk_config"] = str(self.run)
        with self.assertRaisesRegex(ValueError, "outside the project"):
            auth.validate_environment(self.spec)

    def test_deadline_cannot_be_changed(self):
        self.spec["deadline_utc"] = "2099-01-01T00:00:00+00:00"
        self.write(self.path, self.spec)
        with self.assertRaisesRegex(ValueError, "deadline/manifest"):
            self.verify()

    def test_cloud_probe_failure_stops_before_resume(self):
        self.coordinator.GCS().metadata.side_effect = RuntimeError("expired credentials")
        with self.assertRaisesRegex(RuntimeError, "expired credentials"):
            auth.run(self.path, auth.history.sha(self.path), execute=True)
        self.coordinator.run_after_boundary.assert_not_called()

    def test_execute_reuses_original_controller_under_shared_lock(self):
        before = (self.previous/"status.json").read_bytes()
        self.assertEqual(auth.run(self.path, auth.history.sha(self.path), execute=True)["state"], "COMPLETE")
        self.assertEqual(before, (self.previous/"status.json").read_bytes())
        self.recovery.boundary_lock.assert_called_once_with(self.run/"repairs/boundary-v2/.boundary.lock")
        self.coordinator.run_after_boundary.assert_called_once()
        self.assertFalse(self.coordinator.run_after_boundary.call_args.kwargs["activate"])
        self.assertTrue((self.path.parent/"activation.json").is_file())

    def test_existing_output_never_replaced(self):
        with self.assertRaises(FileExistsError):
            auth.prepare(self.run, "synthetic.service", self.config, self.account)


class ServiceIdentityTests(unittest.TestCase):
    def test_active_unknown_or_nonzero_pid_fails(self):
        for output in ("LoadState=loaded\nActiveState=active\nMainPID=42\n",
                       "LoadState=not-found\nActiveState=inactive\nMainPID=0\n",
                       "LoadState=loaded\nActiveState=failed\nMainPID=42\n"):
            with patch.object(auth.subprocess, "run", return_value=types.SimpleNamespace(stdout=output)):
                with self.assertRaisesRegex(ValueError, "active or unknown"):
                    auth.inactive_service("synthetic.service")

    def test_failed_inactive_service_is_admissible(self):
        with patch.object(auth.subprocess, "run", return_value=types.SimpleNamespace(stdout="LoadState=loaded\nActiveState=failed\nMainPID=0\n")):
            self.assertEqual(auth.inactive_service("synthetic.service")["MainPID"], "0")


if __name__ == "__main__":
    unittest.main()
