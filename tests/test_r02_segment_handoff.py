"""Real fictional processes only: never signal an existing DNABR process."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import r02_segment_handoff as handoff
from test_r02_segment_campaign import CampaignFixture, STAGE_PROGRAM

PARENT = '''import json,pathlib,subprocess,sys,time
path=pathlib.Path(sys.argv[1]); config=json.loads(path.read_text()); children=[]
for job in config["jobs"][:2]:
 trace=str(path.parent/job["id"]/"attempt01.annotation.trace.tsv")
 command=[trace if x=="{trace}" else x for x in job["annotation"]["command"]]
 children.append(subprocess.Popen(command,cwd=job["run_dir"],start_new_session=True))
(path.parent/"fixture_pids.json").write_text(json.dumps([p.pid for p in children]))
while any(p.poll() is None for p in children): time.sleep(.02)
(path.parent/"pending_started.txt").write_text("must not happen after transfer")
time.sleep(30)
'''

PROGRAM = STAGE_PROGRAM.replace('root/"work"/"aa"/"bb"', 'root/"work"/"aa"/"bbbbbbbbbbbbbbbb"').replace(
    'time.sleep(p.get("sleep",0.01))',
    'while stage=="annotation" and not pathlib.Path(p["release"]).exists(): time.sleep(.01)\n'
    'time.sleep(p.get("sleep",0.01))') + '''
if stage=="annotation":
 (out.parent/".exitcode").write_text(str(p.get("task_exit",0)))
 pathlib.Path(sys.argv[3]).write_text("hash\\tstatus\\texit\\n"+p.get("trace_hash","aa/bbbbbb")+"\\tCOMPLETED\\t"+str(p.get("trace_exit",0))+"\\n")
 sys.exit(p.get("nextflow_exit",0))
'''


def eventually(condition, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for fictional fixture")


class HandoffFixture:
    def __init__(self, root, amendments=None):
        self.root = Path(root)
        self.fx = CampaignFixture(self.root / "campaign", n=3)
        self.fx.script.write_text(PROGRAM)
        self.fx.config["files"][str(self.fx.script)] = handoff.operations.sha(self.fx.script)
        self.jobs = self.fx.config["jobs"]
        for index, job in enumerate(self.jobs):
            self.fx.amend_fixture(index, release=str(Path(job["run_dir"]) / "release"), **(amendments or {}).get(index, {}))
            job["annotation"]["command"].append("{trace}")
        self.fx.save()
        for job in self.jobs[:2]:
            folder = Path(job["run_dir"])
            subprocess.run(job["geometry"]["command"], cwd=folder, check=True)
            receipt = handoff.campaign.accept_geometry(job)
            handoff.operations.write_new(folder / "geometry.complete.json", dict(campaign_sha256=self.fx.digest,
                receipt=str(receipt), receipt_sha256=handoff.operations.sha(receipt)))
            command = [str(folder / "attempt01.annotation.trace.tsv") if x == "{trace}" else x
                       for x in job["annotation"]["command"]]
            handoff.operations.status(folder, "RUNNING_ANNOTATION", job_id=job["id"], campaign_sha256=self.fx.digest, command=command)
        handoff.operations.status(self.fx.root, "RUNNING", campaign_sha256=self.fx.digest,
                                  running=["chr1", "chr2"], pending=["chr3"])
        self.parent_script = self.root / "fictional_parent.py"
        self.parent_script.write_text(PARENT)
        self.parent = subprocess.Popen([sys.executable, str(self.parent_script), str(self.fx.path)], start_new_session=True)
        eventually(lambda: (self.fx.root / "fixture_pids.json").exists())
        self.pids = json.loads((self.fx.root / "fixture_pids.json").read_text())
        eventually(lambda: all((Path(j["run_dir"]) / "annotation.pid").exists() for j in self.jobs[:2]))
        self.folder = self.root / "handoff"
        self.folder.mkdir(mode=0o700)
        self.path = self.folder / "handoff.json"
        self.config = dict(schema=handoff.SCHEMA, handoff_id="fictional-transfer", mode="SYNTHETIC_TEST",
            campaign_path=str(self.fx.path), campaign_sha256=self.fx.digest,
            supervisor=handoff.process_identity.identity(handoff.process_identity.process_info(self.parent.pid)),
            files={str(self.parent_script): handoff.operations.sha(self.parent_script)},
            adopted_jobs=[dict(job_id=job["id"], process=handoff.process_identity.identity(handoff.process_identity.process_info(pid)),
                               trace_path=str(Path(job["run_dir"]) / "attempt01.annotation.trace.tsv"))
                          for job, pid in zip(self.jobs, self.pids)], delegated_job_ids=["chr3"],
            resources=dict(timeout_seconds=10, min_free_disk_gib=.0001, min_available_memory_gib=.0001,
                           poll_seconds=.02, child_term_grace_seconds=.1))
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.config))
        self.digest = handoff.operations.sha(self.path)

    def release(self):
        for job in self.jobs[:2]:
            (Path(job["run_dir"]) / "release").touch()

    def cleanup(self):
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.parent.poll() is None:
            self.parent.kill()
        self.parent.wait(timeout=5)

    def run_releasing_after_transfer(self):
        failures = []

        def release_later():
            try:
                eventually(lambda: (self.folder / "queue_transfer.json").exists())
                if handoff.process_identity.process_info(self.parent.pid)["state"] != "T":
                    raise AssertionError("Parent was not stopped before delegation")
                self.release()
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=release_later)
        thread.start()
        try:
            with mock.patch.object(handoff.campaign, "publish_bundle", side_effect=CampaignFixture.fake_publish):
                result = handoff.run(self.path, self.digest)
        finally:
            thread.join()
        if failures:
            raise failures[0]
        return result


class HandoffTests(unittest.TestCase):
    def fixture(self, **kwargs):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        fixture = HandoffFixture(temp.name, **kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def test_inspection_preserves_running_parent_and_children(self):
        fx = self.fixture()
        result = handoff.run(fx.path, fx.digest, inspect_only=True)
        self.assertEqual(result["stage"], "INSPECTED_NOT_APPLIED")
        self.assertNotEqual(handoff.process_identity.process_info(fx.parent.pid)["state"], "T")
        self.assertFalse((fx.folder / "queue_transfer.json").exists())
        self.assertFalse((fx.folder / "prepared.json").exists())

    def test_pause_only_supervisor_children_finish_and_both_publish(self):
        fx = self.fixture()
        result = fx.run_releasing_after_transfer()
        self.assertEqual(result["stage"], "COMPLETE_LOCAL_ANNOTATIONS_PUBLISHED_QUEUE_TRANSFERRED")
        self.assertEqual(set(result["adopted"]), {"chr1", "chr2"})
        self.assertEqual(fx.parent.wait(timeout=3), -signal.SIGKILL)
        self.assertFalse((fx.fx.root / "pending_started.txt").exists())
        for job in fx.jobs[:2]:
            self.assertTrue((Path(job["run_dir"]) / "publication.json").exists())
            self.assertTrue((Path(job["run_dir"]) / "geometry.complete.json").exists())
        self.assertFalse((Path(fx.jobs[2]["run_dir"]) / "annotation.pid").exists())

    def test_nonzero_nextflow_exit_never_adopted_despite_good_task_manifest(self):
        fx = self.fixture(amendments={0: dict(nextflow_exit=7)})
        result = fx.run_releasing_after_transfer()
        self.assertEqual(result["stage"], "PARTIAL_LOCAL_ANNOTATIONS_QUEUE_TRANSFERRED")
        self.assertIn("Nextflow", result["errors"]["chr1"])
        self.assertEqual(set(result["adopted"]), {"chr2"})
        self.assertFalse((Path(fx.jobs[0]["run_dir"]) / "publication.json").exists())

    def test_bad_task_exit_trace_exit_and_trace_hash_are_rejected(self):
        for change in (dict(task_exit=9), dict(trace_exit=2), dict(trace_hash="ab/cccccc")):
            with self.subTest(change=change):
                fx = self.fixture(amendments={0: change})
                result = fx.run_releasing_after_transfer()
                self.assertEqual(set(result["adopted"]), {"chr2"})
                self.assertIn("chr1", result["errors"])

    def test_wrong_configuration_or_start_ticks_never_sends_signal(self):
        fx = self.fixture()
        with mock.patch.object(signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                handoff.run(fx.path, "0" * 64)
            fx.config["supervisor"]["start_ticks"] += 1
            fx.save()
            with self.assertRaises(ValueError):
                handoff.run(fx.path, fx.digest)
            send.assert_not_called()

    def test_wrong_child_command_or_changed_queue_never_pauses(self):
        fx = self.fixture()
        fx.config["adopted_jobs"][0]["process"]["cmdline_sha256"] = "0" * 64
        fx.save()
        with mock.patch.object(signal, "pidfd_send_signal") as send:
            with self.assertRaisesRegex(ValueError, "command identity"):
                handoff.run(fx.path, fx.digest)
            send.assert_not_called()

    def test_duplicate_guard_and_missing_geometry_fail_closed(self):
        fx = self.fixture()
        with handoff.transfer_lock(fx.fx.root):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                handoff.run(fx.path, fx.digest)
        (Path(fx.jobs[0]["run_dir"]) / "geometry.complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "geometry checkpoint"):
            handoff.run(fx.path, fx.digest)
        self.assertNotEqual(handoff.process_identity.process_info(fx.parent.pid)["state"], "T")

    def test_resource_failure_after_transfer_stops_only_owned_processes(self):
        fx = self.fixture()
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self.addCleanup(lambda: (unrelated.kill(), unrelated.wait()))
        available = handoff.campaign.available_memory_bytes
        with mock.patch.object(handoff.campaign, "available_memory_bytes",
                               side_effect=lambda: 0 if (fx.folder / "queue_transfer.json").exists() else available()):
            with self.assertRaisesRegex(ValueError, "reserve|interrupted"):
                handoff.run(fx.path, fx.digest)
        self.assertIsNone(unrelated.poll())
        self.assertEqual(fx.parent.wait(timeout=3), -signal.SIGKILL)
        self.assertFalse((fx.fx.root / "pending_started.txt").exists())
        status = json.loads((fx.folder / "status.json").read_text())
        self.assertEqual(status["stage"], "HANDOFF_STOPPED_WITH_ERROR")
        self.assertIsNone(status["cleanup_error"])

    def test_guard_restart_after_pause_before_marker_preserves_original_deadline(self):
        fx = self.fixture()
        started = time.time() - .5
        handoff.publication.write_receipt_new(fx.folder / "prepared.json", dict(schema=handoff.SCHEMA,
            handoff_sha256=fx.digest, campaign_sha256=fx.fx.digest, started_unix=started))
        handoff.process_identity.send(fx.config["supervisor"], signal.SIGSTOP)
        handoff.process_identity.stopped(fx.config["supervisor"])
        result = fx.run_releasing_after_transfer()
        self.assertFalse(result["errors"])
        marker = json.loads((fx.folder / "queue_transfer.json").read_text())
        self.assertEqual(marker["started_unix"], started)

    def test_sealed_source_change_rejected_before_pause(self):
        fx = self.fixture()
        fx.parent_script.write_text(PARENT + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Sealed file changed"):
            handoff.run(fx.path, fx.digest)
        self.assertNotEqual(handoff.process_identity.process_info(fx.parent.pid)["state"], "T")

    def test_publication_is_watched_and_stop_interrupts_it(self):
        fx = self.fixture()
        stop = threading.Event()

        def publishing(job, manifest, directory, env, event):
            stop.set()
            eventually(event.is_set)
            raise ValueError("publication interrupted")

        timer = threading.Thread(target=lambda: (eventually(lambda: (fx.folder / "queue_transfer.json").exists()), fx.release()))
        timer.start()
        try:
            with mock.patch.object(handoff.campaign, "publish_bundle", side_effect=publishing):
                with self.assertRaisesRegex(ValueError, "interrupted"):
                    handoff.run(fx.path, fx.digest, stop=stop)
        finally:
            timer.join()
        self.assertEqual(fx.parent.wait(timeout=3), -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
