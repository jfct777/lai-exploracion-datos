"""Bounded local orchestration tests with fictional jobs and no cloud calls."""
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
import r02_segment_campaign as campaign


STAGE_PROGRAM = '''import hashlib,json,os,pathlib,sys,time
stage,pfile=sys.argv[1:3]
p=json.loads(pathlib.Path(pfile).read_text()); root=pathlib.Path.cwd()
with (root/(stage+".invocations")).open("a") as f: f.write("called\\n")
(root/(stage+".pid")).write_text(str(os.getpid()))
(root/(stage+".last_argv.json")).write_text(json.dumps(sys.argv))
time.sleep(p.get("sleep",0.01))
if p.get("fail_stage")==stage: sys.exit(3)
name="geometry" if stage=="geometry" else "segment_evidence"
out=root/"work"/"aa"/"bb"/name; out.mkdir(parents=True,exist_ok=False)
inputs={"synthetic":{"path":p["input"],"sha256":hashlib.sha256(pathlib.Path(p["input"]).read_bytes()).hexdigest()}}
data={"chrom":p["chrom"],"n_samples":100,"inputs":inputs,"checks":{"counts":{"input_chain_rows":7}}}
if stage=="geometry":
 data.update(status="PREFLIGHT_GEOMETRY_ONLY_NOT_EVIDENCE",parameters={"block_bp":1000})
 data["checks"]["counts"].update(peak_active_chain_rows_upper_bound=p.get("peak",10),blocks_exceeding_active_row_bound=0)
 target=out/"preflight.json"
else:
 (out/"evidence.tsv").write_text("synthetic\\n")
 if p.get("partial_failure"): sys.exit(4)
 data.update(schema="r02_segment_evidence_v1",status="COMPLETE_DESCRIPTIVE_NOT_VALIDATED",outputs_sha256={"evidence.tsv":hashlib.sha256((out/"evidence.tsv").read_bytes()).hexdigest()})
 target=out/"manifest.json"
target.write_text(json.dumps(data))
'''


class CampaignFixture:
    def __init__(self, root, n=2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.script = self.root / "synthetic_stage.py"
        self.script.write_text(STAGE_PROGRAM)
        self.source = self.root / "input.synthetic.txt"
        self.source.write_text("fictional markers only\n")
        self.path = self.root / "campaign.json"
        self.config = dict(schema=campaign.SCHEMA, campaign_id="synthetic-campaign", mode="SYNTHETIC_TEST",
                           files={str(self.script): campaign.operations.sha(self.script)}, environment={},
                           resources=dict(max_concurrent=2, memory_budget_gib=2, reserve_available_memory_gib=.0001,
                                          min_free_disk_gib=.0001, poll_seconds=.02, admission_timeout_seconds=.15), jobs=[])
        for chrom in range(1, n + 1):
            folder = self.root / f"chr{chrom}"
            folder.mkdir()
            params = folder / "fixture.json"
            params.write_text(json.dumps(dict(input=str(self.source), chrom=str(chrom))))
            job = dict(id=f"chr{chrom}", run_dir=str(folder), files={str(params): campaign.operations.sha(params)},
                       input_files={str(self.source): dict(bytes=self.source.stat().st_size, sha256=campaign.operations.sha(self.source))},
                       memory_gib=.001, scratch_gib=.001, container_label=f"dnabr_r02_campaign=synthetic-campaign.chr{chrom}",
                       geometry=dict(command=[sys.executable, str(self.script), "geometry", str(params)], timeout_seconds=3,
                                     receipt_glob="work/*/*/geometry/preflight.json",
                                     expected=dict(chrom=str(chrom), n_samples=100, input_chain_rows=7, max_active_intervals=50, block_bp=1000)),
                       annotation=dict(command=[sys.executable, str(self.script), "annotation", str(params)], timeout_seconds=3,
                                       manifest_glob="work/*/*/segment_evidence/manifest.json", expected_status="COMPLETE_DESCRIPTIVE_NOT_VALIDATED"),
                       destination=campaign.publication.PREFIX + f"synthetic-campaign/chr{chrom}")
            self.config["jobs"].append(job)
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.config))
        self.digest = campaign.operations.sha(self.path)

    def amend_fixture(self, index, **values):
        job = self.config["jobs"][index]
        path = next(iter(job["files"]))
        data = json.loads(Path(path).read_text())
        data.update(values)
        Path(path).write_text(json.dumps(data))
        job["files"][path] = campaign.operations.sha(path)
        self.save()

    @staticmethod
    def fake_publish(job, manifest, directory, env, stop):
        path = directory / "publication.json"
        if path.exists():
            return campaign.verify_publication_receipt(path, manifest, job["destination"])
        result = dict(schema=campaign.publication.SCHEMA, status="PUBLISHED_VERIFIED",
                      manifest_sha256=campaign.operations.sha(manifest), destination=job["destination"])
        campaign.operations.write_new(path, result)
        return result

    def run(self, **kwargs):
        with mock.patch.object(campaign, "publish_bundle", side_effect=self.fake_publish):
            return campaign.run_campaign(self.path, self.digest, **kwargs)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.fx = CampaignFixture(temp.name)

    def test_real_processes_completed_stages_reused_on_explicit_resume(self):
        first = self.fx.run()
        self.assertEqual(first["stage"], "COMPLETE_PUBLISHED_ALL_JOBS")
        second = self.fx.run(resume=True)
        self.assertEqual(second["stage"], first["stage"])
        for job in self.fx.config["jobs"]:
            folder = Path(job["run_dir"])
            for stage in ("geometry", "annotation"):
                self.assertEqual((folder / f"{stage}.invocations").read_text(), "called\n")
                self.assertTrue((folder / f"{stage}.complete.json").is_file())
            status = json.loads((folder / "status.json").read_text())
            self.assertEqual(status["stage"], campaign.SUCCESS)
            self.assertFalse(status["biological_validation_complete"])
        with self.assertRaisesRegex(ValueError, "explicit --resume"):
            self.fx.run()

    def test_one_failed_chromosome_does_not_stop_the_other(self):
        self.fx.amend_fixture(0, fail_stage="annotation")
        result = self.fx.run()
        self.assertEqual(result["stage"], "PARTIAL_OR_STOPPED")
        self.assertEqual(result["results"]["chr1"]["stage"], "STOPPED_WITH_ERROR")
        self.assertEqual(result["results"]["chr2"]["stage"], campaign.SUCCESS)
        self.assertTrue((self.fx.root / "chr1/geometry.complete.json").exists())
        self.assertFalse((self.fx.root / "chr1/annotation.complete.json").exists())

    def test_geometry_failure_prevents_annotation_and_publication(self):
        self.fx.amend_fixture(0, peak=51)
        result = self.fx.run()
        self.assertIn("Geometry scope", result["results"]["chr1"]["error"])
        self.assertFalse((self.fx.root / "chr1/annotation.invocations").exists())
        self.assertFalse((self.fx.root / "chr1/publication.json").exists())

    def test_partial_annotation_is_preserved_and_not_blindly_resumed(self):
        self.fx.amend_fixture(0, partial_failure=True)
        self.fx.run()
        result = self.fx.run(resume=True)
        self.assertIn("explicit recovery amendment", result["results"]["chr1"]["error"])
        self.assertEqual((self.fx.root / "chr1/annotation.invocations").read_text(), "called\n")
        self.assertTrue((self.fx.root / "chr1/work/aa/bb/segment_evidence/evidence.tsv").exists())

    def test_failed_stage_without_outputs_receives_explicit_resume_flag(self):
        self.fx.amend_fixture(0, fail_stage="annotation")
        self.fx.run()
        self.fx.run(resume=True)
        self.assertEqual((self.fx.root / "chr1/geometry.invocations").read_text(), "called\n")
        self.assertEqual((self.fx.root / "chr1/annotation.invocations").read_text(), "called\ncalled\n")
        argv = json.loads((self.fx.root / "chr1/annotation.last_argv.json").read_text())
        self.assertEqual(argv[-1], "-resume")

    def test_seal_mismatch_cannot_execute(self):
        self.fx.script.write_text(self.fx.script.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Sealed file changed"):
            self.fx.run()
        self.assertFalse((self.fx.root / "status.json").exists())

    def test_input_mismatch_stops_each_affected_job(self):
        self.fx.source.write_text("changed\n")
        result = self.fx.run()
        self.assertTrue(all(item["stage"] == "STOPPED_WITH_ERROR" for item in result["results"].values()))
        self.assertFalse((self.fx.root / "chr1/geometry.invocations").exists())

    def test_campaign_hash_and_resumed_configuration_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Campaign SHA256 mismatch"):
            campaign.run_campaign(self.fx.path, "0" * 64)
        self.fx.run()
        self.fx.config["resources"]["poll_seconds"] = .03
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "configuration changed since prior"):
            self.fx.run(resume=True)

    def test_resource_reasons_and_external_slot_release(self):
        job = self.fx.config["jobs"][0]
        status = self.fx.root / "external.json"
        self.fx.config["external_reservations"] = [dict(status_file=str(status), release_stages=[campaign.SUCCESS], slots=1,
                                                      memory_gib=.001, scratch_gib=.001)]
        allowed, detail = campaign.admission(self.fx.config, job, [job], self.fx.root, free_bytes=campaign.GIB, memory_bytes=campaign.GIB)
        self.assertFalse(allowed)
        self.assertIn("concurrency_reservation", detail["reasons"])
        status.write_text(json.dumps(dict(stage="STOPPED_WITH_ERROR")))
        self.assertEqual(len(campaign.external_reservations(self.fx.config)), 1)
        status.write_text(json.dumps(dict(stage=campaign.SUCCESS)))
        self.assertEqual(campaign.external_reservations(self.fx.config), [])
        allowed, detail = campaign.admission(self.fx.config, job, [], self.fx.root, free_bytes=0, memory_bytes=0)
        self.assertFalse(allowed)
        self.assertEqual(set(detail["reasons"]), {"available_memory", "available_disk_with_reservations"})

    def test_resource_blocked_exit_has_no_fake_running_jobs(self):
        with mock.patch.object(campaign, "available_memory_bytes", return_value=0):
            result = self.fx.run()
        self.assertEqual(result["stage"], "PARTIAL_OR_STOPPED")
        self.assertTrue(all(row["stage"] == "NOT_STARTED_RESOURCE_BLOCKED" for row in result["results"].values()))

    def test_large_blocked_first_job_is_not_bypassed_by_later_small_job(self):
        self.fx.config["jobs"][0]["scratch_gib"] = 10**6
        self.fx.save()
        with mock.patch.object(campaign, "execute_job") as execute:
            result = self.fx.run()
        execute.assert_not_called()
        self.assertEqual(result["results"]["chr2"]["blocked_queue_head"], "chr1")

    def test_external_job_wait_does_not_consume_idle_admission_timeout(self):
        status = self.fx.root / "external.json"
        status.write_text(json.dumps(dict(stage="RUNNING")))
        self.fx.config["external_reservations"] = [dict(status_file=str(status), release_stages=[campaign.SUCCESS],
                                                      slots=1, memory_gib=2, scratch_gib=.001)]
        self.fx.save()
        timer = threading.Timer(.3, lambda: status.write_text(json.dumps(dict(stage=campaign.SUCCESS))))
        timer.start()
        try:
            result = self.fx.run()
        finally:
            timer.join()
        self.assertEqual(result["stage"], "COMPLETE_PUBLISHED_ALL_JOBS")

    def test_concurrency_never_exceeds_two(self):
        self.fx = CampaignFixture(self.fx.root / "larger", n=4)
        active, peak, lock = 0, 0, threading.Lock()
        def bounded(config, job, digest, stop, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.08)
            with lock:
                active -= 1
            return {"stage": campaign.SUCCESS}
        with mock.patch.object(campaign, "execute_job", side_effect=bounded):
            result = self.fx.run()
        self.assertEqual(peak, 2)
        self.assertEqual(result["stage"], "COMPLETE_PUBLISHED_ALL_JOBS")

    def test_duplicate_controller_does_not_overwrite_active_status(self):
        marker = self.fx.root / "status.json"
        marker.write_text("owner active\n")
        with campaign.operations.execution_lock(self.fx.root):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                self.fx.run()
        self.assertEqual(marker.read_text(), "owner active\n")

    def test_duplicate_job_lock_preserves_its_status(self):
        folder = self.fx.root / "chr1"
        (folder / "status.json").write_text("active job\n")
        with campaign.operations.execution_lock(folder):
            result = self.fx.run()
        self.assertEqual((folder / "status.json").read_text(), "active job\n")
        self.assertIn("already owns", result["results"]["chr1"]["error"])

    def test_timeout_terminates_owned_child_without_touching_other_process(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(10)"], start_new_session=True)
        try:
            with self.assertRaises(TimeoutError):
                campaign.run_process([sys.executable, "-c", "import time;time.sleep(10)"], self.fx.root,
                                     self.fx.root / "timeout.log", .05, threading.Event(), os.environ.copy())
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_signal_event_interrupts_running_job(self):
        stop = threading.Event()
        timer = threading.Timer(.05, stop.set)
        timer.start()
        try:
            with self.assertRaises(InterruptedError):
                campaign.run_process([sys.executable, "-c", "import time;time.sleep(10)"], self.fx.root,
                                     self.fx.root / "interrupt.log", 5, stop, os.environ.copy())
        finally:
            timer.join()

    def test_real_sigterm_stops_campaign_and_owned_stage_processes(self):
        self.fx.amend_fixture(0, sleep=10)
        self.fx.amend_fixture(1, sleep=10)
        process = subprocess.Popen([sys.executable, str(ROOT / "bin/r02_segment_campaign.py"),
                                    "--campaign", str(self.fx.path), "--expected-sha256", self.fx.digest],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            paths = [self.fx.root / f"chr{i}/geometry.pid" for i in (1, 2)]
            while not all(path.exists() for path in paths) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(all(path.exists() for path in paths))
            pids = [int(path.read_text()) for path in paths]
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 2, stdout + stderr)
            self.assertEqual(json.loads((self.fx.root / "status.json").read_text())["stage"], "PARTIAL_OR_STOPPED")
            for pid in pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_completed_manifest_tampering_not_adopted_on_resume(self):
        self.fx.run()
        path = self.fx.root / "chr1/work/aa/bb/segment_evidence/evidence.tsv"
        path.write_text("tampered\n")
        result = self.fx.run(resume=True)
        self.assertEqual(result["results"]["chr1"]["stage"], "STOPPED_WITH_ERROR")
        self.assertIn("SHA256 mismatch", result["results"]["chr1"]["error"])

    def test_container_cleanup_uses_only_exact_campaign_label(self):
        calls = []
        def command(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="aabbccddeeff\n" if "ps" in argv else "", stderr="")
        label = self.fx.config["jobs"][0]["container_label"]
        with mock.patch.object(campaign.subprocess, "run", side_effect=command):
            campaign.cleanup_containers(label)
        self.assertEqual(calls[0], ["docker", "ps", "-q", "--filter", "label=" + label])
        self.assertEqual(calls[1], ["docker", "stop", "--time", "10", "aabbccddeeff"])

    def test_label_and_output_directory_collisions_rejected(self):
        self.fx.config["jobs"][0]["container_label"] = "unrelated=active-chr22"
        self.fx.save()
        with self.assertRaisesRegex(ValueError, "uniquely owned"):
            self.fx.run()


if __name__ == "__main__":
    unittest.main()
