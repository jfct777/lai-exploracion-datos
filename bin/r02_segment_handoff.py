#!/usr/bin/env python3
"""Transfer a sealed pending queue while preserving its running annotation jobs.

Only the authenticated supervisor is paused. Its already-running Nextflow
children and labelled containers continue. A separate guard adopts completed
annotations only after kernel exit, zero task exit and matching trace evidence,
then uses the existing create-only publisher. It never edits scientific sources,
restarts a genotype scan, deletes outputs, or launches delegated chromosomes.

Run in an independent service, KillMode=process and TimeoutStopSec>=180. A new
service must not inherit the original campaign's cgroup. A killed guard may be
restarted using the same sealed handoff; its original wall-clock deadline is
preserved. Failure closes only authenticated child handles and exact job labels.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time

import r02_optimization_boundary as process_identity
import r02_segment_campaign as campaign

operations = campaign.operations
publication = campaign.publication
SCHEMA = "r02_segment_handoff_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def check_identity(spec):
    require(isinstance(spec, dict) and set(spec) == {"pid", "start_ticks", "cmdline_sha256"}
            and type(spec["pid"]) is int and spec["pid"] > 1
            and type(spec["start_ticks"]) is int and spec["start_ticks"] > 0
            and re.fullmatch(r"[a-f0-9]{64}", spec["cmdline_sha256"]), "Invalid process identity")


def direct_children(pid):
    """Popen children can belong to different worker threads in Linux procfs."""
    result = set()
    for task in (Path("/proc") / str(pid) / "task").iterdir():
        try:
            result.update(int(value) for value in (task / "children").read_text().split())
        except FileNotFoundError:
            continue
    return result


def bind_process(spec, *, allow_exited=False):
    check_identity(spec)
    fd = os.pidfd_open(spec["pid"])
    try:
        actual = process_identity.same_incarnation(spec)
        require(actual is not None, "Process disappeared before ownership binding")
        if actual["state"] in ("Z", "X"):
            require(allow_exited and process_identity.pidfd_exited(fd), "Process already exited")
        else:
            require(actual["cmdline_sha256"] == spec["cmdline_sha256"], "Process command identity changed")
        return fd, actual
    except BaseException:
        os.close(fd)
        raise


def kernel_exit_code(spec, fd):
    """Read the unreaped child's wait status without reaping another process.

    Pausing its original parent leaves an exited child as a zombie until this
    guard has authenticated its exit. Missing /proc state is not assumed zero.
    """
    if not process_identity.pidfd_exited(fd):
        return None
    fields = (Path("/proc") / str(spec["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
    require(len(fields) >= 50 and int(fields[19]) == spec["start_ticks"] and fields[0] in ("Z", "X"),
            "No authenticated unreaped child exit status")
    return os.waitstatus_to_exitcode(int(fields[49]))


def load(path, digest):
    path = Path(path).resolve()
    require(operations.sha(path) == publication.validate_sha(digest), "Handoff configuration SHA256 mismatch")
    config = campaign.read_json(path)
    require(config.get("schema") == SCHEMA, "Unsupported handoff schema")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,100}", config.get("handoff_id", "")), "Invalid handoff ID")
    check_identity(config["supervisor"])
    source, source_root = campaign.validate_config(config["campaign_path"], config["campaign_sha256"])
    require(config.get("mode", "PRODUCTION") == source.get("mode", "PRODUCTION"),
            "Handoff mode differs from the authenticated campaign")
    require(path.parent != source_root, "Use a separate directory for handoff records")
    require(not any(p.is_symlink() for p in (path.parent, *path.parent.parents)), "Handoff directory contains a symlink")
    campaign.sealed_files(config["files"])
    jobs = {job["id"]: job for job in source["jobs"]}
    adopted = config.get("adopted_jobs", [])
    require(isinstance(adopted, list) and 1 <= len(adopted) <= 2, "Adopt one or two running jobs only")
    ids, pids = set(), {config["supervisor"]["pid"]}
    for item in adopted:
        require(item["job_id"] in jobs and item["job_id"] not in ids, "Duplicate or unknown adopted job")
        ids.add(item["job_id"])
        check_identity(item["process"])
        require(item["process"]["pid"] not in pids, "Duplicate process identity")
        pids.add(item["process"]["pid"])
        trace = Path(item["trace_path"])
        require(trace.is_absolute() and trace.parent == Path(jobs[item["job_id"]]["run_dir"])
                and re.fullmatch(r"attempt[0-9]+\.annotation\.trace.tsv", trace.name), "Trace is not scoped to the adopted annotation")
    delegated = config.get("delegated_job_ids")
    require(isinstance(delegated, list) and delegated and len(set(delegated)) == len(delegated)
            and not ids.intersection(delegated) and set(delegated) <= set(jobs), "Invalid delegated queue")
    r = config.get("resources", {})
    for key in ("timeout_seconds", "min_free_disk_gib", "min_available_memory_gib", "poll_seconds", "child_term_grace_seconds"):
        require(type(r.get(key)) in (int, float) and r[key] > 0, "Invalid handoff resource: " + key)
    require(r["timeout_seconds"] <= 172800 and r["poll_seconds"] <= 30 and r["child_term_grace_seconds"] <= 60,
            "Handoff resource bounds exceed permitted scope")
    for jid in ids:
        campaign.authenticate_job(source, jobs[jid])
        require(campaign.completed_stage(Path(jobs[jid]["run_dir"]), "geometry", config["campaign_sha256"],
                                         campaign.accept_geometry, jobs[jid]) is not None,
                "Adopted annotation lacks an authenticated geometry checkpoint")
    return config, source, source_root, jobs, path.parent


def check_queue(config, source_root, jobs, *, paused=False):
    supervisor = process_identity.authenticated(config["supervisor"])
    require(supervisor is not None and (not paused or supervisor["state"] == "T"), "Supervisor is not in the expected live/stopped state")
    state = campaign.read_json(source_root / "status.json")
    ids = {item["job_id"] for item in config["adopted_jobs"]}
    require(state.get("campaign_sha256") == config["campaign_sha256"] and set(state.get("running", [])) == ids
            and state.get("pending") == config["delegated_job_ids"], "Campaign queue changed before transfer")
    require(direct_children(supervisor["pid"]) == {item["process"]["pid"] for item in config["adopted_jobs"]},
            "Supervisor has unexpected or missing direct children")
    for item in config["adopted_jobs"]:
        child = process_identity.same_incarnation(item["process"])
        require(child is not None and child["ppid"] == supervisor["pid"], "Adopted child no longer belongs to supervisor")
        require(child["pgid"] == child["sid"] == child["pid"],
                "Adopted Nextflow is not the leader of its own process session")
        if child["state"] not in ("Z", "X"):
            require(child["cmdline_sha256"] == item["process"]["cmdline_sha256"], "Child command identity changed")
        job = jobs[item["job_id"]]
        status = campaign.read_json(Path(job["run_dir"]) / "status.json")
        require(status.get("stage") == "RUNNING_ANNOTATION" and status.get("job_id") == item["job_id"]
                and status.get("campaign_sha256") == config["campaign_sha256"], "Adopt only the current annotation stage")
        command = status.get("command", [])
        expected = [item["trace_path"] if token == "{trace}" else token for token in job["annotation"]["command"]]
        require(command == expected, "Observed annotation command differs from frozen campaign")


@contextmanager
def transfer_lock(source_root):
    with (source_root / ".segment-handoff.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another handoff guard already owns this campaign") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def accepted_completion(item, job, child_fd):
    code = kernel_exit_code(item["process"], child_fd)
    require(code == 0, "Nextflow has not exited successfully")
    manifest = campaign.accept_annotation(job)
    task = manifest.parent.parent
    require((task / ".exitcode").read_text().strip() == "0", "Annotation task exit code is not zero")
    with Path(item["trace_path"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == 1 and rows[0].get("status") in ("COMPLETED", "CACHED") and rows[0].get("exit") == "0",
            "Annotation trace does not record one successful task")
    task_hash = task.relative_to(Path(job["run_dir"]) / "work").as_posix()
    trace_hash = rows[0].get("hash", "")
    require(re.fullmatch(r"[a-f0-9]{2}/[a-f0-9]{6,}", trace_hash) and task_hash.startswith(trace_hash),
            "Trace does not identify the completed annotation work directory")
    return manifest


def close_owned_children(config, handles, jobs):
    """On failure close exact pidfds and exact labels, never the VM or others."""
    sessions = {item["process"]["pid"] for item in config["adopted_jobs"]}
    descendants = []
    # Every adopted Nextflow started with start_new_session=True. Processes in
    # that session are its children, not unrelated jobs on this VM. Bind them
    # individually so neither an exited leader nor PID reuse broadens a signal.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in sessions:
            continue
        try:
            info = process_identity.process_info(int(entry.name))
            if info and info["sid"] in sessions and info["state"] not in ("Z", "X"):
                fd, _ = bind_process(process_identity.identity(info))
                descendants.append(fd)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    all_handles = list(handles.values()) + descendants
    try:
        _stop_owned_handles(config, handles, jobs, all_handles)
    finally:
        for fd in descendants:
            os.close(fd)


def _stop_owned_handles(config, handles, jobs, all_handles):
    for fd in all_handles:
        if not process_identity.pidfd_exited(fd):
            signal.pidfd_send_signal(fd, signal.SIGTERM)
    deadline = time.monotonic() + config["resources"]["child_term_grace_seconds"]
    while time.monotonic() < deadline and any(not process_identity.pidfd_exited(fd) for fd in all_handles):
        time.sleep(.05)
    for fd in all_handles:
        if not process_identity.pidfd_exited(fd):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
    if config.get("mode", "PRODUCTION") != "SYNTHETIC_TEST":
        for item in config["adopted_jobs"]:
            campaign.cleanup_containers(jobs[item["job_id"]]["container_label"])
    require(all(process_identity.pidfd_exited(fd, 5000) for fd in all_handles), "An owned Nextflow process did not stop")


def stop_paused_supervisor(spec, fd, child_handles):
    require(all(process_identity.pidfd_exited(child) for child in child_handles.values()), "Cannot end supervisor while children run")
    actual = process_identity.authenticated(spec)
    require(actual is not None and actual["state"] == "T", "Supervisor lost the authenticated pause")
    signal.pidfd_send_signal(fd, signal.SIGKILL)
    require(process_identity.pidfd_exited(fd, 5000), "Paused supervisor did not exit")


def guard_resources(config, started_unix, source_root, stop):
    require(not stop.is_set(), "Handoff interrupted")
    require(0 <= time.time() - started_unix < config["resources"]["timeout_seconds"], "Handoff wall-clock deadline reached")
    require(shutil.disk_usage(source_root).free >= config["resources"]["min_free_disk_gib"] * campaign.GIB,
            "Handoff free-disk reserve breached")
    require(campaign.available_memory_bytes() >= config["resources"]["min_available_memory_gib"] * campaign.GIB,
            "Handoff available-memory reserve breached")


def run(path, digest, *, inspect_only=False, stop=None):
    config, source, source_root, jobs, output = load(path, digest)
    stop = stop or threading.Event()
    with transfer_lock(source_root), ExitStack() as stack:
        marker = output / "queue_transfer.json"
        prior = campaign.read_json(marker) if marker.exists() else None
        armed_path = output / "prepared.json"
        armed = campaign.read_json(armed_path) if armed_path.exists() else None
        for record in (prior, armed):
            if record:
                require(record.get("handoff_sha256") == digest and record.get("campaign_sha256") == config["campaign_sha256"],
                        "Existing transfer belongs to another contract")
        require(not prior or armed, "Transfer marker lacks its preparation record")
        check_queue(config, source_root, jobs, paused=bool(prior))
        supervisor_fd, supervisor = bind_process(config["supervisor"])
        stack.callback(os.close, supervisor_fd)
        require(supervisor["state"] != "T" or armed, "Supervisor was already stopped outside this handoff")
        handles = {}
        for item in config["adopted_jobs"]:
            fd, _ = bind_process(item["process"], allow_exited=bool(armed and supervisor["state"] == "T"))
            handles[item["job_id"]] = fd
            stack.callback(os.close, fd)
        if inspect_only:
            return {"stage": "INSPECTED_NOT_APPLIED", "adopted_jobs": list(handles), "delegated_jobs": config["delegated_job_ids"]}
        started = armed["started_unix"] if armed else time.time()
        paused, transferred = supervisor["state"] == "T", bool(prior)
        guard_done, watchdog = threading.Event(), None
        guard_failures = []

        def watch():
            while not guard_done.wait(config["resources"]["poll_seconds"]):
                try:
                    guard_resources(config, started, source_root, stop)
                    check_queue(config, source_root, jobs, paused=True)
                    require(operations.sha(path) == digest, "Handoff configuration changed during monitoring")
                except BaseException as error:
                    guard_failures.append(str(error))
                    stop.set()
                    return

        try:
            guard_resources(config, started, source_root, stop)
            if not armed:
                publication.write_receipt_new(armed_path, dict(schema=SCHEMA, stage="PREPARED_NOT_TRANSFERRED",
                    handoff_sha256=digest, campaign_sha256=config["campaign_sha256"], started_unix=started))
            if not prior:
                if not paused:
                    signal.pidfd_send_signal(supervisor_fd, signal.SIGSTOP)
                    paused = True
                process_identity.stopped(config["supervisor"])
                check_queue(config, source_root, jobs, paused=True)
                campaign.sealed_files(config["files"])
                publication.write_receipt_new(marker, dict(schema=SCHEMA, stage="QUEUE_TRANSFERRED_SUPERVISOR_PAUSED",
                    handoff_sha256=digest, campaign_sha256=config["campaign_sha256"], started_unix=started,
                    supervisor=config["supervisor"], adopted_jobs=config["adopted_jobs"],
                    delegated_job_ids=config["delegated_job_ids"], no_scientific_sources_modified=True))
                transferred = True
            watchdog = threading.Thread(target=watch, name="handoff-resource-guard", daemon=True)
            watchdog.start()
            while not all(process_identity.pidfd_exited(fd) for fd in handles.values()):
                guard_resources(config, started, source_root, stop)
                check_queue(config, source_root, jobs, paused=True)
                require(operations.sha(path) == digest, "Handoff configuration changed during monitoring")
                operations.status(output, "WAITING_FOR_OWNED_ANNOTATIONS", handoff_sha256=digest,
                                  exited={jid: process_identity.pidfd_exited(fd) for jid, fd in handles.items()},
                                  delegated_job_ids=config["delegated_job_ids"], supervisor_paused=True)
                stop.wait(config["resources"]["poll_seconds"])
            manifests, receipts, errors = {}, {}, {}
            for item in config["adopted_jobs"]:
                guard_resources(config, started, source_root, stop)
                job = jobs[item["job_id"]]
                try:
                    campaign.authenticate_job(source, job)
                    manifests[item["job_id"]] = accepted_completion(item, job, handles[item["job_id"]])
                except Exception as error:
                    errors[item["job_id"]] = str(error)
            for item in config["adopted_jobs"]:
                job, jid = dict(jobs[item["job_id"]]), item["job_id"]
                if config.get("mode", "PRODUCTION") != "SYNTHETIC_TEST":
                    campaign.cleanup_containers(job["container_label"])
                if jid not in manifests:
                    continue
                job["publication_timeout_seconds"] = min(job.get("publication_timeout_seconds", 14400),
                                                        max(1, config["resources"]["timeout_seconds"] - (time.time() - started)))
                operations.status(output, "PUBLISHING_COMPLETED_ANNOTATION", handoff_sha256=digest, job_id=jid,
                                  manifest=str(manifests[jid]), supervisor_paused=True)
                env = {**os.environ, **source.get("environment", {})}
                try:
                    receipts[jid] = campaign.publish_bundle(job, manifests[jid], Path(job["run_dir"]), env, stop)
                except Exception as error:
                    errors[jid] = str(error)
            guard_resources(config, started, source_root, stop)
            guard_done.set()
            watchdog.join()
            stop_paused_supervisor(config["supervisor"], supervisor_fd, handles)
            result = dict(stage="PARTIAL_LOCAL_ANNOTATIONS_QUEUE_TRANSFERRED" if errors else
                          "COMPLETE_LOCAL_ANNOTATIONS_PUBLISHED_QUEUE_TRANSFERRED", handoff_sha256=digest,
                          adopted={jid: dict(manifest=str(manifests[jid]), manifest_sha256=operations.sha(manifests[jid]),
                                             destination=value["destination"]) for jid, value in receipts.items()},
                          errors=errors, delegated_job_ids=config["delegated_job_ids"], biological_validation_complete=False)
            operations.status(output, **result)
            return result
        except BaseException as error:
            guard_done.set()
            if watchdog:
                watchdog.join()
            cleanup_error = None
            if transferred:
                try:
                    close_owned_children(config, handles, jobs)
                    stop_paused_supervisor(config["supervisor"], supervisor_fd, handles)
                except BaseException as failure:
                    cleanup_error = str(failure)
            elif paused:
                # No delegation marker was exposed; restoring the unchanged
                # original supervisor is the safe rollback before transfer.
                signal.pidfd_send_signal(supervisor_fd, signal.SIGCONT)
            operations.status(output, "HANDOFF_STOPPED_WITH_ERROR", handoff_sha256=digest,
                              error_type=type(error).__name__, error=str(error), cleanup_error=cleanup_error,
                              guard_failures=guard_failures,
                              queue_transferred=transferred, original_artifacts_preserved=True)
            raise
        finally:
            guard_done.set()
            if watchdog:
                watchdog.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda signum, frame: stop.set())
    result = run(args.handoff, args.expected_sha256, inspect_only=args.inspect_only, stop=stop)
    print(json.dumps(result, sort_keys=True))
    return 1 if result["stage"].startswith("PARTIAL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
