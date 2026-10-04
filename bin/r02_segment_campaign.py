#!/usr/bin/env python3
"""Bounded execution of sealed per-chromosome Nextflow evidence jobs.

This controller changes concurrency, not scientific parameters. Each job runs
geometry, annotation and create-only publication. Explicit resume reuses only
authenticated finished stages; Nextflow may reuse completed tasks, never a
partially processed Python loop. Inputs and work directories are never deleted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import r02_autosome_pipeline as operations
import r02_publish_evidence as publication

SCHEMA = "r02_segment_campaign_v1"
GIB = 1024**3
SUCCESS = "COMPLETE_PUBLISHED"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return publication.read_unique_json(Path(path))


def sealed_files(files):
    require(isinstance(files, dict) and files, "Empty sealed file inventory")
    for path, digest in files.items():
        require(Path(path).is_absolute() and Path(path).is_file(), "Missing absolute sealed file")
        require(operations.sha(path) == publication.validate_sha(digest), "Sealed file changed: " + str(path))


def authenticate_job(config, job):
    sealed_files(config["files"])
    sealed_files(job["files"])
    require(job["input_files"], "Empty job input inventory")
    for path, receipt in job["input_files"].items():
        p = Path(path)
        require(p.is_absolute() and p.is_file() and p.stat().st_size == receipt["bytes"]
                and operations.sha(p) == receipt["sha256"], "Input identity changed: " + p.name)


def validate_config(path, expected_sha):
    path = Path(path).resolve()
    require(operations.sha(path) == publication.validate_sha(expected_sha), "Campaign SHA256 mismatch")
    config = read_json(path)
    require(config.get("schema") == SCHEMA, "Unknown campaign schema")
    cid = config.get("campaign_id", "")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,100}", cid), "Invalid campaign ID")
    root = path.parent
    require(not any(p.is_symlink() for p in (root, *root.parents)), "Campaign directory contains a symlink")
    resources = config.get("resources", {})
    require(type(resources.get("max_concurrent")) is int and 1 <= resources["max_concurrent"] <= 2,
            "This controller permits one or two concurrent jobs")
    for field in ("memory_budget_gib", "reserve_available_memory_gib", "min_free_disk_gib", "poll_seconds", "admission_timeout_seconds"):
        require(type(resources.get(field)) in (int, float) and resources[field] > 0, "Invalid resource field: " + field)
    require(resources["poll_seconds"] <= 30, "Resource poll interval exceeds 30 seconds")
    require(type(resources.get("campaign_timeout_seconds", 345600)) in (int, float)
            and resources.get("campaign_timeout_seconds", 345600) > 0, "Invalid campaign timeout")
    env = config.get("environment", {})
    require(isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), "Invalid environment")
    require(not {"HOME", "home", "CODEX_HOME", "PATH"} & set(env), "Do not repurpose system environment variables")
    mode = config.get("mode", "PRODUCTION")
    require(mode in ("PRODUCTION", "SYNTHETIC_TEST"), "Unknown execution mode")
    ids, directories, destinations = set(), set(), set()
    require(isinstance(config.get("jobs"), list) and config["jobs"], "No jobs configured")
    for job in config["jobs"]:
        jid = job.get("id", "")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", jid) and jid not in ids, "Invalid or duplicate job ID")
        ids.add(jid)
        directory = Path(job["run_dir"])
        require(directory.is_absolute() and directory.resolve().is_relative_to(root) and directory.resolve() != root
                and directory.is_dir() and directory.resolve() == directory, "Job directory must be a real campaign subdirectory")
        require(str(directory) not in directories, "Jobs share a working directory")
        directories.add(str(directory))
        require(job["container_label"] == f"dnabr_r02_campaign={cid}.{jid}", "Container label is not uniquely owned by this campaign job")
        for name in ("memory_gib", "scratch_gib"):
            require(type(job.get(name)) in (int, float) and job[name] > 0, "Invalid declared job resource")
        require(job["memory_gib"] <= resources["memory_budget_gib"], "Job exceeds campaign memory budget")
        for stage in ("geometry", "annotation"):
            entry = job.get(stage, {})
            command = entry.get("command")
            require(isinstance(command, list) and command and all(isinstance(arg, str) and arg for arg in command), "Stage command must be a nonempty argv list")
            require(Path(command[0]).is_absolute(), "Command executable must be absolute")
            require(mode == "SYNTHETIC_TEST" or Path(command[0]).name == "nextflow", "Production analysis must run through Nextflow")
            require(type(entry.get("timeout_seconds")) in (int, float) and entry["timeout_seconds"] > 0, "Missing stage timeout")
            pattern = entry.get("receipt_glob" if stage == "geometry" else "manifest_glob", "")
            require(pattern and not Path(pattern).is_absolute() and ".." not in Path(pattern).parts, "Unsafe stage receipt pattern")
        expected = job["geometry"].get("expected", {})
        require(str(expected.get("chrom")) in {str(c) for c in range(1, 23)}, "Invalid expected chromosome")
        for field in ("n_samples", "input_chain_rows", "max_active_intervals", "block_bp"):
            require(type(expected.get(field)) is int and expected[field] > 0, "Invalid geometry expectation: " + field)
        require(job["annotation"].get("expected_status") == "COMPLETE_DESCRIPTIVE_NOT_VALIDATED", "Unsupported scientific status")
        destination = publication.destination_prefix(job["destination"])
        require(destination not in destinations, "Jobs share a publication destination")
        destinations.add(destination)
        require(type(job.get("publication_timeout_seconds", 14400)) in (int, float)
                and job.get("publication_timeout_seconds", 14400) > 0, "Invalid publication timeout")
        require(isinstance(job.get("files"), dict) and job["files"] and isinstance(job.get("input_files"), dict), "Missing job seal")
    for external in config.get("external_reservations", []):
        require(Path(external["status_file"]).is_absolute() and external.get("release_stages") == [SUCCESS],
                "External reservation releases only on explicit completed publication")
        require(type(external.get("slots")) is int and 0 < external["slots"] < resources["max_concurrent"], "Invalid external slot reservation")
        require(external.get("memory_gib", 0) >= 0 and external.get("scratch_gib", 0) >= 0, "Invalid external resource reservation")
    sealed_files(config["files"])
    return config, root


def available_memory_bytes():
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine available memory")


def external_reservations(config):
    active = []
    for entry in config.get("external_reservations", []):
        try:
            status = read_json(entry["status_file"])
            released = isinstance(status, dict) and status.get("stage") in entry["release_stages"]
        except (OSError, ValueError, TypeError):
            released = False
        if not released:
            active.append(entry)
    return active


def admission(config, job, running_jobs, root, *, free_bytes=None, memory_bytes=None):
    """Conservative reservation: current free space minus full new peak estimates.

    Existing running jobs retain their reservation until publication ends. This
    may underuse disk but never spends the same remaining headroom twice.
    External scratch_gib is explicitly its remaining, not already used, demand.
    """
    r = config["resources"]
    external = external_reservations(config)
    slots = len(running_jobs) + sum(item["slots"] for item in external)
    memory = sum(item["memory_gib"] for item in running_jobs) + sum(item["memory_gib"] for item in external)
    scratch = sum(item["scratch_gib"] for item in running_jobs) + sum(item["scratch_gib"] for item in external)
    free = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    available = available_memory_bytes() if memory_bytes is None else memory_bytes
    reasons = []
    if slots >= r["max_concurrent"]:
        reasons.append("concurrency_reservation")
    if memory + job["memory_gib"] > r["memory_budget_gib"]:
        reasons.append("declared_memory_budget")
    if available < (job["memory_gib"] + r["reserve_available_memory_gib"]) * GIB:
        reasons.append("available_memory")
    if free < (scratch + job["scratch_gib"] + r["min_free_disk_gib"]) * GIB:
        reasons.append("available_disk_with_reservations")
    return not reasons, dict(reasons=reasons, free_disk_bytes=free, available_memory_bytes=available,
                             occupied_slots=slots, reserved_scratch_gib=scratch, reserved_memory_gib=memory)


def stop_process_group(process, grace_seconds=2):
    """Only a child started with start_new_session=True is accepted here."""
    try:
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            group = process.pid  # Its children can still occupy our original group.
        require(group == process.pid, "Refusing to signal an unowned process group")
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait(timeout=1)
            return
        process.poll()
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def cleanup_containers(label):
    result = subprocess.run(["docker", "ps", "-q", "--filter", "label=" + label],
                            text=True, capture_output=True, timeout=15, check=False)
    require(result.returncode == 0, "Cannot inspect containers owned by this job")
    identifiers = result.stdout.split()
    require(all(re.fullmatch(r"[0-9a-f]{12,64}", item) for item in identifiers), "Invalid Docker IDs")
    if identifiers:
        subprocess.run(["docker", "stop", "--time", "10", *identifiers], check=True, timeout=45)


def run_process(command, cwd, log_path, timeout_seconds, stop, env):
    started = time.monotonic()
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            while True:
                if stop.is_set():
                    raise InterruptedError("Campaign interrupted")
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError("Stage wall-clock limit exceeded")
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise subprocess.CalledProcessError(code, command)
                    return {"pid": process.pid, "exit_code": code, "elapsed_seconds": time.monotonic() - started}
                stop.wait(0.1)
        except BaseException:
            stop_process_group(process)
            raise


def one_receipt(directory, pattern):
    paths = list(directory.glob(pattern))
    require(len(paths) == 1 and paths[0].is_file() and paths[0].resolve().is_relative_to(directory),
            "Expected one unambiguous completed stage receipt")
    return paths[0]


def authenticated_stage_inputs(data, job):
    require(isinstance(data.get("inputs"), dict) and data["inputs"], "Stage receipt lacks source identity")
    for value in data["inputs"].values():
        declared = job["input_files"].get(value["path"])
        require(declared is not None and declared["sha256"] == value["sha256"], "Stage input differs from sealed job")


def accept_geometry(job):
    path = one_receipt(Path(job["run_dir"]), job["geometry"]["receipt_glob"])
    data, expected = read_json(path), job["geometry"]["expected"]
    counts = data.get("checks", {}).get("counts", {})
    require(data.get("status") == "PREFLIGHT_GEOMETRY_ONLY_NOT_EVIDENCE"
            and str(data.get("chrom")) == str(expected["chrom"])
            and data.get("n_samples") == expected["n_samples"]
            and counts.get("input_chain_rows") == expected["input_chain_rows"]
            and counts.get("blocks_exceeding_active_row_bound", 0) == 0
            and 0 < counts.get("peak_active_chain_rows_upper_bound", 0) <= expected["max_active_intervals"]
            and data.get("parameters", {}).get("block_bp") == expected["block_bp"],
            "Geometry scope or resource acceptance failed")
    authenticated_stage_inputs(data, job)
    return path


def accept_annotation(job):
    path = one_receipt(Path(job["run_dir"]), job["annotation"]["manifest_glob"])
    data, expected = read_json(path), job["geometry"]["expected"]
    require(data.get("schema") == "r02_segment_evidence_v1"
            and data.get("status") == job["annotation"]["expected_status"]
            and str(data.get("chrom")) == str(expected["chrom"])
            and data.get("n_samples") == expected["n_samples"]
            and data.get("checks", {}).get("counts", {}).get("input_chain_rows") == expected["input_chain_rows"],
            "Annotation scope differs from sealed job")
    authenticated_stage_inputs(data, job)
    publication.authenticated_bundle(path.parent, operations.sha(path))
    return path


def completed_stage(directory, stage, campaign_sha, accept, job):
    checkpoint = directory / f"{stage}.complete.json"
    if not checkpoint.exists():
        return None
    data = read_json(checkpoint)
    require(data.get("campaign_sha256") == campaign_sha, "Completed stage belongs to another campaign configuration")
    path = accept(job)
    require(str(path) == data.get("receipt") and operations.sha(path) == data.get("receipt_sha256"), "Completed stage receipt changed")
    return path


def next_attempt(directory, stage):
    attempts = []
    for path in directory.glob(f"attempt*.{stage}.log"):
        match = re.fullmatch(r"attempt([0-9]+)\." + stage + r"\.log", path.name)
        if match:
            attempts.append(int(match[1]))
    return max(attempts, default=0) + 1


def verify_publication_receipt(path, manifest, destination):
    receipt = read_json(path)
    require(receipt.get("schema") == publication.SCHEMA and receipt.get("status") == "PUBLISHED_VERIFIED"
            and receipt.get("manifest_sha256") == operations.sha(manifest)
            and receipt.get("destination") == publication.destination_prefix(destination), "Publication receipt scope mismatch")
    return receipt


def publish_bundle(job, manifest, directory, env, stop):
    receipt = directory / "publication.json"
    existed = receipt.exists()
    if not existed:
        attempt = next_attempt(directory, "publication")
        command = [sys.executable, str(Path(publication.__file__).resolve()), "--output-dir", str(manifest.parent),
                   "--expected-manifest-sha256", operations.sha(manifest), "--destination", job["destination"], "--receipt", str(receipt)]
        run_process(command, directory, directory / f"attempt{attempt:02d}.publication.log",
                    job.get("publication_timeout_seconds", 14400), stop, env)
    record = verify_publication_receipt(receipt, manifest, job["destination"])
    if existed:
        require(record.get("files"), "Prior publication receipt has no pinned objects")
        _, _, local = publication.authenticated_bundle(manifest.parent, operations.sha(manifest))
        local = {item["name"]: item for item in local}
        require(len(record["files"]) == len(local), "Prior publication inventory differs from local bundle")
        seen = set()
        for item in record["files"]:
            require(not stop.is_set(), "Campaign interrupted during publication verification")
            name = item.get("name")
            require(name in local and name not in seen and all(item.get(key) == local[name][key]
                    for key in ("bytes", "sha256", "md5_base64"))
                    and item.get("uri") == job["destination"].rstrip("/") + "/" + name,
                    "Prior publication object differs from authenticated local source")
            seen.add(name)
            publication.verify_remote(item, pinned=True)
    return record


def execute_job(config, job, campaign_sha, stop, *, resume=False):
    directory = Path(job["run_dir"])
    # A duplicate invoker must not overwrite the active owner's status.
    with operations.execution_lock(directory):
        if (directory / "status.json").exists() and not resume:
            raise ValueError("Existing job state requires explicit --resume")
        started_analysis = False
        env = {**os.environ, **config.get("environment", {})}
        try:
            authenticate_job(config, job)
            for stage, accept in (("geometry", accept_geometry), ("annotation", accept_annotation)):
                path = completed_stage(directory, stage, campaign_sha, accept, job) if resume else None
                if path is not None:
                    continue
                authenticate_job(config, job)
                attempt = next_attempt(directory, stage)
                if stage == "annotation" and attempt > 1:
                    partial_pattern = str(Path(job[stage]["manifest_glob"]).parent)
                    require(not list(directory.glob(partial_pattern)),
                            "Uncheckpointed annotation output preserved; explicit recovery amendment required, not automatic resume")
                trace = directory / f"attempt{attempt:02d}.{stage}.trace.tsv"
                command = [str(trace) if arg == "{trace}" else arg for arg in job[stage]["command"]]
                if attempt > 1:
                    require(resume, "Prior stage attempt requires explicit --resume")
                    command += ["-resume"]
                operations.status(directory, "RUNNING_" + stage.upper(), job_id=job["id"],
                                  campaign_sha256=campaign_sha, attempt=attempt, command=command)
                started_analysis = True
                result = run_process(command, directory, directory / f"attempt{attempt:02d}.{stage}.log",
                                     job[stage]["timeout_seconds"], stop, env)
                authenticate_job(config, job)
                path = accept(job)
                operations.write_new(directory / f"{stage}.complete.json",
                                     dict(campaign_sha256=campaign_sha, receipt=str(path), receipt_sha256=operations.sha(path),
                                          completed_utc=operations.now(), process=result))
            manifest = accept_annotation(job)
            authenticate_job(config, job)
            if config.get("mode", "PRODUCTION") != "SYNTHETIC_TEST":
                cleanup_containers(job["container_label"])
            operations.status(directory, "PUBLISHING", job_id=job["id"], campaign_sha256=campaign_sha, manifest=str(manifest))
            receipt = publish_bundle(job, manifest, directory, env, stop)
            operations.status(directory, SUCCESS, job_id=job["id"], campaign_sha256=campaign_sha, manifest=str(manifest),
                              manifest_sha256=operations.sha(manifest), destination=receipt["destination"],
                              publication_receipt_sha256=operations.sha(directory / "publication.json"),
                              biological_validation_complete=False)
            return {"stage": SUCCESS, "manifest": str(manifest), "destination": receipt["destination"]}
        except BaseException as error:
            cleanup_error = None
            if started_analysis and config.get("mode", "PRODUCTION") != "SYNTHETIC_TEST":
                try:
                    cleanup_containers(job["container_label"])
                except Exception as cleanup:
                    cleanup_error = str(cleanup)
                    stop.set()  # Capacity is not safely released if ownership cleanup failed.
            operations.status(directory, "STOPPED_WITH_ERROR", job_id=job["id"], campaign_sha256=campaign_sha, error_type=type(error).__name__,
                              error=str(error), cleanup_error=cleanup_error, partial_outputs_not_adoptable=True)
            raise


def run_campaign(path, expected_sha, *, resume=False, stop=None):
    config, root = validate_config(path, expected_sha)
    stop = stop or threading.Event()
    with operations.execution_lock(root):
        state = root / "status.json"
        if state.exists():
            require(resume, "Existing campaign state requires explicit --resume")
            prior = read_json(state)
            require(prior.get("campaign_sha256") == expected_sha, "Campaign configuration changed since prior attempt")
        pending = list(config["jobs"])
        running, results = {}, {}
        waiting_since = started = time.monotonic()
        with ThreadPoolExecutor(max_workers=config["resources"]["max_concurrent"]) as pool:
            try:
                while pending or running:
                    require(operations.sha(path) == expected_sha, "Campaign configuration changed while running")
                    if time.monotonic() - started >= config["resources"].get("campaign_timeout_seconds", 345600):
                        stop.set()
                    for future in list(running):
                        if not future.done():
                            continue
                        job = running.pop(future)
                        try:
                            results[job["id"]] = future.result()
                        except BaseException as error:
                            results[job["id"]] = {"stage": "STOPPED_WITH_ERROR", "error_type": type(error).__name__, "error": str(error)}
                        waiting_since = time.monotonic()
                    if stop.is_set():
                        for job in pending:
                            results[job["id"]] = {"stage": "NOT_STARTED_INTERRUPTED"}
                        pending.clear()
                    reasons = {}
                    for job in list(pending):
                        allowed, detail = admission(config, job, list(running.values()), root)
                        reasons[job["id"]] = detail
                        if not allowed:
                            # Preserve the declared order: bypassing a large job
                            # can leave insufficient disk after smaller outputs accumulate.
                            break
                        future = pool.submit(execute_job, config, job, expected_sha, stop, resume=resume)
                        running[future] = job
                        pending.remove(job)
                        waiting_since = time.monotonic()
                    if pending and not running:
                        if external_reservations(config):
                            # An authorized external job may legitimately own
                            # the needed slot for hours; the global deadline
                            # still bounds an absent or never-completed receipt.
                            waiting_since = time.monotonic()
                        elif time.monotonic() - waiting_since >= config["resources"]["admission_timeout_seconds"]:
                            blocked_head = pending[0]["id"]
                            for job in pending:
                                results[job["id"]] = {"stage": "NOT_STARTED_RESOURCE_BLOCKED", "blocked_queue_head": blocked_head,
                                                      "admission": reasons.get(blocked_head)}
                            pending.clear()
                    operations.status(root, "RUNNING" if running else "WAITING_FOR_RESOURCES", campaign_sha256=expected_sha,
                                      campaign_id=config["campaign_id"], running=[job["id"] for job in running.values()],
                                      pending=[job["id"] for job in pending], results=results, admission=reasons,
                                      biological_validation_complete=False)
                    if pending or running:
                        # An interruption should wake workers, not busy-loop the supervisor.
                        time.sleep(min(0.1, config["resources"]["poll_seconds"]) if stop.is_set()
                                   else config["resources"]["poll_seconds"])
            except BaseException as error:
                stop.set()
                operations.status(root, "STOPPED_WITH_ERROR", campaign_sha256=expected_sha,
                                  campaign_id=config["campaign_id"], error_type=type(error).__name__, error=str(error),
                                  partial_outputs_not_adoptable=True, biological_validation_complete=False)
                raise
        complete = len(results) == len(config["jobs"]) and all(row["stage"] == SUCCESS for row in results.values())
        stage = "COMPLETE_PUBLISHED_ALL_JOBS" if complete else "PARTIAL_OR_STOPPED"
        operations.status(root, stage, campaign_sha256=expected_sha, campaign_id=config["campaign_id"],
                          running=[], pending=[], results=results, biological_validation_complete=False)
        return {"stage": stage, "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.validate_only:
        config, root = validate_config(args.campaign, args.expected_sha256)
        for job in config["jobs"]:
            authenticate_job(config, job)
        print(json.dumps({"stage": "CONFIGURATION_AND_INPUTS_VERIFIED_NOT_EXECUTED", "jobs": len(config["jobs"])}))
        return 0
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda signum, frame: stop.set())
    try:
        result = run_campaign(args.campaign, args.expected_sha256, resume=args.resume, stop=stop)
    except BaseException:
        stop.set()
        raise
    return 0 if result["stage"] == "COMPLETE_PUBLISHED_ALL_JOBS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
