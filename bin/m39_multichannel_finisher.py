#!/usr/bin/env python3
"""Bounded, detached local reporting companion for one multichannel campaign.

The campaign owns A -> audit -> B -> audit and all cloud cleanup. This companion
only waits, runs the existing primary reporter, and reads exact owned-resource
state. It never launches training, cancels cloud jobs, publishes, or opens SCORE.
It survives SSH/chat disconnects through tmux, not shutdown of the development VM.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from m39_gpu_launch import native_auth, native_env_prefix, observed_job_ids, validate_target, write_json
from m39_launch_ordered_training import sha256
from m39_ordered_campaign import private_path

SCHEMA = "m39-multichannel-finisher-v1"
SUCCESS = "TWO_STAGES_AUDITED_EXPLORATORY_ONLY"
REPORT_FILES = ("report.json", "report.md", "cases.csv", "followup-table.csv", "contrasts.csv", "learning-curves.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    require(path.is_file() and not path.is_symlink(), "regular non-symlinked JSON source required")
    with path.open() as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "JSON object required")
    return value


def source_inventory(directory: Path) -> dict:
    require(directory.is_dir() and not directory.is_symlink(), "frozen source directory required")
    files = sorted(directory.glob("*.py"))
    require(all(path.is_file() and not path.is_symlink() for path in files), "symlinked frozen source")
    result = {path.name: sha256(path) for path in files}
    require({"m39_multichannel_finisher.py", "m39_ordered_multichannel_report.py",
             "m39_ordered_multichannel_sweep.py", "m39_ordered_multichannel_training.py"} <= set(result),
            "frozen finisher/report source closure missing")
    return result


def configuration(campaign: Path, source_dir: Path, calibration_dir: Path, calibration_hash: str) -> dict:
    cfg = read_json(campaign)
    require(cfg.get("schema_version") == "m39-multichannel-gpu-campaign-v1", "multichannel campaign required")
    repo = Path(cfg["repository"]).resolve(strict=True)
    run = private_path(cfg["run_dir"], repo)
    require(private_path(str(campaign), repo).parent == run, "campaign must belong to exact private run")
    source = private_path(str(source_dir), repo)
    calibration = private_path(str(calibration_dir), repo)
    require(source.is_dir() and calibration.is_dir(), "source and calibration directories required")
    require(re.fullmatch(r"[0-9a-f]{40}", cfg["source_commit"]) is not None, "full source commit required")
    require(type(cfg["wall_timeout_seconds"]) is int and 60 <= cfg["wall_timeout_seconds"] <= 46800,
            "campaign deadline differs")
    sources = source_inventory(source)
    for relative, digest in cfg["source_sha256"].items():
        if relative.startswith("bin/"):
            require(sources.get(Path(relative).name) == digest, "frozen campaign source differs")
    require(re.fullmatch(r"[0-9a-f]{64}", calibration_hash) is not None
            and sha256(calibration / "report.json") == calibration_hash, "calibration report hash differs")
    require(re.fullmatch(r"us-central1-docker\.pkg\.dev/uspbr-242713/dnabr-lai/[a-z0-9-]+@sha256:[a-f0-9]{64}",
                         cfg["cpu_image"]) is not None, "pinned project CPU image required")
    for stage in (cfg["screen"], cfg["followup"]):
        stage_run = private_path(stage["run_dir"], repo)
        require(stage_run != run, "training stage cannot be the campaign root")
        validate_target(stage_run.name, cfg["gpu_image"])
    require(cfg["screen"]["run_dir"] != cfg["followup"]["run_dir"], "stage run IDs overlap")
    return {"cfg": cfg, "run": run, "repo": repo, "source_dir": source,
            "calibration_dir": calibration, "calibration_report_sha256": calibration_hash,
            "campaign_sha256": sha256(campaign), "source_sha256": sources}


def wait_terminal(path: Path, campaign_hash: str, source_commit: str, seconds: int, poll_seconds: int) -> dict:
    require(type(seconds) is int and 1 <= seconds <= 47400
            and type(poll_seconds) is int and 1 <= poll_seconds <= 60, "finisher wait ceilings differ")
    deadline = time.monotonic() + seconds
    while True:
        if path.exists():
            try:
                terminal = read_json(path)
            except json.JSONDecodeError:
                # The campaign's exclusive writer may have created but not flushed
                # its file yet. A permanently malformed receipt remains a failure.
                terminal = None
            if terminal is not None:
                require(terminal.get("schema_version") == "m39-ordered-campaign-completion-v1"
                        and terminal.get("campaign_sha256") == campaign_hash
                        and terminal.get("source_commit") == source_commit
                        and terminal.get("SCORE_opened") is False,
                        "terminal campaign binding or scope differs")
                return terminal
        remaining = deadline - time.monotonic()
        require(remaining > 0, "no valid terminal campaign receipt before finisher deadline")
        time.sleep(min(poll_seconds, remaining))


def event_value(run: Path, event: str, key: str) -> str:
    matches = list(run.glob(f"event-*-{event}.json"))
    require(len(matches) == 1, f"missing or duplicate campaign event {event}")
    value = read_json(matches[0])
    require(value.get("event") == event and isinstance(value.get(key), str), "campaign event fields differ")
    return value[key]


def report_command(bound: dict, terminal: dict, report_root: Path) -> list[str]:
    cfg, run, repo = bound["cfg"], bound["run"], bound["repo"]
    require(terminal["status"] == SUCCESS, "both audited stages required before final report")
    paths = {"screen-plan": Path(cfg["screen"]["plan"]),
             "followup-plan": run / "cpu-prepare-b/result/plan.json"}
    for short, long in (("a", "screen"), ("b", "followup")):
        outputs = private_path(event_value(run, f"stage-{short}-downloaded", "outputs"), repo)
        comparison = private_path(terminal[f"comparison_{short}"], repo)
        require(outputs == run / f"primary-{short}"
                and comparison == run / f"cpu-audit-{short}/result/comparison.json",
                "campaign primary/audit location differs")
        require(private_path(event_value(run, f"stage-{short}-start", "receipt"), repo)
                == Path(cfg[long]["run_dir"]) / "gpu.launch.json", "stage receipt location differs")
        paths[f"{long}-outputs"], paths[f"{long}-comparison"] = outputs, comparison
    mounts = {"/code": bound["source_dir"], "/calibration": bound["calibration_dir"]}
    arguments = []
    for label, source in paths.items():
        private_path(str(source), repo)
        # Plans contain relative neighboring config filenames, so mount their directory.
        if label.endswith("-plan"):
            target = "/" + label
            mounts[target] = source.parent
            arguments += ["--" + label, target + "/" + source.name]
        else:
            mounts["/" + label] = source
            arguments += ["--" + label, "/" + label]
    command = ["docker", "run", "--rm", "--network", "none", "--cpus", "2", "--memory", "4g",
               "--cidfile", str(report_root / "report.cid"), "--user", f"{os.getuid()}:{os.getgid()}",
               "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=/code",
               "--env", "TORCHINDUCTOR_CACHE_DIR=/tmp/cache", "--env", "OMP_NUM_THREADS=2",
               "--env", "OPENBLAS_NUM_THREADS=2"]
    for target, source in mounts.items():
        require(source.exists() and not source.is_symlink(), "missing or symlinked report mount")
        command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
    return command + ["--mount", f"type=bind,src={report_root},dst=/output", cfg["cpu_image"],
        "python3", "/code/m39_ordered_multichannel_report.py", *arguments,
        "--calibration-dir", "/calibration", "--calibration-report-sha256", bound["calibration_report_sha256"],
        "--outdir", "/output/report"]


def owned_workers(cfg: dict, *, timeout_seconds: int, require_success: bool = False) -> dict:
    """Read-only terminal check; auth failures are never evidence of zero workers."""
    auth = native_auth(Path(cfg["native_auth_dir"]), cfg["service_account"], Path(cfg["repository"]))
    prefix = [*native_env_prefix(auth), "gcloud", f"--account={auth['service_account']}"]
    deadline, result = time.monotonic() + timeout_seconds, {}

    def query(arguments):
        remaining = deadline - time.monotonic()
        require(remaining > 0, "owned-worker verification deadline exceeded")
        return json.loads(subprocess.check_output([*prefix, *arguments], timeout=min(30, remaining),
                                                  text=True, stderr=subprocess.STDOUT))

    for role in ("screen", "followup"):
        stage = Path(cfg[role]["run_dir"])
        run_id = stage.name
        receipt_path = stage / "gpu.launch.json"
        expected = {}
        declared = None
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            require(receipt.get("run_id") == run_id and receipt.get("source_commit") == cfg["source_commit"]
                    and receipt.get("process_name") == "M39_ORDERED_GPU_TRAINING"
                    and receipt.get("SCORE_staged") is False, "owned stage launch binding differs")
            expected = observed_job_ids(stage, run_id, process_name=receipt["process_name"],
                                        max_jobs=receipt["resources"]["max_jobs"])
            declared = receipt["resources"]["max_jobs"]
        if require_success:
            require(declared is not None and len(expected) == declared
                    and all(uid is not None for uid in expected.values()),
                    "successful campaign lacks complete native job IDs/UIDs")
        jobs, absent = [], []
        # Broad Batch list can paginate the whole project even with a label
        # filter. Describe only IDs emitted by this exact stage, under one budget.
        for name, uid in expected.items():
            try:
                job = query(["batch", "jobs", "describe", name, "--project=uspbr-242713",
                             "--location=us-central1", "--format=json"])
            except subprocess.CalledProcessError as exc:
                if re.search(r"\bNOT_FOUND\b|\b404\b", str(exc.output)):
                    absent.append(name)
                    continue
                raise
            require(isinstance(job, dict), "Batch description must be an object")
            require(job.get("labels", {}).get("m39_run") == run_id
                    and job["labels"].get("team") == "frank", "Batch inventory ownership differs")
            require(job.get("name", "").rsplit("/", 1)[-1] == name and bool(job.get("uid")),
                    "Batch job identity missing or different")
            if uid is not None:
                require(uid == job["uid"], "Batch native UID differs")
            state = job.get("status", {}).get("state")
            require(state in (("SUCCEEDED",) if require_success else ("SUCCEEDED", "FAILED")),
                    "owned Batch job is not terminal")
            jobs.append({"name": name, "uid": job["uid"], "state": state})
        machines = query(["compute", "instances", "list", "--project=uspbr-242713",
                         f"--filter=labels.m39_run={run_id}", "--format=json(name,status,labels)"])
        require(isinstance(machines, list) and all(vm.get("labels", {}).get("m39_run") == run_id
                and vm["labels"].get("team") == "frank" for vm in machines), "VM inventory ownership differs")
        require(all(vm.get("status") == "TERMINATED" for vm in machines), "owned workers are not retired")
        require(not require_success or not absent, "successful campaign contains absent native jobs")
        result[role] = {"run_id": run_id, "terminal_jobs_present": len(jobs),
                        "observed_native_ids": len(expected), "nonretired_instances": 0,
                        "declared_jobs": declared, "verified_jobs": jobs, "native_jobs_returned_NOT_FOUND": absent,
                        "no_native_ids_is_not_proof_of_no_jobs": not bool(expected)}
    return {"status": "OWNED_JOBS_TERMINAL_WORKERS_RETIRED", "stages": result,
            "read_only": True, "checked_utc": utc_now()}


def watch(args, bound: dict) -> dict:
    run = bound["run"]
    require(not (run / "finisher.completion.json").exists(), "finisher already completed")
    result = {"schema_version": SCHEMA, "status": "FAILED_PRESERVED_FOR_REVIEW", "SCORE_opened": False,
              "campaign_sha256": bound["campaign_sha256"], "report_status": "NOT_RUN",
              "publication_status": "LOCAL_ONLY", "independent_scientific_POST": "NOT_AN_AUTOMATIC_COUNCIL_VERDICT"}
    report_root = run / "final-reading"
    report_started = False
    try:
        launch = read_json(run / "finisher.launch.json")
        require(launch["campaign_sha256"] == bound["campaign_sha256"]
                and launch["source_sha256"] == source_inventory(bound["source_dir"])
                and launch["calibration_report_sha256"] == sha256(bound["calibration_dir"] / "report.json"),
                "finisher launch bindings changed")
        require(type(launch["report_timeout_seconds"]) is int
                and 1 <= launch["report_timeout_seconds"] <= 1800, "finisher report ceiling differs")
        terminal = wait_terminal(run / "campaign.completion.json", bound["campaign_sha256"],
            bound["cfg"]["source_commit"], launch["wait_seconds"], launch["poll_seconds"])
        require(sha256(args.campaign) == bound["campaign_sha256"]
                and source_inventory(bound["source_dir"]) == launch["source_sha256"],
                "campaign or frozen source changed while waiting")
        result["campaign_status"] = terminal["status"]
        result["campaign_completion_sha256"] = sha256(run / "campaign.completion.json")
        if terminal["status"] != SUCCESS:
            result.update(status="CAMPAIGN_FAILED_OR_INCOMPLETE", campaign_failure=terminal.get("failure"))
        else:
            report_root.mkdir(mode=0o700, exist_ok=False)
            command = report_command(bound, terminal, report_root)
            write_json(report_root / "command.json", {"argv": command, "timeout_seconds": launch["report_timeout_seconds"]})
            with (report_root / "controller.log").open("x") as log:
                report_started = True
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=launch["report_timeout_seconds"])
            report = read_json(report_root / "report/report.json")
            require(report.get("schema_version") == "m39-ordered-multichannel-report-v1"
                    and report.get("status") == "AUDITED_EXPLORATORY_SELECT_ONLY"
                    and report.get("generator_sha256") == launch["source_sha256"]["m39_ordered_multichannel_report.py"],
                    "final report schema/status/source differs")
            result.update(status="FINAL_REPORT_VERIFIED", report_status=report["status"], counts=report["counts"],
                report_files_sha256={name: sha256(report_root / "report" / name) for name in REPORT_FILES})
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        result.update(failure_type=type(exc).__name__, failure=str(exc))
    finally:
        try:
            cidfile = report_root / "report.cid"
            if report_started and cidfile.is_file() and not cidfile.is_symlink():
                identifier = cidfile.read_text().strip()
                require(re.fullmatch(r"[0-9a-f]{64}", identifier) is not None, "invalid own report container ID")
                subprocess.run(["docker", "rm", "-f", identifier], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30, check=False)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result.update(status=("REPORT_VERIFIED_CONTAINER_CLEANUP_PENDING" if result["report_status"] ==
                          "AUDITED_EXPLORATORY_SELECT_ONLY" else "FAILED_PRESERVED_FOR_REVIEW"),
                          container_cleanup_failure=str(exc))
        try:
            result["owned_worker_check"] = owned_workers(bound["cfg"], timeout_seconds=120,
                require_success=result.get("campaign_status") == SUCCESS)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            result.update(status=("REPORT_VERIFIED_EXTERNAL_CHECK_PENDING" if result["report_status"] ==
                          "AUDITED_EXPLORATORY_SELECT_ONLY" else "FAILED_PRESERVED_FOR_REVIEW"),
                          owned_worker_check={"status": "UNVERIFIED_EXTERNAL_STATE", "failure": str(exc)})
        result["finished_utc"] = utc_now()
        write_json(run / "finisher.completion.json", result)
    print(json.dumps(result), flush=True)
    return result


def bootstrap_failure(args, error: Exception) -> None:
    """Persist a pre-watch failure only in the launch-bound private campaign."""
    cfg = read_json(args.campaign)
    repo = Path(cfg["repository"]).resolve(strict=True)
    campaign = private_path(str(args.campaign), repo)
    run = private_path(cfg["run_dir"], repo)
    require(campaign.parent == run, "bootstrap campaign directory differs")
    launch = read_json(run / "finisher.launch.json")
    argv = launch["argv"]
    require(launch.get("schema_version") == SCHEMA and isinstance(argv, list)
            and argv.count("--campaign") == 1
            and argv[argv.index("--campaign") + 1] == str(campaign), "bootstrap launch is not bound to campaign")
    require(not (run / "finisher.completion.json").exists(), "finisher already completed")
    write_json(run / "finisher.completion.json", {"schema_version": SCHEMA,
        "status": "BOOTSTRAP_FAILED_PRESERVED_FOR_REVIEW", "failure_type": type(error).__name__,
        "failure": str(error), "campaign_sha256": launch["campaign_sha256"],
        "observed_campaign_sha256": sha256(campaign), "report_status": "NOT_RUN", "SCORE_opened": False,
        "owned_worker_check": {"status": "UNVERIFIED_BOOTSTRAP_FAILED"}, "finished_utc": utc_now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "source-dir", "calibration-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--calibration-report-sha256", required=True)
    parser.add_argument("--wait-seconds", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--report-timeout-seconds", type=int, default=1200)
    parser.add_argument("--watch", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        bound = configuration(args.campaign, args.source_dir, args.calibration_dir, args.calibration_report_sha256)
        require(bound["source_sha256"]["m39_multichannel_finisher.py"] == sha256(Path(__file__)),
                "executing finisher differs from frozen source")
    except (OSError, ValueError, KeyError) as exc:
        if args.watch:
            bootstrap_failure(args, exc)
        raise
    if args.watch:
        def interrupt(signum, frame):
            raise InterruptedError(f"Finisher received signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        result = watch(args, bound)
        raise SystemExit(0 if result["status"] == "FINAL_REPORT_VERIFIED" else 1)
    run = bound["run"]
    wait = bound["cfg"]["wall_timeout_seconds"] + 600 if args.wait_seconds is None else args.wait_seconds
    require(type(wait) is int and 1 <= wait <= 47400 and 1 <= args.poll_seconds <= 60
            and 1 <= args.report_timeout_seconds <= 1800, "finisher runtime ceilings differ")
    require(not (run / "finisher.completion.json").exists(), "finisher already completed")
    command = [sys.executable, str(bound["source_dir"] / "m39_multichannel_finisher.py"), "--watch",
        "--campaign", str(args.campaign.resolve()), "--source-dir", str(bound["source_dir"]),
        "--calibration-dir", str(bound["calibration_dir"]),
        "--calibration-report-sha256", args.calibration_report_sha256]
    marker = {"schema_version": SCHEMA, "campaign_sha256": bound["campaign_sha256"],
        "source_sha256": bound["source_sha256"], "calibration_report_sha256": args.calibration_report_sha256,
        "wait_seconds": wait, "poll_seconds": args.poll_seconds,
        "report_timeout_seconds": args.report_timeout_seconds,
        "owned_worker_check_timeout_seconds": 120, "container_cleanup_timeout_seconds": 30,
        "launched_utc": utc_now(), "argv": command}
    write_json(run / "finisher.launch.json", marker)
    shell = shlex.join(command) + " > " + shlex.quote(str(run / "finisher.controller.log")) + " 2>&1"
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", run.name + "-report", "-c", str(run), shell],
                       check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        write_json(run / "finisher.completion.json", {"schema_version": SCHEMA, "status": "DETACH_FAILED",
            "failure": str(exc), "SCORE_opened": False, "finished_utc": utc_now()})
        raise
    print(json.dumps({"status": "DETACHED_MULTICHANNEL_FINISHER_STARTED", "run": str(run)}))


if __name__ == "__main__":
    main()
