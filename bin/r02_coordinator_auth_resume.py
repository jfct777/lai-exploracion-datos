#!/usr/bin/env python3
"""Resume an authenticated coordinator after its explicit missing-account error.

Only authentication environment and operational provenance location change.
The previous failed status, checkpoints, imports and all scientific snapshots
remain immutable. No credentials are read or copied by this helper. The original
404 recovery is validated as history, never executed again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import r02_coordinator_resume as history

SCHEMA = "r02_coordinator_auth_recovery_v1"
PREVIOUS = "repairs/recovery-20261002/coordinator"
RELATIVE = "repairs/auth-recovery-20261002/coordinator"
DESTINATION = "00_datos_y_diseno/recovery-20261002/auth-recovery"
COPIES = ("manifest.json", "coordinator.py", "preprocess_count_validation.py", "adoption.json")


def authenticate_failure(failure, old_spec_sha256, coordinator):
    history.require(failure.get("state") == "FAILED" and failure.get("resume_manifest_sha256") == old_spec_sha256,
                    "Failure is not the authenticated previous resumed coordinator")
    for entry in coordinator["remote_chromosomes"]:
        prefix = "Cannot authenticate cloud object: " + entry["completion_uri"] + ": "
        if failure.get("error", "").startswith(prefix):
            detail = failure["error"][len(prefix):]
            history.require(detail.startswith("ERROR: (gcloud.storage.objects.describe) You do not currently have an active account selected."),
                            "Recovery is limited to the explicit missing-account failure")
            return entry["completion_uri"]
    raise ValueError("Failure refers to an unassigned completion object")


def inactive_service(name):
    result = subprocess.run(["systemctl", "--user", "show", name, "--property=LoadState,ActiveState,MainPID"],
                            capture_output=True, text=True, check=True)
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    history.require(state.get("LoadState") == "loaded" and state.get("ActiveState") in ("failed", "inactive")
                    and state.get("MainPID") == "0", "Previous coordinator service is active or unknown")
    return state


def validate_environment(spec):
    path = Path(spec["cloudsdk_config"])
    history.require(path.is_absolute() and path.resolve() == path and path.is_dir()
                    and not path.is_relative_to(Path(spec["run_dir"]).parents[2]), "Authentication config must remain outside the project")
    history.require(os.environ.get("CLOUDSDK_CONFIG") == str(path)
                    and os.environ.get("CLOUDSDK_CORE_ACCOUNT") == spec["account"], "Authentication environment differs from recovery contract")


def protected_paths(run):
    previous = run/PREVIOUS
    paths = [previous/name for name in ("status.json", "resume.manifest.json", "resume.py", "resume_activation.json", "coordinator_activation.json") + COPIES]
    paths += sorted((run/"checkpoints").glob("*.json")) + sorted((run/"repairs/parallel-v1/imports").glob("chr*_import.json"))
    if (run/"status.json").exists():
        paths.append(run/"status.json")
    return paths


def prepare(run, service, config, account):
    run = Path(run).resolve()
    previous = run/PREVIOUS
    old_spec_path = previous/"resume.manifest.json"
    old_spec = history.read(old_spec_path)
    history.checked_file(previous/"resume.py", old_spec["resume_sha256"])
    current = history.read(previous/"manifest.json")
    history.checked_file(previous/"manifest.json", old_spec["coordinator_manifest_sha256"])
    failed_uri = authenticate_failure(history.read(previous/"status.json"), history.sha(old_spec_path), current)
    inactive_service(service)
    imports = run/"repairs/parallel-v1/imports"
    receipts = sorted(imports.glob("chr*_import.json"))
    history.require(receipts, "Need existing authenticated imports for cloud-access probe")
    protected = protected_paths(run)
    directory = run/RELATIVE
    directory.mkdir(parents=True, exist_ok=False)
    for name in COPIES:
        history.copy_bytes(directory/name, (previous/name).read_bytes())
    history.copy_bytes(directory/"resume_auth.py", Path(__file__).read_bytes())
    # The dependency is frozen, so future repository edits cannot alter validation.
    history.copy_bytes(directory/"r02_coordinator_resume.py", (previous/"resume.py").read_bytes())
    snapshot = directory/"previous_evidence"
    snapshot.mkdir()
    for source in protected:
        target = snapshot/source.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        history.copy_bytes(target, source.read_bytes())
    spec = dict(schema=SCHEMA, run_dir=str(run), helper_sha256=history.sha(directory/"resume_auth.py"),
                history_helper_sha256=old_spec["resume_sha256"], previous_resume_sha256=history.sha(old_spec_path),
                coordinator_manifest_sha256=old_spec["coordinator_manifest_sha256"],
                previous_service=service, cloudsdk_config=str(Path(config).resolve()), account=account,
                failed_uri=failed_uri, deadline_utc=current["deadline_utc"],
                cloud_probe=history.read(receipts[0])["completion"],
                protected_sha256={str(path.relative_to(run)): history.sha(path) for path in protected})
    path = directory/"auth.manifest.json"
    history.fixed(path, spec)
    return path


def validate_spec(path, expected, *, cloud_probe=True):
    path = history.checked_file(path, expected)
    spec = history.read(path)
    required = {"schema", "run_dir", "helper_sha256", "history_helper_sha256", "previous_resume_sha256",
                "coordinator_manifest_sha256", "previous_service", "cloudsdk_config", "account",
                "failed_uri", "deadline_utc", "cloud_probe", "protected_sha256"}
    history.require(set(spec) == required and spec["schema"] == SCHEMA and spec["helper_sha256"] == history.sha(__file__), "Wrong auth-recovery source/schema")
    run = Path(spec["run_dir"])
    history.require(run.is_absolute() and run.resolve() == run and path.parent == run/RELATIVE, "Wrong recovery path")
    history.checked_file(Path(history.__file__), spec["history_helper_sha256"])
    validate_environment(spec)
    inactive_service(spec["previous_service"])
    history.require(set(spec["protected_sha256"]) == {str(p.relative_to(run)) for p in protected_paths(run)}, "Incomplete/changed protected checkpoint and import set")
    for relative, digest in spec["protected_sha256"].items():
        history.require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe protected path")
        history.checked_file(run/relative, digest)
        history.checked_file(path.parent/"previous_evidence"/relative, digest)
    previous = run/PREVIOUS
    # This authenticates historical adoption, scientific snapshots, completed
    # chr21/22 outputs, mount and absence of old controllers. It does NOT call run.
    context = history.validate_spec(previous/"resume.manifest.json", spec["previous_resume_sha256"])
    _, current, coordinator, recovery, old, handoff, pipeline, _ = context
    history.require(history.sha(previous/"manifest.json") == spec["coordinator_manifest_sha256"]
                    and current["deadline_utc"] == spec["deadline_utc"], "Original deadline/manifest changed")
    history.require(authenticate_failure(history.read(previous/"status.json"), spec["previous_resume_sha256"], current) == spec["failed_uri"], "Auth failure changed")
    for name in COPIES:
        history.checked_file(path.parent/name, history.sha(previous/name))
    for relative in spec["protected_sha256"]:
        if relative.startswith("repairs/parallel-v1/imports/"):
            pipeline.verify_files(history.read(run/relative)["outputs"])
    if cloud_probe:
        metadata = coordinator.GCS().metadata(spec["cloud_probe"]["uri"])
        coordinator.validate_metadata(metadata, spec["cloud_probe"])
    return spec, current, coordinator, recovery, old, handoff


def run(path, expected, *, execute=False):
    spec, current, coordinator, recovery, old, handoff = validate_spec(path, expected)
    if not execute:
        return dict(state="VERIFIED_AUTH_RECOVERY_NOT_EXECUTED", deadline_utc=spec["deadline_utc"],
                    account=spec["account"], scientific_parameters_changed=False)
    directory = Path(path).parent
    class Status:
        def state(self, state, **extra):
            value = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                         auth_manifest_sha256=expected, **extra)
            handoff.write_json(directory/"status.json", value)
            print(json.dumps(value), flush=True)
    status = Status()
    def expired(_signal, _frame):
        raise TimeoutError("Original coordinator deadline reached")
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, max(.01, coordinator.timestamp(spec["deadline_utc"]) - time.time()))
    try:
        with recovery.boundary_lock(Path(spec["run_dir"])/"repairs/boundary-v2/.boundary.lock"):
            validate_spec(path, expected)
            recovery.validate_mount(old)
            provenance = directory/"provenance"
            provenance.mkdir(exist_ok=True)
            for name in ("resume_auth.py", "r02_coordinator_resume.py", "auth.manifest.json", "coordinator.py"):
                history.copy_bytes(provenance/name, (directory/name).read_bytes())
            history.fixed(directory/"activation.json", dict(schema=SCHEMA, auth_manifest_sha256=expected,
                previous_failure_sha256=spec["protected_sha256"][PREVIOUS + "/status.json"],
                original_deadline_utc=spec["deadline_utc"], authentication_config_copied=False,
                scientific_parameters_changed=False, checkpoints_rewritten=False,
                original_controller_executed_again=False, local_chromosomes_reused=[21, 22]))
            history.copy_bytes(provenance/"activation.json", (directory/"activation.json").read_bytes())
            status.state("AUTH_RECOVERY_VERIFIED_WAITING_REMOTE")
            coordinator.run_after_boundary(current, directory, status, activate=False,
                                           provenance_destination=DESTINATION)
    except BaseException as exc:
        status.state("FAILED", error=str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
    return dict(state="COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "run"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--previous-service")
    parser.add_argument("--cloudsdk-config", type=Path)
    parser.add_argument("--account")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--expected-spec-sha256")
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "prepare":
        history.require(all((args.run_dir, args.previous_service, args.cloudsdk_config, args.account)), "Missing explicit preparation inputs")
        path = prepare(args.run_dir, args.previous_service, args.cloudsdk_config, args.account)
        result = dict(state="PREPARED_NOT_EXECUTED", spec=str(path), sha256=history.sha(path))
    else:
        history.require(args.spec and args.expected_spec_sha256, "Hash-bound recovery spec required")
        result = run(args.spec, args.expected_spec_sha256, execute=args.mode == "run")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
