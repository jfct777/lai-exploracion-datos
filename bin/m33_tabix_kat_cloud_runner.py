#!/usr/bin/env python3
"""Submit and externally close the synthetic M33 Tabix controller job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import time
import hashlib
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT = "uspbr-242713"
REGION = "us-central1"
CONTROLLER_SERVICE_ACCOUNT = "dnabr-m33-controller-frank@uspbr-242713.iam.gserviceaccount.com"
TERMINAL_STATES = {"SUCCEEDED", "FAILED"}
LAUNCHER_SERVICE_ACCOUNT = "653458115080-compute@developer.gserviceaccount.com"
EXTERNAL_CLOSER_FILES = {
    "bin/m33_batch_postflight.py",
    "bin/m33_gcs_append_only.py",
    "bin/m33_storage_policy.py",
    "bin/m33_tabix_kat_cloud_runner.py",
    "conf/m33_storage_namespace_policy.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("m33_batch_postflight_runner", path)
    require(specification is not None and specification.loader is not None, "postflight module cannot load")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def gcloud_json(arguments: list[str]) -> Any:
    completed = subprocess.run(
        ["gcloud", *arguments, f"--project={PROJECT}", "--format=json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout or "null")


def list_jobs() -> list[dict[str, Any]]:
    payload = gcloud_json(["batch", "jobs", "list", f"--location={REGION}"])
    require(isinstance(payload, list), "gcloud Batch listing is not a list")
    return payload


def describe_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detailed = []
    for job in jobs:
        name = str(job.get("name", ""))
        job_id = name.rsplit("/", 1)[-1]
        require(job_id and job_id != name, "Batch job name is not fully qualified")
        detailed.append(gcloud_json(["batch", "jobs", "describe", job_id, f"--location={REGION}"]))
    return detailed


def team_job_names(jobs: list[dict[str, Any]]) -> set[str]:
    return {
        str(job.get("name")) for job in jobs
        if (job.get("labels") or {}).get("team") == "frank" and job.get("name")
    }


def validate_team_delta(
    before_names: set[str], after_jobs: list[dict[str, Any]], expected_run_names: set[str]
) -> set[str]:
    created = team_job_names(after_jobs) - before_names
    require(created == expected_run_names, "unexpected new team=frank job or missing run job")
    return created


def controller_job_spec(
    *, run_id: str, runtime_image: str, controller_image: str, labels: dict[str, str]
) -> dict[str, Any]:
    commands = [
        "--run-id", run_id,
        "--runtime-image", runtime_image,
        "--controller-image", controller_image,
        "--receipt", "/tmp/m33-controller.receipt.json",
    ]
    return {
        "taskGroups": [{
            "taskCount": 1,
            "parallelism": 1,
            "taskSpec": {
                "runnables": [{"container": {"imageUri": controller_image, "commands": commands}}],
                "computeResource": {"cpuMilli": 2000, "memoryMib": 4096},
                "maxRetryCount": 0,
                "maxRunDuration": "1800s",
            },
        }],
        "allocationPolicy": {
            "instances": [{"policy": {"machineType": "e2-standard-2", "provisioningModel": "STANDARD"}}],
            "serviceAccount": {"email": CONTROLLER_SERVICE_ACCOUNT},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": labels,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_launch_authorization(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(set(payload) == {
        "schema_version", "stage", "status", "controller_image", "runtime_image",
        "external_closer_files", "source_auth_sha256", "impersonation_boundary",
    }, "launch authorization keys differ")
    require(payload["schema_version"] == "1.0.0", "launch authorization version differs")
    require(payload["stage"] == "M33_TABIX_KAT_CLOUD_LAUNCH_AUTHORIZATION", "launch stage differs")
    require(payload["status"] == "AUTHORIZED_EXACT_IMMUTABLE_IMAGES", "cloud launch remains blocked")
    require(re.fullmatch(
        r"us-central1-docker\.pkg\.dev/uspbr-242713/dnabr-lai/m33-controller@sha256:[0-9a-f]{64}",
        str(payload["controller_image"]),
    ) is not None, "controller image repository/digest differs")
    require(re.fullmatch(
        r"us-central1-docker\.pkg\.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:[0-9a-f]{64}",
        str(payload["runtime_image"]),
    ) is not None, "runtime image repository/digest differs")
    files = payload["external_closer_files"]
    require(isinstance(files, dict) and set(files) == EXTERNAL_CLOSER_FILES, "external closer inventory differs")
    for relative, expected_hash in files.items():
        require(re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)) is not None, "external closer hash is invalid")
        require(sha256_file(repo_root / relative) == expected_hash, f"external closer source drifted: {relative}")
    require(
        re.fullmatch(r"[0-9a-f]{64}", str(payload["source_auth_sha256"])) is not None
        and sha256_file(repo_root / "conf" / "m33_tabix_kat_source_auth.json") == payload["source_auth_sha256"],
        "source authorization hash differs",
    )
    require(payload["impersonation_boundary"] == {
        "launcher_service_account": LAUNCHER_SERVICE_ACCOUNT,
        "target_service_account": CONTROLLER_SERVICE_ACCOUNT,
        "required_role": "roles/iam.serviceAccountTokenCreator",
        "scope": "controller_service_account_only_all_controller_permissions",
    }, "controller impersonation trust boundary differs")
    return payload


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def wait_controller(job_id: str, timeout_seconds: int = 2400) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = gcloud_json(["batch", "jobs", "describe", job_id, f"--location={REGION}"])
        state = str(job.get("status", {}).get("state", ""))
        if state in TERMINAL_STATES:
            return job
        time.sleep(10)
    raise TimeoutError("controller job did not reach a terminal state before the external timeout")


def drain_run_jobs(
    postflight: Any,
    *,
    run_id: str,
    controller_job_name: str,
    timeout_seconds: int = 2400,
) -> list[dict[str, Any]]:
    """Wait fail-closed until every observed run job is terminal and stable."""
    deadline = time.monotonic() + timeout_seconds
    previous = None
    stable_terminal_reads = 0
    while time.monotonic() < deadline:
        selected = postflight.select_run_jobs(list_jobs(), run_id)
        signature = postflight.inventory_signature(selected)
        states = {str(job.get("status", {}).get("state", "")) for job in selected}
        names = {str(job.get("name", "")) for job in selected}
        all_terminal = bool(selected) and states <= TERMINAL_STATES and controller_job_name in names
        if all_terminal and signature == previous:
            stable_terminal_reads += 1
            if stable_terminal_reads >= 2:
                return selected
        else:
            stable_terminal_reads = 0
        previous = signature
        time.sleep(5)
    raise TimeoutError("run jobs did not become terminal and stable after controller failure")


def active_account() -> str:
    completed = subprocess.run(
        ["gcloud", "auth", "print-access-token"], check=True, capture_output=True, text=True
    )
    token = completed.stdout.strip()
    require(token, "gcloud launcher access token is empty")
    query = urllib.parse.urlencode({"access_token": token})
    with urllib.request.urlopen(f"https://oauth2.googleapis.com/tokeninfo?{query}", timeout=15) as response:
        payload = json.loads(response.read())
    account = str(payload.get("email", ""))
    require(account, "launcher token has no authenticated email")
    return account


def impersonated_controller_token() -> str:
    completed = subprocess.run([
        "gcloud", "auth", "print-access-token",
        f"--impersonate-service-account={CONTROLLER_SERVICE_ACCOUNT}",
    ], check=True, capture_output=True, text=True)
    token = completed.stdout.strip()
    require(token, "controller impersonation returned an empty token")
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launch-authorization", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), "external runner receipt already exists")

    repo_root = Path(__file__).resolve().parents[1]
    authorization = load_launch_authorization(args.launch_authorization, repo_root)
    postflight = load_module(repo_root / "bin" / "m33_batch_postflight.py")
    publisher = load_module(repo_root / "bin" / "m33_gcs_append_only.py")
    controller_module = load_module(repo_root / "bin" / "m33_tabix_kat_controller.py")
    _, observed_source_auth = controller_module.validate_source_auth(repo_root)
    require(
        observed_source_auth == authorization["source_auth_sha256"],
        "external closer source bundle is not exactly authorized",
    )
    runtime_image = authorization["runtime_image"]
    controller_image = authorization["controller_image"]
    labels = postflight.required_labels(args.run_id)
    before = list_jobs()
    launcher_identity = active_account()
    require(launcher_identity == LAUNCHER_SERVICE_ACCOUNT, "launcher identity differs from authorization")
    existing = postflight.select_run_jobs(before, args.run_id)
    require(not existing, "run label already exists in Batch; choose a new run ID")
    before_team_names = team_job_names(before)
    spec = controller_job_spec(
        run_id=args.run_id,
        runtime_image=runtime_image,
        controller_image=controller_image,
        labels=labels,
    )
    job_id = f"m33-kat-{postflight.run_label(args.run_id)}"
    with tempfile.TemporaryDirectory(prefix="m33-kat-submit-") as temporary:
        config = Path(temporary) / "controller-job.json"
        config.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run([
            "gcloud", "batch", "jobs", "submit", job_id,
            f"--location={REGION}", f"--config={config}", f"--project={PROJECT}",
        ], check=True)
    controller_job_name = f"projects/{PROJECT}/locations/{REGION}/jobs/{job_id}"
    ready_created = False
    try:
        controller = wait_controller(job_id)
        require(controller.get("status", {}).get("state") == "SUCCEEDED", "controller job failed; no external PASS")
        stable = postflight.stable_inventory(list_jobs, run_id=args.run_id, attempts=8, interval_seconds=3)
        detailed = describe_jobs(stable)
        inventory = postflight.validate_inventory(
            detailed,
            run_id=args.run_id,
            runtime_image=runtime_image,
            controller_image=controller_image,
            controller_may_be_running=False,
        )
        require(inventory["controller_state"] == "SUCCEEDED", "controller is still active after completion")
        after = list_jobs()
        job_names = {job["name"] for job in detailed}
        validate_team_delta(before_team_names, after, job_names)
        external_close = {
            "schema_version": "1.0.0",
            "stage": "M33_TABIX_SYNTHETIC_KAT_EXTERNAL_CLOSE",
            "status": "PASS_ALL_RUN_JOBS_SUCCEEDED_ZERO_ACTIVE",
            "run_id": args.run_id,
            "run_label": postflight.run_label(args.run_id),
            "controller_job": controller.get("name"),
            "launcher_identity": launcher_identity,
            "finalizer_identity": CONTROLLER_SERVICE_ACCOUNT,
            "job_names": sorted(job_names),
            "inventory": inventory,
            "preexisting_team_job_count": len(before_team_names),
            "preexisting_team_jobs_sha256": hashlib.sha256(
                "\n".join(sorted(before_team_names)).encode("utf-8")
            ).hexdigest(),
            "persistent_write_after_ready": False,
            "contains_real_genomic_data": False,
        }
        final = publisher.finalize_ready(
            run_id=args.run_id,
            storage_policy=repo_root / "conf" / "m33_storage_namespace_policy.json",
            storage_validator=repo_root / "bin" / "m33_storage_policy.py",
            external_close=external_close,
            token=impersonated_controller_token(),
        )
        ready_created = True
        result = {
            **external_close,
            "status": "PASS_ALL_RUN_JOBS_SUCCEEDED_ZERO_ACTIVE_READY_FINALIZED",
            "ready_generation": final["ready"]["generation"],
            "external_close_generation": final["external_close"]["generation"],
        }
        write_json_exclusive(args.receipt, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        failure: dict[str, Any] = {
            "schema_version": "1.0.0",
            "stage": "M33_TABIX_SYNTHETIC_KAT_EXTERNAL_CLOSE",
            "status": "STOP_FAIL_CLOSED_NO_READY",
            "run_id": args.run_id,
            "run_label": postflight.run_label(args.run_id),
            "controller_job": controller_job_name,
            "error_type": type(error).__name__,
            "error": str(error),
            "cleanup_policy": "NO_DELETE_WAIT_UNTIL_TERMINAL",
            "ready_created": ready_created,
        }
        try:
            terminal = drain_run_jobs(
                postflight, run_id=args.run_id, controller_job_name=controller_job_name
            )
            failure["terminal_jobs"] = [
                {"name": job.get("name"), "state": job.get("status", {}).get("state")}
                for job in terminal
            ]
            failure["active_jobs_after_wait"] = []
        except BaseException as drain_error:
            failure["drain_error_type"] = type(drain_error).__name__
            failure["drain_error"] = str(drain_error)
        if not args.receipt.exists():
            write_json_exclusive(args.receipt, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
