#!/usr/bin/env python3
"""Authenticate the exact Google Batch inventory for the synthetic M33 KAT."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterable


PROJECT = "uspbr-242713"
REGION = "us-central1"
CONTROLLER_SERVICE_ACCOUNT = "dnabr-m33-controller-frank@uspbr-242713.iam.gserviceaccount.com"
RUNTIME_SERVICE_ACCOUNT = "dnabr-m33-frank@uspbr-242713.iam.gserviceaccount.com"
PIPELINE_LABEL = "m33-tabix-kat"
EXPECTED_CHILD_JOBS = 4
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "DELETION_IN_PROGRESS"}
EXPECTED_KNOWN_ANSWERS = {
    "source_vcf_sha256": "52868ddaad1bb9641ecfe499d61817736af36169d9d51e717b1d9112bf06a108",
    "independent_tbi_sha256": "ecab3b3f84174efb992be57a46e237c764791d0f81387a16a063baca71b7cc3b",
    "record_sha256": "a3bbc3a262733a3017c1ffd7faf8adaeea063ad81f8a935a4d790f4478b6f3cf",
    "record_count": 4,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run_label(run_id: str) -> str:
    require(isinstance(run_id, str) and run_id, "run ID is empty")
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]


def required_labels(run_id: str) -> dict[str, str]:
    return {"team": "frank", "pipeline": PIPELINE_LABEL, "run": run_label(run_id)}


def _service_account(job: dict[str, Any]) -> str:
    return str(job.get("allocationPolicy", {}).get("serviceAccount", {}).get("email", ""))


def _state(job: dict[str, Any]) -> str:
    return str(job.get("status", {}).get("state", ""))


def _images(job: dict[str, Any]) -> set[str]:
    images: set[str] = set()
    for group in job.get("taskGroups", []):
        for runnable in group.get("taskSpec", {}).get("runnables", []):
            image = runnable.get("container", {}).get("imageUri")
            if image:
                images.add(str(image))
    return images


def _matches_run(job: dict[str, Any], expected: dict[str, str]) -> bool:
    labels = job.get("labels") or {}
    # Select only by the run identity. Team and pipeline are then validated, so
    # corrupting either label cannot make an unexpected same-run job invisible.
    return labels.get("run") == expected["run"]


def inventory_signature(jobs: Iterable[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(
        (str(job.get("name", "")), _state(job), _service_account(job)) for job in jobs
    ))


def select_run_jobs(jobs: Iterable[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    expected = required_labels(run_id)
    return [job for job in jobs if _matches_run(job, expected)]


def validate_inventory(
    jobs: list[dict[str, Any]],
    *,
    run_id: str,
    runtime_image: str,
    controller_image: str | None = None,
    controller_may_be_running: bool,
) -> dict[str, Any]:
    expected = required_labels(run_id)
    selected = select_run_jobs(jobs, run_id)
    require(len(selected) == EXPECTED_CHILD_JOBS + 1, "run inventory is missing or contains an extra job")
    require(all(job.get("name") for job in selected), "a Batch job has no name")
    require(len({job["name"] for job in selected}) == len(selected), "duplicate Batch job name")
    for job in selected:
        labels = job.get("labels") or {}
        require(all(labels.get(key) == value for key, value in expected.items()), "Batch labels differ")

    controllers = [job for job in selected if _service_account(job) == CONTROLLER_SERVICE_ACCOUNT]
    children = [job for job in selected if _service_account(job) == RUNTIME_SERVICE_ACCOUNT]
    require(len(controllers) == 1, "exactly one controller job is required")
    require(len(children) == EXPECTED_CHILD_JOBS, "exactly four runtime child jobs are required")
    require(len(controllers) + len(children) == len(selected), "unexpected service account in run inventory")

    controller = controllers[0]
    controller_state = _state(controller)
    allowed_controller = {"RUNNING", "SUCCEEDED"} if controller_may_be_running else {"SUCCEEDED"}
    require(controller_state in allowed_controller, "controller job is not in the required state")
    if controller_image is not None:
        require(_images(controller) == {controller_image}, "controller image differs from immutable digest")
    for child in children:
        require(_state(child) == "SUCCEEDED", "a runtime child job did not succeed")
        require(_images(child) == {runtime_image}, "runtime image differs from immutable digest")

    return {
        "controller_job": controller["name"],
        "controller_state": controller_state,
        "child_jobs": sorted(child["name"] for child in children),
        "child_states": {child["name"]: _state(child) for child in children},
        "active_child_jobs": [],
        "labels": expected,
    }


def validate_kat_receipt(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    require(payload.get("stage") == "M33_TABIX_SYNTHETIC_KAT", "KAT receipt stage differs")
    require(payload.get("status") == "PASS_INDEPENDENT_INDEX_AND_QUERY_PARITY", "KAT receipt did not pass")
    require(payload.get("contains_real_genomic_data") is False, "real genomic data entered the KAT")
    require(payload.get("cloud_context_authenticated") is True, "KAT was not cloud-authenticated")
    require(payload.get("runtime_service_account") == RUNTIME_SERVICE_ACCOUNT, "KAT runtime identity differs")
    builds = payload.get("build_receipts")
    require(isinstance(builds, list) and len(builds) == 2, "KAT requires two build receipts")
    require({row.get("replica") for row in builds} == {"A", "B"}, "KAT replica inventory differs")
    require(len({row.get("task_hash") for row in builds}) == 2, "KAT builds reused a task hash")
    require(len({row.get("task_work_dir") for row in builds}) == 2, "KAT builds reused a work directory")
    work_prefix = f"gs://teams-usp/frank/lai-exploracion-datos/work/nextflow/{run_id}/"
    require(all(str(row.get("task_work_dir", "")).startswith(work_prefix) for row in builds), "KAT workdir escaped run prefix")
    observed = {
        "source_vcf_sha256": payload.get("source_vcf_sha256"),
        "independent_tbi_sha256": payload.get("independent_tbi_sha256"),
        "record_sha256": payload.get("record_sha256"),
        "record_count": payload.get("record_count"),
    }
    require(observed == EXPECTED_KNOWN_ANSWERS, "KAT known answers differ")
    return observed


class BatchAPI:
    def __init__(self, token: str):
        self.token = token

    def _get(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def list_jobs(self) -> list[dict[str, Any]]:
        base = f"https://batch.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/jobs"
        jobs: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            query = urllib.parse.urlencode({"pageSize": 100, **({"pageToken": page_token} if page_token else {})})
            payload = self._get(f"{base}?{query}")
            jobs.extend(payload.get("jobs", []))
            page_token = payload.get("nextPageToken")
            if not page_token:
                return jobs

    def get_job(self, name: str) -> dict[str, Any]:
        required_prefix = f"projects/{PROJECT}/locations/{REGION}/jobs/"
        require(name.startswith(required_prefix) and name != required_prefix, "Batch job name escaped project/region")
        return self._get(f"https://batch.googleapis.com/v1/{name}")


def stable_inventory(
    fetch: Callable[[], list[dict[str, Any]]],
    *,
    run_id: str,
    attempts: int = 8,
    interval_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    previous: tuple[tuple[str, str, str], ...] | None = None
    for _ in range(attempts):
        selected = select_run_jobs(fetch(), run_id)
        signature = inventory_signature(selected)
        if signature and signature == previous:
            return selected
        previous = signature
        time.sleep(interval_seconds)
    raise ValueError("Batch run inventory did not stabilize across two reads")


def make_postflight(
    *,
    jobs: list[dict[str, Any]],
    run_id: str,
    runtime_image: str,
    controller_image: str | None,
    kat_receipt: dict[str, Any],
) -> dict[str, Any]:
    inventory = validate_inventory(
        jobs,
        run_id=run_id,
        runtime_image=runtime_image,
        controller_image=controller_image,
        controller_may_be_running=True,
    )
    known_answers = validate_kat_receipt(kat_receipt, run_id)
    require(inventory["controller_state"] == "RUNNING", "controller must still be running during in-controller postflight")
    return {
        "schema_version": "1.0.0",
        "stage": "M33_TABIX_SYNTHETIC_KAT_POSTFLIGHT",
        "status": "PASS_CHILD_JOBS_TERMINAL_AND_AUTHENTICATED",
        "run_id": run_id,
        "run_label": run_label(run_id),
        "controller_service_account": CONTROLLER_SERVICE_ACCOUNT,
        "runtime_service_account": RUNTIME_SERVICE_ACCOUNT,
        "inventory": inventory,
        "known_answers": known_answers,
        "contains_real_genomic_data": False,
        "lab_asset_read_authorized": False,
        "controller_still_running": True,
        "external_zero_active_check_required": True,
    }
