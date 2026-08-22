#!/usr/bin/env python3
"""Publish and finalize the synthetic M33 KAT with immutable GCS generations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


RUNTIME_SERVICE_ACCOUNT = "dnabr-m33-frank@uspbr-242713.iam.gserviceaccount.com"
CONTROLLER_SERVICE_ACCOUNT = "dnabr-m33-controller-frank@uspbr-242713.iam.gserviceaccount.com"
METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
TOKEN_URL = f"{METADATA_ROOT}/instance/service-accounts/default/token"
EMAIL_URL = f"{METADATA_ROOT}/instance/service-accounts/default/email"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def metadata(path: str) -> tuple[bytes, Any]:
    request = urllib.request.Request(path, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=10) as response:
        require(response.headers.get("Metadata-Flavor") == "Google", "metadata response is unauthenticated")
        return response.read(), response.headers


def authenticate_service_account(expected: str) -> str:
    raw, _ = metadata(EMAIL_URL)
    email = raw.decode("ascii").strip()
    role = "runtime" if expected == RUNTIME_SERVICE_ACCOUNT else "controller"
    require(email == expected, f"publisher is not using the authorized {role} service account")
    return email


def authenticate_runtime() -> str:
    return authenticate_service_account(RUNTIME_SERVICE_ACCOUNT)


def access_token() -> str:
    raw, _ = metadata(TOKEN_URL)
    payload = json.loads(raw)
    require(payload.get("token_type") == "Bearer", "metadata token type drifted")
    token = payload.get("access_token")
    require(isinstance(token, str) and token, "metadata access token is missing")
    return token


class AppendOnlyGCS:
    def __init__(self, token: str):
        self.token = token

    def _request(self, url: str, *, data: bytes | None = None, content_type: str | None = None) -> tuple[bytes, Any]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(url, headers=headers, data=data)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(), response.headers

    def create_and_verify(self, bucket: str, object_name: str, content: bytes, content_type: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({
            "uploadType": "media",
            "name": object_name,
            "ifGenerationMatch": "0",
        })
        raw, _ = self._request(
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?{query}",
            data=content,
            content_type=content_type,
        )
        created = json.loads(raw)
        generation = str(created.get("generation", ""))
        require(generation.isdigit() and int(generation) > 0, "GCS did not return an immutable generation")
        quoted = urllib.parse.quote(object_name, safe="")
        metadata_query = urllib.parse.urlencode({"generation": generation})
        metadata_raw, _ = self._request(
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quoted}?{metadata_query}"
        )
        observed = json.loads(metadata_raw)
        require(observed.get("name") == object_name, "reopened GCS object name differs")
        require(str(observed.get("generation")) == generation, "reopened GCS generation differs")
        require(int(observed.get("size", -1)) == len(content), "reopened GCS object size differs")
        media_query = urllib.parse.urlencode({"alt": "media", "generation": generation})
        reopened, _ = self._request(
            f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{quoted}?{media_query}"
        )
        require(reopened == content, "reopened GCS bytes differ")
        return {
            "gcs_uri": f"gs://{bucket}/{object_name}",
            "generation": generation,
            "size_bytes": len(content),
            "sha256": sha256_bytes(content),
            "md5_base64": base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode("ascii"),
        }

    def record_for_existing(self, bucket: str, object_name: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(object_name, safe="")
        raw, _ = self._request(f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quoted}")
        metadata_payload = json.loads(raw)
        generation = str(metadata_payload.get("generation", ""))
        require(generation.isdigit() and int(generation) > 0, "GCS object has no immutable generation")
        media_query = urllib.parse.urlencode({"alt": "media", "generation": generation})
        content, _ = self._request(
            f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{quoted}?{media_query}"
        )
        require(int(metadata_payload.get("size", -1)) == len(content), "existing GCS object size differs")
        return {
            "gcs_uri": f"gs://{bucket}/{object_name}",
            "generation": generation,
            "size_bytes": len(content),
            "sha256": sha256_bytes(content),
            "md5_base64": base64.b64encode(
                hashlib.md5(content, usedforsecurity=False).digest()
            ).decode("ascii"),
            "content": content,
        }

    def reopen_record(self, record: dict[str, Any]) -> bytes:
        uri = urllib.parse.urlsplit(record["gcs_uri"])
        require(uri.scheme == "gs" and uri.netloc and uri.path.startswith("/"), "invalid GCS record URI")
        object_name = uri.path[1:]
        quoted = urllib.parse.quote(object_name, safe="")
        query = urllib.parse.urlencode({"alt": "media", "generation": str(record["generation"])})
        content, _ = self._request(
            f"https://storage.googleapis.com/download/storage/v1/b/{uri.netloc}/o/{quoted}?{query}"
        )
        require(len(content) == int(record["size_bytes"]), "reopened record size differs")
        require(sha256_bytes(content) == record["sha256"], "reopened record hash differs")
        return content


def encode_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encode_json(payload))
        handle.flush()
        os.fsync(handle.fileno())


def publish(args: argparse.Namespace) -> dict[str, Any]:
    storage = load_module("m33_storage_policy", args.storage_validator)
    policy = storage.load_policy(args.storage_policy)
    storage.validate_run_id(args.run_id)
    runtime_email = authenticate_runtime()
    client = AppendOnlyGCS(access_token())
    bucket = "teams-usp"
    root = "frank/lai-exploracion-datos"
    destinations = [
        ("runs", f"{root}/runs/{args.run_id}/m33_tabix_kat.receipt.json", args.kat_receipt, "application/json"),
        ("logs", f"{root}/logs/{args.run_id}/controller.receipt.json", args.controller_receipt, "application/json"),
    ]
    published: list[dict[str, Any]] = []
    ordered_uris: list[str] = []
    for namespace, object_name, source, content_type in destinations:
        uri = f"gs://{bucket}/{object_name}"
        storage.validate_write_uri(uri, namespace, policy, run_id=args.run_id)
        ordered_uris.append(uri)
        published.append(client.create_and_verify(bucket, object_name, source.read_bytes(), content_type))

    manifest = {
        "schema_version": "1.0.0",
        "stage": "M33_TABIX_SYNTHETIC_KAT_PUBLICATION",
        "status": "PASS_APPEND_ONLY_REOPENED_AND_HASHED",
        "run_id": args.run_id,
        "runtime_service_account": runtime_email,
        "contains_real_genomic_data": False,
        "published_before_manifest": published,
    }
    manifest_object = f"{root}/manifests/{args.run_id}/publication.manifest.json"
    manifest_uri = f"gs://{bucket}/{manifest_object}"
    storage.validate_write_uri(manifest_uri, "manifests", policy, run_id=args.run_id)
    ordered_uris.append(manifest_uri)
    manifest_record = client.create_and_verify(bucket, manifest_object, encode_json(manifest), "application/json")

    storage.validate_publication_order(ordered_uris)
    result = {
        "stage": "M33_TABIX_SYNTHETIC_KAT_PUBLISHER",
        "status": "PASS_PUBLICATION_CANDIDATE_NO_READY",
        "run_id": args.run_id,
        "runtime_service_account": runtime_email,
        "objects": [*published, manifest_record],
        "manifest": manifest_record,
        "ready_created": False,
        "contains_real_genomic_data": False,
    }
    write_json_exclusive(args.output, result)
    return result


def finalize_candidate(
    *,
    run_id: str,
    storage_policy: Path,
    storage_validator: Path,
    postflight: dict[str, Any],
) -> dict[str, Any]:
    """Publish the internal postflight candidate from the authenticated controller."""
    storage = load_module("m33_storage_policy_finalizer", storage_validator)
    policy = storage.load_policy(storage_policy)
    storage.validate_run_id(run_id)
    controller_email = authenticate_service_account(CONTROLLER_SERVICE_ACCOUNT)
    client = AppendOnlyGCS(access_token())
    bucket = "teams-usp"
    root = "frank/lai-exploracion-datos"
    manifest_object = f"{root}/manifests/{run_id}/publication.manifest.json"
    manifest_record = client.record_for_existing(bucket, manifest_object)
    manifest_content = client.reopen_record(manifest_record)
    manifest = json.loads(manifest_content)
    require(manifest.get("run_id") == run_id, "publication manifest run ID differs")
    require(
        manifest.get("status") == "PASS_APPEND_ONLY_REOPENED_AND_HASHED",
        "publication manifest is not a passing candidate",
    )
    manifest_record.pop("content", None)

    require(postflight.get("run_id") == run_id, "postflight run ID differs")
    require(postflight.get("status") == "PASS_CHILD_JOBS_TERMINAL_AND_AUTHENTICATED", "postflight did not pass")
    postflight_object = f"{root}/logs/{run_id}/postflight.json"
    postflight_uri = f"gs://{bucket}/{postflight_object}"
    storage.validate_write_uri(postflight_uri, "logs", policy, run_id=run_id)
    postflight_record = client.create_and_verify(
        bucket, postflight_object, encode_json(postflight), "application/json"
    )

    # Reauthenticate before the non-consumable candidate marker.
    require(
        authenticate_service_account(CONTROLLER_SERVICE_ACCOUNT) == controller_email,
        "controller identity changed before candidate marker",
    )
    candidate_payload = {
        "stage": "M33_TABIX_SYNTHETIC_KAT_READY_CANDIDATE",
        "status": "NON_CONSUMABLE_CANDIDATE",
        "run_id": run_id,
        "run_label": postflight["run_label"],
        "controller_service_account": controller_email,
        "runtime_service_account": RUNTIME_SERVICE_ACCOUNT,
        "manifest": manifest_record,
        "postflight": postflight_record,
        "known_answers": postflight["known_answers"],
        "controller_still_running": True,
        "external_close_required_before_ready": True,
    }
    candidate_object = f"{root}/runs/{run_id}/READY_CANDIDATE"
    candidate_uri = f"gs://{bucket}/{candidate_object}"
    storage.validate_write_uri(candidate_uri, "runs", policy, run_id=run_id)
    storage.validate_publication_order([manifest_record["gcs_uri"], postflight_uri, candidate_uri])
    candidate_record = client.create_and_verify(
        bucket, candidate_object, encode_json(candidate_payload), "application/json"
    )
    return {
        "stage": "M33_TABIX_SYNTHETIC_KAT_INTERNAL_CLOSE",
        "status": "PASS_CANDIDATE_CREATED_NO_READY",
        "run_id": run_id,
        "controller_service_account": controller_email,
        "manifest": manifest_record,
        "postflight": postflight_record,
        "candidate": candidate_record,
        "ready_created": False,
    }


def token_email(token: str) -> str:
    require(isinstance(token, str) and token, "access token is empty")
    query = urllib.parse.urlencode({"access_token": token})
    with urllib.request.urlopen(f"https://oauth2.googleapis.com/tokeninfo?{query}", timeout=15) as response:
        payload = json.loads(response.read())
    email = str(payload.get("email", ""))
    require(email == CONTROLLER_SERVICE_ACCOUNT, "finalizer token is not the controller service account")
    return email


def finalize_ready(
    *,
    run_id: str,
    storage_policy: Path,
    storage_validator: Path,
    external_close: dict[str, Any],
    token: str,
) -> dict[str, Any]:
    """Publish the external close and terminal READY after every Batch job ended."""
    storage = load_module("m33_storage_policy_external_finalizer", storage_validator)
    policy = storage.load_policy(storage_policy)
    storage.validate_run_id(run_id)
    controller_email = token_email(token)
    client = AppendOnlyGCS(token)
    bucket = "teams-usp"
    root = "frank/lai-exploracion-datos"

    def existing_record(namespace: str, object_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        uri = f"gs://{bucket}/{object_name}"
        storage.validate_write_uri(uri, namespace, policy, run_id=run_id)
        record = client.record_for_existing(bucket, object_name)
        payload = json.loads(client.reopen_record(record))
        record.pop("content", None)
        return record, payload

    manifest_record, manifest = existing_record(
        "manifests", f"{root}/manifests/{run_id}/publication.manifest.json"
    )
    postflight_record, postflight = existing_record(
        "logs", f"{root}/logs/{run_id}/postflight.json"
    )
    candidate_record, candidate = existing_record(
        "runs", f"{root}/runs/{run_id}/READY_CANDIDATE"
    )
    require(manifest.get("run_id") == run_id, "manifest run ID differs")
    require(postflight.get("run_id") == run_id, "internal postflight run ID differs")
    require(candidate.get("status") == "NON_CONSUMABLE_CANDIDATE", "candidate marker differs")
    require(external_close.get("run_id") == run_id, "external close run ID differs")
    require(external_close.get("status") == "PASS_ALL_RUN_JOBS_SUCCEEDED_ZERO_ACTIVE", "external close did not pass")
    internal_jobs = {
        postflight["inventory"]["controller_job"],
        *postflight["inventory"]["child_jobs"],
    }
    require(internal_jobs == set(external_close["job_names"]), "external and internal job inventories differ")

    close_object = f"{root}/logs/{run_id}/external_close.json"
    close_uri = f"gs://{bucket}/{close_object}"
    storage.validate_write_uri(close_uri, "logs", policy, run_id=run_id)
    close_record = client.create_and_verify(
        bucket, close_object, encode_json(external_close), "application/json"
    )
    require(token_email(token) == controller_email, "finalizer identity changed before READY")
    ready_payload = {
        "stage": "M33_TABIX_SYNTHETIC_KAT_READY",
        "status": "READY",
        "run_id": run_id,
        "run_label": postflight["run_label"],
        "controller_service_account": controller_email,
        "runtime_service_account": RUNTIME_SERVICE_ACCOUNT,
        "manifest": manifest_record,
        "internal_postflight": postflight_record,
        "candidate": candidate_record,
        "external_close": close_record,
        "known_answers": postflight["known_answers"],
        "all_batch_jobs_terminal_before_ready": True,
        "persistent_write_after_ready": False,
    }
    ready_object = f"{root}/runs/{run_id}/READY"
    ready_uri = f"gs://{bucket}/{ready_object}"
    storage.validate_write_uri(ready_uri, "runs", policy, run_id=run_id)
    storage.validate_publication_order([
        manifest_record["gcs_uri"], postflight_record["gcs_uri"], candidate_record["gcs_uri"],
        close_record["gcs_uri"], ready_uri,
    ])
    ready_record = client.create_and_verify(
        bucket, ready_object, encode_json(ready_payload), "application/json"
    )
    return {
        "stage": "M33_TABIX_SYNTHETIC_KAT_EXTERNAL_FINALIZER",
        "status": "PASS_READY_CREATED_AFTER_ZERO_ACTIVE",
        "run_id": run_id,
        "external_close": close_record,
        "ready": ready_record,
        "persistent_write_after_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kat-receipt", required=True, type=Path)
    parser.add_argument("--controller-receipt", required=True, type=Path)
    parser.add_argument("--storage-policy", required=True, type=Path)
    parser.add_argument("--storage-validator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = publish(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
