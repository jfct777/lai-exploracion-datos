#!/usr/bin/env python3
"""Publish the synthetic M33 KAT atomically and verify every stored generation."""

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


def authenticate_runtime() -> str:
    raw, _ = metadata(EMAIL_URL)
    email = raw.decode("ascii").strip()
    require(email == RUNTIME_SERVICE_ACCOUNT, "publisher is not using the authorized runtime service account")
    return email


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

    ready_payload = {
        "stage": "M33_TABIX_SYNTHETIC_KAT_READY",
        "status": "READY",
        "run_id": args.run_id,
        "manifest": manifest_record,
    }
    ready_object = f"{root}/runs/{args.run_id}/READY"
    ready_uri = f"gs://{bucket}/{ready_object}"
    storage.validate_write_uri(ready_uri, "runs", policy, run_id=args.run_id)
    ordered_uris.append(ready_uri)
    storage.validate_publication_order(ordered_uris)
    ready_record = client.create_and_verify(bucket, ready_object, encode_json(ready_payload), "application/json")
    result = {
        "stage": "M33_TABIX_SYNTHETIC_KAT_PUBLISHER",
        "status": "PASS_READY_CREATED_LAST",
        "run_id": args.run_id,
        "runtime_service_account": runtime_email,
        "objects": [*published, manifest_record, ready_record],
        "ready_is_last": True,
        "contains_real_genomic_data": False,
    }
    write_json_exclusive(args.output, result)
    return result


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
