#!/usr/bin/env python3
"""Bounded, read-only GCS metadata verifier for R02 retirement gates.

The verifier reads current object metadata through the authenticated GCS JSON
API.  It never requests an older generation: a retained historical generation
must still be the current object generation to pass retirement verification.
One persistent HTTPS connection is kept per worker thread.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import re
import subprocess
import threading
import time
from urllib.parse import quote


IMPLEMENTATION_VERSION = "1.0.0"
API_HOST = "storage.googleapis.com"
ALLOWED_WORKERS = {2, 4}
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
SIGNATURE_FIELDS = ("bytes", "md5_base64", "generation")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def parse_gs_uri(uri):
    require(isinstance(uri, str) and uri.startswith("gs://") and
            not any(char in uri for char in "\n\r#?"), "Invalid GCS metadata URI")
    bucket, separator, name = uri[5:].partition("/")
    require(separator and bucket and name and re.fullmatch(r"[a-z0-9][a-z0-9._-]+", bucket),
            "Invalid GCS metadata URI")
    return bucket, name


def access_token(account, project, *, runner=subprocess.run):
    require(isinstance(account, str) and account and "\n" not in account,
            "Explicit GCP account required")
    require(isinstance(project, str) and project and "\n" not in project,
            "Explicit GCP project required")
    result = runner([
        "gcloud", "auth", "print-access-token",
        "--account=" + account, "--project=" + project,
    ], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError("Cannot obtain access token for retirement metadata verification: " +
                           (result.stderr or "")[-500:])
    token = result.stdout.strip()
    require(token and not any(char.isspace() for char in token), "Invalid access token response")
    return token


class GCSMetadataVerifier:
    """Verify current GCS size, MD5 and generation with bounded concurrency."""

    def __init__(self, *, account, project, workers=4, timeout_seconds=30,
                 retries=3, token_provider=access_token, connection_factory=None,
                 sleeper=time.sleep):
        require(workers in ALLOWED_WORKERS, "Metadata workers must be 2 or 4")
        require(type(timeout_seconds) in {int, float} and 0 < timeout_seconds <= 120,
                "Metadata timeout must be within (0, 120] seconds")
        require(type(retries) is int and 1 <= retries <= 5,
                "Metadata retries must be within 1..5")
        require(os.environ.get("CLOUDSDK_CORE_ACCOUNT") == account,
                "Set CLOUDSDK_CORE_ACCOUNT to the recorded account")
        require(os.environ.get("CLOUDSDK_CORE_PROJECT") == project,
                "Set CLOUDSDK_CORE_PROJECT to the recorded project")
        self.account = account
        self.project = project
        self.workers = workers
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._token = token_provider(account, project)
        self._connection_factory = connection_factory or (
            lambda: http.client.HTTPSConnection(API_HOST, timeout=timeout_seconds))
        self._local = threading.local()
        self._sleeper = sleeper
        self._executor = ThreadPoolExecutor(max_workers=self.workers,
                                            thread_name_prefix="r02-gcs-metadata")

    def _connection(self):
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connection_factory()
            self._local.connection = connection
        return connection

    def _reset_connection(self):
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            try:
                connection.close()
            finally:
                self._local.connection = None

    def metadata(self, uri):
        bucket, name = parse_gs_uri(uri)
        path = ("/storage/v1/b/" + quote(bucket, safe="") + "/o/" +
                quote(name, safe="") + "?fields=bucket%2Cname%2Csize%2Cmd5Hash%2Cgeneration")
        for attempt in range(self.retries):
            try:
                connection = self._connection()
                connection.request("GET", path, headers={
                    "Authorization": "Bearer " + self._token,
                    "Accept": "application/json",
                    "User-Agent": "dnabr-r02-retirement-metadata/" + IMPLEMENTATION_VERSION,
                })
                response = connection.getresponse()
                payload = response.read()
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                self._reset_connection()
                if attempt + 1 == self.retries:
                    raise RuntimeError("GCS metadata request failed for " + uri) from error
                self._sleeper(2 ** attempt)
                continue
            if response.status == 200:
                try:
                    value = json.loads(payload)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    self._reset_connection()
                    raise RuntimeError("Invalid GCS metadata response for " + uri) from error
                require(isinstance(value, dict), "Invalid GCS metadata response for " + uri)
                return value
            if response.status == 404:
                raise RuntimeError("GCS object missing: " + uri)
            if response.status in TRANSIENT_STATUS and attempt + 1 < self.retries:
                self._reset_connection()
                self._sleeper(2 ** attempt)
                continue
            raise RuntimeError("GCS metadata HTTP " + str(response.status) + " for " + uri)
        raise AssertionError("unreachable")

    def __call__(self, record):
        metadata = self.metadata(record["uri"])
        size = metadata.get("size")
        generation = metadata.get("generation")
        require(not isinstance(size, bool) and re.fullmatch(r"[0-9]+", str(size)) and
                int(size) == record["bytes"], "Published GCS size differs: " + record["uri"])
        require(metadata.get("md5Hash") == record["md5_base64"],
                "Published GCS MD5 differs or is missing: " + record["uri"])
        require(not isinstance(generation, bool) and str(generation) == record["generation"],
                "Published GCS generation changed: " + record["uri"])
        return str(generation)

    def verify_all(self, records):
        unique = {}
        for record in records:
            uri = record.get("uri")
            signature = tuple(record.get(field) for field in SIGNATURE_FIELDS)
            require(uri not in unique or unique[uri][0] == signature,
                    "Conflicting metadata expectations for " + str(uri))
            unique[uri] = signature, record
        selected = [value[1] for value in unique.values()]
        generations = list(self._executor.map(self, selected))
        return dict(backend="GCS_JSON_API_CURRENT_METADATA",
                    implementation_version=IMPLEMENTATION_VERSION,
                    workers=self.workers, verified_objects=len(selected),
                    generations=generations)

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def for_plan(plan, **kwargs):
    require(plan.get("team") == "frank", "Retirement plan must retain team=frank")
    return GCSMetadataVerifier(account=plan["account"], project=plan["project"], **kwargs)
