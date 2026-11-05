#!/usr/bin/env python3
"""Authenticated, restartable disk cache for one reconstructed LAI region.

The cache is an operational copy of arrays already authenticated by
``load_training_link``.  Its identity cannot admit a different data source or
configuration, and an unpublished staging directory is never read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import Lock
from typing import Mapping

import numpy as np


CACHE_SCHEMA = "r02_lai_duration_region_cache_v1"
PROGRESS_SCHEMA = "r02_lai_duration_region_cache_progress_v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
BLOCK_BYTES = 1024 * 1024
MINIMUM_FREE_BYTES_AFTER_REGION = 2 * 1024 ** 3
_CACHE_SAVE_LOCK = Lock()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            result.update(block)
    return result.hexdigest()


def array_sha256_streaming(value: np.ndarray) -> str:
    """Match regional_channel_bridge.array_sha256 without materializing bytes."""
    array = np.asarray(value)
    _require(array.flags.c_contiguous, "cached arrays must be C contiguous")
    result = hashlib.sha256()
    result.update(array.dtype.str.encode())
    result.update(b"\0")
    result.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    result.update(b"\0")
    if array.size == 0:
        return result.hexdigest()
    view = memoryview(array).cast("B")
    try:
        for start in range(0, len(view), BLOCK_BYTES):
            result.update(view[start:start + BLOCK_BYTES])
    finally:
        view.release()
    return result.hexdigest()


def region_cache_identity(*, region_id: str, entry: Mapping[str, object],
                          contract_sha256: str, resolved_sha256: str,
                          emission_semantics: str,
                          code_sha256: Mapping[str, str]) -> dict[str, object]:
    _require(isinstance(region_id, str) and region_id, "region_id is required")
    _require(isinstance(entry, Mapping)
             and SHA256.fullmatch(str(entry.get("sha256", ""))) is not None
             and SHA256.fullmatch(str(entry.get("receipt_sha256", ""))) is not None,
             "cache identity requires request and wrapper receipt SHA256 values")
    _require(SHA256.fullmatch(contract_sha256 or "") is not None
             and SHA256.fullmatch(resolved_sha256 or "") is not None,
             "cache identity requires contract and resolved SHA256 values")
    _require(isinstance(emission_semantics, str) and emission_semantics,
             "cache identity requires emission semantics")
    _require(isinstance(code_sha256, Mapping) and code_sha256
             and all(isinstance(name, str) and name
                     and SHA256.fullmatch(value or "") is not None
                     for name, value in code_sha256.items()),
             "cache identity requires code SHA256 values")
    return {
        "schema": CACHE_SCHEMA,
        "region_id": region_id,
        "request_sha256": entry["sha256"],
        "wrapper_receipt_sha256": entry["receipt_sha256"],
        "contract_sha256": contract_sha256,
        "resolved_sha256": resolved_sha256,
        "emission_semantics": emission_semantics,
        "code_sha256": dict(sorted(code_sha256.items())),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        # The temporary file is deliberately retained as failure evidence.  It
        # is never a readable cache because only the final name is consulted.
        raise


class DurationRegionCache:
    """Store one exact reconstructed region as independently mmap-able arrays."""

    def __init__(self, root: str | Path, identity: Mapping[str, object],
                 expected_array_sha256: Mapping[str, str]):
        self.root = Path(root).resolve()
        self.identity = dict(identity)
        _require(self.identity.get("schema") == CACHE_SCHEMA,
                 "cache identity schema differs")
        self.identity_sha256 = digest(self.identity)
        _require(isinstance(expected_array_sha256, Mapping)
                 and expected_array_sha256
                 and all(isinstance(name, str) and name
                         and SHA256.fullmatch(value or "") is not None
                         for name, value in expected_array_sha256.items()),
                 "expected reconstructed-array SHA256 values are required")
        self.expected = dict(sorted(expected_array_sha256.items()))
        self.directory = self.root / self.identity_sha256

    def _load_directory(self, directory: Path) -> dict[str, np.ndarray]:
        manifest_path = directory / "manifest.json"
        _require(manifest_path.is_file(), "published cache manifest is absent")
        manifest = json.loads(manifest_path.read_text())
        _require(isinstance(manifest, dict)
                 and manifest.get("schema") == CACHE_SCHEMA
                 and manifest.get("status") == "COMPLETE"
                 and manifest.get("identity") == self.identity
                 and manifest.get("identity_sha256") == self.identity_sha256
                 and manifest.get("expected_array_sha256") == self.expected,
                 "published cache identity or expected hashes differ")
        rows = manifest.get("arrays")
        _require(isinstance(rows, list) and len(rows) == len(self.expected),
                 "published cache array manifest differs")
        loaded: dict[str, np.ndarray] = {}
        for row in rows:
            _require(isinstance(row, dict)
                     and set(row) == {"name", "file", "shape", "dtype", "nbytes",
                                     "file_sha256", "array_sha256"},
                     "published cache array row differs")
            name = row["name"]
            _require(name in self.expected and name not in loaded
                     and row["file"] == f"{len(loaded):03d}.npy"
                     and row["array_sha256"] == self.expected[name]
                     and SHA256.fullmatch(row["file_sha256"] or "") is not None,
                     "published cache array name or hash differs")
            path = directory / row["file"]
            _require(path.is_file() and file_sha256(path) == row["file_sha256"],
                     f"cached array file SHA256 differs: {name}")
            value = np.load(path, allow_pickle=False, mmap_mode="r")
            _require(list(value.shape) == row["shape"]
                     and value.dtype.str == row["dtype"]
                     and int(value.nbytes) == row["nbytes"]
                     and value.flags.c_contiguous,
                     f"cached array shape, dtype, or size differs: {name}")
            _require(array_sha256_streaming(value) == self.expected[name],
                     f"cached reconstructed-array SHA256 differs: {name}")
            loaded[name] = value
        _require(set(loaded) == set(self.expected),
                 "published cache omits or adds arrays")
        return loaded

    def load(self) -> dict[str, np.ndarray] | None:
        if not self.directory.exists():
            return None
        _require(self.directory.is_dir(), "published cache path is not a directory")
        return self._load_directory(self.directory)

    def save(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        # One admitted process can reconstruct two regions concurrently.  Keep
        # its capacity check and publication indivisible across both writers;
        # source reads and reconstruction still run in parallel.
        with _CACHE_SAVE_LOCK:
            return self._save_locked(arrays)

    def _save_locked(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        existing = self.load()
        if existing is not None:
            return existing
        _require(isinstance(arrays, Mapping) and set(arrays) == set(self.expected),
                 "reconstructed arrays differ from authenticated receipt names")
        self.root.mkdir(parents=True, exist_ok=True)
        required = (sum(int(np.asarray(value).nbytes) for value in arrays.values())
                    + 4096 * len(arrays) + BLOCK_BYTES)
        _require(shutil.disk_usage(self.root).free >= required + MINIMUM_FREE_BYTES_AFTER_REGION,
                 "region cache would consume the required 2 GiB free-space reserve")
        staging = Path(tempfile.mkdtemp(prefix=".region-staging-", dir=self.root))
        rows = []
        for index, name in enumerate(sorted(arrays)):
            value = np.asarray(arrays[name])
            # ascontiguousarray promotes a scalar from shape () to (1,).
            # Shape is part of the authenticated content, including metadata.
            if not value.flags.c_contiguous:
                value = np.ascontiguousarray(value)
            _require(array_sha256_streaming(value) == self.expected[name],
                     f"reconstructed array differs from authenticated receipt: {name}")
            filename = f"{index:03d}.npy"
            path = staging / filename
            with path.open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            rows.append({
                "name": name, "file": filename, "shape": list(value.shape),
                "dtype": value.dtype.str, "nbytes": int(value.nbytes),
                "file_sha256": file_sha256(path),
                "array_sha256": self.expected[name],
            })
        manifest = {
            "schema": CACHE_SCHEMA, "status": "COMPLETE",
            "identity": self.identity, "identity_sha256": self.identity_sha256,
            "expected_array_sha256": self.expected, "arrays": rows,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("wb") as stream:
            stream.write(_canonical(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        # rename publishes all files and the manifest as one directory entry.
        # Input arrays and written files were hashed above; the final load
        # verifies persisted content.  Do not read and hash the whole region
        # again before the atomic rename.
        os.rename(staging, self.directory)
        _fsync_directory(self.root)
        return self._load_directory(self.directory)


def write_region_progress(path: str | Path, *, case_id: str, region_id: str,
                          cache_identity_sha256: str, result: str,
                          completed_regions: int, source_loader_calls: int,
                          disk_cache_hits: int, disk_cache_writes: int,
                          elapsed_seconds: float | None = None) -> None:
    _require(result in {"SOURCE_AUTHENTICATED_AND_CACHED", "CACHE_VERIFIED_AND_REUSED"},
             "unknown region cache result")
    value = {
        "schema": PROGRESS_SCHEMA, "case_id": case_id,
        "last_completed_region_id": region_id,
        "last_cache_identity_sha256": cache_identity_sha256,
        "last_result": result, "completed_regions": completed_regions,
        "source_loader_calls": source_loader_calls,
        "disk_cache_hits": disk_cache_hits, "disk_cache_writes": disk_cache_writes,
        "contains_person_or_component_identifiers": False,
        "last_region_elapsed_seconds": elapsed_seconds,
    }
    atomic_json(path, value)
