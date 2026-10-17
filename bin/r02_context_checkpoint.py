#!/usr/bin/env python3
"""Safe local checkpoints for one authenticated R02 training task.

Checkpoint arrays are data-only NPZ files loaded with ``allow_pickle=False``.
An immutable generation becomes visible only when its manifest and payload are
durable and CURRENT is atomically replaced.  Orphan generations are harmless.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import random
import re
import tempfile
import zipfile

import numpy as np
import torch


SCHEMA = "r02_context_internal_checkpoint_v1"
MAX_UNCOMPRESSED_BYTES = 16 * 1024**3
GENERATION_FILE = re.compile(r"^(checkpoint-[0-9]{12})\.(json|npz)$")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_digest(identity):
    return _sha_bytes(_canonical(identity))


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path, content):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path, arrays):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_finite_arrays(arrays):
    """Reject state that the strict restore path could not load."""
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.inexact):
            _require(np.isfinite(value).all(),
                     f"refusing to promote nonfinite checkpoint array: {key}")


def _encode(value, arrays, prefix):
    if torch.is_tensor(value):
        key = f"{prefix}_{len(arrays):06d}"
        array = value.detach().cpu().numpy()
        arrays[key] = array
        return {"type": "tensor", "key": key, "dtype": str(array.dtype), "shape": list(array.shape)}
    if isinstance(value, np.ndarray):
        key = f"{prefix}_{len(arrays):06d}"
        arrays[key] = value
        return {"type": "ndarray", "key": key, "dtype": str(value.dtype), "shape": list(value.shape)}
    if value is None or isinstance(value, (bool, str, int, float)):
        _canonical(value)  # Reject non-finite floats.
        return {"type": "scalar", "value": value}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item, arrays, prefix) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item, arrays, prefix) for item in value]}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[_encode(key, arrays, prefix), _encode(item, arrays, prefix)]
                                              for key, item in value.items()]}
    raise TypeError(f"unsupported checkpoint value type: {type(value).__name__}")


def _decode(spec, arrays, used):
    _require(isinstance(spec, dict) and isinstance(spec.get("type"), str), "invalid checkpoint state tree")
    kind = spec["type"]
    if kind in ("tensor", "ndarray"):
        key = spec.get("key")
        _require(key in arrays and key not in used, "missing or duplicate checkpoint array")
        value = arrays[key]
        _require(str(value.dtype) == spec.get("dtype") and list(value.shape) == spec.get("shape"),
                 "checkpoint array dtype/shape differs from manifest")
        _require(value.dtype != object, "object checkpoint arrays are forbidden")
        if np.issubdtype(value.dtype, np.inexact):
            _require(np.isfinite(value).all(), "nonfinite checkpoint array")
        used.add(key)
        copied = value.copy()
        return torch.from_numpy(copied) if kind == "tensor" else copied
    if kind == "scalar":
        value = spec.get("value")
        _require(value is None or isinstance(value, (bool, str, int, float)), "invalid checkpoint scalar")
        _canonical(value)
        return value
    if kind in ("tuple", "list"):
        items = spec.get("items")
        _require(isinstance(items, list), "invalid checkpoint sequence")
        decoded = [_decode(item, arrays, used) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        items = spec.get("items")
        _require(isinstance(items, list), "invalid checkpoint mapping")
        decoded = {}
        for pair in items:
            _require(isinstance(pair, list) and len(pair) == 2, "invalid checkpoint mapping item")
            key = _decode(pair[0], arrays, used)
            _require(key not in decoded, "duplicate checkpoint mapping key")
            decoded[key] = _decode(pair[1], arrays, used)
        return decoded
    raise ValueError("unsupported checkpoint state tag")


class CheckpointStore:
    """Atomic generation store bound to one exact training identity."""

    def __init__(self, root, identity):
        self.root = Path(root)
        _require("://" not in str(root), "checkpoint directory must be a local filesystem path")
        self.root.mkdir(parents=True, exist_ok=True)
        _require(self.root.is_dir() and not self.root.is_symlink(), "checkpoint root must be a real directory")
        self._lock = (self.root / ".LOCK").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            raise ValueError("another writer owns this task checkpoint directory") from error
        self.identity = identity
        self.identity_sha256 = identity_digest(identity)
        self._current_generation = None

    def close(self):
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()

    def _cleanup_generations(self, keep):
        """Bound storage after CURRENT is durable; extra/orphan files are safe until then."""
        for path in self.root.iterdir():
            match = GENERATION_FILE.fullmatch(path.name)
            if match and match.group(1) not in keep:
                path.unlink()
        _fsync_directory(self.root)

    def save(self, model, optimizer, next_step, history):
        _require(type(next_step) is int and next_step > 0 and len(history) == next_step,
                 "checkpoint step/history mismatch")
        arrays = {}
        state = {
            "model": _encode(model.state_dict(), arrays, "model"),
            "optimizer": _encode(optimizer.state_dict(), arrays, "optimizer"),
            "python_rng": _encode(random.getstate(), arrays, "python_rng"),
            "numpy_rng": _encode(np.random.get_state(), arrays, "numpy_rng"),
            "torch_rng": _encode(torch.get_rng_state(), arrays, "torch_rng"),
        }
        # Validate before writing a payload, manifest, or CURRENT. This keeps
        # the save and restore barriers symmetric and prevents a nonfinite Adam
        # second moment from becoming the advertised resumable generation.
        _require_finite_arrays(arrays)
        generation = f"checkpoint-{next_step:012d}"
        payload = self.root / f"{generation}.npz"
        manifest_path = self.root / f"{generation}.json"
        _atomic_npz(payload, arrays)
        manifest = {"schema": SCHEMA, "generation": generation, "next_step": next_step,
                    "history": list(history), "identity": self.identity,
                    "identity_sha256": self.identity_sha256,
                    "payload": payload.name, "payload_sha256": _sha_file(payload),
                    "array_keys": sorted(arrays), "state": state}
        manifest_bytes = _canonical(manifest)
        _atomic_bytes(manifest_path, manifest_bytes)
        pointer = {"schema": SCHEMA, "manifest": manifest_path.name,
                   "manifest_sha256": _sha_bytes(manifest_bytes), "next_step": next_step,
                   "identity_sha256": self.identity_sha256}
        _atomic_bytes(self.root / "CURRENT", _canonical(pointer))
        keep = {generation}
        if self._current_generation is not None:
            keep.add(self._current_generation)
        self._cleanup_generations(keep)
        self._current_generation = generation
        return manifest

    def restore(self, model, optimizer):
        current = self.root / "CURRENT"
        if not current.exists():
            return 0, []
        try:
            pointer_bytes = current.read_bytes()
            pointer = json.loads(pointer_bytes)
            _require(pointer.get("schema") == SCHEMA and
                     pointer.get("identity_sha256") == self.identity_sha256,
                     "checkpoint identity is incompatible with this task")
            manifest_name = pointer.get("manifest")
            _require(isinstance(manifest_name, str) and Path(manifest_name).name == manifest_name,
                     "invalid checkpoint manifest name")
            manifest_path = self.root / manifest_name
            manifest_bytes = manifest_path.read_bytes()
            _require(_sha_bytes(manifest_bytes) == pointer.get("manifest_sha256"),
                     "checkpoint manifest integrity failure")
            manifest = json.loads(manifest_bytes)
            _require(manifest.get("schema") == SCHEMA and manifest.get("identity") == self.identity and
                     manifest.get("identity_sha256") == self.identity_sha256,
                     "checkpoint manifest identity is incompatible")
            _require(manifest.get("next_step") == pointer.get("next_step") and
                     manifest.get("generation") == Path(manifest_name).stem,
                     "checkpoint step/generation mismatch")
            payload_name = manifest.get("payload")
            _require(isinstance(payload_name, str) and Path(payload_name).name == payload_name,
                     "invalid checkpoint payload name")
            payload = self.root / payload_name
            _require(_sha_file(payload) == manifest.get("payload_sha256"),
                     "checkpoint payload integrity failure")
            with zipfile.ZipFile(payload) as archive:
                _require(sum(item.file_size for item in archive.infolist()) <= MAX_UNCOMPRESSED_BYTES,
                         "checkpoint exceeds safety bound")
            with np.load(payload, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            _require(sorted(arrays) == manifest.get("array_keys"), "checkpoint array inventory differs")
            used = set()
            state = manifest.get("state", {})
            model_state = _decode(state.get("model"), arrays, used)
            optimizer_state = _decode(state.get("optimizer"), arrays, used)
            python_rng = _decode(state.get("python_rng"), arrays, used)
            numpy_rng = _decode(state.get("numpy_rng"), arrays, used)
            torch_rng = _decode(state.get("torch_rng"), arrays, used)
            _require(used == set(arrays), "unreferenced checkpoint arrays")
            next_step, history = manifest.get("next_step"), manifest.get("history")
            _require(type(next_step) is int and isinstance(history, list) and len(history) == next_step and
                     all(type(value) is float and math_isfinite(value) for value in history),
                     "invalid checkpoint history")
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            self._current_generation = manifest["generation"]
            return next_step, history
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
            raise ValueError(f"invalid internal checkpoint: {error}") from error


def math_isfinite(value):
    # Avoid importing a broad serialization mechanism merely for one predicate.
    return value == value and value not in (float("inf"), float("-inf"))
