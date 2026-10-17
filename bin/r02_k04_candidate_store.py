#!/usr/bin/env python3
"""Crash-safe per-candidate state for one authenticated K04 FIT/SELECT task.

The store owns only operational continuation state.  It cannot alter the W
family, targets, loss, tolerances or convergence rule supplied by the caller.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


STATE_SCHEMA = "r02_k04_candidate_store_v1"
RECEIPT_SCHEMA = "r02_k04_completed_candidate_v1"
OUTPUT_SCHEMA = "r02_k04_fit_select_owned_output_v1"
HEX = re.compile(r"[0-9a-f]{64}")
TOKEN = re.compile(r"[A-Za-z0-9_]+")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                              dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_model(path, intercept, coefficients):
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    value = np.asarray(coefficients, dtype=np.float64)
    require(value.ndim == 1 and np.isfinite(value).all() and math.isfinite(float(intercept)),
            "Finite one-dimensional candidate model required")
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                              dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, intercept=np.asarray(float(intercept)), coefficients=value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    target = Path(path)
    require(target.is_file() and not target.is_symlink(), "Real JSON file required: " + str(target))
    with target.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "JSON object required: " + str(target))
    return value


def _budgets(value):
    require(set(value) == {"max_iterations", "max_seconds_per_candidate"}
            and type(value["max_iterations"]) is int and value["max_iterations"] > 0
            and type(value["max_seconds_per_candidate"]) in (int, float)
            and math.isfinite(value["max_seconds_per_candidate"])
            and value["max_seconds_per_candidate"] > 0,
            "Exact positive K04 continuation budgets required")
    return {"max_iterations": value["max_iterations"],
            "max_seconds_per_candidate": float(value["max_seconds_per_candidate"])}


class CandidateStore:
    """Single-writer store with immutable task identity and monotone budgets."""

    def __init__(self, root, immutable_identity, plan_sha256, budgets,
                 allow_compatible_budget_extension=False):
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.parent.is_dir()
                and not self.root.parent.is_symlink(),
                "Candidate state root must have a real existing absolute parent")
        require(isinstance(immutable_identity, dict), "Immutable task identity required")
        require(HEX.fullmatch(plan_sha256) is not None, "Exact fit-plan SHA-256 required")
        require(isinstance(allow_compatible_budget_extension, bool),
                "Explicit budget-extension flag must be boolean")
        self.identity = immutable_identity
        self.identity_sha256 = canonical_sha256(immutable_identity)
        self.plan_sha256 = plan_sha256
        self.budgets = _budgets(budgets)
        self.allow_extension = allow_compatible_budget_extension
        self.lock_stream = None
        self.state = None

    def __enter__(self):
        if self.root.exists():
            require(self.root.is_dir() and not self.root.is_symlink(),
                    "Candidate state root must be a real directory")
        else:
            self.root.mkdir(mode=0o700)
            _fsync_directory(self.root.parent)
        lock_path = self.root / ".lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        self.lock_stream = os.fdopen(descriptor, "r+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock_stream.close(); self.lock_stream = None
            raise ValueError("Another writer owns this K04 candidate state") from error
        self.lock_stream.seek(0); self.lock_stream.truncate()
        self.lock_stream.write(json.dumps({"pid": os.getpid(),
                                           "identity_sha256": self.identity_sha256}) + "\n")
        self.lock_stream.flush(); os.fsync(self.lock_stream.fileno())
        try:
            self._open_state()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lock_stream is not None:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
            self.lock_stream.close(); self.lock_stream = None

    def _open_state(self):
        path = self.root / "state.json"
        if not path.exists():
            self.state = {"schema": STATE_SCHEMA, "immutable_identity": self.identity,
                "immutable_identity_sha256": self.identity_sha256,
                "current_plan_sha256": self.plan_sha256, "current_budgets": self.budgets,
                "budget_amendments": []}
            atomic_json(path, self.state)
            return
        value = _read_json(path)
        require(set(value) == {"schema", "immutable_identity", "immutable_identity_sha256",
                "current_plan_sha256", "current_budgets", "budget_amendments"}
                and value["schema"] == STATE_SCHEMA
                and value["immutable_identity_sha256"] == canonical_sha256(value["immutable_identity"])
                and value["immutable_identity_sha256"] == self.identity_sha256
                and value["immutable_identity"] == self.identity
                and isinstance(value["budget_amendments"], list),
                "Candidate state identity is incompatible or corrupt")
        current = _budgets(value["current_budgets"])
        if value["current_plan_sha256"] == self.plan_sha256:
            require(current == self.budgets, "Same fit-plan hash has different technical budgets")
        else:
            require(self.allow_extension,
                    "Changed fit plan requires --allow-compatible-budget-extension")
            require(all(self.budgets[key] >= current[key] for key in current)
                    and any(self.budgets[key] > current[key] for key in current),
                    "Amended technical budgets must be a strict monotone extension")
            value["budget_amendments"].append({
                "from_plan_sha256": value["current_plan_sha256"],
                "to_plan_sha256": self.plan_sha256,
                "from_budgets": current, "to_budgets": self.budgets})
            value["current_plan_sha256"] = self.plan_sha256
            value["current_budgets"] = self.budgets
            atomic_json(path, value)
        self.state = value

    def _candidate_directory(self, arm, candidate_id):
        require(TOKEN.fullmatch(arm) is not None and TOKEN.fullmatch(candidate_id) is not None,
                "Canonical candidate arm and ID required")
        path = self.root / "candidates" / arm / candidate_id
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(path.is_dir() and not path.is_symlink(), "Candidate directory must be real")
        return path

    def _candidate_identity(self, arm, candidate_id, penalty):
        require(penalty is None or isinstance(penalty, float) and math.isfinite(penalty) and penalty > 0,
                "Explicit candidate penalty required")
        return {"schema": "r02_k04_candidate_identity_v1",
                "task_identity_sha256": self.identity_sha256, "arm": arm,
                "candidate_id": candidate_id,
                "penalty_hex": None if penalty is None else penalty.hex()}

    def checkpoint_path(self, arm, candidate_id):
        return self._candidate_directory(arm, candidate_id) / "checkpoint.npz"

    def load_complete(self, arm, candidate_id, penalty, n_coefficients):
        folder = self._candidate_directory(arm, candidate_id)
        receipt_path = folder / "receipt.json"
        if not receipt_path.exists():
            return None
        receipt = _read_json(receipt_path)
        identity = self._candidate_identity(arm, candidate_id, penalty)
        require(set(receipt) == {"schema", "status", "identity", "identity_sha256",
                "model", "result"} and receipt["schema"] == RECEIPT_SCHEMA
                and receipt["status"] == "COMPLETE_CONVERGED_CANDIDATE"
                and receipt["identity"] == identity
                and receipt["identity_sha256"] == canonical_sha256(identity),
                "Completed candidate receipt identity differs")
        model = receipt["model"]
        require(set(model) == {"path", "bytes", "sha256"} and model["path"] == "model.npz"
                and type(model["bytes"]) is int and model["bytes"] > 0
                and HEX.fullmatch(model["sha256"]) is not None,
                "Completed candidate model descriptor differs")
        model_path = folder / model["path"]
        require(model_path.is_file() and not model_path.is_symlink()
                and model_path.stat().st_size == model["bytes"]
                and file_sha256(model_path) == model["sha256"],
                "Completed candidate model is missing or corrupt")
        with np.load(model_path, allow_pickle=False) as saved:
            require(set(saved.files) == {"intercept", "coefficients"},
                    "Completed candidate model fields differ")
            intercept = float(saved["intercept"])
            coefficients = np.asarray(saved["coefficients"], dtype=np.float64).copy()
        require(coefficients.shape == (n_coefficients,) and np.isfinite(coefficients).all()
                and math.isfinite(intercept), "Completed candidate model shape/value differs")
        result = receipt["result"]
        require(set(result) == {"status", "coefficient_error_bound", "iterations",
                "elapsed_seconds", "diagnostics", "provenance"}
                and str(result["status"]).startswith("CONVERGED")
                and type(result["iterations"]) is int and result["iterations"] >= 0
                and type(result["elapsed_seconds"]) in (int, float)
                and math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= 0
                and type(result["coefficient_error_bound"]) in (int, float)
                and math.isfinite(result["coefficient_error_bound"])
                and result["coefficient_error_bound"] >= 0
                and isinstance(result["diagnostics"], dict)
                and isinstance(result["provenance"], dict),
                "Completed candidate result differs")
        return {"id": candidate_id, "penalty": penalty, "status": result["status"],
                "intercept": intercept, "coefficients": coefficients,
                "coefficient_error_bound": float(result["coefficient_error_bound"]),
                "iterations": result["iterations"],
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "diagnostics": result["diagnostics"], "provenance": result["provenance"],
                "reused_completed_candidate": True, "state_model_path": str(model_path)}

    def save_complete(self, arm, candidate):
        candidate_id, penalty = candidate["id"], candidate["penalty"]
        identity = self._candidate_identity(arm, candidate_id, penalty)
        require(str(candidate["status"]).startswith("CONVERGED")
                and math.isfinite(float(candidate["coefficient_error_bound"]))
                and float(candidate["coefficient_error_bound"]) >= 0,
                "Only converged bounded candidates can be completed")
        folder = self._candidate_directory(arm, candidate_id)
        receipt_path, model_path = folder / "receipt.json", folder / "model.npz"
        require(not receipt_path.exists(), "Completed candidate receipt is immutable")
        atomic_model(model_path, candidate["intercept"], candidate["coefficients"])
        result = {"status": candidate["status"],
            "coefficient_error_bound": float(candidate["coefficient_error_bound"]),
            "iterations": int(candidate.get("iterations", 0)),
            "elapsed_seconds": float(candidate.get("elapsed_seconds", 0.)),
            "diagnostics": candidate.get("diagnostics", {}),
            "provenance": candidate.get("provenance", {})}
        receipt = {"schema": RECEIPT_SCHEMA, "status": "COMPLETE_CONVERGED_CANDIDATE",
            "identity": identity, "identity_sha256": canonical_sha256(identity),
            "model": {"path": "model.npz", "bytes": model_path.stat().st_size,
                      "sha256": file_sha256(model_path)}, "result": result}
        atomic_json(receipt_path, receipt)
        return receipt

    def prepare_output(self, output):
        path = Path(output)
        marker_path = path / ".r02-k04-output.json"
        if path.exists():
            require(path.is_dir() and not path.is_symlink() and not (path / "manifest.json").exists(),
                    "Existing K04 output must be an owned incomplete directory")
            if not marker_path.exists():
                require(not any(path.iterdir()),
                        "Unmarked existing K04 output is not an empty interrupted creation")
                atomic_json(marker_path, {"schema": OUTPUT_SCHEMA,
                                          "task_identity_sha256": self.identity_sha256})
            marker = _read_json(marker_path)
            require(marker == {"schema": OUTPUT_SCHEMA,
                                "task_identity_sha256": self.identity_sha256},
                    "Existing K04 output belongs to another task")
        else:
            require(path.parent.is_dir() and not path.parent.is_symlink(),
                    "K04 output parent must be a real directory")
            path.mkdir(mode=0o700)
            atomic_json(marker_path, {"schema": OUTPUT_SCHEMA,
                                      "task_identity_sha256": self.identity_sha256})
        return path

    def materialize_model(self, arm, candidate, output_path):
        source = candidate.get("state_model_path")
        target = Path(output_path)
        if source is not None:
            source = Path(source)
            require(source.is_file() and not source.is_symlink(), "Reusable candidate model missing")
            descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                                      dir=target.parent)
            try:
                with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
                    shutil.copyfileobj(reader, writer, 8 << 20)
                    writer.flush(); os.fsync(writer.fileno())
                os.chmod(temporary, 0o600); os.replace(temporary, target)
                _fsync_directory(target.parent)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        else:
            atomic_model(target, candidate["intercept"], candidate["coefficients"])
        return file_sha256(target)
