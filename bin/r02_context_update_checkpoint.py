#!/usr/bin/env python3
"""Atomic, data-only progress for one in-flight autosomal optimizer update.

The payload contains tensors and scalar metadata only.  Autograd graphs are
never serialized: tensors are detached by the shared checkpoint encoder and a
resume reconstructs the graph for the next chromosome tile.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np
import torch

from r02_context_checkpoint import (_atomic_bytes, _atomic_npz, _canonical,
    _decode, _encode, _fsync_directory, _require, _require_finite_arrays,
    _sha_bytes, _sha_file, identity_digest, MAX_UNCOMPRESSED_BYTES)


SCHEMA = "r02_context_partial_update_v1"
BURDEN_SCHEMA = "r02_context_autosomal_burden_counts_v1"
BUDGET_SCHEMA = "r02_context_partial_update_budget_v1"
BUDGET_FILE = "DURABLE_GRADIENT_BUDGET.json"


def _state_digest(value):
    """Content digest for a nested tensor state without pickle or object IDs."""
    arrays = {}
    tree = _encode(value, arrays, "state")
    digest = hashlib.sha256(_canonical(tree))
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(_canonical([key, array.dtype.str, list(array.shape)]))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def optimizer_start_digest(model, optimizer):
    """Bind partial progress to the exact pre-update weights and Adam state."""
    return _state_digest({"model": model.state_dict(),
                          "optimizer": optimizer.state_dict()})


def update_identity(training_identity, update_index, fit_batch_index, entry,
                    completed_history, model_optimizer_sha256):
    """Name one update using frozen sources, recipe, code and pre-update state."""
    _require(type(update_index) is int and update_index >= 0,
             "nonnegative update index required")
    _require(type(fit_batch_index) is int and fit_batch_index >= 0,
             "nonnegative FIT batch index required")
    source = {key: entry.get(key) for key in (
        "path", "sha256", "receipt_sha256", "autosomal_request_sha256")}
    _require(all(source.values()), "partial update requires exact target/request identities")
    history = [float(value) for value in completed_history]
    _require(len(history) == update_index and all(np.isfinite(history)),
             "completed history does not identify this update")
    _require(isinstance(model_optimizer_sha256, str) and
             len(model_optimizer_sha256) == 64,
             "pre-update model/optimizer digest required")
    code = training_identity.get("code_sha256", {})
    burden_code_names = ["r02_context_train.py", "r02_context_autosomal_dataset.py",
                         "r02_genomic_event_inputs.py"]
    if training_identity["dataset_mode"] == "AUTOSOMAL_PERSON_BATCHES_V1":
        burden_code_names.append("r02_event_mmap_reader.py")
    _require(all(isinstance(code.get(name), str) and len(code[name]) == 64
                 for name in burden_code_names),
             "exact burden calculation/reader code hashes required")
    return {
        "schema": SCHEMA,
        "training_identity_sha256": identity_digest(training_identity),
        "contract_sha256": training_identity["contract_sha256"],
        "resolved_sha256": training_identity["resolved_sha256"],
        "arm": training_identity["arm"],
        "seed": training_identity["seed"],
        "adam_state_precision": training_identity["adam_state_precision"],
        "dataset_mode": training_identity["dataset_mode"],
        "update_index": update_index,
        "fit_batch_index": fit_batch_index,
        "source_and_masks": source,
        "burden_code_sha256": {name: code[name] for name in burden_code_names},
        "burden_numpy_version": training_identity.get("numpy_version"),
        "completed_history_sha256": _sha_bytes(_canonical(history)),
        "model_optimizer_start_sha256": model_optimizer_sha256,
    }


def burden_identity(update):
    """Identify raw burden counts without binding them to model weights.

    The authenticated autosomal request fixes people/order, counted allele,
    stores, cores, halos and exclusions.  The target and receipt hashes retain
    the exact paired response axis.  Model recipe, seed, weights and Adam state
    are deliberately absent because ``event_burden_counts`` reads genotypes and
    masks only.
    """
    _require(isinstance(update, dict) and update.get("schema") == SCHEMA,
             "partial update identity required for burden counts")
    return {"schema": BURDEN_SCHEMA,
            "contract_sha256": update["contract_sha256"],
            "dataset_mode": update["dataset_mode"],
            "source_and_masks": update["source_and_masks"],
            "calculation": "event_burden_counts_v1_carriers_copies_observed_loci",
            "calculation_and_reader_code_sha256": update["burden_code_sha256"],
            "numpy_version": update["burden_numpy_version"]}


def capture_parameter_gradients(model):
    """Return detached gradients keyed by the stable parameter names."""
    return {name: (None if parameter.grad is None else parameter.grad.detach().clone())
            for name, parameter in model.named_parameters()}


def restore_parameter_gradients(model, gradients):
    expected = dict(model.named_parameters())
    _require(isinstance(gradients, dict) and set(gradients) == set(expected),
             "partial gradient parameter inventory differs")
    for name, parameter in expected.items():
        value = gradients[name]
        if value is None:
            parameter.grad = None
            continue
        _require(torch.is_tensor(value) and value.shape == parameter.shape and
                 value.dtype == parameter.dtype and bool(torch.isfinite(value).all()),
                 f"partial gradient shape/dtype/value differs: {name}")
        parameter.grad = value.detach().clone()


class PartialUpdateStore:
    """One-writer atomic generations for a single unfinished update."""

    def __init__(self, root, identity):
        from r02_context_checkpoint import CheckpointStore
        # Reuse the proven local-directory and lock behavior without coupling
        # this partial schema to completed-step manifests.
        self._lock_owner = CheckpointStore(root, identity)
        self.root = self._lock_owner.root
        self.identity = identity
        self.identity_sha256 = identity_digest(identity)
        self._generation = 0
        self.last_progress = None

    def close(self):
        self._lock_owner.close()

    def save(self, model, optimizer, progress, state):
        _require(isinstance(progress, dict) and
                 progress.get("phase") in ("AUTOSOMAL_BURDENS",
                                            "AUTOSOMAL_AGGREGATION",
                                            "AUTOSOMAL_RECOMPUTATION",
                                            "DURATION_PERSON_CHUNKS",
                                            "LAI_PERSON_MICROBATCH") and
                 type(progress.get("next_person_batch_index")) is int and
                 type(progress.get("next_tile_index")) is int and
                 progress["next_person_batch_index"] >= 0 and
                 progress["next_tile_index"] >= 0,
                 "invalid partial-update progress")
        arrays = {}
        encoded = {
            "model": _encode(model.state_dict(), arrays, "model"),
            "optimizer": _encode(optimizer.state_dict(), arrays, "optimizer"),
            "python_rng": _encode(random.getstate(), arrays, "python_rng"),
            "numpy_rng": _encode(np.random.get_state(), arrays, "numpy_rng"),
            "torch_rng": _encode(torch.get_rng_state(), arrays, "torch_rng"),
            "partial": _encode(state, arrays, "partial"),
        }
        _require_finite_arrays(arrays)
        self._generation += 1
        generation = f"update-progress-{self._generation:012d}"
        payload = self.root / f"{generation}.npz"
        manifest_path = self.root / f"{generation}.json"
        _atomic_npz(payload, arrays)
        manifest = {
            "schema": SCHEMA, "generation": generation,
            "identity": self.identity, "identity_sha256": self.identity_sha256,
            "progress": progress, "payload": payload.name,
            "payload_sha256": _sha_file(payload), "array_keys": sorted(arrays),
            "state": encoded,
        }
        manifest_bytes = _canonical(manifest)
        _atomic_bytes(manifest_path, manifest_bytes)
        pointer = {"schema": SCHEMA, "manifest": manifest_path.name,
                   "manifest_sha256": _sha_bytes(manifest_bytes),
                   "identity_sha256": self.identity_sha256,
                   "phase": progress["phase"],
                   "next_person_batch_index": progress["next_person_batch_index"],
                   "next_tile_index": progress["next_tile_index"],
                   "next_recompute_index": progress.get("next_recompute_index")}
        _atomic_bytes(self.root / "CURRENT", _canonical(pointer))
        self._cleanup(generation)
        self.last_progress = dict(progress)
        return manifest

    def _cleanup(self, keep):
        for path in self.root.iterdir():
            if path.name.startswith("update-progress-") and path.stem != keep:
                path.unlink()
        _fsync_directory(self.root)

    def restore(self, model, optimizer):
        current = self.root / "CURRENT"
        if not current.exists():
            return None
        try:
            pointer_bytes = current.read_bytes()
            pointer = json.loads(pointer_bytes)
            _require(pointer.get("schema") == SCHEMA and
                     pointer.get("identity_sha256") == self.identity_sha256,
                     "partial update identity is incompatible with this task")
            name = pointer.get("manifest")
            _require(isinstance(name, str) and Path(name).name == name,
                     "invalid partial update manifest name")
            manifest_bytes = (self.root / name).read_bytes()
            _require(_sha_bytes(manifest_bytes) == pointer.get("manifest_sha256"),
                     "partial update manifest integrity failure")
            manifest = json.loads(manifest_bytes)
            _require(manifest.get("schema") == SCHEMA and
                     manifest.get("identity") == self.identity and
                     manifest.get("identity_sha256") == self.identity_sha256,
                     "partial update manifest identity is incompatible")
            payload_name = manifest.get("payload")
            _require(isinstance(payload_name, str) and Path(payload_name).name == payload_name,
                     "invalid partial update payload name")
            payload = self.root / payload_name
            _require(_sha_file(payload) == manifest.get("payload_sha256"),
                     "partial update payload integrity failure")
            with zipfile.ZipFile(payload) as archive:
                _require(sum(item.file_size for item in archive.infolist()) <= MAX_UNCOMPRESSED_BYTES,
                         "partial update exceeds safety bound")
            with np.load(payload, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            _require(sorted(arrays) == manifest.get("array_keys"),
                     "partial update array inventory differs")
            used = set(); encoded = manifest.get("state", {})
            model_state = _decode(encoded.get("model"), arrays, used)
            optimizer_state = _decode(encoded.get("optimizer"), arrays, used)
            python_rng = _decode(encoded.get("python_rng"), arrays, used)
            numpy_rng = _decode(encoded.get("numpy_rng"), arrays, used)
            torch_rng = _decode(encoded.get("torch_rng"), arrays, used)
            state = _decode(encoded.get("partial"), arrays, used)
            _require(used == set(arrays), "unreferenced partial update arrays")
            progress = manifest.get("progress")
            _require(isinstance(progress, dict) and
                     progress.get("phase") == pointer.get("phase") and
                     progress.get("next_person_batch_index") ==
                        pointer.get("next_person_batch_index") and
                     progress.get("next_tile_index") == pointer.get("next_tile_index") and
                     progress.get("next_recompute_index") == pointer.get("next_recompute_index"),
                     "partial update cursor differs")
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            random.setstate(python_rng); np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            self._generation = int(manifest["generation"].rsplit("-", 1)[1])
            self.last_progress = dict(progress)
            return progress, state
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
                zipfile.BadZipFile) as error:
            raise ValueError(f"invalid partial update checkpoint: {error}") from error

    @property
    def burden_cache_root(self):
        """Shared sibling cache: reusable across exact updates in one run."""
        return self.root.parent / "burden-counts"

    def save_burdens(self, identity, value):
        """Publish complete weight-independent counts as one immutable object."""
        _require(identity == burden_identity(self.identity),
                 "burden count identity differs from this update")
        arrays = {}
        encoded = _encode(value, arrays, "burdens")
        _require_finite_arrays(arrays)
        digest = identity_digest(identity)
        root = self.burden_cache_root
        root.mkdir(parents=True, exist_ok=True)
        payload = root / f"{digest}.npz"
        manifest_path = root / f"{digest}.json"
        if manifest_path.exists():
            existing = self.restore_burdens(identity)
            _require(_state_digest(existing) == _state_digest(value),
                     "existing burden counts differ under the same identity")
            return json.loads(manifest_path.read_bytes())
        _atomic_npz(payload, arrays)
        manifest = {"schema": BURDEN_SCHEMA, "identity": identity,
                    "identity_sha256": digest, "payload": payload.name,
                    "payload_sha256": _sha_file(payload),
                    "array_keys": sorted(arrays), "state": encoded}
        _atomic_bytes(manifest_path, _canonical(manifest))
        return manifest

    def restore_burdens(self, identity):
        """Load only the cache whose exact non-weight identity was requested."""
        _require(identity == burden_identity(self.identity),
                 "burden count identity differs from this update")
        digest = identity_digest(identity)
        manifest_path = self.burden_cache_root / f"{digest}.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_bytes())
            _require(manifest.get("schema") == BURDEN_SCHEMA and
                     manifest.get("identity") == identity and
                     manifest.get("identity_sha256") == digest,
                     "burden count manifest identity is incompatible")
            payload_name = manifest.get("payload")
            _require(isinstance(payload_name, str) and Path(payload_name).name == payload_name,
                     "invalid burden count payload name")
            payload = self.burden_cache_root / payload_name
            _require(_sha_file(payload) == manifest.get("payload_sha256"),
                     "burden count payload integrity failure")
            with zipfile.ZipFile(payload) as archive:
                _require(sum(item.file_size for item in archive.infolist()) <=
                         MAX_UNCOMPRESSED_BYTES, "burden counts exceed safety bound")
            with np.load(payload, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            _require(sorted(arrays) == manifest.get("array_keys"),
                     "burden count array inventory differs")
            used = set()
            value = _decode(manifest.get("state"), arrays, used)
            _require(used == set(arrays), "unreferenced burden count arrays")
            return value
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
                zipfile.BadZipFile) as error:
            raise ValueError(f"invalid burden count cache: {error}") from error

    def clear(self):
        """Remove only this update's generations after a completed checkpoint is durable."""
        for path in tuple(self.root.iterdir()):
            if path.name == "CURRENT" or path.name.startswith("update-progress-"):
                path.unlink()
        _fsync_directory(self.root)
        self.last_progress = None

    def save_budget(self):
        """Persist durable lower bounds before confirming the complete Adam checkpoint.

        Work lost before a safe point is intentionally not inferred. External
        elapsed/resource measurements remain necessary for a total cost.
        """
        progress = self.last_progress
        _require(isinstance(progress, dict) and
                 progress.get("next_recompute_index") ==
                 progress.get("total_recompute_tiles"),
                 "cannot finalize budget before every tile gradient is durable")
        receipt = {"schema": BUDGET_SCHEMA,
                   "training_identity_sha256": self.identity["training_identity_sha256"],
                   "update_identity_sha256": self.identity_sha256,
                   "update_index": self.identity["update_index"],
                   "progress": progress}
        _atomic_bytes(self.root / BUDGET_FILE, _canonical(receipt))
        return receipt


def load_completed_budgets(root, training_identity_sha256, completed_steps):
    """Read durable work lower bounds across process restarts."""
    root = Path(root)
    _require(type(completed_steps) is int and completed_steps >= 0,
             "nonnegative completed-step count required for budget ledger")
    if not root.exists():
        return []
    rows = {}
    for path in root.glob(f"update-*/{BUDGET_FILE}"):
        try:
            value = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid partial update budget: {error}") from error
        _require(value.get("schema") == BUDGET_SCHEMA and
                 value.get("training_identity_sha256") == training_identity_sha256,
                 "partial update budget identity differs")
        index = value.get("update_index"); progress = value.get("progress")
        _require(type(index) is int and 0 <= index < completed_steps and index not in rows and
                 isinstance(progress, dict) and
                 progress.get("next_recompute_index") == progress.get("total_recompute_tiles"),
                 "partial update budget index/progress differs")
        rows[index] = progress
    _require(set(rows) == set(range(completed_steps)),
             "partial update budget ledger omits a completed optimizer step")
    return [dict(rows[index], update_index=index) for index in range(completed_steps)]
