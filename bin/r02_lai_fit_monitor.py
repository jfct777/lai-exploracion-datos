#!/usr/bin/env python3
"""Deterministic, read-only scoring of an explicitly pinned LAI FIT subset.

The monitor is deliberately independent from the training loop.  Callers supply
the data validation, prediction and additive-loss functions, which keeps this
module free of dataset imports and circular dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


SCHEMA = "r02_lai_fixed_fit_monitor_v1"
REQUIRED_STATISTICS = frozenset(("log_loss_sum", "observed_count"))
OPTIONAL_STATISTICS = frozenset(("brier_sum", "state_error_sum", "dosage_abs_sum"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def object_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def model_state_sha256(model):
    """Hash parameters and buffers without using pickle serialization."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        require(torch.is_tensor(value), "model state must contain tensors only")
        tensor = value.detach().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(canonical_bytes(list(tensor.shape)))
        # Flatten first because viewing a scalar tensor as bytes is otherwise
        # unsupported when its element size differs from uint8.
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _tensor_bytes(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {_clone_tree(key): _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported optimizer state value: {type(value).__name__}")


def _hash_tree(digest, value):
    if torch.is_tensor(value):
        digest.update(b"tensor\0" + str(value.dtype).encode("ascii") + b"\0")
        digest.update(canonical_bytes(list(value.shape)))
        digest.update(_tensor_bytes(value))
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_tree(digest, key)
            _hash_tree(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update((b"list\0" if isinstance(value, list) else b"tuple\0"))
        for item in value:
            _hash_tree(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(canonical_bytes(value))
    else:
        raise TypeError(f"unsupported optimizer state value: {type(value).__name__}")


def optimizer_state_sha256(optimizer):
    require(isinstance(optimizer, torch.optim.Optimizer), "Adam optimizer is required")
    digest = hashlib.sha256()
    _hash_tree(digest, optimizer.state_dict())
    return digest.hexdigest()


def optimizer_step_counters(optimizer):
    values = []
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        step = state["step"]
        value = float(step.detach().cpu()) if torch.is_tensor(step) else float(step)
        require(math.isfinite(value) and value >= 0 and value.is_integer(),
                "optimizer step counter is invalid")
        values.append(int(value))
    require(values, "optimizer has no Adam step counters")
    return sorted(set(values))


def training_state_sha256(model):
    """Hash modes, gradients and RNG state for a pre/post noncontamination audit."""
    digest = hashlib.sha256()
    digest.update(model_state_sha256(model).encode("ascii"))
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8") + b"\0")
        if parameter.grad is None:
            digest.update(b"NO_GRAD\0")
        else:
            digest.update(str(parameter.grad.dtype).encode("ascii") + b"\0")
            digest.update(canonical_bytes(list(parameter.grad.shape)))
            digest.update(_tensor_bytes(parameter.grad))
    digest.update(canonical_bytes([module.training for module in model.modules()]))
    digest.update(repr(random.getstate()).encode("ascii"))
    numpy_rng = np.random.get_state()
    digest.update(str(numpy_rng[0]).encode("ascii") + b"\0")
    digest.update(np.asarray(numpy_rng[1]).tobytes())
    digest.update(canonical_bytes([int(numpy_rng[2]), int(numpy_rng[3]), float(numpy_rng[4])]))
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def _atomic_idempotent_json(path, value):
    path = Path(path)
    require(path.parent.is_dir(), "monitor output parent must already exist")
    content = canonical_bytes(value)
    if path.exists():
        require(path.is_file() and not path.is_symlink(), "monitor output must be a regular file")
        require(path.read_bytes() == content,
                "monitor output already exists with different content")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _clone_model_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _state_equal(model, saved):
    current = model.state_dict()
    return (current.keys() == saved.keys()
            and all(torch.equal(current[name].detach(), value) for name, value in saved.items()))


def _restore_model_state(model, saved):
    if not _state_equal(model, saved):
        model.load_state_dict(saved, strict=True)


def _clone_gradients(model):
    return [None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()]


def _restore_gradients(model, saved):
    parameters = list(model.parameters())
    require(len(parameters) == len(saved), "model parameter inventory changed during FIT monitoring")
    for parameter, gradient in zip(parameters, saved):
        if gradient is None:
            parameter.grad = None
        elif parameter.grad is None:
            parameter.grad = gradient.clone()
        else:
            parameter.grad.detach().copy_(gradient)


def _snapshot_execution(model, optimizer):
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": cuda_rng,
        "modes": [(module, bool(module.training)) for module in model.modules()],
        "model_state": _clone_model_state(model),
        "gradients": _clone_gradients(model),
        "optimizer_state": _clone_tree(optimizer.state_dict()),
    }


def _gradients_equal(model, saved):
    current = [parameter.grad for parameter in model.parameters()]
    return len(current) == len(saved) and all(
        (left is None and right is None) or
        (left is not None and right is not None and torch.equal(left.detach(), right))
        for left, right in zip(current, saved))


def _restore_execution(model, optimizer, saved):
    _restore_model_state(model, saved["model_state"])
    _restore_gradients(model, saved["gradients"])
    if optimizer_state_sha256(optimizer) != optimizer_state_sha256_from_value(saved["optimizer_state"]):
        optimizer.load_state_dict(saved["optimizer_state"])
    for module, training in saved["modes"]:
        module.training = training
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def optimizer_state_sha256_from_value(value):
    digest = hashlib.sha256()
    _hash_tree(digest, value)
    return digest.hexdigest()


def _validate_pinned_fit_entries(contract, fit_entries):
    require(isinstance(contract, dict) and isinstance(contract.get("batches"), dict),
            "contract batches are required")
    canonical_fit = contract["batches"].get("FIT")
    select = contract["batches"].get("SELECT", [])
    require(isinstance(canonical_fit, list) and canonical_fit, "contract has no FIT entries")
    require(isinstance(select, list), "contract SELECT entries must be a list")
    entries = list(fit_entries)
    require(entries and all(isinstance(entry, dict) for entry in entries),
            "explicit FIT entries are required")
    fit_by_hash = {object_sha256(entry): entry for entry in canonical_fit}
    select_hashes = {object_sha256(entry) for entry in select}
    hashes = [object_sha256(entry) for entry in entries]
    require(len(hashes) == len(set(hashes)), "duplicate FIT monitor entry")
    require(all(value in fit_by_hash for value in hashes),
            "monitor entry is not an exact contract FIT entry")
    require(not set(hashes).intersection(select_hashes), "SELECT entry cannot enter FIT monitor")
    require(all(entry.get("stage", "FIT") == "FIT" for entry in entries),
            "monitor entries must be explicitly FIT")
    return entries, hashes


def _validate_statistics(value):
    require(isinstance(value, dict), "loss callback must return additive statistics")
    keys = set(value)
    require(REQUIRED_STATISTICS <= keys and keys <= REQUIRED_STATISTICS | OPTIONAL_STATISTICS,
            "loss callback statistic names differ")
    count = value["observed_count"]
    require(type(count) is int and count > 0, "observed_count must be a positive integer")
    result = {"observed_count": count}
    for name in ("log_loss_sum", "brier_sum", "state_error_sum"):
        if name in value:
            number = float(value[name])
            require(math.isfinite(number) and number >= 0, f"invalid {name}")
            result[name] = number
    if "dosage_abs_sum" in value:
        dosage = [float(item) for item in value["dosage_abs_sum"]]
        require(len(dosage) == 3 and all(math.isfinite(item) and item >= 0 for item in dosage),
                "dosage_abs_sum must contain three finite nonnegative sums")
        result["dosage_abs_sum"] = dosage
    return result


def score_fixed_fit(*, model, optimizer, contract, resolved, fit_entries, arm, people_per_batch,
                    step, output, validate_entry, predict_entry, loss_statistics):
    """Score one immutable FIT subset while preserving all training process state.

    ``validate_entry(entry, contract, resolved, people_per_batch)`` returns an
    opaque validated value. ``predict_entry(model, validated, entry, contract,
    resolved, arm, people_per_batch)`` returns an opaque prediction value.
    ``loss_statistics(prediction, validated, entry)`` returns additive sums.
    """
    require(isinstance(model, torch.nn.Module), "TaskModel must be a torch module")
    require(isinstance(optimizer, torch.optim.Optimizer), "Adam optimizer is required")
    require(isinstance(resolved, dict), "resolved configuration must be a mapping")
    require(isinstance(arm, str) and arm, "arm is required")
    require(type(people_per_batch) is int and people_per_batch > 0,
            "people_per_batch must be a positive integer")
    require(type(step) is int and step >= 0, "step must be a nonnegative integer")
    require(all(callable(item) for item in (validate_entry, predict_entry, loss_statistics)),
            "validation, prediction and loss callbacks are required")
    entries, entry_hashes = _validate_pinned_fit_entries(contract, fit_entries)
    before_sha = model_state_sha256(model)
    training_before_sha = training_state_sha256(model)
    optimizer_before_sha = optimizer_state_sha256(optimizer)
    optimizer_steps_before = optimizer_step_counters(optimizer)
    saved = _snapshot_execution(model, optimizer)
    totals = None
    model_mutated = gradients_mutated = optimizer_mutated = False
    try:
        model.eval()
        with torch.no_grad():
            for entry in entries:
                validated = validate_entry(entry, contract, resolved, people_per_batch)
                prediction = predict_entry(model, validated, entry, contract, resolved,
                                           arm, people_per_batch)
                statistics = _validate_statistics(loss_statistics(prediction, validated, entry))
                if totals is None:
                    totals = {key: ([0.0, 0.0, 0.0] if key == "dosage_abs_sum" else 0)
                              for key in statistics}
                require(set(statistics) == set(totals),
                        "loss callback returned inconsistent metrics across FIT entries")
                for key, value in statistics.items():
                    if key == "dosage_abs_sum":
                        totals[key] = [left + right for left, right in zip(totals[key], value)]
                    else:
                        totals[key] += value
    finally:
        model_mutated = not _state_equal(model, saved["model_state"])
        gradients_mutated = not _gradients_equal(model, saved["gradients"])
        optimizer_mutated = optimizer_state_sha256(optimizer) != optimizer_before_sha
        _restore_execution(model, optimizer, saved)
    after_sha = model_state_sha256(model)
    training_after_sha = training_state_sha256(model)
    optimizer_after_sha = optimizer_state_sha256(optimizer)
    optimizer_steps_after = optimizer_step_counters(optimizer)
    require(before_sha == after_sha, "model weights or buffers changed during FIT monitoring")
    require(training_before_sha == training_after_sha,
            "gradient, mode, or RNG state changed during FIT monitoring")
    require(optimizer_before_sha == optimizer_after_sha
            and optimizer_steps_before == optimizer_steps_after,
            "Adam state changed during FIT monitoring")
    require(not model_mutated and not gradients_mutated and not optimizer_mutated,
            "monitor callback attempted to mutate model, gradients, or Adam state")
    count = totals["observed_count"]
    metrics = {"mean_log_loss": totals["log_loss_sum"] / count,
               "observed_count": count}
    if "brier_sum" in totals:
        metrics["mean_brier"] = totals["brier_sum"] / count
    if "state_error_sum" in totals:
        metrics["state_error"] = totals["state_error_sum"] / count
    if "dosage_abs_sum" in totals:
        metrics["dosage_mae"] = [value / count for value in totals["dosage_abs_sum"]]
    receipt = {
        "schema": SCHEMA,
        "status": "FIXED_FIT_DESCRIPTIVE_MONITOR_ONLY",
        "step": step,
        "arm": arm,
        "people_per_batch": people_per_batch,
        "contract_sha256": object_sha256(contract),
        "resolved_sha256": object_sha256(resolved),
        "fit_entry_sha256": entry_hashes,
        "fit_entries": len(entries),
        "model_state_sha256_before": before_sha,
        "model_state_sha256_after": after_sha,
        "training_state_sha256_before": training_before_sha,
        "training_state_sha256_after": training_after_sha,
        "optimizer_state_sha256_before": optimizer_before_sha,
        "optimizer_state_sha256_after": optimizer_after_sha,
        "optimizer_step_counters_before": optimizer_steps_before,
        "optimizer_step_counters_after": optimizer_steps_after,
        "training_state_preserved": True,
        "select_accessed": False,
        "metrics": metrics,
        "interpretation_limit": "FIXED_FIT_NUMERICAL_TRAJECTORY_NOT_SELECTION_OR_BIOLOGICAL_EVALUATION",
    }
    _atomic_idempotent_json(output, receipt)
    return receipt
