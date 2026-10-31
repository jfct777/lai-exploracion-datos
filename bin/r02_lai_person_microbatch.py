#!/usr/bin/env python3
"""Bound LAI autograd memory by processing consecutive people separately.

The scientific unit remains one authenticated regional batch.  This module
only changes when each person's independent graph is materialized and released;
it does not divide common/event/query positions within a person.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


AMENDMENT_SCHEMA = "r02_lai_person_microbatch_execution_v1"
CHECKPOINT_BINDING_SCHEMA = "r02_lai_person_microbatch_checkpoint_binding_v1"
CHECKPOINT_BINDING_NAME = "LAI_PERSON_MICROBATCH_EXECUTION.json"

# These are the concrete modules used by the two admitted LAI families.  A new
# module must be reviewed before it can enter a person-separated update because
# an operation across the first tensor axis would change the calculation.
_INDEPENDENT_MODULES = {
    ("r02_context_train", "TaskModel"),
    ("bin.r02_context_train", "TaskModel"),
    ("__main__", "TaskModel"),
    ("genomic_context_runtime", "ContextualModel"),
    ("genomic_context_runtime", "LocalAttention"),
    ("genomic_context_runtime", "MultiscaleQueryConvolution"),
    ("r02_lai_common_context", "EventQueryCommonAttention"),
    ("torch.nn.modules.container", "Sequential"),
    ("torch.nn.modules.container", "ModuleList"),
    ("torch.nn.modules.linear", "Linear"),
    ("torch.nn.modules.activation", "GELU"),
    ("torch.nn.modules.dropout", "Dropout"),
    ("torch.nn.modules.conv", "Conv1d"),
    ("torch.nn.modules.normalization", "LayerNorm"),
    ("torch.nn.modules.activation", "MultiheadAttention"),
    ("torch.nn.modules.linear", "NonDynamicallyQuantizableLinear"),
}

_REQUIRED_PERSON_FIELDS = {
    "person_id", "component_id", "common_features", "common_cm",
    "common_chrom", "common_valid", "common_core", "event_metadata",
    "event_dosage", "event_cm", "event_chrom", "event_valid", "event_core",
    "event_reference_features", "query_features", "query_cm", "query_chrom",
    "query_valid", "query_core", "baseline_probabilities", "query_burden",
    "truth_state",
}

_OPTIONAL_PERSON_FIELDS = {
    "role", "common_dosage", "common_pos_bp", "common_padding",
    "event_pos_bp", "event_padding", "query_pos_bp", "truth_observed",
    # This is joined from the truth overlay with the same person and query
    # indices as truth_state/truth_observed.  It is not a model input or part of
    # the first-update loss, but it must follow the corresponding person rows.
    "query_score_weight_cm",
}

# Arrays below describe shared site axes or their vocabularies.  They are
# deliberately retained whole for every person microbatch.  Naming every real
# field and checking its axis prevents an unknown person array from being kept
# merely because its first dimension happens not to equal this batch's N.
_COMMON_SITE_FIELDS = {
    "common_alt", "common_global_index", "common_is_alt", "common_ref",
}
_EVENT_SITE_FIELDS = {
    "event_alt", "event_annotation_is_padding", "event_copyable_in_case_fold",
    "event_copyable_known", "event_counted_allele", "event_global_index",
    "event_is_alt", "event_origin", "event_phase_status", "event_quality_status",
    "event_ref", "event_source_catalogue_sha256",
}
_EVENT_SITE_ANCESTRY_FIELDS = {
    "event_reference_ac", "event_reference_an", "event_reference_known",
}
_EVENT_SITE_ROLE_FIELDS = {
    "event_call_provenance_counts", "event_call_provenance_known",
}
_QUERY_SITE_FIELDS = {"query_global_index"}
_SCHEMA_METADATA_FIELDS = {
    "event_annotation_schema", "event_call_provenance_count_order",
    "event_call_provenance_role_order", "event_reference_ancestry_order",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def validate_person_separation(model, objective, people_per_microbatch):
    """Reject configurations whose output may depend on another person row."""
    _require(objective == "lai", "person microbatching is admitted for LAI only")
    _require(type(people_per_microbatch) is int and people_per_microbatch > 0,
             "LAI people per microbatch must be a positive integer")
    config = getattr(getattr(model, "encoder", None), "config", None)
    _require(config is not None and config.family in {"lai_cnn", "lai_local_attention"},
             "person microbatching requires an admitted LAI architecture")
    _require(config.dropout == 0, "person microbatching requires dropout=0")
    for module in model.modules():
        _require(not isinstance(module, nn.modules.batchnorm._BatchNorm),
                 "person microbatching rejects BatchNorm across people")
        key = (type(module).__module__, type(module).__name__)
        _require(key in _INDEPENDENT_MODULES,
                 "person microbatching rejects an unreviewed person-coupling module: "
                 + ".".join(key))
        if isinstance(module, nn.Dropout):
            _require(module.p == 0, "person microbatching requires every Dropout p=0")
        if isinstance(module, nn.MultiheadAttention):
            _require(module.dropout == 0,
                     "person microbatching requires attention dropout=0")
    return {
        "family": config.family,
        "dropout": float(config.dropout),
        "batchnorm_modules": 0,
        "module_types_reviewed": sorted(".".join(value) for value in
                                        { (type(m).__module__, type(m).__name__)
                                          for m in model.modules() }),
        "person_coupling_review": (
            "STATIC_ALLOWLIST_OF_CURRENT_MODULE_TYPES_WITH_NO_REVIEWED_"
            "BATCH_AXIS_REDUCTION;SOURCE_HASHES_REQUIRED_IN_EXECUTION_IDENTITY"),
    }


def slice_people(data, start, end):
    """Return views for consecutive people while retaining complete site axes."""
    _require(isinstance(data, dict) and "person_id" in data,
             "LAI batch requires a person axis")
    people = len(data["person_id"])
    _require(0 <= start < end <= people, "invalid LAI person microbatch")
    _require(_REQUIRED_PERSON_FIELDS <= set(data),
             "LAI person microbatch lacks required arrays")
    fields = _REQUIRED_PERSON_FIELDS | (_OPTIONAL_PERSON_FIELDS & set(data))
    shared = (_COMMON_SITE_FIELDS | _EVENT_SITE_FIELDS |
              _EVENT_SITE_ANCESTRY_FIELDS | _EVENT_SITE_ROLE_FIELDS |
              _QUERY_SITE_FIELDS | _SCHEMA_METADATA_FIELDS)
    _require(set(data) <= fields | shared,
             "unclassified array(s): " + ",".join(sorted(set(data) - fields - shared)))

    common_sites = np.asarray(data["common_valid"]).shape[1]
    event_sites = np.asarray(data["event_valid"]).shape[1]
    query_shape = np.asarray(data["query_valid"]).shape
    _require(query_shape[0] == people, "query_valid person axis differs")
    for name in _COMMON_SITE_FIELDS & set(data):
        _require(np.asarray(data[name]).shape == (common_sites,),
                 f"shared COMMON-site axis differs: {name}")
    for name in _EVENT_SITE_FIELDS & set(data):
        _require(np.asarray(data[name]).shape == (event_sites,),
                 f"shared EVENT-site axis differs: {name}")
    ancestry = np.asarray(data.get("event_reference_ancestry_order", [])).shape
    roles = np.asarray(data.get("event_call_provenance_role_order", [])).shape
    counts = np.asarray(data.get("event_call_provenance_count_order", [])).shape
    for name in _EVENT_SITE_ANCESTRY_FIELDS & set(data):
        _require(len(ancestry) == 1 and np.asarray(data[name]).shape ==
                 (event_sites, ancestry[0]), f"EVENT-by-ancestry axis differs: {name}")
    for name in _EVENT_SITE_ROLE_FIELDS & set(data):
        _require(len(roles) == 1 and
                 (name != "event_call_provenance_counts" or len(counts) == 1),
                 f"EVENT metadata vocabulary differs: {name}")
        tail = (roles[0], counts[0]) if name.endswith("counts") else (roles[0],)
        _require(np.asarray(data[name]).shape == (event_sites, *tail),
                 f"EVENT-by-role metadata axis differs: {name}")
    for name in _QUERY_SITE_FIELDS & set(data):
        _require(np.asarray(data[name]).shape == (query_shape[1],),
                 f"shared query-site axis differs: {name}")
    if "event_annotation_schema" in data:
        _require(np.asarray(data["event_annotation_schema"]).shape == (),
                 "event annotation schema must be scalar")
    for name in (_SCHEMA_METADATA_FIELDS - {"event_annotation_schema"}) & set(data):
        _require(np.asarray(data[name]).ndim == 1,
                 f"schema vocabulary must be one-dimensional: {name}")
    result = dict(data)
    for name in fields:
        value = np.asarray(data[name])
        _require(value.ndim >= 1 and value.shape[0] == people,
                 f"LAI person-axis array differs: {name}")
        if name.startswith("query_") or name.startswith("truth_") or name == "baseline_probabilities":
            _require(value.ndim >= 2 and value.shape[:2] == query_shape,
                     f"LAI person-by-query axes differ: {name}")
        result[name] = value[start:end]
    return result


@dataclass(frozen=True)
class PersonMicrobatchBackward:
    loss: torch.Tensor
    prediction: torch.Tensor
    persons: int
    microbatches: int


def backward_person_microbatches(model, data, arm, people_per_microbatch,
                                 person_losses):
    """Accumulate \(sum_i loss_i/N\); caller owns zero_grad and Adam step."""
    validate_person_separation(model, "lai", people_per_microbatch)
    people = len(data["person_id"])
    _require(people > 0, "empty LAI person axis")
    predictions = []
    total = torch.zeros((), dtype=torch.float64)
    microbatches = 0
    for start in range(0, people, people_per_microbatch):
        micro = slice_people(data, start, min(start + people_per_microbatch, people))
        prediction = model(micro, arm)
        values = person_losses(prediction, micro)
        _require(values.shape == (len(micro["person_id"]),)
                 and bool(torch.isfinite(values).all()),
                 "LAI microbatch must produce one finite loss per person")
        (values.sum() / people).backward()
        total += values.detach().double().sum()
        predictions.append(prediction.detach())
        microbatches += 1
        del values, prediction, micro
    return PersonMicrobatchBackward(
        loss=(total / people), prediction=torch.cat(predictions, 0),
        persons=people, microbatches=microbatches)


@torch.inference_mode()
def predict_person_microbatches(model, data, arm, people_per_microbatch):
    """Run inference with the same per-person partition and original row order."""
    validate_person_separation(model, "lai", people_per_microbatch)
    people = len(data["person_id"])
    _require(people > 0, "empty LAI person axis")
    predictions = []
    for start in range(0, people, people_per_microbatch):
        micro = slice_people(data, start, min(start + people_per_microbatch, people))
        prediction = model(micro, arm)
        valid = torch.from_numpy(np.asarray(micro["query_valid"]))
        _require(prediction.shape[:2] == valid.shape
                 and bool(torch.isfinite(prediction[valid]).all()),
                 "nonfinite valid-query or reordered LAI microbatch prediction")
        predictions.append(prediction.detach())
    return torch.cat(predictions, 0)


def amendment_identity(model, people_per_microbatch, code_sha256):
    """Describe technical execution separately from scientific checkpoint state."""
    audit = validate_person_separation(model, "lai", people_per_microbatch)
    value = {
        "schema": AMENDMENT_SCHEMA,
        "people_per_microbatch": people_per_microbatch,
        "loss": "MEAN_PERSON_OF_MEAN_OWNED_OBSERVED_QUERY_NEGATIVE_LOG_PROBABILITY",
        "gradient_accumulation": "SUM_PERSON_LOSSES_DIVIDED_BY_COMPLETE_BATCH_PERSON_COUNT",
        "optimizer_boundary": "ONE_ZERO_GRAD_AND_ONE_ADAM_STEP_PER_AUTHENTICATED_REGION",
        "gradient_clipping": "ONCE_AFTER_LAST_PERSON_MICROBATCH_IF_CONFIGURED",
        "site_partition": "NONE_COMPLETE_COMMON_EVENT_QUERY_CONTEXT_PER_PERSON",
        "person_order": "AUTHENTICATED_ORIGINAL_ORDER",
        "architecture_audit": audit,
        "reviewed_code_sha256": code_sha256,
        "scientific_checkpoint_identity_changed_by_people_per_microbatch": False,
    }
    return {**value, "sha256": hashlib.sha256(_canonical(value)).hexdigest()}


def bind_checkpoint_amendment(checkpoint_root, scientific_identity_sha256,
                              amendment):
    """Durably bind a checkpoint root before restore or the first Adam step."""
    _require(isinstance(scientific_identity_sha256, str)
             and len(scientific_identity_sha256) == 64,
             "scientific checkpoint identity digest differs")
    _require(isinstance(amendment, dict)
             and amendment.get("schema") == AMENDMENT_SCHEMA
             and isinstance(amendment.get("sha256"), str)
             and len(amendment["sha256"]) == 64,
             "LAI person-microbatch amendment identity differs")
    root = Path(checkpoint_root)
    _require(root.is_dir() and not root.is_symlink(),
             "checkpoint amendment root must be a real directory")
    expected = {
        "schema": CHECKPOINT_BINDING_SCHEMA,
        "scientific_checkpoint_identity_sha256": scientific_identity_sha256,
        "technical_amendment_sha256": amendment["sha256"],
        "people_per_microbatch": amendment["people_per_microbatch"],
        "reviewed_code_sha256": amendment["reviewed_code_sha256"],
        "optimizer_boundary": amendment["optimizer_boundary"],
    }
    content = _canonical(expected)
    path = root / CHECKPOINT_BINDING_NAME
    superseded = None
    if path.exists():
        try:
            actual = json.loads(path.read_bytes(),
                object_pairs_hook=lambda pairs: _unique_json_pairs(pairs))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid LAI checkpoint amendment binding: {error}") from error
        if actual != expected:
            # A pre-Adam failure can leave only this technical binding and the
            # checkpoint lock.  Preserve that receipt, but never reinterpret or
            # replace a binding after a scientific CURRENT generation exists.
            allowed_prefix = CHECKPOINT_BINDING_NAME + ".superseded-"
            unexpected = [item.name for item in root.iterdir()
                          if item.name not in {".LOCK", CHECKPOINT_BINDING_NAME}
                          and not item.name.startswith(allowed_prefix)]
            _require(not (root / "CURRENT").exists() and not unexpected,
                     "LAI checkpoint technical amendment differs after checkpoint state exists")
            old_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            archive = root / (allowed_prefix + old_sha + ".json")
            if archive.exists():
                _require(archive.read_bytes() == path.read_bytes(),
                         "archived pre-Adam technical binding differs")
                path.unlink()
            else:
                os.replace(path, archive)
            superseded = {"path": str(archive.resolve()), "sha256": old_sha,
                          "reason": "PRE_ADAM_BINDING_WITHOUT_CURRENT"}
    if not path.exists():
        _require(not (root / "CURRENT").exists(),
                 "existing checkpoint lacks reviewed LAI technical amendment binding")
        try:
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except FileExistsError:
            _require(path.read_bytes() == content,
                     "concurrent LAI checkpoint amendment binding differs")
    return {"path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "superseded_pre_adam_binding": superseded,
            **expected}


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate checkpoint amendment JSON key")
        result[key] = value
    return result
