"""Authenticate FI96 model weights for a new LAI training trajectory.

This is weight initialization, never checkpoint continuation.  The historical
Adam state and random-number states in the FI checkpoint are deliberately not
decoded.  Ordinary continuation remains the responsibility of CheckpointStore.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from r02_context_checkpoint import _decode
from r02_lai_common_context import LOCAL_MEAN_V1
from r02_lai_query_event_attention import (
    LAI_QUERY_EVENT_MODES, initialize_from_mean_model,
)


SCHEMA = "r02_lai_fi96_weight_initialization_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _read_descriptor(descriptor, label):
    _require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"},
             f"{label} descriptor fields differ")
    _require(_sha(descriptor["path"]) == descriptor["sha256"],
             f"{label} SHA256 differs")
    return json.loads(Path(descriptor["path"]).read_text())


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalized_recipe(recipe):
    value = dict(recipe)
    value["lai_query_event_mode"] = value.get("lai_query_event_mode", LOCAL_MEAN_V1)
    return value


def _authenticated_target_personal_dose_control(source_contract, target_contract):
    """Authenticate the one admitted extra target block, without widening drift."""
    block = "lai_personal_dose_control"
    _require(block not in source_contract,
             "source FI96 contract must not contain a personal-dose intervention")
    if block not in target_contract:
        return None
    from r02_lai_dose_control_training import (
        BLOCK, checkpoint_binding_identity, validate_binding_spec,
    )

    _require(BLOCK == block, "personal-dose block name differs")
    validate_binding_spec(target_contract)
    binding = checkpoint_binding_identity(target_contract)
    _require(binding is not None,
             "target personal-dose intervention identity is absent")
    return binding


def initialize_from_fi96(model, request_path, request_sha256, *,
                         target_contract_sha256, target_contract, target_resolved,
                         apply_weights=True):
    """Authenticate a FU position request and optionally import only its model.

    ``apply_weights=False`` is used when the destination already has a durable
    ``CURRENT`` checkpoint.  It still authenticates the source identity, but
    leaves the restored destination model as the sole source of parameters.
    """
    _require(_sha(request_path) == request_sha256,
             "weight-initialization request SHA256 differs")
    request = json.loads(Path(request_path).read_text())
    _require(not ({"optimizer", "optimizer_state", "adam", "adam_state"} & set(request)),
             "weight initialization must not request historical Adam state")
    required = {"case", "contract", "resolved", "checkpoint", "checkpoint_payload"}
    _require(required <= set(request), "weight-initialization request fields differ")

    case = _read_descriptor(request["case"], "source case")
    source_contract = _read_descriptor(request["contract"], "source contract")
    source_resolved = _read_descriptor(request["resolved"], "source resolved recipe")
    metadata = _read_descriptor(request["checkpoint"], "source checkpoint metadata")
    receipt = _read_descriptor(case["training_receipt"], "source training receipt")

    contract_hash = request["contract"]["sha256"]
    resolved_hash = request["resolved"]["sha256"]
    initialization_contract = target_contract.get("lai_weight_initialization")
    expected_initialization_contract = {
        "schema": "r02_lai_fi96_weight_initialization_contract_v1",
        "source_request": {"path": str(Path(request_path).resolve()),
                           "sha256": request_sha256},
        "source_step": 96, "source_role": "FIT", "source_arm": "NONE",
        "optimizer_state_imported": False,
    }
    source_contract_comparable = dict(source_contract)
    target_contract_comparable = dict(target_contract)
    target_dose_binding = _authenticated_target_personal_dose_control(
        source_contract, target_contract)
    target_contract_comparable.pop("lai_weight_initialization", None)
    target_contract_comparable.pop("query_event_resolved_amendment", None)
    if target_dose_binding is not None:
        target_contract_comparable.pop("lai_personal_dose_control")
    _require(initialization_contract == expected_initialization_contract
             and target_contract_sha256 != contract_hash
             and case.get("contract_sha256") == contract_hash
             and source_contract.get("objective") == "lai",
             "source/target weight-initialization contract identity differs")
    _require(_canonical(source_contract_comparable) ==
             _canonical(target_contract_comparable),
             "target contract differs from FI96 beyond admitted initialization amendments")
    _require(receipt.get("schema") == "r02_context_training_v1"
             and receipt.get("contract_sha256") == contract_hash
             and receipt.get("resolved_sha256") == resolved_hash
             and receipt.get("score_role") == "FIT"
             and receipt.get("select_accessed") is False
             and receipt.get("arm") == "NONE"
             and receipt.get("steps") == 96
             and receipt.get("internal_checkpoint", {}).get("final_confirmed_step") == 96,
             "source receipt is not the authenticated FI96 FIT model")
    identity = metadata.get("identity", {})
    _require(identity.get("contract_sha256") == contract_hash
             and identity.get("resolved_sha256") == resolved_hash
             and identity.get("objective") == "lai"
             and identity.get("arm") == "NONE"
             and metadata.get("next_step") == 96,
             "source checkpoint identity or step differs")
    payload = request["checkpoint_payload"]
    _require(isinstance(payload, dict) and set(payload) == {"path", "sha256"}
             and _sha(payload["path"]) == payload["sha256"]
             and metadata.get("payload_sha256") == payload["sha256"],
             "source checkpoint payload differs")

    source_recipe = _normalized_recipe(source_resolved)
    target_recipe = _normalized_recipe(target_resolved)
    source_mode = source_recipe.pop("lai_query_event_mode")
    target_mode = target_recipe.pop("lai_query_event_mode")
    _require(source_mode == LOCAL_MEAN_V1 and target_mode in LAI_QUERY_EVENT_MODES
             and _canonical(source_recipe) == _canonical(target_recipe),
             "target recipe must differ from FI96 only by lai_query_event_mode")

    migration = None
    if apply_weights:
        with np.load(payload["path"], allow_pickle=False) as arrays:
            state = _decode(metadata["state"]["model"], arrays, set())
        if target_mode == LOCAL_MEAN_V1:
            model.load_state_dict(state, strict=True)
            migration = {"added_state_keys": [], "optimizer_state_imported": False,
                         "semantics": "STRICT_WEIGHT_INITIALIZATION_NOT_TRAINING_RESUME"}
        else:
            _require(getattr(model, "head", None) is None,
                     "attention initialization expects the LAI encoder-only TaskModel")
            encoder_state = {key.removeprefix("encoder."): value
                             for key, value in state.items() if key.startswith("encoder.")}
            _require(len(encoder_state) == len(state),
                     "FI96 state contains unexpected non-encoder tensors")
            migration = initialize_from_mean_model(model.encoder, encoder_state)

    source = {
        "schema": SCHEMA,
        "request": {"path": str(Path(request_path).resolve()), "sha256": request_sha256},
        "contract": request["contract"], "resolved": request["resolved"],
        "checkpoint": request["checkpoint"], "checkpoint_payload": payload,
        "training_receipt": case["training_receipt"],
        "source_step": 96, "source_mode": source_mode, "target_mode": target_mode,
        "source_role": "FIT", "source_arm": "NONE",
        "optimizer_state_imported": False,
        "application_policy": "APPLY_ONLY_WHEN_DESTINATION_CURRENT_IS_ABSENT",
    }
    if target_dose_binding is not None:
        source["target_personal_dose_control"] = target_dose_binding
    source["identity_sha256"] = hashlib.sha256(_canonical(source).encode()).hexdigest()
    source["weights_applied_this_process"] = bool(apply_weights)
    source["migration"] = migration
    return source


def require_fresh_optimizer(optimizer, initialization):
    _require(initialization.get("optimizer_state_imported") is False,
             "weight initialization cannot import Adam")
    _require(not optimizer.state,
             "Adam must be new before destination checkpoint restore")
    return True
