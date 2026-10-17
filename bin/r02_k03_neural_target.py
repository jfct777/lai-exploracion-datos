#!/usr/bin/env python3
"""Authenticated K03 neural targets and their unchanged weighted MSE.

The observed-marker denominator is validated and retained as provenance.  It
does not multiply the loss: K03 already defines the response as a mean over
jointly observed markers and supplies the sole pair weight separately.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import torch


FIELDS = {"person_id", "component_id", "role", "pairs", "target",
          "pair_weight", "joint_marker_count"}
HEX = re.compile(r"[0-9a-f]{64}")
LOSS_POLICY = "K03_COMPONENT_PAIR_WEIGHTED_MSE"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, digest):
    path = Path(path)
    require(HEX.fullmatch(digest or "") and path.is_file() and sha(path) == digest,
            "pair target JSON SHA256 mismatch")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate pair target JSON key")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"nonfinite JSON value {value}")))


def component_pair_weights(pairs, components):
    labels = [tuple(sorted((str(components[a]), str(components[b])))) for a, b in pairs]
    counts = Counter(labels)
    raw = np.asarray([1 / counts[label] for label in labels], dtype=np.float64)
    return raw / raw.mean()


def load_pair_target(entry, contract, stage, request):
    """Load the complete target axis independently from autosomal feature stores."""
    require(stage in {"FIT", "SELECT"} and set(entry) >=
            {"path", "sha256", "receipt_path", "receipt_sha256"},
            "explicit FIT/SELECT pair target descriptor required")
    path = Path(entry["path"])
    require(HEX.fullmatch(entry["sha256"] or "") and path.is_file()
            and sha(path) == entry["sha256"], "pair target SHA256 mismatch")
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) == len(FIELDS)
                and {item.filename for item in archive.infolist()} ==
                {field + ".npy" for field in FIELDS}
                and sum(item.file_size for item in archive.infolist()) <=
                contract["limits"]["max_batch_bytes"],
                "exact bounded K03 neural pair target fields required")
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    persons = data["person_id"].astype(str)
    components = data["component_id"].astype(str)
    roles = data["role"].astype(str)
    pairs = np.asarray(data["pairs"])
    target = np.asarray(data["target"])
    weights = np.asarray(data["pair_weight"])
    support = np.asarray(data["joint_marker_count"])
    require(tuple(persons) == tuple(request["person_ids"])
            and tuple(components) == tuple(request["component_ids"])
            and tuple(roles) == tuple(request["roles"]) and set(roles) == {stage},
            "pair target and autosomal person axes differ")
    require(pairs.ndim == 2 and pairs.shape[1] == 2
            and np.issubdtype(pairs.dtype, np.integer) and len(pairs) > 0
            and len(pairs) <= contract["limits"]["max_targets"]
            and np.all(pairs >= 0) and np.all(pairs < len(persons))
            and np.all(pairs[:, 0] < pairs[:, 1])
            and len(set(map(tuple, pairs.tolist()))) == len(pairs),
            "pair target indices differ")
    require(np.all(components[pairs[:, 0]] != components[pairs[:, 1]])
            and target.shape == weights.shape == support.shape == (len(pairs),)
            and np.isfinite(target).all() and np.isfinite(weights).all()
            and np.all(weights > 0) and np.issubdtype(support.dtype, np.integer)
            and np.all(support > 0), "pair target values/support differ")
    expected_weight = component_pair_weights(pairs, components)
    require(np.allclose(weights, expected_weight, rtol=1e-13, atol=0),
            "K03 component-pair weights differ")

    receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
    require(receipt.get("schema") == "r02_k03_neural_pair_target_v1"
            and receipt.get("status") == "TARGET_REPRESENTATION_COMPLETE_NOT_TRAINING_ADMISSION"
            and receipt.get("role") == stage and receipt.get("arrays_sha256") == entry["sha256"]
            and receipt.get("target_contract_sha256") == contract.get("target_contract_sha256")
            and receipt.get("block_id") == request["block_id"]
            and receipt.get("exclusion_scope") == request["exclusion_scope"]
            and receipt.get("n_people") == len(persons) and receipt.get("n_pairs") == len(pairs)
            and receipt.get("fields") == sorted(data)
            and receipt.get("evaluate_targets_computed") is False
            and receipt.get("training_allowed") is False,
            "K03 pair target receipt/task lineage differs")
    recorded = receipt.get("support", {})
    require(recorded.get("role_people") == len(persons)
            and recorded.get("emitted_between_component_pairs") == len(pairs)
            and recorded.get("minimum_joint_marker_count") == int(support.min())
            and recorded.get("maximum_joint_marker_count") == int(support.max())
            and recorded.get("target_missing_values") == 0
            and np.isclose(recorded.get("pair_weight_sum", np.nan), weights.sum(),
                           rtol=1e-13, atol=0),
            "K03 pair target support receipt differs")
    semantics = receipt.get("neural_consumer_semantics", {})
    require(semantics == {
        "used_by_current_loss": ["pairs", "target", "pair_weight"],
        "authenticated_but_not_used_by_current_loss": ["joint_marker_count"],
    }, "K03 weighted target consumer semantics differ")
    require(contract.get("pair_loss_policy") == LOSS_POLICY,
            "K03 weighted pair loss policy was not admitted")
    return data


def pair_squared_errors(prediction, data):
    """Per-pair squared errors for reporting; no weight or denominator applied."""
    target = torch.as_tensor(data["target"], dtype=prediction.dtype,
                             device=prediction.device)
    require(tuple(target.shape) == tuple(prediction.shape)
            and bool(torch.isfinite(target).all()), "pair prediction/target differs")
    return (prediction - target).square()


def weighted_pair_mse(prediction, data):
    """Exact K03 sum(w*error²)/sum(w), with the denominator used once only."""
    errors = pair_squared_errors(prediction, data)
    weights = torch.as_tensor(data["pair_weight"], dtype=prediction.dtype,
                              device=prediction.device)
    support = np.asarray(data["joint_marker_count"])
    require(tuple(weights.shape) == tuple(errors.shape)
            and bool(torch.isfinite(weights).all()) and bool((weights > 0).all())
            and support.shape == tuple(errors.shape)
            and np.issubdtype(support.dtype, np.integer) and np.all(support > 0),
            "K03 pair weights/support differ")
    return (weights * errors).sum() / weights.sum()


def structure_training_loss(prediction, data):
    """Preserve legacy mean loss unless the authenticated K03 fields are present."""
    if "pair_weight" not in data:
        return pair_squared_errors(prediction, data).mean()
    require(set(("pair_weight", "joint_marker_count")) <= set(data),
            "incomplete K03 weighted target fields")
    return weighted_pair_mse(prediction, data)
