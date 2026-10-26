#!/usr/bin/env python3
"""Materialize one authenticated regional LAI predictor view for PREPARE.

This module transports geometry already fixed by ``RegionalReader``.  It does
not join targets or controls, train, select, score, or infer people, recipes,
radii, kernels, or regions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r02_lai_neural_channel_adapter as adapter
from r02_lai_regional_inputs import RegionalReader

REQUEST_SCHEMA = "r02_lai_regional_predictor_request_v1"
RECEIPT_SCHEMA = "r02_lai_regional_predictor_receipt_v1"
STATUS = "REGIONAL_PREDICTORS_PREPARED_NO_TARGETS_CONTROLS_TRAINING_SELECTION_OR_EVALUATION"
REFERENCE_HASHES = adapter.REFERENCE_HASHES
FALSE_FIELDS = ("training_allowed", "controls_materialized", "truth_loaded",
                "selection_performed", "evaluate_opened")
REQUEST_FIELDS = {
    "schema", "operation", "layout", "task_id", "case_id", "direction",
    "role", "chromosome", "family", "scale_cm", "region_id",
    "person_indices", "expected_person_ids", "expected_component_ids",
    "expected_layout_recipe", "resolved_model_config", "reference_identity", *FALSE_FIELDS,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array, np):
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode() + b"\0")
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode() + b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_request(path, expected_sha256):
    require(sha256(path) == expected_sha256, "request hash differs")
    request = json.loads(Path(path).read_text())
    require(set(request) == REQUEST_FIELDS, "request fields differ")
    require(request["schema"] == REQUEST_SCHEMA and request["operation"] == "PREPARE",
            "only PREPARE is accepted")
    require(all(request[name] is False for name in FALSE_FIELDS),
            "truth/controls/training/selection/evaluation must remain false")
    descriptor = request["layout"]
    require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"}
            and re.fullmatch(r"[0-9a-f]{64}", descriptor.get("sha256", "")),
            "layout descriptor differs")
    people = request["person_indices"]
    require(isinstance(people, list) and people and all(type(x) is int for x in people)
            and len(set(people)) == len(people), "person indices differ")
    require(isinstance(request["expected_person_ids"], list)
            and len(request["expected_person_ids"]) == len(people)
            and isinstance(request["expected_component_ids"], list)
            and len(request["expected_component_ids"]) == len(people),
            "expected person/component axes differ")
    reference = request["reference_identity"]
    require(isinstance(reference, dict) and set(reference) == set(REFERENCE_HASHES)
            and all(re.fullmatch(r"[0-9a-f]{64}", reference.get(name, ""))
                    for name in REFERENCE_HASHES), "reference identity differs")
    return request


def _authenticate_layout(request):
    descriptor = request["layout"]
    reader = RegionalReader(descriptor["path"], descriptor["sha256"], authenticate=True)
    layout = reader.layout
    require(layout.get("status") == "REGIONAL_GEOMETRY_COMPLETE_NO_TRAINING_NO_SELECTION"
            and layout.get("truth_arrays_loaded") is False
            and layout.get("training_authorized") is False
            and layout.get("selection_performed") is False
            and layout.get("evaluate_opened") is False,
            "regional layout is not PREPARE-only")
    for key in ("task_id", "case_id", "direction", "role"):
        require(layout.get(key) == request[key], "layout identity differs: " + key)
    require(str(layout.get("chromosome")) == str(request["chromosome"]),
            "layout chromosome differs")
    receipt_desc = layout["sources"].get("factorized_receipt")
    require(isinstance(receipt_desc, dict) and set(receipt_desc) == {"path", "sha256"}
            and sha256(receipt_desc["path"]) == receipt_desc["sha256"],
            "factorized receipt differs")
    receipt = json.loads(Path(receipt_desc["path"]).read_text())
    require(receipt.get("schema") == "r02_lai_factorized_predictors_v1"
            and receipt.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING"
            and receipt.get("event_annotation_schema") == "r02_lai_event_annotations_v1"
            and receipt.get("training_authorized") is False
            and receipt.get("controls_materialized") is False
            and receipt.get("truth_arrays_loaded") is False
            and receipt.get("evaluate_opened") is False,
            "revised factorized receipt differs")
    require(receipt.get("reference_identity") == request["reference_identity"],
            "factorized reference identity differs")
    return reader, receipt_desc, receipt


def prepare(request_path, request_sha256, output_dir):
    import numpy as np
    request = _load_request(request_path, request_sha256)
    reader, factorized_desc, factorized_receipt = _authenticate_layout(request)
    raw = reader.load_raw(request["family"], request["scale_cm"], request["region_id"],
                          request["person_indices"], request["resolved_model_config"])
    require(raw["person_id"].astype(str).tolist() == request["expected_person_ids"],
            "selected person identity differs")
    require(raw["component_id"].astype(str).tolist() == request["expected_component_ids"],
            "selected component identity differs")
    require(set(raw["role"].astype(str).tolist()) == {request["role"]},
            "selected role differs")
    require(raw["layout_recipe"] == request["expected_layout_recipe"]
            and raw["layout_recipe"] == reader._recipe(request["family"],
            request["scale_cm"])["recipe"], "regional recipe differs")
    resolved = request["resolved_model_config"]
    require(isinstance(resolved, dict) and resolved.get("family") == request["family"]
            and resolved.get("common_radius_cm") == raw["layout_recipe"].get("common_radius_cm")
            and resolved.get("rare_radius_cm") == raw["layout_recipe"].get("rare_radius_cm")
            and resolved.get("depth") == raw["layout_recipe"].get("depth")
            and resolved.get("query_max_gap_cm") == raw["layout_recipe"].get("query_max_gap_cm"),
            "resolved model geometry differs from regional recipe")
    arrays = adapter.prepare_predictor_arrays(
        {key: value for key, value in raw.items()
         if key in adapter.PREDICTOR_FIELDS | adapter.QUERY_FIELDS |
            {"person_id", "role", "component_id", "baseline_probabilities"}},
        np, include_queries=True)
    out = Path(output_dir)
    require(not out.exists() and out.parent.is_dir(), "fresh output directory required")
    out.mkdir(mode=0o700)
    batch = out / "regional-source.npz"
    np.savez_compressed(batch, **arrays)
    array_hashes = {name: array_sha256(value, np) for name, value in arrays.items()}
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": STATUS,
        "request_sha256": request_sha256, "layout_sha256": request["layout"]["sha256"],
        "factorized_receipt_sha256": factorized_desc["sha256"],
        "batch_sha256": sha256(batch), "arrays_sha256": array_hashes,
        "arrays": sorted(arrays), "task_id": request["task_id"],
        "case_id": request["case_id"], "direction": request["direction"],
        "role": request["role"], "chromosome": str(request["chromosome"]),
        "family": request["family"], "scale_cm": request["scale_cm"],
        "layout_recipe": request["expected_layout_recipe"],
        "resolved_model_config": request["resolved_model_config"],
        "region_id": request["region_id"], "person_indices": request["person_indices"],
        "person_axis_sha256": adapter.axis_sha(arrays["person_id"], np),
        "query_global_index_sha256": array_hashes["query_global_index"],
        "query_core_sha256": array_hashes["query_core"],
        "query_valid_sha256": array_hashes["query_valid"],
        "reference_identity": request["reference_identity"],
        "reference_join_key": ["chromosome", "position_bp", "REF", "ALT"],
        "event_reference_feature_schema": "r02_lai_event_reference_features_v1",
        "event_reference_features": 15,
        "source_catalogue_scope": factorized_receipt["catalogue_scope"],
        "training_allowed": False, "controls_materialized": False,
        "truth_loaded": False, "selection_performed": False,
        "evaluate_opened": False, "training_integration_implemented": False,
        "target_integration_implemented": False,
        "consumer_tested": "bin/r02_context_train.py:TaskModel.forward",
        "scientific_limit": "PREPARE transport only; no targets, controls, training, selection, evaluation, or biological validation.",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def load(request_path, request_sha256, receipt_path, receipt_sha256, batch_path,
         action="PREPARE"):
    """Authenticate and load the PREPARE-only regional source mapping."""
    import numpy as np
    require(action == "PREPARE", "regional source only permits PREPARE; TRAIN/EVALUATE are closed")
    request = _load_request(request_path, request_sha256)
    _, factorized_desc, _ = _authenticate_layout(request)
    require(sha256(receipt_path) == receipt_sha256, "regional receipt hash differs")
    receipt = json.loads(Path(receipt_path).read_text())
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == STATUS,
            "regional receipt differs")
    linked = {
        "task_id": request["task_id"], "case_id": request["case_id"],
        "direction": request["direction"], "role": request["role"],
        "chromosome": str(request["chromosome"]), "family": request["family"],
        "scale_cm": request["scale_cm"], "region_id": request["region_id"],
        "person_indices": request["person_indices"],
        "layout_recipe": request["expected_layout_recipe"],
        "resolved_model_config": request["resolved_model_config"],
        "reference_identity": request["reference_identity"],
        "factorized_receipt_sha256": factorized_desc["sha256"],
    }
    require(all(receipt.get(name) == value for name, value in linked.items()),
            "regional receipt metadata differs from request/layout")
    require(receipt.get("request_sha256") == request_sha256
            and receipt.get("layout_sha256") == request["layout"]["sha256"]
            and receipt.get("batch_sha256") == sha256(batch_path),
            "regional receipt/input hash differs")
    require(all(receipt.get(name) is False for name in FALSE_FIELDS)
            and receipt.get("training_integration_implemented") is False
            and receipt.get("target_integration_implemented") is False,
            "regional receipt opened a forbidden operation")
    with np.load(batch_path, allow_pickle=False) as source:
        require(sorted(source.files) == receipt.get("arrays"), "regional array inventory differs")
        arrays = {name: source[name] for name in source.files}
    require(not any("truth" in key.lower() or key.lower().startswith("control") for key in arrays),
            "truth/control array is forbidden")
    checked = adapter.prepare_predictor_arrays(arrays, np, include_queries=True)
    require(set(checked) == set(arrays), "derived regional array inventory differs")
    for name, value in checked.items():
        require(array_sha256(value, np) == receipt["arrays_sha256"].get(name),
                "regional array differs: " + name)
    require(checked["person_id"].astype(str).tolist() == request["expected_person_ids"]
            and checked["component_id"].astype(str).tolist() == request["expected_component_ids"]
            and set(checked["role"].astype(str).tolist()) == {request["role"]},
            "regional selected axes differ")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); os.umask(0o077)
    print(json.dumps(prepare(args.request, args.request_sha256, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
