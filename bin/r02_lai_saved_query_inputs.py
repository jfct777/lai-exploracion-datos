#!/usr/bin/env python3
"""Load authenticated saved LAI query inputs without genotype reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE, array_sha256, load_prefix, project


RETURNED = ("baseline_probabilities", "query_cm", "query_core", "query_global_index",
            "query_burden", "truth_state", "truth_observed", "query_valid", "person_id", "role")
_SHA_CACHE = {}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    path = Path(path); key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    if key not in _SHA_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(block)
        _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def _path(spec, name):
    _require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, name + " descriptor differs")
    path = Path(spec["path"])
    _require(_sha(path) == spec["sha256"], name + " SHA-256 differs")
    return path


def _json(spec, name):
    return json.loads(_path(spec, name).read_text())


def _members(path, names, np):
    with np.load(path, allow_pickle=False) as source:
        _require(set(names) <= set(source.files), "saved array members are incomplete")
        return {name: source[name] for name in names}


def _selected(value, people, queries, np):
    value = np.asarray(value)
    _require(value.ndim >= 2 and value.shape[0] > max(people, default=-1)
             and value.shape[1] > max(queries, default=-1), "overlay axes are out of range")
    return value[np.asarray(people)[:, None], np.asarray(queries)[None, :]]


def _identity(request, receipt, name):
    for key in ("task_id", "case_id", "direction", "role"):
        _require(receipt.get(key) == request[key], name + " identity differs: " + key)
    _require(str(receipt.get("chromosome")) == str(request["chromosome"]), name + " chromosome differs")


def load_saved_queries(input_descriptor, contract, np):
    """Return ten saved FIT/NONE arrays, authenticated to the original PREPARE chain."""
    _require(isinstance(contract, dict), "diagnostic contract differs")
    _require(input_descriptor.get("format") == "LAI_FACTORIZED_TRAINING_LINK_V1"
             and input_descriptor.get("role") == "FIT", "input is not a saved FIT training link")
    request_path = _path({"path": input_descriptor["path"], "sha256": input_descriptor["sha256"]},
                         "training request")
    wrapper = _json({"path": input_descriptor["receipt_path"],
                     "sha256": input_descriptor["receipt_sha256"]}, "training wrapper")
    request = json.loads(request_path.read_text())
    _require(request.get("schema") == "r02_lai_regional_truth_join_request_v2"
             and request.get("operation") == "PREPARE" and request.get("condition") == "NONE"
             and request.get("role") == "FIT" and request.get("rare_radius_cm") == 1.0
             and request.get("lai_burden_scope") == SCOPE
             and request.get("lai_burden_boundary") == BOUNDARY
             and request.get("lai_burden_count_names") == list(COUNT_NAMES)
             and all(request.get(k) is False for k in
                     ("training_authorized", "selection_authorized", "evaluate_opened", "controls_materialized")),
             "request is not the fixed-radius FIT/NONE PREPARE product")
    if "rare_allele" in contract:
        _require(contract["rare_allele"] == "DNABR2723_FIXED_SOURCE_MINOR", "rare allele contract differs")
    if "count_scope" in contract:
        _require(contract["count_scope"] ==
                 "query-local DNABR2723 fixed-source-minor event counts within 1 cM",
                 "query burden contract differs")
    allowed = contract.get("cases", [contract.get("case")])
    _require(isinstance(allowed,list) and allowed and all(isinstance(v,str) and v for v in allowed)
             and request["case_id"] in allowed, "explicit diagnostic cases required")
    _require(wrapper.get("schema") == "r02_lai_factorized_training_link_receipt_v1"
             and wrapper.get("status") == "FACTORIZED_LINK_BOUND_TO_AUTHENTICATED_TRAINING_PROTOCOL"
             and wrapper.get("request_sha256") == input_descriptor["sha256"]
             and wrapper.get("case_id") == input_descriptor.get("lai_case_id") == request["case_id"]
             and wrapper.get("condition") == "NONE" and wrapper.get("role") == "FIT"
             and wrapper.get("evaluate_opened") is False, "training wrapper identity differs")
    _path(wrapper["protocol"], "training protocol")
    if "lai_training_protocol" in contract:
        _require(contract["lai_training_protocol"] == wrapper["protocol"], "training protocol contract differs")
    source = _json(wrapper["source_prepare_receipt"], "source PREPARE receipt")
    _require(source.get("schema") == "r02_lai_regional_truth_factorized_link_receipt_v1"
             and source.get("status") ==
                 "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
             and source.get("request_sha256") == input_descriptor["sha256"]
             and source.get("reconstructed_arrays_sha256") == wrapper.get("reconstructed_arrays_sha256")
             and source.get("arrays") == wrapper.get("arrays")
             and source.get("factorized_link") is True and source.get("truth_joined") is True
             and all(source.get(k) is False for k in
                     ("training_authorized", "selection_authorized", "evaluate_opened")),
             "source PREPARE receipt differs")
    for key in ("regional_request", "regional_receipt", "regional_batch", "overlay", "overlay_receipt",
                "control_request", "control_receipt", "control_batch", "event_burden_prefix",
                "event_burden_prefix_receipt"):
        _path(request[key], key)
    regional_request = _json(request["regional_request"], "regional request")
    regional_receipt = _json(request["regional_receipt"], "regional receipt")
    overlay_receipt = _json(request["overlay_receipt"], "overlay receipt")
    control_receipt = _json(request["control_receipt"], "control receipt")
    _require(regional_receipt.get("schema") == "r02_lai_regional_predictor_receipt_v1"
             and regional_receipt.get("request_sha256") == request["regional_request"]["sha256"]
             and regional_receipt.get("batch_sha256") == request["regional_batch"]["sha256"]
             and regional_receipt.get("region_id") == wrapper.get("region_id") == regional_request.get("region_id"),
             "regional manifest chain differs")
    _identity(request, regional_request, "regional request")
    _require(source.get("regional_receipt_sha256") == request["regional_receipt"]["sha256"]
             and source.get("overlay_receipt_sha256") == request["overlay_receipt"]["sha256"]
             and source.get("control_receipt_sha256") == request["control_receipt"]["sha256"]
             and source.get("event_burden_prefix_receipt_sha256") ==
                 request["event_burden_prefix_receipt"]["sha256"], "source receipt chain differs")
    for value in (regional_receipt.get("scale_cm"), regional_receipt.get("layout_recipe", {}).get("rare_radius_cm"),
                  regional_receipt.get("resolved_model_config", {}).get("rare_radius_cm")):
        _require(value == 1.0, "regional radius differs from fixed 1 cM")
    _identity(request, regional_receipt, "regional receipt"); _identity(request, control_receipt, "control receipt")
    _require(control_receipt.get("request_sha256") == request["control_request"]["sha256"]
             and control_receipt.get("batch_sha256") == request["control_batch"]["sha256"],
             "control manifest chain differs")
    _require(overlay_receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2"
             and overlay_receipt.get("output_sha256") == request["overlay"]["sha256"]
             and overlay_receipt.get("roles") == ["FIT"]
             and overlay_receipt.get("truth_read_for_query_geometry") is False,
             "overlay manifest chain differs")
    if "people" in contract:
        _require(overlay_receipt.get("n_people") == contract["people"], "people contract differs")
    if "queries" in contract:
        _require(overlay_receipt.get("n_queries") == contract["queries"], "query contract differs")
    rnames = ("baseline_probabilities", "query_cm", "query_chrom", "query_core", "query_global_index",
              "query_valid", "person_id", "role", "component_id", "event_global_index", "event_chrom", "event_cm")
    regional = _members(request["regional_batch"]["path"], rnames, np)
    for name in ("query_valid", "query_core"):
        _require(regional[name].dtype == np.bool_, name + " must be boolean")
    _require(np.all(regional["query_chrom"][regional["query_valid"]] == int(request["chromosome"])),
             "query chromosome differs from declared chromosome")
    for name, value in regional.items():
        _require(array_sha256(value, np) == regional_receipt["arrays_sha256"].get(name),
                 "regional array hash differs: " + name)
    people_n, queries_n = regional["query_valid"].shape
    _require(regional["baseline_probabilities"].shape == (people_n, queries_n, 6)
             and regional["query_cm"].shape == regional["query_chrom"].shape ==
                 regional["query_core"].shape == (people_n, queries_n)
             and len(regional["person_id"]) == len(regional["role"]) == people_n,
             "regional person/query dimensions differ")
    qidx = np.asarray(regional["query_global_index"])
    _require(qidx.ndim == 1 and np.issubdtype(qidx.dtype, np.integer)
             and len(np.unique(qidx)) == len(qidx) and (qidx >= 0).all(), "regional query axis differs")
    onames = ("person_id", "role", "component_id", "query_cm", "query_chrom", "query_valid",
              "truth_state", "truth_observed")
    overlay = _members(request["overlay"]["path"], onames, np)
    _require(overlay["query_valid"].dtype == overlay["truth_observed"].dtype == np.bool_,
             "overlay observation masks must be boolean")
    ids = overlay["person_id"].astype(str).tolist(); _require(len(ids) == len(set(ids)), "overlay people duplicate")
    lookup = {value: index for index, value in enumerate(ids)}
    rids = regional["person_id"].astype(str).tolist()
    if "people" in contract:
        _require(len(rids)==contract["people"], "declared person count differs")
    _require(len(rids) == len(set(rids)) and all(value in lookup for value in rids), "regional people differ")
    people = [lookup[value] for value in rids]
    for name in ("role", "component_id"):
        _require(np.array_equal(regional[name].astype(str), overlay[name][people].astype(str)), name + " differs")
    for name in ("query_cm", "query_chrom", "query_valid"):
        _require(np.array_equal(regional[name], _selected(overlay[name], people, qidx.tolist(), np), equal_nan=True),
                 name + " differs between regional source and overlay")
    truth = {name: _selected(overlay[name], people, qidx.tolist(), np)
             for name in ("truth_state", "truth_observed")}
    _require(not (truth["truth_observed"] & ~regional["query_valid"]).any(), "truth extends query support")
    prefix, prefix_receipt = load_prefix(request["event_burden_prefix"]["path"],
                                         request["event_burden_prefix_receipt"]["path"], "NONE")
    _identity(request, prefix_receipt, "burden-prefix receipt")
    _require(prefix_receipt["event_input_sha256"] == request["control_batch"]["sha256"]
             and prefix_receipt["control_receipt_sha256"] == request["control_receipt"]["sha256"]
             and prefix_receipt["event_axis_receipt_sha256"] == regional_receipt["factorized_receipt_sha256"]
             and prefix_receipt["reference_identity"] == control_receipt["reference_identity"],
             "burden-prefix source chain differs")
    b_ids = prefix["person_id"].astype(str).tolist(); b_lookup = {v: i for i, v in enumerate(b_ids)}
    _require(len(b_ids) == len(b_lookup) and all(v in b_lookup for v in rids), "burden people differ")
    prefix = dict(prefix); prefix["prefix_counts"] = prefix["prefix_counts"][[b_lookup[v] for v in rids]]
    inverse = np.empty(len(prefix["event_order"]), dtype=np.int64); inverse[prefix["event_order"]] = np.arange(len(inverse))
    event = inverse[np.asarray(regional["event_global_index"])]
    for name in ("event_chrom", "event_cm"):
        value=regional[name]
        _require(value.shape==(len(rids),len(event))
                 and np.array_equal(value,np.broadcast_to(value[:1],value.shape)),
                 name + " must be shared across all people")
    _require(np.array_equal(prefix["event_chrom"][event], regional["event_chrom"][0])
             and np.array_equal(prefix["event_cm"][event], regional["event_cm"][0]), "burden EVENT axis differs")
    result = {name: regional[name] for name in RETURNED if name in regional}
    result.update(truth); result["query_burden"] = project(prefix, regional["query_cm"], regional["query_chrom"], 1.0, np)
    _require(set(result) == set(RETURNED), "returned query inputs differ")
    for name, value in result.items():
        _require(array_sha256(value, np) == wrapper["reconstructed_arrays_sha256"].get(name),
                 "reconstructed array hash differs: " + name)
    return result
