#!/usr/bin/env python3
"""Bind admitted COMMON/EVENT material and a projected FLARE predictor.

This is the bridge between ``assemble_explicit_channels.py`` and the
truth-blind dense query grid.  It copies only COMMON/EVENT predictor arrays,
checks that the producer and the overlay describe the same query axis, and
binds the separately projected FLARE probabilities.  Channel selection and
FLARE initialization are upstream decisions and are never inferred here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

SCHEMA = "r02_lai_neural_channel_adapter_request_v1"


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def resolve(item, name, staged=None):
    require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "invalid descriptor: " + name)
    path = Path(item["path"])
    if staged:
        matches = [p for p in Path(staged).iterdir() if p.is_file() and sha(p) == item["sha256"]]
        require(len(matches) == 1, "staged input absent or ambiguous: " + name); path = matches[0]
    require(path.is_file() and sha(path) == item["sha256"], "input differs: " + name)
    return path


def axis_sha(values, np):
    h = hashlib.sha256()
    for value in values.astype(str).tolist(): h.update(value.encode() + b"\0")
    return h.hexdigest()


def run(request_path, request_sha, output_dir, staged=None):
    import numpy as np
    require(sha(request_path) == request_sha, "request SHA-256 differs")
    request = json.loads(Path(request_path).read_text())
    require(request.get("schema") == SCHEMA and request.get("operation") == "BIND_EXISTING_CHANNELS_NO_SELECTION",
            "adapter request differs")
    require("controls" not in request,
            "this L1/L2 adapter does not implement or certify training controls")
    inputs = request.get("inputs", {})
    shared = {"overlay", "overlay_receipt", "flare_probabilities", "flare_receipt"}
    explicit_names = {"source_batch", "source_receipt"}
    legacy_names = {"legacy_batch", "legacy_receipt"}
    require(set(inputs) in (shared | explicit_names, shared | legacy_names), "exact adapter inputs required")
    paths = {name: resolve(value, name, staged) for name, value in inputs.items()}
    source_names = explicit_names if explicit_names <= set(inputs) else legacy_names
    batch_name, receipt_name = sorted(source_names)
    # Sorting puts *_batch before *_receipt for either accepted pair.
    producer_receipt = json.loads(paths[receipt_name].read_text())
    overlay_receipt = json.loads(paths["overlay_receipt"].read_text())
    flare_receipt = json.loads(paths["flare_receipt"].read_text())
    producer_schema = producer_receipt.get("schema")
    if producer_schema == "r02_lai_predictor_channels_receipt_v1":
        require(source_names == explicit_names
                and producer_receipt.get("arrays_sha256") == inputs[batch_name]["sha256"]
                and producer_receipt.get("training_allowed") is False
                and producer_receipt.get("truth_loaded") is False
                and producer_receipt.get("selection_performed") is False,
                "explicit-channel receipt differs")
    else:
        require(source_names == legacy_names and producer_schema == "r02_genomic_event_training_batch_v1"
                and producer_receipt.get("arrays_sha256") == inputs[batch_name]["sha256"]
                and producer_receipt.get("channel_policy") == "EXPLICIT_PRESELECTED_NO_INFERENCE",
                "legacy channel receipt differs")
    require(overlay_receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2"
            and overlay_receipt.get("output_sha256") == inputs["overlay"]["sha256"]
            and overlay_receipt.get("truth_read_for_query_geometry") is False, "overlay receipt differs")
    require(flare_receipt.get("schema") == "r02_lai_flare_query_projection_receipt_v1"
            and flare_receipt.get("probabilities_sha256") == inputs["flare_probabilities"]["sha256"]
            and flare_receipt.get("truth_used_for_projection") is False
            and flare_receipt.get("selection_performed") is False, "FLARE projection receipt differs")
    with np.load(paths[batch_name], allow_pickle=False) as source:
        keep = {key: source[key] for key in source.files if key.startswith(("common_", "event_"))}
        legacy_people = source["person_id"].astype(str)
        producer_query = {key: source[key] for key in
                          ("query_cm", "query_chrom", "query_pos_bp", "query_valid")
                          if key in source.files}
        producer_baseline = source["baseline_probabilities"] if "baseline_probabilities" in source.files else None
    with np.load(paths["overlay"], allow_pickle=False) as source:
        people = source["person_id"].astype(str); role = source["role"].astype(str)
        component = source["component_id"].astype(str); queries = source["query_valid"].shape[1]
        overlay_query = {key: source[key] for key in
                         ("query_cm", "query_chrom", "query_pos_bp", "query_valid")}
        # Deliberately do not read truth_state or truth_observed.
    require(np.array_equal(legacy_people, people), "legacy channels and overlay person axes differ")
    if producer_schema == "r02_lai_predictor_channels_receipt_v1":
        require(set(producer_query) == set(overlay_query), "explicit producer query axis is incomplete")
        for key in sorted(overlay_query):
            require(np.array_equal(producer_query[key], overlay_query[key]),
                    "explicit producer and overlay differ: " + key)
    expected_role = request.get("expected_role")
    require(expected_role in ("FIT", "SELECT") and set(role.tolist()) == {expected_role}, "role differs")
    probability = np.load(paths["flare_probabilities"], mmap_mode="r", allow_pickle=False)
    require(probability.shape == (len(people), queries, 6) and np.isfinite(probability).all()
            and (probability >= 0).all() and np.allclose(probability.sum(-1), 1, atol=5e-5),
            "FLARE comparator axes/domain differ")
    if producer_baseline is not None:
        require(producer_baseline.shape == probability.shape
                and np.allclose(producer_baseline, probability, atol=0, rtol=0),
                "explicit producer and requested FLARE probabilities differ")
    arrays = {"person_id": people, "role": role, "component_id": component, **keep,
              "baseline_probabilities": np.asarray(probability, dtype=np.float32)}
    needed = {"common_features", "common_cm", "common_chrom", "common_pos_bp", "common_valid", "common_core",
              "event_metadata", "event_dosage", "event_cm", "event_chrom", "event_pos_bp", "event_valid", "event_core"}
    require(needed <= set(arrays), "legacy COMMON/EVENT predictor arrays incomplete")
    require(not any("truth" in key or key.startswith("query_") for key in arrays),
            "target/query array leaked into channel batch")
    out = Path(output_dir); require(not out.exists() and out.parent.is_dir(), "fresh output required")
    out.mkdir(mode=0o700); batch = out / "channel-batch.npz"; np.savez_compressed(batch, **arrays)
    result = {"schema":"r02_lai_neural_channel_receipt_v1",
        "status":"CHANNELS_AND_FLARE_BOUND_NO_TRAINING_NO_SELECTION", "batch_sha256":sha(batch),
        "case_id":request["case_id"], "direction":request["direction"], "expected_role":expected_role,
        "source_genotype_material_sha256":request["source_genotype_material_sha256"],
        "source_overlay_sha256":inputs["overlay"]["sha256"], "person_axis_sha256":axis_sha(people, np),
        "source_channel_batch_sha256":inputs[batch_name]["sha256"],
        "source_channel_receipt_schema":producer_schema,
        "source_flare_projection_sha256":inputs["flare_probabilities"]["sha256"],
        "flare_initialization_label":flare_receipt["initialization_label"],
        "missing_policy":"OBSERVED_MASK_SEPARATE_FROM_ZERO_DOSAGE",
        "flare_semantics":"COMPARATOR_INPUT_NOT_TRUTH", "truth_read_for_site_selection":False,
        "truth_arrays_loaded":False, "selection_performed":False,
        "controls_materialized":False,
        "scientific_limit":"Channel identities are inherited from the explicit upstream contract; this adapter does not admit them."}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--request",required=True)
    p.add_argument("--request-sha256",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--staged-inputs");a=p.parse_args();os.umask(0o077)
    print(json.dumps(run(a.request,a.request_sha256,a.output_dir,a.staged_inputs),sort_keys=True))


if __name__ == "__main__": main()
