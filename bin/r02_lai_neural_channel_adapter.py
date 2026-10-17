#!/usr/bin/env python3
"""Bind admitted COMMON/EVENT material and a projected FLARE comparator.

This is the bridge between the existing explicit-channel producer and the new
truth-blind dense query grid.  It copies only predictor arrays from the legacy
batch, discards its query/target arrays, and binds the separately projected
FLARE probabilities.  Channel selection and FLARE initialization selection are
upstream decisions and are never inferred here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

SCHEMA = "r02_lai_neural_channel_adapter_request_v1"
CONTROLS = {
    "NONE": "ALL_ADMITTED_CONTEXT", "NO_RARE": "COMMON_AND_FLARE_ONLY",
    "COUNT_MATCHED": "RARE_COUNT_FIXED_IDENTITY_REPLACED",
    "DENSITY_MATCHED": "RARE_DENSITY_FIXED_IDENTITY_REPLACED",
    "IDENTITY_SHAM": "RARE_LOCUS_AND_COUNT_FIXED_PERSON_RELATION_BROKEN",
}


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
    expected_controls = [{"name": name, "operation": operation, "truth_used": False,
                          "preserves_people": True} for name, operation in CONTROLS.items()]
    require(request.get("controls") == expected_controls, "registered controls differ")
    inputs = request.get("inputs", {})
    required = {"legacy_batch", "legacy_receipt", "overlay", "overlay_receipt",
                "flare_probabilities", "flare_receipt"}
    require(set(inputs) == required, "exact adapter inputs required")
    paths = {name: resolve(value, name, staged) for name, value in inputs.items()}
    legacy_receipt = json.loads(paths["legacy_receipt"].read_text())
    overlay_receipt = json.loads(paths["overlay_receipt"].read_text())
    flare_receipt = json.loads(paths["flare_receipt"].read_text())
    require(legacy_receipt.get("schema") == "r02_genomic_event_training_batch_v1"
            and legacy_receipt.get("arrays_sha256") == inputs["legacy_batch"]["sha256"]
            and legacy_receipt.get("channel_policy") == "EXPLICIT_PRESELECTED_NO_INFERENCE",
            "legacy channel receipt differs")
    require(overlay_receipt.get("schema") == "r02_lai_truthblind_context_receipt_v2"
            and overlay_receipt.get("output_sha256") == inputs["overlay"]["sha256"]
            and overlay_receipt.get("truth_read_for_query_geometry") is False, "overlay receipt differs")
    require(flare_receipt.get("schema") == "r02_lai_flare_query_projection_receipt_v1"
            and flare_receipt.get("probabilities_sha256") == inputs["flare_probabilities"]["sha256"]
            and flare_receipt.get("truth_used_for_projection") is False
            and flare_receipt.get("selection_performed") is False, "FLARE projection receipt differs")
    with np.load(paths["legacy_batch"], allow_pickle=False) as source:
        keep = {key: source[key] for key in source.files if key.startswith(("common_", "event_"))}
        legacy_people = source["person_id"].astype(str)
    with np.load(paths["overlay"], allow_pickle=False) as source:
        people = source["person_id"].astype(str); role = source["role"].astype(str)
        component = source["component_id"].astype(str); queries = source["query_valid"].shape[1]
        # Deliberately do not read truth_state or truth_observed.
    require(np.array_equal(legacy_people, people), "legacy channels and overlay person axes differ")
    expected_role = request.get("expected_role")
    require(expected_role in ("FIT", "SELECT") and set(role.tolist()) == {expected_role}, "role differs")
    probability = np.load(paths["flare_probabilities"], mmap_mode="r", allow_pickle=False)
    require(probability.shape == (len(people), queries, 6) and np.isfinite(probability).all()
            and (probability >= 0).all() and np.allclose(probability.sum(-1), 1, atol=5e-5),
            "FLARE comparator axes/domain differ")
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
        "source_legacy_batch_sha256":inputs["legacy_batch"]["sha256"],
        "source_flare_projection_sha256":inputs["flare_probabilities"]["sha256"],
        "flare_initialization_label":flare_receipt["initialization_label"],
        "missing_policy":"OBSERVED_MASK_SEPARATE_FROM_ZERO_DOSAGE",
        "flare_semantics":"COMPARATOR_INPUT_NOT_TRUTH", "truth_read_for_site_selection":False,
        "truth_arrays_loaded":False, "selection_performed":False, "controls":expected_controls,
        "scientific_limit":"Channel identities are inherited from the explicit upstream contract; this adapter does not admit them."}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--request",required=True)
    p.add_argument("--request-sha256",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--staged-inputs");a=p.parse_args();os.umask(0o077)
    print(json.dumps(run(a.request,a.request_sha256,a.output_dir,a.staged_inputs),sort_keys=True))


if __name__ == "__main__": main()
