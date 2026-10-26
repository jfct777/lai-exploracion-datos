#!/usr/bin/env python3
"""Prepare four defined LAI EVENT views; never read targets or authorize training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r02_lai_factorized_inputs import PredictorIndex

REQUEST_SCHEMA = "r02_lai_four_control_request_v1"
PLAN_SCHEMA = "r02_lai_four_control_plan_v1"
RECEIPT_SCHEMA = "r02_lai_four_control_receipt_v1"
STATUS = "FOUR_DEFINED_CONDITIONS_MATERIALIZED_DENSITY_BLOCKED"
ALGORITHM = "SHA256_PCG64_PERMUTATION_V1"
CONDITIONS = ("NONE", "NO_RARE", "COUNT_MATCHED", "IDENTITY_SHAM")
CONDITION_OPERATIONS = {
    "NONE": "FULL_ADMITTED_COMMON_EVENT_AND_FIXED_REVISED_BASE",
    "NO_RARE": "NO_INDIVIDUAL_EVENT_CHANNEL_COMMON_AND_FIXED_REVISED_BASE_RETAINED",
    "COUNT_MATCHED": "EVENT_DOSAGE_PERMUTED_WITHIN_PERSON_OBSERVED_LOCUS_MASK_FIXED",
    "IDENTITY_SHAM": "EVENT_DOSAGE_AND_OBSERVATION_PERMUTED_JOINTLY_BETWEEN_PERSONS_PER_LOCUS",
}
FALSE_FIELDS = ("training_authorized", "selection_authorized", "evaluate_opened",
                "controls_materialized", "truth_loaded")
REQUEST_FIELDS = {"schema", "operation", "task_id", "case_id", "direction", "role",
                  "chromosome", "source_plan", "factorized_receipt", "seed",
                  "algorithm_version", "conditions", *FALSE_FIELDS}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value, np):
    a = np.asarray(value); h = hashlib.sha256()
    h.update(a.dtype.str.encode() + b"\0")
    h.update(json.dumps(a.shape, separators=(",", ":")).encode() + b"\0")
    h.update(a.tobytes(order="C")); return h.hexdigest()


def _descriptor(value, name):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")), name + " descriptor differs")
    require(Path(value["path"]).is_file() and sha256(value["path"]) == value["sha256"], name + " differs")
    return Path(value["path"])


def _load_request(path, expected):
    require(sha256(path) == expected, "request hash differs")
    r = json.loads(Path(path).read_text())
    require(set(r) == REQUEST_FIELDS and r.get("schema") == REQUEST_SCHEMA
            and r.get("operation") == "PREPARE", "four-control PREPARE request differs")
    require(r.get("conditions") == list(CONDITIONS), "exact four-condition order required")
    require(r.get("algorithm_version") == ALGORITHM, "algorithm version differs")
    require(type(r.get("seed")) is int and 0 <= r["seed"] < 2**63, "seed differs")
    require(all(r.get(k) is False for k in FALSE_FIELDS), "forbidden operation is open")
    plan_path = _descriptor(r["source_plan"], "source plan")
    receipt_path = _descriptor(r["factorized_receipt"], "factorized receipt")
    plan = json.loads(plan_path.read_text())
    require(plan.get("schema") == "r02_lai_training_inputs_plan_v1"
            and plan.get("status") == "REVISED_FACTORIZED_INPUTS_READY", "revised source plan differs")
    task = next((x for x in plan.get("tasks", []) if x.get("task_id") == r["task_id"]), None)
    require(task is not None, "task absent from source plan")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("schema") == "r02_lai_factorized_predictors_v1"
            and receipt.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING"
            and receipt.get("plan_sha256") == r["source_plan"]["sha256"]
            and receipt.get("event_annotation_schema") == "r02_lai_event_annotations_v1",
            "revised factorized receipt differs")
    for key in ("task_id", "case_id", "direction", "role"):
        require(task.get(key) == r[key] and receipt.get(key) == r[key], key + " identity differs")
    require(str(task.get("chromosome")) == str(r["chromosome"])
            and str(receipt.get("chromosome")) == str(r["chromosome"]), "chromosome differs")
    require(receipt.get("training_authorized") is False
            and receipt.get("controls_materialized") is False
            and receipt.get("truth_arrays_loaded") is False
            and receipt.get("evaluate_opened") is False, "factorized receipt opened forbidden data/action")
    return r, receipt_path, receipt


def _rng(seed, algorithm, task_id, condition, axis):
    material = f"{algorithm}\0{seed}\0{task_id}\0{condition}\0{axis}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:16], "little")
    import numpy as np
    return np.random.Generator(np.random.PCG64(derived))


def transform(dose, valid, seed, algorithm, task_id, np):
    dose = np.asarray(dose); valid = np.asarray(valid, dtype=bool)
    require(dose.shape == valid.shape and dose.ndim == 2, "EVENT dose/observed shapes differ")
    out = {"NONE": (dose.copy(), valid.copy()),
           "NO_RARE": (np.zeros_like(dose), np.zeros_like(valid))}
    count = dose.copy()
    moved_count = changed_count = eligible_count = 0
    for person in range(dose.shape[0]):
        loci = np.flatnonzero(valid[person]); eligible_count += len(loci)
        if len(loci) > 1:
            perm = _rng(seed, algorithm, task_id, "COUNT_MATCHED", person).permutation(len(loci))
            source = loci[perm]
            moved_count += int(np.count_nonzero(source != loci))
            old = count[person, loci].copy(); count[person, loci] = dose[person, source]
            changed_count += int(np.count_nonzero(old != count[person, loci]))
    out["COUNT_MATCHED"] = (count, valid.copy())
    identity_dose = dose.copy(); identity_valid = valid.copy(); moved_identity = changed_identity = 0
    for locus in range(dose.shape[1]):
        if dose.shape[0] > 1:
            perm = _rng(seed, algorithm, task_id, "IDENTITY_SHAM", locus).permutation(dose.shape[0])
            moved_identity += int(np.count_nonzero(perm != np.arange(dose.shape[0])))
            old_d, old_v = dose[:, locus], valid[:, locus]
            identity_dose[:, locus], identity_valid[:, locus] = old_d[perm], old_v[perm]
            changed_identity += int(np.count_nonzero((old_d != identity_dose[:, locus]) |
                                                     (old_v != identity_valid[:, locus])))
    out["IDENTITY_SHAM"] = (identity_dose, identity_valid)
    metrics = {"COUNT_MATCHED": {"eligible_cells": eligible_count, "moved_indices": moved_count,
                                  "values_changed": changed_count},
               "IDENTITY_SHAM": {"eligible_cells": dose.size, "moved_indices": moved_identity,
                                  "values_changed": changed_identity}}
    return out, metrics


def prepare(request_path, request_sha256, output_dir):
    import numpy as np
    r, receipt_path, source_receipt = _load_request(request_path, request_sha256)
    index = PredictorIndex(receipt_path)
    people = np.arange(source_receipt["people"], dtype=np.int64)
    event_count = int(source_receipt["loci"]["event"])
    maximum = int(source_receipt.get("maximum_loci_per_read", 8192))
    doses, valids, metadata = [], [], None
    for begin in range(0, event_count, maximum):
        d, v, m = index.read_loci("event", people, begin, min(event_count, begin + maximum))
        doses.append(d); valids.append(v)
        if metadata is None: metadata = {k: [] for k, x in m.items() if hasattr(x, "shape") and x.ndim == 1 and len(x) == len(m["source_index"])}
        for k in metadata: metadata[k].append(m[k])
    dose = np.concatenate(doses, axis=1); valid = np.concatenate(valids, axis=1)
    metadata = {k: np.concatenate(v) for k, v in metadata.items()}
    require(dose.shape == (source_receipt["people"], event_count), "complete EVENT axis differs")
    views, metrics = transform(dose, valid, r["seed"], r["algorithm_version"], r["task_id"], np)
    # Exact invariants, including degenerate axes.
    require(np.array_equal(views["NONE"][0], dose) and np.array_equal(views["NONE"][1], valid), "NONE differs")
    require(not views["NO_RARE"][1].any() and not views["NO_RARE"][0].any(), "NO_RARE differs")
    require(np.array_equal(views["COUNT_MATCHED"][1], valid), "COUNT_MATCHED mask differs")
    for person in range(len(people)):
        require(np.array_equal(np.sort(views["COUNT_MATCHED"][0][person, valid[person]]),
                               np.sort(dose[person, valid[person]])), "COUNT_MATCHED histogram differs")
    for locus in range(event_count):
        before = sorted(zip(dose[:, locus].tolist(), valid[:, locus].tolist()))
        after = sorted(zip(views["IDENTITY_SHAM"][0][:, locus].tolist(), views["IDENTITY_SHAM"][1][:, locus].tolist()))
        require(before == after, "IDENTITY_SHAM locus multiset differs")
    arrays = {"person_id": index.index["person_id"],
              "event_global_index": np.arange(event_count, dtype=np.int64)}
    for key in ("source_index", "chromosome", "position_bp", "ref", "alt", "counted_allele", "is_alt"):
        require(key in metadata, "EVENT identity missing: " + key); arrays["event_" + key] = metadata[key]
    for name, (d, v) in views.items():
        arrays[name + "_event_dosage"] = d; arrays[name + "_event_valid"] = v
    out = Path(output_dir); require(not out.exists() and out.parent.is_dir(), "fresh output directory required")
    out.mkdir(mode=0o700); batch = out / "control-views.npz"; np.savez_compressed(batch, **arrays)
    hashes = {k: array_sha256(v, np) for k, v in arrays.items()}
    receipt = {"schema": RECEIPT_SCHEMA, "status": STATUS, "request_sha256": request_sha256,
               "source_plan_sha256": r["source_plan"]["sha256"],
               "factorized_receipt_sha256": r["factorized_receipt"]["sha256"],
               "batch_sha256": sha256(batch), "arrays": sorted(arrays), "arrays_sha256": hashes,
               "task_id": r["task_id"], "case_id": r["case_id"], "direction": r["direction"],
               "role": r["role"], "chromosome": str(r["chromosome"]), "people": len(people),
               "event_loci": event_count, "conditions_materialized": list(CONDITIONS),
               "condition_operations": CONDITION_OPERATIONS,
               "condition_display_aliases": {"NONE": "FULL"}, "condition_metrics": metrics,
               "seed": r["seed"], "algorithm_version": r["algorithm_version"],
               "event_axis_sha256": hashes["event_global_index"],
               "person_axis_sha256": hashes["person_id"],
               "reference_identity": source_receipt["reference_identity"],
               "event_annotation_schema": source_receipt["event_annotation_schema"],
               "fixed_revised_base": {key: source_receipt["flare"].get(key) for key in
                   ("initial_gen", "marker_condition", "sha256", "projection_receipt_sha256")},
               "no_rare_retained_channels": ["COMMON", "FIXED_REVISED_BASE_GEN10"],
               "no_rare_serialized_but_model_masked_event_fields":
                   ["EVENT_COORDINATES", "EVENT_12_METADATA", "EVENT_15_REFERENCE_FEATURES"],
               "maximum_loci_per_read": maximum, "read_all_loci_in_blocks_without_truncation": True,
               "density_matched_status": "BLOCKED_NO_ADMITTED_SCALE_OR_OPERATION",
               "training_authorized": False, "selection_authorized": False,
               "evaluate_opened": False, "controls_materialized": False, "truth_loaded": False,
               "scientific_limit": "Four diagnostic information destructions only; no null-model, stability, training, selection, evaluation, or biological claim."}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_digest = sha256(out / "receipt.json")
    checked, _ = load(request_path, request_sha256, out / "receipt.json", receipt_digest,
                      batch, action="PREPARE")
    validation = {"schema": "r02_lai_four_control_readback_v1",
                  "status": "REQUEST_RECEIPT_BATCH_AND_ARRAY_HASHES_VERIFIED",
                  "request_sha256": request_sha256, "receipt_sha256": receipt_digest,
                  "batch_sha256": receipt["batch_sha256"], "arrays_verified": len(checked),
                  "action": "PREPARE", "training_authorized": False}
    (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    return receipt


def load(request_path, request_sha256, receipt_path, receipt_sha256, batch_path, action="PREPARE"):
    """Authenticate a partial four-condition product; TRAIN/SELECT/EVALUATE stay closed."""
    import numpy as np
    require(action == "PREPARE", "four-control product only permits PREPARE")
    request, _, source = _load_request(request_path, request_sha256)
    require(sha256(receipt_path) == receipt_sha256, "four-control receipt hash differs")
    receipt = json.loads(Path(receipt_path).read_text())
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == STATUS,
            "four-control receipt differs")
    linked = {"request_sha256": request_sha256, "source_plan_sha256": request["source_plan"]["sha256"],
              "factorized_receipt_sha256": request["factorized_receipt"]["sha256"],
              "task_id": request["task_id"], "case_id": request["case_id"],
              "direction": request["direction"], "role": request["role"],
              "chromosome": str(request["chromosome"]), "seed": request["seed"],
              "algorithm_version": request["algorithm_version"],
              "reference_identity": source["reference_identity"]}
    require(all(receipt.get(k) == v for k, v in linked.items()), "four-control receipt metadata differs")
    require(receipt.get("conditions_materialized") == list(CONDITIONS)
            and receipt.get("condition_operations") == CONDITION_OPERATIONS
            and all(receipt.get(k) is False for k in FALSE_FIELDS), "partial product opened forbidden operation")
    require(receipt.get("batch_sha256") == sha256(batch_path), "four-control batch hash differs")
    with np.load(batch_path, allow_pickle=False) as z:
        require(sorted(z.files) == receipt.get("arrays"), "four-control array inventory differs")
        arrays = {k: z[k] for k in z.files}
    for k, v in arrays.items():
        require(array_sha256(v, np) == receipt["arrays_sha256"].get(k), "four-control array differs: " + k)
    require(np.array_equal(arrays["event_global_index"], np.arange(receipt["event_loci"])), "EVENT global axis differs")
    return arrays, receipt


def build_plan(source_plan_path, completion_path, seed, algorithm_version, output_dir):
    require(algorithm_version == ALGORITHM and type(seed) is int and 0 <= seed < 2**63, "seed/version differs")
    source_sha = sha256(source_plan_path); source = json.loads(Path(source_plan_path).read_text())
    completion = json.loads(Path(completion_path).read_text())
    require(source.get("schema") == "r02_lai_training_inputs_plan_v1"
            and source.get("status") == "REVISED_FACTORIZED_INPUTS_READY"
            and completion.get("status") == "TWELVE_FACTORIZED_INDICES_VERIFIED", "source inputs incomplete")
    products = {x["task_id"]: x for x in completion["products"]}
    out = Path(output_dir); require(not out.exists() and out.parent.is_dir(), "fresh plan directory required")
    (out / "requests").mkdir(parents=True, mode=0o700)
    rows = []
    for task in source["tasks"]:
        product = products.get(task["task_id"]); require(product is not None, "factorized product missing")
        require(sha256(product["receipt"]) == product["receipt_sha256"], "factorized receipt changed")
        request = {"schema": REQUEST_SCHEMA, "operation": "PREPARE", "task_id": task["task_id"],
                   "case_id": task["case_id"], "direction": task["direction"], "role": task["role"],
                   "chromosome": str(task["chromosome"]),
                   "source_plan": {"path": str(Path(source_plan_path).resolve()), "sha256": source_sha},
                   "factorized_receipt": {"path": product["receipt"], "sha256": product["receipt_sha256"]},
                   "seed": seed, "algorithm_version": algorithm_version,
                   "conditions": list(CONDITIONS), **{k: False for k in FALSE_FIELDS}}
        path = out / "requests" / (task["task_id"] + ".json")
        path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        rows.append({"task_id": task["task_id"], "request": {"path": str(path.resolve()), "sha256": sha256(path)}})
    require(len(rows) == 12, "exact 12-case plan required")
    plan = {"schema": PLAN_SCHEMA, "status": "TWELVE_CASES_READY_FOR_FOUR_CONTROL_PREPARE",
            "source_plan_sha256": source_sha, "index_completion_sha256": sha256(completion_path),
            "seed": seed, "algorithm_version": algorithm_version, "conditions": list(CONDITIONS),
            "condition_operations": CONDITION_OPERATIONS,
            "tasks": rows, "training_authorized": False, "selection_authorized": False,
            "evaluate_opened": False, "controls_materialized": False, "truth_loaded": False}
    (out / "PLAN.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-plan"); p.add_argument("--source-plan", required=True); p.add_argument("--index-completion", required=True)
    p.add_argument("--seed", required=True, type=int); p.add_argument("--algorithm-version", default=ALGORITHM); p.add_argument("--output-dir", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--request", required=True); q.add_argument("--request-sha256", required=True); q.add_argument("--output-dir", required=True)
    a = parser.parse_args(); os.umask(0o077)
    result = (build_plan(a.source_plan, a.index_completion, a.seed, a.algorithm_version, a.output_dir)
              if a.command == "build-plan" else prepare(a.request, a.request_sha256, a.output_dir))
    print(json.dumps({k: result[k] for k in ("schema", "status")}, sort_keys=True))


if __name__ == "__main__": main()
