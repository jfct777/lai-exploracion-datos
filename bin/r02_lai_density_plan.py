#!/usr/bin/env python3
"""Build the admitted 12-case by 3-radius DENSITY_MATCHED PREPARE plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from r02_lai_density_control import ALGORITHM_VERSION, MATERIAL_REQUEST_SCHEMA

RADII_CM = (0.05, 0.2, 1.0)
SEED = 20261021
PLAN_SCHEMA = "r02_lai_density_plan_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_file(), "file absent: " + str(path))
    return {"path": str(path), "sha256": sha256(path)}


def build(control_root, index_root, output_dir):
    control_root, index_root, out = map(Path, (control_root, index_root, output_dir))
    require(not out.exists() and out.parent.is_dir(), "fresh density plan directory required")
    controls = {path.parent.parent.name: path for path in control_root.glob("*/prepared/receipt.json")}
    indices = {path.parent.name: path for path in index_root.glob("*/receipt.json")}
    require(len(controls) == len(indices) == 12 and set(controls) == set(indices),
            "exact matching twelve control/index cases required")
    (out / "requests").mkdir(parents=True, mode=0o700)
    tasks = []
    for task_id in sorted(controls):
        control_path, axis_receipt_path = controls[task_id], indices[task_id]
        control = json.loads(control_path.read_text())
        axis = json.loads(axis_receipt_path.read_text())
        require(control.get("schema") == "r02_lai_four_control_receipt_v1"
                and control.get("status") == "FOUR_DEFINED_CONDITIONS_MATERIALIZED_DENSITY_BLOCKED",
                "four-control receipt differs: " + task_id)
        require(axis.get("schema") == "r02_lai_factorized_predictors_v1"
                and axis.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING",
                "factorized receipt differs: " + task_id)
        for key in ("task_id", "case_id", "direction", "role", "reference_identity"):
            require(control.get(key) == axis.get(key), key + " differs: " + task_id)
        require(str(control.get("chromosome")) == str(axis.get("chromosome")),
                "chromosome differs: " + task_id)
        control_batch = control_path.parent / "control-views.npz"
        event_axis = axis_receipt_path.parent / "indices.npz"
        require(sha256(control_batch) == control["batch_sha256"]
                and sha256(event_axis) == axis["outputs_sha256"]["indices.npz"],
                "source batch changed: " + task_id)
        for radius in RADII_CM:
            scale = format(radius, ".12g")
            request = {
                "schema": MATERIAL_REQUEST_SCHEMA, "operation": "PREPARE",
                "task_id": task_id, "case_id": axis["case_id"],
                "direction": axis["direction"], "role": axis["role"],
                "chromosome": str(axis["chromosome"]), "radius_cm": radius,
                "width_cm": 2 * radius, "offset_cm": 0.0, "seed": SEED,
                "algorithm_version": ALGORITHM_VERSION,
                "four_control_receipt": descriptor(control_path),
                "four_control_batch": descriptor(control_batch),
                "event_axis": descriptor(event_axis),
                "event_axis_receipt": descriptor(axis_receipt_path),
                "training_authorized": False, "selection_authorized": False,
                "evaluate_opened": False, "truth_loaded": False,
            }
            name = task_id + ".r" + scale
            path = out / "requests" / (name + ".json")
            path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
            tasks.append({"task_id": task_id, "density_task_id": name,
                          "radius_cm": radius, "request": descriptor(path)})
    require(len(tasks) == 36, "exact 36 density tasks required")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "THIRTY_SIX_RADIUS_SPECIFIC_DENSITY_PREPARE_REQUESTS",
        "scientific_design": {
            "radii_cm": list(RADII_CM), "width_rule": "width_cm=2*radius_cm",
            "offset_cm": 0.0, "seed": SEED,
            "operation": "PERMUTE_OBSERVED_DIPLOID_DOSES_WITHIN_PERSON_CHROMOSOME_CM_BIN_MASK_FIXED",
            "preserves": "PER_PERSON_PER_BIN_OBSERVED_DOSE_HISTOGRAM_CARRIER_COUNT_AND_COPY_SUM",
            "removes": "DOSE_TO_LOCUS_ASSIGNMENT_WITHIN_EACH_NONOVERLAPPING_BIN",
            "does_not_preserve": "EVERY_OVERLAPPING_NETWORK_WINDOW_OR_PER_LOCUS_FREQUENCY",
        },
        "tasks": tasks, "training_authorized": False, "selection_authorized": False,
        "evaluate_opened": False, "truth_loaded": False,
    }
    (out / "PLAN.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); os.umask(0o077)
    result = build(args.control_root, args.index_root, args.output_dir)
    print(json.dumps({"schema": result["schema"], "tasks": len(result["tasks"])}))


if __name__ == "__main__":
    main()
