#!/usr/bin/env python3
"""Create the 12-case regional-layout v2 PREPARE batch from revised indices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RADII = (0.05, 0.2, 1.0)
KERNELS = {0.05: [3, 7], 0.2: [3, 21], 1.0: [3, 101]}
QUERY_STEP_CM = {"1": 0.009999973242978901, "22": 0.009998872144233416}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def build(index_root, output):
    root = Path(index_root)
    receipts = sorted(root.glob("*/receipt.json"))
    require(len(receipts) == 12, "exact twelve revised factorized receipts required")
    tasks = []
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get("schema") == "r02_lai_factorized_predictors_v1"
                and receipt.get("status") == "PREDICTOR_INDEX_COMPLETE_NOT_TRAINING"
                and receipt.get("event_annotation_schema") == "r02_lai_event_annotations_v1",
                "revised factorized receipt differs")
        index_path = receipt_path.parent / "indices.npz"
        require(sha256(index_path) == receipt["outputs_sha256"]["indices.npz"],
                "factorized index changed")
        chrom = str(receipt["chromosome"])
        require(chrom in QUERY_STEP_CM, "query-step evidence absent for chromosome " + chrom)
        configs = {str(radius): {"schema": "lai_query_multiscale_cnn_v1",
                                 "kernel_sizes": KERNELS[radius],
                                 "query_step_cm": QUERY_STEP_CM[chrom]}
                   for radius in RADII}
        tasks.append({
            "task_id": receipt["task_id"], "role": receipt["role"],
            "index_dir": str(receipt_path.parent.resolve()),
            "scales_cm": list(RADII), "core_chunks": 16, "depth": 2,
            "query_max_gap_cm": 0.01, "cnn_query_configs_by_scale": configs,
        })
    batch = {
        "schema": "r02_lai_regional_layout_batch_v2",
        "status": "PREPARE_ONLY_NO_TRAINING_NO_SELECTION", "tasks": tasks,
        "scientific_design": {
            "radii_cm": list(RADII), "common_radius_equals_rare_radius": True,
            "cnn_depth": 2, "cnn_kernel_sizes_by_radius":
                {str(key): value for key, value in KERNELS.items()},
            "query_step_cm_by_chromosome": QUERY_STEP_CM,
            "candidate_rule": "ENUMERATE_ALL_THREE_ON_FIT;SELECT_BY_FROZEN_RULE_LATER",
            "not_claimed": "OPTIMAL_RADIUS_OR_BIOLOGICAL_BOUNDARY",
        },
        "training_authorized": False, "selection_performed": False,
        "evaluate_opened": False,
    }
    path = Path(output)
    require(not path.exists() and path.parent.is_dir(), "fresh regional batch output required")
    path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = build(args.index_root, args.output)
    print(json.dumps({"schema": result["schema"], "tasks": len(result["tasks"])}))


if __name__ == "__main__":
    main()
