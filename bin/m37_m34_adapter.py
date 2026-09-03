#!/usr/bin/env python3
"""Fail-closed adapter from M34 phased truth to TRACE unordered diploid truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import m34_labels_to_states, require


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt(truth_path: Path, f0_path: Path, output: Path) -> dict[str, object]:
    with np.load(truth_path, allow_pickle=False) as truth, np.load(f0_path, allow_pickle=False) as f0:
        require({"sample_key_sha256", "marker_pos", "labels"}.issubset(truth.files), "M34 truth members differ")
        require({"sample_key_sha256", "marker_pos", "F0"}.issubset(f0.files), "M34 F0 members differ")
        labels = np.ascontiguousarray(truth["labels"])
        states = m34_labels_to_states(labels)
        require(np.array_equal(truth["sample_key_sha256"], f0["sample_key_sha256"]) and
                np.array_equal(truth["marker_pos"], f0["marker_pos"]) and
                f0["F0"].shape[:3] == labels.shape, "M34 truth/F0 axes differ")
        payload = {"sample_key_sha256": np.ascontiguousarray(truth["sample_key_sha256"]),
                   "marker_pos": np.ascontiguousarray(truth["marker_pos"]),
                   "state_labels": np.ascontiguousarray(states),
                   "state_order": np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"])}
    require(not output.exists(), "refusing to overwrite M37 truth adapter output")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(output, payload)
    receipt = {"schema_version": "1.0.0", "stage": "M37_M34_TO_TRACE_TRUTH",
               "status": "PASS_PHASE_ORDER_MARGINALIZED", "input_truth_sha256": sha256(truth_path),
               "input_f0_sha256": sha256(f0_path), "output_sha256": sha256(output),
               "target_axes_checked": True, "rare_phase_assignment": "forbidden"}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m34-truth", type=Path, required=True)
    parser.add_argument("--m34-f0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adapt(args.m34_truth, args.m34_f0, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
