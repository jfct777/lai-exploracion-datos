#!/usr/bin/env python3
"""Bind physical marker positions to genetic coordinates for M37."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import require


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_sha256(marker_pos: np.ndarray, marker_cm: np.ndarray) -> str:
    """Hash the ordered position/cM pairs with dtype and shape delimiters."""
    digest = hashlib.sha256()
    for value in (np.asarray(marker_pos), np.asarray(marker_cm)):
        contiguous = np.ascontiguousarray(value)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def verify_joint_source_receipt(f0_path: Path, marker_cm_path: Path,
                                receipt_path: Path) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("stage") in {"M34_PARSE_FLARE_F0", "M37_MARKER_AXIS_SOURCE"},
            "marker-axis source receipt stage differs")
    outputs = receipt.get("outputs")
    require(isinstance(outputs, dict) and
            outputs.get(f0_path.name, {}).get("sha256") == sha256(f0_path) and
            outputs.get(marker_cm_path.name, {}).get("sha256") == sha256(marker_cm_path),
            "marker-axis source receipt does not bind F0 and marker cM")
    return receipt


def bind_marker_axis(f0_path: Path, marker_cm_path: Path, output: Path,
                     source_receipt_path: Path | None = None) -> dict[str, object]:
    source_receipt = (verify_joint_source_receipt(f0_path, marker_cm_path, source_receipt_path)
                      if source_receipt_path is not None else None)
    with np.load(f0_path, allow_pickle=False) as f0, np.load(marker_cm_path, allow_pickle=False) as marker:
        require({"marker_pos", "F0"}.issubset(f0.files), "F0 lacks its physical marker axis")
        require("marker_cM" in marker.files, "genetic marker archive lacks marker_cM")
        marker_pos = np.ascontiguousarray(f0["marker_pos"])
        marker_cm = np.ascontiguousarray(marker["marker_cM"], dtype=np.float64)
        require(marker_pos.ndim == marker_cm.ndim == 1 and len(marker_pos) == len(marker_cm) > 0,
                "physical/genetic marker axes differ")
        require(np.issubdtype(marker_pos.dtype, np.integer) and np.all(np.diff(marker_pos) > 0),
                "physical marker positions must be strictly increasing")
        require(np.isfinite(marker_cm).all() and np.all(np.diff(marker_cm) >= 0),
                "genetic marker coordinates must be finite and nondecreasing")
        require(f0["F0"].ndim == 4 and f0["F0"].shape[2] == len(marker_pos),
                "F0 tensor and physical marker axis differ")
        if "marker_pos" in marker.files:
            require(np.array_equal(marker["marker_pos"], marker_pos),
                    "marker archive physical axis differs from F0")
            binding_evidence = "SELF_CONTAINED_POSITION_CM_PAIRS"
        else:
            require(source_receipt is not None,
                    "marker cM without marker_pos needs a joint F0/marker source receipt")
            binding_evidence = "JOINT_UPSTREAM_F0_MARKER_RECEIPT"
        if source_receipt is not None:
            require(int(source_receipt.get("marker_count", -1)) == len(marker_pos),
                    "marker-axis source receipt count differs")
        pair_digest = axis_sha256(marker_pos, marker_cm)
        payload = {
            "marker_pos": marker_pos,
            "marker_cM": marker_cm,
            "marker_axis_sha256": np.asarray([pair_digest]),
        }
    require(not output.exists(), "refusing to overwrite bound marker axis")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(output, payload)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M37_BIND_MARKER_AXIS",
        "status": "PASS_PHYSICAL_GENETIC_AXIS_BOUND",
        "f0_source_sha256": sha256(f0_path),
        "marker_cm_source_sha256": sha256(marker_cm_path),
        "marker_axis_sha256": pair_digest,
        "marker_count": int(len(marker_pos)),
        "binding_evidence": binding_evidence,
        "source_receipt_sha256": sha256(source_receipt_path) if source_receipt_path else None,
        "source_receipt_stage": source_receipt.get("stage") if source_receipt else None,
        "output_sha256": sha256(output),
    }
    output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--marker-cm", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bind_marker_axis(args.f0, args.marker_cm, args.output,
                                      args.source_receipt), sort_keys=True))


if __name__ == "__main__":
    main()
