#!/usr/bin/env python3
"""Expose the frozen FLARE baseline in the common M34 scoring schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m33_safe_bridge_core as core


F0_MEMBERS = {
    "sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref",
    "marker_alt", "F0",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pack(f0_path: Path, marker_cm_path: Path, output: Path,
         ancestry_names: tuple[str, ...] = ("AFR", "EUR", "NAM")) -> None:
    require(not output.exists(), "refusing to overwrite baseline prediction")
    with np.load(f0_path, allow_pickle=False) as archive:
        require(set(archive.files) == F0_MEMBERS, "F0 members differ")
        f0 = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(marker_cm_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"marker_cM"}, "marker-cM members differ")
        marker_cm = np.ascontiguousarray(archive["marker_cM"], dtype="<f8")
    probabilities = np.ascontiguousarray(f0["F0"], dtype="<f4")
    require(
        probabilities.ndim == 4
        and probabilities.shape[:3]
        == (len(f0["sample_key_sha256"]), 2, len(f0["marker_pos"]))
        and probabilities.shape[3] == len(ancestry_names),
        "F0 dimensions differ",
    )
    require(
        marker_cm.shape == f0["marker_pos"].shape
        and np.all(np.isfinite(marker_cm))
        and np.all(np.diff(marker_cm) >= 0),
        "marker-cM axis differs",
    )
    require(
        np.all(np.isfinite(probabilities))
        and np.all(probabilities >= 0)
        and np.max(np.abs(probabilities.sum(axis=3) - 1.0)) <= 5e-6,
        "F0 probabilities differ",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    core.write_deterministic_npz(output, {
        "sample_key_sha256": f0["sample_key_sha256"],
        "marker_pos": f0["marker_pos"],
        "marker_cM": marker_cm,
        "ancestry_names": np.asarray(
            [name.encode("ascii") for name in ancestry_names], dtype="|S32"
        ),
        "probabilities": probabilities,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--marker-cm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pack(args.f0, args.marker_cm, args.output)
    print(json.dumps({"status": "PASS_BASELINE_SCORING_SCHEMA", "output": str(args.output)}))
