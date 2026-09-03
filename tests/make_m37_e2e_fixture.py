#!/usr/bin/env python3
"""Create a tiny deterministic AFR/EUR/NAM fixture for the M37 real-config test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-prefix", default="fixture-person")
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)

    people, markers = 8, 13
    sample_keys = np.asarray([f"{args.sample_prefix}-{index:02d}".encode("ascii") for index in range(people)])
    marker_pos = np.arange(10_000, 10_000 + 1_000 * markers, 1_000, dtype=np.int64)
    marker_cm = np.arange(markers, dtype=np.float64) / 100.0
    locus_id = np.asarray([101, 102, 103, 104], dtype=np.int64)
    locus_cm = np.asarray([.015, .045, .075, .105], dtype=np.float64)

    dosage = np.asarray([
        [0, 1, 0, 2], [1, 0, 0, 1], [0, 0, 1, 0], [2, 1, 0, 0],
        [0, 1, 1, 0], [1, 0, 2, 0], [0, 0, 1, 1], [1, 1, 0, 0],
    ], dtype=np.int8)
    observed = np.ones_like(dosage, dtype=np.uint8)
    observed[2, 3] = 0
    dosage[2, 3] = 0

    haploid = np.full((people, 2, markers, 3), .05, dtype=np.float32)
    for person in range(people):
        first = person % 3
        second = (person // 3) % 3
        haploid[person, 0, :, first] = .90
        haploid[person, 1, :, second] = .90
    haploid /= haploid.sum(axis=3, keepdims=True)
    state_index = np.asarray(((0, 1, 2), (1, 3, 4), (2, 4, 5)), dtype=np.uint8)
    truth = np.empty((people, markers), dtype=np.uint8)
    for person in range(people):
        truth[person] = state_index[person % 3, (person // 3) % 3]

    np.savez(root / "selected.npz", locus_id=locus_id, cM=locus_cm,
             context_7mer=np.asarray([17, 31, 49, 63], dtype=np.uint16),
             carrier_support=np.asarray([.7, .6, .8, .5], dtype=np.float32),
             origin_support=np.asarray([.4, .5, .3, .6], dtype=np.float32))
    np.savez(root / "target.npz", sample_key_sha256=sample_keys, locus_id=locus_id,
             minor_dosage=dosage, observed_mask=observed)
    np.savez(root / "reference.npz", ancestry=np.asarray(["AFR", "EUR", "NAM"]),
             locus_id=locus_id,
             minor_ac=np.asarray([[1, 3, 1, 2], [4, 1, 2, 1], [1, 2, 5, 3]], dtype=np.int16),
             callable_an=np.full((3, len(locus_id)), 40, dtype=np.int16))
    np.savez(root / "f0.npz", sample_key_sha256=sample_keys, marker_pos=marker_pos,
             F0=haploid, baseline_method=np.asarray(["fixture_F0"]))
    np.savez(root / "marker_axis_source.npz", marker_pos=marker_pos, marker_cM=marker_cm)
    np.savez(root / "truth.npz", sample_key_sha256=sample_keys, marker_pos=marker_pos,
             state_labels=truth, state_order=np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"]))
    f0_path, marker_path = root / "f0.npz", root / "marker_axis_source.npz"
    receipt = {
        "schema_version": "1.0.0", "stage": "M37_MARKER_AXIS_SOURCE",
        "marker_count": markers,
        "outputs": {
            f0_path.name: {"sha256": hashlib.sha256(f0_path.read_bytes()).hexdigest()},
            marker_path.name: {"sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest()},
        },
    }
    (root / "f0.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
