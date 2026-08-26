#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_pack_f0_prediction as subject


class M34PackF0PredictionTests(unittest.TestCase):
    def test_known_answer_uses_common_scoring_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = np.asarray([b"sample"], dtype="|S64")
            probabilities = np.full((1, 2, 2, 3), np.float32(1 / 3), dtype="<f4")
            f0 = root / "f0.npz"
            marker = root / "marker.npz"
            output = root / "prediction.npz"
            np.savez(
                f0, sample_key_sha256=keys, marker_chrom=np.asarray([22, 22], dtype="|u1"),
                marker_pos=np.asarray([100, 200], dtype="<i8"),
                marker_ref=np.asarray([b"A", b"C"], dtype="|S1"),
                marker_alt=np.asarray([b"G", b"T"], dtype="|S1"), F0=probabilities,
            )
            np.savez(marker, marker_cM=np.asarray([0.1, 0.2], dtype="<f8"))
            subject.pack(f0, marker, output)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {
                    "sample_key_sha256", "marker_pos", "marker_cM",
                    "ancestry_names", "probabilities",
                })
                np.testing.assert_array_equal(archive["probabilities"], probabilities)
                np.testing.assert_array_equal(archive["ancestry_names"], [b"AFR", b"EUR", b"NAM"])

    def test_axis_and_probability_drift_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            f0 = root / "f0.npz"
            marker = root / "marker.npz"
            np.savez(
                f0, sample_key_sha256=np.asarray([b"sample"], dtype="|S64"),
                marker_chrom=np.asarray([22], dtype="|u1"),
                marker_pos=np.asarray([100], dtype="<i8"),
                marker_ref=np.asarray([b"A"], dtype="|S1"),
                marker_alt=np.asarray([b"G"], dtype="|S1"),
                F0=np.ones((1, 2, 1, 3), dtype="<f4"),
            )
            np.savez(marker, marker_cM=np.asarray([0.1], dtype="<f8"))
            with self.assertRaisesRegex(ValueError, "probabilities"):
                subject.pack(f0, marker, root / "out.npz")


if __name__ == "__main__":
    unittest.main()
