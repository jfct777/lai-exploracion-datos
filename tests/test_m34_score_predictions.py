#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_score_predictions as subject


class M34ScoringTests(unittest.TestCase):
    def write_pair(self, directory: Path, labels: np.ndarray,
                   hard: np.ndarray | None = None) -> tuple[Path, Path]:
        samples, haplotypes, markers = labels.shape
        ancestries = ("AFR", "EUR", "NAM")
        probabilities = np.eye(len(ancestries), dtype=np.float32)[labels]
        if hard is not None:
            probabilities = np.eye(len(ancestries), dtype=np.float32)[hard]
        keys = np.asarray([f"sample-{index}".encode() for index in range(samples)], dtype="|S64")
        positions = np.arange(100, 100 + markers, dtype=np.int64)
        prediction = directory / "prediction.npz"
        truth = directory / "truth.npz"
        np.savez_compressed(
            prediction, sample_key_sha256=keys, marker_pos=positions,
            marker_cM=np.arange(markers, dtype=np.float64) * 0.1,
            ancestry_names=np.asarray(ancestries, dtype="|S32"),
            probabilities=probabilities,
        )
        np.savez_compressed(truth, sample_key_sha256=keys, marker_pos=positions,
                            labels=labels.astype(np.int8))
        return prediction, truth

    def test_perfect_known_answer(self):
        labels = np.asarray([[[0, 0, 1, 1, 2], [2, 2, 1, 1, 0]]], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            prediction, truth = self.write_pair(Path(directory), labels)
            result = subject.score(prediction, truth)
        self.assertEqual(result["macro_ancestry_dose_MAE"], 0.0)
        self.assertEqual(result["NAM_truth_present_MAE"], 0.0)
        self.assertEqual(result["haplotype_Brier"], 0.0)
        for value in result["boundary"].values():
            self.assertEqual(value["f1"], 1.0)
            self.assertEqual(value["false_transitions_per_cM"], 0.0)

    def test_boundary_tolerance_and_false_transition_known_answer(self):
        labels = np.asarray([[[0, 0, 0, 1, 1, 1], [2, 2, 2, 2, 2, 2]]], dtype=np.int8)
        hard = labels.copy()
        hard[0, 0] = [0, 0, 0, 0, 1, 2]
        with tempfile.TemporaryDirectory() as directory:
            prediction, truth = self.write_pair(Path(directory), labels, hard)
            result = subject.score(prediction, truth)
        # The true 0->1 boundary and predicted 0->1 boundary differ by exactly 0.1 cM.
        self.assertEqual(result["boundary"]["0.1"]["matched"], 1)
        self.assertEqual(result["boundary"]["0.1"]["predicted"], 2)
        self.assertGreater(result["boundary"]["0.1"]["false_transitions_per_cM"], 0.0)
        self.assertGreater(result["macro_ancestry_dose_MAE"], 0.0)
        self.assertGreater(result["haplotype_Brier"], 0.0)

    def test_axis_mismatch_fails_closed(self):
        labels = np.zeros((1, 2, 3), dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            prediction, truth = self.write_pair(Path(directory), labels)
            with np.load(truth, allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            payload["marker_pos"] = payload["marker_pos"] + 1
            np.savez_compressed(truth, **payload)
            with self.assertRaisesRegex(ValueError, "marker axes"):
                subject.score(prediction, truth)


if __name__ == "__main__":
    unittest.main()
