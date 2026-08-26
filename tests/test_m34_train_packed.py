#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_adaptive_sweep as sweep
import m34_train_packed as subject


CONTRACT = ROOT / "conf" / "m34_adaptive_sweep_contract.json"


def write_shard(path: Path, truth_path: Path, prefix: str, samples: int,
                markers: int, arm: str = "RE") -> None:
    ancestries, haplotypes, channels = 3, 2, 13
    keys = np.asarray([f"{prefix}-{index}".encode() for index in range(samples)], dtype="|S64")
    marker_pos = np.arange(100, 100 + markers, dtype=np.int64)
    labels = np.empty((samples, haplotypes, markers), dtype=np.int8)
    for sample in range(samples):
        for hap in range(haplotypes):
            labels[sample, hap] = (np.arange(markers) + sample + hap) % ancestries
    baseline = np.full((samples, haplotypes, markers, ancestries), 0.1, dtype=np.float32)
    for sample in range(samples):
        for hap in range(haplotypes):
            baseline[sample, hap, np.arange(markers), labels[sample, hap]] = 0.8
    rows = samples * markers
    lengths = (np.arange(rows) % 3) + 1
    row_ptr = np.concatenate(([0], np.cumsum(lengths))).astype(np.uint64)
    rng = np.random.default_rng(44 + samples)
    tokens = rng.normal(size=(int(row_ptr[-1]), channels)).astype(np.float32)
    if arm == "RD":
        columns = (0, 2, 3, 5, 6, 8, 9)
        tokens[:, columns] = 0
    np.savez_compressed(
        path, sample_key_sha256=keys,
        marker_chrom=np.asarray([b"22"] * markers, dtype="|S2"),
        marker_pos=marker_pos, marker_ref=np.asarray([b"A"] * markers, dtype="|S1"),
        marker_alt=np.asarray([b"G"] * markers, dtype="|S1"),
        marker_cM=np.arange(markers, dtype=np.float64) * 0.1,
        radius_cM=np.asarray([0.2], dtype=np.float32), rare_tokens=tokens,
        rare_mask=np.ones(len(tokens), dtype=np.uint8),
        rare_locus_index=np.arange(len(tokens), dtype=np.uint64), row_ptr=row_ptr,
        row_sample_index=np.repeat(np.arange(samples, dtype=np.uint32), markers),
        row_marker_index=np.tile(np.arange(markers, dtype=np.uint32), samples), F0=baseline,
    )
    np.savez_compressed(truth_path, sample_key_sha256=keys, marker_pos=marker_pos,
                        labels=labels)


class M34TrainPackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def fixture(self, base: Path, family: str, config_id: str) -> Namespace:
        fit, fit_truth = base / "fit.npz", base / "fit.truth.npz"
        valid, valid_truth = base / "valid.npz", base / "valid.truth.npz"
        write_shard(fit, fit_truth, "fit", 2, 4)
        write_shard(valid, valid_truth, "valid", 1, 4)
        manifest = base / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": "1.0.0", "ancestry_names": ["AFR", "EUR", "NAM"],
            "haplotypes": 2,
            "splits": {"FIT": [{"packed": str(fit), "truth": str(fit_truth)}],
                       "VALID": [{"packed": str(valid), "truth": str(valid_truth)}]},
        }), encoding="utf-8")
        contract = sweep.validate_contract(sweep.strict_json(CONTRACT))
        task = sweep.triage_plan(contract)["tasks"]
        selected = next(row for row in task if row["family"] == family and
                        row["config_id"] == config_id and row["arm"] == "RE")
        # Three updates are sufficient here to test the real optimizer/validation path.
        local_contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        local_contract["stages"]["triage"]["maximum_updates"] = 3
        local_contract["boundary_loss"] = {
            "target_transition_weight_share": 0.01,
            "formula": "beta=q*(1-p)/(p*(1-q))-1_fit_truth_only",
            "provenance": "M33_PRE4B",
        }
        local_contract_path = base / "contract.json"
        local_contract_path.write_text(json.dumps(local_contract), encoding="utf-8")
        selected["maximum_updates"] = 3
        task_path = base / "task.json"
        task_path.write_text(json.dumps(selected), encoding="utf-8")
        return Namespace(
            contract=local_contract_path, manifest=manifest, task=task_path,
            outdir=base / "output", threads=1, maximum_rows_per_batch=4,
            maximum_tokens_per_batch=100, validation_every=1,
        )

    def run_family(self, family: str, config_id: str):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), family, config_id)
            # The production validator intentionally fixes the declared budget.  For this
            # focused test, retain all other contract checks and substitute the local one.
            original = sweep.validate_contract
            sweep.validate_contract = lambda value: value
            try:
                receipt = subject.run(args)
            finally:
                sweep.validate_contract = original
            with np.load(args.outdir / "valid.prediction.npz", allow_pickle=False) as archive:
                probabilities = archive["probabilities"]
            self.assertEqual(receipt["status"], "PASS_TRAINED_VALID_ONLY")
            self.assertFalse(receipt["test_opened"])
            self.assertEqual(receipt["updates_executed"], 3)
            self.assertEqual(probabilities.shape, (1, 2, 4, 3))
            self.assertTrue(np.allclose(probabilities.sum(axis=3), 1.0, atol=1e-5))

    def test_real_linear_training_path(self):
        self.run_family("local_linear", "linear_r0")

    def test_real_cnn_training_path(self):
        self.run_family("residual_cnn_1d", "rescnn_r0")

    def test_test_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "manifest.json"
            path.write_text(json.dumps({
                "schema_version": "1", "ancestry_names": ["AFR", "EUR", "NAM"],
                "haplotypes": 2, "splits": {"FIT": [], "VALID": [], "TEST": []},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "FIT and VALID only"):
                subject.load_manifest(path)

    def test_row_batching_respects_both_limits(self):
        row_ptr = np.asarray([0, 2, 5, 9, 10], dtype=np.uint64)
        self.assertEqual(subject.plan_row_batches(row_ptr, 2, 6), [(0, 2), (2, 4)])

    def test_boundary_formula_reaches_frozen_weight_share(self):
        labels = np.asarray([[[0, 0, 0, 1, 1], [2, 2, 2, 2, 2]]], dtype=np.int8)
        result = subject.boundary_parameters([labels], 0.01)
        self.assertEqual(result["transition_cells"], 1)
        self.assertEqual(result["possible_cells"], 10)
        weighted = (result["transition_prevalence"] * result["transition_multiplier"] /
                    ((1 - result["transition_prevalence"]) +
                     result["transition_prevalence"] * result["transition_multiplier"]))
        self.assertAlmostEqual(weighted, 0.01, places=14)

    def test_transition_at_marker_tile_boundary_is_retained(self):
        first = ({"sample_key_sha256": np.asarray([b"s"], dtype="|S64"),
                  "marker_pos": np.asarray([10, 20])},
                 np.asarray([[[0, 0], [2, 2]]], dtype=np.int8))
        second = ({"sample_key_sha256": np.asarray([b"s"], dtype="|S64"),
                   "marker_pos": np.asarray([30, 40])},
                  np.asarray([[[1, 1], [2, 0]]], dtype=np.int8))
        attached = subject.attach_global_transitions([first, second])
        self.assertEqual(attached[1][2][0, 0, 0], 1)
        self.assertEqual(attached[1][2][0, 1, 0], 0)
        self.assertEqual(attached[1][2][0, 1, 1], 1)


if __name__ == "__main__":
    unittest.main()
