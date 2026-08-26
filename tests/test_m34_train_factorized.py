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
import m34_materialize as materialize
import m34_train_factorized as subject


CONTRACT = ROOT / "conf" / "m34_adaptive_sweep_contract.json"


def arrays(prefix: str, sample_count: int) -> tuple[dict[str, np.ndarray], ...]:
    loci = np.asarray([10, 11, 12], dtype="<u8")
    keys = np.asarray(
        [f"{prefix}-{index}".encode() for index in range(sample_count)], dtype="|S64",
    )
    selected = {
        "locus_id": loci,
        "chrom": np.full(3, 22, dtype="|u1"),
        "pos": np.asarray([100, 200, 300], dtype="<i8"),
        "ref": np.asarray([b"A", b"C", b"G"], dtype="|S1"),
        "alt": np.asarray([b"T", b"G", b"A"], dtype="|S1"),
        "cM": np.asarray([0.05, 0.10, 0.20], dtype="<f8"),
    }
    dosage = np.empty((sample_count, 3), dtype="|i1")
    for sample in range(sample_count):
        dosage[sample] = (np.arange(3) + sample) % 3
    target = {
        "sample_key_sha256": keys,
        "locus_id": loci,
        "minor_dosage": dosage,
        "observed_mask": np.ones((sample_count, 3), dtype="|u1"),
    }
    ac = np.asarray([[1, 2, 0], [2, 1, 1], [0, 1, 3]], dtype="<u2")
    an = np.full((3, 3), 8, dtype="<u2")
    reference = {
        "ancestry": np.asarray([b"AFR", b"EUR", b"NAM"], dtype="|S4"),
        "locus_id": loci,
        "minor_ac": ac,
        "callable_an": an,
        "minor_af": ac.astype("<f8") / an,
        "observed_mask": np.ones((3, 3), dtype="|u1"),
        "no_support": (ac == 0).astype("|u1"),
    }
    marker_pos = np.asarray([150, 250], dtype="<i8")
    labels = np.empty((sample_count, 2, 2), dtype="|i1")
    for sample in range(sample_count):
        labels[sample, 0] = [sample % 3, (sample + 1) % 3]
        labels[sample, 1] = [(sample + 2) % 3, (sample + 2) % 3]
    f0_values = np.full((sample_count, 2, 2, 3), 0.1, dtype="<f4")
    for sample in range(sample_count):
        for haplotype in range(2):
            f0_values[sample, haplotype, np.arange(2), labels[sample, haplotype]] = 0.8
    f0 = {
        "sample_key_sha256": keys,
        "marker_chrom": np.full(2, 22, dtype="|u1"),
        "marker_pos": marker_pos,
        "marker_ref": np.asarray([b"A", b"C"], dtype="|S1"),
        "marker_alt": np.asarray([b"G", b"T"], dtype="|S1"),
        "F0": f0_values,
    }
    marker = {"marker_cM": np.asarray([0.10, 0.15], dtype="<f8")}
    truth = {"sample_key_sha256": keys, "marker_pos": marker_pos, "labels": labels}
    return selected, target, reference, f0, marker, truth


def write_factor_row(base: Path, prefix: str, sample_count: int) -> dict[str, str]:
    names = ("selected_variant", "target", "reference", "f0", "marker_cm", "truth")
    result: dict[str, str] = {}
    for name, value in zip(names, arrays(prefix, sample_count)):
        path = base / f"{prefix}.{name}.npz"
        np.savez_compressed(path, **value)
        result[name] = str(path)
    return result


def local_task(base: Path, arm: str) -> tuple[Path, Path]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["stages"]["triage"]["maximum_updates"] = 2
    contract["training"]["batch_people"] = 2
    contract["training"]["marker_shard_size"] = 1
    contract["training"]["maximum_rows_per_microbatch"] = 4
    contract["training"]["maximum_padded_tokens_per_microbatch"] = 100
    contract["training"]["validation_every_updates"] = 1
    contract_path = base / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    declared = sweep.triage_plan(sweep.validate_contract(
        sweep.strict_json(CONTRACT),
    ))["tasks"]
    task = next(row for row in declared
                if row["family"] == "local_linear" and
                row["config_id"] == "linear_r0" and row["arm"] == arm)
    task["maximum_updates"] = 2
    task_path = base / f"task.{arm}.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return contract_path, task_path


def fixture(base: Path, arm: str = "RE") -> Namespace:
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "ancestry_names": ["AFR", "EUR", "NAM"],
        "haplotypes": 2,
        "rotation": "R0",
        "splits": {
            "FIT": [write_factor_row(base, "fit", 3)],
            "VALID": [write_factor_row(base, "valid", 2)],
        },
    }), encoding="utf-8")
    contract_path, task_path = local_task(base, arm)
    return Namespace(
        contract=contract_path,
        manifest=manifest_path,
        task=task_path,
        outdir=base / f"output-{arm}",
        device="cpu",
        threads=1,
        sample_shard_size=2,
        marker_shard_size=1,
        maximum_rows_per_batch=4,
        maximum_tokens_per_batch=100,
        validation_every=1,
    )


class M34FactorizedTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def run_fixture(self, args: Namespace) -> dict:
        original = sweep.validate_contract
        sweep.validate_contract = lambda value: value
        try:
            return subject.run(args)
        finally:
            sweep.validate_contract = original

    def test_real_factorized_training_has_no_expanded_input_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory), "RE")
            receipt = self.run_fixture(args)
            self.assertEqual(receipt["status"],
                             "PASS_TRAINED_VALID_ONLY_FACTORIZED_LAZY")
            self.assertFalse(receipt["test_opened"])
            self.assertFalse(receipt["expanded_input_artifacts_written"])
            self.assertEqual(receipt["fit_sample_count"], 3)
            self.assertEqual(receipt["valid_sample_count"], 2)
            self.assertEqual(receipt["updates_executed"], 2)
            self.assertEqual({path.name for path in args.outdir.iterdir()},
                             {"model.pt", "valid.prediction.npz", "train.receipt.json"})
            with np.load(args.outdir / "valid.prediction.npz", allow_pickle=False) as archive:
                probabilities = archive["probabilities"]
            self.assertEqual(probabilities.shape, (2, 2, 2, 3))
            self.assertTrue(np.allclose(probabilities.sum(axis=3), 1.0, atol=1e-5))

    def test_rd_re_are_derived_together_and_only_declared_channels_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory), "RE")
            manifest = subject.load_manifest(args.manifest)
            bundles = subject.load_bundles(manifest)
            maxima = materialize.derive_fit_max_callable(
                [bundles["FIT"][0].reference], manifest["ancestry_names"],
            )
            shard = subject.plan_shards(bundles["FIT"], 2, 1)[0]
            rd_view, re_view, labels, transitions = subject.build_pair(
                bundles["FIT"][0], shard, manifest["ancestry_names"], maxima, 0.2,
            )
            canonical_rd, canonical_re = materialize.build_paired_shard(
                bundles["FIT"][0].selected, bundles["FIT"][0].target,
                bundles["FIT"][0].reference, bundles["FIT"][0].f0,
                bundles["FIT"][0].marker_cm, manifest["ancestry_names"], maxima,
                0.2, shard.sample_start, shard.sample_end_exclusive,
                shard.marker_start, shard.marker_end_exclusive,
            )
            for name in canonical_re:
                np.testing.assert_array_equal(re_view[name], canonical_re[name], err_msg=name)
                np.testing.assert_array_equal(rd_view[name], canonical_rd[name], err_msg=name)
            materialize.validate_control_pair(rd_view, re_view, manifest["ancestry_names"])
            rare = materialize.rare_value_channel_indices(3)
            kept = tuple(index for index in range(13) if index not in rare)
            self.assertTrue(np.all(rd_view["rare_tokens"][:, rare] == 0))
            np.testing.assert_array_equal(rd_view["rare_tokens"][:, kept],
                                          re_view["rare_tokens"][:, kept])
            self.assertEqual(labels.shape, transitions.shape)

    def test_arm_pair_uses_same_pairing_key_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            re_args = fixture(base, "RE")
            re_receipt = self.run_fixture(re_args)
            _contract, rd_task = local_task(base, "RD")
            rd_args = Namespace(**{**vars(re_args), "task": rd_task,
                                   "outdir": base / "output-RD"})
            rd_receipt = self.run_fixture(rd_args)
            self.assertEqual(re_receipt["paired_task_sha256_without_arm"],
                             rd_receipt["paired_task_sha256_without_arm"])
            self.assertEqual(re_receipt["rd_zero_channel_indices"],
                             rd_receipt["rd_zero_channel_indices"])
            self.assertEqual(re_receipt["fit_lazy_shard_count"],
                             rd_receipt["fit_lazy_shard_count"])

    def test_manifest_rejects_test_and_rotation_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            invalid = base / "invalid.json"
            invalid.write_text(json.dumps({
                "schema_version": "1.0.0",
                "ancestry_names": ["AFR", "EUR", "NAM"],
                "haplotypes": 2,
                "rotation": "R0",
                "splits": {"FIT": [], "VALID": [], "TEST": []},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "FIT and VALID only"):
                subject.load_manifest(invalid)
            args = fixture(base, "RE")
            payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            payload["rotation"] = "R1"
            args.manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rotations differ"):
                self.run_fixture(args)

    def test_sample_and_marker_shards_are_variable_and_gap_free(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory), "RE")
            manifest = subject.load_manifest(args.manifest)
            bundles = subject.load_bundles(manifest)
            shards = subject.plan_shards(bundles["FIT"], 2, 1)
            observed = {
                (shard.sample_start, shard.sample_end_exclusive,
                 shard.marker_start, shard.marker_end_exclusive)
                for shard in shards
            }
            self.assertEqual(observed, {
                (0, 2, 0, 1), (0, 2, 1, 2),
                (2, 3, 0, 1), (2, 3, 1, 2),
            })

    def test_sample_shard_size_must_match_frozen_batch_people(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory), "RE")
            args.sample_shard_size = 3
            with self.assertRaisesRegex(ValueError, "batch_people"):
                self.run_fixture(args)

    def test_marker_microbatch_and_validation_controls_are_frozen(self):
        mutations = {
            "marker_shard_size": 2,
            "maximum_rows_per_batch": 5,
            "maximum_tokens_per_batch": 101,
            "validation_every": 2,
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = fixture(Path(directory), "RE")
                setattr(args, field, value)
                with self.assertRaisesRegex(ValueError, "frozen"):
                    self.run_fixture(args)

if __name__ == "__main__":
    unittest.main()
