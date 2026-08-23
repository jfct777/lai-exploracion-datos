#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m33_materialize", ROOT / "bin" / "m33_materialize.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixture():
    loci = np.asarray([10, 11, 12], dtype="<u8")
    samples = np.asarray([b"a" * 64, b"b" * 64], dtype="|S64")
    selected = {
        "locus_id": loci, "chrom": np.full(3, 22, dtype="|u1"),
        "pos": np.asarray([100, 200, 300], dtype="<i8"),
        "ref": np.asarray([b"A", b"C", b"G"], dtype="|S1"),
        "alt": np.asarray([b"T", b"G", b"A"], dtype="|S1"),
        "cM": np.asarray([0.05, 0.10, 0.20], dtype="<f8"),
    }
    target = {
        "sample_key_sha256": samples, "locus_id": loci,
        "minor_dosage": np.asarray([[2, 1, 0], [0, 0, 2]], dtype="|i1"),
        "observed_mask": np.asarray([[1, 1, 1], [1, 0, 1]], dtype="|u1"),
    }
    ac = np.asarray([[1, 0, 3], [2, 1, 0], [0, 2, 1]], dtype="<u2")
    an = np.full((3, 3), 4, dtype="<u2")
    reference = {
        "ancestry": np.asarray([b"AFR", b"EUR", b"ASIA"], dtype="|S4"), "locus_id": loci,
        "minor_ac": ac, "callable_an": an, "minor_af": ac.astype("<f8") / an,
        "observed_mask": np.ones((3, 3), dtype="|u1"),
        "no_support": (ac == 0).astype("|u1"),
    }
    f0_values = np.full((2, 2, 2, 3), np.float32(1 / 3), dtype="<f4")
    f0 = {
        "sample_key_sha256": samples, "marker_chrom": np.full(2, 22, dtype="|u1"),
        "marker_pos": np.asarray([150, 250], dtype="<i8"),
        "marker_ref": np.asarray([b"A", b"C"], dtype="|S1"),
        "marker_alt": np.asarray([b"G", b"T"], dtype="|S1"), "F0": f0_values,
    }
    return selected, target, reference, f0, np.asarray([0.10, 0.15], dtype="<f8")


class MaterializeOracleTests(unittest.TestCase):
    def test_exact_channels_and_packed_rows(self):
        selected, target, reference, f0, marker_cm = fixture()
        shard = MODULE.build_packed_shard(
            selected, target, reference, f0, marker_cm,
            {"AFR": 4, "EUR": 4, "ASIA": 4}, 0.05, 0, 2, 0, 2,
        )
        self.assertEqual(shard["rare_tokens"].shape, (8, 13))
        np.testing.assert_array_equal(shard["row_ptr"], [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(shard["row_sample_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(shard["row_marker_index"], [0, 1, 0, 1])
        np.testing.assert_array_equal(shard["rare_locus_index"], [0, 1, 1, 2, 0, 1, 1, 2])
        np.testing.assert_allclose(shard["rare_tokens"][0, :2], [1.0, 1.0], rtol=0, atol=0)
        np.testing.assert_allclose(shard["rare_tokens"][4, :2], [0.0, 1.0], rtol=0, atol=0)
        np.testing.assert_allclose(shard["rare_tokens"][6, :2], [0.0, 0.0], rtol=0, atol=0)
        np.testing.assert_allclose(shard["rare_tokens"][:2, 11], [-1.0, 0.0], rtol=0, atol=1e-7)
        np.testing.assert_allclose(shard["rare_tokens"][:2, 12], [0.0, 1.0], rtol=0, atol=1e-7)

    def test_technical_kat_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "technical.npz"
            np.savez(path, locus_key_sha256=np.asarray([b"a" * 64], dtype="|S64"),
                     chrom=np.asarray([22], dtype="|u1"), pos=np.asarray([1], dtype="<i8"),
                     ref=np.asarray([b"A"], dtype="|S1"), alt=np.asarray([b"C"], dtype="|S1"),
                     cM=np.asarray([0.1], dtype="<f8"), minor_code=np.asarray([0], dtype="|i1"))
            with self.assertRaisesRegex(ValueError, "productive schema"):
                MODULE.load_productive_npz(path, "selected")

    def test_missingness_and_ref_semantics_fail_closed(self):
        selected, target, reference, f0, marker_cm = fixture()
        target["minor_dosage"][1, 1] = 1
        with self.assertRaisesRegex(ValueError, "missing TARGET"):
            MODULE.validate_inputs(selected, target, reference, f0, marker_cm)
        target["minor_dosage"][1, 1] = 0
        reference["minor_af"][0, 0] = 0.99
        with self.assertRaisesRegex(ValueError, "AC/AN/AF"):
            MODULE.validate_inputs(selected, target, reference, f0, marker_cm)

    def test_storage_estimate_matches_root17_scale(self):
        assignments = {0.05: 36_229_121, 0.1: 58_587_636, 0.2: 92_883_536, 0.5: 175_066_132}
        result = MODULE.estimate_packed_storage(assignments, 30)
        self.assertEqual(result["total_token_array_bytes"], 663_862_557_750)
        self.assertEqual(result["minimum_shards"], 41_520)
        nine = MODULE.estimate_packed_storage(assignments, 30, copies=9)
        self.assertEqual(nine["total_token_array_bytes"], 9 * result["total_token_array_bytes"])
        self.assertEqual(nine["minimum_shards"], 9 * result["minimum_shards"])

    def test_unfrozen_radius_and_oversized_shard_fail(self):
        selected, target, reference, f0, marker_cm = fixture()
        with self.assertRaisesRegex(ValueError, "radius"):
            MODULE.build_packed_shard(selected, target, reference, f0, marker_cm,
                                      {"AFR": 4, "EUR": 4, "ASIA": 4}, 0.3, 0, 2, 0, 2)

    def test_factorized_lazy_view_is_byte_exact_to_oracle(self):
        selected, target, reference, f0, marker_cm = fixture()
        intervals = MODULE.build_interval_table(selected["cM"], marker_cm)
        normalization = {"AFR": 4, "EUR": 4, "ASIA": 4}
        for radius in MODULE.RADII:
            oracle = MODULE.build_packed_shard(
                selected, target, reference, f0, marker_cm, normalization,
                radius, 0, 2, 0, 2,
            )
            lazy = MODULE.build_lazy_packed_shard(
                selected, target, reference, f0, marker_cm, intervals, normalization,
                radius, 0, 2, 0, 2,
            )
            self.assertEqual(set(oracle), set(lazy))
            for name in oracle:
                self.assertEqual(oracle[name].dtype, lazy[name].dtype, name)
                self.assertEqual(oracle[name].tobytes(order="C"), lazy[name].tobytes(order="C"), name)

            # Fixed consumer: equality reaches predictions, loss and gradient.
            weights = np.arange(39, dtype="<f4").reshape(13, 3) / np.float32(39)
            logits_oracle = oracle["rare_tokens"] @ weights
            logits_lazy = lazy["rare_tokens"] @ weights
            np.testing.assert_array_equal(logits_oracle, logits_lazy)
            loss_oracle = np.mean(logits_oracle * logits_oracle, dtype="<f8")
            loss_lazy = np.mean(logits_lazy * logits_lazy, dtype="<f8")
            self.assertEqual(loss_oracle, loss_lazy)
            scale = np.float32(2.0 / logits_oracle.size)
            gradient_oracle = scale * (logits_oracle @ weights.T)
            gradient_lazy = scale * (logits_lazy @ weights.T)
            np.testing.assert_array_equal(gradient_oracle, gradient_lazy)

    def test_interval_table_is_inclusive_nested_and_supports_empty_contexts(self):
        selected, target, reference, f0, marker_cm = fixture()
        intervals = MODULE.build_interval_table(selected["cM"], marker_cm)
        MODULE.validate_interval_table(intervals, selected["cM"], marker_cm)
        np.testing.assert_array_equal(intervals["context_start"][0], [0, 1])
        np.testing.assert_array_equal(intervals["context_stop"][0], [2, 3])
        self.assertTrue(np.all(intervals["context_start"][1:] <=
                               intervals["context_start"][:-1]))
        self.assertTrue(np.all(intervals["context_stop"][1:] >=
                               intervals["context_stop"][:-1]))

        far_marker_cm = np.asarray([10.0, 11.0], dtype="<f8")
        far = MODULE.build_interval_table(selected["cM"], far_marker_cm)
        lazy = MODULE.build_lazy_packed_shard(
            selected, target, reference, f0, far_marker_cm, far,
            {"AFR": 4, "EUR": 4, "ASIA": 4}, 0.05, 0, 2, 0, 2,
        )
        self.assertEqual(lazy["rare_tokens"].shape, (0, 13))
        np.testing.assert_array_equal(lazy["row_ptr"], np.zeros(5, dtype="<u8"))
        corrupted = {name: value.copy() for name, value in intervals.items()}
        corrupted["context_start"][:] = 0
        corrupted["context_stop"][:] = 0
        with self.assertRaisesRegex(ValueError, "content differs"):
            MODULE.validate_interval_table(corrupted, selected["cM"], marker_cm)

    def test_fit_normalization_rejects_score_and_eval_leakage(self):
        _, _, reference, _, _ = fixture()
        fit_a = {name: value.copy() for name, value in reference.items()}
        fit_b = {name: value.copy() for name, value in reference.items()}
        fit_b["callable_an"][:, :] = np.asarray([[8], [10], [12]], dtype="<u2")
        fit_b["minor_af"] = fit_b["minor_ac"].astype("<f8") / fit_b["callable_an"]
        fit_b["observed_mask"][:] = 1
        fit_b["no_support"] = (fit_b["minor_ac"] == 0).astype("|u1")
        observed = MODULE.derive_fit_max_callable(
            {2024931463: fit_a, 1324432253: fit_b}, "R0",
            {2024931463: "a" * 64, 1324432253: "b" * 64}, [201, 202],
        )
        self.assertEqual(observed, {"AFR": 8, "EUR": 10, "ASIA": 12})
        with self.assertRaisesRegex(ValueError, "exactly the FIT"):
            MODULE.derive_fit_max_callable(
                {2024931463: fit_a, 1324432253: fit_b, 386357765: reference}, "R0",
                {2024931463: "a" * 64, 1324432253: "b" * 64}, [201],
            )
        with self.assertRaisesRegex(ValueError, "EVAL root"):
            MODULE.derive_fit_max_callable(
                {2024931463: fit_a, 1324432253: fit_b}, "R0",
                {2024931463: "a" * 64, 1324432253: "b" * 64}, [1324432253],
            )

    def test_prepared_batch_reuse_and_deterministic_microbatch_plan(self):
        selected, target, reference, f0, marker_cm = fixture()
        normalization = {"AFR": 4, "EUR": 4, "ASIA": 4}
        intervals = MODULE.build_interval_table(selected["cM"], marker_cm)
        manifest_sha = "c" * 64
        prepared = MODULE.prepare_person_batch_channels(
            target, reference, normalization, 0, 2, root_seed=20260817,
            rotation_id="TECHNICAL_KAT", fit_normalization_manifest_sha256=manifest_sha,
        )
        direct = MODULE.build_lazy_packed_shard(
            selected, target, reference, f0, marker_cm, intervals, normalization,
            0.05, 0, 2, 0, 2,
        )
        reused = MODULE.build_lazy_packed_shard(
            selected, target, reference, f0, marker_cm, intervals, normalization,
            0.05, 0, 2, 0, 2, prepared_channels=prepared,
            expected_root_seed=20260817, expected_rotation_id="TECHNICAL_KAT",
            expected_fit_normalization_manifest_sha256=manifest_sha,
        )
        for name in direct:
            self.assertEqual(direct[name].tobytes(order="C"), reused[name].tobytes(order="C"))
        prepared_second = MODULE.prepare_person_batch_channels(
            target, reference, normalization, 1, 2, root_seed=20260817,
            rotation_id="TECHNICAL_KAT", fit_normalization_manifest_sha256=manifest_sha,
        )
        direct_second = MODULE.build_lazy_packed_shard(
            selected, target, reference, f0, marker_cm, intervals, normalization,
            0.05, 1, 2, 0, 2,
        )
        reused_second = MODULE.build_lazy_packed_shard(
            selected, target, reference, f0, marker_cm, intervals, normalization,
            0.05, 1, 2, 0, 2, prepared_channels=prepared_second,
            expected_root_seed=20260817, expected_rotation_id="TECHNICAL_KAT",
            expected_fit_normalization_manifest_sha256=manifest_sha,
        )
        for name in direct_second:
            self.assertEqual(direct_second[name].tobytes(order="C"),
                             reused_second[name].tobytes(order="C"))
        plan = MODULE.plan_lazy_marker_chunks(
            intervals, selected["cM"], marker_cm, 0.05, 2,
            central_marker_block=2, token_budget=4,
        )
        self.assertEqual([(row["marker_start"], row["marker_end_exclusive"])
                          for row in plan], [(0, 1), (1, 2)])
        self.assertEqual([row["optimizer_step_after"] for row in plan], [False, True])
        with self.assertRaisesRegex(ValueError, "provenance binding"):
            MODULE.build_lazy_packed_shard(
                selected, target, reference, f0, marker_cm, intervals, normalization,
                0.05, 0, 2, 0, 2, prepared_channels=prepared,
                expected_root_seed=20260818, expected_rotation_id="TECHNICAL_KAT",
                expected_fit_normalization_manifest_sha256=manifest_sha,
            )

    def test_split_logical_block_preserves_weighted_loss_and_input_gradient(self):
        rng = np.random.default_rng(17)
        tokens = rng.normal(size=(12, 13)).astype("<f8")
        weights = rng.normal(size=(13, 3)).astype("<f8")
        row_weight = np.linspace(1.0, 2.0, 12, dtype="<f8")
        logits = tokens @ weights
        denominator = float(row_weight.sum() * logits.shape[1])
        full_loss = float(np.sum(row_weight[:, None] * logits * logits) / denominator)
        full_gradient = 2.0 * row_weight[:, None] * (logits @ weights.T) / denominator

        slices = (slice(0, 5), slice(5, 9), slice(9, 12))
        numerator = sum(float(np.sum(row_weight[part, None] *
                                     (tokens[part] @ weights) ** 2)) for part in slices)
        split_denominator = sum(float(row_weight[part].sum() * logits.shape[1])
                                for part in slices)
        split_loss = numerator / split_denominator
        split_gradient = np.concatenate([
            2.0 * row_weight[part, None] * ((tokens[part] @ weights) @ weights.T) /
            split_denominator for part in slices
        ], axis=0)
        self.assertAlmostEqual(full_loss, split_loss, places=14)
        np.testing.assert_allclose(full_gradient, split_gradient, rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
