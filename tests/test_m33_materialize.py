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


if __name__ == "__main__":
    unittest.main()
