#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M34 = load_module("m34_materialize", ROOT / "bin" / "m34_materialize.py")
M33 = load_module("m33_materialize_compatibility", ROOT / "bin" / "m33_materialize.py")


def fixture(sample_count: int = 30, ancestries=("AFR", "EUR", "NAM")):
    loci = np.asarray([10, 11, 12], dtype="<u8")
    samples = np.asarray([f"{index:064x}".encode() for index in range(sample_count)], dtype="|S64")
    selected = {
        "locus_id": loci, "chrom": np.full(3, 22, dtype="|u1"),
        "pos": np.asarray([100, 200, 300], dtype="<i8"),
        "ref": np.asarray([b"A", b"C", b"G"], dtype="|S1"),
        "alt": np.asarray([b"T", b"G", b"A"], dtype="|S1"),
        "cM": np.asarray([0.05, 0.10, 0.20], dtype="<f8"),
    }
    dosage = np.tile(np.asarray([[2, 1, 0]], dtype="|i1"), (sample_count, 1))
    observed = np.ones((sample_count, 3), dtype="|u1")
    observed[-1, 1] = 0
    dosage[-1, 1] = 0
    target = {"sample_key_sha256": samples, "locus_id": loci,
              "minor_dosage": dosage, "observed_mask": observed}
    ancestry_count = len(ancestries)
    ac = np.arange(ancestry_count * 3, dtype="<u2").reshape(ancestry_count, 3) % 4
    an = np.full((ancestry_count, 3), 8, dtype="<u2")
    reference = {
        "ancestry": np.asarray([value.encode() for value in ancestries], dtype="|S4"),
        "locus_id": loci, "minor_ac": ac, "callable_an": an,
        "minor_af": ac.astype("<f8") / an,
        "observed_mask": np.ones((ancestry_count, 3), dtype="|u1"),
        "no_support": (ac == 0).astype("|u1"),
    }
    probabilities = np.full((sample_count, 2, 2, ancestry_count),
                            np.float32(1 / ancestry_count), dtype="<f4")
    f0 = {
        "sample_key_sha256": samples, "marker_chrom": np.full(2, 22, dtype="|u1"),
        "marker_pos": np.asarray([150, 250], dtype="<i8"),
        "marker_ref": np.asarray([b"A", b"C"], dtype="|S1"),
        "marker_alt": np.asarray([b"G", b"T"], dtype="|S1"), "F0": probabilities,
    }
    return selected, target, reference, f0, np.asarray([0.10, 0.15], dtype="<f8")


class M34DataLayerTests(unittest.TestCase):
    def test_thirty_sample_compatibility_with_m33(self):
        factors = fixture(30, ("AFR", "EUR", "ASIA"))
        selected, target, reference, f0, marker_cm = factors
        maxima = {"AFR": 8, "EUR": 8, "ASIA": 8}
        rd, re = M34.build_paired_shard(
            *factors, ("AFR", "EUR", "ASIA"), maxima, 0.05, 0, 8, 0, 2,
        )
        oracle = M33.build_packed_shard(
            selected, target, reference, f0, marker_cm, maxima, 0.05, 0, 8, 0, 2,
        )
        self.assertEqual(re["rare_tokens"].shape[1], 13)
        for name in oracle:
            np.testing.assert_array_equal(re[name], oracle[name], err_msg=name)
        M34.validate_control_pair(rd, re, ("AFR", "EUR", "ASIA"))

    def test_256_samples_are_sharded_without_hardcoding(self):
        selected, target, reference, f0, marker_cm = fixture(256)
        dimensions = M34.validate_inputs(
            selected, target, reference, f0, marker_cm, ("AFR", "EUR", "NAM")
        )
        self.assertEqual(dimensions["sample_count"], 256)
        shards = M34.plan_sample_shards(dimensions["sample_count"], 8)
        self.assertEqual(len(shards), 32)
        self.assertEqual(shards[0], {"sample_start": 0, "sample_end_exclusive": 8})
        self.assertEqual(shards[-1], {"sample_start": 248, "sample_end_exclusive": 256})
        covered = [index for shard in shards
                   for index in range(shard["sample_start"], shard["sample_end_exclusive"])]
        self.assertEqual(covered, list(range(256)))

    def test_ancestry_order_is_contractual(self):
        factors = list(fixture())
        reference = {name: value.copy() for name, value in factors[2].items()}
        reference["ancestry"] = reference["ancestry"][[1, 0, 2]]
        for name in ("minor_ac", "callable_an", "minor_af", "observed_mask", "no_support"):
            reference[name] = reference[name][[1, 0, 2]]
        factors[2] = reference
        with self.assertRaisesRegex(ValueError, "ancestry order"):
            M34.validate_inputs(*factors, ("AFR", "EUR", "NAM"))

    def test_rd_re_keep_identical_axes_and_masks(self):
        factors = fixture()
        rd, re = M34.build_paired_shard(
            *factors, ("AFR", "EUR", "NAM"), {"AFR": 8, "EUR": 8, "NAM": 8},
            0.1, 0, 8, 0, 2,
        )
        M34.validate_control_pair(rd, re, ("AFR", "EUR", "NAM"))
        for name in set(re) - {"rare_tokens"}:
            np.testing.assert_array_equal(rd[name], re[name], err_msg=name)
        corrupted = {name: value.copy() for name, value in rd.items()}
        corrupted["rare_mask"][0] = 0
        with self.assertRaisesRegex(ValueError, "axis or mask"):
            M34.validate_control_pair(corrupted, re, ("AFR", "EUR", "NAM"))

    def test_role_overlap_and_dimension_failures(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            M34.validate_role_disjointness({"REF": ["a", "b"], "SCORE": ["b", "c"]})
        factors = list(fixture())
        broken_target = {name: value.copy() for name, value in factors[1].items()}
        broken_target["minor_dosage"] = broken_target["minor_dosage"][:-1]
        factors[1] = broken_target
        with self.assertRaisesRegex(ValueError, "TARGET dimensions"):
            M34.validate_inputs(*factors, ("AFR", "EUR", "NAM"))
        with self.assertRaisesRegex(ValueError, "overlap, contain gaps"):
            M34.validate_sample_shards([
                {"sample_start": 0, "sample_end_exclusive": 8},
                {"sample_start": 7, "sample_end_exclusive": 16},
            ], 16, 8)

    def test_contract_forbids_configured_sample_count(self):
        contract_path = ROOT / "conf" / "m34_nam_data_contract.json"
        observed = M34.load_contract(contract_path)
        self.assertEqual(observed["ancestry_names"], ["AFR", "EUR", "NAM"])
        self.assertNotIn("sample_count", observed)
        with tempfile.TemporaryDirectory() as raw:
            modified = json.loads(contract_path.read_text(encoding="utf-8"))
            modified["sample_count"] = 30
            path = Path(raw) / "invalid.json"
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract members|sample_count"):
                M34.load_contract(path)


if __name__ == "__main__":
    unittest.main()
