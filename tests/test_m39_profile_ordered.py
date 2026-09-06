"""Ordered-profile safety checks using artificial inputs, never model fitting."""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import m39_profile_ordered as P
from m33_safe_bridge_core import sample_key
from test_m39_carrier_context import fixture as bridge_fixture


def synthetic_keys(count: int = 96) -> np.ndarray:
    return np.asarray([hashlib.sha256(f"ordered-profile-fixture-{i}".encode()).hexdigest()
                       for i in range(count)], dtype="S64")


def fold_payload(keys: np.ndarray) -> dict[str, np.ndarray]:
    """Three deterministic rotations, with no genotype or outcome input."""
    n = len(keys)
    if n != 96:
        raise ValueError("fixture requires the 96-person library")
    roles = np.full((3, n), "TRAIN", dtype="U6")
    for fold in range(3):
        roles[fold, fold * 32:(fold + 1) * 32] = "SCORE"
        remaining = np.flatnonzero(roles[fold] != "SCORE")
        roles[fold, remaining[:16]] = "SELECT"
    return {"sample_key_sha256": keys, "roles": roles,
            "outer_fold": np.arange(3, dtype=np.uint8),
            "inner_split_seed": np.asarray([11, 12, 13], dtype=np.int64),
            "outer_seed": np.asarray([7], dtype=np.int64)}


class OrderedProfileContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.keys = synthetic_keys()
        self.payload = fold_payload(self.keys)
        self.folds = self.root / "folds.npz"
        self.counts = {"TRAIN": 48, "SELECT": 16, "SCORE": 32}
        self.write_folds()
        self.config = json.loads((ROOT / "conf/m39_ordered_profile.json").read_text())
        self.config["folds"] = {"path": self.folds.name, "uri": "local:synthetic-folds",
                                "sha256": self.digest}
        self.config_path = self.root / "profile.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write_folds(self, payload=None):
        np.savez(self.folds, **(self.payload if payload is None else payload))
        self.digest = P.sha256_file(self.folds)

    def indices(self, *, keys=None, fold=0, counts=None, digest=None):
        return P.training_indices(self.folds, self.keys if keys is None else keys,
                                  self.digest if digest is None else digest, fold,
                                  self.counts if counts is None else counts)

    def load_config(self, config=None):
        self.config_path.write_text(json.dumps(self.config if config is None else config))
        return P.load_profile(self.config_path)

    def test_keyed_join_selects_nested_train_prefixes_independent_of_row_order(self):
        fit = self.indices()
        expected = sorted(self.keys[self.payload["roles"][0] == "TRAIN"].tolist())
        self.assertEqual(self.keys[fit].tolist(), expected)
        prefixes = [set(self.keys[fit[:n]].tolist()) for n in (16, 32, 48)]
        self.assertTrue(prefixes[0] < prefixes[1] < prefixes[2])
        self.assertTrue(np.all(self.payload["roles"][0, fit] == "TRAIN"))
        target_permutation = np.random.default_rng(919).permutation(96)
        reordered_keys = self.keys[target_permutation]
        reordered_fit = self.indices(keys=reordered_keys)
        self.assertEqual(reordered_keys[reordered_fit].tolist(), expected)
        source_permutation = np.random.default_rng(920).permutation(96)
        reordered = copy.deepcopy(self.payload)
        reordered["sample_key_sha256"] = reordered["sample_key_sha256"][source_permutation]
        reordered["roles"] = reordered["roles"][:, source_permutation]
        self.write_folds(reordered)
        self.assertEqual(reordered_keys[self.indices(keys=reordered_keys)].tolist(), expected)

    def test_fold_hash_is_verified_before_npz_open(self):
        with patch.object(P.np, "load", side_effect=AssertionError("archive opened")):
            with self.assertRaisesRegex(ValueError, "hash"):
                self.indices(digest="0" * 64)

    def test_same_size_wrong_or_duplicate_key_universes_are_rejected(self):
        wrong = self.keys.copy()
        wrong[0] = hashlib.sha256(b"unrelated artificial person").hexdigest().encode()
        duplicate = self.keys.copy()
        duplicate[0] = duplicate[1]
        for keys in (wrong, duplicate):
            with self.subTest(kind="target-universe"), self.assertRaises(ValueError):
                self.indices(keys=keys)
        malformed = copy.deepcopy(self.payload)
        malformed["sample_key_sha256"][0] = malformed["sample_key_sha256"][1]
        self.write_folds(malformed)
        with self.assertRaises(ValueError):
            self.indices()

    def test_partial_intersection_and_wrong_sample_schema_are_rejected(self):
        for keys in (self.keys[:-1], self.keys.astype("U64"), self.keys.reshape(8, 12)):
            with self.subTest(shape=keys.shape), self.assertRaises(ValueError):
                self.indices(keys=keys)

    def test_invalid_hash_key_encoding_is_rejected_even_for_matching_universes(self):
        keys = self.keys.copy()
        keys[0] = b"z" * 64
        self.payload["sample_key_sha256"] = keys
        self.write_folds()
        with self.assertRaisesRegex(ValueError, "encoding"):
            self.indices(keys=keys)

    def test_role_counts_and_types_are_not_coerced(self):
        cases = ({"TRAIN": 47, "SELECT": 17, "SCORE": 32},
                 {"TRAIN": 48.0, "SELECT": 16, "SCORE": 32},
                 {"TRAIN": True, "SELECT": 16, "SCORE": 32},
                 {"TRAIN": 48, "SELECT": 16, "SCORE": 0},
                 {"TRAIN": 48, "SELECT": 16, "SCORE": 32, "OTHER": 1})
        for counts in cases:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.indices(counts=counts)
        for fold in (True, 0.0, -1, 3):
            with self.subTest(fold=fold), self.assertRaises(ValueError):
                self.indices(fold=fold)

    def test_role_inventory_dimensions_and_score_rotation_fail_closed(self):
        cases = []
        unknown = copy.deepcopy(self.payload)
        unknown["roles"][0, 0] = "OTHER"
        cases.append(unknown)
        dimensions = copy.deepcopy(self.payload)
        dimensions["roles"] = dimensions["roles"][:2]
        cases.append(dimensions)
        repeated_score = copy.deepcopy(self.payload)
        repeated_score["roles"][1] = repeated_score["roles"][0]
        cases.append(repeated_score)
        extra = copy.deepcopy(self.payload)
        extra["undeclared"] = np.asarray([1])
        cases.append(extra)
        for payload in cases:
            self.write_folds(payload)
            with self.subTest(inventory=sorted(payload)), self.assertRaises(ValueError):
                self.indices()

    def test_query_subset_keeps_all_references_and_does_not_mutate_input(self):
        data = {"sample_key_sha256": self.keys[:6],
                "query_dosage": np.arange(18, dtype=np.int8).reshape(6, 3),
                "query_observed": np.ones((6, 3), dtype=bool),
                "common_target": np.arange(48, dtype=np.int8).reshape(4, 6, 2),
                "common_ref": np.arange(56, dtype=np.int8).reshape(4, 7, 2),
                "reference_sample_key_sha256": self.keys[6:13],
                "ref_dosage": np.arange(21, dtype=np.int8).reshape(3, 7),
                "ref_observed": np.ones((3, 7), dtype=bool),
                "reference_ancestry": np.asarray([0, 0, 1, 1, 2, 2, 2]),
                "common_cm": np.arange(4, dtype=np.float64)}
        before = {key: value.copy() for key, value in data.items()}
        indices = np.asarray([4, 1], dtype=np.int64)
        out = P.subset_queries(data, indices)
        for key in ("sample_key_sha256", "query_dosage", "query_observed"):
            np.testing.assert_array_equal(out[key], before[key][indices])
        np.testing.assert_array_equal(out["common_target"], before["common_target"][:, indices])
        for key in ("common_ref", "ref_dosage", "ref_observed", "reference_ancestry",
                    "reference_sample_key_sha256", "common_cm"):
            np.testing.assert_array_equal(out[key], before[key])
        out["common_target"].fill(0)
        out["query_dosage"].fill(0)
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])

    def test_query_subset_rejects_duplicate_out_of_bounds_and_noninteger_indices(self):
        data = {"sample_key_sha256": self.keys[:6]}
        cases = (np.asarray([1, 1]), np.asarray([-1, 2]), np.asarray([0, 6]),
                 np.asarray([0.0, 1.0]), np.asarray([True, False]), np.asarray([[0, 1]]))
        for indices in cases:
            with self.subTest(dtype=str(indices.dtype)), self.assertRaises(ValueError):
                P.subset_queries(data, indices)

    def test_profile_accepts_checked_repository_contract(self):
        self.assertEqual(self.load_config()["people"], [16, 32, 48])

    def test_profile_rejects_extra_fields_and_invalid_role_contract_before_data(self):
        cases = []
        extra = copy.deepcopy(self.config)
        extra["truth_path"] = "forbidden.npz"
        cases.append(extra)
        for counts in ({"TRAIN": 48, "SELECT": 16, "SCORE": 32, "OTHER": 1},
                       {"TRAIN": 48, "SELECT": 16},
                       {"TRAIN": 48.0, "SELECT": 16, "SCORE": 32},
                       {"TRAIN": 48, "SELECT": True, "SCORE": 32},
                       {"TRAIN": 48, "SELECT": 16, "SCORE": 0}):
            invalid = copy.deepcopy(self.config)
            invalid["role_counts"] = counts
            cases.append(invalid)
        for config in cases:
            with self.subTest(keys=sorted(config)), self.assertRaises(ValueError):
                self.load_config(config)

    def test_profile_rejects_non_nested_sizes_boolean_knobs_and_oversized_streams(self):
        cases = {"people": ([32, 16], [16, 16], [49], [True], []),
                 "fold": (True, 1, 0.0), "K": (True, 0, 9, 1.5),
                 "radii_cm": ([True], [0.2, 0.2], [0], [1.1], [float("nan")]),
                 "site_chunk_size": (True, 0, 4097),
                 "max_window_bytes": (True, 0, 64 * 1024**2 + 1)}
        for key, values in cases.items():
            for value in values:
                invalid = copy.deepcopy(self.config)
                invalid[key] = value
                with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                    self.load_config(invalid)

    def test_profile_rejects_unstaged_paths_bad_hashes_and_descriptor_extras(self):
        for field, value in (("path", "../folds.npz"), ("path", "/tmp/folds.npz"),
                             ("path", ""), ("path", "."), ("path", ".."), ("path", True),
                             ("sha256", "0" * 63), ("sha256", "q" * 64), ("sha256", 1),
                             ("uri", ""), ("uri", False)):
            invalid = copy.deepcopy(self.config)
            invalid["folds"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load_config(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["folds"]["truth"] = "not-an-input"
        with self.assertRaises(ValueError):
            self.load_config(invalid)

    def test_runner_has_no_model_or_scoring_import(self):
        tree = ast.parse((ROOT / "bin/m39_profile_ordered.py").read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        forbidden = {"torch", "m39_carrier_models", "m39_anchor_score", "m39_anchor_screen",
                     "m39_bind_anchor_truth", "m34_parse_flare_truth"}
        self.assertFalse(forbidden.intersection(imports))


class OrderedProfileSyntheticFlowTests(unittest.TestCase):
    def test_real_runner_and_store_path_with_artificial_library_no_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge_path, bridge = bridge_fixture(root)
            target_path = root / bridge["inputs"]["target_vcf"]["path"]
            names = [f"M34_R0_FIT_{i:04d}" for i in range(96)]
            with gzip.open(target_path, "rt") as stream:
                lines = stream.read().splitlines()
            expanded = []
            for line in lines:
                fields = line.split("\t")
                if line.startswith("#CHROM"):
                    fields = fields[:9] + names
                elif not line.startswith("#"):
                    fields = fields[:9] + fields[9:] * 48
                expanded.append("\t".join(fields))
            with gzip.open(target_path, "wt") as stream:
                stream.write("\n".join(expanded) + "\n")
            rare_path = root / bridge["inputs"]["target_rare"]["path"]
            with np.load(rare_path, allow_pickle=False) as archive:
                rare = {name: archive[name].copy() for name in archive.files}
            rare["sample_key_sha256"] = np.asarray(
                [sample_key(name) for name in names], dtype="S64")
            for name in ("minor_dosage", "observed_mask"):
                rare[name] = np.tile(rare[name], (48, 1))
            rare_path = root / "expanded_target_rare.npz"
            np.savez(rare_path, **rare)
            bridge["inputs"]["target_rare"]["path"] = rare_path.name
            for name in ("target_vcf", "target_rare"):
                bridge["inputs"][name]["sha256"] = P.sha256_file(root / bridge["inputs"][name]["path"])
            bridge["parameters"]["expected_target_people"] = 96
            bridge_path.write_text(json.dumps(bridge))
            folds = root / "profile-folds.npz"
            np.savez(folds, **fold_payload(rare["sample_key_sha256"]))
            config = json.loads((ROOT / "conf/m39_ordered_profile.json").read_text())
            config["folds"] = {"path": folds.name, "uri": "local:synthetic-folds",
                               "sha256": P.sha256_file(folds)}
            config.update(people=[16, 32, 48], radii_cm=[0.5], K=2, site_chunk_size=2)
            profile_path = root / "profile.json"
            profile_path.write_text(json.dumps(config))
            outdir = root / "ordered-output"
            receipt = P.profile(bridge_path, profile_path, root, outdir)
            self.assertEqual(receipt["decision"], "PASS_ORDERED_INPUT_TECHNICAL_ONLY")
            self.assertTrue(receipt["input_dir_files_unchanged"])
            self.assertEqual(receipt["boundaries"]["feature_computation_roles"], ["TRAIN"])
            self.assertEqual(receipt["boundaries"]["source_shared_target_genotypes_parsed"], 96)
            for key in ("truth_opened", "predictions_opened", "model_training",
                        "training_memory_profiled", "rare_phase_assigned"):
                self.assertIs(receipt["boundaries"][key], False)
            self.assertEqual(len(receipt["profiles"]), 3)
            roles = fold_payload(rare["sample_key_sha256"])["roles"]
            expected = sorted(rare["sample_key_sha256"][roles[0] == "TRAIN"].tolist())
            for row, count in zip(receipt["profiles"], (16, 32, 48)):
                self.assertEqual((row["people"], row["anchors"]), (count, 2))
                self.assertGreater(row["stream_chunks"], 0)
                self.assertGreater(row["expected_query_anchor_sites"], 0)
                self.assertEqual(row["streamed_query_anchor_sites"], row["expected_query_anchor_sites"])
                self.assertLessEqual(row["largest_channel_chunk_bytes"], config["max_window_bytes"])
                self.assertTrue((outdir / row["case"] / "manifest.json").is_file())
                opened = P.OrderedContextStore.open(outdir / row["case"],
                    expected_manifest_sha256=row["store_manifest_sha256"])
                self.assertEqual(opened.arrays["sample_key_sha256"].tolist(), expected[:count])
                self.assertEqual(len(opened.arrays["reference_sample_key_sha256"]), 6)
            public_receipt = (outdir / "receipt.json").read_text()
            self.assertNotIn(names[0], public_receipt)
            self.assertNotIn(rare["sample_key_sha256"][0].decode(), public_receipt)
            with self.assertRaisesRegex(ValueError, "already exists"):
                P.profile(bridge_path, profile_path, root, outdir)


if __name__ == "__main__":
    unittest.main()
