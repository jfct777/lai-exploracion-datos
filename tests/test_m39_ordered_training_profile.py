"""Technical profile contract/selection checks without optimization or data reads."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m39_profile_ordered_training as P


class FakeArchive(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class OrderedTrainingProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile_path = ROOT / "conf" / "m39_ordered_training_profile.json"
        self.cfg = json.loads(self.profile_path.read_text())

    def load_changed(self, cfg):
        with patch.object(Path, "read_text", return_value=json.dumps(cfg)):
            return P.load_profile(Path("artificial-profile.json"))

    def test_frozen_profile_has_all_sixteen_unique_technical_cases(self):
        cfg = P.load_profile(self.profile_path)
        self.assertEqual(cfg["scope"], "technical_optimization_synthetic_labels_inner_TRAIN_only")
        self.assertEqual(cfg["weight_decay"], 0)
        self.assertEqual(len({case["id"] for case in cfg["cases"]}), 16)
        observed = {(case["family"], case["size"], case["window"], case["arm"])
                    for case in cfg["cases"]}
        expected = {(family, size, window, arm)
                    for family in ("cnn", "attention") for size in ("small", "medium")
                    for window in ("median", "maximum") for arm in ("common", "real")}
        self.assertEqual(observed, expected)

    def test_inventory_scope_hash_and_data_regime_drift_are_rejected(self):
        changes = ({"scope": "ancestry_training"}, {"schema_version": "old"},
                   {"parent_receipt_sha256": "A" * 64}, {"folds_sha256": "0" * 63},
                   {"store_manifest_sha256": "../manifest"}, {"fold": 1},
                   {"fold": False}, {"store_case": "people_16/radius_1cm"},
                   {"new_truth_file": "forbidden"})
        for change in changes:
            cfg = copy.deepcopy(self.cfg)
            cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load_changed(cfg)
        cfg = copy.deepcopy(self.cfg)
        del cfg["scope"]
        with self.assertRaises(ValueError):
            self.load_changed(cfg)

    def test_resource_and_optimizer_envelope_is_closed(self):
        changes = ({"steps": 1}, {"steps": 6}, {"steps": True}, {"torch_threads": 3},
                   {"max_seconds": 901}, {"max_rss_kib": 6710887},
                   {"max_input_bytes": 64 * 1024 ** 2 + 1}, {"seed": -1},
                   {"learning_rate": 0.01}, {"weight_decay": 0.01},
                   {"equivalence_atol": 0.01}, {"equivalence_rtol": 0.01})
        for change in changes:
            cfg = copy.deepcopy(self.cfg)
            cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load_changed(cfg)

    def test_case_identity_duplicates_and_unsupported_arms_fail(self):
        changed_cases = ({"arm": "pooled"}, {"window": "cropped"}, {"family": "global"},
                         {"batch_size": True}, {"batch_size": 3}, {"core_sites": 128},
                         {"id": "unbound-name"}, {"hidden_labels": "unused"})
        for change in changed_cases:
            cfg = copy.deepcopy(self.cfg)
            cfg["cases"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load_changed(cfg)
        for mode in ("duplicate", "missing"):
            cfg = copy.deepcopy(self.cfg)
            if mode == "duplicate":
                cfg["cases"][1] = copy.deepcopy(cfg["cases"][0])
            else:
                cfg["cases"].pop()
            with self.assertRaises(ValueError):
                self.load_changed(cfg)

    def test_capacity_and_attention_recipe_drift_fail(self):
        for size, change in (("small", {"width": 64}), ("medium", {"depth": 2}),
                             ("medium", {"attention_radius_tokens": 16}),
                             ("small", {"heads": 2}), ("small", {"kernels": [3]}),
                             ("medium", {"dilations": [1, 1, 1, 1]}),
                             ("small", {"dropout": 0.1})):
            cfg = copy.deepcopy(self.cfg)
            cfg["recipes"][size].update(change)
            with self.subTest(size=size, change=change), self.assertRaises(ValueError):
                self.load_changed(cfg)

    def test_choose_pairs_uses_complete_window_lengths_and_first_tie(self):
        lengths = [4, 10, 6, 6, 10]
        store = SimpleNamespace(shape=(48, len(lengths), 8),
                                window_bounds=lambda j: (100 * j, 100 * j + lengths[j]))
        pairs, returned = P.choose_pairs(store, {"window": "median", "batch_size": 1})
        self.assertEqual(pairs, [(0, 2)])
        np.testing.assert_array_equal(returned, lengths)
        pairs, _ = P.choose_pairs(store, {"window": "maximum", "batch_size": 2})
        self.assertEqual(pairs, [(0, 1), (1, 1)])
        store.window_bounds = lambda j: (0, 0 if j == 2 else 5)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            P.choose_pairs(store, {"window": "median", "batch_size": 1})

    def authentication_fixture(self):
        keys = np.asarray([f"{index:064x}".encode() for index in range(96)], dtype="S64")
        roles = np.full((3, 96), "SELECT", dtype="U8")
        roles[:, :48] = "TRAIN"
        archive = FakeArchive(sample_key_sha256=keys, roles=roles, outer_fold=np.arange(3),
                              inner_split_seed=np.array([1]), outer_seed=np.array([2]))
        store = SimpleNamespace(shape=(48, 660, 8), arrays={
            "sample_key_sha256": keys[:48].copy(), "reference_ancestry": np.zeros(753, dtype=np.int8)})
        parent = {"decision": "PASS_ORDERED_INPUT_TECHNICAL_ONLY", "boundaries": {"truth_opened": False},
                  "profiles": [{"case": self.cfg["store_case"],
                                "store_manifest_sha256": self.cfg["store_manifest_sha256"]}],
                  "input_hashes": {"folds": self.cfg["folds_sha256"], "common_source": "c" * 64}}
        manifest = {"source_sha256": dict(parent["input_hashes"])}
        return store, archive, parent, manifest

    def authenticate_fake(self, store, archive, parent, manifest, *, parent_hash=None, fold_hash=None):
        store_path, parent_path, folds_path = Path("store"), Path("parent.json"), Path("folds.npz")
        hashes = {parent_path: self.cfg["parent_receipt_sha256"] if parent_hash is None else parent_hash,
                  folds_path: self.cfg["folds_sha256"] if fold_hash is None else fold_hash}
        with patch.object(P, "sha256_file", side_effect=lambda path: hashes[path]), \
                patch.object(Path, "read_text", side_effect=[json.dumps(parent), json.dumps(manifest)]), \
                patch.object(P.OrderedContextStore, "open", return_value=store) as opened, \
                patch.object(P.np, "load", return_value=archive) as loaded:
            result, returned_manifest = P.authenticate(store_path, parent_path, folds_path, self.cfg)
            opened.assert_called_once_with(store_path, expected_manifest_sha256=self.cfg["store_manifest_sha256"])
            loaded.assert_called_once_with(folds_path, allow_pickle=False)
            self.assertIs(result, store)
            self.assertEqual(returned_manifest, manifest)

    def test_authentication_binds_parent_sources_manifest_and_exact_train_roster(self):
        self.authenticate_fake(*self.authentication_fixture())

    def test_authentication_rejects_parent_truth_and_parent_or_fold_hash_drift(self):
        for what in ("parent_hash", "fold_hash", "truth", "decision", "duplicated_store", "store_hash"):
            store, archive, parent, manifest = self.authentication_fixture()
            kwargs = {}
            if what in ("parent_hash", "fold_hash"):
                kwargs[what] = "f" * 64
            elif what == "truth":
                parent["boundaries"]["truth_opened"] = True
            elif what == "decision":
                parent["decision"] = "PASS"
            elif what == "duplicated_store":
                parent["profiles"] *= 2
            else:
                parent["profiles"][0]["store_manifest_sha256"] = "f" * 64
            with self.subTest(what=what), self.assertRaises(ValueError):
                self.authenticate_fake(store, archive, parent, manifest, **kwargs)

    def test_authentication_rejects_source_roster_axis_or_reference_regime_drift(self):
        for what in ("sources", "query", "roles", "duplicate_key", "fold_axis", "ref_count", "k", "archive_fields"):
            store, archive, parent, manifest = self.authentication_fixture()
            if what == "sources":
                manifest["source_sha256"]["common_source"] = "f" * 64
            elif what == "query":
                store.arrays["sample_key_sha256"][0] = archive["sample_key_sha256"][48]
            elif what == "roles":
                archive["roles"][0, 0] = "SELECT"
            elif what == "duplicate_key":
                archive["sample_key_sha256"][1] = archive["sample_key_sha256"][0]
            elif what == "fold_axis":
                archive["outer_fold"] = np.array([1, 0, 2])
            elif what == "ref_count":
                store.arrays["reference_ancestry"] = np.zeros(752, dtype=np.int8)
            elif what == "k":
                store.shape = (48, 660, 4)
            else:
                archive["extra"] = np.array([0])
            with self.subTest(what=what), self.assertRaises(ValueError):
                self.authenticate_fake(store, archive, parent, manifest)


if __name__ == "__main__":
    unittest.main()
