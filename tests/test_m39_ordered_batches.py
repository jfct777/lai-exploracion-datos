"""Artificial-input batch adapter checks; no ancestry labels or model fitting."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import m39_carrier_context as B
import m39_ordered_context as O
import m39_ordered_batches as A
from test_m39_carrier_context import fixture

FIELDS = {"channels", "delta_cm", "site_mask", "candidate_mask", "reference_dosage",
          "reference_observed", "query_dosage", "query_observed", "radius_cm", "pooled_af", "pooled_observed"}
MASKS = {"site_mask", "candidate_mask", "reference_observed", "query_observed", "pooled_observed"}


class OrderedBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path, _ = fixture(self.root)
        config, paths = B.load_config(path, self.root)
        self.data = B.extract_inputs(paths, config)

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, data=None, *, radius=0.5, k=3):
        data = O.canonicalize_reference_homologs(self.data if data is None else data)
        candidates, _ = B.materialize_radius(data, np.arange(2, dtype=np.int64), k, radius)
        return O.build_factorized_store(data, candidates)

    def pack(self, store=None, pairs=None, **kwargs):
        return A.pack_batch(self.store() if store is None else store,
                            [(0, 0), (1, 1)] if pairs is None else pairs,
                            kwargs.pop("max_input_bytes", 1_000_000), **kwargs)

    def test_exact_fields_shapes_types_and_right_padding(self):
        store = self.store()
        pairs = [(0, 0), (1, 1)]
        batch = self.pack(store, pairs)
        windows = [store.read_window(q, j, max_window_bytes=1_000_000) for q, j in pairs]
        maximum = max(w.window_site_count for w in windows)
        self.assertEqual(set(batch), FIELDS)
        self.assertEqual(tuple(batch["channels"].shape), (2, 2, 3, 3, maximum, 5))
        for name, tensor in batch.items():
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, torch.bool if name in MASKS else torch.float32)
            self.assertTrue(bool(torch.isfinite(tensor).all()))
        for row, window in enumerate(windows):
            length = window.window_site_count
            np.testing.assert_array_equal(batch["channels"][row, ..., :length, :].numpy(), window.channels)
            np.testing.assert_array_equal(batch["delta_cm"][row, :length].numpy(), window.delta_cm.astype(np.float32))
            self.assertTrue(bool(batch["site_mask"][row, :length].all()))
            self.assertFalse(bool(batch["site_mask"][row, length:].any()))
            self.assertFalse(bool(batch["channels"][row, ..., length:, :].any()))
            self.assertFalse(bool(batch["delta_cm"][row, length:].any()))
            np.testing.assert_array_equal(batch["reference_dosage"][row].numpy(), window.ref_dosage)
            np.testing.assert_array_equal(batch["reference_observed"][row].numpy(), window.ref_observed)

    def test_memory_guard_precedes_window_or_padded_allocations(self):
        store = self.store()
        pairs = [(0, 0), (1, 1)]
        plan = A.estimate_batch_bytes(store, pairs)
        self.assertEqual(plan["required_bytes"], sum(plan[name] for name in (
            "numpy_staging_bytes", "tensor_bytes", "reader_workspace_bytes",
            "validation_workspace_bytes", "pooled_workspace_bytes")))
        with patch.object(store, "read_window", side_effect=AssertionError("window read")), \
                patch.object(A.np, "zeros", side_effect=AssertionError("padded array allocated")), \
                patch.object(A.torch, "tensor", side_effect=AssertionError("tensor allocated")):
            with self.assertRaisesRegex(ValueError, "byte ceiling"):
                self.pack(store, pairs, max_input_bytes=plan["required_bytes"] - 1)
        batch = self.pack(store, pairs, max_input_bytes=plan["required_bytes"])
        self.assertEqual(sum(x.numel() * x.element_size() for x in batch.values()), plan["tensor_bytes"])

    def test_guard_counts_padding_and_all_rows_not_only_largest_window(self):
        store = self.store()
        one = A.estimate_batch_bytes(store, [(0, 1)])
        pairs = [(0, 0), (1, 1)] * 8
        many = A.estimate_batch_bytes(store, pairs)
        self.assertGreater(many["numpy_staging_bytes"], one["numpy_staging_bytes"])
        with self.assertRaisesRegex(ValueError, "byte ceiling"):
            self.pack(store, pairs, max_input_bytes=one["required_bytes"])

    def test_complete_windows_are_not_truncated_to_an_input_limit(self):
        store = self.store(radius=1.0)
        batch = self.pack(store)
        self.assertTrue(bool(batch["site_mask"].all()))
        self.assertEqual(batch["channels"].shape[-2], len(store.arrays["common_cm"]))

    def test_empty_windows_have_one_fully_masked_padding_position(self):
        store = self.store(radius=0.05, k=8)
        batch = self.pack(store)
        self.assertEqual(tuple(batch["channels"].shape), (2, 2, 3, 8, 1, 5))
        for name in ("channels", "delta_cm", "site_mask", "candidate_mask", "reference_observed"):
            self.assertFalse(bool(batch[name].any()))
        self.assertTrue(bool(torch.isfinite(batch["pooled_af"]).all()))

    def test_pooled_af_uses_all_observed_diploid_people_not_top_k(self):
        data = copy.deepcopy(self.data)
        data["ref_observed"][:] = True
        data["ref_dosage"][:] = np.asarray([0, 2, 1, 1, 0, 0])[None, :]
        small = self.pack(self.store(data, radius=1.0, k=1))
        larger = self.pack(self.store(data, radius=1.0, k=3))
        expected = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        torch.testing.assert_close(small["pooled_af"], expected, rtol=0, atol=0)
        torch.testing.assert_close(small["pooled_af"], larger["pooled_af"], rtol=0, atol=0)
        self.assertTrue(bool(small["pooled_observed"].all()))
        # One observed homozygote counts one diploid person: AC2 / AN2 = 1.
        data["ref_observed"][:, 0] = False
        data["ref_dosage"][:, 0] = 0
        batch = self.pack(self.store(data, radius=1.0, k=1))
        torch.testing.assert_close(batch["pooled_af"][:, 0], torch.ones(2), rtol=0, atol=0)

    def test_no_pooled_support_is_masked_not_zero_frequency_evidence(self):
        data = copy.deepcopy(self.data)
        data["ref_observed"][:, data["reference_ancestry"] == 2] = False
        data["ref_dosage"][:, data["reference_ancestry"] == 2] = 0
        batch = self.pack(self.store(data))
        self.assertFalse(bool(batch["pooled_observed"][:, 2].any()))
        self.assertFalse(bool(batch["pooled_af"][:, 2].any()))

    def test_delta_subtraction_precedes_float32_conversion(self):
        data = copy.deepcopy(self.data)
        offset = float(2**24)
        data["common_cm"] = data["common_cm"] + offset
        data["selected"]["cM"] = data["selected"]["cM"] + offset
        store = self.store(data, radius=1.0)
        batch = self.pack(store, [(0, 0)])
        expected = (store.arrays["common_cm"] - float(store.arrays["cM"][0])).astype(np.float32)
        prematurely_cast = store.arrays["common_cm"].astype(np.float32) - np.float32(store.arrays["cM"][0])
        self.assertFalse(np.array_equal(expected, prematurely_cast))
        np.testing.assert_array_equal(batch["delta_cm"][0].numpy(), expected)

    def test_tensor_mutation_cannot_modify_mmap_store(self):
        source = self.store()
        directory = self.root / "saved"
        receipt = source.save(directory, source_hashes={"fixture": "a" * 64})
        store = O.OrderedContextStore.open(directory, expected_manifest_sha256=receipt["manifest_sha256"])
        before = {name: array.copy() for name, array in store.arrays.items()}
        batch = self.pack(store)
        for tensor in batch.values():
            tensor.fill_(0)
        for name, value in store.arrays.items():
            np.testing.assert_array_equal(value, before[name])
            self.assertFalse(value.flags.writeable)

    def test_rare_changes_leave_common_batch_fields_identical(self):
        original = self.pack()
        changed = copy.deepcopy(self.data)
        for dosage, observed in (("query_dosage", "query_observed"), ("ref_dosage", "ref_observed")):
            changed[dosage] = np.where(changed[observed], 2 - changed[dosage], 0).astype(np.int8)
        changed_batch = self.pack(self.store(changed))
        for name in ("channels", "delta_cm", "site_mask", "candidate_mask", "radius_cm"):
            torch.testing.assert_close(original[name], changed_batch[name], rtol=0, atol=0)

    def test_bad_pairs_budgets_and_nonmaterialized_devices_fail(self):
        store = self.store()
        for pairs in ([], [(True, 0)], [(0.0, 0)], [(0,)], [(2, 0)], [(0, 2)], [(-1, 0)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                self.pack(store, pairs)
        for budget in (True, 0, -1, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.pack(store, max_input_bytes=budget)
        with self.assertRaises(ValueError):
            self.pack(store, device="meta")

    def test_nonfinite_and_invalid_masks_from_reader_are_rejected(self):
        store = self.store()
        window = store.read_window(0, 0, max_window_bytes=1_000_000)
        delta = window.delta_cm.copy()
        delta[0] = np.nan
        bad_channels = window.channels.copy()
        bad_channels.flat[0] = 2
        candidates = window.candidate_mask.astype(np.uint8)
        for bad in (replace(window, delta_cm=delta), replace(window, channels=bad_channels),
                    replace(window, candidate_mask=candidates)):
            with patch.object(store, "read_window", return_value=bad):
                with self.assertRaises(ValueError):
                    self.pack(store, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
