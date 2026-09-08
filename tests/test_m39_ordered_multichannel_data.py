"""Synthetic technical contracts, with no efficacy claims or historical writes."""
from __future__ import annotations

import copy
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
import m39_carrier_context as C
import m39_ordered_context as O
import m39_ordered_multichannel_data as M
from m39_ordered_batches import pack_batch
from m39_ordered_multichannel import MultichannelConfig, OrderedMultichannelAdapter
from m39_ordered_models import OrderedLAIModel, OrderedModelConfig
from m39_ordered_training_data import reference_link_permutation
from test_m39_carrier_context import fixture


class MultichannelDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path, _ = fixture(self.root)
        config, paths = C.load_config(path, self.root)
        self.data = C.extract_inputs(paths, config)

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, data=None, *, k=3, radius=.5):
        data = O.canonicalize_reference_homologs(self.data if data is None else data)
        candidates, _ = C.materialize_radius(data, np.arange(2, dtype=np.int64), k, radius)
        return O.build_factorized_store(data, candidates)

    def baseline(self, store):
        return np.full((store.shape[0], store.shape[1], 6), 1 / 6, dtype=np.float32)

    def pack(self, store=None, *, mode="BOTH", pairs=None, summary=None, baseline=None, **kwargs):
        store = self.store() if store is None else store
        summary = M.ReferenceSummary.from_store(store) if summary is None else summary
        return M.pack_multichannel_batch(store, [(0, 0), (1, 1)] if pairs is None else pairs,
                                        self.baseline(store) if baseline is None else baseline, summary,
                                        mode=mode, max_input_bytes=kwargs.pop("max_input_bytes", 1_000_000),
                                        **kwargs)

    def test_exact_model_inventory_shapes_and_types(self):
        expected = {
            "NONE": M.COMMON_FIELDS | {"baseline"},
            "OFF": {"baseline"},
            "SUMMARY": M.COMMON_FIELDS | M.QUERY_FIELDS | M.SUMMARY_FIELDS | {"baseline"},
            "DETAIL": M.COMMON_FIELDS | M.QUERY_FIELDS | M.DETAIL_FIELDS | {"baseline"},
            "BOTH": M.COMMON_FIELDS | M.QUERY_FIELDS | M.SUMMARY_FIELDS | M.DETAIL_FIELDS | {"baseline"},
        }
        for mode, fields in expected.items():
            with self.subTest(mode=mode):
                packed = self.pack(mode=mode)
                self.assertEqual(set(packed), fields)
                self.assertEqual(tuple(packed["baseline"].shape), (2, 6))
                self.assertEqual(packed["baseline"].dtype, torch.float32)
                for name in M.SUMMARY_FIELDS & fields:
                    self.assertEqual(packed[name].dtype, torch.int64)
                self.assertFalse(any("truth" in name or "index" in name or "key" in name for name in packed))
        packed = self.pack()
        self.assertEqual(tuple(packed["ref_dosage_counts"].shape), (2, 3, 3))
        self.assertEqual(tuple(packed["ref_eligible"].shape), (2, 3))
        self.assertEqual(tuple(packed["reference_dosage"].shape), (2, 2, 3, 3))

    def test_global_summary_uses_unique_people_not_retrieved_haplotypes(self):
        data = copy.deepcopy(self.data)
        data["ref_observed"][:] = True
        data["ref_dosage"][:] = [0, 2, 1, 1, 0, 0]
        one, many = self.store(data, k=1, radius=1), self.store(data, k=4, radius=1)
        small, large = self.pack(one), self.pack(many)
        expected = np.asarray([[[1, 0, 1], [0, 2, 0], [2, 0, 0]]] * 2, dtype=np.int64)
        np.testing.assert_array_equal(small["ref_dosage_counts"].numpy(), expected)
        torch.testing.assert_close(small["ref_dosage_counts"], large["ref_dosage_counts"], rtol=0, atol=0)
        np.testing.assert_array_equal(large["ref_eligible"].numpy(), [[2, 2, 2]] * 2)
        self.assertGreater(int(large["candidate_mask"].sum()), int(small["candidate_mask"].sum()))
        # Four candidate haplotypes per ancestry still represent two REF persons.
        self.assertEqual(int(large["reference_observed"][0].sum()), 24)
        self.assertEqual(int(large["ref_dosage_counts"][0].sum()), 6)

    def test_missing_does_not_become_observed_zero_or_leave_eligibility(self):
        data = copy.deepcopy(self.data)
        data["ref_observed"][:] = True
        data["ref_dosage"][:] = 0
        data["ref_observed"][:, 0] = False
        data["ref_observed"][:, 4:] = False
        batch = self.pack(self.store(data))
        np.testing.assert_array_equal(batch["ref_eligible"].numpy(), [[2, 2, 2]] * 2)
        np.testing.assert_array_equal(batch["ref_dosage_counts"][:, :, 0].numpy(), [[1, 2, 0]] * 2)
        self.assertFalse(bool(batch["ref_dosage_counts"][:, :, 1:].any()))
        self.assertFalse(bool(batch["reference_observed"][:, :, 2].any()))

    def test_missing_query_retains_its_mask(self):
        data = copy.deepcopy(self.data)
        data["query_dosage"][:] = 0
        data["query_observed"][:] = False
        batch = self.pack(self.store(data))
        self.assertFalse(bool(batch["query_observed"].any()))
        self.assertFalse(bool(batch["query_dosage"].any()))

    def test_duplicate_reference_people_rejected_by_store_contract(self):
        a = {key: value.copy() for key, value in self.store().arrays.items()}
        a["reference_sample_key_sha256"][1] = a["reference_sample_key_sha256"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            O.OrderedContextStore(a)

    def test_all_common_fields_and_real_detail_match_historical_packer_exactly(self):
        store = self.store()
        old = pack_batch(store, [(0, 0), (1, 1)], 1_000_000)
        for mode in ("NONE", "SUMMARY", "DETAIL", "BOTH"):
            batch = self.pack(store, mode=mode)
            for key in M.COMMON_FIELDS | (M.DETAIL_FIELDS if mode in ("DETAIL", "BOTH") else set()):
                torch.testing.assert_close(batch[key], old[key], rtol=0, atol=0)

    def test_pairs_order_duplicates_and_exact_baseline_zero_are_preserved(self):
        store = self.store()
        baseline = np.eye(6, dtype=np.float32)[np.asarray([[0, 1], [2, 5]])]
        pairs = [(1, 1), (0, 0), (1, 1)]
        batch = self.pack(store, baseline=baseline, pairs=pairs)
        np.testing.assert_array_equal(batch["baseline"].numpy(), baseline[[1, 0, 1], [1, 0, 1]])
        torch.testing.assert_close(batch["reference_dosage"][0], batch["reference_dosage"][2], rtol=0, atol=0)
        self.assertFalse(batch["baseline"].requires_grad)

    def test_sham_preserves_summary_common_masks_and_shared_person_doses(self):
        data = copy.deepcopy(self.data)
        data["ref_observed"][:] = True
        data["ref_dosage"][:] = [0, 2, 0, 2, 0, 2]
        store = self.store(data, k=4, radius=1)
        # Explicit within-ancestry bijection makes every diploid genotype change.
        permutation = np.tile(np.asarray([1, 0, 3, 2, 5, 4]), (2, 1))
        real = self.pack(store)
        sham = self.pack(store, mode="SHAM", sham_permutation=permutation)
        for key in M.COMMON_FIELDS | M.SUMMARY_FIELDS | M.QUERY_FIELDS | {"baseline", "reference_observed"}:
            torch.testing.assert_close(real[key], sham[key], rtol=0, atol=0)
        a = store.arrays
        for row, (query, anchor) in enumerate([(0, 0), (1, 1)]):
            active = a["candidate_mask"][query, anchor].astype(bool)
            people = a["candidate_ref_index"][query, anchor][active]
            expected = a["ref_dosage"][anchor, permutation[anchor, people]]
            np.testing.assert_array_equal(sham["reference_dosage"][row].numpy()[active], expected)
        self.assertTrue(bool((real["reference_dosage"] != sham["reference_dosage"]).any()))

    def test_existing_reference_link_permutation_is_accepted(self):
        store = self.store()
        mapping = reference_link_permutation(store, 17)
        self.pack(store, mode="SHAM", sham_permutation=mapping)

    def test_nonbijective_or_cross_ancestry_or_cross_observation_sham_rejected(self):
        store = self.store()
        identity = np.tile(np.arange(6), (2, 1))
        duplicate = identity.copy()
        duplicate[:, 0] = 1
        cross = identity.copy()
        cross[:, [0, 2]] = cross[:, [2, 0]]
        data = copy.deepcopy(self.data)
        data["ref_observed"][:, 0] = False
        data["ref_dosage"][:, 0] = 0
        data["ref_observed"][:, 1] = True
        missing_store = self.store(data)
        mask_swap = identity.copy()
        mask_swap[:, [0, 1]] = mask_swap[:, [1, 0]]
        for candidate, source, message in ((duplicate, store, "bijection"), (cross, store, "ancestry"),
                                           (mask_swap, missing_store, "strata")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.pack(source, mode="SHAM", sham_permutation=candidate)

    def test_summary_can_share_across_identical_refs_but_not_changed_genotypes_or_axes(self):
        first, second = self.store(), self.store()
        summary = M.ReferenceSummary.from_store(first)
        self.pack(second, summary=summary)
        data = copy.deepcopy(self.data)
        data["ref_dosage"] = np.where(data["ref_observed"], 2 - data["ref_dosage"], 0).astype(np.int8)
        with self.assertRaisesRegex(ValueError, "summary REF/locus"):
            self.pack(self.store(data), summary=summary)
        data = copy.deepcopy(self.data)
        data["minor_is_alt"] = 1 - data["minor_is_alt"]
        with self.assertRaisesRegex(ValueError, "summary REF/locus"):
            self.pack(self.store(data), summary=summary)

    def test_pack_mutation_never_changes_store_baseline_or_cached_counts(self):
        store = self.store()
        summary, baseline = M.ReferenceSummary.from_store(store), self.baseline(store)
        source = {key: value.copy() for key, value in store.arrays.items()}
        counts = summary.ref_dosage_counts.copy()
        for tensor in self.pack(store, summary=summary, baseline=baseline).values():
            tensor.fill_(0)
        for key in source:
            np.testing.assert_array_equal(source[key], store.arrays[key])
        np.testing.assert_array_equal(counts, summary.ref_dosage_counts)
        np.testing.assert_array_equal(baseline, self.baseline(store))
        self.assertFalse(summary.ref_dosage_counts.flags.writeable)
        self.assertFalse(summary.ref_eligible.flags.writeable)

    def test_byte_guard_precedes_packing_and_accounts_for_new_tensors(self):
        store = self.store()
        pairs = [(0, 0), (1, 1)]
        plan = M.estimate_multichannel_batch_bytes(store, pairs, mode="BOTH")
        self.assertGreater(plan["multichannel_extra_bytes"], 0)
        with patch.object(M, "pack_batch", side_effect=AssertionError("packing before budget guard")):
            with self.assertRaisesRegex(ValueError, "byte ceiling"):
                self.pack(store, max_input_bytes=plan["required_bytes"] - 1)
        self.pack(store, max_input_bytes=plan["required_bytes"])

    def test_invalid_baselines_modes_pairs_permutations_and_budgets_fail_closed(self):
        store = self.store()
        baseline = self.baseline(store)
        for bad in (baseline.astype(np.float64), baseline[:, :, :5], baseline + .1,
                    np.full_like(baseline, np.nan)):
            with self.assertRaisesRegex(ValueError, "baseline"):
                self.pack(store, baseline=bad)
        for kwargs in ({"mode": "real"}, {"pairs": []}, {"pairs": [(True, 0)]},
                       {"mode": "SHAM"}, {"sham_permutation": np.tile(np.arange(6), (2, 1))},
                       {"max_input_bytes": True}, {"device": "meta"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pack(store, **kwargs)

    def test_no_summary_needed_for_none_or_detail_and_off_never_reads_window(self):
        store = self.store()
        for mode in ("NONE", "DETAIL"):
            M.pack_multichannel_batch(store, [(0, 0)], self.baseline(store), None,
                                      mode=mode, max_input_bytes=1_000_000)
        with patch.object(store, "read_window", side_effect=AssertionError("OFF read common window")):
            self.pack(store, mode="OFF")

    def test_real_pack_drives_all_adapter_branches_and_empty_common_falls_back(self):
        for architecture in ("cnn", "attention"):
            encoder = OrderedLAIModel(OrderedModelConfig(family=architecture, width=4, depth=1,
                                      kernels=(3,), dilations=(1,), heads=1, core_sites=2))
            model = OrderedMultichannelAdapter(encoder, MultichannelConfig(4, .1))
            for mode in ("NONE", "SUMMARY", "DETAIL", "BOTH", "OFF"):
                batch = self.pack(mode=mode)
                output = model(batch, batch["baseline"], batch, mode=mode)
                self.assertTrue(bool(torch.isfinite(output["probabilities"]).all()))
                if mode != "OFF":
                    (-output["log_probabilities"][:, 0].mean()).backward()
            batch = self.pack(self.store(radius=.05))
            output = model(batch, batch["baseline"], batch, mode="BOTH")
            torch.testing.assert_close(output["probabilities"], batch["baseline"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
