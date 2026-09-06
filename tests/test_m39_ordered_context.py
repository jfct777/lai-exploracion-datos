"""Synthetic ordered-context known answers; no production assets or training."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
import m39_carrier_context as B
import m39_ordered_context as O
from test_m39_carrier_context import fixture


class OrderedContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path, _ = fixture(self.root)
        config, paths = B.load_config(path, self.root)
        self.data = B.extract_inputs(paths, config)

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, data=None, *, radius=0.5, k=3, canonicalize=True):
        data = self.data if data is None else data
        if canonicalize:
            data = O.canonicalize_reference_homologs(data)
        candidates, _ = B.materialize_radius(data, np.arange(2, dtype=np.int64), k, radius)
        return O.build_factorized_store(data, candidates), candidates

    def window(self, store, q=0, j=0, **kwargs):
        return store.read_window(q, j, max_window_bytes=1_000_000, **kwargs)

    def saved(self):
        store, _ = self.store()
        directory = self.root / "stores" / "case"
        receipt = store.save(directory, source_hashes={"fixture": "a" * 64})
        return store, directory, receipt

    def test_factorized_matches_deltas_and_diploid_values_have_exact_known_answers(self):
        store, candidates = self.store()
        for q, j in np.ndindex(2, 2):
            w = self.window(store, q, j)
            a = store.arrays
            sites = np.flatnonzero(np.abs(a["common_cm"] - a["cM"][j]) <= 0.5)
            np.testing.assert_array_equal(w.common_site_index, sites)
            np.testing.assert_array_equal(w.delta_cm, a["common_cm"][sites] - a["cM"][j])
            self.assertEqual(w.channels.shape, (2, 3, 3, len(sites), 5))
            self.assertEqual(w.channels.dtype, np.dtype("uint8"))
            for h, anc, k in np.ndindex(2, 3, 3):
                if not w.candidate_mask[h, anc, k]:
                    self.assertTrue((w.channels[h, anc, k] == 0).all())
                    self.assertEqual(w.candidate_original_hap[h, anc, k], -1)
                    continue
                p, lane = w.candidate_ref_index[h, anc, k], w.candidate_hap[h, anc, k]
                for offset, site in enumerate(sites):
                    qa, ra = a["common_target"][site, q, h], a["common_ref"][site, p, lane]
                    joint = qa >= 0 and ra >= 0
                    expected = [joint and qa == ra, joint and qa != ra, qa < 0, ra < 0, joint]
                    np.testing.assert_array_equal(w.channels[h, anc, k, offset], expected)
                self.assertEqual(w.ref_dosage[h, anc, k], a["ref_dosage"][j, p])
                self.assertEqual(w.ref_observed[h, anc, k], a["ref_observed"][j, p])
                self.assertEqual(w.candidate_original_hap[h, anc, k], a["reference_hap_original_index"][p, lane])
            np.testing.assert_array_equal(w.ref_dosage, candidates["ref_dosage"][q, j])
            np.testing.assert_array_equal(w.ref_observed, candidates["rare_ref_observed"][q, j])

    def test_equal_mismatch_counts_do_not_erase_order(self):
        first = copy.deepcopy(self.data)
        first["common_ref"][:] = np.asarray([0, 0, 1, 1])[:, None, None]
        first["common_target"][:, :, 0] = 0
        first["common_target"][:, :, 1] = 1
        second = copy.deepcopy(first)
        second["common_ref"][:] = np.asarray([0, 1, 0, 1])[:, None, None]
        a, ac = self.store(first, radius=1.0)
        b, bc = self.store(second, radius=1.0)
        np.testing.assert_array_equal(ac["common_context"], bc["common_context"])
        np.testing.assert_array_equal(ac["candidate_ref_index"], bc["candidate_ref_index"])
        wa, wb = self.window(a), self.window(b)
        np.testing.assert_array_equal(wa.channels.sum(axis=3), wb.channels.sum(axis=3))
        self.assertFalse(np.array_equal(wa.channels, wb.channels))

    def test_missing_is_not_a_reference_allele_or_a_match(self):
        store, _ = self.store(radius=1.0)
        w = self.window(store)
        query_missing = w.channels[..., 2].astype(bool)
        ref_missing = w.channels[..., 3].astype(bool)
        self.assertTrue(query_missing.any())
        self.assertTrue(ref_missing.any())
        missing = query_missing | ref_missing
        self.assertFalse(w.channels[..., 0][missing].any())
        self.assertFalse(w.channels[..., 1][missing].any())
        self.assertFalse(w.channels[..., 4][missing].any())
        np.testing.assert_array_equal(w.channels[..., 0] + w.channels[..., 1], w.channels[..., 4])

    def test_changing_rare_values_cannot_change_common_order_or_retrieval(self):
        changed = copy.deepcopy(self.data)
        for dosage, mask in (("ref_dosage", "ref_observed"), ("query_dosage", "query_observed")):
            changed[dosage] = np.where(changed[mask], 2 - changed[dosage], 0).astype(np.int8)
        a, ac = self.store()
        b, bc = self.store(changed)
        for name in ("common_context", "candidate_mask", "candidate_ref_index", "candidate_hap"):
            np.testing.assert_array_equal(ac[name], bc[name])
        for q, j in np.ndindex(2, 2):
            np.testing.assert_array_equal(self.window(a, q, j).channels, self.window(b, q, j).channels)
        self.assertFalse(np.array_equal(a.arrays["query_dosage"], b.arrays["query_dosage"]))

    def test_query_homolog_swap_is_equivariant_without_changing_rare_dosage(self):
        changed = copy.deepcopy(self.data)
        changed["common_target"] = changed["common_target"][:, :, ::-1].copy()
        original, _ = self.store()
        swapped, _ = self.store(changed)
        for q, j in np.ndindex(2, 2):
            a, b = self.window(original, q, j), self.window(swapped, q, j)
            for name in ("channels", "candidate_ref_index", "candidate_hap", "ref_dosage", "ref_observed"):
                np.testing.assert_array_equal(getattr(a, name)[::-1], getattr(b, name))
            self.assertEqual(a.query_dosage, b.query_dosage)

    def test_reference_homolog_canonicalization_fixes_ties_crossing_k(self):
        original = copy.deepcopy(self.data)
        original["common_target"][:] = 0
        # Both lanes have equal mismatch count but different ordered sequence.
        original["common_ref"][:, :, 0] = np.asarray([0, 0, 1, 1])[:, None]
        original["common_ref"][:, :, 1] = np.asarray([0, 1, 0, 1])[:, None]
        changed = copy.deepcopy(original)
        changed["common_ref"] = changed["common_ref"][:, :, ::-1].copy()
        a, ac = self.store(original, radius=1.0, k=1)
        b, bc = self.store(changed, radius=1.0, k=1)
        np.testing.assert_array_equal(a.arrays["common_ref"], b.arrays["common_ref"])
        for name in ("candidate_ref_index", "candidate_hap", "common_context"):
            np.testing.assert_array_equal(ac[name], bc[name])
        np.testing.assert_array_equal(self.window(a).channels, self.window(b).channels)
        valid = self.window(a).candidate_mask
        np.testing.assert_array_equal(self.window(a).candidate_original_hap[valid],
                                      1 - self.window(b).candidate_original_hap[valid])

    def test_identical_reference_lanes_are_interchangeable_and_canonicalization_idempotent(self):
        data = copy.deepcopy(self.data)
        data["common_ref"][:, :, 1] = data["common_ref"][:, :, 0]
        original = data["common_ref"].copy()
        once = O.canonicalize_reference_homologs(data)
        twice = O.canonicalize_reference_homologs(once)
        np.testing.assert_array_equal(data["common_ref"], original)
        np.testing.assert_array_equal(once["common_ref"], twice["common_ref"])
        np.testing.assert_array_equal(once["reference_hap_original_index"], twice["reference_hap_original_index"])
        swapped = copy.deepcopy(data)
        swapped["common_ref"] = swapped["common_ref"][:, :, ::-1].copy()
        a, _ = self.store(data)
        b, _ = self.store(swapped)
        np.testing.assert_array_equal(self.window(a).channels, self.window(b).channels)

    def test_common_cm_ties_keep_distinct_site_identity_in_genomic_order(self):
        data = copy.deepcopy(self.data)
        data["common_cm"][2] = data["common_cm"][1]
        store, _ = self.store(data)
        w = self.window(store)
        self.assertIn(1, w.common_site_index)
        self.assertIn(2, w.common_site_index)
        np.testing.assert_array_equal(w.common_site_index, [0, 1, 2])
        self.assertEqual(w.delta_cm[1], w.delta_cm[2])
        self.assertNotEqual(store.arrays["common_locus_id"][1], store.arrays["common_locus_id"][2])

    def test_empty_windows_padding_and_unsupported_queries(self):
        store, _ = self.store(radius=0.05, k=8)
        windows = list(store.iter_windows(site_chunk_size=3, max_window_bytes=1_000_000))
        self.assertEqual(len(windows), 4)
        for w in windows:
            self.assertEqual(w.channels.shape, (2, 3, 8, 0, 5))
            self.assertEqual(w.window_site_count, 0)
            self.assertTrue((w.candidate_ref_index == -1).all())
            self.assertFalse(w.candidate_mask.any())
            self.assertFalse(w.ref_observed.any())
        data = copy.deepcopy(self.data)
        data["common_target"][:] = -1
        unsupported, _ = self.store(data)
        self.assertFalse(self.window(unsupported).candidate_mask.any())
        self.assertFalse(self.window(unsupported).channels.any())

    def test_chunk_invariance_and_candidate_identity_are_exact(self):
        store, _ = self.store(radius=1.0)
        full = self.window(store)
        for chunk_size in (1, 2, 3, 10):
            chunks = list(store.iter_windows(query_indices=[0], anchor_indices=[0],
                                            site_chunk_size=chunk_size, max_window_bytes=1_000_000))
            np.testing.assert_array_equal(np.concatenate([w.channels for w in chunks], axis=3), full.channels)
            np.testing.assert_array_equal(np.concatenate([w.delta_cm for w in chunks]), full.delta_cm)
            np.testing.assert_array_equal(np.concatenate([w.common_site_index for w in chunks]), full.common_site_index)
            for w in chunks:
                np.testing.assert_array_equal(w.candidate_ref_index, full.candidate_ref_index)
                np.testing.assert_array_equal(w.candidate_hap, full.candidate_hap)
                self.assertEqual(w.window_site_count, full.window_site_count)

    def test_byte_ceiling_checked_before_channel_allocation_and_auto_chunks(self):
        store, _ = self.store(radius=1.0)
        with patch.object(O.np, "zeros", side_effect=AssertionError("allocated channels")):
            with self.assertRaisesRegex(ValueError, "byte ceiling"):
                store.read_window(0, 0, max_window_bytes=store.window_working_bytes(1))
        ceiling = store.window_working_bytes(1)
        chunks = list(store.iter_windows(query_indices=[0], anchor_indices=[0], site_chunk_size=100,
                                        max_window_bytes=ceiling))
        self.assertEqual(len(chunks), len(store.arrays["common_cm"]))
        self.assertTrue(all(w.nbytes <= ceiling and w.channels.shape[3] == 1 for w in chunks))
        with self.assertRaisesRegex(ValueError, "one site"):
            list(store.iter_windows(site_chunk_size=1, max_window_bytes=store.window_working_bytes(0)))

    def test_store_shares_common_matrices_without_global_expansion(self):
        store, _ = self.store(canonicalize=False)
        self.assertTrue(np.shares_memory(store.arrays["common_ref"], self.data["common_ref"]))
        self.assertTrue(np.shares_memory(store.arrays["common_target"], self.data["common_target"]))
        self.assertEqual(store.arrays["common_ref"].ndim, 3)
        self.assertTrue(all(a.ndim <= 5 for a in store.arrays.values()))
        self.assertNotIn("common_context", store.arrays)
        self.assertNotIn("pooled_summary", store.arrays)
        self.assertEqual(store.factorized_nbytes, sum(a.nbytes for a in store.arrays.values()))

    def test_mmap_roundtrip_deterministic_hashes_readonly_and_no_overwrite(self):
        store, directory, receipt = self.saved()
        opened = O.OrderedContextStore.open(directory, expected_manifest_sha256=receipt["manifest_sha256"])
        for name, array in store.arrays.items():
            np.testing.assert_array_equal(array, opened.arrays[name])
            self.assertFalse(opened.arrays[name].flags.writeable)
            self.assertIsInstance(opened.arrays[name], np.memmap)
        np.testing.assert_array_equal(self.window(store).channels, self.window(opened).channels)
        second = store.save(self.root / "second", source_hashes={"fixture": "a" * 64})
        self.assertEqual(receipt, second)
        with self.assertRaisesRegex(ValueError, "already exists"):
            store.save(directory, source_hashes={"fixture": "a" * 64})
        with self.assertRaises(ValueError):
            opened.arrays["common_ref"][0, 0, 0] = 1

    def test_tampered_array_rejected_before_any_numpy_load(self):
        _, directory, receipt = self.saved()
        path = directory / "common_ref.npy"
        os.chmod(path, 0o600)
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            byte = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([byte[0] ^ 1]))
        with patch.object(O.np, "load", side_effect=AssertionError("unverified array loaded")):
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                O.OrderedContextStore.open(directory, expected_manifest_sha256=receipt["manifest_sha256"])

    def test_manifest_tampering_unexpected_files_and_symlinks_fail_closed(self):
        _, directory, receipt = self.saved()
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            O.OrderedContextStore.open(directory, expected_manifest_sha256="f" * 64)
        extra = directory / "undeclared.txt"
        extra.write_text("synthetic extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            O.OrderedContextStore.open(directory, expected_manifest_sha256=receipt["manifest_sha256"])
        extra.unlink()
        file = directory / "common_ref.npy"
        moved = self.root / "reference.npy"
        file.rename(moved)
        file.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, "hash/size"):
            O.OrderedContextStore.open(directory, expected_manifest_sha256=receipt["manifest_sha256"])

    def test_reauthenticated_manifest_still_rejects_path_traversal_and_metadata_drift(self):
        _, directory, _ = self.saved()
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["arrays"]["common_ref"]["file"] = "../reference.npy"
        os.chmod(path, 0o600)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "descriptor/path"):
            O.OrderedContextStore.open(directory, expected_manifest_sha256=B.sha256_file(path))
        manifest["arrays"]["common_ref"]["file"] = "common_ref.npy"
        manifest["arrays"]["common_ref"]["shape"][0] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            O.OrderedContextStore.open(directory, expected_manifest_sha256=B.sha256_file(path))

    def test_invalid_candidate_keys_rare_values_phase_or_unknown_inputs_are_rejected(self):
        data = O.canonicalize_reference_homologs(self.data)
        _, original = self.store(data, canonicalize=False)
        for change, message in (
            (lambda c: c.update(truth=np.zeros(1)), "undeclared"),
            (lambda c: c["sample_key_sha256"].__setitem__(0, b"0" * 64), "axes differ"),
            (lambda c: c["candidate_ref_index"].__setitem__((0, 0, 0, 0, 0), 99), "candidate person"),
            (lambda c: c["ref_dosage"].__setitem__((0, 0, 0, 0, 0), 2), "attached diploid"),
            (lambda c: c.update(rare_semantics=np.asarray(["phased"], dtype="S64")), "semantics"),
        ):
            candidates = copy.deepcopy(original)
            change(candidates)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                O.build_factorized_store(data, candidates)

    def test_invalid_common_axis_masks_and_k_are_rejected(self):
        for key, mutate, message in (
            ("common_cm", lambda x: x.__setitem__(0, 99), "common cM"),
            ("common_ref", lambda x: x.__setitem__((0, 0, 0), 2), "alleles"),
            ("common_locus_id", lambda x: x.__setitem__(0, 1), "canonical locus"),
            ("candidate_mask", lambda x: x.__setitem__((0, 0, 0, 0, 0), 2), "binary"),
            ("query_observed", lambda x: x.__setitem__((0, 0), 2), "binary"),
        ):
            store, _ = self.store()
            arrays = {name: a.copy() for name, a in store.arrays.items()}
            if arrays[key].dtype.kind == "b":
                arrays[key] = arrays[key].astype(np.uint8)
            mutate(arrays[key])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                O.OrderedContextStore(arrays)
        with self.assertRaisesRegex(ValueError, "K ceiling"):
            self.store(k=9)


if __name__ == "__main__":
    unittest.main()
