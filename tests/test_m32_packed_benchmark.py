import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
SPEC = importlib.util.spec_from_file_location("m32_packed_benchmark", ROOT / "bin" / "m32_packed_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class M32PackedCoreTests(unittest.TestCase):
    def test_contract_freezes_truth_free_grid(self):
        contract, root = MODULE.load_contract(
            ROOT / "conf" / "m32_packed_benchmark_preregistration.json", "root17", 20260817
        )
        self.assertFalse(contract["truth_policy"]["training_authorized"])
        self.assertFalse(contract["truth_policy"]["selects_radius"])
        self.assertEqual(root["role"], "consumed_technical_only")

    def test_interval_is_inclusive_and_matches_explicit_csr_with_ties(self):
        grid = np.asarray([0.10, 0.20, 0.30])
        rare = np.asarray([0.00, 0.10, 0.10, 0.20, 0.30, 0.40])
        starts, stops = MODULE.interval_arrays(grid, rare, 0.10)
        oracle_starts, oracle_stops = MODULE.streaming_interval_oracle(grid, rare, 0.10)
        np.testing.assert_array_equal(starts, oracle_starts)
        np.testing.assert_array_equal(stops, oracle_stops)
        self.assertEqual(starts.tolist(), [0, 1, 3])
        self.assertEqual(stops.tolist(), [4, 5, 6])
        indptr, indices = MODULE.explicit_csr_oracle(starts, stops)
        reconstructed = [indices[indptr[i]:indptr[i + 1]].tolist() for i in range(grid.size)]
        self.assertEqual(reconstructed, [list(range(a, b)) for a, b in zip(starts, stops)])

    def test_target_missing_never_becomes_hom_ref(self):
        answer = MODULE.missing_known_answer()
        self.assertTrue(answer["target_missing_is_not_hom_ref"])
        self.assertTrue(answer["ref_zero_count_with_denominator_is_observed_absence"])
        self.assertTrue(answer["ref_zero_denominator_is_unobserved"])
        self.assertTrue(answer["flare_h0_h1_distinct"])

    def test_reference_denominators_are_alleles_not_people(self):
        dosage = np.asarray([[0, 2, 1, 0, 2, 2], [1, 1, 0, 0, 0, 0]], dtype=np.int8)
        labels = ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"]
        channels = MODULE.reference_channels(dosage, labels)
        self.assertEqual(channels["ref_minor_count"].tolist(), [[2, 1, 4], [2, 0, 0]])
        self.assertEqual(channels["ref_callable_alleles"].tolist(), [[4, 4, 4], [4, 4, 4]])
        np.testing.assert_allclose(channels["ref_support"], [[0.5, 0.25, 1.0], [0.5, 0.0, 0.0]])
        self.assertEqual(channels["ref_dosage"].shape, (2, 6))
        self.assertEqual(channels["ref_label_codes"].tolist(), [0, 0, 1, 1, 2, 2])

    def test_uint16_overflow_is_rejected(self):
        dosage = np.zeros((1, 32770), dtype=np.int8)
        labels = ["AFR"] * 32768 + ["EUR", "ASIA"]
        with self.assertRaisesRegex(ValueError, "overflows uint16"):
            MODULE.reference_channels(dosage, labels)

    def test_every_batch_budget_backend_preserves_semantic_totals(self):
        arrays = {
            "target_minor_dosage": np.asarray([[0, 1, 2], [1, 0, 1], [2, 2, 0], [0, 1, 1]], dtype=np.int8),
            "target_observed_mask": np.ones((4, 3), dtype=bool),
            "target_haplotype_presence": np.zeros((4, 3, 2), dtype=np.int8),
            "minor_codes": np.asarray([0, 1, 0, 1], dtype=np.int8),
            "rare_bp": np.asarray([10, 20, 30, 40], dtype=np.int64),
            "ref_dosage": np.asarray([[0, 1, 2, 0, 1, 2]] * 4, dtype=np.int8),
            "ref_minor_count": np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint16),
            "ref_callable_alleles": np.full((4, 3), 4, dtype=np.uint16),
            "ref_observed": np.ones((4, 6), dtype=bool),
            "ref_support": np.asarray([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25], [0.25, 0.25, 0.25]], dtype=np.float32),
            "ref_support_observed_mask": np.ones((4, 3), dtype=bool),
            "ref_minor_supported": np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool),
            "rare_cm": np.asarray([0.0, 0.1, 0.2, 0.3]),
            "grid_cm": np.asarray([0.05, 0.15, 0.25]),
            "flare_raw": np.full((3, 3, 2, 3), 1 / 3, dtype=np.float32),
        }
        starts = np.asarray([0, 1, 1], dtype=np.int32)
        stops = np.asarray([2, 3, 4], dtype=np.int32)
        markers = np.arange(3, dtype=np.int32)
        observed = []
        for backend in ("contiguous_packed", "length_sorted_packed"):
            for batch in (1, 2, 3):
                for budget in (9, 12, 24):
                    output = MODULE.consume_once(arrays, starts, stops, markers, batch, budget, backend)
                    observed.append({key: output[key] for key in (
                        "tokens", "marker_person_contexts", "dosage_sum", "observed_count",
                        "ref_minor_count_sum", "ref_callable_alleles_sum", "truncations",
                        "assignment_sha256",
                    )})
        self.assertTrue(all(item == observed[0] for item in observed))
        self.assertEqual(observed[0]["truncations"], 0)

    def test_single_context_over_budget_fails_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "single context"):
            MODULE.marker_chunks(np.asarray([0]), np.asarray([9]), people=8, budget=64)

    def test_marker_sampling_is_deterministic_and_keeps_extremes(self):
        lengths = np.asarray([10, 1, 8, 3, 7, 2, 9, 4, 6, 5])
        first = MODULE.deterministic_marker_sample(lengths, 5)
        second = MODULE.deterministic_marker_sample(lengths, 5)
        np.testing.assert_array_equal(first, second)
        self.assertIn(int(np.argmin(lengths)), first)
        self.assertIn(int(np.argmax(lengths)), first)

    def test_atomic_writer_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            MODULE.write_json_atomic(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                MODULE.write_json_atomic(path, {"a": 2})

    def test_atomic_npz_refuses_overwrite_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor.npz"
            arrays = {"x": np.arange(6, dtype=np.int8).reshape(2, 3)}
            MODULE.write_npz_atomic(path, arrays)
            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["x"], arrays["x"])
            with self.assertRaises(FileExistsError):
                MODULE.write_npz_atomic(path, arrays)

    def test_padded_known_answer_masks_every_sentinel(self):
        self.assertTrue(MODULE.padded_roundtrip_known_answer()["packed_equals_padded_after_mask"])

    def test_contract_rejects_load_bearing_mutation(self):
        original = json.loads((ROOT / "conf" / "m32_packed_benchmark_preregistration.json").read_text())
        mutations = (
            (lambda value: value["memory"].__setitem__("stop_fraction", 0.9), "memory contract"),
            (lambda value: value["dtypes"].__setitem__("ref_support", "float64"), "dtype contract"),
            (lambda value: value["reference_support_policy"].__setitem__(
                "freq_and_ref_counts_are_not_required_to_match", False), "reference support policy"),
            (lambda value: value["stop_conditions"].pop(), "stop-condition contract"),
        )
        for mutate, message in mutations:
            contract = json.loads(json.dumps(original))
            mutate(contract)
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "contract.json"
                path.write_text(json.dumps(contract))
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.load_contract(path, "root17", 20260817)

    def test_content_hash_detects_equal_sum_locus_permutation(self):
        arrays = {
            "grid_cm": np.asarray([0.1]), "rare_bp": np.asarray([10, 20]),
            "rare_cm": np.asarray([0.05, 0.15]), "minor_codes": np.asarray([0, 1], dtype=np.int8),
            "target_minor_dosage": np.asarray([[0], [2]], dtype=np.int8),
            "target_observed_mask": np.ones((2, 1), dtype=bool),
            "target_haplotype_presence": np.asarray([[[0, 0]], [[1, 1]]], dtype=np.int8),
            "ref_dosage": np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int8),
            "ref_observed": np.ones((2, 3), dtype=bool),
            "ref_minor_count": np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.uint16),
            "ref_callable_alleles": np.full((2, 3), 2, dtype=np.uint16),
            "ref_support": np.asarray([[0, .5, 1], [1, .5, 0]], dtype=np.float32),
            "ref_support_observed_mask": np.ones((2, 3), dtype=bool),
            "ref_minor_supported": np.asarray([[0, 1, 1], [1, 1, 0]], dtype=bool),
            "flare_raw": np.asarray([[[[.8, .1, .1], [.1, .8, .1]]]], dtype=np.float32),
        }
        starts, stops, markers = np.asarray([0]), np.asarray([2]), np.asarray([0])
        first = MODULE.expected_content_hash(arrays, starts, stops, markers)
        arrays["target_minor_dosage"] = arrays["target_minor_dosage"][::-1].copy()
        second = MODULE.expected_content_hash(arrays, starts, stops, markers)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
