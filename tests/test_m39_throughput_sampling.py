"""Sampling and finite-population cost arithmetic on artificial metadata only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import m39_throughput_sampling as S


class MetadataStore:
    """No genotype matrices or labels exist in this artificial store."""
    shape = (48, 660, 8)

    def __init__(self, lengths=None):
        self.lengths = np.asarray(lengths if lengths is not None else
                                  [100 + (j * 37) % 2000 for j in range(660)], dtype=np.int64)
        self.arrays = {"sample_key_sha256": np.asarray([
            hashlib.sha256(f"artificial query {i}".encode()).hexdigest() for i in range(48)], dtype="S64"),
            "locus_id": np.arange(660, 0, -1, dtype=np.uint64)}
        for value in self.arrays.values():
            value.flags.writeable = False

    def window_bounds(self, anchor):
        return 3, 3 + int(self.lengths[anchor])


def inventory(rows):
    return sorted(tuple(pair) for row in rows for pair in row["pairs"])


class ThroughputSamplingTests(unittest.TestCase):
    def setUp(self):
        self.store = MetadataStore()
        self.manifest = S.build_sampling_manifest(self.store, 20260907)

    def rows(self, batch=1, value=1.0):
        rows = S.steps_for_case(self.manifest, batch, "grouped")
        for row in rows:
            row["e2e_seconds"] = value
        return rows

    def test_reproducible_independent_streams_and_json_roundtrip(self):
        before = np.random.get_state()
        repeated = S.build_sampling_manifest(self.store, 20260907)
        after = np.random.get_state()
        self.assertEqual(repeated, self.manifest)
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertNotEqual(S.build_sampling_manifest(self.store, 20260908)["anchor_samples"],
                            self.manifest["anchor_samples"])
        restored = json.loads(json.dumps(self.manifest))
        self.assertEqual(S.steps_for_case(restored, 2, "random"),
                         S.steps_for_case(self.manifest, 2, "random"))

    def test_sixteen_distinct_anchors_and_thirty_two_distinct_queries(self):
        samples = self.manifest["anchor_samples"]
        self.assertEqual(len({sample["anchor_index"] for sample in samples}), 16)
        pairs = [pair for sample in samples for pair in sample["pairs"]]
        self.assertEqual(len(pairs), 32)
        self.assertEqual(len({q for q, _ in pairs}), 32)
        for h in range(4):
            stratum = self.manifest["strata"][h]
            self.assertEqual(stratum["anchor_count"], 165)
            self.assertEqual(stratum["pair_count"], 7920)
            self.assertEqual(stratum["anchor_inclusion_fraction"], [4, 165])
            self.assertEqual(sum(sample["stratum"] == h for sample in samples), 4)
        for sample in samples:
            for pair, key in zip(sample["pairs"], sample["query_key_sha256"]):
                self.assertEqual(key, bytes(self.store.arrays["sample_key_sha256"][pair[0]]).decode())

    def test_equal_lengths_are_rank_partitioned_with_locus_tie_break(self):
        store = MetadataStore([100] * 660)
        manifest = S.build_sampling_manifest(store, 23)
        order = [j for stratum in manifest["strata"] for j in stratum["anchor_indices"]]
        self.assertEqual(order, list(range(659, -1, -1)))
        self.assertTrue(all(x["anchor_count"] == 165 for x in manifest["strata"]))

    def test_all_policies_have_identical_pair_inventory(self):
        single = S.steps_for_case(self.manifest, 1, "grouped")
        grouped = S.steps_for_case(self.manifest, 2, "grouped")
        random = S.steps_for_case(self.manifest, 2, "random")
        self.assertEqual((len(single), len(grouped), len(random)), (32, 16, 16))
        self.assertEqual(inventory(single), inventory(grouped))
        self.assertEqual(inventory(grouped), inventory(random))
        self.assertTrue(all(row["stratum"] is None for row in random))
        self.assertTrue(all(len({j for _, j in row["pairs"]}) == 1 for row in grouped))
        for rows in (single, grouped, random):
            for row in rows:
                self.assertEqual(row["anchor_lengths"], [int(self.store.lengths[j]) for _, j in row["pairs"]])

    def test_warmup_is_separate_dense_and_not_sampled(self):
        warmup = self.manifest["warmup"]
        self.assertIsNone(warmup["stratum"])
        self.assertEqual(warmup["anchor_lengths"], [int(self.store.lengths.max())] * 2)
        self.assertTrue(set(map(tuple, warmup["pairs"])).isdisjoint(inventory(self.rows())))

    def test_manifest_source_and_returned_rows_are_not_mutated(self):
        original = copy.deepcopy(self.manifest)
        arrays = {name: value.copy() for name, value in self.store.arrays.items()}
        rows = S.steps_for_case(self.manifest, 2, "grouped")
        rows[0]["pairs"][0][0] = -1
        rows[0]["anchor_lengths"][0] = -1
        self.assertEqual(self.manifest, original)
        for name, array in self.store.arrays.items():
            np.testing.assert_array_equal(array, arrays[name])
            self.assertFalse(array.flags.writeable)

    def test_constant_time_epoch_estimates_use_batch_denominators(self):
        for batch in (1, 2):
            result = S.estimate_grouped_epoch(self.rows(batch, 3.0), self.manifest, batch)
            self.assertEqual(result["estimated_grouped_pass_seconds"], 31680 / batch * 3)
            self.assertEqual(result["estimated_pairs_per_second"], batch / 3)
            for stratum in result["strata"]:
                self.assertEqual(stratum["sampled_anchors"], 4)
                self.assertEqual(stratum["measured_batches"], 8 // batch)
                self.assertEqual(stratum["population_batches"], 7920 // batch)

    def test_arithmetic_stratified_weighting_not_mean_throughput(self):
        rows = self.rows(batch=2)
        for row in rows:
            row["e2e_seconds"] = float(2 ** row["stratum"])
        result = S.estimate_grouped_epoch(rows, self.manifest, 2)
        expected = 3960 * (1 + 2 + 4 + 8)
        self.assertEqual(result["estimated_grouped_pass_seconds"], expected)
        self.assertEqual(result["estimated_pairs_per_second"], 31680 / expected)
        self.assertEqual(result["descriptive_minmax_pass_seconds"], [expected, expected])
        self.assertIn("not CI", result["uncertainty"])

    def test_estimate_is_independent_of_measurement_row_order(self):
        rows = self.rows(1)
        for i, row in enumerate(rows):
            row["e2e_seconds"] = 0.1 + i
        self.assertEqual(S.estimate_grouped_epoch(rows, self.manifest, 1),
                         S.estimate_grouped_epoch(list(reversed(rows)), self.manifest, 1))

    def test_invalid_seed_shape_keys_windows_and_policy_are_rejected(self):
        for seed in (True, -1, 2**32, 3.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                S.build_sampling_manifest(self.store, seed)
        for batch, policy in ((True, "grouped"), (3, "grouped"), (1, "random"), (2, "other")):
            with self.subTest(batch=batch, policy=policy), self.assertRaises(ValueError):
                S.steps_for_case(self.manifest, batch, policy)
        invalid = MetadataStore()
        invalid.shape = (96, 660, 8)
        with self.assertRaises(ValueError):
            S.build_sampling_manifest(invalid, 0)
        invalid = MetadataStore([0] * 660)
        with self.assertRaises(ValueError):
            S.build_sampling_manifest(invalid, 0)

    def test_manifest_weights_duplicates_and_length_drift_are_rejected(self):
        examples = []
        bad = copy.deepcopy(self.manifest)
        bad["strata"][0]["pair_count"] = 7921
        examples.append(bad)
        bad = copy.deepcopy(self.manifest)
        bad["anchor_samples"][0] = copy.deepcopy(bad["anchor_samples"][1])
        examples.append(bad)
        bad = copy.deepcopy(self.manifest)
        bad["anchor_lengths"][0] += 1
        examples.append(bad)
        bad = copy.deepcopy(self.manifest)
        bad["random_pair_order"][0] = bad["random_pair_order"][1]
        examples.append(bad)
        for example in examples:
            with self.assertRaises(ValueError):
                S.steps_for_case(example, 2, "grouped")

    def test_estimator_rejects_warmup_random_missing_or_invalid_times(self):
        bad_rows = []
        bad_rows.append(self.rows(2)[:-1])
        bad = self.rows(2)
        bad[0] = copy.deepcopy(bad[1])
        bad_rows.append(bad)
        bad_rows.append([{**r, "e2e_seconds": 1.0} for r in S.steps_for_case(self.manifest, 2, "random")])
        for value in (0.0, -1.0, True, float("nan"), float("inf"), "1"):
            bad = self.rows(2)
            bad[0]["e2e_seconds"] = value
            bad_rows.append(bad)
        bad = self.rows(2)
        bad[0]["stratum"] = None
        bad_rows.append(bad)
        bad = self.rows(2)
        bad[0]["pairs"][0][0] = float(bad[0]["pairs"][0][0])
        bad_rows.append(bad)
        for example in bad_rows:
            with self.assertRaises(ValueError):
                S.estimate_grouped_epoch(example, self.manifest, 2)


if __name__ == "__main__":
    unittest.main()
