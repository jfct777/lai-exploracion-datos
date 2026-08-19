import importlib.util
from pathlib import Path
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "bin" / "preflight_p1a_continuous_structure.py"
SPEC = importlib.util.spec_from_file_location("p1a_preflight", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class TestP1APreflight(unittest.TestCase):
    def test_component_labels_respects_threshold(self):
        kin = pd.DataFrame(
            {
                "ID1": ["a", "b", "c"],
                "ID2": ["b", "c", "d"],
                "kin": ["0.05", "0.03", "0.09"],
            }
        )
        labels = MOD.component_labels(["a", "b", "c", "d"], kin, 0.0442)
        self.assertEqual(labels["a"], labels["b"])
        self.assertNotEqual(labels["b"], labels["c"])
        self.assertEqual(labels["c"], labels["d"])

    def test_projection_ignores_eval_eval_edges_and_labels(self):
        samples = pd.DataFrame(
            {
                "sample_id": ["e0", "t1", "e1", "t0"],
                "fold": [0, 1, 1, 0],
                "region": ["S", "S", "NE", "NE"],
            }
        )
        pairs = pd.DataFrame(
            {
                "sample_a": ["e0", "e1"],
                "sample_b": ["t1", "t0"],
            }
        )
        base = MOD.projection_summary(
            pairs, samples, {"e0", "t1", "e1", "t0"}, ["S", "NE"], "all_dnabr"
        )
        # An extra e0--t0 edge is EVAL--EVAL in fold 0 and must not help projection.
        changed = pd.concat(
            [pairs, pd.DataFrame({"sample_a": ["e0"], "sample_b": ["t0"]})], ignore_index=True
        )
        changed_result = MOD.projection_summary(
            changed,
            samples.assign(region=["NE", "S", "NE", "S"]),
            {"e0", "t1", "e1", "t0"},
            ["S", "NE"],
            "all_dnabr",
        )
        base_all = base[base["region"].eq("__ALL__")].reset_index(drop=True)
        changed_all = changed_result[changed_result["region"].eq("__ALL__")].reset_index(drop=True)
        pd.testing.assert_frame_equal(base_all, changed_all)

    def test_projection_is_invariant_to_pair_orientation(self):
        samples = pd.DataFrame(
            {
                "sample_id": ["RHT_1", "1001"],
                "fold": [0, 1],
                "region": ["S", "S"],
            }
        )
        forward = pd.DataFrame({"sample_a": ["RHT_1"], "sample_b": ["1001"]})
        reverse = forward.rename(columns={"sample_a": "sample_b", "sample_b": "sample_a"})[
            ["sample_a", "sample_b"]
        ]
        expected = MOD.projection_summary(
            forward, samples, {"RHT_1", "1001"}, ["S"], "all_dnabr"
        )
        observed = MOD.projection_summary(
            reverse, samples, {"RHT_1", "1001"}, ["S"], "all_dnabr"
        )
        pd.testing.assert_frame_equal(expected, observed)

    def test_reserved_fold_is_rejected_from_analysis_ids(self):
        samples = pd.DataFrame(
            {
                "sample_id": ["dev", "reserved"],
                "fold": [0, MOD.RESERVED_FOLD],
                "region": ["S", "S"],
            }
        )
        pairs = pd.DataFrame({"sample_a": ["dev"], "sample_b": ["reserved"]})
        with self.assertRaises(MOD.ContractError):
            MOD.projection_summary(
                pairs, samples, {"dev", "reserved"}, ["S"], "all_dnabr"
            )

    def test_identical_duplicate_is_allowed_but_conflict_fails(self):
        frame = pd.DataFrame({"ID": ["x", "x"], "value": [1, 1]})
        result = MOD.deduplicate_identical(frame, "ID", "metadata")
        self.assertEqual(len(result), 1)
        conflicting = pd.DataFrame({"ID": ["x", "x"], "value": [1, 2]})
        with self.assertRaises(MOD.ContractError):
            MOD.deduplicate_identical(conflicting, "ID", "metadata")


if __name__ == "__main__":
    unittest.main()
