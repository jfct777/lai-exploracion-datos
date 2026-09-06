from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from m33_safe_bridge_core import sample_key, write_deterministic_npz  # noqa: E402
from m39_bind_anchor_truth import bind, project_probabilities, sample_order, sha256, SCHEMA  # noqa: E402
from m37_trace_core import baseline_to_states  # noqa: E402


def make_fixture(root: Path, *, gap: bool = False) -> Path:
    people = ["synthetic-0", "synthetic-1", "synthetic-2"]
    keys = np.asarray([sample_key(p) for p in people], dtype="|S64")
    selected = {"chrom": np.full(2, 22, dtype=np.uint8), "pos": np.array([20, 40], dtype=np.int64),
                "ref": np.array([b"A", b"A"], dtype="|S1"), "alt": np.array([b"C", b"C"], dtype="|S1"),
                "cM": np.array([1., 2.]), "locus_id": np.array([110, 120], dtype=np.uint64)}
    features = {**selected, "sample_key_sha256": keys, "anchor_indices": np.arange(2, dtype=np.int64),
                "ancestry": np.array([b"AFR", b"EUR", b"NAM"], dtype="|S4"),
                "rare_semantics": np.array([b"diploid_reference_feature_not_phased_allele"], dtype="|S64")}
    write_deterministic_npz(root / "features.npz", features)
    write_deterministic_npz(root / "selected_loci.npz", selected)
    bridge = {"decision": "PASS_TECHNICAL_BRIDGE_ONLY", "boundaries": {"truth_opened": False, "target_partition": "R0_FIT"},
              "profiles": [{"sha256": sha256(root / "features.npz"), "tensor_shape": [3, 2, 2, 3, 1, 4]}]}
    (root / "bridge_receipt.json").write_text(json.dumps(bridge))
    (root / "genetic_map.tsv").write_text("22 10 0\n22 20 1\n22 30 1\n22 40 2\n22 50 3\n22 60 4\n")
    ancestry = ("AFR", "EUR", "NAM")
    with gzip.open(root / "truth_segments.tsv.gz", "wt") as handle:
        handle.write("target_id\thaplotype\tchrom\tstart_bp\tend_bp_exclusive\tancestry\n")
        for i, person in enumerate(people):
            for hap, boundary, following in ((0, 25, (i + 1) % 3), (1, 40, (i + 2) % 3)):
                handle.write(f"{person}\t{hap}\t22\t10\t{boundary}\t{ancestry[i]}\n")
                handle.write(f"{person}\t{hap}\t22\t{boundary + int(gap)}\t60\t{ancestry[following]}\n")
    permutation = np.array([2, 0, 1])
    for name, positions, cm in (("full", [10, 20, 30, 40, 50], [0., 1., 1., 2., 3.]),
                                ("fminus", [10, 30, 50], [0., 1., 3.])):
        probability = np.zeros((3, 2, len(positions), 3), dtype=np.float32)
        for row, person in enumerate(permutation):
            for marker, position in enumerate(positions):
                probability[row, 0, marker, person if position < 25 else (person + 1) % 3] = 1
                probability[row, 1, marker, person if position < 40 else (person + 2) % 3] = 1
        write_deterministic_npz(root / f"{name}.npz", {
            "sample_key_sha256": keys[permutation], "marker_chrom": np.full(len(positions), 22, dtype=np.uint8),
            "marker_pos": np.array(positions, dtype=np.int64), "marker_ref": np.full(len(positions), b"A", dtype="|S1"),
            "marker_alt": np.full(len(positions), b"C", dtype="|S1"), "F0": probability})
        write_deterministic_npz(root / f"{name}_cm.npz", {"marker_cM": np.array(cm, dtype=np.float64)})
    fold_permutation = np.array([1, 2, 0])
    roles = np.stack([np.roll(np.array(["TRAIN", "SELECT", "SCORE"]), i) for i in range(3)])
    write_deterministic_npz(root / "folds.npz", {"sample_key_sha256": keys[fold_permutation],
        "roles": roles[:, fold_permutation], "outer_fold": np.arange(3, dtype=np.uint8),
        "inner_split_seed": np.array([7, 8, 9]), "outer_seed": np.array([6])})
    filenames = {"features": "features.npz", "selected_loci": "selected_loci.npz", "bridge_receipt": "bridge_receipt.json",
                 "genetic_map": "genetic_map.tsv", "truth_segments": "truth_segments.tsv.gz",
                 "fminus": "fminus.npz", "full": "full.npz", "fminus_cm": "fminus_cm.npz",
                 "full_cm": "full_cm.npz", "folds": "folds.npz"}
    manifest = {"schema_version": SCHEMA, "scope": "exploratory_chr22_R0_FIT_no_training",
                "inputs": {name: {"path": filename, "sha256": sha256(root / filename), "uri": f"synthetic://{name}"}
                           for name, filename in filenames.items()},
                "parameters": {"people": 3, "anchors": 2, "fminus_markers": 3, "full_markers": 5, "fold": 0,
                               "role_counts": {"TRAIN": 1, "SELECT": 1, "SCORE": 1}, "write_all": False}}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class AnchorTruthBinderTest(unittest.TestCase):
    def test_probability_ties_interpolation_extremes_and_exact_override(self):
        p = np.eye(6, dtype=np.float32)[[0, 1, 3, 5]][None]
        result, audit = project_probabilities(p, np.array([0., 1., 1., 3.]),
                                              np.array([-1., 1., 2., 4.]))
        np.testing.assert_allclose(result[0, 1], .5 * (p[0, 1] + p[0, 2]))
        np.testing.assert_allclose(result[0, 2], .25 * (p[0, 1] + p[0, 2]) + .5 * p[0, 3])
        np.testing.assert_array_equal(result[0, 0], p[0, 0])
        np.testing.assert_array_equal(result[0, -1], p[0, -1])
        self.assertEqual((audit["clamped_left"], audit["clamped_right"]), (1, 1))
        exact, _ = project_probabilities(p, np.array([0., 1., 1., 3.]), np.array([1.]), np.array([1]))
        np.testing.assert_array_equal(exact[0, 0], p[0, 1])

    def test_six_state_projection_does_not_invent_cross_marker_haplotypes(self):
        hap = np.zeros((1, 2, 2, 3), dtype=np.float32)
        hap[:, :, 0, 0] = 1
        hap[:, :, 1, 1] = 1
        result, _ = project_probabilities(baseline_to_states(hap), np.array([0., 2.]), np.array([1.]))
        np.testing.assert_array_equal(result[0, 0], np.array([.5, 0., 0., .5, 0., 0.]))

    def test_sample_hash_join_reorders_and_rejects_duplicates_or_missing(self):
        keys = np.asarray([sample_key("a"), sample_key("b")], dtype="|S64")
        np.testing.assert_array_equal(sample_order(keys[::-1], keys), [1, 0])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            sample_order(keys[[0, 0]], keys)
        with self.assertRaisesRegex(ValueError, "universes"):
            sample_order(keys[:1], keys)

    def test_end_to_end_exact_truth_and_physical_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_fixture(root)
            receipt = bind(manifest, root, root / "bound")
            self.assertFalse((root / "bound/anchor-data.npz").exists())
            with np.load(root / "bound/development.npz", allow_pickle=False) as dev, \
                    np.load(root / "bound/score.npz", allow_pickle=False) as score:
                np.testing.assert_array_equal(dev["source_indices"], [0, 1])
                np.testing.assert_array_equal(dev["train_indices"], [0])
                np.testing.assert_array_equal(dev["select_indices"], [1])
                self.assertNotIn("score_indices", dev.files)
                np.testing.assert_array_equal(score["source_indices"], [2])
                np.testing.assert_array_equal(score["score_indices"], [0])
                self.assertNotIn("train_indices", score.files)
                self.assertNotIn("select_indices", score.files)
                np.testing.assert_array_equal(dev["truth_state"], [[0, 4], [3, 2]])
                np.testing.assert_array_equal(score["truth_state"], [[5, 1]])
                # Anchor20 is before the recorded breakpoint25, unlike source marker30.
                self.assertEqual(dev["truth_state"][0, 0], 0)
                self.assertEqual(dev["baseline"][0, 0].argmax(), 1)
                np.testing.assert_array_equal(dev["full_baseline"].argmax(-1), dev["truth_state"])
                self.assertEqual(len(set(dev["sample_key_sha256"]) & set(score["sample_key_sha256"])), 0)
            self.assertEqual(receipt["projection_audit"]["full"]["exact_locus"], 2)
            self.assertFalse(receipt["boundaries"]["training_performed"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                bind(manifest, root, root / "bound")

    def test_truth_gap_and_input_tampering_fail_before_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_fixture(root, gap=True)
            with self.assertRaisesRegex(ValueError, "gap or overlap"):
                bind(manifest, root, root / "bound")
            self.assertFalse((root / "bound").exists())
            altered = json.loads(manifest.read_text())
            altered["inputs"]["features"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                bind(manifest, root, root / "bound")

    def test_nonmonotonic_cm_and_bad_simplex_are_rejected(self):
        p = np.ones((1, 2, 6), dtype=np.float32) / 6
        with self.assertRaisesRegex(ValueError, "ordered"):
            project_probabilities(p, np.array([1., 0.]), np.array([.5]))
        with self.assertRaisesRegex(ValueError, "simplex"):
            project_probabilities(p * 2, np.array([0., 1.]), np.array([.5]))


if __name__ == "__main__":
    unittest.main()
