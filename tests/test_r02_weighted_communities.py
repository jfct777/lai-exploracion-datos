"""Synthetic graph mathematics and authenticated M16.5 weighted execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import r02_genomic_pair_evidence as evidence
import r02_weighted_communities as weighted


class WeightedCommunitiesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = list("abcdef")
        self.samples_path = self.root / "samples.txt"
        self.samples_path.write_text("\n".join(self.samples) + "\n")
        self.settings_path = self.root / "settings.json"
        self.settings_path.write_text(json.dumps(dict(expected_samples=6, n_seeds=3, dpi=72)))

    def arrays(self):
        # Last person has measured rare variants, none shared with anybody;
        # observed J=0 is distinct from an undefined pair with U=0.
        c = np.array([[1, 1, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 1]], dtype=np.int64)
        i = c @ c.T
        u = c.sum(axis=1)[:, None] + c.sum(axis=1)[None, :] - i
        k = np.full((6, 6), -.1)
        for a, b, value in ((0, 1, .8), (1, 2, .2), (2, 3, .1), (3, 4, .7)):
            k[a, b] = k[b, a] = value
        np.fill_diagonal(k, 1.)
        n = np.full((6, 6), 10, dtype=np.int64)
        return dict(I=i, U=u, Q=np.full((6, 6), 6, dtype=np.int64),
                    J=evidence.masked_jaccard(i, u), K=k, common_N=n,
                    common_numerator=k * n, pair_eligible_R_C=np.ones((6, 6), bool))

    def bundle(self, arrays=None, chromosomes=None):
        prefix = self.root / "aggregate"
        evidence.write_bundle(prefix, self.arrays() if arrays is None else arrays,
                              self.samples, weighted.AUTOSOMES if chromosomes is None else chromosomes,
                              "aggregate", {})
        return prefix.with_suffix(".npz")

    def test_defaults_preserve_three_gammas_and_twentyfive_seeds(self):
        self.settings_path.write_text("{}")
        result = weighted.settings_from(self.settings_path)
        self.assertEqual(result["n_seeds"], 25)
        self.assertEqual(result["expected_samples"], 2619)
        self.assertTrue(result["include_rare_graph"])
        self.assertEqual(result["resolutions"], [.5, 1., 2.])
        self.settings_path.write_text('{"resolutions":[0.5,1,2,3]}')
        with self.assertRaisesRegex(ValueError, "exactly gamma"):
            weighted.settings_from(self.settings_path)

    def test_exact_weight_formulas_and_source_unchanged(self):
        arrays = self.arrays()
        original = {k: v.copy() for k, v in arrays.items()}
        graphs, info = weighted.matrices_from(arrays, 6, True)
        self.assertEqual(set(graphs), {"R", "C", "R_plus_C"})
        self.assertEqual(graphs["R"][0, 1], 1.)
        self.assertAlmostEqual(graphs["R"][0, 2], 1. / 3.)
        self.assertEqual(graphs["C"][0, 2], 0.)
        self.assertEqual(graphs["C"][0, 1], .8)
        for graph in graphs.values():
            np.testing.assert_array_equal(np.diag(graph), np.zeros(6))
        s_r = graphs["R"][graphs["R"] > 0].mean()
        s_c = graphs["C"][graphs["C"] > 0].mean()
        np.testing.assert_allclose(graphs["R_plus_C"], .5 * graphs["R"] / s_r + .5 * graphs["C"] / s_c)
        self.assertEqual(info["n_possible_pairs"], 15)
        self.assertEqual(info["common_negative_pairs"], 11)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], original[key])

    def test_unknown_pair_rejected_not_zero_imputed(self):
        arrays = self.arrays()
        arrays["common_N"][0, 1] = arrays["common_N"][1, 0] = 0
        arrays["common_numerator"][0, 1] = arrays["common_numerator"][1, 0] = 0
        arrays["K"][0, 1] = arrays["K"][1, 0] = np.nan
        arrays["pair_eligible_R_C"][0, 1] = arrays["pair_eligible_R_C"][1, 0] = False
        with self.assertRaisesRegex(ValueError, "unevaluable"):
            weighted.matrices_from(arrays, 6, True)

    def test_similarity_must_reproduce_denominator_and_eligibility(self):
        arrays = self.arrays()
        arrays["J"][0, 1] = arrays["J"][1, 0] = .2
        with self.assertRaisesRegex(ValueError, "sufficient statistics"):
            weighted.matrices_from(arrays, 6, True)
        arrays = self.arrays()
        arrays["pair_eligible_R_C"][0, 1] = False
        with self.assertRaisesRegex(ValueError, "eligibility"):
            weighted.matrices_from(arrays, 6, True)

    def test_no_positive_common_weights_is_not_fabricated_combination(self):
        arrays = self.arrays()
        arrays["K"][:] = -.1
        arrays["common_numerator"] = arrays["K"] * arrays["common_N"]
        graphs, info = weighted.matrices_from(arrays, 6, True)
        self.assertEqual(set(graphs), {"R", "C"})
        self.assertFalse(graphs["C"].any())
        self.assertEqual(info["combination_status"], "NOT_ESTIMABLE_NO_POSITIVE_WEIGHTS")

    def test_sample_order_and_autosomal_coverage_fail_before_output(self):
        path = self.bundle(chromosomes=["22"])
        with self.assertRaisesRegex(ValueError, "22 autosomes"):
            weighted.run(path, self.samples_path, self.settings_path, self.root / "no_output")
        self.samples_path.write_text("\n".join(reversed(self.samples)) + "\n")
        with self.assertRaisesRegex(ValueError, "sample order"):
            weighted.run(path, self.samples_path, self.settings_path, self.root / "no_output")
        self.assertFalse((self.root / "no_output").exists())

    def test_synthetic_actual_core_and_render_all_three_graphs(self):
        path = self.bundle()
        output = self.root / "communities"
        result = weighted.run(path, self.samples_path, self.settings_path, output)
        self.assertEqual(result["status"], "COMPLETE_DESCRIPTIVE_WEIGHTED_COMMUNITIES")
        self.assertTrue(result["no_bp_fields"])
        self.assertTrue(result["no_missing_pairs_imputed_zero"])
        self.assertEqual(len(result["graphs"]), 3)
        for name in ("R", "C", "R_plus_C"):
            directory = output / name
            graph = sparse.load_npz(directory / "weights.npz")
            self.assertEqual(graph.shape, (6, 6))
            self.assertEqual(graph[-1].nnz, 0)
            summary = pd.read_csv(directory / "resolution_summary.tsv", sep="\t")
            self.assertTrue((summary.n_isolated == 1).all())
            self.assertEqual(summary.resolution.tolist(), [.5, 1., 2.])
            raw = pd.read_csv(directory / "seed_memberships.private.tsv.gz", sep="\t")
            self.assertEqual(len(raw), 3 * 3 * 6)
            self.assertTrue((raw.loc[raw.sample_id == "f", "community_raw"] == -1).all())
            coordinates = np.load(directory / "coordinates.private.npz", allow_pickle=False)
            self.assertEqual(coordinates["coordinates"].shape, (6, 2))
            self.assertTrue(coordinates["isolated"][-1])
            for gamma in (.5, 1., 2.):
                stem = directory / f"{name}_autosomes22_gamma{gamma:g}"
                for extension in (".png", ".pdf", ".descripcion.md"):
                    self.assertTrue(Path(str(stem) + extension).exists())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["chromosomes"], weighted.AUTOSOMES)
            for relative, digest in manifest["outputs_sha256"].items():
                self.assertEqual(evidence.sha256(directory / relative), digest)
        with self.assertRaisesRegex(ValueError, "Output exists"):
            weighted.run(path, self.samples_path, self.settings_path, output)

    def test_no_edge_graph_is_explicit_not_dummy_community(self):
        arrays = self.arrays()
        arrays["K"][:] = -.1
        arrays["common_numerator"] = arrays["K"] * arrays["common_N"]
        path = self.bundle(arrays)
        # Core execution is real; render-only behavior has its own exact test below.
        with patch.object(weighted, "draw_graph", return_value={"synthetic_render_skipped": True}):
            result = weighted.run(path, self.samples_path, self.settings_path, self.root / "empty")
        self.assertEqual(result["graphs"][1]["status"], "COMPLETE_NO_POSITIVE_EDGES")
        summary = pd.read_csv(self.root / "empty/C/resolution_summary.tsv", sep="\t")
        self.assertTrue((summary.n_active == 0).all())
        self.assertTrue((summary.n_assigned == 0).all())
        self.assertEqual(result["measurement"]["combination_status"], "NOT_ESTIMABLE_NO_POSITIVE_WEIGHTS")

    def test_common_sensitivity_skips_only_standalone_rare_execution(self):
        self.settings_path.write_text(json.dumps(dict(expected_samples=6, n_seeds=3, dpi=72,
                                                      include_rare_graph=False)))
        path = self.bundle()
        output = self.root / "sensitivity"
        expected, info = weighted.matrices_from(self.arrays(), 6, True)
        with patch.object(weighted, "draw_graph", return_value={"synthetic_render_skipped": True}):
            result = weighted.run(path, self.samples_path, self.settings_path, output)
        self.assertEqual([g["graph"] for g in result["graphs"]], ["C", "R_plus_C"])
        self.assertFalse((output / "R").exists())
        actual = sparse.load_npz(output / "R_plus_C/weights.npz").toarray()
        np.testing.assert_allclose(actual, expected["R_plus_C"])
        self.assertEqual(result["measurement"]["positive_weight_scales"], info["positive_weight_scales"])

    def test_empty_render_is_labelled_without_coordinates(self):
        directory = self.root / "plots"
        directory.mkdir()
        assignments = pd.DataFrame({f"community_res_{r:g}": np.full(6, -1) for r in (.5, 1., 2.)})
        info = weighted.draw_graph(None, sparse.csr_matrix((6, 6)), assignments, "C",
                                   weighted.settings_from(self.settings_path), directory)
        self.assertEqual(info["status"], "NO_EDGES_NO_INFORMATIVE_PROJECTION")
        self.assertFalse((directory / "coordinates.private.npz").exists())
        self.assertEqual(len(list(directory.glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
