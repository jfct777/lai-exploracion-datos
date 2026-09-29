"""Small numerical regressions; no DNABR inputs, cloud or real clustering.

Run in the existing M16.5 image with unittest. AST loading isolates the
production functions from optional plotting/UMAP imports and logger setup;
the tested bodies are compiled verbatim from the current source file.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import logging
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import warnings

try:
    import igraph as ig
    import leidenalg as la
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import adjusted_rand_score
    HAVE_NUMERICAL_DEPS = True
except ImportError:
    HAVE_NUMERICAL_DEPS = False


SOURCE = Path(__file__).resolve().parents[1] / "bin" / "ibd_community_enhanced.py"


def load_core():
    tree = ast.parse(SOURCE.read_text())
    wanted = {
        "_fail", "load_individuals", "load_pair_summary", "_validate_pair_rows",
        "load_segments_aggregated", "_compute_edge_weight", "aggregate_pair_weights",
        "build_sparse_matrix", "sparse_to_igraph", "_relabel_small_communities",
        "_leiden_single", "run_leiden_multiresolution", "compute_ari_multi_seed",
        "compute_assignment_confidence", "validate_intra_vs_inter",
        "_random_nonneg_init", "_nndsvd_init", "symnmf", "laplacian_normalize",
        "_symnmf_dominant_components", "run_symnmf_cophenetic", "_select_recommended_k",
        "save_symnmf", "load_symnmf",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    module = types.ModuleType("m165_numerical_fixture")
    module.__dict__.update({k: v for k, v in globals().items() if not k.startswith("__")})
    module.LOG = logging.getLogger("m165-test")
    module.NOISE_LABEL = -1
    module._HAS_SKLEARN = True
    module.NAME_NMF_SOFT_TPL = "symnmf_k{k}.tsv"
    module.NAME_NMF_ERR = "symnmf_errors.tsv"
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(SOURCE), "exec"), module.__dict__)
    return module


@unittest.skipUnless(HAVE_NUMERICAL_DEPS, "requires the existing M16.5 numerical image")
class NumericalCorrectness(unittest.TestCase):
    def setUp(self):
        self.core = load_core()

    @staticmethod
    def graph(n=4):
        g = ig.Graph(n=n, edges=[(0, 1)], directed=False)
        g.es["weight"] = [2.0]
        return g

    @staticmethod
    def pairs(rows=None):
        return pd.DataFrame(rows or [("a", "b", 1, 100, 1.0)],
                            columns=["sample_a", "sample_b", "n_segments",
                                     "total_shared_bp", "mean_jaccard"])

    def test_rb_quality_is_not_legacy_modularity(self):
        g = ig.Graph.Full(4)
        g.es["weight"] = [100, 1, 1, 1, 1, 100]
        part = la.RBConfigurationVertexPartition(
            g, initial_membership=[0, 0, 1, 1], weights="weight", resolution_parameter=.5)
        with patch.object(la, "find_partition", return_value=part):
            membership, quality = self.core._leiden_single(g, .5, 17)
        np.testing.assert_array_equal(membership, [0, 0, 1, 1])
        self.assertAlmostEqual(quality, 298.0)
        self.assertAlmostEqual(part.modularity, -1 / 6)
        self.assertAlmostEqual(g.modularity(membership, weights="weight"), .4803921568627451)

    def test_ranking_uses_rb_and_separates_descriptive_scores(self):
        g = ig.Graph.Full(4)
        g.es["weight"] = [100, 1, 1, 1, 1, 100]
        results = [(np.array([0, 0, 0, 0]), 204.), (np.array([0, 0, 1, 1]), 298.)]
        with patch.object(self.core, "_leiden_single", side_effect=results):
            assignments, scores, _, descriptor, _ = self.core.run_leiden_multiresolution(
                g, [.5], 2, 1, 42, .5)
        np.testing.assert_array_equal(assignments["community_res_0.5"], [0, 0, 1, 1])
        self.assertEqual(scores["is_representative"].tolist(), [False, True])
        self.assertEqual(scores["rb_quality"].tolist(), [204., 298.])
        self.assertGreater(scores.iloc[0].modularity, scores.iloc[1].modularity)
        self.assertAlmostEqual(descriptor, -1 / 6)

    def test_empty_graph_is_explicit_and_never_optimized(self):
        for n in (0, 4):
            with self.subTest(n=n), patch.object(la, "find_partition") as optimizer:
                a, scores, consensus, descriptor, memberships = self.core.run_leiden_multiresolution(
                    ig.Graph(n=n), [1.], 2, 1, 42, 1.)
                optimizer.assert_not_called()
                self.assertTrue((a.to_numpy() == -1).all())
                self.assertEqual(consensus.nnz, 0)
                self.assertTrue(np.isnan(descriptor))
                self.assertEqual(scores.status.tolist(), ["NO_GRAPH_SUPPORT"] * 2)
                self.assertEqual(scores.rb_quality.tolist(), [0., 0.])
                ari = self.core.compute_ari_multi_seed(memberships)
                self.assertEqual(int(ari.iloc[0].n_nodes), 0)
                self.assertTrue(np.isnan(ari.iloc[0].median_ari))

    def test_isolates_stay_unassigned_even_minimum_size_one(self):
        a, scores, consensus, _, memberships = self.core.run_leiden_multiresolution(
            self.graph(), [1.], 3, 1, 42, 1.)
        np.testing.assert_array_equal(a.iloc[2:, 0], [-1, -1])
        self.assertEqual(consensus[2:, :].nnz, 0)
        self.assertEqual(consensus[:, 2:].nnz, 0)
        self.assertEqual(scores.n_active_nodes.tolist(), [2] * 3)
        ari = self.core.compute_ari_multi_seed(memberships)
        self.assertEqual(int(ari.iloc[0].n_nodes), 2)
        self.assertAlmostEqual(ari.iloc[0].median_ari, 1.)
        confidence = self.core.compute_assignment_confidence(consensus, a.iloc[:, 0].to_numpy())
        self.assertTrue(np.isnan(confidence[2:]).all())

    def test_sparse_consensus_also_excludes_isolates(self):
        _, _, consensus, _, _ = self.core.run_leiden_multiresolution(
            self.graph(n=5001), [1.], 1, 1, 42, 1.)
        self.assertEqual(consensus.nnz, 4)
        self.assertEqual(consensus[2:, :].nnz, 0)

    def test_ari_ignores_noise_not_counted_as_one_stable_group(self):
        memberships = {1.: [np.array([0, 0, 1, -1, -1]), np.array([0, 1, 1, -1, -1])]}
        result = self.core.compute_ari_multi_seed(memberships).iloc[0]
        self.assertEqual(result.n_nodes, 3)
        self.assertAlmostEqual(result.median_ari, adjusted_rand_score([0, 0, 1], [0, 1, 1]))

    def test_invalid_leiden_arguments_fail_closed(self):
        for resolutions, seeds in (([], 1), ([1., 1.], 1), ([float("nan")], 1), ([1.], 0)):
            with self.subTest(resolutions=resolutions, seeds=seeds), self.assertRaises(ValueError):
                self.core.run_leiden_multiresolution(self.graph(), resolutions, seeds, 1, 42, 1.)
        for weight in (0., -1., float("nan")):
            g = self.graph(); g.es["weight"] = [weight]
            with self.assertRaises(ValueError):
                self.core.run_leiden_multiresolution(g, [1.], 1, 1, 42, 1.)

    def test_duplicate_and_reverse_pair_rows_are_rejected(self):
        for second in (("a", "b", 1, 100, 1.), ("b", "a", 1, 100, 1.)):
            df = self.pairs([("a", "b", 1, 100, 1.), second])
            with self.assertRaisesRegex(ValueError, "duplicate unordered"):
                self.core.aggregate_pair_weights(df, None, "log1p")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "pairs.tsv"; df.to_csv(path, sep="\t", index=False)
                with self.assertRaisesRegex(ValueError, "duplicate unordered"):
                    self.core.load_pair_summary(path)

    def test_self_pairs_missing_endpoints_and_invalid_values_rejected(self):
        for row in (("a", "a", 1, 10, 1.), (None, "b", 1, 10, 1.),
                    (" ", "b", 1, 10, 1.), ("a", "b", 1, -1, 1.),
                    ("a", "b", 1.5, 10, 1.), ("a", "b", 1, 10, float("inf"))):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.core.aggregate_pair_weights(self.pairs([row]), None, "raw")

    def test_unavailable_jaccard_is_not_fabricated_or_used_as_weight(self):
        df = self.pairs(); df["mean_jaccard"] = np.nan
        for mode, expected in (("raw", 100.), ("log1p", np.log1p(100))):
            weighted = self.core.aggregate_pair_weights(df, None, mode)
            self.assertAlmostEqual(weighted.weight.iloc[0], expected)
            self.assertTrue(weighted.mean_jaccard.isna().all())
        with self.assertRaisesRegex(ValueError, "requires finite observed"):
            self.core.aggregate_pair_weights(df, None, "mean_jaccard_weighted")

    def test_unknown_endpoint_rejected_even_below_threshold(self):
        df = self.core.aggregate_pair_weights(self.pairs(), None, "log1p")
        with self.assertRaisesRegex(ValueError, "outside"):
            self.core.build_sparse_matrix(df, ["a", "c"], 1000)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.core.build_sparse_matrix(df, ["a", "b", "b"], 0)

    def test_graph_builder_itself_rejects_duplicates(self):
        df = self.pairs([("a", "b", 1, 100, 1.), ("b", "a", 1, 100, 1.)])
        df["weight"] = np.log1p(100)
        with self.assertRaisesRegex(ValueError, "duplicate unordered"):
            self.core.build_sparse_matrix(df, ["a", "b"], 0)

    def test_filter_is_inclusive_and_and_not_segment_count(self):
        df = self.pairs([("b", "a", 1, 100, 1.), ("a", "c", 100, 99, 1.),
                         ("b", "c", 100, 101, 1.)])
        seg = df.copy(); seg["max_segment_bp"] = [50, 50, 49]
        weighted = self.core.aggregate_pair_weights(df, seg, "log1p", 50)
        S, kept = self.core.build_sparse_matrix(weighted, ["a", "b", "c", "d"], 100)
        self.assertEqual(kept.tolist(), [True, False])
        self.assertEqual(S.nnz, 2)
        self.assertAlmostEqual(S[0, 1], np.log1p(100))
        self.assertEqual(S[3, :].nnz, 0)

    def test_segment_summary_disagreement_rejected(self):
        for column, value in (("n_segments", 2), ("total_shared_bp", 101), ("mean_jaccard", .5), ("sample_b", "c")):
            seg = self.pairs(); seg.loc[0, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "summary and segment aggregate"):
                self.core.aggregate_pair_weights(self.pairs(), seg, "raw")

    def test_empty_pair_table_preserves_all_isolates(self):
        df = self.pairs().iloc[:0]
        weighted = self.core.aggregate_pair_weights(df, None, "log1p")
        S, kept = self.core.build_sparse_matrix(weighted, ["a", "b"], 0)
        self.assertEqual(S.shape, (2, 2)); self.assertEqual(S.nnz, 0)
        self.assertEqual(kept.size, 0)

    def test_absent_or_zero_inter_denominator_is_undefined(self):
        cases = [(self.pairs(), {"a": 0, "b": 0}),
                 (self.pairs(), {"a": 0, "b": 1}),
                 (self.pairs([("a", "b", 1, 100, 1.), ("a", "c", 1, 0, 1.)]),
                  {"a": 0, "b": 0, "c": 1})]
        for df, assignments in cases:
            labels = pd.DataFrame({"sample_id": list(assignments),
                                   "community_res_1": list(assignments.values())})
            result = self.core.validate_intra_vs_inter(df, labels, [1.]).iloc[0]
            self.assertTrue(np.isnan(result.ratio_median_intra_inter))

    def test_supported_ratio_keeps_exact_value(self):
        df = self.pairs([("a", "b", 1, 100, 1.), ("a", "c", 1, 20, 1.)])
        labels = pd.DataFrame({"sample_id": ["a", "b", "c"], "community_res_1": [0, 0, 1]})
        result = self.core.validate_intra_vs_inter(df, labels, [1.]).iloc[0]
        self.assertEqual(result.ratio_median_intra_inter, 5.)
        self.assertEqual((result.n_intra, result.n_inter), (1, 1))

    def nmf(self, matrix, **kwargs):
        options = dict(k_values=[2], n_inits=3, max_iter=20, tol=1e-5,
                       base_seed=42, laplacian=False, init_mode="random-cophenetic")
        options.update(kwargs)
        return self.core.run_symnmf_cophenetic(sp.csr_matrix(matrix), **options)

    def test_zero_mass_nmf_rows_are_unassigned(self):
        np.testing.assert_array_equal(self.core._symnmf_dominant_components(
            np.array([[0., 0.], [1., 1.], [0., 2.]])), [-1, 0, 1])

    def test_nmf_invariant_to_inserting_isolates(self):
        active = np.array([[0., 2., 1.], [2., 0., .5], [1., .5, 0.]])
        with_isolates = np.zeros((5, 5)); indices = [0, 2, 4]
        with_isolates[np.ix_(indices, indices)] = active
        for laplacian in (False, True):
            H, errors, stats = self.nmf(active, laplacian=laplacian)
            full_H, full_errors, full_stats = self.nmf(with_isolates, laplacian=laplacian)
            np.testing.assert_array_equal(H[2], full_H[2][indices])
            np.testing.assert_array_equal(full_H[2][[1, 3]], np.zeros((2, 2)))
            pd.testing.assert_frame_equal(errors, full_errors)
            for column in ("cophenetic_correlation", "dispersion_index"):
                np.testing.assert_allclose(stats[column], full_stats[column], equal_nan=True)
            self.assertEqual(full_stats.n_supported.tolist(), [3])
            self.assertEqual(full_stats.n_isolated.tolist(), [2])

    def test_nmf_empty_support_no_fit_no_recommendation(self):
        for n in (0, 4):
            with patch.object(self.core, "symnmf") as fitter:
                H, errors, stats = self.nmf(np.zeros((n, n)), k_operational=2)
            fitter.assert_not_called()
            self.assertEqual(H[2].shape, (n, 2)); self.assertFalse(H[2].any())
            self.assertTrue(errors.empty)
            self.assertEqual(stats.status.tolist(), ["NO_GRAPH_SUPPORT"])
            self.assertFalse(stats.is_recommended_k.any())
            self.assertTrue(stats.cophenetic_correlation.isna().all())

    def test_two_supported_nmf_nodes_not_four_for_diagnostics(self):
        matrix = np.zeros((4, 4)); matrix[0, 1] = matrix[1, 0] = 2.
        _, _, stats = self.nmf(matrix)
        self.assertEqual(stats.n_supported.tolist(), [2])
        self.assertTrue(stats.cophenetic_correlation.isna().all())
        self.assertFalse(stats.is_recommended_k.any())

    def test_nmf_excessive_k_and_invalid_matrix_rejected(self):
        with self.assertRaisesRegex(ValueError, "supported nodes"):
            self.nmf([[0., 1.], [1., 0.]], k_values=[3])
        for matrix in ([[0., -1.], [-1., 0.]], [[0., 1.], [0., 0.]], [[float("nan")]]):
            with self.assertRaises(ValueError):
                self.nmf(matrix)

    def test_supported_zero_loading_fit_is_rejected(self):
        with patch.object(self.core, "symnmf", return_value=(np.zeros((2, 2)), [1.])):
            with self.assertRaisesRegex(ValueError, "zero loadings"):
                self.nmf([[0., 1.], [1., 0.]])

    def test_nmf_saved_zero_rows_keep_sample_order_and_k_columns(self):
        matrix = np.zeros((4, 4)); matrix[0, 2] = matrix[2, 0] = 2.
        H, errors, _ = self.nmf(matrix)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.core.save_symnmf(output, ["a", "b", "c", "d"], H, errors)
            loaded = self.core.load_symnmf(output, 2)
        self.assertEqual(loaded.columns.tolist(), ["sample_id", "component_1", "component_2"])
        self.assertEqual(loaded.sample_id.tolist(), ["a", "b", "c", "d"])
        np.testing.assert_array_equal(loaded.iloc[[1, 3], 1:].to_numpy(), np.zeros((2, 2)))

    def test_leiden_actual_runs_are_deterministic_and_rank_within_gamma(self):
        g = ig.Graph.Full(4); g.es["weight"] = [100, 1, 1, 1, 1, 100]
        first = self.core.run_leiden_multiresolution(g, [.5, 1., 2.], 3, 1, 42, 1.)
        second = self.core.run_leiden_multiresolution(g, [.5, 1., 2.], 3, 1, 42, 1.)
        pd.testing.assert_frame_equal(first[0], second[0])
        for resolution, rows in first[1].groupby("resolution"):
            selected = rows[rows.is_representative]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected.rb_quality.iloc[0], rows.rb_quality.max())
            membership = first[0][f"community_res_{resolution:g}"].tolist()
            partition = la.RBConfigurationVertexPartition(
                g, initial_membership=membership, weights="weight", resolution_parameter=resolution)
            self.assertAlmostEqual(selected.rb_quality.iloc[0], partition.quality())


if __name__ == "__main__":
    unittest.main()
