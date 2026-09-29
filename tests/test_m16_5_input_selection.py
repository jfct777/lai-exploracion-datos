"""Selección de una única fuente de estadísticas, sin datos de personas reales."""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_m16_5_numerical_correctness as numerical


def load_selection_core():
    core = numerical.load_core()
    tree = ast.parse(numerical.SOURCE.read_text())
    names = {"load_graph_pair_summary", "do_build_graph"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *functions,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(numerical.SOURCE), "exec"), core.__dict__)
    # Exercise the actual cache branch without invoking the full CLI or plots.
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    ensure = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "_ensure_graph")
    wrapper = ast.parse("""
def run_cached_graph(args, inputs, out_dir):
    S = g = samples = pair_summary = metadata_values = metadata_name = None
    metadata_warnings = []
""").body[0]
    wrapper.body.extend([ensure, *ast.parse("_ensure_graph()\nreturn pair_summary").body])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 str(numerical.SOURCE), "exec"), core.__dict__)
    return core


@unittest.skipUnless(numerical.HAVE_NUMERICAL_DEPS, "requires the existing M16.5 image")
class InputSelection(unittest.TestCase):
    def setUp(self):
        self.core = load_selection_core()
        self.inputs = SimpleNamespace(
            pair_summary=Path("pairs.tsv"), segments=Path("segments.tsv.gz"),
            individual_summary=Path("individuals.tsv"),
        )

    @staticmethod
    def args(transform="log1p", maximum=0):
        return SimpleNamespace(edge_weight_transform=transform,
                               min_max_segment_bp=maximum,
                               segments_chunk_rows=2, min_edge_bp=0)

    def summary(self):
        return numerical.pd.DataFrame({
            "sample_a": ["a", "a"], "sample_b": ["b", "c"],
            "n_segments": [2, 1], "total_shared_bp": [150, 25],
            "mean_jaccard": [.5, .25], "n_shared_variants_total": [7, 2],
            "max_segment_bp": [100, 25],
        })

    def test_only_the_required_source_is_opened(self):
        for transform in ("raw", "log1p", "n_shared_variants", "mean_jaccard_weighted"):
            for maximum in (0, 50):
                with self.subTest(transform=transform, maximum=maximum):
                    streaming = transform == "n_shared_variants" or maximum > 0
                    selected = "load_segments_aggregated" if streaming else "load_pair_summary"
                    omitted = "load_pair_summary" if streaming else "load_segments_aggregated"
                    with patch.object(self.core, selected, return_value=self.summary()) as reader, \
                         patch.object(self.core, omitted, side_effect=AssertionError("redundant read")):
                        actual = self.core.load_graph_pair_summary(self.args(transform, maximum), self.inputs)
                    reader.assert_called_once()
                    self.assertEqual(reader.call_args.kwargs["include_jaccard"],
                                     transform == "mean_jaccard_weighted")
                    numerical.pd.testing.assert_frame_equal(actual, self.summary())

    def test_streamed_summary_still_rejects_invalid_pairs(self):
        bad = self.summary()
        bad.loc[0, "sample_b"] = "a"
        with patch.object(self.core, "load_segments_aggregated", return_value=bad), \
             patch.object(self.core, "load_pair_summary", side_effect=AssertionError("redundant read")), \
             self.assertRaises(ValueError):
            self.core.load_graph_pair_summary(self.args(maximum=50), self.inputs)

    def test_graph_matches_legacy_aggregation_and_returns_unfiltered_summary(self):
        full = self.summary()
        samples = ["a", "b", "c", "isolated"]
        legacy = self.core.aggregate_pair_weights(full, full, "log1p", min_max_segment_bp=50)
        expected, _ = self.core.build_sparse_matrix(legacy, samples, min_edge_bp=0)
        with patch.object(self.core, "load_segments_aggregated", return_value=full), \
             patch.object(self.core, "load_pair_summary", side_effect=AssertionError("redundant read")), \
             patch.object(self.core, "load_individuals", return_value=samples), \
             patch.object(self.core, "save_graph", create=True) as save:
            matrix, graph, order, returned = self.core.do_build_graph(
                self.args(maximum=50), self.inputs, Path("unused"))
        numerical.np.testing.assert_array_equal(matrix.toarray(), expected.toarray())
        self.assertEqual(order, samples)
        self.assertEqual(graph.vcount(), 4)
        self.assertEqual(graph.ecount(), 1)
        self.assertEqual(len(save.call_args.kwargs["pair_df_with_weight"]), 1)
        numerical.pd.testing.assert_frame_equal(returned, full)
        self.assertEqual(len(returned), 2)  # Validation receives both pairs, not just the graph edge.

    def test_real_segment_file_does_not_require_opening_pair_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = SimpleNamespace(segments=Path(tmp) / "segments.tsv.gz",
                                     pair_summary=Path(tmp) / "not_opened.tsv")
            numerical.pd.DataFrame({
                "sample_a": ["a", "a", "a"], "sample_b": ["b", "b", "c"],
                "length_bp": [100, 50, 25], "n_shared_variants": [4, 3, 2],
                "jaccard": [.5, .5, .25],
            }).to_csv(inputs.segments, sep="\t", index=False)
            actual = self.core.load_graph_pair_summary(self.args(maximum=50), inputs)
            expected = self.summary().loc[:, actual.columns].copy()
            expected["mean_jaccard"] = numerical.np.nan
            numerical.pd.testing.assert_frame_equal(actual, expected)

    def test_unused_jaccard_is_not_parsed_or_required_from_either_source(self):
        for transform in ("raw", "log1p", "n_shared_variants"):
            for maximum in (0, 50):
                for include_column in (False, True):
                    with self.subTest(transform=transform, maximum=maximum,
                                      include_column=include_column), tempfile.TemporaryDirectory() as tmp:
                        inputs = SimpleNamespace(segments=Path(tmp) / "segments.tsv.gz",
                                                 pair_summary=Path(tmp) / "pairs.tsv")
                        segments = numerical.pd.DataFrame({
                            "sample_a": ["a", "a", "a"], "sample_b": ["b", "b", "c"],
                            "length_bp": [100, 50, 25], "n_shared_variants": [4, 3, 2],
                        })
                        pairs = self.summary().drop(columns="mean_jaccard")
                        if include_column:
                            # Non-numeric sentinel proves the unused column is not parsed.
                            segments["jaccard"] = "unused-not-a-number"
                            pairs["mean_jaccard"] = "unused-not-a-number"
                        segments.to_csv(inputs.segments, sep="\t", index=False)
                        pairs.to_csv(inputs.pair_summary, sep="\t", index=False)
                        actual = self.core.load_graph_pair_summary(self.args(transform, maximum), inputs)
                        self.assertTrue(actual.mean_jaccard.isna().all())
                        weights = self.core.aggregate_pair_weights(actual, None, transform, maximum)
                        legacy = self.core.aggregate_pair_weights(self.summary(), None, transform, maximum)
                        numerical.np.testing.assert_array_equal(weights.weight, legacy.weight)
                        self.assertEqual(list(weights.columns), list(legacy.columns))
                        samples = ["a", "b", "c", "isolated"]
                        matrix, _ = self.core.build_sparse_matrix(weights, samples, 0)
                        expected, _ = self.core.build_sparse_matrix(legacy, samples, 0)
                        numerical.np.testing.assert_array_equal(matrix.toarray(), expected.toarray())

    def test_explicit_jaccard_preserves_values_across_chunks_and_requires_observations(self):
        for maximum in (0, 50):
            with self.subTest(maximum=maximum), tempfile.TemporaryDirectory() as tmp:
                inputs = SimpleNamespace(segments=Path(tmp) / "segments.tsv.gz",
                                         pair_summary=Path(tmp) / "pairs.tsv")
                segments = numerical.pd.DataFrame({
                    "sample_a": ["a", "a", "a"], "sample_b": ["b", "c", "b"],
                    "length_bp": [100, 25, 50], "n_shared_variants": [4, 2, 3],
                    "jaccard": [.25, .25, .75],
                })
                segments.to_csv(inputs.segments, sep="\t", index=False)
                self.summary().to_csv(inputs.pair_summary, sep="\t", index=False)
                args = self.args("mean_jaccard_weighted", maximum)
                actual = self.core.load_graph_pair_summary(args, inputs)
                numerical.np.testing.assert_array_equal(actual.mean_jaccard, [.5, .25])
                weights = self.core.aggregate_pair_weights(actual, None, args.edge_weight_transform, maximum)
                numerical.np.testing.assert_array_equal(weights.weight, [1.] if maximum else [1., .25])
                if maximum:
                    segments.loc[0, "jaccard"] = numerical.np.nan
                    segments.to_csv(inputs.segments, sep="\t", index=False)
                else:
                    bad = self.summary()
                    bad["mean_jaccard"] = numerical.np.nan
                    bad.to_csv(inputs.pair_summary, sep="\t", index=False)
                with self.assertRaises(ValueError):
                    actual = self.core.load_graph_pair_summary(args, inputs)
                    self.core.aggregate_pair_weights(actual, None, args.edge_weight_transform, maximum)
                if maximum:
                    segments.drop(columns="jaccard").to_csv(inputs.segments, sep="\t", index=False)
                else:
                    self.summary().drop(columns="mean_jaccard").to_csv(inputs.pair_summary, sep="\t", index=False)
                with self.assertRaises((ValueError, SystemExit)):
                    self.core.load_graph_pair_summary(args, inputs)

    def test_empty_segments_preserve_schema_with_or_without_jaccard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "segments.tsv.gz"
            numerical.pd.DataFrame(columns=["sample_a", "sample_b", "length_bp",
                                            "n_shared_variants", "jaccard"]).to_csv(path, sep="\t", index=False)
            for include_jaccard in (False, True):
                result = self.core.load_segments_aggregated(path, 2, include_jaccard=include_jaccard)
                self.assertTrue(result.empty)
                self.assertIn("mean_jaccard", result.columns)

    def test_all_driver_segment_reads_choose_jaccard_explicitly(self):
        tree = ast.parse(numerical.SOURCE.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "load_segments_aggregated"]
        self.assertEqual(len(calls), 2)  # Graph statistics and validation diagnostics.
        for call in calls:
            self.assertIn("include_jaccard", {keyword.arg for keyword in call.keywords})

    def test_unused_jaccard_is_excluded_by_parser_column_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.tsv"
            self.summary().to_csv(path, sep="\t", index=False)
            with patch.object(self.core.pd, "read_csv", wraps=numerical.pd.read_csv) as reader:
                self.core.load_pair_summary(path, include_jaccard=False)
            self.assertFalse(reader.call_args.kwargs["usecols"]("mean_jaccard"))
            self.assertTrue(reader.call_args.kwargs["usecols"]("total_shared_bp"))

    def test_cached_diagnostics_use_same_source_and_clustering_skips_statistics(self):
        for mode in ("validate", "plot", "all", "leiden", "nmf", "report"):
            for maximum in (0, 50):
                with self.subTest(mode=mode, maximum=maximum), tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp)
                    (out / "cached_graph.npz").touch()
                    args = self.args(maximum=maximum)
                    args.mode, args.metadata_file, args.plot_color_by = mode, None, None
                    full = self.summary()
                    selected = "load_segments_aggregated" if maximum else "load_pair_summary"
                    omitted = "load_pair_summary" if maximum else "load_segments_aggregated"
                    with patch.object(self.core, "NAME_GRAPH_MATRIX", "cached_graph.npz", create=True), \
                         patch.object(self.core, "load_graph", return_value=(object(), object(), ["a", "b"]), create=True), \
                         patch.object(self.core, "load_metadata_safe", return_value=(None, None, []), create=True), \
                         patch.object(self.core, selected, return_value=full) as reader, \
                         patch.object(self.core, omitted, side_effect=AssertionError("redundant read")):
                        actual = self.core.run_cached_graph(args, self.inputs, out)
                    if mode in ("validate", "plot", "all"):
                        numerical.pd.testing.assert_frame_equal(actual, full)
                        reader.assert_called_once()
                    else:
                        self.assertIsNone(actual)
                        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
