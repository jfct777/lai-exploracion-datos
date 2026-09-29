"""Synthetic inline legend checks. No projections, genotypes or clustering."""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'bin/ibd_community_enhanced.py'
FROZEN_STAGE11_STATIC_AST = '1a17a20e18ad2036766318a762cd41536b3ad771956229147013e9fab6a2957f'
sys.path.insert(0, str(ROOT / 'bin'))

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import ibd_community_enhanced as core
    HAS_STACK = True
except ImportError:
    HAS_STACK = False


class DefaultInlinePath(ast.NodeTransformer):
    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == 'inline_metadata_legend':
            return [self.visit(statement) for statement in node.orelse]
        return self.generic_visit(node)


def legacy_ast():
    node = copy.deepcopy(next(node for node in ast.parse(CORE.read_text()).body
                              if isinstance(node, ast.FunctionDef) and node.name == '_plot_network_static'))
    retained = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if arg.arg not in {'inline_metadata_legend', 'legend_wrap_chars'}]
    node.args.kwonlyargs = [arg for arg, _ in retained]
    node.args.kw_defaults = [default for _, default in retained]
    return DefaultInlinePath().visit(node)


class DefaultContractTests(unittest.TestCase):
    def test_default_off_is_ast_identical_to_frozen_stage11(self):
        node = legacy_ast()
        digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest, FROZEN_STAGE11_STATIC_AST)
        current = next(node for node in ast.parse(CORE.read_text()).body
                       if isinstance(node, ast.FunctionDef) and node.name == '_plot_network_static')
        defaults = dict(zip((arg.arg for arg in current.args.kwonlyargs), current.args.kw_defaults))
        self.assertIs(defaults['inline_metadata_legend'].value, False)
        self.assertEqual(defaults['legend_wrap_chars'].value, 56)


@unittest.skipUnless(HAS_STACK, 'requires existing pinned M16.5 image')
class InlineLegendTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.addCleanup(plt.close, 'all')

    def kwargs(self, membership, metadata, indices=None):
        membership = np.asarray(membership, dtype=int)
        indices = np.arange(len(membership)) if indices is None else np.asarray(indices)
        coords = np.column_stack((np.cos(indices), np.sin(indices)))
        return dict(coords=coords, sub_idx=indices, sub_memb=membership[indices],
            sub_wdeg=np.arange(len(indices), dtype=float), membership=membership,
            out_path=self.base/'synthetic.png', confidence=np.linspace(0, 1, len(membership)),
            metadata_values=None if metadata is None else np.asarray(metadata, dtype=object),
            metadata_name=None if metadata is None else 'finestructure_clusters', dpi=60,
            width_in=8., height_in=6., export_pdf=False, export_svg=False, label_min_size=10,
            community_annotations=None, adjust_labels=False,
            figure_title='Resultado descriptivo sintético', figure_note='Sin inferencia biológica.')

    def figure(self, kwargs, **optional):
        with patch.object(core, '_save_fig') as save:
            core._plot_network_static(**kwargs, **optional)
        self.assertEqual(save.call_count, 1)
        return save.call_args.args[0]

    def test_full_counts_mixed_missing_and_noise_do_not_equal_missing_metadata(self):
        membership = np.array([0, 0, 0, 0, 1, 1, -1, -1, -1])
        metadata = np.array(['BrazilB', 'BrazilA', 'BrazilA', None,
                             'BrazilX', 'BrazilX', 'BrazilB', np.nan, 'BrazilA'], dtype=object)
        before = membership.copy(); metadata_before = metadata.copy()
        result = core._inline_community_metadata_labels(membership, metadata)
        self.assertTrue(result['abbreviated'])
        self.assertEqual(result['rows'][0]['counts'], [('BrazilA', 2), (None, 1), ('BrazilB', 1)])
        self.assertIn('C0 (n=4)', result['rows'][0]['label'])
        self.assertIn('A: 2', result['rows'][0]['label'])
        self.assertIn('sin dato: 1', result['rows'][0]['label'])
        self.assertEqual(result['rows'][-1]['n'], 3)
        self.assertIn('Noise (n=3)', result['rows'][-1]['label'])
        self.assertIn('A = BrazilA', result['header'])
        self.assertIn('X = BrazilX', result['header'])
        for row in result['rows'].values():
            self.assertEqual(sum(count for _, count in row['counts']), row['n'])
        np.testing.assert_array_equal(membership, before)
        self.assertEqual(metadata[0], metadata_before[0])
        self.assertIsNone(metadata[3])
        self.assertTrue(np.isnan(metadata[7]))

    def test_arbitrary_category_prevents_ambiguous_abbreviation_and_preserves_all_names(self):
        result = core._inline_community_metadata_labels(np.array([0, 0, 0]),
            np.array(['BrazilA', 'A', 'A different long category'], dtype=object), 30)
        self.assertFalse(result['abbreviated'])
        self.assertEqual(result['codes']['BrazilA'], 'BrazilA')
        label = result['rows'][0]['label']
        self.assertIn('BrazilA: 1', label)
        self.assertIn('A: 1', label)
        self.assertIn('A different long category: 1', ' '.join(label.split()))
        self.assertTrue(all(len(line) <= 30 for line in label.splitlines()[1:]))

    def test_full_cohort_counts_and_groups_retained_in_subsample_view(self):
        kwargs = self.kwargs([0, 0, 1, 1, -1], ['BrazilA'] * 5, [0, 4])
        fig = self.figure(kwargs, inline_metadata_legend=True)
        labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
        self.assertTrue(any('C1 (n=2)' in text for text in labels))
        self.assertTrue(any('C0 (n=2)' in text for text in labels))
        self.assertTrue(any('Noise (n=1)' in text for text in labels))

    def test_all_65_groups_and_noise_visible_readably_without_overlap(self):
        membership = np.concatenate((np.repeat(np.arange(65), 3), [-1, -1, -1]))
        metadata = np.tile(np.array(['BrazilA', 'BrazilB', 'BrazilX']), 66)
        fig = self.figure(self.kwargs(membership, metadata), inline_metadata_legend=True)
        self.assertEqual(len(fig.axes), 2)
        legends = [ax.get_legend() for ax in fig.axes]
        texts = legends[0].get_texts()
        self.assertEqual(len(texts), 66)
        self.assertTrue(all(text.get_fontsize() >= 7 for text in texts))
        self.assertTrue(any(text.get_text().startswith('C64 (n=3)') for text in texts))
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer) for text in texts]
        self.assertTrue(all(not first.overlaps(second) for first, second in zip(boxes, boxes[1:])))
        for legend in legends:
            box = legend.get_window_extent(renderer)
            self.assertGreaterEqual(box.x0, 0); self.assertGreaterEqual(box.y0, 0)
            self.assertLessEqual(box.x1, fig.bbox.width); self.assertLessEqual(box.y1, fig.bbox.height)
        self.assertFalse(legends[0].get_window_extent(renderer).overlaps(fig.axes[1].get_window_extent(renderer)))
        self.assertFalse(legends[0].get_window_extent(renderer).overlaps(fig.axes[0].get_window_extent(renderer)))
        np.testing.assert_allclose(fig.axes[0].get_position().size, fig.axes[1].get_position().size)
        self.assertTrue(any('mezcla, no equivalencia' in text.get_text() for text in fig.texts))

    def test_scatter_coordinates_colours_sizes_order_alpha_and_limits_unchanged(self):
        kwargs = self.kwargs([0, 0, 1, 1, -1, -1], ['BrazilA', 'BrazilB', 'BrazilX', None, 'BrazilA', None])
        before = {key: value.copy() for key, value in kwargs.items() if isinstance(value, np.ndarray)}
        old = self.figure(kwargs)
        new = self.figure(kwargs, inline_metadata_legend=True)
        for old_ax, new_ax in zip(old.axes, new.axes):
            self.assertEqual(old_ax.get_xlim(), new_ax.get_xlim())
            self.assertEqual(old_ax.get_ylim(), new_ax.get_ylim())
            self.assertEqual(len(old_ax.collections), len(new_ax.collections))
            for old_collection, new_collection in zip(old_ax.collections, new_ax.collections):
                for attr in ('get_offsets', 'get_sizes', 'get_facecolors', 'get_edgecolors', 'get_linewidths'):
                    np.testing.assert_array_equal(getattr(old_collection, attr)(), getattr(new_collection, attr)())
                self.assertEqual(old_collection.get_alpha(), new_collection.get_alpha())
                self.assertEqual(old_collection.get_zorder(), new_collection.get_zorder())
        for key, value in before.items():
            np.testing.assert_array_equal(kwargs[key], value)
        right_labels = [text.get_text() for text in new.axes[1].get_legend().get_texts()]
        self.assertIn('None (2)', right_labels)

    def test_default_off_png_is_byte_identical_to_frozen_path_with_and_without_metadata(self):
        node = legacy_ast()
        namespace = dict(vars(core))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<frozen-default-path>', 'exec'), namespace)
        legacy = namespace['_plot_network_static']
        for metadata in (None, ['BrazilA', 'BrazilB', 'BrazilX', 'BrazilA']):
            with self.subTest(metadata=metadata is not None):
                kwargs = self.kwargs([0, 0, 1, -1], metadata)
                legacy(**kwargs)
                expected = kwargs['out_path'].read_bytes()
                kwargs['out_path'] = self.base/'new.png'
                core._plot_network_static(**kwargs, inline_metadata_legend=False, legend_wrap_chars=56)
                self.assertEqual(kwargs['out_path'].read_bytes(), expected)

    def test_invalid_alignment_missing_metadata_and_wrap_width_fail_closed(self):
        for width in (0, 19, 121, True, '56'):
            with self.assertRaises(ValueError):
                core._inline_community_metadata_labels(np.array([0]), np.array(['BrazilA']), width)
        with self.assertRaises(ValueError):
            core._inline_community_metadata_labels(np.array([0, 1]), np.array(['BrazilA']))
        with self.assertRaises(ValueError):
            self.figure(self.kwargs([0, 1], None), inline_metadata_legend=True)
        kwargs = self.kwargs([0, 1], ['BrazilA', 'BrazilB']); kwargs['sub_memb'] = np.array([1, 0])
        with self.assertRaises(ValueError):
            self.figure(kwargs, inline_metadata_legend=True)


if __name__ == '__main__':
    unittest.main()
