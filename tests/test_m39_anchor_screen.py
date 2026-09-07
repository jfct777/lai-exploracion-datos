"""Synthetic unit checks for development-only screening; no biological training."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import m39_anchor_screen as screen
from m33_safe_bridge_core import write_deterministic_npz


def development_fixture() -> tuple[dict, dict]:
    n, j, k, f = 4, 5, 2, 4
    keys = np.asarray([b'c', b'a', b'd', b'b'], dtype='S32')
    shape = (n, j, 2, 3, k)
    features = {
        'sample_key_sha256': keys,
        'pos': np.arange(j) * 100 + 1,
        'anchor_indices': np.arange(j) * 2,
        'locus_id': np.asarray([f'22:{100*i+1}:A:C' for i in range(j)]),
        'common_context': np.arange(np.prod((*shape, f)), dtype=np.float32).reshape(*shape, f),
        'candidate_mask': np.ones(shape, dtype=bool),
        'ref_dosage': np.zeros(shape, dtype=np.float32),
        'rare_ref_observed': np.ones(shape, dtype=bool),
        'query_dosage': np.zeros((n, j), dtype=np.float32),
        'query_observed': np.ones((n, j), dtype=bool),
        'pooled_summary': np.broadcast_to(np.asarray([1., 0., 0., 1.]), (j, 3, 4)).copy(),
    }
    data = {name: features[name].copy() for name in ('pos', 'anchor_indices', 'locus_id')}
    data.update(sample_key_sha256=keys[[1, 3, 0, 2]],
                truth_state=np.arange(n*j, dtype=np.int64).reshape(n, j) % 6,
                baseline=np.full((n, j, 6), 1/6, dtype=np.float32),
                coords=np.arange(j, dtype=np.float64) * .01,
                train_indices=np.asarray([0, 2]), select_indices=np.asarray([1, 3]))
    return features, data


class AnchorScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exhaustive_ragged_epochs_same_exposure_and_ordered_coordinates(self):
        people = np.asarray([8, 2, 7, 1, 5])
        original = people.copy()
        expected = Counter((int(p), j) for p in people for j in range(11))
        for seed in (0, 17):
            emitted = list(screen.batches(people, 11, 2, 4, np.random.default_rng(seed)))
            self.assertEqual(Counter((int(p), int(j)) for rows, cols in emitted
                                     for p in rows for j in cols), expected)
            for rows, cols in emitted:
                self.assertLessEqual(len(rows), 2)
                self.assertLessEqual(len(cols), 4)
                self.assertTrue(np.all(np.diff(cols) > 0))
            repeated = list(screen.batches(people, 11, 2, 4, np.random.default_rng(seed)))
            for first, second in zip(emitted, repeated):
                for a, b in zip(first, second):
                    np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(people, original)

    def test_normalization_and_prior_are_train_only(self):
        features, data = development_fixture()
        train = data['train_indices']
        context, mask = features['common_context'], features['candidate_mask']
        mean, scale = screen.normalizer(context, mask, train)
        changed = context.copy()
        changed[data['select_indices']] = np.nan
        changed_mean, changed_scale = screen.normalizer(changed, mask, train)
        np.testing.assert_array_equal(changed_mean, mean)
        np.testing.assert_array_equal(changed_scale, scale)
        values = context[train].reshape(-1, 4).astype(np.float64)
        np.testing.assert_allclose(mean, values.mean(0))
        np.testing.assert_allclose(scale, values.std(0))
        labels = data['truth_state'].copy()
        expected = np.bincount(labels[train].ravel(), minlength=6) + 1
        prior = screen.ancestry_prior(labels, train)
        np.testing.assert_allclose(prior, expected / expected.sum())
        labels[data['select_indices']] = 5
        np.testing.assert_array_equal(screen.ancestry_prior(labels, train), prior)
        self.assertTrue(np.all(screen.ancestry_prior(np.zeros_like(labels), train) > 0))
        for alpha in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                screen.ancestry_prior(labels, train, alpha)

    def test_masked_train_values_ignored_and_no_support_rejected(self):
        context = np.asarray([[[1., 4.], [np.nan, np.nan]]])
        mask = np.asarray([[True, False]])
        mean, scale = screen.normalizer(context, mask, np.asarray([0]))
        np.testing.assert_array_equal(mean, [1., 4.])
        np.testing.assert_array_equal(scale, [1e-6, 1e-6])
        with self.assertRaises(ValueError):
            screen.normalizer(context, np.zeros_like(mask), np.asarray([0]))

    def test_bound_sample_order_and_global_pooled_broadcast(self):
        features, data = development_fixture()
        with tempfile.TemporaryDirectory() as directory:
            fp, dp = Path(directory)/'features.npz', Path(directory)/'development.npz'
            np.savez(fp, **features)
            np.savez(dp, **data)
            batch, labels, _, train, select, mean, scale = screen.load_inputs(fp, dp)
        order = np.asarray([1, 3, 0, 2])
        expected = features['common_context'][order]
        expected_mean, expected_scale = screen.normalizer(expected, features['candidate_mask'][order], train)
        np.testing.assert_array_equal(mean, expected_mean)
        np.testing.assert_array_equal(scale, expected_scale)
        np.testing.assert_allclose(batch['common_context'].numpy(), (expected-mean)/scale, atol=1e-6)
        np.testing.assert_array_equal(labels.numpy(), data['truth_state'])
        np.testing.assert_array_equal(select, data['select_indices'])
        np.testing.assert_array_equal(batch['pooled_summary'].numpy(),
                                      np.broadcast_to(features['pooled_summary'], (4, 5, 3, 4)))

    def test_score_roles_forbidden_even_if_empty(self):
        _, data = development_fixture()
        for score in (np.asarray([], dtype=np.int64), np.asarray([3])):
            with self.assertRaisesRegex(ValueError, 'SCORE'):
                screen.validate_partition({**data, 'score_indices': score})

    def test_fractional_truth_cannot_be_silently_truncated_to_six_states(self):
        _, data = development_fixture()
        fractional = data['truth_state'].astype(np.float64)
        fractional[0, 0] = .5
        with self.assertRaises(ValueError):
            screen.validate_partition({**data, 'truth_state': fractional})

    def test_roles_exhaustive_disjoint_integer_and_keys_unique(self):
        _, data = development_fixture()
        for train, select in (([0, 1], [1, 3]), ([0], [1, 2]), ([0, 4], [1, 2]), ([], [0, 1, 2, 3])):
            with self.assertRaises(ValueError):
                screen.validate_partition({**data, 'train_indices': np.asarray(train),
                                            'select_indices': np.asarray(select)})
        with self.assertRaises(ValueError):
            screen.validate_partition({**data, 'train_indices': np.asarray([0., 2.])})
        for source, requested in (([b'a', b'a'], [b'a']), ([b'a', b'b'], [b'a', b'a']),
                                  ([b'a', b'b'], [b'c'])):
            with self.assertRaises(ValueError):
                screen.match_samples(np.asarray(source), np.asarray(requested))

    def test_prediction_stitches_nonmonotonic_people_and_ragged_loci(self):
        class EchoBaseline:
            def eval(self):
                return self
            def __call__(self, batch, arm):
                assert arm == 'carrier'
                assert bool((batch['coords'][1:] >= batch['coords'][:-1]).all())
                return batch['baseline']
        probabilities = torch.arange(1, 4*7*6+1, dtype=torch.float32).reshape(4, 7, 6)
        probabilities /= probabilities.sum(-1, keepdim=True)
        batch = {'baseline': probabilities, 'coords': torch.arange(7).double()}
        people = np.asarray([3, 0, 2])
        actual = screen.predict(EchoBaseline(), batch, people, 'carrier', 2, 3)
        np.testing.assert_array_equal(actual, probabilities[people].numpy())

    def test_loss_brier_and_diploid_dosage_formulas(self):
        p = np.asarray([[[.5, .25, .25, 0, 0, 0], [0, 0, 0, 0, 0, 1]],
                        [[1/6]*6, [.1, .2, .1, .2, .3, .1]]])
        labels = np.asarray([[1, 0], [2, 4]])
        original = p.copy()
        actual = screen.metrics(p, labels)
        ll = -np.log(np.maximum(np.asarray([[.25, 0], [1/6, .3]]), 1e-12))
        brier = np.square(p-np.eye(6)[labels]).sum(-1)
        dosage = np.abs(p @ screen.STATE_DOSAGES - screen.STATE_DOSAGES[labels]).mean((0, 1))
        self.assertAlmostEqual(actual['log_loss'], ll.mean())
        self.assertAlmostEqual(actual['log_loss'], np.mean(actual['log_loss_per_person']))
        self.assertAlmostEqual(actual['brier'], brier.mean())
        np.testing.assert_allclose(actual['dosage_mae'], dosage)
        np.testing.assert_allclose(actual['log_loss_per_person'], ll.mean(1))
        self.assertEqual(actual['fraction_true_probability_below_floor'], .25)
        self.assertEqual(actual['weighting'], 'equal_people_equal_selected_anchors')
        np.testing.assert_array_equal(p, original)

    def test_metric_range_simplex_and_roundoff(self):
        p = np.zeros((1, 1, 6), dtype=np.float32)
        p[..., 0] = 1 + 1e-7
        self.assertEqual(screen.metrics(p, np.zeros((1, 1), dtype=int))['log_loss'], 0.)
        for invalid in (np.full((1, 1, 6), np.nan), np.full((1, 1, 6), .5),
                        np.asarray([[[-1, 2, 0, 0, 0, 0]]])):
            with self.assertRaises(ValueError):
                screen.metrics(invalid, np.zeros((1, 1), dtype=int))

    def test_metrics_are_exactly_independent_of_memory_layout(self):
        # A one-person fixture cannot exercise this: its label array can be
        # both C and F contiguous. Use the actual development dimensions, but
        # entirely synthetic probabilities and labels (no genomic records).
        rng = np.random.default_rng(20260907)
        p = rng.random((64, 660, 6), dtype=np.float32)
        p /= p.sum(-1, keepdims=True)
        truth = rng.integers(0, 6, size=(64, 660), dtype=np.uint8)
        original_p, original_truth = p.copy(), truth.copy()
        p_storage = np.empty((64, 1320, 6), dtype=p.dtype)
        truth_storage = np.empty((64, 1320), dtype=truth.dtype)
        p_storage[:, ::2], truth_storage[:, ::2] = p, truth
        probabilities = (p, np.asfortranarray(p), p_storage[:, ::2])
        labels = (truth, np.asfortranarray(truth), truth_storage[:, ::2],
                  truth[:, np.arange(660)])
        expected = screen.metrics(p, truth)
        for p_view in probabilities:
            for truth_view in labels:
                with self.subTest(p_strides=p_view.strides, truth_strides=truth_view.strides):
                    self.assertEqual(screen.metrics(p_view, truth_view), expected)
        np.testing.assert_array_equal(p, original_p)
        np.testing.assert_array_equal(truth, original_truth)

    def test_metrics_remain_exact_after_canonical_npz_roundtrip(self):
        rng = np.random.default_rng(15840)
        p = rng.random((64, 660, 6), dtype=np.float32)
        p /= p.sum(-1, keepdims=True)
        truth = rng.integers(0, 6, size=(64, 660), dtype=np.uint8)
        # This is the exact SELECT operation that produced column-contiguous
        # truth before the writer stored it row-contiguously.
        p, truth = p[48:].copy(), truth[48:][:, np.arange(660)]
        self.assertTrue(truth.flags.f_contiguous)
        self.assertFalse(truth.flags.c_contiguous)
        expected = screen.metrics(p, truth)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'select.predictions.npz'
            write_deterministic_npz(path, {'probabilities': p, 'truth_state': truth})
            with np.load(path, allow_pickle=False) as z:
                self.assertTrue(z['truth_state'].flags.c_contiguous)
                np.testing.assert_array_equal(z['truth_state'], truth)
                np.testing.assert_array_equal(z['probabilities'], p)
                self.assertEqual(screen.metrics(z['probabilities'], z['truth_state']), expected)
                corrupted = z['probabilities'].copy()
        # Exact agreement must still reject an actual probability change.
        corrupted[0, 0] = np.roll(corrupted[0, 0], 1)
        self.assertNotEqual(screen.metrics(corrupted, truth), expected)


if __name__ == '__main__':
    unittest.main()
