"""Synthetic, local-only tests; make_fixture is reusable for the Nextflow KAT."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import m39_calibration as cal


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def update_npz(path, changes):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    values.update(changes)
    np.savez_compressed(path, **values)


def make_fixture(root: Path) -> dict[str, Path]:
    """Make an authenticated artificial TRAIN/SELECT/SCORE + historical receipt chain.

    Only eight synthetic people and six invented anchors, no project input reads.
    The returned mapping is suitable for the production CLI/Nextflow wrapper.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {key: root / name for key, name in (
        ('development', 'development.npz'), ('score', 'score.npz'), ('plan', 'plan.json'),
        ('comparators', 'comparators.json'), ('selection', 'historical.selection.json'),
        ('score_receipt', 'historical.score.receipt.json'))}
    n, j = 8, 6
    labels = (np.arange(n)[:, None] + np.arange(j)[None, :]) % 6
    probabilities = np.eye(6)[labels].astype(np.float64)
    # Some correct, overconfident wrong, tiny true, and diffuse probabilities.
    probabilities[:, 1] = np.eye(6)[(labels[:, 1] + 1) % 6]
    probabilities[:, 2] = 1 / 6
    probabilities[:, 3] = .01
    probabilities[np.arange(n), 3, labels[:, 3]] = .95
    probabilities[:, 4] = np.eye(6)[(labels[:, 4] + 1) % 6]
    probabilities[np.arange(n), 4, labels[:, 4]] = 1e-15
    probabilities[:, 4] /= probabilities[:, 4].sum(-1, keepdims=True)
    full = .8 * probabilities + .2 / 6
    common = {'chrom': np.full(j, 22, dtype=np.int64), 'pos': np.arange(1, j + 1, dtype=np.int64) * 100,
              'ref': np.asarray(['A'] * j, dtype='S1'), 'alt': np.asarray(['C'] * j, dtype='S1'),
              'coords': np.arange(j, dtype=np.float64) * .1,
              'locus_id': np.arange(10001, 10001 + j, dtype=np.uint64),
              'anchor_indices': np.arange(j, dtype=np.int64),
              'state_names': np.asarray(cal.STATE_NAMES, dtype='S2')}
    for key, indices in (('development', np.arange(5)), ('score', np.arange(5, n))):
        payload = dict(common, baseline=probabilities[indices], full_baseline=full[indices],
                       truth_state=labels[indices], source_indices=indices,
                       sample_key_sha256=np.asarray([hashlib.sha256(f'fixture-{i}'.encode()).hexdigest()
                                                     for i in indices], dtype='S64'))
        if key == 'development':
            payload.update(train_indices=np.arange(3), select_indices=np.arange(3, 5))
        else:
            payload.update(score_indices=np.arange(3))
        np.savez_compressed(paths[key], **payload)
    chosen = {arm: f'synthetic-{arm}' for arm in ('common', 'pooled', 'carrier')}
    selection = {'schema_version': 'm39-anchor-selection-lock-v1', 'score_read': False,
                 'expected_score_sha256': cal.sha256(paths['score']), 'best_by_arm': chosen}
    write_json(paths['selection'], selection)
    with np.load(paths['score'], allow_pickle=False) as score:
        records, descriptors = [], []
        for index, arm in enumerate(chosen):
            path = root / f'{arm}.npz'
            np.savez_compressed(path, probabilities=(.7 - index * .1) * score['baseline'] + (.3 + index * .1) / 6,
                                truth_state=score['truth_state'], sample_key_sha256=score['sample_key_sha256'],
                                coords=score['coords'])
            descriptors.append({'arm': arm, 'case_id': chosen[arm], 'path': path.name, 'sha256': cal.sha256(path)})
            records.append({'case': chosen[arm], 'arm': arm, 'selected_by_arm': True,
                            'predictions_sha256': cal.sha256(path)})
    receipt = {'schema_version': 'm39-anchor-score-v1', 'score_sha256': cal.sha256(paths['score']),
               'selection_lock_sha256': cal.sha256(paths['selection']), 'selected': chosen, 'records': records}
    write_json(paths['score_receipt'], receipt)
    manifest = {'schema_version': 'm39-calibration-comparators-v1',
                'selection': {'path': paths['selection'].name, 'sha256': cal.sha256(paths['selection'])},
                'score_receipt': {'path': paths['score_receipt'].name, 'sha256': cal.sha256(paths['score_receipt'])},
                'comparators': descriptors}
    write_json(paths['comparators'], manifest)
    plan = {'schema_version': 'm39-calibration-plan-v1', 'scope': cal.SCOPE,
            'development_sha256': cal.sha256(paths['development']), 'score_sha256': cal.sha256(paths['score']),
            'comparators_manifest_sha256': cal.sha256(paths['comparators']),
            'temperature_grid': [.25, .5, 1., 2., 4., 8., 16.],
            'temperature_endpoints': [.125, 32.],
            'alpha_grid': [0., .0001, .001, .01, .03, .1, .2, .4, .6, .8, 1.],
            'prior_pseudocount': 1., 'floor': 1e-12, 'tie_tolerance': 1e-12,
            'bootstrap_replicates': 10000, 'bootstrap_seed': 39062027,
            'reliability_bins': [5, 10, 20]}
    write_json(paths['plan'], plan)
    return paths


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.p = np.asarray([[[.2, .3, .1, .05, .35, 0.], [1., 0, 0, 0, 0, 0]]], dtype=np.float32)
        self.prior = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64) / 21

    def test_identity_and_noop_are_byte_exact_nonmutating(self):
        before = self.p.tobytes()
        for family in cal.FAMILIES:
            q = cal.transform(self.p, family, prior=self.prior)
            self.assertEqual(q.tobytes(), before)
            self.assertEqual(q.dtype, self.p.dtype)
            self.assertFalse(np.shares_memory(q, self.p))
        self.assertEqual(before, self.p.tobytes())

    def test_temperature_reference_and_argmax(self):
        for temperature in [.125, .25, .7, 2, 32]:
            q = cal.transform(self.p, 'temperature', temperature)
            expected = self.p.astype(float) ** (1 / temperature)
            expected /= expected.sum(-1, keepdims=True)
            np.testing.assert_allclose(q, expected, rtol=1e-13, atol=1e-16)
            np.testing.assert_array_equal(q == 0, self.p == 0)
            np.testing.assert_array_equal(q.argmax(-1), self.p.argmax(-1))

    def test_smallest_subnormal_support_is_preserved(self):
        p = np.asarray([[[1., np.nextafter(0., 1.), 0, 0, 0, 0]]])
        for temperature in [.125, 32.]:
            q = cal.transform(p, 'temperature', temperature)
            self.assertTrue(np.isfinite(q).all())
            self.assertGreater(q[0, 0, 1], 0)
            np.testing.assert_array_equal(q == 0, p == 0)
            np.testing.assert_allclose(q.sum(-1), 1, atol=1e-14)

    def test_mixture_endpoints_and_exact_prior(self):
        for family in ('uniform_mix', 'prior_mix', 'temperature_prior_mix'):
            q = cal.transform(self.p, family, 1, 1, self.prior)
            expected = np.full(6, 1 / 6) if family == 'uniform_mix' else self.prior
            np.testing.assert_array_equal(q, np.broadcast_to(expected, q.shape))
            self.assertTrue((cal.transform(self.p, family, 1, .0001, self.prior) > 0).all())

    def test_mixture_reference(self):
        p = self.p.astype(float)
        p /= p.sum(-1, keepdims=True)
        expected = .8 * p + .2 * self.prior
        np.testing.assert_allclose(cal.transform(self.p, 'prior_mix', alpha=.2, prior=self.prior), expected)

    def test_bad_parameters_and_probabilities_rejected(self):
        for invalid in (-1, 0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                cal.transform(self.p, 'temperature', invalid)
        for alpha in (-.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                cal.transform(self.p, 'prior_mix', alpha=alpha, prior=self.prior)
        for value in (math.nan, math.inf, -.1, 1.1):
            p = self.p.copy()
            p[0, 0, 0] = value
            with self.assertRaises(ValueError):
                cal.transform(p, 'identity')
        with self.assertRaises(ValueError):
            cal.transform(self.p / 2, 'identity')
        with self.assertRaises(ValueError):
            cal.transform(self.p, 'prior_mix', alpha=.1, prior=np.zeros(6))


class MetricTests(unittest.TestCase):
    def test_arithmetic_reference_no_torch(self):
        p = np.asarray([[[.5, .3, .1, .1, 0, 0], [0, 0, 0, 0, 0, 1]],
                        [[.2, .2, .2, .2, .1, .1], [1, 0, 0, 0, 0, 0]]])
        y = np.asarray([[0, 1], [5, 0]])
        result = cal.evaluate(p, y)
        losses, briers, dosages = [], [], []
        correct = 0
        for i in range(2):
            row_loss, row_brier = [], []
            for j in range(2):
                q, state = p[i, j], y[i, j]
                row_loss.append(-math.log(max(q[state], 1e-12)))
                row_brier.append(sum((q[k] - int(state == k))**2 for k in range(6)))
                dosages.append([abs(sum(q[k] * cal.DOSAGES[k, a] for k in range(6)) - cal.DOSAGES[state, a])
                                for a in range(3)])
                correct += int(np.argmax(q) == state)
            losses.append(sum(row_loss) / 2)
            briers.append(sum(row_brier) / 2)
        np.testing.assert_allclose(result['log_loss_per_person'], losses, atol=1e-14)
        self.assertAlmostEqual(result['brier'], sum(briers) / 2)
        self.assertEqual(result['accuracy'], correct / 4)
        np.testing.assert_allclose(result['dosage_mae'], np.mean(dosages, axis=0))
        self.assertIsNone(result['raw_log_loss'])
        self.assertTrue(result['raw_infinite'])
        self.assertEqual(result['true_zero_count'], 1)
        json.dumps(result, allow_nan=False)

    def test_invalid_labels_are_not_truncated(self):
        p = np.ones((1, 2, 6)) / 6
        for labels in (np.asarray([[.1, 1.]]), np.asarray([[1., 1.]]), np.asarray([[0, 6]]),
                       np.asarray([[-1, 2]]), np.asarray([[True, False]])):
            with self.assertRaises(ValueError):
                cal.evaluate(p, labels)

    def test_raw_positive_and_strata_reconstruct_loss(self):
        y = np.zeros((1, 3), dtype=int)
        p = np.asarray([[[0, 1, 0, 0, 0, 0], [1e-15, 1 - 1e-15, 0, 0, 0, 0], [.8, .2, 0, 0, 0, 0]]])
        q = cal.transform(p, 'uniform_mix', alpha=.1)
        rows = cal.strata_metrics(q, p, y, 1e-12)
        self.assertEqual([r['count'] for r in rows], [1, 1, 1])
        self.assertAlmostEqual(sum(r['weighted_log_loss_contribution'] for r in rows), cal.evaluate(q, y)['log_loss'])
        self.assertFalse(cal.evaluate(q, y)['raw_infinite'])
        self.assertAlmostEqual(cal.evaluate(q, y)['raw_log_loss'], cal.evaluate(q, y)['log_loss'])

    def test_reliability_bin_edges_empty_bins_and_denominators(self):
        p = np.asarray([[[1, 0, 0, 0, 0, 0], [.2, .2, .2, .2, .1, .1]]])
        rows, ece = cal.reliability(p, np.asarray([[0, 0]]), 10)
        self.assertEqual(sum(r['count'] for r in rows), 2)
        self.assertEqual(rows[9]['count'], 1)
        self.assertEqual(rows[2]['count'], 1)
        self.assertIsNone(rows[0]['accuracy'])
        self.assertAlmostEqual(ece, .4)

    def test_paired_bootstrap_exact_same_person_draws(self):
        delta = np.asarray([-.3, .2, .1])
        result = cal.paired_summary({'log_loss_per_person': delta.tolist()},
                                    {'log_loss_per_person': [0., 0., 0.]}, 10000, 37)
        rng = np.random.default_rng(37)
        indices = rng.integers(3, size=(10000, 3))
        expected = np.quantile(np.mean(delta[indices], axis=1), [.025, .975])
        np.testing.assert_array_equal(result['conditional_person_bootstrap_95'], expected)
        self.assertEqual(result['people_improved'], 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def fit(self, suffix='fit'):
        return cal.fit_select(self.paths['development'], self.paths['plan'], self.root / suffix)

    def score(self, suffix='scored'):
        return cal.score_locked(self.root / 'fit/lock.json', self.paths['score'],
                                self.paths['comparators'], self.root / suffix)

    def refresh_development_hash(self):
        plan = json.loads(self.paths['plan'].read_text())
        plan['development_sha256'] = cal.sha256(self.paths['development'])
        write_json(self.paths['plan'], plan)

    def test_end_to_end_replay_and_source_immutability(self):
        original_hashes = {k: cal.sha256(v) for k, v in self.paths.items()}
        fit = self.fit()
        score = self.score()
        self.assertFalse(fit['score_read'])
        self.assertEqual(fit['role_counts'], {'TRAIN': 3, 'SELECT': 2})
        self.assertEqual(len(fit['finalists']), 5)
        self.assertEqual(set(score['results']), {'Fminus', 'Ffull', 'COMMON', 'POOLED', 'CARRIER', 'CALIBRATED'})
        self.assertEqual(len(score['contrasts']), 5)
        with np.load(self.paths['score'], allow_pickle=False) as archive:
            m = fit['model']
            expected = cal.transform(archive['baseline'], m['family'], m['temperature'], m['alpha'], np.asarray(m['prior']))
        with np.load(self.root / 'scored/prediction.npz', allow_pickle=False) as output:
            np.testing.assert_array_equal(expected, output['probabilities'])
        self.assertEqual(original_hashes, {k: cal.sha256(v) for k, v in self.paths.items()})
        self.assertEqual(self.score('replay')['results'], score['results'])
        self.assertTrue(all(p['bootstrap_draws'] == 10000 for p in score['contrasts'].values()))
        with self.assertRaises(FileExistsError):
            self.fit()
        with self.assertRaises(FileExistsError):
            self.score()

    def test_fit_never_opens_score_or_comparators(self):
        forbidden = {self.paths['score'].resolve(), self.paths['comparators'].resolve(),
                     self.paths['score_receipt'].resolve(), self.paths['selection'].resolve()}
        original = Path.open

        def guarded(path, *args, **kwargs):
            self.assertNotIn(path.resolve(), forbidden)
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, 'open', guarded):
            first = self.fit()
        self.paths['score'].write_bytes(b'SCORE now unavailable; fit must never inspect it')
        second = self.fit('fit-again')
        self.assertEqual(first['model'], second['model'])
        self.assertEqual(first['finalists'], second['finalists'])

    def test_select_truth_changes_cannot_change_fitted_parameters_or_prior(self):
        first = self.fit()
        with np.load(self.paths['development']) as archive:
            truth = archive['truth_state'].copy()
        truth[3:] = (truth[3:] + 3) % 6
        update_npz(self.paths['development'], {'truth_state': truth})
        self.refresh_development_hash()
        second = self.fit('fit-other-select')
        for old, new in zip(first['finalists'], second['finalists']):
            for key in ('family', 'temperature', 'alpha', 'train_log_loss', 'stage'):
                self.assertEqual(old[key], new[key])
        self.assertEqual(first['model']['prior'], second['model']['prior'])
        with np.load(self.paths['development']) as archive:
            labels = archive['truth_state'][archive['train_indices']]
        counts = np.bincount(labels.ravel(), minlength=6) + 1
        np.testing.assert_array_equal(first['model']['prior'], counts / counts.sum())

    def test_refinement_cartesian_and_endpoints_are_bounded(self):
        self.assertEqual(cal.neighbors([.25, 1, 16], .25, True, [.125, 32]), [.125, .25, .5])
        self.assertEqual(cal.neighbors([.25, 1, 16], 16, True, [.125, 32]), [4., 16, 32])
        self.assertEqual(cal.neighbors([0, .1, 1], .1, False), [.05, .1, .55])
        self.fit()
        with (self.root / 'fit/candidates.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        for family in cal.FAMILIES:
            entries = [r for r in rows if r['family'] == family]
            self.assertEqual(sum(r['train_finalist'] == 'True' for r in entries), 1)
            self.assertLessEqual(sum(r['stage'] != 'initial' for r in entries), 8)
        self.assertTrue(all(.125 <= float(r['temperature']) <= 32 for r in rows))

    def test_tie_rule_favors_identity_then_fixed_family_order(self):
        records = [{'family': f, 'temperature': 1., 'alpha': 0., 'loss': 1.} for f in reversed(cal.FAMILIES)]
        records[0]['loss'] -= 5e-13
        self.assertEqual(cal.choose(records, 'loss', 1e-12)['family'], 'identity')

    def test_bad_role_indices_labels_states_or_extra_features_rejected(self):
        mutations = [{'train_indices': np.asarray([0., 1., 2.])},
                     {'select_indices': np.asarray([2, 4])},
                     {'truth_state': np.full((5, 6), .5)},
                     {'state_names': np.asarray(cal.STATE_NAMES[::-1], dtype='S2')},
                     {'candidate_mask': np.asarray([2], dtype=np.uint8)},
                     {'score_indices': np.arange(5)},
                     {'coords': np.asarray([0, .1, .2, .3, .4, np.nan])}]
        for index, changes in enumerate(mutations):
            with self.subTest(changes=list(changes)):
                make_fixture(self.root)
                update_npz(self.paths['development'], changes)
                self.refresh_development_hash()
                with self.assertRaises(ValueError):
                    self.fit(f'bad-{index}')

    def test_duplicate_person_or_locus_rejected(self):
        with np.load(self.paths['development']) as archive:
            keys = archive['sample_key_sha256'].copy()
            loci = archive['locus_id'].copy()
        keys[1] = keys[0]
        loci[1] = loci[0]
        for key, values in [('sample_key_sha256', keys), ('locus_id', loci)]:
            make_fixture(self.root)
            update_npz(self.paths['development'], {key: values})
            self.refresh_development_hash()
            with self.assertRaises(ValueError):
                self.fit()

    def test_authentication_and_lock_tampering_rejected(self):
        self.fit()
        lock_path = self.root / 'fit/lock.json'
        lock = json.loads(lock_path.read_text())
        lock['model']['alpha'] = .777
        write_json(lock_path, lock)
        with self.assertRaisesRegex(ValueError, 'lock payload changed'):
            self.score()
        make_fixture(self.root)
        self.paths['development'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
            self.fit('bad-development')

    def test_changed_score_hash_rejected_before_load(self):
        self.fit()
        self.paths['score'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
            self.score()

    def test_score_overlap_rejected_even_with_updated_hash_chain(self):
        with np.load(self.paths['development']) as archive:
            key = archive['sample_key_sha256'][0]
        with np.load(self.paths['score']) as archive:
            keys = archive['sample_key_sha256'].copy()
        keys[0] = key
        update_npz(self.paths['score'], {'sample_key_sha256': keys})
        plan = json.loads(self.paths['plan'].read_text())
        plan['score_sha256'] = cal.sha256(self.paths['score'])
        write_json(self.paths['plan'], plan)
        self.fit()
        with self.assertRaisesRegex(ValueError, 'overlap development'):
            self.score()

    def test_comparator_must_be_historically_selected(self):
        manifest = json.loads(self.paths['comparators'].read_text())
        manifest['comparators'][0]['case_id'] = 'posthoc-other'
        write_json(self.paths['comparators'], manifest)
        plan = json.loads(self.paths['plan'].read_text())
        plan['comparators_manifest_sha256'] = cal.sha256(self.paths['comparators'])
        write_json(self.paths['plan'], plan)
        self.fit()
        with self.assertRaisesRegex(ValueError, 'not historically selected'):
            self.score()

    def test_comparator_alignment_even_when_authenticated(self):
        with np.load(self.root / 'common.npz') as archive:
            coords = archive['coords'][::-1]
        update_npz(self.root / 'common.npz', {'coords': coords})
        receipt = json.loads(self.paths['score_receipt'].read_text())
        receipt['records'][0]['predictions_sha256'] = cal.sha256(self.root / 'common.npz')
        write_json(self.paths['score_receipt'], receipt)
        manifest = json.loads(self.paths['comparators'].read_text())
        manifest['score_receipt']['sha256'] = cal.sha256(self.paths['score_receipt'])
        manifest['comparators'][0]['sha256'] = cal.sha256(self.root / 'common.npz')
        write_json(self.paths['comparators'], manifest)
        plan = json.loads(self.paths['plan'].read_text())
        plan['comparators_manifest_sha256'] = cal.sha256(self.paths['comparators'])
        write_json(self.paths['plan'], plan)
        self.fit()
        with self.assertRaisesRegex(ValueError, 'coords alignment differs'):
            self.score()

    def test_cli_fixture(self):
        source = Path(cal.__file__)
        fit = subprocess.run([sys.executable, str(source), 'fit-select', '--development', str(self.paths['development']),
                              '--plan', str(self.paths['plan']), '--outdir', str(self.root / 'fit')],
                             capture_output=True, text=True)
        self.assertEqual(fit.returncode, 0, fit.stderr)
        score = subprocess.run([sys.executable, str(source), 'score', '--lock', str(self.root / 'fit/lock.json'),
                                '--score', str(self.paths['score']), '--comparators-manifest', str(self.paths['comparators']),
                                '--outdir', str(self.root / 'scored')], capture_output=True, text=True)
        self.assertEqual(score.returncode, 0, score.stderr)
        self.assertEqual(json.loads(score.stdout)['schema_version'], 'm39-calibration-score-v1')


if __name__ == '__main__':
    unittest.main()
