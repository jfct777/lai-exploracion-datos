"""Synthetic checks for SELECT-only adaptation, immutable selection and contrasts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import m39_anchor_plan as plan
from m39_anchor_score import paired_summary, score_locked


def write_case(root: Path, config: dict, *, carrier_loss: float = .7) -> Path:
    """Create inert checkpoint bytes and receipt, never train or load a model."""
    folder = root / config['id']
    folder.mkdir()
    rows = []
    for index, arm in enumerate(plan.ARMS):
        checkpoint = folder / f'{arm}.pt'
        checkpoint.write_bytes(f'synthetic-inert-{config["id"]}-{arm}'.encode())
        rows.append({'arm': arm, 'select': {'log_loss': carrier_loss + .1 * (2-index)},
                     'checkpoint_sha256': plan.sha256(checkpoint)})
    receipt = {'schema_version': 'm39-anchor-training-v1', 'score_read': False,
               'config': dict(config), 'results': rows,
               'input_sha256': {'development': 'd'*64, 'features': 'f'*64},
               'source_sha256': {'m39_anchor_screen.py': 's'*64,
                                 'm39_carrier_models.py': 'm'*64}}
    (folder / 'training.receipt.json').write_text(json.dumps(receipt))
    return folder


def rewrite_receipt(folder: Path, change) -> None:
    path = folder / 'training.receipt.json'
    receipt = json.loads(path.read_text())
    change(receipt)
    path.write_text(json.dumps(receipt))


def write_binding(root: Path, development_sha: str = 'd'*64) -> Path:
    path = root/'binding.receipt.json'
    path.write_text(json.dumps({'decision': 'PASS_EXPLORATORY_EXACT_ANCHOR_BINDING',
        'outputs': {'development.npz': {'sha256': development_sha},
                    'score.npz': {'sha256': 'c'*64}}}))
    return path


class AnchorPlanTest(unittest.TestCase):
    def test_initial_plan_bounded_grid_and_profile_are_explicit(self):
        initial = plan.initial_plan()
        self.assertEqual(len(initial['cases']), 12)
        self.assertEqual(len({c['id'] for c in initial['cases']}), 12)
        self.assertEqual(initial['max_total_cases'], 18)
        self.assertEqual(initial['adaptive_max_cases'], 6)
        self.assertFalse(initial['score_for_adaptation'])
        self.assertEqual(initial['arms'], list(plan.ARMS))
        for variant in plan.VARIANTS:
            family = [c for c in initial['cases'] if c['variant'] == variant]
            self.assertEqual({(c['width'], c['radius_cm']) for c in family},
                             {(32, .05), (32, .2), (32, .5), (64, .2)})
            for config in family:
                plan.validate_config(config)
                self.assertEqual((config['initial_epochs'], config['max_epochs']), (8, 16))
        profile = plan.initial_plan(profile=True)
        self.assertEqual(len(profile['cases']), 1)
        self.assertEqual(profile['cases'][0]['max_epochs'], 1)
        self.assertEqual(profile['adaptive_max_cases'], 0)

    def test_adaptation_is_order_invariant_and_only_grows_supported_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = []
            growing = plan.VARIANTS[0]
            for config in plan.initial_plan()['cases']:
                value = .6 if config['width'] == 32 else (.5 if config['variant'] == growing else .8)
                folders.append(write_case(root, config, carrier_loss=value))
            forward = plan.adapt(folders, root/'adapt-forward.json')
            reverse = plan.adapt(list(reversed(folders)), root/'adapt-reverse.json')
            self.assertEqual(forward['cases'], reverse['cases'])
            self.assertEqual(forward['reasons'], reverse['reasons'])
            self.assertFalse(forward['score_read'])
            self.assertEqual(len(forward['training_receipts']), 12)
            for variant in plan.VARIANTS:
                family = [c for c in forward['cases'] if c['variant'] == variant]
                self.assertEqual(len(family), 2)
                self.assertEqual(sum(c['width'] == 128 for c in family), int(variant == growing))
                expected_rates = {.0003, .001} if variant == growing else {.0003, .003}
                self.assertEqual({c['learning_rate'] for c in family}, expected_rates)

    def test_incomplete_or_score_tainted_training_cannot_adapt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = [write_case(root, c) for c in plan.initial_plan()['cases']]
            with self.assertRaisesRegex(ValueError, 'twelve'):
                plan.adapt(folders[:-1], root/'incomplete.json')
            rewrite_receipt(folders[0], lambda r: r.update(score_read=True))
            with self.assertRaisesRegex(ValueError, 'score'):
                plan.adapt(folders, root/'tainted.json')

    def test_duplicate_development_or_source_mismatch_rejected(self):
        for mismatch in ('duplicate', 'development', 'source'):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                folders = [write_case(Path(directory), c) for c in plan.initial_plan()['cases'][:2]]
                if mismatch == 'duplicate':
                    folders[1] = folders[0]
                elif mismatch == 'development':
                    rewrite_receipt(folders[1], lambda r: r['input_sha256'].update(development='other'))
                else:
                    rewrite_receipt(folders[1], lambda r: r['source_sha256'].update(changed='other'))
                with self.assertRaises(ValueError):
                    plan.read_results(folders)

    def test_lock_ties_use_case_id_and_checkpoint_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = [write_case(root, c) for c in plan.initial_plan()['cases']]
            extra = plan.adapt(folders, root/'adapt.json')
            folders.extend(write_case(root, c) for c in extra['cases'])
            binding = write_binding(root)
            locked = plan.lock(list(reversed(folders)), root/'lock.json', binding)
            expected = min(path.name for path in folders)
            self.assertEqual(set(locked['best_by_arm'].values()), {expected})
            self.assertEqual(len(locked['cases']), 18)
            self.assertFalse(locked['score_read'])
            self.assertEqual(locked['expected_score_sha256'], 'c'*64)
            self.assertEqual(locked['binding_receipt_sha256'], plan.sha256(binding))
            with self.assertRaises(FileExistsError):
                plan.lock(folders, root/'lock.json', binding)
            (folders[0]/'carrier.pt').write_bytes(b'tampered-synthetic-checkpoint')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                plan.lock(folders, root/'tampered-lock.json', binding)

    def test_initial_inventory_rejects_in_range_config_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = [write_case(root, c) for c in plan.initial_plan()['cases']]
            rewrite_receipt(folders[0], lambda r: r['config'].update(seed=39062027))
            with self.assertRaisesRegex(ValueError, 'frozen screen plan'):
                plan.adapt(folders, root/'substituted.json')

    def test_lock_rejects_development_binding_or_adaptive_inventory_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = [write_case(root, c) for c in plan.initial_plan()['cases']]
            extra = plan.adapt(folders, root/'adapt.json')
            folders.extend(write_case(root, c) for c in extra['cases'])
            binding = write_binding(root, development_sha='other-development')
            with self.assertRaisesRegex(ValueError, 'authenticated binding development'):
                plan.lock(folders, root/'bad-development-lock.json', binding)
            rewrite_receipt(folders[-1], lambda r: r['config'].update(seed=39062027))
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                plan.lock(folders, root/'bad-inventory-lock.json', binding)

    def test_paired_summary_is_person_weighted_conditional_and_reproducible(self):
        baseline = {'log_loss_per_person': [1., 2., 3., 4.]}
        candidate = {'log_loss_per_person': [.5, 1.5, 2.5, 3.5]}
        result = paired_summary(candidate, baseline)
        self.assertEqual(result['delta_log_loss'], -.5)
        self.assertEqual(result['conditional_person_bootstrap_95'], [-.5, -.5])
        self.assertEqual((result['people'], result['people_improved']), (4, 4))
        self.assertEqual(result['bootstrap_draws'], 10000)
        self.assertIn('not_NAM_population_interval', result['scope'])
        variable = {'log_loss_per_person': [1.2, 1.5, 3.2, 4.4]}
        self.assertEqual(paired_summary(variable, baseline), paired_summary(variable, baseline))
        np.testing.assert_allclose(paired_summary(variable, baseline)['delta_per_person'],
                                   [.2, -.5, .2, .4])

    def test_scorer_rejects_invalid_lock_before_opening_score_or_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root/'lock.json'
            lock_path.write_text(json.dumps({'schema_version': 'm39-anchor-selection-lock-v1',
                                             'score_read': True}))
            with self.assertRaisesRegex(ValueError, 'pre-SCORE'):
                score_locked(lock_path, [], [root/'missing-features.npz'],
                             root/'missing-score.npz', root/'bad-lock-output')
            score_path = root/'inert-score.npz'
            score_path.write_bytes(b'inert synthetic score, not an NPZ')
            lock_path.write_text(json.dumps({'schema_version': 'm39-anchor-selection-lock-v1',
                'score_read': False, 'expected_score_sha256': plan.sha256(score_path),
                'cases': [{'id': 'missing-case'}]}))
            with self.assertRaisesRegex(ValueError, 'case set mismatch'):
                score_locked(lock_path, [], [root/'missing-features.npz'],
                             score_path, root/'bad-cases-output')

    def test_scorer_rejects_replaced_score_before_loading_any_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_path = root/'score.npz'
            score_path.write_bytes(b'original inert score')
            lock_path = root/'lock.json'
            lock_path.write_text(json.dumps({'schema_version': 'm39-anchor-selection-lock-v1',
                'score_read': False, 'expected_score_sha256': plan.sha256(score_path), 'cases': []}))
            score_path.write_bytes(b'replaced inert score')
            with self.assertRaisesRegex(ValueError, 'authenticated exact truth binding'):
                score_locked(lock_path, [], [root/'missing-features.npz'], score_path, root/'out')

    def test_scorer_rejects_invalid_role_exhaustion_before_model_instantiation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = plan.initial_plan()['cases'][1]
            folder = write_case(root, config)
            feature_path = root/'features.npz'
            np.savez(feature_path, radius_cm=np.asarray([.2]))
            rewrite_receipt(folder, lambda r: r.update(source_sha256={},
                input_sha256={'features': plan.sha256(feature_path), 'development': 'd'*64}))
            score_path = root/'score.npz'
            score_path.write_bytes(b'inert score for mocked loader')
            case = {'id': folder.name, 'config': config,
                'receipt_sha256': plan.sha256(folder/'training.receipt.json'),
                'checkpoint_sha256': {a: plan.sha256(folder/f'{a}.pt') for a in plan.ARMS}}
            lock_path = root/'lock.json'
            lock_path.write_text(json.dumps({'schema_version': 'm39-anchor-selection-lock-v1',
                'score_read': False, 'expected_score_sha256': plan.sha256(score_path), 'cases': [case]}))
            saved = {'config': config, 'arm': 'common', 'normalization_mean': [0.]*4,
                     'normalization_scale': [1.]*4}
            for index, data in enumerate(({'score_indices': np.asarray([0, 0])},
                                          {'score_indices': np.asarray([0])},
                                          {'score_indices': np.arange(2), 'train_indices': np.asarray([])})):
                loaded = ({}, torch.zeros((2, 3), dtype=torch.int64), data, None, None, None, None)
                with self.subTest(data_index=index), patch('m39_anchor_score.torch.load', return_value=saved), \
                        patch('m39_anchor_score.load_inputs', return_value=loaded), \
                        patch('m39_anchor_score.CarrierContextModel') as model:
                    with self.assertRaisesRegex(ValueError, 'SCORE'):
                        score_locked(lock_path, [folder], [feature_path], score_path, root/f'out{index}')
                    model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
