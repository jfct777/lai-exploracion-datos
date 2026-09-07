"""Development feature routing tests; all inputs are artificial fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bin'))
sys.path.insert(0, str(ROOT/'tests'))
import m39_ordered_training_store as S
from m39_carrier_context import extract_inputs, load_config
from m39_ordered_context import OrderedContextStore
from test_m39_carrier_context import fixture


class DevelopmentStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bridge, _ = fixture(self.root)
        config, paths = load_config(self.bridge, self.root)
        self.data = extract_inputs(paths, config)
        self.keys = np.asarray([hashlib.sha256(f'query-{i}'.encode()).hexdigest().encode()
                                for i in range(6)], dtype='S64')
        self.roles = np.asarray([['TRAIN', 'TRAIN', 'SELECT', 'SELECT', 'SCORE', 'SCORE'],
                                 ['SCORE', 'SCORE', 'TRAIN', 'TRAIN', 'SELECT', 'SELECT'],
                                 ['SELECT', 'SELECT', 'SCORE', 'SCORE', 'TRAIN', 'TRAIN']])
        self.folds = self.root/'folds.npz'
        np.savez(self.folds, sample_key_sha256=self.keys, roles=self.roles,
                 outer_fold=np.arange(3), inner_split_seed=np.arange(3), outer_seed=np.asarray([7]))
        self.counts = dict(TRAIN=2, SELECT=2, SCORE=2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roles_are_authenticated_joined_and_key_sorted(self):
        shuffled = self.keys[[3, 4, 0, 5, 2, 1]]
        for role in ('TRAIN', 'SELECT'):
            indices = S.role_indices(self.folds, shuffled, S.sha256_file(self.folds), 0, self.counts, role)
            np.testing.assert_array_equal(shuffled[indices], np.sort(self.keys[self.roles[0] == role]))

    def test_score_and_wrong_fold_hash_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'development'):
            S.role_indices(self.folds, self.keys, S.sha256_file(self.folds), 0, self.counts, 'SCORE')
        with self.assertRaisesRegex(ValueError, 'hash'):
            S.role_indices(self.folds, self.keys, 'a'*64, 0, self.counts, 'SELECT')

    def test_materializer_only_calls_requested_roles_and_preserves_all_ref(self):
        data = copy.deepcopy(self.data)
        data['sample_key_sha256'] = self.keys
        data['common_target'] = np.tile(data['common_target'], (1, 3, 1))
        data['query_dosage'] = np.tile(data['query_dosage'], (3, 1))
        data['query_observed'] = np.tile(data['query_observed'], (3, 1))
        profile = {'folds': {'path': self.folds.name, 'sha256': S.sha256_file(self.folds)},
                   'fold': 0, 'role_counts': self.counts, 'radii_cm': [.5], 'K': 3}
        profile_path = self.root/'profile.json'
        profile_path.write_text(json.dumps(profile))
        with patch.object(S, 'load_profile', return_value=profile), patch.object(S, 'extract_inputs', return_value=data):
            receipt = S.materialize(self.bridge, profile_path, self.root, ['SELECT'], self.root/'output')
        self.assertEqual(receipt['boundaries']['feature_computation_roles'], ['SELECT'])
        self.assertFalse(receipt['boundaries']['truth_opened'])
        self.assertFalse(receipt['boundaries']['SCORE_features_created'])
        self.assertEqual(len(receipt['profiles']), 1)
        row = receipt['profiles'][0]
        store = OrderedContextStore.open(self.root/'output'/row['relative_path'],
                                         expected_manifest_sha256=row['manifest_sha256'])
        np.testing.assert_array_equal(store.arrays['sample_key_sha256'], np.sort(self.keys[2:4]))
        np.testing.assert_array_equal(store.arrays['reference_sample_key_sha256'], data['reference_sample_key_sha256'])
        self.assertEqual(store.shape[:2], (2, 2))

    def test_nextflow_explicit_roles_and_input_list_add_are_kept(self):
        text = (ROOT/'workflows/m39_ordered_development_store.nf').read_text()
        self.assertIn("['TRAIN', 'SELECT']", text)
        self.assertIn('inputs.add(folds)', text)
        self.assertNotIn('inputs += folds', text)


if __name__ == '__main__':
    unittest.main()
