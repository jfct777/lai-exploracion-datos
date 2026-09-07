"""Synthetic end-to-end checks of role binding, controls and budgeted fitting."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
sys.path.insert(0, str(ROOT / 'tests'))
import m39_ordered_training as T
import m39_ordered_training_data as D
from m39_ordered_batches import pack_batch
from m39_ordered_models import OrderedLAIModel, OrderedModelConfig
from m39_ordered_context import build_factorized_store, canonicalize_reference_homologs
from m39_carrier_context import extract_inputs, load_config, materialize_radius
from m39_profile_ordered import subset_queries
from test_m39_carrier_context import fixture


class OrderedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path, _ = fixture(self.root)
        config, paths = load_config(path, self.root)
        self.data = canonicalize_reference_homologs(extract_inputs(paths, config))

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, data=None, indices=None):
        data = self.data if data is None else data
        if indices is not None:
            data = subset_queries(data, np.asarray(indices, dtype=np.int64))
        candidate, _ = materialize_radius(data, np.arange(2, dtype=np.int64), 3, .5)
        return build_factorized_store(data, candidate)

    def setup_training(self):
        train, select = self.store(indices=[0]), self.store(indices=[1])
        train_path, select_path = self.root/'train', self.root/'select'
        train_saved = train.save(train_path, source_hashes={'fixture': 'a'*64})
        select_saved = select.save(select_path, source_hashes={'fixture': 'a'*64})
        # Reversed binding order verifies the hash-key join, not positional reuse.
        a = train.arrays
        data = {key: a[key].copy() for key in ('alt', 'anchor_indices', 'chrom', 'locus_id', 'pos', 'ref')}
        data.update(coords=a['cM'].copy(), sample_key_sha256=self.data['sample_key_sha256'][::-1].copy(),
                    truth_state=np.asarray([[1, 4], [0, 5]], dtype=np.uint8),
                    baseline=np.full((2, 2, 6), 1/6, dtype=np.float32),
                    full_baseline=np.full((2, 2, 6), 1/6, dtype=np.float32),
                    train_indices=np.asarray([1]), select_indices=np.asarray([0]),
                    source_indices=np.asarray([1, 0]), state_names=np.asarray(T.STATE_NAMES, dtype='S2'))
        binding = self.root/'development.npz'
        np.savez(binding, **data)
        cfg = {'schema_version': T.SCHEMA, 'case_id': 'fixture',
               'scope': 'exploratory_chr22_R0_development_anchors_only', 'arm': 'real',
               'model': asdict(OrderedModelConfig(width=4, depth=1, kernels=(3,), dilations=(1,),
                                                heads=1, core_sites=2)),
               'seed': 7, 'pair_seed': 8, 'anchor_seed': 9, 'sham_seed': 10,
               'anchor_count': 2, 'steps': 2, 'evaluate_every_steps': 1, 'batch_size': 2,
               'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip_norm': 1.,
               'device': 'cpu', 'cpu_threads': 1, 'max_input_bytes': 8*1024**2,
               'max_device_bytes': 0, 'max_rss_bytes': 3*1024**3, 'max_runtime_seconds': 30,
               'train_manifest_sha256': train_saved['manifest_sha256'],
               'select_manifest_sha256': select_saved['manifest_sha256'],
               'development_sha256': T.sha256(binding), 'selection_metric': 'brier',
               'paired_budget_id': 'fixture-budget'}
        config_path = self.root/'training.json'
        config_path.write_text(json.dumps(cfg))
        return train, select, train_path, select_path, binding, config_path, cfg

    def test_binding_joins_role_keys_and_exact_allele_axis(self):
        train, select, _, _, binding, _, cfg = self.setup_training()
        data = D.bind_development(binding, cfg['development_sha256'], train, select)
        np.testing.assert_array_equal(data['train']['truth_state'], [[0, 5]])
        np.testing.assert_array_equal(data['select']['truth_state'], [[1, 4]])
        with self.assertRaisesRegex(ValueError, 'sample universe'):
            D.bind_development(binding, cfg['development_sha256'], select, train)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            D.bind_development(binding, cfg['development_sha256'], train, train)
        with self.assertRaisesRegex(ValueError, 'hash'):
            D.bind_development(binding, 'b'*64, train, select)

    def test_score_field_is_forbidden_even_if_empty(self):
        train, select, _, _, binding, _, _ = self.setup_training()
        with np.load(binding, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        data['score_indices'] = np.asarray([], dtype=np.int64)
        np.savez(binding, **data)
        with self.assertRaisesRegex(ValueError, 'SCORE'):
            D.bind_development(binding, T.sha256(binding), train, select)

    def test_fractional_truth_cannot_be_truncated(self):
        train, select, _, _, binding, _, _ = self.setup_training()
        with np.load(binding, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        data['truth_state'] = data['truth_state'].astype(float) + .1
        np.savez(binding, **data)
        with self.assertRaisesRegex(ValueError, 'integer'):
            D.bind_development(binding, T.sha256(binding), train, select)

    def test_pair_stream_is_paired_exhaustive_and_does_not_drop_remainder(self):
        anchors = np.asarray([2, 7, 13])
        one = list(D.paired_batches(3, anchors, 2, 6, 8))
        two = list(D.paired_batches(3, anchors, 2, 6, 8))
        self.assertEqual(one, two)
        self.assertEqual([len(x) for x in one], [2, 2, 2, 1, 1, 1])
        self.assertTrue(all(len({anchor for _, anchor in batch}) == 1 for batch in one))
        self.assertEqual({batch[0][1] for batch in one[:3]}, set(anchors))
        self.assertEqual(Counter(p for batch in one for p in batch), Counter((p, j) for p in range(3) for j in anchors))
        self.assertEqual(len(list(D.paired_batches(3, anchors, 4, 5, 8))), 5)

    def test_short_budget_covers_every_anchor_before_repeating_an_anchor(self):
        anchors = np.arange(96, dtype=np.int64)
        prefix = list(D.paired_batches(48, anchors, 2, 96, 381))
        extended = list(D.paired_batches(48, anchors, 2, 192, 381))
        self.assertEqual(prefix, extended[:96])
        self.assertEqual({j for batch in prefix for _, j in batch}, set(anchors))
        self.assertEqual(len({(q, j) for batch in extended for q, j in batch}), 384)

    def test_anchor_subset_fixed_without_features(self):
        a = D.fixed_anchor_subset(4, 30, 11)
        np.testing.assert_array_equal(a, D.fixed_anchor_subset(4, 30, 11))
        self.assertTrue(np.all(np.diff(a) > 0))
        np.testing.assert_array_equal(D.fixed_anchor_subset(30, 30, 5), np.arange(30))
        for args in ((0, 30, 0), (31, 30, 0), (True, 30, 0)):
            with self.assertRaises(ValueError):
                D.fixed_anchor_subset(*args)

    def test_sham_preserves_locus_ancestry_masks_and_global_allele_counts(self):
        store = self.store()
        a = store.arrays
        permutation = D.reference_link_permutation(store, 17)
        np.testing.assert_array_equal(permutation, D.reference_link_permutation(store, 17))
        for anchor, row in enumerate(permutation):
            np.testing.assert_array_equal(np.sort(row), np.arange(len(row)))
            np.testing.assert_array_equal(a['reference_ancestry'][row], a['reference_ancestry'])
            np.testing.assert_array_equal(a['ref_observed'][anchor, row], a['ref_observed'][anchor])
            for ancestry in range(3):
                group = a['reference_ancestry'] == ancestry
                self.assertEqual(a['ref_dosage'][anchor, row[group]].sum(), a['ref_dosage'][anchor, group].sum())

    def test_sham_consistent_between_queries_and_homologs_and_does_not_mutate(self):
        store = self.store()
        pairs = [(0, 0), (1, 0), (0, 1)]
        original = pack_batch(store, pairs, 8*1024**2)
        before = {k: v.clone() for k, v in original.items()}
        permutation = D.reference_link_permutation(store, 5)
        changed = D.apply_reference_link_sham(original, store, pairs, permutation)
        for key in original:
            torch.testing.assert_close(original[key], before[key], atol=0, rtol=0)
            if key != 'reference_dosage':
                torch.testing.assert_close(original[key], changed[key], atol=0, rtol=0)
        seen = {}
        for row, (query, anchor) in enumerate(pairs):
            for h, ancestry, k in np.ndindex(2, 3, 3):
                if store.arrays['candidate_mask'][query, anchor, h, ancestry, k]:
                    person = int(store.arrays['candidate_ref_index'][query, anchor, h, ancestry, k])
                    value = float(changed['reference_dosage'][row, h, ancestry, k])
                    identity = (anchor, person)
                    if identity in seen:
                        self.assertEqual(value, seen[identity])
                    seen[identity] = value

    def test_common_predictions_invariant_to_reference_sham(self):
        store = self.store()
        pairs = [(0, 0), (1, 1)]
        batch = pack_batch(store, pairs, 8*1024**2)
        changed = D.apply_reference_link_sham(batch, store, pairs, D.reference_link_permutation(store, 5))
        model = OrderedLAIModel(OrderedModelConfig(width=4, depth=1, kernels=(3,), dilations=(1,), heads=1)).eval()
        for arm in ('common', 'pooled'):
            with torch.no_grad():
                torch.testing.assert_close(model(batch, arm=arm), model(changed, arm=arm), atol=0, rtol=0)

    def test_config_requires_explicit_model_and_rejects_chunked_dropout(self):
        *_, path, cfg = self.setup_training()
        self.assertEqual(T.load_config(path)['steps'], 2)
        for key, value in (('arm', 'score'), ('steps', True), ('selection_metric', 'accuracy')):
            bad = copy.deepcopy(cfg); bad[key] = value; path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): T.load_config(path)
        bad = copy.deepcopy(cfg); bad['model']['dropout'] = .1; path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'dropout'): T.load_config(path)

    def test_synthetic_training_runs_all_arms_with_same_initialization_and_pair_stream(self):
        _, _, train, select, binding, path, cfg = self.setup_training()
        receipts = []
        for arm in T.ARMS:
            cfg['arm'] = arm
            path.write_text(json.dumps(cfg))
            output = self.root/arm
            receipt = T.run_case(train, select, binding, path, output)
            self.assertEqual(receipt['decision'], 'COMPLETED_EXPLORATORY_DEVELOPMENT_CASE')
            self.assertFalse(receipt['scope']['SCORE_opened'])
            self.assertFalse(receipt['scope']['dense_LAI_or_border_F1_evaluated'])
            self.assertEqual(receipt['training_observations'], 2)
            self.assertEqual(len(receipt['curve']), 2)
            self.assertEqual(receipt['selected_step'], min(receipt['curve'],
                key=lambda row:(row['SELECT']['brier'], row['SELECT']['log_loss'], row['step']))['step'])
            checkpoint = torch.load(output/'checkpoint.pt', weights_only=True)
            self.assertEqual(checkpoint['config']['arm'], arm)
            receipts.append(receipt)
        self.assertEqual(len({r['initial_state_sha256'] for r in receipts}), 1)
        self.assertEqual(len({r['training_pair_stream_sha256'] for r in receipts}), 1)

    def test_deadline_failure_cannot_emit_completed_receipt(self):
        _, _, train, select, binding, path, _ = self.setup_training()
        with patch.object(T, '_limits', side_effect=ValueError('time ceiling exceeded')):
            with self.assertRaisesRegex(ValueError, 'time ceiling'):
                T.run_case(train, select, binding, path, self.root/'interrupted')
        self.assertFalse((self.root/'interrupted'/'training.receipt.json').exists())
        self.assertTrue((self.root/'interrupted'/'started.json').exists())


if __name__ == '__main__':
    unittest.main()
