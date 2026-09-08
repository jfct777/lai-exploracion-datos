"""Synthetic inventory, paired-primary and selection tests; no cloud or real truth."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
sys.path.insert(0, str(ROOT / 'tests'))

import m39_ordered_multichannel_sweep as M
import m39_ordered_multichannel_training as T
import m39_ordered_training_sweep as S
import test_m39_ordered_training_sweep as legacy
import test_m39_ordered_multichannel_training as fixtures


class MultichannelSweepTests(unittest.TestCase):
    tearDown = legacy.SweepTests.tearDown

    def setUp(self):
        legacy.SweepTests.setUp(self)
        self.cfg.update(schema_version=T.SCHEMA, arm='both',
                        multichannel={'hidden_width': 32, 'initial_gate': .01})
        self.base.write_text(json.dumps(self.cfg))

    def groups(self):
        return [dict(id=f'{family}-lr{i}', family=family, learning_rate=lr, seed=17,
                     BOTH_metrics={'brier': both, 'log_loss': both},
                     NONE_metrics={'brier': none, 'log_loss': none})
                for family in ('cnn', 'attention')
                for i, (lr, both, none) in enumerate(((.00003,.10,.90), (.0001,.20,.20),
                                                     (.0003,.20,.20)))]

    def test_screen_inventory_is_six_groups_twelve_fits_with_all_explicit_parameters(self):
        plan_path = M.freeze_plan(self.base, self.resources, self.recipes,
                                  'multichannel_screen', self.root/'screen')
        plan = M.load_plan(plan_path)
        self.assertEqual(len(plan['groups']), 6)
        self.assertEqual(sum(len(group['configs']) for group in plan['groups']), 12)
        for group in plan['groups']:
            configs = [T.load_config(plan_path.parent/spec['file']) for spec in group['configs']]
            self.assertEqual([cfg['arm'] for cfg in configs], ['none', 'both'])
            self.assertTrue(all(cfg['multichannel'] == self.cfg['multichannel'] for cfg in configs))
            self.assertEqual({k:v for k,v in configs[0].items() if k not in ('arm','case_id')},
                             {k:v for k,v in configs[1].items() if k not in ('arm','case_id')})
        with self.assertRaisesRegex(ValueError, 'multichannel stage'):
            M.freeze_plan(self.base, self.resources, self.recipes,
                          'exploratory_screen', self.root/'wrong')

    def test_five_arm_followup_inventory_and_no_overwrite(self):
        plan_path = M.freeze_plan(self.base, self.resources, self.recipes[:1],
                                  'multichannel_followup', self.root/'followup')
        self.assertEqual([T.load_config(plan_path.parent/spec['file'])['arm']
                          for spec in M.load_plan(plan_path)['groups'][0]['configs']], list(T.ARMS))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            M.freeze_plan(self.base, self.resources, self.recipes[:1],
                          'multichannel_followup', plan_path.parent)

    def test_technical_stage_requires_tiny_run_and_cannot_select_parameters(self):
        self.cfg.update(steps=4, evaluate_every_steps=4, evaluate_initial=True)
        self.base.write_text(json.dumps(self.cfg))
        path = M.freeze_plan(self.base, self.resources, self.recipes[:1],
                             'multichannel_technical', self.root/'technical')
        self.assertEqual(len(M.load_plan(path)['groups'][0]['configs']), 5)
        with self.assertRaisesRegex(ValueError, 'requires screening'):
            M.selected_learning_rates({'stage':'multichannel_technical', 'groups':self.groups()})

    def test_selection_means_NONE_BOTH_not_rare_gain_and_keeps_both_families(self):
        summary = {'stage':'multichannel_screen', 'groups':self.groups()}
        chosen = M.selected_learning_rates(summary)
        self.assertEqual(set(chosen), {'cnn','attention'})
        for result in chosen.values():
            self.assertEqual(result['learning_rate'], .0001)
            self.assertEqual(result['mean_NONE_BOTH_brier'], .2)
            self.assertIn('NONE_BOTH', result['selection_metric'])
            self.assertNotIn('REAL', json.dumps(result))
        incomplete = copy.deepcopy(summary)
        incomplete['groups'] = incomplete['groups'][:3]
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            M.selected_learning_rates(incomplete)
        bad = copy.deepcopy(summary)
        bad['groups'][0]['BOTH_metrics']['brier'] = math.nan
        with self.assertRaisesRegex(ValueError, 'invalid'):
            M.selected_learning_rates(bad)

    def test_followup_twenty_fits_uses_fresh_seeds_and_preserves_geometry(self):
        screen = M.freeze_plan(self.base, self.resources, self.recipes,
                               'multichannel_screen', self.root/'screen')
        summary = {'stage':'multichannel_screen', 'groups':self.groups()}
        with patch.object(S, 'audit_results', return_value=summary):
            path = M.freeze_followup(self.base, self.resources, screen,
                                     self.root/'unused', [18,19], self.root/'followup')
            plan = M.load_plan(path)
            self.assertEqual(len(plan['groups']), 4)
            self.assertEqual(sum(len(group['configs']) for group in plan['groups']), 20)
            for group in plan['groups']:
                cfg = T.load_config(path.parent/group['configs'][0]['file'])
                self.assertEqual(cfg['learning_rate'], .0001)
                self.assertEqual(cfg['anchor_count'], self.cfg['anchor_count'])
            with self.assertRaisesRegex(ValueError, 'differ from screening'):
                M.freeze_followup(self.base, self.resources, screen,
                                  self.root/'unused', [17,19], self.root/'reuse-seed')
            self.cfg['anchor_count'] = 128
            self.base.write_text(json.dumps(self.cfg))
            with self.assertRaisesRegex(ValueError, 'geometry/input: anchor_count'):
                M.freeze_followup(self.base, self.resources, screen,
                                  self.root/'unused', [20,21], self.root/'changed-grid')

    def test_complete_pass_schedule_handles_remainder_minibatches(self):
        # Three people, two anchors, batch two: four steps/pass, not 6/2.
        cfg = {'steps':8, 'evaluate_every_steps':4, 'batch_size':2, 'evaluate_initial':True}
        curve = []
        for passes in range(3):
            exposure = {'available_pairs':6, 'minimum_visits_per_pair':passes,
                        'maximum_visits_per_pair':passes, 'complete_passes_over_declared_pairs':passes,
                        'observations':6*passes}
            curve.append({'step':passes*4, 'train_observations_cumulative':6*passes,
                          'TRAIN_exposure':exposure})
        receipt = {'TRAIN_exposure':curve[-1]['TRAIN_exposure'], 'anchor_indices':[0,1],
                   'curve':curve, 'training_observations':12}
        S.verify_complete_pass_schedule(receipt, cfg)
        with self.assertRaisesRegex(ValueError, 'close complete passes'):
            S.verify_complete_pass_schedule(receipt, dict(cfg, evaluate_every_steps=3))
        changed = copy.deepcopy(receipt)
        changed['curve'][1]['TRAIN_exposure']['minimum_visits_per_pair'] = 0
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            S.verify_complete_pass_schedule(changed, cfg)

    def test_shared_auditor_reopens_all_five_fits_and_rejects_corrupt_primaries(self):
        fixture = fixtures.MultichannelTrainingTests()
        fixture.setUp()
        try:
            _, _, train, select, binding, _, base = fixture.setup_multichannel()
            plan_dir = self.root/'synthetic-plan'; plan_dir.mkdir()
            output = self.root/'primary'/'training-synthetic-group'; output.mkdir(parents=True)
            specs, hashes = [], {}
            for arm in T.ARMS:
                cfg = copy.deepcopy(base)
                cfg.update(case_id=f'synthetic-{arm}', paired_budget_id='synthetic-group',
                           arm=arm, train_probe_people=1, evaluate_initial=True)
                path = plan_dir/f'{arm}.json'; path.write_text(json.dumps(cfg))
                T.run_case(train, select, binding, path, output/arm)
                specs.append({'file':path.name, 'sha256':S.sha256(path)})
                hashes[arm] = S.sha256(output/arm/'training.receipt.json')
            plan = {'stage':'multichannel_technical',
                    'groups':[{'id':'synthetic-group', 'configs':specs}]}
            plan_path = plan_dir/'plan.json'; plan_path.write_text(json.dumps(plan))
            completion = {'status':'COMPLETED_DECLARED_PAIRED_ARMS_NEEDS_SCIENTIFIC_POST',
                          'group_id':'synthetic-group', 'plan_sha256':S.sha256(plan_path),
                          'stage':plan['stage'], 'completed_arms':list(T.ARMS),
                          'case_receipt_sha256':hashes}
            (output/'group.completion.json').write_text(json.dumps(completion))
            # CPU-only biological-free fixtures; production manifests require CUDA.
            with patch.object(S, 'load_plan', return_value=plan), patch.object(M, 'load_plan', return_value=plan):
                result = M.audit_results(plan_path, output.parent, self.root/'audit')
                self.assertEqual(len(result['cases']), 5)
                group = result['groups'][0]
                self.assertIn('BOTH_metrics', group)
                self.assertIn('NONE_metrics', group)
                self.assertEqual(set(group['control_minus_BOTH']),
                                 {'none','summary','detail','sham','Fminus_SELECT','Ffull_SELECT'})
                self.assertNotIn('REAL_metrics', group)
                self.assertNotIn('COMMON_metrics', group)
                self.assertTrue(all(row['selected_step'] > 0 for row in result['cases']))
                self.assertTrue(result['scope']['technical_e2e_only'])
                self.assertFalse(result['scope']['SCORE_opened'])
                original = S._verified_case
                def wrong_axes(*args, **kwargs):
                    receipt, arrays, differences = original(*args, **kwargs)
                    if receipt['config']['arm'] == 'both':
                        arrays['truth_state'].flat[0] = (arrays['truth_state'].flat[0] + 1) % 6
                    return receipt, arrays, differences
                with patch.object(S, '_verified_case', side_effect=wrong_axes):
                    with self.assertRaisesRegex(ValueError, 'paired prediction axes'):
                        M.audit_results(plan_path, output.parent)
                curve = output/'both/curve-step-0000000.json'
                curve.chmod(0o600); curve.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'curve checkpoint'):
                    M.audit_results(plan_path, output.parent)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
