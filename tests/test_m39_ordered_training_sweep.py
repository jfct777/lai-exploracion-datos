"""Synthetic checks of explicit recipes, paired result audit and LR selection."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from m39_ordered_models import OrderedModelConfig
import m39_ordered_training_sweep as S
from m39_ordered_training import SCHEMA
import m39_ordered_training as T
import test_m39_ordered_training as fixtures


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = {'schema_version': SCHEMA, 'case_id': 'test-base', 'scope': S.SCOPE,
            'arm': 'real', 'model': asdict(OrderedModelConfig()), 'seed': 17, 'pair_seed': 18,
            'anchor_seed': 19, 'sham_seed': 20, 'anchor_count': 660, 'steps': 31680,
            'evaluate_every_steps': 15840, 'batch_size': 2, 'learning_rate': .001,
            'weight_decay': .0001, 'gradient_clip_norm': 1., 'device': 'cuda:0', 'cpu_threads': 2,
            'max_input_bytes': 2**28, 'max_device_bytes': 2**31, 'max_rss_bytes': 2**33,
            'max_runtime_seconds': 6000, 'train_manifest_sha256': 'a'*64,
            'select_manifest_sha256': 'b'*64, 'development_sha256': 'c'*64,
            'selection_metric': 'brier', 'paired_budget_id': 'test-base',
            'train_probe_people': 4, 'evaluate_initial': True}
        self.base = self.root/'base.json'; self.base.write_text(json.dumps(self.cfg))
        self.resources = {'max_workers': 2, 'task_seconds': 26000, 'controller_seconds': 40000}
        self.recipes = [dict(id=f'{family}-lr{i}', family=family, learning_rate=lr, seed=17, pair_seed=18)
                        for family in ('cnn', 'attention') for i, lr in enumerate((.0001,.0003,.001))]

    def tearDown(self):
        self.temp.cleanup()

    def test_screen_freezes_two_arms_and_all_explicit_architecture_values(self):
        path = S.freeze_plan(self.base, self.resources, self.recipes, 'exploratory_screen', self.root/'plan')
        plan = S.load_plan(path)
        self.assertEqual(len(plan['groups']), 6)
        for group in plan['groups']:
            cfgs = [json.loads((path.parent/spec['file']).read_text()) for spec in group['configs']]
            self.assertEqual([cfg['arm'] for cfg in cfgs], ['common', 'real'])
            self.assertTrue(all(cfg['steps'] == 31680 and cfg['anchor_count'] == 660 for cfg in cfgs))
            self.assertEqual({key:value for key,value in cfgs[0].items() if key not in ('arm','case_id')},
                             {key:value for key,value in cfgs[1].items() if key not in ('arm','case_id')})
        self.assertFalse(json.loads((path.parent/'preparation.receipt.json').read_text())['training_launched'])

    def test_followup_requires_all_four_arms_and_refuses_overwrite(self):
        path = S.freeze_plan(self.base, self.resources, self.recipes[:1], 'controlled_followup', self.root/'follow')
        self.assertEqual(len(S.load_plan(path)['groups'][0]['configs']), 4)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            S.freeze_plan(self.base, self.resources, self.recipes[:1], 'controlled_followup', path.parent)

    def test_technical_e2e_is_complete_but_cannot_select_scientific_parameters(self):
        self.cfg.update(steps=4, evaluate_every_steps=4, evaluate_initial=True)
        self.base.write_text(json.dumps(self.cfg))
        path = S.freeze_plan(self.base, self.resources, self.recipes[:1],
                             'technical_e2e', self.root/'technical')
        plan = S.load_plan(path)
        self.assertEqual(len(plan['groups'][0]['configs']), 4)
        with self.assertRaisesRegex(ValueError, 'requires screening'):
            S.selected_learning_rates({'stage': 'technical_e2e', 'groups': []})

    def test_technical_e2e_cannot_hide_a_large_training_run(self):
        with self.assertRaisesRegex(ValueError, 'tiny complete run'):
            S.freeze_plan(self.base, self.resources, self.recipes[:1],
                          'technical_e2e', self.root/'not-technical')

    def test_selection_minimizes_mean_absolute_error_not_largest_rare_gain(self):
        groups = []
        for family in ('cnn', 'attention'):
            for lr, real, common in ((.0001,.10,.90),(.0003,.20,.20),(.001,.20,.20)):
                groups.append({'id': f'{family}-{lr}', 'family': family, 'learning_rate': lr,
                    'REAL_metrics': {'brier': real, 'log_loss': real},
                    'COMMON_metrics': {'brier': common, 'log_loss': common}})
        summary = {'stage': 'exploratory_screen', 'groups': groups}
        chosen = S.selected_learning_rates(summary)
        self.assertEqual(set(chosen), {'cnn','attention'})
        self.assertTrue(all(value['learning_rate'] == .0003 for value in chosen.values()))
        incomplete = copy.deepcopy(summary)
        incomplete['groups'] = [group for group in groups if group['family'] == 'cnn']
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            S.selected_learning_rates(incomplete)

    def test_primary_audit_reopens_predictions_curves_and_vector_dosage_errors(self):
        fixture = fixtures.OrderedTrainingTests()
        fixture.setUp()
        try:
            _, _, train, select, binding, _, base = fixture.setup_training()
            plan_dir = self.root/'synthetic-plan'; plan_dir.mkdir()
            output = self.root/'primary'/'training-synthetic-group'; output.mkdir(parents=True)
            specs, receipt_hashes = [], {}
            for arm in ('common', 'real'):
                cfg = copy.deepcopy(base)
                cfg.update(case_id=f'synthetic-{arm}', paired_budget_id='synthetic-group',
                           arm=arm, train_probe_people=1, evaluate_initial=True)
                config_path = plan_dir/f'{arm}.json'; config_path.write_text(json.dumps(cfg))
                T.run_case(train, select, binding, config_path, output/arm)
                specs.append({'file': config_path.name, 'sha256': S.sha256(config_path)})
                receipt_hashes[arm] = S.sha256(output/arm/'training.receipt.json')
            plan = {'stage': 'exploratory_screen', 'groups': [{'id': 'synthetic-group', 'configs': specs}]}
            plan_path = plan_dir/'plan.json'; plan_path.write_text(json.dumps(plan))
            completion = {'status': 'COMPLETED_DECLARED_PAIRED_ARMS_NEEDS_SCIENTIFIC_POST',
                'group_id': 'synthetic-group', 'plan_sha256': S.sha256(plan_path),
                'stage': 'exploratory_screen', 'completed_arms': ['common','real'],
                'case_receipt_sha256': receipt_hashes}
            (output/'group.completion.json').write_text(json.dumps(completion))
            # The GPU-only manifest validator is separately tested above; these
            # biological-free optimizer fixtures deliberately execute on CPU.
            with patch.object(S, 'load_plan', return_value=plan):
                summary = S.audit_results(plan_path, output.parent, self.root/'audit')
                self.assertEqual(len(summary['cases']), 2)
                self.assertEqual(len(summary['groups'][0]['control_minus_REAL']['common']['dosage_mae']), 3)
                self.assertTrue((self.root/'audit/cases.csv').is_file())
                curve = output/'real/curve-step-0000000.json'
                curve.chmod(0o600)
                curve.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'curve'):
                    S.audit_results(plan_path, output.parent)
        finally:
            fixture.tearDown()

    def test_followup_generates_both_families_four_arms_and_two_fresh_seeds(self):
        groups = []
        for family in ('cnn', 'attention'):
            for lr in (.0001, .0003):
                groups.append({'id': f'{family}-{lr}', 'family': family, 'learning_rate': lr,
                    'seed': 17, 'REAL_metrics': {'brier': lr, 'log_loss': lr},
                    'COMMON_metrics': {'brier': lr, 'log_loss': lr}})
        summary = {'stage': 'exploratory_screen', 'groups': groups}
        screen_plan = self.root/'screen.json'; screen_plan.write_text('{}')
        with patch.object(S, 'audit_results', return_value=summary):
            path = S.freeze_followup(self.base, self.resources, screen_plan,
                                     self.root/'unused-fixtures', [18, 19], self.root/'followup')
            plan = S.load_plan(path)
            self.assertEqual(len(plan['groups']), 4)
            self.assertTrue(all(len(group['configs']) == 4 for group in plan['groups']))
            for group in plan['groups']:
                cfg = json.loads((path.parent/group['configs'][0]['file']).read_text())
                self.assertEqual(cfg['learning_rate'], .0001)
                self.assertIn(cfg['seed'], (18, 19))
                self.assertEqual(cfg['pair_seed'], cfg['seed'] + 100)
                self.assertEqual(cfg['sham_seed'], self.cfg['sham_seed'])
            with self.assertRaisesRegex(ValueError, 'differ from screening'):
                S.freeze_followup(self.base, self.resources, screen_plan,
                                  self.root/'unused-fixtures', [17, 18], self.root/'bad-seed')


if __name__ == '__main__':
    unittest.main()
