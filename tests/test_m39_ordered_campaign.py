"""Campaign sequencing, budget and failure tests without Docker, GCS or GPU calls."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import m39_ordered_campaign as campaign


class TestOrderedCampaign(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='m39-campaign-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.item = campaign.Campaign.__new__(campaign.Campaign)
        self.item.path, self.item.run, self.item.repo = self.root / 'campaign.json', self.root, self.root
        self.item.started, self.item.active_receipt, self.item.events = time.monotonic(), None, 0
        self.item.config_hash = 'a' * 64
        self.item.cfg = {'source_commit': 'b' * 40, 'wall_timeout_seconds': 39600,
            'costs': {'worker_hourly_usd': .853624312, 'worker_count_ceiling': 2, 'max_usd': 20,
                      'reserve_usd': 1.2, 'screen_forecast_usd': 8.33, 'followup_forecast_usd': 8.5},
            'screen': {'plan': str(self.root / 'plan-a/plan.json')},
            'followup': {'run_dir': str(self.root / 'run-b'), 'train_store': str(self.root / 'train-b'),
                         'select_store': str(self.root / 'select-b'), 'development': str(self.root / 'dev.npz'),
                         'base_config': str(self.root / 'base.json'),
                         'resources_file': str(self.root / 'resources.json'), 'seeds': [11, 12]}}

    def test_success_sequences_audit_before_followup_and_audits_final(self):
        calls = []
        def stage(name, settings):
            calls.append('stage-' + name)
            return self.root / ('primary-' + name)
        def cpu(name, arguments, mounts):
            calls.append(name)
            self.assertNotIn('/repository', mounts)
            return self.root / ('cpu-' + name) / 'result'
        with patch.object(self.item, 'stage', side_effect=stage), \
             patch.object(self.item, 'cpu', side_effect=cpu), \
             patch.object(self.item, 'measured_screen_cost', return_value=8.33), \
             patch.object(self.item, 'check_sources') as check, \
             patch.object(campaign, 'load_plan', return_value={'stage': 'controlled_followup'}):
            result = self.item.run_all()
        self.assertEqual(calls, ['stage-a', 'audit-a', 'prepare-b', 'stage-b', 'audit-b'])
        self.assertEqual(result['status'], 'TWO_STAGES_AUDITED_EXPLORATORY_ONLY')
        self.assertFalse(result['SCORE_opened'])
        self.assertTrue((self.root / 'campaign.completion.json').is_file())
        check.assert_called_once()

    def test_budget_gate_retains_screen_audit_without_launching_followup(self):
        self.item.started -= 7 * 3600
        with patch.object(self.item, 'stage', return_value=self.root / 'primary-a') as stage, \
             patch.object(self.item, 'measured_screen_cost', return_value=8.33), \
             patch.object(self.item, 'cpu', return_value=self.root / 'audit-a') as cpu:
            result = self.item.run_all()
        self.assertEqual(result['status'], 'SCREEN_AUDITED_FOLLOWUP_STOPPED_BY_BUDGET')
        self.assertEqual(stage.call_count, 1)
        self.assertEqual(cpu.call_count, 1)
        self.assertGreater(result['followup_budget']['followup_projected_usd'], 20)

    def test_slow_actual_screen_increases_followup_forecast_instead_of_hiding_overrun(self):
        self.item.screen_compute_cost_usd = 16.66
        result = self.item.budget_allows_followup()
        self.assertEqual(result['observed_slowdown_multiplier'], 2)
        self.assertEqual(result['followup_forecast_usd'], 17)

    def test_failed_scientific_audit_prevents_followup_and_preserves_failure(self):
        with patch.object(self.item, 'stage', return_value=self.root / 'primary-a') as stage, \
             patch.object(self.item, 'cpu', side_effect=ValueError('paired primary mismatch')):
            result = self.item.run_all()
        self.assertEqual(result['status'], 'FAILED')
        self.assertEqual(result['failure'], 'paired primary mismatch')
        self.assertEqual(stage.call_count, 1)
        self.assertEqual(json.loads((self.root / 'campaign.completion.json').read_text())['status'], 'FAILED')

    def test_global_deadline_terminates_watcher_with_cleanup_grace(self):
        process = Mock()
        process.poll.return_value = None
        self.item.started = 0
        with patch.object(campaign.time, 'monotonic', side_effect=[0, 0, 2]), \
             patch.object(campaign.subprocess, 'Popen', return_value=process):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.item.execute(['fixture-watcher'], self.root / 'watch.log', timeout=1, gpu_watcher=True)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=180)

    def test_retirement_is_bound_to_exact_native_uid_and_labels(self):
        run_id = 'm39-gpu-campaign-test'
        receipt = self.root / 'gpu.launch.json'
        receipt.write_text(json.dumps({'run_id': run_id, 'process_name': 'M39_ORDERED_GPU_TRAINING',
                                      'resources': {'max_jobs': 1}}))
        job = {'uid': 'expected-uid', 'labels': {'m39_run': run_id, 'team': 'frank'},
               'status': {'state': 'SUCCEEDED'}}
        with patch.object(self.item, 'gcloud', return_value=['fixture-gcloud']), \
             patch.object(campaign, 'observed_job_ids', return_value={'job-1': 'expected-uid'}), \
             patch.object(campaign.subprocess, 'check_output', side_effect=[json.dumps(job), '[]']) as read:
            self.item.ensure_retired(receipt)
        self.assertEqual(read.call_count, 2)
        self.assertIn('--filter=labels.m39_run=' + run_id, read.call_args.args[0])
        job['uid'] = 'unexpected-uid'
        with patch.object(self.item, 'gcloud', return_value=['fixture-gcloud']), \
             patch.object(campaign, 'observed_job_ids', return_value={'job-1': 'expected-uid'}), \
             patch.object(campaign.subprocess, 'check_output', return_value=json.dumps(job)):
            with self.assertRaisesRegex(ValueError, 'not terminal or owned'):
                self.item.ensure_retired(receipt)


if __name__ == '__main__':
    unittest.main()
