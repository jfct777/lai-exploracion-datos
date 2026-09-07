"""Isolated development-training orchestration tests; never launch cloud work."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import m39_gpu_launch as shared
import m39_ordered_gpu_launch as launch
import m39_ordered_gpu_manifest as manifest
import m39_ordered_gpu_worker as worker

REPO = Path(__file__).resolve().parents[1]
IMAGE = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/test@sha256:' + 'a' * 64
ACCOUNT = '123456789012-compute@developer.gserviceaccount.com'


def make_plan(root: Path, groups=2) -> Path:
    inputs = {name: '1' * 64 for name in ('train_manifest_sha256', 'select_manifest_sha256', 'development_sha256')}
    plan = {'schema_version': manifest.SCHEMA, 'scope': manifest.SCOPE, 'inputs': inputs,
            'stage': 'controlled_followup',
            'resources': {'max_workers': 2, 'task_seconds': 600, 'controller_seconds': 1800}, 'groups': []}
    for i in range(groups):
        name = f'candidate-{i}'
        group = {'id': name, 'configs': []}
        for arm in manifest.ARMS:
            cfg = {'schema_version': 'm39-ordered-anchor-training-v1', 'scope': manifest.SCOPE,
                   'arm': arm, 'case_id': name + '-' + arm, 'paired_budget_id': name,
                   'device': 'cuda:0', 'max_runtime_seconds': 100, 'seed': 1, 'pair_seed': 2,
                   'anchor_seed': 3, 'model': {'family': 'cnn'}, **inputs}
            target = root / (cfg['case_id'] + '.json')
            target.write_text(json.dumps(cfg))
            group['configs'].append({'file': target.name, 'sha256': manifest.sha256(target)})
        plan['groups'].append(group)
    path = root / 'plan.json'
    path.write_text(json.dumps(plan))
    return path


class TestOrderedGPUExecution(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='m39-training-fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = make_plan(self.root)

    def test_four_arms_remain_paired_across_queued_configurations(self):
        self.assertEqual(len(manifest.load_plan(self.path)['groups']), 2)
        self.assertEqual(manifest.ARMS, ('common', 'pooled', 'real', 'sham'))

    def test_changed_config_hash_and_unpaired_seed_rejected(self):
        plan = json.loads(self.path.read_text())
        spec = plan['groups'][0]['configs'][1]
        path = self.root / spec['file']
        cfg = json.loads(path.read_text())
        cfg['pair_seed'] += 1
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'hash'):
            manifest.load_plan(self.path)
        spec['sha256'] = manifest.sha256(path)
        self.path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'different budgets or models'):
            manifest.load_plan(self.path)

    def test_screen_accepts_only_common_real_followup_requires_four(self):
        plan = json.loads(self.path.read_text())
        plan['stage'] = 'exploratory_screen'
        for group in plan['groups']:
            group['configs'] = [group['configs'][0], group['configs'][2]]
        self.path.write_text(json.dumps(plan))
        self.assertEqual(manifest.load_plan(self.path)['stage'], 'exploratory_screen')
        plan['stage'] = 'controlled_followup'
        self.path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'inventory'):
            manifest.load_plan(self.path)

    def test_unbounded_or_duplicated_plans_rejected(self):
        original = json.loads(self.path.read_text())
        for key, value in (('max_workers', 3), ('task_seconds', 999999), ('controller_seconds', 999999)):
            plan = copy.deepcopy(original)
            plan['resources'][key] = value
            self.path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                manifest.load_plan(self.path)
        plan = copy.deepcopy(original)
        plan['groups'].append(plan['groups'][0])
        self.path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            manifest.load_plan(self.path)

    def test_development_inventory_rejects_score_and_unexpected_members(self):
        path = self.root / 'development.npz'
        with zipfile.ZipFile(path, 'w') as archive:
            for name in manifest.DEVELOPMENT_FIELDS:
                archive.writestr(name + '.npy', b'fixture-not-genotypes')
        self.assertGreater(manifest.development_inventory(path, manifest.sha256(path)), 0)
        with zipfile.ZipFile(path, 'a') as archive:
            archive.writestr('score_indices.npy', b'forbidden')
        with self.assertRaisesRegex(ValueError, 'SCORE'):
            manifest.development_inventory(path, manifest.sha256(path))

    def test_exact_job_recovery_accepts_multiple_owned_jobs(self):
        run_id = 'm39-gpu-training-test'
        log = self.root / 'gpu.nextflow.log'
        log.write_text('\n'.join(f'[GOOGLE BATCH] Process `M39_ORDERED_GPU_TRAINING (candidate-{i})` submitted > '
            f'job=job-{i}; uid=uid-{i}; work-dir={shared.OWN_RUNS}{run_id}/work/{i}/'
            for i in range(3)))
        jobs = shared.observed_job_ids(self.root, run_id, process_name='M39_ORDERED_GPU_TRAINING', max_jobs=3)
        self.assertEqual(jobs, {f'job-{i}': f'uid-{i}' for i in range(3)})
        with self.assertRaisesRegex(ValueError, 'excess'):
            shared.observed_job_ids(self.root, run_id, process_name='M39_ORDERED_GPU_TRAINING', max_jobs=2)

    def test_frozen_launcher_import_closure_needs_no_tensor_libraries(self):
        frozen = self.root / 'frozen'
        frozen.mkdir()
        for relative in launch.SOURCES:
            if relative.startswith('bin/'):
                shutil.copyfile(REPO / relative, frozen / Path(relative).name)
        result = subprocess.run([sys.executable, '-I', '-c',
            f'import sys; sys.path.insert(0, {str(frozen)!r}); import m39_ordered_gpu_launch; '
            'assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
            cwd=self.root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prepare_seals_exact_inputs_and_never_stages_score_or_folds(self):
        repo = self.root / 'repository'
        private = repo / '.claude/runs'
        private.mkdir(parents=True)
        for name in launch.SOURCES:
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
        design = private / 'design'
        design.mkdir()
        plan_path = make_plan(design)
        plan = json.loads(plan_path.read_text())
        stores = {}
        for role in ('train', 'select'):
            store = private / role
            store.mkdir()
            arrays = {}
            for i in range(28):
                member = store / f'fixture_{i}.npy'
                member.write_bytes(b'fixture-only-no-genotypes')
                arrays[f'fixture_{i}'] = {'file': member.name, 'sha256': manifest.sha256(member)}
            target = store / 'manifest.json'
            target.write_text(json.dumps({'schema_version': 'm39-ordered-context-factorized-v1', 'arrays': arrays}))
            plan['inputs'][role + '_manifest_sha256'] = manifest.sha256(target)
            stores[role + '_store'] = store
        development = private / 'development.npz'
        with zipfile.ZipFile(development, 'w') as archive:
            for name in manifest.DEVELOPMENT_FIELDS:
                archive.writestr(name + '.npy', b'fixture-only')
        plan['inputs']['development_sha256'] = manifest.sha256(development)
        for group in plan['groups']:
            for spec in group['configs']:
                path = design / spec['file']
                config = json.loads(path.read_text())
                config.update(plan['inputs'])
                path.write_text(json.dumps(config))
                spec['sha256'] = manifest.sha256(path)
        plan_path.write_text(json.dumps(plan))
        auth = self.root / 'native-auth'
        auth.mkdir(mode=0o700)
        run = private / 'm39-gpu-training-test'
        run.mkdir()
        args = argparse.Namespace(run_dir=run, image=IMAGE, plan=plan_path, native_auth_dir=auth,
            service_account=ACCOUNT, development=development, **stores)
        def git(command, **kwargs):
            if command[1] == 'status':
                return ''
            if command[1] == 'rev-parse':
                return '1' * 40
            return (repo / command[2].split(':', 1)[1]).read_bytes()
        with patch.object(launch.subprocess, 'check_output', side_effect=git):
            receipt_path = launch.prepare(args, repo)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['resources']['max_concurrent_workers'], 2)
        self.assertEqual(receipt['process_name'], 'M39_ORDERED_GPU_TRAINING')
        self.assertEqual(set(receipt['inputs']), {'train_store', 'select_store', 'development'})
        self.assertFalse(receipt['SCORE_staged'])
        self.assertNotIn('--m39_folds', receipt['argv'])
        self.assertEqual(receipt['argv'][:len(shared.native_env_prefix(receipt['native_auth']))],
                         shared.native_env_prefix(receipt['native_auth']))
        config = next(path for path in (run / 'training-plan').iterdir() if path.name != 'plan.json')
        config.write_text('changed')
        with patch.object(shared.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(ValueError, 'Frozen run artifact'):
                shared.watch(receipt_path)
        start.assert_not_called()

    def test_one_failed_job_lookup_does_not_block_other_owned_job_cleanup(self):
        run_id = 'm39-gpu-training-test'
        (self.root / 'gpu.nextflow.log').write_text('\n'.join(
            f'[GOOGLE BATCH] Process `M39_ORDERED_GPU_TRAINING (candidate-{i})` submitted > '
            f'job=job-{i}; uid=uid-{i}; work-dir={shared.OWN_RUNS}{run_id}/work/{i}/' for i in range(2)))
        authdir = self.root / 'auth'
        authdir.mkdir(mode=0o700)
        repo = self.root / 'repo'
        repo.mkdir()
        auth = shared.native_auth(authdir, ACCOUNT, repo)
        job = {'name': 'projects/uspbr-242713/locations/us-central1/jobs/job-1', 'uid': 'uid-1',
               'labels': {'team': 'frank', 'm39_run': run_id}, 'status': {'state': 'RUNNING'}}
        with patch.object(shared.subprocess, 'check_output', side_effect=[
                subprocess.CalledProcessError(1, 'fixture-describe'), json.dumps(job)]), \
             patch.object(shared.subprocess, 'run') as delete:
            result = shared.cancel_owned_jobs(run_id, self.root, auth=auth,
                    process_name='M39_ORDERED_GPU_TRAINING', max_jobs=2)
        self.assertEqual(result['deleted_unfinished_jobs'], [job['name']])
        self.assertEqual(delete.call_count, 1)
        self.assertEqual(result['errors'][0]['job'], 'job-0')

    def test_delete_timeout_does_not_leave_second_owned_job_running(self):
        run_id = 'm39-gpu-training-test'
        (self.root / 'gpu.nextflow.log').write_text('\n'.join(
            f'[GOOGLE BATCH] Process `M39_ORDERED_GPU_TRAINING (candidate-{i})` submitted > '
            f'job=job-{i}; uid=uid-{i}; work-dir={shared.OWN_RUNS}{run_id}/work/{i}/' for i in range(2)))
        authdir = self.root / 'auth'
        authdir.mkdir(mode=0o700)
        repo = self.root / 'repo'
        repo.mkdir()
        auth = shared.native_auth(authdir, ACCOUNT, repo)
        jobs = [{'name': f'projects/uspbr-242713/locations/us-central1/jobs/job-{i}',
                 'uid': f'uid-{i}', 'labels': {'team': 'frank', 'm39_run': run_id},
                 'status': {'state': 'RUNNING'}} for i in range(2)]
        with patch.object(shared.subprocess, 'check_output', side_effect=list(map(json.dumps, jobs))), \
             patch.object(shared.subprocess, 'run', side_effect=[subprocess.TimeoutExpired('delete', 30), Mock()]) as delete:
            result = shared.cancel_owned_jobs(run_id, self.root, auth=auth,
                process_name='M39_ORDERED_GPU_TRAINING', max_jobs=2)
        self.assertEqual(delete.call_count, 2)
        self.assertEqual(result['deleted_unfinished_jobs'], [jobs[1]['name']])
        self.assertEqual(result['errors'][0]['delete_failure'], 'TimeoutExpired')

    def test_failure_transport_preserves_failure_instead_of_forging_success(self):
        output = self.root / 'failed-worker'
        output.mkdir()
        receipt = {'status': 'FAILED', 'exit_code': 124, 'completed_arms': ['common']}
        (output / 'group.completion.json').write_text(json.dumps(receipt))
        self.assertEqual(worker.transport_exit_code(output, 124, False), 124)
        self.assertEqual(worker.transport_exit_code(output, 124, True), 0)
        self.assertEqual(json.loads((output / 'group.completion.json').read_text()), receipt)
        with self.assertRaisesRegex(ValueError, 'failure receipt'):
            worker.transport_exit_code(output, 1, True)

    def _worker_args(self):
        seal = {'schema_version': 'm39-gpu-source-seal-v1', 'source_commit': 'a' * 40,
                'profile_sha256': manifest.sha256(self.path), 'source_sha256': {}}
        path = self.root / 'source-seal.json'
        path.write_text(json.dumps(seal))
        return argparse.Namespace(source_seal=path, plan=self.path, group_id='candidate-0',
            train_store=self.root / 'train', select_store=self.root / 'select',
            development=self.root / 'development.npz', outdir=self.root / 'out'), seal

    def test_worker_stops_after_first_failed_arm(self):
        args, seal = self._worker_args()
        process = Mock()
        process.wait.return_value = process.poll.return_value = 2
        with patch.object(worker, 'verify_seal', return_value=seal), \
             patch.object(worker.subprocess, 'Popen', return_value=process) as constructor:
            self.assertEqual(worker.run(args), 1)
        self.assertEqual(constructor.call_count, 1)
        receipt = json.loads((args.outdir / 'group.completion.json').read_text())
        self.assertEqual(receipt['completed_arms'], [])
        self.assertFalse(receipt['SCORE_opened'])

    def test_worker_audits_pair_stream_and_initialization_before_completion(self):
        args, seal = self._worker_args()
        def complete(command, **kwargs):
            config = Path(command[command.index('--config') + 1])
            cfg = json.loads(config.read_text())
            outdir = Path(command[command.index('--outdir') + 1])
            outdir.mkdir()
            receipt = {'decision': 'COMPLETED_EXPLORATORY_DEVELOPMENT_CASE', 'config': cfg,
                'sources': {'config_sha256': manifest.sha256(config)}, 'scope': {'SCORE_opened': False},
                'initial_state_sha256': 'a' * 64, 'training_pair_stream_sha256': 'b' * 64,
                'anchor_indices': [1, 4], 'training_observations': 4, 'distinct_train_queries_exposed': 2,
                'distinct_train_anchors_exposed': 2, 'batch_policy': 'same-for-fixture'}
            (outdir / 'training.receipt.json').write_text(json.dumps(receipt))
            process = Mock()
            process.wait.return_value = process.poll.return_value = 0
            return process
        with patch.object(worker, 'verify_seal', return_value=seal), \
             patch.object(worker.subprocess, 'Popen', side_effect=complete) as constructor:
            self.assertEqual(worker.run(args), 0)
        self.assertEqual(constructor.call_count, 4)
        receipt = json.loads((args.outdir / 'group.completion.json').read_text())
        self.assertEqual(receipt['completed_arms'], list(manifest.ARMS))

    @unittest.skipUnless(shutil.which('nextflow'), 'Nextflow unavailable')
    def test_native_cos_config_limits_and_no_duplicate_gpu_flags(self):
        env = {**os.environ, 'NXF_VER': '26.04.6', 'NXF_OFFLINE': 'true',
            'M39_GPU_SERVICE_ACCOUNT': ACCOUNT, 'M39_GPU_RUN_ID': 'm39-gpu-training-test',
            'M39_GPU_IMAGE': IMAGE, 'M39_TRAIN_TASK_SECONDS': '600', 'M39_TRAIN_MAX_WORKERS': '2'}
        command = ['nextflow', '-C', str(REPO / launch.CONFIG), 'config', '-flat']
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for expected in ("process.containerOptions = '--user 0:0'", 'process.maxForks = 2',
                         'executor.queueSize = 2', "process.time = '600s'", 'installGpuDrivers = true'):
            self.assertIn(expected, result.stdout)
        for forbidden in ('--gpus', '--runtime', '--device', 'gs://projects-usp/'):
            self.assertNotIn(forbidden, result.stdout)

    @unittest.skipUnless(shutil.which('nextflow'), 'Nextflow unavailable')
    def test_nextflow_stub_queues_two_complete_configurations(self):
        config = self.root / 'stub.config'
        config.write_text("process { executor='local'; cpus=1; memory='128 MB'; maxForks=2 }; "
            "docker.enabled=false; params { m39_gpu_run_id='m39-gpu-training-test'; "
            f"m39_gpu_image='{IMAGE}'; m39_max_workers=2; m39_task_seconds=600 }}")
        train, select = self.root / 'train', self.root / 'select'
        train.mkdir(); select.mkdir()
        development = self.root / 'development.npz'
        development.write_bytes(b'STUB_ONLY')
        seal = self.root / 'seal.json'
        seal.write_text(json.dumps({'source_commit': '1' * 40, 'profile_sha256': manifest.sha256(self.path),
            'source_sha256': {name: manifest.sha256(REPO / 'bin' / name) for name in launch.RUNTIME}}))
        command = ['nextflow', '-C', str(config), '-log', str(self.root / 'nf.log'),
            'run', str(REPO / launch.WORKFLOW), '-stub-run', '-work-dir', str(self.root / 'work'),
            '-with-trace', str(self.root / 'trace.tsv')]
        for key, value in {'train_store': train, 'select_store': select, 'development': development,
                'training_plan': self.path, 'source_seal': seal, 'source_commit': '1' * 40,
                'plan_sha256': manifest.sha256(self.path), 'output_dir': self.root / 'outputs'}.items():
            command.extend(('--m39_' + key, str(value)))
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=75,
            env={**os.environ, 'NXF_VER': '26.04.6', 'NXF_OFFLINE': 'true'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for i in range(2):
            self.assertTrue((self.root / f'outputs/training-candidate-{i}/STUB_ONLY.txt').is_file())
        rows = (self.root / 'trace.tsv').read_text().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all('COMPLETED' in row for row in rows[1:]))


if __name__ == '__main__':
    unittest.main()
