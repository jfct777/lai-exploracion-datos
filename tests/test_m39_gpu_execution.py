"""Local contract tests for one-worker GPU execution; no cloud calls are made."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import os
import shutil
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import m39_gpu_launch as launch
import m39_gpu_serial_profile as serial

REPO = Path(__file__).resolve().parents[1]
IMAGE = 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m39-gpu@sha256:' + 'a' * 64
RUN_ID = 'm39-gpu-profile-test'
SERVICE_ACCOUNT = '123456789012-compute@developer.gserviceaccount.com'


class TestGPUExecution(unittest.TestCase):
    def setUp(self):
        self.native_tmp = tempfile.TemporaryDirectory(prefix='m39-native-fixture-')
        self.addCleanup(self.native_tmp.cleanup)
        self.auth_path = Path(self.native_tmp.name)
        self.auth = launch.native_auth(self.auth_path, SERVICE_ACCOUNT, REPO)
        self.profile = json.loads((REPO / launch.PROFILE).read_text())
        self.args = argparse.Namespace(store_dir=Path('/store'), parent_receipt=Path('/parent.json'),
            folds=Path('/folds.npz'), profile_config=Path('/config.json'),
            output_dir=Path('/out'), source_commit='1' * 40)

    def test_native_directory_is_private_external_and_never_contains_adc(self):
        self.assertEqual(self.auth['mode'], 'attached_vm_metadata')
        adc = self.auth_path / 'application_default_credentials.json'
        adc.write_text('fixture-only-not-a-credential')
        with self.assertRaisesRegex(ValueError, 'ADC'):
            launch.native_auth(self.auth_path, SERVICE_ACCOUNT, REPO)
        adc.unlink()
        self.auth_path.chmod(0o755)
        with self.assertRaisesRegex(ValueError, '0700'):
            launch.native_auth(self.auth_path, SERVICE_ACCOUNT, REPO)
        self.auth_path.chmod(0o700)
        with self.assertRaisesRegex(ValueError, 'outside'):
            launch.native_auth(self.auth_path, SERVICE_ACCOUNT, self.auth_path)

    def test_native_directory_rejects_symlinks_and_human_or_unsafe_accounts(self):
        alias = self.auth_path / 'alias'
        alias.symlink_to(self.auth_path, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            launch.native_auth(self.auth_path, SERVICE_ACCOUNT, REPO)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            launch.native_auth(alias, SERVICE_ACCOUNT, REPO)
        for account in ('person@example.org', SERVICE_ACCOUNT + ';id', '', None):
            with self.assertRaisesRegex(ValueError, 'service-account'):
                launch.native_auth(self.auth_path, account, REPO)

    def test_process_identity_overrides_stale_tmux_env_without_changing_parent(self):
        # Invalid inherited settings are harmless placeholders, not real credentials.
        inherited = {**os.environ, **{key: 'fixture-stale' for key in launch.AUTH_OVERRIDES},
                     'CLOUDSDK_CONFIG': '/fixture/stale', 'CLOUDSDK_CORE_ACCOUNT': 'person@example.org'}
        prefix = launch.native_env_prefix(self.auth)
        program = ('import os,json; print(json.dumps({k:os.environ.get(k) for k in '
                   + repr([*launch.AUTH_OVERRIDES, 'CLOUDSDK_CONFIG', 'CLOUDSDK_CORE_ACCOUNT',
                           'M39_GPU_SERVICE_ACCOUNT']) + '}))')
        before = dict(os.environ)
        actual = json.loads(subprocess.check_output([*prefix, sys.executable, '-c', program],
                                                     env=inherited, text=True))
        self.assertTrue(all(actual[key] is None for key in launch.AUTH_OVERRIDES))
        self.assertEqual(actual['CLOUDSDK_CONFIG'], str(self.auth_path))
        self.assertEqual(actual['CLOUDSDK_CORE_ACCOUNT'], SERVICE_ACCOUNT)
        self.assertEqual(actual['M39_GPU_SERVICE_ACCOUNT'], SERVICE_ACCOUNT)
        self.assertEqual(dict(os.environ), before)
        receipt = Path('/fixture/run/gpu.launch.json')
        self.assertEqual(launch.watcher_command(receipt, self.auth)[:len(prefix)], prefix)

    def test_prepare_seals_native_identity_for_nextflow_and_watcher(self):
        with tempfile.TemporaryDirectory(prefix='m39-source-fixture-') as td:
            repo = Path(td)
            private = repo / '.claude/runs'
            run, store = private / RUN_ID, private / 'store'
            run.mkdir(parents=True); store.mkdir()
            parent, folds = private / 'parent.json', private / 'folds.npz'
            parent.write_text('{}'); folds.write_bytes(b'fixture-folds')
            arrays = {}
            for index in range(28):
                path = store / f'array_{index}.npy'
                path.write_bytes(f'fixture-{index}'.encode())
                arrays[str(index)] = {'file': path.name, 'sha256': serial.sha256(path)}
            manifest = store / 'manifest.json'
            manifest.write_text(json.dumps({'arrays': arrays}))
            profile = {**self.profile, 'store_manifest_sha256': serial.sha256(manifest),
                       'parent_receipt_sha256': serial.sha256(parent), 'folds_sha256': serial.sha256(folds)}
            blobs = {name: (REPO / name).read_bytes() for name in launch.SOURCES}
            blobs[launch.PROFILE] = json.dumps(profile).encode()
            for name, blob in blobs.items():
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
            def git_result(command, **kwargs):
                if command[1] == 'status': return ''
                if command[1] == 'rev-parse': return '1'*40
                self.assertEqual(command[1], 'show')
                return blobs[command[2].split(':', 1)[1]]
            args = argparse.Namespace(run_dir=run, image=IMAGE, store_dir=store,
                parent_receipt=parent, folds=folds, native_auth_dir=self.auth_path,
                service_account=SERVICE_ACCOUNT)
            with patch.object(launch.subprocess, 'check_output', side_effect=git_result):
                receipt_path = launch.prepare(args, repo)
            receipt = json.loads(receipt_path.read_text())
            prefix = launch.native_env_prefix(receipt['native_auth'])
            self.assertEqual(receipt['argv'][:len(prefix)], prefix)
            self.assertEqual(launch.watcher_command(receipt_path, receipt['native_auth'])[:len(prefix)], prefix)
            self.assertEqual(receipt['native_auth']['service_account'], SERVICE_ACCOUNT)
            self.assertEqual(len(list(self.auth_path.iterdir())), 0)
            self.assertEqual(len(receipt['case_ids']), 6)
            self.assertNotIn('credentials', (run/'gpu.launch.json').read_text())

    def test_same_six_cases_no_duplicates(self):
        commands = serial.case_commands(self.args, self.profile)
        self.assertEqual(len(commands), 6)
        self.assertEqual([name for name, _ in commands], [case['id'] for case in self.profile['cases']])
        for name, command in commands:
            self.assertEqual(command[command.index('--case-id') + 1], name)
            self.assertEqual(command[command.index('--store-dir') + 1], '/store')
            self.assertEqual(command[command.index('--output-dir') + 1], '/out/' + name)
            self.assertNotIn('--truth', command)

    def test_cpu_profile_rejected(self):
        self.profile['device'] = 'cpu'
        with self.assertRaises(ValueError):
            serial.case_commands(self.args, self.profile)

    def test_parallel_worker_rejected(self):
        self.profile['max_workers'] = 2
        with self.assertRaises(ValueError):
            serial.case_commands(self.args, self.profile)

    def test_reordered_or_missing_cases_rejected(self):
        for cases in (self.profile['cases'][::-1], self.profile['cases'][:-1], self.profile['cases'][:1] * 6):
            cfg = dict(self.profile, cases=cases)
            with self.assertRaises(ValueError):
                serial.case_commands(self.args, cfg)

    def test_own_cloud_prefix_and_digest(self):
        self.assertEqual(launch.validate_target(RUN_ID, IMAGE), launch.OWN_RUNS + RUN_ID)

    def test_unsafe_image_and_paths_rejected(self):
        for image in (IMAGE.replace('@sha256:', ':'), IMAGE.replace('uspbr-242713', 'another-project'),
                      'ubuntu:latest', IMAGE + '; rm -rf /'):
            with self.assertRaises(ValueError):
                launch.validate_target(RUN_ID, image)
        for run_id in ('../../lab', 'm39-gpu-../lab', 'm39-gpu-', 'other-run', 'm39-gpu-x;echo'):
            with self.assertRaises(ValueError):
                launch.validate_target(run_id, IMAGE)

    def test_only_owned_unfinished_jobs_cancelled(self):
        name = 'projects/uspbr-242713/locations/us-central1/jobs/job-one'
        jobs = [{'name': name, 'labels': {'team': 'frank', 'm39_run': RUN_ID},
                 'status': {'state': state}} for state in ('RUNNING', 'SUCCEEDED', 'FAILED')]
        with patch.object(launch.subprocess, 'check_output', return_value=json.dumps(jobs).encode()), \
             patch.object(launch.subprocess, 'run') as run:
            result = launch.cancel_owned_jobs(RUN_ID, auth=self.auth)
        self.assertEqual(result['deleted_unfinished_jobs'], [name])
        self.assertEqual(run.call_count, 1)
        self.assertIn('job-one', run.call_args.args[0])
        self.assertNotIn('storage', run.call_args.args[0])
        self.assertIn('--account=' + SERVICE_ACCOUNT, run.call_args.args[0])
        prefix = launch.native_env_prefix(self.auth)
        self.assertEqual(run.call_args.args[0][:len(prefix)], prefix)

    def test_foreign_job_is_never_cancelled(self):
        jobs = [{'name': 'projects/uspbr-242713/locations/us-central1/jobs/foreign',
                 'labels': {'team': 'somebody', 'm39_run': RUN_ID}, 'status': {'state': 'RUNNING'}}]
        with patch.object(launch.subprocess, 'check_output', return_value=json.dumps(jobs).encode()), \
             patch.object(launch.subprocess, 'run') as run:
            result = launch.cancel_owned_jobs(RUN_ID, auth=self.auth)
        run.assert_not_called()
        self.assertEqual(result['errors'], ['ValueError'])

    def test_unknown_cloud_state_is_not_zero_jobs(self):
        with patch.object(launch.subprocess, 'check_output', side_effect=subprocess.TimeoutExpired('gcloud', 25)):
            result = launch.cancel_owned_jobs(RUN_ID, auth=self.auth)
        self.assertEqual(result['errors'], ['TimeoutExpired'])

    def test_exact_job_lookup_avoids_slow_historical_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            (run / 'gpu.nextflow.log').write_text(
                '[GOOGLE BATCH] Process `M39_GPU_SERIAL_PROFILE (six-frozen-cases-one-l4)` '
                f'submitted > job=nf-123; uid=uid-123; work-dir={launch.OWN_RUNS}{RUN_ID}/work/12/abc\n')
            job = {'name': 'projects/uspbr-242713/locations/us-central1/jobs/nf-123', 'uid': 'uid-123',
                   'labels': {'team': 'frank', 'm39_run': RUN_ID}, 'status': {'state': 'RUNNING'}}
            with patch.object(launch.subprocess, 'check_output', return_value=json.dumps(job).encode()) as query, \
                 patch.object(launch.subprocess, 'run') as cancel:
                result = launch.cancel_owned_jobs(RUN_ID, run, auth=self.auth)
            self.assertIn('describe', query.call_args.args[0])
            self.assertNotIn('list', query.call_args.args[0])
            self.assertIn('--account=' + SERVICE_ACCOUNT, query.call_args.args[0])
            self.assertEqual(cancel.call_count, 1)
            self.assertEqual(result['errors'], [])

    def test_cancellation_without_sealed_identity_fails_before_cloud_call(self):
        with patch.object(launch.subprocess, 'check_output') as query:
            result = launch.cancel_owned_jobs(RUN_ID)
        query.assert_not_called()
        self.assertEqual(result['errors'], ['ValueError'])

    def test_watcher_rejects_identity_drift_before_starting_nextflow(self):
        with tempfile.TemporaryDirectory() as td:
            receipt = Path(td) / 'gpu.launch.json'
            receipt.write_text(json.dumps({'schema_version': 'm39-gpu-launch-v1',
                'run_id': RUN_ID, 'image': IMAGE, 'native_auth': self.auth,
                'argv': ['env', 'CLOUDSDK_CORE_ACCOUNT=person@example.org', 'nextflow']}))
            with patch.object(launch.subprocess, 'Popen') as process:
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    launch.watch(receipt)
            process.assert_not_called()

    def test_frozen_launcher_dependency_closure_imports_without_repo_path(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            for name in ('m39_gpu_launch.py', 'm39_launch_ordered_training.py',
                         'm39_launch_ordered_throughput.py'):
                (directory / name).write_bytes((REPO / 'bin' / name).read_bytes())
            result = subprocess.run([sys.executable, str(directory / 'm39_gpu_launch.py'), '--help'],
                cwd=directory, env={**os.environ, 'PYTHONPATH': ''}, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_seal_detects_tampering_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            source = directory / 'm39_fixture.py'
            source.write_text('test')
            value = {'schema_version': 'm39-gpu-source-seal-v1', 'source_commit': '1' * 40,
                     'source_sha256': {source.name: serial.sha256(source)}}
            seal = directory / 'seal.json'
            seal.write_text(json.dumps(value))
            self.assertEqual(serial.verify_seal(seal, directory), value)
            source.write_text('changed')
            with self.assertRaises(ValueError):
                serial.verify_seal(seal, directory)
            value['source_sha256'] = {'../outside.py': 'a' * 64}
            seal.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                serial.verify_seal(seal, directory)

    def test_resource_config_has_one_worker_and_bounded_timeout(self):
        text = (REPO / launch.CONFIG).read_text()
        for expected in ("machineType = 'g2-standard-8'", "accelerator = [request: 1, type: 'nvidia-l4']",
                         "time = '30m'", 'maxRetries = 0', 'maxForks = 1', "team: 'frank'",
                         'installGpuDrivers = true', launch.OWN_RUNS):
            self.assertIn(expected, text)
        self.assertNotIn('gs://projects-usp/', text)
        self.assertEqual(launch.CONTROLLER_SECONDS, 3600)
        self.assertIn("System.getenv('M39_GPU_SERVICE_ACCOUNT')", text)
        self.assertIn('serviceAccountEmail', text)
        self.assertNotIn(SERVICE_ACCOUNT, text)

    def test_worker_failure_stops_later_cases_and_writes_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            args = copy.copy(self.args)
            args.output_dir = Path(td) / 'out'
            args.source_seal = Path(td) / 'seal.json'
            args.profile_config = Path(td) / 'profile.json'
            args.profile_config.write_text(json.dumps(self.profile))
            seal = {'source_commit': args.source_commit, 'profile_sha256': serial.sha256(args.profile_config)}
            args.source_seal.write_text(json.dumps(seal))
            args.max_seconds = 1800
            with patch.object(serial, 'verify_seal', return_value=seal), \
                 patch.object(serial.subprocess, 'Popen') as constructor:
                process = constructor.return_value
                process.wait.return_value = 2
                process.poll.return_value = 2
                self.assertEqual(serial.run(args), 1)
            self.assertEqual(constructor.call_count, 1)
            result = json.loads((args.output_dir / 'gpu-serial-completion.json').read_text())
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(result['completed_cases'], [])
            self.assertFalse(result['accuracy_evaluated'])

    @unittest.skipUnless(shutil.which('nextflow'), 'Nextflow executable unavailable')
    def test_native_worker_account_config_accepts_valid_and_rejects_invalid_values(self):
        base = {**os.environ, 'NXF_VER': '26.04.6', 'NXF_OFFLINE': 'true',
                'M39_GPU_RUN_ID': RUN_ID, 'M39_GPU_IMAGE': IMAGE}
        command = ['nextflow', '-C', str(REPO / launch.CONFIG), 'config', '-flat']
        for account in (SERVICE_ACCOUNT, '', 'person@example.org'):
            result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=30,
                                    env={**base, 'M39_GPU_SERVICE_ACCOUNT': account})
            if account == SERVICE_ACCOUNT:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("google.batch.serviceAccountEmail = '" + account + "'", result.stdout)
            else:
                self.assertNotEqual(result.returncode, 0)
                # Nextflow wraps the validation exception as a config error.
                self.assertIn('Unable to parse config file', result.stdout + result.stderr)
                self.assertNotIn('google.batch.serviceAccountEmail = ', result.stdout)

    @unittest.skipUnless(shutil.which('nextflow'), 'Nextflow executable unavailable')
    def test_nextflow_stub_one_process_no_gpu(self):
        with tempfile.TemporaryDirectory(prefix='m39-gpu-stub-') as td:
            root = Path(td)
            store = root / 'store'
            store.mkdir()
            parent, folds, seal = (root / name for name in ('parent.json', 'folds.npz', 'seal.json'))
            parent.write_text('{}')
            folds.write_bytes(b'STUB_ONLY')
            profile = REPO / launch.PROFILE
            seal.write_text(json.dumps({'source_commit': '1' * 40, 'profile_sha256': serial.sha256(profile),
                'source_sha256': {name: serial.sha256(REPO / 'bin' / name) for name in launch.RUNTIME}}))
            command = ['nextflow', '-C', str(REPO / 'conf/m39_gpu_profile_stub.config'),
                '-log', str(root / 'nextflow.log'), 'run', str(REPO / launch.WORKFLOW),
                '-stub-run', '-work-dir', str(root / 'work'), '-with-trace', str(root / 'trace.tsv')]
            for name, value in {'store_dir': store, 'parent_receipt': parent, 'folds': folds,
                'profile_config': profile, 'source_seal': seal, 'source_commit': '1' * 40,
                'profile_sha256': serial.sha256(profile), 'output_dir': root / 'output'}.items():
                command.extend((f'--m39_{name}', str(value)))
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=75,
                env={**os.environ, 'NXF_VER': '26.04.6', 'NXF_OFFLINE': 'true'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((root / 'output/gpu-profile/STUB_ONLY.txt').is_file())
            rows = (root / 'trace.tsv').read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn('COMPLETED', rows[1])


if __name__ == '__main__':
    unittest.main()
