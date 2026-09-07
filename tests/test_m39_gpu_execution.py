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


class TestGPUExecution(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((REPO / launch.PROFILE).read_text())
        self.args = argparse.Namespace(store_dir=Path('/store'), parent_receipt=Path('/parent.json'),
            folds=Path('/folds.npz'), profile_config=Path('/config.json'),
            output_dir=Path('/out'), source_commit='1' * 40)

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
            result = launch.cancel_owned_jobs(RUN_ID)
        self.assertEqual(result['deleted_unfinished_jobs'], [name])
        self.assertEqual(run.call_count, 1)
        self.assertIn('job-one', run.call_args.args[0])
        self.assertNotIn('storage', run.call_args.args[0])

    def test_foreign_job_is_never_cancelled(self):
        jobs = [{'name': 'projects/uspbr-242713/locations/us-central1/jobs/foreign',
                 'labels': {'team': 'somebody', 'm39_run': RUN_ID}, 'status': {'state': 'RUNNING'}}]
        with patch.object(launch.subprocess, 'check_output', return_value=json.dumps(jobs).encode()), \
             patch.object(launch.subprocess, 'run') as run:
            result = launch.cancel_owned_jobs(RUN_ID)
        run.assert_not_called()
        self.assertEqual(result['errors'], ['ValueError'])

    def test_unknown_cloud_state_is_not_zero_jobs(self):
        with patch.object(launch.subprocess, 'check_output', side_effect=subprocess.TimeoutExpired('gcloud', 25)):
            result = launch.cancel_owned_jobs(RUN_ID)
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
                result = launch.cancel_owned_jobs(RUN_ID, run)
            self.assertIn('describe', query.call_args.args[0])
            self.assertNotIn('list', query.call_args.args[0])
            self.assertEqual(cancel.call_count, 1)
            self.assertEqual(result['errors'], [])

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
