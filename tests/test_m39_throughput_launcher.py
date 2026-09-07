"""Local controller tests only: no models, Docker tasks or private genotype reads."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
SPEC = importlib.util.spec_from_file_location('throughput_launcher', ROOT / 'bin/m39_launch_ordered_throughput.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='m39-throughput-launch-test-')
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        for name in MODULE.SOURCES:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(name)
        self.profile = json.loads((ROOT / MODULE.PROFILE).read_text())
        self.profile_path = self.repo / MODULE.PROFILE
        private = self.repo / '.claude/runs'
        source = private / 'source-run/ordered-output/ordered-profile'
        self.store = source / 'people_48/radius_1cm'
        self.store.mkdir(parents=True)
        (self.store / 'manifest.json').write_text('{}')
        self.parent = source / 'receipt.json'
        self.parent.write_text('{}')
        self.folds = private / 'folds.npz'
        self.folds.write_bytes(b'synthetic-fold-placeholder')
        self.run = private / 'new-throughput-run'
        self.run.mkdir()
        self.args = argparse.Namespace(run_dir=self.run, store_dir=self.store,
            parent_receipt=self.parent, folds=self.folds, profile_config=self.profile_path)
        for name, path in [('parent_receipt', self.parent), ('store_manifest', self.store / 'manifest.json'),
                           ('folds', self.folds)]:
            self.profile[f'{name}_sha256'] = MODULE.sha256(path)
        self.profile_path.write_text(json.dumps(self.profile))
        self.blobs = {name: (self.repo / name).read_bytes() for name in MODULE.SOURCES}
        self.dirty = ''
        self.commit = 'a' * 40
        def git_output(command, **kwargs):
            if command[1] == 'status':
                return self.dirty
            if command[1] == 'rev-parse':
                return self.commit
            if command[1] == 'show':
                name = command[2].split(':', 1)[1]
                if name not in self.blobs:
                    raise subprocess.CalledProcessError(128, command)
                return self.blobs[name]
            raise AssertionError(command)
        mock = patch.object(MODULE.subprocess, 'check_output', side_effect=git_output)
        mock.start()
        self.addCleanup(mock.stop)

    def prepare(self):
        return MODULE.prepare(self.args, repo=self.repo)

    def test_seals_exact_sources_inputs_and_watchdog_argv(self):
        command, receipt_path, session = self.prepare()
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['watchdog_argv'], command)
        self.assertIn('--watch-receipt', command)
        self.assertIn('NXF_OFFLINE=true', receipt['argv'])
        self.assertNotIn('-resume', receipt['argv'])
        self.assertEqual(set(receipt['case_ids']), set(MODULE.EXPECTED_CASES))
        self.assertEqual(receipt['resources']['controller_timeout_seconds'], 1800)
        self.assertEqual(receipt['resources']['time_seconds_per_task'], 900)
        self.assertEqual(receipt['resources']['cpus_total'], 4)
        self.assertEqual(receipt['resources']['memory_gib_total'], 16)
        self.assertFalse(receipt['full_epoch'])
        self.assertFalse(receipt['gpu'])
        self.assertEqual(receipt['store_container_path'], '/m39-ordered-store')
        for source in MODULE.SOURCES:
            frozen = self.run / 'frozen-source' / source
            self.assertEqual(frozen.read_bytes(), self.blobs[source])
            self.assertEqual(MODULE.sha256(frozen), receipt['source_sha256'][source])
        self.assertFalse((self.run / 'frozen-source/ordered-store').exists())
        self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
        self.assertRegex(session, r'^m39-throughput-[0-9a-f]{16}$')

    def test_rejects_dirty_sources(self):
        self.dirty = ' M bin/m39_ordered_models.py'
        with self.assertRaisesRegex(ValueError, 'Commit tracked'):
            self.prepare()

    def test_rejects_source_mismatch_and_unversioned_source(self):
        source = 'bin/m39_ordered_models.py'
        (self.repo / source).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'differs from source commit'):
            self.prepare()
        (self.repo / source).write_bytes(self.blobs[source])
        del self.blobs[source]
        with self.assertRaisesRegex(ValueError, 'Commit required source'):
            self.prepare()

    def test_rejects_reusing_any_immutable_artifact(self):
        (self.run / 'ordered-throughput.completion.json').symlink_to(self.run / 'missing')
        with self.assertRaisesRegex(ValueError, 'Immutable'):
            self.prepare()

    def test_rejects_path_escape_and_unrelated_receipt(self):
        self.args.run_dir = self.repo
        with self.assertRaisesRegex(ValueError, 'private project runs'):
            self.prepare()
        self.args.run_dir = self.run
        self.args.parent_receipt = self.folds
        with self.assertRaisesRegex(ValueError, 'belong to'):
            self.prepare()

    def test_rejects_unsealed_input_and_identity_injection(self):
        with patch.dict(os.environ, {'M39_CONTAINER_USER': '1:2 --privileged'}):
            with self.assertRaisesRegex(ValueError, 'numeric uid:gid'):
                self.prepare()
        self.folds.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'folds differs'):
            self.prepare()

    def test_rejects_case_scope_drift(self):
        for key, value in [('arm', 'common'), ('family', 'global'), ('policy', 'random'),
                           ('batch_size', True), ('size', 'medium'), ('id', '../unsafe')]:
            with self.subTest(key=key):
                profile = json.loads(json.dumps(self.profile))
                profile['cases'][0][key] = value
                with self.assertRaises(ValueError):
                    MODULE._case_ids(profile)
        profile = json.loads(json.dumps(self.profile))
        profile['cases'][1] = profile['cases'][0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            MODULE._case_ids(profile)

    def test_main_detaches_frozen_watchdog_not_unmanaged_nextflow(self):
        argv = []
        for key, value in vars(self.args).items():
            argv.extend((f'--{key.replace("_", "-")}', str(value)))
        with patch.object(MODULE, '__file__', str(self.repo / 'bin/m39_launch_ordered_throughput.py')), \
                patch.object(MODULE.subprocess, 'run') as launch, patch('builtins.print'):
            previous = os.umask(0o077)
            try:
                MODULE.main(argv)
            finally:
                os.umask(previous)
        receipt = json.loads((self.run / 'ordered-throughput.launch.json').read_text())
        actual = launch.call_args.args[0]
        self.assertEqual(actual[:3], ['tmux', 'new-session', '-d'])
        self.assertEqual(shlex.split(actual[-1]), receipt['watchdog_argv'])

    def run_watch(self, wait_result=0, publish=True, timeout=False, popen_error=None):
        _, path, _ = self.prepare()
        if publish:
            for case in MODULE.EXPECTED_CASES:
                output = self.run / 'ordered-throughput-output' / case
                output.mkdir(parents=True)
                (output / 'profile.json').write_text(json.dumps({'decision': 'PASS_ORDERED_THROUGHPUT_TECHNICAL_ONLY'}))
                (output / 'sampling-manifest.json').write_text('{}')
        process = MagicMock(pid=123456)
        process.poll.return_value = None if timeout else wait_result
        if timeout:
            process.wait.side_effect = subprocess.TimeoutExpired('fake-controller', 1800)
        else:
            process.wait.return_value = wait_result
        with patch.object(MODULE.subprocess, 'Popen', return_value=process, side_effect=popen_error) as launch, \
                patch.object(MODULE, 'stop_process_group') as stop, \
                patch.object(MODULE, 'cleanup_containers', return_value={'stopped': [], 'errors': []}) as cleanup:
            code = MODULE.watch_receipt(path)
        completion = json.loads((self.run / 'ordered-throughput.completion.json').read_text())
        return code, completion, launch, stop, cleanup

    def test_watcher_completes_exact_six_and_scoped_cleanup(self):
        code, result, launch, stop, cleanup = self.run_watch()
        self.assertEqual(code, 0)
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(set(result['completed_cases']), set(MODULE.EXPECTED_CASES))
        self.assertTrue(launch.call_args.kwargs['start_new_session'])
        cleanup.assert_called_once_with(result['run_token'])
        stop.assert_not_called()

    def test_watcher_timeout_writes_final_without_waiting_30_minutes(self):
        code, result, launch, stop, cleanup = self.run_watch(timeout=True, publish=False)
        self.assertEqual((code, result['status']), (124, 'TIMEOUT'))
        self.assertEqual(len(result['missing_cases']), 6)
        self.assertEqual(stop.call_count, 1)
        self.assertEqual(cleanup.call_count, 1)

    def test_watcher_failure_and_missing_outputs_never_claim_completion(self):
        code, result, *_ = self.run_watch(publish=False)
        self.assertEqual((code, result['status']), (1, 'INCOMPLETE'))

    def test_watcher_spawn_exception_still_writes_final(self):
        code, result, *_ = self.run_watch(publish=False, popen_error=OSError('synthetic spawn error'))
        self.assertEqual((code, result['status']), (1, 'FAILED'))
        self.assertEqual(result['failure_type'], 'OSError')


class WatchdogSafetyTests(unittest.TestCase):
    def test_cleanup_rejects_unsafe_token_before_docker(self):
        with patch.object(MODULE.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                MODULE.cleanup_containers('x --all')
            run.assert_not_called()

    def test_cleanup_only_exact_labeled_container(self):
        token = 'a' * 16
        cid = 'b' * 64
        responses = [subprocess.CompletedProcess([], 0, stdout=cid + '\n'),
                     subprocess.CompletedProcess([], 0),
                     subprocess.CompletedProcess([], 0, stdout='')]
        with patch.object(MODULE.subprocess, 'run', side_effect=responses) as run:
            result = MODULE.cleanup_containers(token)
        self.assertEqual(result['stopped'], [cid])
        self.assertEqual(run.call_args_list[0].args[0],
                         ['docker', 'ps', '-q', '--no-trunc', '--filter', f'label=dnabr.m39_run={token}'])
        self.assertEqual(run.call_args_list[1].args[0], ['docker', 'stop', '--time', '10', cid])
        self.assertEqual(run.call_args_list[2].args[0], run.call_args_list[0].args[0])

    def test_hanging_process_group_gets_term_then_kill(self):
        process = MagicMock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired('fake', 15), 0]
        with patch.object(MODULE.os, 'killpg') as kill:
            MODULE.stop_process_group(process)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_real_tiny_process_group_stops_without_cloud_or_models(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            MODULE.stop_process_group(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class NextflowContractTests(unittest.TestCase):
    def test_limits_exact_readonly_bind_and_run_label(self):
        cfg = (ROOT / MODULE.CONFIG).read_text()
        module = (ROOT / 'modules/39_ORDERED_THROUGHPUT.nf').read_text()
        for expected in ("stageInMode = 'copy'", 'docker.writableInputMounts = false',
                         'executor.cpus = 4', "executor.memory = '16 GB'", '--network none',
                         '--pull never', '--memory 8g --memory-swap 8g',
                         '--label dnabr.m39_run=${params.m39_run_token}',
                         '--mount type=bind,src=${params.m39_store_dir},dst=/m39-ordered-store,readonly'):
            self.assertIn(expected, cfg)
        for expected in ('cpus 2', "memory '8 GB'", "time '15m'", 'maxForks 2',
                         'val store_dir', "--store-dir '/m39-ordered-store'",
                         'TORCHINDUCTOR_CACHE_DIR=/tmp/m39-torch-cache', 'overwrite: false'):
            self.assertIn(expected, module)
        self.assertNotIn('path store_dir', module)
        self.assertNotRegex(module, r'\b(?:HOME|USER|LOGNAME)\s*=')

    def test_staged_source_inventory_and_local_import_closure(self):
        workflow = (ROOT / MODULE.WORKFLOW).read_text()
        self.assertEqual(len(MODULE.RUNTIME_SOURCES), 11)
        for name in MODULE.RUNTIME_SOURCES:
            self.assertIn(f"'{name}'", workflow)
            self.assertIn(f'bin/{name}', MODULE.SOURCES)
            path = ROOT / 'bin' / name
            self.assertTrue(path.is_file(), f'Missing runtime source {name}')
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    local = ROOT / 'bin' / f'{node.module}.py'
                    if local.is_file():
                        self.assertIn(local.name, MODULE.RUNTIME_SOURCES)
        self.assertIn('channel.value(store.toString())', workflow)
        self.assertIn('store.toRealPath().toString()', workflow)
        self.assertIn('Profile differs from launch seal', workflow)
        self.assertIn('channel.fromList(ids)', workflow)


if __name__ == '__main__':
    unittest.main()
