"""Recovery contracts; no external VM, container, signal or worker launch."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reboot_recovery', ROOT / 'bin/r02_worker_reboot_resume.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RebootRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'worker12'
        self.directory = self.run / 'repairs/reboot-test'
        self.directory.mkdir(parents=True)
        helper = self.directory / 'r02_optimization_boundary.py'
        helper.write_bytes((ROOT / 'bin/r02_optimization_boundary.py').read_bytes())
        old = self.run / 'repairs/preprocess-v2'
        old.mkdir()
        script, manifest = old / 'r02_optimized_worker.py', old / 'manifest.json'
        script.write_text('# frozen worker test double; never executed\n')
        manifest.write_text('{}')
        failure = old / 'events.jsonl'
        failure.write_text('{"state":"FAILED_CLOSED","error":"original observation"}\n')
        checkpoints = self.run / 'checkpoints'
        checkpoints.mkdir()
        checkpoint = checkpoints / 'chr09_M01_M02_M021.json'
        checkpoint.write_text('{"returncode":0,"test_fixture_only":true}')
        bulk = self.run / 'local_bulk'
        bulk.mkdir()
        account = pwd.getpwuid(os.getuid())
        self.account = account
        started = datetime.now(timezone.utc) - timedelta(hours=1)
        self.config = dict(run_id='r02-test-worker12', bulk=str(bulk),
            destination='/home/jose.tantalean/gcs-dnabr/refined/DNABR_QC/test/worker12')
        self.worker_spec = dict(schema='r02_optimized_worker_v2', run_dir=str(self.run))
        self.worker = types.SimpleNamespace(COUNT_SCHEMA='r02_optimized_worker_v2',
            validate_spec=Mock(return_value=(self.worker_spec, self.config, {}, {}, [])), validate_idle=Mock())
        self.s = dict(schema='r02_worker_reboot_resume_v1', run_dir=str(self.run),
            wrapper_sha256=digest(m.__file__), helper_sha256=digest(helper),
            instance=dict(id='123', name='only-test-worker', zone='test-zone', project='test-project'),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            original_start_utc=started.isoformat(), deadline_utc=(started + timedelta(hours=72)).isoformat(),
            user=account.pw_name, uid=account.pw_uid, gid=account.pw_gid,
            replacement=dict(script_path=str(script), script_sha256=digest(script),
                manifest_path=str(manifest), manifest_sha256=digest(manifest), completion_path=str(old / 'completed.json')),
            failure_events_path=str(failure), failure_events_sha256=digest(failure),
            checkpoint_baseline={checkpoint.name: digest(checkpoint)}, local_bulk=str(bulk), bulk=str(bulk),
            publication_prefix='gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/test/worker12/00_worker_provenance/reboot_recovery/test')
        r = self.s['replacement']
        r['command'] = ['/usr/bin/python3', r['script_path'], '--manifest', r['manifest_path'],
                        '--manifest-sha256', r['manifest_sha256'], '--run']
        self.path = self.directory / 'spec.json'
        self.reload()

    def reload(self):
        self.path.write_text(json.dumps(self.s))
        self.recovery = m.RebootRecovery(self.path, digest(self.path))
        return self.recovery

    def environment(self):
        h = self.recovery
        stack = ExitStack()
        original_load = m.load_verified_module
        stack.enter_context(patch.object(m, 'load_verified_module', side_effect=lambda path, expected, name:
            self.worker if name == '_frozen_reboot_worker' else original_load(path, expected, name)))
        stack.enter_context(patch.object(h.b, 'metadata_identity', return_value=self.s['instance']))
        stack.enter_context(patch.object(h.b, 'run_checked', side_effect=lambda command, **kwargs:
            types.SimpleNamespace(stdout='ext4\n' if command[0] == 'findmnt' else '', returncode=0)))
        stack.enter_context(patch.object(m.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=100 * 1024**3)))
        return stack

    def test_preflight_is_read_only_and_preserves_checkpoints_and_frozen_worker(self):
        before = {p: digest(p) for p in self.run.rglob('*') if p.is_file()}
        with self.environment(), patch.object(m.subprocess, 'Popen') as launch:
            h = self.recovery.validate()
            self.assertFalse(h.directory.exists())
            launch.assert_not_called()
        self.assertEqual(before, {p: digest(p) for p in self.run.rglob('*') if p.is_file()})
        self.worker.validate_idle.assert_called_once()

    def test_wrong_manifest_hash_is_rejected_before_module_load(self):
        with patch.object(m, 'load_verified_module') as load, self.assertRaisesRegex(ValueError, 'hash/path'):
            m.RebootRecovery(self.path, '0' * 64)
        load.assert_not_called()

    def test_changed_sibling_helper_is_rejected(self):
        (self.directory / 'r02_optimization_boundary.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError, 'Module hash'):
            m.RebootRecovery(self.path, digest(self.path))

    def test_wrong_vm_is_rejected(self):
        with self.environment(), patch.object(self.recovery.b, 'metadata_identity', return_value={}), \
             self.assertRaisesRegex(ValueError, 'authorized temporary VM'):
            self.recovery.validate()

    def test_wrong_boot_is_rejected(self):
        self.s['boot_id'] = '0' * 36
        self.reload()
        with self.environment(), self.assertRaisesRegex(ValueError, 'boot identity'):
            self.recovery.validate()

    def test_extended_deadline_is_rejected(self):
        self.s['deadline_utc'] = (datetime.fromisoformat(self.s['deadline_utc']) + timedelta(seconds=1)).isoformat()
        self.reload()
        with self.environment(), self.assertRaisesRegex(ValueError, '72-hour'):
            self.recovery.validate()

    def test_expired_original_deadline_is_rejected(self):
        start = datetime.now(timezone.utc) - timedelta(hours=73)
        self.s.update(original_start_utc=start.isoformat(), deadline_utc=(start + timedelta(hours=72)).isoformat())
        self.reload()
        with self.environment(), self.assertRaisesRegex(ValueError, '72-hour'):
            self.recovery.validate()

    def test_changed_worker_bytes_are_rejected(self):
        Path(self.s['replacement']['script_path']).write_text('# different')
        with self.environment(), self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.recovery.validate()

    def test_changed_command_is_rejected(self):
        self.s['replacement']['command'].append('--unchecked')
        self.reload()
        with self.environment(), self.assertRaisesRegex(ValueError, 'command changed'):
            self.recovery.validate()

    def test_changed_checkpoint_is_rejected(self):
        (self.run / 'checkpoints/chr09_M01_M02_M021.json').write_text('{}')
        with self.environment(), self.assertRaisesRegex(ValueError, 'checkpoint inventory changed'):
            self.recovery.validate()

    def test_changed_failure_history_is_rejected(self):
        Path(self.s['failure_events_path']).write_text('{}\n')
        with self.environment(), self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.recovery.validate()

    def test_completion_exists_cannot_relaunch_worker(self):
        Path(self.s['replacement']['completion_path']).write_text('{}')
        with self.environment(), self.assertRaisesRegex(ValueError, 'completion exists'):
            self.recovery.validate()

    def test_unbound_or_nonlocal_scratch_is_rejected(self):
        for mode in ('unbound', 'fuse', 'low_space'):
            with self.subTest(mode=mode), self.environment(), ExitStack() as stack:
                if mode == 'unbound':
                    stack.enter_context(patch.object(Path, 'samefile', return_value=False))
                    expected = 'bind is absent'
                elif mode == 'fuse':
                    stack.enter_context(patch.object(self.recovery.b, 'run_checked',
                        return_value=types.SimpleNamespace(stdout='fuse.gcsfuse\n')))
                    expected = 'not ext4'
                else:
                    stack.enter_context(patch.object(m.shutil, 'disk_usage',
                        return_value=types.SimpleNamespace(free=12 * 1024**3 - 1)))
                    expected = 'below 12 GiB'
                with self.assertRaisesRegex(ValueError, expected):
                    self.recovery.validate()

    def test_live_old_worker_rejects_preflight(self):
        self.worker.validate_idle.side_effect = ValueError('Old process remains alive')
        with self.environment(), self.assertRaisesRegex(ValueError, 'Old process'):
            self.recovery.validate()

    def test_occupied_original_runner_lock_is_not_stolen(self):
        import fcntl
        with (self.run / '.runner.lock').open('w') as lock:
            lock.write('original owner evidence')
            lock.flush()
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.environment(), self.assertRaises(BlockingIOError):
                self.recovery.validate()
        self.assertEqual((self.run / '.runner.lock').read_text(), 'original owner evidence')

    def test_apply_success_uses_real_exit_status_and_authenticated_completion(self):
        self.exercise_apply(0)

    def test_apply_failure_keeps_old_records_and_stops_vm_without_retry(self):
        self.exercise_apply(9)

    def exercise_apply(self, code):
        h = self.recovery
        before = {p: digest(p) for p in self.run.rglob('*') if p.is_file()}
        child = Mock(pid=54321, returncode=code)
        child.poll.return_value = code
        with self.environment(), ExitStack() as stack:
            h.validate()
            stack.enter_context(patch.object(m.os, 'geteuid', return_value=0))
            launch = stack.enter_context(patch.object(m.subprocess, 'Popen', return_value=child))
            completion = stack.enter_context(patch.object(h.b.Boundary, 'validate_successor_completion'))
            publication = stack.enter_context(patch.object(h.b.Boundary, 'publish'))
            cleanup = stack.enter_context(patch.object(h, 'stop_owned_work', return_value=[]))
            stack.enter_context(patch.object(h.b, 'containers', return_value=[]))
            # The frozen worker's completed.json is produced by the child, not
            # the recovery wrapper. Avoid manufacturing it in this mock test.
            original_sha = h.b.sha
            stack.enter_context(patch.object(h.b, 'sha', side_effect=lambda path:
                'a' * 64 if str(path) == self.s['replacement']['completion_path'] else original_sha(path)))
            if code:
                with self.assertRaisesRegex(ValueError, 'Frozen worker failed'):
                    h.apply()
                completion.assert_not_called()
                cleanup.assert_called_once()
                self.assertTrue((h.directory / 'failed.json').is_file())
            else:
                h.apply()
                completion.assert_called_once_with(h)
                cleanup.assert_not_called()
                self.assertTrue((h.directory / 'completed.json').is_file())
            publication.assert_called_once_with(h)
            h.b.run_checked.assert_any_call(['/usr/sbin/shutdown', '-h', 'now'])
            self.assertEqual(launch.call_args.args[0], ['/usr/sbin/runuser', '-u', self.account.pw_name,
                '--', *self.s['replacement']['command']])
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertEqual(launch.call_args.kwargs['env']['NXF_VER'], '26.04.6')
        for path, expected in before.items():
            self.assertEqual(digest(path), expected)

    def test_publication_error_does_not_bypass_shutdown(self):
        h = self.recovery
        with self.environment(), patch.object(m.os, 'geteuid', return_value=0), \
             patch.object(m.subprocess, 'Popen', side_effect=RuntimeError('launch failed')), \
             patch.object(h, 'stop_owned_work', return_value=[]), \
             patch.object(h.b.Boundary, 'publish', side_effect=RuntimeError('publication failed')):
            h.validate()
            with self.assertRaisesRegex(RuntimeError, 'publication failed'):
                h.apply()
            h.b.run_checked.assert_any_call(['/usr/sbin/shutdown', '-h', 'now'])


if __name__ == '__main__':
    unittest.main()
