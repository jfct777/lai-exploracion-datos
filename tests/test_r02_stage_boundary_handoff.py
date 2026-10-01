"""Local fixtures only: these tests never signal a real R02 process or use Docker."""
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_boundary_handoff', ROOT / 'bin/r02_stage_boundary_handoff.py')
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / 'r02-test'
        self.amendment = self.run / 'amendments/v2'
        self.amendment.mkdir(parents=True)
        (self.run / 'checkpoints').mkdir()
        (self.run / 'chr21').mkdir()
        (self.run / 'source/workflows').mkdir(parents=True)
        self.put(self.run / 'run.json', {'run_id': 'r02-test', 'free_disk_min_gib': 12})
        self.put(self.run / 'chr21/parameters.json', {'chromosome': 21})
        (self.run / 'chr21/runtime.config').write_text('executor = local\n')
        (self.run / 'source/workflows/r02_preprocess_autosome.nf').write_text('// immutable original\n')
        self.put(self.run / 'source.sha256.json', {
            'workflows/r02_preprocess_autosome.nf': handoff.sha(self.run / 'source/workflows/r02_preprocess_autosome.nf')})
        self.put(self.run / 'frozen.sha256.json', {
            name: handoff.sha(self.run / name) for name in ('run.json', 'source.sha256.json')})
        (self.amendment / 'source/bin').mkdir(parents=True)
        for name in ('r02_stage_boundary_handoff.py', 'r02_apply_amendment.py'):
            (self.amendment / 'source/bin' / name).write_text('# synthetic, not executable\n')
        self.put(self.amendment / 'source.sha256.json', {
            'bin/' + name: handoff.sha(self.amendment / 'source/bin' / name)
            for name in ('r02_stage_boundary_handoff.py', 'r02_apply_amendment.py')})
        self.request = dict(schema_version=1, run_dir=str(self.run), amendment_dir=str(self.amendment),
                            supervisor=dict(pid=12345, start_ticks=123, cmdline_sha256='a' * 64),
                            child=dict(pid=23456, start_ticks=456, cmdline_sha256='b' * 64),
                            boundary_stage='chr21_M01_M02_M021', free_disk_min_gib=12,
                            original_limits=[128172, 128172],
                            amendment_source_manifest_sha256=handoff.sha(self.amendment / 'source.sha256.json'),
                            amendment_template=dict(schema_version=1, run_dir=str(self.run),
                                original_frozen_sha256=handoff.sha(self.run / 'frozen.sha256.json'),
                                original_source_manifest_sha256=handoff.sha(self.run / 'source.sha256.json'),
                                boundary=dict(chromosome=21, stage='chr21_M01_M02_M021',
                                    parameters_sha256=handoff.sha(self.run / 'chr21/parameters.json'),
                                    runtime_config_sha256=handoff.sha(self.run / 'chr21/runtime.config')),
                                overrides={'analysis_protocol_version': 2}))
        self.put(self.amendment / 'request.json', self.request)
        self.digest = handoff.sha(self.amendment / 'request.json')
        self.h = handoff.Handoff(self.amendment / 'request.json', self.digest).validate()
        self.put(self.run / 'checkpoints/chr22_complete.json', {'historical_fixture': True})
        self.put(self.run / 'status.json', dict(stage='chr21_M01_M02_M021', state='RUNNING', pid=23456))

    @staticmethod
    def put(path, value):
        path.write_text(json.dumps(value))

    def parent(self, state='S'):
        return dict(self.request['supervisor'], state=state, ppid=1, pgid=12345, sid=12345,
                    uids=[os.getuid()] * 4, gids=[os.getgid()] * 4, threads=1, cap_eff=0, cap_prm=0)

    def child(self):
        return dict(self.request['child'], state='S', ppid=12345, pgid=23456, sid=23456,
                    uids=[os.getuid()] * 4, gids=[os.getgid()] * 4, threads=6, cap_eff=0, cap_prm=0)

    def armed(self):
        value = dict(request_sha256=self.digest, original_limits=[128172, 128172],
                     child_limits=[128172, 128172], checkpoint_baseline=handoff.checkpoint_inventory(self.run))
        self.h.armed = value
        return value

    def completed(self):
        self.armed()
        folder = self.run / 'chr21'
        rare = folder / 'preprocess/lai_rare'
        rare.mkdir(parents=True)
        outputs = []
        for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv'):
            path = rare / ('dnabr.hg38.2723.chr21.rare.minor' + suffix)
            path.write_text('synthetic output ' + suffix)
            outputs.append(dict(path=str(path), bytes=path.stat().st_size, sha256=handoff.sha(path)))
        command = ['/usr/local/bin/nextflow', '-log', str(folder / 'nextflow.log'), '-C', str(folder / 'runtime.config'),
                   'run', str(self.run / 'source/workflows/r02_preprocess_autosome.nf'), '-params-file',
                   str(folder / 'parameters.json'), '-work-dir', str(folder / 'work'), '-ansi-log', 'false',
                   '-with-trace', str(folder / 'trace.tsv'), '-resume']
        cp = self.run / 'checkpoints/chr21_M01_M02_M021.json'
        self.put(cp, dict(returncode=0, command=command, outputs=outputs))
        self.put(self.run / 'status.json', dict(state='FAILED', stage='FAILED', failed_stage='chr21_rare_J',
                                               error='[Errno 11] Resource temporarily unavailable'))
        return cp

    def test_request_hash_rejected_before_signals(self):
        with patch.object(handoff, 'send') as send:
            with self.assertRaisesRegex(ValueError, 'Request hash mismatch'):
                handoff.Handoff(self.amendment / 'request.json', '0' * 64).validate()
            send.assert_not_called()

    def test_snapshot_mutation_rejected_before_signals(self):
        (self.amendment / 'source/bin/r02_apply_amendment.py').write_text('changed\n')
        with patch.object(handoff, 'send') as send:
            with self.assertRaisesRegex(ValueError, 'Manifest hash mismatch'):
                self.h.validate()
            send.assert_not_called()

    def test_original_parameters_cannot_change(self):
        self.put(self.run / 'chr21/parameters.json', {'chromosome': 20})
        with self.assertRaisesRegex(ValueError, 'preprocessing parameters changed'):
            self.h.validate()

    def test_atomic_immutable_records_cannot_be_replaced(self):
        path = self.amendment / 'record.json'
        handoff.write_json(path, {'a': 1}, once=True)
        handoff.write_json(path, {'a': 1}, once=True)
        with self.assertRaisesRegex(ValueError, 'Immutable handoff record differs'):
            handoff.write_json(path, {'a': 2}, once=True)
        self.assertEqual(handoff.read_json(path), {'a': 1})

    def test_arm_changes_only_parent_limit_and_continues_after_record(self):
        limits = {12345: (128172, 128172), 23456: (128172, 128172)}
        calls = []
        def limit(pid, value=None):
            old = limits[pid]
            if value is not None:
                calls.append(('limit', pid, tuple(value)))
                limits[pid] = tuple(value)
            return old if value is not None else limits[pid]
        def send(spec, sig):
            calls.append(('signal', spec['pid'], sig))
            if sig == signal.SIGCONT:
                self.assertTrue((self.amendment / 'armed.json').exists())
            return True
        with patch.object(self.h, 'eligible'), patch.object(handoff, 'nproc', side_effect=limit), \
                patch.object(handoff, 'send', side_effect=send), patch.object(handoff, 'wait_stopped'), \
                patch.object(handoff, 'authenticated', return_value=self.parent()):
            self.h.arm()
        self.assertEqual(limits[12345], (0, 128172))
        self.assertEqual(limits[23456], (128172, 128172))
        self.assertEqual(calls, [('signal', 12345, signal.SIGSTOP), ('limit', 12345, (0, 128172)),
                                 ('signal', 12345, signal.SIGCONT)])
        self.assertEqual(set(self.h.armed['checkpoint_baseline']), {'chr22_complete.json'})

    def test_restarted_armed_watcher_resumes_only_protected_parent(self):
        self.put(self.amendment / 'armed.json', self.armed())
        with patch.object(handoff, 'authenticated', return_value=self.parent('T')), \
                patch.object(handoff, 'nproc', return_value=(0, 128172)), patch.object(handoff, 'send') as send:
            self.h.arm()
            send.assert_called_once_with(self.request['supervisor'], signal.SIGCONT)
        with patch.object(handoff, 'authenticated', return_value=self.parent('T')), \
                patch.object(handoff, 'nproc', return_value=(128172, 128172)), patch.object(handoff, 'send') as send:
            with self.assertRaisesRegex(RuntimeError, 'removed barrier'):
                self.h.arm()
            send.assert_not_called()

    def test_completion_rejects_later_science_checkpoints(self):
        self.completed()
        self.put(self.run / 'checkpoints/chr21_rare_J.json', {'returncode': 0})
        with patch.object(handoff, 'authenticated', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected new'):
                self.h.validate_completion()

    def test_completion_rejects_changed_output_or_wrong_exit(self):
        cp = self.completed()
        record = handoff.read_json(cp)
        Path(record['outputs'][0]['path']).write_text('changed')
        with patch.object(handoff, 'authenticated', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Boundary output changed'):
                self.h.validate_completion()

    def test_completion_rejects_unexpected_parent_failure(self):
        self.completed()
        self.put(self.run / 'status.json', dict(state='FAILED', failed_stage='chr21_rare_J', error='Some other failure'))
        with patch.object(handoff, 'authenticated', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'expected blocked next fork'):
                self.h.validate_completion()

    def test_completion_rejects_live_containers(self):
        self.completed()
        with patch.object(handoff, 'authenticated', return_value=None), \
                patch.object(handoff, 'containers', return_value=['c' * 64]):
            with self.assertRaisesRegex(RuntimeError, 'containers remain active'):
                self.h.validate_completion()

    def test_success_seals_same_run_and_execs_only_frozen_adapter(self):
        cp = self.completed()
        with patch.object(handoff, 'authenticated', return_value=None), \
                patch.object(handoff, 'containers', return_value=[]), patch.object(os, 'execv') as execute:
            self.h.activate()
        amendment = handoff.read_json(self.amendment / 'amendment.json')
        self.assertEqual(amendment['boundary']['checkpoint_sha256'], handoff.sha(cp))
        self.assertEqual(amendment['run_dir'], str(self.run))
        handoff.verify_manifest(self.amendment, self.amendment / 'frozen.sha256.json')
        execute.assert_called_once_with(sys.executable, [sys.executable,
            str(self.amendment / 'source/bin/r02_apply_amendment.py'), '--run-dir', str(self.run),
            '--amendment-dir', str(self.amendment), '--run'])

    def test_emergency_never_restores_then_continues_unterminated_parent(self):
        self.armed()
        calls = []
        def auth(spec):
            return self.parent('T') if spec['pid'] == 12345 else self.child()
        with patch.object(handoff, 'authenticated', side_effect=auth), \
                patch.object(handoff, 'send', side_effect=lambda spec, sig: (calls.append(('signal', spec['pid'], sig)), True)[1]), \
                patch.object(handoff, 'wait_stopped'), \
                patch.object(handoff, 'nproc', side_effect=lambda pid, limits: calls.append(('limit', pid, limits))), \
                patch.object(handoff, 'stop_containers', side_effect=lambda rid: calls.append(('containers', rid))), \
                patch.object(os, 'killpg', side_effect=lambda pid, sig: calls.append(('group', pid, sig))):
            with self.assertRaisesRegex(RuntimeError, 'disk'):
                self.h.emergency('disk below guard')
        self.assertEqual(calls, [('signal', 12345, signal.SIGSTOP), ('containers', 'r02-test'),
            ('group', 23456, signal.SIGTERM), ('limit', 12345, [128172, 128172]),
            ('signal', 12345, signal.SIGTERM), ('signal', 12345, signal.SIGCONT)])

    def test_failed_installation_before_limit_change_resumes_unarmed_current_work(self):
        calls = []
        def limit(pid, value=None):
            if value is not None:
                raise PermissionError('synthetic prlimit denied')
            return (128172, 128172)
        def auth(spec):
            return self.parent('T') if spec['pid'] == 12345 else self.child()
        with patch.object(self.h, 'eligible'), patch.object(handoff, 'nproc', side_effect=limit), \
                patch.object(handoff, 'send', side_effect=lambda spec, sig: (calls.append(sig), True)[1]), \
                patch.object(handoff, 'wait_stopped'), patch.object(handoff, 'authenticated', side_effect=auth), \
                patch.object(handoff, 'stop_containers') as stop:
            with self.assertRaisesRegex(PermissionError, 'synthetic prlimit'):
                self.h.arm()
            stop.assert_not_called()
        self.assertEqual(calls, [signal.SIGSTOP, signal.SIGCONT])
        self.assertFalse(self.h.interference_started)
        self.assertEqual(handoff.read_json(self.amendment / 'boundary_status.json')['state'], 'NOT_ARMED')

    def test_failure_after_limit_change_terminates_instead_of_stranding_parent(self):
        limits = {12345: (128172, 128172), 23456: (128172, 128172)}
        signals = []
        original_write = handoff.write_json
        def write(path, value, **kwargs):
            if Path(path).name == 'armed.json':
                raise OSError('synthetic armed record failure')
            return original_write(path, value, **kwargs)
        def limit(pid, value=None):
            old = limits[pid]
            if value is not None:
                limits[pid] = tuple(value)
            return old if value is not None else limits[pid]
        def auth(spec):
            return self.parent('T') if spec['pid'] == 12345 else self.child()
        with patch.object(self.h, 'eligible'), patch.object(handoff, 'write_json', side_effect=write), \
                patch.object(handoff, 'nproc', side_effect=limit), \
                patch.object(handoff, 'send', side_effect=lambda spec, sig: (signals.append(sig), True)[1]), \
                patch.object(handoff, 'wait_stopped'), patch.object(handoff, 'authenticated', side_effect=auth), \
                patch.object(handoff, 'stop_containers'), patch.object(os, 'killpg') as group:
            with self.assertRaisesRegex(RuntimeError, 'durable boundary'):
                self.h.arm()
            group.assert_called_once_with(23456, signal.SIGTERM)
        self.assertEqual(signals[-2:], [signal.SIGTERM, signal.SIGCONT])
        self.assertEqual(limits[12345], (128172, 128172))
        self.assertTrue(self.h.aborted)

    def test_unexpected_monitor_error_invokes_authenticated_emergency(self):
        self.h.interference_started = True
        with patch.object(self.h, 'emergency', side_effect=RuntimeError('aborted')) as emergency:
            self.h.fail_closed(OSError('synthetic monitor error'))
        emergency.assert_called_once_with('Unexpected handoff failure: synthetic monitor error')
        self.assertEqual(handoff.read_json(self.amendment / 'boundary_status.json')['state'], 'FAILED_CLOSED')

    def test_pid_reuse_never_signals(self):
        changed = dict(self.parent(), start_ticks=999)
        with patch.object(handoff, 'process_info', return_value=changed), patch.object(os, 'pidfd_open') as opened:
            with self.assertRaisesRegex(RuntimeError, 'PID reuse'):
                handoff.send(self.request['supervisor'], signal.SIGSTOP)
            opened.assert_not_called()


class LinuxBarrierFixture(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and os.getuid() != 0, 'Requires unprivileged Linux RLIMIT_NPROC')
    def test_existing_child_forks_after_parent_barrier_and_checkpoint_is_written(self):
        child_code = '''import json, resource, subprocess, sys
sys.stdin.readline()
result = subprocess.check_output([sys.executable, '-c', 'print("GRANDCHILD_OK")'], text=True).strip()
print(json.dumps({"limits": resource.getrlimit(resource.RLIMIT_NPROC), "grandchild": result}), flush=True)
'''
        parent_code = '''import errno, json, os, resource, subprocess, sys
p = subprocess.Popen([sys.executable, '-c', sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
print(json.dumps({"child": p.pid}), flush=True)
sys.stdin.readline()
stdout, _ = p.communicate('go\\n', timeout=10)
assert p.returncode == 0
os.write(int(sys.argv[2]), stdout.encode()); os.fsync(int(sys.argv[2]))
try:
    forbidden = subprocess.Popen([sys.executable, '-c', 'print("FORBIDDEN")'])
except OSError as error:
    assert error.errno == errno.EAGAIN
    print('NEXT_FORK_BLOCKED', flush=True)
else:
    forbidden.wait()
    raise AssertionError('Old next stage was not blocked')
'''
        with tempfile.TemporaryFile() as checkpoint:
            parent = subprocess.Popen([sys.executable, '-c', parent_code, child_code, str(checkpoint.fileno())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                pass_fds=(checkpoint.fileno(),))
            original = None
            try:
                child = json.loads(parent.stdout.readline())['child']
                original = resource.prlimit(parent.pid, resource.RLIMIT_NPROC)
                child_original = resource.prlimit(child, resource.RLIMIT_NPROC)
                info = handoff.process_info(parent.pid)
                spec = {key: info[key] for key in ('pid', 'start_ticks', 'cmdline_sha256')}
                handoff.send(spec, signal.SIGSTOP)
                handoff.wait_stopped(spec)
                resource.prlimit(parent.pid, resource.RLIMIT_NPROC, (0, original[1]))
                self.assertEqual(resource.prlimit(child, resource.RLIMIT_NPROC), child_original)
                handoff.send(spec, signal.SIGCONT)
                stdout, stderr = parent.communicate('continue\n', timeout=15)
                self.assertEqual(parent.returncode, 0, stderr)
                self.assertEqual(stdout.strip(), 'NEXT_FORK_BLOCKED')
                checkpoint.seek(0)
                saved = json.load(checkpoint)
                self.assertEqual(saved['grandchild'], 'GRANDCHILD_OK')
                self.assertEqual(tuple(saved['limits']), child_original)
            finally:
                if parent.poll() is None:
                    if original is not None:
                        resource.prlimit(parent.pid, resource.RLIMIT_NPROC, original)
                    parent.terminate()
                    os.kill(parent.pid, signal.SIGCONT)
                    parent.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
