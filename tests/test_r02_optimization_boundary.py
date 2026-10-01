"""Synthetic operation tests: no real VM, Docker operation or process signal."""
import copy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('optimization_boundary', Path(__file__).resolve().parents[1] / 'bin/r02_optimization_boundary.py')
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / 'worker01'
        self.directory = self.run / 'repairs/preprocess-v1'
        self.directory.mkdir(parents=True)
        (self.run / 'source/workflows').mkdir(parents=True)
        (self.run / 'source/workflows/r02_preprocess_autosome.nf').write_text('// fixture\n')
        (self.run / 'checkpoints').mkdir()
        self.folder = self.run / 'chr02'
        self.folder.mkdir()
        self.put(self.folder / 'parameters.json', {'r02_chrom': 2})
        (self.folder / 'runtime.config').write_text('fixture')
        self.put(self.run / 'run.json', dict(run_id='r02-fixture-worker01', free_disk_min_gib=12,
            destination='/home/jose.tantalean/gcs-dnabr/refined/DNABR_QC/presentacion/biologico/R02/parallel/worker01',
            parallel_worker=dict(worker_id='worker01', assigned_chromosomes=[2])))
        self.put(self.run / 'source.sha256.json', {'workflows/r02_preprocess_autosome.nf': b.sha(self.run / 'source/workflows/r02_preprocess_autosome.nf')})
        self.put(self.run / 'frozen.sha256.json', {'run.json': b.sha(self.run / 'run.json')})
        (self.directory / 'r02_optimized_worker.py').write_text('# not executable fixture\n')
        self.put(self.directory / 'manifest.json', {'fixture': True})
        self.put(self.directory / 'source.sha256.json', {'fixture': 'x'})
        command = ['/usr/local/bin/nextflow', '-log', str(self.folder / 'nextflow.log'), '-C', str(self.folder / 'runtime.config'),
                   'run', str(self.run / 'source/workflows/r02_preprocess_autosome.nf'), '-params-file', str(self.folder / 'parameters.json'),
                   '-work-dir', str(self.folder / 'work'), '-ansi-log', 'false', '-with-trace', str(self.folder / 'trace.tsv'), '-resume']
        self.s = dict(schema='r02_optimization_boundary_v1', run_dir=str(self.run), helper_sha256=b.sha(b.__file__),
            worker_run_sha256=b.sha(self.run / 'run.json'), frozen_sha256=b.sha(self.run / 'frozen.sha256.json'),
            source_manifest_sha256=b.sha(self.run / 'source.sha256.json'), uid=1017, gid=1020, user='fixtureuser',
            startup=self.ident(1111), supervisor=self.ident(3333), child=self.ident(4444), ancestors=[self.ident(2222)],
            original_limits=[99999, 99999], chromosome=2, next_failed_stage='chr02_rare_J', boundary_command=command,
            parameters_sha256=b.sha(self.folder / 'parameters.json'), runtime_config_sha256=b.sha(self.folder / 'runtime.config'),
            free_disk_min_gib=12, deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
            instance=dict(id='1234', name='fixture-vm', zone='us-central1-a', project='fixture-project'),
            publication_prefix='gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/R02/parallel/worker01/00_worker_provenance/optimization_boundary/fixture',
            replacement=dict(script_path=str(self.directory / 'r02_optimized_worker.py'), script_sha256=b.sha(self.directory / 'r02_optimized_worker.py'),
                 manifest_path=str(self.directory / 'manifest.json'), manifest_sha256=b.sha(self.directory / 'manifest.json'),
                 completion_path=str(self.directory / 'completed.json')))
        r = self.s['replacement']
        r['command'] = ['/usr/bin/python3', r['script_path'], '--manifest', r['manifest_path'], '--manifest-sha256', r['manifest_sha256'], '--run']
        self.put(self.run / 'status.json', dict(stage='chr02_M01_M02_M021', state='RUNNING', pid=4444))
        self.path = self.directory / 'boundary_spec.json'
        self.boundary = self.load()

    @staticmethod
    def ident(pid):
        return dict(pid=pid, start_ticks=pid + 100, cmdline_sha256=hashlib.sha256(str(pid).encode()).hexdigest())

    @staticmethod
    def put(path, value):
        path.write_text(json.dumps(value))

    def load(self):
        self.put(self.path, self.s)
        h = b.Boundary(self.path, b.sha(self.path))
        with patch.object(b.pwd, 'getpwnam', return_value=types.SimpleNamespace(pw_uid=1017, pw_gid=1020)), \
             patch.object(b, 'metadata_identity', return_value=self.s['instance']):
            return h.validate()

    def info(self, role, state='S'):
        spec = self.s[role] if role != 'ancestor' else self.s['ancestors'][0]
        parent = {'startup': 1000, 'ancestor': 1111, 'supervisor': 2222, 'child': 3333}[role]
        uid = 0 if role in ('startup', 'ancestor') else 1017
        return dict(spec, state=state, ppid=parent, pgid=spec['pid'], sid=spec['pid'], uids=[uid] * 4,
                    gids=[0 if uid == 0 else 1020] * 4, threads=1, cap_eff=0, cap_prm=0)

    def completed(self):
        h = self.boundary
        h.baseline = b.checkpoint_inventory(self.run)
        output = self.folder / 'preprocess/lai_rare'
        output.mkdir(parents=True)
        records = []
        for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv'):
            path = output / ('dnabr.hg38.2723.chr2.rare.minor' + suffix)
            path.write_text('synthetic ' + suffix)
            records.append(dict(path=str(path), bytes=path.stat().st_size, sha256=b.sha(path)))
        checkpoint = self.run / 'checkpoints/chr02_M01_M02_M021.json'
        self.put(checkpoint, dict(returncode=0, command=h.command, outputs=records))
        self.put(self.run / 'status.json', dict(state='FAILED', failed_stage='chr02_rare_J', error='[Errno 11] Resource temporarily unavailable'))
        return checkpoint

    def test_preflight_does_not_create_boundary_or_signal(self):
        self.assertFalse(self.boundary.directory.exists())
        with patch.object(b, 'send') as send:
            self.load()
            send.assert_not_called()

    def test_bad_spec_digest_fails_before_mutation(self):
        with patch.object(b, 'send') as send:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                b.Boundary(self.path, '0' * 64).validate()
            send.assert_not_called()

    def test_changed_frozen_source_is_rejected(self):
        (self.run / 'source/workflows/r02_preprocess_autosome.nf').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.load()

    def test_wrong_replacement_command_is_rejected(self):
        self.s['replacement']['command'].append('--ignore-checks')
        with self.assertRaisesRegex(ValueError, 'fixed and explicit'):
            self.load()

    def test_wrong_original_command_is_rejected(self):
        self.s['boundary_command'][-1] = '--run'
        with self.assertRaisesRegex(ValueError, 'Boundary command'):
            self.load()

    def test_wrong_vm_is_rejected(self):
        with patch.object(b.pwd, 'getpwnam', return_value=types.SimpleNamespace(pw_uid=1017, pw_gid=1020)), \
             patch.object(b, 'metadata_identity', return_value={}):
            with self.assertRaisesRegex(ValueError, 'authorized temporary VM'):
                self.boundary.validate()

    def test_other_publication_prefix_rejected(self):
        self.s['publication_prefix'] = 'gs://other-bucket/all'
        with self.assertRaisesRegex(ValueError, 'Publication escaped'):
            self.load()

    def test_bad_deadline_rejected(self):
        self.s['deadline_utc'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with self.assertRaisesRegex(ValueError, 'Deadline'):
            self.load()

    def test_other_uid_rejected(self):
        self.s['uid'] = 0
        with self.assertRaisesRegex(ValueError, 'Unprivileged'):
            self.load()

    def test_symlink_spec_rejected(self):
        link = self.directory / 'link.json'
        link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, 'nonsymlink'):
            b.Boundary(link, b.sha(self.path)).validate()

    def test_duplicate_json_keys_rejected(self):
        self.path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            b.read(self.path)

    def test_pid_reuse_is_never_signalled(self):
        current = self.info('supervisor')
        current['start_ticks'] += 1
        with patch.object(b, 'process_info', return_value=current), patch.object(os, 'pidfd_open') as opening:
            with self.assertRaisesRegex(ValueError, 'PID reused'):
                b.send(self.s['supervisor'], signal.SIGTERM)
            opening.assert_not_called()

    def test_zombie_is_not_live(self):
        with patch.object(b, 'process_info', return_value=self.info('supervisor', 'Z')):
            self.assertIsNone(b.authenticated(self.s['supervisor']))

    def test_unprivileged_child_may_be_paused_by_storage_helper(self):
        h = self.boundary
        infos = {1111: self.info('startup'), 2222: self.info('ancestor'), 3333: self.info('supervisor'), 4444: self.info('child', 'T')}
        original_read = Path.read_text
        def read_text(path, *args, **kwargs):
            if str(path) == '/proc/3333/task/3333/children':
                return '4444'
            return original_read(path, *args, **kwargs)
        with patch.object(b, 'authenticated', side_effect=lambda s: infos[s['pid']]), patch.object(b, 'nproc', return_value=(99999, 99999)), \
             patch.object(b, 'process_info', return_value=None), patch.object(Path, 'read_text', read_text), patch.object(h, 'health'):
            self.assertEqual(h.eligible()['state'], 'T')
            infos[3333]['cap_eff'] = 1
            with self.assertRaisesRegex(ValueError, 'unprivileged'):
                h.eligible()

    def test_arm_order_and_no_child_signal_or_limit_change(self):
        h = self.boundary
        h.directory.mkdir()
        limits = {3333: (99999, 99999), 4444: (99999, 99999)}
        calls = []
        def limit(pid, values=None):
            old = limits[pid]
            if values is not None:
                calls.append(('limit', pid, tuple(values)))
                limits[pid] = tuple(values)
            return old
        with patch.object(h, 'eligible', return_value=self.info('child', 'T')), patch.object(b, 'nproc', side_effect=limit), \
             patch.object(b, 'stopped'), patch.object(b, 'send', side_effect=lambda spec, sig: calls.append(('signal', spec['pid'], sig)) or True):
            h.arm()
        self.assertEqual(calls, [('signal', 1111, signal.SIGSTOP), ('signal', 3333, signal.SIGSTOP),
                                 ('limit', 3333, (0, 99999)), ('signal', 3333, signal.SIGCONT)])
        self.assertEqual(limits[4444], (99999, 99999))
        self.assertTrue((h.directory / 'armed.json').is_file())

    def test_completion_accepts_genuine_checkpoint_not_expected_failure_alone(self):
        checkpoint = self.completed()
        with patch.object(b, 'authenticated', return_value=None), patch.object(b, 'containers', return_value=[]):
            self.assertEqual(self.boundary.completion(), b.sha(checkpoint))

    def test_completion_missing_output_rejected(self):
        checkpoint = self.completed()
        value = b.read(checkpoint)
        value['outputs'].pop()
        self.put(checkpoint, value)
        with patch.object(b, 'authenticated', return_value=None):
            with self.assertRaisesRegex(ValueError, 'output contract'):
                self.boundary.completion()

    def test_completion_changed_output_rejected(self):
        checkpoint = self.completed()
        Path(b.read(checkpoint)['outputs'][0]['path']).write_text('changed')
        with patch.object(b, 'authenticated', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Output size'):
                self.boundary.completion()

    def test_later_analysis_checkpoint_rejected(self):
        self.completed()
        self.put(self.run / 'checkpoints/chr02_rare_J.json', {'returncode': 0})
        with patch.object(b, 'authenticated', return_value=None):
            with self.assertRaisesRegex(ValueError, 'checkpoint changes'):
                self.boundary.completion()

    def test_preprocessing_failure_must_not_launch_successor(self):
        self.completed()
        self.put(self.run / 'status.json', dict(state='FAILED', failed_stage='chr02_M01_M02_M021', error='exit 1'))
        with patch.object(b, 'authenticated', return_value=None):
            with self.assertRaisesRegex(ValueError, 'expected next-stage EAGAIN'):
                self.boundary.completion()

    def test_running_original_rejected(self):
        self.completed()
        with patch.object(b, 'authenticated', return_value=self.info('supervisor')):
            with self.assertRaisesRegex(ValueError, 'must exit'):
                self.boundary.completion()

    def test_container_still_running_rejected(self):
        self.completed()
        with patch.object(b, 'authenticated', return_value=None), patch.object(b, 'containers', return_value=['a' * 64]):
            with self.assertRaisesRegex(ValueError, 'containers still active'):
                self.boundary.completion()

    def test_second_applicator_cannot_acquire_same_lock(self):
        h = self.boundary
        h.directory.mkdir()
        with (h.run / '.optimization_boundary.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(os, 'geteuid', return_value=0), patch.object(h, 'arm') as arm:
                with self.assertRaises(BlockingIOError):
                    h.apply()
                arm.assert_not_called()

    def test_interrupted_application_is_not_restarted(self):
        h = self.boundary
        h.directory.mkdir()
        self.put(h.directory / 'intent.json', {})
        with patch.object(os, 'geteuid', return_value=0), patch.object(h, 'arm') as arm:
            with self.assertRaisesRegex(ValueError, 'manual reconciliation'):
                h.apply()
            arm.assert_not_called()

    def test_disk_guard_and_deadline(self):
        with patch.object(b.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=11 * 1024**3)):
            with self.assertRaisesRegex(ValueError, '12 GiB'):
                self.boundary.health()
        self.boundary.deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, 'deadline'):
            self.boundary.health()

    def test_event_failure_does_not_bypass_authenticated_cleanup(self):
        h = self.boundary
        with patch.object(h, 'event', side_effect=OSError('disk full')), patch.object(b, 'descendants', return_value=[]), \
             patch.object(b, 'send', return_value=False) as send, patch.object(b, 'stop_containers') as stop:
            h.emergency(RuntimeError('fixture error'))
        stop.assert_called_once_with(h.run_id)
        self.assertIn((h.s['startup'], signal.SIGKILL), [x.args for x in send.call_args_list])

    def test_failure_stops_vm_without_deleting_disk(self):
        h = self.boundary
        def fail():
            h.interfered = True
            raise RuntimeError('fixture failure')
        with patch.object(os, 'geteuid', return_value=0), patch.object(h, 'arm', side_effect=fail), \
             patch.object(h, 'emergency') as emergency, patch.object(h, 'publish', side_effect=OSError('publication down')), \
             patch.object(b, 'metadata_identity', return_value=h.s['instance']), patch.object(b, 'run_checked') as command:
            with self.assertRaisesRegex(OSError, 'publication down'):
                h.apply()
        emergency.assert_called_once()
        command.assert_called_once_with(['/usr/sbin/shutdown', '-h', 'now'])

    def test_runuser_continued_without_resuming_startup_or_nextflow(self):
        h = self.boundary
        h.child_limits = [99999, 99999]
        infos = {1111: self.info('startup', 'T'), 2222: self.info('ancestor', 'T'),
                 3333: self.info('supervisor'), 4444: self.info('child', 'T')}
        calls = []
        def send(spec, sig):
            calls.append((spec['pid'], sig))
            infos[spec['pid']]['state'] = 'S'
            return True
        with patch.object(b, 'authenticated', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'authenticated_runuser', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'nproc', side_effect=lambda pid: (0, 99999) if pid == 3333 else (99999, 99999)), \
             patch.object(b, 'send', side_effect=send), patch.object(h, 'event'):
            self.assertTrue(h.resume_stopped_runuser())
            self.assertFalse(h.resume_stopped_runuser())
        self.assertEqual(calls, [(2222, signal.SIGCONT)])
        self.assertEqual(infos[1111]['state'], 'T')
        self.assertEqual(infos[4444]['state'], 'T')

    def test_runuser_not_continued_if_supervisor_barrier_missing(self):
        h = self.boundary
        h.child_limits = [99999, 99999]
        infos = {1111: self.info('startup', 'T'), 2222: self.info('ancestor', 'T'),
                 3333: self.info('supervisor'), 4444: self.info('child')}
        with patch.object(b, 'authenticated', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'authenticated_runuser', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'nproc', return_value=(99999, 99999)), patch.object(b, 'send') as send:
            with self.assertRaisesRegex(ValueError, 'fork barrier'):
                h.resume_stopped_runuser()
            send.assert_not_called()

    def test_runuser_not_continued_if_startup_resumed(self):
        h = self.boundary
        infos = {1111: self.info('startup'), 2222: self.info('ancestor', 'T')}
        with patch.object(b, 'authenticated', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'authenticated_runuser', side_effect=lambda s: infos[s['pid']]), patch.object(b, 'send') as send:
            with self.assertRaisesRegex(ValueError, 'must remain stopped'):
                h.resume_stopped_runuser()
            send.assert_not_called()

    def test_runuser_after_supervisor_exit_requires_genuine_boundary(self):
        h = self.boundary
        self.completed()
        infos = {1111: self.info('startup', 'T'), 2222: self.info('ancestor', 'T'), 3333: None, 4444: None}
        with patch.object(b, 'authenticated', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'authenticated_runuser', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'containers', return_value=[]), patch.object(h, 'event'), patch.object(b, 'send', return_value=True) as send:
            self.assertTrue(h.resume_stopped_runuser())
            send.assert_called_once_with(self.s['ancestors'][0], signal.SIGCONT)
        self.put(self.run / 'status.json', dict(state='FAILED', failed_stage='chr02_M01_M02_M021', error='exit 1'))
        with patch.object(b, 'authenticated', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'authenticated_runuser', side_effect=lambda s: infos[s['pid']]), patch.object(b, 'send') as send:
            with self.assertRaisesRegex(ValueError, 'expected next-stage EAGAIN'):
                h.resume_stopped_runuser()
            send.assert_not_called()

    def test_runuser_not_continued_if_nextflow_limits_changed(self):
        h = self.boundary
        h.child_limits = [99999, 99999]
        infos = {1111: self.info('startup', 'T'), 2222: self.info('ancestor', 'T'),
                 3333: self.info('supervisor'), 4444: self.info('child')}
        with patch.object(b, 'authenticated', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'authenticated_runuser', side_effect=lambda s: infos[s['pid']]), \
             patch.object(b, 'nproc', return_value=(0, 99999)), patch.object(b, 'send') as send:
            with self.assertRaisesRegex(ValueError, 'Nextflow identity or limits'):
                h.resume_stopped_runuser()
            send.assert_not_called()

    def test_build_spec_does_not_signal(self):
        inventory = dict(run=str(self.run), worker='worker01', config_sha256=self.s['worker_run_sha256'],
            frozen_sha256=self.s['frozen_sha256'], source_manifest_sha256=self.s['source_manifest_sha256'],
            status=dict(stage='chr02_M01_M02_M021'), supervisor=self.s['supervisor'],
            child=dict(self.s['child'], args=['java', '-jar', '/nextflow.jar', *self.s['boundary_command'][1:]]),
            ancestors=[dict(self.s['ancestors'][0], args=['runuser', '-u', 'jose.tantalean']),
                       dict(self.s['startup'], args=['bash', '/tmp/metadata-scripts123/startup-script'])],
            original_limits=[99999, 99999], uid=1017, gid=1020, parameters_sha256=self.s['parameters_sha256'],
            runtime_config_sha256=self.s['runtime_config_sha256'],
            instance={'instance/id': '1234', 'instance/name': 'fixture-vm', 'instance/zone': 'projects/p/zones/us-central1-a',
                      'project/project-id': 'fixture-project'})
        with patch.object(b, 'send') as send:
            result = b.build_spec(inventory, self.directory / 'manifest.json', self.s['replacement']['manifest_sha256'],
                                  b.__file__, self.s['deadline_utc'])
            send.assert_not_called()
        self.assertEqual(result['boundary_command'], self.s['boundary_command'])
        self.assertEqual(result['startup'], self.s['startup'])
        self.assertEqual(result['ancestors'], self.s['ancestors'])
        self.assertEqual(result['replacement']['completion_path'], str(self.directory / 'completed.json'))


if __name__ == '__main__':
    unittest.main()
