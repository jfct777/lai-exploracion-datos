"""Synthetic local recovery contracts; no cloud access, jobs or genomic inputs."""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


resume = load_module('cloud_resume_tests', ROOT/'bin/r02_coordinator_resume.py')
cloud = load_module('cloud_resume_coordinator_tests', ROOT/'bin/r02_parallel_coordinator.py')
OLD_CODE = '''import json
class GCS:
    def metadata(self, uri, optional=False):
        if result.returncode:
            if optional and any(token in result.stderr for token in ('HTTPError 404',)):
                return None
            raise RuntimeError('failure')
        return json.loads(result.stdout)
def run_after_boundary(spec, directory, boundary, *, activate=True):
    destination = 'original'
    return destination
def scientific_function():
    return 'unchanged'
'''
NEW_CODE = ('from urllib.parse import quote\n' + OLD_CODE.replace(
    "optional and any(token in result.stderr for token in ('HTTPError 404',))",
    'optional and missing_object_error(result.stderr, uri)').replace(
    'activate=True):', 'activate=True, provenance_destination=None):').replace(
    "destination = 'original'", "destination = provenance_destination or 'original'") +
    '\ndef missing_object_error(stderr, uri):\n    return True\n')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CoordinatorResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)/'run'
        self.previous_dir = self.run_dir/resume.PREVIOUS
        self.directory = self.run_dir/resume.RELATIVE
        self.previous_dir.mkdir(parents=True)
        self.new_source = Path(self.temp.name)/'new_coordinator.py'
        self.new_source.write_text(NEW_CODE)
        (self.previous_dir/'coordinator.py').write_text(OLD_CODE)
        (self.previous_dir/'preprocess_count_validation.py').write_text('count fixture')
        (self.previous_dir/'r02_count_recovery.py').write_text('recovery fixture')
        count = {'helper_sha256': resume.sha(self.previous_dir/'r02_count_recovery.py')}
        put(self.previous_dir/'recovery.json', count)
        self.uri = 'gs://bucket/completion.json'
        self.old_spec = {'schema': 'fixture', 'run_dir': str(self.run_dir),
            'coordinator_sha256': resume.sha(self.previous_dir/'coordinator.py'),
            'deadline_utc': (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
            'remote_chromosomes': [{'chromosome': 1, 'completion_uri': self.uri}],
            'operational_amendments': {'worker01': {'fixture': 'unchanged'}}}
        for chrom in (21, 22):
            output = self.run_dir/f'chr{chrom}/result.txt'
            output.parent.mkdir()
            output.write_text('synthetic')
            put(self.run_dir/f'checkpoints/chr{chrom}_complete.json', {
                'chromosome': chrom, 'completed_utc': 'original',
                'outputs': [{'path': str(output), 'bytes': output.stat().st_size, 'sha256': resume.sha(output)}]})
        put(self.run_dir/'checkpoints/chr21_M01_M02_M021.json', {'original': True})
        (self.run_dir/'chr21/trace.tsv').write_text('original trace')
        adoption = {'original_checkpoint_reused': True, 'preprocessing_reexecuted': False,
            'scientific_parameters_changed': False,
            'recovery_manifest_sha256': resume.sha(self.previous_dir/'recovery.json'),
            'checkpoint_sha256': resume.sha(self.run_dir/'checkpoints/chr21_M01_M02_M021.json'),
            'trace_sha256': resume.sha(self.run_dir/'chr21/trace.tsv'),
            'validator_sha256': resume.sha(self.previous_dir/'preprocess_count_validation.py')}
        put(self.previous_dir/'adoption.json', adoption)
        self.old_spec['record_count_repair'] = {
            'validator_sha256': adoption['validator_sha256'],
            'adoption_sha256': resume.sha(self.previous_dir/'adoption.json')}
        put(self.previous_dir/'manifest.json', self.old_spec)
        put(self.previous_dir/'status.json', {'state': 'FAILED',
            'coordinator_manifest_sha256': resume.sha(self.previous_dir/'manifest.json'),
            'error': 'Cannot authenticate cloud object: ' + self.uri + ': '
                     'ERROR: (gcloud.storage.objects.describe) ' + self.uri + ' not found: 404.\n'})
        put(self.previous_dir/'coordinator_activation.json', {'original': True})
        boundary = self.run_dir/'repairs/boundary-v2'
        put(boundary/'activation.sha256.json', {'old': 'seal'})
        put(boundary/'source.sha256.json', {'bin/r02_apply_amendment.py': 'a'*64})
        self.spec_path = resume.prepare(self.run_dir, self.new_source, 99999999)
        self.spec = resume.read(self.spec_path)
        self.old_coordinator = types.SimpleNamespace(validate_manifest=Mock())
        self.coordinator = types.SimpleNamespace(missing_object_error=cloud.missing_object_error,
            timestamp=cloud.timestamp, write_fixed=lambda p,v:put(p,v), run_after_boundary=Mock())
        self.pipeline = types.SimpleNamespace(verify_files=cloud_module_verify_files)
        self.handoff = types.SimpleNamespace(write_json=lambda p,v:put(p,v))
        self.recovery = types.SimpleNamespace(boundary_lock=Mock(return_value=nullcontext()), validate_mount=Mock())
        self.count = types.SimpleNamespace(context=Mock(return_value=(
            {}, {}, self.recovery, self.handoff, self.pipeline, None)))
        self.adapter = types.SimpleNamespace(build_runner=Mock())
        modules = {'_old_cloud_wait_coordinator': self.old_coordinator,
            '_recovered_cloud_wait_coordinator': self.coordinator,
            '_original_count_recovery': self.count, '_recovery_frozen_amendment': self.adapter}
        self.module_patch = patch.object(resume, 'module', side_effect=lambda p,h,n:modules[n])
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def verify(self):
        return resume.validate_spec(self.spec_path, resume.sha(self.spec_path))

    def reseal_current(self, key, value):
        path = self.directory/'manifest.json'
        current = resume.read(path)
        current[key] = value
        put(path, current)
        self.spec['coordinator_manifest_sha256'] = resume.sha(path)
        put(self.spec_path, self.spec)

    def test_prepare_is_idempotent_and_preserves_old_artifacts(self):
        before = {p: p.read_bytes() for p in self.previous_dir.iterdir()}
        self.assertEqual(resume.prepare(self.run_dir, self.new_source, 99999999), self.spec_path)
        self.assertEqual(before, {p: p.read_bytes() for p in self.previous_dir.iterdir()})
        self.verify()

    def test_verify_is_read_only_and_reuses_genuine_completions(self):
        before = {p: p.read_bytes() for p in self.run_dir.rglob('*') if p.is_file()}
        result = resume.run(self.spec_path, resume.sha(self.spec_path))
        self.assertEqual(result['state'], 'VERIFIED_NOT_EXECUTED')
        self.assertEqual(result['local_chromosomes_reused'], [21, 22])
        self.coordinator.run_after_boundary.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.run_dir.rglob('*') if p.is_file()})

    def test_live_pid_is_rejected(self):
        self.spec['previous_controller_pid'] = resume.os.getpid()
        put(self.spec_path, self.spec)
        with self.assertRaisesRegex(ValueError, 'PID is still present'):
            self.verify()

    def test_wrong_manifest_hash_is_rejected_before_loading_modules(self):
        with patch.object(resume, 'module') as imported:
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                resume.validate_spec(self.spec_path, '0'*64)
            imported.assert_not_called()

    def test_changed_deadline_assignments_and_operational_settings_rejected(self):
        for key, value in [('deadline_utc', '2099-01-01T00:00:00+00:00'),
                           ('remote_chromosomes', []), ('operational_amendments', {})]:
            with self.subTest(key=key):
                self.reseal_current(key, value)
                with self.assertRaisesRegex(ValueError, 'deadline, assignments'):
                    self.verify()
                self.reseal_current(key, self.old_spec[key])

    def test_changed_checkpoint_trace_adoption_and_previous_activation_rejected(self):
        for relative in self.spec['protected_sha256']:
            path = self.run_dir/relative
            old = path.read_bytes()
            with self.subTest(relative=relative):
                path.write_bytes(old+b' ')
                with self.assertRaisesRegex(ValueError, 'evidence changed'):
                    self.verify()
                path.write_bytes(old)

    def test_changed_local_result_rejected_even_if_checkpoint_unchanged(self):
        (self.run_dir/'chr21/result.txt').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            self.verify()

    def test_missing_completed_chromosome_never_triggers_recompute(self):
        (self.run_dir/'checkpoints/chr22_complete.json').unlink()
        with self.assertRaises(ValueError):
            self.verify()
        self.coordinator.run_after_boundary.assert_not_called()

    def test_other_failures_cannot_be_adopted_even_if_resealed(self):
        status_path = self.previous_dir/'status.json'
        status = resume.read(status_path)
        status['error'] = status['error'].replace('not found: 404.', 'Permission denied: 403.')
        put(status_path, status)
        self.spec['protected_sha256'][resume.PREVIOUS+'/status.json'] = resume.sha(status_path)
        put(self.spec_path, self.spec)
        with self.assertRaisesRegex(ValueError, 'optional-object 404'):
            self.verify()

    def test_unrelated_code_change_is_rejected(self):
        path = self.directory/'coordinator.py'
        path.write_text(NEW_CODE.replace("return 'unchanged'", "return 'changed'"))
        self.reseal_current('coordinator_sha256', resume.sha(path))
        with self.assertRaisesRegex(ValueError, 'outside the authorized operational delta'):
            self.verify()

    def test_execute_rechecks_under_lock_and_disables_old_activation(self):
        previous = {p: p.read_bytes() for p in self.previous_dir.iterdir()}
        with patch.object(resume.signal, 'signal'), \
                patch.object(resume.signal, 'setitimer', return_value=(0, 0)):
            result = resume.run(self.spec_path, resume.sha(self.spec_path), execute=True)
        self.assertEqual(result, {'state': 'COMPLETE'})
        self.recovery.boundary_lock.assert_called_once_with(self.run_dir/'repairs/boundary-v2/.boundary.lock')
        self.coordinator.run_after_boundary.assert_called_once()
        self.assertEqual(self.coordinator.run_after_boundary.call_args.kwargs,
                         {'activate': False, 'provenance_destination': resume.DESTINATION})
        self.assertEqual(self.adapter.build_runner.call_count, 2)
        self.assertEqual(self.recovery.validate_mount.call_count, 2)
        self.assertEqual(previous, {p: p.read_bytes() for p in self.previous_dir.iterdir()})
        for name in ('resume.py', 'coordinator.py', 'resume.manifest.json', 'resume_activation.json'):
            self.assertEqual((self.directory/'provenance'/name).read_bytes(), (self.directory/name).read_bytes())
        self.assertEqual((self.directory/'provenance/previous_status.json').read_bytes(),
                         (self.previous_dir/'status.json').read_bytes())

    def test_failed_new_run_only_writes_new_failure(self):
        original = (self.previous_dir/'status.json').read_bytes()
        self.coordinator.run_after_boundary.side_effect = RuntimeError('transient service error')
        with patch.object(resume.signal, 'signal'), \
                patch.object(resume.signal, 'setitimer', return_value=(0, 0)):
            with self.assertRaisesRegex(RuntimeError, 'transient service error'):
                resume.run(self.spec_path, resume.sha(self.spec_path), execute=True)
        self.assertEqual(resume.read(self.directory/'status.json')['state'], 'FAILED')
        self.assertEqual((self.previous_dir/'status.json').read_bytes(), original)


def cloud_module_verify_files(records):
    for record in records:
        path = Path(record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes'] or resume.sha(path) != record['sha256']:
            raise ValueError('Checkpoint output changed')


if __name__ == '__main__':
    unittest.main()
