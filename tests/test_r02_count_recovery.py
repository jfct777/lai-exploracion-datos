"""Synthetic chr21 adoption contracts: no cloud, containers or process signals.

Orchestration dependencies are mocked; real checkpoint hashes, trace validation,
execution locks and Runner.command's cached path are exercised on tiny files.
"""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


r = load('count_recovery_tested', ROOT/'bin/r02_count_recovery.py')
local = load('count_recovery_local_fixture', ROOT/'bin/r02_local_preprocess_resume.py')
original = load('count_recovery_pipeline_fixture', ROOT/'bin/r02_autosome_pipeline.py')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CompletedCountRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)/'run'
        self.directory = self.run/'repairs/preprocess-v2/coordinator'
        self.directory.mkdir(parents=True)
        self.previous = self.run/'repairs/local-preprocess-v1'
        self.boundary = self.run/'repairs/boundary-v2'
        self.path = self.directory/'recovery.json'
        self.controller = dict(pid=987654321, start_ticks=123, cmdline_sha256='a'*64)
        self.old = dict(wrapper_sha256='b'*64)
        put(self.previous/'manifest.json', self.old)
        self.failure = self.previous/'status.json'
        put(self.failure, dict(state='FAILED', error=
            'chr21: source index count does not match completed sequential M01 annotation'))
        previous_controller = self.run/'repairs/preprocess-v1/controller-boundary/manifest.json'
        put(previous_controller, dict(supervisor=self.controller))
        put(self.boundary/'source.sha256.json', {'bin/r02_apply_amendment.py': 'c'*64})
        put(self.boundary/'request.json', dict(amendment_template=dict(
            schema_version=1, run_dir=str(self.run), boundary=dict(chromosome=21), overrides={'fixture': True})))
        self.spec = dict(schema=r.SCHEMA, run_dir=str(self.run), helper_sha256=r.sha(r.__file__),
            validator_sha256='d'*64, recovery_manifest_sha256=r.sha(self.previous/'manifest.json'),
            failed_status_sha256=r.sha(self.failure), previous_controller=self.controller,
            previous_controller_manifest_sha256=r.sha(previous_controller))
        put(self.path, self.spec)
        self.command = ['nextflow', 'authenticated-original-workflow', '-resume']
        self.outputs = []
        for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv'):
            output = self.run/'chr21/preprocess/lai_rare'/('dnabr.hg38.2723.chr21.rare.minor'+suffix)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('synthetic '+suffix)
            self.outputs.append(dict(path=str(output), bytes=output.stat().st_size, sha256=r.sha(output)))
        self.checkpoint = self.run/'checkpoints/chr21_M01_M02_M021.json'
        put(self.checkpoint, dict(command=self.command, returncode=0,
            completed_utc='2026-10-01T19:50:00+00:00', outputs=self.outputs))
        self.trace = self.run/'chr21/trace.tsv'
        self.trace.write_text('name\tstatus\texit\thash\n'+
            f'{local.M01}\tCACHED\t0\tab/123456\n'+
            f'{local.M02}\tCOMPLETED\t0\tcd/123456\n'+
            f'{local.M021}\tCOMPLETED\t0\tef/123456\n')
        self.audit = dict(raw_record_count_source='sequential_m01_with_missing_tbi_statistics',
                          raw_records=10, m01_records=10, fixture=True)
        self.helper = types.SimpleNamespace(validate_raw_record_count=Mock(return_value=self.audit))
        self.handoff = types.SimpleNamespace(authenticated=Mock(return_value=None),
            write_json=lambda path, value: put(path, value))
        self.runner = types.SimpleNamespace(run=self.run)
        self.runner.preprocess = Mock(side_effect=self.cached_preprocess)
        self.pipeline = types.SimpleNamespace(Runner=Mock(return_value=self.runner),
            verify_files=original.verify_files, execution_lock=original.execution_lock,
            validate_raw_record_count='original-validator')
        self.recovery = types.SimpleNamespace(validate_idle=Mock(), validate_mount=Mock(),
            check_trace=local.check_trace, boundary_lock=local.boundary_lock)
        self.recovery.validate_spec = Mock(return_value=(self.old, {}, None, self.handoff, self.pipeline, []))
        self.adapter = types.SimpleNamespace(expected_boundary_command=lambda *_: self.command)
        self.coordinator = types.SimpleNamespace(validate_manifest=Mock(), timestamp=lambda _: r.time.time()+100,
                                                  run_after_boundary=Mock())
        def dependency(path, expected, name):
            return {'_original_recovery': self.recovery, '_count_validator': self.helper,
                    '_original_amendment': self.adapter, '_count_coordinator': self.coordinator}[name]
        self.module_patch = patch.object(r, 'module', side_effect=dependency)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def cached_preprocess(self, chrom):
        self.assertEqual(chrom, 21)
        original.Runner.command(self.runner, 'chr21_M01_M02_M021', self.command,
                                expected_outputs=[Path(item['path']) for item in self.outputs])
        return self.pipeline.validate_raw_record_count(self.run/'chr21', 21)

    def adopt(self, execute=False):
        return r.adopt(self.path, r.sha(self.path), execute=execute)

    def authenticate_failure(self, **changes):
        failure = r.read(self.failure)
        failure.update(changes)
        put(self.failure, failure)
        self.spec['failed_status_sha256'] = r.sha(self.failure)
        put(self.path, self.spec)

    def prepare_coordinator(self):
        self.adopt(execute=True)
        self.coordinator_spec = dict(coordinator_sha256='e'*64, deadline_utc='future',
            record_count_repair=dict(validator_sha256=self.spec['validator_sha256'],
                                    adoption_sha256=r.sha(self.directory/'adoption.json')))
        put(self.directory/'manifest.json', self.coordinator_spec)
        self.coordinator.validate_manifest.return_value = self.coordinator_spec

    def test_preflight_is_read_only_and_never_invokes_runner(self):
        before = {str(p): r.sha(p) for p in self.run.rglob('*') if p.is_file()}
        result = self.adopt()
        self.assertEqual(result['state'], 'VERIFIED_NOT_ADOPTED')
        self.assertFalse(result['preprocessing_reexecuted'])
        self.assertEqual(before, {str(p): r.sha(p) for p in self.run.rglob('*') if p.is_file()})
        self.pipeline.Runner.assert_not_called()

    def test_adoption_uses_cached_command_without_launch_and_preserves_history(self):
        original_paths = [self.failure, self.trace, self.checkpoint, self.boundary/'request.json',
                          self.boundary/'source.sha256.json', self.previous/'manifest.json']
        before = {path: r.sha(path) for path in original_paths}
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('Must not launch')) as launch:
            result = self.adopt(execute=True)
        launch.assert_not_called()
        self.assertEqual(result['state'], 'GENUINE_COMPLETION_ADOPTED')
        self.assertEqual(before, {path: r.sha(path) for path in original_paths})
        self.assertIs(self.pipeline.validate_raw_record_count, self.helper.validate_raw_record_count)
        self.assertFalse((self.previous/'completed.json').exists())
        self.assertFalse((self.boundary/'preprocess_storage_recovery.json').exists())
        seal = r.read(self.boundary/'frozen.sha256.json')
        self.assertEqual(set(seal), {'amendment.json', 'source.sha256.json', 'request.json',
                                     'preprocess_count_recovery.json'})
        self.assertTrue(all(r.sha(self.boundary/name) == digest for name, digest in seal.items()))
        self.assertEqual(r.read(self.boundary/'amendment.json')['boundary']['checkpoint_sha256'], r.sha(self.checkpoint))

    def test_adoption_replay_is_idempotent_and_still_does_not_launch(self):
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('Must not launch')):
            first = self.adopt(execute=True)
            before = r.sha(self.directory/'adoption.json')
            second = self.adopt(execute=True)
        self.assertEqual(first, second)
        self.assertEqual(r.sha(self.directory/'adoption.json'), before)

    def test_manifest_symlink_alias_and_relative_path_are_rejected(self):
        alias = self.run/'alias.json'
        alias.symlink_to(self.path)
        for path in (alias, Path('relative/manifest.json')):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'canonical absolute'):
                r.context(path, r.sha(self.path))

    def test_wrong_manifest_hash_rejected_before_dependency_loading(self):
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            r.context(self.path, '0'*64)
        self.recovery.validate_spec.assert_not_called()

    def test_generic_or_prefixed_failure_is_never_adopted(self):
        for message in ('exit 1', 'prefix: chr21: source index count does not match completed sequential M01 annotation'):
            self.authenticate_failure(error=message)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, 'exact post-checkpoint'):
                self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_live_previous_controller_is_rejected(self):
        self.handoff.authenticated.return_value = self.controller
        with self.assertRaisesRegex(ValueError, 'still active'):
            self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_preflight_does_not_treat_index_agreement_as_missing_statistics(self):
        self.helper.validate_raw_record_count.return_value = dict(raw_record_count_source='source_index_and_sequential_m01')
        with self.assertRaisesRegex(ValueError, 'demonstrated absent'):
            self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_checkpoint_exit_zero_alone_is_not_completion(self):
        original_record = r.read(self.checkpoint)
        bad_records = [dict(original_record, returncode=True), dict(original_record, command=['other']),
                       dict(original_record, completed_utc=''), dict(original_record, outputs=self.outputs[:-1])]
        for record in bad_records:
            put(self.checkpoint, record)
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, 'original preprocessing'):
                self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_changed_checkpoint_output_rejected(self):
        Path(self.outputs[0]['path']).write_text('corrupt')
        with self.assertRaises(ValueError):
            self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_recomputed_m01_or_failed_downstream_trace_rejected(self):
        original_trace = self.trace.read_text()
        for trace in (original_trace.replace('CACHED', 'COMPLETED'), original_trace.replace('COMPLETED\t0', 'FAILED\t1', 1)):
            self.trace.write_text(trace)
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_count_evidence_changed_while_acquiring_locks_is_rejected(self):
        self.helper.validate_raw_record_count.side_effect = [self.audit, dict(self.audit, raw_records=11)]
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()
        self.assertFalse((self.directory/'adoption.json').exists())

    def test_failure_evidence_changed_while_acquiring_locks_is_rejected(self):
        @contextmanager
        def changing_lock(_):
            put(self.failure, {'state': 'FAILED', 'error': 'different'})
            yield
        self.recovery.boundary_lock = changing_lock
        with self.assertRaisesRegex(ValueError, 'failure evidence changed'):
            self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_changed_trace_during_cached_path_blocks_seal(self):
        self.runner.preprocess.side_effect = lambda _: self.trace.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'must not change checkpoint or trace'):
            self.adopt(execute=True)
        self.assertFalse((self.directory/'adoption.json').exists())
        self.assertFalse((self.boundary/'frozen.sha256.json').exists())

    def test_competing_coordinator_lock_is_not_stolen(self):
        with local.boundary_lock(self.boundary/'.boundary.lock'):
            with self.assertRaisesRegex(RuntimeError, 'owns the lock'):
                self.adopt(execute=True)
        self.pipeline.Runner.assert_not_called()

    def test_conflicting_adoption_evidence_is_not_overwritten(self):
        put(self.directory/'adoption.json', {'other': True})
        before = r.sha(self.directory/'adoption.json')
        with self.assertRaisesRegex(ValueError, 'evidence differs'):
            self.adopt(execute=True)
        self.assertEqual(r.sha(self.directory/'adoption.json'), before)

    def test_resume_requires_declared_adoption_identity(self):
        self.prepare_coordinator()
        self.coordinator_spec['record_count_repair']['adoption_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'count repair differs'):
            r.resume(self.path, r.sha(self.path), r.sha(self.directory/'manifest.json'))
        self.coordinator.run_after_boundary.assert_not_called()

    def test_resume_rejects_changed_adopted_trace(self):
        self.prepare_coordinator()
        self.trace.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Adopted completion changed'):
            r.resume(self.path, r.sha(self.path), r.sha(self.directory/'manifest.json'))
        self.coordinator.run_after_boundary.assert_not_called()

    def test_resume_uses_existing_coordinator_with_original_deadline(self):
        self.prepare_coordinator()
        with patch.object(r.signal, 'signal') as signals, patch.object(r.signal, 'setitimer') as timer:
            r.resume(self.path, r.sha(self.path), r.sha(self.directory/'manifest.json'))
        self.coordinator.run_after_boundary.assert_called_once()
        args = self.coordinator.run_after_boundary.call_args.args
        self.assertEqual(args[:2], (self.coordinator_spec, self.directory))
        self.assertEqual(signals.call_args.args[0], r.signal.SIGALRM)
        self.assertGreater(timer.call_args.args[1], 0)
        self.assertLessEqual(timer.call_args.args[1], 100)

    def test_resume_rejects_lost_local_overlay_before_continuing(self):
        self.prepare_coordinator()
        self.recovery.validate_mount.side_effect = ValueError('Bulk is not the exact local bind mount')
        with patch.object(r.signal, 'signal'), patch.object(r.signal, 'setitimer'):
            with self.assertRaisesRegex(ValueError, 'exact local bind'):
                r.resume(self.path, r.sha(self.path), r.sha(self.directory/'manifest.json'))
        self.coordinator.run_after_boundary.assert_not_called()


if __name__ == '__main__':
    unittest.main()
