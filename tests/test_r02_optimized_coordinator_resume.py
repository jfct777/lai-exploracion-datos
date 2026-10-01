"""Synthetic continuation safety: no jobs, signals, VM operations or GCS calls."""
from contextlib import nullcontext
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


resume = load_module('optimized_coordinator_test', ROOT/'bin/r02_optimized_coordinator_resume.py')
recovery = load_module('original_recovery_test', ROOT/'bin/r02_local_preprocess_resume.py')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ResumeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)/'run'
        self.directory = self.run/'repairs/preprocess-v1/coordinator'
        self.directory.mkdir(parents=True)
        self.old = dict(run_dir=str(self.run), wrapper_sha256='f'*64,
                       old_processes={})
        self.spec = dict(schema=resume.SCHEMA, run_dir=str(self.run), resume_sha256=resume.sha(resume.__file__),
            coordinator_manifest_sha256='a'*64, recovery_manifest_sha256='b'*64,
            previous_controller=dict(pid=12345, start_ticks=30, cmdline_sha256='c'*64), poll_seconds=1)
        self.spec_path = self.directory/'resume.manifest.json'
        put(self.spec_path, self.spec)
        self.trace = self.run/'chr21/trace.tsv'
        self.trace.parent.mkdir()
        self.trace.write_text('name\tstatus\texit\thash\n' +
            f'{recovery.M01}\tCACHED\t0\tab/cdef\n' +
            f'{recovery.M02}\tCOMPLETED\t0\tcd/efab\n' +
            f'{recovery.M021}\tCOMPLETED\t0\tef/abcd\n')
        self.checkpoint = self.run/'checkpoints/chr21_M01_M02_M021.json'
        self.output = self.run/'chr21/result.txt'
        self.output.write_text('synthetic output')
        self.command = ['nextflow', 'original-workflow', '-resume']
        put(self.checkpoint, dict(returncode=0, command=self.command,
            outputs=[dict(path=str(self.output), sha256=resume.sha(self.output))]))
        self.evidence_path = self.run/'repairs/local-preprocess-v1/completed.json'
        self.evidence = dict(recovery_spec_sha256=self.spec['recovery_manifest_sha256'],
            wrapper_sha256=self.old['wrapper_sha256'], checkpoint_sha256=resume.sha(self.checkpoint),
            checkpoint_created_by='frozen_original_Runner.preprocess(21)',
            historical_blocked_fork_status_fabricated=False, scientific_parameters_changed=False,
            command=self.command, trace_sha256=resume.sha(self.trace),
            trace=recovery.check_trace(self.trace, recovered=True))
        put(self.evidence_path, self.evidence)
        boundary = self.run/'repairs/boundary-v2'
        put(boundary/'source.sha256.json', {'bin/r02_apply_amendment.py':'d'*64})
        put(boundary/'amendment.json', dict(boundary=dict(checkpoint_sha256=resume.sha(self.checkpoint))))
        put(boundary/'preprocess_storage_recovery.json', self.evidence)
        put(boundary/'frozen.sha256.json', {'synthetic':'seal'})
        self.handoff = types.SimpleNamespace(authenticated=Mock(return_value=None),
            write_json=lambda p,v,**_:put(p,v))
        def verify_files(records):
            for record in records:
                if resume.sha(record['path']) != record['sha256']:
                    raise ValueError('output changed')
        self.pipeline = types.SimpleNamespace(verify_files=Mock(side_effect=verify_files))
        self.adapter = types.SimpleNamespace(expected_boundary_command=lambda *_:self.command,
                                             validate_snapshot=Mock())
        self.patches = [patch.object(recovery, 'validate_idle'),
                        patch.object(resume, 'module', return_value=self.adapter)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def verify(self):
        return resume.validate_completion(self.spec, recovery, self.old, self.handoff, self.pipeline)

    def test_existing_completed_checkpoint_is_verified_never_recreated(self):
        originals = {p: p.read_bytes() for p in (self.checkpoint, self.evidence_path, self.trace)}
        result = self.verify()
        self.assertEqual(result['checkpoint_sha256'], resume.sha(self.checkpoint))
        self.adapter.validate_snapshot.assert_called_once()
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)

    def test_live_controller_rejected_before_outputs(self):
        self.handoff.authenticated.return_value = {'pid':12345}
        with self.assertRaisesRegex(ValueError, 'still active'):
            self.verify()
        self.pipeline.verify_files.assert_not_called()

    def test_missing_completion_does_not_create_checkpoint_or_evidence(self):
        self.evidence_path.unlink()
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            self.verify()
        self.assertFalse(self.evidence_path.exists())

    def test_changed_checkpoint_rejected(self):
        put(self.checkpoint, dict(returncode=0, command=['different'], outputs=[]))
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.verify()

    def test_changed_output_rejected(self):
        self.output.write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            self.verify()

    def test_changed_trace_rejected(self):
        self.trace.write_text(self.trace.read_text().replace('COMPLETED', 'FAILED'))
        with self.assertRaisesRegex(ValueError, 'trace changed'):
            self.verify()

    def test_unsealed_completion_rejected(self):
        put(self.run/'repairs/boundary-v2/preprocess_storage_recovery.json', {'wrong':True})
        with self.assertRaisesRegex(ValueError, 'Boundary seal'):
            self.verify()

    def test_manifest_hash_is_checked_before_imports(self):
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            resume.validate_spec(self.spec_path, '0'*64)

    def test_check_mode_does_not_wait_or_start_coordinator(self):
        coordinator = types.SimpleNamespace(run_after_boundary=Mock())
        context = (self.spec, {}, coordinator, recovery, self.old, self.handoff, self.pipeline)
        with patch.object(resume, 'validate_spec', return_value=context):
            result = resume.run(self.spec_path, resume.sha(self.spec_path))
        self.assertEqual(result['state'], 'VERIFIED_NOT_EXECUTED')
        coordinator.run_after_boundary.assert_not_called()
        self.assertFalse((self.directory/'resume_activation.json').exists())

    def test_execute_waits_and_uses_original_lock_then_new_coordinator(self):
        events = []
        coordinator = types.SimpleNamespace(timestamp=lambda _:resume.time.time()+120,
            write_fixed=lambda p,v:put(p,v),
            run_after_boundary=Mock(side_effect=lambda *_:events.append('new coordinator')))
        context = (self.spec, {'deadline_utc':'synthetic'}, coordinator,
                   recovery, self.old, self.handoff, self.pipeline)
        self.handoff.authenticated.side_effect = [{'pid':12345}, None, None]
        with patch.object(resume, 'validate_spec', return_value=context), \
             patch.object(resume.time, 'sleep', side_effect=lambda _:events.append('wait')), \
             patch.object(resume.signal, 'signal'), patch.object(resume.signal, 'setitimer'), \
             patch.object(recovery, 'boundary_lock', return_value=nullcontext()) as lock:
            self.assertEqual(resume.run(self.spec_path, 'a'*64, execute=True), {'state':'COMPLETE'})
        self.assertEqual(events, ['wait', 'new coordinator'])
        lock.assert_called_once_with(self.run/'repairs/boundary-v2/.boundary.lock')
        self.assertTrue((self.directory/'resume_activation.json').is_file())


if __name__ == '__main__':
    unittest.main()
