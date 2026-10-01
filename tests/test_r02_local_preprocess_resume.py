"""Synthetic storage recovery tests: no GCP, mounts, genotypes or process signals."""
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_local_resume', ROOT/'bin/r02_local_preprocess_resume.py')
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)/'run'
        self.directory = self.run/'repairs/local-preprocess-v1'
        self.directory.mkdir(parents=True)
        self.bulk = self.run/'local_bulk'
        self.bulk.mkdir()
        put(self.run/'run.json', dict(run_id='synthetic', bulk=str(self.bulk)))
        self.task = self.run/'chr21/work/ab/abcdef'
        self.task.mkdir(parents=True)
        (self.task/'.exitcode').write_text('0\n')
        self.m01 = self.bulk/'m01_chr21.testing'
        self.m01.mkdir()
        (self.task/'dnabr.hg38.2723.chr21.m01.large_temp_dir.txt').write_text(str(self.m01)+'\n')
        files = []
        for suffix in ('vcf.gz', 'vcf.gz.tbi', 'log'):
            path = self.m01/('dnabr.hg38.2723.chr21.norm.'+suffix)
            path.write_text('synthetic-'+suffix)
            files.append(dict(path=str(path), bytes=path.stat().st_size,
                              sha256=resume.sha(path), mtime_ns=path.stat().st_mtime_ns))
        self.spec = dict(schema=resume.SCHEMA, run_dir=str(self.run), local_backing_dir=str(self.bulk),
                         m01_task_dir=str(self.task), m01_files=files, wrapper_sha256='f'*64,
                         storage_receipt=dict(path='synthetic', sha256='a'*64),
                         old_processes={name:dict(pid=n+100, start_ticks=30+n, cmdline_sha256='b'*64)
                                        for n, name in enumerate(('supervisor','child','coordinator'))})
        self.spec_path = self.directory/'manifest.json'
        put(self.spec_path, self.spec)
        self.trace = self.run/'chr21/trace.tsv'
        self.write_trace()

    def write_trace(self, norm='CACHED', second='COMPLETED', third='COMPLETED'):
        self.trace.write_text('name\tstatus\texit\thash\n'+
            f'{resume.M01}\t{norm}\t0\tab/cdef\n'+
            f'{resume.M02}\t{second}\t0\tcd/efab\n'+
            f'{resume.M021}\t{third}\t0\tef/abcd\n')

    def test_trace_requires_m01_cached_not_recomputed(self):
        resume.check_trace(self.trace, recovered=True)
        self.write_trace(norm='COMPLETED')
        with self.assertRaisesRegex(ValueError, 'genuinely cached'):
            resume.check_trace(self.trace, recovered=True)

    def test_trace_rejects_failed_downstream(self):
        self.write_trace(second='FAILED')
        with self.assertRaisesRegex(ValueError, 'did not finish'):
            resume.check_trace(self.trace, recovered=True)

    def test_trace_rejects_extra_tasks(self):
        with self.trace.open('a') as stream:
            stream.write('OTHER\tCOMPLETED\t0\tXX/YY\n')
        with self.assertRaisesRegex(ValueError, 'exactly M01'):
            resume.check_trace(self.trace, recovered=True)

    def test_original_trace_must_be_completed(self):
        with self.assertRaises(ValueError):
            resume.check_trace(self.trace, recovered=False)
        self.write_trace(norm='COMPLETED')
        resume.check_trace(self.trace, recovered=False)

    def test_m01_three_outputs_and_metadata_verified(self):
        self.assertEqual(resume.validate_m01(self.spec, self.bulk), self.spec['m01_files'])
        self.spec['m01_files'][0]['mtime_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'modification time'):
            resume.validate_m01(self.spec, self.bulk)

    def test_m01_changed_hash_rejected(self):
        self.spec['m01_files'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'content'):
            resume.validate_m01(self.spec, self.bulk)

    def test_m01_nonzero_exit_rejected(self):
        (self.task/'.exitcode').write_text('143\n')
        with self.assertRaisesRegex(ValueError, 'exitcode'):
            resume.validate_m01(self.spec, self.bulk)

    def test_abandoned_m02_marker_needs_empty_overlay_directory(self):
        task = self.run/'chr21/work/22/oldtask'
        task.mkdir(parents=True)
        prior = self.bulk/'m02_chr21.oldattempt'
        (task/'chr21.large_temp_dir.txt').write_text(str(prior))
        with self.assertRaisesRegex(ValueError, 'previous temporary'):
            resume.validate_m01(self.spec, self.bulk)
        prior.mkdir()
        resume.validate_m01(self.spec, self.bulk)

    def test_mount_must_be_exact_local_filesystem(self):
        command = Mock(return_value=json.dumps(dict(filesystems=[dict(target=str(self.bulk),fstype='ext4')])))
        self.assertEqual(resume.validate_mount(self.spec, command=command), self.bulk)
        command.return_value = json.dumps(dict(filesystems=[dict(target=str(self.bulk),fstype='fuse.gcsfuse')]))
        with self.assertRaisesRegex(ValueError, 'exact local bind'):
            resume.validate_mount(self.spec, command=command)

    def test_mount_parent_not_accepted(self):
        command = Mock(return_value=json.dumps(dict(filesystems=[dict(target=str(self.run),fstype='ext4')])))
        with self.assertRaises(ValueError):
            resume.validate_mount(self.spec, command=command)

    def test_live_old_coordinator_rejected(self):
        handoff = types.SimpleNamespace(authenticated=Mock(side_effect=[None,None,dict(pid=102)]),
                                        containers=Mock(return_value=[]))
        pipeline = types.SimpleNamespace(Runner=Mock())
        with self.assertRaisesRegex(ValueError, 'coordinator'):
            resume.validate_idle(self.spec, handoff, pipeline)
        pipeline.Runner.assert_not_called()

    def test_active_original_container_rejected(self):
        handoff = types.SimpleNamespace(authenticated=Mock(return_value=None),
                                        containers=Mock(return_value=['active']))
        with self.assertRaisesRegex(ValueError, 'active containers'):
            resume.validate_idle(self.spec, handoff, types.SimpleNamespace(Runner=Mock()))

    def test_boundary_lock_never_stolen(self):
        lock = self.directory/'lock'
        with resume.boundary_lock(lock):
            with self.assertRaisesRegex(RuntimeError, 'owns the lock'):
                with resume.boundary_lock(lock):
                    self.fail('second holder acquired lock')

    def test_process_identity_has_exact_fields(self):
        good = self.spec['old_processes']['child']
        resume.identity(good)
        with self.assertRaises(ValueError):
            resume.identity(dict(good, extra=True))

    def test_spec_wrong_hash_fails_before_loading_code(self):
        with self.assertRaisesRegex(ValueError, 'spec hash'):
            resume.validate_spec(self.spec_path, '0'*64)

    def test_spec_wrong_schema_fails_before_loading_code(self):
        put(self.spec_path, dict(self.spec, schema='wrong'))
        with self.assertRaisesRegex(ValueError, 'schema'):
            resume.validate_spec(self.spec_path, resume.sha(self.spec_path))

    def test_spec_wrong_wrapper_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, 'wrapper changed'):
            resume.validate_spec(self.spec_path, resume.sha(self.spec_path))

    def test_duplicate_json_keys_rejected(self):
        self.spec_path.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            resume.load(self.spec_path)

    def test_only_frozen_runner_creates_genuine_checkpoint(self):
        checkpoint = self.run/'checkpoints'/f'{resume.STAGE}.json'
        source = self.run/'repairs/boundary-v2'
        put(source/'source.sha256.json', {'bin/r02_apply_amendment.py':'a'*64})
        expected_command = ['nextflow', 'original-workflow', '-resume']
        def run_preprocess(chrom):
            self.assertEqual(chrom, 21)
            put(checkpoint, dict(command=expected_command, returncode=0, outputs=[{'synthetic':True}]))
        runner = types.SimpleNamespace(preprocess=Mock(side_effect=run_preprocess))
        pipeline = types.SimpleNamespace(Runner=Mock(return_value=runner), execution_lock=lambda _:nullcontext(),
                                         verify_files=Mock())
        handoff = types.SimpleNamespace(authenticated=Mock(return_value=None),containers=Mock(return_value=[]),
                                        write_json=lambda p,v,**_:put(p,v))
        adapter = types.SimpleNamespace(expected_boundary_command=lambda *_:expected_command)
        with patch.object(resume,'validate_mount',return_value=self.bulk), \
                patch.object(resume,'authenticated_module',return_value=adapter):
            evidence = resume.complete_preprocess(self.spec,{},None,handoff,pipeline,self.spec_path)
        runner.preprocess.assert_called_once_with(21)
        self.assertEqual(evidence['command'],expected_command)
        self.assertFalse(evidence['historical_blocked_fork_status_fabricated'])
        self.assertFalse(evidence['scientific_parameters_changed'])

    def test_seal_retains_original_request_and_records_recovery(self):
        source = self.run/'repairs/boundary-v2'
        request = dict(amendment_template=dict(boundary=dict(chromosome=21)))
        put(source/'request.json',request)
        put(source/'source.sha256.json',{'a':'b'})
        original = resume.sha(source/'request.json')
        handoff = types.SimpleNamespace(write_json=lambda p,v,**_:put(p,v))
        evidence = dict(checkpoint_sha256='c'*64)
        resume.seal_boundary(self.spec,evidence,handoff)
        self.assertEqual(resume.sha(source/'request.json'),original)
        seal = resume.load(source/'frozen.sha256.json')
        self.assertIn('preprocess_storage_recovery.json',seal)
        self.assertEqual(resume.load(source/'preprocess_storage_recovery.json'),evidence)


if __name__ == '__main__':
    unittest.main()
