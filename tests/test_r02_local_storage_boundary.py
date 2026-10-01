"""Synthetic storage boundary checks; no signals, mounts, containers or cloud jobs."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_storage', ROOT/'bin/r02_local_storage_boundary.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def write(path, value):
    path.write_text(json.dumps(value))


class StorageBoundaryTests(unittest.TestCase):
    def fixture(self, root):
        mount = root/'gcs'; run = root/'parent/parallel/worker01'; run.mkdir(parents=True)
        target = mount/'transient/DNABR_QC/R02_20260930/parent/parallel/worker01'
        target.mkdir(parents=True)
        task = run/'chr02/work/ab/123456abcdef'; task.mkdir(parents=True)
        (task/'.command.sh').write_text('M01 fixture\n')
        for name in ('parameters.json', 'runtime.config'):
            (run/'chr02'/name).write_text('{}')
        config = dict(run_id='parent-worker01', bulk=str(target), processing_order=[2],
            raw_dir=str(mount/'raw'), parallel_worker=dict(parent_run_id='parent', worker_id='worker01'))
        write(run/'run.json', config)
        write(run/'status.json', dict(stage='chr02_M01_M02_M021',state='RUNNING'))
        write(run/'input_objects.json', {str(mount/'raw/dnabr.hg38.2723.chr2.vcf.gz'): dict(size=1)})
        write(run/'frozen.sha256.json', {name:m.sha(run/name) for name in ('run.json','input_objects.json')})
        ident = dict(pid=12345, start_ticks=10, cmdline_sha256='a'*64)
        spec = dict(schema=m.SCHEMA, run_dir=str(run), helper_sha256=m.sha(m.__file__), chromosome=2,
            worker_frozen_sha256=m.sha(run/'frozen.sha256.json'), worker_run_sha256=m.sha(run/'run.json'),
            task_dir=str(task), task_command_sha256=m.sha(task/'.command.sh'),
            parameters_json_sha256=m.sha(run/'chr02/parameters.json'), runtime_config_sha256=m.sha(run/'chr02/runtime.config'),
            nextflow=ident, runner={**ident, 'pid':12346}, timeout_seconds=60, poll_seconds=1)
        path=root/'spec.json'; write(path,spec)
        with patch.object(m,'MOUNT',mount):
            boundary=m.Boundary(path,m.sha(path))
        return boundary

    def processes(self, boundary, state='R'):
        def actual(identity):
            token=b'nextflow' if identity['pid']==12345 else b'worker.py'
            return {**identity, 'state':state, 'command': token+b' '+str(boundary.run).encode()}
        return actual

    def populated(self, boundary):
        source=boundary.target/'m01_chr2.ABC123'; source.mkdir()
        base='dnabr.hg38.2723.chr2'
        for ending in ('.norm.vcf.gz','.norm.vcf.gz.tbi','.norm.log','.original.bcf'):
            (source/(base+ending)).write_bytes(('synthetic'+ending).encode())
        (boundary.task/(base+'.m01.large_temp_dir.txt')).write_text(str(source)+'\n')
        return source

    def metadata(self,path):
        data=Path(path).read_bytes()
        return dict(uri='gs://fixture/'+Path(path).name,generation='1',bytes=len(data),
                    md5_base64=base64.b64encode(hashlib.md5(data).digest()).decode())

    def test_preflight_is_read_only_and_rejects_insufficient_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.fixture(Path(tmp))
            with patch.object(m,'authenticate',side_effect=self.processes(b)), \
                 patch.object(m,'mount_info',return_value={'fstype':'fuse.gcsfuse'}), \
                 patch.object(m.shutil,'disk_usage') as disk:
                disk.return_value.free=100*m.GIB
                self.assertEqual(b.preflight()['state'],'PREFLIGHT_VERIFIED_NOT_APPLIED')
                self.assertFalse(b.backing.exists()); self.assertFalse(b.receipt_dir.exists())
                disk.return_value.free=1
                with self.assertRaisesRegex(ValueError,'Insufficient'): b.preflight()

    def test_changed_pid_identity_rejected(self):
        original=dict(pid=12345,start_ticks=10,cmdline_sha256='a'*64)
        for actual in (None,{**original,'state':'R','start_ticks':11},{**original,'state':'Z'}):
            with patch.object(m,'process_identity',return_value=actual), self.assertRaises(ValueError):
                m.authenticate(original)

    def test_pidfd_opens_before_reauthentication_and_does_not_use_kill(self):
        identity=dict(pid=12345,start_ticks=10,cmdline_sha256='a'*64)
        events=[]
        with patch.object(m.os,'pidfd_open',side_effect=lambda pid:events.append('open') or 42), \
             patch.object(m,'authenticate',side_effect=lambda ident:events.append('authenticate')), \
             patch.object(m.signal,'pidfd_send_signal',side_effect=lambda fd,num:events.append(('signal',fd,num))), \
             patch.object(m.os,'close',side_effect=lambda fd:events.append('close')), \
             patch.object(m.os,'kill',side_effect=AssertionError('numeric PID signalling forbidden')):
            m.signal_owned(identity,signal.SIGSTOP)
        self.assertEqual(events,['open','authenticate',('signal',42,signal.SIGSTOP),'close'])

    def test_wait_is_bounded_and_requires_zero_exit_and_no_live_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.fixture(Path(tmp));b.deadline=20
            with patch.object(m.time,'monotonic',return_value=21):
                with self.assertRaisesRegex(ValueError,'Timed out'):b.wait_finished()
            (b.task/'.exitcode').write_text('1')
            with patch.object(m.time,'monotonic',return_value=10), \
                 patch.object(m,'authenticate',side_effect=self.processes(b,'T')):
                with self.assertRaisesRegex(ValueError,'M01 failed'):b.wait_finished()
            (b.task/'.exitcode').write_text('0')
            with patch.object(m.time,'monotonic',return_value=10), \
                 patch.object(m,'authenticate',side_effect=self.processes(b,'T')), \
                 patch.object(m,'active_containers',side_effect=[['fixture'],[]]), \
                 patch.object(m.time,'sleep') as sleep:
                b.wait_finished();sleep.assert_called_once_with(1)

    def test_source_copy_preserves_bytes_mtime_permissions_and_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'; target=Path(tmp)/'copy'; source.write_bytes(b'synthetic')
            os.utime(source,ns=(1000000000,1234567890000000000))
            with patch.object(m,'cloud_metadata',side_effect=self.metadata):
                record=m.copy_verified(source,target)
            self.assertEqual(m.file_metadata(source),m.file_metadata(target))
            self.assertEqual(record['sha256'],m.sha(target)); self.assertTrue(source.exists())

    def test_changed_generation_or_md5_refuses_copy(self):
        for change in ('generation','md5'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                source=Path(tmp)/'source'; target=Path(tmp)/'copy'; source.write_bytes(b'synthetic')
                original=self.metadata(source); revised={**original,'generation':'2'}
                values=[original,revised] if change=='generation' else [{**original,'md5_base64':'incorrect'}]*2
                with patch.object(m,'cloud_metadata',side_effect=values),self.assertRaises(ValueError):
                    m.copy_verified(source,target)
                self.assertTrue(source.exists())
                self.assertFalse(target.exists())

    def test_selected_outputs_only_original_bcf_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.fixture(Path(tmp)); source=self.populated(b); b.receipt_dir.mkdir(parents=True)
            with patch.object(m,'cloud_metadata',side_effect=self.metadata),patch.object(m.shutil,'disk_usage') as disk:
                disk.return_value.free=100*m.GIB
                records=b.copy_outputs()
            self.assertEqual(len(records),3)
            self.assertFalse((b.backing/source.name/'dnabr.hg38.2723.chr2.original.bcf').exists())
            self.assertTrue((source/'dnabr.hg38.2723.chr2.original.bcf').exists())
            self.assertEqual(b.backing.stat().st_uid,b.target.stat().st_uid)

    def test_symlink_and_unknown_files_rejected(self):
        for change in ('symlink','unknown','other_task'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                b=self.fixture(Path(tmp)); source=self.populated(b)
                if change=='symlink':
                    p=source/'dnabr.hg38.2723.chr2.norm.log';p.unlink();p.symlink_to(source/'dnabr.hg38.2723.chr2.norm.vcf.gz')
                elif change=='unknown':(source/'foreign_file').write_text('x')
                else:(b.target/'m02_chr2.UNRELATED').mkdir()
                with self.assertRaises(ValueError):b.copy_outputs()

    def test_downstream_task_prevents_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.fixture(Path(tmp));other=b.task.parent.parent/'ac/abcdef123456';other.mkdir(parents=True)
            (other/'.command.sh').write_text('M02')
            with self.assertRaisesRegex(ValueError,'Another workflow task'):b.no_downstream()

    def apply_case(self,mode):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.fixture(Path(tmp));signals=[];mounts=[]
            def command(argv,**kwargs):
                mounts.append(argv[0])
                if mode=='mount_fails' and argv[0]=='mount':raise RuntimeError('mount failed')
            def check(records):
                if mode=='validation_fails':raise ValueError('bound data differ')
                return dict(target=str(b.target),fstype='ext4')
            with patch.object(m.os,'geteuid',return_value=0), \
                 patch.object(b,'preflight',return_value={'state':'PREFLIGHT'}), \
                 patch.object(m,'process_identity',return_value=dict(pid=1,start_ticks=1,cmdline_sha256='a'*64)), \
                 patch.object(m,'signal_owned',side_effect=lambda who,num:signals.append((who['pid'],num))), \
                 patch.object(m,'authenticate',side_effect=self.processes(b,'T')), \
                 patch.object(b,'wait_finished',side_effect=ValueError('M01 failed') if mode=='m01_fails' else None), \
                 patch.object(b,'copy_outputs',return_value=[]),patch.object(b,'no_downstream'), \
                 patch.object(m,'active_containers',return_value=[]),patch.object(m.subprocess,'run',side_effect=command), \
                 patch.object(b,'validate_mount',side_effect=check), \
                 patch.object(m,'mount_info',return_value={'fstype':'fuse.gcsfuse'}):
                if mode=='success':b.apply()
                else:
                    with self.assertRaises((ValueError,RuntimeError)):b.apply()
            self.assertEqual(signals,[(12345,signal.SIGSTOP),(12345,signal.SIGCONT)])
            self.assertFalse(b.owned_stop)
            if mode=='validation_fails':self.assertEqual(mounts,['mount','umount'])
            if mode=='m01_fails':self.assertEqual(mounts,[])
            events=[json.loads(line) for line in b.events.read_text().splitlines()]
            self.assertEqual(events[-1]['state'],'COMPLETE_JVM_RESUMED' if mode=='success' else 'ABORT_JVM_RESUMED_ORIGINAL_STORAGE')
            with patch.object(m.os,'geteuid',return_value=0),self.assertRaisesRegex(ValueError,'previous transition'):
                b.apply()

    def test_m01_failure_resumes_only_owned_jvm(self):self.apply_case('m01_fails')
    def test_mount_failure_resumes_original_storage(self):self.apply_case('mount_fails')
    def test_bound_validation_failure_rolls_back_then_resumes(self):self.apply_case('validation_fails')
    def test_success_mounts_then_resumes_without_changing_parameters(self):self.apply_case('success')

    def test_failed_rollback_keeps_jvm_paused_for_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.fixture(Path(tmp));signals=[]
            def command(argv,**kwargs):
                if argv[0]=='umount':raise RuntimeError('device busy')
            with patch.object(m.os,'geteuid',return_value=0), \
                 patch.object(b,'preflight',return_value={'state':'PREFLIGHT'}), \
                 patch.object(m,'process_identity',return_value=dict(pid=1,start_ticks=1,cmdline_sha256='a'*64)), \
                 patch.object(m,'signal_owned',side_effect=lambda who,num:signals.append(num)), \
                 patch.object(m,'authenticate',side_effect=self.processes(b,'T')), \
                 patch.object(b,'wait_finished'),patch.object(b,'copy_outputs',return_value=[]), \
                 patch.object(b,'no_downstream'),patch.object(m,'active_containers',return_value=[]), \
                 patch.object(m.subprocess,'run',side_effect=command), \
                 patch.object(b,'validate_mount',side_effect=ValueError('bad mount')):
                with self.assertRaisesRegex(RuntimeError,'rollback failed'):b.apply()
            self.assertEqual(signals,[signal.SIGSTOP])
            self.assertIn('BLOCKED_ROLLBACK_FAILED_JVM_REMAINS_PAUSED',b.events.read_text())


if __name__=='__main__':unittest.main()
