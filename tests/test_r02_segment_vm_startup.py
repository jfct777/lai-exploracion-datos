"""Offline bootstrap contracts; never create VMs, publish, install or shut down."""
import io
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'bin/r02_segment_vm_startup.sh'


class StartupContractTests(unittest.TestCase):
    def shell(self, command, variables=None):
        assignments = '\n'.join(f'{key}={shlex.quote(value)}' for key,value in (variables or {}).items())
        return subprocess.run(['bash','-c',f'source {shlex.quote(str(SCRIPT))}\n{assignments}\n{command}'],
                              text=True,capture_output=True)

    def assignment(self):
        return dict(worker='chr01',instance_name='dnabr-m142-1003-c01',instance_id='1234',
                    project_id='uspbr-242713',zone='us-central1-a',
                    run_dir='/home/jose.tantalean/projects/lai-exploracion-datos/.claude/runs/r02-m142-fleet-20261003a/chr01',
                    bundle_sha='a'*64,campaign_sha='b'*64,
                    runtime_sha='8c1f85de9e192be2fcc19e5a278d29ae28fc586a5b76c40eec54ef5f0b384916',
                    bundle_uri='gs://projects-usp/dnaBr-lai/datalake/transient/DNABR_QC/R02_20260930/r02-m142-fleet-20261003a/chr01.tar.gz',
                    runtime_uri='gs://projects-usp/dnaBr-lai/datalake/transient/DNABR_QC/R02_20260930/r02-autosomes-20261001b/parallel-launch-20261001/runtime.tar.gz',
                    log_uri='gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-m142-fleet-20261003a/chr01/00_vm_logs')

    def test_bash_syntax(self):
        result = subprocess.run(['bash','-n',str(SCRIPT)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_source_does_not_bootstrap(self):
        result = self.shell('printf source_only')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'source_only')

    def test_expected_assignment(self):
        self.assertEqual(self.shell('validate_assignment',self.assignment()).returncode,0)

    def test_bad_assignment_fails_even_when_function_is_conditional(self):
        for field,bad in [('worker','chr19'),('instance_name','development-vm'),('instance_id','x'),
                          ('project_id','other-project'),('zone','us-central1-b'),('run_dir','/tmp/x'),
                          ('campaign_sha','x'),('runtime_sha','a'*64),('runtime_uri','gs://projects-usp/other'),
                          ('bundle_uri','gs://projects-usp/dnaBr-lai/datalake/transient/DNABR_QC/R02_20260930/r02-m142-fleet-20261003a/../bad'),
                          ('log_uri','gs://public-bucket/log')]:
            with self.subTest(field=field):
                values=self.assignment();values[field]=bad
                self.assertNotEqual(self.shell('if validate_assignment; then exit 0; else exit 4; fi',values).returncode,0)

    def extract(self,members,ceiling=10000,root='runtime'):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        folder=Path(temporary.name);archive=folder/'a.tar.gz';destination=folder/'out';destination.mkdir()
        with tarfile.open(archive,'w:gz') as writer:
            for member,content in members:
                writer.addfile(member,io.BytesIO(content) if member.isfile() else None)
        result=self.shell('safe_extract '+ ' '.join(shlex.quote(str(x)) for x in (archive,destination,root,ceiling)))
        return result,destination

    @staticmethod
    def regular(name,content=b'hello'):
        member=tarfile.TarInfo(name);member.size=len(content)
        return member,content

    def test_regular_runtime_extracts(self):
        directory=tarfile.TarInfo('runtime');directory.type=tarfile.DIRTYPE
        result,destination=self.extract([(directory,b''),self.regular('runtime/nextflow')])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((destination/'runtime/nextflow').read_bytes(),b'hello')

    def test_reject_traversal_absolute_and_wrong_root(self):
        for name in ('../outside','/tmp/outside','runtime/../../outside','wrong/file'):
            with self.subTest(name=name):
                self.assertNotEqual(self.extract([self.regular(name)])[0].returncode,0)

    def test_reject_symlink_hardlink_device(self):
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.CHRTYPE):
            member=tarfile.TarInfo('runtime/bad');member.type=kind;member.linkname='/tmp/outside'
            with self.subTest(kind=kind):
                self.assertNotEqual(self.extract([(member,b'')])[0].returncode,0)

    def test_reject_duplicates_and_oversize(self):
        self.assertNotEqual(self.extract([self.regular('runtime/a'),self.regular('runtime/a')])[0].returncode,0)
        self.assertNotEqual(self.extract([self.regular('runtime/a')],ceiling=4)[0].returncode,0)

    def test_reject_reextract_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);archive=folder/'a.tar.gz';out=folder/'out';out.mkdir();(out/'x').write_text('original')
            with tarfile.open(archive,'w:gz') as writer:
                member,content=self.regular('x');writer.addfile(member,io.BytesIO(content))
            result=self.shell(f'safe_extract {archive} {out} "" 100')
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((out/'x').read_text(),'original')

    def own_instance(self,**changes):
        result=dict(name='dnabr-m142-1003-c01',id='1234',
                    zone='https://www.googleapis.com/compute/v1/projects/uspbr-242713/zones/us-central1-a',
                    labels=dict(role='m142-worker',team='frank',round='r02'),
                    disks=[dict(boot=True,source='https://www.googleapis.com/compute/v1/projects/uspbr-242713/zones/us-central1-a/disks/dnabr-m142-1003-c01')])
        result.update(changes);return result

    def check_instance(self,record):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'instance.json';file.write_text(json.dumps(record))
            return self.shell(f'verify_own_instance {file}',self.assignment())

    def test_own_instance_identity(self):
        result=self.check_instance(self.own_instance())
        self.assertEqual(result.returncode,0,result.stderr)

    def test_refuse_other_instance_or_extra_disks(self):
        for changes in [dict(id='1235'),dict(name='important-vm'),dict(labels=dict(role='other')),
                        dict(zone='https://www.googleapis.com/compute/v1/projects/other/zones/us-central1-a'),
                        dict(disks=[dict(boot=True,source='wrong')]),
                        dict(disks=[dict(boot=True,source='https://www.googleapis.com/compute/v1/projects/uspbr-242713/zones/us-central1-a/disks/unrelated-disk')]),
                        dict(disks=self.own_instance()['disks']*2)]:
            with self.subTest(changes=changes):
                self.assertNotEqual(self.check_instance(self.own_instance(**changes)).returncode,0)

    def test_safety_contract_in_startup(self):
        source=SCRIPT.read_text()
        self.assertNotIn('gcsfuse',source)
        for expected in ('--expected-sha256 "$campaign_sha"','--validate-only',
                         '--minimum-free-gib 12','--delete-disks=boot','publication.verify_remote(item, pinned=True)',
                         "assert status['stage'] == 'COMPLETE_PUBLISHED_ALL_JOBS'",'165600 - SECONDS',
                         'verify_own_instance "$bootstrap_dir/instance-before-delete.json"',
                         '[[ "$complete" == 1 && "$published" == 1 ]]','shutdown -h now',
                         'path.stat().st_uid == 1017 and path.stat().st_gid == 1020',
                         'with tempfile.TemporaryFile(dir=path) as probe:',
                         '/home/jose.tantalean/.nextflow/framework /home/jose.tantalean/.nextflow/framework/26.04.6'):
            self.assertIn(expected,source)
        self.assertNotIn('rm -rf',source)

    def test_publisher_uses_mutated_generation_not_return_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);module=root/'frozen/bin';module.mkdir(parents=True)
            (root/'sample.log').write_text('bounded log')
            (module/'r02_autosome_pipeline.py').write_text('def publication_digests(path):\n return dict(bytes=11,sha256="a"*64,md5_base64="fixture")\n')
            (module/'r02_publish_evidence.py').write_text('def upload_one(record):\n record["generation"]="123"\ndef verify_remote(record,pinned=False):\n assert pinned and record["generation"]=="123"\n')
            command='runuser() { shift 3; "$@"; }\npublish_private_file '+shlex.quote(str(root/'sample.log'))+' gs://fixture/log'
            result=self.shell(command,dict(run_dir=str(root)))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['generation'],'123')

    def test_single_chromosome_campaign_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            record=dict(mode='PRODUCTION',resources=dict(max_concurrent=1,memory_budget_gib=12),
                        jobs=[dict(run_dir=str(root/'job'),geometry=dict(expected=dict(chrom='1',n_samples=2619)))])
            for invalid in (False,True):
                if invalid:record['jobs'][0]['geometry']['expected']['chrom']='2'
                path=root/'campaign.json';path.write_text(json.dumps(record))
                result=self.shell('validate_campaign_assignment',dict(run_dir=str(root),worker='chr01',campaign_sha=hashlib.sha256(path.read_bytes()).hexdigest()))
                self.assertEqual(result.returncode==0,not invalid,result.stderr)


if __name__=='__main__':
    unittest.main()
