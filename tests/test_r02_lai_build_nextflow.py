"""Opt-in tiny REF/FASTA audit and cached receipt, no actual reference access."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9'


@unittest.skipUnless(os.environ.get('RUN_R02_NEXTFLOW_TEST') == '1', 'Explicit Docker/Nextflow opt-in required')
class LaiBuildNextflow(unittest.TestCase):
    def test_ref_audit_and_cached_receipt(self):
        base = Path(tempfile.mkdtemp(prefix='r02-l1-build-nextflow-', dir=ROOT/'.claude/runs'))
        setup = """
import sys
sys.path.insert(0,sys.argv[1]+'/tests')
from test_r02_lai_build_verification import BuildAuditFixture
BuildAuditFixture(sys.argv[2]+'/inputs')
"""
        subprocess.run(['docker','run','--rm','--network','none','--cpus','1','--memory','2g',
                        '--user',f'{os.getuid()}:{os.getgid()}','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
                        '-v',f'{ROOT}:{ROOT}:ro','-v',f'{base}:{base}',IMAGE,'python3','-c',setup,str(ROOT),str(base)],
                       check=True,text=True,capture_output=True,timeout=60)
        data, source = base/'inputs', base/'source_bin'
        source.mkdir()
        for name in ('r02_lai_build_verification.py','r02_lai_allele_support.py','r02_genomic_pair_evidence.py','preprocess_storage_guard.py'):
            shutil.copy2(ROOT/'bin'/name, source/name)
        inputs = dict(vcf=data/'genotype_panel.vcf',fasta=data/'reference.fa',fai=data/'reference.fa.fai',
                      reference_contract=data/'reference_identity.json',contig_map=data/'contigs.json')
        params = dict(source_bin=str(source),**{key:str(path) for key,path in inputs.items()},chromosome='22',
                      reference_contig='chr22',max_records=10,max_line_bytes=8388608,timeout_seconds=30,
                      audit_memory_mb=2048,audit_time='3m',min_free_gib=0)
        for key in ('vcf','reference_contract','contig_map'):
            params['expected_'+key+'_sha256'] = hashlib.sha256(inputs[key].read_bytes()).hexdigest()
        (base/'params.json').write_text(json.dumps(params))
        (base/'resources.config').write_text(f"""process.executor = 'local'
process.container = '{IMAGE}'
process.errorStrategy = 'terminate'
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()} --cpus 1 --memory 2g'
env.PYTHONDONTWRITEBYTECODE = '1'
env.OMP_NUM_THREADS = '1'
env.OPENBLAS_NUM_THREADS = '1'
""")
        env = {**os.environ,'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true','NXF_SYNTAX_PARSER':'v1',
               'NXF_OPTS':'-Xms64m -Xmx512m'}
        command = ['nextflow','-C',str(base/'resources.config'),'run',str(ROOT/'workflows/r02_lai_build_verification.nf'),
                   '-params-file',str(base/'params.json'),'-work-dir',str(base/'work'),'-ansi-log','false']
        print('Synthetic L1 REF audit integration:',base,flush=True)
        for index in range(2):
            result = subprocess.run(command+['-with-trace',str(base/f'trace{index}.tsv')]+([] if index==0 else ['-resume']),
                                    cwd=base,env=env,capture_output=True,text=True,timeout=180)
            (base/f'run{index}.log').write_text(result.stdout+result.stderr)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            with (base/f'trace{index}.tsv').open() as handle:
                rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['status'],'COMPLETED' if index==0 else 'CACHED')
        receipts=list((base/'work').glob('*/*/reference_verification.json'))
        self.assertEqual(len(receipts),1)
        result=json.loads(receipts[0].read_text())
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(result['reference_audit']['records_checked'],3)
        self.assertEqual(result['reference_audit']['mismatches'],0)


if __name__ == '__main__':
    unittest.main()
