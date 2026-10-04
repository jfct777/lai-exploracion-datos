"""Opt-in Nextflow execution/resume test using invented individuals only."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:76a0fddd300d3634363b5a13a1d28983122644a42e7d6c18339f9c8ac08548a7'


@unittest.skipUnless(os.environ.get('RUN_R02_NEXTFLOW_TEST') == '1', 'Explicit Docker/Nextflow opt-in required')
class CommunitySummaryNextflow(unittest.TestCase):
    def test_complete_bundle_and_resume(self):
        base = Path(tempfile.mkdtemp(prefix='r02-community-summary-nextflow-', dir=ROOT/'.claude/runs'))
        setup = ("import sys,json;sys.path.insert(0,sys.argv[1]+'/tests');"
                 "from test_r02_community_evidence import SyntheticFixture,community;"
                 "f=SyntheticFixture(sys.argv[2]+'/input');"
                 "out=sys.argv[2]+'/evidence';community.run(f.contract_path,f.contract_hash,out,2048);"
                 "print(json.dumps({'path':out,'sha256':community.sha256(out+'/manifest.json')}))")
        result = subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
            '-v',f'{ROOT}:{ROOT}:ro','-v',f'{base}:{base}',IMAGE,
            'python3','-c',setup,str(ROOT),str(base)], check=True, text=True, capture_output=True, timeout=60)
        fixture = json.loads(result.stdout)
        (base/'params.json').write_text(json.dumps(dict(source_bin=str(ROOT/'bin'), evidence_dir=fixture['path'],
            evidence_manifest_sha256=fixture['sha256'], min_free_gib=0, max_memory_mb=2048)))
        (base/'resources.config').write_text(f"""
process.executor = 'local'
process.container = '{IMAGE}'
process.memory = '3 GB'
process.time = '3m'
process.errorStrategy = 'terminate'
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()} --cpus 1 --memory 3g'
env.PYTHONDONTWRITEBYTECODE = '1'
env.OMP_NUM_THREADS = '1'
env.OPENBLAS_NUM_THREADS = '1'
""")
        env = {**os.environ, 'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true',
               'NXF_SYNTAX_PARSER':'v1','NXF_OPTS':'-Xms64m -Xmx512m'}
        command = ['nextflow','-C',str(base/'resources.config'),'run',str(ROOT/'workflows/r02_community_summary.nf'),
                   '-params-file',str(base/'params.json'),'-work-dir',str(base/'work'),'-ansi-log','false']
        print('Synthetic summary/cache integration:',base,flush=True)
        for i, resume in enumerate([[],['-resume']]):
            result = subprocess.run(command+['-with-trace',str(base/f'trace{i}.tsv')]+resume,
                cwd=base,env=env,capture_output=True,text=True,timeout=210)
            (base/f'captured{i}.log').write_text(result.stdout+result.stderr)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            with (base/f'trace{i}.tsv').open() as handle:
                rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['status'],'COMPLETED' if i==0 else 'CACHED')
        manifests=list((base/'work').glob('*/*/community_summary/manifest.json'))
        self.assertEqual(len(manifests),1)
        manifest=json.loads(manifests[0].read_text())
        self.assertEqual(manifest['status'],'COMPLETE_DESCRIPTIVE_COMMUNITY_SUMMARY_NOT_VALIDATED')
        self.assertEqual(manifest['n_cohort'],6)
        self.assertTrue(manifest['no_reclustering'])
        self.assertFalse(manifest['public_distribution_allowed'])


if __name__ == '__main__':
    unittest.main()
