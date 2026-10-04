"""Opt-in Nextflow single-chromosome geometry/annotation/cache integration."""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9'


@unittest.skipUnless(os.environ.get('RUN_R02_NEXTFLOW_TEST') == '1', 'Explicit Docker/Nextflow opt-in required')
class SegmentSingleNextflow(unittest.TestCase):
    def test_geometry_annotation_and_resume(self):
        base = Path(tempfile.mkdtemp(prefix='r02-segment-single-nextflow-', dir=ROOT/'.claude/runs'))
        setup = """
import sys,shutil
sys.path.insert(0,sys.argv[1]+'/tests')
from test_r02_segment_evidence import SegmentEvidenceTests
t=SegmentEvidenceTests();t.setUp();t.spanning_chain();t.map_fixture()
shutil.copytree(t.root,sys.argv[2]+'/input')
"""
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
            '-v',f'{ROOT}:{ROOT}:ro','-v',f'{base}:{base}',IMAGE,
            'python3','-c',setup,str(ROOT),str(base)], check=True, text=True, capture_output=True, timeout=60)
        data=base/'input'
        params=dict(source_bin=str(ROOT/'bin'), chains=str(data/'candidate_chains.tsv.gz'),
            configurations=str(data/'configuration_summary.tsv'), rare_vcf=str(data/'spanning.vcf.gz'),
            rare_index=str(data/'spanning.vcf.gz.tbi'), samples=str(data/'samples.ids'),
            chromosome='22',expected_source_samples=101,genome_build='GRCh38',block_bp=150,chunk_sites=1,
            max_active_intervals=1000,min_free_gib=0,min_free_disk_mb=0,max_db_mb=32,
            preflight_memory_mb=16,max_preflight_blocks=1000,resource_check_rows=1)
        (base/'resources.config').write_text(f"""
process.executor = 'local'
process.container = '{IMAGE}'
process.memory = '2 GB'
process.time = '3m'
process.errorStrategy = 'terminate'
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()} --cpus 1 --memory 2g'
env.PYTHONDONTWRITEBYTECODE = '1'
env.OMP_NUM_THREADS = '1'
env.OPENBLAS_NUM_THREADS = '1'
""")
        env={**os.environ,'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true',
             'NXF_SYNTAX_PARSER':'v1','NXF_OPTS':'-Xms64m -Xmx512m'}
        print('Synthetic M14.2 single-chromosome integration:',base,flush=True)
        for scenario, mode, extra in [
            ('geometry','geometry',{}), ('annotation','annotation',{}),
            ('annotation_map','annotation',dict(genetic_map=str(data/'map.tsv'),
                map_contract=str(data/'map.json'),max_map_knots=3)),
        ]:
            (base/f'{scenario}.json').write_text(json.dumps(dict(params,mode=mode,**extra)))
            command=['nextflow','-C',str(base/'resources.config'),'run',str(ROOT/'workflows/r02_segment_evidence_single.nf'),
                     '-params-file',str(base/f'{scenario}.json'),'-work-dir',str(base/'work'),'-ansi-log','false']
            for i,resume in enumerate([[],['-resume']]):
                result=subprocess.run(command+['-with-trace',str(base/f'{scenario}{i}.tsv')]+resume,
                    cwd=base,env=env,capture_output=True,text=True,timeout=180)
                (base/f'{scenario}{i}.log').write_text(result.stdout+result.stderr)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                with (base/f'{scenario}{i}.tsv').open() as handle:
                    rows=list(csv.DictReader(handle,delimiter='\t'))
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['status'],'COMPLETED' if i==0 else 'CACHED')
        preflight=list((base/'work').glob('*/*/geometry/preflight.json'))
        evidence=list((base/'work').glob('*/*/segment_evidence/manifest.json'))
        self.assertEqual(len(preflight),1)
        self.assertEqual(len(evidence),2)
        self.assertEqual(json.loads(preflight[0].read_text())['checks']['genotype_records_read'],0)
        self.assertFalse((preflight[0].parent/'manifest.json').exists())
        for path in evidence:
            manifest=json.loads(path.read_text())
            self.assertEqual(manifest['status'],'COMPLETE_DESCRIPTIVE_NOT_VALIDATED')
            with gzip.open(path.parent/'segment_evidence.tsv.gz','rt') as handle:
                rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual(len(rows),1)
            if 'genetic_map' in manifest['inputs']:
                self.assertEqual(rows[0]['map_status'],'OK')
                self.assertEqual(float(rows[0]['length_cm']),1.0)
                self.assertEqual(manifest['parameters']['max_map_knots'],3)
            else:
                self.assertEqual(rows[0]['map_status'],'NO_EVALUABLE:no_map')
                self.assertEqual(rows[0]['length_cm'],'NA')

        # Reject an unpaired input before task submission; no silently ignored map.
        (base/'invalid_map.json').write_text(json.dumps(dict(params,mode='annotation',
            genetic_map=str(data/'map.tsv'))))
        invalid=subprocess.run(['nextflow','-C',str(base/'resources.config'),'run',
            str(ROOT/'workflows/r02_segment_evidence_single.nf'),'-params-file',
            str(base/'invalid_map.json'),'-work-dir',str(base/'work'),'-ansi-log','false'],
            cwd=base,env=env,capture_output=True,text=True,timeout=180)
        self.assertNotEqual(invalid.returncode,0)
        self.assertIn('Map and contract must be supplied together',invalid.stdout+invalid.stderr)


if __name__ == '__main__':
    unittest.main()
