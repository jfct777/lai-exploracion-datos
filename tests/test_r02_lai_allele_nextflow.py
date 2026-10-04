"""Opt-in catalogue-only L1 integration; synthetic fixtures, no cloud writes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:b672cfc6b0c5e4cc7d16c90d453b6f2036b2bed119a54525cec4bd531a1df5d9'


@unittest.skipUnless(os.environ.get('RUN_R02_NEXTFLOW_TEST') == '1', 'Explicit Docker/Nextflow opt-in required')
class LaiCatalogueNextflow(unittest.TestCase):
    def test_correspondence_and_cached_resume(self):
        base=Path(tempfile.mkdtemp(prefix='r02-l1-nextflow-',dir=ROOT/'.claude/runs'))
        setup="""
import sys
sys.path.insert(0,sys.argv[1]+'/tests')
from test_r02_lai_allele_support import CatalogueFixture
CatalogueFixture(sys.argv[2]+'/inputs')
"""
        subprocess.run(['docker','run','--rm','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
            '-v',f'{ROOT}:{ROOT}:ro','-v',f'{base}:{base}',IMAGE,
            'python3','-c',setup,str(ROOT),str(base)],check=True,text=True,capture_output=True,timeout=60)
        data=base/'inputs'
        source=base/'source_bin'
        source.mkdir()
        for name in ('r02_lai_allele_support.py','r02_lai_genotype_support.py','r02_lai_build_verification.py',
                     'r02_genomic_pair_evidence.py','preprocess_storage_guard.py'):
            shutil.copy2(ROOT/'bin'/name,source/name)
        probe=source/'r02_lai_allele_support.py'
        probe.write_text(probe.read_text()+'\n# Synthetic cache probe: A\n')
        inputs=dict(rare_vcf=data/'rare.vcf.gz',rare_contract=data/'rare.contract.json',
                    panel_vcf=data/'panel.vcf.gz',roles=data/'roles.tsv',build_evidence=data/'build.json',
                    contig_map=data/'contigs.json')
        hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in inputs.items()}
        (data/'hashes.json').write_text(json.dumps(hashes))
        params=dict(source_bin=str(source),**{k:str(v) for k,v in inputs.items()},input_hashes=str(data/'hashes.json'),
            chromosome='22',panel_role='REF_TRAIN',expected_source_samples=101,expected_rare_sites=4,
            max_panel_records=1000,max_panel_bytes=1048576,max_panel_index_bytes=1048576,max_line_bytes=8388608,
            min_free_gib=0,resource_check_rows=1)
        (base/'params.json').write_text(json.dumps(params))
        (base/'resources.config').write_text(f"""process.executor = 'local'
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
        env={**os.environ,'NXF_OFFLINE':'true','NXF_DISABLE_CHECK_LATEST':'true','NXF_SYNTAX_PARSER':'v1',
             'NXF_OPTS':'-Xms64m -Xmx512m'}
        command=['nextflow','-C',str(base/'resources.config'),'run',str(ROOT/'workflows/r02_lai_allele_support.nf'),
                 '-params-file',str(base/'params.json'),'-work-dir',str(base/'work'),'-ansi-log','false']
        print('Synthetic L1 integration:',base,flush=True)
        for index,resume in enumerate([[],['-resume'],['-resume']]):
            if index==2:
                metadata=probe.stat()
                probe.write_text(probe.read_text().replace('Synthetic cache probe: A','Synthetic cache probe: B'))
                os.utime(probe,ns=(metadata.st_atime_ns,metadata.st_mtime_ns))
                self.assertEqual(probe.stat().st_size,metadata.st_size)
                self.assertEqual(probe.stat().st_mtime_ns,metadata.st_mtime_ns)
            result=subprocess.run(command+['-with-trace',str(base/f'trace{index}.tsv')]+resume,
                cwd=base,env=env,capture_output=True,text=True,timeout=180)
            (base/f'run{index}.log').write_text(result.stdout+result.stderr)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            with (base/f'trace{index}.tsv').open() as handle:
                rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['status'],'CACHED' if index==1 else 'COMPLETED')
        manifests=list((base/'work').glob('*/*/catalogue_correspondence/manifest.json'))
        self.assertEqual(len(manifests),2)
        for path in manifests:
            manifest=json.loads(path.read_text())
            self.assertEqual(manifest['output_rows'],4)
            self.assertEqual(manifest['genotype_fields_decoded'],0)
            self.assertEqual(manifest['genotype_support'],'NOT_EVALUATED')
        self.assertEqual((manifests[0].parent/'correspondencia_alelos_dnabr.tsv').read_bytes(),
                         (manifests[1].parent/'correspondencia_alelos_dnabr.tsv').read_bytes())


if __name__=='__main__': unittest.main()
