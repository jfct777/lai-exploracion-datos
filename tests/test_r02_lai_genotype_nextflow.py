"""Opt-in synthetic L1 genotype integration: deep resume and QC invalidation."""
import csv
import gzip
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
class LaiGenotypeNextflow(unittest.TestCase):
    def test_genotype_cached_resume_and_quality_change(self):
        base = Path(tempfile.mkdtemp(prefix='r02-l1-genotype-nextflow-', dir=ROOT/'.claude/runs'))
        setup = """
import sys
sys.path.insert(0,sys.argv[1]+'/tests')
from test_r02_lai_genotype_support import GenotypeFixture
GenotypeFixture(sys.argv[2]+'/inputs')
"""
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--cpus', '1', '--memory', '2g',
                        '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                        '-e', 'OPENBLAS_NUM_THREADS=1', '-v', f'{ROOT}:{ROOT}:ro', '-v', f'{base}:{base}',
                        IMAGE, 'python3', '-c', setup, str(ROOT), str(base)], check=True, text=True,
                       capture_output=True, timeout=60)
        data = base/'inputs'
        source = base/'source_bin'
        source.mkdir()
        for name in ('r02_lai_allele_support.py','r02_lai_genotype_support.py','r02_lai_build_verification.py',
                     'r02_genomic_pair_evidence.py','preprocess_storage_guard.py'):
            shutil.copy2(ROOT/'bin'/name, source/name)
        probe = source/'r02_lai_genotype_support.py'
        probe.write_text(probe.read_text()+'\n# Synthetic cache probe: A\n')
        inputs = dict(rare_vcf=data/'rare.vcf.gz', rare_contract=data/'rare.contract.json',
                      panel_vcf=data/'genotype_panel.vcf', roles=data/'roles.tsv', build_evidence=data/'build.json',
                      contig_map=data/'contigs.json', genotype_contract=data/'genotype_contract.json',
                      source_build_verification=data/'source_build_proof.json', panel_build_verification=data/'panel_build_proof.json')
        # The real audit producer emits the same basename for both VCFs.
        # Keep distinct provenance by input role, not by caller-side renaming.
        for role in ('source', 'panel'):
            name = role + '_build_verification'
            folder = data/role
            folder.mkdir()
            target = folder/'reference_verification.json'
            shutil.copy2(inputs[name], target)
            inputs[name] = target
        self.assertEqual(inputs['source_build_verification'].name,
                         inputs['panel_build_verification'].name)
        self.assertNotEqual(inputs['source_build_verification'].read_bytes(),
                            inputs['panel_build_verification'].read_bytes())
        (data/'hashes.json').write_text(json.dumps({
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}))
        params = dict(mode='genotype_support', source_bin=str(source), **{k: str(v) for k, v in inputs.items()},
                      input_hashes=str(data/'hashes.json'), chromosome='22', panel_role='REF_TRAIN',
                      expected_source_samples=101, expected_rare_sites=4, max_panel_records=1000,
                      max_panel_bytes=1048576, max_panel_index_bytes=1048576, max_line_bytes=8388608,
                      min_free_gib=0, resource_check_rows=1)
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
        env = {**os.environ, 'NXF_OFFLINE': 'true', 'NXF_DISABLE_CHECK_LATEST': 'true', 'NXF_SYNTAX_PARSER': 'v1',
               'NXF_OPTS': '-Xms64m -Xmx512m'}
        command = ['nextflow', '-C', str(base/'resources.config'), 'run', str(ROOT/'workflows/r02_lai_allele_support.nf'),
                   '-params-file', str(base/'params.json'), '-work-dir', str(base/'work'), '-ansi-log', 'false']
        print('Synthetic L1 genotype integration:', base, flush=True)
        for index in range(4):
            if index == 2:
                quality = json.loads(inputs['genotype_contract'].read_text())
                quality['quality']['min_dp'] = 30
                quality['quality']['contract_id'] = 'synthetic_dp30_gq20'
                inputs['genotype_contract'].write_text(json.dumps(quality))
                hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
                (data/'hashes.json').write_text(json.dumps(hashes))
            if index == 3:
                metadata = probe.stat()
                probe.write_text(probe.read_text().replace('Synthetic cache probe: A','Synthetic cache probe: B'))
                os.utime(probe,ns=(metadata.st_atime_ns,metadata.st_mtime_ns))
                self.assertEqual(probe.stat().st_size,metadata.st_size)
                self.assertEqual(probe.stat().st_mtime_ns,metadata.st_mtime_ns)
            result = subprocess.run(command + ['-with-trace', str(base/f'trace{index}.tsv')] + ([] if index == 0 else ['-resume']),
                                    cwd=base, env=env, capture_output=True, text=True, timeout=180)
            (base/f'run{index}.log').write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (base/f'trace{index}.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'CACHED' if index == 1 else 'COMPLETED')
        manifests = list((base/'work').glob('*/*/genotype_support/manifest.json'))
        self.assertEqual(len(manifests), 3)
        for path in manifests:
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest['output_rows'], 12)
            self.assertEqual(manifest['genotype_fields_decoded'], 16)
            self.assertEqual(manifest['package_versions']['pysam'], '0.23.3')
            threshold = manifest['genotype_contract']['quality']['min_dp']
            with gzip.open(path.parent/'soporte_alelos_dnabr.tsv.gz', 'rt') as handle:
                row = next(row for row in csv.DictReader(handle, delimiter='\t')
                           if row['position_bp'] == '10' and row['panel_ancestry'] == 'ALL')
            self.assertEqual(row['an_evaluable'], '4' if threshold == 10 else '0')
            self.assertEqual(row['ac_counted_evaluable'], '1' if threshold == 10 else 'NA')


if __name__ == '__main__':
    unittest.main()
