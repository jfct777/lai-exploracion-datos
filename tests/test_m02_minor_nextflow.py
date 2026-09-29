"""Opt-in local Nextflow execution of real M01 -> M02 -> M02.1 processes.

No human inputs or cloud execution. Requires locally built dnabr-m02-minor image.
RUN_M02_NEXTFLOW=1 python3 -m unittest tests.test_m02_minor_nextflow -v
"""
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('RUN_M02_NEXTFLOW') == '1' and shutil.which('nextflow'),
                     'opt-in synthetic local Nextflow/Docker integration')
class NextflowMinorContractTests(unittest.TestCase):
    def test_presentation_config_only_redirects_new_outputs(self):
        with tempfile.TemporaryDirectory(prefix='m02-presentation-config-') as folder:
            base = Path(folder)
            config = base / 'presentation-test.config'
            config.write_text(f'''
params.outdir = '/tmp/old-output'
params.painting_results_dir = '/tmp/old-output/14_rare_allele_sharing_painting'
params.feature_build_results_dir = '/tmp/old-output/20_feature_store'
params.painting_input_dir = '/tmp/unchanged-input'
params.presentation_category = 'smokes'
params.presentation_run_id = 'synthetic-config-test'
includeConfig '{ROOT}/conf/presentation.config'
''')
            result = subprocess.run(['nextflow', '-C', str(config), 'config', '-flat'],
                                    cwd=base, capture_output=True, text=True, timeout=60,
                                    env={**os.environ, 'NXF_SYNTAX_PARSER': 'v1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('presentacion/smokes/synthetic-config-test/20_feature_store', result.stdout)
            self.assertIn('presentacion/smokes/synthetic-config-test/14_rare_allele_sharing_painting', result.stdout)
            self.assertIn("params.painting_input_dir = '/tmp/unchanged-input'", result.stdout)

    def test_main_workflow_initializes_with_legacy_parser_without_jobs(self):
        # main.nf predates the v2 syntax parser; no migration is implied here.
        with tempfile.TemporaryDirectory(prefix='m02-minor-init-') as folder:
            base = Path(folder)
            env = {**os.environ, 'NXF_OFFLINE': 'true', 'NXF_DISABLE_CHECK_LATEST': 'true',
                   'NXF_SYNTAX_PARSER': 'v1'}
            result = subprocess.run([
                'nextflow', '-log', str(base / 'nextflow.log'), 'run', str(ROOT / 'main.nf'),
                '-work-dir', str(base / 'work'), '-ansi-log', 'false',
                '--run_qc', 'false', '--run_downstream', 'false',
                '--outdir', str(base / 'outputs'), '--ref_fasta', str(base / 'unused.fa'),
            ], cwd=base, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_process_chain_and_staged_sample_selection(self):
        with tempfile.TemporaryDirectory(prefix='m02-minor-nf-') as folder:
            base = Path(folder)
            (base / 'ref.fa').write_text('>chr22\n' + 'A' * 1000 + '\n')
            samples = [f's{i}' for i in range(50)]
            (base / 'keep.txt').write_text('\n'.join(samples) + '\n')
            header = ('##fileformat=VCFv4.2\n##contig=<ID=chr22,length=1000>\n'
                      '##FILTER=<ID=q10,Description="Rejected fixture">\n'
                      '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                      '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples) + '\n')
            rows = []
            for pos, alt, filt, gts in [
                (100, 'C', 'PASS', ['0/1'] * 2 + ['0/0'] * 48),
                (200, 'C', 'PASS', ['0/1'] * 2 + ['1/1'] * 48),
                (300, 'C,G', 'PASS', ['0/1'] * 2 + ['0/0'] * 48),
                (400, 'C', 'PASS', ['0/1'] * 2 + ['0/0'] * 48),
                (400, 'G', 'q10', ['0/1'] * 2 + ['0/0'] * 48),
            ]:
                rows.append(f'chr22\t{pos}\t.\tA\t{alt}\t.\t{filt}\t.\tGT\t' + '\t'.join(gts))
            (base / 'input.vcf').write_text(header + '\n'.join(rows) + '\n')
            (base / 'nf.config').write_text(f'''
process.executor = 'local'
process.container = 'dnabr-m02-minor:20260922'
docker.enabled = true
docker.runOptions = '--network none --user {os.getuid()}:{os.getgid()}'
params.outdir = '{base}/outputs'
params.cpus = 1
params.memory = '1 GB'
params.time = '5m'
params.resources = [:]
params.bcftools_min_alleles = 2
params.plink_max_alleles = 2
params.plink_snps_only = true
params.max_maf = null
params.keep_pass = true
params.lai_rare_max_maf = 0.03
params.lai_rare_min_mac = 2
params.lai_rare_remove_info = false
params.lai_rare_keep_format = 'GT'
''')
            (base / 'test.nf').write_text(f'''
nextflow.enable.dsl=2
include {{ PREPROCESS_NORM_LEFTALIGN }} from '{ROOT}/modules/01_preprocess_norm_leftalign'
include {{ PREPROCESS_FILTER_SNV_BIALLELIC_PASS }} from '{ROOT}/modules/02_preprocess_filter_snv_biallelic_pass'
include {{ LAI_RARE_BIALELIC_ONLY }} from '{ROOT}/modules/lai_rare_bialelic_only'
include {{ VALIDATE_RARE_CONSUMERS }} from '{ROOT}/modules/02_1_VALIDATE_RARE_CONSUMERS'
process PACK {{
    input: path vcf
    output: tuple val('22'), path('input.vcf.gz'), path('input.vcf.gz.tbi')
    script:
    """
    bcftools view -Oz -o input.vcf.gz ${{vcf}}
    bcftools index -t input.vcf.gz
    """
}}
workflow {{
    raw = PACK(file('{base}/input.vcf'))
    norm = PREPROCESS_NORM_LEFTALIGN(raw.combine(Channel.value(file('{base}/ref.fa'))), file('{ROOT}/bin/mark_original_alleles.py'))
    filtered = PREPROCESS_FILTER_SNV_BIALLELIC_PASS(raw.join(norm[0]))
    rare = LAI_RARE_BIALELIC_ONLY(filtered.map {{ c, counts, v, idx -> tuple(c,v,idx) }}, file('{base}/keep.txt'), file('{ROOT}/bin/select_rare_minor.py'))
    VALIDATE_RARE_CONSUMERS(rare.rare_vcfs, '', 'source_minor', file('{ROOT}/bin/validate_rare_contract.py'))
}}
''')
            env = {**os.environ, 'NXF_OFFLINE': 'true', 'NXF_DISABLE_CHECK_LATEST': 'true'}
            run = subprocess.run(['nextflow', '-log', str(base / 'nextflow.log'), '-C', str(base / 'nf.config'),
                                  'run', str(base / 'test.nf'), '-work-dir', str(base / 'work'), '-ansi-log', 'false'],
                                 cwd=base, env=env, capture_output=True, text=True, timeout=180)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            out = base / 'outputs' / 'lai_rare' / 'dnabr.hg38.2723.chr22.rare.minor.vcf.gz'
            self.assertTrue(Path(str(out) + '.tbi').exists())
            with gzip.open(out, 'rt') as handle:
                data = [line.rstrip().split('\t') for line in handle if not line.startswith('#')]
            self.assertEqual([int(row[1]) for row in data], [100, 200])
            self.assertIn('RARE_ALLELE=0', data[1][7])
            self.assertEqual(data[1][10], '0/1:1')
            self.assertEqual(data[1][11], '1/1:0')
            report = json.loads(out.with_name(out.name.replace('.vcf.gz', '.contract.json')).read_text())
            self.assertEqual(report['counts']['excluded_original_multiallelic'], 3)
            self.assertEqual(report['cohort_samples_after'], 50)


if __name__ == '__main__':
    unittest.main()
