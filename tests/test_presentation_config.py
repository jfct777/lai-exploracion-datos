"""Evaluate presentation destinations with Nextflow config, never run processes.

No Docker, GPU, input data or cloud access is needed. Fixtures and Nextflow logs
live in fresh temporary directories, not historical run directories.
"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = ROOT / 'conf' / 'presentation.config'
PREFIX = 'gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion'


def groovy_literal(value):
    if value is None:
        return 'null'
    return "'" + str(value).replace('\\', '\\\\').replace("'", "\\'") + "'"


@unittest.skipUnless(shutil.which('nextflow'), 'Nextflow is required for config-only tests')
class PresentationConfigTests(unittest.TestCase):
    def evaluate(self, overrides=None, include_project=False):
        params = {
            'presentation_run_id': 'synthetic-run_01',
            'painting_input_dir': '/tmp/unchanged-input',
            'm14_followup_rare_vcf': '/tmp/unchanged-rare.vcf.gz',
            'm14_followup_pcrelate': '/tmp/unchanged-pcrelate.tsv',
            'm14_followup_results_dir': '/tmp/old-followup-output',
        }
        params.update(overrides or {})
        with tempfile.TemporaryDirectory(prefix='presentation-config-only-') as folder:
            base = Path(folder)
            config = base / 'test.config'
            lines = [f'includeConfig {groovy_literal(ROOT / "nextflow.config")}'] if include_project else []
            lines += [f'params.{key} = {groovy_literal(value)}' for key, value in params.items()]
            lines += ["workDir = '/tmp/unchanged-work'", f'includeConfig {groovy_literal(PRESENTATION)}',
                      'docker.enabled = false', 'singularity.enabled = false']
            config.write_text('\n'.join(lines) + '\n')
            result = subprocess.run(
                ['nextflow', '-log', str(base/'nextflow.log'), '-C', str(config), 'config', '-flat'],
                cwd=base, capture_output=True, text=True, timeout=90,
                env={**os.environ, 'NXF_SYNTAX_PARSER': 'v1', 'NXF_OFFLINE': 'true',
                     'NXF_DISABLE_CHECK_LATEST': 'true'})
            self.assertFalse((base/'work').exists(), 'config-only checks must not create task work')
            return result

    def assert_destination(self, result, relative):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        root = f'{PREFIX}/{relative}'
        for key, suffix in [('outdir', ''), ('m14_followup_results_dir', ''),
                            ('painting_results_dir', '/14_rare_allele_sharing_painting'),
                            ('feature_build_results_dir', '/20_feature_store')]:
            self.assertIn(f"params.{key} = '{root}{suffix}'", result.stdout)
        for key, value in [('painting_input_dir', '/tmp/unchanged-input'),
                           ('m14_followup_rare_vcf', '/tmp/unchanged-rare.vcf.gz'),
                           ('m14_followup_pcrelate', '/tmp/unchanged-pcrelate.tsv')]:
            self.assertIn(f"params.{key} = '{value}'", result.stdout)
        self.assertIn("workDir = '/tmp/unchanged-work'", result.stdout)

    def test_legacy_flat_default_category(self):
        self.assert_destination(self.evaluate(), 'smokes/synthetic-run_01')

    def test_legacy_flat_with_project_config(self):
        self.assert_destination(self.evaluate(include_project=True), 'smokes/synthetic-run_01')

    def test_legacy_flat_explicit_null_pair_and_biologico(self):
        self.assert_destination(self.evaluate({
            'presentation_category': 'biologico', 'presentation_family_id': None,
            'presentation_stage_id': None}), 'biologico/synthetic-run_01')

    def test_family_stage_and_run_with_project_config(self):
        self.assert_destination(self.evaluate({
            'presentation_category': 'biologico', 'presentation_family_id': 'M14-family_01',
            'presentation_stage_id': '03-relatedness'}, include_project=True),
            'biologico/M14-family_01/03-relatedness/synthetic-run_01')

    def test_component_length_boundary(self):
        identifier = 'a' + '0' * 127
        self.assert_destination(self.evaluate({
            'presentation_family_id': identifier, 'presentation_stage_id': 's',
            'presentation_run_id': identifier}), f'smokes/{identifier}/s/{identifier}')

    def check_rejections(self, cases):
        # Two small config evaluators at a time; no workflow or task is launched.
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self.evaluate, [params for params, _ in cases]))
        for (params, expected), result in zip(cases, results):
            with self.subTest(params=params):
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(expected, result.stdout + result.stderr)

    def test_partial_and_empty_family_stage_are_rejected(self):
        cases = []
        for family, stage in [('family', None), (None, 'stage'), ('', None), (None, '')]:
            cases.append(({'presentation_family_id': family, 'presentation_stage_id': stage},
                          'deben definirse juntas'))
        for family, stage, field in [('', '', 'presentation_family_id'),
                                     ('', 'stage', 'presentation_family_id'),
                                     ('family', '', 'presentation_stage_id')]:
            cases.append(({'presentation_family_id': family, 'presentation_stage_id': stage}, field))
        self.check_rejections(cases)

    def test_unsafe_components_are_rejected(self):
        cases = []
        for field in ['presentation_run_id', 'presentation_family_id', 'presentation_stage_id']:
            for value in ['.', '..', '../escape', 'parent/child', 'a' * 129]:
                params = {'presentation_family_id': 'family', 'presentation_stage_id': 'stage',
                          field: value}
                cases.append((params, f'{field} debe ser un nombre simple seguro'))
        for field, value in [('presentation_run_id', None), ('presentation_run_id', ''),
                             ('presentation_family_id', 'family\\child'),
                             ('presentation_stage_id', 'white space'),
                             ('presentation_stage_id', 'stage.name')]:
            cases.append(({'presentation_family_id': 'family', 'presentation_stage_id': 'stage',
                           field: value}, f'{field} debe ser un nombre simple seguro'))
        self.check_rejections(cases)

    def test_unknown_or_traversing_category_is_rejected(self):
        self.check_rejections([({'presentation_category': category},
                                'presentation_category debe ser smokes o biologico')
                               for category in ['science', '../biologico']])


if __name__ == '__main__':
    unittest.main()
