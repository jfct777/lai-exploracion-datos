"""Synthetic, local amendment contracts. No subprocesses or human genotypes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_amendment', ROOT/'bin/r02_apply_amendment.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True) + '\n'
    path.write_text(value)


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.run = self.base/'original'
        self.amend = self.run/'repairs/protocol2'
        self.old = self.run/'source'
        self.new = self.amend/'source'
        source = (ROOT/'bin/r02_autosome_pipeline.py').read_text()
        for folder in (self.old, self.new):
            write(folder/'bin/r02_autosome_pipeline.py', source)
            for name in adapter.PROTECTED:
                write(folder/name, '# identical synthetic source\n')
        write(self.new/'bin/r02_apply_amendment.py', (ROOT/'bin/r02_apply_amendment.py').read_text())
        for name in adapter.ANALYSIS_TOOLS:
            write(self.new/'bin'/name, '# synthetic tool; never executed\n')
        self.config = dict(run_id='synthetic', destination=str(self.base/'destination'),
            raw_dir=str(self.base/'raw'), ref=str(self.base/'reference.fa'), bulk=str(self.base/'bulk'),
            processing_order=[22] + list(range(21, 0, -1)), analytical_samples=2619,
            m14=dict(lengths=[100000, 250000, 500000, 1000000, 2000000],
                     gaps=[25000, 50000, 100000], counts=[5, 10, 20, 50, 100, 200]))
        write(self.run/'run.json', self.config)
        write(self.run/'input_objects.json', {})
        write(self.run/'samples.txt', 'synthetic-only\n')
        write(self.run/'h_settings.json', {'original_six': True})
        self.hash_source(self.old, self.run/'source.sha256.json')
        write(self.run/'frozen.sha256.json', {name: adapter.sha(self.run/name) for name in
            ('run.json', 'input_objects.json', 'samples.txt', 'h_settings.json', 'source.sha256.json')})
        folder = self.run/'chr21'
        write(folder/'parameters.json', adapter.expected_preprocess_parameters(self.run, self.config, 21))
        write(folder/'runtime.config', '# immutable synthetic runtime\n')
        self.outputs = []
        for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv'):
            output = folder/'preprocess/lai_rare'/('dnabr.hg38.2723.chr21.rare.minor' + suffix)
            write(output, 'synthetic tiny bytes\n')
            self.outputs.append(dict(path=str(output), bytes=output.stat().st_size, sha256=adapter.sha(output)))
        self.cp = self.run/'checkpoints/chr21_M01_M02_M021.json'
        write(self.cp, dict(returncode=0, completed_utc='synthetic-time',
            command=adapter.expected_boundary_command(self.run, 21), outputs=self.outputs))
        prior = self.run/'chr22/completed.txt'
        write(prior, 'synthetic completed result\n')
        write(self.run/'checkpoints/chr22_complete.json', dict(chromosome=22,
            outputs=[dict(path=str(prior), bytes=prior.stat().st_size, sha256=adapter.sha(prior))]))
        self.request = dict(schema_version=1, run_dir=str(self.run),
            original_frozen_sha256=adapter.sha(self.run/'frozen.sha256.json'),
            original_source_manifest_sha256=adapter.sha(self.run/'source.sha256.json'),
            overrides=copy.deepcopy(adapter.PROTOCOL_DELTA), boundary=dict(chromosome=21,
                stage='chr21_M01_M02_M021', checkpoint_sha256=adapter.sha(self.cp),
                parameters_sha256=adapter.sha(folder/'parameters.json'),
                runtime_config_sha256=adapter.sha(folder/'runtime.config')))
        self.seal()

    def hash_source(self, folder, manifest):
        write(manifest, {str(p.relative_to(folder)): adapter.sha(p)
                        for p in sorted(folder.rglob('*')) if p.is_file()})

    def seal(self):
        self.hash_source(self.new, self.amend/'source.sha256.json')
        write(self.amend/'amendment.json', self.request)
        write(self.amend/'request.json', {'synthetic_request': True})
        write(self.amend/'frozen.sha256.json', {name: adapter.sha(self.amend/name) for name in
            ('amendment.json', 'source.sha256.json', 'request.json')})

    def change_checkpoint(self, **updates):
        value = json.loads(self.cp.read_text())
        value.update(updates)
        write(self.cp, value)
        self.request['boundary']['checkpoint_sha256'] = adapter.sha(self.cp)
        self.seal()

    def build(self, **kwargs):
        return adapter.build_runner(self.run, self.amend, **kwargs)

    def test_dry_validation_is_read_only_and_keeps_original_inputs(self):
        before = {str(p): p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        runner, _, evidence = self.build()
        after = {str(p): p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(runner.source, self.old)
        self.assertEqual(runner.bin, self.new/'bin')
        self.assertEqual(runner.dest, Path(self.config['destination']))
        self.assertEqual(runner.c['analysis_protocol_version'], 2)
        self.assertEqual(evidence['completed_chromosomes'][0]['chromosome'], 22)

    def test_activation_is_additive_idempotent_and_resumable(self):
        original = {name: (self.run/name).read_bytes() for name in
                    ('run.json', 'source.sha256.json', 'frozen.sha256.json', 'h_settings.json')}
        self.build(activate=True)
        settings = json.loads((self.run/'m14_diagnostic_settings.json').read_text())
        self.assertEqual(len(settings['configurations']), 74)
        self.build(activate=True)
        # Amended work is legal after the initial activation receipt, not before it.
        write(self.run/'chr21/rare_evidence.npz', 'synthetic partial amended output')
        write(self.run/'stages/chr21_rare_J__r02v2/nextflow.log', 'synthetic')
        self.build()
        for name, value in original.items():
            self.assertEqual((self.run/name).read_bytes(), value)

    def test_analysis_namespace_and_preprocess_original_bin(self):
        runner, pipeline, _ = self.build()
        with patch.object(pipeline.Runner, 'docker', return_value='mock') as task:
            self.assertEqual(runner.docker('chr21_rare_J', ['python3', 'fake']), 'mock')
            self.assertEqual(task.call_args.args[0], 'chr21_rare_J__r02v2')
        with patch.object(pipeline.Runner, 'preprocess', side_effect=lambda c: (c, runner.bin)):
            self.assertEqual(runner.preprocess(20), (20, self.old/'bin'))
        self.assertEqual(runner.bin, self.new/'bin')
        with patch.object(pipeline.Runner, 'preprocess', side_effect=ValueError('synthetic')):
            with self.assertRaises(ValueError):
                runner.preprocess(20)
        self.assertEqual(runner.bin, self.new/'bin')

    def test_completed_chr22_reused_without_new_commands(self):
        runner, _, _ = self.build()
        with patch.object(runner, 'preprocess', side_effect=AssertionError('must reuse')):
            runner.analyze_chromosome(22)

    def test_legacy_prepared_rare_j_is_allowed_but_execution_evidence_rejected(self):
        stage = self.run/'stages/chr21_rare_J'
        for name in ('command.json', 'parameters.json', 'runtime.config'):
            write(stage/name, '{}\n')
        self.build()
        write(stage/'nextflow.log', '')
        with self.assertRaisesRegex(ValueError, 'execution evidence'):
            self.build()

    def test_boundary_absent_failed_changed_command_and_missing_output(self):
        for label, updates in (
            ('failed', {'returncode': 1}), ('boolean returncode', {'returncode': False}),
            ('missing outputs', {'outputs': []}), ('wrong command', {'command': ['true']}),
            ('missing time', {'completed_utc': ''})):
            with self.subTest(label=label):
                original = self.cp.read_text()
                self.change_checkpoint(**updates)
                with self.assertRaises(ValueError):
                    self.build()
                write(self.cp, original)
        self.cp.unlink()
        with self.assertRaisesRegex(ValueError, 'not ready'):
            self.build()

    def test_tampered_boundary_output_rejected(self):
        write(Path(self.outputs[0]['path']), 'different')
        with self.assertRaisesRegex(ValueError, 'changed or is missing'):
            self.build()

    def test_resealed_wrong_preprocess_parameters_rejected(self):
        path = self.run/'chr21/parameters.json'
        value = json.loads(path.read_text())
        value['lai_rare_max_maf'] = .02
        write(path, value)
        self.request['boundary']['parameters_sha256'] = adapter.sha(path)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Preprocessing parameters'):
            self.build()

    def test_science_or_aggregate_state_blocks_first_activation(self):
        for name in ('checkpoints/chr21_rare_J.json', 'checkpoints/chr20_M01_M02_M021.json',
                     'checkpoints/publish_01_estructura_desarrollo_por_cromosoma_chr21.json',
                     'checkpoints/all22_H_aggregate.json', 'chr21/rare_evidence.npz',
                     'aggregate_H/results.txt', 'stages/all22_H_aggregate/command.json'):
            with self.subTest(path=name):
                path = self.run/name
                write(path, '{}\n')
                with self.assertRaises(ValueError):
                    self.build()
                path.unlink()
                # Keep later subtests independent of the directories they generated.
                for directory in (self.run/'aggregate_H', self.run/'stages/all22_H_aggregate'):
                    if directory.exists():
                        directory.rmdir()

    def test_out_of_scope_delta_rejected(self):
        self.request['overrides']['m14'] = {'gaps': [123]}
        self.seal()
        with self.assertRaisesRegex(ValueError, 'explicit protocol-2 delta'):
            self.build()

    def test_changed_frozen_code_and_unlisted_code_rejected(self):
        path = self.new/'bin/r02_autosome_pipeline.py'
        original = path.read_text()
        write(path, original + '\n# tampered\n')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            self.build()
        write(path, original)
        write(self.new/'bin/unlisted.py', '# not authenticated\n')
        with self.assertRaisesRegex(ValueError, 'unauthenticated files'):
            self.build()

    def test_resealed_preprocess_source_change_rejected(self):
        write(self.new/adapter.PROTECTED[0], '# changed workflow')
        self.seal()
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.build()

    def test_resealed_runner_preprocess_change_rejected(self):
        path = self.new/'bin/r02_autosome_pipeline.py'
        write(path, path.read_text().replace("outdir=str(folder/'preprocess'), cpus=6", "outdir=str(folder/'preprocess'), cpus=7"))
        self.seal()
        with self.assertRaisesRegex(ValueError, 'preprocessing implementation'):
            self.build()

    def test_original_configuration_change_rejected(self):
        write(self.run/'run.json', dict(self.config, analytical_samples=2))
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            self.build()

    def test_path_escape_and_symlink_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe manifest path'):
            adapter.verify_manifest(self.new, {'../escape': 'a'*64})
        path = self.new/'bin/select_rare_minor.py'
        path.unlink()
        path.symlink_to(self.old/'bin/select_rare_minor.py')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.build()

    def test_activation_tamper_and_conflicting_settings_rejected(self):
        self.build(activate=True)
        write(self.amend/'resolved_configuration.json', {'wrong': True})
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            self.build()

    def test_conflicting_diagnostic_settings_rejected(self):
        write(self.run/'m14_diagnostic_settings.json', {'wrong': True})
        with self.assertRaisesRegex(ValueError, 'diagnostic settings conflict'):
            self.build()

    def test_live_lock_blocks_run_before_activation(self):
        _, pipeline, _ = self.build()
        with pipeline.execution_lock(self.run):
            with self.assertRaisesRegex(RuntimeError, 'already owns'):
                adapter.main(['--run-dir', str(self.run), '--amendment-dir', str(self.amend), '--run'])
        self.assertFalse((self.amend/'activation.json').exists())

    def test_run_activates_and_calls_original_execution_under_lock(self):
        frozen = adapter.validate_snapshot(self.run, self.amend)
        pipeline = frozen[1]
        observations = []

        def execute(runner):
            self.assertTrue((self.amend/'activation.json').is_file())
            with self.assertRaisesRegex(RuntimeError, 'already owns'):
                with pipeline.execution_lock(self.run):
                    pass
            observations.append((runner.run, runner.bin, runner.source))

        with patch.object(adapter, 'validate_snapshot', return_value=frozen), \
             patch.object(pipeline.Runner, 'execute', execute):
            adapter.main(['--run-dir', str(self.run), '--amendment-dir', str(self.amend), '--run'])
        self.assertEqual(observations, [(self.run, self.new/'bin', self.old)])

    def test_duplicate_json_key_rejected(self):
        write(self.amend/'duplicate.json', '{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            adapter.load_json(self.amend/'duplicate.json')


if __name__ == '__main__':
    unittest.main()
