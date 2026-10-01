"""Synthetic amendment checks: no VCF, cloud, process signals or VM operations."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('optimized_worker', ROOT/'bin/r02_optimized_worker.py')
OPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPT)


class OptimizedWorkerTests(unittest.TestCase):
    def fixture(self, root, *, single=False):
        run = root/'worker01' if single else root/'worker05'
        folder = run/'repairs/preprocess-v1'
        (run/'source/bin').mkdir(parents=True)
        (run/'checkpoints').mkdir()
        folder.mkdir(parents=True)
        assigned = [2] if single else [5, 20]
        (run/'samples.txt').write_text('synthetic_a\nsynthetic_b\n')
        for name in ('input_objects.json', 'h_settings.json', 'weighted_settings_primary.json',
                     'weighted_settings_sensitivity.json', 'm14_diagnostic_settings.json'):
            OPT.fixed(run/name, {})
        shutil.copy2(ROOT/'bin/r02_parallel_worker.py', run/'worker.py')
        shutil.copy2(ROOT/'bin/r02_autosome_pipeline.py', run/'source/bin/r02_autosome_pipeline.py')
        (run/'source/bin/mark_original_alleles.py').write_text('# original fixture\n')
        (run/'source/bin/scientific_unchanged.py').write_text('# scientific fixture\n')
        old_source = {str(p.relative_to(run/'source')): OPT.sha(p)
                      for p in (run/'source').rglob('*') if p.is_file()}
        OPT.fixed(run/'source.sha256.json', old_source)
        config = dict(run_id='fixture-' + run.name, destination=str(root/'published'),
            bulk=str(root/'local-bulk'), samples_sha256=OPT.sha(run/'samples.txt'),
            processing_order=assigned,
            parallel_worker=dict(worker_id=run.name, assigned_chromosomes=assigned,
                                 aggregate_allowed=False, parent_run_id='fixture'))
        worker = OPT.load_module(run/'worker.py', OPT.sha(run/'worker.py'), '_fixture_original_worker')
        config.update(worker.PROTOCOL_DELTA)
        OPT.fixed(run/'run.json', config)
        OPT.fixed(run/'frozen.sha256.json', {str(p.relative_to(run)): OPT.sha(p)
                  for p in run.iterdir() if p.is_file()})
        shutil.copytree(run/'source', folder/'source')
        for name in OPT.ALLOWED_SOURCE_DELTA - {'bin/r02_autosome_pipeline.py'}:
            target = folder/'source'/name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_text('# new operational fixture\n')
        new_source = {str(p.relative_to(folder/'source')): OPT.sha(p)
                      for p in (folder/'source').rglob('*') if p.is_file()}
        OPT.fixed(folder/'source.sha256.json', new_source)
        spec = dict(schema=OPT.SCHEMA, run_dir=str(run), worker_id=run.name,
            wrapper_sha256=OPT.sha(OPT.__file__), original_run_sha256=OPT.sha(run/'run.json'),
            original_frozen_sha256=OPT.sha(run/'frozen.sha256.json'),
            original_source_manifest_sha256=OPT.sha(run/'source.sha256.json'),
            source_manifest_sha256=OPT.sha(folder/'source.sha256.json'),
            new_preprocess_chromosomes=[] if single else [20], legacy_chromosomes=assigned[:1],
            overrides=OPT.OVERRIDES.copy(), old_processes={key: dict(pid=90000+i,
                start_ticks=1, cmdline_sha256='a'*64) for i, key in enumerate(('supervisor', 'child', 'startup'))})
        OPT.fixed(folder/'manifest.json', spec)
        return run, folder, spec, worker

    def check(self, folder):
        return OPT.validate_spec(folder/'manifest.json', OPT.sha(folder/'manifest.json'))

    def rewrite(self, folder, spec):
        (folder/'manifest.json').write_text(json.dumps(spec))

    def test_valid_manifest_leaves_original_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, folder, spec, _ = self.fixture(Path(tmp))
            seal = OPT.sha(run/'frozen.sha256.json')
            observed = self.check(folder)
            self.assertEqual(observed[0], spec)
            self.assertEqual(OPT.sha(run/'frozen.sha256.json'), seal)
            self.assertFalse((folder/'activation.json').exists())

    def test_single_legacy_worker_can_have_no_future_chromosome(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, _, _ = self.fixture(Path(tmp), single=True)
            self.assertEqual(self.check(folder)[0]['new_preprocess_chromosomes'], [])

    def test_changed_hash_or_scientific_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, spec, _ = self.fixture(Path(tmp))
            source = folder/'source/bin/scientific_unchanged.py'
            source.write_text('# changed science\n')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                self.check(folder)
            sources = OPT.read(folder/'source.sha256.json')
            sources['bin/scientific_unchanged.py'] = OPT.sha(source)
            (folder/'source.sha256.json').write_text(json.dumps(sources))
            spec['source_manifest_sha256'] = OPT.sha(folder/'source.sha256.json')
            self.rewrite(folder, spec)
            with self.assertRaisesRegex(ValueError, 'allowlist'):
                self.check(folder)

    def test_original_configuration_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, folder, _, _ = self.fixture(Path(tmp))
            (run/'run.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Original run changed'):
                self.check(folder)

    def test_changed_scientific_runner_method_fails_ast_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, folder, _, _ = self.fixture(Path(tmp))
            new = folder/'source/bin/r02_autosome_pipeline.py'
            new.write_text(new.read_text().replace('def analyze_chromosome(self,c):',
                                                    'def analyze_chromosome(self,c):\n        raise ValueError("changed")'))
            with self.assertRaisesRegex(ValueError, 'Scientific Runner method changed'):
                OPT.validate_pipeline_delta(run/'source/bin/r02_autosome_pipeline.py', new)

    def test_rejects_overlap_wrong_order_extra_config_and_started_new_chromosome(self):
        for failure in ('overlap', 'empty_legacy', 'extra_parameter', 'started'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                run, folder, spec, _ = self.fixture(Path(tmp))
                if failure == 'overlap': spec['new_preprocess_chromosomes'] = [5, 20]
                if failure == 'empty_legacy': spec['legacy_chromosomes'] = []
                if failure == 'extra_parameter': spec['overrides']['min_mac'] = 1
                if failure == 'started': (run/'chr20').mkdir()
                self.rewrite(folder, spec)
                with self.assertRaises(ValueError): self.check(folder)

    def test_override_is_temporarily_applied_and_restored_on_success_and_error(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                run, folder, spec, _ = self.fixture(Path(tmp))
                events = []
                class OldRunner:
                    def __init__(self, path):
                        self.source = path/'source'; self.bin = self.source/'bin'
                        self.c = {'nested': {'value': 1}}
                    def preprocess(self, c):
                        events.append(('old', c, self.source)); return 'legacy'
                class NewRunner:
                    def preprocess(self, c):
                        events.append(('new', c, self.source))
                        assert self.c['preprocess_checkpointed_m01'] is True
                        self.c['nested']['value'] = 2
                        if failure: raise RuntimeError('fixture')
                        return 'new'
                runner = OPT.build_runner(run, folder, spec, types.SimpleNamespace(Runner=OldRunner),
                                          types.SimpleNamespace(Runner=NewRunner))
                before = runner.source, runner.bin, runner.c
                self.assertEqual(runner.preprocess(5), 'legacy')
                if failure:
                    with self.assertRaises(RuntimeError): runner.preprocess(20)
                else: self.assertEqual(runner.preprocess(20), 'new')
                self.assertEqual((runner.source, runner.bin, runner.c), before)
                self.assertIs(runner.c, before[2])
                self.assertEqual(runner.c['nested']['value'], 1)
                self.assertEqual(events, [('old', 5, run/'source'), ('new', 20, folder/'source')])
                with self.assertRaisesRegex(ValueError, 'Unassigned'): runner.preprocess(21)

    def test_alive_predecessor_or_container_prevents_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, folder, spec, _ = self.fixture(Path(tmp))
            with patch.object(OPT, 'old_process_live', return_value=True):
                with self.assertRaisesRegex(ValueError, 'remains alive'):
                    OPT.validate_idle(spec, {'run_id': 'fixture'})
            with patch.object(OPT, 'old_process_live', return_value=False), patch.object(
                OPT.subprocess, 'check_output', return_value='active-container\n'
            ):
                with self.assertRaisesRegex(ValueError, 'active containers'):
                    OPT.validate_idle(spec, {'run_id': 'fixture'})

    def test_run_receipt_identifies_both_sources_and_publishes_operational_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, folder, spec, worker = self.fixture(Path(tmp))
            original_config = (run/'run.json').read_bytes()
            old = worker.load_pipeline(run/'source')
            calls = []
            class FakeRunner:
                def __init__(self, path):
                    self.run = path; self.source = path/'source'; self.bin = self.source/'bin'
                    self.c = OPT.read(path/'run.json'); self.current_stage = 'fixture'
                def preprocess(self, c): calls.append(('old', c, self.source))
                def analyze_chromosome(self, c):
                    checkpoint = self.run/'checkpoints'/f'chr{c:02d}_complete.json'
                    if checkpoint.exists(): return
                    self.preprocess(c)
                    chromosome = self.run/f'chr{c:02d}'; chromosome.mkdir()
                    result = chromosome/'fixture.tsv'; result.write_text('synthetic\n')
                    self.publish(chromosome, f'01_estructura/desarrollo/por_cromosoma/chr{c:02d}')
                    OPT.fixed(checkpoint, {'outputs': old.files_manifest([result])})
                def publish(self, source, relative):
                    receipt = self.run/'checkpoints'/('publish_' + relative.replace('/', '_') + '.json')
                    if receipt.exists(): return
                    records = []
                    for p in sorted(source.rglob('*')):
                        if p.is_file(): records.append(dict(path=str(p), bytes=p.stat().st_size,
                            uri='gs://fixture/' + relative + '/' + str(p.relative_to(source)), generation='1',
                            sha256=OPT.sha(p), md5_base64=base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()))
                    OPT.fixed(receipt, {'files': records})
            class FakeNew:
                def preprocess(self, c):
                    calls.append(('new', c, self.source))
                    assert self.c['preprocess_checkpointed_m01'] is True
            old.Runner = FakeRunner
            modules = {'_original_worker': worker, '_original_pipeline': old,
                       '_optimized_preprocess_pipeline': types.SimpleNamespace(Runner=FakeNew)}
            with patch.object(OPT, 'load_module', side_effect=lambda p, h, name: modules[name]), patch.object(
                OPT, 'validate_idle'
            ):
                result = OPT.run_worker(folder/'manifest.json', OPT.sha(folder/'manifest.json'))
                repeated = OPT.run_worker(folder/'manifest.json', OPT.sha(folder/'manifest.json'))
            self.assertEqual(result, repeated)
            self.assertEqual(calls, [('old', 5, run/'source'), ('new', 20, folder/'source')])
            self.assertEqual((run/'run.json').read_bytes(), original_config)
            self.assertEqual(result['source_manifest_sha256'], spec['original_source_manifest_sha256'])
            self.assertEqual(result['operational_amendment']['source_manifest_sha256'], spec['source_manifest_sha256'])
            self.assertEqual(result['chromosomes'], [5, 20])
            self.assertFalse(result['aggregate_executed'])
            required = {'manifest.json', 'source.sha256.json', 'optimized_worker.py'}
            self.assertTrue(required <= {r['uri'].rsplit('/', 1)[-1] for r in result['operational_provenance']})
            self.assertTrue((run/'checkpoints/publish_00_worker_provenance.json').is_file())
            self.assertEqual(OPT.read(run/'status.json')['state'], 'COMPLETE')
            finished = OPT.read(folder/'completed.json')
            self.assertTrue(finished['published'])
            self.assertEqual(finished['manifest_sha256'], OPT.sha(folder/'manifest.json'))
            self.assertEqual(finished['worker_completion_sha256'],
                             OPT.sha(run/'worker_provenance/worker_completion.json'))


if __name__ == '__main__':
    unittest.main()
