"""Synthetic orchestration checks; no human genotypes, GCS calls or VM creation."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_parallel_worker', ROOT/'bin/r02_parallel_worker.py')
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def fixture(self, root):
        parent, snapshot, run = root/'parent', root/'amendment', root/'worker'
        parent.mkdir(); snapshot.mkdir(); (parent/'checkpoints').mkdir()
        for base in (parent, snapshot):
            (base/'source/bin').mkdir(parents=True)
            (base/'source/modules').mkdir()
            shutil.copy2(ROOT/'bin/r02_autosome_pipeline.py', base/'source/bin/r02_autosome_pipeline.py')
            (base/'source/modules/example.nf').write_text('process FIXTURE {}\n')
            worker.write_new(base/'source.sha256.json', {str(p.relative_to(base/'source')): worker.sha(p)
                for p in (base/'source').rglob('*') if p.is_file()})
        (parent/'samples.txt').write_text('a\nb\n')
        for name in worker.COPIED_INPUTS[1:]:
            worker.write_new(parent/name, {})
        config = dict(run_id='r02-fixture', chromosomes=list(range(1, 23)),
            processing_order=list(range(22, 0, -1)), destination=str(root/'published'), bulk=str(root/'bulk'),
            container_cpus=6, container_memory_gib=24, samples_sha256=worker.sha(parent/'samples.txt'),
            analytical_samples=2, m14=dict(gaps=[50000], lengths=[250000], counts=[10, 20]),
            prep_image='sha256:'+'a'*64, analysis_image='sha256:'+'b'*64,
            rare_definition='minor source', common_definition='common analytical',
            common_ld={'primary': [500, 1, .2], 'sensitivity': [200, 1, .5]})
        worker.write_new(parent/'run.json', config)
        worker.write_new(parent/'frozen.sha256.json', {name: worker.sha(parent/name)
            for name in (*worker.COPIED_INPUTS, 'run.json', 'source.sha256.json')})
        worker.write_new(snapshot/'request.json', dict(
            amendment_source_manifest_sha256=worker.sha(snapshot/'source.sha256.json'),
            amendment_template=dict(run_dir=str(parent), original_frozen_sha256=worker.sha(parent/'frozen.sha256.json'),
                original_source_manifest_sha256=worker.sha(parent/'source.sha256.json'), overrides=worker.PROTOCOL_DELTA)))
        return parent, snapshot, run, config

    def prepare(self, root, assigned='1,20'):
        parent, snapshot, run, config = self.fixture(root)
        worker.prepare(parent, snapshot, run, 'worker01', assigned,
                       Path(config['destination'])/'parallel/worker01', Path(config['bulk'])/'parallel/worker01')
        return parent, snapshot, run, config

    def test_chromosome_contract_excludes_existing_and_duplicates(self):
        self.assertEqual(worker.chromosomes('1,20'), [1, 20])
        for bad in ('', '21', '22', '0', '1,1', '2,23', 'a', '-1'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                worker.chromosomes(bad)

    def test_preparation_preserves_science_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, snapshot, run, old = self.prepare(Path(tmp))
            new = worker.read_json(run/'run.json')
            allowed = {'run_id', 'created_utc', 'destination', 'bulk', 'processing_order', 'parallel_worker',
                       *worker.PROTOCOL_DELTA}
            self.assertEqual({k: v for k, v in old.items() if k not in allowed},
                             {k: v for k, v in new.items() if k not in allowed})
            self.assertEqual(worker.read_json(parent/'run.json'), old)
            self.assertEqual(new['processing_order'], [1, 20])
            self.assertFalse(new['parallel_worker']['aggregate_allowed'])
            worker.verify_manifest(run, worker.read_json(run/'frozen.sha256.json'))
            self.assertEqual(worker.sha(run/'samples.txt'), old['samples_sha256'])
            self.assertEqual(worker.sha(run/'source.sha256.json'), worker.sha(snapshot/'source.sha256.json'))

    def test_rejects_assigned_chromosome_already_prepared(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, snapshot, run, config = self.fixture(Path(tmp)); (parent/'chr01').mkdir()
            with self.assertRaisesRegex(ValueError, 'already prepared'):
                worker.prepare(parent, snapshot, run, 'worker01', '1',
                               Path(config['destination'])/'parallel/worker01', Path(config['bulk'])/'parallel/worker01')
            self.assertFalse(run.exists())

    def test_rejects_wrong_destinations_and_nested_run(self):
        for case in ('destination', 'bulk', 'nested'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                parent, snapshot, run, config = self.fixture(Path(tmp))
                destination = Path(config['destination'])/'parallel/worker01'
                bulk = Path(config['bulk'])/'parallel/worker01'
                if case == 'destination': destination = Path(config['destination'])
                if case == 'bulk': bulk = Path(config['bulk'])
                if case == 'nested': run = parent/'nested'
                with self.assertRaises(ValueError):
                    worker.prepare(parent, snapshot, run, 'worker01', '1', destination, bulk)

    def test_rejects_changed_source_even_with_updated_manifest_unless_request_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, snapshot, run, config = self.fixture(Path(tmp))
            (snapshot/'source/modules/example.nf').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
                worker.prepare(parent, snapshot, run, 'worker01', '1',
                               Path(config['destination'])/'parallel/worker01', Path(config['bulk'])/'parallel/worker01')

    def test_execute_only_assigned_and_does_not_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, snapshot, run, config = self.prepare(Path(tmp))
            pipeline = worker.load_pipeline(run/'source')
            events = []

            class FakeRunner:
                def __init__(self, path):
                    self.current_stage = 'fixture'
                def execute(self):
                    raise AssertionError('Full Runner.execute must never be called')
                def aggregate(self):
                    raise AssertionError('Aggregate must never be called')
                def analyze_chromosome(self, c):
                    events.append(c)
                    folder = run/f'chr{c:02d}'; folder.mkdir()
                    item = folder/'evidence.npz'; item.write_bytes(b'synthetic')
                    pipeline.write_new(run/'checkpoints'/f'chr{c:02d}_complete.json',
                        dict(outputs=pipeline.files_manifest([item])))
                    relative = f'01_estructura/desarrollo/por_cromosoma/chr{c:02d}'
                    pipeline.write_new(run/'checkpoints'/('publish_'+relative.replace('/', '_')+'.json'),
                        dict(files=[dict(path=str(item), uri='gs://fixture/'+str(c), bytes=9,
                                         sha256=worker.sha(item), generation='1', md5_base64='mock')]))
                def publish(self, folder, relative):
                    events.append(relative)
            pipeline.Runner = FakeRunner
            with patch.object(worker, 'load_pipeline', return_value=pipeline):
                worker.execute(run)
            self.assertEqual([event for event in events if isinstance(event, int)], [1, 20])
            result = worker.read_json(run/'worker_provenance/worker_completion.json')
            self.assertFalse(result['aggregate_executed'])
            self.assertEqual(result['chromosomes'], [1, 20])
            self.assertEqual(worker.read_json(run/'status.json')['state'], 'COMPLETE')

    def test_changed_frozen_assignment_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, snapshot, run, config = self.prepare(Path(tmp))
            (run/'run.json').write_text('{}')
            with patch.object(worker, 'load_pipeline', side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
                    worker.execute(run)

    def test_manifest_rejects_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'file').write_text('data'); (root/'link').symlink_to(root/'file')
            for entry in ('../elsewhere', '/etc/passwd', 'link'):
                with self.assertRaises(ValueError):
                    worker.verify_manifest(root, {entry: worker.sha(root/'file')})


if __name__ == '__main__':
    unittest.main()
