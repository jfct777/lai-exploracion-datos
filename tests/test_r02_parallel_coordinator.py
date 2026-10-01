"""Synthetic local tests; no GCP calls, human data reads, or process signals."""
import base64
import copy
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('parallel_coordinator', ROOT/'bin/r02_parallel_coordinator.py')
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeCloud:
    def __init__(self):
        self.data = {}
        self.meta = {}
        self.downloads = []

    def add(self, uri, payload):
        self.data[uri] = payload
        self.meta[uri] = dict(generation='12345', size=len(payload),
            md5_hash=base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode())

    def metadata(self, uri, optional=False):
        if optional and uri not in self.data:
            return None
        return self.meta[uri]

    def read(self, uri, generation):
        assert str(generation) == str(self.meta[uri]['generation'])
        return self.data[uri]

    def download(self, uri, generation, path):
        self.downloads.append(uri)
        Path(path).write_bytes(self.read(uri, generation))


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)/'r02-test'
        self.directory = self.run/'repairs/parallel-v1'
        self.directory.mkdir(parents=True)
        (self.run/'checkpoints').mkdir()
        (self.run/'samples.txt').write_text('sample-A\nsample-B\n')
        self.sample_hash = coordinator.sha(self.run/'samples.txt')
        destination = '/home/jose.tantalean/gcs-dnabr/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-test'
        put(self.run/'run.json', dict(destination=destination, samples_sha256=self.sample_hash))
        self.request = self.run/'repairs/boundary-v2/request.json'
        put(self.request, {'fixture': True})
        self.source = self.run/'repairs/boundary-v2/source.sha256.json'
        put(self.source, {'bin/r02_apply_amendment.py': 'a'*64})
        prefix = 'gs://projects-usp/dnaBr-lai/datalake/' + destination.split('/gcs-dnabr/')[1]
        remote = []
        for chrom in range(1, 21):
            worker = f'worker{min(chrom, 12):02d}'
            target = prefix + '/parallel/' + worker
            remote.append(dict(chromosome=chrom, worker_id=worker, publication_prefix=target,
                completion_uri=target+'/00_worker_provenance/worker_completion.json', worker_run_sha256='b'*64))
        self.spec = dict(schema=coordinator.SCHEMA, run_dir=str(self.run), request=str(self.request),
            request_sha256=coordinator.sha(self.request), coordinator_sha256=coordinator.sha(coordinator.__file__),
            source_manifest_sha256=coordinator.sha(self.source), samples_sha256=self.sample_hash,
            deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(), poll_seconds=1,
            max_import_bytes=1024**2, old_watcher=dict(pid=99999, start_ticks=10, cmdline_sha256='c'*64),
            remote_chromosomes=remote)
        self.manifest = self.directory/'manifest.json'
        put(self.manifest, self.spec)
        self.imports = self.directory/'imports'
        self.imports.mkdir()
        self.entry = remote[0]
        self.cloud = FakeCloud()
        self.records = []
        base = self.entry['publication_prefix']+'/01_estructura/desarrollo/por_cromosoma/chr01/'
        for name in sorted(coordinator.ESSENTIAL | {'preprocess/lai_rare/fixture.vcf.gz'}):
            data = ('synthetic-'+name).encode()
            uri = base + name
            self.cloud.add(uri, data)
            self.records.append(dict(relative_path=name, path='/worker/chr01/'+name, uri=uri,
                bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                md5_base64=self.cloud.meta[uri]['md5_hash'], generation='12345'))
        self.payload = dict(schema_version=1, worker_id='worker01', parent_run_id=self.run.name,
            chromosomes=[1], analysis_protocol_version=2, source_manifest_sha256=self.spec['source_manifest_sha256'],
            samples_sha256=self.sample_hash, worker_run_sha256='b'*64, worker_frozen_sha256='d'*64,
            aggregate_executed=False, biological_validation_complete=False,
            artifacts=[dict(chromosome=1, outputs=self.records, completion_sha256='e'*64, publication_sha256='f'*64)])
        self.publish_receipt()

    def publish_receipt(self):
        self.cloud.add(self.entry['completion_uri'], json.dumps(self.payload).encode())

    def enable_operational(self):
        source_code = b'operational annotation fixture\n'
        source_manifest = json.dumps({'bin/mark_original_alleles.py': hashlib.sha256(source_code).hexdigest()}).encode()
        wrapper = b'operational wrapper fixture\n'
        manifest = b'{"schema":"synthetic-approved-operational-manifest"}\n'
        amendment = dict(schema=coordinator.OPERATIONAL_SCHEMA,
            manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            source_manifest_sha256=hashlib.sha256(source_manifest).hexdigest(),
            new_preprocess_chromosomes=[1], legacy_chromosomes=[])
        self.spec['operational_amendments'] = {'worker01': copy.deepcopy(amendment)}
        self.spec['operational_wrapper_sha256'] = {'worker01': hashlib.sha256(wrapper).hexdigest()}
        self.payload['operational_amendment'] = copy.deepcopy(amendment)
        activation = json.dumps(dict(schema=coordinator.OPERATIONAL_SCHEMA,
            manifest_sha256=amendment['manifest_sha256'],
            original_source_manifest_sha256=self.spec['source_manifest_sha256'],
            new_source_manifest_sha256=amendment['source_manifest_sha256'],
            changed_source_files=['bin/mark_original_alleles.py'], operational_only=True,
            biological_validation_complete=False)).encode()
        prefix = (self.entry['publication_prefix'] + '/00_worker_provenance/operational_amendments/'
                  + amendment['manifest_sha256'] + '/')
        records = []
        for name, data in [('manifest.json', manifest), ('source.sha256.json', source_manifest),
                           ('optimized_worker.py', wrapper), ('activation.json', activation),
                           ('source/bin/mark_original_alleles.py', source_code)]:
            uri = prefix + name
            self.cloud.add(uri, data)
            records.append(dict(uri=uri, generation='12345', bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), md5_base64=self.cloud.meta[uri]['md5_hash']))
        self.payload['operational_provenance'] = records
        self.manifest = self.run/'repairs/preprocess-v1/coordinator/manifest.json'
        self.publish_receipt()
        return amendment

    def verify(self):
        put(self.manifest, self.spec)
        return coordinator.validate_manifest(self.manifest, coordinator.sha(self.manifest))

    def import_one(self):
        return coordinator.import_chromosome(self.spec, self.entry, self.imports, self.cloud)

    def test_manifest_valid(self):
        self.assertEqual(self.verify()['remote_chromosomes'], self.spec['remote_chromosomes'])

    def test_operational_manifest_and_import_preserve_base_and_explicit_new_source(self):
        amendment = self.enable_operational()
        self.verify()
        self.import_one()
        proof = coordinator.load(self.imports/'chr01_import.json')
        self.assertEqual(proof['operational_amendment'], amendment)
        self.assertEqual(proof['preprocess_source_manifest_sha256'], amendment['source_manifest_sha256'])
        self.assertEqual(self.payload['source_manifest_sha256'], self.spec['source_manifest_sha256'])
        self.assertEqual(proof['operational_provenance'], self.payload['operational_provenance'])

    def test_operational_amendment_requires_versioned_directory_and_exact_assignment(self):
        self.enable_operational()
        self.manifest = self.directory/'manifest.json'
        with self.assertRaisesRegex(ValueError, 'own coordinator directory'):
            self.verify()
        self.manifest = self.run/'repairs/preprocess-v1/coordinator/manifest.json'
        for key, value in [('new_preprocess_chromosomes', [1, 1]), ('legacy_chromosomes', [1]),
                           ('new_preprocess_chromosomes', []), ('new_preprocess_chromosomes', [True])]:
            old = self.spec['operational_amendments']['worker01'][key]
            self.spec['operational_amendments']['worker01'][key] = value
            with self.assertRaisesRegex(ValueError, 'partition'):
                self.verify()
            self.spec['operational_amendments']['worker01'][key] = old

    def test_missing_unapproved_or_changed_operational_receipt_rejected(self):
        self.enable_operational()
        approved = copy.deepcopy(self.payload)
        del self.payload['operational_amendment']
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'approved per-chromosome'):
            self.import_one()
        self.payload = approved
        self.spec.pop('operational_amendments')
        self.spec.pop('operational_wrapper_sha256')
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            self.import_one()

    def test_operational_object_generation_and_content_are_checked(self):
        self.enable_operational()
        record = self.payload['operational_provenance'][-1]
        self.cloud.meta[record['uri']]['generation'] = '67890'
        with self.assertRaisesRegex(ValueError, 'generation'):
            self.import_one()
        self.cloud.meta[record['uri']]['generation'] = '12345'
        self.cloud.data[record['uri']] = b'changed'
        with self.assertRaisesRegex(ValueError, 'content differs'):
            self.import_one()
        self.assertFalse((self.imports/'chr01_import.json').exists())

    def test_operational_missing_wrapper_and_extra_paths_are_rejected(self):
        self.enable_operational()
        self.payload['operational_provenance'] = [r for r in self.payload['operational_provenance']
                                                 if not r['uri'].endswith('/optimized_worker.py')]
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'source identities'):
            self.import_one()
        self.enable_operational()
        self.payload['operational_provenance'][-1]['uri'] = 'gs://outside/source.py'
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'approved prefix'):
            self.import_one()

    def test_changed_base_source_is_not_accepted_by_operational_amendment(self):
        self.enable_operational()
        self.payload['source_manifest_sha256'] = self.payload['operational_amendment']['source_manifest_sha256']
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'scientific source'):
            self.import_one()

    def test_operational_wrapper_can_preserve_only_legacy_chromosomes(self):
        self.enable_operational()
        for owner in (self.spec['operational_amendments']['worker01'], self.payload['operational_amendment']):
            owner['new_preprocess_chromosomes'] = []
            owner['legacy_chromosomes'] = [1]
        self.publish_receipt()
        self.verify()
        self.import_one()
        proof = coordinator.load(self.imports/'chr01_import.json')
        self.assertEqual(proof['preprocess_source_manifest_sha256'], self.spec['source_manifest_sha256'])

    def test_operational_activation_must_describe_exact_published_delta(self):
        self.enable_operational()
        record = next(r for r in self.payload['operational_provenance'] if r['uri'].endswith('/activation.json'))
        activation = json.loads(self.cloud.data[record['uri']])
        activation['changed_source_files'] = ['bin/another_script.py']
        data = json.dumps(activation).encode()
        self.cloud.add(record['uri'], data)
        record.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                      md5_base64=self.cloud.meta[record['uri']]['md5_hash'])
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'activation differs'):
            self.import_one()

    def test_manifest_source_self_hash_required(self):
        self.spec['coordinator_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.verify()

    def test_manifest_request_hash_required(self):
        self.request.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'request changed'):
            self.verify()

    def test_manifest_samples_match(self):
        self.spec['samples_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Samples changed'):
            self.verify()

    def test_exact_chromosomes_and_workers_required(self):
        self.spec['remote_chromosomes'][0]['chromosome'] = 21
        with self.assertRaisesRegex(ValueError, 'exactly1'):
            self.verify()

    def test_publication_prefix_cannot_escape(self):
        self.entry['publication_prefix'] = 'gs://public/example'
        with self.assertRaisesRegex(ValueError, 'escaped'):
            self.verify()

    def test_expired_deadline_rejected(self):
        self.spec['deadline_utc'] = '2000-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError, 'deadline'):
            self.verify()

    def test_unbounded_deadline_rejected(self):
        self.spec['deadline_utc'] = (datetime.now(timezone.utc)+timedelta(hours=81)).isoformat()
        with self.assertRaisesRegex(ValueError, '80 hours'):
            self.verify()

    def test_import_essential_only_and_no_fake_compute_checkpoint(self):
        self.assertTrue(self.import_one())
        self.assertEqual(len(self.cloud.downloads), len(coordinator.ESSENTIAL))
        self.assertFalse((self.run/'chr01/preprocess').exists())
        self.assertEqual(list((self.run/'checkpoints').iterdir()), [])
        proof = coordinator.load(self.imports/'chr01_import.json')
        self.assertEqual(proof['all_published_objects_verified'], 10)
        self.assertFalse(proof['local_compute_checkpoint_created'])
        self.assertEqual(proof['outputs'][0]['source_path'], '/worker/chr01/'+proof['outputs'][0]['relative_path'])

    def test_import_is_idempotent(self):
        self.import_one()
        original = (self.imports/'chr01_import.json').read_bytes()
        self.import_one()
        self.assertEqual(len(self.cloud.downloads), len(coordinator.ESSENTIAL))
        self.assertEqual(original, (self.imports/'chr01_import.json').read_bytes())

    def test_absent_completion_is_pending(self):
        self.cloud.data.pop(self.entry['completion_uri'])
        self.assertFalse(self.import_one())
        self.assertFalse((self.imports/'chr01_import.json').exists())

    def test_changed_generation_rejected(self):
        self.cloud.meta[self.records[0]['uri']]['generation'] = '987'
        with self.assertRaisesRegex(ValueError, 'generation'):
            self.import_one()

    def test_download_sha256_checked(self):
        self.records[0]['sha256'] = '0'*64
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.import_one()

    def test_wrong_worker_source_rejected(self):
        self.payload['source_manifest_sha256'] = '0'*64
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'scientific source'):
            self.import_one()

    def test_wrong_worker_assignment_rejected(self):
        self.payload['chromosomes'] = [1, 2]
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'assignment'):
            self.import_one()

    def test_missing_required_file_rejected(self):
        self.records.pop(0)
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'required aggregation'):
            self.import_one()

    def test_path_traversal_rejected(self):
        self.records[0]['relative_path'] = '../escaped'
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'relative path'):
            self.import_one()

    def test_output_uri_escape_rejected(self):
        self.records[0]['uri'] = 'gs://wrong/data'
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'escaped chromosome'):
            self.import_one()

    def test_local_compute_collision_rejected(self):
        put(self.run/'checkpoints/chr01_M01_M02_M021.json', {'completed': True})
        with self.assertRaisesRegex(ValueError, 'execution already'):
            self.import_one()

    def test_local_changed_file_not_overwritten(self):
        self.import_one()
        path = self.run/'chr01/rare_evidence.npz'
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Existing import differs'):
            self.import_one()
        self.assertEqual(path.read_bytes(), b'changed')

    def test_import_budget_and_disk_guards(self):
        self.spec['max_import_bytes'] = 1
        with self.assertRaisesRegex(ValueError, 'byte limit'):
            self.import_one()
        self.spec['max_import_bytes'] = 1024**2
        with patch.object(coordinator.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=100)):
            with self.assertRaisesRegex(ValueError, 'local disk'):
                self.import_one()

    def test_duplicate_output_rejected(self):
        self.records.append(dict(self.records[0]))
        self.publish_receipt()
        with self.assertRaisesRegex(ValueError, 'Duplicate output'):
            self.import_one()

    def test_total_deadline_controls_import(self):
        with self.assertRaisesRegex(ValueError, 'deadline expired'):
            coordinator.import_chromosome(self.spec, self.entry, self.imports, self.cloud, deadline=1)

    def test_local_only22_then21_before_import_and_aggregation(self):
        calls = []
        runner = types.SimpleNamespace(run=self.run, dest=Path('/fake/destination'), current_stage='test',
            analyze_chromosome=lambda c: calls.append(('local', c)),
            aggregate=lambda: calls.append(('aggregate',)),
            publish=lambda *args: calls.append(('publish',)))
        pipeline = types.SimpleNamespace(execution_lock=lambda _: nullcontext(), status=Mock(), uri=str)
        adapter = types.SimpleNamespace(validate_snapshot=lambda *args: ({}, pipeline, {}),
            build_runner=lambda *args, **kwargs: (runner, pipeline, {'amendment_sha256': 'a'*64}))
        with patch.object(coordinator, 'load_module', return_value=adapter), \
             patch.object(coordinator, 'wait_and_import', side_effect=lambda *args: calls.append(('import',))):
            coordinator.run_after_boundary(self.spec, self.directory, types.SimpleNamespace(state=Mock()))
        self.assertEqual(calls[:4], [('local',22), ('local',21), ('import',), ('aggregate',)])
        self.assertFalse(any(c[0] == 'execute' for c in calls))


if __name__ == '__main__':
    unittest.main()
