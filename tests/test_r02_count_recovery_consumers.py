"""Synthetic v2 recovery contracts; no cloud, VM, genotype or service operations."""
import copy
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


worker_fixtures = module('count_worker_fixtures', ROOT/'tests/test_r02_optimized_worker.py')
coordinator_fixtures = module('count_coordinator_fixtures', ROOT/'tests/test_r02_parallel_coordinator.py')
cleanup_fixtures = module('count_cleanup_fixtures', ROOT/'tests/test_r02_parallel_cleanup.py')
worker = worker_fixtures.OPT
coordinator = coordinator_fixtures.coordinator
cleanup = cleanup_fixtures.cleanup


def json_put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def worker_v2(root):
    fixture = worker_fixtures.OptimizedWorkerTests()
    run, previous, original, _ = fixture.fixture(root)
    directory = run/'repairs/preprocess-v2'
    shutil.copytree(previous, directory)
    helper = directory/'preprocess_count_validation.py'
    helper.write_text('def validate_raw_record_count(folder, chrom, *, source_index=None):\n'
                      '    return dict(raw_records=9, count_source="synthetic", checked_chromosome=chrom)\n')
    spec = {**original, 'schema':worker.COUNT_SCHEMA,
            'count_validator_sha256':worker.sha(helper),
            'previous_manifest_sha256':worker.sha(previous/'manifest.json')}
    json_put(directory/'manifest.json', spec)
    return run, directory, previous, spec


def coordinator_v2(fixture):
    fixture.enable_operational()
    helper = b'# authenticated synthetic record-count validator\n'
    previous = b'{"schema":"synthetic-predecessor"}\n'
    extras = dict(count_validator_sha256=hashlib.sha256(helper).hexdigest(),
                  previous_manifest_sha256=hashlib.sha256(previous).hexdigest())
    for owner in (fixture.spec['operational_amendments']['worker01'],
                  fixture.payload['operational_amendment']):
        owner.update(schema=coordinator.COUNT_OPERATIONAL_SCHEMA, **extras)
    amendment = fixture.payload['operational_amendment']
    prefix = (fixture.entry['publication_prefix']+'/00_worker_provenance/operational_amendments/'
              + amendment['manifest_sha256']+'/')
    records = fixture.payload['operational_provenance']
    activation_record = next(r for r in records if r['uri'].endswith('/activation.json'))
    activation = json.loads(fixture.cloud.data[activation_record['uri']])
    activation.update(schema=coordinator.COUNT_OPERATIONAL_SCHEMA, **extras)
    for name, data in [('activation.json',json.dumps(activation).encode()),
                       ('preprocess_count_validation.py',helper),('previous_manifest.json',previous)]:
        uri = prefix+name
        fixture.cloud.add(uri,data)
        record = dict(uri=uri, generation='12345', bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(),md5_base64=fixture.cloud.meta[uri]['md5_hash'])
        if name == 'activation.json':activation_record.update(record)
        else:records.append(record)
    fixture.manifest = fixture.run/'repairs/preprocess-v2/coordinator/manifest.json'
    directory = fixture.manifest.parent
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'preprocess_count_validation.py').write_bytes(helper)
    json_put(directory/'adoption.json', {'synthetic':True})
    fixture.spec['record_count_repair'] = dict(validator_sha256=hashlib.sha256(helper).hexdigest(),
                                              adoption_sha256=coordinator.sha(directory/'adoption.json'))
    fixture.publish_receipt()
    return amendment


class WorkerCountRecoveryTests(unittest.TestCase):
    def check(self,directory):
        path=directory/'manifest.json'
        return worker.validate_spec(path,worker.sha(path))

    def test_v2_accepts_preserved_settings_and_frozen_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,directory,previous,spec=worker_v2(Path(tmp))
            before={str(p):worker.sha(p) for root in (run/'source',previous) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(self.check(directory)[0],spec)
            old=types.SimpleNamespace(validate_raw_record_count='old')
            new=types.SimpleNamespace(validate_raw_record_count='new')
            helper=worker.bind_count_validator(directory,spec,old,new)
            self.assertIs(old.validate_raw_record_count,helper.validate_raw_record_count)
            self.assertIs(new.validate_raw_record_count,helper.validate_raw_record_count)
            self.assertEqual(old.validate_raw_record_count(run,5)['raw_records'],9)
            self.assertEqual(before,{str(p):worker.sha(p) for root in (run/'source',previous) for p in root.rglob('*') if p.is_file()})
            self.assertEqual(worker.validation_delta(spec),{key:spec[key] for key in
                ('count_validator_sha256','previous_manifest_sha256')})

    def test_legacy_schema_does_not_bind_or_claim_count_repair(self):
        old=types.SimpleNamespace(validate_raw_record_count='unchanged')
        self.assertIsNone(worker.bind_count_validator('/not/read',{'schema':worker.SCHEMA},old))
        self.assertEqual(old.validate_raw_record_count,'unchanged')
        self.assertEqual(worker.validation_delta({'schema':worker.SCHEMA}),{})

    def test_changed_count_helper_and_predecessor_are_rejected(self):
        for name in ('helper','previous','source'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                _,directory,previous,_=worker_v2(Path(tmp))
                path={'helper':directory/'preprocess_count_validation.py',
                      'previous':previous/'manifest.json',
                      'source':directory/'source/bin/mark_original_alleles.py'}[name]
                path.write_text(path.read_text()+'\n# modified\n')
                with self.assertRaises(ValueError):self.check(directory)

    def test_v2_cannot_change_existing_settings_even_with_new_manifest_hash(self):
        for key,value in [('legacy_chromosomes',[6]),('new_preprocess_chromosomes',[19]),
                          ('overrides',{'min_mac':1}),('source_manifest_sha256','0'*64)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                _,directory,_,spec=worker_v2(Path(tmp))
                spec[key]=value;json_put(directory/'manifest.json',spec)
                with self.assertRaisesRegex(ValueError,'changed previous settings'):self.check(directory)

    def test_v2_must_be_in_its_versioned_directory_and_have_both_digests(self):
        for field in ('count_validator_sha256','previous_manifest_sha256','directory'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                _,directory,_,spec=worker_v2(Path(tmp))
                if field=='directory':directory=directory.rename(directory.with_name('unexpected'))
                else:spec.pop(field);json_put(directory/'manifest.json',spec)
                with self.assertRaises(ValueError):self.check(directory)

    def test_runtime_binding_rechecks_hash_after_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,directory,_,spec=worker_v2(Path(tmp));self.check(directory)
            (directory/'preprocess_count_validation.py').write_text('# replaced\n')
            old=types.SimpleNamespace(validate_raw_record_count='old')
            new=types.SimpleNamespace(validate_raw_record_count='new')
            with self.assertRaises(ValueError):worker.bind_count_validator(directory,spec,old,new)
            self.assertEqual((old.validate_raw_record_count,new.validate_raw_record_count),('old','new'))

    def test_read_only_cli_accepts_only_legacy_chromosome_and_never_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,directory,_,_=worker_v2(Path(tmp));manifest=directory/'manifest.json'
            command=[sys.executable,str(ROOT/'bin/r02_optimized_worker.py'),'--manifest',str(manifest),
                     '--manifest-sha256',worker.sha(manifest),'--validate-legacy-count']
            result=subprocess.run(command+['5'],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(json.loads(result.stdout)['legacy_count_validated'])
            for args in (['20'],['21'],['5','--run']):
                result=subprocess.run(command+args,capture_output=True,text=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
            self.assertFalse((directory/'activation.json').exists())
            self.assertFalse((run/'chr05').exists())

    def test_v2_run_publishes_helper_predecessor_and_both_runtime_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,directory,_,spec=worker_v2(Path(tmp))
            frozen_worker=worker.load_module(run/'worker.py',worker.sha(run/'worker.py'),'_fixture_worker')
            old=frozen_worker.load_pipeline(run/'source')
            new=types.SimpleNamespace()
            calls=[]
            class FakeRunner:
                def __init__(self,path):
                    self.run=path;self.source=path/'source';self.bin=self.source/'bin'
                    self.c=worker.read(path/'run.json');self.current_stage='fixture'
                def preprocess(self,chrom):
                    calls.append(('old',chrom,old.validate_raw_record_count(self.run,chrom)))
                def analyze_chromosome(self,chrom):
                    checkpoint=self.run/'checkpoints'/f'chr{chrom:02d}_complete.json'
                    if checkpoint.exists():return
                    self.preprocess(chrom)
                    folder=self.run/f'chr{chrom:02d}';folder.mkdir()
                    result=folder/'fixture.tsv';result.write_text('synthetic\n')
                    self.publish(folder,f'01_estructura/desarrollo/por_cromosoma/chr{chrom:02d}')
                    worker.fixed(checkpoint,{'outputs':old.files_manifest([result])})
                def publish(self,source,relative):
                    checkpoint=self.run/'checkpoints'/('publish_'+relative.replace('/','_')+'.json')
                    if checkpoint.exists():return
                    records=[]
                    for path in sorted(source.rglob('*')):
                        if path.is_file():records.append(dict(path=str(path),bytes=path.stat().st_size,
                            uri='gs://fixture/'+relative+'/'+str(path.relative_to(source)),generation='1',
                            sha256=worker.sha(path),md5_base64=base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()))
                    worker.fixed(checkpoint,{'files':records})
            class FakeNew:
                def preprocess(self,chrom):
                    calls.append(('new',chrom,new.validate_raw_record_count(self.run,chrom)))
                    assert self.c['preprocess_checkpointed_m01'] is True
            old.Runner=FakeRunner;new.Runner=FakeNew
            original_loader=worker.load_module
            modules={'_original_worker':frozen_worker,'_original_pipeline':old,'_optimized_preprocess_pipeline':new}
            def loader(path,checksum,name):
                return modules[name] if name in modules else original_loader(path,checksum,name)
            with patch.object(worker,'load_module',side_effect=loader),patch.object(worker,'validate_idle'):
                receipt=worker.run_worker(directory/'manifest.json',worker.sha(directory/'manifest.json'))
                again=worker.run_worker(directory/'manifest.json',worker.sha(directory/'manifest.json'))
            self.assertEqual(receipt,again)
            self.assertEqual([(kind,chrom) for kind,chrom,_ in calls],[('old',5),('new',20)])
            self.assertTrue(all(audit['raw_records']==9 for _,_,audit in calls))
            for evidence in (receipt['operational_amendment'],worker.read(directory/'activation.json')):
                self.assertEqual(evidence['schema'],worker.COUNT_AMENDMENT_SCHEMA)
                self.assertEqual(evidence['count_validator_sha256'],spec['count_validator_sha256'])
                self.assertEqual(evidence['previous_manifest_sha256'],spec['previous_manifest_sha256'])
            names={record['uri'].rsplit('/',1)[-1]:record for record in receipt['operational_provenance']}
            self.assertEqual(names['preprocess_count_validation.py']['sha256'],spec['count_validator_sha256'])
            self.assertEqual(names['previous_manifest.json']['sha256'],spec['previous_manifest_sha256'])
            self.assertTrue(worker.read(directory/'completed.json')['published'])


class CoordinatorCountRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=coordinator_fixtures.CoordinatorTests(methodName='test_manifest_valid')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        coordinator_v2(self.fixture)

    def test_v2_manifest_provenance_import_preserves_approved_count_repair(self):
        f=self.fixture;f.verify();f.import_one()
        proof=coordinator.load(f.imports/'chr01_import.json')
        self.assertEqual(proof['operational_amendment'],f.payload['operational_amendment'])
        records=coordinator.verify_operational_provenance(f.payload,f.entry,f.spec,f.cloud)
        self.assertIn('preprocess_count_validation.py',records)
        self.assertIn('previous_manifest.json',records)

    def test_missing_or_changed_extra_provenance_is_rejected(self):
        f=self.fixture;original=copy.deepcopy(f.payload['operational_provenance'])
        for name in ('preprocess_count_validation.py','previous_manifest.json'):
            with self.subTest(name=name):
                f.payload['operational_provenance']=[r for r in original if not r['uri'].endswith('/'+name)]
                with self.assertRaisesRegex(ValueError,'source identities'):
                    coordinator.operational_records(f.payload,f.entry,f.spec)
                f.payload['operational_provenance']=copy.deepcopy(original)
                record=next(r for r in f.payload['operational_provenance'] if r['uri'].endswith('/'+name))
                record['sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'source identities'):
                    coordinator.operational_records(f.payload,f.entry,f.spec)

    def test_remote_helper_content_and_generation_rechecked(self):
        f=self.fixture
        record=next(r for r in f.payload['operational_provenance'] if r['uri'].endswith('/preprocess_count_validation.py'))
        f.cloud.meta[record['uri']]['generation']='777'
        with self.assertRaisesRegex(ValueError,'generation'):
            coordinator.verify_operational_provenance(f.payload,f.entry,f.spec,f.cloud)
        f.cloud.meta[record['uri']]['generation']='12345';f.cloud.data[record['uri']]=b'# replaced\n'
        with self.assertRaisesRegex(ValueError,'content differs'):
            coordinator.verify_operational_provenance(f.payload,f.entry,f.spec,f.cloud)

    def test_activation_cannot_omit_count_identity_even_with_valid_object_hash(self):
        f=self.fixture
        record=next(r for r in f.payload['operational_provenance'] if r['uri'].endswith('/activation.json'))
        value=json.loads(f.cloud.data[record['uri']]);value.pop('count_validator_sha256')
        data=json.dumps(value).encode();f.cloud.add(record['uri'],data)
        record.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),md5_base64=f.cloud.meta[record['uri']]['md5_hash'])
        with self.assertRaisesRegex(ValueError,'activation differs'):
            coordinator.verify_operational_provenance(f.payload,f.entry,f.spec,f.cloud)

    def test_local_repair_identity_and_directory_are_required(self):
        f=self.fixture;original=f.manifest
        f.manifest=f.run/'repairs/preprocess-v1/coordinator/manifest.json'
        with self.assertRaisesRegex(ValueError,'versioned coordinator directory'):f.verify()
        f.manifest=original
        for name in ('adoption.json','preprocess_count_validation.py'):
            path=f.manifest.parent/name;before=path.read_bytes();path.write_bytes(before+b'changed')
            with self.assertRaisesRegex(ValueError,'count-repair evidence'):f.verify()
            path.write_bytes(before)

    def test_count_schema_requires_both_valid_digests(self):
        f=self.fixture;original=copy.deepcopy(f.spec['operational_amendments']['worker01'])
        for key in ('count_validator_sha256','previous_manifest_sha256'):
            for mutation in ('missing','invalid'):
                with self.subTest(key=key,mutation=mutation):
                    value=copy.deepcopy(original)
                    if mutation=='missing':value.pop(key)
                    else:value[key]='not-a-digest'
                    f.spec['operational_amendments']['worker01']=value
                    with self.assertRaises(ValueError):coordinator.validate_operational_amendments(f.spec)


class CleanupCountRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=cleanup_fixtures.CleanupSafetyTests(methodName='test_prepare_freezes_exact_twelve_identities_and_dependencies')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.make_creation_fixtures()
        coordinator_v2(self.fixture.fixture)
        self.fixture.fixture.verify()

    def test_v2_cleanup_freezes_and_revalidates_matching_consumer(self):
        f=self.fixture
        directory=f.run/'repairs/preprocess-v2/cleanup'
        path=cleanup.prepare(f.run,coordinator.__file__,coordinator_manifest=f.fixture.manifest,cleanup_directory=directory)
        spec,observed,_=cleanup.validate(path,cleanup.sha(path))
        self.assertEqual(spec['coordinator_manifest'],str(f.fixture.manifest))
        self.assertEqual(observed['record_count_repair'],f.fixture.spec['record_count_repair'])
        self.assertEqual(len(spec['workers']),12)
        self.assertEqual(f.compute.deleted,[])

    def test_cross_version_cleanup_is_rejected_before_creation(self):
        f=self.fixture
        for version in ('parallel-v1','preprocess-v1'):
            directory=f.run/'repairs'/version/'cleanup'
            with self.assertRaisesRegex(ValueError,'matching approved directories'):
                cleanup.prepare(f.run,coordinator.__file__,coordinator_manifest=f.fixture.manifest,cleanup_directory=directory)
            self.assertFalse(directory.exists())

    def test_resealed_cleanup_manifest_cannot_point_to_previous_coordinator(self):
        f=self.fixture
        path=cleanup.prepare(f.run,coordinator.__file__,coordinator_manifest=f.fixture.manifest,
                             cleanup_directory=f.run/'repairs/preprocess-v2/cleanup')
        spec=cleanup.read(path);spec['coordinator_manifest']=str(f.run/'repairs/preprocess-v1/coordinator/manifest.json')
        json_put(path,spec)
        with self.assertRaisesRegex(ValueError,'Wrong version'):cleanup.validate(path,cleanup.sha(path))


if __name__=='__main__':unittest.main()
