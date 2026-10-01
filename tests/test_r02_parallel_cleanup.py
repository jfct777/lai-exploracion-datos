"""Synthetic safety tests: no real cloud objects, machines, or deletion calls."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r02_cleanup',ROOT/'bin/r02_parallel_cleanup.py')
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)
FIXTURE = importlib.util.spec_from_file_location('cleanup_coordinator_fixtures',ROOT/'tests/test_r02_parallel_coordinator.py')
fixtures = importlib.util.module_from_spec(FIXTURE)
FIXTURE.loader.exec_module(fixtures)


class FakeCompute:
    def __init__(self,vm):
        self.vm=copy.deepcopy(vm)
        self.disk={'id':'777'}
        self.deleted=[]
        self.retain_disk=False

    def describe(self,worker,resource='instances'):
        return copy.deepcopy(self.vm if resource=='instances' else self.disk)

    def delete(self,worker):
        self.deleted.append(worker['name'])
        self.vm=None
        if not self.retain_disk:
            self.disk=None


class CleanupSafetyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.CoordinatorTests(methodName='test_manifest_valid')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.import_one()
        self.run=self.fixture.run
        self.folder=self.fixture.directory/'cleanup'
        self.folder.mkdir()
        self.spec={'parent_run':str(self.run)}
        self.worker=dict(worker_id='worker01',name='dnabr-r02p-1001-w01',project='uspbr-242713',zone='us-central1-a',
            instance_id='1234',boot_disk_name='dnabr-r02p-1001-w01',chromosomes=[1],
            completion_uri=self.fixture.entry['completion_uri'],
            boot_disk_source='https://www.googleapis.com/compute/v1/projects/uspbr-242713/zones/us-central1-a/disks/dnabr-r02p-1001-w01')
        self.vm=dict(id='1234',name=self.worker['name'],status='TERMINATED',labels=dict(team='frank',round='r02',role='worker'),
            selfLink='https://www.googleapis.com/compute/v1/projects/uspbr-242713/zones/us-central1-a/instances/'+self.worker['name'],
            disks=[dict(boot=True,autoDelete=True,source=self.worker['boot_disk_source'])])
        self.compute=FakeCompute(self.vm)

    def step(self):
        return cleanup.cleanup_one(self.worker,self.spec,self.fixture.spec,fixtures.coordinator,self.folder,self.fixture.cloud,self.compute)

    def test_delete_only_verified_success_and_stopped_vm(self):
        self.assertEqual(self.step(),'DELETED_VERIFIED')
        self.assertEqual(self.compute.deleted,[self.worker['name']])
        self.assertTrue((self.folder/'worker01.delete_intent.json').exists())
        self.assertTrue((self.folder/'worker01.deleted.json').exists())

    def test_running_vm_never_deleted(self):
        self.compute.vm['status']='RUNNING'
        self.assertEqual(self.step(),'RETAINED_WAITING_VM_STOP')
        self.assertEqual(self.compute.deleted,[])

    def test_missing_import_never_deleted(self):
        (self.fixture.imports/'chr01_import.json').unlink()
        self.assertEqual(self.step(),'RETAINED_WAITING_VERIFIED_IMPORTS')
        self.assertEqual(self.compute.deleted,[])

    def test_all_assigned_imports_required(self):
        self.worker['chromosomes']=[1,2]
        self.assertEqual(self.step(),'RETAINED_WAITING_VERIFIED_IMPORTS')
        self.assertEqual(self.compute.deleted,[])

    def test_changed_native_id_rejected(self):
        self.compute.vm['id']='9999'
        with self.assertRaisesRegex(ValueError,'native identity'):
            self.step()
        self.assertEqual(self.compute.deleted,[])

    def test_changed_label_rejected(self):
        self.compute.vm['labels']['role']='development'
        with self.assertRaisesRegex(ValueError,'ownership'):
            self.step()
        self.assertEqual(self.compute.deleted,[])

    def test_extra_disk_rejected(self):
        self.compute.vm['disks'].append(dict(source='important-data',autoDelete=True,boot=False))
        with self.assertRaisesRegex(ValueError,'one auto-delete'):
            self.step()
        self.assertEqual(self.compute.deleted,[])

    def test_boot_not_auto_delete_rejected(self):
        self.compute.vm['disks'][0]['autoDelete']=False
        with self.assertRaisesRegex(ValueError,'auto-delete'):
            self.step()

    def test_changed_boot_disk_rejected(self):
        self.compute.vm['disks'][0]['source']='different'
        with self.assertRaisesRegex(ValueError,'Boot disk changed'):
            self.step()

    def test_changed_local_data_prevents_delete(self):
        (self.run/'chr01/rare_evidence.npz').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'imported result changed'):
            self.step()
        self.assertEqual(self.compute.deleted,[])

    def test_changed_remote_data_prevents_delete(self):
        uri=self.fixture.records[0]['uri']
        self.fixture.cloud.meta[uri]['generation']='55555'
        with self.assertRaisesRegex(ValueError,'generation'):
            self.step()
        self.assertEqual(self.compute.deleted,[])

    def test_remote_vcf_also_revalidated_before_delete(self):
        record=next(r for r in self.fixture.records if r['relative_path'].endswith('fixture.vcf.gz'))
        self.fixture.cloud.meta[record['uri']]['generation']='88888'
        with self.assertRaisesRegex(ValueError,'generation'):
            self.step()
        self.assertEqual(self.compute.deleted,[])

    def test_idempotent_deletion_receipt(self):
        self.step()
        self.assertEqual(self.step(),'DELETED_VERIFIED')
        self.assertEqual(len(self.compute.deleted),1)

    def test_absent_vm_without_intent_not_claimed_deleted(self):
        self.compute.vm=None
        self.assertEqual(self.step(),'ALREADY_ABSENT_NOT_DELETED_BY_CLEANUP')
        self.assertFalse((self.folder/'worker01.deleted.json').exists())

    def test_remaining_disk_never_deleted_separately(self):
        self.compute.retain_disk=True
        self.assertEqual(self.step(),'VM_ABSENT_BOOT_DISK_STILL_PRESENT')
        self.assertEqual(self.step(),'VM_ABSENT_BOOT_DISK_STILL_PRESENT')
        self.assertEqual(len(self.compute.deleted),1)
        self.assertFalse((self.folder/'worker01.deleted.json').exists())

    def test_creation_must_be_single_success(self):
        for value in ({'returncode':1,'response':[self.vm]}, {'returncode':0,'response':[]},
                      {'returncode':0,'response':[self.vm,self.vm]}, {'returncode':False,'response':[self.vm]}):
            with self.assertRaises(ValueError):
                cleanup.instance_from_creation(value)

    def test_boot_disk_postcheck_completes_after_delayed_auto_delete(self):
        self.compute.retain_disk=True
        self.step()
        self.compute.disk=None
        self.assertEqual(self.step(),'DELETED_VERIFIED')
        self.assertEqual(len(self.compute.deleted),1)

    def test_replace_vm_after_intent_is_rejected(self):
        old=self.compute.describe
        self.count=0
        def describe(worker,resource='instances'):
            self.count+=1
            value=old(worker,resource)
            if self.count>=2 and resource=='instances':
                value['id']='replacement'
            return value
        self.compute.describe=describe
        with self.assertRaisesRegex(ValueError,'native identity'):
            self.step()
        self.assertEqual(self.compute.deleted,[])

    def make_creation_fixtures(self):
        self.folder.rename(self.folder.with_name('cleanup-unit-test'))
        workers=[]
        for index in range(1,13):
            worker=f'worker{index:02d}'
            name=f'dnabr-r02p-1001-w{index:02d}'
            entries=[item for item in self.fixture.spec['remote_chromosomes'] if item['worker_id']==worker]
            entry=dict(worker=worker,name=name,chromosomes=[item['chromosome'] for item in entries],
                run_sha256=entries[0]['worker_run_sha256'],completion_uri=entries[0]['completion_uri'])
            workers.append(entry)
            vm=copy.deepcopy(self.vm)
            vm.update(id=str(1234+index),name=name)
            vm['selfLink']=vm['selfLink'].rsplit('/',1)[0]+'/'+name
            vm['disks'][0]['source']=vm['disks'][0]['source'].rsplit('/',1)[0]+'/'+name
            fixtures.put(self.fixture.directory/(worker+'.creation.json'),dict(worker=worker,returncode=0,response=[vm]))
        fixtures.put(self.fixture.directory/'fleet.json',dict(schema='r02_fleet_v1',project='uspbr-242713',
            zone='us-central1-a',workers=workers))

    def test_prepare_freezes_exact_twelve_identities_and_dependencies(self):
        self.make_creation_fixtures()
        path=cleanup.prepare(self.run,fixtures.coordinator.__file__)
        spec,coordinator,helper=cleanup.validate(path,cleanup.sha(path))
        self.assertEqual(len(spec['workers']),12)
        self.assertEqual(spec['coordinator_manifest_sha256'],cleanup.sha(self.fixture.manifest))
        self.assertEqual(helper.ESSENTIAL,fixtures.coordinator.ESSENTIAL)
        self.assertEqual(self.compute.deleted,[])

    def test_prepare_rejects_partial_fleet_creation(self):
        self.make_creation_fixtures()
        (self.fixture.directory/'worker12.creation.json').unlink()
        with self.assertRaisesRegex(ValueError,'twelve creation'):
            cleanup.prepare(self.run,fixtures.coordinator.__file__)
        self.assertFalse((self.fixture.directory/'cleanup').exists())

    def test_manifest_rejects_changed_creation_after_prepare(self):
        self.make_creation_fixtures()
        path=cleanup.prepare(self.run,fixtures.coordinator.__file__)
        creation=self.fixture.directory/'worker01.creation.json'
        value=cleanup.read(creation)
        value['response'][0]['id']='999999'
        fixtures.put(creation,value)
        with self.assertRaisesRegex(ValueError,'Creation receipt changed'):
            cleanup.validate(path,cleanup.sha(path))

    def test_operational_cleanup_prepared_separately_keeps_original_dependencies(self):
        self.make_creation_fixtures()
        original = cleanup.prepare(self.run, fixtures.coordinator.__file__)
        before = original.read_bytes()
        self.fixture.enable_operational()
        self.fixture.verify()
        folder = self.run/'repairs/preprocess-v1/cleanup'
        path = cleanup.prepare(self.run, fixtures.coordinator.__file__,
            coordinator_manifest=self.fixture.manifest, cleanup_directory=folder)
        spec, coordinator, helper = cleanup.validate(path, cleanup.sha(path))
        self.assertEqual(path.parent, folder)
        self.assertEqual(spec['coordinator_manifest'], str(self.fixture.manifest))
        self.assertEqual(coordinator['operational_amendments'], self.fixture.spec['operational_amendments'])
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(self.compute.deleted, [])

    def test_operational_cleanup_rechecks_provenance_and_source_per_chromosome(self):
        self.fixture.enable_operational()
        (self.fixture.imports/'chr01_import.json').unlink()
        self.fixture.import_one()
        path = self.fixture.payload['operational_provenance'][0]['uri']
        self.fixture.cloud.meta[path]['generation'] = '888'
        with self.assertRaisesRegex(ValueError, 'generation'):
            self.step()
        self.assertEqual(self.compute.deleted, [])
        self.fixture.cloud.meta[path]['generation'] = '12345'
        proof_path = self.fixture.imports/'chr01_import.json'
        proof = cleanup.read(proof_path)
        proof['preprocess_source_manifest_sha256'] = self.fixture.spec['source_manifest_sha256']
        fixtures.put(proof_path, proof)
        with self.assertRaisesRegex(ValueError, 'per-chromosome'):
            self.step()
        self.assertEqual(self.compute.deleted, [])

    def test_operational_cleanup_accepts_only_verified_complete_worker(self):
        self.fixture.enable_operational()
        (self.fixture.imports/'chr01_import.json').unlink()
        self.fixture.import_one()
        self.assertEqual(self.step(), 'DELETED_VERIFIED')
        self.assertEqual(self.compute.deleted, [self.worker['name']])

    def test_cleanup_rejects_mixing_manifest_and_directory_versions(self):
        self.make_creation_fixtures()
        self.fixture.enable_operational()
        self.fixture.verify()
        with self.assertRaisesRegex(ValueError, 'matching approved directories'):
            cleanup.prepare(self.run, fixtures.coordinator.__file__,
                            coordinator_manifest=self.fixture.manifest)


if __name__=='__main__':
    unittest.main()
