"""Synthetic publication tests: no GCP calls, real genomes, boots or cleanup."""
import base64
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).parents[1]/'bin/r02_publish_retained_m02.py'
MODULE = importlib.util.spec_from_file_location('publication', SOURCE)
p = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(p)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True)+'\n')
    return p.file_sha(path)


class Cloud:
    def __init__(self):
        self.objects = {}
        self.uploads = []
        self.lose_ack = False
        self.after_upload = None

    def metadata(self, uri, optional=False):
        if uri not in self.objects:
            if optional:
                return None
            raise RuntimeError('404')
        return dict(self.objects[uri])

    def upload(self, source, uri):
        if uri in self.objects:
            raise RuntimeError('412 precondition')
        data = Path(source).read_bytes()
        self.objects[uri] = dict(size=len(data), generation=str(100+len(self.objects)),
                                 md5_hash=base64.b64encode(hashlib.md5(data).digest()).decode())
        self.uploads.append(uri)
        if self.after_upload:
            self.after_upload(source, uri)
        if self.lose_ack:
            raise RuntimeError('Connection lost after successful create')


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/p.RUN_ID/'parallel/worker05'
        self.old_folder = self.run/'repairs/m02-preservation-20261002'
        self.folder = self.run/'repairs'/p.FOLDER
        self.held_folder = self.old_folder/'held'
        self.held_folder.mkdir(parents=True)
        self.folder.mkdir(parents=True)
        self.bulk = self.root/'original_bulk'
        self.cfg = dict(parallel_worker=dict(worker_id='worker05'), processing_order=[5, 20], bulk=str(self.bulk))
        run_sha = write(self.run/'run.json', self.cfg)
        self.instance = dict(id='4726203795318754785', name='dnabr-r02p-1001-w05',
                             project='uspbr-242713', zone='us-central1-a')
        (self.old_folder/'helper.py').write_text('raise AssertionError("Never execute historical helper")\n')
        self.historical_sha = p.file_sha(self.old_folder/'helper.py')
        self.old = dict(schema='r02_m02_preservation_worker_v1', helper_sha256=self.historical_sha,
                        worker='worker05', chromosome=20, run_dir=str(self.run), run_sha256=run_sha,
                        boot_id='00000000-0000-0000-0000-000000000000',
                        instance_id=self.instance['id'], instance_name=self.instance['name'], files=[])
        held_files = []
        base = 'dnabr.hg38.2723.chr20.snv.bi.pass.vcf.gz'
        for index, name in enumerate((base, base+'.tbi')):
            source = self.held_folder/name
            source.write_bytes(b'synthetic-fixture-'+bytes([index]))
            original = self.bulk/'m02_chr20.ABC'/name
            self.old['files'].append(dict(name=name, source=str(original), bytes=source.stat().st_size))
            held_files.append(dict(name=name, path=str(source), source=str(original),
                                   bytes=source.stat().st_size, inode=source.stat().st_ino,
                                   device=999999))  # Reboot can change the old device number.
        old_sha = write(self.old_folder/'spec.json', self.old)
        self.held = dict(schema='r02_m02_held_v1', worker='worker05', chromosome=20,
                         spec_sha256=old_sha, files=held_files, m02_trace_sha256='a'*64,
                         m02_trace_task=dict(name='PREPROCESS_FILTER_SNV_BIALLELIC_PASS (chr20)',
                                             status='COMPLETED', exit='0'))
        held_sha = write(self.old_folder/'held.json', self.held)
        self.disk = dict(schema='r02_m02_disk_identity_v1', instance=self.instance,
                         disk=dict(id='1234', name='worker05-disk', device_name='persistent-disk-0',
                                   filesystem_uuid='85b8a44b-802c-4f59-aab0-ec802b57b6c6',
                                   source='https://www.googleapis.com/compute/v1/projects/uspbr-242713/'
                                          'zones/us-central1-a/disks/worker05-disk'))
        disk_sha = write(self.folder/'disk.json', self.disk)
        self.spec = dict(schema=p.SCHEMA, helper_sha256=p.file_sha(SOURCE.absolute()),
                         dependency_sha256={name:p.file_sha((SOURCE.parent/name).absolute()) for name in p.DEPENDENCIES},
                         worker='worker05', chromosome=20, run_dir=str(self.run), run_sha256=run_sha,
                         old_spec_path=str(self.old_folder/'spec.json'), old_spec_sha256=old_sha,
                         old_held_path=str(self.old_folder/'held.json'), old_held_sha256=held_sha,
                         instance=self.instance, boot_id='64fa668b-ce20-4930-9734-43dc9ae7f733',
                         disk_manifest_path=str(self.folder/'disk.json'), disk_manifest_sha256=disk_sha,
                         destination=p.PREFIX+'worker05/chr20/')
        self.path = self.folder/'spec.json'
        self.expected = write(self.path, self.spec)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(p, 'metadata_identity', return_value=dict(self.instance)))
        self.stack.enter_context(patch.object(p, 'boot_id', return_value=self.spec['boot_id']))
        self.mount = dict(source='/dev/sda1', fstype='ext4', filesystem_uuid=self.disk['disk']['filesystem_uuid'],
                          device_name='persistent-disk-0', block_device='/dev/sda')
        self.disk_mock = self.stack.enter_context(patch.object(p, 'check_disk', return_value=self.mount))
        self.stack.enter_context(patch.dict(os.environ))

    def update_spec(self):
        self.expected = write(self.path, self.spec)

    def update_old(self):
        self.spec['old_spec_sha256'] = write(self.old_folder/'spec.json', self.old)
        self.held['spec_sha256'] = self.spec['old_spec_sha256']
        self.update_held()

    def update_held(self):
        self.spec['old_held_sha256'] = write(self.old_folder/'held.json', self.held)
        self.update_spec()

    def validate(self):
        return p.validate(self.path, self.expected)

    def publish(self, cloud=None):
        return p.publish(self.path, self.expected, cloud or Cloud())

    def test_reboot_adopts_held_without_originals_or_old_device(self):
        result = self.validate()
        self.assertNotEqual(self.old['boot_id'], self.spec['boot_id'])
        self.assertFalse(self.bulk.exists())
        self.assertEqual(len(result['files']), 2)
        self.assertNotEqual(result['files'][0]['device'], result['files'][0]['original_device'])
        self.assertEqual(result['old_helper_sha256'], self.historical_sha)
        self.assertNotEqual(self.old['helper_sha256'], self.spec['dependency_sha256']['r02_preserve_m02.py'])

    def test_exact_eight_assignments(self):
        self.assertEqual(p.APPROVED_CHROMOSOMES, dict(worker05=20, worker06=19, worker07=18, worker08=15,
                                                    worker09=17, worker10=14, worker11=16, worker12=13))
        self.spec['chromosome'] = 19
        self.update_spec()
        with self.assertRaisesRegex(ValueError, 'assignment'):
            self.validate()

    def test_wrong_vm_or_new_boot_rejected(self):
        with patch.object(p, 'metadata_identity', return_value=dict(self.instance, id='9')):
            with self.assertRaisesRegex(ValueError, 'metadata'):
                self.validate()
        with patch.object(p, 'boot_id', return_value=self.old['boot_id']):
            with self.assertRaisesRegex(ValueError, 'boot'):
                self.validate()

    def test_namespace_and_worker_path_strict(self):
        self.spec['destination'] = self.spec['destination'].replace('/m02-preservation-', '/00_datos_y_diseno/m02-preservation-')
        self.update_spec()
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.validate()
        self.spec['destination'] = p.PREFIX+'worker05/chr20/'
        self.spec['run_dir'] = str(self.run.parent)
        self.update_spec()
        with self.assertRaisesRegex(ValueError, 'Run directory'):
            self.validate()

    def test_spec_dependency_old_run_and_held_hashes_bound(self):
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            p.validate(self.path, 'f'*64)
        self.spec['dependency_sha256']['r02_parallel_coordinator.py'] = 'f'*64
        self.update_spec()
        with self.assertRaisesRegex(ValueError, 'Dependency'):
            self.validate()
        self.spec['dependency_sha256']['r02_parallel_coordinator.py'] = p.file_sha((SOURCE.parent/'r02_parallel_coordinator.py').absolute())
        self.spec['old_held_sha256'] = 'f'*64
        self.update_spec()
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            self.validate()

    def test_original_relationship_and_duplicate_files_rejected(self):
        self.held['files'][0]['source'] += '.other'
        self.update_held()
        with self.assertRaisesRegex(ValueError, 'relationship'):
            self.validate()
        self.held['files'][0]['source'] = self.old['files'][0]['source']
        self.held['files'].append(self.held['files'][0])
        self.update_held()
        with self.assertRaisesRegex(ValueError, 'exactly'):
            self.validate()

    def test_changed_inode_size_and_symlink_rejected(self):
        source = Path(self.held['files'][0]['path'])
        source.write_bytes(b'different-size')
        with self.assertRaisesRegex(ValueError, 'inode/size'):
            self.validate()
        target = source.with_name('target')
        source.rename(target)
        source.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.validate()

    def test_json_duplicate_keys_rejected(self):
        self.path.write_text('{"schema":"one","schema":"two"}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.validate(self.path, p.file_sha(self.path))

    def test_disk_manifest_changed_vm_or_api_identity_rejected(self):
        self.disk['disk']['source'] = self.disk['disk']['source'].replace('uspbr-242713', 'another-project')
        self.spec['disk_manifest_sha256'] = write(self.folder/'disk.json', self.disk)
        self.update_spec()
        with self.assertRaisesRegex(ValueError, 'Disk API identity'):
            self.validate()
        self.assertFalse(self.disk_mock.called)

    def test_hash_publish_receipt_and_strict_idempotency(self):
        cloud = Cloud()
        first = self.publish(cloud)
        self.assertEqual(first['state'], 'PUBLISHED_VERIFIED')
        self.assertEqual(len(first['files']), 2)
        self.assertEqual(len(cloud.uploads), 3)
        self.assertEqual(self.publish(cloud), first)
        self.assertEqual(len(cloud.uploads), 3)
        receipt = json.loads((self.folder/'preservation.json').read_text())
        self.assertEqual(receipt['files'], first['files'])
        self.assertEqual(receipt['old_helper_sha256'], self.historical_sha)
        self.assertFalse(first['cleanup_authorized'])
        self.assertEqual(os.environ['CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED'], 'False')
        self.assertEqual(os.environ['CLOUDSDK_STORAGE_PROCESS_COUNT'], '1')
        self.assertEqual(os.environ['CLOUDSDK_STORAGE_THREAD_COUNT'], '1')

    def test_ack_lost_adopts_only_matching_object(self):
        cloud = Cloud()
        cloud.lose_ack = True
        self.assertEqual(self.publish(cloud)['state'], 'PUBLISHED_VERIFIED')
        self.assertEqual(len(cloud.uploads), 3)

    def test_corrupt_object_rejected_without_overwrite(self):
        cloud = Cloud()
        uri = self.spec['destination']+self.held['files'][0]['name']
        cloud.objects[uri] = dict(size=self.held['files'][0]['bytes'], generation='111', md5_hash='wrong')
        with self.assertRaisesRegex(ValueError, 'MD5'):
            self.publish(cloud)
        self.assertEqual(cloud.uploads, [])
        self.assertFalse((self.folder/'publication.json').exists())
        self.assertEqual(json.loads((self.folder/'status.json').read_text())['state'], 'FAILED_CLOSED')

    def test_existing_generation_changed_or_missing_never_recreated(self):
        cloud = Cloud()
        result = self.publish(cloud)
        uri = result['files'][0]['uri']
        cloud.objects[uri]['generation'] = '888'
        with self.assertRaisesRegex(ValueError, 'generation changed'):
            self.publish(cloud)
        del cloud.objects[uri]
        with self.assertRaisesRegex(ValueError, 'disappeared'):
            self.publish(cloud)
        self.assertEqual(len(cloud.uploads), 3)

    def test_upload_failure_cannot_mark_published(self):
        cloud = Cloud()
        with patch.object(cloud, 'upload', side_effect=RuntimeError('403 forbidden')):
            with self.assertRaises(RuntimeError):
                self.publish(cloud)
        self.assertFalse((self.folder/'publication.json').exists())
        self.assertEqual(json.loads((self.folder/'status.json').read_text())['state'], 'FAILED_CLOSED')

    def test_source_mutation_during_upload_rejected(self):
        cloud = Cloud()
        def mutate(source, uri):
            if not str(source).endswith('preservation.json'):
                with Path(source).open('ab') as stream:
                    stream.write(b'changed')
        cloud.after_upload = mutate
        with self.assertRaisesRegex(ValueError, 'changed while publishing'):
            self.publish(cloud)
        self.assertFalse((self.folder/'publication.json').exists())

    def test_receipt_upload_failure_not_final_success(self):
        cloud = Cloud()
        original = cloud.upload
        def upload(source, uri):
            if uri.endswith('preservation.json'):
                raise RuntimeError('receipt upload failed')
            return original(source, uri)
        with patch.object(cloud, 'upload', side_effect=upload):
            with self.assertRaises(RuntimeError):
                self.publish(cloud)
        self.assertTrue((self.folder/'preservation.json').exists())
        self.assertFalse((self.folder/'publication.json').exists())
        self.assertEqual(json.loads((self.folder/'status.json').read_text())['state'], 'FAILED_CLOSED')
        self.assertEqual(self.publish(cloud)['state'], 'PUBLISHED_VERIFIED')

    def test_exclusive_lock_rejects_second_publisher(self):
        with p.exclusive(self.folder):
            with self.assertRaises(BlockingIOError):
                self.publish()

    def test_cli_verify_writes_nothing_and_cannot_hold_or_cleanup(self):
        before = sorted(str(v) for v in self.root.rglob('*'))
        output = io.StringIO()
        args = ['publisher', 'verify', '--spec', str(self.path), '--spec-sha256', self.expected]
        with patch('sys.argv', args), redirect_stdout(output):
            p.main()
        self.assertEqual(json.loads(output.getvalue())['state'], 'READY_NOT_PUBLISHED')
        self.assertEqual(sorted(str(v) for v in self.root.rglob('*')), before)
        with patch('sys.argv', ['publisher', 'hold']), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                p.main()

    def test_real_gcs_methods_create_only_and_403_not_optional_404(self):
        context = self.validate()
        cloud = p.cloud_client(context['preserve'], context['coordinator'])
        error = subprocess.CompletedProcess([], 1, '', 'ERROR: HTTPError 403 forbidden gs://bucket/404')
        with patch.object(subprocess, 'run', return_value=error):
            with self.assertRaisesRegex(RuntimeError, 'Cannot authenticate'):
                cloud.metadata('gs://bucket/404', optional=True)
        success = subprocess.CompletedProcess([], 0, '', '')
        with patch.object(subprocess, 'run', return_value=success) as call:
            cloud.upload(Path('/tmp/synthetic'), 'gs://bucket/object')
        self.assertEqual(call.call_args.args[0], ['gcloud', 'storage', 'cp', '--if-generation-match=0',
                                               '/tmp/synthetic', 'gs://bucket/object'])

    def test_signature_includes_ctime_independent_of_size_and_mtime(self):
        # A same-tick rewrite may leave actual FS timestamps indistinguishable.
        # Test the signature contract, not a guaranteed clock resolution.
        source = Mock()
        fields = dict(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
        source.stat.side_effect = [SimpleNamespace(**fields, st_ctime_ns=5),
                                   SimpleNamespace(**fields, st_ctime_ns=6)]
        with patch.object(p, 'safe_path', return_value=source):
            before, after = p.signature('/synthetic'), p.signature('/synthetic')
        self.assertEqual(before['mtime_ns'], after['mtime_ns'])
        self.assertEqual(before['bytes'], after['bytes'])
        self.assertNotEqual(before['ctime_ns'], after['ctime_ns'])

    def test_historical_helper_hash_is_independently_checked(self):
        (self.old_folder/'helper.py').write_text('changed historical producer\n')
        with self.assertRaisesRegex(ValueError, 'Historical retention helper'):
            self.validate()

    def test_source_mutation_during_hashing_rejected(self):
        original_dependencies = p.dependencies
        def changed_dependencies(spec):
            preserve, coordinator = original_dependencies(spec)
            digest = preserve.digests
            def digests(path):
                result = digest(path)
                with Path(path).open('ab') as stream:
                    stream.write(b'changed')
                return result
            preserve.digests = digests
            return preserve, coordinator
        cloud = Cloud()
        with patch.object(p, 'dependencies', side_effect=changed_dependencies):
            with self.assertRaisesRegex(ValueError, 'changed while hashing'):
                self.publish(cloud)
        self.assertFalse(cloud.uploads)

    def test_disk_partition_parent_uuid_and_ext4_real_validator(self):
        # Directly exercise the validator; only host utilities/device paths mocked.
        context = self.validate()
        preserve = context['preserve']
        mount = dict(source='/dev/sda1', fstype='ext4', uuid=self.disk['disk']['filesystem_uuid'])
        original_resolve = Path.resolve
        def resolved(path, *args, **kwargs):
            if str(path) == '/dev/disk/by-id/google-persistent-disk-0':
                return Path('/dev/sda')
            return original_resolve(path, *args, **kwargs)
        # check_disk was patched at setup; obtain the unchanged source function.
        pristine = importlib.util.module_from_spec(MODULE)
        MODULE.loader.exec_module(pristine)
        for fstype, uuid, parent, succeeds in [
                ('ext4', mount['uuid'], 'sda', True),
                ('fuse', mount['uuid'], 'sda', False),
                ('ext4', 'wrong', 'sda', False),
                ('ext4', mount['uuid'], 'sdb', False)]:
            with self.subTest(fstype=fstype, uuid=uuid, parent=parent):
                response = json.dumps(dict(filesystems=[dict(mount, fstype=fstype, uuid=uuid)]))
                with patch.object(preserve, 'command', side_effect=[response, parent+'\n']), \
                        patch.object(Path, 'exists', return_value=True), patch.object(Path, 'resolve', resolved):
                    if succeeds:
                        self.assertEqual(pristine.check_disk(self.held_folder, self.disk['disk'], preserve)['block_device'],
                                         '/dev/sda')
                    else:
                        with self.assertRaises(ValueError):
                            pristine.check_disk(self.held_folder, self.disk['disk'], preserve)

    def test_publish_never_invokes_old_hold_publish_or_cleanup(self):
        original_dependencies = p.dependencies
        def no_legacy_mutation(spec):
            preserve, coordinator = original_dependencies(spec)
            def forbidden(*args, **kwargs):
                raise AssertionError('Historical operational action must never be invoked')
            preserve.retain = preserve.publish = preserve.cleanup_guard = preserve.load_spec = forbidden
            return preserve, coordinator
        with patch.object(p, 'dependencies', side_effect=no_legacy_mutation):
            self.assertEqual(self.publish()['state'], 'PUBLISHED_VERIFIED')


if __name__ == '__main__':
    unittest.main()
