#!/usr/bin/env python3
"""Publish authenticated, previously held M02 bytes; never retain, clean or boot.

Delta from retention v1: an explicit new-boot/disk adoption and the newly
authorized destination. Neither the old receipts nor scientific files change.
The controller authenticates disk ID/source through the Compute API; this
helper verifies that attestation's hash, live VM metadata and local ext4 UUID.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import types
from urllib.request import Request, urlopen


SCHEMA = 'r02_m02_publication_worker_v2'
FOLDER = 'm02-publication-approved-20261002'
RUN_ID = 'r02-autosomes-20261001b'
PREFIX = ('gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/'
          'biologico/R02_20260930/'+RUN_ID+'/m02-preservation-20261002/')
DEPENDENCIES = ('r02_preserve_m02.py', 'r02_parallel_coordinator.py')
APPROVED_CHROMOSOMES = {'worker05': 20, 'worker06': 19, 'worker07': 18, 'worker08': 15,
                        'worker09': 17, 'worker10': 14, 'worker11': 16, 'worker12': 13}


def require(value, message):
    if not value:
        raise ValueError(message)


def safe_path(value, *, file=True):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and path.resolve() == path,
            'Path is not absolute/canonical or contains symlinks')
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), 'Symlink in authenticated path')
    if file:
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'Not a regular file')
    return path


def file_sha(path):
    with safe_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(safe_path(path).read_text(), object_pairs_hook=unique)


def authenticated_json(path, expected):
    require(re.fullmatch('[0-9a-f]{64}', expected or ''), 'Invalid SHA256')
    require(file_sha(path) == expected, 'Authenticated JSON hash changed')
    return read_json(path)


def dependencies(spec):
    require(set(spec['dependency_sha256']) == set(DEPENDENCIES), 'Dependency set differs')
    modules = []
    for name in DEPENDENCIES:
        path = Path(__file__).resolve().parent/name
        # Authenticate bytes before executing them (no mutable import search path).
        source = safe_path(path).read_bytes()
        require(hashlib.sha256(source).hexdigest() == spec['dependency_sha256'][name],
                'Dependency source changed: '+name)
        module = types.ModuleType('_publication_'+path.stem)
        module.__file__ = str(path)
        exec(compile(source, str(path), 'exec'), module.__dict__)
        modules.append(module)
    return tuple(modules)


def signature(path):
    value = safe_path(path).stat()
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def metadata_identity():
    values = {}
    for name, key in (('id', 'instance/id'), ('name', 'instance/name'),
                      ('zone', 'instance/zone'), ('project', 'project/project-id')):
        request = Request('http://metadata.google.internal/computeMetadata/v1/'+key,
                          headers={'Metadata-Flavor': 'Google'})
        with urlopen(request, timeout=5) as response:
            require(response.headers.get('Metadata-Flavor') == 'Google', 'Not GCE metadata')
            values[name] = response.read(4096).decode().strip()
    values['zone'] = values['zone'].split('/')[-1]
    return values


def check_disk(held_directory, disk, preserve):
    mount = json.loads(preserve.command(['findmnt', '-J', '-T', str(held_directory),
                                         '-o', 'SOURCE,FSTYPE,UUID']))['filesystems']
    require(len(mount) == 1, 'Ambiguous held mount')
    mount = mount[0]
    require(mount['fstype'] == 'ext4' and mount['uuid'] == disk['filesystem_uuid'],
            'Held files are not on attested ext4 UUID')
    require(re.fullmatch('[A-Za-z0-9_-]+', disk['device_name']), 'Unsafe disk device name')
    source = Path(mount['source'])
    require(source.is_absolute() and str(source).startswith('/dev/'), 'Not a local block device')
    device = Path('/dev/disk/by-id/google-'+disk['device_name'])
    require(device.exists(), 'Attested Google disk device is absent')
    parent = preserve.command(['lsblk', '-n', '-o', 'PKNAME', str(source)]).strip()
    require(len(parent.splitlines()) <= 1, 'Ambiguous block-device parent')
    actual = (Path('/dev')/parent if parent and not parent.startswith('/') else
              Path(parent) if parent else source)
    require(actual.resolve() == device.resolve(), 'Filesystem is on another disk')
    return dict(source=str(source), filesystem_uuid=mount['uuid'], fstype='ext4',
                device_name=disk['device_name'], block_device=str(actual.resolve()))


def validate(spec_path, expected):
    path = safe_path(spec_path)
    spec = authenticated_json(path, expected)
    require(spec['schema'] == SCHEMA, 'Wrong publication schema')
    require(file_sha(Path(__file__).absolute()) == spec['helper_sha256'], 'Publisher source changed')
    preserve, coordinator = dependencies(spec)
    run = safe_path(spec['run_dir'], file=False)
    worker, chromosome = spec['worker'], spec['chromosome']
    require(re.fullmatch(r'worker(?:0[5-9]|1[0-2])', worker or ''), 'Worker outside approval')
    require(type(chromosome) is int and chromosome == APPROVED_CHROMOSOMES[worker],
            'Worker/chromosome outside approved assignment')
    require(run.name == worker and run.parent.name == 'parallel' and run.parent.parent.name == RUN_ID,
            'Run directory does not identify the approved worker')
    require(path == run/'repairs'/FOLDER/'spec.json', 'Wrong publication specification location')
    cfg = authenticated_json(run/'run.json', spec['run_sha256'])
    require(cfg['parallel_worker']['worker_id'] == worker and chromosome in cfg['processing_order'],
            'Frozen run identity differs')
    instance = spec['instance']
    require(set(instance) == {'id', 'name', 'zone', 'project'} and
            all(isinstance(value, str) and value for value in instance.values()), 'Invalid instance identity')
    require(instance['name'] == 'dnabr-r02p-1001-w'+worker[-2:] and instance['id'].isdigit(),
            'Wrong instance assignment')
    require(metadata_identity() == instance, 'Live VM metadata differs')
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', spec['boot_id'])
            and boot_id() == spec['boot_id'], 'Publication boot differs')
    root = PREFIX+worker+f'/chr{chromosome:02d}/'
    require(spec['destination'] == root, 'Destination escaped newly authorized prefix')
    old_folder = run/'repairs/m02-preservation-20261002'
    require(Path(spec['old_spec_path']) == old_folder/'spec.json' and
            Path(spec['old_held_path']) == old_folder/'held.json', 'Wrong retained provenance paths')
    old = authenticated_json(spec['old_spec_path'], spec['old_spec_sha256'])
    held = authenticated_json(spec['old_held_path'], spec['old_held_sha256'])
    require(old['schema'] == 'r02_m02_preservation_worker_v1' and held['schema'] == 'r02_m02_held_v1',
            'Wrong retention provenance schema')
    require(re.fullmatch('[0-9a-f]{64}', old['helper_sha256']), 'Invalid historical helper hash')
    require(file_sha(old_folder/'helper.py') == old['helper_sha256'], 'Historical retention helper changed')
    for key in ('worker', 'chromosome', 'run_dir', 'run_sha256'):
        require(old[key] == spec[key], 'Original retention/run relationship differs')
    require(str(old['instance_id']) == instance['id'] and old['instance_name'] == instance['name'],
            'Original retention VM differs')
    require(held['spec_sha256'] == spec['old_spec_sha256'] and held['worker'] == worker
            and held['chromosome'] == chromosome, 'Held receipt relationship differs')
    require(held['m02_trace_task']['name'] == f'PREPROCESS_FILTER_SNV_BIALLELIC_PASS (chr{chromosome})'
            and held['m02_trace_task']['status'] in ('COMPLETED', 'CACHED')
            and held['m02_trace_task']['exit'] == '0', 'Held M02 was not successful')
    require(re.fullmatch('[0-9a-f]{64}', held['m02_trace_sha256']), 'Invalid retained trace hash')
    disk_manifest = authenticated_json(spec['disk_manifest_path'], spec['disk_manifest_sha256'])
    require(disk_manifest['schema'] == 'r02_m02_disk_identity_v1'
            and disk_manifest['instance'] == instance, 'Disk attestation VM differs')
    disk = disk_manifest['disk']
    require(str(disk['id']).isdigit() and re.fullmatch('[a-z][a-z0-9-]*', disk['name'])
            and re.fullmatch(r'[0-9a-fA-F-]{36}', disk['filesystem_uuid']), 'Invalid disk attestation')
    suffix = '/projects/'+instance['project']+'/zones/'+instance['zone']+'/disks/'+disk['name']
    require(disk['source'] in ('https://www.googleapis.com/compute/v1'+suffix,
                              'https://compute.googleapis.com/compute/v1'+suffix), 'Disk API identity differs')
    mount = check_disk(old_folder/'held', disk, preserve)
    base = f'dnabr.hg38.2723.chr{chromosome}.snv.bi.pass.vcf.gz'
    names = {base, base+'.tbi'}
    require(len(old['files']) == len(held['files']) == 2 and
            {v['name'] for v in old['files']} == {v['name'] for v in held['files']} == names,
            'Require exactly the original VCF and TBI')
    originals = {v['name']: v for v in old['files']}
    current = []
    bulk = Path(cfg['bulk'])
    require(bulk.is_absolute() and '..' not in bulk.parts, 'Invalid original bulk path')
    for record in sorted(held['files'], key=lambda value: value['name']):
        original, source = originals[record['name']], Path(record['path'])
        require(source == old_folder/'held'/record['name'], 'Held path escaped original retention')
        before = signature(source)
        require(before['device'] == (old_folder/'held').stat().st_dev,
                'Held file is not on the authenticated filesystem')
        require(type(record['bytes']) is int and record['bytes'] > 0 and
                before['bytes'] == record['bytes'] == original['bytes'] and
                before['inode'] == record['inode'], 'Held inode/size differs')
        original_path = Path(original['source'])
        require(record['source'] == original['source'] and original_path.name == record['name']
                and original_path.parent.parent == bulk and
                re.fullmatch('m02_chr'+str(chromosome)+r'\.[A-Za-z0-9]+', original_path.parent.name),
                'Original file relationship differs')
        # st_dev can change after reboot. Disk ID/source + UUID authenticate the
        # new mapping, while the retained inode and size must remain unchanged.
        current.append(dict(name=record['name'], path=str(source), original_device=record['device'],
                            **before))
    return dict(path=path, spec=spec, held=held, old_helper_sha256=old['helper_sha256'], files=current, mount=mount,
                preserve=preserve, coordinator=coordinator, expected=expected)


@contextmanager
def exclusive(folder):
    lock = safe_path(Path(folder)/'publish.lock', file=False)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def configure_uploads():
    # Composite uploads create/delete temporary objects and omit object MD5.
    # These process-local settings avoid both behaviors; no gcloud config write.
    os.environ.update(CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED='False',
                      CLOUDSDK_STORAGE_PROCESS_COUNT='1', CLOUDSDK_STORAGE_THREAD_COUNT='1')


def cloud_client(preserve, coordinator):
    class GCS(preserve.GCS):
        metadata = coordinator.GCS.metadata
    return GCS()


def verified_object(cloud, source, uri, record, preserve):
    metadata = cloud.metadata(uri, optional=True)
    if metadata is None:
        require('generation' not in record, 'Previously verified GCS generation disappeared')
        try:
            cloud.upload(source, uri)
        except (RuntimeError, subprocess.TimeoutExpired):
            # ACK loss / racing create: never repeat with overwrite semantics.
            # Unknown metadata, 403 and transient errors stay errors.
            metadata = cloud.metadata(uri, optional=True)
            if metadata is None:
                raise
        else:
            metadata = cloud.metadata(uri)
    preserve.verify_metadata(metadata, record)
    require(re.fullmatch('[1-9][0-9]*', str(metadata.get('generation', ''))), 'Invalid GCS generation')
    return str(metadata['generation'])


def publish(spec_path, expected, cloud=None):
    # Lock before adoption, hashes or cloud writes. Invalid specs cannot redirect it.
    initial = validate(spec_path, expected)
    folder = initial['path'].parent
    with exclusive(folder):
        context = validate(spec_path, expected)
        p, spec = context['preserve'], context['spec']
        configure_uploads()
        cloud = cloud or cloud_client(p, context['coordinator'])
        adoption = dict(schema='r02_m02_publication_adoption_v1', spec_sha256=expected,
                        run_sha256=spec['run_sha256'], instance=spec['instance'], boot_id=spec['boot_id'],
                        old_held_sha256=spec['old_held_sha256'], old_spec_sha256=spec['old_spec_sha256'],
                        old_helper_sha256=context['old_helper_sha256'],
                        disk_manifest_sha256=spec['disk_manifest_sha256'], mount=context['mount'],
                        files=context['files'])
        p.fixed(folder/'adoption.json', adoption)
        objects = safe_path(folder/'objects', file=False)
        objects.mkdir(mode=0o700, exist_ok=True)
        records = []
        try:
            for item in context['files']:
                source = Path(item['path'])
                before = signature(source)
                require(all(before[key] == item[key] for key in before), 'Held source changed after adoption')
                p.status(folder, state='HASHING', file=item['name'], spec_sha256=expected)
                record = dict(name=item['name'], uri=spec['destination']+item['name'], **p.digests(source))
                require(signature(source) == before, 'Held source changed while hashing')
                object_receipt = objects/(item['name']+'.json')
                if object_receipt.exists():
                    prior = read_json(object_receipt)
                    require(prior['spec_sha256'] == expected and
                            {k: prior[k] for k in record} == record, 'Existing object receipt differs')
                    record['generation'] = prior['generation']
                p.status(folder, state='UPLOADING_CREATE_ONLY', file=item['name'], spec_sha256=expected)
                record['generation'] = verified_object(cloud, source, record['uri'], record, p)
                require(signature(source) == before, 'Held source changed while publishing')
                p.fixed(object_receipt, dict(spec_sha256=expected, **record))
                records.append(record)
            receipt = dict(schema='r02_m02_preserved_v2', state='PUBLISHED_VERIFIED',
                           worker=spec['worker'], chromosome=spec['chromosome'], spec_sha256=expected,
                           run_sha256=spec['run_sha256'], helper_sha256=spec['helper_sha256'],
                           dependency_sha256=spec['dependency_sha256'], boot_id=spec['boot_id'],
                           instance=spec['instance'], old_spec_sha256=spec['old_spec_sha256'],
                           old_held_sha256=spec['old_held_sha256'],
                           old_helper_sha256=context['old_helper_sha256'],
                           disk_manifest_sha256=spec['disk_manifest_sha256'],
                           adoption_sha256=p.sha(folder/'adoption.json'), destination=spec['destination'],
                           files=records, scientific_parameters_changed=False, cleanup_authorized=False)
            receipt_path = folder/'preservation.json'
            p.fixed(receipt_path, receipt)
            p.status(folder, state='PUBLISHING_RECEIPT', spec_sha256=expected)
            final_path = folder/'publication.json'
            receipt_record = dict(uri=spec['destination']+'preservation.json', **p.digests(receipt_path))
            if final_path.exists():
                final = read_json(final_path)
                require(final['spec_sha256'] == expected and
                        {k: final['receipt'][k] for k in receipt_record} == receipt_record,
                        'Existing final publication proof differs')
                receipt_record['generation'] = final['receipt']['generation']
            receipt_record['generation'] = verified_object(cloud, receipt_path, receipt_record['uri'],
                                                          receipt_record, p)
            # Recheck both files and remote generations after the receipt upload.
            for item, record in zip(context['files'], records):
                current = signature(item['path'])
                require(all(current[key] == item[key] for key in current),
                        'Held source changed before final verification')
                p.verify_metadata(cloud.metadata(record['uri']), record)
            final = dict(schema='r02_m02_publication_complete_v1', state='PUBLISHED_VERIFIED',
                         spec_sha256=expected, worker=spec['worker'], chromosome=spec['chromosome'],
                         receipt=receipt_record, files=records, cleanup_authorized=False)
            p.fixed(final_path, final)
            p.status(folder, **final)
            return final
        except Exception as error:
            p.status(folder, state='FAILED_CLOSED', spec_sha256=expected, error_type=type(error).__name__)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'publish'))
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == 'verify':
        context = validate(args.spec, args.spec_sha256)
        result = dict(schema='r02_m02_publication_preflight_v1', state='READY_NOT_PUBLISHED',
                      spec_sha256=args.spec_sha256, worker=context['spec']['worker'],
                      chromosome=context['spec']['chromosome'], files=context['files'], mount=context['mount'])
    else:
        result = publish(args.spec, args.spec_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
