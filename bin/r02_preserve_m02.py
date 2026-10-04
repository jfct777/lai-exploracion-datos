#!/usr/bin/env python3
"""Operational M02 retention/publication; no genotype analysis or VM mutations.

Worker publication is create-only and independently resumable. This helper
never reactivates cleanup: deletion requires a separate authorized action.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fixed(path, value):
    path = Path(path)
    if path.exists():
        require(not path.is_symlink() and read(path) == value, 'Existing immutable receipt differs')
        return
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def status(folder, **value):
    fd, temporary = tempfile.mkstemp(prefix='.preservation-', dir=folder)
    with os.fdopen(fd, 'w') as stream:
        json.dump(dict(updated_utc=now(), **value), stream, sort_keys=True, allow_nan=False)
    os.replace(temporary, Path(folder)/'status.json')


def command(args, timeout=120):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Command failed: '+result.stderr[-1500:])
    return result.stdout


def signature(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def digests(path):
    before = signature(path)
    h, m = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
            m.update(block)
    require(signature(path) == before, 'Completed M02 changed while hashing')
    return dict(bytes=before[2], sha256=h.hexdigest(), md5_base64=base64.b64encode(m.digest()).decode())


def validate_m02_trace(path, chromosome):
    expected = f'PREPROCESS_FILTER_SNV_BIALLELIC_PASS (chr{chromosome})'
    with Path(path).open() as stream:
        rows = [r for r in csv.DictReader(stream, delimiter='\t') if r['name'] == expected]
    require(rows and rows[-1]['status'] in ('COMPLETED', 'CACHED')
            and rows[-1].get('exit') == '0', 'M02 has no genuine successful trace')
    return rows[-1]


def load_spec(path, expected, schema):
    path = Path(path).resolve()
    require(sha(path) == expected, 'Preservation specification changed')
    spec = read(path)
    require(spec['schema'] == schema, 'Wrong preservation schema')
    require(sha(__file__) == spec['helper_sha256'], 'Preservation helper source changed')
    return path, spec


def retain(spec_path, expected):
    path, spec = load_spec(spec_path, expected, 'r02_m02_preservation_worker_v1')
    run = Path(spec['run_dir'])
    require(run.is_absolute() and run.resolve() == run, 'Invalid worker run')
    require(sha(run/'run.json') == spec['run_sha256'], 'Frozen worker run changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == spec['boot_id'], 'Worker rebooted')
    require(sha(spec['reboot_spec_path']) == spec['reboot_spec_sha256'], 'Storage identity changed')
    reboot, cfg = read(spec['reboot_spec_path']), read(run/'run.json')
    chromosome = spec['chromosome']
    require(type(chromosome) is int and 13 <= chromosome <= 20
            and chromosome in cfg['processing_order'], 'Chromosome outside authorized retention')
    require(cfg['parallel_worker']['worker_id'] == spec['worker'], 'Worker identity differs')
    require(path.parent == run/'repairs/m02-preservation-20261002', 'Unexpected retention folder')
    local, bulk = Path(reboot['local_bulk']), Path(cfg['bulk'])
    require(local.samefile(bulk), 'Bulk is not the recorded local mount')
    mount = json.loads(command(['findmnt', '-J', '-T', str(local), '-o', 'FSTYPE']))['filesystems'][0]
    require(mount['fstype'] == 'ext4', 'Retention requires local ext4')
    trace = run/f'chr{chromosome:02d}/trace.tsv'
    task = validate_m02_trace(trace, chromosome)
    base = f'dnabr.hg38.2723.chr{chromosome}.snv.bi.pass.vcf.gz'
    require({x['name'] for x in spec['files']} == {base, base+'.tbi'}, 'Require exact VCF and TBI')
    held = path.parent/'held'
    held.mkdir(mode=0o700, exist_ok=True)
    files = []
    for record in spec['files']:
        source = Path(record['source'])
        require(source.parent.parent == bulk and source.name == record['name']
                and re.fullmatch(r'm02_chr'+str(chromosome)+r'\.[A-Za-z0-9]+', source.parent.name),
                'Source escaped authenticated M02 directory')
        physical = local/source.relative_to(bulk)
        target = held/record['name']
        # Reuse a held inode even if the normal analysis has since unlinked bulk.
        if target.exists():
            require(not target.is_symlink() and target.stat().st_size == record['bytes'], 'Retained file changed')
            if physical.exists():
                require(physical.samefile(target), 'Retention points to another inode')
        else:
            require(physical.is_file() and not physical.is_symlink()
                    and source.samefile(physical) and physical.stat().st_size == record['bytes'],
                    'Completed source missing or changed')
            require(physical.stat().st_dev == held.stat().st_dev, 'Hardlink crosses filesystems')
            os.link(physical, target)
            require(physical.samefile(target), 'Hardlink retention failed')
        files.append(dict(name=record['name'], path=str(target), source=str(source),
                          bytes=target.stat().st_size, inode=target.stat().st_ino,
                          device=target.stat().st_dev))
    hold_path = path.parent/'held.json'
    if hold_path.exists():
        held_receipt = read(hold_path)
        require(held_receipt['spec_sha256'] == expected and held_receipt['files'] == files,
                'Prior held receipt differs')
    else:
        held_receipt = dict(schema='r02_m02_held_v1', worker=spec['worker'], chromosome=chromosome,
                            spec_sha256=expected, utc=now(), files=files,
                            m02_trace_task=task, m02_trace_sha256=sha(trace))
        fixed(hold_path, held_receipt)
    status(path.parent, state='HELD_LOCAL_NOT_YET_PUBLISHED', worker=spec['worker'])
    return path, spec, held_receipt


class GCS:
    def metadata(self, uri, optional=False):
        result = subprocess.run(['gcloud', 'storage', 'objects', 'describe', uri, '--format=json'],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            if optional and ('not found: 404' in result.stderr or 'NOT_FOUND: 404' in result.stderr):
                return None
            raise RuntimeError(result.stderr[-1500:])
        return json.loads(result.stdout)

    def read(self, uri, generation):
        return json.loads(command(['gcloud', 'storage', 'cat', uri+'#'+str(generation)]))

    def upload(self, source, uri):
        command(['gcloud', 'storage', 'cp', '--if-generation-match=0', str(source), uri], timeout=14400)


def verify_metadata(metadata, record):
    require(int(metadata['size']) == record['bytes'], 'Published size mismatch')
    require(metadata.get('md5_hash', metadata.get('md5Hash')) == record['md5_base64'], 'Published MD5 mismatch')
    if 'generation' in record:
        require(str(metadata['generation']) == str(record['generation']), 'Published generation changed')


def destination(spec):
    root = ('gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/'
            'R02_20260930/r02-autosomes-20261001b/00_datos_y_diseno/m02-preservation-20261002/')
    expected = root+spec['worker']+f'/chr{spec["chromosome"]:02d}/'
    require(spec['destination'] == expected, 'Destination escaped authorized preservation prefix')
    return expected


def publish(spec_path, expected, cloud=None):
    path, spec, held = retain(spec_path, expected)
    cloud = cloud or GCS()
    root = destination(spec)
    records = []
    for item in held['files']:
        source = Path(item['path'])
        status(path.parent, state='HASHING', file=item['name'])
        record = dict(name=item['name'], uri=root+item['name'], **digests(source))
        require(record['bytes'] == item['bytes'], 'Held size changed')
        metadata = cloud.metadata(record['uri'], optional=True)
        if metadata is None:
            status(path.parent, state='UPLOADING_CREATE_ONLY', file=item['name'])
            cloud.upload(source, record['uri'])
            metadata = cloud.metadata(record['uri'])
        verify_metadata(metadata, record)
        record['generation'] = str(metadata['generation'])
        records.append(record)
    receipt_path = path.parent/'preservation.json'
    receipt = dict(schema='r02_m02_preserved_v1', worker=spec['worker'], chromosome=spec['chromosome'],
                   spec_sha256=expected, run_sha256=spec['run_sha256'], helper_sha256=spec['helper_sha256'],
                   held_receipt_sha256=sha(path.parent/'held.json'), files=records,
                   state='PUBLISHED_VERIFIED', scientific_parameters_changed=False)
    fixed(receipt_path, receipt)
    metadata = cloud.metadata(root+'preservation.json', optional=True)
    if metadata is None:
        cloud.upload(receipt_path, root+'preservation.json')
        metadata = cloud.metadata(root+'preservation.json')
    verify_metadata(metadata, digests(receipt_path))
    status(path.parent, state='PUBLISHED_VERIFIED', receipt_uri=root+'preservation.json',
           receipt_generation=str(metadata['generation']), receipt_sha256=sha(receipt_path))
    return receipt


def verify_preservation(target, cloud):
    uri = target['destination']+'preservation.json'
    metadata = cloud.metadata(uri)
    receipt = cloud.read(uri, metadata['generation'])
    require(receipt['schema'] == 'r02_m02_preserved_v1' and receipt['state'] == 'PUBLISHED_VERIFIED'
            and receipt['spec_sha256'] == target['spec_sha256']
            and receipt['worker'] == target['worker'] and receipt['chromosome'] == target['chromosome'],
            'Preservation receipt identity differs')
    expected = {x['name']: x for x in target['files']}
    require(len(receipt['files']) == 2 and {x['name'] for x in receipt['files']} == set(expected),
            'Incomplete preservation receipt')
    for record in receipt['files']:
        require(record['uri'] == target['destination']+record['name']
                and record['bytes'] == expected[record['name']]['bytes']
                and re.fullmatch('[a-f0-9]{64}', record['sha256']), 'Unexpected preserved object')
        verify_metadata(cloud.metadata(record['uri']), record)
    return receipt


def cleanup_guard(spec_path, expected):
    raise PermissionError('Cleanup is disabled; publication does not authorize deleting VMs or disks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('hold', 'publish', 'cleanup-guard'))
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == 'hold':
        _, _, receipt = retain(args.spec, args.spec_sha256)
        print(json.dumps(receipt))
    elif args.mode == 'publish':
        print(json.dumps(publish(args.spec, args.spec_sha256)))
    else:
        cleanup_guard(args.spec, args.spec_sha256)


if __name__ == '__main__':
    main()
