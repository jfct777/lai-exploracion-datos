#!/usr/bin/env python3
"""Run the frozen storage helper and durably publish its small operational evidence.

No scientific manifest is edited. No genotype files, credentials, arbitrary
directories or environment variables are uploaded. Default is verification only;
--run invokes the already authorized helper and create-only private publication.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


HELPER_SHA256 = '796c4eac50fb002cf5190bc3dbeb84d7b8d49ffb06725e51b515fc35215fd932'
MOUNT = Path('/home/jose.tantalean/gcs-dnabr')
BUCKET = 'gs://projects-usp/dnaBr-lai/datalake/'
MAX_SMALL_BYTES = 8*1024*1024
SPEC_FIELDS = {'schema', 'run_dir', 'helper_sha256', 'worker_frozen_sha256',
               'worker_run_sha256', 'chromosome', 'task_dir', 'task_command_sha256',
               'parameters_json_sha256', 'runtime_config_sha256', 'nextflow', 'runner',
               'timeout_seconds', 'poll_seconds'}
PUBLISHED_NAMES = ('helper.py', 'spec.json', 'events.jsonl', 'wrapper_execution.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def small_file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_SMALL_BYTES,
            'Expected a regular, small operational evidence file')
    return path


def read(path):
    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, 'Duplicate JSON key')
            output[key] = value
        return output
    return json.loads(small_file(path).read_text(), object_pairs_hook=unique)


def write_fixed(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Existing operational evidence differs')
        return
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def copy_fixed(source, destination):
    small_file(source)
    if destination.exists():
        require(sha(small_file(destination)) == sha(source), 'Frozen local evidence differs')
        return
    with destination.open('xb') as output, Path(source).open('rb') as original:
        shutil.copyfileobj(original, output)
    require(sha(destination) == sha(source), 'Frozen evidence copy differs')


def digests(path):
    data = small_file(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                md5_base64=base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode())


def validate(helper, spec, spec_sha256):
    helper, spec = small_file(helper).resolve(), small_file(spec).resolve()
    require(re.fullmatch(r'[a-f0-9]{64}', spec_sha256) is not None and sha(spec) == spec_sha256,
            'Specification hash mismatch')
    require(sha(helper) == HELPER_SHA256, 'Helper is not the frozen approved version')
    settings = read(spec)
    require(settings.get('schema') == 'r02_local_storage_boundary_v1'
            and set(settings) <= SPEC_FIELDS, 'Unexpected storage specification fields')
    require(settings['helper_sha256'] == HELPER_SHA256, 'Specification names a different helper')
    run = Path(settings['run_dir'])
    require(run.is_absolute() and run.resolve() == run and run.is_dir(), 'Invalid worker directory')
    require(sha(small_file(run/'run.json')) == settings['worker_run_sha256']
            and sha(small_file(run/'frozen.sha256.json')) == settings['worker_frozen_sha256'],
            'Worker scientific manifest changed')
    config = read(run/'run.json')
    worker = config['parallel_worker']['worker_id']
    parent = config['parallel_worker']['parent_run_id']
    require(re.fullmatch(r'worker[0-9]{2}', worker) is not None
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{1,100}', parent) is not None,
            'Unsafe worker identity')
    destination = MOUNT/'refined/DNABR_QC/presentacion/biologico/R02_20260930'/parent/'parallel'/worker
    require(Path(config['destination']) == destination, 'Publication escaped the existing private worker destination')
    prefix = BUCKET + str(destination.relative_to(MOUNT)) + '/00_worker_provenance/storage_transition/' + spec_sha256
    return dict(helper=helper, spec=spec, spec_sha256=spec_sha256, run=run,
                directory=run/'storage_transition'/spec_sha256[:16], prefix=prefix,
                worker_id=worker, worker_run_sha256=settings['worker_run_sha256'])


class GCS:
    def upload(self, source, uri):
        digest = digests(source)
        result = subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0',
                                 '--content-md5='+digest['md5_base64'], str(source), uri],
                                capture_output=True, text=True, timeout=600,
                                env={**os.environ, 'CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED':'false'})
        require(result.returncode == 0 or 'HTTPError 412:' in result.stderr,
                'Create-only operational evidence upload failed (credential output suppressed)')
        description = subprocess.run(['gcloud', 'storage', 'objects', 'describe', uri, '--format=json'],
                                     capture_output=True, text=True, timeout=120)
        require(description.returncode == 0, 'Operational evidence metadata verification failed')
        meta = json.loads(description.stdout)
        require(str(meta.get('generation', '')).isdigit()
                and int(meta.get('size', -1)) == digest['bytes']
                and meta.get('md5_hash', meta.get('md5Hash')) == digest['md5_base64'],
                'Remote operational evidence differs from the local file')
        require(digests(source) == digest, 'Local evidence changed during publication')
        return dict(name=Path(source).name, uri=uri, generation=str(meta['generation']), **digest)


def publish(context, cloud):
    directory, prefix = context['directory'], context['prefix']
    records = []
    for name in PUBLISHED_NAMES:
        path = directory/name
        if name == 'events.jsonl' and not path.exists():
            continue  # Helper may have failed before creating its first event.
        require(path.parent == directory, 'Unexpected evidence parent')
        records.append(cloud.upload(small_file(path), prefix+'/'+name))
    record = dict(schema='r02_storage_publication_v1', spec_sha256=context['spec_sha256'],
                  worker_run_sha256=context['worker_run_sha256'], worker_id=context['worker_id'],
                  scientific_manifests_modified=False, genotype_files_uploaded=False,
                  helper_events_present=any(item['name']=='events.jsonl' for item in records),
                  files=records)
    receipt = directory/'publication.json'
    write_fixed(receipt, record)
    cloud.upload(receipt, prefix+'/publication.json')
    return record


def run(context, cloud=None):
    directory = context['directory']
    require(not directory.is_symlink(), 'Unsafe operational evidence directory')
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (directory/'publisher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        copy_fixed(context['helper'], directory/'helper.py')
        copy_fixed(context['spec'], directory/'spec.json')
        receipt_path = directory/'wrapper_execution.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt['spec_sha256'] == context['spec_sha256']
                    and receipt['helper_sha256'] == HELPER_SHA256
                    and type(receipt['helper_returncode']) is int, 'Previous helper execution differs')
            returncode = receipt['helper_returncode']
        else:
            started = directory/'wrapper_started.json'
            require(not started.exists(), 'Interrupted wrapper execution needs inspection; helper will not be repeated')
            write_fixed(started, dict(started_utc=utc(), pid=os.getpid(), spec_sha256=context['spec_sha256'],
                                     wrapper_sha256=sha(__file__)))
            command = [sys.executable, str(directory/'helper.py'), '--spec', str(directory/'spec.json'),
                       '--spec-sha256', context['spec_sha256'], '--apply']
            try:
                completed = subprocess.run(command, check=False)
                returncode = completed.returncode
            except OSError:
                returncode = 127
            # Persist the original code before publication. A failed transfer can
            # retry without repeating the JVM pause, copy or mount operation.
            write_fixed(receipt_path, dict(schema='r02_storage_execution_v1', completed_utc=utc(),
                helper_returncode=returncode, helper_sha256=HELPER_SHA256,
                spec_sha256=context['spec_sha256'], wrapper_sha256=sha(__file__),
                worker_run_sha256=context['worker_run_sha256'], scientific_manifests_modified=False,
                helper_events_present=(directory/'events.jsonl').is_file()))
        try:
            publish(context, cloud or GCS())
        except Exception as error:
            # No exception text or environment is uploaded: authentication errors
            # must not accidentally persist secrets in project/bucket files.
            print(json.dumps(dict(status='OPERATIONAL_EVIDENCE_PUBLICATION_FAILED',
                                  helper_returncode=returncode, error_type=type(error).__name__,
                                  local_evidence=str(directory))), file=sys.stderr)
            return returncode if returncode else 74
        print(json.dumps(dict(status='OPERATIONAL_EVIDENCE_PUBLISHED', helper_returncode=returncode,
                              publication_prefix=context['prefix'])))
        return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', required=True, type=Path)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    context = validate(args.helper, args.spec, args.spec_sha256)
    if not args.run:
        print(json.dumps(dict(status='VERIFIED_NOT_EXECUTED', publication_prefix=context['prefix'])))
        return 0
    result = run(context)
    return result if result >= 0 else 128-result


if __name__ == '__main__':
    raise SystemExit(main())
