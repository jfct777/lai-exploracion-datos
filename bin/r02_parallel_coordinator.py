#!/usr/bin/env python3
"""Continue the authenticated chr21 boundary and import disjoint cloud workers.

This is an additive coordinator, not a replacement for any frozen R02 source.
It never launches remote workers, signals the old watcher, changes scientific
parameters, or invents compute checkpoints. The genuine local chr21 workflow
must finish first. Only authenticated files needed for the all-autosome analysis
are downloaded; large worker VCFs remain in their private publication prefixes.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import tempfile
import time
import types
from urllib.parse import quote


ESSENTIAL = frozenset({
    'rare_evidence.npz', 'rare_evidence.manifest.json',
    'common_primary_evidence.npz', 'common_primary_evidence.manifest.json',
    'common_sensitivity_evidence.npz', 'common_sensitivity_evidence.manifest.json',
    'sensitivity/pair_configuration_summary.tsv.gz',
    'sensitivity/configuration_summary.tsv', 'sensitivity/summary.json',
})
SCHEMA = 'r02_parallel_coordinator_v1'
OPERATIONAL_SCHEMA = 'r02_preprocess_operational_amendment_v1'
COUNT_OPERATIONAL_SCHEMA = 'r02_preprocess_operational_amendment_v2'
OPERATIONAL_SOURCE_FILES = frozenset({
    'bin/r02_autosome_pipeline.py', 'bin/mark_original_alleles.py',
    'bin/preprocess_storage_guard.py', 'bin/preprocess_audit.py',
    'modules/01_preprocess_checkpointed.nf', 'workflows/r02_preprocess_autosome.nf',
})


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    def unique(pairs):
        values = {}
        for key, value in pairs:
            require(key not in values, 'Duplicate JSON key')
            values[key] = value
        return values
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Deadline needs an explicit timezone')
    return parsed.timestamp()


def write_fixed(path, value):
    path = Path(path)
    if path.exists():
        require(not path.is_symlink() and load(path) == value, 'Existing record changed: ' + str(path))
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def load_module(path, expected, name):
    require(sha(path) == expected, 'Frozen module differs: ' + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def validate_operational_amendments(spec):
    """An approved per-worker delta, never a replacement for the base snapshot."""
    amendments = spec.get('operational_amendments', {})
    require(isinstance(amendments, dict), 'Invalid operational amendments')
    wrappers = spec.get('operational_wrapper_sha256', {})
    require(isinstance(wrappers, dict) and set(wrappers) == set(amendments)
            and all(digest(value) for value in wrappers.values()),
            'Approved operational wrapper hashes must cover exactly the amended workers')
    for worker, amendment in amendments.items():
        assigned = sorted(item['chromosome'] for item in spec['remote_chromosomes']
                          if item['worker_id'] == worker)
        require(assigned, 'Operational amendment names an unassigned worker')
        repaired = isinstance(amendment, dict) and amendment.get('schema') == COUNT_OPERATIONAL_SCHEMA
        extra = {'count_validator_sha256', 'previous_manifest_sha256'} if repaired else set()
        require(isinstance(amendment, dict) and set(amendment) == {
            'schema', 'manifest_sha256', 'source_manifest_sha256',
            'new_preprocess_chromosomes', 'legacy_chromosomes'} | extra,
            'Operational amendment fields differ')
        require(amendment['schema'] in (OPERATIONAL_SCHEMA, COUNT_OPERATIONAL_SCHEMA)
                and digest(amendment['manifest_sha256'])
                and digest(amendment['source_manifest_sha256']), 'Invalid operational amendment identity')
        require(all(digest(amendment[key]) for key in extra), 'Invalid count repair identity')
        new, old = amendment['new_preprocess_chromosomes'], amendment['legacy_chromosomes']
        require(isinstance(new, list) and isinstance(old, list)
                and all(type(c) is int for c in new + old)
                and new == sorted(set(new)) and old == sorted(set(old))
                and not set(new).intersection(old) and sorted(new + old) == assigned,
                'Operational chromosome lists must partition the original assignment')
    return amendments


def operational_records(payload, entry, spec):
    """Validate the explicit small provenance list, without accepting arbitrary prefixes."""
    amendment = validate_operational_amendments(spec).get(entry['worker_id'])
    if amendment is None:
        require('operational_amendment' not in payload and 'operational_provenance' not in payload,
                'Worker supplied an undeclared operational amendment')
        return {}
    require(payload.get('operational_amendment') == amendment,
            'Worker operational amendment differs from the approved per-chromosome delta')
    listed = payload.get('operational_provenance')
    require(isinstance(listed, list) and 4 <= len(listed) <= 128,
            'Operational provenance files missing or unexpectedly numerous')
    prefix = (entry['publication_prefix'] + '/00_worker_provenance/operational_amendments/'
              + amendment['manifest_sha256'] + '/')
    records = {}
    for record in listed:
        require(isinstance(record, dict) and set(record) == {
            'uri', 'generation', 'bytes', 'sha256', 'md5_base64'}, 'Unexpected operational provenance fields')
        uri = record['uri']
        require(isinstance(uri, str) and uri.startswith(prefix), 'Operational provenance escaped its approved prefix')
        relative = PurePosixPath(uri[len(prefix):])
        name = str(relative)
        require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
                and name == uri[len(prefix):] and name not in records,
                'Unsafe or duplicate operational provenance path')
        extra_files = ({'preprocess_count_validation.py', 'previous_manifest.json'}
                       if amendment['schema'] == COUNT_OPERATIONAL_SCHEMA else set())
        require(name in {'manifest.json', 'source.sha256.json', 'optimized_worker.py', 'activation.json'} | extra_files
                or (name.startswith('source/') and name[len('source/'):] in OPERATIONAL_SOURCE_FILES),
                'Non-code file in operational provenance')
        require(type(record['bytes']) is int and 0 < record['bytes'] <= 16 * 1024**2
                and digest(record['sha256']) and re.fullmatch(r'[0-9]+', str(record['generation']))
                and isinstance(record['md5_base64'], str), 'Invalid operational provenance digest or size')
        records[name] = record
    expected = {'manifest.json': amendment['manifest_sha256'],
                'source.sha256.json': amendment['source_manifest_sha256'],
                'optimized_worker.py': spec['operational_wrapper_sha256'][entry['worker_id']]}
    if amendment['schema'] == COUNT_OPERATIONAL_SCHEMA:
        expected.update({'preprocess_count_validation.py': amendment['count_validator_sha256'],
                         'previous_manifest.json': amendment['previous_manifest_sha256']})
    require(all(name in records and records[name]['sha256'] == checksum
                for name, checksum in expected.items()) and 'activation.json' in records,
            'Operational provenance lacks its approved source identities')
    return records


def verify_operational_provenance(payload, entry, spec, cloud):
    records = operational_records(payload, entry, spec)
    raw = {}
    for name, record in records.items():
        validate_metadata(cloud.metadata(record['uri']), record)
        data = cloud.read(record['uri'], record['generation'])
        require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
                and base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode()
                    == record['md5_base64'], 'Operational provenance object content differs')
        raw[name] = data
    if records:
        source = json.loads(raw['source.sha256.json'])
        require(isinstance(source, dict) and source, 'Operational source manifest is empty')
        amendment = payload['operational_amendment']
        activation = json.loads(raw['activation.json'])
        changed = sorted(name[len('source/'):] for name in records if name.startswith('source/'))
        extra = ({key: amendment[key] for key in ('count_validator_sha256', 'previous_manifest_sha256')}
                 if amendment['schema'] == COUNT_OPERATIONAL_SCHEMA else {})
        require(changed and isinstance(activation, dict) and activation == {
            'schema': amendment['schema'], 'manifest_sha256': amendment['manifest_sha256'],
            'original_source_manifest_sha256': spec['source_manifest_sha256'],
            'new_source_manifest_sha256': amendment['source_manifest_sha256'],
            'changed_source_files': changed, 'operational_only': True,
            'biological_validation_complete': False, **extra}, 'Operational activation differs from approved source delta')
        for name, record in records.items():
            if name.startswith('source/'):
                require(source.get(name[len('source/'):]) == record['sha256'],
                        'Published operational source differs from its source manifest')
    return records


def validate_manifest(path, expected_sha256):
    path = Path(path).resolve()
    require(digest(expected_sha256) and sha(path) == expected_sha256, 'Coordinator manifest hash mismatch')
    spec = load(path)
    require(spec.get('schema') == SCHEMA, 'Invalid coordinator schema')
    require(spec.get('coordinator_sha256') == sha(__file__), 'Coordinator source changed')
    run = Path(spec['run_dir'])
    require(run.is_absolute() and run.resolve() == run and run.is_dir(), 'Invalid parent run path')
    allowed = {run/'repairs/parallel-v1', run/'repairs/preprocess-v1/coordinator',
               run/'repairs/preprocess-v2/coordinator'}
    require(path.parent in allowed, 'Unexpected coordinator directory')
    repair = spec.get('record_count_repair')
    if repair is not None:
        require(path.parent == run/'repairs/preprocess-v2/coordinator'
                and isinstance(repair, dict) and set(repair) == {'validator_sha256', 'adoption_sha256'},
                'Count repair requires its versioned coordinator directory')
        for name, key in [('preprocess_count_validation.py', 'validator_sha256'),
                          ('adoption.json', 'adoption_sha256')]:
            target = path.parent/name
            require(digest(repair[key]) and target.is_file() and not target.is_symlink()
                    and sha(target) == repair[key], 'Changed coordinator count-repair evidence')
    request = Path(spec['request'])
    require(request == run/'repairs/boundary-v2/request.json', 'Unexpected boundary request')
    require(digest(spec['request_sha256']) and sha(request) == spec['request_sha256'], 'Boundary request changed')
    config = load(run/'run.json')
    sources = run/'repairs/boundary-v2/source.sha256.json'
    require(sha(sources) == spec['source_manifest_sha256'], 'Amended source manifest differs')
    require(sha(run/'samples.txt') == spec['samples_sha256'] == config['samples_sha256'], 'Samples changed')
    require(0 < timestamp(spec['deadline_utc']) - time.time() <= 80 * 3600,
            'Coordinator deadline must be future and at most80 hours away')
    require(0 < spec.get('poll_seconds', 30) <= 60, 'Polling interval must be at most60 seconds')
    require(spec.get('max_import_bytes', 0) > 0, 'Explicit local import byte limit is required')
    old = spec['old_watcher']
    require(set(old) == {'pid', 'start_ticks', 'cmdline_sha256'}
            and type(old['pid']) is int and old['pid'] > 1
            and type(old['start_ticks']) is int and old['start_ticks'] > 0
            and digest(old['cmdline_sha256']), 'Invalid old watcher identity')
    remote = spec['remote_chromosomes']
    require(len(remote) == 20 and {x['chromosome'] for x in remote} == set(range(1, 21)),
            'Remote chromosomes must cover exactly1–20, without duplicates')
    parent_uri = 'gs://projects-usp/dnaBr-lai/datalake/' + str(
        Path(config['destination']).relative_to('/home/jose.tantalean/gcs-dnabr'))
    workers = {}
    for entry in remote:
        require(type(entry['chromosome']) is int, 'Chromosome must be an integer')
        worker = entry['worker_id']
        require(re.fullmatch(r'[a-z][a-z0-9-]{2,62}', worker), 'Unsafe worker ID')
        prefix = parent_uri + '/parallel/' + worker
        require(entry['publication_prefix'] == prefix, 'Worker publication escaped parent run')
        require(entry['completion_uri'] == prefix + '/00_worker_provenance/worker_completion.json',
                'Unexpected worker completion path')
        require(digest(entry['worker_run_sha256']), 'Worker run hash is required')
        identity = (entry['completion_uri'], entry['worker_run_sha256'])
        require(worker not in workers or workers[worker] == identity, 'Conflicting worker declarations')
        workers[worker] = identity
    require(len(workers) == 12, 'This authorization is for exactly12 workers')
    amendments = validate_operational_amendments(spec)
    require(not amendments or path.parent in {run/'repairs/preprocess-v1/coordinator',
                                             run/'repairs/preprocess-v2/coordinator'},
            'Operational changes need their own coordinator directory')
    return spec


def missing_object_error(stderr, uri):
    """Recognize a describe 404, never a status mentioned inside another error.

    gcloud formats successful output as JSON but still emits errors as text.
    Keep unknown or conflicting diagnostics fail-closed, and bind its current
    URI-first error format to the object that was actually requested.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith('ERROR:')]
    candidates = errors or lines
    if len(candidates) != 1:
        return False
    message = re.sub(r'^ERROR:\s*(?:\(gcloud\.storage\.objects\.describe\)\s*)?',
                     '', candidates[0])
    bucket, separator, object_name = uri.removeprefix('gs://').partition('/')
    encoded_uri = 'gs://' + bucket + separator + quote(object_name, safe='')
    return bool(any(re.fullmatch(re.escape(name) + r'\s+not found:\s*404\.?', message)
                    for name in (uri, encoded_uri))
                or re.match(r'(?:HTTPError\s+404|NOT_FOUND:\s*404|'
                            r'ResponseError:\s*status\s*=\s*404)\b', message))


class GCS:
    """Read-only cloud operations; all transfers are tied to object generations."""

    def metadata(self, uri, *, optional=False):
        result = subprocess.run(['gcloud', 'storage', 'objects', 'describe', uri, '--format=json'],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            if optional and missing_object_error(result.stderr, uri):
                return None
            raise RuntimeError('Cannot authenticate cloud object: ' + uri + ': ' + result.stderr[-1000:])
        return json.loads(result.stdout)

    def read(self, uri, generation):
        return subprocess.check_output(['gcloud', 'storage', 'cat', uri + '#' + str(generation)], timeout=120)

    def download(self, uri, generation, target):
        subprocess.run(['gcloud', 'storage', 'cp', uri + '#' + str(generation), str(target)],
                       check=True, capture_output=True, text=True, timeout=7200)


def md5_file(path):
    checksum = hashlib.md5(usedforsecurity=False)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return base64.b64encode(checksum.digest()).decode('ascii')


def validate_metadata(metadata, record):
    require(str(metadata.get('generation')) == str(record['generation']), 'GCS generation changed')
    require(int(metadata.get('size', -1)) == record['bytes'], 'GCS size differs')
    require(metadata.get('md5_hash', metadata.get('md5Hash')) == record['md5_base64'], 'GCS checksum differs')


def validate_worker_completion(payload, entry, spec):
    require(payload.get('schema_version') == 1 and payload.get('worker_id') == entry['worker_id'],
            'Worker completion identity differs')
    require(payload.get('parent_run_id') == Path(spec['run_dir']).name
            and payload.get('analysis_protocol_version') == 2, 'Wrong parent or analysis protocol')
    require(payload.get('source_manifest_sha256') == spec['source_manifest_sha256'],
            'Worker used a different scientific source snapshot')
    require(payload.get('samples_sha256') == spec['samples_sha256'], 'Worker samples differ')
    require(payload.get('worker_run_sha256') == entry['worker_run_sha256'], 'Worker run differs')
    operational_records(payload, entry, spec)
    require(digest(payload.get('worker_frozen_sha256')), 'Worker frozen manifest hash missing')
    require(payload.get('aggregate_executed') is False and payload.get('biological_validation_complete') is False,
            'Worker exceeded its declared role')
    assigned = sorted(item['chromosome'] for item in spec['remote_chromosomes']
                      if item['worker_id'] == entry['worker_id'])
    require(sorted(payload.get('chromosomes', [])) == assigned, 'Worker assignment changed')
    artifacts = payload.get('artifacts', [])
    require(len(artifacts) == len(assigned)
            and sorted(item['chromosome'] for item in artifacts) == assigned, 'Incomplete/duplicate worker artifacts')
    artifact = next(item for item in artifacts if item['chromosome'] == entry['chromosome'])
    require(digest(artifact.get('completion_sha256')) and digest(artifact.get('publication_sha256')),
            'Missing worker completion/publication hashes')
    outputs = artifact['outputs']
    require(isinstance(outputs, list) and outputs, 'Worker has no published outputs')
    records = {}
    prefix = entry['publication_prefix'] + f'/01_estructura/desarrollo/por_cromosoma/chr{entry["chromosome"]:02d}/'
    for record in outputs:
        relative = PurePosixPath(record['relative_path'])
        require(not relative.is_absolute() and relative.parts and '..' not in relative.parts
                and str(relative) == record['relative_path'], 'Unsafe worker relative path')
        require(str(relative) not in records, 'Duplicate output path')
        require(record['uri'] == prefix + str(relative), 'Worker output escaped chromosome prefix')
        require(type(record['bytes']) is int and record['bytes'] >= 0 and digest(record['sha256']), 'Invalid output digest')
        require(re.fullmatch(r'[0-9]+', str(record['generation']))
                and isinstance(record['md5_base64'], str), 'Missing generation/checksum')
        records[str(relative)] = record
    require(ESSENTIAL <= records.keys(), 'Worker lacks required aggregation files')
    return records


def import_chromosome(spec, entry, receipt_dir, cloud, *, deadline=None):
    """Return False only for an absent completion; errors never mean success."""
    deadline = deadline or timestamp(spec['deadline_utc'])
    require(time.time() < deadline, 'Import deadline expired')
    uri = entry['completion_uri']
    metadata = cloud.metadata(uri, optional=True)
    if metadata is None:
        return False
    raw = cloud.read(uri, metadata['generation'])
    completion_record = dict(uri=uri, generation=str(metadata['generation']), bytes=len(raw),
        md5_base64=base64.b64encode(hashlib.md5(raw, usedforsecurity=False).digest()).decode(),
        sha256=hashlib.sha256(raw).hexdigest())
    validate_metadata(metadata, completion_record)
    payload = json.loads(raw)
    records = validate_worker_completion(payload, entry, spec)
    verify_operational_provenance(payload, entry, spec, cloud)
    chromosome = entry['chromosome']
    root = Path(spec['run_dir'])/f'chr{chromosome:02d}'
    require(not root.is_symlink() and (not root.exists() or root.is_dir()), 'Invalid chromosome destination')
    # Never collide with a local scientific computation or pretend it ran here.
    cp = Path(spec['run_dir'])/'checkpoints'
    require(not list(cp.glob(f'chr{chromosome:02d}_*.json')), 'Local chromosome execution already exists')
    prior_bytes = sum(item['bytes'] for receipt in Path(receipt_dir).glob('chr*_import.json')
                      if receipt.name != f'chr{chromosome:02d}_import.json'
                      for item in load(receipt)['outputs'])
    require(prior_bytes + sum(records[name]['bytes'] for name in ESSENTIAL) <= spec['max_import_bytes'],
            'Combined imports exceed byte limit')
    root.mkdir(exist_ok=True)
    installed = []
    for name, record in sorted(records.items()):
        require(time.time() < deadline, 'Import deadline expired')
        validate_metadata(cloud.metadata(record['uri']), record)
        if name not in ESSENTIAL:
            continue
        target = root/name
        require(not any(p.is_symlink() for p in (target, *target.parents)), 'Symlink in import destination')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(target.is_file() and target.stat().st_size == record['bytes']
                    and sha(target) == record['sha256'] and md5_file(target) == record['md5_base64'],
                    'Existing import differs: ' + name)
        else:
            free = shutil.disk_usage(root).free
            require(free - record['bytes'] >= 12 * 1024**3, 'Insufficient local disk for authenticated import')
            fd, temporary = tempfile.mkstemp(prefix='.import-', dir=target.parent)
            os.close(fd)
            try:
                cloud.download(record['uri'], record['generation'], temporary)
                temp = Path(temporary)
                require(temp.stat().st_size == record['bytes'] and sha(temp) == record['sha256']
                        and md5_file(temp) == record['md5_base64'], 'Downloaded output checksum differs')
                os.link(temporary, target)  # Exclusive installation; never overwrite an existing result.
            finally:
                Path(temporary).unlink(missing_ok=True)
        installed.append(dict(record, source_path=record.get('path'), path=str(target)))
    proof = dict(schema='r02_remote_import_v1', chromosome=chromosome,
        worker_id=entry['worker_id'], completion=completion_record,
        worker_run_sha256=entry['worker_run_sha256'],
        publication_sha256=next(a['publication_sha256'] for a in payload['artifacts'] if a['chromosome'] == chromosome),
        outputs=installed, all_published_objects_verified=len(records),
        local_compute_checkpoint_created=False, biological_validation_complete=False)
    if 'operational_amendment' in payload:
        proof['operational_amendment'] = payload['operational_amendment']
        proof['operational_provenance'] = payload['operational_provenance']
        proof['preprocess_source_manifest_sha256'] = (
            payload['operational_amendment']['source_manifest_sha256']
            if chromosome in payload['operational_amendment']['new_preprocess_chromosomes']
            else spec['source_manifest_sha256'])
    write_fixed(Path(receipt_dir)/f'chr{chromosome:02d}_import.json', proof)
    return True


def wait_and_import(spec, receipt_dir, state, cloud=None):
    cloud = cloud or GCS()
    pending = {item['chromosome']: item for item in spec['remote_chromosomes']}
    deadline = timestamp(spec['deadline_utc'])
    while pending:
        require(time.time() < deadline, 'Remote workers did not complete before coordinator deadline')
        for chrom in list(pending):
            if import_chromosome(spec, pending[chrom], receipt_dir, cloud, deadline=deadline):
                del pending[chrom]
        state('WAITING_REMOTE' if pending else 'ALL_REMOTE_IMPORTED', pending_chromosomes=sorted(pending))
        if pending:
            time.sleep(min(spec.get('poll_seconds', 30), max(0, deadline-time.time())))


def run_after_boundary(spec, directory, boundary, *, activate=True, provenance_destination=None):
    source = Path(spec['run_dir'])/'repairs/boundary-v2'
    hashes = load(source/'source.sha256.json')
    adapter = load_module(source/'source/bin/r02_apply_amendment.py',
                          hashes['bin/r02_apply_amendment.py'], '_r02_parallel_adapter')
    _, pipeline, _ = adapter.validate_snapshot(Path(spec['run_dir']), source)
    with pipeline.execution_lock(Path(spec['run_dir'])):
        runner, pipeline, evidence = adapter.build_runner(Path(spec['run_dir']), source, activate=activate)
        repair = spec.get('record_count_repair')
        if repair is not None:
            helper = load_module(directory/'preprocess_count_validation.py', repair['validator_sha256'],
                                 '_r02_repaired_record_count')
            pipeline.validate_raw_record_count = helper.validate_raw_record_count
        activation = dict(schema=SCHEMA, source_manifest_sha256=spec['source_manifest_sha256'],
                          coordinator_sha256=spec['coordinator_sha256'], remote_chromosomes=list(range(1, 21)),
                          local_chromosomes=[22, 21], amendment_sha256=evidence['amendment_sha256'])
        if repair is not None:
            activation['record_count_repair'] = repair
        write_fixed(directory/'coordinator_activation.json', activation)
        try:
            runner.analyze_chromosome(22)
            runner.analyze_chromosome(21)
            # Import receipts retain one canonical location across operational
            # controller versions. A new directory cannot authorize recompute.
            imports = Path(spec['run_dir'])/'repairs/parallel-v1/imports'
            imports.mkdir(exist_ok=True)
            wait_and_import(spec, imports, boundary.state)
            require(time.time() < timestamp(spec['deadline_utc']), 'No time remains for aggregation')
            runner.aggregate()
            provenance = directory/'provenance'
            provenance.mkdir(exist_ok=True)
            names = ['manifest.json', 'coordinator_activation.json']
            if repair is not None:
                names += ['preprocess_count_validation.py', 'adoption.json']
            for name in names:
                source_file = directory/name
                if source_file.is_file():
                    target = provenance/name
                    if target.exists():
                        require(sha(target) == sha(source_file), 'Provenance source changed')
                    else:
                        with target.open('xb') as handle:
                            handle.write(source_file.read_bytes())
            for source_file in sorted(imports.glob('*.json')):
                write_fixed(provenance/source_file.name, load(source_file))
            destination = provenance_destination or ('00_datos_y_diseno/preprocess-v2/parallel' if repair is not None else
                           '00_datos_y_diseno/preprocess-v1/parallel'
                           if spec.get('operational_amendments') else '00_datos_y_diseno/parallel-v1')
            runner.publish(provenance, destination)
            pipeline.status(runner.run, 'COMPLETE_PUBLISHED', state='COMPLETE',
                destination=pipeline.uri(runner.dest), biological_validation_complete=False,
                scope='DESCRIPTIVE_NOT_CONFIRMATORY', remote_chromosomes=list(range(1, 21)))
            boundary.state('PARALLEL_AGGREGATION_COMPLETE', biological_validation_complete=False)
        except BaseException as error:
            pipeline.status(runner.run, 'FAILED', state='FAILED', failed_stage=runner.current_stage, error=str(error))
            raise


def successor(spec, directory, module):
    class ParallelHandoff(module.Handoff):
        def state(self, state, **extra):
            module.write_json(directory/'coordinator_status.json',
                dict(state=state, updated_utc=module.now(), request_sha256=self.expected_sha256, **extra))
            print(json.dumps(dict(state=state, **extra)), flush=True)

        def activate(self):
            self.validate()
            checkpoint_hash = self.validate_completion()
            amendment = copy.deepcopy(self.request['amendment_template'])
            amendment['boundary']['checkpoint_sha256'] = checkpoint_hash
            module.write_json(self.amendment/'amendment.json', amendment, once=True)
            module.write_json(self.amendment/'frozen.sha256.json',
                {name: module.sha(self.amendment/name)
                 for name in ('amendment.json', 'source.sha256.json', 'request.json')}, once=True)
            self.state('LOCAL_BOUNDARY_VERIFIED', checkpoint_sha256=checkpoint_hash)
            run_after_boundary(spec, directory, self)

        def monitor(self, poll_seconds=10):
            # Same authenticated barrier checks as the existing watcher, with
            # one additional finite deadline. No second arm or new fork barrier.
            while True:
                if time.time() >= timestamp(spec['deadline_utc']):
                    self.emergency('Parallel coordinator deadline reached')
                free = shutil.disk_usage(self.run).free / 1024**3
                parent = module.authenticated(self.request['supervisor'])
                child = module.authenticated(self.request['child'])
                if free < self.request['free_disk_min_gib']:
                    self.emergency('Free disk below original12 GiB guard')
                if parent is None:
                    if child is not None:
                        self.emergency('Supervisor exited before its authenticated Nextflow child')
                    return self.activate()
                if module.nproc(parent['pid']) != (0, self.request['original_limits'][1]):
                    self.emergency('Process-creation barrier was removed')
                if child is not None and list(module.nproc(child['pid'])) != self.armed['child_limits']:
                    self.emergency('Existing Nextflow limits changed')
                if parent['state'] == 'T':
                    module.send(self.request['supervisor'], module.signal.SIGCONT)
                self.state('WAITING_CURRENT_WORKFLOW', free_disk_gib=round(free, 2), child_active=child is not None)
                time.sleep(poll_seconds)

    return ParallelHandoff(spec['request'], spec['request_sha256']).validate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    spec = validate_manifest(args.manifest, args.expected_manifest_sha256)
    directory = args.manifest.resolve().parent
    source = Path(spec['run_dir'])/'repairs/boundary-v2'
    hashes = load(source/'source.sha256.json')
    module = load_module(source/'source/bin/r02_stage_boundary_handoff.py',
                         hashes['bin/r02_stage_boundary_handoff.py'], '_r02_parallel_boundary')
    handoff = successor(spec, directory, module)
    require((source/'armed.json').is_file(), 'Original boundary must already be armed')
    if not args.run:
        print(json.dumps(dict(state='VERIFIED_NOT_EXECUTED', chromosomes=list(range(1, 23)))))
        return 0
    def expired(_signal, _frame):
        raise TimeoutError('Coordinator total deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, timestamp(spec['deadline_utc']) - time.time()))
    old = module.authenticated(spec['old_watcher'])
    if old is not None:
        children = Path(f'/proc/{old["pid"]}/task/{old["pid"]}/children').read_text().split()
        require(not children, 'Old watcher has children; unsafe to hand off')
    module.write_json(directory/'ready.json', dict(state='READY_WAITING_BOUNDARY_LOCK',
        pid=os.getpid(), manifest_sha256=args.expected_manifest_sha256, ready_utc=module.now()))
    with (source/'.boundary.lock').open('a+') as lock:
        while True:
            require(time.time() < timestamp(spec['deadline_utc']), 'Deadline waiting for original watcher lock')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        require(module.authenticated(spec['old_watcher']) is None, 'Old watcher still alive after lock acquisition')
        try:
            if (directory/'coordinator_activation.json').exists():
                run_after_boundary(spec, directory, handoff)
            else:
                require(not (source/'activation.json').exists(), 'A different controller already activated boundary')
                handoff.arm()
                handoff.monitor()
        except BaseException as error:
            if not (directory/'coordinator_activation.json').exists():
                handoff.fail_closed(error)
            else:
                handoff.state('FAILED', error=str(error))
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
