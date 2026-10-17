#!/usr/bin/env python3
"""Delegate pending M02 chromosomes while adopting the unchanged active Nextflow.

Run in a separate systemd service (KillMode=process), never the startup cgroup.
Only the authenticated Python supervisor is paused. Once queue_transfer.json is
sealed it must NEVER be resumed: its pending work belongs to the new delegate.
This guard runs no filter, starts no VM and edits no frozen source. It verifies
the active child's kernel exit, task traces and task exit codes, then invokes
the existing publication finalizer. Failure preserves all unpublished files.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager, ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import r02_m02_recovery_worker as worker
import r02_segment_handoff as identities

process = identities.process_identity
publication = worker.publication
require = publication.require
SCHEMA = 'r02_m02_active_handoff_v1'
TRANSFER_SCHEMA = 'r02_m02_pending_handoff_v1'


def handoff_identity(config):
    """Derive a bounded publication destination; never accept an arbitrary URI."""
    require(config.get('schema') == SCHEMA, 'Unknown handoff schema')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,100}', config['handoff_id']), 'Invalid handoff ID')
    require(isinstance(config.get('source_instance_id'), str) and
            re.fullmatch(r'[1-9][0-9]+', config['source_instance_id']), 'Original VM numeric ID required')
    match = re.fullmatch(r'dnabr-r02-m02-([e-h])-1004', config['delegate_instance'])
    require(match is not None, 'Unexpected delegate instance')
    owner = dict(e='a', f='b', g='c', h='d')[match[1]]
    return ('dnabr-r02-m02-' + owner + '-1004',
            worker.RECOVERY_PREFIX + 'worker-' + owner + '/handoff/' + config['handoff_id'])


def publish_record(path, uri, name):
    record = dict(name=name, path=str(publication.safe_local(path)), uri=uri,
                  **worker.pipeline.publication_digests(path))
    publication.upload_one(record)
    publication.verify_remote(record, pinned=True)
    return record


def publish_transfer(path, config, output):
    _, prefix = handoff_identity(config)
    records = []
    for name, local in [('handoff.json', path), ('queue_transfer.json', output / 'queue_transfer.json')]:
        records.append(publish_record(local, prefix + '/' + name, name))
    return records


def fetch_json(uri, record=None):
    """Read small pinned JSON evidence, not genomic payloads or local VM paths."""
    require(uri.startswith(worker.RECOVERY_PREFIX) and uri.endswith('.json'), 'Unexpected evidence URI')
    metadata = publication.remote_metadata(uri)
    size, generation = int(metadata['size']), str(metadata['generation'])
    require(0 < size <= 4 * 1024 * 1024 and re.fullmatch(r'[1-9][0-9]*', generation), 'Invalid evidence size/generation')
    if record is not None:
        require(record['uri'] == uri, 'Evidence URI differs')
        publication.verify_remote(record, pinned=True)
    response = subprocess.run(['gcloud', 'storage', 'cat', uri + '#' + generation],
                              capture_output=True, check=True, timeout=120)
    data = response.stdout
    actual = dict(uri=uri, bytes=len(data), generation=generation, sha256=hashlib.sha256(data).hexdigest(),
                  md5_base64=base64.b64encode(hashlib.md5(data).digest()).decode())
    require(len(data) == size, 'Evidence length changed')
    if record is not None:
        require(all(actual[k] == record[k] for k in actual), 'Evidence bytes or generation changed')
    publication.verify_remote(actual, pinned=True)
    return json.loads(data), actual


def verify_complete(path, digest):
    """Read-only retirement proof; works after the original VM has stopped.

    Only the adopted chromosome must be complete. The pending queue must have
    been irrevocably delegated. This is NOT an all-chromosome completion claim.
    An external controller must compare instance_id before deleting anything.
    """
    path = publication.safe_local(path)
    require(worker.sha(path) == publication.validate_sha(digest), 'Handoff SHA256 mismatch')
    config = worker.read(path)
    instance, prefix = handoff_identity(config)
    result, completion_record = fetch_json(prefix + '/completion.json')
    require(result['status'] == 'ACTIVE_M02_PUBLISHED_PENDING_DELEGATED' and
            result['handoff_contract_sha256'] == digest and result['instance'] == instance and
            result['instance_id'] == config['source_instance_id'] and
            result['active_chromosome'] == config['active_chromosome'] and
            result['delegated_chromosomes'] == config['delegated_chromosomes'] and
            result['no_filter_replayed'] is True and result['all_original_chromosomes_complete'] is False,
            'Completion scope/identity changed')
    records = result['handoff_files']
    expected_names = {'handoff.json', 'queue_transfer.json', 'accepted-exit.json', 'finalized.json'}
    require(len(records) == 4 and {r['name'] for r in records} == expected_names, 'Incomplete handoff publication')
    contents = {}
    for record in records:
        require(record['uri'] == prefix + '/' + record['name'], 'Handoff evidence escaped its prefix')
        contents[record['name']], _ = fetch_json(record['uri'], record)
    contract = next(r for r in records if r['name'] == 'handoff.json')
    require(contract['sha256'] == digest and contents['handoff.json'] == config, 'Published contract differs')
    transfer = contents['queue_transfer.json']
    require(transfer['schema'] == TRANSFER_SCHEMA and transfer['status'] == 'PENDING_M02_DELEGATED_SUPERVISOR_PAUSED' and
            transfer['original_instance'] == instance and transfer['source_instance_id'] == config['source_instance_id'] and
            transfer['delegate_instance'] == config['delegate_instance'] and
            transfer['chromosomes'] == config['delegated_chromosomes'] and
            transfer['adopted_chromosome'] == config['active_chromosome'] and
            transfer['original_launch_sha256'] == config['original_launch_sha256'] and
            transfer['handoff_contract_sha256'] == digest and transfer['supervisor'] == config['supervisor'] and
            transfer['adopted_nextflow'] == config['nextflow'] and
            transfer['original_deadline_utc'] == config['original_deadline_utc'], 'Published queue transfer differs')
    proof, finalized = contents['accepted-exit.json'], contents['finalized.json']
    require(proof['handoff_sha256'] == digest and proof['kernel_exit'] == 0 and len(proof['task_exits']) == 2 and
            finalized['handoff_sha256'] == digest and finalized['no_filter_replayed'] is True and
            finalized['published'] == result['published'], 'Exit/finalization proof differs')
    chrom_prefix = prefix.split('/handoff/')[0] + '/chr' + config['active_chromosome'].zfill(2) + '/'
    require(result['published']['uri'] == chrom_prefix + 'completed.json', 'Wrong chromosome completion URI')
    active, _ = fetch_json(result['published']['uri'], result['published'])
    require(active['status'] == 'M02_RECOVERED_PUBLISHED_VERIFIED_NOT_FIT_SELECTED' and
            active['chromosome'] == config['active_chromosome'] and
            active['launch_sha256'] == config['original_launch_sha256'], 'Active M02 completion differs')
    files = active['files']
    basename = 'dnabr.hg38.2723.chr' + config['active_chromosome']
    attempt = config['attempt']
    expected_files = {basename + '.snv.bi.pass.vcf.gz', basename + '.snv.bi.pass.vcf.gz.tbi', basename + '.counts.tsv',
                      'source-receipt.json', 'params.json', 'trace.' + attempt + '.tsv', 'nextflow.' + attempt + '.log',
                      'console.' + attempt + '.log', 'execution-amendment.' + attempt + '.json',
                      'raw-count-evidence.' + attempt + '.json', 'params.' + attempt + '.json'}
    require(len(files) == 11 and {r['name'] for r in files} == expected_files and
            len({r['uri'] for r in files}) == len(files), 'Missing/duplicate active output files')
    for record in files:
        require(record['uri'] == chrom_prefix + str(publication.safe_relative(record['name'])), 'Active output escaped its prefix')
        publication.validate_sha(record['sha256'])
        publication.verify_remote(record, pinned=True)
    return dict(status='SAFE_TO_RETIRE_OWN_VM', instance=instance, instance_id=config['source_instance_id'],
                project=worker.PROJECT, zone='us-central1-a', adopted_chromosome=config['active_chromosome'],
                pending_delegated=config['delegated_chromosomes'], delegate_instance=config['delegate_instance'],
                all_original_chromosomes_complete=False, completion=completion_record,
                verified_active_files=len(files), verified_utc=utc())


def publish_completion(path, digest, config, output, result):
    """All irreversible retirement evidence is durable before old startup exits."""
    _, prefix = handoff_identity(config)
    records = publish_transfer(path, config, output)
    for name in ('accepted-exit.json', 'finalized.json'):
        records.append(publish_record(output / name, prefix + '/' + name, name))
    # Upload operation may be CREATED then REUSED on retry; it is not identity.
    result['handoff_files'] = [{k:v for k,v in record.items() if k != 'operation'} for record in records]
    complete = output / 'completion.json'
    if complete.exists():
        previous = worker.read(complete)
        require({k:v for k,v in previous.items() if k != 'completion_utc'} ==
                {k:v for k,v in result.items() if k != 'completion_utc'}, 'Completion record differs')
        result = previous
    save_once(complete, result)
    record = publish_record(complete, prefix + '/completion.json', 'completion.json')
    save_once(output / 'completion.publication.json', {k:v for k,v in record.items() if k != 'operation'})
    proof = verify_complete(path, digest)
    proof_path = output / 'retirement-proof.json'
    if proof_path.exists():
        previous = worker.read(proof_path)
        require({k:v for k,v in previous.items() if k != 'verified_utc'} ==
                {k:v for k,v in proof.items() if k != 'verified_utc'}, 'Retirement evidence changed')
        proof = previous
    save_once(proof_path, proof)
    return result


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_once(path, value):
    if path.exists():
        require(worker.read(path) == value, 'Immutable handoff record changed')
    else:
        publication.write_receipt_new(path, value)


def load(path, expected):
    path = publication.safe_local(path)
    require(worker.sha(path) == publication.validate_sha(expected), 'Handoff SHA256 mismatch')
    config = worker.read(path)
    require(config.get('schema') == SCHEMA, 'Unknown handoff schema')
    original_instance, _ = handoff_identity(config)
    require(config.get('mode', 'PRODUCTION') in {'PRODUCTION', 'SYNTHETIC_TEST'}, 'Invalid handoff mode')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,100}', config['handoff_id']), 'Invalid handoff ID')
    root = publication.safe_local(config['source_root'], directory=True)
    require(path.parent != root and not path.parent.is_relative_to(root), 'Handoff records must be outside the frozen source root')
    manifest, inventory = worker.verify_bundle(root, config['original_launch_sha256'])
    require(manifest['instance'] == original_instance, 'Original/delegated identities differ')
    require(manifest['schema'] == 'r02_m02_recovery_launch_v1', 'Adopt only an original v1 worker with v4 amendment')
    amendment = config['runtime_amendment']
    worker.apply_runtime_amendment(root, manifest, Path(amendment['path']), amendment['sha256'],
                                   runtime_worker_path=Path(config['runtime_worker_path']))
    require(manifest.get('runtime_source'), 'Authenticated legacy-count amendment required')
    require(re.fullmatch(r'resume[1-9][0-9]*', config['attempt']), 'Explicit adopted attempt required')
    owner = manifest['instance'].split('-')[-2]
    require(config['delegate_instance'] == 'dnabr-r02-m02-' + dict(a='e', b='f', c='g', d='h')[owner] + '-1004',
            'Delegate identity must correspond to its original owner')
    active = config['active_chromosome']
    require(active in manifest['chromosomes'], 'Active chromosome not assigned')
    index = manifest['chromosomes'].index(active)
    require(config['delegated_chromosomes'] == manifest['chromosomes'][index + 1:] and config['delegated_chromosomes'],
            'Delegate exactly all pending chromosomes; never the active or completed chromosome')
    for chrom in manifest['chromosomes'][:index]:
        receipt = worker.read(root / 'chromosomes' / ('chr' + chrom.zfill(2)) / 'completed.publication.json')
        publication.verify_remote(receipt, pinned=True)
    identities.check_identity(config['supervisor']); identities.check_identity(config['nextflow'])
    require(config['supervisor']['pid'] != config['nextflow']['pid'], 'Supervisor and Nextflow must differ')
    identities.campaign.sealed_files(config['files'])
    if config.get('mode', 'PRODUCTION') == 'PRODUCTION':
        modules = [Path(__file__), Path(worker.__file__), Path(identities.__file__), Path(process.__file__),
                   Path(identities.campaign.__file__), Path(publication.__file__), Path(worker.pipeline.__file__)]
        require({str(p.resolve()) for p in modules} <= set(config['files']), 'Handoff/finalizer dependencies must be sealed')
        own_cgroup = Path('/proc/self/cgroup').read_text()
        old_cgroup = (Path('/proc') / str(config['supervisor']['pid']) / 'cgroup').read_text()
        require(own_cgroup != old_cgroup and '.service' in own_cgroup,
                'Use an independent systemd service, not the original startup cgroup')
    deadline = datetime.fromisoformat(config['original_deadline_utc'].replace('Z', '+00:00'))
    require(deadline.tzinfo is not None, 'Timezone-aware original deadline required')
    require(config['original_deadline_utc'] == manifest['runtime_amendment']['original_deadline_utc'], 'Original deadline changed')
    resources = config['resources']
    for key in ('min_free_disk_gib', 'min_available_memory_gib', 'poll_seconds', 'child_term_grace_seconds', 'publication_reserve_seconds'):
        require(type(resources.get(key)) in (int, float) and resources[key] > 0, 'Invalid resource ' + key)
    require(resources['poll_seconds'] <= 10 and resources['child_term_grace_seconds'] <= 60 and
            300 <= resources['publication_reserve_seconds'] <= 3600, 'Unsafe guard interval/deadline margin')
    if config.get('mode', 'PRODUCTION') == 'PRODUCTION':
        require(resources['min_free_disk_gib'] >= 12 and resources['min_available_memory_gib'] >= .25, 'Insufficient resource reserve')
    entry = next(e for e in inventory['entries'] if e['chromosome'] == active)
    return config, root, manifest, entry, path.parent


def folder_for(root, chrom):
    return root / 'chromosomes' / ('chr' + chrom.zfill(2))


def check_queue(config, root, manifest, *, paused=False, allow_publication=False):
    parent = process.authenticated(config['supervisor'])
    require(parent is not None and (not paused or parent['state'] == 'T'), 'Authenticated supervisor must remain paused')
    require(identities.direct_children(parent['pid']) == {config['nextflow']['pid']}, 'Supervisor child set changed')
    child = process.same_incarnation(config['nextflow'])
    require(child is not None and child['ppid'] == parent['pid'] and child['sid'] == child['pgid'] == child['pid'],
            'Nextflow ownership/session changed')
    if child['state'] not in ('Z', 'X'):
        require(child['cmdline_sha256'] == config['nextflow']['cmdline_sha256'], 'Nextflow command identity changed')
    for chrom in config['delegated_chromosomes']:
        require(not folder_for(root, chrom).exists(), 'Pending chromosome was already started; do not delegate it')
    folder = folder_for(root, config['active_chromosome'])
    if not allow_publication:
        require(not (folder / 'publication-plan.json').exists() and not (folder / 'completed.json').exists(),
                'Active chromosome entered publication before handoff')
    if config.get('mode', 'PRODUCTION') == 'PRODUCTION' and child['state'] not in ('Z', 'X'):
        argv = (Path('/proc') / str(child['pid']) / 'cmdline').read_bytes().decode().strip('\0').split('\0')
        require(Path(argv[0]).name == 'java', 'Adopt a Nextflow Java process only')
        expected = {'-C': manifest['runtime_config'], '-params-file': str(folder / ('params.' + config['attempt'] + '.json')),
                    '-work-dir': str(folder / 'work'), '-with-trace': str(folder / ('trace.' + config['attempt'] + '.tsv')),
                    'run': str(Path(manifest['runtime_source']) / 'workflows/r02_recover_m02.nf')}
        for flag, value in expected.items():
            require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, 'Active Nextflow command differs at ' + flag)


def resource_guard(config, root, stop):
    require(not stop.is_set(), 'Handoff interrupted')
    deadline = datetime.fromisoformat(config['original_deadline_utc'].replace('Z', '+00:00')).timestamp()
    require(time.time() < deadline - config['resources']['publication_reserve_seconds'], 'Original deadline reserve reached')
    require(shutil.disk_usage(root).free >= config['resources']['min_free_disk_gib'] * worker.GIB, 'Disk reserve reached')
    require(identities.campaign.available_memory_bytes() >= config['resources']['min_available_memory_gib'] * worker.GIB,
            'Memory reserve reached')


def accepted_exit(config, root, fd):
    code = identities.kernel_exit_code(config['nextflow'], fd)
    require(code == 0, 'Active Nextflow kernel exit was not zero')
    folder = folder_for(root, config['active_chromosome'])
    trace = folder / ('trace.' + config['attempt'] + '.tsv')
    with trace.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    require(len(rows) == 2 and all(r['status'] in {'COMPLETED', 'CACHED'} and r['exit'] == '0' for r in rows),
            'Expected two successful M02 trace tasks')
    require({r['name'].split(' (')[0].split(':')[-1] for r in rows} ==
            {'RECOVER_M02_ADMISSION', 'PREPROCESS_FILTER_SNV_BIALLELIC_PASS'}, 'Unexpected task scope')
    exits = []
    for row in rows:
        task_hash = row['hash']
        require(re.fullmatch(r'[a-f0-9]{2}/[a-f0-9]{6,}', task_hash), 'Malformed task hash')
        matches = list((folder / 'work').glob(task_hash + '*'))
        require(len(matches) == 1 and matches[0].resolve() == matches[0], 'Task hash does not identify one work directory')
        path = matches[0] / '.exitcode'
        require(path.is_file() and path.read_text().strip() == '0', 'Task exit was not zero')
        exits.append(dict(task_hash=task_hash, exitcode_sha256=worker.sha(path)))
    return dict(kernel_exit=code, trace_sha256=worker.sha(trace), task_exits=exits)


@contextmanager
def transfer_lock(root):
    with (root / '.m02-handoff.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def close_children(config, manifest, handles):
    adopted = [dict(job_id=name, process=spec) for name, spec in handles['specs'].items()]
    shim = dict(mode=config.get('mode', 'PRODUCTION'), adopted_jobs=adopted, resources=config['resources'])
    jobs = {row['job_id']: dict(container_label=manifest['container_label']) for row in adopted}
    identities.close_owned_children(shim, handles['fds'], jobs)


def finalize_only(path, digest):
    config, root, manifest, entry, output = load(path, digest)
    receipt = worker.read(output / 'queue_transfer.json')
    require(receipt['handoff_contract_sha256'] == digest, 'Transfer belongs to another guard')
    check_queue(config, root, manifest, paused=True, allow_publication=True)
    finalized = output / 'finalized.json'
    if finalized.exists():
        previous = worker.read(finalized)
        require(previous['handoff_sha256'] == digest and previous['no_filter_replayed'] is True, 'Finalized record differs')
        receipt = previous['published']
        local = folder_for(root, config['active_chromosome']) / 'completed.json'
        require(worker.sha(local) == receipt['sha256'], 'Completed receipt changed')
        for record in worker.read(local)['files']:
            publication.verify_remote(record, pinned=True)
        publication.verify_remote(receipt, pinned=True)
        return
    fd, _ = identities.bind_process(config['nextflow'], allow_exited=True)
    try:
        proof = worker.read(output / 'accepted-exit.json')
        require(proof['handoff_sha256'] == digest and identities.kernel_exit_code(config['nextflow'], fd) == 0,
                'No authenticated zero-exit adoption proof')
        if not (folder_for(root, config['active_chromosome']) / 'publication-plan.json').exists():
            require(proof == dict(handoff_sha256=digest, **accepted_exit(config, root, fd)), 'Task exit proof changed')
        result = worker.finalize_chromosome(root, manifest, entry, config['attempt'],
                                           runtime_worker_path=Path(config['runtime_worker_path']))
        save_once(output / 'finalized.json', dict(handoff_sha256=digest, published=result, no_filter_replayed=True))
    finally:
        os.close(fd)


def publish_finished(path, digest, config, root, manifest, output, stop, handles):
    args = [sys.executable, str(Path(__file__).resolve()), '--handoff', str(path), '--expected-sha256', digest, '--finalize-only']
    log = output / ('finalization.' + str(time.time_ns()) + '.log')
    with log.open('x') as stream:
        child = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        info = process.process_info(child.pid)
        require(info is not None, 'Publication finalizer disappeared')
        spec = process.identity(info)
        fd, _ = identities.bind_process(spec, allow_exited=True)
        handles['specs']['publication'] = spec; handles['fds']['publication'] = fd
        try:
            while child.poll() is None:
                resource_guard(config, root, stop)
                check_queue(config, root, manifest, paused=True, allow_publication=True)
                require(worker.sha(path) == digest, 'Handoff contract changed')
                stop.wait(config['resources']['poll_seconds'])
            require(child.returncode == 0, 'Publication failed; preserve outputs and publication plan')
        finally:
            if child.poll() is None:
                close_children(config, manifest, handles)
                child.wait(timeout=10)
            os.close(fd)
            handles['fds'].pop('publication'); handles['specs'].pop('publication')
    result = worker.read(output / 'finalized.json')
    require(result['handoff_sha256'] == digest and result['no_filter_replayed'] is True, 'Finalization scope differs')
    publication.verify_remote(result['published'], pinned=True)
    return result['published']


def run(path, digest, *, inspect_only=False, stop=None):
    path = Path(path).absolute()
    config, root, manifest, entry, output = load(path, digest)
    stop = stop or threading.Event()
    with transfer_lock(root), ExitStack() as stack:
        prior = worker.read(output / 'queue_transfer.json') if (output / 'queue_transfer.json').exists() else None
        armed = worker.read(output / 'prepared.json') if (output / 'prepared.json').exists() else None
        for record in (prior, armed):
            if record:
                require(record.get('handoff_contract_sha256') == digest, 'Existing handoff differs')
        require(not prior or armed, 'Transfer lacks preparation record')
        check_queue(config, root, manifest, paused=bool(prior), allow_publication=bool(prior))
        parent_fd, parent = identities.bind_process(config['supervisor']); stack.callback(os.close, parent_fd)
        require(parent['state'] != 'T' or armed, 'Supervisor was paused outside this contract')
        child_fd, _ = identities.bind_process(config['nextflow'], allow_exited=bool(armed and parent['state'] == 'T'))
        stack.callback(os.close, child_fd)
        handles = dict(specs={'active': config['nextflow']}, fds={'active': child_fd})
        if inspect_only:
            return dict(status='INSPECTED_NO_SIGNAL_NO_TRANSFER', active=config['active_chromosome'], pending=config['delegated_chromosomes'])
        paused, transferred = parent['state'] == 'T', bool(prior)
        try:
            resource_guard(config, root, stop)
            if not armed:
                save_once(output / 'prepared.json', dict(handoff_contract_sha256=digest, original_deadline_utc=config['original_deadline_utc']))
            if not transferred:
                if not paused:
                    signal.pidfd_send_signal(parent_fd, signal.SIGSTOP); paused = True
                process.stopped(config['supervisor'])
                check_queue(config, root, manifest, paused=True)
                identities.campaign.sealed_files(config['files'])
                marker = dict(schema=TRANSFER_SCHEMA, status='PENDING_M02_DELEGATED_SUPERVISOR_PAUSED',
                    original_instance=manifest['instance'], delegate_instance=config['delegate_instance'],
                    source_instance_id=config['source_instance_id'],
                    chromosomes=config['delegated_chromosomes'], original_launch_sha256=config['original_launch_sha256'],
                    source_inventory_sha256=worker.sha(root / 'sources.json'), handoff_contract_sha256=digest,
                    adopted_chromosome=config['active_chromosome'], supervisor=config['supervisor'], adopted_nextflow=config['nextflow'],
                    created_utc=utc(), original_deadline_utc=config['original_deadline_utc'])
                save_once(output / 'queue_transfer.json', marker)
                transferred = True
            publish_transfer(path, config, output)
            while not process.pidfd_exited(child_fd):
                resource_guard(config, root, stop)
                check_queue(config, root, manifest, paused=True)
                require(worker.sha(path) == digest, 'Handoff contract changed')
                stop.wait(config['resources']['poll_seconds'])
            resource_guard(config, root, stop)
            exit_path = output / 'accepted-exit.json'
            if not exit_path.exists():
                save_once(exit_path, dict(handoff_sha256=digest, **accepted_exit(config, root, child_fd)))
            else:
                require(worker.read(exit_path)['handoff_sha256'] == digest and identities.kernel_exit_code(config['nextflow'], child_fd) == 0,
                        'Existing exit proof differs')
            receipt = publish_finished(path, digest, config, root, manifest, output, stop, handles)
            result = dict(status='ACTIVE_M02_PUBLISHED_PENDING_DELEGATED', handoff_contract_sha256=digest,
                instance=manifest['instance'], instance_id=config['source_instance_id'],
                active_chromosome=config['active_chromosome'], published=receipt,
                delegated_chromosomes=config['delegated_chromosomes'], no_filter_replayed=True,
                all_original_chromosomes_complete=False, completion_utc=utc())
            # The original startup shuts down the VM when its worker exits. All
            # durable records MUST therefore precede the supervisor's termination.
            result = publish_completion(path, digest, config, output, result)
            identities.stop_paused_supervisor(config['supervisor'], parent_fd, handles['fds'])
            return result
        except BaseException as error:
            save_once(output / ('failure.' + str(time.time_ns()) + '.json'), dict(handoff_contract_sha256=digest,
                error_type=type(error).__name__, error=str(error), queue_transferred=transferred,
                original_artifacts_preserved=True, observed_utc=utc()))
            if transferred:
                close_children(config, manifest, handles)
                identities.stop_paused_supervisor(config['supervisor'], parent_fd, handles['fds'])
            elif paused:
                signal.pidfd_send_signal(parent_fd, signal.SIGCONT)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--verify-complete', action='store_true', help='Read-only GCS proof; requires no live original VM')
    args = parser.parse_args()
    require(sum((args.inspect_only, args.finalize_only, args.verify_complete)) <= 1, 'Choose one operation')
    os.umask(0o077)
    os.environ.update(CLOUDSDK_CORE_ACCOUNT=worker.ACCOUNT, CLOUDSDK_CORE_PROJECT=worker.PROJECT)
    if args.verify_complete:
        print(json.dumps(verify_complete(args.handoff, args.expected_sha256)))
        return
    if args.finalize_only:
        finalize_only(args.handoff, args.expected_sha256)
        return
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    print(json.dumps(run(args.handoff, args.expected_sha256, inspect_only=args.inspect_only, stop=stop)))


if __name__ == '__main__':
    main()
