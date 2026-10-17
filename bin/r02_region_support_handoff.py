#!/usr/bin/env python3
"""Adopt an active FIT-support task and delegate untouched chromosomes.

The original controller remains paused permanently. Its active Nextflow child
finishes unchanged; successful kernel/task exits are required before publication.
Only orchestration changes. All analysis code, roles, sources and thresholds
remain the frozen original. Run in an independent systemd service.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import r02_optimization_boundary as identity
import r02_region_support_queue as worker

require = worker.require


def bind(spec, exited=False):
    fd = os.pidfd_open(spec['pid'])
    try:
        actual = identity.same_incarnation(spec)
        require(actual is not None, 'Process disappeared')
        if actual['state'] in {'Z', 'X'}:
            require(exited and identity.pidfd_exited(fd), 'Unexpected exited process')
        else:
            require(actual['cmdline_sha256'] == spec['cmdline_sha256'], 'Process identity changed')
        return fd, actual
    except BaseException:
        os.close(fd)
        raise


def children(pid):
    return {int(v) for t in (Path('/proc') / str(pid) / 'task').iterdir()
            for v in (t / 'children').read_text().split()}


def validate_assignments(config):
    assignments = config['assignments']
    require(len(assignments) == 2, 'Expected two delegates')
    require([(a['instance'], a['chromosomes']) for a in assignments] ==
            [('dnabr-r02-fit-a-1005', ['13', '14']), ('dnabr-r02-fit-b-1005', ['15', '16'])],
            'Unexpected or overlapping assignments')
    for a in assignments:
        require(a['output_prefix'] == worker.PREFIX + '/worker-' + a['instance'].removeprefix('dnabr-r02-'),
                'Delegation prefix differs')


def paused(config):
    actual = identity.authenticated(config['supervisor'])
    require(actual is not None and actual['state'] == 'T', 'Original supervisor is no longer paused')


def remaining(config):
    seconds = (datetime.fromisoformat(config['deadline_utc']) - datetime.now(timezone.utc)).total_seconds()
    require(seconds > 0, 'Original execution deadline reached')
    return seconds


def load(config_path, digest):
    require(worker.sha(config_path) == digest, 'Handoff configuration changed')
    config = worker.read(config_path)
    require(config['schema'] == 'r02_region_support_handoff_v1', 'Wrong handoff schema')
    require(config['original_instance'] == 'dnabr-r02-support-1004s' and
            re.fullmatch('[1-9][0-9]+', config['source_instance_id']), 'Original instance identity missing')
    validate_assignments(config)
    root = Path(config['source_root'])
    manifest = worker.verify_bundle(root, config['original_launch_sha256'])
    require(config_path.parent != root and not config_path.parent.is_relative_to(root), 'Use a separate amendment directory')
    require(config['active_chromosome'] in manifest['chromosomes'] and
            config['active_chromosome'] not in {'13', '14', '15', '16'}, 'Active task cannot be delegated')
    require(set(config['code_sha256']) == {'r02_region_support_handoff.py', 'r02_region_support_queue.py',
            'r02_optimization_boundary.py'}, 'Incomplete original orchestration hashes')
    # The transfer's original code remains immutable. A separately authenticated
    # runtime revision may fix orchestration without rewriting that transfer.
    for name, expected in config['code_sha256'].items():
        require(worker.sha(config_path.parent / 'code' / worker.publication.safe_relative(name)) == expected,
                'Original amendment code changed')
    return config, root, manifest


def prepare(config_path, digest):
    config, root, manifest = load(config_path, digest)
    target = config_path.parent / 'queue_transfer.json'
    with ExitStack() as stack:
        parent_fd, parent = bind(config['supervisor']); stack.callback(os.close, parent_fd)
        child_fd, child = bind(config['nextflow'], exited=target.exists()); stack.callback(os.close, child_fd)
        if target.exists():
            paused(config)
            require(worker.read(target)['handoff_sha256'] == digest, 'Different transfer already exists')
            return worker.read(target)
        require(parent['state'] != 'T', 'Controller paused without this transfer')
        require(child['ppid'] == parent['pid'] and child['sid'] == child['pgid'] == child['pid'],
                'Active task does not have an independent authenticated session')
        args = (Path('/proc') / str(child['pid']) / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        folder = root / 'chromosomes' / ('chr' + config['active_chromosome'].zfill(2))
        require(str(folder / 'params.json') in args and str(folder / 'trace.initial.tsv') in args and
                str(root / 'frozen/workflows/r02_structure_region_support.nf') in args, 'Active task scope differs')
        require(children(parent['pid']) == {child['pid']}, 'Unexpected active controller children')
        for c in ('13', '14', '15', '16'):
            require(not (root / 'chromosomes' / ('chr' + c)).exists(), 'Delegated chromosome already started')
        worker.save_once(config_path.parent / 'active-command.json', args)
        signal.pidfd_send_signal(parent_fd, signal.SIGSTOP)
        try:
            for _ in range(100):
                if identity.authenticated(config['supervisor'])['state'] == 'T': break
                time.sleep(.01)
            paused(config)
            require(children(parent['pid']) == {child['pid']}, 'Child set changed during pause')
            for c in ('13', '14', '15', '16'):
                require(not (root / 'chromosomes' / ('chr' + c)).exists(), 'Delegated chromosome started during pause')
            value = dict(schema='r02_region_support_transfer_v1',
                         status='PENDING_SUPPORT_DELEGATED_SUPERVISOR_PAUSED',
                         original_instance=config['original_instance'], source_instance_id=config['source_instance_id'],
                         original_launch_sha256=config['original_launch_sha256'], supervisor=config['supervisor'],
                         nextflow=config['nextflow'], assignments=config['assignments'], handoff_sha256=digest,
                         deadline_utc=config['deadline_utc'], created_utc=datetime.now(timezone.utc).isoformat())
            worker.save_once(target, value)
            return value
        except BaseException:
            if not target.exists(): signal.pidfd_send_signal(parent_fd, signal.SIGCONT)
            raise


def accepted_exit(config, root, fd):
    require(identity.pidfd_exited(fd), 'Active Nextflow still running')
    spec = config['nextflow']
    fields = (Path('/proc') / str(spec['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    require(int(fields[19]) == spec['start_ticks'] and fields[0] in {'Z', 'X'} and
            os.waitstatus_to_exitcode(int(fields[49])) == 0, 'Active Nextflow kernel exit not zero')
    folder = root / 'chromosomes' / ('chr' + config['active_chromosome'].zfill(2))
    trace = folder / 'trace.initial.tsv'
    with trace.open() as handle: rows = list(csv.DictReader(handle, delimiter='\t'))
    require(len(rows) == 1 and rows[0]['status'] in {'COMPLETED', 'CACHED'} and rows[0]['exit'] == '0' and
            rows[0]['name'].split(' (')[0].split(':')[-1] == 'R02_STRUCTURE_REGION_SUPPORT', 'Unexpected task trace')
    task_hash = rows[0]['hash']
    require(re.fullmatch('[a-f0-9]{2}/[a-f0-9]{6,}', task_hash), 'Invalid task hash')
    paths = list((folder / 'work').glob(task_hash + '*'))
    require(len(paths) == 1 and (paths[0] / '.exitcode').read_text().strip() == '0', 'Task did not exit successfully')
    return dict(kernel_exit=0, trace_sha256=worker.sha(trace), exitcode_sha256=worker.sha(paths[0] / '.exitcode'))


def fetch_delegate(root, manifest, chrom, delegate, transfer_sha):
    uri = delegate['output_prefix'] + '/chr' + chrom.zfill(2) + '/completed.json'
    response = subprocess.run(['gcloud', 'storage', 'objects', 'describe', uri, '--format=json'],
                              capture_output=True, text=True, timeout=120)
    if response.returncode:
        message = response.stderr + response.stdout
        require(worker.missing_gcs_object(response, uri),
                'Delegate publication inaccessible; not a missing-product assumption: ' + message[-2000:])
        return None
    meta = json.loads(response.stdout)
    require(0 < int(meta['size']) < 8 * 2**20, 'Delegate receipt too large')
    data = subprocess.check_output(['gcloud', 'storage', 'cat', uri + '#' + str(meta['generation'])], timeout=120)
    folder = root / 'chromosomes' / ('chr' + chrom.zfill(2)); folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    local = folder / 'completed.json'
    if local.exists(): require(local.read_bytes() == data, 'Delegate receipt changed')
    else:
        with local.open('xb') as handle: handle.write(data)
    receipt = dict(name='completed.json', path=str(local), uri=uri, generation=str(meta['generation']),
                   **worker.pipeline.publication_digests(local))
    worker.publication.verify_remote(receipt, pinned=True)
    result = worker.read(local)
    require(result['chromosome'] == chrom and result['launch_sha256'] == delegate['launch_sha256'] and
            result['status'] == 'FIT_SUPPORT_PUBLISHED_VERIFIED_NOT_MODEL_EVALUATION', 'Delegate scope differs')
    names = {r['name'] for r in result['files']}
    mandatory = {'fit_eligible.tsv', 'retained.tsv', 'target_support.tsv', 'ld_diagnostic.tsv',
                 'manifest.json', 'provenance.json', 'source-receipt.json', 'contract.json'}
    traces = {n for n in names if re.fullmatch(r'trace\.(initial|resume[1-9][0-9]*)\.tsv', n)}
    require(len(traces) == 1 and len(result['files']) == len(names) and mandatory <= names and
            names <= mandatory | traces | {'ld_tool.log'}, 'Incomplete/duplicate delegate outputs')
    for item in result['files']:
        require(item['uri'] == delegate['output_prefix'] + '/chr' + chrom.zfill(2) + '/' + str(worker.publication.safe_relative(item['name'])),
                'Delegate output escaped assignment')
        worker.publication.verify_remote(item, pinned=True)
    provenance = next(r for r in result['files'] if r['name'] == 'provenance.json')
    content = subprocess.check_output(['gcloud', 'storage', 'cat', provenance['uri'] + '#' + provenance['generation']], timeout=120)
    require(hashlib.sha256(content).hexdigest() == provenance['sha256'] and
            json.loads(content)['ownership_receipt_sha256'] == transfer_sha, 'Delegate does not reference this transfer')
    return receipt


def run(config_path, digest, delegate_path, delegate_sha, runtime_sha=None):
    config, root, manifest = load(config_path, digest)
    paused(config)
    require(worker.read(config_path.parent / 'queue_transfer.json')['handoff_sha256'] == digest, 'Transfer not sealed')
    transfer_sha = worker.sha(config_path.parent / 'queue_transfer.json')
    require(worker.sha(delegate_path) == delegate_sha, 'Delegation launches changed')
    delegates = worker.read(delegate_path)
    require(len(delegates) == 2, 'Expected two frozen delegate launches')
    for expected, actual in zip(config['assignments'], delegates):
        require(all(actual[k] == v for k, v in expected.items()), 'Delegate launch changes assignment')
        worker.publication.validate_sha(actual['launch_sha256'])
    manifest['max_runtime_seconds'] = remaining(config)
    manifest['execution_amendment_sha256'] = runtime_sha or digest
    started = time.monotonic()
    adopted = config_path.parent / 'active-published.json'
    if not adopted.exists():
        fd, _ = bind(config['nextflow'], exited=True)
        try:
            while not identity.pidfd_exited(fd):
                paused(config); remaining(config); worker.check_disk(root, manifest); time.sleep(10)
            folder = root / 'chromosomes' / ('chr' + config['active_chromosome'].zfill(2))
            if not (folder / 'publication-plan.json').exists():
                worker.save_once(config_path.parent / 'accepted-exit.json', accepted_exit(config, root, fd))
                names = {key: folder / ('%s.initial.%s' % (key, ext))
                         for key, ext in (('trace', 'tsv'), ('nextflow', 'log'), ('console', 'log'))}
                args = worker.read(config_path.parent / 'active-command.json')
                result = worker.finalize_chromosome(root, manifest, folder, config['active_chromosome'], args, names, 'initial')
            else:
                result = worker.publish_chromosome(root, manifest, folder, worker.read(folder / 'publication-plan.json'))
            worker.save_once(adopted, result)
        finally: os.close(fd)
    results = {}
    while len(results) != len(manifest['entries']):
        changed = False
        for entry in manifest['entries']:
            paused(config); remaining(config)
            c = entry['chromosome']
            if c in results: continue
            delegate = next((d for d in delegates if c in d['chromosomes']), None)
            result = fetch_delegate(root, manifest, c, delegate, transfer_sha) if delegate else worker.run_chromosome(root, manifest, entry, started, 'resume1')
            if result is not None:
                results[c] = result; changed = True
                print(json.dumps(dict(chromosome=c, completed=len(results), delegated=bool(delegate))), flush=True)
        if not changed: time.sleep(min(120, remaining(config)))
    worker.save_once(root / 'complete.json', dict(schema=worker.SCHEMA, status='ALL_ASSIGNED_SUPPORT_PUBLISHED_VERIFIED',
                     chromosomes=manifest['chromosomes'], launch_sha256=manifest['authenticated_launch_sha256'],
                     results=[results[c] for c in manifest['chromosomes']], handoff_sha256=digest, delegated_launches_sha256=delegate_sha,
                     execution_runtime_sha256=runtime_sha))
    receipt = dict(name='complete.json', path=str(root / 'complete.json'), uri=manifest['output_prefix'] + '/complete.json',
                   **worker.pipeline.publication_digests(root / 'complete.json'))
    if (root / 'complete.publication.json').exists(): receipt = worker.read(root / 'complete.publication.json')
    else:
        worker.publication.upload_one(receipt); worker.publication.verify_remote(receipt, pinned=True)
        worker.save_once(root / 'complete.publication.json', receipt)
    return worker.verify_completion(root, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--sha256', required=True)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--delegates', type=Path)
    parser.add_argument('--delegates-sha256')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(CLOUDSDK_CORE_ACCOUNT='653458115080-compute@developer.gserviceaccount.com', CLOUDSDK_CORE_PROJECT='uspbr-242713',
                      PYTHONDONTWRITEBYTECODE='1', R02_WORKER_UID=str(os.getuid()), R02_WORKER_GID=str(os.getgid()))
    require(worker.sha(args.runtime) == worker.publication.validate_sha(args.runtime_sha256), 'Runtime manifest changed')
    runtime = worker.read(args.runtime)
    require(runtime['schema'] == 'r02_support_handoff_runtime_v1' and
            set(runtime['files']) == {'r02_region_support_handoff.py','r02_region_support_queue.py','r02_optimization_boundary.py'},
            'Incomplete runtime declaration')
    for name, expected in runtime['files'].items():
        require(worker.sha(Path(__file__).parent / name) == expected, 'Runtime code changed')
    with (args.config.parent / '.handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(prepare(args.config, args.sha256) if args.prepare else
                         run(args.config, args.sha256, args.delegates, args.delegates_sha256, args.runtime_sha256)), flush=True)


if __name__ == '__main__': main()
