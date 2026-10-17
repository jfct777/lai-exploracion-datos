#!/usr/bin/env python3
"""Run one bounded, frozen Nextflow evidence workflow and publish its products.

The caller authorizes an exact contract. This controller does not choose
scientific parameters, alter existing jobs, create VMs, or remove work/results.
Only containers with this campaign's exact label can be stopped on failure.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

SCHEMA = 'r02_local_evidence_campaign_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def payload(path):
    total = 0
    for folder, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(folder)/name).is_symlink()]
        for name in files:
            try:
                info = (Path(folder)/name).lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def validate(path, expected):
    path = Path(path).absolute()
    require(path.resolve() == path and sha(path) == expected, 'Campaign contract hash/path mismatch')
    contract = read(path)
    root = path.parent
    require(contract['schema'] == SCHEMA, 'Unsupported campaign schema')
    require(re.fullmatch(r'r02-[a-z0-9-]+', contract['run_id']), 'Unsafe campaign label')
    require(root.name == contract['run_id'], 'Campaign directory/label mismatch')
    for key in ('max_seconds', 'max_new_bytes', 'reserve_bytes', 'expected_tasks'):
        require(type(contract[key]) is int and contract[key] > 0, 'Invalid bound: '+key)
    require(contract['products'] and contract['frozen_sha256'], 'Empty campaign inventory')
    for name in ('workflow', 'config', 'params', 'publisher'):
        source = Path(contract[name])
        require(source.is_relative_to(root) and source.resolve() == source, 'Frozen path escaped campaign')
        require(str(source) in contract['frozen_sha256'], 'Unauthenticated program/configuration')
    products = contract['products']
    require(len({item['label'] for item in products}) == len(products), 'Duplicate product label')
    for item in products:
        require(re.fullmatch(r'[A-Za-z0-9_-]+', item['label']), 'Unsafe product label')
        out = Path(item['path'])
        require(out.is_relative_to(root) and '..' not in out.parts, 'Output escaped campaign')
    for item in contract.get('other_storage_budgets', []):
        require(Path(item['path']).is_absolute() and type(item['cap_bytes']) is int
                and item['cap_bytes'] > 0, 'Invalid sibling storage budget')
    return root, contract


def verify_sources(contract):
    for path, digest in contract['frozen_sha256'].items():
        require(sha(path) == digest, 'Frozen source/input changed: '+Path(path).name)


def disk_guard(root, contract):
    own = payload(root)
    require(own <= contract['max_new_bytes'], 'Campaign storage limit exceeded')
    remaining = contract['max_new_bytes'] - own
    for sibling in contract.get('other_storage_budgets', []):
        used = payload(sibling['path'])
        require(used <= sibling['cap_bytes'], 'Concurrent campaign storage limit exceeded')
        remaining += sibling['cap_bytes'] - used
    free = shutil.disk_usage(root).free
    require(free >= contract['reserve_bytes'] + remaining,
            'Insufficient jointly reserved local space')
    return dict(used_bytes=own, free_bytes=free, reserved_remaining_bytes=remaining)


def stop_owned(process, label):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    result = subprocess.run(['docker', 'ps', '-q', '--filter', 'label=dnabr.run='+label],
                            capture_output=True, text=True, timeout=15, check=True)
    containers = result.stdout.split()
    require(all(re.fullmatch(r'[a-f0-9]{12,64}', item) for item in containers), 'Invalid container identity')
    if containers:
        subprocess.run(['docker', 'stop', '-t', '10', *containers], check=True, timeout=45)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--contract-sha256', required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    root, contract = validate(args.contract, args.contract_sha256)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock = root/'budget_clock.json'
        if not clock.exists():
            save(clock, dict(started_epoch=time.time(), contract_sha256=args.contract_sha256))
        budget = read(clock)
        require(budget['contract_sha256'] == args.contract_sha256, 'Resume contract changed')
        deadline = budget['started_epoch'] + contract['max_seconds']
        require(time.time() < deadline, 'Original campaign deadline exceeded')
        attempt = root/('attempt_'+str(time.time_ns()))
        attempt.mkdir(mode=0o700)
        process = None
        env = {**os.environ, 'NXF_OFFLINE':'true', 'NXF_DISABLE_CHECK_LATEST':'true',
               'NXF_SYNTAX_PARSER':'v1', 'NXF_OPTS':'-Xms64m -Xmx384m',
               'PYTHONDONTWRITEBYTECODE':'1', 'OMP_NUM_THREADS':'1',
               'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
               'CLOUDSDK_CORE_ACCOUNT':contract['gcp_account'],
               'CLOUDSDK_CORE_PROJECT':contract['gcp_project']}

        def execute(command, name):
            nonlocal process
            save(attempt/(name+'_command.json'), dict(command=command, started_epoch=time.time()))
            with (attempt/(name+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    require(time.time() < deadline, 'Campaign deadline exceeded')
                    disk_guard(root, contract)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                require(process.returncode == 0, 'Command failed: '+name)
                process = None

        def interrupted(signum, frame):
            raise RuntimeError('Controller interrupted by signal '+str(signum))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline-time.time()))
        try:
            require(str(Path(__file__).resolve()) in contract['frozen_sha256'],
                    'Executed controller is not authenticated')
            verify_sources(contract)
            filesystem = subprocess.check_output(
                ['findmnt', '-T', str(root), '-n', '-o', 'FSTYPE'], text=True, timeout=10).strip()
            require(filesystem in ('ext4', 'xfs'), 'Local block filesystem required')
            disk_guard(root, contract)
            command = ['nextflow', '-C', contract['config'], '-log', str(attempt/'nextflow.log'),
                       'run', contract['workflow'], '-params-file', contract['params'],
                       '-work-dir', str(root/'work'), '-with-trace', str(attempt/'trace.tsv'),
                       '-ansi-log', 'false']
            if (root/'.nextflow/cache').exists():
                command.append('-resume')
            execute(command, 'workflow')
            import csv
            with (attempt/'trace.tsv').open() as trace:
                tasks = list(csv.DictReader(trace, delimiter='\t'))
            require(len(tasks) == contract['expected_tasks'] and
                    all(t['status'] in ('COMPLETED', 'CACHED') and t['exit'] == '0' for t in tasks),
                    'Incomplete Nextflow task inventory')
            verify_sources(contract)
            sys.path.insert(0, str(Path(contract['publisher']).parent))
            spec = importlib.util.spec_from_file_location('campaign_publisher', contract['publisher'])
            publisher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(publisher)
            os.environ.update({key:env[key] for key in ('CLOUDSDK_CORE_ACCOUNT', 'CLOUDSDK_CORE_PROJECT')})
            products = []
            for item in contract['products']:
                output = Path(item['path'])
                digest = sha(output/'manifest.json')
                _, _, files = publisher.authenticated_bundle(output, digest)
                receipt = root/('publication_'+item['label']+'.json')
                destination = contract['destination']+'/'+item['label']
                if receipt.exists():
                    previous = read(receipt)
                    require(previous['status'] == 'PUBLISHED_VERIFIED' and
                            previous['manifest_sha256'] == digest and previous['destination'] == destination,
                            'Prior publication contract changed')
                    expected = {r['name']:r for r in files}
                    require(set(expected) == {r['name'] for r in previous['files']}, 'Publication inventory changed')
                    for record in previous['files']:
                        require(all(record[k] == expected[record['name']][k] for k in ('sha256','bytes','md5_base64')),
                                'Previously published content changed')
                        publisher.verify_remote(record, pinned=True)
                else:
                    execute([sys.executable, contract['publisher'], '--output-dir', str(output),
                             '--expected-manifest-sha256', digest, '--destination', destination,
                             '--receipt', str(receipt)], 'publish_'+item['label'])
                products.append(dict(label=item['label'], manifest_sha256=digest, receipt_sha256=sha(receipt)))
            verify_sources(contract)
            result = dict(status='COMPLETE_PUBLISHED_NOT_BIOLOGICAL_VALIDATION',
                          completed_epoch=time.time(), products=products, tasks=len(tasks),
                          simulations=False, training=False, contract_sha256=args.contract_sha256)
            save(attempt/'complete.json', result)
            if not (root/'complete.json').exists():
                save(root/'complete.json', result)
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                stop_owned(process, contract['run_id'])
            finally:
                save(attempt/'failed.json', dict(error_type=type(error).__name__,
                     message=str(error), epoch=time.time(), results_preserved=True))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
