#!/usr/bin/env python3
"""Inspect the authorized R02 workers without changing scientific files or jobs.

Remote commands only read files, process metadata and Docker state. gcloud SSH
uses the supplied temporary key and an explicit two-hour authorization expiry.
Each invocation writes a new local timestamped verification record.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

PROJECT = 'uspbr-242713'
ZONE = 'us-central1-a'
MARKER = 'R02_VERIFICATION_JSON='
CORE_FILES = (
    'modules/01_preprocess_norm_leftalign.nf',
    'modules/02_preprocess_filter_snv_biallelic_pass.nf',
    'modules/lai_rare_bialelic_only.nf',
    'bin/r02_autosome_pipeline.py', 'bin/rare_allele_sharing_painter.py',
    'bin/rare_segment_sensitivity.py', 'bin/r02_genomic_pair_evidence.py',
    'bin/r02_common_grm.sh', 'bin/r02_m14_configuration_diagnostics.py',
    'bin/m165_autosome_sweep.py', 'bin/m165_chr22_sweep.py',
    'bin/r02_weighted_communities.py', 'bin/m165_spectral_figures.py',
)

# Self-contained code: transmitted in the SSH command, never saved remotely.
REMOTE_SCRIPT = r'''
import hashlib, json, os, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
run = Path(sys.argv[1]).resolve()
core = json.loads(sys.argv[2])
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
def verify(root, manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Empty or invalid manifest')
    failures, hashes = [], {}
    for name, expected in manifest.items():
        rel = Path(name)
        path = root / rel
        if rel.is_absolute() or '..' in rel.parts or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            failures.append({'path': name, 'reason': 'missing_or_unsafe'})
            continue
        actual = sha(path)
        if actual != expected:
            failures.append({'path': name, 'reason': 'sha256_mismatch'})
        if name in core:
            hashes[name] = {'actual_sha256': actual, 'manifest_sha256': expected}
    return {'files_checked': len(manifest), 'failures': failures, 'core_hashes': hashes}
config = read(run / 'run.json')
source = verify(run / 'source', read(run / 'source.sha256.json'))
frozen = verify(run, read(run / 'frozen.sha256.json'))
status = read(run / 'status.json')
containers, docker_error = [], None
try:
    ids = subprocess.check_output(['docker', 'ps', '-q', '--filter', 'label=dnabr-r02=' + config['run_id']], text=True, timeout=15).split()
    if ids:
        lines = subprocess.check_output(['docker', 'inspect', '--format', '{{.Id}}\t{{.Image}}\t{{.State.Status}}', *ids], text=True, timeout=15).splitlines()
        for line in lines:
            ident, image, state = line.split('\t')
            containers.append({'id': ident, 'image_id': image, 'state': state,
                               'expected_image': image in [config['prep_image'], config['analysis_image']]})
except (subprocess.SubprocessError, OSError, ValueError) as error:
    docker_error = type(error).__name__
submissions = []
for chromosome in config['processing_order']:
    log = run / ('chr%02d' % chromosome) / 'nextflow.log'
    if not log.is_file():
        continue
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 262144))
        tail = stream.read().decode('utf-8', 'replace')
    lines = [line for line in tail.splitlines() if 'Submitted process >' in line]
    submissions.append({'chromosome': chromosome, 'log': str(log),
                        'submitted_lines': lines[-12:],
                        'm01_submitted': any('PREPROCESS_NORM_LEFTALIGN' in line for line in lines)})
result = {'checked_utc': datetime.now(timezone.utc).isoformat(),
          'worker_run_id': config['run_id'], 'run_sha256': sha(run / 'run.json'),
          'source_manifest_sha256': sha(run / 'source.sha256.json'),
          'frozen_manifest_sha256': sha(run / 'frozen.sha256.json'),
          'source_verification': source, 'frozen_verification': frozen,
          'status': {key: status.get(key) for key in ('updated_utc', 'stage', 'state', 'pid', 'free_disk_gib')},
          'containers': containers, 'docker_error': docker_error,
          'nextflow_submissions': submissions,
          'm01_submission_confirmed': any(item['m01_submitted'] for item in submissions),
          'active_container_confirmed': any(item['state'] == 'running' for item in containers)}
print('R02_VERIFICATION_JSON=' + json.dumps(result, sort_keys=True))
'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_workers(value):
    try:
        indices = [int(item) for item in value.split(',')]
    except ValueError as error:
        raise ValueError('Workers must be distinct indices 1–12') from error
    if not indices or len(indices) != len(set(indices)) or any(i < 1 or i > 12 for i in indices):
        raise ValueError('Workers must be distinct indices 1–12')
    return indices


def read(path):
    return json.loads(Path(path).read_text())


def remote_command(run_dir):
    return 'PYTHONDONTWRITEBYTECODE=1 python3 -c ' + shlex.quote(REMOTE_SCRIPT) + ' ' + shlex.quote(run_dir) + ' ' + shlex.quote(json.dumps(CORE_FILES))


def parse_remote(stdout):
    matches = [line[len(MARKER):] for line in stdout.splitlines() if line.startswith(MARKER)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one remote verification record')
    return json.loads(matches[0])


def verify_worker(spec, worker, folder, key, workspace):
    started = time.monotonic()
    result = {'worker': worker['worker'], 'name': worker['name'], 'verification': 'UNVERIFIED'}

    def invoke(args):
        remaining = 120 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Worker verification exceeded 120 seconds')
        completed = subprocess.run(args, capture_output=True, text=True, timeout=remaining)
        if completed.returncode:
            # Avoid recording arbitrary gcloud output, identities or credential messages.
            raise RuntimeError('Diagnostic command failed with exit ' + str(completed.returncode))
        return completed.stdout

    try:
        receipt = read(folder / (worker['worker'] + '.creation.json'))
        created = receipt.get('response') or []
        matching = [item for item in created if item.get('name') == worker['name']]
        if receipt.get('returncode') != 0 or len(matching) != 1:
            raise ValueError('Successful matching creation receipt required')
        described = json.loads(invoke(['gcloud', 'compute', 'instances', 'describe', worker['name'],
            '--project=' + PROJECT, '--zone=' + ZONE, '--format=json', '--quiet']))
        if str(described.get('id')) != str(matching[0].get('id')) or described.get('name') != worker['name']:
            raise ValueError('Live instance identity differs from creation receipt')
        result['instance'] = {key: described.get(key) for key in ('id', 'name', 'status', 'lastStartTimestamp')}
        if described.get('status') != 'RUNNING':
            result['verification'] = 'INSTANCE_NOT_RUNNING'
            return result
        stdout = invoke(['gcloud', 'compute', 'ssh', 'jose.tantalean@' + worker['name'],
            '--project=' + PROJECT, '--zone=' + ZONE, '--ssh-key-file=' + str(key),
            '--ssh-key-expire-after=2h', '--quiet', '--command=' + remote_command(worker['run_dir']),
            '--', '-o', 'ConnectTimeout=20', '-o', 'ConnectionAttempts=1'])
        remote = parse_remote(stdout)
        result['remote'] = remote
        source = remote['source_verification']
        if (remote['run_sha256'] != worker['run_sha256']
                or remote['source_manifest_sha256'] != spec['scientific_source_manifest_sha256']
                or source['failures'] or remote['frozen_verification']['failures']):
            result['verification'] = 'CODE_OR_CONTRACT_MISMATCH'
            return result
        if set(source['core_hashes']) != set(CORE_FILES):
            raise ValueError('Missing required core files in verified source')
        for relative, hashes in source['core_hashes'].items():
            path = workspace / relative
            current = sha(path) if path.is_file() else None
            hashes['workspace_sha256'] = current
            hashes['same_as_workspace'] = current == hashes['actual_sha256']
        if any(not item['expected_image'] for item in remote['containers']):
            result['verification'] = 'UNEXPECTED_RUNNING_IMAGE'
        elif remote['m01_submission_confirmed'] and remote['active_container_confirmed']:
            result['verification'] = 'VERIFIED_CODE_AND_ACTIVE_ANALYSIS'
        elif remote['status']['state'] == 'COMPLETE':
            result['verification'] = 'VERIFIED_CODE_REPORTED_COMPLETE'
        else:
            result['verification'] = 'VERIFIED_CODE_ANALYSIS_NOT_CONFIRMED_ACTIVE'
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        result['error_type'] = type(error).__name__
        result['error'] = str(error) if not isinstance(error, subprocess.SubprocessError) else 'Diagnostic command timed out'
    finally:
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--ssh-key-file', required=True, type=Path)
    parser.add_argument('--workers', required=True)
    args = parser.parse_args()
    try:
        indices = parse_workers(args.workers)
    except ValueError as error:
        parser.error(str(error))
    key = args.ssh_key_file.resolve()
    if not key.is_file() or not Path(str(key) + '.pub').is_file():
        parser.error('Existing temporary SSH private/public key files are required')
    run = args.run_dir.resolve()
    folder = run / 'repairs/parallel-v1'
    spec = read(folder / 'fleet.json')
    if spec['project'] != PROJECT or spec['zone'] != ZONE:
        parser.error('Unexpected project or zone')
    workers = {item['worker']: item for item in spec['workers']}
    selected = [workers['worker%02d' % index] for index in indices]
    workspace = Path(__file__).resolve().parents[1]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda worker: verify_worker(spec, worker, folder, key, workspace), selected))
    timestamp = datetime.now(timezone.utc)
    output = folder / ('runtime_verification_' + timestamp.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    record = {'schema_version': 1, 'checked_utc': timestamp.isoformat(),
        'fleet_sha256': sha(folder / 'fleet.json'), 'verifier_sha256': sha(Path(__file__)),
        'remote_scientific_files_modified': False, 'max_concurrency': 3,
        'per_worker_timeout_seconds': 120, 'workers': results,
        'interpretation': 'Operational provenance and activity verification; not biological validation'}
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'record': str(output), 'workers': [
        {'worker': item['worker'], 'verification': item['verification']} for item in results]}, indent=2))
    if any(item['verification'] not in ('VERIFIED_CODE_AND_ACTIVE_ANALYSIS', 'VERIFIED_CODE_REPORTED_COMPLETE') for item in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
