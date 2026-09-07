#!/usr/bin/env python3
"""Seal a single-worker GPU benchmark and detach its Nextflow controller.

Google Batch owns the disposable worker and enforces the task timeout even if
this development VM disappears. The additional controller deadline includes
queueing and setup while this VM remains available; it is not a cloud billing
guarantee. Only this run's labelled unfinished job may be cancelled.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import time

from m39_launch_ordered_training import _private_path, sha256, write_exclusive
from m39_launch_ordered_throughput import RUNTIME_SOURCES, _case_ids, stop_process_group


PROFILE = 'conf/m39_ordered_throughput_gpu_profile.json'
CONFIG = 'conf/m39_gpu_profile_google_batch.config'
WORKFLOW = 'workflows/m39_gpu_profile.nf'
RUNTIME = (*RUNTIME_SOURCES, 'm39_gpu_serial_profile.py')
SOURCES = (PROFILE, CONFIG, WORKFLOW, 'modules/39_GPU_SERIAL_PROFILE.nf',
           'bin/m39_gpu_launch.py', 'bin/m39_launch_ordered_training.py',
           'bin/m39_launch_ordered_throughput.py', *(f'bin/{name}' for name in RUNTIME))
OWN_RUNS = 'gs://teams-usp/frank/lai-exploracion-datos/runs/'
CONTROLLER_SECONDS = 3600
AUTH_OVERRIDES = ('GOOGLE_APPLICATION_CREDENTIALS', 'DEVSHELL_CLIENT_PORT', 'NO_GCE_CHECK',
                  'CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT', 'CLOUDSDK_AUTH_ACCESS_TOKEN',
                  'CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE', 'CLOUDSDK_AUTH_ACCESS_TOKEN_FILE')


def write_json(path: Path, value: dict) -> None:
    write_exclusive(path, (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())


def validate_target(run_id: str, image: str) -> str:
    if not re.fullmatch(r'm39-gpu-[a-z0-9-]{3,45}', run_id):
        raise ValueError('Use a safe, unique m39-gpu-* run identifier')
    if not re.fullmatch(r'us-central1-docker\.pkg\.dev/uspbr-242713/dnabr-lai/[a-z0-9-]+@sha256:[a-f0-9]{64}', image):
        raise ValueError('GPU image must be pinned by digest in the project registry')
    return OWN_RUNS + run_id


def native_auth(path: Path, service_account: str, repo: Path) -> dict:
    """Select native VM credentials without reading or moving a credential file."""
    if not isinstance(service_account, str) or not re.fullmatch(
            r'[a-z0-9][a-z0-9-]*@(?:developer\.gserviceaccount\.com|'
            r'[a-z0-9][a-z0-9-]*\.iam\.gserviceaccount\.com)', service_account):
        raise ValueError('Expected an explicit service-account email')
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('Native authentication directory cannot use symlinks')
    directory, repository = path.resolve(strict=True), repo.resolve(strict=True)
    if not directory.is_dir() or directory.is_relative_to(repository):
        raise ValueError('Native authentication directory must be outside the repository')
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError('Native authentication directory must be owned privately with mode 0700')
    if any(item.is_symlink() or item.name == 'application_default_credentials.json'
           for item in directory.rglob('*')):
        raise ValueError('Native authentication directory contains ADC or a symlink')
    return {'mode': 'attached_vm_metadata', 'directory': str(directory),
            'service_account': service_account, 'repository_root': str(repository)}


def native_env_prefix(auth: dict) -> list[str]:
    """Seal process-local identity; tmux servers need not inherit the caller's env."""
    checked = native_auth(Path(auth['directory']), auth['service_account'], Path(auth['repository_root']))
    if auth != checked:
        raise ValueError('Native authentication declaration differs')
    unset = [word for name in AUTH_OVERRIDES for word in ('-u', name)]
    return ['env', *unset, f"CLOUDSDK_CONFIG={auth['directory']}",
            f"CLOUDSDK_CORE_ACCOUNT={auth['service_account']}",
            f"M39_GPU_SERVICE_ACCOUNT={auth['service_account']}"]


def watcher_command(receipt: Path, auth: dict) -> list[str]:
    watcher = receipt.parent / 'frozen-source/bin/m39_gpu_launch.py'
    return [*native_env_prefix(auth), sys.executable, str(watcher), '--watch', str(receipt)]


def prepare(args: argparse.Namespace, repo: Path | None = None) -> Path:
    """Freeze committed code and exact inputs locally, without starting cloud work."""
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    auth = native_auth(args.native_auth_dir, args.service_account, repo)
    run = _private_path(args.run_dir, repo / '.claude/runs', directory=True)
    cloud = validate_target(run.name, args.image)
    if any((run / name).exists() for name in ('frozen-source', 'gpu.launch.json', 'gpu.completion.json')):
        raise ValueError('Use a new immutable run directory')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip():
        raise ValueError('Commit tracked source changes before preparing a launch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    blobs = {}
    for name in SOURCES:
        source = repo / name
        blob = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=repo)
        if source.is_symlink() or not source.is_file() or source.read_bytes() != blob:
            raise ValueError(f'Source differs from committed regular file: {name}')
        blobs[name] = blob
    cfg = json.loads(blobs[PROFILE])
    cases = _case_ids(cfg)
    if cfg.get('schema_version') != 'm39-ordered-throughput-gpu-profile-v1' or cfg.get('device') != 'cuda:0':
        raise ValueError('Expected the frozen GPU profile')
    inputs = {}
    for name, path, directory in (('store_dir', args.store_dir, True),
                                   ('parent_receipt', args.parent_receipt, False), ('folds', args.folds, False)):
        inputs[name] = _private_path(path, repo / '.claude/runs', directory=directory)
    store = inputs['store_dir']
    manifest_path = store / 'manifest.json'
    for name, path in (('store_manifest', manifest_path), ('parent_receipt', inputs['parent_receipt']),
                       ('folds', inputs['folds'])):
        if path.is_symlink() or sha256(path) != cfg[f'{name}_sha256']:
            raise ValueError(f'Frozen input differs: {name}')
    manifest = json.loads(manifest_path.read_text())
    files = {'manifest.json'}
    total_bytes = manifest_path.stat().st_size
    for spec in manifest['arrays'].values():
        name = spec['file']
        if not re.fullmatch(r'[A-Za-z0-9_]+\.npy', name) or name in files:
            raise ValueError('Unsafe or duplicate store member')
        path = store / name
        if path.is_symlink() or not path.is_file() or sha256(path) != spec['sha256']:
            raise ValueError(f'Store member differs: {name}')
        files.add(name)
        total_bytes += path.stat().st_size
    if {path.name for path in store.iterdir()} != files or len(files) != 29 or total_bytes > 100 * 1024**2:
        raise ValueError('Unexpected store inventory or staging size')
    frozen = run / 'frozen-source'
    frozen.mkdir(mode=0o700)
    for name, blob in blobs.items():
        target = frozen / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_exclusive(target, blob)
    source_hashes = {name: hashlib.sha256(blobs[f'bin/{name}']).hexdigest() for name in RUNTIME}
    profile_sha = hashlib.sha256(blobs[PROFILE]).hexdigest()
    seal = {'schema_version': 'm39-gpu-source-seal-v1', 'source_commit': commit,
            'profile_sha256': profile_sha, 'source_sha256': source_hashes}
    write_json(run / 'source-seal.json', seal)
    command = [*native_env_prefix(auth), 'NXF_VER=26.04.6', 'NXF_OFFLINE=true', f'M39_GPU_RUN_ID={run.name}',
               f'M39_GPU_IMAGE={args.image}', 'nextflow', '-C', str(frozen / CONFIG),
               '-log', str(run / 'gpu.nextflow.log'), 'run', str(frozen / WORKFLOW),
               '-work-dir', cloud + '/work', '-with-trace', str(run / 'gpu.trace.tsv'),
               '-with-report', str(run / 'gpu.report.html'), '-with-timeline', str(run / 'gpu.timeline.html')]
    values = {**inputs, 'profile_config': frozen / PROFILE, 'source_seal': run / 'source-seal.json',
              'source_commit': commit, 'profile_sha256': profile_sha}
    for name, value in values.items():
        command.extend((f'--m39_{name}', str(value)))
    receipt = {'schema_version': 'm39-gpu-launch-v1', 'source_commit': commit,
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'run_id': run.name, 'image': args.image, 'argv': command, 'cloud_prefix': cloud,
               'native_auth': auth,
               'source_sha256': {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
               'source_seal_sha256': sha256(run / 'source-seal.json'), 'case_ids': cases,
               'input_bytes': total_bytes + inputs['folds'].stat().st_size + inputs['parent_receipt'].stat().st_size,
               'scope': 'technical_GPU_profile_synthetic_labels_no_accuracy',
               'resources': {'new_batch_workers': 1, 'machine_type': 'g2-standard-8', 'gpu': 'NVIDIA L4',
                             'gpu_count': 1, 'cpus_per_task': 2, 'memory_gib_per_task': 8,
                             'disk_gb': 50, 'task_timeout_seconds': 1800,
                             'controller_timeout_seconds': CONTROLLER_SECONDS, 'max_retries': 0,
                             'provisioning_time_not_covered_by_task_timeout': True},
               'outputs': cloud + '/outputs', 'labels': {'team': 'frank', 'm39_run': run.name}}
    write_json(run / 'gpu.launch.json', receipt)
    return run / 'gpu.launch.json'


def observed_job_ids(run: Path, run_id: str) -> dict[str, str | None]:
    """Recover exact native IDs before considering a potentially slow inventory."""
    jobs = {}
    log = run / 'gpu.nextflow.log'
    if log.is_file():
        pattern = re.compile(r'\[GOOGLE BATCH\] Process `M39_GPU_SERIAL_PROFILE[^`]*` submitted > '
                             r'job=([a-z0-9-]+); uid=([a-z0-9-]+); work-dir=(gs://[^\s]+)')
        for match in pattern.finditer(log.read_text(errors='replace')):
            if not match[3].startswith(OWN_RUNS + run_id + '/work/'):
                raise ValueError('Submitted job work directory differs from this run')
            jobs[match[1]] = match[2]
    trace = run / 'gpu.trace.tsv'
    if trace.is_file():
        with trace.open() as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                name = row.get('native_id', '')
                if re.fullmatch(r'[a-z0-9-]+', name) and name != '-':
                    jobs.setdefault(name, None)
    if len(jobs) > 1:
        raise ValueError('Expected one Batch job, found several native IDs')
    return jobs


def cancel_owned_jobs(run_id: str, run: Path | None = None, *, auth: dict | None = None) -> dict:
    """Cancel only unfinished jobs bearing both exact ownership labels."""
    validate_target(run_id, 'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/check@sha256:' + '0' * 64)
    result = {'deleted_unfinished_jobs': [], 'errors': []}
    try:
        if auth is None:
            raise ValueError('Cancellation requires the sealed native identity')
        command = [*native_env_prefix(auth), 'gcloud', f"--account={auth['service_account']}"]
        exact = observed_job_ids(run, run_id) if run is not None else {}
        if exact:
            jobs = [json.loads(subprocess.check_output([*command, 'batch', 'jobs', 'describe', name,
                    '--project=uspbr-242713', '--location=us-central1', '--format=json'], timeout=20))
                    for name in exact]
            result['lookup'] = 'exact_native_id_from_sealed_run'
        else:
            jobs = json.loads(subprocess.check_output([*command, 'batch', 'jobs', 'list',
                    '--project=uspbr-242713', '--location=us-central1',
                    f'--filter=labels.m39_run={run_id}', '--format=json'], timeout=25))
            result['lookup'] = 'label_fallback_native_id_unavailable'
        for job in jobs:
            if job.get('labels', {}).get('m39_run') != run_id or job['labels'].get('team') != 'frank':
                raise ValueError('Batch query returned a job outside the exact owned run')
            short_name = job.get('name', '').rsplit('/', 1)[-1]
            if exact and (short_name not in exact or
                          (exact[short_name] is not None and job.get('uid') != exact[short_name])):
                raise ValueError('Batch identity differs from the submitted native ID/UID')
            if job.get('status', {}).get('state') in ('SUCCEEDED', 'FAILED', 'DELETION_IN_PROGRESS'):
                continue
            name = job['name']
            if not re.fullmatch(r'projects/uspbr-242713/locations/us-central1/jobs/[a-z0-9-]+', name):
                raise ValueError('Unexpected Batch resource path')
            subprocess.run([*command, 'batch', 'jobs', 'delete', name.rsplit('/', 1)[-1],
                            '--project=uspbr-242713', '--location=us-central1', '--quiet'], check=True, timeout=30)
            result['deleted_unfinished_jobs'].append(name)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result['errors'].append(type(exc).__name__)
    return result


def watch(path: Path) -> int:
    receipt = json.loads(path.read_text())
    if receipt.get('schema_version') != 'm39-gpu-launch-v1' or path.name != 'gpu.launch.json':
        raise ValueError('Expected a GPU launch receipt')
    validate_target(receipt['run_id'], receipt['image'])
    auth = receipt['native_auth']
    prefix = native_env_prefix(auth)
    if receipt['argv'][:len(prefix)] != prefix:
        raise ValueError('Nextflow identity differs from the sealed native identity')
    run = path.parent
    for name, digest in receipt['source_sha256'].items():
        if sha256(run / 'frozen-source' / name) != digest:
            raise ValueError('Frozen source changed before detached launch')
    if sha256(run / 'source-seal.json') != receipt['source_seal_sha256']:
        raise ValueError('Source seal changed before detached launch')
    start = time.monotonic()
    result = {'schema_version': 'm39-gpu-controller-completion-v1', 'run_id': receipt['run_id'],
              'status': 'FAILED', 'exit_code': 1, 'launch_sha256': sha256(path)}
    process = None
    previous = {}
    def interrupted(signum, frame):
        raise InterruptedError(f'GPU controller received signal {signum}')
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        with (run / 'gpu.controller.log').open('x') as log:
            process = subprocess.Popen(receipt['argv'], cwd=run, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            result['exit_code'] = process.wait(timeout=CONTROLLER_SECONDS)
            result['status'] = 'NEXTFLOW_COMPLETED_NEEDS_PRIMARY_POST' if result['exit_code'] == 0 else 'FAILED'
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(failure_type=type(exc).__name__, exit_code=124 if isinstance(exc, subprocess.TimeoutExpired) else 1)
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            # Nextflow first receives SIGTERM and may cancel its own exact task.
            stop_process_group(process)
        if result['exit_code']:
            result['cloud_cleanup'] = cancel_owned_jobs(receipt['run_id'], run, auth=auth)
        result['elapsed_seconds'] = time.monotonic() - start
        write_json(run / 'gpu.completion.json', result)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return result['exit_code'] if result['exit_code'] >= 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--prepare-only', action='store_true')
    for name in ('run-dir', 'store-dir', 'parent-receipt', 'folds'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--image')
    parser.add_argument('--native-auth-dir', type=Path,
                        help='Existing private directory outside the repository, without ADC')
    parser.add_argument('--service-account', help='Verified service account attached to this VM')
    args = parser.parse_args()
    os.umask(0o077)
    if args.watch:
        raise SystemExit(watch(args.watch.resolve(strict=True)))
    if any(getattr(args, name) is None for name in ('run_dir', 'store_dir', 'parent_receipt', 'folds',
                                                  'image', 'native_auth_dir', 'service_account')):
        parser.error('run-dir, store-dir, parent-receipt, folds, image, native-auth-dir and service-account are required')
    receipt = prepare(args)
    if not args.prepare_only:
        auth = json.loads(receipt.read_text())['native_auth']
        subprocess.run(['tmux', 'new-session', '-d', '-s', receipt.parent.name, '-c', str(receipt.parent),
                        shlex.join(watcher_command(receipt, auth))], check=True)
    print(json.dumps({'status': 'PREPARED_ONLY' if args.prepare_only else 'DETACHED_CONTROLLER_STARTED',
                      'receipt': str(receipt)}))


if __name__ == '__main__':
    main()
