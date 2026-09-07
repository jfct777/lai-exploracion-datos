#!/usr/bin/env python3
"""Seal and supervise private, local-only ordered throughput through Nextflow.

The historical profiler stays immutable. This controller reuses its pure path,
hash and exclusive-write helpers, adding an independent detached watchdog.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

from m39_launch_ordered_training import _private_path, sha256, write_exclusive


PROFILE = 'conf/m39_ordered_throughput_profile.json'
CONFIG = 'conf/m39_ordered_throughput_local.config'
WORKFLOW = 'workflows/m39_ordered_throughput.nf'
RUNTIME_SOURCES = (
    'm39_profile_ordered_throughput.py', 'm39_throughput_sampling.py',
    'm39_profile_device.py',
    'm39_profile_ordered_training.py', 'm39_ordered_models.py',
    'm39_ordered_batches.py', 'm39_ordered_context.py', 'm39_carrier_context.py',
    'm34_prepare_panel_factors.py', 'm34_generate_mosaics.py', 'm33_safe_bridge_core.py',
)
SOURCES = (PROFILE, CONFIG, WORKFLOW, 'modules/39_ORDERED_THROUGHPUT.nf',
           'bin/m39_launch_ordered_throughput.py', 'bin/m39_launch_ordered_training.py',
           *(f'bin/{name}' for name in RUNTIME_SOURCES))
LANE = 'ordered-throughput'
CONTROLLER_TIMEOUT_SECONDS = 1800
TASK_TIMEOUT_SECONDS = 900
EXPECTED_CASES = tuple(f'{family}-small-{ordering}-b{batch}-real'
                      for family in ('cnn', 'attention')
                      for ordering, batch in (('grouped', 1), ('grouped', 2), ('random', 2)))


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def _case_ids(profile: dict) -> list[str]:
    cases = profile.get('cases') if isinstance(profile, dict) else None
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError('Exactly six authorized throughput cases are required')
    ids = []
    for case in cases:
        if not isinstance(case, dict) or case.get('id') not in EXPECTED_CASES:
            raise ValueError('Unexpected or unsafe throughput case identifier')
        if (set(case) != {'id', 'family', 'policy', 'batch_size', 'arm'} or
                case.get('family') not in ('cnn', 'attention') or
                type(case.get('batch_size')) is not int or case['batch_size'] not in (1, 2) or
                case.get('policy') not in ('grouped', 'random') or case.get('arm') != 'real' or
                case['id'] != f"{case['family']}-small-{case['policy']}-b{case['batch_size']}-real"):
            raise ValueError('Case exceeds the authorized small CPU throughput profile')
        ids.append(case['id'])
    if set(ids) != set(EXPECTED_CASES):
        raise ValueError('Duplicate or missing throughput case identifiers')
    if profile.get('core_sites') != 256 or profile.get('max_seconds') != TASK_TIMEOUT_SECONDS:
        raise ValueError('Unexpected throughput geometry or task time limit')
    return ids


def prepare(args: argparse.Namespace, repo: Path | None = None) -> tuple[list[str], Path, str]:
    """Validate and freeze without starting tmux, Docker or Nextflow."""
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    private = repo / '.claude/runs'
    run = _private_path(args.run_dir, private, directory=True)
    if args.run_dir.is_symlink() or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{2,69}', run.name):
        raise ValueError('Use an existing private run directory with a safe identifier')
    store = _private_path(args.store_dir, private, directory=True)
    parent = _private_path(args.parent_receipt, private, directory=False)
    folds = _private_path(args.folds, private, directory=False)
    if store.name != 'radius_1cm' or store.parent.name != 'people_48':
        raise ValueError('Requires the frozen people_48/radius_1cm store')
    if parent != store.parent.parent / 'receipt.json':
        raise ValueError('Parent receipt must belong to the selected ordered store')
    manifest = _private_path(store / 'manifest.json', private, directory=False)
    if folds.suffix != '.npz':
        raise ValueError('Historical folds must be an NPZ file')
    if args.profile_config.resolve(strict=True) != repo / PROFILE:
        raise ValueError(f'Use the versioned {PROFILE}')
    reserved = ['frozen-source', *(f'{LANE}{suffix}' for suffix in (
        '-output', '-work', '.launch.json', '.started.json', '.completion.json',
        '.controller.log', '.nextflow.log', '.trace.tsv', '.report.html', '.timeline.html'))]
    if any(os.path.lexists(run / name) for name in reserved):
        raise ValueError('Immutable run artifacts already exist; choose a new run directory')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=repo, text=True).strip():
        raise ValueError('Commit tracked changes before launching')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Could not resolve an immutable source commit')
    blobs = {}
    for relative in SOURCES:
        source = repo / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Missing regular versioned source: {relative}')
        try:
            blob = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=repo,
                                           stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f'Commit required source before launching: {relative}') from exc
        if source.read_bytes() != blob:
            raise ValueError(f'Source differs from source commit: {relative}')
        blobs[relative] = blob
    profile = json.loads(blobs[PROFILE])
    cases = _case_ids(profile)
    inputs = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in
              [('parent_receipt', parent), ('store_manifest', manifest), ('folds', folds)]}
    for name, record in inputs.items():
        if profile.get(f'{name}_sha256') != record['sha256']:
            raise ValueError(f'{name} differs from the frozen profile input seal')
    user = os.environ.get('M39_CONTAINER_USER', '1017:1020')
    if not re.fullmatch(r'[0-9]{1,10}:[0-9]{1,10}', user):
        raise ValueError('M39_CONTAINER_USER must be a numeric uid:gid')
    frozen = run / 'frozen-source'
    token = hashlib.sha256(str(run).encode()).hexdigest()[:16]
    session = 'm39-throughput-' + token
    digest = hashlib.sha256(blobs[PROFILE]).hexdigest()
    command = ['env', 'NXF_OFFLINE=true', 'NXF_VER=26.04.6', 'nextflow', '-C', str(frozen / CONFIG),
               '-log', str(run / f'{LANE}.nextflow.log'), 'run', str(frozen / WORKFLOW),
               '-work-dir', str(run / f'{LANE}-work'), '-with-trace', str(run / f'{LANE}.trace.tsv'),
               '-with-report', str(run / f'{LANE}.report.html'),
               '-with-timeline', str(run / f'{LANE}.timeline.html')]
    values = {'m39_store_dir': store, 'm39_parent_receipt': parent, 'm39_folds': folds,
              'm39_profile_config': frozen / PROFILE, 'm39_profile_sha256': digest,
              'm39_source_commit': commit, 'm39_output_dir': run / f'{LANE}-output',
              'm39_container_user': user, 'm39_run_token': token}
    for key, value in values.items():
        command.extend((f'--{key}', str(value)))
    receipt_path = run / f'{LANE}.launch.json'
    watch_argv = [sys.executable, str(frozen / 'bin/m39_launch_ordered_throughput.py'),
                  '--watch-receipt', str(receipt_path)]
    receipt = {'schema_version': 'm39-ordered-throughput-launch-v1', 'utc': utc(),
        'status': 'launch_requested_not_completion', 'scope': 'technical_synthetic_labels_no_accuracy',
        'argv': command, 'watchdog_argv': watch_argv, 'tmux_session': session, 'run_token': token,
        'source_commit': commit, 'source_sha256': {name: hashlib.sha256(blob).hexdigest()
                                                for name, blob in blobs.items()},
        'profile_sha256': digest, 'case_ids': cases, 'inputs': inputs, 'container_user': user,
        'store_dir': str(store), 'staging': 'small_inputs_copy_exact_store_readonly_bind_no_store_copy',
        'store_container_path': '/m39-ordered-store', 'frozen_source_dir': str(frozen),
        'labels': 'artificial_six_state_for_resources_only', 'full_epoch': False,
        'new_cloud_instances': 0, 'gpu': False,
        'resources': {'cpus_per_task': 2, 'memory_gib_per_task': 8, 'max_forks': 2,
                      'cpus_total': 4, 'memory_gib_total': 16, 'rss_stop_gib': 6.4,
                      'time_seconds_per_task': TASK_TIMEOUT_SECONDS,
                      'controller_timeout_seconds': CONTROLLER_TIMEOUT_SECONDS,
                      'cleanup_grace_seconds': 90}}
    frozen.mkdir(mode=0o700)
    for relative, blob in blobs.items():
        destination = frozen / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_exclusive(destination, blob)
    write_json(receipt_path, receipt)
    return watch_argv, receipt_path, session


def cleanup_containers(token: str) -> dict:
    """Stop only containers owned by this exact sealed run; never list credentials."""
    if not re.fullmatch(r'[0-9a-f]{16}', token):
        raise ValueError('Invalid cleanup run token')
    result = {'stopped': [], 'errors': []}
    query = ['docker', 'ps', '-q', '--no-trunc', '--filter', f'label=dnabr.m39_run={token}']
    def active_ids() -> list[str]:
        found = subprocess.run(query, check=True, capture_output=True, text=True, timeout=10)
        ids = found.stdout.split()
        if any(not re.fullmatch(r'[0-9a-f]{64}', cid) for cid in ids):
            raise ValueError('Unexpected scoped Docker inventory; refusing broad cleanup')
        return ids
    try:
        ids = active_ids()
        if ids:
            try:
                subprocess.run(['docker', 'stop', '--time', '10', *ids], check=True,
                               capture_output=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                remaining = active_ids()
                if remaining:
                    subprocess.run(['docker', 'kill', *remaining], check=True,
                                   capture_output=True, timeout=10)
            if active_ids():
                raise ValueError('Scoped Docker workers remain after cleanup')
            result['stopped'] = ids
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result['errors'].append(type(exc).__name__)
    return result


def stop_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # A terminated leader does not prove its child process group is gone.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=5)


def watch_receipt(path: Path) -> int:
    """Own the controller until completion/timeout, independent of chat and SSH."""
    path = path.resolve(strict=True)
    run = path.parent
    receipt = json.loads(path.read_text())
    if path.name != f'{LANE}.launch.json' or receipt.get('schema_version') != 'm39-ordered-throughput-launch-v1':
        raise ValueError('Expected a sealed throughput launch receipt')
    if not re.fullmatch(r'[0-9a-f]{16}', receipt.get('run_token', '')):
        raise ValueError('Invalid sealed run token')
    frozen = run / 'frozen-source'
    start = time.monotonic()
    result = {'schema_version': 'm39-ordered-throughput-completion-v1', 'started_utc': utc(),
              'status': 'FAILED', 'source_commit': receipt['source_commit'],
              'profile_sha256': receipt['profile_sha256'], 'run_token': receipt['run_token'],
              'expected_cases': receipt['case_ids'], 'completed_cases': [], 'exit_code': 1,
              'full_epoch': False, 'accuracy_evaluated': False, 'cleanup': {}}
    process = None
    handlers = {}
    def interrupted(signum, frame):
        raise InterruptedError(f'Controller received signal {signum}')
    try:
        for name in SOURCES:
            if sha256(frozen / name) != receipt['source_sha256'].get(name):
                raise ValueError('Frozen source differs from launch receipt')
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, interrupted)
        write_json(run / f'{LANE}.started.json', {'utc': utc(), 'pid': os.getpid(),
                   'launch_sha256': sha256(path), 'timeout_seconds': CONTROLLER_TIMEOUT_SECONDS})
        descriptor = os.open(run / f'{LANE}.controller.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as log:
            process = subprocess.Popen(receipt['argv'], cwd=run, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result['exit_code'] = process.wait(timeout=CONTROLLER_TIMEOUT_SECONDS)
                result['status'] = 'COMPLETED' if result['exit_code'] == 0 else 'FAILED'
            except subprocess.TimeoutExpired:
                result.update(status='TIMEOUT', exit_code=124)
        for case in receipt['case_ids']:
            directory = run / f'{LANE}-output' / case
            if (directory / 'profile.json').is_file() and (directory / 'sampling-manifest.json').is_file():
                report = json.loads((directory / 'profile.json').read_text())
                if report.get('decision') == 'PASS_ORDERED_THROUGHPUT_TECHNICAL_ONLY':
                    result['completed_cases'].append(case)
        if result['status'] == 'COMPLETED' and set(result['completed_cases']) != set(receipt['case_ids']):
            result.update(status='INCOMPLETE', exit_code=1)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        result.update(status='INTERRUPTED' if isinstance(exc, InterruptedError) else 'FAILED',
                      exit_code=130 if isinstance(exc, InterruptedError) else 1,
                      failure_type=type(exc).__name__)
    finally:
        # Repeated terminal signals must not interrupt the bounded cleanup/receipt.
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            try:
                stop_process_group(process)
            except (OSError, subprocess.SubprocessError) as exc:
                result['process_cleanup_error'] = type(exc).__name__
        try:
            result['cleanup'] = cleanup_containers(receipt['run_token'])
        except Exception as exc:
            result['cleanup'] = {'stopped': [], 'errors': [type(exc).__name__]}
        if result['cleanup']['errors'] and result['status'] == 'COMPLETED':
            result.update(status='CLEANUP_UNVERIFIED', exit_code=1)
        result['finished_utc'] = utc()
        result['elapsed_seconds'] = time.monotonic() - start
        result['missing_cases'] = [case for case in receipt['case_ids'] if case not in result['completed_cases']]
        write_json(run / f'{LANE}.completion.json', result)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return result['exit_code'] if result['exit_code'] >= 0 else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-receipt', type=Path, help=argparse.SUPPRESS)
    for name in ('run-dir', 'profile-config', 'store-dir', 'parent-receipt', 'folds'):
        parser.add_argument(f'--{name}', type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.watch_receipt:
        raise SystemExit(watch_receipt(args.watch_receipt))
    if any(getattr(args, key) is None for key in ('run_dir', 'profile_config', 'store_dir', 'parent_receipt', 'folds')):
        parser.error('run-dir, profile-config, store-dir, parent-receipt and folds are required')
    try:
        command, receipt, session = prepare(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    try:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(receipt.parent),
                        shlex.join(command)], check=True)
    except (OSError, subprocess.CalledProcessError):
        write_json(receipt.parent / f'{LANE}.completion.json',
                   {'schema_version': 'm39-ordered-throughput-completion-v1',
                    'utc': utc(), 'status': 'LAUNCH_FAILED', 'exit_code': 1})
        raise
    print(json.dumps({'status': 'tmux_started', 'receipt': str(receipt), 'session': session}))


if __name__ == '__main__':
    main()
