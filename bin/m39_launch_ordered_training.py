#!/usr/bin/env python3
"""Seal and detach the bounded local ordered-model resource profile.

This launcher creates no cloud resources and never copies the genomic store.
Only committed code/configuration is snapshotted. No ancestry truth is an input.
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
import subprocess


PROFILE = 'conf/m39_ordered_training_profile.json'
CONFIG = 'conf/m39_ordered_training_local.config'
WORKFLOW = 'workflows/m39_ordered_training_profile.nf'
SOURCES = (PROFILE, CONFIG, WORKFLOW, 'modules/39_ORDERED_TRAINING_PROFILE.nf',
           'bin/m39_launch_ordered_training.py', 'bin/m39_profile_ordered_training.py',
           'bin/m39_ordered_models.py', 'bin/m39_ordered_batches.py',
           'bin/m39_ordered_context.py', 'bin/m39_carrier_context.py',
           'bin/m34_prepare_panel_factors.py', 'bin/m34_generate_mosaics.py',
           'bin/m33_safe_bridge_core.py')
LANE = 'ordered-training'
_SAFE_CASE = re.compile(r'[a-z0-9][a-z0-9_-]{0,79}\Z')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(data)


def _private_path(path: Path, private: Path, *, directory: bool) -> Path:
    resolved = path.resolve(strict=True)
    if (not resolved.is_relative_to(private) or resolved == private or
            not re.fullmatch(r'[A-Za-z0-9_./-]+', str(resolved))):
        raise ValueError('Input/output paths must remain within private project runs')
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise ValueError('Expected a private directory' if directory else 'Expected a private file')
    return resolved


def _case_ids(profile: dict) -> list[str]:
    cases = profile.get('cases') if isinstance(profile, dict) else None
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError('Expected a bounded nonempty cases array (at most 32)')
    ids = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not _SAFE_CASE.fullmatch(case['id']):
            raise ValueError('Unsafe case identifier')
        if (case.get('family') not in ('cnn', 'attention') or
                case.get('size') not in ('small', 'medium') or
                type(case.get('batch_size')) is not int or case['batch_size'] not in (1, 2) or
                case.get('window') not in ('median', 'maximum') or
                case.get('arm') not in ('common', 'real') or
                type(case.get('core_sites')) is not int or case['core_sites'] != 256):
            raise ValueError('Case exceeds the authorized technical profile')
        ids.append(case['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate case identifiers')
    return ids


def prepare(args: argparse.Namespace, repo: Path | None = None) -> tuple[list[str], Path, str]:
    """Validate and seal; does not start Nextflow, Docker or tmux."""
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    private = repo / '.claude' / 'runs'
    run_dir = _private_path(args.run_dir, private, directory=True)
    if args.run_dir.is_symlink() or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{2,69}', run_dir.name):
        raise ValueError('Use an existing private run directory with a safe identifier')
    store = _private_path(args.store_dir, private, directory=True)
    parent = _private_path(args.parent_receipt, private, directory=False)
    folds = _private_path(args.folds, private, directory=False)
    if store.name != 'radius_1cm' or store.parent.name != 'people_48':
        raise ValueError('This profile requires the frozen people_48/radius_1cm store')
    if parent != store.parent.parent / 'receipt.json':
        raise ValueError('Parent receipt must belong to the selected ordered store')
    manifest = _private_path(store / 'manifest.json', private, directory=False)
    if folds.suffix != '.npz':
        raise ValueError('Historical folds must be an NPZ file')
    if args.profile_config.resolve(strict=True) != repo / PROFILE:
        raise ValueError('Use the versioned conf/m39_ordered_training_profile.json')
    # Reject reuse even after a failed/interrupted launch: receipts are append-only evidence.
    reserved = ['frozen-source', f'{LANE}-output', f'{LANE}-work', f'{LANE}.launch.json',
                f'{LANE}.nextflow.log', f'{LANE}.trace.tsv', f'{LANE}.report.html', f'{LANE}.timeline.html']
    if any(os.path.lexists(run_dir / name) for name in reserved):
        raise ValueError('Immutable run artifacts already exist; choose a new run directory')
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                     cwd=repo, text=True)
    if status.strip():
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
            committed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=repo,
                                                 stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f'Commit required source before launching: {relative}') from exc
        if source.read_bytes() != committed:
            raise ValueError(f'Source differs from source commit: {relative}')
        blobs[relative] = committed
    profile = json.loads(blobs[PROFILE])
    cases = _case_ids(profile)
    input_records = {name: {'path': str(path), 'sha256': sha256(path)}
                     for name, path in [('parent_receipt', parent), ('store_manifest', manifest), ('folds', folds)]}
    for name, record in input_records.items():
        if profile.get(f'{name}_sha256') != record['sha256']:
            raise ValueError(f'{name} differs from the frozen profile input seal')
    container_user = os.environ.get('M39_CONTAINER_USER', '1017:1020')
    if not re.fullmatch(r'[0-9]{1,10}:[0-9]{1,10}', container_user):
        raise ValueError('M39_CONTAINER_USER must be a numeric uid:gid')
    profile_digest = hashlib.sha256(blobs[PROFILE]).hexdigest()
    frozen = run_dir / 'frozen-source'
    output = run_dir / f'{LANE}-output'
    command = ['env', 'NXF_OFFLINE=true', 'NXF_VER=26.04.6', 'nextflow',
               '-C', str(frozen / CONFIG), '-log', str(run_dir / f'{LANE}.nextflow.log'),
               'run', str(frozen / WORKFLOW), '-work-dir', str(run_dir / f'{LANE}-work'),
               '-with-trace', str(run_dir / f'{LANE}.trace.tsv'),
               '-with-report', str(run_dir / f'{LANE}.report.html'),
               '-with-timeline', str(run_dir / f'{LANE}.timeline.html')]
    values = {'m39_store_dir': store, 'm39_parent_receipt': parent, 'm39_folds': folds,
              'm39_profile_config': frozen / PROFILE, 'm39_profile_sha256': profile_digest,
              'm39_source_commit': commit, 'm39_output_dir': output,
              'm39_container_user': container_user}
    for key, value in values.items():
        command.extend((f'--{key}', str(value)))
    # Short deterministic session name; periods in user run names cannot alias tmux targets.
    session = 'm39-training-' + hashlib.sha256(str(run_dir).encode()).hexdigest()[:16]
    receipt = {'schema_version': 'm39-ordered-training-launch-v1',
               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status': 'launch_requested_not_completion', 'scope': 'technical_synthetic_labels_no_accuracy',
               'argv': command, 'tmux_session': session, 'source_commit': commit,
               'source_sha256': {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
               'profile_sha256': profile_digest, 'case_ids': cases,
               'inputs': input_records, 'container_user': container_user,
               'store_dir': str(store), 'staging': 'small_inputs_copy_exact_store_readonly_bind_no_store_copy',
               'store_container_path': '/m39-ordered-store',
               'frozen_source_dir': str(frozen), 'labels': 'artificial_six_state_for_resources_only',
               'full_epoch': False, 'new_cloud_instances': 0, 'gpu': False,
               'resources': {'cpus_per_task': 2, 'memory_gib_per_task': 8, 'max_forks': 2,
                             'cpus_total': 4, 'memory_gib_total': 16, 'time_minutes_per_task': 20}}
    frozen.mkdir(mode=0o700)
    for relative, blob in blobs.items():
        destination = frozen / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_exclusive(destination, blob)
    receipt_path = run_dir / f'{LANE}.launch.json'
    write_exclusive(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    return command, receipt_path, session


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'profile-config', 'store-dir', 'parent-receipt', 'folds'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        command, receipt, session = prepare(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    # tmux, not the SSH/IDE shell, owns the controller. A failed start retains its receipt.
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(receipt.parent),
                    shlex.join(command)], check=True)
    print(json.dumps({'status': 'tmux_started', 'receipt': str(receipt), 'session': session}))


if __name__ == '__main__':
    main()
