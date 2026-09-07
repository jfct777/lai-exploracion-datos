#!/usr/bin/env python3
"""Freeze development-only training inputs and detach bounded Nextflow jobs."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from m39_gpu_launch import native_auth, native_env_prefix, validate_target, watcher_command, write_json
from m39_launch_ordered_training import _private_path, sha256, write_exclusive
from m39_ordered_gpu_manifest import development_inventory, load_plan, require, store_inventory

CONFIG = 'conf/m39_ordered_gpu_google_batch.config'
WORKFLOW = 'workflows/m39_ordered_gpu_training.nf'
RUNTIME = ('m39_ordered_training.py', 'm39_ordered_training_data.py', 'm39_ordered_models.py',
    'm39_ordered_batches.py', 'm39_ordered_context.py', 'm39_profile_device.py',
    'm39_anchor_screen.py', 'm39_carrier_models.py', 'm33_safe_bridge_core.py',
    'm34_prepare_panel_factors.py', 'm34_generate_mosaics.py', 'm39_carrier_context.py',
    'm39_ordered_gpu_manifest.py', 'm39_ordered_gpu_worker.py', 'm39_gpu_serial_profile.py')
SOURCES = (CONFIG, WORKFLOW, 'modules/39_ORDERED_GPU_TRAINING.nf',
    'bin/m39_ordered_gpu_launch.py', 'bin/m39_gpu_launch.py',
    'bin/m39_launch_ordered_training.py', 'bin/m39_launch_ordered_throughput.py',
    *(f'bin/{name}' for name in RUNTIME))


def prepare(args: argparse.Namespace, repo: Path | None = None) -> Path:
    repo = (repo or Path(__file__).resolve().parents[1]).resolve()
    private = repo / '.claude/runs'
    auth = native_auth(args.native_auth_dir, args.service_account, repo)
    run = _private_path(args.run_dir, private, directory=True)
    cloud = validate_target(run.name, args.image)
    require(not any((run / name).exists() for name in ('frozen-source', 'training-plan',
            'gpu.launch.json', 'gpu.completion.json', 'source-seal.json')), 'use a new immutable run')
    plan_path = _private_path(args.plan, private, directory=False)
    require(not plan_path.is_symlink(), 'plan must not be a symlink')
    plan = load_plan(plan_path)
    plan_hash = sha256(plan_path)
    inputs = {key: _private_path(getattr(args, key), private, directory=key != 'development')
              for key in ('train_store', 'select_store', 'development')}
    require(inputs['train_store'] != inputs['select_store'], 'TRAIN and SELECT stores must differ')
    input_bytes = sum(store_inventory(inputs[role + '_store'], plan['inputs'][role + '_manifest_sha256'])
                      for role in ('train', 'select'))
    input_bytes += development_inventory(inputs['development'], plan['inputs']['development_sha256'])
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=repo, text=True).strip(), 'commit tracked source before launch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    blobs = {}
    for name in SOURCES:
        source = repo / name
        blob = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=repo)
        require(source.is_file() and not source.is_symlink() and source.read_bytes() == blob,
                f'source differs from committed file: {name}')
        blobs[name] = blob
    frozen = run / 'frozen-source'
    frozen.mkdir(mode=0o700)
    for name, blob in blobs.items():
        target = frozen / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_exclusive(target, blob)
    config_dir = run / 'training-plan'
    config_dir.mkdir(mode=0o700)
    write_exclusive(config_dir / 'plan.json', plan_path.read_bytes())
    for group in plan['groups']:
        for spec in group['configs']:
            write_exclusive(config_dir / spec['file'], (plan_path.parent / spec['file']).read_bytes())
    load_plan(config_dir / 'plan.json')
    require(sha256(config_dir / 'plan.json') == plan_hash, 'plan changed during sealing')
    seal = {'schema_version': 'm39-gpu-source-seal-v1', 'source_commit': commit,
            'profile_sha256': plan_hash,
            'source_sha256': {name: hashlib.sha256(blobs['bin/' + name]).hexdigest() for name in RUNTIME}}
    write_json(run / 'source-seal.json', seal)
    command = [*native_env_prefix(auth), 'NXF_VER=26.04.6', 'NXF_OFFLINE=true',
        f'M39_GPU_RUN_ID={run.name}', f'M39_GPU_IMAGE={args.image}',
        f"M39_TRAIN_TASK_SECONDS={plan['resources']['task_seconds']}",
        f"M39_TRAIN_MAX_WORKERS={plan['resources']['max_workers']}",
        'nextflow', '-C', str(frozen / CONFIG), '-log', str(run / 'gpu.nextflow.log'),
        'run', str(frozen / WORKFLOW), '-work-dir', cloud + '/work',
        '-with-trace', str(run / 'gpu.trace.tsv'), '-with-report', str(run / 'gpu.report.html'),
        '-with-timeline', str(run / 'gpu.timeline.html')]
    values = {**inputs, 'training_plan': config_dir / 'plan.json', 'source_seal': run / 'source-seal.json',
              'source_commit': commit, 'plan_sha256': plan_hash}
    for key, value in values.items():
        command.extend(('--m39_' + key, str(value)))
    receipt = {'schema_version': 'm39-gpu-launch-v1', 'source_commit': commit,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'run_id': run.name, 'image': args.image, 'argv': command, 'cloud_prefix': cloud,
        'native_auth': auth, 'source_sha256': {key: hashlib.sha256(value).hexdigest() for key, value in blobs.items()},
        'source_seal_sha256': sha256(run / 'source-seal.json'), 'plan_sha256': plan_hash,
        'group_ids': [group['id'] for group in plan['groups']], 'input_bytes': input_bytes,
        'scope': plan['scope'], 'stage': plan['stage'], 'process_name': 'M39_ORDERED_GPU_TRAINING',
        'resources': {'max_jobs': len(plan['groups']), 'new_batch_workers': len(plan['groups']),
            'max_concurrent_workers': plan['resources']['max_workers'], 'machine_type': 'g2-standard-8',
            'gpu': 'NVIDIA L4', 'gpu_count_per_worker': 1, 'cpus_per_task': 2, 'memory_gib_per_task': 8,
            'disk_gb': 50, 'task_timeout_seconds': plan['resources']['task_seconds'],
            'controller_timeout_seconds': plan['resources']['controller_seconds'], 'max_retries': 0,
            'provisioning_time_not_covered_by_task_timeout': True},
        'outputs': cloud + '/outputs', 'labels': {'team': 'frank', 'm39_run': run.name},
        'inputs': {key: str(value) for key, value in inputs.items()}, 'input_sha256': plan['inputs'],
        'SCORE_staged': False, 'frozen_training_plan': str(config_dir / 'plan.json')}
    receipt['frozen_artifact_sha256'] = {str(path.relative_to(run)): sha256(path) for path in config_dir.iterdir()}
    write_json(run / 'gpu.launch.json', receipt)
    return run / 'gpu.launch.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'train-store', 'select-store', 'development', 'plan', 'native-auth-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--service-account', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    receipt = prepare(args)
    if not args.prepare_only:
        auth = json.loads(receipt.read_text())['native_auth']
        subprocess.run(['tmux', 'new-session', '-d', '-s', receipt.parent.name, '-c', str(receipt.parent),
                        shlex.join(watcher_command(receipt, auth))], check=True)
    print(json.dumps({'status': 'PREPARED_ONLY' if args.prepare_only else 'DETACHED_CONTROLLER_STARTED',
                      'receipt': str(receipt)}))


if __name__ == '__main__':
    main()
