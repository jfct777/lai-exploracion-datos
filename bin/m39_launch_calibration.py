#!/usr/bin/env python3
"""Freeze and detach a bounded probability-only M39 calibration workflow."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time


SOURCES = (
    'bin/m39_calibration.py', 'bin/m39_launch_calibration.py',
    'workflows/m39_calibration.nf', 'conf/m39_calibration_local.config',
)


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def frozen_copy(source: Path, target: Path, expected: str) -> None:
    if sha256(source) != expected:
        raise ValueError(f'Input hash differs: {source.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f'Duplicate frozen input name: {target.name}')
    shutil.copyfile(source, target)
    if sha256(target) != expected:
        raise ValueError(f'Copy verification failed: {target.name}')
    target.chmod(0o444)


def supervise(run: Path) -> None:
    request = json.loads((run / 'launch.json').read_text())
    for name, expected in request['frozen_sha256'].items():
        if sha256(run / name) != expected:
            raise ValueError(f'Frozen payload changed: {name}')
    start = time.monotonic()
    with (run / 'controller.log').open('x') as log:
        child = subprocess.Popen(request['argv'], cwd=run, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write_json(run / 'running.json', {'pid': child.pid,
                   'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        try:
            code = child.wait(timeout=request['max_minutes'] * 60)
            reason = 'nextflow_exited'
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            reason = 'wall_time_budget_reached'
            try:
                code = child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                code = None
                reason = 'controller_did_not_exit_check_its_tasks'
    write_json(run / 'completion.json', {
        'exit_code': code, 'reason': reason, 'seconds': time.monotonic() - start,
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'new_instances': 0,
        'vm_must_remain_running': True,
    })


def prepare(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    if (not run.is_relative_to(repo / '.claude' / 'runs') or run.exists()
            or not re.fullmatch('[a-z0-9._-]{3,70}', run.name)):
        raise ValueError('Use a fresh named directory under .claude/runs')
    if not 0 < args.max_minutes <= 15:
        raise ValueError('Wall-time budget must be in (0,15] minutes')
    if any(p is None for p in (args.development, args.score, args.plan, args.comparators)):
        raise ValueError('Development, score, plan, comparators are required')
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                      cwd=repo, text=True)
    if changed.strip():
        raise ValueError('Commit tracked changes before freezing the workflow')
    subprocess.run(['git', 'ls-files', '--error-unmatch', *SOURCES], cwd=repo,
                   check=True, stdout=subprocess.DEVNULL)
    plan = json.loads(args.plan.read_text())
    if sha256(args.comparators) != plan['comparators_manifest_sha256']:
        raise ValueError('Comparator manifest differs from plan')
    manifest = json.loads(args.comparators.read_text())
    run.mkdir(mode=0o700, parents=True)
    original = {str(p.resolve()): sha256(p) for p in
                (args.development, args.score, args.plan, args.comparators)}
    for name in SOURCES:
        frozen_copy(repo / name, run / 'source' / name, sha256(repo / name))
    for role in ('development', 'score'):
        frozen_copy(getattr(args, role), run / 'inputs' / f'{role}.npz',
                    plan[f'{role}_sha256'])
    # Flat, self-contained comparator inputs are only staged into the scoring task.
    entries = [manifest['selection'], manifest['score_receipt'], *manifest['comparators']]
    for entry in entries:
        source = (args.comparators.parent / entry['path']).resolve()
        original[str(source)] = entry['sha256']
        frozen_copy(source, run / 'comparators' / source.name, entry['sha256'])
        entry['path'] = source.name
    write_json(run / 'comparators' / 'manifest.json', manifest)
    plan['comparators_manifest_sha256'] = sha256(run / 'comparators' / 'manifest.json')
    write_json(run / 'plan.json', plan)
    frozen = {str(p.relative_to(run)): sha256(p) for p in run.rglob('*') if p.is_file()}
    argv = ['env', 'NXF_OFFLINE=true', 'NXF_VER=26.04.6', 'nextflow',
            '-C', str(run / 'source/conf/m39_calibration_local.config'),
            '-log', str(run / 'nextflow.log'), 'run', str(run / 'source/workflows/m39_calibration.nf'),
            '-work-dir', str(run / 'work'), '-with-trace', str(run / 'trace.tsv'),
            '-with-report', str(run / 'report.html'), '-with-timeline', str(run / 'timeline.html'),
            '--m39cal_output_dir', str(run / 'output'), '--m39cal_plan', str(run / 'plan.json'),
            '--m39cal_development', str(run / 'inputs/development.npz'),
            '--m39cal_score', str(run / 'inputs/score.npz'),
            '--m39cal_comparators_dir', str(run / 'comparators')]
    write_json(run / 'launch.json', {
        'schema_version': 'm39-calibration-launch-v1', 'argv': argv,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'frozen_sha256': frozen, 'original_sha256': original,
        'max_minutes': args.max_minutes, 'new_instances': 0, 'new_gpu': False,
        'scope': 'historical_R0_exploratory_post_hoc',
        'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--development', type=Path)
    parser.add_argument('--score', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--comparators', type=Path)
    parser.add_argument('--max-minutes', type=float, default=15.)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    if args.supervise:
        supervise(args.run_dir.resolve())
        return
    run = prepare(args)
    if not args.prepare_only:
        command = shlex.join(['python3', str(run / 'source/bin/m39_launch_calibration.py'),
                              '--run-dir', str(run), '--supervise'])
        subprocess.run(['tmux', 'new-session', '-d', '-s', run.name, command], check=True)
    print(json.dumps({'run_dir': str(run), 'detached': not args.prepare_only}))


if __name__ == '__main__':
    main()
