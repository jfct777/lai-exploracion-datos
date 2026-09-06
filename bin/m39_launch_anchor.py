#!/usr/bin/env python3
"""Detach an immutable M39 workflow snapshot on the existing development VM."""
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def supervise(run: Path):
    request = json.loads((run/'launch.json').read_text())
    for name, expected in request['snapshot_sha256'].items():
        if digest(run/'source'/name) != expected:
            raise ValueError('Frozen workflow source changed')
    if digest(run/'plan.json') != request['plan_sha256']:
        raise ValueError('Frozen plan changed')
    for name, expected in request['input_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError('Scientific input changed before launch')
    start = time.monotonic()
    with (run/'controller.log').open('x') as log:
        child = subprocess.Popen(request['argv'], cwd=run, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write(run/'running.json', {'pid': child.pid, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        try:
            code = child.wait(timeout=request['max_hours']*3600)
            reason = 'nextflow_exited'
        except subprocess.TimeoutExpired:
            # Only this workflow's controller group is signalled; Nextflow handles its tasks.
            os.killpg(child.pid, signal.SIGTERM)
            reason = 'wall_time_budget_reached'
            try:
                code = child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                code = None
                reason = 'controller_did_not_exit_after_termination_request_check_tasks'
    write(run/'completion.json', {'exit_code': code, 'reason': reason,
                                  'seconds': time.monotonic()-start,
                                  'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                  'new_instances': 0, 'vm_must_remain_running': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--binding-dir', type=Path)
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--max-hours', type=float, default=6.)
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.supervise:
        supervise(run)
        return
    repo = Path(__file__).resolve().parents[1]
    if not run.is_relative_to(repo/'.claude'/'runs') or run.exists():
        parser.error('Use a fresh directory under the private project runs')
    if not re.fullmatch('[a-z0-9._-]{3,70}', run.name) or not 0 < args.max_hours <= 8:
        parser.error('Invalid run id or wall-time budget')
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                     cwd=repo, text=True)
    if status.strip():
        parser.error('Commit changes before freezing the workflow')
    for path in (args.plan, args.binding_dir, args.feature_dir):
        if path is None or not path.resolve().is_relative_to(repo/'.claude'/'runs') or not path.exists():
            parser.error('Plan and data must exist in the private project runs')
    plan = json.loads(args.plan.read_text())
    if len(plan['cases']) != (1 if args.profile_only else 12):
        parser.error('Wrong plan for this execution mode')
    run.mkdir(mode=0o700)
    names = ['bin/m39_anchor_screen.py', 'bin/m39_carrier_models.py', 'bin/m39_anchor_plan.py',
             'bin/m39_anchor_score.py', 'bin/m39_launch_anchor.py',
             'workflows/m39_anchor_screen.nf', 'conf/m39_anchor_local.config']
    subprocess.run(['git', 'ls-files', '--error-unmatch', *names], cwd=repo,
                   check=True, stdout=subprocess.DEVNULL)
    hashes = {}
    for name in names:
        target = run/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, target)
        target.chmod(0o444)
        hashes[name] = digest(target)
    shutil.copy2(args.plan, run/'plan.json')
    argv = ['env', 'NXF_OFFLINE=true', 'NXF_VER=26.04.6', 'nextflow',
            '-C', str(run/'source/conf/m39_anchor_local.config'),
            '-log', str(run/'nextflow.log'), 'run', str(run/'source/workflows/m39_anchor_screen.nf'),
            '-work-dir', str(run/'work'), '-with-trace', str(run/'trace.tsv'),
            '-with-report', str(run/'report.html'), '-with-timeline', str(run/'timeline.html'),
            '--m39_output_dir', str(run/'output'), '--m39_plan', str(run/'plan.json'),
            '--m39_binding_dir', str(args.binding_dir.resolve()),
            '--m39_feature_dir', str(args.feature_dir.resolve()),
            '--m39_profile_only', str(args.profile_only).lower()]
    session = run.name
    write(run/'launch.json', {'schema_version': 'm39-anchor-launch-v1', 'argv': argv,
                             'snapshot_sha256': hashes, 'plan_sha256': digest(run/'plan.json'),
                             'input_sha256': {str(p.resolve()): digest(p) for p in
                                 [args.binding_dir/'development.npz', args.binding_dir/'receipt.json'] +
                                 [args.feature_dir/f'radius_{r}cm/features.npz' for r in (.05, .2, .5)]},
                             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                             'tmux_session': session, 'max_hours': args.max_hours,
                             'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                             'scope': 'exploratory_anchor_profile' if args.profile_only else 'exploratory_anchor_screen'})
    command = ['python3', str(run/'source/bin/m39_launch_anchor.py'), '--run-dir', str(run), '--supervise']
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(run), shlex.join(command)], check=True)
    print(json.dumps({'status': 'detached_controller_started', 'run_dir': str(run), 'session': session}))


if __name__ == '__main__':
    main()
