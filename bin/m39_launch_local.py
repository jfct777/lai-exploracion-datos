#!/usr/bin/env python3
"""Launch the two bounded M39 diagnostics in detached, auditable tmux sessions.

No cloud instance is created. Processes survive an SSH/IDE disconnect while
the existing development VM stays on. Biological training is not launched.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--lane', choices=('bridge', 'capacity'), required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--input-dir', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(repo / '.claude' / 'runs') or not run_dir.is_dir():
        parser.error('Use an existing private project run directory')
    if not re.fullmatch(r'[a-z0-9._-]{3,70}', run_dir.name):
        parser.error('Unsafe run identifier')
    tracked = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                     cwd=repo, text=True)
    if tracked.strip():
        parser.error('Commit tracked code changes before a production diagnostic')
    names = {
        'bridge': ('conf/m39_carrier_context_local.config', 'workflows/m39_carrier_context_bridge.nf'),
        'capacity': ('conf/m39_capacity_local.config', 'workflows/m39_capacity_screen.nf'),
    }
    cfg, workflow = names[args.lane]
    output = run_dir / f'{args.lane}-output'
    if output.exists():
        parser.error('Output already exists; choose a new run directory')
    command = [
        'env', 'NXF_OFFLINE=true', 'NXF_VER=26.04.6', 'nextflow', '-C', str(repo/cfg),
        '-log', str(run_dir/f'{args.lane}.nextflow.log'), 'run', str(repo/workflow),
        '-work-dir', str(run_dir/f'{args.lane}-work'),
        '-with-trace', str(run_dir/f'{args.lane}.trace.tsv'),
        '-with-report', str(run_dir/f'{args.lane}.report.html'),
        '-with-timeline', str(run_dir/f'{args.lane}.timeline.html'),
        '--m39_output_dir', str(output),
    ]
    if args.lane == 'bridge':
        input_dir = (args.input_dir or run_dir/'inputs').resolve()
        if not input_dir.is_relative_to(repo/'.claude'/'runs') or not input_dir.is_dir():
            parser.error('Bridge inputs must be staged in the private project runs')
        command += ['--m39_input_dir', str(input_dir)]
        contract = (args.contract or repo/'conf/m39_carrier_context_bridge.json').resolve()
        if not contract.is_relative_to(repo) or not contract.is_file():
            parser.error('Bridge contract must be an existing file in this project')
        command += ['--m39_contract', str(contract)]
    elif args.contract or args.input_dir:
        parser.error('The synthetic lane cannot read bridge inputs or contracts')
    session = f'{run_dir.name}-{args.lane}'
    receipt_path = run_dir/f'{args.lane}.launch.json'
    descriptor = os.open(receipt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    # Keep argv and immutable source revision even if tmux cannot start.
    receipt = {'schema_version': 'm39-local-launch-v1', 'lane': args.lane,
               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'argv': command, 'tmux_session': session,
               'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
               'config_sha256': hashlib.sha256((repo/cfg).read_bytes()).hexdigest(),
               'scope': 'technical_or_synthetic_only',
               'new_cloud_instances': 0, 'status': 'launch_requested'}
    if args.lane == 'bridge':
        receipt['bridge_contract_sha256'] = hashlib.sha256(contract.read_bytes()).hexdigest()
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    # A tmux server, not the invoking shell, owns the scientific controller.
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(repo),
                    shlex.join(command)], check=True)
    print(json.dumps({'session': session, 'receipt': str(receipt_path), 'status': 'tmux_started'}))


if __name__ == '__main__':
    main()
