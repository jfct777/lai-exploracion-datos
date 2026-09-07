#!/usr/bin/env python3
"""Execute the declared paired arms serially on one disposable GPU worker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from m39_gpu_serial_profile import sha256, stop_group, verify_seal
from m39_ordered_gpu_manifest import load_plan, require


def run(args: argparse.Namespace) -> int:
    seal = verify_seal(args.source_seal, Path(__file__).parent)
    require(sha256(args.plan) == seal['profile_sha256'], 'plan differs from source seal')
    plan = load_plan(args.plan)
    groups = [group for group in plan['groups'] if group['id'] == args.group_id]
    require(len(groups) == 1, 'expected one declared configuration group')
    group = groups[0]
    args.outdir.mkdir(parents=True, mode=0o700, exist_ok=False)
    started, process = time.monotonic(), None
    result = {'schema_version': 'm39-ordered-gpu-group-completion-v1', 'status': 'FAILED',
              'stage': plan['stage'],
              'group_id': args.group_id, 'source_commit': seal['source_commit'],
              'source_seal_sha256': sha256(args.source_seal), 'plan_sha256': sha256(args.plan),
              'completed_arms': [], 'paired_checks': {}, 'case_receipt_sha256': {},
              'SCORE_opened': False, 'new_batch_workers': 1, 'exit_code': 1}
    previous = {}
    def interrupted(signum, frame):
        raise InterruptedError(f'GPU group received signal {signum}')
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        for spec in group['configs']:
            config_path = args.plan.parent / spec['file']
            cfg = json.loads(config_path.read_text())
            remaining = plan['resources']['task_seconds'] - (time.monotonic() - started)
            require(remaining > 0, 'group wall-time budget exhausted')
            output = args.outdir / cfg['arm']
            command = [sys.executable, str(Path(__file__).with_name('m39_ordered_training.py'))]
            for key, value in (('train-store', args.train_store), ('select-store', args.select_store),
                               ('development', args.development), ('config', config_path), ('outdir', output)):
                command.extend(('--' + key, str(value)))
            print(json.dumps({'event': 'arm_start', 'group': args.group_id, 'arm': cfg['arm']}), flush=True)
            process = subprocess.Popen(command, start_new_session=True)
            exit_code = process.wait(timeout=min(remaining, cfg['max_runtime_seconds'] + 10))
            require(exit_code == 0, f"arm {cfg['arm']} failed: {exit_code}")
            receipt_path = output / 'training.receipt.json'
            receipt = json.loads(receipt_path.read_text())
            require(receipt.get('decision') == 'COMPLETED_EXPLORATORY_DEVELOPMENT_CASE'
                    and receipt.get('config') == cfg
                    and receipt.get('sources', {}).get('config_sha256') == spec['sha256']
                    and receipt.get('scope', {}).get('SCORE_opened') is False,
                    'training receipt binding or scope differs')
            paired = {key: receipt[key] for key in ('initial_state_sha256', 'training_pair_stream_sha256',
                'anchor_indices', 'training_observations', 'distinct_train_queries_exposed',
                'distinct_train_anchors_exposed', 'batch_policy')}
            require(not result['paired_checks'] or paired == result['paired_checks'],
                    'paired arms used different initialization, observations or anchors')
            result['paired_checks'] = paired
            result['completed_arms'].append(cfg['arm'])
            result['case_receipt_sha256'][cfg['arm']] = sha256(receipt_path)
        verify_seal(args.source_seal, Path(__file__).parent)
        require(sha256(args.plan) == seal['profile_sha256'], 'plan changed during run')
        load_plan(args.plan)
        result.update(status='COMPLETED_DECLARED_PAIRED_ARMS_NEEDS_SCIENTIFIC_POST', exit_code=0)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result.update(failure_type=type(exc).__name__, failure=str(exc),
                      exit_code=124 if isinstance(exc, subprocess.TimeoutExpired) else 1)
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            stop_group(process)
        result['elapsed_seconds'] = time.monotonic() - started
        with (args.outdir / 'group.completion.json').open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write('\n')
        print(json.dumps({'event': 'ordered_group_completion', 'receipt': result}), flush=True)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return result['exit_code']


def transport_exit_code(outdir: Path, code: int, preserve: bool) -> int:
    """Copy partial artifacts before a downstream scientific audit rejects the failure."""
    if code == 0 or not preserve:
        return code
    receipt = json.loads((outdir / 'group.completion.json').read_text())
    require(receipt.get('status') == 'FAILED' and receipt.get('exit_code') == code,
            'failure transport requires an explicit matching failure receipt')
    print(json.dumps({'event': 'failed_group_preserved_not_success', 'scientific_exit_code': code}), flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train-store', 'select-store', 'development', 'plan', 'source-seal', 'outdir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--group-id', required=True)
    parser.add_argument('--preserve-failure-output', action='store_true',
                        help='Transport a recorded failure for downstream audit, not scientific success')
    os.umask(0o077)
    args = parser.parse_args()
    code = run(args)
    raise SystemExit(transport_exit_code(args.outdir, code, args.preserve_failure_output))


if __name__ == '__main__':
    main()
