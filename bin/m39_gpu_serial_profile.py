#!/usr/bin/env python3
"""Run the six frozen resource profiles serially on one Batch GPU worker.

The biological models and per-case profiler are reused unchanged. Each case has
its own process so allocator state and thread-pool configuration cannot leak to
the next measurement. This is a resource benchmark with artificial labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(path: Path, source_dir: Path) -> dict:
    seal = json.loads(path.read_text())
    if (seal.get('schema_version') != 'm39-gpu-source-seal-v1' or
            not re.fullmatch(r'[0-9a-f]{40}', seal.get('source_commit', ''))):
        raise ValueError('Invalid immutable source seal')
    for name, digest in seal['source_sha256'].items():
        if not re.fullmatch(r'm[0-9]+_[a-z0-9_]+\.py', name):
            raise ValueError('Unsafe staged source name')
        if sha256(source_dir / name) != digest:
            raise ValueError(f'Staged source differs: {name}')
    return seal


def case_commands(args: argparse.Namespace, profile: dict) -> list[tuple[str, list[str]]]:
    expected = [f'{family}-small-{policy}-b{batch}-real'
                for family in ('cnn', 'attention')
                for policy, batch in (('grouped', 1), ('grouped', 2), ('random', 2))]
    if (profile.get('schema_version') != 'm39-ordered-throughput-gpu-profile-v1' or
            profile.get('device') != 'cuda:0' or profile.get('max_workers') != 1 or
            [case['id'] for case in profile.get('cases', [])] != expected):
        raise ValueError('Requires the frozen six-case, one-worker GPU profile')
    commands = []
    for name in expected:
        command = [sys.executable, str(Path(__file__).with_name('m39_profile_ordered_throughput.py'))]
        for key, value in (('store-dir', args.store_dir), ('parent-receipt', args.parent_receipt),
                           ('folds', args.folds), ('profile-config', args.profile_config),
                           ('case-id', name), ('output-dir', args.output_dir / name),
                           ('source-commit', args.source_commit)):
            command.extend((f'--{key}', str(value)))
        commands.append((name, command))
    return commands


def stop_group(process: subprocess.Popen) -> None:
    """Terminate this worker's process group, not other project processes."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.max_seconds <= 1800:
        raise ValueError('Serial runtime ceiling must be at most 1800 seconds')
    seal = verify_seal(args.source_seal, Path(__file__).parent)
    if seal['source_commit'] != args.source_commit:
        raise ValueError('Source commit differs from seal')
    profile = json.loads(args.profile_config.read_text())
    if sha256(args.profile_config) != seal['profile_sha256']:
        raise ValueError('Profile differs from source seal')
    commands = case_commands(args, profile)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    start = time.monotonic()
    result = {'schema_version': 'm39-gpu-serial-completion-v1', 'status': 'FAILED',
              'source_commit': args.source_commit, 'source_seal_sha256': sha256(args.source_seal),
              'expected_cases': [name for name, _ in commands], 'completed_cases': [],
              'case_exit_codes': {}, 'max_seconds': args.max_seconds,
              'new_cloud_instances': 1, 'gpu_count': 1, 'concurrent_workers': 1,
              'accuracy_evaluated': False, 'weights_saved': False, 'exit_code': 1}
    process = None
    previous = {}
    def interrupted(signum, frame):
        raise InterruptedError(f'Serial worker received signal {signum}')
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        for name, command in commands:
            remaining = args.max_seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, args.max_seconds)
            print(json.dumps({'event': 'case_start', 'case': name}), flush=True)
            process = subprocess.Popen(command, start_new_session=True)
            code = process.wait(timeout=min(remaining, profile['max_seconds'] + 10))
            result['case_exit_codes'][name] = code
            if code:
                raise RuntimeError(f'Case {name} failed with exit code {code}')
            report = json.loads((args.output_dir / name / 'profile.json').read_text())
            if (report.get('decision') != 'PASS_ORDERED_THROUGHPUT_TECHNICAL_ONLY' or
                    report.get('runtime', {}).get('device') != 'cuda:0' or
                    report.get('provenance', {}).get('source_commit') != args.source_commit):
                raise ValueError('Case report did not authenticate successful GPU execution')
            result['completed_cases'].append(name)
        verify_seal(args.source_seal, Path(__file__).parent)
        result.update(status='PASS_GPU_RESOURCE_PROFILE_ONLY', exit_code=0)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result.update(failure_type=type(exc).__name__, failure=str(exc),
                      exit_code=124 if isinstance(exc, subprocess.TimeoutExpired) else 1)
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            stop_group(process)
        result['elapsed_seconds'] = time.monotonic() - start
        with (args.output_dir / 'gpu-serial-completion.json').open('x') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write('\n')
        # Batch preserves the command log even when failed scratch outputs cannot
        # be published. This aggregate receipt contains no genotypes or predictions.
        print(json.dumps({'event': 'gpu_serial_completion', 'receipt': result}, sort_keys=True), flush=True)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return result['exit_code']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('store-dir', 'parent-receipt', 'folds', 'profile-config', 'source-seal', 'output-dir'):
        parser.add_argument(f'--{key}', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--max-seconds', required=True, type=int)
    os.umask(0o077)
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
