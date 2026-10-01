#!/usr/bin/env python3
"""Resume the R02 coordinator after a genuine, sealed chr21 preprocessing result.

No signals, new preprocessing, artificial checkpoints, VM operations or snapshot
replacement are performed here. The previous controller must exit itself before
this successor acquires the original boundary lock. Remote operational deltas
are accepted only through the separately authenticated coordinator manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time
import types
import hashlib
import json


SCHEMA = 'r02_optimized_coordinator_resume_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def module(path, checksum, name):
    require(Path(path).is_file() and not Path(path).is_symlink()
            and sha(path) == checksum, 'Authenticated continuation source changed')
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), result.__dict__)
    return result


def validate_spec(path, expected):
    path = Path(path).resolve()
    require(not path.is_symlink() and sha(path) == expected, 'Continuation manifest changed')
    spec = load(path)
    require(set(spec) == {'schema', 'run_dir', 'resume_sha256', 'coordinator_manifest_sha256',
                         'recovery_manifest_sha256', 'previous_controller', 'poll_seconds'},
            'Unexpected continuation manifest fields')
    require(spec['schema'] == SCHEMA and spec['resume_sha256'] == sha(__file__),
            'Wrong continuation schema or executable')
    run = Path(spec['run_dir'])
    require(run.is_absolute() and run.resolve() == run and run.is_dir()
            and path.parent == run/'repairs/preprocess-v1/coordinator', 'Unexpected continuation directory')
    require(type(spec['poll_seconds']) is int and 1 <= spec['poll_seconds'] <= 60,
            'Continuation polling must be between1 and60 seconds')
    recovery_path = run/'repairs/local-preprocess-v1/manifest.json'
    require(sha(recovery_path) == spec['recovery_manifest_sha256'], 'Recovery manifest changed')
    recovery_spec = load(recovery_path)
    recovery = module(recovery_path.parent/'resume.py', recovery_spec['wrapper_sha256'], '_previous_local_recovery')
    recovery.identity(spec['previous_controller'])
    old_spec, _, _, handoff, pipeline, _ = recovery.validate_spec(recovery_path, spec['recovery_manifest_sha256'])
    coordinator_path = path.parent/'manifest.json'
    require(sha(coordinator_path) == spec['coordinator_manifest_sha256'], 'New coordinator manifest changed')
    coordinator_spec = load(coordinator_path)
    coordinator = module(path.parent/'coordinator_snapshot.py', coordinator_spec['coordinator_sha256'], '_new_parallel_coordinator')
    coordinator_spec = coordinator.validate_manifest(coordinator_path, spec['coordinator_manifest_sha256'])
    require(coordinator_spec['run_dir'] == str(run), 'Continuation and coordinator runs differ')
    return spec, coordinator_spec, coordinator, recovery, old_spec, handoff, pipeline


def validate_completion(spec, recovery, old_spec, handoff, pipeline):
    """Only consume the previous controller's completed receipt; never create it."""
    run = Path(spec['run_dir'])
    require(handoff.authenticated(spec['previous_controller']) is None, 'Previous controller is still active')
    recovery.validate_idle(old_spec, handoff, pipeline)
    folder = run/'repairs/local-preprocess-v1'
    evidence_path = folder/'completed.json'
    require(evidence_path.is_file() and not evidence_path.is_symlink(), 'Previous chr21 recovery did not complete')
    evidence = load(evidence_path)
    checkpoint = run/'checkpoints/chr21_M01_M02_M021.json'
    require(checkpoint.is_file() and not checkpoint.is_symlink()
            and evidence.get('recovery_spec_sha256') == spec['recovery_manifest_sha256']
            and evidence.get('wrapper_sha256') == old_spec['wrapper_sha256']
            and evidence.get('checkpoint_sha256') == sha(checkpoint), 'Recovery completion identity changed')
    require(evidence.get('checkpoint_created_by') == 'frozen_original_Runner.preprocess(21)'
            and evidence.get('historical_blocked_fork_status_fabricated') is False
            and evidence.get('scientific_parameters_changed') is False, 'Recovery did not preserve the original computation')
    receipt = load(checkpoint)
    require(type(receipt.get('returncode')) is int and receipt['returncode'] == 0,
            'Recovered preprocessing checkpoint is not successful')
    pipeline.verify_files(receipt['outputs'])
    trace = run/'chr21/trace.tsv'
    require(evidence.get('trace_sha256') == sha(trace)
            and evidence.get('trace') == recovery.check_trace(trace, recovered=True), 'Recovered chr21 trace changed')
    boundary = run/'repairs/boundary-v2'
    hashes = load(boundary/'source.sha256.json')
    adapter = module(boundary/'source/bin/r02_apply_amendment.py',
                     hashes['bin/r02_apply_amendment.py'], '_sealed_chr21_amendment')
    require(receipt.get('command') == evidence.get('command') == adapter.expected_boundary_command(run, 21),
            'Recovered preprocessing command differs')
    adapter.validate_snapshot(run, boundary)
    require(load(boundary/'amendment.json')['boundary']['checkpoint_sha256'] == sha(checkpoint)
            and load(boundary/'preprocess_storage_recovery.json') == evidence,
            'Boundary seal does not refer to this genuine recovery')
    return dict(recovery_completion_sha256=sha(evidence_path), checkpoint_sha256=sha(checkpoint),
                trace_sha256=sha(trace), boundary_seal_sha256=sha(boundary/'frozen.sha256.json'))


def run(spec_path, expected, *, execute=False):
    context = validate_spec(spec_path, expected)
    spec, coordinator_spec, coordinator, recovery, old_spec, handoff, pipeline = context
    directory = Path(spec_path).resolve().parent
    class Status:
        def state(self, state, **extra):
            value = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                         continuation_manifest_sha256=expected, **extra)
            handoff.write_json(directory/'resume_status.json', value)
            print(json.dumps(value), flush=True)
    if not execute:
        return dict(state='VERIFIED_NOT_EXECUTED', previous_controller_active=
                    handoff.authenticated(spec['previous_controller']) is not None)
    status = Status()
    deadline = coordinator.timestamp(coordinator_spec['deadline_utc'])
    def expired(_signal, _frame):
        raise TimeoutError('Original parallel deadline reached during coordinator handoff')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.time()))
    while handoff.authenticated(spec['previous_controller']) is not None:
        require(time.time() < deadline, 'Coordinator handoff deadline expired')
        status.state('WAITING_PREVIOUS_CONTROLLER_EXIT')
        time.sleep(min(spec['poll_seconds'], max(0, deadline-time.time())))
    with recovery.boundary_lock(Path(spec['run_dir'])/'repairs/boundary-v2/.boundary.lock'):
        try:
            evidence = validate_completion(spec, recovery, old_spec, handoff, pipeline)
            coordinator.write_fixed(directory/'resume_activation.json',
                dict(schema=SCHEMA, continuation_manifest_sha256=expected, **evidence))
            status.state('RECOVERY_VERIFIED_NEW_COORDINATOR_ACTIVE', **evidence)
            coordinator.run_after_boundary(coordinator_spec, directory, status)
            return dict(state='COMPLETE')
        except BaseException as error:
            status.state('FAILED', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--expected-spec-sha256', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    print(json.dumps(run(args.spec, args.expected_spec_sha256, execute=args.run)))


if __name__ == '__main__':
    main()
