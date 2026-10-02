#!/usr/bin/env python3
"""Adopt genuine chr21 completion after a legacy-index statistics failure.

No raw data, checkpoints, frozen sources or failed status records are rewritten.
The completed Nextflow checkpoint is authenticated before calling its cached
Runner path with a declared, hash-bound count validator. The new coordinator
uses the same validator; all scientific functions and settings remain frozen.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import types


SCHEMA = 'r02_completed_preprocess_count_recovery_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(path, expected, name):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path) == expected,
            'Recovery dependency changed: ' + str(path))
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), result.__dict__)
    return result


def fixed(path, value):
    path = Path(path)
    if path.exists():
        require(not path.is_symlink() and read(path) == value, 'Existing recovery evidence differs')
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def context(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(),
            'Recovery manifest must use its canonical absolute path')
    require(sha(path) == expected, 'Count recovery manifest differs')
    spec = read(path)
    require(set(spec) == {'schema', 'run_dir', 'helper_sha256', 'validator_sha256',
                         'recovery_manifest_sha256', 'failed_status_sha256',
                         'previous_controller', 'previous_controller_manifest_sha256'},
            'Unexpected count recovery fields')
    run = Path(spec['run_dir'])
    require(spec['schema'] == SCHEMA and run.resolve() == run and run.is_dir()
            and path.parent == run/'repairs/preprocess-v2/coordinator'
            and sha(__file__) == spec['helper_sha256'], 'Wrong recovery source or location')
    original = run/'repairs/local-preprocess-v1/manifest.json'
    require(sha(original) == spec['recovery_manifest_sha256'], 'Previous recovery manifest changed')
    old = read(original)
    recovery = module(original.parent/'resume.py', old['wrapper_sha256'], '_original_recovery')
    old, _, _, handoff, pipeline, _ = recovery.validate_spec(original, spec['recovery_manifest_sha256'])
    previous = run/'repairs/preprocess-v1/controller-boundary/manifest.json'
    require(sha(previous) == spec['previous_controller_manifest_sha256']
            and read(previous)['supervisor'] == spec['previous_controller'],
            'Previous controller identity changed')
    require(handoff.authenticated(spec['previous_controller']) is None, 'Previous controller is still active')
    recovery.validate_idle(old, handoff, pipeline)
    helper = module(path.parent/'preprocess_count_validation.py', spec['validator_sha256'], '_count_validator')
    failure = run/'repairs/local-preprocess-v1/status.json'
    require(sha(failure) == spec['failed_status_sha256'], 'Previous failure evidence changed')
    status = read(failure)
    require(status.get('state') == 'FAILED' and status.get('error') ==
            'chr21: source index count does not match completed sequential M01 annotation',
            'Recovery only accepts the exact post-checkpoint index-statistics failure')
    return spec, old, recovery, handoff, pipeline, helper


def completed_checkpoint(run, recovery, pipeline):
    checkpoint = run/'checkpoints/chr21_M01_M02_M021.json'
    require(checkpoint.is_file() and not checkpoint.is_symlink(), 'No genuine completed checkpoint')
    record = read(checkpoint)
    source = run/'repairs/boundary-v2'
    sources = read(source/'source.sha256.json')
    adapter = module(source/'source/bin/r02_apply_amendment.py',
                     sources['bin/r02_apply_amendment.py'], '_original_amendment')
    prefix = run/'chr21/preprocess/lai_rare/dnabr.hg38.2723.chr21.rare.minor'
    outputs = {str(prefix) + suffix for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv')}
    require(type(record.get('returncode')) is int and record['returncode'] == 0
            and record.get('command') == adapter.expected_boundary_command(run, 21)
            and record.get('completed_utc') and len(record.get('outputs', [])) == 4
            and {item.get('path') for item in record['outputs']} == outputs,
            'Completed checkpoint does not authenticate the original preprocessing')
    pipeline.verify_files(record['outputs'])
    recovery.check_trace(run/'chr21/trace.tsv', recovered=True)
    return checkpoint, record


def adopt(path, expected, *, execute=False):
    spec, old, recovery, handoff, pipeline, helper = context(path, expected)
    run, directory = Path(spec['run_dir']), Path(path).parent
    recovery.validate_mount(old)
    checkpoint, record = completed_checkpoint(run, recovery, pipeline)
    checkpoint_hash = sha(checkpoint)
    audit = helper.validate_raw_record_count(run/'chr21', 21)
    require(audit['raw_record_count_source'] == 'sequential_m01_with_missing_tbi_statistics',
            'This recovery requires demonstrated absent legacy TBI statistics')
    evidence = dict(schema=SCHEMA, recovery_manifest_sha256=expected, checkpoint_sha256=checkpoint_hash,
                    trace_sha256=sha(run/'chr21/trace.tsv'), validator_sha256=spec['validator_sha256'],
                    original_checkpoint_reused=True, preprocessing_reexecuted=False,
                    historical_failed_status_rewritten=False, scientific_parameters_changed=False,
                    biological_validation_complete=False, count_validation=audit)
    if not execute:
        return dict(state='VERIFIED_NOT_ADOPTED', **evidence)
    with recovery.boundary_lock(run/'repairs/boundary-v2/.boundary.lock'), pipeline.execution_lock(run):
        checked, _, _, _, _, _ = context(path, expected)
        require(checked == spec and helper.validate_raw_record_count(run/'chr21', 21) == audit,
                'Recovery inputs changed while acquiring locks')
        recovery.validate_idle(old, handoff, pipeline)
        require(sha(checkpoint) == checkpoint_hash, 'Checkpoint changed before adoption')
        pipeline.validate_raw_record_count = helper.validate_raw_record_count
        runner = pipeline.Runner(run)
        # Existing exact command+outputs are checked again by Runner.command.
        # This returns from its checkpoint, never invokes Nextflow again.
        runner.preprocess(21)
        require(sha(checkpoint) == checkpoint_hash and sha(run/'chr21/trace.tsv') == evidence['trace_sha256'],
                'Recovery must not change checkpoint or trace')
        fixed(directory/'adoption.json', evidence)
        boundary = run/'repairs/boundary-v2'
        request = read(boundary/'request.json')
        amendment = copy.deepcopy(request['amendment_template'])
        amendment['boundary']['checkpoint_sha256'] = checkpoint_hash
        fixed(boundary/'preprocess_count_recovery.json', evidence)
        fixed(boundary/'amendment.json', amendment)
        fixed(boundary/'frozen.sha256.json', {name: sha(boundary/name) for name in
              ('amendment.json', 'source.sha256.json', 'request.json', 'preprocess_count_recovery.json')})
    return dict(state='GENUINE_COMPLETION_ADOPTED', **evidence)


def resume(path, expected, coordinator_expected):
    spec, old, recovery, handoff, pipeline, _ = context(path, expected)
    run, directory = Path(spec['run_dir']), Path(path).parent
    manifest = directory/'manifest.json'
    require(sha(manifest) == coordinator_expected, 'Coordinator manifest changed')
    coordinator_spec = read(manifest)
    coordinator = module(directory/'coordinator.py', coordinator_spec['coordinator_sha256'], '_count_coordinator')
    coordinator_spec = coordinator.validate_manifest(manifest, coordinator_expected)
    require(coordinator_spec['record_count_repair'] == {
            'validator_sha256': spec['validator_sha256'], 'adoption_sha256': sha(directory/'adoption.json')},
            'Coordinator count repair differs')
    evidence = read(directory/'adoption.json')
    require(evidence['recovery_manifest_sha256'] == expected
            and evidence['checkpoint_sha256'] == sha(run/'checkpoints/chr21_M01_M02_M021.json')
            and evidence['trace_sha256'] == sha(run/'chr21/trace.tsv'), 'Adopted completion changed')
    class Status:
        def state(self, state, **extra):
            value = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                         coordinator_manifest_sha256=coordinator_expected, **extra)
            handoff.write_json(directory/'status.json', value)
            print(json.dumps(value), flush=True)
    status = Status()
    deadline = coordinator.timestamp(coordinator_spec['deadline_utc'])
    def expired(_signal, _frame):
        raise TimeoutError('Original coordinator deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline-time.time()))
    with recovery.boundary_lock(run/'repairs/boundary-v2/.boundary.lock'):
        try:
            recovery.validate_mount(old)
            status.state('COUNT_REPAIR_VERIFIED_CONTINUING')
            coordinator.run_after_boundary(coordinator_spec, directory, status)
        except BaseException as error:
            status.state('FAILED', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--adopt', action='store_true')
    modes.add_argument('--run', action='store_true')
    parser.add_argument('--coordinator-sha256')
    args = parser.parse_args()
    os.umask(0o077)
    if args.run:
        require(args.coordinator_sha256 is not None, 'Coordinator hash required for execution')
        resume(args.spec, args.spec_sha256, args.coordinator_sha256)
    else:
        print(json.dumps(adopt(args.spec, args.spec_sha256, execute=args.adopt)))


if __name__ == '__main__':
    main()
