#!/usr/bin/env python3
"""Resume only the authenticated R02 optional-GCS-404 failure.

Preserve the previous failed state, count adoption, completed chromosomes and
scientific snapshots. The new coordinator differs only in missing-object
classification and a separately named provenance destination. No VM operations,
signals to other processes, fresh local chromosome analyses or new deadlines.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import types


SCHEMA = 'r02_failed_cloud_wait_resume_v1'
RELATIVE = 'repairs/recovery-20261002/coordinator'
PREVIOUS = 'repairs/preprocess-v2/coordinator'
DESTINATION = '00_datos_y_diseno/recovery-20261002/parallel'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'Duplicate manifest field')
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def checked_file(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
            and not path.is_symlink() and sha(path) == expected,
            'Recovery evidence changed: ' + str(path))
    return path


def module(path, expected, name):
    path = checked_file(path, expected)
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), result.__dict__)
    return result


def fixed(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    copy_bytes(path, raw)


def copy_bytes(path, raw):
    path = Path(path)
    require(path.parent.resolve() == path.parent and not path.is_symlink(), 'Unsafe recovery output')
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, 'Existing recovery file differs')
        return
    with path.open('xb') as handle:
        handle.write(raw)


def validate_code_delta(previous_path, current_path):
    """Allow only the reviewed cloud classification and provenance routing delta."""
    previous = ast.parse(Path(previous_path).read_text())
    current = ast.parse(Path(current_path).read_text())
    previous_functions = {node.name: node for node in previous.body if isinstance(node, ast.FunctionDef)}
    current.body = [node for node in current.body
                    if not (isinstance(node, ast.FunctionDef) and node.name == 'missing_object_error')
                    and not (isinstance(node, ast.ImportFrom) and node.module == 'urllib.parse'
                             and [(item.name, item.asname) for item in node.names] == [('quote', None)])]
    before_class = next(node for node in previous.body if isinstance(node, ast.ClassDef) and node.name == 'GCS')
    after_class = next(node for node in current.body if isinstance(node, ast.ClassDef) and node.name == 'GCS')
    before_metadata = next(node for node in before_class.body if isinstance(node, ast.FunctionDef) and node.name == 'metadata')
    after_metadata = next(node for node in after_class.body if isinstance(node, ast.FunctionDef) and node.name == 'metadata')
    old_guard = next(node for node in ast.walk(before_metadata) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.BoolOp))
    new_guard = next(node for node in ast.walk(after_metadata) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.BoolOp))
    expected_guard = ast.parse('optional and missing_object_error(result.stderr, uri)', mode='eval').body
    require(ast.dump(new_guard.test) == ast.dump(expected_guard), 'Unreviewed cloud error handling')
    new_guard.test = copy.deepcopy(old_guard.test)
    function = next(node for node in current.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'run_after_boundary')
    require(function.args.kwonlyargs[-1].arg == 'provenance_destination'
            and isinstance(function.args.kw_defaults[-1], ast.Constant)
            and function.args.kw_defaults[-1].value is None, 'Unreviewed provenance parameter')
    function.args.kwonlyargs.pop()
    function.args.kw_defaults.pop()
    assignment = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                      and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id == 'destination')
    require(isinstance(assignment.value, ast.BoolOp) and isinstance(assignment.value.op, ast.Or)
            and len(assignment.value.values) == 2
            and isinstance(assignment.value.values[0], ast.Name)
            and assignment.value.values[0].id == 'provenance_destination', 'Unreviewed provenance routing')
    assignment.value = assignment.value.values[1]
    require(ast.dump(function) == ast.dump(previous_functions['run_after_boundary']),
            'Coordinator execution behavior changed beyond provenance routing')
    require(ast.dump(current) == ast.dump(previous), 'Coordinator changed outside the authorized operational delta')


def failure_uri(failure, previous, coordinator):
    require(failure.get('state') == 'FAILED', 'Previous coordinator is not failed')
    error = failure.get('error', '')
    for entry in previous['remote_chromosomes']:
        prefix = 'Cannot authenticate cloud object: ' + entry['completion_uri'] + ': '
        if error.startswith(prefix) and coordinator.missing_object_error(
                error[len(prefix):], entry['completion_uri']):
            return entry['completion_uri']
    raise ValueError('Recovery requires the authenticated optional-object 404 failure')


def validate_spec(path, expected):
    path = checked_file(path, expected)
    spec = read(path)
    require(set(spec) == {'schema', 'run_dir', 'resume_sha256', 'coordinator_manifest_sha256',
                         'previous_coordinator_manifest_sha256', 'count_recovery_manifest_sha256',
                         'previous_controller_pid', 'protected_sha256'}, 'Unexpected recovery fields')
    run = Path(spec['run_dir'])
    require(spec['schema'] == SCHEMA and spec['resume_sha256'] == sha(__file__)
            and run.is_absolute() and run.resolve() == run and run.is_dir()
            and path.parent == run/RELATIVE, 'Wrong recovery source or directory')
    pid = spec['previous_controller_pid']
    require(type(pid) is int and pid > 1 and not Path('/proc', str(pid)).exists(),
            'Previous controller PID is still present or invalid')
    previous_dir = run/PREVIOUS
    old_path = checked_file(previous_dir/'manifest.json', spec['previous_coordinator_manifest_sha256'])
    previous = read(old_path)
    old_coordinator = module(previous_dir/'coordinator.py', previous['coordinator_sha256'], '_old_cloud_wait_coordinator')
    old_coordinator.validate_manifest(old_path, spec['previous_coordinator_manifest_sha256'])
    new_path = checked_file(path.parent/'manifest.json', spec['coordinator_manifest_sha256'])
    current = read(new_path)
    require(set(current) == set(previous) and
            {key: value for key, value in current.items() if key != 'coordinator_sha256'} ==
            {key: value for key, value in previous.items() if key != 'coordinator_sha256'},
            'Coordinator recovery changed the deadline, assignments or other settings')
    coordinator_path = checked_file(path.parent/'coordinator.py', current['coordinator_sha256'])
    validate_code_delta(previous_dir/'coordinator.py', coordinator_path)
    coordinator = module(coordinator_path, current['coordinator_sha256'], '_recovered_cloud_wait_coordinator')
    protected = spec['protected_sha256']
    required = {PREVIOUS + '/' + name for name in ('status.json', 'adoption.json', 'coordinator_activation.json')}
    required |= {'checkpoints/chr21_complete.json', 'checkpoints/chr22_complete.json',
                 'checkpoints/chr21_M01_M02_M021.json', 'chr21/trace.tsv',
                 'repairs/boundary-v2/activation.sha256.json'}
    require(isinstance(protected, dict) and set(protected) == required, 'Incomplete protected recovery evidence')
    for relative, checksum in protected.items():
        checked_file(run/relative, checksum)
    failure = read(previous_dir/'status.json')
    require(failure.get('coordinator_manifest_sha256') == spec['previous_coordinator_manifest_sha256'],
            'Failure refers to another coordinator')
    failed_uri = failure_uri(failure, previous, coordinator)
    count_path = checked_file(previous_dir/'recovery.json', spec['count_recovery_manifest_sha256'])
    count_spec = read(count_path)
    count_recovery = module(previous_dir/'r02_count_recovery.py', count_spec['helper_sha256'], '_original_count_recovery')
    _, old, recovery, handoff, pipeline, _ = count_recovery.context(count_path, spec['count_recovery_manifest_sha256'])
    recovery.validate_mount(old)
    adoption = read(previous_dir/'adoption.json')
    require(adoption.get('recovery_manifest_sha256') == spec['count_recovery_manifest_sha256']
            and adoption.get('checkpoint_sha256') == sha(run/'checkpoints/chr21_M01_M02_M021.json')
            and adoption.get('trace_sha256') == sha(run/'chr21/trace.tsv')
            and adoption.get('validator_sha256') == current['record_count_repair']['validator_sha256']
            and adoption.get('original_checkpoint_reused') is True
            and adoption.get('preprocessing_reexecuted') is False
            and adoption.get('scientific_parameters_changed') is False,
            'Count adoption or completed preprocessing changed')
    for filename, key in (('adoption.json', 'adoption_sha256'),
                          ('preprocess_count_validation.py', 'validator_sha256')):
        checked_file(path.parent/filename, current['record_count_repair'][key])
    for chrom in (21, 22):
        record = read(run/f'checkpoints/chr{chrom}_complete.json')
        require(record.get('chromosome') == chrom and record.get('completed_utc')
                and record.get('outputs'), 'Local chromosome is not genuinely complete')
        for output in record['outputs']:
            target = Path(output['path'])
            require(target.is_absolute() and target.resolve() == target
                    and target.is_relative_to(run/f'chr{chrom}'), 'Local chromosome output escaped its directory')
        pipeline.verify_files(record['outputs'])
    boundary = run/'repairs/boundary-v2'
    sources = read(boundary/'source.sha256.json')
    adapter = module(boundary/'source/bin/r02_apply_amendment.py',
                     sources['bin/r02_apply_amendment.py'], '_recovery_frozen_amendment')
    adapter.build_runner(run, boundary, activate=False)
    return spec, current, coordinator, recovery, old, handoff, pipeline, failed_uri


def prepare(run, coordinator_source, previous_controller_pid):
    run, coordinator_source = Path(run).resolve(), Path(coordinator_source).resolve()
    previous_dir, directory = run/PREVIOUS, run/RELATIVE
    previous = read(previous_dir/'manifest.json')
    validate_code_delta(previous_dir/'coordinator.py', coordinator_source)
    require(type(previous_controller_pid) is int and previous_controller_pid > 1
            and not Path('/proc', str(previous_controller_pid)).exists(), 'Previous controller PID remains present')
    directory.mkdir(parents=True, exist_ok=True)
    for filename, source in (('resume.py', Path(__file__)), ('coordinator.py', coordinator_source),
                              ('adoption.json', previous_dir/'adoption.json'),
                              ('preprocess_count_validation.py', previous_dir/'preprocess_count_validation.py')):
        copy_bytes(directory/filename, source.read_bytes())
    current = dict(previous, coordinator_sha256=sha(coordinator_source))
    fixed(directory/'manifest.json', current)
    protected = [PREVIOUS + '/' + name for name in ('status.json', 'adoption.json', 'coordinator_activation.json')]
    protected += ['checkpoints/chr21_complete.json', 'checkpoints/chr22_complete.json',
                  'checkpoints/chr21_M01_M02_M021.json', 'chr21/trace.tsv',
                  'repairs/boundary-v2/activation.sha256.json']
    spec = dict(schema=SCHEMA, run_dir=str(run), resume_sha256=sha(__file__),
                coordinator_manifest_sha256=sha(directory/'manifest.json'),
                previous_coordinator_manifest_sha256=sha(previous_dir/'manifest.json'),
                count_recovery_manifest_sha256=sha(previous_dir/'recovery.json'),
                previous_controller_pid=previous_controller_pid,
                protected_sha256={relative: sha(run/relative) for relative in protected})
    target = directory/'resume.manifest.json'
    fixed(target, spec)
    return target


def run(path, expected, *, execute=False):
    context = validate_spec(path, expected)
    spec, current, coordinator, recovery, old, handoff, pipeline, failed_uri = context
    directory = Path(path).parent
    if not execute:
        return dict(state='VERIFIED_NOT_EXECUTED', failed_uri=failed_uri,
                    deadline_utc=current['deadline_utc'], local_chromosomes_reused=[21, 22])
    class Status:
        def state(self, state, **extra):
            value = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                         resume_manifest_sha256=expected, **extra)
            handoff.write_json(directory/'status.json', value)
            print(json.dumps(value), flush=True)
    status = Status()
    deadline = coordinator.timestamp(current['deadline_utc'])
    def expired(_signal, _frame):
        raise TimeoutError('Original coordinator deadline reached')
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.time()))
    try:
        with recovery.boundary_lock(Path(spec['run_dir'])/'repairs/boundary-v2/.boundary.lock'):
            validate_spec(path, expected)  # Recheck evidence and absence under the shared lock.
            provenance = directory/'provenance'
            provenance.mkdir(exist_ok=True)
            for name in ('resume.py', 'resume.manifest.json', 'coordinator.py'):
                copy_bytes(provenance/name, (directory/name).read_bytes())
            previous_dir = Path(spec['run_dir'])/PREVIOUS
            for name in ('status.json', 'manifest.json', 'recovery.json', 'coordinator_activation.json'):
                copy_bytes(provenance/('previous_' + name), (previous_dir/name).read_bytes())
            coordinator.write_fixed(directory/'resume_activation.json',
                dict(schema=SCHEMA, resume_manifest_sha256=expected, failed_uri=failed_uri,
                     previous_coordinator_manifest_sha256=spec['previous_coordinator_manifest_sha256'],
                     coordinator_manifest_sha256=spec['coordinator_manifest_sha256'],
                     local_chromosomes_reused=[21, 22], deadline_utc=current['deadline_utc'],
                     previous_evidence_rewritten=False, scientific_parameters_changed=False))
            copy_bytes(provenance/'resume_activation.json', (directory/'resume_activation.json').read_bytes())
            status.state('RECOVERY_VERIFIED_WAITING_REMOTE')
            coordinator.run_after_boundary(current, directory, status, activate=False,
                                           provenance_destination=DESTINATION)
            return dict(state='COMPLETE')
    except BaseException as error:
        status.state('FAILED', error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'run'))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--coordinator-source', type=Path)
    parser.add_argument('--previous-controller-pid', type=int)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--expected-spec-sha256')
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == 'prepare':
        require(args.run_dir and args.coordinator_source and args.previous_controller_pid,
                'Prepare requires run, reviewed coordinator source and previous PID')
        path = prepare(args.run_dir, args.coordinator_source, args.previous_controller_pid)
        print(json.dumps(dict(state='PREPARED_NOT_VERIFIED_OR_RUNNING', spec=str(path), sha256=sha(path))))
    else:
        require(args.spec and args.expected_spec_sha256, 'Verification/execution needs the sealed manifest hash')
        print(json.dumps(run(args.spec, args.expected_spec_sha256, execute=args.mode == 'run')))


if __name__ == '__main__':
    main()
