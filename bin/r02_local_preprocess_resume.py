#!/usr/bin/env python3
"""Resume authenticated chr21 preprocessing after a storage-only repair.

This program neither signals processes nor mounts storage. It requires the old
supervisor, Nextflow and coordinator to have stopped, and an authenticated local
bind mount prepared separately. It calls the frozen original Runner.preprocess
with unchanged parameters and command, verifies genuine Nextflow reuse of M01,
then continues the frozen parallel coordinator. No completion checkpoint or
historical blocked-fork status is synthesized.

CLI: --spec /absolute/manifest.json --expected-spec-sha256 SHA [--run].
The manifest lives under RUN/repairs/local-preprocess-v1. Required fields are
schema, run_dir, wrapper_sha256, parallel_manifest_sha256, local_backing_dir,
storage_receipt {path,sha256}, archived_trace {path,sha256}, m01_task_dir,
m01_files [{path,bytes,sha256,mtime_ns}], old_processes with supervisor, child,
and coordinator process identities {pid,start_ticks,cmdline_sha256}.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import types

SCHEMA = 'r02_local_preprocess_resume_v1'
STAGE = 'chr21_M01_M02_M021'
M01 = 'PREPROCESS_NORM_LEFTALIGN (chr21)'
M02 = 'PREPROCESS_FILTER_SNV_BIALLELIC_PASS (chr21)'
M021 = 'LAI_RARE_BIALELIC_ONLY (chr21)'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def authenticated_module(path, digest, name):
    require(sha(path) == digest, 'Module hash mismatch: ' + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def identity(value):
    require(set(value) == {'pid', 'start_ticks', 'cmdline_sha256'}
            and type(value['pid']) is int and value['pid'] > 1
            and type(value['start_ticks']) is int and value['start_ticks'] > 0
            and re.fullmatch('[0-9a-f]{64}', value['cmdline_sha256']), 'Invalid process identity')


def trace_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def check_trace(path, *, recovered):
    rows = trace_rows(path)
    names = [row.get('name') for row in rows]
    require(names.count(M01) == 1, 'Trace must contain exactly one chr21 M01 task')
    norm = next(row for row in rows if row['name'] == M01)
    require(norm.get('status') == ('CACHED' if recovered else 'COMPLETED')
            and norm.get('exit') == '0', 'M01 was not genuinely cached/completed as required')
    if recovered:
        require(set(names) == {M01, M02, M021} and len(rows) == 3,
                'Recovered trace must contain exactly M01, M02 and M02.1')
        for name in (M02, M021):
            row = next(row for row in rows if row['name'] == name)
            require(row.get('status') in ('COMPLETED', 'CACHED') and row.get('exit') == '0',
                    'Recovered task did not finish successfully: ' + name)
    return [dict(name=row['name'], status=row['status'], exit=row.get('exit'),
                 hash=row.get('hash')) for row in rows]


def verify_record(record, root):
    path = Path(record['path'])
    require(path.is_absolute() and path.resolve().is_relative_to(root)
            and path.is_file() and not path.is_symlink(), 'Unsafe evidence record')
    require(sha(path) == record['sha256'], 'Evidence hash mismatch: ' + str(path))
    return path


def validate_spec(path, expected):
    path = Path(path).resolve()
    require(re.fullmatch('[0-9a-f]{64}', expected) and sha(path) == expected,
            'Recovery spec hash mismatch')
    spec = load(path)
    require(spec.get('schema') == SCHEMA, 'Invalid recovery schema')
    require(spec.get('wrapper_sha256') == sha(__file__), 'Recovery wrapper changed')
    run = Path(spec['run_dir'])
    require(run.is_absolute() and run.resolve() == run and run.is_dir(), 'Invalid run directory')
    require(path.parent == run/'repairs/local-preprocess-v1', 'Unexpected recovery directory')
    require(set(spec['old_processes']) == {'supervisor', 'child', 'coordinator'},
            'All three old process identities are required')
    for item in spec['old_processes'].values():
        identity(item)
    require(len({p['pid'] for p in spec['old_processes'].values()}) == 3, 'Duplicated old process PID')
    backing = Path(spec['local_backing_dir'])
    require(backing.is_absolute() and backing.resolve() == backing and backing.is_dir()
            and backing.is_relative_to(run), 'Local backing directory escaped the run')
    verify_record(spec['storage_receipt'], run)
    verify_record(spec['archived_trace'], run)
    original_trace = check_trace(spec['archived_trace']['path'], recovered=False)
    pdir = run/'repairs/parallel-v1'
    require(sha(pdir/'manifest.json') == spec['parallel_manifest_sha256'], 'Parallel manifest changed')
    p_spec = load(pdir/'manifest.json')
    parallel = authenticated_module(pdir/'coordinator.py', p_spec['coordinator_sha256'], '_local_parallel')
    p_spec = parallel.validate_manifest(pdir/'manifest.json', spec['parallel_manifest_sha256'])
    boundary = run/'repairs/boundary-v2'
    hashes = load(boundary/'source.sha256.json')
    handoff = authenticated_module(boundary/'source/bin/r02_stage_boundary_handoff.py',
        hashes['bin/r02_stage_boundary_handoff.py'], '_local_handoff')
    request = load(boundary/'request.json')
    require(spec['old_processes']['supervisor'] == request['supervisor']
            and spec['old_processes']['child'] == request['child'], 'Original process identities differ')
    fixed_boundary = request['amendment_template']['boundary']
    require(fixed_boundary['chromosome'] == 21 and fixed_boundary['stage'] == STAGE,
            'Recovery is restricted to the original chr21 boundary')
    for name, field in (('parameters.json', 'parameters_sha256'),
                        ('runtime.config', 'runtime_config_sha256')):
        require(sha(run/'chr21'/name) == fixed_boundary[field],
                'Original preprocessing configuration changed: ' + name)
    ready = load(pdir/'ready.json')
    require(ready['pid'] == spec['old_processes']['coordinator']['pid']
            and ready['manifest_sha256'] == spec['parallel_manifest_sha256'],
            'Replaced coordinator differs from its original readiness record')
    original = load(run/'source.sha256.json')
    pipeline = authenticated_module(run/'source/bin/r02_autosome_pipeline.py',
        original['bin/r02_autosome_pipeline.py'], '_local_original_pipeline')
    return spec, p_spec, parallel, handoff, pipeline, original_trace


def validate_idle(spec, handoff, pipeline, *, containers=True):
    for name, item in spec['old_processes'].items():
        require(handoff.authenticated(item) is None, 'Old process remains active: ' + name)
    if containers:
        require(not handoff.containers(load(Path(spec['run_dir'])/'run.json')['run_id']),
                'Original run still has active containers')
    pipeline.Runner(Path(spec['run_dir']))  # Read-only authentication of all original manifests.


def validate_mount(spec, *, command=subprocess.check_output):
    run = Path(spec['run_dir'])
    target = Path(load(run/'run.json')['bulk'])
    backing = Path(spec['local_backing_dir'])
    info = json.loads(command(['findmnt', '--target', str(target), '--json',
                              '--output', 'TARGET,FSTYPE'], text=True))
    mounts = info.get('filesystems', [])
    require(len(mounts) == 1 and mounts[0].get('target') == str(target)
            and mounts[0].get('fstype') in ('ext4', 'xfs'), 'Bulk is not the exact local bind mount')
    a, b = target.stat(), backing.stat()
    require((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), 'Bind target does not match authenticated backing')
    return target


def validate_m01(spec, target):
    run = Path(spec['run_dir'])
    task = Path(spec['m01_task_dir'])
    require(task.is_absolute() and task.resolve() == task and task.is_relative_to(run/'chr21/work'),
            'M01 task path escaped original work directory')
    require((task/'.exitcode').read_text().strip() == '0', 'Original M01 exitcode is not zero')
    marker = task/'dnabr.hg38.2723.chr21.m01.large_temp_dir.txt'
    directory = Path(marker.read_text().strip())
    require(directory.parent == target and re.fullmatch('m01_chr21\.[A-Za-z0-9]+', directory.name),
            'M01 marker differs from bulk mount')
    names = {'dnabr.hg38.2723.chr21.norm.vcf.gz', 'dnabr.hg38.2723.chr21.norm.vcf.gz.tbi',
             'dnabr.hg38.2723.chr21.norm.log'}
    records = spec['m01_files']
    require(len(records) == 3 and {Path(r['path']).name for r in records} == names,
            'Exactly the three completed M01 outputs must be authenticated')
    for record in records:
        path = Path(record['path'])
        require(path.parent == directory and path.is_file() and not path.is_symlink(), 'Unsafe M01 output')
        require(path.stat().st_size == record['bytes'] and path.stat().st_mtime_ns == record['mtime_ns']
                and sha(path) == record['sha256'], 'M01 content or modification time changed')
    # The frozen Runner cleanup visits markers from all attempts. Preserve their
    # directory names even when an abandoned partial GCS output was not copied.
    for prior_marker in (run/'chr21/work').glob('*/*/*.large_temp_dir.txt'):
        prior = Path(prior_marker.read_text().strip())
        require(prior.parent == target and re.fullmatch('m0[12]_chr21\.[A-Za-z0-9]+', prior.name)
                and prior.is_dir() and not prior.is_symlink(),
                'A previous temporary-directory marker is missing from the local overlay')
    return records


def boundary_lock(path):
    class Lock:
        def __enter__(self):
            self.stream = Path(path).open('a+')
            try:
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                self.stream.close()
                raise RuntimeError('Another boundary coordinator owns the lock')
            return self

        def __exit__(self, *_):
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
    return Lock()


def complete_preprocess(spec, p_spec, parallel, handoff, pipeline, spec_path):
    """Only the original Runner creates the real preprocessing checkpoint."""
    run = Path(spec['run_dir'])
    directory = Path(spec_path).parent
    evidence_path = directory/'completed.json'
    checkpoint = run/'checkpoints'/f'{STAGE}.json'
    if evidence_path.exists():
        evidence = load(evidence_path)
        require(evidence['recovery_spec_sha256'] == sha(spec_path)
                and evidence['checkpoint_sha256'] == sha(checkpoint), 'Recovery completion changed')
        pipeline.verify_files(load(checkpoint)['outputs'])
        return evidence
    target = validate_mount(spec)
    validate_m01(spec, target)
    require(not checkpoint.exists(), 'Unauthenticated preprocessing checkpoint already exists')
    with pipeline.execution_lock(run):
        validate_idle(spec, handoff, pipeline)
        runner = pipeline.Runner(run)
        runner.preprocess(21)
        require(checkpoint.is_file(), 'Original Runner did not create preprocessing checkpoint')
        receipt = load(checkpoint)
        source = run/'repairs/boundary-v2'
        hashes = load(source/'source.sha256.json')
        adapter = authenticated_module(source/'source/bin/r02_apply_amendment.py',
            hashes['bin/r02_apply_amendment.py'], '_local_amendment')
        require(receipt['command'] == adapter.expected_boundary_command(run, 21)
                and receipt['returncode'] == 0, 'Recovered command or exit status differs')
        pipeline.verify_files(receipt['outputs'])
        records = check_trace(run/'chr21/trace.tsv', recovered=True)
        evidence = dict(schema=SCHEMA, completed_utc=datetime.now(timezone.utc).isoformat(),
            recovery_spec_sha256=sha(spec_path), wrapper_sha256=spec['wrapper_sha256'],
            storage_receipt=spec['storage_receipt'], checkpoint_sha256=sha(checkpoint),
            original_processes=spec['old_processes'], trace=records,
            trace_sha256=sha(run/'chr21/trace.tsv'), command=receipt['command'],
            checkpoint_created_by='frozen_original_Runner.preprocess(21)',
            historical_blocked_fork_status_fabricated=False, scientific_parameters_changed=False,
            biological_validation_complete=False,
            cleanup_scope='Local bulk overlay only; original GCS intermediate copies remain retained')
        handoff.write_json(evidence_path, evidence, once=True)
        return evidence


def seal_boundary(spec, evidence, handoff):
    run = Path(spec['run_dir'])
    directory = run/'repairs/boundary-v2'
    request = load(directory/'request.json')
    amendment = copy.deepcopy(request['amendment_template'])
    amendment['boundary']['checkpoint_sha256'] = evidence['checkpoint_sha256']
    handoff.write_json(directory/'preprocess_storage_recovery.json', evidence, once=True)
    handoff.write_json(directory/'amendment.json', amendment, once=True)
    handoff.write_json(directory/'frozen.sha256.json', {name: sha(directory/name) for name in
        ('amendment.json', 'source.sha256.json', 'request.json', 'preprocess_storage_recovery.json')}, once=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--expected-spec-sha256', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    spec, p_spec, parallel, handoff, pipeline, _ = validate_spec(args.spec, args.expected_spec_sha256)
    run = Path(spec['run_dir'])
    validate_idle(spec, handoff, pipeline)
    if not (args.spec.parent/'completed.json').exists():
        validate_m01(spec, validate_mount(spec))
    if not args.run:
        print(json.dumps(dict(state='VERIFIED_NOT_EXECUTED', schema=SCHEMA)))
        return 0

    class Status:
        def state(self, state, **extra):
            payload = dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                           recovery_spec_sha256=args.expected_spec_sha256, **extra)
            handoff.write_json(args.spec.parent/'status.json', payload)
            handoff.write_json(run/'repairs/parallel-v1/coordinator_status.json', payload)
            print(json.dumps(payload), flush=True)

    status = Status()
    def expired(_sig, _frame):
        raise TimeoutError('Original parallel deadline reached during local recovery')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, parallel.timestamp(p_spec['deadline_utc']) - time.time()))
    with boundary_lock(run/'repairs/boundary-v2/.boundary.lock'):
        try:
            status.state('LOCAL_PREPROCESS_RESUMING')
            evidence = complete_preprocess(spec, p_spec, parallel, handoff, pipeline, args.spec)
            seal_boundary(spec, evidence, handoff)
            provenance = run/'repairs/parallel-v1/provenance'
            provenance.mkdir(exist_ok=True)
            for name, value in [('local_preprocess_recovery.json', evidence),
                                ('local_preprocess_recovery_spec.json', spec),
                                ('local_preprocess_storage_receipt.json', load(spec['storage_receipt']['path']))]:
                handoff.write_json(provenance/name, value, once=True)
            status.state('LOCAL_PREPROCESS_RECOVERED', checkpoint_sha256=evidence['checkpoint_sha256'])
            parallel.run_after_boundary(p_spec, run/'repairs/parallel-v1', status)
        except BaseException as error:
            status.state('FAILED', error=str(error))
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
