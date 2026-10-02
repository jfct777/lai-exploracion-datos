#!/usr/bin/env python3
"""Apply an authenticated preprocessing-only amendment to an existing R02 worker.

No signals, VM operations, parameter selection or historical-file mutation are
performed here. A separately authenticated handoff must stop the old startup
shell, supervisor and child before --run. The original Runner finishes legacy
chromosomes and all scientific analyses. Only preprocessing of explicitly new
chromosomes uses the new frozen source, with original paths and sample lists.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import types


SCHEMA = 'r02_optimized_worker_v1'
AMENDMENT_SCHEMA = 'r02_preprocess_operational_amendment_v1'
COUNT_SCHEMA = 'r02_optimized_worker_v2'
COUNT_AMENDMENT_SCHEMA = 'r02_preprocess_operational_amendment_v2'
OVERRIDES = dict(preprocess_checkpointed_m01=True,
                 preprocess_scratch_input_multiplier=3, preprocess_scratch_reserve_gib=32)
ALLOWED_SOURCE_DELTA = frozenset({
    'bin/r02_autosome_pipeline.py', 'bin/mark_original_alleles.py',
    'bin/preprocess_storage_guard.py', 'bin/preprocess_audit.py',
    'modules/01_preprocess_checkpointed.nf', 'workflows/r02_preprocess_autosome.nf',
})
SPEC_FIELDS = frozenset({
    'schema', 'run_dir', 'worker_id', 'wrapper_sha256', 'original_run_sha256',
    'original_frozen_sha256', 'original_source_manifest_sha256', 'source_manifest_sha256',
    'new_preprocess_chromosomes', 'legacy_chromosomes', 'overrides', 'old_processes',
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 1024**2,
            'Unsafe or oversized metadata file: ' + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def fixed(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Existing amendment evidence differs: ' + str(path))
    else:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')


def safe_file(root, relative):
    root, relative = Path(root), Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
            'Unsafe relative source path')
    path = root / relative
    require(path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]
                                     if p == root or root in p.parents), 'Unsafe source file')
    require(path.resolve().is_relative_to(root.resolve()), 'Source escaped its root')
    return path


def verify_manifest(root, values):
    require(isinstance(values, dict) and values, 'Empty source manifest')
    for name, expected in values.items():
        require(is_digest(expected) and sha(safe_file(root, name)) == expected,
                'Source hash changed: ' + str(name))


def load_module(path, expected, name):
    require(is_digest(expected) and sha(path) == expected, 'Module hash differs')
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), result.__dict__)
    return result


def validate_pipeline_delta(old_path, new_path):
    """Do not allow operational packaging to change downstream algorithms."""
    old, new = [ast.parse(Path(path).read_text()) for path in (old_path, new_path)]
    def functions(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    a, b = functions(old), functions(new)
    for name in (a.keys() | b.keys()) - {'prepare', 'preprocessing_resources'}:
        require(a.get(name) == b.get(name), 'Unexpected top-level pipeline change: ' + name)
    def methods(tree):
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Runner']
        require(len(classes) == 1, 'Expected one Runner class')
        return {n.name: ast.dump(n, include_attributes=False) for n in classes[0].body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    a, b = methods(old), methods(new)
    for name in (a.keys() | b.keys()) - {'preprocess', 'cleanup_bulk'}:
        require(a.get(name) == b.get(name), 'Scientific Runner method changed: ' + name)
    require('preprocess' in b and 'preprocessing_resources' in functions(new),
            'New source does not implement checkpointed preprocessing')


def validate_identity(value):
    require(isinstance(value, dict) and set(value) == {'pid', 'start_ticks', 'cmdline_sha256'}
            and type(value['pid']) is int and value['pid'] > 1
            and type(value['start_ticks']) is int and value['start_ticks'] > 0
            and is_digest(value['cmdline_sha256']), 'Invalid old process identity')


def old_process_live(value):
    """Check the sealed identity; never signal a process or mistake PID reuse for ownership."""
    validate_identity(value)
    directory = Path('/proc') / str(value['pid'])
    try:
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if int(fields[19]) != value['start_ticks'] or fields[0] in {'Z', 'X'}:
            return False
        command = (directory / 'cmdline').read_bytes()
    except FileNotFoundError:
        return False
    require(hashlib.sha256(command).hexdigest() == value['cmdline_sha256'],
            'Old process command changed without PID/start-time change')
    return True


def validate_idle(spec, configuration):
    for name, identity in spec['old_processes'].items():
        require(not old_process_live(identity), 'Old process remains alive: ' + name)
    result = subprocess.check_output(['docker', 'ps', '-q', '--filter',
        'label=dnabr-r02=' + configuration['run_id']], text=True, timeout=30)
    require(not result.strip(), 'Worker still has active containers')


def validate_spec(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and is_digest(expected)
            and sha(path) == expected, 'Amendment manifest hash/path differs')
    spec = read(path)
    repaired = spec.get('schema') == COUNT_SCHEMA
    extra = {'count_validator_sha256', 'previous_manifest_sha256'} if repaired else set()
    require(set(spec) == SPEC_FIELDS | extra and spec['schema'] in (SCHEMA, COUNT_SCHEMA),
            'Invalid amendment manifest schema')
    require(spec['wrapper_sha256'] == sha(__file__), 'Optimized wrapper changed')
    run = Path(spec['run_dir'])
    require(run.is_absolute() and run.resolve() == run and run.is_dir(), 'Invalid original run directory')
    version = 'preprocess-v2' if repaired else 'preprocess-v1'
    require(path == run/'repairs'/version/'manifest.json', 'Unexpected amendment directory')
    if repaired:
        require(is_digest(spec['count_validator_sha256'])
                and sha(safe_file(path.parent, 'preprocess_count_validation.py')) == spec['count_validator_sha256'],
                'Count validator changed')
        previous = run/'repairs/preprocess-v1/manifest.json'
        require(is_digest(spec['previous_manifest_sha256']) and sha(previous) == spec['previous_manifest_sha256'],
                'Previous preprocessing amendment changed')
        old = read(previous)
        require(old.get('schema') == SCHEMA, 'Expected version1 predecessor')
        for key in SPEC_FIELDS - {'schema', 'wrapper_sha256'}:
            require(old[key] == spec[key], 'Count repair changed previous settings: ' + key)
    for name, key in [('run.json', 'original_run_sha256'), ('frozen.sha256.json', 'original_frozen_sha256'),
                      ('source.sha256.json', 'original_source_manifest_sha256')]:
        require(is_digest(spec[key]) and sha(safe_file(run, name)) == spec[key], 'Original run changed: ' + name)
    original_seal = read(run/'frozen.sha256.json')
    verify_manifest(run, original_seal)
    configuration = read(run/'run.json')
    worker = configuration['parallel_worker']
    require(spec['worker_id'] == worker['worker_id']
            and re.fullmatch(r'worker(?:0[1-9]|1[0-2])', spec['worker_id']), 'Worker identity differs')
    assigned = worker['assigned_chromosomes']
    require(worker['aggregate_allowed'] is False and assigned == configuration['processing_order'],
            'Worker role or processing order changed')
    new, legacy = spec['new_preprocess_chromosomes'], spec['legacy_chromosomes']
    for values in (assigned, new, legacy):
        require(isinstance(values, list) and all(type(c) is int for c in values)
                and len(values) == len(set(values)), 'Invalid chromosome partition')
    require(assigned and legacy and new == sorted(new) and legacy == sorted(legacy)
            and set(new).isdisjoint(legacy) and set(new) | set(legacy) == set(assigned),
            'Chromosomes must form one disjoint exhaustive partition')
    require(all(13 <= c <= 20 for c in new) and all(1 <= c <= 12 for c in legacy)
            and assigned == legacy + new, 'Amendment only applies after the original active chromosome')
    require(spec['overrides'] == OVERRIDES, 'Only the explicit operational settings are permitted')
    require(set(spec['old_processes']) == {'supervisor', 'child', 'startup'}, 'Three old process identities required')
    for identity in spec['old_processes'].values():
        validate_identity(identity)
    require(len({p['pid'] for p in spec['old_processes'].values()}) == 3, 'Old process identities overlap')
    base_source = read(run/'source.sha256.json')
    verify_manifest(run/'source', base_source)
    new_manifest = path.parent/'source.sha256.json'
    require(is_digest(spec['source_manifest_sha256']) and sha(new_manifest) == spec['source_manifest_sha256'],
            'New source manifest changed')
    new_source = read(new_manifest)
    verify_manifest(path.parent/'source', new_source)
    require(set(base_source) <= set(new_source), 'New snapshot removed original source files')
    changed = sorted(name for name in new_source if new_source[name] != base_source.get(name))
    require(changed and set(changed) <= ALLOWED_SOURCE_DELTA, 'Source delta exceeds preprocessing allowlist')
    require(ALLOWED_SOURCE_DELTA <= new_source.keys(), 'Missing operational preprocessing source')
    validate_pipeline_delta(run/'source/bin/r02_autosome_pipeline.py',
                            path.parent/'source/bin/r02_autosome_pipeline.py')
    activation = path.parent/'activation.json'
    if activation.exists():
        require(read(activation)['manifest_sha256'] == expected, 'Prior activation belongs to a different amendment')
    else:
        for c in new:
            require(not (run/f'chr{c:02d}').exists()
                    and not list((run/'checkpoints').glob(f'chr{c:02d}_*'))
                    and not list((run/'stages').glob(f'chr{c:02d}_*')),
                    'New preprocessing chromosome already has execution state')
        require(not (run/'worker_provenance/worker_completion.json').exists(),
                'Worker already has a final completion receipt')
    require(sha(run/'samples.txt') == configuration['samples_sha256'], 'Analytical sample list changed')
    return spec, configuration, base_source, new_source, changed


def build_runner(run, directory, spec, old_pipeline, new_pipeline):
    """Restore source/configuration even on failure; downstream calls always use the original code."""
    class OperationalRunner(old_pipeline.Runner):
        def preprocess(self, chromosome):
            if chromosome not in spec['new_preprocess_chromosomes']:
                require(chromosome in spec['legacy_chromosomes'], 'Unassigned preprocessing chromosome')
                return super().preprocess(chromosome)
            saved = self.source, self.bin, self.c
            try:
                self.source = directory/'source'
                self.bin = self.source/'bin'
                self.c = copy.deepcopy(self.c)
                self.c.update(spec['overrides'])
                return new_pipeline.Runner.preprocess(self, chromosome)
            finally:
                self.source, self.bin, self.c = saved
    return OperationalRunner(run)


def copy_unchanged(source, target):
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink(), 'Unsafe provenance source')
    if target.exists():
        require(target.is_file() and not target.is_symlink() and sha(source) == sha(target),
                'Existing provenance changed')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)


def operational_evidence(runner, directory, manifest_path, spec, changed):
    folder = directory/'publication'
    folder.mkdir(exist_ok=True)
    for name, source in [('manifest.json', manifest_path), ('source.sha256.json', directory/'source.sha256.json'),
                         ('optimized_worker.py', Path(__file__)), ('activation.json', directory/'activation.json')]:
        copy_unchanged(source, folder/name)
    for relative in changed:
        copy_unchanged(directory/'source'/relative, folder/'source'/relative)
    if spec['schema'] == COUNT_SCHEMA:
        copy_unchanged(directory/'preprocess_count_validation.py', folder/'preprocess_count_validation.py')
        copy_unchanged(runner.run/'repairs/preprocess-v1/manifest.json', folder/'previous_manifest.json')
    relative = '00_worker_provenance/operational_amendments/' + sha(manifest_path)
    runner.publish(folder, relative)
    receipt = read(runner.run/'checkpoints'/('publish_' + relative.replace('/', '_') + '.json'))
    return [{key: item[key] for key in ('uri', 'generation', 'bytes', 'sha256', 'md5_base64')}
            for item in receipt['files']]


def validation_delta(spec):
    return ({key: spec[key] for key in ('count_validator_sha256', 'previous_manifest_sha256')}
            if spec['schema'] == COUNT_SCHEMA else {})


def bind_count_validator(directory, spec, *pipelines):
    """Explicit authenticated runtime delta; frozen source bytes stay intact."""
    if spec['schema'] != COUNT_SCHEMA:
        return None
    helper = load_module(Path(directory)/'preprocess_count_validation.py',
                         spec['count_validator_sha256'], '_repaired_record_count')
    for pipeline in pipelines:
        pipeline.validate_raw_record_count = helper.validate_raw_record_count
    return helper


def validate_legacy_count(manifest_path, expected, chromosome):
    spec, _, _, _, _ = validate_spec(manifest_path, expected)
    require(chromosome in spec['legacy_chromosomes'] and spec['schema'] == COUNT_SCHEMA,
            'Only a legacy chromosome with an explicit count repair can be checked')
    helper = bind_count_validator(Path(manifest_path).parent, spec)
    audit = helper.validate_raw_record_count(Path(spec['run_dir'])/f'chr{chromosome:02d}', chromosome)
    return dict(chromosome=chromosome, legacy_count_validated=True, **audit)


def run_worker(manifest_path, expected):
    spec, configuration, base_source, new_source, changed = validate_spec(manifest_path, expected)
    run, directory = Path(spec['run_dir']), Path(manifest_path).parent
    frozen_worker = load_module(run/'worker.py', read(run/'frozen.sha256.json')['worker.py'], '_original_worker')
    require(all(configuration.get(k) == v for k, v in frozen_worker.PROTOCOL_DELTA.items()),
            'Worker no longer uses scientific protocol 2')
    old = load_module(run/'source/bin/r02_autosome_pipeline.py', base_source['bin/r02_autosome_pipeline.py'],
                      '_original_pipeline')
    new = load_module(directory/'source/bin/r02_autosome_pipeline.py', new_source['bin/r02_autosome_pipeline.py'],
                      '_optimized_preprocess_pipeline')
    bind_count_validator(directory, spec, old, new)
    with old.execution_lock(run):
        validate_idle(spec, configuration)
        runner = build_runner(run, directory, spec, old, new)
        amendment_schema = COUNT_AMENDMENT_SCHEMA if spec['schema'] == COUNT_SCHEMA else AMENDMENT_SCHEMA
        fixed(directory/'activation.json', dict(schema=amendment_schema, manifest_sha256=expected,
            original_source_manifest_sha256=spec['original_source_manifest_sha256'],
            new_source_manifest_sha256=spec['source_manifest_sha256'], changed_source_files=changed,
            operational_only=True, biological_validation_complete=False, **validation_delta(spec)))
        try:
            operational_provenance = operational_evidence(runner, directory, manifest_path, spec, changed)
            for c in configuration['processing_order']:
                runner.analyze_chromosome(c)
            artifacts = []
            for c in configuration['processing_order']:
                complete = run/'checkpoints'/f'chr{c:02d}_complete.json'
                old.verify_files(read(complete)['outputs'])
                relative = f'01_estructura/desarrollo/por_cromosoma/chr{c:02d}'
                publication = run/'checkpoints'/('publish_' + relative.replace('/', '_') + '.json')
                runner.publish(run/f'chr{c:02d}', relative)
                outputs = [{**item, 'relative_path': str(Path(item['path']).relative_to(run/f'chr{c:02d}'))}
                           for item in read(publication)['files']]
                artifacts.append(dict(chromosome=c, outputs=outputs, completion_sha256=sha(complete),
                                      publication_sha256=sha(publication)))
            provenance = run/'worker_provenance'
            provenance.mkdir(exist_ok=True)
            for name in (*frozen_worker.COPIED_INPUTS, 'run.json', 'source.sha256.json', 'frozen.sha256.json',
                         'worker.py', 'm14_diagnostic_settings.json'):
                copy_unchanged(run/name, provenance/name)
            worker = configuration['parallel_worker']
            amendment = dict(schema=amendment_schema, manifest_sha256=expected,
                source_manifest_sha256=spec['source_manifest_sha256'],
                new_preprocess_chromosomes=spec['new_preprocess_chromosomes'],
                legacy_chromosomes=spec['legacy_chromosomes'], **validation_delta(spec))
            receipt = dict(schema_version=1, worker_id=worker['worker_id'], parent_run_id=worker['parent_run_id'],
                chromosomes=configuration['processing_order'], analysis_protocol_version=2,
                source_manifest_sha256=spec['original_source_manifest_sha256'],
                worker_run_sha256=spec['original_run_sha256'], worker_frozen_sha256=spec['original_frozen_sha256'],
                samples_sha256=configuration['samples_sha256'], artifacts=artifacts,
                aggregate_executed=False, biological_validation_complete=False,
                operational_amendment=amendment, operational_provenance=operational_provenance)
            fixed(provenance/'worker_completion.json', receipt)
            runner.publish(provenance, '00_worker_provenance')
            publication_checkpoint = run/'checkpoints/publish_00_worker_provenance.json'
            publication = read(publication_checkpoint)
            completion_path = provenance/'worker_completion.json'
            published = [item for item in publication['files']
                         if item.get('path') == str(completion_path)]
            completion_bytes = completion_path.read_bytes()
            require(len(published) == 1 and published[0]['sha256'] == sha(completion_path)
                    and published[0]['bytes'] == len(completion_bytes)
                    and published[0]['md5_base64'] == base64.b64encode(
                        hashlib.md5(completion_bytes, usedforsecurity=False).digest()).decode()
                    and re.fullmatch(r'[0-9]+', str(published[0]['generation'])),
                    'Final completion is missing from verified publication')
            old.status(run, 'WORKER_COMPLETE_PUBLISHED', state='COMPLETE',
                assigned_chromosomes=configuration['processing_order'], aggregate_executed=False,
                operational_amendment_sha256=expected)
            fixed(directory/'completed.json', dict(manifest_sha256=expected, published=True,
                worker_completion_sha256=sha(completion_path),
                publication_checkpoint_sha256=sha(publication_checkpoint)))
            return receipt
        except BaseException as error:
            old.status(run, 'WORKER_FAILED', state='FAILED', failed_stage=runner.current_stage, error=str(error),
                       operational_amendment_sha256=expected)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--validate-legacy-count', type=int)
    args = parser.parse_args()
    os.umask(0o077)
    if args.validate_legacy_count is not None:
        require(not args.run, 'Read-only validation cannot also execute the worker')
        print(json.dumps(validate_legacy_count(args.manifest, args.manifest_sha256, args.validate_legacy_count)))
    elif args.run:
        run_worker(args.manifest, args.manifest_sha256)
    else:
        spec, _, _, _, changed = validate_spec(args.manifest, args.manifest_sha256)
        print(json.dumps(dict(status='VERIFIED_NOT_EXECUTED', worker_id=spec['worker_id'],
                              changed_source_files=changed,
                              new_preprocess_chromosomes=spec['new_preprocess_chromosomes'])))


if __name__ == '__main__':
    main()
