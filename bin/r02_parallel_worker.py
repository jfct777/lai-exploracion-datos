#!/usr/bin/env python3
"""Execute a fixed subset of R02 autosomes using the authenticated protocol-2 Runner.

This program never aggregates chromosomes, chooses scientific parameters, creates
VMs, or modifies the ongoing serial run. Preparation writes a new private worker
directory; execution keeps Nextflow and the existing chromosome implementation.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import types


PROTOCOL_DELTA = {
    'analysis_protocol_version': 2,
    'm14_skip_legacy_windows': True,
    'evaluate_all_m14_configurations': True,
    'm14_diagnostic_edge_thresholds_bp': [0, 250000, 500000, 750000, 1000000],
    'm14_diagnostic_kinship_thresholds': [.0221, .0442],
}
COPIED_INPUTS = ('samples.txt', 'input_objects.json', 'h_settings.json',
                 'weighted_settings_primary.json', 'weighted_settings_sensitivity.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_manifest(root, manifest):
    root = Path(root).resolve()
    require(isinstance(manifest, dict) and bool(manifest), 'Empty hash manifest')
    for name, digest in manifest.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Unsafe manifest path')
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root), 'Missing or unsafe frozen file: ' + name)
        require(re.fullmatch(r'[a-f0-9]{64}', digest) is not None
                and sha(path) == digest, 'Frozen file changed: ' + name)


def chromosomes(value):
    tokens = value.split(',') if isinstance(value, str) else value
    try:
        result = [int(item) for item in tokens]
    except (ValueError, TypeError):
        raise ValueError('Chromosomes must be a comma-separated integer list') from None
    require(result and len(set(result)) == len(result), 'Empty or repeated chromosomes')
    require(all(1 <= item <= 20 for item in result),
            'Workers accept only chr1–20; chr21 and chr22 remain with the existing controller')
    return result


def method_ast(path, class_name, method):
    tree = ast.parse(Path(path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
    return ast.dump(node, include_attributes=False)


def load_pipeline(source):
    path = Path(source) / 'bin/r02_autosome_pipeline.py'
    module = types.ModuleType('_r02_frozen_worker_pipeline')
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def prepare(parent, snapshot, run, worker_id, assigned, destination, bulk):
    """Copy immutable source/settings only; no GCS reads, writes or cohort computation."""
    parent, snapshot, run = (Path(value).resolve() for value in (parent, snapshot, run))
    destination, bulk = Path(destination), Path(bulk)
    assigned = chromosomes(assigned)
    require(re.fullmatch(r'[a-z][a-z0-9-]{1,45}', worker_id) is not None, 'Unsafe worker identifier')
    require(not run.exists(), 'Worker directory already exists; refusing overwrite')
    require(not run.is_relative_to(parent) or run == parent/'parallel'/worker_id,
            'Nested workers must use the dedicated parallel/worker-id directory')
    frozen = read_json(parent/'frozen.sha256.json')
    verify_manifest(parent, frozen)
    old_source = read_json(parent/'source.sha256.json')
    verify_manifest(parent/'source', old_source)
    original = read_json(parent/'run.json')
    require(original['chromosomes'] == list(range(1, 23)), 'Parent is not the all-autosome R02 run')
    require(sha(parent/'samples.txt') == original['samples_sha256'], 'Analytical sample list changed')
    for c in assigned:
        require(not (parent/f'chr{c:02d}').exists()
                and not any((parent/'checkpoints').glob(f'chr{c:02d}_*')),
                f'chr{c}: existing controller already prepared or processed this chromosome')
    require(destination.is_absolute() and destination == Path(original['destination'])/'parallel'/worker_id,
            'Destination must be the private parent-run parallel/worker-id subtree')
    require(bulk.is_absolute() and bulk == Path(original['bulk'])/'parallel'/worker_id,
            'Bulk must be the parent-run transient parallel/worker-id subtree')

    # Authenticate the same amendment already staged at the chr21 boundary.
    request = read_json(snapshot/'request.json')
    template = request['amendment_template']
    require(template['run_dir'] == str(parent)
            and template['original_frozen_sha256'] == sha(parent/'frozen.sha256.json')
            and template['original_source_manifest_sha256'] == sha(parent/'source.sha256.json'),
            'Source amendment belongs to a different serial run')
    require(template['overrides'] == PROTOCOL_DELTA, 'Scientific amendment differs from protocol 2')
    require(request['amendment_source_manifest_sha256'] == sha(snapshot/'source.sha256.json'),
            'Amendment source manifest changed')
    source_manifest = read_json(snapshot/'source.sha256.json')
    verify_manifest(snapshot/'source', source_manifest)
    for name, digest in old_source.items():
        if name.startswith(('modules/', 'workflows/')):
            require(source_manifest.get(name) == digest, 'Existing workflow/module changed: ' + name)
    require(method_ast(parent/'source/bin/r02_autosome_pipeline.py', 'Runner', 'preprocess')
            == method_ast(snapshot/'source/bin/r02_autosome_pipeline.py', 'Runner', 'preprocess'),
            'Preprocessing implementation changed')
    require(original['container_cpus'] == 6 and original['container_memory_gib'] == 24,
            'Worker requires the existing 6 CPU / 24 GiB task contract')

    run.mkdir(parents=True, mode=0o700)
    (run/'source').mkdir()
    for relative in source_manifest:
        target = run/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot/'source'/relative, target)
    write_new(run/'source.sha256.json', source_manifest)
    for name in COPIED_INPUTS:
        shutil.copy2(parent/name, run/name)
    shutil.copy2(Path(__file__), run/'worker.py')
    configuration = {**original, **PROTOCOL_DELTA,
                     'run_id': original['run_id'] + '-' + worker_id,
                     'created_utc': datetime.now(timezone.utc).isoformat(),
                     'destination': str(destination), 'bulk': str(bulk),
                     'processing_order': assigned,
                     'parallel_worker': dict(schema_version=1, worker_id=worker_id,
                         parent_run_id=original['run_id'], assigned_chromosomes=assigned,
                         parent_frozen_sha256=sha(parent/'frozen.sha256.json'),
                         parent_config_sha256=sha(parent/'run.json'),
                         amendment_request_sha256=sha(snapshot/'request.json'),
                         source_snapshot_manifest_sha256=sha(snapshot/'source.sha256.json'),
                         aggregate_allowed=False)}
    write_new(run/'run.json', configuration)
    pipeline = load_pipeline(run/'source')
    write_new(run/'m14_diagnostic_settings.json', pipeline.all_m14_diagnostic_settings(
        configuration['m14'], configuration['analytical_samples']))
    write_new(run/'frozen.sha256.json', {name: sha(run/name) for name in
        (*COPIED_INPUTS, 'worker.py', 'run.json', 'source.sha256.json', 'm14_diagnostic_settings.json')})
    (run/'logs').mkdir()
    (run/'checkpoints').mkdir()
    pipeline.status(run, 'WORKER_PREPARED_NOT_STARTED', assigned_chromosomes=assigned)
    return configuration


def execute(run):
    """Execute assigned chromosomes only. All-autosome aggregation belongs to the coordinator."""
    run = Path(run).resolve()
    verify_manifest(run, read_json(run/'frozen.sha256.json'))
    require(sha(Path(__file__)) == sha(run/'worker.py'), 'Worker executable differs from frozen copy')
    verify_manifest(run/'source', read_json(run/'source.sha256.json'))
    configuration = read_json(run/'run.json')
    worker = configuration['parallel_worker']
    assigned = chromosomes(worker['assigned_chromosomes'])
    require(configuration['processing_order'] == assigned and worker['aggregate_allowed'] is False,
            'Worker assignment or aggregation policy changed')
    require(all(configuration.get(key) == value for key, value in PROTOCOL_DELTA.items()),
            'Worker is not using protocol 2')
    pipeline = load_pipeline(run/'source')
    with pipeline.execution_lock(run):
        runner = pipeline.Runner(run)
        try:
            # M01/M02 require their explicitly assigned transient parent to
            # exist. This creates only this worker's authorized prefix, never
            # an input or historical-results directory.
            bulk = Path(configuration['bulk'])
            require(bulk.is_absolute() and bulk.name == worker['worker_id']
                    and bulk.parent.name == 'parallel' and not bulk.is_symlink(),
                    'Unsafe worker bulk directory')
            bulk.mkdir(parents=True, exist_ok=True)
            for c in assigned:
                runner.analyze_chromosome(c)
            records = []
            for c in assigned:
                complete = read_json(run/'checkpoints'/f'chr{c:02d}_complete.json')
                pipeline.verify_files(complete['outputs'])
                relative = f'01_estructura/desarrollo/por_cromosoma/chr{c:02d}'
                publication = run/'checkpoints'/('publish_'+relative.replace('/', '_')+'.json')
                published = read_json(publication)
                # Recheck cloud generations even when a local completion checkpoint exists.
                runner.publish(run/f'chr{c:02d}', relative)
                outputs = [{**item, 'relative_path': str(Path(item['path']).relative_to(run/f'chr{c:02d}'))}
                           for item in published['files']]
                records.append(dict(chromosome=c, outputs=outputs,
                    completion_sha256=sha(run/'checkpoints'/f'chr{c:02d}_complete.json'),
                    publication_sha256=sha(publication)))
            provenance = run/'worker_provenance'
            provenance.mkdir(exist_ok=True)
            for name in (*COPIED_INPUTS, 'run.json', 'source.sha256.json', 'frozen.sha256.json',
                         'worker.py', 'm14_diagnostic_settings.json'):
                target = provenance/name
                if target.exists():
                    require(sha(target) == sha(run/name), 'Existing worker provenance changed')
                else:
                    shutil.copy2(run/name, target)
            receipt = dict(schema_version=1, worker_id=worker['worker_id'],
                           parent_run_id=worker['parent_run_id'], chromosomes=assigned,
                           analysis_protocol_version=2, source_manifest_sha256=sha(run/'source.sha256.json'),
                           worker_run_sha256=sha(run/'run.json'), worker_frozen_sha256=sha(run/'frozen.sha256.json'),
                           samples_sha256=configuration['samples_sha256'], artifacts=records,
                           aggregate_executed=False, biological_validation_complete=False)
            receipt_path = provenance/'worker_completion.json'
            if receipt_path.exists():
                require(read_json(receipt_path) == receipt, 'Worker completion receipt changed')
            else:
                write_new(receipt_path, receipt)
            runner.publish(provenance, '00_worker_provenance')
            pipeline.status(run, 'WORKER_COMPLETE_PUBLISHED', state='COMPLETE',
                            assigned_chromosomes=assigned, aggregate_executed=False)
        except BaseException as error:
            pipeline.status(run, 'WORKER_FAILED', state='FAILED', failed_stage=runner.current_stage,
                            error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--parent-run', type=Path)
    parser.add_argument('--source-snapshot', type=Path)
    parser.add_argument('--worker-id')
    parser.add_argument('--chromosomes')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--bulk', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == 'prepare':
        required = ('parent_run', 'source_snapshot', 'worker_id', 'chromosomes', 'destination', 'bulk')
        if any(getattr(args, name) is None for name in required):
            parser.error('prepare requires parent-run, source-snapshot, worker-id, chromosomes, destination and bulk')
        prepare(args.parent_run, args.source_snapshot, args.run_dir, args.worker_id,
                args.chromosomes, args.destination, args.bulk)
    else:
        execute(args.run_dir)


if __name__ == '__main__':
    main()
