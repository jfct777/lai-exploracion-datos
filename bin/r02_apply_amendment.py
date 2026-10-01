#!/usr/bin/env python3
"""Apply one frozen R02 protocol-2 delta at an authenticated preprocessing boundary.

No preparation, copying, stopping processes, cloud access or execution is done by
default. --run acquires the original execution lock, writes additive provenance
and continues its existing destination/checkpoints using the frozen new Runner.

amendment.json schema (all paths are absolute except source manifest entries):
  schema_version: 1; run_dir; original_frozen_sha256;
  original_source_manifest_sha256; overrides: exactly PROTOCOL_DELTA below;
  boundary: {chromosome, stage, checkpoint_sha256, parameters_sha256,
             runtime_config_sha256}.
source.sha256.json maps relative source paths to SHA256. frozen.sha256.json
authenticates amendment.json, source.sha256.json and optionally the request. The watcher may
create these final files only after the genuine successful boundary checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import types


PROTOCOL_DELTA = {
    'analysis_protocol_version': 2,
    'm14_skip_legacy_windows': True,
    'evaluate_all_m14_configurations': True,
    'm14_diagnostic_edge_thresholds_bp': [0, 250000, 500000, 750000, 1000000],
    'm14_diagnostic_kinship_thresholds': [.0221, .0442],
}
PROTECTED = (
    'workflows/r02_preprocess_autosome.nf', 'workflows/r02_analysis_task.nf',
    'modules/01_preprocess_norm_leftalign.nf',
    'modules/02_preprocess_filter_snv_biallelic_pass.nf',
    'modules/lai_rare_bialelic_only.nf',
    'bin/mark_original_alleles.py', 'bin/select_rare_minor.py',
)
SUFFIX = '__r02v2'
ANALYSIS_TOOLS = (
    'r02_exec_task.py', 'r02_genomic_pair_evidence.py', 'r02_common_grm.sh',
    'rare_allele_sharing_painter.py', 'rare_segment_sensitivity.py',
    'm165_autosome_sweep.py', 'm165_chr22_sweep.py', 'm165_spectral_figures.py',
    'm165_sweep_figures.py', 'm165_graph_kinship.py', 'ibd_community_enhanced.py',
    'r02_weighted_communities.py', 'r02_weighted_kinship.py',
    'r02_m14_configuration_diagnostics.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def safe_file(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
            'Unsafe manifest path')
    path = root / relative
    require(path.is_file(), 'Missing manifest file: ' + str(relative))
    require(all(not item.is_symlink() for item in [path, *path.parents]
                if item == root or root in item.parents), 'Symlink in frozen manifest')
    require(path.resolve().is_relative_to(root.resolve()), 'Manifest escaped its root')
    return path


def verify_manifest(root, manifest):
    require(isinstance(manifest, dict) and manifest, 'Empty or invalid hash manifest')
    for name, digest in manifest.items():
        require(isinstance(digest, str) and re.fullmatch(r'[a-f0-9]{64}', digest),
                'Invalid SHA256')
        require(sha(safe_file(root, name)) == digest, 'Frozen file changed: ' + name)


def write_fixed(path, value):
    if path.exists():
        require(not path.is_symlink() and load_json(path) == value,
                'Existing amendment evidence changed: ' + path.name)
        return
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def method_ast(path, class_name, method):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return ast.dump(node, include_attributes=False)


def expected_preprocess_parameters(run, configuration, chromosome):
    folder = run / f'chr{chromosome:02d}'
    return dict(outdir=str(folder/'preprocess'), cpus=6, memory='24 GB', time='72h',
        resources={name: {'threads': 6} for name in (
            'preprocess_norm_leftalign', 'preprocess_filter_snv_biallelic_pass', 'lai_rare_bialelic_only')},
        r02_chrom=chromosome,
        r02_raw_vcf=str(Path(configuration['raw_dir'])/f'dnabr.hg38.2723.chr{chromosome}.vcf.gz'),
        r02_bin_dir=str(run/'source/bin'), ref_fasta=configuration['ref'],
        preprocess_large_temp_dir=configuration['bulk'], bcftools_min_alleles=2,
        plink_max_alleles=2, plink_snps_only=True, max_maf=None, keep_pass=True,
        lai_rare_max_maf=.01, lai_rare_min_mac=2, lai_rare_keep_format='GT', lai_rare_remove_info=False)


def expected_boundary_command(run, chromosome):
    folder = run/f'chr{chromosome:02d}'
    return [str(x) for x in ('/usr/local/bin/nextflow', '-log', folder/'nextflow.log',
        '-C', folder/'runtime.config', 'run', run/'source/workflows/r02_preprocess_autosome.nf',
        '-params-file', folder/'parameters.json', '-work-dir', folder/'work',
        '-ansi-log', 'false', '-with-trace', folder/'trace.tsv', '-resume')]


def validate_boundary(runner, amendment, pipeline, *, initial):
    """Authenticate the actual successful checkpoint; never synthesize one."""
    run = runner.run
    boundary = amendment['boundary']
    require(set(boundary) == {'chromosome', 'stage', 'checkpoint_sha256',
                             'parameters_sha256', 'runtime_config_sha256'}, 'Invalid boundary schema')
    chromosome = boundary['chromosome']
    require(type(chromosome) is int and 1 <= chromosome <= 21, 'Invalid boundary chromosome')
    stage = f'chr{chromosome:02d}_M01_M02_M021'
    require(boundary['stage'] == stage, 'Boundary must be the preprocessing checkpoint')
    checkpoint = run/'checkpoints'/f'{stage}.json'
    require(checkpoint.is_file() and not checkpoint.is_symlink(), 'Boundary checkpoint not ready')
    require(sha(checkpoint) == boundary['checkpoint_sha256'], 'Boundary checkpoint hash changed')
    receipt = load_json(checkpoint)
    require(type(receipt.get('returncode')) is int and receipt['returncode'] == 0,
            'Boundary checkpoint is not successful')
    require(receipt.get('command') == expected_boundary_command(run, chromosome),
            'Boundary checkpoint command changed')
    require(bool(receipt.get('completed_utc')), 'Boundary checkpoint lacks completion time')
    folder = run/f'chr{chromosome:02d}'
    for filename, field in [('parameters.json', 'parameters_sha256'),
                            ('runtime.config', 'runtime_config_sha256')]:
        require(sha(safe_file(folder, filename)) == boundary[field], 'Boundary settings hash changed')
    require(load_json(folder/'parameters.json') == expected_preprocess_parameters(run, runner.c, chromosome),
            'Preprocessing parameters differ from original protocol')
    base = f'dnabr.hg38.2723.chr{chromosome}.rare.minor'
    expected = {str(folder/'preprocess/lai_rare'/f'{base}{suffix}')
                for suffix in ('.vcf.gz', '.vcf.gz.tbi', '.contract.json', '.counts.tsv')}
    records = receipt.get('outputs', [])
    require(isinstance(records, list) and len(records) == len(expected)
            and {item.get('path') for item in records} == expected,
            'Boundary does not authenticate the four durable rare outputs')
    pipeline.verify_files(records)
    order = runner.c['processing_order']
    require(isinstance(order, list) and len(order) == 22 and set(order) == set(range(1, 23)),
            'Original run is not an all-autosome run')
    previous = order[:order.index(chromosome)]
    completed = []
    for prior in previous:
        done = run/'checkpoints'/f'chr{prior:02d}_complete.json'
        require(done.is_file(), 'Previous chromosome is not complete')
        payload = load_json(done)
        require(payload.get('chromosome') == prior and payload.get('outputs'),
                'Previous chromosome completion is unauthenticated')
        pipeline.verify_files(payload['outputs'])
        completed.append(dict(chromosome=prior, checkpoint_sha256=sha(done)))
    if initial:
        remaining = order[order.index(chromosome):]
        for item in (run/'checkpoints').iterdir():
            if item.name.startswith('all22_') or item.name.startswith('publish_01_estructura_desarrollo_aggregate'):
                raise ValueError('Aggregate checkpoint already exists')
            if any(item.name.startswith(f'chr{c:02d}_')
                   or item.name == f'publish_01_estructura_desarrollo_por_cromosoma_chr{c:02d}.json'
                   for c in remaining):
                require(item == checkpoint, 'Science or later chromosome checkpoint already exists')
        for c in remaining:
            cf = run/f'chr{c:02d}'
            for pattern in ('rare_evidence*', 'common*', 'anchor', 'sensitivity'):
                require(not any(cf.glob(pattern)), 'Scientific output already exists at boundary')
        for pattern in ('aggregate*', 'communities*', 'plots*', 'diagnostics*', 'autosome_inputs.json'):
            require(not any(run.glob(pattern)), 'Aggregate output already exists')
        stages = run/'stages'
        if stages.exists():
            for item in stages.iterdir():
                if item.name.startswith('all22_'):
                    raise ValueError('Aggregate task already prepared')
                if any(item.name.startswith(f'chr{c:02d}_') for c in remaining):
                    # Old supervisor can prepare rare_J settings before failing to fork.
                    require(item.name == f'chr{chromosome:02d}_rare_J' and item.is_dir(),
                            'Unexpected analysis task at boundary')
                    require(all(p.name in {'command.json', 'parameters.json', 'runtime.config'}
                                and p.is_file() and not p.is_symlink() for p in item.iterdir()),
                            'Old analysis task has execution evidence')
    return dict(boundary_checkpoint=str(checkpoint), boundary_sha256=sha(checkpoint),
                completed_chromosomes=completed)


def validate_snapshot(run, directory):
    """Only local immutable files are read; module code loads after hash checks."""
    require(run.is_dir() and directory.is_dir(), 'Run and amendment directories must exist')
    seal = load_json(safe_file(directory, 'frozen.sha256.json'))
    require({'amendment.json', 'source.sha256.json'} <= set(seal), 'Invalid amendment seal')
    verify_manifest(directory, seal)
    amendment = load_json(directory/'amendment.json')
    require(set(amendment) == {'schema_version', 'run_dir', 'original_frozen_sha256',
                             'original_source_manifest_sha256', 'boundary', 'overrides'},
            'Invalid amendment schema')
    require(amendment['schema_version'] == 1 and amendment['run_dir'] == str(run),
            'Amendment belongs to another run or schema')
    require(json.dumps(amendment['overrides'], sort_keys=True) == json.dumps(PROTOCOL_DELTA, sort_keys=True),
            'Only the explicit protocol-2 delta is allowed')
    require(sha(run/'frozen.sha256.json') == amendment['original_frozen_sha256'],
            'Original frozen manifest changed')
    require(sha(run/'source.sha256.json') == amendment['original_source_manifest_sha256'],
            'Original source manifest changed')
    old_manifest = load_json(run/'source.sha256.json')
    verify_manifest(run, load_json(run/'frozen.sha256.json'))
    verify_manifest(run/'source', old_manifest)
    source = directory/'source'
    new_manifest = load_json(directory/'source.sha256.json')
    verify_manifest(source, new_manifest)
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()
              and '__pycache__' not in p.relative_to(source).parts}
    require(actual == set(new_manifest), 'Snapshot has unauthenticated files')
    require(all('bin/' + name in new_manifest for name in ANALYSIS_TOOLS),
            'Frozen analysis tool missing')
    for name in PROTECTED:
        require(name in old_manifest and name in new_manifest and old_manifest[name] == new_manifest[name],
                'Preprocessing/workflow source changed: ' + name)
    for name in set(old_manifest) & set(new_manifest):
        if name.startswith(('modules/', 'workflows/')):
            require(old_manifest[name] == new_manifest[name], 'Existing workflow/module changed: ' + name)
    runner_path = source/'bin/r02_autosome_pipeline.py'
    require('bin/r02_autosome_pipeline.py' in new_manifest, 'Frozen new Runner missing')
    require(method_ast(run/'source/bin/r02_autosome_pipeline.py', 'Runner', 'preprocess')
            == method_ast(runner_path, 'Runner', 'preprocess'), 'Runner preprocessing implementation changed')
    require('bin/r02_apply_amendment.py' in new_manifest
            and new_manifest['bin/r02_apply_amendment.py'] == sha(Path(__file__)),
            'Adapter differs from the frozen snapshot')
    pipeline = types.ModuleType('_r02_amended_pipeline')
    pipeline.__file__ = str(runner_path)
    exec(compile(runner_path.read_bytes(), str(runner_path), 'exec'), pipeline.__dict__)
    delta = {name: {'before': old_manifest.get(name), 'after': digest}
             for name, digest in sorted(new_manifest.items()) if old_manifest.get(name) != digest}
    return amendment, pipeline, delta


def build_runner(run, directory, *, activate=False):
    run, directory = Path(run).resolve(), Path(directory).resolve()
    amendment, pipeline, delta = validate_snapshot(run, directory)

    class AmendedRunner(pipeline.Runner):
        def preprocess(self, chromosome):
            # Both existing and future M01/M02/M021 use unchanged original scripts.
            analysis_bin, self.bin = self.bin, self.source/'bin'
            try:
                return super().preprocess(chromosome)
            finally:
                self.bin = analysis_bin

        def docker(self, stage, args, **kwargs):
            require(not stage.endswith(SUFFIX), 'Stage namespace already amended')
            return super().docker(stage + SUFFIX, args, **kwargs)

    runner = AmendedRunner(run)  # Base validates original files, without changing any.
    evidence_path = directory/'activation.json'
    activated = evidence_path.exists()
    if activated:
        activation_seal = load_json(directory/'activation.sha256.json')
        require(set(activation_seal) == {'activation.json', 'resolved_configuration.json', 'code_delta.json'},
                'Invalid activation seal')
        verify_manifest(directory, activation_seal)
        require(load_json(evidence_path)['amendment_sha256'] == sha(directory/'amendment.json'),
                'Activation belongs to a different amendment')
    boundary_evidence = validate_boundary(runner, amendment, pipeline, initial=not activated)
    runner.c = dict(runner.c, **amendment['overrides'])
    runner.bin = directory/'source/bin'
    runner.env['PYTHONDONTWRITEBYTECODE'] = '1'
    settings = pipeline.all_m14_diagnostic_settings(runner.c['m14'], runner.c['analytical_samples'])
    require(len(settings['configurations']) == 74, 'Expected all 74 effective M14 configurations')
    root_settings = run/'m14_diagnostic_settings.json'
    if root_settings.exists():
        require(not root_settings.is_symlink() and load_json(root_settings) == settings,
                'Existing diagnostic settings conflict with amendment')
    evidence = dict(schema_version=1, status='AMENDMENT_ACTIVATED_NOT_BIOLOGICALLY_VALIDATED',
        amendment_sha256=sha(directory/'amendment.json'),
        amendment_seal_sha256=sha(directory/'frozen.sha256.json'),
        original_source=str(runner.source), analysis_bin=str(runner.bin),
        analysis_stage_suffix=SUFFIX, **boundary_evidence)
    if activated:
        require(load_json(evidence_path) == evidence, 'Activation evidence changed')
        require(load_json(directory/'resolved_configuration.json') == runner.c,
                'Resolved configuration changed')
        require(load_json(directory/'code_delta.json') == delta, 'Code delta changed')
    if activate:
        # Never edit root run.json, original settings, source, or their hash seals.
        write_fixed(directory/'resolved_configuration.json', runner.c)
        write_fixed(directory/'code_delta.json', delta)
        write_fixed(root_settings, settings)
        write_fixed(evidence_path, evidence)
        write_fixed(directory/'activation.sha256.json', {
            name: sha(directory/name) for name in ('activation.json', 'resolved_configuration.json', 'code_delta.json')})
    return runner, pipeline, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--amendment-dir', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Continue only after successful boundary validation')
    args = parser.parse_args(argv)
    os.umask(0o077)
    run, directory = args.run_dir.resolve(), args.amendment_dir.resolve()
    if not args.run:
        _, _, evidence = build_runner(run, directory)
        print(json.dumps(dict(status='VERIFIED_NOT_EXECUTED', **{k: v for k, v in evidence.items() if k != 'status'})))
        return
    # Load only authenticated code to obtain the original, non-stealing lock.
    _, pipeline, _ = validate_snapshot(run, directory)
    with pipeline.execution_lock(run):
        runner, pipeline, _ = build_runner(run, directory, activate=True)
        try:
            runner.execute()
        except BaseException as error:
            pipeline.status(run, 'FAILED', state='FAILED', failed_stage=runner.current_stage, error=str(error))
            raise


if __name__ == '__main__':
    main()
