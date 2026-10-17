#!/usr/bin/env python3
"""Execute one authenticated FLARE plan and audit predictions without reading truth.

The preflight and inference are separate Nextflow tasks. This adapter never
selects a demographic scenario or optimizes parameters. Successful execution is
an ancestry estimate, not biological validation; scoring belongs downstream.
"""
from __future__ import annotations

import argparse
import gzip
from itertools import zip_longest
import math
import os
from pathlib import Path
import time

import r02_lai_flare_plan as plan
import r02_lai_simulate as sim
from r02_lai_support_review import require, read, write, sha

CODE_FILES = (*plan.CODE_FILES, 'r02_lai_flare_run.py')


def audit_predictions(path, target_path, ancestry_names, resources, out, started):
    """Stream marker/sample axes, copied GT and both posterior vectors, O(samples)."""
    target = sim.vcf_rows(target_path)
    target_samples, sentinel = next(target)
    require(sentinel is None, 'Target header missing')
    expected_header = '##ANCESTRY=<' + ','.join(f'{a}={i}' for i, a in enumerate(ancestry_names)) + '>'
    count = 0
    headers = set()
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.startswith('##'):
                headers.add(line.rstrip('\n'))
            elif line.startswith('#CHROM\t'):
                require(line.rstrip('\n').split('\t')[9:] == target_samples, 'Prediction sample order differs')
                break
            else:
                raise ValueError('Prediction header missing or malformed')
        else:
            raise ValueError('Prediction sample header missing')
        require(expected_header in headers, 'Prediction ancestry order differs')
        for prediction, original in zip_longest(stream, target):
            require(prediction is not None and original is not None, 'Prediction marker count differs')
            samples, row = original
            fields = prediction.rstrip('\n').split('\t')
            require(samples == target_samples and row is not None and len(fields) == 9 + len(samples),
                    'Prediction row width or target header differs')
            require(tuple(fields[i] for i in (0, 1, 3, 4)) == tuple(row[i] for i in (0, 1, 3, 4)),
                    'Prediction marker/allele axis differs')
            fmt = fields[8].split(':')
            require(len(set(fmt)) == len(fmt) and {'GT','AN1','AN2','ANP1','ANP2'} <= set(fmt),
                    'Prediction FORMAT incomplete')
            indexes = {name: fmt.index(name) for name in ('GT','AN1','AN2','ANP1','ANP2')}
            for sample, expected_gt in zip(fields[9:], row[9:]):
                values = sample.split(':')
                require(len(values) == len(fmt) and values[indexes['GT']] == expected_gt,
                        'Prediction changes target genotype or homolog order')
                for hard_name, prob_name in (('AN1','ANP1'),('AN2','ANP2')):
                    probabilities = [float(x) for x in values[indexes[prob_name]].split(',')]
                    hard = int(values[indexes[hard_name]])
                    require(len(probabilities) == len(ancestry_names) and 0 <= hard < len(ancestry_names),
                            'Prediction ancestry dimension differs')
                    require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities),
                            'Invalid ancestry probability')
                    # FLARE serializes rounded probabilities. Preserve these values;
                    # neither renormalize nor silently manufacture missing mass.
                    require(abs(sum(probabilities)-1) <= .010000001, 'Invalid posterior probability mass')
                    require(probabilities[hard] >= max(probabilities)-1e-9, 'Hard call is not posterior argmax')
            count += 1
            if count % 1024 == 0:
                sim.resource_guard(out, resources, started)
    require(count > 0, 'Empty prediction VCF')
    return dict(marker_count=count, target_samples=len(target_samples),
                haplotype_probability_cells=2*count*len(target_samples),
                genotype_and_homolog_order_preserved=True,
                probability_values_renormalized=False)


def execute(contract_path, contract_digest, plan_dir, manifest_digest, output_dir):
    started = time.monotonic()
    contract_path = Path(contract_path).resolve()
    config = read(sim.descriptor(dict(path=str(contract_path), sha256=contract_digest)))
    parameters, bounds = plan.validate(config)
    plan_dir = Path(plan_dir).resolve()
    manifest = read(sim.descriptor(dict(path=str(plan_dir/'manifest.json'), sha256=manifest_digest)))
    require(manifest.get('schema') == 'r02_lai_flare_plan_manifest_v1' and
            manifest.get('status') == 'PREFLIGHT_ONLY_NOT_INFERENCE_OR_SCIENTIFIC_APPROVAL',
            'An authenticated completed preflight is required')
    require(set(manifest.get('outputs_sha256', {})) == {'plan.json','genetic.map','reference-panel.tsv'},
            'Unexpected preflight output inventory')
    for name, digest in manifest['outputs_sha256'].items():
        sim.descriptor(dict(path=str(plan_dir/name), sha256=digest))
    for name in plan.CODE_FILES:
        require(manifest.get('code_sha256', {}).get(name) == sha(Path(__file__).parent/name),
                'Preflight/executor code drift; regenerate preflight, not silently reuse cache')
    prepared = read(plan_dir/'plan.json')
    require(prepared.get('input_contract_sha256') == contract_digest and
            prepared.get('parameters') == parameters and prepared.get('scope') == config['scope'],
            'Preflight does not bind the supplied execution contract')
    files = {key: sim.descriptor(entry) for key, entry in config['inputs'].items()}
    export = plan.validate_export(config,files)
    require(prepared.get('probability_export') == export, 'Preflight/executor probability export differs')
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists() and output_dir.parent.is_dir(), 'Fresh output directory required')
    sim.local_filesystem(output_dir.parent)
    require(sim.shutil.disk_usage(output_dir.parent).free >= bounds['disk_reserve_gib']*2**30,
            'Insufficient disk reserve')
    output_dir.mkdir(mode=0o700)
    limits = dict(timeout_seconds=bounds['max_wall_seconds'], disk_reserve_gib=bounds['disk_reserve_gib'],
                  max_output_gib=bounds['max_output_bytes']/2**30)
    command = plan.legacy.build_command('java', files['flare_jar'], files['reference_vcf'], files['target_vcf'],
        plan_dir/'reference-panel.tsv', plan_dir/'genetic.map', output_dir/'inference', parameters)
    command.insert(1, f"-Xmx{bounds['java_heap_mib']}m")
    command = [f'{key}={parameters[key]}' if '=' in item and
               (key := item.split('=', 1)[0]) in {'gen','min-maf'} else item for item in command]
    command.append('panel-cm='+str(parameters['panel-cm']))
    sim.execute_tool(command, output_dir, limits, started, log_name='flare.private.log', tool_name='FLARE')
    for suffix in ('log','model','anc.vcf.gz','global.anc.gz'):
        path = output_dir/('inference.'+suffix)
        require(path.is_file() and path.stat().st_size > 0, 'Missing FLARE output '+suffix)
    audit = audit_predictions(output_dir/'inference.anc.vcf.gz', files['target_vcf'],
                              config['ancestry_names'], limits, output_dir, started)
    require(audit['marker_count'] == prepared['shape']['marker_count'] and
            audit['target_samples'] == prepared['target_samples'], 'Prediction shape differs from preflight')
    sim.descriptor(dict(path=str(contract_path), sha256=contract_digest))
    for entry in config['inputs'].values():
        sim.descriptor(entry)
    sim.resource_guard(output_dir, limits, started)
    result = dict(schema='r02_lai_flare_run_v1',
        status='COMPLETE_TRUTH_BLIND_FLARE_NOT_LAI_VALIDATION', scope=config['scope'],
        chromosome=config['chromosome'], genome_build=config['genome_build'],
        ancestry_names=config['ancestry_names'], marker_condition=config['marker_condition'],
        input_contract_sha256=contract_digest, preflight_manifest_sha256=manifest_digest,
        parameters=parameters, generation_policy=config['generation_policy'], probability_export=export,
        inference_executed=True, truth_accessed=False, scoring_performed=False,
        genotype_quality_or_phase_accuracy_validated=False, audit=audit, command_argv=command,
        elapsed_seconds=time.monotonic()-started,
        outputs_sha256={p.name: sha(p) for p in output_dir.iterdir() if p.is_file()},
        code_sha256={name: sha(Path(__file__).parent/name) for name in CODE_FILES},
        contains_sample_identifiers=True, public_distribution_allowed=False)
    write(output_dir/'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--plan-dir', type=Path, required=True)
    parser.add_argument('--plan-manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    execute(args.contract, args.contract_sha256, args.plan_dir, args.plan_manifest_sha256, args.output_dir)


if __name__ == '__main__':
    main()
