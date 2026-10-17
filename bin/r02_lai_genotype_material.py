#!/usr/bin/env python3
"""Export diploid ALT dosage and recorded LAI truth without learning or filtering.

All loci of an authenticated simulation are retained. Truth is a separate output,
never an input feature. This material does not authorize neural fitting or create
an independent evaluation set from previously exposed donor panels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

import numpy as np
import r02_lai_flare_evaluation_material as evaluation
from r02_lai_support_review import read, require, sha, write

SCHEMA = 'r02_lai_genotype_material_request_v1'
CODE_FILES = tuple(dict.fromkeys(('r02_lai_genotype_material.py', *evaluation.CODE_FILES)))
GT = re.compile(r'^([01.])([/|])([01.])$')


def implementation():
    return {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def dosage(calls):
    """Discard homolog order explicitly; preserve incomplete calls as missing."""
    x = np.asarray(calls)
    require(x.ndim == 2 and x.shape[1] == 3, 'Only diploid calls with phase flag supported')
    alleles = x[:, :2]
    require(np.isin(alleles, (-1, 0, 1)).all(), 'Non-biallelic genotype in simulation')
    observed = (alleles >= 0).all(axis=1)
    result = np.full(len(x), -1, dtype=np.int8)
    result[observed] = alleles[observed].sum(axis=1)
    return result, observed


def parse_gt(value):
    """Parse one explicit diploid biallelic GT while retaining its phase flag."""
    match = GT.fullmatch(value)
    require(match is not None, 'Only diploid biallelic GT supported')
    left, separator, right = match.groups()
    return [-1 if left == '.' else int(left), -1 if right == '.' else int(right), separator == '|']


def bcftools_version():
    value = subprocess.run(['bcftools', '--version-only'], check=True, text=True,
                           capture_output=True, timeout=30).stdout.strip()
    require(value and '\n' not in value, 'Invalid bcftools version')
    return value


def prepare(request_path, digest, output, source_overrides=None):
    started = time.monotonic()
    request_path = Path(request_path)
    require(sha(request_path) == digest, 'Request SHA differs')
    request = read(request_path)
    require(request['schema'] == SCHEMA and request['operation'] == 'PREPARE_ONLY_NO_TRAINING',
            'Explicit preparation-only request required')
    require(request['implementation_sha256'] == implementation(), 'Frozen implementation differs')
    inputs = request['inputs']
    require(set(inputs) == {'manifest', 'genotypes', 'truth'}, 'Exact source inventory required')
    if source_overrides is not None:
        require(set(source_overrides) == set(inputs), 'Exact staged source inventory required')
    paths = {}
    for key, item in inputs.items():
        path = Path(source_overrides[key] if source_overrides is not None else item['path'])
        require(path.is_file() and sha(path) == item['sha256'], 'Source differs: ' + key)
        paths[key] = path
    source = read(paths['manifest'])
    require(source['schema'] == 'r02_lai_simulate_output_v1' and
            source['status'] == 'COMPLETE_SIMULATED_SCENARIO_NOT_LAI_VALIDATION' and
            source['scope'] in {'DNABR_AUTHORIZED_SIMULATION', 'SYNTHETIC_TEST'},
            'Completed registered simulation required')
    for key, filename in [('genotypes', 'genotypes.vcf.gz'), ('truth', 'truth.private.tsv.gz')]:
        require(source['outputs_sha256'][filename] == inputs[key]['sha256'],
                'Genotypes/truth do not originate from the same simulation')
    require(len(source['chromosomes']) == 1, 'One chromosome per recoverable task')
    chromosome = source['chromosomes'][0]
    n = source['summary']['simulated_individuals']
    m = source['summary']['variants_by_chromosome'][chromosome]
    limits = request['resources']
    require(set(limits) == {'max_people', 'max_loci', 'max_output_bytes', 'min_free_bytes',
                           'timeout_seconds', 'max_memory_bytes', 'chunk_loci'} and
            all(type(v) is int and v > 0 for v in limits.values()), 'Explicit positive resource bounds required')
    require(0 < n <= limits['max_people'] and 0 < m <= limits['max_loci'], 'Shape exceeds resource limit')
    # Dosage, observed GT, truth, observed truth, temporary two homolog truth plus axes.
    projected = 6 * n * m + 100 * m + 2**20
    out = Path(output).resolve()
    require(not out.exists() and out.parent.is_dir(), 'Fresh output directory required')
    evaluation.sim.local_filesystem(out.parent)
    require(projected <= limits['max_output_bytes'] and
            shutil.disk_usage(out.parent).free >= projected + limits['min_free_bytes'],
            'Insufficient local disk reserve')
    out.mkdir(mode=0o700)

    def guard():
        require(time.monotonic() - started < limits['timeout_seconds'], 'Preparation deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= limits['max_memory_bytes'],
                'Preparation memory limit')
        require(shutil.disk_usage(out).free >= limits['min_free_bytes'], 'Local disk reserve breached')

    samples = subprocess.run(['bcftools', 'query', '-l', str(paths['genotypes'])], check=True,
                             text=True, capture_output=True, timeout=60).stdout.splitlines()
    require(len(samples) == n and len(set(samples)) == n, 'Unique simulation person axis required')
    genotypes = np.lib.format.open_memmap(out/'alt_dosage.npy', mode='w+', dtype=np.int8, shape=(n, m))
    observed = np.lib.format.open_memmap(out/'genotype_observed.npy', mode='w+', dtype=np.bool_, shape=(n, m))
    positions = np.empty(m, dtype=np.int64)
    block_size = min(m, limits['chunk_loci'])
    block_gt = np.empty((n, block_size), dtype=np.int8)
    block_mask = np.empty_like(block_gt, dtype=np.bool_)
    count = start = previous = missing = phase_flags = 0
    with (out/'loci.tsv').open('x') as stream:
        table = csv.writer(stream, delimiter='\t', lineterminator='\n')
        table.writerow(('chromosome', 'position_bp', 'REF', 'ALT', 'counted_allele'))
        with (out/'bcftools.query.log').open('x') as query_log:
            query = subprocess.Popen(['bcftools', 'query', '-f', '%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n',
                                      str(paths['genotypes'])], text=True, stdout=subprocess.PIPE,
                                     stderr=query_log, env={**os.environ, 'OMP_NUM_THREADS':'1'})
            require(query.stdout is not None, 'bcftools query stream unavailable')
            for line in query.stdout:
                fields = line.rstrip('\n').split('\t')
                require(len(fields) == 4+n, 'bcftools GT row/sample axis differs')
                chrom, raw_pos, ref, alt = fields[:4]
                position = int(raw_pos)
                require(count < m and chrom == chromosome and position > previous,
                    'Chromosome, order, duplicate position or locus count differs')
                require(len(ref) == 1 and len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT' and ref != alt,
                    'Expected original biallelic SNV axis')
                calls = [parse_gt(value) for value in fields[4:]]
                gt, mask = dosage(calls)
                block_gt[:, count-start] = gt
                block_mask[:, count-start] = mask
                missing += int((~mask).sum())
                phase_flags += sum(bool(x[2]) for x in calls)
                positions[count] = previous = position
                table.writerow((chromosome, previous, ref, alt, 'ALT'))
                count += 1
                if count-start == block_size or count == m:
                    genotypes[:, start:count] = block_gt[:, :count-start]
                    observed[:, start:count] = block_mask[:, :count-start]
                    start = count
                    guard()
            query.stdout.close()
            require(query.wait(timeout=60) == 0, 'bcftools query failed; inspect log')
    require(count == m, 'Simulation locus count differs')
    genotypes.flush(); observed.flush()
    del genotypes, observed
    truth = evaluation.truth_arrays(paths['truth'], samples, positions, chromosome, out, guard)
    with (out/'persons.tsv').open('x') as stream:
        stream.write('sample_key_sha256\tdonor_fold\tuse\n')
        for sample in samples:
            stream.write(hashlib.sha256(sample.encode()).hexdigest() + '\t' + source['donor_split'] + '\tCONDITIONAL_DEVELOPMENT\n')
    result = dict(schema='r02_lai_genotype_material_v1',
        status='COMPLETE_PAIRED_MATERIAL_NOT_NEURAL_TRAINING_OR_VALIDATION',
        source_manifest_sha256=inputs['manifest']['sha256'], request_sha256=digest,
        implementation_sha256=implementation(), case_id=source['job_id'],
        chromosome=chromosome, genome_build=source['genome_build'], n_people=n, n_loci=m,
        gt_missing_cells=missing, recorded_gt_phase_flags=phase_flags,
        homolog_order_used=False, parental_phase_validated=False,
        donor_split=source['donor_split'], truth_coverage=truth,
        genotype_reader='bcftools query', bcftools_version=bcftools_version(),
        predictor_files=['alt_dosage.npy','genotype_observed.npy','loci.tsv'],
        separate_target_files=['truth_state.npy','observed_mask.npy'],
        rare_channels_assigned=False, train_select_test_split_assigned=False,
        parameters_changed=False, independent_evaluation=False,
        elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    require(sum(p.stat().st_size for p in out.iterdir()) <= limits['max_output_bytes'], 'Output limit exceeded')
    write(out/'manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest-source')
    parser.add_argument('--genotypes-source')
    parser.add_argument('--truth-source')
    args = parser.parse_args()
    os.umask(0o077)
    values = (args.manifest_source, args.genotypes_source, args.truth_source)
    require(not any(values) or all(values), 'All staged source overrides are required together')
    overrides = None if not any(values) else dict(manifest=args.manifest_source,
        genotypes=args.genotypes_source, truth=args.truth_source)
    value = prepare(args.request, args.request_sha256, args.output, overrides)
    print(json.dumps({k:value[k] for k in ('status','case_id','n_people','n_loci','elapsed_seconds')}))
