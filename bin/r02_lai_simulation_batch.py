#!/usr/bin/env python3
"""Prepare a bounded simulation batch from an authenticated donor design.

This writes valid demographic model files and explicit job requests, not a
simulation campaign. Binding a request to checked parental VCFs and scientific
admission remains a separate step. No genotypes are opened, changed or sampled.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import resource
import shutil
import time

import r02_lai_simulate as sim
from r02_lai_support_review import require, read, write, sha

SCHEMA = 'r02_lai_simulation_batch_input_v1'
STATUS = 'BOUNDED_DEVELOPMENT_BATCH_PREPARED_NOT_SIMULATION_AUTHORIZATION'
CODE_FILES = ('r02_lai_simulation_batch.py', *sim.CODE_FILES)


def authenticate_design(entry):
    path = sim.descriptor(entry); manifest = read(path)
    require(manifest.get('schema') == 'r02_lai_development_design_v1' and
            manifest.get('status') == 'DEVELOPMENT_DESIGN_PREPARED_NOT_SIMULATION_AUTHORIZATION',
            'Completed development donor design required')
    summary = manifest.get('summary', {})
    for key in ('source_test_genotypes_opened', 'source_test_rows_emitted',
                'original_roles_changed', 'original_units_changed', 'independent_test_created'):
        require(summary.get(key) is False, 'Incompatible donor design: '+key)
    outputs = manifest.get('outputs_sha256')
    require(isinstance(outputs, dict) and outputs, 'Design output inventory required')
    files = {}
    for name, digest in outputs.items():
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name or '') and name != 'manifest.json',
                'Unsafe design output name')
        files[name] = sim.descriptor(dict(path=str(path.parent/name), sha256=digest))
    require({'roles.private.tsv', 'units.private.tsv', 'scenarios.json', 'summary.json',
             'support_summary.json', 'support_by_donor_fold.tsv'} <= set(files),
            'Incomplete donor design outputs')
    return manifest, files


def positive_integer(value, name):
    require(type(value) is int and value > 0, 'Positive integer required: '+name)
    return value


def seed_for(namespace, task_id):
    """A deterministic positive uint32; collision checks are mandatory upstream."""
    digest = hashlib.sha256((namespace+'\0'+task_id).encode()).digest()
    return int.from_bytes(digest[:8], 'big') % (2**32-1) + 1


def read_rows(path):
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), 'Malformed design table')
            yield row


def guard(out, limits, started):
    require(time.monotonic()-started <= limits['timeout_seconds'], 'Preparation timeout exceeded')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= limits['memory_gib']*2**30,
            'Preparation memory cap exceeded')
    require(shutil.disk_usage(out.parent).free >= limits['disk_reserve_gib']*2**30,
            'Preparation disk reserve violated')
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    require(size <= limits['max_output_gib']*2**30, 'Preparation output cap exceeded')


def prepare(path, digest, out):
    started = time.monotonic()
    contract_path = sim.descriptor(dict(path=str(Path(path).resolve()), sha256=digest))
    c = read(contract_path)
    require(c.get('schema') == SCHEMA and set(c) ==
            {'schema', 'design', 'batch', 'job_resources', 'preparation_resources', 'budget', 'interpretation', 'consumer_policy'},
            'Explicit batch contract fields required')
    design, files = authenticate_design(c['design']); templates = read(files['scenarios.json'])
    require(templates.get('scope') == 'CONDITIONAL_DEVELOPMENT_ONLY', 'Development templates required')
    b = c['batch']
    require(set(b) == {'batch_id', 'scenario_ids', 'chromosomes', 'folds', 'replicates', 'simulated_people',
                      'seed_namespace', 'replacement', 'genome_build', 'haptools_version', 'input_policy',
                      'future_breakpoints_popsize_sensitivity', 'benchmark_chromosome'}, 'Unexpected batch fields')
    for field in ('batch_id', 'seed_namespace'):
        require(isinstance(b[field], str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,39}', b[field]),
                'Safe bounded namespace required: '+field)
    n = positive_integer(b['simulated_people'], 'simulated_people')
    replicas = positive_integer(b['replicates'], 'replicates')
    scenario_ids = b['scenario_ids']; chromosomes = b['chromosomes']; folds = b['folds']
    require(isinstance(scenario_ids, list) and scenario_ids and len(set(scenario_ids)) == len(scenario_ids),
            'Unique nonempty scenario selection required')
    require(isinstance(chromosomes, list) and chromosomes and chromosomes == sorted(set(chromosomes), key=int),
            'Sorted nonempty chromosome selection required')
    require(b['benchmark_chromosome'] in chromosomes, 'Benchmark chromosome must belong to this batch')
    available_folds = templates['policy']['fold_labels']
    require(isinstance(folds, list) and folds == available_folds, 'Preserve both original donor folds and order')
    scenarios = {s['scenario_id']: s for s in templates['scenarios']}
    require(len(scenarios) == len(templates['scenarios']) and set(scenario_ids) <= set(scenarios),
            'Unknown or duplicate frozen scenario')
    for scenario in scenarios.values():
        require(scenario.get('executable') is False and scenario.get('fitted_to_lai_performance') is False,
                'Frozen design scenarios must not imply execution or outcome-selected demographics')
    for value in b['future_breakpoints_popsize_sensitivity']:
        positive_integer(value, 'future_popsize')
        require(value >= 10*n, 'Future popsize would be silently clamped by haptools')
    require(len(set(b['future_breakpoints_popsize_sensitivity'])) == len(b['future_breakpoints_popsize_sensitivity']),
            'Duplicate future popsize sensitivity')
    budget = c['budget']
    require(set(budget) == {'max_tasks', 'max_cpu_hours', 'max_concurrent_tasks', 'max_total_output_gib',
                           'launch_authorized', 'expansion_requires_measurement'}, 'Explicit batch budget required')
    require(budget['launch_authorized'] is False and budget['expansion_requires_measurement'] is True,
            'Preparation cannot grant launch or automatic expansion')
    for key in ('max_tasks', 'max_concurrent_tasks'): positive_integer(budget[key], key)
    for key in ('max_cpu_hours', 'max_total_output_gib'):
        require(type(budget[key]) in (int,float) and math.isfinite(budget[key]) and budget[key] > 0,
                'Positive finite budget required: '+key)
    tasks_count = len(scenario_ids)*len(chromosomes)*len(folds)*replicas
    require(tasks_count <= budget['max_tasks'] and budget['max_concurrent_tasks'] <= tasks_count,
            'Task count or concurrency exceeds declared batch')
    r = c['job_resources']
    base = dict(schema=sim.SCHEMA_V2, scope='DNABR_AUTHORIZED_SIMULATION', job_id=b['batch_id'],
                sample_namespace=b['batch_id'], donor_split=folds[0], chromosomes=chromosomes,
                genome_build=b['genome_build'], seed=1, replacement=b['replacement'],
                haptools_version=b['haptools_version'], input_policy=b['input_policy'], resources=r)
    sim.validate_parameters(base)
    require(r['popsize'] >= 10*n, 'Internal popsize would be silently clamped')
    require(tasks_count*r['cpus']*r['timeout_seconds']/3600 <= budget['max_cpu_hours'],
            'Sum of task time caps exceeds CPU-hour cap')
    require(tasks_count*r['max_output_gib'] <= budget['max_total_output_gib'],
            'Sum of task output caps exceeds batch output cap')
    prep = c['preparation_resources']
    require(set(prep) == {'memory_gib','disk_reserve_gib','max_output_gib','timeout_seconds'},
            'Explicit preparation resource limits required')
    for key,value in prep.items():
        require(type(value) in (int,float) and math.isfinite(value) and value > 0,
                'Positive finite preparation cap required: '+key)
    require(isinstance(c['interpretation'], dict) and set(c['interpretation']) ==
            {'size_justification','popsize_justification','replication_unit','replacement_justification',
             'expansion_rule','limitations','references'} and
            all(isinstance(c['interpretation'][key], str) and c['interpretation'][key].strip()
                for key in c['interpretation'] if key not in {'limitations','references'}) and
            isinstance(c['interpretation']['limitations'], list) and c['interpretation']['limitations'] and
            isinstance(c['interpretation']['references'], list) and c['interpretation']['references'],
            'Explicit scientific interpretation and references required')
    consumer=c['consumer_policy']
    require(consumer == dict(parental_markers='ALL_ELIGIBLE_COMMON_AND_RARE',
        flare_incremental_baseline='EXCLUDE_EXACT_ALLELE_IDENTITIES_OF_EVALUATED_RARE_CATALOGUE',
        flare_full_marker_comparator='SEPARATE_COMPARATOR_NOT_COMMON_ONLY',
        comparisons='SAME_REALIZED_MOSAICS_REFERENCES_AND_EVALUATION_TERRITORY',
        rare_catalogue_hash_required_before_flare=True, chromosome_identity='NO_CROSS_CHROMOSOME_PERSON_ID_MERGE'),
        'Explicit non-circular consumer comparison policy required')
    donor_rows = {fold:[] for fold in folds}; counts = defaultdict(Counter); units = defaultdict(set)
    seen = set(); assignments = {}
    for row in read_rows(files['roles.private.tsv']):
        sid = row['sample_id']; require(sid and sid not in seen, 'Duplicate or empty design identity'); seen.add(sid)
        require(row['role'] in {'SOURCE_VALID','REF_TRAIN'}, 'Unsupported role; test metadata must not be emitted')
        if row['role'] == 'REF_TRAIN': continue
        fold = row['split']; require(fold in donor_rows, 'Unexpected donor fold')
        unit = row['family_component']; require(unit, 'Missing original donor unit')
        require(assignments.setdefault(unit, fold) == fold, 'Original donor unit crosses folds')
        donor_rows[fold].append(sid); counts[fold][row['ancestry']] += 1; units[(fold,row['ancestry'])].add(unit)
    ancestry_order = templates['policy']['ancestry_order']
    for fold in folds:
        require(set(counts[fold]) == set(ancestry_order), 'A donor ancestry is absent from a fold')
        require(len(donor_rows[fold]) == design['summary']['fold_people'][fold], 'Donor count differs from design')
        if not b['replacement']:
            require(min(counts[fold].values()) >= n, 'haptools no-replacement requires at least N donors per ancestry')
    out = Path(out).resolve(); require(not out.exists() and out.parent.is_dir(), 'Fresh output directory required')
    sim.local_filesystem(out.parent); guard(out,prep,started); out.mkdir(mode=0o700)
    shutil.copyfile(files['roles.private.tsv'], out/'roles.private.tsv')
    for fold,sids in donor_rows.items():
        (out/(fold+'.samples.private.txt')).write_text(''.join(sid+'\n' for sid in sorted(sids)))
    model_files = {}; requests = []; seeds = set()
    for scenario_id in scenario_ids:
        scenario = scenarios[scenario_id]; model = out/(scenario_id+'.model.dat')
        model.write_text('\t'.join([str(n),'Admixed',*ancestry_order])+'\n'+
                         '\t'.join([str(scenario['generations']),'0',
                                     *(format(scenario['fractions'][a],'.17g') for a in ancestry_order)])+'\n')
        require(sim.read_model(model,r['popsize']) == (n,ancestry_order), 'Generated model interpretation changed')
        model_files[scenario_id] = dict(file=model.name,sha256=sha(model))
        for chrom in chromosomes:
            for fold in folds:
                for replicate in range(1,replicas+1):
                    task_id = f'{b["batch_id"]}.{scenario_id}.chr{chrom}.{fold}.r{replicate}'
                    seed = seed_for(b['seed_namespace'],task_id)
                    require(seed not in seeds, 'Deterministic seed collision'); seeds.add(seed)
                    request = {**base, 'job_id':task_id, 'sample_namespace':task_id,
                               'donor_split':fold, 'chromosomes':[chrom], 'seed':seed}
                    sim.validate_parameters(request)
                    requests.append(dict(task_id=task_id, replicate=replicate, scenario_id=scenario_id,
                        parameters=request, model=model_files[scenario_id],
                        roles=dict(file='roles.private.tsv',sha256=sha(out/'roles.private.tsv')),
                        donor_samples=dict(file=fold+'.samples.private.txt',sha256=sha(out/(fold+'.samples.private.txt'))),
                        resource_release='BENCHMARK_CHROMOSOME' if chrom==b['benchmark_chromosome'] else
                            'REQUIRES_MEASURED_BENCHMARK_CHROMOSOME_WITHIN_CAPS',
                        required_binding='EXACT_FOLD_VCF_INDEX_MAP_AND_SCIENTIFIC_REVIEW', executable=False))
    scope_rows=[]
    for fold in folds:
        for ancestry in ancestry_order:
            scope_rows.append(dict(fold=fold,ancestry=ancestry,donor_people=counts[fold][ancestry],
                                   original_units=len(units[(fold,ancestry)])))
    write(out/'requests.json',dict(schema='r02_lai_simulation_requests_v1',status=STATUS,requests=requests,
        frozen_design=c['design'], available_scenarios=list(scenarios), selected_scenarios=scenario_ids,
        original_material_report=design['inputs']['material_report'], original_roles=design['inputs']['original_roles'],
        benchmark_chromosome=b['benchmark_chromosome'], consumer_policy=consumer,
        scenarios_changed=False, donor_folds_changed=False, launch_authorized=False))
    write(out/'scope.json',dict(scope='CONDITIONAL_DEVELOPMENT_ONLY',donor_support=scope_rows,
        inherited_design_summary=design['summary'], rare_support=read(files['support_summary.json']),
        interpretation=c['interpretation'], phase_accuracy_validated=False, genotype_quality_validated=False,
        original_donor_people=design['summary']['donors'], simulated_output_records_requested=tasks_count*n,
        simulated_outputs_are_independent_biological_people=False,
        chromosome_specific_samples_not_a_phased_whole_genome_cohort=True,
        consumer_policy=consumer, reuse_same_realized_mosaics_for_all_lai_comparators=True))
    write(out/'release_requirements.json',dict(status='NO_SIMULATION_LAUNCHED',
        automatic_preparation_checks=['Exact immutable parental manifest for each pilot chromosome; no need to wait for unused chromosomes',
            'Exact original roles and population/kinship units; selected donor fold in the VCF header',
            'Every parental site/allele/FORMAT and selected GT copied exactly, including monomorphic sites',
            'Accessible indexes, authenticated map, no extrapolation and explicit marker territory',
            'Reference unknowns preserved and no genotype imputation',
            'Actual counts and map-point work proxy within explicit independent memory/time/output caps'],
        explicit_conditional_review_checks={key:'Required interpretation bound to the exact scenario and evidence; not an automatic biological PASS'
            for key in sim.CHECKS},
        acceptable_scope='Previously exposed donor-unit development only; supplied phase and GT_ONLY limitations explicit',
        kinship_rule='Keep20 reference-donor links sensitive toPC4 documented; do not choosePC8 to erase them or certify independence',
        nam_rule='One NAM population unit per fold,2 versus12 people; no broad NAM power or generalization claim',
        benchmark_rule='Release benchmark chromosome first; other chromosomes require measured runtime,peakRSS,output bytes and support before launch',
        consumer_policy=consumer, review_none_is_executable=False))
    write(out/'budget.json',dict(**budget, tasks_prepared=tasks_count, simulated_diploids_per_task=n,
        replicas_per_fold_chromosome=replicas, internal_simulator_popsize=r['popsize'],
        declared_cpu_hours_cap=tasks_count*r['cpus']*r['timeout_seconds']/3600,
        declared_total_output_gib_cap=tasks_count*r['max_output_gib'], runtime_estimated=False,
        work_proxy_definition='internal_popsize x generations x map_points; not measured operations or elapsed time',
        monetary_cost_estimated=False, initial_batch_not_full_9_scenario_22_autosome_grid=True,
        future_popsize_breakpoints_only=b['future_breakpoints_popsize_sensitivity'],
        future_popsize_tasks_created=0, simulations_executed=False))
    # All raw requests deliberately lack genotype inputs and are not simulation contracts.
    authenticate_design(c['design']); sim.descriptor(dict(path=str(contract_path),sha256=digest)); guard(out,prep,started)
    manifest=dict(schema='r02_lai_simulation_batch_v1',status=STATUS,contract_sha256=digest,
        design=c['design'], tasks_prepared=tasks_count, simulations_executed=False, training_executed=False,
        contains_sample_identifiers=True, public_distribution_allowed=False,
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        code_sha256={name:sha(Path(__file__).parent/name) for name in CODE_FILES})
    write(out/'manifest.json',manifest); return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); os.umask(0o077)
    prepare(args.contract,args.contract_sha256,args.output_dir)


if __name__ == '__main__': main()
