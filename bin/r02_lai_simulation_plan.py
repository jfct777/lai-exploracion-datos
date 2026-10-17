#!/usr/bin/env python3
"""Bridge an authenticated L2 report and explicit scenario to a simulation plan.

No demographics, review decisions, genotypes or donor assignments are inferred.
An incomplete review produces a non-executable plan, never a synthetic approval.
Even a runnable contract is preparation only: this module does not run haptools.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path
import re
import time

import r02_lai_simulate as sim
from r02_lai_support_review import require, read, write, sha

SCHEMA = 'r02_lai_simulation_plan_input_v1'
REPORT_SCHEMA = 'r02_lai_admission_v1'
REPORT_STATUS = 'COMPLETE_CONDITIONAL_MATERIAL_ADMISSION_NOT_SIMULATION_AUTHORIZATION'
CODE_FILES = ('r02_lai_simulation_plan.py', *sim.CODE_FILES)


def table(path, required):
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        require(set(required) <= set(reader.fieldnames or []), 'Missing table columns: '+path.name)
        for row in reader:
            require(None not in row and all(row.get(k) is not None for k in required), 'Malformed table')
            yield row


def authenticate_report(entry):
    path = sim.descriptor(entry); report = read(path)
    require(report.get('schema') == REPORT_SCHEMA and report.get('status') == REPORT_STATUS,
            'Completed conditional L2 material report required')
    require(report.get('source_test_genotypes_opened') is False and
            report.get('roles_changed') is False and report.get('donor_units_changed') is False,
            'L2 report changed authorized roles or opened test genotypes')
    outputs = report.get('outputs_sha256')
    require(isinstance(outputs, dict) and outputs, 'L2 report output inventory required')
    paths = {}
    for name, digest in outputs.items():
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name or '') and name != 'manifest.json',
                'Unsafe or self-referential L2 output name')
        paths[name] = sim.descriptor(dict(path=str(path.parent/name), sha256=digest))
    require({'donor_chromosome_admission.private.tsv', 'criteria.tsv', 'limitations.json'} <= set(paths),
            'L2 donor identities, criteria and limitations required')
    return report, paths


def match_roles(report, paths, original_entry, selected_roles, split, populations, chromosomes, synthetic):
    original = sim.descriptor(original_entry)
    require(original_entry['sha256'] in report.get('inputs_sha256', {}).values(),
            'Original roles are not authenticated by L2')
    identities = {}
    for row in table(original, ('sample_id','role','ancestry','atomic_unit_id','canonical_population')):
        if row['role'] not in {'SOURCE_VALID','REF_TRAIN'}: continue  # metadata only; no test genotypes
        sid = row['sample_id']
        require(sid not in identities and row['atomic_unit_id'] and row['canonical_population'],
                'Duplicate identity or absent original biological unit')
        identities[sid] = row
    supplied = {}
    for row in table(selected_roles, ('sample_id','role','ancestry','split','family_component')):
        sid = row['sample_id']
        require(sid in identities and sid not in supplied, 'Unknown or duplicate simulation role identity')
        expected = identities[sid]
        require((row['role'],row['ancestry'],row['family_component']) ==
                (expected['role'],expected['ancestry'],expected['atomic_unit_id']),
                'Simulation roles changed ancestry, role or original unit')
        supplied[sid] = row
    require(set(supplied) == set(identities), 'Simulation roles must retain all authorized original people')
    donors = sim.read_roles(selected_roles, split, populations, synthetic)
    seen = set()
    for row in table(paths['donor_chromosome_admission.private.tsv'],
                     ('sample_id','role','ancestry','atomic_unit_id','canonical_population','chromosome','admission')):
        key = (row['sample_id'], row['chromosome'])
        require(key not in seen, 'Duplicate donor/chromosome in L2 report')
        seen.add(key)
        expected = identities.get(row['sample_id'])
        require(expected is not None and all(row[k] == expected[k] for k in
                ('role','ancestry','atomic_unit_id','canonical_population')), 'L2 donor identity differs from original')
        require(row['role'] == 'SOURCE_VALID' and row['admission'] == 'LIMITED_DEVELOPMENT_MATERIAL',
                'Unsupported or rejected donor admission')
    require(all((sid,ch) in seen for sid in donors for ch in chromosomes), 'Donor chromosome absent from L2')
    counts = {}
    for ancestry in populations:
        selected = [identities[sid] for sid in donors if donors[sid] == ancestry]
        counts[ancestry] = dict(people=len(selected), original_units=len({x['atomic_unit_id'] for x in selected}),
                               original_populations=len({x['canonical_population'] for x in selected}))
    return donors, counts


def prepare(path, digest, out):
    started = time.monotonic()
    config = read(sim.descriptor(dict(path=str(Path(path).resolve()), sha256=digest)))
    require(config.get('schema') == SCHEMA, 'Explicit simulation plan contract required')
    require(set(config) == {'schema','material_report','original_roles','scenario','review'}, 'Unexpected plan contract fields')
    report, paths = authenticate_report(config['material_report'])
    scenario_path = sim.descriptor(config['scenario']); job = read(scenario_path)
    require(job.get('schema') == sim.SCHEMA_V2, 'New plans require explicit v2 input policies')
    sim.validate_parameters(job)
    require('preparation_provenance' not in job, 'Planner supplies authenticated preparation provenance')
    require(set(job.get('inputs', {})) == {'donor_vcf','donor_index','roles','model','maps'},
            'Scenario must explicitly supply donor VCF/index, roles, model and maps; no embedded admission')
    files = {key: sim.descriptor(job['inputs'][key]) for key in ('donor_vcf','donor_index','roles','model')}
    require(str(files['donor_vcf']).endswith('.vcf.gz') and str(files['donor_index']) in
            {str(files['donor_vcf'])+'.tbi', str(files['donor_vcf'])+'.csi'}, 'Indexed donor VCF required')
    require(set(job['inputs']['maps']) == set(job['chromosomes']), 'One map per declared chromosome required')
    files['maps'] = {ch: sim.descriptor(value) for ch,value in job['inputs']['maps'].items()}
    require(files['model'].stat().st_size <= 2**20, 'Demographic model unexpectedly large')
    n, populations = sim.read_model(files['model'], job['resources']['popsize'])
    donors, counts = match_roles(report, paths, config['original_roles'], files['roles'], job['donor_split'],
                                populations, job['chromosomes'], job['scope'] == 'SYNTHETIC_TEST')
    maps = {ch: sim.read_map(p,ch) for ch,p in files['maps'].items()}
    # The plan validates configuration and existing role metadata, not VCF calls.
    # The execution preflight still authenticates and checks every supplied call.
    job['preparation_provenance'] = {k: copy.deepcopy(config[k]) for k in ('material_report','original_roles')}
    open_criteria = {}; seen_criteria = set()
    for row in table(paths['criteria.tsv'], ('criterion','state','next_action')):
        require(row['criterion'] not in seen_criteria, 'Duplicate L2 criterion')
        seen_criteria.add(row['criterion'])
        require(row['state'] in {'PASS','LIMITED','UNRESOLVED'}, 'Unknown L2 criterion state')
        if row['state'] != 'PASS': open_criteria[row['criterion']] = dict(state=row['state'],next_action=row['next_action'])
    review = config['review']; unresolved = []
    if review is None:
        unresolved.append('SCIENTIFIC_REVIEW_NOT_SUPPLIED')
    else:
        review_path = sim.descriptor(review); admission = read(review_path)
        require(admission.get('material_report_sha256') == config['material_report']['sha256'],
                'Review is not bound to the completed L2 report')
        require(admission.get('allowed_claim') == 'CONDITIONAL_DEVELOPMENT_ONLY',
                'Previously exposed donors cannot become a new confirmatory panel')
        acknowledged = admission.get('acknowledged_material_criteria', {})
        require(isinstance(acknowledged, dict) and set(acknowledged) == set(open_criteria) and
                all(isinstance(value,str) and value.strip() for value in acknowledged.values()),
                'Every limited/unresolved L2 criterion requires an explicit interpretation, not an automatic PASS')
        unresolved.extend(key for key in sim.CHECKS if admission.get('checks',{}).get(key,{}).get('status') != 'ACCEPTED_FOR_SCENARIO')
        expected_review = 'SYNTHETIC_FIXTURE_ONLY' if job['scope'] == 'SYNTHETIC_TEST' else 'REVIEWED_FOR_DECLARED_SCENARIO'
        if admission.get('status') != expected_review:
            unresolved.append('SCIENTIFIC_REVIEW_NOT_ACCEPTED')
        job['inputs']['admission'] = copy.deepcopy(review)
    out = Path(out).resolve()
    require(not out.exists() and out.parent.is_dir(), 'Fresh output directory required')
    sim.local_filesystem(out.parent)
    require(sim.shutil.disk_usage(out.parent).free >= job['resources']['disk_reserve_gib']*2**30,
            'Disk reserve violated before plan preparation')
    out.mkdir(mode=0o700)
    state = 'PLAN_REQUIRES_EXPLICIT_REVIEW' if unresolved else 'PLAN_REVIEWED_READY_FOR_EXECUTION_PREFLIGHT'
    summary = dict(status=state, scenario_model_sha256=job['inputs']['model']['sha256'],
        simulated_people_requested=n, chromosomes=job['chromosomes'], donor_counts_by_ancestry=counts,
        input_policy=sim.input_policy(job), unresolved_review_checks=unresolved,
        inherited_limited_or_unresolved_criteria=open_criteria,
        metadata_only_preparation=True, donor_calls_checked_here=False,
        simulations_executed=False, source_test_genotypes_opened=False, phase_accuracy_validated=False,
        original_roles_changed=False, original_units_changed=False, genotype_quality_validated=False,
        allowed_claim='CONDITIONAL_DEVELOPMENT_ONLY',
        source_valid_is_untouched_test=False, maps_points={ch:len(value) for ch,value in maps.items()})
    write(out/'plan.json', dict(summary=summary, scenario=job, inherited_material_limits=read(paths['limitations.json'])))
    if not unresolved:
        write(out/'simulation.json', job)
        sim.load_contract(out/'simulation.json', sha(out/'simulation.json'))
        write(out/'campaign.json', dict(schema='r02_lai_simulate_campaign_v1',
            jobs=[dict(path=str(out/'simulation.json'),sha256=sha(out/'simulation.json'))]))
    # Recheck immutable sources before sealing a plan.
    sim.descriptor(dict(path=str(Path(path).resolve()),sha256=digest))
    authenticate_report(config['material_report'])
    for entry in (config['original_roles'], config['scenario']): sim.descriptor(entry)
    if review: sim.descriptor(review)
    for key in ('donor_vcf','donor_index','roles','model'): sim.descriptor(job['inputs'][key])
    for entry in job['inputs']['maps'].values(): sim.descriptor(entry)
    sim.resource_guard(out,job['resources'],started)
    manifest = dict(schema='r02_lai_simulation_plan_v1', status=state, summary=summary,
        plan_contract_sha256=digest, material_report_sha256=config['material_report']['sha256'],
        scenario_sha256=config['scenario']['sha256'],
        original_roles_sha256=config['original_roles']['sha256'],
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        code_sha256={name:sha(Path(__file__).parent/name) for name in CODE_FILES},
        simulations_executed=False, contains_sample_identifiers=True, public_distribution_allowed=False)
    write(out/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args(); os.umask(0o077)
    prepare(args.contract,args.contract_sha256,args.output_dir)


if __name__ == '__main__': main()
