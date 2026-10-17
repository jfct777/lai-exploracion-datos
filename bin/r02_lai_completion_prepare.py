#!/usr/bin/env python3
"""Bind the remaining predeclared LAI simulations to authenticated copied inputs.

No simulation, role change, new scenario or scientific approval is performed by
this preparer. A separate root review must admit the exact four parental axes.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil

from r02_lai_support_review import require, read, write, sha
import r02_lai_simulate as sim

SCHEMA = 'r02_lai_completion_review_v1'
COMPLETED_ID = 'R02L2pilot.PULSE_T10_ANCHOR.chr22.FOLD_A.r1'


def remaining_requests(requests, completed):
    require(completed.get('job_id') == COMPLETED_ID and
            completed.get('status') == 'ONE_SCENARIO_COPIED_VERIFIED_PUBLISHED_NOT_LAI_VALIDATION',
            'Exact completed first scenario required')
    rows = requests['requests']; ids = [r['task_id'] for r in rows]
    expected = {f'R02L2pilot.PULSE_T10_ANCHOR.chr{c}.{f}.r{r}'
                for c in ('1','22') for f in ('FOLD_A','FOLD_B') for r in (1,2,3)}
    require(len(rows) == 12 and len(set(ids)) == 12 and set(ids) == expected,
            'Preserve the exact original twelve requests')
    for row in rows:
        p = row['parameters']; sim.validate_parameters(p)
        require(p['job_id'] == row['task_id'] == p['sample_namespace'] and
                row['scenario_id'] == 'PULSE_T10_ANCHOR', 'Request identity or scenario differs')
    return [r for r in rows if r['task_id'] != COMPLETED_ID]


def validate_review(review, input_audit, requests_sha, completed_sha):
    require(review['schema'] == SCHEMA and review['status'] == 'REVIEWED_FOR_DECLARED_SCENARIOS',
            'Independent exact root review required')
    require(review['original_requests_sha256'] == requests_sha and
            review['completed_receipt_sha256'] == completed_sha and
            review['material_preflight_sha256'] == sha(input_audit), 'Review source binding differs')
    expected = {(c,f) for c in ('1','22') for f in ('FOLD_A','FOLD_B')}
    require({(r['chromosome'],r['fold']) for r in review['admitted_axes']} == expected,
            'All and only the four parental axes must be reviewed')
    require(review['reviewer'] and review['limitations'] and set(review['checks']) == set(sim.CHECKS),
            'Explicit reviewer, limitations and all scientific checks required')
    for item in review['checks'].values():
        require(item['status'] == 'ACCEPTED_FOR_SCENARIO' and item['interpretation'],
                'No silent acceptance of an unresolved scientific check')
    require(review['scientific_parameters_changed'] is False and
            review['original_roles_changed'] is False and review['source_test_genotypes_used'] is False,
            'No changed scenario, roles or test data admitted')


def prepare(stage, requests_path, completed_path, review_path, output, runtime_root):
    stage, output, runtime_root = map(Path,(stage, output, runtime_root))
    require(stage.is_dir() and not output.exists() and output.parent.is_dir(), 'Fresh local package required')
    require(runtime_root.is_absolute() and '..' not in runtime_root.parts, 'Absolute canonical runtime required')
    requests = read(requests_path); completed = read(completed_path); review = read(review_path)
    selected = remaining_requests(requests, completed)
    audit = stage/'material-preflight.json'
    validate_review(review, audit, sha(requests_path), sha(completed_path))
    preflight = read(audit)
    require(preflight['status'] == 'FOUR_PARENTAL_FILES_COMPLETE_DONORS_MAP_ALLELES_FILTER_PHASE_FORMAT_VERIFIED_NO_SIMULATION',
            'Exact donor call and coordinate preflight required')
    require({(r['chromosome'],r['fold']) for r in preflight['results']} ==
            {(c,f) for c in ('1','22') for f in ('FOLD_A','FOLD_B')}, 'Input audit axes differ')
    for r in preflight['sources']:
        p=Path(r['local_path']); require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'], 'Frozen parental source changed')
    output.mkdir(mode=0o700)
    def stage_file(source, relative, expected=None):
        source=Path(source); digest=sha(source)
        require(expected is None or digest==expected,'Source digest differs: '+str(source))
        target=output/relative; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): require(sha(target)==digest, 'Conflicting staged input')
        else: shutil.copyfile(source,target)
        return dict(path=str(runtime_root/relative),sha256=digest)
    root_review=stage_file(review_path,'review/ROOT_REVIEW.json')
    stage_file(audit,'review/material-preflight.json')
    stage_file(requests_path,'review/original-requests.json')
    stage_file(completed_path,'review/completed-first-receipt.json')
    jobs=[]
    for row in selected:
        c=copy.deepcopy(row['parameters']); chrom=c['chromosomes'][0]; fold=c['donor_split']; base=stage/'parentals'/('chr'+chrom)
        original_plan=read(base/(row['task_id']+'.plan/plan.json'))
        source=original_plan['scenario']
        require({k:v for k,v in source.items() if k not in ('inputs','preparation_provenance')}==c,
                'Original plan changed parameters or resources')
        c['inputs']={}
        for key in ('donor_vcf','donor_index','roles'):
            old=source['inputs'][key]; local=base/Path(old['path']).name
            c['inputs'][key]=stage_file(local,'inputs/chr'+chrom+'/'+local.name,old['sha256'])
        c['inputs']['model']=stage_file(stage/'first/bundle/source/model.dat','inputs/common/model.dat',source['inputs']['model']['sha256'])
        c['inputs']['maps']={chrom:stage_file(base/'genetic.map','inputs/chr'+chrom+'/genetic.map',source['inputs']['maps'][chrom]['sha256'])}
        c['preparation_provenance']={
            'material_report':stage_file(stage/'first/bundle/material/manifest.json','inputs/common/material-report.json',source['preparation_provenance']['material_report']['sha256']),
            'original_roles':stage_file(stage/'first/bundle/source/original_roles.tsv','inputs/common/original-roles.tsv',source['preparation_provenance']['original_roles']['sha256'])}
        sources={k:c['inputs'][k]['sha256'] for k in ('donor_vcf','donor_index','roles','model')}
        sources['maps']={chrom:c['inputs']['maps'][chrom]['sha256']}
        admission=dict(schema='r02_lai_simulation_admission_v1',status='REVIEWED_FOR_DECLARED_SCENARIO',
            scope=c['scope'],genome_build=c['genome_build'],donor_split=fold,source_sha256=sources,
            preparation_sha256={k:v['sha256']for k,v in c['preparation_provenance'].items()},input_policy=c['input_policy'],
            reviewer=review['reviewer'],limitations=review['limitations'],copy_provenance_policy='DONOR_EXACT_HOMOLOG_COMPATIBILITY_SET',
            checks={k:dict(v,evidence=root_review)for k,v in review['checks'].items()},independent_review=root_review)
        admission_path=output/'admissions'/(row['task_id']+'.json');admission_path.parent.mkdir(exist_ok=True);write(admission_path,admission)
        c['inputs']['admission']=dict(path=str(runtime_root/'admissions'/admission_path.name),sha256=sha(admission_path))
        contract=output/'contracts'/(row['task_id']+'.json');contract.parent.mkdir(exist_ok=True);write(contract,c)
        jobs.append(dict(path=str(runtime_root/'contracts'/contract.name),sha256=sha(contract)))
    write(output/'campaign.json',dict(schema='r02_lai_simulate_campaign_v1',jobs=jobs))
    benchmark_id='R02L2pilot.PULSE_T10_ANCHOR.chr1.FOLD_A.r1'
    benchmark=[j for j in jobs if Path(j['path']).name==benchmark_id+'.json']
    require(len(benchmark)==1,'Exactly one longest-chromosome first task required')
    write(output/'campaign-chr1-first.json',dict(schema='r02_lai_simulate_campaign_v1',jobs=benchmark))
    write(output/'campaign-after-chr1.json',dict(schema='r02_lai_simulate_campaign_v1',jobs=[j for j in jobs if j not in benchmark]))
    receipt=dict(schema='r02_lai_completion_prepared_v1',status='PREPARED_NOT_LAUNCHED',
        new_tasks=len(jobs),reused_task=COMPLETED_ID,first_remaining_task=benchmark_id,
        following_tasks_require_measured_chr1_success=True,original_requests_sha256=sha(requests_path),
        independent_root_review_sha256=sha(review_path),runtime_root=str(runtime_root),
        scientific_parameters_changed=False,source_test_used=False,training_allowed=False,flare_allowed=False,
        files_sha256={str(p.relative_to(output)):sha(p)for p in sorted(output.rglob('*'))if p.is_file()},
        code_sha256={n:sha(Path(__file__).with_name(n))for n in ('r02_lai_completion_prepare.py',*sim.CODE_FILES)})
    write(output/'prepared.json',receipt); return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','requests','completed','review','output','runtime-root'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();prepare(a.stage,a.requests,a.completed,a.review,a.output,a.runtime_root)

if __name__=='__main__':main()
