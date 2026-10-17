#!/usr/bin/env python3
"""Extract immutable donor folds once per chromosome and bind simulation plans.

Uses bcftools sample projection, without allele/marker filtering or frequency
recalculation. Every retained site and selected genotype is compared as text
with the authenticated parental input. Plans remain non-executable pending
their separate scientific admission; this module never invokes haptools.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import r02_lai_simulation_batch as batch
import r02_lai_simulation_plan as planner
import r02_lai_role_subset as subset
import r02_lai_simulate as sim
from r02_lai_support_review import require,read,write,sha

SCHEMA='r02_lai_fold_material_input_v1'
SCHEMA_V2='r02_lai_fold_material_input_v2'
STATUS='COMPLETE_EXACT_DONOR_FOLDS_AND_PLANS_PENDING_SCIENTIFIC_REVIEW'
CODE_FILES=('r02_lai_fold_material.py','r02_lai_simulation_batch.py','r02_lai_simulation_plan.py',
            'r02_lai_role_subset.py','r02_lai_simulate.py','r02_lai_support_review.py')


def evidence(entry,schema,status):
    source=sim.descriptor(entry); value=read(source)
    require(value.get('schema')==schema and value.get('status')==status,'Wrong completed evidence schema/status')
    declared=value.get('outputs_sha256');require(isinstance(declared,dict) and declared,'Output inventory required')
    for name,digest in declared.items():
        require(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*',name)
                and name!='manifest.json' and re.fullmatch(r'[0-9a-f]{64}',digest),'Unsafe evidence output')
    return source,value


def output(source,manifest,name):
    require(name in manifest['outputs_sha256'],'Expected evidence product absent: '+name)
    return sim.descriptor(dict(path=str(source.parent/name),sha256=manifest['outputs_sha256'][name]))


def desc(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def prepare(path,digest,out):
    started=time.monotonic(); contract_path=sim.descriptor(dict(path=str(Path(path).resolve()),sha256=digest)); c=read(contract_path)
    expected={'schema','chromosome','batch','material','material_report',
              'original_roles','resources','bcftools_version','code_sha256'}
    require(c.get('schema') in {SCHEMA,SCHEMA_V2} and
            set(c)==expected|({'extract_folds'} if c['schema']==SCHEMA_V2 else set()),
            'Explicit fold material contract required')
    chrom=c['chromosome'];require(isinstance(chrom,str) and re.fullmatch(r'([1-9]|1[0-9]|2[0-2])',chrom),'Autosome required')
    r=c['resources']
    require(set(r)=={'cpus','memory_gib','disk_reserve_gib','max_output_gib','timeout_seconds','max_input_bytes',
            'max_records','max_samples','max_line_bytes'},'Explicit fold resource limits required')
    require(r['cpus']==1,'Sequential bcftools projection uses one computation CPU')
    for key in ('timeout_seconds','max_input_bytes','max_records','max_samples','max_line_bytes'):
        require(type(r[key]) is int and r[key]>0,'Positive integer resource limit required: '+key)
    for key in ('memory_gib','disk_reserve_gib','max_output_gib'):
        require(type(r[key]) in (int,float) and math.isfinite(r[key]) and r[key]>0,'Positive resource limit required')
    require(c['code_sha256']=={name:sha(Path(__file__).parent/name) for name in CODE_FILES},'Fold code changed')
    batch_path,bm=evidence(c['batch'],'r02_lai_simulation_batch_v1',batch.STATUS)
    # The batch contains small metadata only, and all of it participates in the contract.
    batch_files={name:output(batch_path,bm,name) for name in bm['outputs_sha256']}
    requests=read(batch_files['requests.json'])
    require(requests.get('schema')=='r02_lai_simulation_requests_v1' and requests.get('launch_authorized') is False,
            'Non-executable development requests required')
    selected=[j for j in requests['requests'] if j['parameters']['chromosomes']==[chrom]]
    require(len({j['task_id'] for j in selected})==len(selected),'Duplicate chromosome requests')
    if c['schema']==SCHEMA:
        require(selected,'No unique chromosome requests')
    material_path,material=evidence(c['material'],'r02_lai_real_material_v1',
        'COMPLETE_PARENTAL_MATERIAL_NOT_SIMULATION_AUTHORIZATION')
    require(material['chromosome']==chrom and material.get('source_test_opened') is False and
        material.get('roles_changed') is False and material.get('genotypes_modified') is False,
            'Parental chromosome/roles/content mismatch')
    require(material.get('scope') in {'REAL_PARENTAL_PREPARATION','SYNTHETIC_CODE_TEST'},'Unexpected parental scope')
    require(material.get('reference_unknowns_preserved') is True and material.get('reference_genotype_filtering') is False,
            'Parental reference unknowns must be preserved without genotype filtering')
    original=sim.descriptor(c['original_roles']); report=sim.descriptor(c['material_report'])
    require(c['original_roles']['sha256']==requests['original_roles']['sha256'] and
            c['material_report']['sha256']==requests['original_material_report']['sha256'] and
            c['original_roles']['sha256'] in material['input_sha256'].values(),
            'Parentals, batch and material admission do not share original roles/report')
    planner.authenticate_report(c['material_report'])
    source=output(material_path,material,f'source_valid.chr{chrom}.vcf.gz')
    source_index=output(material_path,material,source.name+'.csi')
    genetic_map=output(material_path,material,'genetic.map')
    require(source_index==Path(str(source)+'.csi'),'Adjacent original index required')
    source_people=subset.header(source,r['max_line_bytes'])
    rows=list(batch.read_rows(batch_files['roles.private.tsv']))
    declared_folds=sorted({row['split'] for row in rows if row['role']=='SOURCE_VALID'})
    if c['schema']==SCHEMA_V2:
        # Extraction is reusable preparation, not authorization to add scenarios.
        # All original donor folds must be retained even for an autosome with
        # no request in the bounded simulation batch.
        require(isinstance(c['extract_folds'],list) and c['extract_folds']==declared_folds and
                all(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',x or '') for x in declared_folds),
                'Explicit extraction folds must equal the authenticated donor design')
        folds=c['extract_folds']
        require(all(j['parameters']['donor_split'] in folds for j in selected),
                'A requested donor fold is not prepared')
    else:
        folds=sorted({j['parameters']['donor_split'] for j in selected})
    all_donors={row['sample_id'] for row in rows if row['role']=='SOURCE_VALID'}
    require(len(source_people)==len(set(source_people)) and set(source_people)==all_donors and
            len(source_people)==material['role_counts']['SOURCE_VALID'] and len(source_people)<=r['max_samples'],
            'Parentals do not contain exactly the authorized full donor panel')
    require(subset.axis_hash(source_people)==material['sample_order_sha256']['SOURCE_VALID'],'Parental order changed')
    require(0<material['retained_records']<=r['max_records'],'Parental marker count exceeds cap')
    required=[contract_path,batch_path,material_path,original,report,source,source_index,genetic_map,*batch_files.values()]
    require(sum(p.stat().st_size for p in set(required))<=r['max_input_bytes'],'Input byte budget exceeded')
    versions=subprocess.check_output(['bcftools','--version'],text=True,timeout=30).splitlines()[0]
    require(versions==c['bcftools_version'],'bcftools version mismatch')
    indexed=int(subprocess.check_output(['bcftools','index','-n',str(source)],text=True,timeout=30))
    require(indexed==material['retained_records'],'Source index record count mismatch')
    out=Path(out).resolve();require(not out.exists() and out.parent.is_dir(),'Fresh fold output required')
    sim.local_filesystem(out.parent);batch.guard(out,r,started);out.mkdir(mode=0o700)
    shutil.copyfile(genetic_map,out/'genetic.map');shutil.copyfile(batch_files['roles.private.tsv'],out/'roles.private.tsv')
    commands=[]; results={}; release_rows=[]; previous_alarm=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):raise TimeoutError('Fold preparation exceeded its complete-stage timeout')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,r['timeout_seconds']-(time.monotonic()-started)))
    try:
        for fold in folds:
            sample_file=batch_files[fold+'.samples.private.txt']; ids=sample_file.read_text().splitlines()
            expected=sorted(row['sample_id'] for row in rows if row['role']=='SOURCE_VALID' and row['split']==fold)
            require(ids==expected and len(ids)==len(set(ids)) and ids,'Fold list differs from authenticated donor design')
            require({row['family_component'] for row in rows if row['role']=='SOURCE_VALID' and row['split']==fold}.isdisjoint(
                {row['family_component'] for row in rows if row['role']=='SOURCE_VALID' and row['split']!=fold}),
                'An original biological donor unit crosses folds')
            vcf=out/f'{fold}.chr{chrom}.vcf.gz'
            cmd=['bcftools','view','--no-update','--no-version','--samples-file',str(sample_file),
                 '--output-type','z','--threads','0','--output',str(vcf),str(source)]
            subprocess.run(cmd,check=True,capture_output=True,timeout=max(1,r['timeout_seconds']-(time.monotonic()-started)))
            commands.append(cmd)
            count,axis=subset.verify_projection(source,vcf,ids,material['retained_records'],r['max_line_bytes'],chromosome=chrom)
            require(axis==material['retained_axis_sha256'],'Fold allele/position axis differs from parental material')
            subprocess.run(['bcftools','index','--csi',str(vcf)],check=True,capture_output=True,timeout=60)
            require(int(subprocess.check_output(['bcftools','index','-n',str(vcf)],text=True,timeout=30))==count,
                    'Fold index record count mismatch')
            # Exercise the random-access index as well as its summary count.
            first_contig,first=next(line.split('\t')[:2] for line in subset.lines(vcf,r['max_line_bytes']) if not line.startswith('#'))
            hits=subprocess.check_output(['bcftools','query','--regions',f'{first_contig}:{first}-{first}','--format','%POS\n',str(vcf)],text=True,timeout=30).splitlines()
            require(hits and all(x==first for x in hits),'Fold index cannot retrieve the first marker')
            results[fold]=dict(vcf=desc(vcf),index=desc(Path(str(vcf)+'.csi')),people=len(ids),records=count,
                              allele_axis_sha256=axis,sample_order_sha256=subset.axis_hash(ids))
            batch.guard(out,r,started)
        for j in selected:
            parameters=j['parameters'];fold=parameters['donor_split']
            require(parameters['genome_build']==material['genome_build'] and
                    parameters['input_policy']==material['input_policy'],'Scenario/material assembly or filter policy differs')
            model=batch_files[j['model']['file']]
            require(sha(model)==j['model']['sha256'] and
                    sha(batch_files['roles.private.tsv'])==j['roles']['sha256'],'Request model/roles changed')
            simulated_people,_=sim.read_model(model,parameters['resources']['popsize'])
            generations=int(model.read_text().splitlines()[-1].split()[0])
            genotype_cells=2*(results[fold]['people']+simulated_people)*material['retained_records']
            map_points=len(sim.read_map(out/'genetic.map',chrom))
            proxy_work=parameters['resources']['popsize']*generations*map_points
            require(material['retained_records']<=parameters['resources']['max_variants'] and
                    genotype_cells<=parameters['resources']['max_genotype_cells'] and
                    proxy_work<=parameters['resources']['max_generation_marker_cells'],
                    'Real parental dimensions exceed requested simulation bounds')
            scenario={**parameters,'inputs':dict(donor_vcf=results[fold]['vcf'],donor_index=results[fold]['index'],
                roles=desc(out/'roles.private.tsv'),model=desc(model),maps={chrom:desc(out/'genetic.map')})}
            scenario_path=out/(j['task_id']+'.scenario.json');write(scenario_path,scenario)
            contract=dict(schema=planner.SCHEMA,material_report=c['material_report'],original_roles=c['original_roles'],
                          scenario=desc(scenario_path),review=None)
            plan_path=out/(j['task_id']+'.plan-input.json');write(plan_path,contract)
            result=planner.prepare(plan_path,sha(plan_path),out/(j['task_id']+'.plan'))
            require(result['status']=='PLAN_REQUIRES_EXPLICIT_REVIEW','Unexpected executable simulation plan')
            release_rows.append(dict(task_id=j['task_id'],parental_integrity='VERIFIED',fold_identity='VERIFIED',
                all_parental_sites_and_selected_gt='TEXT_IDENTICAL',index='VERIFIED',map='AUTHENTICATED_NO_NEW_EXTRAPOLATION',
                source_genotypes=material['retained_records'],selected_donors=results[fold]['people'],map_points=map_points,
                genotype_cells_bound_proxy=genotype_cells,popsize_generation_map_points_proxy=proxy_work,
                work_proxy_is_measured_operations=False,resource_bounds='DIMENSIONS_WITHIN_CAPS_NOT_RUNTIME_BENCHMARK',
                reference_unknowns_preserved=material['reference_unknowns_preserved'],
                scientific_review='REQUIRES_EXPLICIT_CONDITIONAL_SCENARIO_REVIEW',
                inherited_criteria=result['summary']['inherited_limited_or_unresolved_criteria'],
                resource_release=j.get('resource_release','SYNTHETIC_FIXTURE_ONLY'),
                simulation_executable=False))
        write(out/'release_checklist.json',dict(status='PREPARATION_CHECKED_NOT_SIMULATION_RELEASED',tasks=release_rows,
            chromosome=chrom,extracted_folds=folds,no_simulation_requested_for_chromosome=not selected,
            required_review_checks=list(sim.CHECKS),
            phase_notation_does_not_certify_accuracy=True,gt_only_does_not_certify_quality=True,
            donor_count_does_not_certify_population_power=True,
            refs_unknown_must_remain_unknown=True,
            no_scientific_validation_from_operational_checks=True))
        batch.guard(out,r,started)
        for entry in (c['batch'],c['material'],c['original_roles'],c['material_report']):sim.descriptor(entry)
        for name in (source.name,source_index.name,'genetic.map'):output(material_path,material,name)
        for name in bm['outputs_sha256']:output(batch_path,bm,name)
        sim.descriptor(dict(path=str(contract_path),sha256=digest))
        summary=dict(schema='r02_lai_fold_material_v1',status=STATUS,chromosome=chrom,
            input_contract_sha256=digest,parental_manifest=c['material'],batch_manifest=c['batch'],
            folds=results,plans_prepared=len(selected),simulations_executed=False,training_executed=False,
            extraction_contract_schema=c['schema'],simulation_requests_added=0,
            no_simulation_requested_for_chromosome=not selected,
            source_test_opened=False,donor_roles_changed=False,donor_units_changed=False,
            selected_genotype_text_preserved=True,all_parental_sites_retained=True,monomorphic_sites_retained=True,
            inherited_info_scope=material['inherited_info_scope'],
            source_genotype_scope='ALL_SOURCE_VALID_DONORS',source_donor_people=len(source_people),
            inherited_info_is_fold_frequency=False,
            inherited_info_policy='DO_NOT_USE_AS_FOLD_AC_AN_AF; compute separate GT-derived counts if needed; do not redefine the original rare catalogue',
            plan_path_scope='LOCAL_VERIFIED_WORK_PATHS; preserve work/cache or rebind exact bytes by hash after relocating; not directly executable from GCS',
            input_policy=material['input_policy'],
            bcftools_version=versions,commands=commands,elapsed_seconds=time.monotonic()-started,
            contains_sample_identifiers=True,public_distribution_allowed=False,
            code_sha256=c['code_sha256'],outputs_sha256={p.relative_to(out).as_posix():sha(p)
                for p in sorted(out.rglob('*')) if p.is_file()})
        write(out/'manifest.json',summary);return summary
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);prepare(args.contract,args.contract_sha256,args.output_dir)


if __name__=='__main__':main()
