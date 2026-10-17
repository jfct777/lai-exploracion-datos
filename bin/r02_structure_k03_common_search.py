#!/usr/bin/env python3
"""FIT-only Gram scale, finite Ridge family, and complete-family SELECT closure.

The source is an immutable common-only bridge. Policy and resource values have
no defaults. EVALUATE arrays are never read or accepted by this adapter.
"""
from pathlib import Path
import argparse
import json
import math
import os
import shutil

import numpy as np

import r02_pair_operator as operator
import r02_ridge_finite_family as family
import r02_structure_increment as core
import r02_structure_k03_fit as fit

require=core.require
CODE_FILES=('r02_structure_k03_common_search.py','r02_ridge_finite_family.py',
            'r02_structure_k03_fit.py','r02_structure_k03_design.py',
            'r02_structure_increment.py','r02_pair_operator.py','r02_pair_preconditioner.py')


def implementation():
    return {name:core.sha(Path(__file__).with_name(name)) for name in CODE_FILES}


def _hex(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _pin(path,digest,limit,pins):
    path=Path(path).resolve()
    require(_hex(digest) and path.is_file() and path.stat().st_size<=limit and
            core.sha(path)==digest,'Pinned common-search input changed or exceeds bound')
    pins[str(path)]=digest
    return path


def _load_source(request,pins):
    limits=request['resources'];desc=request['source']
    require(set(desc)=={'directory','manifest_sha256'},'Exact immutable bridge descriptor required')
    root=Path(desc['directory']).resolve()
    manifest=core.load_json(_pin(root/'manifest.json',desc['manifest_sha256'],limits['max_metadata_bytes'],pins))
    require(manifest.get('schema')=='r02_k03_fit_select_arrays_v1' and
            manifest.get('stage')=='FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS' and
            manifest.get('evidence_scope')==request['evidence_scope'] and
            manifest.get('scientific_spec_sha256')==request['source_scientific_spec_sha256'] and
            manifest.get('evaluate_targets_computed') is False and
            manifest.get('alpha_selection_performed') is False and
            manifest.get('arm')=='CONTINUOUS' and
            set(manifest.get('outputs_sha256',{}))=={'persons.tsv','arrays.private.npz'},
            'Source is not the declared common-only FIT/SELECT bridge')
    shim=dict(schema='r02_k03_fit_select_request_v1',evidence_scope=request['evidence_scope'],
              scientific_spec_sha256=request['source_scientific_spec_sha256'])
    persons,points,records,_=fit.load_legacy_inputs(desc,
        lambda path,digest,limit:_pin(path,digest,limit,pins),shim,limits)
    return root,manifest,persons,points,records


def _validate_request(request,request_sha256,admission):
    required={'schema','evidence_scope','scientific_spec_sha256','source_scientific_spec_sha256',
              'source','implementation_sha256','fit_implementation_sha256','policy',
              'parameters','resources'}
    require(set(request)==required and request['schema']=='r02_k03_common_finite_search_request_v1'
            and request['implementation_sha256']==implementation()
            and request['fit_implementation_sha256']==fit.implementation()
            and request['evidence_scope'] in {'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'}
            and _hex(request['scientific_spec_sha256']) and
            _hex(request['source_scientific_spec_sha256']),
            'Exact common finite-search request/code required')
    require(admission==dict(schema='r02_k03_common_finite_search_admission_v1',
            status='FIT_SELECT_FINITE_FAMILY_ADMITTED',evidence_scope=request['evidence_scope'],
            scientific_spec_sha256=request['scientific_spec_sha256'],
            request_sha256=request_sha256,evaluate_allowed=False),
            'Exact external FIT/SELECT finite-family admission required')
    policy=request['policy']
    require(set(policy)=={'epsilon','subdivisions_per_decade','guard_decades','max_penalties',
            'plausibility_rule','selection_rule','boundary_action'} and
            policy['plausibility_rule']=='LOWER_BOUND_LE_MINIMUM_UPPER_BOUND' and
            policy['selection_rule']=='RETAIN_NUMERICALLY_PLAUSIBLE_SET' and
            policy['boundary_action']=='REQUIRE_EXTENSION_NO_SELECTION',
            'Explicit reviewed finite-family selection policy required')
    limits=request['resources']
    require(set(limits)=={'max_metadata_bytes','max_array_bytes','max_input_bytes',
            'max_output_bytes','min_free_bytes','task_cpus','statistics_max_seconds'} and
            all(type(limits[k]) is int and limits[k]>0 for k in limits),
            'Explicit positive search resource limits required')


def prepare(request_path,request_sha256,admission_path,admission_sha256,output):
    """Compute FIT statistics once and materialize an authenticated model batch."""
    os.umask(0o077);pins={}
    request=core.load_json(_pin(request_path,request_sha256,2**20,pins))
    admission=core.load_json(_pin(admission_path,admission_sha256,2**20,pins))
    _validate_request(request,request_sha256,admission)
    root,source_manifest,persons,points,records=_load_source(request,pins)
    roles=core.validate_roles(persons)
    i,j,_,_,weights,extra=fit.role_pairs(persons,'FIT',records['FIT'])
    indices=np.asarray(roles['FIT'],dtype=np.int64)
    pair_design=operator.PairDesign(np.asarray(points)[indices],np.searchsorted(indices,i),
        np.searchsorted(indices,j),extra=extra,**{k:request['parameters'][k] for k in
        ('rows_per_chunk','max_working_mib','backend','preprocessing_backend','workers')})
    identity=family.content_sha256(dict(source_manifest_sha256=request['source']['manifest_sha256'],
        source_outputs_sha256=source_manifest['outputs_sha256'],role='FIT'))
    stats=family.fit_gram_statistics(pair_design,weights,role='FIT',input_identity=identity,
        max_seconds=request['resources']['statistics_max_seconds'])
    stats_hash=family.content_sha256(stats);policy=request['policy']
    plan=family.finite_family(stats,expected_statistics_sha256=stats_hash,
        epsilon=policy['epsilon'],subdivisions_per_decade=policy['subdivisions_per_decade'],
        guard_decades=policy['guard_decades'],max_penalties=policy['max_penalties'])
    require(plan['status']=='PLANNED_NOT_FITTED' and plan['penalties'],
            'Finite family is not runnable under the admitted penalty cap')
    out=Path(output).resolve()
    require(not out.exists() and out.parent.is_dir() and
            shutil.disk_usage(out.parent).free>=request['resources']['min_free_bytes'],
            'New search output and disk reserve required')
    out.mkdir(mode=0o700);(out/'models').mkdir()
    core.write_json(out/'statistics.json',stats);core.write_json(out/'family.json',plan)
    family_hash=core.sha(out/'family.json');models=[]
    for index,penalty in enumerate(plan['penalties']):
        safe_key=penalty['key'].replace('-','neg')
        model_id=f"p{index:03d}_{safe_key}";model_dir=out/'models'/model_id
        model_dir.mkdir()
        semantics=dict(schema='r02_ridge_finite_penalty_binding_v1',family_sha256=family_hash,
            penalty_key=penalty['key'],normalized_lambda=penalty['normalized_lambda'],
            fit_weight_sum=plan['weight_sum'],alpha_derivation='alpha=fit_weight_sum*normalized_lambda')
        semantics.update(source_scientific_spec_sha256=request['source_scientific_spec_sha256'],
                         source_manifest_sha256=request['source']['manifest_sha256'])
        model_request=dict(schema='r02_k03_fit_select_request_v3',
            evidence_scope=request['evidence_scope'],scientific_spec_sha256=request['scientific_spec_sha256'],
            inputs=request['source'],implementation_sha256=request['fit_implementation_sha256'],
            alpha=penalty['alpha'],penalty_semantics=semantics,
            parameters=request['parameters'],resources={k:request['resources'][k] for k in
            ('max_metadata_bytes','max_array_bytes','max_input_bytes','max_output_bytes','min_free_bytes','task_cpus')})
        core.write_json(model_dir/'request.json',model_request);request_pin=core.sha(model_dir/'request.json')
        model_admission=dict(schema='r02_k03_fit_select_admission_v1',status='FIT_SELECT_ONLY_ADMITTED',
            evidence_scope=request['evidence_scope'],scientific_spec_sha256=request['scientific_spec_sha256'],
            request_sha256=request_pin,evaluate_allowed=False)
        core.write_json(model_dir/'admission.json',model_admission)
        models.append(dict(id=model_id,penalty_key=penalty['key'],request=str(model_dir/'request.json'),
            request_sha256=request_pin,admission=str(model_dir/'admission.json'),
            admission_sha256=core.sha(model_dir/'admission.json')))
    batch=dict(schema='r02_k03_common_finite_fit_batch_v1',
        scientific_spec_sha256=request['scientific_spec_sha256'],family_root_sha256=family_hash,
        current_family_sha256=family_hash,
        source_manifest_sha256=request['source']['manifest_sha256'],
        implementation_sha256=request['fit_implementation_sha256'],models=models,
        evaluate_allowed=False)
    core.write_json(out/'fit-batch.json',batch)
    binding=dict(schema='r02_k03_common_finite_search_binding_v1',request_sha256=request_sha256,
        admission_sha256=admission_sha256,source_directory=str(root),
        source_manifest_sha256=request['source']['manifest_sha256'],
        source_scientific_spec_sha256=request['source_scientific_spec_sha256'],source_input_identity=identity,
        statistics_sha256=core.sha(out/'statistics.json'),family_root_sha256=family_hash,
        current_family_sha256=family_hash,
        fit_batch_sha256=core.sha(out/'fit-batch.json'),n_penalties=len(models),
        alpha_derivation='alpha=fit_weight_sum*normalized_lambda',evaluate_targets_computed=False)
    core.write_json(out/'binding.json',binding)
    require(all(core.sha(p)==h for p,h in pins.items()) and implementation()==request['implementation_sha256'],
            'Search inputs/code changed during FIT statistics')
    outputs={str(p.relative_to(out)):core.sha(p) for p in out.rglob('*') if p.is_file()}
    require(sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=request['resources']['max_output_bytes'],
            'Search planning outputs exceed explicit byte bound')
    receipt=dict(schema='r02_k03_common_finite_search_plan_v1',stage='FIT_STATISTICS_AND_FINITE_BATCH_COMPLETE',
        request_sha256=request_sha256,admission_sha256=admission_sha256,
        scientific_spec_sha256=request['scientific_spec_sha256'],source_manifest_sha256=request['source']['manifest_sha256'],
        statistics_sha256=binding['statistics_sha256'],family_root_sha256=family_hash,
        current_family_sha256=family_hash,
        fit_batch_sha256=binding['fit_batch_sha256'],n_penalties=len(models),outputs_sha256=outputs,
        complete_family_fitted=False,selection_performed=False,evaluate_targets_computed=False,
        biological_validation=False)
    core.write_json(out/'manifest.json',receipt)
    return receipt


def select(plan_directory,fit_directories,output):
    """Close SELECT only after every family member completed and passed bounds."""
    root=Path(plan_directory).resolve();binding=core.load_json(root/'binding.json')
    plan=core.load_json(root/'family.json');batch=core.load_json(root/'fit-batch.json')
    require(core.sha(root/'family.json')==binding['current_family_sha256'] and
            core.sha(root/'fit-batch.json')==binding['fit_batch_sha256'] and
            plan['status']=='PLANNED_NOT_FITTED' and len(batch['models'])==plan['n_penalties'],
            'Pinned finite-family plan/batch required')
    require(len(fit_directories)==len(batch['models']),'Exact finite-family result count required')
    by_request={};rows=[]
    for directory in map(Path,fit_directories):
        manifest=core.load_json(directory/'manifest.json');metrics=core.load_json(directory/'metrics.json')
        require(manifest['schema']=='r02_k03_fit_select_result_v2' and
                manifest['stage']=='ONE_MODEL_COMPLETE_SELECTION_PENDING' and
                manifest['evaluate_targets_computed'] is False and
                manifest['alpha_selection_performed'] is False and
                all((directory/name).is_file() and core.sha(directory/name)==digest
                    for name,digest in manifest['outputs_sha256'].items()),
                'Incomplete or altered finite-family fit result')
        require(manifest['request_sha256'] not in by_request,'Duplicate finite-family result forbidden')
        by_request[manifest['request_sha256']]=(directory,manifest,metrics)
    require(len(by_request)==len(batch['models']) and
            set(by_request)=={m['request_sha256'] for m in batch['models']},
            'Every declared penalty must complete exactly once before SELECT')
    models_by_key={model['penalty_key']:model for model in batch['models']}
    require(len(models_by_key)==len(batch['models']) and
            set(models_by_key)=={penalty['key'] for penalty in plan['penalties']},
            'Batch and finite-family penalty keys differ')
    for penalty in plan['penalties']:
        model=models_by_key[penalty['key']]
        directory,manifest,metrics=by_request[model['request_sha256']];interval=metrics['numerical_loss']
        model_request=core.load_json(model['request']);model_admission=core.load_json(model['admission'])
        diagnostics=metrics['numerical_diagnostics']
        require(core.sha(model['request'])==model['request_sha256'] and
                core.sha(model['admission'])==model['admission_sha256'] and
                model_admission['request_sha256']==model['request_sha256'] and
                model_admission['evaluate_allowed'] is False and
                manifest['admission_sha256']==model['admission_sha256'] and
                manifest['scientific_spec_sha256']==batch['scientific_spec_sha256'] and
                manifest['implementation_sha256']==batch['implementation_sha256'] and
                manifest['inputs_manifest_sha256']==binding['source_manifest_sha256'] and
                model_request['penalty_semantics']['source_manifest_sha256']==binding['source_manifest_sha256'] and
                model_request['penalty_semantics']['source_scientific_spec_sha256']==binding['source_scientific_spec_sha256'] and
                diagnostics['lsqr_status'] in (0,1,2,4,5) and
                diagnostics['relative_penalized_gradient']<=model_request['parameters']['gradient_tolerance'],
                'Fit request/admission/source or numerical convergence differs')
        require(metrics['penalty_key']==penalty['key'] and
                metrics['finite_family_sha256']==binding['family_root_sha256'] and
                math.isclose(metrics['normalized_lambda'],penalty['normalized_lambda'],rel_tol=1e-14) and
                interval['exact_ridge_mse_lower']<=interval['weighted_mse']<=interval['exact_ridge_mse_upper'] and
                math.isfinite(interval['exact_ridge_mse_lower']) and math.isfinite(interval['exact_ridge_mse_upper']),
                'Fit penalty binding or numerical loss interval differs')
        rows.append(dict(model_id=model['id'],penalty_key=penalty['key'],
            normalized_lambda=penalty['normalized_lambda'],alpha=penalty['alpha'],
            weighted_select_mse=metrics['weighted_select_mse'],
            numerical_lower=interval['exact_ridge_mse_lower'],numerical_upper=interval['exact_ridge_mse_upper'],
            result_manifest_sha256=core.sha(directory/'manifest.json')))
    min_upper=min(row['numerical_upper'] for row in rows)
    plausible=[row for row in rows if row['numerical_lower']<=min_upper]
    proposal=family.propose_extension(plan,expected_plan_sha256=family.content_sha256(plan),
        completed_penalty_keys=[r['penalty_key'] for r in rows],
        plausible_penalty_keys=[r['penalty_key'] for r in plausible])
    selected=None
    if proposal['status']=='DECLARED_FINITE_SEARCH_INTERIOR':
        status='FINITE_FAMILY_SELECT_COMPLETE'
    else:
        status='EXTENSION_REQUIRED_NO_SELECTION'
    out=Path(output).resolve();require(not out.exists() and out.parent.is_dir(),'New SELECT output required')
    out.mkdir(mode=0o700)
    result=dict(schema='r02_k03_common_finite_selection_v1',status=status,
        family_root_sha256=binding['family_root_sha256'],current_family_sha256=binding['current_family_sha256'],fit_batch_sha256=binding['fit_batch_sha256'],
        numerical_plausibility_rule='LOWER_BOUND_LE_MINIMUM_UPPER_BOUND',minimum_numerical_upper=min_upper,
        plausible_penalty_keys=[r['penalty_key'] for r in plausible],selected=selected,
        extension=proposal,models=rows,complete_family_fitted=True,
        select_used_for_model_choice=True,winner_selected=False,evaluate_targets_computed=False,
        biological_validation=False,scope='ONE_COMMON_CONTINUOUS_VIEW_FIT_SELECT_ONLY')
    core.write_json(out/'selection.json',result)
    core.write_json(out/'manifest.json',dict(schema='r02_k03_common_finite_selection_receipt_v1',
        stage=status,selection_sha256=core.sha(out/'selection.json'),family_root_sha256=binding['family_root_sha256'],current_family_sha256=binding['current_family_sha256'],
        complete_family_fitted=True,evaluate_targets_computed=False,biological_validation=False))
    return result


def extend(plan_directory,selection_directory,output):
    """Materialize only boundary-extension requests while retaining old results."""
    root=Path(plan_directory).resolve();selection_root=Path(selection_directory).resolve()
    binding=core.load_json(root/'binding.json');batch=core.load_json(root/'fit-batch.json')
    selection_receipt=core.load_json(selection_root/'manifest.json')
    selection=core.load_json(selection_root/'selection.json')
    require(core.sha(root/'family.json')==binding['current_family_sha256'] and
            core.sha(root/'fit-batch.json')==binding['fit_batch_sha256'] and
            selection_receipt['schema']=='r02_k03_common_finite_selection_receipt_v1' and
            core.sha(selection_root/'selection.json')==selection_receipt['selection_sha256'] and
            all(core.sha(model['request'])==model['request_sha256'] and
                core.sha(model['admission'])==model['admission_sha256'] for model in batch['models']),
            'Pinned parent plan, selection receipt and model requests required')
    require(selection['status']=='EXTENSION_REQUIRED_NO_SELECTION' and
            selection['current_family_sha256']==binding['current_family_sha256'] and
            selection['fit_batch_sha256']==binding['fit_batch_sha256'] and
            selection['extension']['status']=='NEW_REQUESTS_REQUIRE_FAMILY_BARRIER',
            'Authenticated boundary extension request required')
    amended=selection['extension']['amended_plan'];new_penalties=selection['extension']['required_new_penalties']
    require(new_penalties and amended['status']=='PLANNED_NOT_FITTED','Nonempty runnable extension required')
    out=Path(output).resolve();require(not out.exists() and out.parent.is_dir(),'New extension output required')
    out.mkdir(mode=0o700);(out/'models').mkdir()
    core.write_json(out/'family.json',amended);current_hash=core.sha(out/'family.json')
    template=core.load_json(batch['models'][0]['request']);models=list(batch['models']);new_models=[]
    extension_number=len(amended['extensions'])
    for index,penalty in enumerate(new_penalties):
        safe_key=penalty['key'].replace('-','neg');model_id=f"e{extension_number:02d}_{index:03d}_{safe_key}"
        model_dir=out/'models'/model_id;model_dir.mkdir()
        semantics=dict(template['penalty_semantics'],penalty_key=penalty['key'],
            normalized_lambda=penalty['normalized_lambda'],fit_weight_sum=amended['weight_sum'])
        request=dict(template,alpha=penalty['alpha'],penalty_semantics=semantics)
        core.write_json(model_dir/'request.json',request);request_pin=core.sha(model_dir/'request.json')
        admission=dict(schema='r02_k03_fit_select_admission_v1',status='FIT_SELECT_ONLY_ADMITTED',
            evidence_scope=request['evidence_scope'],scientific_spec_sha256=request['scientific_spec_sha256'],
            request_sha256=request_pin,evaluate_allowed=False)
        core.write_json(model_dir/'admission.json',admission)
        descriptor=dict(id=model_id,penalty_key=penalty['key'],request=str(model_dir/'request.json'),
            request_sha256=request_pin,admission=str(model_dir/'admission.json'),
            admission_sha256=core.sha(model_dir/'admission.json'))
        models.append(descriptor);new_models.append(descriptor)
    full_batch=dict(batch,current_family_sha256=current_hash,models=models)
    core.write_json(out/'fit-batch.json',full_batch)
    core.write_json(out/'new-fit-batch.json',dict(full_batch,models=new_models))
    new_binding=dict(binding,parent_current_family_sha256=binding['current_family_sha256'],
        current_family_sha256=current_hash,fit_batch_sha256=core.sha(out/'fit-batch.json'),
        n_penalties=len(models))
    core.write_json(out/'binding.json',new_binding)
    receipt=dict(schema='r02_k03_common_finite_extension_plan_v1',stage='BOUNDARY_EXTENSION_REQUESTS_READY',
        family_root_sha256=binding['family_root_sha256'],parent_current_family_sha256=binding['current_family_sha256'],
        current_family_sha256=current_hash,fit_batch_sha256=new_binding['fit_batch_sha256'],
        new_fit_batch_sha256=core.sha(out/'new-fit-batch.json'),new_penalties=len(new_models),
        total_penalties=len(models),prior_results_retained=True,selection_performed=False,
        evaluate_targets_computed=False,biological_validation=False)
    core.write_json(out/'manifest.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('request','request-sha256','admission','admission-sha256','output'):p.add_argument('--'+name,required=True)
    s=sub.add_parser('select');s.add_argument('--plan-directory',required=True);s.add_argument('--fit-directory',action='append',required=True);s.add_argument('--output',required=True)
    e=sub.add_parser('extend');e.add_argument('--plan-directory',required=True);e.add_argument('--selection-directory',required=True);e.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        result=prepare(args.request,args.request_sha256,args.admission,args.admission_sha256,args.output)
    elif args.command=='select':result=select(args.plan_directory,args.fit_directory,args.output)
    else:result=extend(args.plan_directory,args.selection_directory,args.output)
    print(json.dumps(result,sort_keys=True))
