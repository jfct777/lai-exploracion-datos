#!/usr/bin/env python3
"""One resumable K03 Ridge FIT/SELECT task; EVALUATE is deliberately absent.

Numerical API only until an authenticated upstream preparation and explicit
scientific admission supply its inputs. Every alpha/candidate is a separate
checkpoint; interrupted solves are repeated, completed models are reusable.
"""
from pathlib import Path
import json
import math
import os
import time
import numpy as np
import r02_structure_increment as core
import r02_pair_operator as operator
import r02_pair_preconditioner as preconditioner
import r02_structure_k03_design as design

require=core.require


def _write_progress(path, value):
    """Replace a small attempt receipt without changing scientific inputs."""
    if path is None:return
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temporary,path)


def role_pairs(persons, role, record):
    """Reject crossed roles, family pairs and changed weighting before fitting."""
    require(role in {'FIT','SELECT'} and set(record)=={'i','j','y','n','w','extra'},
            'Explicit FIT/SELECT pair arrays required; EVALUATE not accepted')
    roles=core.validate_roles(persons);i,j,y,n,w,extra=(np.asarray(record[k])for k in ('i','j','y','n','w','extra'))
    require(i.ndim==j.ndim==y.ndim==n.ndim==w.ndim==1 and i.shape==j.shape==y.shape==n.shape==w.shape and len(i)>0,
            'Pair target/denominator dimensions differ')
    require(i.dtype.kind in 'iu'and j.dtype.kind in 'iu'and n.dtype.kind in 'iu'and
            np.all(n>0)and np.all(i<j)and np.isin(i,roles[role]).all()and np.isin(j,roles[role]).all(),
            'Target pairs must be canonical and belong only to the declared role')
    keys=i*len(persons)+j
    require(np.all(np.diff(keys)>0),'Ordered unique target pairs required')
    components=[p['component']for p in persons]
    require(all(components[a]!=components[b]for a,b in zip(i,j)),'Within-component target pair forbidden')
    expected,_=core.component_pair_weights(i,j,components)
    require(np.isfinite(y).all()and np.isfinite(w).all()and np.allclose(w,expected,rtol=1e-13,atol=0),
            'Finite targets and original component-pair weights required')
    require(extra.ndim==2 and len(extra)==len(i)and np.isfinite(extra).all(),'Finite FIT-imputed paired controls required')
    return i,j,y,n,w,extra


def _penalty_metadata(penalty_semantics, alpha, fit_weight_sum):
    """Validate an optional finite-family alpha binding; preserve legacy meaning."""
    if penalty_semantics is None:
        return dict(
            alpha_interpretation='ABSOLUTE_INHERITED_NUMERICAL_PROFILE_POINT_NOT_HUMAN_NORMALIZED_LAMBDA')
    required={'schema','family_sha256','penalty_key','normalized_lambda',
              'fit_weight_sum','alpha_derivation','source_scientific_spec_sha256',
              'source_manifest_sha256'}
    require(set(penalty_semantics)==required and
            penalty_semantics['schema']=='r02_ridge_finite_penalty_binding_v1' and
            penalty_semantics['alpha_derivation']=='alpha=fit_weight_sum*normalized_lambda' and
            isinstance(penalty_semantics['family_sha256'],str) and
            len(penalty_semantics['family_sha256'])==64 and
            isinstance(penalty_semantics['source_scientific_spec_sha256'],str) and
            len(penalty_semantics['source_scientific_spec_sha256'])==64 and
            isinstance(penalty_semantics['source_manifest_sha256'],str) and
            len(penalty_semantics['source_manifest_sha256'])==64 and
            isinstance(penalty_semantics['penalty_key'],str) and
            penalty_semantics['penalty_key'], 'Invalid finite-family penalty binding')
    lam=penalty_semantics['normalized_lambda'];bound_w=penalty_semantics['fit_weight_sum']
    require(type(lam) in (int,float) and math.isfinite(lam) and lam>0 and
            type(bound_w) in (int,float) and math.isfinite(bound_w) and bound_w>0 and
            math.isclose(bound_w,fit_weight_sum,rel_tol=1e-14,abs_tol=0) and
            math.isclose(alpha,bound_w*lam,rel_tol=1e-14,abs_tol=0),
            'Finite-family alpha does not equal authenticated FIT W*lambda')
    return dict(alpha_interpretation='FINITE_FAMILY_ALPHA_EQUALS_FIT_WEIGHT_SUM_TIMES_NORMALIZED_LAMBDA',
                normalized_lambda=float(lam),finite_family_sha256=penalty_semantics['family_sha256'],
                penalty_key=penalty_semantics['penalty_key'],alpha_derivation=penalty_semantics['alpha_derivation'],
                source_scientific_spec_sha256=penalty_semantics['source_scientific_spec_sha256'],
                source_manifest_sha256=penalty_semantics['source_manifest_sha256'])


def fit_select_arrays(persons,points,fit,select,*,alpha,parameters,task_cpus,
                      progress_path=None,penalty_semantics=None):
    """Fit one specified model, preserving the complete supplied feature family.

    FIT and SELECT use separately indexed person matrices, so query points
    cannot affect FIT preprocessing and do not inflate FIT operator dimensions.
    No alpha search, rank truncation, random seed or fallback is chosen here.
    """
    started=time.monotonic();roles=core.validate_roles(persons)
    x=np.asarray(points,dtype=float)
    require(x.ndim==2 and len(x)==len(persons)and np.isfinite(x).all(),'Finite full-roster points required')
    fields={'rows_per_chunk','max_working_mib','backend','preprocessing_backend',
            'atol','btol','max_iter','gradient_tolerance','residual_roundoff_allowance','workers'}
    extended=fields|{'solver','max_preconditioner_mib'}
    require(set(parameters)in(fields,extended),'Exact explicit numerical settings required; no defaults')
    # Historical requests retain their original solver. New solver requests must
    # name both the algorithm and its workspace; failure never triggers fallback.
    solver=parameters['solver'] if set(parameters)==extended else 'LSQR'
    require(solver in {'LSQR','EXACT_MEAN_BLOCK_LSQR'},'Unknown explicit K03 solver')
    if set(parameters)==extended:
        workspace=parameters['max_preconditioner_mib']
        require(type(workspace)in(int,float) and np.isfinite(workspace) and
                ((solver=='LSQR' and workspace==0) or
                 (solver=='EXACT_MEAN_BLOCK_LSQR' and workspace>0)),
                'Explicit solver workspace differs: LSQR=0; preconditioned solver>0 MiB')
    require(type(task_cpus)is int and type(parameters['workers'])is int and 0<parameters['workers']<=task_cpus,
            'Explicit numerical workers must fit reserved task CPUs')
    require(type(alpha)in(int,float)and np.isfinite(alpha)and alpha>0,'Explicit positive alpha required')
    records={r:role_pairs(persons,r,d)for r,d in [('FIT',fit),('SELECT',select)]}
    require(records['FIT'][-1].shape[1]==records['SELECT'][-1].shape[1],'FIT/SELECT paired columns differ')
    designs={}
    for role in ('FIT','SELECT'):
        i,j,_,_,_,extra=records[role];indices=np.asarray(roles[role],np.int64)
        left,right=np.searchsorted(indices,i),np.searchsorted(indices,j)
        designs[role]=operator.PairDesign(x[indices],left,right,extra=extra,
            **{k:parameters[k]for k in ('rows_per_chunk','max_working_mib','backend','preprocessing_backend','workers')})
    _,_,fy,_,fw,_=records['FIT'];si,sj,sy,sn,sw,_=records['SELECT']
    penalty_metadata=_penalty_metadata(penalty_semantics,float(alpha),float(fw.sum()))
    solve=preconditioner.fit_mean_preconditioned if solver=='EXACT_MEAN_BLOCK_LSQR' else operator.fit_ridge
    solver_args={k:parameters[k]for k in ('atol','btol','max_iter','gradient_tolerance')}
    if solver=='EXACT_MEAN_BLOCK_LSQR':
        solver_args['max_preconditioner_mib']=parameters['max_preconditioner_mib']
    progress=dict(stage='SOLVER_STARTED_FULL_INFORMATION',solver=solver,
        alpha=float(alpha),parameters=parameters,fit_pairs=len(fy),select_pairs=len(sy),
        fit_weight_sum=float(fw.sum()),alpha_over_fit_weight_sum=float(alpha/fw.sum()),
        **penalty_metadata,
        individual_axes=x.shape[1],pair_columns=designs['FIT'].shape[1],
        fit_operator_people=len(roles['FIT']),select_operator_people=len(roles['SELECT']),
        evaluate_targets_computed=False,features_truncated=False,
        started_monotonic=started)
    _write_progress(progress_path,progress)
    try:model=solve(designs['FIT'],fy,fw,alpha,**solver_args)
    except Exception as error:
        _write_progress(progress_path,dict(progress,stage='SOLVER_FAILED_OR_STOPPED_NO_MODEL',
            elapsed_seconds=time.monotonic()-started,error_type=type(error).__name__,error=str(error)))
        raise
    prediction=model.predict(designs['SELECT'])
    numerical=model.numerical_loss_interval(designs['SELECT'],sy,sw,
        residual_roundoff_allowance=parameters['residual_roundoff_allowance'])
    _write_progress(progress_path,dict(progress,stage='SOLVER_COMPLETE_NUMERICALLY_ADMITTED',
        elapsed_seconds=time.monotonic()-started,numerical_diagnostics=model.diagnostics))
    return dict(model=model,predictions=prediction,
        metrics=dict(alpha=float(alpha),weighted_select_mse=core.macro_loss(sy,prediction,sw),
            fit_pairs=len(fy),select_pairs=len(sy),select_min_joint_markers=int(sn.min()),
            fit_weight_sum=float(fw.sum()),select_weight_sum=float(sw.sum()),
            alpha_over_fit_weight_sum=float(alpha/fw.sum()),
            **penalty_metadata,
            individual_axes=x.shape[1],pair_columns=designs['FIT'].shape[1],
            fit_operator_people=len(roles['FIT']),select_operator_people=len(roles['SELECT']),
            numerical_loss=numerical,numerical_diagnostics=model.diagnostics,
            seconds=time.monotonic()-started,evaluate_targets_computed=False,
            selection_performed=False,biological_validation=False,
            scope='ONE_DECLARED_CANDIDATE_ALPHA_FIT_SELECT_NOT_COMPLETE_K03'))


CODE_FILES=('r02_structure_k03_fit.py','r02_structure_k03_design.py','r02_structure_increment.py',
            'r02_pair_operator.py','r02_pair_preconditioner.py')

def implementation():
    return {name:core.sha(Path(__file__).with_name(name))for name in CODE_FILES}


def run_request(request_path,request_sha256,admission_path,admission_sha256,output):
    """Authenticated one-model checkpoint. Real data always require admission."""
    import json,os,shutil,zipfile
    os.umask(0o077)
    pins={}
    def pin(path,digest,limit):
        path=Path(path).resolve()
        require(core.sha(path)==digest and path.stat().st_size<=limit,'K03 FIT/SELECT input changed or exceeds bound')
        pins[str(path)]=digest;return path
    path=pin(request_path,request_sha256,2**20);r=core.load_json(path)
    legacy={'schema','evidence_scope','scientific_spec_sha256','inputs','implementation_sha256','alpha','parameters','resources'}
    finite=legacy|{'penalty_semantics'}
    require(set(r) in (legacy,finite)
        and r['schema']in{'r02_k03_fit_select_request_v1','r02_k03_fit_select_request_v2','r02_k03_fit_select_request_v3'}
        and ((r['schema']=='r02_k03_fit_select_request_v3')==(set(r)==finite))
        and r['implementation_sha256']==implementation(),
        'Exact K03 FIT/SELECT request and implementation required')
    require(r['evidence_scope']in{'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'},
        'Explicit K03 evidence scope required')
    require(len(r['scientific_spec_sha256'])==64 and all(c in '0123456789abcdef'for c in r['scientific_spec_sha256']),
        'Frozen scientific specification hash required')
    a=core.load_json(pin(admission_path,admission_sha256,2**20))
    require(set(a)=={'schema','status','evidence_scope','scientific_spec_sha256','request_sha256','evaluate_allowed'}and
        a['schema']=='r02_k03_fit_select_admission_v1'and a['status']=='FIT_SELECT_ONLY_ADMITTED'and
        a['evidence_scope']==r['evidence_scope']and a['scientific_spec_sha256']==r['scientific_spec_sha256']and
        a['request_sha256']==request_sha256 and a['evaluate_allowed']is False,
        'Exact external FIT/SELECT admission required; EVALUATE remains closed')
    limits=r['resources']
    require(set(limits)=={'max_metadata_bytes','max_array_bytes','max_input_bytes','max_output_bytes','min_free_bytes','task_cpus'}and
        all(type(v)is int and v>0 for v in limits.values()),'Explicit positive K03 task byte limits required')
    d=r['inputs']
    if r['schema']=='r02_k03_fit_select_request_v2':
        persons,points,records,m,binding=load_factored_inputs(d,pin,r,limits)
    else:
        persons,points,records,m=load_legacy_inputs(d,pin,r,limits)
        binding={}
    out=Path(output).resolve();require(not out.exists()and out.parent.is_dir()and
        shutil.disk_usage(out.parent).free>=limits['min_free_bytes']+limits['max_output_bytes'],'New private output and disk reserve required')
    out.mkdir(mode=0o700)
    result=fit_select_arrays(persons,points,records['FIT'],records['SELECT'],alpha=r['alpha'],parameters=r['parameters'],task_cpus=limits['task_cpus'],progress_path=out/'progress.json',penalty_semantics=r.get('penalty_semantics'))
    model=result['model'];np.savez(out/'model.private.npz',coefficient=model.coefficient,center=model.center,
        scale=model.scale,response_mean=np.asarray(model.response_mean))
    np.savez(out/'select_predictions.private.npz',i=records['SELECT']['i'],j=records['SELECT']['j'],prediction=result['predictions'])
    metrics=dict(result['metrics'],arm=m['arm'],candidate_id=m['candidate_id'],block_id=m['block_id'],exclusion_scope=m['exclusion_scope'],**binding)
    core.write_json(out/'metrics.json',metrics)
    require(all(core.sha(p)==h for p,h in pins.items())and implementation()==r['implementation_sha256'],'K03 source changed during task')
    require(sum(p.stat().st_size for p in out.iterdir())<=limits['max_output_bytes']and
        shutil.disk_usage(out).free>=limits['min_free_bytes'],'K03 output or disk bound exceeded')
    receipt=dict(schema=('r02_k03_fit_select_result_v2' if r['schema']=='r02_k03_fit_select_request_v3' else 'r02_k03_fit_select_result_v1'),stage='ONE_MODEL_COMPLETE_SELECTION_PENDING',
        request_sha256=request_sha256,admission_sha256=admission_sha256,scientific_spec_sha256=r['scientific_spec_sha256'],
        implementation_sha256=r['implementation_sha256'],inputs_manifest_sha256=d['base']['manifest_sha256']if binding else d['manifest_sha256'],
        outputs_sha256={p.name:core.sha(p)for p in out.iterdir()},**binding,
        evaluate_targets_computed=False,alpha_selection_performed=False,biological_validation=False,
        **({'penalty_semantics':r['penalty_semantics']} if 'penalty_semantics' in r else {}))
    core.write_json(out/'manifest.json',receipt);return receipt


def load_legacy_inputs(d,pin,r,limits):
    import zipfile
    require(set(d)=={'directory','manifest_sha256'},'Exact K03 prepared input descriptor required')
    root=Path(d['directory']).resolve();m=core.load_json(pin(root/'manifest.json',d['manifest_sha256'],limits['max_metadata_bytes']))
    source_spec=(r['penalty_semantics']['source_scientific_spec_sha256']
                 if r['schema']=='r02_k03_fit_select_request_v3' else r['scientific_spec_sha256'])
    require((r['schema']!='r02_k03_fit_select_request_v3' or
             r['penalty_semantics']['source_manifest_sha256']==d['manifest_sha256']) and
        m['schema']=='r02_k03_fit_select_arrays_v1'and m['stage']=='FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS'and
        m['evidence_scope']==r['evidence_scope']and m['scientific_spec_sha256']==source_spec and
        m['evaluate_targets_computed']is False and m['alpha_selection_performed']is False and
        m['arm']in{'CONTINUOUS','COMMUNITY_A','COMMUNITY_B'}and
        set(m['outputs_sha256'])=={'persons.tsv','arrays.private.npz'},'Prepared K03 feature/target scope differs')
    people_path=pin(root/'persons.tsv',m['outputs_sha256']['persons.tsv'],limits['max_metadata_bytes'])
    arrays_path=pin(root/'arrays.private.npz',m['outputs_sha256']['arrays.private.npz'],limits['max_input_bytes'])
    persons=core.table(people_path,['sample_id','component','role']);roles=core.validate_roles(persons)
    if r['evidence_scope']=='DNABR_R02':
        require(len(persons)==2619 and {k:len(v)for k,v in roles.items()}=={'FIT':1460,'SELECT':476,'EVALUATE':683},
            'Original DNABR roster must be preserved')
    fields={'points'}|{role+'_'+name for role in ('FIT','SELECT')for name in ('i','j','y','n','w','extra')}
    with zipfile.ZipFile(arrays_path)as z:
        require({p.filename for p in z.infolist()}=={k+'.npy'for k in fields}and
            len(z.infolist())==len(fields)and sum(p.file_size for p in z.infolist())<=limits['max_array_bytes'],
            'Exact bounded FIT/SELECT array inventory required; EVALUATE arrays forbidden')
    with np.load(arrays_path,allow_pickle=False)as z:
        points=z['points'];records={role:{name:z[role+'_'+name]for name in ('i','j','y','n','w','extra')}for role in ('FIT','SELECT')}
    return persons,points,records,m


def load_factored_inputs(d,pin,r,limits):
    import zipfile
    require(set(d)=={'base','candidate','arm'}and d['arm']in{'CONTINUOUS','COMMUNITY_A','COMMUNITY_B'},'Exact factored K03 descriptor required')
    manifests={};roots={}
    for kind in ('base','candidate'):
        desc=d[kind];require(set(desc)=={'directory','manifest_sha256'},'Exact factored artifact pin required')
        root=Path(desc['directory']);roots[kind]=root
        manifests[kind]=core.load_json(pin(root/'manifest.json',desc['manifest_sha256'],limits['max_metadata_bytes']))
    bm,cm=manifests['base'],manifests['candidate']
    require(bm['schema']=='r02_k03_shared_base_v1'and bm['stage']=='FROZEN_FIT_SELECT_BASE_NO_EVALUATE_TARGETS'
        and bm['evaluate_targets_computed']is False and bm['alpha_selection_performed']is False
        and cm['schema']=='r02_k03_projected_membership_v1'and cm['stage']=='FROZEN_PROJECTED_MEMBERSHIP_NO_SELECTION'
        and cm['base_manifest_sha256']==d['base']['manifest_sha256']and cm['target_outcomes_used_for_membership']is False
        and cm['alpha_selection_performed']is False,'Factored K03 preparation scope differs')
    for key in ('evidence_scope','scientific_spec_sha256'):
        require(bm[key]==cm[key]==r[key],'Factored K03 scientific identity differs')
    require(all(bm[k]==cm[k]for k in ('block_id','exclusion_scope')),'Candidate target geometry differs')
    require(set(bm['outputs_sha256'])=={'persons.tsv','arrays.private.npz','support.json'}
        and set(cm['outputs_sha256'])=={'membership.private.npz'},'Factored artifact inventory differs')
    paths={};total=0
    for kind,manifest in manifests.items():
        for name,h in manifest['outputs_sha256'].items():
            p=pin(roots[kind]/name,h,limits['max_input_bytes']if name.endswith('.npz')else limits['max_metadata_bytes'])
            total+=p.stat().st_size;paths[name]=p
    require(total<=limits['max_input_bytes'],'Factored aggregate input byte bound exceeded')
    fields={'base_points'}|{role+'_'+name for role in ('FIT','SELECT')for name in ('i','j','y','n','w','extra')}
    size=0
    for name,names in [('arrays.private.npz',fields),('membership.private.npz',{'codes'})]:
        with zipfile.ZipFile(paths[name])as z:
            require({v.filename for v in z.infolist()}=={k+'.npy'for k in names}and len(z.infolist())==len(names),'Factored array inventory differs')
            size+=sum(v.file_size for v in z.infolist())
    require(size<=limits['max_array_bytes'],'Factored decompression bound exceeded')
    persons=core.table(paths['persons.tsv'],['sample_id','component','role']);roles=core.validate_roles(persons)
    if r['evidence_scope']=='DNABR_R02':require(len(persons)==2619 and {k:len(v)for k,v in roles.items()}=={'FIT':1460,'SELECT':476,'EVALUATE':683},'Original DNABR roster required')
    with np.load(paths['arrays.private.npz'],allow_pickle=False)as z:
        base=z['base_points'];records={role:{name:z[role+'_'+name]for name in ('i','j','y','n','w','extra')}for role in ('FIT','SELECT')}
    with np.load(paths['membership.private.npz'],allow_pickle=False)as z:codes=z['codes']
    require(len(cm['categories'])==cm['n_categories']and len(set(cm['categories']))==cm['n_categories']
        and design.baseline_identity(d['base']['manifest_sha256'],codes)==cm['baseline_identity'],'Categorical/baseline identity differs')
    points=None
    for role,record in records.items():
        result=design.assemble(base,codes,cm['n_categories'],record['i'],record['j'],record['extra'],arm=d['arm'],max_output_bytes=limits['max_array_bytes'])
        if points is None:points=result['points']
        else:require(np.array_equal(points,result['points']),'Role-specific individual features differ')
        record['extra']=result['extra']
    binding={k:cm[k]for k in ('baseline_identity','control','parent_candidate_id','sham_seed')}
    binding.update(base_manifest_sha256=d['base']['manifest_sha256'],candidate_manifest_sha256=d['candidate']['manifest_sha256'],
        target_identity=bm['target_identity'],mask_control_identity=bm['mask_control_identity'])
    binding['select_target_identity']=bm['select_target_identity']
    return persons,points,records,dict(bm,arm=d['arm'],candidate_id=cm['candidate_id']),binding


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(run_request(args.request,args.request_sha256,args.admission,args.admission_sha256,args.output)))
