#!/usr/bin/env python3
"""One registered SELECT-only alpha refinement, reusing frozen FIT targets.

Does not generate an EVALUATE acceptance. Initial SVDs were not persisted, so
they are refitted from the unchanged FIT data. Original trials are preserved.
"""
from functools import partial
import argparse
import json
import math
from pathlib import Path
import shutil
import numpy as np

import r02_structure_campaign as campaign
core=campaign.core
SCHEMA='r02_structure_select_refinement_v1'


def implementation():
    return {**campaign.implementation(),'r02_structure_refine.py':core.sha(__file__)}


def additional_alphas(trials,alphas):
    """PLAN 12.6: two adjacent geometric midpoints or registered edge expansion."""
    grid=sorted(alphas)
    core.require(len(grid)>=2 and len(set(grid))==len(grid)
                 and all(math.isfinite(v)and v>0 for v in grid),'Invalid initial SELECT alpha domain')
    chosen=core._choose_ridge(trials,grid);extra=set()
    for arm,alpha in chosen.items():
        at=grid.index(alpha)
        if at==0:extra.add(alpha/100)
        elif at==len(grid)-1:extra.update([alpha*10,alpha*100])
        else:extra.update([math.sqrt(alpha)*math.sqrt(grid[at-1]),math.sqrt(alpha)*math.sqrt(grid[at+1])])
    extra-=set(grid)
    core.require(0<len(extra)<=6 and all(math.isfinite(v)and v>0 for v in extra),'Refinement exceeds six-candidate budget')
    return sorted(extra)


def initial_selection(request,digest,selection,selection_sha,*,verify_cache):
    r,store=campaign.load(request,digest);selection=Path(selection).resolve()
    core.require(core.sha(selection/'manifest.json')==selection_sha,'Initial SELECT manifest changed')
    m=core.load_json(selection/'manifest.json')
    core.require(m.get('campaign_request_sha256')==digest and m.get('campaign_implementation_sha256')==campaign.implementation()
        and m.get('source_contract_sha256')==r['bundle']['sha256'] and m.get('evaluation_contract_sha256')==r['evaluation']['sha256']
        and m.get('evaluate_targets_computed')is False and m.get('evaluate_predictions_computed')is False
        and m.get('block_ids')==[b['block_id']for b in store.blocks],'Initial SELECT source/coverage/reserved role differs')
    core.require(core.load_json(selection/'complete.json')==dict(status=core.SELECT_STATUS,manifest_sha256=selection_sha),
                 'Initial SELECT completion marker differs')
    expected={'frozen_selection.json','support.json'}|{f'fit_targets/fit_{k:04d}.npz'for k in range(len(store.blocks))}
    core.require(set(m['outputs_sha256'])==expected,'Initial SELECT artifact inventory differs')
    for name,sha in m['outputs_sha256'].items():
        if verify_cache or not name.startswith('fit_targets/'):
            core.require(core.sha(selection/name)==sha,'Initial SELECT artifact changed')
    selected=core.load_json(selection/'frozen_selection.json');trials=selected['trials']
    grid={(b['block_id'],a,alpha)for b in store.blocks for a in core.ARMS for alpha in store.evaluation['ridge_alphas']}
    core.require(len(trials)==len(grid) and {(v['block_id'],v['arm'],v['alpha'])for v in trials}==grid
        and all(math.isfinite(v['loss'])and v['loss']>=0 for v in trials)
        and selected['chosen']==core._choose_ridge(trials,store.evaluation['ridge_alphas']),
        'Initial SELECT trial grid/choice differs')
    core.require(not any('refinement_plan_sha256' in v for v in trials),'Only one refinement is registered')
    return r,store,m,selected


def make_plan(request,digest,selection,selection_sha,output):
    r,store,m,selected=initial_selection(request,digest,selection,selection_sha,verify_cache=True)
    additions=additional_alphas(selected['trials'],store.evaluation['ridge_alphas'])
    output=campaign.new_output(output,r).resolve()
    evaluation={**store.evaluation,'ridge_alphas':sorted(store.evaluation['ridge_alphas']+additions),
        'range_rationale':store.evaluation['range_rationale']+' One SELECT-only refinement by PLAN12.6; original SELECT manifest '+selection_sha+'; no EVALUATE acceptance.'}
    core.write_json(output/'evaluation.json',evaluation)
    new={**r,'evaluation':dict(path=str(output/'evaluation.json'),sha256=core.sha(output/'evaluation.json'))}
    core.write_json(output/'request.json',new)
    plan=dict(schema=SCHEMA,iteration=1,implementation_sha256=implementation(),
        initial_request=dict(path=str(Path(request).resolve()),sha256=digest),
        initial_selection=dict(path=str(Path(selection).resolve()),sha256=selection_sha),
        request=dict(path=str(output/'request.json'),sha256=core.sha(output/'request.json')),
        initial_alphas=store.evaluation['ridge_alphas'],additional_alphas=additions,
        rule='INTERIOR_GEOMETRIC_NEIGHBORS_LOWER_DIV100_UPPER_TIMES10_AND100_MAX6_SHARED',
        evaluate_allowed=False,refinement_count_limit=1)
    core.write_json(output/'plan.json',plan)
    return plan


def load_plan(path,digest):
    core.require(core.sha(path)==digest,'Refinement plan changed')
    p=core.load_json(path)
    core.require(p['schema']==SCHEMA and p['iteration']==p['refinement_count_limit']==1
        and p['implementation_sha256']==implementation() and p['evaluate_allowed']is False,'Refinement identity/limit differs')
    old,store,m,selected=initial_selection(p['initial_request']['path'],p['initial_request']['sha256'],
        p['initial_selection']['path'],p['initial_selection']['sha256'],verify_cache=False)
    r,new_store=campaign.load(p['request']['path'],p['request']['sha256'])
    extra=additional_alphas(selected['trials'],store.evaluation['ridge_alphas'])
    core.require(p['initial_alphas']==store.evaluation['ridge_alphas'] and p['additional_alphas']==extra
        and r['bundle']==old['bundle'] and r['resources']==old['resources']
        and new_store.evaluation['ridge_alphas']==sorted(p['initial_alphas']+extra),
        'Refinement changed sources/resources or deviated from registered candidate rule')
    for key,value in store.evaluation.items():
        if key not in {'ridge_alphas','range_rationale'}:
            core.require(new_store.evaluation[key]==value,'Refinement changed a non-alpha evaluation decision')
    return p,r,new_store,m,selected


def refine_block(plan_path,plan_sha,index,output):
    p,r,store,m,selected=load_plan(plan_path,plan_sha)
    core.require(type(index)is int and 0<=index<len(store.blocks),'Unknown refinement target')
    name=store.blocks[index]['block_id'];selection=Path(p['initial_selection']['path'])
    cached=selection/'fit_targets'/f'fit_{index:04d}.npz'
    core.require(core.sha(cached)==m['outputs_sha256'][f'fit_targets/fit_{index:04d}.npz'],'Frozen FIT target cache changed')
    values,c=store.load(index);persons,roles,blocks,markers,gt,features,pairs=values
    components=[v['component']for v in persons];paired=pairs[name]
    with np.load(cached,allow_pickle=False)as fit:
        fi,fj,fy,fw=(fit[k]for k in ('i','j','y','w'))
    gt=core._block_genotypes(blocks[0],markers,gt)
    si,sj,sy,sn,info=campaign.numeric.target_pairs(gt,roles['FIT'],roles['SELECT'],components,
        c['min_fit_called'],c['min_pair_markers'],c['max_pairs'],max_workspace_bytes=r['resources']['target_workspace_bytes'])
    sw,_=core.component_pair_weights(si,sj,components)
    trials=[dict(row,initial_selection_manifest_sha256=p['initial_selection']['sha256'])
            for row in selected['trials']if row['block_id']==name]
    for arm in core.ARMS:
        x=features[f'{name}_{arm}'];xf=core.design_features(x,fi,fj,c['max_matrix_mb'],paired,arm,require_fit_support=True)
        xs=core.design_features(x,si,sj,c['max_matrix_mb'],paired,arm)
        model=campaign.numeric.WeightedRidgePath(xf,fy,fw,paired=paired[arm].shape[1]>0,
            max_workspace_bytes=r['resources']['ridge_workspace_bytes'])
        for alpha in p['additional_alphas']:
            trials.append(dict(block_id=name,arm=arm,alpha=alpha,loss=core.macro_loss(sy,model.predict(xs,alpha),sw),
                n_symmetric_features=xf.shape[1],refinement_plan_sha256=plan_sha,
                refinement_implementation_sha256=implementation(),initial_selection_manifest_sha256=p['initial_selection']['sha256']))
    output=campaign.new_output(output,r);(output/'fit_targets').mkdir()
    shutil.copyfile(cached,output/'fit_targets'/cached.name)
    support=[v for v in core.load_json(selection/'support.json')if v['block_id']==name]
    prior=next(v for v in support if v['role']=='SELECT')
    core.require(prior['n_pairs']==len(sy) and prior['eligible_markers']==info['eligible_markers']
        and prior['min_joint_markers']==int(sn.min()) and prior['max_joint_markers']==int(sn.max()),
        'SELECT support changed during refinement')
    core.write_json(output/'support.json',support);core.write_json(output/'trials.json',trials)
    campaign.check_output(output,r);campaign.verify_task_inputs(p['request']['path'],p['request']['sha256'],r,store,index)
    core.require(core.sha(plan_path)==plan_sha and core.sha(cached)==m['outputs_sha256'][f'fit_targets/{cached.name}'],
                 'Refinement source changed during computation')
    return campaign.seal(output,'SELECT_BLOCK',p['request']['sha256'],index=index,block_id=name,
        evaluate_targets_computed=False,refinement_plan_sha256=plan_sha,refinement_implementation_sha256=implementation(),
        fit_target_recomputed=False,fit_model_refitted=True)


def boundary_review(plan_path,plan_sha,selection,selection_sha):
    p,r,store,m,prior=load_plan(plan_path,plan_sha);selection=Path(selection)
    core.require(core.sha(selection/'manifest.json')==selection_sha,'Refined selection manifest changed')
    manifest=core.load_json(selection/'manifest.json')
    core.require(manifest['campaign_request_sha256']==p['request']['sha256']
        and core.sha(selection/'frozen_selection.json')==manifest['outputs_sha256']['frozen_selection.json'],
        'Refined candidate identity differs')
    result=core.load_json(selection/'frozen_selection.json');chosen=result['chosen']
    core.require(chosen==core._choose_ridge(result['trials'],store.evaluation['ridge_alphas']),
                 'Refined choice differs from recorded curves')
    upper=max(store.evaluation['ridge_alphas'])
    unresolved=[arm for arm in core.ARMS if chosen[arm]==upper and upper>max(p['initial_alphas'])]
    return dict(schema=SCHEMA,stage='SELECT_BOUNDARY_REVIEW_REQUIRED',selection_manifest_sha256=selection_sha,
        upper_boundary_preference=unresolved,stop_before_evaluate=bool(unresolved),
        acceptance_generated=False,additional_search_automatically_allowed=False,
        criterion='Any persistent preference at the expanded upper boundary stops before EVALUATE; all curves retained')


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--stage',choices=['plan','block','boundary'],required=True)
    q.add_argument('--request');q.add_argument('--request-sha256');q.add_argument('--selection');q.add_argument('--selection-sha256')
    q.add_argument('--plan');q.add_argument('--plan-sha256');q.add_argument('--index',type=int);q.add_argument('--output',type=Path)
    a=q.parse_args()
    if a.stage=='plan':r=make_plan(a.request,a.request_sha256,a.selection,a.selection_sha256,a.output)
    elif a.stage=='block':r=refine_block(a.plan,a.plan_sha256,a.index,a.output)
    else:r=boundary_review(a.plan,a.plan_sha256,a.selection,a.selection_sha256)
    print(json.dumps(r))


if __name__=='__main__':main()
