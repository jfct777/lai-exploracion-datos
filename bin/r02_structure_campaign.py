#!/usr/bin/env python3
"""Checkpointed R02 scalar comparison, one authenticated target per task.

Uses the existing B0/B1/B2 specification and review gate. Completed blocks are
Nextflow cache units; no checkpoint resumes a partially computed block.
"""
from __future__ import annotations

import argparse
import csv
from functools import partial
import json
from pathlib import Path
import shutil
import time

import numpy as np

import r02_structure_increment as core
import r02_structure_numerics as numeric

SCHEMA='r02_structure_campaign_v1'
CODE=('r02_structure_campaign.py','r02_structure_numerics.py','r02_structure_increment.py')


def implementation():
    return {n:core.sha(Path(__file__).with_name(n)) for n in CODE}


def load(request, digest):
    request=Path(request).resolve()
    core.require(core.sha(request)==digest,'Campaign request SHA256 differs')
    r=core.load_json(request)
    core.require(set(r)=={'schema','bundle','evaluation','implementation_sha256','resources'}
        and r['schema']==SCHEMA and r['implementation_sha256']==implementation(),
        'Exact campaign request/current implementation required')
    for key in ('bundle','evaluation'):
        d=r[key]
        core.require(set(d)=={'path','sha256'} and Path(d['path']).is_absolute(),'Absolute pinned campaign input required')
        p=Path(d['path'])/ 'contract.json' if key=='bundle' else Path(d['path'])
        core.require(core.sha(p)==d['sha256'],'Campaign input identity differs: '+key)
    resources=r['resources']
    core.require(set(resources)=={'target_workspace_bytes','ridge_workspace_bytes','bootstrap_batch_bytes',
        'max_checkpoint_bytes','min_free_bytes'} and all(type(v)is int and v>0 for v in resources.values()),
        'Exact positive operational resource bounds required')
    bundle=Path(r['bundle']['path']);contract=core.load_json(bundle/'contract.json')
    evaluation=core.load_json(r['evaluation']['path'])
    core.require(evaluation['prepared_contract_sha256']==r['bundle']['sha256'],'Evaluation belongs to another catalogue')
    store=core.BlockBundles(bundle,contract,evaluation)
    return r,store


def outputs(root):
    return {p.relative_to(root).as_posix():core.sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name not in ('checkpoint.json',)}


def seal(root,kind,request_sha,**metadata):
    value=dict(schema=SCHEMA,kind=kind,request_sha256=request_sha,
               implementation_sha256=implementation(),outputs_sha256=outputs(root),**metadata)
    core.write_json(root/'checkpoint.json',value)
    return value


def authenticate_checkpoint(root,kind,request_sha):
    root=Path(root).resolve();m=core.load_json(root/'checkpoint.json')
    core.require(m.get('schema')==SCHEMA and m.get('kind')==kind and m.get('request_sha256')==request_sha
        and m.get('implementation_sha256')==implementation(),'Checkpoint execution/source identity differs')
    core.require(not any(p.is_symlink() for p in root.rglob('*')),'Symlink inside immutable checkpoint')
    core.require(outputs(root)==m['outputs_sha256'],'Checkpoint inventory/content changed')
    return m


def new_output(output,r):
    output=Path(output)
    core.require(not output.exists(),'Output exists; resume completed tasks through Nextflow')
    core.require(shutil.disk_usage(output.parent).free >= r['resources']['min_free_bytes'],
                 'Insufficient free space for campaign reserve')
    output.mkdir(mode=0o700)
    return output


def check_output(output,r):
    core.require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<=
                 r['resources']['max_checkpoint_bytes'],'Checkpoint byte budget exceeded')
    core.require(shutil.disk_usage(output).free >= r['resources']['min_free_bytes'],
                 'Campaign disk reserve exhausted')


def verify_task_inputs(request,digest,r,store,index=None):
    core.require(core.sha(request)==digest and r['implementation_sha256']==implementation(),
                 'Campaign source/configuration changed during task')
    store.verify()
    if index is not None:
        path=store.path(store.entries[index])
        for name,value in core.load_json(path/'manifest.json')['outputs_sha256'].items():
            core.require(core.sha(path/name)==value,'Consumed shard changed during task')


def select_block(request,digest,index,output):
    started=time.monotonic();r,store=load(request,digest)
    core.require(type(index)is int and 0<=index<len(store.blocks),'Unknown campaign target index')
    output=new_output(output,r);work=output/'fit_targets';work.mkdir()
    support,trials=[],[]
    core._stream_select(store,index,work,support,trials,
        target_function=partial(numeric.target_pairs,max_workspace_bytes=r['resources']['target_workspace_bytes']),
        path_factory=partial(numeric.WeightedRidgePath,max_workspace_bytes=r['resources']['ridge_workspace_bytes']))
    core.write_json(output/'support.json',support);core.write_json(output/'trials.json',trials)
    check_output(output,r);verify_task_inputs(request,digest,r,store,index)
    return seal(output,'SELECT_BLOCK',digest,index=index,block_id=store.blocks[index]['block_id'],
                seconds=time.monotonic()-started,evaluate_targets_computed=False)


def merge_select(request,digest,checkpoints,output):
    r,store=load(request,digest);accepted={}
    for path in checkpoints:
        m=authenticate_checkpoint(path,'SELECT_BLOCK',digest)
        k=m['index'];core.require(k not in accepted and 0<=k<len(store.blocks)
            and m['block_id']==store.blocks[k]['block_id'] and m['evaluate_targets_computed']is False,
            'Duplicate, unexpected or evaluated SELECT block')
        accepted[k]=Path(path)
    core.require(set(accepted)==set(range(len(store.blocks))),'Incomplete SELECT target coverage')
    output=new_output(output,r);(output/'fit_targets').mkdir()
    trials,support=[],[]
    for k,path in sorted(accepted.items()):
        rows=core.load_json(path/'trials.json')
        expected={(store.blocks[k]['block_id'],a,alpha) for a in core.ARMS for alpha in store.evaluation['ridge_alphas']}
        core.require(len(rows)==len(expected) and {(x['block_id'],x['arm'],x['alpha'])for x in rows}==expected,
                     'SELECT trial grid differs')
        trials.extend(rows);support.extend(core.load_json(path/'support.json'))
        shutil.copyfile(path/'fit_targets'/f'fit_{k:04d}.npz',output/'fit_targets'/f'fit_{k:04d}.npz')
    chosen=core._choose_ridge(trials,store.evaluation['ridge_alphas'])
    core.write_json(output/'frozen_selection.json',core._selection_record(chosen,trials,r['bundle']['sha256'],r['evaluation']['sha256']))
    core.write_json(output/'support.json',support)
    manifest=core._select_manifest(output,store,r['bundle']['sha256'],r['evaluation']['sha256'],core.sha(core.__file__),
        extra_metadata=dict(campaign_request_sha256=digest,campaign_implementation_sha256=implementation(),
                            checkpointed_blocks=len(accepted)))
    check_output(output,r);verify_task_inputs(request,digest,r,store)
    return manifest


def admit(request,digest,selection,selection_hash,review,review_hash,output):
    r,store=load(request,digest)
    m=core.load_json(Path(selection)/'manifest.json')
    core.require(m.get('campaign_request_sha256')==digest and
        m.get('campaign_implementation_sha256')==implementation(),'SELECT campaign identity differs')
    selected,support=core._admit_selection(store,selection,selection_hash,review,review_hash,
        r['bundle']['sha256'],r['evaluation']['sha256'],core.sha(core.__file__))
    output=new_output(output,r)
    core.write_json(output/'admission.json',dict(selection=str(Path(selection).resolve()),
        selection_manifest_sha256=selection_hash,review=str(Path(review).resolve()),review_sha256=review_hash,
        selected=selected,support=support,evaluation_targets_computed=False))
    verify_task_inputs(request,digest,r,store)
    return seal(output,'EVALUATE_ADMITTED',digest)


def verify_review(admission):
    a=core.load_json(Path(admission)/'admission.json')
    core.require(core.sha(Path(a['selection'])/'manifest.json')==a['selection_manifest_sha256']
        and core.sha(a['review'])==a['review_sha256'],'Review/SELECT changed since admission')
    return a


def evaluate_block(request,digest,index,admission,output):
    started=time.monotonic();r,store=load(request,digest)
    authenticate_checkpoint(admission,'EVALUATE_ADMITTED',digest)
    a=verify_review(admission);selection=Path(a['selection'])
    core.require(type(index)is int and 0<=index<len(store.blocks),'Unknown evaluation target')
    manifest=core.load_json(selection/'manifest.json');cache=f'fit_targets/fit_{index:04d}.npz'
    core.require(core.sha(selection/cache)==manifest['outputs_sha256'][cache],'FIT target cache changed')
    output=new_output(output,r);work=output/'groups';work.mkdir()
    support,fitted,metadata=[],[],{}
    matrix_mb=core.load_json(store.path(store.entries[index])/'contract.json')['max_matrix_mb']
    statistics=core.BlockErrors(work,store.persons,{**store.evaluation,'max_matrix_mb':matrix_mb})
    fields=['block_id','chromosome','sample_a','sample_b','component_a','component_b','target','n_joint_markers',*core.ARMS]
    with (output/'predictions.private.tsv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t');writer.writeheader()
        core._stream_predict(store,index,work,a['selected']['chosen'],support,fitted,writer,statistics,metadata,
            fit_root=selection/'fit_targets',
            target_function=partial(numeric.target_pairs,max_workspace_bytes=r['resources']['target_workspace_bytes']))
    np.savez(output/'statistics.private.npz',person_macro=np.asarray(statistics.person_macro),
        seen_people=statistics.seen_people,seen_pairs=statistics.seen_pairs)
    core.write_json(output/'support.json',support);core.write_json(output/'models.json',fitted)
    core.write_json(output/'metadata.json',metadata)
    check_output(output,r);verify_task_inputs(request,digest,r,store,index);verify_review(admission)
    return seal(output,'EVALUATE_BLOCK',digest,index=index,block_id=store.blocks[index]['block_id'],
        chromosome=store.blocks[index]['chromosome'],total=statistics.total,
        admission_sha256=core.sha(Path(admission)/'checkpoint.json'),seconds=time.monotonic()-started)


def merge_evaluate(request,digest,checkpoints,admission,output):
    r,store=load(request,digest);authenticate_checkpoint(admission,'EVALUATE_ADMITTED',digest)
    verify_review(admission)
    ahash=core.sha(Path(admission)/'checkpoint.json');accepted={}
    for path in checkpoints:
        m=authenticate_checkpoint(path,'EVALUATE_BLOCK',digest);k=m['index']
        core.require(k not in accepted and 0<=k<len(store.blocks) and
            m['block_id']==store.blocks[k]['block_id'] and m['admission_sha256']==ahash,
            'Duplicate or incompatible EVALUATE block')
        accepted[k]=(Path(path),m)
    core.require(set(accepted)==set(range(len(store.blocks))),'Incomplete EVALUATE target coverage')
    output=new_output(output,r)
    matrix_mb=core.load_json(store.path(store.entries[0])/'contract.json')['max_matrix_mb']
    statistics=core.BlockErrors(output,store.persons,{**store.evaluation,'max_matrix_mb':matrix_mb})
    support,models,metadata=[],[],{}
    header=None
    with (output/'predictions.private.tsv').open('xb') as destination:
        for k,(path,m) in sorted(accepted.items()):
            with (path/'predictions.private.tsv').open('rb')as handle:
                current=handle.readline()
                if header is None:header=current;destination.write(header)
                core.require(current==header,'Prediction columns differ across targets')
                shutil.copyfileobj(handle,destination)
            statistics.entries.append(dict(block_id=m['block_id'],chromosome=m['chromosome'],stem=path/'groups/group_0000'))
            statistics.total+=m['total']
            with np.load(path/'statistics.private.npz',allow_pickle=False)as z:
                statistics.person_macro.extend(z['person_macro']);statistics.seen_people|=z['seen_people'];statistics.seen_pairs|=z['seen_pairs']
            support.extend(core.load_json(path/'support.json'));models.extend(core.load_json(path/'models.json'))
            metadata.update(core.load_json(path/'metadata.json'))
    core.require(statistics.total<=store.evaluation['max_prediction_records'],'Global prediction budget exceeded')
    result=statistics.evaluate(audit_dir=output,draw_reducer=partial(numeric.crossed_draw_means,
        max_batch_bytes=r['resources']['bootstrap_batch_bytes']))
    # The 2026-10-06 prespecified synthetic calibration showed undercoverage for
    # chromosome effects and overcoverage for degenerate dyadic mechanisms.
    # Preserve every estimate/draw; suppress unsupported detectability labels.
    result['legacy_numerical_support_status']=result['status']
    result['status']='CONDITIONAL_POINT_ESTIMATE_WITH_UNCALIBRATED_RESAMPLING_DIAGNOSTIC'
    result['contrast_interpretation']={k:'NO_INFERENTIAL_DETECTABILITY_ADMISSION' for k in result['deltas']}
    result['percentile_ci95_inferentially_admitted']=False
    result['limitations'].append('Crossed percentile bands failed nominal coverage in prespecified synthetic chromosome/degenerate cases; bands are diagnostics only')
    for name,value in [('evaluation.private.json',result),('support.json',support),('models.json',models),('metadata.json',metadata)]:
        core.write_json(output/name,value)
    store.verify_shards();check_output(output,r);verify_task_inputs(request,digest,r,store);verify_review(admission)
    return seal(output,'COMPLETE_CONDITIONAL_COMPARISON',digest,n_blocks=len(accepted),
        prepared_contract_sha256=r['bundle']['sha256'],evaluation_contract_sha256=r['evaluation']['sha256'],
        admission_sha256=ahash,claim=store.evaluation['evaluation_claim'],
        interpretation_required_read='bootstrap_monte_carlo.private.json',
        biological_generalization_certified=False,K03_discrete_structure_evaluated=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--stage',choices=['select-block','merge-select','admit','evaluate-block','merge-evaluate'],required=True)
    p.add_argument('--index',type=int);p.add_argument('--checkpoints',type=Path,nargs='*')
    p.add_argument('--admission',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--selection-sha256')
    p.add_argument('--review',type=Path);p.add_argument('--review-sha256');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();args=(a.request,a.request_sha256)
    if a.stage=='select-block':r=select_block(*args,a.index,a.output)
    elif a.stage=='merge-select':r=merge_select(*args,a.checkpoints,a.output)
    elif a.stage=='admit':r=admit(*args,a.selection,a.selection_sha256,a.review,a.review_sha256,a.output)
    elif a.stage=='evaluate-block':r=evaluate_block(*args,a.index,a.admission,a.output)
    else:r=merge_evaluate(*args,a.checkpoints,a.admission,a.output)
    print(json.dumps(r))


if __name__=='__main__':main()
