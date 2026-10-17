#!/usr/bin/env python3
"""Reduce authenticated one-block SCORE checkpoints to one scope-wide SELECT choice."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math

ARMS=('B0','B1','B2')
SCORE_SCHEMA='r02_structure_pair_score_checkpoint_v1'
SCORE_STATUS='FIT_SELECT_SCORES_COMPLETE'

def require(value,message):
    if not value: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def load(path):
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,'Duplicate JSON key');out[key]=value
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda _:require(False,'Nonfinite JSON'))

def write(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')

def choose(trials,alphas):
    selected={}
    for arm in ARMS:
        options=[(sum(r['loss'] for r in trials if r['arm']==arm and r['alpha']==alpha)/
                  len({r['block_id'] for r in trials}),alpha) for alpha in alphas]
        selected[arm]=min(options,key=lambda x:(x[0],-x[1]))[1]
    return selected

def run(plan_path,scope,score_dirs,output):
    require(scope in {'BLOCK_AND_LD_MARGIN','WHOLE_CHROMOSOME'},'Unknown exclusion scope')
    plan=load(plan_path);expected=[t for t in plan['tasks'] if t['exclusion_scope']==scope]
    require(len(plan['tasks'])==88 and len(expected)==44 and plan.get('evaluate_allowed') is False,
            'Exact 88-view plan with 44 requested scope blocks and EVALUATE closed required')
    expected_ids={t['block_id'] for t in expected};require(len(expected_ids)==44,'Duplicate expected block')
    require(len(score_dirs)==44*len(ARMS),'Exactly 132 block-arm score checkpoints required')
    trials=[];sources={};common_identity=None;seen=set();alphas=None
    identity_fields=('code_sha256','versions','numerical_backend','operator_backend','preprocessing_backend',
        'numerical_parameters','solver','numerical_config_sha256','pair_operator_sha256',
        'pair_preconditioner_sha256','run_scope_amendment_sha256')
    for directory in sorted(map(Path,score_dirs)):
        manifest=load(directory/'manifest.json')
        require(manifest.get('schema')==SCORE_SCHEMA and manifest.get('status')==SCORE_STATUS and
                manifest.get('exclusion_scope')==scope and manifest.get('evaluate_targets_computed') is False and
                manifest.get('alpha_selected') is False,'Incompatible score checkpoint')
        require(load(directory/'complete.json')=={'status':SCORE_STATUS,
                'manifest_sha256':sha(directory/'manifest.json')},'Score completion marker differs')
        block,arm=manifest['block_id'],manifest.get('arm')
        key=(block,arm);require(block in expected_ids and arm in ARMS and key not in seen,
                'Unexpected/duplicate score block-arm')
        seen.add(key)
        for relative,digest in manifest['outputs_sha256'].items():
            path=directory/relative
            require(path.is_file() and sha(path)==digest,'Score output changed: '+relative)
        record=load(directory/'candidate_scores.json')
        require(record.get('schema')==SCORE_SCHEMA and record.get('status')==SCORE_STATUS and
                record.get('block_id')==block and record.get('arm')==arm and record.get('exclusion_scope')==scope and
                record.get('evaluate_targets_computed') is False and record.get('alpha_selected') is False,
                'Score record identity differs')
        if alphas is None:alphas=record['ridge_alphas']
        require(record['ridge_alphas']==alphas and len(alphas)==9 and len(set(alphas))==9 and
                all(type(a) in (int,float) and math.isfinite(a) and a>0 for a in alphas),
                'Finite nine-point Ridge family differs')
        block_trials=record['trials'];grid={(block,arm,alpha) for alpha in alphas}
        require(len(block_trials)==len(grid) and
                {(r['block_id'],r['arm'],r['alpha']) for r in block_trials}==grid and
                all(math.isfinite(r['loss']) and r['loss']>=0 for r in block_trials),
                'Score trial grid/loss differs')
        identity={k:manifest.get(k) for k in identity_fields}
        require(isinstance(identity['run_scope_amendment_sha256'],str) and
                len(identity['run_scope_amendment_sha256'])==64 and
                all(c in '0123456789abcdef' for c in identity['run_scope_amendment_sha256']),
                'Authenticated full-common run-scope amendment required')
        if common_identity is None:common_identity=identity
        require(identity==common_identity,'Numerical implementation/configuration differs across blocks')
        trials.extend(block_trials);sources[block+':'+arm]=dict(manifest_sha256=sha(directory/'manifest.json'),
            candidate_scores_sha256=sha(directory/'candidate_scores.json'),
            source_contract_sha256=manifest['source_contract_sha256'],
            evaluation_contract_sha256=manifest['evaluation_contract_sha256'])
    require(seen=={(block,arm) for block in expected_ids for arm in ARMS},'Scope block-arm coverage incomplete')
    output=Path(output);require(not output.exists(),'Output exists');output.mkdir(parents=True,mode=0o700)
    selected=dict(schema='r02_structure_distributed_ridge_selection_v1',source='SELECT_ONLY',
        exclusion_scope=scope,chosen=choose(trials,alphas),ridge_alphas=alphas,trials=trials,
        n_blocks=44,frozen_before_evaluation=True,evaluate_targets_computed=False,
        score_sources=sources)
    write(output/'frozen_selection.json',selected)
    manifest=dict(schema='r02_structure_distributed_select_checkpoint_v1',
        status='GLOBAL_SELECT_COMPLETE_AWAITING_REVIEW',exclusion_scope=scope,n_blocks=44,
        block_ids=sorted(expected_ids),score_checkpoint_count=len(seen),assembly_plan_sha256=sha(plan_path),numerical_identity=common_identity,
        frozen_selection_sha256=sha(output/'frozen_selection.json'),
        created_at_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
        evaluate_targets_computed=False,biological_result=False)
    write(output/'manifest.json',manifest)
    write(output/'complete.json',{'status':manifest['status'],'manifest_sha256':sha(output/'manifest.json')})
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--scope',required=True)
    p.add_argument('--score-dir',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.plan,a.scope,a.score_dir,a.output),sort_keys=True))
