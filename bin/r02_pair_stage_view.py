#!/usr/bin/env python3
"""Stage one portable K02/K07 view from generation-pinned GCS objects."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

REUSED='chr01_part01__WHOLE_CHROMOSOME'
ACCOUNT='653458115080-compute@developer.gserviceaccount.com'
PROJECT='uspbr-242713'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())

def resolve(plan_path,objects_path,reused_path):
    plan=load(plan_path);inventory=load(objects_path);reused=Path(reused_path)
    require(len(plan.get('tasks',[]))==88 and plan.get('evaluate_allowed') is False,
            'Exact 88-view EVALUATE-closed plan required')
    published={}
    for item in inventory.get('objects',[]):
        if item['path'].endswith('/arrays.private.npz'):
            require(item['view_id'] not in published,'Duplicate published common array')
            published[item['view_id']]=item
    require(len(published)==87 and REUSED not in published,'Expected exactly 87 newly published common arrays')
    require(reused.is_file(),'Reused full-common array absent')
    sources={}
    for task in plan['tasks']:
        view=task['view_id']
        if view==REUSED:
            require(task['common_key']=='scores' and sha(reused)==task['common_npz_sha256'],
                    'Reused common source differs')
            sources[view]=dict(kind='PACKAGED_REUSED',path=str(reused),bytes=reused.stat().st_size,
                               sha256=task['common_npz_sha256'])
        else:
            obj=published.get(view);require(obj is not None,'Published common source absent: '+view)
            require(task['common_key']=='points' and obj['sha256']==task['common_npz_sha256'] and
                    type(obj['bytes']) is int and obj['bytes']>0 and str(obj['generation']).isdigit() and
                    obj['uri'].startswith('gs://'),'Published common identity differs: '+view)
            sources[view]=dict(kind='GCS_GENERATION_PINNED',uri=obj['uri'],generation=str(obj['generation']),
                               bytes=obj['bytes'],sha256=obj['sha256'])
    require(len(sources)==88,'Common source coverage differs')
    return plan,sources

def copy_gcs(obj,destination):
    subprocess.run(['gcloud','--account='+ACCOUNT,'--project='+PROJECT,'storage','cp',
                    obj['uri']+'#'+obj['generation'],str(destination)],check=True)

def stage(plan_path,objects_path,reused_path,view_id,out,evaluation_block,evaluation_whole):
    plan,sources=resolve(plan_path,objects_path,reused_path)
    matches=[t for t in plan['tasks'] if t['view_id']==view_id];require(len(matches)==1,'Exact view required')
    task=matches[0];out=Path(out);require(not out.exists(),'Output exists');(out/'prepared').mkdir(parents=True)
    for objects,base in ((task['prepared_root_objects'],out/'prepared'),
                         (task['prepared_shard_objects'],out/'prepared'/'shards'/task['block_id'])):
        base.mkdir(parents=True,exist_ok=True)
        for name,obj in objects.items():
            destination=base/name;copy_gcs(dict(uri=obj['uri'],generation=str(obj['generation'])),destination)
            require(destination.stat().st_size==obj['bytes'] and sha(destination)==obj['sha256'],
                    'Staged rare object differs: '+name)
    source=sources[view_id];common=out/'common.npz'
    if source['kind']=='PACKAGED_REUSED':shutil.copyfile(source['path'],common)
    else:copy_gcs(source,common)
    require(common.stat().st_size==source['bytes'] and sha(common)==source['sha256'],
            'Staged common array differs')
    evaluation=Path(evaluation_block if task['exclusion_scope']=='BLOCK_AND_LD_MARGIN' else evaluation_whole)
    require(evaluation.is_file(),'Packaged evaluation configuration absent');shutil.copyfile(evaluation,out/'evaluation.json')
    receipt=dict(schema='r02_pair_portable_stage_v1',view_id=view_id,block_id=task['block_id'],
        exclusion_scope=task['exclusion_scope'],common_source=source,common_sha256=sha(common),
        evaluation_sha256=sha(out/'evaluation.json'),rare_objects_verified=True)
    (out/'stage-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--common-objects',required=True);p.add_argument('--reused-common',required=True)
    p.add_argument('--view-id');p.add_argument('--out');p.add_argument('--evaluation-block')
    p.add_argument('--evaluation-whole');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    if a.verify_only:
        plan,sources=resolve(a.plan,a.common_objects,a.reused_common)
        print(json.dumps({'status':'COMPLETE_88_COMMON_SOURCES','views':len(sources),
            'gcs_generation_pinned':sum(x['kind']=='GCS_GENERATION_PINNED' for x in sources.values()),
            'packaged_reused':sum(x['kind']=='PACKAGED_REUSED' for x in sources.values()),
            'total_common_bytes':sum(x['bytes'] for x in sources.values())},sort_keys=True))
    else:
        require(all((a.view_id,a.out,a.evaluation_block,a.evaluation_whole)),'Stage arguments required')
        print(json.dumps(stage(a.plan,a.common_objects,a.reused_common,a.view_id,a.out,
                               a.evaluation_block,a.evaluation_whole),sort_keys=True))
