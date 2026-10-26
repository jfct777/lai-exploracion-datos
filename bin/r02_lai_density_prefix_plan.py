#!/usr/bin/env python3
"""Bind verified radius-specific density products to EVENT axes for prefix preparation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def require(value, message):
    if not value: raise ValueError(message)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def desc(path):
    path=Path(path).resolve(); require(path.is_file(),'file absent: '+str(path))
    return {'path':str(path),'sha256':sha256(path)}

def build(density_plan_path,density_root,index_root,output):
    density_plan_path=Path(density_plan_path); plan=json.loads(density_plan_path.read_text())
    require(plan.get('schema')=='r02_lai_density_plan_v1' and len(plan.get('tasks',[]))==36,
            'density plan differs')
    tasks=[]
    for row in plan['tasks']:
        prepared=Path(density_root)/row['density_task_id']/'prepared'
        receipt_path=prepared/'receipt.json'; batch_path=prepared/'control-views.npz'
        receipt=json.loads(receipt_path.read_text())
        require(receipt.get('schema')=='r02_lai_five_control_receipt_v2'
                and receipt.get('radius_cm')==row['radius_cm']
                and receipt.get('batch_sha256')==sha256(batch_path),'density product differs')
        axis_dir=Path(index_root)/row['task_id']; axis_receipt=axis_dir/'receipt.json'; axis=axis_dir/'indices.npz'
        tasks.append({'density_task_id':row['density_task_id'],'task_id':row['task_id'],
                      'radius_cm':row['radius_cm'],'density_batch':desc(batch_path),
                      'density_receipt':desc(receipt_path),'event_axis':desc(axis),
                      'event_axis_receipt':desc(axis_receipt)})
    result={'schema':'r02_lai_density_prefix_plan_v1',
            'status':'THIRTY_SIX_DENSITY_PREFIX_REQUESTS','source_plan_sha256':sha256(density_plan_path),
            'tasks':tasks,'condition':'DENSITY_MATCHED','radius_embedded_in_prefix':False,
            'training_authorized':False,'selection_authorized':False,'evaluate_opened':False}
    out=Path(output);require(not out.exists() and out.parent.is_dir(),'fresh prefix plan required')
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--density-plan',required=True)
    p.add_argument('--density-root',required=True);p.add_argument('--index-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=build(a.density_plan,a.density_root,a.index_root,a.output);print(json.dumps({'tasks':len(r['tasks'])}))
if __name__=='__main__':main()
