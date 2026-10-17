#!/usr/bin/env python3
"""Bind, admit and close finite common searches for the frozen 88 territories."""
import argparse, hashlib, json, os
from pathlib import Path

FIRST = "chr01_part01__WHOLE_CHROMOSOME"
UPSTREAM_COMPLETE_URI = "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-common-88-preparation-20261008ak/products/COMMON_88_PREPARATION_COMPLETE.json"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise ValueError("new output required: "+str(path))
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def require(v,m):
    if not v: raise ValueError(m)

def bridge(root,view,source_spec):
    directory=Path(root).resolve()/('common_'+view);m=directory/'manifest.json';b=directory/'bridge-manifest.json'
    require(m.is_file()and b.is_file(),'missing complete bridge: '+view)
    manifest=load(m);receipt=load(b)
    require(manifest.get('schema')=='r02_k03_fit_select_arrays_v1'and manifest.get('stage')=='FROZEN_FIT_SELECT_ARRAYS_NO_EVALUATE_TARGETS'
      and manifest.get('scientific_spec_sha256')==source_spec and manifest.get('block_id')+'__'+manifest.get('exclusion_scope')==view
      and manifest.get('evaluate_targets_computed')is False and manifest.get('model_fit_performed')is False
      and manifest.get('penalty_present')is False and set(manifest.get('outputs_sha256',{}))=={'persons.tsv','arrays.private.npz'},
      'bridge semantic identity differs: '+view)
    require(receipt.get('schema')=='r02_k03_common_target_bridge_v1'and receipt.get('input_manifest_sha256')==sha(m)
      and receipt.get('block_id')+'__'+receipt.get('exclusion_scope')==view and receipt.get('model_fit_performed')is False
      and receipt.get('penalty_present')is False and receipt.get('evaluate_targets_computed')is False,
      'bridge completion receipt differs: '+view)
    for name,digest in manifest['outputs_sha256'].items():require(sha(directory/name)==digest,'bridge file differs: '+view+'/'+name)
    return directory,sha(m)

def generate(catalogue_path,bridge_root,prototype_path,spec_path,output):
    catalogue=load(catalogue_path);prototype=load(prototype_path);spec=sha(spec_path)
    require(catalogue.get('view_count')==88 and catalogue.get('evaluate_allowed')is False,'exact 88 catalogue required')
    views=[v['view_id'] for v in catalogue['views']];require(len(views)==len(set(views))==88 and FIRST in views,'ordered 88 views differ')
    source_spec='61d3e3d8939f7b57f021af1802c0d7ff2f73b00e9955b7baf0b2332d2869ef93'
    out=Path(output).resolve();require(not out.exists()and out.parent.is_dir(),'new batch output required');out.mkdir()
    rows=[]
    for view in views:
        if view==FIRST:continue
        directory,manifest_sha=bridge(bridge_root,view,source_spec)
        request=dict(prototype);request['parameters']=dict(prototype['parameters']);request['parameters']['workers']=1
        request['scientific_spec_sha256']=spec;request['source_scientific_spec_sha256']=source_spec
        request['source']={'directory':str(directory),'manifest_sha256':manifest_sha}
        path=out/'requests'/(view+'.json');write(path,request)
        rows.append({'id':view,'request':str(path),'request_sha256':sha(path),'admission_status':'NOT_GRANTED'})
    require(len(rows)==87,'exact 87 new search requests required')
    batch={'schema':'r02_common_finite_search_territories_batch_v1','status':'REQUESTS_BOUND_ADMISSIONS_REQUIRED',
      'scientific_spec_sha256':spec,'source_scientific_spec_sha256':source_spec,'view_count':87,'ordered_views':rows,
      'derivation':{'catalogue_sha256':sha(catalogue_path),'prototype_sha256':sha(prototype_path),
        'generator_sha256':sha(Path(__file__).resolve()),'workers_transform':{'path':'parameters.workers','from':prototype['parameters']['workers'],
          'to':1,'kind':'TECHNICAL_SINGLE_THREAD_EQUIVALENT'}},
      'reused_first_view':FIRST,'evaluate_allowed':False}
    write(out/'batch.json',batch);write(out/'manifest.json',{'schema':'r02_common_finite_search_territories_request_receipt_v1',
      'batch_sha256':sha(out/'batch.json'),'view_count':87,'evaluate_allowed':False})
    return batch

def admit(batch_path,review_path,output):
    batch=load(batch_path);review=load(review_path);pin=sha(batch_path)
    require(review=={'schema':'r02_common_finite_search_territories_root_review_v1','status':'ROOT_APPROVED_FIT_SELECT_NO_EVALUATE',
      'request_batch_sha256':pin,'scientific_spec_sha256':batch['scientific_spec_sha256'],'ordered_request_sha256':[r['request_sha256']for r in batch['ordered_views']],
      'evaluate_allowed':False},'exact root review of all 87 request hashes required')
    out=Path(output).resolve();require(not out.exists()and out.parent.is_dir(),'new admissions output required');out.mkdir()
    rows=[]
    for row in batch['ordered_views']:
        a={'schema':'r02_k03_common_finite_search_admission_v1','status':'FIT_SELECT_FINITE_FAMILY_ADMITTED','evidence_scope':'DNABR_R02',
          'scientific_spec_sha256':batch['scientific_spec_sha256'],'request_sha256':row['request_sha256'],'evaluate_allowed':False}
        p=out/(row['id']+'.json');write(p,a);rows.append({**row,'admission':str(p),'admission_sha256':sha(p),'admission_status':'EXACT_REQUEST_ADMITTED'})
    write(out/'batch.json',{'schema':'r02_common_finite_search_territories_admitted_batch_v1','status':'EXACT_87_REQUESTS_ADMITTED',
      'request_batch_sha256':pin,'view_count':87,'views':rows,'evaluate_allowed':False})

def admit_prepare(batch_path,review_path,output):
    """Admit only calculation of FIT statistics and the frozen finite grid."""
    batch=load(batch_path);review=load(review_path);pin=sha(batch_path)
    ordered=[r['request_sha256'] for r in batch['ordered_views']]
    require(batch.get('schema')=='r02_common_finite_search_territories_batch_v1'
      and batch.get('status')=='REQUESTS_BOUND_ADMISSIONS_REQUIRED' and batch.get('view_count')==87
      and batch.get('evaluate_allowed')is False and len(ordered)==len(set(ordered))==87,
      'exact bound 87-request batch required')
    here=Path(__file__).resolve().parent;workflow=here.parent/'workflows/r02_common_search_prepare_only.nf'
    require(review=={'schema':'r02_common_prepare_only_parametric_root_review_v1',
      'status':'ROOT_APPROVED_DETERMINISTIC_FIT_STATISTICS_AND_GRID_ONLY',
      'scientific_spec_sha256':batch['scientific_spec_sha256'],'derivation':batch['derivation'],
      'upstream_completion_uri':UPSTREAM_COMPLETE_URI,
      'upstream_completion_status':'COMMON_88_PREPARATION_COMPLETE_NO_EVALUATE_NO_TRAINING',
      'view_count':87,'reused_complete_view':FIRST,'cpus_per_prepare':1,'memory_gb_per_prepare':2,
      'max_parallel_prepares':4,'prepare_module_sha256':sha(here/'r02_common_search_prepare_only.py'),
      'workflow_sha256':sha(workflow),'container_reference':'us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/dnabr-qc@sha256:3e9381d165b1e664e75093d885a62d8c180711c1c3f495324e8c08bdb5bd2af2',
      'container_image_id':'sha256:76a0fddd300d3634363b5a13a1d28983122644a42e7d6c18339f9c8ac08548a7',
      'fit_allowed':False,'select_allowed':False,'evaluate_allowed':False},
      'exact root review of deterministic prepare-only derivation/resources required')
    out=Path(output).resolve();require(not out.exists()and out.parent.is_dir(),'new prepare admissions output required');out.mkdir()
    rows=[]
    for row in batch['ordered_views']:
        a={'schema':'r02_k03_common_prepare_only_admission_v1','status':'FIT_STATISTICS_AND_GRID_ONLY_ADMITTED',
          'evidence_scope':'DNABR_R02','scientific_spec_sha256':batch['scientific_spec_sha256'],
          'request_sha256':row['request_sha256'],'fit_allowed':False,'select_allowed':False,'evaluate_allowed':False}
        p=out/(row['id']+'.json');write(p,a)
        rows.append({**row,'admission':str(p),'admission_sha256':sha(p),'admission_status':'PREPARE_ONLY_EXACT_REQUEST_ADMITTED'})
    implementations=[load(r['request'])['implementation_sha256'] for r in batch['ordered_views']]
    require(implementations and all(x==implementations[0] for x in implementations),
      'all prepare requests must bind the same frozen implementation')
    code_root=Path(__file__).resolve().parent
    write(out/'batch.json',{'schema':'r02_common_prepare_only_admitted_batch_v1','status':'EXACT_87_PREPARES_ADMITTED',
      'request_batch_sha256':pin,'scientific_spec_sha256':batch['scientific_spec_sha256'],'view_count':87,'views':rows,
      'implementation_sha256':implementations[0],
      'prepare_module_sha256':sha(code_root/'r02_common_search_prepare_only.py'),'cpus_per_prepare':1,
      'memory_gb_per_prepare':2,'max_parallel_prepares':4,'fit_allowed':False,'select_allowed':False,'evaluate_allowed':False})

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
 g=s.add_parser('generate');[g.add_argument('--'+n,required=True)for n in ('catalogue','bridge-root','prototype','spec','output')]
 a=s.add_parser('admit');[a.add_argument('--'+n,required=True)for n in ('batch','review','output')]
 q=s.add_parser('admit-prepare');[q.add_argument('--'+n,required=True)for n in ('batch','review','output')]
 x=p.parse_args()
 if x.cmd=='generate':generate(x.catalogue,x.bridge_root,x.prototype,x.spec,x.output)
 elif x.cmd=='admit':admit(x.batch,x.review,x.output)
 else:admit_prepare(x.batch,x.review,x.output)
if __name__=='__main__':main()
