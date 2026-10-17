#!/usr/bin/env python3
"""Turn an explicit root review of 88 exact requests into an executable batch."""
import argparse,hashlib,json
from pathlib import Path

def require(v,m):
    if not v: raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);require(not p.exists(),'Refuse overwrite: '+str(p));p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');return sha(p)
def admitted_requests(decision,batch,request_batch_sha):
    expected=[{'id':r['id'],'request_sha256':r['request_sha256']} for r in batch['views']]
    if decision.get('schema')=='r02_k03_common_target_bridge_root_review_v1':
        require(decision=={'schema':'r02_k03_common_target_bridge_root_review_v1',
            'status':'ADMITTED_EXACT_PENALTY_FREE_TARGET_REQUESTS',
            'request_batch_sha256':request_batch_sha,'requests':expected,
            'evaluate_allowed':False},'Root review must enumerate the exact ordered request hashes')
        return expected
    require(decision.get('schema')=='r02_common_88_root_campaign_review_v1'
        and decision.get('status')=='ADMITTED_EXACT_COMMON_88_PREPARATION_PLAN',
        'Exact admitted 88-view campaign review required')
    require(decision.get('request_batch_sha256')==request_batch_sha
        and decision.get('scientific_spec_sha256')==batch['scientific_spec_sha256']
        and decision.get('producer_implementation_sha256')==batch['implementation_sha256'],
        'Campaign review differs from request batch or producer')
    require(decision.get('source_pins')=={
        'assembly_sources_sha256':'d322b0e06118729450d6aaedf60b060b10d4bdf1dd9d874ddfce30184a0865bc',
        'common_archive':{'generation':'1791324684503839','sha256':'a8f95e533df8620f324976090e0298e1a1c22081020523f4429a017cf9ab4480'},
        'prepared_scope_contracts_sha256':'935c278fd228fe5d96e42d89ba8445437ffb924fe65a5c4089773a0a88612a34'},
        'Campaign review source pins differ')
    require(decision.get('resources')=={'machine_type':'e2-standard-8','memory_gib':32,
        'disk_gib':150,'minimum_free_bytes_before_stage':100000000000,
        'bridge_memory_gb':30,'bridge_max_forks':1,'adapter_memory_gb':4,
        'adapter_max_forks':4,'native_stop_seconds':43200,
        'operator_timeout_seconds':39600,'cost_ceiling_usd':10.0},
        'Campaign review resources differ')
    require(all(decision.get(k)is False for k in ('evaluate_allowed','model_fit_allowed',
        'alpha_selection_allowed','training_allowed')),'Campaign review opened a forbidden operation')
    rows=decision.get('ordered_views',[]);build=[{'id':r['id'],'request_sha256':r['request_sha256']}
        for r in rows if r.get('action')=='BUILD_PENALTY_FREE_BRIDGE_AND_ADAPT_FIT_SELECT']
    reused=[r for r in rows if r.get('action')=='REUSE_BYTE_VERIFIED_NEURAL_TARGET']
    require(len(rows)==88 and build==expected and reused==[{'id':'chr01_part01__WHOLE_CHROMOSOME',
        'action':'REUSE_BYTE_VERIFIED_NEURAL_TARGET',
        'target_manifest_sha256':'2fdb14ca7050e458b23862cc4cc91716e6d73c69fd4a73b04fbb487c7e1eae6d',
        'archive':{'bytes':11882608,'generation':'1791422735032792',
          'md5_base64':'dliN0PFZfNa1175G866O1A==',
          'sha256':'cb45ac0579f242b01d3e3caba20fe162f6670c16a9f1c9296020b36e38e78474',
          'uri':'gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/presentacion/biologico/R02_20260930/r02-neural-human-targets-20261008ac/useful-products.tar.gz'}}],
        'Campaign review must bind 87 builds plus the one exact reused target')
    return expected
def build(request_batch,request_batch_sha,review,review_sha,output):
    require(sha(request_batch)==request_batch_sha and sha(review)==review_sha,'Pinned review inputs differ')
    batch=load(request_batch);decision=load(review)
    require(batch.get('schema')=='r02_k03_common_target_bridge_request_batch_v1' and
            batch.get('status')=='REQUESTS_COMPLETE_ADMISSIONS_REQUIRED' and batch.get('view_count') in (87,88),
            'Exact 87-new or 88 request batch required')
    admitted_requests(decision,batch,request_batch_sha)
    out=Path(output).resolve();require(out.parent.is_dir(),'Admission output parent required')
    resume=out.exists();require(not resume or out.is_dir(),'Admission output must be a directory')
    if not resume:out.mkdir()
    rows=[]
    for row in batch['views']:
        admission={'schema':'r02_k03_common_target_bridge_admission_v1',
          'status':'COMMON_FIT_SELECT_TARGET_INPUT_ADMITTED','evidence_scope':'DNABR_R02',
          'scientific_spec_sha256':batch['scientific_spec_sha256'],
          'request_sha256':row['request_sha256'],'evaluate_allowed':False}
        path=out/(row['id']+'.admission.json')
        if resume:
            require(path.is_file()and load(path)==admission,'Existing exact admission differs');digest=sha(path)
        else:digest=write(path,admission)
        rows.append({'id':row['id'],'request':row['request'],'request_sha256':row['request_sha256'],
                     'admission':str(path),'admission_sha256':digest})
    executable={'schema':'r02_k03_common_target_bridge_batch_v1',
      'status':'EXACT_PENALTY_FREE_TARGET_REQUESTS_ADMITTED','view_count':batch['view_count'],
      'scientific_spec_sha256':batch['scientific_spec_sha256'],
      'implementation_sha256':batch['implementation_sha256'],'views':rows,
      'review_sha256':review_sha,'evaluate_allowed':False}
    batch_path=out/'batch.json'
    if resume:
        require(batch_path.is_file()and load(batch_path)==executable,'Existing executable batch differs');batch_digest=sha(batch_path)
    else:batch_digest=write(batch_path,executable)
    result={'schema':'r02_k03_common_target_bridge_admission_receipt_v1',
      'stage':'EXACT_ADMISSIONS_MATERIALIZED','batch_sha256':batch_digest,
      'view_count':batch['view_count'],'evaluate_allowed':False}
    manifest=out/'manifest.json'
    if resume:require(manifest.is_file()and load(manifest)==result,'Existing admission receipt differs')
    else:write(manifest,result)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('request-batch','request-batch-sha256','review','review-sha256','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(build(a.request_batch,a.request_batch_sha256,a.review,a.review_sha256,a.output),sort_keys=True))
