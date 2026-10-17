#!/usr/bin/env python3
"""Replace scalar-common columns in one authenticated v4 shard by full common axes.

Rare-count/pattern suffixes and paired controls are copied byte-for-byte. This is
only data/config adaptation; fitting remains in r02_structure_increment.py.
"""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
ARMS=('B0','B1','B2')
VERIFIED_INPUTS=('persons.tsv','blocks.tsv','markers.tsv','targets.npz','features.npz','lineage.json',
 'pair_features.npz','pair_lineage.json')
def require(x,m):
 if not x: raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_adapted_features(path,sample_ids,block,arrays):
 with Path(path).open('wb') as f:np.savez(f,sample_ids=sample_ids,**{f'{block}_{a}':arrays[a] for a in ARMS})
def rebind_verification(dst,common,amendment_sha):
 dst,common=Path(dst),Path(common)
 sc=load(dst/'contract.json')
 source_verification_sha=sha(dst/'upstream_verification.json')
 verification=load(dst/'upstream_verification.json')
 require(verification.get('status')=='COMPLETE_FEATURE_EXCLUSION_AND_SPLIT_CHECK' and
  verification.get('checks') and all(verification['checks'].values()),
  'Source materialization verification is incomplete')
 adapter_sha=sha(__file__)
 verification['source_upstream_verification_sha256']=source_verification_sha
 verification['source_materializer_code_sha256']=verification['materializer_code_sha256']
 verification['materializer_code_sha256']=adapter_sha
 verification['common_source_sha256']=sha(common)
 verification['run_scope_amendment_sha256']=amendment_sha
 verification['input_sha256']={n:sha(dst/n) for n in VERIFIED_INPUTS}
 verification['verification_scope']=('Authenticated full-common replacement after verified source preparation; '
  'persons, roles, targets, rare suffixes, paired features, masks and exclusions preserved; no fit')
 write(dst/'upstream_verification.json',verification)
 sc['materializer_code_sha256']=adapter_sha
 sc['materializer_verification_sha256']=sha(dst/'upstream_verification.json')
 for n in list(sc['sha256']):sc['sha256'][n]=sha(dst/n)
 write(dst/'contract.json',sc)
 return sc,verification
def apply_amendment(evaluation_path,amendment_path,scope,axes):
 e=load(evaluation_path);a=load(amendment_path)
 required={'schema','status','reason','base_evaluation_sha256','full_common_axes',
  'symmetric_transform_families','reference_fit_pairs','dense_base_design_bytes','dense_base_design_gib',
  'replacement_capacity_comparison','preserved_scientific_parameters','dependence_limit','evaluate_allowed'}
 require(set(a)==required and a['schema']=='r02_k02_k07_fullcommon_run_scope_amendment_v1' and
  a['status']=='REVIEW_REQUIRED_WITH_PAIR_SELECT_ADMISSION' and a['evaluate_allowed'] is False,
  'Exact reviewed full-common run-scope amendment required')
 require(a['base_evaluation_sha256']=={
  'BLOCK_AND_LD_MARGIN':'619427a4c575a8e69d905ba2d8ac4f5ade81c2ba0350e47b6901860fe6e01392',
  'WHOLE_CHROMOSOME':'8ae794eb3e2282c826d5827959091ee6cd649f2432d479efde6fc0af0bf74172'},
  'Frozen evaluation identities differ from amendment')
 require(sha(evaluation_path)==a['base_evaluation_sha256'][scope] and a['full_common_axes']==axes and
  a['symmetric_transform_families']==3 and a['reference_fit_pairs']==1064710 and
  a['dense_base_design_bytes']==1064710*axes*3*8 and
  abs(a['dense_base_design_gib']-a['dense_base_design_bytes']/2**30)<1e-12 and
  isinstance(a['replacement_capacity_comparison'],str) and len(a['replacement_capacity_comparison'])>200,
  'Full-common capacity amendment calculation differs')
 e['capacity_comparison']=a['replacement_capacity_comparison']
 return e,sha(amendment_path),sha(evaluation_path)
def run(root,common,block,out,evaluation,common_key,amendment):
 root,common,out=map(Path,(root,common,out));require(not out.exists(),'Output exists')
 rc=load(root/'contract.json');require(rc.get('schema')=='r02_structure_increment_v4','v4 prepared root required')
 idx=load(root/'block_index.json'); matches=[x for x in idx['blocks'] if x['block_id']==block];require(len(matches)==1,'Exact block required')
 src=root/matches[0]['path']; sm=load(src/'manifest.json')
 require(sha(src/'manifest.json')==matches[0]['manifest_sha256'],'Source manifest changed')
 for n,d in sm['outputs_sha256'].items():require(sha(src/n)==d,'Source shard changed: '+n)
 with np.load(common,allow_pickle=False) as z:
  require(common_key in z.files,'Full-common axis array absent');points=np.array(z[common_key],copy=True)
 persons=(root/'persons.tsv').read_bytes(); common_people=common.parent/'persons.tsv'
 if common_people.exists():require(common_people.read_bytes()==persons,'Full-common people/order differs')
 require(points.ndim==2 and points.shape[0]==sum(1 for _ in persons.splitlines())-1 and np.isfinite(points).all(),'Invalid full-common axes')
 lineage=load(src/'lineage.json');L=lineage[block]
 with np.load(src/'features.npz',allow_pickle=False) as z:
  require(set(z.files)=={'sample_ids'}|{f'{block}_{a}' for a in ARMS},'Source feature tensor keys differ')
  sample_ids=np.array(z['sample_ids'],copy=True)
  old={a:np.array(z[f'{block}_{a}'],copy=True) for a in ARMS}
 for a in ARMS:require(old[a].shape[1]==len(L[a]),'Source lineage width differs')
 n0,n1=len(L['B0']),len(L['B1']);require(L['B1'][:n0]==L['B0'] and L['B2'][:n1]==L['B1'],'Source arms not nested')
 common_cols=[c for c in L['B0'] if c['kind']=='common'];measurement=[(i,c) for i,c in enumerate(L['B0']) if c['kind']=='measurement']
 require(common_cols and measurement and len(common_cols)+len(measurement)==n0,'B0 must contain only common and measurement')
 intervals=[]
 for c in common_cols:
  for x in c['source_intervals']:
   if x not in intervals:intervals.append(x)
 fit_hash=common_cols[0]['fit_sample_ids_sha256'];require(all(c['fit_sample_ids_sha256']==fit_hash for c in common_cols),'Common FIT lineage differs')
 axes=[{'id':f'full_common_axis_{i+1:04d}','kind':'common','source_intervals':intervals,'fit_sample_ids_sha256':fit_hash,'derivation':f'Frozen full numerical common axis {i+1} from authenticated common bridge {sha(common)}'} for i in range(points.shape[1])]
 meas=np.column_stack([old['B0'][:,i] for i,_ in measurement]);meas_cols=[c for _,c in measurement]
 b0=np.column_stack((points,meas));add1=old['B1'][:,n0:];add2=old['B2'][:,n1:]
 arrays={'B0':b0,'B1':np.column_stack((b0,add1)),'B2':np.column_stack((b0,add1,add2))}
 cols={'B0':axes+meas_cols,'B1':axes+meas_cols+L['B1'][n0:],'B2':axes+meas_cols+L['B1'][n0:]+L['B2'][n1:]}
 require(all(arrays[a].shape[1]==len(cols[a]) for a in ARMS),'Adapted width mismatch')
 (out/'shards').mkdir(parents=True);dst=out/'shards'/block;shutil.copytree(src,dst)
 write_adapted_features(dst/'features.npz',sample_ids,block,arrays)
 lineage[block]=cols;write(dst/'lineage.json',lineage)
 e=load(evaluation);require(e.get('schema')=='r02_structure_evaluation_config_v1','Frozen evaluation config required')
 scope=load(dst/'contract.json')['exclusion_scope']
 e,amendment_sha,base_evaluation_sha=apply_amendment(evaluation,amendment,scope,points.shape[1])
 sc=load(dst/'contract.json');sc['full_common_binding']={'source_npz_sha256':sha(common),'array':common_key,'axes':points.shape[1],'replacement':'COMMON_COLUMNS_ONLY_RARE_SUFFIXES_PRESERVED'}
 sc['run_scope_amendment_sha256']=amendment_sha
 write(dst/'contract.json',sc)
 sc,_=rebind_verification(dst,common,amendment_sha)
 outputs=sm['outputs_sha256']
 for n in outputs:
  if n=='contract.json':continue
  outputs[n]=sha(dst/n)
 outputs['contract.json']=sha(dst/'contract.json')
 sm.update(schema='r02_structure_fullcommon_adapted_block_v1',stage='COMPLETE_PREPARED_FULLCOMMON_B0_B1_B2_NO_TRAINING',source_manifest_sha256=matches[0]['manifest_sha256'],common_source_sha256=sha(common),outputs_sha256=outputs,training_allowed=False)
 write(dst/'manifest.json',sm)
 shutil.copyfile(root/'persons.tsv',out/'persons.tsv');shutil.copyfile(root/'protocol.md',out/'protocol.md')
 with (root/'blocks.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
 row=[x for x in rows if x['block_id']==block];require(len(row)==1,'Root block row missing')
 with (out/'blocks.tsv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row[0]),delimiter='\t');w.writeheader();w.writerows(row)
 write(out/'block_index.json',{'schema':'r02_structure_block_index_v1','blocks':[{'block_id':block,'path':'shards/'+block,'contract_sha256':sha(dst/'contract.json'),'manifest_sha256':sha(dst/'manifest.json')}]})
 c=dict(rc);c['pair_missing_indicator_scope']={block:sc['pair_missing_indicator_scope'][block]};c['full_common_binding']=sc['full_common_binding'];c['run_scope_amendment_sha256']=amendment_sha;c['sha256']={n:sha(out/n) for n in ('persons.tsv','blocks.tsv','block_index.json','protocol.md')};write(out/'contract.json',c)
 e['prepared_contract_sha256']=sha(out/'contract.json');write(out/'evaluation.json',e)
 write(out/'assembly-manifest.json',{'schema':'r02_structure_fullcommon_v4_single_view_v1','status':'ASSEMBLED_NO_FIT','block_id':block,'arms':{'B0':arrays['B0'].shape[1],'B1':arrays['B1'].shape[1],'B2':arrays['B2'].shape[1]},'source_manifest_sha256':matches[0]['manifest_sha256'],'common_source_sha256':sha(common),'contract_sha256':sha(out/'contract.json'),'base_evaluation_sha256':base_evaluation_sha,'run_scope_amendment_sha256':amendment_sha,'evaluation_sha256':sha(out/'evaluation.json'),'evaluate_allowed':False})
 return load(out/'assembly-manifest.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared-root',required=True);p.add_argument('--common-npz',required=True);p.add_argument('--common-key',choices=('points','scores'),default='points');p.add_argument('--block-id',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--run-scope-amendment',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.prepared_root,a.common_npz,a.block_id,a.output,a.evaluation,a.common_key,a.run_scope_amendment)))
