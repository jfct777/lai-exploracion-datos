#!/usr/bin/env python3
"""Emit authenticated C and R+C profile requests from completed frozen graphs.

This is a pure planner: it neither builds graphs nor runs the adaptive profile.
The caller supplies an authenticated graph registry and a separately admitted
numerical policy.  No R-family admission or numerical value is inherited.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import r02_structure_fit_graph as graph_io
import r02_structure_profile_queue as queue

REGISTRY_SCHEMA='r02_fit_family_graph_registry_v1'
POLICY_SCHEMA='r02_fit_family_profile_numerical_policy_v1'
POLICY_ADMISSION_SCHEMA='r02_fit_family_profile_numerical_policy_admission_v1'
FAMILIES={'C':88,'R_PLUS_C':264}
require=graph_io.io.require


def sha(path):return graph_io.io.sha(Path(path))
def load(path):return graph_io.io.load_json(Path(path))
def pin(record,maximum=32*2**20):
    require(set(record)=={'path','sha256'},'Exact path/SHA descriptor required')
    return graph_io.pin(record['path'],record['sha256'],maximum)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if path.exists():require(path.read_bytes()==raw,'Refuse to change immutable generated record')
    else:path.write_bytes(raw)
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest()}


def validate_code(directory,expected):
    raw=Path(directory);require(raw.is_absolute() and '..' not in raw.parts,'Safe absolute frozen code directory required')
    directory=raw.resolve();require(expected and directory.is_dir(),'Frozen code directory required')
    for name,digest in expected.items():
        require((directory/name).is_file() and sha(directory/name)==digest,'Frozen implementation differs: '+name)


def validate_registry(path,digest):
    registry=load(graph_io.pin(path,digest,64*2**20))
    require(set(registry)=={'schema','status','evidence_scope','producer_code_dir',
        'producer_implementation_sha256','completion_evidence','graphs'}
        and registry['schema']==REGISTRY_SCHEMA and registry['status']=='AUTHENTICATED_COMPLETE'
        and registry['evidence_scope']in {'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'},'Graph registry differs')
    expected=graph_io.implementation_hashes()
    require(registry['producer_implementation_sha256']==expected,
        'Graph producer differs from the producer frozen with this queue')
    validate_code(registry['producer_code_dir'],expected)
    require(isinstance(registry['completion_evidence'],list) and registry['completion_evidence'],'Completion evidence required')
    for record in registry['completion_evidence']:
        evidence=load(pin(record));require(evidence.get('status')=='COMPLETE','Graph completion evidence is not complete')
    graphs=registry['graphs'];require(isinstance(graphs,list) and graphs,'Graph records required')
    seen=set();counts={family:0 for family in FAMILIES}
    for record in graphs:
        require(set(record)=={'family','coverage','graph','view_id','source_sha256','territory'},'Exact graph registry record required')
        family=record['family'];require(family in FAMILIES and type(record['coverage'])is int and record['coverage']>0,'Supported explicit family coverage required')
        require(isinstance(record['view_id'],str) and re.fullmatch(r'[A-Za-z0-9._-]+',record['view_id']),
            'Safe explicit view identifier required')
        require(set(record['graph'])=={'directory','manifest_sha256'},'Exact graph directory/manifest descriptor required')
        graph_dir=Path(record['graph']['directory'])
        require(graph_dir.is_absolute() and '..' not in graph_dir.parts,'Safe absolute graph directory required')
        manifest_path=graph_dir/'graph_manifest.json'
        manifest=load(graph_io.pin(manifest_path,record['graph']['manifest_sha256'],32*2**20))
        require(manifest.get('schema')==graph_io.FAMILY_GRAPH_SCHEMA and manifest.get('stage')=='FIT_GRAPH_ONLY_NO_COMMUNITIES_OR_SELECTION'
            and manifest.get('implementation_sha256')==expected and manifest.get('evidence_scope')==registry['evidence_scope']
            and manifest.get('family')==family and manifest.get('view_id')==record['view_id']
            and manifest.get('source_sha256')==record['source_sha256'] and manifest.get('territory')==record['territory'],
            'Registry/graph family, source, territory or frozen producer differs')
        identity=(family,record['view_id']);require(identity not in seen,'Duplicate family view');seen.add(identity);counts[family]+=1
    if registry['evidence_scope']=='DNABR_R02':require(counts==FAMILIES,'R02 registry must contain all 88 C and 264 R+C graphs')
    require(all(record['coverage']==counts[record['family']] for record in graphs),'Per-record family coverage differs')
    return registry,counts


def validate_policy(path,digest,admission_path,admission_digest,counts):
    policy=load(graph_io.pin(path,digest,4*2**20));admission=load(graph_io.pin(admission_path,admission_digest,4*2**20))
    required={'schema','status','evidence_scope','scientific_spec_sha256','families','search',
              'candidate_resources','resources','code_dir','implementation_sha256','work_root'}
    require(set(policy)==required and policy['schema']==POLICY_SCHEMA and policy['status']=='DECLARED_NUMERICAL_POLICY'
        and policy['families']==counts and graph_io.io.valid_hash(policy['scientific_spec_sha256']),'Exact declared family policy required')
    expected=queue.implementation_hashes();require(policy['implementation_sha256']==expected,'Queue implementation policy differs')
    validate_code(policy['code_dir'],expected)
    search=policy['search'];require(set(search)=={'family_error','missing_mass','max_queries','max_episodes_per_query','max_total_seeds','max_iterations'},'Exact common numerical search policy required')
    queue.policy.probability(search['family_error'],'family_error');queue.policy.probability(search['missing_mass'],'missing_mass')
    require(all(type(search[k])is int and search[k]>0 for k in ('max_queries','max_episodes_per_query','max_total_seeds','max_iterations')),'Positive search stops required')
    resource_fields={'max_input_bytes','max_metadata_bytes','max_memory_bytes','max_output_bytes','min_free_bytes'}
    for key in ('candidate_resources','resources'):
        value=policy[key];require(set(value)==resource_fields and all(type(v)is int and v>0 for v in value.values()),
            'Explicit positive '+key+' required')
    work_root=Path(policy['work_root'])
    require(work_root.is_absolute() and '..' not in work_root.parts,'Safe absolute policy work root required')
    require(set(admission)=={'schema','status','policy_sha256','evidence_scope','scientific_spec_sha256','families','evaluate_allowed','numerical_policy_admitted'}
        and admission['schema']==POLICY_ADMISSION_SCHEMA and admission['status']=='FIT_PROFILE_POLICY_ADMITTED'
        and admission['policy_sha256']==digest and admission['evidence_scope']==policy['evidence_scope']
        and admission['scientific_spec_sha256']==policy['scientific_spec_sha256'] and admission['families']==counts
        and admission['evaluate_allowed']is False and admission['numerical_policy_admitted']is True,
        'Separate C/R+C numerical policy admission required')
    return policy,admission


def validate_graph_payloads(registry,policy):
    limits=policy['resources']
    for record in registry['graphs']:
        _graph,_persons,manifest=graph_io.load_graph(
            record['graph']['directory'],record['graph']['manifest_sha256'],
            max_input_bytes=limits['max_input_bytes'],
            max_metadata_bytes=limits['max_metadata_bytes'],
            max_memory_bytes=limits['max_memory_bytes'])
        require(manifest['family']==record['family'] and manifest['source_sha256']==record['source_sha256']
            and manifest['territory']==record['territory'],'Authenticated graph payload identity differs')


def generate(registry_path,registry_sha256,policy_path,policy_sha256,policy_admission,
             policy_admission_sha256,output_dir):
    registry,counts=validate_registry(registry_path,registry_sha256)
    policy,_=validate_policy(policy_path,policy_sha256,policy_admission,policy_admission_sha256,counts)
    require(policy['evidence_scope']==registry['evidence_scope'],'Policy and graphs have different evidence scopes')
    validate_graph_payloads(registry,policy)
    out=Path(output_dir).resolve();require(not out.exists() and out.parent.is_dir(),'Fresh output directory required');out.mkdir()
    coverage={}
    for family in FAMILIES:
        rows=sorted(({'view_id':x['view_id'],'source_sha256':x['source_sha256']} for x in registry['graphs'] if x['family']==family),key=lambda x:x['view_id'])
        coverage[family]=write(out/'coverage'/(family+'.json'),{'schema':'r02_fit_family_profile_coverage_v1','family':family,'views':rows})
    tasks=[]
    for graph in sorted(registry['graphs'],key=lambda x:(x['family'],x['view_id'])):
        family=graph['family'];task_id=family+'_'+graph['view_id']
        request={'schema':queue.FAMILY_QUEUE,'evidence_scope':registry['evidence_scope'],
            'scientific_spec_sha256':policy['scientific_spec_sha256'],'implementation_sha256':policy['implementation_sha256'],
            'graph':graph['graph'],'coverage':coverage[family],
            'work_root':str((Path(policy['work_root'])/task_id).resolve()),'code_dir':str(Path(policy['code_dir']).resolve()),
            'search':dict(policy['search'],views=counts[family]),'candidate_resources':policy['candidate_resources'],'resources':policy['resources']}
        request_pin=write(out/'requests'/family/(task_id+'.json'),request)
        admission={'schema':'r02_fit_family_profile_queue_admission_v2','status':'FIT_PROFILE_POLICY_ADMITTED',
            'request_sha256':request_pin['sha256'],'scientific_spec_sha256':policy['scientific_spec_sha256'],
            'evidence_scope':registry['evidence_scope'],'evaluate_allowed':False,'coverage':coverage[family],
            'family':family,'source_sha256':graph['source_sha256'],'territory':graph['territory'],
            'numerical_policy':{'path':str(Path(policy_path).resolve()),'sha256':policy_sha256},
            'numerical_policy_admission':{'path':str(Path(policy_admission).resolve()),'sha256':policy_admission_sha256}}
        admission_pin=write(out/'admissions'/family/(task_id+'.json'),admission)
        tasks.append({'task_id':task_id,'family':family,'coverage':counts[family],'view_id':graph['view_id'],
            'source_sha256':graph['source_sha256'],'territory':graph['territory'],
            'request':request_pin,'admission':admission_pin})
    result={'schema':'r02_fit_family_profile_request_batch_v1','status':'READY_AUTHENTICATED_GRAPH_REUSE',
        'task_count':len(tasks),'family_counts':counts,'registry':{'path':str(Path(registry_path).resolve()),'sha256':registry_sha256},
        'policy':{'path':str(Path(policy_path).resolve()),'sha256':policy_sha256},'graphs_rebuilt':False,
        'evaluate_allowed':False,'tasks':tasks}
    write(out/'REQUEST_BATCH.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registry','registry-sha256','policy','policy-sha256','policy-admission','policy-admission-sha256','output-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(generate(a.registry,a.registry_sha256,a.policy,a.policy_sha256,a.policy_admission,a.policy_admission_sha256,a.output_dir),indent=2))
if __name__=='__main__':main()
