#!/usr/bin/env python3
"""Freeze a tiny lazy R+C specification over authenticated C and R providers."""
import argparse
import json
import math
from pathlib import Path
import re

import r02_structure_affinity_io as affinity

discrete=affinity.discrete
require,sha,load_json=discrete.require,discrete.sha,discrete.load_json
SCHEMA='r02_structure_lazy_composite_request_v1'
CODE_FILES=tuple(dict.fromkeys(('r02_structure_composite_affinities.py',
    'r02_structure_affinity_io.py',*discrete.CODE_FILES)))


def implementation_hashes():
    return {name:sha(Path(__file__).with_name(name))for name in CODE_FILES}


def pin(path,digest,maximum):
    path=Path(path).resolve()
    require(path.is_absolute()and path.is_file()and discrete.valid_hash(digest)
            and path.stat().st_size<=maximum and sha(path)==digest,
            'Lazy composite request changed or exceeds metadata bound')
    return path


def freeze(request_path,request_sha256,output,*,source_cache=None,defer_consumer_check=False):
    request_path=pin(request_path,request_sha256,2**20);r=load_json(request_path)
    fields={'schema','evidence_scope','scientific_spec_sha256','implementation_sha256',
            'mix_weight','normalization_policy','common_inputs','rare_inputs','resources'}
    require(set(r)==fields and r['schema']==SCHEMA
            and r['implementation_sha256']==implementation_hashes()
            and r['evidence_scope']in{'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'}
            and discrete.valid_hash(r['scientific_spec_sha256'])
            and type(r['mix_weight'])in(int,float)and math.isfinite(r['mix_weight'])
            and 0<=r['mix_weight']<=1
            and r['normalization_policy']=='COMPONENT_PAIR_EQUAL_POSITIVE_MEAN',
            'Exact explicit lazy R+C request required')
    limits=r['resources']
    require(set(limits)=={'max_input_bytes','max_metadata_bytes'}
            and all(type(v)is int and v>0 for v in limits.values()),
            'Explicit positive lazy R+C input bounds required')
    cache_key=json.dumps(dict(evidence_scope=r['evidence_scope'],common_inputs=r['common_inputs'],
        rare_inputs=r['rare_inputs'],resources=limits),sort_keys=True,separators=(',',':'))
    prepared=None if source_cache is None else source_cache.get(cache_key)
    if prepared is None:
        common=affinity.AuthenticatedAffinityView(r['common_inputs'],evidence_scope=r['evidence_scope'],
            max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'],expected_family='C')
        rare=affinity.AuthenticatedRareView(r['rare_inputs'],evidence_scope=r['evidence_scope'],
            max_input_bytes=limits['max_input_bytes'],max_metadata_bytes=limits['max_metadata_bytes'])
        rs,rn,rm=affinity.component_pair_positive_scale(rare.fit_edges(),rare.persons)
        cs,cn,cm=affinity.component_pair_positive_scale(common.fit_edges(),common.persons)
        prepared=(common,rare,rs,rn,rm,cs,cn,cm)
        if source_cache is not None:source_cache[cache_key]=prepared
    common,rare,rs,rn,rm,cs,cn,cm=prepared
    territory=common.upstream['territory']
    require(common.persons==rare.persons and rare.upstream['block_id']==territory['block_id']
            and rare.upstream['exclusion_scope']==territory['exclusion_scope']
            and rare.upstream['source_contract_sha256']==territory['source_contract_sha256'],
            'Lazy R+C common/rare roster or territory differs')
    payload=dict(schema=affinity.COMPOSITE_SCHEMA,
        stage='LAZY_COMPOSITE_READY_NOT_SCIENTIFICALLY_ADMITTED',family='R_PLUS_C',
        mix_weight=r['mix_weight'],normalization_policy=r['normalization_policy'],
        normalization=dict(policy=r['normalization_policy'],
            scope='ELIGIBLE_CROSS_COMPONENT_FIT_EDGES_ONLY',
            fit_positive_rare_scale=rs,fit_positive_common_scale=cs,
            n_positive_rare_fit_edges=rn,n_positive_common_fit_edges=cn,
            positive_rare_fit_component_pair_mass=rm,
            positive_common_fit_component_pair_mass=cm,queries_used_for_scale=False),
        common_inputs=r['common_inputs'],rare_inputs=r['rare_inputs'],territory=territory,
        role_counts=common.upstream['role_counts'],
        persons_content_sha256=common.upstream['persons_content_sha256'],
        capacity=dict(fit_edges=common.expected_edges,
            query_fit_edges=len(common.query_indices)*len(common.fit_set),
            materialized_pair_rows=0,provider='LAZY_PINNED_C_AND_R'),
        target_outcomes_read=False,evaluate_scored=False,scientific_admission=False,
        biological_validation=False)
    output=Path(output)
    require(not output.exists()and output.parent.is_dir(),'Preserve existing lazy composite specification')
    output.mkdir(mode=0o700)
    path=output/'composite_affinity.json'
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');path.chmod(0o600)
    if defer_consumer_check:
        require(load_json(path)==payload and sha(request_path)==request_sha256
                and r['implementation_sha256']==implementation_hashes(),
                'Lazy R+C request/code/spec changed during grouped freeze')
    else:
        view=affinity.AuthenticatedCompositeAffinityView(dict(path=str(path.resolve()),sha256=sha(path)),
            evidence_scope=r['evidence_scope'],max_input_bytes=limits['max_input_bytes'],
            max_metadata_bytes=limits['max_metadata_bytes'])
        require(view.upstream==payload and sha(request_path)==request_sha256
                and r['implementation_sha256']==implementation_hashes(),
                'Lazy R+C request/code/spec changed during freeze')
    return dict(manifest_sha256=sha(path),materialized_pair_rows=0,
                fit_edges=common.expected_edges,query_fit_edges=payload['capacity']['query_fit_edges'],
                scientific_admission=False,biological_validation=False)


def freeze_set(request_set_path,request_set_sha256,output_root):
    request_set_path=pin(request_set_path,request_set_sha256,2**20);body=load_json(request_set_path)
    require(set(body)=={'schema','territory_id','requests'}
            and body['schema']=='r02_structure_lazy_composite_request_set_v1'
            and isinstance(body['territory_id'],str) and re.fullmatch(r'[A-Za-z0-9_-]+',body['territory_id'])
            and isinstance(body['requests'],list) and len(body['requests'])==3,
            'Exact three-weight lazy R+C request set required')
    output_root=Path(output_root);require(output_root.is_dir(),'Existing grouped R+C output root required')
    cache={};results=[];seen=set();weights=[]
    for entry in body['requests']:
        require(set(entry)=={'id','path','sha256'} and isinstance(entry['id'],str)
                and re.fullmatch(r'[A-Za-z0-9_-]+',entry['id']) and entry['id'] not in seen,
                'Unique exact grouped R+C request descriptor required')
        seen.add(entry['id'])
        weights.append(load_json(pin(entry['path'],entry['sha256'],2**20))['mix_weight'])
        results.append(freeze(entry['path'],entry['sha256'],output_root/('composite_'+entry['id']),
                              source_cache=cache,defer_consumer_check=True))
    require(len(cache)==1 and sorted(weights)==[.25,.5,.75],
            'Grouped R+C requests must share one territory and exact finite weights')
    for common,rare,*_ in cache.values():common.verify();rare.verify()
    return dict(territory_id=body['territory_id'],products=len(results),
                materialized_pair_rows=0,scientific_admission=False,biological_validation=False)


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--request');group.add_argument('--request-set')
    p.add_argument('--request-sha256');p.add_argument('--request-set-sha256');p.add_argument('--output',required=True);a=p.parse_args()
    if a.request:
        require(a.request_sha256 is not None and a.request_set_sha256 is None,'Single request SHA required')
        result=freeze(a.request,a.request_sha256,a.output)
    else:
        require(a.request_set_sha256 is not None and a.request_sha256 is None,'Request-set SHA required')
        result=freeze_set(a.request_set,a.request_set_sha256,a.output)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
