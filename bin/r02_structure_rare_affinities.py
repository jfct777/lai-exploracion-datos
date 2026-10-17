#!/usr/bin/env python3
"""Assemble both K03 rare geometries from verified additive count products.

Sum original chromosome counts once. Each view removes its target chromosome
and, for BLOCK_AND_LD_MARGIN, adds only that chromosome's exact allowed partial.
Jaccard is formed after integer aggregation. No community or target is read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
import r02_structure_rare_inputs as producer
import r02_structure_affinity_io as affinity

material=producer.material; evidence=producer.evidence; require=producer.require
SCHEMA='r02_structure_rare_affinity_assembly_v1'
CODE=tuple(dict.fromkeys(('r02_structure_rare_affinities.py',*producer.CODE)))


def implementation():
    return {name:material.sha(Path(__file__).with_name(name))for name in CODE}


def pin(desc, limit):
    require(set(desc)=={'path','sha256'},'Exact rare assembly pin required')
    return producer.continuous.pinned(Path(desc['path']).resolve(),desc['sha256'],limit)


def load_counts(path, digest, ids, chrom, limits):
    path=pin(dict(path=str(path),sha256=digest),limits['max_input_bytes'])
    with zipfile.ZipFile(path)as z:
        require(sum(m.file_size for m in z.infolist())<=limits['max_array_bytes'],'Rare count decompression bound exceeded')
    with np.load(path,allow_pickle=False)as z:
        require(set(z.files)==producer.reuse.ARRAYS and z['samples'].tolist()==ids and
                str(z['sample_hash'])==evidence.sample_hash(ids)and z['chromosomes'].tolist()==[chrom]
                and str(z['kind'])=='rare'and str(z['schema'])==evidence.SCHEMA,'Rare count roster/source chromosome differs')
        a={k:z[k]for k in ('I','U','Q','individual_carrier','individual_callable','n_sites')}
    evidence.validate_rare_counts(a['I'],a['U'],a['Q'])
    require(all(a[k].shape==(len(ids),len(ids))for k in ('I','U','Q'))and
            a['n_sites'].shape==()and a['n_sites'].dtype.kind in 'iu'and 0<=int(a['n_sites'])<=np.iinfo(np.int64).max and
            np.all(a['Q']<=int(a['n_sites']))and
            all(a[k].shape==(len(ids),)and a[k].dtype.kind in 'iu'for k in ('individual_carrier','individual_callable'))and
            np.array_equal(np.diag(a['I']),a['individual_carrier'])and
            np.array_equal(np.diag(a['U']),a['individual_carrier'])and
            np.array_equal(np.diag(a['Q']),a['individual_callable']),'Rare count denominator/diagonal differs')
    require(material.sha(path)==digest,'Rare count input changed while reading')
    return a


def inspect(bundle,request,digest):
    request=pin(dict(path=str(request),sha256=digest),2**20);r=material.load_json(request)
    require(set(r)=={'schema','evidence_scope','source_contract_sha256','implementation_sha256','chromosomes','resources'}and
            r['schema']==SCHEMA and r['implementation_sha256']==implementation(),'Rare assembly request/code differs')
    limits=r['resources']
    require(set(limits)=={'max_metadata_bytes','max_input_bytes','max_array_bytes','max_memory_bytes',
                         'max_output_bytes','max_pairs','min_free_bytes'}and
            all(type(v)is int and v>0 for v in limits.values()),'Explicit rare assembly resource bounds required')
    bundle=Path(bundle).resolve();pin(dict(path=str(bundle/'materialize.json'),sha256=r['source_contract_sha256']),limits['max_metadata_bytes'])
    source,persons,roles,blocks,regions=material.load_context(bundle)
    require(source['roles_state']=='EXPLICITLY_FROZEN'and source['exclusion_scope']=='BLOCK_AND_LD_MARGIN'and
            source.get('additional_exclusion_scopes')==['WHOLE_CHROMOSOME'],'Both frozen rare geometries required')
    require(r['evidence_scope']in{'SYNTHETIC_INSTRUMENT_ONLY','DNABR_R02'},'Explicit rare evidence scope required')
    if r['evidence_scope']=='DNABR_R02':
        require(len(persons)==2619 and {k:len(v)for k,v in roles.items()}=={'FIT':1460,'SELECT':476,'EVALUATE':683}
                and len(blocks)==44,'R02 rare cohort/roles/targets differ')
    require(set(r['chromosomes'])==set(source['sources']),'Every source chromosome required for global counts')
    ids=[p['sample_id']for p in persons];tasks,_=material.make_tasks(source,regions);catalogue={};total=0
    for chrom,desc in r['chromosomes'].items():
        require(set(desc)=={'directory','manifest_sha256','producer_code_sha256'},'Exact chromosome producer pins required')
        directory=Path(desc['directory']).resolve();m=material.load_json(pin(dict(path=str(directory/'manifest.json'),sha256=desc['manifest_sha256']),limits['max_metadata_bytes']))
        require(m['schema']==producer.SCHEMA and m['stage']=='EXACT_INTERVAL_COUNTS_NOT_K03_FIT'and
                m['implementation_sha256']==desc['producer_code_sha256']and m['chromosome']==chrom and
                m['source_contract_sha256']==r['source_contract_sha256']and m['persons_sha256']==evidence.sample_hash(ids)and
                m['n_people']==len(ids)and m['source_rare_rule']==source['source_rare_rule'],
                'Rare chromosome interval producer identity differs')
        expected={k:v for k,v in tasks.items()if v['kind']=='features'and v['chromosome']==chrom}
        require(set(m['tasks'])==set(expected),'Rare interval task coverage differs')
        entries={}
        for key,row in m['tasks'].items():
            require(row['request']==expected[key],'Rare interval geometry differs')
            if 'reused_original_m02'in row:
                require(row['request']['intervals']==[[1,source['sources'][chrom]['length_bp']]],'Only full chromosome can reuse M02')
                old=row['reused_original_m02'];hm=material.load_json(pin(old['manifest'],limits['max_metadata_bytes']))
                p=pin(old['npz'],limits['max_input_bytes'])
                require(hm['input_vcf_sha256']==source['sources'][chrom]['rare_vcf']['sha256']and
                        hm['output_npz_sha256']==old['npz']['sha256'],'Original chromosome source differs')
                record=dict(path=str(p),sha256=old['npz']['sha256'])
            else:
                require(Path(row['path']).name==row['path'],'Unsafe interval product path')
                p=pin(dict(path=str(directory/row['path']),sha256=row['sha256']),limits['max_input_bytes'])
                require(p.stat().st_size==row['bytes'],'Interval product size differs')
                record=dict(path=str(p),sha256=row['sha256'])
            total+=p.stat().st_size;entries[key]=dict(**record,request=row['request'],candidate_sites=row['candidate_sites'])
        full=[k for k,v in entries.items()if v['request']['intervals']==[[1,source['sources'][chrom]['length_bp']]]]
        require(len(full)==1,'Unique original full-chromosome counts required')
        catalogue[chrom]=dict(entries=entries,full_id=full[0],producer_manifest=dict(path=str(directory/'manifest.json'),sha256=desc['manifest_sha256']))
    require(total<=limits['max_input_bytes'],'Rare assembly total input bound exceeded')
    # Global matrices + one decoded chromosome + pair vectors + compression.
    n=len(ids);require(160*n*n+48*limits['max_pairs']<=limits['max_memory_bytes'],'Rare assembly memory bound exceeded')
    return r,source,persons,roles,blocks,catalogue


def reduce(bundle,request,digest,output):
    r,source,persons,roles,blocks,catalogue=inspect(bundle,request,digest)
    output=material.new_directory(output);limits=r['resources'];ids=[p['sample_id']for p in persons]
    require(shutil.disk_usage(output).free>=limits['min_free_bytes'],'Rare reduction disk reserve exhausted')
    total=None
    for chrom in sorted(catalogue,key=int):
        row=catalogue[chrom]['entries'][catalogue[chrom]['full_id']]
        counts=load_counts(row['path'],row['sha256'],ids,chrom,limits)
        require(int(counts['n_sites'])==row['candidate_sites'],'Full chromosome site count differs')
        if total is None:total={k:a.astype(np.int64,copy=True)for k,a in counts.items()}
        else:
            require(int(counts['n_sites'])<=np.iinfo(np.int64).max-int(total['n_sites']),'Rare global count overflow')
            for k in total:total[k]+=counts[k]
        del counts
    evidence.validate_rare_counts(total['I'],total['U'],total['Q'])
    np.savez_compressed(output/'global.private.npz',**total)
    require((output/'global.private.npz').stat().st_size<=limits['max_output_bytes'],'Rare global output bound exceeded')
    m=dict(schema=SCHEMA,stage='ADDITIVE_GLOBAL_RARE_COUNTS_NOT_TRAINING',request_sha256=digest,
        source_contract_sha256=r['source_contract_sha256'],persons_content_sha256=material.digest(persons),
        implementation_sha256=implementation(),catalogue=catalogue,n_sites=int(total['n_sites']),
        array_sha256=material.sha(output/'global.private.npz'),array_bytes=(output/'global.private.npz').stat().st_size,
        request_path=str(Path(request).resolve()),source_metadata={str((Path(bundle)/name).resolve()):material.sha(Path(bundle)/name)
            for name in ('materialize.json',*source['metadata_sha256'])},resources=limits,evidence_scope=r['evidence_scope'],
        outcomes_read=False,training_allowed=False)
    require(material.sha(request)==digest and implementation()==r['implementation_sha256'],'Rare assembly request/code changed')
    material.write_json(output/'manifest.json',m);return m


def view(bundle,global_dir,global_sha256,scope,block_id,output):
    global_dir=Path(global_dir).resolve();m=material.load_json(pin(dict(path=str(global_dir/'manifest.json'),sha256=global_sha256),2**24))
    require(m['schema']==SCHEMA and m['stage']=='ADDITIVE_GLOBAL_RARE_COUNTS_NOT_TRAINING'and
            m['implementation_sha256']==implementation()and scope in affinity.discrete.SCOPES,
            'Current rare aggregate/geometry required')
    limits=m['resources'];require(material.sha(m['request_path'])==m['request_sha256']and
        all(material.sha(p)==h for p,h in m['source_metadata'].items()),'Rare aggregate source metadata changed')
    source,persons,roles,blocks,_=material.load_context(bundle)
    require(material.sha(Path(bundle)/'materialize.json')==m['source_contract_sha256']and
            material.digest(persons)==m['persons_content_sha256'],'Rare aggregate cohort/source differs')
    recipes=material.input_plan.plan_regions(blocks,{c:v['length_bp']for c,v in source['sources'].items()},scope)
    recipes=[v for v in recipes if v['block_id']==block_id];require(len(recipes)==1,'Unknown rare target view')
    recipe=recipes[0];chrom=recipe['target'][0];ids=[p['sample_id']for p in persons]
    gp=pin(dict(path=str(global_dir/'global.private.npz'),sha256=m['array_sha256']),limits['max_input_bytes'])
    with zipfile.ZipFile(gp)as z:require(sum(v.file_size for v in z.infolist())<=limits['max_array_bytes'],'Global rare decompression bound exceeded')
    with np.load(gp,allow_pickle=False)as z:counts={k:z[k].astype(np.int64,copy=True)for k in ('I','U','Q','individual_carrier','individual_callable','n_sites')}
    catalogue=m['catalogue'][chrom];pin(catalogue['producer_manifest'],limits['max_metadata_bytes'])
    full=catalogue['entries'][catalogue['full_id']];remove=load_counts(full['path'],full['sha256'],ids,chrom,limits)
    for k in counts:counts[k]-=remove[k]
    del remove
    included=[full['request']for c,v in m['catalogue'].items()if c!=chrom for full in [v['entries'][v['full_id']]]]
    allowed=[v[1:]for v in recipe['allowed_input_intervals']if v[0]==chrom]
    if allowed:
        matches=[v for v in catalogue['entries'].values()if v['request']['intervals']==allowed]
        require(len(matches)==1,'Exact allowed rare interval count missing')
        add=load_counts(matches[0]['path'],matches[0]['sha256'],ids,chrom,limits)
        for k in counts:counts[k]+=add[k]
        del add;included.append(matches[0]['request'])
    evidence.validate_rare_counts(counts['I'],counts['U'],counts['Q'])
    nsites=int(counts['n_sites']);require(nsites>0 and np.all(counts['Q']<=nsites),'Rare view candidate denominator invalid')
    keys=affinity.pair_indices(persons,roles,'EXCLUDE')
    require(sum(len(a)for a,b in keys)<=limits['max_pairs'],'Rare view pair bound exceeded')
    totals=[np.vstack([counts[k][a,b]for k in ('I','U','Q')])for a,b in keys];del counts
    output=material.new_directory(output);affinity.write_pair_tables(output,keys,totals,nsites,limits)
    material.input_plan.write_table(output/'persons.tsv',persons);material.write_json(output/'exclusion.json',recipe)
    result=dict(schema=affinity.NEW_SCHEMA,stage=affinity.STAGE,block_id=block_id,exclusion_scope=scope,exclusion=recipe,
        source_contract_sha256=m['source_contract_sha256'],persons_content_sha256=material.digest(persons),
        global_manifest_sha256=global_sha256,implementation_sha256=implementation(),candidate_sites=nsites,
        included_partials=included,same_component_fit_edges='EXCLUDE',missing_edge_policy='UNKNOWN_NOT_ZERO',
        projection_rule=affinity.discrete.PROJECTION_RULE,
        affinity=dict(source_measure='SOURCE_MINOR_JACCARD',transformation='IDENTITY_NONNEGATIVE_INPUT',
            definition='Sum of shared carriage I divided by sum of jointly observed carriage union U; U=0 unknown',
            code_sha256=material.sha(__file__),maximum=1.),
        sha256={p.name:material.sha(p)for p in output.iterdir()if p.is_file()},
        n_fit_edges=len(keys[0][0]),n_query_edges=len(keys[1][0]),training_allowed=False,outcomes_read=False,
        missing_for_projection_bundle=['communities.private.tsv','frozen_communities.json','community_method.json'])
    require(material.sha(global_dir/'manifest.json')==global_sha256 and material.sha(gp)==m['array_sha256']and
            implementation()==m['implementation_sha256'],'Rare aggregate/code changed during view construction')
    material.write_json(output/'affinity_inputs.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['reduce','view'],required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--request',type=Path);p.add_argument('--request-sha256')
    p.add_argument('--global-dir',type=Path);p.add_argument('--global-sha256');p.add_argument('--scope');p.add_argument('--block-id')
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();os.umask(0o077)
    result=reduce(a.bundle,a.request,a.request_sha256,a.output_dir)if a.stage=='reduce'else view(a.bundle,a.global_dir,a.global_sha256,a.scope,a.block_id,a.output_dir)
    print(json.dumps({k:result[k]for k in ('schema','stage')}))


if __name__=='__main__':main()
