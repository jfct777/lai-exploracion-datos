#!/usr/bin/env python3
"""Recover exact selected common loci absent from the mapped event cache.

Consumes a previously authenticated indexed source slice. No LD selection,
reorientation, map interpolation, cohort filter or model fitting is performed.
NaN genetic-map coordinates explicitly mean unavailable and are never imputed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

import r02_structure_common_inputs as common

material=common.material;shared=common.shared;require=common.require
SCHEMA='r02_structure_common_supplement_v1'


def implementation():
    return {**common.implementation(),Path(__file__).name:shared.sha(__file__)}


def extract(bundle,request,digest,output):
    request=common.continuous.pinned(Path(request).resolve(),digest,2**20);r=shared.load_json(request)
    require(set(r)=={'schema','source_contract_sha256','implementation_sha256','chromosome','slice_request','slice_manifest','slice_vcf','loci','resources'}
            and r['schema']==SCHEMA and r['implementation_sha256']==implementation(),'Exact common supplement specification required')
    common.continuous.pinned(Path(bundle).resolve()/'materialize.json',r['source_contract_sha256'],2**20)
    source,persons,roles,_,_=material.load_context(bundle);chrom=r['chromosome']
    require(source['roles_state']=='EXPLICITLY_FROZEN' and chrom in source['sources'],'Frozen original source roles/chromosome required')
    limits=r['resources'];require(set(limits)=={'max_input_bytes','max_loci','max_output_bytes','max_memory_bytes','min_free_bytes'}
        and all(type(v)is int and v>0 for v in limits.values()),'Explicit supplement bounds required')
    pins={str(request):digest,str(Path(bundle).resolve()/'materialize.json'):r['source_contract_sha256']}
    for name,value in source['metadata_sha256'].items():pins[str(Path(bundle).resolve()/name)]=value
    for key in ('slice_request','slice_manifest','slice_vcf','loci'):
        d=r[key];require(set(d)=={'path','sha256'},'Exact pinned supplement input required')
        path=common.continuous.pinned(Path(d['path']).resolve(),d['sha256'],limits['max_input_bytes']);pins[str(path)]=d['sha256']
    receipt=shared.load_json(r['slice_manifest']['path'])
    slice_request=shared.load_json(r['slice_request']['path'])
    require(receipt['schema']=='r02_structure_indexed_source_slice_v1' and receipt['stage']=='INDEXED_SOURCE_SLICE_NOT_FIT'
            and slice_request['schema']==receipt['schema'] and receipt['request_sha256']==r['slice_request']['sha256']
            and receipt['source']==slice_request['source']
            and receipt['implementation_sha256']==slice_request['implementation_sha256']
            and receipt['source']['sha256']==source['sources'][chrom]['common_vcf']['sha256']
            and receipt['output_sha256']==r['slice_vcf']['sha256'],'Slice does not bind original common source bytes')
    wanted=shared.load_json(r['loci']['path'])
    require(isinstance(wanted,list) and 0<len(wanted)<=limits['max_loci'],'Exact nonempty bounded locus list required')
    previous=0;by_position={}
    for row in wanted:
        require(set(row)=={'variant_id','position_bp','fit_alt_frequency'},'Exact selected-locus identity required')
        parts=row['variant_id'].split(':');pos=row['position_bp'];p=float(row['fit_alt_frequency'])
        require(len(parts)==4 and parts[0]==chrom and parts[1]==str(pos) and type(pos)is int and pos>previous
                and all(len(v)==1 and v in 'ACGT'for v in parts[2:]) and parts[2]!=parts[3]
                and np.isfinite(p) and min(p,1-p)>=source['common']['min_maf'],'Selected supplement allele/position/FIT frequency differs')
        by_position[pos]=row;previous=pos
    n=len(persons);count=len(wanted)
    require(n*count*32+2**20<=limits['max_memory_bytes'],'Supplement allocation bound exceeded')
    output=Path(output);require(not output.exists(),'Preserve completed common supplement')
    require(shutil.disk_usage(output.parent).free>=limits['min_free_bytes']+limits['max_output_bytes'],'Supplement disk reserve insufficient')
    output.mkdir(mode=0o700)
    tool=material.checked_tool('bcftools',source['tools']['bcftools'])
    actual=subprocess.check_output([tool,'query','-l',r['slice_vcf']['path']],text=True).splitlines()
    require(actual==[p['sample_id']for p in persons],'Sliced VCF sample order differs from original analysis roster')
    samples=output/'samples.txt';samples.write_text(''.join(p['sample_id']+'\n'for p in persons))
    arrays=dict(dosage=np.full((n,count),255,np.uint8),observed=np.zeros((n,count),bool),
        position_bp=np.asarray(list(by_position),np.int64),position_cm=np.full(count,np.nan),ref=np.zeros(count,np.uint8),
        alt=np.zeros(count,np.uint8),counted_allele=np.ones(count,np.uint8),allele_count=np.zeros(count,np.int64),
        allele_number=np.zeros(count,np.int64),frequency=np.zeros(count,np.float64))
    index={p:i for i,p in enumerate(by_position)};seen=set();src=source['sources'][chrom]
    fmt='%CHROM\t%POS\t%REF\t%ALT\t%FILTER\t%INFO/ORIG_NALLELES[\t%GT]\n'
    with(output/'bcftools.log').open('x')as log:
        proc=subprocess.Popen([tool,'query','-S',str(samples),'-f',fmt,r['slice_vcf']['path']],stdout=subprocess.PIPE,stderr=log,text=True)
        try:
            previous=0
            for line in proc.stdout:
                fields=line.rstrip('\n').split('\t');require(len(fields)==6+n,'Supplement genotype columns differ')
                pos,eligible=material.basic_site(fields,src,previous);previous=pos
                if not eligible:continue
                key=f'{chrom}:{pos}:{fields[2]}:{fields[3]}'
                require(pos in by_position and key==by_position[pos]['variant_id'] and pos not in seen,'Unexpected/duplicate selected supplement locus')
                gt=material.decode_gt(fields[6:]);keep,p=material.common_eligible(gt,roles['FIT'],source['common'])
                require(keep and abs(p-float(by_position[pos]['fit_alt_frequency']))<=1e-10,'Supplement FIT eligibility/frequency differs')
                j=index[pos];mask=np.isfinite(gt);arrays['observed'][:,j]=mask;arrays['dosage'][mask,j]=gt[mask].astype(np.uint8)
                arrays['ref'][j]='ACGT'.index(fields[2]);arrays['alt'][j]='ACGT'.index(fields[3]);arrays['frequency'][j]=p
                fit=roles['FIT'];arrays['allele_count'][j]=int(np.nansum(gt[fit]));arrays['allele_number'][j]=2*int(mask[fit].sum());seen.add(pos)
            require(proc.wait()==0 and seen==set(by_position),'Selected source slice lacks required loci')
        finally:
            proc.stdout.close()
            if proc.poll()is None:proc.kill();proc.wait(timeout=10)
    path=output/'supplement.npz';np.savez(path,**arrays);require(path.stat().st_size<=limits['max_output_bytes'],'Supplement output bound exceeded')
    require(all(shared.sha(p)==h for p,h in pins.items())and implementation()==r['implementation_sha256'],'Supplement input/code changed')
    manifest=dict(schema=SCHEMA,stage='EXACT_SELECTED_COMMON_LOCI_NO_RESELECTION',request_sha256=digest,
        source_contract_sha256=r['source_contract_sha256'],implementation_sha256=implementation(),chromosome=chrom,
        sample_ids_sha256=material.evidence.sample_hash([p['sample_id']for p in persons]),n_people=n,n_sites=count,
        position_bp=arrays['position_bp'].tolist(),
        counted_allele='ALT',position_cm_policy='UNAVAILABLE_NAN_NOT_USED_FOR_COMMON_SVD',selected_loci_sha256=r['loci']['sha256'],
        original_source_sha256=receipt['source']['sha256'],slice_manifest_sha256=r['slice_manifest']['sha256'],
        original_source=receipt['source'],slice_request_sha256=r['slice_request']['sha256'],
        array=dict(path=path.name,sha256=shared.sha(path),bytes=path.stat().st_size,n_sites=count,
                   start_bp=min(seen),end_bp=max(seen)),target_outcomes_read=False,selection_changed=False)
    material.write_json(output/'manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--request',required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args();os.umask(0o077)
    print(json.dumps(extract(a.bundle,a.request,a.request_sha256,a.output)))
