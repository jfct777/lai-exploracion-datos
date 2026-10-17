#!/usr/bin/env python3
"""Common multiaxis inputs using exactly the materializer's FIT marker lists.

One feature partial produces reusable Gram/cross statistics. A target combines
the authenticated partials specified by its original exclusion geometry before
a single decomposition. No target outcome, predictive rank or winner is chosen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

import r02_structure_continuous as continuous

material=continuous.discrete.materialize
shared=continuous.discrete.shared
require=continuous.require
SCHEMA='r02_structure_materialized_common_v2'
MERGE_SCHEMA='r02_structure_materialized_common_merge_v2'
BATCH_SCHEMA='r02_structure_materialized_common_batch_v2'
CODE=('r02_structure_common_inputs.py',*continuous.CODE_FILES)
FIELDS={'schema','evidence_scope','source_contract_sha256','implementation_sha256',
        'task_id','weighting','selection','events','supplement','resources'}
LIMITS={'max_metadata_bytes','max_input_bytes','max_chunk_bytes','max_chunk_sites',
        'max_selected_sites','max_memory_bytes','max_output_bytes','min_free_bytes'}


def implementation():
    return {name:shared.sha(Path(__file__).with_name(name)) for name in CODE}


def request_metadata(bundle,request,digest):
    path=continuous.pinned(Path(request).resolve(),digest,2**20)
    r=shared.load_json(path)
    require(set(r)==FIELDS and r['schema']==SCHEMA and r['implementation_sha256']==implementation(),
            'Exact materialized-common request/current implementation required')
    require(r['evidence_scope'] in {'DNABR_R02','SYNTHETIC_INSTRUMENT_ONLY'},'Explicit evidence scope required')
    limits=r['resources']
    require(set(limits)==LIMITS and all(type(v)is int and v>0 for v in limits.values()),
            'Explicit positive common-input resource bounds required')
    continuous.pinned(Path(bundle).resolve()/'materialize.json',r['source_contract_sha256'],limits['max_metadata_bytes'])
    source,persons,roles,blocks,regions=material.load_context(bundle)
    require(source['roles_state']=='EXPLICITLY_FROZEN','Frozen source roles required')
    if r['evidence_scope']=='DNABR_R02':
        require(len(persons)==2619 and {k:len(v)for k,v in roles.items()}=={'FIT':1460,'SELECT':476,'EVALUATE':683}
                and len(blocks)==44,'DNABR cohort/role/target dimensions differ')
    tasks,_=material.make_tasks(source,regions)
    require(r['task_id'] in tasks and tasks[r['task_id']]['kind']=='features','Unknown common feature partial')
    require(r['weighting'] in continuous.POLICIES,'Explicit FIT centering weight required')
    return r,source,persons,roles,blocks,regions,tasks[r['task_id']]


class MaterializedCommonInput:
    def __init__(self,bundle,request,digest):
        self.bundle=Path(bundle).resolve();self.request=Path(request).resolve();self.digest=digest
        self.r,self.source,self.persons,self.roles,self.blocks,self.regions,self.task=request_metadata(bundle,request,digest)
        r=self.r;limits=r['resources'];chrom=self.task['chromosome']
        self.pins={str(self.request):digest,str(self.bundle/'materialize.json'):r['source_contract_sha256']}
        self.pins.update({str(self.bundle/n):v for n,v in self.source['metadata_sha256'].items()})
        def pin(path,digest,limit):
            path=continuous.pinned(Path(path).resolve(),digest,limit);self.pins[str(path)]=digest;return path
        selection=r['selection'];events=r['events']
        for descriptor in (selection,events):
            require(set(descriptor)=={'directory','manifest_sha256','producer_code_sha256'}
                    and Path(descriptor['directory']).is_absolute()
                    and isinstance(descriptor['producer_code_sha256'],dict) and descriptor['producer_code_sha256']
                    and all(continuous.discrete.valid_hash(v)for v in descriptor['producer_code_sha256'].values()),
                    'Exact original producer directory/manifest/code pins required')
        directory=Path(selection['directory']).resolve()
        manifest=shared.load_json(pin(directory/'manifest.json',selection['manifest_sha256'],limits['max_metadata_bytes']))
        require(manifest['schema']==material.PARTIAL and manifest['stage']=='COMPLETE_SLICE'
                and manifest['request']==self.task and manifest['contract_sha256']==r['source_contract_sha256']
                and manifest['code_sha256']==selection['producer_code_sha256']
                and manifest['source_intervals']==[[chrom,*v]for v in self.task['intervals']]
                and manifest['sample_ids_sha256']==material.evidence.sample_hash([p['sample_id']for p in self.persons])
                and manifest['fit_ids_sha256']==material.evidence.sample_hash(sorted(self.persons[i]['sample_id']for i in self.roles['FIT']))
                and manifest['selection_used']=='FIT_ONLY' and manifest['evaluated_outcomes_for_selection']is False
                and manifest['roles_state']=='EXPLICITLY_FROZEN','Common selection source/territory/roles differ')
        marker_path=pin(directory/'common_markers.tsv',manifest['outputs_sha256']['common_markers.tsv'],limits['max_metadata_bytes'])
        audit=shared.load_json(pin(directory/'selection.json',manifest['outputs_sha256']['selection.json'],limits['max_metadata_bytes']))
        require(len(audit)==len(self.task['intervals']),'Common selection interval count differs')
        self.selected={chrom:{}};previous=0
        for row in shared.table(marker_path,['variant_id','chromosome','position_bp','fit_alt_frequency']):
            pos=int(row['position_bp']);key=row['variant_id'].split(':');freq=float(row['fit_alt_frequency'])
            require(len(key)==4 and row['chromosome']==key[0]==chrom and key[1]==str(pos)
                    and key[2] in 'ACGT' and key[3] in 'ACGT' and len(key[2])==len(key[3])==1 and key[2]!=key[3]
                    and pos>previous and any(a<=pos<=b for a,b in self.task['intervals'])
                    and np.isfinite(freq) and min(freq,1-freq)>=self.source['common']['min_maf'],
                    'Selected common locus identity/order/allowed interval/FIT frequency differs')
            self.selected[chrom][pos]=dict(chromosome=chrom,position_bp=str(pos),variant_id=row['variant_id'],
                                           ref=key[2],alt=key[3],fit_alt_af=freq)
            previous=pos
        self.n_sites=len(self.selected[chrom])
        require(0<self.n_sites<=limits['max_selected_sites'],'Selected common locus allocation bound exceeded')
        require(self.n_sites==sum(int(v['retained_sites'])for v in audit),
                'Materialized selected common count differs from FIT selection audit')
        directory=Path(events['directory']).resolve()
        event=shared.load_json(pin(directory/'manifest.json',events['manifest_sha256'],limits['max_metadata_bytes']))
        require(event['schema']=='r02_genomic_event_chromosome_v1'
                and event['stage']=='COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING'
                and event['chromosome']==chrom and event['source_contract_sha256']==r['source_contract_sha256']
                and event['code_sha256']==events['producer_code_sha256'] and event['n_people']==len(self.persons)
                and event['sample_ids_sha256']==material.evidence.sample_hash([p['sample_id']for p in self.persons]),
                'Event genotype source/people/chromosome/producer differs')
        positions=np.asarray(sorted(self.selected[chrom]),np.int64);previous=0;total=0;self.chunks=[]
        for desc in event['channels']['common']:
            require(set(desc)=={'path','sha256','n_sites','bytes','start_bp','end_bp'}
                    and previous<desc['start_bp']<=desc['end_bp'] and 0<desc['n_sites']<=limits['max_chunk_sites'],
                    'Common chunk order/schema/site budget differs')
            previous=desc['end_bp']
            if np.searchsorted(positions,desc['start_bp'])==np.searchsorted(positions,desc['end_bp'],side='right'):continue
            rel=Path(desc['path']);path=(directory/rel).resolve()
            require(not rel.is_absolute() and '..'not in rel.parts and path.is_relative_to(directory)
                    and event['outputs_sha256'].get(desc['path'])==desc['sha256'],'Unsafe or unauthenticated event chunk')
            pin(path,desc['sha256'],limits['max_chunk_bytes'])
            require(path.stat().st_size==desc['bytes'],'Common chunk size differs')
            with zipfile.ZipFile(path)as archive:
                members=archive.infolist()
                require(len(members)==len(continuous.EVENT_FIELDS)
                        and {v.filename for v in members}=={k+'.npy'for k in continuous.EVENT_FIELDS}
                        and sum(v.file_size for v in members)*4<=limits['max_chunk_bytes'],
                        'Common chunk decompression/inventory bound differs')
            total+=desc['bytes'];require(total<=limits['max_input_bytes'],'Common genotype input byte bound exceeded')
            self.chunks.append((chrom,path,desc))
        supplement=r['supplement']
        if supplement is not None:
            require(set(supplement)=={'directory','manifest_sha256','producer_code_sha256'}
                    and Path(supplement['directory']).is_absolute(),'Explicit pinned common supplement required')
            directory=Path(supplement['directory']).resolve()
            sm=shared.load_json(pin(directory/'manifest.json',supplement['manifest_sha256'],limits['max_metadata_bytes']))
            require(sm['schema']=='r02_structure_common_supplement_v1' and sm['stage']=='EXACT_SELECTED_COMMON_LOCI_NO_RESELECTION'
                    and sm['source_contract_sha256']==r['source_contract_sha256'] and sm['chromosome']==chrom
                    and sm['sample_ids_sha256']==material.evidence.sample_hash([p['sample_id']for p in self.persons])
                    and sm['n_people']==len(self.persons) and sm['original_source_sha256']==self.source['sources'][chrom]['common_vcf']['sha256']
                    and sm['implementation_sha256']==supplement['producer_code_sha256']
                    and sm['counted_allele']=='ALT' and sm['position_cm_policy']=='UNAVAILABLE_NAN_NOT_USED_FOR_COMMON_SVD'
                    and sm['target_outcomes_read']is False and sm['selection_changed']is False,
                    'Supplement source/alleles/people/provenance differs')
            desc=sm['array'];require(set(desc)=={'path','sha256','bytes','n_sites','start_bp','end_bp'}
                and desc['path']=='supplement.npz' and 0<desc['n_sites']<=limits['max_chunk_sites'],'Supplement chunk descriptor differs')
            require(len(sm['position_bp'])==desc['n_sites'] and sm['position_bp']==sorted(set(sm['position_bp']))
                    and all(type(v)is int and v>0 for v in sm['position_bp'])
                    and sm['position_bp'][0]==desc['start_bp'] and sm['position_bp'][-1]==desc['end_bp'],
                    'Supplement locus coordinate inventory differs')
            path=pin(directory/desc['path'],desc['sha256'],limits['max_chunk_bytes'])
            require(path.stat().st_size==desc['bytes'],'Supplement chunk byte count differs')
            with zipfile.ZipFile(path)as archive:
                entries=archive.infolist()
                require(len(entries)==len(continuous.EVENT_FIELDS)
                        and {v.filename for v in entries}=={k+'.npy'for k in continuous.EVENT_FIELDS}
                        and sum(v.file_size for v in entries)*4<=limits['max_chunk_bytes'],'Supplement decompression/inventory bound differs')
            total+=desc['bytes'];require(total<=limits['max_input_bytes'],'Supplemented common input byte bound exceeded')
            if any(p in self.selected[chrom]for p in sm['position_bp']):
                self.chunks.append((chrom,path,desc))
        require(self.chunks,'No genotype chunks cover the selected common loci')

    def verify(self):
        require(self.r['implementation_sha256']==implementation()
                and all(shared.sha(path)==digest for path,digest in self.pins.items()),'Common input/source changed')

    def accumulate(self,output):
        limits=self.r['resources']
        accumulator=continuous.DualContinuousAccumulator(self.persons,weighting=self.r['weighting'],
            max_chunk_sites=limits['max_chunk_sites'],max_memory_bytes=limits['max_memory_bytes']-limits['max_chunk_bytes'])
        continuous.accumulate_selected_chunks(self.persons,self.roles['FIT'],self.selected,self.chunks,accumulator,
            min_fit_called=max(self.source['common']['min_fit_called'],
                               int(np.ceil(len(self.roles['FIT'])*(1-self.source['common']['max_fit_missing'])))))
        self.verify()
        record=accumulator.chromosome_statistics(self.task['chromosome'],self.digest)
        return continuous.save_chromosome_statistics(output,record,max_output_bytes=limits['max_output_bytes'],
                                                     min_free_bytes=limits['min_free_bytes'])


def merge(bundle,request,digest,output):
    path=continuous.pinned(Path(request).resolve(),digest,2**20);r=shared.load_json(path)
    require(set(r)=={'schema','source_contract_sha256','implementation_sha256','block_id','exclusion_scope',
                    'weighting','partials','resources'} and r['schema']==MERGE_SCHEMA
            and r['implementation_sha256']==implementation(),'Exact target common merge specification required')
    limits=r['resources']
    require(set(limits)=={'max_input_bytes','max_memory_bytes','max_output_bytes','min_free_bytes'}
            and all(type(v)is int and v>0 for v in limits.values()),'Explicit merge byte bounds required')
    continuous.pinned(Path(bundle).resolve()/'materialize.json',r['source_contract_sha256'],2**20)
    source,persons,roles,blocks,regions=material.load_context(bundle)
    _,scoped,scope=material.assembly_scope(source,blocks,regions,r['block_id'],r['exclusion_scope'])
    tasks,membership=material.make_tasks(source,scoped)
    require(set(r['partials'])==set(membership[r['block_id']]['features']),'Target common partial coverage differs')
    inventories={};expected={};total=0
    for task_id,d in r['partials'].items():
        require(set(d)=={'request','request_sha256','directory','manifest_sha256'},'Exact common cache descriptor required')
        old,*_=request_metadata(bundle,d['request'],d['request_sha256'])
        require(old['task_id']==task_id and old['source_contract_sha256']==r['source_contract_sha256']
                and old['weighting']==r['weighting'],'Common partial source/weight/territory differs')
        chrom=tasks[task_id]['chromosome'];require(chrom not in inventories,'Overlapping chromosome feature partials')
        manifest=continuous.pinned(Path(d['directory']).resolve()/'manifest.json',d['manifest_sha256'],2**20)
        m=shared.load_json(manifest);total+=m['arrays_bytes']
        require(total<=limits['max_input_bytes'],'Merged common cache input byte bound exceeded')
        inventories[chrom]=d;expected[chrom]=d['request_sha256']
    included=sorted(inventories,key=int)
    records=(continuous.load_chromosome_statistics(inventories[c]['directory'],inventories[c]['manifest_sha256'],
              max_input_bytes=limits['max_input_bytes'])for c in included)
    result=continuous.merge_chromosome_statistics(persons,records,weighting=r['weighting'],included_chromosomes=included,
                expected_sources=expected,max_memory_bytes=limits['max_memory_bytes'])
    output=Path(output);require(not output.exists(),'Preserve existing common target output')
    require(shutil.disk_usage(output.parent).free>=limits['min_free_bytes']+limits['max_output_bytes'],'Common merge disk reserve insufficient')
    output.mkdir(mode=0o700);arrays=output/'continuous.private.npz'
    np.savez(arrays,**{k:v for k,v in result.items()if isinstance(v,np.ndarray)})
    require(arrays.stat().st_size<=limits['max_output_bytes'],'Common merge output byte bound exceeded')
    require(shared.sha(path)==digest and r['implementation_sha256']==implementation(),'Common merge request/code changed')
    metadata=dict(schema=MERGE_SCHEMA,stage='FULL_NUMERICAL_COMMON_SPECTRUM_NOT_PREDICTIVE_SELECTION',
        request_sha256=digest,source_contract_sha256=r['source_contract_sha256'],implementation_sha256=implementation(),
        block_id=r['block_id'],exclusion_scope=scope,allowed_input_intervals=scoped[0]['allowed_input_intervals'],
        partials=r['partials'],n_sites=result['n_sites'],numerical_rank=len(result['eigenvalues']),
        numerical_tolerance=result['numerical_tolerance'],weighting=r['weighting'],targets_read=False,
        outputs_sha256={arrays.name:shared.sha(arrays)},biological_validation=False,
        common_information='EXACT_MATERIALIZED_FIT_SELECTED_LOCI_WITH_FIT_MEAN_IMPUTATION')
    material.write_json(output/'manifest.json',metadata);return metadata


def plan_merges(bundle,request,digest,partials,output):
    """Freeze target membership after reusable statistics complete; no arrays read."""
    path=continuous.pinned(Path(request).resolve(),digest,2**20);r=shared.load_json(path)
    require(set(r)=={'schema','bundle','implementation_sha256','partials','merges'}
            and r['schema']==BATCH_SCHEMA and r['implementation_sha256']==implementation(),
            'Exact common batch specification required')
    require(r['bundle']==dict(path=str(Path(bundle).resolve()),sha256=shared.sha(Path(bundle)/'materialize.json')),
            'Common batch source differs')
    source,persons,roles,blocks,regions=material.load_context(bundle)
    by_source={}
    for directory in partials:
        directory=Path(directory).resolve();manifest=shared.load_json(directory/'manifest.json')
        source_sha=manifest['source_sha256']
        require(source_sha not in by_source and manifest['schema']=='r02_common_gram_chromosome_v1',
                'Duplicate or wrong common cache')
        require(shared.sha(directory/manifest['arrays_file'])==manifest['arrays_sha256'],
                'Common cache changed before merge planning')
        by_source[source_sha]=dict(directory=str(directory),manifest_sha256=shared.sha(directory/'manifest.json'))
    descriptors={}
    for d in r['partials']:
        require(set(d)=={'request','request_sha256'},'Exact partial request descriptor required')
        old,*_=request_metadata(bundle,d['request'],d['request_sha256'])
        require(d['request_sha256']in by_source and old['task_id']not in descriptors,'Missing/duplicate common partial')
        descriptors[old['task_id']]={**d,**by_source[d['request_sha256']]}
    require(len(descriptors)==len(by_source),'Unexpected common cache supplied')
    output=Path(output);require(not output.exists(),'Preserve existing common merge plan');output.mkdir(mode=0o700)
    catalogue=[];seen=set();used=set()
    for target in r['merges']:
        require(set(target)=={'block_id','exclusion_scope','weighting','resources'},'Exact target merge specification required')
        _,scoped,scope=material.assembly_scope(source,blocks,regions,target['block_id'],target['exclusion_scope'])
        _,membership=material.make_tasks(source,scoped)
        needed=set(membership[target['block_id']]['features'])
        require(needed<=descriptors.keys(),'Incomplete common target territory')
        name=target['block_id']+'__'+scope
        require(name not in seen,'Repeated common target');seen.add(name);used.update(needed)
        record=dict(schema=MERGE_SCHEMA,source_contract_sha256=r['bundle']['sha256'],implementation_sha256=implementation(),
                    **target,partials={k:descriptors[k]for k in sorted(needed)})
        target_path=output/(name+'.json');material.write_json(target_path,record)
        catalogue.append(dict(name=name,request=str(target_path.resolve()),request_sha256=shared.sha(target_path),
                              directories=[descriptors[k]['directory']for k in sorted(needed)]))
    require(used==descriptors.keys() and catalogue,'Unused common partials or empty targets')
    require(shared.sha(path)==digest and r['implementation_sha256']==implementation(),'Common batch source changed')
    result=dict(schema=BATCH_SCHEMA,batch_sha256=digest,targets=catalogue,arrays_read=False)
    material.write_json(output/'manifest.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--stage',choices=['partial','merge','plan-merges'],required=True)
    p.add_argument('--partials',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='partial':result={'manifest_sha256':MaterializedCommonInput(a.bundle,a.request,a.request_sha256).accumulate(a.output)}
    elif a.stage=='merge':result=merge(a.bundle,a.request,a.request_sha256,a.output)
    else:result=plan_merges(a.bundle,a.request,a.request_sha256,a.partials,a.output)
    print(json.dumps(result))


if __name__=='__main__':main()
