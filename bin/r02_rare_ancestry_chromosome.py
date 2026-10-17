#!/usr/bin/env python3
"""Aggregate conditional rare-copy identification over a whole frozen chromosome.

All source rare-catalogue loci are retained, including loci outside ancestry
windows. These contribute explicit missing-ancestry counts, never silent zeros.
This describes identifiability; it is not a fine-structure association test.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import tarfile
import tempfile
import time
import numpy as np
import r02_rare_ancestry_identifiability as core

SCHEMA='r02_rare_ancestry_chromosome_input_v1'
CODE_FILES=(*core.CODE_FILES,'r02_rare_ancestry_chromosome.py')
require=core.require


def merge_counts(total,block):
    for dst,src in zip(total,block):
        # Counter.update adds values, including explicitly present zeros.
        dst.update(src)


def render_counts(counters):
    counts,by_dose,joint,ancestry_copies,lower,upper=counters
    return dict(counts=[dict(ancestry=a,status=s,person_locus_ancestry_rows=n)for(a,s),n in sorted(counts.items())],
      counts_by_observed_dosage=[dict(ancestry=a,observed_minor_dosage=d,status=s,person_locus_ancestry_rows=n)for(a,d,s),n in sorted(by_dose.items())],
      joint_counts=[dict(observed_minor_dosage=d,status=s,person_locus_rows=n)for(d,s),n in sorted(joint.items())],
      jointly_observed_conditional_ancestry_copies=dict(ancestry_copies),
      marginal_copy_sum_lower=dict(lower),marginal_copy_sum_upper=dict(upper))


def posterior_bounds(data, starts, ends, posterior, tolerance):
    """Outer expected-copy bounds, summed by person and constant MSP window.

    This is algebraically the locus-wise sum, avoiding a 3D posterior copy for
    every noncarrier. No phase or homolog independence distribution is assumed.
    """
    d=data['dosage'];observed=data['observed'];position=data['position_bp']
    index=np.searchsorted(starts,position,side='right')-1
    covered=(index>=0)&(position<=ends[np.maximum(index,0)])
    index=np.where(covered,index,-1)
    coverage=[dict(observed_minor_dosage=str(dose),all_observed_person_locus=0,
        outside_msp=0,missing_fb_inside_msp=0,two_fb_marginals_available=0)for dose in range(3)]
    bounds=[dict(ancestry=a,observed_minor_dosage=str(dose),person_locus_rows=0,
        expected_copy_sum_lower=0.0,expected_copy_sum_upper=0.0)
        for dose in range(3)for a in core.ancestry.POPULATIONS]
    mass_max=None
    for window in np.unique(index):
        columns=np.flatnonzero(index==window)
        calls=d[:,columns];mask=observed[:,columns]
        counts=[np.count_nonzero(mask&(calls==dose),axis=1)for dose in range(3)]
        if window<0:
            for dose,n in enumerate(counts):
                value=int(n.sum());coverage[dose]['all_observed_person_locus']+=value;coverage[dose]['outside_msp']+=value
            continue
        p0=np.asarray(posterior[window,0::2,:]);p1=np.asarray(posterior[window,1::2,:])
        available=np.isfinite(p0).all(axis=1)&np.isfinite(p1).all(axis=1)
        usable=available&((counts[0]+counts[1]+counts[2])>0)
        require(np.all(np.abs(p0[usable].sum(axis=1)-1)<=tolerance) and
                np.all(np.abs(p1[usable].sum(axis=1)-1)<=tolerance),'FB marginal mass differs')
        total=p0+p1;lo=np.maximum(0,total-1);hi=np.minimum(1,total)
        one=available&(counts[1]>0)
        require(np.all(lo[one].sum(axis=1)<=1+2*tolerance+1e-12) and
                np.all(hi[one].sum(axis=1)>=1-2*tolerance-1e-12),'Posterior bounds fail total-copy consistency')
        mass=np.abs(total[usable].sum(axis=1)-2)
        if len(mass):mass_max=max(mass_max or 0,float(mass.max()))
        for dose,n in enumerate(counts):
            all_count=int(n.sum());valid_count=int(n[available].sum())
            coverage[dose]['all_observed_person_locus']+=all_count
            coverage[dose]['missing_fb_inside_msp']+=all_count-valid_count
            coverage[dose]['two_fb_marginals_available']+=valid_count
            for a in range(3):
                row=bounds[dose*3+a];row['person_locus_rows']+=valid_count
                if dose:
                    lower=lo[:,a]if dose==1 else total[:,a]
                    upper=hi[:,a]if dose==1 else total[:,a]
                    row['expected_copy_sum_lower']+=float(np.dot(n[available],lower[available]))
                    row['expected_copy_sum_upper']+=float(np.dot(n[available],upper[available]))
    return dict(coverage_by_dosage=coverage,genotype_missing=int(np.count_nonzero(~observed)),
                bounds_by_dosage=bounds,posterior_two_homolog_mass_residual_max=mass_max)


def merge_posterior(total, block):
    total['genotype_missing']+=block['genotype_missing']
    for key in ('coverage_by_dosage','bounds_by_dosage'):
        if not total[key]:total[key]=block[key]
        else:
            for left,right in zip(total[key],block[key]):
                for field,value in right.items():
                    if field not in {'ancestry','observed_minor_dosage'}:left[field]+=value
    m=block['posterior_two_homolog_mass_residual_max']
    if m is not None:total['posterior_two_homolog_mass_residual_max']=max(total['posterior_two_homolog_mass_residual_max']or 0,m)


def run(path,digest,inputs_dir,output,scratch):
    started=time.monotonic();path=Path(path).resolve();output=Path(output);scratch=Path(scratch)
    require(core.sha256(path)==digest,'Chromosome specification SHA256 differs')
    c=json.loads(path.read_text());require(c['schema']==SCHEMA,'Unexpected chromosome schema')
    expected=(set(core.FILES)-{'event_chunk'})|{'event_archive'}
    require(set(c['inputs'])==expected,'Exact complete-chromosome input inventory required')
    limits=c['resources'];require(set(limits)=={'max_array_mb','max_window_matrix_mb','min_free_mb','max_archive_bytes','timeout_seconds'}and all(type(v)is int and v>0 for v in limits.values()),'Explicit positive resource limits required')
    paths={key:core.authenticate(desc,inputs_dir)for key,desc in c['inputs'].items()}
    require(paths['event_archive'].stat().st_size<=limits['max_archive_bytes'],'Archive exceeds bound')
    code=[Path(__file__).with_name(n)for n in CODE_FILES]
    before=core.input_snapshot([path,*paths.values(),*code])
    anc,event,material,samples,mapping=core.validate_sources(paths)
    chrom=str(anc['chromosome']);prefix=c['archive_member_prefix']
    require(prefix=='published/chr'+chrom,'Exact scientific product prefix required')
    entries=event['channels']['rare'];require(entries,'Nonempty rare catalogue required')
    lookup={prefix+'/'+Path(r['path']).name:r for r in entries}
    require(len(lookup)==len(entries),'Duplicate rare chunk identities')
    previous=0
    for e in entries:
        require(type(e['n_sites'])is int and e['n_sites']>0 and e['start_bp']>previous and e['end_bp']>=e['start_bp'],'Source chunk order/overlap invalid')
        previous=e['end_bp']
    require(scratch.is_dir() and not output.exists(),'Fresh output and existing local scratch required')
    core.ancestry.reserve(scratch,limits['min_free_mb']);output.mkdir(parents=True,mode=0o700)
    counters=tuple(Counter()for _ in range(6));seen=set();coordinate_covered=0;loci=0
    soft=dict(genotype_missing=0,coverage_by_dosage=[],bounds_by_dosage=[],posterior_two_homolog_mass_residual_max=None)
    with tempfile.TemporaryDirectory(prefix='k04-chromosome-',dir=scratch)as td:
        td=Path(td)
        windows,hard,posterior,window_receipt=core.ancestry.prepare_windows(paths,anc,samples,mapping,td,limits['max_window_matrix_mb'],limits['min_free_mb'])
        starts=np.asarray([w[0]for w in windows]);ends=np.asarray([w[1]for w in windows]);require(len(starts)>0,'Ancestry windows absent')
        # First sequential pass authenticates exact locus coverage over the
        # COMPLETE catalogue before loading any dosage array. A second stream
        # is deliberate: bounded disk use and no selection by observed biology.
        preflight={}
        with tarfile.open(paths['event_archive'],'r|gz') as archive:
            for member in archive:
                if member.name not in lookup:continue
                require(member.isfile() and member.name not in preflight,'Repeated or nonregular preflight chunk')
                e=lookup[member.name];require(member.size==e['bytes'],'Preflight chunk size differs')
                target=td/'coordinates.npz';checksum=hashlib.sha256()
                with archive.extractfile(member) as source,target.open('wb') as dest:
                    while block:=source.read(2**20):checksum.update(block);dest.write(block)
                require(checksum.hexdigest()==e['sha256'],'Preflight chunk SHA256 differs')
                preflight[member.name]=core.coordinate_preflight(target,paths['msp'],chrom,e,limits['max_array_mb'],require_overlap=False,intervals=windows)
                require(time.monotonic()-started<=limits['timeout_seconds'],'Coordinate preflight timeout exceeded')
        require(set(preflight)==set(lookup),'Incomplete archive: rare catalogue chunks missing before genotype read')
        require(sum(v['loci_in_msp'] for v in preflight.values())>0,'NO_COORDINATE_OVERLAP: chromosome has no joint territory; no genotype arrays read')
        # Streaming tar avoids quadratic seek/decompression across thousands of
        # chunks. Non-scientific members are skipped rather than extracted.
        with (output/'chunks.jsonl').open('x')as records,tarfile.open(paths['event_archive'],'r|gz')as archive:
            for member in archive:
                if member.name not in lookup:continue
                require(member.isfile() and member.name not in seen,'Repeated or nonregular rare chunk')
                e=lookup[member.name];require(member.size==e['bytes'],'Chunk byte count differs')
                target=td/'chunk.npz';checksum=hashlib.sha256()
                with archive.extractfile(member)as source,target.open('wb')as dest:
                    while block:=source.read(2**20):checksum.update(block);dest.write(block)
                require(checksum.hexdigest()==e['sha256'],'Chunk SHA256 differs')
                data=core.load_chunk(target,len(samples),e,material['source_rare_rule'],material['source_n_samples'],limits['max_array_mb'])
                block=core.aggregate_chunk(data,starts,ends,hard);merge_counts(counters,block)
                soft_block=posterior_bounds(data,starts,ends,posterior,anc['posterior_sum_tolerance']);merge_posterior(soft,soft_block)
                idx=np.searchsorted(starts,data['position_bp'],side='right')-1
                ncovered=int(np.count_nonzero((idx>=0)&(data['position_bp']<=ends[np.maximum(idx,0)])))
                require(ncovered==preflight[member.name]['loci_in_msp'],'Coordinate preflight changed')
                coordinate_covered+=ncovered;loci+=e['n_sites'];seen.add(member.name)
                records.write(json.dumps(dict(chunk=Path(member.name).name,start_bp=e['start_bp'],end_bp=e['end_bp'],n_loci=e['n_sites'],loci_in_msp=ncovered,posterior_bounds=soft_block,**render_counts(block)),sort_keys=True)+'\n');records.flush()
                require(time.monotonic()-started<=limits['timeout_seconds'],'Chromosome timeout exceeded')
                core.ancestry.reserve(output,limits['min_free_mb'])
        require(seen==set(lookup),'Incomplete archive: rare catalogue chunks missing')
    require(loci==sum(e['n_sites']for e in entries),'Whole-catalogue locus total differs')
    require(sum(counters[0].values())==len(samples)*loci*3 and sum(counters[2].values())==len(samples)*loci,'Aggregate denominators differ')
    usable=sum(n for(_,s),n in counters[0].items()if s in('POINT_CONDITIONAL','AMBIGUOUS_CONDITIONAL'))
    result=dict(schema='r02_rare_ancestry_chromosome_v1',implementation_version=core.IMPLEMENTATION_VERSION,
      status='COMPLETE_CONDITIONAL_IDENTIFICATION_NOT_FINE_STRUCTURE_VALIDATION'if usable else'NO_USABLE_JOINT_GENOTYPE_ANCESTRY_SUPPORT',
      chromosome=chrom,n_people=len(samples),source_n_people=material['source_n_samples'],n_loci=loci,n_chunks=len(entries),loci_in_msp=coordinate_covered,loci_outside_msp=loci-coordinate_covered,
      **render_counts(counters),posterior_bounds=soft,
      posterior_interpretation='OUTER_EXPECTED_COPY_BOUNDS_FROM_WINDOW_GNOMIX_MARGINALS_NO_JOINT_OR_PHASE_ASSUMPTION_NOT_CONFIDENCE_INTERVALS',
      posterior_normalized=False,posterior_mass_tolerance=anc['posterior_sum_tolerance'],
      omitted_ancestry='EAS_NOT_MODELLED_AND_NOT_COVERED_BY_THESE_BOUNDS',scope='ALL_AUTHENTICATED_SOURCE_RARE_CATALOGUE_LOCI_ON_THIS_CHROMOSOME',
      denominator='SOURCE_RARE_CATALOGUE_NOT_GENOMIC_CALLABILITY; EACH_PERSON_LOCUS_COUNTED_ONCE_IN_JOINT_COUNTS',
      bounds='MARGINAL_COPY_BOUNDS_CONDITIONAL_ON_GNOMIX_HARDCALLS_NOT_CONFIDENCE_INTERVALS',
      ancestry_is='GNOMIX_ESTIMATED_AFR_EUR_NAM_NO_EAS_CATEGORY',phase_accuracy='NOT_VALIDATED',
      no_rephasing=True,no_imputation=True,source_rare_rule=material['source_rare_rule'],windows=window_receipt,
      seconds=time.monotonic()-started,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    core.verify_unchanged([path,*paths.values(),*code],before)
    manifest=dict(schema='r02_rare_ancestry_chromosome_v1',status=result['status'],contract_sha256=digest,sources=before,resources=limits,outputs_sha256={p.name:core.sha256(p)for p in output.iterdir()if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('contract','contract-sha256','inputs-dir','output-dir','scratch'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(run(a.contract,a.contract_sha256,a.inputs_dir,a.output_dir,a.scratch)))
if __name__=='__main__':main()
