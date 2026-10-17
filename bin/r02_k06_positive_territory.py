#!/usr/bin/env python3
"""Map all identified simulated copy overlaps to cM, retaining unknown territory.

Prepares positive inputs for detector calibration. It does not label different
donors as negative or interpret copied segments as a population genealogy.
"""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import tempfile

import r02_k06_observability as provenance
from r02_segment_evidence import GeneticMap

require = provenance.require
sha = provenance.sha


def run(request_path, request_sha, output):
    request_path=Path(request_path)
    require(sha(request_path)==request_sha,'Request changed')
    request=json.loads(request_path.read_text())
    require(request['schema']=='r02_k06_positive_territory_request_v1'
            and request['genome_build']=='GRCh38'
            and request['coordinate_system']=='1-based-inclusive', 'Explicit reference required')
    summary_path=provenance.desc(request['summary'],'copy summary')
    overlap_path=provenance.desc(request['overlaps'],'copy overlaps')
    map_path=provenance.desc(request['map'],'genetic map')
    summary=json.loads(summary_path.read_text())
    require(summary['schema']=='r02_identified_donor_copy_truth_v1'
            and summary['job_id']==request['job_id']
            and summary['negative_truth_available'] is False
            and summary['outputs_sha256']['identified_copy_overlaps.private.tsv.gz']==sha(overlap_path),
            'Authenticated conditional copy truth required')
    chrom=request['chromosome']
    require(summary['chromosomes']==[chrom], 'Chromosome differs')
    lo,hi,knots,_=provenance.map_range(map_path,chrom)
    output=Path(output);require(not output.exists(),'Output already exists')
    output.mkdir(parents=True)
    # Adapt the source format; interpolation remains the established implementation.
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        canonical=Path(temporary)/'map.tsv'
        with canonical.open('w') as dst,provenance.open_text(map_path) as src:
            dst.write('chrom\tpos\tcm\n')
            for line in src:
                fields=line.split();dst.write('\t'.join(fields[:3])+'\n')
        genetic=GeneticMap(canonical,chrom,knots)
    counts=Counter();sums=Counter();people=set();pairs=set();rows=0
    target=output/'positive-copy-territory.private.tsv.gz'
    with gzip.open(overlap_path,'rt',newline='') as src,gzip.open(target,'wt',newline='') as dst:
        reader=csv.DictReader(src,delimiter='\t');fields=reader.fieldnames
        expected=['chromosome','sample_a','haplotype_a','sample_b','haplotype_b',
                  'start_bp_inclusive','end_bp_inclusive','donor_id','donor_copy','relationship']
        require(fields==expected,'Copy overlap columns differ')
        writer=csv.DictWriter(dst,fieldnames=fields+['map_support','mapped_start_bp',
            'mapped_end_bp','mapped_length_cm'],delimiter='\t',lineterminator='\n')
        writer.writeheader()
        for row in reader:
            rows+=1;require(rows<=request['max_rows'],'Input exceeds admitted row budget')
            require(row['chromosome']==chrom,'Mixed chromosome')
            start,end=int(row['start_bp_inclusive']),int(row['end_bp_inclusive'])
            require(0<start<=end and row['relationship'] in
                    {'BETWEEN_PERSON_COPY_MATCH','WITHIN_PERSON_COPY_MATCH'},'Malformed positive interval')
            a,b=row['sample_a'],row['sample_b'];ha,hb=int(row['haplotype_a']),int(row['haplotype_b'])
            require(ha in (1,2) and hb in (1,2) and (a,ha)<(b,hb),'Noncanonical haplotype pair')
            require((a==b)==(row['relationship']=='WITHIN_PERSON_COPY_MATCH'),'Relationship differs')
            people.update((a,b));pairs.add((a,b))
            left,right=max(start,lo),min(end,hi)
            status='OUTSIDE_MAP' if left>right else 'FULL_MAP' if (left,right)==(start,end) else 'CLIPPED_TO_MAP'
            length=None if left>right else genetic.length(left,right)
            counts[row['relationship']+':'+status]+=1
            if length is not None:sums[row['relationship']]+=length
            writer.writerow(dict(row,map_support=status,mapped_start_bp='NA' if length is None else left,
                mapped_end_bp='NA' if length is None else right,mapped_length_cm='NA' if length is None else repr(length)))
    require(rows==sum(summary['overlap_rows'].values()),'Incomplete copy source')
    require(sha(request_path)==request_sha and sha(summary_path)==request['summary']['sha256']
            and sha(overlap_path)==request['overlaps']['sha256'] and sha(map_path)==request['map']['sha256'],
            'Input changed during preparation')
    result=dict(schema='r02_k06_positive_territory_v1',job_id=request['job_id'],chromosome=chrom,
        status='POSITIVE_TERRITORY_PREPARED_NOT_DETECTOR_CALIBRATION',request_sha256=request_sha,
        source_summary_sha256=sha(summary_path),genetic_map_sha256=sha(map_path),rows=rows,
        observed_people=len(people),observed_person_pairs=len(pairs),counts=dict(counts),
        sum_copy_pair_cm_not_independent_segments=dict(sums),
        outputs_sha256={target.name:sha(target)},negative_truth_available=False,
        detector_calibration_completed=False,demographic_inference=False,
        limits=['Different donor copies are unknown, not genealogical negatives.',
                'Clipped rows cannot be treated as complete segment lengths.',
                'Rows and repeated mosaics are not independent demographic replicates.',
                'Adjacent donor-copy rows are not merged into genealogical IBD segments.'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','request-sha256','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=run(a.request,a.request_sha256,a.output)
    print(json.dumps({k:r[k] for k in ('status','job_id','rows')}))
