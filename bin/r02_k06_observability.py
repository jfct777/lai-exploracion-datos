#!/usr/bin/env python3
"""Validate K06 reference/map/person/segment observability and calibration inputs."""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json, math
from collections import Counter
from pathlib import Path


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def desc(x, name):
    require(set(x)=={'path','sha256'}, 'Exact descriptor required: '+name)
    p=Path(x['path']); require(p.is_absolute() and p.is_file() and sha(p)==x['sha256'], 'Input differs: '+name)
    return p


def open_text(p): return gzip.open(p,'rt') if p.suffix=='.gz' else p.open()


def map_range(path, chrom):
    first=last=None; previous_bp=0; previous_cm=-math.inf; n=0
    with open_text(path) as f:
        for line in f:
            fields=line.split(); require(len(fields)>=3 and fields[0].removeprefix('chr')==str(chrom),'Map schema/chromosome differs')
            bp,cm=int(fields[1]),float(fields[2]); require(bp>previous_bp and math.isfinite(cm) and cm>=previous_cm,'Map is unordered')
            first=bp if first is None else first; last=bp; previous_bp,previous_cm=bp,cm; n+=1
    require(n>1,'Map needs at least two points')
    return first,last,n,previous_cm


def roster(path):
    with path.open(newline='') as f:
        reader=csv.DictReader(f,delimiter='\t'); require('sample_id' in (reader.fieldnames or []),'sample_id missing')
        ids=[r['sample_id'] for r in reader]
    require(ids and len(ids)==len(set(ids)),'Empty or duplicate people'); return set(ids)


def interval_audit(path, chrom, people, lo, hi):
    counts=Counter(rows=0, people_outside=0, wholly_in_map=0, clipped_to_map=0, outside_map=0)
    observed_people=set(); pairs=set(); bp=0
    with open_text(path) as f:
        reader=csv.DictReader(f,delimiter='\t')
        require({'chrom','sample_a','sample_b','start_pos','end_pos'} <= set(reader.fieldnames or []),'Segment fields missing')
        for r in reader:
            require(r['chrom'].removeprefix('chr')==str(chrom),'Segment chromosome differs')
            a,b=r['sample_a'],r['sample_b']; x,y=int(r['start_pos']),int(r['end_pos'])
            require(0<x<=y and a!=b,'Invalid segment')
            counts['rows']+=1
            if a not in people or b not in people: counts['people_outside']+=1; continue
            observed_people.update((a,b)); pairs.add(tuple(sorted((a,b))))
            left,right=max(x,lo),min(y,hi)
            if left>right: counts['outside_map']+=1
            else:
                bp+=right-left+1
                counts['wholly_in_map' if x>=lo and y<=hi else 'clipped_to_map']+=1
    return dict(counts, observed_people=len(observed_people), observed_pairs=len(pairs),
                segment_bp_sum_within_map_not_union=bp)


def validate_calibration_table(path, people, kind):
    expected=['chrom','sample_a','sample_b','start_bp','end_bp','length_cm']
    count=0
    with open_text(path) as f:
        reader=csv.DictReader(f,delimiter='\t'); require(list(reader.fieldnames or [])==expected,kind+' columns differ')
        for r in reader:
            require(r['sample_a']<r['sample_b'] and {r['sample_a'],r['sample_b']}<=people,'Calibration pair/person differs')
            require(1<=int(r['chrom'])<=22 and 0<int(r['start_bp'])<=int(r['end_bp']) and float(r['length_cm'])>=0, 'Invalid calibration interval')
            count+=1
    return count


def run(contract_path, expected_sha, out):
    require(sha(contract_path)==expected_sha,'Contract changed')
    c=json.loads(Path(contract_path).read_text()); require(c.get('schema')=='r02_k06_observability_v1','Unsupported K06 contract')
    require(c.get('genome_build')=='GRCh38' and c.get('coordinate_system')=='1-based-inclusive','Explicit GRCh38 inclusive coordinates required')
    people_path=desc(c['people'],'people'); people=roster(people_path)
    require(set(c['chromosomes'])=={str(i) for i in range(1,23)},'Exactly 22 autosomes required')
    reports=[]; totals=Counter()
    for chrom in map(str,range(1,23)):
        item=c['chromosomes'][chrom]
        mp=desc(item['genetic_map'],'map chr'+chrom); lo,hi,nmap,lastcm=map_range(mp,chrom)
        contract=desc(item['segments_contract'],'segment contract chr'+chrom); sc=json.loads(contract.read_text())
        require(sc['schema']=='r02_segment_ibd_v2' and sc['genome_build']=='GRCh38' and sc['chromosome']==chrom and sc['analytical_samples_sha256']==c['analytical_samples_sha256'],'Segment contract identity differs')
        intervals=desc(item['segments'],'segments chr'+chrom); require(sha(intervals)==sc['intervals']['sha256'],'Segment contract/file differs')
        provenance=desc(item['reference_provenance'],'reference provenance chr'+chrom)
        with provenance.open(newline='') as f:
            rr=csv.DictReader(f,delimiter='\t'); require({'chromosome','position_bp','status'}<=set(rr.fieldnames or []),'Reference provenance columns missing')
            positions=[int(r['position_bp']) for r in rr if r['status'].startswith('EXACT')]
        require(positions,'No reference positions'); ref_in=len({x for x in positions if lo<=x<=hi})
        seg=interval_audit(intervals,chrom,people,lo,hi)
        row=dict(chromosome=chrom,map_points=nmap,map_start_bp=lo,map_end_bp=hi,map_end_cm=lastcm,
                 reference_positions=len(set(positions)),reference_positions_in_map_domain=len(set(x for x in positions if lo<=x<=hi)),**seg)
        reports.append(row)
        totals.update(map_points=nmap, reference_positions=len(set(positions)),
            reference_positions_in_map_domain=ref_in, segment_rows=seg['rows'],
            segments_people_outside=seg['people_outside'], segments_wholly_in_map=seg['wholly_in_map'],
            segments_clipped_to_map=seg['clipped_to_map'], segments_outside_map=seg['outside_map'],
            observed_person_chromosome_units=seg['observed_people'],
            observed_pair_chromosome_units=seg['observed_pairs'],
            segment_bp_sum_within_map_not_union=seg['segment_bp_sum_within_map_not_union'])
    calibration=c.get('calibration')
    if calibration is None:
        calibration_report=dict(status='BLOCKED_TRUTH_DETECTIONS_AND_ADMITTED_MATCHING_POLICY_REQUIRED',metrics_computed=False,
            required_columns=['chrom','sample_a','sample_b','start_bp','end_bp','length_cm'],
            required_policy=['matching_rule','endpoint_tolerance_unit_and_value','length_bins','uncertainty_unit'])
    else:
        require(calibration.get('policy',{}).get('status')=='ADMITTED','Calibration policy must be admitted')
        truth=desc(calibration['truth'],'truth'); detections=desc(calibration['detections'],'detections')
        nt=validate_calibration_table(truth,people,'truth'); nd=validate_calibration_table(detections,people,'detections')
        calibration_report=dict(status='FORMAT_BRIDGE_VALIDATED_METRIC_EXECUTION_SEPARATE',truth_segments=nt,detection_segments=nd,
            policy=calibration['policy'],metrics_computed=False,
            existing_metric_primitives=['m37_trace_score.match_pairs','r02_lai_evaluate.boundary_rates'])
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    result=dict(schema='r02_k06_observability_result_v1',status='REAL_INPUT_OBSERVABILITY_VALIDATED_NO_DEMOGRAPHIC_INFERENCE',
        people=len(people),chromosomes=22,genome_build='GRCh38',coordinate_system='1-based-inclusive',
        totals=dict(totals),chromosome_reports=reports,calibration=calibration_report,
        biological_validation=False,demographic_history_inferred=False,
        limits=['Reported IBD geometry does not define callable negative territory.',
                'Map-domain overlap does not validate segment sensitivity or length error.',
                'Related dyads and segments are not independent replicates.'],
        inputs_sha256={'contract':expected_sha,'people':sha(people_path)})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'calibration-format.tsv').write_text('chrom\tsample_a\tsample_b\tstart_bp\tend_bp\tlength_cm\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--contract',type=Path,required=True);p.add_argument('--contract-sha256',required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.contract,a.contract_sha256,a.output_dir)))

