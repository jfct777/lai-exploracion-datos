#!/usr/bin/env python3
"""L2 private extraction of observed donor copies; no simulation or new split.

Consumes sealed L1 candidate decisions and exactly the SOURCE_VALID projection.
Uses the existing L1 allele mapping, genotype counting and quality rules. Copy
1/2 is a VCF chromosome-local index, NOT a parent, global homologue or validated
phase block. Reserved genotypes are not opened. Missing calls remain unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import itertools
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import r02_lai_material_decision as decision
import r02_lai_genotype_support as gt

require, sha, read, write = decision.require, decision.sha, decision.read, decision.write
SCHEMA = 'r02_lai_donor_haplotypes_v1'
STATUS = 'COMPLETE_PRIVATE_DONOR_HAPLOTYPES_NOT_SIMULATION_READY'
AGG_SCHEMA = 'r02_lai_donor_haplotypes_aggregate_v1'
AGG_STATUS = 'COMPLETE_AUTOSOMAL_PRIVATE_DONOR_PREPARATION_NOT_SIMULATION_READY'
CANDIDATE = 'CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE'
ROLE = 'SOURCE_VALID'
KEY = decision.joint.KEY
ROSTER = ('sample_id', 'role', 'ancestry', 'atomic_unit_id', 'canonical_population')
COUNTS = ('candidate_loci', 'complete_calls', 'missing_calls', 'non_diploid_calls',
          'counted_copies', 'carrier_loci', 'phased_carrier_loci', 'observed_carrier_copies',
          'unphased_carrier_loci', 'quality_evaluable_calls')
FILES = {'donors.private.tsv', 'calls.private.tsv.gz', 'carriages.private.tsv.gz',
         'donor_support.private.tsv', 'loci.tsv.gz', 'summary.json', 'definitions.json'}
AGG_FILES = {'donors.private.tsv', 'donor_support.private.tsv', 'summary.json', 'definitions.json'}
CODE_FILES = ('r02_lai_donor_haplotypes.py', 'r02_lai_material_decision.py',
              'r02_lai_simulation_material.py', 'r02_lai_support_review.py',
              'r02_lai_genotype_support.py', 'r02_lai_allele_support.py',
              'r02_lai_build_verification.py', 'r02_genomic_pair_evidence.py')


def definitions():
    return dict(scope='Existing SOURCE_VALID donors and ALL conditional L1 candidates only; no new split or phasing.',
        genotype_roles_read=[ROLE], source_test_opened=False, reference_genotypes_opened=False,
        quality='GT_ONLY means quality unmeasured. Missing FILTER is allowed only under inherited PASS_OR_UNFILTERED, not certified PASS.',
        counted_allele='Exact DNABR REF or ALT recorded in L1; no strand inference or imputation.',
        copies='Observed phased diploid GT copy 1/2 within this chromosome. No cross-chromosome identity or parental identity.',
        phase_set='Existing FORMAT/PS retained; absent PS is unknown, not evidence of chromosome-wide phase accuracy.',
        phase='Observed phase flags, not validated haplotypes; unphased homozygotes keep dose but receive no phased-copy assignment.',
        zeros='Complete noncarrier GT has zero dose. Missing/partial/non-diploid calls have NA dose and copy.',
        source_exposure='SOURCE_VALID is historical development material, not an untouched evaluation panel.',
        ancestry='Historical donor label from the role table, not an independently verified ancestry of each allele or segment.',
        dependency='Declared atomic_unit_id/population is not a certified family or independent biological replicate.',
        simulation_ready=False, new_donor_split_assigned=False, phase_validated=False,
        remaining_for_simulation=['Document actual donor independence and exposure before any new split.',
          'Specify the simulation demographic design, recombination tool/map and full common-marker haplotypes.',
          'Specify phase-error and genotype-quality sensitivity; this extraction does not certify either.',
          'Implement and qualify the simulation stage separately; no simulation is authorized by this receipt.'])


def writer(path, fields):
    handle = gzip.open(path, 'xt', newline='') if path.name.endswith('.gz') else path.open('x', newline='')
    value = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
    value.writeheader()
    return handle, value


def roster(path):
    selected, seen = {}, set()
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require(set(ROSTER) <= set(reader.fieldnames or []), 'Missing historical donor metadata')
        for row in reader:
            sample = row['sample_id']
            require(sample and sample not in seen, 'Duplicate person across historical roles')
            seen.add(sample)
            if row['role'] == ROLE:
                require(row['ancestry'] and row['ancestry'] != 'ALL', 'Unknown/reserved ancestry label')
                selected[sample] = {k: row[k] for k in ROSTER}
    require(selected, 'No historical donors')
    return selected


def authenticate(contract_path, digest):
    inputs = decision.Inputs(); cp = inputs.add(contract_path, digest); c = read(cp)
    require(set(c) == {'schema', 'chromosome', 'decision_manifest', 'donor_manifest', 'roles', 'resources'}, 'Unexpected L2 contract fields')
    require(c['schema'] == 'r02_lai_donor_haplotypes_input_v1' and c['chromosome'] in {str(i) for i in range(1,23)}, 'Invalid chromosome contract')
    resources = c['resources']
    required = {'cpus', 'max_panel_bytes', 'max_panel_records', 'max_candidate_rows', 'max_private_rows',
                'max_output_bytes', 'min_free_gib', 'timeout_seconds'}
    require(set(resources) == required and resources['cpus'] == 1, 'Explicit single-reader resource contract required')
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= (0 if k == 'min_free_gib' else 1)
                for k,v in resources.items()), 'Invalid resource bounds')
    require(all(isinstance(resources[k], int) for k in required-{'min_free_gib'}), 'Integer resource bounds required')
    dp = inputs.descriptor(c['decision_manifest']); d = read(dp)
    require(d.get('schema') == decision.SCHEMA and d.get('status') == decision.STATUS and
            d.get('chromosome') == c['chromosome'] and d.get('source_test_opened') is False and
            d.get('simulation_ready') is False, 'L1 decision is incomplete or incompatible')
    smp = inputs.descriptor(c['donor_manifest']); source = read(smp)
    require(d['inputs_sha256'].get(str(smp)) == c['donor_manifest']['sha256'], 'Donor source not authenticated by L1 decision')
    require(source.get('schema') == 'r02_lai_genotype_support_v1' and
            source.get('status') == 'COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION' and
            source['chromosome'] == c['chromosome'] and source['genotype_contract']['panel_role'] == ROLE and
            source['build_status'] == 'BUILD_VERIFIED_BY_SUPPLIED_EVIDENCE' and
            source['donor_phase_support'] == 'OBSERVED_PHASE_ONLY_NOT_CERTIFIED', 'Invalid donor evidence')
    require(d['quality_modes'][ROLE] == source['genotype_contract']['quality']['mode'], 'L1 quality mode differs')
    rp = inputs.descriptor(c['roles'])
    require(c['roles']['sha256'] == d['roles_sha256'] == source['input_files']['roles']['sha256'], 'Historical roles changed')
    table = inputs.add(dp.parent/'candidate_alleles.tsv.gz', d['outputs_sha256']['candidate_alleles.tsv.gz'])
    info = read(inputs.add(dp.parent/'role_support.json', d['outputs_sha256']['role_support.json']))
    require(info['disjointness']['overlap_individuals'] == 0 and
            info['disjointness']['roles_sha256'] == c['roles']['sha256'], 'Role disjointness is not verified')
    projection = source['input_files']['panel_vcf']
    require(Path(projection['path']).stat().st_size <= resources['max_panel_bytes'], 'Projected VCF exceeds byte budget')
    panel = inputs.add(projection['path'], projection['sha256'])
    require(d['inputs_sha256'].get(str(panel)) == projection['sha256'], 'Projected donor VCF differs from L1')
    donors = roster(rp)
    expected = source['genotype_contract']['panel_sample_members_sha256']
    require(gt.base.sample_hash(sorted(donors)) == expected == d['role_members_sha256'][ROLE], 'Donor membership differs')
    require(len(donors) == source['ancestry_strata']['ALL'] == d['role_members'][ROLE]['ALL'], 'Donor denominator differs')
    require(Counter(row['ancestry'] for row in donors.values()) ==
            {a:n for a,n in source['ancestry_strata'].items() if a != 'ALL'}, 'Donor ancestry labels changed')
    return inputs, c, d, source, panel, donors, table


def candidates(path, manifest, maximum):
    result = {}; rows = 0
    with gzip.open(path, 'rt') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require(set(decision.FIELDS+decision.DECISION_FIELDS) == set(reader.fieldnames or []), 'Candidate schema differs')
        for key, group in itertools.groupby(reader, lambda r: tuple(r[k] for k in KEY)):
            group = list(group); rows += len(group)
            require(rows <= maximum and len(group) == len(manifest['role_members'][ROLE]), 'Candidate row/stratum budget differs')
            strata = {r['panel_ancestry']:r for r in group}
            require(len(strata) == len(group) and set(strata) == set(manifest['role_members'][ROLE]), 'Missing/duplicate ancestry stratum')
            row = strata['ALL']
            if row['decision'] != CANDIDATE: continue
            require(key not in result and row['chromosome'] == manifest['chromosome'], 'Duplicate/wrong-chromosome candidate')
            pos = int(row['position_bp']); selected = int(row['source_rare_allele_index'])
            require(selected in (0,1) and row['counted_allele'] == (row['REF'], row['ALT'])[selected], 'Counted allele orientation differs')
            require(row['map_range_status'] == 'INSIDE' and manifest['map_first_bp'] <= pos <= manifest['map_last_bp'], 'Candidate outside authenticated map')
            result[key] = strata
    require(rows == manifest['sparse_rows'] and len(result) == manifest['conditional_candidate_loci']['ALL'], 'L1 candidate denominator differs')
    return result


def call_state(record, sample, allele_index, contract):
    call = record.samples[sample]; raw = call.get('GT')
    if not raw or any(a is None for a in raw): state = 'MISSING_OR_PARTIAL'
    elif len(raw) != 2: state = 'NON_DIPLOID'
    else: state = 'COMPLETE'
    complete = state == 'COMPLETE'
    dose = sum(a == allele_index for a in raw) if complete else 'NA'
    observed = bool(complete and call.phased)
    quality = 'NOT_MEASURED_GT_ONLY' if contract['quality']['mode'] == 'GT_ONLY' else (
        gt.quality_state(call, contract['quality']).upper() if complete else 'NOT_EVALUABLE_GT')
    allowed = gt.site_allowed(record, contract['site_filter_policy'])
    return dict(gt_state=state, counted_dose=dose, observed_phased=observed,
        copy1_counted=int(raw[0] == allele_index) if observed else 'NA',
        copy2_counted=int(raw[1] == allele_index) if observed else 'NA',
        phase_set=call.get('PS') if call.get('PS') is not None else 'NA',
        phase_set_scope='CHROMOSOME_LOCAL_OBSERVED_NOT_VALIDATED', quality_state=quality,
        site_filter=';'.join(record.filter.keys()) or '.', site_allowed=allowed,
        DP=call.get('DP') if call.get('DP') is not None else 'NA',
        GQ=call.get('GQ') if call.get('GQ') is not None else 'NA')


def run(contract_path, digest, outdir, cpus=1):
    import pysam
    started = time.monotonic(); out = Path(outdir)
    require(not out.exists() and out.parent.is_dir(), 'Output exists or parent missing; use Nextflow resume')
    inputs,c,d,source,panel,donors,table = authenticate(contract_path,digest)
    limits = c['resources']; require(cpus == limits['cpus'], 'Reserved CPU and actual reader threads differ')
    selected = candidates(table,d,limits['max_candidate_rows'])
    require(len(selected)*len(donors) <= limits['max_private_rows'], 'Private row budget exceeded')
    decision.support.check_space(out.parent, limits['min_free_gib'])
    out.mkdir(mode=0o700)
    def guard():
        decision.guard(out,started,limits['min_free_gib'],limits['timeout_seconds'],limits['max_output_bytes'])
    # Validate role/order BEFORE decoding any record; only this projection is opened.
    with pysam.VariantFile(str(panel), threads=1) as vcf:
        samples = list(vcf.header.samples)
        require(set(samples) == set(donors) and len(samples) == len(donors), 'Projection contains reserved or missing people')
        gcpath = out/'genotype_contract.json'; write(gcpath,source['genotype_contract'])
        contract = gt.genotype_contract(SimpleNamespace(genotype_contract=gcpath,panel_role=ROLE,panel_vcf=panel),samples)
        gcpath.unlink()  # locally created validation helper, never an input or result
        groups = {'ALL':list(range(len(samples)))}
        for i,s in enumerate(samples): groups.setdefault(donors[s]['ancestry'],[]).append(i)
        require('GT' in vcf.header.formats, 'VCF lacks GT')
        counts = {s:Counter() for s in samples}; seen=set(); record_count=0; ncarriages=0
        by_pos = defaultdict(list)
        for key in selected: by_pos[int(key[1])].append(key)
        handles=[]
        try:
            h,rosterwriter=writer(out/'donors.private.tsv',ROSTER); handles.append(h)
            for s in samples: rosterwriter.writerow(donors[s])
            callfields=(*KEY,'sample_id','ancestry','atomic_unit_id','gt_state','counted_dose','observed_phased',
                'copy1_counted','copy2_counted','phase_set','phase_set_scope','quality_state','site_filter','site_allowed','DP','GQ')
            h,cw=writer(out/'calls.private.tsv.gz',callfields); handles.append(h)
            h,hw=writer(out/'carriages.private.tsv.gz',(*KEY,'sample_id','ancestry','atomic_unit_id','copy_index',
                'phase_set','quality_state','site_filter','reference_scenario')); handles.append(h)
            h,lw=writer(out/'loci.tsv.gz',(*KEY,'ancestry',*gt.COUNTERS,'n_role_members',
                'n_declared_carrier_units','carrier_units_missing_label','reference_scenario','map_range_status')); handles.append(h)
            previous=0
            for pos, records in itertools.groupby(vcf,lambda r:r.pos):
                batch=list(records); record_count += len(batch)
                require(pos >= previous and record_count <= limits['max_panel_records'], 'Unsorted VCF or record budget exceeded'); previous=pos
                require(all(str(r.contig).removeprefix('chr') == c['chromosome'] for r in batch), 'Mixed chromosome VCF')
                if record_count % 2000 == 0: guard()
                for key in by_pos.get(pos,[]):
                    require(key not in seen, 'Candidate encountered twice'); seen.add(key)
                    record,status = gt.choose_record(batch,key)
                    require(record is not None, 'Candidate allele absent from authenticated donor VCF')
                    selected_index=int(selected[key]['ALL']['source_rare_allele_index'])
                    rep=contract['allele_representation']['status']
                    require(not(selected_index==0 and (rep=='UNKNOWN' or (len(batch)>1 and rep=='ORIGINAL_UNSPLIT'))), 'Candidate REF dosage unresolved')
                    allele_index=record.alleles.index(key[4])
                    individual=gt.count_record(record,allele_index,samples,contract)
                    values=dict(zip(KEY,key))
                    for a,indices in groups.items():
                        total=gt.group_support(individual,indices,contract); old=selected[key][a]
                        for field in (*gt.COUNTERS,'n_role_members'):
                            require(str(total[field])==str(old['SOURCE_VALID_'+field]), 'L1 count reconciliation failed: '+field)
                        require(status == old['SOURCE_VALID_correspondence_status'], 'L1 allele mapping differs')
                        units={donors[samples[i]]['atomic_unit_id'] for i in indices if individual[i]['n_carriers_called'] and donors[samples[i]]['atomic_unit_id']}
                        unknown=sum(individual[i]['n_carriers_called'] for i in indices if not donors[samples[i]]['atomic_unit_id'])
                        lw.writerow(dict(values,ancestry=a,**{k:total[k] for k in (*gt.COUNTERS,'n_role_members')},
                            n_declared_carrier_units=len(units),carrier_units_missing_label=unknown,
                            reference_scenario=old['reference_scenario'],map_range_status='INSIDE'))
                    for i,s in enumerate(samples):
                        state=call_state(record,s,allele_index,contract)
                        cw.writerow(dict(values,sample_id=s,ancestry=donors[s]['ancestry'],atomic_unit_id=donors[s]['atomic_unit_id'],**state))
                        cc=counts[s]; one=individual[i]
                        cc.update(candidate_loci=1,complete_calls=one['n_gt_complete'],missing_calls=one['n_gt_missing'],
                            non_diploid_calls=one['n_gt_non_diploid'],counted_copies=one['ac_counted_called'],
                            carrier_loci=one['n_carriers_called'],phased_carrier_loci=one['n_phased_carriers_called'],
                            unphased_carrier_loci=one['n_carriers_called']-one['n_phased_carriers_called'],
                            quality_evaluable_calls=one['n_quality_evaluable'])
                        for copy in (1,2):
                            if state['copy'+str(copy)+'_counted'] == 1:
                                cc['observed_carrier_copies']+=1; ncarriages+=1
                                hw.writerow(dict(values,sample_id=s,ancestry=donors[s]['ancestry'],atomic_unit_id=donors[s]['atomic_unit_id'],copy_index=copy,
                                    phase_set=state['phase_set'],quality_state=state['quality_state'],site_filter=state['site_filter'],
                                    reference_scenario=selected[key][donors[s]['ancestry']]['reference_scenario']))
                    guard()
            require(seen==set(selected), 'Not all candidate alleles recovered from donor projection')
            require(record_count==source['panel_counts']['records'], 'Donor VCF record denominator differs from L1')
        finally:
            for h in handles: h.close()
    h,w=writer(out/'donor_support.private.tsv',(*ROSTER,'chromosome',*COUNTS))
    with h:
        for s in samples:
            row=dict(donors[s],chromosome=c['chromosome'],**{k:counts[s][k] for k in COUNTS})
            if contract['quality']['mode']=='GT_ONLY': row['quality_evaluable_calls']='NA'
            w.writerow(row)
    summary=dict(chromosome=c['chromosome'],distinct_donors=len(samples),candidate_loci=len(selected),
        person_locus_rows=len(samples)*len(selected),observed_carrier_copy_rows=ncarriages,
        donor_members_sha256=gt.base.sample_hash(sorted(samples)),donors_with_no_candidate_carriage=sum(not counts[s]['carrier_loci'] for s in samples),
        reference_zero_candidates=sum(v['ALL']['reference_scenario'].startswith('REFERENCE_ZERO') for v in selected.values()),
        all_candidate_counts_reconciled=True,source_test_opened=False,simulation_ready=False)
    write(out/'summary.json',summary); write(out/'definitions.json',definitions())
    guard(); inputs.unchanged()
    result=dict(schema=SCHEMA,status=STATUS,chromosome=c['chromosome'],source_contract_sha256=digest,
        decision_manifest_sha256=c['decision_manifest']['sha256'],donor_manifest_sha256=c['donor_manifest']['sha256'],
        roles_sha256=c['roles']['sha256'],map_sha256=d['map_sha256'],map_first_bp=d['map_first_bp'],map_last_bp=d['map_last_bp'],
        genotype_contract=contract,source_catalogue_sha256=d['source_catalogue_sha256'],summary=summary,
        contains_sample_identifiers=True,contains_individual_genotypes=True,public_distribution_allowed=False,
        genotype_roles_read=[ROLE],source_test_opened=False,reference_genotypes_opened=False,
        new_donor_split_assigned=False,phase_validated=False,simulation_ready=False,simulations_executed=False,
        resources=limits,actual_reader_threads=1,panel_records_read=record_count,
        elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        inputs_sha256={p:v[2] for p,v in inputs.entries.items()},
        code_sha256={name:sha(Path(__file__).parent/name) for name in CODE_FILES},outputs_sha256={n:sha(out/n) for n in sorted(FILES)})
    guard()
    require(sum(p.stat().st_size for p in out.iterdir()) + len((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
            <= limits['max_output_bytes'], 'Final manifest exceeds output budget')
    write(out/'manifest.json',result)
    return result


def aggregate(campaign_path,digest,manifests,outdir):
    inputs=decision.Inputs(); cp=inputs.add(campaign_path,digest); campaign=read(cp)
    require(campaign.get('schema')=='r02_lai_donor_haplotypes_campaign_v1','Invalid donor campaign')
    jobs=campaign['jobs']; expected={j['chromosome']:j for j in jobs}
    require(len(expected)==len(jobs)==22 and set(expected)=={str(i) for i in range(1,23)},'Aggregation requires exactly22 autosomes')
    require(len(manifests)==22,'Missing autosomal preparation')
    found={}; common=None; metadata=None; counts=defaultdict(Counter)
    for path in manifests:
        mp=Path(path); m=read(inputs.add(mp,sha(mp))); chrom=m.get('chromosome')
        require(m.get('schema')==SCHEMA and m.get('status')==STATUS and chrom in expected and chrom not in found,'Incomplete/duplicate donor preparation')
        require(m['source_contract_sha256']==expected[chrom]['contract']['sha256'],'Campaign source contract differs')
        inputs.descriptor(expected[chrom]['contract'])
        require(m['genotype_roles_read']==[ROLE] and all(m[k] is False for k in
                ('source_test_opened','new_donor_split_assigned','phase_validated','reference_genotypes_opened',
                 'simulation_ready','simulations_executed')),'Unexpected role/split/inference')
        binding=(m['roles_sha256'],m['summary']['donor_members_sha256'],m['genotype_contract']['quality'],m['code_sha256'])
        require(common is None or binding==common,'Autosomal roles/quality differ');common=binding
        require(set(m['outputs_sha256'])==FILES,'Incomplete private products')
        for name,value in m['outputs_sha256'].items(): inputs.add(mp.parent/name,value)
        with (mp.parent/'donors.private.tsv').open() as handle:
            rows=list(csv.DictReader(handle,delimiter='\t'))
        roster_now={r['sample_id']:r for r in rows}
        require(len(roster_now)==len(rows)==m['summary']['distinct_donors'] and (metadata is None or metadata==roster_now),'Donor roster mismatch')
        metadata=roster_now
        with (mp.parent/'donor_support.private.tsv').open() as handle:
            support_rows=list(csv.DictReader(handle,delimiter='\t'))
        require(len(support_rows)==len(metadata) and {r['sample_id'] for r in support_rows}==set(metadata),'Incomplete donor support')
        for row in support_rows:
            require(row['chromosome']==chrom and all(row[k]==metadata[row['sample_id']][k] for k in ROSTER),'Donor support metadata differs')
            for key in COUNTS:
                if row[key]!='NA': counts[row['sample_id']][key]+=int(row[key])
        found[chrom]=dict(manifest_sha256=sha(mp),summary=m['summary'])
    out=Path(outdir);require(not out.exists(),'Aggregate exists; use resume');out.mkdir(mode=0o700)
    for name,fields in (('donors.private.tsv',ROSTER),('donor_support.private.tsv',(*ROSTER,'chromosome',*COUNTS))):
        h,w=writer(out/name,fields)
        with h:
            for s in sorted(metadata):
                row=dict(metadata[s])
                if name.startswith('donor_support'):
                    row.update(chromosome='AUTOSOMES',**{k:counts[s][k] for k in COUNTS})
                    if common[2]['mode']=='GT_ONLY':row['quality_evaluable_calls']='NA'
                w.writerow(row)
    summary=dict(expected_chromosomes=list(map(str,range(1,23))),products=found,distinct_donors=len(metadata),
        candidate_loci=sum(x['summary']['candidate_loci'] for x in found.values()),
        person_locus_rows=sum(x['summary']['person_locus_rows'] for x in found.values()),
        observed_carrier_copy_rows=sum(x['summary']['observed_carrier_copy_rows'] for x in found.values()),
        source_test_opened=False,simulation_ready=False,phase_validated=False,new_donor_split_assigned=False)
    write(out/'summary.json',summary);write(out/'definitions.json',definitions());inputs.unchanged()
    result=dict(schema=AGG_SCHEMA,status=AGG_STATUS,source_contract_sha256=digest,summary=summary,
        contains_sample_identifiers=True,contains_individual_genotypes=False,public_distribution_allowed=False,
        source_test_opened=False,simulation_ready=False,phase_validated=False,
        outputs_sha256={n:sha(out/n) for n in sorted(AGG_FILES)},inputs_sha256={p:v[2] for p,v in inputs.entries.items()})
    write(out/'manifest.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('chromosome');p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--cpus',type=int,required=True)
    p=sub.add_parser('aggregate');p.add_argument('--campaign',required=True);p.add_argument('--campaign-sha256',required=True)
    p.add_argument('--manifest',action='append',required=True);p.add_argument('--output-dir',required=True)
    args=parser.parse_args();os.umask(0o077)
    result=run(args.contract,args.contract_sha256,args.output_dir,args.cpus) if args.command=='chromosome' else aggregate(args.campaign,args.campaign_sha256,args.manifest,args.output_dir)
    print(json.dumps({'schema':result['schema'],'status':result['status']}))


if __name__=='__main__':main()
