#!/usr/bin/env python3
"""Audit every marker in existing donor chromosomes, without changing genotypes.

The exact SOURCE_VALID projection and its earlier reference-assembly audit are
authenticated before genotype decoding. Missing calls are unknown, homozygous
calls do not need phase to assign identical alleles, and absent PS does not mean
bad phase. Observed GT/PS can establish format compatibility, not phase accuracy,
donor independence or natural local-ancestry truth. Outputs remain private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
from pathlib import Path
import resource
import time

import r02_lai_material_decision as base
from r02_lai_phased_audit import read_map as map_endpoints

require, sha, read, write = base.require, base.sha, base.read, base.write
SCHEMA = 'r02_lai_full_chromosome_audit_v1'
STATUS = 'COMPLETE_PRIVATE_FULL_MARKER_AUDIT_NOT_PHASE_OR_SIMULATION_VALIDATION'
AGG_SCHEMA = 'r02_lai_full_chromosome_audit_aggregate_v1'
AGG_STATUS = 'COMPLETE_AUTOSOMAL_FULL_MARKER_AUDIT_NOT_SIMULATION_READY'
INPUT_SCHEMA = 'r02_lai_full_chromosome_audit_input_v1'
CAMPAIGN_SCHEMA = 'r02_lai_full_chromosome_audit_campaign_v1'
ROLE = 'SOURCE_VALID'
CODE_FILES = ('r02_lai_full_chromosome_audit.py', 'r02_lai_material_decision.py',
              'r02_lai_simulation_material.py', 'r02_lai_support_review.py',
              'r02_lai_phased_audit.py', 'r02_lai_allele_support.py',
              'r02_lai_build_verification.py', 'r02_lai_role_subset.py',
              'r02_lai_genotype_support.py', 'r02_genomic_pair_evidence.py')
ROSTER = ('sample_id', 'role', 'ancestry', 'atomic_unit_id', 'canonical_population')
FIELDS = ('records', 'complete_diploid_calls', 'missing_or_partial_calls',
          'non_diploid_calls', 'homozygous_calls', 'heterozygous_calls',
          'phased_heterozygous_calls', 'unphased_heterozygous_calls',
          'heterozygous_calls_with_ps', 'phased_heterozygous_calls_without_ps',
          'explicit_ps_runs', 'copyable_observed_calls', 'eligible_marker_calls',
          'eligible_copyable_calls', 'unusable_eligible_calls')
BLOCK_FIELDS = ('sample_id', 'chromosome', 'phase_set', 'first_het_bp', 'last_het_bp',
                'phased_heterozygotes', 'scope')


def definitions():
    return dict(scope='Every record in the authenticated donor chromosome projection; not only rare candidates.',
        source_test_opened=False, simulations=False, training=False, simulation_ready=False,
        phase_validated=False, donor_split_assigned=False, genotypes_modified=False,
        public_distribution_allowed=False, contains_sample_identifiers=True,
        contains_individual_genotypes=False, source_role=ROLE,
        quality='GT_ONLY inherited: DP/GQ not measured, not certified high-quality genotypes.',
        copyable='Complete diploid homozygote or phased heterozygote; format compatibility only.',
        eligible_marker='Biallelic SNV, inherited PASS_OR_UNFILTERED and inside inclusive map endpoints. No marker is removed by this audit.',
        missing='Missing or partial GT remains unknown. Ploidy and missingness counts can overlap.',
        phase_sets='Consecutive observed phased heterozygotes carrying the same explicit PS form a reported run. Missing/unphased heterozygotes break a run. Homozygotes do not. Runs are not verified phase blocks.',
        no_ps='PS absence is unrecorded block support, not automatic failure and not chromosome-wide accuracy.',
        map='Coverage between first and last map points only; no extrapolation, ancestry accuracy or uniform recombination claim.',
        denominators='Per-person rows count all records; rates use the named call/marker denominator. Calls/markers are not independent biological replicates.',
        ancestry='Historical individual ancestry label, not ancestry of each copied marker.',
        exposure='SOURCE_VALID was previously used for development; not an untouched test cohort.',
        interpretation='Supports defining usable simulated marker territory and phase sensitivities. Does not validate LAI accuracy or donor independence.')


def table(path, fields, rows):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'xt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def read_rows(path):
    with Path(path).open() as handle:
        yield from csv.DictReader(handle, delimiter='\t')


def authenticate(contract_path, digest):
    inputs = base.Inputs(); cp = inputs.add(contract_path, digest); c = read(cp)
    require(set(c) == {'schema', 'chromosome', 'donor_manifest', 'roles', 'genetic_map',
                      'map_support', 'resources', 'phase_provenance'}, 'Unexpected audit contract fields')
    require(c['schema'] == INPUT_SCHEMA and c['chromosome'] in {str(i) for i in range(1,23)}, 'Wrong audit schema/chromosome')
    r = c['resources']
    require(set(r) == {'cpus','max_samples','max_panel_bytes','max_panel_records','max_output_bytes',
                      'min_free_gib','timeout_seconds','guard_every_records'}, 'Explicit resource bounds required')
    require(r['cpus'] == 1 and all(type(v) in (int,float) and v >= (0 if k == 'min_free_gib' else 1)
            for k,v in r.items()), 'Invalid resource limits')
    require(all(type(r[k]) is int for k in r if k != 'min_free_gib'), 'Integer resource bounds required')
    mp = inputs.descriptor(c['donor_manifest']); m = read(mp)
    require(m.get('schema') == 'r02_lai_genotype_support_v1' and
        m.get('status') == 'COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION' and
        m.get('chromosome') == c['chromosome'] and m['genotype_contract']['panel_role'] == ROLE and
        m.get('build_status') == 'BUILD_VERIFIED_BY_SUPPLIED_EVIDENCE', 'Incompatible donor source')
    require(m['genotype_contract']['quality']['mode'] == 'GT_ONLY' and
            m['genotype_contract']['site_filter_policy'] == 'PASS_OR_UNFILTERED', 'Inherited quality/filter changed')
    rp = inputs.descriptor(c['roles'])
    require(c['roles']['sha256'] == m['input_files']['roles']['sha256'], 'Role metadata changed')
    roster, all_people = {}, set()
    for row in read_rows(rp):
        require(set(ROSTER) <= set(row) and row['sample_id'] not in all_people, 'Invalid/duplicate role row')
        all_people.add(row['sample_id'])
        if row['role'] == ROLE: roster[row['sample_id']] = {k:row[k] for k in ROSTER}
    require(0 < len(roster) == m['ancestry_strata']['ALL'] <= r['max_samples'], 'Donor denominator differs')
    require(Counter(x['ancestry'] for x in roster.values()) == {k:v for k,v in m['ancestry_strata'].items() if k != 'ALL'}, 'Donor ancestry labels differ')
    pp = m['input_files']['panel_vcf']; panel = inputs.add(pp['path'],pp['sha256'])
    require(panel.stat().st_size == pp['size_bytes'] <= r['max_panel_bytes'], 'Donor VCF byte bound differs')
    build = m['input_files']['panel_build_verification']
    build_path = inputs.add(build['path'],build['sha256']); b = read(build_path)
    require(b.get('schema') == 'r02_lai_build_verification_v1' and b.get('status') == 'VERIFIED' and
        b.get('method') == 'bcftools_norm_check_ref_e_all_records_v1' and b.get('build') == 'GRCh38' and
        b['reference_audit']['scope'] == 'ALL_VCF_RECORDS' and b['reference_audit']['mismatches'] == 0,
        'Full donor REF verification absent')
    require(b['vcf_sha256'] == pp['sha256'], 'Build evidence belongs to another VCF')
    mapped = read(inputs.descriptor(c['map_support'])); map_path = inputs.descriptor(c['genetic_map'])
    low, high, points = map_endpoints(map_path,c['chromosome'])
    require(mapped['schema'] == 'r02_lai_phased_map_support_v1' and mapped['chromosome'] == c['chromosome'] and
        mapped['map_sha256'] == c['genetic_map']['sha256'] and
        (low,high) == (mapped['first_bp'],mapped['last_bp']) and mapped['extrapolation'] is False, 'Map provenance differs')
    provenance = c['phase_provenance']
    require(set(provenance) == {'status','evidence'} and provenance['status'] in
            ('NOT_LINKED_TO_EXACT_PROJECTED_BYTES','LINKED_PROCESSING_RECORD_NOT_ACCURACY'), 'Explicit phase provenance required')
    require(isinstance(provenance['evidence'],list), 'Phase evidence must be a descriptor list')
    if provenance['status'].startswith('LINKED'): require(provenance['evidence'], 'Linked phase origin needs evidence')
    for item in provenance['evidence']: inputs.descriptor(item)
    return inputs,c,m,panel,roster,(low,high,points),b


class Person:
    """Bounded per-person counters and one open explicit-PS run."""
    def __init__(self, row, chromosome, block_writer):
        self.row, self.chromosome, self.writer = row, chromosome, block_writer
        self.counts = dict.fromkeys(FIELDS,0)
        self.block = None
        self.first_het_bp, self.last_het_bp = None,None

    def close(self):
        if self.block is not None:
            ps,start,end,n = self.block
            self.writer.writerow(dict(sample_id=self.row['sample_id'],chromosome=self.chromosome,
                phase_set=ps,first_het_bp=start,last_het_bp=end,phased_heterozygotes=n,
                scope='OBSERVED_CONSECUTIVE_PS_RUN_NOT_ACCURACY_VALIDATED'))
            self.block = None

    def observe(self, call, position, eligible, ps_present):
        c = self.counts; c['records'] += 1
        raw = call.get('GT') or ()
        missing = not raw or any(a is None for a in raw)
        diploid = len(raw) == 2
        if missing: c['missing_or_partial_calls'] += 1
        if not diploid: c['non_diploid_calls'] += 1
        if eligible: c['eligible_marker_calls'] += 1
        if missing or not diploid:
            if eligible: c['unusable_eligible_calls'] += 1
            self.close(); return
        c['complete_diploid_calls'] += 1
        het = raw[0] != raw[1]
        phased = bool(call.phased)
        copyable = not het or phased
        if copyable: c['copyable_observed_calls'] += 1
        if eligible: c['eligible_copyable_calls' if copyable else 'unusable_eligible_calls'] += 1
        if not het: c['homozygous_calls'] += 1; return
        c['heterozygous_calls'] += 1
        self.first_het_bp = position if self.first_het_bp is None else self.first_het_bp
        self.last_het_bp = position
        c['phased_heterozygous_calls' if phased else 'unphased_heterozygous_calls'] += 1
        ps = call.get('PS') if ps_present else None
        if ps is not None: c['heterozygous_calls_with_ps'] += 1
        if phased and ps is None: c['phased_heterozygous_calls_without_ps'] += 1
        if not phased or ps is None: self.close(); return
        if self.block is None or self.block[0] != ps:
            self.close(); self.block = [ps,position,position,1]; c['explicit_ps_runs'] += 1
        else: self.block[2] = position; self.block[3] += 1

    def result(self):
        return dict(self.row,**self.counts,first_het_bp=self.first_het_bp or 'NA',last_het_bp=self.last_het_bp or 'NA')


def seal(out, inputs, summary, chromosome=None):
    write(out/'summary.json',summary); write(out/'definitions.json',definitions())
    inputs.unchanged()
    result = dict(schema=SCHEMA if chromosome else AGG_SCHEMA,status=STATUS if chromosome else AGG_STATUS,
        evidence_level='FEASIBILITY_FULL_MARKER_DONOR_AUDIT',**definitions(),
        chromosome=chromosome,summary=summary,
        inputs_sha256={p:s[2] for p,s in inputs.entries.items()},
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        code_sha256={p:sha(Path(__file__).parent/p) for p in CODE_FILES})
    write(out/'manifest.json',result); return result


def audit(contract_path,digest,outdir,cpus=1):
    import pysam
    start=time.monotonic(); out=Path(outdir)
    require(not out.exists() and out.parent.is_dir(), 'Output exists or parent missing; use Nextflow resume')
    inputs,c,m,panel,roster,mapped,build=authenticate(contract_path,digest)
    r=c['resources']; require(cpus == r['cpus'], 'Reader/reserved CPUs differ')
    base.support.check_space(out.parent,r['min_free_gib']); out.mkdir(mode=0o700)
    guard=lambda:base.guard(out,start,r['min_free_gib'],r['timeout_seconds'],r['max_output_bytes'])
    sites=Counter(); formats=Counter(); previous=0; at_position=set(); first=None
    with pysam.VariantFile(str(panel),threads=1) as vcf, gzip.open(out/'phase_sets.private.tsv.gz','xt',newline='') as bh:
        samples=list(vcf.header.samples)
        require(len(samples)==len(roster) and set(samples)==set(roster), 'VCF includes wrong or reserved people')
        require('GT' in vcf.header.formats,'VCF lacks GT')
        writer=csv.DictWriter(bh,fieldnames=BLOCK_FIELDS,delimiter='\t',lineterminator='\n'); writer.writeheader()
        people=[Person(roster[s],c['chromosome'],writer) for s in samples]
        for record in vcf:
            sites['records']+=1
            require(sites['records']<=r['max_panel_records'], 'Record budget exceeded')
            require(record.chrom in (c['chromosome'],'chr'+c['chromosome']) and record.pos>=previous,'Wrong chromosome or unsorted VCF')
            if record.pos!=previous: at_position.clear()
            key=(record.ref,record.alts)
            require(key not in at_position,'Duplicate exact marker'); at_position.add(key)
            first=record.pos if first is None else first; previous=record.pos
            inside=mapped[0]<=record.pos<=mapped[1]
            sites['inside_map' if inside else ('before_map' if record.pos<mapped[0] else 'after_map')]+=1
            snv=bool(record.alts and len(record.alts)==1 and all(len(a)==1 and a in 'ACGT' for a in (record.ref,*record.alts)))
            sites['biallelic_snv' if snv else 'other_variant']+=1
            filters=set(record.filter.keys()); allowed=not filters or filters=={'PASS'}
            sites['filter_unfiltered' if not filters else ('filter_pass' if filters=={'PASS'} else 'filter_rejected')]+=1
            eligible=bool(inside and snv and allowed)
            sites['eligible_markers' if eligible else 'ineligible_markers']+=1
            for field in record.format.keys(): formats[field]+=1
            ps_present='PS' in record.format
            for person,call in zip(people,record.samples.values()): person.observe(call,record.pos,eligible,ps_present)
            if sites['records']%r['guard_every_records']==0: guard()
        for person in people: person.close()
    require(sites['records']>0 and sites['records']==build['reference_audit']['records_checked'],'Full-marker denominator differs from authenticated REF audit')
    perperson=[p.result() for p in people]
    table(out/'people.private.tsv',(*ROSTER,*FIELDS,'first_het_bp','last_het_bp'),perperson)
    totals={k:sum(p.counts[k] for p in people) for k in FIELDS}
    require(totals['records']==sites['records']*len(samples),'Call denominator mismatch')
    summary=dict(chromosome=c['chromosome'],donors=len(samples),ancestry_counts=dict(Counter(x['ancestry'] for x in roster.values())),
        markers=dict(sites),calls=totals,first_marker_bp=first,last_marker_bp=previous,
        map_first_bp=mapped[0],map_last_bp=mapped[1],map_points=mapped[2],format_record_counts=dict(formats),
        phase_provenance=c['phase_provenance'],full_marker_haplotypes_audited=True,
        elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        noncopyable_calls_require_explicit_masking=totals['unusable_eligible_calls']>0,
        full_chromosome_phase_accuracy='NOT_MEASURED',quality='NOT_MEASURED_GT_ONLY')
    guard(); return seal(out,inputs,summary,c['chromosome'])


def aggregate(campaign_path,digest,manifests,outdir):
    inputs=base.Inputs(); cp=inputs.add(campaign_path,digest); campaign=read(cp)
    require(campaign.get('schema')==CAMPAIGN_SCHEMA,'Wrong campaign schema')
    expected={j['chromosome']:j for j in campaign['jobs']}
    require(len(expected)==len(campaign['jobs'])==22 and set(expected)=={str(i) for i in range(1,23)},'Full aggregate needs22autosomes')
    out=Path(outdir); require(not out.exists() and out.parent.is_dir(),'Output exists or parent missing')
    seen=set(); rows=[]; donors=None; combined=None
    for path in manifests:
        p=Path(path).resolve(strict=True); m=read(inputs.add(p,sha(p))); chrom=m.get('chromosome')
        require(m.get('schema')==SCHEMA and m.get('status')==STATUS and chrom in expected and chrom not in seen,'Invalid/duplicate chromosome result')
        contract=inputs.descriptor(expected[chrom]['contract']); c=read(contract)
        require(m['inputs_sha256'].get(str(contract))==expected[chrom]['contract']['sha256'],'Different chromosome contract')
        require(m['source_test_opened'] is False and m['simulation_ready'] is False,'Scope differs')
        for name,d in m['outputs_sha256'].items(): inputs.add(p.parent/name,d)
        current=list(read_rows(p.parent/'people.private.tsv'))
        roster={x['sample_id']:tuple(x[k] for k in ROSTER) for x in current}
        require(len(roster)==len(current)==m['summary']['donors'],'Person denominator differs')
        if donors is None:
            donors=roster; combined={s:dict(zip(ROSTER,r),**dict.fromkeys(FIELDS,0)) for s,r in roster.items()}
        require(donors==roster,'Donor memberships/metadata differ between chromosomes')
        for row in current:
            for k in FIELDS: combined[row['sample_id']][k]+=int(row[k])
        rows.append(dict(chromosome=chrom,donors=len(roster),**m['summary']['markers'],
            **{'calls_'+k:v for k,v in m['summary']['calls'].items()})); seen.add(chrom)
    require(seen==set(expected),'Incomplete chromosome aggregate')
    out.mkdir(mode=0o700)
    fields=sorted({k for r in rows for k in r})
    table(out/'chromosomes.tsv',fields,sorted(rows,key=lambda r:int(r['chromosome'])))
    table(out/'people.private.tsv',(*ROSTER,*FIELDS),sorted(combined.values(),key=lambda r:r['sample_id']))
    summary=dict(chromosomes=22,donors=len(donors),full_marker_haplotypes_audited=True,
        calls={k:sum(row[k] for row in combined.values()) for k in FIELDS},
        marker_records=sum(r['records'] for r in rows),
        full_chromosome_phase_accuracy='NOT_MEASURED',quality='NOT_MEASURED_GT_ONLY')
    return seal(out,inputs,summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('chromosome'); p.add_argument('--contract',required=True);p.add_argument('--contract-sha256',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--cpus',type=int,default=1)
    p=sub.add_parser('aggregate');p.add_argument('--campaign',required=True);p.add_argument('--campaign-sha256',required=True)
    p.add_argument('--manifest',action='append',required=True);p.add_argument('--output-dir',required=True)
    a=parser.parse_args()
    if a.command=='chromosome': audit(a.contract,a.contract_sha256,a.output_dir,a.cpus)
    else: aggregate(a.campaign,a.campaign_sha256,a.manifest,a.output_dir)


if __name__=='__main__': main()
