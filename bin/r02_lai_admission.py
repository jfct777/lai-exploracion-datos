#!/usr/bin/env python3
"""Reconcile completed LAI preparation evidence without simulating or changing roles.

Admission is scoped to supplied development material. Genotype format, phase
accuracy, donor separation and population representativeness are distinct gates.
Only authenticated summaries/private inventories are read; no VCF is decoded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import math
import os
from pathlib import Path
import resource
import time

import r02_lai_material_decision as base

require, sha, read, write = base.require, base.sha, base.read, base.write
SCHEMA = 'r02_lai_admission_input_v1'
OUTPUT_SCHEMA = 'r02_lai_admission_v1'
STATUS = 'COMPLETE_CONDITIONAL_MATERIAL_ADMISSION_NOT_SIMULATION_AUTHORIZATION'
ROLES = ('REF_TRAIN', 'SOURCE_VALID')
ANCESTRIES = ('African', 'European', 'Native_American')
SCENARIOS = {(p, f) for p in (4, 8, 12) for f in (0.0221, 0.0442)}
IDENTITY = ('sample_id', 'role', 'ancestry', 'atomic_unit_id', 'canonical_population')
KEY = ('chromosome', 'position_bp', 'REF', 'ALT', 'counted_allele')
COUNTS = ('records', 'complete_diploid_calls', 'missing_or_partial_calls',
          'non_diploid_calls', 'homozygous_calls', 'heterozygous_calls', 'phased_heterozygous_calls',
          'unphased_heterozygous_calls', 'heterozygous_calls_with_ps',
          'copyable_observed_calls', 'eligible_marker_calls', 'eligible_copyable_calls',
          'unusable_eligible_calls')
CODE_FILES = ('r02_lai_admission.py', 'r02_lai_material_decision.py',
              'r02_lai_simulation_material.py', 'r02_lai_support_review.py')


def rows(path, required=()):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)) and set(required) <= set(fields), 'Table columns differ')
        for row in reader:
            require(None not in row and None not in row.values(), 'Malformed table')
            yield row


def table(path, values):
    require(values, 'Empty output table')
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def integer(value):
    require(str(value).isdigit(), 'Expected nonnegative count')
    return int(value)


def truth(value):
    require(str(value) in ('True', 'False'), 'Expected explicit Boolean')
    return str(value) == 'True'


def keyed_people(path, expected):
    result = {}
    for row in rows(path, IDENTITY):
        sid = row['sample_id']
        require(sid not in result and sid in expected, 'Duplicate or unexpected donor')
        require(all(row[k] == expected[sid][k] for k in IDENTITY), 'Donor identity/unit/ancestry changed')
        result[sid] = row
    require(set(result) == set(expected), 'Incomplete donor inventory')
    return result


def evaluate_format(row):
    """Compatibility of the declared eligible calls, not quality or phase accuracy."""
    n = {k: integer(row[k]) for k in COUNTS}
    require(all(v <= n['records'] for v in n.values()) and
            n['homozygous_calls'] + n['heterozygous_calls'] == n['complete_diploid_calls'] and
            n['eligible_copyable_calls'] <= n['copyable_observed_calls'] <= n['complete_diploid_calls'] and
            n['complete_diploid_calls'] + max(n['missing_or_partial_calls'],n['non_diploid_calls']) <= n['records'] <=
            n['complete_diploid_calls'] + n['missing_or_partial_calls'] + n['non_diploid_calls'] and
            n['eligible_marker_calls'] <= n['records'] and
            n['eligible_copyable_calls'] + n['unusable_eligible_calls'] == n['eligible_marker_calls'] and
            n['phased_heterozygous_calls'] + n['unphased_heterozygous_calls'] == n['heterozygous_calls'] and
            n['heterozygous_calls_with_ps'] <= n['phased_heterozygous_calls'], 'Inconsistent call denominators')
    if n['eligible_marker_calls'] == 0 or n['eligible_copyable_calls'] == 0:
        return 'BLOCKED_NO_COPYABLE_TERRITORY'
    if n['unusable_eligible_calls']:
        return 'LIMITED_REQUIRES_EXPLICIT_CALL_MASK'
    return 'PASS_FORMAT_WITHIN_ELIGIBLE_TERRITORY'


def unique_phase(path, donors):
    result={}
    for row in rows(path, ('sample_id','chromosome')):
        sid=row['sample_id']
        require(sid in donors and sid not in result, 'Unknown/duplicate donor phase row')
        result[sid]=row
    require(set(result)==set(donors), 'Candidate phase membership differs')
    return result


class Admission:
    def __init__(self, contract, digest, output):
        self.started = time.monotonic()
        self.inputs = base.Inputs()
        self.cp = self.inputs.add(contract, digest)
        self.c = read(self.cp)
        require(self.c.get('schema') == SCHEMA, 'Unsupported admission contract')
        require(self.c['expected_chromosomes'] == [str(i) for i in range(1, 23)], 'All 22 autosomes required')
        require(self.c['policy'] == 'PRESERVE_M27F_CONDITIONAL_DEVELOPMENT_NO_SIMULATION', 'Unapproved policy')
        self.out = Path(output)
        require(not self.out.exists() and self.out.parent.is_dir(), 'Output exists; use Nextflow resume')
        r = self.c['resources']
        require(set(r) == {'cpus', 'max_output_bytes', 'min_free_gib', 'timeout_seconds', 'max_table_rows'},
                'Explicit resource bounds required')
        require(r['cpus'] == 1 and all(type(v) in (int, float) and v > 0 for v in r.values()), 'Invalid resources')
        base.support.check_space(self.out.parent, r['min_free_gib'])
        self.out.mkdir(mode=0o700)
        self.bundles = {}

    def guard(self):
        r = self.c['resources']
        base.guard(self.out, self.started, r['min_free_gib'], r['timeout_seconds'], r['max_output_bytes'])

    def bundle(self, desc, schema):
        path = self.inputs.descriptor(desc)
        m = read(path)
        require(m.get('schema') == schema and m.get('status', '').startswith('COMPLETE_'), 'Wrong source schema/status')
        require(m.get('source_test_opened', False) is False and
                m.get('source_test_genotypes_opened', False) is False, 'Reserved genotypes forbidden')
        self.bundles[str(path)] = m
        return path, m

    def artifact(self, source, name):
        path, m = source
        require(name == Path(name).name and name in m['outputs_sha256'], 'Unlisted/unsafe artifact')
        return self.inputs.add(path.parent/name, m['outputs_sha256'][name])

    def json_artifact(self, source, name='summary.json'):
        return read(self.artifact(source, name))


def role_inventory(task):
    roles = task.inputs.descriptor(task.c['roles'])
    policy = read(task.inputs.descriptor(task.c['m27f_policy']))
    require(policy['stage'] == 'M27F_BLIND_ROLE_SPLIT' and policy['version'] == 4 and
            policy['block_policy']['union_canonical_population'] is True and
            policy['block_policy']['max_segment_floor_cm'] == 10 and
            policy['block_policy']['kinship_floor'] == 0.04419417382415922, 'Frozen M27F policy differs')
    people = {}
    unit_roles, population_roles = {}, {}
    for row in rows(roles, IDENTITY):
        # Metadata may list other frozen roles; their genotypes are never inputs.
        if row['role'] not in ROLES:
            continue
        sid = row['sample_id']
        require(sid and sid not in people and row['ancestry'] in ANCESTRIES, 'Invalid role member')
        require(row['atomic_unit_id'] and row['canonical_population'], 'Missing historical unit')
        for field, store in [('atomic_unit_id', unit_roles), ('canonical_population', population_roles)]:
            require(store.get(row[field], row['role']) == row['role'], 'Historical unit/population crosses roles')
            store[row[field]] = row['role']
        people[sid] = {k: row[k] for k in IDENTITY}
    for role in ROLES:
        actual = Counter(p['ancestry'] for p in people.values() if p['role'] == role)
        require(dict(actual) == task.c['expected_role_counts'][role], 'Role/ancestry denominator differs')
    return people, policy


def dependency(task, people):
    source = task.bundle(task.c['dependency'], 'r02_lai_donor_dependency_v1')
    s = task.json_artifact(source)
    require(s['pc_counts'] == [4, 8, 12] and s['thresholds'] == [0.0221, 0.0442], 'Missing PC/phi sensitivity')
    require(s['n_samples'] == len(people) and s['pair_sensitivity']['all_pairs'] == len(people)*(len(people)-1)//2,
            'Kinship person/pair universe differs')
    records = defaultdict(dict)
    for row in rows(task.artifact(source, 'person_components.private.tsv'), ('sample_id', 'pc', 'phi')):
        sid = row['sample_id']; scenario = (int(row['pc']), float(row['phi']))
        require(sid in people and scenario in SCENARIOS and scenario not in records[sid], 'Invalid dependency membership')
        require(row['role'] == people[sid]['role'] and row['ancestry'] == people[sid]['ancestry'] and
                row['population'] == people[sid]['canonical_population'], 'Dependency person metadata differs')
        if people[sid]['role'] == 'SOURCE_VALID':
            require(row['declared_atomic_unit'] == people[sid]['atomic_unit_id'], 'Dependency replaced historical unit')
        records[sid][scenario] = row
    require(set(records) == set(people) and all(set(v) == SCENARIOS for v in records.values()), 'Incomplete six-scenario inventory')
    review = task.bundle(task.c['kinship_review'], 'r02_lai_donor_review_private_v1')
    rs = task.json_artifact(review)
    require(rs['pc_counts'] == [4,8,12] and rs['thresholds'] == [0.0221,0.0442] and rs['people'] == len(people),
            'Review scenario/universe differs')
    selected = {}
    for row in rows(task.artifact(review, 'pair_review.private.tsv'), ('sample_a', 'sample_b')):
        if row['role_pair'] == 'REF_TRAIN--SOURCE_VALID' and row['status_phi_00221'] == 'PC_SENSITIVE':
            pair = tuple(sorted((row['sample_a'],row['sample_b'])))
            require(pair not in selected and all(sid in people for sid in pair), 'Duplicate/unknown sensitive pair')
            selected[pair] = row
    ibd = task.bundle(task.c['ibd_review'], 'r02_lai_external_ibd_pairs_private_v1')
    bs = task.json_artifact(ibd)
    require(bs['authenticated_ibd'] == bs['authenticated_maps'] == bs['authenticated_logs'] == 22,
            'Incomplete external IBD source coverage')
    pairs = {}
    by_donor = defaultdict(Counter)
    for row in rows(task.artifact(ibd, 'pairs.private.tsv'), ('sample_a','sample_b','m27f_primary_edge')):
        pair = tuple(sorted((row['sample_a'],row['sample_b'])))
        require(pair in selected and pair not in pairs, 'Missing/duplicate/unexpected IBD diagnostic pair')
        for pc in (4,8,12):
            require(float(row[f'kin_pc{pc}']) == float(selected[pair][f'kin_pc{pc}']), 'IBD/PC pair differs')
        require(integer(row['source_autosomes_authenticated']) == 22, 'Incomplete IBD source coverage')
        require(row['sample_a_original_unit'] == people[row['sample_a']]['atomic_unit_id'] and
                row['sample_b_original_unit'] == people[row['sample_b']]['atomic_unit_id'], 'IBD original unit differs')
        primary = truth(row['m27f_primary_edge'])
        historical = (float(row['max_reported_cm']) >= 10 and
                      float(row['m27e_reported_phi']) >= 0.04419417382415922)
        require(primary == historical, 'Historical edge flag does not match fixed M27F rule')
        sid = next(s for s in pair if people[s]['role'] == 'SOURCE_VALID')
        by_donor[sid]['reviewed_pairs'] += 1
        by_donor[sid]['reported_pairs'] += int(integer(row['reported_records']) > 0)
        by_donor[sid]['historical_cross_role_edges'] += int(primary)
        pairs[pair] = row
    require(set(pairs) == set(selected) and len(pairs) == bs['pairs'], 'IBD review not complete for selected pairs')
    require(sum(v['historical_cross_role_edges'] for v in by_donor.values()) == bs['pairs_m27f_primary'], 'IBD summary differs')
    return records, s, bs, by_donor


def chromosome(task, job, donors, people, dep, ibd_people, maps):
    chrom = job['chromosome']
    full = task.bundle(job['full_audit'], 'r02_lai_full_chromosome_audit_v1')
    readiness = task.bundle(job['readiness'], 'r02_lai_donor_readiness_v1')
    copies = task.bundle(job['donor_haplotypes'], 'r02_lai_donor_haplotypes_v1')
    decision = task.bundle(job['material_decision'], 'r02_lai_material_decision_v1')
    fs, rs, cs = (task.json_artifact(x) for x in (full, readiness, copies))
    dm = decision[1]
    for obj in (full[1], readiness[1], copies[1], dm):
        require(obj['chromosome'] == chrom, 'Chromosome mismatch')
    require(fs['donors'] == rs['distinct_donors'] == cs['distinct_donors'] == len(donors), 'Donor denominator differs')
    require(rs['candidate_loci'] == cs['candidate_loci'] == dm['conditional_candidate_loci']['ALL'], 'Candidate denominator differs')
    require(dm['roles_sha256'] == task.c['roles']['sha256'], 'Decision used different roles')
    for source in (full, copies):
        require(task.c['roles']['sha256'] in source[1]['inputs_sha256'].values() and
                rs['source_panel_sha256'] in source[1]['inputs_sha256'].values(),
                'Full-marker and candidate evidence do not authenticate the same role/VCF bytes')
    for role in ROLES:
        require(dm['role_members'][role] == dict(task.c['expected_role_counts'][role],
                    ALL=sum(task.c['expected_role_counts'][role].values())), 'L1 ancestry/role count differs')
    m = maps[chrom]
    require(dm['map_sha256'] == m['local_sha256'] and
            (fs['map_first_bp'],fs['map_last_bp']) == (m['first_bp_cm'][0],m['last_bp_cm'][0]) and
            (dm['map_first_bp'],dm['map_last_bp']) == (fs['map_first_bp'],fs['map_last_bp']), 'Map identity/endpoints differ')
    require(m['ordered_numeric_equal'] is True and m['mismatching_ordered_tuples'] == 0 and
            m['nonincreasing_bp'] == m['decreasing_cm'] == 0, 'Map audit failed')
    audit_people = keyed_people(task.artifact(full,'people.private.tsv'), donors)
    support = keyed_people(task.artifact(copies,'donor_support.private.tsv'), donors)
    phase = unique_phase(task.artifact(readiness,'donor_phase.private.tsv'), donors)
    counts = Counter(); result = []
    for sid in sorted(donors):
        row, rare, rp = audit_people[sid], support[sid], phase[sid]
        require(rare['chromosome'] == rp['chromosome'] == chrom and
                integer(rare['candidate_loci']) == integer(rp['candidate_calls']) == cs['candidate_loci'], 'Candidate inventory differs')
        require(integer(rare['complete_calls']) == integer(rp['complete_calls']) and
                integer(rare['missing_calls']) == integer(rp['missing_or_partial_calls']) and
                integer(rare['non_diploid_calls']) == integer(rp['non_diploid_calls']), 'Candidate call evidence differs')
        format_status = evaluate_format(row)
        n = {k:integer(row[k]) for k in COUNTS}; counts.update(n)
        sensitive = [f'PC{pc}_phi{phi}' for pc,phi in sorted(SCENARIOS)
                     if integer(dep[sid][(pc,phi)]['n_references'])]
        unknown = max(integer(dep[sid][k]['nonfinite_pairs']) for k in SCENARIOS)
        cross = ibd_people[sid]['historical_cross_role_edges']
        separation = 'BLOCKED_HISTORICAL_CROSS_ROLE_IBD' if cross else 'LIMITED_HISTORICAL_UNITS_PRESERVED'
        if unknown:
            separation = 'UNRESOLVED_NONFINITE_KINSHIP'
        admission = ('LIMITED_DEVELOPMENT_MATERIAL' if format_status.startswith('PASS') and not cross and not unknown
                     else 'UNRESOLVED_MATERIAL_MASK_OR_SEPARATION')
        result.append(dict(**donors[sid], chromosome=chrom, **n,
            excluded_marker_calls=n['records']-n['eligible_marker_calls'],
            candidate_rare_loci=integer(rare['candidate_loci']), rare_carrier_loci=integer(rare['carrier_loci']),
            observed_rare_copies=integer(rare['observed_carrier_copies']),
            genotype_format=format_status, quality='UNRESOLVED_DP_GQ_NOT_MEASURED',
            phase_accuracy='UNRESOLVED_NOT_MEASURED', ps_absence_is_automatic_failure=False,
            supplied_phase_provenance=fs['phase_provenance']['status'], historical_separation=separation,
            reference_connected_scenarios=','.join(sensitive) or 'NONE_AT_EXAMINED_CUTS',
            direct_ibd_reviewed_pairs=ibd_people[sid]['reviewed_pairs'],
            direct_ibd_reported_pairs=ibd_people[sid]['reported_pairs'],
            historical_cross_role_ibd_edges=cross, nonfinite_kinship_pairs=unknown,
            admission=admission, simulation_authorized=False))
    for key in COUNTS:
        require(counts[key] == fs['calls'][key], 'Full audit person/call aggregate differs')
    require(counts['records'] == fs['markers']['records']*len(donors), 'Marker/call denominator differs')
    require(sum(integer(r['observed_carrier_copies']) for r in support.values()) == cs['observed_carrier_copy_rows'],
            'Rare copy denominator differs')
    decision_rows = list(rows(task.artifact(decision,'decision_summary.tsv'), ('chromosome','panel_ancestry','category','loci')))
    for row in decision_rows:
        require(row['chromosome'] == chrom and row['panel_ancestry'] in ('ALL',*ANCESTRIES) and
                integer(row['source_loci_denominator']) == dm['source_loci'], 'L1 summary denominator differs')
    ancestry_rows = []
    for ancestry in ('ALL',*ANCESTRIES):
        subset = [r for r in result if ancestry == 'ALL' or r['ancestry'] == ancestry]
        carriers = [r for r in subset if r['rare_carrier_loci'] > 0]
        refs = [r for r in people.values() if r['role']=='REF_TRAIN' and (ancestry=='ALL' or r['ancestry']==ancestry)]
        ancestry_rows.append(dict(chromosome=chrom, ancestry=ancestry, donors=len(subset),
            declared_donor_units=len({r['atomic_unit_id'] for r in subset}),
            declared_donor_populations=len({r['canonical_population'] for r in subset}),
            references=len(refs), declared_reference_units=len({r['atomic_unit_id'] for r in refs}),
            rare_carrier_donors=len(carriers), declared_rare_carrier_units=len({r['atomic_unit_id'] for r in carriers}),
            candidate_rare_loci=dm['conditional_candidate_loci'].get(ancestry,0),
            source_rare_loci_denominator=dm['source_loci'],
            full_marker_records=fs['markers']['records'], eligible_marker_records=fs['markers']['eligible_markers'],
            eligible_copyable_calls=sum(r['eligible_copyable_calls'] for r in subset),
            eligible_marker_calls_denominator=sum(r['eligible_marker_calls'] for r in subset),
            phase_accuracy='NOT_MEASURED', effective_independent_donors='UNKNOWN',
            interpretation='Conditional on supplied populations and haplotypes; not ancestry-wide validation'))
    task.guard()
    return result, ancestry_rows, decision_rows, fs, rs


def run(contract, digest, output):
    task = Admission(contract,digest,output)
    people, policy = role_inventory(task)
    donors = {s:r for s,r in people.items() if r['role']=='SOURCE_VALID'}
    dep, ds, ibd, ibd_people = dependency(task,people)
    map_audit = read(task.inputs.descriptor(task.c['map_audit']))
    require(map_audit['all_22_ordered_numeric_equal'] is True, 'Map audit incomplete')
    maps = {m['chromosome']:m for m in map_audit['maps']}
    require(len(maps)==len(map_audit['maps'])==22 and set(maps)==set(task.c['expected_chromosomes']), 'Map inventory differs')
    jobs = task.c['jobs']
    require([j['chromosome'] for j in jobs] == task.c['expected_chromosomes'], 'Missing/duplicate/out-of-order chromosome')
    donor_rows=[]; ancestry_rows=[]; l1_rows=[]; total_calls=Counter(); marker_count=0; candidate_count=0
    for job in jobs:
        d,a,l,fs,rs = chromosome(task,job,donors,people,dep,ibd_people,maps)
        donor_rows.extend(d); ancestry_rows.extend(a); l1_rows.extend(l)
        total_calls.update({k:fs['calls'][k] for k in COUNTS})
        marker_count += fs['markers']['records']; candidate_count += rs['candidate_loci']
        require(len(donor_rows)+len(ancestry_rows)+len(l1_rows) <= task.c['resources']['max_table_rows'], 'Table budget exceeded')
    fa = task.bundle(task.c['full_aggregate'],'r02_lai_full_chromosome_audit_aggregate_v1')
    ra = task.bundle(task.c['readiness_aggregate'],'r02_lai_donor_readiness_aggregate_v1')
    fas,ras = task.json_artifact(fa),task.json_artifact(ra)
    require(fas['chromosomes']==22 and fas['donors']==len(donors) and fas['marker_records']==marker_count and
            all(fas['calls'][k]==total_calls[k] for k in COUNTS), 'Full aggregate differs')
    require(ras['chromosomes']==task.c['expected_chromosomes'] and ras['candidate_loci']==candidate_count and
            ras['distinct_donors']==len(donors), 'Readiness aggregate differs')
    unit_rows=[]
    for role in ROLES:
        for ancestry in ANCESTRIES:
            subset=[r for r in people.values() if r['role']==role and r['ancestry']==ancestry]
            unit_rows.append(dict(role=role,ancestry=ancestry,people=len(subset),
                declared_units=len({r['atomic_unit_id'] for r in subset}),
                declared_populations=len({r['canonical_population'] for r in subset}),
                biological_independence='NOT_CERTIFIED',exposure='HISTORICALLY_EXPOSED_DEVELOPMENT'))
    nam=[r for r in donors.values() if r['ancestry']=='Native_American']
    require(Counter(r['canonical_population'] for r in nam)==task.c['expected_nam_populations'] and
            len({r['atomic_unit_id'] for r in nam})==task.c['expected_nam_units'], 'NAM original support differs')
    sensitivity=[]
    for pc,phi in sorted(SCENARIOS):
        for ancestry in ('ALL',*ANCESTRIES):
            ids=[s for s,r in donors.items() if ancestry=='ALL' or r['ancestry']==ancestry]
            sensitivity.append(dict(pc=pc,phi=phi,ancestry=ancestry,donors=len(ids),
                reference_connected_donors=sum(integer(dep[s][(pc,phi)]['n_references'])>0 for s in ids),
                nonfinite_pair_donors=sum(integer(dep[s][(pc,phi)]['nonfinite_pairs'])>0 for s in ids),
                pc_components=len({dep[s][(pc,phi)]['component_id'] for s in ids}),
                historical_units=len({donors[s]['atomic_unit_id'] for s in ids}),
                selected_for_admission=False))
    criteria=[]
    def criterion(identifier,state,numerator,denominator,measure,conclusion,next_action):
        criteria.append(dict(criterion=identifier,state=state,numerator=numerator,denominator=denominator,
            measured_quantity=measure,allowed_conclusion=conclusion,next_action=next_action))
    criterion('IDENTITY_ROLES','PASS',len(people),len(people),'authenticated permitted people',
              'M27F roles and original population-plus-IBD units preserved','No reassignment; SOURCE_TEST genotypes remain closed')
    criterion('ALL_AUTOSOMES','PASS',22,22,'chromosomes with reconciled evidence','All completed sources integrated','Use source manifests for later extraction')
    eligible=total_calls['eligible_marker_calls']; copyable=total_calls['eligible_copyable_calls']
    criterion('COPYABLE_ELIGIBLE_GT','PASS' if copyable==eligible and eligible else 'LIMITED',copyable,eligible,
              'complete homozygous or phased heterozygous eligible calls','Supplied calls readable as copies within declared territory',
              'Apply inherited marker policy and explicit missing-call mask before simulation; do not change originals')
    criterion('PHASE_FORMAT','PASS' if total_calls['unphased_heterozygous_calls']==0 else 'LIMITED',
              total_calls['phased_heterozygous_calls'],total_calls['heterozygous_calls'],'heterozygotes with supplied phase notation',
              'Observed format, not phase accuracy','Retain supplied phase and exact provenance; do not rephase automatically')
    criterion('PHASE_ACCURACY','UNRESOLVED','NA','NA','switch error/phase accuracy not measured',
              'No validated phase accuracy claim; absence of PS is not automatic failure','Use an explicit supplied-phase development scope; prospective phase sensitivity before generalization')
    criterion('GENOTYPE_QUALITY','UNRESOLVED','NA','NA','DP/GQ unavailable in these GT-only projections',
              'Not certified high-quality calls; unfiltered is not PASS','Retain GT_ONLY and inherited filter policy; do not invent thresholds or quality values')
    criterion('MAP_IDENTITY','PASS',22,22,'numeric equality to authenticated GRCh38 maps',
              'Known coordinate domain, no silent extrapolation','Mask out-of-map markers; map is not ancestry-aware')
    criterion('KINSHIP_SENSITIVITY','LIMITED',6,6,'PC4/8/12 by phi0.0221/0.0442 retained',
              'Sensitivity documented; no PC chosen by fewer links','Preserve each scenario and original M27F units')
    criterion('HISTORICAL_SPLIT_COMPATIBILITY','LIMITED' if ibd['pairs_m27f_primary']==0 else 'UNRESOLVED',
              ibd['pairs_m27f_primary'],ibd['pairs'],'selected cross-role pairs meeting fixed historical segment rule',
              'No new edge under that rule does not prove independence; IBD callability unknown','Preserve original units unless contradictory evidence is found')
    criterion('NAM_REPRESENTATION','LIMITED',len({r['atomic_unit_id'] for r in nam}),len(nam),
              'historical population-plus-IBD units / donor people; not a success fraction',
              'Karitiana/Zapotec-conditional development only, not all Brazilian NAM','Report units separately; synthetic mosaics do not increase independent donor diversity')
    criterion('RARE_SUPPORT','LIMITED',candidate_count,sum(integer(r['source_rare_loci_denominator']) for r in ancestry_rows if r['ancestry']=='ALL'),
              'source DNABR rare loci with conditional observed donor copies',
              'Simulation support for represented alleles only; absent record is not zero','Retain reference-present, zero-observed, unknown and no-record categories separately')
    criterion('INCREMENTAL_REFERENCE_UNIT_SUPPORT','UNRESOLVED','NA','NA','current baseline-excluded rare loci across reference/donor units',
              'Aggregate reference counts do not identify reference carrier units or incremental baseline exclusion',
              'Freeze exact baseline markers and authenticate unit-level reference carriage before claiming incremental NAM support; do not import historical thresholds silently')
    criterion('CONFIRMATORY_EXPOSURE','LIMITED',0,len(donors),'untouched donors established by this evidence',
              'SOURCE_VALID historically exposed; no confirmatory holdout created','Use only development conclusions; independent material required for external confirmation')
    criterion('SIMULATION_DESIGN','UNRESOLVED','NA','NA','no demographic scenario or new simulation contract approved here',
              'Admission report is not execution authorization or proof of biological accuracy','Before simulation specify demographic model, copied haplotypes, truth provenance and paired evaluation')
    for name,values in [('donor_chromosome_admission.private.tsv',donor_rows),('ancestry_chromosome_admission.tsv',ancestry_rows),
                        ('l1_support_categories.tsv',l1_rows),('declared_units.tsv',unit_rows),
                        ('kinship_sensitivity.tsv',sensitivity),('criteria.tsv',criteria)]:
        table(task.out/name,values)
    summary=dict(chromosomes=22,donors=len(donors),references=len(people)-len(donors),
        donor_chromosome_rows=len(donor_rows),marker_records=marker_count,calls=dict(total_calls),
        candidate_rare_loci=candidate_count,nam_people=len(nam),nam_units=len({r['atomic_unit_id'] for r in nam}),
        nam_populations=dict(Counter(r['canonical_population'] for r in nam)),
        admission_states=dict(Counter(r['admission'] for r in donor_rows)),
        genotype_format_states=dict(Counter(r['genotype_format'] for r in donor_rows)),
        criteria_states=dict(Counter(r['state'] for r in criteria)),ibd_review=ibd,
        interpretation='Conditional development material, not independent ancestry truth or confirmatory LAI validation')
    limitations=dict(simulations=False,training=False,simulation_authorized=False,source_test_genotypes_opened=False,
        roles_changed=False,donor_units_changed=False,phase_validated=False,genotype_quality_validated=False,
        independence_certified=False,public_distribution_allowed=False,contains_sample_identifiers=True,
        contains_individual_genotypes=False,pc_configuration_selected=False,
        missing_genotype_is_reference=False,source_valid_is_untouched_test=False)
    write(task.out/'summary.json',summary); write(task.out/'limitations.json',limitations)
    task.guard();task.inputs.unchanged()
    result=dict(schema=OUTPUT_SCHEMA,status=STATUS,evidence_level='CONDITIONAL_MATERIAL_FEASIBILITY',**limitations,
        contract_sha256=sha(task.cp),summary=summary,
        inputs_sha256={p:v[2] for p,v in task.inputs.entries.items()},
        outputs_sha256={p.name:sha(p) for p in sorted(task.out.iterdir()) if p.is_file()},
        code_sha256={n:sha(Path(__file__).parent/n) for n in CODE_FILES},
        elapsed_seconds=time.monotonic()-task.started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(task.out/'manifest.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True);parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    os.umask(0o077)
    result=run(args.contract,args.contract_sha256,args.output_dir)
    print(result['status'])


if __name__=='__main__':
    main()
