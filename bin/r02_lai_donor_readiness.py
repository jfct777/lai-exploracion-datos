#!/usr/bin/env python3
"""Integrate sealed donor copies with existing PC-Relate; never assign a split.

All outputs are private preparation. Components are thresholded diagnostics,
not certified families. A pipe/PS at candidate sites is not chromosome-wide
phase continuity. No genotype, reserved role or truth of ancestry is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import itertools
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import r02_lai_material_decision as base

require, sha, read, write = base.require, base.sha, base.read, base.write
SCHEMA = 'r02_lai_donor_readiness_input_v1'
STAGES = {
    'dependency': ('r02_lai_donor_dependency_v1', 'COMPLETE_PRIVATE_DONOR_DEPENDENCY_DIAGNOSTIC_NOT_SPLIT'),
    'chromosome': ('r02_lai_donor_readiness_v1', 'COMPLETE_PRIVATE_DONOR_READINESS_NOT_SIMULATION_READY'),
    'aggregate': ('r02_lai_donor_readiness_aggregate_v1', 'COMPLETE_AUTOSOMAL_DONOR_READINESS_NOT_SIMULATION_READY'),
}
CODE_FILES = ('r02_lai_donor_readiness.py', 'r02_lai_material_decision.py',
              'r02_lai_simulation_material.py', 'r02_lai_support_review.py')
KEY = ('chromosome', 'position_bp', 'REF', 'ALT', 'counted_allele')
PCS = (4, 8, 12)
DONOR = 'SOURCE_VALID'
REFERENCE = 'REF_TRAIN'


def rows(path, required=()):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)) and set(required) <= set(fields), 'Missing/duplicate table columns')
        for row in reader:
            require(None not in row and None not in row.values(), 'Malformed table row')
            yield row


def write_rows(path, fields, values):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'xt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def boolean(value):
    require(value in ('True', 'False', 'TRUE', 'FALSE'), 'Invalid explicit Boolean')
    return value in ('True', 'TRUE')


def number(value):
    if value in ('NA', 'NaN', 'nan', 'Inf', '-Inf', ''):
        return float('nan')
    return float(value)


def limitations():
    return dict(simulation_ready=False, new_donor_split_assigned=False, donors_removed=False,
        source_test_opened=False, simulations=False, training=False, phase_validated=False,
        independence_certified=False, ancestry_relabelled=False,
        component_scope='Thresholded PC-Relate connected components, not certified families or independent donors.',
        role_scope='SOURCE_VALID is historically exposed development material, not untouched evaluation.',
        ancestry_scope='Historical individual label, not validated local ancestry of a copied allele.',
        phase_scope='Candidate-site observed copy index/PS only; no chromosome-wide continuity or accuracy certification.',
        full_marker_haplotypes_audited=False,
        remaining_conditions=['Choose donor-unit separation policy before assigning a simulation split.',
            'Resolve reference-connected and unknown-relatedness donor units under that policy.',
            'Audit full unpruned simulation-marker haplotypes, missingness and phase continuity.',
            'Justify demographic scenarios and uncertainty handling before simulation.'])


class Task:
    def __init__(self, contract, digest, outdir):
        self.started = time.monotonic()
        self.inputs = base.Inputs()
        self.contract_path = self.inputs.add(contract, digest)
        self.c = read(self.contract_path)
        require(self.c.get('schema') == SCHEMA, 'Unsupported readiness contract')
        require(self.c.get('pc_counts') == list(PCS), 'Preserve all three PC sensitivities')
        require(self.c.get('thresholds') == [0.0221, 0.0442], 'Only explicitly inherited P06 diagnostic thresholds')
        criteria = self.inputs.descriptor(self.c['criteria'])
        require(any(row['parametro_id'] == 'P06' for row in rows(criteria, ('parametro_id',)))
                if 'parametro_id' in criteria.read_text().splitlines()[0].split('\t')
                else '\nP06\t' in criteria.read_text(), 'P06 provenance is required')
        self.out = Path(outdir)
        require(not self.out.exists() and self.out.parent.is_dir(), 'Output exists or parent missing; use Nextflow resume')
        r = self.c['resources']
        require(set(r) == {'cpus', 'max_samples', 'max_pairs', 'max_private_rows', 'max_output_bytes',
                          'min_free_gib', 'timeout_seconds'}, 'Explicit resource bounds required')
        require(r['cpus'] == 1 and all(type(v) in (int, float) and v > 0 for v in r.values()), 'Invalid resource limits')
        base.support.check_space(self.out.parent, r['min_free_gib'])
        self.out.mkdir(mode=0o700)

    def guard(self):
        r = self.c['resources']
        base.guard(self.out, self.started, r['min_free_gib'], r['timeout_seconds'], r['max_output_bytes'])

    def manifest(self, descriptor, schema, status):
        p = self.inputs.descriptor(descriptor)
        value = read(p)
        require(value.get('schema') == schema and value.get('status') == status, 'Wrong input schema/status')
        require(value.get('source_test_opened') is not True, 'Reserved-role input forbidden')
        return p, value

    def artifact(self, p, manifest, name):
        require(name in manifest['outputs_sha256'], 'Unlisted input artifact')
        return self.inputs.add(p.parent/name, manifest['outputs_sha256'][name])

    def seal(self, stage, summary, **extra):
        self.guard()
        self.inputs.unchanged()
        write(self.out/'summary.json', summary)
        write(self.out/'definitions.json', limitations())
        schema, status = STAGES[stage]
        result = dict(schema=schema, status=status, evidence_level='DESCRIPTIVO_PREPARACION',
            **limitations(), **extra, contract_sha256=sha(self.contract_path),
            inputs_sha256={name:info[2] for name,info in self.inputs.entries.items()},
            outputs_sha256={p.name:sha(p) for p in sorted(self.out.iterdir()) if p.is_file()},
            code_sha256={name:sha(Path(__file__).parent/name) for name in CODE_FILES},
            summary=summary, elapsed_seconds=time.monotonic()-self.started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(self.out/'manifest.json', result)
        return result


def read_people(task):
    p, m = task.manifest(task.c['pcair'], 'r02_lai_panel_pcair_v1',
                         'COMPLETE_PCAIR_CHECKPOINT_NOT_DONOR_INDEPENDENCE')
    people = {}
    for row in rows(task.artifact(p,m,'pcair_roles.private.tsv'),
                    ('sample_id','role','ancestry','population','pcair_status')):
        sid = row['sample_id']
        require(sid and sid not in people and row['role'] in (DONOR,REFERENCE), 'Duplicate or reserved person')
        require(row['pcair_status'] in ('FIT_UNRELATED','PROJECTED_RELATED'), 'Unknown PC-AiR status')
        people[sid] = row
    require(len(people) == m['n_samples'] <= task.c['resources']['max_samples'], 'Person denominator/budget differs')
    ap, am = task.manifest(task.c['donor_aggregate'], 'r02_lai_donor_haplotypes_aggregate_v1',
        'COMPLETE_AUTOSOMAL_PRIVATE_DONOR_PREPARATION_NOT_SIMULATION_READY')
    donor_rows = list(rows(task.artifact(ap,am,'donors.private.tsv'),
        ('sample_id','role','ancestry','atomic_unit_id','canonical_population')))
    donors = {r['sample_id']:r for r in donor_rows}
    require(len(donors) == len(donor_rows) and set(donors) == {s for s,r in people.items() if r['role']==DONOR},
            'Exact donor membership differs')
    for sid, row in donors.items():
        require(row['role']==DONOR and row['ancestry']==people[sid]['ancestry'] and
                row['canonical_population']==people[sid]['population'], 'Historical ancestry/population differs')
    return people, donors, m


def pair_matrix(path, people, maximum):
    samples = list(people); index = {s:i for i,s in enumerate(samples)}; n = len(samples)
    require(n*(n-1)//2 <= maximum, 'Pair budget exceeded')
    values = np.full((n,n), np.nan)
    finite = np.zeros((n,n),dtype=bool); seen = np.zeros((n,n),dtype=bool)
    count = 0
    for row in rows(path, ('ID1','ID2','kin','k0','k2','role1','role2','estimate_finite')):
        a,b = row['ID1'],row['ID2']
        require(a in index and b in index and a != b, 'Pair outside exact role universe')
        i,j = index[a],index[b]
        require(not seen[i,j], 'Duplicate/reversed pair')
        require(row['role1']==people[a]['role'] and row['role2']==people[b]['role'], 'Pair role differs')
        v = [number(row[k]) for k in ('kin','k0','k2')]
        valid = all(math.isfinite(x) for x in v)
        require(valid == boolean(row['estimate_finite']), 'Finite estimate flag differs')
        seen[i,j] = seen[j,i] = True
        values[i,j] = values[j,i] = v[0]
        finite[i,j] = finite[j,i] = valid
        count += 1
        require(count<=maximum, 'Pair budget exceeded')
    require(count == n*(n-1)//2, 'Incomplete pair inventory; absent is not unrelated')
    return values,finite


def dependency_components(samples, people, matrix, finite, threshold):
    adjacency = finite & (matrix >= threshold)
    np.fill_diagonal(adjacency,False)
    _, labels = connected_components(csr_matrix(adjacency),directed=False)
    groups = defaultdict(list)
    for sid,label in zip(samples,labels): groups[int(label)].append(sid)
    members, details = {}, {}
    # IDs use exactly the project helper's deterministic sorted membership hash.
    import hashlib
    for group in groups.values():
        digest = hashlib.sha256(('\n'.join(sorted(group))+'\n').encode()).hexdigest()[:20]
        component = 'dep_'+digest
        counts = Counter(people[s]['role'] for s in group)
        details[component] = dict(component_id=component, n_people=len(group),
            n_donors=counts[DONOR], n_references=counts[REFERENCE])
        for sid in group: members[sid] = component
    return members,details


def dependency(task):
    people,donors,pm = read_people(task)
    samples = list(people); n=len(samples); values={}; finite={}; descriptions=[]; assignments=[]
    pair_summaries=[]
    for pc in PCS:
        p,m=task.manifest(task.c['pcrelate'][str(pc)], 'r02_lai_panel_pcrelate_v1',
            'COMPLETE_PCRELATE_DESCRIPTIVE_NOT_DONOR_INDEPENDENCE')
        summary=read(task.artifact(p,m,'summary.json'))
        require(m['input_sha256']==pm['input_sha256'] and m['n_samples']==n and
                m['n_markers']==pm['n_markers'] and summary['n_pcs']==pc and
                summary['checkpoint_sha256']==read(task.artifact(Path(task.c['pcair']['path']),pm,'summary.json'))['checkpoint_sha256'],
                'PC sensitivity inputs/checkpoint differ')
        matrix,valid=pair_matrix(task.artifact(p,m,'pcrelate_pairs.private.tsv.gz'),people,task.c['resources']['max_pairs'])
        require(int(np.triu(valid,1).sum())==summary['n_pairs_finite'], 'Finite-pair denominator differs')
        values[pc],finite[pc]=matrix,valid
        for threshold in task.c['thresholds']:
            members,details=dependency_components(samples,people,matrix,valid,threshold)
            for comp in details.values(): descriptions.append(dict(pc=pc,phi=threshold,**comp))
            groups=defaultdict(Counter)
            for i,sid in enumerate(samples):
                d=details[members[sid]]
                row=dict(pc=pc,phi=threshold,**people[sid],**d,
                    nonfinite_pairs=int((~valid[i]).sum())-1,
                    declared_atomic_unit=donors.get(sid,{}).get('atomic_unit_id','NA'))
                assignments.append(row)
            for i,a in enumerate(samples):
                for j in range(i+1,n):
                    b=samples[j]; key=tuple(sorted((people[a]['role'],people[b]['role'])))
                    groups[key]['all_pairs']+=1
                    groups[key]['finite_pairs']+=int(valid[i,j])
                    groups[key]['above_cut_pairs']+=int(valid[i,j] and matrix[i,j]>=threshold)
            for (ra,rb),counts in sorted(groups.items()):
                pair_summaries.append(dict(pc=pc,phi=threshold,role_a=ra,role_b=rb,**counts))
        task.guard()
    write_rows(task.out/'person_components.private.tsv',tuple(assignments[0]),assignments)
    write_rows(task.out/'component_summary.private.tsv',tuple(descriptions[0]),descriptions)
    write_rows(task.out/'pair_role_summary.tsv',tuple(pair_summaries[0]),pair_summaries)
    sensitivity=Counter()
    fields=('sample_a','sample_b','role_a','role_b','kin_pc4','kin_pc8','kin_pc12','kin_span',
            'finite_all_pcs','crosses_phi_0.0221','crosses_phi_0.0442')
    def pairs():
        for i,a in enumerate(samples):
            for j in range(i+1,n):
                vs=[values[pc][i,j] for pc in PCS]; fs=[bool(finite[pc][i,j]) for pc in PCS]
                item=dict(sample_a=a,sample_b=samples[j],role_a=people[a]['role'],role_b=people[samples[j]]['role'],
                    **{f'kin_pc{pc}':float(v) if math.isfinite(v) else 'NA' for pc,v in zip(PCS,vs)},
                    kin_span=float(max(vs)-min(vs)) if all(fs) else 'NA',finite_all_pcs=all(fs))
                sensitivity['all_pairs']+=1; sensitivity['finite_all_pcs']+=int(all(fs))
                for threshold in task.c['thresholds']:
                    key=f'crosses_phi_{threshold:g}'
                    item[key]=len({v>=threshold for v in vs})>1 if all(fs) else 'NA'
                    sensitivity[key]+=int(item[key] is True)
                yield item
    write_rows(task.out/'pair_sensitivity.private.tsv.gz',fields,pairs())
    representation=Counter((r['role'],r['ancestry'],r['population'],r['pcair_status']) for r in people.values())
    write_rows(task.out/'pcair_representation.tsv',('role','ancestry','population','pcair_status','n_people'),
        (dict(zip(('role','ancestry','population','pcair_status','n_people'),(*key,count)))
         for key,count in sorted(representation.items())))
    return task.seal('dependency',dict(n_samples=n,n_donors=len(donors),n_references=n-len(donors),
        n_markers=pm['n_markers'],pc_counts=list(PCS),thresholds=task.c['thresholds'],
        pair_sensitivity=dict(sensitivity),component_scenarios=[dict(pc=pc,phi=t,
            n_components=sum(r['pc']==pc and r['phi']==t for r in descriptions),
            reference_connected_donors=sum(r['pc']==pc and r['phi']==t and r['role']==DONOR and r['n_references']>0 for r in assignments))
            for pc in PCS for t in task.c['thresholds']]))


def dependency_rows(task,path):
    p=Path(path); m=read(p)
    require((m.get('schema'),m.get('status'))==STAGES['dependency'] and
            m['contract_sha256']==sha(task.contract_path), 'Dependency result contract differs')
    task.inputs.add(p,sha(p))
    result=defaultdict(list)
    for row in rows(task.artifact(p,m,'person_components.private.tsv'),('sample_id','pc','phi','component_id','n_references','nonfinite_pairs')):
        for key in ('pc','n_people','n_donors','n_references','nonfinite_pairs'): row[key]=int(row[key])
        row['phi']=float(row['phi']); result[row['sample_id']].append(row)
    for values in result.values():
        require({(r['pc'],r['phi']) for r in values}=={(pc,t) for pc in PCS for t in task.c['thresholds']}
                and len(values)==6, 'Missing/duplicate person dependency scenario')
    return result


def chromosome(task,chrom,dependency_manifest):
    job=next((j for j in task.c['jobs'] if j['chromosome']==chrom),None)
    require(job is not None, 'Undeclared chromosome')
    p,m=task.manifest(job['donor_manifest'],'r02_lai_donor_haplotypes_v1',
        'COMPLETE_PRIVATE_DONOR_HAPLOTYPES_NOT_SIMULATION_READY')
    lp,lm=task.manifest(job['decision_manifest'],base.SCHEMA,base.STATUS)
    sp,sm=task.manifest(job['source_manifest'],'r02_lai_genotype_support_v1',
        'COMPLETE_GENOTYPE_AUDIT_NOT_PHASE_OR_BIOLOGICAL_VALIDATION')
    require(m['chromosome']==lm['chromosome']==sm['chromosome']==chrom and
        m['inputs_sha256'].get(str(lp))==sha(lp) and m['inputs_sha256'].get(str(sp))==sha(sp), 'L1/L2 provenance differs')
    require(sm['genotype_contract']['panel_role']==DONOR and sm['build_status']=='BUILD_VERIFIED_BY_SUPPLIED_EVIDENCE',
            'Wrong donor role or unverified build')
    deps=dependency_rows(task,dependency_manifest)
    donors={row['sample_id']:row for row in rows(task.artifact(p,m,'donors.private.tsv'),('sample_id','role','ancestry'))}
    require(set(donors)=={s for s,r in deps.items() if r[0]['role']==DONOR}, 'Donor universe differs')
    require(all(donors[s]['ancestry']==deps[s][0]['ancestry'] for s in donors),'Donor ancestry changed')
    stats={s:Counter() for s in donors}; ps={s:set() for s in donors}; all_keys=set(); n_calls=0; observed_copies={}
    calls=rows(task.artifact(p,m,'calls.private.tsv.gz'),(*KEY,'sample_id','gt_state','observed_phased',
        'phase_set','quality_state','counted_dose','copy1_counted','copy2_counted'))
    for key,group in itertools.groupby(calls,lambda r:tuple(r[k] for k in KEY)):
        require(key not in all_keys and key[0]==chrom,'Repeated or foreign call locus'); all_keys.add(key)
        batch=list(group)
        require(len(batch)==len(donors) and {r['sample_id'] for r in batch}==set(donors),'Missing/duplicate donor call')
        require(lm['map_first_bp']<=int(key[1])<=lm['map_last_bp'],'Candidate outside map')
        for row in batch:
            sid=row['sample_id']; phased=boolean(row['observed_phased'])
            require(row['gt_state'] in ('COMPLETE','MISSING_OR_PARTIAL','NON_DIPLOID'),'Unknown genotype state')
            require(not phased or row['gt_state']=='COMPLETE','Incomplete call marked phased')
            s=stats[sid]; s['candidate_calls']+=1; s['gt_'+row['gt_state']]+=1
            s['observed_phased_calls']+=int(phased); s['quality_'+row['quality_state']]+=1
            if row['phase_set'] not in ('NA','.',''):
                ps[sid].add(row['phase_set']); s['calls_with_phase_set']+=1
            if phased:
                copies=[row['copy1_counted'],row['copy2_counted']]
                require(all(c in ('0','1') for c in copies) and sum(map(int,copies))==int(row['counted_dose']),'Phased copy/dose mismatch')
                for idx,value in enumerate(copies,1):
                    if value=='1': observed_copies[(key,sid,str(idx))]=(row['phase_set'],row['quality_state'])
            else:
                require(row['copy1_counted']==row['copy2_counted']=='NA','Unphased copy assignment forbidden')
            n_calls+=1
            require(n_calls<=task.c['resources']['max_private_rows'],'Candidate call row budget exceeded')
        if n_calls%10000<len(donors): task.guard()
    expected=m['summary']; require(n_calls==expected['person_locus_rows'] and len(all_keys)==expected['candidate_loci'], 'Call denominator differs')
    require(all(s['candidate_calls']==len(all_keys) for s in stats.values()), 'Calls missing for donor')
    phasefields=('sample_id','chromosome','candidate_calls','complete_calls','missing_or_partial_calls',
        'non_diploid_calls','observed_phased_calls','calls_with_phase_set','distinct_observed_phase_sets','quality_unmeasured_calls',
        'phase_continuity','full_marker_haplotypes_audited')
    write_rows(task.out/'donor_phase.private.tsv',phasefields,(dict(sample_id=sid,chromosome=chrom,
        candidate_calls=s['candidate_calls'],complete_calls=s['gt_COMPLETE'],missing_or_partial_calls=s['gt_MISSING_OR_PARTIAL'],
        non_diploid_calls=s['gt_NON_DIPLOID'],observed_phased_calls=s['observed_phased_calls'],calls_with_phase_set=s['calls_with_phase_set'],
        distinct_observed_phase_sets=len(ps[sid]),quality_unmeasured_calls=s['quality_NOT_MEASURED_GT_ONLY'],
        phase_continuity='NOT_CERTIFIED',full_marker_haplotypes_audited=False) for sid,s in sorted(stats.items())))
    carriers=rows(task.artifact(p,m,'carriages.private.tsv.gz'),(*KEY,'sample_id','copy_index','reference_scenario','phase_set'))
    summaries=[]; n_carriages=0; seen_keys=set()
    copyfields=(*KEY,'sample_id','ancestry','atomic_unit_id','copy_index','phase_set','quality_state','site_filter',
        'reference_scenario','pc','phi','component_id','component_reference_members','component_donor_members','unknown_pair_estimates',
        'reference_separation_status','phase_continuity')
    def annotated():
        nonlocal n_carriages
        for key,group in itertools.groupby(carriers,lambda r:tuple(r[k] for k in KEY)):
            batch=list(group); require(key in all_keys and key not in seen_keys,'Unknown/repeated carriage locus'); seen_keys.add(key)
            seen=set()
            for row in batch:
                sid=row['sample_id']; require(sid in donors and row['copy_index'] in ('1','2'),'Unknown donor/copy')
                require((sid,row['copy_index']) not in seen,'Duplicate observed copy'); seen.add((sid,row['copy_index']))
                require(row['ancestry']==donors[sid]['ancestry'],'Carriage ancestry differs')
                require(observed_copies.pop((key,sid,row['copy_index']),None)==(row['phase_set'],row['quality_state']),
                        'Carriage differs from observed phased call')
                n_carriages+=1
                for d in deps[sid]:
                    status='RELATEDNESS_UNRESOLVED' if d['nonfinite_pairs'] else (
                        'REFERENCE_CONNECTED_COMPONENT' if d['n_references'] else 'NO_REFERENCE_LINK_AT_THIS_PC_AND_CUT')
                    yield dict(row,pc=d['pc'],phi=d['phi'],component_id=d['component_id'],
                        component_reference_members=d['n_references'],component_donor_members=d['n_donors'],
                        unknown_pair_estimates=d['nonfinite_pairs'],reference_separation_status=status,phase_continuity='NOT_CERTIFIED')
            for pc,t in itertools.product(PCS,task.c['thresholds']):
                for ancestry in ['ALL',*sorted({r['ancestry'] for r in batch})]:
                    selected=[r for r in batch if ancestry=='ALL' or r['ancestry']==ancestry]
                    ids={r['sample_id'] for r in selected}
                    ds={sid:next(d for d in deps[sid] if d['pc']==pc and d['phi']==t) for sid in ids}
                    eligible={s:d for s,d in ds.items() if d['n_references']==0 and d['nonfinite_pairs']==0}
                    summaries.append(dict(zip(KEY,key),pc=pc,phi=t,ancestry=ancestry,
                        observed_copies=len(selected),carrier_donors=len(ids),carrier_components=len({d['component_id'] for d in ds.values()}),
                        no_reference_link_donors=len(eligible),no_reference_link_components=len({d['component_id'] for d in eligible.values()}),
                        reference_connected_donors=sum(d['n_references']>0 for d in ds.values()),
                        unknown_relatedness_donors=sum(d['nonfinite_pairs']>0 for d in ds.values())))
            task.guard()
    write_rows(task.out/'carriage_dependency.private.tsv.gz',copyfields,annotated())
    require(not observed_copies and n_carriages==expected['observed_carrier_copy_rows'] and len(seen_keys)==len(all_keys),'Carriage denominator differs')
    fields=(*KEY,'pc','phi','ancestry','observed_copies','carrier_donors','carrier_components','no_reference_link_donors',
            'no_reference_link_components','reference_connected_donors','unknown_relatedness_donors')
    write_rows(task.out/'candidate_separation.tsv.gz',fields,summaries)
    summary=dict(chromosome=chrom,candidate_loci=len(all_keys),distinct_donors=len(donors),person_locus_rows=n_calls,
        observed_carrier_copy_rows=n_carriages,annotated_copy_scenario_rows=n_carriages*6,
        observed_phased_calls=sum(s['observed_phased_calls'] for s in stats.values()),
        calls_with_phase_set=sum(s['calls_with_phase_set'] for s in stats.values()),
        source_panel_records=sm['panel_counts']['records'],source_panel_sha256=sm['input_files']['panel_vcf']['sha256'],
        map_first_bp=lm['map_first_bp'],map_last_bp=lm['map_last_bp'],map_no_extrapolation=True,
        candidate_separation=[dict(pc=pc,phi=t,
            candidate_loci=sum(r['pc']==pc and r['phi']==t and r['ancestry']=='ALL' for r in summaries),
            loci_with_at_least_two_no_reference_link_components=sum(r['pc']==pc and r['phi']==t and r['ancestry']=='ALL' and r['no_reference_link_components']>=2 for r in summaries),
            loci_without_no_reference_link_components=sum(r['pc']==pc and r['phi']==t and r['ancestry']=='ALL' and r['no_reference_link_components']==0 for r in summaries))
            for pc,t in itertools.product(PCS,task.c['thresholds'])])
    return task.seal('chromosome',summary,chromosome=chrom,dependency_manifest_sha256=sha(dependency_manifest))


def aggregate(task,manifests,dependency_manifest):
    by_chrom={}; summaries=[]
    for raw in manifests:
        p=task.inputs.add(raw,sha(raw)); m=read(p)
        require((m.get('schema'),m.get('status'))==STAGES['chromosome'] and m['contract_sha256']==sha(task.contract_path),
            'Incompatible chromosome readiness')
        c=m['chromosome']; require(c not in by_chrom,'Duplicate chromosome readiness')
        require(m['dependency_manifest_sha256']==sha(dependency_manifest),'Dependency sensitivity changed')
        summary=read(task.artifact(p,m,'summary.json')); by_chrom[c]=summary
        summaries.append(summary)
    expected={j['chromosome'] for j in task.c['jobs']}
    require(set(by_chrom)==expected,'Missing chromosome readiness')
    dep=read(task.inputs.add(dependency_manifest,sha(dependency_manifest)))
    require(dep['contract_sha256']==sha(task.contract_path),'Aggregate dependency contract differs')
    totals={name:sum(s[name] for s in summaries) for name in ('candidate_loci','person_locus_rows','observed_carrier_copy_rows',
            'annotated_copy_scenario_rows','observed_phased_calls','calls_with_phase_set','source_panel_records')}
    summary=dict(chromosomes=sorted(expected,key=int),distinct_donors=dep['summary']['n_donors'],
        distinct_references=dep['summary']['n_references'],**totals,
        dependency=dep['summary'],simulation_ready=False)
    fields=('chromosome','candidate_loci','distinct_donors','person_locus_rows','observed_carrier_copy_rows',
        'observed_phased_calls','calls_with_phase_set','source_panel_records','map_first_bp','map_last_bp')
    write_rows(task.out/'chromosome_readiness.tsv',fields,({k:s[k] for k in fields} for s in sorted(summaries,key=lambda x:int(x['chromosome']))))
    rows_out=[]
    for pc,t in itertools.product(PCS,task.c['thresholds']):
        items=[r for s in summaries for r in s['candidate_separation'] if r['pc']==pc and r['phi']==t]
        rows_out.append(dict(pc=pc,phi=t,**{k:sum(r[k] for r in items) for k in items[0] if k not in ('pc','phi')}))
    write_rows(task.out/'candidate_separation_summary.tsv',tuple(rows_out[0]),rows_out)
    summary['candidate_separation']=rows_out
    return task.seal('aggregate',summary)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=STAGES)
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--chromosome')
    parser.add_argument('--dependency-manifest',type=Path)
    parser.add_argument('--manifest',action='append',type=Path,default=[])
    args=parser.parse_args(argv); os.umask(0o077)
    task=Task(args.contract,args.contract_sha256,args.output_dir)
    if args.stage=='dependency': dependency(task)
    elif args.stage=='chromosome': chromosome(task,args.chromosome,args.dependency_manifest)
    else: aggregate(task,args.manifest,args.dependency_manifest)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
