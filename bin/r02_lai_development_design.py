#!/usr/bin/env python3
"""Freeze metadata-only donor folds and conditional LAI demographic scenarios.

Original reference/donor roles and population-plus-IBD units are immutable.
The output is preparation, not a claim of independent test material or a command
to simulate. In particular, scenario fractions are not fitted migration rates.
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
import re
import resource
import time

import r02_lai_simulate as sim
from r02_lai_simulation_plan import authenticate_report, table
from r02_lai_support_review import read, require, sha, write

SCHEMA = 'r02_lai_development_design_input_v1'
ALGORITHM = 'ANCESTRY_ORDER_LARGEST_UNIT_FIRST_MIN_ANCESTRY_THEN_TOTAL_V1'
CODE_FILES = ('r02_lai_development_design.py', 'r02_lai_simulation_plan.py', *sim.CODE_FILES)
LOCUS = ('chromosome','position_bp','REF','ALT','counted_allele')
CALL_COUNTS = ('complete_gt','unknown_gt','observed_copies','carrier_people',
               'copyable_site_allowed_copies','quality_evaluable_gt','quality_pass_gt',
               'quality_failed_gt','quality_missing_gt','quality_not_evaluable_gt','quality_unmeasured_gt')


def resource_guard(out, limits, started):
    require(time.monotonic()-started <= limits['timeout_seconds'], 'Development-design deadline exceeded')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= limits['memory_gib']*2**30,
            'Development-design memory budget exceeded')
    require(sim.shutil.disk_usage(out).free >= limits['disk_reserve_gib']*2**30, 'Disk reserve violated')
    require(sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= limits['max_output_bytes'],
            'Development-design output budget exceeded')


def bounded_table(path, required, maximum):
    rows = []
    for row in table(path, required):
        require(len(rows) < maximum, 'Metadata row limit exceeded: '+path.name)
        rows.append(row)
    return rows


def policy_checked(policy):
    required = {'algorithm', 'ancestry_order', 'fold_labels', 'reference_split',
                'generations', 'anchor_generation', 'ancestry_weights', 'focal_ancestry',
                'focal_sensitivities', 'kinship_pcs', 'kinship_thresholds', 'scope',
                'demographic_model', 'scenario_interpretation', 'references'}
    require(set(policy) == required, 'Explicit scientific policy fields required')
    require(policy['algorithm'] == ALGORITHM, 'Unsupported allocation algorithm')
    ancestors = policy['ancestry_order']; folds = policy['fold_labels']
    require(len(ancestors) == 3 and len(set(ancestors)) == 3, 'Three distinct ancestries required')
    require(len(folds) == 2 and len(set(folds)) == 2 and policy['reference_split'] not in folds,
            'Two donor folds and distinct reference label required')
    require(all(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', x or '') for x in
                [*ancestors, *folds, policy['reference_split']]), 'Invalid ancestry/split label')
    require(set(policy['ancestry_weights']) == set(ancestors), 'Ancestry weights do not match labels')
    require(all(isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(x) and x > 0
                for x in policy['ancestry_weights'].values()), 'Weights must be finite and positive')
    generations = policy['generations']
    require(generations == sorted(set(generations)) and all(type(g) is int and g > 0 for g in generations)
            and policy['anchor_generation'] in generations, 'Explicit ordered generation grid required')
    require(policy['focal_ancestry'] in ancestors, 'Focal ancestry not in declared labels')
    sensitivities = policy['focal_sensitivities']
    require(len(sensitivities) == len(set(sensitivities)) and
            all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0 < x < 1
                for x in sensitivities), 'Sensitivity fractions must be distinct and between zero and one')
    require(policy['kinship_pcs'] and policy['kinship_pcs'] == sorted(set(policy['kinship_pcs'])) and
            all(type(x) is int and x > 0 for x in policy['kinship_pcs']), 'Explicit PC sensitivity required')
    require(policy['kinship_thresholds'] and policy['kinship_thresholds'] == sorted(set(policy['kinship_thresholds'])) and
            all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0 < x < 0.5
                for x in policy['kinship_thresholds']), 'Explicit kinship thresholds required')
    require(policy['scope'] == 'CONDITIONAL_DEVELOPMENT_ONLY' and
            policy['demographic_model'] == 'SINGLE_PULSE_RANDOM_MATING', 'Unsupported scientific scope')
    require(isinstance(policy['scenario_interpretation'],str) and policy['scenario_interpretation'].strip(),
            'Demographic simplification must be explained')
    require(isinstance(policy['references'],list) and policy['references'] and
            all(set(x) == {'url','support','not_supported'} and x['url'].startswith('https://') and
                x['support'].strip() and x['not_supported'].strip() for x in policy['references']),
            'Primary references must state support and non-transferable claims')
    return policy


def allocate(identities, policy):
    """Allocate intact units without genotypes, outcomes, or rare-allele counts."""
    units = defaultdict(list)
    for row in identities.values():
        if row['role'] == 'SOURCE_VALID': units[row['atomic_unit_id']].append(row)
    totals = {fold:0 for fold in policy['fold_labels']}
    ancestry_totals = {a:{fold:0 for fold in totals} for a in policy['ancestry_order']}
    assigned = {}
    for ancestry in policy['ancestry_order']:
        selected = [(uid,rows) for uid,rows in units.items() if rows[0]['ancestry'] == ancestry]
        require(len(selected) >= len(totals), 'Insufficient intact donor units for every fold: '+ancestry)
        for uid,rows in sorted(selected, key=lambda pair:(-len(pair[1]),pair[0])):
            fold = min(totals, key=lambda f:(ancestry_totals[ancestry][f], totals[f],
                                            policy['fold_labels'].index(f)))
            assigned[uid] = fold; ancestry_totals[ancestry][fold] += len(rows); totals[fold] += len(rows)
    return assigned, ancestry_totals, totals


def scenarios(policy):
    total = sum(policy['ancestry_weights'].values())
    anchor = {a:policy['ancestry_weights'][a]/total for a in policy['ancestry_order']}
    focal = policy['focal_ancestry']; nonfocal = [a for a in anchor if a != focal]
    require(all(abs(x-anchor[focal]) > 1e-12 for x in policy['focal_sensitivities']),
            'Sensitivity duplicates normalized anchor fraction')
    fractions = [('ANCHOR', anchor)]
    for index,value in enumerate(policy['focal_sensitivities'],1):
        weights = {a:(1-value)*anchor[a]/(1-anchor[focal]) for a in nonfocal}; weights[focal] = value
        fractions.append((f'SENSITIVITY_{index}',weights))
    result = []
    for generation in policy['generations']:
        for label,weights in fractions:
            result.append(dict(scenario_id=f'PULSE_T{generation}_{label}', generations=generation,
                fractions={a:weights[a] for a in policy['ancestry_order']},
                is_anchor=generation == policy['anchor_generation'] and label == 'ANCHOR',
                model=policy['demographic_model'], simulated_people=None, popsize=None,
                seed=None, executable=False, fitted_to_lai_performance=False))
    return result


def authenticate_pair_review(entry, report):
    path = sim.descriptor(entry); manifest = read(path)
    require(entry['sha256'] in report['inputs_sha256'].values(), 'Kinship review not authenticated by L2')
    require(manifest.get('schema') == 'r02_lai_donor_review_private_v1' and
            manifest.get('status') == 'COMPLETE_DESCRIPTIVE_REVIEW_NOT_DONOR_SELECTION' and
            manifest.get('source_test_opened') is False, 'Completed permitted kinship review required')
    outputs = {}
    for name,digest in manifest.get('outputs_sha256',{}).items():
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name or ''), 'Unsafe review output')
        outputs[name] = sim.descriptor(dict(path=str(path.parent/name),sha256=digest))
    require({'summary.json','pair_review.private.tsv'} <= set(outputs), 'Missing kinship summary or pairs')
    return outputs


def write_table(path, fields, rows):
    with path.open('x') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def candidate_call_counts(call):
    """Keep observed dosage distinct from quality certification and missingness."""
    require(call['gt_state'] in {'COMPLETE','MISSING_OR_PARTIAL','NON_DIPLOID'}, 'Unknown genotype state')
    require(call['site_allowed'] in {'True','False'} and call['observed_phased'] in {'True','False'},
            'Invalid candidate boolean')
    quality = call['quality_state']
    require(quality in {'PASS','FAILED','MISSING','NOT_EVALUABLE_GT','NOT_MEASURED_GT_ONLY'},
            'Unknown candidate quality state')
    value = Counter()
    if call['gt_state']=='COMPLETE':
        require(call['counted_dose'] in {'0','1','2'}, 'Invalid counted dose')
        dose = int(call['counted_dose']); value['complete_gt']=1; value['observed_copies']=dose
        value['carrier_people']=int(dose>0)
        if call['site_allowed']=='True' and (call['observed_phased']=='True' or dose in (0,2)):
            value['copyable_site_allowed_copies']=dose
    else:
        require(call['counted_dose']=='NA' and call['observed_phased']=='False',
                'Unknown genotype must not become zero dosage or phased observation')
        value['unknown_gt']=1
    value[{'PASS':'quality_pass_gt','FAILED':'quality_failed_gt','MISSING':'quality_missing_gt',
           'NOT_EVALUABLE_GT':'quality_not_evaluable_gt','NOT_MEASURED_GT_ONLY':'quality_unmeasured_gt'}[quality]]=1
    value['quality_evaluable_gt']=int(quality in {'PASS','FAILED'})
    return value


def fold_support(report, identities, by_sample, policy, limits, out, started):
    """Reuse authenticated candidate-call tables; never reread full VCFs.

    Candidate absence means zero observed copies only when every permitted donor
    genotype in that fold/ancestry is known. GT_ONLY quality remains unmeasured.
    All counts are conditional on the preselected candidate catalogue, not all
    DNABR rare variants. Reference categories are inherited, not reconstructed.
    """
    sources = []; chromosomes = set()
    for name,digest in report['inputs_sha256'].items():
        path = Path(name)
        if path.name != 'manifest.json' or path.parent.name != 'private_preparation': continue
        sim.descriptor(dict(path=name,sha256=digest)); source = read(path)
        require(source.get('schema') == 'r02_lai_donor_haplotypes_v1' and
                source.get('status') == 'COMPLETE_PRIVATE_DONOR_HAPLOTYPES_NOT_SIMULATION_READY' and
                source.get('source_test_opened') is False, 'Incompatible donor-candidate source')
        chromosome = source['chromosome']; require(chromosome not in chromosomes, 'Duplicate donor-candidate chromosome')
        chromosomes.add(chromosome)
        files = {}
        for filename,value in source['outputs_sha256'].items():
            require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',filename), 'Unsafe donor-candidate output')
            files[filename] = sim.descriptor(dict(path=str(path.parent/filename),sha256=value))
        require({'loci.tsv.gz','calls.private.tsv.gz'} <= set(files), 'Candidate calls/loci missing')
        sources.append((chromosome,path,digest,source,files))
    require(len(chromosomes) == report['summary']['chromosomes'], 'Incomplete donor-candidate chromosome inventory')
    donors = {sid:r for sid,r in identities.items() if r['role']=='SOURCE_VALID'}
    groups = {(fold,anc):{sid for sid,r in donors.items() if by_sample[sid]==fold and r['ancestry']==anc}
              for fold in policy['fold_labels'] for anc in policy['ancestry_order']}
    count_rows = 0; total_loci = 0; aggregate = []; cross = Counter()
    cross_ancestry = {ancestry:Counter() for ancestry in policy['ancestry_order']}
    fields = (*LOCUS,'fold','ancestry','donor_people',*CALL_COUNTS,
              'observed_carrier_units','support_state','reference_scenario','map_range_status')
    output = out/'support_loci_by_donor_fold.tsv.gz'
    with output.open('xb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        import io
        with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as text:
            writer = csv.DictWriter(text,fieldnames=fields,delimiter='\t',lineterminator='\n'); writer.writeheader()
            for chromosome,path,digest,source,files in sorted(sources,key=lambda item:int(item[0])):
                loci = {}
                with gzip.open(files['loci.tsv.gz'],'rt') as stream:
                    for row in csv.DictReader(stream,delimiter='\t'):
                        if row['ancestry'] != 'ALL': continue
                        key = tuple(row[k] for k in LOCUS)
                        require(key not in loci and key[0]==chromosome and len(loci)<limits['max_loci'], 'Invalid candidate locus')
                        loci[key] = row
                require(len(loci)==source['summary']['candidate_loci'], 'Candidate denominator mismatch')
                local = {key:Counter() for key in groups}; seen_loci = set()
                with gzip.open(files['calls.private.tsv.gz'],'rt') as stream:
                    for locus,calls in itertools.groupby(csv.DictReader(stream,delimiter='\t'),lambda r:tuple(r[k] for k in LOCUS)):
                        require(locus in loci and locus not in seen_loci, 'Duplicate, unordered or unknown candidate calls')
                        seen_loci.add(locus); seen_people = set(); values = {key:Counter() for key in groups}; units = {key:set() for key in groups}
                        for call in calls:
                            count_rows += 1; require(count_rows <= limits['max_call_rows'], 'Candidate call row budget exceeded')
                            sid = call['sample_id']; require(sid in donors and sid not in seen_people, 'Duplicate or unauthorized candidate donor')
                            seen_people.add(sid); donor = donors[sid]
                            require(call['ancestry']==donor['ancestry'] and call['atomic_unit_id']==donor['atomic_unit_id'], 'Candidate identity mismatch')
                            key = (by_sample[sid],donor['ancestry']); counts = candidate_call_counts(call)
                            values[key].update(counts)
                            if counts['carrier_people']: units[key].add(donor['atomic_unit_id'])
                        require(seen_people == set(donors),'Incomplete donor-call denominator at candidate locus')
                        supported_folds = set()
                        for key,value in values.items():
                            fold,ancestry = key; state = 'OBSERVED_COPIES' if value['observed_copies'] else 'ZERO_OBSERVED_ALL_GT_KNOWN' if value['unknown_gt']==0 else 'NO_COPY_OBSERVED_WITH_UNKNOWN_GT'
                            row = dict(zip(LOCUS,locus)); row.update(fold=fold,ancestry=ancestry,donor_people=len(groups[key]),
                                observed_carrier_units=len(units[key]),support_state=state,
                                reference_scenario=loci[locus]['reference_scenario'],map_range_status=loci[locus]['map_range_status'])
                            row.update({field:value[field] for field in CALL_COUNTS})
                            writer.writerow(row); local[key]['candidate_loci']+=1; local[key][state]+=1
                            local[key].update({field:value[field] for field in CALL_COUNTS})
                            if value['observed_copies']:supported_folds.add(fold)
                        cross['both_folds' if len(supported_folds)==2 else 'one_fold' if len(supported_folds)==1 else 'no_fold']+=1
                        for ancestry in policy['ancestry_order']:
                            supported=sum(values[(fold,ancestry)]['observed_copies']>0 for fold in policy['fold_labels'])
                            cross_ancestry[ancestry][('no_fold','one_fold','both_folds')[supported]]+=1
                    require(seen_loci == set(loci),'Candidate with no donor-call rows')
                total_loci += len(loci)
                for (fold,ancestry),values in local.items():
                    aggregate.append(dict(chromosome=chromosome,fold=fold,ancestry=ancestry,donor_people=len(groups[(fold,ancestry)]),
                        **{key:values[key] for key in ('candidate_loci','OBSERVED_COPIES','ZERO_OBSERVED_ALL_GT_KNOWN',
                            'NO_COPY_OBSERVED_WITH_UNKNOWN_GT',*CALL_COUNTS)}))
                resource_guard(out,limits,started)
    require(total_loci == report['summary']['candidate_rare_loci'], 'Autosomal candidate denominator mismatch')
    write_table(out/'support_by_donor_fold.tsv',tuple(aggregate[0]),aggregate)
    for chromosome,path,digest,source,files in sources:
        sim.descriptor(dict(path=str(path),sha256=digest))
        for name,file in files.items(): sim.descriptor(dict(path=str(file),sha256=source['outputs_sha256'][name]))
    return dict(candidate_loci=total_loci, donor_locus_rows=count_rows, locus_support_across_folds=dict(cross),
        locus_support_across_folds_by_ancestry={ancestry:dict(counts) for ancestry,counts in cross_ancestry.items()},
        sources=[dict(path=str(path),sha256=digest) for chromosome,path,digest,source,files in sources],
        reference_categories_preserved=True,genotype_quality_certified=False,
        copyable_site_allowed_definition='Observed dosage at an allowed site with phased GT notation or homozygous GT; does not certify phase accuracy or genotype quality',
        scope='Candidate catalogue only; no variant removed because absent in one donor fold')


def prepare(contract, digest, out):
    started = time.monotonic()
    config = read(sim.descriptor(dict(path=str(Path(contract).resolve()),sha256=digest)))
    require(set(config) == {'schema','material_report','original_roles','pair_review','scientific_policy','resources'}
            and config['schema'] == SCHEMA, 'Invalid development-design contract')
    policy = policy_checked(config['scientific_policy']); limits = config['resources']
    require(set(limits) == {'max_rows','max_output_bytes','disk_reserve_gib','memory_gib','timeout_seconds','max_call_rows','max_loci'} and
            all(type(x) is int and x > 0 for x in limits.values()), 'Explicit positive integer resource limits required')
    report, material = authenticate_report(config['material_report'])
    original = sim.descriptor(config['original_roles'])
    require(config['original_roles']['sha256'] in report['inputs_sha256'].values(), 'Original roles not authenticated by L2')
    reviews = authenticate_pair_review(config['pair_review'],report)
    source_rows = bounded_table(original,('sample_id','role','ancestry','atomic_unit_id','canonical_population'),limits['max_rows'])
    identities = {}; all_ids = set(); units = defaultdict(set)
    for row in source_rows:
        sid = row['sample_id']; require(sid and sid not in all_ids, 'Duplicate original identity'); all_ids.add(sid)
        if row['role'] not in {'REF_TRAIN','SOURCE_VALID'}: continue
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',sid) and row['ancestry'] in policy['ancestry_order'] and
                row['atomic_unit_id'] and row['canonical_population'], 'Invalid permitted identity metadata')
        identities[sid] = row; units[row['atomic_unit_id']].add((row['role'],row['ancestry']))
    require(all(len(x) == 1 for x in units.values()), 'Original unit crosses role or ancestry')
    counts = Counter(r['role'] for r in identities.values())
    require(counts == Counter(REF_TRAIN=report['summary']['references'], SOURCE_VALID=report['summary']['donors']),
            'Allowed identity count does not match completed L2 report')
    assigned, ancestry_totals, totals = allocate(identities,policy)
    output_rows = [dict(sample_id=sid, role=r['role'], ancestry=r['ancestry'],
        split=assigned[r['atomic_unit_id']] if r['role']=='SOURCE_VALID' else policy['reference_split'],
        family_component=r['atomic_unit_id']) for sid,r in sorted(identities.items())]
    by_sample = {r['sample_id']:r['split'] for r in output_rows}
    pairs = bounded_table(reviews['pair_review.private.tsv'],
        ('sample_a','sample_b','role_pair',*[f'kin_pc{k}' for k in policy['kinship_pcs']]),limits['max_rows'])
    kinship = Counter(); seen_pairs = set()
    for row in pairs:
        a,b = row['sample_a'],row['sample_b']; key = tuple(sorted((a,b)))
        require(a != b and a in identities and b in identities and key not in seen_pairs, 'Invalid or duplicate reviewed pair')
        seen_pairs.add(key)
        roles = '--'.join(sorted((identities[a]['role'],identities[b]['role'])))
        require(roles == row['role_pair'], 'Reviewed pair role mismatch')
        for phi in policy['kinship_thresholds']:
            above = []
            for pc in policy['kinship_pcs']:
                value = float(row[f'kin_pc{pc}']); require(math.isfinite(value),'Nonfinite reviewed kinship')
                above.append(value >= phi)
            state = 'STABLE_ABOVE' if all(above) else 'PC_SENSITIVE' if any(above) else 'STABLE_BELOW'
            relation = 'CROSS_DONOR_FOLD' if roles == 'SOURCE_VALID--SOURCE_VALID' and by_sample[a] != by_sample[b] else roles
            kinship[(phi,relation,state)] += 1
            require(not(relation == 'CROSS_DONOR_FOLD' and any(above)),
                    'Flagged donor relationship crosses proposed folds; scientific review required')
    require(len(pairs) == read(reviews['summary.json'])['pair_review_rows'], 'Reviewed pair inventory mismatch')
    inherited = bounded_table(material['criteria.tsv'],('criterion','state','next_action'),limits['max_rows'])
    require(len({r['criterion'] for r in inherited}) == len(inherited) and
            all(r['state'] in {'PASS','LIMITED','UNRESOLVED'} for r in inherited), 'Invalid material criteria')
    out = Path(out).resolve(); require(not out.exists() and out.parent.is_dir(), 'Fresh output directory required')
    sim.local_filesystem(out.parent)
    require(sim.shutil.disk_usage(out.parent).free >= limits['disk_reserve_gib']*2**30, 'Disk reserve insufficient')
    out.mkdir(mode=0o700)
    write_table(out/'roles.private.tsv',('sample_id','role','ancestry','split','family_component'),output_rows)
    # Reuse the downstream reader and role authenticator, including donor/chromosome identity checks.
    for fold in policy['fold_labels']:
        from r02_lai_simulation_plan import match_roles
        match_roles(report,material,config['original_roles'],out/'roles.private.tsv',fold,
                    policy['ancestry_order'],[str(i) for i in range(1,report['summary']['chromosomes']+1)],False)
    unit_rows = []
    for uid,fold in sorted(assigned.items()):
        rows = [r for r in identities.values() if r['role']=='SOURCE_VALID' and r['atomic_unit_id']==uid]
        unit_rows.append(dict(unit_id=uid,fold=fold,ancestry=rows[0]['ancestry'],people=len(rows),
                             populations=';'.join(sorted({r['canonical_population'] for r in rows}))))
    write_table(out/'units.private.tsv',('unit_id','fold','ancestry','people','populations'),unit_rows)
    scenario_rows = scenarios(policy)
    write(out/'scenarios.json',dict(schema='r02_lai_demography_templates_v1',scope=policy['scope'],
        policy=policy,scenarios=scenario_rows,scenarios_executed=0,
        reciprocal_folds=[dict(fit=a,select=b,confirmatory_test=None) for a,b in
                          [policy['fold_labels'],list(reversed(policy['fold_labels']))]],
        interpretation='Reciprocal population-unit development, not independent confirmation; shared references retained.'))
    cross_reference = sum(n for (phi,roles,state),n in kinship.items() if roles=='REF_TRAIN--SOURCE_VALID' and
                          state!='STABLE_BELOW' and phi == policy['kinship_thresholds'][0])
    summary = dict(status='DEVELOPMENT_DESIGN_PREPARED_NOT_SIMULATION_AUTHORIZATION',
        donors=counts['SOURCE_VALID'], references=counts['REF_TRAIN'], fold_people=totals,
        fold_ancestry_people=ancestry_totals, donor_units=len(assigned), scenarios=len(scenario_rows),
        reviewed_pair_rows=len(pairs), cross_donor_fold_flagged_pairs=0,
        cross_reference_donor_flagged_pairs_at_lowest_phi=cross_reference,
        original_roles_changed=False,original_units_changed=False,source_test_genotypes_opened=False,
        source_test_rows_emitted=False, independent_test_created=False, independence_certified=False,
        phase_accuracy_validated=False,genotype_quality_validated=False,simulation_authorized=False,
        simulations_executed=False,actual_genotypes_prepared_here=False,
        scope=policy['scope'], inherited_criteria=inherited,
        limitations=read(material['limitations.json']))
    write(out/'summary.json',summary)
    write(out/'kinship_checks.json',dict(denominator='Flagged pairs in authenticated review; not all pairwise coefficients',
        pcs=policy['kinship_pcs'], thresholds=policy['kinship_thresholds'],
        rows=[dict(phi=k[0],role_relation=k[1],state=k[2],pairs=v) for k,v in sorted(kinship.items())],
        configuration_selected=False,independence_certified=False))
    support = fold_support(report,identities,by_sample,policy,limits,out,started)
    write(out/'support_summary.json',support)
    for entry in (config['original_roles'],config['material_report'],config['pair_review']): sim.descriptor(entry)
    authenticate_report(config['material_report']); authenticate_pair_review(config['pair_review'],report)
    sim.descriptor(dict(path=str(Path(contract).resolve()),sha256=digest))
    resource_guard(out,limits,started)
    manifest = dict(schema='r02_lai_development_design_v1', status=summary['status'],
        contract_sha256=digest,inputs={k:config[k] for k in ('material_report','original_roles','pair_review')},
        code_sha256={name:sha(Path(__file__).parent/name) for name in CODE_FILES},
        outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir())},
        contains_sample_identifiers=True,public_distribution_allowed=False,summary=summary)
    write(out/'manifest.json',manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args(); os.umask(0o077)
    prepare(args.contract,args.contract_sha256,args.output_dir)


if __name__ == '__main__': main()
