#!/usr/bin/env python3
"""Trace recorded PC-Relate sensitivity through existing donor units.

This reuses recorded coefficients, never estimates new kinship or chooses the
configuration yielding fewer conflicts. Components represent possible dependencies,
not certified pedigrees. Unlisted pairs remain unassessed.
"""
from collections import Counter, defaultdict
import math

from r02_lai_target_dependencies import require


def link_existing_ibd(sensitivity_result, ibd_rows, summary):
    """Join existing reported segments; zero calls never certifies absence."""
    key = lambda row: tuple(sorted((row['sample_a'], row['sample_b'])))
    reference = {key(r): r for r in sensitivity_result['reference_donor_pairs']}
    indexed = {key(r): r for r in ibd_rows}
    require(len(indexed) == len(ibd_rows) and set(indexed) == set(reference),
            'IBD and kinship pair universes differ')
    for pair, row in indexed.items():
        require(row['role_pair'] == reference[pair]['role_pair'] and
                int(row['source_autosomes_authenticated']) == 22, 'IBD source scope differs')
        require(all(float(row['kin_pc' + pc]) == value for pc, value in
                    reference[pair]['kinship_by_pc'].items()), 'IBD and PC-Relate lineage differs')
        require(row['callability_status'] == summary['callability_status'], 'IBD callability metadata differs')
    require(len(ibd_rows) == summary['pairs'], 'IBD pair count differs')
    return {'status': 'EXISTING_IBD_REUSED_DETECTABILITY_UNRESOLVED',
            'matched_pairs': len(indexed), 'authenticated_autosomes': 22,
            'pairs_with_reported_segments': sum(int(r['reported_records']) > 0 for r in ibd_rows),
            'reported_records': sum(int(r['reported_records']) for r in ibd_rows),
            'maximum_reported_segment_cm': max(float(r['max_reported_cm']) for r in ibd_rows),
            'callability_status': summary['callability_status'], 'new_ibd_called': False,
            'absence_of_relatedness_certified': False,
            'limits': ['Reported absence is not biological absence.',
                       'IBD length/kinship rules and PC-Relate coefficients are different estimators.',
                       'Detection probability and false positive rates are not calibrated by this join.']}


def components(roster, original, pairs):
    parent = {person: person for person in roster}

    def root(person):
        while parent[person] != person:
            parent[person] = parent[parent[person]]
            person = parent[person]
        return person

    def join(a, b):
        parent[root(a)] = root(b)

    units = defaultdict(list)
    for person in roster:
        units[original[person]['atomic_unit_id']].append(person)
    for people in units.values():
        for person in people[1:]:
            join(people[0], person)
    for a, b in pairs:
        join(a, b)
    groups = defaultdict(list)
    for person in roster:
        groups[root(person)].append(person)
    cross = []
    for people in groups.values():
        roles = Counter(roster[p]['split'] if roster[p]['role'] == 'SOURCE_VALID'
                        else 'REF_TRAIN' for p in people)
        if len(roles) > 1:
            cross.append({'people': sorted(people), 'role_counts': dict(sorted(roles.items())),
                          'original_units': sorted({original[p]['atomic_unit_id'] for p in people})})
    return {'component_count': len(groups), 'cross_role_components': sorted(cross, key=lambda x: x['people']),
            'donor_fold_bridges': sum('FOLD_A' in x['role_counts'] and 'FOLD_B' in x['role_counts'] for x in cross)}


def assess(roster_rows, original_rows, pair_rows, sensitivity):
    roster = {r['sample_id']: r for r in roster_rows}
    original = {r['sample_id']: r for r in original_rows}
    require(len(roster) == len(roster_rows) and len(original) == len(original_rows), 'duplicate roster identity')
    require(all(p in original and r['role'] in ('REF_TRAIN', 'SOURCE_VALID') and
                original[p]['role'] == r['role'] and original[p]['atomic_unit_id']
                for p, r in roster.items()), 'invalid or held-out role in development assessment')
    require(all(r['role'] != 'SOURCE_VALID' or r['split'] in ('FOLD_A', 'FOLD_B')
                for r in roster.values()), 'unknown donor fold')
    pcs, thresholds = sensitivity['pcs'], sensitivity['thresholds']
    require(pcs and len(set(pcs)) == len(pcs) and thresholds and
            all(math.isfinite(t) and 0 < t < 1 for t in thresholds), 'invalid recorded sensitivity')
    seen, clean = set(), []
    for row in pair_rows:
        a, b = row['sample_a'], row['sample_b']
        require(a in roster and b in roster and a != b, 'unmapped or held-out reviewed pair')
        key = tuple(sorted((a, b)))
        require(key not in seen, 'duplicate reviewed pair'); seen.add(key)
        relation = '--'.join(sorted((roster[a]['role'], roster[b]['role'])))
        require(relation == row['role_pair'], 'pair roles differ from roster')
        values = {str(pc): float(row[f'kin_pc{pc}']) for pc in pcs}
        require(all(math.isfinite(v) for v in values.values()), 'nonfinite kinship')
        clean.append({'sample_a': a, 'sample_b': b, 'role_pair': relation,
                      'population_pair': '--'.join(sorted((original[a]['canonical_population'],
                                                          original[b]['canonical_population']))),
                      'kinship_by_pc': values})
    cross = [r for r in clean if r['role_pair'] == 'REF_TRAIN--SOURCE_VALID']
    scenarios = []
    for threshold in thresholds:
        for selection in [*[str(p) for p in pcs], 'ANY_RECORDED_PC']:
            selected = [r for r in clean if (any(v >= threshold for v in r['kinship_by_pc'].values())
                         if selection == 'ANY_RECORDED_PC' else r['kinship_by_pc'][selection] >= threshold)]
            graph = components(roster, original, [(r['sample_a'], r['sample_b']) for r in selected])
            scenarios.append({'phi': threshold, 'pcs_rule': selection, 'flagged_pairs': len(selected),
                              'reference_donor_pairs': sum(r['role_pair'] == 'REF_TRAIN--SOURCE_VALID' for r in selected),
                              **graph})
    return {'schema': 'r02_lai_kinship_dependency_sensitivity_v1',
            'status': 'RECORDED_SENSITIVITY_TRACED_NOT_INDEPENDENCE_CERTIFICATION',
            'people': len(roster), 'reviewed_pairs': len(clean), 'reference_donor_pairs': cross,
            'population_pair_counts': dict(Counter(r['population_pair'] for r in cross)),
            'pcs': pcs, 'thresholds': thresholds, 'scenarios': scenarios,
            'configuration_selected': False, 'people_removed': 0, 'roles_changed': False,
            'evaluate_opened': False, 'independence_certified': False,
            'limits': ['Only previously flagged pairs are supplied; unlisted pairs are unknown.',
                       'Sensitivity edges are possible dependencies, not established relatives.',
                       'A smaller conflict count does not justify selecting PCs or thresholds.',
                       'Recorded population labels have not been biologically harmonized.']}
