#!/usr/bin/env python3
"""Join completed K05 fits to their audited support, without refitting/ranking."""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def synthesize(plan_path, plan_sha256, output):
    require(sha(plan_path) == plan_sha256, 'Preparation specification changed')
    plan = json.loads(Path(plan_path).read_text())
    require(plan['scope'] == 'EXISTING_DESCRIPTIVE_RESULTS_NO_REFIT_OR_RANKING', 'Invalid scope')
    for name, descriptor in plan['inputs'].items():
        require(sha(descriptor['path']) == descriptor['sha256'], 'Input changed: ' + name)
    require(sha(__file__) == plan['code_sha256'], 'Implementation changed')
    inputs = {name: value['path'] for name, value in plan['inputs'].items()}
    coverage = table(inputs['coverage'])
    fits = table(inputs['associations'])
    summary = json.loads(Path(inputs['community_summary']).read_text())
    coverage_receipt = json.loads(Path(inputs['coverage_receipt']).read_text())
    # The old metadata audit is reused, including its underlying identity pins.
    for path, digest in coverage_receipt['sources_sha256'].items():
        require(sha(path) == digest, 'Audited coverage source changed')
    ledger = {row['partition_id']: row for row in summary['partition_ledger']}
    support = {row['partition_id']: row for row in coverage}
    require(len(support) == len(coverage) == len(ledger) == summary['n_partitions'], 'Partition inventory differs')
    require(set(support) == set(ledger), 'Coverage and community partitions differ')
    thresholds = [str(x) for x in summary['kinship_thresholds']]
    full, sensitivity = 'COMPLETE_REPORTED_CALLSET', 'REPORTED_ONLY_SENSITIVITY'
    adjusted = 'ANCESTRY_COLLECTION_GT_AVAILABILITY'
    models = ['CRUDE', 'ANCESTRY', adjusted]
    relations = ['ALL_KNOWN_COMPONENT', 'DIFFERENT_COMPONENT']
    expected = set(itertools.product(ledger, thresholds, [full, sensitivity], models, relations))
    keyed = {}
    for row in fits:
        key = tuple(row[name] for name in ['partition_id', 'kinship_threshold', 'call_scope', 'model', 'relation'])
        require(key not in keyed and key in expected, 'Duplicate or unexpected completed comparison')
        keyed[key] = row
        item, cov = ledger[row['partition_id']], support[row['partition_id']]
        require(row['graph_id'] == item['graph_id'] == cov['graph_id'] and
                float(row['resolution']) == float(item['resolution']) == float(cov['resolution']),
                'Graph or resolution differs across completed sources')
        require(int(row['support_people']) == int(cov['assigned_n']), 'Fit/support people differ')
        column = ('assigned_pairs' if row['relation'] == 'ALL_KNOWN_COMPONENT' else
                  'assigned_phi' + row['kinship_threshold'] + '_between_component_pairs')
        actual_pairs, possible_pairs = int(row['n_pairs']), int(cov[column])
        require(actual_pairs == possible_pairs if row['call_scope'] == full else
                0 <= actual_pairs <= possible_pairs, 'Fit/support pair denominator differs')
    require(set(keyed) == expected, 'Completed K05 family has missing comparisons')
    rows = []
    for partition in sorted(ledger):
        cov, item = support[partition], ledger[partition]
        for threshold in thresholds:
            base = keyed[(partition, threshold, full, adjusted, 'DIFFERENT_COMPONENT')]
            all_pairs = keyed[(partition, threshold, full, adjusted, 'ALL_KNOWN_COMPONENT')]
            reported = keyed[(partition, threshold, sensitivity, adjusted, 'DIFFERENT_COMPONENT')]
            row = dict(partition_id=partition, family=item['family'], graph_id=item['graph_id'],
                resolution=item['resolution'], cohort_people=summary['n_cohort'],
                assigned_people=cov['assigned_n'], unassigned_people=summary['n_cohort']-int(cov['assigned_n']),
                assigned_communities=cov['assigned_communities'], kinship_threshold=threshold,
                relatedness_components=cov['assigned_phi'+threshold+'_components'],
                eligible_between_component_pairs=base['n_pairs'], status=base['status'],
                conditional_coefficient=base['coefficient'], partial_r2=base['partial_r2'],
                leave_one_component_sign_changes=base['loo_sign_changes'],
                leave_one_component_valid=base['loo_valid'], leave_one_component_invalid=base['loo_invalid'],
                maximum_component_score_fraction=base['max_component_score_fraction'],
                all_known_component_pairs=all_pairs['n_pairs'],
                all_known_component_coefficient=all_pairs['coefficient'],
                reported_only_pairs=reported['n_pairs'],
                reported_only_coefficient=reported['coefficient'], reported_only_status=reported['status'],
                interpretation='DESCRIPTIVE_OWN_SUPPORT_NOT_A_CONFIGURATION_RANKING')
            rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    destination = output/'partition_support_and_kinship.tsv'
    with destination.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    result = dict(schema='r02_k05_support_synthesis_v1',
        status='COMPLETE_JOIN_OF_EXISTING_DESCRIPTIVE_RESULTS_NOT_BIOLOGICAL_SELECTION',
        partitions=len(ledger), source_comparisons=len(fits), integrated_rows=len(rows),
        partitions_by_family=dict(Counter(x['family'] for x in ledger.values())),
        integrated_status_counts=dict(Counter(x['status'] for x in rows)),
        support_people_range=[min(int(x['assigned_n']) for x in coverage),max(int(x['assigned_n']) for x in coverage)],
        fits_rerun=0, universal_intersection_filter_applied=False, winner_selected=False,
        confidence_intervals_calibrated=False, human_genotypes_loaded=False,
        source_pins=plan['inputs'], output_sha256=sha(destination), code_sha256=sha(__file__),
        limitations=['Different assigned supports are not exchangeable comparison populations',
                    'Reported-only sensitivity conditions on observing an IBD call and has a smaller pair denominator',
                    'Shared partitions, people and IBD calls are dependent observations',
                    'Leave-component diagnostics and coefficients do not identify causal demography',
                    'Original fixed partitions were learned globally, not in held-out FIT-only training'])
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(synthesize(args.plan, args.plan_sha256, args.output)['status'])
