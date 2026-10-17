#!/usr/bin/env python3
"""Known copied territory and exact reported overlap; no genealogical negatives.

Reuse existing inclusive-interval primitives. Keep chromosome, person and copy
identity: neither adjacent rows nor repeated mosaics become independent IBD.
The optional detector input is a complete, normalized callset admitted elsewhere.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

from r02_k06_observability import require, sha
from r02_segment_evidence import merge_intervals, intersect_intervals, interval_bp


def pair(row):
    a, b = (row['sample_a'], int(row['haplotype_a'])), (row['sample_b'], int(row['haplotype_b']))
    require(a[1] in (1, 2) and b[1] in (1, 2) and a < b, 'Noncanonical chromosome-copy pair')
    return row['chromosome'], a, b


def known_territory(path, chromosome, max_rows):
    intervals, counts = defaultdict(list), defaultdict(int)
    with gzip.open(path, 'rt', newline='') as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter='\t'), 1):
            require(index <= max_rows, 'Positive territory row budget exceeded')
            key = pair(row)
            require(key[0] == chromosome, 'Positive chromosome mismatch')
            relationship = 'WITHIN_PERSON_COPY_MATCH' if key[1][0] == key[2][0] else 'BETWEEN_PERSON_COPY_MATCH'
            require(row['relationship'] == relationship, 'Positive relationship mismatch')
            start, end = int(row['start_bp_inclusive']), int(row['end_bp_inclusive'])
            require(0 < start <= end, 'Invalid original positive coordinates')
            status = row['map_support']
            require(status in ('FULL_MAP', 'CLIPPED_TO_MAP', 'OUTSIDE_MAP'), 'Unknown map support')
            counts[status] += 1
            if status == 'OUTSIDE_MAP':
                require(row['mapped_start_bp'] == row['mapped_end_bp'] == 'NA', 'Outside-map coordinates must be unknown')
                continue
            left, right = int(row['mapped_start_bp']), int(row['mapped_end_bp'])
            require(start <= left <= right <= end, 'Mapped interval extends beyond positive source')
            require(((left, right) == (start, end)) == (status == 'FULL_MAP'), 'Map clipping status differs')
            intervals[key].append((left, right))
    return {key: merge_intervals(value) for key, value in intervals.items()}, dict(counts)


def summarize(positives, calls=None):
    """Count a base once per chromosome-copy pair, not once per source row."""
    groups = {kind: dict(copy_pairs=0, positive_copy_pair_bp=0,
                        reported_overlap_copy_pair_bp=None if calls is None else 0)
              for kind in ('between_person', 'within_person')}
    for key, territory in positives.items():
        kind = 'within_person' if key[1][0] == key[2][0] else 'between_person'
        group = groups[kind]
        group['copy_pairs'] += 1
        group['positive_copy_pair_bp'] += interval_bp(territory)
        if calls is not None:
            group['reported_overlap_copy_pair_bp'] += interval_bp(intersect_intervals(territory, calls.get(key, [])))
    for value in groups.values():
        denominator, numerator = value['positive_copy_pair_bp'], value['reported_overlap_copy_pair_bp']
        value['known_positive_coverage_fraction'] = None if numerator is None or not denominator else numerator / denominator
    return groups


def read_calls(request, path, chromosome):
    require(request['schema'] == 'r02_k06_complete_normalized_calls_v1'
            and request['complete_callset'] is True
            and request['genome_build'] == 'GRCh38'
            and request['coordinate_system'] == '1-based-inclusive'
            and request['chromosome'] == chromosome, 'Explicit complete compatible callset required')
    require(sha(path) == request['calls_sha256'], 'Detector calls changed')
    require(request['detector_parameters_admitted'] is True
            and len(request['detector_parameter_spec_sha256']) == 64,
            'Admitted detector specification required; format alone is insufficient')
    spec_path = Path(request['detector_parameter_spec_path'])
    require(sha(spec_path) == request['detector_parameter_spec_sha256'], 'Detector parameter specification changed')
    spec = json.loads(spec_path.read_text())
    require(spec['status'] == 'ADMITTED' and spec['scientific_scope'] == 'CONDITIONAL_POSITIVE_COPY_COVERAGE'
            and spec['case_id'] == request['case_id'], 'Scope/case of detector admission differs')
    receipt_path = Path(request['producer_receipt_path'])
    require(sha(receipt_path) == request['producer_receipt_sha256'], 'Producer receipt changed')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['status'] == 'COMPLETE' and receipt['case_id'] == request['case_id']
            and receipt['calls_sha256'] == request['calls_sha256']
            and receipt['parameter_spec_sha256'] == request['detector_parameter_spec_sha256']
            and receipt['genotypes_sha256'] == request['genotypes_sha256']
            and receipt['sample_ids'] == request['sample_ids'], 'Producer identity/completion differs')
    calls = defaultdict(list)
    people = set(request['sample_ids'])
    require(len(people) == len(request['sample_ids']) and people, 'Unique detector roster required')
    with gzip.open(path, 'rt', newline='') as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter='\t'), 1):
            require(index <= request['max_rows'], 'Call row budget exceeded')
            key = pair(row)
            require(key[0] == chromosome and key[1][0] in people and key[2][0] in people,
                    'Call chromosome/person outside admitted roster')
            left, right = int(row['start_bp_inclusive']), int(row['end_bp_inclusive'])
            require(0 < left <= right, 'Invalid call interval')
            calls[key].append((left, right))
    return calls, people


def run(report, expected_sha, case_id, output, calls_request=None):
    require(sha(report) == expected_sha, 'Preflight report changed')
    source = json.loads(Path(report).read_text())
    matches = [case for case in source['cases'] if case['case_id'] == case_id]
    require(len(matches) == 1, 'Exactly one known case required')
    case = matches[0]
    descriptor = case['files']['positive_territory']
    path = Path(descriptor['path'])
    require(sha(path) == descriptor['sha256'], 'Positive territory changed')
    positives, counts = known_territory(path, case['chromosome'], case['positive_rows'])
    require(sum(counts.values()) == case['positive_rows'], 'Incomplete positive territory')
    calls, call_identity = None, None
    if calls_request is not None:
        call_spec = json.loads(Path(calls_request).read_text())
        require(call_spec['case_id'] == case_id
                and call_spec['positive_territory_sha256'] == descriptor['sha256']
                and call_spec['genotypes_sha256'] == case['files']['genotypes']['sha256'],
                'Calls and positive/genotype identities differ')
        calls, people = read_calls(call_spec, call_spec['calls_path'], case['chromosome'])
        require(all(key[1][0] in people and key[2][0] in people for key in positives), 'Positive person absent from calls roster')
        call_identity = dict(request_sha256=sha(calls_request), calls_sha256=sha(call_spec['calls_path']))
    result = dict(schema='r02_k06_positive_copy_coverage_v1', case_id=case_id,
        status='POSITIVE_DENOMINATOR_READY' if calls is None else 'CONDITIONAL_REPORTED_POSITIVE_COVERAGE',
        source_report_sha256=expected_sha, positive_sha256=descriptor['sha256'], calls=call_identity,
        chromosome=case['chromosome'], dependency_component=case['component_id'],
        phase_accuracy_validated=case['phase_accuracy_validated'],
        positive_rows=case['positive_rows'], map_support_rows=counts, groups=summarize(positives, calls),
        coordinate_system='1-based-inclusive', length_unit='copy-pair bp', endpoint_tolerance_bp=0,
        negative_truth_available=False, precision=None, specificity=None,
        detector_calibration_complete=False, demographic_inference=False,
        limits=['Exact territory overlap is not segment sensitivity or endpoint accuracy.',
                'Unmatched or differently copied territory is unknown, not a genealogical negative.',
                'Mapped support is not a callability certificate.',
                'Mosaics and copy pairs are not independent biological replicates.',
                'No detector parameters or biological selection are admitted by this consumer.'])
    require(sha(report) == expected_sha and sha(path) == descriptor['sha256'], 'Input changed while computing')
    if calls_request is not None:
        require(sha(calls_request) == call_identity['request_sha256']
                and sha(call_spec['calls_path']) == call_identity['calls_sha256'], 'Calls changed while computing')
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    destination = Path(output)
    if destination.exists():
        require(destination.read_text() == encoded, 'Existing output differs; preserve prior result')
    else:
        with destination.open('x') as handle:
            handle.write(encoded)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'report-sha256', 'case-id', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--calls-request')
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.report_sha256, args.case_id, args.output, args.calls_request)['status']))
