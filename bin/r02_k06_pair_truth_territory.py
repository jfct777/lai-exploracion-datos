#!/usr/bin/env python3
"""Partition existing simulated copy-pair territory by source identifiability.

This is an exhaustive denominator, NOT detector callability or negative IBD
truth. It reads no genotypes/calls and changes no detector parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from itertools import combinations
import json
from pathlib import Path
import resource
import tempfile
import time

from r02_donor_copy_truth import FIELDS, require, sha
from r02_k06_score_refined_ibd_positive import GeneticMap, pair, union_bp

STATES = ('same_identified_copy_bp', 'distinct_identified_copies_unknown_ibd_bp',
          'ambiguous_copy_bp', 'no_observed_marker_bp')
UNIQUE = 'UNIQUE_BY_ALLELE_COMPATIBILITY'
AMBIGUOUS = 'AMBIGUOUS_IDENTICAL_ALLELES'
UNOBSERVED = 'NO_OBSERVED_MARKER'


def load_blocks(path, manifest, chromosome, bounds, max_rows):
    """Require the complete diploid roster and a contiguous common map domain."""
    axes, counts = defaultdict(list), Counter()
    with gzip.open(path, 'rt', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require(reader.fieldnames == FIELDS, 'Truth columns differ')
        for index, row in enumerate(reader, 1):
            require(index <= max_rows, 'Truth row budget exceeded')
            a = (row['sample_id'], int(row['haplotype']))
            lo, hi, markers = (int(row[x]) for x in
                ('start_bp_inclusive', 'end_bp_inclusive', 'markers'))
            state, donor, copy = (row[x] for x in
                ('copy_identity_status', 'donor_id', 'compatible_donor_copies_zero_based'))
            require(row['chromosome'] == chromosome and a[1] in (1, 2) and
                    a[0].startswith(manifest['sample_namespace'] + '__'), 'Copy identity differs')
            require(bounds[0] <= lo <= hi <= bounds[1], 'Source block outside map domain')
            require(lo == (axes[a][-1][1] + 1 if axes[a] else bounds[0]),
                    'Missing, overlapping, or unordered source territory')
            if state == UNIQUE:
                require(copy in ('0', '1') and markers > 0 and donor != 'UNKNOWN', 'Unique copy lacks evidence')
            elif state == AMBIGUOUS:
                require(copy == '0,1' and markers > 0 and donor != 'UNKNOWN', 'Ambiguous copy lacks evidence')
            else:
                require(state == UNOBSERVED and copy == '.' and markers == 0 and donor == 'UNKNOWN',
                        'Unobserved copy contains unsupported identity')
            axes[a].append((lo, hi, state, donor, copy)); counts[state] += 1
    people = {a[0] for a in axes}
    require(len(people) == manifest['summary']['simulated_individuals'] and
            len(axes) == manifest['summary']['copied_haplotypes'] == 2 * len(people), 'Roster size differs')
    require(set(axes) == {(p, h) for p in people for h in (1, 2)}, 'Incomplete diploid roster')
    require(all(rows[-1][1] == bounds[1] for rows in axes.values()), 'Incomplete terminal territory')
    require(dict(counts) == manifest['verification']['blocks_by_copy_status'], 'Block counts differ')
    return dict(axes), dict(counts)


def pair_counts(left, right):
    """Linear merge of two complete inclusive interval partitions; no bp array."""
    counts = dict.fromkeys(STATES, 0)
    i = j = cells = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        lo, hi = max(a[0], b[0]), min(a[1], b[1])
        require(lo <= hi, 'Source partitions differ')
        # Missing markers take precedence over ambiguity: categories are disjoint.
        if UNOBSERVED in (a[2], b[2]): state = STATES[3]
        elif AMBIGUOUS in (a[2], b[2]): state = STATES[2]
        else: state = STATES[0] if a[3:] == b[3:] else STATES[1]
        counts[state] += hi - lo + 1; cells += 1
        if a[1] == hi: i += 1
        if b[1] == hi: j += 1
    require(i == len(left) and j == len(right), 'Unequal source domain')
    return counts, cells


def reference_positives(path, chromosome, bounds, expected_rows):
    """Reuse existing AU positives as an independent conservation identity."""
    intervals = defaultdict(list); n = 0
    with gzip.open(path, 'rt', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            n += 1; key = pair(row)
            lo, hi = int(row['mapped_start_bp']), int(row['mapped_end_bp'])
            require(key[0] == chromosome and row['map_support'] == 'FULL_MAP' and
                    bounds[0] <= lo <= hi <= bounds[1], 'Existing positive support differs')
            require(lo == int(row['start_bp_inclusive']) and hi == int(row['end_bp_inclusive']),
                    'Existing positive coordinates differ')
            intervals[key[1:]].append((lo, hi))
    require(n == expected_rows, 'Existing positive row count differs')
    return {key: union_bp(value) for key, value in intervals.items()}


def run(report_path, report_sha, case_id, output, max_rows=20000, timeout_seconds=120, staged_input_dir=None):
    started = time.monotonic()
    require(max_rows > 0 and timeout_seconds > 0, 'Positive technical bounds required')
    require(sha(report_path) == report_sha, 'BM source report changed')
    matches = [x for x in json.loads(Path(report_path).read_text())['cases'] if x['case_id'] == case_id]
    require(len(matches) == 1, 'Exactly one admitted case required')
    case = matches[0]; sources, paths = {}, {}
    for name in ('manifest', 'simulation_truth', 'genetic_map', 'positive_territory'):
        item = case['files'][name]; original = Path(item['path']); path = original
        require(original.is_absolute(), 'Absolute source descriptor required')
        if staged_input_dir is not None:
            candidates = [p for p in Path(staged_input_dir).iterdir() if p.is_file() and sha(p) == item['sha256']]
            require(len(candidates) == 1, 'Staged source identity differs: ' + name)
            path = candidates[0]
        require(sha(path) == item['sha256'], 'Source changed: ' + name)
        sources[name] = {'path': str(original), 'sha256': item['sha256']}; paths[name] = path
    manifest = json.loads(paths['manifest'].read_text())
    require(manifest['schema'] == 'r02_lai_simulate_output_v1' and
            manifest['status'] == 'COMPLETE_SIMULATED_SCENARIO_NOT_LAI_VALIDATION' and
            manifest['job_id'] == case_id and manifest['genome_build'] == 'GRCh38' and
            manifest['chromosomes'] == [case['chromosome']] and manifest['donor_split'] == case['donor_fold'] and
            manifest['verification']['truth_is_simulated_not_observed_dnabr'] is True and
            manifest['outputs_sha256']['truth.private.tsv.gz'] == sources['simulation_truth']['sha256'],
            'Authenticated case manifest differs')
    identity = {'report_sha256': report_sha, 'case_id': case_id, 'sources': sources,
                'code_sha256': {p.name: sha(p) for p in (Path(__file__),
                    Path(__file__).with_name('r02_donor_copy_truth.py'),
                    Path(__file__).with_name('r02_k06_score_refined_ibd_positive.py'))}}
    destination = Path(output)
    if destination.exists():
        result = json.loads((destination / 'summary.json').read_text())
        require(result['identity'] == identity and result['status'] == 'COMPLETE_PAIR_TRUTH_TERRITORY',
                'Existing output identity differs')
        require(all(sha(destination / name) == digest for name, digest in result['outputs_sha256'].items()),
                'Existing output changed')
        return result
    genetic = GeneticMap(paths['genetic_map'], case['chromosome'])
    bounds = genetic.bp[0], genetic.bp[-1]; territory = bounds[1] - bounds[0] + 1
    axes, blocks = load_blocks(paths['simulation_truth'], manifest,
                              case['chromosome'], bounds, max_rows)
    positives = reference_positives(paths['positive_territory'], case['chromosome'],
                                   bounds, case['positive_rows'])
    require(all(a in axes and b in axes for a, b in positives), 'Positive pair outside roster')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + '.partial-', dir=destination.parent))
    groups = {g: dict(copy_pairs=0, total_copy_pair_bp=0, **dict.fromkeys(STATES, 0))
              for g in ('between_person', 'within_person')}
    fields = ['chromosome', 'sample_a', 'haplotype_a', 'sample_b', 'haplotype_b',
              'relationship', 'total_copy_pair_bp', *STATES]
    cell_count = 0
    with gzip.open(temporary / 'pair-truth-territory.private.tsv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n'); writer.writeheader()
        for index, (a, b) in enumerate(combinations(sorted(axes), 2), 1):
            if index % 128 == 1:
                require(time.monotonic() - started < timeout_seconds, 'Case technical deadline exceeded')
            counts, cells = pair_counts(axes[a], axes[b]); cell_count += cells
            require(sum(counts.values()) == territory, 'Pair denominator not conserved')
            require(counts[STATES[0]] == positives.get((a, b), 0), 'Pair positive territory differs from AU')
            group = 'within_person' if a[0] == b[0] else 'between_person'
            writer.writerow(dict(chromosome=case['chromosome'], sample_a=a[0], haplotype_a=a[1],
                sample_b=b[0], haplotype_b=b[1], relationship=group, total_copy_pair_bp=territory, **counts))
            groups[group]['copy_pairs'] += 1; groups[group]['total_copy_pair_bp'] += territory
            for state in STATES: groups[group][state] += counts[state]
    n = len(axes) // 2
    require(groups['within_person']['copy_pairs'] == n and
            groups['between_person']['copy_pairs'] == 4 * n * (n - 1) // 2, 'Pair roster not exhaustive')
    require(sha(report_path) == report_sha and all(sha(paths[k]) == v['sha256'] for k, v in sources.items()),
            'Source changed during calculation')
    result = dict(schema='r02_k06_pair_truth_territory_v1', status='COMPLETE_PAIR_TRUTH_TERRITORY',
        identity=identity, chromosome=case['chromosome'], donor_fold=case['donor_fold'],
        dependency_component=case['component_id'], coordinate_system='1-based-inclusive',
        source_map_domain_bp=list(bounds), length_unit='copy-pair bp', groups=groups,
        block_counts=blocks, interval_intersections=cell_count,
        technical_limits=dict(max_rows=max_rows, timeout_seconds=timeout_seconds),
        elapsed_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        negative_truth_available=False, detector_callability_estimated=False, precision=None,
        specificity=None, demographic_inference=False, independent_replicates_claimed=False,
        limits=['Different identified donor copies remain genealogically unknown.',
                'Copy identifiability is conditional on alleles; phase accuracy is unvalidated.',
                'A marker count greater than zero does not establish local detector sensitivity.',
                'Do not divide CE >=2-cM positives by this all-length denominator.',
                'Only existing simulated chr1/chr22 scenarios; no transfer to observed DNABR.'],
        outputs_sha256={'pair-truth-territory.private.tsv.gz': sha(temporary / 'pair-truth-territory.private.tsv.gz')})
    (temporary / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.rename(destination)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--report-sha256', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--staged-input-dir', type=Path)
    args = parser.parse_args()
    result = run(args.report, args.report_sha256, args.case_id, args.output, args.max_rows, args.timeout_seconds,
                 args.staged_input_dir)
    print(json.dumps({'status': result['status'], 'case_id': args.case_id, 'elapsed_seconds': result['elapsed_seconds']}))
