#!/usr/bin/env python3
"""Aggregate-only PC-Relate description of an existing M14 sensitivity sweep.

Two streaming passes over the sweep pair table validate unique pair/config rows
and aggregate their *existing* segments. One PC-Relate pass retains only pairs
in that table. No genotypes, segment redetection, IBD calls, family assignment,
threshold optimization or significance tests occur. The operational kinship
cutoffs do not certify families, and PC-Relate including chr22 is not independent
validation of the chr22 sweep.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


MEASURES = ('n_pairs', 'n_segments', 'total_shared_bp')
GROUPS = ('kin_ge_threshold', 'kin_lt_threshold', 'missing')
CONFIG_FIELDS = ('config_id', 'max_gap_bp', 'min_length_bp', 'min_shared_effective')
OUTPUT_FIELDS = (*CONFIG_FIELDS, 'kinship_threshold', 'kinship_group',
                 *MEASURES, 'n_ids', 'n_pairs_total', 'n_segments_total',
                 'total_shared_bp_total', 'n_ids_total',
                 'pair_fraction', 'segment_fraction', 'shared_bp_fraction')
UNSEEN = object()


class AuditError(ValueError):
    """Input integrity failure; messages deliberately omit sample identifiers."""


def current_rss():
    try:
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class Guard:
    def __init__(self, max_pairs, memory_mb):
        if max_pairs <= 0 or memory_mb <= 0:
            raise AuditError('Resource limits must be positive')
        self.max_pairs = int(max_pairs)
        self.memory_limit_bytes = int(memory_mb) * 1024**2
        self.peak_checked_rss_bytes = 0

    def check(self, n_pairs=0):
        rss = current_rss()
        self.peak_checked_rss_bytes = max(self.peak_checked_rss_bytes, rss)
        if n_pairs > self.max_pairs or rss > self.memory_limit_bytes:
            raise AuditError('Resource limit exceeded; no pair truncation is permitted')


class TableStream:
    """Hash exact uncompressed bytes during iteration, with no extra full-file pass."""
    def __init__(self, path, required, whitespace=False):
        self.path = Path(path)
        self.required = tuple(required)
        self.whitespace = whitespace
        self.digest = hashlib.sha256()
        self.n_rows = 0
        self.completed = False
        self.initial_stat = self.path.stat()

    def __iter__(self):
        opener = gzip.open if self.path.suffix == '.gz' else open
        with opener(self.path, 'rb') as stream:
            header = None
            delimiter = None
            for line_number, raw in enumerate(stream, 1):
                self.digest.update(raw)
                text = raw.decode('utf-8').rstrip('\r\n')
                if not text.strip():
                    continue
                if header is None:
                    delimiter = '\t' if '\t' in text or not self.whitespace else None
                fields = next(csv.reader([text], delimiter='\t')) if delimiter else text.split()
                if header is None:
                    header = fields
                    if len(header) != len(set(header)) or not set(self.required).issubset(header):
                        raise AuditError('Missing or duplicate required table columns')
                    continue
                if len(fields) != len(header):
                    raise AuditError(f'Wrong table field count at line {line_number}')
                self.n_rows += 1
                yield dict(zip(header, fields))
            if header is None:
                raise AuditError('Table has no header')
        observed = self.path.stat()
        if (observed.st_size, observed.st_mtime_ns) != (
                self.initial_stat.st_size, self.initial_stat.st_mtime_ns):
            raise AuditError('Input changed while streaming')
        self.completed = True

    def provenance(self):
        if not self.completed:
            raise AuditError('Cannot certify an incompletely read table')
        return dict(path=str(self.path.resolve()), source_bytes=self.initial_stat.st_size,
                    sha256_uncompressed_content=self.digest.hexdigest(), n_rows=self.n_rows,
                    compressed=self.path.suffix == '.gz')


def nonnegative_integer(value, field):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f'Invalid integer in {field}') from exc
    if result < 0:
        raise AuditError(f'Negative value in {field}')
    return result


def load_cohort(path, expected_samples):
    data = Path(path).read_bytes()
    samples = []
    for line in data.decode('utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 1:
            raise AuditError('Keep file must contain exactly one identifier per row')
        samples.append(fields[0])
    if len(samples) != len(set(samples)):
        raise AuditError('Duplicate identifiers in keep file')
    if not samples or len(samples) != expected_samples:
        raise AuditError('Keep cohort size does not match expected-samples')
    return {sample: i for i, sample in enumerate(samples)}, dict(
        n_ids=len(samples), file_sha256=hashlib.sha256(data).hexdigest(),
        ordered_ids_sha256=hashlib.sha256(('\n'.join(samples)+'\n').encode()).hexdigest())


def load_configurations(path):
    stream = TableStream(path, (*CONFIG_FIELDS, *MEASURES))
    configurations = {}
    for row in stream:
        config_id = row['config_id']
        if not config_id or config_id in configurations:
            raise AuditError('Empty or duplicate configuration identifier')
        config = {key: nonnegative_integer(row[key], key)
                  for key in (*CONFIG_FIELDS[1:], *MEASURES)}
        if min(config[key] for key in CONFIG_FIELDS[1:]) == 0:
            raise AuditError('Configuration cutoffs must be positive')
        config['config_id'] = config_id
        configurations[config_id] = config
    if not configurations:
        raise AuditError('No sensitivity configurations supplied')
    return configurations, stream.provenance()


def pair_key(left, right, cohort):
    if left not in cohort or right not in cohort:
        raise AuditError('A sweep pair identifier is absent from the fixed keep cohort')
    a, b = cohort[left], cohort[right]
    if a == b:
        raise AuditError('Self-pair in the sweep table')
    return min(a, b) * len(cohort) + max(a, b), (1 << a) | (1 << b)


def pair_rows(path, configurations, cohort):
    stream = TableStream(path, ('config_id', 'sample_a', 'sample_b', 'n_segments', 'total_shared_bp'))

    def records():
        for row in stream:
            config_id = row['config_id']
            if config_id not in configurations:
                raise AuditError('Unknown configuration in pair table')
            key, ids = pair_key(row['sample_a'], row['sample_b'], cohort)
            n_segments = nonnegative_integer(row['n_segments'], 'n_segments')
            total_bp = nonnegative_integer(row['total_shared_bp'], 'total_shared_bp')
            if n_segments == 0 or total_bp < n_segments:
                raise AuditError('Pair rows require positive segments and inclusive bp support')
            yield config_id, key, ids, n_segments, total_bp
    return records(), stream


def collect_sweep_pairs(path, configurations, cohort, guard):
    config_bits = {name: 1 << i for i, name in enumerate(configurations)}
    wanted = {}
    totals = {name: dict(n_pairs=0, n_segments=0, total_shared_bp=0, ids_mask=0)
              for name in configurations}
    records, stream = pair_rows(path, configurations, cohort)
    for i, (name, key, ids, n_segments, total_bp) in enumerate(records, 1):
        previous = wanted.get(key, 0)
        if previous & config_bits[name]:
            raise AuditError('Duplicate unordered pair/configuration row')
        wanted[key] = previous | config_bits[name]
        target = totals[name]
        target['n_pairs'] += 1
        target['n_segments'] += n_segments
        target['total_shared_bp'] += total_bp
        target['ids_mask'] |= ids
        if i % 65536 == 0:
            guard.check(len(wanted))
    guard.check(len(wanted))
    for name, config in configurations.items():
        if any(totals[name][measure] != config[measure] for measure in MEASURES):
            raise AuditError('Configuration summary and pair table totals disagree')
    for key in wanted:
        wanted[key] = UNSEEN
    return wanted, totals, stream.provenance()


def load_selected_kinship(path, wanted, cohort, guard):
    stream = TableStream(path, ('ID1', 'ID2', 'kin', 'k0', 'k2'), whitespace=True)
    counts = Counter()
    started = time.monotonic()
    ids_seen = 0
    for row in stream:
        left, right = row['ID1'], row['ID2']
        for sample in (left, right):
            if sample in cohort:
                ids_seen |= 1 << cohort[sample]
        if left not in cohort or right not in cohort:
            counts['rows_outside_keep'] += 1
        elif left == right:
            counts['self_rows_ignored'] += 1
        else:
            a, b = cohort[left], cohort[right]
            key = min(a, b) * len(cohort) + max(a, b)
            if key not in wanted:
                counts['in_keep_rows_not_in_sweep'] += 1
            else:
                if wanted[key] is not UNSEEN:
                    raise AuditError('Duplicate PC-Relate row for an unordered sweep pair')
                token = row['kin'].strip()
                if token.lower() in {'', 'na', 'nan', '.', 'n/a'}:
                    value = None
                else:
                    try:
                        value = float(token)
                    except ValueError as exc:
                        raise AuditError('Invalid kinship value for a sweep pair') from exc
                    if not math.isfinite(value):
                        value = None
                wanted[key] = value
                counts['matched_pair_rows'] += 1
                counts['matched_nonfinite_or_missing_kinship'] += value is None
        if stream.n_rows % 500000 == 0:
            guard.check(len(wanted))
            print(f'[rare_segment_kinship_summary] PCRelate_rows={stream.n_rows} '
                  f'elapsed_s={time.monotonic()-started:.1f} '
                  f'rss_mb={current_rss()/1024**2:.1f}', file=sys.stderr, flush=True)
    if stream.n_rows == 0:
        raise AuditError('PC-Relate input has no pair rows')
    guard.check(len(wanted))
    counts['n_ids_in_keep_observed_in_pcrelate'] = ids_seen.bit_count()
    counts['sweep_pairs_absent_from_pcrelate'] = sum(value is UNSEEN for value in wanted.values())
    counts['sweep_pairs_without_finite_kinship'] = sum(value is UNSEEN or value is None
                                                     for value in wanted.values())
    return dict(counts), stream.provenance()


def aggregate_pairs(path, configurations, cohort, wanted, thresholds, totals, initial_hash, guard):
    aggregate = {(name, threshold, group): dict(n_pairs=0, n_segments=0,
                                               total_shared_bp=0, ids_mask=0)
                 for name in configurations for threshold in thresholds for group in GROUPS}
    records, stream = pair_rows(path, configurations, cohort)
    for i, (name, key, ids, n_segments, total_bp) in enumerate(records, 1):
        if key not in wanted:
            raise AuditError('Sweep pair table changed between passes')
        kinship = wanted[key]
        missing = kinship is UNSEEN or kinship is None
        for threshold in thresholds:
            group = 'missing' if missing else ('kin_ge_threshold' if kinship >= threshold else 'kin_lt_threshold')
            result = aggregate[name, threshold, group]
            result['n_pairs'] += 1
            result['n_segments'] += n_segments
            result['total_shared_bp'] += total_bp
            result['ids_mask'] |= ids
        if i % 65536 == 0:
            guard.check(len(wanted))
    if stream.provenance()['sha256_uncompressed_content'] != initial_hash:
        raise AuditError('Sweep pair table content changed between passes')
    output_rows = []
    for name, config in configurations.items():
        for threshold in thresholds:
            for measure in MEASURES:
                if sum(aggregate[name, threshold, group][measure] for group in GROUPS) != totals[name][measure]:
                    raise AuditError('Kinship partitions fail conservation of input support')
            for group in GROUPS:
                result = aggregate[name, threshold, group]
                total = totals[name]
                output_rows.append(dict(
                    **{field: config[field] for field in CONFIG_FIELDS},
                    kinship_threshold=threshold, kinship_group=group,
                    **{measure: result[measure] for measure in MEASURES},
                    n_ids=result['ids_mask'].bit_count(), n_ids_total=total['ids_mask'].bit_count(),
                    **{measure+'_total': total[measure] for measure in MEASURES},
                    **{fraction: result[measure]/total[measure] if total[measure] else None
                       for fraction, measure in [('pair_fraction', 'n_pairs'),
                                                 ('segment_fraction', 'n_segments'),
                                                 ('shared_bp_fraction', 'total_shared_bp')]}))
    return output_rows


def parse_thresholds(text):
    try:
        values = tuple(float(item) for item in text.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Thresholds must be comma-separated finite numbers') from exc
    if not values or len(set(values)) != len(values) or any(not math.isfinite(x) or x < 0 or x > .5 for x in values):
        raise argparse.ArgumentTypeError('Thresholds must be unique finite numbers in [0, 0.5]')
    return tuple(sorted(values))


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configuration-summary', required=True)
    p.add_argument('--pair-configuration-summary', required=True)
    p.add_argument('--pcrelate', required=True)
    p.add_argument('--sample-ids-file', required=True)
    p.add_argument('--expected-samples', type=int, default=2619)
    p.add_argument('--thresholds', type=parse_thresholds, default=(.0221, .0442, .0884, .177))
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-unique-pairs', type=int, default=3_500_000)
    p.add_argument('--max-memory-mb', type=int, default=4096)
    return p


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    guard = Guard(args.max_unique_pairs, args.max_memory_mb)
    try:
        guard.check()
        cohort, cohort_receipt = load_cohort(args.sample_ids_file, args.expected_samples)
        configs, config_source = load_configurations(args.configuration_summary)
        wanted, totals, pair_source = collect_sweep_pairs(
            args.pair_configuration_summary, configs, cohort, guard)
        counts, kinship_source = load_selected_kinship(args.pcrelate, wanted, cohort, guard)
        rows = aggregate_pairs(args.pair_configuration_summary, configs, cohort, wanted,
                               args.thresholds, totals, pair_source['sha256_uncompressed_content'], guard)
        complete = counts['sweep_pairs_without_finite_kinship'] == 0
        aggregate_path = output / 'kinship_by_configuration.tsv'
        with aggregate_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter='\t', lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        summary = dict(
            schema_version=1, status='COMPLETE_DESCRIPTIVE_NOT_INDEPENDENT_VALIDATION' if complete else 'FAILED_INCOMPLETE_PCRELATE',
            generated_utc=datetime.now(timezone.utc).isoformat(),
            cohort=cohort_receipt, n_configurations=len(configs), n_unique_sweep_pairs=len(wanted),
            thresholds=list(args.thresholds), n_output_rows=len(rows), pcrelate_audit=counts,
            sources=dict(configuration_summary=config_source, pair_configuration_summary=pair_source,
                         pcrelate=kinship_source),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
            resource_guard=vars(guard),
            semantics=dict(
                cutoffs='operational pairwise PC-Relate thresholds; not certified families or independent units',
                categories='kin>=threshold versus kin<threshold; missing never reassigned to below-threshold',
                shared_bp='sum of existing segment lengths over unordered pairs; not unique genomic coverage',
                fractions='within each configuration, each measure divided by all input support including missing',
                individuals='unique IDs per category; categories can share individuals so n_ids is not additive',
                independence='PC-Relate includes chr22: descriptive overlap, NOT an independent validation',
                inference='no p-values, no optimal configuration, no segment changes, no IBD/family assignment',
                privacy='aggregate-only outputs; no sample identifiers or pair-level kinships exported'))
        guard.check(len(wanted))
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary
    except (AuditError, OSError, MemoryError, UnicodeError, csv.Error) as exc:
        # Exception messages above contain counts/field names, never source IDs.
        (output/'failure.json').write_text(json.dumps(dict(status='FAILED_INPUT_AUDIT',
                                                         error_type=type(exc).__name__, message=str(exc)), indent=2)+'\n')
        raise


def main(argv=None):
    args = argument_parser().parse_args(argv)
    try:
        summary = run(args)
    except (AuditError, OSError, MemoryError, UnicodeError, csv.Error) as exc:
        print(f'[rare_segment_kinship_summary] {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    return 0 if summary['status'].startswith('COMPLETE_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
