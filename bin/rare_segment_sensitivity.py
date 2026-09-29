#!/usr/bin/env python3
"""One-pass, source_minor-only M14 segment sensitivity (not an IBD significance test).

The caller (normally Nextflow) freezes samples, upstream QC and the grid. Carrier
parsing and maximal-chain detection are the existing M14 implementations. One
VCF genotype pass feeds three DISTINCT distance populations: all rare loci,
within-individual carrier loci, and within-pair shared loci. Only positive,
within-chromosome consecutive-position distances enter their distributions.

Pair intersections are computed once; maximal chains are enumerated once per
gap. All chains contribute to diagnostics, including singletons. The chain TSV
stores only candidates meeting the grid's smallest length/count, which cannot
remove a final result. Effective thresholds are geometric constraints, not
independent evidence, p-values, LD pruning, or a recommended optimal gap.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp

import rare_allele_sharing_painter as painter


DISTANCE_BINS = (0, 1000, 5000, 10000, 25000, 50000, 100000, 200000,
                 500000, 1000000, 2000000, 5000000, 10000000, 50000000)
COUNT_BINS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 10000)
UNIVERSES = ("catalogue_all_rare", "individual_carrier_all_rare",
             "pair_shared_before_segment_filter")
CONFIG_COLUMNS = ("config_id", "max_gap_bp", "min_length_bp", "min_shared_effective",
                  "min_shared_nominal_aliases", "n_geo", "n_segments", "n_pairs",
                  "total_shared_bp", "n_shared_variants_total", "n_segments_lt_1mb",
                  "total_shared_bp_lt_1mb", "mean_length_bp",
                  "min_length_observed_bp", "max_length_observed_bp")
DISTANCE_COLUMNS = ("universe", "n_units", "n_positions", "n_distances",
                    "mean_distance_bp", "median_distance_bp", "p90_distance_bp",
                    "p95_distance_bp", "min_distance_bp", "max_distance_bp", "denominator")


class ResourceLimitError(RuntimeError):
    """An operational budget was exceeded: fail the whole run, never truncate."""


@dataclass(frozen=True)
class Configuration:
    config_id: str
    max_gap_bp: int
    min_length_bp: int
    min_shared_effective: int
    min_shared_nominal_aliases: tuple[int, ...]
    n_geo: int


def geometric_minimum(length_bp, gap_bp):
    if length_bp < 1 or gap_bp < 1:
        raise ValueError("Length and gap must be positive integers in bp")
    return (int(length_bp) - 2 + int(gap_bp)) // int(gap_bp) + 1


def build_grid(lengths, gaps, nominal_counts):
    axes = [sorted(set(map(int, values))) for values in (lengths, gaps, nominal_counts)]
    if any(not values or min(values) < 1 for values in axes):
        raise ValueError("Grid axes must be nonempty positive integers")
    configurations, mapping = [], []
    for gap in axes[1]:
        for length in axes[0]:
            n_geo = geometric_minimum(length, gap)
            groups = {}
            for nominal in axes[2]:
                groups.setdefault(max(nominal, n_geo), []).append(nominal)
            for effective, aliases in groups.items():
                config_id = f"L{length}_G{gap}_N{effective}"
                configurations.append(Configuration(config_id, gap, length, effective,
                                                     tuple(aliases), n_geo))
                mapping.extend(dict(config_id=config_id, max_gap_bp=gap,
                                    min_length_bp=length, min_shared_nominal=nominal,
                                    min_shared_effective=effective, n_geo=n_geo)
                               for nominal in aliases)
    return configurations, mapping


def rss_bytes():
    # Current RSS, not VMS (scientific libraries can reserve large address spaces).
    try:
        import os
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def progress(stage, completed, started, total=None):
    denominator = f'/{total}' if total is not None else ''
    print(f'[rare_segment_sensitivity] {stage}={completed}{denominator} '
          f'elapsed_s={time.monotonic()-started:.1f} rss_mb={rss_bytes()/1024**2:.1f}',
          file=sys.stderr, flush=True)


class ResourceGuard:
    def __init__(self, max_pair_events=250_000_000, max_memory_mb=12288,
                 max_output_rows=25_000_000):
        if min(max_pair_events, max_memory_mb, max_output_rows) <= 0:
            raise ValueError("Resource budgets must be positive")
        self.max_pair_events = int(max_pair_events)
        self.max_memory_bytes = int(max_memory_mb * 1024**2)
        self.max_output_rows = int(max_output_rows)
        self.pair_events = self.output_rows = self.peak_checked_rss_bytes = 0

    def check(self, additional_bytes=0):
        current = rss_bytes()
        self.peak_checked_rss_bytes = max(self.peak_checked_rss_bytes, current)
        if current + additional_bytes > self.max_memory_bytes:
            raise ResourceLimitError("Memory budget exceeded (RSS plus estimated next allocation)")

    def add_site(self, n_carriers):
        self.pair_events += n_carriers * (n_carriers - 1) // 2
        if self.pair_events > self.max_pair_events:
            raise ResourceLimitError("max-pair-events exceeded; no sample/locus/pair truncation allowed")

    def rows(self, n=1):
        self.output_rows += int(n)
        if self.output_rows > self.max_output_rows:
            raise ResourceLimitError("max-output-rows exceeded; partial outputs are NOT complete")


class ExactDistribution:
    """Exact integer-frequency distribution; quantiles match numpy linear interpolation."""
    def __init__(self):
        self.counts = Counter()
        self.n = self.total = 0

    def add(self, values):
        values = np.asarray(values, dtype=np.int64)
        if values.size == 0:
            return
        if np.any(values <= 0):
            raise ValueError("Only positive distances/lengths/counts are allowed")
        keys, counts = np.unique(values, return_counts=True)
        self.counts.update({int(k): int(n) for k, n in zip(keys, counts)})
        self.n += int(values.size)
        self.total += int(values.sum())

    def repeat(self, value, count):
        """Accumulate a proven repeated value without expanding observations."""
        if value <= 0 or count < 0:
            raise ValueError("Repeated observations must have positive value/nonnegative count")
        if count:
            self.counts[int(value)] += int(count)
            self.n += int(count)
            self.total += int(value) * int(count)

    def stats(self):
        if not self.n:
            return dict(n_distances=0, mean_distance_bp=None, median_distance_bp=None,
                        p90_distance_bp=None, p95_distance_bp=None,
                        min_distance_bp=None, max_distance_bp=None)
        keys = np.array(sorted(self.counts), dtype=np.int64)
        cumulative = np.cumsum([self.counts[int(k)] for k in keys], dtype=np.int64)

        def quantile(q):
            rank = (self.n - 1) * q
            lo, hi = math.floor(rank), math.ceil(rank)
            a = int(keys[np.searchsorted(cumulative, lo + 1)])
            b = int(keys[np.searchsorted(cumulative, hi + 1)])
            return a + (b - a) * (rank - lo)

        return dict(n_distances=self.n, mean_distance_bp=self.total / self.n,
                    median_distance_bp=quantile(.5), p90_distance_bp=quantile(.9),
                    p95_distance_bp=quantile(.95), min_distance_bp=int(keys[0]),
                    max_distance_bp=int(keys[-1]))

    def histogram(self, edges):
        counts = np.zeros(len(edges), dtype=np.int64)
        for value, count in self.counts.items():
            counts[np.searchsorted(edges, value, side='right') - 1] += count
        return [(int(left), int(edges[i + 1]) if i + 1 < len(edges) else 'inf', int(counts[i]))
                for i, left in enumerate(edges)]


def position_stats(positions):
    values = np.asarray(positions, dtype=np.int64)
    differences = np.diff(values)
    if np.any(differences <= 0):
        raise ValueError("Positions must be strictly increasing within one chromosome")
    distribution = ExactDistribution()
    distribution.add(differences)
    return distribution.stats(), differences


class SiteCollector:
    """Observer called by the M14 parser AFTER its source_minor GT/RD validation."""
    def __init__(self, n_samples, guard):
        self.guard = guard
        self.started = time.monotonic()
        self.catalogue = array('q')
        self.individual_positions = [array('q') for _ in range(n_samples)]

    def __call__(self, pos, carriers):
        if self.catalogue and pos <= self.catalogue[-1]:
            raise ValueError("Duplicate/unsorted source locus")
        self.guard.add_site(len(carriers))
        self.catalogue.append(pos)
        for index in carriers:
            self.individual_positions[index].append(pos)
        if len(self.catalogue) % 4096 == 0:
            self.guard.check()
        if len(self.catalogue) % 50000 == 0:
            progress('source_minor_sites', len(self.catalogue), self.started)


def open_table(stack, path, columns):
    opener = gzip.open if str(path).endswith('.gz') else open
    handle = stack.enter_context(opener(path, 'wt', encoding='utf-8', newline=''))
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t', lineterminator='\n')
    writer.writeheader()
    return writer


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def segment_key(chrom, a, b, start, end, length, n_shared):
    return (str(chrom).removeprefix('chr'), *sorted((str(a), str(b))),
            int(start), int(end), int(length), int(n_shared))


def load_anchor(path, max_rows=1_000_000):
    opener = gzip.open if str(path).endswith('.gz') else open
    records = Counter()
    with opener(path, 'rt') as handle:
        for i, row in enumerate(csv.DictReader(handle, delimiter='\t')):
            if i >= max_rows:
                raise ResourceLimitError("Anchor table exceeds bounded comparison size")
            records[segment_key(row['chrom'], row['sample_a'], row['sample_b'],
                                row['start_pos'], row['end_pos'], row['length_bp'],
                                row['n_shared_variants'])] += 1
    return records


def validate_anchor(summary, selected_samples, contract, chrom):
    if summary.get('carrier_allele_mode') != 'source_minor':
        raise ValueError("Anchor must use source_minor, not historical/recomputed ALT orientation")
    if str(summary.get('chrom')).removeprefix('chr') != str(chrom).removeprefix('chr'):
        raise ValueError("Anchor chromosome mismatch")
    if summary.get('selected_samples') != selected_samples:
        raise ValueError("Anchor cohort/order mismatch")
    if summary.get('source_rare_contract') != contract:
        raise ValueError("Anchor source-cohort contract mismatch")
    p = summary.get('parameters_used', {})
    for key, value in [('min_segment_bp', 1_000_000), ('max_gap_bp', 50_000),
                       ('min_shared_variants', 10)]:
        if p.get(key) != value:
            raise ValueError(f"Unexpected anchor parameter: {key}")


def scan_grid(variants, samples, chrom, configurations, mapping, collector, output_dir,
              guard, anchor_records=None, distance_bins=DISTANCE_BINS):
    """Analyze already authenticated carrier sets. Tests use synthetic sets only."""
    output_dir = Path(output_dir)
    started = time.monotonic()
    gaps = sorted({c.max_gap_bp for c in configurations} | {50_000})
    groups = {gap: [(i, c) for i, c in enumerate(configurations) if c.max_gap_bp == gap]
              for gap in gaps}
    min_length = min(c.min_length_bp for c in configurations)
    min_count = min(min(c.min_shared_nominal_aliases) for c in configurations)
    incidences = sum(len(cs) for _, cs in variants)
    # Conservative temporary-allocation estimate, NOT a claim of measured peak RAM.
    guard.check(incidences * 40 + len(samples)**2 * 64)
    C, positions = painter._build_carrier_matrix(variants, len(samples))
    S = (C @ C.T).tocsr()
    triangle = sp.triu(S, k=1).tocoo()
    order = np.lexsort((triangle.col, triangle.row))
    pair_a, pair_b = triangle.row[order], triangle.col[order]
    pair_counts = triangle.data[order]
    observed_events = int(triangle.data.sum())
    if observed_events != guard.pair_events:
        raise ValueError("Pair-event preflight and carrier matrix disagree")
    del S, triangle, order
    guard.check()
    progress('pair_events_preflight', observed_events, started, guard.max_pair_events)
    totals = [dict(n_segments=0, n_pairs=0, total_shared_bp=0, n_shared_variants_total=0,
                   n_segments_lt_1mb=0, total_shared_bp_lt_1mb=0,
                   min_length_observed_bp=None, max_length_observed_bp=None)
              for _ in configurations]
    pooled = {u: ExactDistribution() for u in UNIVERSES}
    pooled_positions = {UNIVERSES[0]: len(collector.catalogue), UNIVERSES[1]: 0,
                        UNIVERSES[2]: observed_events}
    chain_lengths = {gap: ExactDistribution() for gap in gaps}
    chain_counts = {gap: ExactDistribution() for gap in gaps}
    n_singleton_pairs = int(np.count_nonzero(pair_counts == 1))
    # A pair sharing exactly one locus has one (1 bp, 1 marker) chain for EVERY
    # gap, no distance, and cannot satisfy a grid cell with minimum length >1.
    # Skip its intersection entirely while retaining exact diagnostic counts.
    singleton_fast_path = min_length > 1
    if singleton_fast_path:
        for gap in gaps:
            chain_lengths[gap].repeat(1, n_singleton_pairs)
            chain_counts[gap].repeat(1, n_singleton_pairs)
    anchor_observed = Counter()
    chain_candidate_rows = pair_config_rows = 0
    with ExitStack() as stack:
        nominal = open_table(stack, output_dir / 'nominal_to_effective.tsv',
                             tuple(mapping[0]))
        nominal.writerows(mapping)
        individual = open_table(stack, output_dir / 'individual_distance_summary.tsv',
                                ('sample_id', 'n_positions', *tuple(ExactDistribution().stats())))
        pair_distance = open_table(stack, output_dir / 'pair_distance_summary.tsv.gz',
                                   ('sample_a', 'sample_b', 'n_positions',
                                    *tuple(ExactDistribution().stats())))
        chain_writer = open_table(stack, output_dir / 'candidate_chains.tsv.gz',
                                  ('chrom', 'sample_a', 'sample_b', 'max_gap_bp', 'start_pos',
                                   'end_pos', 'length_bp', 'n_shared_variants', 'n_geo_for_span'))
        pair_config = open_table(stack, output_dir / 'pair_configuration_summary.tsv.gz',
                                 ('config_id', 'sample_a', 'sample_b', 'n_segments',
                                  'total_shared_bp', 'n_shared_variants_total',
                                  'n_segments_lt_1mb', 'total_shared_bp_lt_1mb',
                                  'max_segment_bp'))
        _, distances = position_stats(collector.catalogue)
        pooled[UNIVERSES[0]].add(distances)
        for i, sample in enumerate(samples):
            sample_positions = collector.individual_positions[i]
            stats, distances = position_stats(sample_positions)
            pooled[UNIVERSES[1]].add(distances)
            pooled_positions[UNIVERSES[1]] += len(sample_positions)
            individual.writerow(dict(sample_id=sample, n_positions=len(sample_positions), **stats))
            collector.individual_positions[i] = array('q')
            guard.check()
        for pi, (a, b) in enumerate(zip(pair_a, pair_b)):
            if pi % 100000 == 0:
                progress('pairs', pi, started, len(pair_a))
            a, b = int(a), int(b)
            if singleton_fast_path and pair_counts[pi] == 1:
                guard.rows()
                pair_distance.writerow(dict(sample_a=samples[a], sample_b=samples[b],
                                             n_positions=1, **ExactDistribution().stats()))
                if pi % 128 == 0:
                    guard.check()
                continue
            shared = np.intersect1d(C.indices[C.indptr[a]:C.indptr[a + 1]],
                                    C.indices[C.indptr[b]:C.indptr[b + 1]], assume_unique=True)
            shared_positions = positions[shared]
            stats, distances = position_stats(shared_positions)
            pooled[UNIVERSES[2]].add(distances)
            guard.rows()
            pair_distance.writerow(dict(sample_a=samples[a], sample_b=samples[b],
                                         n_positions=int(shared.size), **stats))
            for gap in gaps:
                chains = np.asarray(list(painter._detect_segments_for_pair(
                    shared_positions, gap, 1, 1)), dtype=np.int64).reshape(-1, 4)
                lengths, counts = chains[:, 2], chains[:, 3]
                chain_lengths[gap].add(lengths)
                chain_counts[gap].add(counts)
                candidate_mask = (lengths >= min_length) & (counts >= min_count)
                for start, end, length, count in chains[candidate_mask]:
                    guard.rows()
                    chain_candidate_rows += 1
                    chain_writer.writerow(dict(chrom=chrom, sample_a=samples[a], sample_b=samples[b],
                                                max_gap_bp=gap, start_pos=int(start), end_pos=int(end),
                                                length_bp=int(length), n_shared_variants=int(count),
                                                n_geo_for_span=geometric_minimum(length, gap)))
                if gap == 50_000:
                    for start, end, length, count in chains[(lengths >= 1_000_000) & (counts >= 10)]:
                        anchor_observed[segment_key(chrom, samples[a], samples[b], start,
                                                    end, length, count)] += 1
                if not np.any(candidate_mask):
                    continue  # Most pairs cannot enter ANY grid cell; retain their diagnostics.
                for ci, config in groups[gap]:
                    mask = (lengths >= config.min_length_bp) & (counts >= config.min_shared_effective)
                    if not np.any(mask):
                        continue
                    selected_lengths = lengths[mask]
                    short = selected_lengths < 1_000_000
                    values = dict(n_segments=int(mask.sum()), n_pairs=1,
                                  total_shared_bp=int(selected_lengths.sum()),
                                  n_shared_variants_total=int(counts[mask].sum()),
                                  n_segments_lt_1mb=int(short.sum()),
                                  total_shared_bp_lt_1mb=int(selected_lengths[short].sum()))
                    total = totals[ci]
                    for key, value in values.items():
                        total[key] += value
                    lo, hi = int(selected_lengths.min()), int(selected_lengths.max())
                    total['min_length_observed_bp'] = min(total['min_length_observed_bp'] or lo, lo)
                    total['max_length_observed_bp'] = max(total['max_length_observed_bp'] or hi, hi)
                    guard.rows()
                    pair_config_rows += 1
                    pair_config.writerow(dict(config_id=config.config_id, sample_a=samples[a],
                                               sample_b=samples[b], max_segment_bp=hi,
                                               **{k: v for k, v in values.items() if k != 'n_pairs'}))
            if pi % 128 == 0:
                guard.check()
        progress('pairs', len(pair_a), started, len(pair_a))
        summary_writer = open_table(stack, output_dir / 'configuration_summary.tsv', CONFIG_COLUMNS)
        for config, values in zip(configurations, totals):
            row = asdict(config)
            row['min_shared_nominal_aliases'] = ','.join(map(str, config.min_shared_nominal_aliases))
            summary_writer.writerow(dict(**row, **values, mean_length_bp=(
                values['total_shared_bp'] / values['n_segments'] if values['n_segments'] else None)))
        dist_summary = open_table(stack, output_dir / 'distance_summary.tsv', DISTANCE_COLUMNS)
        histogram = open_table(stack, output_dir / 'distance_histogram.tsv',
                               ('universe', 'bin_start_bp', 'bin_end_bp', 'count'))
        for universe, units in zip(UNIVERSES, [1, len(samples), len(pair_a)]):
            stats = pooled[universe].stats()
            dist_summary.writerow(dict(universe=universe, n_units=units,
                                        n_positions=pooled_positions[universe], **stats,
                                        denominator='pooled_positive_within_unit_consecutive_distances'))
            histogram.writerows(dict(universe=universe, bin_start_bp=lo, bin_end_bp=hi, count=n)
                                for lo, hi, n in pooled[universe].histogram(distance_bins))
        chain_hist = open_table(stack, output_dir / 'chain_histogram.tsv',
                                ('max_gap_bp', 'metric', 'bin_start', 'bin_end', 'count'))
        chain_summary = open_table(stack, output_dir / 'chain_summary.tsv',
                                   ('max_gap_bp', 'metric', 'n_chains', 'mean', 'median', 'p90',
                                    'p95', 'min', 'max'))
        for gap in gaps:
            for metric, dist, bins in [('length_bp', chain_lengths[gap], distance_bins),
                                       ('n_shared_variants', chain_counts[gap], COUNT_BINS)]:
                stats = dist.stats()
                chain_summary.writerow(dict(max_gap_bp=gap, metric=metric, n_chains=dist.n,
                                             **{out: stats[key] for out, key in [
                                                 ('mean', 'mean_distance_bp'), ('median', 'median_distance_bp'),
                                                 ('p90', 'p90_distance_bp'), ('p95', 'p95_distance_bp'),
                                                 ('min', 'min_distance_bp'), ('max', 'max_distance_bp')]}))
                chain_hist.writerows(dict(max_gap_bp=gap, metric=metric, bin_start=lo,
                                           bin_end=hi, count=n) for lo, hi, n in dist.histogram(bins))
    anchor_match = anchor_records is None or anchor_records == anchor_observed
    anchor = dict(requested=anchor_records is not None, matched=anchor_match,
                  observed_n_segments=sum(anchor_observed.values()),
                  expected_n_segments=sum(anchor_records.values()) if anchor_records is not None else None,
                  comparison='multiset(chrom,unordered_pair,start,end,inclusive_length,n_shared); ignore segment_id')
    if not anchor_match:
        raise ValueError("Anchor segment identity failed; outputs are not approved/complete")
    return dict(n_effective_configurations=len(configurations), n_nominal_configurations=len(mapping),
                n_pairs_with_any_shared_locus=len(pair_a), n_pair_events=observed_events,
                distance_pair_denominators=dict(total_possible_pairs=len(samples)*(len(samples)-1)//2,
                                                pairs_with_zero_shared_loci=len(samples)*(len(samples)-1)//2-len(pair_a),
                                                pairs_with_one_shared_locus=n_singleton_pairs,
                                                pairs_with_distances=len(pair_a)-n_singleton_pairs),
                candidate_chain_rows=chain_candidate_rows, pair_configuration_rows=pair_config_rows,
                candidate_chain_tsv_filter=dict(min_length_bp=min_length, min_shared_variants=min_count,
                                                preserves_all_grid_results=True),
                anchor_identity=anchor, n_distance_observations={u: pooled[u].n for u in UNIVERSES})


def positive_csv(value):
    try:
        values = tuple(int(v) for v in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from exc
    if not values or min(values) < 1 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Grid values must be unique positive integers")
    return values


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--chr', required=True)
    p.add_argument('--sample-ids-file', required=True)
    p.add_argument('--expected-samples', type=int, default=2619)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--gaps-bp', type=positive_csv, default=(25000, 50000, 100000))
    p.add_argument('--lengths-bp', type=positive_csv, default=(100000, 250000, 500000, 1000000, 2000000))
    p.add_argument('--min-shared', type=positive_csv, default=(5, 10, 20, 50, 100, 200))
    p.add_argument('--anchor-segments', required=True)
    p.add_argument('--anchor-summary', required=True)
    p.add_argument('--max-pair-events', type=int, default=250_000_000)
    p.add_argument('--max-memory-mb', type=int, default=12288)
    p.add_argument('--max-output-rows', type=int, default=25_000_000)
    return p


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    guard = ResourceGuard(args.max_pair_events, args.max_memory_mb, args.max_output_rows)
    try:
        guard.check()
        painter.validate_input_schema(args.input, 'vcf_rare')
        contract = painter.read_source_minor_contract(args.input, 'source_minor')
        header_samples = painter._read_header_samples(args.input)
        samples = painter.load_selected_samples(header_samples, args.sample_ids_file, None)
        if len(samples) != args.expected_samples:
            raise ValueError("Selected sample count differs from expected-samples")
        anchor_summary = json.loads(Path(args.anchor_summary).read_text())
        validate_anchor(anchor_summary, samples, contract, args.chr)
        anchor_records = load_anchor(args.anchor_segments)
        if sum(anchor_records.values()) != anchor_summary.get('n_segments'):
            raise ValueError("Anchor summary/segment row count mismatch")
        configs, mapping = build_grid(args.lengths_bp, args.gaps_bp, args.min_shared)
        collector = SiteCollector(len(samples), guard)
        chrom, variants, total, lo, hi, orientation = painter.parse_genotypes_carrier_sets(
            args.input, str(args.chr).removeprefix('chr'), samples, 'source_minor',
            return_orientation_qc=True, site_callback=collector)
        if len(collector.catalogue) != total:
            raise ValueError("Site observer did not receive the complete rare catalogue")
        result = scan_grid(variants, samples, str(chrom).removeprefix('chr'), configs, mapping,
                           collector, output, guard, anchor_records)
        result.update(schema_version=1, status='COMPLETE_EXPLORATORY_NOT_VALIDATED',
                      generated_utc=datetime.now(timezone.utc).isoformat(), chrom=str(chrom),
                      n_samples=len(samples), total_variants_in_input=total,
                      n_shared_carrier_variants=len(variants), chrom_extent=[lo, hi],
                      carrier_allele_mode='source_minor', source_rare_contract=contract,
                      selected_samples_order_sha256=hashlib.sha256(('\n'.join(samples)+'\n').encode()).hexdigest(),
                      orientation_qc=orientation,
                      input=dict(path=str(Path(args.input).resolve()), bytes=Path(args.input).stat().st_size,
                                 genotype_passes=1, full_input_hash_recomputed=False),
                      source_sha256={Path(__file__).name: sha256(__file__),
                                     Path(painter.__file__).name: sha256(painter.__file__)},
                      anchor_sha256=dict(summary=sha256(args.anchor_summary),
                                         segments=sha256(args.anchor_segments)),
                      resource_guard=vars(guard),
                      semantics=dict(total_shared_bp='sum of inclusive segment lengths over unordered pairs; not unique genomic coverage',
                                     distances='pooled positive consecutive distances within each unit, before any segment length/count filtering',
                                     zero_distance_policy='reject duplicates, not silently discard',
                                     zero_shared_pairs='absent from pair distance table; have no inter-event distance',
                                     chains='all maximal chains counted in histograms; candidate TSV uses only grid lower bounds',
                                     anchor='always checked separately at L=1Mb/G=50kb/n=10, even if absent from grid',
                                     n_geo='necessary geometry, not statistical evidence',
                                     inference='descriptive sensitivity only; no optimal gap, significance, phasing, LD pruning or population validation'))
        guard.check()
        (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return result
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(
            status='FAILED_NO_COMPLETE_RESULTS', error_type=type(exc).__name__, message=str(exc),
            resource_guard=vars(guard)), indent=2)+'\n')
        raise


def main(argv=None):
    args = argument_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, ResourceLimitError, MemoryError, OSError) as exc:
        print(f'[rare_segment_sensitivity] {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
