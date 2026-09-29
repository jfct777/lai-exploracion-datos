#!/usr/bin/env python3
"""Synthetic-only tests for the one-pass M14 sensitivity helper."""
import csv
import gzip
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
try:
    import numpy as np
    import rare_segment_sensitivity as sweep
except ModuleNotFoundError:
    sweep = None


HEADER = (
    '##fileformat=VCFv4.2\n##contig=<ID=22,length=10000000>\n'
    '##dnabr_rare_contract=minor_v1\n'
    '##dnabr_rare_cohort_sha256=' + 'a' * 64 + '\n'
    '##dnabr_rare_cohort_n_samples=3\n'
    '##INFO=<ID=RARE_ALLELE,Number=1,Type=Integer,Description="Source allele">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n'
    '##FORMAT=<ID=RD,Number=1,Type=Integer,Description="Dose">\n'
    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\n'
)


def read_tsv(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


@unittest.skipIf(sweep is None, 'requires the M14 scientific Python stack')
class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        # Intervening nonshared loci MUST NOT break the a/b pair's chain.
        sites = [(10000 + 40000 * i, frozenset({0, 1})) for i in range(36)]
        sites += [(15000, frozenset({1, 2})), (16000, frozenset({2})),
                  (18000, frozenset())]
        sites.sort()
        guard = sweep.ResourceGuard(max_memory_mb=4096)
        collector = sweep.SiteCollector(3, guard)
        for pos, carriers in sites:
            collector(pos, carriers)
        variants = [(pos, carriers) for pos, carriers in sites if len(carriers) >= 2]
        return variants, collector, guard

    def test_geometric_bound_uses_inclusive_coordinates(self):
        self.assertEqual(sweep.geometric_minimum(1, 5), 1)
        self.assertEqual(sweep.geometric_minimum(50001, 50000), 2)
        self.assertEqual(sweep.geometric_minimum(50002, 50000), 3)
        self.assertEqual(sweep.geometric_minimum(1000000, 50000), 21)
        for length, gap in [(0, 1), (1, 0)]:
            with self.assertRaises(ValueError):
                sweep.geometric_minimum(length, gap)

    def test_ninety_nominal_seventy_four_effective(self):
        configs, mapping = sweep.build_grid([100000, 250000, 500000, 1000000, 2000000],
                                            [25000, 50000, 100000], [5, 10, 20, 50, 100, 200])
        self.assertEqual((len(mapping), len(configs)), (90, 74))
        n = [sum(c.min_length_bp == length for c in configs)
             for length in [100000, 250000, 500000, 1000000, 2000000]]
        self.assertEqual(n, [18, 17, 15, 13, 11])
        anchor = next(c for c in configs if c.config_id == 'L1000000_G50000_N21')
        self.assertEqual(anchor.min_shared_nominal_aliases, (5, 10, 20))

    def test_invalid_grid_fails(self):
        for axes in [([], [1], [1]), ([1], [0], [1]), ([1], [1], [-1])]:
            with self.assertRaises(ValueError):
                sweep.build_grid(*axes)

    def test_exact_quantiles_are_pooled_linear_not_average_of_units(self):
        d = sweep.ExactDistribution()
        values = [1, 1, 2, 10, 50, 1000, 1000]
        d.add(values[:2]); d.add(values[2:])
        stats = d.stats()
        for key, q in [('median_distance_bp', .5), ('p90_distance_bp', .9), ('p95_distance_bp', .95)]:
            self.assertAlmostEqual(stats[key], np.quantile(values, q))
        self.assertEqual(stats['mean_distance_bp'], np.mean(values))
        self.assertEqual(sum(row[2] for row in d.histogram((0, 2, 50, 1000))), len(values))
        self.assertEqual(d.histogram((0, 2, 50, 1000))[-1], (1000, 'inf', 2))

    def test_no_distance_for_zero_or_one_locus(self):
        for positions in [[], [10]]:
            stats, gaps = sweep.position_stats(positions)
            self.assertEqual(stats['n_distances'], 0)
            self.assertIsNone(stats['median_distance_bp'])
            self.assertEqual(gaps.size, 0)
        for positions in [[1, 1], [2, 1]]:
            with self.assertRaises(ValueError):
                sweep.position_stats(positions)

    def test_observer_includes_zero_one_carrier_sites(self):
        variants, collector, guard = self.fixture()
        self.assertEqual(len(collector.catalogue), len(variants) + 2)
        self.assertIn(16000, collector.individual_positions[2])
        self.assertEqual(guard.pair_events, 37)

    def test_resource_guards_fail_not_truncate(self):
        guard = sweep.ResourceGuard(max_pair_events=2, max_output_rows=1)
        with self.assertRaises(sweep.ResourceLimitError):
            guard.add_site(3)
        guard.rows()
        with self.assertRaises(sweep.ResourceLimitError):
            guard.rows()
        with patch.object(sweep, 'rss_bytes', return_value=2 * 1024**2):
            with self.assertRaises(sweep.ResourceLimitError):
                sweep.ResourceGuard(max_memory_mb=1).check()

    def test_every_grid_cell_equals_existing_m14_detector(self):
        variants, collector, guard = self.fixture()
        configs, mapping = sweep.build_grid([100000, 250000, 500000, 1000000, 2000000],
                                            [25000, 50000, 100000], [5, 10, 20, 50, 100, 200])
        with redirect_stderr(io.StringIO()):
            anchor = sweep.painter.detect_pairwise_segments_direct('22', variants, ['a', 'b', 'c'],
                                                                   50000, 1000000, 10)
        anchor_path = self.root / 'anchor.tsv.gz'
        anchor.to_csv(anchor_path, index=False, sep='\t', compression='gzip')
        result = sweep.scan_grid(variants, ['a', 'b', 'c'], '22', configs, mapping,
                                 collector, self.root, guard, sweep.load_anchor(anchor_path))
        self.assertTrue(result['anchor_identity']['matched'])
        table = {row['config_id']: row for row in read_tsv(self.root/'configuration_summary.tsv')}
        with redirect_stderr(io.StringIO()):
            for config in configs:
                expected = sweep.painter.detect_pairwise_segments_direct(
                    '22', variants, ['a', 'b', 'c'], config.max_gap_bp, config.min_length_bp,
                    config.min_shared_effective)
                row = table[config.config_id]
                self.assertEqual(int(row['n_segments']), len(expected))
                self.assertEqual(int(row['total_shared_bp']), int(expected.length_bp.sum()))
                expected_pairs = expected.groupby(['sample_a', 'sample_b']).ngroups
                self.assertEqual(int(row['n_pairs']), expected_pairs)
        diagnostics = {row['universe']: row for row in read_tsv(self.root/'distance_summary.tsv')}
        self.assertEqual(int(diagnostics['catalogue_all_rare']['n_positions']), 39)
        self.assertEqual(int(diagnostics['catalogue_all_rare']['n_distances']), 38)
        self.assertEqual(int(diagnostics['pair_shared_before_segment_filter']['n_positions']), 37)
        self.assertEqual(int(diagnostics['pair_shared_before_segment_filter']['n_distances']), 35)
        chains = read_tsv(self.root/'candidate_chains.tsv.gz')
        self.assertTrue(all(int(r['length_bp']) >= 100000 and int(r['n_shared_variants']) >= 5
                            for r in chains))
        all_chain_stats = read_tsv(self.root/'chain_summary.tsv')
        gap25 = next(r for r in all_chain_stats if r['max_gap_bp']=='25000' and r['metric']=='length_bp')
        self.assertEqual(int(gap25['n_chains']), 37)
        self.assertEqual(float(gap25['max']), 1.0)

    def test_anchor_mismatch_is_fatal(self):
        variants, collector, guard = self.fixture()
        configs, mapping = sweep.build_grid([100000], [50000], [5])
        with self.assertRaisesRegex(ValueError, 'Anchor segment identity'):
            sweep.scan_grid(variants, ['a', 'b', 'c'], '22', configs, mapping,
                            collector, self.root, guard, sweep.Counter())

    def test_anchor_is_checked_even_if_not_in_grid(self):
        variants, collector, guard = self.fixture()
        configs, mapping = sweep.build_grid([2000000], [25000], [200])
        result = sweep.scan_grid(variants, ['a', 'b', 'c'], '22', configs, mapping,
                                 collector, self.root, guard)
        self.assertEqual(result['anchor_identity']['observed_n_segments'], 1)
        self.assertEqual(result['n_effective_configurations'], 1)

    def test_output_is_create_only(self):
        args = sweep.argument_parser().parse_args([
            '--input', 'unused', '--chr', '22', '--sample-ids-file', 'unused',
            '--output-dir', str(self.root), '--anchor-segments', 'unused', '--anchor-summary', 'unused'])
        with self.assertRaises(FileExistsError):
            sweep.run(args)

    @unittest.skipUnless(shutil.which('bcftools'), 'requires bcftools')
    def test_real_bcftools_one_pass_ref_minor_and_complete_output(self):
        vcf = self.root/'fixture.vcf'
        rows = []
        for i in range(31):
            # REF counts must not be replaced by ALT counts in subset processing.
            rare = i % 2
            gt = '0/0:2\t0|1:1\t1/1:0' if rare == 0 else '1/1:2\t0|1:1\t0/0:0'
            rows.append(f'22\t{10000+i*40000}\t.\tA\tC\t.\tPASS\tRARE_ALLELE={rare}\tGT:RD\t{gt}\n')
        vcf.write_text(HEADER+''.join(rows))
        keep = self.root/'keep.txt'; keep.write_text('a\nb\nc\n')
        summary = dict(chrom='22', carrier_allele_mode='source_minor', selected_samples=['a','b','c'],
                       source_rare_contract=dict(contract='minor_v1', cohort_sha256='a'*64, cohort_n_samples=3),
                       parameters_used=dict(min_segment_bp=1000000, max_gap_bp=50000, min_shared_variants=10),
                       n_segments=1)
        anchor_summary = self.root/'anchor.json'; anchor_summary.write_text(json.dumps(summary))
        anchor = self.root/'anchor.tsv'
        anchor.write_text('chrom\tsample_a\tsample_b\tstart_pos\tend_pos\tlength_bp\tn_shared_variants\n'
                          '22\ta\tb\t10000\t1210000\t1200001\t31\n')
        args = sweep.argument_parser().parse_args([
            '--input', str(vcf), '--chr', '22', '--sample-ids-file', str(keep),
            '--expected-samples', '3', '--output-dir', str(self.root/'output'),
            '--anchor-segments', str(anchor), '--anchor-summary', str(anchor_summary),
            '--lengths-bp', '1000000', '--gaps-bp', '50000', '--min-shared', '10'])
        original = sweep.painter.parse_genotypes_carrier_sets
        with patch.object(sweep.painter, 'parse_genotypes_carrier_sets', wraps=original) as loader:
            result = sweep.run(args)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(result['orientation_qc']['source_ref_sites'], 16)
        self.assertEqual(result['orientation_qc']['source_alt_sites'], 15)
        self.assertEqual(result['anchor_identity']['observed_n_segments'], 1)
        self.assertTrue((self.root/'output'/'summary.json').exists())
        self.assertFalse((self.root/'output'/'failure.json').exists())
        self.assertNotIn('selected_samples', result)


if __name__ == '__main__':
    unittest.main()
