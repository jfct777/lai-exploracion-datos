#!/usr/bin/env python3
"""All fixtures are synthetic; no PC-Relate/DNABR production input is opened."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
import rare_segment_kinship_summary as kin


class KinshipSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.configs = self.root/'configurations.tsv'
        self.pairs = self.root/'pairs.tsv.gz'
        self.pcrelate = self.root/'pcrelate.tsv'
        self.keep = self.root/'keep.txt'
        self.keep.write_text('private_alpha\nprivate_beta\nprivate_gamma\n')
        self.configs.write_text(
            'config_id\tmax_gap_bp\tmin_length_bp\tmin_shared_effective\tn_pairs\tn_segments\ttotal_shared_bp\n'
            'c1\t50000\t100000\t5\t2\t3\t400000\n'
            'c2\t50000\t1000000\t21\t1\t1\t1000000\n'
            'empty\t25000\t2000000\t81\t0\t0\t0\n')
        self.pair_text = (
            'config_id\tsample_a\tsample_b\tn_segments\ttotal_shared_bp\n'
            'c1\tprivate_alpha\tprivate_beta\t2\t300000\n'
            'c1\tprivate_alpha\tprivate_gamma\t1\t100000\n'
            'c2\tprivate_beta\tprivate_alpha\t1\t1000000\n')
        self.write_pairs(self.pair_text)
        self.pc_text = ('ID1\tID2\tkin\tk0\tk2\n'
                        'private_beta\tprivate_alpha\t0.0442\t0\t0\n'
                        'private_alpha\tprivate_gamma\t-0.005\t0\t0\n'
                        'private_beta\tprivate_gamma\t0.7\t0\t0\n'
                        'private_external_zeta\tprivate_alpha\tNA\tNA\tNA\n')
        self.pcrelate.write_text(self.pc_text)

    def write_pairs(self, text):
        with gzip.open(self.pairs, 'wt') as handle:
            handle.write(text)

    def args(self):
        return kin.argument_parser().parse_args([
            '--configuration-summary', str(self.configs),
            '--pair-configuration-summary', str(self.pairs), '--pcrelate', str(self.pcrelate),
            '--sample-ids-file', str(self.keep), '--expected-samples', '3',
            '--output-dir', str(self.root/'result')])

    def rows(self):
        with (self.root/'result'/'kinship_by_configuration.tsv').open() as handle:
            return list(csv.DictReader(handle, delimiter='\t'))

    def row(self, config, threshold, group):
        return next(row for row in self.rows() if row['config_id']==config
                    and float(row['kinship_threshold'])==threshold and row['kinship_group']==group)

    def test_complete_aggregation_inclusive_threshold_and_negative_kinship(self):
        summary = kin.run(self.args())
        self.assertTrue(summary['status'].startswith('COMPLETE_'))
        self.assertEqual(summary['n_unique_sweep_pairs'], 2)
        self.assertEqual(summary['n_output_rows'], 3*4*3)
        above = self.row('c1', .0442, 'kin_ge_threshold')
        below = self.row('c1', .0442, 'kin_lt_threshold')
        self.assertEqual([int(above[k]) for k in ['n_pairs','n_segments','total_shared_bp','n_ids']], [1,2,300000,2])
        self.assertEqual([int(below[k]) for k in ['n_pairs','n_segments','total_shared_bp','n_ids']], [1,1,100000,2])
        self.assertEqual(int(above['n_ids_total']), 3)
        self.assertEqual(float(above['pair_fraction']), .5)
        self.assertAlmostEqual(float(above['segment_fraction']), 2/3)
        self.assertEqual(float(above['shared_bp_fraction']), .75)
        self.assertEqual(int(self.row('c2', .0884, 'kin_ge_threshold')['n_pairs']), 0)
        self.assertEqual(int(self.row('c2', .0884, 'kin_lt_threshold')['total_shared_bp']), 1000000)
        self.assertEqual(summary['pcrelate_audit']['rows_outside_keep'], 1)
        self.assertEqual(summary['pcrelate_audit']['in_keep_rows_not_in_sweep'], 1)
        self.assertEqual(summary['sources']['pcrelate']['sha256_uncompressed_content'],
                         hashlib.sha256(self.pc_text.encode()).hexdigest())

    def test_conservation_for_every_configuration_threshold(self):
        kin.run(self.args())
        rows = self.rows()
        for config in ['c1','c2','empty']:
            for threshold in [.0221,.0442,.0884,.177]:
                subset = [r for r in rows if r['config_id']==config and float(r['kinship_threshold'])==threshold]
                for measure in kin.MEASURES:
                    self.assertEqual(sum(int(r[measure]) for r in subset), int(subset[0][measure+'_total']))

    def test_zero_support_yields_null_fraction_not_division_by_zero(self):
        kin.run(self.args())
        row = self.row('empty', .0221, 'kin_ge_threshold')
        self.assertEqual(row['pair_fraction'], '')
        self.assertEqual(row['shared_bp_fraction'], '')
        self.assertEqual(int(row['n_ids_total']), 0)

    def test_output_contains_no_source_sample_identifiers(self):
        kin.run(self.args())
        for path in (self.root/'result').iterdir():
            content = path.read_text()
            for identifier in ['private_alpha','private_beta','private_gamma','private_external_zeta']:
                self.assertNotIn(identifier, content)

    def test_missing_pair_stays_missing_and_run_is_failed(self):
        self.pcrelate.write_text(self.pc_text.replace('private_alpha\tprivate_gamma\t-0.005\t0\t0\n',''))
        summary = kin.run(self.args())
        self.assertEqual(summary['status'], 'FAILED_INCOMPLETE_PCRELATE')
        self.assertEqual(summary['pcrelate_audit']['sweep_pairs_absent_from_pcrelate'], 1)
        self.assertEqual(int(self.row('c1', .0442, 'missing')['n_segments']), 1)
        self.assertEqual(int(self.row('c1', .0442, 'kin_lt_threshold')['n_pairs']), 0)

    def test_nonfinite_kinship_is_not_assumed_unrelated(self):
        self.pcrelate.write_text(self.pc_text.replace('-0.005', 'NA'))
        summary = kin.run(self.args())
        self.assertEqual(summary['status'], 'FAILED_INCOMPLETE_PCRELATE')
        self.assertEqual(summary['pcrelate_audit']['sweep_pairs_absent_from_pcrelate'], 0)
        self.assertEqual(summary['pcrelate_audit']['matched_nonfinite_or_missing_kinship'], 1)

    def test_duplicate_pcrelate_pair_is_fatal_even_if_equal(self):
        self.pcrelate.write_text(self.pc_text+'private_alpha\tprivate_beta\t0.0442\t0\t0\n')
        with self.assertRaisesRegex(kin.AuditError, 'Duplicate PC-Relate'):
            kin.run(self.args())
        self.assertFalse((self.root/'result'/'summary.json').exists())
        self.assertTrue((self.root/'result'/'failure.json').exists())

    def test_duplicate_sweep_unordered_pair_config_is_fatal(self):
        self.write_pairs(self.pair_text+'c1\tprivate_beta\tprivate_alpha\t2\t300000\n')
        with self.assertRaisesRegex(kin.AuditError, 'Duplicate unordered'):
            kin.run(self.args())

    def test_summary_mismatch_is_detected_before_pc_scan(self):
        self.configs.write_text(self.configs.read_text().replace('400000','400001'))
        with patch.object(kin, 'load_selected_kinship') as loader:
            with self.assertRaisesRegex(kin.AuditError, 'totals disagree'):
                kin.run(self.args())
            loader.assert_not_called()

    def test_outside_keep_sweep_pair_is_fatal_without_exporting_id(self):
        self.write_pairs(self.pair_text.replace('private_gamma','not_in_keep'))
        with self.assertRaisesRegex(kin.AuditError, 'absent from'):
            kin.run(self.args())
        self.assertNotIn('not_in_keep',(self.root/'result'/'failure.json').read_text())

    def test_cohort_size_and_duplicate_keep_are_rejected(self):
        self.keep.write_text('private_alpha\nprivate_alpha\nprivate_gamma\n')
        with self.assertRaisesRegex(kin.AuditError, 'Duplicate identifiers'):
            kin.run(self.args())

    def test_whitespace_pc_format_and_unused_rows_not_retained(self):
        self.pcrelate.write_text(self.pc_text.replace('\t',' '))
        summary = kin.run(self.args())
        self.assertEqual(summary['pcrelate_audit']['matched_pair_rows'], 2)
        self.assertEqual(summary['sources']['pcrelate']['n_rows'], 4)

    def test_threshold_validation(self):
        self.assertEqual(kin.parse_thresholds('.177,.0221'), (.0221,.177))
        for value in ['nan', '.1,.1', '-.1', '.51', 'bad']:
            with self.assertRaises(Exception):
                kin.parse_thresholds(value)

    def test_guard_budget_exceeded_aborts_without_partial_approval(self):
        args = self.args(); args.max_unique_pairs = 1
        with self.assertRaisesRegex(kin.AuditError, 'Resource limit'):
            kin.run(args)
        self.assertFalse((self.root/'result'/'summary.json').exists())

    def test_output_directory_is_create_only(self):
        kin.run(self.args())
        with self.assertRaises(FileExistsError):
            kin.run(self.args())

    def test_content_change_between_sweep_passes_is_fatal(self):
        original = kin.load_selected_kinship
        def change_after_pc(*args, **kwargs):
            result = original(*args, **kwargs)
            self.write_pairs(self.pair_text.replace('300000','300001'))
            return result
        with patch.object(kin, 'load_selected_kinship', side_effect=change_after_pc):
            with self.assertRaisesRegex(kin.AuditError, 'content changed'):
                kin.run(self.args())

    def test_cli_exit_two_when_missing(self):
        args = self.args()
        self.pcrelate.write_text('ID1\tID2\tkin\tk0\tk2\nprivate_beta\tprivate_alpha\t0.0442\t0\t0\n')
        with patch.object(kin, 'argument_parser') as parser:
            parser.return_value.parse_args.return_value = args
            self.assertEqual(kin.main([]), 2)


if __name__ == '__main__':
    unittest.main()
