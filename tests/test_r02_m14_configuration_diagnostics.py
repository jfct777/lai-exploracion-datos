"""Synthetic-only M14 ledger, streaming order, provenance and denominator tests."""
import csv
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import r02_m14_configuration_diagnostics as diagnostics
import m165_autosome_sweep as aggregation


class ConfigurationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.samples = self.root / "samples.txt"
        # Deliberately not lexical: aggregate and raw M14 use different orders.
        self.samples.write_text("D\nC\nA\nB\nE\n")
        self.pairs = self.root / "pair_configuration_summary.tsv.gz"
        self.configs = self.root / "configuration_summary.tsv"
        self.receipt = self.root / "aggregation.json"
        self.kin = self.root / "kin.tsv"
        self.kin.write_text("ID1\tID2\tkin\nA\tB\t0.05\nA\tC\t0.03\nA\tD\t0.01\nB\tC\tNA\n")
        self.config = "L10_G10_N2"
        self.zero = "L20_G10_N3"
        self.rows = [dict(config_id=self.config, sample_a=a, sample_b=b, n_segments=1,
                          total_shared_bp=bp, n_shared_variants_total=2, max_segment_bp=bp)
                     for a, b, bp in [("A", "B", 100), ("A", "C", 200), ("A", "D", 500),
                                      ("B", "C", 300), ("C", "D", 400)]]
        self.ledger = [dict(config_id=self.config, max_gap_bp=10, min_length_bp=10,
                           min_shared_effective=2, n_pairs=5, n_segments=5,
                           total_shared_bp=1500, n_shared_variants_total=10),
                       dict(config_id=self.zero, max_gap_bp=10, min_length_bp=20,
                            min_shared_effective=3, n_pairs=0, n_segments=0,
                            total_shared_bp=0, n_shared_variants_total=0)]
        self.write_inputs()

    def write_inputs(self):
        with gzip.open(self.pairs, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostics.PAIR_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.rows)
        with self.configs.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=diagnostics.CONFIG_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.ledger)
        self.receipt.write_text(json.dumps(dict(status="COMPLETE_AUTOSOMAL_AGGREGATION",
            chromosomes=diagnostics.AUTOSOMES, n_samples=5,
            sample_ids_sha256=diagnostics.sweep.sha256(self.samples),
            selected_source_configurations=sorted(row["config_id"] for row in self.ledger),
            source_rare_contract=dict(contract="minor_v1", cohort_n_samples=5, cohort_sha256="f"*64),
            outputs_sha256={self.pairs.name: diagnostics.sweep.sha256(self.pairs),
                            self.configs.name: diagnostics.sweep.sha256(self.configs)})))

    def args(self):
        return dict(pair_summary=self.pairs, configuration_summary=self.configs,
                    sample_ids=self.samples, expected_samples=5, expected_configurations=len(self.ledger),
                    pcrelate_file=self.kin, expected_pcrelate_sha256=diagnostics.sweep.sha256(self.kin),
                    aggregation_receipt=self.receipt, output_dir=self.root / "diagnostics")

    def execute(self, **updates):
        args = self.args()
        args.update(updates)
        return diagnostics.run(**args)

    def read_rows(self, name="configuration_diagnostics.tsv"):
        with (self.root / "diagnostics" / name).open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def test_denominators_lengths_log_weight_and_unknown_states(self):
        result = self.execute()
        self.assertEqual(result["status"], diagnostics.STATUS)
        rows = self.read_rows()
        row = next(row for row in rows if row["config_id"] == self.config and row["kinship_threshold"] == "0.0221")
        self.assertEqual(int(row["n_people"]), 5)
        self.assertEqual(int(row["n_possible_pairs"]), 10)
        self.assertEqual(int(row["n_active"]), 4)
        self.assertEqual(int(row["n_isolated"]), 1)
        self.assertEqual(int(row["n_kin_ge_threshold"]), 2)
        self.assertEqual(int(row["n_missing_absent"]), 1)
        self.assertEqual(int(row["n_missing_nonfinite"]), 1)
        self.assertAlmostEqual(float(row["fraction_n_kin_ge_all_pairs"]), 2/5)
        self.assertAlmostEqual(float(row["fraction_n_kin_ge_observed_pairs"]), 2/3)
        self.assertAlmostEqual(float(row["fraction_bp_kin_ge_all_pairs"]), 300/1500)
        self.assertAlmostEqual(float(row["fraction_bp_kin_ge_observed_pairs"]), 300/800)
        total_weight = sum(math.log1p(bp) for bp in (100, 200, 300, 400, 500))
        max_degree = sum(math.log1p(bp) for bp in (200, 300, 400))
        self.assertAlmostEqual(float(row["sum_weight"]), total_weight)
        self.assertAlmostEqual(float(row["max_weighted_degree_share"]), max_degree/(2*total_weight))
        self.assertAlmostEqual(float(row["fraction_weight_kin_ge_all_pairs"]),
                               (math.log1p(100)+math.log1p(200))/total_weight)
        self.assertTrue(result["no_pvalues"] and result["no_winner_selected"])
        self.assertEqual(result["evidence_gaps"]["local_ibd"], "NOT_EVALUATED")
        self.assertEqual(result["evidence_gaps"]["quality_and_batch"], "NOT_EVALUATED")
        self.assertFalse(result["contains_individual_identifiers"])
        self.assertEqual(result["pcrelate_audit"]["n_retained_union_edges"], 10)

    def test_zero_edge_configurations_and_thresholds_never_disappear(self):
        result = self.execute(edge_thresholds_bp=(1000, 0, 350))
        self.assertEqual(result["n_diagnostic_rows"], 2*3*2)
        rows = self.read_rows()
        for row in rows:
            if row["config_id"] == self.zero or int(row["min_edge_bp"]) == 1000:
                self.assertEqual(row["n_pairs"], "0")
                self.assertEqual(row["n_isolated"], "5")
                self.assertEqual(row["fraction_n_kin_ge_all_pairs"], "NA")
                self.assertEqual(row["max_weighted_degree_share"], "NA")
        retained = next(row for row in rows if row["config_id"] == self.config and row["min_edge_bp"] == "350")
        self.assertEqual(retained["n_pairs"], "2")
        self.assertEqual(retained["n_active"], "3")

    def test_74_configuration_ledger_includes_all_empty_settings_sorted(self):
        self.rows = []
        self.ledger = [dict(config_id=f"L{length}_G100_N2", max_gap_bp=100,
                            min_length_bp=length, min_shared_effective=2, n_pairs=0,
                            n_segments=0, total_shared_bp=0, n_shared_variants_total=0)
                       for length in range(74, 0, -1)]
        self.write_inputs()
        result = self.execute(expected_configurations=74)
        ledger = self.read_rows("configuration_ledger.tsv")
        self.assertEqual(len(ledger), 74)
        self.assertEqual(result["n_diagnostic_rows"], 148)
        self.assertEqual([row["config_id"] for row in ledger], sorted(row["config_id"] for row in ledger))
        self.assertEqual(result["n_rows_scanned"], 0)

    def test_single_chromosome_original_pair_major_order_and_hashes(self):
        sample_index = {sample: i for i, sample in enumerate(self.samples.read_text().splitlines())}
        self.rows.sort(key=lambda row: tuple(sorted((sample_index[row["sample_a"]], sample_index[row["sample_b"]]))))
        self.write_inputs()
        source = self.root / "summary.json"
        source.write_text(json.dumps(dict(status="COMPLETE_EXPLORATORY_NOT_VALIDATED", chrom="chr22",
            n_samples=5, n_effective_configurations=2, carrier_allele_mode="source_minor",
            selected_samples_order_sha256=diagnostics.sweep.sha256(self.samples),
            source_rare_contract=dict(contract="minor_v1"))))
        result = self.execute(aggregation_receipt=None, source_receipt=source,
            pair_sha256=diagnostics.sweep.sha256(self.pairs),
            configuration_sha256=diagnostics.sweep.sha256(self.configs))
        self.assertEqual(result["chromosomes"], ["22"])
        self.assertEqual(result["n_rows_scanned"], 5)

    def test_reversed_duplicate_pair_is_rejected(self):
        duplicate = dict(self.rows[0], sample_a="B", sample_b="A")
        self.rows.insert(1, duplicate)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Duplicate/reversed"):
            self.execute()
        self.assertFalse((self.root / "diagnostics").exists())

    def test_unsorted_pair_groups_rejected_not_silently_accumulated(self):
        self.rows[0], self.rows[-1] = self.rows[-1], self.rows[0]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "unsorted"):
            self.execute()

    def test_configuration_group_reappearance_rejected(self):
        order = diagnostics.PairOrder(True)
        order.check("L10", "A", "B", {})
        order.check("L20", "A", "B", {})
        with self.assertRaisesRegex(ValueError, "unsorted"):
            order.check("L10", "A", "C", {})

    def test_unknown_id_and_self_pair_rejected(self):
        for a, b in (("OUTSIDE", "B"), ("A", "A")):
            with self.subTest(a=a, b=b):
                self.rows[0].update(sample_a=a, sample_b=b)
                self.write_inputs()
                with self.assertRaisesRegex(ValueError, "Self-pair or ID"):
                    self.execute()

    def test_duplicate_sample_ids_and_ledger_config_rejected(self):
        self.samples.write_text("D\nC\nA\nB\nB\n")
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "uniqueness"):
            self.execute()
        self.samples.write_text("D\nC\nA\nB\nE\n")
        self.ledger.append(dict(self.ledger[0]))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Duplicate configuration"):
            self.execute()

    def test_kinship_reversed_duplicates_even_when_first_unknown_rejected(self):
        self.kin.write_text("ID1\tID2\tkin\nA\tB\tNA\nB\tA\t0.1\n")
        with self.assertRaisesRegex(ValueError, "Duplicate retained unordered"):
            self.execute()

    def test_all_source_hashes_and_sample_order_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "PC-Relate full-file SHA256"):
            self.execute(expected_pcrelate_sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "source receipt SHA256"):
            self.execute(receipt_sha256="0"*64)
        with self.assertRaisesRegex(ValueError, "sample IDs SHA256"):
            self.execute(sample_ids_sha256="0"*64)
        record = json.loads(self.receipt.read_text())
        record["outputs_sha256"][self.pairs.name] = "0"*64
        self.receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "compressed-byte SHA256"):
            self.execute()
        self.write_inputs()
        self.samples.write_text("C\nD\nA\nB\nE\n")
        with self.assertRaisesRegex(ValueError, "sample-ID file hash"):
            self.execute()

    def test_configuration_count_totals_and_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "ledger is incomplete"):
            self.execute(expected_configurations=74)
        self.ledger[0]["total_shared_bp"] += 1
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "totals disagree"):
            self.execute()
        self.ledger[0]["min_length_bp"] = 11
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "ID/numeric"):
            self.execute()

    def test_nonfinite_and_absent_kinship_remain_distinct(self):
        self.kin.write_text("ID1\tID2\tkin\nA\tB\tInf\nA\tC\tNaN\nA\tD\t-0.1\n")
        self.execute()
        row = next(row for row in self.read_rows() if row["config_id"] == self.config)
        self.assertEqual(row["n_missing_nonfinite"], "2")
        self.assertEqual(row["n_missing_absent"], "2")
        self.assertEqual(row["n_kin_lt_threshold"], "1")

    def test_no_overwrite_and_bad_thresholds_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "baseline zero"):
            self.execute(edge_thresholds_bp=(100,))
        with self.assertRaisesRegex(ValueError, "kinship thresholds"):
            self.execute(thresholds=(float("nan"),))
        self.execute()
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            self.execute()

    def test_cli_contract_without_numeric_dependencies(self):
        command = [sys.executable, str(ROOT / "bin/r02_m14_configuration_diagnostics.py"),
            "--pair-summary", str(self.pairs), "--configuration-summary", str(self.configs),
            "--sample-ids", str(self.samples), "--expected-samples", "5", "--expected-configurations", "2",
            "--pcrelate-file", str(self.kin), "--expected-pcrelate-sha256", diagnostics.sweep.sha256(self.kin),
            "--aggregation-receipt", str(self.receipt), "--edge-thresholds-bp", "0,250,500",
            "--thresholds", "0.0221,0.0442", "--output-dir", str(self.root / "diagnostics")]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], diagnostics.STATUS)
        self.assertEqual(receipt["n_diagnostic_rows"], 12)
        self.assertNotIn("sample_a", result.stdout)

    def test_actual_autosome_aggregator_output_is_consumed_without_resorting(self):
        settings = self.root / "settings.json"
        settings.write_text(json.dumps(dict(expected_samples=5, resolutions=[1.0], n_seeds=1,
            seed=1, min_community_size=3, consensus_resolution=1.0,
            configurations=[dict(length_bp=row["min_length_bp"], gap_bp=row["max_gap_bp"],
                min_shared=row["min_shared_effective"], min_edge_bp=1, min_max_segment_bp=0)
                for row in self.ledger])))
        entries = []
        for chrom in diagnostics.AUTOSOMES:
            summary = self.root / f"source_chr{chrom}.json"
            summary.write_text(json.dumps(dict(chrom=chrom, status="COMPLETE_EXPLORATORY_NOT_VALIDATED",
                n_samples=5, selected_samples_order_sha256=diagnostics.sweep.sha256(self.samples),
                carrier_allele_mode="source_minor", source_rare_contract=dict(contract="minor_v1"))))
            entries.append(dict(chromosome=chrom, pair_summary=str(self.pairs),
                configuration_summary=str(self.configs), summary=str(summary),
                sha256=dict(pair_summary=diagnostics.sweep.sha256(self.pairs),
                    configuration_summary=diagnostics.sweep.sha256(self.configs),
                    summary=diagnostics.sweep.sha256(summary))))
        manifest = self.root / "inputs.json"
        manifest.write_text(json.dumps(dict(schema_version=1, chromosomes=entries)))
        aggregated = self.root / "aggregated"
        aggregation.aggregate(manifest, settings, self.samples, aggregated)
        result = self.execute(pair_summary=aggregated / self.pairs.name,
            configuration_summary=aggregated / self.configs.name,
            aggregation_receipt=aggregated / "aggregation.json")
        self.assertEqual(result["n_rows_scanned"], 5)
        row = next(row for row in self.read_rows() if row["config_id"] == self.config)
        self.assertEqual(int(row["n_pairs"]), 5)
        self.assertEqual(int(row["n_segments"]), 110)
        self.assertEqual(int(row["total_shared_bp"]), 33000)
        self.assertEqual(int(row["max_segment_bp"]), 500)


if __name__ == "__main__":
    unittest.main()
