"""Small, synthetic-only retained-edge PC-Relate summaries; no clustering."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_m165_sweep_figures import fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin/m165_graph_kinship.py"
sys.path.insert(0, str(ROOT / "bin"))
try:
    spec = importlib.util.spec_from_file_location("m165_graph_kinship_tested", SCRIPT)
    kinship = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kinship)
finally:
    sys.path.pop(0)


class KinshipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m165-kinship-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "kin.tsv"

    def make_source(self, rows, header=("ID1", "ID2", "kin", "k0", "k2")):
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def read(self, rows):
        expected = self.make_source(rows)
        return kinship.stream_kinship(self.source, {"a": 0, "b": 1, "c": 2}, {(0, 1), (0, 2)}, expected)

    def test_reversed_source_orientation_maps_to_one_unordered_edge(self):
        values, audit = self.read([("b", "a", .0221, 0, 0), ("c", "a", -.01, 0, 0),
                                   ("other", "outside", "ignored", 0, 0)])
        self.assertEqual(values, {(0, 1): .0221, (0, 2): -.01})
        self.assertEqual(audit["n_source_rows"], 3)
        self.assertEqual(audit["n_rows_not_target"], 1)
        self.assertEqual(audit["n_retained_union_edges"], 2)
        self.assertEqual(kinship.kinship_state((0, 1), values, .0221), "kin_ge_threshold")
        self.assertEqual(kinship.kinship_state((0, 2), values, .0221), "kin_lt_threshold")

    def test_identical_reversed_and_conflicting_retained_duplicates_fail(self):
        for first, second in ((.1, .1), (.1, .2), ("NA", .1)):
            for endpoints in (("a", "b"), ("b", "a")):
                with self.subTest(first=first, second=second, reversed=endpoints[0] == "b"):
                    with self.assertRaisesRegex(ValueError, "Duplicate retained unordered"):
                        self.read([("a", "b", first, 0, 0), (*endpoints, second, 0, 0)])

    def test_non_target_duplicates_are_not_retained(self):
        values, audit = self.read([("b", "c", .1, 0, 0), ("c", "b", .2, 0, 0)])
        self.assertEqual(values, {})
        self.assertEqual(audit["n_rows_not_target"], 2)

    def test_missing_absent_and_nonfinite_are_not_below_threshold(self):
        for missing in ("NA", "nan", "inf", "-inf", "", "."):
            with self.subTest(missing=missing):
                values, audit = self.read([("a", "b", missing, 0, 0)])
                self.assertIsNone(values[(0, 1)])
                self.assertNotIn((0, 2), values)
                self.assertEqual(audit["n_retained_missing_absent"], 1)
                self.assertEqual(audit["n_retained_missing_nonfinite"], 1)

    def test_full_source_hash_includes_unmatched_tail(self):
        expected = self.make_source([("a", "b", .1, 0, 0), ("outside1", "outside2", .2, 0, 0)])
        with self.source.open("a") as handle:
            handle.write("outside3\toutside4\t0.0\t0\t0\n")
        with self.assertRaisesRegex(ValueError, "full-file SHA256 mismatch"):
            kinship.stream_kinship(self.source, {"a": 0, "b": 1}, {(0, 1)}, expected)

    def test_source_is_opened_once_for_full_stream_and_hash(self):
        expected = self.make_source([("a", "b", .1, 0, 0)])
        original = Path.open
        calls = []
        def counted(path, *args, **kwargs):
            if path == self.source:
                calls.append(args)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", counted):
            kinship.stream_kinship(self.source, {"a": 0, "b": 1}, {(0, 1)}, expected)
        self.assertEqual(calls, [("rb",)])

    def test_bad_schema_values_and_expected_hash_fail_without_ids_in_error(self):
        for header in (("ID1", "ID2", "other", "k0", "k2"), ("ID1", "ID2", "kin", "kin", "k2")):
            expected = self.make_source([], header)
            with self.assertRaisesRegex(ValueError, "unique ID1, ID2 and kin"):
                kinship.stream_kinship(self.source, {}, set(), expected)
        with self.assertRaisesRegex(ValueError, "kinship value") as error:
            self.read([("a", "b", "PRIVATE_INVALID_VALUE", 0, 0)])
        self.assertNotIn("PRIVATE_INVALID_VALUE", str(error.exception))
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            kinship.stream_kinship(self.source, {}, set(), "bad")

    def test_within_between_and_unassigned_are_disjoint_with_correct_denominators(self):
        graph = dict(config_id="synthetic", length_bp=250000, threshold_bp=500000,
                     n_cohort=8, n_active=8, edges=[(0, 1), (0, 3), (3, 4), (4, 5), (6, 7), (2, 6)],
                     labels={1.: [0, 0, 0, 1, 1, 1, -1, -1]})
        values = {(0, 1): .0221, (0, 3): -.01, (3, 4): None, (4, 5): .01, (2, 6): .2}
        graphs, partitions = kinship.summarize({"graphs": [graph]}, values, .0221)
        total, row = graphs[0], partitions[0]
        self.assertEqual((total["n_edges"], total["n_kin_ge_threshold"], total["n_kin_lt_threshold"]), (6, 2, 2))
        self.assertEqual(total["n_missing_kinship"], 2)
        self.assertEqual(total["fraction_kin_ge_all_edges"], 2/6)
        self.assertEqual(total["fraction_kin_ge_observed_edges"], 2/4)
        self.assertEqual(row["within_assigned_n_edges"], 3)
        self.assertEqual(row["between_assigned_n_edges"], 1)
        self.assertEqual(row["with_unassigned_n_edges"], 2)
        self.assertEqual(row["with_unassigned_n_missing_absent"], 1)
        self.assertEqual(row["within_assigned_n_missing_nonfinite"], 1)
        self.assertEqual(row["within_assigned_fraction_kin_ge_all_edges"], 1/3)
        self.assertEqual(row["within_assigned_fraction_kin_ge_observed_edges"], 1/2)

    def test_zero_denominator_is_na_not_zero(self):
        metrics = kinship.count_metrics(kinship.Counter())
        self.assertIsNone(metrics["fraction_kin_ge_all_edges"])
        self.assertIsNone(metrics["fraction_kin_ge_observed_edges"])
        path = self.base / "empty_class.tsv"
        kinship.write_table(path, [metrics])
        self.assertIn("\tNA\tNA\n", path.read_text())

    def test_complete_authenticated_fixture_outputs_six_and_42_rows_without_ids(self):
        root = fixture(self.base / "results")
        ids = [f"PRIVATE_FIXTURE_SAMPLE_{i}" for i in range(5)]
        expected = self.make_source([(ids[0], ids[1], .0221, 0, 0), (ids[0], ids[2], -.1, 0, 0),
                                     (ids[1], ids[2], "NA", 0, 0)])
        output = self.base / "output"
        receipt = kinship.run(root, self.source, expected, output)
        graphs = kinship.saved.table(output / "graph_kinship_summary.tsv")
        partitions = kinship.saved.table(output / "partition_kinship_summary.tsv")
        self.assertEqual((len(graphs), len(partitions)), (6, 42))
        self.assertTrue(receipt["no_reclustering"] and receipt["no_new_kinship"] and receipt["no_pvalues"])
        self.assertEqual(receipt["pcrelate_audit"]["n_retained_union_edges"], 3)
        for row in graphs:
            self.assertEqual((row["n_edges"], row["n_kin_ge_threshold"], row["n_missing_nonfinite"]), ("3", "1", "1"))
        for path in output.iterdir():
            for sample in ids:
                self.assertNotIn(sample, path.read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["outputs_sha256"], receipt["outputs_sha256"])

    def test_input_hash_mismatch_or_changed_graph_produces_no_output(self):
        root = fixture(self.base / "results")
        expected = self.make_source([])
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            kinship.run(root, self.source, "0"*64, self.base / "bad_hash")
        self.assertFalse((self.base / "bad_hash").exists())
        original = kinship.stream_kinship
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            with next(root.glob("*/graph_nodes.tsv")).open("a") as handle:
                handle.write("changed\n")
            return result
        with patch.object(kinship, "stream_kinship", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "source changed"):
                kinship.run(root, self.source, expected, self.base / "changed")
        self.assertFalse((self.base / "changed").exists())

    def test_existing_output_and_invalid_threshold_rejected(self):
        existing = self.base / "existing"; existing.mkdir()
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            kinship.run(self.base / "missing", self.source, "0"*64, existing)
        for threshold in (math.nan, math.inf, -.1, .6):
            with self.assertRaisesRegex(ValueError, "threshold"):
                kinship.run(self.base / "missing", self.source, "0"*64, self.base / "new", threshold)

    def test_cli_only_reads_synthetic_fixture(self):
        root = fixture(self.base / "results")
        expected = self.make_source([])
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--results-dir", str(root),
                                 "--pcrelate-file", str(self.source), "--expected-sha256", expected,
                                 "--threshold", ".0221", "--output-dir", str(self.base / "cli")],
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["n_partitions"], 42)
        self.assertNotIn("PRIVATE_FIXTURE_SAMPLE", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
