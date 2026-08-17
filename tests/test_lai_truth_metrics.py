import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "lai_truth_metrics.py"
SPEC = importlib.util.spec_from_file_location("lai_truth_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LaiTruthMetricsTest(unittest.TestCase):
    def write_rows(self, directory: Path, name: str, rows: list[str]) -> Path:
        path = directory / name
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_summary_counts_transitions_per_haplotype(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            truth = self.write_rows(
                directory,
                "truth.tsv",
                [
                    "22 10 0 1",
                    "22 20 0 1",
                    "22 30 2 1",
                    "22 40 2 0",
                ],
            )
            summary = MODULE.summarize_truth(truth)
            self.assertEqual(summary["n_rows"], 4)
            self.assertEqual(summary["n_haplotypes"], 2)
            self.assertEqual(summary["label_counts"], {"0": 3, "1": 3, "2": 2})
            self.assertEqual(summary["n_transitions"], 2)
            self.assertEqual(summary["transitions_per_haplotype"], [1, 1])

    def test_identical_calls_are_perfect(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            truth = self.write_rows(
                directory,
                "truth.tsv",
                ["22 10 0 1", "22 20 0 1", "22 30 2 1", "22 40 2 0"],
            )
            result = MODULE.compare_calls(truth, truth, tolerance_bp=0)
            self.assertEqual(result["site_accuracy"], 1.0)
            self.assertEqual(result["macro_f1"], 1.0)
            self.assertEqual(result["n_truth_boundaries"], 2)
            self.assertEqual(result["n_matched_boundaries"], 2)
            self.assertEqual(result["boundary_mean_abs_error_bp"], 0.0)

    def test_shifted_boundary_is_matched_within_tolerance(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            truth = self.write_rows(
                directory,
                "truth.tsv",
                ["22 10 0", "22 20 0", "22 30 1", "22 40 1"],
            )
            prediction = self.write_rows(
                directory,
                "prediction.tsv",
                ["22 10 0", "22 20 0", "22 30 0", "22 40 1"],
            )
            result = MODULE.compare_calls(truth, prediction, tolerance_bp=10)
            self.assertEqual(result["n_matched_boundaries"], 1)
            self.assertEqual(result["boundary_mean_abs_error_bp"], 10.0)
            self.assertEqual(result["boundary_precision"], 1.0)
            self.assertEqual(result["boundary_recall"], 1.0)
            self.assertEqual(result["site_accuracy"], 0.75)

    def test_coordinate_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            truth = self.write_rows(directory, "truth.tsv", ["22 10 0", "22 20 1"])
            prediction = self.write_rows(
                directory, "prediction.tsv", ["22 10 0", "22 21 1"]
            )
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                MODULE.compare_calls(truth, prediction, tolerance_bp=0)

    def test_unsorted_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            truth = self.write_rows(directory, "truth.tsv", ["22 20 0", "22 10 1"])
            with self.assertRaisesRegex(ValueError, "not strictly increasing"):
                MODULE.summarize_truth(truth)


if __name__ == "__main__":
    unittest.main()
