"""Synthetic-only compatibility tests for the optional segment-only M14 scan."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
try:
    import pandas as pd
    import rare_allele_sharing_painter as painter
    import rare_segment_sensitivity as sensitivity
except ModuleNotFoundError:
    painter = None


@unittest.skipIf(painter is None, "requires the M14 scientific Python stack")
class SkipWindowsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = ["a", "b", "c"]
        self.contract = dict(contract="minor_v1", cohort_sha256="a" * 64,
                             cohort_n_samples=3)
        self.variants = [(10000 + 40000 * i, frozenset({0, 1})) for i in range(31)]

    def arguments(self, name, extra=()):
        folder = self.root / name
        argv = ["painter", "--mode", "scan", "--input", str(self.root / "fixture.vcf"),
                "--chr", "22", "--carrier-allele-mode", "source_minor",
                "--expected-samples", "3", "--min-shared-variants", "10",
                "--max-gap-bp", "50000", "--min-segment-bp", "1000000",
                "--out-sharing-windows", str(folder / "windows.tsv.gz"),
                "--out-pairwise-segments", str(folder / "segments.tsv.gz"),
                "--out-summary-json", str(folder / "summary.json"),
                "--output-dir", str(folder), "--skip-plots", "true", *extra]
        with patch.object(sys, "argv", argv):
            return painter.parse_args()

    def scan(self, name, extra=(), variants=None):
        args = self.arguments(name, extra)
        sites = self.variants if variants is None else variants
        parsed = ("22", sites, len(sites), sites[0][0] if sites else None,
                  sites[-1][0] if sites else None, {})
        with patch.object(painter, "validate_input_schema"), \
             patch.object(painter, "read_source_minor_contract", return_value=self.contract), \
             patch.object(painter, "_read_header_samples", return_value=self.samples), \
             patch.object(painter, "parse_genotypes_carrier_sets", return_value=parsed), \
             redirect_stderr(io.StringIO()):
            painter.scan_mode(args)
        return args, json.loads(Path(args.out_summary_json).read_text())

    @staticmethod
    def tsv_bytes(path):
        with gzip.open(path, "rb") as handle:
            return handle.read()

    def test_flag_is_boolean_and_defaults_to_legacy_windows(self):
        self.assertFalse(self.arguments("default").skip_windows)
        self.assertFalse(self.arguments("false", ["--skip-windows", "false"]).skip_windows)
        self.assertTrue(self.arguments("true", ["--skip-windows", "true"]).skip_windows)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.arguments("bad", ["--skip-windows", "maybe"])

    def test_default_and_explicit_false_have_identical_outputs(self):
        default, summary_default = self.scan("default")
        explicit, summary_explicit = self.scan("false", ["--skip-windows", "false"])
        for field in ("out_sharing_windows", "out_pairwise_segments"):
            self.assertEqual(self.tsv_bytes(getattr(default, field)),
                             self.tsv_bytes(getattr(explicit, field)))
        summary_default.pop("analysis_date")
        summary_explicit.pop("analysis_date")
        self.assertEqual(summary_default, summary_explicit)
        self.assertEqual(summary_default["windows_status"], "COMPUTED")
        self.assertGreater(summary_default["n_windows"], 0)

    def test_skip_does_not_call_windows_and_preserves_every_segment(self):
        baseline, reference = self.scan("baseline")
        with patch.object(painter, "compute_sharing_windows",
                          side_effect=AssertionError("windows must not execute")):
            skipped, summary = self.scan("skipped", ["--skip-windows", "true"])
        self.assertEqual(self.tsv_bytes(baseline.out_pairwise_segments),
                         self.tsv_bytes(skipped.out_pairwise_segments))
        self.assertEqual(summary["n_segments"], 1)
        for key in ("n_segments", "n_sharing_pairs", "total_shared_bp", "ordered_samples",
                    "carrier_allele_mode", "source_rare_contract", "orientation_qc"):
            self.assertEqual(summary[key], reference[key])
        self.assertIsNone(summary["n_windows"])
        self.assertEqual(summary["windows_status"], "NOT_COMPUTED")
        self.assertTrue(summary["parameters_used"]["skip_windows"])
        for key in ("window_size_bp", "step_size_bp", "min_jaccard"):
            self.assertIsNone(summary["parameters_used"][key])
        self.assertEqual(summary["segment_jaccard_status"], "LEGACY_PLACEHOLDER_NOT_SIMILARITY")
        self.assertEqual(self.tsv_bytes(skipped.out_sharing_windows).decode(),
                         "\t".join(painter.SHARING_WINDOW_COLUMNS) + "\n")
        with (Path(skipped.output_dir) / "chr22.scan_summary.tsv").open() as handle:
            row = next(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(row["n_windows"], "")
        self.assertEqual(list(row), painter.SCAN_SUMMARY_COLUMNS)

    def test_evaluated_empty_is_distinct_from_not_computed(self):
        _, evaluated = self.scan("empty_evaluated", variants=[])
        _, skipped = self.scan("empty_skipped", ["--skip-windows", "true"], variants=[])
        self.assertEqual(evaluated["n_windows"], 0)
        self.assertEqual(evaluated["windows_status"], "COMPUTED")
        self.assertIsNone(skipped["n_windows"])
        self.assertEqual(skipped["windows_status"], "NOT_COMPUTED")
        self.assertEqual(skipped["n_segments"], 0)

    def test_canonical_segment_only_summary_does_not_claim_window_parameters(self):
        args, _ = self.scan("canonical", ["--skip-windows", "true"])
        self.assertEqual(painter.load_and_validate_canonical_summary(
            args.out_summary_json, "22", args), self.samples)
        args.skip_windows = False
        with self.assertRaisesRegex(SystemExit, "lacks load-bearing parameter"):
            painter.load_and_validate_canonical_summary(args.out_summary_json, "22", args)

    def test_segment_consumers_accept_skipped_summary_and_unchanged_schema(self):
        args, summary = self.scan("consumer", ["--skip-windows", "true"])
        sensitivity.validate_anchor(summary, self.samples, self.contract, "22")
        self.assertEqual(sum(sensitivity.load_anchor(args.out_pairwise_segments).values()), 1)
        with patch.object(sys, "argv", ["painter", "--mode", "aggregate",
                    "--pairwise-segments", args.out_pairwise_segments,
                    "--per-chr-summary", args.out_summary_json,
                    "--output-dir", str(self.root / "aggregate")]):
            aggregate_args = painter.parse_args()
        with patch.object(painter, "write_aggregate_report"), redirect_stderr(io.StringIO()):
            painter.aggregate_mode(aggregate_args)
        table = pd.read_csv(self.root / "aggregate/chromosome_sharing_summary.tsv", sep="\t")
        self.assertTrue(pd.isna(table.iloc[0]["n_windows"]))
        self.assertEqual(int(table.iloc[0]["n_segments"]), 1)

    @unittest.skipUnless(shutil.which("bcftools"), "requires bcftools for real synthetic VCF scan")
    def test_real_vcf_scan_preserves_segments_when_windows_are_skipped(self):
        header = ("##fileformat=VCFv4.2\n##contig=<ID=22,length=2000000>\n"
                  "##dnabr_rare_contract=minor_v1\n"
                  "##dnabr_rare_cohort_sha256=" + "a" * 64 + "\n"
                  "##dnabr_rare_cohort_n_samples=3\n"
                  '##INFO=<ID=RARE_ALLELE,Number=1,Type=Integer,Description="Source allele">\n'
                  '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                  '##FORMAT=<ID=RD,Number=1,Type=Integer,Description="Source minor dose">\n'
                  "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\n")
        rows = []
        for i, (pos, _) in enumerate(self.variants):
            # REF-minor and ALT-minor records share a/b without reorientation.
            allele = i % 2
            gt = "0/0:2\t0|1:1\t1/1:0" if allele == 0 else "1/1:2\t0|1:1\t0/0:0"
            rows.append(f"22\t{pos}\t.\tA\tC\t.\tPASS\tRARE_ALLELE={allele}\tGT:RD\t{gt}\n")
        (self.root / "fixture.vcf").write_text(header + "".join(rows))
        outputs = []
        for name, skip in (("real_default", False), ("real_skipped", True)):
            args = self.arguments(name, ["--skip-windows", str(skip).lower()])
            with redirect_stderr(io.StringIO()):
                painter.scan_mode(args)
            outputs.append(args)
        self.assertEqual(self.tsv_bytes(outputs[0].out_pairwise_segments),
                         self.tsv_bytes(outputs[1].out_pairwise_segments))
        summary = json.loads(Path(outputs[1].out_summary_json).read_text())
        self.assertEqual(summary["n_segments"], 1)
        self.assertEqual(summary["windows_status"], "NOT_COMPUTED")


if __name__ == "__main__":
    unittest.main()
