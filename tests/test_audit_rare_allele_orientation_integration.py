#!/usr/bin/env python3

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

try:
    import pandas as pd
    from rare_allele_sharing_painter import (
        compute_sharing_windows,
        detect_pairwise_segments_direct,
        parse_genotypes_carrier_sets,
    )
    from audit_rare_allele_orientation import interval_overlap_summary, window_comparison
except ModuleNotFoundError:
    pd = None


@unittest.skipUnless(
    pd is not None and shutil.which("bcftools") and shutil.which("samtools"),
    "integration test requires the production M14 Python stack, bcftools and samtools",
)
class AuditIntegrationTest(unittest.TestCase):
    def test_interval_and_window_comparisons_are_border_aware(self):
        reference = pd.DataFrame({
            "chrom": ["22", "22"],
            "sample_a": ["a", "a"],
            "sample_b": ["b", "b"],
            "start_pos": [100, 300],
            "end_pos": [200, 400],
        })
        current = pd.DataFrame({
            "chrom": ["22"],
            "sample_a": ["a"],
            "sample_b": ["b"],
            "start_pos": [150],
            "end_pos": [350],
        })
        overlap = interval_overlap_summary(reference, current)
        self.assertEqual(overlap["pairwise_interval_overlap_bp"], 102)
        self.assertAlmostEqual(
            overlap["historical_pairwise_bp_fraction_overlapped"], 102 / 202
        )
        self.assertAlmostEqual(
            overlap["current_pairwise_bp_fraction_overlapped"], 102 / 201
        )
        self.assertEqual(overlap["exact_interval_record_jaccard_vs_historical"], 0.0)

        historical_windows = pd.DataFrame({
            "chrom": ["22"] * 3,
            "start_pos": [1, 101, 201],
            "end_pos": [100, 200, 300],
            "n_sharing_pairs": [1, 2, 3],
        })
        current_windows = historical_windows.copy()
        current_windows["n_sharing_pairs"] = [1, 4, 2]
        comparison = window_comparison(historical_windows, current_windows)
        self.assertEqual(comparison["windows_with_changed_pair_count"], 2)
        self.assertEqual(comparison["total_window_pair_count_historical"], 6)
        self.assertEqual(comparison["total_window_pair_count_current"], 7)
        self.assertIsNotNone(comparison["window_pair_count_spearman_vs_historical"])

    def test_end_to_end_reproduces_historical_and_detects_alt_major(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / "ref.fa"
            fasta.write_text(">chr22\n" + "A" * 2000 + "\n", encoding="ascii")
            subprocess.run(["samtools", "faidx", str(fasta)], check=True)

            samples = ["s1", "s2", "s3", "s4"]
            vcf = root / "chr22.vcf"
            vcf.write_text(
                "##fileformat=VCFv4.2\n"
                "##contig=<ID=chr22,length=2000>\n"
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(samples)
                + "\n"
                + "chr22\t100\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/1\t0/0\t0/0\n"
                + "chr22\t250\t.\tA\tG\t.\tPASS\t.\tGT\t1/1\t1/1\t0/1\t1/1\n"
                + "chr22\t400\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1\t0/0\t0/0\n"
                + "chr22\t550\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/1\t./.\t./.\n",
                encoding="ascii",
            )

            sample_file = root / "samples.txt"
            sample_file.write_text("\n".join(samples) + "\n", encoding="utf-8")
            _, historical, total, lo, hi = parse_genotypes_carrier_sets(vcf, "22", samples)
            self.assertEqual(total, 4)

            params = {
                "window_size_bp": 1000,
                "step_size_bp": 1000,
                "min_shared_variants": 2,
                "min_jaccard": 0.0,
                "max_gap_bp": 500,
                "min_segment_bp": 100,
            }
            canonical_segments = detect_pairwise_segments_direct(
                "22", historical, samples,
                params["max_gap_bp"], params["min_segment_bp"],
                params["min_shared_variants"], n_jobs=1,
            )
            canonical_windows = compute_sharing_windows(
                "22", historical, samples,
                params["window_size_bp"], params["step_size_bp"],
                params["min_shared_variants"], params["min_jaccard"],
            )
            segments_path = root / "canonical.segments.tsv.gz"
            windows_path = root / "canonical.windows.tsv.gz"
            canonical_segments.to_csv(segments_path, sep="\t", index=False, compression="gzip")
            canonical_windows.to_csv(windows_path, sep="\t", index=False, compression="gzip")

            summary_path = root / "canonical.summary.json"
            summary_path.write_text(
                json.dumps({
                    "chrom": "22",
                    "selected_samples": samples,
                    "chrom_extent": [lo, hi],
                    "parameters_used": params,
                }),
                encoding="utf-8",
            )
            split = root / "split.tsv"
            pd.DataFrame({"sample_id": samples, "split": ["TRAIN"] * 4}).to_csv(
                split, sep="\t", index=False
            )

            outdir = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(BIN / "audit_rare_allele_orientation.py"),
                    "--vcf", str(vcf),
                    "--reference-fasta", str(fasta),
                    "--canonical-summary", str(summary_path),
                    "--canonical-windows", str(windows_path),
                    "--canonical-segments", str(segments_path),
                    "--split-manifest", str(split),
                    "--chrom", "22",
                    "--n-jobs", "1",
                    "--outdir", str(outdir),
                ],
                check=True,
                cwd=REPO,
            )

            report = json.loads((outdir / "chr22.audit_report.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertGreater(report["resource_usage"]["analysis_wall_seconds"], 0)
            self.assertGreater(report["resource_usage"]["self_max_rss_kib"], 0)
            self.assertTrue(report["historical_reproduction"]["segments_equal"])
            self.assertTrue(report["historical_reproduction"]["windows_equal"])
            self.assertEqual(
                report["mode_comparisons"]["historical_alt"][
                    "historical_pairwise_bp_fraction_overlapped"
                ],
                1.0,
            )
            self.assertEqual(report["orientation_summary"]["counts"]["alt_major_sites"], 1)
            self.assertEqual(report["orientation_summary"]["counts"]["tie_sites"], 1)
            self.assertLess(
                report["mode_comparisons"]["minor_allele"]["n_variants_with_at_least_two_carriers"],
                report["mode_comparisons"]["historical_alt"]["n_variants_with_at_least_two_carriers"],
            )


if __name__ == "__main__":
    unittest.main()
