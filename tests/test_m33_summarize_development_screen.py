from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import m33_summarize_development_screen as subject


ANCESTRIES = ("AFR", "EUR", "ASIA")


def metrics(f1: float, false_rate: float, mae: float) -> dict:
    return {
        "truth_opened_only_by_scorer": True,
        "boundary": {"0.2": {"f1": f1, "false_transitions_per_cM": false_rate}},
        "macro_ancestry_dose_MAE": mae,
        "per_ancestry_truth_present_MAE": {ancestry: mae for ancestry in ANCESTRIES},
    }


class DevelopmentScreenSummaryTests(unittest.TestCase):
    def run_summary(self, candidate: dict, baseline: dict | None = None) -> dict:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metric_dir = root / "metrics"
            metric_dir.mkdir()
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline or metrics(0.70, 0.02, 0.03)))
            for family in ("local_linear", "small_residual_cnn_1d"):
                for radius in (0.05, 0.1, 0.2, 0.5):
                    for arm in ("RD", "RE"):
                        payload = candidate[arm]
                        name = f"R0.{family}.r{radius}.qq01.{arm}.metrics.json"
                        (metric_dir / name).write_text(json.dumps(payload))
            args = argparse.Namespace(
                baseline=baseline_path,
                metrics_dir=[metric_dir],
                output_json=root / "summary.json",
                output_tsv=root / "summary.tsv",
            )
            return subject.summarize(args)

    def test_requires_incremental_rare_signal_and_baseline_improvement(self):
        result = self.run_summary({
            "RD": metrics(0.69, 0.02, 0.03),
            "RE": metrics(0.695, 0.02, 0.03),
        })
        self.assertEqual(result["status"], "STOP_SCREEN_NO_CANDIDATE")
        self.assertEqual(result["promoted_candidate_count"], 0)
        self.assertTrue(result["candidates"][0]["checks"]["RE_F1_gt_RD"])
        self.assertFalse(result["candidates"][0]["checks"]["RE_F1_gt_F0"])

    def test_promotes_only_when_every_guardrail_passes(self):
        result = self.run_summary({
            "RD": metrics(0.69, 0.021, 0.031),
            "RE": metrics(0.71, 0.019, 0.029),
        })
        self.assertEqual(result["status"], "PASS_CANDIDATE_FOR_FULL_DEVELOPMENT")
        self.assertEqual(result["promoted_candidate_count"], 8)
        self.assertTrue(all(row["passes_screen_promotion"] for row in result["candidates"]))

    def test_rejects_metrics_without_truth_barrier_receipt(self):
        invalid = metrics(0.71, 0.019, 0.029)
        invalid["truth_opened_only_by_scorer"] = False
        with self.assertRaisesRegex(ValueError, "truth barrier"):
            self.run_summary({"RD": metrics(0.69, 0.021, 0.031), "RE": invalid})

    def test_rejects_an_unfrozen_family_or_radius_even_with_eight_pairs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metric_dir = root / "metrics"
            metric_dir.mkdir()
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(metrics(0.70, 0.02, 0.03)))
            pairs = list(subject.EXPECTED_PAIRS - {("local_linear", 0.5)})
            pairs.append(("local_linear", 1.0))
            for family, radius in pairs:
                for arm in ("RD", "RE"):
                    name = f"R0.{family}.r{radius}.qq01.{arm}.metrics.json"
                    (metric_dir / name).write_text(json.dumps(metrics(0.71, 0.019, 0.029)))
            args = argparse.Namespace(
                baseline=baseline_path,
                metrics_dir=[metric_dir],
                output_json=root / "summary.json",
                output_tsv=root / "summary.tsv",
            )
            with self.assertRaisesRegex(ValueError, "frozen screen"):
                subject.summarize(args)


if __name__ == "__main__":
    unittest.main()
