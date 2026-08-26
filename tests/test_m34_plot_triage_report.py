#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_plot_triage_report as subject  # noqa: E402


CONTRACT = ROOT / "conf" / "m34_adaptive_sweep_contract.json"


def build_rows(contract: dict, broken_guardrail: bool = False) -> list[dict[str, str]]:
    metrics = [contract["metrics"]["primary"], *contract["metrics"]["sensitivities"],
               *contract["metrics"]["guardrails"]]
    rows = []
    for family, config_id, rank in subject.triage_config_order(contract):
        for metric in metrics:
            f0 = 0.75 if metric.startswith("boundary_F1") else 0.10
            rd = f0 - 0.01 if metric.startswith("boundary_F1") else f0 + 0.002
            re = f0 + 0.01 if metric.startswith("boundary_F1") else f0 - 0.002
            if broken_guardrail and family == "local_linear" and rank == 0 and \
                    metric == "haplotype_Brier":
                re = rd + 0.003
            rows.append({
                "family": family,
                "config_id": config_id,
                "seed": "1103",
                "root": "R0",
                "radius_cM": "0.2",
                "sweep_stage": "triage",
                "maximum_updates": "300",
                "metric": metric,
                "F0": str(f0),
                "RD": str(rd),
                "RE": str(re),
                "RE_minus_RD": str(re - rd),
                "RE_minus_F0": str(re - f0),
            })
    return rows


def write_comparison(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_aggregate_receipt(path: Path, comparison: Path) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "stage": "M34_AGGREGATE_ADAPTIVE_STAGE_METRICS",
        "source_plan_stage": "M34_TRIAGE_PLAN",
        "status": "PASS_EXACT_TRIAGE_GRID_F0_RD_RE",
        "claim_level": "exploratory",
        "evaluation_split": "VALID",
        "test_opened": False,
        "pair_count": 21,
        "long_table_row_count": 147,
        "output_sha256": {"table": subject.sha256_file(comparison)},
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


class M34PlotTriageReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = subject.read_contract(CONTRACT)

    def test_summary_has_exact_grid_primary_effects_and_guardrails(self):
        summaries = subject.summarize(build_rows(self.contract), self.contract)
        self.assertEqual(len(summaries), 21)
        first = summaries[0]
        self.assertEqual((first["family"], first["config_id"]),
                         ("local_linear", "linear_r0"))
        self.assertAlmostEqual(first["RE_minus_RD_boundary_F1_0.2cM"], 0.02)
        self.assertAlmostEqual(first["RE_minus_F0_boundary_F1_0.2cM"], 0.01)
        self.assertTrue(first["primary_delta_pass"])
        self.assertTrue(first["beats_F0"])
        self.assertTrue(first["all_guardrails_pass"])
        self.assertEqual(first["failed_guardrails"], "")
        table = subject.summary_tsv(summaries, self.contract)
        self.assertEqual(len(table.splitlines()), 22)
        self.assertIn("threshold_haplotype_Brier", table.splitlines()[0])

    def test_guardrail_worsening_is_flagged_by_declared_threshold(self):
        summaries = subject.summarize(
            build_rows(self.contract, broken_guardrail=True), self.contract,
        )
        first = summaries[0]
        self.assertFalse(first["pass_haplotype_Brier"])
        self.assertFalse(first["all_guardrails_pass"])
        self.assertEqual(first["failed_guardrails"], "haplotype_Brier")

    def test_missing_duplicate_and_inconsistent_arithmetic_are_rejected(self):
        rows = build_rows(self.contract)
        with self.assertRaisesRegex(subject.ReportError, "incomplete"):
            subject.summarize(rows[:-7], self.contract)
        with self.assertRaisesRegex(subject.ReportError, "duplicate"):
            subject.summarize(rows + [dict(rows[0])], self.contract)
        altered = [dict(row) for row in rows]
        altered[0]["RE_minus_RD"] = "0.5"
        with self.assertRaisesRegex(subject.ReportError, "arithmetic"):
            subject.summarize(altered, self.contract)

    def test_aggregate_receipt_must_bind_table_and_close_test(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            comparison = write_comparison(root / "comparison.tsv",
                                          build_rows(self.contract))
            receipt_path = write_aggregate_receipt(root / "aggregate.json", comparison)
            receipt = subject.validate_aggregate_receipt(receipt_path, comparison)
            self.assertFalse(receipt["test_opened"])
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            payload["output_sha256"]["table"] = "0" * 64
            receipt_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(subject.ReportError, "not bound"):
                subject.validate_aggregate_receipt(receipt_path, comparison)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None,
                         "matplotlib is supplied by the analysis image")
    def test_rendered_outputs_and_receipt_are_complete(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            comparison = write_comparison(
                root / "comparison.tsv", build_rows(self.contract),
            )
            aggregate = write_aggregate_receipt(
                root / "aggregate.receipt.json", comparison,
            )
            summary = root / "summary.tsv"
            png = root / "figure.png"
            pdf = root / "figure.pdf"
            receipt_path = root / "receipt.json"
            receipt = subject.write_artifacts(
                comparison, CONTRACT, aggregate, summary, png, pdf, receipt_path,
            )
            self.assertEqual(receipt["configuration_count"], 21)
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            stored = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["output_sha256"]["figure_png"],
                             subject.sha256_file(png))
            with self.assertRaisesRegex(subject.ReportError, "overwrite"):
                subject.write_artifacts(
                    comparison, CONTRACT, aggregate, summary, png, pdf, receipt_path,
                )


if __name__ == "__main__":
    unittest.main()
