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
import m34_adaptive_sweep as adaptive  # noqa: E402


CONTRACT = ROOT / "conf" / "m34_adaptive_sweep_contract.json"


def build_plan(contract: dict, stage: str = "M34_TRIAGE_PLAN") -> dict:
    stage_name = subject.PLAN_STAGE_SPECS[stage]["sweep_stage"]
    stage_contract = contract["stages"][stage_name]
    configuration_ids: dict[str, list[str]] = {}
    anchors: dict[str, list[str]] = {}
    selected: dict[str, list[str]] = {}
    for family, specification in contract["families"].items():
        if stage == "M34_TRIAGE_PLAN":
            configuration_ids[family] = list(specification["triage_ids"])
            continue
        by_rank = {config["complexity_rank"]: config["id"]
                   for config in specification["configs"]}
        anchor = specification["triage_ids"][1]
        anchor_rank = next(config["complexity_rank"]
                           for config in specification["configs"]
                           if config["id"] == anchor)
        neighbor = by_rank[anchor_rank - 1]
        anchors[family] = [anchor]
        selected[family] = [neighbor]
        configuration_ids[family] = [anchor, neighbor]

    tasks = []
    for family, ids in configuration_ids.items():
        by_id = {config["id"]: config
                 for config in contract["families"][family]["configs"]}
        for config_id in ids:
            training = dict(contract["training"])
            training.update(by_id[config_id].get("training_overrides", {}))
            for arm in ("RD", "RE"):
                tasks.append({
                    "family": family,
                    "config_id": config_id,
                    "arm": arm,
                    "seed": stage_contract["seed"],
                    "rotation": stage_contract["rotation"],
                    "radius_cM": stage_contract["radius_cM"],
                    "sweep_stage": stage_name,
                    "maximum_updates": stage_contract["maximum_updates"],
                    "learning_rate": training["learning_rate"],
                    "weight_decay": training["weight_decay"],
                })
    plan = {
        "stage": stage,
        "status": "PLAN_ONLY_NO_EXECUTION",
        "task_count": len(tasks),
        "tasks": tasks,
    }
    if stage == "M34_LOCAL_EXPANSION_PLAN":
        plan.update({
            "anchor_config_ids_by_family": anchors,
            "selected_config_ids_by_family": selected,
            "medium_budget_config_ids_by_family": configuration_ids,
        })
    return plan


def build_rows(contract: dict, plan: dict,
               plan_source_metrics: dict | None = None,
               broken_guardrail: bool = False) -> list[dict[str, str]]:
    metrics = [contract["metrics"]["primary"], *contract["metrics"]["sensitivities"],
               *contract["metrics"]["guardrails"]]
    task_by_pair = {
        (task["family"], task["config_id"], float(task["radius_cM"])): task
        for task in plan["tasks"] if task["arm"] == "RD"
    }
    rows = []
    for family, config_id, rank, radius in subject.validate_plan(
            plan, contract, plan_source_metrics):
        task = task_by_pair[(family, config_id, radius)]
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
                "seed": str(task["seed"]),
                "root": str(task["rotation"]),
                "radius_cM": str(task["radius_cM"]),
                "sweep_stage": str(task["sweep_stage"]),
                "maximum_updates": str(task["maximum_updates"]),
                "metric": metric,
                "F0": str(f0),
                "RD": str(rd),
                "RE": str(re),
                "RE_minus_RD": str(re - rd),
                "RE_minus_F0": str(re - f0),
            })
    return rows


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_comparison(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def aggregate_payload(comparison: Path, contract_path: Path, plan_path: Path,
                      contract: dict, plan: dict,
                      plan_source_path: Path | None = None,
                      plan_source_metrics: dict | None = None) -> dict:
    stage_pairs = len(subject.validate_plan(plan, contract, plan_source_metrics))
    if plan["stage"] == "M34_TRIAGE_PLAN":
        prior_pairs = 0
    elif plan["stage"] == "M34_LOCAL_EXPANSION_PLAN":
        prior_pairs = len(subject.triage_config_order(contract))
    else:
        prior_pairs = int(plan_source_metrics["pair_count"])
    pair_count = prior_pairs + stage_pairs
    metric_count = 1 + len(contract["metrics"]["sensitivities"]) + len(
        contract["metrics"]["guardrails"]
    )
    return {
        "schema_version": "1.0.0",
        "stage": "M34_AGGREGATE_ADAPTIVE_STAGE_METRICS",
        "source_plan_stage": plan["stage"],
        "status": subject.PLAN_STAGE_SPECS[plan["stage"]]["aggregate_status"],
        "claim_level": "exploratory",
        "evaluation_split": "VALID",
        "test_opened": False,
        "record_count": pair_count * 2,
        "pair_count": pair_count,
        "stage_record_count": stage_pairs * 2,
        "stage_pair_count": stage_pairs,
        "long_table_row_count": stage_pairs * metric_count,
        "input_sha256": {
            "adaptive_contract": subject.sha256_file(contract_path),
            "triage_plan": subject.sha256_file(plan_path),
            "plan_source_metrics": (
                subject.sha256_file(plan_source_path)
                if plan_source_path is not None else None
            ),
        },
        "output_sha256": {"table": subject.sha256_file(comparison)},
    }


def metric_records(plan: dict, deltas: dict[tuple[str, str], float]) -> list[dict]:
    records = []
    for task in plan["tasks"]:
        delta = deltas.get((task["family"], task["config_id"]), 0.0)
        record = {
            name: task[name] for name in (
                "family", "config_id", "arm", "seed", "rotation", "radius_cM",
                "sweep_stage", "maximum_updates",
            )
        }
        for metric in adaptive.METRIC_KEYS:
            base = 0.70 if metric in adaptive.F1_KEYS else 0.10
            record[metric] = base + (delta if task["arm"] == "RE" and
                                     metric in adaptive.F1_KEYS else 0.0)
        records.append(record)
    return records


def build_radius_fixture(contract: dict) -> tuple[dict, dict]:
    triage = adaptive.triage_plan(contract)
    desired_anchors = {
        "bilstm_r0", "lainns_r4", "linear_r4", "rescnn_r2", "tcn_r2",
        "transformer_r2", "unet_r2",
    }
    triage_deltas = {
        (task["family"], task["config_id"]): (
            0.002 if task["config_id"] in desired_anchors else 0.0
        )
        for task in triage["tasks"]
    }
    records = metric_records(triage, triage_deltas)
    expansion = adaptive.expansion_plan(
        contract, adaptive.load_metric_pairs(contract, {"records": records}),
    )
    promotion_deltas = {
        ("bilstm", "bilstm_r1"): 0.020,
        ("unet_1d", "unet_r1"): 0.015,
    }
    expansion_records = metric_records(expansion, promotion_deltas)
    records.extend(expansion_records)
    source = {
        "schema_version": "1.0.0",
        "stage": "M34_ADAPTIVE_STAGE_METRICS_PAYLOAD",
        "source_plan_stage": "M34_LOCAL_EXPANSION_PLAN",
        "status": "PASS_EXACT_LOCAL_EXPANSION_GRID",
        "claim_level": "exploratory",
        "evaluation_split": "VALID",
        "test_opened": False,
        "record_count": len(records),
        "pair_count": len(records) // 2,
        "stage_record_count": len(expansion_records),
        "stage_pair_count": len(expansion_records) // 2,
        "records": records,
    }
    radius = adaptive.radius_sensitivity_plan(
        contract, adaptive.load_metric_pairs(contract, source),
    )
    assert [(row["family"], row["config_id"])
            for row in radius["selected_architectures"]] == [
                ("bilstm", "bilstm_r1"), ("unet_1d", "unet_r1"),
            ]
    return source, radius


class M34PlotTriageReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = subject.read_contract(CONTRACT)
        self.triage_plan = build_plan(self.contract)

    def test_triage_summary_keeps_exact_grid_order_and_guardrails(self):
        summaries = subject.summarize(
            build_rows(self.contract, self.triage_plan),
            self.contract, self.triage_plan,
        )
        self.assertEqual(len(summaries), 21)
        first = summaries[0]
        self.assertEqual((first["family"], first["config_id"]),
                         ("local_linear", "linear_r0"))
        self.assertAlmostEqual(first["RE_minus_RD_boundary_F1_0.2cM"], 0.02)
        self.assertAlmostEqual(first["RE_minus_F0_boundary_F1_0.2cM"], 0.01)
        self.assertTrue(first["primary_delta_pass"])
        self.assertTrue(first["beats_F0"])
        self.assertTrue(first["all_guardrails_pass"])
        table = subject.summary_tsv(summaries, self.contract)
        self.assertEqual(len(table.splitlines()), 22)

    def test_local_expansion_uses_only_plan_configurations_in_plan_order(self):
        plan = build_plan(self.contract, "M34_LOCAL_EXPANSION_PLAN")
        summaries = subject.summarize(
            list(reversed(build_rows(self.contract, plan))), self.contract, plan,
        )
        self.assertEqual(len(summaries), 14)
        expected = [(family, config_id)
                    for family, config_id, _rank, _radius in subject.validate_plan(
                        plan, self.contract,
                    )]
        self.assertEqual([(row["family"], row["config_id"]) for row in summaries],
                         expected)
        self.assertEqual(summaries[0]["maximum_updates"], 800)
        self.assertEqual(summaries[0]["sweep_stage"], "local_expansion")

    def test_guardrail_worsening_is_flagged_by_declared_threshold(self):
        summaries = subject.summarize(
            build_rows(self.contract, self.triage_plan, broken_guardrail=True),
            self.contract, self.triage_plan,
        )
        first = summaries[0]
        self.assertFalse(first["pass_haplotype_Brier"])
        self.assertFalse(first["all_guardrails_pass"])
        self.assertEqual(first["failed_guardrails"], "haplotype_Brier")

    def test_missing_duplicate_arithmetic_and_out_of_plan_are_rejected(self):
        rows = build_rows(self.contract, self.triage_plan)
        with self.assertRaisesRegex(subject.ReportError, "incomplete"):
            subject.summarize(rows[:-7], self.contract, self.triage_plan)
        with self.assertRaisesRegex(subject.ReportError, "duplicate"):
            subject.summarize(rows + [dict(rows[0])], self.contract,
                              self.triage_plan)
        altered = [dict(row) for row in rows]
        altered[0]["RE_minus_RD"] = "0.5"
        with self.assertRaisesRegex(subject.ReportError, "arithmetic"):
            subject.summarize(altered, self.contract, self.triage_plan)

        expansion = build_plan(self.contract, "M34_LOCAL_EXPANSION_PLAN")
        extra = build_rows(self.contract, self.triage_plan)[0]
        with self.assertRaisesRegex(subject.ReportError, "absent from the adaptive plan"):
            subject.summarize(
                build_rows(self.contract, expansion) + [extra],
                self.contract, expansion,
            )

    def test_local_expansion_receipt_dimensions_are_strict(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan = build_plan(self.contract, "M34_LOCAL_EXPANSION_PLAN")
            plan_path = write_json(root / "plan.json", plan)
            comparison = write_comparison(
                root / "comparison.tsv", build_rows(self.contract, plan),
            )
            payload = aggregate_payload(
                comparison, CONTRACT, plan_path, self.contract, plan,
            )
            receipt_path = write_json(root / "aggregate.json", payload)
            receipt = subject.validate_aggregate_receipt(
                receipt_path, comparison, CONTRACT, plan_path, plan, self.contract,
            )
            self.assertEqual((receipt["pair_count"], receipt["stage_pair_count"],
                              receipt["long_table_row_count"]), (35, 14, 98))
            mutations = {
                "status": "PASS_EXACT_TRIAGE_GRID_F0_RD_RE",
                "pair_count": 34,
                "stage_pair_count": 13,
                "long_table_row_count": 97,
            }
            for member, wrong_value in mutations.items():
                with self.subTest(member=member):
                    changed = dict(payload)
                    changed[member] = wrong_value
                    write_json(receipt_path, changed)
                    with self.assertRaises(subject.ReportError):
                        subject.validate_aggregate_receipt(
                            receipt_path, comparison, CONTRACT, plan_path,
                            plan, self.contract,
                        )

    def test_radius_plan_and_rows_require_exact_distinct_radii(self):
        source, plan = build_radius_fixture(self.contract)
        order = subject.validate_plan(plan, self.contract, source)
        self.assertEqual(len(order), 6)
        self.assertEqual({row[3] for row in order}, {0.05, 0.1, 0.5})

        duplicated = json.loads(json.dumps(plan))
        duplicated["tasks"][2]["radius_cM"] = 0.05
        with self.assertRaisesRegex(subject.ReportError, "differs"):
            subject.validate_plan(duplicated, self.contract, source)

        missing = json.loads(json.dumps(plan))
        missing["tasks"] = [task for task in missing["tasks"]
                            if task["radius_cM"] != 0.5]
        missing["task_count"] = len(missing["tasks"])
        with self.assertRaisesRegex(subject.ReportError, "differs"):
            subject.validate_plan(missing, self.contract, source)

        stage_rows = build_rows(self.contract, plan, source)
        combined = stage_rows + subject.reused_radius_rows(
            stage_rows, plan, self.contract, source,
        )
        with self.assertRaisesRegex(subject.ReportError, "duplicate"):
            subject.summarize(combined + [dict(combined[0])], self.contract,
                              plan, source)
        absent = [row for row in combined if not (
            row["family"] == "bilstm" and row["config_id"] == "bilstm_r1" and
            float(row["radius_cM"]) == 0.1
        )]
        with self.assertRaisesRegex(subject.ReportError, "incomplete"):
            subject.summarize(absent, self.contract, plan, source)

    def test_radius_receipt_hashes_and_counts_are_strict(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, plan = build_radius_fixture(self.contract)
            source_path = write_json(root / "source.json", source)
            plan_path = write_json(root / "plan.json", plan)
            comparison = write_comparison(
                root / "comparison.tsv",
                build_rows(self.contract, plan, source),
            )
            payload = aggregate_payload(
                comparison, CONTRACT, plan_path, self.contract, plan,
                source_path, source,
            )
            receipt_path = write_json(root / "aggregate.json", payload)
            receipt = subject.validate_aggregate_receipt(
                receipt_path, comparison, CONTRACT, plan_path, plan, self.contract,
                source_path, source,
            )
            self.assertEqual((receipt["pair_count"], receipt["stage_pair_count"],
                              receipt["long_table_row_count"]), (41, 6, 42))
            mutations = {
                "record_count": 80,
                "pair_count": 40,
                "stage_record_count": 10,
                "stage_pair_count": 5,
                "long_table_row_count": 41,
            }
            for member, wrong_value in mutations.items():
                with self.subTest(member=member):
                    changed = json.loads(json.dumps(payload))
                    changed[member] = wrong_value
                    write_json(receipt_path, changed)
                    with self.assertRaises(subject.ReportError):
                        subject.validate_aggregate_receipt(
                            receipt_path, comparison, CONTRACT, plan_path, plan,
                            self.contract, source_path, source,
                        )
            for hash_member in ("plan_source_metrics", "triage_plan"):
                with self.subTest(hash_member=hash_member):
                    changed = json.loads(json.dumps(payload))
                    changed["input_sha256"][hash_member] = "0" * 64
                    write_json(receipt_path, changed)
                    with self.assertRaises(subject.ReportError):
                        subject.validate_aggregate_receipt(
                            receipt_path, comparison, CONTRACT, plan_path, plan,
                            self.contract, source_path, source,
                        )
            changed = json.loads(json.dumps(payload))
            changed["output_sha256"]["table"] = "0" * 64
            write_json(receipt_path, changed)
            with self.assertRaises(subject.ReportError):
                subject.validate_aggregate_receipt(
                    receipt_path, comparison, CONTRACT, plan_path, plan,
                    self.contract, source_path, source,
                )

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None,
                         "matplotlib is supplied by the analysis image")
    def test_local_expansion_outputs_and_receipt_are_complete(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan = build_plan(self.contract, "M34_LOCAL_EXPANSION_PLAN")
            plan_path = write_json(root / "plan.json", plan)
            comparison = write_comparison(
                root / "comparison.tsv", build_rows(self.contract, plan),
            )
            aggregate = write_json(
                root / "aggregate.receipt.json",
                aggregate_payload(comparison, CONTRACT, plan_path,
                                  self.contract, plan),
            )
            summary = root / "summary.tsv"
            png = root / "figure.png"
            pdf = root / "figure.pdf"
            receipt_path = root / "receipt.json"
            receipt = subject.write_artifacts(
                comparison, CONTRACT, plan_path, aggregate, summary, png, pdf,
                receipt_path,
            )
            self.assertEqual(receipt["configuration_count"], 14)
            self.assertEqual(receipt["source_plan_stage"],
                             "M34_LOCAL_EXPANSION_PLAN")
            self.assertEqual(receipt["status"],
                             "PASS_LOCAL_EXPANSION_REPORT_RENDERED")
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            stored = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["output_sha256"]["figure_png"],
                             subject.sha256_file(png))
            with self.assertRaisesRegex(subject.ReportError, "overwrite"):
                subject.write_artifacts(
                    comparison, CONTRACT, plan_path, aggregate, summary, png, pdf,
                    receipt_path,
                )

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None,
                         "matplotlib is supplied by the analysis image")
    def test_radius_outputs_include_reused_radius_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source, plan = build_radius_fixture(self.contract)
            source_path = write_json(root / "source.json", source)
            plan_path = write_json(root / "plan.json", plan)
            comparison = write_comparison(
                root / "comparison.tsv",
                build_rows(self.contract, plan, source),
            )
            aggregate = write_json(
                root / "aggregate.receipt.json",
                aggregate_payload(comparison, CONTRACT, plan_path, self.contract,
                                  plan, source_path, source),
            )
            summary = root / "summary.tsv"
            png = root / "figure.png"
            pdf = root / "figure.pdf"
            receipt_path = root / "receipt.json"
            receipt = subject.write_artifacts(
                comparison, CONTRACT, plan_path, aggregate, summary, png, pdf,
                receipt_path, source_path,
            )
            self.assertEqual((receipt["configuration_count"],
                              receipt["stage_pair_count"],
                              receipt["reused_pair_count"]), (8, 6, 2))
            self.assertEqual(receipt["status"],
                             "PASS_RADIUS_SENSITIVITY_REPORT_RENDERED")
            with summary.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 8)
            by_config = {}
            for row in rows:
                by_config.setdefault(row["config_id"], set()).add(
                    float(row["radius_cM"])
                )
            self.assertEqual(by_config, {
                "bilstm_r1": {0.05, 0.1, 0.2, 0.5},
                "unet_r1": {0.05, 0.1, 0.2, 0.5},
            })
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            self.assertEqual(receipt["input_sha256"]["plan_source_metrics"],
                             subject.sha256_file(source_path))
            with self.assertRaisesRegex(subject.ReportError, "overwrite"):
                subject.write_artifacts(
                    comparison, CONTRACT, plan_path, aggregate, summary, png, pdf,
                    receipt_path, source_path,
                )


if __name__ == "__main__":
    unittest.main()
