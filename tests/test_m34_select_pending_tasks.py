#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_adaptive_sweep as sweep
import m34_select_pending_tasks as subject
import m34_train_factorized as trainer


CONTRACT = ROOT / "conf" / "m34_adaptive_sweep_contract.json"


def write_manifest(path: Path) -> Path:
    path.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    return path


def receipt_payload(task: dict, prediction: Path, manifest: Path) -> dict:
    paired = {name: value for name, value in task.items() if name != "arm"}
    return {
        "status": "PASS_TRAINED_VALID_ONLY_FACTORIZED_LAZY",
        "test_opened": False,
        "task": task,
        "valid_prediction_sha256": trainer.sha256_file(prediction),
        "contract_sha256": trainer.sha256_file(CONTRACT),
        "manifest_sha256": trainer.sha256_file(manifest),
        "paired_task_sha256_without_arm": trainer.canonical_sha256(paired),
    }


class PendingTaskSelectionTests(unittest.TestCase):
    def test_exact_complement_is_selected_and_prediction_is_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            contract = sweep.validate_contract(sweep.strict_json(CONTRACT))
            plan = sweep.triage_plan(contract)
            plan_path = base / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            manifest = write_manifest(base / "factorized.manifest.json")
            receipt_paths = []
            for index, task in enumerate(plan["tasks"][:3]):
                directory = base / str(index)
                directory.mkdir()
                prediction = directory / "valid.prediction.npz"
                prediction.write_bytes(f"prediction-{index}".encode())
                receipt = receipt_payload(task, prediction, manifest)
                path = directory / "train.receipt.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")
                receipt_paths.append(path)
            result = subject.select(CONTRACT, plan_path, receipt_paths, manifest)
            self.assertEqual((result["completed_count"], result["pending_count"]), (3, 39))
            self.assertEqual(result["pending_tasks"], plan["tasks"][3:])
            self.assertEqual(len(result["completed"]), 3)

    def test_duplicate_or_modified_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            contract = sweep.validate_contract(sweep.strict_json(CONTRACT))
            plan = sweep.triage_plan(contract)
            plan_path = base / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            manifest = write_manifest(base / "factorized.manifest.json")
            prediction = base / "valid.prediction.npz"
            prediction.write_bytes(b"before")
            receipt = base / "train.receipt.json"
            receipt.write_text(json.dumps(
                receipt_payload(plan["tasks"][0], prediction, manifest)
            ), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                subject.select(CONTRACT, plan_path, [receipt, receipt], manifest)
            prediction.write_bytes(b"after")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                subject.select(CONTRACT, plan_path, [receipt], manifest)

    def test_exact_local_expansion_plan_is_supported(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            contract = sweep.validate_contract(sweep.strict_json(CONTRACT))
            triage = sweep.triage_plan(contract)
            records = []
            for task in triage["tasks"]:
                record = dict(task)
                # A small positive RE effect gives the planner deterministic trends.
                effect = 0.01 if task["arm"] == "RE" else 0.0
                record.update({
                    "boundary_F1_0.1cM": 0.70 + effect,
                    "boundary_F1_0.2cM": 0.75 + effect,
                    "boundary_F1_0.5cM": 0.80 + effect,
                    "macro_ancestry_dose_MAE": 0.10,
                    "NAM_truth_present_MAE": 0.12,
                    "false_transitions_per_cM": 0.02,
                    "haplotype_Brier": 0.15,
                })
                records.append(record)
            metrics_path = base / "metrics.json"
            metrics_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            pairs = sweep.load_metric_pairs(contract, {"records": records})
            plan = sweep.expansion_plan(contract, pairs)
            plan_path = base / "expansion.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            manifest = write_manifest(base / "factorized.manifest.json")

            result = subject.select(CONTRACT, plan_path, [], manifest, metrics_path)

            self.assertEqual(result["source_plan_stage"], "M34_LOCAL_EXPANSION_PLAN")
            self.assertEqual(result["completed_count"], 0)
            self.assertEqual(result["pending_count"], plan["task_count"])

    def test_adaptive_plan_requires_the_metrics_that_created_it(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            contract = sweep.validate_contract(sweep.strict_json(CONTRACT))
            plan_path = base / "plan.json"
            plan_path.write_text(json.dumps({
                "stage": "M34_LOCAL_EXPANSION_PLAN", "tasks": [],
            }), encoding="utf-8")
            manifest = write_manifest(base / "factorized.manifest.json")
            with self.assertRaisesRegex(ValueError, "--metrics is required"):
                subject.select(CONTRACT, plan_path, [], manifest)


if __name__ == "__main__":
    unittest.main()
