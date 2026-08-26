#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import m34_adaptive_sweep as sweep  # noqa: E402
import m34_aggregate_triage as subject  # noqa: E402


CONTRACT = ROOT / "conf" / "m34_adaptive_sweep_contract.json"
TRUTH_SHA256 = hashlib.sha256(b"VALID truth").hexdigest()


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def score_payload(prediction_sha256: str, arm: str = "F0") -> dict:
    offset = {"F0": 0.0, "RD": -0.01, "RE": 0.01}[arm]
    mae_offset = {"F0": 0.0, "RD": 0.002, "RE": -0.002}[arm]
    return {
        "schema_version": "1.0.0",
        "stage": "M34_EXPLORATORY_SCORING",
        "status": "PASS_SCORED",
        "claim_level": "exploratory",
        "sample_count": 8,
        "haplotype_count": 2,
        "marker_count": 17,
        "ancestry_names": ["AFR", "EUR", "NAM"],
        "cm_span": 4.5,
        "boundary": {
            "0.1": {"f1": 0.70 + offset, "false_transitions_per_cM": 0.030},
            "0.2": {"f1": 0.75 + offset,
                    "false_transitions_per_cM": 0.020 + mae_offset},
            "0.5": {"f1": 0.80 + offset, "false_transitions_per_cM": 0.010},
        },
        "macro_ancestry_dose_MAE": 0.10 + mae_offset,
        "per_ancestry_MAE": {
            "AFR": 0.08 + mae_offset,
            "EUR": 0.09 + mae_offset,
            "NAM": 0.13 + mae_offset,
        },
        "NAM_truth_present_MAE": 0.12 + mae_offset,
        "haplotype_Brier": 0.15 + mae_offset,
        "input_sha256": {
            "prediction": prediction_sha256,
            "truth": TRUTH_SHA256,
        },
        "truth_opened_only_by_scorer": True,
    }


class Fixture:
    def __init__(self, root: Path, plan_payload: dict | None = None,
                 plan_source_metrics: dict | None = None) -> None:
        self.root = root
        self.contract = root / "contract.json"
        self.contract.write_bytes(CONTRACT.read_bytes())
        contract = sweep.validate_contract(sweep.strict_json(self.contract))
        self.plan_payload = plan_payload or sweep.triage_plan(contract)
        self.plan = write_json(root / "triage.plan.json", self.plan_payload)
        self.plan_source_metrics = (
            write_json(root / "plan-source.records.json", plan_source_metrics)
            if plan_source_metrics is not None else None
        )
        self.manifest = write_json(root / "factorized.manifest.json", {
            "schema_version": "1.0.0",
            "ancestry_names": ["AFR", "EUR", "NAM"],
            "haplotypes": 2,
            "rotation": "R0",
            "splits": {
                "FIT": [{name: f"FIT/{name}.npz" for name in
                         ("selected_variant", "target", "reference", "f0",
                          "marker_cm", "truth")}],
                "VALID": [{name: f"VALID/{name}.npz" for name in
                           ("selected_variant", "target", "reference", "f0",
                            "marker_cm", "truth")}],
            },
        })
        self.baseline = write_json(
            root / "FLARE.F0.F0.metrics.json",
            score_payload(hashlib.sha256(b"F0 prediction").hexdigest()),
        )
        manifest_hash = subject.sha256_file(self.manifest)
        contract_hash = subject.sha256_file(self.contract)
        self.metrics: list[Path] = []
        self.receipts: list[Path] = []
        self.transformer_batching: list[Path] = []
        for index, task in enumerate(self.plan_payload["tasks"]):
            identity = subject.task_id(task)
            prediction_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            metric_name = (
                f"{task['family']}.{task['config_id']}.{task['arm']}.metrics.json"
            )
            self.metrics.append(write_json(
                root / "metrics" / f"task-{index:02d}" / metric_name,
                score_payload(prediction_hash, task["arm"]),
            ))
            paired_task = {name: value for name, value in task.items() if name != "arm"}
            receipt_path = write_json(
                root / "receipts" / f"receipt-{index:02d}.json",
                {
                    "schema_version": "1.0.0",
                    "stage": "M34_EXPLORATORY_TRAIN_FACTORIZED_LAZY",
                    "status": "PASS_TRAINED_VALID_ONLY_FACTORIZED_LAZY",
                    "claim_level": "exploratory",
                    "task": task,
                    "paired_task_sha256_without_arm":
                        subject.canonical_sha256(paired_task),
                    "fit_factor_count": 1,
                    "valid_factor_count": 1,
                    "fit_sample_count": 24,
                    "valid_sample_count": 8,
                    "updates_executed": task["maximum_updates"],
                    "selected_update": task["maximum_updates"] - 50,
                    "selected_valid_loss": 0.25,
                    "contract_sha256": contract_hash,
                    "manifest_sha256": manifest_hash,
                    "task_sha256": hashlib.sha256(
                        json.dumps(task, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "valid_prediction_sha256": prediction_hash,
                    "rd_re_pair_policy":
                        "same_factors_axes_masks_geometry_F0_seed_and_task_except_arm",
                    "test_opened": False,
                },
            )
            self.receipts.append(receipt_path)
            if task["family"] == "transformer_small":
                cap = {"transformer_r0": 256,
                       "transformer_r2": 170,
                       "transformer_r4": 64}[task["config_id"]]
                self.transformer_batching.append(write_json(
                    root / "batching" / f"batching-{index:02d}.json",
                    {
                        "schema_version": "1.0.0",
                        "stage": "M34_TRANSFORMER_PHYSICAL_BATCHING",
                        "status": "PASS_BOUNDED_ATTENTION_BATCHING",
                        "family": task["family"],
                        "config_id": task["config_id"],
                        "arm": task["arm"],
                        "policy": "test_budgeted_policy",
                        "declared_maximum_rows_per_logical_microbatch": 2048,
                        "effective_maximum_rows_per_physical_microbatch": cap,
                        "maximum_tokens_per_microbatch": 262144,
                        "logical_optimizer_updates": task["maximum_updates"],
                        "maximum_updates": task["maximum_updates"],
                        "optimizer_step_per_logical_shard_unchanged": True,
                        "task_sha256": hashlib.sha256(
                            json.dumps(task, sort_keys=True).encode("utf-8")
                        ).hexdigest(),
                        "paired_task_sha256_without_arm":
                            subject.canonical_sha256(paired_task),
                        "train_receipt_sha256": subject.sha256_file(receipt_path),
                        "test_opened": False,
                    },
                ))

    def aggregate(self):
        return subject.aggregate(
            self.contract, self.plan, self.manifest, self.baseline,
            self.metrics, self.receipts, self.transformer_batching,
            self.plan_source_metrics,
        )


def record_for_task(task: dict, primary: float) -> dict:
    guardrail = 0.02
    return {
        "family": task["family"], "config_id": task["config_id"],
        "seed": task["seed"], "rotation": task["rotation"],
        "arm": task["arm"], "radius_cM": task["radius_cM"],
        "sweep_stage": task["sweep_stage"],
        "maximum_updates": task["maximum_updates"],
        "boundary_F1_0.1cM": primary - 0.03,
        "boundary_F1_0.2cM": primary,
        "boundary_F1_0.5cM": primary + 0.03,
        "macro_ancestry_dose_MAE": guardrail,
        "NAM_truth_present_MAE": guardrail,
        "false_transitions_per_cM": guardrail,
        "haplotype_Brier": guardrail,
    }


def radius_plan_fixture(contract: dict) -> tuple[dict, dict]:
    records = []
    triage = sweep.triage_plan(contract)
    for task in triage["tasks"]:
        primary = 0.70 if task["arm"] == "RD" else 0.701
        records.append(record_for_task(task, primary))
    expansion = sweep.expansion_plan(
        contract, sweep.load_metric_pairs(contract, {"records": records}),
    )
    for task in expansion["tasks"]:
        primary = 0.70 if task["arm"] == "RD" else 0.72
        records.append(record_for_task(task, primary))
    source = {"records": records}
    plan = sweep.radius_sensitivity_plan(
        contract, sweep.load_metric_pairs(contract, source),
    )
    return plan, source


def mutate(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    write_json(path, payload)


class M34AggregateTriageTests(unittest.TestCase):
    def test_cli_accepts_result_roots_without_shell_generated_argument_lists(self):
        parser_source = (ROOT / "bin" / "m34_aggregate_triage.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('--candidate-root', parser_source)
        self.assertIn('glob("metrics/**/*.metrics.json")', parser_source)
        self.assertIn('--train-root', parser_source)
        self.assertIn('glob("models/**/train.receipt.json")', parser_source)
        self.assertIn('--transformer-batching-root', parser_source)

    def test_known_answer_emits_exact_records_and_long_f0_table(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            payload, table, receipt = fixture.aggregate()

        self.assertEqual(payload["record_count"], 42)
        self.assertEqual(payload["pair_count"], 21)
        self.assertEqual(payload["stage_record_count"], 42)
        self.assertEqual(payload["stage_pair_count"], 21)
        self.assertFalse(payload["test_opened"])
        contract = sweep.validate_contract(sweep.strict_json(CONTRACT))
        pairs = sweep.load_metric_pairs(contract, payload)
        self.assertEqual(len(pairs), 21)
        lines = table.splitlines()
        self.assertEqual(len(lines), 1 + 21 * len(sweep.METRIC_KEYS))
        header = lines[0].split("\t")
        row = dict(zip(header, lines[1].split("\t")))
        self.assertEqual(row["metric"], "boundary_F1_0.1cM")
        self.assertAlmostEqual(float(row["RE_minus_RD"]), 0.02)
        self.assertAlmostEqual(float(row["RE_minus_F0"]), 0.01)
        self.assertEqual(receipt["evaluation_split"], "VALID")
        self.assertEqual(payload["source_plan_stage"], "M34_TRIAGE_PLAN")
        self.assertEqual(receipt["factor_splits"], ["FIT", "VALID"])
        self.assertEqual(receipt["long_table_row_count"], 147)

    def test_writes_hashed_outputs_without_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = Fixture(root)
            payload, table, receipt = fixture.aggregate()
            records = root / "out" / "records.json"
            tsv = root / "out" / "comparison.tsv"
            audit = root / "out" / "aggregate.receipt.json"
            final = subject.write_outputs(payload, table, receipt, records, tsv, audit)
            self.assertEqual(final["output_sha256"]["records"],
                             subject.sha256_file(records))
            self.assertEqual(final["output_sha256"]["table"], subject.sha256_file(tsv))
            self.assertEqual(len(final["semantic_sha256"]), 64)
            with self.assertRaisesRegex(subject.AggregateError, "overwrite"):
                subject.write_outputs(payload, table, receipt, records, tsv, audit)

    def test_missing_or_extra_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            fixture.metrics.pop()
            with self.assertRaisesRegex(subject.AggregateError, "metric count"):
                fixture.aggregate()

    def test_plan_drift_is_rejected_before_metrics(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.plan,
                   lambda value: value["tasks"][0].update(maximum_updates=299))
            with self.assertRaisesRegex(subject.AggregateError, "frozen adaptive contract"):
                fixture.aggregate()

    def test_test_split_in_manifest_or_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.manifest,
                   lambda value: value["splits"].update(TEST=value["splits"]["VALID"]))
            with self.assertRaisesRegex(subject.AggregateError, "FIT and VALID only"):
                fixture.aggregate()
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.receipts[0], lambda value: value.update(test_opened=True))
            with self.assertRaisesRegex(subject.AggregateError, "close TEST"):
                fixture.aggregate()

    def test_prediction_binding_and_truth_geometry_are_enforced(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.metrics[0], lambda value: value["input_sha256"].update(
                prediction=hashlib.sha256(b"different prediction").hexdigest()
            ))
            with self.assertRaisesRegex(subject.AggregateError, "not bound"):
                fixture.aggregate()
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.metrics[0], lambda value: value["input_sha256"].update(
                truth=hashlib.sha256(b"different truth").hexdigest()
            ))
            with self.assertRaisesRegex(subject.AggregateError, "VALID truth differs"):
                fixture.aggregate()

    def test_repeated_radii_require_embedded_task_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            contract = sweep.validate_contract(sweep.strict_json(CONTRACT))
            plan, source = radius_plan_fixture(contract)
            fixture = Fixture(base, plan, source)
            with self.assertRaisesRegex(subject.AggregateError,
                                        "legacy candidate metric filename.*ambiguous"):
                fixture.aggregate()

            for task, path in zip(plan["tasks"], fixture.metrics):
                mutate(path, lambda value, exact=task: value.update(task=exact))
            payload, _table, receipt = fixture.aggregate()
            self.assertEqual(payload["stage_record_count"], plan["task_count"])
            self.assertEqual(receipt["source_plan_stage"],
                             "M34_RADIUS_SENSITIVITY_PLAN")

    def test_embedded_identity_disambiguates_equal_prediction_hashes(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            shared = "a" * 64
            for index in (0, 1):
                task = fixture.plan_payload["tasks"][index]
                mutate(fixture.metrics[index], lambda value, exact=task: (
                    value.update(task=exact),
                    value["input_sha256"].update(prediction=shared),
                ))
                mutate(fixture.receipts[index],
                       lambda value: value.update(valid_prediction_sha256=shared))
            payload, _table, _receipt = fixture.aggregate()
            self.assertEqual(payload["stage_record_count"], 42)

    def test_update_budget_and_pair_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.receipts[0],
                   lambda value: value.update(updates_executed=299))
            with self.assertRaisesRegex(subject.AggregateError, "update budget"):
                fixture.aggregate()
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.receipts[0], lambda value: value.update(
                paired_task_sha256_without_arm="0" * 64
            ))
            with self.assertRaisesRegex(subject.AggregateError, "task hash"):
                fixture.aggregate()

    def test_training_receipt_must_bind_the_adaptive_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.receipts[0], lambda value: value.update(
                contract_sha256="0" * 64
            ))
            with self.assertRaisesRegex(subject.AggregateError, "contract SHA-256"):
                fixture.aggregate()

    def test_nonfinite_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.metrics[0],
                   lambda value: value.update(NAM_truth_present_MAE=None))
            with self.assertRaisesRegex(subject.AggregateError, "must be numeric"):
                fixture.aggregate()

    def test_transformer_batching_must_match_rd_re_and_training(self):
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.transformer_batching[0], lambda value: value.update(
                effective_maximum_rows_per_physical_microbatch=63
            ))
            with self.assertRaisesRegex(subject.AggregateError,
                                        "RD/RE batching policies differ"):
                fixture.aggregate()
        with tempfile.TemporaryDirectory() as raw:
            fixture = Fixture(Path(raw))
            mutate(fixture.transformer_batching[0], lambda value: value.update(
                train_receipt_sha256="0" * 64
            ))
            with self.assertRaisesRegex(subject.AggregateError,
                                        "not bound to training"):
                fixture.aggregate()

    def test_local_expansion_plan_is_rebuilt_from_prior_metrics(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            fixture = Fixture(base)
            triage_records, _table, _receipt = fixture.aggregate()
            write_json(base / "triage.records.json", triage_records)
            contract = sweep.validate_contract(sweep.strict_json(fixture.contract))
            pairs = sweep.load_metric_pairs(contract, triage_records)
            plan = sweep.expansion_plan(contract, pairs)
            plan_path = write_json(base / "expansion.plan.json", plan)
            rebuilt = subject.validate_plan(
                contract, sweep.strict_json(plan_path), triage_records,
            )
            self.assertEqual(len(rebuilt), plan["task_count"])
            self.assertEqual({task["sweep_stage"] for task in rebuilt},
                             {"local_expansion"})

            with self.assertRaisesRegex(subject.AggregateError, "source metrics"):
                subject.validate_plan(contract, sweep.strict_json(plan_path))


if __name__ == "__main__":
    unittest.main()
