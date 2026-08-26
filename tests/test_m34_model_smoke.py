#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
sys.path.insert(0, str(BIN))


def load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


SUBJECT = load("m34_model_smoke", "bin/m34_model_smoke.py")
SWEEP = sys.modules["m34_adaptive_sweep"]
CONTRACT_PATH = ROOT / "conf/m34_adaptive_sweep_contract.json"


def contract():
    return SWEEP.validate_contract(SWEEP.strict_json(CONTRACT_PATH))


def triage_tasks_by_family():
    plan = SWEEP.triage_plan(contract())
    tasks = []
    seen = set()
    for task in plan["tasks"]:
        if task["family"] not in seen and task["arm"] == "RE":
            tasks.append(task)
            seen.add(task["family"])
    return tasks


class M34ModelSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def run_receipt(self, tasks):
        return SUBJECT.run_smoke(
            contract(), tasks,
            contract_sha256="a" * 64,
            task_source_sha256="b" * 64,
            channels=7,
            ancestries=3,
            batch_size=4,
            context_length=17,
            haplotypes=2,
        )

    def test_every_family_completes_forward_backward_and_optimizer_step(self):
        receipt = self.run_receipt(triage_tasks_by_family())
        deterministic = receipt["deterministic"]
        self.assertEqual(deterministic["status"], SUBJECT.PASS_STATUS)
        self.assertFalse(deterministic["scientific_result"])
        self.assertEqual(deterministic["task_count"], 7)
        self.assertEqual(
            {row["task"]["family"] for row in deterministic["results"]},
            set(SWEEP.EXPECTED_FAMILIES),
        )
        for result in deterministic["results"]:
            with self.subTest(family=result["task"]["family"]):
                self.assertGreater(result["parameter_count"], 0)
                self.assertGreater(result["baseline_zero_count"], 0)
                self.assertLessEqual(result["zero_head_baseline_max_abs_error"], 5e-7)
                self.assertTrue(result["gradients_finite"])
                self.assertEqual(result["optimizer_steps_executed"], 1)
                self.assertNotEqual(
                    result["parameter_sha256_before_step"],
                    result["parameter_sha256_after_step"],
                )

    def test_deterministic_content_repeats_while_telemetry_is_separate(self):
        tasks = triage_tasks_by_family()[:2]
        first = self.run_receipt(tasks)
        second = self.run_receipt(list(reversed(tasks)))
        self.assertEqual(first["deterministic"], second["deterministic"])
        self.assertEqual(first["deterministic_sha256"], second["deterministic_sha256"])
        self.assertTrue(first["telemetry"]["excluded_from_deterministic_sha256"])
        self.assertNotIn("wall_seconds", first["deterministic"])
        self.assertNotIn("rss", json.dumps(first["deterministic"]).lower())

    def test_paired_arms_share_fixture_and_initialization_seed(self):
        tasks = SUBJECT.tasks_for_config(
            contract(), "residual_cnn_1d", "rescnn_r0", "triage",
            seed=None, rotation=None, arm="both", radius_cM=None)
        receipt = self.run_receipt(tasks)
        left, right = receipt["deterministic"]["results"]
        self.assertEqual({left["task"]["arm"], right["task"]["arm"]}, {"RD", "RE"})
        self.assertEqual(left["synthetic_pairing_seed"], right["synthetic_pairing_seed"])
        self.assertEqual(left["parameter_sha256_before_step"],
                         right["parameter_sha256_before_step"])
        self.assertEqual(left["prediction_sha256_after_step"],
                         right["prediction_sha256_after_step"])

    def test_radius_and_stage_are_part_of_task_identity(self):
        tasks = SUBJECT.tasks_for_config(
            contract(), "tcn", "tcn_r0", "radius_sensitivity",
            seed=None, rotation=None, arm="RE", radius_cM=None)
        self.assertEqual(len(tasks), 4)
        self.assertEqual({row["radius_cM"] for row in tasks}, {0.05, 0.1, 0.2, 0.5})
        self.assertEqual({row["sweep_stage"] for row in tasks}, {"radius_sensitivity"})
        receipt = self.run_receipt(tasks)
        results = receipt["deterministic"]["results"]
        self.assertEqual(len({row["synthetic_pairing_seed"] for row in results}), 4)
        triage = SUBJECT.tasks_for_config(
            contract(), "tcn", "tcn_r0", "triage",
            seed=None, rotation=None, arm="RE", radius_cM=0.2)
        expansion = SUBJECT.tasks_for_config(
            contract(), "tcn", "tcn_r0", "local_expansion",
            seed=None, rotation=None, arm="RE", radius_cM=0.2)
        cross_stage = SUBJECT.normalize_tasks(contract(), triage + expansion)
        self.assertEqual(len(cross_stage), 2)
        self.assertNotEqual(SUBJECT._task_seed(cross_stage[0]), SUBJECT._task_seed(cross_stage[1]))

    def test_undeclared_duplicate_and_drifted_tasks_fail_closed(self):
        base = SWEEP.triage_plan(contract())["tasks"][0]
        cases = []
        unknown = dict(base, config_id="invented")
        cases.append(([unknown], "undeclared"))
        cases.append(([base, dict(base)], "duplicate"))
        cases.append(([dict(base, learning_rate=0.02)], "learning rate"))
        cases.append(([dict(base, maximum_updates=301)], "declared stage"))
        cases.append(([dict(base, seed=999)], "declared stage"))
        for tasks, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(SUBJECT.SmokeError, message):
                SUBJECT.normalize_tasks(contract(), tasks)
        with self.assertRaisesRegex(SUBJECT.SmokeError, "seed is not declared"):
            SUBJECT.tasks_for_config(
                contract(), "local_linear", "linear_r0", "triage",
                seed=999, rotation=None, arm="both", radius_cM=None)

    def test_cli_supports_single_config_and_task_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            single_output = base / "single.json"
            command = [
                sys.executable, str(ROOT / "bin/m34_model_smoke.py"),
                "--contract", str(CONTRACT_PATH),
                "--family", "local_linear", "--config-id", "linear_r0",
                "--arm", "both", "--channels", "7", "--ancestries", "3",
                "--batch-size", "3", "--context-length", "11",
                "--output", str(single_output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(single_output.read_text())["deterministic"]["task_count"], 2)

            manifest = SWEEP.triage_plan(contract())
            manifest["tasks"] = manifest["tasks"][:1]
            manifest["task_count"] = 1
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_output = base / "manifest-receipt.json"
            command = [
                sys.executable, str(ROOT / "bin/m34_model_smoke.py"),
                "--contract", str(CONTRACT_PATH), "--manifest", str(manifest_path),
                "--channels", "7", "--ancestries", "3",
                "--batch-size", "3", "--context-length", "11",
                "--output", str(manifest_output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(manifest_output.read_text())["deterministic"]["task_count"], 1)

            manifest["task_count"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest count", result.stderr)


if __name__ == "__main__":
    unittest.main()
