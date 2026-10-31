#!/usr/bin/env python3
"""Focused technical tests for exact LAI training-plan case universes.

These synthetic identities test input validation only.  They do not execute
FLARE, train a network, select a model, or provide biological evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import r02_lai_training_inputs_plan as plan


def make_cases(chromosomes):
    rows = []
    for chromosome in chromosomes:
        for fold, role in (("FOLD_A", "FIT"), ("FOLD_B", "SELECT")):
            for replicate in range(1, 4):
                case_id = (f"R02L2pilot.PULSE_T10_ANCHOR.chr{chromosome}."
                           f"{fold}.r{replicate}")
                rows.append({
                    "task_id": case_id + ".FOLD_A_TO_FOLD_B",
                    "case_id": case_id,
                    "chromosome": str(chromosome),
                    "direction": plan.DIRECTION,
                    "role": role,
                })
    return rows


def expected_manifest(cases):
    chromosomes = sorted({row["chromosome"] for row in cases}, key=int)
    return {
        "schema": "r02_lai_expected_case_universe_v1",
        "status": "EXACT_CASE_UNIVERSE_RECORDED_PREPARATION_ONLY",
        "operation": "PREPARE_ONLY_NO_TRAINING",
        "direction": plan.DIRECTION,
        "case_count": len(cases),
        "role_counts": {
            "FIT": sum(row["role"] == "FIT" for row in cases),
            "SELECT": sum(row["role"] == "SELECT" for row in cases),
        },
        "chromosomes": chromosomes,
        "training_authorized": False,
        "selection_authorized": False,
        "evaluate_opened": False,
        "cases": cases,
    }


class ExpectedCaseUniverseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def resolve(self, inventory_cases, manifest_cases=None, mutate_manifest=None,
                supplied_sha=None):
        inventory = {"tasks": inventory_cases}
        if manifest_cases is None:
            args = SimpleNamespace(expected_cases=None, expected_cases_sha256=None)
            return plan.resolve_expected_cases(args, inventory)
        value = expected_manifest(manifest_cases)
        if mutate_manifest:
            mutate_manifest(value)
        path = self.root / "expected-cases.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        args = SimpleNamespace(expected_cases=str(path),
                               expected_cases_sha256=supplied_sha or digest)
        return plan.resolve_expected_cases(args, inventory)

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return path

    def test_historical_route_remains_exactly_six_fit_and_six_select(self):
        cases = make_cases((1, 22))
        resolved, source = self.resolve(cases)
        self.assertEqual(len(resolved), 12)
        self.assertEqual(sum(row["role"] == "FIT" for row in resolved.values()), 6)
        self.assertIsNone(source)

    def test_sha_pinned_120_and_combined_132_case_universes(self):
        for chromosomes, expected_count in ((range(2, 22), 120), (range(1, 23), 132)):
            with self.subTest(expected_count=expected_count):
                cases = make_cases(chromosomes)
                resolved, source = self.resolve(cases, cases)
                self.assertEqual(len(resolved), expected_count)
                self.assertEqual(sum(row["role"] == "FIT" for row in resolved.values()),
                                 expected_count // 2)
                self.assertEqual(source["sha256"], plan.sha(source["path"]))

    def test_separate_expected_sha_is_mandatory_and_verified(self):
        cases = make_cases(range(2, 22))
        with self.assertRaisesRegex(ValueError, "recorded SHA-256 differs"):
            self.resolve(cases, cases, supplied_sha="0" * 64)
        args = SimpleNamespace(expected_cases=str(self.root / "absent.json"),
                               expected_cases_sha256=None)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            plan.resolve_expected_cases(args, {"tasks": cases})

    def test_empty_explicit_case_universe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "universe is empty"):
            self.resolve([], [])

    def test_missing_extra_and_contradictory_region_or_role_are_rejected(self):
        cases = make_cases(range(2, 22))
        variants = []
        variants.append((cases[:-1], cases, "missing"))
        extra = copy.deepcopy(cases)
        extra.append(make_cases((1,))[0])
        variants.append((extra, cases, "extra"))
        wrong_role = copy.deepcopy(cases)
        wrong_role[0]["role"] = "SELECT"
        variants.append((wrong_role, cases, "role"))
        wrong_chromosome = copy.deepcopy(cases)
        wrong_chromosome[0]["chromosome"] = "3"
        variants.append((wrong_chromosome, cases, "chromosome"))
        for inventory_cases, manifest_cases, name in variants:
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "missing, extra, or contradictory"):
                self.resolve(inventory_cases, manifest_cases)

    def test_duplicates_are_rejected_even_when_total_count_is_unchanged(self):
        cases = make_cases(range(2, 22))
        duplicate_task = copy.deepcopy(cases)
        duplicate_task[-1]["task_id"] = duplicate_task[0]["task_id"]
        with self.assertRaisesRegex(ValueError, "duplicate expected task_id"):
            self.resolve(cases, duplicate_task)
        duplicate_case = copy.deepcopy(cases)
        duplicate_case[-1]["case_id"] = duplicate_case[0]["case_id"]
        with self.assertRaisesRegex(ValueError, "duplicate expected case_id"):
            self.resolve(cases, duplicate_case)

    def test_manifest_cannot_open_training_selection_or_evaluate(self):
        cases = make_cases(range(2, 22))
        for field in ("training_authorized", "selection_authorized", "evaluate_opened"):
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "preparation-only specification"):
                self.resolve(cases, cases,
                             mutate_manifest=lambda value, field=field: value.__setitem__(field, True))

    def test_build_plan_accepts_exact_120_case_preparation_universe(self):
        cases = make_cases(range(2, 22))
        inventory_tasks = []
        fit_units = []
        reader_tasks = []
        registry_records = []
        for case in cases:
            alternatives = []
            for initial_gen in ("5", "10", "16"):
                alternatives.append({
                    **plan.PRIMARY_FLARE,
                    "initial_gen": initial_gen,
                    "probability_uri": f"gs://fixture/{case['case_id']}/gen{initial_gen}.npy",
                    "probability_generation": "1",
                    "probability_sha256": initial_gen.zfill(64),
                    "probability_bytes": 1,
                    "receipt_uri": f"gs://fixture/{case['case_id']}/gen{initial_gen}.json",
                    "receipt_generation": "1",
                    "receipt_sha256": (initial_gen + "f").zfill(64),
                })
            inventory_tasks.append({
                **case,
                "people": 128,
                "available": {
                    "flare_projection_alternatives": alternatives,
                    "event_candidate_table": {"path": f"/fixture/{case['case_id']}/events.tsv",
                                              "observed_sha256": "e" * 64},
                    "simulation_predictors": {"alt_dosage.npy": {
                        "path": f"/fixture/material/{case['case_id']}/arrays/alt_dosage.npy"}},
                    "query_overlay": {"path": f"/fixture/overlay/{case['case_id']}/data/query.npz",
                                      "recorded_sha256": "q" * 64},
                    "query_overlay_receipt": {"path": f"/fixture/{case['case_id']}/overlay.json",
                                              "observed_sha256": "o" * 64},
                    "genetic_map": {"path": f"/fixture/chr{case['chromosome']}.map",
                                    "sha256": "m" * 64},
                },
            })
            fit_units.append({
                "unit_id": case["task_id"], "case_id": case["case_id"],
                "chromosome": case["chromosome"], "direction": plan.DIRECTION,
                "target_role": case["role"],
                "simulated_fit_statistics_job": case["task_id"],
            })
            reader_tasks.append({
                **case,
                "flare_identity": plan.PRIMARY_FLARE,
                "inputs": {
                    "flare_probabilities": {
                        "path": f"/fixture/{case['case_id']}/gen10/probabilities.npy",
                        "sha256": "10".zfill(64),
                    },
                    "reference_manifest": {"path": "/fixture/reference.json", "sha256": "r" * 64},
                    "reference_index": {"path": "/fixture/reference-index.json", "sha256": "i" * 64},
                },
            })
            registry_records.append({
                "case_id": case["case_id"], "initial_gen": 10,
                "policy": "ALL",
                "marker_condition": "BASELINE_EXCLUDING_EVALUATED_DNABR_RARE",
                "archive": {"uri": f"gs://fixture/{case['case_id']}.tar", "generation": "1",
                            "sha256": "a" * 64, "bytes": 2048},
                "members": {"manifest": {"path": "manifest.json", "sha256": "b" * 64,
                                         "bytes": 10, "tar_header_offset": 0}},
                "archive_inventory": {"uri": f"gs://fixture/{case['case_id']}.inventory.json",
                                      "generation": "1", "sha256": "c" * 64, "bytes": 1},
            })

        inventory = self.write_json("inventory.json", {
            "schema": "r02_lai_channel_input_inventory_v1", "tasks": inventory_tasks})
        fit_plan = self.write_json("fit.json", {
            "schema": "r02_lai_fit_statistics_plan_v1", "logical_units": fit_units})
        reader = self.write_json("reader.json", {
            "schema": "r02_lai_regional_reader_plan_v1", "tasks": reader_tasks})
        registry = self.write_json("registry.json", {
            "status": "COMPLETE_FITTED_MODELS_RANGE_VERIFIED", "records": registry_records})
        decision = self.write_json("decision.json", {
            "schema": "r02_lai_flare_initialization_decision_cx_v1",
            "status": "PROSPECTIVE_DESIGN_DECISION_REQUIRES_EXACT_EXISTING_INPUT_BINDING"})
        universe = self.write_json("universe.json", expected_manifest(cases))
        output = self.root / "plan.json"
        args = SimpleNamespace(
            inventory=str(inventory), fit_plan=str(fit_plan), reader_plan=str(reader),
            flare_registry=str(registry), flare_decision=str(decision),
            expected_cases=str(universe), expected_cases_sha256=plan.sha(universe),
            catalogue_receipt=None, flare_receipt_staging=None,
            output=str(output), case_manifest_output=None, product_request_output=None)
        result = plan.build_plan(args)
        self.assertEqual(len(result["tasks"]), 120)
        self.assertEqual(result["sources"]["expected_case_universe"]["sha256"], plan.sha(universe))
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["selection_authorized"])
        self.assertFalse(result["evaluate_opened"])

    def test_receipt_rows_require_the_exact_task_set_and_no_duplicates(self):
        cases = make_cases((1, 22))
        expected = plan.index_case_identities(cases, "expected")
        rows = [{"task_id": row["task_id"]} for row in cases]
        self.assertEqual(set(plan.exact_task_rows(rows, expected, "receipt")), set(expected))
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            plan.exact_task_rows(rows[:-1], expected, "receipt")
        with self.assertRaisesRegex(ValueError, "duplicate receipt task_id"):
            plan.exact_task_rows(rows[:-1] + [rows[0]], expected, "receipt")

    def test_historical_registry_still_requires_all_108_records(self):
        cases = make_cases((1, 22))
        expected = plan.index_case_identities(cases, "expected")
        records = []
        for case in cases:
            for initial_gen in (5, 10, 16):
                for policy in ("ALL", "AFR", "EUR"):
                    records.append({
                        "case_id": case["case_id"], "initial_gen": initial_gen,
                        "policy": policy,
                        "marker_condition": "BASELINE_EXCLUDING_EVALUATED_DNABR_RARE",
                    })
        registry = {"status": "COMPLETE_108_FITTED_MODELS_RANGE_VERIFIED",
                    "records": records}
        self.assertEqual(len(plan.validate_flare_registry(registry, expected, False)), 108)
        registry["records"] = records[:-1]
        with self.assertRaisesRegex(ValueError, "must contain 108"):
            plan.validate_flare_registry(registry, expected, False)

    def test_expanded_registry_rejects_duplicate_model_identity(self):
        case = make_cases((2,))[0]
        expected = plan.index_case_identities([case], "expected")
        record = {
            "case_id": case["case_id"], "initial_gen": 10, "policy": "ALL",
            "marker_condition": "BASELINE_EXCLUDING_EVALUATED_DNABR_RARE",
        }
        registry = {"status": "COMPLETE_FITTED_MODELS_RANGE_VERIFIED",
                    "records": [record, dict(record)]}
        with self.assertRaisesRegex(ValueError, "duplicate FLARE fitted-model identity"):
            plan.validate_flare_registry(registry, expected, True)


if __name__ == "__main__":
    unittest.main()
