"""Focused tests for the one-update LAI FIT resource measurement."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import r02_lai_fit_budget as budget


def fixture_data(people=2):
    common, event, query = 3, 2, 3
    data = {
        "person_id": np.asarray([f"p-{index}" for index in range(people)]),
        "component_id": np.asarray([f"component-{index}" for index in range(people)]),
        "role": np.asarray(["FIT"] * people),
        "common_features": np.zeros((people, common, 12), np.float32),
        "common_cm": np.broadcast_to(np.asarray([0.00, 0.01, 0.02]),
                                      (people, common)).copy(),
        "common_chrom": np.full((people, common), 22, np.int64),
        "common_pos_bp": np.broadcast_to(np.asarray([100, 200, 300]),
                                          (people, common)).copy(),
        "common_valid": np.ones((people, common), bool),
        "common_core": np.ones((people, common), bool),
        "common_padding": np.zeros((people, common), bool),
        "event_metadata": np.zeros((people, event, 12), np.float32),
        "event_reference_features": np.zeros((people, event, 15), np.float32),
        "event_dosage": np.asarray([[0, 1], [2, 1]][:people], np.int8),
        "event_cm": np.broadcast_to(np.asarray([0.00, 0.02]),
                                     (people, event)).copy(),
        "event_chrom": np.full((people, event), 22, np.int64),
        "event_pos_bp": np.broadcast_to(np.asarray([400, 500]),
                                         (people, event)).copy(),
        "event_valid": np.ones((people, event), bool),
        "event_core": np.ones((people, event), bool),
        "event_padding": np.zeros((people, event), bool),
        "query_features": np.zeros((people, query, 1), np.float32),
        "query_cm": np.broadcast_to(np.asarray([0.00, 0.01, 0.02]),
                                     (people, query)).copy(),
        "query_chrom": np.full((people, query), 22, np.int64),
        "query_pos_bp": np.broadcast_to(np.asarray([600, 700, 800]),
                                         (people, query)).copy(),
        "query_valid": np.ones((people, query), bool),
        "query_core": np.ones((people, query), bool),
        "truth_state": np.asarray([[0, 1, 2], [3, 4, 5]][:people], np.int64),
        "truth_observed": np.ones((people, query), bool),
        "query_burden": np.zeros((people, query, 3), np.int64),
        "baseline_probabilities": np.full((people, query, 6), 1 / 6, np.float32),
    }
    return data


def resolved(family):
    value = {
        "family": family, "common_radius_cm": 1.0, "rare_radius_cm": 1.0,
        "width": 8, "depth": 2, "dropout": 0.0,
        "heads": 2 if family == "lai_local_attention" else None,
        "learning_rate": budget.MEASUREMENT_LEARNING_RATE, "optimizer": "Adam",
        "learning_rate_measurement_rule":
            "LOW_ENDPOINT_OF_FROZEN_GEOMETRIC_DOMAIN_FOR_COST_MEASUREMENT_NOT_SELECTION",
    }
    if family == "lai_cnn":
        value.update(query_max_gap_cm=0.011,
                     cnn_query_config={"schema": "lai_query_multiscale_cnn_v1",
                                       "kernel_sizes": [3, 7], "query_step_cm": 0.01})
    return value


class LAIFitBudgetTest(unittest.TestCase):
    def test_factorized_source_remains_prepare_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {
                "schema": "r02_lai_regional_truth_join_request_v2",
                "operation": "PREPARE", "role": "FIT", "condition": "NONE",
                "training_authorized": False, "selection_authorized": False,
                "evaluate_opened": False,
            }
            request_spec = budget.write_fresh(root / "request.json", request)
            receipt = {
                "schema": "r02_lai_regional_truth_factorized_link_receipt_v1",
                "status": "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION",
                "request_sha256": request_spec["sha256"], "role": "FIT",
                "condition": "NONE", "truth_joined": True, "factorized_link": True,
                "training_authorized": False, "selection_authorized": False,
                "evaluate_opened": False,
            }
            receipt_spec = budget.write_fresh(root / "receipt.json", receipt)
            selected = {"link_request": request_spec, "link_receipt": receipt_spec}
            self.assertEqual(budget.authenticate_link_source(selected),
                             (request_spec, receipt_spec))
            receipt["selection_authorized"] = True
            changed = budget.write_fresh(root / "changed.json", receipt)
            with self.assertRaisesRegex(ValueError, "complete factorized source"):
                budget.authenticate_link_source(
                    {"link_request": request_spec, "link_receipt": changed})

    def test_family_specific_cost_proxy_penalizes_attention_event_square(self):
        small = {"people": 128, "common_sites": 10, "event_sites": 2, "query_sites": 10}
        large = dict(small, event_sites=8)
        self.assertGreater(budget.cost_proxy("lai_local_attention", large),
                           budget.cost_proxy("lai_local_attention", small))
        expected = 128 * 64 * (10 + 10 + 8 + 2 * 8 * 8)
        self.assertEqual(budget.cost_proxy("lai_local_attention", large), expected)

    def test_pre_cannot_open_select_or_change_resources(self):
        inventory = {"path": "/tmp/inventory.json", "sha256": "a" * 64}
        value = {
            "schema": budget.PRE_SCHEMA, "status": "ADMIT_ONE_TECHNICAL_FIT_UPDATE",
            "scope": "DEVELOPMENT_BUDGET_MEASUREMENT", "inventory": inventory,
            "family": "lai_cnn", "chromosome": "22", "role": "FIT",
            "condition": "NONE", "radius_cm": 1.0, "width": 64, "updates": 1,
            "threads": 1, "memory_bytes": 8 * 1024**3, "timeout_seconds": 600,
            "seed": 1701,
            "learning_rate_domain": {"low": 1e-4, "high": 1e-3,
                                     "spacing": "GEOMETRIC"},
            "measurement_learning_rate": budget.MEASUREMENT_LEARNING_RATE,
            "checkpoint_recovery": budget.CHECKPOINT_RECOVERY,
            "selection_opened": False, "evaluate_opened": False,
            "scientific_use": budget.EXCLUSION,
        }
        budget.validate_pre(value, inventory, "lai_cnn")
        value["selection_opened"] = True
        with self.assertRaisesRegex(ValueError, "PRE does not admit"):
            budget.validate_pre(value, inventory, "lai_cnn")

    def test_both_real_model_families_restore_excluded_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in budget.FAMILIES:
                receipt = budget.run_update(
                    fixture_data(), resolved(family), root / family,
                    source_identity={"synthetic_code_test": True, "family": family})
                self.assertEqual(receipt["status"],
                                 "ONE_TECHNICAL_FIT_UPDATE_COMPLETE_AND_RECOVERABLE")
                self.assertEqual(receipt["scientific_use"], budget.EXCLUSION)
                self.assertFalse(receipt["selection_eligible"])
                self.assertFalse(receipt["select_accessed"])
                self.assertFalse(receipt["evaluate_opened"])
                self.assertTrue(receipt["checkpoint_restore_exact"])
                self.assertTrue(all(receipt["checkpoint_restore"].values()))
                self.assertEqual(receipt["core_objectives_total"], 6)
                self.assertEqual(receipt["core_objectives_per_mosaic_minimum"], 3)
                self.assertEqual(receipt["valid_core_queries_total"], 6)
                self.assertEqual(receipt["valid_core_queries_per_mosaic_minimum"], 3)
                self.assertEqual(receipt["checkpoint_recovery_boundary"],
                                 budget.CHECKPOINT_RECOVERY)
                persisted = json.loads((root / family / "measurement.receipt.json").read_text())
                self.assertEqual(persisted["updates"], 1)

    def test_human_measurement_requires_all_128_retained_mosaics(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rejected"
            with self.assertRaisesRegex(ValueError, "128 mosaics"):
                budget.run_update(fixture_data(), resolved("lai_cnn"),
                                  output,
                                  source_identity={"predictor_id": "human-source"})
            self.assertFalse(output.exists())

    def test_measurement_requires_an_observed_core_objective_per_mosaic(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = fixture_data()
            data["query_core"][0] = False
            output = Path(temporary) / "rejected"
            with self.assertRaisesRegex(ValueError, "observed core objective"):
                budget.run_update(
                    data, resolved("lai_cnn"), output,
                    source_identity={"synthetic_code_test": True})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
