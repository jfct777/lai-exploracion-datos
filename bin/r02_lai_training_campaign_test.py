"""Small tests for protocol-bound LAI training; no human data or long campaign."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import r02_context_train as trainer
import r02_lai_neural_input as neural_input
import r02_lai_training_campaign as campaign
from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE
from r02_lai_reference_features import reference_features
from r02_lai_regional_channel_bridge import array_sha256


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"path": str(path.resolve()), "sha256": campaign.sha(path)}


class LAITrainingCampaignTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cw2 = write(self.root / "cw2.json", {"source": "CW2"})
        self.ec = write(self.root / "ec.json", {"source": "EC"})
        self.protocol_value = {
            "schema": campaign.PROTOCOL_SCHEMA,
            "scope": "SYNTHETIC_CODE_TEST",
            "execution_mode": campaign.SYNTHETIC_FULL,
            "direction": "FOLD_A_TO_FOLD_B",
            "chromosome": "22",
            "roles": {"FIT": "FOLD_A", "SELECT": "FOLD_B"},
            "conditions": list(campaign.CONDITIONS),
            "evaluate_opened": False,
            "input_plan": None, "input_output_root": None,
            "input_completion": None,
            "objective": {
                "state_order": ["AA", "AE", "AN", "EE", "EN", "NN"],
                "primary_loss": "SIX_STATE_CROSS_ENTROPY_WITHOUT_CLIPPING",
                "truth_support": "TRUTH_OBSERVED_AND_QUERY_VALID",
                "query_ownership": "EACH_CORE_QUERY_EXACTLY_ONCE",
                "training_loss_aggregation":
                    "MEAN_CE_OVER_OBSERVED_CORE_QUERIES_PER_PERSON_THEN_MEAN_128_PEOPLE_PER_REGION_UPDATE",
                "fit_score_aggregation":
                    "SUM_OWNED_QUERY_LOSS_PER_PERSON_THEN_MEAN_COMPONENTS_THEN_MEAN_THREE_CASES",
                "halo_policy": "VISIBLE_TO_PREDICTOR_EXCLUDED_FROM_LOSS_AND_FIT_SCORE",
                "secondary_metrics": list(campaign.SECONDARY_METRICS),
                "boundary_tolerance_cm": 0.01,
                "boundary_metrics_status": "ENABLED",
            },
            "training_unit": {
                "fit_cases": 3, "select_cases": 3, "regions_per_case": 16,
                "fit_batch_order": "FIXED_CASE_THEN_REGION_CYCLE",
                "fit_loss_aggregation":
                    "MEAN_CE_OVER_OBSERVED_QUERIES_PER_PERSON_THEN_MEAN_PEOPLE_PER_BATCH",
                "selection_loss_aggregation":
                    "MEAN_PERSON_LOSS_ACROSS_BATCHES_THEN_MEAN_COMPONENTS",
            },
            "recipe_domain": {
                "families": list(campaign.FAMILIES), "radii_cm": list(campaign.RADII),
                "widths": list(campaign.WIDTHS), "depth": 2, "attention_heads": 4,
                "dropout": 0.0,
                "learning_rate": {"low": 0.0001, "high": 0.001, "log": True},
                "cnn_kernels": campaign.CNN_KERNELS,
                "cw2_source": self.cw2, "ec_source": self.ec,
            },
            "optimization": {
                "name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0.0, "adam_state_precision": "fp32",
                "optimizer_updates": 48, "checkpoint_every_steps": 1,
                "stopping_rule": "FIXED_48_UPDATES_FOR_SYNTHETIC_CODE_TEST",
                "training_seeds": [1701], "stability_seeds": [1701, 1702, 1703],
            },
            "selection": {
                "primary_metric": "component_log_loss",
                "condition_weights": {name: 0.2 for name in campaign.CONDITIONS},
                "require_all_conditions": True, "require_complete_recipe_family": True,
                "expected_recipe_ids": ["cnn-r005-w32"], "search_seed": 20260930,
                "max_finalists": 2,
                "tie_break": "LOWEST_WEIGHTED_LOG_LOSS_THEN_LEXICOGRAPHIC_RECIPE_ID",
            },
            "resources": {
                "threads": 1, "max_batch_bytes": 8_000_000,
                "memory_bytes": 16_000_000, "cpus": 1,
                "measured_update_seconds": 0.01, "estimated_total_cost_usd": 0.0,
            },
            "campaign_budget": None,
            "evidence": {},
        }
        self.protocol = write(self.root / "protocol.json", self.protocol_value)
        self.resolved_value = {
            "family": "lai_cnn", "common_radius_cm": 0.05, "rare_radius_cm": 0.05,
            "width": 32, "depth": 2, "dropout": 0.0, "heads": None,
            "learning_rate": 0.0001, "optimizer": "Adam", "query_max_gap_cm": 0.011,
            "cnn_query_config": {"schema": "lai_query_multiscale_cnn_v1",
                                 "kernel_sizes": [3, 7], "query_step_cm": 0.01},
        }
        self.resolved = write(self.root / "resolved.json", self.resolved_value)
        self.data = {role: self.make_data(role) for role in ("FIT", "SELECT")}

    @staticmethod
    def make_data(role):
        people = np.asarray([role.lower() + "-1", role.lower() + "-2"])
        components = np.asarray([role.lower() + "-component-1", role.lower() + "-component-2"])
        n, common, event, query = 2, 2, 2, 3
        data = {
            "person_id": people, "component_id": components,
            "role": np.asarray([role] * n),
            "common_features": np.zeros((n, common, 12), np.float32),
            "common_cm": np.broadcast_to(np.asarray([0.00, 0.01]), (n, common)).copy(),
            "common_chrom": np.full((n, common), 22, np.int64),
            "common_pos_bp": np.broadcast_to(np.asarray([100, 200]), (n, common)).copy(),
            "common_valid": np.ones((n, common), bool),
            "common_core": np.ones((n, common), bool),
            "common_padding": np.zeros((n, common), bool),
            "event_metadata": np.zeros((n, event, 12), np.float32),
            "event_dosage": np.asarray([[0, 1], [2, 1]], np.int8),
            "event_cm": np.broadcast_to(np.asarray([0.00, 0.01]), (n, event)).copy(),
            "event_chrom": np.full((n, event), 22, np.int64),
            "event_pos_bp": np.broadcast_to(np.asarray([300, 400]), (n, event)).copy(),
            "event_valid": np.ones((n, event), bool),
            "event_core": np.ones((n, event), bool),
            "event_padding": np.zeros((n, event), bool),
            "event_reference_ac": np.asarray([[1, 2, 3], [2, 3, 4]], np.int64),
            "event_reference_an": np.asarray([[10, 10, 10], [10, 10, 10]], np.int64),
            "event_reference_known": np.ones((event, 3), bool),
            "event_reference_ancestry_order": np.asarray(
                ["African", "European", "Native_American"]),
            "event_counted_allele": np.asarray(["G", "T"]),
            "event_ref": np.asarray(["A", "C"]), "event_alt": np.asarray(["G", "T"]),
            "event_is_alt": np.asarray([True, True]),
            "event_global_index": np.asarray([0, 1], np.int64),
            "query_features": np.zeros((n, query, 1), np.float32),
            "query_cm": np.broadcast_to(np.asarray([0.00, 0.01, 0.02]), (n, query)).copy(),
            "query_chrom": np.full((n, query), 22, np.int64),
            "query_pos_bp": np.broadcast_to(np.asarray([500, 600, 700]), (n, query)).copy(),
            "query_valid": np.ones((n, query), bool),
            "truth_state": np.asarray([[0, 1, 2], [3, 4, 5]], np.int64),
            "truth_observed": np.ones((n, query), bool),
            "query_burden": np.zeros((n, query, 3), np.int64),
            "baseline_probabilities": np.full((n, query, 6), 1 / 6, np.float32),
        }
        data["event_reference_features"] = reference_features(data, np)
        return data

    def source_receipt(self, role, case_index, region_index, condition="NONE"):
        data = self.data[role]
        request = write(self.root / f"{role}-{case_index}-{region_index}-{condition}.request.json",
                        {"fixture": True, "role": role, "case": case_index,
                         "region": region_index, "condition": condition})
        receipt_value = {
            "schema": "r02_lai_regional_truth_factorized_link_receipt_v1",
            "status": "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION",
            "request_sha256": request["sha256"], "role": role, "condition": condition,
            "direction": "FOLD_A_TO_FOLD_B", "chromosome": "22",
            "case_id": f"R02L2pilot.PULSE_T10_ANCHOR.chr22.FOLD_{'A' if role == 'FIT' else 'B'}.r{case_index + 1}",
            "region_id": f"region-{region_index:02d}",
            "task_id": f"fixture-{role}-{case_index}-{region_index}",
            "arrays": sorted(data),
            "reconstructed_arrays_sha256": {name: array_sha256(value, np)
                                               for name, value in data.items()},
            "factorized_link": True, "truth_joined": True,
            "training_authorized": False, "selection_authorized": False,
            "synthetic_code_test": True,
            "lai_burden_scope": SCOPE, "lai_burden_boundary": BOUNDARY,
            "lai_burden_count_names": list(COUNT_NAMES),
            "event_burden_prefix_receipt_sha256": "f" * 64,
            "rare_radius_cm": 0.05,
            "density_matched_status": ("MATERIALIZED_FOR_THIS_RADIUS_PREPARE_ONLY"
                                       if condition == "DENSITY_MATCHED" else
                                       "SEPARATE_RADIUS_SPECIFIC_PRODUCT"),
            "evaluate_opened": False,
        }
        receipt = write(self.root / f"{role}-{case_index}-{region_index}-{condition}.prepare.json",
                        receipt_value)
        return request, receipt, receipt_value

    def training_entry(self, role, case_index, region_index, condition="NONE"):
        request, source, source_value = self.source_receipt(role, case_index, region_index, condition)
        wrapper_value = {
            "schema": campaign.TRAIN_LINK_SCHEMA, "status": campaign.TRAIN_LINK_STATUS,
            "request_sha256": request["sha256"], "source_prepare_receipt": source,
            "protocol": self.protocol,
            "protocol_decision_sha256": campaign.digest(
                campaign.decision_payload(self.protocol_value)),
            "resolved_configuration_sha256": campaign.digest(self.resolved_value),
            "direction": "FOLD_A_TO_FOLD_B", "chromosome": "22", "role": role,
            "condition": condition, "case_id": source_value["case_id"],
            "region_id": source_value["region_id"], "predictor_id": source_value["task_id"],
            "arrays": source_value["arrays"],
            "reconstructed_arrays_sha256": source_value["reconstructed_arrays_sha256"],
            "evaluate_opened": False,
        }
        wrapper = write(self.root / f"{role}-{case_index}-{region_index}-{condition}.train.json",
                        wrapper_value)
        return {"format": "LAI_FACTORIZED_TRAINING_LINK_V1",
                "path": request["path"], "sha256": request["sha256"],
                "receipt_path": wrapper["path"], "receipt_sha256": wrapper["sha256"]}

    def make_contract(self, condition="NONE"):
        batches = {role: [self.training_entry(role, case_index, region_index, condition)
                          for case_index in range(3) for region_index in range(16)]
                   for role in ("FIT", "SELECT")}
        persons = {}
        for role in ("FIT", "SELECT"):
            for person, component in zip(self.data[role]["person_id"],
                                         self.data[role]["component_id"]):
                persons[str(person)] = {"role": role, "component": str(component)}
        identity = {name: character * 64 for name, character in zip(
            ("event_catalogue_sha256", "ref_train_receipt_sha256",
             "ref_support_by_provenance_sha256", "reference_vcf_sha256",
             "reference_roles_sha256"), "abcde")}
        return {
            "schema": trainer.SCHEMA, "scope": "SYNTHETIC_CODE_TEST", "objective": "lai",
            "admission": {"status": "CODE_TEST_ONLY"},
            "target_definition": "six simulated diploid AFR/EUR/NAM ancestry states",
            "count_scope": "query-local DNABR2723 source-minor counts",
            "feature_definition": "COMMON12 EVENT12+dosage+REF15 and revised FLARE probabilities",
            "persons": persons, "batches": batches, "steps": 48, "threads": 1,
            "dataset_declared_scope": "BOUNDED_CONTEXT_PILOT",
            "limits": {"max_common_sites": 8, "max_event_sites": 8,
                       "max_targets": 8, "max_batch_bytes": 8_000_000},
            "common_features": 12, "event_metadata_features": 12,
            "event_reference_features": 15,
            "event_reference_feature_schema": "r02_lai_event_reference_features_v1",
            "event_reference_feature_formula":
                "PER_ANCESTRY(AC_0,AN_0,AF_0,KNOWN,FREQUENCY_DEFINED);AF_0=AC/AN_IFF_KNOWN_AND_AN_GT_0",
            "event_reference_ancestry_order": ["African", "European", "Native_American"],
            "reference_identity": identity, "lai_burden_scope": SCOPE,
            "lai_burden_boundary": BOUNDARY, "lai_burden_count_names": list(COUNT_NAMES),
            "lai_burden_transform": "LOG1P_IN_MODEL",
            "rare_allele": "DNABR2723_FIXED_SOURCE_MINOR",
            "rare_phase": "DIPLOID_UNPHASED_NO_HAPLOTYPE_ASSIGNMENT",
            "query_max_gap_cm": 0.011, "information_condition": condition,
            "lai_training_protocol": self.protocol,
        }

    def test_protocol_rejects_reciprocal_and_unreviewed_development(self):
        changed = copy.deepcopy(self.protocol_value)
        changed["direction"] = "FOLD_B_TO_FOLD_A"
        with self.assertRaisesRegex(ValueError, "current campaign"):
            campaign.validate_protocol_value(changed)
        changed = copy.deepcopy(self.protocol_value)
        changed["scope"] = "DEVELOPMENT"
        changed["execution_mode"] = campaign.FIT_ONLY_INITIAL
        with self.assertRaisesRegex(ValueError, "chromosome PREPARE completion"):
            campaign.validate_protocol_value(changed)
        changed["input_completion"] = write(self.root / "complete.json", {
            "schema": "r02_lai_factorized_chr22_prepare_completion_v2",
            "status": "PREPARE_COMPLETE_NO_TRAIN_NO_EVALUATE",
            "completed_at": "fixture", "predictors": 576, "links": 2880,
            "combined_condition_batches": 0, "chromosomes": ["22"],
            "training_started": False, "evaluate_opened": False,
            "publication_layout": "predictors/ID/prepared and links/ID/linked",
            "verification":
                "ALL_NESTED_PREDICTOR_BATCH_AND_REQUEST_HASHES_AND_FIVE_LINK_IDENTITIES_MATCH",
        })
        with self.assertRaisesRegex(ValueError, "factorized PREPARE plan"):
            campaign.validate_protocol_value(changed)
        pending = json.loads((ROOT / "bin" /
                              "r02_lai_training_campaign_PROTOCOL_PENDING.json").read_text())
        with self.assertRaisesRegex(ValueError, "invalid chromosome PREPARE completion descriptor"):
            campaign.validate_protocol_value(pending)

    def test_fit_only_protocol_defers_select_and_boundary_metrics(self):
        changed = copy.deepcopy(self.protocol_value)
        changed["execution_mode"] = campaign.FIT_ONLY_INITIAL
        changed["objective"].update(
            secondary_metrics=[], boundary_tolerance_cm=None,
            boundary_metrics_status="DEFERRED_BEFORE_SELECT")
        changed["selection"] = {
            "status": "DEFERRED_BEFORE_SELECT", "select_metadata_bound": True,
            "select_values_loaded": False, "select_scoring_allowed": False,
            "evaluate_allowed": False, "preserved_recipe_domain_only": True,
        }
        changed["campaign_budget"] = {
            "fits": 10, "families": 2, "conditions": 5, "workers": 2,
            "cpu_per_worker": 1,
            "memory_bytes_per_worker": changed["resources"]["memory_bytes"],
            "shared_wall_seconds": 72000,
            "countdown_starts": "FIRST_SCIENTIFIC_TRAINING_LAUNCH",
            "retries_reset_budget": False,
            "budget_includes": ["48_INITIAL_FIT_LOADS_PER_FIT",
                                "96_ADAM_UPDATES_PER_FIT",
                                "48_FINAL_FIT_SCORES_PER_FIT"],
            "select_scoring_included": False,
        }
        campaign.validate_protocol_value(changed)
        bad = copy.deepcopy(changed)
        bad["selection"]["select_scoring_allowed"] = True
        with self.assertRaisesRegex(ValueError, "keep SELECT"):
            campaign.validate_protocol_value(bad)

    def test_complete_unit_and_recipe_guards(self):
        contract = self.make_contract()
        campaign.validate_campaign_contract(contract)
        missing = copy.deepcopy(contract)
        missing["batches"]["FIT"].pop()
        with self.assertRaisesRegex(ValueError, "complete case-region"):
            campaign.validate_campaign_contract(missing)
        attention = copy.deepcopy(self.resolved_value)
        attention.update(family="lai_local_attention", heads=4)
        attention.pop("cnn_query_config")
        campaign.validate_resolved(self.protocol_value, attention)
        wrong = copy.deepcopy(self.resolved_value)
        wrong["cnn_query_config"]["kernel_sizes"] = [3, 21]
        with self.assertRaisesRegex(ValueError, "kernels"):
            campaign.validate_resolved(self.protocol_value, wrong)

    def test_training_link_uses_protocol_chromosome_and_rejects_foreign_source(self):
        protocol_value = copy.deepcopy(self.protocol_value)
        protocol_value["chromosome"] = "1"
        protocol = write(self.root / "chr1-protocol.json", protocol_value)
        request, source, source_value = self.source_receipt("FIT", 0, 0)
        source_value["chromosome"] = "1"
        source_value["case_id"] = source_value["case_id"].replace("chr22.", "chr1.")
        source = write(self.root / "chr1-source.json", source_value)
        result = campaign.create_training_link(
            request["path"], request["sha256"], source["path"], source["sha256"],
            protocol["path"], protocol["sha256"], self.resolved["path"],
            self.resolved["sha256"], "FIT", "NONE", self.root / "chr1-link.json")
        self.assertEqual(result["chromosome"], "1")
        foreign_value = copy.deepcopy(source_value)
        foreign_value["chromosome"] = "22"
        foreign = write(self.root / "foreign-source.json", foreign_value)
        with self.assertRaisesRegex(ValueError, "source PREPARE"):
            campaign.create_training_link(
                request["path"], request["sha256"], foreign["path"], foreign["sha256"],
                protocol["path"], protocol["sha256"], self.resolved["path"],
                self.resolved["sha256"], "FIT", "NONE", self.root / "foreign-link.json")

    def test_case_specific_identity_rejects_crossed_realization_before_arrays(self):
        fields = sorted(campaign.REFERENCE_IDENTITY_FIELDS)
        first = {name: "a" * 64 for name in fields}
        second = {name: "b" * 64 for name in fields}
        contract = {"reference_identity_by_case": {"case-1": first, "case-2": second}}
        campaign.validate_case_reference_identity(contract, "case-1", first)
        with self.assertRaisesRegex(ValueError, "case-specific"):
            campaign.validate_case_reference_identity(contract, "case-1", second)

    def test_factorized_link_resolves_unique_plan_predictor_descriptor(self):
        predictor = write(self.root / "regional-predictor.json", {
            "task_id": "case.direction", "case_id": "case", "role": "FIT"})
        link = write(self.root / "factorized-link.json", {
            "schema": "r02_lai_regional_truth_join_request_v2",
            "task_id": "case.direction", "case_id": "case", "role": "FIT",
            "regional_request": predictor})
        planned = {"case.direction.lai_cnn.r1.region-00": predictor}
        reverse = {(str(Path(predictor["path"]).resolve()), predictor["sha256"]):
                   "case.direction.lai_cnn.r1.region-00"}
        observed, value = campaign.canonical_planned_predictor(link, planned, reverse)
        self.assertEqual(observed, "case.direction.lai_cnn.r1.region-00")
        self.assertEqual(value["task_id"], "case.direction")
        with self.assertRaisesRegex(ValueError, "one authenticated PLAN predictor"):
            campaign.canonical_planned_predictor(link, planned, {})

    def test_existing_checkpoint_recovers_protocol_bound_factorized_training(self):
        contract_value = self.make_contract()
        contract = write(self.root / "contract.json", contract_value)

        def reconstructed(request_path, request_sha256, receipt_path, receipt_sha256,
                          *, reconstruction_limits, action):
            self.assertEqual(action, "PREPARE")
            self.assertEqual(set(reconstruction_limits), {"max_factorized_source_bytes",
                "max_reconstructed_join_bytes", "max_source_plus_reconstructed_bytes"})
            role = "FIT" if Path(request_path).name.startswith("FIT-") else "SELECT"
            return copy.deepcopy(self.data[role])

        def run(name, checkpoint, stop):
            with patch.object(neural_input, "load_factorized_link", side_effect=reconstructed):
                return campaign.execute(contract["path"], contract["sha256"],
                    self.resolved["path"], self.resolved["sha256"], "NONE", 1701,
                    self.root / name, self.root / checkpoint, fit_only=True,
                    stop_after_steps=stop)

        first = run("first", "continued", 1)
        resumed = run("resumed", "continued", 2)
        continuous = run("continuous", "fresh", 2)
        self.assertEqual(first["status"], "FIT_CHECKPOINT")
        self.assertEqual(resumed["internal_checkpoint"]["restored_steps"], 1)
        self.assertEqual(resumed["training_losses"], continuous["training_losses"])
        with np.load(self.root / "resumed/weights.npz") as left, \
             np.load(self.root / "continuous/weights.npz") as right:
            self.assertEqual(set(left.files), set(right.files))
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])

    def test_public_trainer_enforces_protocol_arm_and_runtime_parameters(self):
        contract = write(self.root / "direct-contract.json", self.make_contract())
        common = (contract["path"], contract["sha256"], self.resolved["path"],
                  self.resolved["sha256"])
        with self.assertRaisesRegex(ValueError, "LAI arm, seed, scope"):
            trainer.train(*common, "NO_RARE", 1701, self.root / "wrong-arm",
                          fit_only=True, stop_after_steps=1)
        with self.assertRaisesRegex(ValueError, "LAI arm, seed, scope"):
            trainer.train(*common, "NONE", 1701, self.root / "wrong-adam",
                          adam_state_precision="fp64", fit_only=True,
                          stop_after_steps=1)

    def test_campaign_cli_propagates_person_microbatch_checkpoint_only(self):
        with patch.object(campaign, "execute", return_value={"status": "fixture"}) as execute:
            campaign.main([
                "execute", "--contract", "contract.json",
                "--contract-sha256", "a" * 64, "--resolved", "resolved.json",
                "--resolved-sha256", "b" * 64, "--condition", "NONE",
                "--seed", "1701", "--output", "output", "--checkpoint-dir", "checkpoint",
                "--fit-only", "--stop-after-steps", "1",
                "--lai-people-per-microbatch", "8", "--checkpoint-only",
            ])
        self.assertEqual(execute.call_args.kwargs["lai_people_per_microbatch"], 8)
        self.assertTrue(execute.call_args.kwargs["checkpoint_only"])

    def test_public_trainer_rejects_fit_only_protocol_before_arrays(self):
        contract_value = self.make_contract()
        contract = write(self.root / "fit-only-direct-contract.json", contract_value)
        protocol = copy.deepcopy(self.protocol_value)
        protocol["execution_mode"] = campaign.FIT_ONLY_INITIAL
        with patch.object(trainer, "validate_contract", return_value=contract_value), \
             patch.object(campaign, "validate_protocol", return_value=(protocol, "d" * 64)), \
             patch.object(neural_input, "load_factorized_link",
                          side_effect=AssertionError("arrays must remain unopened")):
            with self.assertRaisesRegex(ValueError, "requires fit_only before any batch"):
                trainer.train(contract["path"], contract["sha256"],
                              self.resolved["path"], self.resolved["sha256"],
                              "NONE", 1701, self.root / "forbidden-select-entry",
                              fit_only=False, stop_after_steps=1)

    def test_selection_requires_all_five_conditions_and_primary_seed(self):
        results = {}
        for index, condition in enumerate(campaign.CONDITIONS):
            contract = write(self.root / f"contract-{condition}.json",
                             self.make_contract(condition))
            receipt = write(self.root / f"result-{condition}.json", {
                "status": "COMPLETE", "score_role": "SELECT", "select_accessed": True,
                "arm": condition, "resolved_sha256": self.resolved["sha256"],
                "contract_sha256": contract["sha256"], "scope": "SYNTHETIC_CODE_TEST",
                "frozen_training_ceiling_steps": 48, "steps": 48, "seed": 1701,
                "metrics": {"component_log_loss": float(index + 1)},
            })
            results[condition] = [{"receipt": receipt, "contract": contract}]
        manifest_value = {"schema": campaign.SELECT_MANIFEST_SCHEMA,
                          "protocol": self.protocol, "evaluate_opened": False,
                          "recipes": [{"recipe_id": "cnn-r005-w32",
                                       "resolved_sha256": self.resolved["sha256"],
                                       "results": results}]}
        manifest = write(self.root / "select.json", manifest_value)
        selected = campaign.select(manifest["path"], manifest["sha256"],
                                   self.root / "selection.json")
        self.assertEqual(selected["selected_recipe_id"], "cnn-r005-w32")
        incomplete = copy.deepcopy(manifest_value)
        incomplete["recipes"][0]["results"].pop("IDENTITY_SHAM")
        incomplete_path = write(self.root / "select-incomplete.json", incomplete)
        with self.assertRaisesRegex(ValueError, "five paired"):
            campaign.select(incomplete_path["path"], incomplete_path["sha256"],
                            self.root / "selection-incomplete.json")


if __name__ == "__main__":
    unittest.main()
