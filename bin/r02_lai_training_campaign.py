#!/usr/bin/env python3
"""Bind factorized LAI regions to authenticated, resumable FIT/SELECT training.

The existing factorized receipts remain PREPARE-only.  This module creates and
checks a second, small receipt that binds one PREPARE link to an exact scientific
protocol and resolved model configuration.  ``r02_context_train`` accepts only
that second receipt for training.  No operation in this module opens EVALUATE.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import time


PROTOCOL_SCHEMA = "r02_lai_training_protocol_v2"
FIRST_UPDATE_PROTOCOL_SCHEMA = "r02_lai_first_retained_update_protocol_v1"
EVIDENCE_SCHEMA = "r02_lai_training_admission_evidence_v1"
FIRST_UPDATE_EVIDENCE_SCHEMA = "r02_lai_first_update_admission_evidence_v1"
FIRST_UPDATE_EXECUTION_BUDGET_SCHEMA = "r02_lai_first_update_execution_budget_v1"
REPAIR_EXECUTION_BUDGET_SCHEMA = "r02_lai_repair_same_fit_execution_budget_v1"
RECONSTRUCTION_EXECUTION_LIMIT_SCHEMA = "r02_lai_reconstruction_execution_limit_v2"
TRAIN_LINK_SCHEMA = "r02_lai_factorized_training_link_receipt_v1"
TRAIN_LINK_STATUS = "FACTORIZED_LINK_BOUND_TO_AUTHENTICATED_TRAINING_PROTOCOL"
SELECT_MANIFEST_SCHEMA = "r02_lai_training_selection_manifest_v1"
SELECTION_SCHEMA = "r02_lai_training_selection_v1"
CONDITIONS = ("NONE", "NO_RARE", "COUNT_MATCHED", "DENSITY_MATCHED", "IDENTITY_SHAM")
FAMILIES = ("lai_cnn", "lai_local_attention")
RADII = (0.05, 0.2, 1.0)
WIDTHS = (32, 64)
CNN_KERNELS = {"0.05": [3, 7], "0.2": [3, 21], "1": [3, 101]}
TRAINING_SEEDS = (1701,)
STABILITY_SEEDS = (1701, 1702, 1703)
SECONDARY_METRICS = ("six_state_brier", "dosage_mae_AFR", "dosage_mae_EUR",
                     "dosage_mae_NAM", "boundary_metric_with_protocol_tolerance")
FIT_ONLY_INITIAL = "FIT_ONLY_INITIAL"
SYNTHETIC_FULL = "SYNTHETIC_FULL"
HASH = re.compile(r"[0-9a-f]{64}")
REFERENCE_IDENTITY_FIELDS = {
    "event_catalogue_sha256", "ref_train_receipt_sha256",
    "ref_support_by_provenance_sha256", "reference_vcf_sha256",
    "reference_roles_sha256",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path, expected_sha=None):
    path = Path(path)
    if expected_sha is not None:
        require(sha(path) == expected_sha, "JSON SHA-256 differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("nonfinite JSON value: " + value)))


def descriptor(value, name):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and value["path"]
            and bool(HASH.fullmatch(value["sha256"] or "")),
            f"invalid {name} descriptor")
    path = Path(value["path"])
    require(path.is_file() and sha(path) == value["sha256"], f"{name} differs")
    return path


def decision_payload(protocol):
    return {key: value for key, value in protocol.items() if key != "evidence"}


def _validate_evidence(protocol, decision_sha256):
    evidence = protocol["evidence"]
    if protocol["scope"] == "SYNTHETIC_CODE_TEST":
        require(evidence == {}, "synthetic protocol cannot claim scientific review")
        return
    if protocol["schema"] == FIRST_UPDATE_PROTOCOL_SCHEMA:
        expected = {"council_root": "COUNCIL_ROOT_DECISION",
                    "claude": "CLAUDE_EXTERNAL_REVIEW"}
        require(set(evidence) == set(expected),
                "first-update protocol requires root and external review records")
        for name, role in expected.items():
            record = read_json(descriptor(evidence[name], name + " evidence"),
                               evidence[name]["sha256"])
            require(set(record) == {"schema", "role", "scope", "outcome",
                                    "decision_sha256", "evaluate_opened"}
                    and record["schema"] == FIRST_UPDATE_EVIDENCE_SCHEMA
                    and record["role"] == role and record["scope"] == "DEVELOPMENT"
                    and record["outcome"] == "ADMIT_TWO_FIRST_RETAINED_UPDATES"
                    and record["decision_sha256"] == decision_sha256
                    and record["evaluate_opened"] is False,
                    f"{name} evidence does not admit this exact first-update decision")
        return
    require(set(evidence) == {"consejo", "claude", "independent_audit", "fit_budget_measurement"},
            "development protocol requires four authenticated evidence records")
    expected = {
        "consejo": "CONSEJO_PRE",
        "claude": "CLAUDE_REVIEW",
        "independent_audit": "INDEPENDENT_AUDIT",
        "fit_budget_measurement": "FIT_BUDGET_MEASUREMENT",
    }
    for name, role in expected.items():
        record = read_json(descriptor(evidence[name], name + " evidence"),
                           evidence[name]["sha256"])
        require(set(record) == {"schema", "role", "scope", "outcome",
                                "decision_sha256", "evaluate_opened"}
                and record["schema"] == EVIDENCE_SCHEMA
                and record["role"] == role and record["scope"] == "DEVELOPMENT"
                and record["outcome"] == ("SUPPORTS_FIXED_BUDGET" if
                    role == "FIT_BUDGET_MEASUREMENT" else "ADMIT")
                and record["decision_sha256"] == decision_sha256
                and record["evaluate_opened"] is False,
                f"{name} evidence does not admit this exact decision")


def validate_protocol_value(protocol):
    common_fields = {"schema", "scope", "execution_mode", "direction", "chromosome", "roles", "conditions",
              "evaluate_opened", "objective", "training_unit", "recipe_domain",
              "optimization", "selection", "resources", "input_plan",
              "input_output_root", "input_completion", "campaign_budget", "evidence"}
    first_update = (isinstance(protocol, dict)
                    and protocol.get("schema") == FIRST_UPDATE_PROTOCOL_SCHEMA)
    fields = common_fields
    require(isinstance(protocol, dict) and set(protocol) == fields
            and protocol["schema"] in {PROTOCOL_SCHEMA, FIRST_UPDATE_PROTOCOL_SCHEMA},
            "training protocol fields/schema differ")
    require(protocol["scope"] in {"SYNTHETIC_CODE_TEST", "DEVELOPMENT"},
            "unknown training protocol scope")
    require(protocol["execution_mode"] in {FIT_ONLY_INITIAL, SYNTHETIC_FULL}
            and (protocol["scope"] != "DEVELOPMENT" or
                 protocol["execution_mode"] == FIT_ONLY_INITIAL),
            "human DEVELOPMENT currently permits only FIT_ONLY_INITIAL")
    chromosome = str(protocol.get("chromosome"))
    require(protocol["direction"] == "FOLD_A_TO_FOLD_B"
            and chromosome.isdigit() and 1 <= int(chromosome) <= 22
            and chromosome == str(int(chromosome))
            and protocol["roles"] == {"FIT": "FOLD_A", "SELECT": "FOLD_B"},
            "current campaign requires one canonical autosome and FOLD_A_TO_FOLD_B")
    require(protocol["conditions"] == list(CONDITIONS)
            and protocol["evaluate_opened"] is False,
            "five ordered conditions required and EVALUATE must remain closed")
    completion_spec = protocol["input_completion"]
    if protocol["scope"] == "SYNTHETIC_CODE_TEST":
        require(completion_spec is None and protocol["input_plan"] is None
                and protocol["input_output_root"] is None,
                "synthetic code test cannot claim the human PREPARE completion")
    else:
        completion = read_json(descriptor(completion_spec, "chromosome PREPARE completion"),
                               completion_spec["sha256"])
        valid_completion_schema = (
            completion.get("schema") ==
                "r02_lai_factorized_chromosome_prepare_completion_v1" or
            (chromosome == "22" and completion.get("schema") ==
                "r02_lai_factorized_chr22_prepare_completion_v2"))
        require(valid_completion_schema
                and completion.get("status") == "PREPARE_COMPLETE_NO_TRAIN_NO_EVALUATE"
                and completion.get("predictors") == 576
                and completion.get("links") == 2880
                and completion.get("combined_condition_batches") == 0
                and completion.get("chromosomes") == [chromosome]
                and completion.get("training_started") is False
                and completion.get("evaluate_opened") is False
                and completion.get("publication_layout") ==
                    "predictors/ID/prepared and links/ID/linked"
                and completion.get("verification") ==
                    "ALL_NESTED_PREDICTOR_BATCH_AND_REQUEST_HASHES_AND_FIVE_LINK_IDENTITIES_MATCH",
                "DEVELOPMENT requires the verified 576-predictor/2880-link completion for its chromosome")
        plan_spec = protocol["input_plan"]
        plan = read_json(descriptor(plan_spec, "factorized PREPARE plan"),
                         plan_spec["sha256"])
        chromosome_predictors = [row for row in plan.get("predictors", [])
                                 if str(row.get("chromosome")) == chromosome]
        chromosome_links = [row for row in plan.get("links", [])
                            if str(row.get("chromosome")) == chromosome]
        require(plan.get("conditions") == list(CONDITIONS)
                and plan.get("evaluate_opened") is False
                and len(chromosome_predictors) == 576
                and len(chromosome_links) == 2880,
                "factorized PREPARE plan lacks the exact chromosome five-condition inventory")
        output_root = Path(protocol["input_output_root"])
        require(output_root.is_absolute()
                and Path(completion_spec["path"]).resolve() ==
                    (output_root.parent / "COMPLETE.json").resolve(),
                "PREPARE completion is not rooted beside its authenticated output inventory")

    objective = protocol["objective"]
    expected_objective = {
        "state_order": ["AA", "AE", "AN", "EE", "EN", "NN"],
        "primary_loss": "SIX_STATE_CROSS_ENTROPY_WITHOUT_CLIPPING",
        "truth_support": "TRUTH_OBSERVED_AND_QUERY_VALID",
        "query_ownership": "EACH_CORE_QUERY_EXACTLY_ONCE",
        "training_loss_aggregation":
            "MEAN_CE_OVER_OBSERVED_CORE_QUERIES_PER_PERSON_THEN_MEAN_128_PEOPLE_PER_REGION_UPDATE",
        "fit_score_aggregation":
            "SUM_OWNED_QUERY_LOSS_PER_PERSON_THEN_MEAN_COMPONENTS_THEN_MEAN_THREE_CASES",
        "halo_policy": "VISIBLE_TO_PREDICTOR_EXCLUDED_FROM_LOSS_AND_FIT_SCORE",
    }
    if protocol["execution_mode"] == FIT_ONLY_INITIAL:
        expected_objective.update(secondary_metrics=[], boundary_tolerance_cm=None,
                                  boundary_metrics_status="DEFERRED_BEFORE_SELECT")
    else:
        boundary_tolerance = objective.get("boundary_tolerance_cm") if isinstance(objective, dict) else None
        require(type(boundary_tolerance) in (int, float)
                and math.isfinite(boundary_tolerance) and boundary_tolerance > 0,
                "boundary tolerance in cM must be fixed before full scoring")
        expected_objective.update(secondary_metrics=list(SECONDARY_METRICS),
                                  boundary_tolerance_cm=boundary_tolerance,
                                  boundary_metrics_status="ENABLED")
    require(objective == expected_objective,
            "LAI target, loss support, ownership, aggregation, halo, or metrics differ")

    unit = protocol["training_unit"]
    require(set(unit) == {"fit_cases", "select_cases", "regions_per_case",
                          "fit_batch_order", "fit_loss_aggregation",
                          "selection_loss_aggregation"}
            and unit["fit_cases"] == 3 and unit["select_cases"] == 3
            and unit["regions_per_case"] == 16
            and unit["fit_batch_order"] == "FIXED_CASE_THEN_REGION_CYCLE"
            and unit["fit_loss_aggregation"] ==
                "MEAN_CE_OVER_OBSERVED_QUERIES_PER_PERSON_THEN_MEAN_PEOPLE_PER_BATCH"
            and unit["selection_loss_aggregation"] in {
                "MEAN_PERSON_LOSS_ACROSS_BATCHES_THEN_MEAN_COMPONENTS",
                "QUERY_SUM_PER_PERSON_THEN_MEAN_COMPONENTS_THEN_MEAN_CASES_V2"},
            "training unit or implemented loss aggregation differs")
    if protocol["scope"] == "DEVELOPMENT":
        require(unit["selection_loss_aggregation"] ==
                "QUERY_SUM_PER_PERSON_THEN_MEAN_COMPONENTS_THEN_MEAN_CASES_V2",
                "human regional scoring requires unique-query aggregation")

    domain = protocol["recipe_domain"]
    require(set(domain) == {"families", "radii_cm", "widths", "depth", "attention_heads",
                            "dropout", "learning_rate", "cnn_kernels", "cw2_source",
                            "ec_source"}
            and domain["families"] == list(FAMILIES)
            and domain["radii_cm"] == list(RADII) and domain["widths"] == list(WIDTHS)
            and domain["depth"] == 2 and domain["attention_heads"] == 4
            and domain["dropout"] == 0.0
            and domain["learning_rate"] == {"low": 0.0001, "high": 0.001, "log": True}
            and domain["cnn_kernels"] == CNN_KERNELS,
            "CW2/EC recipe domain differs")
    descriptor(domain["cw2_source"], "CW2 source")
    descriptor(domain["ec_source"], "EC source")

    optimization = protocol["optimization"]
    require(set(optimization) == {"name", "betas", "eps", "weight_decay",
                                  "adam_state_precision", "optimizer_updates",
                                  "checkpoint_every_steps", "stopping_rule",
                                  "training_seeds", "stability_seeds"}
            and optimization["name"] == "Adam"
            and optimization["betas"] == [0.9, 0.999]
            and optimization["eps"] == 1e-8 and optimization["weight_decay"] == 0.0
            and optimization["adam_state_precision"] in {"fp32", "fp64"}
            and type(optimization["optimizer_updates"]) is int
            and optimization["optimizer_updates"] >= 48
            and type(optimization["checkpoint_every_steps"]) is int
            and 1 <= optimization["checkpoint_every_steps"] <= optimization["optimizer_updates"]
            and isinstance(optimization["stopping_rule"], str)
            and bool(optimization["stopping_rule"].strip())
            and optimization["training_seeds"] == list(TRAINING_SEEDS)
            and optimization["stability_seeds"] == list(STABILITY_SEEDS),
            "optimizer, measured update budget, checkpoint cadence, or seeds differ")

    selection = protocol["selection"]
    if protocol["execution_mode"] == FIT_ONLY_INITIAL:
        require(selection == {
            "status": "DEFERRED_BEFORE_SELECT",
            "select_metadata_bound": True,
            "select_values_loaded": False,
            "select_scoring_allowed": False,
            "evaluate_allowed": False,
            "preserved_recipe_domain_only": True,
        }, "FIT_ONLY_INITIAL must keep SELECT and EVALUATE closed")
    else:
        weights = selection.get("condition_weights") if isinstance(selection, dict) else None
        require(isinstance(weights, dict) and set(weights) == set(CONDITIONS)
                and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                        for value in weights.values())
                and math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12),
                "five explicit positive condition weights must sum to one")
        require(set(selection) == {"primary_metric", "condition_weights", "require_all_conditions",
                                   "require_complete_recipe_family", "expected_recipe_ids",
                                   "search_seed", "max_finalists", "tie_break"}
                and selection["primary_metric"] == "component_log_loss"
                and selection["require_all_conditions"] is True
                and selection["require_complete_recipe_family"] is True
                and isinstance(selection["expected_recipe_ids"], list)
                and selection["expected_recipe_ids"]
                and len(set(selection["expected_recipe_ids"])) == len(selection["expected_recipe_ids"])
                and all(isinstance(value, str) and value for value in selection["expected_recipe_ids"])
                and selection["search_seed"] == 20260930
                and selection["max_finalists"] == 2
                and selection["tie_break"] ==
                    "LOWEST_WEIGHTED_LOG_LOSS_THEN_LEXICOGRAPHIC_RECIPE_ID",
                "selection rule or complete recipe inventory differs")

    resources = protocol["resources"]
    resource_fields = ({"threads", "max_batch_bytes", "memory_bytes", "cpus"}
                       if first_update else
                       {"threads", "max_batch_bytes", "memory_bytes", "cpus",
                        "measured_update_seconds", "estimated_total_cost_usd"})
    require(set(resources) == resource_fields
            and type(resources["threads"]) is int and 1 <= resources["threads"] <= 8
            and type(resources["max_batch_bytes"]) is int and resources["max_batch_bytes"] > 0
            and type(resources["memory_bytes"]) is int
            and resources["memory_bytes"] >= resources["max_batch_bytes"]
            and type(resources["cpus"]) is int and resources["cpus"] >= resources["threads"]
            and (first_update or
                 (type(resources["measured_update_seconds"]) in (int, float)
                  and math.isfinite(resources["measured_update_seconds"])
                  and resources["measured_update_seconds"] > 0
                  and type(resources["estimated_total_cost_usd"]) in (int, float)
                  and math.isfinite(resources["estimated_total_cost_usd"])
                  and resources["estimated_total_cost_usd"] >= 0)),
            "measured resource and cost fields are incomplete")

    campaign_budget = protocol["campaign_budget"]
    if first_update:
        require(campaign_budget == {
                    "status": "PENDING_FIRST_RETAINED_UPDATE_MEASUREMENT",
                    "fits": 10, "optimizer_updates_per_fit": 96,
                    "full_wall_seconds": None, "full_cost_usd": None},
                "first-update protocol must not invent a full campaign budget")
        require(resources == {"threads": 1, "max_batch_bytes": 350925967,
                              "memory_bytes": 8589934592, "cpus": 1},
                "first-update measured input bound or hard resources differ")
    elif protocol["execution_mode"] == FIT_ONLY_INITIAL:
        require(isinstance(campaign_budget, dict)
                and set(campaign_budget) == {
                    "fits", "families", "conditions", "workers", "cpu_per_worker",
                    "memory_bytes_per_worker", "shared_wall_seconds", "countdown_starts",
                    "retries_reset_budget", "budget_includes", "select_scoring_included"}
                and campaign_budget["fits"] == 10
                and campaign_budget["families"] == 2
                and campaign_budget["conditions"] == 5
                and type(campaign_budget["workers"]) is int
                and campaign_budget["workers"] >= 1
                and campaign_budget["cpu_per_worker"] == resources["cpus"]
                and campaign_budget["memory_bytes_per_worker"] == resources["memory_bytes"]
                and type(campaign_budget["shared_wall_seconds"]) is int
                and campaign_budget["shared_wall_seconds"] > 0
                and campaign_budget["countdown_starts"] ==
                    "FIRST_SCIENTIFIC_TRAINING_LAUNCH"
                and campaign_budget["retries_reset_budget"] is False
                and campaign_budget["budget_includes"] == [
                    "48_INITIAL_FIT_LOADS_PER_FIT", "96_ADAM_UPDATES_PER_FIT",
                    "48_FINAL_FIT_SCORES_PER_FIT"]
                and campaign_budget["select_scoring_included"] is False,
                "FIT_ONLY_INITIAL shared campaign budget differs")
    else:
        require(campaign_budget is None,
                "synthetic full protocol cannot claim the human shared budget")

    decision_sha256 = digest(decision_payload(protocol))
    _validate_evidence(protocol, decision_sha256)
    return protocol, decision_sha256


def validate_first_update_execution_budget(path, expected_sha, protocol_spec,
                                           contract_sha256, resolved_sha256, family,
                                           expected_condition=None):
    """Authorize a bounded stage without changing the scientific checkpoint identity."""
    budget = read_json(path, expected_sha)
    common = {"schema", "stage", "protocol", "contract", "resolved", "family",
              "condition", "seed", "workers", "cpu_per_worker",
              "memory_bytes_per_worker", "people_per_microbatch",
              "stop_after_steps", "checkpoint_only", "select_values_loaded",
              "evaluate_opened", "shared_wall_seconds", "retries_reset_budget",
              "scientific_ceiling_updates", "same_contract_required_for_continuation",
              "deadline_epoch_seconds", "accumulated_wall_seconds", "accumulated_cost"}
    require(budget.get("schema") == FIRST_UPDATE_EXECUTION_BUDGET_SCHEMA
            and budget.get("protocol") == protocol_spec
            and budget.get("contract") == contract_sha256
            and budget.get("resolved") == resolved_sha256
            and budget.get("family") == family in FAMILIES
            and budget.get("condition") in CONDITIONS
            and (expected_condition is None or
                 budget.get("condition") == expected_condition in CONDITIONS)
            and budget.get("seed") == 1701
            and budget.get("workers") == 1 and budget.get("cpu_per_worker") == 1
            and budget.get("memory_bytes_per_worker") == 8589934592
            and budget.get("people_per_microbatch") == 8
            and budget.get("select_values_loaded") is False
            and budget.get("evaluate_opened") is False
            and type(budget.get("shared_wall_seconds")) is int
            and budget["shared_wall_seconds"] > 0
            and budget.get("retries_reset_budget") is False
            and budget.get("scientific_ceiling_updates") == 96
            and budget.get("same_contract_required_for_continuation") is True
            and type(budget.get("deadline_epoch_seconds")) in (int, float)
            and math.isfinite(budget["deadline_epoch_seconds"])
            and budget["deadline_epoch_seconds"] > time.time()
            and type(budget.get("accumulated_wall_seconds")) in (int, float)
            and math.isfinite(budget["accumulated_wall_seconds"])
            and budget["accumulated_wall_seconds"] >= 0
            and isinstance(budget.get("accumulated_cost"), dict)
            and budget["accumulated_cost"].get("kind") in {
                "UNKNOWN_SHARED_VM_MARGINAL_COST", "ESTIMATED_ALLOCATED_REFERENCE"}
            and ((budget["accumulated_cost"]["kind"] == "UNKNOWN_SHARED_VM_MARGINAL_COST"
                  and budget["accumulated_cost"].get("usd") is None)
                 or (budget["accumulated_cost"]["kind"] == "ESTIMATED_ALLOCATED_REFERENCE"
                     and type(budget["accumulated_cost"].get("usd")) in (int, float)
                     and math.isfinite(budget["accumulated_cost"]["usd"])
                     and budget["accumulated_cost"]["usd"] >= 0)),
            "execution budget differs from the exact contract-bound stage")
    if budget["stage"] == "FIRST_RETAINED_UPDATE_MEASUREMENT":
        require(set(budget) == common | {"fits", "fit_scoring_performed", "reference_hourly_cost_usd"}
                and budget["fits"] == 1 and budget["stop_after_steps"] == 1
                and budget["checkpoint_only"] is True
                and budget["fit_scoring_performed"] is False
                and (budget["reference_hourly_cost_usd"] is None
                     or (type(budget["reference_hourly_cost_usd"]) in (int, float)
                         and math.isfinite(budget["reference_hourly_cost_usd"])
                         and budget["reference_hourly_cost_usd"] >= 0))
                and budget["shared_wall_seconds"] == 7200
                and budget["shared_wall_seconds"] <=
                    budget["deadline_epoch_seconds"] - time.time() + 1
                and budget["accumulated_wall_seconds"] == 0
                and budget["accumulated_cost"] in (
                    {"kind": "UNKNOWN_SHARED_VM_MARGINAL_COST", "usd": None},
                    {"kind": "ESTIMATED_ALLOCATED_REFERENCE", "usd": 0}),
                "first-update execution budget differs")
    elif budget["stage"] == "CONTINUE_SAME_CONTRACT_TO_SCIENTIFIC_CEILING":
        require(set(budget) == common | {"first_update_measurement", "fit_scoring_performed"}
                and budget["stop_after_steps"] is None
                and budget["checkpoint_only"] is False
                and budget["fit_scoring_performed"] is True,
                "continuation execution budget differs")
        measurement_spec = budget["first_update_measurement"]
        measurement = read_json(descriptor(measurement_spec, "first update measurement"),
                                measurement_spec["sha256"])
        receipt_spec = measurement.get("receipt")
        receipt = read_json(descriptor(receipt_spec, "first retained update receipt"),
                            receipt_spec["sha256"])
        elapsed = measurement.get("elapsed_wall_seconds")
        cost = measurement.get("cost")
        deadline = budget.get("deadline_epoch_seconds")
        require(measurement.get("schema") == "r02_lai_chr1_first_retained_update_v2"
                and measurement.get("contract_sha256") == contract_sha256
                and measurement.get("resolved_sha256") == resolved_sha256
                and type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0
                and isinstance(cost, dict)
                and cost.get("kind") in {"UNKNOWN_SHARED_VM_MARGINAL_COST",
                                         "ESTIMATED_ALLOCATED_REFERENCE"}
                and ((cost["kind"] == "UNKNOWN_SHARED_VM_MARGINAL_COST" and cost.get("usd") is None)
                     or (cost["kind"] == "ESTIMATED_ALLOCATED_REFERENCE"
                         and type(cost.get("usd")) in (int, float)
                         and math.isfinite(cost["usd"]) and cost["usd"] >= 0))
                and isinstance(measurement.get("cgroup"), dict)
                and type(measurement["cgroup"].get("memory_peak_bytes")) is int
                and measurement["cgroup"]["memory_peak_bytes"] > 0
                and receipt.get("status") == "FIT_CHECKPOINT_ONLY"
                and receipt.get("contract_sha256") == contract_sha256
                and receipt.get("resolved_sha256") == resolved_sha256
                and receipt.get("steps") == 1
                and receipt.get("frozen_training_ceiling_steps") == 96
                and receipt.get("fit_scoring_performed") is False
                and receipt.get("select_accessed") is False
                and budget["accumulated_wall_seconds"] >= elapsed
                and budget["accumulated_cost"]["kind"] == cost["kind"]
                and (cost["usd"] is None or budget["accumulated_cost"]["usd"] >= cost["usd"]),
                "continuation does not bind a live cumulative budget and authenticated first update")
    else:
        raise ValueError("unknown execution-budget stage")
    return budget


def validate_repair_execution_budget(path, expected_sha, protocol_spec,
                                     contract_sha256, resolved_sha256, family,
                                     expected_condition=None):
    """Admit one new 96-update FIT repair from zero using historical resources only."""
    budget = read_json(path, expected_sha)
    fields = {"schema", "stage", "protocol", "contract", "resolved", "family",
              "condition", "seed", "workers", "campaign_parallel_workers",
              "cpu_per_worker", "memory_bytes_per_worker", "people_per_microbatch",
              "stop_after_steps", "checkpoint_only", "fit_scoring_performed",
              "select_values_loaded", "evaluate_opened", "max_task_wall_seconds",
              "retries_reset_budget", "scientific_ceiling_updates", "starts_from_zero",
              "checkpoint_reuse", "current_contract_first_update_measurement",
              "resource_basis"}
    require(set(budget) == fields
            and budget.get("schema") == REPAIR_EXECUTION_BUDGET_SCHEMA
            and budget.get("stage") == "REPAIR_SAME_FIT_FULL_BUDGET"
            and budget.get("protocol") == protocol_spec
            and budget.get("contract") == contract_sha256
            and budget.get("resolved") == resolved_sha256
            and budget.get("family") == family in FAMILIES
            and budget.get("condition") in CONDITIONS
            and (expected_condition is None or
                 budget.get("condition") == expected_condition in CONDITIONS)
            and budget.get("seed") == 1701
            and budget.get("workers") == 1 and budget.get("campaign_parallel_workers") == 2
            and budget.get("cpu_per_worker") == 1
            and budget.get("memory_bytes_per_worker") == 8589934592
            and budget.get("people_per_microbatch") == 8
            and budget.get("stop_after_steps") is None
            and budget.get("checkpoint_only") is False
            and budget.get("fit_scoring_performed") is True
            and budget.get("select_values_loaded") is False
            and budget.get("evaluate_opened") is False
            and budget.get("max_task_wall_seconds") == 43200
            and budget.get("retries_reset_budget") is False
            and budget.get("scientific_ceiling_updates") == 96
            and budget.get("starts_from_zero") is True
            and budget.get("checkpoint_reuse") is False
            and budget.get("current_contract_first_update_measurement") is None,
            "repair execution budget differs from the exact fresh 96-update FIT stage")
    basis_spec = budget["resource_basis"]
    basis = read_json(descriptor(basis_spec, "historical resource basis"), basis_spec["sha256"])
    require(set(basis) == {"schema", "evidence_role", "family", "historical_contract",
                           "historical_receipt", "trace", "trace_task"}
            and basis.get("schema") == "r02_lai_repair_resource_basis_v1"
            and basis.get("evidence_role") ==
                "HISTORICAL_RESOURCE_ENVELOPE_ONLY_DIFFERENT_CONTRACT"
            and basis.get("family") == family,
            "repair resource evidence is not explicitly historical and family-matched")
    old_contract_spec = basis["historical_contract"]
    old_contract = read_json(descriptor(old_contract_spec, "historical contract"),
                             old_contract_spec["sha256"])
    old_receipt_spec = basis["historical_receipt"]
    old_receipt = read_json(descriptor(old_receipt_spec, "historical receipt"),
                            old_receipt_spec["sha256"])
    require(old_contract_spec["sha256"] != contract_sha256
            and old_contract.get("objective") == "lai" and old_contract.get("steps") == 96
            and old_receipt.get("contract_sha256") == old_contract_spec["sha256"]
            and old_receipt.get("resolved_sha256") == resolved_sha256
            and old_receipt.get("status") == "FIT_CHECKPOINT"
            and old_receipt.get("steps") == 96 and old_receipt.get("select_accessed") is False,
            "resource basis is not a completed different-contract 96-update FIT")
    trace_spec = basis["trace"]
    trace_path = Path(descriptor(trace_spec, "historical Nextflow trace"))
    require(sha(trace_path) == trace_spec["sha256"], "historical trace SHA-256 differs")
    task = basis["trace_task"]
    require(set(task) == {"name", "duration_seconds", "peak_rss_bytes"}
            and type(task["duration_seconds"]) is int and 0 < task["duration_seconds"] < 43200
            and type(task["peak_rss_bytes"]) is int
            and 0 < task["peak_rss_bytes"] < 8589934592,
            "historical task resource values exceed the repair envelope")
    rows = trace_path.read_text().splitlines()
    require(rows and "name" in rows[0].split("\t"), "historical trace header differs")
    header = rows[0].split("\t")
    observed = [dict(zip(header, row.split("\t"))) for row in rows[1:]
                if len(row.split("\t")) == len(header)]
    matched = [row for row in observed if row.get("name") == task["name"]]
    require(len(matched) == 1 and matched[0].get("status") == "COMPLETED"
            and matched[0].get("peak_rss") not in {None, "-"},
            "historical completed task is absent from the pinned trace")
    return budget


def validate_reconstruction_execution_limit(path, expected_sha, contract_sha256,
                                            family, condition):
    """Preserve v18's measured source, reconstructed, and combined limits."""
    value = read_json(path, expected_sha)
    required = {"schema", "status", "metadata_scan", "contract_sha256_by_id",
                "superseded_estimate", "max_factorized_source_bytes",
                "max_reconstructed_join_bytes", "max_source_plus_reconstructed_bytes",
                "memory_bytes", "derivation", "select_values_loaded", "evaluate_opened"}
    require(set(value) == required and value.get("schema") == RECONSTRUCTION_EXECUTION_LIMIT_SCHEMA
            and value.get("status") == "TECHNICAL_LIMITS_FROM_ALL_480_RECONSTRUCTED_FIT_LINK_HEADERS"
            and value.get("memory_bytes") == 8589934592
            and value.get("derivation") == "maximum over each link of factorized_source_bytes + reconstructed_join_bytes"
            and value["max_source_plus_reconstructed_bytes"] < value["memory_bytes"]
            and value.get("select_values_loaded") is False and value.get("evaluate_opened") is False,
            "reconstruction execution limit differs")
    scan_spec = value["metadata_scan"]
    scan = read_json(descriptor(scan_spec, "factorized-source metadata scan"), scan_spec["sha256"])
    require(scan.get("status") == "NPZ_NPY_HEADERS_ONLY_NO_ARRAY_BODY_OR_GENOTYPE_READ"
            and scan.get("links_scanned") == 480
            and scan.get("global_maximum_reconstructed", {}).get("reconstructed_join_bytes") == value["max_reconstructed_join_bytes"]
            and scan.get("global_maximum_source_plus_reconstructed", {}).get("factorized_source_bytes") == value["max_factorized_source_bytes"]
            and scan.get("global_maximum_source_plus_reconstructed", {}).get("source_plus_reconstructed_bytes") == value["max_source_plus_reconstructed_bytes"],
            "reconstruction metadata evidence differs")
    require(value["contract_sha256_by_id"].get(family + "." + condition) == contract_sha256,
            "reconstruction limit is not bound to this exact contract")
    return {key: value[key] for key in ("max_factorized_source_bytes",
        "max_reconstructed_join_bytes", "max_source_plus_reconstructed_bytes")}


@lru_cache(maxsize=16)
def _validate_protocol_cached(path, expected_sha):
    return validate_protocol_value(read_json(path, expected_sha))


def validate_protocol(path, expected_sha):
    return _validate_protocol_cached(str(Path(path).resolve()), expected_sha)


def validate_resolved(protocol, resolved):
    from genomic_context_runtime import ContextConfig
    from r02_lai_cnn_config import validate_cnn_config
    config = ContextConfig.from_recipe(resolved)
    domain = protocol["recipe_domain"]
    require(config.family in domain["families"]
            and float(config.common_radius_cm) in domain["radii_cm"]
            and float(config.rare_radius_cm) == float(config.common_radius_cm)
            and config.width in domain["widths"] and config.depth == domain["depth"]
            and float(config.dropout) == domain["dropout"]
            and domain["learning_rate"]["low"] <= config.learning_rate <=
                domain["learning_rate"]["high"]
            and resolved.get("optimizer", "Adam") == "Adam",
            "resolved model is outside the authenticated CW2/EC domain")
    if config.family == "lai_cnn":
        require(config.heads is None, "CNN cannot declare attention heads")
        cnn = validate_cnn_config(resolved.get("cnn_query_config"))
        key = format(float(config.common_radius_cm), "g")
        require(cnn is not None and cnn["kernel_sizes"] == domain["cnn_kernels"][key],
                "CNN kernels do not match the authenticated radius")
    else:
        require(config.heads == domain["attention_heads"]
                and "cnn_query_config" not in resolved,
                "attention heads or CNN-only configuration differ")
    return config


def validate_case_reference_identity(contract, case_id, observed):
    """Match one prepared realization to its five source/reference hashes.

    The newer regional campaign has distinct event catalogues and donor-role
    identities for each realization.  A single representative digest would
    conceal those differences.  Legacy single-realization contracts remain
    valid through ``reference_identity``.
    """
    require(isinstance(observed, dict) and set(observed) == REFERENCE_IDENTITY_FIELDS
            and all(HASH.fullmatch(value or "") for value in observed.values()),
            "predictor request has invalid source/reference identity")
    by_case = contract.get("reference_identity_by_case")
    if by_case is not None:
        require(isinstance(by_case, dict) and case_id in by_case
                and by_case[case_id] == observed,
                "case-specific source/reference identity differs")
    else:
        require(contract.get("reference_identity") == observed,
                "legacy source/reference identity differs")
    return observed


def canonical_planned_predictor(entry, planned_predictors, planned_by_request):
    """Resolve a factorized link to the exact PLAN predictor descriptor.

    ``task_id`` names the case-level source and is intentionally shared by
    families, radii and regions.  The nested ``regional_request`` descriptor
    identifies the unique predictor that COMPLETE v2 authenticated.
    """
    request = read_json(entry["path"], entry["sha256"])
    regional_spec = request.get("regional_request")
    regional_path = descriptor(regional_spec, "regional predictor request")
    key = (str(regional_path.resolve()), regional_spec["sha256"])
    predictor_id = planned_by_request.get(key)
    require(request.get("schema") == "r02_lai_regional_truth_join_request_v2"
            and predictor_id in planned_predictors
            and planned_predictors[predictor_id] == regional_spec,
            "factorized link does not identify one authenticated PLAN predictor")
    predictor = read_json(regional_path, regional_spec["sha256"])
    require(request.get("task_id") == predictor.get("task_id")
            and request.get("case_id") == predictor.get("case_id")
            and request.get("role") == predictor.get("role"),
            "factorized link and regional predictor identities differ")
    return predictor_id, predictor


def _validate_training_receipt(entry, receipt, contract, resolved, stage, action):
    require(action == "TRAIN" and stage in {"FIT", "SELECT"}
            and entry.get("format") == "LAI_FACTORIZED_TRAINING_LINK_V1",
            "training link is valid only for FIT/SELECT TRAIN")
    required = {"schema", "status", "request_sha256", "source_prepare_receipt",
                "protocol", "protocol_decision_sha256", "resolved_configuration_sha256",
                "direction", "chromosome", "role", "condition", "case_id", "region_id",
                "predictor_id", "arrays",
                "reconstructed_arrays_sha256", "evaluate_opened"}
    require(set(receipt) == required and receipt["schema"] == TRAIN_LINK_SCHEMA
            and receipt["status"] == TRAIN_LINK_STATUS
            and receipt["request_sha256"] == entry["sha256"]
            and receipt["role"] == stage and receipt["direction"] == "FOLD_A_TO_FOLD_B"
            and str(receipt["chromosome"]) ==
                str(contract.get("lai_chromosome", "22"))
            and receipt["evaluate_opened"] is False,
            "training-link receipt identity differs")
    protocol_spec = contract.get("lai_training_protocol")
    require(protocol_spec == receipt["protocol"], "contract and training link use different protocols")
    protocol, decision_sha256 = validate_protocol(
        descriptor(protocol_spec, "training protocol"), protocol_spec["sha256"])
    require(receipt["protocol_decision_sha256"] == decision_sha256
            and contract.get("information_condition") == receipt["condition"] in CONDITIONS,
            "training link condition or protocol decision differs")
    validate_resolved(protocol, resolved)
    _validate_runtime_only_resolved_amendment(
        receipt["resolved_configuration_sha256"], contract, resolved)
    require(contract.get("steps") == protocol["optimization"]["optimizer_updates"]
            and contract.get("threads") == protocol["resources"]["threads"]
            and contract.get("limits", {}).get("max_batch_bytes") ==
                protocol["resources"]["max_batch_bytes"],
            "contract budget/resources differ from authenticated protocol")
    return protocol


def _validate_runtime_only_resolved_amendment(prepared_digest, contract, resolved):
    """Authenticate the sole runtime-only EVENT mode added after PREPARE."""
    if prepared_digest == digest(resolved):
        return
    amendment = contract.get("runtime_only_resolved_amendment")
    required = {"schema", "added_field", "added_value", "source_resolved",
                "source_resolved_configuration_sha256", "runtime_resolved",
                "runtime_resolved_configuration_sha256",
                "prepared_receipts_unchanged"}
    require(isinstance(amendment, dict) and set(amendment) == required
            and amendment.get("schema") ==
                "r02_lai_runtime_only_resolved_amendment_v1"
            and amendment.get("added_field") == "lai_common_context_mode"
            and amendment.get("added_value") == "EVENT_QUERY_COMMON_KV_V1"
            and amendment.get("prepared_receipts_unchanged") is True,
            "runtime resolved amendment is absent or exceeds the admitted field")
    source = read_json(descriptor(amendment["source_resolved"], "source resolved"),
                       amendment["source_resolved"]["sha256"])
    runtime = read_json(descriptor(amendment["runtime_resolved"], "runtime resolved"),
                        amendment["runtime_resolved"]["sha256"])
    expected_runtime = dict(source)
    expected_runtime[amendment["added_field"]] = amendment["added_value"]
    require(runtime == expected_runtime and resolved == runtime
            and prepared_digest == digest(source) ==
                amendment["source_resolved_configuration_sha256"]
            and digest(runtime) ==
                amendment["runtime_resolved_configuration_sha256"],
            "prepared and runtime resolved recipes differ beyond the admitted mode")


def _authenticate_source_prepare(receipt, protocol, resolved=None):
    """Authenticate wrapper metadata without opening predictor arrays."""
    source_spec = receipt["source_prepare_receipt"]
    source = read_json(descriptor(source_spec, "source PREPARE receipt"),
                       source_spec["sha256"])
    require(source.get("schema") == "r02_lai_regional_truth_factorized_link_receipt_v1"
            and source.get("status") ==
                "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
            and source.get("factorized_link") is True
            and source.get("truth_joined") is True
            and source.get("training_authorized") is False
            and source.get("selection_authorized") is False
            and source.get("evaluate_opened") is False,
            "source is not the authenticated PREPARE-only factorized link")
    bound = {
        "request_sha256": receipt["request_sha256"],
        "task_id": receipt["predictor_id"], "case_id": receipt["case_id"],
        "direction": receipt["direction"], "role": receipt["role"],
        "chromosome": receipt["chromosome"], "condition": receipt["condition"],
        "region_id": receipt["region_id"], "arrays": receipt["arrays"],
        "reconstructed_arrays_sha256": receipt["reconstructed_arrays_sha256"],
    }
    require(all(str(source.get(key)) == str(value) if key == "chromosome"
                else source.get(key) == value for key, value in bound.items()),
            "training wrapper and source PREPARE identity differ")
    # This marker exists only in deliberately constructed code-test fixtures.
    # Human PREPARE receipts cannot be relabelled as synthetic to skip review.
    require((source.get("synthetic_code_test") is True) ==
            (protocol["scope"] == "SYNTHETIC_CODE_TEST"),
            "synthetic protocol/source provenance differs")
    from r02_lai_query_burden import BOUNDARY, COUNT_NAMES, SCOPE
    require(source.get("lai_burden_scope") == SCOPE
            and source.get("lai_burden_boundary") == BOUNDARY
            and source.get("lai_burden_count_names") == list(COUNT_NAMES)
            and bool(HASH.fullmatch(source.get("event_burden_prefix_receipt_sha256") or "")),
            "source is not the complete v2 query-burden product")
    if resolved is not None:
        require(source.get("rare_radius_cm") == resolved.get("rare_radius_cm")
                and source.get("density_matched_status") ==
                    ("MATERIALIZED_FOR_THIS_RADIUS_PREPARE_ONLY" if
                     receipt["condition"] == "DENSITY_MATCHED" else
                     "SEPARATE_RADIUS_SPECIFIC_PRODUCT"),
                "source radius or radius-specific DENSITY product differs")
    return source


def validate_campaign_contract(contract):
    """Require the complete 3-case x 16-region FIT and SELECT unit for one condition."""
    protocol_spec = contract.get("lai_training_protocol")
    protocol, _ = validate_protocol(descriptor(protocol_spec, "training protocol"),
                                    protocol_spec["sha256"])
    require(contract.get("objective") == "lai"
            and contract.get("scope") == protocol["scope"]
            and str(contract.get("lai_chromosome", protocol["chromosome"])) ==
                str(protocol["chromosome"])
            and contract.get("information_condition") in CONDITIONS
            and set(contract.get("batches", {})) == {"FIT", "SELECT"}
            and contract.get("steps") == protocol["optimization"]["optimizer_updates"]
            and contract.get("threads") == protocol["resources"]["threads"]
            and contract.get("limits", {}).get("max_batch_bytes") ==
                protocol["resources"]["max_batch_bytes"],
            "campaign contract differs from protocol objective, condition, or resources")
    expected_cases = {"FIT": protocol["training_unit"]["fit_cases"],
                      "SELECT": protocol["training_unit"]["select_cases"]}
    regions_per_case = protocol["training_unit"]["regions_per_case"]
    planned_links = None
    planned_predictors = None
    planned_by_request = None
    if protocol["scope"] == "DEVELOPMENT":
        plan_spec = protocol["input_plan"]
        plan = read_json(descriptor(plan_spec, "factorized PREPARE plan"),
                         plan_spec["sha256"])
        chromosome = str(protocol["chromosome"])
        planned_links = {(row["predictor_id"], row["condition"]): row["link_id"]
                         for row in plan["links"]
                         if str(row["chromosome"]) == chromosome}
        planned_predictors = {row["predictor_id"]: row["request"]
                              for row in plan["predictors"]
                              if str(row["chromosome"]) == chromosome}
        planned_by_request = {
            (str(Path(spec["path"]).resolve()), spec["sha256"]): predictor_id
            for predictor_id, spec in planned_predictors.items()}
        require(len(planned_links) == 2880,
                "chromosome plan link identities are not unique")
        require(len(planned_predictors) == 576,
                "chromosome plan predictor identities are not unique")
        require(len(planned_by_request) == 576,
                "chromosome predictor request descriptors are not unique")
    observed_cases = set()
    for stage in ("FIT", "SELECT"):
        entries = contract["batches"][stage]
        require(len(entries) == expected_cases[stage] * regions_per_case,
                f"{stage} does not contain the complete case-region unit")
        by_case = {}
        source_sha256 = set()
        predictor_ids = set()
        for entry in entries:
            require(entry.get("format") == "LAI_FACTORIZED_TRAINING_LINK_V1"
                    and sha(entry["path"]) == entry["sha256"],
                    "campaign contains a non-training-link batch")
            receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
            require(receipt.get("schema") == TRAIN_LINK_SCHEMA
                    and receipt.get("status") == TRAIN_LINK_STATUS
                    and receipt.get("request_sha256") == entry["sha256"]
                    and receipt.get("protocol") == protocol_spec
                    and receipt.get("protocol_decision_sha256") ==
                        digest(decision_payload(protocol))
                    and receipt.get("role") == stage
                    and receipt.get("condition") == contract["information_condition"]
                    and receipt.get("direction") == "FOLD_A_TO_FOLD_B"
                    and str(receipt.get("chromosome")) == str(protocol["chromosome"])
                    and receipt.get("evaluate_opened") is False,
                    "campaign batch receipt differs from stage, condition, or protocol")
            source = _authenticate_source_prepare(receipt, protocol)
            if planned_links is not None:
                predictor_id, predictor = canonical_planned_predictor(
                    entry, planned_predictors, planned_by_request)
                key = (predictor_id, source["condition"])
                require(key in planned_links
                        and Path(receipt["source_prepare_receipt"]["path"]).resolve() ==
                            (Path(protocol["input_output_root"]) / "links" /
                             planned_links[key] / "linked" / "receipt.json").resolve(),
                        "training source is absent from the authenticated COMPLETE-v2 inventory")
                require(predictor.get("task_id") == source["task_id"]
                        and predictor.get("case_id") == source["case_id"]
                        and predictor.get("role") == stage
                        and str(predictor.get("chromosome")) == str(protocol["chromosome"]),
                        "regional predictor request identity differs")
                validate_case_reference_identity(
                    contract, source["case_id"], predictor.get("reference_identity"))
                observed_cases.add(source["case_id"])
            from r02_lai_neural_input import parse_case
            scenario, chromosome, donor_fold, replicate = parse_case(receipt["case_id"])
            expected_fold = "FOLD_A" if stage == "FIT" else "FOLD_B"
            require(scenario == "PULSE_T10_ANCHOR"
                    and chromosome == str(protocol["chromosome"])
                    and donor_fold == expected_fold and replicate in {1, 2, 3}
                    and re.fullmatch(r"region-(?:0[0-9]|1[0-5])", receipt["region_id"]),
                    f"{stage} case, realization, fold, or region differs")
            by_case.setdefault(receipt["case_id"], set()).add(receipt["region_id"])
            source_sha256.add(receipt["source_prepare_receipt"]["sha256"])
            predictor_ids.add(predictor_id if planned_links is not None
                              else source["task_id"])
        expected_regions = {f"region-{index:02d}" for index in range(regions_per_case)}
        require(len(by_case) == expected_cases[stage]
                and {parse_case(case)[3] for case in by_case} == {1, 2, 3}
                and all(regions == expected_regions for regions in by_case.values())
                and len(source_sha256) == len(entries)
                and len(predictor_ids) == len(entries),
                f"{stage} lacks exact, uniquely sourced 3-realization x 16-region coverage")
    if contract.get("reference_identity_by_case") is not None:
        require(set(contract["reference_identity_by_case"]) == observed_cases,
                "case-specific source/reference map omits or adds a realization")
    return protocol


def validate_loaded_training_link(data, entry, receipt, contract, resolved, stage, action):
    _validate_training_receipt(entry, receipt, contract, resolved, stage, action)
    from r02_lai_regional_channel_bridge import array_sha256
    require(receipt["arrays"] == sorted(data)
            and receipt["reconstructed_arrays_sha256"] ==
                {name: array_sha256(value, __import__("numpy")) for name, value in data.items()},
            "loaded training arrays differ from authenticated PREPARE reconstruction")


def load_training_link(entry, contract, resolved, stage, *, max_reconstruction_bytes):
    receipt = read_json(entry["receipt_path"], entry["receipt_sha256"])
    protocol = _validate_training_receipt(entry, receipt, contract, resolved, stage, "TRAIN")
    _authenticate_source_prepare(receipt, protocol, resolved)
    source = receipt["source_prepare_receipt"]
    source_path = descriptor(source, "source PREPARE receipt")
    from r02_lai_neural_input import load_factorized_link
    return load_factorized_link(entry["path"], entry["sha256"], source_path,
                                source["sha256"], reconstruction_limits=max_reconstruction_bytes,
                                action="PREPARE")


def create_training_link(request, request_sha256, source_receipt, source_receipt_sha256,
                         protocol_path, protocol_sha256, resolved_path, resolved_sha256,
                         stage, condition, output):
    require(stage in {"FIT", "SELECT"} and condition in CONDITIONS,
            "unknown development role or condition")
    require(sha(request) == request_sha256, "factorized request differs")
    protocol, decision_sha256 = validate_protocol(protocol_path, protocol_sha256)
    chromosome = str(protocol["chromosome"])
    source = read_json(source_receipt, source_receipt_sha256)
    require(source.get("schema") == "r02_lai_regional_truth_factorized_link_receipt_v1"
            and source.get("status") ==
                "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
            and source.get("request_sha256") == request_sha256
            and source.get("role") == stage and source.get("condition") == condition
            and source.get("direction") == "FOLD_A_TO_FOLD_B"
            and str(source.get("chromosome")) == chromosome
            and source.get("evaluate_opened") is False,
            "source PREPARE receipt does not match requested training unit")
    resolved = read_json(resolved_path, resolved_sha256)
    validate_resolved(protocol, resolved)
    receipt = {
        "schema": TRAIN_LINK_SCHEMA, "status": TRAIN_LINK_STATUS,
        "request_sha256": request_sha256,
        "source_prepare_receipt": {"path": str(Path(source_receipt).resolve()),
                                   "sha256": source_receipt_sha256},
        "protocol": {"path": str(Path(protocol_path).resolve()), "sha256": protocol_sha256},
        "protocol_decision_sha256": decision_sha256,
        "resolved_configuration_sha256": digest(resolved),
        "direction": "FOLD_A_TO_FOLD_B", "chromosome": chromosome, "role": stage,
        "condition": condition, "case_id": source["case_id"],
        "region_id": source["region_id"], "predictor_id": source["task_id"],
        "arrays": source["arrays"],
        "reconstructed_arrays_sha256": source["reconstructed_arrays_sha256"],
        "evaluate_opened": False,
    }
    output = Path(output)
    require(not output.exists() and output.parent.is_dir(), "fresh receipt path required")
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def execute(contract_path, contract_sha256, resolved_path, resolved_sha256, condition, seed,
            output, checkpoint_dir=None, *, fit_only=False, stop_after_steps=None,
            lai_people_per_microbatch=None, checkpoint_only=False,
            execution_budget=None, execution_budget_sha256=None,
            reconstruction_limit=None, reconstruction_limit_sha256=None):
    from r02_context_train import read_json as read_pinned, train
    contract = read_pinned(contract_path, contract_sha256)
    resolved = read_pinned(resolved_path, resolved_sha256)
    protocol_spec = contract.get("lai_training_protocol")
    protocol, _ = validate_protocol(descriptor(protocol_spec, "training protocol"),
                                    protocol_spec["sha256"])
    require(protocol["execution_mode"] != FIT_ONLY_INITIAL or fit_only,
            "FIT_ONLY_INITIAL protocol rejects SELECT scoring")
    validate_resolved(protocol, resolved)
    validate_campaign_contract(contract)
    require(contract.get("objective") == "lai"
            and contract.get("information_condition") == condition in CONDITIONS,
            "contract objective or information condition differs")
    allowed = protocol["optimization"]["training_seeds"]
    legacy = contract["limits"]["max_batch_bytes"]
    max_reconstruction_bytes = {"max_factorized_source_bytes": legacy,
        "max_reconstructed_join_bytes": legacy,
        "max_source_plus_reconstructed_bytes": 2 * legacy}
    if protocol["schema"] == FIRST_UPDATE_PROTOCOL_SCHEMA:
        require(execution_budget is not None and execution_budget_sha256 is not None,
                "first-update execution requires a separate exact technical budget")
        budget_value = read_json(execution_budget, execution_budget_sha256)
        if budget_value.get("schema") == REPAIR_EXECUTION_BUDGET_SCHEMA:
            stage = validate_repair_execution_budget(
                execution_budget, execution_budget_sha256, protocol_spec,
                contract_sha256, resolved_sha256, resolved.get("family"), condition)
        else:
            stage = validate_first_update_execution_budget(
                execution_budget, execution_budget_sha256, protocol_spec,
                contract_sha256, resolved_sha256, resolved.get("family"), condition)
        require(reconstruction_limit is not None and reconstruction_limit_sha256 is not None,
                "factorized execution requires an authenticated three-part reconstruction limit")
        max_reconstruction_bytes = validate_reconstruction_execution_limit(
            reconstruction_limit, reconstruction_limit_sha256, contract_sha256,
            resolved.get("family"), condition)
        require(condition == stage["condition"] and seed == stage["seed"]
                and fit_only and stop_after_steps == stage["stop_after_steps"]
                and lai_people_per_microbatch == stage["people_per_microbatch"]
                and checkpoint_only is stage["checkpoint_only"],
                "execution exceeds the exact contract-bound stage budget")
    if not fit_only:
        allowed = sorted(set(allowed + protocol["optimization"]["stability_seeds"]))
    require(seed in allowed, "seed is outside the authenticated campaign phase")
    return train(contract_path, contract_sha256, resolved_path, resolved_sha256,
                 condition, seed, output, checkpoint_dir=checkpoint_dir,
                 checkpoint_every_steps=protocol["optimization"]["checkpoint_every_steps"],
                 adam_state_precision=protocol["optimization"]["adam_state_precision"],
                 fit_only=fit_only, stop_after_steps=stop_after_steps,
                 lai_people_per_microbatch=lai_people_per_microbatch,
                 checkpoint_only=checkpoint_only,
                 max_reconstruction_bytes=max_reconstruction_bytes)


def select(manifest_path, manifest_sha256, output):
    manifest = read_json(manifest_path, manifest_sha256)
    require(set(manifest) == {"schema", "protocol", "recipes", "evaluate_opened"}
            and manifest["schema"] == SELECT_MANIFEST_SCHEMA
            and manifest["evaluate_opened"] is False,
            "selection manifest fields differ or EVALUATE is open")
    protocol_spec = manifest["protocol"]
    protocol, decision_sha256 = validate_protocol(
        descriptor(protocol_spec, "training protocol"), protocol_spec["sha256"])
    require(protocol["scope"] == "SYNTHETIC_CODE_TEST"
            and protocol["execution_mode"] == SYNTHETIC_FULL,
            "DEVELOPMENT selection remains closed until the LAI weighting and guard policy is adjudicated")
    policy = protocol["selection"]
    expected = set(policy["expected_recipe_ids"])
    require({row.get("recipe_id") for row in manifest["recipes"]} == expected
            and len(manifest["recipes"]) == len(expected),
            "selection requires the complete authenticated recipe family")
    rows = []
    for recipe in manifest["recipes"]:
        require(set(recipe) == {"recipe_id", "resolved_sha256", "results"}
                and set(recipe["results"]) == set(CONDITIONS),
                "recipe lacks one of the five paired conditions")
        condition_values = {}
        for condition in CONDITIONS:
            records = recipe["results"][condition]
            expected_seeds = protocol["optimization"]["training_seeds"]
            require(isinstance(records, list) and len(records) == len(expected_seeds),
                    "condition lacks the exact primary-search seed inventory")
            values = []
            seen_seeds = set()
            for item in records:
                require(isinstance(item, dict) and set(item) == {"receipt", "contract"},
                        "selection result must pin its receipt and training contract")
                receipt_spec, contract_spec = item["receipt"], item["contract"]
                receipt = read_json(descriptor(receipt_spec, "training result"),
                                    receipt_spec["sha256"])
                contract = read_json(descriptor(contract_spec, "training contract"),
                                     contract_spec["sha256"])
                validate_campaign_contract(contract)
                require(receipt.get("status") == "COMPLETE"
                        and receipt.get("score_role") == "SELECT"
                        and receipt.get("select_accessed") is True
                        and receipt.get("arm") == condition
                        and receipt.get("resolved_sha256") == recipe["resolved_sha256"]
                        and receipt.get("contract_sha256") == contract_spec["sha256"]
                        and receipt.get("scope") == contract.get("scope") == protocol["scope"]
                        and contract.get("lai_training_protocol") == protocol_spec
                        and contract.get("information_condition") == condition
                        and receipt.get("frozen_training_ceiling_steps") == contract["steps"]
                        and receipt.get("steps") == contract["steps"]
                        and receipt.get("seed") in expected_seeds
                        and receipt.get("metrics", {}).get("component_log_loss") is not None,
                        "only completed SELECT receipts for the exact condition/recipe are selectable")
                value = receipt["metrics"]["component_log_loss"]
                require(type(value) in (int, float) and math.isfinite(value),
                        "selection metric is nonfinite")
                values.append(float(value))
                seen_seeds.add(receipt["seed"])
            require(seen_seeds == set(expected_seeds),
                    "condition duplicates or omits a primary-search seed")
            condition_values[condition] = sum(values) / len(values)
        score = sum(policy["condition_weights"][condition] * condition_values[condition]
                    for condition in CONDITIONS)
        rows.append({"recipe_id": recipe["recipe_id"],
                     "resolved_sha256": recipe["resolved_sha256"],
                     "condition_log_loss": condition_values,
                     "weighted_complete_condition_log_loss": score})
    rows.sort(key=lambda row: (row["weighted_complete_condition_log_loss"], row["recipe_id"]))
    result = {"schema": SELECTION_SCHEMA, "protocol_sha256": protocol_spec["sha256"],
              "protocol_decision_sha256": decision_sha256,
              "scope": "DEVELOPMENT_SELECTION_NOT_INDEPENDENT_EVALUATION",
              "metric": "component_log_loss", "condition_weights": policy["condition_weights"],
              "tie_break": policy["tie_break"],
              "selected_recipe_id": rows[0]["recipe_id"], "recipes": rows,
              "evaluate_opened": False}
    output = Path(output)
    require(not output.exists() and output.parent.is_dir(), "fresh selection output required")
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wrap = commands.add_parser("wrap")
    for name in ("request", "request_sha256", "source_receipt", "source_receipt_sha256",
                 "protocol", "protocol_sha256", "resolved", "resolved_sha256", "stage",
                 "condition", "output"):
        wrap.add_argument("--" + name.replace("_", "-"), required=True)
    run = commands.add_parser("execute")
    for name in ("contract", "contract_sha256", "resolved", "resolved_sha256", "condition",
                 "seed", "output"):
        run.add_argument("--" + name.replace("_", "-"), required=True)
    run.add_argument("--checkpoint-dir"); run.add_argument("--fit-only", action="store_true")
    run.add_argument("--stop-after-steps", type=int)
    run.add_argument("--lai-people-per-microbatch", type=int)
    run.add_argument("--checkpoint-only", action="store_true")
    run.add_argument("--execution-budget"); run.add_argument("--execution-budget-sha256")
    run.add_argument("--reconstruction-limit"); run.add_argument("--reconstruction-limit-sha256")
    choose = commands.add_parser("select")
    choose.add_argument("--manifest", required=True); choose.add_argument("--manifest-sha256", required=True)
    choose.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "wrap":
        value = create_training_link(args.request, args.request_sha256, args.source_receipt,
            args.source_receipt_sha256, args.protocol, args.protocol_sha256, args.resolved,
            args.resolved_sha256, args.stage, args.condition, args.output)
    elif args.command == "execute":
        value = execute(args.contract, args.contract_sha256, args.resolved, args.resolved_sha256,
                        args.condition, int(args.seed), args.output, args.checkpoint_dir,
                        fit_only=args.fit_only, stop_after_steps=args.stop_after_steps,
                        lai_people_per_microbatch=args.lai_people_per_microbatch,
                        checkpoint_only=args.checkpoint_only,
                        execution_budget=args.execution_budget,
                        execution_budget_sha256=args.execution_budget_sha256,
                        reconstruction_limit=args.reconstruction_limit,
                        reconstruction_limit_sha256=args.reconstruction_limit_sha256)
    else:
        value = select(args.manifest, args.manifest_sha256, args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
