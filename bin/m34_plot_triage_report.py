#!/usr/bin/env python3
"""Summarize and plot one audited M34 adaptive-stage comparison table."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import m34_adaptive_sweep as adaptive


PRIMARY_METRIC = "boundary_F1_0.2cM"
PLAN_STAGE_SPECS = {
    "M34_TRIAGE_PLAN": {
        "sweep_stage": "triage",
        "aggregate_status": "PASS_EXACT_TRIAGE_GRID_F0_RD_RE",
        "report_stage": "M34_TRIAGE_REPORT",
        "report_status": "PASS_REPORT_RENDERED",
        "label": "Triaje exploratorio",
    },
    "M34_LOCAL_EXPANSION_PLAN": {
        "sweep_stage": "local_expansion",
        "aggregate_status": "PASS_EXACT_LOCAL_EXPANSION_GRID_F0_RD_RE",
        "report_stage": "M34_LOCAL_EXPANSION_REPORT",
        "report_status": "PASS_LOCAL_EXPANSION_REPORT_RENDERED",
        "label": "Expansión local exploratoria",
    },
    "M34_RADIUS_SENSITIVITY_PLAN": {
        "sweep_stage": "radius_sensitivity",
        "aggregate_status": "PASS_EXACT_RADIUS_SENSITIVITY_GRID_F0_RD_RE",
        "report_stage": "M34_RADIUS_SENSITIVITY_REPORT",
        "report_status": "PASS_RADIUS_SENSITIVITY_REPORT_RENDERED",
        "label": "Sensibilidad exploratoria al radio",
    },
}
REQUIRED_COLUMNS = {
    "family", "config_id", "seed", "root", "radius_cM", "sweep_stage",
    "maximum_updates", "metric", "F0", "RD", "RE", "RE_minus_RD",
    "RE_minus_F0",
}


class ReportError(ValueError):
    """Raised when the comparison table cannot support an M34 report."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ReportError(f"non-finite JSON constant in {path}: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read JSON document: {path}") from error
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{label} must be numeric") from error
    require(math.isfinite(result), f"{label} must be finite")
    return result


def read_contract(path: Path) -> dict[str, Any]:
    contract = strict_json(path)
    require(contract.get("experiment_id") == "M34_NAM_ADAPTIVE_MODEL_SWEEP",
            "unexpected adaptive sweep contract")
    stages = contract.get("stages")
    metrics = contract.get("metrics")
    selection = contract.get("selection")
    families = contract.get("families")
    require(isinstance(stages, dict) and isinstance(metrics, dict) and
            isinstance(selection, dict) and isinstance(families, dict),
            "contract is missing reporting members")
    require(metrics.get("primary") == PRIMARY_METRIC,
            "contract primary metric differs from the report")
    sensitivities = metrics.get("sensitivities")
    guardrails = metrics.get("guardrails")
    thresholds = selection.get("maximum_guardrail_worsening")
    require(isinstance(sensitivities, list) and len(sensitivities) > 0 and
            len(sensitivities) == len(set(sensitivities)) and
            isinstance(guardrails, list) and len(guardrails) > 0 and
            len(guardrails) == len(set(guardrails)) and
            isinstance(thresholds, dict) and set(guardrails) == set(thresholds),
            "contract metrics, guardrails or thresholds differ")
    require(not ({PRIMARY_METRIC, *sensitivities} & set(guardrails)),
            "contract metric roles overlap")
    require(all(finite_float(thresholds[name], f"threshold/{name}") > 0.0
                for name in guardrails),
            "guardrail thresholds must be positive")
    return contract


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(reader.fieldnames is not None and
                    REQUIRED_COLUMNS.issubset(reader.fieldnames),
                    "comparison table columns differ")
            rows = list(reader)
    except OSError as error:
        raise ReportError(f"cannot read comparison table: {path}") from error
    require(rows, "comparison table is empty")
    return rows


def triage_config_order(contract: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    ordered: list[tuple[str, str, int]] = []
    for family, specification in contract["families"].items():
        triage_ids = specification.get("triage_ids")
        configs = specification.get("configs")
        require(isinstance(triage_ids, list) and isinstance(configs, list),
                f"{family} configuration space is malformed")
        ranks = {
            str(config.get("id")): int(config.get("complexity_rank"))
            for config in configs
        }
        require(set(triage_ids).issubset(ranks),
                f"{family} triage IDs are absent from the declared space")
        ordered.extend((family, config_id, ranks[config_id])
                       for config_id in triage_ids)
    require(len(ordered) == len({(family, config) for family, config, _ in ordered}),
            "contract contains duplicate triage configurations")
    return ordered


def _declared_configs(
    contract: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[int, Mapping[str, Any]]]:
    declared: dict[tuple[str, str], tuple[int, Mapping[str, Any]]] = {}
    for family, specification in contract["families"].items():
        configs = specification.get("configs")
        require(isinstance(configs, list) and configs,
                f"{family} configuration space is malformed")
        ranks: set[int] = set()
        for config in configs:
            require(isinstance(config, Mapping),
                    f"{family} contains a malformed configuration")
            config_id = str(config.get("id"))
            rank = config.get("complexity_rank")
            require(type(rank) is int and rank >= 0,
                    f"{family}/{config_id} complexity rank is malformed")
            require((family, config_id) not in declared and rank not in ranks,
                    f"{family} contains a duplicate configuration or rank")
            declared[(family, config_id)] = (rank, config)
            ranks.add(rank)
    return declared


def _radius_plan_from_source(
    plan: Mapping[str, Any], contract: Mapping[str, Any],
    plan_source_metrics: Mapping[str, Any] | None,
) -> None:
    """Bind a radius plan to the accumulated metrics that selected it."""
    require(plan_source_metrics is not None,
            "radius sensitivity requires plan-source metrics")
    try:
        pairs = adaptive.load_metric_pairs(dict(contract), dict(plan_source_metrics))
        expected = adaptive.radius_sensitivity_plan(dict(contract), pairs)
    except (adaptive.ContractError, KeyError, TypeError, ValueError) as error:
        raise ReportError("radius plan-source metrics are invalid") from error
    require(dict(plan) == expected,
            "radius plan differs from the contract and plan-source metrics")


def validate_plan(
    plan: Mapping[str, Any], contract: Mapping[str, Any],
    plan_source_metrics: Mapping[str, Any] | None = None,
) -> list[tuple[str, str, int, float]]:
    """Validate an adaptive plan and return its stage-specific report order."""
    plan_stage = plan.get("stage")
    require(plan_stage in PLAN_STAGE_SPECS,
            f"unsupported report plan stage: {plan_stage}")
    require(plan.get("status") == "PLAN_ONLY_NO_EXECUTION",
            "adaptive plan status differs")
    tasks = plan.get("tasks")
    require(isinstance(tasks, list) and tasks and len(tasks) % 2 == 0 and
            plan.get("task_count") == len(tasks),
            "adaptive plan task count differs")

    stage_spec = PLAN_STAGE_SPECS[str(plan_stage)]
    contract_stage = contract["stages"][stage_spec["sweep_stage"]]
    if plan_stage == "M34_RADIUS_SENSITIVITY_PLAN":
        _radius_plan_from_source(plan, contract, plan_source_metrics)
    declared = _declared_configs(contract)
    task_pairs: dict[tuple[str, str, float], dict[str, Mapping[str, Any]]] = {}
    first_seen: dict[str, list[str]] = {
        family: [] for family in contract["families"]
    }
    for task in tasks:
        require(isinstance(task, Mapping), "adaptive plan contains a malformed task")
        family, config_id = str(task.get("family")), str(task.get("config_id"))
        pair = (family, config_id)
        require(pair in declared,
                f"plan configuration is absent from the contract: {pair}")
        radius = finite_float(task.get("radius_cM"), f"{pair}/radius_cM")
        identity = (family, config_id, radius)
        arm = task.get("arm")
        require(arm in {"RD", "RE"} and
                arm not in task_pairs.setdefault(identity, {}),
                f"plan arms are duplicated or malformed for {identity}")
        task_pairs[identity][str(arm)] = task
        if config_id not in first_seen[family]:
            first_seen[family].append(config_id)

        require(task.get("seed") == contract_stage["seed"] and
                task.get("rotation") == contract_stage["rotation"] and
                task.get("sweep_stage") == stage_spec["sweep_stage"] and
                task.get("maximum_updates") == contract_stage["maximum_updates"],
                f"plan task identity differs from the contract for {identity}/{arm}")
        if plan_stage == "M34_RADIUS_SENSITIVITY_PLAN":
            allowed_radii = {
                finite_float(value, "radius_sensitivity/new_radius_cM")
                for value in plan.get("new_radii_cM", [])
            }
            require(radius in allowed_radii,
                    f"plan radius differs from the contract for {identity}/{arm}")
        else:
            require(radius == finite_float(contract_stage["radius_cM"],
                                           f"{stage_spec['sweep_stage']}/radius_cM"),
                    f"plan radius differs from the contract for {identity}/{arm}")
        _, config = declared[pair]
        training = dict(contract["training"])
        overrides = config.get("training_overrides", {})
        require(isinstance(overrides, Mapping),
                f"training overrides are malformed for {pair}")
        training.update(overrides)
        require(finite_float(task.get("learning_rate"),
                             f"{pair}/{arm}/learning_rate") ==
                finite_float(training["learning_rate"], "training/learning_rate") and
                finite_float(task.get("weight_decay"),
                             f"{identity}/{arm}/weight_decay") ==
                finite_float(training["weight_decay"], "training/weight_decay"),
                f"plan optimizer settings differ from the contract for {identity}/{arm}")

    require(all(set(arms) == {"RD", "RE"} for arms in task_pairs.values()),
            "adaptive plan contains an incomplete RD/RE pair")

    if plan_stage == "M34_TRIAGE_PLAN":
        radius = finite_float(contract_stage["radius_cM"], "triage/radius_cM")
        expected = [(family, config_id, rank, radius)
                    for family, config_id, rank in triage_config_order(contract)]
        require(set(task_pairs) == {(family, config_id, radius_value)
                                    for family, config_id, _, radius_value in expected},
                "triage plan configuration grid differs from the contract")
        return expected

    if plan_stage == "M34_RADIUS_SENSITIVITY_PLAN":
        selected = plan.get("selected_architectures")
        new_radii = plan.get("new_radii_cM")
        require(isinstance(selected, list) and selected and
                isinstance(new_radii, list) and new_radii,
                "radius sensitivity selection is malformed")
        ordered = []
        for candidate in selected:
            require(isinstance(candidate, Mapping),
                    "radius sensitivity candidate is malformed")
            family = str(candidate.get("family"))
            config_id = str(candidate.get("config_id"))
            require((family, config_id) in declared,
                    f"radius candidate is absent from the contract: {family}/{config_id}")
            rank = declared[(family, config_id)][0]
            ordered.extend((family, config_id, rank,
                            finite_float(radius, "radius_sensitivity/new_radius_cM"))
                           for radius in new_radii)
        require(set(task_pairs) == {
            (family, config_id, radius)
            for family, config_id, _rank, radius in ordered
        }, "radius sensitivity task grid differs from its selection")
        return ordered

    families = list(contract["families"])
    anchors = plan.get("anchor_config_ids_by_family")
    selected = plan.get("selected_config_ids_by_family")
    medium = plan.get("medium_budget_config_ids_by_family")
    require(all(isinstance(member, Mapping) and set(member) == set(families)
                for member in (anchors, selected, medium)),
            "local expansion configuration maps differ from the contract")
    stage = contract["stages"]["local_expansion"]
    radius = finite_float(stage["radius_cM"], "local_expansion/radius_cM")
    ordered: list[tuple[str, str, int, float]] = []
    declared_pairs: set[tuple[str, str]] = set()
    for family in families:
        family_anchors = anchors[family]
        family_selected = selected[family]
        family_medium = medium[family]
        require(isinstance(family_anchors, list) and
                len(family_anchors) == stage["anchor_count_per_family"] and
                isinstance(family_selected, list) and
                len(family_selected) == stage["maximum_new_configs_per_family"] and
                isinstance(family_medium, list) and
                family_medium == [*family_anchors, *family_selected] and
                len(family_medium) == len(set(family_medium)),
                f"local expansion declaration is malformed for {family}")
        require(all((family, config_id) in declared for config_id in family_medium),
                f"local expansion contains an undeclared config for {family}")
        require(first_seen[family] == family_medium,
                f"local expansion task order differs for {family}")
        anchor_ranks = [declared[(family, config_id)][0]
                        for config_id in family_anchors]
        for config_id in family_selected:
            rank = declared.get((family, config_id), (-999, {}))[0]
            require(any(abs(rank - anchor_rank) == 1 for anchor_rank in anchor_ranks),
                    f"local expansion neighbor is not adjacent for {family}/{config_id}")
        for config_id in family_medium:
            pair = (family, config_id)
            require(pair in declared, f"undeclared local expansion pair: {pair}")
            declared_pairs.add(pair)
            ordered.append((family, config_id, declared[pair][0], radius))
    require(set(task_pairs) == {(family, config_id, radius)
                                for family, config_id in declared_pairs},
            "local expansion task grid differs from its configuration maps")
    return ordered


def read_plan(
    path: Path, contract: Mapping[str, Any],
    plan_source_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = strict_json(path)
    validate_plan(plan, contract, plan_source_metrics)
    return plan


def validate_plan_source_metrics(
    payload: Mapping[str, Any], contract: Mapping[str, Any],
) -> dict[tuple[str, str, float, int, str, str], dict[str, dict[str, float]]]:
    """Validate the accumulated expansion payload reused by radius sensitivity."""
    expansion_pairs = len(contract["families"]) * (
        int(contract["stages"]["local_expansion"]["anchor_count_per_family"]) +
        int(contract["stages"]["local_expansion"]["maximum_new_configs_per_family"])
    )
    prior_pairs = len(triage_config_order(contract)) + expansion_pairs
    require(payload.get("schema_version") == "1.0.0" and
            payload.get("stage") == "M34_ADAPTIVE_STAGE_METRICS_PAYLOAD" and
            payload.get("source_plan_stage") == "M34_LOCAL_EXPANSION_PLAN" and
            payload.get("status") == "PASS_EXACT_LOCAL_EXPANSION_GRID" and
            payload.get("claim_level") == "exploratory" and
            payload.get("evaluation_split") == "VALID" and
            payload.get("test_opened") is False,
            "plan-source metrics identity differs")
    records = payload.get("records")
    require(isinstance(records, list) and
            payload.get("record_count") == 2 * prior_pairs and
            payload.get("pair_count") == prior_pairs and
            payload.get("stage_record_count") == 2 * expansion_pairs and
            payload.get("stage_pair_count") == expansion_pairs and
            len(records) == 2 * prior_pairs,
            "plan-source metrics dimensions differ")
    try:
        pairs = adaptive.load_metric_pairs(dict(contract), dict(payload))
    except (adaptive.ContractError, KeyError, TypeError, ValueError) as error:
        raise ReportError("plan-source metric records are invalid") from error
    require(len(pairs) == prior_pairs,
            "plan-source metrics contain an unexpected pair grid")
    return pairs


def report_order(
    plan: Mapping[str, Any], contract: Mapping[str, Any],
    plan_source_metrics: Mapping[str, Any] | None = None,
) -> list[tuple[str, str, int, float]]:
    """Return rows shown in the report, including the reused 0.2 cM radius."""
    stage_order = validate_plan(plan, contract, plan_source_metrics)
    if plan.get("stage") != "M34_RADIUS_SENSITIVITY_PLAN":
        return stage_order
    declared = _declared_configs(contract)
    radii = [finite_float(value, "radius_sensitivity/radius_cM")
             for value in contract["stages"]["radius_sensitivity"]["radii_cM"]]
    selected = [(str(row["family"]), str(row["config_id"]))
                for row in plan["selected_architectures"]]
    return [
        (family, config_id, declared[(family, config_id)][0], radius)
        for family, config_id in selected
        for radius in radii
    ]


def _baseline_by_metric(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    baseline: dict[str, float] = {}
    for row in rows:
        metric = str(row["metric"])
        value = finite_float(row["F0"], f"F0/{metric}")
        if metric in baseline:
            require(abs(baseline[metric] - value) <= 1e-12,
                    f"F0 differs across radius rows for {metric}")
        else:
            baseline[metric] = value
    return baseline


def reused_radius_rows(
    comparison_rows: Sequence[Mapping[str, str]], plan: Mapping[str, Any],
    contract: Mapping[str, Any], plan_source_metrics: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Materialize the frozen 0.2 cM rows from the accumulated expansion payload."""
    pairs = validate_plan_source_metrics(plan_source_metrics, contract)
    _radius_plan_from_source(plan, contract, plan_source_metrics)
    baseline = _baseline_by_metric(comparison_rows)
    required_metrics = [contract["metrics"]["primary"],
                        *contract["metrics"]["sensitivities"],
                        *contract["metrics"]["guardrails"]]
    require(set(baseline) == set(required_metrics),
            "comparison table does not define one F0 per declared metric")
    radius_stage = contract["stages"]["radius_sensitivity"]
    expansion_stage = contract["stages"]["local_expansion"]
    reused_radius = finite_float(
        plan.get("reused_radius_cM_from_local_expansion"),
        "radius_sensitivity/reused_radius_cM",
    )
    require(reused_radius == finite_float(
        radius_stage["reuse_radius_cM_from_local_expansion"],
        "contract/reused_radius_cM",
    ), "reused radius differs from the contract")
    rows: list[dict[str, str]] = []
    for candidate in plan["selected_architectures"]:
        family, config_id = str(candidate["family"]), str(candidate["config_id"])
        key = (family, config_id, reused_radius, expansion_stage["seed"],
               expansion_stage["rotation"], "local_expansion")
        require(key in pairs,
                f"reused local-expansion pair is missing: {key}")
        arms = pairs[key]
        for metric in required_metrics:
            f0, rd, re = baseline[metric], arms["RD"][metric], arms["RE"][metric]
            rows.append({
                "family": family, "config_id": config_id,
                "seed": str(expansion_stage["seed"]),
                "root": str(expansion_stage["rotation"]),
                "radius_cM": str(reused_radius),
                "sweep_stage": "local_expansion",
                "maximum_updates": str(expansion_stage["maximum_updates"]),
                "metric": metric, "F0": str(f0), "RD": str(rd), "RE": str(re),
                "RE_minus_RD": str(re - rd), "RE_minus_F0": str(re - f0),
            })
    return rows


def validate_aggregate_receipt(
    path: Path, comparison_path: Path, contract_path: Path, plan_path: Path,
    plan: Mapping[str, Any], contract: Mapping[str, Any],
    plan_source_metrics_path: Path | None = None,
    plan_source_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = strict_json(path)
    plan_stage = str(plan["stage"])
    specification = PLAN_STAGE_SPECS[plan_stage]
    stage_pairs = len(validate_plan(plan, contract, plan_source_metrics))
    if plan_stage == "M34_TRIAGE_PLAN":
        prior_pairs = 0
    elif plan_stage == "M34_LOCAL_EXPANSION_PLAN":
        prior_pairs = len(triage_config_order(contract))
    else:
        require(plan_source_metrics_path is not None and
                plan_source_metrics is not None,
                "radius aggregate receipt requires plan-source metrics")
        validate_plan_source_metrics(plan_source_metrics, contract)
        prior_pairs = int(plan_source_metrics["pair_count"])
    pair_count = prior_pairs + stage_pairs
    metric_count = 1 + len(contract["metrics"]["sensitivities"]) + len(
        contract["metrics"]["guardrails"]
    )
    require(receipt.get("stage") == "M34_AGGREGATE_ADAPTIVE_STAGE_METRICS" and
            receipt.get("source_plan_stage") == plan_stage and
            receipt.get("status") == specification["aggregate_status"] and
            receipt.get("claim_level") == "exploratory",
            "aggregate receipt identity differs")
    require(receipt.get("evaluation_split") == "VALID" and
            receipt.get("test_opened") is False,
            "aggregate receipt does not close TEST")
    input_hashes = receipt.get("input_sha256")
    output_hashes = receipt.get("output_sha256")
    require(isinstance(input_hashes, dict) and
            input_hashes.get("triage_plan") == sha256_file(plan_path) and
            input_hashes.get("adaptive_contract") == sha256_file(contract_path),
            "aggregate receipt is not bound to the adaptive plan")
    if plan_stage == "M34_RADIUS_SENSITIVITY_PLAN":
        require(input_hashes.get("plan_source_metrics") ==
                sha256_file(plan_source_metrics_path),
                "aggregate receipt is not bound to plan-source metrics")
    require(isinstance(output_hashes, dict) and
            output_hashes.get("table") == sha256_file(comparison_path),
            "aggregate receipt is not bound to the comparison table")
    require(receipt.get("record_count") == 2 * pair_count and
            receipt.get("pair_count") == pair_count and
            receipt.get("stage_record_count") == 2 * stage_pairs and
            receipt.get("stage_pair_count") == stage_pairs and
            receipt.get("long_table_row_count") == stage_pairs * metric_count,
            f"aggregate receipt dimensions differ for {plan_stage}")
    return receipt


def summarize(rows: Sequence[Mapping[str, str]],
              contract: Mapping[str, Any],
              plan: Mapping[str, Any],
              plan_source_metrics: Mapping[str, Any] | None = None,
              ) -> list[dict[str, Any]]:
    """Build one exact report row per family, configuration and radius."""
    guardrails = list(contract["metrics"]["guardrails"])
    sensitivities = list(contract["metrics"]["sensitivities"])
    thresholds = contract["selection"]["maximum_guardrail_worsening"]
    required_metrics = {PRIMARY_METRIC, *sensitivities, *guardrails}
    expected = report_order(plan, contract, plan_source_metrics)
    expected_pairs = {(family, config, radius)
                      for family, config, _rank, radius in expected}
    task_by_pair: dict[tuple[str, str, float], Mapping[str, Any]] = {
        (str(task["family"]), str(task["config_id"]),
         finite_float(task["radius_cM"], "plan/radius_cM")): task
        for task in plan["tasks"] if task["arm"] == "RD"
    }
    if plan.get("stage") == "M34_RADIUS_SENSITIVITY_PLAN":
        expansion = contract["stages"]["local_expansion"]
        reused = finite_float(plan["reused_radius_cM_from_local_expansion"],
                              "reused_radius_cM")
        for candidate in plan["selected_architectures"]:
            pair = (str(candidate["family"]), str(candidate["config_id"]), reused)
            task_by_pair[pair] = {
                "seed": expansion["seed"], "rotation": expansion["rotation"],
                "radius_cM": reused, "sweep_stage": "local_expansion",
                "maximum_updates": expansion["maximum_updates"],
            }
    by_config: dict[tuple[str, str, float], dict[str, Mapping[str, str]]] = {}
    metadata: dict[tuple[str, str, float], tuple[str, ...]] = {}

    for index, row in enumerate(rows, start=2):
        pair = (str(row["family"]), str(row["config_id"]),
                finite_float(row["radius_cM"], f"line_{index}/radius_cM"))
        require(pair in expected_pairs,
                f"configuration/radius absent from the adaptive plan on line {index}: {pair}")
        metric = str(row["metric"])
        require(metric in required_metrics,
                f"undeclared metric on line {index}: {metric}")
        require(metric not in by_config.setdefault(pair, {}),
                f"duplicate {metric} row for {pair}")
        by_config[pair][metric] = row
        identity = tuple(str(row[name]) for name in
                         ("seed", "root", "radius_cM", "sweep_stage",
                          "maximum_updates"))
        if pair in metadata:
            require(metadata[pair] == identity,
                    f"inconsistent task identity for {pair}")
        else:
            metadata[pair] = identity

    require(set(by_config) == expected_pairs,
            "comparison table contains an incomplete adaptive-stage grid")
    summaries: list[dict[str, Any]] = []
    for family, config_id, complexity_rank, expected_radius in expected:
        pair = (family, config_id, expected_radius)
        observed = by_config[pair]
        require(set(observed) == required_metrics,
                f"required metrics are incomplete for {pair}")
        seed, root, radius, stage, updates = metadata[pair]
        planned = task_by_pair[pair]
        require(int(seed) == planned["seed"] and root == planned["rotation"] and
                finite_float(radius, f"{pair}/radius_cM") ==
                finite_float(planned["radius_cM"], f"{pair}/planned_radius_cM") and
                stage == planned["sweep_stage"] and
                int(updates) == planned["maximum_updates"],
                f"comparison task identity differs from the plan for {pair}")
        primary = metric_values(observed[PRIMARY_METRIC], pair, PRIMARY_METRIC)
        summary: dict[str, Any] = {
            "family": family,
            "config_id": config_id,
            "complexity_rank": complexity_rank,
            "seed": int(seed),
            "root": root,
            "radius_cM": finite_float(radius, f"{pair}/radius_cM"),
            "sweep_stage": stage,
            "maximum_updates": int(updates),
            **prefixed_values("boundary_F1_0.2cM", primary),
        }
        failed = []
        for metric in guardrails:
            values = metric_values(observed[metric], pair, metric)
            threshold = finite_float(thresholds[metric], f"threshold/{metric}")
            passed = values["RE_minus_RD"] <= threshold + 1e-15
            summary.update(prefixed_values(metric, values))
            summary[f"threshold_{metric}"] = threshold
            summary[f"pass_{metric}"] = passed
            if not passed:
                failed.append(metric)
        minimum_delta = finite_float(
            contract["selection"]["provisional_minimum_delta_F1"],
            "provisional_minimum_delta_F1",
        )
        summary["primary_delta_pass"] = primary["RE_minus_RD"] >= minimum_delta
        summary["beats_F0"] = primary["RE_minus_F0"] > 0.0
        summary["all_guardrails_pass"] = not failed
        summary["failed_guardrails"] = ";".join(failed)
        summaries.append(summary)
    return summaries


def metric_values(row: Mapping[str, str], pair: tuple[str, str, float],
                  metric: str) -> dict[str, float]:
    values = {
        name: finite_float(row[name], f"{pair}/{metric}/{name}")
        for name in ("F0", "RD", "RE", "RE_minus_RD", "RE_minus_F0")
    }
    tolerance = 64.0 * math.ulp(max(1.0, abs(values["RE"]), abs(values["RD"]),
                                    abs(values["F0"])))
    require(abs((values["RE"] - values["RD"]) - values["RE_minus_RD"]) <= tolerance,
            f"RE-RD arithmetic differs for {pair}/{metric}")
    require(abs((values["RE"] - values["F0"]) - values["RE_minus_F0"]) <= tolerance,
            f"RE-F0 arithmetic differs for {pair}/{metric}")
    return values


def prefixed_values(metric: str, values: Mapping[str, float]) -> dict[str, float]:
    return {f"{name}_{metric}": value for name, value in values.items()}


def summary_columns(contract: Mapping[str, Any]) -> list[str]:
    columns = [
        "family", "config_id", "complexity_rank", "seed", "root",
        "radius_cM", "sweep_stage", "maximum_updates",
    ]
    value_names = ("F0", "RD", "RE", "RE_minus_RD", "RE_minus_F0")
    columns.extend(f"{name}_{PRIMARY_METRIC}" for name in value_names)
    for metric in contract["metrics"]["guardrails"]:
        columns.extend(f"{name}_{metric}" for name in value_names)
        columns.extend((f"threshold_{metric}", f"pass_{metric}"))
    columns.extend(("primary_delta_pass", "beats_F0", "all_guardrails_pass",
                    "failed_guardrails"))
    return columns


def format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def summary_tsv(summaries: Sequence[Mapping[str, Any]],
                contract: Mapping[str, Any]) -> str:
    columns = summary_columns(contract)
    lines = ["\t".join(columns)]
    lines.extend("\t".join(format_cell(row[name]) for name in columns)
                 for row in summaries)
    return "\n".join(lines) + "\n"


def render_figure(summaries: Sequence[Mapping[str, Any]],
                  contract: Mapping[str, Any], plan: Mapping[str, Any], png_path: Path,
                  pdf_path: Path) -> None:
    """Render absolute performance, paired effects and guardrails together."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    guardrails = list(contract["metrics"]["guardrails"])
    show_radius = plan.get("stage") == "M34_RADIUS_SENSITIVITY_PLAN"
    labels = [
        f"{row['family']} | {row['config_id']} | r={row['radius_cM']:g} cM"
        if show_radius else f"{row['family']} | {row['config_id']}"
        for row in summaries
    ]
    y = np.arange(len(summaries))
    f0 = np.asarray([row[f"F0_{PRIMARY_METRIC}"] for row in summaries])
    rd = np.asarray([row[f"RD_{PRIMARY_METRIC}"] for row in summaries])
    re = np.asarray([row[f"RE_{PRIMARY_METRIC}"] for row in summaries])
    delta_rd = np.asarray([row[f"RE_minus_RD_{PRIMARY_METRIC}"]
                           for row in summaries])
    delta_f0 = np.asarray([row[f"RE_minus_F0_{PRIMARY_METRIC}"]
                           for row in summaries])
    guardrail_ratio = np.asarray([
        [row[f"RE_minus_RD_{metric}"] / row[f"threshold_{metric}"]
         for metric in guardrails]
        for row in summaries
    ])

    ink = "#26333D"
    muted = "#667681"
    grid = "#D9E0E5"
    f0_color = "#424B54"
    rd_color = "#D0832F"
    re_color = "#2C6E9B"
    fig = plt.figure(figsize=(18.0, 11.8), facecolor="white")
    layout = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.1), wspace=0.23)
    ax_absolute = fig.add_subplot(layout[0, 0])
    ax_delta = fig.add_subplot(layout[0, 1], sharey=ax_absolute)
    ax_guard = fig.add_subplot(layout[0, 2], sharey=ax_absolute)

    ax_absolute.scatter(rd, y, s=36, marker="o", facecolors="white",
                        edgecolors=rd_color, linewidths=1.5, label="RD")
    ax_absolute.scatter(re, y, s=36, marker="s", color=re_color, label="RE")
    for index in range(len(y)):
        ax_absolute.plot([rd[index], re[index]], [y[index], y[index]],
                         color=grid, linewidth=1.0, zorder=0)
    f0_reference = float(f0[0])
    require(np.allclose(f0, f0_reference, rtol=0.0, atol=1e-12),
            "F0 differs across adaptive-stage configurations")
    ax_absolute.axvline(f0_reference, color=f0_color, linestyle="--",
                        linewidth=1.4, label=f"F0 = {f0_reference:.4f}")
    ax_absolute.set_xlim(0.0, 1.0)
    ax_absolute.set_xlabel("F1 de bordes a 0,2 cM", color=ink)
    ax_absolute.set_yticks(y, labels, fontsize=7.4)
    ax_absolute.invert_yaxis()
    ax_absolute.set_title("A. Rendimiento absoluto", loc="left", color=ink,
                          fontweight="bold")
    ax_absolute.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    ax_delta.axvline(0.0, color=f0_color, linestyle="--", linewidth=1.0)
    ax_delta.scatter(delta_rd, y, s=34, marker="o", color=re_color,
                     label="RE − RD")
    ax_delta.scatter(delta_f0, y, s=34, marker="D", facecolors="white",
                     edgecolors=ink, linewidths=1.2, label="RE − F0")
    delta_extent = max(0.005, float(np.max(np.abs(
        np.concatenate((delta_rd, delta_f0))))) * 1.18)
    ax_delta.set_xlim(-delta_extent, delta_extent)
    ax_delta.set_xlabel("Diferencia en F1 (positivo = mejor)", color=ink)
    ax_delta.set_title("B. Diferencia al habilitar valores raros", loc="left", color=ink,
                       fontweight="bold")
    ax_delta.legend(frameon=False, fontsize=8, loc="lower left")
    ax_delta.tick_params(axis="y", labelleft=False)

    limit = max(1.25, float(np.max(np.abs(guardrail_ratio))))
    image = ax_guard.imshow(guardrail_ratio, aspect="auto", cmap="PuOr_r",
                            vmin=-limit, vmax=limit)
    short_names = {
        "macro_ancestry_dose_MAE": "MAE\nmacro",
        "NAM_truth_present_MAE": "MAE NAM\ncon verdad",
        "false_transitions_per_cM": "Transiciones\nfalsas/cM",
        "haplotype_Brier": "Brier\nhaplotipo",
    }
    ax_guard.set_xticks(range(len(guardrails)),
                        [short_names.get(name, name) for name in guardrails],
                        fontsize=7.6)
    ax_guard.tick_params(axis="y", labelleft=False)
    ax_guard.set_title("C. Controles de error", loc="left", color=ink,
                       fontweight="bold")
    for row_index in range(guardrail_ratio.shape[0]):
        for column_index in range(guardrail_ratio.shape[1]):
            ratio = guardrail_ratio[row_index, column_index]
            delta = summaries[row_index][f"RE_minus_RD_{guardrails[column_index]}"]
            marker = "!" if ratio > 1.0 else ""
            ax_guard.text(column_index, row_index, f"{delta:+.3g}{marker}",
                          ha="center", va="center", fontsize=6.4,
                          fontweight="bold" if marker else "normal",
                          color="white" if abs(ratio) > 0.58 * limit else ink)
    colorbar = fig.colorbar(image, ax=ax_guard, fraction=0.045, pad=0.03)
    colorbar.set_label("(RE − RD) / límite permitido", fontsize=8, color=ink)
    colorbar.ax.tick_params(labelsize=7)

    families = [row["family"] for row in summaries]
    for index in range(1, len(families)):
        if families[index] != families[index - 1]:
            for axis in (ax_absolute, ax_delta, ax_guard):
                axis.axhline(index - 0.5, color=grid, linewidth=0.9)
    for axis in (ax_absolute, ax_delta):
        axis.grid(axis="x", color=grid, linewidth=0.7)
        axis.set_axisbelow(True)
    for axis in (ax_absolute, ax_delta, ax_guard):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=muted)

    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.20, right=0.96)
    fig.suptitle("M34 chr22 — comparación exploratoria AFR/EUR/NAM",
                 x=0.055, y=0.965, ha="left", fontsize=17,
                 fontweight="bold", color=ink)
    plan_spec = PLAN_STAGE_SPECS[str(plan["stage"])]
    roots = {str(row["root"]) for row in summaries}
    seeds = {int(row["seed"]) for row in summaries}
    updates = {int(row["maximum_updates"]) for row in summaries}
    require(len(roots) == len(seeds) == len(updates) == 1,
            "report configurations do not share one plan identity")
    fig.text(
        0.055, 0.925,
        f"{plan_spec['label']} con una raíz ({next(iter(roots))}), una semilla "
        f"({next(iter(seeds))}) y {next(iter(updates))} actualizaciones; VALID "
        "separado de FIT. F0: FLARE; RD: valores raros desactivados, conservando "
        "loci y máscaras; RE: valores raros habilitados."
        + (" El radio 0,2 cM se reutiliza de la expansión local; los demás "
           "pertenecen a la sensibilidad." if show_radius else ""),
        ha="left", fontsize=10.2, color=muted,
    )
    minimum_delta = float(contract["selection"]["provisional_minimum_delta_F1"])
    fig.text(
        0.055, 0.025,
        f"El contrato exploratorio exige RE−RD ≥ {minimum_delta:.3f} y controles "
        "dentro de sus umbrales. Para superar F0 también se necesita RE−F0 > 0; "
        "aun así, la señal sigue siendo exploratoria. En los controles de error, "
        "valores menores son mejores; "
        "! indica que RE−RD supera el umbral preespecificado. TEST no se abrió.",
        ha="left", fontsize=9.0, color=ink,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight",
                facecolor="white", metadata={"Software": "M34 report"})
    fixed_date = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white",
                metadata={"Creator": "M34 report", "CreationDate": fixed_date,
                          "ModDate": fixed_date})
    plt.close(fig)


def write_artifacts(comparison_path: Path, contract_path: Path, plan_path: Path,
                    aggregate_receipt_path: Path,
                    summary_path: Path, png_path: Path, pdf_path: Path,
                    receipt_path: Path,
                    plan_source_metrics_path: Path | None = None) -> dict[str, Any]:
    outputs = (summary_path, png_path, pdf_path, receipt_path)
    require(len({path.resolve() for path in outputs}) == len(outputs),
            "report output paths must be distinct")
    require(not any(path.exists() for path in outputs),
            "refusing to overwrite report outputs")
    contract = read_contract(contract_path)
    plan_source_metrics = (
        strict_json(plan_source_metrics_path)
        if plan_source_metrics_path is not None else None
    )
    plan = read_plan(plan_path, contract, plan_source_metrics)
    validate_aggregate_receipt(
        aggregate_receipt_path, comparison_path, contract_path, plan_path,
        plan, contract, plan_source_metrics_path, plan_source_metrics,
    )
    comparison_rows = read_rows(comparison_path)
    combined_rows = list(comparison_rows)
    if plan["stage"] == "M34_RADIUS_SENSITIVITY_PLAN":
        require(plan_source_metrics is not None,
                "radius report requires plan-source metrics")
        combined_rows.extend(reused_radius_rows(
            comparison_rows, plan, contract, plan_source_metrics,
        ))
    summaries = summarize(combined_rows, contract, plan, plan_source_metrics)
    stage_pair_count = len(validate_plan(plan, contract, plan_source_metrics))
    plan_spec = PLAN_STAGE_SPECS[str(plan["stage"])]
    temporary = tuple(path.with_name(f".{path.stem}.tmp{path.suffix}")
                      for path in outputs)
    require(not any(path.exists() for path in temporary),
            "temporary report output already exists")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary[0].write_text(summary_tsv(summaries, contract), encoding="utf-8")
        render_figure(summaries, contract, plan, temporary[1], temporary[2])
        receipt = {
            "schema_version": "1.0.0",
            "stage": plan_spec["report_stage"],
            "source_plan_stage": plan["stage"],
            "sweep_stage": plan_spec["sweep_stage"],
            "status": plan_spec["report_status"],
            "claim_level": contract["scope"]["claim_level"],
            "evaluation_split": "VALID",
            "test_opened": False,
            "configuration_count": len(summaries),
            "stage_pair_count": stage_pair_count,
            "reused_pair_count": len(summaries) - stage_pair_count,
            "primary_metric": PRIMARY_METRIC,
            "input_sha256": {
                "comparison_table": sha256_file(comparison_path),
                "adaptive_contract": sha256_file(contract_path),
                "adaptive_plan": sha256_file(plan_path),
                "aggregate_receipt": sha256_file(aggregate_receipt_path),
                "plan_source_metrics": (
                    sha256_file(plan_source_metrics_path)
                    if plan_source_metrics_path is not None else None
                ),
            },
            "output_sha256": {
                "summary_table": sha256_file(temporary[0]),
                "figure_png": sha256_file(temporary[1]),
                "figure_pdf": sha256_file(temporary[2]),
            },
        }
        temporary[3].write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for source, target in zip(temporary, outputs):
            os.replace(source, target)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--aggregate-receipt", type=Path, required=True)
    parser.add_argument("--plan-source-metrics", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = write_artifacts(
        args.comparison, args.contract, args.plan, args.aggregate_receipt, args.summary,
        args.png, args.pdf, args.receipt, args.plan_source_metrics,
    )
    print(json.dumps({
        "status": receipt["status"],
        "configuration_count": receipt["configuration_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
