#!/usr/bin/env python3
"""Validate and plan the finite M34 adaptive model sweep.

The planner never trains a model.  It turns a frozen JSON contract and, after
triage, paired RD/RE metrics into deterministic task manifests.  No candidate
may be synthesized outside the contract after metrics have been observed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("RD", "RE")
F1_KEYS = ("boundary_F1_0.1cM", "boundary_F1_0.2cM", "boundary_F1_0.5cM")
GUARDRAIL_KEYS = (
    "macro_ancestry_dose_MAE",
    "NAM_truth_present_MAE",
    "false_transitions_per_cM",
    "haplotype_Brier",
)
METRIC_KEYS = F1_KEYS + GUARDRAIL_KEYS
EXPECTED_FAMILIES = (
    "local_linear",
    "residual_cnn_1d",
    "lainns_cnn_1d",
    "unet_1d",
    "bilstm",
    "tcn",
    "transformer_small",
)
MODEL_SPEC_FIELDS = (
    "hidden_dim",
    "depth",
    "kernel_size",
    "dilations",
    "dropout",
    "lstm_layers",
    "transformer_heads",
    "transformer_ff_dim",
    "transformer_max_tokens",
)


class ContractError(ValueError):
    """Raised when a contract or metric input violates the frozen design."""


def _reject_constant(value: str) -> None:
    raise ContractError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"metric must be numeric, not boolean: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"metric must be numeric: {label}") from error
    if not math.isfinite(number):
        raise ContractError(f"metric must be finite: {label}")
    return number


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("status") != "CONTRACT_ONLY_NO_REAL_RESULTS_NO_EXECUTION":
        raise ContractError("contract status does not close real execution")
    scope = contract.get("scope", {})
    if scope.get("ancestries") != ["AFR", "EUR", "NAM"]:
        raise ContractError("M34 ancestry order must be AFR/EUR/NAM")
    if scope.get("arms") != list(ARMS):
        raise ContractError("M34 requires paired RD/RE arms")
    if scope.get("rotation_field_semantics") != (
        "independent_mosaic_realization_root_not_donor_unit_rotation"
    ):
        raise ContractError("R0/R1/R2 must denote independent mosaic roots")

    stages = contract.get("stages", {})
    triage = stages.get("triage", {})
    expansion = stages.get("local_expansion", {})
    radius_sensitivity = stages.get("radius_sensitivity", {})
    finalists = stages.get("finalists", {})
    minimum = int(triage.get("minimum_evaluated_configs_per_family", 0))
    if minimum < 3:
        raise ContractError("at least three triage points per family are required")
    if int(expansion.get("anchor_count_per_family", 0)) < 1:
        raise ContractError("local expansion needs at least one anchor")
    if int(expansion.get("maximum_new_configs_per_family", 0)) < 1:
        raise ContractError("local expansion must be possible")
    if int(finalists.get("maximum_families", 0)) > 2:
        raise ContractError("at most two families may be promoted")
    if triage.get("radius_cM") != 0.2 or expansion.get("radius_cM") != 0.2:
        raise ContractError("architecture search must remain anchored at 0.2 cM")
    if radius_sensitivity.get("radii_cM") != [0.05, 0.1, 0.2, 0.5]:
        raise ContractError("radius sensitivity must retain the four M33 radii")
    if radius_sensitivity.get("reuse_radius_cM_from_local_expansion") != expansion.get(
        "radius_cM"
    ):
        raise ContractError("radius reuse must match the local-expansion radius")
    if int(radius_sensitivity.get("maximum_families", 0)) > 2:
        raise ContractError("radius sensitivity may receive at most two families")
    if len(set(finalists.get("seeds", []))) != 3:
        raise ContractError("finalists require exactly three unique seeds")
    if len(set(finalists.get("rotations", []))) != 3:
        raise ContractError("finalists require exactly three unique rotations")
    replication = stages.get("replication_128", {})
    if replication != {
        "maximum_families": 2,
        "maximum_updates": 3200,
        "seeds": [1103],
        "rotations": ["R0", "R1", "R2"],
        "radius_cM": 0.2,
        "one_config_per_family": True,
        "training_overrides": {
            "warmup_updates": 400,
            "validation_every_updates": 200,
        },
        "purpose": (
            "preserve the 800-update small-pilot exposure per FIT person at "
            "96 FIT people"
        ),
    }:
        raise ContractError("128-person replication stage differs")
    budgets = [
        int(triage.get("maximum_updates", 0)),
        int(expansion.get("maximum_updates", 0)),
        int(radius_sensitivity.get("maximum_updates", 0)),
        int(finalists.get("maximum_updates", 0)),
    ]
    if not (0 < budgets[0] < budgets[1] == budgets[2] < budgets[3]):
        raise ContractError(
            "budgets must be triage < expansion = radius sensitivity < finalists"
        )

    selection = contract.get("selection", {})
    if not selection.get("no_family_elimination_from_two_or_fewer_points"):
        raise ContractError("two-point family elimination must be forbidden")
    if not selection.get("no_retuning_after_finalist_promotion"):
        raise ContractError("post-promotion retuning must be forbidden")
    training = contract.get("training", {})
    if training.get("learning_rate_policy") != "one_common_rate_no_posthoc_family_specific_search":
        raise ContractError("one common learning-rate policy must be explicit")
    if not (0.0 < float(training.get("learning_rate", 0.0)) <= 0.01):
        raise ContractError("common learning rate is outside the finite safe range")
    boundary_loss = contract.get("boundary_loss", {})
    if boundary_loss != {
        "target_transition_weight_share": 0.01,
        "formula": "beta=q*(1-p)/(p*(1-q))-1_fit_truth_only",
        "provenance": "M33_PRE4B",
    }:
        raise ContractError("boundary loss differs from the FIT-only M33 PRE4B rule")

    families = contract.get("families")
    if not isinstance(families, dict) or not families:
        raise ContractError("finite family space is missing")
    if tuple(families) != EXPECTED_FAMILIES:
        raise ContractError("family registry differs from the executable ModelSpec registry")
    if tuple(contract.get("model_spec_fields", [])) != MODEL_SPEC_FIELDS:
        raise ContractError("ModelSpec field registry differs from the executable API")
    global_ids: set[str] = set()
    for family, specification in families.items():
        configs = specification.get("configs", [])
        triage_ids = specification.get("triage_ids", [])
        if len(configs) < minimum + 1:
            raise ContractError(f"{family}: no room remains for local expansion")
        if len(triage_ids) < minimum:
            raise ContractError(f"{family}: fewer than three triage points")
        ids = [row.get("id") for row in configs]
        ranks = [row.get("complexity_rank") for row in configs]
        if len(ids) != len(set(ids)) or any(not isinstance(value, str) for value in ids):
            raise ContractError(f"{family}: config identifiers must be unique strings")
        if len(ranks) != len(set(ranks)) or sorted(ranks) != list(range(len(configs))):
            raise ContractError(f"{family}: complexity ranks must be contiguous from zero")
        if not set(triage_ids).issubset(ids):
            raise ContractError(f"{family}: triage references an undeclared config")
        for config in configs:
            if tuple(config.get("model_spec", {})) != MODEL_SPEC_FIELDS:
                raise ContractError(f"{family}/{config['id']}: ModelSpec fields differ from API")
            if "parameter_count" in config:
                raise ContractError(
                    f"{family}/{config['id']}: fixed parameter counts require model construction"
                )
            if "learning_rate" in config.get("training_overrides", {}):
                raise ContractError(
                    f"{family}/{config['id']}: per-config learning-rate search is forbidden"
                )
        if global_ids.intersection(ids):
            raise ContractError("config identifiers must also be globally unique")
        global_ids.update(ids)

    return contract


def _config_lookup(contract: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (family, config["id"]): config
        for family, specification in contract["families"].items()
        for config in specification["configs"]
    }


def _task(
    contract: dict[str, Any], family: str, config_id: str, seed: int,
    rotation: str, arm: str, maximum_updates: int, radius_cM: float,
    sweep_stage: str,
) -> dict[str, Any]:
    config = _config_lookup(contract)[(family, config_id)]
    training = contract["training"]
    return {
        "family": family,
        "config_id": config_id,
        "seed": seed,
        "rotation": rotation,
        "arm": arm,
        "radius_cM": radius_cM,
        "sweep_stage": sweep_stage,
        "maximum_updates": maximum_updates,
        "learning_rate": training["learning_rate"],
        "weight_decay": config.get("training_overrides", {}).get(
            "weight_decay", training["weight_decay"]
        ),
    }


def triage_plan(contract: dict[str, Any]) -> dict[str, Any]:
    stage = contract["stages"]["triage"]
    tasks = []
    for family, specification in sorted(contract["families"].items()):
        for config_id in specification["triage_ids"]:
            for arm in ARMS:
                tasks.append(_task(
                    contract, family, config_id, stage["seed"], stage["rotation"], arm,
                    stage["maximum_updates"], stage["radius_cM"], "triage",
                ))
    return {
        "schema_version": "1.0.0",
        "stage": "M34_TRIAGE_PLAN",
        "status": "PLAN_ONLY_NO_EXECUTION",
        "task_count": len(tasks),
        "tasks": tasks,
    }


def load_metric_pairs(
    contract: dict[str, Any], metrics_payload: dict[str, Any]
) -> dict[tuple[str, str, float, int, str, str], dict[str, dict[str, float]]]:
    records = metrics_payload.get("records")
    if not isinstance(records, list):
        raise ContractError("metrics JSON requires a records list")
    declared = _config_lookup(contract)
    pairs: dict[
        tuple[str, str, float, int, str, str], dict[str, dict[str, float]]
    ] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ContractError(f"metric record {index} is not an object")
        family = record.get("family")
        config_id = record.get("config_id")
        if (family, config_id) not in declared:
            raise ContractError(f"undeclared metric candidate: {family}/{config_id}")
        arm = record.get("arm")
        if arm not in ARMS:
            raise ContractError(f"invalid arm in metric record {index}: {arm}")
        seed = int(record.get("seed"))
        rotation = str(record.get("rotation"))
        radius_cM = _finite_number(record.get("radius_cM"), f"record {index}/radius_cM")
        sweep_stage = str(record.get("sweep_stage"))
        if sweep_stage not in (
            "triage", "local_expansion", "radius_sensitivity", "finalists",
            "replication_128",
        ):
            raise ContractError(f"invalid sweep stage in metric record {index}: {sweep_stage}")
        expected_updates = int(contract["stages"][sweep_stage]["maximum_updates"])
        if int(record.get("maximum_updates")) != expected_updates:
            raise ContractError(f"metric update budget differs from contract: record {index}")
        key = (family, config_id, radius_cM, seed, rotation, sweep_stage)
        bucket = pairs.setdefault(key, {})
        if arm in bucket:
            raise ContractError(f"duplicate metric record: {key}/{arm}")
        bucket[arm] = {
            metric: _finite_number(record.get(metric), f"record {index}/{metric}")
            for metric in METRIC_KEYS
        }
    incomplete = [key for key, arms in pairs.items() if set(arms) != set(ARMS)]
    if incomplete:
        raise ContractError(f"missing paired RD/RE metric records: {sorted(incomplete)}")
    return pairs


def _pair_deltas(arms: dict[str, dict[str, float]]) -> dict[str, float]:
    return {metric: arms["RE"][metric] - arms["RD"][metric] for metric in METRIC_KEYS}


def _guardrails_pass(contract: dict[str, Any], deltas: dict[str, float]) -> bool:
    ceilings = contract["selection"]["maximum_guardrail_worsening"]
    return all(deltas[key] <= float(ceilings[key]) for key in GUARDRAIL_KEYS)


def _score_key(
    contract: dict[str, Any], family: str, config_id: str, arms: dict[str, dict[str, float]]
) -> tuple[Any, ...]:
    deltas = _pair_deltas(arms)
    rank = _config_lookup(contract)[(family, config_id)]["complexity_rank"]
    return (
        not _guardrails_pass(contract, deltas),
        -deltas["boundary_F1_0.2cM"],
        -min(deltas["boundary_F1_0.1cM"], deltas["boundary_F1_0.5cM"]),
        deltas["NAM_truth_present_MAE"],
        deltas["false_transitions_per_cM"],
        rank,
        f"{family}/{config_id}",
    )


def _triage_pairs(
    contract: dict[str, Any],
    pairs: dict[tuple[str, str, float, int, str, str], dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    stage = contract["stages"]["triage"]
    result: dict[str, dict[str, dict[str, float]]] = {}
    for family, specification in contract["families"].items():
        family_pairs = {}
        for config_id in specification["triage_ids"]:
            key = (
                family, config_id, stage["radius_cM"], stage["seed"],
                stage["rotation"], "triage",
            )
            if key not in pairs:
                raise ContractError(f"triage pair is missing: {key}")
            family_pairs[config_id] = pairs[key]
        result[family] = family_pairs
    return result


def _expansion_ids(
    contract: dict[str, Any], triage: dict[str, dict[str, dict[str, float]]]
) -> dict[str, list[str]]:
    lookup = _config_lookup(contract)
    stage = contract["stages"]["local_expansion"]
    anchor_count = int(stage["anchor_count_per_family"])
    limit = int(stage["maximum_new_configs_per_family"])
    result: dict[str, list[str]] = {}
    for family, measured in sorted(triage.items()):
        anchors = sorted(
            measured,
            key=lambda config_id: _score_key(contract, family, config_id, measured[config_id]),
        )[:anchor_count]
        ranked = {
            config["complexity_rank"]: config["id"]
            for config in contract["families"][family]["configs"]
        }
        candidates = []
        for anchor_priority, anchor in enumerate(anchors):
            rank = lookup[(family, anchor)]["complexity_rank"]
            for direction_priority, neighbor_rank in enumerate((rank - 1, rank + 1)):
                neighbor = ranked.get(neighbor_rank)
                if neighbor is not None and neighbor not in measured:
                    candidates.append((anchor_priority, direction_priority, neighbor_rank, neighbor))
        selected = []
        for _anchor, _direction, _rank, config_id in sorted(candidates):
            if config_id not in selected:
                selected.append(config_id)
            if len(selected) == limit:
                break
        if len(selected) < limit:
            remaining = [
                config["id"]
                for config in contract["families"][family]["configs"]
                if config["id"] not in measured and config["id"] not in selected
            ]
            selected.extend(remaining[: limit - len(selected)])
        result[family] = selected
    return result


def _anchor_ids(
    contract: dict[str, Any], triage: dict[str, dict[str, dict[str, float]]]
) -> dict[str, list[str]]:
    count = int(contract["stages"]["local_expansion"]["anchor_count_per_family"])
    return {
        family: sorted(
            measured,
            key=lambda config_id: _score_key(contract, family, config_id, measured[config_id]),
        )[:count]
        for family, measured in sorted(triage.items())
    }


def expansion_plan(
    contract: dict[str, Any],
    pairs: dict[tuple[str, str, float, int, str, str], dict[str, dict[str, float]]],
) -> dict[str, Any]:
    triage = _triage_pairs(contract, pairs)
    selected = _expansion_ids(contract, triage)
    anchors = _anchor_ids(contract, triage)
    medium_ids = {
        family: anchors[family] + selected[family]
        for family in sorted(contract["families"])
    }
    stage = contract["stages"]["local_expansion"]
    tasks = [
        _task(
            contract, family, config_id, stage["seed"], stage["rotation"], arm,
            stage["maximum_updates"], stage["radius_cM"], "local_expansion",
        )
        for family, config_ids in sorted(medium_ids.items())
        for config_id in config_ids
        for arm in ARMS
    ]
    return {
        "schema_version": "1.0.0",
        "stage": "M34_LOCAL_EXPANSION_PLAN",
        "status": "PLAN_ONLY_NO_EXECUTION",
        "anchor_config_ids_by_family": anchors,
        "selected_config_ids_by_family": selected,
        "medium_budget_config_ids_by_family": medium_ids,
        "task_count": len(tasks),
        "tasks": tasks,
    }


def _within_ceilings(deltas: dict[str, float], ceilings: dict[str, Any]) -> bool:
    return all(deltas[key] <= float(ceilings[key]) for key in GUARDRAIL_KEYS)


def _architecture_candidates(
    contract: dict[str, Any],
    pairs: dict[tuple[str, str, float, int, str, str], dict[str, dict[str, float]]],
) -> tuple[str, list[tuple[tuple[Any, ...], str, str, dict[str, float]]]]:
    triage = _triage_pairs(contract, pairs)
    expected_expansion = _expansion_ids(contract, triage)
    anchors = _anchor_ids(contract, triage)
    stage = contract["stages"]["local_expansion"]
    minimum_delta = float(contract["selection"]["provisional_minimum_delta_F1"])
    family_best = []
    minimum_points = int(contract["stages"]["triage"]["minimum_evaluated_configs_per_family"])
    for family in sorted(contract["families"]):
        if len(triage[family]) < minimum_points:
            raise ContractError(f"{family}: fewer than three triage configurations")
        evaluated_ids = anchors[family] + expected_expansion[family]
        if len(set(evaluated_ids)) != 2:
            raise ContractError(f"{family}: medium-budget comparison must contain anchor and neighbor")
        measured = {}
        for config_id in evaluated_ids:
            key = (
                family, config_id, stage["radius_cM"], stage["seed"],
                stage["rotation"], "local_expansion",
            )
            if key not in pairs:
                raise ContractError(f"local expansion pair is missing: {key}")
            measured[config_id] = pairs[key]
        best_id = min(
            measured,
            key=lambda config_id: _score_key(contract, family, config_id, measured[config_id]),
        )
        deltas = _pair_deltas(measured[best_id])
        family_best.append((
            _score_key(contract, family, best_id, measured[best_id]), family, best_id, deltas
        ))

    maximum = int(contract["stages"]["radius_sensitivity"]["maximum_families"])
    promoted = [
        row for row in family_best
        if _guardrails_pass(contract, row[3])
        and row[3]["boundary_F1_0.2cM"] >= minimum_delta
    ]
    if promoted:
        return "promoted", sorted(promoted)[:maximum]
    fallback = contract["selection"]["rank_only_fallback"]
    exploratory = [
        row for row in family_best
        if row[3]["boundary_F1_0.2cM"] > 0.0
        and _within_ceilings(row[3], fallback["maximum_guardrail_worsening"])
    ]
    return fallback["label"], sorted(exploratory)[: int(fallback["maximum_families"])]


def radius_sensitivity_plan(
    contract: dict[str, Any],
    pairs: dict[tuple[str, str, float, int, str, str], dict[str, dict[str, float]]],
) -> dict[str, Any]:
    selection_mode, candidates = _architecture_candidates(contract, pairs)
    stage = contract["stages"]["radius_sensitivity"]
    reused_radius = stage["reuse_radius_cM_from_local_expansion"]
    new_radii = [radius for radius in stage["radii_cM"] if radius != reused_radius]
    tasks = [
        _task(
            contract, family, config_id, stage["seed"], stage["rotation"], arm,
            stage["maximum_updates"], radius, "radius_sensitivity",
        )
        for _score, family, config_id, _deltas in candidates
        for radius in new_radii
        for arm in ARMS
    ]
    return {
        "schema_version": "1.0.0",
        "stage": "M34_RADIUS_SENSITIVITY_PLAN",
        "status": "PLAN_ONLY_NO_EXECUTION" if candidates else "STOP_NO_POSITIVE_FAMILY",
        "selection_mode": selection_mode,
        "family_count": len(candidates),
        "reused_radius_cM_from_local_expansion": reused_radius,
        "new_radii_cM": new_radii,
        "selected_architectures": [
            {"family": family, "config_id": config_id, "screen_RE_minus_RD": deltas}
            for _score, family, config_id, deltas in candidates
        ],
        "task_count": len(tasks),
        "tasks": tasks,
    }


def finalist_plan(
    contract: dict[str, Any],
    pairs: dict[tuple[str, str, float, int, str, str], dict[str, dict[str, float]]],
) -> dict[str, Any]:
    selection_mode, architecture_candidates = _architecture_candidates(contract, pairs)
    radius_stage = contract["stages"]["radius_sensitivity"]
    selected = []
    for _architecture_score, family, config_id, _screen_deltas in architecture_candidates:
        measured = {}
        for radius in radius_stage["radii_cM"]:
            source_stage = (
                "local_expansion"
                if radius == radius_stage["reuse_radius_cM_from_local_expansion"]
                else "radius_sensitivity"
            )
            key = (
                family, config_id, radius, radius_stage["seed"],
                radius_stage["rotation"], source_stage,
            )
            if key not in pairs:
                raise ContractError(f"radius sensitivity pair is missing: {key}")
            measured[radius] = pairs[key]
        best_radius = min(
            measured,
            key=lambda radius: _score_key(contract, family, config_id, measured[radius]),
        )
        deltas = _pair_deltas(measured[best_radius])
        selected.append((
            _score_key(contract, family, config_id, measured[best_radius]),
            family, config_id, best_radius, deltas,
        ))

    final_stage = contract["stages"]["finalists"]
    tasks = [
        _task(
            contract, family, config_id, seed, rotation, arm,
            final_stage["maximum_updates"], radius, "finalists",
        )
        for _score, family, config_id, radius, _deltas in selected
        for seed in final_stage["seeds"]
        for rotation in final_stage["rotations"]
        for arm in ARMS
    ]
    finalists = [
        {
            "family": family,
            "config_id": config_id,
            "radius_cM": radius,
            "radius_sensitivity_RE_minus_RD": deltas,
            "selection_label": selection_mode,
        }
        for _score, family, config_id, radius, deltas in selected
    ]
    return {
        "schema_version": "1.0.0",
        "stage": "M34_FINALIST_PLAN",
        "status": "PLAN_ONLY_NO_EXECUTION" if finalists else "STOP_NO_PROVISIONAL_FAMILY",
        "finalist_count": len(finalists),
        "finalists": finalists,
        "reuse_identical_completed_tasks": True,
        "task_count": len(tasks),
        "tasks": tasks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("validate", "triage", "expand", "radius_sensitivity", "finalists"),
        required=True,
    )
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = validate_contract(strict_json(args.contract))
    if args.stage == "validate":
        output = {
            "schema_version": "1.0.0",
            "stage": "M34_CONTRACT_VALIDATION",
            "status": "PASS_CONTRACT_ONLY_NO_EXECUTION",
            "family_count": len(contract["families"]),
            "declared_config_count": sum(
                len(value["configs"]) for value in contract["families"].values()
            ),
        }
    elif args.stage == "triage":
        output = triage_plan(contract)
    else:
        if args.metrics is None:
            raise ContractError(f"--metrics is required for stage {args.stage}")
        pairs = load_metric_pairs(contract, strict_json(args.metrics))
        if args.stage == "expand":
            output = expansion_plan(contract, pairs)
        elif args.stage == "radius_sensitivity":
            output = radius_sensitivity_plan(contract, pairs)
        else:
            output = finalist_plan(contract, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("stage", "status")}, sort_keys=True))


if __name__ == "__main__":
    main()
