#!/usr/bin/env python3
"""Assemble the exact M34 triage metric grid and its audit table.

Candidate scores do not carry model identity themselves.  This boundary binds
each score to its training receipt through the prediction SHA-256, verifies the
full task against the frozen plan, and emits the flat ``records`` payload used
by ``m34_adaptive_sweep.py``.  F0 is retained as a common baseline in the audit
table; it is not inserted as a candidate arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import m34_adaptive_sweep as sweep


SCORE_STAGE = "M34_EXPLORATORY_SCORING"
SCORE_STATUS = "PASS_SCORED"
TRAIN_STAGE = "M34_EXPLORATORY_TRAIN_FACTORIZED_LAZY"
TRAIN_STATUS = "PASS_TRAINED_VALID_ONLY_FACTORIZED_LAZY"
TRANSFORMER_BATCH_STAGE = "M34_TRANSFORMER_PHYSICAL_BATCHING"
TRANSFORMER_BATCH_STATUS = "PASS_BOUNDED_ATTENTION_BATCHING"
BASELINE_LOGICAL_ID = "FLARE/F0/F0"
SCORER_METRIC_PATHS = {
    "boundary_F1_0.1cM": ("boundary", "0.1", "f1"),
    "boundary_F1_0.2cM": ("boundary", "0.2", "f1"),
    "boundary_F1_0.5cM": ("boundary", "0.5", "f1"),
    "macro_ancestry_dose_MAE": ("macro_ancestry_dose_MAE",),
    "NAM_truth_present_MAE": ("NAM_truth_present_MAE",),
    # The transition guardrail follows the primary 0.2 cM boundary tolerance.
    "false_transitions_per_cM": ("boundary", "0.2", "false_transitions_per_cM"),
    "haplotype_Brier": ("haplotype_Brier",),
}
MANIFEST_MEMBERS = {
    "schema_version", "ancestry_names", "haplotypes", "rotation", "splits",
}


class AggregateError(ValueError):
    """Raised when triage evidence is incomplete or does not match its plan."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, label: str) -> float:
    require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AggregateError(f"{label} must be numeric") from error
    require(math.isfinite(result), f"{label} must be finite")
    return result


def _sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _nested(payload: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    value: Any = payload
    for member in path:
        require(isinstance(value, Mapping) and member in value,
                f"{label} is missing {'.'.join(path)}")
        value = value[member]
    return value


def task_key(task: Mapping[str, Any]) -> tuple[str, str, str, int, str, float, str, int]:
    """Identity fields that must match one task, one receipt and one score."""
    return (
        str(task.get("family")), str(task.get("config_id")), str(task.get("arm")),
        int(task.get("seed")), str(task.get("rotation")), float(task.get("radius_cM")),
        str(task.get("sweep_stage")), int(task.get("maximum_updates")),
    )


def task_id(task: Mapping[str, Any]) -> str:
    family, config, arm, seed, root, radius, stage, updates = task_key(task)
    return f"{family}/{config}/{arm}/seed{seed}/{root}/r{radius:g}/{stage}/u{updates}"


def validate_plan(
    contract: dict[str, Any], plan: dict[str, Any],
    plan_source_metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stage = plan.get("stage")
    if stage == "M34_TRIAGE_PLAN":
        expected = sweep.triage_plan(contract)
    else:
        require(plan_source_metrics is not None,
                f"source metrics are required to verify {stage}")
        pairs = sweep.load_metric_pairs(contract, plan_source_metrics)
        planners = {
            "M34_LOCAL_EXPANSION_PLAN": sweep.expansion_plan,
            "M34_RADIUS_SENSITIVITY_PLAN": sweep.radius_sensitivity_plan,
            "M34_FINALIST_PLAN": sweep.finalist_plan,
        }
        require(stage in planners, f"unsupported adaptive plan stage: {stage}")
        expected = planners[stage](contract, pairs)
    require(plan == expected,
            "adaptive plan differs from the frozen adaptive contract and source metrics")
    tasks = plan.get("tasks")
    require(isinstance(tasks, list) and tasks and len(tasks) % 2 == 0,
            "adaptive plan must contain a positive even task count")
    keys = [task_key(task) for task in tasks]
    require(len(keys) == len(set(keys)), "triage plan contains duplicate task identities")
    return tasks


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = sweep.strict_json(path)
    require(set(manifest) == MANIFEST_MEMBERS,
            "factorized manifest members differ")
    require(manifest.get("schema_version") == "1.0.0",
            "factorized manifest schema differs")
    require(manifest.get("ancestry_names") == ["AFR", "EUR", "NAM"] and
            manifest.get("haplotypes") == 2 and manifest.get("rotation") == "R0",
            "factorized manifest axes or root differ")
    splits = manifest.get("splits")
    require(isinstance(splits, dict) and set(splits) == {"FIT", "VALID"},
            "factorized manifest must contain FIT and VALID only")
    require(all(isinstance(splits[name], list) and splits[name]
                for name in ("FIT", "VALID")),
            "factorized manifest has an empty FIT or VALID factor list")
    return manifest


def score_values(payload: dict[str, Any], label: str) -> dict[str, float]:
    require(payload.get("schema_version") == "1.0.0" and
            payload.get("stage") == SCORE_STAGE and
            payload.get("status") == SCORE_STATUS and
            payload.get("claim_level") == "exploratory",
            f"{label} scorer identity differs")
    require(payload.get("truth_opened_only_by_scorer") is True,
            f"{label} truth barrier receipt is missing")
    require(payload.get("ancestry_names") == ["AFR", "EUR", "NAM"],
            f"{label} ancestry axis differs")
    require(type(payload.get("sample_count")) is int and payload["sample_count"] > 0 and
            payload.get("haplotype_count") == 2 and
            type(payload.get("marker_count")) is int and payload["marker_count"] > 1,
            f"{label} scoring dimensions differ")
    require(_finite(payload.get("cm_span"), f"{label}/cm_span") > 0.0,
            f"{label} genetic span must be positive")
    input_hashes = payload.get("input_sha256")
    require(isinstance(input_hashes, dict) and set(input_hashes) == {"prediction", "truth"},
            f"{label} scorer input hashes differ")
    _sha256(input_hashes["prediction"], f"{label}/prediction")
    _sha256(input_hashes["truth"], f"{label}/truth")

    values = {
        metric: _finite(_nested(payload, path, label), f"{label}/{metric}")
        for metric, path in SCORER_METRIC_PATHS.items()
    }
    require(all(0.0 <= values[name] <= 1.0 for name in sweep.F1_KEYS),
            f"{label} boundary F1 is outside [0,1]")
    require(all(values[name] >= 0.0 for name in sweep.GUARDRAIL_KEYS),
            f"{label} guardrail metric is negative")
    ancestry_mae = payload.get("per_ancestry_MAE")
    require(isinstance(ancestry_mae, dict) and
            set(ancestry_mae) == {"AFR", "EUR", "NAM"} and
            all(_finite(ancestry_mae[name], f"{label}/{name}_MAE") >= 0.0
                for name in ("AFR", "EUR", "NAM")),
            f"{label} per-ancestry MAE differs")
    return values


def scoring_geometry(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        payload["sample_count"], payload["haplotype_count"], payload["marker_count"],
        tuple(payload["ancestry_names"]), float(payload["cm_span"]),
        payload["input_sha256"]["truth"],
    )


def validate_receipt(
    receipt: dict[str, Any], expected_task: dict[str, Any],
    contract_sha256: str, manifest_sha256: str,
) -> None:
    label = task_id(expected_task)
    require(receipt.get("schema_version") == "1.0.0" and
            receipt.get("stage") == TRAIN_STAGE and
            receipt.get("status") == TRAIN_STATUS and
            receipt.get("claim_level") == "exploratory",
            f"{label} training receipt identity differs")
    require(receipt.get("task") == expected_task,
            f"{label} receipt task differs from the frozen plan")
    require(receipt.get("test_opened") is False,
            f"{label} training receipt does not close TEST")
    require(receipt.get("contract_sha256") == contract_sha256,
            f"{label} adaptive contract SHA-256 differs")
    require(receipt.get("manifest_sha256") == manifest_sha256,
            f"{label} factorized manifest SHA-256 differs")
    require(receipt.get("fit_factor_count", 0) > 0 and
            receipt.get("valid_factor_count", 0) > 0 and
            receipt.get("fit_sample_count", 0) > 0 and
            receipt.get("valid_sample_count", 0) > 0,
            f"{label} FIT/VALID receipt counts differ")
    require(receipt.get("updates_executed") == expected_task["maximum_updates"],
            f"{label} executed update budget differs")
    selected = receipt.get("selected_update")
    require(type(selected) is int and 0 < selected <= expected_task["maximum_updates"],
            f"{label} selected update is outside the frozen budget")
    require(_finite(receipt.get("selected_valid_loss"), f"{label}/selected_valid_loss") >= 0.0,
            f"{label} VALID loss is negative")
    require(receipt.get("rd_re_pair_policy") ==
            "same_factors_axes_masks_geometry_F0_seed_and_task_except_arm",
            f"{label} RD/RE pairing policy differs")
    expected_pair_hash = canonical_sha256({
        name: value for name, value in expected_task.items() if name != "arm"
    })
    require(receipt.get("paired_task_sha256_without_arm") == expected_pair_hash,
            f"{label} arm-independent task hash differs")
    _sha256(receipt.get("task_sha256"), f"{label}/task_sha256")
    _sha256(receipt.get("valid_prediction_sha256"),
            f"{label}/valid_prediction_sha256")


def aggregate(
    contract_path: Path,
    plan_path: Path,
    manifest_path: Path,
    baseline_path: Path,
    metric_paths: Sequence[Path],
    receipt_paths: Sequence[Path],
    transformer_batching_paths: Sequence[Path],
    plan_source_metrics_path: Path | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    contract = sweep.validate_contract(sweep.strict_json(contract_path))
    plan = sweep.strict_json(plan_path)
    plan_source_metrics = (
        sweep.strict_json(plan_source_metrics_path)
        if plan_source_metrics_path is not None else None
    )
    tasks = validate_plan(contract, plan, plan_source_metrics)
    manifest = validate_manifest(manifest_path)
    contract_hash = sha256_file(contract_path)
    manifest_hash = sha256_file(manifest_path)

    expected_by_key = {task_key(task): task for task in tasks}
    expected_names = {
        f"{task['family']}.{task['config_id']}.{task['arm']}.metrics.json": task
        for task in tasks
    }
    require(len(metric_paths) == len(expected_names),
            "candidate metric count differs from the frozen plan")
    metrics_by_key: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
    observed_names: set[str] = set()
    for path in metric_paths:
        require(path.name in expected_names,
                f"unexpected candidate metric filename: {path.name}")
        require(path.name not in observed_names,
                f"duplicate candidate metric filename: {path.name}")
        observed_names.add(path.name)
        task = expected_names[path.name]
        metrics_by_key[task_key(task)] = (path, sweep.strict_json(path))
    require(observed_names == set(expected_names),
            "candidate metric filename grid is incomplete")

    require(len(receipt_paths) == len(expected_by_key),
            "training receipt count differs from the frozen plan")
    receipts_by_key: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
    for path in receipt_paths:
        receipt = sweep.strict_json(path)
        task = receipt.get("task")
        require(isinstance(task, dict), f"training receipt task is missing: {path}")
        key = task_key(task)
        require(key in expected_by_key,
                f"training receipt contains an undeclared task: {task_id(task)}")
        require(key not in receipts_by_key,
                f"duplicate training receipt: {task_id(task)}")
        validate_receipt(
            receipt, expected_by_key[key], contract_hash, manifest_hash,
        )
        receipts_by_key[key] = (path, receipt)
    require(set(receipts_by_key) == set(expected_by_key),
            "training receipt task grid is incomplete")

    transformer_tasks = {
        (task["family"], task["config_id"], task["arm"]): task
        for task in tasks if task["family"] == "transformer_small"
    }
    require(len(transformer_batching_paths) == len(transformer_tasks),
            "Transformer batching receipt count differs from the frozen plan")
    transformer_batching: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]] = {}
    for path in transformer_batching_paths:
        batching = sweep.strict_json(path)
        identity = (
            batching.get("family"), batching.get("config_id"), batching.get("arm"),
        )
        require(identity in transformer_tasks,
                f"unexpected Transformer batching receipt: {path}")
        require(identity not in transformer_batching,
                f"duplicate Transformer batching receipt: {identity}")
        task = transformer_tasks[identity]
        training_path, training = receipts_by_key[task_key(task)]
        require(
            batching.get("schema_version") == "1.0.0" and
            batching.get("stage") == TRANSFORMER_BATCH_STAGE and
            batching.get("status") == TRANSFORMER_BATCH_STATUS and
            batching.get("test_opened") is False,
            f"Transformer batching receipt identity differs: {path}",
        )
        require(batching.get("task_sha256") == training.get("task_sha256") and
                batching.get("paired_task_sha256_without_arm") ==
                training.get("paired_task_sha256_without_arm") and
                batching.get("train_receipt_sha256") == sha256_file(training_path),
                f"Transformer batching receipt is not bound to training: {path}")
        require(batching.get("logical_optimizer_updates") == task["maximum_updates"] and
                batching.get("maximum_updates") == task["maximum_updates"] and
                batching.get("optimizer_step_per_logical_shard_unchanged") is True,
                f"Transformer batching changed the logical update budget: {path}")
        require(type(batching.get("effective_maximum_rows_per_physical_microbatch")) is int and
                0 < batching["effective_maximum_rows_per_physical_microbatch"] <=
                int(batching.get("declared_maximum_rows_per_logical_microbatch", 0)),
                f"Transformer physical row cap differs: {path}")
        transformer_batching[identity] = (path, batching)
    for config_id in sorted({key[1] for key in transformer_tasks}):
        rd = transformer_batching[("transformer_small", config_id, "RD")][1]
        re = transformer_batching[("transformer_small", config_id, "RE")][1]
        require(
            (rd.get("policy"),
             rd.get("declared_maximum_rows_per_logical_microbatch"),
             rd.get("effective_maximum_rows_per_physical_microbatch"),
             rd.get("maximum_tokens_per_microbatch")) ==
            (re.get("policy"),
             re.get("declared_maximum_rows_per_logical_microbatch"),
             re.get("effective_maximum_rows_per_physical_microbatch"),
             re.get("maximum_tokens_per_microbatch")),
            f"Transformer {config_id} RD/RE batching policies differ",
        )

    baseline = sweep.strict_json(baseline_path)
    baseline_values = score_values(baseline, BASELINE_LOGICAL_ID)
    baseline_geometry = scoring_geometry(baseline)
    records: list[dict[str, Any]] = []
    input_audit: dict[str, dict[str, str]] = {}
    for task in tasks:
        key = task_key(task)
        metric_path, metric_payload = metrics_by_key[key]
        receipt_path, receipt = receipts_by_key[key]
        label = task_id(task)
        values = score_values(metric_payload, label)
        require(scoring_geometry(metric_payload) == baseline_geometry,
                f"{label} scoring geometry or VALID truth differs from F0")
        require(metric_payload["input_sha256"]["prediction"] ==
                receipt["valid_prediction_sha256"],
                f"{label} score is not bound to its training prediction")
        record = {
            "family": task["family"], "config_id": task["config_id"],
            "arm": task["arm"], "seed": task["seed"],
            "rotation": task["rotation"], "radius_cM": task["radius_cM"],
            "sweep_stage": task["sweep_stage"],
            "maximum_updates": task["maximum_updates"],
        } | values
        records.append(record)
        input_audit[label] = {
            "metrics_sha256": sha256_file(metric_path),
            "training_receipt_sha256": sha256_file(receipt_path),
            "prediction_sha256": receipt["valid_prediction_sha256"],
            "task_sha256": receipt["task_sha256"],
        }
        batching = transformer_batching.get(
            (task["family"], task["config_id"], task["arm"])
        )
        if batching is not None:
            input_audit[label]["transformer_batching_receipt_sha256"] = (
                sha256_file(batching[0])
            )

    pairs = sweep.load_metric_pairs(contract, {"records": records})
    require(len(pairs) == len(tasks) // 2,
            "aggregated RD/RE pair count differs from the frozen plan")
    for pair_key in pairs:
        family, config, radius, seed, root, stage = pair_key
        hashes = {
            receipts_by_key[(family, config, arm, seed, root, radius, stage,
                             contract["stages"][stage]["maximum_updates"])][1]
            ["paired_task_sha256_without_arm"]
            for arm in sweep.ARMS
        }
        require(len(hashes) == 1,
                f"{family}/{config}/{root}/r{radius:g} RD/RE receipts are not paired")

    rows = []
    for pair_key, arms in sorted(pairs.items()):
        family, config, radius, seed, root, stage = pair_key
        updates = contract["stages"][stage]["maximum_updates"]
        for metric in sweep.METRIC_KEYS:
            f0 = baseline_values[metric]
            rd = arms["RD"][metric]
            re = arms["RE"][metric]
            rows.append({
                "family": family, "config_id": config, "seed": seed,
                "root": root, "radius_cM": radius, "sweep_stage": stage,
                "maximum_updates": updates, "metric": metric,
                "F0": f0, "RD": rd, "RE": re,
                "RE_minus_RD": re - rd, "RE_minus_F0": re - f0,
            })
    header = (
        "family", "config_id", "seed", "root", "radius_cM", "sweep_stage",
        "maximum_updates", "metric", "F0", "RD", "RE", "RE_minus_RD",
        "RE_minus_F0",
    )
    tsv = "\t".join(header) + "\n" + "".join(
        "\t".join(str(row[name]) for name in header) + "\n" for row in rows
    )
    source_stage = plan["stage"]
    exact_status = {
        "M34_TRIAGE_PLAN": "PASS_EXACT_TRIAGE_GRID",
        "M34_LOCAL_EXPANSION_PLAN": "PASS_EXACT_LOCAL_EXPANSION_GRID",
        "M34_RADIUS_SENSITIVITY_PLAN": "PASS_EXACT_RADIUS_SENSITIVITY_GRID",
        "M34_FINALIST_PLAN": "PASS_EXACT_FINALIST_GRID",
    }[source_stage]
    prior_records = []
    if plan_source_metrics is not None:
        prior_records = plan_source_metrics.get("records")
        require(isinstance(prior_records, list),
                "plan source metrics require a records list")
    accumulated_records = list(prior_records) + records
    accumulated_pairs = sweep.load_metric_pairs(
        contract, {"records": accumulated_records},
    )
    require(len(accumulated_pairs) == len(accumulated_records) // 2,
            "accumulated adaptive metric pairs are incomplete")
    payload = {
        "schema_version": "1.0.0",
        "stage": "M34_ADAPTIVE_STAGE_METRICS_PAYLOAD",
        "source_plan_stage": source_stage,
        "status": exact_status,
        "claim_level": "exploratory",
        "evaluation_split": "VALID",
        "test_opened": False,
        "record_count": len(accumulated_records),
        "pair_count": len(accumulated_pairs),
        "stage_record_count": len(records),
        "stage_pair_count": len(pairs),
        "records": accumulated_records,
    }
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M34_AGGREGATE_ADAPTIVE_STAGE_METRICS",
        "source_plan_stage": source_stage,
        "status": f"{exact_status}_F0_RD_RE",
        "claim_level": "exploratory",
        "evaluation_split": "VALID",
        "factor_splits": sorted(manifest["splits"]),
        "test_opened": False,
        "record_count": len(accumulated_records),
        "pair_count": len(accumulated_pairs),
        "stage_record_count": len(records),
        "stage_pair_count": len(pairs),
        "long_table_row_count": len(rows),
        "false_transition_guardrail_tolerance_cM": 0.2,
        "input_sha256": {
            "adaptive_contract": contract_hash,
            "triage_plan": sha256_file(plan_path),
            "plan_source_metrics": (
                sha256_file(plan_source_metrics_path)
                if plan_source_metrics_path is not None else None
            ),
            "factorized_manifest": manifest_hash,
            "F0_metrics": sha256_file(baseline_path),
            "VALID_truth": baseline_geometry[-1],
            "tasks": input_audit,
        },
    }
    return payload, tsv, receipt


def write_outputs(
    payload: dict[str, Any], tsv: str, receipt: dict[str, Any],
    records_path: Path, table_path: Path, receipt_path: Path,
) -> dict[str, Any]:
    paths = (records_path, table_path, receipt_path)
    require(len({path.resolve() for path in paths}) == 3,
            "aggregate output paths must be distinct")
    require(not any(path.exists() for path in paths),
            "refusing to overwrite aggregate outputs")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        records_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        written.append(records_path)
        table_path.write_text(tsv, encoding="utf-8")
        written.append(table_path)
        result = dict(receipt)
        result["output_sha256"] = {
            "records": sha256_file(records_path),
            "table": sha256_file(table_path),
        }
        result["semantic_sha256"] = canonical_sha256(result)
        receipt_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        written.append(receipt_path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--plan-source-metrics", type=Path,
        help="Prior-stage records used to deterministically rebuild an adaptive plan.",
    )
    parser.add_argument("--factorized-manifest", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metric", type=Path, action="append", default=[])
    parser.add_argument(
        "--candidate-root", type=Path, action="append", default=[],
        help="Run root searched below metrics/ for candidate metric JSON files.",
    )
    parser.add_argument("--train-receipt", type=Path, action="append", default=[])
    parser.add_argument(
        "--train-root", type=Path, action="append", default=[],
        help="Run root searched below models/ for immutable training receipts.",
    )
    parser.add_argument(
        "--transformer-batching-receipt", type=Path, action="append", default=[],
    )
    parser.add_argument(
        "--transformer-batching-root", type=Path, action="append", default=[],
        help="Run root searched below models/ for Transformer batching receipts.",
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_paths = list(args.candidate_metric)
    for root in args.candidate_root:
        require(root.is_dir(), f"candidate result root is missing: {root}")
        metric_paths.extend(sorted(root.glob("metrics/**/*.metrics.json")))
    receipt_paths = list(args.train_receipt)
    for root in args.train_root:
        require(root.is_dir(), f"training result root is missing: {root}")
        receipt_paths.extend(sorted(root.glob("models/**/train.receipt.json")))
    require(len(metric_paths) == len({path.resolve() for path in metric_paths}),
            "candidate metric inputs contain duplicate paths")
    require(len(receipt_paths) == len({path.resolve() for path in receipt_paths}),
            "training receipt inputs contain duplicate paths")
    batching_paths = list(args.transformer_batching_receipt)
    for root in args.transformer_batching_root:
        require(root.is_dir(), f"Transformer batching result root is missing: {root}")
        batching_paths.extend(sorted(
            root.glob("models/**/transformer_batching.receipt.json")
        ))
    require(len(batching_paths) == len({path.resolve() for path in batching_paths}),
            "Transformer batching inputs contain duplicate paths")
    payload, table, receipt = aggregate(
        args.contract, args.plan, args.factorized_manifest, args.baseline_metrics,
        metric_paths, receipt_paths, batching_paths, args.plan_source_metrics,
    )
    final = write_outputs(
        payload, table, receipt, args.records, args.table, args.receipt,
    )
    print(json.dumps({
        "status": final["status"], "record_count": final["record_count"],
        "pair_count": final["pair_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
