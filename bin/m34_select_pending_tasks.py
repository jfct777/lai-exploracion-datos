#!/usr/bin/env python3
"""Select the exact unfinished M34 tasks from immutable training receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import m34_adaptive_sweep as sweep
import m34_train_factorized as trainer


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task.get("family"), task.get("config_id"), task.get("arm"),
        task.get("seed"), task.get("rotation"), task.get("radius_cM"),
        task.get("sweep_stage"), task.get("maximum_updates"),
    )


def expected_plan(
    contract: dict[str, Any], plan: dict[str, Any],
    metrics_path: Path | None,
) -> dict[str, Any]:
    """Rebuild a declared adaptive plan before selecting unfinished tasks."""
    stage = plan.get("stage")
    if stage == "M34_TRIAGE_PLAN":
        return sweep.triage_plan(contract)
    require(metrics_path is not None,
            f"--metrics is required to verify {stage}")
    pairs = sweep.load_metric_pairs(contract, sweep.strict_json(metrics_path))
    planners = {
        "M34_LOCAL_EXPANSION_PLAN": sweep.expansion_plan,
        "M34_RADIUS_SENSITIVITY_PLAN": sweep.radius_sensitivity_plan,
        "M34_FINALIST_PLAN": sweep.finalist_plan,
    }
    require(stage in planners, f"unsupported adaptive plan stage: {stage}")
    return planners[stage](contract, pairs)


def select(contract_path: Path, plan_path: Path,
           receipt_paths: Sequence[Path],
           manifest_path: Path,
           metrics_path: Path | None = None) -> dict[str, Any]:
    contract = sweep.validate_contract(sweep.strict_json(contract_path))
    plan = sweep.strict_json(plan_path)
    rebuilt = expected_plan(contract, plan, metrics_path)
    require(plan == rebuilt, "adaptive plan differs from its contract and metrics")
    require(manifest_path.is_file(), "factorized manifest is missing")
    contract_sha256 = trainer.sha256_file(contract_path)
    manifest_sha256 = trainer.sha256_file(manifest_path)
    expected = {key(task): task for task in plan["tasks"]}
    completed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for receipt_path in receipt_paths:
        receipt = sweep.strict_json(receipt_path)
        require(receipt.get("status") == "PASS_TRAINED_VALID_ONLY_FACTORIZED_LAZY" and
                receipt.get("test_opened") is False,
                f"training receipt is not consumable: {receipt_path}")
        task = receipt.get("task")
        require(isinstance(task, dict) and key(task) in expected,
                f"receipt task is absent from the frozen plan: {receipt_path}")
        require(task == expected[key(task)],
                f"receipt task differs from the frozen plan: {receipt_path}")
        require(receipt.get("contract_sha256") == contract_sha256,
                f"receipt contract hash differs: {receipt_path}")
        require(receipt.get("manifest_sha256") == manifest_sha256,
                f"receipt factorized manifest hash differs: {receipt_path}")
        paired = {name: value for name, value in task.items() if name != "arm"}
        require(receipt.get("paired_task_sha256_without_arm") ==
                trainer.canonical_sha256(paired),
                f"receipt paired-task hash differs: {receipt_path}")
        require(key(task) not in completed,
                f"duplicate completed task: {task['family']}/{task['config_id']}/{task['arm']}")
        prediction = receipt_path.parent / "valid.prediction.npz"
        require(prediction.is_file(), f"completed prediction is missing: {prediction}")
        require(trainer.sha256_file(prediction) == receipt.get("valid_prediction_sha256"),
                f"completed prediction hash differs: {prediction}")
        completed[key(task)] = {
            "task": task,
            "training_receipt": str(receipt_path.resolve()),
            "prediction": str(prediction.resolve()),
        }
    pending = [task for task in plan["tasks"] if key(task) not in completed]
    require(len(completed) + len(pending) == len(expected),
            "completed and pending task counts do not close the triage grid")
    return {
        "schema_version": "1.0.0",
        "stage": "M34_PENDING_TASK_SELECTION",
        "status": "PASS_EXACT_COMPLEMENT",
        "claim_level": "exploratory",
        "test_opened": False,
        "source_plan_stage": plan["stage"],
        "task_count": len(expected),
        "completed_count": len(completed),
        "pending_count": len(pending),
        "completed": [completed[key(task)] for task in plan["tasks"] if key(task) in completed],
        "pending_tasks": pending,
        "input_sha256": {
            "contract": contract_sha256,
            "factorized_manifest": manifest_sha256,
            "plan": trainer.sha256_file(plan_path),
            "metrics": (
                trainer.sha256_file(metrics_path) if metrics_path is not None else None
            ),
            "training_receipts": {
                str(path.resolve()): trainer.sha256_file(path) for path in receipt_paths
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--factorized-manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--completed-receipt", type=Path, action="append", default=[])
    parser.add_argument("--completed-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_paths = list(args.completed_receipt)
    for root in args.completed_root:
        require(root.is_dir(), f"completed result root is missing: {root}")
        receipt_paths.extend(sorted(root.glob("models/**/train.receipt.json")))
    require(len(receipt_paths) == len({path.resolve() for path in receipt_paths}),
            "completed receipt inputs contain duplicate paths")
    payload = select(
        args.contract, args.plan, receipt_paths,
        args.factorized_manifest, args.metrics,
    )
    trainer.core.write_exclusive_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "completed_count": payload["completed_count"],
        "pending_count": payload["pending_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
