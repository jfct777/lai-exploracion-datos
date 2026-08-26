#!/usr/bin/env python3
"""Run the M34 Transformer with bounded physical attention microbatches."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import m34_train_factorized as trainer


ATTENTION_ELEMENT_BUDGET = 134_217_728
POLICY = "transformer_attention_budgeted_rows_gradient_accumulated_per_logical_shard_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def capped_batches(
    planner: Callable[[Any, int, int], list[tuple[int, int]]],
    row_ptr: Any,
    declared_rows: int,
    maximum_tokens: int,
    physical_row_cap: int,
) -> list[tuple[int, int]]:
    """Split one logical shard while retaining its single optimizer update."""
    require(declared_rows > 0 and maximum_tokens > 0,
            "declared batch limits must be positive")
    require(physical_row_cap > 0, "physical row cap must be positive")
    return planner(row_ptr, min(declared_rows, physical_row_cap), maximum_tokens)


def physical_row_cap(contract: dict[str, Any], task: dict[str, Any],
                     declared_rows: int) -> int:
    family = contract["families"].get(task.get("family"), {})
    configurations = {
        row.get("id"): row for row in family.get("configs", [])
    }
    require(task.get("config_id") in configurations,
            "Transformer task is absent from the declared configuration space")
    specification = configurations[task["config_id"]]["model_spec"]
    depth = int(specification["depth"])
    heads = int(specification["transformer_heads"])
    tokens = int(specification["transformer_max_tokens"])
    denominator = depth * heads * tokens * tokens
    require(denominator > 0, "Transformer attention geometry must be positive")
    return max(1, min(declared_rows, ATTENTION_ELEMENT_BUDGET // denominator))


@contextmanager
def transformer_batching(physical_row_cap: int) -> Iterator[None]:
    original = trainer.packed_train.plan_row_batches

    def bounded(row_ptr: Any, maximum_rows: int,
                maximum_tokens: int) -> list[tuple[int, int]]:
        return capped_batches(
            original, row_ptr, maximum_rows, maximum_tokens, physical_row_cap,
        )

    trainer.packed_train.plan_row_batches = bounded
    try:
        yield
    finally:
        trainer.packed_train.plan_row_batches = original


def run(args: Any) -> dict[str, Any]:
    task = trainer.sweep.strict_json(args.task)
    require(task.get("family") == "transformer_small",
            "the bounded-attention runner accepts Transformer tasks only")
    contract = trainer.sweep.strict_json(args.contract)
    cap = physical_row_cap(contract, task, args.maximum_rows_per_batch)
    with transformer_batching(cap):
        receipt = trainer.run(args)
    train_receipt = args.outdir / "train.receipt.json"
    audit = {
        "schema_version": "1.0.0",
        "stage": "M34_TRANSFORMER_PHYSICAL_BATCHING",
        "status": "PASS_BOUNDED_ATTENTION_BATCHING",
        "claim_level": "exploratory",
        "family": task["family"],
        "config_id": task["config_id"],
        "arm": task["arm"],
        "task": task,
        "declared_maximum_rows_per_logical_microbatch": args.maximum_rows_per_batch,
        "effective_maximum_rows_per_physical_microbatch": min(
            args.maximum_rows_per_batch, cap,
        ),
        "maximum_tokens_per_microbatch": args.maximum_tokens_per_batch,
        "policy": POLICY,
        "logical_optimizer_updates": receipt["updates_executed"],
        "maximum_updates": task["maximum_updates"],
        "optimizer_step_per_logical_shard_unchanged": True,
        "task_sha256": receipt["task_sha256"],
        "paired_task_sha256_without_arm": receipt["paired_task_sha256_without_arm"],
        "train_receipt_sha256": trainer.sha256_file(train_receipt),
        "test_opened": False,
    }
    audit["semantic_sha256"] = trainer.canonical_sha256(audit)
    trainer.core.write_exclusive_json(
        args.outdir / "transformer_batching.receipt.json", audit,
    )
    return receipt


def main() -> None:
    args = trainer.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result["status"],
        "selected_update": result["selected_update"],
        "valid_loss": result["selected_valid_loss"],
        "physical_row_cap": physical_row_cap(
            trainer.sweep.strict_json(args.contract), result["task"],
            args.maximum_rows_per_batch,
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
