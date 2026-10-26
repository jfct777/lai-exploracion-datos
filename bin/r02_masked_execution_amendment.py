#!/usr/bin/env python3
"""Install the reviewed exact q=1 execution amendment, then run frozen training.

The amendment changes only temporary query blocking.  It delegates every
neighbor predicate, reduction, attention operation, model, optimizer and
checkpoint identity to the frozen core package.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SCHEMA = "r02_masked_chr22_exact_query_subblock_amendment_request_v1"
STATUS = "IMPLEMENTED_AWAITING_EXACT_POST"
POST_SCHEMA = "r02_masked_chr22_exact_query_subblock_amendment_post_v1"
POST_STATUS = "ACCEPTED_EXACT_QUERY_SUBDIVISION_EXECUTION_ONLY"
TELEMETRY = {"attention_min_query_block_size": 16,
             "context_min_query_block_size": 16,
             "attention_budget_fallbacks": 0,
             "context_budget_fallbacks": 0,
             "attention_rejected_call_seconds": 0.0,
             "context_rejected_call_seconds": 0.0,
             "attention_worst_mask_elements": 0,
             "context_worst_mask_elements": 0,
             "requested_query_block_size": 16}
OBSERVATION = {"operation": None, "multiplier": 1}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_amendment(path, post_path):
    import genomic_context_runtime as runtime
    import r02_local_attention_blocks as blocks
    import r02_masked_training_request as training
    value = json.loads(Path(path).read_text())
    required = {"schema", "status", "query_block_size_before",
        "query_block_size_after", "max_mask_elements", "operation",
        "scientific_values_changed", "checkpoint_identity_changed",
        "frozen_files", "original_package", "pre_sha256", "post_sha256",
        "affected_fit_ids", "original_runtime_runner",
        "amended_runtime_runner", "measured_q1_bounds", "select_opened",
        "evaluate_opened"}
    require(set(value) == required and value["schema"] == SCHEMA
            and value["status"] == STATUS
            and value["query_block_size_before"] == 16
            and value["query_block_size_after"] == [16, 8, 4, 2, 1]
            and value["max_mask_elements"] == 1_000_000
            and value["operation"] ==
                "SAME_EXACT_NEIGHBORS_AND_OPERATIONS_RECURSIVELY_HALVE_ONLY_REJECTED_QUERY_BLOCKS"
            and value["scientific_values_changed"] is False
            and value["checkpoint_identity_changed"] is False
            and value["affected_fit_ids"] == [
                "local_attention__count_only", "local_attention__full",
                "local_attention__full__rare_identity",
                "local_deepsets__count_only", "local_deepsets__full",
                "local_deepsets__full__rare_identity"]
            and value["measured_q1_bounds"] == {
                "attention_update1_cells": [6248, 304],
                "attention_max_elements_upper_bound": 111648,
                "deepsets_update5_cells": [4898, 736],
                "deepsets_max_elements_upper_bound": 77688}
            and value["select_opened"] is False
            and value["evaluate_opened"] is False,
            "execution amendment differs from reviewed exact q=1 design")
    actual = {"genomic_context_runtime.py": runtime.__file__,
              "r02_local_attention_blocks.py": blocks.__file__,
              "r02_masked_training_request.py": training.__file__}
    require(set(value["frozen_files"]) == set(actual),
            "frozen numerical file inventory differs")
    for name, path_actual in actual.items():
        require(value["frozen_files"][name] == sha(path_actual),
                name + " differs from frozen core")
    require(Path(value["original_package"]["path"]).is_file()
            and value["original_package"]["sha256"] ==
                sha(value["original_package"]["path"]),
            "original package differs")
    require(Path(value["original_runtime_runner"]["path"]).is_file()
            and value["original_runtime_runner"]["sha256"] ==
                sha(value["original_runtime_runner"]["path"]),
            "original runtime runner differs")
    require(Path(value["amended_runtime_runner"]["path"]).is_file()
            and value["amended_runtime_runner"]["sha256"] ==
                sha(value["amended_runtime_runner"]["path"]),
            "amended runtime runner differs")
    post = json.loads(Path(post_path).read_text())
    amendment_root = Path(path).resolve().parent
    require(post == {"schema": POST_SCHEMA, "status": POST_STATUS,
        "amendment_request_sha256": sha(path),
        "amendment_runner_sha256": sha(__file__),
        "amendment_package_sha256": sha(
            amendment_root / "masked-execution-amendment.tgz"),
        "amendment_manifest_sha256": sha(
            amendment_root / "package/MANIFEST.json"),
        "training_pre_sha256": value["pre_sha256"],
        "training_post_sha256": value["post_sha256"],
        "authorized_role": "FIT", "select_opened": False,
        "evaluate_opened": False}, "exact execution amendment POST required")
    return value


def install_exact_query_subblocks():
    import genomic_context_runtime as runtime
    import r02_local_attention_blocks as blocks
    original_attention = blocks.blocked_attention
    original_mean = blocks.blocked_local_mean
    original_candidate_keys = blocks.candidate_keys

    def candidate_keys(qcm, qchrom, qvalid, cm, chrom, valid, radius):
        keys = original_candidate_keys(qcm, qchrom, qvalid, cm, chrom,
                                       valid, radius)
        operation = OBSERVATION["operation"]
        if operation is not None:
            elements = (qcm.shape[0] * qcm.shape[1] * len(keys)
                        * OBSERVATION["multiplier"])
            field = operation + "_worst_mask_elements"
            TELEMETRY[field] = max(TELEMETRY[field], elements)
        return keys

    blocks.candidate_keys = candidate_keys

    def attention(attention_module, clean, cm, chrom, valid, radius, execution):
        require(execution.query_block_size == 16
                and execution.max_mask_elements == 1_000_000,
                "amendment applies only to frozen q=16/mask-budget configuration")
        query_block_size = execution.query_block_size
        while True:
            attempted = time.monotonic()
            try:
                OBSERVATION.update(operation="attention",
                                   multiplier=attention_module.num_heads)
                result = original_attention(attention_module, clean, cm, chrom,
                    valid, radius, replace(execution,
                                           query_block_size=query_block_size))
                TELEMETRY["attention_min_query_block_size"] = min(
                    TELEMETRY["attention_min_query_block_size"], query_block_size)
                return result
            except ValueError as error:
                if str(error) != "attention mask budget exceeded; no neighbors were truncated":
                    raise
                require(query_block_size > 1,
                        "one-query attention mask still exceeds frozen budget")
                TELEMETRY["attention_budget_fallbacks"] += 1
                TELEMETRY["attention_rejected_call_seconds"] += (
                    time.monotonic() - attempted)
                query_block_size = max(1, query_block_size // 2)
            finally:
                OBSERVATION.update(operation=None, multiplier=1)

    def mean(values, query, source, radius, execution):
        require(execution.query_block_size == 16
                and execution.max_mask_elements == 1_000_000,
                "amendment applies only to frozen q=16/mask-budget configuration")
        query_block_size = execution.query_block_size
        while True:
            attempted = time.monotonic()
            try:
                OBSERVATION.update(operation="context", multiplier=1)
                result = original_mean(values, query, source, radius,
                    replace(execution, query_block_size=query_block_size))
                TELEMETRY["context_min_query_block_size"] = min(
                    TELEMETRY["context_min_query_block_size"], query_block_size)
                return result
            except ValueError as error:
                if str(error) != "context mask budget exceeded; no neighbors were truncated":
                    raise
                require(query_block_size > 1,
                        "one-query context mask still exceeds frozen budget")
                TELEMETRY["context_budget_fallbacks"] += 1
                TELEMETRY["context_rejected_call_seconds"] += (
                    time.monotonic() - attempted)
                query_block_size = max(1, query_block_size // 2)
            finally:
                OBSERVATION.update(operation=None, multiplier=1)

    runtime.blocked_attention = attention
    runtime.blocked_local_mean = mean


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--amendment", required=True)
    result.add_argument("--amendment-post", required=True)
    result.add_argument("training_args", nargs=argparse.REMAINDER)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.training_args and args.training_args[0] in {"run", "score"},
            "only frozen training run or paired flank score may be amended")
    amendment = validate_amendment(args.amendment, args.amendment_post)
    output_index = args.training_args.index("--output") + 1
    output = Path(args.training_args[output_index])
    if args.training_args[0] == "score":
        training_receipt_index = args.training_args.index(
            "--training-receipt") + 1
        training_receipt = Path(
            args.training_args[training_receipt_index]).resolve()
        fit_receipt = json.loads(training_receipt.read_text())
        require(fit_receipt.get("fit_id") in amendment["affected_fit_ids"]
                and fit_receipt.get("status") ==
                    "DECLARED_UPDATE_BUDGET_COMPLETE",
                "paired score requires amended completed FULL FIT")
        execution_receipt_path = (
            training_receipt.parent / "EXECUTION_AMENDMENT_RECEIPT.json")
        require(execution_receipt_path.is_file(),
                "FULL FIT lacks execution amendment receipt")
        execution_receipt = json.loads(execution_receipt_path.read_text())
        require(execution_receipt.get("amendment_sha256") ==
                    sha(args.amendment)
                and execution_receipt.get("amendment_post_sha256") ==
                    sha(args.amendment_post)
                and execution_receipt.get("training_receipt_sha256") ==
                    sha(training_receipt)
                and execution_receipt.get("checkpoint_identity_preserved")
                    is True,
                "FULL FIT execution amendment binding differs")
        install_exact_query_subblocks()
        import r02_masked_flank_diagnostic_v2 as flank
        code = flank.main(args.training_args)
        score_receipt = output / "FLANK_SCORE.json"
        require(code == 0 and score_receipt.is_file(),
                "paired score did not produce its exact receipt")
        atomic_json(output / "SCORE_EXECUTION_AMENDMENT.json", {
            "schema": "r02_masked_chr22_score_execution_amendment_v1",
            "amendment_sha256": sha(args.amendment),
            "amendment_post_sha256": sha(args.amendment_post),
            "training_execution_amendment_sha256":
                sha(execution_receipt_path),
            "training_receipt_sha256": sha(training_receipt),
            "score_receipt_sha256": sha(score_receipt),
            "runner_sha256": sha(__file__),
            "query_block_execution": dict(TELEMETRY),
            "select_opened": False, "evaluate_opened": False})
        return code

    request_index = args.training_args.index("--request") + 1
    fit_id = json.loads(Path(args.training_args[request_index]).read_text())["fit_id"]
    require(fit_id in amendment["affected_fit_ids"],
            "fit does not require rare-information execution amendment")
    checkpoint = output / "checkpoint"
    pointer_path = checkpoint / "CURRENT"
    initial = None
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        manifest_path = checkpoint / pointer["manifest"]
        require(manifest_path.is_file()
                and sha(manifest_path) == pointer["manifest_sha256"],
                "initial checkpoint pointer or manifest changed")
        initial = {"current_sha256": sha(pointer_path),
                   "manifest_sha256": sha(manifest_path),
                   "identity_sha256": pointer["identity_sha256"],
                   "next_step": pointer["next_step"]}
    start = {"schema": "r02_masked_chr22_execution_amendment_start_v1",
        "fit_id": fit_id, "amendment_sha256": sha(args.amendment),
        "amendment_post_sha256": sha(args.amendment_post),
        "runner_sha256": sha(__file__), "initial_checkpoint": initial,
        "runtime_runner_sha256": amendment["amended_runtime_runner"]["sha256"],
        "partial_gradient_checkpoint_present": any(
            path.is_file() for directory in checkpoint.glob("partial-*")
            for path in directory.rglob("*")),
        "adam_state_preserved": True, "select_opened": False,
        "evaluate_opened": False}
    start_path = output / "EXECUTION_AMENDMENT_START.json"
    if start_path.is_file():
        existing = json.loads(start_path.read_text())
        require(existing.get("schema") == start["schema"]
                and existing.get("fit_id") == fit_id
                and existing.get("amendment_sha256") == start["amendment_sha256"]
                and existing.get("amendment_post_sha256") ==
                    start["amendment_post_sha256"]
                and existing.get("runner_sha256") == start["runner_sha256"]
                and existing.get("runtime_runner_sha256") ==
                    start["runtime_runner_sha256"]
                and existing.get("adam_state_preserved") is True
                and existing.get("select_opened") is False
                and existing.get("evaluate_opened") is False,
                "execution amendment start binding changed")
    else:
        atomic_json(start_path, start)
    checkpoint_binding_path = output / "EXECUTION_AMENDMENT_CHECKPOINT.json"
    if checkpoint_binding_path.is_file():
        binding = json.loads(checkpoint_binding_path.read_text())
        require(pointer_path.is_file()
                and binding.get("schema") ==
                    "r02_masked_chr22_execution_amendment_checkpoint_v1"
                and binding.get("amendment_start_sha256") == sha(start_path)
                and binding.get("amendment_sha256") == sha(args.amendment)
                and binding.get("amendment_post_sha256") ==
                    sha(args.amendment_post)
                and binding.get("current_sha256") == sha(pointer_path),
                "current checkpoint lacks its execution amendment binding")
    install_exact_query_subblocks()
    import r02_masked_training_request as training
    code = training.main(args.training_args)
    receipt = output / "TRAINING_RECEIPT.json"
    current_binding = None
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        manifest_path = checkpoint / pointer["manifest"]
        require(manifest_path.is_file()
                and sha(manifest_path) == pointer["manifest_sha256"],
                "amended checkpoint pointer or manifest changed")
        current_binding = {
            "schema": "r02_masked_chr22_execution_amendment_checkpoint_v1",
            "fit_id": fit_id, "amendment_start_sha256": sha(start_path),
            "amendment_sha256": sha(args.amendment),
            "amendment_post_sha256": sha(args.amendment_post),
            "current_sha256": sha(pointer_path),
            "manifest_sha256": sha(manifest_path),
            "identity_sha256": pointer["identity_sha256"],
            "next_step": pointer["next_step"],
            "adam_state_preserved": True, "select_opened": False,
            "evaluate_opened": False}
        atomic_json(checkpoint_binding_path, current_binding)
    sidecar = {"schema": "r02_masked_chr22_execution_amendment_receipt_v1",
        "amendment_sha256": sha(args.amendment),
        "amendment_post_sha256": sha(args.amendment_post),
        "runner_sha256": sha(__file__), "training_returncode": code,
        "training_receipt_sha256": sha(receipt) if receipt.is_file() else None,
        "amendment_start_sha256": sha(start_path),
        "amendment_checkpoint_sha256": (
            sha(checkpoint_binding_path) if current_binding else None),
        "query_block_execution": dict(TELEMETRY),
        "checkpoint_identity_preserved": True,
        "select_opened": False, "evaluate_opened": False}
    atomic_json(output / "EXECUTION_AMENDMENT_RECEIPT.json", sidecar)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        raise SystemExit(2)
