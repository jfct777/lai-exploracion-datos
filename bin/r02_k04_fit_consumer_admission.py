#!/usr/bin/env python3
"""Admit the direct K04 FIT/SELECT consumer over published BUILD66 matrices.

This module authenticates reception and dependencies.  It never constructs
matrices, chooses numerical parameters, fits a model, or reads EVALUATE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


TASK = re.compile(r"chr(0[1-9]|1[0-9]|2[0-2])_(AFR|EUR|NAM)")
HEX = re.compile(r"[0-9a-f]{64}")
MATRIX_SCHEMA = "r02_k04_ccqcqr_inputs_v1"
MATRIX_STATUS = "COMPLETE_REVIEWED_MODEL_INPUTS_NOT_MODEL_SELECTION_OR_BIOLOGICAL_VALIDATION"
PLAN_SCHEMA = "r02_k04_fit_select_plan_v1"
PUBLICATION_SCHEMA = "r02_k04_matrix_manifest_publication_receipt_v1"
W_SHA256 = "6540d76c71b41de0fea50d6047658fa8e92d575435f0ee73aa4f2fc1d7c3fe1e"
CODE_NAMES = (
    "r02_k04_model_bridge.py", "r02_k04_predictor_inputs.py", "r02_pair_operator.py",
    "r02_interval_ridge.py", "r02_k04_fit_select.py", "r02_k04_candidate_store.py",
    "r02_k04_precision_policy.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"real file required: {target}")
    with target.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {target}")
    return value


def absolute_file(path, expected):
    target = Path(path)
    if not target.is_absolute() or not target.is_file() or target.is_symlink():
        raise ValueError(f"real absolute file required: {target}")
    if not HEX.fullmatch(str(expected)) or sha256(target) != expected:
        raise ValueError(f"SHA-256 differs: {target}")
    return target


def authenticate_matrix(root, expected_manifest_sha, deep):
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError(f"real absolute matrix directory required: {root}")
    manifest_path = absolute_file(root / "manifest.json", expected_manifest_sha)
    manifest = load(manifest_path)
    if not (manifest.get("schema") == MATRIX_SCHEMA and manifest.get("status") == MATRIX_STATUS
            and manifest.get("evaluate_allowed") is False and manifest.get("model_fit") is False):
        raise ValueError(f"reviewed FIT/SELECT-only matrix manifest required: {root}")
    hashes = manifest.get("output_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"matrix payload hash inventory required: {root}")
    if deep:
        for relative, digest in hashes.items():
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"unsafe matrix payload path: {relative!r}")
            absolute_file(root / relative, digest)
    return manifest


def authenticate_publication(publication, manifest_path, manifest_sha):
    """Bind a pinned local receipt to the exact published manifest object."""
    required = {"uri", "generation", "bytes", "sha256",
                "verification_receipt", "verification_receipt_sha256"}
    if not (isinstance(publication, dict) and set(publication) == required and
            isinstance(publication["uri"], str) and publication["uri"].startswith("gs://") and
            str(publication["generation"]).isdigit() and
            type(publication["bytes"]) is int and publication["bytes"] > 0 and
            publication["sha256"] == manifest_sha):
        raise ValueError("exact GCS manifest publication identity required")
    receipt_path = absolute_file(publication["verification_receipt"],
                                 publication["verification_receipt_sha256"])
    receipt = load(receipt_path)
    expected = {"schema": PUBLICATION_SCHEMA, "status": "VERIFIED",
                "uri": publication["uri"], "generation": str(publication["generation"]),
                "bytes": publication["bytes"], "sha256": manifest_sha}
    if receipt != expected:
        raise ValueError("publication receipt does not authenticate exact URI/generation/bytes/SHA-256")
    if Path(manifest_path).stat().st_size != publication["bytes"]:
        raise ValueError("published manifest byte count differs from received manifest")
    return receipt


def validate_plan(plan, matrix, chromosome, ancestry):
    required = {"schema", "operation", "w_plan", "bridge_request_sha256",
                "target_chromosome", "ancestry", "resources", "numerical", "arms"}
    if not (set(plan) == required and plan.get("schema") == PLAN_SCHEMA and
            plan.get("operation") == "FIT_SELECT_COMPLETE_W_FINITE_FAMILY" and
            plan.get("bridge_request_sha256") == matrix.get("request_sha256") and
            str(plan.get("target_chromosome")) == chromosome and plan.get("ancestry") == ancestry):
        raise ValueError("fit plan identity or fields do not match received matrices")
    w_plan = plan["w_plan"]
    if not (isinstance(w_plan, dict) and set(w_plan) == {"path", "sha256"} and
            w_plan["sha256"] == W_SHA256):
        raise ValueError("authenticated admitted W plan required")
    absolute_file(w_plan["path"], W_SHA256)
    resources = plan["resources"]
    resource_fields = {"rows_per_chunk", "max_working_mib", "workers", "operator_backend"}
    if not (isinstance(resources, dict) and set(resources) == resource_fields and
            all(type(resources[x]) is int and resources[x] > 0
                for x in ("rows_per_chunk", "max_working_mib", "workers")) and
            resources["operator_backend"] in {"factorized", "factorized_admitted_pairs"}):
        raise ValueError("explicit measured operator resources required")
    if resources["operator_backend"] != matrix.get("solver_bridge", {}).get("operator_backend"):
        raise ValueError("fit-plan operator backend differs from published matrices")
    numerical = plan["numerical"]
    numeric_fields = {"step_fraction", "max_iterations", "max_seconds_per_candidate",
        "check_every", "gap_absolute_tolerance", "gap_relative_tolerance",
        "dual_feasibility_tolerance", "centering_tolerance", "checkpoint_every"}
    if not (isinstance(numerical, dict) and set(numerical) == numeric_fields and
            all(type(numerical[x]) in (int, float) and math.isfinite(numerical[x]) and
                numerical[x] > 0 for x in numerical) and
            all(type(numerical[x]) is int for x in
                ("max_iterations", "check_every", "checkpoint_every")) and
            0 < numerical["step_fraction"] < 1):
        raise ValueError("explicit measured convergence and checkpoint controls required")
    arms = plan["arms"]
    arm_fields = {"augmented_operator_norm_bound", "select_augmented_row_norm_bound",
                  "guard_extensions_low", "guard_extensions_high"}
    if not (isinstance(arms, dict) and set(arms) == {"C", "CQ", "CQR"} and
            all(isinstance(value, dict) and set(value) == arm_fields and
                all(type(value[x]) in (int, float) and math.isfinite(value[x]) and value[x] > 0
                    for x in ("augmented_operator_norm_bound", "select_augmented_row_norm_bound")) and
                all(type(value[x]) is int and value[x] >= 0
                    for x in ("guard_extensions_low", "guard_extensions_high"))
                for value in arms.values())):
        raise ValueError("measured arm bounds and explicit W guard state required")


def inspect(spec_path, code_dir, deep):
    spec = load(spec_path)
    if spec.get("schema") != "r02_k04_published_fit_consumer_spec_v1":
        raise ValueError("K04 published-consumer specification v1 required")
    if spec.get("evaluate_allowed") is not False:
        raise ValueError("EVALUATE must remain closed")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("tasks list required")
    blockers, admitted, seen = [], [], set()
    for item in tasks:
        task_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(task_id, str) or not TASK.fullmatch(task_id) or task_id in seen:
            raise ValueError(f"unique canonical task id required: {task_id!r}")
        seen.add(task_id)
        required = {"id", "matrix_dir", "matrix_manifest_sha256", "publication",
                    "fit_plan", "fit_plan_sha256"}
        if set(item) != required:
            raise ValueError(f"exact task fields required for {task_id}")
        publication = item["publication"]
        try:
            matrix = authenticate_matrix(item["matrix_dir"], item["matrix_manifest_sha256"], deep)
            authenticate_publication(publication, Path(item["matrix_dir"]) / "manifest.json",
                                     item["matrix_manifest_sha256"])
        except (ValueError, OSError, json.JSONDecodeError) as error:
            blockers.append({"task_id": task_id, "dependency": "PUBLISHED_MATRIX", "reason": str(error)})
            continue
        chromosome, ancestry = task_id.split("_")
        chromosome = str(int(chromosome[3:]))
        if str(matrix.get("target_chromosome")) != chromosome or matrix.get("ancestry") != ancestry:
            blockers.append({"task_id": task_id, "dependency": "MATRIX_IDENTITY",
                             "reason": "manifest chromosome/ancestry differs"})
            continue
        if item["fit_plan"] is None or item["fit_plan_sha256"] is None:
            blockers.append({"task_id": task_id, "dependency": "ADMITTED_FIT_PLAN",
                "reason": "missing reviewed numerical bounds, convergence criteria and uncertainty controls"})
            continue
        try:
            plan_path = absolute_file(item["fit_plan"], item["fit_plan_sha256"])
            plan = load(plan_path)
            validate_plan(plan, matrix, chromosome, ancestry)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            blockers.append({"task_id": task_id, "dependency": "ADMITTED_FIT_PLAN", "reason": str(error)})
            continue
        w_plan = plan["w_plan"]
        admitted.append({"id": task_id, "matrix_dir": str(Path(item["matrix_dir"]).resolve()),
            "matrix_manifest_sha256": item["matrix_manifest_sha256"],
            "fit_plan": str(plan_path.resolve()), "fit_plan_sha256": item["fit_plan_sha256"],
            "w_plan": str(Path(w_plan["path"]).resolve()), "w_plan_sha256": w_plan["sha256"],
            "publication": publication})
    expected = {f"chr{chromosome:02d}_{ancestry}" for chromosome in range(1, 23)
                for ancestry in ("AFR", "EUR", "NAM")}
    for task_id in sorted(expected - seen):
        blockers.append({"task_id": task_id, "dependency": "PUBLISHED_MATRIX_AND_FIT_PLAN",
                         "reason": "task absent from consumer specification"})
    if seen - expected:
        raise ValueError("unexpected K04 tasks present")
    code_dir = Path(code_dir)
    code = {}
    for name in CODE_NAMES:
        target = code_dir / name
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"consumer implementation missing: {target}")
        code[name] = sha256(target)
    status = "READY_FOR_ROOT_REVIEW_NOT_LAUNCHED" if not blockers and len(admitted) == 66 else "BLOCKED_NOT_LAUNCHABLE"
    return {"schema": "r02_k04_published_fit_consumer_admission_v1", "status": status,
        "launch_allowed": False, "evaluate_allowed": False,
        "matrix_rebuild_allowed": False, "deep_payload_authentication": bool(deep),
        "implementation_sha256": code, "tasks": admitted, "blockers": blockers,
        "scope": "FIT_SELECT_ONLY_NOT_EVALUATE_OR_BIOLOGICAL_VALIDATION"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deep-payload-authentication", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    result = inspect(args.spec, args.code_dir, args.deep_payload_authentication)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "blockers": len(result["blockers"])}, sort_keys=True))


if __name__ == "__main__":
    main()
