#!/usr/bin/env python3
"""Authenticate one K04 BUILD product and derive safe FIT/SELECT norm bounds.

This is a numerical preparation step.  It does not choose penalties, stopping
criteria, guard extensions or resources, fit a model, read EVALUATE, or admit a
FIT/SELECT plan.  The FIT bound is a conservative Frobenius upper bound for the
unweighted augmented operator ``[1, Z_fit]``.  The SELECT bound is a
conservative maximum augmented-row norm derived without allocating a
pairs-by-columns matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import numpy as np


ARMS = ("C", "CQ", "CQR")
INPUT_SCHEMA = "r02_k04_ccqcqr_inputs_v1"
INPUT_STATUS = "COMPLETE_REVIEWED_MODEL_INPUTS_NOT_MODEL_SELECTION_OR_BIOLOGICAL_VALIDATION"
OUTPUT_SCHEMA = "r02_k04_numerical_bounds_v1"
COMPLETE_SCHEMA = "r02_k04_numerical_bounds_66_complete_v1"
TASK = re.compile(r"chr(0[1-9]|1[0-9]|2[0-2])_(AFR|EUR|NAM)\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
FLOAT = np.dtype("float64")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    """Match the producer's shape/dtype/payload identity, distinct from file SHA."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps(list(value.shape)).encode())
    digest.update(value.dtype.str.encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Real JSON file required: " + str(path))
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "JSON object required: " + str(path))
    return value


def atomic_json(path, value):
    path = Path(path)
    require(not path.exists() and path.parent.is_dir(), "Fresh output in an existing directory required")
    temporary = path.with_name("." + path.name + ".tmp-" + str(os.getpid()))
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _up(value):
    value = float(value)
    require(math.isfinite(value), "Finite FP64 value required")
    result = float(np.nextafter(np.float64(value), np.float64(math.inf)))
    require(math.isfinite(result), "FP64 upper-bound operation overflowed")
    return result


def _down(value):
    value = float(value)
    require(math.isfinite(value), "Finite FP64 value required")
    result = float(np.nextafter(np.float64(value), np.float64(-math.inf)))
    require(math.isfinite(result), "FP64 lower-bound operation overflowed")
    return result


def add_up(left, right):
    return _up(float(left) + float(right))


def multiply_up(left, right):
    require(left >= 0 and right >= 0, "Upward product accepts nonnegative operands only")
    return _up(float(left) * float(right))


def square_up(value):
    return multiply_up(abs(float(value)), abs(float(value)))


def divide_up(numerator, denominator):
    require(numerator >= 0 and denominator > 0, "Positive denominator required")
    return _up(float(numerator) / _down(float(denominator)))


def sqrt_up(value):
    require(value >= 0, "Nonnegative square-root input required")
    return _up(math.sqrt(float(value)))


def _load_array(root, relative, expected_sha, *, kind=None, dimensions=None):
    relative_path = Path(relative)
    require(not relative_path.is_absolute() and ".." not in relative_path.parts,
            "Safe relative array path required")
    path = root / relative_path
    require(path.is_file() and not path.is_symlink(), "Real model-input array required: " + str(path))
    require(HEX.fullmatch(str(expected_sha)) is not None and sha256(path) == expected_sha,
            "Model-input array SHA-256 differs: " + str(path))
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(isinstance(value, np.ndarray) and (dimensions is None or value.ndim == dimensions),
            "Array dimensions differ: " + str(path))
    if kind is not None:
        require(value.dtype.kind in kind, "Array dtype differs: " + str(path))
    require(np.isfinite(value).all(), "Finite model-input array required: " + str(path))
    return value


def authenticate(root, manifest_sha):
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), "Real model-input directory required")
    require(HEX.fullmatch(str(manifest_sha)) is not None and
            sha256(root / "manifest.json") == manifest_sha,
            "Authenticated model-input manifest required")
    manifest = read_json(root / "manifest.json")
    require(manifest.get("schema") == INPUT_SCHEMA and manifest.get("status") == INPUT_STATUS,
            "Exact K04 BUILD schema/status required")
    require(manifest.get("evaluate_target_read") is False and
            manifest.get("evaluate_allowed") is False and
            manifest.get("model_fit") is False and manifest.get("model_selection") is False and
            manifest.get("penalty_chosen") is False,
            "Only pre-model FIT/SELECT inputs with EVALUATE closed are accepted")
    require(set(manifest.get("arms", {})) == set(ARMS) and
            manifest.get("rows_identical_across_arms") is True and
            manifest.get("weights_identical_across_arms") is True,
            "Exact shared-row C/CQ/CQR manifest required")
    bridge = manifest.get("solver_bridge", {})
    require(bridge.get("pair_design_operator_row_weights") == "UNIT_IDENTITY" and
            bridge.get("interval_solver_weights") == "EQUAL_ELIGIBLE_PAIRS_NORMALIZED_IN_SOLVER" and
            bridge.get("double_weighting") is False and
            bridge.get("operator_backend") in {"factorized", "factorized_admitted_pairs"},
            "Exact unweighted K04 solver bridge required")
    outputs = manifest.get("output_sha256")
    require(isinstance(outputs, dict) and outputs, "Complete payload SHA-256 inventory required")
    expected_outputs = {"points_C.npy", "points_CQ.npy"}
    expected_outputs |= {f"preprocessing_{arm}_{kind}.npy" for arm in ARMS
                         for kind in ("center", "scale", "quadratic_diagonal")}
    expected_outputs |= {f"{role}/{name}.npy" for role in ("FIT", "SELECT")
                         for name in ("person_i", "person_j", "lower", "upper",
                                      "opportunities", "extra_full")}
    require(set(outputs) == expected_outputs,
            "Exact K04 BUILD payload inventory required")
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    require(set(outputs) == actual, "Model-input payload inventory differs from manifest")
    for relative, digest in outputs.items():
        path = root / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and
                path.is_file() and not path.is_symlink() and HEX.fullmatch(str(digest)) is not None and
                sha256(path) == digest,
                "Model-input payload authentication failed: " + relative)
    require(sum((root / name).stat().st_size for name in outputs) == manifest.get("output_bytes"),
            "Model-input payload byte total differs")
    return root, manifest


def load_pair_operator(code_path, expected_sha):
    code_path = Path(code_path).resolve()
    require(code_path.is_file() and not code_path.is_symlink() and
            HEX.fullmatch(str(expected_sha)) is not None and sha256(code_path) == expected_sha,
            "Pinned PairDesign implementation required")
    spec = importlib.util.spec_from_file_location("_r02_k04_bounds_pair_operator", code_path)
    require(spec is not None and spec.loader is not None, "PairDesign implementation cannot be loaded")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(hasattr(module, "PairDesign"), "Pinned implementation has no PairDesign")
    return module, code_path


def _fit_frobenius_bound(diagonal, n_rows):
    """Upper-bound ||[1,Z]||_F from the saved FP64 mean squared columns.

    The producer formed each diagonal value by squaring n_rows standardized
    FP64 values, summing nonnegative terms and dividing by n_rows.  A worst-case
    sequential gamma bound is deliberately used even if NumPy performs a more
    accurate pairwise reduction.  ``n_rows + 3`` covers square, all additions,
    division and reconstruction; a smallest-subnormal allowance covers gradual
    underflow.  Every subsequent positive operation is rounded toward +infinity.
    """
    require(type(n_rows) is int and n_rows > 0 and diagonal.ndim == 1 and
            np.isfinite(diagonal).all() and np.all(diagonal >= 0),
            "Finite nonnegative FIT quadratic diagonal required")
    unit_roundoff = np.finfo(np.float64).eps / 2.
    operations = n_rows + 3
    require(operations * unit_roundoff < 1., "FIT reduction too large for FP64 gamma bound")
    gamma = (operations * unit_roundoff) / (1. - operations * unit_roundoff)
    denominator = _down(1. - gamma)
    subnormal = float(np.nextafter(np.float64(0.), np.float64(1.)))
    underflow = multiply_up(float(operations), subnormal)
    squared_sum_upper = 0.
    for observed_mean_square in diagonal:
        reconstructed = multiply_up(float(observed_mean_square), float(n_rows))
        column_upper = divide_up(add_up(reconstructed, underflow), denominator)
        squared_sum_upper = add_up(squared_sum_upper, column_upper)
    augmented_squared_upper = add_up(float(n_rows), squared_sum_upper)
    return sqrt_up(augmented_squared_upper), {
        "unit_roundoff": unit_roundoff,
        "worst_case_operations_per_column": operations,
        "gamma": gamma,
        "smallest_subnormal_allowance_per_operation": subnormal,
        "standardized_frobenius_squared_upper": squared_sum_upper,
        "augmented_frobenius_squared_upper": augmented_squared_upper,
    }


def _raw_person_feature_intervals(points):
    """Outward intervals for mean, absolute difference and product columns."""
    lower = np.min(points, axis=0).astype(np.float64, copy=False)
    upper = np.max(points, axis=0).astype(np.float64, copy=False)
    require(np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower <= upper),
            "Finite ordered person-feature extrema required")
    require(float(np.max(np.abs(points))) <= np.finfo(np.float64).max / 2.,
            "Person-feature mean can overflow FP64")
    mean_low = np.nextafter(lower, -np.inf)
    mean_high = np.nextafter(upper, np.inf)
    difference_low = np.zeros_like(lower)
    difference_high = np.nextafter(upper - lower, np.inf)
    products = np.stack((lower * lower, lower * upper, upper * lower, upper * upper))
    product_low = np.nextafter(np.min(products, axis=0), -np.inf)
    product_high = np.nextafter(np.max(products, axis=0), np.inf)
    require(np.isfinite(difference_high).all() and np.isfinite(products).all() and
            np.isfinite(product_low).all() and np.isfinite(product_high).all(),
            "Person-feature interval arithmetic overflowed")
    return np.concatenate((mean_low, difference_low, product_low)), \
        np.concatenate((mean_high, difference_high, product_high))


def _standardized_abs_upper(raw_low, raw_high, center, scale):
    require(raw_low.shape == raw_high.shape == center.shape == scale.shape and
            np.isfinite(raw_low).all() and np.isfinite(raw_high).all() and
            np.isfinite(center).all() and np.isfinite(scale).all() and
            np.all(raw_low <= raw_high) and np.all(scale > 0),
            "Finite ordered raw intervals and positive FIT scales required")
    low = np.nextafter(np.nextafter(raw_low - center, -np.inf) / scale, -np.inf)
    high = np.nextafter(np.nextafter(raw_high - center, np.inf) / scale, np.inf)
    result = np.nextafter(np.maximum(np.abs(low), np.abs(high)), np.inf)
    require(np.isfinite(result).all(), "Standardized feature bound overflowed")
    return result


def _select_row_bound(points, left, right, extra, center, scale):
    left, right = np.asarray(left), np.asarray(right)
    require(left.ndim == right.ndim == 1 and left.shape == right.shape and
            len(left) == len(extra) > 0 and left.dtype.kind in "iu" and right.dtype.kind in "iu" and
            np.all(left >= 0) and np.all(right >= 0) and
            np.all(left < len(points)) and np.all(right < len(points)),
            "Valid aligned SELECT pair indices required")
    select_people = np.unique(np.concatenate((left, right)))
    select_points = points[select_people]
    d = points.shape[1]
    require(center.shape == scale.shape == (3 * d + extra.shape[1],),
            "SELECT design/preprocessing width differs")
    raw_low, raw_high = _raw_person_feature_intervals(select_points)
    extra_low = np.min(extra, axis=0).astype(np.float64, copy=False)
    extra_high = np.max(extra, axis=0).astype(np.float64, copy=False)
    raw_low = np.concatenate((raw_low, np.nextafter(extra_low, -np.inf)))
    raw_high = np.concatenate((raw_high, np.nextafter(extra_high, np.inf)))
    coordinate_upper = _standardized_abs_upper(raw_low, raw_high, center, scale)
    squared_upper = 1.
    for value in coordinate_upper:
        squared_upper = add_up(squared_upper, square_up(float(value)))
    return sqrt_up(squared_upper), {
        "augmented_row_norm_squared_upper": squared_upper,
        "maximum_standardized_coordinate_upper": float(np.max(coordinate_upper)),
        "person_feature_columns": 3 * d,
        "paired_extra_columns": extra.shape[1],
        "select_people_used_for_person_extrema": len(select_people),
    }


def derive(model_inputs, manifest_sha, bounds_code_sha, pair_operator_code, pair_operator_sha,
           rows_per_chunk, max_working_mib, workers):
    require(type(rows_per_chunk) is int and rows_per_chunk > 0 and
            type(max_working_mib) is int and max_working_mib > 0 and
            type(workers) is int and workers > 0,
            "Explicit positive integer operator resources required")
    require(HEX.fullmatch(str(bounds_code_sha)) is not None and
            sha256(Path(__file__).resolve()) == bounds_code_sha,
            "Pinned numerical-bounds implementation required")
    root, manifest = authenticate(model_inputs, manifest_sha)
    operator_module, operator_path = load_pair_operator(pair_operator_code, pair_operator_sha)
    outputs = manifest["output_sha256"]
    arrays = {}
    for role in ("FIT", "SELECT"):
        arrays[(role, "left")] = _load_array(root, f"{role}/person_i.npy",
            outputs[f"{role}/person_i.npy"], kind="iu", dimensions=1)
        arrays[(role, "right")] = _load_array(root, f"{role}/person_j.npy",
            outputs[f"{role}/person_j.npy"], kind="iu", dimensions=1)
        extra_path = manifest["extra_storage"][role]["path"]
        arrays[(role, "extra")] = _load_array(root, extra_path, outputs[extra_path],
                                               kind="f", dimensions=2)
        require(arrays[(role, "left")].shape == arrays[(role, "right")].shape and
                len(arrays[(role, "left")]) == len(arrays[(role, "extra")]) > 0,
                role + " pair/extra row counts differ")
    n_people = manifest.get("cohort", {}).get("n_people")
    require(type(n_people) is int and n_people > 1, "Explicit cohort size required")
    results = {}
    for arm in ARMS:
        arm_manifest = manifest["arms"][arm]
        points_name = manifest["points_storage"][arm]
        points = _load_array(root, points_name, outputs[points_name], kind="f", dimensions=2)
        require(points.shape[0] == n_people and points.shape[1] == len(arm_manifest["points"]),
                arm + " person-feature shape differs")
        prefix = manifest["extra_storage"]["SELECT"]["arm_prefix_columns"][arm]
        require(prefix == len(arm_manifest["extra"]) and
                prefix <= arrays[("FIT", "extra")].shape[1] and
                prefix <= arrays[("SELECT", "extra")].shape[1],
                arm + " paired-extra prefix differs")
        preprocessing = {}
        for kind in ("center", "scale", "quadratic_diagonal"):
            name = f"preprocessing_{arm}_{kind}.npy"
            preprocessing[kind] = _load_array(root, name, outputs[name], kind="f", dimensions=1)
        center, scale, diagonal = (preprocessing[x] for x in
                                   ("center", "scale", "quadratic_diagonal"))
        columns = int(arm_manifest["n_design_columns"])
        require(center.shape == scale.shape == diagonal.shape == (columns,) and
                columns == 3 * points.shape[1] + prefix and np.all(scale > 0),
                arm + " preprocessing width/scale differs")
        declared = arm_manifest["preprocessing"]
        require(declared.get("fit_weighting") == "EQUAL_ELIGIBLE_PAIRS" and
                declared.get("n_columns") == columns and
                declared.get("center_sha256") == array_sha256(center) and
                declared.get("scale_sha256") == array_sha256(scale) and
                declared.get("quadratic_diagonal_sha256") == array_sha256(diagonal) and
                declared.get("L") == float(np.max(diagonal)) and
                declared.get("U") == float(math.fsum(map(float, diagonal))),
                arm + " preprocessing manifest binding differs")
        fit_n = len(arrays[("FIT", "left")]); select_n = len(arrays[("SELECT", "left")])
        fit_bound, fit_fp = _fit_frobenius_bound(diagonal, fit_n)
        select_bound, select_detail = _select_row_bound(
            points, arrays[("SELECT", "left")], arrays[("SELECT", "right")],
            arrays[("SELECT", "extra")][:, :prefix], center, scale)
        operator_shapes = {}
        for role in ("FIT", "SELECT"):
            design = operator_module.PairDesign(points, arrays[(role, "left")],
                arrays[(role, "right")], extra=arrays[(role, "extra")][:, :prefix],
                rows_per_chunk=rows_per_chunk, max_working_mib=max_working_mib,
                backend=manifest["solver_bridge"]["operator_backend"],
                preprocessing_backend="moments", workers=workers)
            require(design.shape == ((fit_n if role == "FIT" else select_n), columns),
                    arm + " actual PairDesign shape differs")
            actual_operator = design.operator(np.ones(design.shape[0], dtype=np.float64),
                                              center, scale)
            require(actual_operator.shape == design.shape and actual_operator.dtype == FLOAT,
                    arm + " actual PairDesign.operator contract differs")
            operator_shapes[role] = list(actual_operator.shape)
        results[arm] = {
            "n_columns": columns,
            "n_fit_rows": fit_n,
            "n_select_rows": select_n,
            "augmented_operator_norm_bound": fit_bound,
            "select_augmented_row_norm_bound": select_bound,
            "fit_bound_method": "FP64_OUTWARD_FROBENIUS_FROM_SAVED_QUADRATIC_DIAGONAL",
            "select_bound_method": "FP64_OUTWARD_PERSON_FEATURE_INTERVALS_PLUS_PAIRED_EXTRA_EXTREMA",
            "fit_fp64": fit_fp,
            "select_interval_details": select_detail,
            "pair_design_operator_constructed": True,
            "pair_design_operator_shapes": operator_shapes,
            "plan_fragment": {
                "augmented_operator_norm_bound": fit_bound,
                "select_augmented_row_norm_bound": select_bound,
            },
        }
    task_id = f"chr{int(manifest['target_chromosome']):02d}_{manifest['ancestry']}"
    require(TASK.fullmatch(task_id) is not None, "Canonical chromosome/ancestry identity required")
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "COMPLETE_SAFE_NUMERICAL_BOUNDS_NOT_FIT_PLAN_NOT_LAUNCHED",
        "task_id": task_id,
        "model_inputs_manifest_sha256": manifest_sha,
        "model_inputs_request_sha256": manifest["request_sha256"],
        "input_schema": INPUT_SCHEMA,
        "implementation_sha256": {
            "r02_k04_numerical_bounds.py": bounds_code_sha,
            "r02_pair_operator.py": pair_operator_sha,
        },
        "execution_pair_operator_path": str(operator_path),
        "authenticated_payload_sha256": outputs,
        "all_payload_files_sha256_verified": True,
        "resources_used_for_operator_contract_check": {
            "rows_per_chunk": rows_per_chunk,
            "max_working_mib": max_working_mib,
            "workers": workers,
            "operator_backend": manifest["solver_bridge"]["operator_backend"],
        },
        "arms": results,
        "proof_scope": {
            "fit": "spectral_norm([1,Z_FIT]) <= Frobenius_norm([1,Z_FIT]); saved diagonal is sum(Z^2)/n with explicit worst-case FP64 reduction and outward-rounding allowance",
            "select": "every mean/absolute-difference/product feature lies in the outward interval induced by person-level extrema; paired extra columns use their actual SELECT extrema; standardized coordinate bounds imply the row Euclidean bound",
            "power_iteration_used": False,
            "dense_pairs_by_columns_materialized": False,
        },
        "evaluate_read": False,
        "fit_plan_admitted": False,
        "parameters_chosen": False,
        "models_fitted": 0,
        "model_selection": False,
        "biological_validation": False,
    }


def gate(bounds_dir):
    paths = sorted(Path(bounds_dir).glob("*.json"))
    rows = [read_json(path) for path in paths]
    expected = {f"chr{chromosome:02d}_{ancestry}" for chromosome in range(1, 23)
                for ancestry in ("AFR", "EUR", "NAM")}
    require(len(rows) == 66 and {row.get("task_id") for row in rows} == expected,
            "Exactly one numerical-bounds artifact per 22x3 task required")
    require(all(row.get("schema") == OUTPUT_SCHEMA and
                row.get("status") == "COMPLETE_SAFE_NUMERICAL_BOUNDS_NOT_FIT_PLAN_NOT_LAUNCHED" and
                row.get("evaluate_read") is False and row.get("models_fitted") == 0 and
                row.get("fit_plan_admitted") is False and set(row.get("arms", {})) == set(ARMS)
                for row in rows),
            "Complete safe pre-plan bounds artifacts required")
    return {
        "schema": COMPLETE_SCHEMA,
        "status": "COMPLETE_66_SAFE_BOUNDS_NOT_FIT_PLAN_NOT_LAUNCHED",
        "tasks": 66,
        "bounds_sha256": {row["task_id"]: sha256(path) for path, row in zip(paths, rows)},
        "model_inputs_manifest_sha256": {
            row["task_id"]: row["model_inputs_manifest_sha256"] for row in rows},
        "evaluate_read": False,
        "fit_plan_admitted": False,
        "models_fitted": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("derive")
    one.add_argument("--model-inputs", required=True)
    one.add_argument("--model-inputs-manifest-sha256", required=True)
    one.add_argument("--bounds-code-sha256", required=True)
    one.add_argument("--pair-operator-code", required=True)
    one.add_argument("--pair-operator-sha256", required=True)
    one.add_argument("--rows-per-chunk", required=True, type=int)
    one.add_argument("--max-working-mib", required=True, type=int)
    one.add_argument("--workers", required=True, type=int)
    one.add_argument("--output", required=True)
    complete = sub.add_parser("gate")
    complete.add_argument("--bounds-dir", required=True)
    complete.add_argument("--output", required=True)
    args = parser.parse_args(); os.umask(0o077)
    if args.command == "derive":
        result = derive(args.model_inputs, args.model_inputs_manifest_sha256,
            args.bounds_code_sha256, args.pair_operator_code, args.pair_operator_sha256,
            args.rows_per_chunk, args.max_working_mib, args.workers)
    else:
        result = gate(args.bounds_dir)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("schema", "status")}, sort_keys=True))


if __name__ == "__main__":
    main()
