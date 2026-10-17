#!/usr/bin/env python3
"""Fit and SELECT one reviewed K04 C/CQ/CQR finite family.

This consumer never reads EVALUATE.  Candidate penalties and numerical limits
come from an authenticated, human-reviewed plan derived from W; this module
does not choose either.  A family can select only after every candidate has
converged and no numerically plausible candidate touches an outer guard.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from r02_interval_ridge import solve_interval_ridge
from r02_k04_candidate_store import CandidateStore, atomic_json, atomic_model
from r02_k04_precision_policy import fp64_intercept_candidate, selection_decision
from r02_k04_model_bridge import EXPECTED_DESIGN_COLUMNS, EXTRA_COLUMNS, OUTPUT_SCHEMA, _make_design


ARMS = ("C", "CQ", "CQR")
W_SHA256 = "6540d76c71b41de0fea50d6047658fa8e92d575435f0ee73aa4f2fc1d7c3fe1e"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_array(path, expected_hash):
    require(sha256(path) == expected_hash, "Model-input array hash differs: " + str(path))
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    require(value.dtype.kind in "iuf" and np.isfinite(value).all(), "Finite numeric array required")
    return value


def interval_risk(prediction, lower, upper):
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 0., 1.)
    return float(np.mean(np.maximum((prediction-lower)**2, (prediction-upper)**2), dtype=np.float64))


def intercept_only(lower, upper):
    """Finite-candidate FP64 solution; no outward numerical error certificate."""
    midpoint = (np.asarray(lower, dtype=np.float64)+np.asarray(upper, dtype=np.float64))/2.
    half = (np.asarray(upper, dtype=np.float64)-np.asarray(lower, dtype=np.float64))/2.
    order = np.argsort(midpoint, kind="mergesort")
    midpoint, half = midpoint[order], half[order]
    n = len(midpoint); mean_midpoint = math.fsum(map(float, midpoint))/n
    total_half = math.fsum(map(float, half)); below = 0.; start = 0
    while start < n:
        stop = start+1
        while stop < n and midpoint[stop] == midpoint[start]:
            stop += 1
        value = float(midpoint[start]); equal = math.fsum(map(float, half[start:stop]))
        above = total_half-below-equal
        lower_gradient = value-mean_midpoint+(below-equal-above)/n
        upper_gradient = value-mean_midpoint+(below+equal-above)/n
        if lower_gradient <= 0 <= upper_gradient:
            return value, interval_risk(np.full(n, value), lower, upper)
        below += equal
        if stop < n:
            root = mean_midpoint-(below-(total_half-below))/n
            if value < root < float(midpoint[stop]):
                return root, interval_risk(np.full(n, root), lower, upper)
        start = stop
    raise ValueError("Intercept-only optimum not located in [0,1]")


def w_penalty_grid(L, U, F0, guard_extensions_low=0, guard_extensions_high=0):
    require(all(type(x) is int and x >= 0 for x in
                (guard_extensions_low, guard_extensions_high)) and 0 <= L <= U and 0 <= F0 <= 0.25,
            "Authenticated L/U/F0 and integer W guard extensions required")
    if U == 0 or F0 == 0:
        return []
    require(L > 0, "Positive L required for a nonconstant design")
    low = 0.01 * L / 0.99
    high = U * F0 / 0.01**2
    require(math.isfinite(low) and math.isfinite(high) and high >= low,
            "Finite ordered W penalty anchors required")
    start = math.floor(math.log10(low))-1-guard_extensions_low
    stop = math.ceil(math.log10(high))+1+guard_extensions_high
    return [10.**(start+i*0.5) for i in range(2*(stop-start)+1)]


def make_design(root, role, arm, manifest, resources):
    points_name = manifest["points_storage"][arm]
    points = load_array(root / points_name, manifest["output_sha256"][points_name])
    names = ["person_i.npy", "person_j.npy", "extra_full.npy"]
    left, right, extra = [load_array(root / role / name,
        manifest["output_sha256"][role + "/" + name]) for name in names]
    design = _make_design(points, left, right,
                          np.asarray(extra[:, :len(EXTRA_COLUMNS[arm])]), resources)
    require(design.shape[1] == EXPECTED_DESIGN_COLUMNS[arm], "K04 arm width differs")
    center_name, scale_name = f"preprocessing_{arm}_center.npy", f"preprocessing_{arm}_scale.npy"
    diagonal_name = f"preprocessing_{arm}_quadratic_diagonal.npy"
    center = load_array(root / center_name, manifest["output_sha256"][center_name])
    scale = load_array(root / scale_name, manifest["output_sha256"][scale_name])
    diagonal = load_array(root / diagonal_name, manifest["output_sha256"][diagonal_name])
    return design, np.asarray(center), np.asarray(scale), np.asarray(diagonal)


def continuation_identity(manifest_sha, manifest, plan):
    """Bind data, code and immutable scientific/numerical configuration.

    Only the two stopping budgets are excluded.  Their monotone amendment is
    recorded separately by CandidateStore and requires an explicit CLI flag.
    """
    scientific_plan = copy.deepcopy(plan)
    numerical = scientific_plan["numerical"]
    budgets = {name: numerical.pop(name) for name in
               ("max_iterations", "max_seconds_per_candidate")}
    scientific_plan["w_plan"] = {"sha256": scientific_plan["w_plan"]["sha256"]}
    code = {}
    for name in ("r02_k04_fit_select.py", "r02_k04_candidate_store.py",
                 "r02_k04_precision_policy.py",
                 "r02_interval_ridge.py", "r02_k04_model_bridge.py",
                 "r02_k04_predictor_inputs.py", "r02_pair_operator.py"):
        path = Path(__file__).resolve().with_name(name)
        require(path.is_file() and not path.is_symlink(), "Required K04 code file missing: " + name)
        code[name] = sha256(path)
    identity = {"schema": "r02_k04_fit_select_continuation_identity_v1",
        "model_inputs_manifest_sha256": manifest_sha,
        "model_inputs_request_sha256": manifest["request_sha256"],
        "target_chromosome": manifest["target_chromosome"],
        "ancestry": manifest["ancestry"], "scientific_plan": scientific_plan,
        "implementation_sha256": code}
    return identity, budgets


def fit_select(model_inputs, manifest_sha, plan_path, plan_sha, output,
               state_root=None, allow_compatible_budget_extension=False):
    root = Path(model_inputs).resolve(); plan_path = Path(plan_path).resolve()
    require(root.is_dir() and sha256(root / "manifest.json") == manifest_sha,
            "Authenticated model-input manifest required")
    require(sha256(plan_path) == plan_sha, "Fit plan hash differs")
    manifest, plan = read(root / "manifest.json"), read(plan_path)
    require(manifest.get("schema") == OUTPUT_SCHEMA and
            manifest.get("status") == "COMPLETE_REVIEWED_MODEL_INPUTS_NOT_MODEL_SELECTION_OR_BIOLOGICAL_VALIDATION" and
            not manifest.get("evaluate_allowed") and not manifest.get("model_fit"),
            "Reviewed FIT/SELECT-only producer output required")
    required = {"schema", "operation", "w_plan", "bridge_request_sha256",
                "target_chromosome", "ancestry", "resources", "numerical", "arms"}
    require(set(plan) == required and plan["schema"] == "r02_k04_fit_select_plan_v1" and
            plan["operation"] == "FIT_SELECT_COMPLETE_W_FINITE_FAMILY" and
            plan["w_plan"].get("sha256") == W_SHA256 and
            sha256(plan["w_plan"]["path"]) == W_SHA256 and
            plan["bridge_request_sha256"] == manifest.get("request_sha256") and
            str(plan["target_chromosome"]) == str(manifest["target_chromosome"]) and
            plan["ancestry"] == manifest["ancestry"] and set(plan["arms"]) == set(ARMS),
            "Fit plan/W/model-input identity differs")
    resources = plan["resources"]
    resource_fields = {"rows_per_chunk", "max_working_mib", "workers"}
    require(set(resources) in (resource_fields, resource_fields | {"operator_backend"}) and
            all(type(resources[x]) is int and resources[x] > 0 for x in resource_fields) and
            resources.get("operator_backend", "factorized") in
                {"factorized", "factorized_admitted_pairs"} and
            resources.get("operator_backend", "factorized") ==
                manifest["solver_bridge"].get("operator_backend", "factorized"),
            "Explicit positive operator resources required")
    numerical = plan["numerical"]
    expected_num = {"step_fraction", "max_iterations", "max_seconds_per_candidate",
                    "check_every", "gap_absolute_tolerance", "gap_relative_tolerance",
                    "dual_feasibility_tolerance", "centering_tolerance",
                    "checkpoint_every"}
    require(set(numerical) == expected_num and all(isinstance(x, (int, float)) and
            math.isfinite(x) and x > 0 for x in numerical.values()),
            "Measured explicit numerical limits required")
    require(all(type(numerical[x]) is int for x in
                ("max_iterations", "check_every", "checkpoint_every")) and
            0 < numerical["step_fraction"] < 1,
            "Integer iteration controls and step fraction in (0,1) required")
    identity, budgets = continuation_identity(manifest_sha, manifest, plan)
    output = Path(output).resolve()
    state_root = (output.parent / ("." + output.name + ".candidate-state")
                  if state_root is None else Path(state_root).resolve())
    with CandidateStore(state_root, identity, plan_sha, budgets,
                        allow_compatible_budget_extension) as store:
        output = store.prepare_output(output)
        return _fit_select_locked(root, manifest_sha, plan_sha, manifest, plan,
                                  resources, numerical, output, store)


def _fit_select_locked(root, manifest_sha, plan_sha, manifest, plan, resources,
                       numerical, output, store):
    lower = {role: load_array(root / role / "lower.npy",
        manifest["output_sha256"][role + "/lower.npy"]) for role in ("FIT", "SELECT")}
    upper = {role: load_array(root / role / "upper.npy",
        manifest["output_sha256"][role + "/upper.npy"]) for role in ("FIT", "SELECT")}
    results = {}
    for arm in ARMS:
        arm_plan = plan["arms"][arm]
        intercept, intercept_loss = intercept_only(lower["FIT"], upper["FIT"])
        require(set(arm_plan) == {"augmented_operator_norm_bound",
                "select_augmented_row_norm_bound", "guard_extensions_low", "guard_extensions_high"} and
                all(math.isfinite(arm_plan[x]) and arm_plan[x] > 0 for x in
                    ("augmented_operator_norm_bound", "select_augmented_row_norm_bound")) and
                all(type(arm_plan[x]) is int and arm_plan[x] >= 0 for x in
                    ("guard_extensions_low", "guard_extensions_high")),
                "Measured bounds and integer W guard-extension state required")
        fit_design, center, scale, diagonal = make_design(root, "FIT", arm, manifest, resources)
        select_design, select_center, select_scale, select_diagonal = make_design(root, "SELECT", arm, manifest, resources)
        observed_L = float(np.max(diagonal)); observed_U = float(math.fsum(map(float, diagonal)))
        require(np.array_equal(center, select_center) and np.array_equal(scale, select_scale) and
                np.array_equal(diagonal, select_diagonal) and
                manifest["arms"][arm]["preprocessing"].get("L") == observed_L and
                manifest["arms"][arm]["preprocessing"].get("U") == observed_U,
                "FIT preprocessing differs in SELECT adapter")
        penalties = w_penalty_grid(observed_L, observed_U, intercept_loss,
            arm_plan["guard_extensions_low"], arm_plan["guard_extensions_high"])
        fit_operator = fit_design.operator(np.ones(fit_design.shape[0]), center, scale)
        select_operator = select_design.operator(np.ones(select_design.shape[0]), center, scale)
        candidates = []
        # Historical receipts labelled this FP64 scalar exact with zero error.
        # Preserve them on disk, but never reuse them as a numerical certificate.
        candidate = fp64_intercept_candidate(intercept, fit_design.shape[1])
        candidates.append(candidate)
        for penalty in penalties:
            candidate_id = "LAMBDA_" + hashlib.sha256(float(penalty).hex().encode()).hexdigest()[:16]
            candidate = store.load_complete(arm, candidate_id, float(penalty), fit_design.shape[1])
            if candidate is None:
                checkpoint = store.checkpoint_path(arm, candidate_id)
                design_identity = hashlib.sha256((store.identity_sha256 + arm).encode()).hexdigest()
                result = solve_interval_ridge(fit_operator, lower["FIT"], upper["FIT"],
                    np.ones(fit_design.shape[0]), penalty=float(penalty), design_identity=design_identity,
                    augmented_operator_norm_bound=float(arm_plan["augmented_operator_norm_bound"]),
                    step_fraction=float(numerical["step_fraction"]),
                    max_iterations=int(numerical["max_iterations"]),
                    max_seconds=float(numerical["max_seconds_per_candidate"]),
                    check_every=int(numerical["check_every"]),
                    gap_absolute_tolerance=float(numerical["gap_absolute_tolerance"]),
                    gap_relative_tolerance=float(numerical["gap_relative_tolerance"]),
                    dual_feasibility_tolerance=float(numerical["dual_feasibility_tolerance"]),
                    centering_tolerance=float(numerical["centering_tolerance"]),
                    checkpoint_every=int(numerical["checkpoint_every"]), checkpoint_path=checkpoint,
                    resume=checkpoint.is_file())
                candidate = {"id": candidate_id, "penalty": float(penalty),
                    "status": result.status, "intercept": result.intercept,
                    "coefficients": result.coefficients,
                    "coefficient_error_bound": result.diagnostics["coefficient_error_bound_exact_arithmetic"],
                    "iterations": result.iterations, "elapsed_seconds": result.elapsed_seconds,
                    "diagnostics": result.diagnostics, "provenance": result.provenance}
                if (candidate["status"].startswith("CONVERGED") and
                        candidate["coefficient_error_bound"] is not None and
                        math.isfinite(candidate["coefficient_error_bound"])):
                    store.save_complete(arm, candidate)
                    candidate = store.load_complete(arm, candidate_id, float(penalty),
                                                    fit_design.shape[1])
                    candidate["reused_completed_candidate"] = False
            candidates.append(candidate)
        summaries = []
        for candidate in candidates:
            bound = candidate["coefficient_error_bound"]
            model_path = output / f"model_{arm}_{candidate['id']}.npz"
            if candidate.get("reused_completed_candidate"):
                model_sha = store.materialize_model(arm, candidate, model_path)
            else:
                atomic_model(model_path, candidate["intercept"], candidate["coefficients"])
                model_sha = sha256(model_path)
            if not candidate["status"].startswith("CONVERGED") or bound is None or not math.isfinite(bound):
                summaries.append({"id": candidate["id"], "penalty": candidate["penalty"],
                    "status": candidate["status"], "select_loss": None,
                    "select_loss_lower": None, "select_loss_upper": None,
                    "model_sha256": model_sha,
                    "reused_completed_candidate": False})
                continue
            require(bound >= 0, "Nonnegative coefficient error bound required for SELECT")
            prediction = candidate["intercept"] + select_operator @ candidate["coefficients"]
            risk = interval_risk(prediction, lower["SELECT"], upper["SELECT"])
            delta = bound * float(arm_plan["select_augmented_row_norm_bound"])
            error = min(1., 2.*delta + delta*delta)
            summaries.append({"id": candidate["id"], "penalty": candidate["penalty"],
                "status": candidate["status"], "select_loss": risk,
                "select_loss_lower": max(0., risk-error), "select_loss_upper": min(1., risk+error),
                "coefficient_error_bound_exact_arithmetic": bound,
                "iterations": candidate.get("iterations", 0), "model_sha256": model_sha,
                "reused_completed_candidate": candidate.get("reused_completed_candidate", False)})
        status, plausible = selection_decision(summaries, penalties)
        results[arm] = {"status": status, "plausible": plausible,
                        "selected": plausible[0] if status == "SELECTED_UNIQUE" else None,
                        "policy_derivation": {"w_plan_sha256": W_SHA256, "L": observed_L,
                            "U": observed_U, "F0": intercept_loss, "epsilon_scale": 0.01,
                            "epsilon_prediction": 0.01, "spacing_log10": 0.5,
                            "guard_decades": 1,
                            "guard_extensions_low": arm_plan["guard_extensions_low"],
                            "guard_extensions_high": arm_plan["guard_extensions_high"]},
                        "candidates": summaries}
    manifest_out = {"schema": "r02_k04_fit_select_result_v1", "status":
        "COMPLETE_SELECTION" if all(x["status"] == "SELECTED_UNIQUE" for x in results.values()) else
        "INCOMPLETE_NO_SELECTION_RELEASED", "model_inputs_manifest_sha256": manifest_sha,
        "fit_plan_sha256": plan_sha, "w_plan_sha256": W_SHA256,
        "target_chromosome": manifest["target_chromosome"], "ancestry": manifest["ancestry"],
        "arms": results, "evaluate_read": False, "evaluate_allowed": False,
        "fit_plus_select_refit": False, "biological_validation": False,
        "continuation": {"schema": "r02_k04_fit_select_continuation_v1",
            "candidate_state_root": str(store.root),
            "immutable_identity_sha256": store.identity_sha256,
            "fit_plan_sha256": plan_sha,
            "reused_completed_candidates": sum(
                int(x.get("reused_completed_candidate", False))
                for arm in results.values() for x in arm["candidates"])}}
    atomic_json(output / "manifest.json", manifest_out)
    return manifest_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-inputs", required=True); p.add_argument("--model-inputs-manifest-sha256", required=True)
    p.add_argument("--fit-plan", required=True); p.add_argument("--fit-plan-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--state-root", required=True)
    p.add_argument("--allow-compatible-budget-extension", action="store_true")
    a = p.parse_args(); os.umask(0o077)
    result = fit_select(a.model_inputs, a.model_inputs_manifest_sha256, a.fit_plan,
                        a.fit_plan_sha256, a.output, a.state_root,
                        a.allow_compatible_budget_extension)
    print(json.dumps({"schema": result["schema"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
