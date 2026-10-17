#!/usr/bin/env python3
"""Load one complete K02/K07 bundle and construct real SCORE designs without fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import r02_structure_increment as core


def run(bundle, numerics, pair_operator, pair_preconditioner):
    bundle = Path(bundle)
    contract_path = bundle / "contract.json"
    evaluation_path = bundle / "evaluation.json"
    contract = core.load_json(contract_path)
    evaluation = core.load_json(evaluation_path)
    store = core.BlockBundles(bundle, contract, evaluation)
    core.require(len(store.blocks) == 1, "Preflight requires one complete block bundle")
    values, scoped = store.load(0)
    persons, roles, blocks, markers, genotypes, features, pairs = values
    block = blocks[0]
    paired = pairs[block["block_id"]]
    genotypes = core._block_genotypes(block, markers, genotypes)
    components = [person["component"] for person in persons]
    targets = {}
    for role in ("FIT", "SELECT"):
        i, j, y, n, info = core.target_pairs(
            genotypes, roles["FIT"], roles[role], components,
            scoped["min_fit_called"], scoped["min_pair_markers"], scoped["max_pairs"])
        weights, _ = core.component_pair_weights(i, j, components)
        targets[role] = i, j, y, n, weights, info
    backend = core.load_pair_operator_backend(
        numerics, core.sha(numerics), pair_operator, core.sha(pair_operator),
        pair_preconditioner, core.sha(pair_preconditioner))
    fi, fj, fy, fn, fw, fit_info = targets["FIT"]
    si, sj, sy, sn, sw, select_info = targets["SELECT"]
    designs = {}
    for arm in core.ARMS:
        matrix = features[f"{block['block_id']}_{arm}"]
        fit_design, imputation = backend.design(
            matrix, fi, fj, paired, arm, roles["FIT"], require_fit_support=True)
        select_design, reused = backend.design(
            matrix, si, sj, paired, arm, roles["SELECT"], imputation=imputation)
        core.require(np.array_equal(reused, imputation), "FIT imputation was not frozen for SELECT")
        designs[arm] = {
            "fit_shape": list(fit_design.shape), "select_shape": list(select_design.shape),
            "fit_weight_sum": float(fw.sum()), "select_weight_sum": float(sw.sum()),
            "imputation_columns": int(len(imputation)),
            "operator_working_bytes_bound": int(fit_design.working_bytes_bound),
        }
    return {
        "schema": "r02_pair_complete_bundle_preflight_v1",
        "status": "COMPLETE_NO_MODEL_FIT",
        "view_id": block["block_id"] + "__" + scoped["exclusion_scope"],
        "roles": {key: int(len(value)) for key, value in roles.items()},
        "ridge_alphas": scoped["ridge_alphas"],
        "fit_pairs": int(len(fy)), "select_pairs": int(len(sy)),
        "fit_eligible_markers": int(fit_info["eligible_markers"]),
        "select_eligible_markers": int(select_info["eligible_markers"]),
        "designs": designs, "evaluate_targets_computed": False,
        "models_fitted": 0, "evaluate_allowed": False,
        "contract_sha256": core.sha(contract_path),
        "evaluation_sha256": core.sha(evaluation_path),
        "numerical_backend": backend.metadata(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--numerics", required=True)
    parser.add_argument("--pair-operator", required=True)
    parser.add_argument("--pair-preconditioner", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = run(args.bundle, args.numerics, args.pair_operator, args.pair_preconditioner)
    Path(args.receipt).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))
