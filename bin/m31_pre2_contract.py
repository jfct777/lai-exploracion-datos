#!/usr/bin/env python3
"""Fail-closed validator for the immutable M31 PRE2 preregistration.

This command validates the contract only.  It never reads genomic inputs and
does not authorize or launch a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SHA256 = "fd2d7b6d287913636be6e83ad542a40ffbc26c961d769e9c181955d89efa76bf"
EXPECTED_SEMANTIC_SHA256 = "cd9ec3c203f6806b575f4467cb904c0aaa70fd5b2f26bc59746b1e6c7d6bc5ac"
EXPECTED_SCHEMA = "2.0.0"
EXPECTED_EXPERIMENT = "M31_ORDERED_LINEAR_DEV_PRE2"
EXPECTED_STATUS = "PREREGISTERED_NOT_RUN"


class ContractError(ValueError):
    """Raised when the immutable PRE2 contract is invalid or has drifted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_json_values(value: Any, label: str = "contract") -> None:
    """Reject non-finite numbers, booleans masquerading as numbers and unsafe keys."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            require(isinstance(key, str) and key, f"{label} contains an invalid key")
            require(".." not in key and not key.startswith("/"), f"{label}.{key} has an unsafe key")
            validate_json_values(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_values(item, f"{label}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"{label} must be finite")
    elif value is not None:
        require(isinstance(value, (str, int, bool)), f"{label} has unsupported type")


def semantic_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate parsed semantics independently of the immutable file hash."""
    require(isinstance(payload, dict), "contract must be a JSON object")
    validate_json_values(payload)
    require(semantic_sha256(payload) == EXPECTED_SEMANTIC_SHA256, "PRE2 contract semantics drifted")
    require(payload.get("schema_version") == EXPECTED_SCHEMA, "schema_version drifted")
    require(payload.get("experiment_id") == EXPECTED_EXPERIMENT, "experiment_id drifted")
    require(payload.get("status") == EXPECTED_STATUS, "status drifted")
    require(payload.get("protocol_class") == "DEVELOPMENT_PRE2_ADAPTIVE_TO_PRE1", "adaptive protocol class drifted")
    require(payload.get("scope") == "chr22_root17_development_root18_one_way_evaluation_no_validation", "scope drifted")
    require(payload.get("estimand") == "if_L_is_guarded_predictive_increment_of_D_over_F0_and_L; otherwise_combined_D_candidate_vs_F0_only", "estimand drifted")
    require(payload["implementation"]["contract_only"] is True, "contract-only gate drifted")
    require(payload["implementation"]["real_run_authorized"] is False, "contract must not authorize a real run")
    require(payload["prior_evidence"]["pre1_is_immutable"] is True, "PRE1 immutability drifted")
    require(payload["roots"]["reciprocal_direction_forbidden"] is True, "one-way rule drifted")
    require(payload["arms"]["scientific"] == ["F0", "L", "D"], "scientific arms drifted")
    require(payload["arms"]["C_is_scientific_comparator"] is False, "C comparator role drifted")
    require(payload["arms"]["excluded"] == ["H", "DSHAM", "HSHAM"], "excluded arms drifted")
    require(payload["selection"]["tau"] == 1e-15, "computational tolerance drifted")
    require(payload["selection"]["tau_role"] == "computational_tolerance_only_not_SESOI", "tolerance role drifted")
    require(payload["selection"]["development_minimum_delta_F1"] == 0.01, "material F1 threshold drifted")
    require(payload["selection"]["development_minimum_delta_F1_sensitivities"] == [0.005, 0.02], "material-threshold sensitivities drifted")
    require(payload["selection"]["empty_guarded_set"] == "NO_FALLBACK", "empty-set rule drifted")
    require(payload["selection"]["lexicographic_order"] == ["-F1", "FT", "MAE", "Brier", "boundary_weight", "-alpha"], "selection order drifted")
    require(payload["root18_decision"]["required"] == [
        "F1_D>=F1_each_applicable_comparator+0.01",
        "MAE_D<=MAE_each_applicable_comparator+tau",
        "FT_D<=FT_each_applicable_comparator+tau",
        "MAE_AFR_D<=MAE_AFR_each_applicable_comparator+tau",
        "MAE_EUR_D<=MAE_EUR_each_applicable_comparator+tau",
        "MAE_ASIA_D<=MAE_ASIA_each_applicable_comparator+tau",
        "MAE_truth_present_AFR_D<=MAE_truth_present_AFR_each_applicable_comparator+tau",
        "MAE_truth_present_EUR_D<=MAE_truth_present_EUR_each_applicable_comparator+tau",
        "MAE_truth_present_ASIA_D<=MAE_truth_present_ASIA_each_applicable_comparator+tau",
    ], "root18 ancestry safeguards drifted")
    require(payload["uncertainty"] == {
        "unit": "complete_diploid_individual",
        "bootstrap_replicates": 10000,
        "bootstrap_type": "paired_resample_individual_count_bundles_then_reconstruct_global_metrics",
        "role": "descriptive_only_not_gate",
    }, "uncertainty contract drifted")
    require(payload["parallelism"]["workers_primary"] == 4, "primary workers drifted")
    require(payload["parallelism"]["workers_screen"] == [1, 4, 8], "worker screen drifted")
    require(all(value == 1 for value in payload["parallelism"]["thread_limits"].values()), "thread limits must all equal one")
    require("rare_support_mechanism_without_DSHAM" in payload["claims_excluded"], "mechanism exclusion is missing")
    require("independent_replication" in payload["claims_excluded"], "independent-replication exclusion is missing")
    return payload


def validate_contract(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"contract is absent: {path}")
    actual_sha256 = sha256_file(path)
    require(actual_sha256 == EXPECTED_SHA256, "immutable PRE2 contract bytes differ from preregistration")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ContractError(f"non-finite JSON constant: {token}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"contract is not valid UTF-8 JSON: {error}") from error
    return validate_payload(payload)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = validate_contract(args.contract)
    report = {
        "schema_version": payload["schema_version"],
        "experiment_id": payload["experiment_id"],
        "status": "PASS_CONTRACT_ONLY_NO_DATA",
        "contract_sha256": sha256_file(args.contract),
        "real_run_authorized": False,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
