#!/usr/bin/env python3
"""Authenticate the frozen M38B contract and its pre-outcome amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from m33_safe_bridge_core import write_exclusive_json


LOAD_BEARING_SOURCE_NAMES = frozenset({
    "m33_safe_bridge_core.py", "m34_generate_mosaics.py", "m34_parse_flare_truth.py",
    "m37_trace_core.py",
    "m37_bind_marker_axis.py", "m37_trace_materialize.py", "m37_trace_train.py",
    "m38b_oof_core.py", "m38b_validate_model_contract.py", "m38b_subset_factors.py",
    "m38b_bind_marker_axis.py", "m38b_strict_sham.py", "m38b_materialize_arm.py",
    "m38b_make_folds.py", "m38b_positive_control.py", "m38b_partition_fold.py",
    "m38b_train_fold.py", "m38b_collect_oof.py", "m38b_pack_scoring.py",
    "m38b_score_oof.py", "m38b_score_positive.py", "m38b_decide.py",
    "m38b_r0_oof_contract.json", "m38b_r0_oof_amendment_1.json",
    "m38b_r0_oof_amendment_2.json", "m38b_r0_oof_models.config",
    "38B_OOF_MODELS.nf", "m38b_r0_oof_models.nf",
})


class M38BModelContractError(ValueError):
    """Raised when the executable model contract differs from preregistration."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BModelContractError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"contract input is not a file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(contract: Path, amendment: Path, amendment_2: Path,
             expected_contract_sha256: str, expected_amendment_sha256: str,
             expected_amendment_2_sha256: str, output: Path,
             source_files: list[Path]) -> dict[str, Any]:
    contract_sha, amendment_sha = sha256(contract), sha256(amendment)
    amendment_2_sha = sha256(amendment_2)
    require(len(source_files) == len(LOAD_BEARING_SOURCE_NAMES)
            and {path.name for path in source_files} == LOAD_BEARING_SOURCE_NAMES,
            "M38B load-bearing source set differs")
    source_manifest = [
        {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(source_files, key=lambda item: item.name)
    ]
    manifest_bytes = json.dumps(
        source_manifest, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    source_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    require(contract_sha == expected_contract_sha256,
            "M38B base contract SHA-256 differs")
    require(amendment_sha == expected_amendment_sha256,
            "M38B amendment SHA-256 differs")
    require(amendment_2_sha == expected_amendment_2_sha256,
            "M38B amendment 2 SHA-256 differs")
    base = json.loads(contract.read_text(encoding="utf-8"))
    delta = json.loads(amendment.read_text(encoding="utf-8"))
    delta_2 = json.loads(amendment_2.read_text(encoding="utf-8"))
    require(
        base.get("experiment_id") == "M38B_S660_INCREMENTAL_LAI_CHR22_R0_FIT"
        and base.get("status") == "PREREGISTERED_AMENDED_BEFORE_OUTCOME_ACCESS"
        and base.get("claim_scope", {}).get("target_partition") == "FIT_ONLY"
        and base.get("claim_scope", {}).get("target_people") == 96
        and base.get("claim_scope", {}).get("valid_opened") is False
        and base.get("claim_scope", {}).get("test_opened") is False,
        "M38B base contract scope differs",
    )
    require(
        delta.get("experiment_id") == base.get("experiment_id")
        and delta.get("amendment_id") == "M38B_AMENDMENT_1_MATCHED_TCN_RD"
        and delta.get("base_contract_sha256") == contract_sha
        and delta.get("timing") == "BEFORE_MODEL_TRAINING_AND_BEFORE_MODEL_OUTCOME_ACCESS"
        and "same folds" in delta.get("delta", {}).get("tcn_RD", "")
        and "receipt-bound arm" in delta.get("delta", {}).get("arm_binding", ""),
        "M38B model amendment differs",
    )
    require(
        delta_2.get("experiment_id") == base.get("experiment_id")
        and delta_2.get("amendment_id") == "M38B_AMENDMENT_2_TCN_OFF_AND_PURE_POSITIVE_CONTROL"
        and delta_2.get("supersedes_amendment") == delta.get("amendment_id")
        and delta_2.get("superseded_statement_only") == "tcn_RD is a fitted matched-capacity TCN"
        and delta_2.get("timing") == "BEFORE_MODEL_TRAINING_AND_BEFORE_MODEL_OUTCOME_ACCESS"
        and "exact F_minus_S660" in delta_2.get("delta", {}).get("tcn_OFF", "")
        and "every biological channel is zero" in delta_2.get("delta", {}).get("positive_delta_nonzero", "")
        and "Every update is centered on one TRAIN event row" in
            delta_2.get("delta", {}).get("tcn_sparse_event_scheduler", "")
        and "dense smoothed evidence field" in
            delta_2.get("delta", {}).get("family_feature_scope", ""),
        "M38B model amendment 2 differs",
    )
    document = {
        "schema_version": "1.0.0",
        "stage": "M38B_AUTHENTICATE_MODEL_CONTRACT",
        "status": "PASS_BASE_AND_PRE_OUTCOME_AMENDMENT_BOUND",
        "experiment_id": base["experiment_id"],
        "base_contract_sha256": contract_sha,
        "amendment_sha256": amendment_sha,
        "amendment_2_sha256": amendment_2_sha,
        "scope": {
            "chromosome": "22", "root": "R0", "partition": "FIT",
            "people": 96, "valid_opened": False, "test_opened": False,
        },
        "family_specific_RD": {
            "analytic": "F_MINUS_S660_LAMBDA_ZERO_NO_FIT",
            "tcn": "F_MINUS_S660_EXACT_NO_TCN_FIT",
        },
        "tcn_estimand": "TOTAL_EFFECT_OF_OPENING_RARE_EVENT_RESIDUAL_ROUTE",
        "positive_control": "PURE_SYNTHETIC_EVIDENCE_ON_MATCHED_EVENT_GEOMETRY",
        "arm_binding_required": True,
        "source_binding": "DETERMINISTIC_LOAD_BEARING_SOURCE_MANIFEST",
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest": source_manifest,
    }
    write_exclusive_json(output, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-2", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--expected-amendment-2-sha256", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.contract, args.amendment, args.amendment_2,
                      args.expected_contract_sha256, args.expected_amendment_sha256,
                      args.expected_amendment_2_sha256, args.output, args.source)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
