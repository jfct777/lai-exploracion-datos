#!/usr/bin/env python3
"""Validate the additive M33 M0 boundary for sanitized FLARE probabilities.

This module does not read scientific assets or materialize model inputs.  It
proves that the amendment removes raw FLARE records from MATERIALIZE while
preserving the frozen PRE4 roots, representation, geometry and FIT-only
normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


HEX64 = set("0123456789abcdef")
EXPECTED_BASE_INPUTS = [
    "selected_loci_incremental",
    "target_rare_diploid_incremental",
    "reference_rare_summary_incremental",
    "a0_authenticated_flare_anc",
    "derived_flare_anc_tbi",
    "authenticated_genetic_map",
    "authenticated_fit_callable_normalization_manifest",
]
EXPECTED_AMENDED_INPUTS = [
    "selected_loci_incremental",
    "target_rare_diploid_incremental",
    "reference_rare_summary_incremental",
    "flare_f0_sanitized",
    "authenticated_genetic_map",
    "authenticated_fit_callable_normalization_manifest",
    "safe_bridge_receipt",
    "safe_bridge_independent_verify_receipt",
]
EXPECTED_DEVELOPMENT_ROOTS = [386357765, 2024931463, 1324432253]
EXPECTED_ROTATIONS = {
    "R0": {"fit_root_seeds": [2024931463, 1324432253], "score_only_root_seed": 386357765},
    "R1": {"fit_root_seeds": [386357765, 1324432253], "score_only_root_seed": 2024931463},
    "R2": {"fit_root_seeds": [386357765, 2024931463], "score_only_root_seed": 1324432253},
}
EXPECTED_RADII = [0.05, 0.1, 0.2, 0.5]
EXPECTED_FORBIDDEN_FIELDS = ["GT", "AN1", "AN2", "truth", "hard_call", "target_rare_phase"]
EXPECTED_STOP_RULES = [
    "STOP_if_base_contract_hash_differs",
    "STOP_if_any_scientific_parameter_or_root_registry_changes",
    "STOP_if_MATERIALIZE_can_read_raw_FLARE_or_truth_bearing_fields",
    "STOP_if_F0_sample_or_marker_axis_is_not_exact",
    "STOP_if_SCORE_ONLY_or_EVAL_contributes_to_FIT_normalization",
    "STOP_if_SAFE_BRIDGE_receipt_and_independent_verification_are_not_bound",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    require(isinstance(payload, dict), f"{path} must contain one JSON object")
    return payload


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_keys(payload: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    require(
        set(payload) == set(expected),
        f"{label} keys differ; missing={sorted(set(expected) - set(payload))} "
        f"extra={sorted(set(payload) - set(expected))}",
    )


def validate_contract(amendment: Mapping[str, Any], base: Mapping[str, Any], base_path: Path) -> None:
    exact_keys(
        amendment,
        [
            "schema_version", "stage", "status", "scope", "base_contract",
            "development_roots", "sanitizer_boundary", "materialize_boundary",
            "unchanged_scientific_contract", "execution_authorization", "stop_rules",
        ],
        "amendment",
    )
    require(amendment["schema_version"] == "1.0.0", "wrong amendment schema version")
    require(amendment["stage"] == "M33_M0_F0_SANITIZED_AMENDMENT_CONTRACT", "wrong stage")
    require(amendment["status"] == "CONTRACT_ONLY_NO_REAL_EXECUTION", "execution opened")
    require(amendment["scope"] == "CHR22_DEVELOPMENT_PRE4_THREE_ROOTS", "scope changed")

    anchor = amendment["base_contract"]
    exact_keys(anchor, ["path", "raw_sha256", "semantic_sha256"], "base_contract")
    require(len(anchor["raw_sha256"]) == 64 and set(anchor["raw_sha256"]) <= HEX64,
            "invalid base raw SHA-256")
    require(raw_sha256(base_path) == anchor["raw_sha256"], "base contract raw hash drift")
    require(semantic_sha256(base) == anchor["semantic_sha256"], "base contract semantic hash drift")
    require(base["process_contracts"]["MATERIALIZE"]["input_logical_ids"] == EXPECTED_BASE_INPUTS,
            "base MATERIALIZE input boundary drift")

    roots = amendment["development_roots"]
    exact_keys(roots, ["exact_seeds", "rotations", "forbidden_technical_seeds", "forbidden_eval_seeds"],
               "development_roots")
    require(roots["exact_seeds"] == EXPECTED_DEVELOPMENT_ROOTS, "DEVELOPMENT roots changed")
    require(roots["rotations"] == EXPECTED_ROTATIONS, "DEVELOPMENT rotations changed")
    require(set(roots["forbidden_technical_seeds"]).isdisjoint(roots["exact_seeds"]),
            "technical roots entered DEVELOPMENT")
    require(set(roots["forbidden_eval_seeds"]).isdisjoint(roots["exact_seeds"]),
            "EVAL roots entered DEVELOPMENT")

    sanitizer = amendment["sanitizer_boundary"]
    exact_keys(sanitizer, [
        "process", "raw_inputs", "output_logical_id", "truth_mounted", "allowed_source_fields",
        "forbidden_source_fields", "probability_contract", "identity_contract", "artifact",
    ], "sanitizer_boundary")
    require(sanitizer["process"] == "SANITIZE_FLARE_F0", "sanitizer process changed")
    require(sanitizer["raw_inputs"] == ["a0_authenticated_flare_anc", "derived_flare_anc_tbi"],
            "sanitizer raw boundary changed")
    require(sanitizer["output_logical_id"] == "flare_f0_sanitized", "sanitized output changed")
    require(sanitizer["truth_mounted"] is False, "truth entered sanitizer")
    require(sanitizer["forbidden_source_fields"] == EXPECTED_FORBIDDEN_FIELDS,
            "forbidden F0 dependencies changed")
    require(sanitizer["allowed_source_fields"] == [
        "root_seed", "sample_id", "chrom", "pos", "ref", "alt", "ANP1", "ANP2"
    ], "allowed F0 source fields changed")
    probability = sanitizer["probability_contract"]
    exact_keys(probability, [
        "ancestry_order", "raw_sum_range", "operation",
        "float32_simplex_absolute_tolerance", "haplotype_axis_preserved",
    ], "probability_contract")
    require(probability["ancestry_order"] == ["AFR", "EUR", "ASIA"], "ancestry order changed")
    require(probability["raw_sum_range"] == [0.98, 1.02], "raw probability gate changed")
    require(probability["float32_simplex_absolute_tolerance"] == 0.000005,
            "float32 simplex tolerance changed")
    require(probability["operation"] ==
            "renormalize_each_ANP_vector_to_exact_simplex_in_float64_then_cast_little_endian_float32",
            "F0 normalization operation changed")
    require(probability["haplotype_axis_preserved"] is True, "F0 haplotype axis lost")
    identity = sanitizer["identity_contract"]
    exact_keys(identity, [
        "sample_key", "marker_key", "sample_axis", "marker_axis",
        "duplicate_missing_or_extra_identity",
    ], "identity_contract")
    require(identity == {
        "sample_key": "sha256_DNABR_M33_M0_SAMPLE_V1_pipe_plus_exact_sample_id",
        "marker_key": ["chrom", "pos", "ref", "alt"],
        "sample_axis": "exact_bijection_and_ordered_equality_with_TARGET_axis",
        "marker_axis": "exact_identity_and_order_with_authenticated_FLARE_marker_axis",
        "duplicate_missing_or_extra_identity": "STOP_BEFORE_WRITE",
    }, "F0 identity contract changed")
    artifact = sanitizer["artifact"]
    exact_keys(artifact, [
        "schema_id", "format", "privacy", "arrays", "contains_raw_genotypes",
        "contains_hard_calls", "contains_truth", "append_only",
        "reopen_and_semantic_hash_required",
    ], "sanitized artifact")
    require(artifact["schema_id"] == "m33_m0_flare_f0_sanitized_v1" and
            artifact["format"] == "npz_uncompressed" and artifact["privacy"] == "private",
            "sanitized artifact identity changed")
    require(artifact["arrays"] == {
        "sample_key_sha256": {"axes": ["sample"], "dtype": "|S64"},
        "marker_chrom": {"axes": ["marker"], "dtype": "|u1"},
        "marker_pos": {"axes": ["marker"], "dtype": "<i8"},
        "marker_ref": {"axes": ["marker"], "dtype": "|S1"},
        "marker_alt": {"axes": ["marker"], "dtype": "|S1"},
        "F0": {"axes": ["sample", "haplotype", "marker", "ancestry"], "dtype": "<f4"},
    }, "sanitized artifact arrays changed")
    require(artifact["arrays"]["F0"] == {
        "axes": ["sample", "haplotype", "marker", "ancestry"], "dtype": "<f4"
    }, "F0 physical representation changed")
    require(all(artifact[key] is False for key in
                ("contains_raw_genotypes", "contains_hard_calls", "contains_truth")),
            "forbidden payload entered sanitized artifact")
    require(artifact["append_only"] is True and artifact["reopen_and_semantic_hash_required"] is True,
            "sanitized artifact is not auditable")

    materialize = amendment["materialize_boundary"]
    exact_keys(materialize, [
        "removed_input_logical_ids", "added_input_logical_ids",
        "exact_input_logical_ids_after_amendment", "forbidden_inputs",
        "required_receipt_bindings", "fit_callable_normalization_manifest_role",
        "genetic_map_role",
    ], "materialize_boundary")
    require(materialize["removed_input_logical_ids"] ==
            ["a0_authenticated_flare_anc", "derived_flare_anc_tbi"],
            "raw FLARE removal changed")
    require(materialize["added_input_logical_ids"] == [
        "flare_f0_sanitized", "safe_bridge_receipt", "safe_bridge_independent_verify_receipt"
    ], "amended MATERIALIZE additions changed")
    require(materialize["exact_input_logical_ids_after_amendment"] == EXPECTED_AMENDED_INPUTS,
            "amended MATERIALIZE input boundary changed")
    derived_inputs = [
        value for value in EXPECTED_BASE_INPUTS
        if value not in materialize["removed_input_logical_ids"]
    ]
    derived_inputs[3:3] = materialize["added_input_logical_ids"][:1]
    derived_inputs.extend(materialize["added_input_logical_ids"][1:])
    require(derived_inputs == materialize["exact_input_logical_ids_after_amendment"],
            "declared add/remove delta does not reconstruct amended MATERIALIZE inputs")
    joined = " ".join(materialize["exact_input_logical_ids_after_amendment"]).lower()
    for forbidden in materialize["forbidden_inputs"]:
        require(forbidden.lower() not in joined, f"forbidden MATERIALIZE input present: {forbidden}")
    require("FIT_roots_only" in materialize["fit_callable_normalization_manifest_role"],
            "FIT-only normalization lost")
    require(materialize["genetic_map_role"] ==
            "authenticated_context_geometry_only_no_labels_no_truth",
            "genetic map gained a forbidden role")
    require(set(materialize["required_receipt_bindings"]) == {
        "root_seed", "source_auth_sha256", "artifact_semantic_sha256",
        "sample_axis_semantic_sha256", "marker_axis_semantic_sha256",
    }, "receipt identity binding changed")

    unchanged = amendment["unchanged_scientific_contract"]
    exact_keys(unchanged, [
        "rare_representation", "minor_allele_orientation", "channel_count", "radii_cM",
        "person_batch", "maximum_valid_tokens_per_shard", "models_losses_metrics_controls",
        "pre4_roots_and_rotations", "no_parameter_selection_from_technical_roots",
    ], "unchanged_scientific_contract")
    require(unchanged["rare_representation"] is True and
            unchanged["minor_allele_orientation"] is True and
            unchanged["channel_count"] == 13 and
            unchanged["radii_cM"] == EXPECTED_RADII and
            unchanged["person_batch"] == 8 and
            unchanged["maximum_valid_tokens_per_shard"] == 262144 and
            unchanged["models_losses_metrics_controls"] is True and
            unchanged["pre4_roots_and_rotations"] is True and
            unchanged["no_parameter_selection_from_technical_roots"] is True,
            "scientific contract changed")
    require(base["primary_transferable_input"]["channel_count"] == unchanged["channel_count"] and
            base["packed_loader"]["radii_cm"] == unchanged["radii_cM"] and
            base["packed_loader"]["person_batch"] == unchanged["person_batch"] and
            base["packed_loader"]["token_budget"] == unchanged["maximum_valid_tokens_per_shard"] and
            base["process_contracts"]["MATERIALIZE"]["fit_callable_normalization_manifest"]
                ["development_rotations"] == roots["rotations"],
            "amendment does not preserve the frozen base semantics")
    authorization = amendment["execution_authorization"]
    exact_keys(authorization, [
        "contract_validation", "synthetic_tests", "real_asset_read", "sanitize_f0",
        "materialize", "write_READY", "forward", "backward", "training", "truth_scoring",
    ], "execution_authorization")
    require(authorization["contract_validation"] is True and authorization["synthetic_tests"] is True,
            "contract tests disabled")
    require(all(authorization[key] is False for key in (
        "real_asset_read", "sanitize_f0", "materialize", "write_READY", "forward",
        "backward", "training", "truth_scoring",
    )), "real execution was authorized by a contract-only amendment")
    require(amendment["stop_rules"] == EXPECTED_STOP_RULES, "stop rules changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    amendment = load_json(args.contract)
    base_path = args.repo_root / amendment["base_contract"]["path"]
    base = load_json(base_path)
    validate_contract(amendment, base, base_path)
    print(json.dumps({
        "status": "PASS_CONTRACT_ONLY",
        "stage": amendment["stage"],
        "base_raw_sha256": raw_sha256(base_path),
        "amendment_semantic_sha256": semantic_sha256(amendment),
        "real_execution_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
