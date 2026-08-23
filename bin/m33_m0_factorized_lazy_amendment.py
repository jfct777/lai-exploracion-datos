#!/usr/bin/env python3
"""Validate the physical-only M33 factorized/lazy storage amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_RADII = [0.05, 0.1, 0.2, 0.5]
EXPECTED_ROOTS = [386357765, 2024931463, 1324432253]
EXPECTED_ROTATIONS = {
    "R0": {"fit_root_seeds": [2024931463, 1324432253], "score_only_root_seed": 386357765},
    "R1": {"fit_root_seeds": [386357765, 1324432253], "score_only_root_seed": 2024931463},
    "R2": {"fit_root_seeds": [386357765, 2024931463], "score_only_root_seed": 1324432253},
}
EXPANDED_ARRAYS = {
    "rare_tokens", "rare_mask", "rare_locus_index", "row_ptr",
    "row_sample_index", "row_marker_index",
}
EXPECTED_OUTPUT_NAMESPACE_TEMPLATE = (
    "gs://teams-usp/frank/lai-exploracion-datos/runs/{run_id}/33_m0_factorized_lazy/"
)
EXPECTED_STOP_RULES = [
    "STOP_if_any_base_contract_hash_differs",
    "STOP_if_any_scientific_root_radius_channel_model_loss_metric_or_control_changes",
    "STOP_if_lazy_and_oracle_views_are_not_byte_exact",
    "STOP_if_SCORE_ONLY_or_EVAL_contributes_to_FIT_normalization",
    "STOP_if_any_expanded_context_array_is_persisted",
    "STOP_if_output_namespace_is_not_the_project_bucket",
    "STOP_if_REF_label_sham_is_reconstructed_from_insufficient_aggregate_summaries",
    "STOP_if_peak_RSS_reaches_0.8_of_available_memory",
]
EXPECTED_ROOT_RECEIPT_KEYS = [
    "stage", "schema_id", "status", "run_id", "root_seed", "git_commit",
    "oci_image_digest", "source_auth_sha256", "selected_locus_axis_semantic_sha256",
    "target_sample_axis_semantic_sha256", "FLARE_marker_axis_semantic_sha256",
    "artifact_raw_sha256_by_logical_id", "artifact_semantic_sha256_by_logical_id",
    "gcs_uri_generation_by_logical_id", "reopen_verified", "append_only",
]
EXPECTED_LAZY_RECIPE_KEYS = [
    "stage", "schema_id", "status", "run_id", "root_seed", "rotation_id",
    "role_in_rotation", "radii_cM", "person_batch_maximum",
    "valid_token_budget_per_batch", "central_marker_block",
    "microbatch_plan_semantic_sha256", "optimizer_update_rule", "row_order",
    "token_order", "channel_count", "factorized_root_receipt_sha256",
    "fit_callable_normalization_manifest_sha256", "source_auth_sha256", "semantic_sha256",
]
EXPECTED_READY_KEYS = [
    "stage", "schema_id", "status", "run_id", "root_seed", "rotation_id",
    "role_in_rotation", "factorized_root_receipt_sha256",
    "fit_callable_normalization_manifest_sha256", "lazy_context_recipe_sha256",
    "source_auth_sha256",
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
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs,
                       parse_constant=_reject_constant)
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False,
                         separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    require(set(value) == set(expected),
            f"{label} keys differ; missing={sorted(set(expected) - set(value))} "
            f"extra={sorted(set(value) - set(expected))}")


def validate_contract(amendment: Mapping[str, Any], materializer: Mapping[str, Any],
                      sanitizer: Mapping[str, Any], pre4: Mapping[str, Any],
                      materializer_path: Path, sanitizer_path: Path, pre4_path: Path) -> None:
    exact_keys(amendment, [
        "schema_version", "stage", "status", "scope", "base_contracts",
        "proportionality_gate", "physical_amendment", "contract_precedence",
        "interval_artifact", "lazy_reconstruction", "publication_contract",
        "unchanged_scientific_contract",
        "required_gates_before_real_roots", "ref_label_sham_dependency",
        "execution_authorization", "stop_rules",
    ], "amendment")
    require(amendment["schema_version"] == "1.0.0", "schema version changed")
    require(amendment["stage"] == "M33_M0_FACTORIZED_LAZY_AMENDMENT_CONTRACT",
            "stage changed")
    require(amendment["status"] ==
            "CONTRACT_AND_SYNTHETIC_EQUIVALENCE_ONLY_NO_REAL_ROOT_EXECUTION_NO_TRAINING",
            "real execution was opened")
    require(amendment["scope"] == "CHR22_DEVELOPMENT_PRE4_THREE_ROOTS", "scope changed")

    bases = amendment["base_contracts"]
    exact_keys(bases, ["materializer", "sanitized_f0_amendment", "pre4_preregistration"],
               "base_contracts")
    for label, declared, path, payload in (
        ("materializer", bases["materializer"], materializer_path, materializer),
        ("sanitized_f0_amendment", bases["sanitized_f0_amendment"], sanitizer_path, sanitizer),
        ("pre4_preregistration", bases["pre4_preregistration"], pre4_path, pre4),
    ):
        exact_keys(declared, ["path", "raw_sha256", "semantic_sha256"], label)
        require(raw_sha256(path) == declared["raw_sha256"], f"{label} raw hash drift")
        require(semantic_sha256(payload) == declared["semantic_sha256"],
                f"{label} semantic hash drift")

    gate = amendment["proportionality_gate"]
    exact_keys(gate, [
        "technical_anchor", "target_samples", "context_locus_assignments_by_radius_cM",
        "packed_token_bytes", "minimum_token_array_bytes_per_root_rotation",
        "minimum_shards_per_root_rotation", "development_root_rotation_copies",
        "projected_minimum_token_array_bytes_all_copies",
        "projected_minimum_shards_all_copies", "verdict",
    ], "proportionality_gate")
    expected_assignments = {"0.05": 36229121, "0.10": 58587636,
                            "0.20": 92883536, "0.50": 175066132}
    require(gate["technical_anchor"] == "root17_occupancy_only_not_scientific_selection",
            "technical anchor gained a scientific role")
    require(gate["target_samples"] == 30 and
            gate["context_locus_assignments_by_radius_cM"] == expected_assignments and
            gate["packed_token_bytes"] == 61, "proportionality inputs changed")
    expected_bytes = sum(expected_assignments.values()) * 30 * 61
    require(gate["minimum_token_array_bytes_per_root_rotation"] == expected_bytes == 663862557750,
            "per-copy byte estimate differs")
    require(gate["development_root_rotation_copies"] == 9 and
            gate["projected_minimum_token_array_bytes_all_copies"] == 9 * expected_bytes,
            "development storage estimate differs")
    require(gate["minimum_shards_per_root_rotation"] == 41520 and
            gate["projected_minimum_shards_all_copies"] == 373680,
            "minimum shard estimate differs")
    require(gate["verdict"].startswith("STOP_PERSISTENT_EXPANDED_CONTEXTS"),
            "proportionality stop was removed")

    physical = amendment["physical_amendment"]
    exact_keys(physical, [
        "scientific_estimand_changed", "persistent_packed_rare_context_shard",
        "persistent_bundle_manifest_by_radius",
        "persistent_materialization_receipt_for_expanded_shards",
        "canonical_storage_definition", "lazy_definition", "canonical_root_artifacts",
        "rotation_artifacts", "forbidden_persistence", "output_namespace_template",
    ], "physical_amendment")
    require(physical["scientific_estimand_changed"] is False,
            "physical amendment changed the estimand")
    for key in ("persistent_packed_rare_context_shard",
                "persistent_bundle_manifest_by_radius",
                "persistent_materialization_receipt_for_expanded_shards"):
        require(physical[key] is False, f"expanded persistence reopened: {key}")
    require(set(physical["forbidden_persistence"]) == EXPANDED_ARRAYS,
            "expanded-array persistence inventory changed")
    require(physical["output_namespace_template"] == EXPECTED_OUTPUT_NAMESPACE_TEMPLATE,
            "output is not confined to the project bucket")
    require("context_intervals_all_frozen_radii" in physical["canonical_root_artifacts"] and
            "factorized_READY" in physical["rotation_artifacts"],
            "factorized artifacts are incomplete")

    precedence = amendment["contract_precedence"]
    exact_keys(precedence, [
        "mode", "replaced_base_logical_outputs", "replacement_logical_outputs",
        "base_clauses_preserved", "pre4_preregistration_preserved_by_hash",
    ], "contract_precedence")
    require(precedence == {
        "mode": "additive_amendment_replaces_only_named_physical_clauses",
        "replaced_base_logical_outputs": [
            "packed_rare_context", "bundle_manifest_by_radius", "materialization_receipt", "READY"
        ],
        "replacement_logical_outputs": [
            "canonical_root_factors", "context_intervals_all_frozen_radii",
            "factorized_root_receipt", "fit_callable_normalization_manifest",
            "lazy_context_recipe", "factorized_READY",
        ],
        "base_clauses_preserved": [
            "root_registry", "incremental_partition", "reference_semantics",
            "sample_key_contract", "f0_contract", "control_views",
            "primary_transferable_input", "packed_loader", "privacy_contract",
        ],
        "pre4_preregistration_preserved_by_hash": True,
    }, "contract precedence changed")

    interval = amendment["interval_artifact"]
    exact_keys(interval, [
        "schema_id", "arrays", "radii_cM", "interval", "ordering",
        "nested_radius_invariant", "axis_binding", "truncation",
        "append_only_reopen_and_semantic_hash_required",
    ], "interval_artifact")
    require(interval["schema_id"] == "m33_m0_context_intervals_v1" and
            interval["radii_cM"] == EXPECTED_RADII and
            interval["interval"] ==
            "inclusive_closed_marker_cM_minus_radius_to_marker_cM_plus_radius" and
            interval["ordering"] == "radius_then_authenticated_FLARE_marker_order" and
            interval["nested_radius_invariant"] ==
            "larger_radius_start_le_smaller_radius_start_and_larger_radius_stop_ge_smaller_radius_stop" and
            interval["axis_binding"] ==
            "receipt_binds_selected_locus_axis_semantic_sha256_and_FLARE_marker_axis_semantic_sha256" and
            interval["truncation"] is False and
            interval["append_only_reopen_and_semantic_hash_required"] is True,
            "interval contract changed")
    require(interval["arrays"] == {
        "radii_cM": {"axes": ["radius"], "dtype": "<f8"},
        "context_start": {"axes": ["radius", "marker"], "dtype": "<u8"},
        "context_stop": {"axes": ["radius", "marker"], "dtype": "<u8"},
    }, "interval physical schema changed")

    lazy = amendment["lazy_reconstruction"]
    exact_keys(lazy, [
        "channel_count", "channel_semantics", "row_order", "token_order",
        "person_batch_maximum", "valid_token_budget_per_batch", "empty_context",
        "single_context_over_budget", "minor_allele_orientation", "target_phase",
        "normalization", "forbidden_dependencies",
    ], "lazy_reconstruction")
    require(lazy["channel_count"] == materializer["primary_transferable_input"]["channel_count"] == 13,
            "channel count changed")
    require(lazy["channel_semantics"] ==
            "exactly_m33_m0_materializer_contract_primary_transferable_input_channels",
            "channel semantics changed")
    require(lazy["row_order"] == "sample_major_then_marker" and
            lazy["token_order"] == "cM_then_bp_then_locus_id" and
            lazy["empty_context"] == "valid_zero_length_row" and
            lazy["single_context_over_budget"] == "STOP_NO_TRUNCATION",
            "packed ordering or no-truncation semantics changed")
    require(lazy["person_batch_maximum"] == materializer["packed_loader"]["person_batch"] == 8 and
            lazy["valid_token_budget_per_batch"] == materializer["packed_loader"]["token_budget"] == 262144,
            "batch or token limit changed")
    require(lazy["minor_allele_orientation"].endswith("minor_code_0_and_1") and
            lazy["target_phase"] == "diploid_minor_dosage_only_no_rare_phase" and
            "FIT_roots_only" in lazy["normalization"], "rare semantics or FIT boundary changed")
    require({"SCORE_ONLY", "EVAL", "truth"} <= set(lazy["forbidden_dependencies"]),
            "forbidden lazy dependencies changed")

    publication = amendment["publication_contract"]
    exact_keys(publication, [
        "append_only_precondition", "reopen_verify", "write_order",
        "lazy_recipe_constants",
        "factorized_root_receipt_required_keys", "lazy_context_recipe_required_keys",
        "factorized_READY_required_keys", "READY_write_rule",
    ], "publication_contract")
    require(publication["append_only_precondition"] == "ifGenerationMatch=0" and
            publication["reopen_verify"] is True and
            publication["write_order"][-1] == "factorized_READY" and
            publication["READY_write_rule"] ==
            "write_last_only_after_all_objects_reopen_and_hash_and_generation_verification",
            "append-only or READY-last publication changed")
    require(publication["lazy_recipe_constants"] == {
        "central_marker_block": 256,
        "optimizer_update_rule": "accumulate_loss_numerator_denominator_and_gradients_across_all_microbatches_then_one_optimizer_step_after_last_microbatch_of_logical_block",
    }, "lazy recipe constants changed")
    require(publication["factorized_root_receipt_required_keys"] == EXPECTED_ROOT_RECEIPT_KEYS,
            "root receipt schema differs")
    require(publication["lazy_context_recipe_required_keys"] == EXPECTED_LAZY_RECIPE_KEYS,
            "lazy recipe schema differs")
    require(publication["factorized_READY_required_keys"] == EXPECTED_READY_KEYS,
            "factorized READY schema differs")

    unchanged = amendment["unchanged_scientific_contract"]
    exact_keys(unchanged, [
        "development_roots", "rotations", "forbidden_technical_roots",
        "forbidden_eval_roots", "radii_cM", "channels_models_losses_metrics_controls",
        "no_parameter_selection_from_technical_roots",
    ], "unchanged_scientific_contract")
    require(unchanged["development_roots"] == EXPECTED_ROOTS and
            unchanged["rotations"] == EXPECTED_ROTATIONS and
            unchanged["radii_cM"] == EXPECTED_RADII and
            unchanged["channels_models_losses_metrics_controls"] is True and
            unchanged["no_parameter_selection_from_technical_roots"] is True,
            "scientific design changed")
    require(sanitizer["development_roots"]["exact_seeds"] == EXPECTED_ROOTS and
            sanitizer["development_roots"]["rotations"] == EXPECTED_ROTATIONS and
            sanitizer["unchanged_scientific_contract"]["radii_cM"] == EXPECTED_RADII,
            "amendment does not preserve the sanitized F0 contract")
    pre4_rotations = {
        row["rotation"]: {
            "fit_root_seeds": row["fit_roots"],
            "score_only_root_seed": row["score_only_root"],
        }
        for row in pre4["root_registry"]["development_rotations"]
    }
    require(pre4["root_registry"]["DEVELOPMENT"] == EXPECTED_ROOTS and
            pre4_rotations == EXPECTED_ROTATIONS and
            pre4["packed_loader"]["radii_cm"] == EXPECTED_RADII and
            pre4["packed_loader"]["person_batch"] == 8 and
            pre4["packed_loader"]["token_budget"] == 262144,
            "PRE4 roots, rotations, radii or loader limits differ")
    require(set(unchanged["forbidden_technical_roots"]).isdisjoint(EXPECTED_ROOTS) and
            set(unchanged["forbidden_eval_roots"]).isdisjoint(EXPECTED_ROOTS),
            "forbidden root entered DEVELOPMENT")

    gates = amendment["required_gates_before_real_roots"]
    exact_keys(gates, [
        "synthetic_exact_equivalence", "consumer_exact_equivalence",
        "fit_leakage_negative_test", "technical_known_answer", "determinism",
        "performance_T0", "memory_warning_fraction", "memory_stop_fraction",
    ], "required_gates_before_real_roots")
    require(gates["memory_warning_fraction"] == 0.7 and
            gates["memory_stop_fraction"] == 0.8 and
            "logits_loss_and_input_gradient_equal" in gates["consumer_exact_equivalence"] and
            gates["performance_T0"].startswith("forward_only_no_gradient_no_truth"),
            "equivalence or memory gate changed")
    sham = amendment["ref_label_sham_dependency"]
    exact_keys(sham, [
        "status", "required_resolution", "individual_reference_genotype_export",
        "modeling_blocked_until_resolved",
    ], "ref_label_sham_dependency")
    require(sham == {
        "status": "NOT_DERIVABLE_FROM_AGGREGATED_REFERENCE_SUMMARIES",
        "required_resolution": "derive_each_preregistered_permutation_inside_SAFE_BRIDGE_from_REF_genotypes_then_export_only_aggregated_permuted_reference_summaries",
        "individual_reference_genotype_export": False,
        "modeling_blocked_until_resolved": True,
    }, "REF-label sham was weakened")
    authorization = amendment["execution_authorization"]
    exact_keys(authorization, [
        "contract_validation", "synthetic_equivalence_tests", "real_root_read",
        "real_root_write", "forward_T0", "training", "truth_scoring",
    ], "execution_authorization")
    require(authorization["contract_validation"] is True and
            authorization["synthetic_equivalence_tests"] is True and
            all(authorization[key] is False for key in
                ("real_root_read", "real_root_write", "forward_T0", "training", "truth_scoring")),
            "contract-only execution boundary changed")
    require(amendment["stop_rules"] == EXPECTED_STOP_RULES, "stop rules changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--materializer-contract", type=Path, required=True)
    parser.add_argument("--sanitized-f0-contract", type=Path, required=True)
    parser.add_argument("--pre4-preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    amendment = load_json(args.contract)
    materializer = load_json(args.materializer_contract)
    sanitizer = load_json(args.sanitized_f0_contract)
    pre4 = load_json(args.pre4_preregistration)
    validate_contract(amendment, materializer, sanitizer, pre4,
                      args.materializer_contract, args.sanitized_f0_contract,
                      args.pre4_preregistration)
    receipt = {
        "stage": amendment["stage"],
        "status": "PASS_CONTRACT_AND_SYNTHETIC_EQUIVALENCE_GATE_ONLY",
        "contract_raw_sha256": raw_sha256(args.contract),
        "contract_semantic_sha256": semantic_sha256(amendment),
        "real_root_execution": False,
        "training": False,
    }
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
