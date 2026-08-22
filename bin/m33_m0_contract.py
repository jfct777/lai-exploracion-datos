#!/usr/bin/env python3
"""Fail-closed validation helpers for the M33 M0 contract.

This module deliberately has no real-data materializer.  It validates the frozen
contract and small synthetic known answers only.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_CHANNELS = [
    "target_minor_dosage_div_2_float32_clip_0_1", "target_observed_mask_float32_0_1",
    "ref_minor_af_AFR_float32_clip_0_1", "log1p_ref_callable_AFR_div_log1p_maxAN_AFR_FIT", "ref_observed_AFR_float32_0_1",
    "ref_minor_af_EUR_float32_clip_0_1", "log1p_ref_callable_EUR_div_log1p_maxAN_EUR_FIT", "ref_observed_EUR_float32_0_1",
    "ref_minor_af_ASIA_float32_clip_0_1", "log1p_ref_callable_ASIA_div_log1p_maxAN_ASIA_FIT", "ref_observed_ASIA_float32_0_1",
    "relative_cM_minus_marker_cM_div_radius_clip_minus1_1", "delta_cM_from_previous_ordered_locus_div_radius_clip_0_2",
]
EXPECTED_RADII = [0.05, 0.1, 0.2, 0.5]
EXPECTED_CONTRACT_SEMANTIC_SHA256 = "46f84a9015908f9056fa01555e3686c4efd5d23a3e6d3d842c28a87189f71c59"
DEVELOPMENT_ROTATIONS = {
    "R0": {"fit_root_seeds": [2024931463, 1324432253], "score_only_root_seed": 386357765},
    "R1": {"fit_root_seeds": [386357765, 1324432253], "score_only_root_seed": 2024931463},
    "R2": {"fit_root_seeds": [386357765, 2024931463], "score_only_root_seed": 1324432253},
}
FORBIDDEN_EVAL_ROOTS = [1341407242, 2049644864, 693524843, 1896826422, 166187460]
RADIUS_KEYS = ["0.05", "0.1", "0.2", "0.5"]
BUNDLE_REQUIRED_KEYS = (
    "stage", "schema_id", "status", "root_label", "root_seed", "rotation_id", "role_in_rotation",
    "radius_cM", "fit_callable_normalization_manifest_sha256", "sample_count",
    "sample_axis_semantic_sha256", "marker_count", "marker_axis_semantic_sha256", "ordered_shards",
    "raw_semantic_sha256", "channel_semantic_sha256", "source_auth_sha256",
)
RECEIPT_REQUIRED_KEYS = (
    "stage", "schema_id", "status", "root_label", "root_seed", "rotation_id", "role_in_rotation",
    "radii_cM", "fit_callable_normalization_manifest_sha256", "sample_count",
    "sample_axis_semantic_sha256", "marker_count", "marker_axis_semantic_sha256",
    "ordered_bundle_manifest_sha256_by_radius", "raw_semantic_sha256", "channel_semantic_sha256",
    "source_auth_sha256", "reopen_verified", "append_only",
)
READY_REQUIRED_KEYS = (
    "stage", "schema_id", "status", "root_label", "root_seed", "rotation_id", "role_in_rotation",
    "fit_callable_normalization_manifest_sha256", "sample_count", "sample_axis_semantic_sha256",
    "marker_count", "marker_axis_semantic_sha256", "materialization_receipt_sha256",
    "ordered_bundle_manifest_sha256_by_radius", "source_auth_sha256",
)
OUTPUT_CHAIN_IDENTITY_FIELDS = (
    "root_label", "root_seed", "rotation_id", "role_in_rotation",
    "fit_callable_normalization_manifest_sha256", "sample_count", "sample_axis_semantic_sha256",
    "marker_count", "marker_axis_semantic_sha256", "source_auth_sha256",
)
SHARD_REQUIRED_KEYS = (
    "schema_id", "shard_ordinal", "person_start", "person_end_exclusive", "marker_start",
    "marker_end_exclusive", "valid_token_count", "gcs_uri", "gcs_generation", "raw_sha256",
    "semantic_sha256",
)
F0_KEY = ("root_seed", "sample_id", "chrom", "pos", "ref", "alt")
LOCUS_KEYS = ("chrom", "pos", "ref", "alt", "locus_id", "cM")
HEX64 = set("0123456789abcdef")
EXPECTED_A0 = {
    "git_commit": "f1b08289ac6635cbe69cb1d8c897f86ea51f22cc",
    "official_receipt_sha256": "4fe79f0dc648caa2c64b9d81c752e144af6d0c9db43e3fa3ae4274887ecfff93",
    "source_auth_sha256": "e9b062c6ac803d7a3c4b50ba087c3553d89efe8d4d83be42e539d2bb74ed1940",
    "index_audit_sha256": "614931d5042999602be6078e93d802b31d27282b33b3cadabfc2a54e2afc7ad3",
    "preregistration_sha256": "c99f890e00c383df25bbdfbb94e9fba3bb181adbfa4db1e1301e4648cfa3d70d",
    "asset_registry_sha256": "44311fe8ef9238c81f630343857439ac16e52b7569c1348a52fb65d744ad93cd",
}
EXPECTED_A0_ROOT18 = {
    "git_commit": "cd324fe156fcb4168b6249d20d1a34144d58f065",
    "official_receipt_sha256": "de1ea4ae2b67179acb9a6f56024bfe9bbb92b00663ef3684b280d077f1726b4b",
    "source_auth_sha256": "03fefb178a50d4427ad9be89a908adf03cbddf1b2ad58ab66e6b23a899f85255",
    "index_audit_sha256": "328b0ab2ba9fa4032798bb54bede1be1daa324082f884484c92e42bc62532a2a",
    "preregistration_sha256": "c99f890e00c383df25bbdfbb94e9fba3bb181adbfa4db1e1301e4648cfa3d70d",
    "asset_registry_sha256": "649993d6e098b3cf92260a95d5bfcf8a89a529f8438f753dce368598958773de",
}
EXPECTED_KAT_COUNTS = {
    "flare_loci": 79791, "selected_all": 94029, "selected_incremental": 94029,
    "flare_overlap": 0, "reference_no_support": 43892, "target_missing_diploid_cells": 0,
}
EXPECTED_KAT_HASHES = {
    "target_diploid_dosage": "ea142d0a87ae4e74a6817b15f4b9dc196467ea6f0c875611aa0813b417e2eff4",
    "reference_ac_an": "6f0d91443fd5f187377111e38133e8ab823399ebb1fe095c3b78e4da06723bdd",
    "flare_anc_raw": "85dfd76df2c14cb8fe0a753910f25c49c88d38edc5708ec6d641053d95cc74e8",
    "genetic_map": "33c7a94e0cbc0ce3cc3ff83cd3838119a881cb7e838825521a778e07a01ee6e9",
}
EXPECTED_KAT18_COUNTS = {
    "flare_loci": 79791, "selected_all": 94703, "selected_incremental": 94703,
    "flare_overlap": 0, "reference_no_support": 43938, "target_missing_diploid_cells": 0,
}
EXPECTED_KAT18_HASHES = {
    "target_diploid_dosage": "cc068025663bd88e3b2903685fdf76cf164a04c9ac52362b80305c0adc52ee46",
    "reference_ac_an": "f2fcf538258114df5410674fe52ef7b6abaf64a35ec4bc251d6a6c25633ccee5",
    "flare_anc_raw": "edc4bcdc62f5ce0ffe04bd27e9d6d6ee892e03282a1474639fc3082fbc3832c9",
    "genetic_map": "33c7a94e0cbc0ce3cc3ff83cd3838119a881cb7e838825521a778e07a01ee6e9",
}


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


def loads_strict_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)


def load_json(path: Path) -> dict[str, Any]:
    payload = loads_strict_json(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "contract must be a JSON object")
    return payload


def exact_keys(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    require(actual == expected_set,
            f"{path} keys differ; missing={sorted(expected_set - actual)} extra={sorted(actual - expected_set)}")


OBJECT_KEYS: dict[tuple[str, ...], set[str]] = {
    (): {"schema_version", "stage", "status", "scope", "supersedes_historical_contracts", "anchors", "root_registry",
         "process_contracts", "incremental_partition", "reference_semantics", "f0_contract", "primary_transferable_input",
         "sample_key_contract", "control_views", "packed_loader", "privacy_contract", "source_auth_policy", "persistence_contract",
         "execution_authorization", "known_contract_tensions"},
    ("anchors",): {"a0_root17", "a0_root18", "known_answer_root17", "known_answer_root18", "pre4_contract"},
    ("anchors", "a0_root17"): set(EXPECTED_A0),
    ("anchors", "a0_root18"): set(EXPECTED_A0_ROOT18),
    ("anchors", "known_answer_root17"): {"profile", "counts", "sha256"},
    ("anchors", "known_answer_root17", "counts"): set(EXPECTED_KAT_COUNTS),
    ("anchors", "known_answer_root17", "sha256"): set(EXPECTED_KAT_HASHES),
    ("anchors", "known_answer_root18"): {"profile", "counts", "sha256"},
    ("anchors", "known_answer_root18", "counts"): set(EXPECTED_KAT18_COUNTS),
    ("anchors", "known_answer_root18", "sha256"): set(EXPECTED_KAT18_HASHES),
    ("anchors", "pre4_contract"): {"git_commit", "preregistration_sha256"},
    ("root_registry",): {"consumed_technical_roots", "root18_status", "consumed_roots_quarantine", "scientific_selection", "radius_selection"},
    ("root_registry", "consumed_technical_roots"): {"root17", "root18"},
    ("process_contracts",): {"I0_DERIVE_AUTHENTICATE_FLARE_INDEX", "SAFE_BRIDGE", "MATERIALIZE"},
    ("process_contracts", "I0_DERIVE_AUTHENTICATE_FLARE_INDEX"): {"implemented", "status", "input_logical_ids", "output_logical_ids", "tool", "exact_command_argv", "query_parity", "write_policy", "requirements", "receipt_required_keys"},
    ("process_contracts", "I0_DERIVE_AUTHENTICATE_FLARE_INDEX", "tool"): {"name", "exact_version", "local_image_id_technical_anchor", "required_pullable_oci_image"},
    ("process_contracts", "SAFE_BRIDGE"): {"implemented", "isolation", "truth_mounted", "physical_inputs", "derivations", "role_firewall", "target_minor_orientation", "reference_minor_orientation", "output_artifacts"},
    ("process_contracts", "SAFE_BRIDGE", "physical_inputs"): {"tree_sequence", "pools", "rare_catalog", "rare_haplotypes", "selected_sites", "target_calls", "common_reference_crosscheck", "ref_pairs", "panel_map", "genetic_map"},
    ("process_contracts", "SAFE_BRIDGE", "physical_inputs", "*"): {"logical_id", "format", "access", "authentication"},
    ("process_contracts", "SAFE_BRIDGE", "derivations"): {"selected_loci_incremental", "target_rare_diploid_incremental", "reference_rare_summary_incremental", "common_reference_crosscheck_scope"},
    ("process_contracts", "SAFE_BRIDGE", "role_firewall"): {"tree_and_pools_allowed_nodes", "required_node_set_equality", "forbidden_genotype_contributors", "freq_usage", "violation"},
    ("process_contracts", "SAFE_BRIDGE", "target_minor_orientation"): {"minor_code_source", "haplotype_formula", "diploid_formula", "missing_formula", "minor_code_domain", "minor_code_exported_to_MATERIALIZE", "invalid_or_ambiguous_orientation"},
    ("process_contracts", "SAFE_BRIDGE", "reference_minor_orientation"): {"minor_code_source", "minor_ac_formula", "callable_an_formula", "minor_af_formula", "missing_rule", "raw_state_domain", "minor_code_domain", "invalid_or_ambiguous_state_or_orientation"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts"): {"selected_loci_incremental", "target_rare_diploid_incremental", "reference_rare_summary_incremental", "safe_bridge_receipt"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "*"): {"schema_id", "format", "axes", "dtypes", "layout", "byte_order", "contains_raw_input_payload", "privacy"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "reference_rare_summary_incremental"): {"schema_id", "format", "axes", "dtypes", "enum_mappings", "layout", "byte_order", "contains_raw_input_payload", "privacy"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "safe_bridge_receipt"): {"schema_id", "format", "axes", "dtypes", "layout", "byte_order", "contains_raw_input_payload", "privacy", "required_keys"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "reference_rare_summary_incremental", "enum_mappings"): {"ancestry"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "reference_rare_summary_incremental", "enum_mappings", "ancestry"): {"0", "1", "2"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "selected_loci_incremental", "dtypes"): {"locus_id", "chrom", "pos", "ref", "alt", "cM"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "target_rare_diploid_incremental", "dtypes"): {"sample_key_sha256", "locus_id", "minor_dosage", "observed_mask"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "reference_rare_summary_incremental", "dtypes"): {"ancestry", "locus_id", "minor_ac", "callable_an", "minor_af", "observed_mask", "no_support"},
    ("process_contracts", "SAFE_BRIDGE", "output_artifacts", "safe_bridge_receipt", "dtypes"): {"semantic_sha256"},
    ("process_contracts", "MATERIALIZE"): {"implemented", "input_logical_ids", "output_logical_ids", "bundle_partitioning", "fit_callable_normalization_manifest", "channel_cast", "output_artifacts", "forbidden_namespace_tokens_case_insensitive"},
    ("process_contracts", "MATERIALIZE", "bundle_partitioning"): {"unit", "row_order", "token_order_within_row", "person_batch", "maximum_valid_tokens_per_shard"},
    ("process_contracts", "MATERIALIZE", "fit_callable_normalization_manifest"): {"schema_id", "required_keys", "forbid_extra_keys", "profiles", "development_rotations", "forbidden_eval_root_seeds", "consumed_technical_root_materialization", "ancestry_order", "max_callable_rule", "score_only_or_eval_contribution"},
    ("process_contracts", "MATERIALIZE", "fit_callable_normalization_manifest", "profiles"): {"DEVELOPMENT_ROTATION"},
    ("process_contracts", "MATERIALIZE", "fit_callable_normalization_manifest", "development_rotations"): {"R0", "R1", "R2"},
    ("process_contracts", "MATERIALIZE", "fit_callable_normalization_manifest", "development_rotations", "*"): {"fit_root_seeds", "score_only_root_seed"},
    ("process_contracts", "MATERIALIZE", "output_artifacts"): {"packed_rare_context_shard", "bundle_manifest", "materialization_receipt", "READY"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "packed_rare_context_shard"): {"schema_id", "format", "privacy", "arrays", "invariants"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "packed_rare_context_shard", "arrays"): {"sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref", "marker_alt", "marker_cM", "radius_cM", "rare_tokens", "rare_mask", "rare_locus_index", "row_ptr", "row_sample_index", "row_marker_index", "F0"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "packed_rare_context_shard", "arrays", "*"): {"axes", "shape", "dtype"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "bundle_manifest"): {"schema_id", "format", "privacy", "required_keys", "forbid_extra_keys", "field_contracts", "ordered_shard_entry_schema", "ordering_and_coverage"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "bundle_manifest", "field_contracts"): {"stage", "schema_id", "status", "root_label", "root_seed", "rotation_id", "role_in_rotation", "radius_cM", "fit_callable_normalization_manifest_sha256", "sample_count", "sample_axis_semantic_sha256", "marker_count", "marker_axis_semantic_sha256", "ordered_shards", "raw_semantic_sha256", "channel_semantic_sha256", "source_auth_sha256"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "bundle_manifest", "ordered_shard_entry_schema"): {"exact_keys", "field_contracts"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "bundle_manifest", "ordered_shard_entry_schema", "field_contracts"): {"schema_id", "shard_ordinal", "person_start", "person_end_exclusive", "marker_start", "marker_end_exclusive", "valid_token_count", "gcs_uri", "gcs_generation", "raw_sha256", "semantic_sha256"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "materialization_receipt"): {"schema_id", "format", "privacy", "required_keys", "forbid_extra_keys", "field_contracts", "radius_manifest_map"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "materialization_receipt", "field_contracts"): {"stage", "schema_id", "status", "root_label", "root_seed", "rotation_id", "role_in_rotation", "radii_cM", "fit_callable_normalization_manifest_sha256", "sample_count", "sample_axis_semantic_sha256", "marker_count", "marker_axis_semantic_sha256", "ordered_bundle_manifest_sha256_by_radius", "raw_semantic_sha256", "channel_semantic_sha256", "source_auth_sha256", "reopen_verified", "append_only"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "materialization_receipt", "radius_manifest_map"): {"exact_order", "value"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "READY"): {"schema_id", "format", "privacy", "required_keys", "forbid_extra_keys", "field_contracts", "write_order"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "READY", "field_contracts"): {"stage", "schema_id", "status", "root_label", "root_seed", "rotation_id", "role_in_rotation", "fit_callable_normalization_manifest_sha256", "sample_count", "sample_axis_semantic_sha256", "marker_count", "marker_axis_semantic_sha256", "materialization_receipt_sha256", "ordered_bundle_manifest_sha256_by_radius", "source_auth_sha256"},
    ("incremental_partition",): {"model_input", "exact_key", "required_identity", "cross_source_identity_guard", "ref_alt_swap_or_allele_mismatch_at_same_chrom_pos", "selected_all_and_flare_overlap", "overlap_loci_must_not_enter_model", "duplicate_key_or_locus_id"},
    ("reference_semantics",): {"raw_bridge_fields", "minor_af", "observed_mask", "no_support", "no_support_is_explicit_bridge_audit_field_not_an_extra_tensor_channel", "dosage_mean_as_af", "callable_normalization"},
    ("reference_semantics", "callable_normalization"): {"technical_denominator_by_root", "technical_status", "future_scientific_denominator", "future_manifest_required_before_materialize", "raw_callable_an_is_persisted_unchanged"},
    ("reference_semantics", "callable_normalization", "technical_denominator_by_root"): {"root17", "root18"},
    ("sample_key_contract",): {"algorithm", "domain_separator", "input_encoding", "output_encoding", "formula", "f0_join", "axis_join", "duplicate_or_collision", "privacy", "not_anonymization"},
    ("f0_contract",): {"join_key", "probability_fields", "ancestry_order", "raw_sum_range", "operation", "float32_simplex_absolute_tolerance", "haplotype_axis_preserved", "forbidden_dependencies"},
    ("control_views",): {"rare_disabled_RD", "rare_enabled_RE", "target_same_locus_sham", "REF_label_sham", "common_matched", "cross_radius_invariant", "control_or_cross_radius_violation"},
    ("control_views", "target_same_locus_sham"): {"replicates", "seeds", "operation"},
    ("control_views", "REF_label_sham"): {"replicates", "seeds", "operation"},
    ("primary_transferable_input",): {"storage", "conceptual_shapes", "phase_policy", "channel_count", "channel_dtype", "calculation_dtype_before_cast", "first_ordered_locus_delta_cM", "channels", "rare_order", "missing_policy"},
    ("primary_transferable_input", "conceptual_shapes"): {"rare_tokens", "rare_mask", "F0"},
    ("packed_loader",): {"mode", "person_batch", "token_budget", "token_definition", "radii_cm", "radius_selection", "interval", "global_padding", "truncation", "empty_context", "single_context_over_budget", "warning_memory_fraction", "stop_memory_fraction"},
    ("privacy_contract",): {"classification", "gcs_access_gate", "encryption", "retention", "external_sharing", "sample_hash_interpretation", "acl_or_privacy_failure"},
    ("source_auth_policy",): {"status", "initial_anchor", "minimum_covered_now", "future_implementation_must_also_cover", "dirty_or_untracked_source"},
    ("persistence_contract",): {"write", "reopen", "semantic_hash", "receipt", "bundle_transaction", "partial_bundle_policy", "gcs_policy"},
    ("persistence_contract", "semantic_hash"): {"algorithm", "canonicalization", "archive_metadata_excluded", "required_for_every_output_artifact"},
    ("execution_authorization",): {"contract_validation", "synthetic_known_answer_tests", "real_asset_read", "derive_index", "safe_bridge", "materialize", "write_READY", "forward", "backward", "training", "truth_scoring"},
}


def _schema_for(path: tuple[str, ...]) -> set[str] | None:
    if path in OBJECT_KEYS:
        return OBJECT_KEYS[path]
    wildcard = path[:-1] + ("*",) if path else path
    return OBJECT_KEYS.get(wildcard)


def validate_exact_keys_recursive(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        schema = _schema_for(path)
        require(schema is not None, f"unregistered object schema at {'.'.join(path) or '<root>'}")
        exact_keys(value, schema, ".".join(path) or "<root>")
        for key, nested in value.items():
            validate_exact_keys_recursive(nested, path + (key,))
    elif isinstance(value, list):
        for nested in value:
            validate_exact_keys_recursive(nested, path)


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _flatten_strings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for nested in value:
            yield from _flatten_strings(nested)


def _valid_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def contract_semantic_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(contract, allow_nan=False, ensure_ascii=False,
                         separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_contract(contract: Mapping[str, Any]) -> None:
    validate_exact_keys_recursive(contract)
    require(contract_semantic_sha256(contract) == EXPECTED_CONTRACT_SEMANTIC_SHA256,
            "contract semantic fingerprint drift")
    require(contract["schema_version"] == "2.0.0", "wrong schema version")
    require(contract["stage"] == "M33_M0_MATERIALIZER_CONTRACT", "wrong stage")
    require(contract["scope"] == "technical_chr22_root17_root18_only", "wrong M0 scope")
    require(contract["status"].startswith("CONTRACT_ONLY_"), "not contract-only")
    require(contract["supersedes_historical_contracts"] is False, "historical contract was superseded")
    anchors = contract["anchors"]
    require(anchors["a0_root17"] == EXPECTED_A0, "A0 anchors differ from official root17")
    require(anchors["a0_root18"] == EXPECTED_A0_ROOT18, "A0 anchors differ from official root18")
    kat17, kat18 = anchors["known_answer_root17"], anchors["known_answer_root18"]
    require(kat17["profile"] == "official_root17_A0_KAT" and kat17["counts"] == EXPECTED_KAT_COUNTS and kat17["sha256"] == EXPECTED_KAT_HASHES, "root17 KAT differs")
    require(kat18["profile"] == "official_root18_A0_KAT" and kat18["counts"] == EXPECTED_KAT18_COUNTS and kat18["sha256"] == EXPECTED_KAT18_HASHES, "root18 KAT differs")
    require(all(_valid_hex64(value) for kat in (kat17, kat18) for value in kat["sha256"].values()), "invalid KAT SHA-256")
    require(anchors["pre4_contract"] == {
        "git_commit": "9f2214f5eaa2c5ab02e5df89528282569cffac4d",
        "preregistration_sha256": "4308bbf33ae28f554f701da33efdc185264f9f407d62661e7048e0345687eb8b",
    }, "PRE4 anchor changed")

    roots = contract["root_registry"]
    require(roots["consumed_technical_roots"] == {"root17": 20260817, "root18": 20260818}, "M0 must consume exactly the two technical A0 roots")
    require(roots["root18_status"] == "TECHNICAL_A0_COMPLETE_CONSUMED_NO_SCIENTIFIC_VALIDATION", "root18 status changed")
    require(roots["consumed_roots_quarantine"] ==
            "root17_and_root18_may_only_serve_as_technical_known_answers_never_fit_validation_test_radius_or_model_selection",
            "consumed technical roots escaped quarantine")
    require(roots["scientific_selection"] is False and roots["radius_selection"] is False, "technical root cannot select science")

    processes = contract["process_contracts"]
    require(list(processes) == ["I0_DERIVE_AUTHENTICATE_FLARE_INDEX", "SAFE_BRIDGE", "MATERIALIZE"], "process order changed")
    require(all(processes[name]["implemented"] is False for name in processes), "contract accidentally claims implementation")
    i0 = processes["I0_DERIVE_AUTHENTICATE_FLARE_INDEX"]
    require(i0["tool"]["name"] == "tabix" and i0["tool"]["exact_version"] == "1.16", "I0 requires exact tabix 1.16")
    require(i0["status"] == "BLOCKED_PENDING_PULLABLE_TABIX_OCI", "I0 was opened without a pullable OCI image")
    require(i0["tool"] == {
        "name": "tabix", "exact_version": "1.16",
        "local_image_id_technical_anchor": "sha256:b89353efc9a4a5953519fa9f066728e2f63d0e9125fc9fc771ef3ea9bb9c961c",
        "required_pullable_oci_image": "BLOCKED_PENDING_REPOSITORY_AT_SHA256_DIGEST",
    }, "I0 tool contract changed")
    require(i0["exact_command_argv"] == ["tabix", "-p", "vcf", "flare.anc.vcf.gz"], "I0 command changed")
    require(i0["input_logical_ids"] == ["a0_authenticated_flare_anc"] and
            i0["output_logical_ids"] == ["derived_flare_anc_tbi", "i0_index_receipt"],
            "I0 logical boundary changed")
    require(i0["write_policy"] == "append_only_fail_if_output_exists_atomic_fsync_reopen", "I0 is not append-only")
    require(i0["requirements"] == [
        "copy_authenticated_flare_anc_without_modification",
        "verify_raw_flare_anc_sha256_before_and_after",
        "build_tbi_twice_in_independent_clean_directories_with_pinned_tabix_image",
        "require_identical_tbi_sha256", "require_indexed_and_sequential_query_parity",
        "record_derived_index_provenance_without_claiming_FLARE_produced_the_index",
    ], "I0 requirements changed")
    require(i0["receipt_required_keys"] == [
        "stage", "status", "root_label", "root_seed", "source_flare_sha256", "tabix_version",
        "tabix_oci_repository_digest", "independent_tbi_sha256", "query_parity_sha256",
        "output_tbi_sha256", "append_only", "reopen_verified",
    ], "I0 receipt keys changed")

    bridge = processes["SAFE_BRIDGE"]
    require(bridge["truth_mounted"] is False, "truth must not enter SAFE_BRIDGE")
    require(set(bridge["physical_inputs"]) >= {"tree_sequence", "pools", "ref_pairs", "panel_map", "genetic_map"}, "SAFE_BRIDGE physical mapping incomplete")
    require(bridge["physical_inputs"]["pools"]["format"] == "tsv", "pools format must be TSV")
    for name in ("rare_catalog", "rare_haplotypes", "selected_sites", "target_calls"):
        require(bridge["physical_inputs"][name]["format"] == "tsv_gzip", f"{name} format must be TSV.gz")
    require(bridge["derivations"]["reference_rare_summary_incremental"] ==
            "for_each_ancestry_and_locus_over_tree_sequence_REF_nodes_sum_indicator_raw_binary_state_equals_authenticated_rare_catalog_minor_code_and_count_nonmissing_binary_states_as_callable_an",
            "REF rare source or minor orientation is wrong")
    require("never_source" in bridge["derivations"]["common_reference_crosscheck_scope"], "common REF VCF could be misused as rare source")
    require(all(item["access"] == "read_only" for item in bridge["physical_inputs"].values()), "raw bridge input is writable")
    require(all(item["authentication"] == "A0_receipt_input_sha256" for item in bridge["physical_inputs"].values()), "SAFE_BRIDGE input authentication changed")
    require(all(item["contains_raw_input_payload"] is False for item in bridge["output_artifacts"].values()), "raw payload escapes SAFE_BRIDGE")
    require(all(item["privacy"] == "private" for item in bridge["output_artifacts"].values()), "SAFE_BRIDGE privacy changed")
    require(all(item["byte_order"] in {"little_endian_for_multibyte_numeric_fields", "not_applicable"} for item in bridge["output_artifacts"].values()), "output byte order is ambiguous")
    require(bridge["role_firewall"]["forbidden_genotype_contributors"] == ["DONOR", "FREQ", "TARGET", "VALID", "TEST"] and
            bridge["role_firewall"]["violation"] == "STOP", "SAFE_BRIDGE role firewall changed")
    orientation = bridge["target_minor_orientation"]
    require(orientation["minor_code_domain"] == [0, 1] and
            orientation["haplotype_formula"] == "minor_indicator_equals_1_if_raw_binary_state_equals_minor_code_else_0" and
            orientation["minor_code_exported_to_MATERIALIZE"] is False and
            orientation["invalid_or_ambiguous_orientation"] == "STOP", "TARGET minor orientation changed")
    ref_orientation = bridge["reference_minor_orientation"]
    require(ref_orientation["minor_code_domain"] == [0, 1] and
            ref_orientation["minor_ac_formula"] ==
                "sum_indicator_raw_REF_haplotype_state_equals_minor_code_within_ancestry_and_locus" and
            ref_orientation["callable_an_formula"] ==
                "count_raw_REF_haplotype_states_in_0_1_within_ancestry_and_locus" and
            ref_orientation["missing_rule"] ==
                "missing_REF_state_contributes_neither_minor_ac_nor_callable_an" and
            ref_orientation["invalid_or_ambiguous_state_or_orientation"] == "STOP",
            "REF minor orientation changed")
    bridge_receipt = bridge["output_artifacts"]["safe_bridge_receipt"]
    require(bridge_receipt["schema_id"] == "m33_m0_safe_bridge_receipt_v1" and
            bridge_receipt["required_keys"] == [
                "stage", "status", "root_label", "root_seed", "expected_ref_node_count",
                "contributing_ref_node_count", "rejected_non_ref_node_count",
                "expected_ref_nodes_semantic_sha256", "contributing_ref_nodes_semantic_sha256",
                "role_firewall_pass", "selected_all_count", "selected_incremental_count",
                "selected_overlap_count", "selected_all_semantic_sha256",
                "selected_incremental_semantic_sha256", "selected_overlap_semantic_sha256",
                "partition_disjoint_union_pass", "minor_code_0_locus_count", "minor_code_1_locus_count",
                "minor_orientation_source_semantic_sha256", "target_minor_dosage_semantic_sha256",
                "reference_minor_summary_semantic_sha256",
                "artifact_semantic_sha256", "reopen_verified", "append_only",
            ], "SAFE_BRIDGE firewall receipt changed")

    materialize = processes["MATERIALIZE"]
    namespace = " ".join(_flatten_strings({"inputs": materialize["input_logical_ids"], "outputs": materialize["output_logical_ids"]})).lower()
    for token in materialize["forbidden_namespace_tokens_case_insensitive"]:
        require(token.lower() not in namespace, f"forbidden MATERIALIZE namespace token: {token}")
    require(materialize["input_logical_ids"] == [
        "selected_loci_incremental", "target_rare_diploid_incremental",
        "reference_rare_summary_incremental", "a0_authenticated_flare_anc",
        "derived_flare_anc_tbi", "authenticated_genetic_map",
        "authenticated_fit_callable_normalization_manifest",
    ], "MATERIALIZE inputs changed")
    require(materialize["output_logical_ids"] == [
        "packed_rare_context", "bundle_manifest_by_radius", "materialization_receipt", "READY",
    ], "MATERIALIZE outputs changed")
    packed_schema = materialize["output_artifacts"]["packed_rare_context_shard"]
    fit_manifest = materialize["fit_callable_normalization_manifest"]
    require(fit_manifest["schema_id"] == "m33_m0_fit_callable_normalization_manifest_v1" and
            fit_manifest["ancestry_order"] == ["AFR", "EUR", "ASIA"] and
            fit_manifest["score_only_or_eval_contribution"] == "STOP_BEFORE_ANY_SHARD_WRITE" and
            fit_manifest["forbid_extra_keys"] is True, "FIT normalization manifest changed")
    require(fit_manifest["profiles"] == {
                "DEVELOPMENT_ROTATION": "derive_each_ancestry_maxAN_only_from_authenticated_FIT_roots_excluding_score_only_and_EVAL"
            } and fit_manifest["development_rotations"] == DEVELOPMENT_ROTATIONS and
            fit_manifest["forbidden_eval_root_seeds"] == FORBIDDEN_EVAL_ROOTS and
            fit_manifest["consumed_technical_root_materialization"] ==
                "STOP_root17_root18_are_SAFE_BRIDGE_KAT_only",
            "FIT rotation registry or quarantine changed")
    require(packed_schema["schema_id"] == "m33_m0_packed_rare_context_shard_v1", "packed schema_id changed")
    require(packed_schema["arrays"]["rare_tokens"] ==
            {"axes": ["valid_token", "channel"], "shape": ["N", 13], "dtype": "<f4"},
            "packed token schema changed")
    require(packed_schema["arrays"]["F0"] ==
            {"axes": ["sample", "haplotype", "marker", "ancestry"],
             "shape": ["B", 2, "M", 3], "dtype": "<f4"}, "F0 physical schema changed")
    require(packed_schema["arrays"]["row_ptr"]["dtype"] == "<u8" and
            packed_schema["arrays"]["rare_mask"]["dtype"] == "|u1", "packed indexing schema changed")
    bundle_manifest = materialize["output_artifacts"]["bundle_manifest"]
    require(bundle_manifest["forbid_extra_keys"] is True and
            bundle_manifest["required_keys"] == list(BUNDLE_REQUIRED_KEYS) and
            "FIT_iff_root" in bundle_manifest["field_contracts"]["role_in_rotation"] and
            bundle_manifest["ordered_shard_entry_schema"]["exact_keys"] == [
                "schema_id", "shard_ordinal", "person_start", "person_end_exclusive",
                "marker_start", "marker_end_exclusive", "valid_token_count", "gcs_uri",
                "gcs_generation", "raw_sha256", "semantic_sha256",
            ], "bundle manifest shard schema changed")
    material_receipt = materialize["output_artifacts"]["materialization_receipt"]
    require(material_receipt["required_keys"] == list(RECEIPT_REQUIRED_KEYS) and
            "READY_sha256" not in material_receipt["required_keys"] and
            "ordered_bundle_manifest_sha256_by_radius" in material_receipt["required_keys"] and
            "FIT_iff_root" in material_receipt["field_contracts"]["role_in_rotation"] and
            material_receipt["radius_manifest_map"]["exact_order"] == ["0.05", "0.1", "0.2", "0.5"],
            "materialization receipt radius manifests changed")
    ready = materialize["output_artifacts"]["READY"]
    require(ready["schema_id"] == "m33_m0_READY_v1" and ready["forbid_extra_keys"] is True and
            ready["required_keys"] == list(READY_REQUIRED_KEYS) and
            "FIT_iff_root" in ready["field_contracts"]["role_in_rotation"] and
            "materialization_receipt_sha256" in ready["required_keys"] and
            ready["write_order"].startswith("last_after_all_shards"), "READY schema/order changed")

    partition = contract["incremental_partition"]
    require(partition["model_input"] == "selected_loci_incremental_only" and partition["overlap_loci_must_not_enter_model"] is True, "model input is not incremental-only")
    require(partition["exact_key"] == ["chrom", "pos", "ref", "alt"], "variant identity changed")
    require(partition["ref_alt_swap_or_allele_mismatch_at_same_chrom_pos"].startswith("STOP"), "REF/ALT mismatch can enter incremental partition")
    require(partition["required_identity"] == "selected_all_equals_disjoint_union_incremental_and_flare_overlap", "partition union invariant missing")
    reference = contract["reference_semantics"]
    require(reference["minor_af"] == "minor_ac_div_callable_an_if_callable_an_gt_0_else_0" and reference["dosage_mean_as_af"] is False, "wrong AF semantics")
    require(reference["raw_bridge_fields"] == ["minor_ac", "callable_an", "minor_af", "observed_mask", "no_support"], "raw AN or no_support missing")
    normalization = reference["callable_normalization"]
    require(normalization["technical_denominator_by_root"] == {"root17": 60, "root18": 60} and
            normalization["technical_status"] == "TECHNICAL_ONLY_EQUIVALENT_NEVER_REUSED_FOR_SCIENTIFIC_NORMALIZATION",
            "technical AN normalizer changed")
    require(normalization["future_manifest_required_before_materialize"] is True and normalization["raw_callable_an_is_persisted_unchanged"] is True, "future FIT/raw AN separation missing")

    f0 = contract["f0_contract"]
    require(f0["join_key"] == list(F0_KEY) and f0["probability_fields"] == ["ANP1", "ANP2"], "F0 source changed")
    require(f0["ancestry_order"] == ["AFR", "EUR", "ASIA"] and f0["raw_sum_range"] == [0.98, 1.02] and
            f0["operation"] == "renormalize_each_ANP_vector_to_exact_simplex_for_probability_operations",
            "F0 probability semantics changed")
    require(f0["float32_simplex_absolute_tolerance"] == 5e-06, "F0 float32 tolerance changed")
    require(f0["haplotype_axis_preserved"] is True, "F0 haplotype axis lost")
    require(f0["forbidden_dependencies"] == ["GT", "AN1", "AN2", "truth", "hard_call", "target_rare_phase"], "F0 forbidden dependencies changed")
    sample_key = contract["sample_key_contract"]
    require(sample_key["algorithm"] == "sha256" and sample_key["domain_separator"] == "DNABR_M33_M0_SAMPLE_V1|" and
            "bijection" in sample_key["f0_join"] and "ordered_target_sample_keys" in sample_key["axis_join"] and
            sample_key["duplicate_or_collision"] == "STOP" and "linkable" in sample_key["not_anonymization"],
            "sample-key join changed")
    primary = contract["primary_transferable_input"]
    require(primary["channel_count"] == 13 and primary["channels"] == EXPECTED_CHANNELS, "13-channel layout changed")
    require(primary["phase_policy"] == "target_rare_diploid_dosage_only_no_target_rare_phase", "Target phase leaked")
    require(primary["channel_dtype"] == "<f4" and primary["calculation_dtype_before_cast"] == "<f8" and
            primary["first_ordered_locus_delta_cM"] == 0.0, "channel cast/delta semantics changed")
    packed = contract["packed_loader"]
    require(packed["radii_cm"] == EXPECTED_RADII and packed["radius_selection"] is False, "four frozen radii changed")
    require(packed["mode"] == "contiguous_packed" and packed["global_padding"] is False and packed["truncation"] is False, "packed/no-truncation contract changed")
    require(packed["single_context_over_budget"] == "STOP", "oversized context must stop")
    require(packed["person_batch"] == 8 and packed["token_budget"] == 262144 and
            packed["warning_memory_fraction"] == 0.7 and packed["stop_memory_fraction"] == 0.8,
            "packed resource gates changed")
    controls = contract["control_views"]
    require("zero_channels_indices_0_2_3_5_6_8_9" in controls["rare_disabled_RD"] and
            controls["control_or_cross_radius_violation"] == "STOP" and
            "order_preserving_subsequence" in controls["cross_radius_invariant"], "control projections changed")
    require(controls["target_same_locus_sham"]["replicates"] == 3 and
            controls["target_same_locus_sham"]["seeds"] == [1277457345, 943666774, 1858042568] and
            controls["REF_label_sham"]["replicates"] == 3 and
            controls["REF_label_sham"]["seeds"] == [79351217, 202307732, 1737132171],
            "PRE4 sham controls changed")
    privacy = contract["privacy_contract"]
    require(privacy["external_sharing"] is False and "no_allUsers" in privacy["gcs_access_gate"] and
            privacy["acl_or_privacy_failure"] == "STOP_BEFORE_WRITE_OR_READY", "privacy gate changed")

    persistence = contract["persistence_contract"]
    require("atomic" in persistence["write"] and "no_overwrite" in persistence["write"] and "reopen" in persistence["reopen"], "atomic reopen contract missing")
    require(persistence["semantic_hash"]["archive_metadata_excluded"] is True and persistence["semantic_hash"]["required_for_every_output_artifact"] is True, "semantic hashes incomplete")
    require("READY" in persistence["bundle_transaction"] and "never_consumable" in persistence["partial_bundle_policy"] and
            "ifGenerationMatch_equals_0" in persistence["gcs_policy"], "bundle publication is not transactional")
    source_auth = contract["source_auth_policy"]
    require(source_auth["status"] == "REQUIRED_FROM_CLEAN_COMMIT_BEFORE_ANY_REAL_ASSET_READ" and
            source_auth["dirty_or_untracked_source"] == "STOP", "future source authentication changed")
    authorization = contract["execution_authorization"]
    require(authorization["contract_validation"] is True and authorization["synthetic_known_answer_tests"] is True, "contract tests not authorized")
    for action in ("real_asset_read", "derive_index", "safe_bridge", "materialize", "write_READY", "forward", "backward", "training", "truth_scoring"):
        require(authorization[action] is False, f"{action} was accidentally authorized")


def canonical_fixture_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


DTYPE_ITEMSIZE = {
    "|u1": 1, "|i1": 1, "|S1": 1, "|S64": 64,
    "<u2": 2, "<u4": 4, "<u8": 8, "<i8": 8, "<f4": 4, "<f8": 8,
}


def canonical_array_bundle_sha256(schema_id: str, arrays: Mapping[str, Mapping[str, Any]]) -> str:
    """Hash physical array semantics and canonical C-order bytes, not ZIP metadata."""
    require(isinstance(schema_id, str) and bool(schema_id), "schema_id is required")
    require(isinstance(arrays, Mapping) and bool(arrays), "array bundle is empty")
    digest = hashlib.sha256()
    digest.update((schema_id + "\n").encode("utf-8"))
    for name in sorted(arrays):
        item = arrays[name]
        exact_keys(item, ("axes", "shape", "dtype", "data"), f"array.{name}")
        axes, shape, dtype, data = item["axes"], item["shape"], item["dtype"], item["data"]
        require(isinstance(name, str) and bool(name), "array name is empty")
        require(isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)) and
                all(isinstance(axis, str) and axis for axis in axes) and len(set(axes)) == len(axes),
                f"invalid axes: {name}")
        require(isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and
                len(shape) == len(axes) and all(type(dim) is int and dim >= 0 for dim in shape),
                f"invalid shape: {name}")
        require(dtype in DTYPE_ITEMSIZE, f"unsupported dtype: {name}")
        require(isinstance(data, bytes), f"canonical data must be bytes: {name}")
        require(len(data) == math.prod(shape) * DTYPE_ITEMSIZE[dtype],
                f"byte length differs from shape/dtype: {name}")
        header = json.dumps({"name": name, "axes": list(axes), "shape": list(shape), "dtype": dtype},
                            separators=(",", ":"), sort_keys=True).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def sample_key_sha256(sample_id: str) -> str:
    require(isinstance(sample_id, str) and bool(sample_id), "sample_id is required")
    return hashlib.sha256(("DNABR_M33_M0_SAMPLE_V1|" + sample_id).encode("utf-8")).hexdigest()


def validate_sample_axis_join(target_sample_keys: Sequence[str], f0_sample_ids: Sequence[str]) -> None:
    require(bool(target_sample_keys) and len(target_sample_keys) == len(f0_sample_ids), "sample axes differ")
    require(all(_valid_hex64(value) for value in target_sample_keys), "invalid target sample key")
    require(len(set(target_sample_keys)) == len(target_sample_keys), "duplicate target sample key")
    expected = [sample_key_sha256(sample_id) for sample_id in f0_sample_ids]
    require(len(set(expected)) == len(expected), "duplicate F0 sample ID or sample-key collision")
    require(list(target_sample_keys) == expected, "ordered TARGET/F0 sample axes differ")


def fit_manifest_semantic_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash all manifest semantics except the self-referential digest field."""
    payload = {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    return canonical_fixture_sha256(payload)


def validate_fit_normalization_manifest(
        manifest: Mapping[str, Any],
        authenticated_source_hashes: Mapping[str, str],
        observed_maxima: Mapping[str, int],
        authenticated_source_auth_sha256: str) -> None:
    required = {
        "stage", "schema_id", "status", "profile", "rotation_id", "fit_root_seeds",
        "score_only_root_seed", "max_callable_an_by_ancestry",
        "source_reference_summary_sha256_by_fit_root", "semantic_sha256", "source_auth_sha256",
    }
    exact_keys(manifest, required, "fit_normalization_manifest")
    require(manifest["schema_id"] == "m33_m0_fit_callable_normalization_manifest_v1", "FIT manifest schema changed")
    require(manifest["status"] == "PASS" and manifest["profile"] == "DEVELOPMENT_ROTATION" and
            manifest["rotation_id"] in DEVELOPMENT_ROTATIONS,
            "FIT manifest status/rotation invalid")
    expected_rotation = DEVELOPMENT_ROTATIONS[manifest["rotation_id"]]
    roots = manifest["fit_root_seeds"]
    score = manifest["score_only_root_seed"]
    require(roots == expected_rotation["fit_root_seeds"] and
            score == expected_rotation["score_only_root_seed"], "FIT rotation roots changed")
    require(not (set(roots) | {score}) & set(FORBIDDEN_EVAL_ROOTS), "EVAL root leaked into development rotation")
    maxima = manifest["max_callable_an_by_ancestry"]
    require(isinstance(observed_maxima, Mapping) and set(observed_maxima) == {"AFR", "EUR", "ASIA"} and
            all(type(value) is int and value > 0 for value in observed_maxima.values()),
            "observed FIT maxAN invalid")
    require(maxima == dict(observed_maxima), "FIT maxAN differs from authenticated summaries")
    sources = manifest["source_reference_summary_sha256_by_fit_root"]
    require(isinstance(authenticated_source_hashes, Mapping) and
            set(authenticated_source_hashes) == {str(root) for root in roots} and
            all(_valid_hex64(value) for value in authenticated_source_hashes.values()),
            "authenticated FIT source hashes invalid")
    require(sources == dict(authenticated_source_hashes), "FIT source hashes differ from authenticated inputs")
    require(_valid_hex64(authenticated_source_auth_sha256) and
            manifest["source_auth_sha256"] == authenticated_source_auth_sha256 and
            manifest["semantic_sha256"] == fit_manifest_semantic_sha256(manifest),
            "FIT manifest hashes invalid")


def validate_development_rotation_role(rotation_id: str, root_seed: int, role: str) -> None:
    """Bind a development root to its immutable PRE4 role."""
    require(rotation_id in DEVELOPMENT_ROTATIONS, "unknown development rotation")
    require(type(root_seed) is int and root_seed not in FORBIDDEN_EVAL_ROOTS and
            root_seed not in {20260817, 20260818}, "EVAL or technical root entered materialization")
    rotation = DEVELOPMENT_ROTATIONS[rotation_id]
    if root_seed in rotation["fit_root_seeds"]:
        require(role == "FIT", "FIT root was relabeled")
    elif root_seed == rotation["score_only_root_seed"]:
        require(role == "SCORE", "SCORE root was relabeled")
    else:
        raise ValueError("root is not registered in development rotation")


def validate_ordered_shards(shards: Sequence[Mapping[str, Any]], sample_count: int,
                            marker_count: int, expected_gcs_prefix: str) -> None:
    """Validate exact shard metadata and complete person-batch/marker coverage."""
    require(isinstance(shards, list) and bool(shards), "ordered shard list is empty or invalid")
    require(type(sample_count) is int and sample_count > 0 and type(marker_count) is int and marker_count > 0,
            "sample or marker count invalid")
    require(isinstance(expected_gcs_prefix, str) and expected_gcs_prefix.startswith("gs://") and
            expected_gcs_prefix.endswith("/"), "expected GCS radius prefix invalid")
    seen_uri: set[str] = set()
    seen_generation: set[tuple[str, int]] = set()
    seen_raw: set[str] = set()
    seen_semantic: set[str] = set()
    groups: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    current_batch: tuple[int, int] | None = None
    intervals: list[tuple[int, int]] = []
    closed_batches: set[tuple[int, int]] = set()
    for ordinal, shard in enumerate(shards):
        require(isinstance(shard, Mapping), "shard entry is not an object")
        exact_keys(shard, SHARD_REQUIRED_KEYS, f"ordered_shards[{ordinal}]")
        require(shard["schema_id"] == "m33_m0_packed_rare_context_shard_v1" and
                type(shard["shard_ordinal"]) is int and shard["shard_ordinal"] == ordinal,
                "shard schema or ordinal invalid")
        ps, pe = shard["person_start"], shard["person_end_exclusive"]
        ms, me = shard["marker_start"], shard["marker_end_exclusive"]
        count, generation = shard["valid_token_count"], shard["gcs_generation"]
        require(all(type(value) is int for value in (ps, pe, ms, me, count, generation)),
                "shard integer field has invalid type")
        require(0 <= ps < pe <= sample_count and pe - ps <= 8 and 0 <= ms < me <= marker_count,
                "shard person/marker range invalid")
        require(0 <= count <= 262144 and generation > 0, "shard token count or generation invalid")
        uri, raw_hash, semantic_hash = shard["gcs_uri"], shard["raw_sha256"], shard["semantic_sha256"]
        require(isinstance(uri, str) and uri.startswith(expected_gcs_prefix) and len(uri) > len(expected_gcs_prefix),
                "shard URI is outside authenticated radius prefix")
        require(_valid_hex64(raw_hash) and _valid_hex64(semantic_hash), "shard hash invalid")
        require(uri not in seen_uri and (uri, generation) not in seen_generation and
                raw_hash not in seen_raw and semantic_hash not in seen_semantic,
                "duplicate shard URI, generation or hash")
        seen_uri.add(uri)
        seen_generation.add((uri, generation))
        seen_raw.add(raw_hash)
        seen_semantic.add(semantic_hash)
        batch = (ps, pe)
        if current_batch != batch:
            if current_batch is not None:
                groups.append((current_batch, intervals))
                closed_batches.add(current_batch)
            require(batch not in closed_batches, "person batch reappears out of order")
            current_batch, intervals = batch, []
        intervals.append((ms, me))
    require(current_batch is not None, "ordered shard list is empty")
    groups.append((current_batch, intervals))
    person_cursor = 0
    for (ps, pe), marker_intervals in groups:
        require(ps == person_cursor, "person batches have a gap or overlap")
        marker_cursor = 0
        for ms, me in marker_intervals:
            require(ms == marker_cursor, "marker shards have a gap, overlap or wrong order")
            marker_cursor = me
        require(marker_cursor == marker_count, "person batch does not cover all markers")
        person_cursor = pe
    require(person_cursor == sample_count, "person batches do not cover all samples")


def validate_materialization_output_chain(
        bundles_by_radius: Mapping[str, Mapping[str, Any]],
        bundle_raw_sha256_by_radius: Mapping[str, str],
        authenticated_shards_by_radius: Mapping[str, Sequence[Mapping[str, Any]]],
        expected_gcs_prefix_by_radius: Mapping[str, str],
        receipt: Mapping[str, Any],
        receipt_raw_sha256: str,
        ready: Mapping[str, Any],
        authenticated_fit_manifest_sha256: str,
        authenticated_source_auth_sha256: str,
        expected_rotation_id: str,
        expected_root_seed: int,
        expected_role: str) -> None:
    """Validate role binding and transitive identity across bundle, receipt and READY."""
    require(list(bundles_by_radius) == RADIUS_KEYS and list(bundle_raw_sha256_by_radius) == RADIUS_KEYS and
            list(authenticated_shards_by_radius) == RADIUS_KEYS and
            list(expected_gcs_prefix_by_radius) == RADIUS_KEYS, "output-chain radius order changed")
    require(all(_valid_hex64(value) for value in bundle_raw_sha256_by_radius.values()) and
            _valid_hex64(receipt_raw_sha256) and _valid_hex64(authenticated_fit_manifest_sha256) and
            _valid_hex64(authenticated_source_auth_sha256), "output-chain authenticated hash invalid")
    exact_keys(receipt, RECEIPT_REQUIRED_KEYS, "materialization_receipt")
    exact_keys(ready, READY_REQUIRED_KEYS, "READY")
    require(receipt["stage"] == "M33_M0_MATERIALIZATION_RECEIPT" and
            receipt["schema_id"] == "m33_m0_materialization_receipt_v1" and receipt["status"] == "PASS" and
            receipt["radii_cM"] == EXPECTED_RADII and receipt["reopen_verified"] is True and
            receipt["append_only"] is True, "materialization receipt semantics changed")
    require(ready["stage"] == "M33_M0_READY" and ready["schema_id"] == "m33_m0_READY_v1" and
            ready["status"] == "PASS", "READY semantics changed")
    validate_development_rotation_role(expected_rotation_id, expected_root_seed, expected_role)
    validate_development_rotation_role(receipt["rotation_id"], receipt["root_seed"], receipt["role_in_rotation"])
    require((receipt["rotation_id"], receipt["root_seed"], receipt["role_in_rotation"]) ==
            (expected_rotation_id, expected_root_seed, expected_role),
            "receipt rotation/root/role differs from authenticated execution plan")
    require(receipt["fit_callable_normalization_manifest_sha256"] == authenticated_fit_manifest_sha256 and
            receipt["source_auth_sha256"] == authenticated_source_auth_sha256,
            "receipt is not bound to authenticated FIT/source manifests")
    require(receipt["ordered_bundle_manifest_sha256_by_radius"] == dict(bundle_raw_sha256_by_radius) and
            ready["ordered_bundle_manifest_sha256_by_radius"] == dict(bundle_raw_sha256_by_radius) and
            ready["materialization_receipt_sha256"] == receipt_raw_sha256,
            "bundle/receipt/READY raw hashes differ")
    require(list(receipt["ordered_bundle_manifest_sha256_by_radius"]) == RADIUS_KEYS and
            list(ready["ordered_bundle_manifest_sha256_by_radius"]) == RADIUS_KEYS,
            "receipt or READY radius-map order changed")
    require(all(type(receipt[key]) is int and receipt[key] > 0 for key in ("sample_count", "marker_count")) and
            all(_valid_hex64(receipt[key]) for key in (
                "fit_callable_normalization_manifest_sha256", "sample_axis_semantic_sha256",
                "marker_axis_semantic_sha256", "raw_semantic_sha256", "channel_semantic_sha256",
                "source_auth_sha256")), "receipt counts or semantic hashes invalid")
    for field in OUTPUT_CHAIN_IDENTITY_FIELDS:
        require(ready[field] == receipt[field], f"READY/receipt identity drift: {field}")
    channel_hash: str | None = None
    for radius_key, expected_radius in zip(RADIUS_KEYS, EXPECTED_RADII):
        bundle = bundles_by_radius[radius_key]
        exact_keys(bundle, BUNDLE_REQUIRED_KEYS, f"bundle[{radius_key}]")
        require(bundle["stage"] == "M33_M0_MATERIALIZE_BUNDLE" and
                bundle["schema_id"] == "m33_m0_bundle_manifest_v1" and bundle["status"] == "PASS" and
                bundle["radius_cM"] == expected_radius and _valid_hex64(bundle["raw_semantic_sha256"]) and
                _valid_hex64(bundle["channel_semantic_sha256"]), f"bundle semantics invalid: {radius_key}")
        require(bundle["ordered_shards"] == list(authenticated_shards_by_radius[radius_key]),
                "bundle shard metadata differs from reopened authenticated shards")
        validate_ordered_shards(bundle["ordered_shards"], receipt["sample_count"], receipt["marker_count"],
                                expected_gcs_prefix_by_radius[radius_key])
        validate_development_rotation_role(bundle["rotation_id"], bundle["root_seed"], bundle["role_in_rotation"])
        for field in OUTPUT_CHAIN_IDENTITY_FIELDS:
            require(bundle[field] == receipt[field], f"bundle/receipt identity drift: {radius_key}:{field}")
        if channel_hash is None:
            channel_hash = bundle["channel_semantic_sha256"]
        require(bundle["channel_semantic_sha256"] == channel_hash == receipt["channel_semantic_sha256"],
                "channel semantics drift across radii or receipt")


def validate_float32_simplex(values: Sequence[float]) -> None:
    require(len(values) == 3 and all(type(value) in (int, float) and not isinstance(value, bool) and
                                    math.isfinite(value) and 0.0 <= value <= 1.0 for value in values),
            "float32 ancestry vector invalid")
    require(abs(sum(values) - 1.0) <= 5e-6, "float32 ancestry vector outside frozen simplex tolerance")


def validate_cross_radius_loci(loci_by_radius: Mapping[float, Sequence[int]]) -> None:
    require(list(loci_by_radius) == EXPECTED_RADII, "radius order/domain changed")
    previous: list[int] = []
    for radius in EXPECTED_RADII:
        observed = list(loci_by_radius[radius])
        require(all(type(value) is int and value >= 0 for value in observed), "invalid radius locus index")
        require(len(set(observed)) == len(observed), "duplicate radius locus index")
        cursor = iter(observed)
        require(all(any(candidate == expected for candidate in cursor) for expected in previous),
                "smaller-radius locus order is not a subsequence")
        previous = observed


def validate_cross_radius_payloads(loci_by_radius: Mapping[float, Sequence[int]],
                                   tokens_by_radius: Mapping[float, Sequence[Sequence[float]]]) -> None:
    validate_cross_radius_loci(loci_by_radius)
    require(list(tokens_by_radius) == EXPECTED_RADII, "token radius order/domain changed")
    value_channels = tuple(range(11))
    baseline: dict[int, tuple[float, ...]] = {}
    for radius in EXPECTED_RADII:
        loci, tokens = list(loci_by_radius[radius]), list(tokens_by_radius[radius])
        require(len(loci) == len(tokens), "radius locus/token lengths differ")
        current: dict[int, tuple[float, ...]] = {}
        for locus_id, token in zip(loci, tokens):
            require(len(token) == 13 and all(type(value) in (int, float) and not isinstance(value, bool) and
                                            math.isfinite(value) for value in token), "invalid cross-radius token")
            current[locus_id] = tuple(float(token[index]) for index in value_channels)
        for locus_id, values in baseline.items():
            if locus_id in current:
                require(current[locus_id] == values, "non-geometry token values changed across radii")
        baseline.update(current)


def reference_summary(minor_ac: int, callable_an: int) -> dict[str, float | int]:
    require(type(minor_ac) is int and type(callable_an) is int, "AC and AN must be non-boolean integers")
    require(callable_an >= 0 and 0 <= minor_ac <= callable_an, "require 0 <= AC <= AN")
    observed = int(callable_an > 0)
    return {"minor_af": minor_ac / callable_an if callable_an else 0.0,
            "observed_mask": observed, "no_support": int(observed == 1 and minor_ac == 0)}


def reference_minor_summary(raw_states: Sequence[int | None], minor_code: int) -> dict[str, float | int]:
    """Derive REF minor AC/AN from authenticated minor orientation, excluding missing states."""
    require(type(minor_code) is int and minor_code in (0, 1), "minor_code must be binary")
    require(isinstance(raw_states, Sequence) and not isinstance(raw_states, (str, bytes)),
            "REF states must be a sequence")
    minor_ac = 0
    callable_an = 0
    for state in raw_states:
        if state is None:
            continue
        require(type(state) is int and state in (0, 1), "REF state must be binary or missing")
        callable_an += 1
        minor_ac += int(state == minor_code)
    return {"minor_ac": minor_ac, "callable_an": callable_an,
            **reference_summary(minor_ac, callable_an)}


def validate_target_cell(minor_dosage: int, observed_mask: int) -> tuple[int, int]:
    require(type(observed_mask) is int and observed_mask in (0, 1), "observed mask must be integer 0/1, not bool")
    require(type(minor_dosage) is int, "dosage must be a non-boolean integer")
    require((observed_mask == 0 and minor_dosage == 0) or (observed_mask == 1 and minor_dosage in (0, 1, 2)),
            "missing dosage must be zero; observed dosage must be 0/1/2")
    return minor_dosage, observed_mask


def diploid_minor_dosage(haplotype0: int | None, haplotype1: int | None,
                         minor_code: int) -> tuple[int, int]:
    require(type(minor_code) is int and minor_code in (0, 1), "minor_code must be non-boolean 0/1")
    if haplotype0 is None or haplotype1 is None:
        return 0, 0
    require(type(haplotype0) is int and type(haplotype1) is int and haplotype0 in (0, 1) and haplotype1 in (0, 1), "haplotype states must be non-boolean binary integers")
    return validate_target_cell(int(haplotype0 == minor_code) + int(haplotype1 == minor_code), 1)


def _variant_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    exact_keys(row, LOCUS_KEYS, "locus")
    require(type(row["chrom"]) is int and row["chrom"] == 22, "CHROM must be integer 22")
    require(type(row["pos"]) is int and row["pos"] > 0, "POS must be a positive non-boolean integer")
    require(type(row["locus_id"]) is int and row["locus_id"] >= 0, "locus_id must be a nonnegative non-boolean integer")
    require(type(row["ref"]) is str and type(row["alt"]) is str and
            len(row["ref"]) == len(row["alt"]) == 1 and
            row["ref"] in "ACGT" and row["alt"] in "ACGT" and row["ref"] != row["alt"],
            "REF/ALT must be distinct SNV alleles")
    require(type(row["cM"]) in (int, float) and not isinstance(row["cM"], bool) and math.isfinite(row["cM"]), "cM must be finite")
    return row["chrom"], row["pos"], row["ref"], row["alt"]


def validate_locus_rows(rows: Sequence[Mapping[str, Any]], bp_domain: tuple[int, int], cm_domain: tuple[float, float]) -> None:
    require(bool(rows), "locus registry is empty")
    require(bp_domain[0] <= bp_domain[1] and cm_domain[0] <= cm_domain[1], "invalid domain")
    keys: set[tuple[int, int, str, str]] = set()
    locus_ids: set[int] = set()
    order: list[tuple[float, int, int]] = []
    for row in rows:
        key = _variant_key(row)
        require(key not in keys, "duplicate variant key")
        require(row["locus_id"] not in locus_ids, "duplicate locus_id")
        require(bp_domain[0] <= row["pos"] <= bp_domain[1] and cm_domain[0] <= float(row["cM"]) <= cm_domain[1], "locus outside map domain")
        keys.add(key)
        locus_ids.add(row["locus_id"])
        order.append((float(row["cM"]), row["pos"], row["locus_id"]))
    require(order == sorted(order), "loci must be ordered by cM, bp, locus_id; cM ties use deterministic tiebreaks")


def partition_incremental(rows: Sequence[Mapping[str, Any]], flare_keys: Sequence[tuple[int, int, str, str]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    require(bool(rows), "selected loci are empty")
    flare_set = set(flare_keys)
    require(len(flare_set) == len(flare_keys), "duplicate FLARE key")
    flare_by_position: dict[tuple[int, int], tuple[str, str]] = {}
    for chrom, pos, ref, alt in flare_keys:
        require(type(chrom) is int and chrom == 22 and type(pos) is int and pos > 0, "invalid FLARE position")
        require(ref in "ACGT" and alt in "ACGT" and len(ref) == len(alt) == 1 and ref != alt,
                "invalid FLARE alleles")
        position = (chrom, pos)
        require(position not in flare_by_position or flare_by_position[position] == (ref, alt),
                "multiple FLARE allele definitions at one position")
        flare_by_position[position] = (ref, alt)
    row_keys = [_variant_key(row) for row in rows]
    require(len(set(row_keys)) == len(row_keys), "duplicate selected key")
    locus_ids = [row["locus_id"] for row in rows]
    require(len(set(locus_ids)) == len(locus_ids), "duplicate selected locus_id")
    for chrom, pos, ref, alt in row_keys:
        observed = flare_by_position.get((chrom, pos))
        require(observed is None or observed == (ref, alt), "REF/ALT mismatch or swap at shared CHROM:POS")
    incremental = [row for row, key in zip(rows, row_keys) if key not in flare_set]
    overlap = [row for row, key in zip(rows, row_keys) if key in flare_set]
    require(len(incremental) + len(overlap) == len(rows), "partition does not reconstruct selected_all")
    require(set(_variant_key(row) for row in incremental).isdisjoint(_variant_key(row) for row in overlap), "partition is not disjoint")
    return incremental, overlap


def validate_genetic_map(rows: Sequence[Mapping[str, Any]], bp_domain: tuple[int, int]) -> None:
    require(bool(rows), "genetic map is empty")
    previous_pos = -1
    previous_cm = -math.inf
    for row in rows:
        exact_keys(row, ("chrom", "pos", "cM"), "genetic_map_row")
        require(type(row["chrom"]) is int and row["chrom"] == 22, "map CHROM must be integer 22")
        require(type(row["pos"]) is int and row["pos"] > previous_pos, "map positions must be strictly increasing")
        require(type(row["cM"]) in (int, float) and not isinstance(row["cM"], bool) and math.isfinite(row["cM"]), "map cM must be finite")
        require(float(row["cM"]) >= previous_cm, "map cM must be nondecreasing")
        previous_pos, previous_cm = row["pos"], float(row["cM"])
    require(rows[0]["pos"] <= bp_domain[0] and rows[-1]["pos"] >= bp_domain[1], "map does not cover requested domain")


def normalize_f0(record: Mapping[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    exact_keys(record, set(F0_KEY) | {"ANP1", "ANP2"}, "F0_record")
    require(type(record["root_seed"]) is int and record["root_seed"] >= 0, "invalid root_seed")
    require(isinstance(record["sample_id"], str) and bool(record["sample_id"]), "invalid sample_id")
    _variant_key({"chrom": record["chrom"], "pos": record["pos"], "ref": record["ref"], "alt": record["alt"], "locus_id": 0, "cM": 0.0})
    normalized: list[tuple[float, float, float]] = []
    for field in ("ANP1", "ANP2"):
        values = record[field]
        require(isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) == 3, f"{field} must have three probabilities")
        require(all(type(value) in (int, float) and not isinstance(value, bool) for value in values), f"{field} must be numeric and non-boolean")
        cast = tuple(float(value) for value in values)
        require(all(math.isfinite(value) and value >= 0 for value in cast), f"{field} is invalid")
        total = sum(cast)
        require(0.98 <= total <= 1.02, f"{field} raw sum outside authenticated range")
        normalized.append(tuple(value / total for value in cast))
    return normalized[0], normalized[1]


def validate_f0_join(records: Sequence[Mapping[str, Any]], expected_keys: set[tuple[Any, ...]], expected_samples: set[str]) -> None:
    require(bool(records) and bool(expected_keys) and bool(expected_samples), "F0 join domains must be nonempty")
    observed: set[tuple[Any, ...]] = set()
    samples: set[str] = set()
    for record in records:
        normalize_f0(record)
        key = tuple(record[field] for field in F0_KEY)
        require(key not in observed, "duplicate F0 exact key")
        observed.add(key)
        samples.add(record["sample_id"])
    require(observed == expected_keys, "F0 exact-key join is incomplete or has extras")
    require(samples == expected_samples, "F0 sample set differs")


def context_intervals(rare_cm: Sequence[float], marker_cm: Sequence[float], radius_cm: float) -> list[tuple[int, int]]:
    require(radius_cm in EXPECTED_RADII, "radius is not frozen")
    require(bool(marker_cm), "marker cM axis is empty")
    require(all(type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value) for value in list(rare_cm) + list(marker_cm)), "cM must be finite")
    require(all(rare_cm[index] <= rare_cm[index + 1] for index in range(len(rare_cm) - 1)), "rare cM is unsorted")
    require(all(marker_cm[index] <= marker_cm[index + 1] for index in range(len(marker_cm) - 1)), "marker cM is unsorted")
    return [(bisect.bisect_left(rare_cm, marker - radius_cm), bisect.bisect_right(rare_cm, marker + radius_cm)) for marker in marker_cm]


def pack_contiguous(context_lengths: Sequence[int], person_batch: int = 8, token_budget: int = 262144) -> list[tuple[int, int, int]]:
    require(type(person_batch) is int and type(token_budget) is int and person_batch > 0 and token_budget > 0, "packed limits must be positive non-boolean integers")
    chunks: list[tuple[int, int, int]] = []
    start = tokens = 0
    for index, raw_length in enumerate(context_lengths):
        require(type(raw_length) is int and raw_length >= 0, "context length must be a nonnegative non-boolean integer")
        marker_tokens = raw_length * person_batch
        require(marker_tokens <= token_budget, "single context exceeds token budget; truncation is forbidden")
        if tokens and tokens + marker_tokens > token_budget:
            chunks.append((start, index, tokens))
            start, tokens = index, 0
        tokens += marker_tokens
    if context_lengths:
        chunks.append((start, len(context_lengths), tokens))
    require(sum(end - begin for begin, end, _ in chunks) == len(context_lengths), "packed chunks lost markers")
    return chunks


def pack_contexts(contexts: Sequence[Sequence[Sequence[float]]]) -> tuple[list[list[float]], list[int]]:
    flat: list[list[float]] = []
    row_ptr = [0]
    for context in contexts:
        for token in context:
            require(len(token) == 13, "token width must be exactly 13")
            values = [float(value) for value in token]
            require(all(math.isfinite(value) for value in values), "packed token is non-finite")
            flat.append(values)
        row_ptr.append(len(flat))
    return flat, row_ptr


def padded_contexts(contexts: Sequence[Sequence[Sequence[float]]], poison: float) -> tuple[list[list[list[float]]], list[list[int]]]:
    width = 13
    maximum = max((len(context) for context in contexts), default=0)
    padded: list[list[list[float]]] = []
    masks: list[list[int]] = []
    for context in contexts:
        require(all(len(token) == width for token in context), "token width must be exactly 13")
        padded.append([[float(value) for value in token] for token in context] + [[poison] * width for _ in range(maximum - len(context))])
        masks.append([1] * len(context) + [0] * (maximum - len(context)))
    return padded, masks


def unpack_masked_padded(padded: Sequence[Sequence[Sequence[float]]], masks: Sequence[Sequence[int]]) -> list[list[list[float]]]:
    require(len(padded) == len(masks), "padded/mask row count differs")
    contexts: list[list[list[float]]] = []
    for row, mask in zip(padded, masks):
        require(len(row) == len(mask), "padded/mask width differs")
        require(all(type(value) is int and value in (0, 1) for value in mask), "mask must be integer 0/1")
        seen_zero = False
        valid: list[list[float]] = []
        for token, flag in zip(row, mask):
            require(len(token) == 13, "padded token width must be exactly 13")
            seen_zero = seen_zero or flag == 0
            require(not (seen_zero and flag == 1), "mask must be right padded")
            if flag:
                values = [float(value) for value in token]
                require(all(math.isfinite(value) for value in values), "valid padded token is non-finite")
                valid.append(values)
        contexts.append(valid)
    return contexts


def validate_ref_node_firewall(tree_nodes: Sequence[int], ref_pair_nodes: Sequence[int],
                               contributing_nodes: Sequence[int]) -> None:
    for label, values in (("tree", tree_nodes), ("REF pairs", ref_pair_nodes), ("contributors", contributing_nodes)):
        require(bool(values), f"{label} node set is empty")
        require(all(type(value) is int and value >= 0 for value in values), f"{label} node set is invalid")
        require(len(set(values)) == len(values), f"{label} node set contains duplicates")
    tree_set, ref_set, contributor_set = set(tree_nodes), set(ref_pair_nodes), set(contributing_nodes)
    require(ref_set <= tree_set, "REF nodes are absent from the tree")
    require(contributor_set == ref_set, "non-REF or missing REF nodes contributed to SAFE_BRIDGE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    validate_contract(load_json(args.contract))
    print(json.dumps({"stage": "M33_M0_CONTRACT_VALIDATION", "status": "PASS_CONTRACT_ONLY"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
