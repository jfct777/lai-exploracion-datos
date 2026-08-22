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
EXPECTED_CONTRACT_SEMANTIC_SHA256 = "bad4ba80adaae711e7ca75eb88ce69f2b42e45f10637cb648d9fe0ffeeadbb0b"
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
         "sample_key_contract", "packed_loader", "source_auth_policy", "persistence_contract",
         "execution_authorization", "known_contract_tensions"},
    ("anchors",): {"a0_root17", "a0_root18", "known_answer_root17", "known_answer_root18"},
    ("anchors", "a0_root17"): set(EXPECTED_A0),
    ("anchors", "a0_root18"): set(EXPECTED_A0_ROOT18),
    ("anchors", "known_answer_root17"): {"profile", "counts", "sha256"},
    ("anchors", "known_answer_root17", "counts"): set(EXPECTED_KAT_COUNTS),
    ("anchors", "known_answer_root17", "sha256"): set(EXPECTED_KAT_HASHES),
    ("anchors", "known_answer_root18"): {"profile", "counts", "sha256"},
    ("anchors", "known_answer_root18", "counts"): set(EXPECTED_KAT18_COUNTS),
    ("anchors", "known_answer_root18", "sha256"): set(EXPECTED_KAT18_HASHES),
    ("root_registry",): {"consumed_technical_roots", "root18_status", "consumed_roots_quarantine", "scientific_selection", "radius_selection"},
    ("root_registry", "consumed_technical_roots"): {"root17", "root18"},
    ("process_contracts",): {"I0_DERIVE_AUTHENTICATE_FLARE_INDEX", "SAFE_BRIDGE", "MATERIALIZE"},
    ("process_contracts", "I0_DERIVE_AUTHENTICATE_FLARE_INDEX"): {"implemented", "status", "input_logical_ids", "output_logical_ids", "tool", "exact_command_argv", "query_parity", "write_policy", "requirements", "receipt_required_keys"},
    ("process_contracts", "I0_DERIVE_AUTHENTICATE_FLARE_INDEX", "tool"): {"name", "exact_version", "local_image_id_technical_anchor", "required_pullable_oci_image"},
    ("process_contracts", "SAFE_BRIDGE"): {"implemented", "isolation", "truth_mounted", "physical_inputs", "derivations", "role_firewall", "output_artifacts"},
    ("process_contracts", "SAFE_BRIDGE", "physical_inputs"): {"tree_sequence", "pools", "rare_catalog", "rare_haplotypes", "selected_sites", "target_calls", "common_reference_crosscheck", "ref_pairs", "panel_map", "genetic_map"},
    ("process_contracts", "SAFE_BRIDGE", "physical_inputs", "*"): {"logical_id", "format", "access", "authentication"},
    ("process_contracts", "SAFE_BRIDGE", "derivations"): {"selected_loci_incremental", "target_rare_diploid_incremental", "reference_rare_summary_incremental", "common_reference_crosscheck_scope"},
    ("process_contracts", "SAFE_BRIDGE", "role_firewall"): {"tree_and_pools_allowed_nodes", "required_node_set_equality", "forbidden_genotype_contributors", "freq_usage", "violation"},
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
    ("process_contracts", "MATERIALIZE"): {"implemented", "input_logical_ids", "output_logical_ids", "bundle_partitioning", "channel_cast", "output_artifacts", "forbidden_namespace_tokens_case_insensitive"},
    ("process_contracts", "MATERIALIZE", "bundle_partitioning"): {"unit", "row_order", "token_order_within_row", "person_batch", "maximum_valid_tokens_per_shard"},
    ("process_contracts", "MATERIALIZE", "output_artifacts"): {"packed_rare_context_shard", "bundle_manifest", "materialization_receipt", "READY"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "packed_rare_context_shard"): {"schema_id", "format", "privacy", "arrays", "invariants"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "packed_rare_context_shard", "arrays"): {"sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref", "marker_alt", "marker_cM", "radius_cM", "rare_tokens", "rare_mask", "rare_locus_index", "row_ptr", "row_sample_index", "row_marker_index", "F0"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "packed_rare_context_shard", "arrays", "*"): {"axes", "shape", "dtype"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "bundle_manifest"): {"schema_id", "format", "privacy", "required_keys", "forbid_extra_keys", "ordered_shard_entry_schema", "ordering_and_coverage"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "bundle_manifest", "ordered_shard_entry_schema"): {"exact_keys", "field_contracts"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "bundle_manifest", "ordered_shard_entry_schema", "field_contracts"): {"schema_id", "shard_ordinal", "person_start", "person_end_exclusive", "marker_start", "marker_end_exclusive", "valid_token_count", "gcs_uri", "gcs_generation", "raw_sha256", "semantic_sha256"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "materialization_receipt"): {"schema_id", "format", "privacy", "required_keys", "forbid_extra_keys", "radius_manifest_map"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "materialization_receipt", "radius_manifest_map"): {"exact_order", "value"},
    ("process_contracts", "MATERIALIZE", "output_artifacts", "READY"): {"schema_id", "format", "privacy", "required_keys", "forbid_extra_keys", "write_order"},
    ("incremental_partition",): {"model_input", "exact_key", "required_identity", "selected_all_and_flare_overlap", "overlap_loci_must_not_enter_model", "duplicate_key_or_locus_id"},
    ("reference_semantics",): {"raw_bridge_fields", "minor_af", "observed_mask", "no_support", "no_support_is_explicit_bridge_audit_field_not_an_extra_tensor_channel", "dosage_mean_as_af", "callable_normalization"},
    ("reference_semantics", "callable_normalization"): {"technical_denominator_by_root", "technical_status", "future_scientific_denominator", "future_manifest_required_before_forward", "raw_callable_an_is_persisted_unchanged"},
    ("reference_semantics", "callable_normalization", "technical_denominator_by_root"): {"root17", "root18"},
    ("sample_key_contract",): {"algorithm", "domain_separator", "input_encoding", "output_encoding", "formula", "f0_join", "duplicate_or_collision", "privacy"},
    ("f0_contract",): {"join_key", "probability_fields", "ancestry_order", "raw_sum_range", "operation", "haplotype_axis_preserved", "forbidden_dependencies"},
    ("primary_transferable_input",): {"storage", "conceptual_shapes", "phase_policy", "channel_count", "channel_dtype", "calculation_dtype_before_cast", "first_ordered_locus_delta_cM", "channels", "rare_order", "missing_policy"},
    ("primary_transferable_input", "conceptual_shapes"): {"rare_tokens", "rare_mask", "F0"},
    ("packed_loader",): {"mode", "person_batch", "token_budget", "token_definition", "radii_cm", "radius_selection", "interval", "global_padding", "truncation", "empty_context", "single_context_over_budget", "warning_memory_fraction", "stop_memory_fraction"},
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
    require(bridge["derivations"]["reference_rare_summary_incremental"].startswith("tree_sequence_plus_pools_plus_ref_pairs"), "REF rare source mapping is wrong")
    require("never_source" in bridge["derivations"]["common_reference_crosscheck_scope"], "common REF VCF could be misused as rare source")
    require(all(item["access"] == "read_only" for item in bridge["physical_inputs"].values()), "raw bridge input is writable")
    require(all(item["authentication"] == "A0_receipt_input_sha256" for item in bridge["physical_inputs"].values()), "SAFE_BRIDGE input authentication changed")
    require(all(item["contains_raw_input_payload"] is False for item in bridge["output_artifacts"].values()), "raw payload escapes SAFE_BRIDGE")
    require(all(item["privacy"] == "private" for item in bridge["output_artifacts"].values()), "SAFE_BRIDGE privacy changed")
    require(all(item["byte_order"] in {"little_endian_for_multibyte_numeric_fields", "not_applicable"} for item in bridge["output_artifacts"].values()), "output byte order is ambiguous")
    require(bridge["role_firewall"]["forbidden_genotype_contributors"] == ["DONOR", "FREQ", "TARGET", "VALID", "TEST"] and
            bridge["role_firewall"]["violation"] == "STOP", "SAFE_BRIDGE role firewall changed")
    bridge_receipt = bridge["output_artifacts"]["safe_bridge_receipt"]
    require(bridge_receipt["schema_id"] == "m33_m0_safe_bridge_receipt_v1" and
            bridge_receipt["required_keys"] == [
                "stage", "status", "root_label", "root_seed", "expected_ref_node_count",
                "contributing_ref_node_count", "rejected_non_ref_node_count",
                "expected_ref_nodes_semantic_sha256", "contributing_ref_nodes_semantic_sha256",
                "role_firewall_pass", "artifact_semantic_sha256", "reopen_verified", "append_only",
            ], "SAFE_BRIDGE firewall receipt changed")

    materialize = processes["MATERIALIZE"]
    namespace = " ".join(_flatten_strings({"inputs": materialize["input_logical_ids"], "outputs": materialize["output_logical_ids"]})).lower()
    for token in materialize["forbidden_namespace_tokens_case_insensitive"]:
        require(token.lower() not in namespace, f"forbidden MATERIALIZE namespace token: {token}")
    require(materialize["input_logical_ids"] == [
        "selected_loci_incremental", "target_rare_diploid_incremental",
        "reference_rare_summary_incremental", "a0_authenticated_flare_anc",
        "derived_flare_anc_tbi", "authenticated_genetic_map",
    ], "MATERIALIZE inputs changed")
    require(materialize["output_logical_ids"] == [
        "packed_rare_context", "bundle_manifest_by_radius", "materialization_receipt", "READY",
    ], "MATERIALIZE outputs changed")
    packed_schema = materialize["output_artifacts"]["packed_rare_context_shard"]
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
            bundle_manifest["ordered_shard_entry_schema"]["exact_keys"] == [
                "schema_id", "shard_ordinal", "person_start", "person_end_exclusive",
                "marker_start", "marker_end_exclusive", "valid_token_count", "gcs_uri",
                "gcs_generation", "raw_sha256", "semantic_sha256",
            ], "bundle manifest shard schema changed")
    material_receipt = materialize["output_artifacts"]["materialization_receipt"]
    require("READY_sha256" not in material_receipt["required_keys"] and
            "ordered_bundle_manifest_sha256_by_radius" in material_receipt["required_keys"] and
            material_receipt["radius_manifest_map"]["exact_order"] == ["0.05", "0.1", "0.2", "0.5"],
            "materialization receipt radius manifests changed")
    ready = materialize["output_artifacts"]["READY"]
    require(ready["schema_id"] == "m33_m0_READY_v1" and ready["forbid_extra_keys"] is True and
            "materialization_receipt_sha256" in ready["required_keys"] and
            ready["write_order"].startswith("last_after_all_shards"), "READY schema/order changed")

    partition = contract["incremental_partition"]
    require(partition["model_input"] == "selected_loci_incremental_only" and partition["overlap_loci_must_not_enter_model"] is True, "model input is not incremental-only")
    require(partition["exact_key"] == ["chrom", "pos", "ref", "alt"], "variant identity changed")
    require(partition["required_identity"] == "selected_all_equals_disjoint_union_incremental_and_flare_overlap", "partition union invariant missing")
    reference = contract["reference_semantics"]
    require(reference["minor_af"] == "minor_ac_div_callable_an_if_callable_an_gt_0_else_0" and reference["dosage_mean_as_af"] is False, "wrong AF semantics")
    require(reference["raw_bridge_fields"] == ["minor_ac", "callable_an", "minor_af", "observed_mask", "no_support"], "raw AN or no_support missing")
    normalization = reference["callable_normalization"]
    require(normalization["technical_denominator_by_root"] == {"root17": 60, "root18": 60} and
            normalization["technical_status"] == "TECHNICAL_ONLY_EQUIVALENT_NEVER_REUSED_FOR_SCIENTIFIC_NORMALIZATION",
            "technical AN normalizer changed")
    require(normalization["future_manifest_required_before_forward"] is True and normalization["raw_callable_an_is_persisted_unchanged"] is True, "future FIT/raw AN separation missing")

    f0 = contract["f0_contract"]
    require(f0["join_key"] == list(F0_KEY) and f0["probability_fields"] == ["ANP1", "ANP2"], "F0 source changed")
    require(f0["ancestry_order"] == ["AFR", "EUR", "ASIA"] and f0["raw_sum_range"] == [0.98, 1.02] and
            f0["operation"] == "renormalize_each_ANP_vector_to_exact_simplex_for_probability_operations",
            "F0 probability semantics changed")
    require(f0["haplotype_axis_preserved"] is True, "F0 haplotype axis lost")
    require(f0["forbidden_dependencies"] == ["GT", "AN1", "AN2", "truth", "hard_call", "target_rare_phase"], "F0 forbidden dependencies changed")
    sample_key = contract["sample_key_contract"]
    require(sample_key["algorithm"] == "sha256" and sample_key["domain_separator"] == "DNABR_M33_M0_SAMPLE_V1|" and
            "bijection" in sample_key["f0_join"] and sample_key["duplicate_or_collision"] == "STOP",
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


def reference_summary(minor_ac: int, callable_an: int) -> dict[str, float | int]:
    require(type(minor_ac) is int and type(callable_an) is int, "AC and AN must be non-boolean integers")
    require(callable_an >= 0 and 0 <= minor_ac <= callable_an, "require 0 <= AC <= AN")
    observed = int(callable_an > 0)
    return {"minor_af": minor_ac / callable_an if callable_an else 0.0,
            "observed_mask": observed, "no_support": int(observed == 1 and minor_ac == 0)}


def validate_target_cell(minor_dosage: int, observed_mask: int) -> tuple[int, int]:
    require(type(observed_mask) is int and observed_mask in (0, 1), "observed mask must be integer 0/1, not bool")
    require(type(minor_dosage) is int, "dosage must be a non-boolean integer")
    require((observed_mask == 0 and minor_dosage == 0) or (observed_mask == 1 and minor_dosage in (0, 1, 2)),
            "missing dosage must be zero; observed dosage must be 0/1/2")
    return minor_dosage, observed_mask


def diploid_minor_dosage(haplotype0: int | None, haplotype1: int | None) -> tuple[int, int]:
    if haplotype0 is None or haplotype1 is None:
        return 0, 0
    require(type(haplotype0) is int and type(haplotype1) is int and haplotype0 in (0, 1) and haplotype1 in (0, 1), "haplotype states must be non-boolean binary integers")
    return validate_target_cell(haplotype0 + haplotype1, 1)


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
    row_keys = [_variant_key(row) for row in rows]
    require(len(set(row_keys)) == len(row_keys), "duplicate selected key")
    locus_ids = [row["locus_id"] for row in rows]
    require(len(set(locus_ids)) == len(locus_ids), "duplicate selected locus_id")
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
