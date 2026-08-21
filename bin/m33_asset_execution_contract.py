#!/usr/bin/env python3
"""Validate the prospective M33 execution contract without reading real assets.

The executable CLI in this phase authenticates contracts and source only.  The
semantic functions accept explicitly injected bytes/records for unit fixtures;
no GCS client, generator, predictor or trainer is reachable from this module.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from m33_asset_manifest_contract import (
    ASSET_FIELDS,
    BASE_CONTRACT_SHA256,
    EXACT_CONTRACT_SHA256,
    derive_rng_streams,
    exact_keys,
    load_contract,
    require,
    sha256_file,
    strict_json,
    valid_sha,
    validate_gcs_uri,
    validate_roles,
    write_exclusive,
)


AMENDMENT_SHA256 = "65b4b3f0648cd813edbcf2aa053c549cccc72de243082718049788cec39e5a60"
STATUS = "CONTRACT_ONLY_NO_REAL_ASSET_READ_NO_GENERATION_NO_FORWARD_NO_TRAINING"
PASS_STATUS = "PASS_EXECUTION_CONTRACT_FIXTURES_ONLY_NO_REAL_ASSET_READ"
EXPECTED_NEXTFLOW_VERSION = "26.04.6"
DEVELOPMENT_ROOTS = (386357765, 2024931463, 1324432253)
ENGINE_COMMIT = "1b88ab16bfa6a19807cf74689cd5476e05d3d2b4"
REQUIRED_SOURCES = {
    "bin/m33_asset_execution_contract.py",
    "bin/m33_asset_execution_source_auth.py",
    "bin/m33_asset_manifest_contract.py",
    "conf/m33_asset_execution_amendment.json",
    "conf/m33_asset_execution_contract.config",
    "conf/m33_asset_manifest_contract.json",
    "modules/33_ASSET_EXECUTION_CONTRACT.nf",
    "workflows/m33_asset_execution_contract.nf",
    "tests/test_m33_asset_execution_contract.py",
    "tests/test_m33_asset_execution_nextflow.py",
}
PLAN_KEYS = {
    "schema_version", "stage", "mode", "root_seed", "plan_id", "asset_set_id",
    "output_prefix", "base_contract_sha256", "asset_contract_sha256",
    "amendment_sha256", "generator_source_auth_sha256", "input_descriptors",
    "rng_seedsequence", "engine", "orchestrator", "creation_precondition",
}
MANIFEST_KEYS = {
    "schema_version", "stage", "mode", "root_seed", "plan_id", "asset_set_id",
    "output_prefix", "plan_manifest_sha256", "assets", "semantic_fingerprints",
    "predict_bundle", "flare_input_bundle", "rare_enabled_model_bundle",
    "private_truth_bundle", "flare_receipt", "final_manifest_sha256",
}
SEMANTIC_KEYS = {
    "normalized_full_tree_sha256", "normalized_genealogy_sha256",
    "root_independent_source_haplotype_sha256",
}
FLARE_RECEIPT_KEYS = {
    "stage", "status", "plan_id", "input_logical_ids", "input_descriptor_sha256",
    "output_descriptor_sha256",
    "flare_version", "flare_reported_build", "flare_jar_sha256", "container_digest",
    "parameters", "ancestry_order", "simulation_engine_commit", "orchestrator_commit",
    "generator_source_auth_sha256", "run_manifest_sha256", "interface_sha256",
    "prediction_sha256", "audit_payload_sha256", "truth_mounted",
    "truth_argument_available", "sealed_before_truth_mount", "sealed_at_utc",
}
READY_KEYS = {
    "schema_version", "stage", "status", "plan_id", "final_manifest_descriptor",
    "plan_manifest_descriptor",
    "final_manifest_reopened_and_verified", "all_prior_descriptors_reopened_and_verified",
    "created_with_if_generation_match", "publication_log",
}
GENERATOR_SOURCE_AUTH_KEYS = {"stage", "status", "git_commit", "source_sha256"}
FLARE_RUN_MANIFEST_KEYS = {
    "schema_version", "stage", "status", "plan_id", "interface_sha256",
    "generator_source_auth_sha256", "simulation_engine_commit", "orchestrator_commit",
    "flare_version", "flare_reported_build", "flare_jar_sha256", "container_digest",
    "command_argv", "input_descriptor_sha256", "output_descriptor_sha256",
    "truth_accessed", "started_at_utc", "finished_at_utc",
}
FLARE_AUDIT_KEYS = {
    "schema_version", "stage", "status", "plan_id", "interface_sha256",
    "run_manifest_sha256", "prediction_sha256", "target_haplotype_count", "locus_count",
    "sample_parity_exact", "locus_parity_exact", "probabilities_finite",
    "probability_simplex_exact_within_tolerance", "simplex_tolerance", "truth_accessed",
    "started_at_utc", "finished_at_utc",
}
SELECTED_SITES_DOCUMENT_KEYS = {"schema_version", "stage", "status", "rows"}
TARGET_RARE_DOCUMENT_KEYS = {
    "schema_version", "stage", "status", "target_haplotype_ids", "rows",
}
TARGET_RARE_ROW_KEYS = {
    "target_haplotype_id", "CHROM", "POS", "REF", "ALT", "minor_allele_presence",
}
ROLES_DOCUMENT_KEYS = {"schema_version", "stage", "status", "roles"}
MOSAIC_DOCUMENT_KEYS = {"schema_version", "stage", "status", "chromosome", "rows"}
TRUTH_DOCUMENT_KEYS = {"schema_version", "stage", "status", "chromosome", "rows"}
PROVENANCE_DOCUMENT_KEYS = {
    "schema_version", "stage", "status", "chromosome", "donor_ancestry",
    "donor_alleles", "target_alleles",
}
ALLELE_ROW_KEYS = {"haplotype_id", "CHROM", "POS", "REF", "ALT", "allele"}
TARGET_VCF_FIXTURE_KEYS = {
    "schema_version", "stage", "status", "chromosome", "target_haplotype_ids", "loci",
}
FLARE_ANC_DOCUMENT_KEYS = {
    "schema_version", "stage", "status", "chromosome", "target_haplotype_ids", "loci", "rows",
}
FLARE_GLOBAL_DOCUMENT_KEYS = {
    "schema_version", "stage", "status", "ancestry_order", "rows",
}
FLARE_GLOBAL_ROW_KEYS = {"target_haplotype_id", "probabilities"}
SELECTED_SITE_KEYS = {
    "CHROM", "POS", "REF", "ALT", "minor_allele_index", "minor_mac",
    "minor_an", "minor_maf", "carrier_people",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def load_amendment(path: str | Path) -> dict[str, Any]:
    require(sha256_file(path) == AMENDMENT_SHA256, "immutable execution amendment byte hash drift")
    amendment = strict_json(path)
    validate_amendment(amendment)
    return amendment


def validate_amendment(amendment: dict[str, Any]) -> None:
    require(amendment["schema_version"] == "1.0.0", "amendment schema drift")
    require(amendment["stage"] == "M33_ASSET_EXECUTION_AMENDMENT", "amendment stage drift")
    require(amendment["status"] == STATUS, "amendment authorization drift")
    immutable = amendment["immutable_inputs"]
    require(immutable["pre4_contract_sha256"] == BASE_CONTRACT_SHA256, "PRE-4 hash drift")
    require(immutable["asset_manifest_contract_sha256"] == EXACT_CONTRACT_SHA256,
            "asset contract hash drift")
    roots = amendment["root_registry"]
    require(tuple(roots["allowed_new_m33_roots"]) == DEVELOPMENT_ROOTS,
            "new M33 DEVELOPMENT root registry drift")
    forbidden = set(roots["technical_roots_forbidden"]) | set(roots["eval_roots_sealed_and_forbidden"])
    require(set(DEVELOPMENT_ROOTS).isdisjoint(forbidden), "allowed and forbidden roots overlap")
    require(roots["any_other_root_forbidden"] is True, "unregistered roots are not fail-closed")
    require(amendment["code_anchors"]["simulation_engine_commit"] == ENGINE_COMMIT,
            "simulation engine commit drift")
    require(amendment["code_anchors"]["orchestrator_commit_policy"] ==
            "future_exact_clean_commit_authenticated_by_generator_source_auth",
            "orchestrator source policy drift")
    auth = amendment["execution_authorization"]
    require(auth == {
        "contract_tests": True,
        "fixture_semantic_validation": True,
        "real_asset_read": False,
        "asset_generation": False,
        "forward": False,
        "training": False,
        "EVAL_open": False,
    }, "execution was authorized beyond fixtures")
    manifest_members = amendment["manifest_members"]
    bundled = {item for values in amendment["bundles"].values() for item in values}
    require(bundled == set(manifest_members) | set(amendment["publication_envelopes"]),
            "typed asset inventory differs from bundle inventory")
    require(set(amendment["scientific_asset_types"]).issubset(set(manifest_members)),
            "scientific asset inventory is not contained in manifest members")
    predict = set(amendment["bundles"]["predict_bundle"])
    forbidden_predict = set(amendment["truth_barrier"]["predict_bundle_forbidden"])
    require(predict.isdisjoint(forbidden_predict), "predict bundle contains a forbidden truth input")
    flare = amendment["flare_contract"]
    require(amendment["bundles"]["flare_input_bundle"] == flare["input_logical_ids"],
            "FLARE input bundle drift")
    require(flare["input_logical_ids"] ==
            ["ref_vcf", "ref_tbi", "target_vcf", "target_tbi", "panel_map", "genetic_map"],
            "FLARE input inventory is not exact")
    require(flare["output_logical_ids"] ==
            ["flare_anc", "flare_anc_tbi", "flare_global", "flare_model", "flare_log",
             "flare_audit"], "FLARE output inventory is not exact")
    require(flare["direct_command_output_logical_ids"] ==
            ["flare_anc", "flare_anc_tbi", "flare_global", "flare_model", "flare_log"],
            "FLARE direct-command output inventory is not exact")
    require(flare["version"] == "0.6.0" and
            flare["jar_sha256"] ==
            "8c804341b555f302591b12cd72e870b1ca7849055d1dcd2b5cfa09b725bd9420",
            "FLARE binary anchor drift")
    require(flare["receipt_is_sealed_before_truth_mount"] is True and
            flare["receipt_bytes_must_be_independently_reopened_against_flare_receipt_descriptor"]
            is True, "FLARE receipt barrier drift")
    require(isinstance(flare["command_argv"], list) and flare["command_argv"] and
            all(isinstance(item, str) and item for item in flare["command_argv"]),
            "FLARE command argv is not frozen")
    require(not any("truth" in item.lower() for item in flare["command_argv"]),
            "FLARE command argv exposes truth")
    rare_bundle = amendment["bundles"]["rare_enabled_model_bundle"]
    validate_rare_enabled_inputs(rare_bundle)
    require("target_rare_incremental" in manifest_members and "target_rare" not in manifest_members,
            "incremental rare target asset was not renamed fail-closed")
    require(amendment["publication"]["terminal_write_order"] == ["final_manifest", "READY"],
            "READY is not last")
    require(amendment["publication"]["object_precondition"] == "ifGenerationMatch=0",
            "append-only publication drift")


def validate_input_descriptor(descriptor: dict[str, Any], logical_id: str) -> None:
    exact_keys(descriptor, ASSET_FIELDS, f"input descriptor {logical_id}")
    require(descriptor["logical_id"] == logical_id, f"input descriptor ID drift: {logical_id}")
    if logical_id == "genetic_map":
        require(descriptor["gcs_uri"] ==
                "gs://projects-usp/dna-do-brasil/dnabr-lai-gnomix/maps/genetic.map.chr22",
                "genetic_map is not the exact legacy read-only exception")
    else:
        validate_gcs_uri(descriptor["gcs_uri"], logical_id)
    require(isinstance(descriptor["gcs_generation"], str) and
            re.fullmatch(r"[1-9][0-9]*", descriptor["gcs_generation"]) is not None,
            f"immutable generation missing: {logical_id}")
    valid_sha(descriptor["sha256_raw"], f"descriptor {logical_id}")
    require(type(descriptor["size_bytes"]) is int and descriptor["size_bytes"] > 0,
            f"invalid size: {logical_id}")
    require(isinstance(descriptor["crc32c"], str) and
            re.fullmatch(r"[A-Za-z0-9+/]{6}==", descriptor["crc32c"]) is not None,
            f"invalid CRC32C: {logical_id}")
    require(isinstance(descriptor["media_type"], str) and descriptor["media_type"],
            f"missing media type: {logical_id}")
    require(descriptor["compression"] in {"none", "gzip", "bgzip"},
            f"unsupported compression: {logical_id}")
    require(isinstance(descriptor["schema_version"], str) and descriptor["schema_version"],
            f"missing schema: {logical_id}")
    require(type(descriptor["record_count"]) is int and descriptor["record_count"] >= 0,
            f"invalid record count: {logical_id}")


def derive_plan_id(root_seed: int, generator_source_auth_sha256: str,
                   input_descriptors: Mapping[str, dict[str, Any]],
                   rng_seedsequence: Mapping[str, Any], amendment_sha256: str) -> str:
    require(root_seed in DEVELOPMENT_ROOTS, "unregistered or forbidden M33 root")
    valid_sha(generator_source_auth_sha256, "generator source auth")
    valid_sha(amendment_sha256, "execution amendment")
    forbidden_terms = {
        "output_uri", "output_generation", "output_size", "output_sha256",
        "generator_manifest_sha256", "final_manifest_sha256", "ready_sha256",
    }
    serialized_inputs = canonical_json(input_descriptors).decode().lower()
    require(not any(term in serialized_inputs for term in forbidden_terms),
            "plan identity depends on an output descriptor")
    fields = [
        BASE_CONTRACT_SHA256,
        EXACT_CONTRACT_SHA256,
        amendment_sha256,
        "DEVELOPMENT",
        str(root_seed),
        generator_source_auth_sha256,
        canonical_json(input_descriptors).decode(),
        canonical_json(rng_seedsequence).decode(),
    ]
    return hashlib.sha256("|".join(fields).encode()).hexdigest()


def expected_prefix(amendment: dict[str, Any], root_seed: int, plan_id: str) -> str:
    return amendment["identity"]["output_prefix_template"].format(
        pre4_contract_sha256=BASE_CONTRACT_SHA256,
        root_seed=root_seed,
        plan_id=plan_id,
    )


def validate_plan(plan: dict[str, Any], amendment: dict[str, Any],
                  asset_contract: dict[str, Any]) -> None:
    exact_keys(plan, PLAN_KEYS, "plan manifest")
    require(plan["schema_version"] == "1.0.0" and plan["stage"] == "M33_PLAN_MANIFEST",
            "plan schema or stage drift")
    require(plan["mode"] == "DEVELOPMENT", "only DEVELOPMENT plans are accepted")
    root = plan["root_seed"]
    require(root in DEVELOPMENT_ROOTS, "unregistered or forbidden M33 root")
    require(plan["base_contract_sha256"] == BASE_CONTRACT_SHA256, "PRE-4 hash drift in plan")
    require(plan["asset_contract_sha256"] == EXACT_CONTRACT_SHA256, "asset contract hash drift in plan")
    require(plan["amendment_sha256"] == AMENDMENT_SHA256, "amendment hash drift in plan")
    required_inputs = set(amendment["plan_inputs"]["required_descriptors"])
    require(set(plan["input_descriptors"]) == required_inputs, "plan input inventory drift")
    for logical_id, descriptor in plan["input_descriptors"].items():
        validate_input_descriptor(descriptor, logical_id)
    map_expected = asset_contract["shared_assets"]
    observed_map = plan["input_descriptors"]["genetic_map"]
    require(observed_map == {
        "logical_id": "genetic_map",
        "gcs_uri": map_expected["genetic_map_uri"],
        "gcs_generation": map_expected["genetic_map_gcs_generation"],
        "size_bytes": map_expected["genetic_map_size_bytes"],
        "sha256_raw": map_expected["genetic_map_sha256"],
        "crc32c": map_expected["genetic_map_crc32c"],
        "media_type": map_expected["genetic_map_media_type"],
        "compression": map_expected["genetic_map_compression"],
        "schema_version": map_expected["genetic_map_schema_version"],
        "record_count": map_expected["genetic_map_record_count"],
    }, "legacy map descriptor drift")
    source_auth = plan["generator_source_auth_sha256"]
    source_auth_descriptor = plan["input_descriptors"]["generator_source_auth"]
    require(source_auth_descriptor["sha256_raw"] == source_auth,
            "source-auth descriptor is not bound to plan identity")
    require(source_auth_descriptor["schema_version"] == "m33_generator_source_auth_v1",
            "generator source-auth schema drift")
    require(source_auth_descriptor["gcs_uri"].startswith(
        asset_contract["shared_assets"]["generator_source_auth_prefix"]
    ), "generator source-auth is outside the exact shared prefix")
    require(plan["rng_seedsequence"] == asset_contract["rng_contract"]["root_streams"][str(root)],
            "root RNG SeedSequence drift")
    derived = derive_plan_id(root, source_auth, plan["input_descriptors"],
                             plan["rng_seedsequence"], AMENDMENT_SHA256)
    require(plan["plan_id"] == derived and plan["asset_set_id"] == derived,
            "plan_id or asset_set_id derivation drift")
    require(plan["output_prefix"] == expected_prefix(amendment, root, derived),
            "output prefix is not determined by plan_id")
    validate_gcs_uri(plan["output_prefix"], "plan output prefix")
    require(plan["creation_precondition"] == "ifGenerationMatch=0", "plan permits overwrite")
    exact_keys(plan["engine"], {"repository", "git_commit", "oci_image_digest"}, "engine anchor")
    require(plan["engine"]["repository"] == amendment["code_anchors"]["simulation_engine_repository"],
            "simulation engine repository drift")
    require(plan["engine"]["git_commit"] == ENGINE_COMMIT, "simulation engine commit drift")
    require(plan["engine"]["oci_image_digest"] == amendment["code_anchors"]["oci_image_digest"],
            "engine image drift")
    exact_keys(plan["orchestrator"], {"repository", "git_commit", "source_auth_sha256"},
               "orchestrator anchor")
    require(plan["orchestrator"]["repository"] ==
            amendment["code_anchors"]["orchestrator_repository"],
            "orchestrator repository drift")
    require(re.fullmatch(r"[0-9a-f]{40}", plan["orchestrator"]["git_commit"]) is not None,
            "orchestrator commit must be an exact future commit")
    require(plan["orchestrator"]["source_auth_sha256"] == source_auth,
            "orchestrator is not authenticated by source auth")
    require(plan["orchestrator"]["git_commit"] != ENGINE_COMMIT,
            "engine and orchestrator commits were conflated")


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def crc32c_base64(data: bytes) -> str:
    return base64.b64encode(_crc32c(data).to_bytes(4, "big")).decode()


def verify_observed_bytes(descriptor: dict[str, Any], payload: bytes,
                          observed_generation: str, observed_crc32c: str) -> None:
    """Verify independently opened bytes; manifest hashes alone are never evidence."""
    validate_input_descriptor(descriptor, descriptor["logical_id"])
    require(observed_generation == descriptor["gcs_generation"], "opened generation differs")
    require(hashlib.sha256(payload).hexdigest() == descriptor["sha256_raw"],
            "opened bytes differ from SHA-256 descriptor")
    require(len(payload) == descriptor["size_bytes"], "opened byte size differs")
    computed_crc = crc32c_base64(payload)
    require(observed_crc32c == computed_crc, "storage CRC32C was not independently reproduced")
    require(descriptor["crc32c"] == computed_crc, "descriptor CRC32C differs from opened bytes")


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Parse authenticated UTF-8 JSON while rejecting duplicate object keys."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}") from exc
    require(isinstance(parsed, dict), f"{label} must be a JSON object")
    return parsed


def validate_generator_source_auth_observation(
    descriptor: dict[str, Any], observation: Mapping[str, Any],
    expected_orchestrator_commit: str, asset_contract: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the shared source inventory before validating any root plan ID."""
    exact_keys(dict(observation), {"payload", "generation", "crc32c"},
               "generator source-auth observation")
    require(descriptor["logical_id"] == "generator_source_auth",
            "wrong descriptor supplied as generator source-auth")
    require(descriptor["schema_version"] == "m33_generator_source_auth_v1",
            "generator source-auth schema drift")
    require(descriptor["gcs_uri"].startswith(
        asset_contract["shared_assets"]["generator_source_auth_prefix"]
    ), "generator source-auth is outside the exact shared prefix")
    verify_observed_bytes(
        descriptor, observation["payload"], observation["generation"], observation["crc32c"]
    )
    auth = strict_json_bytes(observation["payload"], "generator source-auth")
    exact_keys(auth, GENERATOR_SOURCE_AUTH_KEYS, "generator source-auth")
    require(auth["stage"] == "M33_ASSET_EXECUTION_SOURCE_AUTH",
            "generator source-auth stage drift")
    require(auth["status"] == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
            "generator source-auth did not pass")
    require(auth["git_commit"] == expected_orchestrator_commit,
            "generator source-auth orchestrator commit drift")
    require(re.fullmatch(r"[0-9a-f]{40}", auth["git_commit"]) is not None,
            "generator source-auth commit is not exact")
    hashes = auth["source_sha256"]
    require(isinstance(hashes, dict) and set(hashes) == REQUIRED_SOURCES,
            "generator source-auth source inventory drift")
    for relative, digest in hashes.items():
        valid_sha(digest, f"generator source-auth source {relative}")
    require(hashes["conf/m33_asset_execution_amendment.json"] == AMENDMENT_SHA256,
            "generator source-auth does not authenticate this amendment")
    return auth


def variant_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    key = (row["CHROM"], row["POS"], row["REF"], row["ALT"])
    require(key[0] == "chr22" and type(key[1]) is int and key[1] > 0,
            "variant is not GRCh38 chr22")
    require(all(isinstance(allele, str) and allele for allele in key[2:]), "empty allele")
    require("," not in key[3] and key[2] != key[3], "variant is not normalized biallelic")
    return key


def recompute_freq_selected_sites(freq_variants: Iterable[Mapping[str, Any]],
                                  freq_people: set[str]) -> list[dict[str, Any]]:
    """Return all and only FREQ-only variants passing the frozen rare selector."""
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for variant in freq_variants:
        key = variant_key(variant)
        require(key not in seen, "duplicate normalized variant key")
        seen.add(key)
        genotypes = variant["genotypes"]
        require(isinstance(genotypes, Mapping) and set(genotypes) == freq_people,
                "rare selector did not receive exactly FREQ people")
        called: list[tuple[int, int]] = []
        for genotype in genotypes.values():
            if genotype is None:
                continue
            require(isinstance(genotype, (list, tuple)) and len(genotype) == 2 and
                    all(allele in (0, 1) for allele in genotype), "invalid diploid genotype")
            called.append((int(genotype[0]), int(genotype[1])))
        an = 2 * len(called)
        if an == 0:
            continue
        alt_ac = sum(sum(gt) for gt in called)
        ref_ac = an - alt_ac
        if alt_ac == ref_ac:
            continue
        minor_index = 1 if alt_ac < ref_ac else 0
        mac = min(alt_ac, ref_ac)
        maf = mac / an
        carriers = sum(any(allele == minor_index for allele in gt) for gt in called)
        if mac >= 2 and maf < 0.01 and carriers >= 2:
            selected.append({
                "CHROM": key[0], "POS": key[1], "REF": key[2], "ALT": key[3],
                "minor_allele_index": minor_index, "minor_mac": mac, "minor_an": an,
                "minor_maf": maf, "carrier_people": carriers,
            })
    return sorted(selected, key=lambda row: (row["CHROM"], row["POS"], row["REF"], row["ALT"]))


def validate_selected_sites_exhaustive(freq_variants: Iterable[Mapping[str, Any]],
                                       freq_people: set[str],
                                       selected_sites: Sequence[Mapping[str, Any]]) -> None:
    observed = []
    for row in selected_sites:
        exact_keys(dict(row), SELECTED_SITE_KEYS, "selected site")
        observed.append(dict(row))
    ordered_observed = sorted(
        observed, key=lambda row: (row["CHROM"], row["POS"], row["REF"], row["ALT"])
    )
    require(observed == ordered_observed, "selected-sites artifact is not in canonical locus order")
    expected = recompute_freq_selected_sites(freq_variants, freq_people)
    require(len(observed) == len(expected), "selected-sites set is not exhaustive")
    for left, right in zip(observed, expected):
        require({key: left[key] for key in SELECTED_SITE_KEYS - {"minor_maf"}} ==
                {key: right[key] for key in SELECTED_SITE_KEYS - {"minor_maf"}},
                "selected-site identity or count drift")
        require(math.isclose(left["minor_maf"], right["minor_maf"], rel_tol=0, abs_tol=1e-15),
                "selected-site MAF drift")


def partition_selected_sites(selected_sites_all: Sequence[Mapping[str, Any]],
                             flare_grid_keys: set[tuple[str, int, str, str]]) -> tuple[list[dict[str, Any]],
                                                                                     list[dict[str, Any]]]:
    all_rows = [dict(row) for row in selected_sites_all]
    for row in all_rows:
        exact_keys(row, SELECTED_SITE_KEYS, "selected site partition input")
    incremental = [row for row in all_rows if variant_key(row) not in flare_grid_keys]
    overlap = [row for row in all_rows if variant_key(row) in flare_grid_keys]
    return incremental, overlap


def validate_selected_site_partition(selected_sites_all: Sequence[Mapping[str, Any]],
                                     selected_sites_incremental: Sequence[Mapping[str, Any]],
                                     selected_sites_overlap_flare: Sequence[Mapping[str, Any]],
                                     flare_grid_keys: set[tuple[str, int, str, str]]) -> None:
    expected_incremental, expected_overlap = partition_selected_sites(selected_sites_all, flare_grid_keys)
    require(list(selected_sites_incremental) == expected_incremental,
            "selected_sites_incremental is not all minus FLARE grid")
    require(list(selected_sites_overlap_flare) == expected_overlap,
            "selected_sites_overlap_flare is not all intersect FLARE grid")
    left = {variant_key(row) for row in selected_sites_incremental}
    right = {variant_key(row) for row in selected_sites_overlap_flare}
    all_keys = {variant_key(row) for row in selected_sites_all}
    require(left.isdisjoint(right) and left | right == all_keys,
            "selected-site partitions are not a disjoint union")


def validate_rare_enabled_inputs(input_logical_ids: Sequence[str]) -> None:
    """RE consumes only loci not already visible on the frozen FLARE grid."""
    expected = [
        "selected_sites_incremental", "target_rare_incremental", "flare_anc", "flare_anc_tbi",
        "genetic_map",
    ]
    require(list(input_logical_ids) == expected,
            "rare-enabled input bundle is not the exact minimal frozen interface")
    require(set(input_logical_ids).isdisjoint({"selected_sites_all", "selected_sites_overlap_flare"}),
            "rare-enabled arm includes loci already visible to FLARE")


def root_independent_haplotype_sha256(ancestry: str,
                                      alleles: Sequence[tuple[str, int, str, str, int]]) -> str:
    require(ancestry in {"AFR", "EUR", "ASIA"}, "invalid source ancestry")
    ordered = sorted(alleles, key=lambda row: row[:4])
    require(list(alleles) == ordered, "haplotype variants are not in canonical locus order")
    for row in ordered:
        variant_key({"CHROM": row[0], "POS": row[1], "REF": row[2], "ALT": row[3]})
        require(row[4] in (0, 1), "haplotype allele is not phased biallelic")
    return canonical_json_sha256({"ancestry": ancestry, "ordered_locus_alleles": ordered})


def normalized_tree_hashes(tables: Mapping[str, Any]) -> dict[str, str]:
    """Separate scientific full-tree content from genealogy-only content.

    UUID, provenance and serialization metadata are excluded unconditionally.
    Table rows are serialized in deterministic order, so harmless row permutations
    do not alter fixture hashes.  Sites and mutations affect ``normalized_full_tree``
    but never genealogy.  This fixture helper does not claim node-ID isomorphism:
    the future real tskit adapter must first normalize referenced IDs (or use the
    root-independent haplotype fingerprints) before calling this function.
    """
    required = {
        "sequence_length", "time_units", "nodes", "edges", "individuals", "populations",
        "sites", "mutations",
    }
    require(required.issubset(tables), "tree fixture lacks a scientific table")
    require(isinstance(tables["sequence_length"], (int, float)) and
            math.isfinite(tables["sequence_length"]) and tables["sequence_length"] > 0,
            "tree sequence_length is invalid")
    require(isinstance(tables["time_units"], str) and tables["time_units"],
            "tree time_units is absent")

    def normalized_rows(name: str) -> list[Any]:
        rows = tables.get(name, [])
        require(isinstance(rows, (list, tuple)), f"tree table {name} is not a row sequence")
        return sorted(rows, key=lambda row: canonical_json(row))

    common = {
        "sequence_length": tables["sequence_length"],
        "time_units": tables["time_units"],
        **{key: normalized_rows(key)
           for key in ("nodes", "edges", "individuals", "populations", "migrations")},
    }
    full = {**common, "sites": normalized_rows("sites"), "mutations": normalized_rows("mutations")}
    return {
        "normalized_full_tree_sha256": canonical_json_sha256(full),
        "normalized_genealogy_sha256": canonical_json_sha256(common),
    }


def validate_complete_diploid_roles(roles: dict[str, Any], root_seed: int,
                                    asset_contract: dict[str, Any]) -> dict[str, set[str]]:
    """Reuse the frozen PRE-4 counts, ancestry composition and disjunction checks."""
    return validate_roles(roles, root_seed, asset_contract)


EVENT_KEYS = {"target_haplotype_id", "start_bp", "end_bp", "donor_haplotype_id", "ancestry"}
TRUTH_KEYS = {"target_haplotype_id", "start_bp", "end_bp", "ancestry"}


def truth_from_mosaic_events(events: Sequence[Mapping[str, Any]],
                             expected_target_haplotypes: set[str],
                             donor_ancestry: Mapping[str, str],
                             map_start_bp: int, map_end_exclusive: int) -> list[dict[str, Any]]:
    """Project donor events to ancestry truth and merge adjacent equal-ancestry segments."""
    require(len(expected_target_haplotypes) == 60, "M33 requires exactly 60 TARGET haplotypes")
    require(0 <= map_start_bp < map_end_exclusive, "invalid chr22 map domain")
    ordered = sorted((dict(row) for row in events),
                     key=lambda row: (row["target_haplotype_id"], row["start_bp"], row["end_bp"]))
    result: list[dict[str, Any]] = []
    previous_end: dict[str, int] = {}
    observed_targets: set[str] = set()
    observed_donors: set[str] = set()
    for row in ordered:
        exact_keys(row, EVENT_KEYS, "mosaic event")
        target = row["target_haplotype_id"]
        require(target in expected_target_haplotypes, "mosaic event contains an unknown TARGET")
        donor = row["donor_haplotype_id"]
        require(donor in donor_ancestry, "mosaic event contains an unknown DONOR")
        require(donor not in observed_donors,
                "DONOR haplotype was sampled with replacement across mosaic events")
        observed_donors.add(donor)
        require(row["ancestry"] in {"AFR", "EUR", "ASIA"} and
                donor_ancestry[donor] == row["ancestry"],
                "mosaic ancestry differs from DONOR ancestry")
        require(isinstance(row["start_bp"], int) and isinstance(row["end_bp"], int) and
                0 <= row["start_bp"] < row["end_bp"], "invalid half-open mosaic event")
        if target in previous_end:
            require(row["start_bp"] == previous_end[target], "mosaic events contain gap or overlap")
        else:
            require(row["start_bp"] == map_start_bp, "TARGET truth starts after the map domain")
            observed_targets.add(target)
        previous_end[target] = row["end_bp"]
        projected = {key: row[key] for key in TRUTH_KEYS}
        if (result and result[-1]["target_haplotype_id"] == target and
                result[-1]["end_bp"] == row["start_bp"] and
                result[-1]["ancestry"] == row["ancestry"]):
            result[-1]["end_bp"] = row["end_bp"]
        else:
            result.append(projected)
    require(observed_targets == expected_target_haplotypes, "TARGET event inventory is incomplete")
    require(all(previous_end[target] == map_end_exclusive for target in expected_target_haplotypes),
            "TARGET truth ends before or after the map domain")
    return result


def validate_truth_equals_events(events: Sequence[Mapping[str, Any]],
                                 truth: Sequence[Mapping[str, Any]],
                                 expected_target_haplotypes: set[str],
                                 donor_ancestry: Mapping[str, str],
                                 map_start_bp: int, map_end_exclusive: int) -> None:
    observed = [dict(row) for row in truth]
    for row in observed:
        exact_keys(row, TRUTH_KEYS, "truth segment")
    require(observed == truth_from_mosaic_events(
        events, expected_target_haplotypes, donor_ancestry, map_start_bp, map_end_exclusive
    ),
            "truth is not the exact ancestry merge of mosaic events")


def validate_donor_to_target_genotypes(
    events: Sequence[Mapping[str, Any]],
    donor_alleles: Mapping[tuple[str, tuple[str, int, str, str]], int],
    target_alleles: Mapping[tuple[str, tuple[str, int, str, str]], int],
    expected_target_haplotypes: set[str],
    required_loci: set[tuple[str, int, str, str]],
) -> None:
    """Recompute every TARGET allele from its DONOR event at the same locus."""
    require(len(expected_target_haplotypes) == 60 and required_loci,
            "TARGET/locus domain is empty or differs from M33")
    by_target: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        exact_keys(dict(event), EVENT_KEYS, "mosaic event")
        by_target.setdefault(event["target_haplotype_id"], []).append(event)
    require(set(by_target) == expected_target_haplotypes,
            "mosaic event TARGET inventory differs from genotype TARGETs")
    expected_target_keys = {
        (target, locus) for target in expected_target_haplotypes for locus in required_loci
    }
    require(set(target_alleles) == expected_target_keys,
            "TARGET genotype matrix is not the complete target-haplotype by locus product")
    for (target, locus), observed in target_alleles.items():
        require(target in by_target, "TARGET genotype has no mosaic provenance")
        variant_key({"CHROM": locus[0], "POS": locus[1], "REF": locus[2], "ALT": locus[3]})
        matches = [event for event in by_target[target]
                   if event["start_bp"] <= locus[1] < event["end_bp"]]
        require(len(matches) == 1, "TARGET locus does not map to exactly one donor event")
        donor_key = (matches[0]["donor_haplotype_id"], locus)
        require(donor_key in donor_alleles, "mosaic DONOR allele is absent")
        require(observed in (0, 1) and donor_alleles[donor_key] == observed,
                "TARGET allele differs from its DONOR allele")


FLARE_ROW_KEYS = {"target_haplotype_id", "CHROM", "POS", "REF", "ALT", "probabilities"}


def validate_flare_probability_tensor(rows: Sequence[Mapping[str, Any]],
                                      target_haplotypes: set[str],
                                      flare_grid: Sequence[tuple[str, int, str, str]]) -> None:
    """Check exact sample/locus parity plus finite three-ancestry simplexes."""
    require(len(target_haplotypes) == 60, "FLARE must contain exactly 60 TARGET haplotypes")
    grid = list(flare_grid)
    require(len(grid) == len(set(grid)) and grid, "FLARE grid is empty or duplicated")
    expected = {(target, locus) for target in target_haplotypes for locus in grid}
    observed: set[tuple[str, tuple[str, int, str, str]]] = set()
    for raw in rows:
        row = dict(raw)
        exact_keys(row, FLARE_ROW_KEYS, "FLARE probability row")
        locus = variant_key(row)
        pair = (row["target_haplotype_id"], locus)
        require(pair not in observed, "duplicate FLARE sample-locus row")
        observed.add(pair)
        probabilities = row["probabilities"]
        require(isinstance(probabilities, (list, tuple)) and len(probabilities) == 3,
                "FLARE probability width drift")
        require(all(isinstance(value, (int, float)) and math.isfinite(value) and
                    0.0 <= value <= 1.0 for value in probabilities),
                "FLARE probability is nonfinite or outside [0,1]")
        require(math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6),
                "FLARE probabilities do not form a simplex")
    require(observed == expected, "FLARE samples or loci differ from TARGET/grid parity")


def flare_interface_sha256(assets: Mapping[str, dict[str, Any]],
                           amendment: dict[str, Any]) -> str:
    contract = amendment["flare_contract"]
    inputs = contract["input_logical_ids"]
    require(all(logical_id in assets for logical_id in inputs), "FLARE input descriptor absent")
    return canonical_json_sha256({
        "input_descriptors": {logical_id: assets[logical_id] for logical_id in inputs},
        "output_logical_ids": contract["output_logical_ids"],
        "direct_command_output_logical_ids": contract["direct_command_output_logical_ids"],
        "version": contract["version"],
        "reported_build": contract["reported_build"],
        "jar_sha256": contract["jar_sha256"],
        "container_digest": contract["container_digest"],
        "parameters": contract["parameters"],
        "command_argv": contract["command_argv"],
        "ancestry_order": contract["ancestry_order"],
    })


def parse_utc_timestamp(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and
            re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z",
                         value) is not None,
            f"{label} is not a strict UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo == timezone.utc, f"{label} is not UTC")
    return parsed


def validate_flare_truth_blind(receipt: dict[str, Any], plan: dict[str, Any],
                               assets: Mapping[str, dict[str, Any]],
                               amendment: dict[str, Any]) -> None:
    exact_keys(receipt, FLARE_RECEIPT_KEYS, "FLARE receipt")
    require(receipt["stage"] == "M33_FLARE_TRUTH_BLIND" and receipt["status"] == "PASS",
            "FLARE truth-blind receipt did not pass")
    require(receipt["plan_id"] == plan["plan_id"], "FLARE receipt plan drift")
    require(receipt["truth_mounted"] is False and receipt["truth_argument_available"] is False,
            "truth was addressable by FLARE")
    require(receipt["sealed_before_truth_mount"] is True,
            "FLARE receipt was not sealed before the truth mount")
    contract = amendment["flare_contract"]
    inputs = contract["input_logical_ids"]
    require(receipt["input_logical_ids"] == inputs, "FLARE input inventory/order drift")
    require(set(inputs).isdisjoint(set(amendment["truth_barrier"]["flare_forbidden_inputs"])),
            "FLARE receipt includes a truth-bearing input")
    descriptor_hashes = receipt["input_descriptor_sha256"]
    require(isinstance(descriptor_hashes, dict) and set(descriptor_hashes) == set(inputs),
            "FLARE input descriptor-hash inventory drift")
    for logical_id in inputs:
        require(descriptor_hashes[logical_id] == canonical_json_sha256(assets[logical_id]),
                f"FLARE receipt is not bound to exact descriptor: {logical_id}")
    outputs = contract["output_logical_ids"]
    output_hashes = receipt["output_descriptor_sha256"]
    require(isinstance(output_hashes, dict) and set(output_hashes) == set(outputs),
            "FLARE output descriptor-hash inventory drift")
    for logical_id in outputs:
        require(output_hashes[logical_id] == canonical_json_sha256(assets[logical_id]),
                f"FLARE receipt is not bound to exact output descriptor: {logical_id}")
    require(receipt["flare_version"] == contract["version"] and
            receipt["flare_reported_build"] == contract["reported_build"],
            "FLARE version/build drift")
    require(receipt["flare_jar_sha256"] == contract["jar_sha256"], "FLARE JAR drift")
    require(receipt["container_digest"] == contract["container_digest"],
            "FLARE container drift")
    require(receipt["parameters"] == contract["parameters"], "FLARE parameter drift")
    require(receipt["ancestry_order"] == contract["ancestry_order"],
            "FLARE ancestry-order drift")
    require(receipt["simulation_engine_commit"] == plan["engine"]["git_commit"],
            "FLARE receipt simulation-engine anchor drift")
    require(receipt["orchestrator_commit"] == plan["orchestrator"]["git_commit"],
            "FLARE receipt orchestrator anchor drift")
    require(receipt["generator_source_auth_sha256"] ==
            plan["generator_source_auth_sha256"],
            "FLARE receipt source-auth drift")
    require(receipt["run_manifest_sha256"] == assets["flare_run_manifest"]["sha256_raw"],
            "FLARE receipt run-manifest drift")
    require(receipt["interface_sha256"] == flare_interface_sha256(assets, amendment),
            "FLARE interface hash drift")
    require(receipt["prediction_sha256"] == assets["flare_anc"]["sha256_raw"],
            "FLARE receipt prediction hash drift")
    require(receipt["audit_payload_sha256"] == assets["flare_audit"]["sha256_raw"],
            "FLARE receipt audit-payload hash drift")
    parse_utc_timestamp(receipt["sealed_at_utc"], "FLARE receipt sealed_at_utc")
    for key in ("flare_jar_sha256", "generator_source_auth_sha256", "run_manifest_sha256",
                "interface_sha256", "prediction_sha256", "audit_payload_sha256"):
        valid_sha(receipt[key], f"FLARE receipt {key}")


def validate_reopened_flare_receipt(descriptor: dict[str, Any],
                                    observation: Mapping[str, Any],
                                    embedded_receipt: dict[str, Any]) -> None:
    """Authenticate separately stored receipt bytes, not self-reported receipt fields."""
    exact_keys(dict(observation), {"payload", "generation", "crc32c"},
               "reopened FLARE receipt observation")
    verify_observed_bytes(
        descriptor, observation["payload"], observation["generation"], observation["crc32c"]
    )
    require(observation["payload"] == canonical_json(embedded_receipt),
            "reopened FLARE receipt differs from its canonical embedded copy")
    require(strict_json_bytes(observation["payload"], "reopened FLARE receipt") ==
            embedded_receipt, "reopened FLARE receipt semantic drift")


def validate_reopened_flare_run_and_audit(
    plan: dict[str, Any], assets: Mapping[str, dict[str, Any]],
    observations: Mapping[str, Mapping[str, Any]], receipt: dict[str, Any],
    amendment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse and cross-bind the independently reopened FLARE run and audit JSON."""
    for logical_id in ("flare_run_manifest", "flare_audit"):
        require(logical_id in observations, f"READY observation absent: {logical_id}")
        observation = observations[logical_id]
        exact_keys(dict(observation), {"payload", "generation", "crc32c"},
                   f"reopened {logical_id} observation")
        verify_observed_bytes(
            assets[logical_id], observation["payload"], observation["generation"],
            observation["crc32c"],
        )

    run = strict_json_bytes(observations["flare_run_manifest"]["payload"],
                            "FLARE run manifest")
    exact_keys(run, FLARE_RUN_MANIFEST_KEYS, "FLARE run manifest")
    require(run["schema_version"] == "1.0.0" and
            run["stage"] == "M33_FLARE_RUN_MANIFEST" and run["status"] == "PASS",
            "FLARE run manifest schema, stage or status drift")
    require(run["plan_id"] == plan["plan_id"], "FLARE run manifest plan drift")
    require(run["interface_sha256"] == flare_interface_sha256(assets, amendment),
            "FLARE run manifest interface drift")
    require(run["generator_source_auth_sha256"] == plan["generator_source_auth_sha256"],
            "FLARE run manifest source-auth drift")
    require(run["simulation_engine_commit"] == plan["engine"]["git_commit"] and
            run["orchestrator_commit"] == plan["orchestrator"]["git_commit"],
            "FLARE run manifest code anchor drift")
    contract = amendment["flare_contract"]
    require(run["flare_version"] == contract["version"] and
            run["flare_reported_build"] == contract["reported_build"] and
            run["flare_jar_sha256"] == contract["jar_sha256"] and
            run["container_digest"] == contract["container_digest"],
            "FLARE run manifest binary/runtime drift")
    require(run["command_argv"] == contract["command_argv"],
            "FLARE command argv differs from the frozen interface")
    require(not any("truth" in value.lower() for value in run["command_argv"]),
            "FLARE command argv exposes truth")
    require(run["truth_accessed"] is False, "FLARE run accessed truth")
    expected_inputs = {
        logical_id: canonical_json_sha256(assets[logical_id])
        for logical_id in contract["input_logical_ids"]
    }
    expected_outputs = {
        logical_id: canonical_json_sha256(assets[logical_id])
        for logical_id in contract["direct_command_output_logical_ids"]
    }
    require(run["input_descriptor_sha256"] == expected_inputs,
            "FLARE run input descriptors drift")
    require(run["output_descriptor_sha256"] == expected_outputs,
            "FLARE run output descriptors drift")
    require(run["input_descriptor_sha256"] == receipt["input_descriptor_sha256"],
            "FLARE receipt and run manifest input bindings differ")
    require(all(receipt["output_descriptor_sha256"][logical_id] == digest
                for logical_id, digest in run["output_descriptor_sha256"].items()),
            "FLARE receipt and run manifest direct-output bindings differ")
    run_start = parse_utc_timestamp(run["started_at_utc"], "FLARE run started_at_utc")
    run_finish = parse_utc_timestamp(run["finished_at_utc"], "FLARE run finished_at_utc")
    require(run_start <= run_finish, "FLARE run timestamps are reversed")

    audit = strict_json_bytes(observations["flare_audit"]["payload"], "FLARE audit")
    exact_keys(audit, FLARE_AUDIT_KEYS, "FLARE audit")
    require(audit["schema_version"] == "1.0.0" and
            audit["stage"] == "M33_FLARE_TRUTH_BLIND_AUDIT" and audit["status"] == "PASS",
            "FLARE audit schema, stage or status drift")
    require(audit["plan_id"] == plan["plan_id"] and
            audit["interface_sha256"] == run["interface_sha256"],
            "FLARE audit plan/interface drift")
    require(audit["run_manifest_sha256"] == assets["flare_run_manifest"]["sha256_raw"] ==
            receipt["run_manifest_sha256"], "FLARE audit run-manifest binding drift")
    require(audit["prediction_sha256"] == assets["flare_anc"]["sha256_raw"] ==
            receipt["prediction_sha256"], "FLARE audit prediction binding drift")
    require(audit["target_haplotype_count"] == 60 and
            isinstance(audit["locus_count"], int) and audit["locus_count"] > 0,
            "FLARE audit TARGET/locus domain drift")
    require(audit["sample_parity_exact"] is True and audit["locus_parity_exact"] is True and
            audit["probabilities_finite"] is True and
            audit["probability_simplex_exact_within_tolerance"] is True,
            "FLARE audit parity, finiteness or simplex check failed")
    require(isinstance(audit["simplex_tolerance"], (int, float)) and
            math.isclose(audit["simplex_tolerance"], 1e-6, rel_tol=0, abs_tol=0),
            "FLARE audit simplex tolerance drift")
    require(audit["truth_accessed"] is False, "FLARE audit accessed truth")
    audit_start = parse_utc_timestamp(audit["started_at_utc"], "FLARE audit started_at_utc")
    audit_finish = parse_utc_timestamp(audit["finished_at_utc"], "FLARE audit finished_at_utc")
    sealed = parse_utc_timestamp(receipt["sealed_at_utc"], "FLARE receipt sealed_at_utc")
    require(run_start <= run_finish <= audit_start <= audit_finish <= sealed,
            "FLARE run, audit and receipt timestamps are not ordered")
    return run, audit


def parse_selected_sites_document(
    logical_id: str, descriptor: dict[str, Any], observation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    require(logical_id in {"selected_sites_incremental", "selected_sites_overlap_flare"},
            "unsupported selected-sites document")
    verify_observed_bytes(
        descriptor, observation["payload"], observation["generation"], observation["crc32c"]
    )
    document = strict_json_bytes(observation["payload"], logical_id)
    exact_keys(document, SELECTED_SITES_DOCUMENT_KEYS, logical_id)
    expected_stage = ("M33_SELECTED_SITES_INCREMENTAL" if logical_id ==
                      "selected_sites_incremental" else "M33_SELECTED_SITES_OVERLAP_FLARE")
    require(document["schema_version"] == "1.0.0" and
            document["stage"] == expected_stage and document["status"] == "PASS",
            f"{logical_id} document schema, stage or status drift")
    require(isinstance(document["rows"], list), f"{logical_id} rows are absent")
    rows = [dict(row) for row in document["rows"]]
    for row in rows:
        exact_keys(row, SELECTED_SITE_KEYS, logical_id)
        variant_key(row)
        require(type(row["minor_allele_index"]) is int and
                row["minor_allele_index"] in (0, 1),
                f"{logical_id} minor allele orientation is invalid")
        require(type(row["minor_mac"]) is int and row["minor_mac"] >= 2 and
                type(row["minor_an"]) is int and row["minor_an"] > 0 and
                row["minor_an"] % 2 == 0 and row["minor_mac"] * 2 < row["minor_an"],
                f"{logical_id} minor MAC/AN is invalid")
        require(type(row["carrier_people"]) is int and row["carrier_people"] >= 2 and
                row["carrier_people"] <= row["minor_mac"] <= 2 * row["carrier_people"],
                f"{logical_id} carrier count is inconsistent with minor MAC")
        require(type(row["minor_maf"]) in (int, float) and
                math.isfinite(row["minor_maf"]) and 0 < row["minor_maf"] < 0.01 and
                math.isclose(row["minor_maf"], row["minor_mac"] / row["minor_an"],
                             rel_tol=0, abs_tol=1e-15),
                f"{logical_id} minor MAF is inconsistent with MAC/AN or the rare threshold")
    ordered = sorted(rows, key=lambda row: (row["CHROM"], row["POS"], row["REF"], row["ALT"]))
    require(rows == ordered and len({variant_key(row) for row in rows}) == len(rows),
            f"{logical_id} rows are not canonical and unique")
    require(descriptor["record_count"] == len(rows), f"{logical_id} record count drift")
    return rows


def validate_target_rare_incremental_observations(
    assets: Mapping[str, dict[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require the exact TARGET-haplotype by incremental-locus phased matrix."""
    required = {
        "selected_sites_incremental", "selected_sites_overlap_flare",
        "target_rare_incremental",
    }
    require(required.issubset(observations), "incremental rare-channel observation absent")
    incremental = parse_selected_sites_document(
        "selected_sites_incremental", assets["selected_sites_incremental"],
        observations["selected_sites_incremental"],
    )
    overlap = parse_selected_sites_document(
        "selected_sites_overlap_flare", assets["selected_sites_overlap_flare"],
        observations["selected_sites_overlap_flare"],
    )
    incremental_keys = {variant_key(row) for row in incremental}
    overlap_keys = {variant_key(row) for row in overlap}
    require(incremental_keys, "incremental rare-locus channel is empty")
    require(incremental_keys.isdisjoint(overlap_keys),
            "incremental rare loci overlap the frozen FLARE grid")
    descriptor = assets["target_rare_incremental"]
    observation = observations["target_rare_incremental"]
    verify_observed_bytes(
        descriptor, observation["payload"], observation["generation"], observation["crc32c"]
    )
    document = strict_json_bytes(observation["payload"], "target_rare_incremental")
    exact_keys(document, TARGET_RARE_DOCUMENT_KEYS, "target_rare_incremental")
    require(document["schema_version"] == "1.0.0" and
            document["stage"] == "M33_TARGET_RARE_INCREMENTAL" and
            document["status"] == "PASS", "target rare document schema, stage or status drift")
    target_haplotypes = document["target_haplotype_ids"]
    require(isinstance(target_haplotypes, list) and len(target_haplotypes) == 60 and
            target_haplotypes == sorted(target_haplotypes) and
            len(set(target_haplotypes)) == 60 and
            all(isinstance(value, str) and value for value in target_haplotypes),
            "target rare document does not contain exactly 60 canonical TARGET haplotypes")
    rows = document["rows"]
    require(isinstance(rows, list), "target rare rows are absent")
    observed_pairs: set[tuple[str, tuple[str, int, str, str]]] = set()
    for raw in rows:
        row = dict(raw)
        exact_keys(row, TARGET_RARE_ROW_KEYS, "target rare incremental row")
        locus = variant_key(row)
        require(row["target_haplotype_id"] in target_haplotypes,
                "target rare row contains an unknown TARGET haplotype")
        require(locus in incremental_keys and locus not in overlap_keys,
                "target rare row contains a nonincremental or overlapping locus")
        require(isinstance(row["minor_allele_presence"], int) and
                not isinstance(row["minor_allele_presence"], bool) and
                row["minor_allele_presence"] in (0, 1),
                "haplotype minor-allele presence is not 0/1")
        pair = (row["target_haplotype_id"], locus)
        require(pair not in observed_pairs, "duplicate target-haplotype/locus rare row")
        observed_pairs.add(pair)
    expected_pairs = {(target, locus) for target in target_haplotypes for locus in incremental_keys}
    require(observed_pairs == expected_pairs,
            "target rare matrix is not the exact TARGET-haplotype by incremental-locus product")
    require(descriptor["record_count"] == len(rows), "target rare descriptor record count drift")
    return document


def reopen_strict_json_asset(
    logical_id: str, assets: Mapping[str, dict[str, Any]],
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    require(logical_id in assets and logical_id in observations,
            f"reopened JSON asset absent: {logical_id}")
    observation = observations[logical_id]
    exact_keys(dict(observation), {"payload", "generation", "crc32c"},
               f"reopened {logical_id} observation")
    verify_observed_bytes(
        assets[logical_id], observation["payload"], observation["generation"],
        observation["crc32c"],
    )
    return strict_json_bytes(observation["payload"], logical_id)


def parse_roles_observation(
    root_seed: int, assets: Mapping[str, dict[str, Any]],
    observations: Mapping[str, Mapping[str, Any]], asset_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, set[str]], dict[str, str]]:
    document = reopen_strict_json_asset("roles", assets, observations)
    exact_keys(document, ROLES_DOCUMENT_KEYS, "roles document")
    require(document["schema_version"] == "1.0.0" and
            document["stage"] == "M33_COMPLETE_DIPLOID_ROLES" and
            document["status"] == "PASS", "roles document schema, stage or status drift")
    roles = document["roles"]
    require(isinstance(roles, dict), "roles payload is not an object")
    role_haplotypes = validate_complete_diploid_roles(roles, root_seed, asset_contract)
    require(assets["roles"]["record_count"] == sum(len(rows) for rows in roles.values()),
            "roles descriptor record count drift")
    donor_ancestry: dict[str, str] = {}
    for person in roles["DONOR"]:
        for haplotype in person["haplotypes"]:
            donor_ancestry[haplotype["haplotype_id"]] = person["ancestry"]
    require(set(donor_ancestry) == role_haplotypes["DONOR"],
            "DONOR ancestry registry differs from complete roles")
    return roles, role_haplotypes, donor_ancestry


def parse_genetic_map_domain(
    descriptor: dict[str, Any], observation: Mapping[str, Any],
) -> tuple[int, int]:
    verify_observed_bytes(
        descriptor, observation["payload"], observation["generation"], observation["crc32c"]
    )
    try:
        text = observation["payload"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("genetic map fixture is not UTF-8") from exc
    positions: list[int] = []
    previous_bp = -1
    previous_cm = -math.inf
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0].lower() in {"chrom", "chr", "chromosome"}:
            continue
        if fields[0] not in {"22", "chr22"}:
            continue
        try:
            bp = int(fields[1])
            cm = float(fields[2])
        except ValueError as exc:
            raise ValueError("invalid chr22 genetic-map row") from exc
        require(bp > previous_bp and math.isfinite(cm) and cm >= previous_cm,
                "genetic map positions/cM are not monotone")
        positions.append(bp)
        previous_bp, previous_cm = bp, cm
    require(len(positions) >= 2, "genetic map lacks two chr22 boundary positions")
    require(descriptor["record_count"] == len(positions), "genetic map record count drift")
    return positions[0], positions[-1] + 1


def parse_locus_rows(rows: Any, label: str) -> list[tuple[str, int, str, str]]:
    require(isinstance(rows, list), f"{label} loci are absent")
    loci: list[tuple[str, int, str, str]] = []
    for raw in rows:
        row = dict(raw)
        exact_keys(row, {"CHROM", "POS", "REF", "ALT"}, f"{label} locus")
        loci.append(variant_key(row))
    require(loci == sorted(loci) and len(loci) == len(set(loci)) and loci,
            f"{label} loci are empty, duplicated or noncanonical")
    return loci


def validate_probability_simplex(probabilities: Any, label: str) -> None:
    require(isinstance(probabilities, list) and len(probabilities) == 3,
            f"{label} probability width drift")
    require(all(type(value) in (int, float) and math.isfinite(value) and
                0 <= value <= 1 for value in probabilities),
            f"{label} contains invalid probabilities")
    require(math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6),
            f"{label} probabilities do not form a simplex")


def validate_reopened_flare_outputs(
    assets: Mapping[str, dict[str, Any]], observations: Mapping[str, Mapping[str, Any]],
    expected_target_haplotypes: set[str], amendment: dict[str, Any],
) -> list[tuple[str, int, str, str]]:
    target = reopen_strict_json_asset("target_vcf", assets, observations)
    exact_keys(target, TARGET_VCF_FIXTURE_KEYS, "TARGET VCF fixture view")
    require(target["schema_version"] == "1.0.0" and
            target["stage"] == "M33_TARGET_VCF_FIXTURE_VIEW" and target["status"] == "PASS" and
            target["chromosome"] == "chr22", "TARGET VCF fixture view drift")
    target_ids = target["target_haplotype_ids"]
    require(isinstance(target_ids, list) and target_ids == sorted(expected_target_haplotypes),
            "TARGET VCF haplotypes differ from complete roles")
    grid = parse_locus_rows(target["loci"], "FLARE target grid")
    require(assets["target_vcf"]["record_count"] == len(grid),
            "TARGET VCF fixture record count drift")

    anc = reopen_strict_json_asset("flare_anc", assets, observations)
    exact_keys(anc, FLARE_ANC_DOCUMENT_KEYS, "FLARE ancestry document")
    require(anc["schema_version"] == "1.0.0" and anc["stage"] == "M33_FLARE_ANC" and
            anc["status"] == "PASS" and anc["chromosome"] == "chr22",
            "FLARE ancestry document drift")
    require(anc["target_haplotype_ids"] == target_ids and
            parse_locus_rows(anc["loci"], "FLARE ancestry grid") == grid,
            "FLARE ancestry IDs/loci differ from TARGET grid")
    validate_flare_probability_tensor(anc["rows"], expected_target_haplotypes, grid)
    require(assets["flare_anc"]["record_count"] == len(anc["rows"]),
            "FLARE ancestry record count drift")

    global_doc = reopen_strict_json_asset("flare_global", assets, observations)
    exact_keys(global_doc, FLARE_GLOBAL_DOCUMENT_KEYS, "FLARE global document")
    require(global_doc["schema_version"] == "1.0.0" and
            global_doc["stage"] == "M33_FLARE_GLOBAL" and global_doc["status"] == "PASS" and
            global_doc["ancestry_order"] == amendment["flare_contract"]["ancestry_order"],
            "FLARE global document drift")
    rows = global_doc["rows"]
    require(isinstance(rows, list), "FLARE global rows are absent")
    observed_targets: set[str] = set()
    for raw in rows:
        row = dict(raw)
        exact_keys(row, FLARE_GLOBAL_ROW_KEYS, "FLARE global row")
        target_id = row["target_haplotype_id"]
        require(target_id in expected_target_haplotypes and target_id not in observed_targets,
                "FLARE global TARGET inventory is unknown or duplicated")
        observed_targets.add(target_id)
        validate_probability_simplex(row["probabilities"], "FLARE global")
    require(observed_targets == expected_target_haplotypes,
            "FLARE global TARGET inventory is incomplete")
    require(assets["flare_global"]["record_count"] == len(rows),
            "FLARE global record count drift")
    audit = strict_json_bytes(observations["flare_audit"]["payload"], "FLARE audit grid binding")
    require(audit["target_haplotype_count"] == len(expected_target_haplotypes) and
            audit["locus_count"] == len(grid),
            "FLARE audit counts differ from reopened TARGET/grid")

    for logical_id in ("ref_tbi", "target_tbi", "flare_anc_tbi"):
        payload = observations[logical_id]["payload"]
        require(len(payload) >= 16 and payload.startswith(b"TBI\x01"),
                f"{logical_id} lacks the fixture Tabix magic/nontrivial payload")
    model = observations["flare_model"]["payload"]
    require(len(model) >= 64 and model.startswith(b"FLARE_MODEL\t0.6.0\n"),
            "FLARE model lacks its versioned header/nontrivial payload")
    log = observations["flare_log"]["payload"]
    require(len(log) >= 80 and
            b"flare version 0.6.0 [616fcc9d4 03-Nov-2025]" in log and
            b"Analysis finished" in log,
            "FLARE log lacks its exact version/completion markers")
    return grid


def validate_reopened_mosaic_truth_provenance(
    plan: dict[str, Any], assets: Mapping[str, dict[str, Any]],
    observations: Mapping[str, Mapping[str, Any]], target_rare_document: dict[str, Any],
    role_haplotypes: Mapping[str, set[str]], allowed_donor_ancestry: Mapping[str, str],
) -> None:
    expected_targets = set(target_rare_document["target_haplotype_ids"])
    require(expected_targets == role_haplotypes["TARGET"] and len(expected_targets) == 60,
            "mosaic TARGET haplotypes differ from complete roles/rare channel")
    map_start, map_end = parse_genetic_map_domain(
        assets["genetic_map"], observations["genetic_map"]
    )

    mosaic = reopen_strict_json_asset("mosaic_events", assets, observations)
    exact_keys(mosaic, MOSAIC_DOCUMENT_KEYS, "mosaic-events document")
    require(mosaic["schema_version"] == "1.0.0" and
            mosaic["stage"] == "M33_MOSAIC_EVENTS" and mosaic["status"] == "PASS" and
            mosaic["chromosome"] == "chr22", "mosaic-events document drift")
    events = mosaic["rows"]
    require(isinstance(events, list), "mosaic event rows are absent")
    used_donors = {row.get("donor_haplotype_id") for row in events if isinstance(row, dict)}
    require(used_donors and used_donors.issubset(role_haplotypes["DONOR"]),
            "mosaic uses a haplotype outside the DONOR role")
    event_donor_ancestry = {donor: allowed_donor_ancestry[donor] for donor in used_donors}
    expected_truth = truth_from_mosaic_events(
        events, expected_targets, event_donor_ancestry, map_start, map_end,
    )
    require(assets["mosaic_events"]["record_count"] == len(events),
            "mosaic-events descriptor record count drift")

    truth_document = reopen_strict_json_asset("truth", assets, observations)
    exact_keys(truth_document, TRUTH_DOCUMENT_KEYS, "truth document")
    require(truth_document["schema_version"] == "1.0.0" and
            truth_document["stage"] == "M33_HAPLOTYPE_TRUTH" and
            truth_document["status"] == "PASS" and truth_document["chromosome"] == "chr22",
            "truth document drift")
    truth_rows = truth_document["rows"]
    require(isinstance(truth_rows, list), "truth rows are absent")
    validate_truth_equals_events(
        events, truth_rows, expected_targets, event_donor_ancestry, map_start, map_end,
    )
    require(truth_rows == expected_truth and assets["truth"]["record_count"] == len(truth_rows),
            "truth rows/count differ from the exact event-derived truth")

    provenance = reopen_strict_json_asset("donor_to_target_provenance", assets, observations)
    exact_keys(provenance, PROVENANCE_DOCUMENT_KEYS, "donor-target provenance document")
    require(provenance["schema_version"] == "1.0.0" and
            provenance["stage"] == "M33_DONOR_TARGET_PROVENANCE" and
            provenance["status"] == "PASS" and provenance["chromosome"] == "chr22",
            "donor-target provenance document drift")
    require(provenance["donor_ancestry"] == event_donor_ancestry,
            "provenance DONOR registry differs from allowed event DONORs")
    required_loci = {
        variant_key(row) for row in target_rare_document["rows"]
    }
    require(required_loci, "provenance has no incremental rare loci")
    selected_incremental = parse_selected_sites_document(
        "selected_sites_incremental", assets["selected_sites_incremental"],
        observations["selected_sites_incremental"],
    )
    minor_index_by_locus = {
        variant_key(row): row["minor_allele_index"] for row in selected_incremental
    }
    require(set(minor_index_by_locus) == required_loci,
            "rare presence loci differ from the reopened incremental selector")
    rare_presence: dict[tuple[str, tuple[str, int, str, str]], int] = {}
    for row in target_rare_document["rows"]:
        key = (row["target_haplotype_id"], variant_key(row))
        presence = row["minor_allele_presence"]
        require(type(presence) is int and presence in (0, 1),
                "minor-allele presence is not strict integer 0/1")
        require(key not in rare_presence, "duplicate TARGET/locus minor-allele presence")
        rare_presence[key] = presence

    def allele_mapping(rows: Any, allowed_haplotypes: set[str], label: str
                       ) -> dict[tuple[str, tuple[str, int, str, str]], int]:
        require(isinstance(rows, list), f"{label} allele rows are absent")
        result: dict[tuple[str, tuple[str, int, str, str]], int] = {}
        for raw in rows:
            row = dict(raw)
            exact_keys(row, ALLELE_ROW_KEYS, f"{label} allele row")
            haplotype = row["haplotype_id"]
            locus = variant_key(row)
            require(haplotype in allowed_haplotypes, f"{label} allele has unknown haplotype")
            require(type(row["allele"]) is int and row["allele"] in (0, 1),
                    f"{label} allele is not phased 0/1")
            key = (haplotype, locus)
            require(key not in result, f"duplicate {label} haplotype/locus allele")
            result[key] = row["allele"]
        return result

    donor_alleles = allele_mapping(provenance["donor_alleles"], used_donors, "DONOR")
    target_alleles = allele_mapping(provenance["target_alleles"], expected_targets, "TARGET")
    require(set(rare_presence) == set(target_alleles),
            "rare-presence and reconstructed TARGET allele products differ")
    for (target, locus), allele in target_alleles.items():
        minor_index = minor_index_by_locus[locus]
        expected_presence = allele if minor_index == 1 else 1 - allele
        require(rare_presence[(target, locus)] == expected_presence,
                "minor-allele presence differs from reconstructed TARGET allele orientation")
    needed_donor_keys: set[tuple[str, tuple[str, int, str, str]]] = set()
    for target in expected_targets:
        target_events = [event for event in events if event["target_haplotype_id"] == target]
        for locus in required_loci:
            matches = [event for event in target_events
                       if event["start_bp"] <= locus[1] < event["end_bp"]]
            require(len(matches) == 1, "incremental locus does not map to exactly one DONOR tile")
            needed_donor_keys.add((matches[0]["donor_haplotype_id"], locus))
    require(set(donor_alleles) == needed_donor_keys,
            "DONOR allele provenance is not exactly the event/locus requirement")
    validate_donor_to_target_genotypes(
        events, donor_alleles, target_alleles, expected_targets, required_loci,
    )
    require(assets["donor_to_target_provenance"]["record_count"] == len(target_alleles),
            "donor-target provenance descriptor record count drift")


def validate_final_manifest(manifest: dict[str, Any], plan: dict[str, Any],
                            amendment: dict[str, Any]) -> None:
    exact_keys(manifest, MANIFEST_KEYS, "final manifest")
    require(manifest["schema_version"] == "1.0.0" and manifest["stage"] == "M33_FINAL_MANIFEST",
            "final manifest schema or stage drift")
    for key in ("mode", "root_seed", "plan_id", "asset_set_id", "output_prefix"):
        require(manifest[key] == plan[key], f"final manifest changed frozen plan field {key}")
    require(manifest["plan_manifest_sha256"] == canonical_json_sha256(plan),
            "final manifest is not bound to plan bytes")
    expected_assets = set(amendment["manifest_members"])
    assets = manifest["assets"]
    require(set(assets) == expected_assets, "final asset inventory missing or extra")
    root_uris: list[str] = []
    for logical_id, descriptor in assets.items():
        validate_input_descriptor(descriptor, logical_id)
        expected_schema = amendment["manifest_members"][logical_id]["schema"]
        require(descriptor["schema_version"] == expected_schema, f"asset schema drift: {logical_id}")
        if amendment["manifest_members"][logical_id]["root_specific"]:
            require(descriptor["gcs_uri"].startswith(plan["output_prefix"]),
                    f"root-specific asset outside frozen prefix: {logical_id}")
            root_uris.append(descriptor["gcs_uri"])
    require(len(root_uris) == len(set(root_uris)),
            "two root-specific logical assets reuse the same URI within a root")
    require(assets["generator_source_auth"] ==
            plan["input_descriptors"]["generator_source_auth"],
            "final generator source-auth descriptor differs from the plan input")
    require(assets["genetic_map"] == plan["input_descriptors"]["genetic_map"],
            "final genetic-map descriptor differs from the plan input")
    predict = manifest["predict_bundle"]
    flare_inputs = manifest["flare_input_bundle"]
    rare_inputs = manifest["rare_enabled_model_bundle"]
    truth = manifest["private_truth_bundle"]
    require(predict == amendment["bundles"]["predict_bundle"], "predict bundle drift")
    require(flare_inputs == amendment["bundles"]["flare_input_bundle"],
            "FLARE input bundle drift")
    require(rare_inputs == amendment["bundles"]["rare_enabled_model_bundle"],
            "rare-enabled model bundle drift")
    validate_rare_enabled_inputs(rare_inputs)
    require(truth == amendment["bundles"]["private_truth_bundle"], "private truth bundle drift")
    require(set(predict).isdisjoint(set(truth)), "truth entered predict bundle")
    validate_flare_truth_blind(manifest["flare_receipt"], plan, assets, amendment)
    semantic = manifest["semantic_fingerprints"]
    exact_keys(semantic, SEMANTIC_KEYS, "semantic fingerprints")
    valid_sha(semantic["normalized_full_tree_sha256"], "normalized full tree")
    valid_sha(semantic["normalized_genealogy_sha256"], "normalized genealogy")
    require(semantic["normalized_full_tree_sha256"] != semantic["normalized_genealogy_sha256"],
            "full-tree and genealogy fingerprints were conflated")
    haplotypes = semantic["root_independent_source_haplotype_sha256"]
    require(isinstance(haplotypes, list) and
            len(haplotypes) == amendment["semantic_fingerprints"]
            ["required_source_haplotype_fingerprint_count"],
            "source haplotype fingerprint count differs from 2*(300+90+768)=2316")
    require(len(haplotypes) == len(set(haplotypes)), "source haplotype reused within root")
    for value in haplotypes:
        valid_sha(value, "root-independent source haplotype")
    valid_sha(manifest["final_manifest_sha256"], "final manifest")
    payload = {key: value for key, value in manifest.items() if key != "final_manifest_sha256"}
    require(manifest["final_manifest_sha256"] == canonical_json_sha256(payload),
            "final manifest canonical hash drift")


def validate_write_log(write_log: Sequence[Mapping[str, Any]],
                       descriptors: Mapping[str, dict[str, Any]],
                       amendment: dict[str, Any]) -> None:
    require(isinstance(write_log, list) and write_log, "publication log absent")
    logical_ids = [row.get("logical_id") for row in write_log]
    expected = amendment["publication"]["root_objects_before_final_manifest"] + ["final_manifest"]
    require(logical_ids == expected, "objects before READY were absent, duplicated or reordered")
    for row in write_log:
        exact_keys(dict(row), {"logical_id", "if_generation_match", "observed_generation"},
                   "publication event")
        require(type(row["if_generation_match"]) is int and row["if_generation_match"] == 0,
                "object was not created append-only")
        require(isinstance(row["observed_generation"], str) and
                re.fullmatch(r"[1-9][0-9]*", row["observed_generation"]),
                "publication generation missing")
        require(row["logical_id"] in descriptors and
                row["observed_generation"] ==
                descriptors[row["logical_id"]]["gcs_generation"],
                f"publication generation differs from descriptor: {row['logical_id']}")


def validate_ready(ready: dict[str, Any], plan: dict[str, Any], manifest: dict[str, Any],
                   asset_observations: Mapping[str, Mapping[str, Any]],
                   amendment: dict[str, Any], asset_contract: dict[str, Any]) -> None:
    """Validate the sentinel written after, and authenticating, final_manifest."""
    exact_keys(ready, READY_KEYS, "READY sentinel")
    require(ready["schema_version"] == "1.0.0" and ready["stage"] == "M33_READY",
            "READY schema or stage drift")
    require(ready["status"] == "READY" and ready["plan_id"] == plan["plan_id"],
            "READY status or plan drift")
    require(ready["final_manifest_reopened_and_verified"] is True and
            ready["all_prior_descriptors_reopened_and_verified"] is True,
            "READY was emitted before independent verification")
    require(type(ready["created_with_if_generation_match"]) is int and
            ready["created_with_if_generation_match"] == 0,
            "READY was not created append-only")
    final_descriptor = ready["final_manifest_descriptor"]
    plan_descriptor = ready["plan_manifest_descriptor"]
    validate_input_descriptor(final_descriptor, "final_manifest")
    validate_input_descriptor(plan_descriptor, "plan_manifest")
    require(final_descriptor["schema_version"] ==
            amendment["publication_envelopes"]["final_manifest"]["schema"],
            "final manifest descriptor schema drift")
    require(plan_descriptor["schema_version"] ==
            amendment["publication_envelopes"]["plan_manifest"]["schema"],
            "plan manifest descriptor schema drift")
    require(final_descriptor["gcs_uri"].startswith(plan["output_prefix"]),
            "final manifest descriptor outside frozen prefix")
    require(plan_descriptor["gcs_uri"].startswith(plan["output_prefix"]),
            "plan manifest descriptor outside frozen prefix")
    root_uris = [
        descriptor["gcs_uri"] for logical_id, descriptor in manifest["assets"].items()
        if amendment["manifest_members"][logical_id]["root_specific"]
    ] + [plan_descriptor["gcs_uri"], final_descriptor["gcs_uri"]]
    require(len(root_uris) == len(set(root_uris)),
            "root-specific asset/manifest URIs are not unique within the root")
    descriptors = {**manifest["assets"], "plan_manifest": plan_descriptor,
                   "final_manifest": final_descriptor}
    require(set(asset_observations) == set(descriptors),
            "READY lacks an independent observation for one or more assets")
    reopened: dict[str, bytes] = {}
    for logical_id, descriptor in descriptors.items():
        observation = asset_observations[logical_id]
        exact_keys(dict(observation), {"payload", "generation", "crc32c"},
                   f"READY observation {logical_id}")
        verify_observed_bytes(
            descriptor, observation["payload"], observation["generation"],
            observation["crc32c"],
        )
        reopened[logical_id] = observation["payload"]
    require(reopened["plan_manifest"] == canonical_json(plan),
            "independently reopened plan manifest is not canonical plan JSON")
    require(reopened["final_manifest"] == canonical_json(manifest),
            "independently reopened final manifest is not canonical final JSON")
    validate_reopened_flare_receipt(
        manifest["assets"]["flare_receipt"], asset_observations["flare_receipt"],
        manifest["flare_receipt"],
    )
    validate_reopened_flare_run_and_audit(
        plan, manifest["assets"], asset_observations, manifest["flare_receipt"], amendment,
    )
    _, role_haplotypes, allowed_donor_ancestry = parse_roles_observation(
        plan["root_seed"], manifest["assets"], asset_observations, asset_contract,
    )
    validate_reopened_flare_outputs(
        manifest["assets"], asset_observations, role_haplotypes["TARGET"], amendment,
    )
    target_rare_document = validate_target_rare_incremental_observations(
        manifest["assets"], asset_observations
    )
    validate_reopened_mosaic_truth_provenance(
        plan, manifest["assets"], asset_observations, target_rare_document,
        role_haplotypes, allowed_donor_ancestry,
    )
    validate_write_log(ready["publication_log"], descriptors, amendment)


def validate_root_bundle(plans: Sequence[dict[str, Any]], manifests: Sequence[dict[str, Any]],
                         readies: Sequence[dict[str, Any]],
                         ready_observations: Sequence[Mapping[str, Mapping[str, Any]]],
                         amendment: dict[str, Any], asset_contract: dict[str, Any]) -> dict[str, Any]:
    require(len(plans) == len(manifests) == len(readies) == len(ready_observations) == 3,
            "exactly three DEVELOPMENT roots are required")
    require([plan["root_seed"] for plan in plans] == list(DEVELOPMENT_ROOTS), "plan root order drift")
    all_haplotypes: set[str] = set()
    full_trees: set[str] = set()
    genealogies: set[str] = set()
    root_specific_uris: dict[str, set[str]] = {
        logical_id: set() for logical_id, definition in amendment["manifest_members"].items()
        if definition["root_specific"]
    }
    root_specific_hashes: dict[str, set[str]] = {
        logical_id: set() for logical_id, definition in amendment["manifest_members"].items()
        if definition["root_specific"]
    }
    shared_source_auth: dict[str, Any] | None = None
    for plan, manifest, ready, observation in zip(plans, manifests, readies, ready_observations):
        source_auth_descriptor = plan["input_descriptors"]["generator_source_auth"]
        require("generator_source_auth" in observation,
                "source-auth observation absent before plan validation")
        validate_generator_source_auth_observation(
            source_auth_descriptor, observation["generator_source_auth"],
            plan["orchestrator"]["git_commit"], asset_contract,
        )
        validate_plan(plan, amendment, asset_contract)
        validate_final_manifest(manifest, plan, amendment)
        validate_ready(ready, plan, manifest, observation, amendment, asset_contract)
        semantic = manifest["semantic_fingerprints"]
        current = set(semantic["root_independent_source_haplotype_sha256"])
        require(all_haplotypes.isdisjoint(current), "source haplotype reused across roots")
        all_haplotypes.update(current)
        require(semantic["normalized_full_tree_sha256"] not in full_trees,
                "normalized full tree reused across roots")
        require(semantic["normalized_genealogy_sha256"] not in genealogies,
                "normalized genealogy reused across roots")
        full_trees.add(semantic["normalized_full_tree_sha256"])
        genealogies.add(semantic["normalized_genealogy_sha256"])
        current_source_auth = plan["input_descriptors"]["generator_source_auth"]
        if shared_source_auth is None:
            shared_source_auth = current_source_auth
        else:
            require(current_source_auth == shared_source_auth,
                    "generator source-auth descriptor drifted across roots")
        for logical_id in root_specific_uris:
            descriptor = manifest["assets"][logical_id]
            require(descriptor["gcs_uri"] not in root_specific_uris[logical_id],
                    f"root-specific URI reused across roots: {logical_id}")
            require(descriptor["sha256_raw"] not in root_specific_hashes[logical_id],
                    f"root-specific bytes reused across roots: {logical_id}")
            root_specific_uris[logical_id].add(descriptor["gcs_uri"])
            root_specific_hashes[logical_id].add(descriptor["sha256_raw"])
    return {
        "status": PASS_STATUS,
        "root_seeds": list(DEVELOPMENT_ROOTS),
        "plan_ids": [plan["plan_id"] for plan in plans],
        "real_asset_read": False,
        "asset_generation": False,
        "forward": False,
        "training": False,
    }


def validate_execution_source_auth(auth_path: Path, git_commit: str,
                                   staged_sources: Mapping[str, Path],
                                   repository_root: Path) -> dict[str, str]:
    require(re.fullmatch(r"[0-9a-f]{40}", git_commit) is not None, "git commit must be exact")
    auth = strict_json(auth_path)
    exact_keys(auth, {"stage", "status", "git_commit", "source_sha256"}, "execution source auth")
    require(auth["stage"] == "M33_ASSET_EXECUTION_SOURCE_AUTH", "source-auth stage drift")
    require(auth["status"] == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES", "source-auth did not pass")
    require(auth["git_commit"] == git_commit, "source-auth commit drift")
    hashes = auth["source_sha256"]
    require(set(hashes) == REQUIRED_SOURCES and set(staged_sources) == REQUIRED_SOURCES,
            "execution source-auth inventory incomplete")
    head = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    require(head == git_commit, "Git HEAD differs from authenticated orchestrator commit")
    dirty = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain", "--", *sorted(REQUIRED_SOURCES)],
        check=True, capture_output=True, text=True,
    ).stdout
    require(not dirty.strip(), "authenticated execution sources are dirty or untracked")
    for relative in sorted(REQUIRED_SOURCES):
        valid_sha(hashes[relative], relative)
        require(sha256_file(staged_sources[relative]) == hashes[relative],
                f"staged source changed after authentication: {relative}")
        committed = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"{git_commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        require(hashlib.sha256(committed).hexdigest() == hashes[relative],
                f"commit does not contain authenticated source: {relative}")
    require(hashes["conf/m33_asset_execution_amendment.json"] == AMENDMENT_SHA256,
            "authenticated amendment hash drift")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-contract", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--source-auth", type=Path)
    parser.add_argument("--staged-source", action="append", default=[])
    parser.add_argument("--git-commit")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--nextflow-version")
    parser.add_argument("--allow-real-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(not args.allow_real_assets,
            "real asset access is blocked in this contract-only implementation")
    asset_contract = load_contract(args.asset_contract)
    amendment = load_amendment(args.amendment)
    receipt: dict[str, Any] = {
        "status": PASS_STATUS,
        "asset_contract_sha256": EXACT_CONTRACT_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "execution_authorization": amendment["execution_authorization"],
    }
    auth_options = (args.source_auth, args.git_commit, args.repository_root, args.nextflow_version)
    if any(value is not None for value in auth_options) or args.staged_source:
        require(all(value is not None for value in auth_options),
                "source-auth, commit, repository and Nextflow version are jointly required")
        staged = {}
        for item in args.staged_source:
            relative, separator, path = item.partition("=")
            require(bool(separator) and relative in REQUIRED_SOURCES and relative not in staged,
                    "invalid, duplicate or unknown staged source")
            staged[relative] = Path(path)
        hashes = validate_execution_source_auth(
            args.source_auth, args.git_commit, staged, args.repository_root
        )
        require(set(hashes) == REQUIRED_SOURCES, "execution source-auth inventory drift")
        require(args.nextflow_version == EXPECTED_NEXTFLOW_VERSION, "Nextflow version drift")
        receipt.update({
            "git_commit": args.git_commit,
            "source_auth_sha256": sha256_file(args.source_auth),
            "source_sha256": hashes,
            "nextflow_version": args.nextflow_version,
        })
    if args.output:
        write_exclusive(args.output, receipt)
    else:
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
