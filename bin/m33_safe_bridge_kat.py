#!/usr/bin/env python3
"""Run the non-consumable synthetic SAFE_BRIDGE known-answer transform."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import m33_safe_bridge_core as bridge_core

from m33_safe_bridge_core import (
    ANCESTRIES, bind_genetic_map, canonical_loci, orient_target, parse_f0_records,
    parse_reference_state_records, parse_target_state_records, partition_incremental, reopen_npz,
    sample_key, semantic_arrays_sha256, strict_integer_array, summarize_reference,
    write_deterministic_npz, write_exclusive_json,
)


STAGE = "M33_SAFE_BRIDGE_KAT"
STATUS = "PASS_SAFE_BRIDGE_KAT_TECHNICAL_ONLY_NON_CONSUMABLE"
EXPECTED_CONTRACT_SHA256 = "1c6b3d5679d8f64902f3adb7ae40319874ef88bf32ba215402352b6f6a196b25"
EXPECTED_ROOTS = {
    "root17": {
        "root_seed": 20260817,
        "a0_receipt_sha256": "4fe79f0dc648caa2c64b9d81c752e144af6d0c9db43e3fa3ae4274887ecfff93",
        "a0_registry_sha256": "44311fe8ef9238c81f630343857439ac16e52b7569c1348a52fb65d744ad93cd",
        "flare_vcf_sha256": "85dfd76df2c14cb8fe0a753910f25c49c88d38edc5708ec6d641053d95cc74e8",
        "flare_vcf_generation": "1787175566795248",
        "flare_tbi_sha256": "936a5d8256a2f1610898e2a3fe00523001c203cb014a7509c9c76911fe626e52",
        "i0_receipt_sha256": "53425116ccf0e3d9d8df7b4eaab1e248efe425d0fd0d22041bbc330cbcbb3924",
    },
    "root18": {
        "root_seed": 20260818,
        "a0_receipt_sha256": "de1ea4ae2b67179acb9a6f56024bfe9bbb92b00663ef3684b280d077f1726b4b",
        "a0_registry_sha256": "649993d6e098b3cf92260a95d5bfcf8a89a529f8438f753dce368598958773de",
        "flare_vcf_sha256": "edc4bcdc62f5ce0ffe04bd27e9d6d6ee892e03282a1474639fc3082fbc3832c9",
        "flare_vcf_generation": "1787175916753131",
        "flare_tbi_sha256": "9f26af28898adbdcc8eec8ebddc10b321464923c8b244719e7c3b7a6d0dfacaf",
        "i0_receipt_sha256": "c9a0a5bdc74ade07cb98cfadbfadf0ed9e274adca0c18cd4250de6fffce66984",
    },
}
EXPECTED_I0_FACTS = {
    "status": "PASS_I0_TECHNICAL_ONLY",
    "tabix_version": "1.16",
    "tabix_oci": "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:e730c35759e3851a92d7f3a6619333105331d97f1ae44a50dfa8d59745c43e54",
    "record_count_per_root": 79791,
    "publication_prefix": "gs://teams-usp/frank/lai-exploracion-datos/runs/m33-i0-real-20260822a/",
    "publication_receipt_generation": "1787439897641803",
    "publication_receipt_sha256": "024773196a33e4f70cebf3971e1a3550ef5caaf29e24f6d5cadb4a69419b9733",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_strict(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in out, f"duplicate JSON key: {key}")
            out[key] = value
        return out

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"nonfinite JSON value: {value}")), object_pairs_hook=unique)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract.get("stage") == "M33_SAFE_BRIDGE_KAT_CONTRACT" and
            contract.get("status") ==
            "DESIGN_ONLY_KAT_NO_REAL_ASSET_READ_NO_MATERIALIZE_NO_READY_NO_TRAINING",
            "SAFE_BRIDGE KAT contract identity drifted")
    require(contract.get("base_contract_sha256") ==
            "fb74cd610a36b22fe54b8681238a13b48a0243642c9a24cd036597d161361614",
            "base contract anchor drifted")
    require(contract.get("effective_i0_facts") == EXPECTED_I0_FACTS, "effective I0 facts drifted")
    require(contract.get("technical_roots") == EXPECTED_ROOTS, "technical root bundle drifted or swapped")
    require(contract.get("kat_profiles") == {
        "minor0_overlap_missing_firewall": [
            "minor_code_zero", "nonzero_flare_overlap",
            "target_and_reference_missingness", "role_firewall_poison",
        ]
    }, "KAT profile coverage drifted")
    authorization = contract.get("execution_authorization", {})
    require(authorization == {
        "contract_validation": True, "synthetic_kat": True, "real_asset_read": False,
        "technical_root_kat": False, "materialize": False, "write_ready": False,
        "training": False, "truth_scoring": False, "gcs_write": False,
    }, "KAT authorization drifted")
    bridge = contract.get("effective_safe_bridge", {})
    require(bridge.get("truth_mounted") is False and bridge.get("network_enabled") is False and
            bridge.get("staged_inputs_not_modified_by_runner") is True and
            bridge.get("physical_read_only_mount_proven") is False and bridge.get("role_firewall") ==
            "contributing_nodes_exactly_equal_authenticated_REF_nodes" and
            bridge.get("minor_code_binding") == "exact_locus_id_from_authenticated_rare_catalog",
            "SAFE_BRIDGE isolation or role firewall drifted")
    f0 = bridge.get("flare_f0_sanitized", {})
    require(f0.get("allowed_source_fields") == ["ANP1", "ANP2"] and
            f0.get("required_identity_fields") ==
            ["root_seed", "sample_id", "chrom", "pos", "ref", "alt"] and
            f0.get("forbidden_source_fields") ==
            ["GT", "AN1", "AN2", "truth", "hard_call", "target_rare_phase"] and
            f0.get("output_axes") == ["sample", "haplotype", "marker", "ancestry"],
            "sanitized F0 boundary drifted")
    resolution = bridge.get("raw_boundary_resolution", {})
    require(resolution.get("decision") ==
            "SAFE_BRIDGE_parses_only_ANP1_ANP2_and_emits_flare_f0_sanitized" and
            resolution.get("effective_now") is False and
            resolution.get("requires_separate_materialize_amendment") is True,
            "raw FLARE boundary resolution drifted")


def locus_arrays(rows: list[tuple[int, int, str, str, int, float]]) -> dict[str, np.ndarray]:
    return {
        "locus_id": np.asarray([row[4] for row in rows], dtype="<u8"),
        "chrom": np.asarray([row[0] for row in rows], dtype="|u1"),
        "pos": np.asarray([row[1] for row in rows], dtype="<i8"),
        "ref": np.asarray([row[2].encode("ascii") for row in rows], dtype="|S1"),
        "alt": np.asarray([row[3].encode("ascii") for row in rows], dtype="|S1"),
        "cM": np.asarray([row[5] for row in rows], dtype="<f8"),
    }


def run(fixture_path: Path, contract_path: Path, base_contract_path: Path,
        output_dir: Path) -> dict[str, Any]:
    for label, path in (("fixture", fixture_path), ("contract", contract_path),
                        ("base contract", base_contract_path)):
        require(path.is_file() and not path.is_symlink(), f"{label} must be a regular non-symlink file")
    require(not output_dir.exists() and output_dir.parent.is_dir() and not output_dir.parent.is_symlink(),
            "output directory must be new under a regular existing parent")
    contract = load_strict(contract_path)
    require(sha256_file(contract_path) == EXPECTED_CONTRACT_SHA256,
            "SAFE_BRIDGE KAT contract raw hash drifted")
    validate_contract(contract)
    base = load_strict(base_contract_path)
    require(sha256_file(base_contract_path) == contract["base_contract_sha256"],
            "base M0 contract hash drifted")
    assertions = contract["base_assertions"]
    require(base["process_contracts"]["I0_DERIVE_AUTHENTICATE_FLARE_INDEX"]["implemented"] is
            assertions["i0_implemented"], "base I0 assertion drifted")
    require(base["process_contracts"]["SAFE_BRIDGE"]["implemented"] is
            assertions["safe_bridge_implemented"], "base SAFE_BRIDGE assertion drifted")
    require(base["execution_authorization"]["safe_bridge"] is assertions["safe_bridge_authorized"] and
            base["execution_authorization"]["materialize"] is assertions["materialize_authorized"],
            "base execution assertion drifted")
    fixture = load_strict(fixture_path)
    require(set(fixture) == {"profile", "root_seed", "selected_loci", "rare_catalog_records",
                             "flare_loci", "genetic_map_records", "target_sample_ids",
                             "target_haplotype_records", "expected_ref_records",
                             "ref_state_records", "f0_records"},
            "fixture keys differ")
    require(fixture["profile"] == "minor0_overlap_missing_firewall", "fixture profile differs")
    require(type(fixture["root_seed"]) is int and fixture["root_seed"] >= 0, "fixture root seed is invalid")

    require(isinstance(fixture["rare_catalog_records"], list), "rare catalog records are invalid")
    catalog_loci: list[dict[str, Any]] = []
    minor_by_id: dict[int, int] = {}
    for record in fixture["rare_catalog_records"]:
        require(isinstance(record, dict) and set(record) ==
                {"chrom", "pos", "ref", "alt", "locus_id", "cM", "minor_code"},
                "rare catalog record fields are invalid")
        catalog_loci.append({key: record[key] for key in
                             ("chrom", "pos", "ref", "alt", "locus_id", "cM")})
        require(type(record["locus_id"]) is int, "rare catalog locus_id is invalid")
        code = strict_integer_array([record["minor_code"]], name="minor code", allowed={0, 1})
        require(record["locus_id"] not in minor_by_id, "rare catalog locus_id is duplicated")
        minor_by_id[record["locus_id"]] = int(code[0])

    selected_all = bind_genetic_map(fixture["selected_loci"], fixture["genetic_map_records"])
    bind_genetic_map(catalog_loci, fixture["genetic_map_records"])
    bind_genetic_map(fixture["flare_loci"], fixture["genetic_map_records"])
    incremental, overlap = partition_incremental(fixture["selected_loci"], catalog_loci,
                                                 fixture["flare_loci"])
    require(set(minor_by_id) == {row[4] for row in selected_all},
            "minor-code records do not match the authenticated selected loci")
    minor_codes_all = np.asarray([minor_by_id[row[4]] for row in selected_all], dtype=np.int8)
    index = {(row[0], row[1], row[2], row[3]): i for i, row in enumerate(selected_all)}
    keep = np.asarray([index[(row[0], row[1], row[2], row[3])] for row in incremental], dtype=np.int64)
    minor_codes = minor_codes_all[keep]

    raw_target = parse_target_state_records(fixture["target_haplotype_records"],
                                            fixture["target_sample_ids"], selected_all)
    target_dosage, target_observed = orient_target(raw_target[:, keep, :], minor_codes)
    sample_keys = np.asarray([sample_key(value) for value in fixture["target_sample_ids"]], dtype="|S64")
    require(len(sample_keys) == raw_target.shape[0] and len(set(sample_keys.tolist())) == len(sample_keys),
            "TARGET sample axis is invalid")

    raw_ref, ref_node_ids, ref_person_ids, ref_ancestries = parse_reference_state_records(
        fixture["ref_state_records"], fixture["expected_ref_records"], selected_all)
    reference = summarize_reference(raw_ref[:, keep], minor_codes,
                                    ref_node_ids, ref_person_ids, ref_ancestries,
                                    fixture["expected_ref_records"])
    f0 = parse_f0_records(fixture["f0_records"], fixture["target_sample_ids"],
                          canonical_loci(fixture["flare_loci"]), fixture["root_seed"])
    require(f0.shape[0] == len(sample_keys), "F0 sample axis differs")
    flare_loci = canonical_loci(fixture["flare_loci"])
    require(f0.shape[2] == len(flare_loci), "F0 marker axis differs from FLARE loci")

    selected_arrays = locus_arrays(incremental)
    target_arrays = {
        "sample_key_sha256": sample_keys,
        "locus_id": selected_arrays["locus_id"],
        "minor_dosage": target_dosage,
        "observed_mask": target_observed,
    }
    reference_arrays = {
        "ancestry": np.arange(3, dtype="|u1"),
        "locus_id": selected_arrays["locus_id"],
        **reference,
    }
    f0_arrays = {
        "sample_key_sha256": sample_keys,
        "chrom": np.asarray([row[0] for row in flare_loci], dtype="|u1"),
        "pos": np.asarray([row[1] for row in flare_loci], dtype="<i8"),
        "ref": np.asarray([row[2].encode("ascii") for row in flare_loci], dtype="|S1"),
        "alt": np.asarray([row[3].encode("ascii") for row in flare_loci], dtype="|S1"),
        "probabilities": f0,
    }
    outputs = {
        "kat_selected_loci_incremental.npz": ("tests_m33_m0_selected_loci_incremental_v1", selected_arrays),
        "kat_target_rare_diploid_incremental.npz":
            ("tests_m33_m0_target_rare_diploid_incremental_v1", target_arrays),
        "kat_reference_rare_summary_incremental.npz":
            ("tests_m33_m0_reference_rare_summary_incremental_v1", reference_arrays),
        "kat_flare_f0_sanitized.npz":
            ("tests_m33_safe_bridge_flare_f0_sanitized_v1", f0_arrays),
    }
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    artifact_semantics: dict[str, str] = {}
    raw_sha256: dict[str, str] = {}
    for name, (schema, arrays) in outputs.items():
        path = output_dir / name
        write_deterministic_npz(path, arrays)
        reopen_npz(path, arrays)
        artifact_semantics[name] = semantic_arrays_sha256(schema, arrays)
        raw_sha256[name] = sha256_file(path)

    expected_records = sorted((int(row["node_id"]), str(row["person_id"]), str(row["ancestry"]))
                              for row in fixture["expected_ref_records"])
    contributing_records = sorted(zip(ref_node_ids, ref_person_ids, ref_ancestries))
    expected_node_hash = hashlib.sha256(json.dumps(expected_records, separators=(",", ":")).encode()).hexdigest()
    contributing_node_hash = hashlib.sha256(json.dumps(contributing_records, separators=(",", ":")).encode()).hexdigest()
    receipt = {
        "stage": STAGE,
        "schema_id": "m33_safe_bridge_kat_receipt_v1",
        "status": STATUS,
        "profile": fixture["profile"],
        "root_seed": fixture["root_seed"],
        "scientific_evidence": False,
        "consumable": False,
        "ready_emitted": False,
        "truth_read": False,
        "selected_all_count": len(selected_all),
        "selected_incremental_count": len(incremental),
        "selected_overlap_count": len(overlap),
        "partition_disjoint_union_pass": len(selected_all) == len(incremental) + len(overlap),
        "incremental_minor_code_0_locus_count": int((minor_codes == 0).sum()),
        "incremental_minor_code_1_locus_count": int((minor_codes == 1).sum()),
        "target_missing_cells": int((target_observed == 0).sum()),
        "reference_missing_alleles": int((raw_ref[:, keep] < 0).sum()),
        "expected_ref_nodes_semantic_sha256": expected_node_hash,
        "contributing_ref_nodes_semantic_sha256": contributing_node_hash,
        "expected_ref_node_count": len(expected_records),
        "contributing_ref_node_count": len(contributing_records),
        "rejected_non_ref_node_count": 0,
        "role_firewall_pass": True,
        "ancestry_order": list(ANCESTRIES),
        "artifact_semantic_sha256": artifact_semantics,
        "artifact_raw_sha256": raw_sha256,
        "contract_sha256": sha256_file(contract_path),
        "base_contract_sha256": sha256_file(base_contract_path),
        "fixture_sha256": sha256_file(fixture_path),
        "source_sha256": {
            "m33_safe_bridge_core.py": sha256_file(Path(bridge_core.__file__)),
            "m33_safe_bridge_kat.py": sha256_file(Path(__file__)),
        },
        "reopen_verified": True,
        "append_only": True,
    }
    receipt_path = output_dir / "safe_bridge_kat.receipt.json"
    write_exclusive_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--base-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.fixture, args.contract, args.base_contract, args.output_dir), sort_keys=True))
