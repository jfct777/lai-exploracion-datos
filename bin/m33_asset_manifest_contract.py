#!/usr/bin/env python3
"""Freeze and test the fail-closed schema for prospective M33 assets.

This is deliberately not an asset validator yet: it reads no scientific asset,
generates no mosaic, and authorizes neither a forward pass nor training.  It
validates only synthetic manifest fixtures against the future private schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


EXACT_CONTRACT_SHA256 = "3b4ead82305041295b2a997655fe31696eddc15ebbd4c55c0a247f14ea295e4b"
BASE_CONTRACT_SHA256 = "4308bbf33ae28f554f701da33efdc185264f9f407d62661e7048e0345687eb8b"
STATUS = "CONTRACT_ONLY_NO_ASSET_READ_NO_GENERATION_NO_FORWARD_NO_TRAINING"
PASS_STATUS = "PASS_MANIFEST_SCHEMA_FIXTURES_ONLY_NO_ASSET_READ_NO_FORWARD_NO_TRAINING"
EXPECTED_NEXTFLOW_VERSION = "26.04.6"
REQUIRED_SOURCES = {
    "bin/m33_asset_manifest_contract.py",
    "bin/m33_asset_manifest_source_auth.py",
    "conf/m33_asset_manifest_contract.json",
    "conf/m33_asset_manifest_contract.config",
    "modules/33_ASSET_MANIFEST_CONTRACT.nf",
    "workflows/m33_asset_manifest_contract.nf",
    "tests/test_m33_asset_manifest_contract.py",
    "tests/test_m33_asset_manifest_nextflow.py",
}
ROOT_KEYS = {
    "schema_version", "stage", "mode", "root_seed", "root_namespace", "asset_set_id",
    "output_prefix", "creation_precondition", "base_contract_sha256", "generator",
    "assets", "roles", "donor_to_target_haplotypes", "semantic_fingerprints",
    "rare_flare_grid_overlap", "truth_barrier", "private_manifest_sha256",
}
ASSET_FIELDS = {
    "logical_id", "gcs_uri", "gcs_generation", "size_bytes", "sha256_raw", "crc32c",
    "media_type", "compression", "schema_version", "record_count",
}
PERSON_FIELDS = {"person_id", "source_person_sha256", "ancestry", "haplotypes"}
HAPLOTYPE_FIELDS = {"haplotype_id", "canonical_sha256"}
GENERATOR_FIELDS = {
    "repository", "git_commit", "clean_source_auth_sha256", "source_sha256",
    "nextflow_version", "oci_image_digest", "python_version", "stdpopsim_version",
    "msprime_version", "tskit_version", "numpy_version", "root_seed", "rng_streams",
    "rng_seedsequence", "new_tree_sequence_for_root", "generator_receipt_sha256",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_pairs,
        parse_constant=reject_constant,
    )
    require(isinstance(value, dict), "top-level JSON must be an object")
    return value


def exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    require(set(value) == expected, f"{where} key inventory drift")


def valid_sha(value: Any, where: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"invalid sha256: {where}")


def asset_set_id_for(root: int, generator_receipt_sha256: str) -> str:
    valid_sha(generator_receipt_sha256, "generator receipt")
    payload = f"{BASE_CONTRACT_SHA256}|DEVELOPMENT|{root}|{generator_receipt_sha256}"
    return hashlib.sha256(payload.encode()).hexdigest()


def private_manifest_payload_sha256(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "private_manifest_sha256"}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract["schema_version"] == "1.0.0", "schema drift")
    require(contract["stage"] == "M33_ASSET_MANIFEST_CONTRACT", "stage drift")
    require(contract["status"] == STATUS, "contract-only status drift")
    require(contract["base_contract"]["sha256"] == BASE_CONTRACT_SHA256,
            "base contract hash drift")
    require(contract["base_contract"]["git_commit"] ==
            "9f2214f5eaa2c5ab02e5df89528282569cffac4d", "base commit drift")
    roots = contract["root_seeds"]
    forbidden = contract["forbidden_root_seeds"]
    require(roots == [386357765, 2024931463, 1324432253], "DEVELOPMENT roots drift")
    require(len(roots) == len(set(roots)) and set(roots).isdisjoint(forbidden),
            "root registry is not disjoint")
    require(contract["root_is_primary_inference_unit"] is True,
            "persons or markers would be treated as replicates")
    require(contract["creation_precondition"] == "ifGenerationMatch=0",
            "overwrite protection drift")
    rng = contract["rng_contract"]
    require(list(rng["root_streams"]) == [str(root) for root in roots], "RNG root inventory drift")
    full_states = []
    for root in roots:
        root_streams = rng["root_streams"][str(root)]
        require(list(root_streams) == rng["components"], "RNG component order drift")
        for index, component in enumerate(rng["components"]):
            state = root_streams[component]
            require(state["entropy"] == root and state["spawn_key"] == [index], "RNG spawn metadata drift")
            require(len(state["state_uint32x4"]) == 4, "RNG state width drift")
            full_states.append(tuple(state["state_uint32x4"]))
    require(len(full_states) == len(set(full_states)), "RNG component stream collision")
    authorization = contract["execution_authorization"]
    require(authorization == {
        "contract_and_fixture_tests": True,
        "real_asset_read": False,
        "asset_generation": False,
        "no_gradient_forward": False,
        "training": False,
        "EVAL_open": False,
    }, "execution was authorized prematurely")
    require(contract["authorized_output_status"] == PASS_STATUS, "PASS status drift")
    inventory = contract["asset_inventory"]
    require(set(inventory["required"]) == {
        "sites", "targets", "truth", "flare", "map", "roles", "generator_source_auth",
        "generator_manifest", "tree_sequence",
    }, "asset inventory drift")
    require(inventory["shareable_between_roots"] == ["map", "generator_source_auth"],
            "shared asset policy drift")
    require(set(inventory["required_fields"]) == ASSET_FIELDS, "asset fields drift")
    require(contract["role_contract"]["expected_people"] == {
        "FREQ": 300, "REF_LAI": 90, "DONOR": 768, "TARGET": 30,
    }, "role counts drift")
    anchors = contract["generator_anchors"]
    require(anchors["repository"] == "jfct777/lai-exploracion-datos", "repository drift")
    require(set(anchors["required_runtime_fields"]) == GENERATOR_FIELDS - {
        "repository", "source_sha256", "root_seed", "rng_streams", "rng_seedsequence",
        "new_tree_sequence_for_root", "generator_receipt_sha256",
    }, "runtime field policy drift")
    for relative, digest in anchors["source_sha256"].items():
        valid_sha(digest, relative)


def load_contract(path: str | Path) -> dict[str, Any]:
    require(sha256_file(path) == EXACT_CONTRACT_SHA256, "immutable asset contract byte hash drift")
    contract = strict_json(path)
    validate_contract(contract)
    return contract


def validate_gcs_uri(uri: Any, where: str) -> None:
    require(isinstance(uri, str), f"{where} URI must be text")
    require(not any(ord(char) < 32 for char in uri), f"unsafe control character in {where} URI")
    require("\\" not in uri, f"backslash forbidden in {where} URI")
    require(not any(char in uri for char in "%*?[]{}") and not any(char.isspace() for char in uri),
            f"glob, template or whitespace forbidden in {where} URI")
    parsed = urlsplit(uri)
    require(parsed.scheme == "gs" and parsed.netloc == "projects-usp", f"noncanonical {where} URI")
    require(parsed.query == "" and parsed.fragment == "", f"mutable query/fragment in {where} URI")
    decoded = unquote(parsed.path).lower()
    require(decoded.startswith("/dnabr-lai/datalake/"), f"{where} outside canonical datalake")
    require(".." not in decoded.split("/"), f"path traversal in {where} URI")


def expected_prefix(contract: dict[str, Any], root: int, asset_set_id: str) -> str:
    return contract["canonical_prefix_template"].format(
        contract_sha256=BASE_CONTRACT_SHA256,
        root_seed=root,
        asset_set_id=asset_set_id,
    )


def derive_rng_streams(root_seed: int, contract: dict[str, Any]) -> dict[str, int]:
    """Return NumPy-derived streams frozen by the scientific runtime.

    The host-only contract environment intentionally has no NumPy.  These values
    were derived once with the pinned M28 image and are checked against the
    immutable contract instead of silently changing with a local dependency.
    """
    streams = contract["rng_contract"]["root_streams"]
    require(str(root_seed) in streams, "root has no frozen RNG streams")
    value = streams[str(root_seed)]
    require(list(value) == contract["rng_contract"]["components"], "RNG component order drift")
    return {component: record["state_uint32x4"][0] for component, record in value.items()}


def validate_asset(asset: dict[str, Any], logical_id: str, prefix: str,
                   contract: dict[str, Any]) -> None:
    exact_keys(asset, ASSET_FIELDS, f"asset {logical_id}")
    require(asset["logical_id"] == logical_id, f"logical ID mismatch: {logical_id}")
    if logical_id == "map":
        shared = contract["shared_assets"]
        require(asset == {
            "logical_id": "map",
            "gcs_uri": shared["genetic_map_uri"],
            "gcs_generation": shared["genetic_map_gcs_generation"],
            "size_bytes": shared["genetic_map_size_bytes"],
            "sha256_raw": shared["genetic_map_sha256"],
            "crc32c": shared["genetic_map_crc32c"],
            "media_type": shared["genetic_map_media_type"],
            "compression": shared["genetic_map_compression"],
            "schema_version": shared["genetic_map_schema_version"],
            "record_count": shared["genetic_map_record_count"],
        }, "map immutable descriptor drift")
    elif logical_id == "generator_source_auth":
        validate_gcs_uri(asset["gcs_uri"], logical_id)
        require(asset["gcs_uri"].startswith(contract["shared_assets"]["generator_source_auth_prefix"]),
                "generator source auth outside shared prefix")
    else:
        validate_gcs_uri(asset["gcs_uri"], logical_id)
        require(asset["gcs_uri"].startswith(prefix), f"asset outside root prefix: {logical_id}")
    require(isinstance(asset["gcs_generation"], str) and
            re.fullmatch(r"[1-9][0-9]*", asset["gcs_generation"]) is not None,
            f"missing immutable GCS generation: {logical_id}")
    require(isinstance(asset["size_bytes"], int) and asset["size_bytes"] > 0,
            f"empty asset: {logical_id}")
    valid_sha(asset["sha256_raw"], f"asset {logical_id}")
    require(isinstance(asset["crc32c"], str) and
            re.fullmatch(r"[A-Za-z0-9+/]{6}==", asset["crc32c"]) is not None,
            f"invalid crc32c: {logical_id}")
    require(isinstance(asset["media_type"], str) and asset["media_type"],
            f"missing media type: {logical_id}")
    require(asset["compression"] in {"none", "gzip", "bgzip"},
            f"unsupported compression: {logical_id}")
    require(isinstance(asset["schema_version"], str) and asset["schema_version"],
            f"missing asset schema: {logical_id}")
    require(isinstance(asset["record_count"], int) and asset["record_count"] >= 0,
            f"invalid record count: {logical_id}")


def validate_roles(roles: dict[str, Any], root: int, contract: dict[str, Any]) -> dict[str, set[str]]:
    expected_counts = contract["role_contract"]["expected_people"]
    require(set(roles) == set(expected_counts), "role inventory drift")
    seen_people: set[str] = set()
    seen_haplotypes: set[str] = set()
    role_haplotypes: dict[str, set[str]] = {}
    source_people: set[str] = set()
    source_ancestries = set(contract["role_contract"]["allowed_source_ancestries"])
    ancestry_counts = contract["role_contract"]["expected_source_ancestry_counts"]
    for role, count in expected_counts.items():
        people = roles[role]
        require(isinstance(people, list) and len(people) == count, f"wrong {role} person count")
        observed_ancestry: dict[str, int] = {}
        role_haplotypes[role] = set()
        for person in people:
            exact_keys(person, PERSON_FIELDS, f"{role} person")
            person_id = person["person_id"]
            require(isinstance(person_id, str) and
                    person_id.startswith(f"m33:{root}:"), f"non-global person ID in {role}")
            require(person_id not in seen_people, "person appears in multiple roles")
            seen_people.add(person_id)
            ancestry = person["ancestry"]
            allowed = {contract["role_contract"]["target_ancestry_label"]} if role == "TARGET" else source_ancestries
            require(ancestry in allowed, f"invalid ancestry in {role}")
            observed_ancestry[ancestry] = observed_ancestry.get(ancestry, 0) + 1
            haplotypes = person["haplotypes"]
            require(isinstance(haplotypes, list) and len(haplotypes) == 2,
                    f"{role} person is not diploid")
            for haplotype in haplotypes:
                exact_keys(haplotype, HAPLOTYPE_FIELDS, f"{role} haplotype")
                hap_id = haplotype["haplotype_id"]
                require(isinstance(hap_id, str) and hap_id.startswith(person_id + ":h"),
                        f"haplotype not linked to person in {role}")
                require(hap_id not in seen_haplotypes, "haplotype appears in multiple roles")
                seen_haplotypes.add(hap_id)
                role_haplotypes[role].add(hap_id)
                valid_sha(haplotype["canonical_sha256"], f"{role} haplotype")
            if role == "TARGET":
                require(person["source_person_sha256"] is None,
                        "synthetic TARGET cannot claim a source-person fingerprint")
            else:
                valid_sha(person["source_person_sha256"], f"{role} source person")
                expected_person_sha = hashlib.sha256(
                    (ancestry + "|" + "|".join(sorted(
                        hap["canonical_sha256"] for hap in haplotypes
                    ))).encode()
                ).hexdigest()
                require(person["source_person_sha256"] == expected_person_sha,
                        f"{role} source-person fingerprint drift")
                require(person["source_person_sha256"] not in source_people,
                        "source person reused across roles")
                source_people.add(person["source_person_sha256"])
        if role != "TARGET":
            require(observed_ancestry == ancestry_counts[role], f"{role} ancestry counts drift")
    role_haplotypes["_SOURCE_PEOPLE"] = source_people
    return role_haplotypes


def validate_root_manifest(manifest: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    exact_keys(manifest, ROOT_KEYS, "root manifest")
    require(manifest["schema_version"] == "1.0.0", "root manifest schema drift")
    require(manifest["stage"] == "M33_PRIVATE_DEVELOPMENT_ROOT_ASSETS", "root manifest stage drift")
    require(manifest["mode"] == contract["mode"], "root manifest mode drift")
    root = manifest["root_seed"]
    require(root in contract["root_seeds"], "unregistered or forbidden DEVELOPMENT root")
    require(root not in contract["forbidden_root_seeds"], "forbidden root reused")
    namespace = f"m33-development-root-{root}"
    require(manifest["root_namespace"] == namespace, "root namespace drift")
    asset_set_id = manifest["asset_set_id"]
    require(isinstance(asset_set_id, str) and re.fullmatch(r"[0-9a-f]{64}", asset_set_id),
            "asset_set_id must be a content digest")
    require(asset_set_id == asset_set_id_for(root, manifest["generator"]["generator_receipt_sha256"]),
            "asset_set_id provenance drift")
    prefix = expected_prefix(contract, root, asset_set_id)
    require(manifest["output_prefix"] == prefix, "output prefix drift")
    validate_gcs_uri(prefix, "output prefix")
    require(manifest["creation_precondition"] == "ifGenerationMatch=0", "overwrite allowed")
    require(manifest["base_contract_sha256"] == BASE_CONTRACT_SHA256, "base contract drift")

    generator = manifest["generator"]
    exact_keys(generator, GENERATOR_FIELDS, "generator")
    anchors = contract["generator_anchors"]
    require(generator["repository"] == anchors["repository"], "generator repository drift")
    require(generator["source_sha256"] == anchors["source_sha256"], "generator source drift")
    require(generator["git_commit"] == anchors["source_commit"], "generator commit drift")
    valid_sha(generator["clean_source_auth_sha256"], "generator source auth")
    valid_sha(generator["generator_receipt_sha256"], "generator receipt")
    require(generator["root_seed"] == root, "generator root drift")
    expected_streams = derive_rng_streams(root, contract)
    require(generator["rng_streams"] == expected_streams, "RNG derivation drift")
    require(generator["rng_seedsequence"] == contract["rng_contract"]["root_streams"][str(root)],
            "full RNG SeedSequence state drift")
    require(generator["new_tree_sequence_for_root"] is True, "tree was reused")
    require(generator["oci_image_digest"] == anchors["oci_image_digest"], "container digest drift")
    runtime = anchors["runtime"]
    require(generator["nextflow_version"] == runtime["nextflow"], "Nextflow runtime drift")
    require(generator["python_version"] == runtime["python"], "Python runtime drift")
    require(generator["stdpopsim_version"] == runtime["stdpopsim"], "stdpopsim runtime drift")
    require(generator["msprime_version"] == runtime["msprime"], "msprime runtime drift")
    require(generator["tskit_version"] == runtime["tskit"], "tskit runtime drift")
    require(generator["numpy_version"] == runtime["numpy"], "NumPy runtime drift")

    assets = manifest["assets"]
    required_assets = set(contract["asset_inventory"]["required"])
    require(set(assets) == required_assets, "asset inventory missing or extra")
    for logical_id, asset in assets.items():
        validate_asset(asset, logical_id, prefix, contract)
    nonmap_uris = [asset["gcs_uri"] for key, asset in assets.items() if key != "map"]
    require(len(nonmap_uris) == len(set(nonmap_uris)), "two logical assets share one URI")
    require(generator["generator_receipt_sha256"] == assets["generator_manifest"]["sha256_raw"],
            "generator receipt is not bound to its staged asset")
    require(generator["clean_source_auth_sha256"] == assets["generator_source_auth"]["sha256_raw"],
            "generator source auth is not bound to its staged asset")

    role_haps = validate_roles(manifest["roles"], root, contract)
    edges = manifest["donor_to_target_haplotypes"]
    require(isinstance(edges, list) and edges, "TARGET donor provenance is empty")
    linked_targets: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        exact_keys(edge, {"donor_haplotype_id", "target_haplotype_id"}, "donor-target edge")
        pair = (edge["donor_haplotype_id"], edge["target_haplotype_id"])
        require(pair not in seen_edges, "duplicate donor-target edge")
        seen_edges.add(pair)
        require(pair[0] in role_haps["DONOR"], "TARGET references a non-DONOR haplotype")
        require(pair[1] in role_haps["TARGET"], "donor edge references an unknown TARGET")
        linked_targets.add(pair[1])
    require(linked_targets == role_haps["TARGET"], "not every TARGET haplotype has donor provenance")

    fingerprints = manifest["semantic_fingerprints"]
    required_fingerprints = set(contract["semantic_fingerprints"]["required"])
    require(set(fingerprints) == required_fingerprints, "semantic fingerprint inventory drift")
    for key, digest in fingerprints.items():
        valid_sha(digest, key)
    overlap = manifest["rare_flare_grid_overlap"]
    exact_keys(overlap, {"rare_site_count", "flare_grid_site_count", "overlap_site_count",
                         "overlap_fraction_of_rare"}, "rare-FLARE overlap")
    for key in ("rare_site_count", "flare_grid_site_count", "overlap_site_count"):
        require(isinstance(overlap[key], int) and overlap[key] >= 0, f"invalid {key}")
    require(overlap["overlap_site_count"] <= overlap["rare_site_count"], "overlap exceeds rare sites")
    require(overlap["overlap_site_count"] <= overlap["flare_grid_site_count"],
            "overlap exceeds FLARE grid sites")
    expected_fraction = (overlap["overlap_site_count"] / overlap["rare_site_count"]
                         if overlap["rare_site_count"] else 0.0)
    require(isinstance(overlap["overlap_fraction_of_rare"], (int, float)) and
            math.isfinite(overlap["overlap_fraction_of_rare"]) and
            math.isclose(overlap["overlap_fraction_of_rare"], expected_fraction,
                         rel_tol=0.0, abs_tol=1e-12), "rare-FLARE overlap fraction drift")
    require(manifest["truth_barrier"] == {
        "truth_state": "SEALED_PRIVATE_NOT_EXPOSED_TO_PREDICT",
        "predict_view_contains_truth": False,
        "predictor_accepts_truth_argument": False,
    }, "truth barrier drift")
    valid_sha(manifest["private_manifest_sha256"], "private manifest")
    require(manifest["private_manifest_sha256"] == private_manifest_payload_sha256(manifest),
            "private manifest canonical hash drift")
    return {
        "root": root,
        "namespace": namespace,
        "prefix": prefix,
        "people": {person["person_id"] for rows in manifest["roles"].values() for person in rows},
        "haplotypes": {hap["haplotype_id"] for rows in manifest["roles"].values()
                       for person in rows for hap in person["haplotypes"]},
        "founders": {hap["canonical_sha256"] for role in ("FREQ", "REF_LAI", "DONOR")
                     for person in manifest["roles"][role] for hap in person["haplotypes"]},
        "source_people": role_haps["_SOURCE_PEOPLE"],
        "rng_streams": {tuple(record["state_uint32x4"])
                        for record in generator["rng_seedsequence"].values()},
        "tree_raw": assets["tree_sequence"]["sha256_raw"],
        "tree_semantic": fingerprints["normalized_tree_tables_sha256"],
        "asset_raw": {key: (value["gcs_uri"], value["gcs_generation"], value["sha256_raw"])
                      for key, value in assets.items()},
        "private_manifest_sha256": manifest["private_manifest_sha256"],
        "generator_receipt_sha256": generator["generator_receipt_sha256"],
        "counts": {role: len(rows) for role, rows in manifest["roles"].items()},
    }


def _prefixes_overlap(left: str, right: str) -> bool:
    return left.startswith(right) or right.startswith(left)


def validate_bundle(manifests: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    require(len(manifests) == len(contract["root_seeds"]), "bundle must contain exactly three roots")
    rows = [validate_root_manifest(manifest, contract) for manifest in manifests]
    require([row["root"] for row in rows] == contract["root_seeds"],
            "roots must be complete, unique and in frozen order")
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            require(left["namespace"] != right["namespace"], "root namespace reused")
            require(not _prefixes_overlap(left["prefix"], right["prefix"]), "root prefixes overlap")
            require(left["people"].isdisjoint(right["people"]), "person reused across roots")
            require(left["haplotypes"].isdisjoint(right["haplotypes"]), "haplotype reused across roots")
            require(left["source_people"].isdisjoint(right["source_people"]),
                    "source person reused across roots")
            require(left["founders"].isdisjoint(right["founders"]), "founder reused across roots")
            require(left["rng_streams"].isdisjoint(right["rng_streams"]), "RNG stream reused")
            require(left["tree_raw"] != right["tree_raw"], "tree byte hash reused")
            require(left["tree_semantic"] != right["tree_semantic"], "tree genealogy reused")
            for logical_id in set(contract["asset_inventory"]["required"]) - set(
                contract["asset_inventory"]["shareable_between_roots"]
            ):
                left_asset = left["asset_raw"][logical_id]
                right_asset = right["asset_raw"][logical_id]
                require(left_asset[0] != right_asset[0], f"non-shareable URI reused: {logical_id}")
                require(left_asset[2] != right_asset[2], f"non-shareable asset reused: {logical_id}")
    for logical_id in contract["asset_inventory"]["shareable_between_roots"]:
        shared = [manifest["assets"][logical_id] for manifest in manifests]
        require(all(value == shared[0] for value in shared[1:]),
                f"shared descriptor drift across roots: {logical_id}")
    return {
        "status": PASS_STATUS,
        "base_contract_sha256": BASE_CONTRACT_SHA256,
        "asset_contract_sha256": EXACT_CONTRACT_SHA256,
        "root_seeds": [row["root"] for row in rows],
        "aggregate_counts": {role: sum(row["counts"][role] for row in rows)
                             for role in contract["role_contract"]["expected_people"]},
        "private_manifest_sha256": [row["private_manifest_sha256"] for row in rows],
        "generator_receipt_sha256": [row["generator_receipt_sha256"] for row in rows],
        "failure_reasons": [],
    }


def assert_predict_receipt_redacted(receipt: dict[str, Any], contract: dict[str, Any]) -> None:
    require(set(receipt) == set(contract["privacy"]["public_receipt_contains_only"]),
            "predict receipt key inventory drift")
    serialized = json.dumps(receipt, sort_keys=True).lower()
    require("gs://" not in serialized, "predict receipt contains a GCS URI")
    for forbidden in contract["privacy"]["predict_receipt_forbids"]:
        require(forbidden.lower() not in serialized, f"predict receipt leaks {forbidden}")


def validate_source_auth(auth_path: Path, git_commit: str, staged_sources: dict[str, Path],
                         repository_root: Path) -> dict[str, str]:
    require(re.fullmatch(r"[0-9a-f]{40}", git_commit) is not None, "git commit must be exact")
    auth = strict_json(auth_path)
    exact_keys(auth, {"stage", "status", "git_commit", "source_sha256"}, "source auth")
    require(auth["stage"] == "M33_ASSET_MANIFEST_SOURCE_AUTH", "source-auth stage drift")
    require(auth["status"] == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES", "source-auth did not pass")
    require(auth["git_commit"] == git_commit, "source-auth commit drift")
    hashes = auth["source_sha256"]
    require(isinstance(hashes, dict) and set(hashes) == REQUIRED_SOURCES,
            "source-auth inventory incomplete")
    require(set(staged_sources) == REQUIRED_SOURCES, "staged source inventory incomplete")
    head = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(head == git_commit, "Git HEAD differs from authenticated commit")
    dirty = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain", "--", *sorted(REQUIRED_SOURCES)],
        check=True, capture_output=True, text=True,
    ).stdout
    require(not dirty.strip(), "authenticated sources are dirty or untracked")
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
    require(hashes["conf/m33_asset_manifest_contract.json"] == EXACT_CONTRACT_SHA256,
            "authenticated asset contract hash drift")
    return hashes


def parse_staged(items: list[str]) -> dict[str, Path]:
    staged: dict[str, Path] = {}
    for item in items:
        relative, separator, path = item.partition("=")
        require(bool(separator) and relative in REQUIRED_SOURCES and relative not in staged,
                "invalid, duplicate or unknown staged source")
        staged[relative] = Path(path)
    return staged


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-auth", type=Path)
    parser.add_argument("--staged-source", action="append", default=[])
    parser.add_argument("--git-commit")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--nextflow-version")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    receipt = {
        "status": "PASS_ASSET_MANIFEST_CONTRACT_ONLY",
        "asset_contract_sha256": EXACT_CONTRACT_SHA256,
        "execution_authorization": contract["execution_authorization"],
    }
    auth_options = (args.source_auth, args.git_commit, args.repository_root, args.nextflow_version)
    if any(value is not None for value in auth_options) or args.staged_source:
        require(all(value is not None for value in auth_options),
                "source-auth, commit, repository and Nextflow version are jointly required")
        hashes = validate_source_auth(
            args.source_auth, args.git_commit, parse_staged(args.staged_source), args.repository_root
        )
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
