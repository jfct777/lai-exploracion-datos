#!/usr/bin/env python3
"""Validate the frozen M34 NAM asset registry and a local M27F role fixture.

The validator is intentionally offline.  It reads one local JSON registry and
one local TSV fixture; it never resolves a GCS object, generates an asset, or
authorizes training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit


EXPECTED_REGISTRY_SHA256 = "0e56c59d6fe6d11be7a6a3476dc435dfed0a7de7888ae4a736a03ed54cfafb2b"
EXPECTED_ANCESTRIES = ["AFR", "EUR", "NAM"]
EXPECTED_ROLES = ["REF_TRAIN", "SOURCE_VALID", "SOURCE_TEST", "DISCOVERY"]
OWNER_ROOT = "gs://teams-usp/frank/lai-exploracion-datos/runs/"
EXPECTED_DERIVED_PREFIX = OWNER_ROOT + "m34-nam-assets-20260826a/"
PASS_STATUS = "PASS_OFFLINE_REGISTRY_AND_LOCAL_ROLE_FIXTURE"

ROOT_KEYS = {
    "schema_version",
    "stage",
    "status",
    "chromosome",
    "genome_build",
    "ancestries",
    "roles",
    "source_assets",
    "allowed_derived_uri_prefix",
    "derived_destinations",
    "relatedness_contract",
    "role_contract",
    "execution_authorization",
}
SOURCE_ASSET_KEYS = {
    "logical_id",
    "gcs_uri",
    "sha256",
    "hash_evidence",
    "requires_live_hash_resolution",
}
HASH_EVIDENCE_KEYS = {"local_path", "manifest_key"}
DESTINATION_KEYS = {
    "logical_id",
    "gcs_uri",
    "sha256",
    "status",
    "requires_live_hash_resolution",
}
EXPECTED_SOURCE_ASSETS = {
    "phased_chr22_vcf": {
        "gcs_uri": "gs://projects-usp/nam-diversity/shapeit/phased/"
        "natwgs.1000G.sgdp.hgdp.andamanese.hg38.22.norm.PHASED.vcf.gz",
        "sha256": "71161973b2e233963321feb03db63eff04f14bc5c7a561a2f0dbce623fc05395",
    },
    "panel_metadata": {
        "gcs_uri": "gs://projects-usp/nam-diversity/"
        "natwgs.1000G.sgdp.hgdp.hg38/metadata_complete_revised.txt",
        "sha256": "d5035d2effa9855999ec684309b9d3d4a3db07184fbb089e73e74957e6fe13d0",
    },
    "m27f_private_split": {
        "gcs_uri": "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/runs/"
        "m27f-blind-split-20260816a/results/27f_blind_split/m27f_split.private.tsv",
        "sha256": "7e4abe2aa57c7375023268e56e3c1dbf9c7bbedc31c3acf1e07b4b07dabfdd07",
    },
    "genetic_map_chr22": {
        "gcs_uri": "gs://projects-usp/dna-do-brasil/"
        "dnabr-lai-gnomix/maps/genetic.map.chr22",
        "sha256": "33c7a94e0cbc0ce3cc3ff83cd3838119a881cb7e838825521a778e07a01ee6e9",
    },
    "pcrelate_retained_people": {
        "gcs_uri": "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/runs/"
        "m27d-pass0-20260815a/27d_donor_kinship_audit/pass0/private/"
        "m27d_pass0_training_set.private.txt",
        "sha256": "bbe3a9b8dc69700e0602e6e3c42cceec5f9564aeea2c7b05b4db17e5e40dfb08",
    },
}
EXPECTED_DESTINATIONS = {
    "resolved_asset_registry",
    "roles_private",
    "mosaic_events",
    "mosaic_haplotypes",
    "mosaic_truth",
    "flare_reference_vcf",
    "flare_reference_tbi",
    "flare_target_vcf",
    "flare_target_tbi",
    "flare_sample_map",
    "training_factors",
    "training_factor_manifest",
}
EXPECTED_COLUMNS = [
    "sample_id",
    "source",
    "ancestry",
    "population",
    "canonical_population",
    "atomic_unit_id",
    "role",
    "exclusion_reason",
]
EXPECTED_ALIASES = {
    "AFR": "AFR",
    "African": "AFR",
    "EUR": "EUR",
    "European": "EUR",
    "NAM": "NAM",
    "Native_American": "NAM",
}
EXPECTED_PEOPLE = {
    "AFR": {"REF_TRAIN": 341, "SOURCE_VALID": 141, "SOURCE_TEST": 135, "DISCOVERY": 0},
    "EUR": {"REF_TRAIN": 387, "SOURCE_VALID": 151, "SOURCE_TEST": 151, "DISCOVERY": 0},
    "NAM": {"REF_TRAIN": 25, "SOURCE_VALID": 14, "SOURCE_TEST": 11, "DISCOVERY": 149},
}
EXPECTED_ATOMIC_UNITS = {
    "AFR": {"REF_TRAIN": 9, "SOURCE_VALID": 5, "SOURCE_TEST": 5, "DISCOVERY": 0},
    "EUR": {"REF_TRAIN": 15, "SOURCE_VALID": 8, "SOURCE_TEST": 8, "DISCOVERY": 0},
    "NAM": {"REF_TRAIN": 4, "SOURCE_VALID": 2, "SOURCE_TEST": 2, "DISCOVERY": 44},
}
EXPECTED_POPULATIONS = {
    "AFR": {"REF_TRAIN": 9, "SOURCE_VALID": 5, "SOURCE_TEST": 5, "DISCOVERY": 0},
    "EUR": {"REF_TRAIN": 15, "SOURCE_VALID": 8, "SOURCE_TEST": 8, "DISCOVERY": 0},
    "NAM": {"REF_TRAIN": 4, "SOURCE_VALID": 2, "SOURCE_TEST": 2, "DISCOVERY": 45},
}


class AssetContractError(ValueError):
    """Raised when the M34 registry or role fixture fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssetContractError(message)


def exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    require(set(value) == expected, f"{where} key inventory drift")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"registry is absent: {path}")

    def reject_constant(token: str) -> None:
        raise AssetContractError(f"non-finite JSON constant: {token}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetContractError(f"registry is not strict UTF-8 JSON: {error}") from error
    require(isinstance(payload, dict), "registry must be a JSON object")
    return payload


def validate_full_sha256(value: Any, where: str) -> None:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{where} must be a complete lowercase SHA-256, never a prefix",
    )


def validate_gcs_object_uri(uri: Any, where: str) -> None:
    require(isinstance(uri, str) and uri, f"{where} URI must be nonempty text")
    require(not any(character.isspace() or ord(character) < 32 for character in uri),
            f"unsafe whitespace/control character in {where} URI")
    require("\\" not in uri and not any(character in uri for character in "*?[]{}"),
            f"glob or template forbidden in {where} URI")
    parsed = urlsplit(uri)
    require(parsed.scheme == "gs" and parsed.netloc, f"{where} must be an exact gs:// URI")
    require(parsed.query == "" and parsed.fragment == "", f"query/fragment forbidden in {where} URI")
    decoded_parts = unquote(parsed.path).split("/")
    require(".." not in decoded_parts, f"path traversal forbidden in {where} URI")
    require(parsed.path not in {"", "/"} and not parsed.path.endswith("/"),
            f"{where} must name an object, not a prefix")


def validate_count_matrix(value: Any, expected: Mapping[str, Mapping[str, int]], where: str) -> None:
    require(isinstance(value, dict), f"{where} must be an object")
    require(list(value) == EXPECTED_ANCESTRIES, f"{where} ancestry order/inventory drift")
    for ancestry in EXPECTED_ANCESTRIES:
        row = value[ancestry]
        require(isinstance(row, dict), f"{where}.{ancestry} must be an object")
        require(list(row) == EXPECTED_ROLES, f"{where}.{ancestry} role order/inventory drift")
        for role in EXPECTED_ROLES:
            count = row[role]
            require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                    f"{where}.{ancestry}.{role} must be a nonnegative integer")
            require(count == expected[ancestry][role],
                    f"{where}.{ancestry}.{role} drift: {count} != {expected[ancestry][role]}")


def validate_registry_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen registry semantics without reading any referenced asset."""
    exact_keys(payload, ROOT_KEYS, "registry")
    require(payload["schema_version"] == "1.0.0", "schema_version drift")
    require(payload["stage"] == "M34_NAM_ASSET_REGISTRY", "stage drift")
    require(
        payload["status"] == "CONTRACT_ONLY_OFFLINE_VALIDATION_NO_GCS_READ_NO_GENERATION_NO_TRAINING",
        "offline-only status drift",
    )
    require(payload["chromosome"] == "22", "only chr22 is frozen")
    require(payload["genome_build"] == "GRCh38", "genome build drift")
    require(payload["ancestries"] == EXPECTED_ANCESTRIES, "ancestries must be AFR/EUR/NAM")
    require(payload["roles"] == EXPECTED_ROLES, "role inventory/order drift")

    sources = payload["source_assets"]
    require(isinstance(sources, dict) and set(sources) == set(EXPECTED_SOURCE_ASSETS),
            "source asset inventory drift")
    for logical_id, expected in EXPECTED_SOURCE_ASSETS.items():
        asset = sources[logical_id]
        require(isinstance(asset, dict), f"source asset {logical_id} must be an object")
        exact_keys(asset, SOURCE_ASSET_KEYS, f"source asset {logical_id}")
        require(asset["logical_id"] == logical_id, f"logical_id mismatch: {logical_id}")
        validate_gcs_object_uri(asset["gcs_uri"], f"source asset {logical_id}")
        require(asset["gcs_uri"] == expected["gcs_uri"], f"source URI drift: {logical_id}")
        validate_full_sha256(asset["sha256"], f"source asset {logical_id}.sha256")
        require(asset["sha256"] == expected["sha256"], f"source SHA-256 drift: {logical_id}")
        require(asset["requires_live_hash_resolution"] is False,
                f"confirmed source hash marked unresolved: {logical_id}")
        evidence = asset["hash_evidence"]
        require(isinstance(evidence, dict), f"hash evidence missing: {logical_id}")
        exact_keys(evidence, HASH_EVIDENCE_KEYS, f"hash evidence {logical_id}")
        require(isinstance(evidence["local_path"], str) and evidence["local_path"].startswith("/"),
                f"hash evidence path must be absolute: {logical_id}")
        require(isinstance(evidence["manifest_key"], str) and evidence["manifest_key"],
                f"manifest key missing: {logical_id}")

    prefix = payload["allowed_derived_uri_prefix"]
    require(prefix == EXPECTED_DERIVED_PREFIX, "derived run prefix drift")
    require(prefix.startswith(OWNER_ROOT) and prefix.endswith("/"),
            "derived run prefix outside the owned runs namespace")
    destinations = payload["derived_destinations"]
    require(isinstance(destinations, dict) and set(destinations) == EXPECTED_DESTINATIONS,
            "derived destination inventory drift")
    seen_uris: set[str] = set()
    for logical_id, destination in destinations.items():
        require(isinstance(destination, dict), f"destination {logical_id} must be an object")
        exact_keys(destination, DESTINATION_KEYS, f"destination {logical_id}")
        require(destination["logical_id"] == logical_id, f"destination logical_id mismatch: {logical_id}")
        uri = destination["gcs_uri"]
        validate_gcs_object_uri(uri, f"destination {logical_id}")
        require(uri.startswith(prefix) and uri.startswith(OWNER_ROOT),
                f"destination outside owned M34 run prefix: {logical_id}")
        require(uri not in seen_uris, f"duplicate destination URI: {uri}")
        seen_uris.add(uri)
        require(destination["sha256"] is None,
                f"planned destination cannot claim an unobserved hash: {logical_id}")
        require(destination["requires_live_hash_resolution"] is True,
                f"planned destination must require live hash resolution: {logical_id}")
        require(destination["status"] == "PLANNED_NOT_GENERATED",
                f"planned destination status drift: {logical_id}")

    relatedness = payload["relatedness_contract"]
    require(relatedness == {
        "source_methods": ["PC-Relate", "Refined IBD"],
        "pcrelate_without_king": True,
        "king_used": False,
        "forbidden_methods": ["KING"],
        "ibd_atomic_unit_definition": "connected_component_of_canonical_population_and_primary_ibd_edges",
        "ibd_max_segment_floor_cm": 10.0,
        "ibd_kinship_floor": 0.04419417382415922,
    }, "relatedness provenance drift")
    require("KING" not in relatedness["source_methods"], "KING cannot be a relatedness source")

    role_contract = payload["role_contract"]
    require(isinstance(role_contract, dict), "role_contract must be an object")
    exact_keys(role_contract, {
        "input_columns",
        "input_ancestry_aliases",
        "expected_people",
        "expected_atomic_units",
        "expected_populations",
        "nam_disjointness_keys",
        "source_test_is_confirmatory",
        "source_test_status",
    }, "role_contract")
    require(role_contract["input_columns"] == EXPECTED_COLUMNS, "M27F fixture schema drift")
    require(role_contract["input_ancestry_aliases"] == EXPECTED_ALIASES,
            "ancestry normalization drift")
    validate_count_matrix(role_contract["expected_people"], EXPECTED_PEOPLE, "expected_people")
    validate_count_matrix(
        role_contract["expected_atomic_units"], EXPECTED_ATOMIC_UNITS, "expected_atomic_units"
    )
    validate_count_matrix(
        role_contract["expected_populations"], EXPECTED_POPULATIONS, "expected_populations"
    )
    require(role_contract["nam_disjointness_keys"] == [
        "sample_id", "canonical_population", "atomic_unit_id"
    ], "NAM disjointness keys drift")
    require(role_contract["source_test_is_confirmatory"] is False,
            "consumed SOURCE_TEST cannot be called confirmatory")
    require(role_contract["source_test_status"] == "CONSUMED_BY_LATER_EXPLORATORY_SPLIT",
            "SOURCE_TEST status drift")

    require(payload["execution_authorization"] == {
        "registry_validation": True,
        "local_fixture_read": True,
        "gcs_read": False,
        "asset_generation": False,
        "training": False,
        "source_test_open": False,
    }, "execution authorization drift")
    return dict(payload)


def validate_registry(path: Path) -> dict[str, Any]:
    require(sha256_file(path) == EXPECTED_REGISTRY_SHA256,
            "immutable M34 asset registry byte hash drift")
    return validate_registry_payload(strict_json(path))


def empty_count_matrix() -> dict[str, dict[str, int]]:
    return {
        ancestry: {role: 0 for role in EXPECTED_ROLES}
        for ancestry in EXPECTED_ANCESTRIES
    }


def require_nam_disjoint(values_by_role: Mapping[str, set[str]], dimension: str) -> None:
    owner: dict[str, str] = {}
    for role in EXPECTED_ROLES:
        for value in values_by_role[role]:
            previous = owner.get(value)
            require(previous is None or previous == role,
                    f"NAM {dimension} crosses roles: {value} in {previous} and {role}")
            owner[value] = role


def validate_roles_fixture(path: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a local M27F-compatible TSV without opening any GCS asset."""
    require(path.is_file(), f"roles fixture is absent: {path}")
    role_contract = registry["role_contract"]
    aliases = role_contract["input_ancestry_aliases"]
    people = empty_count_matrix()
    units = {
        ancestry: {role: set() for role in EXPECTED_ROLES}
        for ancestry in EXPECTED_ANCESTRIES
    }
    populations = {
        ancestry: {role: set() for role in EXPECTED_ROLES}
        for ancestry in EXPECTED_ANCESTRIES
    }
    nam_people = {role: set() for role in EXPECTED_ROLES}
    seen_people: set[str] = set()
    rows_read = 0
    target_rows = 0

    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except (OSError, UnicodeError) as error:
        raise AssetContractError(f"cannot read local roles fixture: {error}") from error
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == role_contract["input_columns"],
                "roles fixture column inventory/order drift")
        for line_number, row in enumerate(reader, start=2):
            rows_read += 1
            require(None not in row, f"extra TSV field at line {line_number}")
            sample_id = row["sample_id"]
            require(sample_id != "", f"empty sample_id at line {line_number}")
            require(sample_id not in seen_people, f"sample_id appears more than once: {sample_id}")
            seen_people.add(sample_id)

            role = row["role"]
            ancestry_label = row["ancestry"]
            if role not in EXPECTED_ROLES:
                require(role == "EXCLUDED", f"unknown role at line {line_number}: {role}")
                require(ancestry_label not in aliases,
                        f"target ancestry cannot be silently excluded at line {line_number}")
                continue

            require(ancestry_label in aliases,
                    f"unknown target ancestry at line {line_number}: {ancestry_label}")
            ancestry = aliases[ancestry_label]
            require(ancestry in EXPECTED_ANCESTRIES, f"ancestry normalization failed at line {line_number}")
            require(row["canonical_population"] != "",
                    f"empty canonical_population at line {line_number}")
            require(row["atomic_unit_id"] != "", f"empty atomic_unit_id at line {line_number}")
            people[ancestry][role] += 1
            units[ancestry][role].add(row["atomic_unit_id"])
            populations[ancestry][role].add(row["canonical_population"])
            target_rows += 1
            if ancestry == "NAM":
                nam_people[role].add(sample_id)

    require(rows_read > 0, "roles fixture is empty")
    require_nam_disjoint(nam_people, "sample_id")
    require_nam_disjoint(populations["NAM"], "canonical_population")
    require_nam_disjoint(units["NAM"], "atomic_unit_id")
    require(people == role_contract["expected_people"], "person counts by ancestry/role drift")
    observed_units = {
        ancestry: {role: len(units[ancestry][role]) for role in EXPECTED_ROLES}
        for ancestry in EXPECTED_ANCESTRIES
    }
    observed_populations = {
        ancestry: {role: len(populations[ancestry][role]) for role in EXPECTED_ROLES}
        for ancestry in EXPECTED_ANCESTRIES
    }
    require(observed_units == role_contract["expected_atomic_units"],
            "atomic-unit counts by ancestry/role drift")
    require(observed_populations == role_contract["expected_populations"],
            "population counts by ancestry/role drift")
    return {
        "rows_read": rows_read,
        "target_rows": target_rows,
        "people": people,
        "atomic_units": observed_units,
        "populations": observed_populations,
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "conf" / "m34_nam_assets.json",
        help="Local frozen registry JSON (default: conf/m34_nam_assets.json).",
    )
    parser.add_argument(
        "--roles-fixture",
        type=Path,
        required=True,
        help="Local M27F-compatible TSV fixture; GCS URIs are never opened.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = validate_registry(args.registry)
    fixture = validate_roles_fixture(args.roles_fixture, registry)
    report = {
        "schema_version": registry["schema_version"],
        "stage": "M34_NAM_ASSET_VALIDATION",
        "status": PASS_STATUS,
        "registry_sha256": sha256_file(args.registry),
        "rows_read": fixture["rows_read"],
        "target_rows": fixture["target_rows"],
        "people": fixture["people"],
        "gcs_read": False,
        "asset_generation": False,
        "training": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
