#!/usr/bin/env python3
"""Fail-closed M33 I0 index derivation for consumed technical roots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_IMAGE = (
    "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix@sha256:"
    "e730c35759e3851a92d7f3a6619333105331d97f1ae44a50dfa8d59745c43e54"
)
EXPECTED_VERSION = "tabix (htslib) 1.16"
EXPECTED_CONTRACT_SHA256 = "fb74cd610a36b22fe54b8681238a13b48a0243642c9a24cd036597d161361614"
EXPECTED_RECORD_COUNT = 79791
EXPECTED_ROOTS = {
    "root17": {
        "root_seed": 20260817,
        "uri": "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/runs/"
        "m30-flare-baseline-20260819b/30_flare_baseline/root17/flare/root17.flare.anc.vcf.gz",
        "generation": "1787175566795248",
        "size_bytes": 2297753,
        "sha256": "85dfd76df2c14cb8fe0a753910f25c49c88d38edc5708ec6d641053d95cc74e8",
        "crc32c_base64": "QFwzMA==",
        "md5_base64": "8I0LJOoui+5VwVys588tFA==",
    },
    "root18": {
        "root_seed": 20260818,
        "uri": "gs://projects-usp/dnaBr-lai/datalake/refined/DNABR_QC/runs/"
        "m30-flare-baseline-20260819b/30_flare_baseline/root18/flare/root18.flare.anc.vcf.gz",
        "generation": "1787175916753131",
        "size_bytes": 2331603,
        "sha256": "edc4bcdc62f5ce0ffe04bd27e9d6d6ee892e03282a1474639fc3082fbc3832c9",
        "crc32c_base64": "aw/Whw==",
        "md5_base64": "8Yki4megECkiIFX9Erk8SA==",
    },
}
SOURCE_FILES = {
    "bin/m33_i0_index.py",
    "bin/m33_i0_real.py",
    "conf/m33_i0_real.config",
    "conf/m33_i0_real_authorization.json",
    "conf/m33_m0_materializer_contract.json",
    "modules/33_I0_REAL.nf",
    "tests/test_m33_i0_real.py",
    "tests/test_m33_i0_real_integration.py",
    "tests/test_m33_i0_real_nextflow.py",
    "workflows/m33_i0_real.nf",
}
FORBIDDEN_CREDENTIAL_ENV = {
    "CLOUDSDK_AUTH_ACCESS_TOKEN",
    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
    "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT",
    "CLOUDSDK_CONFIG",
    "CLOUDSDK_CORE_ACCOUNT",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_IMPERSONATE_SERVICE_ACCOUNT",
    "GOOGLE_OAUTH_ACCESS_TOKEN",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=no_duplicate_pairs)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixture_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("m33_i0_index_helpers", path)
    require(spec is not None and spec.loader is not None, "cannot load frozen I0 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_contract(contract: Path) -> str:
    observed = sha256_file(contract)
    require(observed == EXPECTED_CONTRACT_SHA256, "base M0 contract hash differs")
    payload = load_json(contract)
    i0 = payload["process_contracts"]["I0_DERIVE_AUTHENTICATE_FLARE_INDEX"]
    require(i0["implemented"] is False, "base contract was modified retroactively")
    require(i0["status"] == "BLOCKED_PENDING_PULLABLE_TABIX_OCI", "base I0 status differs")
    return observed


def load_authorization(path: Path, contract: Path) -> dict[str, Any]:
    payload = load_json(path)
    require(set(payload) == {
        "schema_version", "stage", "status", "base_m0_contract_sha256", "fixture_commit",
        "tabix_image", "tabix_version", "chromosome", "expected_record_count",
        "download_principal", "roots", "execution", "local_output_policy",
    }, "I0 real authorization keys differ")
    require(payload["schema_version"] == "1.0.0", "authorization schema differs")
    require(payload["stage"] == "M33_I0_REAL_AUTHORIZATION", "authorization stage differs")
    require(payload["status"] == "AUTHORIZED_EXACT_CONSUMED_TECHNICAL_ROOTS_LOCAL_ONLY",
            "authorization status differs")
    require(validate_contract(contract) == payload["base_m0_contract_sha256"],
            "authorization is not tied to the base contract")
    require(payload["tabix_image"] == EXPECTED_IMAGE, "Tabix image differs")
    require(payload["tabix_version"] == "1.16", "Tabix version authorization differs")
    require(payload["chromosome"] == "22", "chromosome differs")
    require(payload["expected_record_count"] == EXPECTED_RECORD_COUNT, "record count differs")
    require(payload["download_principal"] == "jcalderonta@ime.usp.br", "download principal differs")
    require(payload["roots"] == EXPECTED_ROOTS, "root descriptors differ")
    require(payload["execution"] == {
        "real_asset_read": True,
        "derive_index": True,
        "local_nextflow_only": True,
        "root_tasks_parallel": True,
        "independent_builds_per_root": 2,
        "cache": False,
        "retries": 0,
        "container_network": False,
        "container_credentials": False,
        "safe_bridge": False,
        "materialize": False,
        "forward": False,
        "backward": False,
        "training": False,
        "truth": False,
        "test": False,
        "global_ready": False,
    }, "execution boundary differs")
    policy = payload["local_output_policy"]
    require(policy == {
        "append_only": True,
        "mode": "0444",
        "summary_only_receipts": True,
        "publish_gcs_during_workflow": False,
        "allowed_future_prefix": "gs://teams-usp/frank/lai-exploracion-datos/runs/",
        "forbidden_source_prefix": "gs://projects-usp/",
        "completion_marker": "I0_REAL_PASS_NON_CONSUMABLE",
    }, "output policy differs")
    return payload


def load_source_auth(path: Path, repo_root: Path) -> str:
    payload = load_json(path)
    require(set(payload) == {"schema_version", "stage", "status", "files"},
            "source-auth keys differ")
    require(payload["schema_version"] == "1.0.0", "source-auth schema differs")
    require(payload["stage"] == "M33_I0_REAL_SOURCE_AUTH", "source-auth stage differs")
    require(payload["status"] == "AUTHORIZED_EXACT_I0_REAL_SOURCES", "source-auth status differs")
    require(set(payload["files"]) == SOURCE_FILES, "source-auth inventory differs")
    for relative, expected in payload["files"].items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"invalid source hash: {relative}")
        require(sha256_file(repo_root / relative) == expected, f"source hash differs: {relative}")
    return sha256_file(path)


def load_runtime_source_auth(
    path: Path, *, real_script: Path, helper_script: Path,
    authorization: Path, contract: Path,
) -> str:
    """Verify the exact authenticated subset staged inside the offline container."""
    payload = load_json(path)
    require(set(payload) == {"schema_version", "stage", "status", "files"},
            "source-auth keys differ")
    require(payload["schema_version"] == "1.0.0", "source-auth schema differs")
    require(payload["stage"] == "M33_I0_REAL_SOURCE_AUTH", "source-auth stage differs")
    require(payload["status"] == "AUTHORIZED_EXACT_I0_REAL_SOURCES", "source-auth status differs")
    require(set(payload["files"]) == SOURCE_FILES, "source-auth inventory differs")
    runtime_files = {
        "bin/m33_i0_real.py": real_script,
        "bin/m33_i0_index.py": helper_script,
        "conf/m33_i0_real_authorization.json": authorization,
        "conf/m33_m0_materializer_contract.json": contract,
    }
    for relative, staged_path in runtime_files.items():
        expected = payload["files"][relative]
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                f"invalid source hash: {relative}")
        require(sha256_file(staged_path) == expected, f"runtime source hash differs: {relative}")
    return sha256_file(path)


def validate_run_id(run_id: str) -> None:
    require(re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", run_id) is not None, "invalid run ID")


def active_gcloud_account() -> str:
    for variable in FORBIDDEN_CREDENTIAL_ENV:
        require(not os.environ.get(variable), f"credential override is forbidden: {variable}")
    completed = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        check=True, capture_output=True, text=True,
    )
    accounts = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    require(len(accounts) == 1, "exactly one active gcloud account is required")
    return accounts[0]


def require_no_gcloud_impersonation() -> None:
    completed = subprocess.run(
        ["gcloud", "config", "get-value", "auth/impersonate_service_account"],
        check=True, capture_output=True, text=True,
    )
    value = completed.stdout.strip()
    require(value in {"", "(unset)"}, "gcloud service-account impersonation is forbidden")


def local_gcloud_hashes(path: Path) -> tuple[str, str]:
    completed = subprocess.run(
        ["gcloud", "storage", "hash", str(path), "--format=json"],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    require(isinstance(payload, list) and len(payload) == 1, "unexpected gcloud hash result")
    return payload[0]["crc32c_hash"], payload[0]["md5_hash"]


def stage_source(
    *, root_label: str, output_vcf: Path, receipt: Path, authorization: Path,
    contract: Path, source_auth: Path, helper_script: Path, repo_root: Path, run_id: str,
) -> dict[str, Any]:
    validate_run_id(run_id)
    auth = load_authorization(authorization, contract)
    full_source_auth_sha = load_source_auth(source_auth, repo_root)
    runtime_source_auth_sha = load_runtime_source_auth(
        source_auth,
        real_script=Path(__file__).resolve(),
        helper_script=helper_script,
        authorization=authorization,
        contract=contract,
    )
    require(full_source_auth_sha == runtime_source_auth_sha, "source-auth verification path differs")
    source_auth_sha = full_source_auth_sha
    helpers = load_fixture_helpers(helper_script)
    require(root_label in EXPECTED_ROOTS, "unauthorized root")
    descriptor = auth["roots"][root_label]
    require(output_vcf.name == f"{root_label}.flare.anc.vcf.gz", "staged VCF basename differs")
    require(receipt.name == f"{root_label}.source.receipt.json", "source receipt basename differs")
    require(output_vcf.parent.absolute() == receipt.parent.absolute(), "stage outputs must share directory")
    require(not output_vcf.exists() and not receipt.exists(), "stage output already exists")
    require_no_gcloud_impersonation()
    account = active_gcloud_account()
    require(account == auth["download_principal"], "active gcloud account is not authorized")

    descriptor_fd, descriptor_name = tempfile.mkstemp(prefix=f".{root_label}.", dir=output_vcf.parent)
    os.close(descriptor_fd)
    temporary = Path(descriptor_name)
    temporary.unlink()
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", f"{descriptor['uri']}#{descriptor['generation']}", str(temporary)],
            check=True,
        )
        require(temporary.is_file(), "gcloud did not stage the source")
        require(temporary.stat().st_size == descriptor["size_bytes"], "staged source size differs")
        require(sha256_file(temporary) == descriptor["sha256"], "staged source SHA-256 differs")
        crc32c, md5 = local_gcloud_hashes(temporary)
        require(crc32c == descriptor["crc32c_base64"], "staged source CRC32C differs")
        require(md5 == descriptor["md5_base64"], "staged source MD5 differs")
        temporary.chmod(stat.S_IRUSR)
        os.link(temporary, output_vcf)
        helpers.fsync_directory(output_vcf.parent)
    finally:
        temporary.unlink(missing_ok=True)

    payload = {
        "schema_version": "1.0.0",
        "stage": "M33_I0_REAL_SOURCE",
        "status": "PASS_EXACT_GENERATION_STAGED_READ_ONLY",
        "run_id": run_id,
        "root_label": root_label,
        "root_seed": descriptor["root_seed"],
        "generation": descriptor["generation"],
        "size_bytes": descriptor["size_bytes"],
        "source_sha256": descriptor["sha256"],
        "crc32c_base64": descriptor["crc32c_base64"],
        "md5_base64": descriptor["md5_base64"],
        "download_principal": account,
        "source_auth_sha256": source_auth_sha,
        "read_only_mode": "0400",
        "source_bucket_write_attempted": False,
        "receipt_contains_genomic_payload": False,
    }
    helpers.atomic_bytes_no_overwrite(
        receipt, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    require(load_json(receipt) == payload, "source receipt reopen differs")
    return payload


def validate_source_receipt(
    receipt: Path, *, root_label: str, run_id: str, descriptor: dict[str, Any],
    source_auth_sha: str,
) -> None:
    payload = load_json(receipt)
    require(payload == {
        "schema_version": "1.0.0",
        "stage": "M33_I0_REAL_SOURCE",
        "status": "PASS_EXACT_GENERATION_STAGED_READ_ONLY",
        "run_id": run_id,
        "root_label": root_label,
        "root_seed": descriptor["root_seed"],
        "generation": descriptor["generation"],
        "size_bytes": descriptor["size_bytes"],
        "source_sha256": descriptor["sha256"],
        "crc32c_base64": descriptor["crc32c_base64"],
        "md5_base64": descriptor["md5_base64"],
        "download_principal": "jcalderonta@ime.usp.br",
        "source_auth_sha256": source_auth_sha,
        "read_only_mode": "0400",
        "source_bucket_write_attempted": False,
        "receipt_contains_genomic_payload": False,
    }, "source receipt differs")


def derive_index(
    *, root_label: str, source: Path, source_receipt: Path, output_tbi: Path,
    receipt: Path, marker: Path, authorization: Path, contract: Path,
    source_auth: Path, helper_script: Path, run_id: str, container_image: str,
) -> dict[str, Any]:
    validate_run_id(run_id)
    auth = load_authorization(authorization, contract)
    source_auth_sha = load_runtime_source_auth(
        source_auth,
        real_script=Path(__file__).resolve(),
        helper_script=helper_script,
        authorization=authorization,
        contract=contract,
    )
    require(root_label in EXPECTED_ROOTS, "unauthorized root")
    descriptor = auth["roots"][root_label]
    require(container_image == auth["tabix_image"] == EXPECTED_IMAGE, "effective image differs")
    require(source.is_file(), "staged source is missing")
    require((source.stat().st_mode & 0o222) == 0, "staged source must be read-only")
    require(source.stat().st_size == descriptor["size_bytes"], "source size differs before indexing")
    require(sha256_file(source) == descriptor["sha256"], "source hash differs before indexing")
    validate_source_receipt(
        source_receipt, root_label=root_label, run_id=run_id,
        descriptor=descriptor, source_auth_sha=source_auth_sha,
    )
    require(output_tbi.name == f"{root_label}.flare.anc.vcf.gz.tbi", "index basename differs")
    require(receipt.name == f"{root_label}.i0_real.receipt.json", "receipt basename differs")
    require(marker.name == f"{root_label.upper()}_I0_REAL_PASS_NON_CONSUMABLE", "marker differs")
    require(len({p.parent.absolute() for p in (source, output_tbi, receipt, marker)}) == 1,
            "I0 inputs and outputs must share a task directory")
    require(not output_tbi.exists() and not receipt.exists() and not marker.exists(),
            "I0 output already exists")

    helpers = load_fixture_helpers(helper_script)
    require(helpers.tabix_version() == EXPECTED_VERSION, "runtime Tabix version differs")
    sequential_count, sequential_sha = helpers.sequential_chr22(source)
    require(sequential_count == EXPECTED_RECORD_COUNT, "sequential known-answer count differs")
    with tempfile.TemporaryDirectory(prefix=f"{root_label}-a-", dir=source.parent) as first, \
         tempfile.TemporaryDirectory(prefix=f"{root_label}-b-", dir=source.parent) as second:
        index_a, count_a, query_a = helpers.build_one_index(source, Path(first), descriptor["sha256"])
        index_b, count_b, query_b = helpers.build_one_index(source, Path(second), descriptor["sha256"])
        tbi_a = sha256_file(index_a)
        tbi_b = sha256_file(index_b)
        require(tbi_a == tbi_b, "independent TBI hashes differ")
        require(count_a == count_b == sequential_count, "indexed/sequential counts differ")
        require(query_a == query_b == sequential_sha, "indexed/sequential record digests differ")
        helpers.atomic_copy_no_overwrite(index_a, output_tbi)

    require(source.stat().st_size == descriptor["size_bytes"], "source size changed")
    require(sha256_file(source) == descriptor["sha256"], "source hash changed")
    require(sha256_file(output_tbi) == tbi_a, "reopened output TBI differs")
    payload = {
        "schema_version": "1.0.0",
        "stage": "M33_I0_REAL_INDEX",
        "status": "PASS_DOUBLE_INDEX_AND_QUERY_PARITY_TECHNICAL_ONLY",
        "run_id": run_id,
        "root_label": root_label,
        "root_seed": descriptor["root_seed"],
        "source_generation": descriptor["generation"],
        "source_flare_sha256": descriptor["sha256"],
        "source_size_bytes": descriptor["size_bytes"],
        "source_auth_sha256": source_auth_sha,
        "tabix_version": EXPECTED_VERSION,
        "tabix_oci_repository_digest": EXPECTED_IMAGE,
        "build_replicates": 2,
        "build_replication_scope": "SAME_TASK_SAME_CONTAINER_SEPARATE_TEMP_DIRECTORIES",
        "independent_tbi_sha256": tbi_a,
        "query_parity_sha256": sequential_sha,
        "indexed_record_count": count_a,
        "sequential_record_count": sequential_count,
        "output_tbi_sha256": sha256_file(output_tbi),
        "append_only": True,
        "reopen_verified": True,
        "scientific_evidence": False,
        "safe_bridge": False,
        "materialize": False,
        "training": False,
        "truth": False,
        "test": False,
        "global_ready": False,
    }
    helpers.atomic_bytes_no_overwrite(
        receipt, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    marker_payload = {
        "stage": "M33_I0_REAL_ROOT_PASS",
        "status": "PASS_TECHNICAL_NON_CONSUMABLE",
        "run_id": run_id,
        "root_label": root_label,
        "receipt_sha256": sha256_file(receipt),
        "global_ready": False,
    }
    helpers.atomic_bytes_no_overwrite(
        marker, (json.dumps(marker_payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    require(load_json(receipt) == payload, "receipt reopen differs")
    require(load_json(marker) == marker_payload, "marker reopen differs")
    return payload


def verify_existing_index(
    *, source: Path, index: Path, descriptor: dict[str, Any],
    expected_query_sha: str, helpers: Any,
) -> None:
    """Reopen one emitted TBI against its exact VCF and a sequential oracle."""
    root_label = source.name.removesuffix(".flare.anc.vcf.gz")
    require(root_label in EXPECTED_ROOTS, "malformed aggregate source basename")
    require(index.name == f"{source.name}.tbi", "source/index basename binding differs")
    require(source.parent.absolute() == index.parent.absolute(),
            "source and index must share the aggregate task directory")
    require(source.is_file() and index.is_file(), "aggregate source or index is missing")
    require((source.stat().st_mode & 0o222) == 0, "aggregate source must be read-only")
    require((index.stat().st_mode & 0o222) == 0, "aggregate index must be read-only")
    require(source.stat().st_size == descriptor["size_bytes"], "aggregate source size differs")
    require(sha256_file(source) == descriptor["sha256"], "aggregate source hash differs")
    require(helpers.tabix_version() == EXPECTED_VERSION, "aggregate Tabix version differs")
    sequential_count, sequential_sha = helpers.sequential_chr22(source)
    indexed_lines = subprocess.run(
        ["tabix", source.name, "22"], cwd=source.parent, check=True, capture_output=True,
    ).stdout.splitlines(keepends=True)
    indexed_count, indexed_sha = helpers.line_digest(indexed_lines)
    require(indexed_count == sequential_count == EXPECTED_RECORD_COUNT,
            "aggregate indexed/sequential counts differ")
    require(indexed_sha == sequential_sha == expected_query_sha,
            "aggregate indexed/sequential record digests differ")


def aggregate(
    *, receipts: list[Path], markers: list[Path], sources: list[Path], indexes: list[Path],
    manifest: Path, completion_marker: Path,
    authorization: Path, contract: Path, source_auth: Path, helper_script: Path,
    run_id: str, container_image: str,
) -> dict[str, Any]:
    validate_run_id(run_id)
    auth = load_authorization(authorization, contract)
    source_auth_sha = load_runtime_source_auth(
        source_auth,
        real_script=Path(__file__).resolve(),
        helper_script=helper_script,
        authorization=authorization,
        contract=contract,
    )
    require(container_image == auth["tabix_image"] == EXPECTED_IMAGE,
            "aggregate effective image differs")
    require(manifest.name == "m33_i0_real.manifest.json", "manifest basename differs")
    require(completion_marker.name == "I0_REAL_PASS_NON_CONSUMABLE", "completion marker differs")
    require(not manifest.exists() and not completion_marker.exists(), "aggregate output already exists")
    require(len(receipts) == len(markers) == len(sources) == len(indexes) == 2,
            "aggregate requires two roots, sources, and indexes")
    receipt_pairs = [(load_json(path), path) for path in receipts]
    marker_pairs = [(load_json(path), path) for path in markers]
    require(all("root_label" in payload for payload, _ in receipt_pairs),
            "root label missing from receipt")
    require(all("root_label" in payload for payload, _ in marker_pairs),
            "root label missing from marker")
    receipt_by_root = {payload["root_label"]: (payload, path) for payload, path in receipt_pairs}
    marker_by_root = {payload["root_label"]: (payload, path) for payload, path in marker_pairs}
    require(len(receipt_by_root) == len(marker_by_root) == 2, "duplicate root evidence")
    index_by_root = {
        path.name.removesuffix(".flare.anc.vcf.gz.tbi"): path for path in indexes
    }
    source_by_root = {
        path.name.removesuffix(".flare.anc.vcf.gz"): path for path in sources
    }
    require(len(index_by_root) == len(source_by_root) == 2,
            "duplicate or malformed root sources/indexes")
    require(set(receipt_by_root) == set(marker_by_root) == set(source_by_root)
            == set(index_by_root) == set(EXPECTED_ROOTS),
            "root set differs")
    helpers = load_fixture_helpers(helper_script)
    roots: dict[str, Any] = {}
    for root_label in sorted(EXPECTED_ROOTS):
        descriptor = auth["roots"][root_label]
        receipt_payload, receipt_path = receipt_by_root[root_label]
        marker_payload, _ = marker_by_root[root_label]
        index_path = index_by_root[root_label]
        digest_fields = (
            receipt_payload.get("independent_tbi_sha256"),
            receipt_payload.get("query_parity_sha256"),
            receipt_payload.get("output_tbi_sha256"),
        )
        require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in digest_fields), "invalid root digest")
        expected_receipt = {
            "schema_version": "1.0.0",
            "stage": "M33_I0_REAL_INDEX",
            "status": "PASS_DOUBLE_INDEX_AND_QUERY_PARITY_TECHNICAL_ONLY",
            "run_id": run_id,
            "root_label": root_label,
            "root_seed": descriptor["root_seed"],
            "source_generation": descriptor["generation"],
            "source_flare_sha256": descriptor["sha256"],
            "source_size_bytes": descriptor["size_bytes"],
            "source_auth_sha256": source_auth_sha,
            "tabix_version": EXPECTED_VERSION,
            "tabix_oci_repository_digest": EXPECTED_IMAGE,
            "build_replicates": 2,
            "build_replication_scope": "SAME_TASK_SAME_CONTAINER_SEPARATE_TEMP_DIRECTORIES",
            "independent_tbi_sha256": receipt_payload["independent_tbi_sha256"],
            "query_parity_sha256": receipt_payload["query_parity_sha256"],
            "indexed_record_count": EXPECTED_RECORD_COUNT,
            "sequential_record_count": EXPECTED_RECORD_COUNT,
            "output_tbi_sha256": receipt_payload["output_tbi_sha256"],
            "append_only": True,
            "reopen_verified": True,
            "scientific_evidence": False,
            "safe_bridge": False,
            "materialize": False,
            "training": False,
            "truth": False,
            "test": False,
            "global_ready": False,
        }
        require(receipt_payload == expected_receipt, f"root receipt schema or value differs: {root_label}")
        require(receipt_payload["independent_tbi_sha256"] == receipt_payload["output_tbi_sha256"],
                "independent/output TBI binding differs")
        require(index_path.is_file(), "root TBI is missing")
        require((index_path.stat().st_mode & 0o222) == 0, "root TBI must be read-only")
        require(sha256_file(index_path) == receipt_payload["output_tbi_sha256"],
                "reopened root TBI hash differs")
        require(marker_payload == {
            "stage": "M33_I0_REAL_ROOT_PASS",
            "status": "PASS_TECHNICAL_NON_CONSUMABLE",
            "run_id": run_id,
            "root_label": root_label,
            "receipt_sha256": sha256_file(receipt_path),
            "global_ready": False,
        }, f"root marker schema or binding differs: {root_label}")
        verify_existing_index(
            source=source_by_root[root_label], index=index_path, descriptor=descriptor,
            expected_query_sha=receipt_payload["query_parity_sha256"], helpers=helpers,
        )
        roots[root_label] = {
            "root_seed": receipt_payload["root_seed"],
            "source_generation": receipt_payload["source_generation"],
            "source_flare_sha256": receipt_payload["source_flare_sha256"],
            "output_tbi_sha256": receipt_payload["output_tbi_sha256"],
            "query_parity_sha256": receipt_payload["query_parity_sha256"],
            "record_count": receipt_payload["indexed_record_count"],
            "status": receipt_payload["status"],
        }
    payload = {
        "schema_version": "1.0.0",
        "stage": "M33_I0_REAL_AGGREGATE",
        "status": "PASS_2_OF_2_TECHNICAL_ROOTS_NO_DOWNSTREAM_OPEN",
        "run_id": run_id,
        "source_auth_sha256": source_auth_sha,
        "roots": roots,
        "root_pass_count": 2,
        "scientific_evidence": False,
        "safe_bridge": False,
        "materialize": False,
        "training": False,
        "truth": False,
        "test": False,
        "global_ready": False,
        "gcs_published": False,
    }
    helpers.atomic_bytes_no_overwrite(
        manifest, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    marker_payload = {
        "stage": "M33_I0_REAL_PASS",
        "status": "PASS_NON_CONSUMABLE_PENDING_POST_AND_PUBLICATION",
        "run_id": run_id,
        "manifest_sha256": sha256_file(manifest),
        "global_ready": False,
    }
    helpers.atomic_bytes_no_overwrite(
        completion_marker,
        (json.dumps(marker_payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    require(load_json(manifest) == payload, "manifest reopen differs")
    require(load_json(completion_marker) == marker_payload, "completion marker reopen differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--authorization", type=Path, required=True)
    common.add_argument("--contract", type=Path, required=True)
    common.add_argument("--source-auth", type=Path, required=True)
    common.add_argument("--run-id", required=True)

    stage_parser = subparsers.add_parser("stage", parents=[common])
    stage_parser.add_argument("--root-label", required=True)
    stage_parser.add_argument("--output-vcf", type=Path, required=True)
    stage_parser.add_argument("--receipt", type=Path, required=True)
    stage_parser.add_argument("--helper-script", type=Path, required=True)
    stage_parser.add_argument("--repo-root", type=Path, required=True)

    index_parser = subparsers.add_parser("index", parents=[common])
    index_parser.add_argument("--root-label", required=True)
    index_parser.add_argument("--source", type=Path, required=True)
    index_parser.add_argument("--source-receipt", type=Path, required=True)
    index_parser.add_argument("--output-tbi", type=Path, required=True)
    index_parser.add_argument("--receipt", type=Path, required=True)
    index_parser.add_argument("--marker", type=Path, required=True)
    index_parser.add_argument("--container-image", required=True)
    index_parser.add_argument("--helper-script", type=Path, required=True)

    aggregate_parser = subparsers.add_parser("aggregate", parents=[common])
    aggregate_parser.add_argument("--receipts", type=Path, nargs=2, required=True)
    aggregate_parser.add_argument("--markers", type=Path, nargs=2, required=True)
    aggregate_parser.add_argument("--sources", type=Path, nargs=2, required=True)
    aggregate_parser.add_argument("--indexes", type=Path, nargs=2, required=True)
    aggregate_parser.add_argument("--manifest", type=Path, required=True)
    aggregate_parser.add_argument("--completion-marker", type=Path, required=True)
    aggregate_parser.add_argument("--helper-script", type=Path, required=True)
    aggregate_parser.add_argument("--container-image", required=True)
    args = parser.parse_args()
    common_kwargs = {
        "authorization": args.authorization,
        "contract": args.contract,
        "source_auth": args.source_auth,
        "run_id": args.run_id,
    }
    if args.command == "stage":
        result = stage_source(
            root_label=args.root_label, output_vcf=args.output_vcf, receipt=args.receipt,
            helper_script=args.helper_script, repo_root=args.repo_root, **common_kwargs,
        )
    elif args.command == "index":
        result = derive_index(
            root_label=args.root_label, source=args.source, source_receipt=args.source_receipt,
            output_tbi=args.output_tbi, receipt=args.receipt, marker=args.marker,
            helper_script=args.helper_script, container_image=args.container_image, **common_kwargs,
        )
    else:
        result = aggregate(
            receipts=args.receipts, markers=args.markers, sources=args.sources, indexes=args.indexes,
            manifest=args.manifest, completion_marker=args.completion_marker,
            helper_script=args.helper_script, container_image=args.container_image, **common_kwargs,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
