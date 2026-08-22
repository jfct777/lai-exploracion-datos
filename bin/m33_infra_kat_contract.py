#!/usr/bin/env python3
"""Validate the narrow M33 infrastructure and synthetic-KAT authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


STATUS = "AUTHORIZED_INFRA_AND_SYNTHETIC_KAT_ONLY_NO_REAL_ASSET_READ"
CONTROLLER_SERVICE_ACCOUNT = "dnabr-m33-controller-frank@uspbr-242713.iam.gserviceaccount.com"
RUNTIME_SERVICE_ACCOUNT = "dnabr-m33-frank@uspbr-242713.iam.gserviceaccount.com"
MANAGED_FOLDER = "gs://teams-usp/frank/lai-exploracion-datos/"
REPOSITORY = "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-tabix"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )


def exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    require(set(value) == expected, f"{where} keys differ")


def validate_authorization(
    payload: Mapping[str, Any],
    *,
    storage_policy: Path,
    m0_contract: Path,
    require_published_digest: bool = False,
) -> Mapping[str, Any]:
    exact_keys(payload, {
        "schema_version", "stage", "status", "parent_contracts", "gcp",
        "runtime", "storage_permissions", "authorization", "kat",
    }, "authorization")
    require(payload["schema_version"] == "1.0.0", "schema version drifted")
    require(payload["stage"] == "M33_INFRA_AND_SYNTHETIC_KAT_AUTHORIZATION", "stage drifted")
    require(payload["status"] == STATUS, "authorization status drifted")

    parents = payload["parent_contracts"]
    require(parents == {
        "storage_namespace_policy_sha256": sha256_file(storage_policy),
        "m0_materializer_contract_sha256": sha256_file(m0_contract),
    }, "parent contract hash drifted")

    gcp = payload["gcp"]
    require(gcp == {
        "project": "uspbr-242713",
        "region": "us-central1",
        "managed_folder": MANAGED_FOLDER,
        "controller_service_account": CONTROLLER_SERVICE_ACCOUNT,
        "runtime_service_account": RUNTIME_SERVICE_ACCOUNT,
        "resource_labels": {"team": "frank"},
    }, "GCP boundary drifted")

    runtime = payload["runtime"]
    exact_keys(runtime, {
        "nextflow_version", "nf_google_version", "tabix_version",
        "oci_repository", "oci_digest",
    }, "runtime")
    require(runtime["nextflow_version"] == "26.04.6", "Nextflow version drifted")
    require(runtime["nf_google_version"] == "1.27.3", "nf-google version drifted")
    require(runtime["tabix_version"] == "1.16", "Tabix version drifted")
    require(runtime["oci_repository"] == REPOSITORY, "OCI repository drifted")
    digest = runtime["oci_digest"]
    require(
        digest == "BLOCKED_PENDING_PUBLISHED_DIGEST" or DIGEST_RE.fullmatch(str(digest)),
        "OCI digest must be blocked or immutable",
    )
    if require_published_digest:
        require(DIGEST_RE.fullmatch(str(digest)) is not None, "published OCI digest is required")

    permissions = payload["storage_permissions"]
    require(permissions == {
        "work_prefix": "work/nextflow/",
        "work_allows_create_get_list": True,
        "work_allows_update": False,
        "work_allows_delete": False,
        "append_only_prefixes": ["runs/", "logs/", "manifests/", "software/containers/"],
        "append_only_precondition": "ifGenerationMatch=0",
        "ready_is_last": True,
        "delete_forbidden_everywhere": True,
        "write_outside_managed_folder": "STOP",
    }, "storage permission boundary drifted")

    authorization = payload["authorization"]
    require(authorization == {
        "create_dedicated_non_editor_controller_service_account": True,
        "create_dedicated_non_editor_runtime_service_account": True,
        "create_managed_folder": True,
        "publish_tabix_oci": True,
        "synthetic_fixture_only": True,
        "synthetic_kat": True,
        "lab_asset_read": False,
        "root17_read": False,
        "root18_read": False,
        "eval_read": False,
        "derived_real_index_write": False,
        "safe_bridge": False,
        "materialization": False,
        "training": False,
    }, "authorization exceeds infrastructure and synthetic KAT")

    kat = payload["kat"]
    exact_keys(kat, {
        "chromosome", "replicas", "independent_nextflow_tasks",
        "shared_task_cache_forbidden", "controller_runner_required",
        "direct_launch_forbidden", "required_checks",
    }, "KAT")
    require(kat["chromosome"] == "22", "KAT chromosome drifted")
    require(kat["replicas"] == ["A", "B"], "KAT replicas drifted")
    require(kat["independent_nextflow_tasks"] is True, "independent KAT tasks required")
    require(kat["shared_task_cache_forbidden"] is True, "shared KAT cache is forbidden")
    require(kat["controller_runner_required"] is True, "controller runner is required")
    require(kat["direct_launch_forbidden"] is True, "direct launch must remain forbidden")
    required = set(kat["required_checks"])
    require(required == {
        "tabix_version_exact", "source_vcf_sha256_equal",
        "independent_tbi_sha256_equal", "indexed_sequential_record_count_equal",
        "indexed_sequential_sha256_equal", "effective_service_account_exact",
        "effective_controller_service_account_exact",
        "resource_label_exact", "workdir_under_project_prefix", "no_lab_asset_reads",
        "no_active_batch_jobs_after_completion",
    }, "KAT checks drifted")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--storage-policy", required=True, type=Path)
    parser.add_argument("--m0-contract", required=True, type=Path)
    parser.add_argument("--require-published-digest", action="store_true")
    args = parser.parse_args()
    payload = load_json(args.authorization)
    validate_authorization(
        payload,
        storage_policy=args.storage_policy,
        m0_contract=args.m0_contract,
        require_published_digest=args.require_published_digest,
    )
    print(json.dumps({
        "stage": payload["stage"],
        "status": "PASS_INFRA_AND_SYNTHETIC_KAT_AUTHORIZATION",
        "real_asset_read": False,
        "training": False,
        "oci_digest": payload["runtime"]["oci_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
