#!/usr/bin/env python3
"""Append-only publisher for the eight non-genomic M33 I0 derived artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_PREFIX = "gs://teams-usp/frank/lai-exploracion-datos/runs/m33-i0-real-20260822a/"
EXPECTED_PRINCIPAL = "jcalderonta@ime.usp.br"
EXPECTED_RUN_ID = "m33-i0-real-20260822a"
EXPECTED_I0_SOURCE_COMMIT = "731757aa89949de9a641568a38697794f31a0fff"
EXPECTED_BASE_POLICY_SHA256 = "ff9143e29fe154cc5d793a7b36efe91cefc08ff012f8024ffa1c39b7b1adc1da"
SOURCE_FILES = {
    "bin/m33_i0_publish.py",
    "conf/m33_i0_publication_authorization.json",
    "conf/m33_storage_namespace_policy.json",
    "tests/test_m33_i0_publish.py",
}
FORBIDDEN_ENV = {
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


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    require(isinstance(payload, dict), f"JSON object required: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_authorization(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    require(set(payload) == {
        "schema_version", "stage", "status", "run_id", "i0_source_commit",
        "publisher_code_commit", "base_storage_policy_sha256", "amendment",
        "source_manifest_sha256", "publisher_principal", "destination_prefix",
        "artifacts", "policy",
    }, "publication authorization keys differ")
    require(payload["schema_version"] == "1.0.0", "authorization schema differs")
    require(payload["stage"] == "M33_I0_PUBLICATION_AUTHORIZATION", "authorization stage differs")
    require(payload["status"] == "AUTHORIZED_APPEND_ONLY_DERIVED_ARTIFACTS_NO_READY",
            "publication is not authorized")
    require(payload["run_id"] == EXPECTED_RUN_ID, "run ID differs")
    require(payload["i0_source_commit"] == EXPECTED_I0_SOURCE_COMMIT, "I0 source commit differs")
    require(re.fullmatch(r"[0-9a-f]{40}", payload["publisher_code_commit"]) is not None,
            "publisher code commit is invalid")
    require(payload["base_storage_policy_sha256"] == EXPECTED_BASE_POLICY_SHA256,
            "base storage policy hash differs")
    require(payload["amendment"] == {
        "base_json_pointer": "/execution_authorization/derived_index_write",
        "base_value": False,
        "amended_value": True,
        "scope": "EXACT_EIGHT_I0_DERIVED_ARTIFACTS_THIS_RUN_ONLY",
    }, "publication amendment differs")
    require(payload["publisher_principal"] == EXPECTED_PRINCIPAL, "publisher principal differs")
    require(payload["destination_prefix"] == EXPECTED_PREFIX, "destination prefix differs")
    require(re.fullmatch(r"[0-9a-f]{64}", payload["source_manifest_sha256"]) is not None,
            "manifest hash is invalid")
    artifacts = payload["artifacts"]
    require(isinstance(artifacts, dict) and len(artifacts) == 8, "exactly eight artifacts required")
    for relative, descriptor in artifacts.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
        require(not relative.endswith(".vcf.gz"), "VCF publication is forbidden")
        require(set(descriptor) == {"size_bytes", "sha256"}, "artifact descriptor differs")
        require(type(descriptor["size_bytes"]) is int and descriptor["size_bytes"] > 0,
                "artifact size is invalid")
        require(re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is not None,
                "artifact SHA-256 is invalid")
    require(payload["policy"] == {
        "expected_file_count": 8,
        "required_mode": "0444",
        "if_generation_match": 0,
        "reopen_exact_generation": True,
        "publication_receipt_last": True,
        "allow_vcf": False,
        "allow_overwrite": False,
        "allow_delete": False,
        "allow_ready": False,
        "safe_bridge": False,
        "materialize": False,
        "training": False,
        "global_ready": False,
    }, "publication policy differs")
    return payload


def load_source_auth(path: Path, repo_root: Path) -> str:
    payload = load_json(path)
    require(set(payload) == {"schema_version", "stage", "status", "files"},
            "source-auth keys differ")
    require(payload["schema_version"] == "1.0.0", "source-auth schema differs")
    require(payload["stage"] == "M33_I0_PUBLICATION_SOURCE_AUTH", "source-auth stage differs")
    require(payload["status"] == "AUTHORIZED_EXACT_PUBLICATION_SOURCES", "source-auth status differs")
    require(set(payload["files"]) == SOURCE_FILES, "source-auth inventory differs")
    for relative, expected in payload["files"].items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "invalid source hash")
        require(sha256_file(repo_root / relative) == expected, f"source hash differs: {relative}")
    return sha256_file(path)


def validate_publisher_commit(repo_root: Path, authorization: dict[str, Any]) -> str:
    require(not subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip(), "tracked worktree must be clean")
    commit = authorization["publisher_code_commit"]
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:bin/m33_i0_publish.py"],
        check=True, capture_output=True,
    )
    require(hashlib.sha256(result.stdout).hexdigest() == sha256_file(repo_root / "bin/m33_i0_publish.py"),
            "publisher script differs from authorized commit")
    return commit


def validate_base_storage_policy(path: Path, authorization: dict[str, Any]) -> None:
    require(sha256_file(path) == authorization["base_storage_policy_sha256"],
            "base storage policy bytes differ")
    payload = load_json(path)
    require(payload["execution_authorization"]["derived_index_write"] is False,
            "base derived-index gate was modified retroactively")
    require(payload["persistent_write_contract"]["append_only"] is True,
            "base append-only policy differs")
    require(payload["persistent_write_contract"]["object_creation_precondition"]
            == "ifGenerationMatch=0", "base create precondition differs")
    require(payload["persistent_write_contract"]["overwrite_forbidden"] is True,
            "base overwrite policy differs")
    require(payload["persistent_write_contract"]["delete_forbidden"] is True,
            "base delete policy differs")


def validate_local_artifacts(root: Path, authorization: dict[str, Any]) -> dict[str, Path]:
    require(root.is_dir() and not root.is_symlink(), "local artifact root is invalid")
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*") if path.is_file() or path.is_symlink()
    }
    require(set(observed) == set(authorization["artifacts"]), "local artifact inventory differs")
    for relative, path in observed.items():
        require(path.is_file() and not path.is_symlink(), f"artifact must be a regular file: {relative}")
        require((path.stat().st_mode & 0o777) == 0o444, f"artifact mode differs: {relative}")
        expected = authorization["artifacts"][relative]
        require(path.stat().st_size == expected["size_bytes"], f"artifact size differs: {relative}")
        require(sha256_file(path) == expected["sha256"], f"artifact hash differs: {relative}")
    require(authorization["artifacts"]["m33_i0_real.manifest.json"]["sha256"]
            == authorization["source_manifest_sha256"], "manifest authorization binding differs")
    return observed


def validate_i0_semantics(root: Path, authorization: dict[str, Any]) -> None:
    manifest = load_json(root / "m33_i0_real.manifest.json")
    require(manifest["run_id"] == authorization["run_id"], "manifest run ID differs")
    require(manifest["status"] == "PASS_2_OF_2_TECHNICAL_ROOTS_NO_DOWNSTREAM_OPEN",
            "I0 aggregate did not pass")
    require(manifest["root_pass_count"] == 2 and set(manifest["roots"]) == {"root17", "root18"},
            "I0 root set differs")
    for flag in ("gcs_published", "global_ready", "materialize", "safe_bridge", "scientific_evidence",
                 "test", "training", "truth"):
        require(manifest[flag] is False, f"forbidden manifest flag opened: {flag}")
    for root_label in ("root17", "root18"):
        receipt = load_json(root / root_label / f"{root_label}.i0_real.receipt.json")
        root_manifest = manifest["roots"][root_label]
        require(receipt["status"] == "PASS_DOUBLE_INDEX_AND_QUERY_PARITY_TECHNICAL_ONLY",
                f"root receipt did not pass: {root_label}")
        require(receipt["indexed_record_count"] == receipt["sequential_record_count"] == 79791,
                f"root count differs: {root_label}")
        require(receipt["independent_tbi_sha256"] == receipt["output_tbi_sha256"],
                f"root TBI builds differ: {root_label}")
        require(receipt["output_tbi_sha256"] == root_manifest["output_tbi_sha256"],
                f"root manifest TBI binding differs: {root_label}")
        require(receipt["query_parity_sha256"] == root_manifest["query_parity_sha256"],
                f"root query binding differs: {root_label}")
        for flag in ("global_ready", "materialize", "safe_bridge", "scientific_evidence",
                     "test", "training", "truth"):
            require(receipt[flag] is False, f"forbidden root flag opened: {root_label}/{flag}")


def active_account() -> str:
    for variable in FORBIDDEN_ENV:
        require(not os.environ.get(variable), f"credential override is forbidden: {variable}")
    impersonation = subprocess.run(
        ["gcloud", "config", "get-value", "auth/impersonate_service_account"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(impersonation in {"", "(unset)"}, "service-account impersonation is forbidden")
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        check=True, capture_output=True, text=True,
    )
    accounts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(accounts == [EXPECTED_PRINCIPAL], "active publisher account differs")
    return accounts[0]


def require_empty_prefix(prefix: str) -> None:
    result = subprocess.run(
        ["gcloud", "storage", "ls", prefix, "--recursive"], capture_output=True, text=True,
    )
    require(result.returncode in {0, 1}, "destination listing failed")
    require(not result.stdout.strip(), "destination prefix is not empty")


def describe_object(uri: str) -> dict[str, Any]:
    result = subprocess.run(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    require(isinstance(payload, dict), "unexpected object description")
    return payload


def publish_one(local: Path, uri: str, expected: dict[str, Any]) -> dict[str, Any]:
    subprocess.run(
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(local), uri], check=True,
    )
    metadata = describe_object(uri)
    generation = str(metadata["generation"])
    require(int(metadata["size"]) == expected["size_bytes"], "remote size differs")
    with tempfile.TemporaryDirectory(prefix="m33-i0-reopen-") as temporary:
        reopened = Path(temporary) / local.name
        subprocess.run(["gcloud", "storage", "cp", f"{uri}#{generation}", str(reopened)], check=True)
        require(reopened.stat().st_size == expected["size_bytes"], "reopened size differs")
        require(sha256_file(reopened) == expected["sha256"], "reopened SHA-256 differs")
    return {
        "uri": uri,
        "generation": generation,
        "size_bytes": expected["size_bytes"],
        "sha256": expected["sha256"],
        "crc32c_base64": metadata["crc32c_hash"],
        "md5_base64": metadata["md5_hash"],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.name == "publication.receipt.json", "publication receipt basename differs")
    require(not path.exists() and not path.is_symlink(), "publication receipt already exists")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(
    *, artifact_root: Path, authorization_path: Path, source_auth_path: Path,
    base_policy_path: Path, repo_root: Path, receipt_path: Path,
) -> dict[str, Any]:
    authorization = load_authorization(authorization_path)
    source_auth_sha = load_source_auth(source_auth_path, repo_root)
    publisher_commit = validate_publisher_commit(repo_root, authorization)
    validate_base_storage_policy(base_policy_path, authorization)
    artifacts = validate_local_artifacts(artifact_root, authorization)
    validate_i0_semantics(artifact_root, authorization)
    principal = active_account()
    require_empty_prefix(authorization["destination_prefix"])
    published = []
    order = sorted(
        artifacts,
        key=lambda relative: (
            2 if relative == "I0_REAL_PASS_NON_CONSUMABLE"
            else 1 if relative == "m33_i0_real.manifest.json"
            else 0,
            relative,
        ),
    )
    for relative in order:
        published.append(publish_one(
            artifacts[relative], authorization["destination_prefix"] + relative,
            authorization["artifacts"][relative],
        ))
    payload = {
        "schema_version": "1.0.0",
        "stage": "M33_I0_PUBLICATION",
        "status": "PASS_APPEND_ONLY_REOPENED_NON_CONSUMABLE",
        "run_id": authorization["run_id"],
        "i0_source_commit": authorization["i0_source_commit"],
        "publisher_code_commit": publisher_commit,
        "base_storage_policy_sha256": authorization["base_storage_policy_sha256"],
        "source_manifest_sha256": authorization["source_manifest_sha256"],
        "publication_source_auth_sha256": source_auth_sha,
        "publisher_principal": principal,
        "destination_prefix": authorization["destination_prefix"],
        "published_object_count": 8,
        "objects": published,
        "source_vcf_published": False,
        "safe_bridge": False,
        "materialize": False,
        "training": False,
        "truth": False,
        "test": False,
        "global_ready": False,
    }
    atomic_json(receipt_path, payload)
    receipt_expected = {"size_bytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path)}
    remote_receipt = publish_one(
        receipt_path, authorization["destination_prefix"] + "publication.receipt.json", receipt_expected,
    )
    result = dict(payload)
    result["publication_receipt"] = remote_receipt
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = publish(
        artifact_root=args.artifact_root,
        authorization_path=args.authorization,
        source_auth_path=args.source_auth,
        base_policy_path=args.base_policy,
        repo_root=args.repo_root,
        receipt_path=args.receipt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
