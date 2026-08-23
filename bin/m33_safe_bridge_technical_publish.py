#!/usr/bin/env python3
"""Publish only the minimal audit evidence for the M33 technical SAFE_BRIDGE KAT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


RUN_ID = "m33-safe-bridge-technical-kat-20260823e"
PREFIX = f"gs://teams-usp/frank/lai-exploracion-datos/runs/{RUN_ID}/"
PRINCIPAL = "jcalderonta@ime.usp.br"
BASE_POLICY_SHA256 = "ff9143e29fe154cc5d793a7b36efe91cefc08ff012f8024ffa1c39b7b1adc1da"
CONFIG_FILE = "conf/m33_safe_bridge_technical_publication_config.json"
AUTH_FILE = "conf/m33_safe_bridge_technical_publication_authorization.json"
SOURCE_AUTH_FILE = "conf/m33_safe_bridge_technical_publication_source_auth.json"
BASE_POLICY_FILE = "conf/m33_storage_namespace_policy.json"
SOURCE_FILES = {
    "bin/m33_safe_bridge_technical_publish.py",
    CONFIG_FILE,
    AUTH_FILE,
    BASE_POLICY_FILE,
    "tests/test_m33_safe_bridge_technical_publish.py",
}
SOURCE_ORDER = (
    "root17/safe_bridge_technical_kat.receipt.json",
    "root18/safe_bridge_technical_kat.receipt.json",
    "verification/root17.independent_verify.receipt.json",
    "verification/root18.independent_verify.receipt.json",
    "provenance/m33_safe_bridge_technical_kat_contract.json",
    "provenance/m33_safe_bridge_technical_kat_authorization.json",
    "provenance/m33_safe_bridge_technical_kat_source_auth.json",
)
MANIFEST = "m33_safe_bridge_technical_kat.manifest.json"
PUBLICATION_RECEIPT = "publication.receipt.json"
FINAL_ORDER = SOURCE_ORDER + (MANIFEST, PUBLICATION_RECEIPT)
RESULT_FILES = {
    "root17/safe_bridge_technical_kat.receipt.json":
        "results/root17.technical_kat/safe_bridge_technical_kat.receipt.json",
    "root18/safe_bridge_technical_kat.receipt.json":
        "results/root18.technical_kat/safe_bridge_technical_kat.receipt.json",
    "verification/root17.independent_verify.receipt.json":
        "verification/root17.independent_verify.receipt.json",
    "verification/root18.independent_verify.receipt.json":
        "verification/root18.independent_verify.receipt.json",
}
PROVENANCE_FILES = {
    "provenance/m33_safe_bridge_technical_kat_contract.json":
        "conf/m33_safe_bridge_technical_kat_contract.json",
    "provenance/m33_safe_bridge_technical_kat_authorization.json":
        "conf/m33_safe_bridge_technical_kat_authorization.json",
    "provenance/m33_safe_bridge_technical_kat_source_auth.json":
        "conf/m33_safe_bridge_technical_kat_source_auth.json",
}
NPZ_NAMES = (
    "technical_kat_selected_loci_incremental.npz",
    "technical_kat_target_rare_diploid_incremental.npz",
    "technical_kat_reference_rare_summary_incremental.npz",
    "technical_kat_flare_f0_sanitized.npz",
)
EMPTY_PREFIX_MESSAGE = "ERROR: (gcloud.storage.ls) One or more URLs matched no objects."
FORBIDDEN_ENV = {
    "CLOUDSDK_AUTH_ACCESS_TOKEN", "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
    "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT", "CLOUDSDK_CONFIG",
    "CLOUDSDK_CORE_ACCOUNT", "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_IMPERSONATE_SERVICE_ACCOUNT", "GOOGLE_OAUTH_ACCESS_TOKEN",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON path: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in payload, f"duplicate JSON key: {key}")
            payload[key] = value
        return payload

    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")),
    )
    require(isinstance(payload, dict), "JSON object required")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    require(payload.get("stage") == "M33_SAFE_BRIDGE_TECHNICAL_PUBLICATION_CONFIG" and
            payload.get("status") == "CONFIGURED_MINIMAL_APPEND_ONLY_AUDIT_PUBLICATION",
            "publication config identity drifted")
    require(payload.get("run_id") == RUN_ID and payload.get("destination_prefix") == PREFIX and
            payload.get("publisher_principal") == PRINCIPAL,
            "publication config scope drifted")
    require(tuple(payload.get("source_artifact_order", [])) == SOURCE_ORDER and
            payload.get("generated_artifact_order") == [MANIFEST, PUBLICATION_RECEIPT],
            "publication order drifted")
    policy = payload.get("policy", {})
    require(policy.get("source_artifact_count") == 7 and
            policy.get("validated_ephemeral_npz_count") == 8 and
            policy.get("final_object_count") == 9 and policy.get("if_generation_match") == 0 and
            policy.get("reopen_exact_generation") is True and
            policy.get("publication_receipt_last") is True,
            "publication config controls drifted")
    for flag in (
        "allow_npz", "allow_code", "allow_input_assets", "allow_vcf_tbi_tree",
        "allow_workdirs", "allow_raw_identifiers", "allow_truth", "allow_ready",
        "allow_overwrite", "allow_delete", "consumable", "scientific_evidence",
        "materialize", "training",
    ):
        require(policy.get(flag) is False, f"publication config opens forbidden gate: {flag}")
    return payload


def load_authorization(path: Path, config_sha256: str) -> dict[str, Any]:
    payload = load_json(path)
    require(payload.get("stage") == "M33_SAFE_BRIDGE_TECHNICAL_PUBLICATION_AUTHORIZATION" and
            payload.get("status") == "AUTHORIZED_MINIMAL_APPEND_ONLY_AUDIT_PUBLICATION",
            "publication authorization is not enabled")
    require(payload.get("run_id") == RUN_ID and payload.get("destination_prefix") == PREFIX and
            payload.get("publisher_principal") == PRINCIPAL,
            "publication authorization scope drifted")
    require(payload.get("publication_config_sha256") == config_sha256 and
            payload.get("base_storage_policy_sha256") == BASE_POLICY_SHA256,
            "publication policy anchor drifted")
    require(re.fullmatch(r"[0-9a-f]{40}", str(payload.get("publisher_code_commit"))) is not None,
            "publisher commit is invalid")
    artifacts = payload.get("artifacts", {})
    require(set(artifacts) == set(SOURCE_ORDER), "publication allowlist drifted")
    for relative, descriptor in artifacts.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts and
                not relative.endswith((".npz", ".vcf", ".vcf.gz", ".tbi", ".trees", ".py", ".nf")) and
                "ready" not in relative.lower() and "truth" not in relative.lower(),
                f"forbidden publication path: {relative}")
        require(set(descriptor) == {"size_bytes", "sha256"} and
                type(descriptor["size_bytes"]) is int and descriptor["size_bytes"] > 0 and
                re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is not None,
                f"invalid descriptor: {relative}")
    policy = payload.get("policy", {})
    require(policy.get("final_object_count") == 9 and policy.get("source_artifact_count") == 7 and
            policy.get("validated_ephemeral_npz_count") == 8 and
            policy.get("if_generation_match") == 0 and
            policy.get("publication_receipt_last") is True,
            "authorization controls drifted")
    for flag in (
        "allow_npz", "allow_code", "allow_input_assets", "allow_vcf_tbi_tree",
        "allow_workdirs", "allow_raw_identifiers", "allow_truth", "allow_ready",
        "allow_overwrite", "allow_delete", "consumable", "scientific_evidence",
        "materialize", "training",
    ):
        require(policy.get(flag) is False, f"authorization opens forbidden gate: {flag}")
    return payload


def load_source_auth(path: Path, repo_root: Path) -> str:
    payload = load_json(path)
    require(payload.get("stage") == "M33_SAFE_BRIDGE_TECHNICAL_PUBLICATION_SOURCE_AUTH" and
            payload.get("status") == "AUTHORIZED_EXACT_PUBLICATION_SOURCES" and
            set(payload.get("files", {})) == SOURCE_FILES,
            "publication source-auth drifted")
    for relative, expected in payload["files"].items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None and
                sha256_file(repo_root / relative) == expected,
                f"publication source hash drifted: {relative}")
    return sha256_file(path)


def validate_commit(repo_root: Path, authorization: dict[str, Any]) -> str:
    require(not subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip(), "worktree must be fully clean")
    for relative in SOURCE_FILES | {SOURCE_AUTH_FILE}:
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", relative],
            check=True, capture_output=True,
        )
        committed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            check=True, capture_output=True,
        ).stdout
        require(hashlib.sha256(committed).hexdigest() == sha256_file(repo_root / relative),
                f"publication source differs from HEAD: {relative}")
    commit = authorization["publisher_code_commit"]
    require(subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
    ).returncode == 0, "publisher commit is not an ancestor of HEAD")
    committed_script = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:bin/m33_safe_bridge_technical_publish.py"],
        check=True, capture_output=True,
    ).stdout
    require(hashlib.sha256(committed_script).hexdigest() ==
            sha256_file(repo_root / "bin/m33_safe_bridge_technical_publish.py"),
            "publisher script differs from authorized commit")
    return commit


def validate_base_policy(path: Path) -> None:
    require(sha256_file(path) == BASE_POLICY_SHA256, "base storage policy drifted")
    payload = load_json(path)["persistent_write_contract"]
    require(payload.get("append_only") is True and
            payload.get("object_creation_precondition") == "ifGenerationMatch=0" and
            payload.get("overwrite_forbidden") is True and payload.get("delete_forbidden") is True,
            "base storage policy is not append-only")


def local_sources(run_root: Path, repo_root: Path,
                  authorization: dict[str, Any]) -> dict[str, Path]:
    mapped = {relative: run_root / source for relative, source in RESULT_FILES.items()}
    mapped.update({relative: repo_root / source for relative, source in PROVENANCE_FILES.items()})
    require(set(mapped) == set(SOURCE_ORDER), "local source mapping drifted")
    for relative, path in mapped.items():
        require(path.is_file() and not path.is_symlink(), f"invalid source: {relative}")
        descriptor = authorization["artifacts"][relative]
        require(path.stat().st_size == descriptor["size_bytes"] and
                sha256_file(path) == descriptor["sha256"],
                f"source descriptor drifted: {relative}")
    return mapped


def validate_receipts(paths: dict[str, Path]) -> list[dict[str, Any]]:
    source_auth_path = paths["provenance/m33_safe_bridge_technical_kat_source_auth.json"]
    source_auth = load_json(source_auth_path)
    source_auth_sha = sha256_file(source_auth_path)
    verifier_sha = source_auth["independent_verifier_files"][
        "bin/m33_safe_bridge_technical_verify.py"]
    ephemeral: list[dict[str, Any]] = []
    for root in ("root17", "root18"):
        bridge_path = paths[f"{root}/safe_bridge_technical_kat.receipt.json"]
        bridge = load_json(bridge_path)
        require(bridge.get("status") ==
                "PASS_SAFE_BRIDGE_TECHNICAL_ROOT_KAT_ONLY_NON_CONSUMABLE" and
                bridge.get("root_label") == root and bridge.get("source_auth_sha256") == source_auth_sha and
                bridge.get("rss_gate_passed") is True and bridge.get("gcs_write") is False,
                f"bridge receipt is not publishable evidence: {root}")
        independent = load_json(paths[f"verification/{root}.independent_verify.receipt.json"])
        require(independent.get("status") == bridge["status"] and
                independent.get("root_label") == root and
                independent.get("bridge_receipt_sha256") == sha256_file(bridge_path) and
                independent.get("source_auth_sha256") == source_auth_sha and
                independent.get("verifier_source_sha256") == verifier_sha,
                f"independent receipt is not sealed: {root}")
        require(set(bridge["artifact_raw_sha256"]) == set(NPZ_NAMES) and
                set(bridge["artifact_semantic_sha256"]) == set(NPZ_NAMES),
                f"ephemeral NPZ inventory drifted: {root}")
        for name in NPZ_NAMES:
            path = run_root_path(bridge_path) / name
            require(path.is_file() and not path.is_symlink(), f"ephemeral NPZ missing: {root}/{name}")
            require(sha256_file(path) == bridge["artifact_raw_sha256"][name],
                    f"ephemeral NPZ hash drifted: {root}/{name}")
            ephemeral.append({
                "root_label": root,
                "name": name,
                "size_bytes": path.stat().st_size,
                "raw_sha256": bridge["artifact_raw_sha256"][name],
                "semantic_sha256": bridge["artifact_semantic_sha256"][name],
                "published": False,
            })
    require(len(ephemeral) == 8, "exactly eight ephemeral NPZ descriptors required")
    return ephemeral


def run_root_path(bridge_receipt: Path) -> Path:
    return bridge_receipt.parent


def atomic_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def active_account() -> str:
    for variable in FORBIDDEN_ENV:
        require(not os.environ.get(variable), f"credential override forbidden: {variable}")
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        check=True, capture_output=True, text=True,
    )
    accounts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(accounts == [PRINCIPAL], "active publisher account differs")
    return accounts[0]


def list_remote(prefix: str) -> dict[str, dict[str, Any]]:
    result = subprocess.run(
        ["gcloud", "storage", "ls", prefix, "--recursive", "--json"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        diagnostic = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
        require(result.returncode == 1 and diagnostic == EMPTY_PREFIX_MESSAGE,
                f"destination listing failed: {diagnostic}")
        return {}
    observed: dict[str, dict[str, Any]] = {}
    for entry in json.loads(result.stdout):
        if entry.get("type") == "prefix":
            continue
        metadata = entry.get("metadata")
        require(entry.get("type") == "cloud_object" and isinstance(metadata, dict),
                "invalid destination listing")
        uri = f"gs://{metadata['bucket']}/{metadata['name']}"
        require(uri.startswith(prefix) and uri not in observed, "destination escaped prefix")
        observed[uri] = metadata
    return observed


def describe(uri: str) -> dict[str, Any]:
    return json.loads(subprocess.run(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)


def verify_remote(uri: str, expected: dict[str, Any]) -> dict[str, Any]:
    metadata = describe(uri)
    generation = str(metadata["generation"])
    require(int(metadata["size"]) == expected["size_bytes"], f"remote size drifted: {uri}")
    with tempfile.TemporaryDirectory(prefix="m33-safe-bridge-reopen-") as temporary:
        reopened = Path(temporary) / "reopened"
        subprocess.run(["gcloud", "storage", "cp", f"{uri}#{generation}", str(reopened)], check=True)
        require(sha256_file(reopened) == expected["sha256"], f"remote SHA-256 drifted: {uri}")
    return {
        "uri": uri, "generation": generation, "size_bytes": expected["size_bytes"],
        "sha256": expected["sha256"], "crc32c_base64": metadata["crc32c_hash"],
        "md5_base64": metadata["md5_hash"],
    }


def publish_one(local: Path, uri: str, expected: dict[str, Any]) -> dict[str, Any]:
    subprocess.run(
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(local), uri], check=True,
    )
    return verify_remote(uri, expected)


def bucket_controls() -> dict[str, Any]:
    payload = json.loads(subprocess.run(
        ["gcloud", "storage", "buckets", "describe", "gs://teams-usp", "--format=json"],
        check=True, capture_output=True, text=True,
    ).stdout)
    require(payload.get("uniform_bucket_level_access") is True and
            payload.get("public_access_prevention") == "enforced",
            "bucket controls drifted")
    return {"uniform_bucket_level_access": True, "public_access_prevention": "enforced"}


def publish(*, run_root: Path, repo_root: Path, manifest_path: Path,
            receipt_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path, authorization_path = repo_root / CONFIG_FILE, repo_root / AUTH_FILE
    source_auth_path, base_policy_path = repo_root / SOURCE_AUTH_FILE, repo_root / BASE_POLICY_FILE
    config = load_config(config_path)
    authorization = load_authorization(authorization_path, sha256_file(config_path))
    publication_source_auth_sha = load_source_auth(source_auth_path, repo_root)
    validate_base_policy(base_policy_path)
    publisher_commit = validate_commit(repo_root, authorization)
    sources = local_sources(run_root, repo_root, authorization)
    ephemeral = validate_receipts(sources)
    require(manifest_path.parent == receipt_path.parent and manifest_path.name == MANIFEST and
            receipt_path.name == PUBLICATION_RECEIPT and manifest_path.parent.is_dir(),
            "generated publication paths are invalid")
    manifest_payload = {
        "schema_version": "1.0.0",
        "stage": "M33_SAFE_BRIDGE_TECHNICAL_KAT_MINIMAL_MANIFEST",
        "status": "PASS_MINIMAL_AUDIT_EVIDENCE_NON_CONSUMABLE",
        "run_id": RUN_ID,
        "publisher_code_commit": publisher_commit,
        "publication_source_auth_sha256": publication_source_auth_sha,
        "published_source_artifacts": authorization["artifacts"],
        "validated_ephemeral_artifacts": ephemeral,
        "npz_published": False,
        "pseudonymous_individual_artifacts_persisted": False,
        "consumable": False,
        "scientific_evidence": False,
        "truth": False,
        "materialize": False,
        "training": False,
        "ready": False,
    }
    manifest_descriptor = atomic_json(manifest_path, manifest_payload)
    local_order = {**sources, MANIFEST: manifest_path}
    descriptors = {**authorization["artifacts"], MANIFEST: manifest_descriptor}
    initial = list_remote(PREFIX)
    ordered_uris = tuple(PREFIX + relative for relative in FINAL_ORDER)
    require(set(initial) <= set(ordered_uris) and
            set(initial) == set(ordered_uris[:len(initial)]),
            "destination is not an exact resumable prefix")
    principal, controls = active_account(), bucket_controls()
    records: list[dict[str, Any]] = []
    verified: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_ORDER + (MANIFEST,):
        uri = PREFIX + relative
        record = (verify_remote(uri, descriptors[relative]) if uri in initial else
                  publish_one(local_order[relative], uri, descriptors[relative]))
        records.append(record)
        verified[uri] = record
    receipt_payload = {
        "schema_version": "1.0.0",
        "stage": "M33_SAFE_BRIDGE_TECHNICAL_MINIMAL_PUBLICATION",
        "status": "PASS_APPEND_ONLY_9_OBJECTS_REOPENED_NON_CONSUMABLE",
        "run_id": RUN_ID,
        "publisher_code_commit": publisher_commit,
        "publication_source_auth_sha256": publication_source_auth_sha,
        "publisher_principal": principal,
        "bucket_controls": controls,
        "destination_prefix": PREFIX,
        "objects_before_receipt": records,
        "object_count_before_receipt": 8,
        "final_object_count": 9,
        "npz_published": False,
        "pseudonymous_individual_artifacts_persisted": False,
        "source_code_published": False,
        "input_assets_published": False,
        "truth_published": False,
        "ready_published": False,
        "consumable": False,
        "scientific_evidence": False,
        "materialize": False,
        "training": False,
    }
    receipt_descriptor = atomic_json(receipt_path, receipt_payload)
    receipt_uri = PREFIX + PUBLICATION_RECEIPT
    receipt_record = (verify_remote(receipt_uri, receipt_descriptor) if receipt_uri in initial else
                      publish_one(receipt_path, receipt_uri, receipt_descriptor))
    verified[receipt_uri] = receipt_record
    expected = {PREFIX + name: descriptor for name, descriptor in {
        **descriptors, PUBLICATION_RECEIPT: receipt_descriptor,
    }.items()}
    listed = list_remote(PREFIX)
    require(set(listed) == set(expected) == set(verified) and len(listed) == 9,
            "final remote inventory is not exactly nine objects")
    result = dict(receipt_payload)
    result["publication_receipt"] = receipt_record
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(
        run_root=args.run_root, repo_root=args.repo_root,
        manifest_path=args.manifest, receipt_path=args.receipt,
    ), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
