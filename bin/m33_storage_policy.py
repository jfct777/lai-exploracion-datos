#!/usr/bin/env python3
"""Fail-closed storage namespace checks for prospective M33 executions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


POLICY_STATUS = "CONTRACT_ONLY_NO_REAL_ASSET_READ_NO_PERSISTENT_WRITE"
PROJECT_WRITE_ROOT = "gs://teams-usp/frank/lai-exploracion-datos/"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
CANONICAL_GCS_URI_RE = re.compile(r"^gs://[a-z0-9][a-z0-9.-]*/[A-Za-z0-9._/-]+/?$")
READ_DESCRIPTOR_KEYS = {
    "logical_id", "gcs_uri", "gcs_generation", "size_bytes", "sha256_raw", "crc32c",
}
TOP_LEVEL_KEYS = {
    "schema_version", "stage", "status", "project_write_root", "lab_data_policy",
    "persistent_namespaces", "namespace_amendment", "persistent_write_contract",
    "ephemeral_scratch", "google_batch", "oci", "execution_authorization",
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


def exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    require(actual == expected,
            f"{where} keys differ; missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def load_policy(path: str | Path) -> dict[str, Any]:
    payload = loads_strict_json(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "storage policy must be a JSON object")
    validate_policy(payload)
    return payload


def validate_policy(policy: Mapping[str, Any]) -> None:
    exact_keys(policy, TOP_LEVEL_KEYS, "storage policy")
    require(policy.get("schema_version") == "1.0.0", "storage policy schema drift")
    require(policy.get("stage") == "M33_STORAGE_NAMESPACE_POLICY", "storage stage drift")
    require(policy.get("status") == POLICY_STATUS, "storage policy authorization drift")
    require(policy.get("project_write_root") == PROJECT_WRITE_ROOT, "project write root drift")

    namespaces = policy.get("persistent_namespaces")
    require(isinstance(namespaces, dict) and namespaces, "persistent namespaces missing")
    expected = {"runs", "work", "logs", "manifests", "software_receipts"}
    require(set(namespaces) == expected, "persistent namespace inventory drift")
    values = list(namespaces.values())
    require(len(values) == len(set(values)), "persistent namespaces overlap")
    for left_index, left in enumerate(values):
        for right in values[left_index + 1:]:
            require(not left.startswith(right) and not right.startswith(left),
                    "persistent namespaces are nested")
    for name, suffix in namespaces.items():
        require(isinstance(suffix, str) and suffix.endswith("/"),
                f"namespace must end in slash: {name}")
        _parse_gcs_uri(PROJECT_WRITE_ROOT + suffix, allow_prefix=True)

    lab = policy.get("lab_data_policy", {})
    exact_keys(lab, {
        "persistent_write_forbidden", "approved_read_buckets",
        "read_descriptor_requirement", "unregistered_input",
    }, "lab data policy")
    require(lab.get("persistent_write_forbidden") is True, "lab writes are not forbidden")
    buckets = lab.get("approved_read_buckets")
    require(buckets == ["projects-usp", "frozen-data-br"], "approved read buckets drift")
    require(lab.get("read_descriptor_requirement") ==
            "exact_logical_id_uri_generation_size_sha256_crc32c",
            "read descriptor contract drift")
    require(lab.get("unregistered_input") == "STOP", "unregistered inputs are not fail-closed")

    amendment = policy.get("namespace_amendment", {})
    exact_keys(amendment, {
        "preserve_prior_objects_in_place", "prior_objects_become_read_only_inputs",
        "supersedes_only", "development_output_prefix_template",
    }, "namespace amendment")
    require(amendment["preserve_prior_objects_in_place"] is True and
            amendment["prior_objects_become_read_only_inputs"] is True,
            "historical object preservation drift")
    require(amendment["development_output_prefix_template"].startswith(
        PROJECT_WRITE_ROOT + "runs/m33/"), "development output template escapes project root")
    require(len(amendment["supersedes_only"]) == 2, "namespace supersession inventory drift")
    for item in amendment["supersedes_only"]:
        exact_keys(item, {"contract_sha256", "json_pointer"}, "superseded field")
        require(re.fullmatch(r"[0-9a-f]{64}", item["contract_sha256"]) is not None,
                "superseded contract hash is invalid")
        require(item["json_pointer"] in {
            "/canonical_prefix_template", "/identity/output_prefix_template"
        }, "unexpected field superseded by storage policy")

    write = policy.get("persistent_write_contract", {})
    exact_keys(write, {
        "append_only", "object_creation_precondition", "ready_is_last",
        "reopen_generation_and_sha256", "delete_forbidden", "overwrite_forbidden",
        "run_id_pattern",
    }, "persistent write contract")
    require(write.get("append_only") is True, "append-only policy missing")
    require(write.get("object_creation_precondition") == "ifGenerationMatch=0",
            "atomic object creation precondition drift")
    require(write.get("ready_is_last") is True, "READY is not terminal")
    require(write.get("reopen_generation_and_sha256") is True,
            "published objects are not reopened and authenticated")
    require(write.get("delete_forbidden") is True, "delete is not forbidden")
    require(write.get("overwrite_forbidden") is True, "overwrite is not forbidden")
    require(write.get("run_id_pattern") == RUN_ID_RE.pattern, "run ID pattern drift")

    batch = policy.get("google_batch", {})
    exact_keys(batch, {
        "region", "resource_labels", "real_run_service_account",
        "service_account_must_have_lab_object_viewer_only", "service_account_write_scope",
    }, "Google Batch policy")
    require(batch.get("region") == "us-central1", "Batch region drift")
    require(batch.get("resource_labels") == {"team": "frank"}, "Batch cost label drift")
    require(batch.get("real_run_service_account") ==
            "BLOCKED_PENDING_DEDICATED_NON_EDITOR_SERVICE_ACCOUNT",
            "real-run service account was opened without a policy amendment")
    require(batch.get("service_account_must_have_lab_object_viewer_only") is True,
            "lab input service-account policy drift")
    require(batch.get("service_account_write_scope") == "exact_project_write_root_only",
            "service-account write scope drift")

    scratch = policy.get("ephemeral_scratch", {})
    exact_keys(scratch, {
        "allowed_roots", "must_not_be_final_artifact", "must_be_removed_after_verified_publication"
    }, "ephemeral scratch policy")
    require(scratch == {
        "allowed_roots": ["/tmp"],
        "must_not_be_final_artifact": True,
        "must_be_removed_after_verified_publication": True,
    }, "ephemeral scratch policy drift")

    oci = policy.get("oci", {})
    exact_keys(oci, {
        "repository_prefix", "runtime_reference_must_use_digest", "receipt_namespace"
    }, "OCI policy")
    require(oci == {
        "repository_prefix": "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-",
        "runtime_reference_must_use_digest": True,
        "receipt_namespace": "software_receipts",
    }, "OCI policy drift")

    auth = policy.get("execution_authorization", {})
    require(auth == {
        "contract_tests": True,
        "fixture_tests": True,
        "create_project_prefix": False,
        "real_asset_read": False,
        "derived_index_write": False,
        "materialization": False,
        "training": False,
    }, "storage policy authorizes a real execution")


def _parse_gcs_uri(uri: str, *, allow_prefix: bool = False) -> tuple[str, str]:
    require(isinstance(uri, str) and uri, "GCS URI is empty")
    require(uri.isascii(), "non-ASCII character in GCS URI")
    require(uri.startswith("gs://"), "GCS URI is not canonical")
    require(all(33 <= ord(char) <= 126 for char in uri), "whitespace or control in GCS URI")
    require(CANONICAL_GCS_URI_RE.fullmatch(uri) is not None,
            "unsupported or ambiguous character in GCS URI")
    parsed = urlsplit(uri)
    require(parsed.scheme == "gs", "GCS URI must use gs://")
    require(parsed.netloc and parsed.netloc == parsed.netloc.lower(), "invalid GCS bucket")
    require(parsed.username is None and parsed.password is None and parsed.port is None,
            "authority is forbidden in GCS URI")
    require(parsed.query == "" and parsed.fragment == "", "query or fragment in GCS URI")
    require(parsed.path.startswith("/") and parsed.path != "/", "GCS object path is empty")
    require("//" not in parsed.path, "empty GCS path component")
    parts = parsed.path[1:].split("/")
    if parts[-1] == "":
        require(allow_prefix, "persistent write URI must name an object")
        parts = parts[:-1]
    require(parts and all(part not in {"", ".", ".."} for part in parts),
            "relative or empty GCS path component")
    return parsed.netloc, "/".join(parts) + ("/" if allow_prefix and uri.endswith("/") else "")


def _validate_read_location(uri: str, policy: Mapping[str, Any]) -> str:
    bucket, _ = _parse_gcs_uri(uri)
    own_bucket = _parse_gcs_uri(policy["project_write_root"], allow_prefix=True)[0]
    allowed = set(policy["lab_data_policy"]["approved_read_buckets"]) | {own_bucket}
    require(bucket in allowed, "read URI bucket is not approved")
    if bucket == own_bucket:
        require(uri.startswith(policy["project_write_root"]),
                "read URI escapes the project prefix")
    return uri


def validate_read_descriptor(
    descriptor: Mapping[str, Any],
    authorized_descriptor: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    exact_keys(descriptor, READ_DESCRIPTOR_KEYS, "read descriptor")
    exact_keys(authorized_descriptor, READ_DESCRIPTOR_KEYS, "authorized read descriptor")
    require(descriptor == authorized_descriptor, "read descriptor is not exactly authorized")
    _validate_read_location(str(descriptor["gcs_uri"]), policy)
    require(isinstance(descriptor["logical_id"], str) and descriptor["logical_id"],
            "read logical ID is empty")
    require(isinstance(descriptor["gcs_generation"], str) and
            re.fullmatch(r"[1-9][0-9]*", descriptor["gcs_generation"]) is not None,
            "read generation is not immutable")
    require(isinstance(descriptor["size_bytes"], int) and
            not isinstance(descriptor["size_bytes"], bool) and descriptor["size_bytes"] > 0,
            "read size is invalid")
    require(isinstance(descriptor["sha256_raw"], str) and
            re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256_raw"]) is not None,
            "read SHA-256 is invalid")
    require(isinstance(descriptor["crc32c"], str) and
            re.fullmatch(r"[A-Za-z0-9+/]{6}==", descriptor["crc32c"]) is not None,
            "read CRC32C is invalid")
    return descriptor


def validate_write_uri(
    uri: str,
    namespace: str,
    policy: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> str:
    bucket, object_name = _parse_gcs_uri(uri)
    root_bucket, root_prefix = _parse_gcs_uri(policy["project_write_root"], allow_prefix=True)
    require(bucket == root_bucket, "persistent write bucket is not the project bucket")
    suffix = policy["persistent_namespaces"].get(namespace)
    require(isinstance(suffix, str), f"unknown persistent namespace: {namespace}")
    if namespace == "software_receipts":
        require(run_id is None, "software receipts must not use a run ID")
        required_prefix = root_prefix + suffix
    else:
        require(run_id is not None, "run-scoped persistent write is missing run ID")
        validate_run_id(run_id)
        required_prefix = root_prefix + suffix + run_id + "/"
    require(object_name.startswith(required_prefix) and object_name != required_prefix,
            f"persistent write escapes namespace: {namespace}")
    return uri


def validate_publication_order(object_names: list[str]) -> list[str]:
    require(object_names and len(object_names) == len(set(object_names)),
            "publication is empty or contains duplicates")
    basenames = [name.rstrip("/").rsplit("/", 1)[-1] for name in object_names]
    ready_positions = [index for index, name in enumerate(basenames) if name == "READY"]
    require(ready_positions in ([], [len(object_names) - 1]), "READY must be written last")
    return object_names


def require_real_execution_authorized(policy: Mapping[str, Any]) -> None:
    auth = policy["execution_authorization"]
    require(auth["real_asset_read"] is True and auth["derived_index_write"] is True,
            "real M33 execution remains blocked by the storage policy")


def validate_run_id(run_id: str) -> str:
    require(RUN_ID_RE.fullmatch(run_id) is not None, "invalid M33 run ID")
    return run_id


def expected_persistent_sinks(run_id: str, policy: Mapping[str, Any]) -> dict[str, str]:
    validate_run_id(run_id)
    root = policy["project_write_root"]
    namespaces = policy["persistent_namespaces"]
    sinks = {
        "results": f"{root}{namespaces['runs']}{run_id}/",
        "work": f"{root}{namespaces['work']}{run_id}/",
        "logs": f"{root}{namespaces['logs']}{run_id}/",
        "manifests": f"{root}{namespaces['manifests']}{run_id}/",
    }
    for name, uri in sinks.items():
        _parse_gcs_uri(uri, allow_prefix=True)
        namespace = "runs" if name == "results" else name
        root_bucket, root_prefix = _parse_gcs_uri(root, allow_prefix=True)
        bucket, object_name = _parse_gcs_uri(uri, allow_prefix=True)
        require(bucket == root_bucket and
                object_name.startswith(root_prefix + namespaces[namespace]),
                f"derived sink escapes policy: {name}")
    return sinks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--read-descriptor", action="append", default=[])
    parser.add_argument("--authorized-read-descriptor", action="append", default=[])
    parser.add_argument("--write", action="append", nargs=2, metavar=("NAMESPACE", "URI"), default=[])
    parser.add_argument("--run-id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = load_policy(args.policy)
    require(len(args.read_descriptor) == len(args.authorized_read_descriptor),
            "each read descriptor requires one exact authorized descriptor")
    for actual_path, authorized_path in zip(
        args.read_descriptor, args.authorized_read_descriptor, strict=True
    ):
        actual = loads_strict_json(Path(actual_path).read_text(encoding="utf-8"))
        authorized = loads_strict_json(Path(authorized_path).read_text(encoding="utf-8"))
        validate_read_descriptor(actual, authorized, policy)
    validated_write_uris = []
    for namespace, uri in args.write:
        validate_write_uri(uri, namespace, policy,
                           run_id=None if namespace == "software_receipts" else args.run_id)
        validated_write_uris.append(uri)
    if validated_write_uris:
        validate_publication_order(validated_write_uris)
    sinks = expected_persistent_sinks(args.run_id, policy) if args.run_id else {}
    print(json.dumps({"status": "PASS_STORAGE_POLICY_CONTRACT_ONLY", "sinks": sinks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
