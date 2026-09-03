#!/usr/bin/env python3
"""Create an immutable provenance binding for M36 materialization 20260903a.

The binder is deliberately additive.  It reads the original materialization
receipt and the evidence that was recovered after execution, but never edits
any of them.  The output is created exclusively and therefore cannot replace
an existing binding.

The invariance proof is a JSON object with the exact fields validated by
``validate_invariance_proof`` below.  It binds the executed and corrected
manifests, the executed source, the empty locus-metadata sentinel, and the
published target hash for each zero-negative ratio.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


class BindingError(ValueError):
    """Raised when M36 provenance evidence is incomplete or inconsistent."""


RUN_ID = "m36-cora-chr22-materialize-20260903a"
STAGE = "M36_CORA_BIND_MATERIALIZATION"
STATUS = "PASS_BOUND_PROVENANCE"

BASE_FILENAMES = {
    "loci": "m36_cora_loci.tsv",
    "carriers": "m36_cora_carriers.tsv",
    "missing": "m36_cora_missing.tsv",
    "covariates": "m36_cora_covariates.tsv",
    "components": "m36_cora_components.tsv",
    "targets": "m36_cora_external_targets.tsv",
    "materialization_receipt": "m36_cora_materialization_receipt.json",
}
SENSITIVITY_FILENAMES = {
    "targets_zero3": "m36_cora_external_targets_zero3.tsv",
    "targets_zero5": "m36_cora_external_targets_zero5.tsv",
}
PUBLISHED_FILENAMES = {**BASE_FILENAMES, **SENSITIVITY_FILENAMES}
RECEIPT_ARTIFACTS = tuple(name for name in BASE_FILENAMES if name != "materialization_receipt")
EXPECTED_PUBLISHED_PREFIX = (
    "gs://teams-usp/frank/lai-exploracion-datos/runs/"
    "m36-cora-chr22-materialize-20260903a/m36_cora_set/"
)
EXPECTED_PUBLISHED_METADATA = {
    "loci": {
        "generation": "1788397908389369", "size_bytes": 27250845,
        "crc32c_base64": "+UhZ7Q==",
        "sha256": "c6e1c5b4b4d66ca7de3d82502472cd152022e7541ccd4846439047b4e9a144bb",
    },
    "carriers": {
        "generation": "1788397908315588", "size_bytes": 39445167,
        "crc32c_base64": "tqsPMA==",
        "sha256": "81c4901d542896367c90684c028c0c16fb50ceae184a69d65e291bde0598a53d",
    },
    "missing": {
        "generation": "1788397908317755", "size_bytes": 134580,
        "crc32c_base64": "ba9xPQ==",
        "sha256": "220655fac9a0e4035027f7c0e057a3e83f951b1928897f4c2d1366ec10b0799e",
    },
    "covariates": {
        "generation": "1788397908326513", "size_bytes": 152916,
        "crc32c_base64": "x56YAg==",
        "sha256": "3a63eb5b35750fc03ef750759a5c15408422120a2d1f29770a76ad41e80d2847",
    },
    "components": {
        "generation": "1788397908351355", "size_bytes": 44728,
        "crc32c_base64": "o133GQ==",
        "sha256": "e06bce8f0790e226189d4258981c8a567ca431972217bf8fb2200208da52f36d",
    },
    "targets": {
        "generation": "1788397908380362", "size_bytes": 883762,
        "crc32c_base64": "ORSfrg==",
        "sha256": "d102891a38f3c8e5130711c477493e34ee966ffe1d338a31df7731ff296e2275",
    },
    "materialization_receipt": {
        "generation": "1788397908368958", "size_bytes": 4886,
        "crc32c_base64": "rqmUSw==",
        "sha256": "e3445e3eda666dc0f717ec0ab290001f22e6c0d57d2013a6c68105d696dc8a5f",
    },
    "targets_zero3": {
        "generation": "1788397908360583", "size_bytes": 1320389,
        "crc32c_base64": "CvfLPA==",
        "sha256": "b20cc9c83037eeb781a1c1005628eb44b3b65194d8eb59f9f7fbcd70517600fb",
    },
    "targets_zero5": {
        "generation": "1788397908409460", "size_bytes": 1756479,
        "crc32c_base64": "4wHcwg==",
        "sha256": "8dad1b6e290698d9eef6d7781497b7531c7f6a3e5197d3c49d34749ce0fc3a2f",
    },
}

EXPECTED_EXECUTED_MANIFEST_SHA256 = "58ca47dc547290751360c7bd07624853ea22b3998aa614b3ac6bd1d81bac4234"
EXPECTED_CORRECTED_MANIFEST_SHA256 = "ecc44ff82c63559d4b9287d51f781358c3fb1b8db95a80532bd9ecf17e494189"
EXPECTED_LOCUS_SENTINEL_SHA256 = "183db7be694e740f48db5d91d6e374c992ed119a67e5f06461dd190fb555d87d"
EXPECTED_EXECUTED_CODE_SHA256 = "199273da2deb91cbe126eb7c99fc5bc213ffbbbb5288357370928bb853f1f578"
EXPECTED_INVARIANCE_COMPARISON_SHA256 = {
    "1": "908cdaf99c454a0961b0a3f4bde5f3cd1bf5780ad89dedefcee270eb6fd83128",
    "3": "f3f5c94188fa0bddc5af8bb2060c0de79ab1d5ed80a8c41be2f9fa3f4fbd1654",
    "5": "0c5f231e57ef8b3ba71ee1df20596b4e1ada2776c0628f074a3be57e2256b29a",
}
EXPECTED_EXECUTED_MAPPING = {
    "anc1.gapfilled_ibd": "AFR",
    "anc2.gapfilled_ibd": "EUR",
    "anc3.gapfilled_ibd": "NAM",
}
EXPECTED_CORRECTED_MAPPING = {
    "anc1.gapfilled_ibd": "EUR",
    "anc2.gapfilled_ibd": "NAM",
    "anc3.gapfilled_ibd": "AFR",
}
EXPECTED_SEGMENT_FILES = list(EXPECTED_EXECUTED_MAPPING)
EXPECTED_LOCUS_SENTINEL = b"chrom\tposition\tref\talt\n"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TARGET_FIELDS = [
    "sample_i", "sample_j", "target_chrom", "target_source", "target_stratum",
    "target_cm", "target_positive", "target",
]


def build_crc32c_table() -> tuple[int, ...]:
    table = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        table.append(crc)
    return tuple(table)


CRC32C_TABLE = build_crc32c_table()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crc32c_base64(data: bytes) -> str:
    """Return the big-endian base64 CRC32C representation used by GCS."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    value = (~crc) & 0xFFFFFFFF
    return base64.b64encode(value.to_bytes(4, "big")).decode("ascii")


def read_input(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(),
            f"missing or non-regular {label}: {path}")
    data = path.read_bytes()
    return data, {"path": str(path), "size_bytes": len(data), "sha256": sha256_bytes(data)}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_value(data: bytes, label: str) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise BindingError(f"invalid {label} JSON constant: {value}")

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BindingError(f"invalid {label} JSON: {error}") from error


def parse_json_object(data: bytes, label: str) -> dict[str, Any]:
    value = parse_json_value(data, label)
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} must be a lowercase SHA-256")
    return value


def positive_size(value: Any, label: str) -> int:
    require(not isinstance(value, bool), f"{label} must be a positive integer")
    if isinstance(value, int):
        result = value
    else:
        require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None,
                f"{label} must be a canonical positive decimal")
        result = int(value)
    require(result > 0, f"{label} must be positive")
    return result


def normalized_descriptor(name: str, value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), f"published descriptor {name} must be an object")
    expected_fields = {"uri", "generation", "size_bytes", "crc32c_base64", "sha256"}
    require(set(value) == expected_fields,
            f"published descriptor {name} fields must be exactly {sorted(expected_fields)}")

    uri = value["uri"]
    require(uri == EXPECTED_PUBLISHED_PREFIX + PUBLISHED_FILENAMES[name],
            f"published descriptor {name} URI differs from the canonical 20260903a object")
    generation = value["generation"]
    require(isinstance(generation, str) and re.fullmatch(r"[1-9][0-9]*", generation) is not None,
            f"published descriptor {name} generation must be a canonical positive decimal string")
    size = positive_size(value["size_bytes"], f"published descriptor {name} size_bytes")
    digest = require_sha256(value["sha256"], f"published descriptor {name} sha256")

    crc32c = value["crc32c_base64"]
    require(isinstance(crc32c, str) and crc32c,
            f"published descriptor {name} crc32c_base64 is missing")
    try:
        decoded_crc = base64.b64decode(crc32c, validate=True)
    except (ValueError, binascii.Error) as error:
        raise BindingError(f"published descriptor {name} crc32c is not valid base64") from error
    require(len(decoded_crc) == 4 and base64.b64encode(decoded_crc).decode("ascii") == crc32c,
            f"published descriptor {name} crc32c is not canonical CRC32C base64")
    return {
        "uri": uri,
        "generation": generation,
        "size_bytes": size,
        "crc32c_base64": crc32c,
        "sha256": digest,
    }


def validate_published_descriptors(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    missing_sensitivity = set(SENSITIVITY_FILENAMES) - set(value)
    require(not missing_sensitivity,
            f"missing required sensitivity descriptor(s): {sorted(missing_sensitivity)}")
    require(set(value) == set(PUBLISHED_FILENAMES),
            f"published descriptors must contain exactly all 9 M36 objects; got {sorted(value)}")
    normalized = {name: normalized_descriptor(name, value[name]) for name in PUBLISHED_FILENAMES}
    for name, descriptor in normalized.items():
        for field, expected_value in EXPECTED_PUBLISHED_METADATA[name].items():
            require(descriptor[field] == expected_value,
                    f"published descriptor {name} {field} differs from audited 20260903a metadata")
    uris = [descriptor["uri"] for descriptor in normalized.values()]
    require(len(uris) == len(set(uris)), "published descriptor GCS URIs must be unique")
    prefixes = {uri.rsplit("/", 1)[0] for uri in uris}
    require(len(prefixes) == 1, "all 9 published objects must share one GCS prefix")
    return normalized


def run_gcloud(arguments: list[str], *, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["gcloud", "storage", *arguments]
    try:
        return subprocess.run(command, check=True, capture_output=capture_output, text=True)
    except subprocess.CalledProcessError as error:
        diagnostic = "\n".join(
            item.strip() for item in (error.stdout, error.stderr)
            if isinstance(item, str) and item.strip()
        )
        raise BindingError(f"read-only GCS command failed: {diagnostic or command}") from error


def list_remote_inventory(prefix: str) -> set[str]:
    result = run_gcloud(["ls", prefix, "--recursive", "--json"])
    payload = parse_json_value(result.stdout.encode("utf-8"), "GCS inventory")
    require(isinstance(payload, list), "GCS inventory must be a JSON array")
    observed: set[str] = set()
    for entry in payload:
        require(isinstance(entry, dict), "GCS inventory entry must be an object")
        if entry.get("type") == "prefix":
            continue
        metadata = entry.get("metadata")
        require(entry.get("type") == "cloud_object" and isinstance(metadata, dict),
                "GCS inventory entry is not a cloud object")
        bucket = metadata.get("bucket")
        name = metadata.get("name")
        require(isinstance(bucket, str) and isinstance(name, str),
                "GCS inventory entry lacks bucket/name")
        uri = f"gs://{bucket}/{name}"
        require(uri.startswith(prefix) and uri not in observed,
                "GCS inventory escaped the prefix or repeated an object")
        observed.add(uri)
    expected = {prefix + filename for filename in PUBLISHED_FILENAMES.values()}
    require(observed == expected, "GCS inventory differs from the exact nine-object publication")
    return observed


def describe_remote_object(uri: str) -> dict[str, Any]:
    result = run_gcloud(["objects", "describe", uri, "--format=json"])
    return parse_json_object(result.stdout.encode("utf-8"), f"GCS description for {uri}")


def validate_remote_metadata(metadata: dict[str, Any], descriptor: dict[str, Any], name: str) -> None:
    require(str(metadata.get("generation")) == descriptor["generation"],
            f"remote generation differs for {name}")
    try:
        remote_size = int(metadata.get("size"))
    except (TypeError, ValueError) as error:
        raise BindingError(f"remote size is invalid for {name}") from error
    require(remote_size == descriptor["size_bytes"], f"remote size differs for {name}")
    require(metadata.get("crc32c_hash") == descriptor["crc32c_base64"],
            f"remote CRC32C differs for {name}")


def parse_target_rows(data: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BindingError(f"{label} is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    require(reader.fieldnames == TARGET_FIELDS, f"{label} target schema differs")
    rows = list(reader)
    require(rows, f"{label} is empty")
    require(all(None not in row and set(row) == set(TARGET_FIELDS) for row in rows),
            f"{label} contains a malformed row")
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        require(row["target_positive"] in {"0", "1"},
                f"{label} target_positive is not binary")
        pair = (row["sample_i"], row["sample_j"])
        require(pair[0] < pair[1] and pair not in seen_pairs,
                f"{label} contains a noncanonical or duplicate pair")
        seen_pairs.add(pair)
        require(row["target_chrom"] == "outside_chr22_total"
                and row["target_source"] == "asibd_refined_ibd_gnomix_stratified_exploratory"
                and row["target_stratum"] in {"between_component", "within_component"},
                f"{label} contains an invalid target identity")
        try:
            target_cm = float(row["target_cm"])
            target = float(row["target"])
        except ValueError as error:
            raise BindingError(f"{label} contains a nonnumeric target") from error
        if row["target_positive"] == "0":
            require(target_cm == 0.0 and target == 0.0,
                    f"{label} sampled-zero row has a nonzero target")
        else:
            require(target_cm > 0.0 and target > 0.0,
                    f"{label} positive row has a nonpositive target")
    return rows


def validate_target_sensitivity(
    receipt: dict[str, Any], target_payloads: dict[str, bytes],
) -> dict[str, Any]:
    receipt_balance = receipt.get("zero_negative_sampling")
    require(isinstance(receipt_balance, dict) and set(receipt_balance) == {"1", "3", "5"},
            "original receipt lacks exact zero-negative sampling evidence")
    logical_name = {"1": "targets", "3": "targets_zero3", "5": "targets_zero5"}
    positive_reference: list[tuple[str, ...]] | None = None
    summary: dict[str, Any] = {}
    for ratio in ("1", "3", "5"):
        rows = parse_target_rows(target_payloads[logical_name[ratio]], logical_name[ratio])
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "zero": 0})
        positive_rows: list[tuple[str, ...]] = []
        positive_pairs: set[tuple[str, str]] = set()
        zero_pairs: set[tuple[str, str]] = set()
        for row in rows:
            polarity = "positive" if row["target_positive"] == "1" else "zero"
            counts[row["target_stratum"]][polarity] += 1
            pair = (row["sample_i"], row["sample_j"])
            (positive_pairs if polarity == "positive" else zero_pairs).add(pair)
            if polarity == "positive":
                positive_rows.append(tuple(row[field] for field in TARGET_FIELDS))
        require(positive_pairs.isdisjoint(zero_pairs),
                f"ratio {ratio} uses a pair as both positive and sampled zero")
        if positive_reference is None:
            positive_reference = positive_rows
        else:
            require(positive_rows == positive_reference,
                    f"ratio {ratio} positive target rows differ from ratio 1")

        expected_ratio_balance = receipt_balance[ratio]
        require(isinstance(expected_ratio_balance, dict)
                and set(expected_ratio_balance) == {"between_component", "within_component"},
                f"receipt ratio {ratio} balance strata differ")
        for stratum in ("between_component", "within_component"):
            expected_stratum = expected_ratio_balance[stratum]
            require(isinstance(expected_stratum, dict)
                    and expected_stratum.get("positive") == counts[stratum]["positive"]
                    and expected_stratum.get("zero") == counts[stratum]["zero"]
                    and expected_stratum.get("requested_zero_to_positive_ratio") == int(ratio),
                    f"ratio {ratio} {stratum} balance differs from original receipt")
        summary[ratio] = {
            "row_count": len(rows),
            "positive_count": len(positive_pairs),
            "zero_count": len(zero_pairs),
            "positive_rows_identical_to_ratio1": True,
        }
    return summary


def verify_remote_publication(
    published: dict[str, dict[str, Any]],
    local_receipt: bytes,
    receipt: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    list_remote_inventory(EXPECTED_PUBLISHED_PREFIX)
    verified: dict[str, dict[str, Any]] = {}
    target_payloads: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="m36-cora-bind-reopen-") as temporary_name:
        temporary = Path(temporary_name)
        for name, descriptor in published.items():
            before = describe_remote_object(descriptor["uri"])
            validate_remote_metadata(before, descriptor, name)
            reopened = temporary / PUBLISHED_FILENAMES[name]
            run_gcloud(["cp", f"{descriptor['uri']}#{descriptor['generation']}", str(reopened)])
            require(reopened.is_file() and not reopened.is_symlink(),
                    f"exact-generation reopen failed for {name}")
            payload = reopened.read_bytes()
            require(len(payload) == descriptor["size_bytes"],
                    f"reopened size differs for {name}")
            require(sha256_bytes(payload) == descriptor["sha256"],
                    f"reopened SHA-256 differs for {name}")
            require(crc32c_base64(payload) == descriptor["crc32c_base64"],
                    f"reopened CRC32C differs for {name}")
            if name == "materialization_receipt":
                require(payload == local_receipt,
                        "reopened materialization receipt differs bytewise from the supplied original")
            if name in {"targets", "targets_zero3", "targets_zero5"}:
                target_payloads[name] = payload
            after = describe_remote_object(descriptor["uri"])
            validate_remote_metadata(after, descriptor, name)
            require(str(after.get("generation")) == str(before.get("generation")),
                    f"remote generation changed while reopening {name}")
            verified[name] = {**descriptor, "verification": "EXACT_GENERATION_REOPENED"}
    list_remote_inventory(EXPECTED_PUBLISHED_PREFIX)
    return verified, validate_target_sensitivity(receipt, target_payloads)


def parse_manifest(data: bytes, label: str) -> tuple[list[str], dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BindingError(f"{label} is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    require(reader.fieldnames == ["gnomix_ancestry", "segment_file"],
            f"{label} must have exactly gnomix_ancestry and segment_file columns")
    rows = list(reader)
    require(len(rows) == 3, f"{label} must contain exactly three ancestry rows")
    require(all(None not in row and set(row) == set(reader.fieldnames) for row in rows),
            f"{label} contains a malformed row")
    for row in rows:
        for field in reader.fieldnames:
            require(bool(row[field]) and row[field] == row[field].strip(),
                    f"{label} contains an empty or padded {field}")
    segment_files = [row["segment_file"] for row in rows]
    ancestries = [row["gnomix_ancestry"] for row in rows]
    require(len(segment_files) == len(set(segment_files)), f"{label} repeats a segment_file")
    require(set(ancestries) == {"AFR", "EUR", "NAM"},
            f"{label} ancestry labels must be exactly AFR, EUR, and NAM")
    return segment_files, {row["segment_file"]: row["gnomix_ancestry"] for row in rows}


def validate_executed_code(data: bytes) -> dict[str, Any]:
    require(sha256_bytes(data) == EXPECTED_EXECUTED_CODE_SHA256,
            "executed code hash differs from M36 20260903a")
    try:
        tree = ast.parse(data.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise BindingError(f"executed code is not parseable Python: {error}") from error
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "materialize_targets"]
    require(len(functions) == 1, "executed code must define materialize_targets exactly once")
    string_literals = {node.value for node in ast.walk(functions[0])
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    require("segment_file" in string_literals,
            "executed materialize_targets does not consume segment_file")
    require("gnomix_ancestry" not in string_literals,
            "executed materialize_targets consumes the corrected label field")
    return {
        "function": "materialize_targets",
        "consumed_manifest_field": "segment_file",
        "unconsumed_manifest_field": "gnomix_ancestry",
    }


def validate_invariance_proof(
    proof: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    published: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version", "stage", "status", "run_id", "method",
        "executed_manifest_sha256", "corrected_manifest_sha256",
        "executed_code_sha256", "locus_sentinel_sha256", "invariant_field",
        "label_field", "sensitivity_ratios", "ratio_comparisons",
    }
    require(set(proof) == expected_fields,
            f"invariance proof fields must be exactly {sorted(expected_fields)}")
    require(proof["schema_version"] == "1.0.0", "invariance proof schema version differs")
    require(proof["stage"] == "M36_CORA_MANIFEST_INVARIANCE_PROOF",
            "invariance proof stage differs")
    require(proof["status"] == "PASS_EXACT_INVARIANCE", "invariance proof did not pass")
    require(proof["run_id"] == RUN_ID, "invariance proof run differs")
    require(proof["method"] == "deterministic_synthetic_regression",
            "invariance proof method differs")
    require(proof["invariant_field"] == "segment_file" and proof["label_field"] == "gnomix_ancestry",
            "invariance proof fields do not describe the manifest correction")
    require(proof["sensitivity_ratios"] == [1, 3, 5],
            "invariance proof must cover sensitivity ratios 1, 3, and 5")

    for proof_field, evidence_name in (
        ("executed_manifest_sha256", "executed_manifest"),
        ("corrected_manifest_sha256", "corrected_manifest"),
        ("executed_code_sha256", "executed_code"),
        ("locus_sentinel_sha256", "locus_sentinel"),
    ):
        require_sha256(proof[proof_field], f"invariance proof {proof_field}")
        require(proof[proof_field] == evidence[evidence_name]["sha256"],
                f"invariance proof does not authenticate {evidence_name}")

    comparisons = proof["ratio_comparisons"]
    require(isinstance(comparisons, dict) and set(comparisons) == {"1", "3", "5"},
            "invariance proof ratio comparisons must be exactly 1, 3, and 5")
    published_name = {"1": "targets", "3": "targets_zero3", "5": "targets_zero5"}
    normalized: dict[str, Any] = {}
    for ratio in ("1", "3", "5"):
        comparison = comparisons[ratio]
        require(isinstance(comparison, dict)
                and set(comparison) == {"exact_equal", "comparison_sha256", "published_sha256"},
                f"invariance proof ratio {ratio} has an invalid comparison record")
        require(comparison["exact_equal"] is True,
                f"invariance proof ratio {ratio} is not exactly invariant")
        digest = require_sha256(comparison["comparison_sha256"],
                                f"invariance proof ratio {ratio} comparison_sha256")
        require(digest == EXPECTED_INVARIANCE_COMPARISON_SHA256[ratio],
                f"invariance proof ratio {ratio} comparison hash differs")
        published_digest = require_sha256(comparison["published_sha256"],
                                          f"invariance proof ratio {ratio} published_sha256")
        require(published_digest == published[published_name[ratio]]["sha256"],
                f"invariance proof ratio {ratio} does not bind its published target")
        normalized[ratio] = dict(comparison)
    return normalized


def bind_materialization(
    materialization_receipt: Path,
    published_descriptors: Path,
    executed_manifest: Path,
    corrected_manifest: Path,
    invariance_proof: Path,
    locus_sentinel: Path,
    executed_code: Path,
    out: Path,
) -> dict[str, Any]:
    """Validate all evidence and exclusively create one provenance binding."""
    require(not out.exists() and not out.is_symlink(),
            f"refusing to overwrite existing binding: {out}")

    paths = {
        "materialization_receipt": materialization_receipt,
        "published_descriptors": published_descriptors,
        "executed_manifest": executed_manifest,
        "corrected_manifest": corrected_manifest,
        "invariance_proof": invariance_proof,
        "locus_sentinel": locus_sentinel,
        "executed_code": executed_code,
    }
    raw: dict[str, bytes] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        raw[name], evidence[name] = read_input(path, name.replace("_", " "))

    receipt = parse_json_object(raw["materialization_receipt"], "materialization receipt")
    descriptor_document = parse_json_object(raw["published_descriptors"], "published descriptors")
    proof = parse_json_object(raw["invariance_proof"], "invariance proof")
    published = validate_published_descriptors(descriptor_document)

    require(receipt.get("stage") == "M36_CORA_MATERIALIZE"
            and receipt.get("status") == "MATERIALIZED_PASS"
            and receipt.get("synthetic") is False,
            "original materialization receipt is not a chainable real-data receipt")
    require(receipt.get("feature_schema") == "m36_factorized_sparse_v1"
            and receipt.get("external_target_schema") == "m36_external_common_pairs_log1p_v3_pair_total",
            "original materialization receipt schema differs from M36 20260903a")
    require(receipt.get("zero_negative_ratio_sensitivity") == [1, 3, 5],
            "original receipt does not declare zero3/zero5 sensitivity")
    receipt_inputs = receipt.get("input_descriptors")
    require(isinstance(receipt_inputs, dict) and set(receipt_inputs) == set(RECEIPT_ARTIFACTS),
            "original receipt must retain exactly its six base input_descriptors")

    verified_base_hashes: list[dict[str, str]] = []
    for name in RECEIPT_ARTIFACTS:
        local = receipt_inputs[name]
        require(isinstance(local, dict), f"original receipt descriptor {name} is invalid")
        require(local.get("uri") == BASE_FILENAMES[name]
                and local.get("generation") == "LOCAL_CHAIN",
                f"original receipt descriptor {name} is not its local materialization output")
        local_digest = require_sha256(local.get("sha256"), f"original receipt descriptor {name} sha256")
        require(published[name]["sha256"] == local_digest,
                f"published hash differs for {name}")
        verified_base_hashes.append({
            "name": name,
            "sha256": local_digest,
            "verified_against": f"original_receipt.input_descriptors.{name}.sha256",
        })

    receipt_descriptor = published["materialization_receipt"]
    require(receipt_descriptor["sha256"] == evidence["materialization_receipt"]["sha256"],
            "published hash differs for materialization_receipt")
    require(receipt_descriptor["size_bytes"] == evidence["materialization_receipt"]["size_bytes"],
            "published size differs for materialization_receipt")
    require(receipt_descriptor["crc32c_base64"] == crc32c_base64(raw["materialization_receipt"]),
            "published CRC32C differs for materialization_receipt")
    verified_base_hashes.append({
        "name": "materialization_receipt",
        "sha256": evidence["materialization_receipt"]["sha256"],
        "verified_against": "local materialization receipt bytes",
    })
    require(len(verified_base_hashes) == 7, "internal error: seven base hashes were not verified")

    source_inputs = receipt.get("source_input_descriptors")
    source_manifest = source_inputs.get("asibd_manifest") if isinstance(source_inputs, dict) else None
    require(isinstance(source_manifest, dict),
            "original receipt does not authenticate the executed asIBD manifest")
    require(source_manifest.get("sha256") == evidence["executed_manifest"]["sha256"],
            "executed manifest hash differs from original receipt")
    require(evidence["executed_manifest"]["sha256"] == EXPECTED_EXECUTED_MANIFEST_SHA256,
            "executed manifest hash differs from M36 20260903a")
    require(evidence["corrected_manifest"]["sha256"] == EXPECTED_CORRECTED_MANIFEST_SHA256,
            "corrected manifest hash differs from the audited correction")
    require(raw["locus_sentinel"] == EXPECTED_LOCUS_SENTINEL
            and evidence["locus_sentinel"]["sha256"] == EXPECTED_LOCUS_SENTINEL_SHA256,
            "locus sentinel must be the authenticated header-only file")

    executed_segments, executed_mapping = parse_manifest(raw["executed_manifest"], "executed manifest")
    corrected_segments, corrected_mapping = parse_manifest(raw["corrected_manifest"], "corrected manifest")
    require(executed_segments == EXPECTED_SEGMENT_FILES
            and executed_mapping == EXPECTED_EXECUTED_MAPPING,
            "executed manifest content differs from M36 20260903a")
    require(corrected_segments == executed_segments
            and corrected_mapping == EXPECTED_CORRECTED_MAPPING,
            "corrected manifest must preserve segment order and apply the audited labels")
    code_semantics = validate_executed_code(raw["executed_code"])
    ratio_comparisons = validate_invariance_proof(proof, evidence, published)
    remote_verified, target_sensitivity = verify_remote_publication(
        published, raw["materialization_receipt"], receipt,
    )

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": STAGE,
        "status": STATUS,
        "run_id": RUN_ID,
        "artifact_role": "NON_CONSUMABLE_PROVENANCE_ADDENDUM",
        "consumable_as_materialization_receipt": False,
        "scope": "technical provenance binding; not a scientific or material verdict",
        "original_receipt": {
            **evidence["materialization_receipt"],
            "stage": receipt["stage"],
            "status": receipt["status"],
            "synthetic": receipt["synthetic"],
        },
        "published_binding": {
            "descriptor_document": evidence["published_descriptors"],
            "gcs_prefix": EXPECTED_PUBLISHED_PREFIX,
            "base_hash_count": len(verified_base_hashes),
            "verified_base_hashes": verified_base_hashes,
            "base_descriptors": {name: published[name] for name in BASE_FILENAMES},
            "sensitivity_descriptors": {name: published[name] for name in SENSITIVITY_FILENAMES},
            "remote_verification": remote_verified,
            "target_sensitivity_validation": target_sensitivity,
        },
        "authenticated_inputs": evidence,
        "manifest_correction": {
            "segment_files": executed_segments,
            "executed_mapping": executed_mapping,
            "corrected_mapping": corrected_mapping,
            "equivalence": "EXACT_LABEL_INVARIANCE_FOR_EXECUTED_PAIR_TOTAL_TARGET",
            "code_semantics": code_semantics,
            "proof_stage": proof["stage"],
            "proof_status": proof["status"],
            "proof_method": proof["method"],
            "ratio_comparisons": ratio_comparisons,
        },
        "immutability": {
            "binding_creation": "EXCLUSIVE_CREATE_NO_OVERWRITE",
            "original_inputs_modified": False,
            "historical_receipt_shape_modified": False,
            "cloud_operations": "READ_ONLY_LIST_DESCRIBE_EXACT_GENERATION_REOPEN",
        },
        "limitations": [
            "The original receipt hashes six factorized outputs but not zero3/zero5; this addendum authenticates and binds their exact GCS generations and bytes.",
            "The invariance evidence establishes computational indifference to the corrected ancestry-label column; it is not a scientific result.",
            "This addendum is not a replacement input for existing M36 training consumers.",
        ],
    }

    serialized = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o400)
        os.link(temporary, out)
    except FileExistsError as error:
        raise BindingError(f"refusing to overwrite existing binding: {out}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--published-descriptors", required=True, type=Path)
    parser.add_argument("--executed-manifest", required=True, type=Path)
    parser.add_argument("--corrected-manifest", required=True, type=Path)
    parser.add_argument("--invariance-proof", required=True, type=Path)
    parser.add_argument("--locus-sentinel", required=True, type=Path)
    parser.add_argument("--executed-code", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = bind_materialization(
            args.materialization_receipt,
            args.published_descriptors,
            args.executed_manifest,
            args.corrected_manifest,
            args.invariance_proof,
            args.locus_sentinel,
            args.executed_code,
            args.out,
        )
    except (BindingError, OSError) as error:
        raise SystemExit(f"M36 materialization binding error: {error}") from error
    print(json.dumps({"stage": payload["stage"], "status": payload["status"], "out": str(args.out)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
