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
import csv
import hashlib
import io
import json
import re
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crc32c_base64(data: bytes) -> str:
    """Return the big-endian base64 CRC32C representation used by GCS."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    value = (~crc) & 0xFFFFFFFF
    return base64.b64encode(value.to_bytes(4, "big")).decode("ascii")


def read_input(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    require(path.is_file(), f"missing {label}: {path}")
    data = path.read_bytes()
    return data, {"path": str(path), "size": len(data), "sha256": sha256_bytes(data)}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BindingError(f"invalid {label} JSON: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} must be a lowercase SHA-256")
    return value


def positive_decimal(value: Any, label: str) -> int:
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
    expected_fields = {"uri", "generation", "size", "crc32c", "sha256"}
    require(set(value) == expected_fields,
            f"published descriptor {name} fields must be exactly {sorted(expected_fields)}")

    uri = value["uri"]
    require(isinstance(uri, str) and uri.startswith("gs://") and uri.rsplit("/", 1)[-1] == PUBLISHED_FILENAMES[name],
            f"published descriptor {name} has an invalid GCS URI or object name")
    generation = positive_decimal(value["generation"], f"published descriptor {name} generation")
    size = positive_decimal(value["size"], f"published descriptor {name} size")
    digest = require_sha256(value["sha256"], f"published descriptor {name} sha256")

    crc32c = value["crc32c"]
    require(isinstance(crc32c, str) and crc32c, f"published descriptor {name} crc32c is missing")
    try:
        decoded_crc = base64.b64decode(crc32c, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise BindingError(f"published descriptor {name} crc32c is not valid base64") from error
    require(len(decoded_crc) == 4 and base64.b64encode(decoded_crc).decode("ascii") == crc32c,
            f"published descriptor {name} crc32c is not canonical CRC32C base64")
    return {
        "uri": uri,
        "generation": str(generation),
        "size": size,
        "crc32c": crc32c,
        "sha256": digest,
    }


def validate_published_descriptors(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    missing_sensitivity = set(SENSITIVITY_FILENAMES) - set(value)
    require(not missing_sensitivity,
            f"missing required sensitivity descriptor(s): {sorted(missing_sensitivity)}")
    require(set(value) == set(PUBLISHED_FILENAMES),
            f"published descriptors must contain exactly all 9 M36 objects; got {sorted(value)}")
    normalized = {name: normalized_descriptor(name, value[name]) for name in PUBLISHED_FILENAMES}
    uris = [descriptor["uri"] for descriptor in normalized.values()]
    require(len(uris) == len(set(uris)), "published descriptor GCS URIs must be unique")
    prefixes = {uri.rsplit("/", 1)[0] for uri in uris}
    require(len(prefixes) == 1, "all 9 published objects must share one GCS prefix")
    return normalized


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
    require(not out.exists(), f"refusing to overwrite existing binding: {out}")

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
    require(receipt_descriptor["size"] == evidence["materialization_receipt"]["size"],
            "published size differs for materialization_receipt")
    require(receipt_descriptor["crc32c"] == crc32c_base64(raw["materialization_receipt"]),
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

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": STAGE,
        "status": STATUS,
        "run_id": RUN_ID,
        "scope": "technical provenance binding; not a scientific or material verdict",
        "original_receipt": {
            **evidence["materialization_receipt"],
            "stage": receipt["stage"],
            "status": receipt["status"],
            "synthetic": receipt["synthetic"],
        },
        "published_binding": {
            "descriptor_document": evidence["published_descriptors"],
            "gcs_prefix": next(iter({item["uri"].rsplit("/", 1)[0] for item in published.values()})),
            "base_hash_count": len(verified_base_hashes),
            "verified_base_hashes": verified_base_hashes,
            "base_descriptors": {name: published[name] for name in BASE_FILENAMES},
            "sensitivity_descriptors": {name: published[name] for name in SENSITIVITY_FILENAMES},
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
        },
        "limitations": [
            "The original receipt hashes six factorized outputs but not zero3/zero5; this receipt additively binds their immutable GCS descriptors.",
            "The invariance evidence establishes computational indifference to the corrected ancestry-label column; it is not a scientific result.",
        ],
    }

    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except FileExistsError as error:
        raise BindingError(f"refusing to overwrite existing binding: {out}") from error
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
