#!/usr/bin/env python3
"""Verify immutable M38B score artifacts and run the pinned decision script."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


class M38BFinalizerError(ValueError):
    """Raised when finalization inputs differ from the immutable run."""


EXPECTED_IDS = (
    "analytic_metrics",
    "analytic_receipt",
    "tcn_metrics",
    "tcn_receipt",
    "positive_metrics",
    "positive_receipt",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BFinalizerError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise M38BFinalizerError(f"non-finite JSON constant in {path}: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M38BFinalizerError(f"cannot read strict JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def parse_artifact(value: str) -> tuple[str, Path]:
    logical_id, separator, raw_path = value.partition("=")
    require(bool(separator) and logical_id in EXPECTED_IDS and bool(raw_path),
            f"invalid --artifact binding: {value}")
    return logical_id, Path(raw_path)


def validate_manifest(manifest: dict, artifact_paths: dict[str, Path], decide: Path) -> list[dict]:
    require(manifest.get("schema_version") == "1.0.0", "unexpected finalizer manifest schema")
    require(manifest.get("stage") == "M38B_FINALIZER_INPUTS", "unexpected finalizer manifest stage")
    source_run_uri = manifest.get("source_run_uri")
    require(
        isinstance(source_run_uri, str)
        and source_run_uri
        == "gs://teams-usp/frank/lai-exploracion-datos/runs/m38b-r0-oof-models-20260903c",
        "finalizer source run is not the immutable M38B run c",
    )
    records = manifest.get("artifacts")
    require(isinstance(records, list), "finalizer artifacts must be a list")
    require(len(records) == len(EXPECTED_IDS), "finalizer must bind exactly six score artifacts")
    by_id: dict[str, dict] = {}
    for record in records:
        require(isinstance(record, dict), "finalizer artifact record must be an object")
        require(set(record) == {"logical_id", "uri", "basename", "sha256"},
                "finalizer artifact record fields differ")
        logical_id = record["logical_id"]
        require(logical_id in EXPECTED_IDS and logical_id not in by_id,
                f"unexpected or duplicate finalizer artifact: {logical_id}")
        require(
            isinstance(record["uri"], str)
            and record["uri"].startswith(source_run_uri + "/")
            and "/valid/" not in record["uri"].lower()
            and "/test/" not in record["uri"].lower(),
            f"artifact URI escapes source run or opens VALID/TEST: {logical_id}",
        )
        require(isinstance(record["basename"], str) and record["basename"],
                f"invalid basename: {logical_id}")
        require(isinstance(record["sha256"], str) and SHA256_RE.fullmatch(record["sha256"]),
                f"invalid SHA-256: {logical_id}")
        by_id[logical_id] = record
    require(tuple(record["logical_id"] for record in records) == EXPECTED_IDS,
            "finalizer artifact order differs")
    require(set(artifact_paths) == set(EXPECTED_IDS), "artifact bindings are incomplete")
    verified: list[dict] = []
    for logical_id in EXPECTED_IDS:
        record = by_id[logical_id]
        path = artifact_paths[logical_id]
        require(path.is_file(), f"staged artifact is missing: {logical_id}")
        require(path.name == record["basename"], f"staged basename differs: {logical_id}")
        observed = sha256(path)
        require(observed == record["sha256"], f"staged SHA-256 differs: {logical_id}")
        verified.append({**record, "observed_sha256": observed})
    decide_spec = manifest.get("decision_script")
    require(isinstance(decide_spec, dict) and set(decide_spec) == {"basename", "sha256"},
            "decision script manifest fields differ")
    require(decide.name == decide_spec["basename"] and decide.is_file(),
            "pinned decision script is missing")
    require(SHA256_RE.fullmatch(str(decide_spec["sha256"])) is not None,
            "decision script SHA-256 is malformed")
    require(sha256(decide) == decide_spec["sha256"], "decision script SHA-256 differs")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifact", action="append", default=[], required=True)
    parser.add_argument("--decision-script", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--provenance-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()

    require(SHA256_RE.fullmatch(args.manifest_sha256) is not None,
            "manifest SHA-256 is malformed")
    require(args.manifest.is_file() and sha256(args.manifest) == args.manifest_sha256,
            "finalizer manifest SHA-256 differs")
    require(COMMIT_RE.fullmatch(args.code_commit) is not None, "code commit must be a full Git SHA")
    require("@sha256:" in args.runtime_image, "runtime image must be pinned by digest")
    bindings = [parse_artifact(value) for value in args.artifact]
    require(len(bindings) == len(EXPECTED_IDS), "exactly six artifact bindings are required")
    artifact_paths = dict(bindings)
    require(len(artifact_paths) == len(bindings), "duplicate artifact binding")

    manifest = load_json(args.manifest)
    verified = validate_manifest(manifest, artifact_paths, args.decision_script)
    receipt = args.output.with_suffix(".receipt.json")
    for path in (args.output, receipt, args.provenance_output):
        require(not path.exists(), f"refusing to overwrite finalizer output: {path}")

    command = [
        sys.executable,
        str(args.decision_script),
        "--analytic", str(artifact_paths["analytic_metrics"]),
        "--analytic-receipt", str(artifact_paths["analytic_receipt"]),
        "--tcn", str(artifact_paths["tcn_metrics"]),
        "--tcn-receipt", str(artifact_paths["tcn_receipt"]),
        "--positive", str(artifact_paths["positive_metrics"]),
        "--positive-receipt", str(artifact_paths["positive_receipt"]),
        "--output", str(args.output),
    ]
    subprocess.run(command, check=True)
    require(args.output.is_file() and receipt.is_file(), "decision script did not emit both outputs")
    decision = load_json(args.output)
    decision_receipt = load_json(receipt)
    require(decision.get("stage") == "M38B_FINAL_PRESPECIFIED_DECISION", "decision stage differs")
    require(decision.get("status") == "PASS_GATES_EVALUATED_NO_FAMILY_SELECTION",
            "decision status differs")
    require(decision_receipt.get("output_sha256") == sha256(args.output),
            "decision receipt does not bind decision output")

    provenance_sources = []
    for path in sorted(args.provenance_source, key=lambda item: item.name):
        require(path.is_file(), f"missing finalizer provenance source: {path}")
        provenance_sources.append({"basename": path.name, "sha256": sha256(path)})
    args.provenance_output.write_text(json.dumps({
        "schema_version": "1.0.0",
        "stage": "M38B_FINALIZER_PROVENANCE",
        "status": "PASS_FINALIZED_IMMUTABLE_SCORES",
        "source_run_uri": manifest["source_run_uri"],
        "code_commit": args.code_commit,
        "runtime_image": args.runtime_image,
        "manifest_sha256": args.manifest_sha256,
        "decision_script_sha256": sha256(args.decision_script),
        "finalizer_script_sha256": sha256(Path(__file__)),
        "artifacts": verified,
        "decision": {
            "basename": args.output.name,
            "sha256": sha256(args.output),
            "receipt_basename": receipt.name,
            "receipt_sha256": sha256(receipt),
        },
        "provenance_sources": provenance_sources,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_FINALIZED_IMMUTABLE_SCORES",
                      "decision_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
