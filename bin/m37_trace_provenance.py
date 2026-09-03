#!/usr/bin/env python3
"""Write a deterministic M37 manifest and READY marker after artifact verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--family", choices=("hmm", "tcn"), required=True)
    parser.add_argument("--arm", choices=("RE", "RD", "POOLED", "SHAM", "GEOMETRY"), required=True)
    parser.add_argument("--auth-file", action="append", type=Path, required=True)
    parser.add_argument("--run-overlay", type=Path, required=True)
    parser.add_argument("--run-overlay-uri", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not args.run_overlay_uri.strip():
        raise ValueError("run overlay URI is empty; refusing READY")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    observed = digest(args.artifact)
    if receipt.get("stage") != "M37_TRACE_SCORE" or receipt.get("output_sha256") != observed:
        raise ValueError("artifact/receipt hash differs; refusing READY")
    if any(receipt.get(key) != expected for key, expected in (
        ("candidate_id", args.candidate_id), ("family", args.family),
        ("root", args.root), ("arm", args.arm)
    )):
        raise ValueError("artifact/receipt candidate identity differs; refusing READY")
    auth_files = {path.name: digest(path) for path in args.auth_file}
    if len(auth_files) != len(args.auth_file):
        raise ValueError("duplicate authenticated source basename")
    manifest = {"schema_version": "1.0.0", "stage": "M37_TRACE_PROVENANCE", "run_id": args.run_id,
                "artifact": args.artifact.name, "artifact_sha256": observed, "receipt_sha256": digest(args.receipt),
                "candidate_id": args.candidate_id, "family": args.family, "root": args.root, "arm": args.arm,
                "receipt_stage": receipt["stage"], "receipt_output_sha256": receipt["output_sha256"],
                "authenticated_sources": auth_files,
                "run_overlay": {"uri": args.run_overlay_uri, "sha256": digest(args.run_overlay)},
                "container_digest": args.container_digest, "status": "READY"}
    # ``Path.with_suffix`` would silently drop the arm from a dotted prefix
    # such as ``candidate.tcn.RE``.  Append the controlled suffix instead.
    manifest_path = Path(f"{args.output_prefix}.manifest.json")
    ready_path = Path(f"{args.output_prefix}.READY.json")
    if manifest_path.exists() or ready_path.exists():
        raise ValueError("refusing to overwrite M37 manifest/READY")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ready_path.write_text(json.dumps({"schema_version": "1.0.0", "status": "READY",
                                      "run_id": args.run_id, "candidate_id": args.candidate_id,
                                      "family": args.family, "root": args.root, "arm": args.arm,
                                      "run_overlay": {"uri": args.run_overlay_uri,
                                                      "sha256": digest(args.run_overlay)},
                                      "artifact_sha256": observed,
                                      "receipt_sha256": digest(args.receipt),
                                      "manifest_sha256": digest(manifest_path)},
                                     indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
