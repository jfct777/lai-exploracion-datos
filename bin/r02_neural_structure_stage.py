#!/usr/bin/env python3
"""Restage one previously validated HUMAN event store for the trainer.

The published FULL22 gate supplies the completed semantic validation.  This
consumer step still authenticates the generation-pinned archive and hashes each
extracted allowlisted file, but deliberately does not repeat the container
reader validation that produced the prior gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import r02_neural_store_validate as store


SCHEMA = "r02_neural_structure_consumer_stage_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def prior_row(gate_path, gate_sha256, chromosome, request_sha256):
    require(re.fullmatch(r"[0-9a-f]{64}", gate_sha256 or "") and
            store.sha256(gate_path) == gate_sha256,
            "published FULL22 gate SHA256 differs")
    gate = load(gate_path)
    require(gate.get("schema") == "r02_neural_input_validation_complete_v1"
            and gate.get("status") == "COMPLETE_22_REMOTE_RECEIPTS_VERIFIED_NO_TRAINING"
            and gate.get("chromosomes") == 22 and gate.get("training_started") is False
            and gate.get("evaluate_opened") is False and len(gate.get("receipts", [])) == 22,
            "prior FULL22 validation gate differs")
    rows = [row for row in gate["receipts"]
            if str(row.get("chromosome")) == str(chromosome)]
    require(len(rows) == 1 and rows[0].get("request_sha256") == request_sha256,
            "prior chromosome validation does not bind this request")
    return rows[0]


def inventory_sizes(root, expected):
    values = {}
    for name, descriptor in expected.items():
        path = root / name
        require(path.is_file() and path.stat().st_size == descriptor["bytes"],
                "retained staged file inventory/size differs: " + name)
        values[name] = path.stat().st_size
    return values


def run(request_path, request_sha256, manifest_override, gate_path, gate_sha256, output):
    require(store.sha256(request_path) == request_sha256, "request SHA256 differs")
    request = load(request_path)
    require(store.sha256(manifest_override) == request["manifest_sha256"],
            "store manifest SHA256 differs")
    request["manifest_path"] = str(Path(manifest_override).resolve())
    archive, manifest, stage_root = store.validate_request(request, request_path)
    previous = prior_row(gate_path, gate_sha256, request["chromosome"], request_sha256)
    stage_root.mkdir(parents=True, exist_ok=True)
    work = stage_root / ("chr" + str(request["chromosome"]) + ".neural-input-validation")
    receipt_path = work / "structure-consumer-stage.json"
    store_root = work / "store"
    expected = store.manifest_files(manifest)
    if receipt_path.is_file():
        receipt = load(receipt_path)
        require(receipt.get("schema") == SCHEMA and receipt.get("request_sha256") == request_sha256
                and receipt.get("prior_full22_gate_sha256") == gate_sha256,
                "retained consumer-stage receipt differs")
        inventory_sizes(store_root, expected)
    else:
        if work.exists():
            marker = work / ".r02-neural-staging-owned"
            require(marker.is_file() and marker.read_text().strip() == store.RECEIPT_SCHEMA,
                    "ambiguous retained stage without ownership marker")
            # An interrupted extraction has no reusable completeness claim.
            # Delete only this explicitly owned, regenerable partial store; a
            # complete store always has the durable consumer receipt above.
            if store_root.exists():
                shutil.rmtree(store_root)
        else:
            work.mkdir()
            (work / ".r02-neural-staging-owned").write_text(store.RECEIPT_SCHEMA + "\n")
        archive_path = work / "source.tar.gz"
        if not archive_path.exists():
            require(shutil.disk_usage(stage_root).free >= request["minimum_free_bytes"],
                    "measured free space is below the frozen staging threshold")
            subprocess.run(["gcloud", "storage", "cp",
                            archive["uri"] + "#" + str(archive["generation"]),
                            str(archive_path)], check=True)
        require(archive_path.stat().st_size == archive["bytes"] and
                store.sha256(archive_path) == archive["sha256"],
                "generation-pinned archive differs")
        actual = store.extract_allowlisted(archive_path, store_root,
                                           Path(request["manifest_path"]), manifest,
                                           request["reserve_free_bytes"])
        archive_path.unlink()
        extracted = sum(value["bytes"] for value in actual.values())
        require(len(actual) == previous["file_count"] and extracted == previous["bytes_hashed"],
                "re-extracted inventory differs from prior validated chromosome")
        receipt = {"schema": SCHEMA, "status": "STAGED_FOR_ONE_AUTHENTICATED_CONSUMER",
                   "chromosome": str(request["chromosome"]),
                   "request_sha256": request_sha256,
                   "manifest_sha256": request["manifest_sha256"],
                   "file_count": len(actual), "extracted_bytes": extracted,
                   "prior_full22_gate_sha256": gate_sha256,
                   "prior_publication": previous["publication"],
                   "semantic_validation_repeated": False,
                   "archive_and_extracted_files_sha256_verified": True,
                   "store_path": str(store_root.resolve()),
                   "training_started": False, "evaluate_opened": False}
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    Path(output).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--manifest-override", type=Path, required=True)
    parser.add_argument("--prior-full22-gate", type=Path, required=True)
    parser.add_argument("--prior-full22-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.request.resolve(), args.request_sha256,
                         args.manifest_override.resolve(), args.prior_full22_gate.resolve(),
                         args.prior_full22_gate_sha256, args.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
