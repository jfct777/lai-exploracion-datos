#!/usr/bin/env python3
"""Close exactly 22 v3-restaged HUMAN stores into a constructor source-path index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "r02_structure_restored_store_index_v1"
RECEIPT_SCHEMA = "r02_neural_structure_consumer_stage_v3"
HEX64 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned(path, expected_sha256):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink() and
            HEX64.fullmatch(expected_sha256 or "") and sha256(path) == expected_sha256,
            "pinned input differs: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def build(request_index_path, request_index_sha256, gate_path, gate_sha256,
          amendment_path, amendment_sha256, stage_root, receipts):
    index = load_pinned(request_index_path, request_index_sha256)
    gate = load_pinned(gate_path, gate_sha256)
    amendment = load_pinned(amendment_path, amendment_sha256)
    require(index.get("schema") == "r02_neural_store_request_index_v1" and
            len(index.get("requests", [])) == 22, "exact 22-store request index required")
    require(gate.get("schema") == "r02_neural_input_validation_complete_v1" and
            gate.get("status") == "COMPLETE_22_REMOTE_RECEIPTS_VERIFIED_NO_TRAINING" and
            gate.get("chromosomes") == 22 and gate.get("training_started") is False and
            gate.get("evaluate_opened") is False, "FULL22 validation gate differs")
    require(amendment.get("schema") == "r02_neural_structure_stage_recovery_amendment_v1" and
            amendment.get("status") == "APPROVED_ENGINEERING_RECOVERY_SAME_SCIENTIFIC_INPUTS" and
            amendment.get("scope") == "STAGE_DENOMINATOR_AND_RESUME_ONLY" and
            amendment.get("policy", {}).get("scientific_inputs_changed") is False and
            amendment.get("policy", {}).get("evaluate_opened") is False,
            "v3 recovery amendment differs")
    expected_requests = {str(row["chromosome"]): row for row in index["requests"]}
    expected_amendment = {str(row["chromosome"]): row for row in amendment["chromosomes"]}
    require(set(expected_requests) == set(expected_amendment) == {str(i) for i in range(1, 23)},
            "request/amendment chromosome universe differs")
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in receipts]
    by_chromosome = {str(row.get("chromosome")): row for row in rows}
    require(len(rows) == len(by_chromosome) == 22 and set(by_chromosome) == set(expected_requests),
            "restaged receipt chromosome universe differs")

    stage_root = Path(stage_root).resolve()
    require(stage_root.is_dir() and not stage_root.is_symlink(), "stage root is absent or unsafe")
    result_rows = []
    for chromosome in map(str, range(1, 23)):
        receipt = by_chromosome[chromosome]
        request = expected_requests[chromosome]
        amended = expected_amendment[chromosome]
        require(receipt.get("schema") == RECEIPT_SCHEMA and
                receipt.get("status") == "STAGED_FOR_ONE_AUTHENTICATED_CONSUMER" and
                receipt.get("request_sha256") == request["request"]["sha256"] == amended["request_sha256"] and
                receipt.get("manifest_sha256") == request["manifest"]["sha256"] == amended["manifest_sha256"] and
                receipt.get("prior_full22_gate_sha256") == gate_sha256 and
                receipt.get("amendment_sha256") == amendment_sha256 and
                receipt.get("file_count") == amended["file_count"] and
                receipt.get("extracted_bytes") == amended["complete_allowlisted_extracted_bytes"] and
                receipt.get("channel_chunk_bytes_hashed") == amended["channel_chunk_bytes_hashed"] and
                receipt.get("archive_and_all_allowlisted_files_sha256_verified") is True and
                receipt.get("training_started") is False and receipt.get("evaluate_opened") is False,
                "restaged receipt lineage/state differs for chromosome " + chromosome)
        expected_work = stage_root / ("chr" + chromosome + ".neural-input-validation")
        store_path = Path(receipt["store_path"]).resolve()
        require(store_path == expected_work / "store" and store_path.is_dir() and
                (store_path / "manifest.json").is_file(),
                "restaged store path differs for chromosome " + chromosome)
        durable = expected_work / "structure-consumer-stage-v2.json"
        require(durable.is_file() and json.loads(durable.read_text(encoding="utf-8")) == receipt,
                "durable v3 receipt differs for chromosome " + chromosome)
        result_rows.append({
            "chromosome": chromosome,
            "store_path": str(store_path),
            "manifest_path": str(store_path / "manifest.json"),
            "manifest_sha256": request["manifest"]["sha256"],
            "request_sha256": request["request"]["sha256"],
            "stage_receipt_path": str(durable),
            "stage_receipt_sha256": sha256(durable),
            "file_count": receipt["file_count"],
            "extracted_bytes": receipt["extracted_bytes"],
        })
    require(sum(row["file_count"] for row in result_rows) == 72625 and
            sum(row["extracted_bytes"] for row in result_rows) == 196389901732,
            "FULL22 restored inventory totals differ")
    return {
        "schema": SCHEMA,
        "status": "COMPLETE_22_RESTORED_AUTHENTICATED_STORE_PATHS_NO_ASSEMBLY_NO_TRAINING",
        "chromosomes": 22,
        "files": 72625,
        "complete_allowlisted_extracted_bytes": 196389901732,
        "channel_chunk_bytes_hashed": 195593032185,
        "stage_root": str(stage_root),
        "request_index": {"path": str(Path(request_index_path).resolve()),
                          "sha256": request_index_sha256},
        "prior_full22_gate": {"path": str(Path(gate_path).resolve()), "sha256": gate_sha256},
        "stage_recovery_amendment": {"path": str(Path(amendment_path).resolve()),
                                     "sha256": amendment_sha256},
        "stores": result_rows,
        "constructor_input": "22 authenticated store_path values; no person batches assembled",
        "assembly_started": False,
        "training_started": False,
        "evaluate_opened": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-index", required=True)
    parser.add_argument("--request-index-sha256", required=True)
    parser.add_argument("--prior-full22-gate", required=True)
    parser.add_argument("--prior-full22-gate-sha256", required=True)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--receipts", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.request_index, args.request_index_sha256,
                   args.prior_full22_gate, args.prior_full22_gate_sha256,
                   args.amendment, args.amendment_sha256, args.stage_root, args.receipts)
    output = Path(args.output)
    require(not output.exists(), "fresh output required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "chromosomes": 22,
                      "complete_allowlisted_extracted_bytes":
                      result["complete_allowlisted_extracted_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
