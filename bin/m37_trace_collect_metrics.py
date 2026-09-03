#!/usr/bin/env python3
"""Collect a complete authenticated set of paired M37 arm metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from m37_trace_core import require


ARMS = ("RE", "RD", "POOLED", "SHAM", "GEOMETRY")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_metrics(metric_paths: list[Path], receipt_paths: list[Path], root: str,
                    expected_evaluation_split: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Authenticate metric/receipt pairs and return deterministic promotion rows."""
    require(root and metric_paths and len(metric_paths) == len(receipt_paths),
            "M37 metric collection inputs differ")
    require(len({path.name for path in metric_paths}) == len(metric_paths) and
            len({path.name for path in receipt_paths}) == len(receipt_paths),
            "M37 metric collection basenames are not unique")
    receipts: dict[tuple[str, str, str, str], tuple[Path, dict[str, object]]] = {}
    for path in receipt_paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        require(receipt.get("stage") == "M37_TRACE_SCORE", "M37 score receipt stage differs")
        key = tuple(str(receipt.get(name, "")) for name in ("candidate_id", "family", "root", "arm"))
        require(all(key) and key[2] == root and key[3] in ARMS and key not in receipts,
                "M37 score receipt identity differs")
        receipts[key] = (path, receipt)

    rows: list[dict[str, object]] = []
    used_receipts: set[Path] = set()
    identities: set[tuple[str, str, str, str]] = set()
    for path in metric_paths:
        metric = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(metric, dict), "M37 metric payload differs")
        key = tuple(str(metric.get(name, "")) for name in ("candidate_id", "family", "root", "arm"))
        require(all(key) and key[2] == root and key[3] in ARMS and key not in identities,
                "M37 metric identity differs")
        require(metric.get("evaluation_split") == expected_evaluation_split,
                "M37 metric evaluation split differs")
        require(key in receipts, "M37 metric lacks its authenticated score receipt")
        receipt_path, receipt = receipts[key]
        require(receipt.get("output_sha256") == sha256(path),
                "M37 metric/score receipt hash differs")
        identities.add(key)
        used_receipts.add(receipt_path)
        rows.append({"candidate_id": key[0], "family": key[1], "root": key[2],
                     "arm": key[3], "metrics": metric})

    require(used_receipts == set(receipt_paths), "M37 score receipt set has an unpaired member")
    by_candidate: dict[tuple[str, str], set[str]] = {}
    for candidate_id, family, observed_root, arm in sorted(identities):
        key = (candidate_id, family)
        by_candidate.setdefault(key, set()).add(arm)
        require(observed_root == root, "M37 metric root differs")
    for key, arms in by_candidate.items():
        require(arms == set(ARMS), f"M37 candidate {key[0]} lacks a complete paired arm family")

    rows.sort(key=lambda row: (str(row["candidate_id"]), str(row["family"]),
                               ARMS.index(str(row["arm"]))))
    evidence = {
        "metric_sha256": {path.name: sha256(path) for path in sorted(metric_paths)},
        "score_receipt_sha256": {path.name: sha256(path) for path in sorted(receipt_paths)},
    }
    return rows, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", action="append", type=Path, required=True)
    parser.add_argument("--receipt", action="append", type=Path, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-evaluation-split", default="FIT_TUNE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite M37 metric collection")
    rows, evidence = collect_metrics(args.metric, args.receipt, args.root,
                                     args.expected_evaluation_split)
    payload = {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_COLLECT_METRICS",
        "root": args.root,
        "evaluation_split": args.expected_evaluation_split,
        "rows": rows,
        "input_evidence": evidence,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M37_TRACE_COLLECT_METRICS",
        "root": args.root,
        "evaluation_split": args.expected_evaluation_split,
        "candidate_count": len({str(row["candidate_id"]) for row in rows}),
        "row_count": len(rows),
        "input_evidence": evidence,
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS_M37_METRIC_COLLECTION", "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
