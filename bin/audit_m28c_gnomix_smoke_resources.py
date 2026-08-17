#!/usr/bin/env python3
"""Audit M28C smoke resource gates from the immutable Nextflow trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


TRAIN_PATTERN = re.compile(r"^TRAIN_M28C_GNOMIX_SMOKE \(m28c_gnomix_smoke_train_([AB])\)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_size_gib(value: str) -> float:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B|[KMGT]iB|B)\s*", value)
    if not match:
        raise ValueError(f"Unsupported Nextflow size: {value!r}")
    number = float(match.group(1))
    unit = match.group(2)
    powers = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4, "KiB": 1, "MiB": 2, "GiB": 3, "TiB": 4}
    base = 1024.0 if "i" in unit else 1000.0
    return number * (base ** powers[unit]) / (1024.0**3)


def parse_duration_seconds(value: str) -> float:
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h|d)", value)
    if not parts or re.sub(r"[0-9.\s]+(?:ms|s|m|h|d)", "", value).strip():
        raise ValueError(f"Unsupported Nextflow duration: {value!r}")
    return sum(float(number) * units[unit] for number, unit in parts)


def audit(trace_path: Path, comparison_path: Path, contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("decision") != "PASS_PREDICTIONS_PENDING_RESOURCE_TRACE_REVIEW":
        raise ValueError("Prediction comparison did not pass before resource review")
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    observed = {}
    for row in rows:
        match = TRAIN_PATTERN.fullmatch(row["name"])
        if not match:
            continue
        replicate = match.group(1)
        if replicate in observed:
            raise ValueError(f"Duplicate trace row for training replicate {replicate}")
        if row["status"] != "COMPLETED" or row["exit"] != "0":
            raise ValueError(f"Training replicate {replicate} did not complete cleanly")
        observed[replicate] = {
            "duration_text": row["duration"],
            "duration_seconds": parse_duration_seconds(row["duration"]),
            "peak_rss_text": row["peak_rss"],
            "peak_rss_gib": parse_size_gib(row["peak_rss"]),
        }
    if sorted(observed) != ["A", "B"]:
        raise ValueError(f"Expected exactly training replicas A and B in trace; observed {sorted(observed)}")

    memory_limit_gib = 6.4
    duration_limit_seconds = 24 * 60
    per_replicate = {}
    for replicate, values in observed.items():
        memory_pass = values["peak_rss_gib"] <= memory_limit_gib
        duration_pass = values["duration_seconds"] <= duration_limit_seconds
        per_replicate[replicate] = {
            **values,
            "memory_review_pass": memory_pass,
            "duration_review_pass": duration_pass,
            "resource_review_pass": memory_pass and duration_pass,
        }
    resource_pass = all(item["resource_review_pass"] for item in per_replicate.values())
    decision = "GO_DESIGN_FULL_B0_RESOURCE_BENCHMARK" if resource_pass else "STOP_RESOURCE_REVIEW"
    return {
        "stage": "M28C_GNOMIX_TRAINING_SMOKE_RESOURCE_AUDIT",
        "scope": contract["scope"],
        "trace_sha256": sha256(trace_path),
        "comparison_sha256": sha256(comparison_path),
        "contract_sha256": sha256(contract_path),
        "thresholds": {
            "peak_rss_gib_max": memory_limit_gib,
            "duration_seconds_max": duration_limit_seconds,
        },
        "training_replicates": per_replicate,
        "gates": {"T6_REPRODUCIBILITY": True, "T7_RESOURCES": resource_pass, "T8_SCOPE": True},
        "target_truth_accessed": False,
        "internal_synthetic_validation_used_for_decision": False,
        "decision": decision,
        "interpretation_limit": "This audit authorizes at most the design of a full-B0 resource benchmark; it does not establish LAI accuracy or rare-variant utility.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.trace, args.comparison, args.preregistration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "stage": report["stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
