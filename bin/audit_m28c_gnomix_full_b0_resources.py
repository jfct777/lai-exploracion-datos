#!/usr/bin/env python3
"""Gate one protected full-B0 Gnomix replica before progression."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from audit_m28c_gnomix_smoke_resources import parse_duration_seconds, parse_size_gib, sha256


def audit(trace: Path, train_report: Path, infer_report: Path, contract_path: Path, replicate: str) -> dict:
    if replicate not in {"A", "B"}:
        raise ValueError("Replicate must be A or B")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    train = json.loads(train_report.read_text(encoding="utf-8"))
    infer = json.loads(infer_report.read_text(encoding="utf-8"))
    contract_hash = sha256(contract_path)
    if (
        contract.get("stage") != "M28C_GNOMIX_FULL_B0_RESOURCE_BENCHMARK"
        or contract.get("status") != "PRE_FROZEN_BEFORE_FULL_B0"
    ):
        raise ValueError("Unexpected or unfrozen full-B0 contract")
    if train.get("replicate") != replicate or infer.get("replicate") != replicate:
        raise ValueError("Report replicate differs from the requested audit")
    if train.get("decision") != "GO_FROZEN_MODEL_INFERENCE_NO_TRUTH":
        raise ValueError("Full-B0 training report did not pass")
    if infer.get("decision") != "GO_REPLICATE_COMPARISON_NO_TRUTH":
        raise ValueError("Full-B0 inference report did not pass")
    if train.get("contract_sha256") != contract_hash or infer.get("contract_sha256") != contract_hash:
        raise ValueError("Full-B0 report contract hash mismatch")
    if infer.get("train_report_sha256") != sha256(train_report):
        raise ValueError("Inference did not authenticate the audited training report")

    with trace.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    pattern = re.compile(
        rf"^TRAIN_M28C_GNOMIX_FULL_B0 \(m28c_gnomix_full_b0_train_{replicate}\)$"
    )
    matched = [row for row in rows if pattern.fullmatch(row["name"])]
    if len(matched) != 1:
        raise ValueError(f"Expected one full-B0 training row for {replicate}; observed {len(matched)}")
    row = matched[0]
    if row["status"] != "COMPLETED" or row["exit"] != "0":
        raise ValueError("Full-B0 training did not complete cleanly")
    duration_seconds = parse_duration_seconds(row["duration"])
    peak_rss_gib = parse_size_gib(row["peak_rss"])
    memory_pass = peak_rss_gib <= 19.2
    duration_pass = duration_seconds <= 48 * 60
    prediction = infer["prediction_audit"]
    dimensions = contract["gnomix_parameters"]["derived_expected"]
    structural_pass = (
        train["model_audit"]["derived_dimensions"]
        == {key: dimensions[key] for key in ("C", "M", "W", "A", "S", "context_markers_each_side")}
        and prediction["target_samples"] == contract["source_panel"]["target_samples"]
        and prediction["windows"] == dimensions["W"]
        and prediction["msp_marker_count_sum"] == dimensions["modeled_markers"]
        and prediction["msp_terminal_window_markers"] == dimensions["terminal_window_markers"]
        and prediction["population_order"] == ["AFR", "EUR", "ASIA"]
    )
    resource_pass = memory_pass and duration_pass
    gates = {
        "F0_AUTH": True,
        "F1_FULL_INPUT": True,
        "F2_RESIDUAL_POLICY": structural_pass,
        "F3_BOUNDARY": train["target_input_present"] is False and train["truth_accessed"] is False,
        "F4_MODEL": True,
        "F5_INFERENCE": structural_pass,
        "F6_RESOURCES": resource_pass,
        "F8_SCOPE": infer["truth_accessed"] is False and infer["target_truth_accuracy_computed"] is False,
    }
    passed = all(gates.values())
    if replicate == "A":
        decision = "GO_LAUNCH_FULL_B0_REPLICATE_B" if passed else "STOP_BEFORE_REPLICATE_B"
    else:
        decision = "GO_COMPARE_FULL_B0_REPLICATES" if passed else "STOP_FULL_B0_BEFORE_COMPARISON"
    return {
        "stage": "M28C_GNOMIX_FULL_B0_RESOURCE_GATE",
        "replicate": replicate,
        "contract_sha256": contract_hash,
        "trace_sha256": sha256(trace),
        "train_report_sha256": sha256(train_report),
        "inference_report_sha256": sha256(infer_report),
        "resources": {
            "duration_text": row["duration"],
            "duration_seconds": duration_seconds,
            "duration_review_max_seconds": 2880,
            "peak_rss_text": row["peak_rss"],
            "peak_rss_gib": peak_rss_gib,
            "peak_rss_review_max_gib": 19.2,
        },
        "gates": gates,
        "internal_synthetic_validation_used_for_decision": False,
        "target_truth_accessed": False,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--train-report", required=True, type=Path)
    parser.add_argument("--inference-report", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--replicate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(
        args.trace,
        args.train_report,
        args.inference_report,
        args.preregistration,
        args.replicate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "stage": report["stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
