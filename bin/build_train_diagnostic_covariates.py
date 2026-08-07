#!/usr/bin/env python3
"""Seal Q and cohort covariates to canonical TRAIN for M25B diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from build_rare_window_features import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--expected-train", type=int, default=2091)
    parser.add_argument("--output", default="train_diagnostic_covariates.tsv")
    parser.add_argument("--audit", default="train_diagnostic_covariates.audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.split_manifest).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "split", "cohort"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("split manifest lacks sample_id/split/cohort")
        split_rows = list(reader)
    train = {row["sample_id"]: row["cohort"] for row in split_rows if row["split"] == "TRAIN"}
    test_ids = {row["sample_id"] for row in split_rows if row["split"] == "TEST"}
    if len(train) != args.expected_train or len(test_ids) != 522:
        raise SystemExit("canonical TRAIN/TEST cardinality gate failed")

    columns = ["sample_id", "Q_NAM", "Q_EUR", "Q_EAS", "Q_AFR", "cohort"]
    selected: dict[str, dict[str, str]] = {}
    rows_seen = 0
    with Path(args.feature_store).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not set(columns).issubset(reader.fieldnames):
            raise SystemExit("feature store lacks diagnostic fields")
        for row in reader:
            rows_seen += 1
            sample_id = row["sample_id"]
            if sample_id not in train:
                continue
            if sample_id in selected:
                raise SystemExit(f"duplicate TRAIN covariate row: {sample_id}")
            values = [float(row[column]) for column in columns[1:5]]
            if any(value < 0 or value > 1 for value in values) or abs(sum(values) - 1.0) > 5e-3:
                raise SystemExit(f"invalid Q vector for {sample_id}")
            if row["cohort"] != train[sample_id]:
                raise SystemExit(f"cohort mismatch for {sample_id}")
            selected[sample_id] = {column: row[column] for column in columns}
    if set(selected) != set(train):
        raise SystemExit("TRAIN covariate membership is incomplete")

    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in split_rows:
            if row["sample_id"] in selected:
                writer.writerow(selected[row["sample_id"]])

    audit = {
        "status": "PASS",
        "scope": "diagnostic Q/cohort covariates sealed to canonical TRAIN",
        "n_feature_store_rows_streamed": rows_seen,
        "n_train_rows_emitted": len(selected),
        "n_test_rows_emitted": 0,
        "n_test_values_used_in_pca_or_diagnostics": 0,
        "fields": columns,
        "historical_rare_fields_used": [],
        "inputs_sha256": {
            "split_manifest": sha256_file(args.split_manifest),
            "feature_store": sha256_file(args.feature_store),
        },
        "output_sha256": sha256_file(args.output),
    }
    write_json(args.audit, audit)
    print(json.dumps({"status": "PASS", "n_train": len(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
