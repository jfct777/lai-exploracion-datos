#!/usr/bin/env python3
"""Bind an M36 training summary to its authenticated materialization receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--train-summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    materialization = json.loads(args.materialization_receipt.read_text(encoding="utf-8"))
    summary = json.loads(args.train_summary.read_text(encoding="utf-8"))
    if materialization.get("status") not in {"MATERIALIZED_PASS", "PUBLISHED_PASS"}:
        raise SystemExit("M36 train receipt error: source materialization is not successful")
    if summary.get("stage") != "M36_CORA_SET_TRAIN" or summary.get("mode") != "train":
        raise SystemExit("M36 train receipt error: summary is not real train output")
    preprocessing = summary.get("fit_only_preprocessing")
    categorical = summary.get("categorical_covariates")
    if not isinstance(preprocessing, list) or "cohort_one_hot_vocabulary" not in preprocessing:
        raise SystemExit("M36 train receipt error: FIT-only cohort encoding is not authenticated")
    if categorical != {
        "cohort": {"encoding": "one_hot", "vocabulary_scope": "FIT_only", "unknown_level": "<UNK>"}
    }:
        raise SystemExit("M36 train receipt error: cohort encoding contract drift")
    runs = summary.get("runs")
    if not isinstance(runs, dict) or not runs or any(
        not isinstance(run.get("fit_only_preprocessing_by_fold"), dict)
        or not isinstance(run.get("target_partition_coverage"), dict)
        for run in runs.values()
    ):
        raise SystemExit("M36 train receipt error: FIT preprocessing or target-partition audit is missing")
    args.out.write_text(json.dumps({
        "stage": "M36_CORA_SET_TRAIN",
        "status": "TRAINED_TECHNICAL_PASS",
        "scope": "exploratory external common-IBD pair-total prediction; no biological or LAI-superiority claim",
        "source_materialization_receipt_sha256": sha256(args.materialization_receipt),
        "train_summary_sha256": sha256(args.train_summary),
        "run_controls": summary.get("controls"),
        "outer_cv": summary.get("outer_cv"),
        "fit_only_preprocessing": preprocessing,
        "categorical_covariates": categorical,
        "fit_only_preprocessing_by_run": {
            label: run["fit_only_preprocessing_by_fold"] for label, run in sorted(runs.items())
        },
        "target_partition_coverage_by_run": {
            label: run["target_partition_coverage"] for label, run in sorted(runs.items())
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
