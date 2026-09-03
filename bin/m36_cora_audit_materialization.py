#!/usr/bin/env python3
"""Audit whether an authenticated M36 materialization can enter training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

for import_root in (Path(__file__).resolve().parent, Path.cwd()):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from m36_cora_set import ContractError, as_float, component_folds, read_tsv
from m36_cora_train import fold_preprocessing_audit, target_partition_coverage


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def audit(receipt_path: Path, covariates_path: Path, components_path: Path,
          targets_path: Path, n_folds: int) -> dict[str, object]:
    """Check hashes, sample axes and component-disjoint target support."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("stage") == "M36_CORA_MATERIALIZE" and
             receipt.get("status") in {"MATERIALIZED_PASS", "PUBLISHED_PASS"} and
             receipt.get("synthetic") is False,
             "M36 audit requires a successful real materialization receipt")
    descriptors = receipt.get("input_descriptors")
    _require(isinstance(descriptors, dict), "M36 materialization descriptors are missing")
    paths = {
        "covariates": covariates_path,
        "components": components_path,
        "targets": targets_path,
    }
    for name, path in paths.items():
        descriptor = descriptors.get(name)
        _require(isinstance(descriptor, dict) and descriptor.get("sha256") == sha256(path),
                 f"M36 materialization hash differs: {name}")

    covariate_rows, component_rows, targets = map(
        read_tsv, (covariates_path, components_path, targets_path)
    )
    covariates = {row["sample_id"]: row for row in covariate_rows}
    component_map = {row["sample_id"]: row["pcrelate_component"] for row in component_rows}
    _require(len(covariates) == len(covariate_rows) and
             len(component_map) == len(component_rows) and
             set(covariates) == set(component_map),
             "M36 covariate/component sample axes differ")
    _require(n_folds >= 2 and len(set(component_map.values())) >= n_folds,
             "M36 lacks enough PC-Relate components for the requested folds")
    for row in targets:
        _require(row["sample_i"] in covariates and row["sample_j"] in covariates and
                 row["sample_i"] != row["sample_j"] and as_float(row["target"], "target") >= 0,
                 "M36 target pair differs")

    assignment = component_folds(component_map, n_folds)
    fold_audit = fold_preprocessing_audit(
        targets, covariates, component_map, assignment, n_folds
    )
    coverage = target_partition_coverage(targets, component_map, assignment)
    readiness: dict[str, dict[str, object]] = {}
    all_ready = True
    for fold, row in sorted(fold_audit.items(), key=lambda item: int(item[0])):
        def outcomes(partition: str) -> Counter[str]:
            counts: Counter[str] = Counter()
            for values in row[f"{partition}_target_counts"].values():
                counts.update(values)
            return counts

        fit_counts, validation_counts = outcomes("fit"), outcomes("validation")
        ready = (fit_counts["positive"] > 0 and fit_counts["zero"] > 0 and
                 validation_counts["positive"] > 0 and validation_counts["zero"] > 0)
        all_ready &= ready
        readiness[fold] = {
            "ready": ready,
            "fit_positive": fit_counts["positive"],
            "fit_zero": fit_counts["zero"],
            "validation_positive": validation_counts["positive"],
            "validation_zero": validation_counts["zero"],
        }

    return {
        "schema_version": "1.0.0",
        "stage": "M36_CORA_MATERIALIZATION_AUDIT",
        "status": "PASS_TRAINABLE" if all_ready else "STOP_INSUFFICIENT_FOLD_SUPPORT",
        "scope": "exploratory cross-chromosome common-asIBD prediction; not LAI validation",
        "n_samples": len(covariates),
        "n_pcrelate_components": len(set(component_map.values())),
        "n_target_pairs": len(targets),
        "n_folds": n_folds,
        "fold_readiness": readiness,
        "fold_audit": fold_audit,
        "target_partition_coverage": coverage,
        "component_assignment_sha256": hashlib.sha256(json.dumps(
            assignment, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "source_sha256": {"receipt": sha256(receipt_path), **{
            name: sha256(path) for name, path in paths.items()
        }},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--covariates", required=True, type=Path)
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite M36 materialization audit")
    payload = audit(args.materialization_receipt, args.covariates, args.components,
                    args.targets, args.n_folds)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "pairs": payload["n_target_pairs"]},
                     sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ContractError as error:
        raise SystemExit(f"M36 materialization audit error: {error}") from error
