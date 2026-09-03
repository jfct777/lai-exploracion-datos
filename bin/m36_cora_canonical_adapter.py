#!/usr/bin/env python3
"""Adapt canonical metadata, M20 and PC-Relate-derived groups for M36.

No genotype or target is read here.  It creates reproducible per-person
covariates and component labels, preserving the canonical sample IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def unique(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        existing = result.get(row[key])
        if existing is not None and existing != row:
            raise ValueError(f"{label} has conflicting duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--m20-feature-store", required=True, type=Path)
    parser.add_argument("--modeling-master", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    metadata_rows = read(args.metadata)
    metadata = unique(metadata_rows, "ID", "metadata")
    m20 = unique(read(args.m20_feature_store), "sample_id", "M20 feature store")
    master = unique(read(args.modeling_master), "sample_id", "modeling master")
    required_m20 = {"Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS", "rare_gt_nonmissing_sites", "rare_missing_sites"}
    if not required_m20.issubset(m20[next(iter(m20))]):
        raise SystemExit(f"M36 canonical adapter error: M20 lacks {sorted(required_m20 - set(m20[next(iter(m20))]))}")
    group = "kinship_group_id_phi0442"
    if group not in master[next(iter(master))]:
        raise SystemExit("M36 canonical adapter error: modeling master lacks PC-Relate component proxy")
    samples = sorted(set(metadata) & set(m20) & set(master))
    if not samples:
        raise SystemExit("M36 canonical adapter error: no shared canonical samples")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "m36_cora_sample_metadata.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cohort", "asibd_id", "rare_callability", "Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS"], delimiter="\t")
        writer.writeheader()
        for sample in samples:
            feature, meta = m20[sample], metadata[sample]
            nonmissing, missing = float(feature["rare_gt_nonmissing_sites"]), float(feature["rare_missing_sites"])
            if nonmissing < 0 or missing < 0 or nonmissing + missing <= 0:
                raise SystemExit(f"M36 canonical adapter error: invalid M20 callability for {sample}")
            cohort = feature.get("cohort") or meta.get("Cohort")
            if not cohort:
                raise SystemExit(f"M36 canonical adapter error: missing cohort for {sample}")
            writer.writerow({"sample_id": sample, "cohort": cohort, "asibd_id": f"{cohort}_{sample}",
                             "rare_callability": f"{nonmissing / (nonmissing + missing):.12g}",
                             "Q_AFR": feature["Q_AFR"], "Q_EUR": feature["Q_EUR"], "Q_NAM": feature["Q_NAM"],
                             "Q_EAS": feature["Q_EAS"]})
    with (args.outdir / "m36_cora_pcrelate_components.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "pcrelate_component"], delimiter="\t")
        writer.writeheader()
        writer.writerows({"sample_id": sample, "pcrelate_component": master[sample][group]} for sample in samples)
    receipt = {
        "stage": "M36_CORA_CANONICAL_ADAPTER",
        "status": "ADAPTED_PASS",
        "n_metadata_rows": len(metadata_rows),
        "n_metadata_unique_ids": len(metadata),
        "n_analysis_samples": len(samples),
        "duplicate_metadata_policy": "identical rows are collapsed; conflicting duplicate IDs fail closed",
        "source_input_descriptors": {
            "metadata": {"uri": str(args.metadata), "sha256": sha256(args.metadata)},
            "m20_feature_store": {"uri": str(args.m20_feature_store), "sha256": sha256(args.m20_feature_store)},
            "modeling_master": {"uri": str(args.modeling_master), "sha256": sha256(args.modeling_master)},
        },
    }
    (args.outdir / "m36_cora_canonical_adapter_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
