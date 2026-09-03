#!/usr/bin/env python3
"""Freeze exact M38B TRAIN/SELECT/SCORE rotations without reading truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import require
from m38b_oof_core import build_outer_roles


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, output: Path, outer_seed: int, inner_seed_start: int,
          source_receipt: Path | None = None) -> dict[str, object]:
    if source_receipt is not None:
        document = json.loads(source_receipt.read_text(encoding="utf-8"))
        require(
            document.get("stage") == "M37_TRACE_MATERIALIZE"
            and document.get("arm") == "RE"
            and document.get("output_sha256") == sha256(source),
            "fold source receipt, arm, or hash differs",
        )
    with np.load(source, allow_pickle=False) as archive:
        require("sample_key_sha256" in archive.files, "fold source lacks sample keys")
        sample_keys = np.ascontiguousarray(archive["sample_key_sha256"])
    roles, inner_seeds = build_outer_roles(
        sample_keys, outer_seed=outer_seed, inner_seed_start=inner_seed_start,
    )
    payload = {
        "sample_key_sha256": sample_keys,
        "roles": roles,
        "outer_fold": np.arange(3, dtype=np.uint8),
        "inner_split_seed": inner_seeds,
        "outer_seed": np.asarray([outer_seed], dtype=np.int64),
    }
    require(not output.exists(), "refusing to overwrite M38B folds")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(output, payload)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M38B_FREEZE_OOF_ROTATION",
        "status": "PASS_TRUTH_BLIND_EXACT_ROTATION",
        "source_sha256": sha256(source),
        "source_receipt_sha256": sha256(source_receipt) if source_receipt else None,
        "source_receipt_required_in_production": True,
        "output_sha256": sha256(output),
        "truth_read": False,
        "target_genotypes_read": False,
        "folds": 3,
        "people": int(len(sample_keys)),
        "per_fold": {"TRAIN": 48, "SELECT": 16, "SCORE": 32},
        "score_appearances_per_person": 1,
        "outer_seed": int(outer_seed),
        "inner_split_seeds": [int(value) for value in inner_seeds],
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--outer-seed", type=int, default=38032026)
    parser.add_argument("--inner-seed-start", type=int, default=38100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.outer_seed,
                           args.inner_seed_start, args.source_receipt), sort_keys=True))


if __name__ == "__main__":
    main()
