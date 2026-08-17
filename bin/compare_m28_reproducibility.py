#!/usr/bin/env python3
"""Compare two independent M28 executions of the same preregistered seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BYTE_IDENTICAL = (
    "m28_pools.private.tsv",
    "m28_mosaic_events.private.tsv.gz",
    "m28_lai_truth.tsv.gz",
    "m28_rare_catalog.tsv.gz",
    "m28_rare_haplotypes.tsv.gz",
    "m28_preflight.public.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(left: Path, right: Path, amendment: Path) -> dict:
    import tskit

    amendment_data = json.loads(amendment.read_text(encoding="utf-8"))
    byte_checks = {}
    for name in BYTE_IDENTICAL:
        left_hash = sha256(left / name)
        right_hash = sha256(right / name)
        byte_checks[name] = {
            "left_sha256": left_hash,
            "right_sha256": right_hash,
            "identical": left_hash == right_hash,
        }
    left_tree_path = left / "m28_sources.trees"
    right_tree_path = right / "m28_sources.trees"
    left_tree = tskit.load(left_tree_path)
    right_tree = tskit.load(right_tree_path)
    tree_check = {
        "left_sha256": sha256(left_tree_path),
        "right_sha256": sha256(right_tree_path),
        "left_file_uuid": left_tree.file_uuid,
        "right_file_uuid": right_tree.file_uuid,
        "semantic_equality": left_tree.equals(right_tree),
        "num_trees": left_tree.num_trees,
        "num_sites": left_tree.num_sites,
        "num_mutations": left_tree.num_mutations,
    }
    passed = tree_check["semantic_equality"] and all(
        item["identical"] for item in byte_checks.values()
    )
    return {
        "stage": amendment_data["stage"],
        "gate": "S1_REPRODUCIBILITY",
        "amendment_sha256": sha256(amendment),
        "byte_checks": byte_checks,
        "tree_sequence_check": tree_check,
        "passed": passed,
        "decision": "GO_PREFLIGHT_COMPLETE" if passed else "STOP_REPRODUCIBILITY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right, args.amendment)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
