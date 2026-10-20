#!/usr/bin/env python3
"""Prepare the ten K03 INNER_FIT jobs from one authenticated shared ALT bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import r02_k03_inner_fit_contract as contract


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-bundle", type=Path, required=True)
    parser.add_argument("--plans-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--memory-gb", type=int, required=True)
    parser.add_argument("--max-forks", type=int, required=True)
    parser.add_argument("--task-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.shared_bundle.is_absolute() and args.plans_dir.is_absolute()
            and args.code_dir.is_absolute() and args.output.is_absolute()
            and not args.output.exists() and args.output.parent.is_dir(),
            "Absolute existing inputs and a new output are required")
    require(re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", args.container) is not None
            and args.memory_gb > 0 and args.max_forks > 0
            and re.fullmatch(r"[1-9][0-9]*[smh]", args.task_time) is not None,
            "Pinned container and positive explicit resources required")
    manifest_path = args.shared_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fixed_catalog = (manifest.get("dosage_allele") == "SOURCE_MINOR"
        and manifest.get("genotype_encoding") == contract.FIXED_SOURCE_MINOR_ENCODING
        and manifest.get("candidate_ascertainment") == contract.FIXED_SOURCE_MINOR_CATALOG)
    new_catalog = (manifest.get("dosage_allele") == "ALT"
        and manifest.get("genotype_encoding") == contract.GT_CODE_ENCODING
        and manifest.get("candidate_ascertainment") == contract.STRICT_INNER_FIT_ASCERTAINMENT)
    require(manifest.get("schema") == "r02_k03_outer_fit_alt_chunks_v1"
            and manifest.get("producer", {}).get("all_autosomes_complete") is True
            and manifest.get("outer_fit_learned_inputs") is False
            and fixed_catalog != new_catalog,
            "Complete fixed-catalogue or INNER_FIT catalogue-learning bundle required")
    implementation = manifest.get("consumer_implementation_sha256")
    require(isinstance(implementation, dict) and implementation,
            "Pinned INNER_FIT implementation map required")
    for name, digest in implementation.items():
        path = args.code_dir / name
        require(re.fullmatch(r"r02_[a-z0-9_]+\.py", name) is not None
                and path.is_file() and sha256(path) == digest,
                "INNER_FIT code bundle differs: " + name)
    expected = {f"G{i}_G{j}" for i in range(5) for j in range(i + 1, 5)}
    contexts = []
    for path in sorted(args.plans_dir.glob("fold-*.plan.private.json")):
        plan = json.loads(path.read_text()); fold = plan.get("fold_id")
        require(fold in expected and plan.get("roster_sha256") == manifest["roster"]["sha256"]
                and plan.get("select_evaluate_access_allowed") is False,
                "Pair-complete plan identity or closed-role policy differs")
        contexts.append({"id": fold, "plan": str(path), "plan_sha256": sha256(path)})
    require({row["id"] for row in contexts} == expected and len(contexts) == 10,
            "All ten and only the pair-complete internal contexts are required")
    result = {"schema": "r02_k03_inner_fit_affinity_batch_v2",
              "implementation_sha256": implementation, "container": args.container,
              "resources": {"cpus": 1, "memory_gb": args.memory_gb,
                            "max_forks": args.max_forks, "task_time": args.task_time},
              "shared_bundle": str(args.shared_bundle),
              "shared_manifest_sha256": sha256(manifest_path), "contexts": contexts}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"batch_sha256": sha256(args.output), "contexts": len(contexts)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
