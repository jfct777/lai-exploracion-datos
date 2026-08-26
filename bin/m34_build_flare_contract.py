#!/usr/bin/env python3
"""Freeze one truth-blind FLARE invocation after its inputs are materialized."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INPUT_ARGUMENTS = (
    "reference_vcf", "reference_tbi", "target_vcf", "target_tbi",
    "sample_map", "genetic_map", "flare_jar",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(experiment_path: Path, paths: dict[str, Path]) -> dict[str, Any]:
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    require(experiment.get("experiment_id") == "M34_NAM_EXPLORATORY_CHR22",
            "M34 experiment identity differs")
    require(experiment.get("status") == "CONTRACT_ONLY_NO_REAL_RESULTS",
            "M34 experiment is not frozen before execution")
    require(experiment.get("ancestry_order") == ["AFR", "EUR", "NAM"],
            "M34 ancestry order differs")
    require(set(paths) == set(INPUT_ARGUMENTS), "FLARE input members differ")
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"invalid FLARE input: {name}")
    source = experiment["baseline"]["parameters"]
    parameters = {
        "array": source["array"],
        "probs": source["probs"],
        "em": source["em"],
        "min-mac": source["min_mac"],
        "min-maf": source["min_maf"],
        "gen": source["generations"],
        "update-p": source["update_p"],
        "panel-probs": source["panel_probs"],
        "seed": source["seed"],
        "nthreads": source["nthreads"],
    }
    return {
        "schema_version": "1.0.0",
        "stage": "M34_AFR_EUR_NAM_FLARE",
        "status": "EXPLORATORY_CONTRACT_BLINDED_TO_LABELS",
        "chromosome": experiment["chromosome"],
        "ancestry_names": experiment["ancestry_order"],
        "parameters": parameters,
        "expected_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    for name in INPUT_ARGUMENTS:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name,
                            type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), "refusing to overwrite the FLARE contract")
    paths = {name: getattr(args, name) for name in INPUT_ARGUMENTS}
    payload = build(args.experiment, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": payload["status"],
                      "inputs": len(payload["expected_sha256"])}, sort_keys=True))


if __name__ == "__main__":
    main()
