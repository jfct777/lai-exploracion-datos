#!/usr/bin/env python3
"""Create a fail-closed provenance receipt for an M36B training result."""

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


def descriptor(path: Path, planned_prefix: str | None = None) -> dict:
    if not path.is_file():
        raise SystemExit(f"M36B receipt error: missing artifact {path}")
    result = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
    if planned_prefix:
        result["planned_uri"] = f"{planned_prefix.rstrip('/')}/{path.name}"
    return result


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw = value.split("=", 1)
    if not name or not raw:
        raise argparse.ArgumentTypeError("expected nonempty NAME=PATH")
    return name, Path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--train-summary", required=True, type=Path)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--design-contract", required=True, type=Path)
    parser.add_argument("--code", required=True, action="append", type=Path)
    parser.add_argument("--input", required=True, action="append", type=named_path)
    parser.add_argument("--output", required=True, action="append", type=Path)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    materialization = json.loads(args.materialization_receipt.read_text(encoding="utf-8"))
    summary = json.loads(args.train_summary.read_text(encoding="utf-8"))
    contract = json.loads(args.design_contract.read_text(encoding="utf-8"))
    if materialization.get("status") not in {"MATERIALIZED_PASS", "PUBLISHED_PASS"}:
        raise SystemExit("M36B receipt error: source materialization did not pass")
    if summary.get("stage") != "M36B_CORA_SET_TRAIN" or summary.get("status") != "TRAINED_EXPLORATORY":
        raise SystemExit("M36B receipt error: training summary is not a completed M36B result")
    parameters = summary.get("effective_parameters")
    if not isinstance(parameters, dict) or not parameters.get("train_seeds") or not parameters.get("halving_budgets"):
        raise SystemExit("M36B receipt error: effective budgets or seeds are absent")
    if contract.get("stage") != "M36B_CORA_SET_EXPLORATORY":
        raise SystemExit("M36B receipt error: design contract stage drift")
    promotion = contract.get("promotion", {})
    if (
        promotion.get("minimum_relative_mse_reduction") != parameters.get("minimum_relative_mse_reduction")
        or promotion.get("minimum_positive_outer_folds") != parameters.get("minimum_positive_folds")
    ):
        raise SystemExit("M36B receipt error: promotion parameters differ from the design contract")
    null_gate = contract.get("controls", {}).get("carrier_permutation_mixing_gate", {})
    if (
        null_gate.get("minimum_moved_carrier_fraction_per_outer_partition")
        != parameters.get("minimum_moved_carrier_fraction")
    ):
        raise SystemExit("M36B receipt error: permutation mixing gate differs from the design contract")
    positive_contract = contract.get("pre_real_positive_controls", {})
    positive_summary = summary.get("pre_real_positive_controls", {})
    if (
        positive_contract.get("controls") != parameters.get("positive_controls")
        or positive_contract.get("budgets") != parameters.get("positive_control_budgets")
        or positive_contract.get("seed") != parameters.get("positive_control_seed")
        or positive_summary.get("required_controls") != positive_contract.get("controls")
        or positive_summary.get("executed_before_real_data_read") is not True
        or positive_summary.get("all_passed") is not True
    ):
        raise SystemExit("M36B receipt error: pre-real positive-control contract was not satisfied")
    positive_runs = positive_summary.get("runs", {})
    if set(positive_runs) != set(positive_contract.get("controls", [])) or any(
        run.get("promotion_gate", {}).get("passed") is not True
        for run in positive_runs.values()
    ):
        raise SystemExit("M36B receipt error: additive and interaction positive controls must both pass")

    inputs = dict(args.input)
    required_inputs = {"loci", "carriers", "missing", "covariates", "components", "targets"}
    if set(inputs) != required_inputs:
        raise SystemExit("M36B receipt error: input set differs from the six materialized tables")
    source_descriptors = materialization.get("input_descriptors", {})
    for name, path in inputs.items():
        expected = source_descriptors.get(name, {}).get("sha256")
        observed = sha256(path)
        if expected != observed:
            raise SystemExit(f"M36B receipt error: {name} differs from materialization receipt")

    output_names = [path.name for path in args.output]
    if len(output_names) != len(set(output_names)) or args.train_summary.name not in output_names:
        raise SystemExit("M36B receipt error: outputs must be unique and include the train summary")
    payload = {
        "stage": "M36B_CORA_SET_TRAIN",
        "status": "TRAINED_EXPLORATORY_PROVENANCE_PASS",
        "scope": "genealogy/structure screen on cross-chromosome common-IBD; not LAI validation",
        "source_materialization_receipt": descriptor(args.materialization_receipt),
        "source_materialization_input_descriptors": source_descriptors,
        "inputs": {name: descriptor(path) for name, path in sorted(inputs.items())},
        "run_config": descriptor(args.run_config),
        "design_contract": descriptor(args.design_contract),
        "code": {path.name: descriptor(path) for path in sorted(args.code)},
        "effective_parameters": parameters,
        "pre_real_positive_controls": positive_summary,
        "outputs": {
            path.name: descriptor(path, args.output_prefix) for path in sorted(args.output)
        },
        "controls": ["rare_enabled", "carrier_permuted", "geometry_only", "baseline_only"],
        "architecture_selection": summary.get("architecture_selection"),
        "uncertainty": summary.get("uncertainty"),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
