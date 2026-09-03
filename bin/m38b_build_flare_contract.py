#!/usr/bin/env python3
"""Freeze the truth-blind FLARE invocation for the M38B F-minus-S660 baseline.

This adapter keeps the tested M34 FLARE implementation as the execution core,
but gives the new baseline its own semantic contract.  It deliberately has no
truth, score, VALID, or TEST argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = "M38B_S660_INCREMENTAL_LAI_CHR22_R0_FIT"
EXPERIMENT_STATUS = "PREREGISTERED_AMENDED_BEFORE_OUTCOME_ACCESS"
STAGE = "M38B_F_MINUS_S660_FLARE"
STATUS = "CONTRACT_FROZEN_TRUTH_BLIND_FIT_ONLY"
INPUT_MEMBERS = (
    "reference_vcf",
    "reference_tbi",
    "target_vcf",
    "target_tbi",
    "sample_map",
    "genetic_map",
    "flare_jar",
)
FIXED_PARAMETERS = {
    "array": False,
    "probs": True,
    "em": True,
    "min-mac": 1,
    "min-maf": 0.0,
    "gen": 12.0,
    "update-p": False,
    "panel-probs": False,
    "seed": 3401103,
    "nthreads": 4,
}
HEX_DIGITS = frozenset("0123456789abcdef")


class M38BFlareContractError(ValueError):
    """Raised when M38B inputs or frozen semantics differ."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BFlareContractError(message)


def sha256_file(path: Path) -> str:
    # Nextflow may stage immutable inputs as symlinks.  Content authentication,
    # not the staging mechanism, defines identity here.
    require(path.is_file(), f"invalid regular input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256(value: str, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS),
        f"{label} SHA-256 is malformed",
    )
    return value


def validate_experiment(experiment: Mapping[str, Any]) -> None:
    require(experiment.get("experiment_id") == EXPERIMENT_ID, "M38B experiment differs")
    require(experiment.get("status") == EXPERIMENT_STATUS, "M38B status differs")
    scope = experiment.get("claim_scope", {})
    require(
        scope.get("analysis_level") == "EXPLORATORY"
        and str(scope.get("chromosome", "")).removeprefix("chr") == "22"
        and scope.get("mosaic_root") == "R0"
        and scope.get("target_partition") == "FIT_ONLY"
        and scope.get("target_people") == 96
        and scope.get("valid_opened") is False
        and scope.get("test_opened") is False,
        "M38B scope is not R0 chr22 FIT-only with VALID/TEST closed",
    )
    universes = experiment.get("locus_universes", {})
    require(
        universes.get("f_full_count") == 42986
        and universes.get("s660_count") == 660
        and universes.get("f_minus_s660_count") == 42326
        and universes.get("f_minus_is_common_only") is False
        and universes.get("variant_key") == ["CHROM", "POS", "REF", "ALT"],
        "M38B locus-universe contract differs",
    )
    parameters = experiment.get("flare_parameters", {})
    observed = {
        "array": parameters.get("array"),
        "probs": parameters.get("probs"),
        "em": parameters.get("em"),
        "min-mac": parameters.get("min_mac"),
        "min-maf": parameters.get("min_maf"),
        "gen": parameters.get("generations"),
        "update-p": parameters.get("update_p"),
        "panel-probs": parameters.get("panel_probs"),
        "seed": parameters.get("seed"),
        "nthreads": parameters.get("nthreads"),
    }
    require(observed == FIXED_PARAMETERS, "M38B FLARE parameters differ from M34")


def build_contract(
    *,
    experiment_path: Path,
    inputs: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
) -> dict[str, Any]:
    require(set(inputs) == set(INPUT_MEMBERS), "runtime FLARE input members differ")
    require(set(expected_sha256) == set(INPUT_MEMBERS), "expected input hashes differ")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    validate_experiment(experiment)
    observed: dict[str, str] = {}
    for name in INPUT_MEMBERS:
        wanted = validate_sha256(expected_sha256[name], name)
        observed[name] = sha256_file(inputs[name])
        require(observed[name] == wanted, f"SHA-256 mismatch for {name}")
    source = experiment.get("source_artifacts", {})
    require(
        source.get("f_minus_reference_vcf_sha256") == observed["reference_vcf"]
        and source.get("f_minus_target_vcf_sha256") == observed["target_vcf"],
        "F-minus-S660 VCF hashes differ from the preregistration",
    )
    return {
        "schema_version": "1.0.0",
        "experiment_id": EXPERIMENT_ID,
        "stage": STAGE,
        "status": STATUS,
        "scope": {
            "claim_level": "exploratory",
            "chromosome": "22",
            "mosaic_root": "R0",
            "target_partition": "FIT",
            "valid_opened": False,
            "test_opened": False,
            "truth_available_to_stage": False,
        },
        "ancestry_names": ["AFR", "EUR", "NAM"],
        "expected_shape": {
            "marker_count": 42326,
            "reference_sample_count": 753,
            "target_sample_count": 96,
        },
        "parameters": dict(FIXED_PARAMETERS),
        "expected_sha256": observed,
        "experiment_contract_sha256": sha256_file(experiment_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    for name in INPUT_MEMBERS:
        option = name.replace("_", "-")
        parser.add_argument(f"--{option}", dest=name, type=Path, required=True)
        parser.add_argument(f"--{option}-sha256", dest=f"{name}_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), "refusing to overwrite the FLARE contract")
    inputs = {name: getattr(args, name) for name in INPUT_MEMBERS}
    hashes = {name: getattr(args, f"{name}_sha256") for name in INPUT_MEMBERS}
    payload = build_contract(
        experiment_path=args.experiment,
        inputs=inputs,
        expected_sha256=hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "inputs": len(inputs)}, sort_keys=True))


if __name__ == "__main__":
    main()
