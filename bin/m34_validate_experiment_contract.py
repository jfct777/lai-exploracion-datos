#!/usr/bin/env python3
"""Authenticate an exact M34 NAM experiment contract before execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_ANCESTRIES = ("AFR", "EUR", "NAM")
EXPECTED_MIXTURE = {"AFR": 0.25, "EUR": 0.60, "NAM": 0.15}
EXPECTED_SEEDS = {
    "R0_FIT": 1439610605,
    "R0_VALID": 1702577247,
    "R1_FIT": 667875703,
    "R1_VALID": 513710823,
    "R2_FIT": 348301061,
    "R2_VALID": 1179260632,
}
EXPECTED_TARGET_SIZES = {
    "small": {"people": 32, "fit": 24, "valid": 8},
    "pilot_128": {"people": 128, "fit": 96, "valid": 32},
    "medium": {"people": 256, "fit": 192, "valid": 64},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def validate(
    contract_path: Path,
    expected_sha256: str,
    root: str = "R0",
    target_size: str = "small",
) -> dict[str, Any]:
    require(contract_path.is_file() and not contract_path.is_symlink(),
            "experiment contract must be a regular non-symbolic file")
    observed_sha256 = sha256_file(contract_path)
    require(observed_sha256 == expected_sha256,
            "M34 experiment contract SHA-256 differs")
    contract = json.loads(
        contract_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    require(contract.get("schema_version") == "1.0.0" and
            contract.get("experiment_id") == "M34_NAM_EXPLORATORY_CHR22" and
            contract.get("status") == "CONTRACT_ONLY_NO_REAL_RESULTS" and
            contract.get("claim_level") == "exploratory",
            "M34 experiment identity differs")
    require(str(contract.get("chromosome")).removeprefix("chr") == "22" and
            tuple(contract.get("ancestry_order", ())) == EXPECTED_ANCESTRIES,
            "M34 chromosome or ancestry axis differs")

    mosaics = contract.get("mosaics", {})
    mixture = mosaics.get("primary_mixture_proportions", {})
    require(set(mixture) == set(EXPECTED_MIXTURE) and all(
        math.isclose(float(mixture[name]), expected, rel_tol=0.0, abs_tol=1e-12)
        for name, expected in EXPECTED_MIXTURE.items()
    ), "M34 primary mosaic mixture differs")
    require(float(mosaics.get("primary_admixture_generations", -1)) == 12.0,
            "M34 primary admixture generations differ")
    require(root in {"R0", "R1", "R2"}, "M34 root is not declared")
    require(target_size in EXPECTED_TARGET_SIZES, "M34 target size is not declared")
    seeds = mosaics.get("seeds", {})
    require(seeds == {"derivation": seeds.get("derivation"), **EXPECTED_SEEDS} and
            seeds.get("derivation") ==
            "uint31(first_4_bytes_sha256('DNABR_M34|MOSAIC|ROLE'))",
            "M34 mosaic seeds differ")
    target_sizes = mosaics.get("target_sizes", {})
    for name, expected in EXPECTED_TARGET_SIZES.items():
        observed = target_sizes.get(name, {})
        require(all(observed.get(key) == value for key, value in expected.items()),
                f"M34 {name} target sizes differ")
    sizes = target_sizes[target_size]

    roles = contract.get("roles", {})
    require(roles.get("reference_and_frequency") == "REF_TRAIN" and
            roles.get("mosaic_fit_donors") == "SOURCE_VALID" and
            roles.get("mosaic_valid_donors") == "SOURCE_TEST",
            "M34 role mapping differs")
    rare = contract.get("rare_definition", {})
    require(rare.get("frequency_scope") == "REF_TRAIN_only" and
            rare.get("minimum_mac") == 2 and
            float(rare.get("maximum_maf_exclusive", -1)) == 0.01 and
            rare.get("target_genotypes_used_for_selection") is False and
            rare.get("truth_or_baseline_used_for_selection") is False,
            "M34 rare-variant definition differs")
    require(contract.get("paired_arms", {}).get("estimand") == "RE_minus_RD",
            "M34 paired-arm estimand differs")
    baseline = contract.get("baseline", {}).get("parameters", {})
    require(float(baseline.get("generations", -1)) == 12.0 and
            baseline.get("min_mac") == 1 and baseline.get("nthreads") == 4,
            "M34 FLARE baseline parameters differ")

    return {
        "schema_version": "1.0.0",
        "stage": "M34_VALIDATE_EXPERIMENT_CONTRACT",
        "status": "PASS_EXACT_SELECTED_ROOT_AND_SIZE_CONTRACT",
        "experiment_contract_sha256": observed_sha256,
        "chromosome": "22",
        "ancestry_order": list(EXPECTED_ANCESTRIES),
        "mixture_proportions": EXPECTED_MIXTURE,
        "admixture_generations": 12.0,
        "root": root,
        "target_size": target_size,
        "splits": {
            "FIT": {
                "donor_role": "SOURCE_VALID",
                "seed": seeds[f"{root}_FIT"],
                "people": sizes["fit"],
            },
            "VALID": {
                "donor_role": "SOURCE_TEST",
                "seed": seeds[f"{root}_VALID"],
                "people": sizes["valid"],
            },
        },
        "rare_definition": {"minimum_mac": 2, "maximum_maf_exclusive": 0.01},
        "paired_arms": ["RD", "RE"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--root", required=True, choices=("R0", "R1", "R2"))
    parser.add_argument(
        "--target-size", required=True, choices=tuple(EXPECTED_TARGET_SIZES),
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), "refusing to overwrite validation receipt")
    receipt = validate(args.contract, args.expected_sha256, args.root, args.target_size)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": receipt["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
