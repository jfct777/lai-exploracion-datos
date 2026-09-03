#!/usr/bin/env python3
"""Run the truth-blind M38B FLARE baseline on F-minus-S660.

The low-level VCF, sample-map, genetic-map, command construction, and FLARE
output checks are reused from the audited M34 implementation.  This adapter
only tightens the experiment identity, counts, output names, and receipt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import m34_run_flare as m34
from m38b_build_flare_contract import (
    FIXED_PARAMETERS,
    INPUT_MEMBERS,
    STAGE,
    STATUS,
    M38BFlareContractError,
    require,
    sha256_file,
)


RUN_STATUS = "PASS_TRUTH_BLIND_FLARE_F_MINUS_S660_FIT"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(
        set(contract)
        == {
            "schema_version",
            "experiment_id",
            "stage",
            "status",
            "scope",
            "ancestry_names",
            "expected_shape",
            "parameters",
            "expected_sha256",
            "experiment_contract_sha256",
        },
        "M38B FLARE contract members differ",
    )
    require(contract["stage"] == STAGE and contract["status"] == STATUS,
            "M38B FLARE contract identity differs")
    scope = contract["scope"]
    require(
        scope == {
            "claim_level": "exploratory",
            "chromosome": "22",
            "mosaic_root": "R0",
            "target_partition": "FIT",
            "valid_opened": False,
            "test_opened": False,
            "truth_available_to_stage": False,
        },
        "M38B FLARE contract is not truth-blind FIT-only chr22 R0",
    )
    require(contract["ancestry_names"] == ["AFR", "EUR", "NAM"],
            "M38B ancestry order differs")
    require(
        contract["expected_shape"]
        == {
            "marker_count": 42326,
            "reference_sample_count": 753,
            "target_sample_count": 96,
        },
        "M38B expected FLARE shape differs",
    )
    require(contract["parameters"] == FIXED_PARAMETERS,
            "M38B FLARE parameters differ from M34")
    require(
        set(contract["expected_sha256"]) == set(INPUT_MEMBERS)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and set(value).issubset("0123456789abcdef")
            for value in contract["expected_sha256"].values()
        ),
        "M38B input hashes differ",
    )
    return contract


def verify_inputs(
    contract: Mapping[str, Any], inputs: Mapping[str, Path]
) -> dict[str, str]:
    require(set(inputs) == set(INPUT_MEMBERS), "runtime FLARE inputs differ")
    observed = {name: sha256_file(inputs[name]) for name in INPUT_MEMBERS}
    require(observed == contract["expected_sha256"], "runtime FLARE hashes differ")
    return observed


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(not args.outdir.exists(), "refusing to overwrite the output directory")
    contract = load_contract(args.contract)
    inputs = {
        "reference_vcf": args.reference_vcf,
        "reference_tbi": args.reference_tbi,
        "target_vcf": args.target_vcf,
        "target_tbi": args.target_tbi,
        "sample_map": args.sample_map,
        "genetic_map": args.genetic_map,
        "flare_jar": args.flare_jar,
    }
    observed_hashes = verify_inputs(contract, inputs)
    reference = m34.scan_vcf(args.reference_vcf, "22")
    target = m34.scan_vcf(args.target_vcf, "22")
    expected = contract["expected_shape"]
    require(reference["loci"] == target["loci"],
            "REF/TARGET locus and allele axes differ")
    require(set(reference["samples"]).isdisjoint(target["samples"]),
            "REF and TARGET sample sets overlap")
    require(len(reference["loci"]) == expected["marker_count"],
            "F-minus-S660 marker count differs")
    require(len(reference["samples"]) == expected["reference_sample_count"],
            "reference sample count differs")
    require(len(target["samples"]) == expected["target_sample_count"],
            "FIT target sample count differs")
    require(reference["minimum_mac"] >= contract["parameters"]["min-mac"],
            "REF contains a marker that FLARE min-mac would remove")

    args.outdir.mkdir(parents=True, exist_ok=False)
    normalized_panel = args.outdir / "m38b.ref-panel.tsv"
    normalized_map = args.outdir / "m38b.flare.map"
    panel_audit = m34.normalize_sample_map(
        args.sample_map,
        normalized_panel,
        reference["samples"],
        contract["ancestry_names"],
    )
    map_audit = m34.normalize_genetic_map(
        args.genetic_map,
        normalized_map,
        "22",
        target["vcf_chromosome"],
        target["first_bp"],
        target["last_bp"],
    )
    prefix = args.outdir / "m38b_f_minus_s660"
    command = m34.build_command(
        args.java,
        args.flare_jar,
        args.reference_vcf,
        args.target_vcf,
        normalized_panel,
        normalized_map,
        prefix,
        contract["parameters"],
    )
    ancestry_audit = None
    status = "PASS_M38B_FLARE_PREFLIGHT_ONLY"
    if not args.preflight_only:
        subprocess.run(command, check=True)
        ancestry_path = Path(f"{prefix}.anc.vcf.gz")
        ancestry_audit = m34.audit_ancestry_vcf(
            ancestry_path, target, contract["ancestry_names"]
        )
        status = RUN_STATUS
    receipt = {
        "schema_version": "1.0.0",
        "stage": STAGE,
        "status": status,
        "claim_level": "exploratory",
        "scope": contract["scope"],
        "ancestry_names": contract["ancestry_names"],
        "shape": {
            "marker_count": len(target["loci"]),
            "reference_sample_count": len(reference["samples"]),
            "target_sample_count": len(target["samples"]),
            "reference_panel_counts": panel_audit["ancestry_counts"],
        },
        "parameters": contract["parameters"],
        "command_argv": command,
        "input_sha256": observed_hashes,
        "contract_sha256": sha256_file(args.contract),
        "derived_input_audit": {
            "sample_map": panel_audit,
            "genetic_map": map_audit,
        },
        "ancestry_vcf_audit": ancestry_audit,
        "truth_argument_available": False,
        "truth_accessed": False,
        "scoring_performed": False,
        "preflight_only": bool(args.preflight_only),
        "wall_seconds": time.monotonic() - started,
    }
    receipt_path = args.outdir / "m38b_flare.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--reference-vcf", type=Path, required=True)
    parser.add_argument("--reference-tbi", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--target-tbi", type=Path, required=True)
    parser.add_argument("--sample-map", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--flare-jar", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        receipt = run(parse_args())
    except m34.FlareContractError as exc:
        raise M38BFlareContractError(str(exc)) from exc
    print(json.dumps({"status": receipt["status"], "shape": receipt["shape"]}, sort_keys=True))


if __name__ == "__main__":
    main()
