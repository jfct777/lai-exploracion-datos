#!/usr/bin/env python3
"""Run and audit frozen FLARE 0.6 on one truth-blind M33 DEVELOPMENT root."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from m30_flare_baseline import (
    EXPECTED_FLARE,
    EXPECTED_PARAMS,
    audit_flare_log,
    audit_flare_model,
    audit_flare_vcf,
    build_flare_command,
    convert_genetic_map,
    normalize_panel_map,
    scan_vcf,
    sha256,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(args: argparse.Namespace) -> dict:
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    require(args.root_seed in pre4["root_registry"]["DEVELOPMENT"],
            "root is not registered for DEVELOPMENT")
    require(sha256(args.flare_jar) == EXPECTED_FLARE["jar_sha256"], "FLARE JAR hash drift")
    reference = scan_vcf(args.reference_vcf, "22")
    target = scan_vcf(args.target_vcf, "22")
    require(reference["loci"] == target["loci"], "REF/TARGET locus axes differ")
    require(len(reference["loci"]) == 79791, "FLARE marker count drift")
    require(len(reference["samples"]) == 90, "REF sample count drift")
    require(len(target["samples"]) == 30, "TARGET sample count drift")
    require(set(reference["samples"]).isdisjoint(target["samples"]), "REF/TARGET overlap")
    require(reference["minimum_mac"] >= 1, "FLARE min-mac would remove a marker")
    require(not args.outdir.exists(), f"refusing to overwrite {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=False)

    panel = args.outdir / "flare.ref-panel.tsv"
    panel_audit = normalize_panel_map(
        args.sample_map, panel, reference["samples"], ["AFR", "EUR", "ASIA"],
        {"AFR": 30, "EUR": 30, "ASIA": 30},
    )
    map4 = args.outdir / "flare.map"
    map_rows = sum(1 for line in args.genetic_map.read_text(encoding="utf-8").splitlines()
                   if line.strip())
    map_audit = convert_genetic_map(
        args.genetic_map, map4, "22", map_rows, target["first_bp"], target["last_bp"]
    )
    prefix = args.outdir / "m33"
    command = build_flare_command(
        args.java, args.flare_jar, args.reference_vcf, args.target_vcf,
        panel, map4, prefix, EXPECTED_PARAMS,
    )
    subprocess.run(command, check=True)
    outputs = {
        "ancestry_vcf": Path(f"{prefix}.anc.vcf.gz"),
        "global_ancestry": Path(f"{prefix}.global.anc.gz"),
        "model": Path(f"{prefix}.model"),
        "log": Path(f"{prefix}.log"),
    }
    for name, path in outputs.items():
        require(path.is_file() and path.stat().st_size > 0, f"missing FLARE output {name}")
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M33_DEVELOPMENT_FLARE",
        "status": "PASS_TRUTH_BLIND_FLARE",
        "root_seed": args.root_seed,
        "pre4_sha256": sha256(args.pre4),
        "inputs_sha256": {
            "reference_vcf": sha256(args.reference_vcf),
            "reference_tbi": sha256(args.reference_tbi),
            "target_vcf": sha256(args.target_vcf),
            "target_tbi": sha256(args.target_tbi),
            "sample_map": sha256(args.sample_map),
            "genetic_map": sha256(args.genetic_map),
        },
        "flare": {
            **EXPECTED_FLARE,
            "parameters": EXPECTED_PARAMS,
            "command_argv": command,
        },
        "panel_audit": panel_audit,
        "map_audit": map_audit,
        "ancestry_vcf_audit": audit_flare_vcf(outputs["ancestry_vcf"], args.target_vcf,
                                                ["AFR", "EUR", "ASIA"]),
        "log_audit": audit_flare_log(outputs["log"], EXPECTED_PARAMS),
        "model_audit": audit_flare_model(outputs["model"], ["AFR", "EUR", "ASIA"]),
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
        "truth_argument_available": False,
        "truth_accessed": False,
        "scoring_performed": False,
    }
    (args.outdir / "m33_flare.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--reference-vcf", type=Path, required=True)
    parser.add_argument("--reference-tbi", type=Path, required=True)
    parser.add_argument("--target-vcf", type=Path, required=True)
    parser.add_argument("--target-tbi", type=Path, required=True)
    parser.add_argument("--sample-map", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--flare-jar", type=Path, default=Path("/opt/flare/flare.jar"))
    parser.add_argument("--java", default="java")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "root_seed": result["root_seed"]}, sort_keys=True))
