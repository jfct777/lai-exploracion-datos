#!/usr/bin/env python3
"""Validate M28C B0 VCFs with BGZF/Tabix and the exact Gnomix loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("stage") != "M28C_B0_GNOMIX_INGEST_AUDIT":
        raise ValueError("Unexpected ingest preregistration stage")
    return contract


def authenticate_upstream(
    reference: Path, target: Path, report_path: Path
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("decision") != "GO_EXTERNAL_GNOMIX_INGEST_VALIDATION":
        raise ValueError("Upstream materialization did not pass")
    expected = report.get("output_sha256", {})
    observed = {
        reference.name: sha256(reference),
        target.name: sha256(target),
    }
    for name, digest in observed.items():
        if expected.get(name) != digest:
            raise ValueError(f"Upstream hash mismatch for {name}")
    return observed


def prepare_bgzf(source: Path, destination: Path) -> Path:
    run_command(["bcftools", "view", "--no-version", "-Oz", "-o", str(destination), str(source)])
    run_command(["bcftools", "index", "--tbi", str(destination)])
    run_command(["bcftools", "view", "--no-version", "-h", str(destination)])
    run_command(["bcftools", "index", "-n", str(destination)])
    return Path(f"{destination}.tbi")


def load_with_gnomix(path: Path, gnomix_root: Path, chromosome: str) -> dict:
    sys.path.insert(0, str(gnomix_root))
    from src.utils import read_vcf

    data = read_vcf(str(path), chm=chromosome, fields="*", verbose=False)
    if data is None:
        raise ValueError(f"Gnomix loader returned no data for {path.name}")
    return data


def audit_loaded_pair(reference: dict, target: dict, expected: dict) -> dict:
    import numpy as np

    ref_gt = reference["calldata/GT"]
    target_gt = target["calldata/GT"]
    expected_markers = int(expected["markers"])
    expected_ploidy = int(expected["ploidy"])
    expected_ref_shape = (
        expected_markers,
        int(expected["reference_samples"]),
        expected_ploidy,
    )
    expected_target_shape = (
        expected_markers,
        int(expected["target_samples"]),
        expected_ploidy,
    )
    ref_pos = reference["variants/POS"]
    target_pos = target["variants/POS"]
    ref_alleles = reference["variants/REF"]
    target_alleles = target["variants/REF"]
    ref_alt = reference["variants/ALT"][:, 0]
    target_alt = target["variants/ALT"][:, 0]
    checks = {
        "reference_shape": tuple(ref_gt.shape) == expected_ref_shape,
        "target_shape": tuple(target_gt.shape) == expected_target_shape,
        "position_parity": np.array_equal(ref_pos, target_pos),
        "ref_parity": np.array_equal(ref_alleles, target_alleles),
        "alt_parity": np.array_equal(ref_alt, target_alt),
        "reference_binary_complete": bool(np.isin(ref_gt, (0, 1)).all()),
        "target_binary_complete": bool(np.isin(target_gt, (0, 1)).all()),
        "reference_encoding": bool((ref_alleles == expected["ref"]).all() and (ref_alt == expected["alt"]).all()),
        "target_encoding": bool((target_alleles == expected["ref"]).all() and (target_alt == expected["alt"]).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Gnomix loader parity failed: {checks}")
    return {
        "checks": checks,
        "reference_shape": list(ref_gt.shape),
        "target_shape": list(target_gt.shape),
        "first_position": int(ref_pos[0]),
        "last_position": int(ref_pos[-1]),
    }


def audit(args: argparse.Namespace) -> dict:
    contract = load_contract(args.preregistration)
    if int(contract["root_seed"]) != args.root_seed:
        raise ValueError("Root seed differs from the ingest contract")
    upstream_hashes = authenticate_upstream(args.reference_vcf, args.target_vcf, args.materialization_report)
    observed_commit = run_command(["git", "-C", str(args.gnomix_root), "rev-parse", "HEAD"])
    expected_commit = contract["software"]["gnomix_commit"]
    if observed_commit != expected_commit:
        raise ValueError(f"Gnomix commit mismatch: {observed_commit}")

    args.outdir.mkdir(parents=True, exist_ok=False)
    reference_bgzf = args.outdir / "m28c_b0_reference.vcf.gz"
    target_bgzf = args.outdir / "m28c_b0_target.vcf.gz"
    reference_tbi = prepare_bgzf(args.reference_vcf, reference_bgzf)
    target_tbi = prepare_bgzf(args.target_vcf, target_bgzf)

    chromosome = str(contract["expected"]["chromosome"])
    reference = load_with_gnomix(reference_bgzf, args.gnomix_root, chromosome)
    target = load_with_gnomix(target_bgzf, args.gnomix_root, chromosome)
    loader_audit = audit_loaded_pair(reference, target, contract["expected"])
    bcftools_version = run_command(["bcftools", "--version"]).splitlines()[0]
    import allel

    outputs = (reference_bgzf, reference_tbi, target_bgzf, target_tbi)
    gates = {
        "G0_UPSTREAM": True,
        "G1_SOFTWARE": observed_commit == expected_commit,
        "G2_BGZF_TABIX": all(path.exists() and path.stat().st_size > 0 for path in outputs),
        "G3_EXACT_LOADER": loader_audit["reference_shape"][0] == int(contract["expected"]["markers"]),
        "G4_EFFECTIVE_PARITY": all(loader_audit["checks"].values()),
        "G5_SCOPE": True,
    }
    decision = contract["decision"]["pass"] if all(gates.values()) else contract["decision"]["fail"]
    report = {
        "stage": contract["stage"],
        "scope": contract["scope"],
        "root_seed": args.root_seed,
        "seed_role": contract["seed_role"],
        "contract_sha256": sha256(args.preregistration),
        "upstream_sha256": upstream_hashes,
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "software": {
            "gnomix_commit": observed_commit,
            "bcftools": bcftools_version,
            "scikit_allel": allel.__version__,
            "python": sys.version.split()[0],
        },
        "loader_audit": loader_audit,
        "merged_truth_table_accessed": False,
        "model_training_performed": False,
        "gates": gates,
        "decision": decision,
    }
    report_path = args.outdir / "m28c_b0_gnomix_ingest.public.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-vcf", required=True, type=Path)
    parser.add_argument("--target-vcf", required=True, type=Path)
    parser.add_argument("--materialization-report", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--gnomix-root", required=True, type=Path)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    report = audit(parse_args())
    print(json.dumps({"decision": report["decision"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
