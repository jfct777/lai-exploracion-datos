#!/usr/bin/env python3
"""Fail-closed preflight for the synthetic M33 T1 backward dry run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_T0B_SHA256 = "4b62813194795c18483110ad43271ba187c2f032e5b2c3c9a3c7a1f401e62d5b"
EXPECTED_CASES = {
    ("local_linear", 0), ("local_linear", 1),
    ("small_residual_cnn_1d", 0), ("small_residual_cnn_1d", 1),
}
EXPECTED_OCI = ("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/"
                "m33-t0a@sha256:c03864a9ed0c56b00fd1a234daee2d17ddfa57d4c426628bd59cd9daf351ee99")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object differs: {path}")
    return value


def validate_contract(path: Path) -> tuple[str, dict[str, Any]]:
    contract = load_json(path)
    scope = contract.get("scope", {})
    stress = contract.get("synthetic_stress", {})
    execution = contract.get("execution", {})
    require(
        contract.get("stage") == "M33_T1_BACKWARD_DRY_RUN_CONTRACT" and
        contract.get("status") == "FROZEN_BEFORE_EXECUTION" and
        {tuple(row) for row in scope.get("cases", [])} == EXPECTED_CASES and
        scope.get("synthetic_only") is True and
        scope.get("subcases") == ["production_zero_head", "private_nonzero_probe_head"] and
        stress.get("people") == 8 and stress.get("central_markers") == 256 and
        stress.get("rows") == 2048 and stress.get("channels") == 13 and
        stress.get("maximum_padded_tokens") == 262_144 and
        stress.get("boundary_loss_weight_beta") == 1.0 and
        stress.get("known_answer_boundary_loss_weights") == [0.0, 1.0] and
        stress.get("target_class_counts") == [1366, 1365, 1365] and
        stress.get("transition_count") == 112 and
        stress.get("transition_count_per_person_haplotype") == 7 and
        stress.get("boundary_weight_sum") == 4208.0 and
        execution.get("oci_image") == EXPECTED_OCI and
        execution.get("process_memory_gib") == 8 and
        execution.get("memory_warning_fraction") == 0.70 and
        execution.get("memory_stop_fraction") == 0.80 and
        execution.get("maximum_parallel_processes") == 1 and
        execution.get("minimum_preflight_mem_available_gib") == 10,
        "T1 frozen contract differs")
    return sha256_file(path), contract


def validate_source_auth(path: Path, commit: str, source_root: Path) -> str:
    payload = load_json(path)
    required = {
        "bin/m33_t0a_models.py", "bin/m33_t1_source_auth.py",
        "bin/m33_t1_preflight.py", "bin/m33_t1_backward.py",
        "bin/m33_t1_compare.py", "conf/m33_pre4_preregistration.json",
        "conf/m33_t1_contract.json", "conf/m33_t1.config",
        "modules/33_T1_BACKWARD_DRY_RUN.nf", "workflows/m33_t1.nf",
        "containers/m33-t0a/Dockerfile", "tests/test_m33_t1.py",
    }
    hashes = payload.get("source_sha256", {})
    require(payload.get("stage") == "M33_T1_SOURCE_AUTH" and
            payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            payload.get("git_commit") == commit and set(hashes) == required and
            all(re.fullmatch(r"[0-9a-f]{64}", value or "") for value in hashes.values()),
            "T1 source authentication differs")
    for relative in ("bin/m33_t1_preflight.py", "conf/m33_t1_contract.json"):
        require(hashes[relative] == sha256_file(source_root / relative),
                f"authenticated T1 preflight source differs: {relative}")
    return sha256_file(path)


def mem_available_gib(path: Path) -> float:
    rows = path.read_text(encoding="ascii").splitlines()
    matches = [row for row in rows if row.startswith("MemAvailable:")]
    require(len(matches) == 1, "MemAvailable is unavailable")
    return int(matches[0].split()[1]) / (1024.0 ** 2)


def validate_t0b(aggregate_path: Path, child_paths: list[Path]) -> None:
    require(sha256_file(aggregate_path) == EXPECTED_T0B_SHA256,
            "T0b aggregate hash differs")
    aggregate = load_json(aggregate_path)
    require(aggregate.get("stage") == "M33_T0B_FULL_CHR22_COMPARISON" and
            aggregate.get("status") == "PASS_T0B_FULL_CHR22_TECHNICAL_ONLY" and
            aggregate.get("marker_count") == 79_791 and
            aggregate.get("t1_open") is False and
            aggregate.get("truth_read") is False and
            aggregate.get("training") is False and
            aggregate.get("gradients") is False and
            aggregate.get("optimizer") is False and
            len(child_paths) == 5, "T0b closure differs")
    expected = {row["name"]: row["sha256"] for row in aggregate.get("child_receipts", [])}
    observed = {path.name: sha256_file(path) for path in child_paths}
    require(len(expected) == 5 and observed == expected,
            "T0b child receipt inventory differs")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit or "") is not None,
            "T1 implementation commit differs")
    contract_sha, contract = validate_contract(args.contract)
    require(args.oci_image == contract["execution"]["oci_image"], "T1 OCI image differs")
    source_auth_sha = validate_source_auth(
        args.source_auth, args.implementation_commit, args.source_root)
    validate_t0b(args.t0b_aggregate, args.t0b_child_receipt)
    available = mem_available_gib(args.meminfo)
    minimum = contract["execution"]["minimum_preflight_mem_available_gib"]
    require(available >= minimum, "T1 MemAvailable is below the sequential gate")
    return {
        "stage": "M33_T1_PREFLIGHT",
        "status": "PASS_T1_PREFLIGHT_SYNTHETIC_ONLY",
        "implementation_commit": args.implementation_commit,
        "oci_image": args.oci_image,
        "contract_sha256": contract_sha,
        "source_auth_sha256": source_auth_sha,
        "t0b_aggregate_sha256": sha256_file(args.t0b_aggregate),
        "t0b_child_receipt_count": len(args.t0b_child_receipt),
        "case_inventory": [list(row) for row in sorted(EXPECTED_CASES)],
        "maximum_parallel_processes": 1,
        "minimum_mem_available_gib": minimum,
        "mem_available_gib": available,
        "synthetic_only": True,
        "truth_read": False,
        "real_data_read": False,
        "training": False,
        "optimizer": False,
        "checkpoint_written": False,
        "predictions_persisted": False,
        "scientific_evidence": False,
        "consumable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--t0b-aggregate", type=Path, required=True)
    parser.add_argument("--t0b-child-receipt", action="append", type=Path, required=True)
    parser.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run_preflight(args), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
