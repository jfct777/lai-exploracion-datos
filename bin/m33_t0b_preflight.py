#!/usr/bin/env python3
"""Fail-closed preflight for the receipt-only M33 T0b full-chr22 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_AGGREGATE_SHA256 = "28035d231f4c7be2040191ff7bd78092244506c33127e607878f05dd8f3d4915"
EXPECTED_T0A_SOURCE_AUTH_SHA256 = "7c80f58c394b05ebf4c9a1272aa9bb25df23de47d5765a755e7ce6a1060e901b"
EXPECTED_MARKERS = 79_791
EXPECTED_CHILDREN = 12
MINIMUM_MEM_AVAILABLE_GIB = 26.0
PROCESS_MEMORY_GIB = 8
MAXIMUM_PARALLEL_FORWARD_PROCESSES = 3
NPZ_NAMES = (
    "technical_kat_flare_f0_sanitized.npz",
    "technical_kat_reference_rare_summary_incremental.npz",
    "technical_kat_selected_loci_incremental.npz",
    "technical_kat_target_rare_diploid_incremental.npz",
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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON object differs: {path}")
    return payload


def mem_available_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    rows = meminfo.read_text(encoding="ascii").splitlines()
    values = {row.split(":", 1)[0]: int(row.split()[1]) for row in rows if ":" in row}
    require("MemAvailable" in values, "MemAvailable is absent")
    return values["MemAvailable"] * 1024 / (1024.0 ** 3)


def marker_count(technical_dir: Path) -> int:
    path = technical_dir / "technical_kat_flare_f0_sanitized.npz"
    require(path.is_file() and not path.is_symlink(), f"technical F0 differs: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {
            "sample_key_sha256", "chrom", "pos", "ref", "alt", "probabilities"},
            "technical F0 marker members differ")
        count = int(archive["pos"].size)
        require(archive["pos"].ndim == 1 and
                archive["probabilities"].ndim == 4 and
                archive["probabilities"].shape[2] == count,
                "technical F0 marker axis differs")
    return count


def authenticate_root_inputs(root_label: str, technical_dir: Path, verify_receipt: Path,
                             genetic_map: Path, expected: dict[str, Any]) -> dict[str, Any]:
    expected_npz = expected.get("technical_npz_sha256", {})
    require(set(expected_npz) == set(NPZ_NAMES), f"{root_label} NPZ contract differs")
    observed_npz = {name: sha256_file(technical_dir / name) for name in NPZ_NAMES}
    bridge = technical_dir / "safe_bridge_technical_kat.receipt.json"
    require(observed_npz == expected_npz and
            sha256_file(bridge) == expected.get("bridge_receipt_sha256") and
            sha256_file(verify_receipt) == expected.get("independent_verify_receipt_sha256") and
            sha256_file(genetic_map) == expected.get("genetic_map_sha256"),
            f"{root_label} input hash differs")
    with np.load(technical_dir / "technical_kat_selected_loci_incremental.npz",
                 allow_pickle=False) as selected:
        rare_locus_count = int(selected["locus_key_sha256"].size)
    with np.load(technical_dir / "technical_kat_target_rare_diploid_incremental.npz",
                 allow_pickle=False) as target:
        target_count = int(target["sample_key_sha256"].size)
    markers = marker_count(technical_dir)
    require(target_count == expected.get("target_count") and
            rare_locus_count == expected.get("rare_locus_count") and
            markers == EXPECTED_MARKERS, f"{root_label} input axes differ")
    return {
        "target_count": target_count, "rare_locus_count": rare_locus_count,
        "marker_count": markers, "bridge_receipt_sha256": sha256_file(bridge),
        "independent_verify_receipt_sha256": sha256_file(verify_receipt),
        "genetic_map_sha256": sha256_file(genetic_map),
        "technical_npz_sha256": observed_npz,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    source_auth = load_json(args.source_auth)
    require(source_auth.get("stage") == "M33_T0B_SOURCE_AUTH" and
            source_auth.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            source_auth.get("git_commit") == args.implementation_commit and
            source_auth.get("source_sha256", {}).get("bin/m33_t0b_preflight.py") ==
            sha256_file(args.source_root / "bin/m33_t0b_preflight.py"),
            "T0b preflight source authentication differs")
    source_auth_sha = sha256_file(args.source_auth)
    contract = load_json(args.contract)
    require(contract.get("stage") == "M33_T0B_FULL_CHR22_CONTRACT" and
            contract.get("status") == "FROZEN_BEFORE_EXECUTION",
            "T0b contract identity differs")
    execution = contract.get("execution", {})
    require(execution.get("process_memory_gib") == PROCESS_MEMORY_GIB and
            execution.get("maximum_parallel_forward_processes") ==
            MAXIMUM_PARALLEL_FORWARD_PROCESSES and
            execution.get("minimum_preflight_mem_available_gib") ==
            MINIMUM_MEM_AVAILABLE_GIB,
            "T0b contract resource identity differs")
    require(sha256_file(args.t0a_aggregate) == EXPECTED_AGGREGATE_SHA256 and
            sha256_file(args.t0a_source_auth) == EXPECTED_T0A_SOURCE_AUTH_SHA256,
            "authenticated T0a anchor differs")
    aggregate = load_json(args.t0a_aggregate)
    require(aggregate.get("stage") == "M33_T0A_CROSS_PROCESS_COMPARISON" and
            aggregate.get("status") == "PASS_T0A_CROSS_PROCESS_TECHNICAL_ONLY" and
            aggregate.get("t0b_open") is False and aggregate.get("scientific_evidence") is False,
            "T0a aggregate semantic identity differs")
    require(len(args.t0a_child_receipt) == EXPECTED_CHILDREN,
            "T0a local child receipt count differs")
    expected_inventory = aggregate.get("child_receipts", [])
    require(len(expected_inventory) == EXPECTED_CHILDREN,
            "T0a aggregate child inventory differs")
    expected_hashes = {row["name"]: row["sha256"] for row in expected_inventory}
    require(len(expected_hashes) == EXPECTED_CHILDREN, "T0a child names are not unique")
    observed_hashes = {path.name: sha256_file(path) for path in args.t0a_child_receipt}
    require(observed_hashes == expected_hashes, "T0a local child receipt identity differs")
    children = [load_json(path) for path in args.t0a_child_receipt]
    require(all(child.get("implementation_commit") == aggregate.get("implementation_commit") and
                child.get("source_auth_sha256") == EXPECTED_T0A_SOURCE_AUTH_SHA256 and
                child.get("oci_image") == aggregate.get("oci_image")
                for child in children), "T0a child execution identity differs")
    forbidden = ("truth_read", "training", "gradients", "optimizer",
                 "predictions_persisted", "consumable")
    require(all(all(child.get(field) is False for field in forbidden) for child in children),
            "T0a child firewall differs")
    expected_inputs = contract.get("expected_inputs", {})
    require(set(expected_inputs) == {"root17", "root18"},
            "T0b expected input roots differ")
    identities = {
        "root17": authenticate_root_inputs(
            "root17", args.root17_technical_dir, args.root17_verify,
            args.root17_map, expected_inputs["root17"]),
        "root18": authenticate_root_inputs(
            "root18", args.root18_technical_dir, args.root18_verify,
            args.root18_map, expected_inputs["root18"]),
    }
    counts = {root: identity["marker_count"] for root, identity in identities.items()}
    require(set(counts.values()) == {EXPECTED_MARKERS},
            "full chromosome 22 marker count differs")
    available = mem_available_gib(args.meminfo)
    require(available >= MINIMUM_MEM_AVAILABLE_GIB,
            f"MemAvailable below T0b three-process floor: {available:.3f} GiB")
    return {
        "stage": "M33_T0B_PREFLIGHT",
        "status": "PASS_T0B_PREFLIGHT_THREE_WAY_ONLY",
        "t0a_aggregate_sha256": EXPECTED_AGGREGATE_SHA256,
        "t0a_source_auth_sha256": EXPECTED_T0A_SOURCE_AUTH_SHA256,
        "t0a_child_receipt_count": len(children),
        "t0a_child_inventory_sha256": hashlib.sha256(json.dumps(
            expected_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "marker_count_by_root": counts,
        "input_identity_by_root": identities,
        "mem_available_gib": available,
        "minimum_mem_available_gib": MINIMUM_MEM_AVAILABLE_GIB,
        "maximum_parallel_forward_processes": MAXIMUM_PARALLEL_FORWARD_PROCESSES,
        "contract_sha256": sha256_file(args.contract),
        "implementation_commit": args.implementation_commit,
        "source_auth_sha256": source_auth_sha,
        "truth_read": False, "training": False, "gradients": False,
        "optimizer": False, "predictions_persisted": False,
        "scientific_evidence": False, "consumable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--t0a-aggregate", type=Path, required=True)
    parser.add_argument("--t0a-source-auth", type=Path, required=True)
    parser.add_argument("--t0a-child-receipt", action="append", type=Path, required=True)
    parser.add_argument("--root17-technical-dir", type=Path, required=True)
    parser.add_argument("--root18-technical-dir", type=Path, required=True)
    parser.add_argument("--root17-verify", type=Path, required=True)
    parser.add_argument("--root18-verify", type=Path, required=True)
    parser.add_argument("--root17-map", type=Path, required=True)
    parser.add_argument("--root18-map", type=Path, required=True)
    parser.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run_preflight(args), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
