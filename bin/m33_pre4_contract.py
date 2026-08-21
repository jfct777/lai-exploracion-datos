#!/usr/bin/env python3
"""Fail-closed validator for the immutable M33 PRE-4 contract.

This program can only issue a contract receipt.  It has no model, truth reader,
asset generator, training entry point, or execution token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


EXPERIMENT = "M33_PRE4_ORDERED_RARE_LAI"
STATUS = "CONTRACT_ONLY_NO_ASSETS_NO_FORWARD_NO_TRAINING"
EXACT_CONTRACT_SHA256 = "4308bbf33ae28f554f701da33efdc185264f9f407d62661e7048e0345687eb8b"
EXPECTED_CONTAINER_DIGEST = "sha256:2c30d018028636ac1b7a4890641e04b3e15be8c79d991dfade35b90db0e17bd1"
EXPECTED_NEXTFLOW_VERSION = "26.04.6"
REQUIRED_SOURCES = {
    "bin/m33_pre4_contract.py",
    "bin/m33_pre4_source_auth.py",
    "conf/m33_pre4_preregistration.json",
    "conf/m33_pre4_contract.config",
    "modules/33_PRE4_CONTRACT.nf",
    "workflows/m33_pre4_contract.nf",
    "tests/test_m33_pre4_contract.py",
    "tests/test_m33_pre4_nextflow.py",
}
ROLES = {"DEVELOPMENT": 3, "EVAL_reserved_not_generated": 5}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def derived_seed(role: str, index: int) -> int:
    digest = hashlib.sha256(f"DNABR_M33_PRE4|{role}|{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    loaded = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_pairs,
        parse_constant=reject_constant,
    )
    require(isinstance(loaded, dict), "top-level JSON must be an object")
    return loaded


def load_contract(path: str | Path) -> dict[str, Any]:
    require(sha256_file(path) == EXACT_CONTRACT_SHA256, "immutable contract byte hash drift")
    payload = strict_json(path)
    validate_contract(payload)
    return payload


def validate_contract(contract: dict[str, Any]) -> None:
    """Check critical invariants in addition to the exact immutable byte hash."""
    require(contract["schema_version"] == "2.0.0", "schema drift")
    require(contract["experiment_id"] == EXPERIMENT, "experiment drift")
    require(contract["status"] == STATUS, "contract-only status drift")
    require(contract["execution_authorization"] == {
        "contract_tests": True,
        "prospective_asset_generation": False,
        "no_gradient_forward": False,
        "training": False,
        "DEVELOPMENT_truth_outside_frozen_rotations": False,
        "EVAL_truth_opening": False,
    }, "execution was authorized prematurely")

    roots = contract["root_registry"]
    observed: list[int] = []
    for role, count in ROLES.items():
        seed_role = "EVAL" if role.startswith("EVAL_") else role
        expected = [derived_seed(seed_role, index) for index in range(count)]
        require(roots[role] == expected, f"{role} seed drift")
        observed.extend(expected)
    require(len(observed) == len(set(observed)), "prospective roots overlap")
    require(set(observed).isdisjoint(roots["consumed_technical_only"]), "consumed root reused")
    expected_rotations = [
        {"rotation": "R0", "fit_roots": [observed[1], observed[2]], "score_only_root": observed[0]},
        {"rotation": "R1", "fit_roots": [observed[0], observed[2]], "score_only_root": observed[1]},
        {"rotation": "R2", "fit_roots": [observed[0], observed[1]], "score_only_root": observed[2]},
    ]
    require(roots["development_rotations"] == expected_rotations, "development rotation drift")
    require(roots["independent_genealogy_required_not_assumed"] is True, "independence was assumed")
    require(contract["asset_manifest_gate"]["status"] == "BLOCKED_PENDING_MANIFESTS",
            "assets opened without verified manifests")
    require(contract["simulation_contract"]["ASIA_is_not_NAM"] is True, "ASIA mislabeled as NAM")
    require(contract["primary_transferable_input"]["rare_haplotype_phase_used"] is False,
            "transferable arm uses unavailable rare phase")
    require(contract["phase_ceiling"]["screen_or_promotion_eligible"] is False,
            "simulation-only phase ceiling entered selection")
    require(contract["model_screen"]["candidate_grid"]["total_candidates"] == 16,
            "candidate budget drift")
    require(contract["metrics"]["primary"] ==
            "paired_boundary_F1_at_0.2_cM_RE_minus_RD_per_simulation_root",
            "primary endpoint drift")
    require(contract["metrics"]["primary_inference_unit"] == "simulation_root",
            "pseudoreplicated inference unit")
    require(contract["promotion_rule"]["two_sided_significance_claim"] is False,
            "unsupported two-sided claim enabled")
    require(contract["technical_gates"]["T0_inference_only"]["status"].startswith("BLOCKED_"),
            "T0 opened prematurely")
    require(contract["technical_gates"]["T1_backward_dry_run"]["status"] == "BLOCKED_UNTIL_T0_POST",
            "T1 opened prematurely")


def validate_source_auth(auth_path: Path, git_commit: str,
                         staged_sources: dict[str, Path],
                         repository_root: Path | None = None) -> dict[str, str]:
    require(re.fullmatch(r"[0-9a-f]{40}", git_commit) is not None, "git commit must be exact")
    auth = strict_json(auth_path)
    require(set(auth) == {"stage", "status", "git_commit", "source_sha256"},
            "source-auth key inventory drift")
    require(auth["stage"] == "M33_PRE4_SOURCE_AUTH", "source-auth stage drift")
    require(auth["status"] == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES", "source-auth did not pass")
    require(auth["git_commit"] == git_commit, "source-auth commit drift")
    hashes = auth["source_sha256"]
    require(isinstance(hashes, dict) and set(hashes) == REQUIRED_SOURCES,
            "source-auth inventory incomplete")
    require(set(staged_sources) == REQUIRED_SOURCES, "staged source inventory incomplete")
    for relative in sorted(REQUIRED_SOURCES):
        expected = hashes[relative]
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                f"invalid source hash: {relative}")
        require(sha256_file(staged_sources[relative]) == expected,
                f"staged source changed after authentication: {relative}")
    require(hashes["conf/m33_pre4_preregistration.json"] == EXACT_CONTRACT_SHA256,
            "authenticated contract hash drift")
    require(hashes["bin/m33_pre4_contract.py"] == sha256_file(Path(__file__)),
            "authenticated validator hash drift")
    if repository_root is not None:
        head = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        require(head == git_commit, "Git HEAD differs from authenticated commit")
        dirty = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain", "--", *sorted(REQUIRED_SOURCES)],
            check=True, capture_output=True, text=True,
        ).stdout
        require(not dirty.strip(), "authenticated sources are dirty or untracked")
        for relative in sorted(REQUIRED_SOURCES):
            committed = subprocess.run(
                ["git", "-C", str(repository_root), "show", f"{git_commit}:{relative}"],
                check=True, capture_output=True,
            ).stdout
            require(hashlib.sha256(committed).hexdigest() == hashes[relative],
                    f"commit does not contain authenticated source: {relative}")
    return hashes


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_staged(items: list[str]) -> dict[str, Path]:
    staged: dict[str, Path] = {}
    for item in items:
        relative, separator, path = item.partition("=")
        require(bool(separator) and relative and path, "invalid staged-source specification")
        require(relative not in staged and relative in REQUIRED_SOURCES, "duplicate or unknown staged source")
        staged[relative] = Path(path)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--source-auth", type=Path)
    parser.add_argument("--staged-source", action="append", default=[])
    parser.add_argument("--git-commit")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--container-digest")
    parser.add_argument("--nextflow-version")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    receipt: dict[str, Any] = {
        "experiment_id": contract["experiment_id"],
        "status": "PASS_CONTRACT_ONLY",
        "contract_sha256": sha256_file(args.contract),
        "execution_authorization": contract["execution_authorization"],
        "root_registry": contract["root_registry"],
    }
    optional = (args.source_auth, args.git_commit, args.repository_root, args.container_digest,
                args.nextflow_version, args.output)
    if any(value is not None for value in optional) or args.staged_source:
        require(all(value is not None for value in optional),
                "source-auth, commit, container, Nextflow and output are jointly required")
        staged = parse_staged(args.staged_source)
        hashes = validate_source_auth(args.source_auth, args.git_commit, staged, args.repository_root)
        require(args.container_digest == EXPECTED_CONTAINER_DIGEST, "planned container digest drift")
        require(args.nextflow_version == EXPECTED_NEXTFLOW_VERSION, "Nextflow version drift")
        receipt.update({
            "git_commit": args.git_commit,
            "source_sha256": hashes,
            "source_auth_sha256": sha256_file(args.source_auth),
            "planned_model_container_digest": args.container_digest,
            "nextflow_version": args.nextflow_version,
        })
        write_exclusive(args.output, receipt)
    else:
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
