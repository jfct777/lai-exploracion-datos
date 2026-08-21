#!/usr/bin/env python3
"""Deterministic synthetic known-answer smoke for the M32 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from m32_locus_contract import EXPECTED_RADII, load_contract, sha256_file, validate_git_commit
from m32_locus_occupancy import occupancy_report
from m32_locus_tensor import (
    apply_phase_switches,
    array_sha256,
    build_ordered_sequence,
    pad_ragged_indices,
    permute_reference_labels,
    phase_aware_minor_presence,
    primary_diploid_channels,
    ragged_context_indices,
    reference_support,
)


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_source_specs(values: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("source must be relative_repository_path=staged_path")
        relative, staged = value.split("=", 1)
        if not relative or relative.startswith("/") or ".." in Path(relative).parts or relative in sources:
            raise ValueError("source repository path is unsafe or duplicated")
        sources[relative] = Path(staged)
    required = {
        "bin/m32_locus_contract.py",
        "bin/m32_locus_tensor.py",
        "bin/m32_locus_occupancy.py",
        "bin/m32_locus_smoke.py",
        "conf/m32_locus_sequence_smoke_preregistration.json",
        "conf/m32_locus_sequence_smoke.config",
        "modules/32_LOCUS_SEQUENCE_SMOKE.nf",
        "workflows/m32_locus_sequence_smoke.nf",
    }
    if set(sources) != required:
        raise ValueError("source set does not authenticate the complete M32 implementation")
    return sources


def authenticate_sources(repository_root: Path, git_commit: str, sources: dict[str, Path]) -> dict[str, str]:
    if not (repository_root / ".git").exists():
        raise ValueError("repository_root is not a Git worktree")
    head = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != git_commit:
        raise ValueError(f"git_commit does not equal repository HEAD: {git_commit} != {head}")
    status = subprocess.run(
        ["git", "-C", str(repository_root), "status", "--porcelain", "--", *sorted(sources)],
        check=True, capture_output=True, text=True,
    ).stdout
    if status.strip():
        raise ValueError("authenticated M32 sources are dirty or untracked")

    hashes: dict[str, str] = {}
    for relative, staged in sorted(sources.items()):
        if not staged.is_file():
            raise ValueError(f"missing staged source: {staged}")
        committed = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"{git_commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        staged_hash = sha256_file(staged)
        if staged_hash != sha256_bytes(committed):
            raise ValueError(f"staged source differs from {git_commit}:{relative}")
        hashes[relative] = staged_hash
    return hashes


def missing_mask(values: Sequence[Sequence[Sequence[int]]]) -> list[list[list[bool]]]:
    return [[[state == -1 for state in pair] for pair in person] for person in values]


def transition_matrix(original: Sequence[str], permuted: Sequence[str]) -> dict[str, dict[str, int]]:
    labels = ("AFR", "EUR", "ASIA")
    return {
        source: {target: sum(a == source and b == target for a, b in zip(original, permuted)) for target in labels}
        for source in labels
    }


def run_known_answers(seed: int) -> tuple[dict, dict]:
    minor_codes = [0, 1, 1, 0]
    target = [
        [[0, 0], [0, 1], [1, 1], [-1, 0]],
        [[1, 0], [1, 1], [0, 0], [1, 1]],
    ]
    primary = primary_diploid_channels(target, minor_codes)
    switches = [[False, True, False, True], [True, False, True, False]]
    switched = apply_phase_switches(target, switches)
    switched_primary = primary_diploid_channels(switched, minor_codes)
    phase = phase_aware_minor_presence(target, minor_codes)
    switched_phase = phase_aware_minor_presence(switched, minor_codes)
    phase_equivariant = all(
        switched_phase[person][locus] == (list(reversed(phase[person][locus])) if switches[person][locus] else phase[person][locus])
        for person in range(len(target)) for locus in range(len(minor_codes))
    )

    reference = [
        [[0, 0], [1, 1], [0, 1], [0, 1]],
        [[0, 1], [1, 0], [1, 1], [0, 0]],
        [[1, 1], [0, 0], [1, 0], [1, 1]],
        [[0, 0], [-1, 1], [0, 0], [0, 1]],
        [[1, 0], [1, 1], [1, 1], [1, 0]],
        [[1, 1], [0, 1], [-1, -1], [0, 0]],
    ]
    labels = ["AFR", "AFR", "EUR", "EUR", "ASIA", "ASIA"]
    reference_hash_before = array_sha256(reference)
    reference_missing_before = array_sha256(missing_mask(reference))
    target_hash_before = array_sha256(target)
    permuted = permute_reference_labels(labels, seed)
    support, observed = reference_support(reference, minor_codes, labels)
    sham_support, sham_observed = reference_support(reference, minor_codes, permuted)
    sham_audit = []
    for replicate in range(8):
        sham_labels = permute_reference_labels(labels, seed + replicate)
        sham_audit.append({
            "seed": seed + replicate,
            "label_sha256": array_sha256(sham_labels),
            "moved_fraction": sum(a != b for a, b in zip(labels, sham_labels)) / len(labels),
            "transition_matrix": transition_matrix(labels, sham_labels),
        })

    marker_ids = [f"m{index}" for index in range(6)]
    marker_bp = [100, 200, 300, 400, 500, 600]
    marker_cm = [0.00, 0.07, 0.14, 0.31, 0.55, 0.90]
    locus_ids = [f"r{index}" for index in range(4)]
    locus_bp = [120, 220, 330, 340]
    locus_cm = [0.02, 0.09, 0.30, 0.30]
    flare = [
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
        [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.7, 0.2, 0.1], [0.8, 0.1, 0.1], [0.2, 0.1, 0.7]],
    ]
    sequence = build_ordered_sequence(
        marker_ids, marker_bp, marker_cm, flare,
        locus_ids, locus_bp, locus_cm, minor_codes,
        target, reference, labels,
    )
    occupancy = occupancy_report(marker_cm, locus_cm, EXPECTED_RADII)
    ragged_roundtrip = True
    ragged_widths: dict[str, int] = {}
    for radius in EXPECTED_RADII:
        rows = ragged_context_indices(marker_cm, locus_cm, radius)
        padded, mask = pad_ragged_indices(rows)
        recovered = [[index for index, keep in zip(row, row_mask) if keep] for row, row_mask in zip(padded, mask)]
        ragged_roundtrip &= recovered == rows
        ragged_widths[str(radius)] = max((len(row) for row in padded), default=0)

    grid_hash_before = array_sha256({"marker_id": marker_ids, "bp": marker_bp, "cm": marker_cm})
    grid_hash_after = array_sha256({key: sequence["grid"][key] for key in ("marker_id", "bp", "cm")})
    locus_hash_before = array_sha256({"locus_id": locus_ids, "bp": locus_bp, "cm": locus_cm})
    locus_hash_after = array_sha256({key: sequence["rare_sequence"][key] for key in ("locus_id", "bp", "cm")})
    invariants = {
        "dosage_0_1_2_known_answer": primary["minor_dosage"] == [[2, 1, 2, None], [1, 2, 0, 0]],
        "missing_is_explicit_null_not_zero": primary["minor_dosage"][0][3] is None and not primary["callable_mask"][0][3],
        "primary_invariant_to_arbitrary_phase_switches": primary == switched_primary,
        "H_is_equivariant_to_arbitrary_phase_switches": phase_equivariant and phase != switched_phase,
        "reference_genotype_matrix_unchanged_by_sham": reference_hash_before == array_sha256(reference),
        "reference_missingness_unchanged_by_sham": reference_missing_before == array_sha256(missing_mask(reference)),
        "reference_LD_and_locus_order_unchanged_by_sham": reference_hash_before == array_sha256(reference),
        "target_unchanged_by_sham": target_hash_before == array_sha256(target),
        "reference_ancestry_group_sizes_unchanged_by_sham": sorted(labels) == sorted(permuted),
        "all_shams_move_at_least_one_label": all(row["moved_fraction"] > 0 for row in sham_audit),
        "sham_changes_genotype_label_association": support != sham_support or observed != sham_observed,
        "full_flare_grid_identity_and_order_preserved": grid_hash_before == grid_hash_after,
        "rare_locus_identity_and_order_preserved": locus_hash_before == locus_hash_after,
        "ragged_padding_mask_roundtrip": ragged_roundtrip,
        "cm_ties_preserved_with_unique_bp": sequence["rare_sequence"]["cm"][-2:] == [0.30, 0.30] and sequence["rare_sequence"]["bp"][-2:] == [330, 340],
    }
    if not all(invariants.values()):
        failed = [name for name, passed in invariants.items() if not passed]
        raise AssertionError(f"M32 known-answer invariants failed: {failed}")
    metrics = {
        "seed": seed,
        "invariants": invariants,
        "array_sha256": {
            "target_haplotypes": target_hash_before,
            "primary_dosage": array_sha256(primary["minor_dosage"]),
            "reference_haplotypes": reference_hash_before,
            "original_labels": array_sha256(labels),
            "permuted_labels": array_sha256(permuted),
            "full_grid": grid_hash_after,
            "rare_loci": locus_hash_after,
        },
        "sham_audit": sham_audit,
        "occupancy": occupancy,
        "ragged_padded_width_by_radius": ragged_widths,
    }
    fixture = {"minor_codes": minor_codes, "marker_bp": marker_bp, "marker_cm": marker_cm, "locus_bp": locus_bp, "locus_cm": locus_cm}
    return metrics, fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=320017)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    git_commit = validate_git_commit(args.git_commit)
    contract = load_contract(args.preregistration)
    sources = parse_source_specs(args.source)
    source_hashes = authenticate_sources(args.repository_root.resolve(), git_commit, sources)
    run_dir = args.outdir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metrics, fixture = run_known_answers(args.seed)
    metrics_path = run_dir / "m32_locus_sequence.occupancy_and_invariants.json"
    provenance_path = run_dir / "m32_locus_sequence.provenance.json"
    manifest_path = run_dir / "m32_locus_sequence.manifest.json"
    receipt_path = run_dir / "m32_locus_sequence.receipt.json"
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(provenance_path, {
        "stage": contract["stage"],
        "status": contract["status"],
        "run_id": args.run_id,
        "seed": args.seed,
        "git_commit": git_commit,
        "source_sha256": source_hashes,
        "preregistration_sha256": sha256_file(args.preregistration),
        "python": platform.python_version(),
        "fixture": fixture,
        "scientific_evidence": False,
    })
    manifest = {
        "files": {
            metrics_path.name: sha256_file(metrics_path),
            provenance_path.name: sha256_file(provenance_path),
        },
        "sources": source_hashes,
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(receipt_path, {
        "stage": contract["stage"],
        "run_id": args.run_id,
        "git_commit": git_commit,
        "status": "PASS_SYNTHETIC_SMOKE_ONLY",
        "all_invariants_passed": True,
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_sha256": sha256_file(provenance_path),
        "authenticated_source_count": len(source_hashes),
        "scientific_run_authorized": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
