#!/usr/bin/env python3
"""Validate and seal the factorized-lazy M33 DEVELOPMENT representation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import m33_materialize as materialize
from m33_safe_bridge_core import reopen_npz, write_deterministic_npz


ROOTS = (386357765, 2024931463, 1324432253)
TARGET_SHAMS = (1277457345, 943666774, 1858042568)
REF_SHAMS = (79351217, 202307732, 1737132171)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_root(root_dir: Path) -> dict[str, Any]:
    selected = materialize.load_productive_npz(root_dir / "selected_loci_incremental.npz", "selected")
    target = materialize.load_productive_npz(root_dir / "target_rare_diploid_incremental.npz", "target")
    reference = materialize.load_productive_npz(root_dir / "reference_rare_summary_incremental.npz", "reference")
    f0 = materialize.load_productive_npz(root_dir / "flare_f0_sanitized.npz", "f0")
    with np.load(root_dir / "marker_cM.npz", allow_pickle=False) as archive:
        require(set(archive.files) == {"marker_cM"}, "marker-cM members differ")
        marker_cm = np.ascontiguousarray(archive["marker_cM"])
    materialize.validate_inputs(selected, target, reference, f0, marker_cm)
    target_shams = {}
    for seed in TARGET_SHAMS:
        value = materialize.load_productive_npz(root_dir / f"target_same_locus_sham_{seed}.npz", "target")
        materialize.validate_inputs(selected, value, reference, f0, marker_cm)
        target_shams[seed] = value
    ref_shams = {}
    for seed in REF_SHAMS:
        value = materialize.load_productive_npz(root_dir / f"reference_label_sham_{seed}.npz", "reference")
        materialize.validate_inputs(selected, target, value, f0, marker_cm)
        ref_shams[seed] = value
    return {
        "selected": selected, "target": target, "reference": reference, "f0": f0,
        "marker_cm": marker_cm, "target_shams": target_shams, "ref_shams": ref_shams,
    }


def plan_hash(intervals: dict[str, np.ndarray], rare_cm: np.ndarray,
              marker_cm: np.ndarray) -> tuple[str, dict[str, int]]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    sample_count = 30
    for radius in materialize.RADII:
        total = 0
        for start in range(0, sample_count, materialize.PERSON_BATCH):
            people = min(materialize.PERSON_BATCH, sample_count - start)
            rows = materialize.plan_lazy_marker_chunks(intervals, rare_cm, marker_cm, radius, people)
            for row in rows:
                digest.update(json.dumps({
                    "radius_cM": radius, "person_start": start, **row,
                }, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            total += len(rows)
        counts[str(radius)] = total
    return digest.hexdigest(), counts


def run(args: argparse.Namespace) -> dict:
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    require(tuple(pre4["root_registry"]["DEVELOPMENT"]) == ROOTS, "DEVELOPMENT roots drifted")
    require(not args.outdir.exists(), f"refusing to overwrite {args.outdir}")
    roots = {seed: load_root(args.bridge_root / f"root-{seed}") for seed in ROOTS}
    args.outdir.mkdir(parents=True, exist_ok=False)
    root_receipts: dict[int, dict[str, Any]] = {}
    for seed, data in roots.items():
        root_out = args.outdir / f"root-{seed}"
        root_out.mkdir()
        intervals = materialize.build_interval_table(data["selected"]["cM"], data["marker_cm"])
        materialize.validate_interval_table(
            intervals, data["selected"]["cM"], data["marker_cm"]
        )
        interval_path = root_out / "context_intervals_all_radii.npz"
        write_deterministic_npz(interval_path, intervals)
        reopen_npz(interval_path, intervals)
        microbatch_sha, shard_counts = plan_hash(
            intervals, data["selected"]["cM"], data["marker_cm"]
        )
        source = args.bridge_root / f"root-{seed}"
        members = [
            "selected_loci_incremental.npz", "target_rare_diploid_incremental.npz",
            "reference_rare_summary_incremental.npz", "flare_f0_sanitized.npz", "marker_cM.npz",
            *[f"target_same_locus_sham_{value}.npz" for value in TARGET_SHAMS],
            *[f"reference_label_sham_{value}.npz" for value in REF_SHAMS],
        ]
        receipt = {
            "schema_version": "1.0.0", "stage": "M33_FACTORIZED_ROOT",
            "status": "PASS_FACTORIZED_NO_EXPANDED_CONTEXTS", "root_seed": seed,
            "pre4_sha256": sha256_file(args.pre4),
            "factor_sha256": {name: sha256_file(source / name) for name in members},
            "context_intervals_sha256": sha256_file(interval_path),
            "microbatch_plan_semantic_sha256": microbatch_sha,
            "shard_count_by_radius": shard_counts,
            "radii_cM": list(materialize.RADII), "person_batch_maximum": materialize.PERSON_BATCH,
            "valid_token_budget": materialize.TOKEN_BUDGET,
            "persistent_expanded_contexts": False, "truth_accessed": False,
        }
        (root_out / "factorized_root.receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        root_receipts[seed] = receipt

    rotation_receipts: list[dict[str, Any]] = []
    forbidden_eval = pre4["root_registry"]["EVAL_reserved_not_generated"]
    for rotation, spec in materialize.DEVELOPMENT_ROTATIONS.items():
        fit = spec["fit_root_seeds"]
        maxima = materialize.derive_fit_max_callable(
            {seed: roots[seed]["reference"] for seed in fit}, rotation,
            {seed: canonical_sha256(root_receipts[seed]) for seed in fit}, forbidden_eval,
        )
        norm = {
            "schema_version": "1.0.0", "stage": "M33_FIT_NORMALIZATION",
            "status": "PASS_FIT_ONLY", "rotation_id": rotation,
            "fit_root_seeds": fit, "score_only_root_seed": spec["score_only_root_seed"],
            "max_callable_an": maxima,
            "source_receipt_sha256": {str(seed): canonical_sha256(root_receipts[seed]) for seed in fit},
            "score_or_eval_contributed": False,
        }
        rotation_dir = args.outdir / rotation
        rotation_dir.mkdir()
        norm_path = rotation_dir / "fit_callable_normalization_manifest.json"
        norm_path.write_text(json.dumps(norm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for seed in ROOTS:
            role = "FIT" if seed in fit else "SCORE_ONLY"
            recipe = {
                "schema_version": "1.0.0", "stage": "M33_LAZY_CONTEXT_RECIPE",
                "status": "PASS", "root_seed": seed, "rotation_id": rotation,
                "role_in_rotation": role, "radii_cM": list(materialize.RADII),
                "person_batch_maximum": materialize.PERSON_BATCH,
                "valid_token_budget_per_batch": materialize.TOKEN_BUDGET,
                "central_marker_block": 256, "row_order": "sample_major_then_marker",
                "token_order": "cM_then_bp_then_locus_id", "channel_count": 13,
                "optimizer_update_rule": "one_step_after_all_microbatches_of_logical_block",
                "microbatch_plan_semantic_sha256": root_receipts[seed]["microbatch_plan_semantic_sha256"],
                "factorized_root_receipt_sha256": canonical_sha256(root_receipts[seed]),
                "fit_callable_normalization_manifest_sha256": sha256_file(norm_path),
            }
            recipe_path = rotation_dir / f"root-{seed}.lazy_recipe.json"
            recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            ready = {
                "schema_version": "1.0.0", "stage": "M33_FACTORIZED_READY", "status": "READY",
                "root_seed": seed, "rotation_id": rotation, "role_in_rotation": role,
                "factorized_root_receipt_sha256": canonical_sha256(root_receipts[seed]),
                "fit_callable_normalization_manifest_sha256": sha256_file(norm_path),
                "lazy_context_recipe_sha256": sha256_file(recipe_path),
            }
            (rotation_dir / f"root-{seed}.READY.json").write_text(
                json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            rotation_receipts.append(ready)
    aggregate = {
        "schema_version": "1.0.0", "stage": "M33_MATERIALIZE_FACTORIZED",
        "status": "PASS_ALL_DEVELOPMENT_ROOTS_ROTATIONS", "roots": list(ROOTS),
        "rotations": list(materialize.DEVELOPMENT_ROTATIONS),
        "root_receipt_sha256": {str(seed): canonical_sha256(root_receipts[seed]) for seed in ROOTS},
        "ready_count": len(rotation_receipts), "persistent_expanded_contexts": False,
        "truth_accessed": False,
    }
    (args.outdir / "materialize.aggregate.receipt.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "ready_count": result["ready_count"]}, sort_keys=True))
