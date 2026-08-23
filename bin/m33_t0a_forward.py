#!/usr/bin/env python3
"""Run one fresh-process, truth-free M33 T0a forward on a technical root."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import m33_materialize as materialize
import m33_t0a_models as models
from m33_m0_factorized_lazy_technical_kat import (
    array_sha256, load_technical, stratified_markers, subset_f0,
)


ROOTS = {"root17": 20260817, "root18": 20260818}
MARKER_COUNT = 512
MAX_PADDED_TOKENS = 262_144


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_limit_bytes() -> int:
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.is_file():
            raw = path.read_text(encoding="ascii").strip()
            if raw != "max" and raw.isdigit() and int(raw) < 1 << 60:
                return int(raw)
    rows = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    return int(next(row for row in rows if row.startswith("MemTotal:")).split()[1]) * 1024


def rss_fraction(limit_bytes: int) -> float:
    fraction = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / limit_bytes
    require(fraction < 0.8, f"T0a peak RSS reached stop fraction: {fraction:.6f}")
    return fraction


def validate_source_auth(path: Path, commit: str, source_root: Path,
                         extra_sources: tuple[str, ...] = ()) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("stage") == "M33_T0A_SOURCE_AUTH" and
            payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            payload.get("git_commit") == commit, "T0a source authentication differs")
    expected = payload.get("source_sha256", {})
    runtime = (
        "bin/m31_ordered_linear.py", "bin/m33_m0_contract.py",
        "bin/m33_materialize.py", "bin/m33_m0_factorized_lazy_technical_kat.py",
        "bin/m33_t0a_models.py", "bin/m33_t0a_forward.py",
        "conf/m33_pre4_preregistration.json",
    ) + extra_sources
    for relative in runtime:
        require(expected.get(relative) == sha256_file(source_root / relative),
                f"authenticated T0a source differs: {relative}")
    return sha256_file(path)


def validate_pre4(path: Path) -> None:
    contract = json.loads(path.read_text(encoding="utf-8"))
    screen = contract.get("model_screen", {})
    gates = contract.get("technical_gates", {}).get("T0_inference_only", {})
    require(screen.get("families") == ["local_linear", "small_residual_cnn_1d"] and
            "Linear29x3_90_parameters" in screen.get("local_linear", "") and
            "1651_parameters" in screen.get("small_residual_cnn_1d", ""),
            "PRE-4 model contract differs")
    require(gates.get("stress_people") == [30, 256, 1024, 2619] and
            gates.get("stress_loci") == 512 and
            set(gates.get("checks", [])) >= {
                "no_truth_interface", "torch_inference_mode", "finite_simplex",
                "zero_residual_reproduces_F0_max_abs_le_1e-6",
                "joint_haplotype_swap_equivariance_max_abs_le_1e-6",
                "person_permutation_equivariance_max_abs_le_1e-6",
                "padding_batch_chunk_invariance_max_abs_le_1e-6",
                "parameter_and_shape_hashes", "peak_RSS_le_0.7_host_RAM",
                "private_tensors_not_published",
            }, "PRE-4 T0 gate differs")


def padded_batches(row_ptr: np.ndarray, maximum_rows: int = 256) -> list[tuple[int, int]]:
    lengths = np.diff(row_ptr.astype("<i8"))
    require(np.all(lengths >= 0), "ragged row pointer differs")
    batches: list[tuple[int, int]] = []
    start = 0
    while start < lengths.size:
        end, maximum = start, 0
        while end < lengths.size and end - start < maximum_rows:
            candidate = max(maximum, int(lengths[end]))
            if candidate > MAX_PADDED_TOKENS:
                raise ValueError("single padded context exceeds token budget")
            if end > start and candidate * (end - start + 1) > MAX_PADDED_TOKENS:
                break
            maximum, end = candidate, end + 1
        require(end > start, "empty padded batch")
        batches.append((start, end))
        start = end
    return batches


def dense_rows(packed: dict[str, np.ndarray], start: int, end: int):
    row_ptr = packed["row_ptr"].astype("<i8")
    lengths = np.diff(row_ptr)[start:end]
    width = max(1, int(lengths.max(initial=0)))
    tokens = np.zeros((end - start, width, 13), dtype="<f4")
    mask = np.zeros((end - start, width), dtype="<f4")
    for local, row in enumerate(range(start, end)):
        left, right = int(row_ptr[row]), int(row_ptr[row + 1])
        length = right - left
        if length:
            tokens[local, :length] = packed["rare_tokens"][left:right]
            mask[local, :length] = 1.0
    sample_count, _haplotypes, marker_count, _ancestries = packed["F0"].shape
    expected_samples = np.repeat(np.arange(sample_count, dtype="<u4"), marker_count)
    expected_markers = np.tile(np.arange(marker_count, dtype="<u4"), sample_count)
    require(np.array_equal(packed["row_sample_index"], expected_samples) and
            np.array_equal(packed["row_marker_index"], expected_markers),
            "packed row index does not match flattened F0")
    f0_rows = np.transpose(packed["F0"], (0, 2, 1, 3)).reshape(-1, 2, 3)[start:end]
    return (torch.from_numpy(tokens), torch.from_numpy(mask),
            torch.from_numpy(np.ascontiguousarray(f0_rows, dtype="<f4")))


def max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.max(torch.abs(first - second)).item()) if first.numel() else 0.0


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a CPU tensor together with its exact dtype and shape."""
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def invariant_checks(model: torch.nn.Module, tokens: torch.Tensor, mask: torch.Tensor,
                     f0: torch.Tensor) -> dict[str, float]:
    limit = min(tokens.shape[0], 12)
    tokens, mask, f0 = tokens[:limit], mask[:limit], f0[:limit]
    with torch.inference_mode():
        base, base_delta, features = model.forward_with_features(tokens, mask, f0)
        padded_tokens = torch.cat((tokens, torch.zeros(tokens.shape[0], 7, 13)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(mask.shape[0], 7)), dim=1)
        padded, padded_delta, padded_features = model.forward_with_features(
            padded_tokens, padded_mask, f0)
        order = torch.arange(tokens.shape[0] - 1, -1, -1)
        permuted, permuted_delta, permuted_features = model.forward_with_features(
            tokens[order], mask[order], f0[order])
        restored = torch.empty_like(permuted); restored[order] = permuted
        restored_delta = torch.empty_like(permuted_delta); restored_delta[order] = permuted_delta
        restored_features = torch.empty_like(permuted_features)
        restored_features[order] = permuted_features
        midpoint = max(1, tokens.shape[0] // 2)
        pieces = [model.forward_with_features(tokens[:midpoint], mask[:midpoint], f0[:midpoint]),
                  model.forward_with_features(tokens[midpoint:], mask[midpoint:], f0[midpoint:])]
        chunked = torch.cat([piece[0] for piece in pieces if piece[0].shape[0]], dim=0)
        chunked_delta = torch.cat([piece[1] for piece in pieces if piece[1].shape[0]], dim=0)
        chunked_features = torch.cat(
            [piece[2] for piece in pieces if piece[2].shape[0]], dim=0)
        swapped, swapped_delta, swapped_features = model.forward_with_features(
            tokens, mask, f0.flip(1))
        swapped = swapped.flip(1)
        swapped_delta = swapped_delta.flip(1)
        swapped_features = swapped_features.flip(1)
    checks = {
        "padding_output_max_abs": max_abs(base, padded),
        "padding_delta_max_abs": max_abs(base_delta, padded_delta),
        "padding_feature_max_abs": max_abs(features, padded_features),
        "person_permutation_output_max_abs": max_abs(base, restored),
        "person_permutation_delta_max_abs": max_abs(base_delta, restored_delta),
        "person_permutation_feature_max_abs": max_abs(features, restored_features),
        "batch_chunk_output_max_abs": max_abs(base, chunked),
        "batch_chunk_delta_max_abs": max_abs(base_delta, chunked_delta),
        "batch_chunk_feature_max_abs": max_abs(features, chunked_features),
        "joint_haplotype_swap_max_abs": max_abs(base, swapped),
        "joint_haplotype_swap_delta_max_abs": max_abs(base_delta, swapped_delta),
        "joint_haplotype_swap_feature_max_abs": max_abs(features, swapped_features),
    }
    require(max(checks.values()) <= 1e-6, "T0a equivariance or padding invariant failed")
    return checks


def run_root(args: argparse.Namespace) -> dict[str, Any]:
    require(ROOTS[args.root_label] == args.root_seed, "technical root identity differs")
    require(args.marker_count == MARKER_COUNT, "T0a requires exactly 512 markers")
    require(args.repetition in (0, 1), "T0a repetition differs")
    require(args.oci_image.startswith("us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:"),
            "T0a image is not fixed by project digest")
    source_auth_sha = validate_source_auth(
        args.source_auth, args.implementation_commit, args.source_root)
    validate_pre4(args.pre4_contract)
    models.configure_deterministic_cpu()
    model = models.build_model(args.model_family)
    require(not model.training and all(parameter.grad is None for parameter in model.parameters()),
            "T0a model is not inference-only")
    loaded = load_technical(args.technical_dir, args.root_label, args.root_seed,
                            args.independent_verify_receipt, args.genetic_map)
    selected, target, reference, f0, marker_cm, bridge_sha, locus_axis_sha = loaded
    marker_indexes = stratified_markers(selected["cM"], marker_cm, MARKER_COUNT)
    marker_cm_kat = np.ascontiguousarray(marker_cm[marker_indexes], dtype="<f8")
    f0_kat = subset_f0(f0, marker_indexes)
    intervals = materialize.build_interval_table(selected["cM"], marker_cm_kat)
    maxima = {name: int(reference["callable_an"][index].max())
              for index, name in enumerate(("AFR", "EUR", "ASIA"))}
    limit_bytes = memory_limit_bytes()
    started = time.monotonic()
    output_digest, feature_digest = hashlib.sha256(), hashlib.sha256()
    total_tokens = total_rows = total_shards = padded_tokens = 0
    maximum_valid_tokens = 0
    zero_residual_max_abs = simplex_max_abs = 0.0
    checks: dict[str, float] | None = None
    probe_checks: dict[str, float] | None = None
    probe_delta_max_abs = 0.0
    probe_output_sha256 = probe_delta_sha256 = probe_feature_sha256 = None
    sample_count = int(target["sample_key_sha256"].size)
    for person_start in range(0, sample_count, materialize.PERSON_BATCH):
        person_end = min(person_start + materialize.PERSON_BATCH, sample_count)
        prepared = materialize.prepare_person_batch_channels(
            target, reference, maxima, person_start, person_end, root_seed=args.root_seed,
            rotation_id="TECHNICAL_KAT", fit_normalization_manifest_sha256=bridge_sha)
        for radius in materialize.RADII:
            plan = materialize.plan_lazy_marker_chunks(
                intervals, selected["cM"], marker_cm_kat, radius, person_end - person_start)
            for chunk in plan:
                packed = materialize.build_lazy_packed_shard(
                    selected, target, reference, f0_kat, marker_cm_kat, intervals, maxima,
                    radius, person_start, person_end, int(chunk["marker_start"]),
                    int(chunk["marker_end_exclusive"]), prepared_channels=prepared,
                    expected_root_seed=args.root_seed, expected_rotation_id="TECHNICAL_KAT",
                    expected_fit_normalization_manifest_sha256=bridge_sha,
                    inputs_already_validated=True)
                total_shards += 1
                valid = int(packed["rare_tokens"].shape[0])
                total_tokens += valid
                maximum_valid_tokens = max(maximum_valid_tokens, valid)
                for row_start, row_end in padded_batches(packed["row_ptr"]):
                    token_tensor, mask_tensor, f0_tensor = dense_rows(packed, row_start, row_end)
                    padded_tokens += int(token_tensor.shape[0] * token_tensor.shape[1])
                    with torch.inference_mode():
                        probabilities, delta, features = model.forward_with_features(
                            token_tensor, mask_tensor, f0_tensor)
                    require(not torch.is_grad_enabled() or all(
                        parameter.grad is None for parameter in model.parameters()),
                        "T0a created gradients")
                    models.assert_probabilities(probabilities)
                    zero_residual_max_abs = max(zero_residual_max_abs,
                                                max_abs(probabilities, f0_tensor))
                    simplex_max_abs = max(simplex_max_abs, float(torch.max(torch.abs(
                        probabilities.sum(dim=2) - 1.0)).item()))
                    require(float(torch.max(torch.abs(delta)).item()) == 0.0,
                            "zero-initialized residual differs")
                    output_digest.update(probabilities.contiguous().numpy().tobytes())
                    feature_digest.update(features.contiguous().numpy().tobytes())
                    total_rows += int(probabilities.shape[0])
                    if checks is None:
                        checks = invariant_checks(model, token_tensor, mask_tensor, f0_tensor)
                        probe = models.build_model(args.model_family)
                        models.activate_deterministic_probe_head(probe)
                        probe_checks = invariant_checks(probe, token_tensor, mask_tensor, f0_tensor)
                        with torch.inference_mode():
                            probe_p, probe_delta, probe_f = probe.forward_with_features(
                                token_tensor, mask_tensor, f0_tensor)
                        probe_delta_max_abs = float(probe_delta.abs().max())
                        require(probe_delta_max_abs > 0.0,
                                "nonzero T0a positive-control residual was not exercised")
                        probe_output_sha256 = tensor_sha256(probe_p)
                        probe_delta_sha256 = tensor_sha256(probe_delta)
                        probe_feature_sha256 = tensor_sha256(probe_f)
                    rss_fraction(limit_bytes)
                del packed
                gc.collect()
    require(checks is not None and probe_checks is not None and
            all(value is not None for value in (
                probe_output_sha256, probe_delta_sha256, probe_feature_sha256)) and
            zero_residual_max_abs <= 1e-6,
            "T0a did not exercise a valid batch or reproduce F0")
    elapsed = time.monotonic() - started
    peak_fraction = rss_fraction(limit_bytes)
    return {
        "stage": "M33_T0A_FORWARD_TECHNICAL_ROOT",
        "status": "PASS_T0A_FORWARD_ONLY_NON_CONSUMABLE",
        "root_label": args.root_label, "root_seed": args.root_seed,
        "model_family": args.model_family, "repetition": args.repetition,
        "marker_count": MARKER_COUNT, "target_count": sample_count,
        "radii_cM": list(materialize.RADII), "channel_count": 13,
        "rare_locus_count": int(selected["locus_id"].size),
        "valid_tokens": total_tokens, "padded_tokens": padded_tokens,
        "row_count": total_rows, "shard_count": total_shards,
        "maximum_valid_tokens_per_shard": maximum_valid_tokens,
        "output_semantic_sha256": output_digest.hexdigest(),
        "feature_semantic_sha256": feature_digest.hexdigest(),
        "marker_index_semantic_sha256": array_sha256(marker_indexes),
        "technical_locus_key_axis_semantic_sha256": locus_axis_sha,
        "technical_locus_id_projection": "first_16_hex_to_uint64_collision_checked_KAT_only",
        "parameter_count": models.parameter_count(model),
        "parameter_shape_sha256": models.parameter_shape_sha256(model),
        "parameter_value_sha256": models.parameter_value_sha256(model),
        "zero_residual_F0_max_abs": zero_residual_max_abs,
        "simplex_max_abs": simplex_max_abs, "invariance_checks": checks,
        "nonzero_probe_invariance_checks": probe_checks,
        "nonzero_probe_delta_max_abs": probe_delta_max_abs,
        "nonzero_probe_output_sha256": probe_output_sha256,
        "nonzero_probe_delta_sha256": probe_delta_sha256,
        "nonzero_probe_feature_sha256": probe_feature_sha256,
        "elapsed_seconds": elapsed,
        "valid_tokens_per_second": total_tokens / elapsed,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2),
        "memory_limit_gib": limit_bytes / (1024.0 ** 3),
        "peak_rss_fraction": peak_fraction, "memory_warning": peak_fraction >= 0.7,
        "memory_stop_fraction": 0.8, "device": "cpu", "vram_applicable": False,
        "torch_version": torch.__version__, "oci_image": args.oci_image,
        "implementation_commit": args.implementation_commit,
        "source_auth_sha256": source_auth_sha,
        "bridge_receipt_sha256": bridge_sha,
        "truth_read": False, "training": False, "gradients": False,
        "optimizer": False, "model_or_radius_selected": False,
        "predictions_persisted": False, "consumable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-label", choices=sorted(ROOTS), required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--technical-dir", type=Path, required=True)
    parser.add_argument("--independent-verify-receipt", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--model-family", choices=models.FAMILIES, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--marker-count", type=int, default=MARKER_COUNT)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pre4-contract", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_root(args)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
