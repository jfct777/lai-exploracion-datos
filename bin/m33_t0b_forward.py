#!/usr/bin/env python3
"""Stream one zero-head M33 T0b model over all 79,791 chr22 markers."""

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
import m33_t0a_forward as t0a
import m33_t0a_models as models
from m33_m0_factorized_lazy_technical_kat import array_sha256, load_technical


ROOTS = {"root17": 20260817, "root18": 20260818}
FULL_MARKER_COUNT = 79_791
RADII_CM = [0.05, 0.1, 0.2, 0.5]
EXPECTED_CASES = {
    ("root17", "local_linear", 0),
    ("root18", "local_linear", 0),
    ("root17", "small_residual_cnn_1d", 0),
    ("root17", "small_residual_cnn_1d", 1),
    ("root18", "small_residual_cnn_1d", 0),
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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON object differs: {path}")
    return payload


def validate_source_auth(path: Path, commit: str, source_root: Path) -> str:
    payload = load_json(path)
    require(payload.get("stage") == "M33_T0B_SOURCE_AUTH" and
            payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            payload.get("git_commit") == commit, "T0b source authentication differs")
    required = (
        "bin/m31_ordered_linear.py", "bin/m33_m0_contract.py",
        "bin/m33_materialize.py", "bin/m33_m0_factorized_lazy_technical_kat.py",
        "bin/m33_t0a_models.py", "bin/m33_t0a_forward.py",
        "bin/m33_t0b_forward.py", "conf/m33_pre4_preregistration.json",
        "conf/m33_t0b_contract.json",
    )
    expected = payload.get("source_sha256", {})
    for relative in required:
        require(expected.get(relative) == sha256_file(source_root / relative),
                f"authenticated T0b source differs: {relative}")
    return sha256_file(path)


def validate_contract(path: Path) -> str:
    payload = load_json(path)
    scope, execution = payload.get("scope", {}), payload.get("execution", {})
    require(payload.get("stage") == "M33_T0B_FULL_CHR22_CONTRACT" and
            payload.get("status") == "FROZEN_BEFORE_EXECUTION" and
            scope.get("marker_count") == FULL_MARKER_COUNT and
            scope.get("radii_cM") == RADII_CM and
            {tuple(case) for case in scope.get("cases", [])} == EXPECTED_CASES and
            execution.get("maximum_padded_tokens_per_batch") == t0a.MAX_PADDED_TOKENS and
            execution.get("memory_warning_fraction") == 0.70 and
            execution.get("memory_stop_fraction") == 0.80,
            "T0b frozen contract differs")
    return sha256_file(path)


def validate_preflight(path: Path, contract_sha256: str, implementation_commit: str,
                       source_auth_sha256: str, expected_inputs: dict[str, Any]) -> str:
    payload = load_json(path)
    require(payload.get("stage") == "M33_T0B_PREFLIGHT" and
            payload.get("status") == "PASS_T0B_PREFLIGHT_THREE_WAY_ONLY" and
            payload.get("marker_count_by_root") == {
                "root17": FULL_MARKER_COUNT, "root18": FULL_MARKER_COUNT} and
            payload.get("maximum_parallel_forward_processes") == 3 and
            payload.get("mem_available_gib", 0) >= 26.0 and
            payload.get("contract_sha256") == contract_sha256 and
            payload.get("implementation_commit") == implementation_commit and
            payload.get("source_auth_sha256") == source_auth_sha256,
            "T0b preflight differs")
    expected_identity = {
        root: {**identity, "marker_count": FULL_MARKER_COUNT}
        for root, identity in expected_inputs.items()
    }
    require(payload.get("input_identity_by_root") == expected_identity,
            "T0b preflight input identity differs")
    require(all(payload.get(field) is False for field in (
        "truth_read", "training", "gradients", "optimizer",
        "predictions_persisted", "scientific_evidence", "consumable")),
        "T0b preflight firewall differs")
    return sha256_file(path)


def validate_root_files(args: argparse.Namespace, expected: dict[str, Any]) -> None:
    expected_npz = expected.get("technical_npz_sha256", {})
    observed_npz = {name: sha256_file(args.technical_dir / name) for name in (
        "technical_kat_flare_f0_sanitized.npz",
        "technical_kat_reference_rare_summary_incremental.npz",
        "technical_kat_selected_loci_incremental.npz",
        "technical_kat_target_rare_diploid_incremental.npz",
    )}
    require(observed_npz == expected_npz and
            sha256_file(args.technical_dir / "safe_bridge_technical_kat.receipt.json") ==
            expected.get("bridge_receipt_sha256") and
            sha256_file(args.independent_verify_receipt) ==
            expected.get("independent_verify_receipt_sha256") and
            sha256_file(args.genetic_map) == expected.get("genetic_map_sha256"),
            f"T0b {args.root_label} input provenance differs")


def update_tensor_digest(digest: "hashlib._Hash", value: torch.Tensor) -> None:
    array = value.detach().cpu().contiguous().numpy()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def sentinel_indexes(marker_count: int) -> tuple[int, int, int]:
    require(marker_count == FULL_MARKER_COUNT, "T0b sentinel marker count differs")
    return 0, marker_count // 2, marker_count - 1


def sentinel_pass(
    model: torch.nn.Module,
    selected: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    f0: dict[str, np.ndarray],
    marker_cm: np.ndarray,
    intervals: dict[str, np.ndarray],
    interval_validation: materialize.ValidatedIntervalTable,
    maxima: dict[str, int],
    root_seed: int,
    bridge_sha: str,
) -> list[dict[str, Any]]:
    """Rebuild and hash first/middle/last markers separately for each radius."""
    results: list[dict[str, Any]] = []
    sample_count = int(target["sample_key_sha256"].size)
    for radius in materialize.RADII:
        for marker_index in sentinel_indexes(marker_cm.size):
            output_digest, feature_digest = hashlib.sha256(), hashlib.sha256()
            valid_tokens = padded_tokens = row_count = shards = 0
            for person_start in range(0, sample_count, materialize.PERSON_BATCH):
                person_end = min(person_start + materialize.PERSON_BATCH, sample_count)
                prepared = materialize.prepare_person_batch_channels(
                    target, reference, maxima, person_start, person_end,
                    root_seed=root_seed, rotation_id="TECHNICAL_KAT",
                    fit_normalization_manifest_sha256=bridge_sha)
                packed = materialize.build_lazy_packed_shard(
                    selected, target, reference, f0, marker_cm, intervals, maxima, radius,
                    person_start, person_end, marker_index, marker_index + 1,
                    prepared_channels=prepared, expected_root_seed=root_seed,
                    expected_rotation_id="TECHNICAL_KAT",
                    expected_fit_normalization_manifest_sha256=bridge_sha,
                    inputs_already_validated=True,
                    interval_validation=interval_validation)
                valid_tokens += int(packed["rare_tokens"].shape[0]); shards += 1
                for row_start, row_end in t0a.padded_batches(packed["row_ptr"]):
                    tokens, mask, f0_rows = t0a.dense_rows(packed, row_start, row_end)
                    padded_tokens += int(tokens.shape[0] * tokens.shape[1])
                    with torch.inference_mode():
                        probabilities, delta, features = model.forward_with_features(
                            tokens, mask, f0_rows)
                    models.assert_probabilities(probabilities)
                    require(float(delta.abs().max()) == 0.0 and
                            t0a.max_abs(probabilities, f0_rows) <= 1e-6,
                            "T0b sentinel zero head differs")
                    update_tensor_digest(output_digest, probabilities)
                    update_tensor_digest(feature_digest, features)
                    row_count += int(probabilities.shape[0])
            results.append({
                "radius_cM": radius, "marker_index": marker_index,
                "output_semantic_sha256": output_digest.hexdigest(),
                "feature_semantic_sha256": feature_digest.hexdigest(),
                "valid_tokens": valid_tokens, "padded_tokens": padded_tokens,
                "row_count": row_count, "shard_count": shards,
            })
    return results


def replay_sentinels_exact(*args: Any) -> list[dict[str, Any]]:
    """Execute two independent in-process sentinel reconstructions and compare exactly."""
    first = sentinel_pass(*args)
    second = sentinel_pass(*args)
    require(first == second, "T0b first/middle/last sentinel replay differs")
    return first


def run_root(args: argparse.Namespace) -> dict[str, Any]:
    identity = (args.root_label, args.model_family, args.repetition)
    require(identity in EXPECTED_CASES, f"T0b case is not preregistered: {identity}")
    require(ROOTS[args.root_label] == args.root_seed, "T0b technical root differs")
    require(args.marker_count == FULL_MARKER_COUNT, "T0b requires exactly 79,791 markers")
    require(args.oci_image.startswith(
        "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:"),
        "T0b image is not fixed by project digest")
    source_auth_sha = validate_source_auth(
        args.source_auth, args.implementation_commit, args.source_root)
    contract_sha = validate_contract(args.contract)
    contract_payload = load_json(args.contract)
    expected_inputs = contract_payload.get("expected_inputs", {})
    require(set(expected_inputs) == {"root17", "root18"},
            "T0b expected input roots differ")
    expected_root = expected_inputs[args.root_label]
    preflight_sha = validate_preflight(
        args.preflight_receipt, contract_sha, args.implementation_commit, source_auth_sha,
        expected_inputs)
    validate_root_files(args, expected_root)
    t0a.validate_pre4(args.pre4_contract)
    models.configure_deterministic_cpu()
    model = models.build_model(args.model_family)
    require(not model.training and all(parameter.grad is None for parameter in model.parameters()),
            "T0b model is not inference-only")
    selected, target, reference, f0, marker_cm, bridge_sha, locus_axis_sha = load_technical(
        args.technical_dir, args.root_label, args.root_seed,
        args.independent_verify_receipt, args.genetic_map)
    require(marker_cm.size == FULL_MARKER_COUNT and f0["F0"].shape[2] == FULL_MARKER_COUNT,
            "T0b full marker axis differs")
    require(target["sample_key_sha256"].size == expected_root.get("target_count") and
            selected["locus_id"].size == expected_root.get("rare_locus_count") and
            bridge_sha == expected_root.get("bridge_receipt_sha256"),
            "T0b authenticated root axes differ")
    marker_indexes = np.arange(FULL_MARKER_COUNT, dtype="<u8")
    intervals, interval_validation = materialize.build_authenticated_interval_table(
        selected["cM"], marker_cm)
    maxima = {name: int(reference["callable_an"][index].max())
              for index, name in enumerate(("AFR", "EUR", "ASIA"))}
    limit_bytes = t0a.memory_limit_bytes()
    require(5.9 <= limit_bytes / (1024.0 ** 3) <= 6.1,
            "T0b process memory limit is not 6 GiB")
    started = time.monotonic()
    output_digest, feature_digest = hashlib.sha256(), hashlib.sha256()
    total_tokens = padded_tokens = total_rows = total_shards = maximum_valid_tokens = 0
    zero_residual_max_abs = simplex_max_abs = 0.0
    invariance_checks: dict[str, float] | None = None
    sample_count = int(target["sample_key_sha256"].size)
    for person_start in range(0, sample_count, materialize.PERSON_BATCH):
        person_end = min(person_start + materialize.PERSON_BATCH, sample_count)
        prepared = materialize.prepare_person_batch_channels(
            target, reference, maxima, person_start, person_end,
            root_seed=args.root_seed, rotation_id="TECHNICAL_KAT",
            fit_normalization_manifest_sha256=bridge_sha)
        for radius in materialize.RADII:
            plan = materialize.plan_lazy_marker_chunks(
                intervals, selected["cM"], marker_cm, radius, person_end - person_start)
            for chunk in plan:
                packed = materialize.build_lazy_packed_shard(
                    selected, target, reference, f0, marker_cm, intervals, maxima, radius,
                    person_start, person_end, int(chunk["marker_start"]),
                    int(chunk["marker_end_exclusive"]), prepared_channels=prepared,
                    expected_root_seed=args.root_seed, expected_rotation_id="TECHNICAL_KAT",
                    expected_fit_normalization_manifest_sha256=bridge_sha,
                    inputs_already_validated=True,
                    interval_validation=interval_validation)
                total_shards += 1
                valid_tokens = int(packed["rare_tokens"].shape[0])
                total_tokens += valid_tokens
                maximum_valid_tokens = max(maximum_valid_tokens, valid_tokens)
                for row_start, row_end in t0a.padded_batches(packed["row_ptr"]):
                    tokens, mask, f0_rows = t0a.dense_rows(packed, row_start, row_end)
                    current_padded = int(tokens.shape[0] * tokens.shape[1])
                    require(current_padded <= t0a.MAX_PADDED_TOKENS,
                            "T0b padded token budget differs")
                    padded_tokens += current_padded
                    with torch.inference_mode():
                        probabilities, delta, features = model.forward_with_features(
                            tokens, mask, f0_rows)
                    models.assert_probabilities(probabilities)
                    zero_residual_max_abs = max(
                        zero_residual_max_abs, t0a.max_abs(probabilities, f0_rows))
                    simplex_max_abs = max(simplex_max_abs, float(torch.max(torch.abs(
                        probabilities.sum(dim=2) - 1.0)).item()))
                    require(float(delta.abs().max()) == 0.0,
                            "T0b zero-initialized residual differs")
                    update_tensor_digest(output_digest, probabilities)
                    update_tensor_digest(feature_digest, features)
                    total_rows += int(probabilities.shape[0])
                    if invariance_checks is None:
                        invariance_checks = t0a.invariant_checks(model, tokens, mask, f0_rows)
                    t0a.rss_fraction(limit_bytes)
                del packed
                gc.collect()
    require(invariance_checks is not None and zero_residual_max_abs <= 1e-6,
            "T0b did not exercise the model or reproduce F0")
    sentinel_first = replay_sentinels_exact(
        model, selected, target, reference, f0, marker_cm, intervals,
        interval_validation, maxima,
        args.root_seed, bridge_sha)
    elapsed = time.monotonic() - started
    peak_fraction = t0a.rss_fraction(limit_bytes)
    return {
        "stage": "M33_T0B_FULL_CHR22_FORWARD",
        "status": "PASS_T0B_FULL_CHR22_FORWARD_ONLY_NON_CONSUMABLE",
        "root_label": args.root_label, "root_seed": args.root_seed,
        "model_family": args.model_family, "repetition": args.repetition,
        "marker_count": FULL_MARKER_COUNT, "target_count": sample_count,
        "radii_cM": RADII_CM, "channel_count": 13,
        "rare_locus_count": int(selected["locus_id"].size),
        "valid_tokens": total_tokens, "padded_tokens": padded_tokens,
        "row_count": total_rows, "shard_count": total_shards,
        "maximum_valid_tokens_per_shard": maximum_valid_tokens,
        "maximum_padded_tokens_per_batch": t0a.MAX_PADDED_TOKENS,
        "output_semantic_sha256": output_digest.hexdigest(),
        "feature_semantic_sha256": feature_digest.hexdigest(),
        "marker_index_semantic_sha256": array_sha256(marker_indexes),
        "technical_locus_key_axis_semantic_sha256": locus_axis_sha,
        "parameter_count": models.parameter_count(model),
        "parameter_shape_sha256": models.parameter_shape_sha256(model),
        "parameter_value_sha256": models.parameter_value_sha256(model),
        "zero_residual_F0_max_abs": zero_residual_max_abs,
        "simplex_max_abs": simplex_max_abs,
        "invariance_checks": invariance_checks,
        "sentinel_replay": sentinel_first, "sentinel_replay_exact": True,
        "sentinel_passes": 2,
        "elapsed_seconds": elapsed,
        "valid_tokens_per_second": total_tokens / elapsed,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2),
        "memory_limit_gib": limit_bytes / (1024.0 ** 3),
        "peak_rss_fraction": peak_fraction, "memory_warning": peak_fraction >= 0.70,
        "memory_warning_fraction": 0.70, "memory_stop_fraction": 0.80,
        "device": "cpu", "vram_applicable": False,
        "torch_version": torch.__version__, "oci_image": args.oci_image,
        "implementation_commit": args.implementation_commit,
        "source_auth_sha256": source_auth_sha,
        "contract_sha256": contract_sha, "preflight_receipt_sha256": preflight_sha,
        "bridge_receipt_sha256": bridge_sha,
        "truth_read": False, "training": False, "gradients": False,
        "optimizer": False, "predictions_persisted": False,
        "model_or_radius_selected": False, "scientific_evidence": False,
        "consumable": False,
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
    parser.add_argument("--marker-count", type=int, default=FULL_MARKER_COUNT)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pre4-contract", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run_root(args), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
