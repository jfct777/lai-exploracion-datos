#!/usr/bin/env python3
"""Run the preregistered truth-free M33 T0a CPU stress sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import torch

import m33_t0a_models as models
from m33_t0a_forward import memory_limit_bytes, require, rss_fraction, validate_source_auth


PEOPLE = (30, 256, 1024, 2619)
MARKERS = 512
CONTEXT = 64
PERSON_BATCH = 8


def synthetic_batch(person_start: int, person_end: int):
    people = torch.arange(person_start, person_end, dtype=torch.float32)[:, None, None, None]
    marker = torch.arange(MARKERS, dtype=torch.float32)[None, :, None, None]
    locus = torch.arange(CONTEXT, dtype=torch.float32)[None, None, :, None]
    channel = torch.arange(13, dtype=torch.float32)[None, None, None, :]
    tokens = torch.remainder(people * 17 + marker * 13 + locus * 7 + channel * 3, 101) / 100
    tokens = tokens.reshape(-1, CONTEXT, 13).contiguous()
    mask = torch.ones(tokens.shape[:2], dtype=torch.float32)
    row_person = torch.arange(person_start, person_end, dtype=torch.float32).repeat_interleave(MARKERS)
    row_marker = torch.arange(MARKERS, dtype=torch.float32).repeat(person_end - person_start)
    first = 0.2 + torch.remainder(row_person, 7) * 0.01
    second = 0.3 + torch.remainder(row_marker, 5) * 0.01
    third = 1.0 - first - second
    base = torch.stack((first, second, third), dim=1)
    f0 = torch.stack((base, torch.roll(base, shifts=1, dims=1)), dim=1).contiguous()
    return tokens, mask, f0


def ragged_geometry_probe(model: torch.nn.Module) -> dict:
    rows, width = 512, 512
    row = torch.arange(rows, dtype=torch.float32)[:, None, None]
    locus = torch.arange(width, dtype=torch.float32)[None, :, None]
    channel = torch.arange(13, dtype=torch.float32)[None, None, :]
    tokens = torch.remainder(row * 11 + locus * 5 + channel * 3, 97) / 96
    lengths = torch.full((rows,), width, dtype=torch.int64)
    lengths[:6] = torch.tensor([0, 1, 63, 64, 65, 511])
    mask = (torch.arange(width)[None, :] < lengths[:, None]).to(torch.float32)
    tokens = tokens * mask[:, :, None]
    f0 = torch.full((rows, 2, 3), 1.0 / 3.0, dtype=torch.float32)
    with torch.inference_mode():
        probabilities, delta, features = model.forward_with_features(tokens, mask, f0)
    models.assert_probabilities(probabilities)
    require(float(delta.abs().max()) == 0.0 and
            float((probabilities - f0).abs().max()) <= 1e-6,
            "ragged geometry zero residual differs")
    probe = models.build_model("local_linear" if isinstance(model, models.LocalLinear)
                               else "small_residual_cnn_1d")
    models.activate_deterministic_probe_head(probe)
    from m33_t0a_forward import invariant_checks
    checks = invariant_checks(probe, tokens[:12], mask[:12], f0[:12])
    return {
        "padded_tokens": rows * width,
        "valid_tokens": int(mask.sum().item()),
        "lengths_include": [0, 1, 63, 64, 65, 511, 512],
        "feature_semantic_sha256": hashlib.sha256(
            features.contiguous().numpy().tobytes()).hexdigest(),
        "nonzero_probe_invariance_checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=models.FAMILIES, required=True)
    parser.add_argument("--repetition", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pre4-contract", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.oci_image.startswith(
        "us-central1-docker.pkg.dev/uspbr-242713/dnabr-lai/m33-t0a@sha256:"),
        "T0a image is not fixed by project digest")
    source_auth_sha = validate_source_auth(
        args.source_auth, args.implementation_commit, args.source_root,
        ("bin/m33_t0a_stress.py",))
    from m33_t0a_forward import validate_pre4
    validate_pre4(args.pre4_contract)
    models.configure_deterministic_cpu()
    model = models.build_model(args.model_family)
    limit_bytes = memory_limit_bytes()
    geometry = ragged_geometry_probe(model)
    require(geometry["padded_tokens"] == 262_144,
            "ragged geometry did not reach the padded-token budget")
    rows = []
    for person_count in PEOPLE:
        started = time.monotonic()
        output_digest = hashlib.sha256()
        token_count = 0
        for person_start in range(0, person_count, PERSON_BATCH):
            person_end = min(person_start + PERSON_BATCH, person_count)
            tokens, mask, f0 = synthetic_batch(person_start, person_end)
            require(tokens.shape[0] == (person_end - person_start) * MARKERS and
                    tokens.shape[0] * tokens.shape[1] <= 262_144,
                    "synthetic stress token budget differs")
            with torch.inference_mode():
                probabilities, delta, _features = model.forward_with_features(tokens, mask, f0)
            models.assert_probabilities(probabilities)
            require(float(delta.abs().max()) == 0.0 and
                    float((probabilities - f0).abs().max()) <= 1e-6,
                    "synthetic stress zero residual differs")
            output_digest.update(probabilities.contiguous().numpy().tobytes())
            token_count += int(tokens.shape[0] * tokens.shape[1])
            rss_fraction(limit_bytes)
        elapsed = time.monotonic() - started
        rows.append({
            "person_count": person_count, "marker_count": MARKERS,
            "context_length": CONTEXT, "valid_tokens": token_count,
            "output_semantic_sha256": output_digest.hexdigest(),
            "elapsed_seconds": elapsed, "valid_tokens_per_second": token_count / elapsed,
            "peak_rss_fraction": rss_fraction(limit_bytes),
        })
    peak_fraction = rss_fraction(limit_bytes)
    receipt = {
        "stage": "M33_T0A_SYNTHETIC_STRESS",
        "status": "PASS_T0A_SYNTHETIC_STRESS_NON_CONSUMABLE",
        "model_family": args.model_family, "repetition": args.repetition,
        "stress_people": list(PEOPLE), "stress_loci": MARKERS,
        "stress_loci_interpretation": "512_central_markers_each_with_streamed_rare_context",
        "synthetic_context_length": CONTEXT,
        "stress_is_streaming_not_full_DNABR_matrix": True,
        "synthetic_people_are_not_individuals": True,
        "rows": rows, "ragged_geometry_probe": geometry,
        "parameter_count": models.parameter_count(model),
        "parameter_shape_sha256": models.parameter_shape_sha256(model),
        "parameter_value_sha256": models.parameter_value_sha256(model),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2),
        "memory_limit_gib": limit_bytes / (1024.0 ** 3),
        "peak_rss_fraction": peak_fraction, "memory_warning": peak_fraction >= 0.7,
        "memory_stop_fraction": 0.8, "device": "cpu", "vram_applicable": False,
        "torch_version": torch.__version__, "oci_image": args.oci_image,
        "implementation_commit": args.implementation_commit,
        "source_auth_sha256": source_auth_sha,
        "truth_read": False, "training": False, "gradients": False,
        "optimizer": False, "predictions_persisted": False, "consumable": False,
        "model_or_radius_selected": False,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
