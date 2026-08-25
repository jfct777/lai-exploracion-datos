#!/usr/bin/env python3
"""Run one synthetic, optimizer-free M33 T1 backward dry-run case."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import torch

import m33_t0a_models as models
import m33_t1_preflight as preflight


PEOPLE = 8
MARKERS = 256
ROWS = PEOPLE * MARKERS
CONTEXT = 128
CHANNELS = 13
PADDED_TOKENS = ROWS * CONTEXT
RUN_SEED = 20260824
EXPECTED_CASES = {
    ("local_linear", 0), ("local_linear", 1),
    ("small_residual_cnn_1d", 0), ("small_residual_cnn_1d", 1),
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


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parameter_gradient_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None, f"missing gradient: {name}")
        gradient = parameter.grad.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(gradient.dtype).encode("ascii"))
        digest.update(json.dumps(list(gradient.shape), separators=(",", ":")).encode("ascii"))
        digest.update(gradient.numpy().tobytes(order="C"))
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
    require(fraction < 0.80, f"T1 peak RSS reached stop fraction: {fraction:.6f}")
    return fraction


def validate_source_auth(path: Path, commit: str, source_root: Path) -> str:
    payload = preflight.load_json(path)
    require(payload.get("stage") == "M33_T1_SOURCE_AUTH" and
            payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            payload.get("git_commit") == commit, "T1 source authentication differs")
    expected = payload.get("source_sha256", {})
    runtime = (
        "bin/m33_t0a_models.py", "bin/m33_t1_preflight.py",
        "bin/m33_t1_backward.py", "conf/m33_pre4_preregistration.json",
        "conf/m33_t1_contract.json",
    )
    for relative in runtime:
        require(expected.get(relative) == sha256_file(source_root / relative),
                f"authenticated T1 source differs: {relative}")
    return sha256_file(path)


def validate_pre4(path: Path) -> None:
    contract = preflight.load_json(path)
    screen = contract.get("model_screen", {})
    training = screen.get("training", {})
    gate = contract.get("technical_gates", {}).get("T1_backward_dry_run", {})
    require(screen.get("families") == list(models.FAMILIES) and
            training.get("loss") == "weighted_per_marker_cross_entropy_defined_by_loss_normalization" and
            training.get("batch_people") == PEOPLE and
            training.get("marker_block_length") == MARKERS and
            gate.get("operation") == "one_forward_backward_without_optimizer_step_on_synthetic_stress" and
            set(gate.get("checks", [])) == {
                "finite_loss", "finite_gradients", "peak_RSS_le_0.8_host_RAM",
                "peak_VRAM_le_0.8_device_RAM", "no_checkpoint_written",
            }, "PRE-4 T1 gate differs")


def validate_preflight_receipt(path: Path, contract_sha: str, source_auth_sha: str,
                               commit: str, oci_image: str) -> str:
    receipt = preflight.load_json(path)
    require(receipt.get("stage") == "M33_T1_PREFLIGHT" and
            receipt.get("status") == "PASS_T1_PREFLIGHT_SYNTHETIC_ONLY" and
            receipt.get("contract_sha256") == contract_sha and
            receipt.get("source_auth_sha256") == source_auth_sha and
            receipt.get("implementation_commit") == commit and
            receipt.get("oci_image") == oci_image and
            receipt.get("t0b_aggregate_sha256") == preflight.EXPECTED_T0B_SHA256 and
            receipt.get("maximum_parallel_processes") == 1 and
            all(receipt.get(field) is False for field in (
                "truth_read", "real_data_read", "training", "optimizer",
                "checkpoint_written", "predictions_persisted", "scientific_evidence",
                "consumable")), "T1 preflight receipt differs")
    return sha256_file(path)


def synthetic_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row = torch.arange(ROWS, dtype=torch.float32)[:, None, None]
    position = torch.arange(CONTEXT, dtype=torch.float32)[None, :, None]
    channel = torch.arange(CHANNELS, dtype=torch.float32)[None, None, :]
    tokens = ((row * 17 + position * 13 + channel * 7).remainder(101) - 50) / 50
    lengths = torch.tensor([0, 1, 64, 128], dtype=torch.int64).repeat(ROWS // 4)
    mask = (torch.arange(CONTEXT)[None, :] < lengths[:, None]).to(torch.float32)

    patterns = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0e-8, 0.4, 0.59999999],
        [1.0 / 3, 1.0 / 3, 1.0 / 3],
        [0.0, 0.2, 0.8],
    ], dtype=torch.float32)
    f0 = patterns[torch.arange(ROWS) % 4][:, None, :].repeat(1, 2, 1)
    f0[:, 1] = f0[:, 1].roll(1, dims=1)
    require(torch.max(torch.abs(f0.sum(dim=2) - 1.0)).item() <= 5e-6,
            "synthetic F0 differs")

    person = torch.arange(PEOPLE)[:, None, None]
    marker = torch.arange(MARKERS)[None, :, None]
    haplotype = torch.arange(2)[None, None, :]
    labels_cube = (person + marker.div(32, rounding_mode="floor") + haplotype) % 3
    # Shift 22 deterministic block boundaries by one marker. This preserves seven
    # internal transitions per person/haplotype while balancing all three classes.
    for source_class, required_shifts in ((1, 11), (2, 11)):
        candidates: list[tuple[int, int, int]] = []
        for person_index in range(PEOPLE):
            for haplotype_index in range(2):
                sequence = labels_cube[person_index, :, haplotype_index]
                for marker_index in range(MARKERS - 1):
                    left, right = int(sequence[marker_index]), int(sequence[marker_index + 1])
                    if left == source_class and right == 0:
                        candidates.append((person_index, marker_index, haplotype_index))
                    elif left == 0 and right == source_class:
                        candidates.append((person_index, marker_index + 1, haplotype_index))
        require(len(candidates) >= required_shifts,
                "synthetic label boundary inventory differs")
        for person_index, marker_index, haplotype_index in candidates[:required_shifts]:
            labels_cube[person_index, marker_index, haplotype_index] = 0
    labels = labels_cube.reshape(ROWS, 2).to(torch.int64)
    counts = torch.bincount(labels.flatten(), minlength=3)
    require(int(counts.max() - counts.min()) <= 1, "synthetic labels are not balanced")
    transition = torch.zeros_like(labels_cube, dtype=torch.float32)
    transition[:, 1:] = (labels_cube[:, 1:] != labels_cube[:, :-1]).to(torch.float32)
    weights = (1.0 + transition).reshape(ROWS, 2)
    require(torch.all(weights.reshape(PEOPLE, MARKERS, 2)[:, 0] == 1).item(),
            "boundary weights cross a person boundary")
    transition_counts = transition.sum(dim=1)
    require(torch.all(transition_counts == 7).item() and int(transition.sum()) == 112 and
            float(weights.sum()) == 4208.0,
            "synthetic boundary weight geometry differs")
    return tokens.contiguous(), mask.contiguous(), f0.contiguous(), labels, weights


def weighted_cross_entropy(f0: torch.Tensor, delta: torch.Tensor,
                           labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    require(f0.shape == delta.shape == (ROWS, 2, 3) and
            labels.shape == weights.shape == (ROWS, 2),
            "T1 loss axes differ")
    logits = torch.log(f0.clamp_min(1e-7)) + delta
    chosen_log_probability = torch.log_softmax(logits, dim=2).gather(
        2, labels.unsqueeze(2)).squeeze(2)
    numerator = (-chosen_log_probability * weights).sum()
    denominator = weights.sum()
    require(float(denominator) > 0, "T1 loss denominator differs")
    return numerator / denominator


def stage_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    names = ["head"] if isinstance(model, models.LocalLinear) else [
        "stem", "block1", "block2", "head1", "head2"]
    totals = {name: 0.0 for name in names}
    for parameter_name, parameter in model.named_parameters():
        require(parameter.grad is not None, f"missing gradient: {parameter_name}")
        require(torch.isfinite(parameter.grad).all().item(),
                f"non-finite gradient: {parameter_name}")
        stage = next((name for name in names if parameter_name == name or
                      parameter_name.startswith(name + ".")), None)
        require(stage is not None, f"unmapped parameter stage: {parameter_name}")
        totals[stage] += float(parameter.grad.detach().double().square().sum().item())
    return {name: math.sqrt(value) for name, value in totals.items()}


def run_subcase(family: str, probe: bool, base_tokens: torch.Tensor, mask: torch.Tensor,
                f0: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> dict[str, Any]:
    torch.manual_seed(RUN_SEED)
    model = models.build_model(family)
    if probe:
        models.activate_deterministic_probe_head(model)
    model.train()
    before = models.parameter_value_sha256(model)
    tokens = base_tokens.detach().clone().requires_grad_(True)
    torch.manual_seed(RUN_SEED)
    probabilities, delta, _features = model.forward_with_features(tokens, mask, f0)
    models.assert_probabilities(probabilities)
    loss = weighted_cross_entropy(f0, delta, labels, weights)
    require(torch.isfinite(loss).item() and float(loss.detach()) > 0,
            "T1 loss is not finite and positive")
    loss.backward()
    norms = stage_gradient_norms(model)
    global_norm = math.sqrt(sum(value * value for value in norms.values()))
    final_stage = "head" if family == "local_linear" else "head2"
    require(global_norm > 0 and norms[final_stage] > 0,
            "T1 final or global gradient is zero")

    gradient = tokens.grad
    require(gradient is not None and torch.isfinite(gradient).all().item(),
            "T1 input gradient differs")
    valid = mask.to(torch.bool)
    valid_norm = float(gradient[valid].double().norm().item()) if valid.any() else 0.0
    padding_max = float(gradient[~valid].abs().max().item()) if (~valid).any() else 0.0
    require(padding_max == 0.0, "T1 padding received input gradient")
    if probe:
        require(all(value > 0 for value in norms.values()) and valid_norm > 0,
                "T1 private probe did not reach every trainable stage")
    elif family == "small_residual_cnn_1d":
        require(all(norms[name] == 0.0 for name in ("stem", "block1", "block2", "head1")),
                "T1 production CNN upstream gradient pattern differs")
    after = models.parameter_value_sha256(model)
    require(before == after, "T1 backward mutated parameter values")
    return {
        "name": "private_nonzero_probe_head" if probe else "production_zero_head",
        "loss": float(loss.detach().item()),
        "probability_sha256": tensor_sha256(probabilities),
        "delta_sha256": tensor_sha256(delta),
        "gradient_sha256": parameter_gradient_sha256(model),
        "stage_gradient_norms": norms,
        "global_gradient_norm": global_norm,
        "valid_input_gradient_norm": valid_norm,
        "padding_input_gradient_max_abs": padding_max,
        "parameter_value_sha256_before": before,
        "parameter_value_sha256_after": after,
        "parameter_count": models.parameter_count(model),
        "parameter_shape_sha256": models.parameter_shape_sha256(model),
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    require((args.model_family, args.repetition) in EXPECTED_CASES,
            "T1 case is not frozen")
    contract_sha, contract = preflight.validate_contract(args.contract)
    require(args.oci_image == contract["execution"]["oci_image"], "T1 OCI image differs")
    source_auth_sha = validate_source_auth(
        args.source_auth, args.implementation_commit, args.source_root)
    validate_pre4(args.pre4_contract)
    preflight_sha = validate_preflight_receipt(
        args.preflight_receipt, contract_sha, source_auth_sha,
        args.implementation_commit, args.oci_image)
    models.configure_deterministic_cpu()
    limit_bytes = memory_limit_bytes()
    limit_gib = limit_bytes / (1024.0 ** 3)
    require(math.isclose(limit_gib, contract["execution"]["process_memory_gib"],
                         rel_tol=0.0, abs_tol=0.05),
            "T1 cgroup memory limit differs from frozen contract")
    started = time.monotonic()
    tokens, mask, f0, labels, weights = synthetic_fixture()
    require(tokens.shape == (ROWS, CONTEXT, CHANNELS) and PADDED_TOKENS == 262_144 and
            int(mask.sum().item()) > 0, "T1 synthetic stress geometry differs")
    production = run_subcase(args.model_family, False, tokens, mask, f0, labels, weights)
    production["peak_rss_fraction_after"] = rss_fraction(limit_bytes)
    gc.collect()
    probe = run_subcase(args.model_family, True, tokens, mask, f0, labels, weights)
    probe["peak_rss_fraction_after"] = rss_fraction(limit_bytes)
    elapsed = time.monotonic() - started
    peak_fraction = rss_fraction(limit_bytes)
    return {
        "stage": "M33_T1_BACKWARD_DRY_RUN",
        "status": "PASS_T1_BACKWARD_CASE_TECHNICAL_ONLY_NON_CONSUMABLE",
        "model_family": args.model_family,
        "repetition": args.repetition,
        "subcases": [production, probe],
        "people": PEOPLE,
        "central_markers": MARKERS,
        "rows": ROWS,
        "channels": CHANNELS,
        "maximum_context": CONTEXT,
        "padded_tokens": PADDED_TOKENS,
        "valid_tokens": int(mask.sum().item()),
        "context_lengths": [0, 1, 64, 128],
        "synthetic_target_class_counts": torch.bincount(labels.flatten(), minlength=3).tolist(),
        "transition_count": int((weights > 1).sum().item()),
        "transition_count_per_person_haplotype": 7,
        "boundary_weight_sum": float(weights.sum().item()),
        "first_marker_weight_max": float(weights.reshape(PEOPLE, MARKERS, 2)[:, 0].max().item()),
        "boundary_weight_beta": 1.0,
        "elapsed_seconds": elapsed,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2),
        "memory_limit_gib": limit_gib,
        "peak_rss_fraction": peak_fraction,
        "memory_warning_fraction": 0.70,
        "memory_stop_fraction": 0.80,
        "memory_warning": peak_fraction >= 0.70,
        "device": "cpu",
        "vram_applicable": False,
        "torch_version": torch.__version__,
        "oci_image": args.oci_image,
        "implementation_commit": args.implementation_commit,
        "source_auth_sha256": source_auth_sha,
        "contract_sha256": contract_sha,
        "preflight_receipt_sha256": preflight_sha,
        "synthetic_only": True,
        "truth_read": False,
        "real_data_read": False,
        "training": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "checkpoint_written": False,
        "predictions_persisted": False,
        "tensors_persisted": False,
        "scientific_evidence": False,
        "consumable": False,
        "development_open": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=models.FAMILIES, required=True)
    parser.add_argument("--repetition", type=int, choices=(0, 1), required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pre4-contract", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--oci-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existing = {path.resolve() for path in Path.cwd().rglob("*") if path.is_file()}
    receipt = run_case(args)
    created = {path.resolve() for path in Path.cwd().rglob("*") if path.is_file()} - existing
    require(not created, f"T1 created an undeclared file before receipt: {sorted(map(str, created))}")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
