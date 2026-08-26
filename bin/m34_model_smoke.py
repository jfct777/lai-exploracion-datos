#!/usr/bin/env python3
"""Run deterministic synthetic forward/backward smokes for M34 model tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

import m34_adaptive_sweep as sweep
import m34_models as models


ARMS = ("RD", "RE")
PASS_STATUS = "PASS_TECHNICAL_SMOKE_ONLY_NO_SCIENTIFIC_RESULT"


class SmokeError(ValueError):
    """Raised when a task or technical invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_bytes(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB. M34 currently runs on Linux containers only.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _config_lookup(contract: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (family, config["id"]): config
        for family, family_spec in contract["families"].items()
        for config in family_spec["configs"]
    }


def _normalize_task(contract: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(task, dict), "each task must be an object")
    family = task.get("family")
    config_id = task.get("config_id")
    lookup = _config_lookup(contract)
    _require((family, config_id) in lookup,
             f"undeclared task configuration: {family}/{config_id}")
    arm = task.get("arm")
    _require(arm in ARMS, f"invalid task arm: {arm}")
    try:
        seed = int(task.get("seed"))
        maximum_updates = int(task.get("maximum_updates"))
        learning_rate = float(task.get("learning_rate"))
        weight_decay = float(task.get("weight_decay"))
        radius_cM = float(task.get("radius_cM"))
    except (TypeError, ValueError) as error:
        raise SmokeError(f"task has invalid numeric fields: {family}/{config_id}/{arm}") from error
    _require(seed >= 0, "task seed must be non-negative")
    _require(math.isfinite(radius_cM) and radius_cM > 0,
             "task radius_cM must be finite and positive")
    sweep_stage = task.get("sweep_stage")
    _require(sweep_stage in contract["stages"],
             f"task sweep_stage is not declared: {sweep_stage}")
    declared_stage = False
    for stage_name, stage in contract["stages"].items():
        if stage_name != sweep_stage:
            continue
        allowed_seeds = stage.get("seeds", [stage.get("seed")])
        allowed_rotations = stage.get("rotations", [stage.get("rotation")])
        if stage_name == "radius_sensitivity":
            allowed_radii = stage["radii_cM"]
        elif stage_name == "finalists":
            allowed_radii = contract["stages"]["radius_sensitivity"]["radii_cM"]
        else:
            allowed_radii = [stage["radius_cM"]]
        if (maximum_updates == int(stage["maximum_updates"])
                and seed in allowed_seeds and task.get("rotation") in allowed_rotations
                and radius_cM in allowed_radii):
            declared_stage = True
            break
    _require(declared_stage,
             "task seed, rotation and update budget do not match a declared stage")
    expected_lr = float(contract["training"]["learning_rate"])
    expected_wd = float(lookup[(family, config_id)].get("training_overrides", {}).get(
        "weight_decay", contract["training"]["weight_decay"]))
    _require(math.isfinite(learning_rate) and learning_rate == expected_lr,
             "task learning rate differs from the contract")
    _require(math.isfinite(weight_decay) and weight_decay == expected_wd,
             "task weight decay differs from the contract")
    rotation = task.get("rotation")
    _require(isinstance(rotation, str) and rotation,
             "task rotation must be a non-empty string")
    return {
        "family": family,
        "config_id": config_id,
        "seed": seed,
        "rotation": rotation,
        "arm": arm,
        "radius_cM": radius_cM,
        "sweep_stage": sweep_stage,
        "maximum_updates": maximum_updates,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


def normalize_tasks(contract: dict[str, Any], tasks: Iterable[dict[str, Any]]
                    ) -> list[dict[str, Any]]:
    normalized = [_normalize_task(contract, task) for task in tasks]
    _require(normalized, "at least one smoke task is required")
    keys = [(
        row["family"], row["config_id"], row["radius_cM"], row["seed"],
        row["rotation"], row["sweep_stage"], row["arm"]
    ) for row in normalized]
    _require(len(keys) == len(set(keys)), "duplicate smoke task")
    return sorted(normalized, key=lambda row: (
        row["family"], row["config_id"], row["radius_cM"], row["seed"],
        row["rotation"], row["sweep_stage"], row["arm"]
    ))


def tasks_for_config(contract: dict[str, Any], family: str, config_id: str,
                     stage_name: str, seed: int | None, rotation: str | None,
                     arm: str, radius_cM: float | None = None) -> list[dict[str, Any]]:
    stages = contract["stages"]
    _require(stage_name in stages, f"unknown contract stage: {stage_name}")
    stage = stages[stage_name]
    declared_seeds = stage.get("seeds", [stage.get("seed")])
    declared_rotations = stage.get("rotations", [stage.get("rotation")])
    chosen_seed = int(seed if seed is not None else declared_seeds[0])
    chosen_rotation = str(rotation if rotation is not None else declared_rotations[0])
    _require(chosen_seed in declared_seeds,
             f"seed is not declared for stage {stage_name}: {chosen_seed}")
    _require(chosen_rotation in declared_rotations,
             f"rotation is not declared for stage {stage_name}: {chosen_rotation}")
    if stage_name == "radius_sensitivity":
        declared_radii = list(stage["radii_cM"])
        chosen_radii = declared_radii if radius_cM is None else [float(radius_cM)]
    elif stage_name == "finalists":
        _require(radius_cM is not None,
                 "finalists smoke requires an explicitly selected radius_cM")
        declared_radii = list(contract["stages"]["radius_sensitivity"]["radii_cM"])
        chosen_radii = [float(radius_cM)]
    else:
        declared_radii = [float(stage["radius_cM"])]
        chosen_radii = declared_radii if radius_cM is None else [float(radius_cM)]
    _require(all(radius in declared_radii for radius in chosen_radii),
             f"radius is not declared for stage {stage_name}")
    chosen_arms = ARMS if arm == "both" else (arm,)
    lookup = _config_lookup(contract)
    _require((family, config_id) in lookup,
             f"undeclared task configuration: {family}/{config_id}")
    config = lookup[(family, config_id)]
    training = contract["training"]
    tasks = [{
        "family": family,
        "config_id": config_id,
        "seed": chosen_seed,
        "rotation": chosen_rotation,
        "arm": selected_arm,
        "radius_cM": selected_radius,
        "sweep_stage": stage_name,
        "maximum_updates": int(stage["maximum_updates"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(config.get("training_overrides", {}).get(
            "weight_decay", training["weight_decay"])),
    } for selected_radius in chosen_radii for selected_arm in chosen_arms]
    return normalize_tasks(contract, tasks)


def _task_seed(task: dict[str, Any]) -> int:
    # Arm is deliberately absent: an RD/RE pair gets identical geometry and initialization.
    identity = {key: task[key] for key in (
        "family", "config_id", "radius_cM", "seed", "rotation", "sweep_stage"
    )}
    return int.from_bytes(hashlib.sha256(_canonical_bytes(identity)).digest()[:8], "big") % (2**31)


def synthetic_fixture(task: dict[str, Any], channels: int, ancestries: int,
                      batch_size: int, context_length: int,
                      haplotypes: int) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
    _require(channels > 0, "channels must be positive")
    _require(ancestries >= 2, "at least two ancestries are required")
    _require(batch_size >= 3, "batch_size must be at least three")
    _require(context_length >= 7, "context_length must be at least seven")
    _require(haplotypes >= 1, "haplotypes must be positive")
    generator = torch.Generator().manual_seed(_task_seed(task))
    tokens = torch.randn(batch_size, context_length, channels, generator=generator)
    mask = torch.zeros(batch_size, context_length, dtype=torch.bool)
    lengths = torch.linspace(0, context_length, batch_size).round().to(torch.int64)
    for row, length in enumerate(lengths.tolist()):
        if length:
            mask[row, :length] = True
    # An interior hole catches implementations that assume every ragged mask is prefix-only.
    if context_length > 8 and lengths[-1] > 5:
        mask[-1, context_length // 2] = False
    tokens = tokens * mask.unsqueeze(-1)
    baseline = torch.softmax(
        torch.randn(batch_size, haplotypes, ancestries, generator=generator), dim=-1)
    zero_class = (
        torch.arange(batch_size)[:, None] + torch.arange(haplotypes)[None, :]
    ) % ancestries
    baseline.scatter_(2, zero_class.unsqueeze(-1), 0.0)
    baseline /= baseline.sum(dim=-1, keepdim=True)
    labels = (zero_class + 1) % ancestries
    return tokens, mask, baseline, labels


def _model_spec(contract: dict[str, Any], task: dict[str, Any], channels: int,
                ancestries: int, zero_init_head: bool) -> models.ModelSpec:
    config = _config_lookup(contract)[(task["family"], task["config_id"])]
    values = dict(config["model_spec"])
    values["dilations"] = tuple(values["dilations"])
    return models.ModelSpec(
        family=task["family"],
        channels=channels,
        ancestries=ancestries,
        zero_init_head=zero_init_head,
        seed=task["seed"],
        **values,
    )


def _simplex_error(probabilities: torch.Tensor) -> float:
    models.assert_simplex(probabilities)
    return float(torch.max(torch.abs(probabilities.sum(dim=-1) - 1.0)).item())


def smoke_task(contract: dict[str, Any], task: dict[str, Any], channels: int,
               ancestries: int, batch_size: int, context_length: int,
               haplotypes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    tokens, mask, baseline, labels = synthetic_fixture(
        task, channels, ancestries, batch_size, context_length, haplotypes)

    zero_spec = _model_spec(contract, task, channels, ancestries, zero_init_head=True)
    zero_model = models.build_model(zero_spec).eval()
    with torch.inference_mode():
        zero_probabilities, zero_delta = zero_model.forward_with_delta(tokens, mask, baseline)
    zero_error = float(torch.max(torch.abs(zero_probabilities - baseline)).item())
    _require(float(zero_delta.abs().max().item()) == 0.0,
             "zero-initialized head produced a nonzero residual")
    _require(zero_error <= 5e-7, "zero-initialized head did not reproduce the baseline")
    zero_simplex = _simplex_error(zero_probabilities)

    active_spec = _model_spec(contract, task, channels, ancestries, zero_init_head=False)
    model = models.build_model(active_spec).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=task["learning_rate"], weight_decay=task["weight_decay"])
    before_parameters = _parameter_sha256(model)
    torch.manual_seed(_task_seed(task) + 1)
    probabilities = model(tokens, mask, baseline)
    pre_simplex = _simplex_error(probabilities)
    selected = probabilities.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    loss = -torch.log(selected.clamp_min(1e-8)).mean()
    _require(torch.isfinite(loss).item(), "training smoke loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    _require(gradients and all(gradient is not None for gradient in gradients),
             "a trainable parameter did not receive a gradient")
    _require(all(torch.isfinite(gradient).all().item() for gradient in gradients
                 if gradient is not None), "a model gradient is non-finite")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(contract["training"]["gradient_clip_norm"])).item())
    _require(math.isfinite(gradient_norm), "gradient norm is non-finite")
    nonzero_gradients = sum(
        int(torch.count_nonzero(gradient).item() > 0)
        for gradient in gradients if gradient is not None)
    _require(nonzero_gradients > 0, "all model gradients are zero")
    optimizer.step()
    after_parameters = _parameter_sha256(model)
    _require(after_parameters != before_parameters, "optimizer step did not change parameters")
    model.eval()
    with torch.inference_mode():
        post_probabilities = model(tokens, mask, baseline)
        post_selected = post_probabilities.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        post_loss = -torch.log(post_selected.clamp_min(1e-8)).mean()
    post_simplex = _simplex_error(post_probabilities)
    _require(torch.isfinite(post_loss).item(), "post-step loss is non-finite")
    zero_positions = baseline == 0
    _require(torch.all(post_probabilities[zero_positions] > 0).item(),
             "active residual head could not reopen a zero baseline class")

    deterministic = {
        "task": task,
        "synthetic_pairing_seed": _task_seed(task),
        "synthetic_arm_semantics": "technical_only_same_fixture_for_RD_and_RE",
        "model_spec": {
            **{key: getattr(active_spec, key) for key in (
                "family", "channels", "ancestries", "hidden_dim", "depth",
                "kernel_size", "dropout", "lstm_layers", "transformer_heads",
                "transformer_ff_dim", "transformer_max_tokens", "zero_init_head", "seed",
            )},
            "dilations": list(active_spec.dilations),
        },
        "parameter_count": models.parameter_count(model),
        "input_shape": list(tokens.shape),
        "baseline_shape": list(baseline.shape),
        "valid_context_tokens": int(mask.sum().item()),
        "empty_context_rows": int((~mask.any(dim=1)).sum().item()),
        "baseline_zero_count": int(zero_positions.sum().item()),
        "zero_head_delta_max_abs": float(zero_delta.abs().max().item()),
        "zero_head_baseline_max_abs_error": zero_error,
        "zero_head_simplex_max_abs_error": zero_simplex,
        "active_pre_step_simplex_max_abs_error": pre_simplex,
        "active_post_step_simplex_max_abs_error": post_simplex,
        "loss_before_step": float(loss.item()),
        "loss_after_step": float(post_loss.item()),
        "gradient_norm_before_clip": gradient_norm,
        "gradient_tensor_count": len(gradients),
        "nonzero_gradient_tensor_count": nonzero_gradients,
        "gradients_finite": True,
        "optimizer": "AdamW",
        "optimizer_steps_executed": 1,
        "declared_maximum_updates_not_executed": task["maximum_updates"],
        "parameter_sha256_before_step": before_parameters,
        "parameter_sha256_after_step": after_parameters,
        "prediction_sha256_after_step": _tensor_sha256(post_probabilities),
    }
    telemetry = {
        "task": {key: task[key] for key in (
            "family", "config_id", "radius_cM", "seed", "rotation", "sweep_stage", "arm")},
        "wall_seconds": time.perf_counter() - started,
        "process_cumulative_peak_rss_mib_after_task": _peak_rss_mib(),
    }
    return deterministic, telemetry


def run_smoke(contract: dict[str, Any], tasks: Iterable[dict[str, Any]], *,
              contract_sha256: str, task_source_sha256: str, channels: int,
              ancestries: int, batch_size: int, context_length: int,
              haplotypes: int) -> dict[str, Any]:
    normalized = normalize_tasks(contract, tasks)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits setting inter-op threads before parallel work starts.
        pass
    torch.use_deterministic_algorithms(True)
    results = []
    telemetry_rows = []
    run_started = time.perf_counter()
    for task in normalized:
        result, telemetry = smoke_task(
            contract, task, channels, ancestries, batch_size, context_length, haplotypes)
        results.append(result)
        telemetry_rows.append(telemetry)
    deterministic = {
        "schema_version": "1.0.0",
        "stage": "M34_MODEL_SMOKE",
        "status": PASS_STATUS,
        "scientific_result": False,
        "contract_sha256": contract_sha256,
        "task_source_sha256": task_source_sha256,
        "torch_version": torch.__version__,
        "task_count": len(results),
        "fixture": {
            "channels": channels,
            "ancestries": ancestries,
            "batch_size": batch_size,
            "context_length": context_length,
            "haplotypes": haplotypes,
            "ragged": True,
            "contains_empty_context": True,
            "contains_baseline_zeros": True,
        },
        "results": results,
    }
    return {
        "deterministic": deterministic,
        "deterministic_sha256": _sha256(_canonical_bytes(deterministic)),
        "telemetry": {
            "excluded_from_deterministic_sha256": True,
            "wall_seconds": time.perf_counter() - run_started,
            "process_peak_rss_mib": _peak_rss_mib(),
            "tasks": telemetry_rows,
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--family", choices=models.FAMILIES)
    parser.add_argument("--config-id")
    parser.add_argument(
        "--contract-stage",
        choices=("triage", "local_expansion", "radius_sensitivity", "finalists"),
        default="triage")
    parser.add_argument("--arm", choices=("RD", "RE", "both"), default="both")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rotation")
    parser.add_argument("--radius-cm", type=float)
    parser.add_argument("--channels", type=int, default=13)
    parser.add_argument("--ancestries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=37)
    parser.add_argument("--haplotypes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_bytes = args.contract.read_bytes()
    contract = sweep.validate_contract(sweep.strict_json(args.contract))
    if args.manifest is not None:
        _require(args.family is None and args.config_id is None,
                 "--manifest cannot be combined with --family or --config-id")
        manifest_bytes = args.manifest.read_bytes()
        manifest = sweep.strict_json(args.manifest)
        tasks = manifest.get("tasks")
        _require(isinstance(tasks, list), "task manifest requires a tasks list")
        _require(manifest.get("status") == "PLAN_ONLY_NO_EXECUTION",
                 "task manifest is not an executable-free M34 plan")
        _require(manifest.get("task_count") == len(tasks),
                 "task manifest count differs from its tasks list")
        task_source_sha256 = _sha256(manifest_bytes)
    else:
        _require(args.family is not None and args.config_id is not None,
                 "provide --manifest or both --family and --config-id")
        tasks = tasks_for_config(
            contract, args.family, args.config_id, args.contract_stage,
            args.seed, args.rotation, args.arm, args.radius_cm)
        task_source_sha256 = _sha256(_canonical_bytes(tasks))
    receipt = run_smoke(
        contract, tasks,
        contract_sha256=_sha256(contract_bytes),
        task_source_sha256=task_source_sha256,
        channels=args.channels,
        ancestries=args.ancestries,
        batch_size=args.batch_size,
        context_length=args.context_length,
        haplotypes=args.haplotypes,
    )
    _write_json_atomic(args.output, receipt)
    print(json.dumps({
        "stage": receipt["deterministic"]["stage"],
        "status": receipt["deterministic"]["status"],
        "task_count": receipt["deterministic"]["task_count"],
        "deterministic_sha256": receipt["deterministic_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
