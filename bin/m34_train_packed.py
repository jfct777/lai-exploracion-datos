#!/usr/bin/env python3
"""Train one declared M34 task from paired, already materialized NPZ shards.

Only FIT and VALID truth are opened here.  TEST is intentionally not part of
the manifest schema, so model selection cannot inspect it accidentally.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m34_adaptive_sweep as sweep
import m34_models as models


TASK_MEMBERS = {
    "family", "config_id", "seed", "rotation", "arm", "radius_cM",
    "sweep_stage", "maximum_updates", "learning_rate", "weight_decay",
}
PACKED_MEMBERS = {
    "sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref",
    "marker_alt", "marker_cM", "radius_cM", "rare_tokens", "rare_mask",
    "rare_locus_index", "row_ptr", "row_sample_index", "row_marker_index",
    "F0",
}
TRUTH_MEMBERS = {"sample_key_sha256", "marker_pos", "labels"}


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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(set(payload) == {"schema_version", "ancestry_names", "haplotypes", "splits"},
            "training manifest members differ")
    names = tuple(str(value) for value in payload["ancestry_names"])
    require(len(names) >= 2 and len(set(names)) == len(names),
            "ancestry names must be unique")
    require(type(payload["haplotypes"]) is int and payload["haplotypes"] >= 1,
            "haplotype count differs")
    require(set(payload["splits"]) == {"FIT", "VALID"},
            "manifest must contain FIT and VALID only")
    base = path.parent.resolve()
    for split in ("FIT", "VALID"):
        rows = payload["splits"][split]
        require(isinstance(rows, list) and rows, f"{split} shard list is empty")
        normalized = []
        for row in rows:
            require(set(row) == {"packed", "truth"}, f"{split} shard members differ")
            packed, truth = _resolved(base, row["packed"]), _resolved(base, row["truth"])
            require(packed.is_file() and truth.is_file(), f"{split} shard input is missing")
            normalized.append({"packed": packed, "truth": truth})
        payload["splits"][split] = normalized
    payload["ancestry_names"] = names
    return payload


def load_task(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    task = json.loads(path.read_text(encoding="utf-8"))
    require(set(task) == TASK_MEMBERS, "task members differ from the declared sweep task")
    lookup = {
        (family, row["id"]): row
        for family, specification in contract["families"].items()
        for row in specification["configs"]
    }
    require((task["family"], task["config_id"]) in lookup, "task model is undeclared")
    require(task["arm"] in contract["scope"]["arms"], "task arm differs")
    require(task["sweep_stage"] in contract["stages"], "task stage differs")
    stage = contract["stages"][task["sweep_stage"]]
    require(int(task["maximum_updates"]) == int(stage["maximum_updates"]),
            "task update budget differs")
    allowed_seeds = stage.get("seeds", [stage.get("seed")])
    allowed_rotations = stage.get("rotations", [stage.get("rotation")])
    if task["sweep_stage"] == "radius_sensitivity":
        allowed_radii = stage["radii_cM"]
    elif task["sweep_stage"] == "finalists":
        allowed_radii = contract["stages"]["radius_sensitivity"]["radii_cM"]
    else:
        allowed_radii = [stage["radius_cM"]]
    require(int(task["seed"]) in allowed_seeds, "task seed differs from its sweep stage")
    require(task["rotation"] in allowed_rotations,
            "task rotation differs from its sweep stage")
    require(float(task["radius_cM"]) in allowed_radii,
            "task radius differs from its sweep stage")
    config = lookup[(task["family"], task["config_id"])]
    expected_lr = float(contract["training"]["learning_rate"])
    expected_wd = float(config.get("training_overrides", {}).get(
        "weight_decay", contract["training"]["weight_decay"]))
    require(float(task["learning_rate"]) == expected_lr,
            "task learning rate differs from the frozen contract")
    require(float(task["weight_decay"]) == expected_wd,
            "task weight decay differs from the frozen contract")
    return task


def training_parameter(
    contract: Mapping[str, Any], task: Mapping[str, Any], name: str,
) -> Any:
    """Return a preregistered stage override or the historical global value."""
    stage = contract["stages"][task["sweep_stage"]]
    return stage.get("training_overrides", {}).get(name, contract["training"][name])


def model_spec(contract: Mapping[str, Any], task: Mapping[str, Any], channels: int,
               ancestries: int) -> models.ModelSpec:
    config = next(row for row in contract["families"][task["family"]]["configs"]
                  if row["id"] == task["config_id"])
    values = dict(config["model_spec"])
    values["dilations"] = tuple(values["dilations"])
    return models.ModelSpec(
        family=task["family"], channels=channels, ancestries=ancestries,
        zero_init_head=False, seed=int(task["seed"]), **values,
    )


def load_pair(packed_path: Path, truth_path: Path, ancestry_count: int,
              haplotypes: int, expected_arm: str, expected_radius: float
              ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(packed_path, allow_pickle=False) as archive:
        require(set(archive.files) == PACKED_MEMBERS, "packed shard members differ")
        packed = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(truth_path, allow_pickle=False) as archive:
        require(set(archive.files) == TRUTH_MEMBERS, "truth shard members differ")
        truth = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    require(np.array_equal(packed["sample_key_sha256"], truth["sample_key_sha256"]),
            "packed/truth sample axes differ")
    require(np.array_equal(packed["marker_pos"], truth["marker_pos"]),
            "packed/truth marker axes differ")
    samples, markers = len(packed["sample_key_sha256"]), len(packed["marker_pos"])
    require(packed["F0"].shape == (samples, haplotypes, markers, ancestry_count),
            "packed F0 dimensions differ")
    require(truth["labels"].shape == (samples, haplotypes, markers),
            "truth label dimensions differ")
    require(np.issubdtype(truth["labels"].dtype, np.integer) and
            np.all((truth["labels"] >= 0) & (truth["labels"] < ancestry_count)),
            "truth labels differ")
    rows = samples * markers
    require(len(packed["row_ptr"]) == rows + 1 and
            np.all(np.diff(packed["row_ptr"].astype(np.int64)) >= 0),
            "packed row pointers differ")
    require(int(packed["row_ptr"][-1]) == len(packed["rare_tokens"]),
            "packed token accounting differs")
    expected_sample = np.repeat(np.arange(samples, dtype=np.uint32), markers)
    expected_marker = np.tile(np.arange(markers, dtype=np.uint32), samples)
    require(np.array_equal(packed["row_sample_index"], expected_sample) and
            np.array_equal(packed["row_marker_index"], expected_marker),
            "packed row order differs")
    require(packed["rare_tokens"].ndim == 2 and
            packed["rare_tokens"].shape[1] == 4 + 3 * ancestry_count,
            "packed token channels differ")
    require(np.all(packed["rare_mask"] == 1), "packed token mask differs")
    require(np.isclose(float(packed["radius_cM"][0]), expected_radius, rtol=0, atol=1e-7),
            "packed radius differs from task")
    if expected_arm == "RD":
        rare_columns = (0, *(index for ancestry in range(ancestry_count)
                             for index in (2 + 3 * ancestry, 3 + 3 * ancestry)))
        require(np.all(packed["rare_tokens"][:, rare_columns] == 0),
                "RD shard contains enabled rare-value channels")
    return packed, truth["labels"].astype(np.int64, copy=False)


def plan_row_batches(row_ptr: np.ndarray, maximum_rows: int,
                     maximum_tokens: int) -> list[tuple[int, int]]:
    require(maximum_rows > 0 and maximum_tokens > 0, "batch limits must be positive")
    pointers = np.asarray(row_ptr, dtype=np.int64)
    batches: list[tuple[int, int]] = []
    start = 0
    while start < len(pointers) - 1:
        end = start
        while end < len(pointers) - 1 and end - start < maximum_rows:
            candidate = end + 1
            token_count = int(pointers[candidate] - pointers[start])
            if candidate > start + 1 and token_count > maximum_tokens:
                break
            require(token_count <= maximum_tokens or candidate == start + 1,
                    "one packed row exceeds the token batch budget")
            end = candidate
        require(end > start, "empty row batch")
        batches.append((start, end))
        start = end
    return batches


def dense_batch(packed: Mapping[str, np.ndarray], labels: np.ndarray,
                transitions: np.ndarray, start: int, end: int,
                beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]:
    pointers = packed["row_ptr"].astype(np.int64, copy=False)
    lengths = np.diff(pointers)[start:end]
    width = max(1, int(lengths.max(initial=0)))
    channels = packed["rare_tokens"].shape[1]
    token = np.zeros((end - start, width, channels), dtype=np.float32)
    mask = np.zeros((end - start, width), dtype=bool)
    for local, row in enumerate(range(start, end)):
        left, right = int(pointers[row]), int(pointers[row + 1])
        if right > left:
            token[local, :right-left] = packed["rare_tokens"][left:right]
            mask[local, :right-left] = True
    samples = packed["row_sample_index"][start:end].astype(np.int64, copy=False)
    markers = packed["row_marker_index"][start:end].astype(np.int64, copy=False)
    baseline = np.transpose(packed["F0"], (0, 2, 1, 3))[samples, markers]
    target = np.transpose(labels, (0, 2, 1))[samples, markers]
    transition = np.transpose(transitions, (0, 2, 1))[samples, markers]
    weights = 1.0 + beta * transition.astype(np.float32)
    return (torch.from_numpy(token), torch.from_numpy(mask),
            torch.from_numpy(np.ascontiguousarray(baseline)),
            torch.from_numpy(np.ascontiguousarray(target)),
            torch.from_numpy(np.ascontiguousarray(weights)))


def transition_mask(labels: np.ndarray) -> np.ndarray:
    require(labels.ndim == 3, "labels must have sample, haplotype and marker axes")
    result = np.zeros(labels.shape, dtype=np.uint8)
    result[:, :, 1:] = labels[:, :, 1:] != labels[:, :, :-1]
    return result


def attach_global_transitions(
    pairs: Sequence[tuple[dict[str, np.ndarray], np.ndarray]],
) -> list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
    """Derive transitions after joining marker tiles, without merging token payloads."""
    cells: dict[bytes, dict[int, np.ndarray]] = {}
    for packed, labels in pairs:
        for sample_index, key_value in enumerate(packed["sample_key_sha256"]):
            sample = cells.setdefault(bytes(key_value), {})
            for marker_index, position in enumerate(packed["marker_pos"].tolist()):
                require(position not in sample, "duplicate sample/marker cell across shards")
                sample[position] = labels[sample_index, :, marker_index].copy()
    global_transition: dict[tuple[bytes, int], np.ndarray] = {}
    for sample_key, marker_rows in cells.items():
        previous: np.ndarray | None = None
        for position in sorted(marker_rows):
            labels = marker_rows[position]
            global_transition[(sample_key, position)] = (
                np.zeros(labels.shape, dtype=np.uint8) if previous is None
                else (labels != previous).astype(np.uint8)
            )
            previous = labels
    result = []
    for packed, labels in pairs:
        transitions = np.empty(labels.shape, dtype=np.uint8)
        for sample_index, key_value in enumerate(packed["sample_key_sha256"]):
            key = bytes(key_value)
            for marker_index, position in enumerate(packed["marker_pos"].tolist()):
                transitions[sample_index, :, marker_index] = global_transition[(key, position)]
        result.append((packed, labels, transitions))
    return result


def boundary_parameters(label_sets: Sequence[np.ndarray], target_share: float
                        ) -> dict[str, float | int]:
    """Derive the PRE4B transition multiplier from FIT truth only."""
    require(0.0 < target_share < 1.0, "boundary target share must be in (0, 1)")
    transitions = [transition_mask(labels) for labels in label_sets]
    possible = sum(value.size for value in transitions)
    observed = sum(int(value.sum()) for value in transitions)
    require(possible > 0 and 0 < observed < possible,
            "FIT must contain both transition and non-transition ancestry cells")
    p = observed / possible
    multiplier = target_share * (1.0 - p) / (p * (1.0 - target_share))
    beta = multiplier - 1.0
    require(math.isfinite(beta) and beta > -1.0, "derived boundary beta differs")
    achieved = (p * multiplier) / ((1.0 - p) + p * multiplier)
    require(abs(achieved - target_share) <= 1e-12,
            "derived transition weighting does not reach its target share")
    return {
        "transition_cells": observed, "possible_cells": possible,
        "transition_prevalence": p, "transition_multiplier": multiplier,
        "beta": beta, "target_weight_share": target_share,
    }


def schedule(base_lr: float, update: int, maximum: int, warmup: int) -> float:
    if warmup > 0 and update <= warmup:
        return base_lr * update / warmup
    progress = (update - warmup) / max(1, maximum - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def validation_loss(model: torch.nn.Module,
                    pairs: Sequence[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
                    maximum_rows: int, maximum_tokens: int, beta: float) -> float:
    numerator = 0.0
    denominator = 0
    model.eval()
    with torch.inference_mode():
        for packed, labels, transitions in pairs:
            for start, end in plan_row_batches(packed["row_ptr"], maximum_rows, maximum_tokens):
                tokens, mask, baseline, target, weights = dense_batch(
                    packed, labels, transitions, start, end, beta)
                probabilities = model(tokens, mask, baseline)
                chosen = probabilities.gather(2, target.unsqueeze(-1)).squeeze(-1)
                numerator += float((-torch.log(chosen.clamp_min(1e-8)) * weights).sum())
                denominator += float(weights.sum())
    require(denominator > 0, "VALID has no labels")
    return numerator / denominator


def predict_validation(model: torch.nn.Module,
                       pairs: Sequence[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
                       ancestry_names: Sequence[str], maximum_rows: int,
                       maximum_tokens: int) -> dict[str, np.ndarray]:
    sample_rows: dict[bytes, dict[int, tuple[float, np.ndarray]]] = {}
    model.eval()
    with torch.inference_mode():
        for packed, labels, transitions in pairs:
            shard = np.empty((*labels.shape[:1], labels.shape[1], labels.shape[2],
                              len(ancestry_names)), dtype=np.float32)
            flat = np.empty((labels.shape[0] * labels.shape[2], labels.shape[1],
                             len(ancestry_names)), dtype=np.float32)
            for start, end in plan_row_batches(packed["row_ptr"], maximum_rows, maximum_tokens):
                tokens, mask, baseline, _target, _weights = dense_batch(
                    packed, labels, transitions, start, end, 0.0)
                flat[start:end] = model(tokens, mask, baseline).cpu().numpy()
            shard[:] = np.transpose(flat.reshape(labels.shape[0], labels.shape[2],
                                                  labels.shape[1], len(ancestry_names)),
                                    (0, 2, 1, 3))
            for sample_index, key_value in enumerate(packed["sample_key_sha256"]):
                key = bytes(key_value)
                row = sample_rows.setdefault(key, {})
                for marker_index, position in enumerate(packed["marker_pos"].tolist()):
                    require(position not in row, "duplicate VALID sample/marker cell")
                    row[position] = (float(packed["marker_cM"][marker_index]),
                                     shard[sample_index, :, marker_index, :].copy())
    keys = sorted(sample_rows)
    require(keys, "VALID prediction is empty")
    positions = sorted(next(iter(sample_rows.values())))
    require(all(sorted(row) == positions for row in sample_rows.values()),
            "VALID shards do not form a complete sample by marker rectangle")
    marker_cm = np.asarray([sample_rows[keys[0]][position][0] for position in positions],
                           dtype=np.float64)
    require(all(np.array_equal(marker_cm, np.asarray([row[position][0] for position in positions]))
                for row in sample_rows.values()), "VALID genetic maps differ across samples")
    probabilities = np.asarray([
        np.stack([sample_rows[key][position][1] for position in positions], axis=1)
        for key in keys
    ], dtype=np.float32)
    return {
        "sample_key_sha256": np.asarray(keys, dtype="|S64"),
        "marker_pos": np.asarray(positions, dtype=np.int64),
        "marker_cM": marker_cm,
        "ancestry_names": np.asarray(tuple(ancestry_names), dtype="|S32"),
        "probabilities": probabilities,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(not args.outdir.exists(), "refusing to overwrite the output directory")
    contract = sweep.validate_contract(sweep.strict_json(args.contract))
    manifest = load_manifest(args.manifest)
    task = load_task(args.task, contract)
    torch.set_num_threads(args.threads)
    random.seed(task["seed"]); np.random.seed(task["seed"]); torch.manual_seed(task["seed"])
    torch.use_deterministic_algorithms(bool(contract["training"]["deterministic_algorithms"]))
    boundary_contract = contract.get("boundary_loss", {})
    require(boundary_contract.get("provenance") == "M33_PRE4B" and
            boundary_contract.get("formula") ==
            "beta=q*(1-p)/(p*(1-q))-1_fit_truth_only",
            "boundary-loss contract differs from M33 PRE4B")
    target_share = float(boundary_contract.get("target_transition_weight_share", 0.0))
    raw_pairs: dict[str, list[tuple[dict[str, np.ndarray], np.ndarray]]] = {
        "FIT": [], "VALID": []}
    hashes: dict[str, dict[str, str]] = {}
    for split in raw_pairs:
        for index, row in enumerate(manifest["splits"][split]):
            packed, labels = load_pair(
                row["packed"], row["truth"], len(manifest["ancestry_names"]),
                manifest["haplotypes"], task["arm"], float(task["radius_cM"]))
            raw_pairs[split].append((packed, labels))
            hashes[f"{split}.{index}"] = {
                "packed": sha256_file(row["packed"]), "truth": sha256_file(row["truth"])}
    pairs = {split: attach_global_transitions(values)
             for split, values in raw_pairs.items()}
    split_keys: dict[str, set[bytes]] = {}
    for split, split_pairs in pairs.items():
        ordered = [bytes(value) for packed, _labels, _transitions in split_pairs
                   for value in packed["sample_key_sha256"]]
        split_keys[split] = set(ordered)
    require(split_keys["FIT"].isdisjoint(split_keys["VALID"]),
            "FIT and VALID sample axes overlap")
    boundary = boundary_parameters([labels for _packed, labels, _transitions in pairs["FIT"]],
                                   target_share)
    beta = float(boundary["beta"])
    channels = pairs["FIT"][0][0]["rare_tokens"].shape[1]
    require(all(pair[0]["rare_tokens"].shape[1] == channels
                for split in pairs.values() for pair in split), "shard channel counts differ")
    specification = model_spec(contract, task, channels, len(manifest["ancestry_names"]))
    model = models.build_model(specification)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(task["learning_rate"]),
                                  weight_decay=float(task["weight_decay"]))
    fit_batches = [(shard_index, start, end)
                   for shard_index, (packed, _labels, _transitions) in enumerate(pairs["FIT"])
                   for start, end in plan_row_batches(packed["row_ptr"],
                                                      args.maximum_rows_per_batch,
                                                      args.maximum_tokens_per_batch)]
    require(fit_batches, "FIT has no batches")
    best_loss, best_update, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())
    update, epoch = 0, 0
    while update < int(task["maximum_updates"]):
        order = list(fit_batches)
        random.Random(int(task["seed"]) + epoch * 1009).shuffle(order)
        for shard_index, start, end in order:
            model.train(); optimizer.zero_grad(set_to_none=True)
            packed, labels, transitions = pairs["FIT"][shard_index]
            tokens, mask, baseline, target, weights = dense_batch(
                packed, labels, transitions, start, end, beta)
            probabilities = model(tokens, mask, baseline)
            chosen = probabilities.gather(2, target.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            loss = (-torch.log(chosen) * weights).sum() / weights.sum()
            require(torch.isfinite(loss).item(), "FIT loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(contract["training"]["gradient_clip_norm"]))
            update += 1
            lr = schedule(float(task["learning_rate"]), update,
                          int(task["maximum_updates"]),
                          min(int(training_parameter(
                              contract, task, "warmup_updates")),
                              int(task["maximum_updates"])))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if update % args.validation_every == 0 or update == int(task["maximum_updates"]):
                value = validation_loss(model, pairs["VALID"], args.maximum_rows_per_batch,
                                        args.maximum_tokens_per_batch, beta)
                if value < best_loss:
                    best_loss, best_update = value, update
                    best_state = copy.deepcopy(model.state_dict())
            if update >= int(task["maximum_updates"]):
                break
        epoch += 1
    model.load_state_dict(best_state)
    prediction = predict_validation(model, pairs["VALID"], manifest["ancestry_names"],
                                    args.maximum_rows_per_batch, args.maximum_tokens_per_batch)
    args.outdir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.outdir / "model.pt"
    torch.save({"model_state_dict": best_state, "model_spec": asdict(specification),
                "ancestry_names": tuple(manifest["ancestry_names"]), "task": task}, checkpoint)
    prediction_path = args.outdir / "valid.prediction.npz"
    np.savez_compressed(prediction_path, **prediction)
    receipt = {
        "schema_version": "1.0.0", "stage": "M34_EXPLORATORY_TRAIN_PACKED",
        "status": "PASS_TRAINED_VALID_ONLY", "claim_level": "exploratory",
        "task": task, "model_spec": asdict(specification),
        "boundary_loss": boundary,
        "parameter_count": models.parameter_count(model), "updates_executed": update,
        "selected_update": best_update, "selected_valid_loss": best_loss,
        "fit_shard_count": len(pairs["FIT"]), "valid_shard_count": len(pairs["VALID"]),
        "input_sha256": hashes, "contract_sha256": sha256_file(args.contract),
        "manifest_sha256": sha256_file(args.manifest), "task_sha256": sha256_file(args.task),
        "checkpoint_sha256": sha256_file(checkpoint),
        "valid_prediction_sha256": sha256_file(prediction_path),
        "test_opened": False, "wall_seconds": time.monotonic() - started,
    }
    receipt["semantic_sha256"] = canonical_sha256({key: value for key, value in receipt.items()
                                                    if key != "wall_seconds"})
    (args.outdir / "train.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--maximum-rows-per-batch", type=int, default=64)
    parser.add_argument("--maximum-tokens-per-batch", type=int, default=1_000_000)
    parser.add_argument("--validation-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "selected_update": result["selected_update"],
                      "valid_loss": result["selected_valid_loss"]}, sort_keys=True))
