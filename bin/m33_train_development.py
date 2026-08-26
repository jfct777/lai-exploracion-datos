#!/usr/bin/env python3
"""Train one frozen M33 DEVELOPMENT arm and seal truth-blind SCORE predictions."""

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m28d_b0_scorer as truth_io
import m30_flare_scorer as flare_io
import m33_materialize as materialize
import m33_safe_bridge_core as bridge_core
import m33_t0a_models as models
from m33_score_development import load_map


ROOTS = (386357765, 2024931463, 1324432253)
RD_ZERO_CHANNELS = (0, 2, 3, 5, 6, 8, 9)
PLAN_CACHE: dict[tuple[int, float, int], list[dict[str, int | bool]]] = {}
CHANNEL_CACHE: dict[tuple[int, int, int, int, str], materialize.PreparedChannelBatch] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RootData:
    seed: int
    selected: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    reference: dict[str, np.ndarray]
    f0: dict[str, np.ndarray]
    marker_cm: np.ndarray
    intervals: dict[str, np.ndarray]
    interval_auth: materialize.ValidatedIntervalTable
    labels: np.ndarray | None
    samples: tuple[str, ...]


def load_root(runtime: Path, seed: int, with_truth: bool) -> RootData:
    root_dir = runtime / "bridge" / f"root-{seed}"
    selected = materialize.load_productive_npz(root_dir / "selected_loci_incremental.npz", "selected")
    target = materialize.load_productive_npz(root_dir / "target_rare_diploid_incremental.npz", "target")
    reference = materialize.load_productive_npz(root_dir / "reference_rare_summary_incremental.npz", "reference")
    f0 = materialize.load_productive_npz(root_dir / "flare_f0_sanitized.npz", "f0")
    with np.load(root_dir / "marker_cM.npz", allow_pickle=False) as archive:
        marker_cm = np.ascontiguousarray(archive["marker_cM"])
    materialize.validate_inputs(selected, target, reference, f0, marker_cm)
    interval_path = runtime / "materialized" / f"root-{seed}" / "context_intervals_all_radii.npz"
    with np.load(interval_path, allow_pickle=False) as archive:
        intervals = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    interval_auth = materialize.authenticate_interval_table(
        intervals, selected["cM"], marker_cm,
    )
    target_grid = flare_io.load_target_grid(runtime / "indexed" / f"root-{seed}" / "target.vcf.gz")
    expected_keys = np.asarray([bridge_core.sample_key(sample) for sample in target_grid.samples],
                               dtype="|S64")
    require(np.array_equal(expected_keys, target["sample_key_sha256"]), "private/public sample binding differs")
    labels = None
    if with_truth:
        truth = truth_io.load_truth(
            runtime / "generation" / f"root-{seed}" / "m28_lai_truth.private.tsv.gz",
            target_grid.samples, "22", int(f0["marker_pos"][0]), int(f0["marker_pos"][-1]) + 1,
        )
        ancestry_index = {value: index for index, value in enumerate(("AFR", "EUR", "ASIA"))}
        labels = np.empty((len(target_grid.samples), 2, len(marker_cm)), dtype=np.int64)
        positions = f0["marker_pos"]
        for sample_index, sample in enumerate(target_grid.samples):
            for hap in (0, 1):
                segments = truth[sample][hap]
                ends = np.asarray([segment.end for segment in segments], dtype=np.int64)
                indexes = np.searchsorted(ends, positions, side="right")
                require(np.all(indexes < len(segments)), "truth does not cover marker axis")
                labels[sample_index, hap] = [ancestry_index[segments[index].ancestry] for index in indexes]
    return RootData(seed, selected, target, reference, f0, marker_cm, intervals,
                    interval_auth, labels, target_grid.samples)


def split_people(root: RootData, rotation: str) -> tuple[np.ndarray, np.ndarray]:
    keys = []
    for index, sample_key in enumerate(root.target["sample_key_sha256"]):
        payload = f"M33|{rotation}|{root.seed}|".encode() + bytes(sample_key)
        keys.append((hashlib.sha256(payload).digest(), index))
    ordered = np.asarray([index for _digest, index in sorted(keys)], dtype=np.int64)
    return ordered[:24], ordered[24:]


def subset(root: RootData, indexes: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    target = {name: np.ascontiguousarray(value[indexes] if name in {
        "sample_key_sha256", "minor_dosage", "observed_mask"
    } else value) for name, value in root.target.items()}
    f0 = {name: np.ascontiguousarray(value[indexes] if name in {"sample_key_sha256", "F0"}
                                     else value) for name, value in root.f0.items()}
    require(root.labels is not None, "FIT truth is unavailable")
    return target, f0, np.ascontiguousarray(root.labels[indexes])


def learning_rate(family: str) -> float:
    return 0.001 if family == "local_linear" else 0.0003


def schedule(base_lr: float, update: int, maximum: int) -> float:
    if update <= 100:
        return base_lr * update / 100.0
    fraction = (update - 100) / max(1, maximum - 100)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, fraction)))


def dense_rows(packed: dict[str, np.ndarray], start: int, end: int):
    row_ptr = packed["row_ptr"].astype("<i8")
    lengths = np.diff(row_ptr)[start:end]
    width = max(1, int(lengths.max(initial=0)))
    tokens = np.zeros((end - start, width, 13), dtype="<f4")
    mask = np.zeros((end - start, width), dtype="<f4")
    for local, row in enumerate(range(start, end)):
        left, right = int(row_ptr[row]), int(row_ptr[row + 1])
        if right > left:
            tokens[local, :right-left] = packed["rare_tokens"][left:right]
            mask[local, :right-left] = 1.0
    f0 = np.transpose(packed["F0"], (0, 2, 1, 3)).reshape(-1, 2, 3)[start:end]
    return torch.from_numpy(tokens), torch.from_numpy(mask), torch.from_numpy(np.ascontiguousarray(f0))


def packed_chunks(root: RootData, target: dict[str, np.ndarray], f0: dict[str, np.ndarray],
                  reference: dict[str, np.ndarray], normalization: dict[str, int], rotation: str,
                  arm: str,
                  normalization_sha: str, radius: float, person_start: int, person_end: int,
                  marker_start: int, marker_end: int):
    person_count = person_end - person_start
    plan_key = (id(root.intervals), radius, person_count)
    if plan_key not in PLAN_CACHE:
        PLAN_CACHE[plan_key] = materialize.plan_lazy_marker_chunks(
            root.intervals, root.selected["cM"], root.marker_cm, radius, person_count,
        )
    plan = PLAN_CACHE[plan_key]
    channel_key = (id(target), id(reference), person_start, person_end, rotation)
    if channel_key not in CHANNEL_CACHE:
        CHANNEL_CACHE[channel_key] = materialize.prepare_person_batch_channels(
            target, reference, normalization, person_start, person_end,
            root_seed=root.seed, rotation_id=rotation,
            fit_normalization_manifest_sha256=normalization_sha,
        )
    prepared = CHANNEL_CACHE[channel_key]
    for row in plan:
        left, right = int(row["marker_start"]), int(row["marker_end_exclusive"])
        if right <= marker_start or left >= marker_end:
            continue
        left, right = max(left, marker_start), min(right, marker_end)
        packed = materialize.build_lazy_packed_shard(
            root.selected, target, reference, f0, root.marker_cm, root.intervals,
            normalization, radius, person_start, person_end, left, right,
            prepared_channels=prepared, expected_root_seed=root.seed,
            expected_rotation_id=rotation,
            expected_fit_normalization_manifest_sha256=normalization_sha,
            inputs_already_validated=True, interval_validation=root.interval_auth,
        )
        if arm == "RD":
            packed["rare_tokens"][:, RD_ZERO_CHANNELS] = 0.0
        yield packed, left, right


def logical_loss(model: torch.nn.Module, root: RootData, target: dict[str, np.ndarray],
                 f0: dict[str, np.ndarray], reference: dict[str, np.ndarray], labels: np.ndarray,
                 normalization: dict[str, int], rotation: str, normalization_sha: str,
                 arm: str, radius: float, beta: float, person_start: int, person_end: int,
                 marker_start: int, marker_end: int, backward: bool) -> tuple[float, float]:
    label_block = labels[person_start:person_end, :, marker_start:marker_end]
    transition = np.zeros_like(label_block, dtype=np.float32)
    if marker_start == 0:
        transition[:, :, 1:] = label_block[:, :, 1:] != label_block[:, :, :-1]
    else:
        previous = labels[person_start:person_end, :, marker_start-1:marker_end-1]
        transition = (label_block != previous).astype(np.float32)
    weights = 1.0 + beta * transition
    denominator = float(weights.sum())
    total = 0.0
    for packed, left, right in packed_chunks(
        root, target, f0, reference, normalization, rotation, arm, normalization_sha,
        radius, person_start, person_end, marker_start, marker_end,
    ):
        local_left, local_right = left - marker_start, right - marker_start
        local_labels = np.transpose(label_block[:, :, local_left:local_right], (0, 2, 1)).reshape(-1, 2)
        local_weights = np.transpose(weights[:, :, local_left:local_right], (0, 2, 1)).reshape(-1, 2)
        row_ptr = packed["row_ptr"].astype("<i8")
        for row_start, row_end in __import__("m33_t0a_forward").padded_batches(row_ptr):
            tokens, mask, baseline = dense_rows(packed, row_start, row_end)
            probabilities, delta, _ = model.forward_with_features(tokens, mask, baseline)
            logits = torch.log(baseline.clamp_min(1e-7)) + delta
            labels_tensor = torch.from_numpy(np.ascontiguousarray(local_labels[row_start:row_end]))
            weights_tensor = torch.from_numpy(np.ascontiguousarray(local_weights[row_start:row_end]))
            chosen = torch.log_softmax(logits, dim=2).gather(2, labels_tensor.unsqueeze(2)).squeeze(2)
            numerator = (-chosen * weights_tensor).sum()
            total += float(numerator.detach())
            if backward:
                (numerator / denominator).backward()
    return total, denominator


def predict_root(model: torch.nn.Module, root: RootData, target: dict[str, np.ndarray],
                 f0: dict[str, np.ndarray], reference: dict[str, np.ndarray],
                 normalization: dict[str, int], rotation: str, normalization_sha: str,
                 arm: str, radius: float) -> np.ndarray:
    sample_count = len(target["sample_key_sha256"])
    marker_count = len(root.marker_cm)
    output = np.empty((sample_count, 2, marker_count, 3), dtype="<f4")
    model.eval()
    with torch.inference_mode():
        for person_start in range(0, sample_count, 8):
            person_end = min(person_start + 8, sample_count)
            for marker_start in range(0, marker_count, 256):
                marker_end = min(marker_start + 256, marker_count)
                for packed, left, right in packed_chunks(
                    root, target, f0, reference, normalization, rotation, arm,
                    normalization_sha, radius, person_start, person_end,
                    marker_start, marker_end,
                ):
                    flat = np.empty(((person_end-person_start) * (right-left), 2, 3), dtype="<f4")
                    row_ptr = packed["row_ptr"].astype("<i8")
                    for row_start, row_end in __import__("m33_t0a_forward").padded_batches(row_ptr):
                        tokens, mask, baseline = dense_rows(packed, row_start, row_end)
                        probabilities = model(tokens, mask, baseline)
                        flat[row_start:row_end] = probabilities.detach().cpu().numpy()
                    cube = flat.reshape(person_end-person_start, right-left, 2, 3)
                    output[person_start:person_end, :, left:right, :] = np.transpose(cube, (0, 2, 1, 3))
    require(np.isfinite(output).all() and np.all(output >= 0) and
            np.max(np.abs(output.sum(axis=3)-1.0)) <= 5e-6,
            "sealed prediction probabilities differ")
    return output


def write_prediction(path: Path, root: RootData, target: dict[str, np.ndarray],
                     probabilities: np.ndarray) -> None:
    np.savez_compressed(
        path, sample_key_sha256=np.ascontiguousarray(target["sample_key_sha256"]),
        marker_pos=np.ascontiguousarray(root.f0["marker_pos"]),
        marker_cM=np.ascontiguousarray(root.marker_cm),
        probabilities=np.ascontiguousarray(probabilities, dtype="<f4"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--rotation", choices=("R0", "R1", "R2"), required=True)
    parser.add_argument("--family", choices=models.FAMILIES, required=True)
    parser.add_argument("--radius", type=float, choices=materialize.RADII, required=True)
    parser.add_argument("--beta", type=float, required=True,
                        help="nonnegative additive weight on a true transition")
    parser.add_argument("--seed", type=int, choices=(1103, 2207, 3301), required=True)
    parser.add_argument("--arm", choices=("RD", "RE"), required=True)
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--validation-start", type=int, default=200)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    os.environ.setdefault("USER", "m33")
    os.environ.setdefault("LOGNAME", "m33")
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    require(math.isfinite(args.beta) and args.beta >= 0.0, "boundary weight must be nonnegative")
    spec = next(row for row in pre4["root_registry"]["development_rotations"]
                if row["rotation"] == args.rotation)
    fit_seeds, score_seed = spec["fit_roots"], spec["score_only_root"]
    fit_roots = {seed: load_root(args.runtime, seed, True) for seed in fit_seeds}
    score_root = None if args.skip_prediction else load_root(args.runtime, score_seed, False)
    norm_path = args.runtime / "materialized" / args.rotation / "fit_callable_normalization_manifest.json"
    norm_manifest = json.loads(norm_path.read_text(encoding="utf-8"))
    normalization = norm_manifest["max_callable_an"]
    normalization_sha = sha256_file(norm_path)
    model = models.build_model(args.family)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(args.family), weight_decay=1e-4)
    train_sets = {}
    val_sets = {}
    for seed, root in fit_roots.items():
        train_index, val_index = split_people(root, args.rotation)
        train_sets[seed] = subset(root, train_index)
        val_sets[seed] = subset(root, val_index)
    best_loss, best_update, best_state = float("inf"), None, None
    update = 0
    epoch = 0
    while update < args.max_updates:
        tasks = []
        for seed in fit_seeds:
            target, f0, labels = train_sets[seed]
            order = np.arange(len(labels)); np.random.default_rng(args.seed + epoch * 1009 + seed).shuffle(order)
            target = {name: np.ascontiguousarray(value[order] if name in {"sample_key_sha256", "minor_dosage", "observed_mask"} else value)
                      for name, value in target.items()}
            f0 = {name: np.ascontiguousarray(value[order] if name in {"sample_key_sha256", "F0"} else value)
                  for name, value in f0.items()}
            labels = np.ascontiguousarray(labels[order])
            for person_start in range(0, len(labels), 8):
                for marker_start in range(0, len(fit_roots[seed].marker_cm), 256):
                    tasks.append((seed, target, f0, labels, person_start, min(person_start+8, len(labels)),
                                  marker_start, min(marker_start+256, len(fit_roots[seed].marker_cm))))
        random.Random(args.seed + epoch).shuffle(tasks)
        for seed, target, f0, labels, person_start, person_end, marker_start, marker_end in tasks:
            optimizer.zero_grad(set_to_none=True)
            logical_loss(model, fit_roots[seed], target, f0, fit_roots[seed].reference, labels,
                         normalization, args.rotation, normalization_sha, args.arm, args.radius, args.beta,
                         person_start, person_end, marker_start, marker_end, True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            update += 1
            lr = schedule(learning_rate(args.family), update, args.max_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if update % 50 == 0 or update == args.max_updates:
                print(json.dumps({"stage": "TRAIN_PROGRESS", "update": update,
                                  "maximum": args.max_updates, "family": args.family,
                                  "rotation": args.rotation, "arm": args.arm}), flush=True)
            if update >= args.validation_start and update % args.validation_every == 0:
                model.eval(); numerator = denominator = 0.0
                with torch.no_grad():
                    for val_seed in fit_seeds:
                        val_target, val_f0, val_labels = val_sets[val_seed]
                        for person_left in range(0, len(val_labels), 8):
                            for marker_left in range(0, len(fit_roots[val_seed].marker_cm), 256):
                                n, d = logical_loss(
                                    model, fit_roots[val_seed], val_target, val_f0,
                                    fit_roots[val_seed].reference, val_labels, normalization,
                                    args.rotation, normalization_sha, args.arm, args.radius, args.beta,
                                    person_left, min(person_left+8, len(val_labels)), marker_left,
                                    min(marker_left+256, len(fit_roots[val_seed].marker_cm)), False,
                                )
                                numerator += n; denominator += d
                value = numerator / denominator
                if value < best_loss - 1e-12:
                    best_loss, best_update = value, update
                    best_state = copy.deepcopy(model.state_dict())
                print(json.dumps({"stage": "INNER_VALIDATION", "update": update,
                                  "loss": value, "best_update": best_update,
                                  "best_loss": best_loss}), flush=True)
                model.train()
            if update >= args.max_updates:
                break
        epoch += 1
    if best_state is None:
        best_update, best_loss, best_state = update, float("nan"), copy.deepcopy(model.state_dict())
    args.outdir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.outdir / "model.pt"
    torch.save(best_state, checkpoint)
    prediction_hashes = {}
    if not args.skip_prediction:
        require(score_root is not None, "SCORE root was not loaded")
        model.load_state_dict(best_state)
        score_probabilities = predict_root(
            model, score_root, score_root.target, score_root.f0, score_root.reference,
            normalization, args.rotation, normalization_sha, args.arm, args.radius,
        )
        score_path = args.outdir / "score.prediction.npz"
        write_prediction(score_path, score_root, score_root.target, score_probabilities)
        prediction_hashes["score"] = sha256_file(score_path)
        for fit_seed in fit_seeds:
            val_target, val_f0, _val_labels = val_sets[fit_seed]
            val_probabilities = predict_root(
                model, fit_roots[fit_seed], val_target, val_f0, fit_roots[fit_seed].reference,
                normalization, args.rotation, normalization_sha, args.arm, args.radius,
            )
            val_path = args.outdir / f"fit-{fit_seed}.inner-val.prediction.npz"
            write_prediction(val_path, fit_roots[fit_seed], val_target, val_probabilities)
            prediction_hashes[f"fit-{fit_seed}.inner-val"] = sha256_file(val_path)
    receipt = {
        "schema_version": "1.0.0", "stage": "M33_DEVELOPMENT_TRAIN",
        "status": "PASS_TRAINED", "rotation": args.rotation, "family": args.family,
        "radius_cM": args.radius, "boundary_loss_weight": args.beta, "seed": args.seed,
        "arm": args.arm, "updates_executed": update, "selected_update": best_update,
        "selected_inner_validation_loss": best_loss, "wall_seconds": time.monotonic() - started,
        "model_sha256": sha256_file(checkpoint), "score_truth_accessed": False,
        "fit_roots": fit_seeds, "score_only_root": score_seed,
        "prediction_sha256": prediction_hashes,
    }
    (args.outdir / "train.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return receipt


if __name__ == "__main__":
    run(parse_args())
