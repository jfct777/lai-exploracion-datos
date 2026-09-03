#!/usr/bin/env python3
"""Train or apply compact phase-free TRACE-LAI candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import (PROBABILITY_FLOOR, TraceSpec, build_tcn,
                            hmm_posterior, m34_labels_to_states, require)


def probability_nll(prediction, labels):
    """Mean NLL with the same additive floor used by TRACE scoring.

    Adding the floor inside the logarithm, instead of clamping the probability
    before ``log``, preserves a finite derivative for a structurally zero F0
    state when an event-supported residual proposes that state.
    """
    selected = prediction.reshape(-1, prediction.shape[-1]).gather(
        1, labels.reshape(-1, 1),
    ).squeeze(1)
    return -(selected + PROBABILITY_FLOOR).log().mean()


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"baseline_states", "evidence_field", "event_values", "event_context_7mer", "event_sample",
                    "event_marker_left", "event_marker_right", "event_delta_left_cM", "event_delta_right_cM",
                    "event_cM", "marker_cM", "marker_pos", "marker_axis_sha256", "sample_key_sha256",
                    "state_names", "schedule_sample", "schedule_marker"}
        require(required.issubset(archive.files), "TRACE feature members differ")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def verify_stage_receipt(artifact: Path, receipt_path: Path, expected_stage: str,
                         expected_arm: str | None = None) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("stage") == expected_stage, f"{expected_stage} receipt stage differs")
    require(receipt.get("output_sha256") == hashlib.sha256(artifact.read_bytes()).hexdigest(),
            f"{expected_stage} artifact/receipt hash differs")
    if expected_arm is not None:
        require(receipt.get("arm") == expected_arm, f"{expected_stage} arm differs")
    return receipt


def load_truth(path: Path) -> np.ndarray:
    """Accept sealed TRACE states or raw sealed M34 labels [N,2,M]."""
    with np.load(path, allow_pickle=False) as archive:
        if "state_labels" in archive.files:
            value = np.ascontiguousarray(archive["state_labels"])
            require(value.ndim == 2 and np.issubdtype(value.dtype, np.integer) and
                    np.all((value >= 0) & (value < 6)), "TRACE truth states differ")
            return value
        require("labels" in archive.files, "truth needs TRACE state_labels or M34 labels [N,2,M]")
        return m34_labels_to_states(np.ascontiguousarray(archive["labels"]))


def authenticate_truth_axes(path: Path, features: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        require({"sample_key_sha256", "marker_pos"}.issubset(archive.files), "truth lacks authenticated sample/marker axes")
        require(np.array_equal(archive["sample_key_sha256"], features["sample_key_sha256"]) and
                np.array_equal(archive["marker_pos"], features["marker_pos"]), "truth/features sample or marker axes differ")


def deterministic_train_tune(features: dict[str, np.ndarray], split_seed: int, tune_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Split FIT individuals by stable sample-key digest, never by VALID truth."""
    require(0.05 <= tune_fraction <= 0.4, "tune fraction must be in [0.05,0.4]")
    require("sample_key_sha256" in features, "TRACE FIT features lack sample keys for deterministic TRAIN/TUNE split")
    keys = np.asarray(features["sample_key_sha256"])
    require(keys.ndim == 1 and len(keys) >= 3 and len(np.unique(keys)) == len(keys), "TRACE FIT sample keys differ")
    threshold = int(round(tune_fraction * 10_000))
    buckets = np.asarray([int.from_bytes(hashlib.sha256(bytes(value) + str(split_seed).encode("ascii")).digest()[:4], "big") % 10_000
                          for value in keys])
    tune = np.flatnonzero(buckets < threshold)
    # Stable repair for a small FIT panel: it is based on the same digest, not labels.
    if len(tune) == 0:
        tune = np.asarray([int(np.argmin(buckets))])
    if len(tune) == len(keys):
        tune = tune[:-1]
    train = np.setdiff1d(np.arange(len(keys)), tune, assume_unique=True)
    require(len(train) > 0 and len(tune) > 0, "FIT TRAIN/TUNE split is empty")
    return train.astype(np.int64), tune.astype(np.int64)


def _event_batch(features: dict[str, np.ndarray], people: np.ndarray, marker_start: int, marker_end: int,
                 evidence_scale: float = 1.0, event_radius_cm: float = .2):
    """Create an event batch and a fixed triangular kernel on the genetic map.

    Every retained event contributes to all markers inside ``event_radius_cm``.
    Kernel weights vary continuously with cM distance and are zero outside the
    declared radius; no physical-window or marker-count surrogate is used.
    """
    import torch
    require(event_radius_cm > 0 and 0 <= marker_start < marker_end <= len(features["marker_cM"]),
            "TRACE event batch interval differs")
    original = features["event_sample"].astype(np.int64)
    remap = np.full(features["baseline_states"].shape[0], -1, dtype=np.int64)
    remap[people] = np.arange(len(people))
    marker_cm = np.asarray(features["marker_cM"], dtype=np.float64)[marker_start:marker_end]
    event_cm = np.asarray(features["event_cM"], dtype=np.float64)
    selected = ((remap[original] >= 0) &
                (event_cm >= marker_cm[0] - event_radius_cm) &
                (event_cm <= marker_cm[-1] + event_radius_cm))
    values = features["event_values"][selected].astype(np.float32).copy()
    values[:, 4:10] *= evidence_scale
    selected_event_cm = event_cm[selected]
    splat_event: list[np.ndarray] = []
    splat_marker: list[np.ndarray] = []
    splat_weight: list[np.ndarray] = []
    for event_index, center in enumerate(selected_event_cm):
        left = int(np.searchsorted(marker_cm, center - event_radius_cm, side="left"))
        right = int(np.searchsorted(marker_cm, center + event_radius_cm, side="right"))
        if right <= left:
            continue
        indices = np.arange(left, right, dtype=np.int64)
        weights = np.maximum(1.0 - np.abs(marker_cm[indices] - center) / event_radius_cm, 0.0)
        positive = weights > 0
        if not positive.any():
            continue
        indices, weights = indices[positive], weights[positive]
        splat_event.append(np.full(len(indices), event_index, dtype=np.int64))
        splat_marker.append(indices)
        splat_weight.append(weights.astype(np.float32))
    packed_event = np.concatenate(splat_event) if splat_event else np.empty(0, dtype=np.int64)
    packed_marker = np.concatenate(splat_marker) if splat_marker else np.empty(0, dtype=np.int64)
    packed_weight = np.concatenate(splat_weight) if splat_weight else np.empty(0, dtype=np.float32)
    return tuple(torch.from_numpy(value) for value in (
        values, features["event_context_7mer"][selected].astype(np.int64),
        remap[original[selected]], packed_event, packed_marker, packed_weight,
    ))


def receptive_halo(spec: TraceSpec) -> int:
    """Symmetric marker halo required by the compact dilated TCN."""
    return sum(dilation * (spec.kernel_size - 1) // 2 for dilation in spec.dilations)


def event_centered_schedule(features: dict[str, np.ndarray], updates: int, seed: int, radius_cm: float,
                            train_people: np.ndarray) -> list[tuple[int, int]]:
    """Shared deterministic calendar derived only from TRAIN individuals."""
    require(radius_cm > 0 and updates > 0 and {"schedule_sample", "schedule_marker"}.issubset(features),
            "TRACE event calendar differs")
    schedule_sample = np.asarray(features["schedule_sample"], dtype=np.int64)
    schedule_marker = np.asarray(features["schedule_marker"], dtype=np.int64)
    require(schedule_sample.shape == schedule_marker.shape and schedule_sample.ndim == 1,
            "TRACE schedule axes differ")
    calendar = schedule_marker[np.isin(schedule_sample, np.asarray(train_people, dtype=np.int64))]
    marker_cm = np.asarray(features["marker_cM"], dtype=float)
    if not len(calendar):
        calendar = np.asarray([len(marker_cm) // 2])
    calendar = np.unique(calendar)
    rng = np.random.default_rng(seed)
    order = rng.permutation(calendar)
    return [(int(np.searchsorted(marker_cm, marker_cm[order[index % len(order)]] - radius_cm, side="left")),
             int(np.searchsorted(marker_cm, marker_cm[order[index % len(order)]] + radius_cm, side="right")))
            for index in range(updates)]


def _predict_batched(model, features: dict[str, np.ndarray], batch_people: int, marker_shard: int,
                     evidence_scale: float, event_radius_cm: float, people_indices: np.ndarray | None = None,
                     halo: int = 0) -> np.ndarray:
    import torch
    baseline = features["baseline_states"]
    selected_people = (np.arange(len(baseline), dtype=np.int64) if people_indices is None else
                       np.asarray(people_indices, dtype=np.int64))
    result = np.empty((len(selected_people), baseline.shape[1], baseline.shape[2]), dtype=baseline.dtype)
    model.eval()
    with torch.no_grad():
        for first in range(0, len(selected_people), batch_people):
            people = selected_people[first:first + batch_people]
            for left in range(0, baseline.shape[1], marker_shard):
                right = min(left + marker_shard, baseline.shape[1])
                expanded_left, expanded_right = max(0, left - halo), min(baseline.shape[1], right + halo)
                event = _event_batch(features, people, expanded_left, expanded_right, evidence_scale, event_radius_cm)
                expanded = model(*event, torch.from_numpy(baseline[people, expanded_left:expanded_right])).numpy()
                result[first:first + len(people), left:right] = expanded[:, left - expanded_left:right - expanded_left]
    return result


def train(features: dict[str, np.ndarray], predict_features: dict[str, np.ndarray], truth: np.ndarray | None, family: str,
          hazard: float, evidence_scale: float, spec: TraceSpec, updates: int, learning_rate: float,
          batch_people: int, marker_shard: int, validation_every: int, patience: int, seed: int,
          checkpoint: Path | None = None, tune_fraction: float = .2, split_seed: int = 3401103,
          event_radius_cm: float = .2,
          training_diagnostics: dict[str, int | bool | None] | None = None,
          early_stopping: bool = True,
          restore_best: bool = True,
          _capacity_loss_marker_mask: np.ndarray | None = None,
          ) -> tuple[np.ndarray, np.ndarray]:
    baseline, evidence, marker = features["baseline_states"], features["evidence_field"], features["marker_cM"]
    if family == "hmm":
        _, tune_people = deterministic_train_tune(features, split_seed, tune_fraction)
        return (hmm_posterior(predict_features["baseline_states"], predict_features["evidence_field"],
                              predict_features["marker_cM"], hazard, evidence_scale), tune_people)
    require(family == "tcn", "TRACE family differs")
    require(truth is not None and truth.shape == baseline.shape[:2], "TCN needs phase-free FIT truth")
    capacity_loss_marker_mask: np.ndarray | None = None
    if _capacity_loss_marker_mask is not None:
        capacity_loss_marker_mask = np.asarray(_capacity_loss_marker_mask)
        require(
            capacity_loss_marker_mask.dtype == np.bool_ and
            capacity_loss_marker_mask.shape == (baseline.shape[1],) and
            bool(capacity_loss_marker_mask.any()) and
            not early_stopping and not restore_best and checkpoint is None,
            "capacity-only loss mask differs",
        )
    import torch
    torch.manual_seed(seed)
    train_people, tune_people = deterministic_train_tune(features, split_seed, tune_fraction)
    model = build_tcn(spec, features["event_values"].shape[1])
    require(sum(parameter.numel() for parameter in model.parameters()) <= 200_000,
            "TRACE TCN exceeds the preregistered compact capacity cap")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    best_loss, best_state, stale = float("inf"), None, 0
    completed_updates = 0
    best_checkpoint_update: int | None = None
    calendar = event_centered_schedule(features, updates, split_seed, event_radius_cm, train_people)
    for update in range(updates):
        people = train_people[np.arange(update * batch_people, update * batch_people + batch_people) % len(train_people)]
        left, right = calendar[update]
        require(right > left, "event-centred cM calendar emitted empty shard")
        halo = receptive_halo(spec)
        expanded_left, expanded_right = max(0, left - halo), min(baseline.shape[1], right + halo)
        event = _event_batch(features, people, expanded_left, expanded_right, evidence_scale, event_radius_cm)
        optimizer.zero_grad()
        expanded = model(*event, torch.from_numpy(baseline[people, expanded_left:expanded_right]))
        prediction = expanded[:, left - expanded_left:right - expanded_left]
        labels = torch.from_numpy(truth[people, left:right].astype(np.int64, copy=False))
        if capacity_loss_marker_mask is not None:
            local_mask = capacity_loss_marker_mask[left:right]
            require(bool(local_mask.any()), "capacity-only loss shard lacks supported markers")
            prediction = prediction[:, local_mask]
            labels = labels[:, local_mask]
        loss = probability_nll(prediction, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_updates = update + 1
        if early_stopping and (update + 1) % validation_every == 0:
            validation = _predict_batched(model, features, batch_people, marker_shard, evidence_scale,
                                          event_radius_cm, tune_people,
                                          receptive_halo(spec))
            tune_truth = truth[tune_people]
            index = np.arange(tune_truth.size)
            # Mirror the differentiable training convention exactly.  The
            # reporting scorer keeps its historical clipping convention so
            # canonical F0/HMM metrics remain comparable.
            observed = -np.log(
                validation.reshape(-1, 6)[index, tune_truth.reshape(-1)] +
                PROBABILITY_FLOOR
            ).mean()
            if observed < best_loss:
                best_loss, best_state, stale = observed, {key: value.detach().clone() for key, value in model.state_dict().items()}, 0
                best_checkpoint_update = update + 1
            else:
                stale += 1
                if stale >= patience:
                    break
            model.train()
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    if training_diagnostics is not None:
        training_diagnostics.clear()
        training_diagnostics.update({
            "requested_updates": int(updates),
            "completed_updates": int(completed_updates),
            "best_checkpoint_update": best_checkpoint_update,
            "validation_every": int(validation_every),
            "early_stopping_patience": int(patience),
            "early_stopping_enabled": bool(early_stopping),
            "restore_best_checkpoint": bool(restore_best),
        })
        if capacity_loss_marker_mask is not None:
            training_diagnostics["capacity_loss_marker_count"] = int(
                capacity_loss_marker_mask.sum(),
            )
    if checkpoint:
        torch.save({"state_dict": model.state_dict(), "event_channels": features["event_values"].shape[1], "spec": spec.__dict__}, checkpoint)
    with torch.no_grad():
        return (_predict_batched(model, predict_features, batch_people, marker_shard, evidence_scale,
                                 event_radius_cm, None,
                                 receptive_halo(spec)).astype(np.float32),
                tune_people)


def authenticate_feature_pair(fit: dict[str, np.ndarray], predict: dict[str, np.ndarray]) -> str:
    """Require identical marker axes and either identical or disjoint people."""
    require(np.array_equal(fit["marker_pos"], predict["marker_pos"]) and
            np.array_equal(fit["marker_cM"], predict["marker_cM"]) and
            np.array_equal(fit["marker_axis_sha256"], predict["marker_axis_sha256"]),
            "FIT/predict physical-genetic marker axes differ")
    fit_keys = np.asarray(fit["sample_key_sha256"])
    predict_keys = np.asarray(predict["sample_key_sha256"])
    if np.array_equal(fit_keys, predict_keys):
        return "FIT_TUNE"
    overlap = np.intersect1d(fit_keys, predict_keys)
    require(len(overlap) == 0, "FIT and VALID sample axes overlap")
    return "SEALED_VALID"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predict-features", type=Path, required=True)
    parser.add_argument("--family", choices=("hmm", "tcn"), required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--arm", choices=("RE", "RD", "POOLED", "SHAM", "GEOMETRY"), required=True)
    parser.add_argument("--features-receipt", type=Path, required=True)
    parser.add_argument("--predict-features-receipt", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--hazard-per-morgan", type=float, default=12.0)
    parser.add_argument("--evidence-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--dilations", default=None)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-people", type=int, default=8)
    parser.add_argument("--marker-shard", type=int, default=256)
    parser.add_argument("--validation-every", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--tune-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--split-seed", type=int, default=3401103)
    parser.add_argument("--event-radius-cm", type=float, default=.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite TRACE predictions")
    truth = None
    if args.truth:
        truth = load_truth(args.truth)
    dilations = tuple(int(value) for value in args.dilations.split(",")) if args.dilations else (1, 2, 4, 8)[:args.depth]
    fit_features = load_features(args.features)
    predict_features = load_features(args.predict_features)
    fit_receipt = verify_stage_receipt(args.features, args.features_receipt, "M37_TRACE_MATERIALIZE", args.arm)
    predict_receipt = verify_stage_receipt(args.predict_features, args.predict_features_receipt,
                                           "M37_TRACE_MATERIALIZE", args.arm)
    require(fit_receipt.get("physical_genetic_axis_sha256") ==
            str(fit_features["marker_axis_sha256"].reshape(-1)[0]) and
            predict_receipt.get("physical_genetic_axis_sha256") ==
            str(predict_features["marker_axis_sha256"].reshape(-1)[0]),
            "materialization receipt/marker axis differs")
    split_name = authenticate_feature_pair(fit_features, predict_features)
    if args.truth:
        authenticate_truth_axes(args.truth, fit_features)
    training_diagnostics: dict[str, int | bool | None] = {}
    result, tune_people = train(fit_features, predict_features, truth, args.family, args.hazard_per_morgan,
                   args.evidence_scale, TraceSpec(args.hidden_dim, args.depth, args.kernel_size, args.dropout, dilations),
                   args.updates, args.learning_rate, args.batch_people, args.marker_shard,
                   args.validation_every, args.early_stopping_patience, args.seed, args.checkpoint,
                   args.tune_fraction, args.split_seed, args.event_radius_cm,
                   training_diagnostics)
    predict_keys = np.asarray(predict_features["sample_key_sha256"])
    fit_keys = np.asarray(fit_features["sample_key_sha256"])
    # A candidate-selection invocation predicts FIT and scores only TUNE.  A
    # frozen-candidate invocation predicts disjoint VALID and scores all of it.
    evaluation_people = tune_people if split_name == "FIT_TUNE" else np.arange(len(predict_keys), dtype=np.int64)
    train_people = np.setdiff1d(np.arange(len(fit_keys)), tune_people, assume_unique=True)
    write_deterministic_npz(args.output, {"probabilities": result, "evaluation_sample_indices": evaluation_people,
                                          "evaluation_split": np.asarray([split_name]),
                                          "sample_key_sha256": predict_features["sample_key_sha256"],
                                          "marker_pos": predict_features["marker_pos"],
                                          "marker_axis_sha256": predict_features["marker_axis_sha256"],
                                          "fit_train_sample_axis_sha256": np.asarray([hashlib.sha256(fit_keys[train_people].tobytes()).hexdigest()]),
                                          "fit_tune_sample_axis_sha256": np.asarray([hashlib.sha256(fit_keys[tune_people].tobytes()).hexdigest()])})
    effective_hyperparameters = ({
        "hazard_per_morgan": args.hazard_per_morgan,
        "evidence_scale": args.evidence_scale,
    } if args.family == "hmm" else {
        "evidence_scale": args.evidence_scale,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "dilations": list(dilations),
        "learning_rate": args.learning_rate,
        "updates": args.updates,
    })
    args.output.with_suffix(".receipt.json").write_text(json.dumps({
        "schema_version": "1.0.0", "stage": "M37_TRACE_TRAIN", "candidate_id": args.candidate_id,
        "family": args.family, "arm": args.arm,
        "seed": args.seed, "batch_people": args.batch_people, "marker_shard": args.marker_shard,
        "split_seed": args.split_seed,
        "event_radius_cM": args.event_radius_cm,
        "effective_hyperparameters": effective_hyperparameters,
        "training_diagnostics": (training_diagnostics if args.family == "tcn" else
                                 "NOT_APPLICABLE_DETERMINISTIC_HMM"),
        "fit_features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "fit_features_receipt_sha256": hashlib.sha256(args.features_receipt.read_bytes()).hexdigest(),
        "predict_features_sha256": hashlib.sha256(args.predict_features.read_bytes()).hexdigest(),
        "predict_features_receipt_sha256": hashlib.sha256(args.predict_features_receipt.read_bytes()).hexdigest(),
        "fit_materialization_output_sha256": fit_receipt["output_sha256"],
        "predict_materialization_output_sha256": predict_receipt["output_sha256"],
        "truth_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest() if args.truth else None,
        "prediction_split": split_name,
        "train_sample_axis_sha256": hashlib.sha256(fit_features["sample_key_sha256"][train_people].tobytes()).hexdigest(),
        "tune_sample_axis_sha256": hashlib.sha256(fit_features["sample_key_sha256"][tune_people].tobytes()).hexdigest(),
        "fit_valid_sample_overlap": 0 if split_name == "SEALED_VALID" else None,
        "marker_axis_sha256": str(predict_features["marker_axis_sha256"].reshape(-1)[0]),
        "early_stopping": {"validation_every": args.validation_every, "patience": args.early_stopping_patience,
                           "split": "FIT_TUNE_ONLY", "tune_fraction": args.tune_fraction},
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_TRACE_TRAIN", "family": args.family,
                      "parameters_cap": 200000}, sort_keys=True))


if __name__ == "__main__":
    main()
