#!/usr/bin/env python3
"""Run small held-out M37 detectability controls at the 200-update rung."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from m37_trace_core import TraceSpec, hmm_posterior, require
from m37_trace_train import authenticate_feature_pair, train


MARKER_CM = np.arange(17, dtype=np.float64) / 10.0
RADIUS_CM = 0.2
SUPPORT = np.abs(MARKER_CM - 0.8) < RADIUS_CM
CALIBRATION_BUDGETS = (200, 400, 800, 1600, 2500)
CONTROL_LEARNING_RATE = 1e-3
CONTROL_VALIDATION_EVERY = 20
CONTROL_PATIENCE = 30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _features(patterns: np.ndarray, prefix: str, disable_values: bool = False,
              interaction: bool = False) -> dict[str, np.ndarray]:
    people = len(patterns)
    event_values: list[np.ndarray] = []
    event_samples: list[int] = []
    event_cm: list[float] = []
    event_anchor: list[int] = []
    for person, pattern in enumerate(patterns):
        bits = (int(pattern),) if not interaction else (int(pattern) // 2, int(pattern) % 2)
        centres = (0.8,) if not interaction else (0.6, 1.0)
        for bit, centre in zip(bits, centres):
            value = np.zeros(20, dtype=np.float32)
            value[0] = 0.0 if disable_values else (-1.0 if bit == 0 else 1.0)
            event_values.append(value)
            event_samples.append(person)
            event_cm.append(centre)
            event_anchor.append(int(np.argmin(np.abs(MARKER_CM - centre))))
    events = len(event_samples)
    baseline = np.full((people, len(MARKER_CM), 6), 1 / 6, dtype=np.float32)
    baseline[:, ~SUPPORT] = np.asarray([.995, .001, .001, .001, .001, .001], dtype=np.float32)
    return {
        "baseline_states": baseline,
        "evidence_field": np.zeros_like(baseline),
        "marker_cM": MARKER_CM.copy(),
        "marker_pos": np.arange(len(MARKER_CM), dtype=np.int64),
        "marker_axis_sha256": np.asarray(["m37-compact-positive-control-axis"]),
        "sample_key_sha256": np.asarray(
            [f"{prefix}-{index:03d}".encode("ascii") for index in range(people)], dtype="S64",
        ),
        "event_values": np.asarray(event_values, dtype=np.float32),
        "event_context_7mer": np.ones(events, dtype=np.uint16),
        "event_sample": np.asarray(event_samples, dtype=np.uint32),
        "event_cM": np.asarray(event_cm, dtype=np.float64),
        "event_marker_left": np.asarray(event_anchor, dtype=np.uint32),
        "event_marker_right": np.asarray(event_anchor, dtype=np.uint32),
        "event_delta_left_cM": np.zeros(events, dtype=np.float32),
        "event_delta_right_cM": np.zeros(events, dtype=np.float32),
        "schedule_sample": np.asarray(event_samples, dtype=np.uint32),
        "schedule_marker": np.asarray(event_anchor, dtype=np.uint32),
        "state_names": np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"]),
    }


def _labels(patterns: np.ndarray, interaction: bool = False) -> np.ndarray:
    state = (((patterns // 2) ^ (patterns % 2)) * 3 if interaction else patterns * 3)
    result = np.zeros((len(patterns), len(MARKER_CM)), dtype=np.uint8)
    result[:, SUPPORT] = state[:, None]
    return result


def _retain_centres(features: dict[str, np.ndarray], centres: tuple[float, ...]
                    ) -> dict[str, np.ndarray]:
    result = {name: value.copy() for name, value in features.items()}
    keep = np.isin(np.round(result["event_cM"], 8), np.round(np.asarray(centres), 8))
    event_count = len(keep)
    for name, value in tuple(result.items()):
        if name.startswith("event_") and value.ndim >= 1 and len(value) == event_count:
            result[name] = value[keep]
    return result


def _balanced_accuracy(probability: np.ndarray, truth: np.ndarray) -> float:
    predicted = probability[:, SUPPORT].argmax(axis=2).reshape(-1)
    observed = truth[:, SUPPORT].reshape(-1)
    return float(np.mean([(predicted[observed == value] == value).mean()
                          for value in np.unique(observed)]))


def _fit_score(fit: dict[str, np.ndarray], valid: dict[str, np.ndarray],
               fit_truth: np.ndarray, valid_truth: np.ndarray, updates: int) -> float:
    require(authenticate_feature_pair(fit, valid) == "SEALED_VALID",
            "positive-control FIT/VALID people are not disjoint")
    probability, _ = train(
        fit, valid, fit_truth, "tcn", 12.0, 1.0,
        TraceSpec(32, 2, 3, 0.0, (1, 2)), updates, CONTROL_LEARNING_RATE,
        8, len(MARKER_CM), CONTROL_VALIDATION_EVERY, CONTROL_PATIENCE, 1103,
        tune_fraction=.2,
        split_seed=3401103, event_radius_cm=RADIUS_CM,
    )
    return _balanced_accuracy(probability, valid_truth)


def run_controls(updates: int = 200) -> dict[str, object]:
    require(updates in CALIBRATION_BUDGETS,
            "M37 compact positive-control budget is not in the prespecified calibration grid")
    import torch
    torch.set_num_threads(1)
    fit_add = np.tile(np.asarray([0, 1], dtype=np.int64), 32)
    valid_add = np.tile(np.asarray([1, 0], dtype=np.int64), 12)
    fit_add_truth, valid_add_truth = _labels(fit_add), _labels(valid_add)
    additive = _fit_score(_features(fit_add, "fit-add"), _features(valid_add, "valid-add"),
                          fit_add_truth, valid_add_truth, updates)
    additive_disabled = _fit_score(
        _features(fit_add, "fit-add-off", disable_values=True),
        _features(valid_add, "valid-add-off", disable_values=True),
        fit_add_truth, valid_add_truth, updates,
    )
    additive_permuted = _fit_score(
        _features(fit_add, "fit-add-permuted"),
        _features(valid_add, "valid-add-permuted"),
        np.roll(fit_add_truth, 1, axis=0), valid_add_truth, updates,
    )
    additive_f0 = _balanced_accuracy(
        _features(valid_add, "valid-add-f0")["baseline_states"], valid_add_truth,
    )
    additive_pass = (additive >= .80 and
                     additive >= max(additive_f0, additive_disabled, additive_permuted) + .25)

    fit_xor = np.tile(np.arange(4, dtype=np.int64), 24)
    valid_xor = np.tile(np.asarray([3, 0, 2, 1], dtype=np.int64), 8)
    fit_xor_truth, valid_xor_truth = _labels(fit_xor, True), _labels(valid_xor, True)
    xor = _fit_score(_features(fit_xor, "fit-xor", interaction=True),
                     _features(valid_xor, "valid-xor", interaction=True),
                     fit_xor_truth, valid_xor_truth, updates)
    xor_zero = _fit_score(
        _retain_centres(_features(fit_xor, "fit-xor-zero", interaction=True), ()),
        _retain_centres(_features(valid_xor, "valid-xor-zero", interaction=True), ()),
        fit_xor_truth, valid_xor_truth, updates,
    )
    xor_one = _fit_score(
        _retain_centres(_features(fit_xor, "fit-xor-one", interaction=True), (0.6,)),
        _retain_centres(_features(valid_xor, "valid-xor-one", interaction=True), (0.6,)),
        fit_xor_truth, valid_xor_truth, updates,
    )
    shuffled = _labels((fit_xor // 2) * 2 + np.roll(fit_xor % 2, 1), True)
    xor_shuffled = _fit_score(
        _features(fit_xor, "fit-xor-shuffled", interaction=True),
        _features(valid_xor, "valid-xor-shuffled", interaction=True),
        shuffled, valid_xor_truth, updates,
    )
    xor_f0 = _balanced_accuracy(
        _features(valid_xor, "valid-xor-f0", interaction=True)["baseline_states"],
        valid_xor_truth,
    )
    xor_pass = (xor >= .75 and
                xor >= max(xor_f0, xor_zero, xor_one, xor_shuffled) + .20)

    baseline = np.full((2, 9, 6), 1 / 6, dtype=np.float32)
    evidence = np.zeros_like(baseline)
    evidence[:, 3:6, 3] = 1.0
    hmm_on = hmm_posterior(baseline, evidence, np.arange(9, dtype=float) / 10, 12.0, 1.0)
    hmm_off = hmm_posterior(baseline, np.zeros_like(evidence),
                            np.arange(9, dtype=float) / 10, 12.0, 1.0)
    hmm_change = float(np.max(np.abs(hmm_on - hmm_off)))
    return {
        "hmm": {
            "status": "PASS_ADDITIVE_DETECTABILITY" if hmm_change > 1e-4 else "FAIL_ADDITIVE_DETECTABILITY",
            "maximum_posterior_change": hmm_change,
            "interaction_control": "NOT_APPLICABLE_NONLEARNED_ADDITIVE_EVIDENCE_MODEL",
        },
        "tcn": {
            "anchor_candidate_id": "tcn_anchor_pc",
            "updates": updates,
            "architecture": {"hidden_dim": 32, "depth": 2, "kernel_size": 3,
                             "dropout": 0.0, "dilations": [1, 2],
                             "learning_rate": CONTROL_LEARNING_RATE, "seed": 1103,
                             "evidence_scale": 1.0,
                             "event_radius_cM": RADIUS_CM,
                             "validation_every": CONTROL_VALIDATION_EVERY,
                             "early_stopping_patience": CONTROL_PATIENCE},
            "additive": {"positive": additive, "F0": additive_f0,
                         "value_disabled": additive_disabled,
                         "label_permuted": additive_permuted,
                         "status": "PASS" if additive_pass else "BUDGET_INSUFFICIENT"},
            "xor_interaction": {"positive": xor, "F0": xor_f0,
                                "zero_event": xor_zero, "one_event": xor_one,
                                "joint_rule_destroyed": xor_shuffled,
                                "status": ("PASS" if xor_pass else
                                           "BUDGET_INSUFFICIENT_FOR_INTERACTION")},
            "scientific_closure_if_failed": "FORBIDDEN",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--contract-amendment", type=Path, required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--updates", type=int, choices=CALIBRATION_BUDGETS, default=200)
    parser.add_argument("--auth-file", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and "@sha256:" in args.container_digest,
            "M37 positive-control output/container differs")
    started = time.perf_counter()
    controls = run_controls(args.updates)
    elapsed_seconds = time.perf_counter() - started
    maximum_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    authenticated_sources = {path.name: sha256(path) for path in args.auth_file}
    require(len(authenticated_sources) == len(args.auth_file),
            "M37 positive-control authenticated source basenames collide")
    payload = {
        "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_POSITIVE_CONTROL",
        "run_id": args.run_id,
        "budget": {"updates": args.updates, "calibration_grid": list(CALIBRATION_BUDGETS)},
        "runtime": {"elapsed_seconds": elapsed_seconds,
                    "maximum_resident_set_size_kib": maximum_rss_kib},
        "controls": controls,
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "parent_contract_sha256": sha256(args.parent_contract),
        "contract_amendment_sha256": sha256(args.contract_amendment),
        "container_digest": args.container_digest,
        "authenticated_source_sha256": authenticated_sources,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".receipt.json").write_text(json.dumps({
        "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_POSITIVE_CONTROL",
        "run_id": args.run_id, "container_digest": args.container_digest,
        "authenticated_source_sha256": authenticated_sources,
        "output_sha256": sha256(args.output),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
