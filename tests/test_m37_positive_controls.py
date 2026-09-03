"""Identifiable M37 controls with people held out from fitting and tuning."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from m37_trace_core import TraceSpec
from m37_trace_train import authenticate_feature_pair, train


MARKERS = 17
MARKER_CM = np.arange(MARKERS, dtype=np.float64) / 100.0
RADIUS_CM = .04
SUPPORT = np.abs(MARKER_CM - .08) < RADIUS_CM


def _features(patterns: np.ndarray, prefix: str, disable_values: bool = False,
              interaction: bool = False) -> dict[str, np.ndarray]:
    people = len(patterns)
    event_values: list[np.ndarray] = []
    event_samples: list[int] = []
    event_cm: list[float] = []
    event_anchor: list[int] = []
    for person, pattern in enumerate(patterns):
        bits = (int(pattern),) if not interaction else (int(pattern) // 2, int(pattern) % 2)
        centres = (.08,) if not interaction else (.06, .10)
        for bit, centre in zip(bits, centres):
            value = np.zeros(20, dtype=np.float32)
            value[0] = 0.0 if disable_values else (-1.0 if bit == 0 else 1.0)
            event_values.append(value)
            event_samples.append(person)
            event_cm.append(centre)
            event_anchor.append(int(np.argmin(np.abs(MARKER_CM - centre))))
    events = len(event_samples)
    axis_digest = np.asarray(["fixture-marker-axis"])
    baseline = np.full((people, MARKERS, 6), 1 / 6, dtype=np.float32)
    # Outside the injected local effect, F0 is already correct.  This makes
    # the control identify whether the residual lane can learn the event rule
    # instead of rewarding a chromosome-wide class bias.
    baseline[:, ~SUPPORT] = np.asarray([.995, .001, .001, .001, .001, .001], dtype=np.float32)
    return {
        "baseline_states": baseline,
        "evidence_field": np.zeros((people, MARKERS, 6), dtype=np.float32),
        "marker_cM": MARKER_CM.copy(),
        "marker_pos": np.arange(MARKERS, dtype=np.int64),
        "marker_axis_sha256": axis_digest,
        "sample_key_sha256": np.asarray([f"{prefix}-{index:03d}".encode() for index in range(people)]),
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
    if interaction:
        state = ((patterns // 2) ^ (patterns % 2)) * 3
    else:
        state = patterns * 3
    result = np.zeros((len(patterns), MARKERS), dtype=np.uint8)
    result[:, SUPPORT] = state[:, None]
    return result


def _balanced_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted = prediction[:, SUPPORT].argmax(axis=2).reshape(-1)
    observed = truth[:, SUPPORT].reshape(-1)
    classes = np.unique(observed)
    return float(np.mean([(predicted[observed == value] == value).mean() for value in classes]))


def _fit_and_score(fit: dict[str, np.ndarray], valid: dict[str, np.ndarray],
                   fit_truth: np.ndarray, valid_truth: np.ndarray,
                   updates: int = 500, patience: int = 8) -> float:
    assert authenticate_feature_pair(fit, valid) == "SEALED_VALID"
    prediction, _ = train(
        fit, valid, fit_truth, "tcn", 12.0, 1.0,
        TraceSpec(32, 2, 3, 0.0, (1, 2)), updates, 1e-3,
        8, MARKERS, 20, patience, 1103, tune_fraction=.2,
        split_seed=3401103, event_radius_cm=RADIUS_CM,
    )
    return _balanced_accuracy(prediction, valid_truth)


def _retain_event_centres(features: dict[str, np.ndarray], centres: tuple[float, ...]
                          ) -> dict[str, np.ndarray]:
    """Remove model events while preserving the shared scheduling calendar."""
    result = {name: value.copy() for name, value in features.items()}
    event_mask = np.isin(np.round(result["event_cM"], 8), np.round(np.asarray(centres), 8))
    event_count = len(event_mask)
    for name, value in tuple(result.items()):
        if name.startswith("event_") and value.ndim >= 1 and len(value) == event_count:
            result[name] = value[event_mask]
    return result


def test_additive_control_generalizes_to_held_out_people_and_negatives_do_not() -> None:
    fit_pattern = np.tile(np.asarray([0, 1], dtype=np.int64), 32)
    valid_pattern = np.tile(np.asarray([1, 0], dtype=np.int64), 12)
    fit_truth, valid_truth = _labels(fit_pattern), _labels(valid_pattern)
    positive = _fit_and_score(_features(fit_pattern, "fit"), _features(valid_pattern, "valid"),
                              fit_truth, valid_truth)
    disabled = _fit_and_score(_features(fit_pattern, "fit-off", True),
                              _features(valid_pattern, "valid-off", True), fit_truth, valid_truth)
    permuted = _fit_and_score(_features(fit_pattern, "fit-permuted"),
                              _features(valid_pattern, "valid-permuted"), np.roll(fit_truth, 1, axis=0),
                              valid_truth)
    f0 = _balanced_accuracy(_features(valid_pattern, "valid-f0")["baseline_states"], valid_truth)
    assert positive >= .80
    assert positive >= f0 + .25 and positive >= disabled + .25 and positive >= permuted + .25
    assert disabled <= .60 and permuted <= .60


def test_nearby_xor_interaction_generalizes_beyond_each_marginal() -> None:
    fit_pattern = np.tile(np.arange(4, dtype=np.int64), 24)
    valid_pattern = np.tile(np.asarray([3, 0, 2, 1], dtype=np.int64), 8)
    fit_truth, valid_truth = _labels(fit_pattern, True), _labels(valid_pattern, True)
    positive = _fit_and_score(_features(fit_pattern, "fit-xor", interaction=True),
                              _features(valid_pattern, "valid-xor", interaction=True),
                              fit_truth, valid_truth, updates=2500, patience=30)
    zero_event = _fit_and_score(
        _retain_event_centres(_features(fit_pattern, "fit-xor-zero", interaction=True), ()),
        _retain_event_centres(_features(valid_pattern, "valid-xor-zero", interaction=True), ()),
        fit_truth, valid_truth, updates=2500, patience=30,
    )
    one_event = _fit_and_score(
        _retain_event_centres(_features(fit_pattern, "fit-xor-one", interaction=True), (.06,)),
        _retain_event_centres(_features(valid_pattern, "valid-xor-one", interaction=True), (.06,)),
        fit_truth, valid_truth, updates=2500, patience=30,
    )
    # Shuffling the second bit destroys the joint rule while preserving both
    # single-bit marginals and the event geometry.
    shuffled_labels = _labels((fit_pattern // 2) * 2 + np.roll(fit_pattern % 2, 1), True)
    joint_negative = _fit_and_score(_features(fit_pattern, "fit-xor-negative", interaction=True),
                                    _features(valid_pattern, "valid-xor-negative", interaction=True),
                                    shuffled_labels, valid_truth, updates=2500, patience=30)
    f0 = _balanced_accuracy(_features(valid_pattern, "valid-xor-f0", interaction=True)["baseline_states"], valid_truth)
    assert positive >= .75
    assert positive >= max(f0, zero_event, one_event, joint_negative) + .20
    assert max(f0, zero_event, one_event, joint_negative) <= .60
