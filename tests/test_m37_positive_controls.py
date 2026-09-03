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
import m37_trace_compact_positive_control as capacity


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


def test_capacity_screen_keeps_every_candidate_without_fixture_ranking() -> None:
    original = capacity._candidate_ladder

    def fixture(_manifest, row, seed):
        return {"seed": seed, "pass": row["candidate_id"] != "b",
                "first_pass_updates": 200, "evaluated_updates": [200],
                "final_capacity_score": {"a": .1, "b": .9, "c": .5}[row["candidate_id"]],
                "final_weakest_margin": .1}

    manifest = {
        "candidates": [
            {"candidate_id": candidate_id, "family": "tcn"}
            for candidate_id in ("c", "a", "b")
        ],
    }
    capacity._candidate_ladder = fixture
    try:
        screen, roster = capacity.run_screen(manifest)
    finally:
        capacity._candidate_ladder = original
    assert screen["candidate_count"] == 3
    assert roster["candidate_ids"] == ["a", "b", "c"]
    assert roster["candidate_count"] == 3
    assert roster["selection_uses_fixture_scores"] is False
    assert "ranked_candidate_ids" not in roster
    assert "maximum_selected_candidates" not in roster


def test_capacity_fixture_matches_real_channels_and_requires_two_loci_for_xor() -> None:
    patterns = np.arange(4, dtype=np.int64)
    radius = .2
    observed = capacity._features(patterns, "xor-schema", radius, "xor")
    values = observed["event_values"]
    assert values.shape == (2 * len(patterns), 23)
    assert np.all(values[:, :4].sum(axis=1) == 1.0)
    assert np.all(values[:, 16:18] == 1.0)
    assert np.all(values[:, 20:23] == 1.0)
    assert np.all(observed["baseline_states"] == np.float32(1.0 / 6.0))
    support = capacity._fixture_support(radius, "xor")
    centre = int(np.argmin(np.abs(observed["marker_cM"] - 2 * radius)))
    assert not support[centre]
    assert int(support.sum()) == 1
    for person in range(len(patterns)):
        centres = observed["event_cM"][observed["event_sample"] == person]
        assert len(centres) == 2
        assert np.isclose(centres[1] - centres[0], 2 * radius)
        marker_cm = observed["marker_cM"]
        left = np.abs(marker_cm - centres[0]) < radius
        right = np.abs(marker_cm - centres[1]) < radius
        assert not np.any(left & right)
        assert np.all(support <= (left | right))
        assert np.allclose(marker_cm[support], centres[0])


def test_every_production_hmm_pair_has_candidate_specific_additive_detectability() -> None:
    import json
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] /
         "conf/m37_trace_compact_candidates.json").read_text(encoding="utf-8")
    )
    observed = capacity._hmm_controls(manifest)
    assert observed["status"] == "PASS_ALL_CANDIDATES_ADDITIVE_DETECTABILITY"
    assert observed["candidate_count"] == 12
    assert len(observed["candidates"]) == 12
    assert all(row["pass"] for row in observed["candidates"].values())


def test_capacity_replication_uses_all_fixed_seeds_without_best_seed_selection() -> None:
    manifest = {
        "execution": {},
        "families": {"hmm": {"hazard_per_morgan": 12.0,
                              "evidence_scale": 1.0}},
        "candidates": [
            {"candidate_id": "hmm", "family": "hmm"},
            {"candidate_id": "a", "family": "tcn"},
            {"candidate_id": "b", "family": "tcn"},
        ],
    }
    screen = {
        "screen_seed": 1103,
        "candidate_count": 2,
        "budget_ladder_updates": [200, 400, 800, 1600],
        "rung_execution": "DETERMINISTIC_RESTART_SAME_SEED_EACH_RUNG",
        "rung_training_policy": (
            "EXACT_REQUESTED_UPDATES_FINAL_STATE_NO_EARLY_STOPPING_NO_BEST_CHECKPOINT_RESTORE"
        ),
        "thresholds": capacity.CONTROL_THRESHOLDS,
        "fixture_oracle": {"pass": True},
        "candidates": {
            "a": {"seed": 1103, "pass": True, "first_pass_updates": 200,
                  "evaluated_updates": [200], "final_capacity_score": .9,
                  "final_weakest_margin": .1},
            "b": {"seed": 1103, "pass": False, "first_pass_updates": None,
                  "evaluated_updates": [200, 400, 800, 1600],
                  "final_capacity_score": .8, "final_weakest_margin": -.1},
        },
    }
    selection = {
        "screen_seed": 1103,
        "candidate_count": 2,
        "candidate_ids": ["a", "b"],
        "selection_uses_real_fit_or_tune_metrics": False,
        "selection_uses_fixture_scores": False,
    }
    original = capacity._candidate_ladder

    def fixture(_manifest, row, seed):
        passed = row["candidate_id"] == "a" or seed == 2207
        return {"seed": seed, "pass": passed,
                "first_pass_updates": 400 if passed else None,
                "evaluated_updates": [200, 400] if passed else [200, 400, 800, 1600],
                "final_capacity_score": .7,
                "final_weakest_margin": .1 if passed else -.1}

    capacity._candidate_ladder = fixture
    try:
        observed = capacity.run_replication(manifest, screen, selection)["tcn"]
    finally:
        capacity._candidate_ladder = original
    assert observed["eligible_candidate_ids"] == ["a"]
    assert observed["candidates"]["a"]["pass_count"] == 3
    assert observed["candidates"]["b"]["pass_count"] == 1
    assert observed["candidates"]["a"]["effective_updates"] == 400
    assert observed["candidate_count"] == 2
    assert observed["evaluated_candidate_ids"] == ["a", "b"]
    assert observed["selection_of_best_candidate"] == "FORBIDDEN"
    assert set(observed["candidates"]["a"]["seed_results"]) == {
        "1103", "2207", "3301",
    }
    assert observed["selection_of_best_seed"] == "FORBIDDEN"


def test_capacity_ladder_stops_at_first_passing_budget() -> None:
    original = capacity._candidate_controls
    calls: list[int] = []

    def fixture(_manifest, _row, seed, updates):
        calls.append(updates)
        passed = updates >= 800
        return {"seed": seed, "pass": passed, "capacity_score": .5,
                "weakest_margin": .1 if passed else -.1}

    capacity._candidate_controls = fixture
    try:
        observed = capacity._candidate_ladder({}, {}, 1103)
    finally:
        capacity._candidate_controls = original
    assert calls == [200, 400, 800]
    assert observed["first_pass_updates"] == 800
    assert observed["evaluated_updates"] == [200, 400, 800]
    assert set(observed["rung_results"]) == {"200", "400", "800"}


def test_capacity_rung_disables_early_stop_and_scores_the_final_update() -> None:
    manifest = {
        "execution": {
            "batch_people": 8, "marker_shard": 17,
            "validation_every": 25, "early_stopping_patience": 4,
            "tune_fraction": .2, "split_seed": 3401103,
            "event_radius_cM": .2,
        },
        "families": {
            "tcn": {
                "evidence_scale": 1.0, "hidden_dim": 32, "depth": 2,
                "kernel_size": 3, "dropout": 0.0, "dilations": [1, 2],
                "seed": 1103, "learning_rate": .001,
            },
        },
    }
    row = {"candidate_id": "fixture", "family": "tcn", "parameters": {}}
    effective = capacity._effective_tcn(manifest, row, 1103, 200)
    original = capacity.train
    observed_kwargs: dict[str, object] = {}

    def fixture_train(*args, **kwargs):
        observed_kwargs.update(kwargs)
        diagnostic = kwargs["training_diagnostics"]
        diagnostic.update({
            "requested_updates": 200,
            "completed_updates": 200,
            "best_checkpoint_update": None,
            "validation_every": 25,
            "early_stopping_patience": 4,
            "early_stopping_enabled": False,
            "restore_best_checkpoint": False,
            "capacity_loss_marker_count": int(
                capacity._fixture_support(.2, "xor").sum(),
            ),
        })
        return args[1]["baseline_states"], np.asarray([0], dtype=np.int64)

    capacity.train = fixture_train
    try:
        result = capacity._fit_score(effective, 1103, "xor")
    finally:
        capacity.train = original
    assert observed_kwargs["early_stopping"] is False
    assert observed_kwargs["restore_best"] is False
    assert np.array_equal(
        observed_kwargs["_capacity_loss_marker_mask"],
        capacity._fixture_support(.2, "xor"),
    )
    assert result["training_diagnostics"]["completed_updates"] == 200
    assert result["training_diagnostics"]["best_checkpoint_update"] is None


def test_candidate_gate_requires_the_runtime_one_bit_ablation() -> None:
    manifest = {
        "execution": {
            "batch_people": 8, "marker_shard": 9,
            "validation_every": 25, "early_stopping_patience": 4,
            "tune_fraction": .2, "split_seed": 3401103,
            "event_radius_cM": .2,
        },
        "families": {
            "tcn": {
                "evidence_scale": 1.0, "hidden_dim": 32, "depth": 2,
                "kernel_size": 3, "dropout": 0.0, "dilations": [1, 2],
                "seed": 1103, "learning_rate": .001,
            },
        },
    }
    row = {"candidate_id": "fixture", "family": "tcn", "parameters": {}}
    original = capacity._fit_score
    ablation_accuracy = 0.8

    def fixture(_effective, _seed, task, xor_ablate_event_index=None):
        common = {
            "balanced_accuracy": 1.0,
            "log_loss": 0.1,
            "baseline_log_loss": 1.0,
            "log_loss_gain": 0.9,
            "mean_true_probability": 0.9,
        }
        if task == "xor" and xor_ablate_event_index == 1:
            return {**common, "balanced_accuracy": ablation_accuracy}
        if task == "zero_revival":
            return {**common, "mean_revived_state_probability": 0.9}
        return common

    capacity._fit_score = fixture
    try:
        failed = capacity._candidate_controls(manifest, row, 1103, 200)
        ablation_accuracy = 0.5
        passed = capacity._candidate_controls(manifest, row, 1103, 200)
    finally:
        capacity._fit_score = original
    assert failed["pass"] is False
    assert failed["margins"]["xor_one_bit_ablation"] < 0
    assert passed["pass"] is True
    assert passed["xor_one_bit_ablation"]["balanced_accuracy"] == .5


def _capacity_effective(spec: TraceSpec, seed: int, updates: int = 800) -> dict[str, object]:
    return {
        "event_radius_cM": .2,
        "evidence_scale": 1.0,
        "hidden_dim": spec.hidden_dim,
        "depth": spec.depth,
        "kernel_size": spec.kernel_size,
        "dropout": spec.dropout,
        "dilations": list(spec.dilations),
        "updates": updates,
        "learning_rate": 1e-3,
        "batch_people": 8,
        "marker_shard": MARKERS,
        "validation_every": 25,
        "early_stopping_patience": 4,
        "tune_fraction": .2,
        "split_seed": 3401103,
        "seed": seed,
    }


def test_capacity_xor_learns_directional_two_locus_rule_in_two_of_three_seeds() -> None:
    spec = TraceSpec(32, 4, 5, 0.0, (1, 2, 4, 8))
    scores = [
        capacity._fit_score(_capacity_effective(spec, seed), seed, "xor")[
            "balanced_accuracy"
        ]
        for seed in capacity.REPLICATION_SEEDS
    ]
    assert sum(score >= .75 for score in scores) >= 2, scores


def test_capacity_xor_stays_at_chance_when_one_bit_is_ablated() -> None:
    spec = TraceSpec(32, 4, 5, 0.0, (1, 2, 4, 8))
    scores = [
        capacity._fit_score(
            _capacity_effective(spec, seed), seed, "xor",
            xor_ablate_event_index=1,
        )[
            "balanced_accuracy"
        ]
        for seed in capacity.REPLICATION_SEEDS
    ]
    assert sum(abs(score - .5) <= .10 for score in scores) >= 2, scores


def test_capacity_xor_is_learnable_by_an_oracle_that_observes_both_bits() -> None:
    observed = capacity._xor_fixture_oracle()
    assert observed["status"] == "PASS_ENCODED_TWO_BIT_ORACLE"
    assert observed["pass"] is True
    assert observed["balanced_accuracy"] == 1.0


def test_capacity_loss_mask_is_private_and_default_training_path_is_unchanged() -> None:
    patterns = np.tile(np.asarray([0, 1], dtype=np.int64), 8)
    features = capacity._features(patterns, "default-path", .2, "additive")
    truth = capacity._labels(patterns, .2, "additive")
    arguments = (
        features, features, truth, "tcn", 12.0, 1.0,
        TraceSpec(32, 2, 3, 0.0, (1, 2)), 20, 1e-3,
        8, MARKERS, 25, 4, 1103,
    )
    default_prediction, _ = train(
        *arguments, tune_fraction=.2, split_seed=3401103,
        event_radius_cm=.2, early_stopping=False, restore_best=False,
    )
    all_markers_prediction, _ = train(
        *arguments, tune_fraction=.2, split_seed=3401103,
        event_radius_cm=.2, early_stopping=False, restore_best=False,
        _capacity_loss_marker_mask=np.ones(
            features["baseline_states"].shape[1], dtype=bool,
        ),
    )
    assert np.array_equal(default_prediction, all_markers_prediction)
    root = Path(__file__).resolve().parents[1]
    production_paths = (
        root / "bin/m37_trace_compact_sweep.py",
        root / "modules/37_TRACE_COMPACT_SWEEP.nf",
        root / "workflows/m37_trace_compact_sweep.nf",
        root / "conf/m37_trace_compact_sweep.config",
        root / "conf/m37_r0_compact_sweep.config",
    )
    assert all(
        "_capacity_loss_marker_mask" not in path.read_text(encoding="utf-8")
        and "capacity-loss-marker-mask" not in path.read_text(encoding="utf-8")
        for path in production_paths
    )
