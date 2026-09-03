#!/usr/bin/env python3
"""Evaluate M37 TCN capacity on sealed synthetic fixtures.

Every declared TCN candidate is evaluated with the fixed seeds 1103, 2207 and
3301.  Within each candidate/seed pair, training follows the prospective ladder
200, 400, 800 and 1600 updates and stops at the first rung that passes all three
controls.  The fixture is only a capacity gate: it never ranks candidates.
Real FIT/TUNE features are not an input to either stage and cannot be opened
before a candidate passes additive evidence, local XOR, the corresponding
one-bit ablation and structural-zero revival in at least two of three seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from m37_trace_core import PROBABILITY_FLOOR, TraceSpec, hmm_posterior, require
from m37_trace_train import authenticate_feature_pair, train


SCREEN_SEED = 1103
REPLICATION_SEEDS = (1103, 2207, 3301)
BUDGET_LADDER = (200, 400, 800, 1600)
FIXTURE_MARKERS = 9
EVENT_CHANNELS = 23
RUNG_EXECUTION = "DETERMINISTIC_RESTART_SAME_SEED_EACH_RUNG"
RUNG_TRAINING_POLICY = (
    "EXACT_REQUESTED_UPDATES_FINAL_STATE_NO_EARLY_STOPPING_NO_BEST_CHECKPOINT_RESTORE"
)
CONTROL_THRESHOLDS = {
    "hmm_additive_maximum_posterior_change": 1e-4,
    "additive_balanced_accuracy": 0.80,
    "xor_balanced_accuracy": 0.75,
    "xor_one_bit_ablation_maximum_distance_from_chance": 0.10,
    "zero_revival_mean_probability": 0.50,
    "zero_revival_log_loss_gain": 0.50,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path.name}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_axis(radius_cm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the same dimensionless fixture geometry for every event radius."""
    require(radius_cm > 0, "capacity fixture radius must be positive")
    marker_cm = np.linspace(0.0, 4.0 * radius_cm, FIXTURE_MARKERS, dtype=np.float64)
    centre = 2.0 * radius_cm
    support = np.abs(marker_cm - centre) < radius_cm
    require(int(support.sum()) >= 3, "capacity fixture support is too small")
    return marker_cm, support, centre


def _fixture_support(radius_cm: float, task: str) -> np.ndarray:
    """Markers used jointly by the capacity-only loss and score."""
    marker_cm, additive_support, centre = _fixture_axis(radius_cm)
    if task != "xor":
        return additive_support
    event_centres = (centre - radius_cm, centre + radius_cm)
    # Score the left event anchor, which has positive gate mass and a fixed
    # direction to the second event.  This is a minimal test of two-locus
    # nonlinear capacity, not a claim about bidirectional genomic range.
    support = np.zeros(len(marker_cm), dtype=bool)
    support[int(np.argmin(np.abs(marker_cm - event_centres[0])))] = True
    centre_marker = int(np.argmin(np.abs(marker_cm - centre)))
    require(
        not bool(support[centre_marker]) and int(support.sum()) == 1,
        "XOR capacity support must be one positive-mass directional anchor",
    )
    return support


def _features(patterns: np.ndarray, prefix: str, radius_cm: float, task: str,
              disable_values: bool = False,
              xor_ablate_event_index: int | None = None) -> dict[str, np.ndarray]:
    require(task in {"additive", "xor", "zero_revival"}, "capacity task differs")
    require(
        xor_ablate_event_index in {None, 0, 1} and
        (task == "xor" or xor_ablate_event_index is None),
        "capacity XOR ablation differs",
    )
    marker_cm, _, centre = _fixture_axis(radius_cm)
    support = _fixture_support(radius_cm, task)
    people = len(patterns)
    event_values: list[np.ndarray] = []
    event_context: list[int] = []
    event_samples: list[int] = []
    event_cm: list[float] = []
    event_anchor: list[int] = []
    for person, pattern in enumerate(patterns):
        def base_event() -> np.ndarray:
            value = np.zeros(EVENT_CHANNELS, dtype=np.float32)
            # Match the materialized encoder schema exactly: one-hot genotype
            # (0:4), ancestry log likelihood (4:10), uncertainty (10:13),
            # reference support (13:16), callability (16:18), carrier/origin
            # support (18:20), and availability flags (20:23).
            value[0] = 1.0
            value[16:18] = 1.0
            value[20:23] = 1.0
            return value

        def encode_contrast(value: np.ndarray, bit: int, channel: int) -> None:
            # Synthetic centred log-likelihood contrasts occupy the real
            # evidence slice (4:10).  Unlike raw genotype one-hot channels,
            # they exercise each candidate's frozen evidence_scale.
            value[channel] = -1.0 if bit == 0 else 1.0

        if task == "xor":
            # Put one bit in each of two neighbouring loci.  Neither event
            # alone predicts XOR, and their positive triangular splats do not
            # overlap; the TCN must integrate their local spatial field.
            bits = (int(pattern) // 2, int(pattern) % 2)
            centres = (centre - radius_cm, centre + radius_cm)
            for event_index, (bit, event_centre) in enumerate(zip(bits, centres)):
                value = base_event()
                if not disable_values and event_index != xor_ablate_event_index:
                    value[4 + event_index] = float(bit)
                event_values.append(value)
                event_context.append(0 if disable_values else 1)
                event_samples.append(person)
                event_cm.append(event_centre)
                event_anchor.append(int(np.argmin(np.abs(marker_cm - event_centre))))
        else:
            value = base_event()
            if not disable_values:
                encode_contrast(value, int(pattern), 4)
            event_values.append(value)
            event_context.append(0 if disable_values else 1)
            event_samples.append(person)
            event_cm.append(centre)
            event_anchor.append(int(np.argmin(np.abs(marker_cm - centre))))
    events = len(event_samples)
    baseline = np.full(
        (people, FIXTURE_MARKERS, 6), np.float32(1.0 / 6.0), dtype=np.float32,
    )
    if task == "zero_revival":
        # The positive class is state NN (index 5), to which F0 assigns an
        # exact zero.  The other five states share the mass uniformly, avoiding
        # an unrelated chromosome-wide AA preference in this synthetic gate.
        baseline[:] = np.asarray([.2, .2, .2, .2, .2, 0.], dtype=np.float32)
    schedule_indices = (
        np.arange(0, events, 2, dtype=np.int64)
        if task == "xor" else np.arange(events, dtype=np.int64)
    )
    return {
        "baseline_states": baseline,
        "evidence_field": np.zeros_like(baseline),
        "marker_cM": marker_cm,
        "marker_pos": np.arange(FIXTURE_MARKERS, dtype=np.int64),
        "marker_axis_sha256": np.asarray([f"m37-capacity-{radius_cm:.8g}"]),
        "sample_key_sha256": np.asarray(
            [f"{prefix}-{index:03d}".encode("ascii") for index in range(people)], dtype="S64",
        ),
        "event_values": np.asarray(event_values, dtype=np.float32),
        "event_context_7mer": np.asarray(event_context, dtype=np.uint16),
        "event_sample": np.asarray(event_samples, dtype=np.uint32),
        "event_cM": np.asarray(event_cm, dtype=np.float64),
        "event_marker_left": np.asarray(event_anchor, dtype=np.uint32),
        "event_marker_right": np.asarray(event_anchor, dtype=np.uint32),
        "event_delta_left_cM": np.zeros(events, dtype=np.float32),
        "event_delta_right_cM": np.zeros(events, dtype=np.float32),
        "schedule_sample": np.asarray(event_samples, dtype=np.uint32)[schedule_indices],
        "schedule_marker": np.asarray(event_anchor, dtype=np.uint32)[schedule_indices],
        "state_names": np.asarray(["AA", "AE", "AN", "EE", "EN", "NN"]),
    }


def _labels(patterns: np.ndarray, radius_cm: float, task: str) -> np.ndarray:
    support = _fixture_support(radius_cm, task)
    if task == "xor":
        state = ((patterns // 2) ^ (patterns % 2)) * 3
    elif task == "zero_revival":
        state = patterns * 5
    else:
        state = patterns * 3
    result = np.zeros((len(patterns), FIXTURE_MARKERS), dtype=np.uint8)
    result[:, support] = state[:, None]
    return result


def _score(probability: np.ndarray, truth: np.ndarray, radius_cm: float, task: str,
           revived_state: int | None = None) -> dict[str, float]:
    support = _fixture_support(radius_cm, task)
    prediction = probability[:, support]
    observed = truth[:, support]
    called = prediction.argmax(axis=2).reshape(-1)
    flattened = observed.reshape(-1)
    balanced_accuracy = float(np.mean([
        (called[flattened == value] == value).mean() for value in np.unique(flattened)
    ]))
    index = np.arange(flattened.size)
    selected = prediction.reshape(-1, 6)[index, flattened]
    result = {
        "balanced_accuracy": balanced_accuracy,
        "log_loss": float(-np.log(selected + PROBABILITY_FLOOR).mean()),
        "mean_true_probability": float(selected.mean()),
    }
    if revived_state is not None:
        mask = flattened == revived_state
        require(bool(mask.any()), "zero-revival fixture lacks its target state")
        result["mean_revived_state_probability"] = float(
            prediction.reshape(-1, 6)[mask, revived_state].mean(),
        )
    return result


def _effective_tcn(manifest: dict[str, Any], row: dict[str, Any], seed: int,
                   updates: int
                   ) -> dict[str, Any]:
    effective = {
        **manifest["execution"],
        **manifest["families"]["tcn"],
        **row.get("parameters", {}),
        "seed": seed,
        "updates": updates,
    }
    require(int(effective["updates"]) in BUDGET_LADDER and
            int(effective["batch_people"]) > 0 and
            int(effective["validation_every"]) > 0 and
            0 < float(effective["event_radius_cM"]) <= 0.5,
            "capacity candidate execution parameters differ")
    return effective


def _fit_score(effective: dict[str, Any], seed: int, task: str,
               xor_ablate_event_index: int | None = None) -> dict[str, Any]:
    radius_cm = float(effective["event_radius_cM"])
    fit_pattern = (np.tile(np.asarray([0, 1], dtype=np.int64), 32) if task != "xor"
                   else np.tile(np.arange(4, dtype=np.int64), 24))
    valid_pattern = (np.tile(np.asarray([1, 0], dtype=np.int64), 12) if task != "xor"
                     else np.tile(np.asarray([3, 0, 2, 1], dtype=np.int64), 8))
    fit = _features(
        fit_pattern, f"fit-{task}-{seed}", radius_cm, task,
        xor_ablate_event_index=xor_ablate_event_index,
    )
    valid = _features(
        valid_pattern, f"valid-{task}-{seed}", radius_cm, task,
        xor_ablate_event_index=xor_ablate_event_index,
    )
    fit_truth = _labels(fit_pattern, radius_cm, task)
    valid_truth = _labels(valid_pattern, radius_cm, task)
    loss_score_support = _fixture_support(radius_cm, task)
    if task == "xor":
        require(
            np.array_equal(np.bincount(fit_pattern, minlength=4), np.full(4, 24)) and
            np.array_equal(np.bincount(valid_pattern, minlength=4), np.full(4, 8)),
            "XOR capacity patterns are not balanced across 00/01/10/11",
        )
    require(authenticate_feature_pair(fit, valid) == "SEALED_VALID",
            "capacity-control FIT/VALID people are not disjoint")
    training_diagnostics: dict[str, int | bool | None] = {}
    probability, _ = train(
        fit, valid, fit_truth, "tcn", 12.0, float(effective["evidence_scale"]),
        TraceSpec(
            int(effective["hidden_dim"]), int(effective["depth"]),
            int(effective["kernel_size"]), float(effective["dropout"]),
            tuple(int(value) for value in effective["dilations"]),
        ),
        int(effective["updates"]), float(effective["learning_rate"]),
        int(effective["batch_people"]), FIXTURE_MARKERS,
        int(effective["validation_every"]), int(effective["early_stopping_patience"]),
        seed, tune_fraction=float(effective["tune_fraction"]),
        split_seed=int(effective["split_seed"]), event_radius_cm=radius_cm,
        training_diagnostics=training_diagnostics,
        early_stopping=False,
        restore_best=False,
        _capacity_loss_marker_mask=loss_score_support,
    )
    require(
        training_diagnostics.get("requested_updates") == int(effective["updates"])
        and training_diagnostics.get("completed_updates") == int(effective["updates"])
        and training_diagnostics.get("best_checkpoint_update") is None
        and training_diagnostics.get("early_stopping_enabled") is False
        and training_diagnostics.get("restore_best_checkpoint") is False
        and training_diagnostics.get("capacity_loss_marker_count") ==
        int(loss_score_support.sum()),
        "capacity-control rung did not execute its exact fixed budget",
    )
    observed = _score(
        probability, valid_truth, radius_cm, task,
        revived_state=5 if task == "zero_revival" else None,
    )
    baseline = _score(
        valid["baseline_states"], valid_truth, radius_cm, task,
        revived_state=5 if task == "zero_revival" else None,
    )
    observed["baseline_balanced_accuracy"] = baseline["balanced_accuracy"]
    observed["baseline_log_loss"] = baseline["log_loss"]
    observed["log_loss_gain"] = baseline["log_loss"] - observed["log_loss"]
    observed["training_diagnostics"] = training_diagnostics
    observed["loss_score_marker_count"] = int(loss_score_support.sum())
    observed["loss_score_marker_mask_sha256"] = hashlib.sha256(
        np.ascontiguousarray(loss_score_support, dtype=np.uint8).tobytes(),
    ).hexdigest()
    return observed


def _candidate_controls(manifest: dict[str, Any], row: dict[str, Any], seed: int,
                        updates: int
                        ) -> dict[str, Any]:
    effective = _effective_tcn(manifest, row, seed, updates)
    additive = _fit_score(effective, seed, "additive")
    xor = _fit_score(effective, seed, "xor")
    xor_ablation = _fit_score(
        effective, seed, "xor", xor_ablate_event_index=1,
    )
    revival = _fit_score(effective, seed, "zero_revival")
    margins = {
        "additive": additive["balanced_accuracy"] -
                    CONTROL_THRESHOLDS["additive_balanced_accuracy"],
        "xor": xor["balanced_accuracy"] -
             CONTROL_THRESHOLDS["xor_balanced_accuracy"],
        "xor_one_bit_ablation": (
            CONTROL_THRESHOLDS[
                "xor_one_bit_ablation_maximum_distance_from_chance"
            ] - abs(xor_ablation["balanced_accuracy"] - 0.5)
        ),
        "zero_revival_probability": revival["mean_revived_state_probability"] -
                                    CONTROL_THRESHOLDS["zero_revival_mean_probability"],
        "zero_revival_log_loss": revival["log_loss_gain"] -
                                 CONTROL_THRESHOLDS["zero_revival_log_loss_gain"],
    }
    relative_nll_gain = {
        name: values["log_loss_gain"] / values["baseline_log_loss"]
        for name, values in (("additive", additive), ("xor", xor),
                             ("zero_revival", revival))
    }
    passed = all(value >= 0.0 for value in margins.values())
    return {
        "seed": seed,
        "status": "PASS_ALL_CAPACITY_CONTROLS" if passed else "FAIL_CAPACITY_CONTROL",
        "pass": passed,
        "weakest_margin": float(min(margins.values())),
        "capacity_score": float(min(relative_nll_gain.values())),
        "margins": margins,
        "relative_log_loss_gain": relative_nll_gain,
        "additive": additive,
        "xor_interaction": xor,
        "xor_one_bit_ablation": xor_ablation,
        "zero_revival": revival,
        "effective_hyperparameters": effective,
    }


def _candidate_ladder(manifest: dict[str, Any], row: dict[str, Any], seed: int
                      ) -> dict[str, Any]:
    """Stop a candidate/seed pair at its first passing capacity rung."""
    rung_results: dict[str, Any] = {}
    first_pass_updates: int | None = None
    for updates in BUDGET_LADDER:
        observed = _candidate_controls(manifest, row, seed, updates)
        rung_results[str(updates)] = observed
        if observed["pass"]:
            first_pass_updates = updates
            break
    final = rung_results[str(first_pass_updates or BUDGET_LADDER[-1])]
    return {
        "seed": seed,
        "status": ("PASS_CAPACITY_AT_FIRST_SUCCESSFUL_RUNG" if first_pass_updates
                   else "FAIL_CAPACITY_AT_MAXIMUM_RUNG"),
        "pass": first_pass_updates is not None,
        "first_pass_updates": first_pass_updates,
        "evaluated_updates": [int(value) for value in rung_results],
        "stopped_after_first_pass": first_pass_updates is not None,
        "final_capacity_score": float(final["capacity_score"]),
        "final_weakest_margin": float(final["weakest_margin"]),
        "rung_results": rung_results,
    }


def _hmm_candidate_control(manifest: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    effective = {
        **manifest["execution"],
        **manifest["families"]["hmm"],
        **row.get("parameters", {}),
    }
    baseline = np.full((2, 9, 6), 1 / 6, dtype=np.float32)
    evidence = np.zeros_like(baseline)
    evidence[:, 3:6, 3] = 1.0
    marker_cm = np.arange(9, dtype=float) / 10
    on = hmm_posterior(
        baseline, evidence, marker_cm,
        float(effective["hazard_per_morgan"]), float(effective["evidence_scale"]),
    )
    off = hmm_posterior(
        baseline, np.zeros_like(evidence), marker_cm,
        float(effective["hazard_per_morgan"]), float(effective["evidence_scale"]),
    )
    change = float(np.max(np.abs(on - off)))
    passed = change > CONTROL_THRESHOLDS["hmm_additive_maximum_posterior_change"]
    return {
        "status": ("PASS_ADDITIVE_DETECTABILITY" if passed else
                   "FAIL_ADDITIVE_DETECTABILITY"),
        "pass": passed,
        "maximum_posterior_change": change,
        "threshold_strictly_greater_than":
            CONTROL_THRESHOLDS["hmm_additive_maximum_posterior_change"],
        "effective_hyperparameters": {
            "hazard_per_morgan": float(effective["hazard_per_morgan"]),
            "evidence_scale": float(effective["evidence_scale"]),
        },
        "interaction_control": "NOT_APPLICABLE_NONLEARNED_ADDITIVE_EVIDENCE_MODEL",
    }


def _hmm_controls(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in manifest.get("candidates", []) if row.get("family") == "hmm"]
    candidate_ids = [str(row.get("candidate_id", "")) for row in rows]
    require(rows and all(candidate_ids) and len(candidate_ids) == len(set(candidate_ids)),
            "HMM capacity candidate identities differ")
    controls = {
        candidate_id: _hmm_candidate_control(manifest, row)
        for candidate_id, row in zip(candidate_ids, rows)
    }
    passed = all(row["pass"] for row in controls.values())
    return {
        "status": ("PASS_ALL_CANDIDATES_ADDITIVE_DETECTABILITY" if passed else
                   "FAIL_AT_LEAST_ONE_CANDIDATE_ADDITIVE_DETECTABILITY"),
        "candidate_count": len(controls),
        "candidates": controls,
        "all_candidates_pass": passed,
        "interaction_control": "NOT_APPLICABLE_NONLEARNED_ADDITIVE_EVIDENCE_MODEL",
    }


def _tcn_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in manifest.get("candidates", []) if row.get("family") == "tcn"]
    identifiers = [str(row.get("candidate_id", "")) for row in rows]
    require(rows and all(identifiers) and len(identifiers) == len(set(identifiers)),
            "capacity screen TCN candidate identities differ")
    return rows


def _xor_fixture_oracle() -> dict[str, Any]:
    """Verify the encoded fixture and labels with an independent tiny MLP."""
    import torch

    torch.manual_seed(SCREEN_SEED)
    fit_pattern = np.tile(np.arange(4, dtype=np.int64), 32)
    valid_pattern = np.tile(np.asarray([3, 0, 2, 1], dtype=np.int64), 8)
    fit = _features(fit_pattern, "oracle-fit", 0.2, "xor")
    valid = _features(valid_pattern, "oracle-valid", 0.2, "xor")
    require(
        authenticate_feature_pair(fit, valid) == "SEALED_VALID",
        "capacity oracle FIT/VALID people are not disjoint",
    )

    def encoded_bits(features: dict[str, np.ndarray]) -> np.ndarray:
        people = len(features["sample_key_sha256"])
        values = np.asarray(features["event_values"], dtype=np.float32)
        require(values.shape == (2 * people, EVENT_CHANNELS),
                "capacity oracle event axes differ")
        paired = values.reshape(people, 2, EVENT_CHANNELS)
        return np.column_stack((paired[:, 0, 4], paired[:, 1, 5])).astype(
            np.float32, copy=False,
        )

    def encoded_truth(patterns: np.ndarray) -> np.ndarray:
        labels = _labels(patterns, 0.2, "xor")
        support = _fixture_support(0.2, "xor")
        require(int(support.sum()) == 1, "capacity oracle support differs")
        states = labels[:, support].reshape(-1)
        require(set(np.unique(states).tolist()) == {0, 3},
                "capacity oracle states differ")
        return (states == 3).astype(np.int64)

    fit_bits, valid_bits = encoded_bits(fit), encoded_bits(valid)
    fit_truth, valid_truth = encoded_truth(fit_pattern), encoded_truth(valid_pattern)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.GELU(), torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    tensor_bits = torch.from_numpy(fit_bits)
    tensor_truth = torch.from_numpy(fit_truth)
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(tensor_bits), tensor_truth)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        called = model(torch.from_numpy(valid_bits)).argmax(dim=1).numpy()
    balanced_accuracy = float(np.mean([
        (called[valid_truth == value] == value).mean()
        for value in np.unique(valid_truth)
    ]))
    passed = balanced_accuracy == 1.0
    return {
        "status": "PASS_ENCODED_TWO_BIT_ORACLE" if passed else
                  "FAIL_ENCODED_TWO_BIT_ORACLE",
        "pass": passed,
        "balanced_accuracy": balanced_accuracy,
        "required_balanced_accuracy": 1.0,
        "inputs": "event_values channels 4 and 5 generated by the sealed fixture",
        "truth": "diploid state generated by the sealed fixture at its score anchor",
        "seed": SCREEN_SEED,
    }


def run_screen(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record the first fixed-seed capacity replicate for every candidate."""
    fixture_oracle = _xor_fixture_oracle()
    require(fixture_oracle["pass"], "encoded XOR fixture oracle did not pass")
    candidates = {
        str(row["candidate_id"]): _candidate_ladder(manifest, row, SCREEN_SEED)
        for row in _tcn_rows(manifest)
    }
    candidate_ids = sorted(candidates)
    screen = {
        "screen_seed": SCREEN_SEED,
        "candidate_count": len(candidates),
        "budget_ladder_updates": list(BUDGET_LADDER),
        "rung_execution": RUNG_EXECUTION,
        "rung_training_policy": RUNG_TRAINING_POLICY,
        "thresholds": CONTROL_THRESHOLDS,
        "capacity_score": "minimum relative held-out log-loss gain across additive, XOR and zero-revival tasks",
        "fixture_oracle": fixture_oracle,
        "candidates": candidates,
    }
    selection = {
        "screen_seed": SCREEN_SEED,
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "selection_uses_real_fit_or_tune_metrics": False,
        "selection_uses_fixture_scores": False,
        "selection_rule": "all declared TCN candidates advance to the three-seed capacity gate; no candidate ranking",
    }
    return screen, selection


def run_replication(manifest: dict[str, Any], screen: dict[str, Any],
                    selection: dict[str, Any]) -> dict[str, Any]:
    """Complete all three fixed-seed replicates for every declared candidate."""
    rows = {str(row["candidate_id"]): row for row in _tcn_rows(manifest)}
    candidate_ids = selection.get("candidate_ids")
    expected_ids = sorted(rows)
    require(screen.get("screen_seed") == SCREEN_SEED and
            screen.get("candidate_count") == len(expected_ids) and
            screen.get("budget_ladder_updates") == list(BUDGET_LADDER) and
            screen.get("rung_execution") ==
            RUNG_EXECUTION and
            screen.get("rung_training_policy") == RUNG_TRAINING_POLICY and
            screen.get("thresholds") == CONTROL_THRESHOLDS and
            screen.get("fixture_oracle", {}).get("pass") is True and
            isinstance(candidate_ids, list) and candidate_ids == expected_ids and
            selection.get("screen_seed") == SCREEN_SEED and
            selection.get("candidate_count") == len(expected_ids) and
            selection.get("selection_uses_real_fit_or_tune_metrics") is False and
            selection.get("selection_uses_fixture_scores") is False and
            sorted(screen.get("candidates", {})) == expected_ids,
            "capacity candidate roster differs")
    replicated: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        seed_results: dict[str, Any] = {
            str(SCREEN_SEED): screen["candidates"][candidate_id],
        }
        for seed in REPLICATION_SEEDS:
            if seed != SCREEN_SEED:
                seed_results[str(seed)] = _candidate_ladder(manifest, rows[candidate_id], seed)
        require(set(seed_results) == {str(seed) for seed in REPLICATION_SEEDS},
                "capacity replication seeds differ")
        pass_count = sum(bool(row["pass"]) for row in seed_results.values())
        passing_updates = sorted(
            int(result["first_pass_updates"])
            for result in seed_results.values() if result["pass"]
        )
        # The effective real-data budget is the smallest rung at which the
        # prespecified 2/3-seed gate has been met.  It is fixed once from the
        # synthetic fixture and is then shared by all five real-data arms.
        effective_updates = passing_updates[1] if pass_count >= 2 else None
        replicated[candidate_id] = {
            "status": ("PASS_CAPACITY_2_OF_3" if pass_count >= 2 else
                       "FAIL_CAPACITY_2_OF_3"),
            "pass_count": pass_count,
            "required_pass_count": 2,
            "effective_updates": effective_updates,
            "effective_updates_rule": "second-smallest first-pass rung; smallest ladder budget meeting the 2-of-3 seed gate",
            "seeds": list(REPLICATION_SEEDS),
            "seed_results": seed_results,
            "aggregation": {
                "selection_of_best_seed": "FORBIDDEN",
                "capacity_score_mean": float(np.mean([
                    row["final_capacity_score"] for row in seed_results.values()
                ])),
                "capacity_score_median": float(np.median([
                    row["final_capacity_score"] for row in seed_results.values()
                ])),
                "weakest_margin_mean": float(np.mean([
                    row["final_weakest_margin"] for row in seed_results.values()
                ])),
                "weakest_margin_median": float(np.median([
                    row["final_weakest_margin"] for row in seed_results.values()
                ])),
            },
        }
    eligible = sorted(
        candidate_id for candidate_id, row in replicated.items()
        if row["status"] == "PASS_CAPACITY_2_OF_3"
    )
    return {
        "hmm": _hmm_controls(manifest),
        "tcn": {
            "status": ("PASS_AT_LEAST_ONE_CANDIDATE" if eligible else
                       "FAIL_NO_CAPABLE_CANDIDATE"),
            "screen_seed": SCREEN_SEED,
            "replication_seeds": list(REPLICATION_SEEDS),
            "budget_ladder_updates": list(BUDGET_LADDER),
            "rung_execution": RUNG_EXECUTION,
            "rung_training_policy": RUNG_TRAINING_POLICY,
            "candidate_count": len(candidate_ids),
            "evaluated_candidate_ids": candidate_ids,
            "eligible_candidate_ids": eligible,
            "candidates": replicated,
            "selection_of_best_candidate": "FORBIDDEN",
            "selection_of_best_seed": "FORBIDDEN",
            "scientific_closure_if_failed": "FORBIDDEN",
            "screen_seed_reuse": (
                "seed 1103 is the first fixed replicate and one of three reported "
                "replications for every candidate; no candidate is ranked by it"
            ),
        },
    }


def _validate_capacity_contract(manifest: dict[str, Any], amendment: dict[str, Any]) -> None:
    capacity = amendment.get("capacity_control")
    require(isinstance(capacity, dict) and
            capacity.get("thresholds") == CONTROL_THRESHOLDS and
            capacity.get("screen_seed") == SCREEN_SEED and
            capacity.get("screen_candidate_count") == len(_tcn_rows(manifest)) and
            capacity.get("budget_ladder_updates") == list(BUDGET_LADDER) and
            capacity.get("rung_execution") ==
            RUNG_EXECUTION and
            capacity.get("rung_training_policy") == RUNG_TRAINING_POLICY and
            int(manifest.get("execution", {}).get("updates", -1)) == BUDGET_LADDER[0] and
            manifest.get("positive_control_status", {}).get("tcn", {}).get(
                "rung_training_policy"
            ) == RUNG_TRAINING_POLICY and
            capacity.get("candidate_evaluation") ==
            "ALL_DECLARED_CANDIDATES_ACROSS_ALL_FIXED_SEEDS_NO_RANKING" and
            capacity.get("replication_seeds") == list(REPLICATION_SEEDS) and
            capacity.get("valid_access") == "FORBIDDEN" and
            amendment.get("candidate_count_by_family", {}).get("hmm") ==
            len([row for row in manifest.get("candidates", []) if row.get("family") == "hmm"]) and
            amendment.get("candidate_count_by_family", {}).get("tcn") ==
            len(_tcn_rows(manifest)),
            "M37 capacity-control contract differs")


def _validate_upstream_artifact(
    artifact: dict[str, Any], receipt: dict[str, Any], *, stage: str,
    run_id: str, container_digest: str, output_sha256: str,
    authenticated_sources: dict[str, str],
) -> None:
    require(
        artifact.get("stage") == receipt.get("stage") == stage and
        artifact.get("run_id") == receipt.get("run_id") == run_id and
        artifact.get("container_digest") == receipt.get("container_digest") ==
        container_digest and
        artifact.get("authenticated_source_sha256") ==
        receipt.get("authenticated_source_sha256") == authenticated_sources and
        receipt.get("output_sha256") == output_sha256 and
        artifact.get("truth_or_real_features_opened") is False,
        f"{stage} artifact/receipt provenance differs",
    )


def _receipt(stage: str, run_id: str, container_digest: str,
             authenticated_sources: dict[str, str], output: Path,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0", "stage": stage, "run_id": run_id,
        "container_digest": container_digest,
        "authenticated_source_sha256": authenticated_sources,
        **(extra or {}),
        "output_sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "replicate"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--contract-amendment", type=Path, required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--auth-file", action="append", type=Path, required=True)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--screen-receipt", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not args.output.exists() and "@sha256:" in args.container_digest,
            "M37 capacity-control output/container differs")
    manifest = _json(args.candidate_manifest)
    amendment = _json(args.contract_amendment)
    require(manifest.get("contract_binding", {}).get("parent_sha256") ==
            sha256(args.parent_contract) and
            manifest.get("contract_binding", {}).get("amendment_sha256") ==
            sha256(args.contract_amendment) and
            amendment.get("capacity_control", {}).get("screen_seed") == SCREEN_SEED and
            amendment.get("capacity_control", {}).get("replication_seeds") ==
            list(REPLICATION_SEEDS),
            "M37 capacity-control contract binding differs")
    _validate_capacity_contract(manifest, amendment)
    authenticated_sources = {path.name: sha256(path) for path in args.auth_file}
    require(len(authenticated_sources) == len(args.auth_file),
            "M37 capacity-control authenticated source basenames collide")
    started = time.perf_counter()
    if args.phase == "screen":
        require(args.selection_output is not None and
                all(value is None for value in (args.screen, args.screen_receipt,
                                                args.selection, args.selection_receipt)),
                "capacity screen invocation differs")
        screen, selection = run_screen(manifest)
        common = {
            "schema_version": "1.0.0", "run_id": args.run_id,
            "candidate_manifest_sha256": sha256(args.candidate_manifest),
            "parent_contract_sha256": sha256(args.parent_contract),
            "contract_amendment_sha256": sha256(args.contract_amendment),
            "container_digest": args.container_digest,
            "authenticated_source_sha256": authenticated_sources,
            "truth_or_real_features_opened": False,
        }
        _write_json(args.output, {
            **common, "stage": "M37_TRACE_CAPACITY_SCREEN",
            "screen": screen,
        })
        _write_json(args.selection_output, {
            **common, "stage": "M37_TRACE_CAPACITY_SELECTION",
            "screen_sha256": sha256(args.output), "selection": selection,
        })
        _write_json(args.output.with_suffix(".receipt.json"), _receipt(
            "M37_TRACE_CAPACITY_SCREEN", args.run_id, args.container_digest,
            authenticated_sources, args.output,
        ))
        _write_json(args.selection_output.with_suffix(".receipt.json"), _receipt(
            "M37_TRACE_CAPACITY_SELECTION", args.run_id, args.container_digest,
            authenticated_sources, args.selection_output,
            {"screen_sha256": sha256(args.output)},
        ))
    else:
        require(args.selection_output is None and
                all(value is not None for value in (args.screen, args.screen_receipt,
                                                    args.selection, args.selection_receipt)),
                "capacity replication invocation differs")
        screen_artifact, selection_artifact = _json(args.screen), _json(args.selection)
        screen_receipt, selection_receipt = _json(args.screen_receipt), _json(args.selection_receipt)
        _validate_upstream_artifact(
            screen_artifact, screen_receipt, stage="M37_TRACE_CAPACITY_SCREEN",
            run_id=args.run_id, container_digest=args.container_digest,
            output_sha256=sha256(args.screen),
            authenticated_sources=authenticated_sources,
        )
        _validate_upstream_artifact(
            selection_artifact, selection_receipt,
            stage="M37_TRACE_CAPACITY_SELECTION", run_id=args.run_id,
            container_digest=args.container_digest,
            output_sha256=sha256(args.selection),
            authenticated_sources=authenticated_sources,
        )
        require(
                selection_artifact.get("screen_sha256") == sha256(args.screen) and
                selection_receipt.get("screen_sha256") == sha256(args.screen) and
                screen_artifact.get("candidate_manifest_sha256") ==
                selection_artifact.get("candidate_manifest_sha256") ==
                sha256(args.candidate_manifest) and
                screen_artifact.get("parent_contract_sha256") ==
                selection_artifact.get("parent_contract_sha256") ==
                sha256(args.parent_contract) and
                screen_artifact.get("contract_amendment_sha256") ==
                selection_artifact.get("contract_amendment_sha256") ==
                sha256(args.contract_amendment),
                "capacity screen/selection provenance differs")
        controls = run_replication(
            manifest, screen_artifact["screen"], selection_artifact["selection"],
        )
        elapsed = time.perf_counter() - started
        _write_json(args.output, {
            "schema_version": "1.0.0", "stage": "M37_TRACE_COMPACT_POSITIVE_CONTROL",
            "run_id": args.run_id,
            "runtime": {
                "elapsed_seconds": elapsed,
                "maximum_resident_set_size_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                ),
            },
            "controls": controls,
            "screen_sha256": sha256(args.screen),
            "screen_receipt_sha256": sha256(args.screen_receipt),
            "selection_sha256": sha256(args.selection),
            "selection_receipt_sha256": sha256(args.selection_receipt),
            "candidate_manifest_sha256": sha256(args.candidate_manifest),
            "parent_contract_sha256": sha256(args.parent_contract),
            "contract_amendment_sha256": sha256(args.contract_amendment),
            "container_digest": args.container_digest,
            "authenticated_source_sha256": authenticated_sources,
            "truth_or_real_features_opened": False,
        })
        _write_json(args.output.with_suffix(".receipt.json"), _receipt(
            "M37_TRACE_COMPACT_POSITIVE_CONTROL", args.run_id,
            args.container_digest, authenticated_sources, args.output,
            {"selection_sha256": sha256(args.selection)},
        ))


if __name__ == "__main__":
    main()
