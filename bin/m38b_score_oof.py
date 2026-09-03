#!/usr/bin/env python3
"""Score paired M38B out-of-fold predictions on the sealed FIT truth.

The statistical unit is a complete synthetic person.  Marker-level losses are
first integrated over genetic distance and are only then averaged or
bootstrapped across people.  No model selection or checkpoint choice is made
in this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m34_parse_flare_truth import sha256_file, write_deterministic_npz
from m37_trace_core import PROBABILITY_FLOOR, STATE_PAIRS, require
from m38b_oof_core import (per_person_log_loss,
                           voronoi_cm_weights as voronoi_cm_widths)


STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")
ANCESTRY_NAMES = ("AFR", "EUR", "NAM")
STATE_DOSAGE = np.zeros((len(STATE_PAIRS), len(ANCESTRY_NAMES)), dtype=np.float64)
for _state, (_left, _right) in enumerate(STATE_PAIRS):
    STATE_DOSAGE[_state, _left] += 1
    STATE_DOSAGE[_state, _right] += 1
BOUNDARY_TOLERANCE_CM = 0.2
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 38_200_103
PER_PERSON_METRICS = (
    "log_loss_cm",
    "log_loss_uniform",
    "brier_cm",
    "ancestry_proportion_mae_macro_cm",
    "ancestry_proportion_mae_nam_cm",
    "ancestry_proportion_mae_nam_truth_present_cm",
    "boundary_f1_0_2cm",
    "boundary_mean_error_0_2cm",
    "false_transitions_per_morgan_0_2cm",
)


def _text_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    require(array.ndim == 1 and len(array) > 0, f"{name} must be a non-empty vector")
    require(array.dtype.kind in "SUiu", f"{name} must contain strings or integers")
    result = np.asarray([
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.tolist()
    ], dtype="U")
    require(len(set(result.tolist())) == len(result), f"{name} values must be unique")
    return result


def _fold_vector(values: np.ndarray, person_count: int) -> np.ndarray:
    array = np.asarray(values)
    require(array.ndim == 1 and len(array) == person_count,
            "fold_ids must have one value per person")
    require(array.dtype.kind in "SUiu", "fold_ids must contain strings or integers")
    return np.asarray([
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.tolist()
    ], dtype="U")


def _person_axis(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    names = [name for name in ("person_ids", "sample_key_sha256") if name in archive.files]
    require(len(names) == 1, "NPZ needs exactly one person axis: person_ids or sample_key_sha256")
    return _text_vector(archive[names[0]], names[0])


def _scalar_text(values: np.ndarray, name: str) -> str:
    array = np.asarray(values).reshape(-1)
    require(array.size == 1, f"{name} must be scalar")
    item = array[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate state probabilities, floor zeros, and restore the simplex."""
    values = np.asarray(probabilities, dtype=np.float64)
    require(values.ndim == 3 and values.shape[2] == len(STATE_NAMES),
            "probabilities must have shape [person, marker, state=6]")
    require(np.isfinite(values).all() and np.all(values >= 0),
            "probabilities must be finite and non-negative")
    row_sums = values.sum(axis=2, keepdims=True)
    require(np.all(row_sums > 0) and np.max(np.abs(row_sums - 1.0)) <= 5e-5,
            "probabilities do not form a simplex")
    values = values / row_sums
    values = np.maximum(values, PROBABILITY_FLOOR)
    return values / values.sum(axis=2, keepdims=True)


def normalised_voronoi_cm_weights(marker_cm: np.ndarray) -> np.ndarray:
    """Normalise the shared M38B genetic widths for integrated metrics."""
    widths = voronoi_cm_widths(marker_cm)
    return widths / widths.sum()


def _states_from_dosage(dosage: np.ndarray) -> np.ndarray:
    values = np.asarray(dosage)
    require(values.ndim == 3 and values.shape[2] == len(ANCESTRY_NAMES),
            "truth dosage must have shape [person, marker, ancestry=3]")
    rounded = np.rint(values).astype(np.int8)
    require(np.isfinite(values).all() and np.allclose(values, rounded, atol=1e-8, rtol=0) and
            np.all((rounded >= 0) & (rounded <= 2)) and
            np.all(rounded.sum(axis=2) == 2),
            "truth dosage must be an exact diploid AFR/EUR/NAM dosage")
    lookup = {tuple(row.astype(int)): index for index, row in enumerate(STATE_DOSAGE)}
    flat = rounded.reshape(-1, 3)
    require(all(tuple(row.astype(int)) in lookup for row in flat),
            "truth dosage contains an unsupported state")
    states = np.asarray(
        [lookup[tuple(row.astype(int))] for row in flat], dtype=np.int8,
    )
    return states.reshape(values.shape[:2])


def _states_from_haplotypes(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    require(values.ndim == 3 and values.shape[1] == 2 and
            np.issubdtype(values.dtype, np.integer) and np.all((values >= 0) & (values < 3)),
            "haplotype truth must have shape [person, haplotype=2, marker]")
    dosage = np.eye(3, dtype=np.int8)[values].sum(axis=1)
    return _states_from_dosage(dosage)


def _truth_states(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    dosage_keys = [name for name in ("truth_dosage", "dosage") if name in archive.files]
    state_keys = [name for name in ("state_labels", "truth_states", "truth") if name in archive.files]
    require(len(dosage_keys) + len(state_keys) <= 1,
            "truth NPZ has more than one state/dosage representation")
    if dosage_keys:
        return _states_from_dosage(archive[dosage_keys[0]])
    if state_keys:
        labels = np.asarray(archive[state_keys[0]])
    else:
        require("labels" in archive.files, "truth NPZ lacks states, dosage, or haplotype labels")
        labels = np.asarray(archive["labels"])
        if labels.ndim == 3:
            return _states_from_haplotypes(labels)
    require(labels.ndim == 2 and np.issubdtype(labels.dtype, np.integer) and
            np.all((labels >= 0) & (labels < len(STATE_NAMES))),
            "truth states must have shape [person, marker] with values 0..5")
    return np.ascontiguousarray(labels, dtype=np.int8)


def boundary_records(labels: np.ndarray, marker_cm: np.ndarray) -> list[tuple[float, int, int]]:
    """Return position and directed before/after state for every transition."""
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return [
        (float((marker_cm[index - 1] + marker_cm[index]) / 2.0),
         int(labels[index - 1]), int(labels[index]))
        for index in changes.tolist()
    ]


def directed_boundary_match(
    truth: Sequence[tuple[float, int, int]],
    predicted: Sequence[tuple[float, int, int]],
    tolerance_cm: float = BOUNDARY_TOLERANCE_CM,
) -> tuple[int, float]:
    """Maximum ordered one-to-one matches, with minimum error as tie-breaker.

    Candidate pairs are limited to the 0.2 cM neighbourhood and combined with
    a Fenwick-tree longest-chain calculation.  This preserves the exact
    dynamic-programming solution without scanning every truth/prediction pair.
    """
    require(tolerance_cm == BOUNDARY_TOLERANCE_CM,
            "M38B boundary matching is fixed at exactly 0.2 cM")
    if not truth or not predicted:
        return 0, 0.0
    predicted_position = np.asarray([row[0] for row in predicted], dtype=np.float64)
    require(np.all(np.diff(predicted_position) >= 0),
            "predicted boundaries must be ordered by cM")
    truth_position = np.asarray([row[0] for row in truth], dtype=np.float64)
    require(np.all(np.diff(truth_position) >= 0),
            "truth boundaries must be ordered by cM")
    tree_count = np.zeros(len(predicted) + 1, dtype=np.int32)
    tree_cost = np.zeros(len(predicted) + 1, dtype=np.float64)

    def better(first: tuple[int, float], second: tuple[int, float]) -> tuple[int, float]:
        return max((first, second), key=lambda value: (value[0], -value[1]))

    def query(exclusive: int) -> tuple[int, float]:
        best = (0, 0.0)
        index = exclusive
        while index > 0:
            best = better(best, (int(tree_count[index]), float(tree_cost[index])))
            index -= index & -index
        return best

    def update(position: int, value: tuple[int, float]) -> None:
        index = position + 1
        while index < len(tree_count):
            best = better((int(tree_count[index]), float(tree_cost[index])), value)
            tree_count[index], tree_cost[index] = best
            index += index & -index

    for truth_row in truth:
        epsilon = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(truth_row[0]))
        left = int(np.searchsorted(
            predicted_position, truth_row[0] - tolerance_cm - epsilon, side="left",
        ))
        right = int(np.searchsorted(
            predicted_position, truth_row[0] + tolerance_cm + epsilon, side="right",
        ))
        pending: list[tuple[int, tuple[int, float]]] = []
        for column in range(left, right):
            predicted_row = predicted[column]
            distance = abs(truth_row[0] - predicted_row[0])
            if (truth_row[1:] == predicted_row[1:] and
                    distance <= tolerance_cm + epsilon):
                count, cost = query(column)
                pending.append((column, (count + 1, cost + distance)))
        # Defer updates so one truth boundary cannot be matched more than once.
        for column, value in pending:
            update(column, value)
    return query(len(predicted))


@dataclass(frozen=True)
class ArmScore:
    summary: dict[str, Any]
    per_person: dict[str, np.ndarray]
    boundary_counts: dict[str, np.ndarray]


def score_arm(probabilities: np.ndarray, truth: np.ndarray,
              marker_cm: np.ndarray) -> ArmScore:
    """Compute all M38B metrics without pooling markers as replicates."""
    probability = normalise_probabilities(probabilities)
    labels = np.asarray(truth, dtype=np.int64)
    cm = np.asarray(marker_cm, dtype=np.float64)
    require(labels.shape == probability.shape[:2] and np.all((labels >= 0) & (labels < 6)),
            "probability and truth axes differ")
    weights = normalised_voronoi_cm_weights(cm)
    people, marker_count = labels.shape
    one_hot = np.eye(6, dtype=np.float64)[labels]
    brier_marker = np.square(probability - one_hot).sum(axis=2)
    expected_dosage = probability @ STATE_DOSAGE / 2.0
    observed_dosage = STATE_DOSAGE[labels] / 2.0
    dosage_error = np.abs(expected_dosage - observed_dosage)
    nam_present = observed_dosage[:, :, 2] > 0
    nam_present_weight = (nam_present * weights[None, :]).sum(axis=1)

    per_person: dict[str, np.ndarray] = {
        "log_loss_cm": per_person_log_loss(probability, labels, cm, weighted=True),
        "log_loss_uniform": per_person_log_loss(probability, labels, cm, weighted=False),
        "brier_cm": brier_marker @ weights,
        "ancestry_proportion_mae_macro_cm": (dosage_error * weights[None, :, None]).sum(axis=1).mean(axis=1),
        "ancestry_proportion_mae_nam_cm": dosage_error[:, :, 2] @ weights,
        "ancestry_proportion_mae_nam_truth_present_cm": np.divide(
            (dosage_error[:, :, 2] * nam_present * weights[None, :]).sum(axis=1),
            nam_present_weight,
            out=np.full(people, np.nan, dtype=np.float64),
            where=nam_present_weight > 0,
        ),
        "_nam_truth_present_error_numerator": (
            dosage_error[:, :, 2] * nam_present * weights[None, :]
        ).sum(axis=1),
        "_nam_truth_present_weight": nam_present_weight,
    }
    hard = probability.argmax(axis=2)
    truth_counts = np.zeros(people, dtype=np.int64)
    predicted_counts = np.zeros(people, dtype=np.int64)
    matched_counts = np.zeros(people, dtype=np.int64)
    boundary_costs = np.zeros(people, dtype=np.float64)
    for person in range(people):
        truth_boundary = boundary_records(labels[person], cm)
        predicted_boundary = boundary_records(hard[person], cm)
        matched, cost = directed_boundary_match(truth_boundary, predicted_boundary)
        truth_counts[person] = len(truth_boundary)
        predicted_counts[person] = len(predicted_boundary)
        matched_counts[person] = matched
        boundary_costs[person] = cost
    denominators = truth_counts + predicted_counts
    per_person["boundary_f1_0_2cm"] = np.divide(
        2.0 * matched_counts, denominators,
        out=np.ones(people, dtype=np.float64), where=denominators > 0,
    )
    per_person["boundary_mean_error_0_2cm"] = np.divide(
        boundary_costs, matched_counts,
        out=np.full(people, np.nan, dtype=np.float64), where=matched_counts > 0,
    )
    span_morgan = (cm[-1] - cm[0]) / 100.0
    require(span_morgan > 0, "marker_cM span must be positive")
    false_counts = predicted_counts - matched_counts
    per_person["false_transitions_per_morgan_0_2cm"] = false_counts / span_morgan

    matched_total = int(matched_counts.sum())
    boundary_denominator = int(truth_counts.sum() + predicted_counts.sum())
    ancestry_mae = (dosage_error * weights[None, :, None]).sum(axis=1).mean(axis=0)
    nam_present_denominator = float(nam_present_weight.sum())
    summary: dict[str, Any] = {
        "log_loss_cm": float(per_person["log_loss_cm"].mean()),
        "log_loss_uniform": float(per_person["log_loss_uniform"].mean()),
        "brier_cm": float(per_person["brier_cm"].mean()),
        "ancestry_proportion_mae_macro_cm": float(per_person["ancestry_proportion_mae_macro_cm"].mean()),
        "ancestry_proportion_mae_nam_cm": float(per_person["ancestry_proportion_mae_nam_cm"].mean()),
        "ancestry_proportion_mae_nam_truth_present_cm": (
            float(np.sum(dosage_error[:, :, 2] * nam_present * weights[None, :]) /
                  nam_present_denominator)
            if nam_present_denominator > 0 else None
        ),
        "ancestry_proportion_mae_by_ancestry_cm": {
            name: float(ancestry_mae[index]) for index, name in enumerate(ANCESTRY_NAMES)
        },
        "boundary_0_2cm": {
            "truth": int(truth_counts.sum()),
            "predicted": int(predicted_counts.sum()),
            "matched_one_to_one_directed": matched_total,
            "f1_micro": float(2.0 * matched_total / boundary_denominator
                              if boundary_denominator else 1.0),
            "mean_error_cm": (float(boundary_costs.sum() / matched_total)
                              if matched_total else None),
            "false_transitions_per_morgan": float(false_counts.sum() / (people * span_morgan)),
        },
    }
    require(all(np.isfinite(per_person[name]).all()
                for name in PER_PERSON_METRICS
                if name not in {"boundary_mean_error_0_2cm", "ancestry_proportion_mae_nam_truth_present_cm"}),
            "a per-person metric is non-finite")
    return ArmScore(
        summary=summary,
        per_person=per_person,
        boundary_counts={
            "truth": truth_counts,
            "predicted": predicted_counts,
            "matched": matched_counts,
        },
    )


def stratified_person_bootstrap_indices(
    fold_ids: np.ndarray, replicates: int, seed: int,
) -> np.ndarray:
    """Resample complete people within folds; one index matrix serves all contrasts."""
    folds = np.asarray(fold_ids)
    require(folds.ndim == 1 and len(folds) > 0, "fold_ids must be a non-empty vector")
    require(replicates >= 100, "bootstrap needs at least 100 replicates")
    unique_folds = sorted(set(folds.tolist()), key=str)
    require(len(unique_folds) == 3, "M38B requires exactly three outer folds")
    groups = [np.flatnonzero(folds == fold) for fold in unique_folds]
    require(all(len(group) > 0 for group in groups), "every outer fold needs people")
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, len(folds)), dtype=np.int64)
    start = 0
    for group in groups:
        result[:, start:start + len(group)] = rng.choice(
            group, size=(replicates, len(group)), replace=True,
        )
        start += len(group)
    return result


def bootstrap_primary_contrasts(
    deltas: np.ndarray, fold_ids: np.ndarray,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    person_indices: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Paired bootstrap with per-contrast one-sided confidence bounds."""
    values = np.asarray(deltas, dtype=np.float64)
    require(values.ndim == 2 and values.shape[0] == len(fold_ids) and
            values.shape[1] > 0 and np.isfinite(values).all(),
            "contrast deltas must be finite [person, contrast]")
    indices = (stratified_person_bootstrap_indices(fold_ids, replicates, seed)
               if person_indices is None else np.asarray(person_indices, dtype=np.int64))
    require(indices.shape == (replicates, len(fold_ids)) and
            np.all((indices >= 0) & (indices < len(fold_ids))),
            "bootstrap person indices differ")
    bootstrap = values[indices].mean(axis=1)
    observed = values.mean(axis=0)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    error = observed[None, :] - bootstrap
    return {
        "observed": observed,
        "percentile_lower_95": lower,
        "percentile_upper_95": upper,
        "one_sided_upper_95": observed + np.quantile(error, 0.95, axis=0),
        "bonferroni_two_candidate_upper_97_5": (
            observed + np.quantile(error, 0.975, axis=0)
        ),
    }, bootstrap, indices


def bootstrap_boundary_f1_contrasts(
    scored: Mapping[str, ArmScore],
    contrasts: Sequence[tuple[str, str, str]],
    person_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap micro-F1 from resampled TP/FP/FN counts, never mean F1."""
    indices = np.asarray(person_indices, dtype=np.int64)
    arm_bootstrap: dict[str, np.ndarray] = {}
    arm_observed: dict[str, float] = {}
    for arm, score in scored.items():
        matched = score.boundary_counts["matched"]
        truth = score.boundary_counts["truth"]
        predicted = score.boundary_counts["predicted"]
        bootstrap_matched = matched[indices].sum(axis=1)
        bootstrap_denominator = (truth[indices].sum(axis=1) +
                                 predicted[indices].sum(axis=1))
        arm_bootstrap[arm] = np.divide(
            2.0 * bootstrap_matched, bootstrap_denominator,
            out=np.ones(len(indices), dtype=np.float64),
            where=bootstrap_denominator > 0,
        )
        denominator = int(truth.sum() + predicted.sum())
        arm_observed[arm] = (2.0 * float(matched.sum()) / denominator
                             if denominator else 1.0)
    observed = np.asarray([
        arm_observed[left] - arm_observed[right] for _, left, right in contrasts
    ])
    bootstrap = np.stack([
        arm_bootstrap[left] - arm_bootstrap[right] for _, left, right in contrasts
    ], axis=1)
    return observed, bootstrap


@dataclass(frozen=True)
class PredictionInput:
    arm: str
    path: Path
    probabilities: np.ndarray
    person_ids: np.ndarray
    fold_ids: np.ndarray
    marker_cm: np.ndarray
    marker_pos: np.ndarray | None


def load_prediction(path: Path, expected_arm: str) -> PredictionInput:
    with np.load(path, allow_pickle=False) as archive:
        required = {"probabilities", "marker_cM", "fold_ids", "arm"}
        require(required.issubset(archive.files),
                f"prediction {path} lacks {sorted(required - set(archive.files))}")
        arm = _scalar_text(archive["arm"], "arm")
        require(arm == expected_arm, f"prediction arm {arm!r} differs from {expected_arm!r}")
        if "state_names" in archive.files:
            state_names = tuple(_text_vector(archive["state_names"], "state_names").tolist())
            require(state_names == STATE_NAMES, "prediction diploid-state order differs")
        person_ids = _person_axis(archive)
        fold_ids = _fold_vector(archive["fold_ids"], len(person_ids))
        probability = normalise_probabilities(archive["probabilities"])
        cm = np.ascontiguousarray(archive["marker_cM"], dtype=np.float64)
        position = (np.ascontiguousarray(archive["marker_pos"], dtype=np.int64)
                    if "marker_pos" in archive.files else None)
    require(probability.shape[:2] == (len(person_ids), len(cm)),
            "prediction probability/person/marker axes differ")
    if position is not None:
        require(position.shape == cm.shape, "prediction marker_pos/marker_cM axes differ")
    voronoi_cm_widths(cm)
    return PredictionInput(expected_arm, path, probability, person_ids, fold_ids, cm, position)


def load_truth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        require({"marker_cM", "fold_ids"}.issubset(archive.files),
                "truth NPZ needs marker_cM and fold_ids")
        people = _person_axis(archive)
        folds = _fold_vector(archive["fold_ids"], len(people))
        cm = np.ascontiguousarray(archive["marker_cM"], dtype=np.float64)
        position = (np.ascontiguousarray(archive["marker_pos"], dtype=np.int64)
                    if "marker_pos" in archive.files else None)
        states = _truth_states(archive)
    require(states.shape == (len(people), len(cm)), "truth person/marker axes differ")
    if position is not None:
        require(position.shape == cm.shape, "truth marker_pos/marker_cM axes differ")
    voronoi_cm_widths(cm)
    return states, people, folds, cm, position


def parse_assignment(value: str, separator: str, kind: str) -> tuple[str, str]:
    require(separator in value, f"{kind} must use NAME{separator}VALUE")
    left, right = value.split(separator, 1)
    require(bool(left) and bool(right), f"{kind} name and value cannot be empty")
    return left, right


def parse_contrast(value: str) -> tuple[str, str, str]:
    name, arms = parse_assignment(value, "=", "contrast")
    require("," in arms, "contrast must use NAME=LEFT,RIGHT")
    left, right = arms.split(",", 1)
    require(left and right and left != right, "contrast needs two distinct arms")
    return name, left, right


def verify_scoring_receipts(
    predictions: Mapping[str, Path], prediction_receipts: Mapping[str, Path],
    truth: Path, truth_receipt: Path, family: str,
) -> dict[str, str]:
    require(set(predictions) == set(prediction_receipts),
            "every prediction needs exactly one receipt")
    contract_rows: list[tuple[str, str, str, str]] = []
    fold_hashes: list[str] = []
    fold_receipt_hashes: list[str] = []
    for arm, prediction in predictions.items():
        receipt_path = prediction_receipts[arm]
        document = json.loads(receipt_path.read_text(encoding="utf-8"))
        digest = sha256_file(prediction)
        if arm in {"full", "minus", "RD"}:
            expected_source = "F_full_projected" if arm == "full" else "F_minus_S660"
            require(
                document.get("stage") == "M38B_PACK_TRUTH_BLIND_OOF_BASELINES"
                and document.get("status") == "PASS_FULL_MINUS_AND_ANALYTIC_RD_PACKED"
                and document.get("truth_read") is False
                and document.get("people") == 96
                and document.get("markers") == 42326
                and document.get("RD_alias") == "OFF_EXACT_F_MINUS_S660_NO_FIT"
                and document.get("outputs", {}).get(prediction.name, {}).get("sha256") == digest
                and document.get("outputs", {}).get(prediction.name, {}).get("source") == expected_source,
                f"M38B {arm} baseline receipt differs",
            )
        else:
            require(
                document.get("stage") == "M38B_COLLECT_TRUTH_BLIND_OOF"
                and document.get("status") == "PASS_EXACT_ONE_OOF_PREDICTION_PER_PERSON"
                and document.get("family") == family
                and document.get("arm") == arm
                and document.get("diagnostic_only") is False
                and document.get("output_sha256") == digest,
                f"M38B {family}/{arm} OOF receipt differs",
            )
        contract_rows.append(tuple(str(document.get(name, "")) for name in (
            "model_contract_receipt_sha256", "base_contract_sha256",
            "amendment_sha256", "amendment_2_sha256",
        )))
        fold_hashes.append(str(document.get("folds_sha256", "")))
        fold_receipt_hashes.append(str(document.get("folds_receipt_sha256", "")))
    truth_document = json.loads(truth_receipt.read_text(encoding="utf-8"))
    require(
        truth_document.get("stage") == "M38B_PACK_OOF_SCORE_TRUTH"
        and truth_document.get("status") == "PASS_TRUTH_SEPARATE_SCORING_BRANCH"
        and truth_document.get("output_sha256") == sha256_file(truth),
        "M38B score-truth receipt differs",
    )
    contract_rows.append(tuple(str(truth_document.get(name, "")) for name in (
        "model_contract_receipt_sha256", "base_contract_sha256",
        "amendment_sha256", "amendment_2_sha256",
    )))
    fold_hashes.append(str(truth_document.get("folds_sha256", "")))
    fold_receipt_hashes.append(str(truth_document.get("folds_receipt_sha256", "")))
    require(len(set(contract_rows)) == 1 and all(len(value) == 64 for value in contract_rows[0]),
            "M38B scoring inputs do not share authenticated contract provenance")
    require(len(set(fold_hashes)) == 1 and len(fold_hashes[0]) == 64,
            "M38B scoring inputs do not share the same frozen folds")
    require(len(set(fold_receipt_hashes)) == 1 and len(fold_receipt_hashes[0]) == 64,
            "M38B scoring inputs do not share the same folds receipt")
    result = dict(zip(("model_contract_receipt_sha256", "base_contract_sha256",
                       "amendment_sha256", "amendment_2_sha256"), contract_rows[0], strict=True))
    result["folds_sha256"] = fold_hashes[0]
    result["folds_receipt_sha256"] = fold_receipt_hashes[0]
    return result


def _axis_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def analyse_files(
    predictions: Mapping[str, Path], truth_path: Path,
    contrasts: Sequence[tuple[str, str, str]],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    expected_person_count: int | None = None,
    expected_fold_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    truth, person_ids, fold_ids, marker_cm, marker_pos = load_truth(truth_path)
    require(len(set(fold_ids.tolist())) == 3, "M38B truth must contain exactly three outer folds")
    if expected_person_count is not None:
        require(len(person_ids) == expected_person_count,
                f"M38B expected {expected_person_count} people, observed {len(person_ids)}")
    fold_counts = {fold: int(np.count_nonzero(fold_ids == fold))
                   for fold in sorted(set(fold_ids.tolist()), key=str)}
    if expected_fold_size is not None:
        require(all(count == expected_fold_size for count in fold_counts.values()),
                f"M38B expected {expected_fold_size} SCORE people per fold")
    loaded = {arm: load_prediction(path, arm) for arm, path in predictions.items()}
    require(len(loaded) >= 2, "at least two prediction arms are required")
    for item in loaded.values():
        require(np.array_equal(item.person_ids, person_ids), f"{item.arm} person axis differs")
        require(np.array_equal(item.fold_ids, fold_ids), f"{item.arm} fold axis differs")
        require(np.array_equal(item.marker_cm, marker_cm), f"{item.arm} marker_cM axis differs")
        if marker_pos is not None:
            require(item.marker_pos is not None and np.array_equal(item.marker_pos, marker_pos),
                    f"{item.arm} physical marker axis differs")
    names = [name for name, _, _ in contrasts]
    require(len(contrasts) > 0 and len(set(names)) == len(names),
            "contrast names must be non-empty and unique")
    require(all(left in loaded and right in loaded for _, left, right in contrasts),
            "a contrast refers to an unavailable arm")

    fold_order = sorted(set(fold_ids.tolist()), key=str)
    scored = {arm: score_arm(item.probabilities, truth, marker_cm)
              for arm, item in loaded.items()}
    nam_weight = next(iter(scored.values())).per_person["_nam_truth_present_weight"]
    require(
        float(nam_weight.sum()) > 0
        and all(float(nam_weight[fold_ids == fold].sum()) > 0 for fold in fold_order),
        "NAM-truth-present metric is unevaluable globally or in an outer fold",
    )
    metric_cube = np.stack([
        np.stack([scored[arm].per_person[name] for name in PER_PERSON_METRICS], axis=1)
        for arm in loaded
    ], axis=0)
    arm_order = tuple(loaded)
    arm_index = {arm: index for index, arm in enumerate(arm_order)}
    contrast_cube = np.stack([
        metric_cube[arm_index[left]] - metric_cube[arm_index[right]]
        for _, left, right in contrasts
    ], axis=0)
    log_loss_index = PER_PERSON_METRICS.index("log_loss_cm")
    primary_deltas = contrast_cube[:, :, log_loss_index].T
    inference, bootstrap, person_indices = bootstrap_primary_contrasts(
        primary_deltas, fold_ids, bootstrap_replicates, bootstrap_seed,
    )
    f1_observed, f1_bootstrap = bootstrap_boundary_f1_contrasts(
        scored, contrasts, person_indices,
    )
    contrast_summary: dict[str, Any] = {}
    for index, (name, left, right) in enumerate(contrasts):
        fold_means = {
            fold: float(primary_deltas[fold_ids == fold, index].mean()) for fold in fold_order
        }
        direction_count = sum(value < 0 for value in fold_means.values())
        candidate_bound = float(
            inference["bonferroni_two_candidate_upper_97_5"][index]
        )
        metric_deltas = {
            metric: (float(np.nanmean(contrast_cube[index, :, metric_index]))
                     if np.isfinite(contrast_cube[index, :, metric_index]).any() else None)
            for metric_index, metric in enumerate(PER_PERSON_METRICS)
            if metric != "boundary_f1_0_2cm"
        }
        metric_deltas["boundary_f1_0_2cm"] = float(f1_observed[index])
        nam_metric = "ancestry_proportion_mae_nam_truth_present_cm"
        nam_num_delta = (
            scored[left].per_person["_nam_truth_present_error_numerator"]
            - scored[right].per_person["_nam_truth_present_error_numerator"]
        )
        metric_deltas[nam_metric] = float(nam_num_delta.sum() / nam_weight.sum())
        metric_fold_means: dict[str, dict[str, float | None]] = {}
        metric_fold_n_eff: dict[str, dict[str, int]] = {}
        for metric_index, metric in enumerate(PER_PERSON_METRICS):
            metric_fold_means[metric] = {}
            metric_fold_n_eff[metric] = {}
            for fold in fold_order:
                values = contrast_cube[index, fold_ids == fold, metric_index]
                if metric == nam_metric:
                    selected_fold = fold_ids == fold
                    denominator = float(nam_weight[selected_fold].sum())
                    metric_fold_n_eff[metric][str(fold)] = int(
                        np.count_nonzero(nam_weight[selected_fold] > 0)
                    )
                    metric_fold_means[metric][str(fold)] = float(
                        nam_num_delta[selected_fold].sum() / denominator
                    )
                else:
                    metric_fold_n_eff[metric][str(fold)] = int(np.isfinite(values).sum())
                    metric_fold_means[metric][str(fold)] = (
                        float(np.nanmean(values)) if np.isfinite(values).any() else None
                    )
        metric_ci95: dict[str, list[float] | None] = {}
        for metric_index, metric in enumerate(PER_PERSON_METRICS):
            if metric == "boundary_f1_0_2cm":
                metric_ci95[metric] = [float(value) for value in np.quantile(
                    f1_bootstrap[:, index], (0.025, 0.975),
                )]
                continue
            values = contrast_cube[index, :, metric_index]
            if metric == nam_metric:
                denominators = nam_weight[person_indices].sum(axis=1)
                require(np.all(denominators > 0),
                        "NAM-truth-present bootstrap contains an unevaluable draw")
                draws = nam_num_delta[person_indices].sum(axis=1) / denominators
                metric_ci95[metric] = [float(value) for value in np.quantile(
                    draws, (0.025, 0.975),
                )]
                continue
            if not np.isfinite(values).any():
                metric_ci95[metric] = None
                continue
            draws = np.nanmean(values[person_indices], axis=1)
            metric_ci95[metric] = [float(value) for value in np.quantile(
                draws[np.isfinite(draws)], (0.025, 0.975),
            )]
        primary_candidate_contrast = name in {"RE-RD", "RE-SHAM"}
        contrast_summary[name] = {
            "left_arm": left,
            "right_arm": right,
            "primary_delta_log_loss_cm_left_minus_right": float(inference["observed"][index]),
            "percentile_ci95": [float(inference["percentile_lower_95"][index]),
                                float(inference["percentile_upper_95"][index])],
            "one_sided_upper_95": float(inference["one_sided_upper_95"][index]),
            "one_sided_upper_97_5_two_family": candidate_bound,
            "bonferroni_two_candidate_upper_97_5": (
                candidate_bound if primary_candidate_contrast else None
            ),
            "fold_mean_deltas": fold_means,
            "negative_direction_folds": direction_count,
            "direction_3_of_3": direction_count == 3,
            "candidate_contrast_gate": (
                direction_count == 3 and candidate_bound < 0
                if primary_candidate_contrast else None
            ),
            "metric_deltas_left_minus_right": metric_deltas,
            "metric_fold_mean_deltas": metric_fold_means,
            "metric_fold_n_eff": metric_fold_n_eff,
            "metric_delta_percentile_ci95": metric_ci95,
            "boundary_f1_micro_delta_percentile_ci95": [
                float(value) for value in np.quantile(
                    f1_bootstrap[:, index], (0.025, 0.975),
                )
            ],
        }

    required_candidate_contrasts = ("RE-RD", "RE-SHAM")
    candidate_gate_available = all(name in contrast_summary
                                   for name in required_candidate_contrasts)
    candidate_gate = (
        all(bool(contrast_summary[name]["candidate_contrast_gate"])
            for name in required_candidate_contrasts)
        if candidate_gate_available else None
    )
    specific = contrast_summary.get("RE-RD")
    no_clear_harm: bool | None = None
    no_reversal: bool | None = None
    error_metrics = (
            "brier_cm", "ancestry_proportion_mae_macro_cm", "ancestry_proportion_mae_nam_cm",
            "ancestry_proportion_mae_nam_truth_present_cm", "false_transitions_per_morgan_0_2cm",
    )
    def no_clear_harm_for(row: dict[str, Any] | None) -> bool | None:
        if row is None:
            return None
        intervals = row["metric_delta_percentile_ci95"]
        metrics = error_metrics + ("boundary_f1_0_2cm",)
        evaluable = all(
            intervals[name] is not None
            and all(count > 0 for count in row["metric_fold_n_eff"][name].values())
            for name in metrics
        )
        return bool(evaluable and all(intervals[name][0] <= 0 for name in error_metrics)
                    and intervals["boundary_f1_0_2cm"][1] >= 0)
    if specific is not None:
        no_clear_harm = no_clear_harm_for(specific)
        reversal_contrasts = [contrast_summary.get(name) for name in ("RE-RD", "RE-SHAM")]
        no_reversal = all(
            row is not None
            and row["metric_deltas_left_minus_right"]["log_loss_uniform"] <= 0
            and all(value is not None and value <= 0
                    for value in row["metric_fold_mean_deltas"]["log_loss_uniform"].values())
            for row in reversal_contrasts
        )
    deploy = contrast_summary.get("RE-full")
    no_clear_harm_deploy = no_clear_harm_for(deploy)
    deploy_gate = None if deploy is None else (
        deploy["negative_direction_folds"] == 3
        and deploy["one_sided_upper_97_5_two_family"] < 0
        and deploy["metric_deltas_left_minus_right"]["log_loss_uniform"] <= 0
        and all(value is not None and value <= 0
                for value in deploy["metric_fold_mean_deltas"]["log_loss_uniform"].values())
    )

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage": "M38B_OOF_SCORE",
        "status": "PASS_SCORED",
        "claim_level": "exploratory_chr22_fit_only_donor_conditional",
        "person_count": len(person_ids),
        "marker_count": len(marker_cm),
        "outer_fold_count": 3,
        "outer_fold_person_counts": fold_counts,
        "person_axis_sha256": _axis_sha256(person_ids.tolist()),
        "marker_cm_sha256": hashlib.sha256(marker_cm.tobytes(order="C")).hexdigest(),
        "state_order": list(STATE_NAMES),
        "ancestry_order": list(ANCESTRY_NAMES),
        "probability_floor_and_renormalisation": PROBABILITY_FLOOR,
        "primary_metric": "per-person log-loss integrated with normalised Voronoi widths in cM",
        "boundary_definition": {
            "tolerance_cM": BOUNDARY_TOLERANCE_CM,
            "matching": "ordered one-to-one with directed before/after diploid states",
            "false_transition": "predicted boundary unmatched at the same 0.2 cM tolerance",
        },
        "truth_usage": "evaluation_only; no model selection or checkpoint choice",
        "arm_metrics": {arm: scored[arm].summary for arm in arm_order},
        "contrasts": contrast_summary,
        "candidate_incremental_gate": {
            "method": (
                "intersection-union: both RE-RD and RE-SHAM must pass; no "
                "multiplicity penalty between these two required conditions"
            ),
            "required_contrasts": list(required_candidate_contrasts),
            "available": candidate_gate_available,
            "pass": candidate_gate,
            "between_candidate_correction": (
                "one-sided 97.5% upper bound for two prespecified candidate "
                "families (TRACE analytic and TCN)"
            ),
            "excluded_from_multiplicity_family": ["full-minus", "RE-full"],
        },
        "secondary_gates": {
            "no_statistically_clear_harm": {
                "pass": no_clear_harm,
                "comparison": "RE-OFF",
                "meaning": (
                    "For error metrics the lower 95% bound is not above zero; "
                    "for boundary F1 the upper 95% bound is not below zero. "
                    "This is not a non-inferiority claim."
                ),
            },
            "no_statistically_clear_harm_vs_full": {
                "pass": no_clear_harm_deploy,
                "comparison": "RE-full",
                "required_for_deploy_claim": True,
            },
            "weighted_uniform_no_sign_reversal": {"pass": no_reversal},
            "deploy_improvement_over_full_flare": {
                "pass": bool(deploy_gate and no_clear_harm_deploy),
                "claim_allowed_only_if_pass": True,
            },
        },
        "bootstrap": {
            "unit": "whole person",
            "paired": True,
            "stratified_by": "outer fold",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "candidate_bound": (
                "per-contrast one-sided basic 97.5% upper bound; Bonferroni "
                "factor two is for the two candidate families, not for the "
                "four reported contrasts"
            ),
            "boundary_f1_aggregation": "resampled micro-F1 from summed TP/FP/FN",
            "per_fold_confidence_intervals_used": False,
        },
        "metric_direction": {
            "lower_is_better": [name for name in PER_PERSON_METRICS
                                if name != "boundary_f1_0_2cm"],
            "higher_is_better": ["boundary_f1_0_2cm"],
            "contrast_orientation": "left arm minus right arm",
        },
        "inputs_sha256": {
            "truth": sha256_file(truth_path),
            "predictions": {arm: sha256_file(path) for arm, path in predictions.items()},
        },
    }
    arrays = {
        "person_ids": person_ids.astype("S"),
        "fold_ids": fold_ids.astype("S"),
        "marker_cM": marker_cm,
        "voronoi_weight": normalised_voronoi_cm_weights(marker_cm),
        "arm_names": np.asarray(arm_order, dtype="S"),
        "metric_names": np.asarray(PER_PERSON_METRICS, dtype="S"),
        "per_person_metrics": metric_cube,
        "contrast_names": np.asarray(names, dtype="S"),
        "contrast_left_arms": np.asarray([left for _, left, _ in contrasts], dtype="S"),
        "contrast_right_arms": np.asarray([right for _, _, right in contrasts], dtype="S"),
        "per_person_contrasts": contrast_cube,
        "bootstrap_primary_deltas": bootstrap,
        "bootstrap_boundary_f1_deltas": f1_bootstrap,
        "bootstrap_person_indices": person_indices,
    }
    return result, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True,
                        help="Repeat as ARM=/path/to/prediction.npz")
    parser.add_argument("--prediction-receipt", action="append", required=True,
                        help="Repeat as ARM=/path/to/receipt.json")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-receipt", type=Path, required=True)
    parser.add_argument("--family", choices=("analytic", "tcn"), required=True)
    parser.add_argument("--contrast", action="append",
                        help="Repeat as NAME=LEFT_ARM,RIGHT_ARM")
    parser.add_argument("--bootstrap-replicates", type=int,
                        default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--expected-person-count", type=int, default=96)
    parser.add_argument("--expected-fold-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-person-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions: dict[str, Path] = {}
    for value in args.prediction:
        arm, path = parse_assignment(value, "=", "prediction")
        require(arm not in predictions, f"duplicate prediction arm {arm}")
        predictions[arm] = Path(path)
    prediction_receipts: dict[str, Path] = {}
    for value in args.prediction_receipt:
        arm, path = parse_assignment(value, "=", "prediction receipt")
        require(arm not in prediction_receipts, f"duplicate prediction receipt arm {arm}")
        prediction_receipts[arm] = Path(path)
    provenance = verify_scoring_receipts(
        predictions, prediction_receipts, args.truth, args.truth_receipt, args.family,
    )
    contrasts = ([parse_contrast(value) for value in args.contrast]
                 if args.contrast else [
                     ("full-minus", "full", "minus"),
                     ("RE-RD", "RE", "RD"),
                     ("RE-SHAM", "RE", "SHAM"),
                     ("RE-full", "RE", "full"),
                 ])
    require(set(predictions) == {"full", "minus", "RD", "RE", "SHAM"},
            "M38B scoring requires exact full/minus/OFF/RE/SHAM arms")
    require(set(contrasts) == {
        ("full-minus", "full", "minus"), ("RE-RD", "RE", "RD"),
        ("RE-SHAM", "RE", "SHAM"), ("RE-full", "RE", "full"),
    }, "M38B scoring contrast set differs")
    output_npz = args.per_person_output or args.output.with_suffix(".per_person.npz")
    require(not args.output.exists() and not output_npz.exists(),
            "refusing to overwrite M38B scoring outputs")
    result, arrays = analyse_files(
        predictions, args.truth, contrasts,
        args.bootstrap_replicates, args.bootstrap_seed,
        args.expected_person_count, args.expected_fold_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_npz(output_npz, arrays)
    result["per_person_output"] = {
        "filename": output_npz.name,
        "sha256": sha256_file(output_npz),
    }
    result["authenticated_model_contract"] = provenance
    result["family"] = args.family
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    receipt = args.output.with_suffix(".receipt.json")
    require(not receipt.exists(), "refusing to overwrite M38B scoring receipt")
    receipt.write_text(json.dumps({
        "schema_version": "1.0.0",
        "stage": "M38B_OOF_SCORE",
        "status": result["status"],
        "family": args.family,
        "arms": sorted(predictions),
        "contrasts": [list(value) for value in contrasts],
        "truth_sha256": result["inputs_sha256"]["truth"],
        "prediction_sha256": result["inputs_sha256"]["predictions"],
        "output_sha256": sha256_file(args.output),
        "per_person_output_sha256": sha256_file(output_npz),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        **provenance,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "person_count": result["person_count"],
        "candidate_incremental_gate": result["candidate_incremental_gate"]["pass"],
        "candidate_contrasts": {
            name: row["candidate_contrast_gate"]
            for name, row in result["contrasts"].items()
            if row["candidate_contrast_gate"] is not None
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
