#!/usr/bin/env python3
"""Score phase-free TRACE probabilities against sealed diploid-state truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m37_trace_core import m34_labels_to_states, require


def verify_receipt(artifact: Path, receipt_path: Path, expected_stage: str,
                   candidate_id: str | None = None, family: str | None = None,
                   arm: str | None = None) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("stage") == expected_stage, f"{expected_stage} receipt stage differs")
    require(receipt.get("output_sha256") == hashlib.sha256(artifact.read_bytes()).hexdigest(),
            f"{expected_stage} artifact/receipt hash differs")
    for key, expected in (("candidate_id", candidate_id), ("family", family), ("arm", arm)):
        if expected is not None:
            require(receipt.get(key) == expected, f"{expected_stage} {key} differs")
    return receipt


def load_truth(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "state_labels" in archive.files:
            value = np.ascontiguousarray(archive["state_labels"])
            require(value.ndim == 2 and np.issubdtype(value.dtype, np.integer) and
                    np.all((value >= 0) & (value < 6)), "TRACE truth states differ")
            return value
        require("labels" in archive.files, "truth needs TRACE state_labels or M34 labels [N,2,M]")
        return m34_labels_to_states(np.ascontiguousarray(archive["labels"]))


def boundaries(labels: np.ndarray, cm: np.ndarray) -> list[tuple[float, int, int]]:
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return [(float((cm[index - 1] + cm[index]) / 2.0), int(labels[index - 1]), int(labels[index]))
            for index in changes]


def match_pairs(truth: list[tuple[float, int, int]], predicted: list[tuple[float, int, int]], tolerance: float) -> list[tuple[int, int]]:
    """Optimal 1:1 boundary pairing requires position and unordered-state transition."""
    table: list[list[tuple[int, float, list[tuple[int, int]]]]] = [[(0, 0.0, []) for _ in range(len(predicted) + 1)]
                                                                     for _ in range(len(truth) + 1)]
    def choose(options):
        return max(options, key=lambda item: (item[0], -item[1]))
    for left in range(1, len(truth) + 1):
        for right in range(1, len(predicted) + 1):
            options = [table[left - 1][right], table[left][right - 1]]
            distance = abs(truth[left - 1][0] - predicted[right - 1][0])
            if distance <= tolerance and truth[left - 1][1:] == predicted[right - 1][1:]:
                count, cost, pairs = table[left - 1][right - 1]
                options.append((count + 1, cost + float(distance), pairs + [(left - 1, right - 1)]))
            table[left][right] = choose(options)
    return table[-1][-1][2]


def match_count(truth: list[tuple[float, int, int]], predicted: list[tuple[float, int, int]], tolerance: float) -> int:
    return len(match_pairs(truth, predicted, tolerance))


def ece(probability: np.ndarray, truth: np.ndarray, bins: int = 15) -> float:
    confidence, prediction = probability.max(axis=1), probability.argmax(axis=1)
    result = 0.0
    for index, left in enumerate(np.linspace(0, 1, bins, endpoint=False)):
        upper = left + 1.0 / bins
        mask = (confidence >= left) & ((confidence <= upper) if index == bins - 1 else (confidence < upper))
        if mask.any():
            result += mask.mean() * abs(confidence[mask].mean() - (prediction[mask] == truth[mask]).mean())
    return float(result)


def score(probabilities: np.ndarray, truth: np.ndarray, marker_cm: np.ndarray,
          tolerances: tuple[float, ...] = (.05, .1, .2, .5)) -> dict[str, object]:
    probabilities, truth, marker_cm = (np.asarray(probabilities, dtype=np.float64),
                                       np.asarray(truth, dtype=np.int64), np.asarray(marker_cm, dtype=np.float64))
    require(probabilities.ndim == 3 and probabilities.shape[2] == 6 and truth.shape == probabilities.shape[:2] and
            marker_cm.shape == (probabilities.shape[1],), "TRACE score axes differ")
    require(np.all(probabilities >= 0) and np.allclose(probabilities.sum(axis=2), 1, atol=5e-6, rtol=0) and
            np.all((truth >= 0) & (truth < 6)) and np.all(np.diff(marker_cm) >= 0), "TRACE score values differ")
    flat_p, flat_y = probabilities.reshape(-1, 6), truth.reshape(-1)
    one_hot = np.eye(6)[flat_y]
    state_dosage = np.array(((2, 0, 0), (1, 1, 0), (1, 0, 1),
                             (0, 2, 0), (0, 1, 1), (0, 0, 2)), dtype=np.float64)
    expected = probabilities @ state_dosage
    actual = state_dosage[truth]
    span_morgan = max((marker_cm[-1] - marker_cm[0]) / 100.0, np.finfo(float).eps)
    predicted_labels = probabilities.argmax(axis=2)
    metric: dict[str, object] = {
        "log_loss": float(-np.log(np.maximum(flat_p[np.arange(len(flat_y)), flat_y], 1e-12)).mean()),
        "brier": float(np.square(flat_p - one_hot).sum(axis=1).mean()),
        "ancestry_dose_mae": {name: float(np.abs(expected[:, :, index] - actual[:, :, index]).mean())
                               for index, name in enumerate(("AFR", "EUR", "NAM"))},
        "macro_ancestry_dose_mae": float(np.abs(expected - actual).mean()),
        "calibration_ece_15": ece(flat_p, flat_y),
        "false_transitions_per_morgan": 0.0,
        "f1_boundary": {},
        "boundary_definition": "unordered_diploid_state_changes_matched_one_to_one_by_cM_and_before_after_transition",
        "mean_boundary_error_cM": None,
    }
    false, total_predicted = 0, 0
    matches_by_tolerance = {value: 0 for value in tolerances}
    truth_count = 0
    boundary_errors = []
    for row in range(len(truth)):
        actual_boundary, predicted_boundary = boundaries(truth[row], marker_cm), boundaries(predicted_labels[row], marker_cm)
        truth_count += len(actual_boundary)
        total_predicted += len(predicted_boundary)
        for tolerance in tolerances:
            matches_by_tolerance[tolerance] += match_count(actual_boundary, predicted_boundary, tolerance)
        false += len(predicted_boundary) - match_count(actual_boundary, predicted_boundary, max(tolerances))
        if len(actual_boundary) and len(predicted_boundary):
            pairs = match_pairs(actual_boundary, predicted_boundary, max(tolerances))
            boundary_errors.extend([abs(actual_boundary[left][0] - predicted_boundary[right][0]) for left, right in pairs])
    metric["false_transitions_per_morgan"] = float(false / (len(truth) * span_morgan))
    metric["mean_boundary_error_cM"] = float(np.mean(boundary_errors)) if boundary_errors else None
    for tolerance, matches in matches_by_tolerance.items():
        precision = matches / total_predicted if total_predicted else 0.0
        recall = matches / truth_count if truth_count else 0.0
        metric["f1_boundary"][str(tolerance)] = float(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-receipt", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--family", choices=("hmm", "tcn"), required=True)
    parser.add_argument("--arm", choices=("RE", "RD", "POOLED", "SHAM", "GEOMETRY"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite TRACE metrics")
    prediction_receipt = verify_receipt(args.prediction, args.prediction_receipt, "M37_TRACE_TRAIN",
                                        args.candidate_id, args.family, args.arm)
    features_receipt = verify_receipt(args.features, args.features_receipt, "M37_TRACE_MATERIALIZE",
                                      arm=args.arm)
    require(prediction_receipt.get("predict_features_sha256") ==
            hashlib.sha256(args.features.read_bytes()).hexdigest() and
            prediction_receipt.get("predict_features_receipt_sha256") ==
            hashlib.sha256(args.features_receipt.read_bytes()).hexdigest(),
            "prediction receipt does not bind the scored feature artifact")
    with np.load(args.prediction, allow_pickle=False) as prediction, np.load(args.features, allow_pickle=False) as features:
        require({"sample_key_sha256", "marker_pos", "marker_cM", "marker_axis_sha256"}.issubset(features.files) and
                {"sample_key_sha256", "marker_pos", "marker_axis_sha256"}.issubset(prediction.files),
                "prediction/features lack authenticated axes")
        require(np.array_equal(prediction["sample_key_sha256"], features["sample_key_sha256"]) and
                np.array_equal(prediction["marker_pos"], features["marker_pos"]) and
                np.array_equal(prediction["marker_axis_sha256"], features["marker_axis_sha256"]),
                "prediction/features sample or marker axes differ")
        with np.load(args.truth, allow_pickle=False) as truth_archive:
            require({"sample_key_sha256", "marker_pos"}.issubset(truth_archive.files), "truth lacks authenticated axes")
            require(np.array_equal(truth_archive["sample_key_sha256"], features["sample_key_sha256"]) and
                    np.array_equal(truth_archive["marker_pos"], features["marker_pos"]), "truth/features sample or marker axes differ")
        labels = load_truth(args.truth)
        probabilities = prediction["probabilities"]
        indices = np.asarray(prediction["evaluation_sample_indices"], dtype=np.int64) if "evaluation_sample_indices" in prediction.files else np.arange(len(labels))
        require(indices.ndim == 1 and len(indices) > 0 and len(np.unique(indices)) == len(indices) and
                np.all((indices >= 0) & (indices < len(labels))), "TRACE evaluation sample indices differ")
        result = score(probabilities[indices], labels[indices], features["marker_cM"])
        result["baseline"] = score(features["baseline_states"][indices], labels[indices], features["marker_cM"])
        result["baseline_metadata"] = {"method": str(features.get("baseline_method", np.asarray(["upstream_baseline_unspecified"]))[0]),
                                       "source_sha256": str(features.get("baseline_source_sha256", np.asarray(["unavailable"]))[0])}
        result["evaluation_split"] = str(np.asarray(prediction["evaluation_split"] if "evaluation_split" in prediction.files else
                                                     ["SEALED_INPUT"])[0])
        result["candidate_id"] = args.candidate_id
        result["root"] = args.root
        result["family"] = args.family
        result["arm"] = args.arm
        result["marker_axis_sha256"] = str(features["marker_axis_sha256"].reshape(-1)[0])
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".receipt.json").write_text(json.dumps({
        "schema_version": "1.0.0", "stage": "M37_TRACE_SCORE",
        "candidate_id": args.candidate_id, "family": args.family, "root": args.root, "arm": args.arm,
        "prediction_sha256": hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
        "prediction_receipt_sha256": hashlib.sha256(args.prediction_receipt.read_bytes()).hexdigest(),
        "prediction_receipt_output_sha256": prediction_receipt["output_sha256"],
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "features_receipt_sha256": hashlib.sha256(args.features_receipt.read_bytes()).hexdigest(),
        "features_receipt_output_sha256": features_receipt["output_sha256"],
        "truth_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_TRACE_SCORE", "log_loss": result["log_loss"]}, sort_keys=True))


if __name__ == "__main__":
    main()
