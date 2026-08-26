#!/usr/bin/env python3
"""Score sealed M34 VALID probabilities against a separately opened truth NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode("ascii") if isinstance(value, bytes) else str(value)
                 for value in values.tolist())


def load_inputs(prediction_path: Path, truth_path: Path
                ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(prediction_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"sample_key_sha256", "marker_pos", "marker_cM",
                                       "ancestry_names", "probabilities"},
                "prediction members differ")
        prediction = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(truth_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"sample_key_sha256", "marker_pos", "labels"},
                "truth members differ")
        truth = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    require(np.array_equal(prediction["sample_key_sha256"], truth["sample_key_sha256"]),
            "prediction/truth sample axes differ")
    require(np.array_equal(prediction["marker_pos"], truth["marker_pos"]),
            "prediction/truth marker axes differ")
    names = _decode(prediction["ancestry_names"])
    probabilities, labels = prediction["probabilities"], truth["labels"]
    require(probabilities.ndim == 4 and labels.shape == probabilities.shape[:3],
            "prediction/truth dimensions differ")
    require(probabilities.shape[3] == len(names) and len(names) >= 2,
            "prediction ancestry axis differs")
    require(np.isfinite(probabilities).all() and np.all(probabilities >= 0) and
            np.max(np.abs(probabilities.sum(axis=3) - 1.0)) <= 5e-6,
            "prediction probabilities differ")
    require(np.issubdtype(labels.dtype, np.integer) and
            np.all((labels >= 0) & (labels < len(names))), "truth labels differ")
    marker_cm = prediction["marker_cM"]
    require(marker_cm.shape == prediction["marker_pos"].shape and
            np.isfinite(marker_cm).all() and np.all(np.diff(marker_cm) >= 0),
            "prediction genetic axis differs")
    return prediction, labels.astype(np.int64, copy=False)


def marker_weights(marker_cm: np.ndarray) -> np.ndarray:
    """Voronoi widths in genetic coordinates for marker-integrated metrics."""
    values = np.asarray(marker_cm, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2 and values[-1] > values[0],
            "scoring needs at least two genetically separated markers")
    boundaries = np.empty(len(values) + 1, dtype=np.float64)
    boundaries[1:-1] = (values[:-1] + values[1:]) / 2.0
    boundaries[0], boundaries[-1] = values[0], values[-1]
    weights = np.diff(boundaries)
    require(np.all(weights >= 0) and weights.sum() > 0, "marker genetic weights differ")
    return weights


def _boundaries(labels: np.ndarray, marker_cm: np.ndarray,
                names: Sequence[str]) -> list[tuple[float, str, str]]:
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return [((float(marker_cm[index - 1]) + float(marker_cm[index])) / 2.0,
             names[int(labels[index - 1])], names[int(labels[index])])
            for index in changes.tolist()]


def ordered_matches(truth: Sequence[tuple[float, str, str]],
                    predicted: Sequence[tuple[float, str, str]], tolerance: float) -> int:
    """Maximum ordered one-to-one matches with transition direction preserved."""
    previous = [0] * (len(predicted) + 1)
    for truth_row in truth:
        current = [0]
        for column, predicted_row in enumerate(predicted, 1):
            compatible = (truth_row[1:] == predicted_row[1:] and
                          abs(truth_row[0] - predicted_row[0]) <=
                          tolerance + 16.0 * np.finfo(np.float64).eps)
            current.append(max(previous[column], current[-1],
                               previous[column - 1] + int(compatible)))
        previous = current
    return previous[-1]


def score(prediction_path: Path, truth_path: Path,
          tolerances: Sequence[float] = (0.1, 0.2, 0.5),
          task: dict[str, Any] | None = None) -> dict[str, Any]:
    prediction, labels = load_inputs(prediction_path, truth_path)
    probabilities = prediction["probabilities"].astype(np.float64, copy=False)
    names = _decode(prediction["ancestry_names"])
    marker_cm = prediction["marker_cM"].astype(np.float64, copy=False)
    weights = marker_weights(marker_cm)
    sample_count, haplotypes, marker_count, ancestry_count = probabilities.shape
    cm_span = float(weights.sum())

    one_hot = np.eye(ancestry_count, dtype=np.float64)[labels]
    predicted_dose = probabilities.sum(axis=1)
    truth_dose = one_hot.sum(axis=1)
    dose_error = np.abs(predicted_dose - truth_dose) / haplotypes
    per_ancestry_mae = (
        np.sum(dose_error * weights[None, :, None], axis=(0, 1)) /
        (sample_count * cm_span)
    )
    macro_mae = float(per_ancestry_mae.mean())

    nam_index = names.index("NAM") if "NAM" in names else None
    nam_truth_present_mae = None
    if nam_index is not None:
        present = truth_dose[:, :, nam_index] > 0
        denominator = float(np.sum(present * weights[None, :]))
        if denominator > 0:
            nam_truth_present_mae = float(np.sum(
                dose_error[:, :, nam_index] * present * weights[None, :]) / denominator)

    brier = float(np.sum(np.square(probabilities - one_hot) *
                         weights[None, None, :, None]) /
                  (sample_count * haplotypes * cm_span))
    hard = np.argmax(probabilities, axis=3)
    totals = {float(value): Counter() for value in tolerances}
    for sample in range(sample_count):
        for haplotype in range(haplotypes):
            true_boundary = _boundaries(labels[sample, haplotype], marker_cm, names)
            predicted_boundary = _boundaries(hard[sample, haplotype], marker_cm, names)
            for tolerance in totals:
                matched = ordered_matches(true_boundary, predicted_boundary, tolerance)
                totals[tolerance].update(truth=len(true_boundary),
                                         predicted=len(predicted_boundary), matched=matched)
    boundary: dict[str, Any] = {}
    for tolerance, counts in totals.items():
        denominator = counts["truth"] + counts["predicted"]
        unmatched = counts["predicted"] - counts["matched"]
        boundary[f"{tolerance:.1f}"] = {
            "truth": counts["truth"], "predicted": counts["predicted"],
            "matched": counts["matched"],
            "f1": (2.0 * counts["matched"] / denominator if denominator else 1.0),
            "false_transitions_per_cM": unmatched /
                (sample_count * haplotypes * cm_span),
        }
    result = {
        "schema_version": "1.0.0", "stage": "M34_EXPLORATORY_SCORING",
        "status": "PASS_SCORED", "claim_level": "exploratory",
        "sample_count": sample_count, "haplotype_count": haplotypes,
        "marker_count": marker_count, "ancestry_names": list(names), "cm_span": cm_span,
        "boundary": boundary, "macro_ancestry_dose_MAE": macro_mae,
        "per_ancestry_MAE": {name: float(per_ancestry_mae[index])
                             for index, name in enumerate(names)},
        "NAM_truth_present_MAE": nam_truth_present_mae,
        "haplotype_Brier": brier,
        "input_sha256": {"prediction": sha256_file(prediction_path),
                          "truth": sha256_file(truth_path)},
        "truth_opened_only_by_scorer": True,
    }
    if task is not None:
        require(isinstance(task, dict) and task,
                "scoring task identity must be a non-empty object")
        result["task"] = task
    require(all(math.isfinite(value["f1"]) and
                math.isfinite(value["false_transitions_per_cM"])
                for value in boundary.values()) and math.isfinite(macro_mae) and
            math.isfinite(brier), "scoring metric is non-finite")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--task", type=Path,
        help="Exact adaptive task JSON embedded in the scoring receipt.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.output.exists(), "refusing to overwrite scoring output")
    task = None
    if args.task is not None:
        require(args.task.is_file(), "scoring task JSON is missing")
        task = json.loads(args.task.read_text(encoding="utf-8"))
        require(isinstance(task, dict), "scoring task JSON must contain an object")
    result = score(args.prediction, args.truth, task=task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "boundary_F1_0.2cM": result["boundary"]["0.2"]["f1"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
