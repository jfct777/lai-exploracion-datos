#!/usr/bin/env python3
"""Stream and compare phased local-ancestry calls against positional truth.

The expected text format has no header.  Each row contains chromosome,
position, and one integer ancestry label per haplotype.  The implementation is
streaming so a full chromosome is not materialized in memory.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, TextIO


@dataclass(frozen=True)
class LaiRow:
    chrom: str
    position: int
    labels: tuple[int, ...]


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_lai_rows(handle: TextIO, source: str) -> Iterator[LaiRow]:
    expected_haplotypes: int | None = None
    previous: tuple[str, int] | None = None
    for line_number, line in enumerate(handle, start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"{source}:{line_number}: expected chrom, position and labels")
        try:
            position = int(fields[1])
            labels = tuple(int(value) for value in fields[2:])
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: non-integer position or label") from exc
        if expected_haplotypes is None:
            expected_haplotypes = len(labels)
            if expected_haplotypes == 0:
                raise ValueError(f"{source}:{line_number}: no haplotype labels")
        elif len(labels) != expected_haplotypes:
            raise ValueError(
                f"{source}:{line_number}: {len(labels)} labels; expected {expected_haplotypes}"
            )
        coordinate = (fields[0], position)
        if previous is not None and coordinate <= previous:
            raise ValueError(f"{source}:{line_number}: coordinates are not strictly increasing")
        previous = coordinate
        yield LaiRow(fields[0], position, labels)


def _empty_transition_lists(n_haplotypes: int) -> list[list[int]]:
    return [[] for _ in range(n_haplotypes)]


def summarize_truth(path: Path) -> dict:
    n_rows = 0
    chrom: str | None = None
    first_position: int | None = None
    last_position: int | None = None
    n_haplotypes: int | None = None
    previous_labels: tuple[int, ...] | None = None
    label_counts: Counter[int] = Counter()
    transitions: list[list[int]] = []

    with open_text(path) as handle:
        for row in iter_lai_rows(handle, str(path)):
            if chrom is None:
                chrom = row.chrom
                first_position = row.position
                n_haplotypes = len(row.labels)
                transitions = _empty_transition_lists(n_haplotypes)
            elif row.chrom != chrom:
                raise ValueError(f"{path}: multiple chromosomes are not supported")
            if previous_labels is not None:
                for haplotype, (before, after) in enumerate(zip(previous_labels, row.labels)):
                    if before != after:
                        transitions[haplotype].append(row.position)
            label_counts.update(row.labels)
            previous_labels = row.labels
            last_position = row.position
            n_rows += 1

    if n_rows == 0 or n_haplotypes is None:
        raise ValueError(f"{path}: no ancestry rows")
    transition_counts = [len(values) for values in transitions]
    return {
        "chrom": chrom,
        "n_rows": n_rows,
        "n_haplotypes": n_haplotypes,
        "first_position": first_position,
        "last_position": last_position,
        "label_counts": {str(label): count for label, count in sorted(label_counts.items())},
        "n_transitions": sum(transition_counts),
        "transitions_per_haplotype": transition_counts,
    }


def _greedy_boundary_distances(
    truth: list[int], prediction: list[int], tolerance_bp: int
) -> tuple[list[int], int, int]:
    """Return one-to-one distances and unmatched counts for sorted boundaries."""
    truth_index = prediction_index = 0
    distances: list[int] = []
    while truth_index < len(truth) and prediction_index < len(prediction):
        delta = prediction[prediction_index] - truth[truth_index]
        if abs(delta) <= tolerance_bp:
            distances.append(abs(delta))
            truth_index += 1
            prediction_index += 1
        elif prediction[prediction_index] < truth[truth_index] - tolerance_bp:
            prediction_index += 1
        else:
            truth_index += 1
    return distances, len(truth) - len(distances), len(prediction) - len(distances)


def compare_calls(truth_path: Path, prediction_path: Path, tolerance_bp: int) -> dict:
    if tolerance_bp < 0:
        raise ValueError("tolerance_bp must be non-negative")
    confusion: Counter[tuple[int, int]] = Counter()
    truth_counts: Counter[int] = Counter()
    prediction_counts: Counter[int] = Counter()
    n_rows = n_haplotypes = 0
    correct = total = 0
    previous_truth: tuple[int, ...] | None = None
    previous_prediction: tuple[int, ...] | None = None
    truth_boundaries: list[list[int]] = []
    prediction_boundaries: list[list[int]] = []

    with ExitStack() as stack:
        truth_handle = stack.enter_context(open_text(truth_path))
        prediction_handle = stack.enter_context(open_text(prediction_path))
        truth_rows = iter_lai_rows(truth_handle, str(truth_path))
        prediction_rows = iter_lai_rows(prediction_handle, str(prediction_path))
        for row_number, pair in enumerate(zip_longest(truth_rows, prediction_rows), start=1):
            truth, prediction = pair
            if truth is None or prediction is None:
                raise ValueError("truth and prediction have different row counts")
            if (truth.chrom, truth.position) != (prediction.chrom, prediction.position):
                raise ValueError(f"row {row_number}: truth and prediction coordinates differ")
            if len(truth.labels) != len(prediction.labels):
                raise ValueError(f"row {row_number}: truth and prediction haplotype counts differ")
            if n_rows == 0:
                n_haplotypes = len(truth.labels)
                truth_boundaries = _empty_transition_lists(n_haplotypes)
                prediction_boundaries = _empty_transition_lists(n_haplotypes)
            if previous_truth is not None and previous_prediction is not None:
                for haplotype, (before, after) in enumerate(zip(previous_truth, truth.labels)):
                    if before != after:
                        truth_boundaries[haplotype].append(truth.position)
                for haplotype, (before, after) in enumerate(
                    zip(previous_prediction, prediction.labels)
                ):
                    if before != after:
                        prediction_boundaries[haplotype].append(prediction.position)
            for expected, observed in zip(truth.labels, prediction.labels):
                confusion[(expected, observed)] += 1
                truth_counts[expected] += 1
                prediction_counts[observed] += 1
                correct += expected == observed
                total += 1
            previous_truth = truth.labels
            previous_prediction = prediction.labels
            n_rows += 1

    labels = sorted(set(truth_counts) | set(prediction_counts))
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = confusion[(label, label)]
        false_positive = prediction_counts[label] - true_positive
        false_negative = truth_counts[label] - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[str(label)] = {
            "support": truth_counts[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    distances: list[int] = []
    missed_truth = extra_prediction = 0
    for truth, prediction in zip(truth_boundaries, prediction_boundaries):
        matched, missed, extra = _greedy_boundary_distances(truth, prediction, tolerance_bp)
        distances.extend(matched)
        missed_truth += missed
        extra_prediction += extra
    n_truth_boundaries = sum(len(values) for values in truth_boundaries)
    n_prediction_boundaries = sum(len(values) for values in prediction_boundaries)
    n_matched = len(distances)
    precision_boundary = n_matched / n_prediction_boundaries if n_prediction_boundaries else 0.0
    recall_boundary = n_matched / n_truth_boundaries if n_truth_boundaries else 0.0

    return {
        "n_rows": n_rows,
        "n_haplotypes": n_haplotypes,
        "n_compared_labels": total,
        "site_accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_label": per_label,
        "boundary_tolerance_bp": tolerance_bp,
        "n_truth_boundaries": n_truth_boundaries,
        "n_prediction_boundaries": n_prediction_boundaries,
        "n_matched_boundaries": n_matched,
        "n_missed_truth_boundaries": missed_truth,
        "n_extra_prediction_boundaries": extra_prediction,
        "boundary_precision": precision_boundary,
        "boundary_recall": recall_boundary,
        "boundary_mean_abs_error_bp": sum(distances) / len(distances) if distances else None,
        "boundary_max_abs_error_bp": max(distances) if distances else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--boundary-tolerance-bp", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "schema_version": 1,
        "truth": str(args.truth),
        "summary": summarize_truth(args.truth),
    }
    if args.prediction is not None:
        result["prediction"] = str(args.prediction)
        result["comparison"] = compare_calls(
            args.truth, args.prediction, args.boundary_tolerance_bp
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
