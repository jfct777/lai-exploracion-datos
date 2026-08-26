#!/usr/bin/env python3
"""Score sealed M33 DEVELOPMENT probabilities against private positional truth.

Prediction files are produced without truth access.  This separate process opens
truth only after the prediction file exists and records hashes for both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m28d_b0_scorer as base
import m30_flare_scorer as flare_io
import m33_safe_bridge_core as bridge_core


ANCESTRIES = ("AFR", "EUR", "ASIA")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_key(sample: str) -> bytes:
    return bridge_core.sample_key(sample)


def load_map(path: Path) -> base.GeneticMap:
    """Read either the three-column scorer map or the four-column FLARE map."""
    points = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            require(len(fields) >= 3, f"map row {line_number} is malformed")
            if len(fields) >= 4 and ":" in fields[1]:
                position, cm = int(fields[3]), float(fields[2])
            else:
                position, cm = int(fields[1]), float(fields[2])
            points.append(base.MapPoint(position, cm))
    return base.GeneticMap(points)


def load_prediction(path: Path, target_vcf: Path) -> tuple[flare_io.TargetGrid, np.ndarray]:
    target = flare_io.load_target_grid(target_vcf)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {
            "sample_key_sha256", "marker_pos", "marker_cM", "probabilities"
        }, "prediction members differ")
        keys = np.ascontiguousarray(archive["sample_key_sha256"])
        positions = np.ascontiguousarray(archive["marker_pos"])
        marker_cm = np.ascontiguousarray(archive["marker_cM"])
        probabilities = np.ascontiguousarray(archive["probabilities"])
    expected_keys = np.asarray([sample_key(sample) for sample in target.samples], dtype="|S64")
    require(np.array_equal(keys, expected_keys), "prediction sample order differs")
    require(np.array_equal(positions, np.asarray(target.positions, dtype="<i8")),
            "prediction marker positions differ")
    require(probabilities.shape == (len(target.samples), 2, len(target.loci), 3),
            "prediction probability axes differ")
    require(probabilities.dtype == np.dtype("<f4") and np.isfinite(probabilities).all(),
            "prediction probability values differ")
    require(np.all(probabilities >= 0) and
            np.max(np.abs(probabilities.sum(axis=3) - 1.0)) <= 5e-6,
            "prediction probabilities do not satisfy the simplex")
    require(marker_cm.shape == positions.shape and np.all(np.diff(marker_cm) >= 0),
            "prediction genetic coordinates differ")
    return target, probabilities


def flare_to_prediction(flare_vcf: Path, target_vcf: Path, map_path: Path,
                        output: Path) -> None:
    target = flare_io.load_target_grid(target_vcf)
    flare = flare_io.load_flare_grid(flare_vcf)
    require(target.loci == flare.loci and target.samples == flare.samples,
            "FLARE and target grids differ")
    values = np.empty((len(target.samples), 2, len(target.loci), 3), dtype="<f4")
    for marker, row in enumerate(flare.probabilities):
        for sample_index, sample in enumerate(target.samples):
            values[sample_index, :, marker, :] = row[sample]
    genetic_map = load_map(map_path)
    marker_cm = np.asarray([genetic_map.cm_at(position) for position in target.positions], dtype="<f8")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, sample_key_sha256=np.asarray([sample_key(sample) for sample in target.samples], dtype="|S64"),
        marker_pos=np.asarray(target.positions, dtype="<i8"), marker_cM=marker_cm,
        probabilities=values,
    )


def score(prediction: Path, target_vcf: Path, truth_path: Path, map_path: Path) -> dict:
    target, probabilities = load_prediction(prediction, target_vcf)
    genetic_map = load_map(map_path)
    truth = base.load_truth(
        truth_path, target.samples, "22", target.positions[0], target.positions[-1] + 1,
    )
    cells = base.discrete_voronoi(target.positions)
    cm_span = genetic_map.cm_at(cells[-1][1]) - genetic_map.cm_at(cells[0][0])
    require(cm_span > 0, "nonpositive scoring span")
    ancestry_index = {value: index for index, value in enumerate(ANCESTRIES)}
    cell_left = np.asarray([value[0] for value in cells], dtype="<i8")
    cell_right = np.asarray([value[1] for value in cells], dtype="<i8")
    map_position = np.asarray(genetic_map.positions, dtype=np.float64)
    map_cm = np.asarray(genetic_map.cms, dtype=np.float64)
    tolerances = (0.1, 0.2, 0.5)
    boundary_totals = {tol: Counter() for tol in tolerances}
    mae_num = np.zeros(3, dtype=np.float64)
    mae_present_num = np.zeros(3, dtype=np.float64)
    mae_present_den = np.zeros(3, dtype=np.float64)
    brier_num = 0.0
    confusion: Counter[tuple[str, str]] = Counter()
    per_person_mae = []
    for sample_index, sample in enumerate(target.samples):
        pair = truth[sample]
        cursors = [0, 0]
        interval_start = cells[0][0]
        marker_parts = []
        weight_parts = []
        label0_parts = []
        label1_parts = []
        while interval_start < cells[-1][1]:
            for hap in (0, 1):
                while pair[hap][cursors[hap]].end <= interval_start:
                    cursors[hap] += 1
            interval_end = min(cells[-1][1], pair[0][cursors[0]].end, pair[1][cursors[1]].end)
            first = int(np.searchsorted(cell_right, interval_start, side="right"))
            stop = int(np.searchsorted(cell_left, interval_end, side="left"))
            indexes = np.arange(first, stop, dtype=np.int64)
            starts = np.maximum(cell_left[indexes], interval_start)
            ends = np.minimum(cell_right[indexes], interval_end)
            weights = np.interp(ends, map_position, map_cm) - np.interp(starts, map_position, map_cm)
            positive = weights > 0
            marker_parts.append(indexes[positive])
            weight_parts.append(weights[positive])
            label0_parts.append(np.full(positive.sum(), ancestry_index[pair[0][cursors[0]].ancestry], dtype=np.int8))
            label1_parts.append(np.full(positive.sum(), ancestry_index[pair[1][cursors[1]].ancestry], dtype=np.int8))
            interval_start = interval_end
        marker_index = np.concatenate(marker_parts)
        weights = np.concatenate(weight_parts)
        label0 = np.concatenate(label0_parts)
        label1 = np.concatenate(label1_parts)
        require(abs(float(weights.sum()) - cm_span) <= max(1e-9, cm_span * 1e-9),
                "integrated truth span differs")
        predicted = probabilities[sample_index, :, marker_index, :]
        truth_one_hot = np.zeros_like(predicted, dtype=np.float64)
        truth_one_hot[np.arange(len(weights)), 0, label0] = 1.0
        truth_one_hot[np.arange(len(weights)), 1, label1] = 1.0
        truth_dose = truth_one_hot.sum(axis=1)
        error = np.abs(predicted.sum(axis=1) - truth_dose) / 2.0
        sample_mae = np.sum(weights[:, None] * error, axis=0)
        mae_num += sample_mae
        present = truth_dose > 0
        mae_present_num += np.sum(weights[:, None] * error * present, axis=0)
        mae_present_den += np.sum(weights[:, None] * present, axis=0)
        brier_num += float(np.sum(weights[:, None, None] * np.square(predicted - truth_one_hot)) / 4.0)
        hard_part = np.argmax(predicted, axis=2)
        for truth0 in range(3):
            for truth1 in range(3):
                truth_mask = (label0 == truth0) & (label1 == truth1)
                if not truth_mask.any():
                    continue
                truth_state = base._diploid_class((ANCESTRIES[truth0], ANCESTRIES[truth1]))
                for pred0 in range(3):
                    for pred1 in range(3):
                        selected = truth_mask & (hard_part[:, 0] == pred0) & (hard_part[:, 1] == pred1)
                        if selected.any():
                            pred_state = base._diploid_class((ANCESTRIES[pred0], ANCESTRIES[pred1]))
                            confusion[(truth_state, pred_state)] += float(weights[selected].sum())
        per_person_mae.append((sample_mae / cm_span).tolist())
        truth_boundaries = tuple(base._truth_boundaries(pair[hap], genetic_map) for hap in (0, 1))
        predicted_boundaries = [[], []]
        hard = np.argmax(probabilities[sample_index], axis=2)
        for hap in (0, 1):
            changes = np.flatnonzero(hard[hap, 1:] != hard[hap, :-1]) + 1
            for marker in changes.tolist():
                before, after = int(hard[hap, marker - 1]), int(hard[hap, marker])
                if before != after:
                    midpoint = (target.positions[marker - 1] + target.positions[marker]) // 2
                    predicted_boundaries[hap].append(
                        base.Boundary(genetic_map.cm_at(midpoint), ANCESTRIES[before], ANCESTRIES[after])
                    )
        for tolerance in tolerances:
            matched = sum(len(base.ordered_boundary_pairs(truth_boundaries[hap],
                                                           predicted_boundaries[hap], tolerance))
                          for hap in (0, 1))
            boundary_totals[tolerance].update(
                truth=sum(len(value) for value in truth_boundaries),
                predicted=sum(len(value) for value in predicted_boundaries), matched=matched,
            )
    boundary = {}
    for tolerance, counts in boundary_totals.items():
        denominator = 2 * counts["matched"] + (counts["predicted"] - counts["matched"]) + (counts["truth"] - counts["matched"])
        boundary[f"{tolerance:.1f}"] = {
            **dict(counts), "f1": 2 * counts["matched"] / denominator if denominator else 1.0,
            "false_transitions_per_cM": (counts["predicted"] - counts["matched"]) /
                                         (2.0 * len(target.samples) * cm_span),
        }
    macro_f1 = flare_io._macro_f1(confusion)
    return {
        "schema_version": "1.0.0", "stage": "M33_DEVELOPMENT_SCORING",
        "sample_count": len(target.samples), "marker_count": len(target.loci), "cm_span": cm_span,
        "boundary": boundary,
        "macro_ancestry_dose_MAE": float(np.mean(mae_num / (len(target.samples) * cm_span))),
        "per_ancestry_MAE": {ancestry: float(mae_num[index] / (len(target.samples) * cm_span))
                             for index, ancestry in enumerate(ANCESTRIES)},
        "per_ancestry_truth_present_MAE": {
            ancestry: (float(mae_present_num[index] / mae_present_den[index])
                       if mae_present_den[index] else None)
            for index, ancestry in enumerate(ANCESTRIES)
        },
        "brier": brier_num / (len(target.samples) * cm_span),
        "diploid_macro_f1": macro_f1["macro_f1_fixed_six"],
        "per_person_macro_MAE": [float(np.mean(value)) for value in per_person_mae],
        "input_sha256": {"prediction": sha256_file(prediction), "target_vcf": sha256_file(target_vcf),
                         "truth": sha256_file(truth_path), "map": sha256_file(map_path)},
        "truth_opened_only_by_scorer": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    project = sub.add_parser("project-flare")
    project.add_argument("--flare-vcf", type=Path, required=True)
    project.add_argument("--target-vcf", type=Path, required=True)
    project.add_argument("--map", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--prediction", type=Path, required=True)
    scoring.add_argument("--target-vcf", type=Path, required=True)
    scoring.add_argument("--truth", type=Path, required=True)
    scoring.add_argument("--map", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "project-flare":
        flare_to_prediction(args.flare_vcf, args.target_vcf, args.map, args.output)
        result = {"status": "PASS_FLARE_PROJECTED", "output": str(args.output)}
    else:
        result = score(args.prediction, args.target_vcf, args.truth, args.map)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {"status": "PASS_SCORED", "boundary_f1_0.2": result["boundary"]["0.2"]["f1"]}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
