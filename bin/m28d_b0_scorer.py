#!/usr/bin/env python3
"""Score frozen M28C B0 predictions against interval LAI truth.

The primary estimand is a base-pair weighted, phase-invariant diploid ancestry
dosage error.  Prediction windows are expanded through the exact discrete
Voronoi support of the ordered B0 markers, then split at every truth boundary.
The implementation intentionally emits no ceiling, SESOI, p-value or BR/BS
decision: the inspected M28 seed is descriptive only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO


ANCESTRIES = ("AFR", "EUR", "ASIA")
ANCESTRY_INDEX = {name: index for index, name in enumerate(ANCESTRIES)}
DIPLOID_CLASSES = ("AA", "EE", "SS", "AE", "AS", "ES")
ANCESTRY_LETTER = {"AFR": "A", "EUR": "E", "ASIA": "S"}
FB_COLUMN_RE = re.compile(r"^(?P<sample>T\d+):::hap(?P<hap>[12]):::(?P<anc>AFR|EUR|ASIA)$")
MSP_COLUMN_RE = re.compile(r"^(?P<sample>T\d+)\.(?P<hap>[01])$")
TRUTH_HAP_RE = re.compile(r"^(?P<sample>T\d+)_h(?P<hap>[01])$")


@dataclass(frozen=True)
class TruthSegment:
    start: int
    end: int
    ancestry: str


@dataclass(frozen=True)
class MapPoint:
    position: int
    cm: float


@dataclass(frozen=True)
class PredictionWindow:
    left: int
    right: int
    marker_start: int
    marker_end: int
    n_markers: int
    probabilities: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
    hard_labels: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class Boundary:
    cm: float
    before: str
    after: str


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} sha256 mismatch: {observed} != {expected}")
    return observed


def normalize_chrom(value: str) -> str:
    return value[3:] if value.startswith("chr") else value


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("stage") != "M28D_B0_DESCRIPTIVE_SCORING":
        raise ValueError("unexpected scoring contract stage")
    if contract.get("status") != "PRE_FROZEN_AMENDED_BEFORE_TRUTH_ACCESS":
        raise ValueError("scoring contract is not frozen before truth access")
    if contract.get("version") != 2:
        raise ValueError("unexpected scoring contract version")
    if contract["unresolved_before_inference"]["SESOI"].startswith("No defensible") is False:
        raise ValueError("contract must not silently fix a SESOI")
    return contract


def load_b0_markers(path: Path, expected_chrom: str) -> tuple[list[int], list[float]]:
    positions: list[int] = []
    cms: list[float] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"arm_component", "chrom", "position", "cm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("B0 marker table has an unexpected header")
        for line_number, row in enumerate(reader, start=2):
            if row["arm_component"] != "B0":
                raise ValueError(f"B0:{line_number}: unexpected arm component")
            if normalize_chrom(row["chrom"]) != normalize_chrom(expected_chrom):
                raise ValueError(f"B0:{line_number}: chromosome mismatch")
            position = int(row["position"])
            cm = float(row["cm"])
            if not math.isfinite(cm):
                raise ValueError(f"B0:{line_number}: nonfinite cM")
            if positions and position <= positions[-1]:
                raise ValueError(f"B0:{line_number}: positions are not strictly increasing")
            if cms and cm < cms[-1]:
                raise ValueError(f"B0:{line_number}: cM is decreasing")
            positions.append(position)
            cms.append(cm)
    if not positions:
        raise ValueError("B0 marker table is empty")
    return positions, cms


def discrete_voronoi(positions: Sequence[int]) -> list[tuple[int, int]]:
    """Return contiguous half-open integer-bp cells with midpoint ties to the left."""
    if not positions:
        raise ValueError("cannot partition an empty marker sequence")
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("marker positions must be strictly increasing")
    cells: list[tuple[int, int]] = []
    left = positions[0]
    for index, position in enumerate(positions):
        if index + 1 == len(positions):
            right = positions[-1] + 1
        else:
            right = (position + positions[index + 1]) // 2 + 1
        if right <= left:
            raise ValueError("nonpositive Voronoi cell")
        cells.append((left, right))
        left = right
    expected = positions[-1] - positions[0] + 1
    observed = sum(right - left for left, right in cells)
    if cells[0][0] != positions[0] or cells[-1][1] != positions[-1] + 1 or observed != expected:
        raise ValueError("Voronoi cells do not reconstruct the inclusive marker domain")
    return cells


class GeneticMap:
    def __init__(self, points: Sequence[MapPoint]):
        if len(points) < 2:
            raise ValueError("genetic map requires at least two points")
        self.positions = [point.position for point in points]
        self.cms = [point.cm for point in points]
        if any(not math.isfinite(value) for value in self.cms):
            raise ValueError("genetic map contains a nonfinite cM value")
        if any(right <= left for left, right in zip(self.positions, self.positions[1:])):
            raise ValueError("genetic-map positions are not strictly increasing")
        if any(right < left for left, right in zip(self.cms, self.cms[1:])):
            raise ValueError("genetic map is decreasing")

    def cm_at(self, position: int | float) -> float:
        import bisect

        if position < self.positions[0] or position > self.positions[-1]:
            raise ValueError(f"position {position} lies outside the genetic map")
        index = bisect.bisect_right(self.positions, position) - 1
        if index == len(self.positions) - 1 or position == self.positions[index]:
            return self.cms[index]
        x0, x1 = self.positions[index], self.positions[index + 1]
        y0, y1 = self.cms[index], self.cms[index + 1]
        fraction = (position - x0) / (x1 - x0)
        value = y0 + fraction * (y1 - y0)
        if not math.isfinite(value):
            raise ValueError("nonfinite interpolated cM")
        return value


def load_genetic_map(path: Path, expected_chrom: str) -> GeneticMap:
    points: list[MapPoint] = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"map:{line_number}: expected chrom position cM")
            if normalize_chrom(fields[0]) != normalize_chrom(expected_chrom):
                raise ValueError(f"map:{line_number}: chromosome mismatch")
            position = int(fields[1])
            cm = float(fields[2])
            if not math.isfinite(cm):
                raise ValueError(f"map:{line_number}: nonfinite cM")
            points.append(MapPoint(position, cm))
    return GeneticMap(points)


def _read_noncomment_table(path: Path) -> tuple[list[str], list[list[str]], list[str]]:
    comments: list[str] = []
    header: list[str] | None = None
    rows: list[list[str]] = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            if stripped.startswith("#"):
                comments.append(stripped)
                continue
            fields = stripped.split("\t")
            if header is None:
                header = fields
            else:
                if len(fields) != len(header):
                    raise ValueError(f"{path}:{line_number}: column count mismatch")
                rows.append(fields)
    if header is None:
        raise ValueError(f"{path}: no header")
    return header, rows, comments


def load_fb(path: Path) -> tuple[list[int], list[float], list[str], list[dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]]]:
    header, rows, _ = _read_noncomment_table(path)
    if header[:4] != ["chromosome", "physical position", "genetic_position", "genetic_marker_index"]:
        raise ValueError("FB has an unexpected fixed header")
    parsed_columns: list[tuple[str, int, str]] = []
    for column in header[4:]:
        match = FB_COLUMN_RE.fullmatch(column)
        if match is None:
            raise ValueError(f"FB unexpected probability column: {column}")
        parsed_columns.append((match.group("sample"), int(match.group("hap")) - 1, match.group("anc")))
    samples = sorted({sample for sample, _, _ in parsed_columns})
    expected_columns = {(sample, hap, anc) for sample in samples for hap in (0, 1) for anc in ANCESTRIES}
    if set(parsed_columns) != expected_columns or len(parsed_columns) != len(expected_columns):
        raise ValueError("FB sample/haplotype/ancestry columns are incomplete or duplicated")

    positions: list[int] = []
    cms: list[float] = []
    windows: list[dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]] = []
    for row_number, row in enumerate(rows, start=2):
        if normalize_chrom(row[0]) != "22":
            raise ValueError(f"FB:{row_number}: unexpected chromosome")
        position = int(row[1])
        cm = float(row[2])
        if not math.isfinite(cm):
            raise ValueError(f"FB:{row_number}: nonfinite genetic position")
        if positions and position <= positions[-1]:
            raise ValueError(f"FB:{row_number}: positions are not increasing")
        if cms and cm < cms[-1]:
            raise ValueError(f"FB:{row_number}: genetic positions are decreasing")
        values: dict[str, list[list[float | None]]] = {
            sample: [[None] * len(ANCESTRIES), [None] * len(ANCESTRIES)] for sample in samples
        }
        for raw, (sample, hap, ancestry) in zip(row[4:], parsed_columns):
            probability = float(raw)
            if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError(f"FB:{row_number}: invalid probability")
            values[sample][hap][ANCESTRY_INDEX[ancestry]] = probability
        frozen: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        for sample, haplotypes in values.items():
            pair: list[tuple[float, float, float]] = []
            for haplotype in haplotypes:
                if any(value is None for value in haplotype):
                    raise ValueError(f"FB:{row_number}: incomplete probability vector")
                vector = tuple(float(value) for value in haplotype)
                if abs(sum(vector) - 1.0) > 1e-6:
                    raise ValueError(f"FB:{row_number}: probabilities do not sum to one")
                pair.append(vector)  # type: ignore[arg-type]
            frozen[sample] = (pair[0], pair[1])
        positions.append(position)
        cms.append(cm)
        windows.append(frozen)
    return positions, cms, samples, windows


def load_msp(path: Path) -> tuple[list[dict[str, int | float]], list[str], list[dict[str, tuple[str, str]]]]:
    data_header: list[str] | None = None
    data_rows: list[list[str]] = []
    code_map: dict[str, str] = {}
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            if stripped.startswith("#Subpopulation order/codes:"):
                payload = stripped.split(":", 1)[1].strip()
                for token in payload.split():
                    ancestry, code = token.split("=", 1)
                    if ancestry not in ANCESTRIES:
                        raise ValueError(f"MSP:{line_number}: unexpected ancestry {ancestry}")
                    code_map[code] = ancestry
                continue
            if stripped.startswith("#chm\t"):
                data_header = stripped[1:].split("\t")
                continue
            if stripped.startswith("#"):
                continue
            if data_header is None:
                raise ValueError("MSP data appeared before its header")
            fields = stripped.split("\t")
            if len(fields) != len(data_header):
                raise ValueError(f"MSP:{line_number}: column count mismatch")
            data_rows.append(fields)
    if data_header is None or set(code_map.values()) != set(ANCESTRIES):
        raise ValueError("MSP lacks a complete header or ancestry code map")
    if data_header[:6] != ["chm", "spos", "epos", "sgpos", "egpos", "n snps"]:
        raise ValueError("MSP has an unexpected fixed header")
    parsed_columns: list[tuple[str, int]] = []
    for column in data_header[6:]:
        match = MSP_COLUMN_RE.fullmatch(column)
        if match is None:
            raise ValueError(f"MSP unexpected haplotype column: {column}")
        parsed_columns.append((match.group("sample"), int(match.group("hap"))))
    samples = sorted({sample for sample, _ in parsed_columns})
    expected_columns = {(sample, hap) for sample in samples for hap in (0, 1)}
    if set(parsed_columns) != expected_columns or len(parsed_columns) != len(expected_columns):
        raise ValueError("MSP sample/haplotype columns are incomplete or duplicated")

    metadata: list[dict[str, int | float]] = []
    labels: list[dict[str, tuple[str, str]]] = []
    for row_number, row in enumerate(data_rows, start=1):
        if normalize_chrom(row[0]) != "22":
            raise ValueError(f"MSP row {row_number}: unexpected chromosome")
        record = {
            "spos": int(row[1]),
            "epos": int(row[2]),
            "sgpos": float(row[3]),
            "egpos": float(row[4]),
            "n_snps": int(row[5]),
        }
        if (
            record["n_snps"] <= 0
            or record["epos"] < record["spos"]
            or not math.isfinite(float(record["sgpos"]))
            or not math.isfinite(float(record["egpos"]))
            or record["egpos"] < record["sgpos"]
        ):
            raise ValueError(f"MSP row {row_number}: invalid interval")
        if metadata and (
            record["spos"] <= metadata[-1]["epos"]
            or record["sgpos"] < metadata[-1]["sgpos"]
            or record["egpos"] < metadata[-1]["egpos"]
        ):
            raise ValueError(f"MSP row {row_number}: windows are not ordered")
        mutable = {sample: [None, None] for sample in samples}
        for raw, (sample, hap) in zip(row[6:], parsed_columns):
            if raw not in code_map:
                raise ValueError(f"MSP row {row_number}: unknown ancestry code {raw}")
            mutable[sample][hap] = code_map[raw]
        frozen: dict[str, tuple[str, str]] = {}
        for sample, pair in mutable.items():
            if pair[0] is None or pair[1] is None:
                raise ValueError(f"MSP row {row_number}: incomplete hard labels")
            frozen[sample] = (str(pair[0]), str(pair[1]))
        metadata.append(record)
        labels.append(frozen)
    return metadata, samples, labels


def build_prediction_windows(
    marker_positions: Sequence[int],
    fb_positions: Sequence[int],
    fb_samples: Sequence[str],
    fb_probabilities: Sequence[dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]],
    msp_metadata: Sequence[dict[str, int | float]],
    msp_samples: Sequence[str],
    msp_labels: Sequence[dict[str, tuple[str, str]]],
) -> list[PredictionWindow]:
    if list(fb_samples) != list(msp_samples):
        raise ValueError("FB/MSP sample orders differ")
    if not (len(fb_positions) == len(fb_probabilities) == len(msp_metadata) == len(msp_labels)):
        raise ValueError("FB/MSP window counts differ")
    cells = discrete_voronoi(marker_positions)
    windows: list[PredictionWindow] = []
    offset = 0
    for index, (fb_position, probabilities, metadata, labels) in enumerate(
        zip(fb_positions, fb_probabilities, msp_metadata, msp_labels)
    ):
        count = int(metadata["n_snps"])
        end_offset = offset + count
        if end_offset > len(marker_positions):
            raise ValueError("MSP n_snps exceeds B0 markers")
        if int(metadata["spos"]) != marker_positions[offset]:
            raise ValueError(f"window {index}: MSP start does not match B0 marker")
        if int(metadata["epos"]) != marker_positions[end_offset - 1]:
            raise ValueError(f"window {index}: MSP end does not match B0 marker")
        if not (int(metadata["spos"]) <= fb_position <= int(metadata["epos"])):
            raise ValueError(f"window {index}: FB representative lies outside MSP markers")
        for sample in fb_samples:
            for hap in (0, 1):
                if _argmax_ancestry(probabilities[sample][hap]) != labels[sample][hap]:
                    raise ValueError(f"window {index}: FB argmax and MSP hard label differ")
        windows.append(
            PredictionWindow(
                left=cells[offset][0],
                right=cells[end_offset - 1][1],
                marker_start=offset,
                marker_end=end_offset,
                n_markers=count,
                probabilities=probabilities,
                hard_labels=labels,
            )
        )
        offset = end_offset
    if offset != len(marker_positions):
        raise ValueError("MSP n_snps does not consume all B0 markers")
    for before, after in zip(windows, windows[1:]):
        if before.right != after.left:
            raise ValueError("prediction windows have a gap or overlap")
    return windows


def validate_genetic_coordinates(
    marker_positions: Sequence[int],
    marker_cms: Sequence[float],
    fb_positions: Sequence[int],
    fb_cms: Sequence[float],
    msp_metadata: Sequence[dict[str, int | float]],
    genetic_map: GeneticMap,
    marker_tolerance_cm: float,
    msp_tolerance_cm: float,
) -> None:
    """Authenticate coordinate systems while the map remains the scoring authority.

    Gnomix writes MSP endpoints rounded to five decimal places. FB genetic
    positions are window representatives, so they must be monotone (checked by
    ``load_fb``) and lie inside the corresponding MSP interval; they are not
    map coordinates for scoring boundary distances.
    """
    if not (
        len(marker_positions) == len(marker_cms)
        and len(fb_positions) == len(fb_cms) == len(msp_metadata)
    ):
        raise ValueError("genetic-coordinate arrays have inconsistent dimensions")
    if marker_tolerance_cm < 0 or msp_tolerance_cm < 0:
        raise ValueError("genetic-coordinate tolerances must be nonnegative")
    for index, (position, observed_cm) in enumerate(zip(marker_positions, marker_cms)):
        if abs(observed_cm - genetic_map.cm_at(position)) > marker_tolerance_cm:
            raise ValueError(f"B0 marker {index}: cM disagrees with authenticated map")
    for index, (position, fb_cm, metadata) in enumerate(
        zip(fb_positions, fb_cms, msp_metadata)
    ):
        if not int(metadata["spos"]) <= position <= int(metadata["epos"]):
            raise ValueError(f"window {index}: FB physical position lies outside MSP")
        if not float(metadata["sgpos"]) <= fb_cm <= float(metadata["egpos"]):
            raise ValueError(f"window {index}: FB genetic representative lies outside MSP")
        start_delta = abs(
            float(metadata["sgpos"]) - genetic_map.cm_at(int(metadata["spos"]))
        )
        end_delta = abs(
            float(metadata["egpos"]) - genetic_map.cm_at(int(metadata["epos"]))
        )
        if start_delta > msp_tolerance_cm or end_delta > msp_tolerance_cm:
            raise ValueError(f"window {index}: MSP cM endpoints disagree with authenticated map")


def load_truth(
    path: Path,
    expected_samples: Sequence[str],
    chrom: str,
    domain_start: int,
    domain_end_exclusive: int,
) -> dict[str, tuple[list[TruthSegment], list[TruthSegment]]]:
    grouped: dict[tuple[str, int], list[TruthSegment]] = defaultdict(list)
    expected_keys = {(sample, hap) for sample in expected_samples for hap in (0, 1)}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"target_haplotype", "chrom", "start_bp", "end_bp_exclusive", "ancestry"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("truth has an unexpected header")
        for line_number, row in enumerate(reader, start=2):
            match = TRUTH_HAP_RE.fullmatch(row["target_haplotype"])
            if match is None:
                raise ValueError(f"truth:{line_number}: invalid target haplotype")
            if normalize_chrom(row["chrom"]) != normalize_chrom(chrom):
                raise ValueError(f"truth:{line_number}: chromosome mismatch")
            ancestry = row["ancestry"]
            if ancestry not in ANCESTRIES:
                raise ValueError(f"truth:{line_number}: unexpected ancestry")
            start, end = int(row["start_bp"]), int(row["end_bp_exclusive"])
            if end <= start:
                raise ValueError(f"truth:{line_number}: nonpositive interval")
            key = (match.group("sample"), int(match.group("hap")))
            if key not in expected_keys:
                raise ValueError(f"truth:{line_number}: unexpected sample or haplotype {key}")
            clipped_start = max(start, domain_start)
            clipped_end = min(end, domain_end_exclusive)
            if clipped_start < clipped_end:
                grouped[key].append(TruthSegment(clipped_start, clipped_end, ancestry))
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise ValueError(f"truth sample/haplotype mismatch: missing={missing}, extra={extra}")
    output: dict[str, tuple[list[TruthSegment], list[TruthSegment]]] = {}
    for sample in expected_samples:
        pair: list[list[TruthSegment]] = []
        for hap in (0, 1):
            segments = sorted(grouped[(sample, hap)], key=lambda segment: (segment.start, segment.end))
            cursor = domain_start
            compact: list[TruthSegment] = []
            for segment in segments:
                if segment.start != cursor:
                    raise ValueError(f"truth {sample} h{hap}: gap or overlap at {cursor}")
                if compact and compact[-1].ancestry == segment.ancestry:
                    previous = compact.pop()
                    compact.append(TruthSegment(previous.start, segment.end, segment.ancestry))
                else:
                    compact.append(segment)
                cursor = segment.end
            if cursor != domain_end_exclusive:
                raise ValueError(f"truth {sample} h{hap}: incomplete terminal coverage")
            pair.append(compact)
        output[sample] = (pair[0], pair[1])
    return output


def ancestry_at(segments: Sequence[TruthSegment], position: int) -> str:
    import bisect

    starts = [segment.start for segment in segments]
    index = bisect.bisect_right(starts, position) - 1
    if index < 0 or not (segments[index].start <= position < segments[index].end):
        raise ValueError(f"truth does not cover position {position}")
    return segments[index].ancestry


def _diploid_class(labels: tuple[str, str]) -> str:
    letters = sorted(ANCESTRY_LETTER[label] for label in labels)
    if letters[0] == letters[1]:
        return letters[0] * 2
    return "".join(letters)


def _argmax_ancestry(probabilities: Sequence[float]) -> str:
    maximum = max(probabilities)
    return ANCESTRIES[next(index for index, value in enumerate(probabilities) if value == maximum)]


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * probability
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _f1_from_precision_recall(
    precision: float | None, recall: float | None
) -> float | None:
    """Return null only for an undefined denominator, and zero for zero matches."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _weighted_f1(confusion: Counter[tuple[str, str]]) -> dict:
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in DIPLOID_CLASSES:
        tp = confusion[(label, label)]
        fp = sum(confusion[(truth, label)] for truth in DIPLOID_CLASSES if truth != label)
        fn = sum(confusion[(label, predicted)] for predicted in DIPLOID_CLASSES if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[(label, predicted)] for predicted in DIPLOID_CLASSES)
        per_class[label] = {
            "support_bp": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    return {"macro_f1_fixed_six": sum(f1_values) / len(f1_values), "per_class": per_class}


def _truth_boundaries(segments: Sequence[TruthSegment], genetic_map: GeneticMap) -> list[Boundary]:
    boundaries: list[Boundary] = []
    for before, after in zip(segments, segments[1:]):
        if before.end != after.start:
            raise ValueError("truth segments are not contiguous")
        if before.ancestry != after.ancestry:
            boundaries.append(Boundary(genetic_map.cm_at(after.start), before.ancestry, after.ancestry))
    return boundaries


def _prediction_boundaries(
    windows: Sequence[PredictionWindow], sample: str, hap: int, genetic_map: GeneticMap
) -> list[Boundary]:
    boundaries: list[Boundary] = []
    previous = windows[0].hard_labels[sample][hap]
    for window in windows[1:]:
        current = window.hard_labels[sample][hap]
        if current != previous:
            boundaries.append(Boundary(genetic_map.cm_at(window.left), previous, current))
        previous = current
    return boundaries


def _better_match(
    left: tuple[int, float, tuple[tuple[int, int], ...]],
    right: tuple[int, float, tuple[tuple[int, int], ...]],
) -> tuple[int, float, tuple[tuple[int, int], ...]]:
    left_key = (left[0], -left[1])
    right_key = (right[0], -right[1])
    if left_key != right_key:
        return left if left_key > right_key else right
    return left if left[2] <= right[2] else right


def ordered_boundary_pairs(
    truth: Sequence[Boundary], prediction: Sequence[Boundary], tolerance_cm: float
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-distance ordered matching by transition label."""
    if tolerance_cm < 0:
        raise ValueError("boundary tolerance must be nonnegative")
    if any(right.cm < left.cm for left, right in zip(truth, truth[1:])):
        raise ValueError("truth boundaries are not ordered")
    if any(right.cm < left.cm for left, right in zip(prediction, prediction[1:])):
        raise ValueError("prediction boundaries are not ordered")
    n, m = len(truth), len(prediction)
    dp: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]] = [
        [(0, 0.0, tuple()) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = _better_match(dp[i - 1][j], dp[i][j - 1])
            truth_boundary = truth[i - 1]
            prediction_boundary = prediction[j - 1]
            distance = abs(truth_boundary.cm - prediction_boundary.cm)
            same_transition = (
                truth_boundary.before == prediction_boundary.before
                and truth_boundary.after == prediction_boundary.after
            )
            if same_transition and distance <= tolerance_cm + 1e-12:
                previous = dp[i - 1][j - 1]
                candidate = (
                    previous[0] + 1,
                    previous[1] + distance,
                    previous[2] + ((i - 1, j - 1),),
                )
                best = _better_match(best, candidate)
            dp[i][j] = best
    return [
        (
            truth_index,
            prediction_index,
            abs(truth[truth_index].cm - prediction[prediction_index].cm),
        )
        for truth_index, prediction_index in dp[n][m][2]
    ]


def ordered_boundary_match(
    truth: Sequence[Boundary], prediction: Sequence[Boundary], tolerance_cm: float
) -> list[float]:
    return [
        distance
        for _, _, distance in ordered_boundary_pairs(truth, prediction, tolerance_cm)
    ]


def _boundary_summary_for_sample(
    truth_pair: tuple[list[Boundary], list[Boundary]],
    prediction_pair: tuple[list[Boundary], list[Boundary]],
    tolerance_cm: float,
) -> dict:
    candidates: list[dict] = []
    for permutation_name, permutation in (("direct", (0, 1)), ("swap", (1, 0))):
        distances: list[float] = []
        matches_by_transition: Counter[tuple[str, str]] = Counter()
        distances_by_transition: dict[tuple[str, str], list[float]] = defaultdict(list)
        for truth_hap, prediction_hap in enumerate(permutation):
            truth_boundaries = truth_pair[truth_hap]
            prediction_boundaries = prediction_pair[prediction_hap]
            pairs = ordered_boundary_pairs(truth_boundaries, prediction_boundaries, tolerance_cm)
            for truth_index, _, distance in pairs:
                transition = (
                    truth_boundaries[truth_index].before,
                    truth_boundaries[truth_index].after,
                )
                matches_by_transition[transition] += 1
                distances_by_transition[transition].append(distance)
                distances.append(distance)
        candidates.append(
            {
                "permutation": permutation_name,
                "matched": len(distances),
                "distance_sum": sum(distances),
                "distances": distances,
                "matches_by_transition": matches_by_transition,
                "distances_by_transition": distances_by_transition,
            }
        )
    candidates.sort(key=lambda item: (-item["matched"], item["distance_sum"], item["permutation"] != "direct"))
    chosen = candidates[0]
    return chosen


def score_objects(
    marker_positions: Sequence[int],
    windows: Sequence[PredictionWindow],
    truth: dict[str, tuple[list[TruthSegment], list[TruthSegment]]],
    genetic_map: GeneticMap,
    tolerances_cm: Sequence[float],
) -> dict:
    if not windows:
        raise ValueError("no prediction windows")
    domain_start, domain_end = marker_positions[0], marker_positions[-1] + 1
    bp_span = domain_end - domain_start
    if windows[0].left != domain_start or windows[-1].right != domain_end:
        raise ValueError("prediction windows do not cover the fixed domain")
    cm_span = genetic_map.cm_at(domain_end) - genetic_map.cm_at(domain_start)
    if cm_span <= 0:
        raise ValueError("genetic span must be positive")
    samples = sorted(windows[0].probabilities)
    if set(samples) != set(truth):
        raise ValueError("prediction/truth samples differ")

    per_ancestry_bp = {ancestry: [] for ancestry in ANCESTRIES}
    per_ancestry_cm = {ancestry: [] for ancestry in ANCESTRIES}
    per_ancestry_mse = {ancestry: [] for ancestry in ANCESTRIES}
    conditional_error = {ancestry: [] for ancestry in ANCESTRIES}
    conditional_support_bp = {ancestry: 0 for ancestry in ANCESTRIES}
    conditional_support_samples = {ancestry: 0 for ancestry in ANCESTRIES}
    brier_by_sample: list[float] = []
    hard_confusion: Counter[tuple[str, str]] = Counter()
    hard_correct_bp = 0
    hard_total_bp = 0
    composition_diagnostic_per_ancestry = {ancestry: [] for ancestry in ANCESTRIES}

    truth_boundary_cache: dict[str, tuple[list[Boundary], list[Boundary]]] = {}
    prediction_boundary_cache: dict[str, tuple[list[Boundary], list[Boundary]]] = {}

    for sample in samples:
        sample_bp_abs = [0.0, 0.0, 0.0]
        sample_cm_abs = [0.0, 0.0, 0.0]
        sample_bp_sq = [0.0, 0.0, 0.0]
        sample_present_abs = [0.0, 0.0, 0.0]
        sample_present_bp = [0, 0, 0]
        truth_dose_bp = [0.0, 0.0, 0.0]
        brier_direct = 0.0
        brier_swap = 0.0
        sample_cm_weight = 0.0
        pieces: list[tuple[int, tuple[int, int, int]]] = []

        truth_pair = truth[sample]
        for window in windows:
            cutpoints = {window.left, window.right}
            for hap_segments in truth_pair:
                for segment in hap_segments:
                    if window.left < segment.start < window.right:
                        cutpoints.add(segment.start)
                    if window.left < segment.end < window.right:
                        cutpoints.add(segment.end)
            ordered = sorted(cutpoints)
            probabilities = window.probabilities[sample]
            for vector in probabilities:
                if (
                    any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in vector)
                    or abs(sum(vector) - 1.0) > 1e-6
                ):
                    raise ValueError("invalid prediction probability vector")
            dosage_hat = [probabilities[0][a] + probabilities[1][a] for a in range(3)]
            if any(value < -1e-12 or value > 2.0 + 1e-12 for value in dosage_hat):
                raise ValueError("predicted dosage outside [0,2]")
            if abs(sum(dosage_hat) - 2.0) > 2e-6:
                raise ValueError("predicted ancestry dosages do not sum to two")
            predicted_hard = (
                _argmax_ancestry(probabilities[0]),
                _argmax_ancestry(probabilities[1]),
            )
            predicted_state = _diploid_class(predicted_hard)
            for left, right in zip(ordered, ordered[1:]):
                if right <= left:
                    raise ValueError("nonpositive scoring overlap")
                truth_labels = (
                    ancestry_at(truth_pair[0], left),
                    ancestry_at(truth_pair[1], left),
                )
                truth_dose = tuple(truth_labels.count(ancestry) for ancestry in ANCESTRIES)
                if sum(truth_dose) != 2:
                    raise ValueError("truth ancestry dosages do not sum to two")
                bp_weight = right - left
                cm_weight = genetic_map.cm_at(right) - genetic_map.cm_at(left)
                if cm_weight < -1e-12:
                    raise ValueError("negative cM overlap weight")
                cm_weight = max(0.0, cm_weight)
                sample_cm_weight += cm_weight
                for ancestry_index in range(3):
                    normalized_error = abs(dosage_hat[ancestry_index] - truth_dose[ancestry_index]) / 2.0
                    squared_error = (
                        dosage_hat[ancestry_index] / 2.0 - truth_dose[ancestry_index] / 2.0
                    ) ** 2
                    sample_bp_abs[ancestry_index] += bp_weight * normalized_error
                    sample_cm_abs[ancestry_index] += cm_weight * normalized_error
                    sample_bp_sq[ancestry_index] += bp_weight * squared_error
                    truth_dose_bp[ancestry_index] += bp_weight * truth_dose[ancestry_index]
                    if truth_dose[ancestry_index] > 0:
                        sample_present_abs[ancestry_index] += bp_weight * normalized_error
                        sample_present_bp[ancestry_index] += bp_weight
                direct_piece = swap_piece = 0.0
                for predicted_hap in (0, 1):
                    direct_truth = truth_labels[predicted_hap]
                    swap_truth = truth_labels[1 - predicted_hap]
                    direct_piece += sum(
                        (probabilities[predicted_hap][a] - (ANCESTRIES[a] == direct_truth)) ** 2
                        for a in range(3)
                    ) / 2.0
                    swap_piece += sum(
                        (probabilities[predicted_hap][a] - (ANCESTRIES[a] == swap_truth)) ** 2
                        for a in range(3)
                    ) / 2.0
                brier_direct += bp_weight * direct_piece / 2.0
                brier_swap += bp_weight * swap_piece / 2.0
                truth_state = _diploid_class(truth_labels)
                hard_confusion[(truth_state, predicted_state)] += bp_weight
                hard_correct_bp += bp_weight * (truth_state == predicted_state)
                hard_total_bp += bp_weight
                pieces.append((bp_weight, truth_dose))

        if sum(weight for weight, _ in pieces) != bp_span:
            raise ValueError(f"sample {sample}: bp weights do not reconstruct the domain")
        if not math.isclose(sample_cm_weight, cm_span, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"sample {sample}: cM weights do not reconstruct the domain")
        for ancestry_index, ancestry in enumerate(ANCESTRIES):
            per_ancestry_bp[ancestry].append(sample_bp_abs[ancestry_index] / bp_span)
            per_ancestry_mse[ancestry].append(sample_bp_sq[ancestry_index] / bp_span)
            per_ancestry_cm[ancestry].append(
                sample_cm_abs[ancestry_index] / cm_span if cm_span > 0 else 0.0
            )
            conditional_error[ancestry].append(
                sample_present_abs[ancestry_index] / sample_present_bp[ancestry_index]
                if sample_present_bp[ancestry_index]
                else None
            )
            conditional_support_bp[ancestry] += sample_present_bp[ancestry_index]
            conditional_support_samples[ancestry] += int(sample_present_bp[ancestry_index] > 0)
            global_q = truth_dose_bp[ancestry_index] / (2.0 * bp_span)
            composition_diagnostic_error = sum(
                weight * abs(global_q - truth_piece[ancestry_index] / 2.0)
                for weight, truth_piece in pieces
            ) / bp_span
            composition_diagnostic_per_ancestry[ancestry].append(
                composition_diagnostic_error
            )
        brier_by_sample.append(min(brier_direct, brier_swap) / bp_span)
        truth_boundary_cache[sample] = (
            _truth_boundaries(truth_pair[0], genetic_map),
            _truth_boundaries(truth_pair[1], genetic_map),
        )
        prediction_boundary_cache[sample] = (
            _prediction_boundaries(windows, sample, 0, genetic_map),
            _prediction_boundaries(windows, sample, 1, genetic_map),
        )

    def mean(values: Iterable[float]) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized) if materialized else 0.0

    bp_by_ancestry = {ancestry: mean(values) for ancestry, values in per_ancestry_bp.items()}
    cm_by_ancestry = {ancestry: mean(values) for ancestry, values in per_ancestry_cm.items()}
    mse_by_ancestry = {ancestry: mean(values) for ancestry, values in per_ancestry_mse.items()}
    conditional_by_ancestry = {}
    for ancestry, values in conditional_error.items():
        supported = [value for value in values if value is not None]
        conditional_by_ancestry[ancestry] = {
            "mae_mean_across_supported_samples": mean(supported) if supported else None,
            "support_bp_across_samples": conditional_support_bp[ancestry],
            "samples_with_support": conditional_support_samples[ancestry],
            "aggregation": "within-sample conditional MAE, then equal mean over samples with support",
        }
    composition_diagnostic_by_ancestry = {
        ancestry: mean(values)
        for ancestry, values in composition_diagnostic_per_ancestry.items()
    }

    boundary_results: dict[str, dict] = {}
    for tolerance in tolerances_cm:
        all_distances: list[float] = []
        total_truth = total_prediction = total_matched = 0
        permutations: Counter[str] = Counter()
        transition_truth: Counter[tuple[str, str]] = Counter()
        transition_prediction: Counter[tuple[str, str]] = Counter()
        transition_matched: Counter[tuple[str, str]] = Counter()
        transition_distances: dict[tuple[str, str], list[float]] = defaultdict(list)
        for sample in samples:
            truth_pair = truth_boundary_cache[sample]
            prediction_pair = prediction_boundary_cache[sample]
            selected = _boundary_summary_for_sample(truth_pair, prediction_pair, tolerance)
            permutations[selected["permutation"]] += 1
            all_distances.extend(selected["distances"])
            total_matched += selected["matched"]
            total_truth += sum(len(values) for values in truth_pair)
            total_prediction += sum(len(values) for values in prediction_pair)
            for hap_boundaries in truth_pair:
                transition_truth.update((item.before, item.after) for item in hap_boundaries)
            for hap_boundaries in prediction_pair:
                transition_prediction.update((item.before, item.after) for item in hap_boundaries)
            transition_matched.update(selected["matches_by_transition"])
            for transition, distances in selected["distances_by_transition"].items():
                transition_distances[transition].extend(distances)
        precision = total_matched / total_prediction if total_prediction else None
        recall = total_matched / total_truth if total_truth else None
        f1 = _f1_from_precision_recall(precision, recall)
        per_transition: dict[str, dict] = {}
        for before in ANCESTRIES:
            for after in ANCESTRIES:
                if before == after:
                    continue
                transition = (before, after)
                n_truth = transition_truth[transition]
                n_prediction = transition_prediction[transition]
                n_matched = transition_matched[transition]
                transition_precision = n_matched / n_prediction if n_prediction else None
                transition_recall = n_matched / n_truth if n_truth else None
                transition_f1 = _f1_from_precision_recall(
                    transition_precision, transition_recall
                )
                distances = transition_distances[transition]
                per_transition[f"{before}->{after}"] = {
                    "truth_boundaries": n_truth,
                    "predicted_boundaries": n_prediction,
                    "matched_boundaries": n_matched,
                    "missed_truth_boundaries": n_truth - n_matched,
                    "extra_predicted_boundaries": n_prediction - n_matched,
                    "precision": transition_precision,
                    "recall": transition_recall,
                    "f1": transition_f1,
                    "matched_distance_median_cm": _percentile(distances, 0.5),
                    "matched_distance_p95_cm": _percentile(distances, 0.95),
                }
        boundary_results[f"{tolerance:.1f}"] = {
            "tolerance_cm": tolerance,
            "truth_boundaries": total_truth,
            "predicted_boundaries": total_prediction,
            "matched_boundaries": total_matched,
            "missed_truth_boundaries": total_truth - total_matched,
            "extra_predicted_boundaries": total_prediction - total_matched,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_distance_median_cm": _percentile(all_distances, 0.5),
            "matched_distance_p95_cm": _percentile(all_distances, 0.95),
            "global_phase_permutation_counts": dict(sorted(permutations.items())),
            "zero_denominator_policy": "undefined precision/recall/F1 are null, never zero",
            "per_directed_transition": per_transition,
        }

    hard_summary = _weighted_f1(hard_confusion)
    hard_summary["accuracy"] = hard_correct_bp / hard_total_bp if hard_total_bp else 0.0
    hard_summary["total_bp_across_samples"] = hard_total_bp

    return {
        "domain": {
            "start_bp_inclusive": domain_start,
            "end_bp_inclusive": domain_end - 1,
            "bp_span": bp_span,
            "cm_span": cm_span,
            "markers": len(marker_positions),
            "windows": len(windows),
            "samples": len(samples),
        },
        "primary": {
            "name": "bp_weighted_macro_normalized_ancestry_dosage_mae",
            "per_ancestry": bp_by_ancestry,
            "macro": mean(bp_by_ancestry.values()),
            "range_max": 2.0 / 3.0,
            "conditional_on_truth_present": conditional_by_ancestry,
        },
        "secondary": {
            "cm_weighted_normalized_ancestry_dosage_mae": {
                "per_ancestry": cm_by_ancestry,
                "macro": mean(cm_by_ancestry.values()),
            },
            "bp_weighted_normalized_ancestry_dosage_mse": {
                "per_ancestry": mse_by_ancestry,
                "macro": mean(mse_by_ancestry.values()),
            },
            "phase_aligned_haplotype_brier": mean(brier_by_sample),
            "hard_unordered_diploid_state": hard_summary,
            "boundaries": boundary_results,
        },
        "truth_informed_diagnostic": {
            "name": "truth_informed_global_composition_diagnostic",
            "not_a_progression_gate": True,
            "primary_scale_per_ancestry": composition_diagnostic_by_ancestry,
            "primary_scale_macro": mean(composition_diagnostic_by_ancestry.values()),
        },
    }


def validate_contract_dimensions(contract: dict, markers: Sequence[int], windows: Sequence[PredictionWindow]) -> None:
    fixed = contract["fixed_domain"]
    if len(markers) != fixed["markers"] or len(windows) != fixed["windows"]:
        raise ValueError("observed marker/window dimensions differ from the contract")
    if markers[0] != fixed["first_b0_marker"] or markers[-1] != fixed["last_b0_marker"]:
        raise ValueError("observed B0 domain differs from the contract")
    span = markers[-1] - markers[0] + 1
    if span != fixed["expected_bp_weight"]:
        raise ValueError("observed bp span differs from the contract")
    samples = sorted(windows[0].probabilities)
    if len(samples) != fixed["target_samples"]:
        raise ValueError("observed target count differs from the contract")
    if sum(window.n_markers for window in windows) != fixed["markers"]:
        raise ValueError("prediction windows do not consume the contracted marker count")


def validate_seed_lineage(
    contract: dict,
    replicate: str,
    simulation_manifest_path: Path,
    b0_preflight_manifest_path: Path,
    ingest_report_path: Path,
    inference_manifest_path: Path,
) -> None:
    seed = contract["seed_policy"]["seed"]
    expected = contract["authenticated_inputs"]
    simulation = json.loads(simulation_manifest_path.read_text(encoding="utf-8"))
    preflight = json.loads(b0_preflight_manifest_path.read_text(encoding="utf-8"))
    ingest = json.loads(ingest_report_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    if (
        simulation.get("stage") != "M28_LAI_SIMULATION_PREFLIGHT"
        or simulation.get("params", {}).get("root_seed") != seed
        or simulation.get("sha256", {}).get("m28_lai_truth.tsv.gz") != expected["truth_sha256"]
        or simulation.get("inputs", {}).get("genetic.map.chr22")
        != expected["genetic_map_sha256"]
    ):
        raise ValueError("simulation manifest does not bind truth, map and root seed")
    if (
        preflight.get("stage") != "M28C_B0_INPUT_PREFLIGHT"
        or preflight.get("params", {}).get("root_seed") != seed
        or preflight.get("inputs", {}).get("m28_sources.trees")
        != simulation.get("sha256", {}).get("m28_sources.trees")
        or preflight.get("inputs", {}).get("m28_mosaic_events.private.tsv.gz")
        != simulation.get("sha256", {}).get("m28_mosaic_events.private.tsv.gz")
        or preflight.get("inputs", {}).get("m28_pools.private.tsv")
        != simulation.get("sha256", {}).get("m28_pools.private.tsv")
        or preflight.get("inputs", {}).get("m28b_v4_validation_B0.tsv.gz")
        != expected["b0_marker_table_sha256"]
    ):
        raise ValueError("B0 preflight does not bind simulation outputs, B0 and root seed")
    if (
        ingest.get("stage") != "M28C_B0_GNOMIX_INGEST_AUDIT"
        or ingest.get("decision") != "GO_B0_GNOMIX_TRAINING_PREREGISTRATION"
        or ingest.get("root_seed") != seed
        or ingest.get("merged_truth_table_accessed") is not False
        or ingest.get("output_sha256", {}).get("m28c_b0_target.vcf.gz")
        != expected["target_vcf_sha256"]
        or ingest.get("upstream_sha256", {}).get("m28c_b0_target.vcf.gz")
        != preflight.get("sha256", {}).get("m28c_b0_target.vcf.gz")
    ):
        raise ValueError("ingest report does not bind target VCF and root seed")
    if (
        inference.get("stage") != f"M28C_GNOMIX_FULL_B0_INFER_{replicate}"
        or inference.get("params", {}).get("replicate") != replicate
        or inference.get("params", {}).get("truth_accessed") is not False
        or inference.get("params", {}).get("target_truth_accuracy_computed") is not False
        or inference.get("inputs", {}).get("m28c_b0_target.vcf.gz")
        != expected["target_vcf_sha256"]
        or inference.get("sha256", {}).get("query_results.fb")
        != expected[f"replicate_{replicate}_fb_sha256"]
        or inference.get("sha256", {}).get("query_results.msp")
        != expected[f"replicate_{replicate}_msp_sha256"]
    ):
        raise ValueError("inference manifest does not bind predictions, target and replicate")


def validate_known_answer_receipt(
    contract: dict, contract_path: Path, receipt_path: Path
) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    implementation = contract["authenticated_implementation"]
    if (
        receipt.get("stage") != "M28D_B0_SCORER_KNOWN_ANSWERS"
        or receipt.get("decision") != "PASS_M28D_SCORER_KNOWN_ANSWERS"
        or receipt.get("real_truth_accessed") is not False
        or not receipt.get("checks")
        or not all(receipt["checks"].values())
        or receipt.get("unit_suite", {}).get("passed") is not True
        or receipt.get("unit_suite", {}).get("tests_run", 0) <= 0
        or receipt.get("scorer_sha256") != sha256_file(Path(__file__))
        or receipt.get("contract_sha256") != sha256_file(contract_path)
        or receipt.get("known_answer_runner_sha256")
        != implementation["known_answer_runner_sha256"]
        or receipt.get("unit_test_file_sha256") != implementation["unit_test_sha256"]
        or receipt.get("scorer_sha256") != implementation["scorer_sha256"]
    ):
        raise ValueError("known-answer receipt does not authenticate the frozen scorer")
    return receipt


def validate_m28c_comparison(path: Path) -> dict:
    comparison = json.loads(path.read_text(encoding="utf-8"))
    if (
        comparison.get("stage") != "M28C_GNOMIX_FULL_B0_RESOURCE_BENCHMARK_COMPARE"
        or comparison.get("decision") != "PASS_FULL_B0_TECHNICAL_BENCHMARK"
        or comparison.get("scope")
        != "full_B0_training_serialization_reload_and_inference_only_no_target_truth_accuracy_screen_or_effect_estimation"
        or comparison.get("truth_accessed") is not False
        or comparison.get("target_truth_accuracy_computed") is not False
        or not comparison.get("gates")
        or not all(comparison["gates"].values())
    ):
        raise ValueError("M28C comparison does not authorize descriptive B0 scoring")
    return comparison


def authenticate_pair_command(args: argparse.Namespace) -> None:
    """Authenticate and structurally parse both replicates without parsing truth."""
    contract = load_contract(args.contract)
    expected = contract["authenticated_inputs"]
    validate_known_answer_receipt(contract, args.contract, args.known_answer_receipt)
    observed_hashes = {
        "contract": sha256_file(args.contract),
        "known_answer_receipt": sha256_file(args.known_answer_receipt),
        "truth": require_hash(args.truth, expected["truth_sha256"], "truth"),
        "b0_markers": require_hash(
            args.b0_markers, expected["b0_marker_table_sha256"], "B0 markers"
        ),
        "genetic_map": require_hash(
            args.genetic_map, expected["genetic_map_sha256"], "genetic map"
        ),
        "m28c_comparison": require_hash(
            args.m28c_comparison, expected["m28c_comparison_sha256"], "M28C comparison"
        ),
        "simulation_manifest": require_hash(
            args.simulation_manifest,
            expected["simulation_manifest_sha256"],
            "simulation manifest",
        ),
        "b0_preflight_manifest": require_hash(
            args.b0_preflight_manifest,
            expected["b0_preflight_manifest_sha256"],
            "B0 preflight manifest",
        ),
        "ingest_report": require_hash(
            args.ingest_report, expected["ingest_report_sha256"], "ingest report"
        ),
        "scorer": sha256_file(Path(__file__)),
    }
    validate_m28c_comparison(args.m28c_comparison)
    fixed = contract["fixed_domain"]
    markers, marker_cms = load_b0_markers(args.b0_markers, fixed["chromosome_truth"])
    genetic_map = load_genetic_map(args.genetic_map, fixed["chromosome_truth"])
    coordinate_policy = contract["genetic_coordinate_validation"]
    structural: dict[str, dict[str, int]] = {}
    for replicate in ("A", "B"):
        suffix = replicate.lower()
        fb_path = getattr(args, f"fb_{suffix}")
        msp_path = getattr(args, f"msp_{suffix}")
        inference_path = getattr(args, f"inference_manifest_{suffix}")
        observed_hashes[f"fb_{replicate}"] = require_hash(
            fb_path, expected[f"replicate_{replicate}_fb_sha256"], f"FB {replicate}"
        )
        observed_hashes[f"msp_{replicate}"] = require_hash(
            msp_path, expected[f"replicate_{replicate}_msp_sha256"], f"MSP {replicate}"
        )
        observed_hashes[f"inference_manifest_{replicate}"] = require_hash(
            inference_path,
            expected[f"replicate_{replicate}_inference_manifest_sha256"],
            f"inference manifest {replicate}",
        )
        validate_seed_lineage(
            contract,
            replicate,
            args.simulation_manifest,
            args.b0_preflight_manifest,
            args.ingest_report,
            inference_path,
        )
        fb_positions, fb_cms, fb_samples, fb_probabilities = load_fb(fb_path)
        msp_metadata, msp_samples, msp_labels = load_msp(msp_path)
        validate_genetic_coordinates(
            markers,
            marker_cms,
            fb_positions,
            fb_cms,
            msp_metadata,
            genetic_map,
            float(coordinate_policy["b0_marker_tolerance_cm"]),
            float(coordinate_policy["msp_endpoint_tolerance_cm"]),
        )
        windows = build_prediction_windows(
            markers,
            fb_positions,
            fb_samples,
            fb_probabilities,
            msp_metadata,
            msp_samples,
            msp_labels,
        )
        validate_contract_dimensions(contract, markers, windows)
        structural[replicate] = {
            "markers": len(markers),
            "windows": len(windows),
            "samples": len(fb_samples),
            "terminal_window_markers": windows[-1].n_markers,
        }
    result = {
        "schema_version": 1,
        "stage": "M28D_B0_PAIR_AUTHENTICATION",
        "decision": "PASS_M28D_B0_PAIR_AUTHENTICATION",
        "real_truth_hashed": True,
        "real_truth_parsed": False,
        "both_replicates_authenticated_before_scoring": True,
        "seed": contract["seed_policy"]["seed"],
        "inputs": observed_hashes,
        "structural": structural,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_command(args: argparse.Namespace) -> None:
    contract = load_contract(args.contract)
    expected = contract["authenticated_inputs"]
    validate_known_answer_receipt(contract, args.contract, args.known_answer_receipt)
    pair_auth = json.loads(args.pair_auth_receipt.read_text(encoding="utf-8"))
    if (
        pair_auth.get("stage") != "M28D_B0_PAIR_AUTHENTICATION"
        or pair_auth.get("decision") != "PASS_M28D_B0_PAIR_AUTHENTICATION"
        or pair_auth.get("real_truth_parsed") is not False
        or pair_auth.get("both_replicates_authenticated_before_scoring") is not True
        or pair_auth.get("seed") != contract["seed_policy"]["seed"]
        or pair_auth.get("inputs", {}).get("contract") != sha256_file(args.contract)
        or pair_auth.get("inputs", {}).get("known_answer_receipt")
        != sha256_file(args.known_answer_receipt)
        or pair_auth.get("inputs", {}).get("scorer") != sha256_file(Path(__file__))
    ):
        raise ValueError("pair-authentication receipt is invalid")
    observed_hashes = {
        "contract": sha256_file(args.contract),
        "known_answer_receipt": sha256_file(args.known_answer_receipt),
        "pair_auth_receipt": sha256_file(args.pair_auth_receipt),
        "truth": require_hash(args.truth, expected["truth_sha256"], "truth"),
        "b0_markers": require_hash(args.b0_markers, expected["b0_marker_table_sha256"], "B0 markers"),
        "genetic_map": require_hash(args.genetic_map, expected["genetic_map_sha256"], "genetic map"),
        "fb": require_hash(args.fb, expected[f"replicate_{args.replicate}_fb_sha256"], "FB"),
        "msp": require_hash(args.msp, expected[f"replicate_{args.replicate}_msp_sha256"], "MSP"),
        "m28c_comparison": require_hash(
            args.m28c_comparison, expected["m28c_comparison_sha256"], "M28C comparison"
        ),
        "simulation_manifest": require_hash(
            args.simulation_manifest,
            expected["simulation_manifest_sha256"],
            "simulation manifest",
        ),
        "b0_preflight_manifest": require_hash(
            args.b0_preflight_manifest,
            expected["b0_preflight_manifest_sha256"],
            "B0 preflight manifest",
        ),
        "ingest_report": require_hash(
            args.ingest_report, expected["ingest_report_sha256"], "ingest report"
        ),
        "inference_manifest": require_hash(
            args.inference_manifest,
            expected[f"replicate_{args.replicate}_inference_manifest_sha256"],
            "inference manifest",
        ),
        "run_provenance": sha256_file(args.run_provenance),
        "scorer": sha256_file(Path(__file__)),
    }
    for key, observed in observed_hashes.items():
        if key in pair_auth.get("inputs", {}) and pair_auth["inputs"][key] != observed:
            raise ValueError(f"pair-authentication receipt changed for {key}")
    for key in ("fb", "msp", "inference_manifest"):
        if pair_auth.get("inputs", {}).get(f"{key}_{args.replicate}") != observed_hashes[key]:
            raise ValueError(f"pair-authentication receipt does not bind {key} {args.replicate}")
    validate_m28c_comparison(args.m28c_comparison)
    validate_seed_lineage(
        contract,
        args.replicate,
        args.simulation_manifest,
        args.b0_preflight_manifest,
        args.ingest_report,
        args.inference_manifest,
    )
    fixed = contract["fixed_domain"]
    markers, marker_cms = load_b0_markers(args.b0_markers, fixed["chromosome_truth"])
    genetic_map = load_genetic_map(args.genetic_map, fixed["chromosome_truth"])
    fb_positions, fb_cms, fb_samples, fb_probabilities = load_fb(args.fb)
    msp_metadata, msp_samples, msp_labels = load_msp(args.msp)
    coordinate_policy = contract["genetic_coordinate_validation"]
    validate_genetic_coordinates(
        markers,
        marker_cms,
        fb_positions,
        fb_cms,
        msp_metadata,
        genetic_map,
        float(coordinate_policy["b0_marker_tolerance_cm"]),
        float(coordinate_policy["msp_endpoint_tolerance_cm"]),
    )
    windows = build_prediction_windows(
        markers,
        fb_positions,
        fb_samples,
        fb_probabilities,
        msp_metadata,
        msp_samples,
        msp_labels,
    )
    validate_contract_dimensions(contract, markers, windows)
    truth = load_truth(
        args.truth,
        fb_samples,
        fixed["chromosome_truth"],
        markers[0],
        markers[-1] + 1,
    )
    tolerances = [
        contract["secondary_estimands"]["boundary_tolerances_cm"]["primary_descriptive"],
        *contract["secondary_estimands"]["boundary_tolerances_cm"]["sensitivities"],
    ]
    tolerances = sorted(set(float(value) for value in tolerances))
    metrics = score_objects(markers, windows, truth, genetic_map, tolerances)
    result = {
        "schema_version": 1,
        "stage": "M28D_B0_DESCRIPTIVE_SCORING",
        "decision": "B0_DESCRIBED_NO_CEILING_INFERENCE",
        "replicate": args.replicate,
        "scope": contract["scope"],
        "seed": contract["seed_policy"]["seed"],
        "scientific_inference_authorized": False,
        "BR_BS_authorized": False,
        "SESOI_fixed": False,
        "independent_seed_count_fixed": False,
        "validation_seed_consumed": True,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "stage": "M28D_B0_DESCRIPTIVE_SCORING_MANIFEST",
        "replicate": args.replicate,
        "inputs": observed_hashes,
        "output": {args.output.name: sha256_file(args.output)},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scientific_payload(document: dict) -> dict:
    return {
        "schema_version": document["schema_version"],
        "stage": document["stage"],
        "decision": document["decision"],
        "scope": document["scope"],
        "seed": document["seed"],
        "scientific_inference_authorized": document["scientific_inference_authorized"],
        "BR_BS_authorized": document["BR_BS_authorized"],
        "SESOI_fixed": document["SESOI_fixed"],
        "independent_seed_count_fixed": document["independent_seed_count_fixed"],
        "validation_seed_consumed": document["validation_seed_consumed"],
        "metrics": document["metrics"],
    }


def validate_score_document(document: dict, replicate: str) -> None:
    expected_scope = "descriptive_scoring_and_scorer_validation_only"
    if (
        document.get("schema_version") != 1
        or document.get("stage") != "M28D_B0_DESCRIPTIVE_SCORING"
        or document.get("decision") != "B0_DESCRIBED_NO_CEILING_INFERENCE"
        or document.get("replicate") != replicate
        or document.get("scope") != expected_scope
        or document.get("seed") != 20260818
        or document.get("scientific_inference_authorized") is not False
        or document.get("BR_BS_authorized") is not False
        or document.get("SESOI_fixed") is not False
        or document.get("independent_seed_count_fixed") is not False
        or document.get("validation_seed_consumed") is not True
        or not isinstance(document.get("metrics"), dict)
    ):
        raise ValueError(f"score document {replicate} violates frozen scope")


def compare_command(args: argparse.Namespace) -> None:
    document_a = json.loads(args.score_a.read_text(encoding="utf-8"))
    document_b = json.loads(args.score_b.read_text(encoding="utf-8"))
    validate_score_document(document_a, "A")
    validate_score_document(document_b, "B")
    payload_a = _scientific_payload(document_a)
    payload_b = _scientific_payload(document_b)
    if payload_a != payload_b:
        raise ValueError("A/B scientific score documents differ")
    result = {
        "schema_version": 1,
        "stage": "M28D_B0_DESCRIPTIVE_SCORING_COMPARE",
        "decision": "PASS_B0_DESCRIPTIVE_SCORER_REPRODUCIBILITY",
        "scientific_payload_exact": True,
        "scientific_inference_authorized": False,
        "BR_BS_authorized": False,
        "validation_seed_consumed": True,
        "score_A_sha256": sha256_file(args.score_a),
        "score_B_sha256": sha256_file(args.score_b),
        "metrics": payload_a["metrics"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "stage": "M28D_B0_DESCRIPTIVE_SCORING_COMPARE_MANIFEST",
        "inputs": {
            args.score_a.name: sha256_file(args.score_a),
            args.score_b.name: sha256_file(args.score_b),
            "scorer": sha256_file(Path(__file__)),
        },
        "output": {args.output.name: sha256_file(args.output)},
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authenticate = subparsers.add_parser(
        "authenticate-pair", help="authenticate both B0 replicas before truth parsing"
    )
    authenticate.add_argument("--contract", required=True, type=Path)
    authenticate.add_argument("--truth", required=True, type=Path)
    authenticate.add_argument("--b0-markers", required=True, type=Path)
    authenticate.add_argument("--genetic-map", required=True, type=Path)
    authenticate.add_argument("--fb-a", required=True, type=Path)
    authenticate.add_argument("--msp-a", required=True, type=Path)
    authenticate.add_argument("--fb-b", required=True, type=Path)
    authenticate.add_argument("--msp-b", required=True, type=Path)
    authenticate.add_argument("--m28c-comparison", required=True, type=Path)
    authenticate.add_argument("--simulation-manifest", required=True, type=Path)
    authenticate.add_argument("--b0-preflight-manifest", required=True, type=Path)
    authenticate.add_argument("--ingest-report", required=True, type=Path)
    authenticate.add_argument("--inference-manifest-a", required=True, type=Path)
    authenticate.add_argument("--inference-manifest-b", required=True, type=Path)
    authenticate.add_argument("--known-answer-receipt", required=True, type=Path)
    authenticate.add_argument("--output", required=True, type=Path)
    authenticate.set_defaults(func=authenticate_pair_command)
    score = subparsers.add_parser("score", help="score one frozen B0 replicate")
    score.add_argument("--contract", required=True, type=Path)
    score.add_argument("--truth", required=True, type=Path)
    score.add_argument("--b0-markers", required=True, type=Path)
    score.add_argument("--genetic-map", required=True, type=Path)
    score.add_argument("--fb", required=True, type=Path)
    score.add_argument("--msp", required=True, type=Path)
    score.add_argument("--m28c-comparison", required=True, type=Path)
    score.add_argument("--simulation-manifest", required=True, type=Path)
    score.add_argument("--b0-preflight-manifest", required=True, type=Path)
    score.add_argument("--ingest-report", required=True, type=Path)
    score.add_argument("--inference-manifest", required=True, type=Path)
    score.add_argument("--known-answer-receipt", required=True, type=Path)
    score.add_argument("--pair-auth-receipt", required=True, type=Path)
    score.add_argument("--run-provenance", required=True, type=Path)
    score.add_argument("--replicate", required=True, choices=("A", "B"))
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--manifest", required=True, type=Path)
    score.set_defaults(func=score_command)
    compare = subparsers.add_parser("compare", help="compare A/B scientific score payloads")
    compare.add_argument("--score-a", required=True, type=Path)
    compare.add_argument("--score-b", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--manifest", required=True, type=Path)
    compare.set_defaults(func=compare_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
