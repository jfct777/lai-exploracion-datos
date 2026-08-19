#!/usr/bin/env python3
"""Frozen paired scorer for the M30 Gnomix-versus-FLARE development comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import m28d_b0_scorer as base


EXPERIMENT_ID = "M30_FLARE_BASELINE"
ANCESTRIES = base.ANCESTRIES
DIPLOID_CLASSES = base.DIPLOID_CLASSES
METHODS = ("Gnomix", "FLARE")
ROOTS = ("root17", "root18")


class ScoringError(RuntimeError):
    """Raised when a frozen alignment, scoring, or provenance invariant fails."""


@dataclass(frozen=True)
class TargetGrid:
    loci: tuple[tuple[str, int, str, str], ...]
    samples: tuple[str, ...]

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(locus[1] for locus in self.loci)


@dataclass(frozen=True)
class PredictionGrid:
    loci: tuple[tuple[str, int, str, str], ...]
    samples: tuple[str, ...]
    probabilities: tuple[dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]], ...]
    hard_labels: tuple[dict[str, tuple[str, str]], ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoringError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def load_contract(path: Path, check_implementation: bool = True) -> dict:
    contract = read_json(path)
    require(contract.get("experiment_id") == EXPERIMENT_ID, "Unexpected M30 contract")
    scoring = contract.get("scoring_contract", {})
    require(scoring.get("status") == "FROZEN_BEFORE_M30_INFERENCE", "Scorer was not frozen before inference")
    require(scoring.get("bootstrap", {}).get("replicates") == 10000, "Bootstrap replicate drift")
    require(scoring.get("bootstrap", {}).get("seed") == 3001702, "Bootstrap seed drift")
    require(scoring.get("metrics", {}).get("boundaries", {}).get("primary_tolerance_cm") == 0.2,
            "Primary boundary tolerance drift")
    require(scoring.get("metrics", {}).get("boundaries", {}).get("sensitivity_tolerances_cm") == [0.1, 0.5],
            "Boundary sensitivity drift")
    if check_implementation:
        require(scoring.get("scorer_sha256") == sha256(Path(__file__)), "Frozen scorer SHA-256 mismatch")
    return contract


def load_target_grid(path: Path) -> TargetGrid:
    samples = None
    loci: list[tuple[str, int, str, str]] = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                samples = fields[9:]
                require(samples and len(samples) == len(set(samples)), "Target VCF sample IDs are missing or duplicated")
                continue
            if line.startswith("#"):
                continue
            require(samples is not None, "Target VCF data precede the header")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"Malformed target VCF row {line_number}")
            chrom, pos_text, _identifier, ref, alt = fields[:5]
            require("," not in alt and alt != ".", f"Non-biallelic target locus at row {line_number}")
            locus = (base.normalize_chrom(chrom), int(pos_text), ref, alt)
            require(not loci or locus[1] > loci[-1][1], "Target loci are not strictly increasing")
            loci.append(locus)
    require(samples is not None and loci, "Target VCF is empty")
    return TargetGrid(tuple(loci), tuple(samples))


def load_flare_grid(path: Path) -> PredictionGrid:
    ancestry_codes: dict[str, str] = {}
    samples = None
    loci: list[tuple[str, int, str, str]] = []
    probabilities = []
    hard_labels = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##ANCESTRY=<"):
                payload = line.strip()[len("##ANCESTRY=<"):-1]
                for token in payload.split(","):
                    ancestry, code = token.split("=", 1)
                    require(ancestry in ANCESTRIES and code not in ancestry_codes,
                            f"Invalid FLARE ancestry header at row {line_number}")
                    ancestry_codes[code] = ancestry
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                require(samples and len(samples) == len(set(samples)), "FLARE sample IDs are missing or duplicated")
                continue
            if line.startswith("#"):
                continue
            require(samples is not None, "FLARE data precede the header")
            require(set(ancestry_codes.values()) == set(ANCESTRIES), "FLARE ancestry header is incomplete")
            require(ancestry_codes == {str(index): ancestry for index, ancestry in enumerate(ANCESTRIES)},
                    "FLARE ancestry codes do not match AFR=0, EUR=1, ASIA=2")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"Malformed FLARE VCF row {line_number}")
            chrom, pos_text, _identifier, ref, alt = fields[:5]
            locus = (base.normalize_chrom(chrom), int(pos_text), ref, alt)
            require(not loci or locus[1] > loci[-1][1], "FLARE loci are not strictly increasing")
            fmt = fields[8].split(":")
            required = ("AN1", "AN2", "ANP1", "ANP2")
            require(all(name in fmt for name in required), f"FLARE FORMAT is incomplete at row {line_number}")
            indices = {name: fmt.index(name) for name in required}
            row_probabilities = {}
            row_hard = {}
            for sample, sample_field in zip(samples, fields[9:]):
                values = sample_field.split(":")
                labels = tuple(ancestry_codes.get(values[indices[name]], "") for name in ("AN1", "AN2"))
                require(all(label in ANCESTRIES for label in labels), f"Invalid FLARE hard ancestry at row {line_number}")
                pair = []
                for name in ("ANP1", "ANP2"):
                    raw = [float(value) for value in values[indices[name]].split(",")]
                    require(len(raw) == len(ANCESTRIES), f"Invalid FLARE probability length at row {line_number}")
                    require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in raw),
                            f"Invalid FLARE probability at row {line_number}")
                    total = sum(raw)
                    require(0.99 - 1e-12 <= total <= 1.01 + 1e-12,
                            f"FLARE rounded probabilities fall outside [0.99,1.01] at row {line_number}")
                    require(total > 0.0, f"FLARE probability vector is zero at row {line_number}")
                    pair.append(tuple(value / total for value in raw))
                for hap, label in enumerate(labels):
                    maximum = max(pair[hap])
                    require(abs(pair[hap][base.ANCESTRY_INDEX[label]] - maximum) <= 1e-12,
                            f"FLARE hard call is not among tied probability maxima at row {line_number}")
                row_probabilities[sample] = (pair[0], pair[1])
                row_hard[sample] = (labels[0], labels[1])
            loci.append(locus)
            probabilities.append(row_probabilities)
            hard_labels.append(row_hard)
    require(samples is not None and loci, "FLARE VCF is empty")
    return PredictionGrid(tuple(loci), tuple(samples), tuple(probabilities), tuple(hard_labels))


def load_gnomix_grid(fb_path: Path, msp_path: Path, target: TargetGrid) -> PredictionGrid:
    fb_positions, _fb_cms, fb_samples, fb_probabilities = base.load_fb(fb_path)
    msp_metadata, msp_samples, msp_labels = base.load_msp(msp_path)
    windows = base.build_prediction_windows(
        target.positions,
        fb_positions,
        fb_samples,
        fb_probabilities,
        msp_metadata,
        msp_samples,
        msp_labels,
    )
    probabilities = []
    hard_labels = []
    for window in windows:
        for _ in range(window.marker_start, window.marker_end):
            probabilities.append(window.probabilities)
            hard_labels.append(window.hard_labels)
    require(len(probabilities) == len(target.loci), "Gnomix windows do not consume the target grid")
    return PredictionGrid(target.loci, tuple(fb_samples), tuple(probabilities), tuple(hard_labels))


def validate_alignment(target: TargetGrid, gnomix: PredictionGrid, flare: PredictionGrid,
                       expected_markers: int | None = None) -> None:
    if expected_markers is not None:
        require(len(target.loci) == expected_markers, "Target marker count differs from contract")
    require(gnomix.loci == target.loci, "Gnomix locus grid differs from target CHROM/POS/REF/ALT")
    require(flare.loci == target.loci, "FLARE locus grid differs from target CHROM/POS/REF/ALT")
    require(gnomix.samples == target.samples, "Gnomix sample order differs from target")
    require(flare.samples == target.samples, "FLARE sample order differs from target")
    for method, grid in (("Gnomix", gnomix), ("FLARE", flare)):
        require(len(grid.probabilities) == len(target.loci), f"{method} probability row count differs")
        require(len(grid.hard_labels) == len(target.loci), f"{method} hard-call row count differs")


def _one_hot(ancestry: str) -> tuple[float, float, float]:
    return tuple(float(item == ancestry) for item in ANCESTRIES)


def _brier(pair, truth_pair: tuple[str, str]) -> float:
    total = 0.0
    for truth_hap in (0, 1):
        target = _one_hot(truth_pair[truth_hap])
        total += sum((pair[truth_hap][index] - target[index]) ** 2 for index in range(3))
    return total / 4.0


def _prediction_boundaries(grid: PredictionGrid, cells: Sequence[tuple[int, int]], sample: str,
                           hap: int, genetic_map: base.GeneticMap) -> list[base.Boundary]:
    boundaries = []
    previous = grid.hard_labels[0][sample][hap]
    for index in range(1, len(cells)):
        current = grid.hard_labels[index][sample][hap]
        if current != previous:
            boundaries.append(base.Boundary(genetic_map.cm_at(cells[index][0]), previous, current))
        previous = current
    return boundaries


def _boundary_metrics(truth_pair, prediction_pair, tolerance: float, cm_span: float) -> dict:
    distances = []
    matches = 0
    for truth_hap, prediction_hap in zip(truth_pair, prediction_pair):
        pairs = base.ordered_boundary_pairs(truth_hap, prediction_hap, tolerance)
        matches += len(pairs)
        distances.extend(distance for _truth_index, _prediction_index, distance in pairs)
    truth_count = sum(len(items) for items in truth_pair)
    prediction_count = sum(len(items) for items in prediction_pair)
    precision = matches / prediction_count if prediction_count else None
    recall = matches / truth_count if truth_count else None
    if truth_count == 0 and prediction_count == 0:
        f1 = 1.0
    elif matches == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "truth": truth_count,
        "predicted": prediction_count,
        "matched": matches,
        "missed": truth_count - matches,
        "extra": prediction_count - matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_transitions_per_cm": (prediction_count - matches) / (2.0 * cm_span),
        "distances": distances,
    }


def score_root(target: TargetGrid, grids: dict[str, PredictionGrid], truth, genetic_map: base.GeneticMap,
               tolerances: Sequence[float]) -> dict:
    validate_alignment(target, grids["Gnomix"], grids["FLARE"])
    cells = base.discrete_voronoi(target.positions)
    domain_start, domain_end = cells[0][0], cells[-1][1]
    cm_span = genetic_map.cm_at(domain_end) - genetic_map.cm_at(domain_start)
    require(cm_span > 0.0, "Scoring domain has nonpositive genetic length")
    per_sample = {}
    for sample in target.samples:
        truth_pair = truth[sample]
        truth_indices = [0, 0]
        method_state = {
            method: {
                "mae_num": [0.0, 0.0, 0.0],
                "present_num": [0.0, 0.0, 0.0],
                "present_den": [0.0, 0.0, 0.0],
                "brier_direct": 0.0,
                "confusion": Counter(),
            }
            for method in METHODS
        }
        total_cm = 0.0
        for marker_index, (left, right) in enumerate(cells):
            cursor = left
            while cursor < right:
                for hap in (0, 1):
                    while truth_pair[hap][truth_indices[hap]].end <= cursor:
                        truth_indices[hap] += 1
                end = min(right, truth_pair[0][truth_indices[0]].end, truth_pair[1][truth_indices[1]].end)
                require(end > cursor, f"Truth partition stalled for {sample} at {cursor}")
                weight = genetic_map.cm_at(end) - genetic_map.cm_at(cursor)
                require(weight >= -1e-12, "Negative interpolated cM weight")
                weight = max(0.0, weight)
                truth_labels = (
                    truth_pair[0][truth_indices[0]].ancestry,
                    truth_pair[1][truth_indices[1]].ancestry,
                )
                truth_dose = [sum(label == ancestry for label in truth_labels) for ancestry in ANCESTRIES]
                truth_state = base._diploid_class(truth_labels)
                for method, grid in grids.items():
                    state = method_state[method]
                    predicted_probabilities = grid.probabilities[marker_index][sample]
                    predicted_dose = [
                        predicted_probabilities[0][index] + predicted_probabilities[1][index]
                        for index in range(3)
                    ]
                    for index in range(3):
                        error = abs(predicted_dose[index] - truth_dose[index]) / 2.0
                        state["mae_num"][index] += weight * error
                        if truth_dose[index] > 0:
                            state["present_num"][index] += weight * error
                            state["present_den"][index] += weight
                    state["brier_direct"] += weight * _brier(predicted_probabilities, truth_labels)
                    predicted_state = base._diploid_class(grid.hard_labels[marker_index][sample])
                    state["confusion"][(truth_state, predicted_state)] += weight
                total_cm += weight
                cursor = end
        require(abs(total_cm - cm_span) <= max(1e-9, cm_span * 1e-9), "Integrated cM weight differs from map span")
        truth_boundaries = (
            base._truth_boundaries(truth_pair[0], genetic_map),
            base._truth_boundaries(truth_pair[1], genetic_map),
        )
        sample_payload = {}
        for method, grid in grids.items():
            state = method_state[method]
            direct = state["brier_direct"] / cm_span
            phase = "lineage_bound_direct"
            predicted_boundaries = (
                _prediction_boundaries(grid, cells, sample, 0, genetic_map),
                _prediction_boundaries(grid, cells, sample, 1, genetic_map),
            )
            boundary = {
                f"{tolerance:.1f}": _boundary_metrics(truth_boundaries, predicted_boundaries, tolerance, cm_span)
                for tolerance in tolerances
            }
            sample_payload[method] = {
                "mae_total": {
                    ancestry: state["mae_num"][index] / cm_span
                    for index, ancestry in enumerate(ANCESTRIES)
                },
                "mae_truth_present": {
                    ancestry: (
                        state["present_num"][index] / state["present_den"][index]
                        if state["present_den"][index] > 0 else None
                    )
                    for index, ancestry in enumerate(ANCESTRIES)
                },
                "brier": direct,
                "phase_permutation": phase,
                "confusion": {f"{truth_state}|{predicted_state}": value / cm_span
                              for (truth_state, predicted_state), value in state["confusion"].items()},
                "boundaries": boundary,
                "cm_span": cm_span,
            }
        per_sample[sample] = sample_payload
    return {"samples": per_sample, "summary": aggregate_root(per_sample)}


def _mean(values: Sequence[float]) -> float:
    require(bool(values), "Cannot average an empty metric")
    return sum(values) / len(values)


def _macro_f1(confusion: Counter[tuple[str, str]]) -> dict:
    per_class = {}
    supported_scores = []
    fixed_scores = []
    truth_supported = []
    for label in DIPLOID_CLASSES:
        tp = confusion[(label, label)]
        fp = sum(confusion[(truth, label)] for truth in DIPLOID_CLASSES if truth != label)
        fn = sum(confusion[(label, predicted)] for predicted in DIPLOID_CLASSES if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        fixed_scores.append(f1)
        truth_support = sum(confusion[(label, predicted)] for predicted in DIPLOID_CLASSES)
        if truth_support > 0:
            truth_supported.append(label)
            supported_scores.append(f1)
    return {
        "macro_f1_truth_supported": _mean(supported_scores),
        "macro_f1_fixed_six": _mean(fixed_scores) if len(truth_supported) == len(DIPLOID_CLASSES) else None,
        "truth_supported_classes": truth_supported,
        "all_six_truth_classes_present": len(truth_supported) == len(DIPLOID_CLASSES),
        "per_class": per_class,
    }


def aggregate_root(per_sample: dict, selected: Sequence[str] | None = None) -> dict:
    sample_ids = list(selected) if selected is not None else sorted(per_sample)
    require(bool(sample_ids), "Root aggregation has no individuals")
    output = {}
    for method in METHODS:
        total = {
            ancestry: _mean([per_sample[sample][method]["mae_total"][ancestry] for sample in sample_ids])
            for ancestry in ANCESTRIES
        }
        present = {}
        for ancestry in ANCESTRIES:
            values = [per_sample[sample][method]["mae_truth_present"][ancestry] for sample in sample_ids]
            supported = [value for value in values if value is not None]
            present[ancestry] = _mean(supported) if supported else None
        confusion = Counter()
        for sample in sample_ids:
            for key, value in per_sample[sample][method]["confusion"].items():
                truth_state, predicted_state = key.split("|", 1)
                confusion[(truth_state, predicted_state)] += value
        tolerance_keys = sorted(per_sample[sample_ids[0]][method]["boundaries"], key=float)
        boundaries = {}
        for tolerance in tolerance_keys:
            records = [per_sample[sample][method]["boundaries"][tolerance] for sample in sample_ids]
            truth_count = sum(record["truth"] for record in records)
            predicted_count = sum(record["predicted"] for record in records)
            matched = sum(record["matched"] for record in records)
            distances = [distance for record in records for distance in record["distances"]]
            precision = matched / predicted_count if predicted_count else None
            recall = matched / truth_count if truth_count else None
            if truth_count == 0 and predicted_count == 0:
                f1 = 1.0
            elif matched == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)
            denominator_cm = sum(2.0 * per_sample[sample][method]["cm_span"] for sample in sample_ids)
            boundaries[tolerance] = {
                "truth": truth_count,
                "predicted": predicted_count,
                "matched": matched,
                "missed": truth_count - matched,
                "extra": predicted_count - matched,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_transitions_per_cm": (predicted_count - matched) / denominator_cm,
                "matched_distance_median_cm": base._percentile(distances, 0.5),
                "matched_distance_p90_cm": base._percentile(distances, 0.9),
            }
        output[method] = {
            "mae_total": total,
            "mae_truth_present": present,
            "primary_macro_mae": _mean(list(total.values())),
            "brier": _mean([per_sample[sample][method]["brier"] for sample in sample_ids]),
            "hard_diploid_state": _macro_f1(confusion),
            "boundaries": boundaries,
            "individuals": len(sample_ids),
        }
    return output


def method_deltas(summary: dict) -> dict:
    gnomix, flare = summary["Gnomix"], summary["FLARE"]
    for ancestry in ANCESTRIES:
        require(gnomix["mae_truth_present"][ancestry] is not None,
                f"Gnomix truth-present MAE is undefined for {ancestry}")
        require(flare["mae_truth_present"][ancestry] is not None,
                f"FLARE truth-present MAE is undefined for {ancestry}")
    return {
        "primary_macro_mae": flare["primary_macro_mae"] - gnomix["primary_macro_mae"],
        "mae_total": {
            ancestry: flare["mae_total"][ancestry] - gnomix["mae_total"][ancestry]
            for ancestry in ANCESTRIES
        },
        "mae_truth_present": {
            ancestry: flare["mae_truth_present"][ancestry] - gnomix["mae_truth_present"][ancestry]
            for ancestry in ANCESTRIES
        },
        "brier": flare["brier"] - gnomix["brier"],
        "macro_f1_truth_supported": (
            flare["hard_diploid_state"]["macro_f1_truth_supported"]
            - gnomix["hard_diploid_state"]["macro_f1_truth_supported"]
        ),
        "boundaries": {
            tolerance: {
                "f1": flare["boundaries"][tolerance]["f1"] - gnomix["boundaries"][tolerance]["f1"],
                "false_transitions_per_cm": (
                    flare["boundaries"][tolerance]["false_transitions_per_cm"]
                    - gnomix["boundaries"][tolerance]["false_transitions_per_cm"]
                ),
            }
            for tolerance in gnomix["boundaries"]
        },
    }


def _average_root_deltas(root_deltas: dict[str, dict]) -> dict:
    def avg(values):
        return sum(values) / len(values)

    return {
        "primary_macro_mae": avg([root_deltas[root]["primary_macro_mae"] for root in ROOTS]),
        "mae_total": {
            ancestry: avg([root_deltas[root]["mae_total"][ancestry] for root in ROOTS])
            for ancestry in ANCESTRIES
        },
        "mae_truth_present": {
            ancestry: avg([root_deltas[root]["mae_truth_present"][ancestry] for root in ROOTS])
            for ancestry in ANCESTRIES
        },
        "brier": avg([root_deltas[root]["brier"] for root in ROOTS]),
        "macro_f1_truth_supported": avg([root_deltas[root]["macro_f1_truth_supported"] for root in ROOTS]),
        "boundaries": {
            tolerance: {
                metric: avg([root_deltas[root]["boundaries"][tolerance][metric] for root in ROOTS])
                for metric in ("f1", "false_transitions_per_cm")
            }
            for tolerance in root_deltas[ROOTS[0]]["boundaries"]
        },
    }


def _flatten_deltas(delta: dict) -> dict[str, float]:
    flat = {
        "primary_macro_mae": delta["primary_macro_mae"],
        "brier": delta["brier"],
        "macro_f1_truth_supported": delta["macro_f1_truth_supported"],
    }
    for ancestry in ANCESTRIES:
        flat[f"mae_total.{ancestry}"] = delta["mae_total"][ancestry]
        flat[f"mae_truth_present.{ancestry}"] = delta["mae_truth_present"][ancestry]
    for tolerance, metrics in delta["boundaries"].items():
        for metric, value in metrics.items():
            flat[f"boundaries.{tolerance}.{metric}"] = value
    return flat


def paired_bootstrap(roots: dict[str, dict], replicates: int, seed: int) -> dict:
    require(set(roots) == set(ROOTS), "Bootstrap requires root17 and root18")
    require(replicates > 0, "Bootstrap replicate count must be positive")
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {}
    for _replicate in range(replicates):
        root_deltas = {}
        for root in ROOTS:
            sample_ids = sorted(roots[root])
            selected = [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
            root_deltas[root] = method_deltas(aggregate_root(roots[root], selected))
        flat = _flatten_deltas(_average_root_deltas(root_deltas))
        for key, value in flat.items():
            require(math.isfinite(value), f"Nonfinite bootstrap statistic: {key}")
            distributions.setdefault(key, []).append(value)
    intervals = {}
    for key, values in sorted(distributions.items()):
        intervals[key] = {
            "lower": base._percentile(values, 0.025),
            "upper": base._percentile(values, 0.975),
        }
    return {
        "scheme": "paired_individual_within_root_equal_root_weight",
        "replicates": replicates,
        "seed": seed,
        "interval": "percentile_95",
        "metrics": intervals,
    }


def decide(root_deltas: dict[str, dict], pooled: dict, bootstrap: dict) -> dict:
    primary_roots = all(root_deltas[root]["primary_macro_mae"] < 0.0 for root in ROOTS)
    primary_interval = bootstrap["metrics"]["primary_macro_mae"]["upper"] < 0.0
    primary_pass = primary_roots and primary_interval
    ancestry_checks = {
        f"{scope}.{ancestry}": pooled[scope][ancestry] <= 0.0
        for scope in ("mae_total", "mae_truth_present")
        for ancestry in ANCESTRIES
    }
    boundary_f1 = pooled["boundaries"]["0.2"]["f1"] >= 0.0
    false_transitions = pooled["boundaries"]["0.2"]["false_transitions_per_cm"] <= 0.0
    safeguards_pass = all(ancestry_checks.values()) and boundary_f1 and false_transitions
    keep_direction = all(root_deltas[root]["primary_macro_mae"] >= 0.0 for root in ROOTS)
    keep_interval = bootstrap["metrics"]["primary_macro_mae"]["lower"] > 0.0
    if primary_pass and safeguards_pass:
        label = "GO_FLARE_NEXT_DEV"
    elif keep_direction or keep_interval:
        label = "KEEP_GNOMIX"
    else:
        label = "INCONCLUSIVE_TRADEOFF"
    return {
        "label": label,
        "primary": {
            "root17_delta_below_zero": root_deltas["root17"]["primary_macro_mae"] < 0.0,
            "root18_delta_below_zero": root_deltas["root18"]["primary_macro_mae"] < 0.0,
            "pooled_ci_upper_below_zero": primary_interval,
        },
        "ancestry_delta_at_or_below_zero": ancestry_checks,
        "boundary_f1_0.2_delta_at_or_above_zero": boundary_f1,
        "false_transitions_0.2_delta_at_or_below_zero": false_transitions,
        "scope": "DEV_ONLY_NOT_INDEPENDENT_VALIDATION",
    }


def validate_known_answer_receipt(path: Path, contract_path: Path) -> dict:
    receipt = read_json(path)
    require(receipt.get("stage") == "M30_SCORER_KNOWN_ANSWERS", "Unexpected known-answer stage")
    require(receipt.get("status") == "PASS", "M30 known-answer tests did not pass")
    require(receipt.get("real_truth_accessed") is False, "Known-answer process accessed real truth")
    require(receipt.get("scorer_sha256") == sha256(Path(__file__)), "Known-answer scorer hash mismatch")
    require(receipt.get("contract_sha256") == sha256(contract_path), "Known-answer contract hash mismatch")
    expected_base = read_json(contract_path)["scoring_contract"]["base_metric_library"]["sha256"]
    require(receipt.get("base_scorer_sha256") == expected_base, "Known-answer base scorer hash mismatch")
    require(receipt.get("checks") and all(receipt["checks"].values()), "Known-answer receipt has a failed check")
    return receipt


def authenticate_root(contract: dict, root_label: str, paths: dict[str, Path]) -> None:
    expected = contract["roots"][root_label]["inputs"]
    truth_expected = contract["roots"][root_label]["scoring_inputs"]["truth"]
    require(sha256(paths["truth"]) == truth_expected["sha256"], f"{root_label} truth SHA-256 mismatch")
    for key in ("target_vcf", "gnomix_binding", "gnomix_fb", "gnomix_msp"):
        require(sha256(paths[key]) == expected[key]["sha256"], f"{root_label} {key} SHA-256 mismatch")
    binding = read_json(paths["gnomix_binding"])
    require(binding.get("stage") == "M29_AUTHENTICATED_B0_BINDING", f"{root_label} Gnomix binding stage mismatch")
    require(binding.get("root_seed") == contract["roots"][root_label]["root_seed"],
            f"{root_label} Gnomix binding seed mismatch")
    require(binding.get("sha256") == {
        "fb": expected["gnomix_fb"]["sha256"],
        "msp": expected["gnomix_msp"]["sha256"],
    }, f"{root_label} Gnomix binding hash mismatch")
    runtime = read_json(paths["runtime_contract"])
    preflight = read_json(paths["preflight_report"])
    flare_audit = read_json(paths["flare_audit"])
    require(runtime.get("experiment_id") == EXPERIMENT_ID and runtime.get("root_label") == root_label,
            f"{root_label} runtime contract mismatch")
    require(runtime.get("status") == "PREFLIGHT_PASS" and runtime.get("truth_accessed") is False,
            f"{root_label} runtime contract did not preserve inference blindness")
    require(runtime.get("inputs_sha256", {}).get("target_vcf") == expected["target_vcf"]["sha256"],
            f"{root_label} runtime contract target mismatch")
    require(preflight.get("runtime_contract_sha256") == sha256(paths["runtime_contract"]),
            f"{root_label} preflight does not authenticate runtime contract")
    require(flare_audit.get("status") == "PASS" and flare_audit.get("root_label") == root_label,
            f"{root_label} FLARE audit did not pass")
    require(flare_audit.get("truth_accessed") is False, f"{root_label} FLARE inference accessed truth")
    require(flare_audit.get("runtime_contract_sha256") == sha256(paths["runtime_contract"]),
            f"{root_label} FLARE audit runtime mismatch")
    require(flare_audit.get("output_sha256", {}).get("ancestry_vcf") == sha256(paths["flare_vcf"]),
            f"{root_label} FLARE VCF is not authenticated")


def score_compare_command(args: argparse.Namespace) -> None:
    contract = load_contract(args.contract)
    scoring = contract["scoring_contract"]
    require(sha256(args.base_scorer) == scoring["base_metric_library"]["sha256"],
            "Base metric library SHA-256 mismatch")
    require(sha256(args.genetic_map) == scoring["alignment"]["genetic_map_sha256"],
            "Genetic map SHA-256 mismatch")
    validate_known_answer_receipt(args.known_answer_receipt, args.contract)
    provenance = read_json(args.run_provenance)
    require(provenance.get("experiment_id") == EXPERIMENT_ID, "Unexpected M30 run provenance")
    root_inputs = {
        root: {
            key: getattr(args, f"{root}_{key}")
            for key in (
                "truth", "target_vcf", "gnomix_binding", "gnomix_fb", "gnomix_msp",
                "flare_vcf", "flare_audit", "runtime_contract", "preflight_report",
            )
        }
        for root in ROOTS
    }
    genetic_map = base.load_genetic_map(args.genetic_map, "22")
    scored_roots = {}
    for root in ROOTS:
        paths = root_inputs[root]
        authenticate_root(contract, root, paths)
        target = load_target_grid(paths["target_vcf"])
        require(len(target.loci) == scoring["alignment"]["marker_count"], f"{root} marker count mismatch")
        gnomix = load_gnomix_grid(paths["gnomix_fb"], paths["gnomix_msp"], target)
        flare = load_flare_grid(paths["flare_vcf"])
        validate_alignment(target, gnomix, flare, scoring["alignment"]["marker_count"])
        truth = base.load_truth(paths["truth"], target.samples, "22", target.positions[0], target.positions[-1] + 1)
        tolerances = sorted({
            scoring["metrics"]["boundaries"]["primary_tolerance_cm"],
            *scoring["metrics"]["boundaries"]["sensitivity_tolerances_cm"],
        })
        scored_roots[root] = score_root(target, {"Gnomix": gnomix, "FLARE": flare}, truth, genetic_map, tolerances)
    root_summaries = {root: scored_roots[root]["summary"] for root in ROOTS}
    root_deltas = {root: method_deltas(root_summaries[root]) for root in ROOTS}
    pooled = _average_root_deltas(root_deltas)
    bootstrap = paired_bootstrap(
        {root: scored_roots[root]["samples"] for root in ROOTS},
        scoring["bootstrap"]["replicates"],
        scoring["bootstrap"]["seed"],
    )
    decision = decide(root_deltas, pooled, bootstrap)
    result = {
        "schema_version": "1.0.0",
        "experiment_id": EXPERIMENT_ID,
        "stage": "M30_FLARE_VS_GNOMIX_DEV_SCORING",
        "decision": decision,
        "roots": root_summaries,
        "root_deltas_flare_minus_gnomix": root_deltas,
        "pooled_equal_root_delta_flare_minus_gnomix": pooled,
        "paired_bootstrap": bootstrap,
        "metric_direction": {
            "mae_brier_false_transitions_distance": "lower_is_better",
            "macro_f1_boundary_f1": "higher_is_better",
        },
        "truth_access_scope": "SCORER_ONLY_AFTER_BOTH_FROZEN_INFERENCES",
        "claim_scope": scoring["claim_scope"],
    }
    write_json(args.output, result)
    manifest = {
        "schema_version": "1.0.0",
        "stage": "M30_FLARE_VS_GNOMIX_DEV_SCORING_MANIFEST",
        "inputs_sha256": {
            "contract": sha256(args.contract),
            "base_scorer": sha256(args.base_scorer),
            "genetic_map": sha256(args.genetic_map),
            "known_answer_receipt": sha256(args.known_answer_receipt),
            "run_provenance": sha256(args.run_provenance),
            **{
                f"{root}.{key}": sha256(path)
                for root, paths in root_inputs.items()
                for key, path in paths.items()
            },
        },
        "scorer_sha256": sha256(Path(__file__)),
        "output_sha256": sha256(args.output),
    }
    write_json(args.manifest, manifest)


def _synthetic_grid(target: TargetGrid, rows: list[dict[str, tuple[str, str]]]) -> PredictionGrid:
    probabilities = []
    for row in rows:
        probabilities.append({sample: (_one_hot(labels[0]), _one_hot(labels[1])) for sample, labels in row.items()})
    return PredictionGrid(target.loci, target.samples, tuple(probabilities), tuple(rows))


def run_known_answers() -> dict[str, bool]:
    target = TargetGrid(
        (("22", 10, "A", "G"), ("22", 20, "C", "T"), ("22", 30, "G", "A")),
        ("T000", "T001"),
    )
    truth = {
        "T000": (
            [base.TruthSegment(10, 16, "AFR"), base.TruthSegment(16, 31, "EUR")],
            [base.TruthSegment(10, 31, "ASIA")],
        ),
        "T001": (
            [base.TruthSegment(10, 16, "EUR"), base.TruthSegment(16, 31, "ASIA")],
            [base.TruthSegment(10, 31, "AFR")],
        ),
    }
    perfect_rows = [
        {"T000": ("AFR", "ASIA"), "T001": ("EUR", "AFR")},
        {"T000": ("EUR", "ASIA"), "T001": ("ASIA", "AFR")},
        {"T000": ("EUR", "ASIA"), "T001": ("ASIA", "AFR")},
    ]
    inferior_rows = [
        {"T000": ("AFR", "ASIA"), "T001": ("EUR", "AFR")},
        {"T000": ("AFR", "ASIA"), "T001": ("EUR", "AFR")},
        {"T000": ("AFR", "ASIA"), "T001": ("EUR", "AFR")},
    ]
    flare = _synthetic_grid(target, perfect_rows)
    gnomix = _synthetic_grid(target, inferior_rows)
    genetic_map = base.GeneticMap([base.MapPoint(0, 0.0), base.MapPoint(100, 1.0)])
    score = score_root(target, {"Gnomix": gnomix, "FLARE": flare}, truth, genetic_map, [0.1, 0.2, 0.5])
    per_root = {root: score["samples"] for root in ROOTS}
    root_deltas = {root: method_deltas(score["summary"]) for root in ROOTS}
    pooled = _average_root_deltas(root_deltas)
    boot_a = paired_bootstrap(per_root, 100, 177)
    boot_b = paired_bootstrap(per_root, 100, 177)
    decision = decide(root_deltas, pooled, boot_a)

    swapped_rows = [
        {sample: (labels[1], labels[0]) for sample, labels in row.items()}
        for row in perfect_rows
    ]
    swapped = _synthetic_grid(target, swapped_rows)
    swapped_score = score_root(
        target, {"Gnomix": swapped, "FLARE": swapped}, truth, genetic_map, [0.2]
    )["summary"]
    one_to_one = base.ordered_boundary_pairs(
        [base.Boundary(0.2, "AFR", "EUR")],
        [base.Boundary(0.19, "AFR", "EUR"), base.Boundary(0.21, "AFR", "EUR")],
        0.2,
    )
    keep_roots = json.loads(json.dumps(root_deltas))
    for root in ROOTS:
        keep_roots[root]["primary_macro_mae"] = 0.01
    trade_pooled = json.loads(json.dumps(pooled))
    trade_pooled["boundaries"]["0.2"]["false_transitions_per_cm"] = 0.01
    return {
        "perfect_flare_zero_primary_mae": score["summary"]["FLARE"]["primary_macro_mae"] == 0.0,
        "inferior_gnomix_positive_primary_mae": score["summary"]["Gnomix"]["primary_macro_mae"] > 0.0,
        "same_grid_exact_alignment": flare.loci == gnomix.loci == target.loci,
        "paired_bootstrap_reproducible": boot_a == boot_b,
        "go_rule_known_answer": decision["label"] == "GO_FLARE_NEXT_DEV",
        "keep_rule_known_answer": decide(keep_roots, pooled, boot_a)["label"] == "KEEP_GNOMIX",
        "tradeoff_rule_known_answer": decide(root_deltas, trade_pooled, boot_a)["label"] == "INCONCLUSIVE_TRADEOFF",
        "lineage_phase_not_swapped_post_truth": swapped_score["FLARE"]["brier"] > 0.0,
        "ordered_boundary_matching_is_one_to_one": len(one_to_one) == 1,
        "all_three_tolerances_reported": set(score["summary"]["FLARE"]["boundaries"]) == {"0.1", "0.2", "0.5"},
    }


def known_answers_command(args: argparse.Namespace) -> None:
    contract = load_contract(args.contract)
    expected_base = contract["scoring_contract"]["base_metric_library"]["sha256"]
    require(sha256(args.base_scorer) == expected_base, "Known-answer base scorer SHA-256 mismatch")
    checks = run_known_answers()
    require(all(checks.values()), f"Known-answer failure: {checks}")
    write_json(args.output, {
        "schema_version": "1.0.0",
        "experiment_id": EXPERIMENT_ID,
        "stage": "M30_SCORER_KNOWN_ANSWERS",
        "status": "PASS",
        "real_truth_accessed": False,
        "checks": checks,
        "scorer_sha256": sha256(Path(__file__)),
        "contract_sha256": sha256(args.contract),
        "base_scorer_sha256": sha256(args.base_scorer),
    })


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    known = sub.add_parser("known-answers")
    known.add_argument("--contract", required=True, type=Path)
    known.add_argument("--base-scorer", required=True, type=Path)
    known.add_argument("--output", required=True, type=Path)
    known.set_defaults(func=known_answers_command)

    score = sub.add_parser("score-compare")
    score.add_argument("--contract", required=True, type=Path)
    score.add_argument("--base-scorer", required=True, type=Path)
    score.add_argument("--genetic-map", required=True, type=Path)
    score.add_argument("--known-answer-receipt", required=True, type=Path)
    score.add_argument("--run-provenance", required=True, type=Path)
    for root in ROOTS:
        for key in (
            "truth", "target-vcf", "gnomix-binding", "gnomix-fb", "gnomix-msp",
            "flare-vcf", "flare-audit", "runtime-contract", "preflight-report",
        ):
            score.add_argument(f"--{root}-{key}", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--manifest", required=True, type=Path)
    score.set_defaults(func=score_compare_command)
    return ap


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except (ScoringError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
