#!/usr/bin/env python3
"""Describe apparent changes between authenticated diploid LAI truth classes.

This program reads only identity, query geometry and truth metadata through
``r02_lai_duration_input``.  It never loads neural inputs, computes logits,
constructs transition matrices or trains a model.  Its interval quantities are
properties of the observed query grid and unordered diploid class endpoints;
they are not haploid transition rates or demographic times.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

import r02_lai_duration_input as duration_input


SCHEMA = "r02_lai_duration_fit_transition_summary_v2"
INTERPRETATION = "APPARENT_INTERVAL_GRID_INTENSITY_NOT_HAPLOID_RATE_NOT_DEMOGRAPHY"
CLASS_NAMES = ("AFR_AFR", "AFR_EUR", "AFR_NAM", "EUR_EUR", "EUR_NAM", "NAM_NAM")
CLASS_DOSAGE = np.asarray(((2, 0, 0), (1, 1, 0), (1, 0, 1),
                           (0, 2, 0), (0, 1, 1), (0, 0, 2)), dtype=np.int8)
PRIVATE_AUTHENTICATION_FUNCTIONS = (
    "_read_json", "_load_region_metadata", "_load_truth_metadata",
    "_validate_loaded_geometry",
)


class SummaryBlocked(ValueError):
    """Fail-closed condition with a stable machine-readable status."""

    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


@dataclass(frozen=True)
class TruthRegion:
    region_id: str
    query_global_indices: tuple[int, ...]
    query_positions_cm: tuple[float, ...]
    truth_state: np.ndarray
    truth_observed: np.ndarray
    metadata_logical_bytes: int = 0
    metadata_archive_uncompressed_bytes_read: int = 0


@dataclass(frozen=True)
class TruthCase:
    case_id: str
    chromosome: int
    inventory_sha256: str
    contract_path: str
    contract_sha256: str
    person_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    query_global_indices: tuple[int, ...]
    query_positions_cm: np.ndarray
    truth_state: np.ndarray
    truth_observed: np.ndarray
    source_region_ids: tuple[str, ...]
    source_region_receipts: tuple[Mapping[str, object], ...] = ()
    metadata_logical_bytes: int = 0
    metadata_archive_uncompressed_bytes_read: int = 0
    peak_region_logical_bytes: int = 0


def assemble_truth_case(inventory: duration_input.DurationCaseInventory,
                        regions: Sequence[TruthRegion]) -> TruthCase:
    """Assemble authenticated regional cores and reject gaps or overlaps."""
    _require(isinstance(inventory, duration_input.DurationCaseInventory),
             "duration inventory is required")
    _require(bool(regions), "at least one authenticated truth region is required")
    people = len(inventory.person_ids)
    _require(len(inventory.component_ids) == people,
             "person and component identity axes differ")
    pieces: dict[int, tuple[float, np.ndarray, np.ndarray, str]] = {}
    for region in regions:
        _require(isinstance(region, TruthRegion), "truth region type differs")
        width = len(region.query_global_indices)
        _require(width == len(region.query_positions_cm)
                 and tuple(region.truth_state.shape) == (people, width)
                 and tuple(region.truth_observed.shape) == (people, width),
                 "truth region axes differ")
        _require(region.truth_observed.dtype == np.bool_,
                 "truth_observed must be boolean")
        for offset, global_index in enumerate(region.query_global_indices):
            _require(global_index not in pieces, "duplicate query_global_index across regions")
            pieces[global_index] = (
                float(region.query_positions_cm[offset]),
                np.asarray(region.truth_state[:, offset]),
                np.asarray(region.truth_observed[:, offset], dtype=bool),
                region.region_id,
            )
    expected = inventory.query_global_indices
    _require(tuple(sorted(pieces)) == expected,
             "authenticated truth regions do not cover the complete inventory axis")
    positions = np.asarray([pieces[index][0] for index in expected], dtype=np.float64)
    declared = np.asarray(inventory.query_positions_cm, dtype=np.float64)
    _require(np.array_equal(positions, declared),
             "truth region coordinates differ from the inventory")
    states = np.stack([pieces[index][1] for index in expected], axis=1).astype(np.int8)
    observed = np.stack([pieces[index][2] for index in expected], axis=1).astype(bool)
    _require(bool(np.isin(states[observed], np.arange(6)).all()),
             "observed truth class is outside 0..5")
    return TruthCase(
        case_id=inventory.case_id, chromosome=inventory.chromosome,
        inventory_sha256=inventory.inventory_sha256,
        contract_path=inventory.contract_path, contract_sha256=inventory.contract_sha256,
        person_ids=inventory.person_ids, component_ids=inventory.component_ids,
        query_global_indices=expected, query_positions_cm=positions,
        truth_state=states, truth_observed=observed,
        source_region_ids=tuple(region.region_id for region in regions),
        source_region_receipts=tuple({
            "region_id": source.region_id,
            "request_path": source.entry.get("path"),
            "request_sha256": source.entry.get("sha256"),
            "wrapper_path": source.entry.get("receipt_path"),
            "wrapper_sha256": source.entry.get("receipt_sha256"),
            "source_receipt_path": source.source_receipt_path,
            "source_receipt_sha256": source.source_receipt_sha256,
            "core_query_positions": len(source.core_global_indices),
        } for source in inventory.regions),
        metadata_logical_bytes=sum(region.metadata_logical_bytes for region in regions),
        metadata_archive_uncompressed_bytes_read=sum(
            region.metadata_archive_uncompressed_bytes_read for region in regions),
        peak_region_logical_bytes=max(region.metadata_logical_bytes for region in regions),
    )


def load_authenticated_truth_case(contract_path: str | Path, contract_sha256: str,
                                  case_id: str, *,
                                  max_region_logical_bytes: int) -> TruthCase:
    """Delegate authentication to the existing duration input implementation."""
    _require(type(max_region_logical_bytes) is int and max_region_logical_bytes > 0,
             "max_region_logical_bytes must be positive")
    inventory = duration_input.load_duration_case_inventory(
        contract_path, contract_sha256, case_id)
    contract = duration_input._read_json(inventory.contract_path, inventory.contract_sha256)
    limit = contract.get("limits", {}).get("max_batch_bytes")
    _require(type(limit) is int and limit > 0, "contract lacks metadata decode limit")
    regions: list[TruthRegion] = []
    peak_region_logical_bytes = 0
    for source in inventory.regions:
        entry = source.entry
        request = duration_input._read_json(entry["path"], entry["sha256"])
        wrapper = duration_input._read_json(entry["receipt_path"], entry["receipt_sha256"])
        expected = wrapper.get("reconstructed_arrays_sha256")
        _require(isinstance(expected, Mapping), "authenticated array hashes are absent")
        regional, regional_archive_bytes = duration_input._load_region_metadata(
            request, expected, limit)
        truth_state, truth_observed, truth_archive_bytes = duration_input._load_truth_metadata(
            request, regional, expected, limit)
        duration_input._validate_loaded_geometry(
            regional, source, inventory.person_ids, inventory.component_ids)
        core = np.asarray(regional["query_core"], dtype=bool)
        valid = np.asarray(regional["query_valid"], dtype=bool)
        _require(bool((core == core[0]).all()), "query_core differs between people")
        owned = core[0]
        observed = core[:, owned] & valid[:, owned] & np.asarray(truth_observed[:, owned], bool)
        counts = tuple(int(value) for value in observed.sum(axis=1, dtype=np.int64))
        _require(counts == source.observed_core_labels_by_person,
                 "authenticated observed-label counts differ from inventory")
        positions = tuple(float(value) for value in
                          np.asarray(regional["query_cm"])[0, owned].tolist())
        _require(positions == source.core_positions_cm,
                 "authenticated positions differ from inventory")
        region_logical_bytes = (sum(np.asarray(value).nbytes for value in regional.values())
                                + truth_state.nbytes + truth_observed.nbytes)
        _require(region_logical_bytes <= max_region_logical_bytes,
                 "region logical arrays exceed max_region_logical_bytes")
        peak_region_logical_bytes = max(peak_region_logical_bytes, region_logical_bytes)
        regions.append(TruthRegion(
            region_id=source.region_id,
            query_global_indices=source.core_global_indices,
            query_positions_cm=positions,
            truth_state=np.asarray(truth_state[:, owned]),
            truth_observed=observed,
            metadata_logical_bytes=region_logical_bytes,
            metadata_archive_uncompressed_bytes_read=(regional_archive_bytes
                                                       + truth_archive_bytes),
        ))
    case = assemble_truth_case(inventory, regions)
    _require(case.peak_region_logical_bytes == peak_region_logical_bytes,
             "peak region logical-byte accounting differs")
    return case


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open maximal observed runs; missing positions are never bridged."""
    _require(mask.ndim == 1 and mask.dtype == np.bool_, "observed row must be boolean")
    padded = np.concatenate((np.asarray([False]), mask, np.asarray([False])))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _missing_blocks(mask: np.ndarray) -> int:
    return len(_runs(~mask))


def _nearest_rank(histogram: Mapping[float, int], probability: float) -> float | None:
    total = sum(histogram.values())
    if total == 0:
        return None
    rank = max(1, math.ceil(probability * total))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= rank:
            return float(value)
    raise AssertionError("weighted rank exceeds histogram")


def _histogram_receipt(histogram: Mapping[float, int]) -> dict[str, object]:
    total = sum(histogram.values())
    ordered = sorted(histogram)
    return {
        "count": total,
        "min_delta_cm": float(ordered[0]) if ordered else None,
        "p50_delta_cm": _nearest_rank(histogram, 0.50),
        "p95_delta_cm": _nearest_rank(histogram, 0.95),
        "max_delta_cm": float(ordered[-1]) if ordered else None,
        "quantile_method": "WEIGHTED_NEAREST_RANK",
        "exact_value_histogram": [
            {"delta_cm": float(value), "delta_cm_float64_hex": float(value).hex(),
             "count": histogram[value]} for value in ordered
        ],
    }


def _empty_accumulator() -> dict[str, object]:
    return {
        "matrix": np.zeros((6, 6), dtype=np.int64),
        "exposure": np.zeros(6, dtype=np.float64),
        "changed_hist": {}, "unchanged_hist": {},
        "included": 0, "excluded": 0, "excluded_morgans": np.float64(0),
        "distance_ge_2": 0, "observed_runs": 0, "missing_blocks": 0,
        "left_boundary_runs": 0, "right_boundary_runs": 0,
    }


def _accumulate_row(positions_cm: np.ndarray, states: np.ndarray, observed: np.ndarray,
                    accumulator: dict[str, object]) -> None:
    deltas_cm = np.diff(positions_cm).astype(np.float64)
    if bool((deltas_cm <= 0).any()):
        raise SummaryBlocked("BLOCKED_ZERO_GENETIC_DISTANCE",
                             "query positions must be strictly increasing")
    runs = _runs(observed)
    accumulator["observed_runs"] += len(runs)
    accumulator["missing_blocks"] += _missing_blocks(observed)
    accumulator["left_boundary_runs"] += int(bool(runs) and runs[0][0] == 0)
    accumulator["right_boundary_runs"] += int(bool(runs) and runs[-1][1] == len(observed))
    for index, delta_cm in enumerate(deltas_cm):
        delta_morgan = np.float64(delta_cm / 100.0)
        if not (observed[index] and observed[index + 1]):
            accumulator["excluded"] += 1
            accumulator["excluded_morgans"] += delta_morgan
            continue
        left, right = int(states[index]), int(states[index + 1])
        _require(0 <= left < 6 and 0 <= right < 6, "observed truth class is outside 0..5")
        accumulator["matrix"][left, right] += 1
        # Explicit trapezoidal endpoint quadrature on the observed grid.
        accumulator["exposure"][left] += np.float64(0.5) * delta_morgan
        accumulator["exposure"][right] += np.float64(0.5) * delta_morgan
        changed = left != right
        histogram = accumulator["changed_hist"] if changed else accumulator["unchanged_hist"]
        key = np.float64(delta_cm)
        histogram[key] = histogram.get(key, 0) + 1
        dosage_distance = int(np.abs(CLASS_DOSAGE[left] - CLASS_DOSAGE[right]).sum() // 2)
        accumulator["distance_ge_2"] += int(dosage_distance >= 2)
        accumulator["included"] += 1


def _accumulator_receipt(accumulator: Mapping[str, object]) -> dict[str, object]:
    matrix = np.asarray(accumulator["matrix"], dtype=np.int64)
    exposure = np.asarray(accumulator["exposure"], dtype=np.float64)
    zero_pairs = [[CLASS_NAMES[left], CLASS_NAMES[right]]
                  for left in range(6) for right in range(6) if matrix[left, right] == 0]
    return {
        "apparent_class_endpoint_matrix": matrix.tolist(),
        "interval_grid_exposure_morgans_by_class": {
            CLASS_NAMES[index]: float(exposure[index]) for index in range(6)},
        "quadrature": "FLOAT64_TRAPEZOIDAL_HALF_INTERVAL_TO_EACH_ENDPOINT_CLASS",
        "included_adjacent_observed_intervals": int(accumulator["included"]),
        "excluded_intervals_due_to_missing_endpoint": int(accumulator["excluded"]),
        "censored_morgans": float(accumulator["excluded_morgans"]),
        "observed_runs": int(accumulator["observed_runs"]),
        "missing_blocks": int(accumulator["missing_blocks"]),
        "runs_touching_grid_axis_left_boundary": int(accumulator["left_boundary_runs"]),
        "runs_touching_grid_axis_right_boundary": int(accumulator["right_boundary_runs"]),
        "chromosome_boundaries_determined": False,
        "bridged_intervals": 0,
        "zero_distance_intervals": 0,
        "intervals_with_class_distance_ge_2": int(accumulator["distance_ge_2"]),
        "delta_cm_by_endpoint_change": {
            "changed": _histogram_receipt(accumulator["changed_hist"]),
            "unchanged": _histogram_receipt(accumulator["unchanged_hist"]),
        },
        "classes_with_zero_outgoing": [CLASS_NAMES[index] for index in range(6)
                                         if int(matrix[index].sum()) == 0],
        "class_pairs_with_zero_support": zero_pairs,
    }


def summarize_truth_case(case: TruthCase) -> dict[str, object]:
    positions = np.asarray(case.query_positions_cm, dtype=np.float64)
    _require(positions.ndim == 1 and len(positions) > 1,
             "complete case needs at least two query positions")
    _require(tuple(case.truth_state.shape) == tuple(case.truth_observed.shape)
             == (len(case.person_ids), len(positions)), "complete truth axes differ")
    deltas = np.diff(positions)
    if bool((deltas <= 0).any()):
        raise SummaryBlocked("BLOCKED_ZERO_GENETIC_DISTANCE",
                             "complete query axis is not strictly increasing")
    aggregate = _empty_accumulator()
    people = []
    initial = np.zeros(6, dtype=np.int64)
    initial_missing = 0
    for row, (person_id, component_id) in enumerate(zip(case.person_ids, case.component_ids)):
        person = _empty_accumulator()
        _accumulate_row(positions, case.truth_state[row], case.truth_observed[row], person)
        for key in ("matrix", "exposure"):
            aggregate[key] += person[key]
        for key in ("included", "excluded", "excluded_morgans", "distance_ge_2",
                    "observed_runs", "missing_blocks", "left_boundary_runs",
                    "right_boundary_runs"):
            aggregate[key] += person[key]
        for key in ("changed_hist", "unchanged_hist"):
            for value, count in person[key].items():
                aggregate[key][value] = aggregate[key].get(value, 0) + count
        if case.truth_observed[row, 0]:
            initial[int(case.truth_state[row, 0])] += 1
        else:
            initial_missing += 1
        receipt = _accumulator_receipt(person)
        people.append({"person_id": person_id, "component_id": component_id,
                       "apparent_class_endpoint_matrix":
                           receipt["apparent_class_endpoint_matrix"],
                       "interval_grid_exposure_morgans_by_class":
                           receipt["interval_grid_exposure_morgans_by_class"],
                       "included_adjacent_observed_intervals":
                           receipt["included_adjacent_observed_intervals"],
                       "excluded_intervals_due_to_missing_endpoint":
                           receipt["excluded_intervals_due_to_missing_endpoint"],
                       "censored_morgans": receipt["censored_morgans"],
                       "observed_runs": receipt["observed_runs"],
                       "missing_blocks": receipt["missing_blocks"],
                       "intervals_with_class_distance_ge_2":
                           receipt["intervals_with_class_distance_ge_2"]})
    summary = _accumulator_receipt(aggregate)
    summary.update({
        "case_id": case.case_id, "chromosome": case.chromosome,
        "inventory_sha256": case.inventory_sha256,
        "contract_path": case.contract_path, "contract_sha256": case.contract_sha256,
        "source_region_ids": list(case.source_region_ids),
        "source_region_receipts": list(case.source_region_receipts),
        "persons": len(case.person_ids), "query_positions": len(positions),
        "query_axis_sha256": _sha256_bytes(_canonical({
            "query_global_indices": case.query_global_indices,
            "query_positions_cm": positions.tolist()})),
        "query_global_index_min": min(case.query_global_indices),
        "query_global_index_max": max(case.query_global_indices),
        "grid_intervals_per_person": len(positions) - 1,
        "possible_grid_intervals": len(case.person_ids) * (len(positions) - 1),
        "chromosome_grid_span_cm": float(positions[-1] - positions[0]),
        "axis_min_delta_cm": float(deltas.min()),
        "axis_p50_delta_cm": float(np.quantile(deltas, 0.50, method="nearest")),
        "axis_p95_delta_cm": float(np.quantile(deltas, 0.95, method="nearest")),
        "axis_max_delta_cm": float(deltas.max()),
        "initial_observed_class_counts": {
            CLASS_NAMES[index]: int(initial[index]) for index in range(6)},
        "initial_position_missing_persons": initial_missing,
        "metadata_logical_bytes_loaded_second_pass": case.metadata_logical_bytes,
        "metadata_archive_uncompressed_bytes_read_second_pass":
            case.metadata_archive_uncompressed_bytes_read,
        "peak_region_logical_bytes_second_pass": case.peak_region_logical_bytes,
        "per_person": people,
    })
    _require(summary["included_adjacent_observed_intervals"]
             + summary["excluded_intervals_due_to_missing_endpoint"]
             == summary["possible_grid_intervals"],
             "included and censored intervals do not cover the grid")
    return summary


def _decode_histogram_delta(entry: Mapping[str, object]) -> np.float64:
    decimal = np.float64(entry["delta_cm"])
    hexadecimal = np.float64(float.fromhex(str(entry["delta_cm_float64_hex"])))
    _require(decimal.view(np.uint64) == hexadecimal.view(np.uint64),
             "decimal and hexadecimal delta_cm representations differ")
    return hexadecimal


def _sum_case_receipts(receipts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    accumulator = _empty_accumulator()
    possible_grid_intervals = 0
    case_order = []
    for receipt in receipts:
        case_order.append(receipt["case_id"])
        possible_grid_intervals += int(receipt["possible_grid_intervals"])
        accumulator["matrix"] += np.asarray(
            receipt["apparent_class_endpoint_matrix"], dtype=np.int64)
        # Float64 exposure is deliberately added left-to-right in receipt order.
        accumulator["exposure"] += np.asarray(
            [receipt["interval_grid_exposure_morgans_by_class"][name]
             for name in CLASS_NAMES], dtype=np.float64)
        for target, source in (
                ("included", "included_adjacent_observed_intervals"),
                ("excluded", "excluded_intervals_due_to_missing_endpoint"),
                ("excluded_morgans", "censored_morgans"),
                ("distance_ge_2", "intervals_with_class_distance_ge_2"),
                ("observed_runs", "observed_runs"),
                ("missing_blocks", "missing_blocks"),
                ("left_boundary_runs", "runs_touching_grid_axis_left_boundary"),
                ("right_boundary_runs", "runs_touching_grid_axis_right_boundary")):
            accumulator[target] += receipt[source]
        for target, source in (("changed_hist", "changed"),
                               ("unchanged_hist", "unchanged")):
            entries = receipt["delta_cm_by_endpoint_change"][source][
                "exact_value_histogram"]
            for entry in entries:
                value, count = _decode_histogram_delta(entry), int(entry["count"])
                accumulator[target][value] = accumulator[target].get(value, 0) + count
    result = _accumulator_receipt(accumulator)
    result.update({
        "possible_grid_intervals": possible_grid_intervals,
        "aggregation_order_case_ids": case_order,
        "float64_exposure_sum_order": "LEFT_TO_RIGHT_IN_AGGREGATION_ORDER_CASE_IDS",
    })
    _require(result["included_adjacent_observed_intervals"]
             + result["excluded_intervals_due_to_missing_endpoint"]
             == possible_grid_intervals,
             "aggregate included and censored intervals do not cover the grid")
    return result


def build_summary(cases: Sequence[TruthCase], *, code_sha256: str,
                  duration_input_sha256: str,
                  max_retained_truth_bytes: int | None = None,
                  max_region_logical_bytes: int | None = None) -> dict[str, object]:
    _require(len(cases) > 0 and len({case.case_id for case in cases}) == len(cases),
             "unique FIT cases are required")
    _require(len({(case.contract_path, case.contract_sha256, case.chromosome)
                  for case in cases}) == 1,
             "FIT cases must share one authenticated contract and chromosome")
    receipts = [summarize_truth_case(case) for case in cases]
    retained_truth_bytes = sum(case.truth_state.nbytes + case.truth_observed.nbytes
                               for case in cases)
    peak_region_logical_bytes = max(case.peak_region_logical_bytes for case in cases)
    if max_retained_truth_bytes is not None:
        _require(type(max_retained_truth_bytes) is int and max_retained_truth_bytes > 0,
                 "max_retained_truth_bytes must be positive")
        _require(retained_truth_bytes <= max_retained_truth_bytes,
                 "retained truth arrays exceed max_retained_truth_bytes")
    if max_region_logical_bytes is not None:
        _require(type(max_region_logical_bytes) is int and max_region_logical_bytes > 0,
                 "max_region_logical_bytes must be positive")
        _require(peak_region_logical_bytes <= max_region_logical_bytes,
                 "peak region logical arrays exceed max_region_logical_bytes")
    leave_one_out = []
    if len(receipts) == 1:
        leave_one_out.append({
            "omitted_case_id": receipts[0]["case_id"], "retained_cases": [],
            "descriptive_only_not_jackknife_inference": True,
            "degenerate_single_case_not_computed": True, "summary": None})
    else:
        for omitted in receipts:
            retained = [receipt for receipt in receipts
                        if receipt["case_id"] != omitted["case_id"]]
            leave_one_out.append({"omitted_case_id": omitted["case_id"],
                                  "retained_cases": [value["case_id"] for value in retained],
                                  "descriptive_only_not_jackknife_inference": True,
                                  "degenerate_single_case_not_computed": False,
                                  "summary": _sum_case_receipts(retained)})
    request = {"contract_path": cases[0].contract_path,
               "contract_sha256": cases[0].contract_sha256,
               "case_ids": [case.case_id for case in cases],
               "inventory_sha256": [case.inventory_sha256 for case in cases]}
    return {
        "schema": SCHEMA, "status": "FIT_TRUTH_GRID_DESCRIBED_NOT_TRAINED",
        "interpretation": INTERPRETATION, "identifies_haploid_events": False,
        "identifies_demographic_time": False, "donor_sharing_across_cases": True,
        "independent_units_claimed": None,
        "ancestry_dosage_by_class": {
            CLASS_NAMES[index]: CLASS_DOSAGE[index].astype(int).tolist() for index in range(6)},
        "dosage_mapping_use": "ENDPOINT_CLASS_DISTANCE_ONLY_NOT_EVENT_ATTRIBUTION",
        "request": request, "request_sha256": _sha256_bytes(_canonical(request)),
        "code_sha256": code_sha256, "duration_input_sha256": duration_input_sha256,
        "duration_input_authentication_api": {
            "public_inventory_function": "load_duration_case_inventory",
            "private_function_names": list(PRIVATE_AUTHENTICATION_FUNCTIONS),
            "module_sha256": duration_input_sha256,
        },
        "resource_accounting": {
            "retained_truth_bytes": retained_truth_bytes,
            "max_retained_truth_bytes": max_retained_truth_bytes,
            "peak_region_logical_bytes": peak_region_logical_bytes,
            "max_region_logical_bytes": max_region_logical_bytes,
            "peak_process_rss_bytes": None,
            "peak_process_rss_measured": False,
        },
        "opened_roles": ["FIT"], "select_opened": False, "evaluate_opened": False,
        "training_performed": False, "neural_forward_performed": False,
        "matrix_exp_performed": False, "optimizer_performed": False,
        "cases": receipts, "aggregate": _sum_case_receipts(receipts),
        "leave_one_case_out_sensitivity": leave_one_out,
    }


def _write_fresh_atomic(output: Path, payload: bytes) -> None:
    """Publish complete bytes without replacing an output won by another process."""
    _require(output.parent.is_dir(), "output parent directory is absent")
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ValueError("fresh output path required") from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--max-retained-truth-bytes", type=int, required=True)
    parser.add_argument("--max-region-logical-bytes", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    _require(not output.exists() and output.parent.is_dir(), "fresh output path required")
    _require(args.max_retained_truth_bytes > 0 and args.max_region_logical_bytes > 0,
             "resource byte limits must be positive")
    cases = []
    for case_id in args.case_id:
        cases.append(load_authenticated_truth_case(
            args.contract, args.contract_sha256, case_id,
            max_region_logical_bytes=args.max_region_logical_bytes))
        retained = sum(case.truth_state.nbytes + case.truth_observed.nbytes
                       for case in cases)
        _require(retained <= args.max_retained_truth_bytes,
                 "retained truth arrays exceed max_retained_truth_bytes")
    result = build_summary(
        cases, code_sha256=_file_sha256(__file__),
        duration_input_sha256=_file_sha256(duration_input.__file__),
        max_retained_truth_bytes=args.max_retained_truth_bytes,
        max_region_logical_bytes=args.max_region_logical_bytes)
    _write_fresh_atomic(output, _canonical(result))


if __name__ == "__main__":
    main()
