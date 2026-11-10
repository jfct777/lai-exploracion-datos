#!/usr/bin/env python3
"""Authenticate GT FIT occurrences and build weighted donor-training plans.

The frozen GT ``PLAN.json`` remains a preparation receipt with
``training_admitted=false``.  A separate explicit reference using
``WEIGHTED_REFERENCE_SCHEMA`` opts into this converter; it does not alter GT.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from r02_lai_donor_context import STATE_NAMES
from r02_lai_donor_train import WeightedLogicalBatchPlan, WeightedMicrobatchPlan


WEIGHTED_REFERENCE_SCHEMA = "r02_lai_fit_weighted_training_reference_v1"
GT_PLAN_SCHEMA = "r02_lai_fit_stratified_block_plan_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _resolved_file(raw: object, base: Path, expected_sha256: object,
                   name: str) -> Path:
    require(isinstance(raw, str) and raw, f"{name} path required")
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    require(path.is_file(), f"{name} is absent")
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and sha256(path) == expected_sha256, f"{name} SHA-256 differs")
    return path


@dataclass(frozen=True)
class WeightedPlanReference:
    plan_json: Path
    occurrences_npz: Path
    plan_sha256: str
    occurrences_sha256: str


def validate_weighted_plan_reference(value: Mapping[str, object],
                                     base: Path) -> WeightedPlanReference:
    require(isinstance(value, Mapping)
            and set(value) == {"schema", "plan_json", "plan_sha256",
                               "occurrences_npz", "occurrences_sha256"}
            and value.get("schema") == WEIGHTED_REFERENCE_SCHEMA,
            "weighted FIT training reference schema or fields differ")
    base = Path(base).resolve()
    plan = _resolved_file(value["plan_json"], base, value["plan_sha256"],
                          "weighted PLAN.json")
    occurrences = _resolved_file(value["occurrences_npz"], base,
                                  value["occurrences_sha256"],
                                  "weighted OCCURRENCES.npz")
    return WeightedPlanReference(plan, occurrences, str(value["plan_sha256"]),
                                 str(value["occurrences_sha256"]))


@dataclass(frozen=True)
class PhysicalQuerySegment:
    region_id: str
    person_index: int
    query_start: int
    query_stop: int

    def receipt(self) -> dict[str, object]:
        return {"region_id": self.region_id, "person_index": self.person_index,
                "query_start": self.query_start, "query_stop": self.query_stop}


@dataclass(frozen=True)
class LoadedWeightedFitPlan:
    logical_batches: tuple[WeightedLogicalBatchPlan, ...]
    physical_segments: tuple[PhysicalQuerySegment, ...]
    plan_metadata: dict[str, object]
    source_receipt: dict[str, object]


def _authenticate_parent(metadata: Mapping[str, object], key: str,
                         actual_path: Path) -> dict[str, str]:
    value = metadata.get(key)
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
            f"GT {key} identity differs")
    actual = _resolved_file(str(actual_path), Path.cwd(), value["sha256"],
                            f"live {key}")
    # A frozen package may relocate the same authenticated bytes.  Preserve
    # the historical path without requiring that workstation path at runtime.
    declared = Path(str(value["path"]))
    if declared.is_file():
        require(sha256(declared) == value["sha256"],
                f"GT historical {key} bytes changed")
    return {"declared_path": str(value["path"]), "live_path": str(actual),
            "sha256": str(value["sha256"])}


def _segments(rows: np.ndarray) -> tuple[tuple[PhysicalQuerySegment, ...],
                                         dict[tuple[int, int], str]]:
    by_person: dict[int, set[int]] = {}
    for person, query in rows[:, 2:4].tolist():
        by_person.setdefault(int(person), set()).add(int(query))
    result: list[PhysicalQuerySegment] = []
    cell_region: dict[tuple[int, int], str] = {}
    for person in sorted(by_person):
        values = sorted(by_person[person])
        groups: list[list[int]] = []
        for query in values:
            if not groups or query != groups[-1][-1] + 1:
                groups.append([query])
            else:
                groups[-1].append(query)
        for number, group in enumerate(groups):
            region = f"weighted_p{person:03d}_segment{number:03d}"
            result.append(PhysicalQuerySegment(region, person, group[0], group[-1] + 1))
            for query in group:
                cell_region[(person, query)] = region
    return tuple(result), cell_region


def _plans(rows: np.ndarray, coefficients: np.ndarray,
           person_ids: Sequence[str]) -> tuple[
               tuple[WeightedLogicalBatchPlan, ...],
               tuple[PhysicalQuerySegment, ...]]:
    segments, cell_region = _segments(rows)
    segment_by_id = {value.region_id: value for value in segments}
    plans: list[WeightedLogicalBatchPlan] = []
    for update in range(int(rows[:, 0].max()) + 1):
        take = np.flatnonzero(rows[:, 0] == update)
        aggregate: dict[tuple[int, int], list[float | int]] = {}
        region_order: list[str] = []
        for position in take.tolist():
            person, query = (int(rows[position, 2]), int(rows[position, 3]))
            cell = (person, query)
            if cell not in aggregate:
                aggregate[cell] = [0.0, 0]
                region = cell_region[cell]
                if region not in region_order:
                    region_order.append(region)
            aggregate[cell][0] = float(aggregate[cell][0]) + float(coefficients[position])
            aggregate[cell][1] = int(aggregate[cell][1]) + 1
        microbatches: list[WeightedMicrobatchPlan] = []
        for region in region_order:
            segment = segment_by_id[region]
            cells = sorted(cell for cell in aggregate if cell_region[cell] == region)
            require(len({person for person, _ in cells}) == 1,
                    "one weighted physical segment must belong to one person")
            person = cells[0][0]
            indices = tuple(query for _, query in cells)
            weights = tuple(float(aggregate[cell][0]) for cell in cells)
            counts = tuple(int(aggregate[cell][1]) for cell in cells)
            receipt = {"schema": "r02_lai_weighted_microbatch_identity_v1",
                       "step_index": update, "region_id": region,
                       "query_start": segment.query_start,
                       "query_stop": segment.query_stop,
                       "person_id": str(person_ids[person]),
                       "scored_query_global_indices": list(indices),
                       "loss_coefficients": list(weights),
                       "occurrence_counts": list(counts)}
            microbatches.append(WeightedMicrobatchPlan(
                canonical_sha256(receipt), region,
                segment.query_start, segment.query_stop,
                (str(person_ids[person]),),
                (indices,), (weights,), (counts,)))
        plans.append(WeightedLogicalBatchPlan(
            update, f"weighted-u{update:03d}", tuple(microbatches), len(take)))
    return tuple(plans), segments


def load_weighted_fit_plan(
        reference: WeightedPlanReference, *, person_ids: Sequence[str],
        query_valid: np.ndarray, truth_observed: np.ndarray, truth_state: np.ndarray,
        source_spec_path: Path, query_truth_overlay_path: Path) -> LoadedWeightedFitPlan:
    """Validate GT algebra/source bindings and return executable weighted plans."""
    metadata = json.loads(reference.plan_json.read_text())
    required = {
        "schema": GT_PLAN_SCHEMA, "role": "FIT",
        "coefficient_uses": "ANCHOR_STATE_NOT_QUERY_STATE",
        "duplicate_rule": "KEEP_OCCURRENCES_OR_SUM_THEIR_COEFFICIENTS_PER_UPDATE_PERSON_QUERY",
        "atomic_update": "ADAM_AND_CLIPPING_ONLY_AFTER_ALL_WEIGHTED_OCCURRENCES_COMPLETE",
        "target": "EQUAL_PERSON_THEN_EQUAL_QUERY_COMPLETE_RECTANGLE",
        "loss": "SUM_OCCURRENCE_COEFFICIENT_TIMES_NLL_NO_RENORMALIZATION",
        "required_training_schema": "EXPLICIT_OCCURRENCE_WEIGHTED_NOT_DONOR_V1",
    }
    require(all(metadata.get(key) == value for key, value in required.items()),
            "GT weighted plan scientific metadata differs")
    require(metadata.get("training_admitted") is False
            and metadata.get("prepared_only") is True
            and metadata.get("trainer_v1_accepts_plan") is False,
            "GT preparation/admission boundary differs")
    require(metadata.get("row_columns") ==
            ["update", "block", "person_index", "query_index", "anchor_query_index"],
            "GT occurrence row columns differ")
    require(metadata.get("state_order") == list(STATE_NAMES),
            "GT state order differs")
    require(metadata.get("occurrences_sha256") == reference.occurrences_sha256,
            "GT metadata does not bind referenced occurrences")

    source_spec = _authenticate_parent(metadata, "source_spec", source_spec_path)
    sources = metadata.get("sources")
    require(isinstance(sources, Mapping), "GT source identities missing")
    overlay = _authenticate_parent(sources, "query_truth_overlay",
                                   query_truth_overlay_path)

    valid = np.asarray(query_valid, dtype=np.bool_)
    observed = np.asarray(truth_observed, dtype=np.bool_)
    truth = np.asarray(truth_state)
    require(valid.shape == observed.shape == truth.shape and valid.ndim == 2
            and tuple(metadata.get("input_shape", ())) == valid.shape
            and len(person_ids) == valid.shape[0],
            "GT plan and live FIT query axes differ")
    require(bool(valid.all()) and bool(observed.all()),
            "weighted GT v1 requires complete valid and observed FIT rectangle")
    require(np.issubdtype(truth.dtype, np.integer)
            and np.isin(truth, np.arange(len(STATE_NAMES))).all(),
            "live FIT truth states differ from six-state GT plan")

    with np.load(reference.occurrences_npz, allow_pickle=False) as stored:
        require(set(stored.files) == {"occurrence_rows", "loss_coefficient"},
                "GT occurrence archive fields differ")
        rows = np.asarray(stored["occurrence_rows"])
        coefficients = np.asarray(stored["loss_coefficient"])
    updates = metadata.get("updates"); blocks = metadata.get("blocks_per_update")
    width = metadata.get("width_queries")
    require(all(type(value) is int and value > 0 for value in (updates, blocks, width))
            and rows.dtype == np.int64 and rows.shape == (updates * blocks * width, 5)
            and coefficients.dtype == np.float64 and coefficients.shape == (len(rows),)
            and np.isfinite(coefficients).all() and (coefficients > 0).all(),
            "GT occurrence arrays, budget or coefficient values differ")
    require(metadata.get("occurrences") == len(rows),
            "GT occurrence count differs")
    require(np.array_equal(rows[:, 0], np.repeat(np.arange(updates), blocks * width))
            and np.array_equal(rows[:, 1], np.tile(np.repeat(np.arange(blocks), width), updates)),
            "GT occurrence update/block order differs")
    require(((0 <= rows[:, 2]) & (rows[:, 2] < valid.shape[0])
             & (0 <= rows[:, 3]) & (rows[:, 3] < valid.shape[1])
             & (0 <= rows[:, 4]) & (rows[:, 4] < valid.shape[1])).all(),
            "GT occurrence indices are outside live FIT axes")

    counts = np.bincount(truth.ravel().astype(np.int64), minlength=len(STATE_NAMES))
    quota = np.asarray(metadata.get("anchor_quota_per_update"), dtype=np.int64)
    require(quota.shape == counts.shape and (quota > 0).all()
            and int(quota.sum()) == blocks and width <= valid.shape[1]
            and metadata.get("state_counts") == counts.tolist(),
            "GT state counts or anchor quotas differ from live FIT truth")
    expected = np.empty(len(rows), dtype=np.float64)
    for update in range(updates):
        realized_anchor_states = np.zeros(len(STATE_NAMES), dtype=np.int64)
        for block in range(blocks):
            start = (update * blocks + block) * width
            stop = start + width
            group = rows[start:stop]
            person, anchor = int(group[0, 2]), int(group[0, 4])
            require((group[:, 2] == person).all() and (group[:, 4] == anchor).all()
                    and np.array_equal(group[:, 3],
                        (anchor + np.arange(width, dtype=np.int64)) % valid.shape[1]),
                    "GT block is not the declared circular ordered query sequence")
            state = int(truth[person, anchor])
            realized_anchor_states[state] += 1
            expected[start:stop] = (counts[state] /
                (truth.size * int(quota[state]) * width))
        require(np.array_equal(realized_anchor_states, quota),
                "GT realized anchor-state quotas differ")
    require(np.array_equal(coefficients, expected),
            "GT occurrence coefficient differs from anchor-state formula")
    sums = [float(coefficients[rows[:, 0] == update].sum())
            for update in range(updates)]
    require(all(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=5e-16)
                for value in sums)
            and np.allclose(sums, metadata.get("weight_sum_per_update"),
                            rtol=0.0, atol=5e-16),
            "GT update coefficient sums differ")
    sampled_states = np.bincount(truth[rows[:, 2], rows[:, 3]].astype(np.int64),
                                 minlength=len(STATE_NAMES))
    require(metadata.get("sampled_state_counts") == sampled_states.tolist()
            and metadata.get("unique_cells") == len(set(map(tuple, rows[:, 2:4])))
            and metadata.get("sampled_people") == len(np.unique(rows[:, 2])),
            "GT sampled occurrence summaries differ")

    logical, segments = _plans(rows, coefficients, person_ids)
    receipt = {
        "schema": "r02_lai_fit_weighted_training_conversion_v1",
        "plan_json": {"path": str(reference.plan_json),
                      "sha256": reference.plan_sha256},
        "occurrences_npz": {"path": str(reference.occurrences_npz),
                            "sha256": reference.occurrences_sha256},
        "source_spec": source_spec, "query_truth_overlay": overlay,
        "updates": len(logical), "physical_segments": len(segments),
        "physical_segment_receipts": [value.receipt() for value in segments],
        "physical_segment_receipts_sha256": canonical_sha256(
            [value.receipt() for value in segments]),
        "executed_context_geometry":
            "CONTIGUOUS_SEGMENTS_OVER_COMPLETE_FROZEN_PLAN_NO_CIRCULAR_JOIN",
        "occurrences": len(rows), "unique_cells": len(set(map(tuple, rows[:, 2:4]))),
        "role": "FIT", "select_or_evaluate_truth_read": False,
    }
    return LoadedWeightedFitPlan(logical, segments, dict(metadata), receipt)
