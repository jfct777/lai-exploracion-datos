#!/usr/bin/env python3
"""Shared primitives for the M38B FIT-only out-of-fold experiment."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from m37_trace_core import PROBABILITY_FLOOR, require


PERSON_MEMBERS = {
    "sample_key_sha256",
    "baseline_states",
    "evidence_field",
    "event_counts",
}
EVENT_INDEX_MEMBERS = {
    "event_sample",
    "schedule_sample",
}


def stable_digest(value: object, seed: int, domain: str) -> bytes:
    """Return a domain-separated digest without consulting labels or outcomes."""
    if isinstance(value, (bytes, np.bytes_)):
        encoded = bytes(value)
    else:
        encoded = str(value).encode("utf-8")
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + str(int(seed)).encode("ascii") + b"\0" + encoded
    ).digest()


def build_outer_roles(
    sample_keys: np.ndarray,
    *,
    outer_seed: int,
    inner_seed_start: int,
    folds: int = 3,
    score_people: int = 32,
    select_people: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Create exact person-level SCORE/TRAIN/SELECT roles from sample hashes only.

    The returned role matrix has one row per outer fold and one column per
    person.  Values are ``TRAIN``, ``SELECT`` or ``SCORE``.  Inner seeds are
    the first deterministic seeds that make M37's threshold splitter select
    exactly the preregistered number of people.
    """
    keys = np.asarray(sample_keys)
    require(
        keys.ndim == 1
        and len(keys) == folds * score_people
        and len(np.unique(keys)) == len(keys),
        "M38B sample axis must contain exactly 96 unique people",
    )
    order = sorted(
        range(len(keys)),
        key=lambda index: stable_digest(keys[index], outer_seed, "M38B_OUTER_SCORE_V1"),
    )
    roles = np.full((folds, len(keys)), "TRAIN", dtype="U6")
    inner_seeds = np.empty(folds, dtype=np.int64)
    for fold in range(folds):
        score = np.asarray(order[fold * score_people:(fold + 1) * score_people], dtype=np.int64)
        roles[fold, score] = "SCORE"
        remaining = np.flatnonzero(roles[fold] != "SCORE")
        require(len(remaining) == len(keys) - score_people, "outer fold size differs")
        chosen_seed: int | None = None
        selected: np.ndarray | None = None
        for seed in range(inner_seed_start, inner_seed_start + 1_000_000):
            buckets = np.asarray([
                int.from_bytes(
                    hashlib.sha256(bytes(keys[index]) + str(seed).encode("ascii")).digest()[:4],
                    "big",
                ) % 10_000
                for index in remaining
            ])
            candidate = remaining[buckets < 2_500]
            if len(candidate) == select_people:
                chosen_seed, selected = seed, candidate
                break
        require(chosen_seed is not None and selected is not None,
                "could not construct the exact inner split")
        roles[fold, selected] = "SELECT"
        inner_seeds[fold] = chosen_seed
        require(
            np.count_nonzero(roles[fold] == "TRAIN") == 48
            and np.count_nonzero(roles[fold] == "SELECT") == select_people
            and np.count_nonzero(roles[fold] == "SCORE") == score_people,
            "M38B TRAIN/SELECT/SCORE counts differ",
        )
    require(
        np.all(np.sum(roles == "SCORE", axis=0) == 1),
        "each FIT person must appear in SCORE exactly once",
    )
    return roles, inner_seeds


def _person_index_map(people: np.ndarray, total: int) -> np.ndarray:
    people = np.asarray(people, dtype=np.int64)
    require(
        people.ndim == 1
        and len(people) > 0
        and len(np.unique(people)) == len(people)
        and np.all((people >= 0) & (people < total)),
        "feature person selection differs",
    )
    remap = np.full(total, -1, dtype=np.int64)
    remap[people] = np.arange(len(people), dtype=np.int64)
    return remap


def slice_features(features: Mapping[str, np.ndarray], people: Sequence[int]) -> dict[str, np.ndarray]:
    """Subset TRACE features and remap all ragged person/event indices."""
    require(PERSON_MEMBERS.issubset(features), "TRACE feature person members differ")
    total = len(np.asarray(features["sample_key_sha256"]))
    selected = np.asarray(people, dtype=np.int64)
    remap = _person_index_map(selected, total)
    event_sample = np.asarray(features.get("event_sample", np.empty(0, dtype=np.int64)), dtype=np.int64)
    schedule_sample = np.asarray(features.get("schedule_sample", np.empty(0, dtype=np.int64)), dtype=np.int64)
    require(
        np.all((event_sample >= 0) & (event_sample < total))
        and np.all((schedule_sample >= 0) & (schedule_sample < total)),
        "TRACE ragged person index differs",
    )
    event_keep = remap[event_sample] >= 0
    schedule_keep = remap[schedule_sample] >= 0
    event_length, schedule_length = len(event_sample), len(schedule_sample)
    payload: dict[str, np.ndarray] = {}
    for name, raw in features.items():
        value = np.asarray(raw)
        if name in PERSON_MEMBERS:
            payload[name] = np.ascontiguousarray(value[selected])
        elif name == "event_sample":
            payload[name] = np.ascontiguousarray(remap[event_sample[event_keep]], dtype=np.uint32)
        elif name == "schedule_sample":
            payload[name] = np.ascontiguousarray(remap[schedule_sample[schedule_keep]], dtype=np.uint32)
        elif name.startswith("event_") and value.ndim >= 1 and len(value) == event_length:
            payload[name] = np.ascontiguousarray(value[event_keep])
        elif name == "schedule_marker" and value.ndim == 1 and len(value) == schedule_length:
            payload[name] = np.ascontiguousarray(value[schedule_keep])
        else:
            payload[name] = np.ascontiguousarray(value)
    require(
        len(payload["event_sample"]) == len(payload.get("event_values", []))
        and len(payload["schedule_sample"]) == len(payload.get("schedule_marker", [])),
        "sliced TRACE ragged axes differ",
    )
    return payload


def voronoi_cm_weights(marker_cm: np.ndarray) -> np.ndarray:
    """Length represented by each marker on an ordered genetic map.

    A recombination-map plateau can place several loci at the same cM.  Their
    shared Voronoi cell is divided equally so the score cannot change merely
    by reordering loci that have identical genetic coordinates.
    """
    position = np.asarray(marker_cm, dtype=np.float64)
    require(
        position.ndim == 1
        and len(position) > 1
        and np.isfinite(position).all()
        and np.all(np.diff(position) >= 0),
        "marker cM axis differs",
    )
    unique, inverse, counts = np.unique(
        position, return_inverse=True, return_counts=True,
    )
    require(len(unique) > 1, "cM Voronoi weights differ")
    midpoint = (unique[:-1] + unique[1:]) / 2.0
    edges = np.concatenate(([unique[0]], midpoint, [unique[-1]]))
    unique_weight = np.diff(edges)
    weight = unique_weight[inverse] / counts[inverse]
    # Tied map positions have zero represented length.  A chromosome with a
    # completely flat map is invalid for a cM-weighted primary endpoint.
    require(np.all(weight >= 0) and float(weight.sum()) > 0, "cM Voronoi weights differ")
    return weight


def per_person_log_loss(probability: np.ndarray, truth: np.ndarray,
                        marker_cm: np.ndarray, *, weighted: bool) -> np.ndarray:
    """Return one diploid log-loss value per person."""
    value = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64)
    require(
        value.ndim == 3
        and value.shape[:2] == labels.shape
        and value.shape[2] == 6
        and np.isfinite(value).all()
        and np.all(value >= 0)
        and np.all((labels >= 0) & (labels < 6)),
        "prediction/truth axes differ",
    )
    value = np.maximum(value, PROBABILITY_FLOOR)
    value /= value.sum(axis=2, keepdims=True)
    selected = np.take_along_axis(value, labels[:, :, None], axis=2)[:, :, 0]
    loss = -np.log(selected)
    if not weighted:
        return loss.mean(axis=1)
    weight = voronoi_cm_weights(marker_cm)
    require(len(weight) == loss.shape[1], "loss/cM axes differ")
    return (loss * weight[None, :]).sum(axis=1) / weight.sum()


def smooth_evidence_triangular(evidence: np.ndarray, marker_cm: np.ndarray,
                               radius_cm: float) -> np.ndarray:
    """Spread anchor evidence continuously over a fixed cM radius."""
    field = np.asarray(evidence, dtype=np.float32)
    position = np.asarray(marker_cm, dtype=np.float64)
    require(
        field.ndim == 3
        and field.shape[1] == len(position)
        and field.shape[2] == 6
        and radius_cm > 0,
        "analytic TRACE evidence axes differ",
    )
    output = np.zeros_like(field)
    anchors = np.flatnonzero(np.any(field != 0, axis=(0, 2)))
    for anchor in anchors.tolist():
        left = int(np.searchsorted(position, position[anchor] - radius_cm, side="left"))
        right = int(np.searchsorted(position, position[anchor] + radius_cm, side="right"))
        local = np.arange(left, right, dtype=np.int64)
        weight = np.maximum(1.0 - np.abs(position[local] - position[anchor]) / radius_cm, 0.0)
        output[:, local] += field[:, anchor, None, :] * weight[None, :, None]
    return output


def analytic_residual(baseline: np.ndarray, evidence: np.ndarray, strength: float) -> np.ndarray:
    """Apply a phase-free multiplicative evidence update to a FLARE baseline."""
    base = np.asarray(baseline, dtype=np.float64)
    field = np.asarray(evidence, dtype=np.float64)
    require(
        base.shape == field.shape
        and base.ndim == 3
        and base.shape[2] == 6
        and strength >= 0
        and np.isfinite(strength),
        "analytic TRACE inputs differ",
    )
    if strength == 0:
        return np.asarray(baseline, dtype=np.float32).copy()
    logits = np.log(np.maximum(base, PROBABILITY_FLOOR)) + float(strength) * field
    logits -= logits.max(axis=2, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=2, keepdims=True)
    return result.astype(np.float32)
