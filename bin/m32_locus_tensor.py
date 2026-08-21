#!/usr/bin/env python3
"""Pure, dependency-free tensor helpers for the M32 synthetic smoke."""

from __future__ import annotations

import hashlib
import json
import math
import random
import bisect
from typing import Sequence


ANCESTRIES = ("AFR", "EUR", "ASIA")
MISSING_STATE = -1


def _validated_haplotypes(
    haplotypes: Sequence[Sequence[Sequence[int]]], minor_codes: Sequence[int]
) -> tuple[list[list[list[int]]], list[int]]:
    hap = [[list(pair) for pair in person] for person in haplotypes]
    minor = list(minor_codes)
    if not hap or not minor or any(len(person) != len(minor) for person in hap):
        raise ValueError("haplotypes must have shape (individual, locus, 2)")
    if any(len(pair) != 2 for person in hap for pair in person):
        raise ValueError("haplotypes must have shape (individual, locus, 2)")
    if any(state not in (MISSING_STATE, 0, 1) for person in hap for pair in person for state in pair):
        raise ValueError("haplotypes must contain only -1, 0 or 1")
    if any(code not in (0, 1) for code in minor):
        raise ValueError("minor_codes must contain only 0 or 1")
    return hap, minor


def phase_aware_minor_presence(
    haplotypes: Sequence[Sequence[Sequence[int]]], minor_codes: Sequence[int]
) -> list[list[list[int | None]]]:
    """Return haplotype-specific minor presence; missing remains None."""
    hap, minor = _validated_haplotypes(haplotypes, minor_codes)
    return [
        [
            [None if state == MISSING_STATE else int(state == minor[locus]) for state in pair]
            for locus, pair in enumerate(person)
        ]
        for person in hap
    ]


def primary_diploid_channels(
    haplotypes: Sequence[Sequence[Sequence[int]]], minor_codes: Sequence[int]
) -> dict[str, list[list[int | bool | None]]]:
    """Build phase-invariant dosage, callability and phase-ambiguity channels."""
    presence = phase_aware_minor_presence(haplotypes, minor_codes)
    dosage: list[list[int | None]] = []
    callable_mask: list[list[bool]] = []
    heterozygous: list[list[bool]] = []
    for person in presence:
        dosage_row: list[int | None] = []
        callable_row: list[bool] = []
        heterozygous_row: list[bool] = []
        for pair in person:
            callable_locus = all(value is not None for value in pair)
            value = sum(int(x) for x in pair) if callable_locus else None
            dosage_row.append(value)
            callable_row.append(callable_locus)
            heterozygous_row.append(callable_locus and value == 1)
        dosage.append(dosage_row)
        callable_mask.append(callable_row)
        heterozygous.append(heterozygous_row)
    return {
        "minor_dosage": dosage,
        "callable_mask": callable_mask,
        "heterozygous_phase_ambiguity": heterozygous,
    }


def swap_homologues(haplotypes: Sequence[Sequence[Sequence[int]]]) -> list[list[list[int]]]:
    hap, _ = _validated_haplotypes(haplotypes, [0] * len(haplotypes[0]))
    return [[[pair[1], pair[0]] for pair in person] for person in hap]


def apply_phase_switches(
    haplotypes: Sequence[Sequence[Sequence[int]]], switches: Sequence[Sequence[bool]]
) -> list[list[list[int]]]:
    """Swap h0/h1 independently for the requested individual-by-locus cells."""
    hap, _ = _validated_haplotypes(haplotypes, [0] * len(haplotypes[0]))
    switch_rows = [list(row) for row in switches]
    if len(switch_rows) != len(hap) or any(len(row) != len(hap[0]) for row in switch_rows):
        raise ValueError("switches must have shape (individual, locus)")
    return [
        [[pair[1], pair[0]] if switch_rows[person_index][locus] else pair.copy() for locus, pair in enumerate(person)]
        for person_index, person in enumerate(hap)
    ]


def permute_reference_labels(labels: Sequence[str], seed: int) -> list[str]:
    """Permute ancestry labels across complete diploid people, preserving counts."""
    original = list(labels)
    if len(original) < 2:
        raise ValueError("at least two reference individuals are required")
    if not set(original).issubset(ANCESTRIES):
        raise ValueError("reference labels contain an unsupported ancestry")
    if len(set(original)) < 2:
        raise ValueError("label permutation cannot break association with one ancestry")
    rng = random.Random(seed)
    for _ in range(64):
        candidate = original.copy()
        rng.shuffle(candidate)
        if candidate != original:
            if sorted(candidate) != sorted(original):
                raise AssertionError("label permutation changed group sizes")
            return candidate
    for shift in range(1, len(original)):
        candidate = original[-shift:] + original[:-shift]
        if candidate != original:
            return candidate
    raise AssertionError("failed to construct a non-identity label permutation")


def reference_support(
    reference_haplotypes: Sequence[Sequence[Sequence[int]]],
    minor_codes: Sequence[int],
    labels: Sequence[str],
) -> tuple[list[list[float]], list[list[bool]]]:
    """Mean minor dosage among callable REF_LAI people at each locus and ancestry."""
    hap, minor = _validated_haplotypes(reference_haplotypes, minor_codes)
    labels_list = list(labels)
    if len(labels_list) != len(hap):
        raise ValueError("reference labels must contain one label per diploid person")
    primary = primary_diploid_channels(hap, minor)
    support = [[0.0 for _ in ANCESTRIES] for _ in minor]
    observed = [[False for _ in ANCESTRIES] for _ in minor]
    for locus in range(len(minor)):
        for ancestry_index, ancestry in enumerate(ANCESTRIES):
            values = [
                primary["minor_dosage"][person][locus]
                for person, label in enumerate(labels_list)
                if label == ancestry and primary["callable_mask"][person][locus]
            ]
            if values:
                support[locus][ancestry_index] = sum(int(value) for value in values) / len(values)
                observed[locus][ancestry_index] = True
    return support, observed


def pad_locus_axis(values: Sequence[Sequence[object]], pad: int, fill: object = 0) -> list[list[object]]:
    if pad < 0:
        raise ValueError("padding must be non-negative")
    return [[fill] * pad + list(person) + [fill] * pad for person in values]


def ragged_context_indices(
    grid_cm: Sequence[float], rare_cm: Sequence[float], radius_cm: float
) -> list[list[int]]:
    if radius_cm <= 0 or not math.isfinite(radius_cm):
        raise ValueError("radius_cm must be positive and finite")
    grid = [float(value) for value in grid_cm]
    rare = [float(value) for value in rare_cm]
    if any(not math.isfinite(value) for value in grid + rare):
        raise ValueError("coordinates contain non-finite cM")
    if any(right < left for left, right in zip(grid, grid[1:])) or any(right < left for left, right in zip(rare, rare[1:])):
        raise ValueError("coordinates must be non-decreasing")
    return [
        list(range(bisect.bisect_left(rare, marker - radius_cm), bisect.bisect_right(rare, marker + radius_cm)))
        for marker in grid
    ]


def pad_ragged_indices(rows: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[bool]]]:
    width = max((len(row) for row in rows), default=0)
    padded = [list(row) + [-1] * (width - len(row)) for row in rows]
    mask = [[index >= 0 for index in row] for row in padded]
    return padded, mask


def build_ordered_sequence(
    marker_ids: Sequence[str],
    marker_bp: Sequence[int],
    marker_cm: Sequence[float],
    flare_posteriors: Sequence[Sequence[Sequence[float]]],
    locus_ids: Sequence[str],
    locus_bp: Sequence[int],
    locus_cm: Sequence[float],
    minor_codes: Sequence[int],
    target_haplotypes: Sequence[Sequence[Sequence[int]]],
    reference_haplotypes: Sequence[Sequence[Sequence[int]]],
    reference_labels: Sequence[str],
) -> dict[str, object]:
    """Construct a same-locus ordered sequence without aggregating rare loci."""
    marker_ids, marker_bp, marker_cm = list(marker_ids), list(marker_bp), [float(x) for x in marker_cm]
    locus_ids, locus_bp, locus_cm = list(locus_ids), list(locus_bp), [float(x) for x in locus_cm]
    if not marker_ids or not locus_ids or not (len(marker_ids) == len(marker_bp) == len(marker_cm)):
        raise ValueError("FLARE marker vectors have inconsistent lengths")
    if not (len(locus_ids) == len(locus_bp) == len(locus_cm) == len(minor_codes)):
        raise ValueError("rare-locus vectors have inconsistent lengths")
    if len(set(marker_ids)) != len(marker_ids) or len(set(locus_ids)) != len(locus_ids):
        raise ValueError("marker and locus identifiers must be unique")
    if any(right <= left for left, right in zip(marker_bp, marker_bp[1:])) or any(right <= left for left, right in zip(locus_bp, locus_bp[1:])):
        raise ValueError("bp coordinates must be unique and strictly increasing")
    if any(right < left for left, right in zip(marker_cm, marker_cm[1:])) or any(right < left for left, right in zip(locus_cm, locus_cm[1:])):
        raise ValueError("cM coordinates must be non-decreasing")
    if any(not math.isfinite(value) for value in marker_cm + locus_cm):
        raise ValueError("coordinates contain non-finite cM")

    primary = primary_diploid_channels(target_haplotypes, minor_codes)
    support, support_observed = reference_support(reference_haplotypes, minor_codes, reference_labels)
    posterior = [[list(row) for row in person] for person in flare_posteriors]
    if len(posterior) != len(primary["minor_dosage"]) or any(len(person) != len(marker_ids) for person in posterior):
        raise ValueError("FLARE posterior grid does not match targets and markers")
    for person in posterior:
        for row in person:
            if len(row) != len(ANCESTRIES) or any(value < 0 or not math.isfinite(value) for value in row) or not math.isclose(sum(row), 1.0, abs_tol=1e-9):
                raise ValueError("FLARE posterior rows must be finite probabilities summing to one")
    entropy = [
        [-sum(value * math.log(value) for value in row if value > 0) for row in person]
        for person in posterior
    ]
    return {
        "grid": {
            "marker_id": marker_ids,
            "bp": marker_bp,
            "cm": marker_cm,
            "delta_cm": [0.0] + [right - left for left, right in zip(marker_cm, marker_cm[1:])],
            "flare_posteriors": posterior,
            "flare_entropy": entropy,
        },
        "rare_sequence": {
            "locus_id": locus_ids,
            "bp": locus_bp,
            "cm": locus_cm,
            "delta_cm": [0.0] + [right - left for left, right in zip(locus_cm, locus_cm[1:])],
            "minor_code": list(minor_codes),
            **primary,
            "reference_support": support,
            "reference_support_observed_mask": support_observed,
        },
    }


def array_sha256(values: object) -> str:
    payload = json.dumps(values, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
