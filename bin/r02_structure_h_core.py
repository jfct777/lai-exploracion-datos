#!/usr/bin/env python3
"""Small dependency-free contracts for the R02 H-chain producer.

The numerical producer imports NumPy/SciPy and the established M14 detector.
This module keeps request validation and indexing independently testable in a
minimal controller environment.  It does not choose a geometry or transform.
"""
from __future__ import annotations

import math
import re


GEOMETRY_FIELDS = {
    "config_id", "max_gap_bp", "min_length_bp", "min_shared_effective"
}
JUSTIFICATION_FIELDS = {
    "question", "mechanism", "source", "applicability", "selection_rule",
    "sensitivity"
}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_justification(value, label):
    require(isinstance(value, dict) and set(value) == JUSTIFICATION_FIELDS,
            f"{label} requires the exact scientific justification fields")
    require(all(isinstance(value[k], str) and value[k].strip()
                for k in JUSTIFICATION_FIELDS),
            f"{label} justification fields must be non-empty")
    return value


def validate_geometry(value):
    require(isinstance(value, dict) and set(value) == GEOMETRY_FIELDS,
            "Exact H geometry fields required")
    require(isinstance(value["config_id"], str) and
            SAFE_ID.fullmatch(value["config_id"]) is not None,
            "Safe H geometry ID required")
    for field in ("max_gap_bp", "min_length_bp", "min_shared_effective"):
        require(type(value[field]) is int and value[field] > 0,
                f"Positive integer {field} required")
    return value


def triangle_size(n):
    require(type(n) is int and n >= 0, "Non-negative roster size required")
    return n * (n - 1) // 2


def triangle_index(left, right, n):
    """Index a canonical off-diagonal pair in np.triu_indices(n, 1) order."""
    require(all(type(v) is int for v in (left, right, n)) and
            0 <= left < right < n, "Canonical triangular pair required")
    return left * (2 * n - left - 1) // 2 + right - left - 1


def log1p_total_bp(total_shared_bp, scale_bp):
    require(type(total_shared_bp) is int and total_shared_bp >= 0,
            "Non-negative integer H length required")
    require(type(scale_bp) is int and scale_bp > 0,
            "Positive integer H transform scale required")
    return math.log1p(total_shared_bp / scale_bp)


def validate_allowed_intervals(intervals, chromosome_length):
    require(type(chromosome_length) is int and chromosome_length > 0 and
            isinstance(intervals, list), "Explicit chromosome intervals required")
    previous = 0
    for interval in intervals:
        require(isinstance(interval, list) and len(interval) == 2 and
                all(type(v) is int for v in interval),
                "Each allowed interval must contain two integers")
        start, end = interval
        require(previous < start <= end <= chromosome_length,
                "Allowed intervals overlap, touch out of order, or exceed chromosome")
        previous = end
    return intervals


def catalogue_geometry(row):
    """Project one admission-catalogue row into the producer geometry."""
    return validate_geometry({
        "config_id": row["config_id"],
        "max_gap_bp": row["max_gap_bp"],
        "min_length_bp": row["min_length_bp"],
        "min_shared_effective": row["min_shared_effective"],
    })
