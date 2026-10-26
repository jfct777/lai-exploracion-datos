"""Versioned LAI query convolutions; no genotype I/O or scientific defaults.

Kernels count query positions, not SNPs or centimorgans. A frozen regular query
step is required for the new multiscale policy. COMMON/EVENT radii remain
separate inputs, and multiple layers add their maximum query half-widths.
"""
from __future__ import annotations

import math

SCHEMA = "lai_query_multiscale_cnn_v1"


def validate_cnn_config(value):
    if value is None:
        return None
    base = {"schema", "kernel_sizes"}
    if not isinstance(value, dict) or set(value) not in (
            base | {"query_step_cm"}, base | {"query_step_cm_by_chromosome"}):
        raise ValueError("explicit CNN schema, kernels and one query-step specification required")
    kernels = value["kernel_sizes"]
    if (value["schema"] != SCHEMA or not isinstance(kernels, list) or len(kernels) < 2
            or any(type(k) is not int or k < 3 or k % 2 != 1 for k in kernels)
            or kernels != sorted(set(kernels))):
        raise ValueError("multiscale kernels must be sorted, distinct odd integers >= 3")
    steps = value.get("query_step_cm_by_chromosome")
    if steps is not None:
        if (not isinstance(steps, dict) or not steps
                or any(k not in {str(i) for i in range(1, 23)} for k in steps)):
            raise ValueError("query steps require explicit autosome keys")
    else:
        steps = {"uniform": value.get("query_step_cm")}
    if any(type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in steps.values()):
        raise ValueError("query steps must be finite and positive")
    result = {"schema": SCHEMA, "kernel_sizes": list(kernels)}
    if "query_step_cm" in value:
        result["query_step_cm"] = float(value["query_step_cm"])
    else:
        result["query_step_cm_by_chromosome"] = {k: float(v) for k, v in sorted(steps.items())}
    return result


def chromosome_step(config, chromosome):
    """Use an authenticated full-axis step, never infer it from a region."""
    if "query_step_cm" in config:
        return config["query_step_cm"]
    try:
        return config["query_step_cm_by_chromosome"][str(int(chromosome))]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("CNN query-step map does not cover this chromosome") from exc


def maximum_step(config):
    return (config["query_step_cm"] if "query_step_cm" in config else
            max(config["query_step_cm_by_chromosome"].values()))


def query_halo(depth, config=None):
    """Exact maximum query-index halo of sequential, parallel-kernel layers."""
    if type(depth) is not int or depth < 0:
        raise ValueError("nonnegative integer CNN depth required")
    config = validate_cnn_config(config)
    return depth * (1 if config is None else (max(config["kernel_sizes"]) - 1) // 2)


def validate_grid_block(coordinates, step, epsilon):
    """Reject compressed/irregular grids; roundoff is coordinate-scale aware."""
    for left, right in zip(coordinates, coordinates[1:]):
        left, right = float(left), float(right)
        tolerance = 2 * epsilon * max(abs(left), abs(right), step)
        if not math.isfinite(left + right) or abs((right - left) - step) > tolerance:
            raise ValueError("multiscale CNN requires the frozen regular query grid")
