"""Stratified FIT block draws with an unchanged equal-person/query objective.

This prepares a new weighted plan, not an executable v1 donor-training plan.
Truth selects FIT anchors only; it must never enter model inputs. Repeated
cells retain multiplicity. Context halos remain the reader's responsibility.
"""
from __future__ import annotations

import numpy as np


SCHEMA = "r02_lai_fit_stratified_block_plan_v1"
STATES = ("AA", "AE", "AN", "EE", "EN", "NN")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def allocate_blocks(state_counts, blocks_per_update):
    """One per present state, then proportional largest remainders; ties by state."""
    counts = np.asarray(state_counts)
    _require(counts.shape == (6,) and np.issubdtype(counts.dtype, np.integer)
             and np.all(counts >= 0) and counts.sum() > 0, "six nonnegative counts required")
    active = counts > 0
    _require(type(blocks_per_update) is int and blocks_per_update >= active.sum(),
             "budget must cover every present state")
    quota = active.astype(np.int64)
    remaining = blocks_per_update - int(quota.sum())
    # Integer arithmetic makes largest-remainder allocation deterministic.
    numerators = counts * remaining
    quota += numerators // counts.sum()
    remainder = numerators % counts.sum()
    order = sorted(np.flatnonzero(active), key=lambda s: (-int(remainder[s]), int(s)))
    for state in order[:blocks_per_update - int(quota.sum())]:
        quota[state] += 1
    return quota


def make_fit_plan(truth, observed, *, role, seed, updates, blocks_per_update, width):
    """Return occurrence rows (update, block, person, query, anchor, coefficient).

    With complete rows, each cell belongs to exactly width circular blocks.
    An anchor in state s gives every occurrence in its block coefficient
    N_s/(P*Q*n_s*width), where n_s blocks are drawn from N_s anchors.
    The expected loss is the full-FIT mean for fixed predictions. Do not divide by
    persons, blocks, or the realized sum of coefficients. Missing/unequal
    eligible query counts require a separately reviewed extension.
    """
    _require(role == "FIT", "reserved labels cannot generate this plan")
    truth, observed = np.asarray(truth), np.asarray(observed)
    _require(observed.dtype == np.bool_ and observed.shape == truth.shape and observed.all(),
             "only complete equal-query FIT rows are admitted by this version")
    _require(type(seed) is int and seed >= 0 and type(updates) is int and updates > 0,
             "explicit nonnegative seed and positive update count required")
    _require(truth.ndim == 2 and truth.size > 0 and
             np.issubdtype(truth.dtype, np.integer) and np.all((truth >= 0) & (truth < 6)),
             "invalid FIT states")
    counts = np.bincount(truth.ravel(), minlength=6)
    quota = allocate_blocks(counts, blocks_per_update)
    _require(type(width) is int and 1 <= width <= truth.shape[1], "invalid block width")
    cells = [np.flatnonzero(truth.ravel() == state) for state in range(6)]
    rng = np.random.default_rng(seed)
    rows, coefficients = [], []
    for update in range(updates):
        anchors = np.concatenate([rng.choice(cells[s], int(quota[s]), replace=True)
                                  for s in range(6) if quota[s]])
        rng.shuffle(anchors)
        for block, anchor in enumerate(anchors):
            person, start = divmod(int(anchor), truth.shape[1])
            state = truth[person, start]
            coefficient = counts[state] / (truth.size * quota[state] * width)
            for query in (start + np.arange(width)) % truth.shape[1]:
                rows.append((update, block, person, int(query), start))
                coefficients.append(coefficient)
    return (np.asarray(rows, dtype=np.int64), np.asarray(coefficients, dtype=np.float64),
            {"schema": SCHEMA, "state_order": list(STATES), "role": role,
             "seed": seed, "updates": updates, "blocks_per_update": blocks_per_update,
             "input_shape": list(truth.shape),
             "width_queries": width, "state_counts": counts.tolist(),
             "anchor_quota_per_update": quota.tolist(),
             "state_mass": (counts / truth.size).tolist(),
             "anchor_probability_within_state": np.divide(1., counts,
                 out=np.zeros(6), where=counts > 0).tolist(),
             "coefficient_uses": "ANCHOR_STATE_NOT_QUERY_STATE",
             "duplicate_rule": "KEEP_OCCURRENCES_OR_SUM_THEIR_COEFFICIENTS_PER_UPDATE_PERSON_QUERY",
             "atomic_update": "ADAM_AND_CLIPPING_ONLY_AFTER_ALL_WEIGHTED_OCCURRENCES_COMPLETE",
             "target": "EQUAL_PERSON_THEN_EQUAL_QUERY_COMPLETE_RECTANGLE",
             "loss": "SUM_OCCURRENCE_COEFFICIENT_TIMES_NLL_NO_RENORMALIZATION",
             "coefficient_sum_per_update": 1.0,
             "coefficient": "ANCHOR_STATE_COUNT/(TOTAL_CELLS*ANCHOR_QUOTA*BLOCK_WIDTH)",
             "training_admitted": False,
             "required_training_schema": "EXPLICIT_OCCURRENCE_WEIGHTED_NOT_DONOR_V1"})
