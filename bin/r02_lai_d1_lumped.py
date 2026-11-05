"""Exact six-class reduction of two identical, independent D=1 ancestry chains.

The objective over correlated FIT mosaics is composite, not an independent
sample likelihood. Fitted rates are model parameters, not demographic estimates.
This module does not read genotypes, choose a neural architecture or train it.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

CLASSES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
OFFDIAGONALS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def generator(log_rates: Tensor) -> Tensor:
    if log_rates.shape != (6,) or log_rates.dtype != torch.float64:
        raise ValueError('six float64 log rates required')
    rates = log_rates.exp()
    if not bool(torch.isfinite(rates).all()) or not bool((rates > 0).all()):
        raise ValueError('nonfinite or boundary rates; no artificial floor')
    q = rates.new_zeros((3, 3))
    for index, (a, b) in enumerate(OFFDIAGONALS):
        q[a, b] = rates[index]
    return q - torch.diag(q.sum(1))


def lumped_generator(q: Tensor) -> Tensor:
    """Each of the two copies can change; equal destinations add multiplicity."""
    if q.shape != (3, 3):
        raise ValueError('D=1 generator must be 3x3')
    result = q.new_zeros((6, 6))
    for source, copies in enumerate(CLASSES):
        for copy in range(2):
            a, other = copies[copy], copies[1 - copy]
            for b in range(3):
                if a != b:
                    dest = CLASSES.index(tuple(sorted((b, other))))
                    result[source, dest] = result[source, dest] + q[a, b]
    return result - torch.diag(result.sum(1))


def transition_probabilities(log_rates: Tensor, deltas: Tensor) -> Tensor:
    if deltas.ndim != 1 or deltas.dtype != torch.float64:
        raise ValueError('float64 vector of distances in Morgans required')
    if not bool(torch.isfinite(deltas).all()) or bool((deltas < 0).any()):
        raise ValueError('invalid distances')
    p = torch.matrix_exp(deltas[:, None, None] * generator(log_rates))
    rows = []
    for a, b in CLASSES:
        columns = []
        for c, d in CLASSES:
            value = p[:, a, c] * p[:, b, d]
            if c != d:
                value = value + p[:, a, d] * p[:, b, c]
            columns.append(value)
        rows.append(torch.stack(columns, dim=-1))
    result = torch.stack(rows, dim=-2)
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError('invalid transition probability; no clipping')
    if not bool(torch.allclose(result.sum(-1), torch.ones_like(result[..., 0]),
                               atol=2e-12, rtol=2e-12)):
        raise ValueError('transition rows do not sum to one')
    return result


def composite_nll(log_rates: Tensor, deltas: Tensor, counts: Tensor) -> Tensor:
    """Unnormalised negative conditional transition log probability.

The frozen initial-state prior adds a constant, so is kept in receipts instead
of rate optimization. Zero counts contribute exactly zero, including at δ=0.
"""
    if counts.shape != (len(deltas), 6, 6) or counts.dtype != torch.int64:
        raise ValueError('integer joint transition counts required')
    if bool((counts < 0).any()) or int(counts.sum()) == 0:
        raise ValueError('invalid or empty counts')
    used = counts > 0
    probabilities = transition_probabilities(log_rates, deltas)
    if bool((probabilities[used] <= 0).any()):
        raise ValueError('observed transition impossible under supplied model')
    return -(counts[used] * probabilities[used].log()).sum()


def full_label_counts(positions_cm: np.ndarray, states: np.ndarray,
                      observed: np.ndarray) -> dict:
    """Count each interval once, keyed by exact canonical float64 subtraction.

Missing truth is deliberately blocked in this first admitted extraction. It
would require propagating the prior/gap probabilities, never a chain restart.
"""
    positions = np.asarray(positions_cm)
    labels, mask = np.asarray(states), np.asarray(observed)
    if positions.dtype != np.float64 or positions.ndim != 1 or len(positions) < 2:
        raise ValueError('complete float64 coordinate axis required')
    if labels.ndim != 2 or labels.shape[1] != len(positions) or labels.shape != mask.shape:
        raise ValueError('truth axes differ')
    if mask.dtype != np.bool_ or not bool(mask.all()):
        raise ValueError('missing truth requires explicit marginalization; blocked')
    if labels.dtype.kind not in 'iu' or not bool(np.isin(labels, np.arange(6)).all()):
        raise ValueError('invalid unordered class')
    delta_cm = np.diff(positions)
    if not bool(np.isfinite(positions).all()) or bool((delta_cm < 0).any()):
        raise ValueError('invalid map ordering')
    delta = delta_cm / np.float64(100)
    # Bit patterns, including near-equal nonidentical distances, never rounded.
    bits, inverse = np.unique(delta.view(np.uint64), return_inverse=True)
    counts = np.zeros((len(bits), 6, 6), dtype=np.int64)
    for person in labels:
        keys = inverse * 36 + person[:-1].astype(np.int64) * 6 + person[1:]
        counts += np.bincount(keys, minlength=len(bits) * 36).reshape(-1, 6, 6)
    if bool(np.any((delta == 0)[None, :] & (labels[:, :-1] != labels[:, 1:]))):
        raise ValueError('changing truth at zero map distance is impossible for D1')
    cm_bits, cm_inverse = np.unique(delta_cm.view(np.uint64), return_inverse=True)
    cm_to_m = {format(int(b), '016x'): format(int(delta.view(np.uint64)[
        np.flatnonzero(cm_inverse == i)[0]]), '016x') for i, b in enumerate(cm_bits)}
    return dict(delta_morgan_bits=[format(int(x), '016x') for x in bits],
                delta_morgan=bits.view(np.float64).tolist(), counts=counts.tolist(),
                delta_cm_to_morgan_bits=cm_to_m,
                initial_counts=np.bincount(labels[:, 0], minlength=6).tolist(),
                intervals=int(counts.sum()), changes=int((labels[:, 1:] != labels[:, :-1]).sum()),
                missing_labels=0, people=int(len(labels)), case_person_rows=int(len(labels)), positions=int(len(positions)),
                summed_interval_morgans=float(delta.sum() * len(labels)))


def combine_counts(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError('no cases')
    keys = sorted({key for case in cases for key in case['delta_morgan_bits']})
    index = {key: i for i, key in enumerate(keys)}
    counts = np.zeros((len(keys), 6, 6), dtype=np.int64)
    for case in cases:
        if sum(case['initial_counts']) != case['people']:
            raise ValueError('initial counts differ from case person rows')
        for key, count in zip(case['delta_morgan_bits'], case['counts']):
            counts[index[key]] += np.asarray(count, dtype=np.int64)
    if int(counts.sum()) != sum(c['intervals'] for c in cases):
        raise ValueError('counts differ from interval total')
    return dict(delta_morgan_bits=keys,
                delta_morgan=np.array([int(k, 16) for k in keys], dtype=np.uint64).view(np.float64).tolist(),
                counts=counts.tolist(),
                initial_counts=np.sum([c['initial_counts'] for c in cases], axis=0).tolist(),
                intervals=sum(c['intervals'] for c in cases),
                changes=sum(c['changes'] for c in cases), missing_labels=0,
                people=sum(c['people'] for c in cases),
                case_person_rows=sum(c['people'] for c in cases),
                case_map_bits=[c['delta_cm_to_morgan_bits'] for c in cases],
                summed_interval_morgans=sum(c['summed_interval_morgans'] for c in cases))


def exact_delta_tensor(count_data: dict) -> Tensor:
    bits = np.array([int(k, 16) for k in count_data['delta_morgan_bits']], dtype=np.uint64)
    values = bits.view(np.float64)
    if (not len(values) or not np.isfinite(values).all() or (values < 0).any()
            or not np.all(np.diff(values) > 0) or len(values) != len(count_data['counts'])):
        raise ValueError('invalid exact distance keys')
    declared = np.asarray(count_data['delta_morgan'],dtype=np.float64)
    if not np.array_equal(declared.view(np.uint64),bits):
        raise ValueError('human readable distances differ from exact keys')
    return torch.from_numpy(values.copy())
