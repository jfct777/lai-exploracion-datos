#!/usr/bin/env python3
"""Bounded numerical kernels for the existing R02 estimands.

No data selection, labels, model family or resampling law is defined here.
The caller authenticates inputs and supplies every workspace budget in bytes.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

import r02_structure_increment as core


def target_pairs(genotypes, fit_rows, role_rows, components, min_fit_called,
                 min_pair_markers, max_pairs, *, max_workspace_bytes):
    """Exact masked standardized product, using bounded matrix row blocks.

    Missing cells contribute neither product nor denominator. Matrix products
    change summation order only; the reference is core.target_pairs.
    """
    require = core.require
    role_rows, fit_rows = np.asarray(role_rows), np.asarray(fit_rows)
    n = len(role_rows)
    require(n*(n-1)//2 <= max_pairs, 'Pair count exceeds declared allocation limit')
    require(type(max_workspace_bytes) is int and max_workspace_bytes > 0,
            'Positive target workspace-byte budget required')
    fit = genotypes[fit_rows]
    called = np.isfinite(fit).sum(axis=0)
    p = np.nansum(fit, axis=0)/np.maximum(2*called, 1)
    eligible = (called >= min_fit_called) & (p > 0) & (p < 1)
    require(eligible.any(), 'No eligible target markers with FIT frequency')
    del fit
    m = int(eligible.sum())
    # z, float64 callability, bool callability, frequency/row arrays and outputs.
    fixed = n*m*17 + len(fit_rows)*genotypes.shape[1]*8 + max_pairs*32 + 64*n + 64*m
    require(fixed + 32*n <= max_workspace_bytes, 'Target numerical workspace exceeds budget')
    block_rows = min(n, max(1, (max_workspace_bytes-fixed)//(32*n)))
    z = np.asarray(genotypes[np.ix_(role_rows, eligible)], dtype=np.float64)
    z -= 2*p[eligible]
    z /= np.sqrt(2*p[eligible]*(1-p[eligible]))
    mask = np.isfinite(z)
    z[~mask] = 0
    measured = mask.astype(np.float64)
    del mask
    component = np.asarray([components[int(i)] for i in role_rows])
    ii, jj, yy, nn = [], [], [], []
    excluded = Counter()
    for start in range(0, n, block_rows):
        stop = min(start+block_rows, n)
        products = z[start:stop] @ z.T
        denominators = measured[start:stop] @ measured.T
        for local, a in enumerate(range(start, stop)):
            candidates = np.arange(a+1, n)
            same = component[candidates] == component[a]
            excluded['within_component'] += int(same.sum())
            candidates = candidates[~same]
            counts = np.rint(denominators[local, candidates]).astype(np.int64)
            keep = counts >= min_pair_markers
            excluded['insufficient_joint_markers'] += int((~keep).sum())
            right = candidates[keep]
            if len(right):
                ii.append(np.full(len(right), role_rows[a], dtype=np.int64))
                jj.append(role_rows[right].astype(np.int64))
                nn.append(counts[keep])
                yy.append(products[local, right]/counts[keep])
    require(yy, 'No evaluable between-component pairs')
    return (np.concatenate(ii), np.concatenate(jj), np.concatenate(yy), np.concatenate(nn),
            dict(eligible_markers=m, excluded_pairs={k:v for k,v in excluded.items() if v}))


class WeightedRidgePath:
    """One FIT imputation/weighted scaling/SVD, many unchanged Ridge penalties.

    The SVD coefficient expression and 1e-15 singular-value floor reproduce
    sklearn 1.8's installed _solve_svd. The floor is numerical, not a rank
    hyperparameter. SELECT/EVALUATE never enter fitted transformations.
    """
    def __init__(self, x, y, weights, *, paired, max_workspace_bytes):
        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
        core.require(x.ndim == 2 and y.shape == weights.shape == (len(x),)
                     and np.isfinite(y).all() and np.isfinite(weights).all()
                     and (weights > 0).all(), 'Invalid weighted Ridge input')
        bound = 8*(8*x.size + 6*min(x.shape)**2 + 16*len(x))
        core.require(type(max_workspace_bytes) is int and bound <= max_workspace_bytes,
                     'Ridge path workspace exceeds explicit byte budget')
        self.transform = core.ridge_model(1.0, paired=paired)[:-1]
        self.transform.fit(x, y, scale__sample_weight=weights)
        z = self.transform.transform(x)
        self.x_offset = np.average(z, axis=0, weights=weights)
        self.y_offset = float(np.average(y, weights=weights))
        z = (z-self.x_offset)*np.sqrt(weights)[:, None]
        centered_y = (y-self.y_offset)*np.sqrt(weights)
        u, self.singular, self.vt = np.linalg.svd(z, full_matrices=False)
        self.uty = u.T @ centered_y
        self.n_features = x.shape[1]

    def coefficients(self, alpha):
        core.require(np.isfinite(alpha) and alpha > 0, 'Positive finite Ridge alpha required')
        factor = np.zeros_like(self.singular)
        keep = self.singular > 1e-15
        factor[keep] = self.singular[keep]/(self.singular[keep]**2+alpha)
        beta = self.vt.T @ (factor*self.uty)
        return beta, self.y_offset-self.x_offset @ beta

    def predict(self, x, alpha):
        beta, intercept = self.coefficients(alpha)
        return self.transform.transform(x) @ beta + intercept


def crossed_draw_means(errors, ca, cb, component_counts, chromosome_counts, *, max_batch_bytes):
    """Same component multiplicities and block means, bounded across draws.

    Counts are supplied, never regenerated. The byte budget limits three dense
    batch-by-pair arrays plus small products; it changes arithmetic batching,
    not the bootstrap population or number of replications.
    """
    errors, ca, cb = np.asarray(errors), np.asarray(ca), np.asarray(cb)
    n = len(errors)
    core.require(errors.shape == (n, 3) and ca.shape == cb.shape == (n,) and n > 0,
                 'Invalid grouped error dimensions')
    per_draw = 24*n + 128
    core.require(type(max_batch_bytes) is int and max_batch_bytes >= per_draw,
                 'Bootstrap batch budget cannot hold one draw')
    batch = max_batch_bytes//per_draw
    active = np.flatnonzero(chromosome_counts)
    numerator = np.zeros((len(chromosome_counts), 3))
    denominator = np.zeros(len(chromosome_counts))
    for start in range(0, len(active), batch):
        indices = active[start:start+batch]
        weight = component_counts[indices[:, None], ca]*component_counts[indices[:, None], cb]
        totals = weight.sum(axis=1)
        valid = totals > 0
        ids = indices[valid]
        numerator[ids] = (weight[valid] @ errors)/totals[valid, None]*chromosome_counts[ids, None]
        denominator[ids] = chromosome_counts[ids]
    return numerator, denominator
