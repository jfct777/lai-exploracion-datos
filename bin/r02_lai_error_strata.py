"""Paired descriptive errors at truth boundaries and existing rare exposure.

FIT diagnosis only: conditioning on known errors cannot define a new test set.
All arrays belong to one complete chromosome and one person, ordered identically.
"""
import numpy as np
from m39_calibration import DOSAGES, validate_probabilities


def truth_boundary_distance(labels, observed, cm):
    """Nearest observed diploid-state change, without bridging missing intervals."""
    y, mask, pos = np.asarray(labels), np.asarray(observed), np.asarray(cm)
    if (y.ndim != 1 or mask.dtype != np.bool_ or y.shape != mask.shape
            or pos.shape != y.shape or not np.isfinite(pos).all()
            or not (np.diff(pos) > 0).all()):
        raise ValueError('ordered chromosome and boolean observation mask required')
    if ((y[mask] < 0) | (y[mask] >= 6)).any():
        raise ValueError('six diploid ancestry states required')
    result = np.full(len(y), np.inf)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for lo, hi in zip(edges[::2], edges[1::2]):
        changes = pos[lo + 1:hi][y[lo + 1:hi] != y[lo:hi - 1]]
        if not len(changes):
            continue
        idx = np.searchsorted(changes, pos[lo:hi])
        a = changes[np.maximum(idx - 1, 0)]
        b = changes[np.minimum(idx, len(changes) - 1)]
        result[lo:hi] = np.minimum(abs(pos[lo:hi] - a), abs(pos[lo:hi] - b))
    return result


def paired_strata(flare, neural, labels, observed, cm, burden, *, cuts_cm, floor):
    """Return additive sums; exact query partition and corrections AND new errors."""
    f = validate_probabilities(np.asarray(flare)[None])[0]
    n = validate_probabilities(np.asarray(neural)[None])[0]
    y, mask, counts = np.asarray(labels), np.asarray(observed), np.asarray(burden)
    d = truth_boundary_distance(y, mask, cm)
    cuts = np.asarray(cuts_cm, dtype=float)
    if (f.shape != n.shape or len(f) != len(y) or counts.shape != (len(y), 3)
            or not np.isfinite(counts).all() or (counts < 0).any()
            or not np.equal(counts, np.floor(counts)).all()
            or (counts[:, 0] > counts[:, 2]).any()
            or (counts[:, 1] < counts[:, 0]).any()
            or (counts[:, 1] > 2 * counts[:, 0]).any()
            or cuts.ndim != 1 or not len(cuts) or not np.isfinite(cuts).all()
            or not (cuts > 0).all() or not (np.diff(cuts) > 0).all()
            or not 0 < floor < 1):
        raise ValueError('probability axes, burden counts, cuts or floor differ')
    distance = np.searchsorted(cuts, d, side='left')
    distance[~np.isfinite(d)] = len(cuts) + 1
    exposure = np.where(counts[:, 2] == 0, 0, np.where(counts[:, 0] == 0, 1, 2))
    yi = y.astype(np.int64)
    # Missing labels are never indexed, even if a caller encodes them as -1/255.
    rows = []
    for a in range(len(cuts) + 2):
        for b in range(3):
            take = mask & (distance == a) & (exposure == b)
            labels_a, fp, np_ = yi[take], f[take], n[take]
            size = len(labels_a)
            fe, ne = fp.argmax(-1) != labels_a, np_.argmax(-1) != labels_a
            row = dict(distance_bin=a, exposure_bin=b, count=size,
                       flare_errors=int(fe.sum()), neural_errors=int(ne.sum()),
                       corrected=int((fe & ~ne).sum()), introduced=int((~fe & ne).sum()),
                       both_wrong=int((fe & ne).sum()), both_correct=int((~fe & ~ne).sum()),
                       truth_state_counts=np.bincount(labels_a,minlength=6).tolist(),
                       carrier_sites_sum=int(counts[take,0].sum()),
                       minor_copies_sum=int(counts[take,1].sum()),
                       observed_sites_sum=int(counts[take,2].sum()))
            for name, p in [('FLARE', fp), ('NEURAL', np_)]:
                truep = p[np.arange(size), labels_a]
                target = np.eye(6)[labels_a]
                row[name] = dict(logloss_sum=float(-np.log(np.maximum(truep, floor)).sum()),
                    true_zero_count=int((truep == 0).sum()),
                    brier_sum=float(((p - target)**2).sum()),
                    dosage_abs_sum=np.abs(p @ DOSAGES - DOSAGES[labels_a]).sum(0).tolist(),
                    argmax_ties=int(((p == p.max(-1,keepdims=True)).sum(-1)>1).sum()),
                    predicted_state_counts=np.bincount(p.argmax(-1),minlength=6).tolist())
            rows.append(row)
    if sum(r['count'] for r in rows) != int(mask.sum()):
        raise ValueError('strata must partition the observed queries exactly')
    return rows
