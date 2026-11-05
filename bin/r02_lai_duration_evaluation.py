"""Descriptive FIT scores for a fixed complete chromosome, before/after training.

Reuse M39 proper scores and M37 boundary matching. Neither a decrease in FIT
loss nor a one-step comparison with FLARE certifies independent performance.
The zero-emission control retains the same initial prior and D1 transitions.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from m39_calibration import DOSAGES, evaluate, validate_probabilities
from r02_lai_boundary_stats import boundary_report_stats as boundary_report
from r02_lai_duration_model import duration_marginals


def baseline_from_provider(provider, index, batch):
    """Join authenticated cached FLARE queries in the frozen provider's order.

    This adapter deliberately binds the provider's internal cache API; callers
    must authenticate its source hash. It never reloads genotypes or imputes
    missing probabilities, and checks identities against the assembled batch.
    """
    case, start, stop = provider._chunk_lookup[index]
    runtime = provider._cases[case]
    if runtime.cache is None or stop - start != 1 or len(batch.person_ids) != 1:
        raise ValueError("one complete cached person required for descriptive scoring")
    indices, probabilities = [], []
    for _, data in runtime.cache:
        core = np.asarray(data['query_core'][start], dtype=bool)
        idx = np.asarray(data['query_global_index'])
        if idx.ndim == 2:
            idx = idx[start]
        indices.append(idx[core])
        probabilities.append(np.asarray(data['baseline_probabilities'][start])[core])
    idx = np.concatenate(indices)
    order = np.argsort(idx, kind='stable')
    expected = batch.query_global_index.detach().cpu().numpy()
    if len(np.unique(idx)) != len(idx) or not np.array_equal(idx[order], expected):
        raise ValueError("cached FLARE and neural query identities differ")
    return validate_probabilities(np.concatenate(probabilities)[order][None])[0]


def score_probabilities(probabilities, labels, mask, cm, *, floor,
                        calibration_bins, boundary_memory_bytes):
    """One person: identical observed queries, raw-loss flags, dosage and borders."""
    p = validate_probabilities(np.asarray(probabilities, dtype=np.float64)[None])[0]
    y, observed, cm = np.asarray(labels), np.asarray(mask), np.asarray(cm)
    if (observed.dtype != np.bool_ or y.shape != observed.shape or
            cm.shape != observed.shape or len(p) != len(y) or not observed.any()
            or not np.isfinite(cm).all() or not (np.diff(cm) > 0).all()):
        raise ValueError("complete ordered query geometry and observed labels required")
    if type(calibration_bins) is not int or calibration_bins < 2:
        raise ValueError("explicit calibration bin count required")
    values = evaluate(p[observed][None], y[observed][None], floor=floor)
    values = {k: v for k, v in values.items() if not k.endswith('_per_person')}
    values['boundaries'] = boundary_report(y, p.argmax(-1), observed, cm,
                                          [('chromosome', 0, len(cm))],
                                          boundary_memory_bytes)
    predicted, actual = p[observed] @ DOSAGES / 2, DOSAGES[y[observed]] / 2
    calibration = {}
    for a, name in enumerate(('AFR', 'EUR', 'NAM')):
        bins = np.minimum((predicted[:, a] * calibration_bins).astype(int),
                          calibration_bins - 1)
        calibration[name] = [dict(
            left=b / calibration_bins, right=(b + 1) / calibration_bins,
            count=int((bins == b).sum()),
            mean_predicted=float(predicted[bins == b, a].mean()) if (bins == b).any() else None,
            mean_observed=float(actual[bins == b, a].mean()) if (bins == b).any() else None)
            for b in range(calibration_bins)]
    values['ancestry_dosage_fraction_calibration'] = calibration
    values['scope'] = 'ONE_FIT_PERSON_DESCRIPTIVE_NOT_INDEPENDENT_REPLICATES'
    return values


class DurationEvaluation:
    """Score neural evidence separately from the same fixed continuity model."""

    def __init__(self, *, floor, calibration_bins, boundary_memory_bytes):
        self.options = dict(floor=floor, calibration_bins=calibration_bins,
                            boundary_memory_bytes=boundary_memory_bytes)
        self._zero = {}

    def __call__(self, chunk, state, baseline_probability):
        batch, d = chunk.batch, state.duration
        if len(batch.roles) != 1 or batch.roles != ('FIT',):
            raise ValueError('exactly one FIT person required')
        if any(p.requires_grad for p in d.parameters()):
            raise ValueError('this comparison requires fixed duration parameters')
        logits = batch.neural_logits.detach()
        kwargs = dict(query_cm=batch.query_cm, query_chromosome=batch.query_chromosome,
                      query_valid=batch.query_valid, log_stage_rates=d.log_stage_rates,
                      switch_logits=d.switch_logits, joint_start_prior=d.joint_start_prior(1),
                      sequence_declarations=batch.sequence_declarations,
                      parameter_declaration=d.core_declaration(), duration_stages=d.duration_stages)
        tensors = [batch.query_cm, batch.query_chromosome, batch.query_valid,
                   d.log_stage_rates, d.switch_logits, d.joint_start_prior(1)]
        key = tuple(hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for v in tensors)
        with torch.no_grad():
            neural = duration_marginals(logits, **kwargs).unordered_marginals[0].cpu().numpy()
            if key not in self._zero:
                self._zero[key] = duration_marginals(torch.zeros_like(logits), **kwargs).unordered_marginals[0].cpu().numpy()
            predictions = dict(NEURAL_RAW=logits.softmax(-1)[0].cpu().numpy(),
                               NEURAL_D1=neural, FLARE_RAW=baseline_probability,
                               ZERO_EMISSION_D1=self._zero[key])
        labels = batch.labels[0].detach().cpu().numpy()
        mask = batch.label_mask[0].detach().cpu().numpy()
        cm = batch.query_cm[0].detach().cpu().numpy()
        return {name: score_probabilities(p, labels, mask, cm, **self.options)
                for name, p in predictions.items()}
