#!/usr/bin/env python3
"""Phase-free TRACE-LAI primitives.

The module deliberately models unordered diploid ancestry states.  A rare
heterozygote is never assigned to a TARGET haplotype: its likelihood is
marginalised over AA, AE, AN, EE, EN and NN.  Reference allele frequencies are
accepted only as FIT-only fold summaries; a caller cannot accidentally label an
aggregate reference panel as cross-fitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


STATE_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
MISSING_GENOTYPE = 3
STATE_INDEX = np.array(((0, 1, 2), (1, 3, 4), (2, 4, 5)), dtype=np.uint8)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def diploid_state_names(ancestry_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in ancestry_names)
    require(len(names) == 3 and len(set(names)) == 3, "TRACE requires AFR/EUR/NAM axes")
    return tuple(names[left] + names[right] for left, right in STATE_PAIRS)


def baseline_to_states(probabilities: np.ndarray) -> np.ndarray:
    """Convert phased baseline probabilities [N,2,M,3] to unordered states."""
    value = np.asarray(probabilities, dtype=np.float64)
    require(value.ndim == 4 and value.shape[1] == 2 and value.shape[3] == 3,
            "baseline must have shape [sample, haplotype, marker, ancestry=3]")
    require(np.isfinite(value).all() and np.all(value >= 0) and
            np.allclose(value.sum(axis=3), 1.0, atol=5e-6, rtol=0),
            "baseline probability simplex differs")
    left, right = value[:, 0], value[:, 1]
    result = np.empty((value.shape[0], value.shape[2], len(STATE_PAIRS)), dtype=np.float32)
    for index, (first, second) in enumerate(STATE_PAIRS):
        result[:, :, index] = (left[:, :, first] * right[:, :, second] if first == second else
                               left[:, :, first] * right[:, :, second] +
                               left[:, :, second] * right[:, :, first])
    require(np.allclose(result.sum(axis=2), 1.0, atol=5e-6, rtol=0),
            "derived diploid state simplex differs")
    return result


def m34_labels_to_states(labels: np.ndarray) -> np.ndarray:
    """Marginalise M34's [N,2,M] labels into unordered diploid states.

    Haplotype order is only used for a symmetric lookup; it cannot assign the
    rare allele to a particular TARGET haplotype.
    """
    value = np.asarray(labels)
    require(value.ndim == 3 and value.shape[1] == 2 and np.issubdtype(value.dtype, np.integer) and
            np.all((value >= 0) & (value < 3)), "M34 truth must be [N,2,M] with AFR/EUR/NAM labels")
    return np.ascontiguousarray(STATE_INDEX[value[:, 0], value[:, 1]])


def _beta_moments(ac: np.ndarray, an: np.ndarray, prior_strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Return E[p] and E[p²] for Beta(ac+a, an-ac+a), with symmetric shrinkage."""
    ac, an = np.asarray(ac, dtype=np.float64), np.asarray(an, dtype=np.float64)
    require(ac.shape == an.shape and np.all(ac >= 0) and np.all(an >= ac),
            "reference AC/AN differs")
    require(prior_strength > 0, "beta prior strength must be positive")
    alpha, beta = ac + prior_strength, an - ac + prior_strength
    total = alpha + beta
    mean = alpha / total
    second = alpha * (alpha + 1.0) / (total * (total + 1.0))
    return mean, second


def _state_likelihood_from_moments(genotype: np.ndarray, mean: np.ndarray,
                                   second: np.ndarray) -> np.ndarray:
    """Return [sample,locus,state] likelihood without assigning rare phase."""
    samples, loci = genotype.shape
    result = np.empty((samples, loci, 6), dtype=np.float64)
    for state_index, (left, right) in enumerate(STATE_PAIRS):
        if left == right:
            p1, p2 = mean[left], second[left]
            likelihood = np.stack((1.0 - 2.0 * p1 + p2, 2.0 * (p1 - p2), p2), axis=1)
        else:
            p, q = mean[left], mean[right]
            likelihood = np.stack(((1.0 - p) * (1.0 - q),
                                   p * (1.0 - q) + (1.0 - p) * q, p * q), axis=1)
        result[:, :, state_index] = likelihood[np.arange(loci)[None, :], np.minimum(genotype, 2)]
    return np.where((genotype != MISSING_GENOTYPE)[:, :, None], result, 1.0)


def reference_state_log_likelihood(genotype: np.ndarray, reference_ac: np.ndarray, reference_an: np.ndarray,
                                   prior_strength: float = 0.5
                                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use REF_TRAIN counts; use folds only when the upstream artifact has them.

    TARGET mosaics are biologically disjoint from REF_TRAIN.  Consequently a
    target-fold assignment would be artificial and is forbidden.  Folds form an
    ensemble solely to quantify reference-frequency instability.
    """
    genotype = np.asarray(genotype, dtype=np.uint8)
    reference_ac, reference_an = np.asarray(reference_ac), np.asarray(reference_an)
    require(genotype.ndim == 2 and reference_ac.shape == reference_an.shape and
            reference_ac.ndim in (2, 3) and reference_ac.shape[-2] == 3 and
            genotype.shape[1] == reference_ac.shape[-1], "TRACE reference count axes differ")
    if reference_ac.ndim == 3:
        require(len(reference_ac) >= 2, "folded reference needs at least two REF_TRAIN folds")
        ac, an = reference_ac.sum(axis=0), reference_an.sum(axis=0)
        fold_means = np.stack([_beta_moments(reference_ac[index], reference_an[index], prior_strength)[0]
                               for index in range(len(reference_ac))])
        ensemble_variance = fold_means.var(axis=0)
    else:
        # M34's reference summary is aggregate [ancestry,locus].  It supports
        # posterior Beta uncertainty but does not identify fold instability.
        ac, an = reference_ac, reference_an
        ensemble_variance = np.zeros_like(ac, dtype=np.float64)
    mean, second = _beta_moments(ac, an, prior_strength)
    state_likelihood = _state_likelihood_from_moments(genotype, mean, second)
    # POOLED recomputes the genotype likelihood from grouped AC/AN.  It is not
    # an AA proxy: all six state columns receive the same pooled likelihood.
    pooled_mean, pooled_second = _beta_moments(ac.sum(axis=0), an.sum(axis=0), prior_strength)
    pooled_by_genotype = np.stack((1.0 - 2.0 * pooled_mean + pooled_second,
                                   2.0 * (pooled_mean - pooled_second), pooled_second), axis=1)
    pooled = pooled_by_genotype[np.arange(genotype.shape[1])[None, :], np.minimum(genotype, 2)]
    pooled = np.where((genotype != MISSING_GENOTYPE)[:, :, None], pooled[:, :, None], 1.0)
    pooled = np.repeat(pooled, 6, axis=2)
    posterior_variance = np.maximum(second - mean * mean, 0.0)
    uncertainty = (posterior_variance + ensemble_variance).transpose(1, 0)[None, :, :]
    # Global denominator: per-ancestry normalization would encode a moving
    # reference scale rather than genuine callable support.
    support = (an / (an.max() + 1e-12)).transpose(1, 0)[None, :, :]
    samples = genotype.shape[0]
    return (np.log(np.maximum(state_likelihood, 1e-12)).astype(np.float32),
            np.log(np.maximum(pooled, 1e-12)).astype(np.float32),
            np.broadcast_to(uncertainty, (samples, *uncertainty.shape[1:])).astype(np.float32),
            np.broadcast_to(support, (samples, *support.shape[1:])).astype(np.float32))


def nearest_marker_indices(locus_cm: np.ndarray, marker_cm: np.ndarray) -> np.ndarray:
    locus_cm, marker_cm = np.asarray(locus_cm), np.asarray(marker_cm)
    require(locus_cm.ndim == marker_cm.ndim == 1 and len(marker_cm) > 0 and
            np.all(np.diff(marker_cm) >= 0), "genetic axes differ")
    right = np.searchsorted(marker_cm, locus_cm, side="left")
    right = np.clip(right, 0, len(marker_cm) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(marker_cm[right] - locus_cm) < np.abs(locus_cm - marker_cm[left]), right, left)


def deposit_evidence(loglik: np.ndarray, pooled_loglik: np.ndarray, genotype: np.ndarray, locus_cm: np.ndarray,
                     marker_cm: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate all genotypes at their nearest common-marker anchor.

    Carrier/missing events remain ragged; genotype-zero likelihood is included
    in the dense field so absence is not silently discarded.
    """
    require(arm in {"RE", "RD", "POOLED", "SHAM", "GEOMETRY"}, "unknown TRACE arm")
    loglik, genotype = np.asarray(loglik, dtype=np.float32), np.asarray(genotype, dtype=np.uint8)
    require(loglik.shape[:2] == genotype.shape and loglik.shape[2] == 6, "TRACE likelihood axes differ")
    samples, loci = genotype.shape
    anchors = nearest_marker_indices(locus_cm, marker_cm)
    field = np.zeros((samples, len(marker_cm), 6), dtype=np.float32)
    counts = np.zeros((samples, len(marker_cm), 1), dtype=np.float32)
    for locus, marker in enumerate(anchors.tolist()):
        observed = genotype[:, locus] != MISSING_GENOTYPE
        if arm == "RD":
            value = np.zeros((samples, 6), dtype=np.float32)
        elif arm == "POOLED":
            value = pooled_loglik[:, locus]
        elif arm == "GEOMETRY":
            value = np.zeros((samples, 6), dtype=np.float32)
        else:
            value = loglik[:, locus]
        field[:, marker] += value * observed[:, None]
        counts[:, marker, 0] += observed
    event_mask = (genotype != 0) | (genotype == MISSING_GENOTYPE)
    event_rows, event_loci = np.nonzero(event_mask)
    return field, counts, np.stack((event_rows, event_loci), axis=1).astype(np.int64)


def transition_matrix(delta_cm: float, hazard_per_morgan: float) -> np.ndarray:
    require(delta_cm >= 0 and hazard_per_morgan > 0, "invalid transition geometry")
    change = 1.0 - math.exp(-hazard_per_morgan * delta_cm / 100.0)
    # Each physical haplotype follows a three-ancestry chain.  Marginalising
    # its two ordered copies prevents one breakpoint from jumping AA -> EE as
    # readily as AA -> AE, while retaining TRACE's phase-free six states.
    haploid = np.full((3, 3), change / 2.0, dtype=np.float64)
    np.fill_diagonal(haploid, 1.0 - change)
    matrix = np.empty((6, 6), dtype=np.float64)
    for source, (left, right) in enumerate(STATE_PAIRS):
        for target, (next_left, next_right) in enumerate(STATE_PAIRS):
            matrix[source, target] = (haploid[left, next_left] * haploid[right, next_right]
                                      if next_left == next_right else
                                      haploid[left, next_left] * haploid[right, next_right] +
                                      haploid[left, next_right] * haploid[right, next_left])
    require(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0), "diploid transition differs")
    return matrix


def hmm_posterior(baseline_states: np.ndarray, evidence: np.ndarray, marker_cm: np.ndarray,
                  hazard_per_morgan: float = 12.0, evidence_scale: float = 1.0) -> np.ndarray:
    """Vectorized forward-backward posterior for phase-free chromosomes."""
    baseline, evidence, marker_cm = (np.asarray(baseline_states, dtype=np.float32),
                                     np.asarray(evidence, dtype=np.float32), np.asarray(marker_cm, dtype=np.float64))
    require(baseline.shape == evidence.shape and baseline.ndim == 3 and baseline.shape[2] == 6 and
            marker_cm.shape == (baseline.shape[1],), "HMM axes differ")
    emission = np.log(np.maximum(baseline, 1e-12)) + np.float32(evidence_scale) * evidence
    samples, markers, _ = baseline.shape
    transitions = np.log(np.maximum(np.stack([
        transition_matrix(marker_cm[index] - marker_cm[index - 1], hazard_per_morgan)
        for index in range(1, markers)
    ]), 1e-12)).astype(np.float32) if markers > 1 else np.empty((0, 6, 6), dtype=np.float32)
    forward = np.empty((samples, markers, 6), dtype=np.float32)
    forward[:, 0] = emission[:, 0] - np.logaddexp.reduce(emission[:, 0], axis=1)[:, None]
    for index in range(1, markers):
        forward[:, index] = emission[:, index] + np.logaddexp.reduce(
            forward[:, index - 1, :, None] + transitions[index - 1][None, :, :], axis=1)
        forward[:, index] -= np.logaddexp.reduce(forward[:, index], axis=1)[:, None]
    # Stream the backward message instead of retaining [N,M,6] twice.  This
    # keeps chr22 R0 within the local/GCP memory budget while retaining batch
    # vectorisation across all target individuals.
    output = np.empty_like(forward)
    backward = np.zeros((samples, 6), dtype=np.float32)
    for index in range(markers - 1, -1, -1):
        logits = forward[:, index] + backward
        output[:, index] = np.exp(logits - np.logaddexp.reduce(logits, axis=1)[:, None])
        if index:
            backward = np.logaddexp.reduce(
                transitions[index - 1][None, :, :] + emission[:, index, None, :] + backward[:, None, :], axis=2)
            backward -= np.logaddexp.reduce(backward, axis=1)[:, None]
    return output.astype(np.float32, copy=False)


def hmm_posterior_reference(baseline_states: np.ndarray, evidence: np.ndarray, marker_cm: np.ndarray,
                            hazard_per_morgan: float = 12.0, evidence_scale: float = 1.0) -> np.ndarray:
    """Slow scalar reference retained solely for numeric-regression tests."""
    baseline, evidence, marker_cm = (np.asarray(baseline_states, dtype=np.float64), np.asarray(evidence, dtype=np.float64),
                                     np.asarray(marker_cm, dtype=np.float64))
    emission = np.log(np.maximum(baseline, 1e-12)) + evidence_scale * evidence
    output = np.empty_like(baseline)
    for sample in range(baseline.shape[0]):
        forward = np.empty((baseline.shape[1], 6), dtype=np.float64)
        forward[0] = emission[sample, 0] - np.logaddexp.reduce(emission[sample, 0])
        for marker in range(1, baseline.shape[1]):
            transition = np.log(np.maximum(transition_matrix(marker_cm[marker] - marker_cm[marker - 1], hazard_per_morgan), 1e-12))
            forward[marker] = emission[sample, marker] + np.logaddexp.reduce(forward[marker - 1][:, None] + transition, axis=0)
            forward[marker] -= np.logaddexp.reduce(forward[marker])
        backward = np.zeros((baseline.shape[1], 6), dtype=np.float64)
        for marker in range(baseline.shape[1] - 2, -1, -1):
            transition = np.log(np.maximum(transition_matrix(marker_cm[marker + 1] - marker_cm[marker], hazard_per_morgan), 1e-12))
            backward[marker] = np.logaddexp.reduce(transition + emission[sample, marker + 1][None, :] + backward[marker + 1][None, :], axis=1)
            backward[marker] -= np.logaddexp.reduce(backward[marker])
        output[sample] = np.exp(forward + backward - np.logaddexp.reduce(forward + backward, axis=1)[:, None])
    return output.astype(np.float32)


@dataclass(frozen=True)
class TraceSpec:
    hidden_dim: int = 64
    depth: int = 3
    kernel_size: int = 5
    dropout: float = 0.1
    dilations: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        require(self.hidden_dim in (32, 64, 96), "hidden dimension must be preregistered")
        require(self.depth in (2, 3, 4) and self.kernel_size in (3, 5) and 0 <= self.dropout <= .2 and
                len(self.dilations) == self.depth and all(value > 0 for value in self.dilations),
                "TRACE TCN capacity is outside compact range")


def build_tcn(spec: TraceSpec, event_channels: int):
    """Compact event encoder, continuous cM splat, TCN and gated residual."""
    import torch
    from torch import nn
    require(event_channels > 0, "TRACE event channels must be positive")
    class Block(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            pad = dilation * (spec.kernel_size - 1) // 2
            self.depth = nn.Conv1d(width, width, spec.kernel_size, padding=pad, dilation=dilation, groups=width)
            self.point = nn.Conv1d(width, width, 1)
            # Normalize channels independently at each marker.  A spatial
            # GroupNorm/InstanceNorm would make a halo shard depend on its
            # length and invalidate full-vs-sharded equivalence.
            self.norm = nn.LayerNorm(width)
            self.drop = nn.Dropout(spec.dropout)
        def forward(self, value):
            convolved = self.point(self.depth(value)).transpose(1, 2)
            return value + self.drop(torch.nn.functional.gelu(self.norm(convolved)).transpose(1, 2))
    class TraceTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.event_encoder = nn.Sequential(nn.Linear(event_channels, spec.hidden_dim), nn.GELU(),
                                               nn.Linear(spec.hidden_dim, spec.hidden_dim), nn.GELU())
            # Hash 7-mer IDs into a compact vocabulary; a 16k embedding alone
            # would violate M37's <200k-cap before the actual encoder exists.
            self.context_embedding = nn.Embedding(257, min(16, spec.hidden_dim))
            self.context_projection = nn.Linear(spec.hidden_dim + min(16, spec.hidden_dim), spec.hidden_dim)
            self.stem = nn.Conv1d(spec.hidden_dim, spec.hidden_dim, 1)
            self.blocks = nn.Sequential(*[Block(spec.hidden_dim, dilation) for dilation in spec.dilations])
            self.head = nn.Conv1d(spec.hidden_dim, 6, 1)
            self.confidence = nn.Conv1d(spec.hidden_dim, 1, 1)
            nn.init.normal_(self.head.weight, std=1e-4)
            nn.init.zeros_(self.head.bias)
        def forward(self, event_values, event_context, event_sample, splat_event,
                    splat_marker, splat_weight, baseline):
            batch, markers = baseline.shape[:2]
            encoded = self.event_encoder(event_values)
            context = self.context_embedding(event_context.remainder(257))
            encoded = self.context_projection(torch.cat((encoded, context), dim=1))
            field = encoded.new_zeros((batch * markers, encoded.shape[1]))
            mass = encoded.new_zeros((batch * markers, 1))
            require(splat_event.ndim == splat_marker.ndim == splat_weight.ndim == 1 and
                    len(splat_event) == len(splat_marker) == len(splat_weight),
                    "TRACE continuous splat axes differ")
            if len(splat_event):
                flat_marker = event_sample[splat_event] * markers + splat_marker
                field.index_add_(0, flat_marker, encoded[splat_event] * splat_weight[:, None])
                mass.index_add_(0, flat_marker, splat_weight[:, None])
            field = field.reshape(batch, markers, -1).transpose(1, 2)
            hidden = self.blocks(self.stem(field))
            delta = self.head(hidden).transpose(1, 2)
            # The fixed triangular cM kernel is continuous inside the declared
            # radius and exactly zero outside it.  GEOMETRY retains this mass;
            # RD has no splats and therefore reproduces F0 exactly.
            spatial_support = mass.reshape(batch, markers, 1).clamp(0.0, 1.0)
            gate = torch.sigmoid(self.confidence(hidden).transpose(1, 2)) * spatial_support
            # Keep the unmodified FLARE posterior as the literal zero-support
            # path.  The earlier softmax(log(clamp(F0, 1e-7))) path silently
            # renormalized RD and could manufacture an apparent RD-vs-F0 gain.
            # A supported residual instead proposes q on the same 1e-12 floor
            # used by the scorer, then mixes q with raw F0.  Consequently RD is
            # exact F0, while RE/POOLED/SHAM/GEOMETRY can still revive a state
            # to which FLARE assigned probability zero.
            proposal = torch.softmax(torch.log(baseline.clamp_min(1e-12)) + delta, dim=-1)
            return (1.0 - gate) * baseline + gate * proposal
    return TraceTCN()
