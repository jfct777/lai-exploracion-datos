"""Loss terms for reconstructing deliberately hidden genotype dosages.

This module only computes cross-entropy sums, objective counts, their balanced
combination, and empirical FIT probabilities.  It does not choose masked cells,
call an optimizer, or perform an optimizer update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ChannelCrossEntropy:
    """Cross-entropy sum and number of scored genotypes for one channel."""

    cross_entropy_sum: torch.Tensor
    objective_count: int


@dataclass(frozen=True)
class MaskedGenotypeLossTerms:
    """Separate common- and rare-genotype loss terms for one fragment."""

    common: ChannelCrossEntropy
    rare: ChannelCrossEntropy


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def masked_cross_entropy_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    score_mask: torch.Tensor,
) -> ChannelCrossEntropy:
    """Return summed three-dose cross-entropy and its scored-objective count.

    ``logits`` has shape ``(..., 3)`` for doses 0, 1 and 2. ``targets`` and
    ``score_mask`` both have shape ``logits.shape[:-1]``. Values outside the
    score mask are never read as targets and nonfinite logits outside that mask
    do not affect the result. An empty mask returns a differentiable scalar zero
    connected to ``logits`` and count zero.
    """

    _require(isinstance(logits, torch.Tensor), "logits must be a torch tensor")
    _require(isinstance(targets, torch.Tensor), "targets must be a torch tensor")
    _require(isinstance(score_mask, torch.Tensor), "score_mask must be a torch tensor")
    _require(torch.is_floating_point(logits), "logits must have a floating dtype")
    _require(logits.ndim >= 1 and logits.shape[-1] == 3,
             "logits must have exactly three dose categories on the last axis")
    expected_shape = logits.shape[:-1]
    _require(targets.shape == expected_shape, "targets shape must equal logits.shape[:-1]")
    _require(score_mask.shape == expected_shape,
             "score_mask shape must equal logits.shape[:-1]")
    _require(targets.dtype == torch.int64, "targets must have torch.int64 dtype")
    _require(score_mask.dtype == torch.bool, "score_mask must have torch.bool dtype")
    _require(targets.device == logits.device and score_mask.device == logits.device,
             "logits, targets and score_mask must be on the same device")

    selected_logits = logits[score_mask]
    selected_targets = targets[score_mask]
    objective_count = int(selected_targets.numel())
    if objective_count == 0:
        return ChannelCrossEntropy(selected_logits.sum(), 0)

    _require(bool(torch.isfinite(selected_logits).all().item()),
             "selected logits must all be finite")
    _require(bool(((selected_targets >= 0) & (selected_targets <= 2)).all().item()),
             "selected targets must be dose categories 0, 1 or 2")
    value = F.cross_entropy(selected_logits, selected_targets, reduction="sum")
    _require(bool(torch.isfinite(value).item()), "selected cross-entropy sum is nonfinite")
    return ChannelCrossEntropy(value, objective_count)


def masked_genotype_loss_terms(
    common_logits: torch.Tensor,
    common_targets: torch.Tensor,
    common_score_mask: torch.Tensor,
    rare_logits: torch.Tensor,
    rare_targets: torch.Tensor,
    rare_score_mask: torch.Tensor,
) -> MaskedGenotypeLossTerms:
    """Compute separate common and rare cross-entropy sums for one fragment."""

    common = masked_cross_entropy_sum(common_logits, common_targets, common_score_mask)
    rare = masked_cross_entropy_sum(rare_logits, rare_targets, rare_score_mask)
    _require(common.cross_entropy_sum.device == rare.cross_entropy_sum.device,
             "common and rare loss sums must be on the same device")
    _require(common.cross_entropy_sum.dtype == rare.cross_entropy_sum.dtype,
             "common and rare loss sums must have the same dtype")
    return MaskedGenotypeLossTerms(common=common, rare=rare)


def balanced_masked_genotype_loss(terms: MaskedGenotypeLossTerms) -> torch.Tensor:
    """Return ``0.5 * S_common/N_common + 0.5 * S_rare/N_rare``.

    This is the final loss for one complete update or evaluation. Both channels
    must contain at least one scored genotype.
    """

    _require(terms.common.objective_count > 0,
             "complete loss requires at least one common-genotype objective")
    _require(terms.rare.objective_count > 0,
             "complete loss requires at least one rare-genotype objective")
    return (0.5 * terms.common.cross_entropy_sum / terms.common.objective_count
            + 0.5 * terms.rare.cross_entropy_sum / terms.rare.objective_count)


def scaled_fragment_loss(
    terms: MaskedGenotypeLossTerms,
    *,
    global_common_count: int,
    global_rare_count: int,
) -> torch.Tensor:
    """Scale one fragment's sums by complete-update denominators.

    Calling ``backward()`` on each newly computed fragment result accumulates
    the same gradient as the unfragmented balanced loss. No graph from an
    earlier fragment needs to be retained. This function neither clears
    gradients nor calls an optimizer.
    """

    _require(isinstance(global_common_count, int) and not isinstance(global_common_count, bool)
             and global_common_count > 0,
             "global_common_count must be a positive integer")
    _require(isinstance(global_rare_count, int) and not isinstance(global_rare_count, bool)
             and global_rare_count > 0,
             "global_rare_count must be a positive integer")
    _require(0 <= terms.common.objective_count <= global_common_count,
             "fragment common count exceeds the complete-update count")
    _require(0 <= terms.rare.objective_count <= global_rare_count,
             "fragment rare count exceeds the complete-update count")
    return (0.5 * terms.common.cross_entropy_sum / global_common_count
            + 0.5 * terms.rare.cross_entropy_sum / global_rare_count)


def fit_empirical_genotype_probabilities(
    fit_counts: np.ndarray,
    *,
    pseudocount_per_category: Real,
    validation_population: str,
    excluded_counts: Optional[np.ndarray],
) -> np.ndarray:
    """Estimate per-locus probabilities for doses 0, 1 and 2 from FIT counts.

    For each locus and category ``k``, the returned value is
    ``(fit_counts[k] - excluded_counts[k] + alpha) / (remaining_total + 3*alpha)``.
    ``alpha`` is the explicit pseudocount added to *each* category.

    ``FIT_INTERNAL`` requires observed excluded counts at every locus.
    ``SELECT_DISJOINT`` also requires an explicit array, normally all zeros, so
    zero exclusion is recorded rather than inferred. Numeric excluded counts do
    not establish the excluded people's identities, uniqueness, or complete
    family-component membership; those checks belong to the data preparation.
    """

    counts = np.asarray(fit_counts)
    _require(counts.ndim >= 1 and counts.shape[-1] == 3 and counts.size > 0,
             "fit_counts must end in three genotype categories")
    _require(counts.dtype.kind in "iu", "fit_counts must use an integer dtype")
    _require(bool(np.all(counts >= 0)), "fit_counts cannot be negative")
    _require(bool(np.all(counts <= np.iinfo(np.int64).max // 3)),
             "fit_counts exceed safe integer accumulation")
    _require(validation_population in {"FIT_INTERNAL", "SELECT_DISJOINT"},
             "validation_population must be FIT_INTERNAL or SELECT_DISJOINT")
    _require(excluded_counts is not None,
             f"{validation_population} requires explicit excluded_counts")
    excluded = np.asarray(excluded_counts)
    _require(excluded.shape == counts.shape,
             "excluded_counts shape must equal fit_counts shape")
    _require(excluded.dtype.kind in "iu", "excluded_counts must use an integer dtype")
    _require(bool(np.all(excluded >= 0)), "excluded_counts cannot be negative")
    _require(bool(np.all(excluded <= counts)),
             "excluded_counts cannot exceed fit_counts in any category")
    _require(isinstance(pseudocount_per_category, Real)
             and not isinstance(pseudocount_per_category, (bool, np.bool_)),
             "pseudocount_per_category must be a real scalar")
    alpha = float(pseudocount_per_category)
    _require(math.isfinite(alpha) and alpha > 0,
             "pseudocount_per_category must be finite and positive")

    remaining = counts.astype(np.int64, copy=False) - excluded.astype(np.int64, copy=False)
    remaining_total = remaining.sum(axis=-1)
    _require(bool(np.all(remaining_total > 0)),
             "every locus must retain at least one observed genotype after exclusion")
    # Scale before addition: finite alpha alone does not make 3*alpha finite.
    scale = np.maximum(remaining_total.astype(np.float64), alpha)[..., np.newaxis]
    numerator = remaining.astype(np.float64) / scale + alpha / scale
    denominator = remaining_total.astype(np.float64)[..., np.newaxis] / scale + 3.0 * (alpha / scale)
    probabilities = numerator / denominator
    _require(bool(np.isfinite(probabilities).all()) and bool(np.all(probabilities > 0)),
             "empirical probabilities must be finite and strictly positive")
    _require(bool(np.allclose(probabilities.sum(axis=-1), 1., rtol=0, atol=1e-12)),
             "empirical probabilities do not sum to one")
    return probabilities
