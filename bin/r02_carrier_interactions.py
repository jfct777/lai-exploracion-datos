"""Fixed FIT-reference carrier-interaction feature for the FF candidate.

The operation keeps one reference-person index across locus pairs before it
aggregates references.  It consumes unphased source-minor dosages and removes
hidden target loci internally.  It does not infer IBD, phase, ancestry, or a
population label, and it does not select scientific radii or reference counts.

``pair_features`` are restricted by the interface to static locus identity or
position metadata.  That declaration can be falsified by a caller, so this
module cannot by itself certify the absence of leakage before the call.

This is a differentiable fixed feature/control, not the complete learned
``psi_theta`` architecture proposed in FF. Blocking limits each forward
temporary; without checkpointing, autograd can retain intermediates from every
block, so block size alone does not bound training peak memory.
"""
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple

import torch


STATIC_LOCUS_IDENTITY = "static_locus_identity"
STATIC_POSITION = "static_position"
STATIC_PAIR_FEATURE_KINDS = frozenset((STATIC_LOCUS_IDENTITY, STATIC_POSITION))


@dataclass(frozen=True)
class CarrierInteractionConfig:
    """Technical block limits and declared static pair-feature meanings."""

    reference_block_size: int
    pair_block_size: int
    pair_feature_kinds: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("reference_block_size", "pair_block_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"positive integer {name} required")
        if type(self.pair_feature_kinds) is not tuple:
            raise ValueError("pair_feature_kinds must be a tuple")
        if any(kind not in STATIC_PAIR_FEATURE_KINDS for kind in self.pair_feature_kinds):
            raise ValueError("pair features must be static locus identity or position metadata")


class CarrierInteractionOutput(NamedTuple):
    """Aggregated features and unnormalized observation denominators.

    ``eligible_reference_counts`` is S for historical shared eligibility and
    B x S when eligibility is explicitly target-specific.
    """

    features: torch.Tensor
    valid_pair_counts: torch.Tensor
    valid_pair_counts_by_reference: torch.Tensor
    observed_reference_counts: torch.Tensor
    eligible_reference_counts: torch.Tensor


class PreparedCarrierInputs(NamedTuple):
    """Validated masks and dosages cleaned before any interaction arithmetic."""

    target_visible: torch.Tensor
    target_clean: torch.Tensor
    reference_clean: torch.Tensor
    segments: int


def _require_bool(name, value, shape, device):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} shape differs")
    if value.device != device:
        raise ValueError(f"{name} device differs")


def _validate_ids(target_ids, reference_ids, batch, references,
                  fit_reference_eligible):
    if target_ids is None or reference_ids is None:
        raise ValueError("target_ids and reference_ids are required")
    target_ids, reference_ids = tuple(map(str, target_ids)), tuple(map(str, reference_ids))
    if len(target_ids) != batch or len(reference_ids) != references:
        raise ValueError("person identifier axis differs")
    if len(set(target_ids)) != batch or len(set(reference_ids)) != references:
        raise ValueError("person identifiers must be unique within each axis")
    overlap = set(target_ids) & set(reference_ids)
    if overlap:
        if fit_reference_eligible.ndim != 3:
            raise ValueError(
                "target person cannot be its own FIT reference; overlapping axes require B x S x R eligibility")
        reference_index = {person: index for index, person in enumerate(reference_ids)}
        for target_index, person in enumerate(target_ids):
            if person in reference_index and bool(
                    fit_reference_eligible[target_index, :, reference_index[person]].any()):
                raise ValueError("target person cannot be its own eligible FIT reference")


def eligibility_by_target(fit_reference_eligible, batch):
    """Return B x S x R eligibility without copying the historical S x R view."""
    return (fit_reference_eligible[None].expand(batch, -1, -1)
            if fit_reference_eligible.ndim == 2 else fit_reference_eligible)


def _validate_inputs(target_dosage, reference_dosage, target_called,
                     reference_called, target_hidden, pairs, pair_segment,
                     fit_reference_eligible, reference_roles, config,
                     pair_features, target_ids, reference_ids):
    if not isinstance(config, CarrierInteractionConfig):
        raise ValueError("CarrierInteractionConfig required")
    if (not isinstance(target_dosage, torch.Tensor)
            or not isinstance(reference_dosage, torch.Tensor)
            or target_dosage.ndim != 2 or reference_dosage.ndim != 2
            or not target_dosage.is_floating_point()
            or reference_dosage.dtype != target_dosage.dtype
            or reference_dosage.device != target_dosage.device):
        raise ValueError("target and reference dosage must be same-device floating matrices")
    batch, loci = target_dosage.shape
    references, reference_loci = reference_dosage.shape
    if batch == 0 or references == 0 or loci == 0 or reference_loci != loci:
        raise ValueError("nonempty target/reference matrices on the same locus axis required")
    device = target_dosage.device
    _require_bool("target_called", target_called, (batch, loci), device)
    _require_bool("reference_called", reference_called, (references, loci), device)
    _require_bool("target_hidden", target_hidden, (batch, loci), device)

    if (not isinstance(pairs, torch.Tensor) or pairs.dtype != torch.int64
            or pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.device != device
            or pairs.shape[0] == 0):
        raise ValueError("pairs must be a nonempty same-device int64 P x 2 tensor")
    if not bool(((pairs[:, 0] >= 0) & (pairs[:, 0] < pairs[:, 1])
                 & (pairs[:, 1] < loci)).all()):
        raise ValueError("each pair must satisfy 0 <= left < right < loci")
    if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
        raise ValueError("duplicate locus pairs are not allowed")
    if (not isinstance(pair_segment, torch.Tensor) or pair_segment.dtype != torch.int64
            or tuple(pair_segment.shape) != (pairs.shape[0],)
            or pair_segment.device != device):
        raise ValueError("pair_segment must be a same-device int64 P-vector")

    if (not isinstance(fit_reference_eligible, torch.Tensor)
            or fit_reference_eligible.dtype != torch.bool
            or fit_reference_eligible.ndim not in (2, 3)
            or fit_reference_eligible.shape[-1] != references
            or fit_reference_eligible.device != device):
        raise ValueError("fit_reference_eligible must be boolean S x R or B x S x R")
    if fit_reference_eligible.ndim == 2:
        segments = fit_reference_eligible.shape[0]
    else:
        if fit_reference_eligible.shape[0] != batch:
            raise ValueError("target-specific reference eligibility batch differs")
        segments = fit_reference_eligible.shape[1]
    if segments == 0:
        raise ValueError("at least one segment required")
    if not bool(((pair_segment >= 0) & (pair_segment < segments)).all()):
        raise ValueError("pair segment outside the declared segment axis")
    present = torch.bincount(pair_segment, minlength=segments) > 0
    if not bool(present.all()):
        raise ValueError("every declared segment requires at least one locus pair")
    eligible_by_target = eligibility_by_target(fit_reference_eligible, batch)
    if not bool(eligible_by_target.any(2).all()):
        raise ValueError("every target and segment requires an admissible FIT reference")

    roles = tuple(reference_roles)
    if len(roles) != references or any(role != "FIT" for role in roles):
        raise ValueError("every reference role must be explicitly FIT")
    _validate_ids(target_ids, reference_ids, batch, references,
                  fit_reference_eligible)

    feature_count = len(config.pair_feature_kinds)
    if feature_count == 0:
        if pair_features is not None:
            raise ValueError("pair_features supplied without declared static feature kinds")
    elif (not isinstance(pair_features, torch.Tensor)
          or pair_features.shape != (pairs.shape[0], feature_count)
          or pair_features.dtype != target_dosage.dtype
          or pair_features.device != device
          or pair_features.requires_grad):
        raise ValueError("static pair_features must be a finite non-trainable P x F tensor")
    elif not bool(torch.isfinite(pair_features).all()):
        raise ValueError("nonfinite static pair feature")

    # Do not inspect target values at hidden cells, including called-but-hidden
    # cells.  They are removed before any arithmetic below.
    target_visible = target_called & ~target_hidden
    visible_target_values = target_dosage[target_visible]
    called_reference_values = reference_dosage[reference_called]
    for name, values in (("visible target", visible_target_values),
                         ("called reference", called_reference_values)):
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"nonfinite {name} dosage")
        if not bool(((values == 0) | (values == 1) | (values == 2)).all()):
            raise ValueError(f"{name} dosage must be 0, 1, or 2")
    return target_visible, segments


def prepare_carrier_inputs(target_dosage: torch.Tensor,
                           reference_dosage: torch.Tensor,
                           target_called: torch.Tensor,
                           reference_called: torch.Tensor,
                           target_hidden: torch.Tensor,
                           pairs: torch.Tensor,
                           pair_segment: torch.Tensor,
                           fit_reference_eligible: torch.Tensor,
                           reference_roles: Sequence[str],
                           config: CarrierInteractionConfig,
                           target_ids: Sequence[str],
                           reference_ids: Sequence[str],
                           pair_features: Optional[torch.Tensor] = None,
                           ) -> PreparedCarrierInputs:
    """Validate shared FG/FH inputs and remove hidden or missing cell values."""
    target_visible, segments = _validate_inputs(
        target_dosage, reference_dosage, target_called, reference_called,
        target_hidden, pairs, pair_segment, fit_reference_eligible,
        reference_roles, config, pair_features, target_ids, reference_ids)
    # torch.where prevents NaN/Inf or arbitrary hidden/missing cells from
    # entering arithmetic. Validation deliberately never indexes hidden cells.
    target_clean = torch.where(target_visible, target_dosage,
                               torch.zeros((), dtype=target_dosage.dtype,
                                           device=target_dosage.device))
    reference_clean = torch.where(reference_called, reference_dosage,
                                  torch.zeros((), dtype=reference_dosage.dtype,
                                              device=reference_dosage.device))
    return PreparedCarrierInputs(target_visible, target_clean,
                                 reference_clean, segments)


def _static_pair_basis(pair_features, count, dtype, device):
    ones = torch.ones((count, 1), dtype=dtype, device=device)
    return ones if pair_features is None else torch.cat((ones, pair_features), dim=1)


def _aggregate_block(target_clean, reference_clean, target_visible,
                     reference_called, pairs, pair_segment,
                     fit_reference_eligible, pair_features, segments,
                     pair_block_size):
    """Return v[i,j,s,c] after summing pairs for the same reference j."""
    batch, references = target_clean.shape[0], reference_clean.shape[0]
    channels = 1 + (0 if pair_features is None else pair_features.shape[1])
    values = target_clean.new_zeros((batch, references, segments, channels))
    counts = torch.zeros((batch, references, segments), dtype=torch.int64,
                         device=target_clean.device)
    for start in range(0, pairs.shape[0], pair_block_size):
        stop = min(start + pair_block_size, pairs.shape[0])
        block_pairs = pairs[start:stop]
        left, right = block_pairs[:, 0], block_pairs[:, 1]
        segment = pair_segment[start:stop]
        target_valid = target_visible[:, left] & target_visible[:, right]
        reference_valid = reference_called[:, left] & reference_called[:, right]
        # fit_reference_eligible is normalized to B x S x R here.
        pair_eligible = fit_reference_eligible[:, segment, :].permute(0, 2, 1)
        valid = (target_valid[:, None, :] & reference_valid[None, :, :]
                 & pair_eligible)
        target_pair = target_clean[:, left] * target_clean[:, right]
        reference_pair = reference_clean[:, left] * reference_clean[:, right]
        shared_pair_dosage = target_pair[:, None, :] * reference_pair[None, :, :]
        block_features = None if pair_features is None else pair_features[start:stop]
        basis = _static_pair_basis(block_features, stop - start,
                                   target_clean.dtype, target_clean.device)
        contribution = (valid.to(target_clean.dtype) * shared_pair_dosage)[..., None]
        contribution = contribution * basis[None, None, :, :]
        values = values.index_add(2, segment, contribution)
        counts = counts.index_add(2, segment, valid.to(torch.int64))
    return values, counts


def carrier_interactions(target_dosage: torch.Tensor,
                         reference_dosage: torch.Tensor,
                         target_called: torch.Tensor,
                         reference_called: torch.Tensor,
                         target_hidden: torch.Tensor,
                         pairs: torch.Tensor,
                         pair_segment: torch.Tensor,
                         fit_reference_eligible: torch.Tensor,
                         reference_roles: Sequence[str],
                         config: CarrierInteractionConfig,
                         target_ids: Sequence[str],
                         reference_ids: Sequence[str],
                         pair_features: Optional[torch.Tensor] = None,
                         ) -> CarrierInteractionOutput:
    """Compute fixed pair interaction and rho=(v, v^2), in bounded blocks.

    For target ``i``, reference ``j`` and segment ``s``, ``v[i,j,s]`` sums
    dosage products over pairs while retaining the same ``j``.  Only then does
    the fixed, differentiable transformation ``rho(v) = (v, v^2)`` aggregate
    references.  Its squared term makes the distribution across references
    observable; neither term is evidence of IBD.

    Both ID axes are mandatory, including for anonymous fixtures, so no caller
    can accidentally omit the self-reference check. ``valid_pair_counts`` is
    returned for downstream handling of observation support; this operation
    does not normalize by it. The square in ``rho`` can dominate in high-count
    segments, so any scaling used by a later model must be fixed from FIT before
    training and recorded separately.
    """
    prepared = prepare_carrier_inputs(
        target_dosage, reference_dosage, target_called, reference_called,
        target_hidden, pairs, pair_segment, fit_reference_eligible,
        reference_roles, config, target_ids, reference_ids, pair_features)
    target_visible, target_clean, reference_clean, segments = prepared
    eligible_by_target = eligibility_by_target(
        fit_reference_eligible, target_dosage.shape[0])
    channels = 1 + len(config.pair_feature_kinds)
    features = target_dosage.new_zeros((target_dosage.shape[0], segments, 2 * channels))
    pair_counts = torch.zeros((target_dosage.shape[0], segments), dtype=torch.int64,
                              device=target_dosage.device)
    counts_by_reference = []
    for start in range(0, reference_dosage.shape[0], config.reference_block_size):
        stop = min(start + config.reference_block_size, reference_dosage.shape[0])
        values, counts = _aggregate_block(
            target_clean, reference_clean[start:stop], target_visible,
            reference_called[start:stop], pairs, pair_segment,
            eligible_by_target[:, :, start:stop], pair_features, segments,
            config.pair_block_size)
        rho = torch.cat((values, values.square()), dim=-1)
        features = features + rho.sum(1)
        pair_counts = pair_counts + counts.sum(1)
        counts_by_reference.append(counts)
    counts_by_reference = torch.cat(counts_by_reference, dim=1)
    return CarrierInteractionOutput(
        features, pair_counts, counts_by_reference,
        (counts_by_reference > 0).sum(1, dtype=torch.int64),
        (fit_reference_eligible.sum(1, dtype=torch.int64)
         if fit_reference_eligible.ndim == 2
         else fit_reference_eligible.sum(2, dtype=torch.int64)))
