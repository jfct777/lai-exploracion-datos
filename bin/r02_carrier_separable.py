"""Exact linear recurrence for a restricted family of visible-locus pairs.

For ordered visible loci ``l < r`` this module sums

    exp(-(c[r] - c[l]) / decay_length_cm) * A[l] * B[r]

elementwise.  ``A`` and ``B`` are explicit per-locus features for one target
person and one eligible FIT reference.  They may be equal, but that symmetry
must be requested through :func:`exponential_separable_same_features`.

The three outputs name genomic territories: pairs within the left visible
flank, pairs across the hidden genomic span, and pairs within the right
visible flank.  They are not haplotypes.  The hidden cell has no feature row;
its width survives only in the cM gap between the two visible flanks.

Pair validity must factor into a fixed target-reference eligibility flag and
one validity flag per locus.  A non-factorable pair mask is outside this
family.  Chunk state is an in-memory autograd object.  Serializing or detaching
that state for a later backward pass is unsupported: training recovery must
recompute the complete objective from a durable model/optimizer checkpoint,
not resume mid-backward.
"""
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


class SeparableCarrierOutput(NamedTuple):
    """Exact features and unweighted valid-pair support by genomic territory."""

    within_left: torch.Tensor
    across_flanks: torch.Tensor
    within_right: torch.Tensor
    within_left_count: torch.Tensor
    across_flanks_count: torch.Tensor
    within_right_count: torch.Tensor

    @property
    def features(self) -> torch.Tensor:
        return torch.stack(
            (self.within_left, self.across_flanks, self.within_right), dim=2)

    @property
    def valid_pair_counts(self) -> torch.Tensor:
        return torch.stack(
            (self.within_left_count, self.across_flanks_count,
             self.within_right_count), dim=2)

    @property
    def has_pair_support(self) -> torch.Tensor:
        return self.valid_pair_counts > 0


@dataclass(frozen=True)
class SeparableCarrierState:
    """In-memory forward state; retaining it also retains its autograd graph."""

    context_id: str
    axis_id: str
    total_loci: int
    next_index: int
    feature_width: int
    decay_length_cm: torch.Tensor
    person_reference_eligible: torch.Tensor
    last_position_cm: Optional[torch.Tensor]
    last_flank_code: int
    left_message: torch.Tensor
    right_message: torch.Tensor
    left_locus_count: torch.Tensor
    right_locus_count: torch.Tensor
    within_left: torch.Tensor
    across_flanks: torch.Tensor
    within_right: torch.Tensor
    within_left_count: torch.Tensor
    across_flanks_count: torch.Tensor
    within_right_count: torch.Tensor


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _identity(name: str, value: str) -> str:
    _require(type(value) is str and bool(value), f"nonempty {name} required")
    return value


def initialize_separable_state(
        *, context_id: str, axis_id: str, total_loci: int,
        feature_width: int, decay_length_cm: torch.Tensor,
        person_reference_eligible: torch.Tensor) -> SeparableCarrierState:
    """Create an empty state bound to one immutable visible-locus context."""
    context_id, axis_id = (_identity("context_id", context_id),
                           _identity("axis_id", axis_id))
    _require(type(total_loci) is int and total_loci > 0,
             "positive integer total_loci required")
    _require(type(feature_width) is int and feature_width > 0,
             "positive integer feature_width required")
    _require(isinstance(decay_length_cm, torch.Tensor)
             and decay_length_cm.ndim == 0
             and decay_length_cm.is_floating_point()
             and bool(torch.isfinite(decay_length_cm.detach()).item())
             and bool((decay_length_cm.detach() > 0).item()),
             "finite positive scalar decay_length_cm required")
    eligible = person_reference_eligible
    _require(isinstance(eligible, torch.Tensor) and eligible.dtype == torch.bool
             and eligible.ndim == 2 and eligible.shape[0] > 0
             and eligible.shape[1] > 0
             and eligible.device == decay_length_cm.device,
             "same-device nonempty boolean B x R eligibility required")
    eligible = eligible.clone()
    batch, references = eligible.shape
    values = torch.zeros(
        (batch, references, feature_width), dtype=decay_length_cm.dtype,
        device=decay_length_cm.device)
    counts = torch.zeros((batch, references), dtype=torch.int64,
                         device=decay_length_cm.device)
    return SeparableCarrierState(
        context_id, axis_id, total_loci, 0, feature_width, decay_length_cm,
        eligible, None, -1, values, values.clone(), counts, counts.clone(),
        values.clone(), values.clone(), values.clone(), counts.clone(),
        counts.clone(), counts.clone())


def _validate_chunk(
        state: SeparableCarrierState, source_features: torch.Tensor,
        destination_features: torch.Tensor, positions_cm: torch.Tensor,
        locus_valid: torch.Tensor, left_flank: torch.Tensor,
        right_flank: torch.Tensor, *, context_id: str, axis_id: str,
        start_index: int) -> int:
    _require(isinstance(state, SeparableCarrierState),
             "SeparableCarrierState required")
    _require(context_id == state.context_id and axis_id == state.axis_id,
             "chunk context or visible-locus axis differs from state")
    _require(type(start_index) is int and start_index == state.next_index,
             "chunk overlaps, skips, or is out of order")
    source, destination = source_features, destination_features
    _require(isinstance(source, torch.Tensor)
             and isinstance(destination, torch.Tensor)
             and source.ndim == 4 and destination.shape == source.shape,
             "source and destination features must share B x R x L x Q")
    batch, references, loci, width = source.shape
    _require(loci > 0 and width == state.feature_width
             and (batch, references) == tuple(
                 state.person_reference_eligible.shape),
             "feature axes differ from state")
    _require(source.dtype == destination.dtype == state.decay_length_cm.dtype
             and source.device == destination.device
             == state.decay_length_cm.device,
             "feature dtype or device differs from state")
    _require(start_index + loci <= state.total_loci,
             "chunk exceeds declared visible-locus axis")
    _require(isinstance(positions_cm, torch.Tensor)
             and positions_cm.ndim == 1 and len(positions_cm) == loci
             and positions_cm.is_floating_point()
             and positions_cm.dtype == source.dtype
             and positions_cm.device == source.device
             and not positions_cm.requires_grad
             and bool(torch.isfinite(positions_cm).all()),
             "finite static same-dtype cM positions required")
    _require(isinstance(locus_valid, torch.Tensor)
             and locus_valid.dtype == torch.bool
             and tuple(locus_valid.shape) == (batch, references, loci)
             and locus_valid.device == source.device,
             "locus_valid must be same-device boolean B x R x L")
    for name, flank in (("left_flank", left_flank),
                        ("right_flank", right_flank)):
        _require(isinstance(flank, torch.Tensor) and flank.dtype == torch.bool
                 and tuple(flank.shape) == (loci,)
                 and flank.device == source.device,
                 f"{name} must be a same-device boolean L-vector")
    _require(not bool((left_flank & right_flank).any()),
             "left and right visible flanks overlap")
    _require(bool((left_flank | right_flank).all()),
             "every supplied feature locus must belong to one visible flank")
    if loci > 1:
        _require(bool((positions_cm[1:] >= positions_cm[:-1]).all()),
                 "visible-locus positions are not ordered")
    if state.last_position_cm is not None:
        _require(bool((positions_cm[0] >= state.last_position_cm).item()),
                 "chunk position overlaps or precedes prior position")
    flank_code = right_flank.to(torch.int8)
    if loci > 1:
        _require(bool((flank_code[1:] >= flank_code[:-1]).all()),
                 "left visible flank occurs after right visible flank")
    _require(int(flank_code[0].item()) >= state.last_flank_code,
             "flank order differs across chunk boundary")
    active = (locus_valid
              & state.person_reference_eligible[:, :, None])
    _require(bool(torch.isfinite(source[active]).all())
             and bool(torch.isfinite(destination[active]).all()),
             "valid visible-locus features must be finite")
    return loci


def accumulate_separable_chunk(
        state: SeparableCarrierState, source_features: torch.Tensor,
        destination_features: torch.Tensor, positions_cm: torch.Tensor,
        locus_valid: torch.Tensor, left_flank: torch.Tensor,
        right_flank: torch.Tensor, *, context_id: str, axis_id: str,
        start_index: int) -> SeparableCarrierState:
    """Accumulate one contiguous chunk without a subtractive suffix.

    ``source_features[..., l, :]`` supplies the earlier-locus factor A_l and
    ``destination_features[..., r, :]`` supplies the later-locus factor B_r.
    The same differentiable ``decay_length_cm`` object remains in ``state``.
    """
    loci = _validate_chunk(
        state, source_features, destination_features, positions_cm,
        locus_valid, left_flank, right_flank, context_id=context_id,
        axis_id=axis_id, start_index=start_index)
    active = locus_valid & state.person_reference_eligible[:, :, None]
    zero = torch.zeros((), dtype=source_features.dtype,
                       device=source_features.device)
    source = torch.where(active[..., None], source_features, zero)
    destination = torch.where(active[..., None], destination_features, zero)

    left_message, right_message = state.left_message, state.right_message
    left_count, right_count = state.left_locus_count, state.right_locus_count
    within_left, across, within_right = (
        state.within_left, state.across_flanks, state.within_right)
    count_ll, count_lr, count_rr = (
        state.within_left_count, state.across_flanks_count,
        state.within_right_count)
    last_position = state.last_position_cm

    for offset in range(loci):
        position = positions_cm[offset]
        if last_position is not None:
            decay = torch.exp(
                -(position - last_position) / state.decay_length_cm)
            left_message = left_message * decay
            right_message = right_message * decay
        valid = active[:, :, offset]
        valid_count = valid.to(torch.int64)
        source_value = source[:, :, offset]
        destination_value = destination[:, :, offset]
        if bool(left_flank[offset].item()):
            within_left = within_left + destination_value * left_message
            count_ll = count_ll + valid_count * left_count
            left_message = left_message + source_value
            left_count = left_count + valid_count
            last_flank_code = 0
        else:
            across = across + destination_value * left_message
            within_right = within_right + destination_value * right_message
            count_lr = count_lr + valid_count * left_count
            count_rr = count_rr + valid_count * right_count
            right_message = right_message + source_value
            right_count = right_count + valid_count
            last_flank_code = 1
        last_position = position.detach()

    return SeparableCarrierState(
        state.context_id, state.axis_id, state.total_loci,
        state.next_index + loci, state.feature_width, state.decay_length_cm,
        state.person_reference_eligible, last_position, last_flank_code,
        left_message, right_message, left_count, right_count, within_left,
        across, within_right, count_ll, count_lr, count_rr)


def finalize_separable_state(
        state: SeparableCarrierState, *, context_id: str,
        axis_id: str) -> SeparableCarrierOutput:
    """Return exact features only after the declared visible axis is complete."""
    _require(isinstance(state, SeparableCarrierState),
             "SeparableCarrierState required")
    _require(context_id == state.context_id and axis_id == state.axis_id,
             "final context or visible-locus axis differs from state")
    _require(state.next_index == state.total_loci,
             "cannot finalize an incomplete visible-locus axis")
    return SeparableCarrierOutput(
        state.within_left, state.across_flanks, state.within_right,
        state.within_left_count, state.across_flanks_count,
        state.within_right_count)


def exponential_separable_pairs(
        source_features: torch.Tensor, destination_features: torch.Tensor,
        positions_cm: torch.Tensor, locus_valid: torch.Tensor,
        left_flank: torch.Tensor, right_flank: torch.Tensor,
        person_reference_eligible: torch.Tensor,
        decay_length_cm: torch.Tensor, *, context_id: str, axis_id: str,
        chunk_size: int) -> SeparableCarrierOutput:
    """Run the exact recurrence over all visible loci in contiguous chunks."""
    _require(isinstance(source_features, torch.Tensor)
             and source_features.ndim == 4 and source_features.shape[2] > 0,
             "source_features must be nonempty B x R x L x Q")
    _require(type(chunk_size) is int and chunk_size > 0,
             "positive integer chunk_size required")
    state = initialize_separable_state(
        context_id=context_id, axis_id=axis_id,
        total_loci=source_features.shape[2],
        feature_width=source_features.shape[3],
        decay_length_cm=decay_length_cm,
        person_reference_eligible=person_reference_eligible)
    for start in range(0, source_features.shape[2], chunk_size):
        stop = min(start + chunk_size, source_features.shape[2])
        state = accumulate_separable_chunk(
            state, source_features[:, :, start:stop],
            destination_features[:, :, start:stop], positions_cm[start:stop],
            locus_valid[:, :, start:stop], left_flank[start:stop],
            right_flank[start:stop], context_id=context_id, axis_id=axis_id,
            start_index=start)
    return finalize_separable_state(
        state, context_id=context_id, axis_id=axis_id)


def exponential_separable_same_features(
        features: torch.Tensor, positions_cm: torch.Tensor,
        locus_valid: torch.Tensor, left_flank: torch.Tensor,
        right_flank: torch.Tensor, person_reference_eligible: torch.Tensor,
        decay_length_cm: torch.Tensor, *, context_id: str, axis_id: str,
        chunk_size: int) -> SeparableCarrierOutput:
    """Explicit symmetric-feature specialization with A_l and B_r both phi."""
    return exponential_separable_pairs(
        features, features, positions_cm, locus_valid, left_flank,
        right_flank, person_reference_eligible, decay_length_cm,
        context_id=context_id, axis_id=axis_id, chunk_size=chunk_size)
