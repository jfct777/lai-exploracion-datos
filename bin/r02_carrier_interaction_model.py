"""Learned, blocked carrier-pair interaction proposed for exploratory FF/FH.

The module retains one FIT reference across visible locus pairs, applies a
shared ``psi_theta`` before summing pairs, then applies a shared ``rho_phi``
before summing references. It consumes unphased DNABR2723 ``source_minor``
dosages and reuses FG validation/masking. Learned values are not IBD, phase,
ancestry, age, or population assignments.

Pair-common context ``b_ij,s`` can enter ``rho_phi`` once after locus pairs for
the same reference have been summed.  A structured declaration checks tensor
axes and that every hidden locus is named in that target person's segment
exclusion.  This
internal consistency check does not authenticate an arbitrary producer; human
training remains blocked until the real producer and exclusions are verified.
Pair/reference selection, network sizes and normalization remain external.
Blocking bounds forward temporaries. The optional checkpoint recomputes only
``psi_theta``; it does not prove a bounded training peak for ``rho_phi`` or the
remaining autograd graph.
"""
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from r02_carrier_interactions import (
    CarrierInteractionConfig, CarrierInteractionOutput,
    eligibility_by_target, prepare_carrier_inputs,
)


@dataclass(frozen=True)
class LearnedCarrierInteractionConfig:
    """Explicit network capacities and technical block choices; no defaults."""

    reference_block_size: int
    pair_block_size: int
    pair_feature_kinds: Tuple[str, ...]
    psi_hidden_widths: Tuple[int, ...]
    psi_output_width: int
    rho_hidden_widths: Tuple[int, ...]
    rho_output_width: int
    checkpoint_blocks: bool
    common_context_width: int = 0

    def __post_init__(self):
        CarrierInteractionConfig(self.reference_block_size, self.pair_block_size,
                                 self.pair_feature_kinds)
        for name in ("psi_hidden_widths", "rho_hidden_widths"):
            value = getattr(self, name)
            if (type(value) is not tuple or not value
                    or any(type(width) is not int or width <= 0 for width in value)):
                raise ValueError(f"{name} must be a nonempty tuple of positive integers")
        for name in ("psi_output_width", "rho_output_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"positive integer {name} required")
        if type(self.checkpoint_blocks) is not bool:
            raise ValueError("checkpoint_blocks must be boolean")
        if type(self.common_context_width) is not int or self.common_context_width < 0:
            raise ValueError("common_context_width must be a nonnegative integer")

    def validation_config(self) -> CarrierInteractionConfig:
        return CarrierInteractionConfig(self.reference_block_size,
                                        self.pair_block_size,
                                        self.pair_feature_kinds)


class CarrierQueryFeatures(NamedTuple):
    """Segment representation and unnormalized support for output queries."""

    features: torch.Tensor
    valid_pair_counts: torch.Tensor
    observed_reference_counts: torch.Tensor
    eligible_reference_counts: torch.Tensor


@dataclass(frozen=True)
class PairCommonContextDeclaration:
    """Declared axes and sparse locus exclusions; not proof of provenance.

    ``excluded_locus_ids_by_target_segment`` is a sparse ``B x S`` tuple: each
    cell names loci removed for one target person and one common-context
    territory.  It never denotes excluded people.  The model checks each row
    only against that person's ``target_hidden`` cells; another target's mask
    cannot enlarge its exclusion.  An external admission must still
    authenticate how the producer calculated the supplied values.
    """

    target_ids: Tuple[str, ...]
    reference_ids: Tuple[str, ...]
    segment_ids: Tuple[str, ...]
    excluded_locus_ids_by_target_segment: Tuple[
        Tuple[Tuple[str, ...], ...], ...]


@dataclass(frozen=True)
class PairCommonContext:
    """Pair-common values, their observation mask and checked declaration."""

    values: torch.Tensor
    valid: torch.Tensor
    declaration: PairCommonContextDeclaration


class PairCommonContextSupport(NamedTuple):
    """Support before and after common-context contribution masking."""

    original_valid_pair_counts: torch.Tensor
    original_valid_pair_counts_by_reference: torch.Tensor
    original_observed_reference_counts: torch.Tensor
    context_valid_eligible_reference_counts: torch.Tensor
    dropped_eligible_reference_counts: torch.Tensor
    dropped_valid_pair_counts: torch.Tensor
    dropped_observed_reference_counts: torch.Tensor


class ContextualCarrierInteractionOutput(NamedTuple):
    """Effective carrier output plus a non-silent lost-support ledger."""

    carrier: CarrierInteractionOutput
    support: PairCommonContextSupport

    @property
    def features(self):
        return self.carrier.features

    @property
    def valid_pair_counts(self):
        return self.carrier.valid_pair_counts

    @property
    def valid_pair_counts_by_reference(self):
        return self.carrier.valid_pair_counts_by_reference

    @property
    def observed_reference_counts(self):
        return self.carrier.observed_reference_counts

    @property
    def eligible_reference_counts(self):
        return self.carrier.eligible_reference_counts


def _mlp(input_width: int, hidden_widths: Tuple[int, ...],
         output_width: int) -> nn.Sequential:
    layers = []
    current = input_width
    for width in hidden_widths:
        layers.extend((nn.Linear(current, width), nn.GELU()))
        current = width
    layers.append(nn.Linear(current, output_width))
    return nn.Sequential(*layers)


def _ordered_ids(name: str, values: Optional[Sequence[str]], length: int) -> Tuple[str, ...]:
    if values is None:
        raise ValueError(f"{name} required with common context")
    result = tuple(map(str, values))
    if len(result) != length or len(set(result)) != length or any(not value for value in result):
        raise ValueError(f"{name} must be a unique nonempty axis of length {length}")
    return result


def _prepare_common_context(
        config: LearnedCarrierInteractionConfig,
        context: Optional[PairCommonContext],
        target_ids: Sequence[str], reference_ids: Sequence[str],
        locus_ids: Optional[Sequence[str]], segment_ids: Optional[Sequence[str]],
        target_hidden: torch.Tensor, fit_reference_eligible: torch.Tensor,
        dtype: torch.dtype, device: torch.device,
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Validate context consistency and mask invalid numerical sentinels.

    The declaration is necessary but cannot prove that an arbitrary producer
    actually excluded the named loci.  Authentication belongs to the future
    human-data adapter and admission.
    """
    if config.common_context_width == 0:
        if context is not None:
            raise ValueError("common_context must be absent when common_context_width is zero")
        return None
    if not isinstance(context, PairCommonContext):
        raise ValueError("PairCommonContext required when common_context_width is positive")
    if not isinstance(context.declaration, PairCommonContextDeclaration):
        raise ValueError("PairCommonContextDeclaration required")

    batch, loci = target_hidden.shape
    if fit_reference_eligible.ndim == 2:
        segments, references = fit_reference_eligible.shape
    else:
        eligibility_batch, segments, references = fit_reference_eligible.shape
        if eligibility_batch != batch:
            raise ValueError("target-specific common-context eligibility batch differs")
    target_axis = _ordered_ids("target_ids", target_ids, batch)
    reference_axis = _ordered_ids("reference_ids", reference_ids, references)
    locus_axis = _ordered_ids("locus_ids", locus_ids, loci)
    segment_axis = _ordered_ids("segment_ids", segment_ids, segments)
    declaration = context.declaration
    if (declaration.target_ids != target_axis
            or declaration.reference_ids != reference_axis
            or declaration.segment_ids != segment_axis):
        raise ValueError("common context target/reference/segment identities differ")
    exclusions = declaration.excluded_locus_ids_by_target_segment
    if type(exclusions) is not tuple or len(exclusions) != batch:
        raise ValueError("one target-person exclusion row required")
    for target_index, target_exclusions in enumerate(exclusions):
        if type(target_exclusions) is not tuple or len(target_exclusions) != segments:
            raise ValueError("one excluded-locus tuple required per target and segment")
        hidden_indices = torch.nonzero(
            target_hidden[target_index], as_tuple=False).flatten().tolist()
        hidden_locus_ids = {locus_axis[index] for index in hidden_indices}
        for excluded in target_exclusions:
            if (type(excluded) is not tuple
                    or any(type(value) is not str or not value for value in excluded)
                    or len(set(excluded)) != len(excluded)):
                raise ValueError("excluded locus identities must be unique nonempty strings")
            if not hidden_locus_ids.issubset(set(excluded)):
                raise ValueError(
                    "target common context territory does not exclude every hidden locus")

    values, valid = context.values, context.valid
    expected_values = (batch, references, segments, config.common_context_width)
    expected_valid = (batch, references, segments)
    if (not isinstance(values, torch.Tensor) or tuple(values.shape) != expected_values
            or values.dtype != dtype or values.device != device):
        raise ValueError("common context values differ in target/reference/segment/width, dtype or device")
    if (not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool
            or tuple(valid.shape) != expected_valid or valid.device != device):
        raise ValueError("common context valid mask differs in axes, dtype or device")
    eligible = eligibility_by_target(
        fit_reference_eligible, batch).permute(0, 2, 1)
    if bool((valid & ~eligible).any()):
        raise ValueError("common context marks an ineligible FIT reference as valid")
    if not bool(torch.isfinite(values[valid]).all()):
        raise ValueError("valid common context values must be finite")
    clean = torch.where(valid[..., None], values, torch.zeros((), dtype=dtype, device=device))
    return clean, valid


class LearnedCarrierInteractionModel(nn.Module):
    """Apply learned pair and per-reference transformations with FIT masks."""

    def __init__(self, config: LearnedCarrierInteractionConfig):
        super().__init__()
        if not isinstance(config, LearnedCarrierInteractionConfig):
            raise ValueError("LearnedCarrierInteractionConfig required")
        self.config = config
        pair_input_width = 4 + len(config.pair_feature_kinds)
        self.psi_theta = _mlp(pair_input_width, config.psi_hidden_widths,
                              config.psi_output_width)
        self.rho_phi = _mlp(config.psi_output_width + 1 + config.common_context_width,
                            config.rho_hidden_widths,
                            config.rho_output_width)

    def _apply_psi(self, pair_input: torch.Tensor) -> torch.Tensor:
        flat = pair_input.reshape(-1, pair_input.shape[-1])
        if self.config.checkpoint_blocks and torch.is_grad_enabled():
            result = checkpoint(self.psi_theta, flat, use_reentrant=False)
        else:
            result = self.psi_theta(flat)
        return result.reshape(*pair_input.shape[:-1], self.config.psi_output_width)

    def forward(self,
                target_dosage: torch.Tensor,
                reference_dosage: torch.Tensor,
                target_called: torch.Tensor,
                reference_called: torch.Tensor,
                target_hidden: torch.Tensor,
                pairs: torch.Tensor,
                pair_segment: torch.Tensor,
                fit_reference_eligible: torch.Tensor,
                reference_roles: Sequence[str],
                target_ids: Sequence[str],
                reference_ids: Sequence[str],
                pair_features: Optional[torch.Tensor] = None,
                locus_ids: Optional[Sequence[str]] = None,
                segment_ids: Optional[Sequence[str]] = None,
                common_context: Optional[PairCommonContext] = None,
                ) -> Union[CarrierInteractionOutput, ContextualCarrierInteractionOutput]:
        validation_config = self.config.validation_config()
        prepared = prepare_carrier_inputs(
            target_dosage, reference_dosage, target_called, reference_called,
            target_hidden, pairs, pair_segment, fit_reference_eligible,
            reference_roles, validation_config, target_ids, reference_ids,
            pair_features)
        target_visible, target_clean, reference_clean, segments = prepared
        eligible_by_target = eligibility_by_target(
            fit_reference_eligible, target_dosage.shape[0])
        parameter = next(self.parameters())
        if parameter.device != target_dosage.device or parameter.dtype != target_dosage.dtype:
            raise ValueError("model parameters and dosages must share device and dtype")
        prepared_context = _prepare_common_context(
            self.config, common_context, target_ids, reference_ids,
            locus_ids, segment_ids, target_hidden, fit_reference_eligible,
            target_dosage.dtype, target_dosage.device)

        batch = target_dosage.shape[0]
        features = target_dosage.new_zeros(
            (batch, segments, self.config.rho_output_width))
        pair_counts = torch.zeros((batch, segments), dtype=torch.int64,
                                  device=target_dosage.device)
        counts_by_reference = []
        original_pair_counts = torch.zeros((batch, segments), dtype=torch.int64,
                                           device=target_dosage.device)
        original_counts_by_reference = []
        for reference_start in range(0, reference_dosage.shape[0],
                                     self.config.reference_block_size):
            reference_stop = min(reference_start + self.config.reference_block_size,
                                 reference_dosage.shape[0])
            block_references = reference_stop - reference_start
            values = target_dosage.new_zeros(
                (batch, block_references, segments, self.config.psi_output_width))
            counts = torch.zeros((batch, block_references, segments),
                                 dtype=torch.int64, device=target_dosage.device)
            reference_block = reference_clean[reference_start:reference_stop]
            reference_called_block = reference_called[reference_start:reference_stop]
            eligible_block = eligible_by_target[:, :, reference_start:reference_stop]

            for pair_start in range(0, pairs.shape[0], self.config.pair_block_size):
                pair_stop = min(pair_start + self.config.pair_block_size,
                                pairs.shape[0])
                block_pairs = pairs[pair_start:pair_stop]
                left, right = block_pairs[:, 0], block_pairs[:, 1]
                segment = pair_segment[pair_start:pair_stop]
                block_pairs_count = pair_stop - pair_start
                target_valid = target_visible[:, left] & target_visible[:, right]
                reference_valid = (reference_called_block[:, left]
                                   & reference_called_block[:, right])
                pair_eligible = eligible_block[:, segment, :].permute(0, 2, 1)
                valid = (target_valid[:, None, :] & reference_valid[None, :, :]
                         & pair_eligible)

                shape = (batch, block_references, block_pairs_count)
                pair_input = torch.stack((
                    target_clean[:, left][:, None, :].expand(shape),
                    target_clean[:, right][:, None, :].expand(shape),
                    reference_block[:, left][None, :, :].expand(shape),
                    reference_block[:, right][None, :, :].expand(shape),
                ), dim=-1)
                if pair_features is not None:
                    static = pair_features[pair_start:pair_stop]
                    static = static[None, None, :, :].expand(
                        batch, block_references, block_pairs_count, -1)
                    pair_input = torch.cat((pair_input, static), dim=-1)
                contribution = self._apply_psi(pair_input)
                contribution = contribution * valid[..., None].to(contribution.dtype)
                values = values.index_add(2, segment, contribution)
                counts = counts.index_add(2, segment, valid.to(torch.int64))

            original_counts_by_reference.append(counts)
            original_pair_counts = original_pair_counts + counts.sum(1)
            support = torch.log1p(counts.to(values.dtype))[..., None]
            rho_parts = [values, support]
            if prepared_context is not None:
                context_values, context_valid = prepared_context
                block_context = context_values[:, reference_start:reference_stop]
                block_context_valid = context_valid[:, reference_start:reference_stop]
                rho_parts.append(block_context)
                effective_counts = counts * block_context_valid.to(torch.int64)
            else:
                block_context_valid = None
                effective_counts = counts
            rho_input = torch.cat(rho_parts, dim=-1)
            transformed = self.rho_phi(
                rho_input.reshape(-1, rho_input.shape[-1])).reshape(
                    batch, block_references, segments,
                    self.config.rho_output_width)
            contributes = effective_counts > 0
            transformed = transformed * contributes[..., None].to(transformed.dtype)
            features = features + transformed.sum(1)
            pair_counts = pair_counts + effective_counts.sum(1)
            counts_by_reference.append(effective_counts)

        counts_by_reference = torch.cat(counts_by_reference, dim=1)
        carrier = CarrierInteractionOutput(
            features, pair_counts, counts_by_reference,
            (counts_by_reference > 0).sum(1, dtype=torch.int64),
            (fit_reference_eligible.sum(1, dtype=torch.int64)
             if fit_reference_eligible.ndim == 2
             else fit_reference_eligible.sum(2, dtype=torch.int64)))
        if prepared_context is None:
            return carrier

        original_counts_by_reference = torch.cat(original_counts_by_reference, dim=1)
        _, context_valid = prepared_context
        eligible = eligible_by_target.permute(0, 2, 1)
        context_valid_eligible = context_valid & eligible
        original_observed = original_counts_by_reference > 0
        effective_observed = counts_by_reference > 0
        support = PairCommonContextSupport(
            original_pair_counts,
            original_counts_by_reference,
            original_observed.sum(1, dtype=torch.int64),
            context_valid_eligible.sum(1, dtype=torch.int64),
            (eligible & ~context_valid).sum(1, dtype=torch.int64),
            (original_pair_counts - pair_counts),
            (original_observed & ~effective_observed).sum(1, dtype=torch.int64))
        return ContextualCarrierInteractionOutput(carrier, support)


def carrier_features_for_queries(output: Union[CarrierInteractionOutput,
                                                ContextualCarrierInteractionOutput],
                                 query_segment: torch.Tensor,
                                 ) -> CarrierQueryFeatures:
    """Gather segment features using a static query-to-segment metadata map."""
    if isinstance(output, ContextualCarrierInteractionOutput):
        output = output.carrier
    if not isinstance(output, CarrierInteractionOutput):
        raise ValueError("CarrierInteractionOutput required")
    features = output.features
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("carrier features must be B x S x D")
    batch, segments, _ = features.shape
    if (not isinstance(query_segment, torch.Tensor)
            or query_segment.dtype != torch.int64 or query_segment.ndim != 1
            or query_segment.device != features.device):
        raise ValueError("query_segment must be a same-device int64 Q-vector")
    if query_segment.numel() and not bool(
            ((query_segment >= 0) & (query_segment < segments)).all()):
        raise ValueError("query segment outside carrier segment axis")
    denominators = (output.valid_pair_counts,
                    output.valid_pair_counts_by_reference,
                    output.observed_reference_counts,
                    output.eligible_reference_counts)
    eligible_shape = tuple(output.eligible_reference_counts.shape)
    if (tuple(output.valid_pair_counts.shape) != (batch, segments)
            or tuple(output.observed_reference_counts.shape) != (batch, segments)
            or eligible_shape not in ((segments,), (batch, segments))
            or output.valid_pair_counts_by_reference.ndim != 3
            or output.valid_pair_counts_by_reference.shape[0] != batch
            or output.valid_pair_counts_by_reference.shape[2] != segments
            or any(value.device != features.device or value.dtype != torch.int64
                   for value in denominators)):
        raise ValueError("carrier denominators differ from feature axes")
    query_eligible = (output.eligible_reference_counts[query_segment]
                      if eligible_shape == (segments,)
                      else output.eligible_reference_counts[:, query_segment])
    return CarrierQueryFeatures(
        features[:, query_segment],
        output.valid_pair_counts[:, query_segment],
        output.observed_reference_counts[:, query_segment],
        query_eligible)
