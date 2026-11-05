"""Pair-free separable carrier features for masked genotype prediction.

This module is a second architecture beside ``r02_masked_carrier_integration``.
It reuses that path's authenticated target/reference tensors, masks, FIT-family
eligibility and COMMON declaration, but requires its enumerated pair tensors to
be empty.  Visible loci are traversed in genomic order by the exact recurrence
in :mod:`r02_carrier_separable`; no ``P x 2`` pair axis is constructed here.

The output is a predictive representation.  It is not phase, IBD, ancestry,
genealogy, or evidence that the chosen width or decay length is biologically
appropriate.  This module does not read stores or authorize human training.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple, Tuple

import torch
from torch import Tensor, nn

from r02_carrier_interaction_model import (
    ContextualCarrierInteractionOutput, PairCommonContextSupport,
    _prepare_common_context, carrier_features_for_queries,
)
from r02_carrier_interactions import CarrierInteractionOutput, eligibility_by_target
from r02_carrier_separable import (
    accumulate_separable_chunk, finalize_separable_state,
    initialize_separable_state,
)
from r02_masked_carrier_batch import CarrierFragmentBatch
from r02_masked_genotype_model import (
    INFORMATION_CONDITIONS, MaskedGenotypeModel, MaskedModelOutput,
    MaskedTensorChannel,
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _positive_widths(name: str, value: Tuple[int, ...]) -> None:
    _require(type(value) is tuple and value
             and all(type(width) is int and width > 0 for width in value),
             f"{name} must be a nonempty tuple of positive integers")


def _mlp(input_width: int, hidden_widths: Tuple[int, ...],
         output_width: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_width
    for width in hidden_widths:
        layers.extend((nn.Linear(current, width), nn.GELU()))
        current = width
    layers.append(nn.Linear(current, output_width))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class SeparableCarrierConfig:
    """All capacity, numerical and blocking choices are explicit.

    ``decay_length_cm`` is an initialization when
    ``learn_decay_length_cm=True`` and a fixed value otherwise.  Neither mode
    implies that the supplied value is optimal for DNABR.
    """

    reference_block_size: int
    locus_chunk_size: int
    factor_hidden_widths: Tuple[int, ...]
    factor_width: int
    rho_hidden_widths: Tuple[int, ...]
    rho_output_width: int
    correction_hidden_widths: Tuple[int, ...]
    decay_length_cm: float
    learn_decay_length_cm: bool
    common_context_width: int

    def __post_init__(self) -> None:
        for name in ("reference_block_size", "locus_chunk_size", "factor_width",
                     "rho_output_width"):
            _require(type(getattr(self, name)) is int and getattr(self, name) > 0,
                     f"positive integer {name} required")
        for name in ("factor_hidden_widths", "rho_hidden_widths",
                     "correction_hidden_widths"):
            _positive_widths(name, getattr(self, name))
        _require(type(self.decay_length_cm) in (int, float)
                 and math.isfinite(self.decay_length_cm)
                 and self.decay_length_cm > 0,
                 "finite positive decay_length_cm required")
        _require(type(self.learn_decay_length_cm) is bool,
                 "learn_decay_length_cm must be boolean")
        _require(type(self.common_context_width) is int
                 and self.common_context_width >= 0,
                 "nonnegative integer common_context_width required")


@dataclass(frozen=True)
class SeparableCarrierGeometry:
    """Linear-size visible-locus geometry for each hidden genomic segment."""

    positions_cm: Tensor                 # L
    left_flank: Tensor                   # S x L
    right_flank: Tensor                  # S x L
    context_id: str
    axis_id: str

    def check(self, rare: MaskedTensorChannel,
              carrier_batch: CarrierFragmentBatch) -> None:
        device = rare.dosage.device
        loci = rare.dosage.shape[1]
        segments = len(carrier_batch.segment_ids)
        _require(isinstance(self.positions_cm, Tensor)
                 and self.positions_cm.is_floating_point()
                 and tuple(self.positions_cm.shape) == (loci,)
                 and self.positions_cm.device == device
                 and self.positions_cm.dtype == carrier_batch.target_dosage.dtype
                 and not self.positions_cm.requires_grad
                 and torch.equal(self.positions_cm, rare.position_cm.to(
                     self.positions_cm.dtype)),
                 "separable cM positions must equal the rare-channel locus axis")
        for name, value in (("left_flank", self.left_flank),
                            ("right_flank", self.right_flank)):
            _require(isinstance(value, Tensor) and value.dtype == torch.bool
                     and tuple(value.shape) == (segments, loci)
                     and value.device == device,
                     f"{name} must be a same-device S x L boolean tensor")
        _require(not bool((self.left_flank & self.right_flank).any()),
                 "left and right genomic flanks overlap")
        for segment in range(segments):
            selected = self.left_flank[segment] | self.right_flank[segment]
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() > 1:
                positions = self.positions_cm[indices]
                _require(bool((positions[1:] >= positions[:-1]).all()),
                         "selected separable loci are not ordered by cM")
                flank_codes = self.right_flank[segment, indices].to(torch.int8)
                _require(bool((flank_codes[1:] >= flank_codes[:-1]).all()),
                         "a left-flank locus occurs after the right flank")
        for name, value in (("context_id", self.context_id),
                            ("axis_id", self.axis_id)):
            _require(type(value) is str and len(value) == 64
                     and all(letter in "0123456789abcdef" for letter in value),
                     f"{name} must be a lowercase SHA-256")


class MaskedSeparableCarrierOutput(NamedTuple):
    common_logits: Tensor
    rare_logits: Tensor
    common_valid_pair_counts: Tensor
    rare_valid_pair_counts: Tensor


class SeparableCarrierInteraction(nn.Module):
    """Create per-locus factors, run the exact recurrence, then sum FIT refs."""

    # The three scalar channels are target dose/2, reference dose/2 and their
    # product.  The remaining channels reuse the base model's locus descriptor.
    DOSAGE_FEATURE_WIDTH = 3

    def __init__(self, descriptor_width: int, config: SeparableCarrierConfig):
        super().__init__()
        _require(type(descriptor_width) is int and descriptor_width > 0,
                 "positive descriptor_width required")
        _require(isinstance(config, SeparableCarrierConfig),
                 "SeparableCarrierConfig required")
        self.config = config
        factor_input = descriptor_width + self.DOSAGE_FEATURE_WIDTH
        self.source_factor = _mlp(
            factor_input, config.factor_hidden_widths, config.factor_width)
        self.destination_factor = _mlp(
            factor_input, config.factor_hidden_widths, config.factor_width)
        rho_input = (3 * config.factor_width + 3
                     + config.common_context_width)
        self.rho_phi = _mlp(
            rho_input, config.rho_hidden_widths, config.rho_output_width)
        log_decay = torch.tensor(math.log(config.decay_length_cm),
                                 dtype=torch.float32)
        if config.learn_decay_length_cm:
            self.log_decay_length_cm = nn.Parameter(log_decay)
        else:
            self.register_buffer("log_decay_length_cm", log_decay)

    @property
    def decay_length_cm(self) -> Tensor:
        return torch.exp(self.log_decay_length_cm)

    def _factor_input(self, descriptor: Tensor, target_clean: Tensor,
                      reference_clean: Tensor, indices: Tensor) -> Tensor:
        batch, references = target_clean.shape[0], reference_clean.shape[0]
        selected_descriptor = descriptor[:, indices]
        target = target_clean[:, indices, None] / 2
        reference = reference_clean[:, indices][None, ..., None] / 2
        target = target[:, None].expand(batch, references, -1, -1)
        reference = reference.expand(batch, references, -1, -1)
        metadata = selected_descriptor[:, None].expand(
            batch, references, -1, -1)
        return torch.cat((metadata, target, reference, target * reference), dim=-1)

    def forward(self, descriptor: Tensor, rare: MaskedTensorChannel,
                carrier_batch: CarrierFragmentBatch,
                geometry: SeparableCarrierGeometry,
                ) -> CarrierInteractionOutput | ContextualCarrierInteractionOutput:
        config = self.config
        batch_size, loci = carrier_batch.target_dosage.shape
        references = carrier_batch.reference_dosage.shape[0]
        segments = len(carrier_batch.segment_ids)
        _require(tuple(descriptor.shape[:2]) == (batch_size, loci)
                 and descriptor.shape[2] > 0,
                 "locus descriptor differs from EVENT person/locus axes")
        parameter = next(self.parameters())
        _require(parameter.device == carrier_batch.target_dosage.device
                 and parameter.dtype == carrier_batch.target_dosage.dtype
                 and descriptor.device == parameter.device
                 and descriptor.dtype == parameter.dtype,
                 "separable model, descriptor and dosage dtype/device differ")

        target_visible = (carrier_batch.target_called
                          & ~carrier_batch.rare_structural_hidden)
        for name, values in (
                ("visible target", carrier_batch.target_dosage[target_visible]),
                ("called reference", carrier_batch.reference_dosage[
                    carrier_batch.reference_called])):
            _require(bool(torch.isfinite(values).all()),
                     f"nonfinite {name} dosage")
            _require(bool(((values == 0) | (values == 1)
                           | (values == 2)).all()),
                     f"{name} dosage must be 0, 1, or 2")
        zero = carrier_batch.target_dosage.new_zeros(())
        target_clean = torch.where(
            target_visible, carrier_batch.target_dosage, zero)
        reference_clean = torch.where(
            carrier_batch.reference_called,
            carrier_batch.reference_dosage, zero)
        eligible = eligibility_by_target(
            carrier_batch.fit_reference_eligible, batch_size)
        prepared_context = _prepare_common_context(
            config, carrier_batch.common_context,
            carrier_batch.target_ids, carrier_batch.reference_ids,
            carrier_batch.locus_ids, carrier_batch.segment_ids,
            carrier_batch.rare_structural_hidden,
            carrier_batch.fit_reference_eligible,
            carrier_batch.target_dosage.dtype,
            carrier_batch.target_dosage.device)

        features = carrier_batch.target_dosage.new_zeros(
            (batch_size, segments, config.rho_output_width))
        effective_counts_by_reference = torch.zeros(
            (batch_size, references, segments), dtype=torch.int64,
            device=parameter.device)
        original_counts_by_reference = torch.zeros_like(
            effective_counts_by_reference)
        valid_locus = (target_visible[:, None, :]
                       & carrier_batch.reference_called[None, :, :]
                       & rare.locus_valid[None, None, :])

        for reference_start in range(0, references,
                                     config.reference_block_size):
            reference_stop = min(reference_start + config.reference_block_size,
                                 references)
            block_reference = reference_clean[reference_start:reference_stop]
            block_valid = valid_locus[:, reference_start:reference_stop]
            block_count = reference_stop - reference_start
            for segment in range(segments):
                selected_mask = (geometry.left_flank[segment]
                                 | geometry.right_flank[segment])
                selected = torch.nonzero(selected_mask, as_tuple=False).flatten()
                if selected.numel() < 2:
                    continue
                block_eligible = eligible[
                    :, segment, reference_start:reference_stop]
                state = initialize_separable_state(
                    context_id=f"{geometry.context_id}:{carrier_batch.segment_ids[segment]}",
                    axis_id=f"{geometry.axis_id}:{carrier_batch.segment_ids[segment]}",
                    total_loci=int(selected.numel()),
                    feature_width=config.factor_width,
                    decay_length_cm=self.decay_length_cm,
                    person_reference_eligible=block_eligible)
                for start in range(0, int(selected.numel()),
                                   config.locus_chunk_size):
                    stop = min(start + config.locus_chunk_size,
                               int(selected.numel()))
                    locus_indices = selected[start:stop]
                    factor_input = self._factor_input(
                        descriptor, target_clean, block_reference, locus_indices)
                    source = self.source_factor(factor_input)
                    destination = self.destination_factor(factor_input)
                    state = accumulate_separable_chunk(
                        state, source, destination,
                        geometry.positions_cm[locus_indices],
                        block_valid[:, :, locus_indices],
                        geometry.left_flank[segment, locus_indices],
                        geometry.right_flank[segment, locus_indices],
                        context_id=state.context_id, axis_id=state.axis_id,
                        start_index=start)
                recurrence = finalize_separable_state(
                    state, context_id=state.context_id, axis_id=state.axis_id)
                values = recurrence.features.reshape(
                    batch_size, block_count, 3 * config.factor_width)
                original_counts = recurrence.valid_pair_counts
                original_total = original_counts.sum(2)
                original_counts_by_reference[
                    :, reference_start:reference_stop, segment] = original_total
                rho_parts = [values, torch.log1p(
                    original_counts.to(values.dtype))]
                if prepared_context is None:
                    effective_total = original_total
                else:
                    context_values, context_valid = prepared_context
                    rho_parts.append(context_values[
                        :, reference_start:reference_stop, segment])
                    effective_total = original_total * context_valid[
                        :, reference_start:reference_stop, segment].to(torch.int64)
                transformed = self.rho_phi(torch.cat(rho_parts, dim=-1))
                transformed = transformed * (effective_total > 0)[..., None].to(
                    transformed.dtype)
                features[:, segment] = (features[:, segment]
                                        + transformed.sum(1))
                effective_counts_by_reference[
                    :, reference_start:reference_stop, segment] = effective_total

        effective_counts = effective_counts_by_reference.sum(1)
        carrier = CarrierInteractionOutput(
            features, effective_counts, effective_counts_by_reference,
            (effective_counts_by_reference > 0).sum(1, dtype=torch.int64),
            carrier_batch.fit_reference_eligible.sum(2, dtype=torch.int64))
        if prepared_context is None:
            return carrier

        _, context_valid = prepared_context
        eligible_brs = eligible.permute(0, 2, 1)
        original_counts = original_counts_by_reference.sum(1)
        original_observed = original_counts_by_reference > 0
        effective_observed = effective_counts_by_reference > 0
        support = PairCommonContextSupport(
            original_counts,
            original_counts_by_reference,
            original_observed.sum(1, dtype=torch.int64),
            (context_valid & eligible_brs).sum(1, dtype=torch.int64),
            (eligible_brs & ~context_valid).sum(1, dtype=torch.int64),
            original_counts - effective_counts,
            (original_observed & ~effective_observed).sum(1, dtype=torch.int64))
        return ContextualCarrierInteractionOutput(carrier, support)


class MaskedSeparableCarrierGenotypeModel(nn.Module):
    """Add a support-gated separable correction to the existing local logits."""

    def __init__(self, base: MaskedGenotypeModel,
                 config: SeparableCarrierConfig):
        super().__init__()
        _require(isinstance(base, MaskedGenotypeModel),
                 "MaskedGenotypeModel base required")
        _require(isinstance(config, SeparableCarrierConfig),
                 "SeparableCarrierConfig required")
        self.base = base
        self.separable_config = config
        self.integration = SeparableCarrierInteraction(base.config.width, config)
        head_input = base.config.width + config.rho_output_width
        self.common_correction = _mlp(
            head_input, config.correction_hidden_widths, 3)
        self.rare_correction = _mlp(
            head_input, config.correction_hidden_widths, 3)
        for head in (self.common_correction, self.rare_correction):
            last = head[-1]
            _require(isinstance(last, nn.Linear),
                     "correction must end in Linear")
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @property
    def config(self):
        return self.base.config

    def _correct(self, channel: MaskedTensorChannel, logits: Tensor,
                 features: Tensor, support: Tensor,
                 head: nn.Module) -> tuple[Tensor, Tensor]:
        descriptor = self.base.locus_descriptor(channel)
        _require(tuple(features.shape[:2]) == tuple(descriptor.shape[:2])
                 and features.shape[2]
                 == self.separable_config.rho_output_width,
                 "separable/query feature axes differ")
        _require(support.dtype == torch.int64
                 and tuple(support.shape) == tuple(descriptor.shape[:2]),
                 "separable query support axes differ")
        raw = head(torch.cat((descriptor, features), dim=-1))
        correction = raw * (support > 0)[..., None].to(raw.dtype)
        return logits + correction, support

    def forward(self, common: MaskedTensorChannel, rare: MaskedTensorChannel,
                *, information_condition: str,
                carrier_batch: CarrierFragmentBatch,
                geometry: SeparableCarrierGeometry,
                ) -> MaskedSeparableCarrierOutput:
        _require(information_condition in INFORMATION_CONDITIONS,
                 "unknown rare-information condition")
        from r02_masked_genotype_update import MaskedFragment, MaskedTargets
        dummy_common = MaskedTargets(
            torch.full_like(common.dosage, -1, dtype=torch.int64),
            torch.zeros_like(common.genotype_visible), common.locus_valid.clone())
        dummy_rare = MaskedTargets(
            torch.full_like(rare.dosage, -1, dtype=torch.int64),
            torch.zeros_like(rare.genotype_visible), rare.locus_valid.clone())
        shell = MaskedFragment(common, rare, dummy_common, dummy_rare,
                               "label-free-separable-forward-shell")
        carrier_batch.check(
            shell, common_context_width=self.separable_config.common_context_width)
        _require(carrier_batch.pairs.numel() == 0
                 and carrier_batch.pair_segment.numel() == 0
                 and carrier_batch.pair_features is None,
                 "separable architecture requires empty enumerated pair tensors")
        geometry.check(rare, carrier_batch)
        local: MaskedModelOutput = self.base(
            common, rare, information_condition=information_condition)
        zeros_common = torch.zeros_like(common.dosage, dtype=torch.int64)
        zeros_rare = torch.zeros_like(rare.dosage, dtype=torch.int64)
        if information_condition != "full":
            return MaskedSeparableCarrierOutput(
                local.common_logits, local.rare_logits,
                zeros_common, zeros_rare)

        descriptor = self.base.locus_descriptor(rare)
        output = self.integration(descriptor, rare, carrier_batch, geometry)
        common_query = carrier_features_for_queries(
            output, carrier_batch.common_query_segment)
        rare_query = carrier_features_for_queries(
            output, carrier_batch.rare_query_segment)
        common_logits, common_support = self._correct(
            common, local.common_logits, common_query.features,
            common_query.valid_pair_counts, self.common_correction)
        rare_logits, rare_support = self._correct(
            rare, local.rare_logits, rare_query.features,
            rare_query.valid_pair_counts, self.rare_correction)
        return MaskedSeparableCarrierOutput(
            common_logits, rare_logits, common_support, rare_support)

    def branch_parameter_count(self) -> int:
        modules = (self.integration, self.common_correction,
                   self.rare_correction)
        return sum(parameter.numel() for module in modules
                   for parameter in module.parameters())
