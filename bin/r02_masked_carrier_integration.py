"""Add carrier interactions to the existing masked-genotype logits.

The carrier branch is active only for the historical ``full`` information
condition because EVENT pair dosages are rare-genotype information.  The
``count_only`` and ``no_rare`` conditions retain their original semantics.
This module chooses no cohort, loci, pair map, widths or training budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import torch
from torch import Tensor, nn

from r02_carrier_interaction_model import (
    LearnedCarrierInteractionConfig, LearnedCarrierInteractionModel,
    carrier_features_for_queries,
)
from r02_masked_carrier_batch import CarrierFragmentBatch
from r02_masked_genotype_model import (
    INFORMATION_CONDITIONS, MaskedGenotypeModel, MaskedModelOutput,
    MaskedTensorChannel,
)


MODES = frozenset(("carrier", "baseline_plus"))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _mlp(input_width: int, hidden_widths: Tuple[int, ...], output_width: int):
    layers: list[nn.Module] = []
    current = input_width
    for width in hidden_widths:
        layers.extend((nn.Linear(current, width), nn.GELU()))
        current = width
    layers.append(nn.Linear(current, output_width))
    return nn.Sequential(*layers)


def _resize_descriptor(values: Tensor, width: int) -> Tensor:
    """Deterministic local-only basis with no learned locus identity table."""
    if values.shape[-1] >= width:
        return values[..., :width]
    copies = (width + values.shape[-1] - 1) // values.shape[-1]
    return values.repeat_interleave(copies, dim=-1)[..., :width]


class _CapacityMatchedLocalControl(nn.Module):
    """Use exactly the psi/rho parameter shapes on existing locus descriptors.

    This is a capacity control, not a carrier model: it never receives a
    reference genotype, reference identity, pair support or COMMON value.
    """

    def __init__(self, config: LearnedCarrierInteractionConfig):
        super().__init__()
        self.config = config
        self.psi_theta = _mlp(4 + len(config.pair_feature_kinds),
                              config.psi_hidden_widths,
                              config.psi_output_width)
        self.rho_phi = _mlp(config.psi_output_width + 1
                            + config.common_context_width,
                            config.rho_hidden_widths,
                            config.rho_output_width)

    def forward(self, descriptor: Tensor) -> Tensor:
        psi_input = _resize_descriptor(
            descriptor, 4 + len(self.config.pair_feature_kinds))
        psi = self.psi_theta(psi_input)
        extra = _resize_descriptor(
            descriptor, 1 + self.config.common_context_width)
        return self.rho_phi(torch.cat((psi, extra), dim=-1))


@dataclass(frozen=True)
class MaskedCarrierIntegrationConfig:
    mode: str
    correction_hidden_widths: Tuple[int, ...]

    def __post_init__(self):
        _require(self.mode in MODES, "carrier or baseline_plus mode required")
        _require(type(self.correction_hidden_widths) is tuple
                 and self.correction_hidden_widths
                 and all(type(value) is int and value > 0
                         for value in self.correction_hidden_widths),
                 "explicit nonempty positive correction hidden widths required")


class MaskedCarrierModelOutput(NamedTuple):
    common_logits: Tensor
    rare_logits: Tensor
    common_valid_pair_counts: Tensor
    rare_valid_pair_counts: Tensor


class MaskedCarrierGenotypeModel(nn.Module):
    """Add a zero-initialized, support-gated correction to local logits."""

    def __init__(self, base: MaskedGenotypeModel,
                 interaction_config: LearnedCarrierInteractionConfig,
                 integration_config: MaskedCarrierIntegrationConfig):
        super().__init__()
        _require(isinstance(base, MaskedGenotypeModel),
                 "MaskedGenotypeModel base required")
        _require(isinstance(interaction_config, LearnedCarrierInteractionConfig),
                 "LearnedCarrierInteractionConfig required")
        _require(isinstance(integration_config, MaskedCarrierIntegrationConfig),
                 "MaskedCarrierIntegrationConfig required")
        self.base = base
        self.interaction_config = interaction_config
        self.integration_config = integration_config
        self.integration = (LearnedCarrierInteractionModel(interaction_config)
                            if integration_config.mode == "carrier" else
                            _CapacityMatchedLocalControl(interaction_config))
        head_input = base.config.width + interaction_config.rho_output_width
        self.common_correction = _mlp(
            head_input, integration_config.correction_hidden_widths, 3)
        self.rare_correction = _mlp(
            head_input, integration_config.correction_hidden_widths, 3)
        for head in (self.common_correction, self.rare_correction):
            last = head[-1]
            _require(isinstance(last, nn.Linear), "correction must end in Linear")
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @property
    def config(self):
        """Compatibility with the existing masked streaming update validator."""
        return self.base.config

    def _correct(self, channel: MaskedTensorChannel, local_logits: Tensor,
                 interaction_features: Tensor, support: Tensor,
                 head: nn.Module) -> tuple[Tensor, Tensor]:
        descriptor = self.base.locus_descriptor(channel)
        _require(interaction_features.shape[:2] == descriptor.shape[:2]
                 and interaction_features.shape[-1]
                 == self.interaction_config.rho_output_width,
                 "interaction/query descriptor axes differ")
        _require(support.dtype == torch.int64
                 and support.shape == descriptor.shape[:2],
                 "query support axis/dtype differs")
        raw = head(torch.cat((descriptor, interaction_features), dim=-1))
        # Gate after the complete head: bias cannot create a correction when no
        # admissible observed reference/pair supports the query.
        correction = raw * (support > 0)[..., None].to(raw.dtype)
        return local_logits + correction, support

    def _carrier_zero_factor(self, like: Tensor) -> Tensor:
        """Exact-one value that gives every carrier parameter a zero gradient.

        Adam distinguishes ``grad=None`` from an explicit zero gradient after
        momentum exists.  The empty-pair path therefore keeps all carrier-only
        parameters in the graph while evaluating their contribution to zero.
        """
        zero = like.new_zeros(())
        modules = (self.integration, self.common_correction,
                   self.rare_correction)
        for module in modules:
            for parameter in module.parameters():
                zero = zero + parameter.reshape(-1)[0].to(like.dtype) * 0
        return zero + 1

    def forward(self, common: MaskedTensorChannel, rare: MaskedTensorChannel,
                *, information_condition: str,
                carrier_batch: CarrierFragmentBatch) -> MaskedCarrierModelOutput:
        _require(information_condition in INFORMATION_CONDITIONS,
                 "unknown rare-information condition")
        # Construct a target-free shell solely for mask/axis checks. The batch
        # boundary never imports labels into the model.
        from r02_masked_genotype_update import MaskedFragment, MaskedTargets
        dummy_common = MaskedTargets(
            torch.full_like(common.dosage, -1, dtype=torch.int64),
            torch.zeros_like(common.genotype_visible),
            common.locus_valid.clone())
        dummy_rare = MaskedTargets(
            torch.full_like(rare.dosage, -1, dtype=torch.int64),
            torch.zeros_like(rare.genotype_visible), rare.locus_valid.clone())
        shell = MaskedFragment(common, rare, dummy_common, dummy_rare,
                               "label-free-forward-shell")
        carrier_batch.check(shell,
            common_context_width=self.interaction_config.common_context_width)
        local: MaskedModelOutput = self.base(
            common, rare, information_condition=information_condition)

        zeros_common = torch.zeros_like(common.dosage, dtype=torch.int64)
        zeros_rare = torch.zeros_like(rare.dosage, dtype=torch.int64)
        if information_condition != "full":
            return MaskedCarrierModelOutput(local.common_logits, local.rare_logits,
                                            zeros_common, zeros_rare)

        if (self.integration_config.mode == "carrier"
                and carrier_batch.pairs.shape[0] == 0):
            factor = self._carrier_zero_factor(local.common_logits)
            return MaskedCarrierModelOutput(
                local.common_logits * factor, local.rare_logits * factor,
                zeros_common, zeros_rare)

        if self.integration_config.mode == "baseline_plus":
            common_descriptor = self.base.locus_descriptor(common)
            rare_descriptor = self.base.locus_descriptor(rare)
            common_features = self.integration(common_descriptor)
            rare_features = self.integration(rare_descriptor)
            common_support = common.locus_valid[None].expand(
                common.dosage.shape[0], -1).to(torch.int64)
            rare_support = rare.locus_valid[None].expand(
                rare.dosage.shape[0], -1).to(torch.int64)
        else:
            output = self.integration(
                target_dosage=carrier_batch.target_dosage,
                reference_dosage=carrier_batch.reference_dosage,
                target_called=carrier_batch.target_called,
                reference_called=carrier_batch.reference_called,
                target_hidden=carrier_batch.rare_structural_hidden,
                pairs=carrier_batch.pairs,
                pair_segment=carrier_batch.pair_segment,
                fit_reference_eligible=carrier_batch.fit_reference_eligible,
                reference_roles=carrier_batch.reference_roles,
                target_ids=carrier_batch.target_ids,
                reference_ids=carrier_batch.reference_ids,
                pair_features=carrier_batch.pair_features,
                locus_ids=carrier_batch.locus_ids,
                segment_ids=carrier_batch.segment_ids,
                common_context=carrier_batch.common_context)
            common_query = carrier_features_for_queries(
                output, carrier_batch.common_query_segment)
            rare_query = carrier_features_for_queries(
                output, carrier_batch.rare_query_segment)
            common_features = common_query.features
            rare_features = rare_query.features
            common_support = common_query.valid_pair_counts
            rare_support = rare_query.valid_pair_counts

        common_logits, common_support = self._correct(
            common, local.common_logits, common_features, common_support,
            self.common_correction)
        rare_logits, rare_support = self._correct(
            rare, local.rare_logits, rare_features, rare_support,
            self.rare_correction)
        return MaskedCarrierModelOutput(common_logits, rare_logits,
                                        common_support, rare_support)

    def branch_parameter_count(self) -> int:
        return sum(value.numel() for value in self.integration.parameters())
