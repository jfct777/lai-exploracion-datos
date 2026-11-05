"""Local genotype predictors for deliberately hidden person-by-cell dosages.

The model receives metadata and visible genotypes only.  Targets remain in the
training caller.  It predicts three logits (diploid doses 0, 1 and 2) at every
declared locus, including loci whose genotype is hidden.  This module chooses
no scientific radius, width, masking schedule, population or learning budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genomic_context_runtime import Sites, local_mean
from r02_local_attention_blocks import ContextExecution


FAMILIES = frozenset(("local_deepsets", "local_attention"))
INFORMATION_CONDITIONS = frozenset(("full", "count_only", "no_rare"))
CHANNELS = frozenset(("common", "rare"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class MaskedModelConfig:
    """Every capacity, scale and execution choice must be supplied explicitly."""

    family: str
    width: int
    projection_depth: int
    attention_depth: int
    dropout: float
    heads: int | None
    common_radius_cm: float
    rare_radius_cm: float
    position_bp_scale: float
    position_cm_scale: float
    max_loci_per_channel: int
    query_block_size: int
    max_mask_elements: int
    checkpoint_blocks: bool

    def __post_init__(self) -> None:
        _require(self.family in FAMILIES, "unknown masked-genotype model family")
        for name in ("width", "projection_depth", "max_loci_per_channel",
                     "query_block_size", "max_mask_elements"):
            value = getattr(self, name)
            _require(type(value) is int and value > 0, f"positive integer {name} required")
        for name in ("common_radius_cm", "rare_radius_cm", "position_bp_scale",
                     "position_cm_scale"):
            value = getattr(self, name)
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     f"positive finite {name} required")
        _require(type(self.dropout) in (int, float) and math.isfinite(self.dropout)
                 and 0 <= self.dropout < 1, "dropout must be in [0,1)")
        _require(type(self.checkpoint_blocks) is bool, "checkpoint_blocks must be boolean")
        if self.family == "local_deepsets":
            _require(self.attention_depth == 0 and self.heads is None,
                     "Deep Sets requires attention_depth=0 and heads=None")
        else:
            _require(type(self.attention_depth) is int and self.attention_depth > 0,
                     "local attention requires positive attention_depth")
            _require(type(self.heads) is int and self.heads > 0 and self.width % self.heads == 0,
                     "attention heads must divide width")
            _require(self.dropout == 0,
                     "bounded exact local-attention execution requires dropout=0")

    @property
    def execution(self) -> ContextExecution:
        return ContextExecution(self.query_block_size, self.max_mask_elements,
                                self.checkpoint_blocks)

    def required_halo_cm(self, channel: str) -> float:
        """Maximum cM reach needed beyond an owned target interval.

        Deep Sets aggregates visible loci once.  Local attention first applies
        ``attention_depth`` same-radius message-passing blocks, then aggregates
        the resulting visible representations once toward each target locus.
        """
        _require(channel in CHANNELS, "common or rare channel required")
        radius = self.common_radius_cm if channel == "common" else self.rare_radius_cm
        hops = 1 if self.family == "local_deepsets" else self.attention_depth + 1
        return hops * radius


@dataclass(frozen=True)
class MaskedTensorChannel:
    """One channel after EB masking; metadata is never erased by visibility."""

    dosage: Tensor
    genotype_visible: Tensor
    locus_valid: Tensor
    chromosome: Tensor
    position_bp: Tensor
    position_cm: Tensor
    ref: Tensor
    alt: Tensor
    counted_allele: Tensor
    cell_index: Tensor
    channel: str
    person_ids: tuple[str, ...]
    plan_identity: str
    lineage: str

    def check(self, max_loci: int) -> None:
        _require(self.channel in CHANNELS, "common or rare channel required")
        _require(self.dosage.ndim == 2 and self.dosage.shape[0] > 0
                 and 0 <= self.dosage.shape[1] <= max_loci, "dosage axes or locus limit differ")
        batch, loci = self.dosage.shape
        _require(self.genotype_visible.shape == (batch, loci)
                 and self.genotype_visible.dtype == torch.bool,
                 "genotype_visible must be boolean person by locus")
        _require(self.dosage.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64),
                 "integer dosage required")
        _require(bool(((self.dosage[self.genotype_visible] >= 0)
                       & (self.dosage[self.genotype_visible] <= 2)).all()),
                 "visible dosage must be 0, 1 or 2")
        metadata = (self.locus_valid, self.chromosome, self.position_bp, self.position_cm,
                    self.ref, self.alt, self.counted_allele, self.cell_index)
        _require(all(value.shape == (loci,) for value in metadata),
                 "one metadata row per locus required")
        _require(self.locus_valid.dtype == torch.bool and self.position_cm.is_floating_point(),
                 "locus_valid must be boolean and position_cm floating")
        _require(self.chromosome.dtype in (torch.int8, torch.uint8, torch.int16,
                                           torch.int32, torch.int64)
                 and self.position_bp.dtype in (torch.int32, torch.int64),
                 "integer chromosome and bp required")
        _require(all(value.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64)
                     for value in (self.ref, self.alt, self.counted_allele, self.cell_index)),
                 "integer allele/cell metadata required")
        valid = self.locus_valid
        _require(bool(torch.isfinite(self.position_cm[valid]).all())
                 and bool(((self.chromosome[valid] >= 1) & (self.chromosome[valid] <= 22)).all())
                 and bool((self.position_bp[valid] > 0).all()),
                 "valid autosomal coordinates required")
        _require(bool(((self.ref[valid] >= 0) & (self.ref[valid] < 4)
                       & (self.alt[valid] >= 0) & (self.alt[valid] < 4)
                       & (self.ref[valid] != self.alt[valid])).all()),
                 "valid REF/ALT nucleotide codes required")
        _require(bool(((self.counted_allele[valid] >= 0)
                       & (self.counted_allele[valid] <= 1)).all()),
                 "counted allele must identify REF or ALT")
        if self.channel == "common":
            _require(bool((self.counted_allele[valid] == 1).all()),
                     "COMMON must retain ALT as the counted allele")
        _require(len(self.person_ids) == batch and len(set(self.person_ids)) == batch
                 and all(isinstance(value, str) and value for value in self.person_ids),
                 "unique nonempty person IDs must match dosage rows")
        _require(isinstance(self.plan_identity, str) and len(self.plan_identity) == 64,
                 "mask-plan SHA-256 identity required")
        _require(isinstance(self.lineage, str) and self.lineage,
                 "authenticated channel lineage required")
        device = self.dosage.device
        _require(all(value.device == device for value in (self.genotype_visible,) + metadata),
                 "channel tensors must share a device")

    @classmethod
    def from_masked_channel(cls, masked, *, person_ids: tuple[str, ...], lineage: str,
                            device: torch.device | str) -> "MaskedTensorChannel":
        """Convert EB's numpy ``MaskedChannel`` without importing its targets."""
        values: Mapping[str, np.ndarray] = masked.inputs
        tensor = lambda name: torch.as_tensor(np.asarray(values[name]), device=device)
        return cls(dosage=tensor("dosage"), genotype_visible=tensor("genotype_visible"),
                   locus_valid=tensor("locus_valid"), chromosome=tensor("chromosome"),
                   position_bp=tensor("position_bp"), position_cm=tensor("position_cm"),
                   ref=tensor("ref"), alt=tensor("alt"),
                   counted_allele=tensor("counted_allele"), cell_index=tensor("cell_index"),
                   channel=masked.channel, person_ids=tuple(person_ids),
                   plan_identity=masked.plan_identity, lineage=lineage)


@dataclass(frozen=True)
class MaskedModelOutput:
    common_logits: Tensor
    rare_logits: Tensor


def _mlp(input_width: int, output_width: int, hidden_width: int, depth: int,
         dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_width
    for _ in range(depth):
        layers.extend((nn.Linear(current, hidden_width), nn.GELU(), nn.Dropout(dropout)))
        current = hidden_width
    layers.append(nn.Linear(current, output_width))
    return nn.Sequential(*layers)


class _ExactLocalAttention(nn.Module):
    """Reuse the bounded, exact ``LocalAttention`` implementation by channel."""

    def __init__(self, config: MaskedModelConfig, radius: float):
        super().__init__()
        from genomic_context_runtime import LocalAttention
        proxy = SimpleNamespace(rare_radius_cm=radius, heads=config.heads,
                                width=config.width, dropout=config.dropout)
        self.blocks = nn.ModuleList(LocalAttention(proxy, config.execution)
                                    for _ in range(config.attention_depth))

    def forward(self, values: Tensor, sites: Sites) -> Tensor:
        for block in self.blocks:
            values = block(values, sites)
        return values


class MaskedGenotypeModel(nn.Module):
    """Predict 0/1/2 at common and rare loci from local visible context."""

    # 22 chromosome + REF4 + ALT4 + counted allele2 + bp + cM + channel2.
    METADATA_WIDTH = 36

    def __init__(self, config: MaskedModelConfig):
        super().__init__()
        self.config = config
        source_width = self.METADATA_WIDTH + 2  # visible dose/2 and explicit visibility
        self.query_encoder = _mlp(self.METADATA_WIDTH, config.width, config.width,
                                  config.projection_depth, config.dropout)
        self.common_encoder = _mlp(source_width, config.width, config.width,
                                   config.projection_depth, config.dropout)
        self.rare_encoder = _mlp(source_width, config.width, config.width,
                                 config.projection_depth, config.dropout)
        self.rare_count_encoder = _mlp(3, config.width, config.width,
                                       config.projection_depth, config.dropout)
        if config.family == "local_attention":
            self.common_attention = _ExactLocalAttention(config, config.common_radius_cm)
            self.rare_attention = _ExactLocalAttention(config, config.rare_radius_cm)
        else:
            self.common_attention = self.rare_attention = None
        self.head = _mlp(3 * config.width + 2, 3, config.width,
                         config.projection_depth, config.dropout)

    def _metadata(self, channel: MaskedTensorChannel) -> Tensor:
        valid = channel.locus_valid
        chrom = F.one_hot(channel.chromosome.clamp(1, 22).to(torch.int64) - 1, 22)
        ref = F.one_hot(channel.ref.clamp(0, 3).to(torch.int64), 4)
        alt = F.one_hot(channel.alt.clamp(0, 3).to(torch.int64), 4)
        counted = F.one_hot(channel.counted_allele.clamp(0, 1).to(torch.int64), 2)
        marker = torch.tensor((1., 0.) if channel.channel == "common" else (0., 1.),
                              device=channel.dosage.device).expand(len(valid), 2)
        values = torch.cat((chrom, ref, alt, counted,
                            (channel.position_bp / self.config.position_bp_scale)[:, None],
                            (channel.position_cm / self.config.position_cm_scale)[:, None], marker), -1)
        values = values.to(next(self.parameters()).dtype)
        return values.unsqueeze(0).expand(channel.dosage.shape[0], -1, -1)

    def locus_descriptor(self, channel: MaskedTensorChannel) -> Tensor:
        """Return the existing learned locus/channel identity representation.

        This public method exposes exactly the query representation already used
        by the historical prediction head.  It does not inspect dosage, target
        labels or reference people and does not add a second locus embedding.
        """
        channel.check(self.config.max_loci_per_channel)
        return self.query_encoder(self._metadata(channel))

    def _sites(self, channel: MaskedTensorChannel, features: Tensor, visible_only: bool) -> Sites:
        batch = channel.dosage.shape[0]
        valid = channel.locus_valid.unsqueeze(0).expand(batch, -1)
        if visible_only:
            valid = valid & channel.genotype_visible
        return Sites(features=features,
                     cm=channel.position_cm.unsqueeze(0).expand(batch, -1),
                     chrom=channel.chromosome.unsqueeze(0).expand(batch, -1).to(torch.int64),
                     valid=valid)

    def _source_values(self, channel: MaskedTensorChannel, encoder: nn.Module) -> tuple[Tensor, Sites]:
        metadata = self._metadata(channel)
        visible = channel.genotype_visible
        source = torch.cat((metadata, (channel.dosage.to(metadata.dtype) / 2)[..., None],
                            visible.to(metadata.dtype)[..., None]), -1)
        values = encoder(source)
        sites = self._sites(channel, values, visible_only=True)
        values = torch.where(sites.valid[..., None], values, 0.)
        return values, sites

    def _context_values(self, common: MaskedTensorChannel, rare: MaskedTensorChannel,
                        condition: str):
        common_values, common_sites = self._source_values(common, self.common_encoder)
        if self.common_attention is not None and common.dosage.shape[1]:
            common_values = self.common_attention(common_values, common_sites)
        if condition == "full":
            rare_values, rare_sites = self._source_values(rare, self.rare_encoder)
            if self.rare_attention is not None and rare.dosage.shape[1]:
                rare_values = self.rare_attention(rare_values, rare_sites)
        else:
            # COUNT_ONLY must not execute the per-locus rare encoder/attention;
            # NO_RARE must not even inspect rare dose or missingness for context.
            # Locus metadata remains solely to locate rare output queries.
            shape = (*rare.dosage.shape, self.config.width)
            rare_values = next(self.parameters()).new_zeros(shape)
            rare_sites = self._sites(rare, rare_values,
                                     visible_only=(condition == "count_only"))
            if condition == "no_rare":
                rare_sites = Sites(rare_sites.features, rare_sites.cm, rare_sites.chrom,
                                   torch.zeros_like(rare_sites.valid))
        return common_values, common_sites, rare_values, rare_sites

    def _local_mean(self, values, query, source, radius):
        # Real cells may contain no loci in a channel. Keep them in the plan;
        # no padding locus may become a genotype or a scored target.
        if source.cm.shape[1] == 0 or query.cm.shape[1] == 0:
            shape = (*query.cm.shape, values.shape[-1])
            return (values.new_zeros(shape) + values.sum() * 0,
                    torch.zeros_like(query.cm, dtype=torch.int64))
        return local_mean(values, query, source, radius, execution=self.config.execution)

    def _predict_channel(self, target: MaskedTensorChannel, common_values: Tensor,
                         common_sites: Sites, rare_values: Tensor, rare_sites: Sites,
                         rare: MaskedTensorChannel, condition: str) -> Tensor:
        query_metadata = self._metadata(target)
        query = self._sites(target, query_metadata, visible_only=False)
        query_values = self.locus_descriptor(target)
        common_summary, common_count = self._local_mean(
            common_values, query, common_sites, self.config.common_radius_cm)
        if condition == "count_only":
            rare_visible = rare_sites.valid
            raw = torch.stack(((rare.dosage > 0) & rare_visible,
                               rare.dosage.to(torch.int64) * rare_visible,
                               rare_visible), -1).to(common_summary.dtype)
            mean, rare_count = self._local_mean(raw, query, rare_sites,
                                               self.config.rare_radius_cm)
            counts = mean * rare_count[..., None].to(mean.dtype)
            rare_summary = self.rare_count_encoder(torch.log1p(counts))
        else:
            rare_summary, rare_count = self._local_mean(
                rare_values, query, rare_sites, self.config.rare_radius_cm)
        fused = torch.cat((query_values, common_summary, rare_summary,
                           torch.log1p(common_count.to(common_summary.dtype))[..., None],
                           torch.log1p(rare_count.to(common_summary.dtype))[..., None]), -1)
        logits = self.head(fused)
        return torch.where(query.valid[..., None], logits, torch.zeros_like(logits))

    def forward(self, common: MaskedTensorChannel, rare: MaskedTensorChannel,
                *, information_condition: str) -> MaskedModelOutput:
        _require(information_condition in INFORMATION_CONDITIONS,
                 "unknown rare-information condition")
        common.check(self.config.max_loci_per_channel)
        rare.check(self.config.max_loci_per_channel)
        _require(common.channel == "common" and rare.channel == "rare",
                 "common and rare channels are not interchangeable")
        _require(common.person_ids == rare.person_ids
                 and common.plan_identity == rare.plan_identity,
                 "channels must share people/order and exact person-by-cell mask")
        values = self._context_values(common, rare, information_condition)
        return MaskedModelOutput(
            common_logits=self._predict_channel(common, *values, rare, information_condition),
            rare_logits=self._predict_channel(rare, *values, rare, information_condition))


def build_masked_adam(model: MaskedGenotypeModel, *, learning_rate: float,
                      adam_state_precision: str):
    """Build Adam only from explicit numerical choices; no learning-rate default."""
    _require(type(learning_rate) in (int, float) and math.isfinite(learning_rate)
             and learning_rate > 0, "positive finite learning_rate required")
    _require(adam_state_precision in {"fp32", "fp64"},
             "adam_state_precision must be fp32 or fp64")
    if adam_state_precision == "fp32":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    from r02_adam_fp64_state import AdamFP64State
    return AdamFP64State(model.parameters(), lr=learning_rate, foreach=False)
