#!/usr/bin/env python3
"""Bounded tensor runtime for development-only contextual genomic models.

Consumes resolved v2 search parameters; never reads VCFs, trains a cohort or
launches jobs. Historical M36/M39 implementations are unchanged. Callers must
authenticate tensor alleles, masks, roles and feature definitions separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from r02_local_attention_blocks import (LOCAL_MEAN_PREFIX_RANGE_V1, ContextExecution,
                                         blocked_attention, blocked_local_mean,
                                         prefix_range_mean)
from r02_context_autosomal_pool import (RARE_POOLING_POLICIES, RARE_POOLING_SUM,
                                         finalize_rare_sum)
from r02_lai_cnn_config import validate_cnn_config, query_halo, chromosome_step, maximum_step
from r02_lai_common_context import (EVENT_QUERY_COMMON_KV_V1, LAI_COMMON_CONTEXT_MODES,
                                    LOCAL_MEAN_V1, EventQueryCommonAttention)
from r02_lai_query_event_attention import (LAI_QUERY_EVENT_MODES, QueryEventAttention)


FAMILIES = {"structure_deep_sets", "structure_local_attention", "lai_cnn", "lai_local_attention"}
FIELDS = ("family", "common_radius_cm", "rare_radius_cm", "width", "depth", "dropout", "heads", "learning_rate")
LOCUS_DESCRIPTOR_LEGACY = "ALLELE_POSITION_12_V0"
LOCUS_DESCRIPTOR_CHROM22 = "ALLELE_POSITION_CHROM22_V1"
LOCUS_DESCRIPTORS = frozenset((LOCUS_DESCRIPTOR_LEGACY, LOCUS_DESCRIPTOR_CHROM22))
AUTOSOME_COUNT = 22


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ContextConfig:
    family: str
    common_radius_cm: float
    rare_radius_cm: float | None
    width: int
    depth: int
    dropout: float
    heads: int | None
    learning_rate: float
    rare_pooling: str = RARE_POOLING_SUM
    locus_descriptor: str = LOCUS_DESCRIPTOR_LEGACY
    lai_common_context_mode: str = LOCAL_MEAN_V1
    lai_query_event_mode: str = LOCAL_MEAN_V1

    def __post_init__(self):
        require(self.family in FAMILIES, "unknown contextual family")
        for key in ("common_radius_cm", "learning_rate"):
            x = getattr(self, key)
            require(type(x) in (float, int) and math.isfinite(x) and x > 0, f"invalid {key}")
        for key in ("width", "depth"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"invalid {key}")
        require(type(self.dropout) in (float, int) and math.isfinite(self.dropout)
                and 0 <= self.dropout < 1, "invalid dropout")
        require(self.rare_pooling in RARE_POOLING_POLICIES, "unknown rare pooling policy")
        require(self.family.startswith("structure_") or self.rare_pooling == RARE_POOLING_SUM,
                "rare pooling applies only to structure families")
        require(self.locus_descriptor in LOCUS_DESCRIPTORS, "unknown locus descriptor")
        require(self.family.startswith("structure_") or self.locus_descriptor == LOCUS_DESCRIPTOR_LEGACY,
                "chromosome-aware locus descriptor applies only to structure families")
        require(self.lai_common_context_mode in LAI_COMMON_CONTEXT_MODES,
                "unknown LAI COMMON-context mode")
        require(self.lai_common_context_mode == LOCAL_MEAN_V1 or self.family.startswith("lai_"),
                "EVENT-query/COMMON-key-value attention applies only to LAI")
        require(self.lai_query_event_mode in LAI_QUERY_EVENT_MODES,
                "unknown LAI query-EVENT readout mode")
        require(self.lai_query_event_mode == LOCAL_MEAN_V1 or self.family.startswith("lai_"),
                "query-EVENT attention applies only to LAI")
        if self.family == "structure_deep_sets":
            require(self.rare_radius_cm is None, "Deep Sets has no rare-neighbor radius")
        else:
            require(type(self.rare_radius_cm) in (int, float) and math.isfinite(self.rare_radius_cm)
                    and self.rare_radius_cm > 0, "positive rare radius required")
        if self.family.endswith("local_attention"):
            require(type(self.heads) is int and self.heads > 0 and self.width % self.heads == 0,
                    "attention heads must divide width")
        else:
            require(self.heads is None, "heads apply only to attention")

    @classmethod
    def from_recipe(cls, resolved: dict):
        """Read explicit resolved fields, not legacy radius_cm defaults."""
        require(all(k in resolved for k in FIELDS), "incomplete v2 resolved configuration")
        require("radius_cm" not in resolved, "legacy radius has no implicit v2 mapping")
        values = {key: resolved[key] for key in FIELDS}
        values["rare_pooling"] = resolved.get("rare_pooling", RARE_POOLING_SUM)
        values["locus_descriptor"] = resolved.get("locus_descriptor", LOCUS_DESCRIPTOR_LEGACY)
        values["lai_common_context_mode"] = resolved.get("lai_common_context_mode", LOCAL_MEAN_V1)
        values["lai_query_event_mode"] = resolved.get("lai_query_event_mode", LOCAL_MEAN_V1)
        return cls(**values)


@dataclass
class Sites:
    """Features [B,N,F], genetic coordinates/chromosomes/mask [B,N].

    valid is a boolean measurement/event mask, not a carrier flag. Zero-dose
    observations remain valid. Nonfinite padding is discarded before arithmetic.
    Chromosomes are autosomal integers 1..22; positions are map cM, not bp.
    """
    features: Tensor
    cm: Tensor
    chrom: Tensor
    valid: Tensor

    def check(self, features: int, max_sites: int):
        require(self.features.ndim == 3 and self.features.shape[-1] == features, "feature shape")
        shape = self.features.shape[:2]
        require(shape[0] > 0 and 0 < shape[1] <= max_sites, "site limit exceeded; never truncate silently")
        require(self.cm.shape == shape and self.chrom.shape == shape and self.valid.shape == shape,
                "coordinate/mask shape")
        require(self.valid.dtype == torch.bool, "valid must be boolean")
        require(self.features.is_floating_point() and self.cm.is_floating_point(), "float features/cM required")
        require(self.chrom.dtype in (torch.int32, torch.int64), "integer chromosome IDs required")
        require(all(x.device == self.features.device for x in (self.cm, self.chrom, self.valid)), "device mismatch")
        require(bool(torch.isfinite(self.features[self.valid]).all()), "nonfinite measured feature")
        require(bool(torch.isfinite(self.cm[self.valid]).all()), "nonfinite measured cM")
        require(bool(((self.chrom[self.valid] >= 1) & (self.chrom[self.valid] <= 22)).all()), "autosomal IDs required")

    def cleaned(self):
        return torch.where(self.valid[..., None], self.features, 0.)


def genetic_neighbors(query: Sites, source: Sites, radius_cm: float) -> Tensor:
    """Explicit same-chromosome genetic-distance mask [B,Q,N]."""
    return (query.valid[:, :, None] & source.valid[:, None, :]
            & (query.chrom[:, :, None] == source.chrom[:, None, :])
            & ((query.cm[:, :, None] - source.cm[:, None, :]).abs() <= radius_cm))


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    clean = torch.where(mask[..., None], values, 0.)
    return clean.sum(1) / mask.sum(1).clamp_min(1)[:, None]


def local_mean(values: Tensor, query: Sites, source: Sites, radius_cm: float, chunk_size: int = 32,
               execution: ContextExecution | None = None):
    """Mean and count over jointly eligible neighbors; bounded query chunks."""
    if execution is not None:
        if execution.local_mean_mode == LOCAL_MEAN_PREFIX_RANGE_V1:
            return prefix_range_mean(values, query, source, radius_cm, execution)
        return blocked_local_mean(values, query, source, radius_cm, execution)
    pooled, counts = [], []
    values = torch.where(source.valid[..., None], values, 0.)
    for start in range(0, query.cm.shape[1], chunk_size):
        end = start + chunk_size
        q = Sites(query.features[:, start:end], query.cm[:, start:end],
                  query.chrom[:, start:end], query.valid[:, start:end])
        allowed = genetic_neighbors(q, source, radius_cm)
        count = allowed.sum(-1)
        pooled.append(torch.bmm(allowed.to(values.dtype), values) / count.clamp_min(1)[..., None])
        counts.append(count)
    return torch.cat(pooled, 1), torch.cat(counts, 1)


class ChromosomeAwareMLP(nn.Sequential):
    """First affine map with implicit chromosome one-hot columns.

    One complete ``Linear(input_dim + 22, width)`` owns initialization and
    checkpoint state. Forward selects its chromosome column instead of
    allocating a dense ``[batch, loci, 22]`` tensor. Invalid padding receives a
    safe index and is erased by the caller's validity mask.
    """
    def __init__(self, base_input_dim, *layers):
        super().__init__(*layers)
        self.base_input_dim = base_input_dim

    def forward(self, values, chromosomes, valid):
        require(values.shape[:-1] == chromosomes.shape == valid.shape,
                "chromosome-aware MLP axes differ")
        require(chromosomes.dtype in (torch.int32, torch.int64) and valid.dtype == torch.bool,
                "chromosome-aware MLP requires integer chromosomes and boolean validity")
        require(bool(((chromosomes[valid] >= 1) & (chromosomes[valid] <= AUTOSOME_COUNT)).all()),
                "valid chromosome-aware loci must be autosomes 1..22")
        first = self[0]
        require(isinstance(first, nn.Linear) and first.in_features == self.base_input_dim + AUTOSOME_COUNT,
                "chromosome-aware first Linear differs")
        safe_index = torch.where(valid, chromosomes - 1, torch.zeros_like(chromosomes))
        output = F.linear(values, first.weight[:, :self.base_input_dim], first.bias)
        output = output + F.embedding(safe_index, first.weight[:, self.base_input_dim:].t())
        for layer in list(self)[1:]:
            output = layer(output)
        return output


def mlp(input_dim, width, depth, dropout, chromosome_aware=False):
    layers = []
    for level in range(depth):
        first_width = input_dim + (AUTOSOME_COUNT if chromosome_aware else 0)
        layers.extend((nn.Linear(first_width if level == 0 else width, width),
                       nn.GELU(), nn.Dropout(dropout)))
    return (ChromosomeAwareMLP(input_dim, *layers) if chromosome_aware
            else nn.Sequential(*layers))


class LocalAttention(nn.Module):
    """Masked same-chromosome attention, one radius per direct connection.

    Multiple blocks extend the effective reach through intermediate events.
    Dense key products are bounded by max_events; no whole-genome scalability
    claim is made by this small-block runtime.
    """
    def __init__(self, config, execution: ContextExecution | None = None):
        super().__init__()
        self.radius = config.rare_radius_cm
        self.heads = config.heads
        self.execution = execution
        require(execution is None or config.dropout == 0,
                "exact block execution requires the frozen dropout=0")
        self.attention = nn.MultiheadAttention(config.width, config.heads, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.width)
        self.ff = mlp(config.width, config.width, 1, config.dropout)

    def forward(self, values, sites):
        clean = torch.where(sites.valid[..., None], values, 0.)
        if self.execution is not None:
            attended = blocked_attention(self.attention, clean, sites.cm, sites.chrom,
                                         sites.valid, self.radius, self.execution)
        else:
            attended = self._dense_attention(clean, sites)
        out = self.norm(clean + attended)
        out = out + self.ff(out)
        return torch.where(sites.valid[..., None], out, 0.)

    def _dense_attention(self, clean, sites):
        allowed = genetic_neighbors(sites, sites, self.radius)
        # Padding queries need one numerical sentinel to avoid softmax(-inf).
        # Its output is erased; no measured query attends missing events.
        safe = allowed.clone()
        safe[:, :, 0] |= ~sites.valid
        blocked = ~safe.repeat_interleave(self.heads, dim=0)
        attended, _ = self.attention(clean, clean, clean, attn_mask=blocked, need_weights=False)
        return attended


class MultiscaleQueryConvolution(nn.Module):
    """Parallel dense-support depthwise filters, then learned channel mixing.

    Separable filtering limits parameters; it does not make capacity identical
    to the historical full kernel-3 convolution or certify LAI accuracy.
    """
    def __init__(self, width, kernels):
        super().__init__()
        self.filters = nn.ModuleList(
            nn.Conv1d(width, width, k, padding=k // 2, groups=width)
            for k in kernels
        )
        self.mix = nn.Conv1d(width * len(kernels), width, 1)

    def forward(self, values):
        return self.mix(torch.cat([layer(values) for layer in self.filters], dim=1))


class ContextualModel(nn.Module):
    """Real tensor computation for both search branches, without a data loader.

    Structure returns a person vector. LAI returns six unordered diploid-state
    probabilities at supplied queries, not at rare-event locations. Reference
    channels, if used, must be explicitly defined in event_features upstream;
    this module does not manufacture reference genotypes or phase.
    """
    def __init__(self, config: ContextConfig, common_features: int, event_features: int,
                 max_common_sites=100000, max_events=2048, max_queries=2048,
                 query_max_gap_cm: float | None = None,
                 context_execution: dict | None = None,
                 cnn_query_config: dict | None = None,
                 lai_query_burden_features: int = 0):
        super().__init__()
        require(common_features > 0 and event_features > 0, "positive feature dimensions required")
        if config.family == "lai_cnn":
            require(query_max_gap_cm is not None, "CNN requires an explicit fixed query_max_gap_cm")
        if query_max_gap_cm is not None:
            require(type(query_max_gap_cm) in (float, int) and math.isfinite(query_max_gap_cm)
                    and query_max_gap_cm > 0, "invalid query_max_gap_cm")
        self.config = config
        self.common_features = common_features
        self.event_features = event_features
        self.limits = (max_common_sites, max_events, max_queries)
        self.query_max_gap_cm = query_max_gap_cm
        self.cnn_query_config = validate_cnn_config(cnn_query_config)
        require(self.cnn_query_config is None or config.family == "lai_cnn",
                "multiscale query convolution applies only to lai_cnn")
        if self.cnn_query_config is not None:
            require(query_max_gap_cm >= maximum_step(self.cnn_query_config),
                    "multiscale query block gap must cover declared query steps")
        self.query_halo_each_side = query_halo(config.depth, self.cnn_query_config)
        require(lai_query_burden_features in (0, 3), "LAI query burden width differs")
        require(lai_query_burden_features == 0 or config.family.startswith("lai_"),
                "query burden applies only to LAI")
        self.lai_query_burden_features = lai_query_burden_features
        execution = None if context_execution is None else ContextExecution(**context_execution)
        self.context_execution = execution
        self.chromosome_aware = config.locus_descriptor == LOCUS_DESCRIPTOR_CHROM22
        require(config.lai_common_context_mode == LOCAL_MEAN_V1 or execution is not None,
                "EVENT-query/COMMON-key-value attention requires bounded context execution")
        self.common_encoder = mlp(common_features, config.width, 1, config.dropout,
                                  self.chromosome_aware)
        common_context_heads = config.heads if config.heads is not None else 1
        self.event_common_attention = (EventQueryCommonAttention(
            event_features, config.width, common_context_heads, config.dropout, execution)
            if config.lai_common_context_mode == EVENT_QUERY_COMMON_KV_V1 else None)
        # Event metadata must retain actual allele identity/position definitions.
        # log1p(number of common neighbors) distinguishes no context from mean zero.
        self.event_encoder = mlp(event_features + config.width + 1, config.width,
                                 config.depth if config.family == "structure_deep_sets" else 1,
                                 config.dropout, self.chromosome_aware)
        self.event_attention = nn.ModuleList(
            LocalAttention(config, execution) for _ in range(config.depth)
        ) if config.family.endswith("local_attention") else nn.ModuleList()
        is_structure = config.family.startswith("structure_")
        self.person_rho = mlp(config.width, config.width, 1, config.dropout) if is_structure else None
        self.query_fusion = mlp(2 * config.width + 8 + lai_query_burden_features,
                                config.width, 1, config.dropout) if not is_structure else None
        self.query_convs = nn.ModuleList(
            (nn.Conv1d(config.width, config.width, 3, padding=1)
             if self.cnn_query_config is None else MultiscaleQueryConvolution(
                 config.width, self.cnn_query_config["kernel_sizes"]))
            for _ in range(config.depth)
        ) if config.family == "lai_cnn" else nn.ModuleList()
        self.query_dropout = nn.Dropout(config.dropout)
        self.state_head = nn.Linear(config.width, 6) if not is_structure else None
        self.gate_head = nn.Linear(config.width, 1) if not is_structure else None
        # Added after historical modules so default initialization and old
        # checkpoints retain their parameter names and random-number sequence.
        self.query_event_attention = (None if config.lai_query_event_mode == LOCAL_MEAN_V1
            else QueryEventAttention(config.width + 7 + lai_query_burden_features,
                config.width, common_context_heads, execution, config.lai_query_event_mode))

    def common_values(self, common):
        common.check(self.common_features, self.limits[0])
        values = (self.common_encoder(common.cleaned(), common.chrom, common.valid)
                  if self.chromosome_aware else self.common_encoder(common.cleaned()))
        return torch.where(common.valid[..., None], values, 0.)

    def event_values(self, common, encoded_common, events):
        events.check(self.event_features, self.limits[1])
        require(common.features.shape[0] == events.features.shape[0], "batch mismatch")
        if self.event_common_attention is None:
            context, count = local_mean(encoded_common, events, common, self.config.common_radius_cm,
                                        execution=self.context_execution)
        else:
            context, count = self.event_common_attention(
                events.features, events.cm, events.chrom, events.valid,
                encoded_common, common.cm, common.chrom, common.valid,
                self.config.common_radius_cm)
        inputs = torch.cat((events.cleaned(), context,
                            torch.log1p(count.to(context.dtype))[..., None]), -1)
        values = (self.event_encoder(inputs, events.chrom, events.valid)
                  if self.chromosome_aware else self.event_encoder(inputs))
        values = torch.where(events.valid[..., None], values, 0.)
        for block in self.event_attention:
            values = block(values, events)
        return values

    def encode_person(self, common: Sites, events: Sites | None = None, include_rare=True):
        require(self.config.family.startswith("structure_"), "structure family required")
        encoded = self.common_values(common)
        baseline = masked_mean(encoded, common.valid)
        if not include_rare:
            return baseline
        require(events is not None, "events required for rare-enabled arm")
        values = self.event_values(common, encoded, events)
        # Shared pooling convention; this alone does not match model capacity.
        rare = finalize_rare_sum(values.sum(1), events.valid.sum(1), self.person_rho,
                                 self.config.rare_pooling)
        return baseline + rare

    @staticmethod
    def _check_unique_queries(queries):
        """One usable prediction per person/chromosome/coordinate, in either family."""
        for batch in range(queries.cm.shape[0]):
            for chrom in torch.unique(queries.chrom[batch][queries.valid[batch]]):
                coords = queries.cm[batch][queries.valid[batch] & (queries.chrom[batch] == chrom)]
                require(torch.unique(coords).numel() == coords.numel(),
                        "duplicate query coordinates are not allowed")

    def _convolve_queries(self, values, queries):
        """Sort within chromosome, convolve, restore caller query order.

        Queries must describe a prespecified dense genomic grid. Historical
        runs use kernel=3; an explicit multiscale configuration changes the
        query halo and requires its authenticated, regular genetic grid.
        A fixed, explicit query_max_gap_cm splits genomic gaps. Invalid queries
        with known coordinates also split blocks, even when the surviving gap
        is smaller than that maximum. NaN padding has no genomic location;
        missing coverage then has to be represented by the surviving gaps or
        by invalid queries with known coordinates. No blocks are joined across
        chromosomes. This does not certify density or truth of the input grid.
        """
        require(self.query_max_gap_cm is not None, "CNN requires an explicit fixed query_max_gap_cm")
        self._check_unique_queries(queries)
        rows = []
        for batch in range(values.shape[0]):
            row = torch.zeros_like(values[batch])
            for chrom in torch.unique(queries.chrom[batch][queries.valid[batch]]):
                located = (queries.chrom[batch] == chrom) & torch.isfinite(queries.cm[batch])
                ordered = torch.where(located)[0]
                ordered = ordered[torch.argsort(queries.cm[batch, ordered], stable=True)]
                kept_positions = torch.where(queries.valid[batch, ordered])[0]
                ids = ordered[kept_positions]
                left, right = queries.cm[batch, ids[:-1]], queries.cm[batch, ids[1:]]
                distances = right - left
                limit = torch.full_like(distances, self.query_max_gap_cm)
                # Bound coordinate-rounding error, not only the small distance's
                # error (e.g. .01 cM steps around 100 cM in float32).
                scale = torch.maximum(torch.maximum(left.abs(), right.abs()), limit)
                roundoff = 2 * torch.finfo(queries.cm.dtype).eps * scale
                too_far = distances > limit + roundoff
                masked_between = kept_positions[1:] > kept_positions[:-1] + 1
                cuts = (torch.where(too_far | masked_between)[0] + 1).tolist()
                for block_ids in torch.tensor_split(ids, cuts):
                    step = (None if self.cnn_query_config is None else
                            chromosome_step(self.cnn_query_config, queries.chrom[batch, block_ids[0]].item()))
                    if self.cnn_query_config is not None and len(block_ids) > 1:
                        coordinates = queries.cm[batch, block_ids]
                        scale = torch.maximum(coordinates[:-1].abs(), coordinates[1:].abs())
                        tolerance = 2 * torch.finfo(coordinates.dtype).eps * torch.clamp(scale, min=step)
                        require(bool(((coordinates[1:] - coordinates[:-1] - step).abs()
                                      <= tolerance).all()),
                                "multiscale CNN requires the frozen regular query grid")
                    block = values[batch, block_ids].T[None]
                    for conv in self.query_convs:
                        block = block + self.query_dropout(torch.nn.functional.gelu(conv(block)))
                    row = row.index_copy(0, block_ids, block[0].T)
            rows.append(row)
        return torch.stack(rows)

    def _lai_query_hidden(self, common: Sites, queries: Sites, baseline_probabilities: Tensor,
                          events: Sites | None = None, include_rare=True,
                          query_burden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return the shared LAI query representation and cleaned FLARE input."""
        require(self.config.family.startswith("lai_"), "LAI family required")
        encoded = self.common_values(common)
        queries.check(queries.features.shape[-1], self.limits[2])
        self._check_unique_queries(queries)
        require(common.features.shape[0] == queries.features.shape[0], "batch mismatch")
        require(baseline_probabilities.shape == (*queries.cm.shape, 6), "six diploid states required")
        p = baseline_probabilities
        require(bool(torch.isfinite(p[queries.valid]).all()) and bool((p[queries.valid] >= 0).all()),
                "invalid measured probabilities")
        require(torch.allclose(p.sum(-1)[queries.valid], torch.ones_like(p.sum(-1)[queries.valid]),
                               atol=1e-5, rtol=1e-5), "probabilities must sum to one")
        p = torch.where(queries.valid[..., None], p, 1 / 6)
        c, nc = local_mean(encoded, queries, common, self.config.common_radius_cm,
                           execution=self.context_execution)
        rare = torch.zeros_like(c)
        nr = torch.zeros_like(nc)
        if self.lai_query_burden_features:
            require(query_burden is not None and query_burden.shape == (*queries.cm.shape, 3)
                    and bool(torch.isfinite(query_burden).all()) and bool((query_burden >= 0).all()),
                    "LAI query burden shape/value differs")
        else:
            require(query_burden is None, "legacy LAI model cannot consume query burden")
        if include_rare:
            require(events is not None, "events required for rare-enabled arm")
            values = self.event_values(common, encoded, events)
            if self.query_event_attention is None:
                rare, nr = local_mean(values, queries, events, self.config.rare_radius_cm,
                                     execution=self.context_execution)
            else:
                query_inputs = [c, p, torch.log1p(nc.to(c.dtype))[..., None]]
                if self.lai_query_burden_features:
                    query_inputs.append(torch.log1p(query_burden.to(c.dtype)))
                rare, nr = self.query_event_attention(
                    torch.cat(query_inputs, -1), queries.cm, queries.chrom, queries.valid,
                    values, events.cm, events.chrom, events.valid, self.config.rare_radius_cm)
        inputs = [c, rare, p, torch.log1p(nc.to(c.dtype))[..., None],
                  torch.log1p(nr.to(c.dtype))[..., None]]
        if self.lai_query_burden_features:
            inputs.append(torch.log1p(query_burden.to(c.dtype)))
        x = self.query_fusion(torch.cat(inputs, -1))
        if self.query_convs:
            x = self._convolve_queries(x, queries)
        return x, p

    def predict_lai_logits(self, common: Sites, queries: Sites, baseline_probabilities: Tensor,
                           events: Sites | None = None, include_rare=True,
                           query_burden: Tensor | None = None) -> Tensor:
        """Return raw six-state neural potentials before softmax or legacy gating.

        FLARE probabilities remain an input to ``query_fusion``.  They must not
        be multiplied into a downstream CRF again as independent evidence.
        ``gate_head`` belongs only to :meth:`predict_lai`'s historical mixture.
        """
        x, _ = self._lai_query_hidden(
            common, queries, baseline_probabilities, events, include_rare, query_burden)
        logits = self.state_head(x)
        return torch.where(queries.valid[..., None], logits,
                           torch.full_like(logits, float("nan")))

    def predict_lai(self, common: Sites, queries: Sites, baseline_probabilities: Tensor,
                    events: Sites | None = None, include_rare=True,
                    query_burden: Tensor | None = None):
        x, p = self._lai_query_hidden(
            common, queries, baseline_probabilities, events, include_rare, query_burden)
        candidate = self.state_head(x).softmax(-1)
        gate = self.gate_head(x).sigmoid()
        result = (1 - gate) * p + gate * candidate
        # Invalid queries carry no biological prediction.
        return torch.where(queries.valid[..., None], result, torch.full_like(result, float("nan")))


def build_model(resolved: dict, common_features: int, event_features: int, **limits):
    # The query-grid gap is a frozen adapter option, never inferred from r_C/r_R
    # or chosen by HPO. Bind it through fixed_config in a resolved recipe.
    if "query_max_gap_cm" in resolved:
        if "query_max_gap_cm" in limits:
            require(limits["query_max_gap_cm"] == resolved["query_max_gap_cm"],
                    "conflicting query_max_gap_cm configuration")
        limits["query_max_gap_cm"] = resolved["query_max_gap_cm"]
    if resolved["family"] == "lai_cnn" and "cnn_query_config" in resolved:
        if "cnn_query_config" in limits:
            require(limits["cnn_query_config"] == resolved["cnn_query_config"],
                    "conflicting multiscale query configuration")
        limits["cnn_query_config"] = resolved["cnn_query_config"]
    return ContextualModel(ContextConfig.from_recipe(resolved), common_features, event_features, **limits)


def build_optimizer(model: ContextualModel):
    """Adam with explicit trial learning rate; no step/run is performed here."""
    return torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
