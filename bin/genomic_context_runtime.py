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


FAMILIES = {"structure_deep_sets", "structure_local_attention", "lai_cnn", "lai_local_attention"}
FIELDS = ("family", "common_radius_cm", "rare_radius_cm", "width", "depth", "dropout", "heads", "learning_rate")


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

    def __post_init__(self):
        require(self.family in FAMILIES, "unknown contextual family")
        for key in ("common_radius_cm", "learning_rate"):
            x = getattr(self, key)
            require(type(x) in (float, int) and math.isfinite(x) and x > 0, f"invalid {key}")
        for key in ("width", "depth"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"invalid {key}")
        require(type(self.dropout) in (float, int) and math.isfinite(self.dropout)
                and 0 <= self.dropout < 1, "invalid dropout")
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
        return cls(**{key: resolved[key] for key in FIELDS})


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


def local_mean(values: Tensor, query: Sites, source: Sites, radius_cm: float, chunk_size: int = 32):
    """Mean and count over jointly eligible neighbors; bounded query chunks."""
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


def mlp(input_dim, width, depth, dropout):
    layers = []
    for level in range(depth):
        layers.extend((nn.Linear(input_dim if level == 0 else width, width), nn.GELU(), nn.Dropout(dropout)))
    return nn.Sequential(*layers)


class LocalAttention(nn.Module):
    """Masked same-chromosome attention, one radius per direct connection.

    Multiple blocks extend the effective reach through intermediate events.
    Dense key products are bounded by max_events; no whole-genome scalability
    claim is made by this small-block runtime.
    """
    def __init__(self, config):
        super().__init__()
        self.radius = config.rare_radius_cm
        self.heads = config.heads
        self.attention = nn.MultiheadAttention(config.width, config.heads, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.width)
        self.ff = mlp(config.width, config.width, 1, config.dropout)

    def forward(self, values, sites):
        allowed = genetic_neighbors(sites, sites, self.radius)
        # Padding queries need one numerical sentinel to avoid softmax(-inf).
        # Its output is erased; no measured query attends missing events.
        safe = allowed.clone()
        safe[:, :, 0] |= ~sites.valid
        blocked = ~safe.repeat_interleave(self.heads, dim=0)
        clean = torch.where(sites.valid[..., None], values, 0.)
        attended, _ = self.attention(clean, clean, clean, attn_mask=blocked, need_weights=False)
        out = self.norm(clean + attended)
        out = out + self.ff(out)
        return torch.where(sites.valid[..., None], out, 0.)


class ContextualModel(nn.Module):
    """Real tensor computation for both search branches, without a data loader.

    Structure returns a person vector. LAI returns six unordered diploid-state
    probabilities at supplied queries, not at rare-event locations. Reference
    channels, if used, must be explicitly defined in event_features upstream;
    this module does not manufacture reference genotypes or phase.
    """
    def __init__(self, config: ContextConfig, common_features: int, event_features: int,
                 max_common_sites=100000, max_events=2048, max_queries=2048,
                 query_max_gap_cm: float | None = None):
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
        self.common_encoder = mlp(common_features, config.width, 1, config.dropout)
        # Event metadata must retain actual allele identity/position definitions.
        # log1p(number of common neighbors) distinguishes no context from mean zero.
        self.event_encoder = mlp(event_features + config.width + 1, config.width,
                                 config.depth if config.family == "structure_deep_sets" else 1, config.dropout)
        self.event_attention = nn.ModuleList(
            LocalAttention(config) for _ in range(config.depth)
        ) if config.family.endswith("local_attention") else nn.ModuleList()
        is_structure = config.family.startswith("structure_")
        self.person_rho = mlp(config.width, config.width, 1, config.dropout) if is_structure else None
        self.query_fusion = mlp(2 * config.width + 8, config.width, 1, config.dropout) if not is_structure else None
        self.query_convs = nn.ModuleList(
            nn.Conv1d(config.width, config.width, 3, padding=1) for _ in range(config.depth)
        ) if config.family == "lai_cnn" else nn.ModuleList()
        self.query_dropout = nn.Dropout(config.dropout)
        self.state_head = nn.Linear(config.width, 6) if not is_structure else None
        self.gate_head = nn.Linear(config.width, 1) if not is_structure else None

    def common_values(self, common):
        common.check(self.common_features, self.limits[0])
        return torch.where(common.valid[..., None], self.common_encoder(common.cleaned()), 0.)

    def event_values(self, common, encoded_common, events):
        events.check(self.event_features, self.limits[1])
        require(common.features.shape[0] == events.features.shape[0], "batch mismatch")
        context, count = local_mean(encoded_common, events, common, self.config.common_radius_cm)
        values = self.event_encoder(torch.cat((events.cleaned(), context,
                                               torch.log1p(count.to(context.dtype))[..., None]), -1))
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
        rare = self.person_rho(values.sum(1))
        rare = torch.where(events.valid.any(1)[:, None], rare, 0.)
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

        Queries must describe a prespecified dense genomic grid. Kernel=3 and
        unit dilation are fixed engineering settings, not searched parameters.
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
                    block = values[batch, block_ids].T[None]
                    for conv in self.query_convs:
                        block = block + self.query_dropout(torch.nn.functional.gelu(conv(block)))
                    row = row.index_copy(0, block_ids, block[0].T)
            rows.append(row)
        return torch.stack(rows)

    def predict_lai(self, common: Sites, queries: Sites, baseline_probabilities: Tensor,
                    events: Sites | None = None, include_rare=True):
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
        c, nc = local_mean(encoded, queries, common, self.config.common_radius_cm)
        rare = torch.zeros_like(c)
        nr = torch.zeros_like(nc)
        if include_rare:
            require(events is not None, "events required for rare-enabled arm")
            values = self.event_values(common, encoded, events)
            rare, nr = local_mean(values, queries, events, self.config.rare_radius_cm)
        x = self.query_fusion(torch.cat((c, rare, p, torch.log1p(nc.to(c.dtype))[..., None],
                                         torch.log1p(nr.to(c.dtype))[..., None]), -1))
        if self.query_convs:
            x = self._convolve_queries(x, queries)
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
    return ContextualModel(ContextConfig.from_recipe(resolved), common_features, event_features, **limits)


def build_optimizer(model: ContextualModel):
    """Adam with explicit trial learning rate; no step/run is performed here."""
    return torch.optim.Adam(model.parameters(), lr=model.config.learning_rate)
