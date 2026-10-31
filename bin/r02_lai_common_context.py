"""Local EVENT-query/COMMON-key-value attention for LAI.

The module operates on diploid locus features and genetic-map coordinates.  It
does not infer haplotypes or phase.  Query blocks reduce temporary attention
storage; every eligible COMMON locus remains in the softmax, and a block
that exceeds its configured mask budget fails instead of truncating neighbors.
Dense and blocked calculations are equivalent up to floating-point rounding.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from r02_local_attention_blocks import ContextExecution, candidate_keys


LOCAL_MEAN_V1 = "LOCAL_MEAN_V1"
EVENT_QUERY_COMMON_KV_V1 = "EVENT_QUERY_COMMON_KV_V1"
LAI_COMMON_CONTEXT_MODES = frozenset((LOCAL_MEAN_V1, EVENT_QUERY_COMMON_KV_V1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def event_common_attention_parameter_count(event_features: int, width: int, heads: int) -> int:
    """Exact trainable parameter count for :class:`EventQueryCommonAttention`."""
    _require(all(type(value) is int and value > 0 for value in (event_features, width, heads)),
             "positive attention dimensions required")
    _require(width % heads == 0, "attention heads must divide width")
    # EVENT query, COMMON key/value and output projections, plus two relative
    # genetic-position inputs and one bias for each head.
    return ((event_features * width + width) + 3 * (width * width + width)
            + 2 * heads + heads)


def _candidate_columns(event_cm, event_chrom, event_valid,
                       common_cm, common_chrom, common_valid, radius_cm):
    """Reuse the exact neighbor predicate and finite sentinel shared by LAI."""
    return candidate_keys(event_cm, event_chrom, event_valid,
                          common_cm, common_chrom, common_valid, radius_cm)


class EventQueryCommonAttention(nn.Module):
    """Attend from each observed EVENT locus to observed nearby COMMON loci.

    EVENT and COMMON remain diploid representations.  Signed and absolute
    relative genetic distance, divided by the already configured COMMON radius,
    add a learned per-head position term to the scaled query-key product.
    """

    def __init__(self, event_features: int, width: int, heads: int, dropout: float = 0.,
                 execution=None):
        super().__init__()
        _require(all(type(value) is int and value > 0
                     for value in (event_features, width, heads)),
                 "positive attention dimensions required")
        _require(width % heads == 0, "attention heads must divide width")
        _require(type(dropout) in (int, float) and math.isfinite(dropout)
                 and 0 <= dropout < 1, "invalid attention dropout")
        _require(execution is None or isinstance(execution, ContextExecution),
                 "execution must be a validated ContextExecution")
        _require(execution is None or dropout == 0,
                 "exact checkpointed/block comparison requires dropout=0")
        self.event_features = event_features
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.dropout = float(dropout)
        self.execution = execution
        self.event_query = nn.Linear(event_features, width)
        self.common_key = nn.Linear(width, width)
        self.common_value = nn.Linear(width, width)
        self.relative_position = nn.Linear(2, heads)
        self.output = nn.Linear(width, width)

    def _validate(self, event_features, event_cm, event_chrom, event_valid,
                  common_values, common_cm, common_chrom, common_valid, radius_cm):
        _require(event_features.ndim == common_values.ndim == 3,
                 "EVENT/COMMON values must have batch, locus and feature axes")
        _require(event_features.shape[-1] == self.event_features
                 and common_values.shape[-1] == self.width,
                 "EVENT/COMMON feature width differs")
        _require(event_features.shape[0] == common_values.shape[0]
                 and event_features.shape[:2] == event_cm.shape == event_chrom.shape == event_valid.shape
                 and common_values.shape[:2] == common_cm.shape == common_chrom.shape == common_valid.shape,
                 "EVENT/COMMON axes differ")
        _require(event_features.shape[1] > 0 and common_values.shape[1] > 0,
                 "nonempty EVENT and COMMON axes required")
        _require(event_valid.dtype == common_valid.dtype == torch.bool,
                 "EVENT/COMMON masks must be boolean")
        _require(event_chrom.dtype in (torch.int32, torch.int64)
                 and common_chrom.dtype in (torch.int32, torch.int64),
                 "EVENT/COMMON chromosomes must be integer")
        tensors = (event_cm, event_chrom, event_valid, common_values,
                   common_cm, common_chrom, common_valid)
        _require(all(value.device == event_features.device for value in tensors),
                 "EVENT/COMMON tensors must share a device")
        _require(type(radius_cm) in (int, float) and math.isfinite(radius_cm) and radius_cm > 0,
                 "positive finite COMMON radius required")
        _require(bool(torch.isfinite(event_features[event_valid]).all())
                 and bool(torch.isfinite(common_values[common_valid]).all())
                 and bool(torch.isfinite(event_cm[event_valid]).all())
                 and bool(torch.isfinite(common_cm[common_valid]).all()),
                 "nonfinite observed EVENT/COMMON value or coordinate")

    def _attend(self, query, key, value, event_cm, event_chrom, event_valid,
                common_cm, common_chrom, common_valid, radius_cm):
        allowed = (event_valid[:, :, None] & common_valid[:, None, :]
                   & (event_chrom[:, :, None] == common_chrom[:, None, :])
                   & ((event_cm[:, :, None] - common_cm[:, None, :]).abs() <= radius_cm))
        count = allowed.sum(-1)
        has_neighbors = count > 0

        # Invalid coordinates may be NaN padding.  Mask before the position
        # projection so those NaNs cannot contaminate finite queries or gradients.
        raw_delta = common_cm[:, None, :] - event_cm[:, :, None]
        scaled_delta = torch.where(allowed, raw_delta / radius_cm,
                                   torch.zeros_like(raw_delta))
        # Genetic-map coordinates are commonly float64 while learned LAI
        # projections are float32.  Position is an input to the learned layer,
        # so use the query projection dtype without changing masks/coordinates.
        position = torch.stack((scaled_delta, scaled_delta.abs()), -1).to(query.dtype)
        position_bias = self.relative_position(position).permute(0, 3, 1, 2)

        scores = torch.einsum("bhed,bhcd->bhec", query, key) / math.sqrt(self.head_width)
        scores = scores + position_bias
        safe = allowed.clone()
        safe[:, :, 0] |= ~has_neighbors
        weights = torch.softmax(scores.masked_fill(~safe[:, None], float("-inf")), -1)
        weights = torch.where(allowed[:, None], weights, torch.zeros_like(weights))
        weights = F.dropout(weights, self.dropout, self.training)
        attended = torch.einsum("bhec,bhcd->bhed", weights, value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            event_cm.shape[0], event_cm.shape[1], self.width)
        result = self.output(attended)
        result = torch.where((event_valid & has_neighbors)[..., None], result,
                             torch.zeros_like(result))
        return result, count

    def forward(self, event_features, event_cm, event_chrom, event_valid,
                common_values, common_cm, common_chrom, common_valid, radius_cm):
        self._validate(event_features, event_cm, event_chrom, event_valid,
                       common_values, common_cm, common_chrom, common_valid, radius_cm)
        clean_events = torch.where(event_valid[..., None], event_features,
                                   torch.zeros_like(event_features))
        clean_common = torch.where(common_valid[..., None], common_values,
                                   torch.zeros_like(common_values))
        query = self.event_query(clean_events).reshape(
            *clean_events.shape[:2], self.heads, self.head_width).permute(0, 2, 1, 3)
        key = self.common_key(clean_common).reshape(
            *clean_common.shape[:2], self.heads, self.head_width).permute(0, 2, 1, 3)
        value = self.common_value(clean_common).reshape(
            *clean_common.shape[:2], self.heads, self.head_width).permute(0, 2, 1, 3)

        if self.execution is None:
            return self._attend(query, key, value, event_cm, event_chrom, event_valid,
                                common_cm, common_chrom, common_valid, radius_cm)

        outputs, counts = [], []
        for start in range(0, event_cm.shape[1], self.execution.query_block_size):
            end = min(start + self.execution.query_block_size, event_cm.shape[1])
            columns = _candidate_columns(event_cm[:, start:end], event_chrom[:, start:end],
                                         event_valid[:, start:end], common_cm, common_chrom,
                                         common_valid, radius_cm)
            elements = (event_cm.shape[0] * self.heads * (end - start) * len(columns))
            if elements > self.execution.max_mask_elements:
                raise ValueError("EVENT-COMMON attention mask budget exceeded; no neighbors were truncated")
            args = (query[:, :, start:end], key[:, :, columns], value[:, :, columns],
                    event_cm[:, start:end], event_chrom[:, start:end], event_valid[:, start:end],
                    common_cm[:, columns], common_chrom[:, columns], common_valid[:, columns],
                    radius_cm)
            if self.execution.checkpoint_blocks and torch.is_grad_enabled():
                result = checkpoint(self._attend, *args, use_reentrant=False)
            else:
                result = self._attend(*args)
            outputs.append(result[0]); counts.append(result[1])
        return torch.cat(outputs, 1), torch.cat(counts, 1)

    @property
    def trainable_parameter_count(self):
        expected = event_common_attention_parameter_count(
            self.event_features, self.width, self.heads)
        actual = sum(parameter.numel() for parameter in self.parameters())
        _require(actual == expected, "EVENT-COMMON attention parameter inventory differs")
        return actual
