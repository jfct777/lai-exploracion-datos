"""Exact query-block execution of existing zero-dropout local attention.

This changes temporary storage, not neighborhoods or learned parameters. The
caller must measure the real workload before admitting training. A block can
still be expensive in dense genetic intervals; no linear-time claim is made.
"""
from dataclasses import dataclass
import math

import torch
from torch.utils.checkpoint import checkpoint


LOCAL_MEAN_BLOCKED_MASK_V1 = "BLOCKED_MASK_V1"
LOCAL_MEAN_PREFIX_RANGE_V1 = "PREFIX_RANGE_MEAN_V1"
LOCAL_MEAN_MODES = frozenset((LOCAL_MEAN_BLOCKED_MASK_V1, LOCAL_MEAN_PREFIX_RANGE_V1))


@dataclass(frozen=True)
class ContextExecution:
    query_block_size: int
    max_mask_elements: int
    checkpoint_blocks: bool
    local_mean_mode: str = LOCAL_MEAN_BLOCKED_MASK_V1

    def __post_init__(self):
        for name in ("query_block_size", "max_mask_elements"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"positive integer {name} required")
        if type(self.checkpoint_blocks) is not bool:
            raise ValueError("checkpoint_blocks must be boolean")
        if self.local_mean_mode not in LOCAL_MEAN_MODES:
            raise ValueError("unknown local mean execution mode")


def candidate_keys(qcm, qchrom, qvalid, cm, chrom, valid, radius):
    """Union of exact neighbor columns, with at most a B*N temporary mask.

    Evaluate the original subtraction predicate directly: rounded q +/- radius
    bounds can omit accepted neighbors near cancellation. No sorted coordinates
    are assumed. Index zero is only a numerical sentinel for invalid queries.
    """
    keep = torch.zeros(cm.shape, dtype=torch.bool, device=cm.device)
    for index in range(qcm.shape[1]):
        keep |= (valid & qvalid[:, index, None]
                 & (chrom == qchrom[:, index, None])
                 & ((qcm[:, index, None] - cm).abs() <= radius))
    columns = keep.any(0)
    columns[0] = True
    return torch.where(columns)[0]


def blocked_attention(attention, clean, cm, chrom, valid, radius, execution):
    if attention.dropout != 0:
        raise ValueError("exact block execution requires the frozen dropout=0")

    def attend(query, source, qcm, qchrom, qvalid, scm, schrom, svalid):
        keys = candidate_keys(qcm, qchrom, qvalid, scm, schrom, svalid, radius)
        elements = query.shape[0] * attention.num_heads * query.shape[1] * len(keys)
        if elements > execution.max_mask_elements:
            raise ValueError("attention mask budget exceeded; no neighbors were truncated")
        allowed = (qvalid[:, :, None] & svalid[:, None, keys]
                   & (qchrom[:, :, None] == schrom[:, None, keys])
                   & ((qcm[:, :, None] - scm[:, None, keys]).abs() <= radius))
        allowed[:, :, 0] |= ~qvalid
        mask = ~allowed.repeat_interleave(attention.num_heads, dim=0)
        values = source[:, keys]
        return attention(query, values, values, attn_mask=mask, need_weights=False)[0]

    outputs = []
    for start in range(0, clean.shape[1], execution.query_block_size):
        end = start + execution.query_block_size
        args = (clean[:, start:end], clean, cm[:, start:end], chrom[:, start:end],
                valid[:, start:end], cm, chrom, valid)
        if execution.checkpoint_blocks and torch.is_grad_enabled():
            outputs.append(checkpoint(attend, *args, use_reentrant=False))
        else:
            outputs.append(attend(*args))
    return torch.cat(outputs, dim=1)


def blocked_local_mean(values, query, source, radius, execution):
    """Same masked sum/count, without retaining every rectangular mask."""
    def mean(v, qcm, qchrom, qvalid, scm, schrom, svalid):
        keys = candidate_keys(qcm, qchrom, qvalid, scm, schrom, svalid, radius)
        if qcm.shape[0] * qcm.shape[1] * len(keys) > execution.max_mask_elements:
            raise ValueError("context mask budget exceeded; no neighbors were truncated")
        allowed = (qvalid[:, :, None] & svalid[:, None, keys]
                   & (qchrom[:, :, None] == schrom[:, None, keys])
                   & ((qcm[:, :, None] - scm[:, None, keys]).abs() <= radius))
        count = allowed.sum(-1)
        selected = torch.where(svalid[:, keys, None], v[:, keys], 0.)
        return torch.bmm(allowed.to(v.dtype), selected) / count.clamp_min(1)[..., None], count

    pooled, counts = [], []
    for start in range(0, query.cm.shape[1], execution.query_block_size):
        end = start + execution.query_block_size
        args = (values, query.cm[:, start:end], query.chrom[:, start:end],
                query.valid[:, start:end], source.cm, source.chrom, source.valid)
        result = (checkpoint(mean, *args, use_reentrant=False)
                  if execution.checkpoint_blocks and torch.is_grad_enabled() else mean(*args))
        pooled.append(result[0]); counts.append(result[1])
    return torch.cat(pooled, 1), torch.cat(counts, 1)


def _literal_neighbor_bounds(sorted_cm, query_cm, radius):
    """Return exact [left, right) bounds for ``abs(q - k) <= radius``.

    Each binary-search comparison evaluates the same rounded subtraction as
    the dense predicate. Computing rounded ``q +/- radius`` first is not
    equivalent at floating-point boundaries.
    """
    size = sorted_cm.numel()

    left_lo = torch.zeros_like(query_cm, dtype=torch.int64)
    left_hi = torch.full_like(left_lo, size)
    while bool((left_lo < left_hi).any()):
        active = left_lo < left_hi
        middle = torch.div(left_lo + left_hi, 2, rounding_mode="floor")
        probe = sorted_cm[middle.clamp_max(size - 1)]
        accepted_or_right = (query_cm - probe) <= radius
        left_hi = torch.where(active & accepted_or_right, middle, left_hi)
        left_lo = torch.where(active & ~accepted_or_right, middle + 1, left_lo)

    right_lo = left_lo.clone()
    right_hi = torch.full_like(right_lo, size)
    while bool((right_lo < right_hi).any()):
        active = right_lo < right_hi
        middle = torch.div(right_lo + right_hi, 2, rounding_mode="floor")
        probe = sorted_cm[middle.clamp_max(size - 1)]
        strictly_right = (query_cm - probe) < -radius
        right_hi = torch.where(active & strictly_right, middle, right_hi)
        right_lo = torch.where(active & ~strictly_right, middle + 1, right_lo)
    return left_lo, right_lo


def prefix_range_mean(values, query, source, radius, execution):
    """Exact-neighbor local means without allocating a ``B x Q x K`` mask.

    Valid source coordinates must be nondecreasing within every person and
    chromosome. FP64 prefix sums and int64 prefix counts are built separately
    for each such sequence. Outputs retain the input value dtype and query
    order; invalid or neighborless queries receive zero.
    """
    if values.ndim != 3 or values.shape[:2] != source.cm.shape:
        raise ValueError("values and source axes differ")
    if query.cm.shape[0] != source.cm.shape[0]:
        raise ValueError("query and source batch axes differ")
    if not values.is_floating_point():
        raise ValueError("floating values required")
    if type(radius) not in (float, int) or not math.isfinite(radius) or not radius > 0:
        raise ValueError("positive radius required")
    if not bool(torch.isfinite(source.cm[source.valid]).all()):
        raise ValueError("nonfinite valid source cM")
    if not bool(torch.isfinite(query.cm[query.valid]).all()):
        raise ValueError("nonfinite valid query cM")

    batch, queries = query.cm.shape
    width = values.shape[-1]
    # Preserve a zero derivative with respect to values even when every source
    # is missing, while never multiplying invalid NaN padding by zero.
    zero_anchor = torch.where(source.valid[:, :1, None], values[:, :1], 0.).sum(1, keepdim=True) * 0
    pooled = zero_anchor.expand(batch, queries, width).clone()
    counts = torch.zeros((batch, queries), dtype=torch.int64, device=values.device)

    for person in range(batch):
        source_chromosomes = torch.unique(source.chrom[person, source.valid[person]])
        for chromosome in source_chromosomes.tolist():
            source_index = torch.where(source.valid[person]
                                       & (source.chrom[person] == chromosome))[0]
            source_cm = source.cm[person, source_index]
            if source_cm.numel() > 1 and not bool((source_cm[1:] >= source_cm[:-1]).all()):
                raise ValueError("source cM must be nondecreasing within person/chromosome")

            query_index = torch.where(query.valid[person]
                                      & (query.chrom[person] == chromosome))[0]
            if query_index.numel() == 0:
                continue
            query_cm = query.cm[person, query_index]
            selected = values[person, source_index].to(torch.float64)
            if not bool(torch.isfinite(selected).all()):
                raise ValueError("nonfinite valid source value")
            value_prefix = torch.cat((selected.new_zeros((1, width)), selected.cumsum(0)), 0)
            count_prefix = torch.arange(source_index.numel() + 1, dtype=torch.int64,
                                        device=values.device)

            pieces = []
            piece_counts = []
            for start in range(0, query_index.numel(), execution.query_block_size):
                block_cm = query_cm[start:start + execution.query_block_size]
                left, right = _literal_neighbor_bounds(source_cm, block_cm, radius)
                block_count = count_prefix[right] - count_prefix[left]
                block_sum = value_prefix[right] - value_prefix[left]
                block_mean = torch.where(block_count[:, None] > 0,
                                         block_sum / block_count.clamp_min(1)[:, None],
                                         torch.zeros_like(block_sum))
                pieces.append(block_mean.to(values.dtype))
                piece_counts.append(block_count)
            pooled[person, query_index] = torch.cat(pieces, 0)
            counts[person, query_index] = torch.cat(piece_counts, 0)
    return pooled, counts
