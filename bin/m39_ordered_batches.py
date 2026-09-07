"""Bounded, label-free tensor batches from authenticated ordered context stores.

Callers open stores with OrderedContextStore.open and an expected manifest hash.
This adapter retains complete windows, pads only at the right, and never sends
sample/candidate IDs to a model. The memory budget covers padded NumPy staging,
copied tensors and conservative reader/check workspaces, not model activations,
the shared source mmap, allocator overhead or total process RSS.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import torch

from m39_ordered_context import OrderedContextStore


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pairs_and_lengths(store: OrderedContextStore, pairs) -> tuple[list[tuple[int, int]], list[int]]:
    require(isinstance(store, OrderedContextStore), "authenticated ordered store required")
    require(isinstance(pairs, (list, tuple)) and len(pairs) > 0, "nonempty list of query/anchor pairs required")
    checked, lengths = [], []
    for pair in pairs:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2, "query/anchor pair required")
        require(all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) for value in pair),
                "pair indices must be integers, not booleans")
        query, anchor = map(int, pair)
        require(0 <= query < store.shape[0] and 0 <= anchor < store.shape[1], "pair index out of range")
        lo, hi = store.window_bounds(anchor)
        checked.append((query, anchor))
        lengths.append(hi - lo)
    return checked, lengths


def _layout(batch: int, k: int, sites: int) -> dict:
    candidates = (batch, 2, 3, k)
    return {
        "channels": ((*candidates, sites, 5), np.dtype("float32")),
        "delta_cm": ((batch, sites), np.dtype("float32")),
        "site_mask": ((batch, sites), np.dtype("bool")),
        "candidate_mask": (candidates, np.dtype("bool")),
        "reference_dosage": (candidates, np.dtype("float32")),
        "reference_observed": (candidates, np.dtype("bool")),
        "query_dosage": ((batch,), np.dtype("float32")),
        "query_observed": ((batch,), np.dtype("bool")),
        "radius_cm": ((batch,), np.dtype("float32")),
        "pooled_af": ((batch, 3), np.dtype("float32")),
        "pooled_observed": ((batch, 3), np.dtype("bool")),
    }


def _memory_plan(store: OrderedContextStore, lengths: list[int]) -> dict[str, int]:
    batch, maximum = len(lengths), max(lengths)
    # All-empty batches have one masked padding position, not a fabricated site.
    padded = max(1, maximum)
    k = store.shape[2]
    elements = lambda shape: int(np.prod(shape, dtype=object))
    staging = sum(elements(shape) * dtype.itemsize for shape, dtype in _layout(batch, k, padded).values())
    window = store.window_working_bytes(maximum)
    validation = 16 * 6 * k * maximum + 4096 + 64 * batch
    pooled = 64 * len(store.arrays["reference_ancestry"])
    return {"batch_size": batch, "padded_sites": padded, "maximum_window_sites": maximum,
            "numpy_staging_bytes": staging, "tensor_bytes": staging,
            "reader_workspace_bytes": window, "validation_workspace_bytes": validation,
            "pooled_workspace_bytes": pooled,
            "required_bytes": 2 * staging + window + validation + pooled}


def estimate_batch_bytes(store: OrderedContextStore, pairs) -> dict[str, int]:
    """Plan the full padded allocation without loading/expanding any window."""
    _, lengths = _pairs_and_lengths(store, pairs)
    return _memory_plan(store, lengths)


def _validate_window(window, query: int, anchor: int, length: int, k: int, radius: float) -> None:
    candidate_shape = (2, 3, k)
    channels = window.channels
    require(window.query_index == query and window.anchor_index == anchor
            and window.site_offset == 0 and window.window_site_count == length,
            "reader did not return the complete requested window")
    require(channels.shape == (*candidate_shape, length, 5) and channels.dtype == np.uint8
            and np.all((channels == 0) | (channels == 1)), "invalid binary common channels")
    require(window.delta_cm.shape == (length,) and window.delta_cm.dtype == np.float64
            and np.isfinite(window.delta_cm).all() and np.all(np.abs(window.delta_cm) <= radius),
            "invalid full-precision relative coordinates")
    require(window.common_site_index.shape == (length,)
            and np.all(np.diff(window.common_site_index) == 1), "window site order differs")
    for mask in (window.candidate_mask, window.ref_observed):
        require(mask.shape == candidate_shape and mask.dtype == np.bool_, "invalid explicit candidate mask")
    require(window.ref_dosage.shape == candidate_shape
            and np.all((window.ref_dosage >= 0) & (window.ref_dosage <= 2))
            and np.all(window.ref_dosage[~window.ref_observed] == 0)
            and np.all(~window.ref_observed | window.candidate_mask), "invalid diploid reference attachment")
    require(0 <= window.query_dosage <= 2 and
            (window.query_observed or window.query_dosage == 0), "invalid diploid query attachment")
    require(not channels[~window.candidate_mask].any(), "unsupported candidate has common channels")
    require(np.array_equal(channels[..., 0] + channels[..., 1], channels[..., 4])
            and not np.any((channels[..., 2] | channels[..., 3]) & channels[..., 4]),
            "called/missing channel semantics differ")
    require(length > 0 or not window.candidate_mask.any(), "empty window has supported candidates")


def pack_batch(store: OrderedContextStore, pairs, max_input_bytes: int,
               device="cpu") -> dict[str, torch.Tensor]:
    """Copy complete query/anchor windows into an ID-free padded tensor dict.

    Delta cM is calculated by the reader in float64 before conversion to float32.
    POOLED frequencies count every observed diploid REF person in each ancestry,
    independent of top-K candidates; zero support is 0 with observed=False.
    Repeated pairs are allowed and preserve the caller's order. Caller-owned
    input/store arrays are never mutated or shared with the returned tensors.
    """
    require(type(max_input_bytes) is int and max_input_bytes > 0, "positive integer input byte ceiling required")
    checked, lengths = _pairs_and_lengths(store, pairs)
    plan = _memory_plan(store, lengths)
    require(plan["required_bytes"] <= max_input_bytes, "complete padded batch exceeds input byte ceiling")
    target = torch.device(device)
    require(target.type in ("cpu", "cuda"), "only materialized CPU/CUDA tensors are supported")
    if target.type == "cuda":
        require(torch.cuda.is_available(), "requested CUDA device is unavailable")
    radius = float(store.arrays["radius_cm"][0])
    require(np.isfinite(radius) and radius > 0, "invalid radius")
    layout = _layout(len(checked), store.shape[2], plan["padded_sites"])
    staging = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in layout.items()}
    staging["radius_cm"].fill(radius)
    ancestry = store.arrays["reference_ancestry"]
    for row, ((query, anchor), length) in enumerate(zip(checked, lengths)):
        window = store.read_window(query, anchor, max_window_bytes=plan["reader_workspace_bytes"])
        _validate_window(window, query, anchor, length, store.shape[2], radius)
        staging["channels"][row, ..., :length, :] = window.channels
        staging["delta_cm"][row, :length] = window.delta_cm
        staging["site_mask"][row, :length] = True
        staging["candidate_mask"][row] = window.candidate_mask
        staging["reference_dosage"][row] = window.ref_dosage
        staging["reference_observed"][row] = window.ref_observed
        staging["query_dosage"][row] = window.query_dosage
        staging["query_observed"][row] = window.query_observed
        # Query and reference homolog multiplicity never enter this denominator.
        for group in range(3):
            observed = (ancestry == group) & store.arrays["ref_observed"][anchor].astype(bool)
            count = int(np.count_nonzero(observed))
            if count:
                total = int(np.sum(store.arrays["ref_dosage"][anchor, observed], dtype=np.int64))
                frequency = total / (2 * count)
                require(np.isfinite(frequency) and 0 <= frequency <= 1, "invalid pooled allele frequency")
                staging["pooled_af"][row, group] = frequency
                staging["pooled_observed"][row, group] = True
        del window
    # Copying is explicit even on CPU: tensor mutation cannot change staging or a source mmap.
    tensors = {name: torch.tensor(array, device=target) for name, array in staging.items()}
    require(sum(array.nbytes for array in staging.values()) == plan["numpy_staging_bytes"],
            "batch memory plan drift")
    return tensors
