"""Label-free anchor batches for the experimental multichannel FLARE adapter.

Open stores with their expected manifest hashes and obtain role-aligned baselines
with ``bind_development`` before calling this bridge. No truth field, person ID,
candidate ID or haplotype-specific rare dosage is forwarded to the model. This
reuses complete ordered windows; it does not construct neighboring rare events
or a dense inference grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import weakref

import numpy as np
import torch

from m39_ordered_batches import estimate_batch_bytes, pack_batch
from m39_ordered_context import OrderedContextStore
from m39_ordered_training_data import apply_reference_link_sham, require


COMMON_FIELDS = frozenset(("channels", "delta_cm", "site_mask", "candidate_mask", "radius_cm"))
QUERY_FIELDS = frozenset(("query_dosage", "query_observed"))
DETAIL_FIELDS = frozenset(("reference_dosage", "reference_observed"))
SUMMARY_FIELDS = frozenset(("ref_dosage_counts", "ref_eligible"))
MODES = ("NONE", "SUMMARY", "DETAIL", "BOTH", "OFF", "SHAM")
_SUMMARY_AXES = ("reference_sample_key_sha256", "reference_ancestry", "ref_dosage",
                 "ref_observed", "anchor_indices", "locus_id", "chrom", "pos", "ref",
                 "alt", "cM", "minor_is_alt")


@dataclass(frozen=True)
class ReferenceSummary:
    """Small, immutable counts cached once from all unique diploid REF persons.

    The store contract rejects duplicate REF-person keys. Homologs and candidate
    slots never enter these counts. Eligible persons include missing rare calls;
    the three genotype bins include only observed calls, including genuine zero.
    A shared summary is allowed across TRAIN/SELECT after exact REF/locus checks.
    Counts describe support, not independent biological sample sizes.
    """

    ref_dosage_counts: np.ndarray
    ref_eligible: np.ndarray
    _source: OrderedContextStore = field(repr=False, compare=False)
    _bound_stores: weakref.WeakSet = field(default_factory=weakref.WeakSet,
                                          repr=False, compare=False)

    @classmethod
    def from_store(cls, store: OrderedContextStore) -> "ReferenceSummary":
        require(isinstance(store, OrderedContextStore), "authenticated ordered store required")
        a = store.arrays
        keys, ancestry = a["reference_sample_key_sha256"], a["reference_ancestry"]
        require(len(np.unique(keys)) == len(keys), "duplicate REF persons")
        counts = np.zeros((store.shape[1], 3, 3), dtype=np.int64)
        eligible = np.zeros((store.shape[1], 3), dtype=np.int64)
        for group in range(3):
            persons = ancestry == group
            eligible[:, group] = np.count_nonzero(persons)
            # Working arrays have [anchor,person], never [query,homolog,K].
            dose = a["ref_dosage"][:, persons]
            observed = a["ref_observed"][:, persons].astype(bool)
            for genotype in range(3):
                counts[:, group, genotype] = np.count_nonzero(observed & (dose == genotype), axis=1)
        require(np.all(counts.sum(axis=-1) <= eligible), "observed count exceeds eligible REF")
        counts.flags.writeable = eligible.flags.writeable = False
        result = cls(counts, eligible, store)
        result._bound_stores.add(store)
        return result

    def bind_store(self, store: OrderedContextStore) -> None:
        """Validate identical summary source once for another immutable store."""
        require(isinstance(store, OrderedContextStore), "authenticated ordered store required")
        if store not in self._bound_stores:
            for key in _SUMMARY_AXES:
                require(np.array_equal(self._source.arrays[key], store.arrays[key]),
                        f"summary REF/locus binding differs: {key}")
            self._bound_stores.add(store)


def _baseline_rows(baseline: np.ndarray, store: OrderedContextStore, pairs) -> np.ndarray:
    # Authentication and exact role/locus joining are bind_development's job;
    # accepting its baseline array does not require its truth dictionary here.
    require(isinstance(baseline, np.ndarray) and baseline.dtype == np.dtype("float32")
            and baseline.shape == (store.shape[0], store.shape[1], 6),
            "role-bound baseline must be float32 [people,anchors,6]")
    rows = baseline[tuple(np.asarray(pairs, dtype=np.int64).T)].copy()
    require(np.isfinite(rows).all() and np.all((rows >= 0) & (rows <= 1))
            and np.allclose(rows.sum(axis=-1), 1, atol=8 * np.finfo(np.float32).eps, rtol=0),
            "baseline rows must be finite probabilities summing to one")
    return rows


def _validate_sham(store: OrderedContextStore, pairs, permutation: np.ndarray) -> None:
    """Check whole-person permutations, not just their retrieved occurrences."""
    a = store.arrays
    require(isinstance(permutation, np.ndarray) and permutation.dtype.kind in "iu"
            and permutation.shape == a["ref_dosage"].shape,
            "invalid reference permutation schema")
    expected = np.arange(len(a["reference_ancestry"]), dtype=np.int64)
    for anchor in sorted({int(anchor) for _, anchor in pairs}):
        row = permutation[anchor]
        require(np.array_equal(np.sort(row), expected), "SHAM must be a bijection of REF persons")
        require(np.array_equal(a["reference_ancestry"][row], a["reference_ancestry"]),
                "SHAM changed REF ancestry")
        require(np.array_equal(a["ref_observed"][anchor, row], a["ref_observed"][anchor]),
                "SHAM changed observed-call strata")


def estimate_multichannel_batch_bytes(store: OrderedContextStore, pairs, *, mode: str) -> dict:
    """Conservative input budget including transient legacy packing and copies.

    As with pack_batch this excludes model activations, source mmaps, the cached
    global summary, allocator overhead and total RSS. The extra allowance includes
    NumPy/tensor copies and a possible CPU-to-device copy for new fields.
    """
    require(mode in MODES, "unknown multichannel mode")
    plan = estimate_batch_bytes(store, pairs)
    per_row = 6 * np.dtype("float32").itemsize
    if mode in ("SUMMARY", "BOTH", "SHAM"):
        per_row += 12 * np.dtype("int64").itemsize
    extra = 3 * len(pairs) * per_row
    # apply_reference_link_sham clones the dosage tensor before attachment.
    if mode == "SHAM":
        extra += len(pairs) * 2 * 3 * store.shape[2] * np.dtype("float32").itemsize
    return {**plan, "multichannel_extra_bytes": int(extra),
            "required_bytes": int(plan["required_bytes"] + extra)}


def pack_multichannel_batch(store: OrderedContextStore, pairs, baseline: np.ndarray,
                           summary: ReferenceSummary | None, *, mode: str,
                           max_input_bytes: int, sham_permutation: np.ndarray | None = None,
                           device="cpu") -> dict[str, torch.Tensor]:
    """Return a flat tensor pack suitable for DeviceRuntime.transfer.

    ``baseline`` is the Fminus array for this exact store/role returned by the
    authenticated development binder. Pass the returned dict as both ``common``
    and ``rare`` to OrderedMultichannelAdapter, and its ``baseline`` separately.
    SHAM maps to model mode BOTH. Labels remain solely in the trainer/evaluator.

    Common retrieval, loci, ancestry lanes and masks are unchanged across arms.
    SHAM uses the existing per-locus reference_link_permutation before attachment
    and preserves the full REF histogram. It tests the individual link conditional
    on SUMMARY, not an exact null of all rare information or interlocus rare LD.
    NONE/OFF expose no rare tensors; SUMMARY/DETAIL expose only their own branch.
    """
    require(type(max_input_bytes) is int and max_input_bytes > 0,
            "positive integer input byte ceiling required")
    plan = estimate_multichannel_batch_bytes(store, pairs, mode=mode)
    require(plan["required_bytes"] <= max_input_bytes,
            "complete multichannel batch exceeds input byte ceiling")
    require((mode == "SHAM") == (sham_permutation is not None),
            "SHAM requires a permutation; other modes forbid it")
    target = torch.device(device)
    require(target.type in ("cpu", "cuda"), "only materialized CPU/CUDA tensors are supported")
    if target.type == "cuda":
        require(torch.cuda.is_available(), "requested CUDA device is unavailable")
    rows = _baseline_rows(baseline, store, pairs)
    if mode in ("SUMMARY", "BOTH", "SHAM"):
        require(isinstance(summary, ReferenceSummary), "global REF summary required")
        summary.bind_store(store)
    if sham_permutation is not None:
        _validate_sham(store, pairs, sham_permutation)
    if mode == "OFF":
        return {"baseline": torch.tensor(rows, device=target)}
    packed = pack_batch(store, pairs, max_input_bytes=max_input_bytes - plan["multichannel_extra_bytes"])
    if sham_permutation is not None:
        packed = apply_reference_link_sham(packed, store, pairs, sham_permutation)
    fields = set(COMMON_FIELDS)
    if mode != "NONE":
        fields.update(QUERY_FIELDS)
    if mode in ("DETAIL", "BOTH", "SHAM"):
        fields.update(DETAIL_FIELDS)
    result = {name: packed[name] for name in sorted(fields)}
    result["baseline"] = torch.tensor(rows)
    if mode in ("SUMMARY", "BOTH", "SHAM"):
        anchors = np.asarray([anchor for _, anchor in pairs], dtype=np.int64)
        result["ref_dosage_counts"] = torch.tensor(summary.ref_dosage_counts[anchors])
        result["ref_eligible"] = torch.tensor(summary.ref_eligible[anchors])
    return {name: value.to(target) for name, value in result.items()}
