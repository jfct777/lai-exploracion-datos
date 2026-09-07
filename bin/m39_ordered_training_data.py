"""Truth binding, paired sampling and a same-locus reference-link control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from m39_anchor_screen import metrics, sha256, validate_partition
from m39_ordered_context import OrderedContextStore
from m39_ordered_models import STATE_NAMES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bind_development(path: Path, expected_sha256: str, train: OrderedContextStore,
                     select: OrderedContextStore) -> dict:
    require(path.is_file() and not path.is_symlink() and sha256(path) == expected_sha256,
            "development binding hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    fields = {"alt", "anchor_indices", "baseline", "chrom", "coords", "full_baseline",
              "locus_id", "pos", "ref", "sample_key_sha256", "select_indices",
              "source_indices", "state_names", "train_indices", "truth_state"}
    require(set(data) == fields, "development field inventory differs; SCORE is forbidden")
    require(np.array_equal(data["state_names"], np.asarray(STATE_NAMES, dtype="S2")),
            "six-state label order differs")
    fit_indices, select_indices = validate_partition(data)
    keys = data["sample_key_sha256"]
    require(keys.dtype == np.dtype("S64") and len(set(keys.tolist())) == len(keys),
            "development key schema differs")
    require(not set(train.arrays["sample_key_sha256"]) & set(select.arrays["sample_key_sha256"]),
            "TRAIN and SELECT stores overlap")
    lookup = {bytes(key): i for i, key in enumerate(keys)}
    result = {}
    for role, store, expected in (("train", train, fit_indices), ("select", select, select_indices)):
        store_keys = store.arrays["sample_key_sha256"]
        require(set(store_keys.tolist()) == set(keys[expected].tolist()), f"{role} sample universe differs")
        order = np.asarray([lookup[bytes(key)] for key in store_keys], dtype=np.int64)
        for source, bound in (("locus_id", "locus_id"), ("pos", "pos"), ("ref", "ref"),
                              ("alt", "alt"), ("chrom", "chrom"), ("anchor_indices", "anchor_indices"),
                              ("cM", "coords")):
            require(np.array_equal(store.arrays[source], data[bound]), f"{role} {source} axis differs")
        result[role] = {key: data[key][order].copy()
                        for key in ("truth_state", "baseline", "full_baseline", "sample_key_sha256")}
        # Validate both comparators before any training. This does not choose a recipe.
        metrics(result[role]["baseline"], result[role]["truth_state"])
        metrics(result[role]["full_baseline"], result[role]["truth_state"])
    require(train.shape[2] == select.shape[2], "TRAIN/SELECT candidate budgets differ")
    for key in ("reference_sample_key_sha256", "reference_ancestry", "reference_hap_original_index",
                "common_ref", "common_cm", "common_pos", "common_ref_allele", "common_alt_allele",
                "common_locus_id", "minor_is_alt", "radius_cm", "ref_dosage", "ref_observed"):
        require(np.array_equal(train.arrays[key], select.arrays[key]), f"shared REF/context {key} differs")
    return result


def fixed_anchor_subset(count: int, available: int, seed: int) -> np.ndarray:
    require(type(count) is int and type(available) is int and 0 < count <= available,
            "invalid anchor subset size")
    require(type(seed) is int and 0 <= seed < 2**63, "invalid sampling seed")
    # No genotype, ancestry label, prediction or outcome is inspected.
    return np.sort(np.random.default_rng(seed).choice(available, count, replace=False)).astype(np.int64)


def paired_batches(people: int, anchors: np.ndarray, batch_size: int,
                   steps: int, seed: int):
    """Interleave shuffled anchors, retaining same-anchor minibatches and remainders.

    Every arm uses the same seed and number of optimizer steps. The receipt records
    actual observations. One minibatch per anchor is emitted in each round before
    another person batch from that anchor. This prevents a short step budget from
    seeing only the first few anchors. Grouping avoids mixed-length padding; these
    gradients and observations remain dependent.
    """
    require(all(type(x) is int and x > 0 for x in (people, batch_size, steps)),
            "positive people/batch/steps required")
    require(anchors.ndim == 1 and anchors.dtype.kind in "iu" and len(anchors) > 0
            and len(np.unique(anchors)) == len(anchors), "invalid anchor inventory")
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < steps:
        person_orders = {int(anchor): rng.permutation(people) for anchor in anchors}
        for begin in range(0, people, batch_size):
            for anchor in rng.permutation(anchors):
                order = person_orders[int(anchor)]
                yield [(int(query), int(anchor)) for query in order[begin:begin + batch_size]]
                produced += 1
                if produced == steps:
                    return


def reference_link_permutation(store: OrderedContextStore, seed: int) -> np.ndarray:
    """Map each REF person to another within ancestry/call-status at every locus.

    The map precedes candidate attachment. It is shared across all query people,
    both homologs and every occurrence of that REF at the same locus. It preserves
    ancestry-specific allele totals and masks, not rare LD across different loci.
    """
    require(type(seed) is int and 0 <= seed < 2**63, "invalid sham seed")
    a = store.arrays
    people = len(a["reference_ancestry"])
    mapping = np.tile(np.arange(people, dtype=np.int64), (len(a["locus_id"]), 1))
    for anchor, locus in enumerate(a["locus_id"]):
        digest = hashlib.sha256(f"m39-reference-link-v1|{seed}|{int(locus)}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        for ancestry in range(3):
            for observed in (False, True):
                group = np.flatnonzero((a["reference_ancestry"] == ancestry)
                                      & (a["ref_observed"][anchor].astype(bool) == observed))
                # Key order makes source row permutations irrelevant.
                group = group[np.argsort(a["reference_sample_key_sha256"][group], kind="stable")]
                mapping[anchor, group] = rng.permutation(group)
    return mapping


def apply_reference_link_sham(batch: dict, store: OrderedContextStore, pairs,
                              permutation: np.ndarray) -> dict:
    require(all(value.device.type == "cpu" for value in batch.values()),
            "sham attachment must precede device transfer")
    a = store.arrays
    require(permutation.shape == a["ref_dosage"].shape and permutation.dtype.kind in "iu",
            "invalid reference permutation schema")
    changed = dict(batch)
    dose = batch["reference_dosage"].clone()
    for row, (query, anchor) in enumerate(pairs):
        mask = a["candidate_mask"][query, anchor].astype(bool)
        reference = a["candidate_ref_index"][query, anchor][mask]
        permuted = permutation[anchor, reference]
        require(np.array_equal(a["reference_ancestry"][reference], a["reference_ancestry"][permuted])
                and np.array_equal(a["ref_observed"][anchor, reference],
                                   a["ref_observed"][anchor, permuted]), "sham changed ancestry or mask")
        values = np.zeros(mask.shape, dtype=np.float32)
        values[mask] = a["ref_dosage"][anchor, permuted]
        dose[row] = torch.from_numpy(values)
    changed["reference_dosage"] = dose
    return changed
