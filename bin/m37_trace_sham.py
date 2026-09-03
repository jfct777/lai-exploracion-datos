#!/usr/bin/env python3
"""Create a deterministic locus-permuted TRACE SHAM reference with invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from itertools import permutations

from m33_safe_bridge_core import write_deterministic_npz
from m37_trace_core import require


def make_sham(source: Path, output: Path, seed: int) -> dict:
    with np.load(source, allow_pickle=False) as archive:
        require({"ancestry", "locus_id"}.issubset(archive.files), "SHAM source lacks axes")
        payload = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    names = tuple(x.decode("ascii") if isinstance(x, bytes) else str(x) for x in payload["ancestry"].tolist())
    require(names == ("AFR", "EUR", "NAM"), "SHAM requires AFR/EUR/NAM order")
    count_names = [name for name in ("minor_ac", "callable_an", "fold_minor_ac", "fold_callable_an") if name in payload]
    require(count_names, "SHAM source lacks reference counts")
    loci = len(payload["locus_id"])
    require(loci > 0, "SHAM reference has no loci")
    rng = np.random.default_rng(seed)
    nonidentity = np.asarray([value for value in permutations(range(3)) if value != (0, 1, 2)], dtype=np.uint8)
    ancestry_permutation = nonidentity[rng.integers(0, len(nonidentity), size=loci)]
    for name in count_names:
        value = payload[name]
        require(value.shape[-2:] == (3, loci), "SHAM reference count axes differ")
        shuffled = np.empty_like(value)
        for locus in range(loci):
            shuffled[..., :, locus] = value[..., ancestry_permutation[locus], locus]
        payload[name] = np.ascontiguousarray(shuffled)
        require(np.array_equal(np.sort(value, axis=-2), np.sort(shuffled, axis=-2)),
                "SHAM per-locus count multiset differs")
    permutation_sha256 = hashlib.sha256(ancestry_permutation.tobytes()).hexdigest()
    payload["sham_seed"] = np.asarray([seed], dtype=np.int64)
    payload["sham_ancestry_permutation_sha256"] = np.asarray([permutation_sha256])
    payload["sham_source_sha256"] = np.asarray([hashlib.sha256(source.read_bytes()).hexdigest()])
    require(not output.exists(), "refusing to overwrite SHAM reference")
    write_deterministic_npz(output, {name: payload[name] for name in sorted(payload)})
    receipt = {"schema_version": "1.0.0", "stage": "M37_TRACE_SHAM_REFERENCE", "status": "PASS_DETERMINISTIC_LOCUS_PERMUTATION",
               "seed": seed, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "locus_axis_sha256": hashlib.sha256(payload["locus_id"].tobytes()).hexdigest(),
               "ancestry_permutation_sha256": permutation_sha256,
               "permutation_unit": "ancestry_labels_within_each_locus",
               "invariants": ["ancestry_order_unchanged", "locus_axis_unchanged",
                              "per_locus_count_multiset_preserved", "pooled_frequency_geometry_preserved"]}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(make_sham(args.reference, args.output, args.seed), sort_keys=True))
