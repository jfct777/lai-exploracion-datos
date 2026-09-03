#!/usr/bin/env python3
"""Create the single frozen M38B SHAM with strict three-way derangements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz, write_exclusive_json
from m37_trace_core import require


DERANGEMENTS = np.asarray(((1, 2, 0), (2, 0, 1)), dtype=np.uint8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_strict_sham(source: Path, source_receipt: Path, output: Path,
                     receipt_output: Path, seed: int) -> dict[str, object]:
    require(seed == 3401103, "M38B SHAM seed differs")
    source_document = json.loads(source_receipt.read_text(encoding="utf-8"))
    require(
        source_document.get("stage") == "M38B_APPLY_FROZEN_LOO_PRIMARY_MASK"
        and source_document.get("decision") == "PASS_PRIMARY_FACTORS_FROZEN_FOR_MODEL"
        and source_document.get("counts", {}).get("primary_loci") == 123
        and source_document.get("outputs", {}).get(source.name, {}).get("sha256") == sha256(source),
        "M38B primary reference receipt differs",
    )
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    names = tuple(value.decode("ascii") if isinstance(value, bytes) else str(value)
                  for value in payload["ancestry"].tolist())
    require(names == ("AFR", "EUR", "NAM"), "M38B SHAM ancestry order differs")
    count_names = [name for name in (
        "minor_ac", "callable_an", "fold_minor_ac", "fold_callable_an",
    ) if name in payload]
    require(bool(count_names), "M38B SHAM source lacks count arrays")
    loci = len(payload["locus_id"])
    require(loci > 0, "M38B SHAM cannot permute zero loci")
    source_ac = np.asarray(payload["minor_ac"], dtype=np.int64)
    source_an = np.asarray(payload["callable_an"], dtype=np.int64)
    require(
        np.all(source_an > 0) and np.all((0 <= source_ac) & (source_ac <= source_an))
        and np.allclose(payload["minor_af"], source_ac / source_an, rtol=0, atol=1e-12)
        and np.array_equal(payload["observed_mask"], (source_an > 0).astype(np.uint8))
        and np.array_equal(payload["no_support"],
                           ((source_an > 0) & (source_ac == 0)).astype(np.uint8)),
        "M38B SHAM source derived frequency fields are inconsistent",
    )
    choices = np.arange(loci, dtype=np.uint8) % 2
    rng = np.random.default_rng(seed)
    choices = choices[rng.permutation(loci)]
    permutations = DERANGEMENTS[choices]
    require(np.all(permutations != np.arange(3, dtype=np.uint8)[None, :]),
            "M38B SHAM contains a fixed ancestry label")
    before_hashes: dict[str, str] = {}
    for name in count_names:
        value = payload[name]
        require(value.shape[-2:] == (3, loci), "M38B SHAM count axes differ")
        before_hashes[name] = hashlib.sha256(
            np.sort(value, axis=-2).tobytes(order="C")
        ).hexdigest()
        shuffled = np.empty_like(value)
        for locus in range(loci):
            shuffled[..., :, locus] = value[..., permutations[locus], locus]
        require(np.array_equal(np.sort(value, axis=-2), np.sort(shuffled, axis=-2)),
                "M38B SHAM changed a per-locus count multiset")
        payload[name] = np.ascontiguousarray(shuffled)
    # These fields are derived from the permuted counts, not independent data.
    # Recompute them so the SHAM artifact cannot carry contradictory semantics.
    shuffled_ac = np.asarray(payload["minor_ac"], dtype=np.int64)
    shuffled_an = np.asarray(payload["callable_an"], dtype=np.int64)
    payload["minor_af"] = np.ascontiguousarray(shuffled_ac / shuffled_an, dtype=np.float64)
    payload["observed_mask"] = np.ascontiguousarray((shuffled_an > 0).astype(np.uint8))
    payload["no_support"] = np.ascontiguousarray(
        ((shuffled_an > 0) & (shuffled_ac == 0)).astype(np.uint8)
    )
    permutation_sha = hashlib.sha256(permutations.tobytes(order="C")).hexdigest()
    payload["sham_seed"] = np.asarray([seed], dtype=np.int64)
    payload["sham_ancestry_permutation_sha256"] = np.asarray([permutation_sha])
    payload["sham_source_sha256"] = np.asarray([sha256(source)])
    require(not output.exists() and not receipt_output.exists(),
            "refusing to overwrite M38B SHAM outputs")
    write_deterministic_npz(output, {name: payload[name] for name in sorted(payload)})
    document: dict[str, object] = {
        "schema_version": "1.0.0",
        "stage": "M38B_STRICT_SHAM_REFERENCE",
        "status": "PASS_SINGLE_FROZEN_STRICT_DERANGEMENT",
        "seed": seed,
        "source_sha256": sha256(source),
        "source_receipt_sha256": sha256(source_receipt),
        "output_sha256": sha256(output),
        "loci": loci,
        "derangements": DERANGEMENTS.tolist(),
        "derangement_counts": [int(np.count_nonzero(choices == value)) for value in (0, 1)],
        "approximately_balanced": abs(int(np.count_nonzero(choices == 0)) -
                                      int(np.count_nonzero(choices == 1))) <= 1,
        "fixed_ancestry_labels": 0,
        "per_locus_count_multiset_preserved": True,
        "permutation_sha256": permutation_sha,
        "sorted_count_multiset_sha256": before_hashes,
        "claim_conditioning": "ONE_PRE_FROZEN_SHAM_REALISATION",
    }
    write_exclusive_json(receipt_output, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    document = make_strict_sham(args.reference, args.source_receipt, args.output,
                                args.receipt, args.seed)
    print(json.dumps({"status": document["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
