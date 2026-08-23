#!/usr/bin/env python3
"""Deterministic M33 packed-context oracle and storage estimator.

This module implements the frozen 13-channel semantics for small fixtures and
for equivalence checks.  It deliberately does not publish GCS objects or emit
READY: the persistent packed layout must pass the proportionality gate before
it can become a production writer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import m33_m0_contract as contract


ANCESTRIES = (b"AFR", b"EUR", b"ASIA")
RADII = (0.05, 0.1, 0.2, 0.5)
PERSON_BATCH = 8
TOKEN_BUDGET = 262_144
TOKEN_BYTES = 13 * 4 + 1 + 8

PRODUCTIVE_MEMBERS = {
    "selected": {"locus_id", "chrom", "pos", "ref", "alt", "cM"},
    "target": {"sample_key_sha256", "locus_id", "minor_dosage", "observed_mask"},
    "reference": {
        "ancestry", "locus_id", "minor_ac", "callable_an", "minor_af",
        "observed_mask", "no_support",
    },
    "f0": {
        "sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref",
        "marker_alt", "F0",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_dtype(value: np.ndarray, dtype: str, name: str) -> None:
    require(value.dtype == np.dtype(dtype), f"{name} dtype differs: {value.dtype} != {dtype}")
    require(value.flags.c_contiguous, f"{name} must be C-contiguous")


def load_productive_npz(path: Path, kind: str) -> dict[str, np.ndarray]:
    """Load one exact productive artifact; technical KAT schemas fail closed."""
    require(kind in PRODUCTIVE_MEMBERS, f"unknown artifact kind: {kind}")
    require(path.is_file() and not path.is_symlink(), f"invalid artifact path: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == PRODUCTIVE_MEMBERS[kind],
                f"{kind} member inventory differs from productive schema")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    require(all(value.dtype.kind != "O" for value in arrays.values()), "object arrays are forbidden")
    return arrays


def validate_inputs(
    selected: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    f0: Mapping[str, np.ndarray],
    marker_cm: np.ndarray,
) -> None:
    """Validate exact axes, dtypes and value domains before any shard is built."""
    require(set(selected) == PRODUCTIVE_MEMBERS["selected"], "selected members differ")
    require(set(target) == PRODUCTIVE_MEMBERS["target"], "target members differ")
    require(set(reference) == PRODUCTIVE_MEMBERS["reference"], "reference members differ")
    require(set(f0) == PRODUCTIVE_MEMBERS["f0"], "F0 members differ")

    dtype_contracts = {
        "selected": {"locus_id": "<u8", "chrom": "|u1", "pos": "<i8", "ref": "|S1",
                     "alt": "|S1", "cM": "<f8"},
        "target": {"sample_key_sha256": "|S64", "locus_id": "<u8", "minor_dosage": "|i1",
                   "observed_mask": "|u1"},
        "reference": {"ancestry": "|S4", "locus_id": "<u8", "minor_ac": "<u2",
                      "callable_an": "<u2", "minor_af": "<f8", "observed_mask": "|u1",
                      "no_support": "|u1"},
        "f0": {"sample_key_sha256": "|S64", "marker_chrom": "|u1", "marker_pos": "<i8",
               "marker_ref": "|S1", "marker_alt": "|S1", "F0": "<f4"},
    }
    for group_name, group in (("selected", selected), ("target", target),
                              ("reference", reference), ("f0", f0)):
        for name, dtype in dtype_contracts[group_name].items():
            _exact_dtype(np.asarray(group[name]), dtype, f"{group_name}.{name}")

    locus_count = int(np.asarray(selected["locus_id"]).size)
    sample_count = int(np.asarray(target["sample_key_sha256"]).size)
    marker_count = int(np.asarray(f0["marker_pos"]).size)
    require(locus_count > 0 and sample_count > 0 and marker_count > 0, "empty axis")
    require(all(np.asarray(selected[name]).shape == (locus_count,) for name in PRODUCTIVE_MEMBERS["selected"]),
            "selected axis differs")
    require(np.asarray(target["locus_id"]).shape == (locus_count,) and
            np.asarray(target["minor_dosage"]).shape == (sample_count, locus_count) and
            np.asarray(target["observed_mask"]).shape == (sample_count, locus_count),
            "TARGET axes differ")
    require(np.asarray(reference["ancestry"]).shape == (3,) and
            np.asarray(reference["locus_id"]).shape == (locus_count,) and
            all(np.asarray(reference[name]).shape == (3, locus_count)
                for name in ("minor_ac", "callable_an", "minor_af", "observed_mask", "no_support")),
            "REF axes differ")
    require(np.asarray(f0["sample_key_sha256"]).shape == (sample_count,) and
            all(np.asarray(f0[name]).shape == (marker_count,)
                for name in ("marker_chrom", "marker_pos", "marker_ref", "marker_alt")) and
            np.asarray(f0["F0"]).shape == (sample_count, 2, marker_count, 3), "F0 axes differ")

    locus_ids = np.asarray(selected["locus_id"])
    order = np.lexsort((locus_ids, np.asarray(selected["pos"]), np.asarray(selected["cM"])))
    require(np.array_equal(order, np.arange(locus_count)), "rare loci are not ordered by cM/bp/locus_id")
    require(np.unique(locus_ids).size == locus_count and
            np.array_equal(np.asarray(target["locus_id"]), locus_ids) and
            np.array_equal(np.asarray(reference["locus_id"]), locus_ids), "locus axes differ")
    require(np.array_equal(np.asarray(target["sample_key_sha256"]),
                           np.asarray(f0["sample_key_sha256"])), "TARGET/F0 sample axes differ")
    require(np.array_equal(np.asarray(reference["ancestry"]), np.asarray(ANCESTRIES, dtype="|S4")),
            "ancestry order differs")
    require(np.all(np.asarray(selected["chrom"]) == 22) and
            np.all(np.asarray(f0["marker_chrom"]) == 22), "only chr22 is accepted")
    require(np.all(np.isin(np.asarray(target["minor_dosage"]), [0, 1, 2])) and
            np.all(np.isin(np.asarray(target["observed_mask"]), [0, 1])), "TARGET domain differs")
    require(np.all((np.asarray(target["observed_mask"]) == 1) |
                   (np.asarray(target["minor_dosage"]) == 0)), "missing TARGET dosage is nonzero")
    ac, an = np.asarray(reference["minor_ac"]), np.asarray(reference["callable_an"])
    af = np.asarray(reference["minor_af"])
    require(np.all(ac <= an) and np.all(np.isfinite(af)) and
            np.allclose(af, np.divide(ac, an, out=np.zeros_like(af), where=an > 0), rtol=0, atol=1e-12),
            "REF AC/AN/AF differ")
    require(np.all(np.isin(np.asarray(reference["observed_mask"]), [0, 1])) and
            np.array_equal(np.asarray(reference["observed_mask"]), (an > 0).astype("|u1")),
            "REF observed mask differs")
    require(np.array_equal(np.asarray(reference["no_support"]),
                           ((an > 0) & (ac == 0)).astype("|u1")), "REF no-support mask differs")
    probabilities = np.asarray(f0["F0"])
    require(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and
            np.allclose(probabilities.sum(axis=3), 1.0, rtol=0, atol=5e-6), "F0 simplex differs")
    marker_cm = np.asarray(marker_cm)
    _exact_dtype(marker_cm, "<f8", "marker_cM")
    require(marker_cm.shape == (marker_count,) and np.all(np.isfinite(marker_cm)) and
            np.all(marker_cm[:-1] <= marker_cm[1:]), "marker cM axis differs")


def base_channels(target: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray],
                  max_callable_an: Mapping[str, int]) -> np.ndarray:
    """Return the eleven non-geometric channels as sample × locus × channel."""
    require(set(max_callable_an) == {"AFR", "EUR", "ASIA"} and
            all(type(value) is int and value > 0 for value in max_callable_an.values()),
            "FIT max callable AN differs")
    dosage = np.asarray(target["minor_dosage"], dtype="<f8")
    observed = np.asarray(target["observed_mask"], dtype="<f8")
    samples, loci = dosage.shape
    values = np.empty((samples, loci, 11), dtype="<f8")
    values[:, :, 0] = np.clip(dosage / 2.0, 0.0, 1.0)
    values[:, :, 1] = observed
    for ancestry_index, ancestry in enumerate(("AFR", "EUR", "ASIA")):
        offset = 2 + ancestry_index * 3
        values[:, :, offset] = np.clip(np.asarray(reference["minor_af"])[ancestry_index], 0.0, 1.0)
        denominator = math.log1p(max_callable_an[ancestry])
        values[:, :, offset + 1] = np.clip(
            np.log1p(np.asarray(reference["callable_an"])[ancestry_index]) / denominator, 0.0, 1.0)
        values[:, :, offset + 2] = np.asarray(reference["observed_mask"])[ancestry_index]
    return np.ascontiguousarray(values.astype("<f4"))


def build_packed_shard(
    selected: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray], f0: Mapping[str, np.ndarray], marker_cm: np.ndarray,
    max_callable_an: Mapping[str, int], radius_cm: float, person_start: int,
    person_end_exclusive: int, marker_start: int, marker_end_exclusive: int,
) -> dict[str, np.ndarray]:
    """Build one exact packed shard; intended as the semantic equivalence oracle."""
    validate_inputs(selected, target, reference, f0, marker_cm)
    require(radius_cm in RADII, "radius is not frozen")
    sample_count = np.asarray(target["sample_key_sha256"]).size
    marker_count = np.asarray(f0["marker_pos"]).size
    require(0 <= person_start < person_end_exclusive <= sample_count and
            person_end_exclusive - person_start <= PERSON_BATCH, "person range differs")
    require(0 <= marker_start < marker_end_exclusive <= marker_count, "marker range differs")
    rare_cm = np.asarray(selected["cM"])
    intervals = contract.context_intervals(rare_cm.tolist(), np.asarray(marker_cm).tolist(), radius_cm)
    channels = base_channels(target, reference, max_callable_an)
    tokens: list[np.ndarray] = []
    locus_indexes: list[np.ndarray] = []
    row_ptr = [0]
    row_samples: list[int] = []
    row_markers: list[int] = []
    for sample_index in range(person_start, person_end_exclusive):
        for marker_index in range(marker_start, marker_end_exclusive):
            left, right = intervals[marker_index]
            indexes = np.arange(left, right, dtype="<u8")
            block = np.empty((right - left, 13), dtype="<f4")
            if right > left:
                block[:, :11] = channels[sample_index, left:right]
                block[:, 11] = np.clip(
                    (rare_cm[left:right] - marker_cm[marker_index]) / radius_cm, -1.0, 1.0,
                ).astype("<f4")
                delta = np.empty(right - left, dtype="<f8")
                delta[0] = 0.0
                if right - left > 1:
                    delta[1:] = np.diff(rare_cm[left:right])
                block[:, 12] = np.clip(delta / radius_cm, 0.0, 2.0).astype("<f4")
            tokens.append(block)
            locus_indexes.append(indexes)
            row_ptr.append(row_ptr[-1] + right - left)
            row_samples.append(sample_index - person_start)
            row_markers.append(marker_index - marker_start)
    valid_tokens = row_ptr[-1]
    require(valid_tokens <= TOKEN_BUDGET, "shard exceeds frozen token budget")
    marker_slice = slice(marker_start, marker_end_exclusive)
    sample_slice = slice(person_start, person_end_exclusive)
    return {
        "sample_key_sha256": np.ascontiguousarray(target["sample_key_sha256"][sample_slice]),
        "marker_chrom": np.ascontiguousarray(f0["marker_chrom"][marker_slice]),
        "marker_pos": np.ascontiguousarray(f0["marker_pos"][marker_slice]),
        "marker_ref": np.ascontiguousarray(f0["marker_ref"][marker_slice]),
        "marker_alt": np.ascontiguousarray(f0["marker_alt"][marker_slice]),
        "marker_cM": np.ascontiguousarray(marker_cm[marker_slice], dtype="<f8"),
        "radius_cM": np.asarray([radius_cm], dtype="<f4"),
        "rare_tokens": np.concatenate(tokens, axis=0) if tokens else np.empty((0, 13), dtype="<f4"),
        "rare_mask": np.ones(valid_tokens, dtype="|u1"),
        "rare_locus_index": (np.concatenate(locus_indexes) if locus_indexes
                             else np.empty(0, dtype="<u8")),
        "row_ptr": np.asarray(row_ptr, dtype="<u8"),
        "row_sample_index": np.asarray(row_samples, dtype="<u4"),
        "row_marker_index": np.asarray(row_markers, dtype="<u4"),
        "F0": np.ascontiguousarray(f0["F0"][sample_slice, :, marker_slice, :], dtype="<f4"),
    }


def estimate_packed_storage(assignments_by_radius: Mapping[float, int], sample_count: int,
                            copies: int = 1) -> dict[str, Any]:
    """Estimate the unavoidable token arrays and minimum shard count."""
    require(list(assignments_by_radius) == list(RADII), "radius order differs")
    require(type(sample_count) is int and sample_count > 0 and type(copies) is int and copies > 0,
            "sample or copy count differs")
    batch_sizes = [min(PERSON_BATCH, sample_count - start)
                   for start in range(0, sample_count, PERSON_BATCH)]
    rows = []
    for radius in RADII:
        assignments = assignments_by_radius[radius]
        require(type(assignments) is int and assignments >= 0, "assignment count differs")
        tokens_per_copy = assignments * sample_count
        minimum_shards_per_copy = sum(
            math.ceil(assignments * batch_size / TOKEN_BUDGET) for batch_size in batch_sizes
        )
        rows.append({"radius_cM": radius, "valid_tokens": tokens_per_copy * copies,
                     "token_array_bytes": tokens_per_copy * TOKEN_BYTES * copies,
                     "minimum_shards": minimum_shards_per_copy * copies})
    return {"token_bytes_each": TOKEN_BYTES, "rows": rows,
            "total_token_array_bytes": sum(row["token_array_bytes"] for row in rows),
            "minimum_shards": sum(row["minimum_shards"] for row in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate frozen M33 packed storage")
    parser.add_argument("--occupancy-json", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--copies", type=int, default=1,
                        help="independent root-by-rotation copies of the packed payload")
    args = parser.parse_args()
    payload = json.loads(args.occupancy_json.read_text(encoding="utf-8"))
    assignments = {float(row["radius_cm"]): int(row["total_context_locus_assignments"])
                   for row in payload["contexts"]}
    print(json.dumps(estimate_packed_storage(assignments, args.sample_count, args.copies),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
