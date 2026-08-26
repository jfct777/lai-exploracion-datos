#!/usr/bin/env python3
"""Generic, phase-invariant M34 rare-context materializer.

M34 keeps the factorized M33 representation semantics while removing two
experiment-specific assumptions: ancestry labels and TARGET sample count.  The
sample count is always derived from authenticated TARGET/F0 axes; it is never a
configuration value.  Expanded RD/RE shards are deterministic derivatives and
remain paired so their axes, masks, geometry and baseline probabilities cannot
drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m33_materialize as m33
from m33_safe_bridge_core import reopen_npz, write_deterministic_npz


PRODUCTIVE_MEMBERS = m33.PRODUCTIVE_MEMBERS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_ancestry_names(values: Sequence[str]) -> tuple[str, ...]:
    """Validate and preserve the exact ancestry-axis order."""
    require(not isinstance(values, (str, bytes)) and len(values) >= 2,
            "at least two ordered ancestry names are required")
    names = tuple(str(value) for value in values)
    require(all(name and name == name.strip() and name.isascii() and
                name.replace("_", "").isalnum() for name in names),
            "ancestry names must be nonempty ASCII identifiers")
    require(len(set(names)) == len(names), "ancestry names must be unique")
    return names


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "stage", "status", "ancestry_names",
        "sample_count_policy", "sample_shard_size", "radii_cM",
        "control_pair_policy",
    }
    require(set(contract) == required, "M34 contract members differ")
    require(contract["stage"] == "M34_GENERIC_RARE_CONTEXT_MATERIALIZER",
            "M34 contract stage differs")
    require(contract["sample_count_policy"] ==
            "derive_from_authenticated_TARGET_and_F0_axes_never_configure",
            "sample-count policy differs")
    require("sample_count" not in contract, "sample_count must not be configured")
    contract["ancestry_names"] = list(normalize_ancestry_names(contract["ancestry_names"]))
    require(type(contract["sample_shard_size"]) is int and
            contract["sample_shard_size"] > 0, "sample shard size differs")
    radii = tuple(float(value) for value in contract["radii_cM"])
    require(radii == m33.RADII, "M34 radii differ from the audited context geometry")
    require(contract["control_pair_policy"] ==
            "RD_and_RE_share_exact_axes_masks_geometry_and_F0",
            "RD/RE control policy differs")
    return contract


def _decode_axis(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode("ascii") if isinstance(value, bytes) else str(value)
                 for value in np.asarray(values).tolist())


def derive_sample_count(target: Mapping[str, np.ndarray],
                        f0: Mapping[str, np.ndarray]) -> int:
    """Derive sample count from both authenticated axes and reject disagreement."""
    target_axis = np.asarray(target["sample_key_sha256"])
    f0_axis = np.asarray(f0["sample_key_sha256"])
    require(target_axis.ndim == f0_axis.ndim == 1 and target_axis.size > 0,
            "sample axes must be nonempty and one-dimensional")
    require(np.array_equal(target_axis, f0_axis), "TARGET/F0 sample axes differ")
    require(np.unique(target_axis).size == target_axis.size, "duplicate TARGET samples")
    return int(target_axis.size)


def plan_sample_shards(sample_count: int, shard_size: int) -> list[dict[str, int]]:
    """Return a gap-free, non-overlapping sample-axis partition."""
    require(type(sample_count) is int and sample_count > 0, "sample count differs")
    require(type(shard_size) is int and shard_size > 0, "sample shard size differs")
    shards = [
        {"sample_start": start,
         "sample_end_exclusive": min(start + shard_size, sample_count)}
        for start in range(0, sample_count, shard_size)
    ]
    validate_sample_shards(shards, sample_count, shard_size)
    return shards


def validate_sample_shards(shards: Sequence[Mapping[str, int]], sample_count: int,
                           shard_size: int) -> None:
    cursor = 0
    for shard in shards:
        require(set(shard) == {"sample_start", "sample_end_exclusive"},
                "sample shard members differ")
        start, end = shard["sample_start"], shard["sample_end_exclusive"]
        require(type(start) is int and type(end) is int and start == cursor and
                start < end <= sample_count and end - start <= shard_size,
                "sample shards overlap, contain gaps or exceed bounds")
        cursor = end
    require(cursor == sample_count, "sample shards do not cover the sample axis")


def validate_role_disjointness(role_samples: Mapping[str, Sequence[str]]) -> None:
    """Fail closed if an individual appears in more than one biological role."""
    require(len(role_samples) >= 2, "at least two roles are required")
    seen: dict[str, str] = {}
    for role, values in role_samples.items():
        require(role and not isinstance(values, (str, bytes)), "role sample list differs")
        current = tuple(str(value) for value in values)
        require(len(set(current)) == len(current), f"duplicate sample inside role {role}")
        for sample in current:
            require(sample not in seen,
                    f"sample overlap between roles {seen.get(sample)} and {role}")
            seen[sample] = role


def validate_inputs(selected: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray],
                    reference: Mapping[str, np.ndarray], f0: Mapping[str, np.ndarray],
                    marker_cm: np.ndarray, ancestry_names: Sequence[str]) -> dict[str, int]:
    """Validate generic factor axes and return their derived dimensions."""
    names = normalize_ancestry_names(ancestry_names)
    require(set(selected) == PRODUCTIVE_MEMBERS["selected"], "selected members differ")
    require(set(target) == PRODUCTIVE_MEMBERS["target"], "target members differ")
    require(set(reference) == PRODUCTIVE_MEMBERS["reference"], "reference members differ")
    require(set(f0) == PRODUCTIVE_MEMBERS["f0"], "F0 members differ")

    sample_count = derive_sample_count(target, f0)
    locus_count = int(np.asarray(selected["locus_id"]).size)
    marker_count = int(np.asarray(f0["marker_pos"]).size)
    ancestry_count = len(names)
    require(locus_count > 0 and marker_count > 0, "locus or marker axis is empty")
    require(all(np.asarray(selected[name]).shape == (locus_count,)
                for name in PRODUCTIVE_MEMBERS["selected"]), "selected dimensions differ")
    require(np.asarray(target["locus_id"]).shape == (locus_count,) and
            np.asarray(target["minor_dosage"]).shape == (sample_count, locus_count) and
            np.asarray(target["observed_mask"]).shape == (sample_count, locus_count),
            "TARGET dimensions differ")
    require(np.asarray(reference["ancestry"]).shape == (ancestry_count,) and
            np.asarray(reference["locus_id"]).shape == (locus_count,) and
            all(np.asarray(reference[name]).shape == (ancestry_count, locus_count)
                for name in ("minor_ac", "callable_an", "minor_af",
                             "observed_mask", "no_support")),
            "reference dimensions differ")
    require(all(np.asarray(f0[name]).shape == (marker_count,)
                for name in ("marker_chrom", "marker_pos", "marker_ref", "marker_alt")) and
            np.asarray(f0["F0"]).shape ==
            (sample_count, 2, marker_count, ancestry_count), "F0 dimensions differ")
    require(np.asarray(marker_cm).shape == (marker_count,), "marker cM dimensions differ")

    require(_decode_axis(np.asarray(reference["ancestry"])) == names,
            "reference ancestry order differs")
    locus_axis = np.asarray(selected["locus_id"])
    require(np.unique(locus_axis).size == locus_count and
            np.array_equal(np.asarray(target["locus_id"]), locus_axis) and
            np.array_equal(np.asarray(reference["locus_id"]), locus_axis),
            "rare-locus axes differ")
    order = np.lexsort((locus_axis, np.asarray(selected["pos"]),
                        np.asarray(selected["cM"])))
    require(np.array_equal(order, np.arange(locus_count)),
            "rare loci are not ordered by cM/bp/locus_id")
    require(np.all(np.isfinite(np.asarray(marker_cm))) and
            np.all(np.asarray(marker_cm)[:-1] <= np.asarray(marker_cm)[1:]),
            "marker cM axis differs")

    dosage = np.asarray(target["minor_dosage"])
    target_mask = np.asarray(target["observed_mask"])
    require(np.all(np.isin(dosage, [0, 1, 2])) and
            np.all(np.isin(target_mask, [0, 1])) and
            np.all((target_mask == 1) | (dosage == 0)), "TARGET values or mask differ")
    ac, an, af = (np.asarray(reference[name]) for name in
                  ("minor_ac", "callable_an", "minor_af"))
    ref_mask = np.asarray(reference["observed_mask"])
    require(np.all(ac <= an) and np.all(np.isfinite(af)) and
            np.allclose(af, np.divide(ac, an, out=np.zeros_like(af), where=an > 0),
                        rtol=0, atol=1e-12), "reference AC/AN/AF differ")
    require(np.array_equal(ref_mask, (an > 0).astype(ref_mask.dtype)) and
            np.array_equal(np.asarray(reference["no_support"]),
                           ((an > 0) & (ac == 0)).astype(np.asarray(reference["no_support"]).dtype)),
            "reference masks differ")
    probabilities = np.asarray(f0["F0"])
    require(np.all(np.isfinite(probabilities)) and np.all(probabilities >= 0) and
            np.allclose(probabilities.sum(axis=3), 1.0, rtol=0, atol=5e-6),
            "F0 probability simplex differs")
    return {"sample_count": sample_count, "locus_count": locus_count,
            "marker_count": marker_count, "ancestry_count": ancestry_count,
            "channel_count": 4 + 3 * ancestry_count}


def derive_fit_max_callable(reference_by_fit: Sequence[Mapping[str, np.ndarray]],
                            ancestry_names: Sequence[str]) -> dict[str, int]:
    names = normalize_ancestry_names(ancestry_names)
    require(len(reference_by_fit) > 0, "FIT reference collection is empty")
    for reference in reference_by_fit:
        require(_decode_axis(np.asarray(reference["ancestry"])) == names,
                "FIT reference ancestry order differs")
        require(np.asarray(reference["callable_an"]).ndim == 2 and
                np.asarray(reference["callable_an"]).shape[0] == len(names),
                "FIT reference dimensions differ")
    maxima: dict[str, int] = {}
    for index, ancestry in enumerate(names):
        values = [int(np.asarray(reference["callable_an"])[index].max())
                  for reference in reference_by_fit]
        maxima[ancestry] = max(values)
    require(all(value > 0 for value in maxima.values()),
            "FIT callable maximum is nonpositive")
    return maxima


def _base_channels(target: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray],
                   ancestry_names: Sequence[str], max_callable_an: Mapping[str, int],
                   sample_start: int, sample_end_exclusive: int) -> np.ndarray:
    names = normalize_ancestry_names(ancestry_names)
    require(tuple(max_callable_an) == names and
            all(type(value) is int and value > 0 for value in max_callable_an.values()),
            "FIT callable normalization ancestry order differs")
    dosage = np.asarray(target["minor_dosage"][sample_start:sample_end_exclusive], dtype="<f8")
    observed = np.asarray(target["observed_mask"][sample_start:sample_end_exclusive], dtype="<f8")
    samples, loci = dosage.shape
    channel_count = 2 + 3 * len(names)
    values = np.empty((samples, loci, channel_count), dtype="<f4")
    values[:, :, 0] = np.clip(dosage / 2.0, 0.0, 1.0)
    values[:, :, 1] = observed
    for index, ancestry in enumerate(names):
        offset = 2 + 3 * index
        values[:, :, offset] = np.clip(reference["minor_af"][index], 0.0, 1.0)
        denominator = math.log1p(max_callable_an[ancestry])
        values[:, :, offset + 1] = np.clip(
            np.log1p(reference["callable_an"][index]) / denominator, 0.0, 1.0)
        values[:, :, offset + 2] = reference["observed_mask"][index]
    return np.ascontiguousarray(values)


def rare_value_channel_indices(ancestry_count: int) -> tuple[int, ...]:
    require(type(ancestry_count) is int and ancestry_count >= 2, "ancestry count differs")
    return (0, *(index for ancestry_index in range(ancestry_count)
                 for index in (2 + 3 * ancestry_index, 3 + 3 * ancestry_index)))


def build_paired_shard(selected: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray],
                       reference: Mapping[str, np.ndarray], f0: Mapping[str, np.ndarray],
                       marker_cm: np.ndarray, ancestry_names: Sequence[str],
                       max_callable_an: Mapping[str, int], radius_cm: float,
                       sample_start: int, sample_end_exclusive: int,
                       marker_start: int, marker_end_exclusive: int) -> tuple[dict[str, np.ndarray],
                                                                              dict[str, np.ndarray]]:
    """Materialize a paired rare-disabled and rare-enabled shard."""
    dimensions = validate_inputs(selected, target, reference, f0, marker_cm, ancestry_names)
    require(radius_cm in m33.RADII, "radius differs from the audited context geometry")
    require(0 <= sample_start < sample_end_exclusive <= dimensions["sample_count"],
            "sample shard bounds differ")
    require(0 <= marker_start < marker_end_exclusive <= dimensions["marker_count"],
            "marker shard bounds differ")
    names = normalize_ancestry_names(ancestry_names)
    channels = _base_channels(target, reference, names, max_callable_an,
                              sample_start, sample_end_exclusive)
    rare_cm = np.asarray(selected["cM"], dtype="<f8")
    marker_cm = np.asarray(marker_cm, dtype="<f8")
    intervals = m33.build_interval_table(rare_cm, marker_cm)
    radius_index = m33.RADII.index(radius_cm)
    starts = intervals["context_start"][radius_index]
    stops = intervals["context_stop"][radius_index]
    expected_tokens = (sample_end_exclusive - sample_start) * int(
        np.sum(stops[marker_start:marker_end_exclusive] -
               starts[marker_start:marker_end_exclusive], dtype="<u8")
    )
    require(expected_tokens <= m33.TOKEN_BUDGET,
            "paired shard exceeds the audited token budget")
    token_blocks: list[np.ndarray] = []
    locus_blocks: list[np.ndarray] = []
    row_ptr = [0]
    row_samples: list[int] = []
    row_markers: list[int] = []
    for sample_index in range(sample_start, sample_end_exclusive):
        for marker_index in range(marker_start, marker_end_exclusive):
            left, right = int(starts[marker_index]), int(stops[marker_index])
            block = np.empty((right - left, dimensions["channel_count"]), dtype="<f4")
            if right > left:
                block[:, :-2] = channels[sample_index - sample_start, left:right]
                block[:, -2] = np.clip(
                    (rare_cm[left:right] - marker_cm[marker_index]) / radius_cm, -1.0, 1.0)
                delta = np.empty(right - left, dtype="<f8")
                delta[0] = 0.0
                if right - left > 1:
                    delta[1:] = np.diff(rare_cm[left:right])
                block[:, -1] = np.clip(delta / radius_cm, 0.0, 2.0)
            token_blocks.append(block)
            locus_blocks.append(np.arange(left, right, dtype="<u8"))
            row_ptr.append(row_ptr[-1] + right - left)
            row_samples.append(sample_index - sample_start)
            row_markers.append(marker_index - marker_start)
    valid_tokens = row_ptr[-1]
    require(valid_tokens == expected_tokens, "paired-shard token accounting differs")
    marker_slice = slice(marker_start, marker_end_exclusive)
    sample_slice = slice(sample_start, sample_end_exclusive)
    re_view = {
        "sample_key_sha256": np.ascontiguousarray(target["sample_key_sha256"][sample_slice]),
        "marker_chrom": np.ascontiguousarray(f0["marker_chrom"][marker_slice]),
        "marker_pos": np.ascontiguousarray(f0["marker_pos"][marker_slice]),
        "marker_ref": np.ascontiguousarray(f0["marker_ref"][marker_slice]),
        "marker_alt": np.ascontiguousarray(f0["marker_alt"][marker_slice]),
        "marker_cM": np.ascontiguousarray(marker_cm[marker_slice], dtype="<f8"),
        "radius_cM": np.asarray([radius_cm], dtype="<f4"),
        "rare_tokens": (np.concatenate(token_blocks, axis=0) if token_blocks
                        else np.empty((0, dimensions["channel_count"]), dtype="<f4")),
        "rare_mask": np.ones(valid_tokens, dtype="|u1"),
        "rare_locus_index": (np.concatenate(locus_blocks) if locus_blocks
                             else np.empty(0, dtype="<u8")),
        "row_ptr": np.asarray(row_ptr, dtype="<u8"),
        "row_sample_index": np.asarray(row_samples, dtype="<u4"),
        "row_marker_index": np.asarray(row_markers, dtype="<u4"),
        "F0": np.ascontiguousarray(f0["F0"][sample_slice, :, marker_slice, :], dtype="<f4"),
    }
    rd_view = {name: np.ascontiguousarray(value.copy()) for name, value in re_view.items()}
    rd_view["rare_tokens"][:, rare_value_channel_indices(len(names))] = np.float32(0.0)
    validate_control_pair(rd_view, re_view, names)
    return rd_view, re_view


def validate_control_pair(rd_view: Mapping[str, np.ndarray],
                          re_view: Mapping[str, np.ndarray],
                          ancestry_names: Sequence[str]) -> None:
    """Prove that RD and RE differ only in authenticated rare-value channels."""
    names = normalize_ancestry_names(ancestry_names)
    require(set(rd_view) == set(re_view), "RD/RE members differ")
    rare_indices = rare_value_channel_indices(len(names))
    for name in rd_view:
        rd_value, re_value = np.asarray(rd_view[name]), np.asarray(re_view[name])
        require(rd_value.shape == re_value.shape and rd_value.dtype == re_value.dtype,
                f"RD/RE dimensions or dtype differ: {name}")
        if name != "rare_tokens":
            require(np.array_equal(rd_value, re_value), f"RD/RE axis or mask differs: {name}")
    rd_tokens, re_tokens = np.asarray(rd_view["rare_tokens"]), np.asarray(re_view["rare_tokens"])
    require(rd_tokens.ndim == 2 and rd_tokens.shape[1] == 4 + 3 * len(names),
            "RD/RE token dimensions differ")
    require(np.all(rd_tokens[:, rare_indices] == 0), "RD rare-value channels are nonzero")
    preserved = tuple(index for index in range(rd_tokens.shape[1]) if index not in rare_indices)
    require(np.array_equal(rd_tokens[:, preserved], re_tokens[:, preserved]),
            "RD/RE observed-mask or geometry channels differ")


def _load_npz(path: Path, expected_members: set[str]) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), f"invalid NPZ path: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected_members, f"NPZ members differ: {path.name}")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    require(not args.outdir.exists(), f"refusing to overwrite {args.outdir}")
    selected = _load_npz(args.selected, PRODUCTIVE_MEMBERS["selected"])
    target = _load_npz(args.target, PRODUCTIVE_MEMBERS["target"])
    reference = _load_npz(args.reference, PRODUCTIVE_MEMBERS["reference"])
    f0 = _load_npz(args.f0, PRODUCTIVE_MEMBERS["f0"])
    marker = _load_npz(args.marker_cm, {"marker_cM"})["marker_cM"]
    names = tuple(contract["ancestry_names"])
    dimensions = validate_inputs(selected, target, reference, f0, marker, names)
    maxima = derive_fit_max_callable([reference], names)
    shards = plan_sample_shards(dimensions["sample_count"], contract["sample_shard_size"])
    require(0 <= args.marker_start < args.marker_end_exclusive <= dimensions["marker_count"],
            "marker materialization bounds differ")
    args.outdir.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        rd, re = build_paired_shard(
            selected, target, reference, f0, marker, names, maxima, args.radius_cm,
            shard["sample_start"], shard["sample_end_exclusive"],
            args.marker_start, args.marker_end_exclusive,
        )
        row: dict[str, Any] = {"shard_index": shard_index, **shard}
        for arm, payload in (("RD", rd), ("RE", re)):
            path = args.outdir / f"sample-{shard_index:04d}.{arm}.npz"
            write_deterministic_npz(path, payload)
            reopen_npz(path, payload)
            row[f"{arm}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs.append(row)
    receipt = {
        "schema_version": "1.0.0", "stage": "M34_GENERIC_MATERIALIZATION",
        "status": "PASS", "ancestry_names": list(names), **dimensions,
        "sample_shard_size": contract["sample_shard_size"],
        "sample_shards": outputs, "radius_cM": args.radius_cm,
        "marker_start": args.marker_start,
        "marker_end_exclusive": args.marker_end_exclusive,
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
    }
    receipt["semantic_sha256"] = canonical_sha256(receipt)
    (args.outdir / "materialization.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--marker-cm", type=Path, required=True)
    parser.add_argument("--radius-cm", type=float, required=True, choices=m33.RADII)
    parser.add_argument("--marker-start", type=int, required=True)
    parser.add_argument("--marker-end-exclusive", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"],
                      "sample_count": result["sample_count"],
                      "shards": len(result["sample_shards"])}, sort_keys=True))
