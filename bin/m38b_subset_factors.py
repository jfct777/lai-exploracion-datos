#!/usr/bin/env python3
"""Freeze the M38B model factors to the preregistered LOO-stable loci.

The leave-one-NAM-unit-out (LOO) artifact is the only selector.  This stage
authenticates the frozen M34 S660 factor bundle, proves that its minor-allele
orientation agrees with the independently rebuilt LOO counts, and applies the
binary ``primary_mask`` without changing its thresholds.  TARGET truth and
model predictions are not inputs.

An empty mask is a valid fail-closed outcome: only a STOP receipt is written,
and no empty model factors are exposed for accidental downstream use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from m33_safe_bridge_core import (
    reopen_npz,
    write_deterministic_npz,
    write_exclusive_json,
)


SELECTED_MEMBERS = {"locus_id", "chrom", "pos", "ref", "alt", "cM"}
TARGET_MEMBERS = {"sample_key_sha256", "locus_id", "minor_dosage", "observed_mask"}
REFERENCE_MEMBERS = {
    "ancestry", "locus_id", "minor_ac", "callable_an", "minor_af",
    "observed_mask", "no_support",
}
LOO_REQUIRED = {
    "ancestry", "omitted_nam_unit", "locus_id", "chrom", "pos", "ref",
    "alt", "cM", "minor_code", "pooled_alt_ac", "pooled_callable_an",
    "full_minor_ac", "full_callable_an", "loo_minor_ac", "loo_callable_an",
    "loo_minor_af", "remaining_nam_carrier_units",
    "q_nam_min_all_priors_omissions", "remaining_nam_carrier_units_min",
    "primary_mask", "primary_locus_id", "primary_pos", "primary_ref",
    "primary_alt", "primary_cM", "primary_minor_code",
}
OUTPUT_NAMES = {
    "selected": "m38b_primary_selected_loci.npz",
    "target": "m38b_primary_target_rare_diploid.npz",
    "reference": "m38b_primary_reference_rare_summary.npz",
    "receipt": "m38b_primary_factor_subset.receipt.json",
}
ANCESTRIES = ("AFR", "EUR", "NAM")
HEX_DIGITS = frozenset("0123456789abcdef")


class M38BFactorSubsetError(ValueError):
    """Raised when an input or a frozen M38B factor invariant differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise M38BFactorSubsetError(message)


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"input is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate(path: Path, expected: str, label: str) -> str:
    require(
        isinstance(expected, str)
        and len(expected) == 64
        and set(expected).issubset(HEX_DIGITS),
        f"{label} expected SHA-256 is malformed",
    )
    observed = sha256_file(path)
    require(observed == expected, f"{label} SHA-256 mismatch")
    return observed


def decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        bytes(value).decode("ascii") if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in np.asarray(values).tolist()
    )


def load_npz(path: Path, expected: set[str], label: str, *, exact: bool) -> dict[str, np.ndarray]:
    require(path.is_file(), f"{label} is not a file")
    with np.load(path, allow_pickle=False) as archive:
        members = set(archive.files)
        require(members == expected if exact else expected.issubset(members),
                f"{label} NPZ members differ")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _validate_original_factors(
    selected: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    expected_loci: int,
) -> None:
    require(
        all(np.asarray(selected[name]).shape == (expected_loci,) for name in SELECTED_MEMBERS),
        "selected factor axes differ",
    )
    require(
        selected["locus_id"].dtype == np.dtype("<u8")
        and selected["chrom"].dtype == np.dtype("|u1")
        and selected["pos"].dtype == np.dtype("<i8")
        and selected["ref"].dtype == np.dtype("|S1")
        and selected["alt"].dtype == np.dtype("|S1")
        and selected["cM"].dtype == np.dtype("<f8"),
        "selected factor dtypes differ",
    )
    require(
        np.unique(selected["locus_id"]).size == expected_loci
        and np.all(selected["chrom"] == 22)
        and np.all(selected["pos"] > 0)
        and np.all(np.diff(selected["cM"]) >= 0)
        and np.isfinite(selected["cM"]).all(),
        "selected factor identity or order differs",
    )
    refs, alts = decode(selected["ref"]), decode(selected["alt"])
    require(
        all(ref in "ACGT" and alt in "ACGT" and ref != alt
            for ref, alt in zip(refs, alts, strict=True)),
        "selected factor allele axis differs",
    )

    sample_count = len(target["sample_key_sha256"])
    require(
        target["sample_key_sha256"].dtype == np.dtype("|S64")
        and target["locus_id"].dtype == np.dtype("<u8")
        and target["minor_dosage"].dtype == np.dtype("|i1")
        and target["observed_mask"].dtype == np.dtype("|u1")
        and target["locus_id"].shape == (expected_loci,)
        and target["minor_dosage"].shape == (sample_count, expected_loci)
        and target["observed_mask"].shape == (sample_count, expected_loci)
        and sample_count > 0
        and np.unique(target["sample_key_sha256"]).size == sample_count,
        "TARGET factor axes or dtypes differ",
    )
    require(
        np.all(np.isin(target["minor_dosage"], (0, 1, 2)))
        and np.all(np.isin(target["observed_mask"], (0, 1)))
        and np.all(target["minor_dosage"][target["observed_mask"] == 0] == 0),
        "TARGET dosage or missingness contract differs",
    )

    require(
        decode(reference["ancestry"]) == ANCESTRIES
        and reference["locus_id"].dtype == np.dtype("<u8")
        and reference["minor_ac"].dtype == np.dtype("<u2")
        and reference["callable_an"].dtype == np.dtype("<u2")
        and reference["minor_af"].dtype == np.dtype("<f8")
        and reference["observed_mask"].dtype == np.dtype("|u1")
        and reference["no_support"].dtype == np.dtype("|u1")
        and reference["locus_id"].shape == (expected_loci,)
        and all(reference[name].shape == (3, expected_loci) for name in (
            "minor_ac", "callable_an", "minor_af", "observed_mask", "no_support"
        )),
        "reference factor axes or dtypes differ",
    )
    ac = np.asarray(reference["minor_ac"], dtype=np.int64)
    an = np.asarray(reference["callable_an"], dtype=np.int64)
    require(
        np.all((0 <= ac) & (ac <= an))
        and np.all(an > 0)
        and np.allclose(reference["minor_af"], ac / an, rtol=0, atol=1e-12)
        and np.array_equal(reference["observed_mask"], (an > 0).astype(np.uint8))
        and np.array_equal(reference["no_support"], ((an > 0) & (ac == 0)).astype(np.uint8)),
        "reference frequency or callability contract differs",
    )
    require(
        np.array_equal(selected["locus_id"], target["locus_id"])
        and np.array_equal(selected["locus_id"], reference["locus_id"]),
        "selected/TARGET/reference locus_id axes differ",
    )


def _validate_loo(
    loo: Mapping[str, np.ndarray],
    selected: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    expected_loci: int,
) -> np.ndarray:
    require(
        decode(loo["ancestry"]) == ANCESTRIES
        and len(loo["omitted_nam_unit"]) == 4
        and len(set(decode(loo["omitted_nam_unit"]))) == 4,
        "LOO ancestry or omitted-unit axis differs",
    )
    for name in ("locus_id", "chrom", "pos", "ref", "alt", "cM"):
        require(
            np.asarray(loo[name]).shape == (expected_loci,)
            and np.array_equal(loo[name], selected[name]),
            f"LOO/original {name} axis differs",
        )
    mask = np.asarray(loo["primary_mask"])
    require(
        mask.shape == (expected_loci,)
        and mask.dtype == np.dtype("|u1")
        and np.all(np.isin(mask, (0, 1))),
        "LOO primary mask differs",
    )
    minor_code = np.asarray(loo["minor_code"])
    full_ac = np.asarray(loo["full_minor_ac"], dtype=np.int64)
    full_an = np.asarray(loo["full_callable_an"], dtype=np.int64)
    alt_ac = np.asarray(loo["pooled_alt_ac"], dtype=np.int64)
    pooled_an = np.asarray(loo["pooled_callable_an"], dtype=np.int64)
    require(
        minor_code.shape == alt_ac.shape == pooled_an.shape == (expected_loci,)
        and np.all(np.isin(minor_code, (0, 1)))
        and full_ac.shape == full_an.shape == (3, expected_loci)
        and np.array_equal(full_ac, reference["minor_ac"])
        and np.array_equal(full_an, reference["callable_an"])
        and np.array_equal(pooled_an, full_an.sum(axis=0)),
        "LOO/reference counts or minor orientation differ",
    )
    pooled_minor = full_ac.sum(axis=0)
    reconstructed_minor = np.where(minor_code == 1, alt_ac, pooled_an - alt_ac)
    require(
        np.array_equal(pooled_minor, reconstructed_minor)
        and np.all(2 * pooled_minor <= pooled_an),
        "LOO minor-code orientation does not reconstruct the minor allele",
    )
    selected_indices = np.flatnonzero(mask)
    for full_name, primary_name in (
        ("locus_id", "primary_locus_id"), ("pos", "primary_pos"),
        ("ref", "primary_ref"), ("alt", "primary_alt"),
        ("cM", "primary_cM"), ("minor_code", "primary_minor_code"),
    ):
        require(
            np.array_equal(np.asarray(loo[full_name])[selected_indices], loo[primary_name]),
            f"LOO stored {primary_name} differs from primary_mask",
        )
    return selected_indices


def _validate_loo_receipt(
    path: Path, loo_path: Path, loo_sha256: str, selected_sha256: str,
    expected_loci: int,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    status = receipt.get("status")
    require(
        receipt.get("stage") == "M38B_REF_TRAIN_LEAVE_ONE_NAM_UNIT_OUT_SUBSET"
        and status in {
            "PASS_PRIMARY_LOO_SUBSET_FROZEN",
            "PASS_ZERO_PRIMARY_LOO_SUBSET_NO_RELAXATION",
        },
        "LOO receipt stage or status differs",
    )
    scope = receipt.get("scope", {})
    selection = receipt.get("selection_contract", {})
    require(
        scope.get("frequency_role") == "REF_TRAIN_only"
        and scope.get("target_genotypes_read") is False
        and scope.get("local_ancestry_truth_read") is False
        and scope.get("predictions_read") is False
        and scope.get("scores_read") is False
        and scope.get("king_used") is False,
        "LOO receipt scope differs",
    )
    require(
        selection.get("beta_priors") == [0.5, 1.0]
        and selection.get("q_top_threshold") == 0.8
        and selection.get("minimum_remaining_NAM_carrier_units") == 2
        and selection.get("all_omissions_required") is True
        and selection.get("all_priors_required") is True
        and selection.get("post_outcome_relaxation_allowed") is False,
        "LOO selection contract was relaxed or differs",
    )
    require(
        receipt.get("counts", {}).get("S660_loci") == expected_loci
        and receipt.get("outputs", {}).get("loo_subset_npz_sha256") == loo_sha256
        and receipt.get("inputs", {}).get("selected_loci_sha256") == selected_sha256,
        "LOO receipt does not authenticate the frozen axes",
    )
    require(sha256_file(loo_path) == loo_sha256, "LOO artifact changed during validation")
    return receipt


def subset_factors(
    *,
    loo_subset: Path,
    loo_receipt: Path,
    selected: Path,
    target: Path,
    reference: Path,
    expected_loo_sha256: str,
    expected_loo_receipt_sha256: str,
    expected_selected_sha256: str,
    expected_target_sha256: str,
    expected_reference_sha256: str,
    outdir: Path,
    expected_loci: int = 660,
) -> dict[str, Any]:
    require(expected_loci > 0, "expected locus count must be positive")
    require(
        not outdir.exists() or (outdir.is_dir() and not any(outdir.iterdir())),
        "output directory must be absent or empty",
    )
    input_hashes = {
        "loo_subset": authenticate(loo_subset, expected_loo_sha256, "LOO subset"),
        "selected": authenticate(selected, expected_selected_sha256, "selected factor"),
        "target": authenticate(target, expected_target_sha256, "TARGET factor"),
        "reference": authenticate(reference, expected_reference_sha256, "reference factor"),
    }
    authenticated_loo_receipt = authenticate(
        loo_receipt, expected_loo_receipt_sha256, "LOO receipt",
    )
    selected_arrays = load_npz(selected, SELECTED_MEMBERS, "selected", exact=True)
    target_arrays = load_npz(target, TARGET_MEMBERS, "TARGET", exact=True)
    reference_arrays = load_npz(reference, REFERENCE_MEMBERS, "reference", exact=True)
    loo_arrays = load_npz(loo_subset, LOO_REQUIRED, "LOO", exact=False)
    _validate_original_factors(
        selected_arrays, target_arrays, reference_arrays, expected_loci,
    )
    indices = _validate_loo(
        loo_arrays, selected_arrays, reference_arrays, expected_loci,
    )
    loo_document = _validate_loo_receipt(
        loo_receipt, loo_subset, input_hashes["loo_subset"],
        input_hashes["selected"], expected_loci,
    )
    require(
        loo_document.get("counts", {}).get("primary_loci") == len(indices),
        "LOO receipt primary count differs from primary_mask",
    )

    outdir.mkdir(parents=True, exist_ok=True)
    receipt_path = outdir / OUTPUT_NAMES["receipt"]
    require(not receipt_path.exists(), "refusing to overwrite M38B factor receipt")
    base_receipt: dict[str, Any] = {
        "schema_version": "m38b_primary_factor_subset_receipt_v1",
        "stage": "M38B_APPLY_FROZEN_LOO_PRIMARY_MASK",
        "scope": {
            "chromosome": "22",
            "target_partition": "FIT_ONLY",
            "target_truth_read": False,
            "predictions_read": False,
            "scores_read": False,
            "king_used": False,
        },
        "selection": {
            "source": "authenticated_LOO_primary_mask_only",
            "thresholds_recomputed": False,
            "post_outcome_relaxation_allowed": False,
        },
        "counts": {
            "S660_loci": expected_loci,
            "primary_loci": int(len(indices)),
            "target_people": int(len(target_arrays["sample_key_sha256"])),
        },
        "orientation": {
            "minor_code_reconstructed_full_reference_counts": True,
            "reference_counts_exactly_equal_LOO_full_counts": True,
            "target_factor_authenticated_by_canonical_SHA256": True,
            "haplotype_phase_invented": False,
        },
        "inputs": {
            **{f"{name}_sha256": digest for name, digest in input_hashes.items()},
            "loo_receipt_sha256": authenticated_loo_receipt,
        },
    }
    if len(indices) == 0:
        require(
            loo_document["status"] == "PASS_ZERO_PRIMARY_LOO_SUBSET_NO_RELAXATION",
            "zero primary mask disagrees with LOO receipt status",
        )
        base_receipt.update({
            "decision": "STOP_MODEL_NO_PRIMARY_LOCI",
            "outputs": {},
            "empty_factor_npz_written": False,
        })
        write_exclusive_json(receipt_path, base_receipt)
        return base_receipt

    require(
        loo_document["status"] == "PASS_PRIMARY_LOO_SUBSET_FROZEN",
        "non-empty primary mask disagrees with LOO receipt status",
    )
    selected_output = {
        name: np.ascontiguousarray(selected_arrays[name][indices])
        for name in SELECTED_MEMBERS
    }
    target_output = {
        "sample_key_sha256": np.ascontiguousarray(target_arrays["sample_key_sha256"]),
        "locus_id": np.ascontiguousarray(target_arrays["locus_id"][indices]),
        "minor_dosage": np.ascontiguousarray(target_arrays["minor_dosage"][:, indices]),
        "observed_mask": np.ascontiguousarray(target_arrays["observed_mask"][:, indices]),
    }
    reference_output = {
        "ancestry": np.ascontiguousarray(reference_arrays["ancestry"]),
        "locus_id": np.ascontiguousarray(reference_arrays["locus_id"][indices]),
        **{
            name: np.ascontiguousarray(reference_arrays[name][:, indices])
            for name in ("minor_ac", "callable_an", "minor_af", "observed_mask", "no_support")
        },
    }
    require(
        np.array_equal(selected_output["locus_id"], loo_arrays["primary_locus_id"])
        and np.array_equal(target_output["locus_id"], loo_arrays["primary_locus_id"])
        and np.array_equal(reference_output["locus_id"], loo_arrays["primary_locus_id"]),
        "primary output locus axes differ",
    )
    paths = {
        "selected": outdir / OUTPUT_NAMES["selected"],
        "target": outdir / OUTPUT_NAMES["target"],
        "reference": outdir / OUTPUT_NAMES["reference"],
    }
    payloads = {
        "selected": selected_output,
        "target": target_output,
        "reference": reference_output,
    }
    for name in ("selected", "target", "reference"):
        write_deterministic_npz(paths[name], payloads[name])
        reopen_npz(paths[name], payloads[name])
    base_receipt.update({
        "decision": "PASS_PRIMARY_FACTORS_FROZEN_FOR_MODEL",
        "empty_factor_npz_written": False,
        "outputs": {
            paths[name].name: {
                "sha256": sha256_file(paths[name]),
                "bytes": paths[name].stat().st_size,
            }
            for name in ("selected", "target", "reference")
        },
    })
    write_exclusive_json(receipt_path, base_receipt)
    return base_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loo-subset", type=Path, required=True)
    parser.add_argument("--loo-receipt", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-loo-sha256", required=True)
    parser.add_argument("--expected-loo-receipt-sha256", required=True)
    parser.add_argument("--expected-selected-sha256", required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--expected-loci", type=int, default=660)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = subset_factors(
        loo_subset=args.loo_subset,
        loo_receipt=args.loo_receipt,
        selected=args.selected,
        target=args.target,
        reference=args.reference,
        expected_loo_sha256=args.expected_loo_sha256,
        expected_loo_receipt_sha256=args.expected_loo_receipt_sha256,
        expected_selected_sha256=args.expected_selected_sha256,
        expected_target_sha256=args.expected_target_sha256,
        expected_reference_sha256=args.expected_reference_sha256,
        outdir=args.outdir,
        expected_loci=args.expected_loci,
    )
    print(json.dumps({
        "decision": receipt["decision"],
        "primary_loci": receipt["counts"]["primary_loci"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
