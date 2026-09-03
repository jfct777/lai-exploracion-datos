#!/usr/bin/env python3
"""Stratify the 660 M34 pooled-rare loci using REF_TRAIN only.

This stage is deliberately descriptive.  It authenticates and reconciles the
M34 factor files with the independent per-locus audit, then estimates how
strongly each locus is enriched in AFR, EUR or NAM and whether it remains rare
within all three reference ancestries.  It never reads TARGET genotypes, local
ancestry truth, baseline predictions or model scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz, write_exclusive_json


ANCESTRIES = ("AFR", "EUR", "NAM")
SELECTED_MEMBERS = {"alt", "cM", "chrom", "locus_id", "pos", "ref"}
REFERENCE_MEMBERS = {
    "ancestry", "callable_an", "locus_id", "minor_ac", "minor_af",
    "no_support", "observed_mask",
}
AUDIT_REQUIRED = {
    "chrom", "position", "ref", "alt", "pooled_minor_ac",
    "pooled_callable_an", "pooled_maf",
    *(f"{ancestry}_{suffix}" for ancestry in ANCESTRIES for suffix in
      ("minor_ac", "callable_an", "minor_af", "carrier_people",
       "carrier_populations", "carrier_units", "max_unit_carrier_share",
       "unit_hhi")),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_axis(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("ascii") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).tolist()
    )


def parse_float_list(value: str, *, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError:
        raise ValueError(f"{label} must be a comma-separated float list") from None
    require(result and all(math.isfinite(item) for item in result),
            f"{label} must contain finite values")
    require(tuple(sorted(set(result))) == result, f"{label} must be unique and sorted")
    return result


def parse_expected_an(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in value.split(","):
        try:
            ancestry, count = item.split("=", 1)
            result[ancestry] = int(count)
        except ValueError:
            raise ValueError("expected AN must use AFR=n,EUR=n,NAM=n") from None
    require(tuple(result) == ANCESTRIES and all(count > 0 for count in result.values()),
            "expected AN must contain positive AFR, EUR and NAM values in that order")
    return result


def validate_exact_locus_partition(
    full_loci: np.ndarray, common_loci: np.ndarray, selected_rare_loci: np.ndarray,
) -> dict[str, int]:
    """Require F_full = F_common disjoint-union selected_rare using real axes.

    M38A has no common-only baseline input and therefore does not call this
    function.  M38B must call it on its authenticated locus axes before any
    model is allowed to run.
    """
    full = np.asarray(full_loci).reshape(-1)
    common = np.asarray(common_loci).reshape(-1)
    rare = np.asarray(selected_rare_loci).reshape(-1)
    require(np.unique(full).size == full.size and np.unique(common).size == common.size and
            np.unique(rare).size == rare.size, "locus partitions contain duplicate identifiers")
    full_set, common_set, rare_set = set(full.tolist()), set(common.tolist()), set(rare.tolist())
    require(not (common_set & rare_set),
            "selected_rare intersects F_common; incremental comparison is contaminated")
    require(common_set | rare_set == full_set,
            "F_full is not the exact union of F_common and selected_rare")
    return {"full_loci": len(full_set), "common_loci": len(common_set),
            "selected_rare_loci": len(rare_set), "overlap_loci": 0}


def load_npz(path: Path, expected_members: set[str]) -> dict[str, np.ndarray]:
    # Nextflow stages immutable inputs as symbolic links inside the task
    # directory.  Content authentication is enforced separately by SHA-256,
    # so rejecting those links would reject the normal production interface.
    require(path.is_file(), f"invalid input: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected_members,
                f"{path.name} members differ")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def authenticate(path: Path, expected_sha256: str) -> str:
    require(len(expected_sha256) == 64, f"missing expected SHA-256 for {path.name}")
    observed = sha256_file(path)
    require(observed == expected_sha256, f"SHA-256 differs for {path.name}")
    return observed


def load_audit(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and AUDIT_REQUIRED.issubset(reader.fieldnames),
                "audit TSV fields differ")
        rows = [dict(row) for row in reader]
        return rows, tuple(reader.fieldnames)


def validate_inputs(
    selected: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray],
    audit_rows: Sequence[Mapping[str, str]], audit_summary: Mapping[str, Any],
    expected_loci: int, expected_an: Mapping[str, int], audit_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(expected_loci == 660, "M38A is frozen to the 660 M34 chr22 loci")
    require(len(selected["locus_id"]) == len(audit_rows) == expected_loci,
            "locus count differs from the frozen M34 universe")
    require(np.array_equal(selected["locus_id"], reference["locus_id"]),
            "selected/reference locus axes differ")
    require(decode_axis(reference["ancestry"]) == ANCESTRIES,
            "reference ancestry axis must be AFR, EUR, NAM")
    require(np.unique(selected["locus_id"]).size == expected_loci,
            "locus identifiers are duplicated")
    require(np.all(np.isfinite(selected["cM"])) and np.all(np.diff(selected["cM"]) >= 0),
            "genetic positions must be finite and ordered")

    ac = np.asarray(reference["minor_ac"], dtype=np.int64)
    an = np.asarray(reference["callable_an"], dtype=np.int64)
    af = np.asarray(reference["minor_af"], dtype=np.float64)
    require(ac.shape == an.shape == af.shape == (3, expected_loci),
            "reference count dimensions differ")
    require(np.all((0 <= ac) & (ac <= an)) and np.all(an > 0),
            "reference AC/AN values differ")
    require(np.allclose(af, ac / an, rtol=0, atol=1e-12),
            "reference AF is inconsistent with AC/AN")
    require(np.array_equal(reference["observed_mask"], (an > 0).astype(np.uint8)) and
            np.array_equal(reference["no_support"], ((an > 0) & (ac == 0)).astype(np.uint8)),
            "reference support masks differ")
    for index, ancestry in enumerate(ANCESTRIES):
        require(np.all(an[index] == expected_an[ancestry]),
                f"{ancestry} callable AN differs from the authenticated audit")

    for index, row in enumerate(audit_rows):
        ref = selected["ref"][index].decode("ascii")
        alt = selected["alt"][index].decode("ascii")
        key = (int(selected["chrom"][index]), int(selected["pos"][index]), ref, alt)
        audit_key = (int(row["chrom"]), int(row["position"]), row["ref"], row["alt"])
        require(key == audit_key, f"audit/NPZ locus axis differs at row {index + 2}")
        for ancestry_index, ancestry in enumerate(ANCESTRIES):
            require(int(row[f"{ancestry}_minor_ac"]) == int(ac[ancestry_index, index]) and
                    int(row[f"{ancestry}_callable_an"]) == int(an[ancestry_index, index]) and
                    math.isclose(float(row[f"{ancestry}_minor_af"]),
                                 float(af[ancestry_index, index]), rel_tol=0, abs_tol=5e-12),
                    f"audit/reference counts differ at row {index + 2}")
        pooled_ac = int(ac[:, index].sum())
        pooled_an = int(an[:, index].sum())
        require(int(row["pooled_minor_ac"]) == pooled_ac and
                int(row["pooled_callable_an"]) == pooled_an and
                math.isclose(float(row["pooled_maf"]), pooled_ac / pooled_an,
                             rel_tol=0, abs_tol=5e-12),
                f"audit pooled counts differ at row {index + 2}")

    require(audit_summary.get("stage") == "M34_RARE_LOCUS_DISTRIBUTION_AUDIT" and
            audit_summary.get("status") == "PASS_DESCRIPTIVE_AUDIT_NO_MODEL_SELECTION",
            "M34 audit status differs")
    scope = audit_summary.get("scope", {})
    require(scope.get("frequency_population") == "REF_TRAIN_only" and
            scope.get("target_mosaics_read") is False and
            scope.get("local_ancestry_truth_read") is False and
            scope.get("predictions_read") is False and
            scope.get("king_used") is False,
            "M34 audit scope is not REF_TRAIN-only or no-KING")
    require(audit_summary.get("selection", {}).get("selected_loci") == expected_loci and
            audit_summary.get("outputs", {}).get("per_locus_tsv_sha256") == audit_sha256,
            "M34 audit summary does not authenticate the locus table")
    return ac, an, af


def posterior_probabilities_mc(
    ac: np.ndarray, an: np.ndarray, locus_id: np.ndarray, *, prior: float,
    rare_af: float, draws: int, seed: int,
) -> dict[str, np.ndarray]:
    """Estimate posterior probabilities with deterministic per-locus draws."""
    require(prior > 0 and math.isfinite(prior), "Beta prior must be positive")
    require(draws >= 4096 and draws % 2 == 0, "q_top draws must be even and at least 4096")
    q_top = np.empty(ac.shape, dtype=np.float64)
    q_top_half_difference = np.empty(ac.shape, dtype=np.float64)
    q_top_standard_error = np.empty(ac.shape, dtype=np.float64)
    q_below = np.empty(ac.shape, dtype=np.float64)
    q_below_half_difference = np.empty(ac.shape, dtype=np.float64)
    q_below_standard_error = np.empty(ac.shape, dtype=np.float64)
    q_all_rare = np.empty(ac.shape[1], dtype=np.float64)
    q_all_rare_half_difference = np.empty(ac.shape[1], dtype=np.float64)
    q_all_rare_standard_error = np.empty(ac.shape[1], dtype=np.float64)
    prior_key = int(round(prior * 1_000_000))
    for locus in range(ac.shape[1]):
        identity = int(locus_id[locus])
        sequence = np.random.SeedSequence([
            int(seed), prior_key, identity & 0xFFFFFFFF, identity >> 32,
        ])
        rng = np.random.default_rng(sequence)
        values = rng.beta(
            ac[:, locus] + prior,
            an[:, locus] - ac[:, locus] + prior,
            size=(draws, len(ANCESTRIES)),
        )
        winners = np.argmax(values, axis=1)
        counts = np.bincount(winners, minlength=len(ANCESTRIES))
        q = counts / draws
        first = np.bincount(winners[:draws // 2], minlength=len(ANCESTRIES)) / (draws // 2)
        second = np.bincount(winners[draws // 2:], minlength=len(ANCESTRIES)) / (draws // 2)
        below = values < rare_af
        below_q = below.mean(axis=0)
        below_first = below[:draws // 2].mean(axis=0)
        below_second = below[draws // 2:].mean(axis=0)
        all_rare = np.all(below, axis=1)
        all_rare_q = float(all_rare.mean())
        q_top[:, locus] = q
        q_top_half_difference[:, locus] = np.abs(first - second)
        q_top_standard_error[:, locus] = np.sqrt(q * (1.0 - q) / draws)
        q_below[:, locus] = below_q
        q_below_half_difference[:, locus] = np.abs(below_first - below_second)
        q_below_standard_error[:, locus] = np.sqrt(below_q * (1.0 - below_q) / draws)
        q_all_rare[locus] = all_rare_q
        q_all_rare_half_difference[locus] = abs(
            float(all_rare[:draws // 2].mean()) - float(all_rare[draws // 2:].mean()))
        q_all_rare_standard_error[locus] = math.sqrt(
            all_rare_q * (1.0 - all_rare_q) / draws)
    require(np.allclose(q_top.sum(axis=0), 1.0, rtol=0, atol=1e-15),
            "q_top Monte Carlo probabilities do not sum to one")
    return {
        "q_top": q_top,
        "q_top_mc_se": q_top_standard_error,
        "q_top_mc_half_difference": q_top_half_difference,
        "q_af_below_rare_cutoff": q_below,
        "q_af_below_rare_cutoff_mc_se": q_below_standard_error,
        "q_af_below_rare_cutoff_mc_half_difference": q_below_half_difference,
        "q_all_ancestries_rare": q_all_rare,
        "q_all_ancestries_rare_mc_se": q_all_rare_standard_error,
        "q_all_ancestries_rare_mc_half_difference": q_all_rare_half_difference,
    }


def posterior_bundle(
    ac: np.ndarray, an: np.ndarray, locus_id: np.ndarray, *, prior: float,
    rare_af: float, draws: int, seed: int,
) -> dict[str, np.ndarray]:
    alpha = ac.astype(np.float64) + prior
    beta = (an - ac).astype(np.float64) + prior
    result = posterior_probabilities_mc(
        ac, an, locus_id, prior=prior, rare_af=rare_af, draws=draws, seed=seed)
    result["posterior_mean"] = alpha / (alpha + beta)
    return result


def threshold_label(value: float) -> str:
    return format(value, ".2f").replace(".", "P")


def prior_label(value: float) -> str:
    return format(value, ".1f").replace(".", "P")


def build_masks(
    posteriors: Mapping[float, Mapping[str, np.ndarray]], af: np.ndarray,
    carrier_units: np.ndarray, q_top_thresholds: Sequence[float],
    q_rare_thresholds: Sequence[float], unit_thresholds: Sequence[int],
) -> tuple[tuple[str, ...], np.ndarray]:
    names: list[str] = []
    values: list[np.ndarray] = []

    def add(name: str, mask: np.ndarray) -> None:
        names.append(name)
        values.append(np.asarray(mask, dtype=np.uint8))

    observed_all_rare = np.all(af < 0.01, axis=0)
    observed_nam_anchor = (af[2] >= 0.05) & (af[0] < 0.01) & (af[1] < 0.01)
    add("ANCHOR_OBS_ALL_AF_LT_0P01", observed_all_rare)
    add("ANCHOR_OBS_NAM_AF_GE_0P05_AFR_EUR_LT_0P01", observed_nam_anchor)
    for units in unit_thresholds:
        add(f"ANCHOR_OBS_NAM_ENRICHED_UNITS_GE_{units}",
            observed_nam_anchor & (carrier_units[2] >= units))

    total_units = carrier_units.sum(axis=0)
    for prior, bundle in posteriors.items():
        suffix = f"PRIOR_{prior_label(prior)}"
        q_top = bundle["q_top"]
        q_rare = bundle["q_all_ancestries_rare"]
        for ancestry_index, ancestry in enumerate(ANCESTRIES):
            for threshold in q_top_thresholds:
                base = q_top[ancestry_index] >= threshold
                add(f"ENRICHED_{ancestry}_QTOP_GE_{threshold_label(threshold)}__{suffix}", base)
                for units in unit_thresholds:
                    add(
                        f"ENRICHED_{ancestry}_QTOP_GE_{threshold_label(threshold)}_UNITS_GE_{units}__{suffix}",
                        base & (carrier_units[ancestry_index] >= units),
                    )
        for threshold in q_rare_thresholds:
            base = q_rare >= threshold
            add(f"ALL_ANCESTRY_RARE_Q_GE_{threshold_label(threshold)}__{suffix}", base)
            for units in unit_thresholds:
                add(
                    f"ALL_ANCESTRY_RARE_Q_GE_{threshold_label(threshold)}_UNITS_GE_{units}__{suffix}",
                    base & (total_units >= units),
                )
    return tuple(names), np.vstack(values).astype(np.uint8)


def nam_status(q_nam_below: np.ndarray) -> np.ndarray:
    result = np.full(q_nam_below.shape, "NAM_UNRESOLVED", dtype="U36")
    result[q_nam_below >= 0.95] = "NAM_RARE_SUPPORTED_AT_Q0P95"
    result[q_nam_below <= 0.05] = "NAM_NOT_RARE_SUPPORTED_AT_Q0P95"
    return result


def write_exclusive_text(path: Path, text: str) -> None:
    require(not path.exists(), "output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        os.chmod(path, 0o400)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    outputs = (args.output_tsv, args.output_npz, args.output_summary, args.output_receipt)
    require(len(set(outputs)) == len(outputs) and not any(path.exists() for path in outputs),
            "refusing to overwrite or alias M38A outputs")
    hashes = {
        "selected": authenticate(args.selected, args.selected_sha256),
        "reference": authenticate(args.reference, args.reference_sha256),
        "audit_tsv": authenticate(args.audit_tsv, args.audit_tsv_sha256),
        "audit_summary": authenticate(args.audit_summary, args.audit_summary_sha256),
    }
    selected = load_npz(args.selected, SELECTED_MEMBERS)
    reference = load_npz(args.reference, REFERENCE_MEMBERS)
    audit_rows, _ = load_audit(args.audit_tsv)
    audit_summary = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    expected_an = parse_expected_an(args.expected_ancestry_an)
    ac, an, af = validate_inputs(
        selected, reference, audit_rows, audit_summary, args.expected_loci,
        expected_an, hashes["audit_tsv"],
    )
    priors = parse_float_list(args.beta_priors, label="Beta priors")
    require(priors == (0.5, 1.0), "M38A priors must be 0.5 and 1.0")
    q_top_thresholds = parse_float_list(args.q_top_thresholds, label="q_top thresholds")
    q_rare_thresholds = parse_float_list(args.q_rare_thresholds, label="q_rare thresholds")
    require(q_top_thresholds == (0.8, 0.9, 0.95) and
            q_rare_thresholds == (0.5, 0.8, 0.95), "frozen probability sweep differs")
    unit_thresholds = tuple(int(item) for item in args.unit_thresholds.split(","))
    require(unit_thresholds == (2, 3), "frozen unit-support sweep differs")
    require(0 < args.rare_af_cutoff < 0.5, "rare-AF cutoff differs")
    require(args.f0_contains_selected_rare_loci,
            "the known F0 locus-overlap limitation must be acknowledged contractually")

    posterior = {
        prior: posterior_bundle(
            ac, an, selected["locus_id"], prior=prior,
            rare_af=args.rare_af_cutoff, draws=args.q_top_draws, seed=args.seed,
        )
        for prior in priors
    }
    carrier_units = np.asarray([
        [int(row[f"{ancestry}_carrier_units"]) for row in audit_rows]
        for ancestry in ANCESTRIES
    ], dtype=np.uint16)
    carrier_people = np.asarray([
        [int(row[f"{ancestry}_carrier_people"]) for row in audit_rows]
        for ancestry in ANCESTRIES
    ], dtype=np.uint16)
    carrier_populations = np.asarray([
        [int(row[f"{ancestry}_carrier_populations"]) for row in audit_rows]
        for ancestry in ANCESTRIES
    ], dtype=np.uint16)
    max_unit_carrier_share = np.asarray([
        [float(row[f"{ancestry}_max_unit_carrier_share"]) for row in audit_rows]
        for ancestry in ANCESTRIES
    ], dtype=np.float64)
    unit_hhi = np.asarray([
        [float(row[f"{ancestry}_unit_hhi"]) for row in audit_rows]
        for ancestry in ANCESTRIES
    ], dtype=np.float64)
    require(np.all((0 <= max_unit_carrier_share) & (max_unit_carrier_share <= 1)) and
            np.all((0 <= unit_hhi) & (unit_hhi <= 1)),
            "unit-concentration metrics lie outside [0,1]")
    mask_names, masks = build_masks(
        posterior, af, carrier_units, q_top_thresholds,
        q_rare_thresholds, unit_thresholds,
    )
    nam_statuses = {prior: nam_status(bundle["q_af_below_rare_cutoff"][2])
                    for prior, bundle in posterior.items()}

    base_fields = [
        "locus_id", "chrom", "position", "ref", "alt", "cM",
        *(f"{ancestry}_{suffix}" for ancestry in ANCESTRIES for suffix in
          ("minor_ac", "callable_an", "minor_af", "carrier_people",
           "carrier_populations", "carrier_units", "max_unit_carrier_share",
           "unit_hhi")),
    ]
    posterior_fields: list[str] = []
    for prior in priors:
        label = prior_label(prior)
        for ancestry in ANCESTRIES:
            posterior_fields.extend((
                f"{ancestry}_posterior_mean_prior_{label}",
                f"{ancestry}_q_top_prior_{label}",
                f"{ancestry}_q_af_lt_0P01_prior_{label}",
            ))
        posterior_fields.extend((f"q_all_af_lt_0P01_prior_{label}",
                                 f"NAM_status_prior_{label}"))
    all_fields = base_fields + posterior_fields + list(mask_names)
    lines = ["\t".join(all_fields)]
    for locus in range(args.expected_loci):
        row: dict[str, Any] = {
            "locus_id": int(selected["locus_id"][locus]),
            "chrom": int(selected["chrom"][locus]),
            "position": int(selected["pos"][locus]),
            "ref": selected["ref"][locus].decode("ascii"),
            "alt": selected["alt"][locus].decode("ascii"),
            "cM": format(float(selected["cM"][locus]), ".12g"),
        }
        for ancestry_index, ancestry in enumerate(ANCESTRIES):
            row[f"{ancestry}_minor_ac"] = int(ac[ancestry_index, locus])
            row[f"{ancestry}_callable_an"] = int(an[ancestry_index, locus])
            row[f"{ancestry}_minor_af"] = format(float(af[ancestry_index, locus]), ".12g")
            row[f"{ancestry}_carrier_people"] = int(carrier_people[ancestry_index, locus])
            row[f"{ancestry}_carrier_populations"] = int(carrier_populations[ancestry_index, locus])
            row[f"{ancestry}_carrier_units"] = int(carrier_units[ancestry_index, locus])
            row[f"{ancestry}_max_unit_carrier_share"] = format(
                float(max_unit_carrier_share[ancestry_index, locus]), ".12g")
            row[f"{ancestry}_unit_hhi"] = format(float(unit_hhi[ancestry_index, locus]), ".12g")
        for prior, bundle in posterior.items():
            label = prior_label(prior)
            for ancestry_index, ancestry in enumerate(ANCESTRIES):
                row[f"{ancestry}_posterior_mean_prior_{label}"] = format(
                    float(bundle["posterior_mean"][ancestry_index, locus]), ".12g")
                row[f"{ancestry}_q_top_prior_{label}"] = format(
                    float(bundle["q_top"][ancestry_index, locus]), ".12g")
                row[f"{ancestry}_q_af_lt_0P01_prior_{label}"] = format(
                    float(bundle["q_af_below_rare_cutoff"][ancestry_index, locus]), ".12g")
            row[f"q_all_af_lt_0P01_prior_{label}"] = format(
                float(bundle["q_all_ancestries_rare"][locus]), ".12g")
            row[f"NAM_status_prior_{label}"] = str(nam_statuses[prior][locus])
        for mask_index, name in enumerate(mask_names):
            row[name] = int(masks[mask_index, locus])
        lines.append("\t".join(str(row[field]) for field in all_fields))
    write_exclusive_text(args.output_tsv, "\n".join(lines) + "\n")

    npz_payload: dict[str, np.ndarray] = {
        "ancestry": np.asarray(ANCESTRIES),
        "locus_id": np.ascontiguousarray(selected["locus_id"]),
        "mask_name": np.asarray(mask_names),
        "mask": masks,
        "carrier_units": carrier_units,
        "carrier_people": carrier_people,
        "carrier_populations": carrier_populations,
        "max_unit_carrier_share": max_unit_carrier_share,
        "unit_hhi": unit_hhi,
    }
    for prior, bundle in posterior.items():
        label = prior_label(prior).lower()
        for name, values in bundle.items():
            npz_payload[f"{name}_prior_{label}"] = np.asarray(values)
        npz_payload[f"nam_status_prior_{label}"] = nam_statuses[prior]
    write_deterministic_npz(args.output_npz, npz_payload)

    mask_counts = {name: int(masks[index].sum()) for index, name in enumerate(mask_names)}
    concentration = {}
    for index, ancestry in enumerate(ANCESTRIES):
        carried = carrier_people[index] > 0
        shares = max_unit_carrier_share[index, carried]
        hhi = unit_hhi[index, carried]
        concentration[ancestry] = {
            "loci_with_carriers": int(carried.sum()),
            "median_max_unit_carrier_share": float(np.median(shares)) if shares.size else None,
            "p95_max_unit_carrier_share": float(np.quantile(shares, 0.95)) if shares.size else None,
            "loci_max_unit_carrier_share_ge_0.5": int(np.sum(shares >= 0.5)),
            "median_unit_hhi": float(np.median(hhi)) if hhi.size else None,
            "p95_unit_hhi": float(np.quantile(hhi, 0.95)) if hhi.size else None,
        }
    mc = {
        f"prior_{prior_label(prior)}": {
            "maximum_standard_error": float(bundle["q_top_mc_se"].max()),
            "maximum_half_sample_difference": float(bundle["q_top_mc_half_difference"].max()),
            "maximum_q_rare_standard_error": float(
                bundle["q_all_ancestries_rare_mc_se"].max()),
            "maximum_q_rare_half_sample_difference": float(
                bundle["q_all_ancestries_rare_mc_half_difference"].max()),
            "q_top_loci_within_2se_of_threshold": {
                threshold_label(threshold): int(np.sum(np.any(
                    np.abs(bundle["q_top"] - threshold) <= 2 * bundle["q_top_mc_se"],
                    axis=0)))
                for threshold in q_top_thresholds
            },
            "q_rare_loci_within_2se_of_threshold": {
                threshold_label(threshold): int(np.sum(
                    np.abs(bundle["q_all_ancestries_rare"] - threshold) <=
                    2 * bundle["q_all_ancestries_rare_mc_se"]))
                for threshold in q_rare_thresholds
            },
            "nam_status_counts": {
                status: int(np.sum(nam_statuses[prior] == status))
                for status in sorted(set(nam_statuses[prior].tolist()))
            },
        }
        for prior, bundle in posterior.items()
    }
    summary = {
        "schema_version": "1.0.0",
        "stage": "M38_RARE_LOCUS_STRATIFICATION",
        "status": "PASS_REF_TRAIN_ONLY_DESCRIPTIVE_STRATIFICATION",
        "scope": {
            "chromosome": 22,
            "loci": args.expected_loci,
            "frequency_population": "REF_TRAIN_only",
            "locus_universe": "pooled_rare_in_REF_TRAIN_MAC_ge_2_MAF_lt_0.01",
            "target_read": False,
            "local_ancestry_truth_read": False,
            "F0_predictions_read": False,
            "model_scores_read": False,
            "king_used": False,
        },
        "contractual_assertions": {
            "M37_F0_CONTAINS_SELECTED_RARE_LOCI": True,
            "source": args.f0_overlap_assertion_source,
        },
        "incremental_value_estimated": False,
        "M38B_required_locus_partition": {
            "identity": "F_full = F_common disjoint_union selected_rare",
            "validation": "must_use_authenticated_real_locus_axes",
            "overlap_allowed": False,
        },
        "posterior": {
            "model": "independent_Beta_binomial_by_ancestry_and_locus",
            "priors": list(priors),
            "principal_prior": 0.5,
            "sensitivity_prior": 1.0,
            "rare_af_cutoff": args.rare_af_cutoff,
            "q_top_definition": "posterior_probability_that_ancestry_AF_is_largest",
            "q_rare_definition": "posterior_probability_all_three_ancestry_AF_are_below_cutoff",
            "q_top_and_q_rare_method": "shared_deterministic_per_locus_Monte_Carlo",
            "draws": args.q_top_draws,
            "seed": args.seed,
            "maximum_theoretical_standard_error": 0.5 / math.sqrt(args.q_top_draws),
            "diagnostics": mc,
        },
        "sweeps": {
            "q_top_thresholds": list(q_top_thresholds),
            "q_rare_thresholds": list(q_rare_thresholds),
            "carrier_unit_thresholds": list(unit_thresholds),
            "mask_counts": mask_counts,
        },
        "carrier_concentration": concentration,
        "M38B_pending_control": "leave_one_atomic_unit_out",
        "limitations": [
            "NAM has 25 REF_TRAIN people and four atomic population/IBD units; posterior rarity is often unresolved.",
            "F0 already contains the 660 selected loci, so M38A cannot estimate incremental value beyond a common-only baseline.",
            "The Beta posteriors treat ancestry-specific reference counts as conditionally independent.",
            "These strata were defined without TARGET, truth, predictions or F1 and are not model-selected results.",
            "Support in two or three units does not by itself establish stability; M38B must add leave-one-atomic-unit-out sensitivity.",
        ],
        "inputs": {**{f"{name}_sha256": digest for name, digest in hashes.items()},
                   "expected_callable_an": dict(expected_an)},
        "outputs": {
            "per_locus_tsv_sha256": sha256_file(args.output_tsv),
            "strata_npz_sha256": sha256_file(args.output_npz),
        },
    }
    write_exclusive_json(args.output_summary, summary)
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M38_RARE_LOCUS_STRATIFICATION_RECEIPT",
        "status": "PASS_OUTPUTS_REOPENED_AND_HASHED",
        "inputs": hashes,
        "outputs": {
            "per_locus_tsv": sha256_file(args.output_tsv),
            "strata_npz": sha256_file(args.output_npz),
            "summary_json": sha256_file(args.output_summary),
        },
        "script_sha256": sha256_file(Path(__file__)),
        "no_overwrite": True,
        "selection_used_target_truth_F0_or_scores": False,
    }
    write_exclusive_json(args.output_receipt, receipt)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--audit-tsv", required=True, type=Path)
    parser.add_argument("--audit-summary", required=True, type=Path)
    parser.add_argument("--selected-sha256", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--audit-tsv-sha256", required=True)
    parser.add_argument("--audit-summary-sha256", required=True)
    parser.add_argument("--expected-loci", type=int, default=660)
    parser.add_argument("--expected-ancestry-an", default="AFR=682,EUR=774,NAM=50")
    parser.add_argument("--beta-priors", default="0.5,1.0")
    parser.add_argument("--rare-af-cutoff", type=float, default=0.01)
    parser.add_argument("--q-top-thresholds", default="0.8,0.9,0.95")
    parser.add_argument("--q-rare-thresholds", default="0.5,0.8,0.95")
    parser.add_argument("--unit-thresholds", default="2,3")
    parser.add_argument("--q-top-draws", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=3801103)
    parser.add_argument("--f0-contains-selected-rare-loci", action="store_true")
    parser.add_argument("--f0-overlap-assertion-source", required=True)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({
        "status": summary["status"],
        "loci": summary["scope"]["loci"],
        "nam_unresolved_principal": summary["posterior"]["diagnostics"]
            ["prior_0P5"]["nam_status_counts"].get("NAM_UNRESOLVED", 0),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
