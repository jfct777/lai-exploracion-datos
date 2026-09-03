#!/usr/bin/env python3
"""Evaluate a REF-frozen site catalog once in SOURCE_VALID; SOURCE_TEST stays sealed."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
from collections import Counter
from pathlib import Path

import audit_m27f_ref_support as refaudit
from audit_rare_scaffold_bridge import parse_gt, parse_record
from m27f_validation_contract import build_validation_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--valid-bcf", type=Path, required=True)
    parser.add_argument("--split-private", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--projection-public", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--ref-support-private", type=Path, required=True)
    parser.add_argument("--ref-primary-catalog", type=Path, required=True)
    parser.add_argument("--ref-public", type=Path, required=True)
    parser.add_argument("--ref-manifest", type=Path, required=True)
    parser.add_argument("--validation-opening", type=Path, required=True)
    parser.add_argument("--claim-py", type=Path, required=True)
    parser.add_argument("--validation-contract-py", type=Path, required=True)
    parser.add_argument("--valid-audit-py", type=Path, required=True)
    parser.add_argument("--ref-audit-py", type=Path, required=True)
    parser.add_argument("--m27e-py", type=Path, required=True)
    parser.add_argument("--bridge-py", type=Path, required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def key_from_row(row: dict[str, object]) -> refaudit.VariantKey:
    return (
        str(row["chrom"]),
        int(row["pos"]),
        str(row["ref"]),
        str(row["alt"]),
    )


def effective_number(counter: Counter[str]) -> float:
    values = [value for value in counter.values() if value > 0]
    total = sum(values)
    return total * total / sum(value * value for value in values) if total else 0.0


def unit_concentration(rows: list[dict[str, object]], column: str) -> dict[str, float | int | None]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(
            unit for unit in str(row[column]).split(";") if unit
        )
    total = sum(counts.values())
    return {
        "n_contributing_units": len(counts),
        "effective_units_by_site_support": effective_number(counts),
        "largest_unit_share_of_site_support": (
            max(counts.values(), default=0) / total if total else None
        ),
    }


def max_sites_in_span(positions: list[int], width: int) -> int:
    ordered = sorted(positions)
    left = 0
    best = 0
    for right, position in enumerate(ordered):
        while position - ordered[left] >= width:
            left += 1
        best = max(best, right - left + 1)
    return best


def spatial_concentration(rows: list[dict[str, object]], widths: list[int]) -> dict[str, object]:
    positions = [int(row["pos"]) for row in rows]
    result: dict[str, object] = {
        "n_sites": len(positions),
        "minimum_bp": min(positions, default=None),
        "maximum_bp": max(positions, default=None),
    }
    for width in widths:
        maximum = max_sites_in_span(positions, width) if positions else 0
        result[f"max_sites_in_any_{width}_bp_span"] = maximum
        result[f"max_fraction_in_any_{width}_bp_span"] = (
            maximum / len(positions) if positions else None
        )
    return result


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    clean = sorted(values)
    return {
        "minimum": min(clean, default=None),
        "median": statistics.median(clean) if clean else None,
        "maximum": max(clean, default=None),
    }


def sensitivity_summary(
    rows: list[dict[str, object]], thresholds: list[int], required_valid_units: int
) -> dict[str, dict[str, int]]:
    result = {}
    for threshold in thresholds:
        selected = [
            row
            for row in rows
            if int(row["ref_nam_carrier_atomic_units"]) >= threshold
        ]
        validated = [
            row
            for row in selected
            if int(row["valid_nam_carrier_atomic_units"])
            >= required_valid_units
        ]
        result[str(threshold)] = {
            "selected_in_ref": len(selected),
            "observed_in_both_valid_units": len(validated),
            "validated_in_frozen_baseline": sum(
                parse_bool(row["in_frozen_baseline"]) for row in validated
            ),
            "validated_outside_frozen_baseline": sum(
                not parse_bool(row["in_frozen_baseline"]) for row in validated
            ),
        }
    return result


def write_private(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    refaudit.write_private_tsv(path, rows, fields)
    os.chmod(path, 0o600)


def classify_valid_decision(
    n_transferred: int,
    n_additional: int,
    n_unresolved_transfer: int,
    n_unresolved_additional: int,
) -> str:
    if not n_transferred:
        return (
            "INCONCLUSIVE_VALID_CALLABILITY"
            if n_unresolved_transfer
            else "STOP_VALID_NO_TRANSFERABLE_SUPPORT"
        )
    if not n_additional:
        return (
            "INCONCLUSIVE_ADDITIONAL_VARIANT_KEY_CALLABILITY"
            if n_unresolved_additional
            else "STOP_NO_ADDITIONAL_VARIANT_KEY_SUPPORT"
        )
    return "GO_SUPPORT_CANDIDATES_ONLY"


def joint_support_sets(
    rows: list[dict[str, object]], ref_threshold: int, valid_threshold: int
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    observed = [
        row
        for row in rows
        if int(row["ref_nam_carrier_atomic_units"]) >= ref_threshold
        and int(row["valid_nam_carrier_atomic_units"]) >= valid_threshold
    ]
    possible = [
        row
        for row in rows
        if int(row["ref_nam_carrier_atomic_units_upper_bound"]) >= ref_threshold
        and int(row["valid_nam_carrier_atomic_units_upper_bound"])
        >= valid_threshold
    ]
    observed_additional = [
        row for row in observed if not parse_bool(row["in_frozen_baseline"])
    ]
    possible_additional = [
        row for row in possible if not parse_bool(row["in_frozen_baseline"])
    ]
    return observed, possible, observed_additional, possible_additional


def require_manifest_hash(
    manifest: dict[str, object], section: str, name: str, expected: str
) -> None:
    values = manifest.get(section)
    if not isinstance(values, dict) or values.get(name) != expected:
        raise ValueError(
            f"STOP_PROVENANCE: manifest {section}.{name} does not authenticate"
        )


def authenticate_before_validation(
    args: argparse.Namespace, prereg: dict[str, object]
) -> tuple[
    dict[str, object], dict[str, object], dict[str, object], dict[str, object]
]:
    """Authenticate the complete REF chain before any VALID BCF access."""
    contract = prereg["upstream_contract"]
    if (
        refaudit.sha256_file(args.split_private)
        != contract["m27f_split_private_sha256"]
        or refaudit.sha256_file(args.split_manifest)
        != contract["m27f_split_manifest_sha256"]
    ):
        raise ValueError("STOP_PROVENANCE: canonical split hash differs")

    projection = json.loads(args.projection_public.read_text(encoding="utf-8"))
    projection_manifest = json.loads(
        args.projection_manifest.read_text(encoding="utf-8")
    )
    ref_public = json.loads(args.ref_public.read_text(encoding="utf-8"))
    ref_manifest = json.loads(args.ref_manifest.read_text(encoding="utf-8"))
    opening = json.loads(args.validation_opening.read_text(encoding="utf-8"))

    if (
        projection.get("stage") != "M27F_REF_VALID_MECHANICAL_PROJECTION"
        or projection.get("decision") != "GO_REF_SUPPORT_AUDIT"
        or any(value != "PASS" for value in projection.get("gates", {}).values())
        or projection.get("source_test_projection_created") is not False
        or projection.get("source_test_samples_in_projected_outputs") != 0
        or any(
            int(row.get("n_records_with_nonempty_info", -1)) != 0
            for row in projection.get("projections", {}).values()
        )
        or projection_manifest.get("stage")
        != "M27F_REF_VALID_MECHANICAL_PROJECTION"
        or ref_public.get("stage") != "M27F_REF_SUPPORT_SELECTION"
        or ref_manifest.get("stage") != "M27F_REF_SUPPORT_SELECTION"
        or opening.get("stage") != "M27F_VALIDATION_OPENING"
        or opening.get("source_valid_genotypes_read_to_create_receipt") is not False
        or opening.get("source_test_genotypes_opened") is not False
    ):
        raise ValueError("STOP_PROVENANCE: invalid stage or leakage receipt")

    projection_hash = refaudit.sha256_file(args.projection_public)
    projection_manifest_hash = refaudit.sha256_file(args.projection_manifest)
    support_hash = refaudit.sha256_file(args.ref_support_private)
    catalog_hash = refaudit.sha256_file(args.ref_primary_catalog)
    ref_public_hash = refaudit.sha256_file(args.ref_public)
    ref_manifest_hash = refaudit.sha256_file(args.ref_manifest)
    prereg_hash = refaudit.sha256_file(args.preregistration)

    require_manifest_hash(
        projection_manifest, "sha256", args.projection_public.name, projection_hash
    )
    require_manifest_hash(
        ref_manifest, "inputs", args.projection_public.name, projection_hash
    )
    require_manifest_hash(
        ref_manifest,
        "inputs",
        args.projection_manifest.name,
        projection_manifest_hash,
    )
    require_manifest_hash(ref_manifest, "inputs", args.preregistration.name, prereg_hash)
    require_manifest_hash(ref_manifest, "sha256", args.ref_support_private.name, support_hash)
    require_manifest_hash(ref_manifest, "sha256", args.ref_primary_catalog.name, catalog_hash)
    require_manifest_hash(ref_manifest, "sha256", args.ref_public.name, ref_public_hash)

    expected_opening = (
        "VALIDATION_OPENING_FROZEN"
        if ref_public.get("decision") == "GO_VALID_SUPPORT_AUDIT"
        else "VALIDATION_NOT_AUTHORIZED"
    )
    expected_openings = 1 if expected_opening == "VALIDATION_OPENING_FROZEN" else 0
    expected_receipt_hashes = {
        "projection_public_sha256": projection_hash,
        "projection_manifest_sha256": projection_manifest_hash,
        "ref_support_private_sha256": support_hash,
        "ref_primary_catalog_sha256": catalog_hash,
        "ref_public_sha256": ref_public_hash,
        "ref_manifest_sha256": ref_manifest_hash,
        "preregistration_sha256": prereg_hash,
    }
    if (
        opening.get("decision") != expected_opening
        or opening.get("ref_decision") != ref_public.get("decision")
        or int(opening.get("authorized_analytical_openings", -1))
        != expected_openings
        or any(opening.get(key) != value for key, value in expected_receipt_hashes.items())
        or int(opening.get("primary_ref_min_atomic_units", -1))
        != int(prereg["support_contract"]["primary_ref_min_atomic_units"])
        or int(opening.get("required_valid_atomic_units", -1))
        != int(prereg["support_contract"]["required_valid_atomic_units"])
        or opening.get("registry_claim_key")
        != prereg["support_contract"]["validation_claim_key"]
        or opening.get("registry_claim_uri")
        != prereg["support_contract"]["validation_claim_uri"]
    ):
        raise ValueError("STOP_PROVENANCE: validation-opening receipt differs")
    validation_plan, validation_plan_hash = build_validation_plan(
        {
            "claim": args.claim_py,
            "validation_contract": args.validation_contract_py,
            "valid_audit": args.valid_audit_py,
            "ref_audit": args.ref_audit_py,
            "m27e_audit": args.m27e_py,
            "bridge": args.bridge_py,
        },
        args.container_image,
        args.container_digest,
        prereg,
    )
    if (
        opening.get("validation_plan") != validation_plan
        or opening.get("validation_plan_sha256") != validation_plan_hash
    ):
        raise ValueError("STOP_PROVENANCE: frozen VALID analysis plan differs")
    return projection, projection_manifest, ref_public, opening


def stopped_before_valid(
    args: argparse.Namespace,
    ref_public: dict[str, object],
    ref_rows: list[dict[str, str]],
    opening: dict[str, object],
) -> dict[str, object]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    private_path = args.outdir / "m27f_ref_valid_support.private.tsv"
    shutil.copyfile(args.ref_support_private, private_path)
    os.chmod(private_path, 0o600)
    public = {
        "stage": "M27F_REF_VALID_SUPPORT_AUDIT",
        "decision": ref_public["decision"],
        "gates": {"V0": "PASS", "V1": "NOT_RUN_REF_STOP"},
        "n_frozen_target_sites": len(ref_rows),
        "source_valid_bcf_mechanically_projected": True,
        "source_valid_genotypes_analyzed": False,
        "source_test_genotypes_opened": False,
        "lai_performed": False,
        "simulation_performed": False,
        "model_training_performed": False,
        "private_combined_support_sha256": refaudit.sha256_file(private_path),
        "validation_opening_receipt_sha256": refaudit.sha256_file(
            args.validation_opening
        ),
        "validation_opening_receipt_verified": True,
        "authorized_analytical_openings": opening["authorized_analytical_openings"],
        "interpretation": "REF did not authorize opening SOURCE_VALID.",
    }
    (args.outdir / "m27f_ref_valid_support.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_REF_VALID_SUPPORT_AUDIT" or prereg.get("version") != 2:
        raise ValueError("Invalid M27F-b preregistration")
    contract = prereg["upstream_contract"]
    projection, projection_manifest, ref_public, opening = (
        authenticate_before_validation(args, prereg)
    )
    ref_rows = read_tsv(args.ref_support_private)
    primary_rows = read_tsv(args.ref_primary_catalog)

    if (
        ref_public.get("private_ref_support_sha256")
        != refaudit.sha256_file(args.ref_support_private)
        or ref_public.get("private_primary_catalog_sha256")
        != refaudit.sha256_file(args.ref_primary_catalog)
        or len(ref_rows) != int(contract["expected_direct_phase_bridge_sites"])
    ):
        raise ValueError("STOP_PROVENANCE: REF catalog authentication failed")

    if ref_public.get("decision") != "GO_VALID_SUPPORT_AUDIT":
        return stopped_before_valid(args, ref_public, ref_rows, opening)

    expected_primary = [
        row for row in ref_rows if parse_bool(row["primary_ref_selected"])
    ]
    if [key_from_row(row) for row in expected_primary] != [
        key_from_row(row) for row in primary_rows
    ]:
        raise ValueError("STOP_CATALOG_FREEZE: primary catalog differs from REF table")

    split_rows = read_tsv(args.split_private)
    valid_rows = [row for row in split_rows if row["role"] == "SOURCE_VALID"]
    test_samples = {
        row["sample_id"] for row in split_rows if row["role"] == "SOURCE_TEST"
    }
    valid_samples = [row["sample_id"] for row in valid_rows]
    if len(valid_samples) != int(contract["expected_valid_samples"]):
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: unexpected VALID sample count")
    observed_valid_samples = refaudit.read_bcf_samples(args.valid_bcf, args.bcftools)
    if observed_valid_samples != valid_samples:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: VALID BCF header differs from split")
    if set(observed_valid_samples) & test_samples:
        raise ValueError("STOP_TEST_LEAKAGE: SOURCE_TEST appeared in VALID BCF")
    if (
        projection["projections"]["SOURCE_VALID"]["bcf_sha256"]
        != refaudit.sha256_file(args.valid_bcf)
    ):
        raise ValueError("STOP_VARIANT_PROJECTION: VALID BCF hash differs")
    require_manifest_hash(
        projection_manifest,
        "sha256",
        args.valid_bcf.name,
        projection["projections"]["SOURCE_VALID"]["bcf_sha256"],
    )

    metadata = {row["sample_id"]: row for row in valid_rows}
    ancestry_indices = {
        ancestry: [
            index
            for index, sample in enumerate(valid_samples)
            if metadata[sample]["ancestry"] == ancestry
        ]
        for ancestry in refaudit.ANCESTRY_PREFIX
    }
    expected_by_ancestry = {
        key: int(value)
        for key, value in contract["expected_samples_by_ancestry_and_role"][
            "SOURCE_VALID"
        ].items()
    }
    if {
        ancestry: len(indices) for ancestry, indices in ancestry_indices.items()
    } != expected_by_ancestry:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: VALID ancestry counts differ")
    valid_nam_units = {
        metadata[sample]["atomic_unit_id"]
        for sample in valid_samples
        if metadata[sample]["ancestry"] == "Native_American"
    }
    if len(valid_nam_units) != int(
        contract["expected_valid_native_american_atomic_units"]
    ):
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: unexpected NAM VALID unit count")

    by_key: dict[refaudit.VariantKey, dict[str, object]] = {
        key_from_row(row): dict(row) for row in ref_rows
    }
    found: set[refaudit.VariantKey] = set()
    with refaudit.open_variant_text(args.valid_bcf, args.bcftools) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, args.valid_bcf)
            row = by_key.get(key)
            if row is None:
                continue
            if key in found:
                raise ValueError("STOP_TARGET_REPRODUCTION: duplicate VALID target key")
            found.add(key)
            parsed = [parse_gt(value) for value in fields[9:]]
            if len(parsed) != len(valid_samples):
                raise ValueError("Unexpected VALID genotype count")
            minor_is_alt = parse_bool(row["minor_is_alt"])
            for ancestry, indices in ancestry_indices.items():
                prefix = refaudit.ANCESTRY_PREFIX[ancestry]
                metrics = refaudit.role_metrics(
                    parsed, indices, valid_samples, metadata, minor_is_alt
                )
                for name, value in metrics.items():
                    row[f"valid_{prefix}_{name}"] = value

    if found != set(by_key):
        raise ValueError("STOP_TARGET_REPRODUCTION: VALID lacks frozen target keys")
    combined = [by_key[key] for key in sorted(by_key, key=lambda item: (int(item[0].removeprefix("chr")), item[1], item[2], item[3]))]
    primary = [
        row
        for row in combined
        if int(row["ref_nam_carrier_atomic_units"])
        >= int(prereg["support_contract"]["primary_ref_min_atomic_units"])
    ]
    primary_outside = [
        row for row in primary if not parse_bool(row["in_frozen_baseline"])
    ]
    ref_threshold = int(prereg["support_contract"]["primary_ref_min_atomic_units"])
    valid_threshold = int(
        prereg["support_contract"]["required_valid_atomic_units"]
    )
    transferred, possible_transfer, additional, possible_additional = (
        joint_support_sets(combined, ref_threshold, valid_threshold)
    )
    unresolved_transfer = len(possible_transfer) - len(transferred)
    unresolved_additional = len(possible_additional) - len(additional)
    decision = classify_valid_decision(
        len(transferred),
        len(additional),
        unresolved_transfer,
        unresolved_additional,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    private_path = args.outdir / "m27f_ref_valid_support.private.tsv"
    write_private(private_path, combined, list(combined[0]))

    widths = [int(value) for value in prereg["diagnostics"]["spatial_span_bp"]]
    public = {
        "stage": "M27F_REF_VALID_SUPPORT_AUDIT",
        "decision": decision,
        "gates": {
            "V0": "PASS",
            "V1": "PASS",
            "V2": "PASS",
            "V3": "PASS" if not unresolved_transfer else "WARN",
            "V4": (
                "PASS"
                if transferred
                else "WARN"
                if decision == "INCONCLUSIVE_VALID_CALLABILITY"
                else "FAIL"
            ),
            "V5": (
                "NOT_EVALUABLE"
                if not transferred
                else "PASS"
                if additional
                else "WARN"
                if decision == "INCONCLUSIVE_ADDITIONAL_VARIANT_KEY_CALLABILITY"
                else "FAIL"
            ),
        },
        "n_frozen_target_sites": len(combined),
        "n_ref_primary_sites": len(primary),
        "n_ref_primary_sites_outside_frozen_baseline": len(primary_outside),
        "n_sites_transferred_ref_to_both_valid_units": len(transferred),
        "n_transferred_sites_in_frozen_baseline": sum(
            parse_bool(row["in_frozen_baseline"]) for row in transferred
        ),
        "n_transferred_exact_variant_keys_outside_frozen_baseline": len(additional),
        "n_valid_native_american_atomic_units": len(valid_nam_units),
        "n_primary_sites_with_all_valid_units_callable": (
            sum(
                int(row["valid_nam_fully_callable_atomic_units"])
                == len(valid_nam_units)
                for row in primary
            )
        ),
        "n_primary_outside_sites_with_all_valid_units_callable": (
            sum(
                int(row["valid_nam_fully_callable_atomic_units"])
                == len(valid_nam_units)
                for row in primary_outside
            )
        ),
        "n_sites_whose_joint_transfer_is_unresolved_by_missingness": unresolved_transfer,
        "n_additional_exact_variant_keys_whose_joint_transfer_is_unresolved_by_missingness": unresolved_additional,
        "ref_threshold_sensitivity": sensitivity_summary(
            combined,
            [int(value) for value in prereg["diagnostics"]["ref_support_thresholds"]],
            int(prereg["support_contract"]["required_valid_atomic_units"]),
        ),
        "valid_frozen_allele_frequency_by_ancestry": {
            ancestry: numeric_summary(
                [
                    float(row[f"valid_{prefix}_minor_af"])
                    for row in combined
                    if row[f"valid_{prefix}_minor_af"] not in ("", None)
                ]
            )
            for ancestry, prefix in refaudit.ANCESTRY_PREFIX.items()
        },
        "transferred_site_spatial_concentration": spatial_concentration(
            transferred, widths
        ),
        "additional_site_spatial_concentration": spatial_concentration(
            additional, widths
        ),
        "ref_unit_concentration_among_additional_sites": unit_concentration(
            additional, "ref_nam_carrier_unit_ids"
        ),
        "valid_unit_concentration_among_additional_sites": unit_concentration(
            additional, "valid_nam_carrier_unit_ids"
        ),
        "private_combined_support_sha256": refaudit.sha256_file(private_path),
        "primary_catalog_sha256_before_validation": refaudit.sha256_file(
            args.ref_primary_catalog
        ),
        "validation_opening_receipt_sha256": refaudit.sha256_file(
            args.validation_opening
        ),
        "validation_opening_receipt_verified": True,
        "authorized_analytical_openings": opening["authorized_analytical_openings"],
        "validation_used_once": True,
        "source_valid_bcf_mechanically_projected": True,
        "source_valid_genotypes_analyzed": True,
        "validation_changed_catalog_or_threshold": False,
        "source_test_projection_created": False,
        "source_test_genotypes_opened": False,
        "m27e_full_panel_counts_used_for_selection": False,
        "lai_performed": False,
        "simulation_performed": False,
        "model_training_performed": False,
        "sample_ids_emitted_publicly": False,
        "population_labels_emitted_publicly": False,
        "holdout_status": "post_screen_holdout_not_independent_external_validation",
        "interpretation": (
            "A site is a feature, not a biological replicate. Transfer is conditional "
            "on four REF and two VALID Native-American atomic units. PASS does not show "
            "LAI improvement, population specificity or generalization to SOURCE_TEST. "
            "An exact variant key outside the frozen baseline is not necessarily a new locus."
        ),
    }
    (args.outdir / "m27f_ref_valid_support.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
