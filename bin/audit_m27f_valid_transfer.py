#!/usr/bin/env python3
"""Audit one-shot VALID transfer of the frozen M27F REF catalog; TEST stays sealed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from audit_m27f_ref_support import (
    catalog_digest,
    open_variant_text,
    read_bcf_samples,
    sha256_file,
    usable_carrier,
    write_private_tsv,
)
from audit_rare_scaffold_bridge import parse_gt, parse_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--valid-bcf", type=Path, required=True)
    parser.add_argument("--split-private", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--projection-public", type=Path, required=True)
    parser.add_argument("--ref-eligible-catalog", type=Path, required=True)
    parser.add_argument("--ref-support-public", type=Path, required=True)
    parser.add_argument("--ref-support-manifest", type=Path, required=True)
    parser.add_argument("--historical-baseline-vcf", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def key_from_row(row: dict[str, str]) -> tuple[str, int, str, str]:
    return (row["chrom"].removeprefix("chr"), int(row["pos"]), row["ref"], row["alt"])


def key_pattern_digest(rows: list[dict[str, str]]) -> str:
    lines = [
        f"{chrom}:{position}:{ref}:{alt}|{row['ref_carrier_pattern_sha256']}"
        for row in rows
        for chrom, position, ref, alt in [key_from_row(row)]
    ]
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode("utf-8")).hexdigest()


def read_variant_keys(path: Path, bcftools: str) -> set[tuple[str, int, str, str]]:
    keys: set[tuple[str, int, str, str]] = set()
    with open_variant_text(path, bcftools) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0].removeprefix("chr")
            for alt in fields[4].split(","):
                keys.add((chrom, int(fields[1]), fields[3], alt))
    return keys


def read_genetic_map(path: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            points.append((int(fields[-2]), float(fields[-1])))
    if len(points) < 2 or points != sorted(points):
        raise ValueError("STOP_PROVENANCE: genetic map is not ordered")
    return points


def interpolate_cm(position: int, points: list[tuple[int, float]]) -> float:
    if position <= points[0][0]:
        return points[0][1]
    if position >= points[-1][0]:
        return points[-1][1]
    left = 0
    right = len(points) - 1
    while right - left > 1:
        middle = (left + right) // 2
        if points[middle][0] <= position:
            left = middle
        else:
            right = middle
    left_bp, left_cm = points[left]
    right_bp, right_cm = points[right]
    fraction = (position - left_bp) / (right_bp - left_bp)
    return left_cm + fraction * (right_cm - left_cm)


def block_state(called: int, carriers: int, unphased_hets: int) -> str:
    if called == 0:
        return "UNEVALUABLE_CALLABILITY"
    if carriers > 0:
        return "PRESENT"
    if unphased_hets > 0:
        return "UNEVALUABLE_PHASE"
    return "ABSENT"


def transfer_decision(
    passing_valid_patterns: set[str],
    potentially_transferable_rows: int,
) -> str:
    if len(passing_valid_patterns) >= 2:
        return "PASS_LOCAL_TRANSFER"
    if len(passing_valid_patterns) + potentially_transferable_rows >= 2:
        return "INCONCLUSIVE_TECHNICAL"
    return "STOP_M27_LOCAL_TRANSFER"


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_VALID_LOCAL_TRANSFER" or prereg.get("version") != 1:
        raise ValueError("Invalid M27F VALID preregistration")
    contract = prereg["upstream_contract"]
    observed = {
        "m27f_split_manifest_sha256": sha256_file(args.split_manifest),
        "m27f_split_private_sha256": sha256_file(args.split_private),
        "m27f_ref_support_manifest_sha256": sha256_file(args.ref_support_manifest),
        "m27f_ref_support_public_sha256": sha256_file(args.ref_support_public),
        "m27f_ref_eligible_catalog_sha256": sha256_file(args.ref_eligible_catalog),
        "historical_baseline_vcf_sha256": sha256_file(args.historical_baseline_vcf),
        "genetic_map_chr22_sha256": sha256_file(args.genetic_map),
    }
    if any(observed[key] != contract[key] for key in observed):
        raise ValueError("STOP_PROVENANCE: an upstream hash differs")
    projection = json.loads(args.projection_public.read_text(encoding="utf-8"))
    if (
        projection.get("decision") != "GO_VALID_TRANSFER_AUDIT"
        or projection.get("valid_bcf_sha256") != sha256_file(args.valid_bcf)
        or projection.get("source_test_opened") is not False
    ):
        raise ValueError("STOP_VARIANT_PROJECTION: VALID projection receipt differs")

    with args.split_private.open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle, delimiter="\t"))
    valid_rows = [row for row in split_rows if row["role"] == "SOURCE_VALID"]
    valid_samples = [row["sample_id"] for row in valid_rows]
    if read_bcf_samples(args.valid_bcf, args.bcftools) != valid_samples:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: VALID BCF header differs")
    metadata = {row["sample_id"]: row for row in valid_rows}
    ancestry_blocks = {
        ancestry: sorted({row["atomic_unit_id"] for row in valid_rows if row["ancestry"] == ancestry})
        for ancestry in ("African", "European", "Native_American")
    }
    nam_blocks = ancestry_blocks["Native_American"]
    if len(nam_blocks) != int(contract["expected_valid_native_american_atomic_units"]):
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: unexpected NAM VALID block count")

    with args.ref_eligible_catalog.open("r", encoding="utf-8", newline="") as handle:
        frozen_rows = list(csv.DictReader(handle, delimiter="\t"))
    catalog = {key_from_row(row): row["minor_is_alt"] == "True" for row in frozen_rows}
    if (
        len(frozen_rows) != int(contract["expected_frozen_sites"])
        or catalog_digest(catalog) != contract["expected_frozen_key_orientation_sha256"]
    ):
        raise ValueError("STOP_FROZEN_CATALOG: key or orientation digest differs")
    frozen_by_key = {key_from_row(row): row for row in frozen_rows}
    baseline_keys = read_variant_keys(args.historical_baseline_vcf, args.bcftools)
    primary_rows = [row for row in frozen_rows if key_from_row(row) not in baseline_keys]
    overlap_rows = [row for row in frozen_rows if key_from_row(row) in baseline_keys]
    primary_catalog = {key_from_row(row): row["minor_is_alt"] == "True" for row in primary_rows}
    if (
        len(overlap_rows) != int(contract["expected_historical_baseline_overlap_sites"])
        or len(primary_rows) != int(contract["expected_primary_historical_baseline_disjoint_sites"])
        or len({row["ref_carrier_pattern_sha256"] for row in primary_rows})
        != int(contract["expected_primary_ref_patterns"])
        or catalog_digest(primary_catalog) != contract["expected_primary_key_orientation_sha256"]
        or key_pattern_digest(primary_rows) != contract["expected_primary_key_ref_pattern_sha256"]
    ):
        raise ValueError("STOP_FROZEN_CATALOG: historical-baseline classification differs")

    points = read_genetic_map(args.genetic_map)
    sample_index = {sample: index for index, sample in enumerate(valid_samples)}
    block_indices = {
        block: [index for index, sample in enumerate(valid_samples) if metadata[sample]["atomic_unit_id"] == block]
        for blocks in ancestry_blocks.values()
        for block in blocks
    }
    block_rank = {
        block: f"{ancestry.lower()}_block_{rank:02d}"
        for ancestry, blocks in ancestry_blocks.items()
        for rank, block in enumerate(blocks, start=1)
    }

    site_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    found: set[tuple[str, int, str, str]] = set()
    with open_variant_text(args.valid_bcf, args.bcftools) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, raw_key = parse_record(line, args.valid_bcf)
            key = (raw_key[0].removeprefix("chr"), raw_key[1], raw_key[2], raw_key[3])
            if key not in catalog:
                continue
            if key in found:
                raise ValueError("STOP_FROZEN_CATALOG: duplicate key in VALID")
            found.add(key)
            parsed = [parse_gt(value) for value in fields[9:]]
            if len(parsed) != len(valid_samples):
                raise ValueError("STOP_OUTPUT_MEMBERSHIP: genotype count differs")
            frozen = frozen_by_key[key]
            minor_is_alt = catalog[key]
            carrier_vector = bytearray(
                255 if not called else 1 if usable_carrier(alt, phased, called, minor_is_alt) else 0
                for alt, phased, called in parsed
            )
            site: dict[str, object] = {
                "chrom": key[0],
                "pos": key[1],
                "ref": key[2],
                "alt": key[3],
                "minor_allele": frozen["minor_allele"],
                "minor_is_alt": minor_is_alt,
                "ref_carrier_pattern_sha256": frozen["ref_carrier_pattern_sha256"],
                "valid_carrier_pattern_sha256": hashlib.sha256(bytes(carrier_vector)).hexdigest(),
                "historical_baseline_exact_key": key in baseline_keys,
                "primary_for_local_transfer": key not in baseline_keys,
                "genetic_position_cm": round(interpolate_cm(key[1], points), 8),
            }
            nam_states: list[str] = []
            for ancestry, blocks in ancestry_blocks.items():
                ancestry_carrier_blocks = 0
                ancestry_callable_blocks = 0
                block_rates: list[float] = []
                for block in blocks:
                    indices = block_indices[block]
                    called = sum(parsed[index][2] for index in indices)
                    carriers = sum(
                        usable_carrier(*parsed[index], minor_is_alt) for index in indices
                    )
                    unphased = sum(
                        parsed[index][2] and parsed[index][0] == 1 and not parsed[index][1]
                        for index in indices
                    )
                    state = block_state(called, carriers, unphased)
                    if called:
                        ancestry_callable_blocks += 1
                        block_rates.append(carriers / called)
                    ancestry_carrier_blocks += carriers > 0
                    if ancestry == "Native_American":
                        nam_states.append(state)
                    block_rows.append(
                        {
                            "chrom": key[0],
                            "pos": key[1],
                            "ref": key[2],
                            "alt": key[3],
                            "primary_for_local_transfer": key not in baseline_keys,
                            "ancestry": ancestry,
                            "block_token": block_rank[block],
                            "n_samples": len(indices),
                            "n_called_samples": called,
                            "n_usable_carriers": carriers,
                            "n_unphased_heterozygotes": unphased,
                            "carrier_rate_among_called": round(carriers / called, 8) if called else "",
                            "state": state,
                        }
                    )
                prefix = {"African": "afr", "European": "eur", "Native_American": "nam"}[ancestry]
                site[f"n_{prefix}_blocks"] = len(blocks)
                site[f"n_{prefix}_callable_blocks"] = ancestry_callable_blocks
                site[f"n_{prefix}_carrier_blocks"] = ancestry_carrier_blocks
                site[f"mean_{prefix}_block_carrier_rate"] = (
                    round(sum(block_rates) / len(block_rates), 8) if block_rates else ""
                )
            site["nam_block_states"] = ";".join(nam_states)
            site["transfers_all_nam_valid_blocks"] = all(state == "PRESENT" for state in nam_states)
            site["has_technical_uncertainty"] = any(state.startswith("UNEVALUABLE") for state in nam_states)
            site["has_explicit_absence"] = "ABSENT" in nam_states
            site_rows.append(site)

    if found != set(catalog):
        raise ValueError("STOP_FROZEN_CATALOG: VALID does not contain all six frozen keys")
    site_rows.sort(key=lambda row: int(row["pos"]))
    block_rows.sort(key=lambda row: (int(row["pos"]), str(row["ancestry"]), str(row["block_token"])))
    primary_site_rows = [row for row in site_rows if row["primary_for_local_transfer"]]
    passing_rows = [row for row in primary_site_rows if row["transfers_all_nam_valid_blocks"]]
    passing_patterns = {str(row["valid_carrier_pattern_sha256"]) for row in passing_rows}
    potentially_transferable = sum(
        bool(row["has_technical_uncertainty"]) and not bool(row["has_explicit_absence"])
        for row in primary_site_rows
    )
    decision = transfer_decision(passing_patterns, potentially_transferable)

    args.outdir.mkdir(parents=True, exist_ok=True)
    site_path = args.outdir / "m27f_valid_site_support.private.tsv"
    block_path = args.outdir / "m27f_valid_block_support.private.tsv"
    write_private_tsv(site_path, site_rows, list(site_rows[0]))
    write_private_tsv(block_path, block_rows, list(block_rows[0]))
    primary_positions = [int(row["pos"]) for row in primary_site_rows]
    primary_cm = [float(row["genetic_position_cm"]) for row in primary_site_rows]
    state_counts = Counter(
        state
        for row in primary_site_rows
        for state in str(row["nam_block_states"]).split(";")
    )
    public = {
        "stage": prereg["stage"],
        "decision": decision,
        "gates": {
            "V0": "PASS",
            "V1": "PASS",
            "V2": "PASS",
            "V3": "PASS",
            "V4": "PASS" if decision == "PASS_LOCAL_TRANSFER" else "INCONCLUSIVE" if decision == "INCONCLUSIVE_TECHNICAL" else "FAIL",
        },
        "n_frozen_sites_audited": len(site_rows),
        "n_historical_baseline_exact_key_controls": len(overlap_rows),
        "n_primary_historical_baseline_disjoint_sites": len(primary_site_rows),
        "n_primary_ref_patterns": len({row["ref_carrier_pattern_sha256"] for row in primary_site_rows}),
        "n_primary_sites_transferred_all_nam_valid_blocks": len(passing_rows),
        "n_distinct_valid_carrier_patterns_transferred": len(passing_patterns),
        "n_valid_samples": len(valid_samples),
        "n_valid_native_american_atomic_units": len(nam_blocks),
        "primary_nam_cell_states": dict(sorted(state_counts.items())),
        "primary_geometry": {
            "span_bp": max(primary_positions) - min(primary_positions),
            "span_cm": round(max(primary_cm) - min(primary_cm), 8),
        },
        "frozen_key_orientation_sha256": catalog_digest(catalog),
        "primary_key_orientation_sha256": catalog_digest(primary_catalog),
        "primary_key_ref_pattern_sha256": key_pattern_digest(primary_rows),
        "private_site_support_sha256": sha256_file(site_path),
        "private_block_support_sha256": sha256_file(block_path),
        "source_valid_opened_once": True,
        "source_test_opened": False,
        "lai_performed": False,
        "simulation_performed": False,
        "model_training_performed": False,
        "sample_ids_emitted_publicly": False,
        "population_labels_emitted_publicly": False,
        "interpretation": (
            "One-shot structural transfer gate. Historical-baseline exact-key disjunction does not prove incremental information. "
            "PASS authorizes only design of a local three-arm comparison."
        ),
    }
    (args.outdir / "m27f_valid_transfer.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
