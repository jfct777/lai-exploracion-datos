#!/usr/bin/env python3
"""Audit the chr22 bridge from raw NatWGS rare variants to a phased scaffold.

The audit is deliberately read-only and fail-closed. It recomputes allele counts
from GT, emits aggregate diagnostics, and never simulates, imputes, trains, runs
LAI, executes relatedness inference, or writes sample identifiers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


MISSING_DOSAGE = 255
VariantKey = tuple[str, int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-wgs-vcf", required=True, type=Path)
    parser.add_argument("--phased-scaffold-vcf", required=True, type=Path)
    parser.add_argument("--gnomix-reference-vcf", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def canonical_contig(value: str) -> str:
    return value.removeprefix("chr")


def canonical_sample_id(value: str) -> str:
    if value.startswith("PSI") and value[3:].isdigit():
        return f"PSI{int(value[3:])}"
    return value


def panel_sample_id(value: str) -> str:
    fields = value.split("_")
    if len(fields) >= 2 and fields[0] == fields[1]:
        return fields[0]
    return value


def baseline_sample_id(value: str) -> str:
    if "_" not in value:
        return value
    return value.split("_", 1)[1]


def canonical_ids(samples: list[str], role: str) -> list[str]:
    if role == "scaffold":
        return [canonical_sample_id(panel_sample_id(sample)) for sample in samples]
    if role == "baseline":
        return [canonical_sample_id(baseline_sample_id(sample)) for sample in samples]
    return [canonical_sample_id(sample) for sample in samples]


def read_vcf_samples(path: Path) -> list[str]:
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
    raise SystemExit(f"Missing #CHROM header in {path.name}")


def parse_gt(sample_field: str) -> tuple[int, bool, bool]:
    gt = sample_field.split(":", 1)[0]
    if "|" in gt:
        separator = "|"
        phased = True
    elif "/" in gt:
        separator = "/"
        phased = False
    else:
        return MISSING_DOSAGE, False, False
    alleles = gt.split(separator)
    if len(alleles) != 2 or any(allele == "." for allele in alleles):
        return MISSING_DOSAGE, phased, False
    try:
        values = [int(allele) for allele in alleles]
    except ValueError:
        return MISSING_DOSAGE, phased, False
    if any(value not in (0, 1) for value in values):
        return MISSING_DOSAGE, phased, False
    return sum(values), phased, True


def info_int(info: str, key: str) -> int | None:
    prefix = f"{key}="
    for field in info.split(";"):
        if field.startswith(prefix):
            raw = field[len(prefix):].split(",", 1)[0]
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def is_biallelic_snv(ref: str, alt: str) -> bool:
    return (
        len(ref) == 1
        and len(alt) == 1
        and ref in "ACGT"
        and alt in "ACGT"
        and "," not in alt
    )


def parse_record(line: str, path: Path) -> tuple[list[str], VariantKey]:
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 8:
        raise SystemExit(f"Malformed VCF record in {path.name}")
    try:
        position = int(fields[1])
    except ValueError as exc:
        raise SystemExit(f"Invalid VCF position in {path.name}") from exc
    key = (canonical_contig(fields[0]), position, fields[3].upper(), fields[4].upper())
    return fields, key


@dataclass(frozen=True)
class ScaffoldGenotypes:
    dosages: bytes
    phased: bytes


@dataclass
class MarkerAudit:
    samples: list[str]
    markers: set[VariantKey]
    positions: set[tuple[str, int]]
    n_records: int
    n_biallelic_snv: int
    n_duplicate_keys: int
    n_records_outside_expected_contig: int
    ordered: bool


def audit_marker_panel(path: Path, expected_contig: str) -> MarkerAudit:
    samples = read_vcf_samples(path)
    markers: set[VariantKey] = set()
    positions: set[tuple[str, int]] = set()
    n_records = n_biallelic = n_duplicates = n_outside = 0
    previous_position: tuple[str, int] | None = None
    ordered = True
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, path)
            chrom, position, ref, alt = key
            n_records += 1
            n_outside += chrom != expected_contig
            current_position = (chrom, position)
            if previous_position is not None and current_position < previous_position:
                ordered = False
            previous_position = current_position
            if not is_biallelic_snv(ref, alt):
                continue
            n_biallelic += 1
            positions.add(current_position)
            if key in markers:
                n_duplicates += 1
            markers.add(key)
    return MarkerAudit(
        samples=samples,
        markers=markers,
        positions=positions,
        n_records=n_records,
        n_biallelic_snv=n_biallelic,
        n_duplicate_keys=n_duplicates,
        n_records_outside_expected_contig=n_outside,
        ordered=ordered,
    )


def audit_scaffold(
    path: Path,
    expected_contig: str,
    raw_canonical_ids: list[str],
) -> tuple[MarkerAudit, dict[VariantKey, ScaffoldGenotypes], dict]:
    samples = read_vcf_samples(path)
    scaffold_ids = canonical_ids(samples, "scaffold")
    indices_by_id: dict[str, list[int]] = {}
    for index, sample_id in enumerate(scaffold_ids):
        indices_by_id.setdefault(sample_id, []).append(index)
    collisions = sum(len(indices) - 1 for indices in indices_by_id.values())
    ambiguous_raw_ids = sum(len(indices_by_id.get(sample_id, [])) > 1 for sample_id in raw_canonical_ids)
    selected_indices = [
        indices_by_id[sample_id][0] if len(indices_by_id.get(sample_id, [])) == 1 else None
        for sample_id in raw_canonical_ids
    ]

    markers: set[VariantKey] = set()
    positions: set[tuple[str, int]] = set()
    genotypes: dict[VariantKey, ScaffoldGenotypes] = {}
    n_records = n_biallelic = n_duplicates = n_outside = 0
    previous_position: tuple[str, int] | None = None
    ordered = True
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, path)
            chrom, position, ref, alt = key
            n_records += 1
            n_outside += chrom != expected_contig
            current_position = (chrom, position)
            if previous_position is not None and current_position < previous_position:
                ordered = False
            previous_position = current_position
            if not is_biallelic_snv(ref, alt):
                continue
            n_biallelic += 1
            positions.add(current_position)
            if key in markers:
                n_duplicates += 1
                continue
            markers.add(key)
            dosages = bytearray()
            phase_flags = bytearray()
            for index in selected_indices:
                if index is None or index + 9 >= len(fields):
                    dosages.append(MISSING_DOSAGE)
                    phase_flags.append(0)
                    continue
                dosage, phased, called = parse_gt(fields[index + 9])
                dosages.append(dosage)
                phase_flags.append(1 if phased and called else 0)
            genotypes[key] = ScaffoldGenotypes(bytes(dosages), bytes(phase_flags))
    audit = MarkerAudit(
        samples=samples,
        markers=markers,
        positions=positions,
        n_records=n_records,
        n_biallelic_snv=n_biallelic,
        n_duplicate_keys=n_duplicates,
        n_records_outside_expected_contig=n_outside,
        ordered=ordered,
    )
    sample_bridge = {
        "n_scaffold_samples": len(samples),
        "n_scaffold_canonical_id_collisions": collisions,
        "n_raw_ids_ambiguous_in_scaffold": ambiguous_raw_ids,
        "n_raw_ids_present_in_scaffold": sum(index is not None for index in selected_indices),
        "n_raw_ids_absent_from_scaffold": sum(index is None for index in selected_indices),
    }
    return audit, genotypes, sample_bridge


def aggregate_concordance(jointly_called: list[int], matches: list[int], floor: int, threshold: float) -> dict:
    concordances = [match / called if called else None for called, match in zip(jointly_called, matches)]
    observed = [value for value in concordances if value is not None]
    below = sum(
        called < floor or value is None or value < threshold
        for called, value in zip(jointly_called, concordances)
    )
    return {
        "n_samples_evaluated": len(concordances),
        "jointly_called_markers_min": min(jointly_called) if jointly_called else 0,
        "jointly_called_markers_median": statistics.median(jointly_called) if jointly_called else 0,
        "jointly_called_markers_max": max(jointly_called) if jointly_called else 0,
        "dosage_concordance_min": min(observed) if observed else None,
        "dosage_concordance_median": statistics.median(observed) if observed else None,
        "dosage_concordance_max": max(observed) if observed else None,
        "n_samples_below_joint_marker_or_concordance_floor": below,
        "sample_ids_emitted": False,
    }


def audit_raw_wgs(
    path: Path,
    expected_contig: str,
    raw_samples: list[str],
    scaffold: dict[VariantKey, ScaffoldGenotypes],
    scaffold_positions: set[tuple[str, int]],
    baseline_markers: set[VariantKey],
    baseline_positions: set[tuple[str, int]],
    rare_min_mac: int,
    rare_maf_threshold: float,
    identity_marker_floor: int,
    identity_concordance_floor: float,
) -> dict:
    n_samples = len(raw_samples)
    jointly_called = [0] * n_samples
    matches = [0] * n_samples
    scaffold_shared_positions: set[tuple[str, int]] = set()
    scaffold_exact_positions: set[tuple[str, int]] = set()
    baseline_shared_positions: set[tuple[str, int]] = set()
    baseline_exact_positions: set[tuple[str, int]] = set()

    counts = {
        "n_records": 0,
        "n_biallelic_snv": 0,
        "n_duplicate_exact_keys": 0,
        "n_records_outside_expected_contig": 0,
        "n_exact_scaffold_markers": 0,
        "n_exact_baseline_markers": 0,
        "n_rare_sites": 0,
        "n_rare_alt_major_sites": 0,
        "n_rare_exact_scaffold_sites": 0,
        "n_rare_exact_baseline_sites": 0,
        "n_direct_phase_bridge_sites": 0,
        "n_rare_heterozygous_gt_phased_raw": 0,
        "n_rare_heterozygous_gt_unphased_raw": 0,
        "n_info_ac_an_checked": 0,
        "n_info_ac_an_mismatch": 0,
    }
    carrier_counts: list[int] = []
    ordered = True
    previous_position: tuple[str, int] | None = None
    keys_at_position: set[VariantKey] = set()
    key_position: tuple[str, int] | None = None

    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, path)
            chrom, position, ref, alt = key
            counts["n_records"] += 1
            counts["n_records_outside_expected_contig"] += chrom != expected_contig
            current_position = (chrom, position)
            if previous_position is not None and current_position < previous_position:
                ordered = False
            previous_position = current_position
            if not is_biallelic_snv(ref, alt):
                continue
            counts["n_biallelic_snv"] += 1
            if current_position != key_position:
                keys_at_position.clear()
                key_position = current_position
            if key in keys_at_position:
                counts["n_duplicate_exact_keys"] += 1
            keys_at_position.add(key)

            sample_fields = fields[9:]
            if len(sample_fields) != n_samples:
                raise SystemExit(f"Unexpected sample-field count in {path.name}")
            parsed = [parse_gt(value) for value in sample_fields]
            alt_dosages = [value[0] for value in parsed]
            called_dosages = [dosage for dosage in alt_dosages if dosage != MISSING_DOSAGE]
            alt_ac = sum(called_dosages)
            an = 2 * len(called_dosages)
            ref_ac = an - alt_ac
            minor_ac = min(alt_ac, ref_ac) if an else 0
            minor_is_alt = alt_ac <= ref_ac
            maf = minor_ac / an if an else None

            info_ac = info_int(fields[7], "AC")
            info_an = info_int(fields[7], "AN")
            if info_ac is not None and info_an is not None:
                counts["n_info_ac_an_checked"] += 1
                counts["n_info_ac_an_mismatch"] += info_ac != alt_ac or info_an != an

            scaffold_entry = scaffold.get(key)
            if current_position in scaffold_positions:
                scaffold_shared_positions.add(current_position)
            if scaffold_entry is not None:
                counts["n_exact_scaffold_markers"] += 1
                scaffold_exact_positions.add(current_position)
                for index, raw_dosage in enumerate(alt_dosages):
                    scaffold_dosage = scaffold_entry.dosages[index]
                    if raw_dosage == MISSING_DOSAGE or scaffold_dosage == MISSING_DOSAGE:
                        continue
                    jointly_called[index] += 1
                    matches[index] += raw_dosage == scaffold_dosage

            if current_position in baseline_positions:
                baseline_shared_positions.add(current_position)
            if key in baseline_markers:
                counts["n_exact_baseline_markers"] += 1
                baseline_exact_positions.add(current_position)

            is_rare = (
                an > 0
                and minor_ac >= rare_min_mac
                and maf is not None
                and maf < rare_maf_threshold
            )
            if not is_rare:
                continue
            counts["n_rare_sites"] += 1
            counts["n_rare_alt_major_sites"] += not minor_is_alt
            minor_dosages = [
                MISSING_DOSAGE
                if dosage == MISSING_DOSAGE
                else dosage if minor_is_alt else 2 - dosage
                for dosage in alt_dosages
            ]
            carriers = [index for index, dosage in enumerate(minor_dosages) if dosage not in (0, MISSING_DOSAGE)]
            carrier_counts.append(len(carriers))
            for dosage, phased, called in parsed:
                if called and dosage == 1:
                    counts[
                        "n_rare_heterozygous_gt_phased_raw"
                        if phased
                        else "n_rare_heterozygous_gt_unphased_raw"
                    ] += 1

            if key in baseline_markers:
                counts["n_rare_exact_baseline_sites"] += 1
            if scaffold_entry is None:
                continue
            counts["n_rare_exact_scaffold_sites"] += 1
            direct_bridge = bool(carriers)
            for index in carriers:
                raw_alt_dosage = alt_dosages[index]
                scaffold_alt_dosage = scaffold_entry.dosages[index]
                if raw_alt_dosage != scaffold_alt_dosage:
                    direct_bridge = False
                    break
                if raw_alt_dosage == 1 and scaffold_entry.phased[index] != 1:
                    direct_bridge = False
                    break
            counts["n_direct_phase_bridge_sites"] += direct_bridge

    sample_concordance = aggregate_concordance(
        jointly_called,
        matches,
        identity_marker_floor,
        identity_concordance_floor,
    )
    counts.update(
        {
            "ordered": ordered,
            "n_shared_scaffold_positions": len(scaffold_shared_positions),
            "n_scaffold_positions_without_exact_allele_match": len(
                scaffold_shared_positions - scaffold_exact_positions
            ),
            "n_shared_baseline_positions": len(baseline_shared_positions),
            "n_baseline_positions_without_exact_allele_match": len(
                baseline_shared_positions - baseline_exact_positions
            ),
            "rare_carrier_count_min": min(carrier_counts) if carrier_counts else None,
            "rare_carrier_count_median": statistics.median(carrier_counts) if carrier_counts else None,
            "rare_carrier_count_max": max(carrier_counts) if carrier_counts else None,
        }
    )
    return {"counts": counts, "sample_concordance": sample_concordance}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gate_table(path: Path, gates: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "name", "status", "reason"), delimiter="\t")
        writer.writeheader()
        writer.writerows(gates)


def run(args: argparse.Namespace) -> dict:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27B_RARE_SCAFFOLD_BRIDGE":
        raise SystemExit("Invalid M27B preregistration stage")
    contract = prereg["frozen_contract"]
    expected_contig = str(contract["chromosome"])

    raw_samples = read_vcf_samples(args.raw_wgs_vcf)
    raw_ids = canonical_ids(raw_samples, "raw")
    raw_id_collisions = len(raw_ids) - len(set(raw_ids))
    scaffold_audit, scaffold_genotypes, sample_bridge = audit_scaffold(
        args.phased_scaffold_vcf,
        expected_contig,
        raw_ids,
    )
    baseline_audit = audit_marker_panel(args.gnomix_reference_vcf, expected_contig)
    baseline_ids = set(canonical_ids(baseline_audit.samples, "baseline"))
    raw_baseline_overlap = len(set(raw_ids) & baseline_ids)

    raw = audit_raw_wgs(
        args.raw_wgs_vcf,
        expected_contig,
        raw_samples,
        scaffold_genotypes,
        scaffold_audit.positions,
        baseline_audit.markers,
        baseline_audit.positions,
        int(contract["rare_min_mac"]),
        float(contract["rare_maf_threshold"]),
        int(contract["identity_min_jointly_called_markers"]),
        float(contract["identity_min_dosage_concordance"]),
    )
    counts = raw["counts"]
    identity = raw["sample_concordance"]

    b0_pass = all(
        (
            counts["n_records"] > 0,
            counts["n_biallelic_snv"] > 0,
            counts["n_records_outside_expected_contig"] == 0,
            counts["n_duplicate_exact_keys"] == 0,
            counts["ordered"],
            scaffold_audit.n_records > 0,
            scaffold_audit.n_records_outside_expected_contig == 0,
            scaffold_audit.n_duplicate_keys == 0,
            scaffold_audit.ordered,
            baseline_audit.n_records > 0,
            baseline_audit.n_records_outside_expected_contig == 0,
            baseline_audit.n_duplicate_keys == 0,
            baseline_audit.ordered,
        )
    )
    b1_pass = all(
        (
            len(raw_samples) == int(contract["expected_raw_samples"]),
            raw_id_collisions == 0,
            sample_bridge["n_raw_ids_ambiguous_in_scaffold"] == 0,
            sample_bridge["n_raw_ids_absent_from_scaffold"] == 0,
            identity["n_samples_below_joint_marker_or_concordance_floor"] == 0,
        )
    )
    b2_pass = counts["n_rare_sites"] > 0
    b3_pass = counts["n_direct_phase_bridge_sites"] > 0
    baseline_fraction = (
        counts["n_exact_baseline_markers"] / len(baseline_audit.markers)
        if baseline_audit.markers
        else 0.0
    )
    b4_pass = baseline_fraction >= float(contract["minimum_pretrained_model_marker_fraction"])

    gates = [
        {
            "gate": "B0",
            "name": "input_and_coordinate_contract",
            "status": "PASS" if b0_pass else "FAIL",
            "reason": "Observed records, contig, order and exact-key uniqueness.",
        },
        {
            "gate": "B1",
            "name": "sample_identity_bridge",
            "status": "PASS" if b1_pass else "FAIL",
            "reason": "Canonical sample overlap and aggregate per-sample dosage fingerprints.",
        },
        {
            "gate": "B2",
            "name": "raw_minor_allele_support",
            "status": "PASS" if b2_pass else "FAIL",
            "reason": "GT-recomputed minor MAC and MAF in the complete 128-sample raw panel; descriptive only.",
        },
        {
            "gate": "B3",
            "name": "direct_rare_phase_bridge",
            "status": "PASS" if b3_pass else "FAIL",
            "reason": "Exact rare markers with concordant carriers and phased informative heterozygotes in the scaffold.",
        },
        {
            "gate": "B4",
            "name": "frozen_baseline_marker_coverage_from_raw_wgs",
            "status": "PASS" if b4_pass else "FAIL",
            "reason": "Exact frozen-model marker coverage from raw WGS; marker-only diagnostic.",
        },
    ]

    if not b0_pass:
        decision = "STOP_INPUT_CONTRACT"
    elif not b1_pass:
        decision = "STOP_SAMPLE_IDENTITY"
    elif not b2_pass:
        decision = "STOP_NO_RAW_RARE_SUPPORT"
    elif not b3_pass:
        decision = "STOP_NO_DIRECT_RARE_PHASE_BRIDGE"
    elif not b4_pass:
        decision = "BRIDGE_PRESENT_BASELINE_REDESIGN_REQUIRED"
    else:
        decision = "GO_PC_RELATE_DONOR_AUDIT_ONLY"

    input_contract = {
        "stage": prereg["stage"],
        "scope": prereg["scope"],
        "chromosome": expected_contig,
        "build_declared": contract["build"],
        "raw_wgs": {
            "n_samples": len(raw_samples),
            "n_canonical_id_collisions": raw_id_collisions,
            **{key: counts[key] for key in (
                "n_records",
                "n_biallelic_snv",
                "n_duplicate_exact_keys",
                "n_records_outside_expected_contig",
                "ordered",
            )},
        },
        "scaffold": {
            "n_samples": len(scaffold_audit.samples),
            "n_records": scaffold_audit.n_records,
            "n_biallelic_snv": scaffold_audit.n_biallelic_snv,
            "n_duplicate_exact_keys": scaffold_audit.n_duplicate_keys,
            "n_records_outside_expected_contig": scaffold_audit.n_records_outside_expected_contig,
            "ordered": scaffold_audit.ordered,
        },
        "baseline": {
            "n_samples": len(baseline_audit.samples),
            "n_markers": len(baseline_audit.markers),
            "n_duplicate_exact_keys": baseline_audit.n_duplicate_keys,
            "n_records_outside_expected_contig": baseline_audit.n_records_outside_expected_contig,
            "ordered": baseline_audit.ordered,
        },
        "imputation_padding_liftover_or_allele_substitution_performed": False,
    }
    sample_identity = {
        **sample_bridge,
        "n_raw_canonical_id_collisions": raw_id_collisions,
        "n_exact_raw_ids_in_frozen_baseline": raw_baseline_overlap,
        "identity_min_jointly_called_markers": int(contract["identity_min_jointly_called_markers"]),
        "identity_min_dosage_concordance": float(contract["identity_min_dosage_concordance"]),
        **identity,
        "sample_ids_emitted": False,
    }
    rare_support = {
        "definition": {
            "orientation": "minor_allele_recomputed_from_GT",
            "minimum_mac": int(contract["rare_min_mac"]),
            "maf_operator": contract["rare_maf_operator"],
            "maf_threshold": float(contract["rare_maf_threshold"]),
            "population": "complete_raw_NatWGS_chr22_panel_diagnostic_only",
        },
        **{key: counts[key] for key in (
            "n_rare_sites",
            "n_rare_alt_major_sites",
            "rare_carrier_count_min",
            "rare_carrier_count_median",
            "rare_carrier_count_max",
            "n_info_ac_an_checked",
            "n_info_ac_an_mismatch",
        )},
        "independent_carrier_units_estimated": False,
        "pcrelate_executed": False,
    }
    phase_bridge = {
        **{key: counts[key] for key in (
            "n_exact_scaffold_markers",
            "n_shared_scaffold_positions",
            "n_scaffold_positions_without_exact_allele_match",
            "n_rare_exact_scaffold_sites",
            "n_direct_phase_bridge_sites",
            "n_rare_heterozygous_gt_phased_raw",
            "n_rare_heterozygous_gt_unphased_raw",
        )},
        "phase_inferred_for_scaffold_absent_rare_sites": False,
        "interpretation": "Only exact rare sites with concordant carriers can establish a direct phase bridge.",
    }
    baseline_overlap = {
        "n_frozen_baseline_markers": len(baseline_audit.markers),
        "n_exact_baseline_markers_in_raw_wgs": counts["n_exact_baseline_markers"],
        "exact_frozen_baseline_marker_fraction": baseline_fraction,
        "minimum_required_fraction": float(contract["minimum_pretrained_model_marker_fraction"]),
        "n_shared_baseline_positions": counts["n_shared_baseline_positions"],
        "n_baseline_positions_without_exact_allele_match": counts[
            "n_baseline_positions_without_exact_allele_match"
        ],
        "n_rare_exact_baseline_sites": counts["n_rare_exact_baseline_sites"],
        "parental_completeness_established": False,
    }
    summary = {
        "stage": prereg["stage"],
        "scope": prereg["scope"],
        "decision": decision,
        "gates": {gate["gate"]: gate["status"] for gate in gates},
        "simulation_performed": False,
        "lai_inference_performed": False,
        "model_training_performed": False,
        "pcrelate_executed": False,
        "test_opened": False,
        "sample_ids_emitted": False,
        "interpretation": (
            "M27B audits a technical bridge only. A PASS cannot establish independent parentals, "
            "locus truth, power or rare-variant improvement of LAI."
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "m27b_input_contract.json", input_contract)
    write_json(args.outdir / "m27b_sample_identity.json", sample_identity)
    write_json(args.outdir / "m27b_rare_support.json", rare_support)
    write_json(args.outdir / "m27b_phase_bridge.json", phase_bridge)
    write_json(args.outdir / "m27b_baseline_overlap.json", baseline_overlap)
    write_json(args.outdir / "m27b_rare_scaffold_bridge_summary.json", summary)
    write_gate_table(args.outdir / "m27b_rare_scaffold_bridge_gates.tsv", gates)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
