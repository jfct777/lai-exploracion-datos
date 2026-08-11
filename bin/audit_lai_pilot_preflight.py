#!/usr/bin/env python3
"""Audit whether the chr22 LAI pilot is identifiable before any simulation.

This program is deliberately fail-closed.  It reads frozen artifacts, emits
aggregate diagnostics, and never simulates mosaics, trains a model, imputes a
marker, or writes individual identifiers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


MISSING_DOSAGE = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnomix-reference-vcf", required=True, type=Path)
    parser.add_argument("--external-panel-vcf", required=True, type=Path)
    parser.add_argument("--gnomix-model", required=True, type=Path)
    parser.add_argument("--gnomix-config", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--top95-nam", required=True, type=Path)
    parser.add_argument("--nam-unrelated-keep", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_contig(value: str) -> str:
    return value.removeprefix("chr")


def canonical_sample_id(value: str) -> str:
    if value.startswith("PSI"):
        suffix = value[3:]
        if suffix.isdigit():
            return f"PSI{int(suffix)}"
    return value


def baseline_sample_id(value: str) -> tuple[str, str]:
    if "_" not in value:
        return "UNKNOWN", value
    ancestry, sample = value.split("_", 1)
    return ancestry, sample


def panel_sample_id(value: str) -> str:
    parts = value.split("_")
    return parts[0] if len(parts) >= 2 and parts[0] == parts[1] else value


def gt_dosage(sample_field: str) -> int:
    gt = sample_field.split(":", 1)[0]
    alleles = gt.replace("|", "/").split("/")
    if len(alleles) != 2 or any(allele == "." for allele in alleles):
        return MISSING_DOSAGE
    try:
        values = [int(allele) for allele in alleles]
    except ValueError:
        return MISSING_DOSAGE
    if any(value not in (0, 1) for value in values):
        return MISSING_DOSAGE
    return sum(values)


def info_float(info: str, key: str) -> float | None:
    prefix = f"{key}="
    for field in info.split(";"):
        if field.startswith(prefix):
            raw = field[len(prefix):].split(",", 1)[0]
            try:
                return float(raw)
            except ValueError:
                return None
    return None


@dataclass
class VcfAudit:
    samples: list[str]
    markers: list[tuple[str, int, str, str]]
    n_records: int
    n_biallelic_snv: int
    n_duplicates: int
    ordered: bool
    contigs: list[str]
    n_global_maf_below_threshold: int
    fingerprints: dict[str, dict[tuple[str, int, str, str], int]]


def audit_vcf(
    path: Path,
    fingerprint_samples: Iterable[str] = (),
    maf_threshold: float = 0.01,
) -> VcfAudit:
    requested = set(fingerprint_samples)
    samples: list[str] = []
    sample_indices: dict[str, int] = {}
    fingerprints = {sample: {} for sample in requested}
    markers: list[tuple[str, int, str, str]] = []
    marker_set: set[tuple[str, int, str, str]] = set()
    contigs: set[str] = set()
    n_records = n_biallelic = n_duplicates = n_rare = 0
    ordered = True
    previous: tuple[str, int] | None = None

    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                index_by_sample = {sample: index + 9 for index, sample in enumerate(samples)}
                missing = requested - set(index_by_sample)
                if missing:
                    raise SystemExit(f"Fingerprint samples absent from {path.name}: {len(missing)}")
                sample_indices = {sample: index_by_sample[sample] for sample in requested}
                continue
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise SystemExit(f"Malformed VCF record in {path.name}")
            chrom = canonical_contig(fields[0])
            try:
                pos = int(fields[1])
            except ValueError as exc:
                raise SystemExit(f"Invalid VCF position in {path.name}") from exc
            ref, alt = fields[3].upper(), fields[4].upper()
            key = (chrom, pos, ref, alt)
            n_records += 1
            contigs.add(chrom)
            if len(ref) == 1 and len(alt) == 1 and "," not in alt and ref in "ACGT" and alt in "ACGT":
                n_biallelic += 1
            if key in marker_set:
                n_duplicates += 1
            else:
                marker_set.add(key)
                markers.append(key)
            current = (chrom, pos)
            if previous is not None and current < previous:
                ordered = False
            previous = current
            af = info_float(fields[7], "AF")
            if af is not None and min(af, 1.0 - af) < maf_threshold:
                n_rare += 1
            for sample, index in sample_indices.items():
                if index >= len(fields):
                    raise SystemExit(f"Missing sample field in {path.name}")
                fingerprints[sample][key] = gt_dosage(fields[index])

    if not samples:
        raise SystemExit(f"Missing #CHROM header in {path.name}")
    return VcfAudit(
        samples=samples,
        markers=markers,
        n_records=n_records,
        n_biallelic_snv=n_biallelic,
        n_duplicates=n_duplicates,
        ordered=ordered,
        contigs=sorted(contigs),
        n_global_maf_below_threshold=n_rare,
        fingerprints=fingerprints,
    )


def read_vcf_samples(path: Path) -> list[str]:
    """Read only the VCF header, avoiding a full first pass over large panels."""
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
    raise SystemExit(f"Missing #CHROM header in {path.name}")


def parse_model_config(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t", 1)
            if len(fields) == 2:
                result[fields[0]] = fields[1]
    return result


def audit_genetic_map(path: Path, expected_chrom: str) -> dict:
    n_rows = 0
    positions: list[int] = []
    cms: list[float] = []
    observed_contigs: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise SystemExit("Genetic map must have chromosome, bp and cM")
            try:
                position = int(float(fields[1]))
                cm = float(fields[2])
            except ValueError as exc:
                raise SystemExit("Invalid numeric value in genetic map") from exc
            observed_contigs.add(canonical_contig(fields[0]))
            positions.append(position)
            cms.append(cm)
            n_rows += 1
    bp_monotonic = all(left < right for left, right in zip(positions, positions[1:]))
    cm_monotonic = all(left <= right for left, right in zip(cms, cms[1:]))
    return {
        "n_rows": n_rows,
        "contigs": sorted(observed_contigs),
        "expected_contig_only": observed_contigs == {expected_chrom},
        "bp_strictly_increasing": bp_monotonic,
        "cm_nondecreasing": cm_monotonic,
        "first_bp": positions[0] if positions else None,
        "last_bp": positions[-1] if positions else None,
        "first_cm": cms[0] if cms else None,
        "last_cm": cms[-1] if cms else None,
    }


def fingerprint_concordance(
    left: dict[tuple[str, int, str, str], int],
    right: dict[tuple[str, int, str, str], int],
) -> dict:
    shared = set(left) & set(right)
    called = [key for key in shared if left[key] != MISSING_DOSAGE and right[key] != MISSING_DOSAGE]
    matches = sum(left[key] == right[key] for key in called)
    return {
        "n_exact_markers": len(shared),
        "n_jointly_called": len(called),
        "dosage_concordance": matches / len(called) if called else None,
    }


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def is_true(value: str | None) -> bool:
    return str(value).strip().upper() == "TRUE"


def ancestry_candidate_counts(rows: list[dict[str, str]], excluded_ids: set[str]) -> dict:
    result = {}
    for ancestry in ("African", "European", "Native_American"):
        selected = [row for row in rows if row.get("Ancestry") == ancestry]
        retained = [
            row for row in selected
            if not is_true(row.get("Exclude")) and row.get("IID") not in excluded_ids
        ]
        result[ancestry] = {
            "metadata_total": len(selected),
            "not_excluded_after_baseline_id_removal": len(retained),
            "maximum_unrelated_dataset": sum(is_true(row.get("Maximum_unrelated_dataset")) for row in retained),
            "maximum_unrelated_dataset_2nd": sum(is_true(row.get("Maximum_unrelated_dataset_2nd")) for row in retained),
        }
    return result


def read_top95(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        return {canonical_sample_id(row[0]) for row in reader if row}


def read_keep(path: Path) -> set[str]:
    result = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if fields:
                result.add(canonical_sample_id(fields[-1]))
    return result


def suppressed_population_counts(rows: list[dict[str, str]], minimum: int) -> list[dict]:
    counts = Counter((row.get("Ancestry", ""), row.get("Population", ""), row.get("Source", "")) for row in rows)
    result = []
    suppressed = 0
    for (ancestry, population, source), count in sorted(counts.items()):
        if count < minimum:
            suppressed += count
        else:
            result.append({"ancestry": ancestry, "population": population, "source": source, "n": count})
    if suppressed:
        result.append({"ancestry": "SUPPRESSED", "population": "SUPPRESSED_LT_N", "source": "SUPPRESSED", "n": suppressed})
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gate_table(path: Path, gates: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "name", "status", "reason"), delimiter="\t")
        writer.writeheader()
        writer.writerows(gates)


def run(args: argparse.Namespace) -> dict:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27_LAI_PILOT_PREFLIGHT":
        raise SystemExit("Invalid M27 preregistration stage")
    contract = prereg["frozen_contract"]
    expected_chrom = str(contract["chromosome"])
    maf_threshold = float(contract["rare_maf_threshold"])

    identity_candidates = prereg.get("identity_candidates", {})
    # Resolve the configured candidate names after reading VCF headers once.
    baseline_samples = read_vcf_samples(args.gnomix_reference_vcf)
    external_samples = read_vcf_samples(args.external_panel_vcf)
    baseline_by_id = {baseline_sample_id(sample)[1]: sample for sample in baseline_samples}
    external_by_id = {panel_sample_id(sample): sample for sample in external_samples}
    baseline_fp_names = [baseline_by_id[left] for left in identity_candidates if left in baseline_by_id]
    external_fp_names = [external_by_id[right] for right in identity_candidates.values() if right in external_by_id]
    baseline = audit_vcf(
        args.gnomix_reference_vcf,
        fingerprint_samples=baseline_fp_names,
        maf_threshold=maf_threshold,
    )
    external = audit_vcf(
        args.external_panel_vcf,
        fingerprint_samples=external_fp_names,
        maf_threshold=maf_threshold,
    )

    config = parse_model_config(args.gnomix_config)
    genetic_map = audit_genetic_map(args.genetic_map, expected_chrom)
    baseline_markers = set(baseline.markers)
    external_markers = set(external.markers)
    exact_overlap = baseline_markers & external_markers
    baseline_positions = {(chrom, pos) for chrom, pos, _ref, _alt in baseline_markers}
    external_positions = {(chrom, pos) for chrom, pos, _ref, _alt in external_markers}
    shared_positions = baseline_positions & external_positions
    overlap_fraction = len(exact_overlap) / len(baseline_markers) if baseline_markers else 0.0
    minimum_fraction = float(contract["minimum_pretrained_model_marker_fraction"])

    c_matches = config.get("C", "").isdigit() and int(config["C"]) == len(baseline_markers)
    g0_pass = all(
        (
            baseline.contigs == [expected_chrom],
            external.contigs == [expected_chrom],
            baseline.n_biallelic_snv == baseline.n_records,
            external.n_biallelic_snv == external.n_records,
            baseline.n_duplicates == 0,
            external.n_duplicates == 0,
            baseline.ordered,
            external.ordered,
            c_matches,
            genetic_map["expected_contig_only"],
            genetic_map["bp_strictly_increasing"],
            genetic_map["cm_nondecreasing"],
        )
    )
    g0 = {
        "gate": "G0",
        "status": "PASS" if g0_pass else "FAIL",
        "build_declared": contract["build"],
        "chromosome": expected_chrom,
        "model_config": config,
        "model_c_matches_reference_marker_count": c_matches,
        "baseline_vcf": {
            "n_samples": len(baseline.samples),
            "n_records": baseline.n_records,
            "n_unique_markers": len(baseline_markers),
            "n_biallelic_snv": baseline.n_biallelic_snv,
            "n_duplicate_keys": baseline.n_duplicates,
            "ordered": baseline.ordered,
            "contigs": baseline.contigs,
        },
        "external_vcf": {
            "n_samples": len(external.samples),
            "n_records": external.n_records,
            "n_unique_markers": len(external_markers),
            "n_biallelic_snv": external.n_biallelic_snv,
            "n_duplicate_keys": external.n_duplicates,
            "ordered": external.ordered,
            "contigs": external.contigs,
            "n_sites_global_maf_below_threshold_diagnostic_only": external.n_global_maf_below_threshold,
        },
        "genetic_map": genetic_map,
        "input_sha256": {
            "gnomix_reference_vcf": sha256(args.gnomix_reference_vcf),
            "external_panel_vcf": sha256(args.external_panel_vcf),
            "gnomix_model": sha256(args.gnomix_model),
            "gnomix_config": sha256(args.gnomix_config),
            "genetic_map": sha256(args.genetic_map),
            "metadata": sha256(args.metadata),
            "top95_nam": sha256(args.top95_nam),
            "nam_unrelated_keep": sha256(args.nam_unrelated_keep),
            "preregistration": sha256(args.preregistration),
        },
    }

    baseline_ids = {baseline_sample_id(sample)[1] for sample in baseline.samples}
    external_ids = {panel_sample_id(sample) for sample in external.samples}
    exact_donor_overlap = baseline_ids & external_ids
    fingerprints = []
    fingerprint_pass = True
    for left_id, right_id in identity_candidates.items():
        left_name = baseline_by_id.get(left_id)
        right_name = external_by_id.get(right_id)
        if not left_name or not right_name:
            result = {"candidate_pair": "REDACTED", "status": "MISSING_CANDIDATE"}
            fingerprint_pass = False
        else:
            metrics = fingerprint_concordance(
                baseline.fingerprints[left_name], external.fingerprints[right_name]
            )
            concordance = metrics["dosage_concordance"]
            status = "MATCH" if concordance is not None and concordance >= 0.99 else "MISMATCH"
            fingerprint_pass &= status == "MATCH"
            result = {"candidate_pair": "REDACTED", "status": status, **metrics}
        fingerprints.append(result)

    metadata = read_metadata(args.metadata)
    candidate_counts = ancestry_candidate_counts(metadata, baseline_ids | set(identity_candidates.values()))
    top95 = read_top95(args.top95_nam)
    nam_keep = read_keep(args.nam_unrelated_keep)
    metadata_by_canonical = {canonical_sample_id(row.get("IID", "")): row for row in metadata}
    top95_in_metadata = [metadata_by_canonical[sample] for sample in top95 if sample in metadata_by_canonical]
    top95_unrelated_1 = sum(is_true(row.get("Maximum_unrelated_dataset")) for row in top95_in_metadata)
    top95_unrelated_2 = sum(is_true(row.get("Maximum_unrelated_dataset_2nd")) for row in top95_in_metadata)
    baseline_ancestry_counts = Counter(baseline_sample_id(sample)[0] for sample in baseline.samples)
    # The current metadata supplies two maximum-unrelated selections, not family component IDs,
    # and global ancestry is not locus-specific truth. G1 therefore remains unresolved.
    g1 = {
        "gate": "G1",
        "status": "FEASIBILITY_UNRESOLVED",
        "n_baseline_donors": len(baseline_ids),
        "n_exact_baseline_donor_ids_in_external_panel": len(exact_donor_overlap),
        "n_unresolved_baseline_ids_before_fingerprint": len(baseline_ids - external_ids),
        "configured_ambiguous_identity_fingerprints": fingerprints,
        "all_configured_fingerprints_match": fingerprint_pass,
        "baseline_ancestry_counts": dict(sorted(baseline_ancestry_counts.items())),
        "candidate_counts_after_baseline_id_removal": candidate_counts,
        "top95_nam": {
            "n_listed": len(top95),
            "n_joined_to_metadata": len(top95_in_metadata),
            "n_in_maximum_unrelated_dataset": top95_unrelated_1,
            "n_in_maximum_unrelated_dataset_2nd": top95_unrelated_2,
            "n_in_unrelated_unadmixed_keep": len(top95 & nam_keep),
        },
        "population_counts_suppressed": suppressed_population_counts(
            [row for row in metadata if row.get("Ancestry") in {"African", "European", "Native_American"}],
            int(prereg["privacy"]["small_cell_suppression_n"]),
        ),
        "sample_ids_emitted": False,
        "reason": "No family-component assignments or independent locus-specific parental truth are present in the supplied metadata.",
    }

    g2_pass = g0_pass and overlap_fraction >= minimum_fraction
    g2 = {
        "gate": "G2",
        "status": "PASS" if g2_pass else "FAIL",
        "n_model_markers": len(baseline_markers),
        "n_external_markers": len(external_markers),
        "n_exact_chrom_pos_ref_alt_overlap": len(exact_overlap),
        "n_shared_chrom_pos": len(shared_positions),
        "n_shared_positions_with_ref_alt_discordance": len(shared_positions) - len(exact_overlap),
        "exact_model_marker_fraction": overlap_fraction,
        "minimum_required_fraction": minimum_fraction,
        "missing_model_marker_count": len(baseline_markers - external_markers),
        "extra_external_marker_count": len(external_markers - baseline_markers),
        "imputation_or_padding_performed": False,
        "source": contract["minimum_marker_fraction_source"],
        "reason": (
            "Frozen baseline marker coverage meets the preregistered contract."
            if g2_pass
            else "External panel is below the preregistered Gnomix marker-coverage floor; simulation and inference remain blocked."
        ),
    }

    if not g0_pass:
        decision = "STOP_INPUT_CONTRACT"
        stop_reasons = ["G0_FAIL"]
    elif not g2_pass:
        decision = "STOP_BASELINE_NOT_EXECUTABLE_FOR_SCIENTIFIC_PILOT"
        stop_reasons = ["G2_MARKER_COVERAGE_BELOW_FLOOR"]
        if g1["status"] != "PASS":
            stop_reasons.append("G1_PARENTAL_INDEPENDENCE_UNRESOLVED")
    elif g1["status"] != "PASS":
        decision = "STOP_PARENTALS_NOT_IDENTIFIED"
        stop_reasons = ["G1_PARENTAL_INDEPENDENCE_UNRESOLVED"]
    else:
        decision = "FEASIBILITY_UNRESOLVED"
        stop_reasons = ["G3_AND_G4_REQUIRE_SEPARATE_WGS_INPUTS"]

    skipped_reason = "Not run because upstream G0-G2 did not authorize rare-WGS or power analysis."
    g3 = {"gate": "G3", "status": "SKIPPED_FAIL_CLOSED", "reason": skipped_reason}
    g4 = {"gate": "G4", "status": "SKIPPED_FAIL_CLOSED", "reason": skipped_reason}
    summary = {
        "stage": prereg["stage"],
        "scope": prereg["scope"],
        "decision": decision,
        "stop_reasons": stop_reasons,
        "simulation_performed": False,
        "model_training_performed": False,
        "gnomix_inference_performed": False,
        "test_opened": False,
        "sample_ids_emitted": False,
        "gates": {gate["gate"]: gate["status"] for gate in (g0, g1, g2, g3, g4)},
        "interpretation": (
            "This decision concerns the frozen Gnomix baseline and supplied external panel only; "
            "it does not refute rare WGS information as a general LAI signal."
        ),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "g0_model_contract.json", g0)
    write_json(args.outdir / "g1_donor_identity_and_parentals.json", g1)
    write_json(args.outdir / "g2_marker_compatibility.json", g2)
    write_json(args.outdir / "g3_wgs_rare_support.json", g3)
    write_json(args.outdir / "g4_identifiability_power.json", g4)
    write_json(args.outdir / "m27_lai_pilot_preflight_summary.json", summary)
    write_gate_table(
        args.outdir / "m27_preflight_gates.tsv",
        [
            {"gate": "G0", "name": "artifact_and_coordinate_contract", "status": g0["status"], "reason": "Frozen input integrity and coordinate checks."},
            {"gate": "G1", "name": "donor_identity_independence_and_parentals", "status": g1["status"], "reason": g1["reason"]},
            {"gate": "G2", "name": "frozen_baseline_marker_compatibility", "status": g2["status"], "reason": g2["reason"]},
            {"gate": "G3", "name": "fit_only_wgs_rare_support_and_phase", "status": g3["status"], "reason": g3["reason"]},
            {"gate": "G4", "name": "identifiability_and_power_envelope", "status": g4["status"], "reason": g4["reason"]},
        ],
    )
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
