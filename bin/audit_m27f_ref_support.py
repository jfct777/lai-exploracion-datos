#!/usr/bin/env python3
"""Reproduce the frozen 954-site catalog in DISCOVERY and audit support in REF only."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import audit_m27e_ibd_rare_transfer as m27e
from audit_rare_scaffold_bridge import MISSING_DOSAGE, parse_gt, parse_record, read_vcf_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--raw-wgs-vcf", type=Path, required=True)
    parser.add_argument("--discovery-bcf", type=Path, required=True)
    parser.add_argument("--ref-bcf", type=Path, required=True)
    parser.add_argument("--split-private", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--projection-public", type=Path, required=True)
    parser.add_argument("--m27e-manifest", type=Path, required=True)
    parser.add_argument("--m27e-support", type=Path, required=True)
    parser.add_argument("--m27e-preregistration", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_variant_text(path: Path, bcftools: str) -> Iterator[TextIO]:
    if path.suffix == ".bcf":
        process = subprocess.Popen(
            [bcftools, "view", "--output-type", "v", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError(f"Could not open {path.name}")
        try:
            yield process.stdout
        finally:
            process.stdout.close()
            stderr = process.stderr.read()
            process.stderr.close()
            if process.wait() != 0:
                raise RuntimeError(f"bcftools view failed for {path.name}: {stderr[:1000]}")
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield handle


def read_bcf_samples(path: Path, bcftools: str) -> list[str]:
    completed = subprocess.run(
        [bcftools, "query", "-l", str(path)], check=True, text=True, capture_output=True
    )
    return completed.stdout.splitlines()


def direct_catalog(
    raw_vcf: Path,
    discovery_bcf: Path,
    bcftools: str,
    minimum_mac: int,
    maf_threshold: float,
) -> dict[tuple[str, int, str, str], bool]:
    raw_samples = read_vcf_samples(raw_vcf)
    discovery_samples = read_bcf_samples(discovery_bcf, bcftools)
    if len(raw_samples) != 128 or set(raw_samples) != set(discovery_samples):
        raise ValueError("STOP_TARGET_REPRODUCTION: DISCOVERY projection differs from NatWGS-128")
    discovery_index = {sample: index for index, sample in enumerate(discovery_samples)}
    raw_indices = [discovery_index[sample] for sample in raw_samples]
    raw_sites = m27e.raw_rare_sites(raw_vcf, "22", len(raw_samples), minimum_mac, maf_threshold)
    catalog: dict[tuple[str, int, str, str], bool] = {}
    with open_variant_text(discovery_bcf, bcftools) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, discovery_bcf)
            raw_site = raw_sites.get(key)
            if raw_site is None:
                continue
            parsed = [parse_gt(value) for value in fields[9:]]
            if len(parsed) != len(discovery_samples):
                raise ValueError("Unexpected DISCOVERY genotype count")
            direct = False
            concordant = True
            for raw_offset, discovery_offset in enumerate(raw_indices):
                raw_alt = raw_site.alt_dosages[raw_offset]
                minor = (
                    raw_alt
                    if raw_site.minor_is_alt
                    else MISSING_DOSAGE if raw_alt == MISSING_DOSAGE else 2 - raw_alt
                )
                if minor in (0, MISSING_DOSAGE):
                    continue
                panel_alt, phased, called = parsed[discovery_offset]
                if not called or panel_alt != raw_alt or (raw_alt == 1 and not phased):
                    concordant = False
                    break
                direct = direct or raw_alt == 1
            if concordant and direct:
                catalog[key] = raw_site.minor_is_alt
    return catalog


def catalog_digest(catalog: dict[tuple[str, int, str, str], bool]) -> str:
    lines = [
        f"{chrom}:{position}:{ref}:{alt}|minor={'ALT' if catalog[key] else 'REF'}"
        for key in catalog
        for chrom, position, ref, alt in [key]
    ]
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode("utf-8")).hexdigest()


def usable_carrier(alt_dosage: int, phased: bool, called: bool, minor_is_alt: bool) -> bool:
    if not called:
        return False
    minor_dosage = alt_dosage if minor_is_alt else 2 - alt_dosage
    return minor_dosage > 0 and (alt_dosage != 1 or phased)


def maximum_gap(positions: list[int]) -> int | None:
    ordered = sorted(set(positions))
    return max((right - left for left, right in zip(ordered, ordered[1:])), default=None)


def write_private_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write a small private table deterministically (no gzip timestamp)."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_REF_SUPPORT_AUDIT" or prereg.get("version") != 1:
        raise ValueError("Invalid M27F REF preregistration")
    contract = prereg["upstream_contract"]
    observed_hashes = {
        "m27f_split_manifest_sha256": sha256_file(args.split_manifest),
        "m27f_split_private_sha256": sha256_file(args.split_private),
        "m27e_manifest_sha256": sha256_file(args.m27e_manifest),
        "m27e_support_sha256": sha256_file(args.m27e_support),
        "raw_wgs_vcf_sha256": sha256_file(args.raw_wgs_vcf),
    }
    if any(observed_hashes[key] != contract[key] for key in observed_hashes):
        raise ValueError("STOP_PROVENANCE: an upstream hash differs")
    projection = json.loads(args.projection_public.read_text(encoding="utf-8"))
    if projection.get("decision") != "GO_REF_SUPPORT_AUDIT":
        raise ValueError("STOP_VARIANT_PROJECTION: projection gate did not pass")
    if (
        projection.get("ref_bcf_sha256") != sha256_file(args.ref_bcf)
        or projection.get("discovery_bcf_sha256") != sha256_file(args.discovery_bcf)
    ):
        raise ValueError("STOP_VARIANT_PROJECTION: projected BCF hash differs")

    m27e_prereg = json.loads(args.m27e_preregistration.read_text(encoding="utf-8"))
    rare_contract = m27e_prereg["rare_contract"]
    catalog = direct_catalog(
        args.raw_wgs_vcf,
        args.discovery_bcf,
        args.bcftools,
        int(rare_contract["minimum_mac"]),
        float(rare_contract["maf_threshold"]),
    )
    observed_catalog_digest = catalog_digest(catalog)
    if (
        len(catalog) != int(contract["expected_direct_phase_bridge_sites"])
        or observed_catalog_digest != contract["expected_direct_bridge_key_orientation_sha256"]
    ):
        raise ValueError("STOP_TARGET_REPRODUCTION: the frozen 954-site digest differs")

    with args.split_private.open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle, delimiter="\t"))
    ref_rows = [row for row in split_rows if row["role"] == "REF_TRAIN"]
    ref_samples = [row["sample_id"] for row in ref_rows]
    if read_bcf_samples(args.ref_bcf, args.bcftools) != ref_samples:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: REF BCF header differs from frozen split")
    metadata = {row["sample_id"]: row for row in ref_rows}
    ancestry_indices = {
        ancestry: [index for index, sample in enumerate(ref_samples) if metadata[sample]["ancestry"] == ancestry]
        for ancestry in ("African", "European", "Native_American")
    }
    nam_blocks = {metadata[sample]["atomic_unit_id"] for sample in ref_samples if metadata[sample]["ancestry"] == "Native_American"}
    if len(nam_blocks) != int(contract["expected_ref_native_american_atomic_units"]):
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: unexpected NAM REF atomic-unit count")

    site_rows: list[dict[str, object]] = []
    found: set[tuple[str, int, str, str]] = set()
    carrier_patterns: set[str] = set()
    unphased_het_excluded = Counter()
    with open_variant_text(args.ref_bcf, args.bcftools) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, args.ref_bcf)
            if key not in catalog:
                continue
            if key in found:
                raise ValueError("STOP_TARGET_REPRODUCTION: duplicate target key in REF")
            found.add(key)
            parsed = [parse_gt(value) for value in fields[9:]]
            if len(parsed) != len(ref_samples):
                raise ValueError("Unexpected REF genotype count")
            minor_is_alt = catalog[key]
            row: dict[str, object] = {
                "chrom": key[0],
                "pos": key[1],
                "ref": key[2],
                "alt": key[3],
                "minor_allele": key[3] if minor_is_alt else key[2],
                "minor_is_alt": minor_is_alt,
            }
            pattern = bytearray()
            for alt_dosage, phased, called in parsed:
                pattern.append(255 if not called else 1 if usable_carrier(alt_dosage, phased, called, minor_is_alt) else 0)
            pattern_digest = hashlib.sha256(bytes(pattern)).hexdigest()
            carrier_patterns.add(pattern_digest)
            row["ref_carrier_pattern_sha256"] = pattern_digest
            for ancestry, indices in ancestry_indices.items():
                called_indices = [index for index in indices if parsed[index][2]]
                carrier_indices = [
                    index
                    for index in indices
                    if usable_carrier(*parsed[index], minor_is_alt)
                ]
                unphased_het_excluded[ancestry] += sum(
                    parsed[index][2] and parsed[index][0] == 1 and not parsed[index][1]
                    for index in indices
                )
                callable_blocks = {metadata[ref_samples[index]]["atomic_unit_id"] for index in called_indices}
                carrier_blocks = {metadata[ref_samples[index]]["atomic_unit_id"] for index in carrier_indices}
                prefix = {"African": "afr", "European": "eur", "Native_American": "nam"}[ancestry]
                row[f"n_{prefix}_called_samples"] = len(called_indices)
                row[f"n_{prefix}_callable_atomic_units"] = len(callable_blocks)
                row[f"n_{prefix}_carrier_samples"] = len(carrier_indices)
                row[f"n_{prefix}_carrier_atomic_units"] = len(carrier_blocks)
            row["nam_supports_both_ref_atomic_units"] = row["n_nam_carrier_atomic_units"] == len(nam_blocks)
            site_rows.append(row)

    if found != set(catalog):
        raise ValueError("STOP_TARGET_REPRODUCTION: REF does not contain all 954 exact keys")
    site_rows.sort(key=lambda row: (int(str(row["chrom"]).removeprefix("chr")), int(row["pos"]), str(row["ref"]), str(row["alt"])))
    args.outdir.mkdir(parents=True, exist_ok=True)
    supported = [row for row in site_rows if row["nam_supports_both_ref_atomic_units"]]
    private_path = args.outdir / "m27f_ref_site_support.private.tsv"
    eligible_path = args.outdir / "m27f_ref_eligible_sites.private.tsv"
    private_fields = list(site_rows[0])
    write_private_tsv(private_path, site_rows, private_fields)
    write_private_tsv(eligible_path, supported, private_fields)
    eligible_catalog = {
        (str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"])):
        bool(row["minor_is_alt"])
        for row in supported
    }
    supported_positions = [int(row["pos"]) for row in supported]
    r5 = bool(supported)
    decision = "GO_VALID_FEASIBILITY_ONLY" if r5 else "STOP_REF_NO_SUPPORT"
    public = {
        "stage": prereg["stage"],
        "decision": decision,
        "gates": {"R0": "PASS", "R1": "PASS", "R2": "PASS", "R3": "PASS", "R4": "PASS", "R5": "PASS" if r5 else "FAIL"},
        "n_frozen_target_sites": len(catalog),
        "target_key_orientation_sha256": observed_catalog_digest,
        "n_ref_samples": len(ref_samples),
        "n_ref_native_american_atomic_units": len(nam_blocks),
        "n_sites_with_any_native_american_ref_carrier": sum(int(row["n_nam_carrier_atomic_units"]) > 0 for row in site_rows),
        "n_sites_with_carriers_in_both_native_american_ref_atomic_units": len(supported),
        "n_unique_ref_carrier_patterns": len(carrier_patterns),
        "supported_position_summary": {
            "n_distinct_positions": len(set(supported_positions)),
            "minimum_bp": min(supported_positions, default=None),
            "maximum_bp": max(supported_positions, default=None),
            "maximum_gap_bp": maximum_gap(supported_positions),
        },
        "unphased_heterozygous_carriers_excluded_by_ancestry": dict(unphased_het_excluded),
        "private_site_support_sha256": sha256_file(private_path),
        "private_eligible_catalog_sha256": sha256_file(eligible_path),
        "eligible_key_orientation_sha256": catalog_digest(eligible_catalog),
        "source_valid_opened": False,
        "source_test_opened": False,
        "lai_performed": False,
        "simulation_performed": False,
        "model_training_performed": False,
        "sample_ids_emitted_publicly": False,
        "population_labels_emitted_publicly": False,
        "interpretation": "REF-only feasibility gate. Sites and orientations remain frozen from NatWGS-128; individuals are nested inside atomic units and are not independent replicates.",
    }
    (args.outdir / "m27f_ref_support.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
