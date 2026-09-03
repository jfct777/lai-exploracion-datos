#!/usr/bin/env python3
"""Reproduce 954 discovery-frozen sites and select support using REF_TRAIN only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import audit_m27e_ibd_rare_transfer as m27e
from audit_rare_scaffold_bridge import (
    MISSING_DOSAGE,
    audit_marker_panel,
    parse_gt,
    parse_record,
    read_vcf_samples,
)


ANCESTRY_PREFIX = {
    "African": "afr",
    "European": "eur",
    "Native_American": "nam",
}
VariantKey = tuple[str, int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--raw-wgs-vcf", type=Path, required=True)
    parser.add_argument("--baseline-vcf", type=Path, required=True)
    parser.add_argument("--discovery-bcf", type=Path, required=True)
    parser.add_argument("--ref-bcf", type=Path, required=True)
    parser.add_argument("--split-private", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--projection-public", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
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
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield handle


def read_bcf_samples(path: Path, bcftools: str) -> list[str]:
    completed = subprocess.run(
        [bcftools, "query", "-l", str(path)],
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.splitlines()


def direct_catalog(
    raw_vcf: Path,
    discovery_bcf: Path,
    bcftools: str,
    minimum_mac: int,
    maf_threshold: float,
) -> dict[VariantKey, bool]:
    raw_samples = read_vcf_samples(raw_vcf)
    discovery_samples = read_bcf_samples(discovery_bcf, bcftools)
    if len(raw_samples) != 128 or set(raw_samples) != set(discovery_samples):
        raise ValueError("STOP_TARGET_REPRODUCTION: DISCOVERY differs from NatWGS-128")
    discovery_index = {sample: index for index, sample in enumerate(discovery_samples)}
    raw_indices = [discovery_index[sample] for sample in raw_samples]
    raw_sites = m27e.raw_rare_sites(
        raw_vcf, "22", len(raw_samples), minimum_mac, maf_threshold
    )
    catalog: dict[VariantKey, bool] = {}
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


def catalog_digest(catalog: dict[VariantKey, bool]) -> str:
    lines = [
        f"{chrom}:{position}:{ref}:{alt}|minor={'ALT' if catalog[key] else 'REF'}"
        for key in catalog
        for chrom, position, ref, alt in [key]
    ]
    payload = "\n".join(sorted(lines))
    if lines:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def minor_dosage(alt_dosage: int, called: bool, minor_is_alt: bool) -> int | None:
    if not called:
        return None
    return alt_dosage if minor_is_alt else 2 - alt_dosage


def usable_carrier(
    alt_dosage: int, phased: bool, called: bool, minor_is_alt: bool
) -> bool:
    dosage = minor_dosage(alt_dosage, called, minor_is_alt)
    return dosage is not None and dosage > 0 and (alt_dosage != 1 or phased)


def role_metrics(
    parsed: list[tuple[int, bool, bool]],
    indices: list[int],
    samples: list[str],
    metadata: dict[str, dict[str, str]],
    minor_is_alt: bool,
) -> dict[str, object]:
    called = [index for index in indices if parsed[index][2]]
    carriers = [
        index for index in indices if usable_carrier(*parsed[index], minor_is_alt)
    ]
    unit_indices: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        unit_indices[metadata[samples[index]]["atomic_unit_id"]].append(index)
    fully_callable_units = sorted(
        unit
        for unit, members in unit_indices.items()
        if all(parsed[index][2] for index in members)
    )
    carrier_units = sorted(
        {metadata[samples[index]]["atomic_unit_id"] for index in carriers}
    )
    units_with_missing = {
        unit
        for unit, members in unit_indices.items()
        if any(not parsed[index][2] for index in members)
    }
    unresolved_units = sorted(units_with_missing - set(carrier_units))
    carrier_units_upper_bound = sorted(set(carrier_units) | units_with_missing)
    dosages = [
        minor_dosage(parsed[index][0], parsed[index][2], minor_is_alt)
        for index in called
    ]
    minor_ac = sum(value for value in dosages if value is not None)
    minor_an = 2 * len(called)
    return {
        "called_samples": len(called),
        "fully_callable_atomic_units": len(fully_callable_units),
        "carrier_samples": len(carriers),
        "carrier_atomic_units": len(carrier_units),
        "carrier_atomic_units_upper_bound": len(carrier_units_upper_bound),
        "unresolved_noncarrier_atomic_units": len(unresolved_units),
        "minor_ac": minor_ac,
        "minor_an": minor_an,
        "minor_af": minor_ac / minor_an if minor_an else None,
        "fully_callable_unit_ids": ";".join(fully_callable_units),
        "carrier_unit_ids": ";".join(carrier_units),
        "unresolved_noncarrier_unit_ids": ";".join(unresolved_units),
        "unphased_het_carriers_excluded": sum(
            parsed[index][2]
            and parsed[index][0] == 1
            and not parsed[index][1]
            and (minor_dosage(parsed[index][0], True, minor_is_alt) or 0) > 0
            for index in indices
        ),
    }


def write_private_tsv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def threshold_summary(
    rows: list[dict[str, object]], thresholds: list[int]
) -> dict[str, dict[str, int]]:
    result = {}
    for threshold in thresholds:
        selected = [
            row
            for row in rows
            if int(row["ref_nam_carrier_atomic_units"]) >= threshold
        ]
        result[str(threshold)] = {
            "all": len(selected),
            "in_frozen_baseline": sum(
                str(row["in_frozen_baseline"]) == "True" for row in selected
            ),
            "outside_frozen_baseline": sum(
                str(row["in_frozen_baseline"]) != "True" for row in selected
            ),
        }
    return result


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    clean = sorted(value for value in values if value is not None)
    return {
        "minimum": min(clean, default=None),
        "median": statistics.median(clean) if clean else None,
        "maximum": max(clean, default=None),
    }


def classify_ref_decision(
    n_primary: int,
    n_unresolved_primary: int,
) -> str:
    if not n_primary:
        return (
            "INCONCLUSIVE_REF_CALLABILITY"
            if n_unresolved_primary
            else "STOP_REF_NO_TRANSFERABLE_SUPPORT"
        )
    return "GO_VALID_SUPPORT_AUDIT"


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27F_REF_VALID_SUPPORT_AUDIT" or prereg.get("version") != 2:
        raise ValueError("Invalid M27F-b preregistration")
    contract = prereg["upstream_contract"]
    observed_hashes = {
        "m27f_split_manifest_sha256": sha256_file(args.split_manifest),
        "m27f_split_private_sha256": sha256_file(args.split_private),
        "m27e_manifest_sha256": sha256_file(args.m27e_manifest),
        "m27e_support_sha256": sha256_file(args.m27e_support),
        "raw_wgs_vcf_sha256": sha256_file(args.raw_wgs_vcf),
        "baseline_vcf_sha256": sha256_file(args.baseline_vcf),
    }
    if any(observed_hashes[key] != contract[key] for key in observed_hashes):
        raise ValueError("STOP_PROVENANCE: an upstream hash differs")

    projection = json.loads(args.projection_public.read_text(encoding="utf-8"))
    projection_manifest = json.loads(
        args.projection_manifest.read_text(encoding="utf-8")
    )
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
        or projection_manifest.get("sha256", {}).get(args.projection_public.name)
        != sha256_file(args.projection_public)
    ):
        raise ValueError("STOP_VARIANT_PROJECTION: projection gate did not pass")
    projections = projection["projections"]
    if (
        projections["DISCOVERY_CORE"]["bcf_sha256"] != sha256_file(args.discovery_bcf)
        or projections["REF_TRAIN"]["bcf_sha256"] != sha256_file(args.ref_bcf)
        or projection_manifest.get("sha256", {}).get(args.discovery_bcf.name)
        != projections["DISCOVERY_CORE"]["bcf_sha256"]
        or projection_manifest.get("sha256", {}).get(args.ref_bcf.name)
        != projections["REF_TRAIN"]["bcf_sha256"]
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
        or observed_catalog_digest
        != contract["expected_direct_bridge_key_orientation_sha256"]
    ):
        raise ValueError("STOP_TARGET_REPRODUCTION: frozen 954-site digest differs")

    baseline = audit_marker_panel(args.baseline_vcf, "22")
    baseline_sites = set(baseline.markers)
    n_in_baseline = sum(key in baseline_sites for key in catalog)
    if (
        n_in_baseline != int(contract["expected_direct_sites_in_frozen_baseline"])
        or len(catalog) - n_in_baseline
        != int(contract["expected_direct_sites_outside_frozen_baseline"])
    ):
        raise ValueError("STOP_BASELINE_REPRODUCTION: frozen 507/447 margin differs")

    m27e_support = json.loads(args.m27e_support.read_text(encoding="utf-8"))
    if (
        m27e_support.get("direct_bridge_key_orientation_sha256")
        != observed_catalog_digest
        or int(m27e_support.get("n_direct_phase_bridge_sites", -1)) != len(catalog)
    ):
        raise ValueError("STOP_TARGET_REPRODUCTION: M27E receipt differs")

    with args.split_private.open("r", encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle, delimiter="\t"))
    ref_rows = [row for row in split_rows if row["role"] == "REF_TRAIN"]
    ref_samples = [row["sample_id"] for row in ref_rows]
    if read_bcf_samples(args.ref_bcf, args.bcftools) != ref_samples:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: REF BCF header differs from split")
    metadata = {row["sample_id"]: row for row in ref_rows}
    ancestry_indices = {
        ancestry: [
            index
            for index, sample in enumerate(ref_samples)
            if metadata[sample]["ancestry"] == ancestry
        ]
        for ancestry in ANCESTRY_PREFIX
    }
    observed_by_ancestry = {
        ancestry: len(indices) for ancestry, indices in ancestry_indices.items()
    }
    expected_by_ancestry = {
        key: int(value)
        for key, value in contract["expected_samples_by_ancestry_and_role"][
            "REF_TRAIN"
        ].items()
    }
    if observed_by_ancestry != expected_by_ancestry:
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: REF ancestry counts differ")
    nam_units = {
        metadata[sample]["atomic_unit_id"]
        for sample in ref_samples
        if metadata[sample]["ancestry"] == "Native_American"
    }
    if len(nam_units) != int(contract["expected_ref_native_american_atomic_units"]):
        raise ValueError("STOP_OUTPUT_MEMBERSHIP: unexpected NAM REF unit count")

    rows: list[dict[str, object]] = []
    found: set[VariantKey] = set()
    unphased_excluded = Counter()
    with open_variant_text(args.ref_bcf, args.bcftools) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, args.ref_bcf)
            if key not in catalog:
                continue
            if key in found:
                raise ValueError("STOP_TARGET_REPRODUCTION: duplicate REF target key")
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
                "in_frozen_baseline": key in baseline_sites,
            }
            for ancestry, indices in ancestry_indices.items():
                prefix = ANCESTRY_PREFIX[ancestry]
                metrics = role_metrics(
                    parsed, indices, ref_samples, metadata, minor_is_alt
                )
                unphased_excluded[ancestry] += int(
                    metrics["unphased_het_carriers_excluded"]
                )
                for name, value in metrics.items():
                    row[f"ref_{prefix}_{name}"] = value
            row["primary_ref_selected"] = (
                int(row["ref_nam_carrier_atomic_units"])
                >= int(prereg["support_contract"]["primary_ref_min_atomic_units"])
            )
            rows.append(row)

    if found != set(catalog):
        raise ValueError("STOP_TARGET_REPRODUCTION: REF lacks frozen target keys")
    rows.sort(
        key=lambda row: (
            int(str(row["chrom"]).removeprefix("chr")),
            int(row["pos"]),
            str(row["ref"]),
            str(row["alt"]),
        )
    )
    primary = [row for row in rows if row["primary_ref_selected"]]
    primary_outside = [
        row for row in primary if str(row["in_frozen_baseline"]) != "True"
    ]
    primary_threshold = int(
        prereg["support_contract"]["primary_ref_min_atomic_units"]
    )
    unresolved_primary = [
        row
        for row in rows
        if int(row["ref_nam_carrier_atomic_units"]) < primary_threshold
        and int(row["ref_nam_carrier_atomic_units_upper_bound"])
        >= primary_threshold
    ]
    unresolved_primary_outside = [
        row
        for row in unresolved_primary
        if str(row["in_frozen_baseline"]) != "True"
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    support_path = args.outdir / "m27f_ref_site_support.private.tsv"
    catalog_path = args.outdir / "m27f_ref_primary_catalog.private.tsv"
    fields = list(rows[0])
    write_private_tsv(support_path, rows, fields)
    write_private_tsv(catalog_path, primary, fields)

    decision = classify_ref_decision(
        len(primary),
        len(unresolved_primary),
    )

    public = {
        "stage": "M27F_REF_SUPPORT_SELECTION",
        "decision": decision,
        "gates": {
            "R0": "PASS",
            "R1": "PASS",
            "R2": "PASS",
            "R3": "PASS",
            "R4": "PASS",
            "R5": "PASS" if primary else "WARN" if unresolved_primary else "FAIL",
            "R6": "PASS" if primary_outside else "WARN" if unresolved_primary_outside else "FAIL",
        },
        "n_frozen_target_sites": len(catalog),
        "n_target_sites_in_frozen_baseline": n_in_baseline,
        "n_target_sites_outside_frozen_baseline": len(catalog) - n_in_baseline,
        "target_key_orientation_sha256": observed_catalog_digest,
        "n_ref_samples": len(ref_samples),
        "n_ref_native_american_atomic_units": len(nam_units),
        "primary_ref_min_atomic_units": int(
            prereg["support_contract"]["primary_ref_min_atomic_units"]
        ),
        "ref_support_sensitivity": threshold_summary(
            rows,
            [int(value) for value in prereg["diagnostics"]["ref_support_thresholds"]],
        ),
        "n_sites_whose_ref_support_is_unresolved_by_missingness": len(
            unresolved_primary
        ),
        "n_outside_sites_whose_ref_support_is_unresolved_by_missingness": len(
            unresolved_primary_outside
        ),
        "ref_frozen_allele_frequency_by_ancestry": {
            ancestry: numeric_summary(
                [
                    float(row[f"ref_{prefix}_minor_af"])
                    for row in rows
                    if row[f"ref_{prefix}_minor_af"] not in ("", None)
                ]
            )
            for ancestry, prefix in ANCESTRY_PREFIX.items()
        },
        "unphased_heterozygous_carriers_excluded_by_ancestry": dict(
            unphased_excluded
        ),
        "private_ref_support_sha256": sha256_file(support_path),
        "private_primary_catalog_sha256": sha256_file(catalog_path),
        "primary_catalog_key_orientation_sha256": catalog_digest(
            {
                (
                    str(row["chrom"]),
                    int(row["pos"]),
                    str(row["ref"]),
                    str(row["alt"]),
                ): str(row["minor_is_alt"]) == "True"
                for row in primary
            }
        ),
        "source_valid_bcf_mechanically_projected": True,
        "source_valid_genotypes_analyzed": False,
        "source_test_genotypes_opened": False,
        "m27e_full_panel_counts_used_for_selection": False,
        "lai_performed": False,
        "simulation_performed": False,
        "model_training_performed": False,
        "sample_ids_emitted_publicly": False,
        "population_labels_emitted_publicly": False,
        "interpretation": (
            "REF-only selection. The allele is rare by the frozen NatWGS-128 "
            "definition; its frequency in REF need not remain below one percent."
        ),
    }
    (args.outdir / "m27f_ref_support.public.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return public


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
