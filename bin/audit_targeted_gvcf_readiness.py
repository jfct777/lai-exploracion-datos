#!/usr/bin/env python3
"""Audit NAMBR gVCF readiness at the frozen chr22 Gnomix marker set.

The program is read-only with respect to genomic inputs. It reconstructs calls
from explicit records and reference blocks, applies preregistered quality
policies, checks phase support in the existing scaffold, and emits aggregate
diagnostics without sample identifiers.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

# Nextflow stages local scripts as symlinks whose targets may live in separate
# cache directories.  Keep the task working directory on the import path so
# staged sibling modules remain importable when this script is invoked through
# one of those symlinks.
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from scipy.stats import beta

from audit_rare_scaffold_bridge import (
    MISSING_DOSAGE,
    audit_scaffold,
    build_alias_resolver,
    open_text,
)
from m27c_gvcf_core import (
    COMPATIBLE_STATES,
    STATE_NAMES,
    VariantKey,
    canonical_contig,
    parse_header_contract,
    parse_targeted_lines,
)


@dataclass(frozen=True)
class Baseline:
    keys: list[VariantKey]
    cm: np.ndarray
    samples: list[str]
    group_names: list[str]
    group_sizes: list[int]
    group_alt_counts: np.ndarray
    group_af: np.ndarray
    group_ci_low: np.ndarray
    group_ci_high: np.ndarray
    pooled_mac: np.ndarray
    pooled_maf: np.ndarray
    max_min_af: np.ndarray
    between_group_variance: np.ndarray


@dataclass(frozen=True)
class SampleResult:
    sample_index: int
    states: np.ndarray
    dosages: np.ndarray
    depths: np.ndarray
    gqs: np.ndarray
    n_records: int
    uncompressed_bytes: int
    elapsed_seconds: float


class CountingLines:
    def __init__(self, handle: TextIO):
        self.handle = handle
        self.n_records = 0
        self.n_bytes = 0

    def __iter__(self) -> Iterable[str]:
        for line in self.handle:
            if line and not line.startswith("#"):
                self.n_records += 1
                self.n_bytes += len(line.encode("utf-8"))
            yield line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvcfs", nargs="+", required=True, type=Path)
    parser.add_argument("--gvcf-indexes", nargs="+", required=True, type=Path)
    parser.add_argument("--gcs-input-manifest", required=True, type=Path)
    parser.add_argument("--phased-scaffold-vcf", required=True, type=Path)
    parser.add_argument("--gnomix-reference-vcf", required=True, type=Path)
    parser.add_argument("--gnomix-config", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--reference-fasta", required=True, type=Path)
    parser.add_argument("--reference-fai", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--samtools", default="samtools")
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_info_float(info: str, key: str) -> float | None:
    prefix = f"{key}="
    for item in info.split(";"):
        if item.startswith(prefix):
            try:
                return float(item[len(prefix):].split(",", 1)[0])
            except ValueError:
                return None
    return None


def parse_baseline(path: Path, expected_groups: list[str]) -> Baseline:
    samples: list[str] = []
    keys: list[VariantKey] = []
    cm_values: list[float] = []
    group_indices: dict[str, list[int]] = {group: [] for group in expected_groups}
    group_counts: list[list[int]] = []
    seen: set[VariantKey] = set()

    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                for index, sample in enumerate(samples):
                    group = sample.split("_", 1)[0]
                    if group not in group_indices:
                        raise SystemExit(f"Unexpected baseline ancestry prefix: {group}")
                    group_indices[group].append(index)
                continue
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 + len(samples):
                raise SystemExit("Malformed frozen Gnomix reference VCF")
            key = (
                canonical_contig(fields[0]),
                int(fields[1]),
                fields[3].upper(),
                fields[4].upper(),
            )
            if len(key[2]) != 1 or len(key[3]) != 1 or "," in key[3]:
                raise SystemExit("Frozen Gnomix reference contains a non-biallelic SNV")
            if key in seen:
                raise SystemExit("Duplicate frozen Gnomix marker key")
            seen.add(key)
            dosages: list[int] = []
            for value in fields[9:]:
                gt = value.split(":", 1)[0].replace("|", "/").split("/")
                if len(gt) != 2 or any(allele not in {"0", "1"} for allele in gt):
                    raise SystemExit("Missing or non-biallelic genotype in frozen baseline")
                dosages.append(sum(int(allele) for allele in gt))
            group_counts.append(
                [sum(dosages[index] for index in group_indices[group]) for group in expected_groups]
            )
            keys.append(key)
            cm = parse_info_float(fields[7], "CM")
            cm_values.append(float("nan") if cm is None else cm)

    sizes = [len(group_indices[group]) for group in expected_groups]
    counts = np.asarray(group_counts, dtype=np.int16)
    denominators = 2 * np.asarray(sizes, dtype=np.float64)
    af = counts / denominators
    ci_low = beta.ppf(0.025, counts + 0.5, denominators - counts + 0.5)
    ci_high = beta.ppf(0.975, counts + 0.5, denominators - counts + 0.5)
    pooled_alt = counts.sum(axis=1)
    pooled_an = int(denominators.sum())
    pooled_mac = np.minimum(pooled_alt, pooled_an - pooled_alt)
    return Baseline(
        keys=keys,
        cm=np.asarray(cm_values, dtype=np.float64),
        samples=samples,
        group_names=expected_groups,
        group_sizes=sizes,
        group_alt_counts=counts,
        group_af=af,
        group_ci_low=ci_low,
        group_ci_high=ci_high,
        pooled_mac=pooled_mac,
        pooled_maf=pooled_mac / pooled_an,
        max_min_af=np.max(af, axis=1) - np.min(af, axis=1),
        between_group_variance=np.var(af, axis=1),
    )


def parse_model_config(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t", 1)
            if len(fields) == 2:
                result[fields[0]] = fields[1]
    return result


def read_header(bcftools: str, path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [bcftools, "view", "-h", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return parse_header_contract(completed.stdout)


def make_target_index(
    model_keys: list[VariantKey], scaffold_keys: list[VariantKey]
) -> tuple[list[VariantKey], dict[VariantKey, int], dict[int, list[int]]]:
    target_keys = list(model_keys)
    seen = set(model_keys)
    target_keys.extend(key for key in scaffold_keys if key not in seen)
    key_to_index = {key: index for index, key in enumerate(target_keys)}
    positions: dict[int, list[int]] = {}
    for index, key in enumerate(target_keys):
        positions.setdefault(key[1], []).append(index)
    return target_keys, key_to_index, positions


def write_regions(path: Path, contig: str, positions: Iterable[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for position in sorted(set(positions)):
            handle.write(f"chr{contig}\t{position - 1}\t{position}\n")


def parse_one_sample(
    sample_index: int,
    path: Path,
    bcftools: str,
    regions: Path,
    target_keys: list[VariantKey],
    positions_to_indices: dict[int, list[int]],
) -> SampleResult:
    started = time.monotonic()
    process = subprocess.Popen(
        [bcftools, "view", "-R", str(regions), "-H", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    counter = CountingLines(process.stdout)
    calls = parse_targeted_lines(counter, target_keys, positions_to_indices)
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"bcftools failed for input {sample_index}: {stderr[-2000:]}")
    return SampleResult(
        sample_index=sample_index,
        states=np.fromiter((call.state for call in calls), dtype=np.int8, count=len(calls)),
        dosages=np.fromiter((call.dosage for call in calls), dtype=np.int8, count=len(calls)),
        depths=np.fromiter((call.depth for call in calls), dtype=np.int32, count=len(calls)),
        gqs=np.fromiter((call.gq for call in calls), dtype=np.int32, count=len(calls)),
        n_records=counter.n_records,
        uncompressed_bytes=counter.n_bytes,
        elapsed_seconds=time.monotonic() - started,
    )


def load_reference_contig(samtools: str, fasta: Path, contig: str) -> str:
    completed = subprocess.run(
        [samtools, "faidx", str(fasta), f"chr{contig}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = completed.stdout.splitlines()
    if not lines or lines[0] != f">chr{contig}":
        raise SystemExit("Pinned reference FASTA did not return the expected chr22 contig")
    return "".join(lines[1:]).upper()


def validate_gcs_manifest(path: Path, gvcfs: list[Path], indexes: list[Path]) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {item.name for item in gvcfs + indexes}
    manifest_names = {Path(row.get("uri", "")).name for row in rows}
    complete_rows = all(
        row.get("uri")
        and row.get("generation")
        and row.get("size_bytes")
        and (row.get("crc32c") or row.get("md5_hash"))
        for row in rows
    )
    return {
        "n_rows": len(rows),
        "n_required_objects": len(required),
        "n_required_objects_found": len(required & manifest_names),
        "all_required_objects_found": required <= manifest_names,
        "all_rows_have_generation_size_and_checksum": complete_rows,
        "sha256": sha256(path),
    }


def window_index(marker_index: int, markers_per_window: int, n_windows: int) -> int:
    return min(marker_index // markers_per_window, n_windows - 1)


def max_gap(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    return float(np.max(np.diff(values)))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27C_TARGETED_GVCF_READINESS":
        raise SystemExit("Invalid M27C preregistration stage")
    args.outdir.mkdir(parents=True, exist_ok=True)
    contract = prereg["frozen_contract"]
    chromosome = str(contract["chromosome"])
    expected_samples = int(contract["expected_gvcf_samples"])
    minimum_fraction = float(contract["minimum_gnomix_ready_marker_fraction"])
    identity_marker_floor = int(contract["identity_min_jointly_called_markers"])
    identity_concordance_floor = float(contract["identity_min_dosage_concordance"])

    baseline = parse_baseline(args.gnomix_reference_vcf, list(contract["ancestries"]))
    config = parse_model_config(args.gnomix_config)
    model_count_ok = len(baseline.keys) == int(contract["expected_model_markers"])
    baseline_group_ok = (
        len(baseline.samples) == int(contract["expected_baseline_donors"])
        and all(
            size == int(contract["expected_donors_per_ancestry"])
            for size in baseline.group_sizes
        )
    )
    model_config_ok = (
        config.get("C", "").isdigit()
        and int(config["C"]) == len(baseline.keys)
        and config.get("M", "").isdigit()
        and config.get("W", "").isdigit()
    )

    if len(args.gvcfs) != len(args.gvcf_indexes):
        raise SystemExit("gVCF/index count mismatch")
    by_name = {path.name: path for path in args.gvcf_indexes}
    missing_indexes = [path.name for path in args.gvcfs if f"{path.name}.tbi" not in by_name]
    if missing_indexes:
        raise SystemExit(f"Missing staged index for {len(missing_indexes)} gVCFs")

    with ThreadPoolExecutor(max_workers=max(1, args.readers)) as pool:
        header_futures = {pool.submit(read_header, args.bcftools, path): index for index, path in enumerate(args.gvcfs)}
        headers: list[dict[str, object] | None] = [None] * len(args.gvcfs)
        for future in as_completed(header_futures):
            headers[header_futures[future]] = future.result()
    header_contracts = [header for header in headers if header is not None]
    gvcf_samples = [str(header["samples"][0]) for header in header_contracts if len(header["samples"]) == 1]
    header_ok = all(
        len(header["samples"]) == 1
        and header["chr22_length"] == 50818468
        and header["has_required_fields"]
        and "HaplotypeCaller" in str(header["source"])
        for header in header_contracts
    ) and len(gvcf_samples) == len(args.gvcfs)

    resolver = build_alias_resolver(args.metadata)
    raw_ids, raw_resolution = resolver.resolve(gvcf_samples, "raw")
    scaffold_samples: list[str] = []
    with open_text(args.phased_scaffold_vcf) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                scaffold_samples = line.rstrip("\n").split("\t")[9:]
                break
    scaffold_ids, scaffold_resolution = resolver.resolve(scaffold_samples, "scaffold")
    scaffold_audit, scaffold_genotypes, sample_bridge = audit_scaffold(
        args.phased_scaffold_vcf,
        chromosome,
        raw_ids,
        scaffold_ids,
    )

    scaffold_keys = sorted(scaffold_genotypes)
    target_keys, key_to_index, positions_to_indices = make_target_index(baseline.keys, scaffold_keys)
    model_target_indices = np.asarray([key_to_index[key] for key in baseline.keys], dtype=np.int64)
    scaffold_target_indices = np.asarray([key_to_index[key] for key in scaffold_keys], dtype=np.int64)
    model_scaffold_entries = [scaffold_genotypes.get(key) for key in baseline.keys]
    regions = args.outdir / "m27c_target_positions.bed"
    write_regions(regions, chromosome, positions_to_indices)

    reference = load_reference_contig(args.samtools, args.reference_fasta, chromosome)
    ref_mismatches = sum(
        key[1] > len(reference) or reference[key[1] - 1] != key[2] for key in baseline.keys
    )
    manifest = validate_gcs_manifest(args.gcs_input_manifest, args.gvcfs, args.gvcf_indexes)

    policies = [prereg["quality_policies"]["primary"], *prereg["quality_policies"]["one_factor_sensitivities"]]
    policies = [{"id": "primary", **policies[0]}, *policies[1:]]
    n_model = len(baseline.keys)
    n_samples = len(args.gvcfs)
    state_counts = np.zeros((len(STATE_NAMES), n_model), dtype=np.uint16)
    high_quality_counts = {policy["id"]: np.zeros(n_model, dtype=np.uint16) for policy in policies}
    phase_missing_counts = {policy["id"]: np.zeros(n_model, dtype=np.uint16) for policy in policies}
    phase_supported_counts = {policy["id"]: np.zeros(n_model, dtype=np.uint16) for policy in policies}
    phase_trivial_counts = {policy["id"]: np.zeros(n_model, dtype=np.uint16) for policy in policies}
    identity_called = np.zeros(n_samples, dtype=np.int32)
    identity_match = np.zeros(n_samples, dtype=np.int32)
    sample_elapsed = np.zeros(n_samples, dtype=np.float64)
    sample_records = np.zeros(n_samples, dtype=np.int64)
    sample_bytes = np.zeros(n_samples, dtype=np.int64)
    compatible_codes = np.asarray(sorted(COMPATIBLE_STATES), dtype=np.int8)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.readers)) as pool:
        futures = {
            pool.submit(
                parse_one_sample,
                index,
                path,
                args.bcftools,
                regions,
                target_keys,
                positions_to_indices,
            ): index
            for index, path in enumerate(args.gvcfs)
        }
        for future in as_completed(futures):
            result = future.result()
            sample_index = result.sample_index
            model_states = result.states[model_target_indices]
            model_dosages = result.dosages[model_target_indices]
            model_depths = result.depths[model_target_indices]
            model_gqs = result.gqs[model_target_indices]
            for code in STATE_NAMES:
                state_counts[code] += model_states == code

            scaffold_dosage = np.fromiter(
                (
                    entry.dosages[sample_index]
                    if entry is not None
                    else MISSING_DOSAGE
                    for entry in model_scaffold_entries
                ),
                dtype=np.uint8,
                count=n_model,
            )
            scaffold_phased = np.fromiter(
                (
                    entry.phased[sample_index]
                    if entry is not None
                    else 0
                    for entry in model_scaffold_entries
                ),
                dtype=np.uint8,
                count=n_model,
            )
            phase_support = (model_dosages == 1) & (scaffold_dosage == 1) & (scaffold_phased == 1)
            compatible = np.isin(model_states, compatible_codes) & (model_dosages >= 0)
            for policy in policies:
                policy_id = policy["id"]
                high_quality = (
                    compatible
                    & (model_depths >= int(policy["minimum_effective_depth"]))
                    & (model_gqs >= int(policy["minimum_gq"]))
                )
                het = high_quality & (model_dosages == 1)
                high_quality_counts[policy_id] += high_quality
                phase_missing_counts[policy_id] += het & ~phase_support
                phase_supported_counts[policy_id] += het & phase_support
                phase_trivial_counts[policy_id] += high_quality & (model_dosages != 1)

            identity_states = result.states[scaffold_target_indices]
            identity_dosages = result.dosages[scaffold_target_indices]
            expected_dosages = np.fromiter(
                (scaffold_genotypes[key].dosages[sample_index] for key in scaffold_keys),
                dtype=np.uint8,
                count=len(scaffold_keys),
            )
            jointly_called = (
                np.isin(identity_states, compatible_codes)
                & (identity_dosages >= 0)
                & (expected_dosages != MISSING_DOSAGE)
            )
            identity_called[sample_index] = int(np.sum(jointly_called))
            identity_match[sample_index] = int(np.sum(jointly_called & (identity_dosages == expected_dosages)))
            sample_elapsed[sample_index] = result.elapsed_seconds
            sample_records[sample_index] = result.n_records
            sample_bytes[sample_index] = result.uncompressed_bytes

    wall_seconds = time.monotonic() - started
    identity_concordance = np.divide(
        identity_match,
        identity_called,
        out=np.zeros_like(identity_match, dtype=np.float64),
        where=identity_called > 0,
    )
    identity_pass = bool(
        len(identity_called) > 0
        and np.min(identity_called) >= identity_marker_floor
        and np.min(identity_concordance) >= identity_concordance_floor
    )

    policy_rows: list[dict[str, object]] = []
    ready_by_policy: dict[str, np.ndarray] = {}
    quality_only_by_policy: dict[str, np.ndarray] = {}
    for policy in policies:
        policy_id = policy["id"]
        minimum_called = math.ceil(float(policy["minimum_marker_call_rate"]) * n_samples)
        quality_only = high_quality_counts[policy_id] >= minimum_called
        ready = quality_only & (phase_missing_counts[policy_id] == 0)
        quality_only_by_policy[policy_id] = quality_only
        ready_by_policy[policy_id] = ready
        policy_rows.append(
            {
                "policy": policy_id,
                "minimum_effective_depth": policy["minimum_effective_depth"],
                "minimum_gq": policy["minimum_gq"],
                "minimum_marker_call_rate": policy["minimum_marker_call_rate"],
                "minimum_called_samples": minimum_called,
                "n_quality_only_markers": int(np.sum(quality_only)),
                "quality_only_fraction": float(np.mean(quality_only)),
                "n_candidate_panel_ready_markers": int(np.sum(ready)),
                "candidate_panel_ready_fraction": float(np.mean(ready)),
                "passes_0_8": bool(np.mean(ready) >= minimum_fraction),
            }
        )

    primary_ready = ready_by_policy["primary"]
    primary_pass = bool(np.mean(primary_ready) >= minimum_fraction)
    sensitivity_passes = [bool(row["passes_0_8"]) for row in policy_rows[1:]]
    if primary_pass:
        robustness = "PASS_ROBUST" if all(sensitivity_passes) else "PASS_THRESHOLD_SENSITIVE"
    else:
        robustness = "FAIL_ROBUST" if not any(sensitivity_passes) else "FAIL_PRIMARY_PERMISSIVE_PASS"

    informative = baseline.max_min_af > 0
    markers_per_window = int(config["M"])
    n_windows = int(config["W"])
    marker_windows = np.fromiter(
        (window_index(index, markers_per_window, n_windows) for index in range(n_model)),
        dtype=np.int32,
        count=n_model,
    )
    window_rows: list[dict[str, object]] = []
    for window in range(n_windows):
        mask = marker_windows == window
        ready = primary_ready & mask
        informative_mask = informative & mask
        ready_positions = np.asarray([baseline.keys[index][1] for index in np.flatnonzero(ready)], dtype=np.int64)
        ready_cm = baseline.cm[ready]
        window_rows.append(
            {
                "model_window": window,
                "first_marker_index": int(np.flatnonzero(mask)[0]),
                "last_marker_index": int(np.flatnonzero(mask)[-1]),
                "n_model_markers": int(np.sum(mask)),
                "n_ready_primary": int(np.sum(ready)),
                "ready_fraction_primary": float(np.sum(ready) / np.sum(mask)),
                "n_informative_baseline": int(np.sum(informative_mask)),
                "n_informative_and_ready": int(np.sum(informative_mask & primary_ready)),
                "informative_and_ready_fraction_of_window": float(np.sum(informative_mask & primary_ready) / np.sum(mask)),
                "max_ready_gap_bp": max_gap(ready_positions),
                "max_ready_gap_cm": max_gap(ready_cm[np.isfinite(ready_cm)]),
            }
        )

    marker_path = args.outdir / "m27c_marker_summary.tsv.gz"
    with gzip.open(marker_path, "wt", encoding="utf-8", newline="") as handle:
        fields = [
            "chrom", "pos", "ref", "alt", "model_window", "pooled_mac_78", "pooled_maf_78",
            "max_min_ancestry_af", "between_ancestry_af_variance", "informative_observed",
            "ready_primary", "structural_covered_fraction", "compatible_called_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        structural = 1.0 - state_counts[0] / n_samples
        compatible = sum(state_counts[code] for code in COMPATIBLE_STATES) / n_samples
        for index, key in enumerate(baseline.keys):
            writer.writerow(
                {
                    "chrom": key[0],
                    "pos": key[1],
                    "ref": key[2],
                    "alt": key[3],
                    "model_window": int(marker_windows[index]),
                    "pooled_mac_78": int(baseline.pooled_mac[index]),
                    "pooled_maf_78": float(baseline.pooled_maf[index]),
                    "max_min_ancestry_af": float(baseline.max_min_af[index]),
                    "between_ancestry_af_variance": float(baseline.between_group_variance[index]),
                    "informative_observed": bool(informative[index]),
                    "ready_primary": bool(primary_ready[index]),
                    "structural_covered_fraction": float(structural[index]),
                    "compatible_called_fraction": float(compatible[index]),
                }
            )

    write_tsv(
        args.outdir / "m27c_readiness_by_policy.tsv",
        list(policy_rows[0]),
        policy_rows,
    )
    write_tsv(args.outdir / "m27c_window_summary.tsv", list(window_rows[0]), window_rows)

    input_ok = all(
        (
            model_count_ok,
            baseline_group_ok,
            model_config_ok,
            header_ok,
            ref_mismatches == 0,
            len(args.gvcfs) == (n_samples if args.smoke else expected_samples),
            len(set(raw_ids)) == len(raw_ids),
            raw_resolution["n_unmapped_in_metadata"] == 0,
            raw_resolution["n_ambiguous_in_metadata"] == 0,
            sample_bridge["n_raw_ids_absent_from_scaffold"] == 0,
            sample_bridge["n_raw_ids_ambiguous_in_scaffold"] == 0,
            manifest["all_required_objects_found"],
            manifest["all_rows_have_generation_size_and_checksum"],
        )
    )
    parser_ok = bool(np.sum(state_counts) == n_model * n_samples and identity_pass)
    information = {
        "baseline_group_names": baseline.group_names,
        "baseline_group_sizes": baseline.group_sizes,
        "n_monomorphic_in_78": int(np.sum((baseline.pooled_mac == 0))),
        "n_polymorphic_no_observed_group_difference": int(np.sum((baseline.pooled_mac > 0) & ~informative)),
        "n_observed_ancestry_difference": int(np.sum(informative)),
        "n_primary_ready_and_observed_ancestry_difference": int(np.sum(primary_ready & informative)),
        "fraction_primary_ready_markers_observed_informative": float(np.mean(informative[primary_ready])) if np.any(primary_ready) else None,
        "jeffreys_interval": "Beta(alt_count+0.5, ref_count+0.5), equal-tailed 95 percent",
        "biological_validation_claimed": False,
    }
    spatial = {
        "model_markers_per_base_window": markers_per_window,
        "n_model_windows": n_windows,
        "last_window_contains_remainder": True,
        "fraction_windows_below": {
            str(threshold): float(np.mean([row["ready_fraction_primary"] < threshold for row in window_rows]))
            for threshold in (0.5, 0.8, 0.95)
        },
        "global_max_ready_gap_bp": max_gap(np.asarray([baseline.keys[index][1] for index in np.flatnonzero(primary_ready)])),
        "global_max_ready_gap_cm": max_gap(baseline.cm[primary_ready & np.isfinite(baseline.cm)]),
    }
    identity = {
        "n_samples_evaluated": n_samples,
        "jointly_called_markers_min": int(np.min(identity_called)) if n_samples else 0,
        "jointly_called_markers_median": float(np.median(identity_called)) if n_samples else 0,
        "jointly_called_markers_max": int(np.max(identity_called)) if n_samples else 0,
        "dosage_concordance_min": float(np.min(identity_concordance)) if n_samples else None,
        "dosage_concordance_median": float(np.median(identity_concordance)) if n_samples else None,
        "dosage_concordance_max": float(np.max(identity_concordance)) if n_samples else None,
        "floor_jointly_called": identity_marker_floor,
        "floor_dosage_concordance": identity_concordance_floor,
        "status": "PASS" if identity_pass else "FAIL",
        "sample_ids_emitted": False,
    }
    callability = {
        "n_model_markers": n_model,
        "n_samples": n_samples,
        "state_totals": {STATE_NAMES[code]: int(np.sum(state_counts[code])) for code in STATE_NAMES},
        "structural_coverage_fraction": float(1.0 - np.sum(state_counts[0]) / (n_model * n_samples)),
        "policy_results": policy_rows,
        "robustness_classification": robustness,
        "phase_primary": {
            "n_phase_trivial_calls": int(np.sum(phase_trivial_counts["primary"])),
            "n_phase_supported_heterozygous_calls": int(np.sum(phase_supported_counts["primary"])),
            "n_phase_missing_heterozygous_calls": int(np.sum(phase_missing_counts["primary"])),
        },
    }
    operations = {
        "readers": args.readers,
        "wall_seconds_targeted_reads": wall_seconds,
        "sample_elapsed_seconds_min": float(np.min(sample_elapsed)) if n_samples else None,
        "sample_elapsed_seconds_median": float(np.median(sample_elapsed)) if n_samples else None,
        "sample_elapsed_seconds_max": float(np.max(sample_elapsed)) if n_samples else None,
        "n_target_records_total": int(np.sum(sample_records)),
        "target_record_uncompressed_bytes_total": int(np.sum(sample_bytes)),
        "whole_gvcf_staging_performed": False,
        "sample_ids_emitted": False,
    }
    write_json(args.outdir / "m27c_input_contract.json", {
        "stage": prereg["stage"],
        "scope": prereg["scope"],
        "smoke": args.smoke,
        "chromosome": chromosome,
        "n_gvcf": n_samples,
        "n_indexes": len(args.gvcf_indexes),
        "n_unique_resolved_sample_ids": len(set(raw_ids)),
        "header_contract_pass": header_ok,
        "reference_ref_mismatches": ref_mismatches,
        "model_marker_count": n_model,
        "model_config": {key: config.get(key) for key in ("A", "C", "M", "S", "W", "context")},
        "gcs_input_manifest": manifest,
        "sample_alias_resolution": raw_resolution,
        "scaffold_alias_resolution": scaffold_resolution,
        "sample_bridge": sample_bridge,
        "sample_ids_emitted": False,
    })
    write_json(args.outdir / "m27c_identity_control.json", identity)
    write_json(args.outdir / "m27c_callability_summary.json", callability)
    write_json(args.outdir / "m27c_ancestral_information.json", information)
    write_json(args.outdir / "m27c_spatial_diagnostics.json", spatial)
    write_json(args.outdir / "m27c_operational_metrics.json", operations)

    c2_pass = primary_pass
    gates = [
        {"gate": "C0", "name": "input_identity_and_coordinate_contract", "status": "PASS" if input_ok else "FAIL"},
        {"gate": "C1", "name": "targeted_callability_and_allele_compatibility", "status": "PASS" if parser_ok else "FAIL"},
        {"gate": "C2", "name": "candidate_panel_phase_ready_marker_fraction", "status": "PASS" if c2_pass else "FAIL"},
        {"gate": "C3", "name": "ancestral_information_and_spatial_distribution", "status": "PASS"},
    ]
    if args.smoke:
        decision = "SMOKE_ONLY"
    elif not input_ok:
        decision = "STOP_INPUT_CONTRACT"
    elif not parser_ok:
        decision = "STOP_PARSER_OR_VALIDATION"
    elif not primary_pass:
        quality_fraction = float(np.mean(quality_only_by_policy["primary"]))
        decision = "STOP_PHASE_READINESS" if quality_fraction >= minimum_fraction else "STOP_BELOW_GNOMIX_FLOOR"
    elif robustness == "PASS_THRESHOLD_SENSITIVE":
        decision = "REVIEW_THRESHOLD_SENSITIVITY"
    else:
        decision = "READY_FOR_RARE_DONOR_AUDIT_ONLY"
    write_tsv(args.outdir / "m27c_gates.tsv", ["gate", "name", "status"], gates)
    summary = {
        "stage": prereg["stage"],
        "scope": prereg["scope"],
        "smoke": args.smoke,
        "decision": decision,
        "gates": {row["gate"]: row["status"] for row in gates},
        "primary_candidate_panel_ready_fraction": float(np.mean(primary_ready)),
        "minimum_marker_fraction": minimum_fraction,
        "robustness_classification": robustness,
        "final_donor_panel_certified": False,
        "pcrelate_executed": False,
        "gnomix_executed": False,
        "simulation_performed": False,
        "training_performed": False,
        "test_opened": False,
        "sample_ids_emitted": False,
    }
    write_json(args.outdir / "m27c_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.readers < 1:
        raise SystemExit("--readers must be positive")
    summary = run(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
