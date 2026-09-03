#!/usr/bin/env python3
"""Materialize factorized M36 CORA-Set chr22 inputs from read-only sources.

The real representation is deliberately sparse: one locus row, carrier rows,
and missing rows.  Non-carriers are implicit zero-evaluable calls, so the
pipeline never writes an individual-by-locus matrix.  asIBD (Refined-IBD
segments stratified by Gnomix ancestry) is an exploratory common-IBD target,
not truth and not evidence of method superiority.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator


class MaterializeError(ValueError):
    """Raised when a sparse M36 input cannot be authenticated or normalized."""


LOCUS_KEY_REQUIRED = {"chrom", "position", "ref", "alt"}
SAMPLE_REQUIRED = {"sample_id", "cohort", "rare_callability", "Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS"}
COMPONENT_REQUIRED = {"sample_id", "pcrelate_component"}
ASIBD_MANIFEST_REQUIRED = {"gnomix_ancestry", "segment_file"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializeError(message)


def canonical_chrom(value: str) -> str:
    return value.removeprefix("chr")


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")


def read_tsv(path: Path) -> list[dict[str, str]]:
    require(path.exists(), f"missing input: {path}")
    with open_text(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows and rows[0], f"empty TSV: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    missing = required - set(rows[0])
    require(not missing, f"{label} missing columns: {sorted(missing)}")


def number(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise MaterializeError(f"{label} must be numeric") from error
    require(math.isfinite(result), f"{label} must be finite")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_gt(raw: str) -> int | None:
    gt = raw.split(":", 1)[0].replace("|", "/")
    if gt in {".", "./.", ".|."}:
        return None
    alleles = gt.split("/")
    require(len(alleles) == 2 and set(alleles) <= {"0", "1"}, "VCF must be diploid biallelic GT")
    return int(alleles[0]) + int(alleles[1])


def vcf_records(path: Path, chromosome: str) -> Iterator[tuple[list[str], dict[str, Any]]]:
    samples: list[str] | None = None
    with open_text(path) as handle:
        for raw in handle:
            if raw.startswith("##"):
                continue
            if raw.startswith("#CHROM"):
                fields = raw.rstrip("\n").split("\t")
                require(len(fields) >= 10, "VCF must contain diploid sample columns")
                samples = fields[9:]
                require(len(samples) == len(set(samples)), "VCF sample IDs are duplicated")
                continue
            if not raw.strip():
                continue
            require(samples is not None, "VCF header #CHROM is missing")
            fields = raw.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), "VCF row/sample count mismatch")
            if canonical_chrom(fields[0]) != canonical_chrom(chromosome):
                continue
            require("," not in fields[4] and fields[4] != ".", "M36 accepts normalized biallelic ALT only")
            yield samples, {
                "chrom": fields[0], "position": fields[1], "ref": fields[3], "alt": fields[4],
                "genotypes": [parse_gt(value) for value in fields[9:]],
            }


def load_optional_locus_metadata(path: Path | None, chromosome: str) -> dict[tuple[str, str, str, str], dict[str, str]]:
    if path is None:
        return {}
    require(path.exists(), f"missing input: {path}")
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None, "locus metadata header is missing")
        rows = list(reader)
    missing_columns = LOCUS_KEY_REQUIRED - set(reader.fieldnames)
    require(not missing_columns, f"locus metadata missing columns: {sorted(missing_columns)}")
    if not rows:
        return {}
    result: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        if canonical_chrom(row["chrom"]) != canonical_chrom(chromosome):
            continue
        key = (canonical_chrom(row["chrom"]), row["position"], row["ref"], row["alt"])
        require(key not in result, f"duplicate locus metadata key: {key}")
        result[key] = row
    return result


def load_genetic_map(path: Path, chromosome: str) -> list[tuple[int, float]]:
    """Read either a named TSV map or Gnomix's headerless CHR BP cM map."""
    require(path.exists(), f"missing input: {path}")
    with open_text(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    require(lines, "genetic map is empty")
    first = lines[0].replace("\t", " ").split()
    header_names = {name.lower() for name in first}
    has_header = bool(header_names & {"chrom", "chr", "chromosome", "position", "pos", "bp", "cm", "genetic_cm"})
    points: list[tuple[int, float]] = []
    if has_header:
        names = {name.lower(): index for index, name in enumerate(first)}
        chrom_column = next((names.get(name) for name in ("chrom", "chr", "chromosome") if name in names), None)
        position_column = next((names.get(name) for name in ("position", "pos", "bp") if name in names), None)
        cm_column = next((names.get(name) for name in ("cm", "genetic_cm") if name in names), None)
        require(chrom_column is not None and position_column is not None and cm_column is not None,
                "genetic map header requires chrom/position/cM columns")
        for line in lines[1:]:
            fields = line.replace("\t", " ").split()
            require(len(fields) > max(chrom_column, position_column, cm_column), "malformed genetic map row")
            if canonical_chrom(fields[chrom_column]) == canonical_chrom(chromosome):
                points.append((int(fields[position_column]), number(fields[cm_column], "genetic map cM")))
    else:
        for line in lines:
            fields = line.replace("\t", " ").split()
            require(len(fields) >= 3, "headerless genetic map requires CHROM POSITION cM columns")
            if canonical_chrom(fields[0]) == canonical_chrom(chromosome):
                points.append((int(fields[1]), number(fields[2], "genetic map cM")))
    require(len(points) >= 2, "genetic map needs at least two points on feature chromosome")
    points.sort()
    require(all(left[0] < right[0] and left[1] <= right[1] for left, right in zip(points, points[1:])),
            "genetic map positions must increase and cM must be nondecreasing")
    return points


def interpolate_cm(points: list[tuple[int, float]], position: int) -> float:
    require(points[0][0] <= position <= points[-1][0], "VCF locus is outside genetic-map support")
    # ``points`` is strictly ordered by base-pair position.  Locate the
    # flanking interval in O(log n); scanning the full map for every rare
    # locus makes a chromosome-scale run unnecessarily quadratic.
    right_index = bisect_right(points, (position, math.inf))
    if right_index == len(points):
        return points[-1][1]
    left_bp, left_cm = points[right_index - 1]
    if left_bp == position:
        return left_cm
    right_bp, right_cm = points[right_index]
    return left_cm + (right_cm - left_cm) * (position - left_bp) / (right_bp - left_bp)


def factorize_vcf(rare_vcf: Path, locus_meta: dict[tuple[str, str, str, str], dict[str, str]],
                  genetic_map: list[tuple[int, float]], chromosome: str,
                  cohort_sample_ids: list[str]):
    loci: list[dict[str, Any]] = []
    carriers: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    sample_ids: list[str] | None = None
    seen_keys: set[tuple[str, str, str, str]] = set()
    for samples, record in vcf_records(rare_vcf, chromosome):
        if sample_ids is None:
            require(len(cohort_sample_ids) == len(set(cohort_sample_ids)), "sample metadata sample_id must be unique")
            sample_index = {sample: index for index, sample in enumerate(samples)}
            absent = set(cohort_sample_ids) - set(sample_index)
            require(not absent, f"canonical analysis samples absent from VCF: {len(absent)}")
            selected_indices = [sample_index[sample] for sample in cohort_sample_ids]
            sample_ids = list(cohort_sample_ids)
        else:
            require(len(samples) >= len(sample_ids), "VCF sample order changed within file")
        key = (canonical_chrom(record["chrom"]), record["position"], record["ref"], record["alt"])
        require(key not in seen_keys, f"duplicate VCF locus: {key}")
        seen_keys.add(key)
        metadata = locus_meta.get(key, {})
        selected_genotypes = [record["genotypes"][index] for index in selected_indices]
        called = [value for value in selected_genotypes if value is not None]
        an = 2 * len(called)
        require(an > 0, f"locus {key} has no called genotype")
        alt_ac = sum(called)
        ref_ac = an - alt_ac
        mac = min(alt_ac, ref_ac)
        if not 2 <= mac <= 10:
            continue
        # The Gnomix map does not cover chr22 telomeres in this release.  A
        # cM-dependent set encoder cannot silently extrapolate beyond it, so
        # those loci are excluded rather than assigned invented geometry.
        if not genetic_map[0][0] <= int(record["position"]) <= genetic_map[-1][0]:
            continue
        # Tie is deterministically ALT; in the selected MAC range it is not a
        # high-frequency ambiguity, and orientation is recorded in the output.
        minor_is_alt = alt_ac <= ref_ac
        minor_allele = record["alt"] if minor_is_alt else record["ref"]
        event_id = f"chr{key[0]}:{key[1]}:{record['ref']}:{record['alt']}:{minor_allele}"
        loci.append({
            "event_id": event_id, "chrom": f"chr{key[0]}", "position": key[1],
            "ref": record["ref"], "alt": record["alt"], "minor_allele": minor_allele,
            "minor_is_alt": int(minor_is_alt), "mac": mac, "an_called": an,
            "callability": f"{len(called) / len(sample_ids):.12g}",
            "mutation_context": metadata.get("mutation_context", "UNAVAILABLE"),
            "mutation_context_available": int("mutation_context" in metadata and bool(metadata["mutation_context"])),
            "cm": f"{interpolate_cm(genetic_map, int(record['position'])):.12g}",
            "common_copying_context": metadata.get("common_copying_context", "0"),
            "common_copying_context_available": int("common_copying_context" in metadata and bool(metadata["common_copying_context"])),
        })
        for sample_id, alt_dosage in zip(sample_ids, selected_genotypes, strict=True):
            if alt_dosage is None:
                missing.append({"sample_id": sample_id, "event_id": event_id})
                continue
            minor_dosage = alt_dosage if minor_is_alt else 2 - alt_dosage
            if minor_dosage:
                carriers.append({"sample_id": sample_id, "event_id": event_id, "minor_dosage": minor_dosage})
    require(sample_ids is not None and loci, "no MAC2-10 loci selected from rare VCF")
    return sample_ids, loci, carriers, missing


def declared_sample_ids(path: Path) -> list[str]:
    rows = read_tsv(path)
    require_columns(rows, SAMPLE_REQUIRED, "sample metadata")
    sample_ids = [row["sample_id"] for row in rows]
    require(len(sample_ids) == len(set(sample_ids)), "sample metadata sample_id must be unique")
    return sample_ids


def load_sample_metadata(path: Path, sample_ids: list[str], rare_burden: dict[str, int]) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = read_tsv(path)
    require_columns(rows, SAMPLE_REQUIRED, "sample metadata")
    by_id = {row["sample_id"]: row for row in rows}
    require(len(by_id) == len(rows), "sample metadata sample_id must be unique")
    require(set(by_id) == set(sample_ids), "sample metadata and VCF samples differ")
    output = []
    for sample in sample_ids:
        row = dict(by_id[sample])
        require(0 <= number(row["rare_callability"], "rare_callability") <= 1, "rare_callability outside [0,1]")
        q_sum = sum(number(row[key], key) for key in ("Q_AFR", "Q_EUR", "Q_NAM", "Q_EAS"))
        require(0.99 <= q_sum <= 1.01, "Q_AFR/Q_EUR/Q_NAM/Q_EAS must sum approximately to one")
        output.append({"sample_id": sample, "rare_burden": str(rare_burden.get(sample, 0)),
                       "rare_callability": row["rare_callability"], "cohort": row["cohort"],
                       "Q_AFR": row["Q_AFR"], "Q_EUR": row["Q_EUR"], "Q_NAM": row["Q_NAM"],
                       "Q_EAS": row["Q_EAS"]})
    asibd_to_sample = {}
    for sample in sample_ids:
        source = by_id[sample]
        asibd_id = source.get("asibd_id") or f"{source['cohort']}_{sample}"
        require(asibd_id not in asibd_to_sample, "metadata maps two samples to one asIBD ID")
        asibd_to_sample[asibd_id] = sample
    return output, asibd_to_sample


def load_components(path: Path, sample_ids: list[str]) -> list[dict[str, str]]:
    rows = read_tsv(path)
    require_columns(rows, COMPONENT_REQUIRED, "PC-Relate components")
    by_id = {row["sample_id"]: row for row in rows}
    require(len(by_id) == len(rows) and set(by_id) == set(sample_ids), "PC-Relate components and VCF samples differ")
    return [{"sample_id": sample, "pcrelate_component": by_id[sample]["pcrelate_component"]} for sample in sample_ids]


def parse_asibd_segments(path: Path, asibd_to_sample: dict[str, str]) -> Iterator[tuple[str, str, str, float]]:
    """Parse Nunes asIBD: ID1 HAP1 ID2 HAP2 CHROM START END cM."""
    with open_text(path) as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            require(len(fields) >= 8, f"asIBD row has fewer than 8 fields: {path.name}")
            left = asibd_to_sample.get(fields[0])
            right = asibd_to_sample.get(fields[2])
            if left is None or right is None or left == right:
                continue
            chrom = fields[4]
            cm = number(fields[7], "asIBD cM")
            require(cm >= 0, "asIBD cM must be nonnegative")
            first, second = (left, right) if left < right else (right, left)
            yield first, second, chrom, cm


def materialize_targets(manifest: Path, segment_paths: list[Path], sample_ids: list[str], asibd_to_sample: dict[str, str], components: list[dict[str, str]],
                        feature_chrom: str, ratio: int, seed: int,
                        max_positives_per_stratum: int) -> list[dict[str, Any]]:
    require(ratio >= 1, "zero-negative ratio must be >= 1")
    require(max_positives_per_stratum >= 1, "max positives per target stratum must be >= 1")
    manifest_rows = read_tsv(manifest)
    require_columns(manifest_rows, ASIBD_MANIFEST_REQUIRED, "asIBD manifest")
    staged = {path.name: path for path in segment_paths}
    component = {row["sample_id"]: row["pcrelate_component"] for row in components}
    # The predictive target is one cross-chromosome common-IBD quantity per
    # pair: total Refined-IBD cM outside the feature chromosome.  Do not turn
    # chromosome or Gnomix ancestry bookkeeping into pseudo-replicates.
    aggregates: dict[tuple[str, str], float] = defaultdict(float)
    for row in manifest_rows:
        path = staged.get(row["segment_file"])
        require(path is not None, f"asIBD manifest file was not staged: {row['segment_file']}")
        for left, right, target_chrom, cm in parse_asibd_segments(path, asibd_to_sample):
            if canonical_chrom(target_chrom) == canonical_chrom(feature_chrom):
                continue
            aggregates[(left, right)] += cm
    require(aggregates, "asIBD manifest produced no in-cohort positive pair")
    # The source has millions of overlapping segments.  The estimand uses a
    # fixed, deterministic sparse pair sample within PC-Relate relation,
    # rather than treating segments as independent observations or
    # materialising all sample pairs.
    positives_by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_by_stratum: dict[str, int] = defaultdict(int)
    sampling_rng = random.Random(seed)
    occupied: set[tuple[str, str]] = set()
    for (left, right), cm in aggregates.items():
        pair_class = "within_component" if component[left] == component[right] else "between_component"
        occupied.add((left, right))
        stratum = pair_class
        row = {"sample_i": left, "sample_j": right, "target_chrom": "outside_chr22_total",
               "target_source": "asibd_refined_ibd_gnomix_stratified_exploratory",
               "target_stratum": pair_class, "target_cm": f"{cm:.12g}",
               "target_positive": 1, "target": f"{math.log1p(cm):.12g}"}
        seen_by_stratum[stratum] += 1
        selected = positives_by_stratum[stratum]
        if len(selected) < max_positives_per_stratum:
            selected.append(row)
        else:
            replacement = sampling_rng.randrange(seen_by_stratum[stratum])
            if replacement < max_positives_per_stratum:
                selected[replacement] = row
    positives = [row for stratum in sorted(positives_by_stratum) for row in positives_by_stratum[stratum]]
    # Sample true zero pairs independently by PC-Relate relation.  Some
    # related components are saturated: every within-component pair has
    # positive asIBD outside chr22.  In that case use the zeros that actually
    # exist instead of fabricating balance.  A bounded reservoir keeps memory
    # proportional to the requested sample while enumerating the finite pair
    # universe exactly.
    rng = random.Random(seed)
    samples = sorted(sample_ids)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        by_stratum[row["target_stratum"]].append(row)
    wanted = {pair_class: ratio * len(rows) for pair_class, rows in by_stratum.items()}
    seen = defaultdict(int)
    reservoirs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for left, right in combinations(samples, 2):
        if (left, right) in occupied:
            continue
        pair_class = "within_component" if component[left] == component[right] else "between_component"
        capacity = wanted.get(pair_class, 0)
        if capacity == 0:
            continue
        seen[pair_class] += 1
        selected = reservoirs[pair_class]
        if len(selected) < capacity:
            selected.append((left, right))
        else:
            replacement = rng.randrange(seen[pair_class])
            if replacement < capacity:
                selected[replacement] = (left, right)
    negatives = []
    for pair_class, stratum_positives in sorted(by_stratum.items()):
        chosen = sorted(reservoirs[pair_class])
        negatives.extend({"sample_i": left, "sample_j": right, "target_chrom": "outside_chr22_total",
                          "target_source": "asibd_refined_ibd_gnomix_stratified_exploratory",
                          "target_stratum": pair_class, "target_cm": "0",
                          "target_positive": 0, "target": "0"} for left, right in chosen)
    return positives + negatives


def target_balance(rows: list[dict[str, Any]], requested_ratio: int) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "zero": 0})
    for row in rows:
        key = "positive" if int(row["target_positive"]) == 1 else "zero"
        counts[str(row["target_stratum"])][key] += 1
    return {
        stratum: {
            **values,
            "requested_zero_to_positive_ratio": requested_ratio,
            "achieved_zero_to_positive_ratio": values["zero"] / values["positive"] if values["positive"] else None,
            "zero_universe_saturated": values["zero"] < requested_ratio * values["positive"],
        }
        for stratum, values in sorted(counts.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rare-vcf", required=True, type=Path)
    parser.add_argument("--locus-metadata", type=Path, help="optional mutation/copying annotations keyed by locus")
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--sample-metadata", required=True, type=Path)
    parser.add_argument("--pcrelate-components", required=True, type=Path)
    parser.add_argument("--asibd-manifest", required=True, type=Path)
    parser.add_argument("--asibd-segments", required=True, nargs="+", type=Path)
    parser.add_argument("--feature-chrom", default="chr22")
    parser.add_argument("--zero-negative-ratio", type=int, default=1)
    parser.add_argument("--zero-negative-ratios", default=None, help="comma-separated initial sensitivity ratios, e.g. 1,3,5")
    parser.add_argument("--max-positives-per-stratum", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    locus_metadata = load_optional_locus_metadata(args.locus_metadata, args.feature_chrom)
    genetic_map = load_genetic_map(args.genetic_map, args.feature_chrom)
    requested_samples = declared_sample_ids(args.sample_metadata)
    samples, loci, carriers, missing = factorize_vcf(
        args.rare_vcf, locus_metadata, genetic_map, args.feature_chrom, requested_samples
    )
    burden: dict[str, int] = defaultdict(int)
    for row in carriers:
        burden[str(row["sample_id"])] += int(row["minor_dosage"])
    covariates, asibd_to_sample = load_sample_metadata(args.sample_metadata, samples, burden)
    components = load_components(args.pcrelate_components, samples)
    requested_ratios = getattr(args, "zero_negative_ratios", None)
    ratios = tuple(int(value) for value in requested_ratios.split(",")) if requested_ratios else (args.zero_negative_ratio,)
    require(ratios and all(value >= 1 for value in ratios) and len(set(ratios)) == len(ratios), "zero-negative ratios must be unique integers >=1")
    max_positives_per_stratum = getattr(args, "max_positives_per_stratum", 2000)
    target_sets = {ratio: materialize_targets(args.asibd_manifest, args.asibd_segments, samples, asibd_to_sample, components,
                                              args.feature_chrom, ratio, args.seed, max_positives_per_stratum)
                   for ratio in ratios}
    targets = target_sets[ratios[0]]
    outputs = {
        "loci": ("m36_cora_loci.tsv", list(loci[0]), loci),
        "carriers": ("m36_cora_carriers.tsv", ["sample_id", "event_id", "minor_dosage"], carriers),
        "missing": ("m36_cora_missing.tsv", ["sample_id", "event_id"], missing),
        "covariates": ("m36_cora_covariates.tsv", list(covariates[0]), covariates),
        "components": ("m36_cora_components.tsv", ["sample_id", "pcrelate_component"], components),
        "targets": ("m36_cora_external_targets.tsv", list(targets[0]), targets),
    }
    descriptors = {}
    for name, (filename, fields, rows) in outputs.items():
        path = args.outdir / filename
        write_tsv(path, fields, rows)
        descriptors[name] = {"uri": filename, "generation": "LOCAL_CHAIN", "sha256": sha256_file(path)}
    for ratio, rows in target_sets.items():
        if ratio != ratios[0]:
            write_tsv(args.outdir / f"m36_cora_external_targets_zero{ratio}.tsv", list(rows[0]), rows)
    source_paths = {
        "rare_vcf": args.rare_vcf, "genetic_map": args.genetic_map,
        "sample_metadata": args.sample_metadata, "pcrelate_components": args.pcrelate_components,
        "asibd_manifest": args.asibd_manifest,
    }
    source_paths.update({f"asibd_segment_{path.name}": path for path in args.asibd_segments})
    receipt = {
        "stage": "M36_CORA_MATERIALIZE", "status": "MATERIALIZED_PASS",
        "synthetic": False, "feature_schema": "m36_factorized_sparse_v1",
        "external_target_schema": "m36_external_common_pairs_log1p_v3_pair_total",
        "feature_chrom": args.feature_chrom, "input_descriptors": descriptors,
        "target_interpretation": "asIBD Refined-IBD stratified by Gnomix; exploratory predictive common-IBD target, not orthogonal truth or superiority evidence",
        "noncarrier_representation": "implicit ZERO_EVALUABLE; no sample_by_locus materialization",
        "missing_representation": "sparse sample_id/event_id rows",
        "geometry_control": "global loci axis shared by all samples; evaluated without individual expansion",
        "target_model": "log1p(total_asIBD_cM outside feature chromosome per pair); stratified sampled-zero rows",
        "map_support_policy": "exclude rare loci outside observed genetic-map support; never extrapolate cM",
        "zero_negative_ratio_sensitivity": list(ratios),
        "zero_negative_sampling": {
            str(ratio): target_balance(rows, ratio) for ratio, rows in target_sets.items()
        },
        "max_positive_pairs_per_target_stratum": max_positives_per_stratum,
        "n_analysis_samples": len(samples),
        "source_input_descriptors": {
            name: {"uri": str(path), "sha256": sha256_file(path)} for name, path in source_paths.items()
        },
    }
    (args.outdir / "m36_cora_materialization_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    try:
        run(parse_args())
    except MaterializeError as error:
        raise SystemExit(f"M36 materialization error: {error}") from error


if __name__ == "__main__":
    main()
