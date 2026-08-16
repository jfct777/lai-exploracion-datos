#!/usr/bin/env python3
"""Audit IBD blocks and rare-allele transferability before an LAI simulation.

M27E is deliberately descriptive and fail-closed.  It reconstructs the M27B
NatWGS-128 rare-allele bridge, summarizes autosomal Refined-IBD with the
published IBD-relatedness definition, and asks whether the frozen alleles occur
outside the discovery panel in distinct Native-American population/kinship
blocks.  It never assigns final split IDs, runs KING/LAI, trains a model, or
emits sample identifiers or variant keys.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

from audit_rare_scaffold_bridge import (
    MISSING_DOSAGE,
    audit_marker_panel,
    canonical_contig,
    is_biallelic_snv,
    parse_gt,
    parse_record,
    read_vcf_samples,
)


VariantKey = tuple[str, int, str, str]
TARGET_ANCESTRIES = ("African", "European", "Native_American")
CHROM_RE = re.compile(r"(?:chr_?|chromosome_?)(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibd-file", action="append", type=Path, required=True)
    parser.add_argument("--ibd-log", action="append", type=Path, required=True)
    parser.add_argument("--genetic-map", action="append", type=Path, required=True)
    parser.add_argument("--raw-wgs-vcf", type=Path, required=True)
    parser.add_argument("--phased-panel-vcf", type=Path, required=True)
    parser.add_argument("--gnomix-reference-vcf", type=Path, required=True)
    parser.add_argument("--resolved-strata", type=Path, required=True)
    parser.add_argument("--resolved-strata-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8"
    )


def chromosome_from_name(path: Path) -> int:
    matches = CHROM_RE.findall(path.name)
    if not matches:
        raise ValueError(f"Cannot identify chromosome from {path.name}")
    chromosome = int(matches[-1])
    if not 1 <= chromosome <= 22:
        raise ValueError(f"Non-autosomal chromosome in {path.name}")
    return chromosome


def indexed_by_chromosome(paths: Iterable[Path], label: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in paths:
        chromosome = chromosome_from_name(path)
        if chromosome in result:
            raise ValueError(f"Duplicate {label} for chromosome {chromosome}")
        result[chromosome] = path
    if set(result) != set(range(1, 23)):
        raise ValueError(f"{label} chromosomes are {sorted(result)}, expected 1..22")
    return result


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


@dataclass(frozen=True)
class RawRareSite:
    minor_is_alt: bool
    alt_dosages: bytes


@dataclass
class PairIbd:
    total_cm: float = 0.0
    max_cm: float = 0.0
    n_segments: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_resolved_strata(
    path: Path,
    manifest_path: Path,
    panel_samples: list[str],
    upstream: dict[str, object],
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    """Load the hash-pinned M27D identity result without inventing populations."""
    expected_table_hash = str(upstream["resolved_strata_sha256"])
    expected_manifest_hash = str(upstream["resolved_strata_manifest_sha256"])
    observed_table_hash = sha256_file(path)
    observed_manifest_hash = sha256_file(manifest_path)
    if observed_table_hash != expected_table_hash:
        raise ValueError("Resolved-strata SHA-256 differs from the preregistration")
    if observed_manifest_hash != expected_manifest_hash:
        raise ValueError("Resolved-strata manifest SHA-256 differs from the preregistration")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "M27D_SAMPLE_STRATA_RESOLUTION":
        raise ValueError("Unexpected resolved-strata manifest stage")
    if manifest.get("sha256", {}).get(path.name) != observed_table_hash:
        raise ValueError("Resolved-strata manifest does not authenticate the table")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "sample_id",
        "match_status",
        "population_interpretable",
        "Source",
        "Ancestry",
        "Population",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("Resolved-strata table lacks required columns")
    by_sample = {row["sample_id"]: row for row in rows}
    if len(by_sample) != len(rows) or set(by_sample) != set(panel_samples):
        raise ValueError("Resolved-strata identities differ from the phased-panel header")

    interpretable = [
        row
        for row in rows
        if row["match_status"] == "MATCHED"
        and row["population_interpretable"] == "TRUE"
    ]
    unresolved = [row for row in rows if row not in interpretable]
    expected_interpretable = int(upstream["expected_population_interpretable_samples"])
    expected_unresolved = int(upstream["expected_population_unresolved_samples"])
    if len(interpretable) != expected_interpretable or len(unresolved) != expected_unresolved:
        raise ValueError("Resolved-strata population coverage differs from the preregistration")
    if any(not all(row[column] for column in ("Source", "Ancestry", "Population")) for row in interpretable):
        raise ValueError("A population-interpretable row lacks a required stratum field")
    if any(row["match_status"] != "UNMATCHED" for row in unresolved):
        raise ValueError("Only documented unmatched rows may lack population interpretation")
    for row in rows:
        row["_population_interpretable"] = str(row in interpretable)
    return by_sample, {
        "n_panel_samples": len(rows),
        "n_population_interpretable": len(interpretable),
        "n_population_unresolved": len(unresolved),
        "n_ambiguous": sum(row["match_status"] == "AMBIGUOUS" for row in rows),
        "table_sha256": observed_table_hash,
        "manifest_sha256": observed_manifest_hash,
        "sample_ids_emitted": False,
    }


def parse_refined_ibd_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    version = re.search(r"(refined-ibd\.[^\s]+\.jar)", text)
    samples = re.search(r"Samples:\s+([\d,]+)", text)
    length = re.search(r"\n\s*length=([0-9.]+)", text)
    lod = re.search(r"\n\s*lod=([0-9.]+)", text)
    if not all((version, samples, length, lod)):
        raise ValueError(f"Incomplete Refined-IBD log {path.name}")
    return {
        "version": version.group(1),
        "n_samples": int(samples.group(1).replace(",", "")),
        "minimum_length_cm": float(length.group(1)),
        "minimum_lod": float(lod.group(1)),
    }


def read_map(path: Path, chromosome: int) -> tuple[list[int], list[float]]:
    positions: list[int] = []
    cms: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 3:
                chrom, position, cm = fields
            elif len(fields) >= 4:
                chrom, cm, position = fields[0], fields[2], fields[3]
            else:
                continue
            if canonical_contig(chrom) != str(chromosome):
                raise ValueError(f"Unexpected chromosome in {path.name}: {chrom}")
            positions.append(int(position))
            cms.append(float(cm))
    if len(positions) < 2 or positions != sorted(positions):
        raise ValueError(f"Invalid genetic map {path.name}")
    return positions, cms


def interpolate_cm(bp: int, positions: list[int], cms: list[float]) -> float:
    index = bisect.bisect_left(positions, bp)
    if index == 0:
        left, right = 0, 1
    elif index >= len(positions):
        left, right = len(positions) - 2, len(positions) - 1
    elif positions[index] == bp:
        return cms[index]
    else:
        left, right = index - 1, index
    span = positions[right] - positions[left]
    if span <= 0:
        raise ValueError("Non-increasing map positions")
    fraction = (bp - positions[left]) / span
    return cms[left] + fraction * (cms[right] - cms[left])


def read_ibd(
    ibd_files: dict[int, Path],
    panel_name_to_id: dict[str, str],
    contract: dict,
) -> tuple[dict[tuple[str, str], PairIbd], dict[int, tuple[int, int]], dict[str, object]]:
    pairs: dict[tuple[str, str], PairIbd] = {}
    endpoints: dict[int, tuple[int, int]] = {}
    observed_ids: set[str] = set()
    segments = duplicate_segments = 0
    segment_keys: set[tuple] = set()
    min_lod = math.inf
    min_cm = math.inf
    for chromosome, path in sorted(ibd_files.items()):
        with open_text(path) as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9:
                    raise ValueError(f"{path.name}:{line_number}: expected 9 fields")
                left_raw, right_raw = fields[0], fields[2]
                if left_raw not in panel_name_to_id or right_raw not in panel_name_to_id:
                    raise ValueError(f"{path.name}: IBD ID absent from phased panel")
                left, right = panel_name_to_id[left_raw], panel_name_to_id[right_raw]
                observed_ids.update((left, right))
                observed_chromosome = int(canonical_contig(fields[4]))
                if observed_chromosome != chromosome:
                    raise ValueError(f"{path.name}: chromosome mismatch")
                start, end = int(fields[5]), int(fields[6])
                lod, length_cm = float(fields[7]), float(fields[8])
                if end < start or lod < float(contract["reported_segment_min_lod"]) or length_cm < float(
                    contract["reported_segment_min_cm"]
                ):
                    raise ValueError(f"{path.name}:{line_number}: segment contract failed")
                segment_key = (
                    left_raw,
                    fields[1],
                    right_raw,
                    fields[3],
                    chromosome,
                    start,
                    end,
                )
                duplicate_segments += segment_key in segment_keys
                segment_keys.add(segment_key)
                key = tuple(sorted((left, right)))
                pair = pairs.setdefault(key, PairIbd())
                pair.total_cm += length_cm
                pair.max_cm = max(pair.max_cm, length_cm)
                pair.n_segments += 1
                previous = endpoints.get(chromosome)
                endpoints[chromosome] = (
                    start if previous is None else min(start, previous[0]),
                    end if previous is None else max(end, previous[1]),
                )
                min_lod, min_cm = min(min_lod, lod), min(min_cm, length_cm)
                segments += 1
    return pairs, endpoints, {
        "n_segments": segments,
        "n_unique_pairs": len(pairs),
        "n_observed_samples": len(observed_ids),
        "n_duplicate_segment_keys": duplicate_segments,
        "minimum_observed_lod": min_lod,
        "minimum_observed_length_cm": min_cm,
        "observed_ids": observed_ids,
    }


def autosomal_span_cm(
    endpoints: dict[int, tuple[int, int]], maps: dict[int, Path]
) -> tuple[float, dict[str, float]]:
    per_chromosome: dict[str, float] = {}
    for chromosome in range(1, 23):
        positions, cms = read_map(maps[chromosome], chromosome)
        start, end = endpoints[chromosome]
        span = interpolate_cm(end, positions, cms) - interpolate_cm(start, positions, cms)
        if span <= 0:
            raise ValueError(f"Non-positive genetic span for chromosome {chromosome}")
        per_chromosome[str(chromosome)] = span
    return sum(per_chromosome.values()), per_chromosome


def population_stratum(row: dict[str, str]) -> str:
    """Biological stratum; Source is batch provenance, not a split boundary."""
    return "|".join((row.get("Ancestry", ""), row.get("Population", "")))


def source_population_stratum(row: dict[str, str]) -> str:
    return "|".join((row.get("Ancestry", ""), row.get("Source", ""), row.get("Population", "")))


def is_usable_minor_carrier(
    alt_dosage: int, phased: bool, called: bool, minor_is_alt: bool
) -> bool:
    if not called:
        return False
    minor_dosage = alt_dosage if minor_is_alt else 2 - alt_dosage
    return minor_dosage > 0 and (alt_dosage != 1 or phased)


def build_blocks(
    samples: list[str],
    metadata: dict[str, dict[str, str]],
    pairs: dict[tuple[str, str], PairIbd],
    genome_cm: float,
    max_segment_floor: float,
    kinship_floor: float,
    union_populations: bool,
) -> tuple[dict[str, str], dict[str, object]]:
    uf = UnionFind(samples)
    if union_populations:
        first_by_stratum: dict[str, str] = {}
        for sample in samples:
            if metadata[sample]["_population_interpretable"] != "True":
                continue
            stratum = population_stratum(metadata[sample])
            if stratum in first_by_stratum:
                uf.union(sample, first_by_stratum[stratum])
            else:
                first_by_stratum[stratum] = sample
    n_kinship_edges = 0
    for (left, right), value in pairs.items():
        kinship = value.total_cm / (4.0 * genome_cm) if value.max_cm >= max_segment_floor else 0.0
        if kinship >= kinship_floor:
            uf.union(left, right)
            n_kinship_edges += 1
    roots = {sample: uf.find(sample) for sample in samples}
    sizes = Counter(roots.values())
    per_ancestry = {}
    for ancestry in TARGET_ANCESTRIES:
        ancestry_samples = [sample for sample in samples if metadata[sample]["Ancestry"] == ancestry]
        ancestry_roots = Counter(roots[sample] for sample in ancestry_samples)
        per_ancestry[ancestry] = {
            "n_samples": len(ancestry_samples),
            "n_blocks": len(ancestry_roots),
            "largest_block_samples": max(ancestry_roots.values(), default=0),
            "n_source_population_strata": len(
                {source_population_stratum(metadata[sample]) for sample in ancestry_samples}
            ),
            "n_canonical_population_strata": len(
                {population_stratum(metadata[sample]) for sample in ancestry_samples}
            ),
        }
    return roots, {
        "max_segment_floor_cm": max_segment_floor,
        "kinship_floor": kinship_floor,
        "union_canonical_population_strata": union_populations,
        "n_kinship_edges": n_kinship_edges,
        "n_blocks_total": len(sizes),
        "largest_block_samples": max(sizes.values(), default=0),
        "per_ancestry": per_ancestry,
    }


def raw_rare_sites(
    path: Path, expected_contig: str, expected_samples: int, minimum_mac: int, maf_threshold: float
) -> dict[VariantKey, RawRareSite]:
    sites: dict[VariantKey, RawRareSite] = {}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, path)
            if key[0] != expected_contig or not is_biallelic_snv(key[2], key[3]):
                continue
            sample_fields = fields[9:]
            if len(sample_fields) != expected_samples:
                raise ValueError("Unexpected NatWGS sample-field count")
            alt = [parse_gt(value)[0] for value in sample_fields]
            called = [value for value in alt if value != MISSING_DOSAGE]
            an = 2 * len(called)
            alt_ac = sum(called)
            ref_ac = an - alt_ac
            minor_ac = min(alt_ac, ref_ac) if an else 0
            if not an or minor_ac < minimum_mac or minor_ac / an >= maf_threshold:
                continue
            if key in sites:
                raise ValueError(f"Duplicate rare key {key}")
            sites[key] = RawRareSite(alt_ac <= ref_ac, bytes(alt))
    return sites


def summarize_panel_bridge(
    path: Path,
    panel_samples: list[str],
    panel_ids: list[str],
    raw_ids: list[str],
    raw_sites: dict[VariantKey, RawRareSite],
    metadata: dict[str, dict[str, str]],
    block_roots: dict[str, dict[str, str]],
    baseline_markers: set[VariantKey],
    primary_policy: str,
) -> dict[str, object]:
    panel_index = {sample: index for index, sample in enumerate(panel_ids)}
    raw_indices = [panel_index[sample] for sample in raw_ids]
    discovery = set(raw_ids)
    discovery_populations = {
        population_stratum(metadata[sample]) for sample in discovery
    }
    discovery_blocks_by_policy = {
        policy: {roots[sample] for sample in discovery}
        for policy, roots in block_roots.items()
    }
    direct_rows: list[dict[str, object]] = []
    n_exact = 0
    called_by_ancestry = Counter()
    possible_by_ancestry = Counter()
    unphased_heterozygous_carriers_excluded = Counter()
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields, key = parse_record(line, path)
            raw_site = raw_sites.get(key)
            if raw_site is None:
                continue
            n_exact += 1
            parsed = [parse_gt(value) for value in fields[9:]]
            if len(parsed) != len(panel_samples):
                raise ValueError("Unexpected phased-panel sample-field count")
            direct = False
            concordant = True
            for raw_offset, panel_offset in enumerate(raw_indices):
                raw_alt = raw_site.alt_dosages[raw_offset]
                minor = raw_alt if raw_site.minor_is_alt else (
                    MISSING_DOSAGE if raw_alt == MISSING_DOSAGE else 2 - raw_alt
                )
                if minor in (0, MISSING_DOSAGE):
                    continue
                panel_alt, phased, called = parsed[panel_offset]
                if not called or panel_alt != raw_alt or (raw_alt == 1 and not phased):
                    concordant = False
                    break
                direct = direct or raw_alt == 1
            if not concordant or not direct:
                continue
            carriers_by_ancestry: dict[str, list[str]] = defaultdict(list)
            for sample, (alt_dosage, phased, called) in zip(panel_ids, parsed):
                ancestry = metadata[sample]["Ancestry"]
                if ancestry in TARGET_ANCESTRIES:
                    possible_by_ancestry[ancestry] += 1
                    called_by_ancestry[ancestry] += called
                if not called:
                    continue
                minor_dosage = alt_dosage if raw_site.minor_is_alt else 2 - alt_dosage
                if minor_dosage > 0 and alt_dosage == 1 and not phased:
                    unphased_heterozygous_carriers_excluded[ancestry] += 1
                    continue
                if is_usable_minor_carrier(
                    alt_dosage, phased, called, raw_site.minor_is_alt
                ):
                    carriers_by_ancestry[ancestry].append(sample)
            row: dict[str, object] = {
                "key": key,
                "minor_is_alt": raw_site.minor_is_alt,
                "in_frozen_baseline": key in baseline_markers,
                "ancestry": {},
            }
            for ancestry in TARGET_ANCESTRIES:
                carriers = carriers_by_ancestry.get(ancestry, [])
                external = [sample for sample in carriers if sample not in discovery]
                discovered = [sample for sample in carriers if sample in discovery]
                policy_counts = {}
                for policy, roots in block_roots.items():
                    discovery_blocks = {roots[sample] for sample in discovered}
                    external_blocks_raw = {roots[sample] for sample in external}
                    external_role_eligible_samples = [
                        sample
                        for sample in external
                        if roots[sample] not in discovery_blocks_by_policy[policy]
                        and population_stratum(metadata[sample])
                        not in discovery_populations
                    ]
                    external_role_eligible_blocks = {
                        roots[sample] for sample in external_role_eligible_samples
                    }
                    external_blocks_by_population: dict[str, set[str]] = defaultdict(set)
                    for sample in external_role_eligible_samples:
                        root = roots[sample]
                        external_blocks_by_population[
                            population_stratum(metadata[sample])
                        ].add(root)
                    minimum_after_leave_one_population_out = min(
                        (
                            len(external_role_eligible_blocks - population_blocks)
                            for population_blocks in external_blocks_by_population.values()
                        ),
                        default=0,
                    )
                    policy_counts[policy] = {
                        "discovery_blocks": len(discovery_blocks),
                        "external_blocks_raw": len(external_blocks_raw),
                        "external_role_eligible_samples": len(
                            external_role_eligible_samples
                        ),
                        "external_role_eligible_blocks": len(
                            external_role_eligible_blocks
                        ),
                        "minimum_external_blocks_after_leave_one_population_out": (
                            minimum_after_leave_one_population_out
                        ),
                        "external_block_ids_private": external_role_eligible_blocks,
                        "external_population_ids_private": set(external_blocks_by_population),
                    }
                row["ancestry"][ancestry] = {
                    "n_carrier_samples": len(carriers),
                    "n_discovery_carriers": len(discovered),
                    "n_external_carriers": len(external),
                    "n_discovery_populations": len(
                        {population_stratum(metadata[sample]) for sample in discovered}
                    ),
                    "n_external_populations": len(
                        {population_stratum(metadata[sample]) for sample in external}
                    ),
                    "policies": policy_counts,
                }
            direct_rows.append(row)

    digest_lines = [
        f"{chrom}:{position}:{ref}:{alt}|minor={'ALT' if row['minor_is_alt'] else 'REF'}"
        for row in direct_rows
        for chrom, position, ref, alt in [row["key"]]
    ]
    digest = hashlib.sha256(("\n".join(sorted(digest_lines)) + "\n").encode()).hexdigest()
    summary_by_policy = {}
    for policy in block_roots:
        per_ancestry = {}
        for ancestry in TARGET_ANCESTRIES:
            transferable_rows = []
            ceiling = baseline_overlap = baseline_disjoint = lopo_robust = 0
            for row in direct_rows:
                counts = row["ancestry"][ancestry]["policies"][policy]
                ceiling += counts["discovery_blocks"] >= 1 and counts[
                    "external_role_eligible_blocks"
                ] >= 1
                if counts["discovery_blocks"] >= 2 and counts[
                    "external_role_eligible_blocks"
                ] >= 2:
                    transferable_rows.append(row)
                    baseline_overlap += bool(row["in_frozen_baseline"])
                    baseline_disjoint += not bool(row["in_frozen_baseline"])
                    lopo_robust += counts[
                        "minimum_external_blocks_after_leave_one_population_out"
                    ] >= 2
            per_ancestry[ancestry] = {
                "n_sites_with_one_fit_and_one_external_block_ceiling": ceiling,
                "n_sites_with_two_fit_and_two_external_blocks": len(transferable_rows),
                "n_transferable_sites_in_frozen_baseline": baseline_overlap,
                "n_transferable_sites_outside_frozen_baseline": baseline_disjoint,
                "n_transferable_sites_leave_one_external_population_out_robust": lopo_robust,
            }
        summary_by_policy[policy] = per_ancestry

    direct_in_baseline = sum(bool(row["in_frozen_baseline"]) for row in direct_rows)
    direct_outside_baseline = len(direct_rows) - direct_in_baseline
    baseline_positions = {(chrom, position) for chrom, position, _ref, _alt in baseline_markers}
    direct_at_baseline_position = sum(
        (row["key"][0], row["key"][1]) in baseline_positions for row in direct_rows
    )
    primary_nam_support = summary_by_policy[primary_policy]["Native_American"]
    transferable_in_baseline = primary_nam_support[
        "n_transferable_sites_in_frozen_baseline"
    ]
    transferable_outside_baseline = primary_nam_support[
        "n_transferable_sites_outside_frozen_baseline"
    ]
    not_transferable_in_baseline = direct_in_baseline - transferable_in_baseline
    not_transferable_outside_baseline = (
        direct_outside_baseline - transferable_outside_baseline
    )
    contingency = {
        "transferable": {
            "in_frozen_baseline": transferable_in_baseline,
            "outside_frozen_baseline": transferable_outside_baseline,
            "total": transferable_in_baseline + transferable_outside_baseline,
        },
        "not_transferable": {
            "in_frozen_baseline": not_transferable_in_baseline,
            "outside_frozen_baseline": not_transferable_outside_baseline,
            "total": not_transferable_in_baseline + not_transferable_outside_baseline,
        },
        "column_totals": {
            "in_frozen_baseline": direct_in_baseline,
            "outside_frozen_baseline": direct_outside_baseline,
        },
        "grand_total": len(direct_rows),
    }
    cell_total = sum(
        contingency[row][column]
        for row in ("transferable", "not_transferable")
        for column in ("in_frozen_baseline", "outside_frozen_baseline")
    )
    if min(
        transferable_in_baseline,
        transferable_outside_baseline,
        not_transferable_in_baseline,
        not_transferable_outside_baseline,
    ) < 0 or cell_total != len(direct_rows):
        raise ValueError("Transferability-by-baseline contingency invariants failed")
    contingency["invariants"] = {
        "four_cells_sum_to_direct_bridge": cell_total == len(direct_rows),
        "transferable_margin_matches_primary_support": contingency["transferable"][
            "total"
        ]
        == primary_nam_support["n_sites_with_two_fit_and_two_external_blocks"],
        "baseline_margin_matches_direct_overlap": contingency["column_totals"][
            "in_frozen_baseline"
        ]
        == direct_in_baseline,
    }
    if not all(contingency["invariants"].values()):
        raise ValueError("Transferability-by-baseline margins failed")

    primary_nam_rows = []
    for row in direct_rows:
        counts = row["ancestry"]["Native_American"]["policies"][primary_policy]
        if (
            counts["discovery_blocks"] >= 2
            and counts["external_role_eligible_blocks"] >= 2
        ):
            primary_nam_rows.append(row)
    block_support = Counter()
    population_support = Counter()
    lopo_robust = 0
    for row in primary_nam_rows:
        counts = row["ancestry"]["Native_American"]["policies"][primary_policy]
        block_support.update(counts["external_block_ids_private"])
        population_support.update(counts["external_population_ids_private"])
        lopo_robust += counts[
            "minimum_external_blocks_after_leave_one_population_out"
        ] >= 2
    block_total = sum(block_support.values())
    population_total = sum(population_support.values())
    concentration = {
        "n_primary_transferable_sites": len(primary_nam_rows),
        "n_sites_robust_to_leave_one_external_population_out": (
            lopo_robust
        ),
        "fraction_robust_among_primary_transferable_sites": (
            lopo_robust / len(primary_nam_rows)
            if primary_nam_rows
            else None
        ),
        "fraction_robust_among_all_direct_bridge_sites": (
            lopo_robust / len(direct_rows) if direct_rows else None
        ),
        "n_contributing_external_blocks": len(block_support),
        "effective_external_blocks_by_site_support": effective_number(block_support.values()),
        "largest_external_block_share_of_site_block_support": (
            max(block_support.values(), default=0) / block_total if block_total else None
        ),
        "n_contributing_external_populations": len(population_support),
        "effective_external_populations_by_site_support": effective_number(
            population_support.values()
        ),
        "largest_external_population_share_of_site_population_support": (
            max(population_support.values(), default=0) / population_total
            if population_total
            else None
        ),
        "population_or_block_labels_emitted": False,
        "sites_are_not_independent_replicates": True,
    }
    return {
        "n_raw_rare_sites": len(raw_sites),
        "n_rare_exact_panel_sites": n_exact,
        "n_direct_phase_bridge_sites": len(direct_rows),
        "direct_bridge_key_orientation_sha256": digest,
        "variant_keys_emitted": False,
        "sample_ids_emitted": False,
        "n_direct_bridge_sites_in_frozen_baseline": direct_in_baseline,
        "n_direct_bridge_sites_outside_frozen_baseline": direct_outside_baseline,
        "n_direct_bridge_sites_at_frozen_baseline_position": direct_at_baseline_position,
        "n_direct_bridge_sites_same_position_but_not_exact_key": (
            direct_at_baseline_position - direct_in_baseline
        ),
        "primary_native_american_transferability_by_baseline": contingency,
        "unphased_heterozygous_carriers_excluded_by_ancestry": {
            ancestry: unphased_heterozygous_carriers_excluded[ancestry]
            for ancestry in TARGET_ANCESTRIES
        },
        "call_rate_over_direct_sites_by_ancestry": {
            ancestry: called_by_ancestry[ancestry] / possible_by_ancestry[ancestry]
            if possible_by_ancestry[ancestry]
            else None
            for ancestry in TARGET_ANCESTRIES
        },
        "support_by_policy": summary_by_policy,
        "primary_policy": primary_policy,
        "primary_native_american_transferable_sites": summary_by_policy[primary_policy][
            "Native_American"
        ]["n_sites_with_two_fit_and_two_external_blocks"],
        "primary_native_american_transferable_sites_in_frozen_baseline": summary_by_policy[
            primary_policy
        ]["Native_American"]["n_transferable_sites_in_frozen_baseline"],
        "primary_native_american_transferable_sites_outside_frozen_baseline": summary_by_policy[
            primary_policy
        ]["Native_American"]["n_transferable_sites_outside_frozen_baseline"],
        "primary_native_american_lopo_robust_sites": lopo_robust,
        "primary_native_american_transferable_concentration": concentration,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gates(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "name", "status", "reason"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def effective_number(counts: Iterable[int]) -> float:
    values = [int(value) for value in counts if int(value) > 0]
    total = sum(values)
    return (total * total / sum(value * value for value in values)) if total else 0.0


def run(args: argparse.Namespace) -> dict[str, object]:
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("stage") != "M27E_IBD_RARE_TRANSFER_FEASIBILITY":
        raise ValueError("Invalid M27E preregistration")
    upstream = prereg["upstream_contract"]
    ibd_contract = prereg["ibd_contract"]
    rare_contract = prereg["rare_contract"]
    ibd_files = indexed_by_chromosome(args.ibd_file, "IBD file")
    ibd_logs = indexed_by_chromosome(args.ibd_log, "IBD log")
    maps = indexed_by_chromosome(args.genetic_map, "genetic map")

    panel_samples = read_vcf_samples(args.phased_panel_vcf)
    if len(panel_samples) != int(upstream["expected_panel_samples"]) or len(set(panel_samples)) != len(panel_samples):
        raise ValueError("Phased-panel header does not contain 3,685 unique samples")
    metadata, strata_receipt = read_resolved_strata(
        args.resolved_strata,
        args.resolved_strata_manifest,
        panel_samples,
        upstream,
    )
    panel_ids = panel_samples
    panel_name_to_id = {sample: sample for sample in panel_samples}
    raw_samples = read_vcf_samples(args.raw_wgs_vcf)
    raw_ids = raw_samples
    if len(raw_ids) != int(upstream["expected_discovery_samples"]) or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("NatWGS header does not contain 128 unique discovery samples")
    if not set(raw_ids) <= set(panel_ids):
        raise ValueError("NatWGS discovery samples are not a subset of the phased panel")
    if any(
        metadata[sample]["_population_interpretable"] != "True"
        or metadata[sample]["Source"] != "NatWGS"
        or metadata[sample]["Ancestry"] != "Native_American"
        for sample in raw_ids
    ):
        raise ValueError("NatWGS discovery samples do not resolve to the frozen Native-American stratum")

    log_receipts = {str(chrom): parse_refined_ibd_log(path) for chrom, path in ibd_logs.items()}
    log_contract_ok = all(
        receipt == {
            "version": ibd_contract["software"],
            "n_samples": int(upstream["expected_panel_samples"]),
            "minimum_length_cm": float(ibd_contract["reported_segment_min_cm"]),
            "minimum_lod": float(ibd_contract["reported_segment_min_lod"]),
        }
        for receipt in log_receipts.values()
    )
    pairs, endpoints, ibd_summary = read_ibd(
        ibd_files, panel_name_to_id, ibd_contract
    )
    observed_ids = ibd_summary.pop("observed_ids")
    genome_cm, per_chromosome_cm = autosomal_span_cm(endpoints, maps)

    raw_component_roots, raw_component_summary = build_blocks(
        panel_ids, metadata, pairs, genome_cm, 0.0, 0.0, False
    )
    del raw_component_roots
    max_segment_floors = sorted(
        {float(ibd_contract["recent_relative_max_segment_cm"]["primary"])}
        | {float(value) for value in ibd_contract["recent_relative_max_segment_cm"]["diagnostic"]}
    )
    kinship_thresholds = {
        name: float(value) for name, value in ibd_contract["kinship_thresholds"].items()
    }
    block_roots: dict[str, dict[str, str]] = {}
    block_summaries: dict[str, object] = {}
    for max_floor in max_segment_floors:
        for threshold_name, kinship_floor in kinship_thresholds.items():
            name = f"maxseg_{max_floor:g}cm__{threshold_name}"
            roots, summary = build_blocks(
                panel_ids,
                metadata,
                pairs,
                genome_cm,
                max_floor,
                kinship_floor,
                True,
            )
            block_roots[name] = roots
            block_summaries[name] = summary
    primary_policy = "maxseg_10cm__primary_third_fourth_midpoint"
    if primary_policy not in block_roots:
        raise ValueError("Primary block policy was not generated")

    raw_sites = raw_rare_sites(
        args.raw_wgs_vcf,
        str(rare_contract["chromosome"]),
        int(upstream["expected_discovery_samples"]),
        int(rare_contract["minimum_mac"]),
        float(rare_contract["maf_threshold"]),
    )
    baseline = audit_marker_panel(args.gnomix_reference_vcf, str(rare_contract["chromosome"]))
    rare_summary = summarize_panel_bridge(
        args.phased_panel_vcf,
        panel_samples,
        panel_ids,
        raw_ids,
        raw_sites,
        metadata,
        block_roots,
        baseline.markers,
        primary_policy,
    )

    e0 = log_contract_ok and all(
        (
            ibd_summary["n_duplicate_segment_keys"] == 0,
            ibd_summary["minimum_observed_lod"] >= float(ibd_contract["reported_segment_min_lod"]),
            ibd_summary["minimum_observed_length_cm"] >= float(ibd_contract["reported_segment_min_cm"]),
        )
    )
    e1 = all(
        (
            len(panel_ids) == int(upstream["expected_panel_samples"]),
            len(observed_ids) == len(panel_ids),
            observed_ids == set(panel_ids),
            len(raw_ids) == int(upstream["expected_discovery_samples"]),
        )
    )
    e2 = rare_summary["n_direct_phase_bridge_sites"] == int(
        upstream["expected_direct_phase_bridge_sites"]
    )
    primary_blocks = block_summaries[primary_policy]["per_ancestry"]
    e3 = all(primary_blocks[ancestry]["n_blocks"] >= 3 for ancestry in TARGET_ANCESTRIES)
    e4 = rare_summary["primary_native_american_transferable_sites"] > 0
    e5 = all(
        rare_summary["primary_native_american_transferability_by_baseline"][
            "invariants"
        ].values()
    )
    e6 = rare_summary["primary_native_american_lopo_robust_sites"] > 0
    gates = [
        {"gate": "E0", "name": "input_and_provenance_contract", "status": "PASS" if e0 else "FAIL", "reason": "22 matched IBD/log/map inputs; frozen Refined-IBD version and segment schema."},
        {"gate": "E1", "name": "identity_and_population_coverage_contract", "status": "PASS" if e1 else "FAIL", "reason": "Panel, IBD and NatWGS identities match the hash-pinned M27D strata; the 10 documented unmatched samples are excluded only from ancestry summaries."},
        {"gate": "E2", "name": "rare_bridge_reproduction", "status": "PASS" if e2 else "FAIL", "reason": "M27B rare definition and direct phased-carrier bridge reproduced with private-key digest."},
        {"gate": "E3", "name": "three_role_block_feasibility", "status": "PASS" if e3 else "FAIL", "reason": "At least three canonical-population/recent-kinship blocks remain per target ancestry; a joint role assignment is still pending."},
        {"gate": "E4", "name": "external_native_american_rare_support", "status": "PASS" if e4 else "FAIL", "reason": "Frozen NatWGS allele appears in >=2 discovery and >=2 external NAM blocks under the primary policy."},
        {"gate": "E5", "name": "baseline_overlap_audit", "status": "PASS" if e5 else "FAIL", "reason": "Exact-key and same-position overlap with the frozen baseline were measured with reconstructable margins; this does not certify the future baseline."},
        {"gate": "E6", "name": "external_population_concentration", "status": "PASS" if e6 else "FAIL", "reason": "At least one primary site retains two external role-eligible blocks after leaving out each carrier biological population."},
    ]
    if not e0 or not e1:
        decision = "STOP_INPUT_OR_IDENTITY_CONTRACT"
    elif not e2:
        decision = "STOP_RARE_BRIDGE_NOT_REPRODUCED"
    elif not e3:
        decision = "STOP_NO_THREE_ROLE_BLOCK_FEASIBILITY"
    elif not e4:
        decision = "STOP_NO_TRANSFERABLE_NAM_SUPPORT"
    elif not e6:
        decision = "STOP_EXTERNAL_POPULATION_CONCENTRATION"
    else:
        decision = "GO_BASELINE_REDESIGN_AND_POWER_ONLY"

    input_contract = {
        "stage": prereg["stage"],
        "n_ibd_files": len(ibd_files),
        "n_ibd_logs": len(ibd_logs),
        "n_genetic_maps": len(maps),
        "log_contract_pass": log_contract_ok,
        "log_receipts": log_receipts,
        "resolved_strata_contract": strata_receipt,
        "n_panel_samples": len(panel_ids),
        "n_discovery_samples": len(raw_ids),
        "n_discovery_population_interpretable_native_american": sum(
            metadata[sample]["_population_interpretable"] == "True"
            and metadata[sample]["Source"] == "NatWGS"
            and metadata[sample]["Ancestry"] == "Native_American"
            for sample in raw_ids
        ),
        "sample_ids_emitted": False,
    }
    relatedness = {
        **ibd_summary,
        "observed_autosomal_map_span_cm": genome_cm,
        "per_chromosome_observed_map_span_cm": per_chromosome_cm,
        "kinship_definition": ibd_contract["kinship_definition"],
        "raw_any_reported_segment_component_diagnostic": raw_component_summary,
        "raw_component_is_not_pedigree": True,
        "block_policies": block_summaries,
        "primary_policy": primary_policy,
        "sample_ids_emitted": False,
    }
    summary = {
        "stage": prereg["stage"],
        "decision": decision,
        "gates": {row["gate"]: row["status"] for row in gates},
        "simulation_performed": False,
        "lai_inference_performed": False,
        "model_training_performed": False,
        "king_executed": False,
        "pcrelate_executed": False,
        "source_test_opened": False,
        "sample_ids_emitted": False,
        "variant_keys_emitted": False,
        "interpretation": "M27E establishes structural transfer feasibility using external role-eligible blocks. Baseline overlap is diagnostic only; it cannot certify pedigree, power, a final split, a compatible common baseline, or rare-channel improvement of LAI.",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "m27e_input_contract.json", input_contract)
    write_json(args.outdir / "m27e_ibd_relatedness_summary.json", relatedness)
    write_json(args.outdir / "m27e_rare_transfer_support.json", rare_summary)
    write_gates(args.outdir / "m27e_gates.tsv", gates)
    write_json(args.outdir / "m27e_summary.json", summary)
    return summary


if __name__ == "__main__":
    run(parse_args())
