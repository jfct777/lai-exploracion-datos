#!/usr/bin/env python3
"""Build exploratory phased chr22 mosaics with exact ancestry truth.

The generator copies alleles from phased donor haplotypes selected by the M27F
role table.  It keeps only a small segment plan in memory and streams the source
VCF twice: once to validate its genomic extent and once to write the targets.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


ANCESTRY_MAP = {
    "African": "AFR",
    "European": "EUR",
    "Native_American": "NAM",
    "AFR": "AFR",
    "EUR": "EUR",
    "NAM": "NAM",
}
ANCESTRIES = ("AFR", "EUR", "NAM")
DEFAULT_FORBIDDEN_ROLES = ("REF_TRAIN", "SOURCE_TEST")


@dataclass(frozen=True)
class GeneticMap:
    chrom: str
    positions_bp: tuple[int, ...]
    positions_cm: tuple[float, ...]

    @property
    def start_bp(self) -> int:
        return self.positions_bp[0]

    @property
    def end_bp(self) -> int:
        return self.positions_bp[-1]

    def bp_to_cm(self, position: int) -> float:
        if not self.start_bp <= position <= self.end_bp:
            raise ValueError(
                f"Position {position} is outside genetic map "
                f"[{self.start_bp}, {self.end_bp}]"
            )
        right = bisect.bisect_right(self.positions_bp, position)
        if right == 0:
            return self.positions_cm[0]
        if right >= len(self.positions_bp):
            return self.positions_cm[-1]
        left = right - 1
        bp0, bp1 = self.positions_bp[left], self.positions_bp[right]
        cm0, cm1 = self.positions_cm[left], self.positions_cm[right]
        fraction = (position - bp0) / (bp1 - bp0)
        return cm0 + fraction * (cm1 - cm0)

    def cm_to_bp(self, value: float) -> int:
        if not self.positions_cm[0] <= value <= self.positions_cm[-1]:
            raise ValueError(f"cM {value} is outside the genetic map")
        right = bisect.bisect_right(self.positions_cm, value)
        if right == 0:
            return self.positions_bp[0]
        if right >= len(self.positions_cm):
            return self.positions_bp[-1]
        left = right - 1
        cm0, cm1 = self.positions_cm[left], self.positions_cm[right]
        bp0, bp1 = self.positions_bp[left], self.positions_bp[right]
        if cm1 == cm0:
            return bp0
        fraction = (value - cm0) / (cm1 - cm0)
        return int(round(bp0 + fraction * (bp1 - bp0)))


@dataclass(frozen=True)
class Donor:
    sample_id: str
    ancestry: str
    atomic_unit_id: str
    haplotype: int


@dataclass(frozen=True)
class Segment:
    target_id: str
    haplotype: int
    start_bp: int
    end_bp_exclusive: int
    ancestry: str
    donor: Donor


@dataclass(frozen=True)
class VcfSummary:
    samples: tuple[str, ...]
    chrom: str
    first_position: int
    last_position: int
    records: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_contig(value: str) -> str:
    contig = value.strip()
    if contig.lower().startswith("chr"):
        contig = contig[3:]
    return contig.upper()


@contextmanager
def open_text(path: Path) -> Iterator[TextIO]:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                yield handle


def read_genetic_map(path: Path, expected_chrom: str) -> GeneticMap:
    """Read either CHROM/BP/cM or PLINK CHROM/ID/cM/BP rows."""
    chroms: list[str] = []
    positions: list[int] = []
    cms: list[float] = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            try:
                if len(fields) == 3:
                    chrom, position, cm = fields
                elif len(fields) >= 4:
                    chrom, cm, position = fields[0], fields[2], fields[3]
                else:
                    raise ValueError
                parsed_position = int(position)
                parsed_cm = float(cm)
            except ValueError:
                if not positions:
                    continue
                raise ValueError(f"Malformed genetic-map row {line_number}") from None
            chroms.append(canonical_contig(chrom))
            positions.append(parsed_position)
            cms.append(parsed_cm)
    if len(positions) < 2:
        raise ValueError("Genetic map needs at least two data rows")
    expected = canonical_contig(expected_chrom)
    if set(chroms) != {expected}:
        raise ValueError("Genetic-map chromosome differs from the requested chromosome")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise ValueError("Genetic-map base-pair positions are not strictly increasing")
    if any(left > right for left, right in zip(cms, cms[1:])):
        raise ValueError("Genetic-map cM positions decrease")
    if not cms[-1] > cms[0]:
        raise ValueError("Genetic map has zero cM span")
    return GeneticMap(expected, tuple(positions), tuple(cms))


def scan_vcf(path: Path, expected_chrom: str) -> VcfSummary:
    samples: tuple[str, ...] | None = None
    first_position: int | None = None
    last_position: int | None = None
    records = 0
    observed_chrom: str | None = None
    previous_position = -1
    expected = canonical_contig(expected_chrom)
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                if samples is not None:
                    raise ValueError("VCF contains more than one #CHROM header")
                header = line.rstrip("\n").split("\t")
                if len(header) < 10 or header[8] != "FORMAT":
                    raise ValueError("VCF must contain FORMAT and phased donor samples")
                samples = tuple(header[9:])
                if len(samples) != len(set(samples)):
                    raise ValueError("VCF sample names are duplicated")
                continue
            if line.startswith("#"):
                continue
            if samples is None:
                raise ValueError("VCF data appeared before #CHROM header")
            fields = line.rstrip("\n").split("\t", 9)
            if len(fields) < 10:
                raise ValueError(f"Malformed VCF row {line_number}")
            chrom = canonical_contig(fields[0])
            if chrom != expected:
                raise ValueError(f"Unexpected VCF chromosome {fields[0]}")
            position = int(fields[1])
            if position < previous_position:
                raise ValueError("VCF positions are not sorted")
            previous_position = position
            observed_chrom = chrom
            first_position = position if first_position is None else first_position
            last_position = position
            records += 1
    if samples is None or records == 0 or first_position is None or last_position is None:
        raise ValueError("VCF has no usable samples or records")
    return VcfSummary(samples, observed_chrom or expected, first_position, last_position, records)


def load_split(
    path: Path,
    donor_role: str,
    forbidden_roles: tuple[str, ...],
    unit_partition: str = "all",
    rotation: int = 0,
    fit_fraction: float = 2.0 / 3.0,
) -> tuple[dict[str, list[Donor]], dict[str, object]]:
    required = {"sample_id", "ancestry", "atomic_unit_id", "role"}
    rows: list[dict[str, str]] = []
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Split TSV lacks required fields: {sorted(required)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("Split TSV is empty")
    sample_ids = [row["sample_id"] for row in rows]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Split sample_id values are empty or duplicated")
    if any(not row["atomic_unit_id"] for row in rows):
        raise ValueError("Split contains an empty atomic_unit_id")

    roles_by_unit: dict[str, set[str]] = {}
    for row in rows:
        roles_by_unit.setdefault(row["atomic_unit_id"], set()).add(row["role"])
    selected_units = {
        row["atomic_unit_id"] for row in rows if row["role"] == donor_role
    }
    forbidden = set(forbidden_roles)
    crossing = {
        unit: sorted(roles_by_unit[unit] & forbidden)
        for unit in selected_units
        if roles_by_unit[unit] & forbidden
    }
    if crossing:
        raise ValueError("Donor atomic units cross a forbidden role")

    if unit_partition not in {"all", "fit", "valid"}:
        raise ValueError("Donor unit partition must be all, fit or valid")
    if type(rotation) is not int or rotation < 0:
        raise ValueError("Rotation must be a non-negative integer")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("FIT unit fraction must lie strictly between zero and one")

    eligible_units_by_ancestry: dict[str, set[str]] = {
        ancestry: set() for ancestry in ANCESTRIES
    }
    for row in rows:
        if row["role"] != donor_role:
            continue
        ancestry = ANCESTRY_MAP.get(row["ancestry"])
        if ancestry is None:
            raise ValueError(f"Unsupported donor ancestry {row['ancestry']!r}")
        eligible_units_by_ancestry[ancestry].add(row["atomic_unit_id"])

    selected_units_by_ancestry: dict[str, set[str]] = {}
    for ancestry in ANCESTRIES:
        units = eligible_units_by_ancestry[ancestry]
        if unit_partition == "all":
            selected_units_by_ancestry[ancestry] = set(units)
            continue
        if len(units) < 2:
            raise ValueError(
                f"Ancestry {ancestry} needs at least two donor units for FIT/VALID separation"
            )
        ordered = sorted(
            units,
            key=lambda unit: hashlib.sha256(
                f"M34|UNIT_PARTITION|R{rotation}|{ancestry}|{unit}".encode("utf-8")
            ).digest(),
        )
        fit_count = max(1, min(len(ordered) - 1, round(len(ordered) * fit_fraction)))
        fit_units = set(ordered[:fit_count])
        selected_units_by_ancestry[ancestry] = (
            fit_units if unit_partition == "fit" else set(ordered[fit_count:])
        )

    donors: dict[str, list[Donor]] = {ancestry: [] for ancestry in ANCESTRIES}
    selected_people_ids: set[str] = set()
    for row in rows:
        if row["role"] != donor_role:
            continue
        ancestry = ANCESTRY_MAP.get(row["ancestry"])
        if ancestry is None:
            raise ValueError(f"Unsupported donor ancestry {row['ancestry']!r}")
        if row["atomic_unit_id"] not in selected_units_by_ancestry[ancestry]:
            continue
        selected_people_ids.add(row["sample_id"])
        donors[ancestry].extend(
            Donor(row["sample_id"], ancestry, row["atomic_unit_id"], haplotype)
            for haplotype in (0, 1)
        )
    missing = [ancestry for ancestry, values in donors.items() if not values]
    if missing:
        raise ValueError(f"Donor role lacks ancestries: {','.join(missing)}")
    audit = {
        "donor_role": donor_role,
        "forbidden_roles": list(forbidden_roles),
        "selected_people": len(selected_people_ids),
        "selected_atomic_units": sum(
            len(units) for units in selected_units_by_ancestry.values()
        ),
        "unit_partition": unit_partition,
        "unit_partition_rotation": rotation,
        "unit_fit_fraction": fit_fraction,
        "atomic_units_crossing_forbidden_roles": 0,
        "donor_people_by_ancestry": {
            ancestry: len(values) // 2 for ancestry, values in donors.items()
        },
        "donor_haplotypes_by_ancestry": {
            ancestry: len(values) for ancestry, values in donors.items()
        },
        "donor_atomic_units_by_ancestry": {
            ancestry: len(selected_units_by_ancestry[ancestry])
            for ancestry in ANCESTRIES
        },
    }
    return donors, audit


def parse_mixture_proportions(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError("Mixture must use AFR=x,EUR=y,NAM=z")
        key, raw = item.split("=", 1)
        key = key.strip().upper()
        if key in result:
            raise argparse.ArgumentTypeError(f"Repeated ancestry {key}")
        try:
            result[key] = float(raw)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"Invalid proportion {raw!r}") from error
    if set(result) != set(ANCESTRIES):
        raise argparse.ArgumentTypeError("Mixture must define AFR, EUR and NAM exactly once")
    if any(not math.isfinite(item) or item <= 0 for item in result.values()):
        raise argparse.ArgumentTypeError("All mixture proportions must be finite and positive")
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise argparse.ArgumentTypeError("Mixture proportions must sum to one")
    return result


def choose_ancestry(rng: random.Random, proportions: dict[str, float]) -> str:
    draw = rng.random()
    cumulative = 0.0
    for ancestry in ANCESTRIES:
        cumulative += proportions[ancestry]
        if draw < cumulative:
            return ancestry
    return ANCESTRIES[-1]


def choose_unit_balanced_donor(rng: random.Random, donors: list[Donor]) -> Donor:
    """Sample an IBD/population unit first, then a haplotype within that unit."""
    by_unit: dict[str, list[Donor]] = {}
    for donor in donors:
        by_unit.setdefault(donor.atomic_unit_id, []).append(donor)
    units = sorted(by_unit)
    unit = units[rng.randrange(len(units))]
    candidates = by_unit[unit]
    return candidates[rng.randrange(len(candidates))]


def draw_breakpoints(
    rng: random.Random,
    genetic_map: GeneticMap,
    start_bp: int,
    end_bp_exclusive: int,
    transitions_per_morgan: float,
) -> tuple[list[int], int]:
    if transitions_per_morgan < 0 or not math.isfinite(transitions_per_morgan):
        raise ValueError("Transition rate must be finite and non-negative")
    start_cm = genetic_map.bp_to_cm(start_bp)
    end_cm = genetic_map.bp_to_cm(end_bp_exclusive - 1)
    if end_cm < start_cm:
        raise ValueError("VCF interval has a negative genetic span")
    if transitions_per_morgan == 0 or end_cm == start_cm:
        return [], 0
    rate_per_cm = transitions_per_morgan / 100.0
    value = start_cm
    raw_events = 0
    breakpoints: set[int] = set()
    while True:
        value += rng.expovariate(rate_per_cm)
        if value >= end_cm:
            break
        raw_events += 1
        position = genetic_map.cm_to_bp(value)
        if start_bp < position < end_bp_exclusive:
            breakpoints.add(position)
    return sorted(breakpoints), raw_events


def build_mosaic_plan(
    genetic_map: GeneticMap,
    first_position: int,
    last_position: int,
    donors: dict[str, list[Donor]],
    proportions: dict[str, float],
    transitions_per_morgan: float,
    seed: int,
    target_individuals: int,
    target_prefix: str = "M34_TARGET",
) -> tuple[dict[str, tuple[list[Segment], list[Segment]]], dict[str, int]]:
    if target_individuals <= 0:
        raise ValueError("Target individual count must be positive")
    if not target_prefix or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for character in target_prefix):
        raise ValueError("Target prefix must use only letters, digits, dot, underscore or dash")
    if first_position < genetic_map.start_bp or last_position > genetic_map.end_bp:
        raise ValueError("VCF extent lies outside the genetic map")
    rng = random.Random(seed)
    plan: dict[str, tuple[list[Segment], list[Segment]]] = {}
    raw_events = 0
    collapsed_events = 0
    end_bp_exclusive = last_position + 1
    for target_index in range(target_individuals):
        target_id = f"{target_prefix}_{target_index:04d}"
        homologues: list[list[Segment]] = []
        for haplotype in (0, 1):
            breakpoints, drawn = draw_breakpoints(
                rng,
                genetic_map,
                first_position,
                end_bp_exclusive,
                transitions_per_morgan,
            )
            raw_events += drawn
            collapsed_events += drawn - len(breakpoints)
            boundaries = [first_position, *breakpoints, end_bp_exclusive]
            segments: list[Segment] = []
            for start, end in zip(boundaries, boundaries[1:]):
                ancestry = choose_ancestry(rng, proportions)
                donor = choose_unit_balanced_donor(rng, donors[ancestry])
                segments.append(
                    Segment(target_id, haplotype, start, end, ancestry, donor)
                )
            homologues.append(segments)
        plan[target_id] = (homologues[0], homologues[1])
    return plan, {
        "raw_recombination_events": raw_events,
        "events_collapsed_to_existing_bp": collapsed_events,
    }


def merge_ancestry_segments(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for segment in segments:
        if merged and merged[-1].ancestry == segment.ancestry:
            previous = merged[-1]
            merged[-1] = Segment(
                previous.target_id,
                previous.haplotype,
                previous.start_bp,
                segment.end_bp_exclusive,
                previous.ancestry,
                previous.donor,
            )
        else:
            merged.append(segment)
    return merged


def write_truth_and_audit(
    truth_path: Path,
    audit_path: Path,
    plan: dict[str, tuple[list[Segment], list[Segment]]],
    chromosome: str,
) -> tuple[int, int, int]:
    truth_segments = 0
    donor_segments = 0
    ancestry_transitions = 0
    with deterministic_gzip_text(truth_path) as truth_handle, audit_path.open(
        "w", encoding="utf-8", newline=""
    ) as audit_handle:
        truth = csv.writer(truth_handle, delimiter="\t", lineterminator="\n")
        audit = csv.writer(audit_handle, delimiter="\t", lineterminator="\n")
        truth.writerow(
            ("target_id", "haplotype", "chrom", "start_bp", "end_bp_exclusive", "ancestry")
        )
        audit.writerow(
            (
                "target_id",
                "haplotype",
                "chrom",
                "start_bp",
                "end_bp_exclusive",
                "ancestry",
                "donor_sample_id",
                "donor_haplotype",
                "donor_atomic_unit_id",
            )
        )
        for target_id, homologues in plan.items():
            for segments in homologues:
                donor_segments += len(segments)
                merged = merge_ancestry_segments(segments)
                truth_segments += len(merged)
                ancestry_transitions += max(len(merged) - 1, 0)
                for segment in merged:
                    truth.writerow(
                        (
                            target_id,
                            segment.haplotype,
                            chromosome,
                            segment.start_bp,
                            segment.end_bp_exclusive,
                            segment.ancestry,
                        )
                    )
                for segment in segments:
                    audit.writerow(
                        (
                            target_id,
                            segment.haplotype,
                            chromosome,
                            segment.start_bp,
                            segment.end_bp_exclusive,
                            segment.ancestry,
                            segment.donor.sample_id,
                            segment.donor.haplotype,
                            segment.donor.atomic_unit_id,
                        )
                    )
    os.chmod(audit_path, 0o600)
    return truth_segments, donor_segments, ancestry_transitions


def parse_phased_gt(sample_field: str, gt_index: int, alt_count: int) -> tuple[str, str]:
    values = sample_field.split(":")
    if gt_index >= len(values):
        raise ValueError("Sample field lacks GT")
    genotype = values[gt_index]
    if "/" in genotype or genotype.count("|") != 1:
        raise ValueError(f"Unphased or non-diploid GT {genotype!r}")
    alleles = tuple(genotype.split("|"))
    if len(alleles) != 2:
        raise ValueError(f"Non-diploid GT {genotype!r}")
    for allele in alleles:
        if allele == ".":
            continue
        if not allele.isdigit() or not 0 <= int(allele) <= alt_count:
            raise ValueError(f"Invalid allele code {allele!r}")
    return alleles[0], alleles[1]


def _segment_for_position(
    segments: list[Segment], position: int, pointer: int
) -> tuple[Segment, int]:
    while pointer + 1 < len(segments) and position >= segments[pointer].end_bp_exclusive:
        pointer += 1
    segment = segments[pointer]
    if not segment.start_bp <= position < segment.end_bp_exclusive:
        raise ValueError("VCF position is not covered by the mosaic plan")
    return segment, pointer


def write_target_vcf(
    source_path: Path,
    output_path: Path,
    summary: VcfSummary,
    plan: dict[str, tuple[list[Segment], list[Segment]]],
    donor_sample_ids: set[str],
    seed: int,
    proportions: dict[str, float],
    transitions_per_morgan: float,
) -> dict[str, int]:
    sample_index = {sample: index for index, sample in enumerate(summary.samples)}
    missing = sorted(donor_sample_ids - set(sample_index))
    if missing:
        raise ValueError(f"Donor samples absent from VCF: {','.join(missing[:5])}")
    selected_columns = {sample: sample_index[sample] for sample in donor_sample_ids}
    pointers = {
        (target, haplotype): 0 for target in plan for haplotype in (0, 1)
    }
    target_ids = tuple(plan)
    records = 0
    phased_donor_gts = 0
    copied_missing_alleles = 0
    with open_text(source_path) as source, deterministic_gzip_text(output_path) as output:
        for line_number, line in enumerate(source, 1):
            if line.startswith("##"):
                output.write(line)
                continue
            if line.startswith("#CHROM"):
                output.write("##m34_scope=exploratory_AFR_EUR_NAM_mosaics\n")
                output.write(f"##m34_seed={seed}\n")
                output.write(
                    "##m34_mixture="
                    + ",".join(f"{name}:{proportions[name]:.12g}" for name in ANCESTRIES)
                    + "\n"
                )
                output.write(f"##m34_transitions_per_morgan={transitions_per_morgan:.12g}\n")
                header = line.rstrip("\n").split("\t")
                output.write("\t".join((*header[:9], *target_ids)) + "\n")
                continue
            if line.startswith("#"):
                output.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 + len(summary.samples):
                raise ValueError(f"VCF sample-field count differs at row {line_number}")
            position = int(fields[1])
            formats = fields[8].split(":")
            if "GT" not in formats:
                raise ValueError(f"VCF row {line_number} lacks GT in FORMAT")
            gt_index = formats.index("GT")
            alt_count = len(fields[4].split(",")) if fields[4] != "." else 0
            donor_genotypes: dict[str, tuple[str, str]] = {}
            for sample_id, index in selected_columns.items():
                donor_genotypes[sample_id] = parse_phased_gt(
                    fields[9 + index], gt_index, alt_count
                )
                phased_donor_gts += 1
            target_genotypes: list[str] = []
            for target_id in target_ids:
                copied: list[str] = []
                for haplotype in (0, 1):
                    segments = plan[target_id][haplotype]
                    segment, pointer = _segment_for_position(
                        segments, position, pointers[(target_id, haplotype)]
                    )
                    pointers[(target_id, haplotype)] = pointer
                    allele = donor_genotypes[segment.donor.sample_id][segment.donor.haplotype]
                    copied_missing_alleles += allele == "."
                    copied.append(allele)
                target_genotypes.append("|".join(copied))
            output.write("\t".join((*fields[:8], "GT", *target_genotypes)) + "\n")
            records += 1
    if records != summary.records:
        raise ValueError("Second VCF pass produced a different record count")
    return {
        "vcf_records": records,
        "phased_donor_gts_validated": phased_donor_gts,
        "copied_missing_alleles": copied_missing_alleles,
    }


def _resolve_mixture(args: argparse.Namespace) -> dict[str, float]:
    value = args.mixture_proportions
    if isinstance(value, str):
        return parse_mixture_proportions(value)
    if set(value) != set(ANCESTRIES):
        raise ValueError("Mixture proportions must define AFR, EUR and NAM")
    result = {ancestry: float(value[ancestry]) for ancestry in ANCESTRIES}
    if any(not math.isfinite(item) or item <= 0 for item in result.values()):
        raise ValueError("All mixture proportions must be finite and positive")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Mixture proportions must sum to one")
    return result


def _resolve_transition_rate(args: argparse.Namespace) -> tuple[float, str]:
    generations = args.admixture_generations
    direct = args.transitions_per_morgan
    if (generations is None) == (direct is None):
        raise ValueError(
            "Provide exactly one of admixture_generations or transitions_per_morgan"
        )
    rate = float(generations if generations is not None else direct)
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Transition parameter must be finite and non-negative")
    return rate, "pulse_generations" if generations is not None else "direct_rate"


def run(args: argparse.Namespace) -> dict[str, object]:
    chromosome = canonical_contig(args.chromosome)
    if chromosome != "22":
        raise ValueError("M34 exploratory mosaic generation is currently restricted to chr22")
    proportions = _resolve_mixture(args)
    transition_rate, transition_parameterization = _resolve_transition_rate(args)
    forbidden_roles = tuple(args.forbidden_role or DEFAULT_FORBIDDEN_ROLES)
    if args.donor_role in forbidden_roles:
        raise ValueError("Donor role cannot also be forbidden")

    genetic_map = read_genetic_map(args.genetic_map, chromosome)
    summary = scan_vcf(args.phased_vcf, chromosome)
    donors, role_audit = load_split(
        args.split_tsv,
        args.donor_role,
        forbidden_roles,
        str(args.donor_unit_partition),
        int(args.rotation),
    )
    donor_sample_ids = {
        donor.sample_id for values in donors.values() for donor in values
    }
    absent = donor_sample_ids - set(summary.samples)
    if absent:
        raise ValueError(f"Split donors absent from VCF: {','.join(sorted(absent)[:5])}")
    target_prefix = str(args.target_prefix)
    generated_ids = {
        f"{target_prefix}_{index:04d}" for index in range(args.target_individuals)
    }
    if donor_sample_ids & generated_ids:
        raise ValueError("A generated target identifier collides with a donor identifier")
    plan, event_counts = build_mosaic_plan(
        genetic_map,
        summary.first_position,
        summary.last_position,
        donors,
        proportions,
        transition_rate,
        int(args.seed),
        int(args.target_individuals),
        target_prefix,
    )

    args.outdir.mkdir(parents=True, exist_ok=False)
    target_path = args.outdir / "m34_target.chr22.vcf.gz"
    truth_path = args.outdir / "m34_truth.chr22.tsv.gz"
    audit_path = args.outdir / "m34_donor_audit.private.tsv"
    receipt_path = args.outdir / "m34_mosaic.receipt.json"
    truth_segments, donor_segments, ancestry_transitions = write_truth_and_audit(
        truth_path, audit_path, plan, chromosome
    )
    vcf_counts = write_target_vcf(
        args.phased_vcf,
        target_path,
        summary,
        plan,
        donor_sample_ids,
        int(args.seed),
        proportions,
        transition_rate,
    )
    unique_donor_haplotypes = {
        (segment.donor.sample_id, segment.donor.haplotype)
        for homologues in plan.values()
        for segments in homologues
        for segment in segments
    }
    outputs = {}
    for path in (target_path, truth_path, audit_path):
        outputs[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    receipt: dict[str, object] = {
        "stage": "M34_NAM_EXPLORATORY_MOSAICS",
        "decision": "PASS_EXPLORATORY_MOSAICS_WITH_LOCAL_TRUTH",
        "scope": {
            "exploratory_only": True,
            "confirmatory_validation": False,
            "holdout_independent": False,
            "reason": "M27F roles are reused after the original holdout was compromised",
            "truth_kind": "simulated_from_recorded_recombination_breakpoints",
            "generalizes_to_dnabr": False,
        },
        "parameters": {
            "chromosome": chromosome,
            "seed": int(args.seed),
            "target_individuals": int(args.target_individuals),
            "target_prefix": target_prefix,
            "donor_unit_partition": str(args.donor_unit_partition),
            "rotation": int(args.rotation),
            "mixture_proportions": proportions,
            "transition_parameterization": transition_parameterization,
            "transitions_per_morgan": transition_rate,
        },
        "inputs": {
            "phased_vcf": {
                "name": args.phased_vcf.name,
                "sha256": sha256_file(args.phased_vcf),
            },
            "split_tsv": {
                "name": args.split_tsv.name,
                "sha256": sha256_file(args.split_tsv),
            },
            "genetic_map": {
                "name": args.genetic_map.name,
                "sha256": sha256_file(args.genetic_map),
            },
        },
        "role_audit": {
            **role_audit,
            "ref_train_used_as_donor": args.donor_role == "REF_TRAIN",
            "source_test_used_as_donor": args.donor_role == "SOURCE_TEST",
            "kinship_or_ibd_reestimated": False,
        },
        "counts": {
            **vcf_counts,
            **event_counts,
            "target_individuals": len(plan),
            "target_haplotypes": 2 * len(plan),
            "donor_segments": donor_segments,
            "truth_ancestry_segments": truth_segments,
            "truth_ancestry_transitions": ancestry_transitions,
            "unique_donor_haplotypes_used": len(unique_donor_haplotypes),
        },
        "vcf_extent": {
            "first_position": summary.first_position,
            "last_position": summary.last_position,
            "genetic_span_cm": genetic_map.bp_to_cm(summary.last_position)
            - genetic_map.bp_to_cm(summary.first_position),
        },
        "ancestry_mapping": {
            "African": "AFR",
            "European": "EUR",
            "Native_American": "NAM",
        },
        "outputs": outputs,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phased-vcf", required=True, type=Path)
    parser.add_argument("--split-tsv", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--chromosome", default="chr22")
    parser.add_argument("--donor-role", default="SOURCE_VALID")
    parser.add_argument("--forbidden-role", action="append")
    parser.add_argument(
        "--donor-unit-partition", choices=("all", "fit", "valid"), default="all"
    )
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--target-individuals", required=True, type=int)
    parser.add_argument("--target-prefix", default="M34_TARGET")
    parser.add_argument(
        "--mixture-proportions",
        required=True,
        type=parse_mixture_proportions,
        metavar="AFR=x,EUR=y,NAM=z",
    )
    transitions = parser.add_mutually_exclusive_group(required=True)
    transitions.add_argument("--admixture-generations", type=float)
    transitions.add_argument("--transitions-per-morgan", type=float)
    return parser.parse_args()


def main() -> None:
    receipt = run(parse_args())
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "counts": receipt["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
