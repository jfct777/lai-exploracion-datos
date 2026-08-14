#!/usr/bin/env python3
"""Pure parsing helpers for the M27C targeted gVCF audit."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


STATE_UNCOVERED = 0
STATE_REFERENCE_BLOCK = 1
STATE_EXPLICIT_EXACT = 2
STATE_EXPLICIT_OTHER_ALT_HOMREF = 3
STATE_ALLELE_INCOMPATIBLE = 4
STATE_MISSING_GENOTYPE = 5
STATE_MALFORMED = 6

STATE_NAMES = {
    STATE_UNCOVERED: "not_covered",
    STATE_REFERENCE_BLOCK: "reference_block",
    STATE_EXPLICIT_EXACT: "explicit_variant_exact",
    STATE_EXPLICIT_OTHER_ALT_HOMREF: "explicit_other_alt_homref",
    STATE_ALLELE_INCOMPATIBLE: "allele_incompatible",
    STATE_MISSING_GENOTYPE: "missing_genotype",
    STATE_MALFORMED: "malformed",
}

COMPATIBLE_STATES = {
    STATE_REFERENCE_BLOCK,
    STATE_EXPLICIT_EXACT,
    STATE_EXPLICIT_OTHER_ALT_HOMREF,
}

VariantKey = tuple[str, int, str, str]


@dataclass(frozen=True)
class ParsedCall:
    state: int
    dosage: int = -1
    depth: int = -1
    gq: int = -1
    phased: bool = False
    precedence: int = 0


def canonical_contig(value: str) -> str:
    return value.removeprefix("chr")


def parse_info(info: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in info.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        elif item and item != ".":
            result[item] = ""
    return result


def parse_int(value: str | None) -> int:
    if value is None or value in {"", "."}:
        return -1
    try:
        return int(value)
    except ValueError:
        return -1


def parse_format(format_field: str, sample_field: str) -> dict[str, str]:
    keys = format_field.split(":")
    values = sample_field.split(":")
    return {key: values[index] if index < len(values) else "." for index, key in enumerate(keys)}


def parse_gt(value: str | None) -> tuple[list[int] | None, bool]:
    if not value or value in {".", "./.", ".|."}:
        return None, False
    if "|" in value:
        separator = "|"
        phased = True
    elif "/" in value:
        separator = "/"
        phased = False
    else:
        return None, False
    fields = value.split(separator)
    if len(fields) != 2 or any(field == "." for field in fields):
        return None, phased
    try:
        return [int(field) for field in fields], phased
    except ValueError:
        return None, phased


def record_end(position: int, ref: str, info: Mapping[str, str]) -> int:
    end = parse_int(info.get("END"))
    return end if end >= position else position + max(1, len(ref)) - 1


def classify_record(
    fields: Sequence[str], target_key: VariantKey, target_position: int
) -> ParsedCall:
    """Classify one gVCF record for one requested biallelic SNP allele."""

    if len(fields) < 10:
        return ParsedCall(STATE_MALFORMED, precedence=1)
    try:
        position = int(fields[1])
    except ValueError:
        return ParsedCall(STATE_MALFORMED, precedence=1)
    ref = fields[3].upper()
    alts = fields[4].upper().split(",")
    info = parse_info(fields[7])
    if not position <= target_position <= record_end(position, ref, info):
        return ParsedCall(STATE_UNCOVERED)

    values = parse_format(fields[8], fields[9])
    alleles, phased = parse_gt(values.get("GT"))
    gq = parse_int(values.get("GQ"))
    model_ref, model_alt = target_key[2], target_key[3]
    is_reference_block = alts == ["<NON_REF>"] and "END" in info

    if is_reference_block:
        depth = parse_int(values.get("MIN_DP"))
        if alleles is None:
            return ParsedCall(STATE_MISSING_GENOTYPE, depth=depth, gq=gq, phased=phased, precedence=2)
        if alleles != [0, 0]:
            return ParsedCall(STATE_ALLELE_INCOMPATIBLE, depth=depth, gq=gq, phased=phased, precedence=2)
        return ParsedCall(STATE_REFERENCE_BLOCK, dosage=0, depth=depth, gq=gq, phased=phased, precedence=2)

    depth = parse_int(values.get("DP"))
    if position != target_position or len(model_ref) != 1 or ref != model_ref:
        return ParsedCall(STATE_ALLELE_INCOMPATIBLE, depth=depth, gq=gq, phased=phased, precedence=1)
    if alleles is None:
        return ParsedCall(STATE_MISSING_GENOTYPE, depth=depth, gq=gq, phased=phased, precedence=3)

    model_indices = [index + 1 for index, alt in enumerate(alts) if alt == model_alt]
    non_reference = [allele for allele in alleles if allele != 0]
    if not model_indices:
        if not non_reference:
            return ParsedCall(
                STATE_EXPLICIT_OTHER_ALT_HOMREF,
                dosage=0,
                depth=depth,
                gq=gq,
                phased=phased,
                precedence=3,
            )
        return ParsedCall(STATE_ALLELE_INCOMPATIBLE, depth=depth, gq=gq, phased=phased, precedence=3)

    model_index = model_indices[0]
    if any(allele not in (0, model_index) for allele in alleles):
        return ParsedCall(STATE_ALLELE_INCOMPATIBLE, depth=depth, gq=gq, phased=phased, precedence=3)
    return ParsedCall(
        STATE_EXPLICIT_EXACT,
        dosage=sum(allele == model_index for allele in alleles),
        depth=depth,
        gq=gq,
        phased=phased,
        precedence=3,
    )


def parse_targeted_lines(
    lines: Iterable[str],
    target_keys: Sequence[VariantKey],
    positions_to_indices: Mapping[int, Sequence[int]],
) -> list[ParsedCall]:
    """Map overlapping gVCF records to all requested alleles at each target."""

    calls = [ParsedCall(STATE_UNCOVERED) for _ in target_keys]
    positions = sorted(positions_to_indices)
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 8:
            continue
        try:
            start = int(fields[1])
        except ValueError:
            continue
        if canonical_contig(fields[0]) != target_keys[0][0]:
            continue
        end = record_end(start, fields[3], parse_info(fields[7]))
        left = bisect.bisect_left(positions, start)
        right = bisect.bisect_right(positions, end)
        for target_position in positions[left:right]:
            for index in positions_to_indices[target_position]:
                candidate = classify_record(fields, target_keys[index], target_position)
                if candidate.precedence >= calls[index].precedence:
                    calls[index] = candidate
    return calls


def is_high_quality(call: ParsedCall, minimum_depth: int, minimum_gq: int) -> bool:
    return (
        call.state in COMPATIBLE_STATES
        and call.dosage >= 0
        and call.depth >= minimum_depth
        and call.gq >= minimum_gq
    )


def parse_header_contract(header: str) -> dict[str, object]:
    samples: list[str] = []
    contig_length = None
    source = None
    reference = None
    format_ids: set[str] = set()
    for line in header.splitlines():
        if line.startswith("##source="):
            source = line.split("=", 1)[1]
        elif line.startswith("##reference="):
            reference = line.split("=", 1)[1]
        elif re.match(r"^##contig=<ID=chr22(?:,|>)", line):
            match = re.search(r"(?:length|Length)=([0-9]+)", line)
            contig_length = int(match.group(1)) if match else None
        elif line.startswith("##FORMAT=<ID="):
            format_ids.add(line.split("##FORMAT=<ID=", 1)[1].split(",", 1)[0])
        elif line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
    return {
        "samples": samples,
        "source": source,
        "reference": reference,
        "chr22_length": contig_length,
        "format_ids": sorted(format_ids),
        "has_required_fields": {"GT", "DP", "GQ", "MIN_DP"}.issubset(format_ids),
    }
