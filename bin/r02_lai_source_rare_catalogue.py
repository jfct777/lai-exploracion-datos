#!/usr/bin/env python3
"""Authenticate complete M02.1 source-rare membership for LAI inputs."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping


SCHEMA = "r02_lai_source_rare_catalogue_v1"
SOURCE_DEFINITION = "DNABR2723_FIXED_SOURCE_MINOR"
TABLE_FIELDS = (
    "chromosome", "position_bp", "REF", "ALT", "counted_allele", "source_build",
)
PROJECTION_TABLE_FIELDS = (
    "chromosome", "position_bp", "REF", "ALT", "source_minor",
)
CONDITIONS = ("ALL_AVAILABLE_SOURCE_RARE", "NO_SOURCE_RARE")
LocusKey = tuple[str, int, str, str]


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(value: object, name: str) -> tuple[Path, str]:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
            name + " pin fields differ")
    path, expected = Path(value["path"]), value["sha256"]
    require(_is_sha256(expected) and path.is_file(),
            name + " pin is invalid or absent")
    require(sha256(path) == expected, name + " SHA-256 differs")
    return path, expected


def _chromosome(value: object) -> str:
    text = str(value)
    require(text.isdigit() and 1 <= int(text) <= 22, "catalogue chromosome differs")
    return str(int(text))


@dataclass(frozen=True)
class SourceRareCatalogue:
    manifest_sha256: str
    table_sha256: str
    original_contract_sha256: str
    source_catalogue_sha256: str
    build: str
    chromosome: str
    counted_allele_by_locus: Mapping[LocusKey, str]

    @property
    def loci(self) -> frozenset[LocusKey]:
        return frozenset(self.counted_allele_by_locus)

    @property
    def source_rare_total(self) -> int:
        return len(self.counted_allele_by_locus)

    @property
    def membership_is_complete_R(self) -> bool:
        return True


@dataclass(frozen=True)
class SourceRareAxisProjection:
    """Authenticated ``R ∩ V`` membership, valid only for one exact locus axis."""

    receipt_sha256: str
    table_sha256: str
    source_catalogue_sha256: str
    build: str
    source_definition: str
    chromosome: str
    source_rare_total: int
    source_rare_ref_total: int
    source_rare_alt_total: int
    projection_axis_sha256: str
    reference_manifest_sha256: str
    reference_source_vcf_sha256: str
    counted_allele_by_locus: Mapping[LocusKey, str]

    @property
    def loci(self) -> frozenset[LocusKey]:
        return frozenset(self.counted_allele_by_locus)

    @property
    def membership_is_complete_R(self) -> bool:
        return False


def canonical_locus_axis_sha256(keys: Iterable[LocusKey]) -> str:
    """Hash the exact sorted SNV axis as used by ``r02_lai_catalogue_axis.py``."""
    normalized = []
    for chromosome, position, ref, alt in keys:
        key = (_chromosome(chromosome), int(position), str(ref), str(alt))
        require(key[1] > 0 and key[2] in "ACGT" and key[3] in "ACGT"
                and len(key[2]) == len(key[3]) == 1 and key[2] != key[3],
                "invalid locus in classified axis")
        normalized.append(key)
    require(len(normalized) == len(set(normalized)), "duplicate locus in classified axis")
    ordered = sorted(normalized, key=lambda key: (int(key[0]), key[1], key[2], key[3]))
    digest = hashlib.sha256()
    for key in ordered:
        digest.update(("\t".join(map(str, key)) + "\n").encode())
    return digest.hexdigest()


def load_source_rare_axis_projection(receipt_path: str | Path,
                                     receipt_sha256: str,
                                     table_path: str | Path,
                                     table_sha256: str) -> SourceRareAxisProjection:
    """Load a GJ ``R ∩ V`` table without opening the original source VCF."""
    receipt_path, table_path = Path(receipt_path), Path(table_path)
    require(receipt_path.is_file() and _is_sha256(receipt_sha256)
            and sha256(receipt_path) == receipt_sha256,
            "source-rare projection receipt SHA-256 differs")
    require(table_path.is_file() and _is_sha256(table_sha256)
            and sha256(table_path) == table_sha256,
            "source-rare projection table SHA-256 differs")
    receipt = json.loads(receipt_path.read_text())
    required = {
        "schema", "status", "chromosome", "source_original", "source_present",
        "source_absent_exact_key", "reference_loci", "reference_axis_sha256",
        "reference_manifest_sha256", "reference_source_vcf_sha256",
        "reference_build_recorded", "source_original_vcf_sha256",
        "outputs_sha256", "arrays_loaded", "genotype_arrays_loaded", "scope",
        "callability_verified", "target_axis_verified", "training_admitted",
    }
    require(isinstance(receipt, dict), "source-rare projection receipt is not an object")
    original_gj = (receipt.get("schema") == "r02_lai_catalogue_axis_receipt_v1"
                   and receipt.get("status") == "COMPLETE_METADATA_INTERSECTION")
    integrated = (receipt.get("schema")
                  == "r02_lai_source_rare_axis_projection_receipt_v1"
                  and receipt.get("status")
                  == "REUSED_AUTHENTICATED_INTERSECTION_NO_RECOMPUTATION")
    require(required <= set(receipt)
            and (original_gj or integrated),
            "source-rare projection receipt schema or status differs")
    if integrated:
        parents = receipt.get("parent_receipts")
        require(receipt.get("source_definition") == SOURCE_DEFINITION
                and receipt.get("intersection_recomputed") is False
                and isinstance(parents, list) and parents
                and all(isinstance(parent, Mapping)
                        and set(parent) == {"path", "sha256"} for parent in parents),
                "integrated source-rare projection derivation differs")
        for number, parent in enumerate(parents):
            _pin(parent, "source-rare projection parent " + str(number))
    chromosome = _chromosome(receipt["chromosome"])
    original, present = receipt["source_original"], receipt["source_present"]
    require(isinstance(original, dict) and isinstance(present, dict)
            and set(original) == {"total", "REF", "ALT"}
            and set(present) == {"total", "REF", "ALT"}
            and all(type(value) is int and value >= 0
                    for value in (*original.values(), *present.values()))
            and original["total"] == original["REF"] + original["ALT"]
            and present["total"] == present["REF"] + present["ALT"]
            and present["total"] <= original["total"]
            and present["REF"] <= original["REF"]
            and present["ALT"] <= original["ALT"]
            and receipt["source_absent_exact_key"]
                == original["total"] - present["total"],
            "source-rare projection full/projected totals are inconsistent")
    hashes = (receipt["reference_axis_sha256"], receipt["reference_manifest_sha256"],
              receipt["reference_source_vcf_sha256"],
              receipt["source_original_vcf_sha256"])
    require(receipt["reference_build_recorded"] == "GRCh38"
            and receipt["scope"] == "REFERENCE_LOCUS_AXIS_ONLY_UNDER_RECORDED_COORDINATES"
            and receipt["arrays_loaded"] == ["chromosome", "position_bp", "ref", "alt"]
            and receipt["genotype_arrays_loaded"] is False
            and receipt["callability_verified"] is False
            and receipt["training_admitted"] is False
            and all(_is_sha256(value) for value in hashes)
            and receipt["outputs_sha256"].get("source-rare-present.tsv.gz")
                == table_sha256,
            "source-rare projection provenance or scope differs")
    if original_gj:
        require(receipt["target_axis_verified"] is False,
                "GJ receipt cannot certify a target axis")
    else:
        binding = receipt.get("target_binding")
        require(receipt["target_axis_verified"] is True
                and isinstance(binding, dict)
                and binding.get("status") == "EXACT_TARGET_AND_REFERENCE_AXIS_MATCH"
                and binding.get("axis_sha256") == receipt["reference_axis_sha256"]
                and _is_sha256(binding.get("target_loci_sha256"))
                and _is_sha256(binding.get("factorized_indices_sha256")),
                "integrated source-rare target-axis binding differs")

    membership: dict[LocusKey, str] = {}
    counts = {"REF": 0, "ALT": 0}
    with gzip.open(table_path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(tuple(reader.fieldnames or ()) == PROJECTION_TABLE_FIELDS,
                "source-rare projection table columns differ")
        for row in reader:
            key = (_chromosome(row["chromosome"]), int(row["position_bp"]),
                   row["REF"], row["ALT"])
            side = row["source_minor"]
            require(key[0] == chromosome and key[1] > 0
                    and len(key[2]) == len(key[3]) == 1
                    and key[2] in "ACGT" and key[3] in "ACGT" and key[2] != key[3]
                    and side in counts, "invalid source-rare projection row")
            require(key not in membership, "duplicate source-rare projection locus")
            membership[key] = side
            counts[side] += 1
    require(len(membership) == present["total"]
            and counts["REF"] == present["REF"] and counts["ALT"] == present["ALT"],
            "source-rare projection row/orientation counts differ")
    return SourceRareAxisProjection(
        receipt_sha256, table_sha256, receipt["source_original_vcf_sha256"],
        "GRCh38", SOURCE_DEFINITION, chromosome, original["total"],
        original["REF"], original["ALT"],
        receipt["reference_axis_sha256"], receipt["reference_manifest_sha256"],
        receipt["reference_source_vcf_sha256"], membership)


def load_source_rare_catalogue(manifest_path: str | Path,
                               manifest_sha256: str) -> SourceRareCatalogue:
    """Authenticate the manifest, canonical table, original contract and VCF."""
    manifest_path = Path(manifest_path)
    require(manifest_path.is_file() and len(manifest_sha256) == 64
            and sha256(manifest_path) == manifest_sha256,
            "source-rare catalogue manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text())
    expected = {"schema", "build", "source_definition", "source_catalogue_sha256",
                "original_contract", "source_table", "expected_rows", "chromosome"}
    require(isinstance(manifest, dict) and set(manifest) == expected
            and manifest["schema"] == SCHEMA
            and manifest["source_definition"] == SOURCE_DEFINITION,
            "source-rare catalogue manifest fields differ")
    build = manifest["build"]
    chromosome = _chromosome(manifest["chromosome"])
    rows_expected = manifest["expected_rows"]
    source_hash = manifest["source_catalogue_sha256"]
    require(build == "GRCh38" and type(rows_expected) is int and rows_expected > 0
            and isinstance(source_hash, str) and len(source_hash) == 64,
            "source-rare build, row count, or source hash differs")

    contract_path, contract_hash = _pin(manifest["original_contract"],
                                        "original M02.1 contract")
    contract = json.loads(contract_path.read_text())
    criteria, counts = contract.get("criteria", {}), contract.get("counts", {})
    require(contract.get("schema") == "m02_1_minor_v1"
            and _chromosome(contract.get("chromosome")) == chromosome
            and contract.get("cohort_samples_before") == 2723
            and contract.get("cohort_samples_after") == 2723
            and counts.get("rare") == rows_expected
            and criteria.get("GT_REF_ALT_preserved") is True,
            "original M02.1 contract identity differs")
    original_vcf = contract_path.parent / str(contract.get("output", ""))
    require(original_vcf.name == contract.get("output") and original_vcf.is_file()
            and sha256(original_vcf) == source_hash,
            "original source-rare VCF SHA-256 differs")

    table_path, table_hash = _pin(manifest["source_table"], "source-rare table")
    opener = gzip.open if table_path.suffix == ".gz" else open
    membership: dict[LocusKey, str] = {}
    with opener(table_path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(tuple(reader.fieldnames or ()) == TABLE_FIELDS,
                "source-rare table columns differ")
        for row in reader:
            chrom = _chromosome(row["chromosome"])
            ref, alt, counted = row["REF"], row["ALT"], row["counted_allele"]
            try:
                position = int(row["position_bp"])
            except ValueError as error:
                raise ValueError("source-rare position is not an integer") from error
            require(chrom == chromosome and position > 0 and row["source_build"] == build,
                    "source-rare row chromosome, position, or build differs")
            require(ref in "ACGT" and alt in "ACGT" and len(ref) == len(alt) == 1
                    and ref != alt and counted in ("REF", "ALT"),
                    "source-rare row is not a biallelic SNV with its counted allele")
            key = (chrom, position, ref, alt)
            require(key not in membership, "duplicate source-rare locus key")
            membership[key] = counted
    require(len(membership) == rows_expected, "source-rare table row count differs")
    return SourceRareCatalogue(manifest_sha256, table_hash, contract_hash, source_hash,
                               build, chromosome, membership)


def catalogue_coverage(catalogue: SourceRareCatalogue | SourceRareAxisProjection,
                       material_rows: Iterable[Mapping[str, str]],
                       common_keys: Iterable[LocusKey],
                       event_keys: Iterable[LocusKey]) -> dict[str, int | str | bool]:
    """Classify V using either complete R or authenticated R-intersect-V rows."""
    material: set[LocusKey] = set()
    for row in material_rows:
        key = (_chromosome(row["chromosome"]), int(row["position_bp"]),
               row["REF"], row["ALT"])
        require(row["counted_allele"] == "ALT", "material must preserve raw ALT dosage")
        require(key not in material, "duplicate material locus key")
        material.add(key)
    common, event = set(common_keys), set(event_keys)
    require(not (common & event), "COMMON/EVENT locus axes overlap")
    require(common | event == material, "COMMON plus EVENT does not equal all material loci")
    rare = catalogue.loci
    source_total = catalogue.source_rare_total
    overlap = rare & material
    result = {
        "schema": "r02_lai_source_rare_coverage_v1",
        "complete_catalogue_covered": len(overlap) == source_total,
        "coverage_scope": "LOCUS_AXIS_MEMBERSHIP_ONLY",
        "reference_callability_checked": False,
        "scientific_admission_complete": False,
        "source_rare_total": source_total,
        "material_loci_total": len(material),
        "source_rare_intersect_material": len(overlap),
        "source_rare_missing_from_material": source_total - len(overlap),
        "historical_event_total": len(event),
        "historical_event_intersect_source_rare": len(event & rare),
        "source_rare_in_historical_common": len(common & rare),
        "historical_event_not_source_rare": len(event - rare),
    }
    if not catalogue.membership_is_complete_R:
        result.update({
            "membership_materialization": "R_INTERSECT_AUTHENTICATED_AXIS",
            "source_rare_membership_rows_available": len(rare),
            "classification_valid_only_for_exact_axis": True,
            "projection_axis_sha256": catalogue.projection_axis_sha256,
        })
    return result
