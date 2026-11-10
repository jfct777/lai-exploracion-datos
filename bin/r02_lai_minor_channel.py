#!/usr/bin/env python3
"""Prepare fixed-source-minor annotations without training or reading genotypes.

The source catalogue orientation is joined by the exact GRCh38 key
``(chromosome, position_bp, REF, ALT)``.  A locus outside the authenticated
projection remains ``UNKNOWN``: absence is not an observed ALT orientation and
is not a zero rare-allele dosage.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from r02_lai_source_rare_catalogue import (
    SOURCE_DEFINITION,
    canonical_locus_axis_sha256,
    load_source_rare_axis_projection,
)


SCHEMA = "r02_lai_minor_channel_v1"
INVENTORY_SCHEMA = "r02_lai_minor_projection_inventory_v1"
INVENTORY_SCHEMA_WITH_TARGETS = "r02_lai_minor_projection_inventory_v2"
AXIS_FIELDS = ("chromosome", "position_bp", "REF", "ALT", "counted_allele")
OUTPUT_FIELDS = (
    "axis_index", "chromosome", "position_bp", "REF", "ALT",
    "rare_membership", "source_minor", "orientation_available",
)
SOURCE_MINOR_VALUES = ("REF", "ALT", "UNKNOWN")
LocusKey = tuple[str, int, str, str]


def require(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _authenticate(path: str | Path, expected_sha256: str, label: str) -> Path:
    path = Path(path)
    require(path.is_file() and _is_sha256(expected_sha256), label + " pin is invalid")
    require(sha256(path) == expected_sha256, label + " SHA-256 differs")
    return path


def _normalize_key(chromosome: object, position: object, ref: object, alt: object) -> LocusKey:
    chrom = str(chromosome).removeprefix("chr")
    ref, alt = str(ref), str(alt)
    require(chrom.isdigit() and 1 <= int(chrom) <= 22, "axis chromosome differs")
    try:
        position = int(position)
    except (TypeError, ValueError) as error:
        raise ValueError("axis position is not an integer") from error
    require(position > 0 and ref in "ACGT" and alt in "ACGT"
            and len(ref) == len(alt) == 1 and ref != alt,
            "axis row is not a biallelic SNV")
    return str(int(chrom)), position, ref, alt


def read_alt_axis(path: str | Path, expected_sha256: str) -> list[LocusKey]:
    """Read an ordered ALT-dosage locus axis and authenticate its file hash."""
    path = _authenticate(path, expected_sha256, "ALT-dosage locus axis")
    rows: list[LocusKey] = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(tuple(reader.fieldnames or ()) == AXIS_FIELDS, "locus axis columns differ")
        for row in reader:
            require(row["counted_allele"] == "ALT", "locus axis does not preserve ALT dosage")
            rows.append(_normalize_key(row["chromosome"], row["position_bp"],
                                       row["REF"], row["ALT"]))
    require(rows and len(rows) == len(set(rows)), "locus axis is empty or contains duplicates")
    require(all(rows[index - 1][1] < rows[index][1] for index in range(1, len(rows))),
            "single-chromosome locus axis is not strictly increasing")
    require(len({row[0] for row in rows}) == 1, "locus axis spans multiple chromosomes")
    return rows


def annotate_axis(axis: Sequence[LocusKey], orientation: Mapping[LocusKey, str]) -> list[dict[str, object]]:
    """Return one explicit annotation per axis row; nonmembers remain UNKNOWN."""
    require(axis and len(axis) == len(set(axis)), "annotation axis is empty or duplicated")
    require(set(orientation) <= set(axis), "source-minor projection is not a subset of the axis")
    result: list[dict[str, object]] = []
    for index, key in enumerate(axis):
        source_minor = orientation.get(key, "UNKNOWN")
        require(source_minor in SOURCE_MINOR_VALUES, "source-minor orientation differs")
        member = source_minor != "UNKNOWN"
        result.append({
            "axis_index": index,
            "chromosome": key[0],
            "position_bp": key[1],
            "REF": key[2],
            "ALT": key[3],
            "rare_membership": member,
            "source_minor": source_minor,
            "orientation_available": member,
        })
    return result


def orient_observed_alt_dose(g_alt: object, observed: bool, source_minor: str) -> float | None:
    """Orient one diploid ALT dosage; return ``None`` when unavailable."""
    require(type(observed) is bool, "observed flag must be boolean")
    require(source_minor in SOURCE_MINOR_VALUES, "source-minor orientation differs")
    if not observed or source_minor == "UNKNOWN":
        return None
    require(g_alt in (0, 1, 2), "observed ALT dosage is outside 0/1/2")
    return float(g_alt if source_minor == "ALT" else 2 - g_alt)


def source_minor_dosage(alt_dosage, observed, source_minor, np):
    """Orient observed diploid ALT dosages; unavailable cells remain NaN.

    ``source_minor`` is a one-dimensional locus axis containing REF, ALT, or
    UNKNOWN.  The returned availability mask is ``observed & orientation_known``.
    Therefore a returned 0 is an observed zero dose, while missing genotype or
    unknown catalogue orientation remains NaN.
    """
    dosage = np.asarray(alt_dosage)
    seen = np.asarray(observed)
    orientation = np.asarray(source_minor).astype(str)
    require(dosage.shape == seen.shape and dosage.ndim >= 1,
            "ALT dosage and observed mask axes differ")
    require(seen.dtype == np.dtype(bool), "observed mask must be boolean")
    require(orientation.ndim == 1 and orientation.shape[0] == dosage.shape[-1]
            and np.isin(orientation, SOURCE_MINOR_VALUES).all(),
            "source-minor orientation axis differs")
    require(np.isin(dosage[seen], (0, 1, 2)).all(), "observed ALT dosage is outside 0/1/2")
    known = orientation != "UNKNOWN"
    available = seen & np.broadcast_to(known, dosage.shape)
    oriented = np.full(dosage.shape, np.nan, dtype=np.float32)
    alt = np.broadcast_to(orientation == "ALT", dosage.shape)
    ref = np.broadcast_to(orientation == "REF", dosage.shape)
    oriented[available & alt] = dosage[available & alt]
    oriented[available & ref] = 2 - dosage[available & ref]
    return oriented, available


def _write_gzip_table(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    require(not path.exists(), "output annotation table exists; refusing overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=OUTPUT_FIELDS, delimiter="\t",
                                        lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({**row,
                                     "rare_membership": str(bool(row["rare_membership"])).lower(),
                                     "orientation_available":
                                         str(bool(row["orientation_available"])).lower()})


def build_channel(*, axis_path: str | Path, axis_sha256: str,
                  projection_receipt: str | Path, projection_receipt_sha256: str,
                  projection_table: str | Path, projection_table_sha256: str,
                  output_table: str | Path, output_receipt: str | Path) -> dict[str, object]:
    """Materialize the full ordered metadata axis without opening genotype arrays."""
    output_table, output_receipt = Path(output_table), Path(output_receipt)
    require(not output_receipt.exists(), "output receipt exists; refusing overwrite")
    axis = read_alt_axis(axis_path, axis_sha256)
    projection = load_source_rare_axis_projection(
        projection_receipt, projection_receipt_sha256,
        projection_table, projection_table_sha256,
    )
    canonical_hash = canonical_locus_axis_sha256(axis)
    require(canonical_hash == projection.projection_axis_sha256,
            "ALT locus axis and source-minor projection axis differ")
    require({key[0] for key in axis} == {projection.chromosome},
            "axis and projection chromosomes differ")
    rows = annotate_axis(axis, projection.counted_allele_by_locus)
    _write_gzip_table(output_table, rows)
    counts = {
        "axis_loci": len(rows),
        "rare_membership": sum(bool(row["rare_membership"]) for row in rows),
        "source_minor_REF": sum(row["source_minor"] == "REF" for row in rows),
        "source_minor_ALT": sum(row["source_minor"] == "ALT" for row in rows),
        "source_minor_UNKNOWN": sum(row["source_minor"] == "UNKNOWN" for row in rows),
    }
    require(counts["rare_membership"] == len(projection.loci)
            and counts["rare_membership"]
                == counts["source_minor_REF"] + counts["source_minor_ALT"]
            and counts["source_minor_UNKNOWN"]
                == counts["axis_loci"] - counts["rare_membership"],
            "annotation counts differ")
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PREPARED_METADATA_ONLY_NO_TRAINING",
        "chromosome": projection.chromosome,
        "build": projection.build,
        "source_definition": projection.source_definition,
        "axis": {
            "path": str(Path(axis_path).resolve()),
            "file_sha256": axis_sha256,
            "canonical_locus_axis_sha256": canonical_hash,
        },
        "source_projection": {
            "receipt": {"path": str(Path(projection_receipt).resolve()),
                        "sha256": projection_receipt_sha256},
            "table": {"path": str(Path(projection_table).resolve()),
                      "sha256": projection_table_sha256},
            "source_catalogue_sha256": projection.source_catalogue_sha256,
            "source_rare_total_original": projection.source_rare_total,
            "source_rare_absent_from_axis": projection.source_rare_total - len(projection.loci),
        },
        "output_table": {"path": str(output_table.resolve()), "sha256": sha256(output_table)},
        "fields": list(OUTPUT_FIELDS),
        "counts": counts,
        "no_source_rare_condition": {
            "excluded_loci": counts["rare_membership"],
            "retained_loci": counts["axis_loci"] - counts["rare_membership"],
            "rule": "exclude every row with rare_membership=true from target, references and context",
        },
        "dosage_rule": {
            "ALT": "g_source_minor = g_ALT for observed genotype",
            "REF": "g_source_minor = 2 - g_ALT for observed genotype",
            "UNKNOWN": "unavailable; never assume ALT and never encode as observed zero",
            "missing_genotype": "unavailable even when orientation is known",
        },
        "identity_already_present": ["chromosome", "position_bp", "REF", "ALT"],
        "identity_not_claimed": "No learned categorical locus ID is added or certified.",
        "genotype_arrays_read": False,
        "individual_people_read": False,
        "source_minor_recalculated": False,
        "training_performed": False,
        "model_changed": False,
    }
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def build_inventory(inventory_path: str | Path, inventory_sha256: str,
                    descriptor_paths: Sequence[str | Path], output: str | Path,
                    target_binding_evidence_path: str | Path | None = None,
                    target_binding_evidence_sha256: str | None = None) -> dict[str, object]:
    """Authenticate existing per-autosome projection descriptors without copying tables."""
    inventory_path = _authenticate(inventory_path, inventory_sha256, "projection inventory")
    recorded = json.loads(inventory_path.read_text())
    require(recorded.get("schema") == "r02_lai_integrated_projection_inventory_v1"
            and recorded.get("status") == "COMPLETE",
            "projection inventory schema or status differs")
    expected = recorded.get("descriptor_sha256_by_chromosome")
    require(isinstance(expected, dict) and set(expected) == {str(i) for i in range(1, 23)},
            "projection inventory chromosome set differs")
    rows = []
    for descriptor_path in descriptor_paths:
        descriptor_path = Path(descriptor_path)
        descriptor = json.loads(descriptor_path.read_text())
        chrom = str(int(descriptor["chromosome"]))
        require(sha256(descriptor_path) == expected.get(chrom),
                "projection descriptor SHA-256 differs for chromosome " + chrom)
        receipt_pin, table_pin = descriptor["projection_receipt"], descriptor["projection_table"]
        projection = load_source_rare_axis_projection(
            receipt_pin["path"], receipt_pin["sha256"], table_pin["path"], table_pin["sha256"])
        receipt = json.loads(Path(receipt_pin["path"]).read_text())
        require(projection.chromosome == chrom
                and descriptor["reference_axis_sha256"] == projection.projection_axis_sha256,
                "projection descriptor identity differs for chromosome " + chrom)
        rows.append({
            "chromosome": chrom,
            "descriptor_path": str(descriptor_path.resolve()),
            "descriptor_sha256": expected[chrom],
            "projection_table_path": str(Path(table_pin["path"]).resolve()),
            "projection_table_sha256": table_pin["sha256"],
            "projection_receipt_sha256": receipt_pin["sha256"],
            "status": descriptor["status"],
            "target_binding_status": descriptor["target_binding"]["status"],
            "reference_loci": receipt["reference_loci"],
            "source_present": len(projection.loci),
            "source_minor_REF": sum(side == "REF" for side in projection.counted_allele_by_locus.values()),
            "source_minor_ALT": sum(side == "ALT" for side in projection.counted_allele_by_locus.values()),
            "source_original": projection.source_rare_total,
            "reference_axis_sha256": projection.projection_axis_sha256,
        })
    rows.sort(key=lambda row: int(row["chromosome"]))
    require([row["chromosome"] for row in rows] == [str(i) for i in range(1, 23)],
            "projection descriptors do not cover chromosomes 1 through 22 exactly once")
    if target_binding_evidence_path is not None:
        evidence_path = _authenticate(target_binding_evidence_path,
                                      str(target_binding_evidence_sha256),
                                      "target binding evidence")
        evidence = json.loads(evidence_path.read_text())
        require(evidence.get("schema") == "r02_chr1_target_axis_authentication_v1"
                and evidence.get("status")
                    == "PASS_3_OF_3_METADATA_ONLY_INPUT_SPEC_STILL_BLOCKED_BY_PERMISSIONS"
                and evidence.get("training_performed") is False
                and evidence.get("genotypes_read") is False
                and evidence.get("people_read") is False
                and evidence.get("permission_policy_resolved") is False
                and evidence.get("related_unit_pairs_resolved") is False,
                "target binding evidence scope differs")
        cases = evidence.get("cases")
        require(isinstance(cases, list) and len(cases) == 3,
                "target binding evidence case count differs")
        chr1 = next(row for row in rows if row["chromosome"] == "1")
        for case in cases:
            classification = case.get("classification", {})
            require(case.get("status") == "EXACT_TARGET_AND_REFERENCE_AXIS_MATCH"
                    and classification.get("classified_material_axis_sha256")
                        == chr1["reference_axis_sha256"]
                    and classification.get("projection_axis_sha256")
                        == chr1["reference_axis_sha256"]
                    and classification.get("material_loci_total") == chr1["reference_loci"]
                    and classification.get("source_rare_intersect_material")
                        == chr1["source_present"]
                    and classification.get("source_rare_projection_receipt_sha256")
                        == chr1["projection_receipt_sha256"]
                    and classification.get("source_rare_projection_table_sha256")
                        == chr1["projection_table_sha256"]
                    and classification.get("training_performed") is False,
                    "chr1 target binding case differs from authenticated projection")
        chr1.update({
            "status": "TARGET_AND_REFERENCE_AXES_AUTHENTICATED_3_CASES_PERMISSIONS_PENDING",
            "target_binding_status": "EXACT_TARGET_AND_REFERENCE_AXIS_MATCH_3_CASES",
            "target_binding_evidence": {
                "path": str(evidence_path.resolve()),
                "sha256": target_binding_evidence_sha256,
                "cases": [
                    {
                        "case_id": case["case_id"],
                        "target_loci_sha256": case["classification"]["target_loci_sha256"],
                        "factorized_indices_sha256":
                            case["classification"]["factorized_indices_sha256"],
                    }
                    for case in cases
                ],
                "permission_policy_resolved": False,
                "related_unit_pairs_resolved": False,
                "training_input_spec_emitted": False,
            },
        })
    target_bound = sum(str(row["target_binding_status"]).startswith(
        "EXACT_TARGET_AND_REFERENCE_AXIS_MATCH") for row in rows)
    result: dict[str, object] = {
        "schema": (INVENTORY_SCHEMA_WITH_TARGETS if target_binding_evidence_path is not None
                   else INVENTORY_SCHEMA),
        "status": "AUTHENTICATED_EXISTING_PROJECTIONS_NO_COPY",
        "source_definition": SOURCE_DEFINITION,
        "input_inventory": {"path": str(inventory_path.resolve()), "sha256": inventory_sha256},
        "chromosomes": rows,
        "totals": {
            "chromosomes": len(rows),
            "reference_loci": sum(int(row["reference_loci"]) for row in rows),
            "source_present": sum(int(row["source_present"]) for row in rows),
            "target_bound_chromosomes": target_bound,
            "target_bound_cases": sum(
                len(row.get("target_binding_evidence", {}).get("cases", []))
                if "target_binding_evidence" in row else 1
                for row in rows
                if str(row["target_binding_status"]).startswith(
                    "EXACT_TARGET_AND_REFERENCE_AXIS_MATCH")),
            "reference_only_chromosomes": len(rows) - target_bound,
        },
        "limit": "An exact target-axis binding does not certify permissions, relatedness, genotype observation or a trainable input specification.",
        "training_performed": False,
        "model_changed": False,
    }
    output = Path(output)
    require(not output.exists(), "inventory output exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    channel = subparsers.add_parser("channel")
    channel.add_argument("--axis", required=True)
    channel.add_argument("--axis-sha256", required=True)
    channel.add_argument("--projection-receipt", required=True)
    channel.add_argument("--projection-receipt-sha256", required=True)
    channel.add_argument("--projection-table", required=True)
    channel.add_argument("--projection-table-sha256", required=True)
    channel.add_argument("--output-table", required=True)
    channel.add_argument("--output-receipt", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--input-inventory", required=True)
    inventory.add_argument("--input-inventory-sha256", required=True)
    inventory.add_argument("--descriptor", action="append", required=True)
    inventory.add_argument("--target-binding-evidence")
    inventory.add_argument("--target-binding-evidence-sha256")
    inventory.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "channel":
        result = build_channel(
            axis_path=args.axis, axis_sha256=args.axis_sha256,
            projection_receipt=args.projection_receipt,
            projection_receipt_sha256=args.projection_receipt_sha256,
            projection_table=args.projection_table,
            projection_table_sha256=args.projection_table_sha256,
            output_table=args.output_table, output_receipt=args.output_receipt,
        )
    else:
        result = build_inventory(args.input_inventory, args.input_inventory_sha256,
                                 args.descriptor, args.output,
                                 args.target_binding_evidence,
                                 args.target_binding_evidence_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
