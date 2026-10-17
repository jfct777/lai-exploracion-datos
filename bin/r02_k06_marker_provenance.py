#!/usr/bin/env python3
"""Join historical K06 BIM rows to a bounded ORIG_SITE query.

The source query must come from an authenticated, indexed VCF and contain no
genotypes.  BIM A1/A2 are checked only as an unordered pair.  REF/ALT
orientation is taken from the historical marker ID only when it has the exact
PLINK ``chrom:pos:REF:ALT`` form documented by the producing log.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


BASES = frozenset("ACGT")
SOURCE_COLUMNS = ("chromosome", "position_bp", "source_id", "REF", "ALT",
                  "FILTER", "ORIG_SITE", "ORIG_NALLELES")
MARKER_COLUMNS = ("bim_row_1based", "chromosome", "position_bp", "marker_id", "A1", "A2")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path, columns: tuple[str, ...]):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"Unexpected columns in {path.name}")
        yield from reader


def marker_orientation(marker_id: str, chromosome: int, position: int):
    match = re.fullmatch(r"(?:chr)?([0-9]+):([0-9]+):([ACGT]):([ACGT])", marker_id)
    if not match or (int(match[1]), int(match[2])) != (chromosome, position):
        return None
    return match[3], match[4]


def run(markers_path: Path, source_path: Path, chromosome: int, output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    markers = []
    for row in read_table(markers_path, MARKER_COLUMNS):
        chrom, pos = int(row["chromosome"]), int(row["position_bp"])
        if chrom == chromosome:
            if pos < 1 or row["A1"] not in BASES or row["A2"] not in BASES:
                raise ValueError("Invalid BIM target")
            markers.append(row)
    if not markers:
        raise ValueError("No markers for requested chromosome")

    sources = defaultdict(list)
    for row in read_table(source_path, SOURCE_COLUMNS):
        chrom = int(row["chromosome"].removeprefix("chr"))
        pos, original_n = int(row["position_bp"]), int(row["ORIG_NALLELES"])
        if chrom != chromosome or pos < 1 or row["REF"] not in BASES or row["ALT"] not in BASES:
            raise ValueError("Invalid source-query coordinate or SNP allele")
        if row["FILTER"] != "PASS" or row["ORIG_SITE"] != f"chr{chrom}|{pos}" or original_n < 2:
            raise ValueError("Source query lacks PASS/original-site provenance")
        row = dict(row, original_n=original_n)
        sources[pos].append(row)

    rows, unresolved, counts = [], [], Counter()
    for marker in markers:
        pos = int(marker["position_bp"])
        oriented = marker_orientation(marker["marker_id"], chromosome, pos)
        candidates = sources.get(pos, [])
        source_counts = {row["original_n"] for row in candidates}
        if len(source_counts) > 1:
            raise ValueError("ORIG_NALLELES differs within source site")
        exact = [row for row in candidates if oriented and
                 (row["REF"], row["ALT"]) == oriented]
        unordered = [row for row in candidates if {row["REF"], row["ALT"]} ==
                     {marker["A1"], marker["A2"]}]
        if len(exact) == 1 and len(unordered) == 1 and exact[0] is unordered[0]:
            status = "EXACT_ID_REF_ALT_AND_UNORDERED_A1_A2_MATCH"
        elif oriented is None:
            status = "UNRESOLVED_MARKER_ID_DOES_NOT_ENCODE_REF_ALT"
        elif not candidates:
            status = "UNRESOLVED_NO_SOURCE_RECORD_AT_POSITION"
        else:
            status = "UNRESOLVED_SOURCE_ALLELES_DIFFER"
        original_n = next(iter(source_counts)) if source_counts else None
        result = {
            **marker,
            "id_REF": oriented[0] if oriented else "",
            "id_ALT": oriented[1] if oriented else "",
            "source_REF": exact[0]["REF"] if len(exact) == 1 else "",
            "source_ALT": exact[0]["ALT"] if len(exact) == 1 else "",
            "source_ID": exact[0]["source_id"] if len(exact) == 1 else "",
            "ORIG_SITE": exact[0]["ORIG_SITE"] if len(exact) == 1 else "",
            "ORIG_NALLELES": original_n if original_n is not None else "",
            "status": status,
        }
        rows.append(result); counts[status] += 1
        if status.startswith("UNRESOLVED"):
            unresolved.append(result)

    positions = {int(row["position_bp"]) for row in markers}
    matched_positions = {int(row["position_bp"]) for row in rows
                         if row["status"].startswith("EXACT")}
    multiallelic_positions = {pos for pos in matched_positions
                              if sources[pos][0]["original_n"] > 2}
    fields = tuple(rows[0])
    for name, values in (("provenance.tsv", rows), ("unresolved.tsv", unresolved)):
        with (output / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(values)
    summary = {
        "schema": "r02_k06_marker_provenance_v1",
        "status": "BOUNDED_SOURCE_CORRESPONDENCE_NOT_HISTORICAL_OBJECT_IDENTITY",
        "chromosome": chromosome,
        "inputs": {
            "markers": {"path": str(markers_path.resolve()), "sha256": sha(markers_path)},
            "source_query": {"path": str(source_path.resolve()), "sha256": sha(source_path)},
        },
        "counts": {
            "target_positions": len(positions),
            "target_marker_rows": len(markers),
            "source_query_rows": sum(len(value) for value in sources.values()),
            "positions_with_all_marker_rows_matched": sum(
                all(row["status"].startswith("EXACT") for row in rows
                    if int(row["position_bp"]) == pos) for pos in positions),
            "matched_marker_rows": counts["EXACT_ID_REF_ALT_AND_UNORDERED_A1_A2_MATCH"],
            "current_source_original_multiallelic_positions": len(multiallelic_positions),
            "unresolved_marker_rows": len(unresolved),
        },
        "status_counts": dict(sorted(counts.items())),
        "allele_interpretation": "REF/ALT_FROM_DOCUMENTED_PLINK_MARKER_ID; A1_A2_UNORDERED_CHECK_ONLY",
        "conclusion": "ORIG_NALLELES certifies multiplicity only in the authenticated current M01 input; coordinate/allele agreement does not prove byte identity with the February 2026 historical source.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", required=True, type=Path)
    parser.add_argument("--source-query", required=True, type=Path)
    parser.add_argument("--chromosome", required=True, type=int, choices=range(1, 23))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.markers, args.source_query, args.chromosome, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
