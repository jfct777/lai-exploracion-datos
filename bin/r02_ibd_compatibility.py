#!/usr/bin/env python3
"""Bounded inventory of existing eight-column asIBD, not new IBD inference.

Keep source labels and homologue fields. A missing record is never certified
zero IBD. The inventory does not authorize interval/ancestry interpretation:
the DNABR gap-fill/ancestry-assignment provenance must be supplied separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_snapshot(paths: list[Path]) -> dict:
    result = {}
    for path in paths:
        stat = path.stat()
        digest = sha256(path)
        after = path.stat()
        require((stat.st_size, stat.st_mtime_ns, stat.st_ino) ==
                (after.st_size, after.st_mtime_ns, after.st_ino), "input changed during authentication")
        result[str(path.resolve())] = {"sha256": digest, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    return result


def verify_unchanged(paths: list[Path], before: dict) -> None:
    require(input_snapshot(paths) == before, "input changed during inventory; results not authenticated")


def rows(path: Path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def unique_rows(path: Path, key: str) -> dict:
    result = {}
    for row in rows(path):
        require(key in row and row[key], "missing metadata identifier")
        require(row[key] not in result or result[row[key]] == row,
                "conflicting duplicate metadata identifier")
        result[row[key]] = row
    return result


def crosswalk(metadata: Path, samples: Path, covariates: Path) -> tuple[dict, list]:
    """Use the documented Cohort_ID rule; do not split or fuzzy-match IDs."""
    wanted = samples.read_text().splitlines()
    require(wanted and all(wanted) and len(set(wanted)) == len(wanted),
            "analytical sample list empty or duplicated")
    meta = unique_rows(metadata, "ID")
    historical = unique_rows(covariates, "sample_id")
    require(set(historical) == set(wanted), "M36 and current analytical ID sets differ")
    result = {}
    for sample in wanted:
        require(sample in meta, "analytical sample lacks source metadata")
        cohort = meta[sample].get("Cohort", "")
        require(cohort and cohort == historical[sample].get("cohort"),
                "cohort source differs between primary metadata and authenticated M36")
        source_id = f"{cohort}_{sample}"
        require(source_id not in result, "source crosswalk collision")
        result[source_id] = sample
    return result, wanted


def msp_roster(path: Path) -> tuple[set, str]:
    lines = path.read_text().splitlines()
    require(len(lines) == 2, "only the complete two-line MSP header is expected")
    columns = lines[1].split("\t")
    require(len(columns) > 6 and (len(columns) - 6) % 2 == 0,
            "MSP header must contain paired haplotype columns")
    result = set()
    for left, right in zip(columns[6::2], columns[7::2]):
        a, ah = left.rsplit(".", 1)
        b, bh = right.rsplit(".", 1)
        require(a == b and {ah, bh} == {"0", "1"}, "MSP haplotype order mismatch")
        require(a not in result, "MSP sample duplicated")
        result.add(a)
    return result, lines[0]


@dataclass(frozen=True)
class Segment:
    id1: str
    hap1: str
    id2: str
    hap2: str
    chromosome: int
    start: int
    end: int
    column8: float


def parse_segment(line: str) -> Segment:
    fields = line.split()
    require(len(fields) == 8, "expected gapfilled eight fields, not native Refined IBD nine")
    chromosome, start, end = map(int, fields[4:7])
    value = float(fields[7])
    require(1 <= chromosome <= 22 and 0 <= start <= end, "invalid autosomal interval")
    require(math.isfinite(value) and value >= 0, "column 8 must be finite nonnegative")
    # 0 can represent an unresolved homologue after merging. Do not fabricate
    # an IBD1/IBD2 assignment or reject all such records as malformed.
    require(fields[1] in {"0", "1", "2"} and fields[3] in {"0", "1", "2"},
            "unrecognized homologue encoding; inspect source before continuing")
    return Segment(fields[0], fields[1], fields[2], fields[3], chromosome, start, end, value)


class PairBits:
    """Exact O(N²) bits, not Python tuples or one object per input record."""
    def __init__(self, size: int):
        self.size = size
        self.bits = bytearray((size * size + 7) // 8)
        self.count = 0

    def add(self, left: int, right: int) -> None:
        if left == right:
            return
        left, right = min(left, right), max(left, right)
        index = left * self.size + right
        byte, bit = divmod(index, 8)
        mask = 1 << bit
        if not self.bits[byte] & mask:
            self.count += 1
            self.bits[byte] |= mask


def authenticate_segments(paths: list[Path], receipt_path: Path, covariates: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    expected = receipt["input_descriptors"]["covariates"]["sha256"]
    require(sha256(covariates) == expected, "M36 covariates hash differs from primary receipt")
    names = [path.name for path in paths]
    require(len(set(names)) == len(names), "duplicate segment file supplied")
    result = {}
    for path in paths:
        key = f"asibd_segment_{path.name}"
        require(key in receipt["source_input_descriptors"], "segment absent from M36 receipt")
        digest = sha256(path)
        require(digest == receipt["source_input_descriptors"][key]["sha256"],
                "segment content differs from authenticated historical M36 source")
        result[path.name] = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                             "sha256": digest}
    return result


def inventory(paths: list[Path], mapping: dict, samples: list, source_roster: set) -> dict:
    require(set(mapping) <= source_roster, "analytical source IDs absent from MSP roster")
    numeric = {source: index for index, source in enumerate(sorted(mapping))}
    overall_pairs = PairBits(len(samples))
    overall_ids, overall_source = set(), set()
    files = {}
    for path in paths:
        seen, source_seen = set(), set()
        pairs = PairBits(len(samples))
        stats = Counter()
        haplotypes = Counter()
        chromosomes = {}
        with path.open() as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                segment = parse_segment(line)
                stats["records"] += 1
                source_seen.update((segment.id1, segment.id2))
                require(segment.id1 in source_roster and segment.id2 in source_roster,
                        "segment ID absent from source MSP roster")
                haplotypes.update((segment.hap1, segment.hap2))
                chrom = chromosomes.setdefault(segment.chromosome, {
                    "records": 0, "both_analytical_records": 0,
                    "first_reported_coordinate": segment.start,
                    "last_reported_coordinate": segment.end,
                    "column8_min": segment.column8, "column8_max": segment.column8,
                })
                chrom["records"] += 1
                chrom["first_reported_coordinate"] = min(chrom["first_reported_coordinate"], segment.start)
                chrom["last_reported_coordinate"] = max(chrom["last_reported_coordinate"], segment.end)
                chrom["column8_min"] = min(chrom["column8_min"], segment.column8)
                chrom["column8_max"] = max(chrom["column8_max"], segment.column8)
                if segment.hap1 == "0" or segment.hap2 == "0":
                    stats["records_with_unknown_homologue_zero"] += 1
                if segment.id1 == segment.id2:
                    stats["self_records"] += 1
                present = [source for source in (segment.id1, segment.id2) if source in mapping]
                seen.update(present)
                if len(present) == 2 and segment.id1 != segment.id2:
                    stats["both_analytical_records"] += 1
                    chrom["both_analytical_records"] += 1
                    pairs.add(numeric[segment.id1], numeric[segment.id2])
                    overall_pairs.add(numeric[segment.id1], numeric[segment.id2])
                else:
                    stats["not_two_distinct_analytical_people_records"] += 1
        overall_ids.update(seen)
        overall_source.update(source_seen)
        files[path.name] = {**dict(stats), "source_ids_with_record": len(source_seen),
            "analytical_ids_with_record": len(seen), "analytical_ids_without_record": len(samples) - len(seen),
            "analytical_pairs_with_record": pairs.count, "homologue_fields": dict(haplotypes),
            "chromosomes": chromosomes, "ancestry_label": "UNVERIFIED_SOURCE_FILE_LABEL"}
    return {"files": files, "analytical_people": len(samples), "source_msp_people": len(source_roster),
            "source_people_observed_in_segments": len(overall_source),
            "analytical_people_present_in_msp": len(mapping),
            "analytical_people_observed_in_segments": len(overall_ids),
            "analytical_people_without_segment_record": len(samples) - len(overall_ids),
            "analytical_pairs_with_record_any_file": overall_pairs.count,
            "analytical_pair_universe": len(samples) * (len(samples) - 1) // 2,
            "no_record_interpretation": "UNKNOWN_NOT_CERTIFIED_ZERO_IBD",
            "_private_source_ids_seen": overall_source, "_private_analytical_ids_seen": overall_ids}


def parse_log(path: Path) -> dict:
    text = path.read_text()
    values = {}
    for key in ("gt", "map", "length", "lod", "nthreads"):
        match = re.search(r"^\s+" + key + r"=(.+)$", text, re.M)
        require(match is not None, f"Refined IBD log lacks {key}")
        values[key] = match.group(1)
    match = re.search(r"Samples:\s+([\d,]+)", text)
    require(match is not None, "log lacks sample denominator")
    values.update(samples=int(match.group(1).replace(",", "")),
                  program=text.splitlines()[0], sha256=sha256(path),
                  build_evidence="GRCh38_DECLARED_IN_LOG_MAP_PATH" if "GRCh38" in values["map"] else "UNKNOWN")
    return values


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", nargs="+", required=True, type=Path)
    parser.add_argument("--m36-receipt", type=Path, required=True)
    parser.add_argument("--m36-receipt-sha256", required=True)
    parser.add_argument("--m36-covariates", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--analytical-samples", type=Path, required=True)
    parser.add_argument("--msp-header", type=Path, required=True)
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    require(len({path.name for path in args.logs}) == len(args.logs), "duplicate log basename")
    all_inputs = [*args.segments, *args.logs, args.m36_receipt, args.m36_covariates,
                  args.metadata, args.analytical_samples, args.msp_header]
    before = input_snapshot(all_inputs)
    require(sha256(args.m36_receipt) == args.m36_receipt_sha256, "M36 primary receipt digest mismatch")
    descriptors = authenticate_segments(args.segments, args.m36_receipt, args.m36_covariates)
    mapping, samples = crosswalk(args.metadata, args.analytical_samples, args.m36_covariates)
    roster, labels = msp_roster(args.msp_header)
    result = inventory(args.segments, mapping, samples, roster)
    source_seen = result.pop("_private_source_ids_seen")
    result.pop("_private_analytical_ids_seen")
    log_evidence = {path.name: parse_log(path) for path in args.logs}
    verify_unchanged(all_inputs, before)
    args.outdir.mkdir(parents=True, exist_ok=False, mode=0o700)
    with (args.outdir / "crosswalk.private.tsv").open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "source_id", "in_source_msp", "has_any_segment_record"])
        for source, sample in sorted(mapping.items()):
            writer.writerow([sample, source, 1, int(source in source_seen)])
    inputs = {name: {"path": str(path.resolve()), "sha256": before[str(path.resolve())]["sha256"]} for name, path in {
        "metadata": args.metadata, "samples": args.analytical_samples, "m36_covariates": args.m36_covariates,
        "m36_receipt": args.m36_receipt, "msp_header": args.msp_header}.items()}
    result.update(schema="r02_ibd_compatibility_inventory_v1", status="INVENTORIED_NOT_READY_FOR_BIOLOGICAL_USE",
        source_files=descriptors, inputs=inputs, logs=log_evidence,
        input_closure=before, input_stability="full SHA256/stat before and after streaming inventory",
        msp_header_ancestry_dictionary_not_gapfilled_mapping=labels,
        crosswalk_rule="metadata.Cohort + '_' + metadata.ID; exact M36 cohort/ID match and MSP membership",
        interval_union="NOT_CALCULATED_PENDING_ENDPOINT_AND_BUILD_PROVENANCE",
        copy_weighted_ibd="NOT_CALCULATED; preserve homologue IDs, including unresolved zero",
        ancestry_assignment_semantics="UNKNOWN_DNABR_GENERATOR; cannot presume homogeneous ancestry intervals",
        column8_unit="historical M36 interprets cM; upstream gapfilled transformation not authenticated",
        joint_callable_territory="NOT_AVAILABLE; no zero filling or denominator fabrication",
        downstream_ready=False,
        blockers=["DNABR gap-fill/filter generator and version", "anc1/2/3 ancestry assignment provenance",
                  "eight-column units and endpoint convention", "map identity and build chain to final intervals",
                  "jointly evaluable territory per person/pair"],
        permitted_conclusion="exact data-format, source-content, ID-overlap and logged upstream settings only")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["resources"] = {"elapsed_seconds": time.monotonic() - started,
        "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
        "maximum_resident_memory_kib_linux": usage.ru_maxrss}
    (args.outdir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "people": len(samples),
        "people_with_record": result["analytical_people_observed_in_segments"],
        "pairs_with_record": result["analytical_pairs_with_record_any_file"]}))


if __name__ == "__main__":
    main()
