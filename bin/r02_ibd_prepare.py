#!/usr/bin/env python3
"""Authenticate native Refined IBD and prepare reported-territory contracts.

This is input preparation, not IBD inference or population validation. Source
coordinates are 1-based inclusive, per Refined IBD's nine-column format. Keep
haplotype pairs in the record table, but count the territorial union only once.
Never manufacture IBD1/IBD2, ancestry, genotype callability, or negative calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from r02_ibd_compatibility import input_snapshot, require, sha256, verify_unchanged
from r02_genomic_pair_evidence import read_samples, sample_hash
from r02_segment_evidence import load_ibd, merge_intervals


SCHEMA = "r02_native_ibd_prepare_v1"
RECORD_COLUMNS = ("chrom", "sample_a", "sample_b", "start_pos", "end_pos",
                  "ibd_state", "hap_a", "hap_b", "lod", "length_cm")


def native_record(line, chrom):
    fields = line.split()
    require(len(fields) == 9, "Expected native Refined IBD nine-column record")
    require(fields[1] in ("1", "2") and fields[3] in ("1", "2"), "Invalid native homologue")
    require(fields[4].removeprefix("chr") == chrom, "Native chromosome mismatch")
    lo, hi = int(fields[5]), int(fields[6])
    lod, cm = float(fields[7]), float(fields[8])
    require(1 <= lo <= hi and math.isfinite(lod) and math.isfinite(cm) and cm >= 0,
            "Invalid native coordinates or score")
    require(fields[0] != fields[2], "Self pair is not between-person IBD")
    return fields[0], fields[1], fields[2], fields[3], lo, hi, lod, cm


def authenticated_path(base, record):
    path = base / record["path"]
    require(path.is_file() and sha256(path) == record["sha256"], "Input authentication failed")
    return path


def prepare(contract_path, expected_sha, output, scratch, max_db_mb, min_free_mb, max_pair_records):
    start = time.monotonic()
    require(sha256(contract_path) == expected_sha, "Preparation contract hash mismatch")
    contract = json.loads(contract_path.read_text())
    require(contract.get("schema") == SCHEMA, "Unsupported preparation schema")
    chrom = str(contract.get("chromosome"))
    require(chrom in {str(i) for i in range(1, 23)}, "Autosome required")
    require(contract.get("genome_build") and contract.get("coordinate_system") == "1-based-inclusive",
            "Explicit build and inclusive coordinates required")
    require(contract.get("caller", {}).get("name") == "Refined IBD" and contract["caller"].get("version"),
            "Native caller/version required")
    paths = {name: authenticated_path(contract_path.parent, contract[name])
             for name in ("native", "audit", "crosswalk", "samples", "log")}
    snapshots = input_snapshot([contract_path, *paths.values()])
    require(snapshots[str(contract_path.resolve())]["sha256"] == expected_sha and
            all(snapshots[str(path.resolve())]["sha256"] == contract[name]["sha256"]
                for name, path in paths.items()), "Input changed during initial authentication")
    audit = json.loads(paths["audit"].read_text())
    require(audit.get("verified_complete_object") is True and str(audit["chrom"]) == chrom,
            "Complete source-object audit required")
    require(audit["sha256"] == contract["native"]["sha256"] and audit["log_sha256"] == contract["log"]["sha256"],
            "Audit source/log differs")
    require(audit["bytes"] == paths["native"].stat().st_size, "Audit byte count differs")
    log = paths["log"].read_text()
    require(f'refined-ibd.{contract["caller"]["version"]}.jar finished' in log, "Caller completion/version differs")
    require(contract["genome_build"] in log, "Build declaration not supported by caller log")
    samples = read_samples(paths["samples"])
    require(sample_hash(samples) == contract["analytical_samples_sha256"], "Ordered sample hash differs")
    order = {sample: i for i, sample in enumerate(samples)}
    mapping = {}
    with paths["crosswalk"].open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require({"source_id", "sample_id"} <= set(reader.fieldnames or []), "Crosswalk fields missing")
        for row in reader:
            require(row["source_id"] and row["source_id"] not in mapping and row["sample_id"] in order,
                    "Invalid/duplicate source mapping")
            mapping[row["source_id"]] = row["sample_id"]
    require(len(mapping) == len(samples) and set(mapping.values()) == set(samples), "Crosswalk must be bijective")
    require(max_db_mb > 0 and min_free_mb >= 0 and max_pair_records > 0, "Invalid resource limit")
    scratch = Path(scratch)
    require(scratch.is_dir(), "Scratch directory missing")
    require(shutil.disk_usage(scratch).free >= min_free_mb * 2**20, "Insufficient scratch reserve")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    counts = Counter(source_records=0, analytical_records=0, excluded_nonanalytical_records=0,
                     pairs_with_reported_ibd=0, union_pieces=0, pair_bp_reported_union=0)
    with tempfile.TemporaryDirectory(prefix="ibd-sort-", dir=scratch) as temporary:
        db = sqlite3.connect(str(Path(temporary) / "records.sqlite"))
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-16384")
            page_size = db.execute("PRAGMA page_size").fetchone()[0]
            db.execute(f"PRAGMA max_page_count={max(1, int(max_db_mb * 2**20 / page_size))}")
            db.execute("CREATE TABLE calls(a INTEGER,b INTEGER,lo INTEGER,hi INTEGER,h1 TEXT,h2 TEXT,lod REAL,cm REAL)")
            opener = gzip.open if paths["native"].suffix == ".gz" else open
            with opener(paths["native"], "rt") as handle:
                for line in handle:
                    counts["source_records"] += 1
                    x, h1, y, h2, lo, hi, lod, cm = native_record(line, chrom)
                    require(lod >= audit["min_lod"] and cm >= audit["min_cm"], "Source audit minima contradicted")
                    require(any(left <= lo <= hi <= right for left, right in audit["nominal_processed_intervals_bp_inclusive"]),
                            "Record outside audited processing domain")
                    if x not in mapping or y not in mapping:
                        counts["excluded_nonanalytical_records"] += 1
                        continue
                    a, b = order[mapping[x]], order[mapping[y]]
                    if a > b:
                        a, b, h1, h2 = b, a, h2, h1
                    db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,?,?)", (a, b, lo, hi, h1, h2, lod, cm))
                    counts["analytical_records"] += 1
                    if counts["source_records"] % 10000 == 0:
                        require(shutil.disk_usage(scratch).free >= min_free_mb * 2**20, "Scratch reserve exhausted")
            require(counts["source_records"] == audit["rows"] and counts["analytical_records"] == audit["analytical_pair_rows"],
                    "Source counts differ from complete audit")
            db.commit()
            db.execute("CREATE INDEX ordered_calls ON calls(a,b,lo,hi)")
            counts["distinct_geometric_records"] = db.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT a,b,lo,hi FROM calls)").fetchone()[0]
            require(shutil.disk_usage(scratch).free >= min_free_mb * 2**20, "Scratch reserve exhausted after sorting")
            with gzip.open(output / "intervals.tsv.gz", "wt", newline="") as record_handle, \
                    gzip.open(output / "reported_union.tsv.gz", "wt", newline="") as union_handle, \
                    gzip.open(output / "pair_territory.tsv.gz", "wt", newline="") as pair_handle:
                records = csv.writer(record_handle, delimiter="\t")
                unions = csv.writer(union_handle, delimiter="\t")
                pairs = csv.writer(pair_handle, delimiter="\t")
                records.writerow(RECORD_COLUMNS)
                unions.writerow(("chrom", "sample_a", "sample_b", "start_pos", "end_pos", "ibd_state"))
                pairs.writerow(("chrom", "sample_a", "sample_b", "reported_ibd_union_bp", "reported_union_pieces", "source_records"))
                cursor = db.execute("SELECT * FROM calls ORDER BY a,b,lo,hi")
                for (a, b), group in itertools.groupby(cursor, key=lambda row: row[:2]):
                    intervals = []
                    for _, _, lo, hi, h1, h2, lod, cm in group:
                        require(len(intervals) < max_pair_records, "Per-pair record limit exceeded")
                        intervals.append((lo, hi))
                        records.writerow((chrom, samples[a], samples[b], lo, hi, "ANY_COPY_REPORTED", h1, h2, lod, cm))
                    merged = merge_intervals(intervals)
                    for lo, hi in merged:
                        unions.writerow((chrom, samples[a], samples[b], lo, hi, "ANY_COPY_REPORTED"))
                    bp = sum(hi - lo + 1 for lo, hi in merged)
                    pairs.writerow((chrom, samples[a], samples[b], bp, len(merged), len(intervals)))
                    counts["pairs_with_reported_ibd"] += 1
                    counts["union_pieces"] += len(merged)
                    counts["pair_bp_reported_union"] += bp
                    if counts["pairs_with_reported_ibd"] % 10000 == 0:
                        require(shutil.disk_usage(scratch).free >= min_free_mb * 2**20, "Scratch reserve exhausted during export")
        finally:
            db.close()
    verify_unchanged([contract_path, *paths.values()], snapshots)
    derived = dict(schema="r02_segment_ibd_v2", genome_build=contract["genome_build"],
        coordinate_system="1-based-inclusive", semantics="ANY_COPY_REPORTED",
        evidence_scope="reported_call_geometry_only",
        absence_semantics="no_reported_call_not_biological_absence",
        callset_scope="complete_reported_calls_for_declared_chromosome_and_samples",
        chromosome=chrom, analytical_samples_sha256=sample_hash(samples), caller=contract["caller"],
        intervals=dict(path="intervals.tsv.gz", sha256=sha256(output / "intervals.tsv.gz")),
        source_prepare_sha256=expected_sha, source_native_sha256=audit["sha256"],
        source_uri=audit["uri"], source_generation=audit["generation"])
    (output / "ibd_contract.json").write_text(json.dumps(derived, indent=2) + "\n")
    # Verify the actual consumer accepts this complete chromosome contract.
    # This does not annotate M14 or compare biological communities.
    with tempfile.TemporaryDirectory(prefix="ibd-reader-", dir=scratch) as temporary:
        db = sqlite3.connect(str(Path(temporary) / "reader.sqlite"))
        try:
            db.execute("PRAGMA cache_size=-16384")
            db.execute("PRAGMA journal_mode=OFF")
            db.execute(f"PRAGMA max_page_count={max(1, int(max_db_mb * 2**20 / db.execute('PRAGMA page_size').fetchone()[0]))}")
            load_ibd(db, output / "ibd_contract.json", derived, samples, chrom, {})
            require(db.execute("SELECT COUNT(*) FROM intervals").fetchone()[0] == counts["distinct_geometric_records"],
                    "Consumer round-trip record count differs")
        finally:
            db.close()
    require(shutil.disk_usage(scratch).free >= min_free_mb * 2**20, "Scratch reserve exhausted at completion")
    verify_unchanged([contract_path, *paths.values()], snapshots)
    report = dict(schema=SCHEMA, status="PREPARED_REPORTED_GEOMETRY_NOT_BIOLOGICAL_VALIDATION",
        chromosome=chrom, analytical_samples=len(samples), **counts,
        possible_analytical_pairs=len(samples) * (len(samples) - 1) // 2,
        callability="UNKNOWN", ancestry_assignment="NOT_PERFORMED", ibd1_ibd2="NOT_INFERRED",
        consumer_contract_checked=True, comparison_with_m14_executed=False,
        input_sha256={key: item["sha256"] for key, item in contract.items() if isinstance(item, dict) and "sha256" in item},
        preparation_contract_sha256=expected_sha, seconds=time.monotonic() - start,
        outputs_sha256={name: sha256(output / name) for name in ("intervals.tsv.gz", "reported_union.tsv.gz", "pair_territory.tsv.gz", "ibd_contract.json")})
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--max-db-mb", type=int, required=True)
    parser.add_argument("--min-free-mb", type=int, required=True)
    parser.add_argument("--max-pair-records", type=int, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    report = prepare(args.contract, args.contract_sha256, args.output_dir, args.scratch_dir,
                     args.max_db_mb, args.min_free_mb, args.max_pair_records)
    print(json.dumps({key: report[key] for key in ("status", "chromosome", "analytical_records", "pairs_with_reported_ibd")}))


if __name__ == "__main__":
    main()
