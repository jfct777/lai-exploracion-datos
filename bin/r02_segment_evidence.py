#!/usr/bin/env python3
"""M14.2: masked local evidence on existing M14 candidate chains, not IBD calls.

Inputs use 1-based inclusive intervals. The native M14 sensitivity tables are
candidate_chains.tsv.gz and configuration_summary.tsv. Indexed VCF/BCF is read
once per occupied genomic block, in bounded site chunks. Only pairs present in
that block are counted; there is no all-pairs-by-segments genotype array.

Optional JSON contracts (all require genome_build and coordinate_system):
* r02_segment_common_v1: vcf_sha256, analytical_samples_sha256, min_maf,
  max_missing, frequency_scope='analytical_complete_diploid', site_filter='PASS',
  original_biallelic=true, mask='complete_diploid_GT'. These explicit filters
  are applied to M02 or to an already filtered common VCF/BCF, without imputation.
* r02_genetic_map_v1: sha256, units='cM', format='tsv_chrom_pos_cm'. TSV columns
  chrom,pos,cm; no interpolation outside the observed map support.
* r02_segment_ibd_v1: semantics='any_copy_ibd_territory', caller={name,version},
  absence_means_no_called_ibd_within_callable=true, intervals={path,sha256},
  callability={path,sha256}. Both TSVs have chrom,sample_a,sample_b,start_pos,
  end_pos; intervals also requires ibd_state (IBD1 or IBD2). The caller's states
  are deliberately collapsed to a territorial union, NOT summed copy lengths.
  Paths are relative to the contract, unless absolute. Callability is pairwise;
  a pair absent from it is not evaluable. A PC-Relate kinship table is not IBD.

NA is explicit; zero union does not imply perfect similarity. Counts and
synthetic tests establish technical properties, not biological validity.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np

from r02_genomic_pair_evidence import (
    _header_scalar, chromosome, rare_dosage, read_samples, require, sample_hash,
    sha256, validate_rare_header, validate_rare_site,
)

SCHEMA = "r02_segment_evidence_v1"
COORDINATES = "1-based-inclusive"
CONFIG_COLUMNS = ("config_id", "max_gap_bp", "min_length_bp", "min_shared_effective",
                  "min_shared_nominal_aliases", "n_geo")
CHAIN_COLUMNS = ("chain_id", "chrom", "sample_a", "sample_b", "max_gap_bp", "start_pos",
                 "end_pos", "length_bp", "n_shared_variants")
EVIDENCE_COLUMNS = CHAIN_COLUMNS + (
    "max_shared_gap_bp", "shared_site_density_per_bp",
    "I", "U", "Q", "J", "rare_status", "rare_catalog_sites", "rare_missing_gt_count",
    "O", "Q_C", "common_status", "length_cm", "map_status", "callable_bp",
    "ibd_union_bp", "ibd_fraction", "ibd_status", "ibd_overlap_pieces",
    "ibd_first_overlap_pos", "ibd_last_overlap_pos", "ibd_left_uncovered_bp",
    "ibd_right_uncovered_bp")
IBD_PAIR_COLUMNS = ("chrom", "sample_a", "sample_b", "callable_bp", "ibd_union_bp", "ibd_pieces", "ibd_status")
COMMON_CRITERIA = ("min_maf", "max_missing", "frequency_scope", "site_filter",
                   "original_biallelic", "mask")
RESOURCE_PARAMETERS = ("max_db_mb", "min_free_disk_mb", "resource_check_rows",
                       "max_preflight_memory_mb", "max_preflight_blocks")


class ResourceProgress:
    """Operational telemetry, never a scientific completion marker.

    Disk reserve checks are periodic; SQLite's page limit additionally bounds its
    main file. Journals and output files still require free disk between checks.
    """
    def __init__(self, output, args):
        self.output, self.args, self.db_path = Path(output), args, None
        self.started = self.last = time.monotonic()
        self.phase, self.phase_seconds, self.peaks = "starting", defaultdict(float), Counter()
        self.counts = {}

    def __enter__(self):
        try:
            self.check("starting", {})
        except (ValueError, OSError):
            self.emit("FAILED", error="Initial resource guard failed")
            raise
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            try:
                self.emit("FAILED", error=str(error))
            except OSError:
                # Preserve the original error if the output filesystem is full
                # or lost; a prior progress record is not a completion marker.
                pass

    def emit(self, status="RUNNING", **details):
        now = time.monotonic()
        self.phase_seconds[self.phase] += now - self.last
        self.last = now
        files = [Path(str(self.db_path) + suffix) for suffix in ("", "-journal", "-wal", "-shm")] if self.db_path else []
        sqlite_bytes = sum(p.stat().st_size for p in files if p.is_file())
        free = shutil.disk_usage(self.output).free
        output_bytes = sum(p.stat().st_size for p in self.output.iterdir()
                           if p.is_file() and not p.name.startswith("progress.json"))
        # ru_maxrss is KiB on the Linux execution images used by this pipeline.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        self.peaks["sqlite_bytes"] = max(self.peaks["sqlite_bytes"], sqlite_bytes)
        self.peaks["rss_bytes"] = max(self.peaks["rss_bytes"], rss)
        record = dict(schema="r02_segment_progress_v1", status=status, phase=self.phase,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=now - self.started, phase_seconds=dict(self.phase_seconds),
            counts=dict(self.counts), sqlite_bytes=sqlite_bytes, output_bytes=output_bytes,
            free_disk_bytes=free, storage_path=str(self.output.resolve()),
            peak=dict(self.peaks), limits={k: getattr(self.args, k) for k in RESOURCE_PARAMETERS},
            resumable=False, **details)
        temporary = self.output / "progress.json.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "progress.json")
        with (self.output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return record

    def check(self, phase, counts, **details):
        now = time.monotonic()
        self.phase_seconds[self.phase] += now - self.last
        self.last, self.phase, self.counts = now, phase, dict(counts)
        record = self.emit(**details)
        require(record["sqlite_bytes"] <= self.args.max_db_mb * 1024**2,
                "SQLite resource limit exceeded (main database plus journal files)")
        require(record["free_disk_bytes"] >= self.args.min_free_disk_mb * 1024**2,
                "Free-disk resource reserve breached")
        return record


def chain_geometry(row, order, chrom):
    """Shared validation and canonical pair order for both execution modes."""
    require(chromosome(row["chrom"]) == chrom, "M14 chain chromosome mismatch")
    a, b = row["sample_a"], row["sample_b"]
    require(a in order and b in order and a != b, "Unknown/self M14 pair")
    a, b = sorted((a, b), key=order.__getitem__)
    gap, lo, hi, length, shared = (int(row[k]) for k in CHAIN_COLUMNS[4:])
    require(1 <= lo <= hi and length == hi - lo + 1 and gap > 0 and shared > 0,
            "Invalid M14 coordinates/counts")
    require(shared <= length, "More shared sites than distinct positions")
    return a, b, gap, lo, hi, length, shared


def table_rows(path, required):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "Missing/duplicate TSV header")
        require(set(required) <= set(reader.fieldnames), f"Missing TSV columns: {path}")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Malformed TSV row")
            yield row


def write_table(stack, path, columns):
    opener = gzip.open if str(path).endswith(".gz") else open
    handle = stack.enter_context(opener(path, "xt", encoding="utf-8", newline=""))
    writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    return writer


def merge_intervals(intervals):
    """Territorial union of inclusive intervals; adjacent bases are continuous."""
    result = []
    for lo, hi in sorted(intervals):
        require(isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi, "Invalid interval")
        if result and lo <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def intersect_intervals(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        lo, hi = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if lo <= hi:
            result.append((lo, hi))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def interval_bp(intervals):
    return sum(hi - lo + 1 for lo, hi in merge_intervals(intervals))


def input_record(path):
    path = Path(path)
    require(path.is_file(), f"Input missing: {path}")
    stat = path.stat()
    return dict(path=str(path.resolve()), sha256=sha256(path), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns)


def indexed_input(path, inputs, key):
    candidates = [Path(str(path) + suffix) for suffix in (".csi", ".tbi") if Path(str(path) + suffix).is_file()]
    require(len(candidates) == 1, "Indexed VCF/BCF required with one unambiguous .csi or .tbi")
    inputs[key + "_index"] = input_record(candidates[0])
    return str(candidates[0])


def checked_contract(path, schema, build, inputs, key):
    inputs[key] = input_record(path)
    result = json.loads(Path(path).read_text())
    require(result.get("schema") == schema, f"Wrong {key} schema")
    require(result.get("genome_build") == build, f"Wrong {key} genome build")
    require(result.get("coordinate_system") == COORDINATES, f"Wrong {key} coordinates")
    return result


def load_configurations(path):
    result = []
    ids = set()
    for row in table_rows(path, CONFIG_COLUMNS):
        require(row["config_id"] and row["config_id"] not in ids, "Repeated/empty configuration")
        ids.add(row["config_id"])
        for key in ("max_gap_bp", "min_length_bp", "min_shared_effective", "n_geo"):
            row[key] = int(row[key])
            require(row[key] > 0, "Nonpositive M14 configuration")
        aliases = [int(v) for v in row["min_shared_nominal_aliases"].split(",")]
        require(aliases and len(set(aliases)) == len(aliases) and min(aliases) > 0, "Invalid nominal aliases")
        n_geo = (row["min_length_bp"] - 2 + row["max_gap_bp"]) // row["max_gap_bp"] + 1
        require(row["n_geo"] == n_geo and all(max(n, n_geo) == row["min_shared_effective"] for n in aliases), "M14 effective geometry/aliases disagree")
        require(row["config_id"] == f"L{row['min_length_bp']}_G{row['max_gap_bp']}_N{row['min_shared_effective']}", "Configuration ID does not encode native M14 geometry")
        row["min_shared_nominal_aliases"] = ",".join(map(str, sorted(aliases)))
        result.append(row)
    require(result, "No M14 configurations")
    return result


def create_database(path, monitor=None):
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-16384")
    if monitor:
        monitor.db_path = Path(path)
        page_size = db.execute("PRAGMA page_size").fetchone()[0]
        db.execute(f"PRAGMA max_page_count={max(1, int(monitor.args.max_db_mb * 1024**2) // page_size)}")
    db.execute("""CREATE TABLE chains (
        chain_id TEXT PRIMARY KEY, chrom TEXT, sample_a TEXT, sample_b TEXT,
        max_gap_bp INTEGER, start_pos INTEGER, end_pos INTEGER, length_bp INTEGER,
        n_shared_variants INTEGER, I INTEGER DEFAULT 0, U INTEGER DEFAULT 0,
        Q INTEGER DEFAULT 0, rare_catalog_sites INTEGER DEFAULT 0,
        rare_missing_gt_count INTEGER DEFAULT 0, O INTEGER DEFAULT 0,
        Q_C INTEGER DEFAULT 0, first_shared INTEGER, last_shared INTEGER,
        max_shared_gap INTEGER DEFAULT 0)""")
    db.execute("CREATE INDEX positions ON chains(start_pos,end_pos)")
    db.execute("CREATE INDEX pairs ON chains(sample_a,sample_b,max_gap_bp,start_pos)")
    db.execute("CREATE TABLE links(chain_id TEXT,config_id TEXT,PRIMARY KEY(chain_id,config_id))")
    db.execute("CREATE INDEX links_config_chain ON links(config_id,chain_id)")
    return db


def load_chains(db, path, configs, samples, chrom, monitor=None):
    order = {name: i for i, name in enumerate(samples)}
    counts = Counter()
    if monitor:
        monitor.check("load_chains", counts)
    for row in table_rows(path, CHAIN_COLUMNS[1:]):
        a, b, gap, lo, hi, length, shared = chain_geometry(row, order, chrom)
        identity = json.dumps([chrom, a, b, lo, hi, gap], separators=(",", ":"))
        chain_id = "chain_" + hashlib.sha256(identity.encode()).hexdigest()
        old = db.execute("SELECT n_shared_variants FROM chains WHERE chain_id=?", (chain_id,)).fetchone()
        counts["input_chain_rows"] += 1
        if monitor and counts["input_chain_rows"] % monitor.args.resource_check_rows == 0:
            monitor.check("load_chains", counts)
        if old:
            require(old[0] == shared, "Conflicting duplicate chain")
            counts["duplicate_chain_rows"] += 1
            continue
        db.execute("INSERT INTO chains(" + ",".join(CHAIN_COLUMNS) + ") VALUES(?,?,?,?,?,?,?,?,?)",
                   (chain_id, chrom, a, b, gap, lo, hi, length, shared))
        for config in configs:
            if gap == config["max_gap_bp"] and length >= config["min_length_bp"] and shared >= config["min_shared_effective"]:
                db.execute("INSERT INTO links VALUES(?,?)", (chain_id, config["config_id"]))
                counts["prepared_configuration_links"] += 1
        counts["unique_chains"] += 1
        if counts["unique_chains"] % 10000 == 0:
            db.commit()
    # Filtered maximal chains for a fixed pair/G cannot overlap or be joinable.
    previous = None
    for number, row in enumerate(db.execute("SELECT * FROM chains ORDER BY sample_a,sample_b,max_gap_bp,start_pos"), 1):
        if monitor and number % monitor.args.resource_check_rows == 0:
            monitor.check("validate_chain_geometry", counts, validated_chains=number)
        key = (row["sample_a"], row["sample_b"], row["max_gap_bp"])
        if previous and previous[0] == key:
            require(row["start_pos"] - previous[1] > row["max_gap_bp"], "Overlapping or nonmaximal M14 chains within pair/G")
        previous = key, row["end_pos"]
    for config in configs:
        if monitor:
            monitor.check("validate_configurations", counts, config_id=config["config_id"])
        row = db.execute("""SELECT count(*),coalesce(sum(length_bp),0),coalesce(sum(n_shared_variants),0)
            FROM chains JOIN links USING(chain_id) WHERE config_id=?""", (config["config_id"],)).fetchone()
        for key, observed in zip(("n_segments", "total_shared_bp", "n_shared_variants_total"), row):
            if key in config and config[key] != "":
                require(int(config[key]) == observed, f"Configuration summary mismatch: {config['config_id']} {key}")
    db.commit()
    if monitor:
        monitor.check("chains_ready", counts)
    return counts


class GeneticMap:
    def __init__(self, path, chrom, max_knots):
        self.positions, self.cm = [], []
        for row in table_rows(path, ("chrom", "pos", "cm")):
            require(chromosome(row["chrom"]) == chrom, "Map chromosome mismatch")
            pos, cm = int(row["pos"]), float(row["cm"])
            require(pos > 0 and math.isfinite(cm) and cm >= 0, "Invalid map coordinate")
            require(not self.positions or (pos > self.positions[-1] and cm >= self.cm[-1]), "Map must have unique increasing bp and nondecreasing cM")
            self.positions.append(pos)
            self.cm.append(cm)
            require(len(self.positions) <= max_knots, "Map knot resource limit exceeded")
        require(len(self.positions) >= 2, "Map requires at least two points")

    def at(self, pos):
        if pos < self.positions[0] or pos > self.positions[-1]:
            return None
        i = bisect_left(self.positions, pos)
        if self.positions[i] == pos:
            return self.cm[i]
        lo, hi = self.positions[i - 1], self.positions[i]
        return self.cm[i - 1] + (pos - lo) / (hi - lo) * (self.cm[i] - self.cm[i - 1])

    def length(self, lo, hi):
        left, right = self.at(lo), self.at(hi)
        return None if left is None or right is None else right - left


def load_ibd(db, contract_path, contract, samples, chrom, inputs, monitor=None):
    require(contract.get("semantics") == "any_copy_ibd_territory", "IBD copy/haplotype semantics must be explicit territorial union")
    require(contract.get("absence_means_no_called_ibd_within_callable") is True, "IBD absence semantics not declared")
    caller = contract.get("caller", {})
    require(caller.get("name") and caller.get("version"), "IBD caller and version required")
    order = {name: i for i, name in enumerate(samples)}
    for kind in ("intervals", "callability"):
        item = contract.get(kind, {})
        require(item.get("path") and item.get("sha256"), "IBD intervals and pair callability with hashes are required")
        path = Path(contract_path).parent / item["path"]
        inputs["ibd_" + kind] = input_record(path)
        require(inputs["ibd_" + kind]["sha256"] == item["sha256"], "IBD table hash mismatch")
        db.execute(f"CREATE TABLE {kind}(sample_a TEXT,sample_b TEXT,start_pos INTEGER,end_pos INTEGER,ibd_state TEXT)")
        db.execute(f"CREATE INDEX {kind}_pair ON {kind}(sample_a,sample_b,start_pos,end_pos)")
        required = ("chrom", "sample_a", "sample_b", "start_pos", "end_pos")
        for number, row in enumerate(table_rows(path, required + (("ibd_state",) if kind == "intervals" else ())), 1):
            if monitor and number % monitor.args.resource_check_rows == 0:
                monitor.check("load_ibd_" + kind, monitor.counts, ibd_rows=number)
            require(chromosome(row["chrom"]) == chrom, "IBD/callability chromosome mismatch")
            a, b = row["sample_a"], row["sample_b"]
            require(a in order and b in order and a != b, "Unknown/self IBD pair")
            a, b = sorted((a, b), key=order.__getitem__)
            lo, hi = int(row["start_pos"]), int(row["end_pos"])
            require(1 <= lo <= hi, "Invalid IBD/callability interval")
            state = row.get("ibd_state", "callable")
            require(kind != "intervals" or state in ("IBD1", "IBD2"), "Unsupported IBD state; do not infer copy semantics")
            db.execute(f"INSERT INTO {kind} VALUES(?,?,?,?,?)", (a, b, lo, hi, state))
    db.commit()
    if monitor:
        monitor.check("ibd_ready", monitor.counts)


def ibd_evidence(db, row, limit):
    pieces = {}
    for kind in ("intervals", "callability"):
        values = db.execute(f"SELECT start_pos,end_pos FROM {kind} WHERE sample_a=? AND sample_b=? AND start_pos<=? AND end_pos>=?",
                            (row["sample_a"], row["sample_b"], row["end_pos"], row["start_pos"])).fetchmany(limit + 1)
        require(len(values) <= limit, "IBD interval resource limit exceeded")
        pieces[kind] = intersect_intervals([(int(v[0]), int(v[1])) for v in values], [(row["start_pos"], row["end_pos"])])
    callable_bp = interval_bp(pieces["callability"])
    boundaries = dict(ibd_overlap_pieces="NA", ibd_first_overlap_pos="NA", ibd_last_overlap_pos="NA",
                      ibd_left_uncovered_bp="NA", ibd_right_uncovered_bp="NA")
    if not callable_bp:
        return dict(callable_bp="NA", ibd_union_bp="NA", ibd_fraction="NA", ibd_status="NO_EVALUABLE:no_pair_callable_territory", **boundaries)
    overlap_intervals = intersect_intervals(pieces["intervals"], pieces["callability"])
    overlap = interval_bp(overlap_intervals)
    boundaries["ibd_overlap_pieces"] = len(overlap_intervals)
    if overlap_intervals:
        boundaries.update(ibd_first_overlap_pos=overlap_intervals[0][0], ibd_last_overlap_pos=overlap_intervals[-1][1],
                          ibd_left_uncovered_bp=overlap_intervals[0][0] - row["start_pos"],
                          ibd_right_uncovered_bp=row["end_pos"] - overlap_intervals[-1][1])
    require(0 <= overlap <= callable_bp <= row["length_bp"], "Invalid IBD territory denominator")
    return dict(callable_bp=callable_bp, ibd_union_bp=overlap, ibd_fraction=overlap / callable_bp, ibd_status="OK", **boundaries)


def ibd_pair_territories(db, chrom, limit):
    """All declared pair territories, including pairs never represented by M14."""
    for a, b in db.execute("SELECT sample_a,sample_b FROM callability UNION SELECT sample_a,sample_b FROM intervals"):
        pieces = {}
        for kind in ("intervals", "callability"):
            rows = db.execute(f"SELECT start_pos,end_pos FROM {kind} WHERE sample_a=? AND sample_b=?", (a, b)).fetchmany(limit + 1)
            require(len(rows) <= limit, "IBD pair territory resource limit exceeded")
            pieces[kind] = merge_intervals([(int(v[0]), int(v[1])) for v in rows])
        callable_bp = interval_bp(pieces["callability"])
        row = dict(chrom=chrom, sample_a=a, sample_b=b, callable_bp="NA", ibd_union_bp="NA",
                   ibd_pieces="NA", ibd_status="NO_EVALUABLE:no_pair_callable_territory")
        if callable_bp:
            overlap = intersect_intervals(pieces["intervals"], pieces["callability"])
            row.update(callable_bp=callable_bp, ibd_union_bp=interval_bp(overlap), ibd_pieces=len(overlap), ibd_status="OK")
        yield row


def vcf_contig(source, chrom):
    candidates = [c for c in source.header.contigs if c in (chrom, "chr" + chrom)]
    require(len(candidates) == 1, "Missing or ambiguous VCF chromosome aliases")
    require(source.index is not None, "Indexed VCF/BCF required")
    return candidates[0]


def common_dosages(record, samples, criteria):
    if (list(record.filter) != ["PASS"] or record.info.get("ORIG_NALLELES") != 2 or
            len(record.alleles) != 2 or not all(len(a) == 1 and a in "ACGT" for a in record.alleles)):
        return None
    values = []
    for sample in samples:
        gt = record.samples[sample].get("GT", ())
        require(len(gt) == 2 and all(a in (None, 0, 1) for a in gt), "Invalid common diploid GT")
        values.append(-1 if None in gt else sum(gt))
    called = [v for v in values if v >= 0]
    if not called or (len(values) - len(called)) / len(values) > criteria["max_missing"]:
        return None
    p = sum(called) / (2 * len(called))
    return values if min(p, 1 - p) >= criteria["min_maf"] else None


def count_chunk(rows, positions, genotypes, samples_index, kind):
    """Prefix counts once per observed pair, then interval queries, int64 sums."""
    positions = np.asarray(positions, dtype=np.int64)
    genotypes = np.asarray(genotypes, dtype=np.int8).T
    grouped = defaultdict(list)
    for row in rows:
        if row["start_pos"] <= positions[-1] and row["end_pos"] >= positions[0]:
            grouped[(row["sample_a"], row["sample_b"])].append(row)
    for (a, b), chains in grouped.items():
        ga, gb = genotypes[samples_index[a]], genotypes[samples_index[b]]
        joint = (ga >= 0) & (gb >= 0)
        if kind == "rare":
            both, either = joint & (ga > 0) & (gb > 0), joint & ((ga > 0) | (gb > 0))
            data = dict(I=both, U=either, Q=joint,
                        rare_missing_gt_count=(ga < 0).astype(np.int8) + (gb < 0))
        else:
            data = dict(O=joint & (((ga == 0) & (gb == 2)) | ((ga == 2) & (gb == 0))), Q_C=joint)
        prefixes = {key: np.concatenate(([0], np.cumsum(value, dtype=np.int64))) for key, value in data.items()}
        for row in chains:
            left, right = np.searchsorted(positions, [row["start_pos"], row["end_pos"]], side="left")
            right = int(np.searchsorted(positions, row["end_pos"], side="right"))
            for key, prefix in prefixes.items():
                row[key] += int(prefix[right] - prefix[left])
            if kind == "rare":
                row["rare_catalog_sites"] += right - int(left)
                shared = positions[left:right][both[left:right]]
                if shared.size:
                    if row["last_shared"] is not None:
                        row["max_shared_gap"] = max(row["max_shared_gap"], int(shared[0]) - row["last_shared"])
                    if shared.size > 1:
                        row["max_shared_gap"] = max(row["max_shared_gap"], int(np.diff(shared).max()))
                    if row["first_shared"] is None:
                        row["first_shared"] = int(shared[0])
                    row["last_shared"] = int(shared[-1])


def annotate_vcf_blocks(db, source, contig, samples, block_bp, chunk_sites, max_active, kind, counts, criteria=None, monitor=None):
    bounds = db.execute("SELECT min(start_pos),max(end_pos) FROM chains").fetchone()
    if bounds[0] is None:
        return
    sample_indices = {sample: i for i, sample in enumerate(samples)}
    block_start = ((bounds[0] - 1) // block_bp) * block_bp + 1
    while block_start <= bounds[1]:
        block_end = block_start + block_bp - 1
        block_started = time.monotonic()
        if monitor:
            monitor.check(kind + "_block", counts, block_start=block_start, block_end=block_end)
        rows = db.execute("SELECT * FROM chains WHERE start_pos<=? AND end_pos>=?", (block_end, block_start)).fetchmany(max_active + 1)
        require(len(rows) <= max_active, "Active-chain resource limit exceeded; lower block-bp or raise explicit limit")
        if not rows:
            nxt = db.execute("SELECT min(start_pos) FROM chains WHERE start_pos>?", (block_end,)).fetchone()[0]
            if nxt is None:
                break
            block_start = ((nxt - 1) // block_bp) * block_bp + 1
            continue
        rows = [dict(row) for row in rows]
        counts[kind + "_occupied_blocks"] += 1
        counts["peak_active_chains"] = max(counts["peak_active_chains"], len(rows))
        positions, genotypes, previous = [], [], 0
        # pysam fetch uses 0-based half-open coordinates; VCF pos is 1-based.
        for record in source.fetch(contig, block_start - 1, block_end):
            counts[kind + "_records_visited"] += 1
            if kind == "rare":
                allele = validate_rare_site(record)
                calls = [rare_dosage(record.samples[s], allele) for s in samples]
                values = [-1 if v is None else v for v in calls]
            else:
                values = common_dosages(record, samples, criteria)
                if values is None:
                    continue
            require(block_start <= record.pos <= block_end and record.pos > previous, "Duplicate/decreasing/out-of-block eligible VCF position")
            previous = record.pos
            positions.append(record.pos)
            genotypes.append(values)
            if len(positions) == chunk_sites:
                count_chunk(rows, positions, genotypes, sample_indices, kind)
                positions, genotypes = [], []
                if monitor:
                    monitor.check(kind + "_block", counts, block_start=block_start, block_end=block_end)
        if positions:
            count_chunk(rows, positions, genotypes, sample_indices, kind)
        fields = ("I", "U", "Q", "rare_catalog_sites", "rare_missing_gt_count", "first_shared", "last_shared", "max_shared_gap") if kind == "rare" else ("O", "Q_C")
        db.executemany("UPDATE chains SET " + ",".join(k + "=?" for k in fields) + " WHERE chain_id=?",
                       [tuple(row[k] for k in fields) + (row["chain_id"],) for row in rows])
        db.commit()
        if monitor:
            monitor.check(kind + "_block_complete", counts, block_start=block_start,
                          block_end=block_end, block_seconds=time.monotonic() - block_started)
        block_start = block_end + 1


def unchanged_inputs(inputs):
    for item in inputs.values():
        stat = Path(item["path"]).stat()
        require(stat.st_size == item["bytes"] and stat.st_mtime_ns == item["mtime_ns"],
                "Input changed while calculating")


def preflight_geometry(args, samples, chrom, inputs, rare_index, configs):
    """Two streaming chain passes, bounded pair/block bitset, no SQLite or GT.

    Chain row counts conservatively include duplicates; pair/block bits do not.
    No deduplication, maximality or genotype-concordance claim is made. Avoiding
    the expensive SQLite load means there is no preparation cache to reuse.
    """
    import pysam
    require(not any((args.common_vcf, args.genetic_map, args.ibd_contract)),
            "Geometry-only preflight accepts rare inputs only; optional evidence is not inspected")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    counts, events = Counter(), Counter()
    order = {name: i for i, name in enumerate(samples)}
    with ResourceProgress(output, args) as monitor, pysam.VariantFile(args.rare_vcf, index_filename=rare_index) as rare:
        source_samples = validate_rare_header(rare.header, samples, args.expected_source_samples)
        contig = vcf_contig(rare, chrom)
        contig_length = rare.header.contigs[contig].length
        max_block = -1
        for row in table_rows(args.chains, CHAIN_COLUMNS[1:]):
            _, _, _, lo, hi, _, _ = chain_geometry(row, order, chrom)
            require(not contig_length or hi <= contig_length, "Chain exceeds declared VCF contig length")
            first, last = (lo - 1) // args.block_bp, (hi - 1) // args.block_bp
            require(last < args.max_preflight_blocks, "Preflight block resource limit exceeded")
            events[first] += 1
            events[last + 1] -= 1
            max_block = max(max_block, last)
            counts["input_chain_rows"] += 1
            if counts["input_chain_rows"] % args.resource_check_rows == 0:
                monitor.check("preflight_chain_geometry", counts)
        monitor.check("preflight_chain_geometry_complete", counts)
        n_blocks, n_pairs = max_block + 1, len(samples) * (len(samples) - 1) // 2
        words = (n_blocks + 63) // 64
        # Include the reduction scratch and block counter array in the explicit
        # NumPy budget. Python/runtime overhead is measured separately by RSS.
        buffer_bytes = words * n_pairs * 8 + n_blocks * 8 + min(n_pairs, 65536) * 8
        require(buffer_bytes <= args.max_preflight_memory_mb * 1024**2,
                "Preflight NumPy-buffer resource limit exceeded")
        bits = np.zeros((words, n_pairs), dtype=np.uint64)
        counts["preflight_numpy_buffer_bytes"] = buffer_bytes
        counts["possible_pairs"] = n_pairs
        for row in table_rows(args.chains, CHAIN_COLUMNS[1:]):
            a, b, _, lo, hi, _, _ = chain_geometry(row, order, chrom)
            i, j = order[a], order[b]
            pair = i * len(samples) - i * (i + 1) // 2 + j - i - 1
            first, last = (lo - 1) // args.block_bp, (hi - 1) // args.block_bp
            require(last < n_blocks, "Input chain bounds changed between preflight passes")
            for word in range(first // 64, last // 64 + 1):
                left, right = max(first, word * 64) % 64, min(last, word * 64 + 63) % 64
                bits[word, pair] |= np.uint64(((1 << (right - left + 1)) - 1) << left)
            counts["pair_bitset_chain_rows"] += 1
            if counts["pair_bitset_chain_rows"] % args.resource_check_rows == 0:
                monitor.check("preflight_pair_blocks", counts)
        require(counts["pair_bitset_chain_rows"] == counts["input_chain_rows"],
                "Input chain row count changed between preflight passes")
        active = 0
        with (output / "preflight_blocks.tsv").open("x", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("block_start", "block_end", "active_chain_rows_upper_bound", "distinct_pairs"))
            for block in range(n_blocks):
                active += events.get(block, 0)
                if not active:
                    continue
                mask = np.uint64(1 << (block % 64))
                pairs = sum(int(np.count_nonzero(bits[block // 64, p:p + 65536] & mask))
                            for p in range(0, n_pairs, 65536))
                writer.writerow((block * args.block_bp + 1, (block + 1) * args.block_bp, active, pairs))
                counts["occupied_blocks"] += 1
                counts["chain_row_blocks_upper_bound"] += active
                counts["distinct_pair_blocks"] += pairs
                counts["peak_active_chain_rows_upper_bound"] = max(counts["peak_active_chain_rows_upper_bound"], active)
                counts["peak_distinct_pairs_per_block"] = max(counts["peak_distinct_pairs_per_block"], pairs)
                if active > args.max_active_intervals:
                    counts["blocks_exceeding_active_row_bound"] += 1
                monitor.check("preflight_block_summary", counts, block_start=block * args.block_bp + 1)
        unchanged_inputs(inputs)
        measured = monitor.check("preflight_complete", counts)
        report = dict(schema="r02_segment_preflight_v1", status="PREFLIGHT_GEOMETRY_ONLY_NOT_EVIDENCE",
            chrom=chrom, genome_build=args.genome_build, n_samples=len(samples),
            source_sample_count=len(source_samples), n_configurations=len(configs),
            inputs=inputs, parameters={k: getattr(args, k) for k in ("block_bp", "chunk_sites", "max_active_intervals") + RESOURCE_PARAMETERS},
            checks=dict(counts=dict(counts), all_chain_rows_geometry_checked=True,
                rare_header_and_index_opened=True, genotype_records_read=0,
                duplicate_and_maximal_chain_validation=False, configuration_summary_totals_validated=False),
            measurement=measured,
            limitations=["Geometry only; not M14.2 scientific evidence or production feasibility certification",
                "Active chain rows include duplicates (upper bound); distinct pair-block counts are exact for these rows",
                "No genotype decoding, locus counts, I/U/Q, local J, SQLite-size measurement or runtime prediction",
                "NumPy buffer budget excludes Python/runtime overhead; process peak RSS is recorded",
                "No SQLite preparation performed; full execution must load and validate chains; no mid-task resume"],
            source_sha256={Path(__file__).name: sha256(__file__), "r02_genomic_pair_evidence.py": sha256(Path(__file__).with_name("r02_genomic_pair_evidence.py"))},
            versions=dict(python=sys.version.split()[0], pysam=pysam.__version__, numpy=np.__version__),
            outputs_sha256={"preflight_blocks.tsv": sha256(output / "preflight_blocks.tsv")})
        report["measurement"] = monitor.emit(report["status"])
        report["operational_outputs_sha256"] = {name: sha256(output / name)
                                                for name in ("progress.json", "progress.jsonl")}
        with (output / "preflight.json").open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return report


def run(args):
    import pysam
    started = time.monotonic()
    for key in ("block_bp", "chunk_sites", "max_active_intervals", "max_map_knots", "expected_source_samples", "resource_check_rows", "max_preflight_blocks"):
        require(getattr(args, key) > 0, f"{key} must be positive")
    for key in ("max_db_mb", "max_preflight_memory_mb", "min_free_disk_mb"):
        value = getattr(args, key)
        require(math.isfinite(value) and (value >= 0 if key == "min_free_disk_mb" else value > 0),
                f"{key} must be finite and {'nonnegative' if key == 'min_free_disk_mb' else 'positive'}")
    require(args.genome_build.strip(), "Genome build must be explicit")
    require(bool(args.common_vcf) == bool(args.common_contract), "common-vcf and common-contract must be paired")
    require(bool(args.genetic_map) == bool(args.map_contract), "genetic-map and map-contract must be paired")
    output = Path(args.output_dir)
    require(not output.exists(), "Output exists; never overwrite a prior result")
    samples, chrom = read_samples(args.samples), chromosome(args.chromosome)
    inputs = {key: input_record(getattr(args, key)) for key in ("chains", "configurations", "rare_vcf", "samples")}
    rare_index = indexed_input(args.rare_vcf, inputs, "rare_vcf")
    configs = load_configurations(args.configurations)
    if args.preflight_only:
        return preflight_geometry(args, samples, chrom, inputs, rare_index, configs)
    common_contract = map_contract = ibd_contract = genetic_map = None
    if args.common_vcf:
        inputs["common_vcf"] = input_record(args.common_vcf)
        common_index = indexed_input(args.common_vcf, inputs, "common_vcf")
        common_contract = checked_contract(args.common_contract, "r02_segment_common_v1", args.genome_build, inputs, "common_contract")
        require(common_contract.get("vcf_sha256") == inputs["common_vcf"]["sha256"], "Common VCF hash mismatch")
        require(common_contract.get("analytical_samples_sha256") == sample_hash(samples), "Common analytical cohort/order mismatch")
        require(common_contract.get("frequency_scope") == "analytical_complete_diploid" and common_contract.get("mask") == "complete_diploid_GT", "Unsupported common frequency/mask semantics")
        require(common_contract.get("site_filter") == "PASS" and common_contract.get("original_biallelic") is True, "Common PASS/original biallelic contract required")
        require(0 < common_contract.get("min_maf", 0) <= .5 and 0 <= common_contract.get("max_missing", -1) < 1, "Invalid explicit common thresholds")
    if args.genetic_map:
        inputs["genetic_map"] = input_record(args.genetic_map)
        map_contract = checked_contract(args.map_contract, "r02_genetic_map_v1", args.genome_build, inputs, "map_contract")
        require(map_contract.get("sha256") == inputs["genetic_map"]["sha256"] and map_contract.get("units") == "cM" and map_contract.get("format") == "tsv_chrom_pos_cm", "Map hash/units/format mismatch")
        genetic_map = GeneticMap(args.genetic_map, chrom, args.max_map_knots)
    if args.ibd_contract:
        ibd_contract = checked_contract(args.ibd_contract, "r02_segment_ibd_v1", args.genome_build, inputs, "ibd_contract")
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        monitor = stack.enter_context(ResourceProgress(output, args))
        temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix=".m142-", dir=output))
        db = create_database(Path(temporary) / "intervals.sqlite", monitor)
        stack.callback(db.close)
        counts = load_chains(db, args.chains, configs, samples, chrom, monitor)
        rare = stack.enter_context(pysam.VariantFile(args.rare_vcf, index_filename=rare_index))
        source_samples = validate_rare_header(rare.header, samples, args.expected_source_samples)
        source_contract = {key: _header_scalar(rare.header, key) for key in ("dnabr_original_alleles", "dnabr_rare_contract", "dnabr_rare_cohort_n_samples", "dnabr_rare_cohort_sha256")}
        source_contract["dnabr_rare_cohort_n_samples"] = int(source_contract["dnabr_rare_cohort_n_samples"])
        contig = vcf_contig(rare, chrom)
        rare.subset_samples(samples)
        annotate_vcf_blocks(db, rare, contig, samples, args.block_bp, args.chunk_sites, args.max_active_intervals, "rare", counts, monitor=monitor)
        if common_contract:
            common = stack.enter_context(pysam.VariantFile(args.common_vcf, index_filename=common_index))
            require(_header_scalar(common.header, "dnabr_original_alleles") == "v1" and "ORIG_NALLELES" in common.header.info and "GT" in common.header.formats, "Uncertified common original alleles/GT")
            require(set(samples) <= set(common.header.samples), "Common VCF lacks analytical samples")
            common_contig = vcf_contig(common, chrom)
            common.subset_samples(samples)
            annotate_vcf_blocks(db, common, common_contig, samples, args.block_bp, args.chunk_sites, args.max_active_intervals, "common", counts, common_contract, monitor)
        if ibd_contract:
            load_ibd(db, args.ibd_contract, ibd_contract, samples, chrom, inputs, monitor)
            territories = write_table(stack, output / "ibd_pair_territory.tsv.gz", IBD_PAIR_COLUMNS)
            for row in ibd_pair_territories(db, chrom, args.max_active_intervals):
                territories.writerow(row)
                counts["ibd_pair_territory_rows"] += 1
                if counts["ibd_pair_territory_rows"] % args.resource_check_rows == 0:
                    monitor.check("write_ibd_territory", counts)
        evidence = write_table(stack, output / "segment_evidence.tsv.gz", EVIDENCE_COLUMNS)
        monitor.check("write_segment_evidence", counts)
        for row in db.execute("SELECT * FROM chains ORDER BY start_pos,end_pos,sample_a,sample_b,max_gap_bp"):
            row = dict(row)
            require(0 <= row["I"] <= row["U"] <= row["Q"] <= row["rare_catalog_sites"], "Invalid local rare denominators")
            require(row["I"] == row["n_shared_variants"] and row["first_shared"] == row["start_pos"] and row["last_shared"] == row["end_pos"] and row["max_shared_gap"] <= row["max_gap_bp"], "M14 chain and source rare genotypes disagree")
            value = {key: row[key] for key in CHAIN_COLUMNS + ("I", "U", "Q", "rare_catalog_sites", "rare_missing_gt_count")}
            value.update(max_shared_gap_bp=row["max_shared_gap"], shared_site_density_per_bp=row["n_shared_variants"] / row["length_bp"])
            value.update(J=row["I"] / row["U"] if row["U"] else "NA", rare_status="OK" if row["U"] else "NO_EVALUABLE:zero_union")
            value.update(O=row["O"] if common_contract else "NA", Q_C=row["Q_C"] if common_contract else "NA", common_status=("OK" if row["Q_C"] else "NO_EVALUABLE:no_joint_common_calls") if common_contract else "NO_EVALUABLE:no_common_catalogue")
            cm = genetic_map.length(row["start_pos"], row["end_pos"]) if genetic_map else None
            value.update(length_cm=cm if cm is not None else "NA", map_status="OK" if cm is not None else "NO_EVALUABLE:outside_map_support" if genetic_map else "NO_EVALUABLE:no_map")
            value.update(ibd_evidence(db, row, args.max_active_intervals) if ibd_contract else dict(callable_bp="NA", ibd_union_bp="NA", ibd_fraction="NA", ibd_status="NO_EVALUABLE:no_ibd_contract",
                **{k: "NA" for k in ("ibd_overlap_pieces", "ibd_first_overlap_pos", "ibd_last_overlap_pos", "ibd_left_uncovered_bp", "ibd_right_uncovered_bp")}))
            evidence.writerow(value)
            counts["evidence_rows"] += 1
            if counts["evidence_rows"] % args.resource_check_rows == 0:
                monitor.check("write_segment_evidence", counts)
        links = write_table(stack, output / "segment_configuration_links.tsv.gz", ("chain_id", "config_id"))
        for row in db.execute("SELECT * FROM links ORDER BY config_id,chain_id"):
            links.writerow(dict(row))
            counts["configuration_links"] += 1
            if counts["configuration_links"] % args.resource_check_rows == 0:
                monitor.check("write_configuration_links", counts)
        config_writer = write_table(stack, output / "configurations.tsv", CONFIG_COLUMNS)
        config_writer.writerows({key: row[key] for key in CONFIG_COLUMNS} for row in configs)
        monitor.check("outputs_written", counts)
    # Manifest is the completion marker; partial files without it are unusable.
    unchanged_inputs(inputs)
    manifest = dict(schema=SCHEMA, status="COMPLETE_DESCRIPTIVE_NOT_VALIDATED", chrom=chrom,
        genome_build=args.genome_build, coordinate_system=COORDINATES,
        n_samples=len(samples), sample_ids_sha256=inputs["samples"]["sha256"],
        analytical_samples_sha256=sample_hash(samples), analytical_members_sha256=sample_hash(sorted(samples)),
        source_rare_contract=source_contract, source_members_sha256=sample_hash(sorted(source_samples)),
        inputs=inputs, parameters={key: getattr(args, key) for key in ("block_bp", "chunk_sites", "max_active_intervals", "max_map_knots", "expected_source_samples") + RESOURCE_PARAMETERS},
        evidence_criteria=dict(rare_allele="fixed_source_minor_minor_v1", rare_gt="complete_diploid_GT_RD_validated",
            rare_missingness="joint_complete_calls_no_imputation", common_criteria={k: common_contract[k] for k in COMMON_CRITERIA} if common_contract else None,
            ibd_quantity="any_copy_ibd_territory" if ibd_contract else None,
            ibd_criteria={k: v for k, v in ibd_contract.items() if k not in ("schema", "genome_build", "coordinate_system", "intervals", "callability")} if ibd_contract else None,
            coordinate_system=COORDINATES, quality_filters="new_DP_GQ_filters_not_applied"),
        checks=dict(counts=counts, chain_shared_sites_and_endpoints_match=True, within_pair_gap_chains_disjoint=True,
            configuration_thresholds_and_available_summary_counts_match=True, no_all_pairs_segment_array=True,
            build_validation="caller-declared genome build; optional map/IBD contracts must match; no FASTA revalidation",
            validation_scope="Only occupied VCF blocks read; no claim about unvisited sites"),
        semantics=dict(J="I/U on jointly complete GT within each inclusive interval; U=0 gives NA",
            Q="number of jointly called rare catalogue sites, not callable bases",
            rare_missing_gt_count="incomplete GT among two persons; denominator 2*rare_catalog_sites",
            common="O=opposite homozygotes; Q_C=joint complete common calls; not certified IBD",
            ibd="union of IBD1/IBD2 territory intersected with pair-callable intervals; no copy-length sum",
            ibd_recall="ibd_pair_territory.tsv.gz provides whole-chromosome caller union and callable territory for ALL declared pairs, including no-M14 pairs" if ibd_contract else "NO_EVALUABLE:no_ibd_contract",
            ibd_overlap_pieces="connected components of caller-union intersected with pair-callability and chain; fragmentation can arise from callability gaps",
            ibd_terminal_spans="chain start to first covered base and last covered base to chain end; may mix non-IBD and uncallable territory; NA if no overlap; NOT true-breakpoint error",
            max_shared_gap_bp="largest consecutive shared-site distance; 0 for one shared site",
            shared_site_density_per_bp="n_shared_variants divided by inclusive length_bp",
            map="linear interpolation within measured support; no extrapolation",
            weights="annotation only; does not replace H weights or multiply H by J"),
        source_sha256={Path(__file__).name: sha256(__file__), "r02_genomic_pair_evidence.py": sha256(Path(__file__).with_name("r02_genomic_pair_evidence.py"))},
        versions=dict(python=sys.version.split()[0], pysam=pysam.__version__, numpy=np.__version__),
        outputs_sha256={name: sha256(output / name) for name in ("segment_evidence.tsv.gz", "segment_configuration_links.tsv.gz", "configurations.tsv") + (("ibd_pair_territory.tsv.gz",) if ibd_contract else ())},
        elapsed_seconds=time.monotonic() - started)
    monitor.emit("COMPLETE_DESCRIPTIVE_NOT_VALIDATED")
    manifest["operational_outputs_sha256"] = {name: sha256(output / name)
                                               for name in ("progress.json", "progress.jsonl")}
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ("chains", "configurations", "rare-vcf", "samples", "chromosome", "genome-build", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--expected-source-samples", type=int, required=True)
    for option in ("common-vcf", "common-contract", "genetic-map", "map-contract", "ibd-contract"):
        parser.add_argument("--" + option)
    parser.add_argument("--block-bp", type=int, default=1_000_000, help="Operational indexed-read block size, bp; not a biological window")
    parser.add_argument("--chunk-sites", type=int, default=2048, help="Maximum genotype rows held at once")
    parser.add_argument("--max-active-intervals", type=int, default=100_000, help="Fail, never truncate, beyond this active-chain/IBD-query bound")
    parser.add_argument("--max-map-knots", type=int, default=1_000_000)
    parser.add_argument("--preflight-only", action="store_true", help="Geometry/header/index inspection only; no genotypes, SQLite, or evidence manifest")
    parser.add_argument("--max-db-mb", type=float, default=8192, help="SQLite main+journal bound, MiB; operational limit, not measured capacity")
    parser.add_argument("--min-free-disk-mb", type=float, default=1024, help="Periodic free-disk reserve, MiB")
    parser.add_argument("--resource-check-rows", type=int, default=10000, help="Resource/progress check interval while loading chain rows")
    parser.add_argument("--max-preflight-memory-mb", type=float, default=256, help="Preflight NumPy buffers only, MiB; RSS includes additional runtime overhead")
    parser.add_argument("--max-preflight-blocks", type=int, default=10000, help="Bound chromosome-coordinate block slots in geometry preflight")
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, sqlite3.Error, MemoryError) as exc:
        print(f"M14.2 FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(status=result["status"], counts=result["checks"]["counts"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
