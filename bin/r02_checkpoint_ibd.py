#!/usr/bin/env python3
"""Compare existing M14 checkpoints with reported IBD, without genotype scans.

This is geometric corroboration, not sensitivity/specificity against truth.
The IBD union counts each physical base once per pair, regardless of copies.
Chromosome tasks retain all pairs; accumulated-length thresholds are applied
only after the autosomal merge. Unknown callability remains unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import heapq
import itertools
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import tempfile
import time

import r02_segment_evidence as segment
import r02_genomic_pair_evidence as genomic

require, sha256 = segment.require, segment.sha256
SCHEMA = "r02_checkpoint_reported_ibd_v1"
CHROM_STATUS = "COMPLETE_REPORTED_GEOMETRY_NOT_BIOLOGICAL_VALIDATION"
FIELDS = ("a", "b", "config_id", "chrom", "m14_union_bp", "ibd_reported_overlap_bp")


def read_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON field")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("Nonfinite JSON value")
    with Path(path).open() as handle:
        return json.load(handle, object_pairs_hook=unique, parse_constant=nonfinite)


def save_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, sort_keys=True)
        handle.write("\n")


def checked(item, parent):
    require(set(item) >= {"path", "sha256"}, "Authenticated input required")
    path = (Path(parent) / item["path"]).resolve()
    require(path.is_file() and sha256(path) == item["sha256"], "Input SHA256 mismatch")
    return path


def configs_checked(configs):
    require(isinstance(configs, dict) and configs, "Configuration ledger missing")
    for name, row in configs.items():
        require(row["config_id"] == name, "Configuration identity mismatch")
        require(all(type(row[k]) is int and row[k] > 0 for k in
                    ("max_gap_bp", "min_length_bp", "min_shared_effective")), "Invalid configuration")
    return configs


def checkpoint_rows(path, chrom, n_samples, expected_count, configs):
    needed = {"a", "b", "chrom", "start_pos", "end_pos", "length_bp",
              "max_gap_bp", "n_shared_variants"}
    previous, count, pair = None, 0, None
    ends = {}
    by_gap = {}
    for name, config in configs.items():
        by_gap.setdefault(config["max_gap_bp"], []).append((name, config))
    with gzip.open(path, "rt") as handle:
        fields = json.loads(next(handle, "null"))
        require(isinstance(fields, list) and len(fields) == len(set(fields)) and
                needed <= set(fields), "Checkpoint columns missing/duplicated")
        for line in handle:
            values = json.loads(line)
            require(isinstance(values, list) and len(values) == len(fields), "Checkpoint width mismatch")
            row = dict(zip(fields, values))
            require(all(type(row[k]) is int for k in needed), "Noninteger checkpoint geometry")
            require(row["chrom"] == int(chrom) and 0 <= row["a"] < row["b"] < n_samples,
                    "Chromosome/sample index mismatch")
            require(1 <= row["start_pos"] <= row["end_pos"] and
                    row["length_bp"] == row["end_pos"] - row["start_pos"] + 1 and
                    row["n_shared_variants"] > 0, "Invalid inclusive interval")
            key = tuple(row[k] for k in ("a", "b", "chrom", "start_pos", "end_pos", "max_gap_bp"))
            require(previous is None or key > previous, "Unsorted/duplicated checkpoint rows")
            if pair != (row["a"], row["b"]):
                pair, ends = (row["a"], row["b"]), {}
            names = []
            require(row["max_gap_bp"] in by_gap, "Unknown checkpoint gap")
            for name, config in by_gap[row["max_gap_bp"]]:
                if row["length_bp"] >= config["min_length_bp"] and row["n_shared_variants"] >= config["min_shared_effective"]:
                    require(name not in ends or row["start_pos"] > ends[name],
                            "Overlapping M14 intervals within configuration/pair")
                    ends[name] = row["end_pos"]
                    names.append(name)
            previous, count = key, count + 1
            yield row, names
    require(count == expected_count, "Checkpoint row count mismatch")


def run_chromosome(contract_path, expected_sha, output_dir, scratch_dir=None,
                   min_free_mb=1024, max_pair_intervals=100000, max_db_mb=1024):
    require(min_free_mb >= 0 and max_pair_intervals > 0 and max_db_mb > 0, "Invalid resource limits")
    contract_path = Path(contract_path)
    require(sha256(contract_path) == expected_sha, "Contract SHA256 mismatch")
    contract = read_json(contract_path)
    require(contract.get("schema") == "r02_checkpoint_ibd_contract_v1", "Unsupported contract")
    parent = contract_path.parent
    samples_path = checked(contract["samples"], parent)
    samples = segment.read_samples(samples_path, contract["n_samples"])
    cp_path = checked(contract["checkpoint"], parent)
    cp = read_json(cp_path)
    chrom = segment.chromosome(contract["chromosome"])
    require(cp.get("schema") == "r02_biological_chromosome_checkpoint_v1" and
            cp.get("status") == "COMPLETE_TECHNICAL_CHECKPOINT_NOT_BIOLOGICAL_RESULT", "Incomplete checkpoint")
    require(cp["chrom"] == chrom and cp["sample_ids_sha256"] == sha256(samples_path) and
            cp["n_samples"] == len(samples), "Checkpoint cohort/chromosome mismatch")
    require(cp["code_sha256"] == contract["checkpoint_code_sha256"] and
            cp["registry"]["identity"] == contract["identity"], "Checkpoint producer/identity mismatch")
    require(cp["no_local_accumulated_bp_threshold"] is True, "Chromosome-local threshold already applied")
    require(cp["registry"]["chromosomes"] == [chrom] and
            cp["registry"]["sources"][0]["chrom"] == chrom, "Checkpoint source chromosome mismatch")
    configs = configs_checked(cp["registry"]["configs"])
    require(len(configs) == contract["n_configurations"] == cp["n_configurations"], "Configuration count mismatch")
    name = cp["streams"]["segments"]
    require(name == f"chr{chrom}.segments.jsonl.gz", "Invalid checkpoint stream name")
    stream = checked(dict(path=name, sha256=cp["outputs"][name]["sha256"]), cp_path.parent)
    require(stream.stat().st_size == cp["outputs"][name]["bytes"], "Checkpoint stream size mismatch")
    ibd_path = checked(contract["ibd_contract"], parent)
    ibd = read_json(ibd_path)
    require(ibd.get("schema") == "r02_segment_ibd_v2" and
            ibd["genome_build"] == contract["identity"]["genome_build"], "Reported IBD build/schema mismatch")
    ibd_intervals = checked(ibd["intervals"], ibd_path.parent)
    immutable = {p: sha256(p) for p in (contract_path, samples_path, cp_path, stream, ibd_path, ibd_intervals)}
    expected = {contract_path: expected_sha, samples_path: contract['samples']['sha256'],
        cp_path: contract['checkpoint']['sha256'], stream: cp['outputs'][name]['sha256'],
        ibd_path: contract['ibd_contract']['sha256'], ibd_intervals: ibd['intervals']['sha256']}
    require(immutable == expected, 'Input changed during authentication')
    output = Path(output_dir)
    require(not output.exists(), "Output exists; no overwrite")
    output.mkdir(parents=True)
    started = time.monotonic()
    def space():
        require(shutil.disk_usage(output).free >= min_free_mb * 1024**2, "Free disk reserve reached")
        if scratch_dir is not None:
            require(shutil.disk_usage(scratch_dir).free >= min_free_mb * 1024**2, "Scratch disk reserve reached")
    inputs = {}
    counts = Counter()
    with tempfile.TemporaryDirectory(dir=scratch_dir, prefix="reported-ibd-") as tmp:
        db = sqlite3.connect(str(Path(tmp) / "ibd.sqlite"))
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute(f"PRAGMA max_page_count={int(max_db_mb * 1024**2 / 4096)}")
            space()
            segment.load_ibd(db, ibd_path, ibd, samples, chrom, inputs)
            with gzip.open(output / "ibd_territory.tsv.gz", "xt") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(("a", "b", "chrom", "ibd_reported_union_bp"))
                query = "SELECT x.ordinal,y.ordinal,i.start_pos,i.end_pos FROM intervals i JOIN ibd_sample_order x ON i.sample_a=x.sample JOIN ibd_sample_order y ON i.sample_b=y.sample ORDER BY x.ordinal,y.ordinal,i.start_pos,i.end_pos"
                for (a, b), rows in itertools.groupby(db.execute(query), lambda row: row[:2]):
                    intervals = []
                    for row in rows:
                        intervals.append((row[2], row[3]))
                        require(len(intervals) <= max_pair_intervals, "Pair IBD resource limit")
                    bp = segment.interval_bp(segment.merge_intervals(intervals))
                    writer.writerow((a, b, chrom, bp))
                    counts["ibd_pairs"] += 1
                    counts["ibd_reported_union_bp"] += bp
            with gzip.open(output / "pair_configurations.tsv.gz", "xt") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(FIELDS)
                rows = checkpoint_rows(stream, chrom, len(samples), cp["registry"]["sources"][0]["n_chains"], configs)
                for (a, b), group in itertools.groupby(rows, lambda value: (value[0]["a"], value[0]["b"])):
                    called = db.execute("SELECT start_pos,end_pos FROM intervals WHERE sample_a=? AND sample_b=?",
                                        (samples[a], samples[b])).fetchmany(max_pair_intervals + 1)
                    require(len(called) <= max_pair_intervals, "Pair IBD resource limit")
                    called = segment.merge_intervals(called)
                    totals = {}
                    for row, memberships in group:
                        counts["checkpoint_rows"] += 1
                        lo, hi = row["start_pos"], row["end_pos"]
                        overlap = segment.interval_bp(segment.intersect_intervals([(lo, hi)], called))
                        for config_id in memberships:
                            value = totals.setdefault(config_id, [0, 0])
                            value[0] += row["length_bp"]
                            value[1] += overlap
                    for config_id, (m_bp, overlap) in sorted(totals.items()):
                        require(0 <= overlap <= m_bp, "Invalid overlap")
                        writer.writerow((a, b, config_id, chrom, m_bp, overlap))
                        counts["pair_configuration_rows"] += 1
                    counts["m14_pairs"] += 1
                    if counts["m14_pairs"] % 4096 == 0:
                        space()
        finally:
            db.close()
    space()
    require(all(sha256(p) == digest for p, digest in immutable.items()), "Input changed during execution")
    manifest = dict(schema=SCHEMA, status=CHROM_STATUS, chromosome=chrom,
        contract_sha256=expected_sha, cohort_sha256=segment.sample_hash(samples), n_samples=len(samples),
        identity=contract["identity"], configs=configs, counts=dict(counts),
        interpretation="reported_geometric_overlap_not_IBD_accuracy_or_population_validation",
        callability="UNKNOWN", copy_dosage="NOT_ESTIMATED", global_threshold_applied=False,
        inputs=dict(checkpoint=contract["checkpoint"], ibd_contract=contract["ibd_contract"], **inputs),
        outputs={p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in output.iterdir()},
        code_sha256={Path(module.__file__).name: sha256(module.__file__) for module in (segment, genomic)} |
                    {Path(__file__).name: sha256(__file__)},
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    manifest['outputs_sha256'] = {name: value['sha256'] for name, value in manifest['outputs'].items()}
    save_json(output / "manifest.json", manifest)
    return manifest


def stream_rows(path, required, chrom, n_samples, expected_count):
    with gzip.open(path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(set(required) == set(reader.fieldnames or ()), "Supplement stream schema mismatch")
        previous, count = None, 0
        for row in reader:
            for name in required:
                if name != "config_id":
                    row[name] = int(row[name])
            require(row["chrom"] == int(chrom) and 0 <= row["a"] < row["b"] < n_samples, "Invalid supplement pair/chromosome")
            key = row["a"], row["b"], row.get("config_id", "")
            require(previous is None or key > previous, "Supplement stream unsorted/duplicated")
            previous = key
            count += 1
            yield row
        require(count == expected_count, "Supplement row count mismatch")


def aggregate(manifests, expected_chromosomes, thresholds, output_dir, min_free_mb=1024):
    wanted = list(map(segment.chromosome, expected_chromosomes))
    require(wanted and len(set(wanted)) == len(wanted), "Chromosomes missing/duplicated")
    require(thresholds and 0 in thresholds and len(set(thresholds)) == len(thresholds) and
            all(type(t) is int and t >= 0 for t in thresholds), "Invalid accumulated-length thresholds")
    require(min_free_mb >= 0, "Invalid disk reserve")
    records, seen, streams, configs, cohort, identity = [], set(), [], None, None, None
    total_ibd_bp, n_samples, code, immutable = 0, None, None, {}
    for item in manifests:
        path = checked(item, Path.cwd())
        m = read_json(path)
        require(m.get("schema") == SCHEMA and m.get("status") == CHROM_STATUS, "Incomplete chromosome supplement")
        chrom = m["chromosome"]
        require(chrom in wanted and chrom not in seen, "Unexpected/duplicate chromosome")
        seen.add(chrom)
        if configs is None:
            configs, cohort, identity = m["configs"], m["cohort_sha256"], m["identity"]
            configs_checked(configs)
            n_samples, code = m["n_samples"], m["code_sha256"]
            require(type(n_samples) is int and n_samples > 1, "Invalid sample count")
        require(m["configs"] == configs and m["cohort_sha256"] == cohort and m["identity"] == identity,
                "Cross-chromosome configuration/cohort mismatch")
        require(m["n_samples"] == n_samples and m["code_sha256"] == code, "Cross-chromosome code/sample count mismatch")
        immutable[path] = item["sha256"]
        require(m["global_threshold_applied"] is False and m["callability"] == "UNKNOWN", "Supplement semantics changed")
        require(set(m["outputs"]) == {"ibd_territory.tsv.gz", "pair_configurations.tsv.gz"}, "Output inventory mismatch")
        for name, fields in (("ibd_territory.tsv.gz", ("a", "b", "chrom", "ibd_reported_union_bp")),
                             ("pair_configurations.tsv.gz", FIELDS)):
            p = checked(dict(path=name, sha256=m["outputs"][name]["sha256"]), path.parent)
            require(p.stat().st_size == m["outputs"][name]["bytes"], "Supplement byte count mismatch")
            count_field = "ibd_pairs" if name == "ibd_territory.tsv.gz" else "pair_configuration_rows"
            streams.append(stream_rows(p, fields, chrom, n_samples, m["counts"].get(count_field, 0)))
            immutable[p] = m["outputs"][name]["sha256"]
        total_ibd_bp += m["counts"].get("ibd_reported_union_bp", 0)
        records.append(dict(path=str(path), sha256=sha256(path)))
    require(seen == set(wanted), "Incomplete chromosome set")
    totals = {(name, t): Counter() for name in configs for t in sorted(thresholds)}
    try:
        merged = heapq.merge(*streams, key=lambda r: (r["a"], r["b"]))
        measured_ibd = 0
        for pair, group in itertools.groupby(merged, lambda r: (r["a"], r["b"])):
            b_bp, per_config = 0, {}
            for row in group:
                if "config_id" not in row:
                    require(row["ibd_reported_union_bp"] > 0, "Nonpositive reported IBD territory")
                    b_bp += row["ibd_reported_union_bp"]
                else:
                    require(row["config_id"] in configs and row["m14_union_bp"] > 0 and 0 <= row["ibd_reported_overlap_bp"] <= row["m14_union_bp"],
                            "Invalid configuration/overlap")
                    value = per_config.setdefault(row["config_id"], [0, 0])
                    value[0] += row["m14_union_bp"]
                    value[1] += row["ibd_reported_overlap_bp"]
            measured_ibd += b_bp
            for name, (m_bp, overlap) in per_config.items():
                require(overlap <= b_bp, "Overlap exceeds whole-genome reported IBD")
                for threshold in thresholds:
                    if m_bp >= threshold:
                        out = totals[name, threshold]
                        out.update(dict(selected_pairs=1, pairs_with_reported_ibd=int(b_bp > 0),
                            m14_union_bp=m_bp, ibd_reported_overlap_bp=overlap,
                            ibd_reported_bp_in_selected_pairs=b_bp))
        require(measured_ibd == total_ibd_bp, "IBD global denominator mismatch")
    finally:
        for stream in streams:
            stream.close()
    output = Path(output_dir)
    require(all(sha256(p) == digest for p, digest in immutable.items()), "Aggregate input changed during execution")
    require(not output.exists(), "Aggregate output exists; no overwrite")
    require(shutil.disk_usage(output.parent).free >= min_free_mb * 1024**2, "Insufficient disk reserve")
    output.mkdir(parents=True)
    rows = []
    for (name, threshold), values in sorted(totals.items()):
        m_bp, overlap, selected_ibd = (values[k] for k in
            ("m14_union_bp", "ibd_reported_overlap_bp", "ibd_reported_bp_in_selected_pairs"))
        rows.append(dict(config_id=name, min_autosomal_m14_bp=threshold,
            selected_pairs=values["selected_pairs"], pairs_with_reported_ibd=values["pairs_with_reported_ibd"],
            m14_union_bp=m_bp, ibd_reported_overlap_bp=overlap,
            ibd_reported_bp_in_selected_pairs=selected_ibd, ibd_reported_bp_all_pairs=total_ibd_bp,
            fraction_m14_overlapping_reported_ibd=overlap / m_bp if m_bp else "NA",
            fraction_reported_ibd_of_selected_pairs_overlapping_m14=overlap / selected_ibd if selected_ibd else "NA",
            fraction_all_reported_ibd_overlapping_selected_m14=overlap / total_ibd_bp if total_ibd_bp else "NA",
            status="DESCRIPTIVE_REPORTED_GEOMETRY_NOT_POPULATION_VALIDATION"))
    target = output / "configuration_ibd_geometry.tsv"
    with target.open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(schema="r02_checkpoint_ibd_aggregate_v1", chromosomes=sorted(wanted, key=int),
        thresholds_bp=sorted(thresholds), configurations=len(configs), cohort_sha256=cohort,
        status="COMPLETE_DESCRIPTIVE_REPORTED_GEOMETRY", input_manifests=records,
        output_sha256=sha256(target), callability="UNKNOWN", independent_validation=False,
        outputs_sha256={target.name:sha256(target)},
        global_threshold_applied_after_autosomal_sum=True,
        limitations=["IBD is an estimate, not truth", "No IBD1/IBD2 dosage", "No ancestry attribution of unphased rare alleles",
                     "M14-selected territory; not a population-structure test"])
    save_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    chrom = sub.add_parser("chromosome")
    chrom.add_argument("--contract", required=True)
    chrom.add_argument("--contract-sha256", required=True)
    chrom.add_argument("--output-dir", required=True)
    chrom.add_argument("--scratch-dir")
    chrom.add_argument("--min-free-mb", type=int, default=1024)
    chrom.add_argument("--max-pair-intervals", type=int, default=100000)
    chrom.add_argument("--max-db-mb", type=int, default=1024)
    reducer = sub.add_parser("aggregate")
    reducer.add_argument("--contract", required=True)
    reducer.add_argument("--contract-sha256", required=True)
    reducer.add_argument("--output-dir", required=True)
    reducer.add_argument("--min-free-mb", type=int, default=1024)
    collect = sub.add_parser("collect")
    collect.add_argument("--manifests", nargs="+", required=True)
    collect.add_argument("--chromosomes", nargs="+", required=True)
    collect.add_argument("--thresholds-bp", nargs="+", type=int, required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--min-free-mb", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.mode == "chromosome":
        result = run_chromosome(args.contract, args.contract_sha256, args.output_dir,
            args.scratch_dir, args.min_free_mb, args.max_pair_intervals, args.max_db_mb)
    elif args.mode == "collect":
        refs = [dict(path=str(Path(p).resolve()), sha256=sha256(p)) for p in args.manifests]
        result = aggregate(refs, args.chromosomes, args.thresholds_bp, args.output_dir, args.min_free_mb)
    else:
        require(sha256(args.contract) == args.contract_sha256, "Aggregate contract SHA256 mismatch")
        contract = read_json(args.contract)
        require(contract.get("schema") == "r02_checkpoint_ibd_aggregate_contract_v1", "Aggregate schema mismatch")
        result = aggregate(contract["manifests"], contract["chromosomes"], contract["thresholds_bp"],
                           args.output_dir, args.min_free_mb)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
