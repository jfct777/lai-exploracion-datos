#!/usr/bin/env python3
"""Combine authenticated per-chromosome M14 summaries, without pooling chromosomes as replicates.

This is the historical length channel only. Lengths remain summed candidate-chain
lengths, not unique IBD coverage or copy-aware IBD. SQLite bounds Python memory.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import m165_chr22_sweep as sweep

AUTOSOMES = [str(c) for c in range(1, 23)]
FIELDS = ("config_id", *sweep.PAIR_FIELDS)
SUM_FIELDS = ("n_segments", "total_shared_bp", "n_shared_variants_total")


def aggregate(manifest_path, settings_path, sample_ids, output):
    manifest = json.loads(Path(manifest_path).read_text())
    settings = json.loads(Path(settings_path).read_text())
    plans = sweep.validate_settings(settings)
    sources = {p["source_config_id"]: p for p in plans}
    samples = sweep.samples_from(sample_ids, settings["expected_samples"])
    sample_hash = sweep.sha256(sample_ids)
    sample_set = set(samples)
    expected_order_hash = hashlib.sha256(("\n".join(samples) + "\n").encode()).hexdigest()
    sweep.require(set(manifest) == {"schema_version", "chromosomes"} and manifest["schema_version"] == 1,
                  "Unexpected autosomal input manifest schema")
    entries = manifest["chromosomes"]
    sweep.require(isinstance(entries, list) and [e["chromosome"] for e in entries] == AUTOSOMES,
                  "Manifest must contain each of the 22 autosomes exactly once in order")
    output = Path(output)
    sweep.require(not output.exists(), "Aggregation output already exists; no overwrite")
    certified = []
    source_contract = None
    # Small metadata are checked before any large table is processed.
    for entry in entries:
        sweep.require(set(entry) == {"chromosome", "pair_summary", "configuration_summary", "summary", "sha256"},
                      "Unexpected chromosome manifest fields")
        sweep.require(set(entry["sha256"]) == {"pair_summary", "configuration_summary", "summary"},
                      "Incomplete per-chromosome hashes")
        for name in ("configuration_summary", "summary"):
            sweep.require(sweep.sha256(entry[name]) == entry["sha256"][name], "Input metadata hash mismatch")
        summary = json.loads(Path(entry["summary"]).read_text())
        sweep.require(str(summary["chrom"]).removeprefix("chr") == entry["chromosome"]
                      and summary["status"] == "COMPLETE_EXPLORATORY_NOT_VALIDATED"
                      and summary["n_samples"] == len(samples)
                      and summary["selected_samples_order_sha256"] == expected_order_hash
                      and summary["carrier_allele_mode"] == "source_minor",
                      "Per-chromosome cohort, orientation, completion or chromosome mismatch")
        contract = summary["source_rare_contract"]
        sweep.require(contract.get("contract") == "minor_v1", "Unsupported source rare contract")
        if source_contract is None:
            source_contract = contract
        sweep.require(contract == source_contract, "Source frequency cohort differs across chromosomes")
        with Path(entry["configuration_summary"]).open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        ids = [r["config_id"] for r in rows]
        sweep.require(len(ids) == len(set(ids)) and set(sources) <= set(ids),
                      "Missing or duplicate source configuration")
        wanted = {r["config_id"]: r for r in rows if r["config_id"] in sources}
        for config, row in wanted.items():
            plan = sources[config]
            sweep.require(int(row["min_length_bp"]) == plan["length_bp"]
                          and int(row["max_gap_bp"]) == plan["gap_bp"]
                          and int(row["min_shared_effective"]) == plan["min_shared_effective"],
                          "Source configuration identity differs from numeric thresholds")
        certified.append((entry, wanted))
    output.mkdir(parents=True, exist_ok=False)
    database = output / "pair_accumulation.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("CREATE TABLE totals (config TEXT, a TEXT, b TEXT, n INTEGER, bp INTEGER, v INTEGER, longest INTEGER, PRIMARY KEY(config,a,b)) WITHOUT ROWID")
    connection.execute("CREATE TABLE seen (config TEXT, a TEXT, b TEXT, PRIMARY KEY(config,a,b)) WITHOUT ROWID")
    audit = []
    try:
        for entry, wanted in certified:
            connection.execute("DELETE FROM seen")
            counters = {c: dict(n_pairs=0, **{k: 0 for k in SUM_FIELDS}) for c in sources}
            scanned = 0
            with Path(entry["pair_summary"]).open("rb") as raw:
                hashed = sweep.HashingReader(raw)
                with gzip.GzipFile(fileobj=hashed, mode="rb") as zipped, io.TextIOWrapper(zipped) as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    sweep.require(set(FIELDS) <= set(reader.fieldnames or []), "Missing pair-summary columns")
                    for row in reader:
                        scanned += 1
                        config = row["config_id"]
                        if config not in sources:
                            continue
                        a, b = sorted((row["sample_a"], row["sample_b"]))
                        sweep.require(a != b and {a, b} <= sample_set, "Self-pair or pair outside cohort")
                        values = [int(row[k]) for k in (*SUM_FIELDS, "max_segment_bp")]
                        n, bp, variants, longest = values
                        sweep.require(min(values) > 0 and longest <= bp and bp <= n * longest,
                                      "Invalid pair-summary length/count values")
                        try:
                            connection.execute("INSERT INTO seen VALUES (?,?,?)", (config, a, b))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("Duplicate or reversed pair within chromosome/configuration") from exc
                        connection.execute("INSERT INTO totals VALUES (?,?,?,?,?,?,?) ON CONFLICT(config,a,b) DO UPDATE SET n=n+excluded.n,bp=bp+excluded.bp,v=v+excluded.v,longest=MAX(longest,excluded.longest)",
                                           (config, a, b, *values))
                        counters[config]["n_pairs"] += 1
                        for key, value in zip(SUM_FIELDS, values):
                            counters[config][key] += value
                        if scanned % 50000 == 0:
                            connection.commit()
                sweep.require(hashed.digest.hexdigest() == entry["sha256"]["pair_summary"],
                              "Pair-summary compressed-byte hash mismatch")
            for config, counts in counters.items():
                sweep.require(all(value == int(wanted[config][key]) for key, value in counts.items()),
                              "Per-chromosome pair-summary aggregate parity failed")
            connection.commit()
            audit.append(dict(chromosome=entry["chromosome"], rows_scanned=scanned,
                              sha256=entry["sha256"], source_totals=counters))
        pair_output = output / "pair_configuration_summary.tsv.gz"
        with pair_output.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped, io.TextIOWrapper(zipped, newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(FIELDS)
            writer.writerows(connection.execute("SELECT config,a,b,n,bp,v,longest FROM totals ORDER BY config,a,b"))
        config_output = output / "configuration_summary.tsv"
        with config_output.open("x", newline="") as handle:
            fields = ("config_id", "max_gap_bp", "min_length_bp", "min_shared_effective", "n_pairs", *SUM_FIELDS)
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for config in sorted(sources):
                n_pairs, n, bp, variants = connection.execute("SELECT COUNT(*),COALESCE(SUM(n),0),COALESCE(SUM(bp),0),COALESCE(SUM(v),0) FROM totals WHERE config=?", (config,)).fetchone()
                plan = sources[config]
                writer.writerow(dict(config_id=config, max_gap_bp=plan["gap_bp"], min_length_bp=plan["length_bp"],
                                     min_shared_effective=plan["min_shared_effective"], n_pairs=n_pairs,
                                     n_segments=n, total_shared_bp=bp, n_shared_variants_total=variants))
    finally:
        connection.close()
    result = dict(status="COMPLETE_AUTOSOMAL_AGGREGATION", schema_version=1, chromosomes=AUTOSOMES,
                  sample_ids_sha256=sample_hash, n_samples=len(samples), source_rare_contract=source_contract,
                  manifest_sha256=sweep.sha256(manifest_path), settings_sha256=sweep.sha256(settings_path),
                  sources=audit, selected_source_configurations=sorted(sources),
                  outputs_sha256={p.name: sweep.sha256(p) for p in (pair_output, config_output)},
                  aggregation="sum n_segments, summed chain bp and shared-record counts; maximum segment length across chromosomes",
                  interval_union_computed=False, haplotype_copies_used=False,
                  interpretation="Descriptive unphased rare-sharing chains; not IBD or validated populations",
                  contains_individual_identifiers=True, public_distribution_allowed=False)
    sweep.write_json(output / "aggregation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["aggregate"])
    for name in ("manifest", "settings", "sample-ids", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.manifest, args.settings, args.sample_ids, args.output)
    print(json.dumps({key: result[key] for key in ("status", "chromosomes", "n_samples")}))


if __name__ == "__main__":
    main()
