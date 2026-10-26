#!/usr/bin/env python3
"""Prepare empirical FIT genotype counts; never train or open reserved roles."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from r02_event_mmap_reader import mapped_arrays, signature
from r02_genomic_event_inputs import ARRAY_FIELDS
from r02_lai_fit_information import counts
from r02_structure_fit_information import (require, select_fit, sha256, write_json)


SCHEMA = "r02_fit_genotype_reference_v1"
BASES = "ACGT"
FIELDS = ("chromosome", "position_bp", "position_cm", "ref", "alt",
          "counted_allele", "allele_count", "allele_number", "frequency",
          "fit_observed_people", "fit_n0", "fit_n1", "fit_n2")
def restored_store(index_path, chromosome, store):
    index_path, store = Path(index_path), Path(store).resolve()
    index = json.loads(index_path.read_text())
    require(index.get("schema") == "r02_structure_restored_store_index_v1" and
            index.get("status") == "COMPLETE_22_RESTORED_AUTHENTICATED_STORE_PATHS_NO_ASSEMBLY_NO_TRAINING" and
            index.get("chromosomes") == 22 and len(index.get("stores", [])) == 22 and
            index.get("training_started") is False and index.get("evaluate_opened") is False,
            "Restored-store index differs")
    matches = [x for x in index["stores"] if str(x["chromosome"]) == chromosome]
    require(len(matches) == 1, "Chromosome absent or duplicated in restored index")
    restored = matches[0]
    manifest = store / "manifest.json"
    require(manifest.is_file() and sha256(manifest) == restored["manifest_sha256"],
            "Store manifest differs from restored index")
    return index, restored, manifest
def fit_indices(store, manifest):
    people_path = store / "source/persons.tsv"
    require(sha256(people_path) == manifest["outputs_sha256"]["source/persons.tsv"],
            "Person metadata differs")
    with people_path.open(newline="") as stream:
        people = list(csv.DictReader(stream, delimiter="\t"))
    require(len(people) == 2619 and len({r["sample_id"] for r in people}) == 2619,
            "Person axis differs")
    observed = {role: sum(r["role"] == role for r in people)
                for role in ("FIT", "SELECT", "EVALUATE")}
    require(observed == {"FIT": 1460, "SELECT": 476, "EVALUATE": 683},
            "Role counts differ")
    return np.flatnonzero(np.asarray([r["role"] == "FIT" for r in people])), people_path
def validate_arrays(data, chunk, people=2619):
    require(set(data) == ARRAY_FIELDS, "NPZ member inventory differs")
    n = chunk["n_sites"]
    for name, value in data.items():
        require(value.shape == ((people, n) if name in {"dosage", "observed"} else (n,)),
                "Array shape differs: " + name)
    require(data["dosage"].dtype == np.uint8 and data["observed"].dtype == np.bool_,
            "Dose/mask type differs")
    require(data["position_bp"].dtype == np.int64 and
            np.all(np.diff(data["position_bp"]) > 0) and
            int(data["position_bp"][0]) == chunk["start_bp"] and
            int(data["position_bp"][-1]) == chunk["end_bp"], "Position identity differs")
    require(np.all(data["ref"] < 4) and np.all(data["alt"] < 4) and
            np.all(data["ref"] != data["alt"]) and np.all(data["counted_allele"] <= 1),
            "Allele identity differs")
    require(np.all(data["allele_number"] > 0) and np.all(data["allele_count"] >= 0) and
            np.all(data["allele_count"] <= data["allele_number"]) and
            np.allclose(data["frequency"], data["allele_count"] / data["allele_number"],
                        rtol=0, atol=1e-12), "Source frequency identity differs")
def output_rows(chromosome, channel, data, counted):
    n = len(data["position_bp"])
    require(all(len(np.asarray(value)) == n for value in counted.values()), "Count length differs")
    require(np.array_equal(counted["n0"] + counted["n1"] + counted["n2"], counted["n"]),
            "Genotype counts do not sum to observed people")
    if channel == "common":
        require(np.all(data["counted_allele"] == 1) and
                np.array_equal(data["allele_count"], counted["n1"] + 2*counted["n2"]) and
                np.array_equal(data["allele_number"], 2*counted["n"]),
                "COMMON ALT FIT AC/AN differ from FIT genotype counts")
    for i in range(n):
        yield {"chromosome": chromosome, "position_bp": int(data["position_bp"][i]),
            "position_cm": float(data["position_cm"][i]), "ref": BASES[int(data["ref"][i])],
            "alt": BASES[int(data["alt"][i])],
            "counted_allele": int(data["counted_allele"][i]),
            "allele_count": int(data["allele_count"][i]),
            "allele_number": int(data["allele_number"][i]),
            "frequency": float(data["frequency"][i]),
            "fit_observed_people": int(counted["n"][i]),
            "fit_n0": int(counted["n0"][i]), "fit_n1": int(counted["n1"][i]),
            "fit_n2": int(counted["n2"][i])}
def write_rows(path, rows):
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
def load_ledger(path):
    """Recover only an interrupted final append; reject earlier corruption."""
    path = Path(path)
    if not path.exists(): return []
    raw = path.read_bytes(); lines = raw.splitlines(keepends=True); records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            require(i == len(lines)-1 and not line.endswith(b"\n"),
                    "Checkpoint ledger has non-final corruption")
            evidence = path.with_name("checkpoints.trailing-partial.json")
            write_json(evidence, {"schema": SCHEMA + "_trailing_partial_v1",
                "ledger": str(path), "valid_records": len(records),
                "trailing_bytes_sha256": __import__("hashlib").sha256(line).hexdigest(),
                "trailing_bytes_hex": line.hex()})
            tmp = path.with_suffix(path.suffix + ".recovered")
            tmp.write_bytes(b"".join(lines[:i])); os.replace(tmp, path)
    if raw and not raw.endswith(b"\n") and len(records) == len(lines):
        tmp = path.with_suffix(path.suffix + ".recovered")
        tmp.write_bytes(raw + b"\n"); os.replace(tmp, path)
    return records
def dependency_hashes():
    root = Path(__file__).resolve().parent
    names = ("r02_event_mmap_reader.py", "r02_genomic_event_inputs.py",
             "r02_lai_fit_information.py", "r02_structure_fit_information.py")
    return {name: sha256(root/name) for name in names}


def process_chunk(store, chromosome, channel, chunk, fit, destination):
    source = store / chunk["path"]
    before = signature(source)
    require(before[2] == chunk["bytes"] and sha256(source) == chunk["sha256"],
            "Chunk size/hash differs")
    with mapped_arrays(source) as data:
        validate_arrays(data, chunk)
        dose, observed = select_fit(data["dosage"], data["observed"], fit)
        counted = counts(dose, observed)
        write_rows(destination, output_rows(chromosome, channel, data, counted))
    require(signature(source) == before, "Chunk changed during preparation")
    return {"source_signature": list(before), "rows_sha256": sha256(destination),
            "observed_genotypes": int(counted["n"].sum()),
            "missing_genotypes": int(len(fit) * chunk["n_sites"] - counted["n"].sum())}


def old_count_stream(summary_path, manifest_sha, loci):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    table = summary_path.parent / "variant-fit-information.tsv.gz"
    require(summary["identity"]["manifest_sha256"] == manifest_sha and
            summary["chromosome"] == "22" and summary["rare_loci"] == loci and
            summary["fit_people"] == 1460 and
            not summary["select_or_evaluate_genotypes_aggregated"] and
            sha256(table) == summary["variant_table_sha256"],
            "Reusable chr22 rare result differs")
    return summary, table, gzip.open(table, "rt")


def reuse_chunk(stream, store, chromosome, chunk, destination):
    source = store / chunk["path"]; before = signature(source)
    require(before[2] == chunk["bytes"] and sha256(source) == chunk["sha256"],
            "Chunk size/hash differs")
    old = [next(stream) for _ in range(chunk["n_sites"])]
    with mapped_arrays(source) as data:
        validate_arrays(data, chunk)
        old_count = {k: np.asarray([int(r[k]) for r in old])
                     for k in ("fit_observed_people", "fit_n0", "fit_n1", "fit_n2")}
        require(np.array_equal(data["position_bp"], [int(r["position_bp"]) for r in old]) and
                np.array_equal(data["counted_allele"], [int(r["counted_allele"]) for r in old]) and
                np.array_equal(data["allele_count"], [int(r["source_ac"]) for r in old]) and
                np.array_equal(data["allele_number"], [int(r["source_an"]) for r in old]) and
                np.allclose(data["frequency"], [float(r["source_frequency"]) for r in old],
                            rtol=0, atol=1e-12),
                "Reusable chr22 rare locus identity differs")
        counted = {"n": old_count["fit_observed_people"], "n0": old_count["fit_n0"],
                   "n1": old_count["fit_n1"], "n2": old_count["fit_n2"]}
        write_rows(destination, output_rows(chromosome, "rare", data, counted))
    require(signature(source) == before, "Chunk changed during metadata reuse")
    return {"source_signature": list(before), "rows_sha256": sha256(destination),
            "observed_genotypes": int(counted["n"].sum()),
            "missing_genotypes": int(1460 * chunk["n_sites"] - counted["n"].sum())}


def run(store, restored_index, chromosome, channel, output, *, reuse_summary=None,
        max_seconds=300, max_rss_mib=512, max_chunks=10000):
    started = time.monotonic(); chromosome = str(chromosome)
    require(chromosome in {str(i) for i in range(1, 23)} and channel in {"common", "rare"},
            "Invalid chromosome/channel")
    store, output = Path(store).resolve(), Path(output).resolve()
    index, _, manifest_path = restored_store(restored_index, chromosome, store)
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == "r02_genomic_event_chromosome_v1" and
            str(manifest["chromosome"]) == chromosome and manifest["n_people"] == 2619 and
            manifest["role_counts"] == {"EVALUATE": 683, "FIT": 1460, "SELECT": 476},
            "Chromosome manifest differs")
    chunks = manifest["channels"][channel]
    require(0 < len(chunks) <= max_chunks and all(manifest["outputs_sha256"].get(c["path"]) == c["sha256"] for c in chunks),
            "Channel inventory exceeds bound or differs")
    fit, people_path = fit_indices(store, manifest)
    reuse = chromosome == "22" and channel == "rare"
    require(not reuse or reuse_summary is not None, "chr22 rare authenticated reuse is required")
    output.mkdir(parents=True, exist_ok=True); checkpoint = output / "chunks"; checkpoint.mkdir(exist_ok=True)
    identity = {"schema": SCHEMA, "chromosome": chromosome, "channel": channel,
        "store_manifest_sha256": sha256(manifest_path), "restored_index_sha256": sha256(restored_index),
        "persons_sha256": sha256(people_path), "code_sha256": sha256(__file__),
        "dependency_sha256": dependency_hashes(),
        "role": "FIT", "fit_people": 1460,
        "allele_definition": "ALT" if channel == "common" else "source_minor_DNABR2723",
        "frequency_origin": "FIT1460" if channel == "common" else "DNABR2723_fixed_catalogue",
        "reuse_summary_sha256": sha256(reuse_summary) if reuse else None}
    identity_path = output / "identity.json"
    if identity_path.exists(): require(json.loads(identity_path.read_text()) == identity, "Resume identity differs")
    else: write_json(identity_path, identity)
    final = output / "summary.json"
    if final.exists():
        saved = json.loads(final.read_text()); require(saved["identity"] == identity and sha256(output/"fit-genotype-reference.tsv.gz") == saved["table_sha256"], "Final output differs")
        return {"status": "FINISHED_SKIP", "summary": str(final)}
    old_handle = None
    if reuse:
        _, _, old_handle = old_count_stream(reuse_summary, identity["store_manifest_sha256"], sum(c["n_sites"] for c in chunks))
        old_stream = csv.DictReader(old_handle, delimiter="\t")
    totals = {"observed_genotypes": 0, "missing_genotypes": 0}
    ledger = output/"checkpoints.jsonl"; completed = load_ledger(ledger)
    require(len(completed) <= len(chunks), "Checkpoint ledger exceeds channel")
    try:
        for i, chunk in enumerate(chunks):
            rows = checkpoint / f"{i:05d}.tsv.gz"
            expected = {"chunk_index": i, "source_path": chunk["path"],
                "source_sha256": chunk["sha256"], "source_bytes": chunk["bytes"],
                "n_sites": chunk["n_sites"], "start_bp": chunk["start_bp"],
                "end_bp": chunk["end_bp"], "reused_counts": reuse}
            if i < len(completed):
                saved = completed[i]
                require(all(saved.get(k) == v for k, v in expected.items()) and
                        list(signature(store/chunk["path"])) == saved["source_signature"] and
                        sha256(rows) == saved["rows_sha256"], "Checkpoint differs")
                if reuse:
                    for _ in range(chunk["n_sites"]): next(old_stream)
            else:
                if time.monotonic() - started >= max_seconds:
                    write_json(output/"progress.json", {"identity": identity, "status": "INCOMPLETE_TIME_LIMIT", "completed_chunks": i, "total_chunks": len(chunks)})
                    return {"status": "INCOMPLETE_TIME_LIMIT", "completed_chunks": i}
                saved = (reuse_chunk(old_stream, store, chromosome, chunk, rows) if reuse else
                         process_chunk(store, chromosome, channel, chunk, fit, rows))
                saved.update(expected)
                with ledger.open("a") as stream:
                    stream.write(json.dumps(saved, separators=(",", ":")) + "\n")
                    stream.flush(); os.fsync(stream.fileno())
                completed.append(saved)
            for key in totals: totals[key] += saved[key]
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= max_rss_mib * 1024,
                    "Memory budget exhausted")
        if reuse: require(next(old_stream, None) is None, "Reusable chr22 rare table has extra rows")
    finally:
        if old_handle is not None: old_handle.close()
    table = output / "fit-genotype-reference.tsv.gz"; tmp = output / "fit-genotype-reference.tmp.gz"
    last = None; rows_total = 0
    with gzip.open(tmp, "wt") as out:
        out.write("\t".join(FIELDS) + "\n")
        for i, chunk in enumerate(chunks):
            with gzip.open(checkpoint/f"{i:05d}.tsv.gz", "rt") as src:
                header = src.readline().rstrip("\n"); require(header == "\t".join(FIELDS), "Chunk header differs")
                for line in src:
                    pos = int(line.split("\t", 3)[1]); require(last is None or pos > last, "Loci not strictly ordered")
                    last = pos; rows_total += 1; out.write(line)
    os.replace(tmp, table)
    require(rows_total == sum(c["n_sites"] for c in chunks), "Aggregated locus count differs")
    result = {"identity": identity, "status": "FIT_EMPIRICAL_GENOTYPE_COUNTS_NO_HWE_NO_TRAINING",
        "loci": rows_total, "chunks": len(chunks), **totals, "table_sha256": sha256(table),
        "training_started": False, "select_or_evaluate_genotypes_aggregated": False,
        "elapsed_seconds": time.monotonic()-started}
    write_json(final, result); return {"status": result["status"], "summary": str(final)}


def result_exit_code(result):
    return 75 if result["status"] == "INCOMPLETE_TIME_LIMIT" else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True); p.add_argument("--restored-index", required=True)
    p.add_argument("--chromosome", required=True); p.add_argument("--channel", choices=("common", "rare"), required=True)
    p.add_argument("--output", required=True); p.add_argument("--reuse-summary")
    p.add_argument("--max-seconds", type=int, default=300); p.add_argument("--max-rss-mib", type=int, default=512)
    p.add_argument("--max-chunks", type=int, default=10000)
    a = p.parse_args(); result = run(a.store, a.restored_index, a.chromosome, a.channel,
        a.output, reuse_summary=a.reuse_summary, max_seconds=a.max_seconds,
        max_rss_mib=a.max_rss_mib, max_chunks=a.max_chunks)
    print(json.dumps(result)); return result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
