#!/usr/bin/env python3
"""Private technical checkpoints for the existing descriptive M14.2 evaluator.

No annotation, cohort selection or per-chromosome accumulated-bp filtering.
Completed checkpoints are immutable inputs, never biological result bundles.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import resource
import shutil
import tempfile
import time

import r02_biological_evaluation as evaluation

SCHEMA = "r02_biological_chromosome_checkpoint_v1"
STATUS = "COMPLETE_TECHNICAL_CHECKPOINT_NOT_BIOLOGICAL_RESULT"
require = evaluation.require
sha256 = evaluation.sha256
SORT_FIELDS = ("a", "b", "chrom", "start_pos", "end_pos", "max_gap_bp")


def code_hashes():
    modules = (evaluation, evaluation.diagnostics, evaluation.diagnostics.sweep,
               evaluation.diagnostics.kinship, evaluation.diagnostics.kinship.saved)
    return {Path(m.__file__).name: sha256(m.__file__) for m in modules} | {
        Path(__file__).name: sha256(__file__)}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    with Path(path).open(encoding="utf-8") as handle:
        def reject_constant(token):
            raise ValueError("Nonfinite JSON value")
        return json.load(handle, object_pairs_hook=unique, parse_constant=reject_constant)


def empty_registry():
    return dict(identity=None, configs=None, optional_criteria={}, chromosomes=[],
                ibd_chromosomes=[], sources=[])


def empty_territory():
    return dict(n_evaluable_pair_chromosomes=0, callable_bp=0, ibd_union_bp=0)


def prepare(manifest_path, expected_manifest_sha256, sample_ids, expected_samples,
            expected_configurations, output_dir, scratch_dir=None,
            max_database_mb=8192, min_free_disk_mb=1024, expected_chromosome=None):
    output = Path(output_dir)
    require(not output.exists(), "Checkpoint output already exists; no overwrite")
    require(math.isfinite(max_database_mb) and max_database_mb > 0 and
            math.isfinite(min_free_disk_mb) and min_free_disk_mb >= 0, "Invalid resource limits")
    require(sha256(manifest_path) == expected_manifest_sha256, "Source manifest SHA256 mismatch")
    if expected_chromosome is not None:
        require(read_json(manifest_path).get("chrom") == str(expected_chromosome),
                "Prepare chromosome/manifest mismatch")
    samples = evaluation.diagnostics.sweep.samples_from(sample_ids, expected_samples)
    require(len(samples) >= 2, "At least two samples required")
    sample_sha, implementation = sha256(sample_ids), code_hashes()
    registry, territory = empty_registry(), empty_territory()
    usage = dict(peak_observed_sqlite_bytes=0, peak_observed_scratch_bytes=0,
                 peak_rss_bytes=0, minimum_observed_free_bytes=None)
    started = time.monotonic()
    output.mkdir(parents=True)
    # On failure partial streams have no completed manifest and cannot be imported.
    with tempfile.TemporaryDirectory(prefix="r02-chromosome-", dir=scratch_dir) as temp:
        db_path = Path(temp)/"evidence.sqlite"
        def check():
            size = evaluation.resources(db_path, max_database_mb, min_free_disk_mb)
            free = min(shutil.disk_usage(output).free, shutil.disk_usage(temp).free)
            require(free >= min_free_disk_mb*1024**2, "Insufficient output free-space reserve")
            scratch = sum(p.stat().st_size for p in Path(temp).iterdir() if p.is_file())
            scratch += sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            usage["peak_observed_sqlite_bytes"] = max(usage["peak_observed_sqlite_bytes"], size)
            usage["peak_observed_scratch_bytes"] = max(usage["peak_observed_scratch_bytes"], scratch)
            usage["peak_rss_bytes"] = max(usage["peak_rss_bytes"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            prior = usage["minimum_observed_free_bytes"]
            usage["minimum_observed_free_bytes"] = free if prior is None else min(prior, free)
        segments, identifiers = evaluation.prepare_chromosome(db_path, output,
            manifest_path, sample_ids, {s: i for i, s in enumerate(samples)},
            expected_configurations, registry, territory, max_database_mb,
            min_free_disk_mb, check)
        check()
    require(sha256(sample_ids) == sample_sha, "Sample file changed during preparation")
    require(sha256(manifest_path) == expected_manifest_sha256, "Source manifest changed during preparation")
    require(code_hashes() == implementation, "Code changed during preparation")
    result = dict(schema=SCHEMA, status=STATUS, chrom=registry["chromosomes"][0],
        n_samples=len(samples), sample_ids_sha256=sample_sha,
        n_configurations=expected_configurations, registry=registry, ibd_territory=territory,
        streams=dict(segments=segments.name, identifiers=identifiers.name),
        sort_fields=list(SORT_FIELDS), code_sha256=implementation,
        outputs={p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in (segments, identifiers)},
        resources={**usage, "elapsed_seconds": time.monotonic()-started,
            "max_database_mb": max_database_mb, "min_free_disk_mb": min_free_disk_mb,
            "mb_unit": "MiB", "rss_scope": "Linux process high-water RSS; includes earlier work in same process",
            "scratch_scope": "SQLite directory plus output streams; transient SQLite sort files may require executor-level accounting"},
        contains_private_pair_indices=True, biological_result=False,
        no_local_accumulated_bp_threshold=True)
    # The marker is written last. Nextflow only caches a successfully ended task.
    evaluation.diagnostics.sweep.write_json(output/"checkpoint.json", result)
    return result


def authenticated_checkpoint(path, expected_sha, sample_sha, n_samples, n_configs):
    path = Path(path)
    require(sha256(path) == expected_sha, "Checkpoint SHA256 mismatch")
    record = read_json(path)
    require(record.get("schema") == SCHEMA and record.get("status") == STATUS,
            "Incomplete or unsupported checkpoint")
    chrom = record.get("chrom")
    require(chrom in evaluation.diagnostics.AUTOSOMES, "Invalid checkpoint chromosome")
    require(record.get("n_samples") == n_samples and record.get("sample_ids_sha256") == sample_sha,
            "Checkpoint sample cohort/order mismatch")
    require(record.get("n_configurations") == n_configs and record.get("code_sha256") == code_hashes(),
            "Checkpoint configuration count or source code mismatch")
    require(record.get("sort_fields") == list(SORT_FIELDS), "Checkpoint sort contract mismatch")
    require(record.get("streams") == {"segments": f"chr{chrom}.segments.jsonl.gz",
            "identifiers": f"chr{chrom}.ids.gz"}, "Checkpoint stream names mismatch")
    names = set(record["streams"].values())
    require(set(record.get("outputs", {})) == names, "Incomplete checkpoint output inventory")
    paths, signatures = {}, {}
    for role, name in record["streams"].items():
        stream = path.parent/name
        signatures[stream] = evaluation.diagnostics.kinship.stat_signature(stream.stat())
        metadata = record["outputs"][name]
        require(stream.is_file() and stream.stat().st_size == metadata["bytes"] and
                sha256(stream) == metadata["sha256"], "Checkpoint stream SHA256/size mismatch")
        paths[role] = stream
    registry = record["registry"]
    require(registry["chromosomes"] == [chrom] and len(registry["sources"]) == 1 and
            registry["sources"][0]["chrom"] == chrom and len(registry["configs"]) == n_configs,
            "Checkpoint registry mismatch")
    require(registry["ibd_chromosomes"] in ([], [chrom]), "Invalid IBD chromosome registry")
    text_configs = ({key: str(value) for key, value in row.items()} for row in registry["configs"].values())
    require(evaluation.configuration_records(text_configs, n_configs) == registry["configs"],
            "Checkpoint configuration ledger mismatch")
    validate_streams(paths, chrom, n_samples, registry["sources"][0]["n_chains"], registry)
    require(all(evaluation.diagnostics.kinship.stat_signature(p.stat()) == s for p, s in signatures.items()),
            "Checkpoint stream changed during verification")
    require(sha256(path) == expected_sha, "Checkpoint changed during verification")
    return record, paths


def validate_streams(paths, chrom, n_samples, expected_count, registry):
    # These are now durable inputs, so do not rely on heapq.merge to detect
    # malformed rows or incorrectly ordered streams.
    db = evaluation.setup_database(":memory:")
    try:
        expected_fields = [r[1] for r in db.execute("PRAGMA table_info(chains)") if r[1] != "chain_id"]
    finally:
        db.close()
    count, previous, previous_pair = 0, None, None
    ends = {}
    sample_indices = {i: i for i in range(n_samples)}
    by_gap = {}
    for name, config in registry["configs"].items():
        by_gap.setdefault(config["max_gap_bp"], []).append((name, config))
    with gzip.open(paths["segments"], "rt", encoding="utf-8") as handle:
        require(json.loads(next(handle, "null")) == expected_fields, "Checkpoint segment schema mismatch")
        for line in handle:
            values = json.loads(line)
            require(isinstance(values, list) and len(values) == len(expected_fields), "Checkpoint row width mismatch")
            row = dict(zip(expected_fields, values))
            require(all(type(row[k]) is int for k in SORT_FIELDS), "Invalid checkpoint sort field")
            require(row["chrom"] == int(chrom) and 0 <= row["a"] < row["b"] < n_samples,
                    "Checkpoint chromosome/person indices mismatch")
            key = tuple(row[k] for k in SORT_FIELDS)
            require(previous is None or key > previous, "Checkpoint segment stream is unsorted or duplicated")
            # Reuse the original scientific validator instead of implementing a
            # second set of numerator/denominator/missingness rules. This dummy
            # ID is only a parsing placeholder; real IDs are checked separately.
            raw = {k: "NA" if v is None else str(v) for k, v in row.items()}
            raw.update(chain_id="chain_"+"0"*64, sample_a=row["a"], sample_b=row["b"],
                J="NA" if not row["U"] else str(row["I"]/row["U"]),
                ibd_fraction="NA" if not row["callable_bp"] or row["ibd_union_bp"] is None
                    else str(row["ibd_union_bp"]/row["callable_bp"]))
            parsed = evaluation.parse_segment(raw, chrom, sample_indices)
            parsed.pop("chain_id")
            require(parsed == row, "Checkpoint segment values differ from canonical parser")
            if row["common_status"] == "OK":
                require(registry["optional_criteria"].get("common_criteria") is not None,
                        "Checkpoint commons lack criteria")
            if row["ibd_status"] == "OK":
                require(chrom in registry["ibd_chromosomes"], "Checkpoint IBD lacks authenticated territory")
            pair = row["a"], row["b"]
            if pair != previous_pair:
                ends.clear()
                previous_pair = pair
            for name, config in by_gap.get(row["max_gap_bp"], ()):
                if row["length_bp"] >= config["min_length_bp"] and row["n_shared_variants"] >= config["min_shared_effective"]:
                    require(name not in ends or row["start_pos"] > ends[name],
                            "Overlapping checkpoint intervals within configuration/pair")
                    ends[name] = row["end_pos"]
            previous, count = key, count+1
    require(count == expected_count, "Checkpoint segment count mismatch")
    count, previous = 0, None
    with gzip.open(paths["identifiers"], "rt", encoding="utf-8") as handle:
        for line in handle:
            require(line.endswith("\n") and line.strip() and
                    (previous is None or line > previous), "Checkpoint ID stream is unsorted or duplicated")
            require(line.startswith("chain_"), "Invalid checkpoint chain ID prefix")
            evaluation.diagnostics.validate_hash(line.strip()[6:], "checkpoint chain ID")
            previous, count = line, count+1
    require(count == expected_count, "Checkpoint identifier count mismatch")


def aggregate(checkpoint_manifests, expected_checkpoint_sha256s, chromosomes,
              sample_ids, expected_samples, pcrelate_file, expected_pcrelate_sha256,
              output_dir, expected_configurations=74, study_contract=None,
              thresholds=evaluation.diagnostics.DEFAULT_THRESHOLDS,
              edge_thresholds_bp=evaluation.DEFAULT_EDGES, scratch_dir=None,
              min_free_disk_mb=1024, expected_genome_build=None,
              expected_source_samples=None, expected_source_cohort_sha256=None):
    require(not Path(output_dir).exists(), "Output already exists; no overwrite")
    wanted = list(map(str, chromosomes))
    require(wanted and len(set(wanted)) == len(wanted) and
            all(c in evaluation.diagnostics.AUTOSOMES for c in wanted), "Invalid chromosome scope")
    require(len(checkpoint_manifests) == len(expected_checkpoint_sha256s) == len(wanted),
            "Missing checkpoint manifests/hashes")
    require(thresholds and len(set(thresholds)) == len(thresholds) and
            all(math.isfinite(t) and 0 <= t <= .5 for t in thresholds), "Invalid kinship thresholds")
    require(edge_thresholds_bp and 0 in edge_thresholds_bp and
            len(set(edge_thresholds_bp)) == len(edge_thresholds_bp) and
            all(type(t) is int and t >= 0 for t in edge_thresholds_bp), "Invalid accumulated-bp thresholds")
    require(math.isfinite(min_free_disk_mb) and min_free_disk_mb >= 0, "Invalid free-space reserve")
    samples = evaluation.diagnostics.sweep.samples_from(sample_ids, expected_samples)
    sample_sha = sha256(sample_ids)
    registry, territory = empty_registry(), empty_territory()
    segments, identifiers, receipts = [], [], []
    usage = dict(peak_observed_sqlite_bytes=0, peak_observed_scratch_bytes=0)
    maximum_database_mb = 0
    signatures = {}
    started = time.monotonic()
    implementation = code_hashes()
    for path, digest in zip(checkpoint_manifests, expected_checkpoint_sha256s):
        record, paths = authenticated_checkpoint(path, digest, sample_sha, len(samples), expected_configurations)
        current, chrom = record["registry"], record["chrom"]
        require(chrom in wanted and chrom not in registry["chromosomes"], "Duplicate or unexpected checkpoint chromosome")
        for field in ("identity", "configs"):
            require(registry[field] is None or registry[field] == current[field], "Incompatible checkpoint "+field)
            registry[field] = current[field]
        for key, value in current["optional_criteria"].items():
            require(key not in registry["optional_criteria"] or registry["optional_criteria"][key] == value,
                    "Incompatible optional evidence criteria")
            registry["optional_criteria"][key] = value
        for field in ("chromosomes", "ibd_chromosomes", "sources"):
            registry[field].extend(current[field])
        for field in territory:
            value = record["ibd_territory"][field]
            require(type(value) is int and value >= 0, "Invalid checkpoint IBD denominator")
            territory[field] += value
        for field in usage:
            usage[field] = max(usage[field], record["resources"][field])
        maximum_database_mb = max(maximum_database_mb, record["resources"]["max_database_mb"])
        segments.append(paths["segments"])
        identifiers.append(paths["identifiers"])
        receipts.append(dict(chrom=chrom, path=str(Path(path).resolve()), sha256=digest))
        for target in (Path(path), *paths.values()):
            signatures[target] = evaluation.diagnostics.kinship.stat_signature(target.stat())
    require(set(registry["chromosomes"]) == set(wanted), "Incomplete chromosome scope")
    identity = registry["identity"]
    rare = identity["source_rare_contract"]
    if expected_genome_build is not None:
        require(identity["genome_build"] == expected_genome_build, "Expected genome build mismatch")
    if expected_source_samples is not None:
        require(str(rare["dnabr_rare_cohort_n_samples"]) == str(expected_source_samples), "Expected source cohort size mismatch")
    if expected_source_cohort_sha256 is not None:
        require(rare["dnabr_rare_cohort_sha256"] == expected_source_cohort_sha256, "Expected source cohort hash mismatch")
    require(sha256(sample_ids) == sample_sha, "Sample file changed during checkpoint verification")
    study = evaluation.read_study(study_contract, samples, expected_pcrelate_sha256, thresholds)
    scratch = Path(scratch_dir or Path(output_dir).parent)
    def check():
        require(shutil.disk_usage(scratch).free >= min_free_disk_mb*1024**2, "Insufficient free-space reserve")
        for path, signature in signatures.items():
            require(evaluation.diagnostics.kinship.stat_signature(path.stat()) == signature,
                    "Checkpoint input changed during aggregation")
    check()
    result = evaluation.finish_prepared(segments, identifiers, registry, territory, samples,
        sample_ids, pcrelate_file, expected_pcrelate_sha256, output_dir, study, thresholds,
        edge_thresholds_bp, usage, maximum_database_mb, min_free_disk_mb, check,
        write_manifest=False)
    result["checkpoint_inputs"] = receipts
    require(code_hashes() == implementation, "Code changed during aggregation")
    result["checkpoint_code_sha256"] = implementation
    result["resources"].update(restart="Per-chromosome completed checkpoints are reusable; global merge has no internal resume",
        aggregate_elapsed_seconds=time.monotonic()-started,
        aggregate_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    evaluation.diagnostics.sweep.write_json(Path(output_dir)/"manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    aggregate_parser = sub.add_parser("aggregate")
    for command in (prepare_parser, aggregate_parser):
        command.add_argument("--sample-ids", required=True)
        command.add_argument("--expected-samples", type=int, required=True)
        command.add_argument("--expected-configurations", type=int, required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--scratch-dir")
        command.add_argument("--min-free-disk-mb", type=int, required=True)
    prepare_parser.add_argument("--manifest-path", required=True)
    prepare_parser.add_argument("--expected-manifest-sha256", required=True)
    prepare_parser.add_argument("--expected-chromosome", required=True)
    prepare_parser.add_argument("--max-database-mb", type=int, required=True)
    aggregate_parser.add_argument("--checkpoint-manifest", dest="checkpoint_manifests", action="append", required=True)
    aggregate_parser.add_argument("--expected-checkpoint-sha256", dest="expected_checkpoint_sha256s", action="append", required=True)
    aggregate_parser.add_argument("--chromosomes", required=True)
    aggregate_parser.add_argument("--pcrelate-file", required=True)
    aggregate_parser.add_argument("--expected-pcrelate-sha256", required=True)
    aggregate_parser.add_argument("--study-contract", required=True)
    aggregate_parser.add_argument("--thresholds", required=True)
    aggregate_parser.add_argument("--edge-thresholds-bp", required=True)
    aggregate_parser.add_argument("--expected-genome-build", required=True)
    aggregate_parser.add_argument("--expected-source-samples", type=int, required=True)
    aggregate_parser.add_argument("--expected-source-cohort-sha256", required=True)
    options = vars(parser.parse_args(argv))
    mode = options.pop("mode")
    if mode == "aggregate":
        options["chromosomes"] = options["chromosomes"].split(",")
        options["thresholds"] = tuple(map(float, options["thresholds"].split(",")))
        options["edge_thresholds_bp"] = tuple(map(int, options["edge_thresholds_bp"].split(",")))
    result = (prepare if mode == "prepare" else aggregate)(**options)
    print(json.dumps({"schema": result["schema"], "status": result["status"]}))


if __name__ == "__main__":
    main()
