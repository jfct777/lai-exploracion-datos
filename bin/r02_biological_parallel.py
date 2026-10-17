#!/usr/bin/env python3
"""Parallel orchestration of the unchanged descriptive M14.2 reduction.

Scientific validation runs once per chromosome. A seal checks global chain IDs;
independent G reducers read the original streams and reuse finish_prepared.
No segment copies, new biological formulas, per-chromosome T, or input edits.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import csv
import heapq
import math
from pathlib import Path
import resource
import shutil
import time

import r02_biological_checkpoint as checkpoint

evaluation = checkpoint.evaluation
require = evaluation.require
sha256 = evaluation.sha256
read_json = checkpoint.read_json
write_json = evaluation.diagnostics.sweep.write_json
VALIDATION_SCHEMA = "r02_biological_parallel_validation_v1"
SEAL_SCHEMA = "r02_biological_parallel_seal_v1"
VALIDATED = "COMPLETE_CHROMOSOME_STREAM_VALIDATION"
SEALED = "COMPLETE_GLOBAL_INPUT_VALIDATION"
TABLES = ("configuration_metrics.tsv", "configuration_kinship.tsv",
          "configuration_chromosomes.tsv", "configuration_components.tsv",
          "configuration_metadata_coverage.tsv", "configuration_metadata_categories.private.tsv")
RESULT_FILES = (*TABLES, "configuration_cards.json")


def implementation():
    return checkpoint.code_hashes() | {Path(__file__).name: sha256(__file__)}


def descriptor(path, expected=None):
    path = Path(path).resolve()
    before = evaluation.diagnostics.kinship.stat_signature(path.stat())
    digest = sha256(path)
    require(expected is None or digest == expected, "Input SHA256 mismatch: " + path.name)
    require(evaluation.diagnostics.kinship.stat_signature(path.stat()) == before,
            "Input changed during authentication")
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def verify_descriptor(item):
    require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"}, "Invalid input descriptor")
    require(type(item["bytes"]) is int and item["bytes"] >= 0, "Invalid input size")
    evaluation.diagnostics.validate_hash(item["sha256"], "input SHA256")
    require(descriptor(item["path"], item["sha256"]) == item, "Input descriptor size/path mismatch")
    return evaluation.diagnostics.kinship.stat_signature(Path(item["path"]).stat())


def watch(items):
    result = {}
    for item in items:
        old = result.get(item["path"])
        require(old is None or old[0] == item, "Conflicting descriptors for one input")
        if old is None:
            result[item["path"]] = (item, verify_descriptor(item))
    return result


def unchanged(watched, hashes=False):
    for path, (item, signature) in watched.items():
        require(evaluation.diagnostics.kinship.stat_signature(Path(path).stat()) == signature,
                "Input changed during parallel stage")
        if hashes:
            verify_descriptor(item)


def resource_guard(output, min_free_disk_mb, watched=()):
    require(type(min_free_disk_mb) in (int, float) and math.isfinite(min_free_disk_mb)
            and min_free_disk_mb >= 0, "Invalid free-space reserve")
    output = Path(output)
    probe = output if output.exists() else output.parent
    require(probe.exists(), "Output parent must exist")
    def check():
        require(shutil.disk_usage(probe).free >= min_free_disk_mb*1024**2,
                "Insufficient free-space reserve")
        if watched:
            unchanged(watched)
    check()
    return check


def new_output(output):
    output = Path(output)
    require(not output.exists(), "Output already exists; no overwrite (use completed Nextflow cache)")
    output.mkdir(mode=0o700)
    return output


def authenticated_record(path, digest, schema, status):
    source = descriptor(path, digest)
    record = read_json(path)
    require(record.get("schema") == schema and record.get("status") == status,
            "Incomplete or unsupported parallel receipt")
    require(record.get("parallel_code_sha256") == implementation(), "Parallel implementation changed")
    require(descriptor(path, digest) == source, "Receipt changed during read")
    return record, source


def validate(checkpoint_manifest, expected_checkpoint_sha256, sample_ids,
             expected_samples, expected_configurations, expected_chromosome,
             output_dir, min_free_disk_mb):
    require(not Path(output_dir).exists(), "Output already exists; no overwrite")
    start, code = time.monotonic(), implementation()
    check = resource_guard(output_dir, min_free_disk_mb)
    samples = evaluation.diagnostics.sweep.samples_from(sample_ids, expected_samples)
    roster = descriptor(sample_ids)
    source = descriptor(checkpoint_manifest, expected_checkpoint_sha256)
    watched = watch([roster, source])
    # This ORIGINAL method includes ORIGINAL validate_streams exactly once.
    record, paths = checkpoint.authenticated_checkpoint(checkpoint_manifest,
        expected_checkpoint_sha256, roster["sha256"], len(samples), expected_configurations)
    require(record["chrom"] == str(expected_chromosome), "Validation chromosome mismatch")
    streams = {role: descriptor(path, record["outputs"][path.name]["sha256"])
               for role, path in paths.items()}
    watched.update(watch(streams.values()))
    check(); unchanged(watched, hashes=True)
    require(implementation() == code, "Code changed during validation")
    result = dict(schema=VALIDATION_SCHEMA, status=VALIDATED, chrom=record["chrom"],
        source_checkpoint=source, sample_ids=roster, checkpoint_record=record,
        streams=streams, parallel_code_sha256=code,
        scientific_validation="original_authenticated_checkpoint_including_validate_streams",
        segment_copies_created=0, resources=dict(elapsed_seconds=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    output = new_output(output_dir)
    write_json(output/"validation.json", result)
    return result


def parse_settings(contract):
    require(contract.get("schema") == "r02_biological_checkpoint_aggregate_v1", "Unknown aggregate contract")
    thresholds = tuple(map(float, contract["kinship_thresholds"].split(",")))
    edges = tuple(map(int, contract["edge_thresholds_bp"].split(",")))
    require(thresholds and len(set(thresholds)) == len(thresholds)
            and all(math.isfinite(v) and 0 <= v <= .5 for v in thresholds), "Invalid kinship thresholds")
    require(edges and 0 in edges and len(set(edges)) == len(edges) and all(v >= 0 for v in edges),
            "Invalid global edge thresholds")
    require(type(contract["expected_samples"]) is int and contract["expected_samples"] >= 2
            and type(contract["expected_configurations"]) is int and contract["expected_configurations"] >= 1,
            "Invalid sample/configuration counts")
    return thresholds, edges


def merge_registry(records):
    registry, territory = checkpoint.empty_registry(), checkpoint.empty_territory()
    usage = dict(peak_observed_sqlite_bytes=0, peak_observed_scratch_bytes=0)
    maximum = 0
    for record in records:
        current = record["registry"]
        for key in ("identity", "configs"):
            require(registry[key] is None or registry[key] == current[key], "Incompatible checkpoint " + key)
            registry[key] = current[key]
        for key, value in current["optional_criteria"].items():
            require(key not in registry["optional_criteria"] or registry["optional_criteria"][key] == value,
                    "Incompatible optional evidence criteria")
            registry["optional_criteria"][key] = value
        for key in ("chromosomes", "ibd_chromosomes", "sources"):
            registry[key].extend(current[key])
        for key in territory:
            value = record["ibd_territory"][key]
            require(type(value) is int and value >= 0, "Invalid checkpoint IBD denominator")
            territory[key] += value
        for key in usage:
            usage[key] = max(usage[key], record["resources"][key])
        maximum = max(maximum, record["resources"]["max_database_mb"])
    require(len(set(registry["chromosomes"])) == len(records), "Duplicate checkpoint chromosome")
    return registry, territory, usage, maximum


def seal(validation_manifests, expected_validation_sha256s, aggregate_contract,
         expected_aggregate_contract_sha256, sample_ids, pcrelate_file,
         chromosomes, output_dir, min_free_disk_mb):
    require(not Path(output_dir).exists(), "Output already exists; no overwrite")
    start, code = time.monotonic(), implementation()
    wanted = list(map(str, chromosomes))
    require(wanted and len(set(wanted)) == len(wanted)
            and all(c in evaluation.diagnostics.AUTOSOMES for c in wanted), "Invalid chromosome scope")
    require(len(validation_manifests) == len(expected_validation_sha256s) == len(wanted),
            "Missing chromosome validation receipts")
    contract_source = descriptor(aggregate_contract, expected_aggregate_contract_sha256)
    contract = read_json(aggregate_contract)
    thresholds, edges = parse_settings(contract)
    samples = evaluation.diagnostics.sweep.samples_from(sample_ids, contract["expected_samples"])
    roster = descriptor(sample_ids, contract["sample_ids_sha256"])
    kinship = descriptor(pcrelate_file, contract["kinship_sha256"])
    expected_checkpoints = {r["chrom"]: r for r in contract["checkpoints"]}
    require(len(expected_checkpoints) == len(contract["checkpoints"])
            and set(expected_checkpoints) == set(wanted), "Contract chromosomes missing or duplicated")
    validations, records, inputs, identifiers = [], [], [contract_source, roster, kinship], []
    for path, digest in zip(validation_manifests, expected_validation_sha256s):
        receipt, item = authenticated_record(path, digest, VALIDATION_SCHEMA, VALIDATED)
        chrom = receipt["chrom"]
        require(chrom in wanted, "Unexpected validated chromosome")
        expected = expected_checkpoints[chrom]
        require(receipt["source_checkpoint"]["path"] == str(Path(expected["path"]).resolve())
                and receipt["source_checkpoint"]["sha256"] == expected["sha256"], "Checkpoint not bound to aggregate contract")
        require(receipt["sample_ids"]["sha256"] == roster["sha256"], "Validation cohort/order mismatch")
        record = receipt["checkpoint_record"]
        require(record == read_json(expected["path"]), "Validation checkpoint record mismatch")
        require(record["n_configurations"] == contract["expected_configurations"]
                and record["n_samples"] == len(samples), "Validation dimensions mismatch")
        require(set(receipt["streams"]) == {"segments", "identifiers"}, "Incomplete validation streams")
        for role, filename in record["streams"].items():
            expected_stream = dict(path=str((Path(expected["path"]).parent/filename).resolve()),
                **record["outputs"][filename])
            require(receipt["streams"][role] == expected_stream,
                    "Validation stream not bound to checkpoint")
        validations.append(dict(receipt=item, chrom=chrom, streams=receipt["streams"],
                                source_checkpoint=receipt["source_checkpoint"]))
        records.append(record)
        inputs.extend([item, receipt["source_checkpoint"], *receipt["streams"].values()])
        identifiers.append(receipt["streams"]["identifiers"]["path"])
    registry, territory, usage, maximum = merge_registry(records)
    require(set(registry["chromosomes"]) == set(wanted), "Incomplete chromosome coverage")
    identity = registry["identity"]
    require(identity["genome_build"] == contract["genome_build"], "Genome build mismatch")
    rare = identity["source_rare_contract"]
    require(str(rare["dnabr_rare_cohort_n_samples"]) == str(contract["expected_source_samples"])
            and rare["dnabr_rare_cohort_sha256"] == contract["source_cohort_sha256"], "Source cohort mismatch")
    study_path = contract["study"]["path"]
    study_source = descriptor(study_path, contract["study"]["sha256"])
    study_people = descriptor(Path(study_path).parent/"persons.private.tsv", contract["study"]["persons_sha256"])
    inputs.extend([study_source, study_people])
    watched = watch(inputs)
    check = resource_guard(output_dir, min_free_disk_mb, watched)
    evaluation.read_study(study_path, samples, contract["kinship_sha256"], thresholds)
    # Original global duplicate-ID check, executed once and authenticated below.
    checkpoint.evaluation.unique_chain_ids(identifiers)
    groups = {}
    for name, config in registry["configs"].items():
        groups.setdefault(str(config["max_gap_bp"]), []).append(name)
    groups = {g: sorted(names) for g, names in groups.items()}
    require(sum(map(len, groups.values())) == contract["expected_configurations"], "Incomplete configuration partition")
    check(); unchanged(watched, hashes=True)
    require(implementation() == code, "Code changed during seal")
    result = dict(schema=SEAL_SCHEMA, status=SEALED, parallel_code_sha256=code,
        chromosomes=sorted(wanted, key=int), registry=registry, ibd_territory=territory,
        inherited_usage=usage, inherited_max_database_mb=maximum, groups=groups,
        global_ids_checked=True, global_ids_method="original_unique_chain_ids",
        validations=validations, immutable_inputs=inputs, aggregate_contract=contract_source,
        sample_ids=roster, pcrelate_file=kinship, study_contract=study_source,
        expected_samples=len(samples), expected_configurations=contract["expected_configurations"],
        thresholds=list(thresholds), edge_thresholds_bp=list(edges),
        resources=dict(elapsed_seconds=time.monotonic()-start, segment_copies_created=0))
    output = new_output(output_dir)
    write_json(output/"seal.json", result)
    return result


def open_seal(path, digest):
    record, item = authenticated_record(path, digest, SEAL_SCHEMA, SEALED)
    require(record.get("global_ids_checked") is True and record.get("global_ids_method") == "original_unique_chain_ids",
            "Global chain IDs were not checked")
    chromosomes = record["chromosomes"]
    validated = [r["chrom"] for r in record["validations"]]
    require(chromosomes and len(set(chromosomes)) == len(chromosomes)
            and set(chromosomes).issubset(evaluation.diagnostics.AUTOSOMES)
            and len(validated) == len(set(validated)) == len(chromosomes)
            and set(validated) == set(chromosomes)
            and set(record["registry"]["chromosomes"]) == set(chromosomes),
            "Sealed chromosome coverage is incomplete or duplicated")
    configs = record["registry"]["configs"]
    ids = [name for names in record["groups"].values() for name in names]
    require(len(ids) == len(set(ids)) == record["expected_configurations"] and set(ids) == set(configs),
            "Configuration groups overlap or are incomplete")
    for gap, names in record["groups"].items():
        require(names and all(str(configs[name]["max_gap_bp"]) == gap for name in names), "Incorrect G group")
    return record, item


def reduce(seal_manifest, expected_seal_sha256, gap_bp, output_dir, min_free_disk_mb):
    require(not Path(output_dir).exists(), "Output already exists; no overwrite")
    start, code = time.monotonic(), implementation()
    record, item = open_seal(seal_manifest, expected_seal_sha256)
    require(str(gap_bp) in record["groups"], "Unknown G reducer")
    watched = watch([item, *record["immutable_inputs"]])
    check = resource_guard(output_dir, min_free_disk_mb, watched)
    samples = evaluation.diagnostics.sweep.samples_from(record["sample_ids"]["path"], record["expected_samples"])
    study = evaluation.read_study(record["study_contract"]["path"], samples,
                                 record["pcrelate_file"]["sha256"], record["thresholds"])
    registry = copy.deepcopy(record["registry"])
    names = record["groups"][str(gap_bp)]
    registry["configs"] = {name: registry["configs"][name] for name in names}
    paths = [r["streams"]["segments"]["path"] for r in record["validations"]]
    # The original reducer selects configurations by G; all 22 chromosome
    # streams remain present. Empty identifiers deliberately means that the
    # once-only ORIGINAL global check is supplied by the authenticated seal.
    result = evaluation.finish_prepared(paths, [], registry, record["ibd_territory"], samples,
        record["sample_ids"]["path"], record["pcrelate_file"]["path"], record["pcrelate_file"]["sha256"],
        output_dir, study, record["thresholds"], record["edge_thresholds_bp"],
        record["inherited_usage"], record["inherited_max_database_mb"], min_free_disk_mb,
        check, write_manifest=False)
    unchanged(watched, hashes=True)
    require(implementation() == code, "Code changed during reducer")
    result.update(parallel_code_sha256=code, parallel_stage="G_REDUCER", gap_bp=gap_bp,
        configuration_ids=names, source_seal=item, global_ids_reused=True)
    result["resources"].update(parallel_elapsed_seconds=time.monotonic()-start,
        parallel_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        restart="Completed chromosome validations and G reducers are independent Nextflow cache units")
    write_json(Path(output_dir)/"manifest.json", result)
    return result


def merged_table(paths, output, allowed):
    """Merge already ordered configuration/T rows, preserving within-group order."""
    with ExitStack() as stack:
        readers = [csv.DictReader(stack.enter_context(Path(p).open(newline="")), delimiter="\t") for p in paths]
        fields = readers[0].fieldnames
        require(fields and all(r.fieldnames == fields for r in readers), "Reducer table schema mismatch")
        def checked(reader, names):
            previous = None
            for row in reader:
                require(None not in row and all(v is not None for v in row.values()), "Malformed reducer table")
                require(row["config_id"] in names, "Reducer table contains another G/configuration")
                key = row["config_id"], int(row["min_edge_bp"])
                require(previous is None or key >= previous, "Unsorted reducer table")
                previous = key
                yield row
        streams = [checked(reader, names) for reader, names in zip(readers, allowed)]
        rows = heapq.merge(*streams, key=lambda r: (r["config_id"], int(r["min_edge_bp"])))
        evaluation.write_table(output, rows, fields)


def collect(seal_manifest, expected_seal_sha256, reducer_manifests,
            expected_reducer_sha256s, output_dir, min_free_disk_mb):
    require(not Path(output_dir).exists(), "Output already exists; no overwrite")
    start, code = time.monotonic(), implementation()
    seal_record, seal_source = open_seal(seal_manifest, expected_seal_sha256)
    require(len(reducer_manifests) == len(expected_reducer_sha256s) == len(seal_record["groups"]),
            "Missing G reducer manifests")
    inputs = [seal_source, *seal_record["immutable_inputs"]]
    records, directories, receipts, seen, allowed = [], [], [], set(), []
    for path, digest in zip(reducer_manifests, expected_reducer_sha256s):
        record, receipt = authenticated_record(path, digest, evaluation.SCHEMA, evaluation.STATUS)
        gap = str(record.get("gap_bp"))
        require(record.get("parallel_stage") == "G_REDUCER" and gap in seal_record["groups"] and gap not in seen,
                "Duplicate, unexpected or incomplete G reducer")
        require(record.get("source_seal") == seal_source and record.get("global_ids_reused") is True,
                "Reducer is not bound to this global seal")
        names = seal_record["groups"][gap]
        require(record.get("configuration_ids") == names and record["n_configurations"] == len(names),
                "Reducer configuration overlap or coverage mismatch")
        require(record["chromosomes"] == seal_record["chromosomes"] and
                record["sample_ids_sha256"] == seal_record["sample_ids"]["sha256"] and
                record["n_samples"] == seal_record["expected_samples"], "Reducer cohort/chromosome mismatch")
        require(record["parameters"]["kinship_thresholds"] == seal_record["thresholds"] and
                record["parameters"]["edge_thresholds_bp"] == sorted(seal_record["edge_thresholds_bp"]),
                "Reducer parameters differ")
        require(set(record["outputs_sha256"]) == set(RESULT_FILES), "Incomplete reducer scientific outputs")
        for filename, sha in record["outputs_sha256"].items():
            inputs.append(descriptor(Path(path).parent/filename, sha))
        inputs.append(receipt); seen.add(gap); records.append(record)
        directories.append(Path(path).parent); receipts.append(receipt); allowed.append(set(names))
    require(seen == set(seal_record["groups"]), "Incomplete G scope")
    for field in ("pcrelate_audit", "study_contract", "optional_evidence_criteria", "input_manifests",
                  "ibd_territory_chromosomes", "genome_build", "source_rare_contract"):
        require(all(r[field] == records[0][field] for r in records), "Incompatible reducer metadata: " + field)
    watched = watch(inputs)
    check = resource_guard(output_dir, min_free_disk_mb, watched)
    cards = []
    for directory, names in zip(directories, allowed):
        payload = read_json(directory/"configuration_cards.json")
        require(payload["schema"] == evaluation.SCHEMA and payload["status"] == evaluation.STATUS,
                "Incomplete cards schema/status")
        keys = [(c["config_id"], c["min_edge_bp"]) for c in payload["cards"]]
        expected = {(name, edge) for name in names for edge in seal_record["edge_thresholds_bp"]}
        require(len(keys) == len(set(keys)) and set(keys) == expected, "Duplicate/missing configuration cards")
        cards.extend(payload["cards"])
    cards.sort(key=lambda c: (c["config_id"], c["min_edge_bp"]))
    output = new_output(output_dir)
    for filename in TABLES:
        merged_table([d/filename for d in directories], output/filename, allowed)
        check()
    write_json(output/"configuration_cards.json", dict(schema=evaluation.SCHEMA, status=evaluation.STATUS, cards=cards))
    # Retain the original scientific manifest/semantics; only the aggregate
    # dimensions and explicit orchestration provenance/resources change.
    result = copy.deepcopy(records[0])
    for key in ("gap_bp", "configuration_ids", "global_ids_reused"):
        result.pop(key)
    result.update(parallel_stage="COLLECTED_ALL_G", n_configurations=seal_record["expected_configurations"],
        n_cards=len(cards), n_metrics=sum(r["n_metrics"] for r in records), reducer_inputs=receipts,
        checkpoint_inputs=[dict(chrom=r["chrom"], path=r["source_checkpoint"]["path"],
                                sha256=r["source_checkpoint"]["sha256"])
                           for r in seal_record["validations"]],
        checkpoint_code_sha256=checkpoint.code_hashes(),
        outputs_sha256={name: sha256(output/name) for name in RESULT_FILES})
    result["resources"].update(collect_elapsed_seconds=time.monotonic()-start,
        strategy="one original scientific validation per chromosome; one original global ID check; independent original reducers by G, all chromosomes before T; original streams reread without copies",
        reducer_resources={str(r["gap_bp"]): r["resources"] for r in records})
    unchanged(watched, hashes=True)
    require(implementation() == code, "Code changed during collection")
    write_json(output/"manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    parsers = {name: commands.add_parser(name) for name in ("validate", "seal", "reduce", "collect")}
    for command in parsers.values():
        command.add_argument("--output-dir", required=True)
        command.add_argument("--min-free-disk-mb", type=int, required=True)
    for name in ("validate", "seal"):
        parsers[name].add_argument("--sample-ids", required=True)
    p = parsers["validate"]
    for name in ("checkpoint-manifest", "expected-checkpoint-sha256", "expected-chromosome"):
        p.add_argument("--"+name, required=True)
    for name in ("expected-samples", "expected-configurations"):
        p.add_argument("--"+name, type=int, required=True)
    p = parsers["seal"]
    p.add_argument("--validation-manifest", dest="validation_manifests", action="append", required=True)
    p.add_argument("--expected-validation-sha256", dest="expected_validation_sha256s", action="append", required=True)
    for name in ("aggregate-contract", "expected-aggregate-contract-sha256", "pcrelate-file", "chromosomes"):
        p.add_argument("--"+name, required=True)
    for name in ("reduce", "collect"):
        parsers[name].add_argument("--seal-manifest", required=True)
        parsers[name].add_argument("--expected-seal-sha256", required=True)
    parsers["reduce"].add_argument("--gap-bp", type=int, required=True)
    parsers["collect"].add_argument("--reducer-manifest", dest="reducer_manifests", action="append", required=True)
    parsers["collect"].add_argument("--expected-reducer-sha256", dest="expected_reducer_sha256s", action="append", required=True)
    options = vars(parser.parse_args(argv)); mode = options.pop("mode")
    if mode == "seal":
        options["chromosomes"] = options["chromosomes"].split(",")
    result = dict(validate=validate, seal=seal, reduce=reduce, collect=collect)[mode](**options)
    print(__import__("json").dumps({"schema": result["schema"], "status": result["status"]}))


if __name__ == "__main__":
    main()
