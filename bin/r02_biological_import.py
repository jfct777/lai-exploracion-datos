#!/usr/bin/env python3
"""Metadata preflight for import-only M14.2 integration; no analysis or selection.

The import contract pins *expected* manifest hashes and an explicit chromosome
scope. The existing evaluator remains responsible for authenticated table bytes,
segment geometry, evidence masks and aggregation. This avoids a second algorithm
or an extra full read of the large compressed interval/link tables.

Contract schema r02_biological_import_v1 contains chromosomes, expected_samples,
expected_source_samples, expected_configurations, genome_build, sample_ids_sha256,
source_cohort_sha256, kinship_sha256, study={path,sha256}, and
segments=[{chrom,path,sha256}, ...]. Paths are explicit local absolute paths to
study_contract.json / manifest.json with their original adjacent tables. Staged
paths may differ; their authenticated contents and chromosome identities may not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import r02_biological_evaluation as evaluation

SCHEMA = "r02_biological_import_v1"
require = evaluation.require
sha256 = evaluation.sha256
validate_hash = evaluation.diagnostics.validate_hash


def unique_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: "+key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def chromosomes(values):
    require(isinstance(values, list) and bool(values), "Explicit chromosome scope is required")
    require(all(isinstance(v, str) and v in evaluation.diagnostics.AUTOSOMES for v in values),
            "Invalid chromosome in import scope")
    require(len(set(values)) == len(values), "Duplicated chromosome in import scope")
    return sorted(values, key=int)


def manifest_entry(entry, basename):
    require(isinstance(entry, dict), "Malformed manifest entry")
    path = entry.get("path")
    require(isinstance(path, str) and Path(path).is_absolute() and ".." not in Path(path).parts and Path(path).name == basename and
            re.search(r"[\n\r'\"`$\\;|&<>*?\[\]{}]", path) is None,
            "Manifest path must be explicit, safe, absolute and have its required filename")
    return validate_hash(entry.get("sha256"), basename)


def validate(import_contract, expected_import_sha256, segment_manifests, study_contract,
             sample_ids, expected_samples, expected_source_samples, expected_configurations,
             pcrelate_file, kinship_sha256, genome_build, scope, thresholds, edge_thresholds_bp):
    """Fail closed on provenance before the evaluator creates a SQLite database."""
    path = Path(import_contract)
    expected = validate_hash(expected_import_sha256, "import contract")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Import contract SHA256 mismatch")
    contract = unique_json(raw)
    require(isinstance(contract, dict) and contract.get("schema") == SCHEMA,
            "Unsupported import contract schema")
    wanted = chromosomes(scope)
    require(chromosomes(contract.get("chromosomes")) == wanted, "Import chromosome scope mismatch")
    values = dict(expected_samples=expected_samples, expected_source_samples=expected_source_samples,
                  expected_configurations=expected_configurations, genome_build=genome_build)
    for key, value in values.items():
        if key != "genome_build":
            require(type(value) is int and value > 0 and type(contract.get(key)) is int,
                    "Invalid declared count: "+key)
        require(contract.get(key) == value, "Import contract/parameter mismatch: "+key)
    require(isinstance(genome_build, str) and bool(genome_build), "Genome build missing")
    sample_ids = Path(sample_ids)
    samples = evaluation.diagnostics.sweep.samples_from(sample_ids, expected_samples)
    sample_digest = sha256(sample_ids)
    require(sample_digest == validate_hash(contract.get("sample_ids_sha256"), "analytical cohort"),
            "Import analytical cohort/order SHA256 mismatch")
    source_digest = validate_hash(contract.get("source_cohort_sha256"), "source cohort")
    kin_digest = validate_hash(kinship_sha256, "PC-Relate")
    require(validate_hash(contract.get("kinship_sha256"), "contract PC-Relate") == kin_digest,
            "Import PC-Relate expectation mismatch")
    require(sha256(Path(pcrelate_file)) == kin_digest, "PC-Relate SHA256 mismatch")
    require(thresholds and len(set(thresholds)) == len(thresholds) and
            all(math.isfinite(t) and 0 <= t <= .5 for t in thresholds), "Invalid kinship thresholds")
    require(edge_thresholds_bp and 0 in edge_thresholds_bp and len(set(edge_thresholds_bp)) == len(edge_thresholds_bp)
            and all(type(t) is int and t >= 0 for t in edge_thresholds_bp), "Invalid accumulated-bp thresholds; baseline zero required")
    edges = edge_thresholds_bp
    study_digest = manifest_entry(contract.get("study"), "study_contract.json")
    require(sha256(Path(study_contract)) == study_digest, "Study contract SHA256 mismatch")
    study = evaluation.read_study(Path(study_contract), samples, kin_digest, thresholds)

    entries = contract.get("segments")
    require(isinstance(entries, list) and all(isinstance(v, dict) for v in entries),
            "Segment manifest entries missing")
    require(chromosomes([v.get("chrom") for v in entries]) == wanted,
            "Segment manifest scope mismatch")
    expected_segments = {v["chrom"]: manifest_entry(v, "manifest.json") for v in entries}
    require(len({v["path"] for v in entries}) == len(entries), "Duplicate source manifest path")
    require(len(segment_manifests) == len(wanted), "Staged manifest count mismatch")
    seen, configs, identities, sources, inventory = set(), None, None, [], []
    for manifest_path in segment_manifests:
        manifest_path = Path(manifest_path)
        raw = manifest_path.read_bytes()
        record = unique_json(raw)
        require(isinstance(record, dict) and record.get("schema") == "r02_segment_evidence_v1" and
                record.get("status") == "COMPLETE_DESCRIPTIVE_NOT_VALIDATED",
                "Incomplete/unsupported M14.2 manifest")
        chrom = str(record.get("chrom", "")).removeprefix("chr")
        require(chrom in expected_segments and chrom not in seen, "Unexpected/duplicate staged chromosome")
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == expected_segments[chrom], "M14.2 manifest SHA256 mismatch: chr"+chrom)
        require(record.get("sample_ids_sha256") == sample_digest and record.get("n_samples") == expected_samples,
                "M14.2 analytical cohort/order mismatch")
        require(record.get("genome_build") == genome_build and record.get("coordinate_system") == "1-based-inclusive",
                "M14.2 build/coordinate mismatch")
        rare = record.get("source_rare_contract", {})
        require(isinstance(rare, dict) and rare.get("dnabr_original_alleles") == "v1" and
                rare.get("dnabr_rare_contract") == "minor_v1", "Unsupported source minor allele contract")
        require(str(rare.get("dnabr_rare_cohort_n_samples")) == str(expected_source_samples) and
                rare.get("dnabr_rare_cohort_sha256") == source_digest, "M14.2 source cohort mismatch")
        criteria = record.get("evidence_criteria", {})
        require(isinstance(criteria, dict) and all(criteria.get(k) is not None for k in evaluation.CORE_CRITERIA),
                "M14.2 evidence criteria incomplete")
        identity = (rare, {key: criteria[key] for key in evaluation.CORE_CRITERIA})
        require(identities is None or identities == identity, "Incompatible M14.2 core evidence criteria")
        identities = identity
        outputs = record.get("outputs_sha256", {})
        require(isinstance(outputs, dict), "M14.2 output digest ledger missing")
        names = (*evaluation.FILES, *((evaluation.IBD_TERRITORY,) if evaluation.IBD_TERRITORY in outputs else ()))
        for name in names:
            validate_hash(outputs.get(name), name)
            require((manifest_path.parent/name).is_file(), "Missing staged M14.2 table: "+name)
            inventory.append(dict(chrom=chrom, name=name, bytes=(manifest_path.parent/name).stat().st_size,
                                  expected_sha256=outputs[name], byte_hash_check="DELEGATED_TO_EVALUATOR"))
        # Small ledger only; full compressed table SHA/content checks occur once
        # in the shared evaluator, including every declared empty configuration.
        ledger_path = manifest_path.parent/evaluation.FILES[2]
        require(sha256(ledger_path) == outputs[evaluation.FILES[2]], "Configuration ledger SHA256 mismatch")
        ledger = evaluation.configuration_ledger(ledger_path, expected_configurations)
        require(configs is None or configs == ledger, "Configuration ledger differs across chromosomes")
        configs = ledger
        seen.add(chrom)
        sources.append(dict(chrom=chrom, manifest_sha256=digest,
                            original_path=next(v["path"] for v in entries if v["chrom"] == chrom)))
    require(sorted(seen, key=int) == wanted, "Incomplete staged chromosome scope")
    return dict(schema="r02_biological_import_receipt_v1", status="PREFLIGHT_METADATA_VERIFIED",
                import_contract_sha256=expected, chromosomes=wanted, **values,
                sample_ids_sha256=sample_digest, source_cohort_sha256=source_digest,
                kinship_sha256=kin_digest, study_contract_sha256=study_digest,
                study_provenance=study["provenance"], segment_manifests=sorted(sources, key=lambda v: int(v["chrom"])),
                thresholds=thresholds, edge_thresholds_bp=edges,
                input_inventory=inventory, segment_table_bytes=sum(v["bytes"] for v in inventory),
                storage_limit="Input sizes are observed bytes, not a bound on SQLite/sort scratch or whole-genome scalability",
                source_sha256={Path(__file__).name: sha256(Path(__file__)),
                               Path(evaluation.__file__).name: sha256(Path(evaluation.__file__))},
                table_validation="Delegated to r02_biological_evaluation; receipt alone is not completed evaluation",
                interpretation="Technical provenance only; no biological validation, selection or training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("import-contract", "study-contract", "sample-ids", "pcrelate-file", "receipt"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--segment-manifest", action="append", type=Path, required=True)
    for name in ("expected-import-sha256", "kinship-sha256", "genome-build", "chromosomes", "thresholds", "edge-thresholds-bp"):
        parser.add_argument("--"+name, required=True)
    for name in ("expected-samples", "expected-source-samples", "expected-configurations"):
        parser.add_argument("--"+name, type=int, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), "Import receipt already exists")
    record = validate(args.import_contract, args.expected_import_sha256, args.segment_manifest,
        args.study_contract, args.sample_ids, args.expected_samples, args.expected_source_samples,
        args.expected_configurations, args.pcrelate_file, args.kinship_sha256, args.genome_build,
        args.chromosomes.split(","), tuple(float(v) for v in args.thresholds.split(",")),
        tuple(int(v) for v in args.edge_thresholds_bp.split(",")))
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
