#!/usr/bin/env python3
"""Stream pre-frequency VCFs into one shared outer-FIT genotype bundle for K03.

This module applies only source-level PASS, original-biallelic and A/C/G/T SNV
rules.  It never counts alleles or selects rare sites.  Each K03 context later
computes MAC, MAF and minor-allele orientation from its own INNER_FIT people.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pysam

import r02_k03_inner_fit_contract as contract


CAMPAIGN_SCHEMA = "r02_k03_prefrequency_campaign_v1"
REPORT_SCHEMA = "r02_k03_prefrequency_chromosome_v1"
TARGET_EXCLUSION = {"chromosome": 1, "start_bp": 1, "end_bp": 115064116}
CONSUMER_CODE = ("r02_genomic_pair_evidence.py", "r02_k03_inner_fit_contract.py",
    "r02_k03_inner_fit_affinities.py", "r02_projection_calibration_split.py",
    "r02_structure_affinity_io.py", "r02_structure_continuous.py",
    "r02_structure_discrete.py", "r02_structure_discrete_reuse.py",
    "r02_structure_increment.py", "r02_structure_input_plan.py",
    "r02_structure_materialize.py", "r02_structure_rare_inputs.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def consumer_implementation_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in CONSUMER_CODE}


def header_value(header: Any, key: str) -> list[str]:
    return [str(row.value) for row in header.records if row.key == key]


def encode_gt(gt: Any) -> int:
    """Encode a diploid GT while retaining its partial called-allele count."""
    require(isinstance(gt, tuple) and len(gt) == 2,
            "Every autosomal GT must be present and exactly diploid")
    require(all(a in (None, 0, 1) for a in gt),
            "A biallelic GT contains an allele other than REF/ALT/missing")
    called = [a for a in gt if a is not None]
    if not called:
        return 255
    if len(called) == 1:
        return 3 + called[0]
    return called[0] + called[1]


def _descriptor(value: Any, base: Path, label: str) -> Path:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            f"Exact {label} descriptor required")
    path = Path(value["path"])
    if not path.is_absolute():
        require(".." not in path.parts, f"Unsafe {label} path")
        path = base / path
    require(path.is_file() and sha256(path) == value["sha256"],
            f"Pinned {label} is absent or changed")
    return path


def _campaign(path: Path) -> tuple[dict[str, Any], Path, list[str]]:
    value = json.loads(path.read_text())
    required = {"schema", "roster", "sources", "output_chunk_sites",
                "max_input_bytes", "max_output_bytes", "target_exclusion", "consumer_resources"}
    require(set(value) == required and value["schema"] == CAMPAIGN_SCHEMA,
            "Exact pre-frequency campaign required")
    require(value["target_exclusion"] == TARGET_EXCLUSION
            and type(value["output_chunk_sites"]) is int and value["output_chunk_sites"] > 0
            and type(value["max_input_bytes"]) is int and value["max_input_bytes"] > 0
            and type(value["max_output_bytes"]) is int and value["max_output_bytes"] > 0,
            "Exact target exclusion and positive byte/chunk bounds required")
    base = path.resolve().parent
    roster = _descriptor(value["roster"], base, "roster")
    with roster.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(rows and set(rows[0]) >= {"sample_id", "role"}, "Roster fields are incomplete")
    ids = [row["sample_id"] for row in rows if row["role"] == "FIT"]
    require(ids and len(ids) == len(set(ids)), "Outer-FIT IDs must be nonempty and unique")
    require(isinstance(value["sources"], list) and len(value["sources"]) == 22
            and sorted(row.get("chromosome") for row in value["sources"]) == list(range(1, 23)),
            "Exactly one pinned source for every autosome is required")
    return value, roster, ids


def materialize_chromosome(campaign_path: Path, chromosome: int, output: Path,
                           source_override: Path | None = None) -> dict[str, Any]:
    campaign_path, output = Path(campaign_path), Path(output)
    campaign, _, fit_ids = _campaign(campaign_path)
    require(1 <= chromosome <= 22 and not output.exists() and output.parent.is_dir(),
            "Autosome and new output directory required")
    source_row = next(row for row in campaign["sources"] if row["chromosome"] == chromosome)
    require(set(source_row) == {"chromosome", "vcf"}, "Exact chromosome source descriptor required")
    source_path = (_descriptor(source_row["vcf"], campaign_path.resolve().parent, "VCF")
                   if source_override is None else Path(source_override))
    require(source_path.is_file() and sha256(source_path) == source_row["vcf"]["sha256"],
            "Staged VCF differs from pinned campaign source")
    require(source_path.stat().st_size <= campaign["max_input_bytes"], "VCF exceeds input byte bound")
    output.mkdir(mode=0o700)
    chunks_dir = output / "chunks"; chunks_dir.mkdir()
    counts: Counter[str] = Counter(); chunks = []
    columns: dict[str, list[Any]] = {name: [] for name in
        ("variant_id", "chromosome", "position_bp", "ref", "alt", "genotype_code")}

    def flush() -> None:
        if not columns["variant_id"]:
            return
        number = len(chunks); target = chunks_dir / f"{number:06d}.npz"
        code = np.stack(columns["genotype_code"], axis=1).astype(np.uint8, copy=False)
        np.savez_compressed(target, sample_ids=np.asarray(fit_ids),
            variant_id=np.asarray(columns["variant_id"]),
            chromosome=np.asarray(columns["chromosome"], np.int16),
            position_bp=np.asarray(columns["position_bp"], np.int64),
            ref=np.asarray(columns["ref"]), alt=np.asarray(columns["alt"]), genotype_code=code)
        chunks.append({"path": str(target.relative_to(output)), "sha256": sha256(target),
                       "sites": len(columns["variant_id"]), "bytes": target.stat().st_size})
        require(sum(row["bytes"] for row in chunks) <= campaign["max_output_bytes"],
                "Chromosome chunks exceed output byte bound")
        for values in columns.values(): values.clear()

    with pysam.VariantFile(str(source_path)) as source:
        require(header_value(source.header, "dnabr_original_alleles") == ["v1"]
                and "ORIG_NALLELES" in source.header.info and "ORIG_SITE" in source.header.info,
                "VCF lacks certified original-site provenance")
        require(not header_value(source.header, "dnabr_rare_contract")
                and not any(name in source.header.info for name in ("RARE_ALLELE", "RARE_AC", "RARE_AN", "RARE_AF"))
                and "RD" not in source.header.formats,
                "VCF is already frequency-selected or oriented")
        require(not (set(fit_ids) - set(source.header.samples)),
                "At least one outer-FIT person is absent from the VCF")
        source.subset_samples(fit_ids)
        require(len(source.header.samples) == len(fit_ids)
                and set(source.header.samples) == set(fit_ids),
                "VCF did not retain exactly the requested outer-FIT people")
        previous_source = 0
        previous_retained = 0
        for record in source:
            counts["input_records"] += 1
            require(record.contig.removeprefix("chr") == str(chromosome),
                    "VCF record differs from declared chromosome")
            require(record.pos >= previous_source,
                    "VCF positions must be nondecreasing before source eligibility filters")
            previous_source = record.pos
            require("ORIG_NALLELES" in record.info and "ORIG_SITE" in record.info,
                    "Record lacks original-site provenance")
            original_n = record.info["ORIG_NALLELES"]
            require(isinstance(original_n, int) and original_n >= 2, "Invalid ORIG_NALLELES")
            if list(record.filter.keys()) != ["PASS"]:
                counts["excluded_not_pass"] += 1; continue
            if original_n != 2:
                counts["excluded_original_multiallelic"] += 1; continue
            if len(record.alleles or ()) != 2 or any(len(a) != 1 or a not in "ACGT" for a in record.alleles):
                counts["excluded_not_biallelic_acgt_snv"] += 1; continue
            if chromosome == 1 and TARGET_EXCLUSION["start_bp"] <= record.pos <= TARGET_EXCLUSION["end_bp"]:
                counts["excluded_target_interval"] += 1; continue
            require(record.pos > previous_retained,
                    "Retained originally-biallelic PASS SNV coordinates must be unique")
            previous_retained = record.pos
            ref, alt = record.alleles
            columns["variant_id"].append(f"{chromosome}:{record.pos}:{ref}:{alt}")
            columns["chromosome"].append(chromosome); columns["position_bp"].append(record.pos)
            columns["ref"].append(ref); columns["alt"].append(alt)
            columns["genotype_code"].append(np.fromiter(
                (encode_gt(record.samples[name].get("GT")) for name in fit_ids), np.uint8, len(fit_ids)))
            counts["retained_pre_frequency_sites"] += 1
            if len(columns["variant_id"]) == campaign["output_chunk_sites"]: flush()
    flush()
    require(chunks, "No pre-frequency site remained after strict source filters")
    report = {"schema": REPORT_SCHEMA, "chromosome": chromosome,
              "campaign_sha256": sha256(campaign_path), "source_vcf_sha256": sha256(source_path),
              "sample_ids_sha256": contract.ordered_ids_sha256(fit_ids),
              "genotype_encoding": contract.GT_CODE_ENCODING,
              "frequency_selection_performed": False, "counts": dict(counts), "chunks": chunks,
              "pysam_version": pysam.__version__}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def assemble(campaign_path: Path, chromosome_dirs: list[Path], output: Path) -> dict[str, Any]:
    campaign_path, output = Path(campaign_path), Path(output)
    campaign, roster, fit_ids = _campaign(campaign_path)
    require(not output.exists() and output.parent.is_dir() and len(chromosome_dirs) == 22,
            "A new output and exactly 22 chromosome products are required")
    reports = []
    for directory in chromosome_dirs:
        report_path = Path(directory) / "report.json"; report = json.loads(report_path.read_text())
        require(report.get("schema") == REPORT_SCHEMA
                and report.get("campaign_sha256") == sha256(campaign_path)
                and report.get("sample_ids_sha256") == contract.ordered_ids_sha256(fit_ids)
                and report.get("frequency_selection_performed") is False,
                "Chromosome product provenance differs")
        reports.append((Path(directory), report))
    require(sorted(r[1]["chromosome"] for r in reports) == list(range(1, 23)),
            "Every autosome must occur exactly once")
    output.mkdir(mode=0o700); (output / "chunks").mkdir()
    shutil.copyfile(roster, output / "persons.tsv")
    descriptors = []
    for directory, report in sorted(reports, key=lambda row: row[1]["chromosome"]):
        chrom_dir = output / "chunks" / f"chr{report['chromosome']:02d}"; chrom_dir.mkdir()
        for row in report["chunks"]:
            source = directory / row["path"]; require(sha256(source) == row["sha256"], "Chunk changed")
            target = chrom_dir / Path(row["path"]).name
            try: os.link(source, target)
            except OSError: shutil.copyfile(source, target)
            descriptors.append({"path": str(target.relative_to(output)), "sha256": sha256(target)})
    manifest = {"schema": contract.SHARED_SCHEMA,
        "roster": {"path": "persons.tsv", "sha256": sha256(output / "persons.tsv")},
        "sample_scope": "OUTER_FIT_GENOTYPES_ONLY", "sample_ids_sha256": contract.ordered_ids_sha256(fit_ids),
        "dosage_allele": "ALT", "genotype_encoding": contract.GT_CODE_ENCODING,
        "variant_universe": "PINNED_ORIGINALLY_BIALLELIC_PASS_TARGET_EXCLUDED_NO_ROLE_SELECTION",
        "candidate_ascertainment": contract.STRICT_INNER_FIT_ASCERTAINMENT,
        "territory": {"view_id": "BLOCK_AND_LD_MARGIN:chr01_part01",
            "excluded_intervals": [TARGET_EXCLUSION], "target_locus_policy": "EXCLUDED_BEFORE_OUTPUT",
            "source_candidate_rule": "ORIGINALLY_BIALLELIC_PASS_BEFORE_MAC_MAF"},
        "learning_scope": "RAW_GENOTYPES_NO_LEARNED_TRANSFORMATIONS", "outer_fit_learned_inputs": False,
        "chunks": descriptors, "resources": campaign["consumer_resources"],
        "producer": {"implementation_sha256": sha256(Path(__file__)),
            "source_campaign_sha256": sha256(campaign_path), "source_frequency_use": "NONE_BEFORE_INNER_FIT",
            "original_allele_filter": "DNABR_V1_ORIG_NALLELES_EQ_2", "site_filter": "PASS_BIALLELIC_ACGT_SNV",
            "all_autosomes_complete": True},
        "consumer_implementation_sha256": consumer_implementation_sha256()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    chrom = sub.add_parser("chromosome"); chrom.add_argument("--campaign", type=Path, required=True)
    chrom.add_argument("--chromosome", type=int, required=True); chrom.add_argument("--output", type=Path, required=True)
    chrom.add_argument("--source-vcf", type=Path)
    join = sub.add_parser("assemble"); join.add_argument("--campaign", type=Path, required=True)
    join.add_argument("--chromosome-dir", type=Path, action="append", required=True)
    join.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    if args.command == "chromosome":
        result = materialize_chromosome(args.campaign, args.chromosome, args.output, args.source_vcf)
        print(json.dumps({"chromosome": result["chromosome"], "counts": result["counts"]}, sort_keys=True))
    else:
        result = assemble(args.campaign, args.chromosome_dir, args.output)
        print(json.dumps({"chunks": len(result["chunks"]), "manifest": sha256(args.output / "manifest.json")}, sort_keys=True))


if __name__ == "__main__": main()
