#!/usr/bin/env python3
"""L1 catalogue correspondence (default) or authenticated genotype audit.

The selected allele and source frequency remain those of M02.1. A missing
panel record is not a reference-homozygote call. Catalogue-only output describes
marker representation and never decodes GT. The explicit genotype_support mode
is implemented in r02_lai_genotype_support.py; neither mode certifies phasing,
donor independence, biological validation or simulation truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time
from types import SimpleNamespace

from r02_genomic_pair_evidence import (
    require, sample_hash, sha256, validate_rare_header, validate_rare_site,
)

SCHEMA = "r02_lai_catalogue_correspondence_v1"
STATUS = "COMPLETE_CATALOGUE_CORRESPONDENCE_ONLY_NOT_SUPPORT"
INPUTS = ("rare_vcf", "rare_contract", "panel_vcf", "roles", "build_evidence", "contig_map")
SUPPORT_FIELDS = (
    "n_role_members", "n_gt_complete", "n_quality_evaluable", "ac_counted_called",
    "an_called", "ac_counted_evaluable", "an_evaluable", "n_carriers_called",
    "n_carriers_evaluable", "n_unresolved", "n_phase_usable_carriers",
    "phase_contract_id", "quality_contract_id",
)
FIELDS = (
    "source_build", "chromosome", "position_bp", "REF", "ALT", "counted_allele",
    "source_rare_allele_index", "source_rare_ac", "source_rare_an",
    "source_cohort_n", "source_cohort_sha256", "source_catalogue_sha256",
    "original_allele_count", "source_site_id", "source_record_id", "panel_role",
    "panel_source_sha256", "panel_records_at_position", "correspondence_status",
    "build_status", "genotype_evidence_status", *SUPPORT_FIELDS, "support_status",
)


class MetadataHeader:
    """Header interface used by the shared M02.1 validators without GT parsing."""
    def __init__(self, lines, samples):
        self.lines, self.samples = lines, samples
        self.info, self.formats = {}, {}
        for line in lines:
            match = re.match(r"##(INFO|FORMAT)=<ID=([^,>]+)[,>]", line)
            if match:
                target = self.info if match[1] == "INFO" else self.formats
                require(match[2] not in target, "Duplicated VCF field declaration")
                target[match[2]] = True

    def __str__(self):
        return "\n".join(self.lines)


class MetadataVCF:
    """Read only the eight fixed columns, leaving sample bytes uninterpreted.

    gzip supports concatenated BGZF members. This sequential first diagnostic
    needs no tabix index and does not fetch or decode a genotype field. The
    bounded binary line reader also refuses unexpectedly large records.
    """
    def __init__(self, path, max_line_bytes):
        self.path, self.max_line_bytes = Path(path), max_line_bytes
        self.handle = None

    def __enter__(self):
        with self.path.open("rb") as handle:
            compressed = handle.read(2) == b"\x1f\x8b"
        self.handle = gzip.open(self.path, "rb") if compressed else self.path.open("rb")
        try:
            lines = []
            while True:
                line = self.readline()
                require(line, "VCF has no column header")
                if line.startswith(b"##"):
                    lines.append(line.decode("utf-8").rstrip("\r\n"))
                    continue
                fields = line.decode("utf-8").rstrip("\r\n").split("\t")
                require(fields[:8] == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"],
                        "Malformed VCF column header")
                require(len(fields) >= 10 and fields[8] == "FORMAT", "VCF must declare samples")
                samples = fields[9:]
                require(len(samples) == len(set(samples)) and all(samples), "Empty or duplicated VCF sample IDs")
                require(all(not any(c.isspace() for c in value) for value in samples), "Invalid VCF sample ID")
                self.header = MetadataHeader(lines, samples)
                return self
        except BaseException:
            self.handle.close()
            raise

    def readline(self):
        line = self.handle.readline(self.max_line_bytes + 1)
        require(len(line) <= self.max_line_bytes, "VCF record exceeds max-line-bytes")
        self.last_line_bytes = len(line)
        return line

    def __iter__(self):
        for line in iter(self.readline, b""):
            require(not line.startswith(b"#") and line.strip(), "Unexpected header or empty VCF row")
            fields = line.rstrip(b"\r\n").split(b"\t", 8)
            require(len(fields) == 9, "VCF row lacks fixed columns or FORMAT/sample tail")
            # The ninth element is deliberately neither decoded nor split.
            yield [field.decode("utf-8") for field in fields[:8]]

    def __exit__(self, *unused):
        self.handle.close()


def read_json(path):
    def unique(pairs):
        data = {}
        for key, value in pairs:
            require(key not in data, "Duplicate JSON key")
            data[key] = value
        return data
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def input_receipts(args, input_names=INPUTS):
    paths = {key: Path(getattr(args, key)) for key in input_names}
    require(all(path.is_file() for path in paths.values()), "A required input is not a file")
    receipts = {key: {"path": str(path.resolve()), "sha256": sha256(path),
                      "size_bytes": path.stat().st_size} for key, path in paths.items()}
    if args.input_hashes:
        expected = read_json(args.input_hashes)
        require(set(expected) == set(input_names), "input-hashes must name every input exactly once")
        for key in input_names:
            require(receipts[key]["sha256"] == expected[key], f"Input SHA256 mismatch: {key}")
    return receipts


def verify_inputs_unchanged(receipts):
    for key, entry in receipts.items():
        path = Path(entry["path"])
        require(path.stat().st_size == entry["size_bytes"] and sha256(path) == entry["sha256"],
                f"Input changed during execution: {key}")


def build_contract(path, receipts, proof_paths=None):
    data = read_json(path)
    require(data.get("schema") == "r02_lai_build_evidence_v1", "Unknown build evidence schema")
    verified_files = {}
    for name in ("source", "panel"):
        entry = data.get(name, {})
        require(entry.get("status") in ("DECLARED", "VERIFIED", "UNVERIFIED"), "Invalid build evidence status")
        require(isinstance(entry.get("evidence"), str) and bool(entry["evidence"].strip()),
                "Build evidence or reason for uncertainty is required")
        require(entry.get("build") is None or isinstance(entry["build"], str), "Invalid build name")
        if entry["status"] == "VERIFIED":
            require(bool(entry.get("build")), "Verified build has no identity")
            proof = entry.get("verification_receipt", {})
            require(isinstance(proof, dict) and proof.get("path") and proof.get("sha256"),
                    "Verified build requires an authenticated verification receipt")
            proof_path = Path((proof_paths or {}).get(name, proof["path"]))
            require(proof_path.is_file() and sha256(proof_path) == proof["sha256"], "Build verification receipt hash mismatch")
            record = read_json(proof_path)
            input_name = "rare_vcf" if name == "source" else "panel_vcf"
            require(record.get("schema") == "r02_lai_build_verification_v1" and record.get("status") == "VERIFIED"
                    and record.get("build") == entry["build"]
                    and record.get("vcf_sha256") == receipts[input_name]["sha256"]
                    and isinstance(record.get("method"), str) and record["method"].strip(),
                    "Build verification receipt does not bind the declared build to this VCF")
            verified_files[f"{name}_build_verification"] = {
                "path": str(proof_path.resolve()), "sha256": proof["sha256"], "size_bytes": proof_path.stat().st_size}
    source, panel = data["source"], data["panel"]
    require(bool(source.get("build")), "Source build declaration is required")
    require(not panel.get("build") or source["build"] == panel["build"], "Source and panel builds differ")
    fasta = [entry.get("fasta_sha256") for entry in (source, panel)]
    require(not all(fasta) or fasta[0] == fasta[1], "Source and panel FASTA digests differ")
    verified = source["status"] == panel["status"] == "VERIFIED"
    return data, "BUILD_VERIFIED_BY_SUPPLIED_EVIDENCE" if verified else "BUILD_UNVERIFIED", verified_files


def contig_mapping(path, chrom):
    values = read_json(path)
    require(isinstance(values, dict) and values, "Explicit contig map is empty")
    require(all(isinstance(k, str) and k and v == chrom for k, v in values.items()),
            "Contig map must map explicit names to the requested chromosome only")
    return values


def record_key(fields, mapping):
    require(fields[0] in mapping, "VCF contig absent from explicit mapping")
    require(fields[1].isdigit() and int(fields[1]) > 0, "Invalid variant position")
    require(fields[3] not in ("", ".") and fields[4] not in ("", "."), "Missing REF or ALT")
    return mapping[fields[0]], int(fields[1]), fields[3], fields[4]


def check_roles(path, samples, role):
    selected, seen, ancestries = set(), set(), Counter()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and {"sample_id", "role", "ancestry"} <= set(reader.fieldnames),
                "Roles table lacks required fields")
        for row in reader:
            sid = row.get("sample_id")
            require(sid and not any(c.isspace() for c in sid), "Invalid sample in roles table")
            require(sid not in seen, "Duplicate identity in roles table")
            seen.add(sid)
            require(bool(row.get("role")), "Missing role")
            if row["role"] == role:
                require(bool(row.get("ancestry")), "Selected role has no ancestry annotation")
                selected.add(sid)
                ancestries[row["ancestry"]] += 1
    require(selected and selected == set(samples), "Panel sample membership is not exactly the selected role")
    return {"role": role, "n_role_members": len(selected), "role_ancestry_counts": dict(ancestries),
            "panel_sample_order_sha256": sample_hash(samples),
            "panel_sample_members_sha256": sample_hash(sorted(samples)), "roles_total_rows": len(seen)}


def free_disk(path, min_free_gib):
    free = shutil.disk_usage(path).free
    require(free >= min_free_gib * 1024**3, "Free disk is below min-free-gib")
    return free


def panel_catalogue(args, mapping):
    require(Path(args.panel_vcf).stat().st_size <= args.max_panel_bytes, "Panel exceeds max-panel-bytes")
    keys, positions, counts = set(), Counter(), Counter()
    object_bytes = 0
    with MetadataVCF(args.panel_vcf, args.max_line_bytes) as reader:
        roles = check_roles(args.roles, reader.header.samples, args.panel_role)
        for fields in reader:
            key = record_key(fields, mapping)
            require(key not in keys, "Duplicate panel allele key; correspondence is ambiguous")
            require(len(keys) < args.max_panel_records, "Panel exceeds max-panel-records")
            object_bytes += sys.getsizeof(key) + sum(sys.getsizeof(value) for value in key)
            if key[:2] not in positions:
                object_bytes += sys.getsizeof(key[:2]) + sum(sys.getsizeof(value) for value in key[:2]) + sys.getsizeof(0)
            keys.add(key)
            positions[key[:2]] += 1
            # Counts shared objects conservatively more than once. Container
            # memory limits remain the guard for interpreter and input buffers.
            accounted = object_bytes + sys.getsizeof(keys) + sys.getsizeof(positions)
            require(accounted <= args.max_panel_index_bytes, "Panel exceeds max-panel-index-bytes")
            counts["records"] += 1
            counts["biallelic_snv" if len(key[2]) == len(key[3]) == 1 and key[2] in "ACGT" and key[3] in "ACGT"
                   else "other_record_types_not_split"] += 1
    require(keys, "Empty panel catalogue")
    counts["conservative_index_bytes"] = accounted
    return keys, positions, dict(counts), roles


def rare_info(fields):
    info = {}
    for item in fields[7].split(";"):
        name, _, value = item.partition("=")
        require(name not in info, "Duplicate INFO key in source record")
        info[name] = value
    for name in ("ORIG_NALLELES", "RARE_ALLELE", "RARE_AC", "RARE_AN"):
        require(name in info and re.fullmatch(r"[0-9]+", info[name]) is not None,
                f"Missing or noninteger INFO/{name}")
        info[name] = int(info[name])
    return info


def validate_source_contract(contract, header, args):
    require(contract.get("schema") == "m02_1_minor_v1", "Unsupported source rare contract")
    require(str(contract.get("chromosome")) == args.chromosome, "Source contract chromosome mismatch")
    for name in ("cohort_samples_before", "cohort_samples_after"):
        require(contract.get(name) == args.expected_source_samples, "Source contract cohort size mismatch")
    samples = validate_rare_header(header, [], args.expected_source_samples)
    require(contract.get("cohort_sha256") == sample_hash(samples), "Source contract cohort hash mismatch")
    criteria = contract.get("criteria", {})
    require(criteria.get("min_mac_inclusive") == 2 and str(criteria.get("max_maf_inclusive")) == "0.01"
            and criteria.get("original_allele_count") == 2 and criteria.get("GT_REF_ALT_preserved") is True
            and criteria.get("frequency_denominator") == "called_alleles", "Unsupported source rarity contract")
    require(contract.get("counts", {}).get("rare") == args.expected_rare_sites, "Source catalogue size mismatch")


def run(args):
    if getattr(args, "mode", "catalogue_only") == "genotype_support":
        from r02_lai_genotype_support import run as genotype_run
        return genotype_run(args)
    started = time.monotonic()
    for value in (args.expected_source_samples, args.expected_rare_sites, args.max_panel_records,
                  args.max_panel_bytes, args.max_panel_index_bytes, args.max_line_bytes, args.resource_check_rows):
        require(value > 0, "Counts and resource limits must be positive")
    require(args.min_free_gib >= 0 and str(args.chromosome) in {str(i) for i in range(1, 23)},
            "Invalid disk reserve or autosome")
    output = Path(args.outdir)
    require(not output.exists(), "Output already exists; no overwrite permitted")
    require(output.parent.is_dir(), "Output parent directory does not exist")
    require(Path(args.panel_vcf).stat().st_size <= args.max_panel_bytes, "Panel exceeds max-panel-bytes")
    receipts = input_receipts(args)
    source_contract = read_json(args.rare_contract)
    builds, build_status, build_receipts = build_contract(args.build_evidence, receipts)
    mapping = contig_mapping(args.contig_map, args.chromosome)
    free_disk(output.parent, args.min_free_gib)
    panel, positions, panel_counts, role_summary = panel_catalogue(args, mapping)
    counts = Counter()
    with MetadataVCF(args.rare_vcf, args.max_line_bytes) as rare:
        validate_source_contract(source_contract, rare.header, args)
        output.mkdir(mode=0o700)
        part = output / "correspondencia_alelos_dnabr.tsv.partial"
        with part.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            previous = 0
            for fields in rare:
                key = record_key(fields, mapping)
                require(key[1] > previous, "Duplicate or decreasing source position")
                previous = key[1]
                info = rare_info(fields)
                selected = validate_rare_site(SimpleNamespace(info=info, alleles=(key[2], key[3])))
                require(info["RARE_AN"] <= 2 * args.expected_source_samples, "Source allele denominator exceeds diploid cohort")
                exact = key in panel
                if exact:
                    status = "EXACT_KEY" if build_status != "BUILD_UNVERIFIED" else "LEXICAL_MATCH_BUILD_UNVERIFIED"
                else:
                    status = "POSITION_PRESENT_ALLELE_MISMATCH" if positions[key[:2]] else "NO_RECORD"
                row = dict(zip(("chromosome", "position_bp", "REF", "ALT"), key))
                row.update(source_build=builds["source"]["build"], counted_allele=(key[2], key[3])[selected],
                           source_rare_allele_index=selected, source_rare_ac=info["RARE_AC"], source_rare_an=info["RARE_AN"],
                           source_cohort_n=args.expected_source_samples, source_cohort_sha256=source_contract["cohort_sha256"],
                           source_catalogue_sha256=receipts["rare_vcf"]["sha256"], original_allele_count=2,
                           source_site_id=":".join(map(str, key)), source_record_id=fields[2], panel_role=args.panel_role,
                           panel_source_sha256=receipts["panel_vcf"]["sha256"], panel_records_at_position=positions[key[:2]],
                           correspondence_status=status, build_status=build_status,
                           genotype_evidence_status="NOT_READ" if exact else "NO_GENOTYPE_EVIDENCE_IN_THIS_SOURCE",
                           support_status="NOT_EVALUATED_CATALOGUE_ONLY", **dict.fromkeys(SUPPORT_FIELDS, "NA"))
                writer.writerow(row)
                counts[status] += 1
                counts["rows"] += 1
                counts["source_minor_ref" if selected == 0 else "source_minor_alt"] += 1
                if counts["rows"] % args.resource_check_rows == 0:
                    free_disk(output, args.min_free_gib)
                    print(f"L1 metadata chr{args.chromosome}: rows={counts['rows']} elapsed_s={time.monotonic()-started:.1f}",
                          file=sys.stderr, flush=True)
        require(counts["rows"] == args.expected_rare_sites, "Observed catalogue count differs from contract")
        for key, counter in (("rare_ref", "source_minor_ref"), ("rare_alt", "source_minor_alt")):
            require(source_contract["counts"].get(key, 0) == counts[counter], "Selected REF/ALT count differs from source contract")
    verify_inputs_unchanged(receipts)
    verify_inputs_unchanged(build_receipts)
    free_disk(output, args.min_free_gib)
    table = output / "correspondencia_alelos_dnabr.tsv"
    part.rename(table)
    manifest = {
        "schema": SCHEMA, "status": STATUS, "mode": "catalogue_only", "evidence_level": "FEASIBILITY",
        "public_distribution_allowed": False, "contains_sample_identifiers": False,
        "contains_individual_genotypes": False, "genotype_fields_decoded": 0,
        "input_files": receipts, "build_evidence": builds, "build_status": build_status,
        "build_verification_files": build_receipts,
        "contig_mapping": mapping, "chromosome": args.chromosome, "counts": dict(counts),
        "panel_counts": panel_counts, "role_validation": role_summary,
        "source_cohort_n": args.expected_source_samples, "source_cohort_sha256": source_contract["cohort_sha256"],
        "source_frequency_unchanged": True, "source_catalogue_all_records_preserved": True,
        "catalogue_correspondence": "COMPLETE", "genotype_support": "NOT_EVALUATED",
        "quality_support": "NOT_EVALUATED", "donor_phase_support": "NOT_EVALUATED", "map_coverage": "NOT_EVALUATED",
        "source_site_id_definition": "Current normalized CHROM:POS:REF:ALT; original biallelicity is certified only relative to the M01 input",
        "missing_record_definition": "No compatible record in this source, not zero copies and not proof the locus was never measured",
        "genotype_scope": "No GT, DP, GQ, AD, phase or panel AC/AN read or inferred; support fields remain NA",
        "role_scope": "Existing role membership authenticated; no role assignment or independence certification",
        "limits": {name: getattr(args, name) for name in ("max_panel_records", "max_panel_bytes", "max_panel_index_bytes", "max_line_bytes", "min_free_gib", "resource_check_rows")},
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "code_sha256": {Path(__file__).name: sha256(__file__), "r02_genomic_pair_evidence.py": sha256(Path(__file__).with_name("r02_genomic_pair_evidence.py"))},
        "outputs_sha256": {table.name: sha256(table)}, "output_rows": counts["rows"],
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        value.add_argument("--" + name.replace("_", "-"), required=True)
    value.add_argument("--mode", choices=("catalogue_only", "genotype_support"), default="catalogue_only")
    value.add_argument("--genotype-contract", help="Required authenticated sample/quality contract in genotype_support mode")
    value.add_argument("--source-build-verification", help="Explicit staged source-build proof in genotype_support mode")
    value.add_argument("--panel-build-verification", help="Explicit staged panel-build proof in genotype_support mode")
    value.add_argument("--input-hashes", help="Expected SHA256 for every named input; mandatory for genotype_support")
    value.add_argument("--panel-role", required=True)
    value.add_argument("--chromosome", required=True)
    value.add_argument("--expected-source-samples", type=int, required=True)
    value.add_argument("--expected-rare-sites", type=int, required=True)
    value.add_argument("--outdir", required=True)
    value.add_argument("--max-panel-records", type=int, default=100000, help="Engineering bound, not a biological cutoff")
    value.add_argument("--max-panel-bytes", type=int, default=1073741824, help="Maximum compressed/on-disk input bytes")
    value.add_argument("--max-panel-index-bytes", type=int, default=134217728, help="Estimated bound for catalogue index, or one-position buffer in genotype mode; not a total RSS cap")
    value.add_argument("--max-line-bytes", type=int, default=8388608)
    value.add_argument("--min-free-gib", type=float, default=2)
    value.add_argument("--resource-check-rows", type=int, default=10000)
    return value


if __name__ == "__main__":
    os.umask(0o077)
    run(parser().parse_args())
