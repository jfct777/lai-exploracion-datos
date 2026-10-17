#!/usr/bin/env python3
"""Audit every REF in an authenticated marker inventory against a FASTA.

The input is a gzip TSV whose first columns are CHROM, POS, REF and ALT. Rare
inventories may carry additional columns; they are recorded and deliberately
ignored. The tool writes a temporary sample-free VCF and asks bcftools norm
``-c w`` to report every REF mismatch. It never filters, repairs or emits a VCF.
ALT is carried through only to make valid VCF records; FASTA compatibility says
nothing about ALT, genotypes, phase, rarity or unique assembly identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile
import time

import r02_lai_allele_support as base
from r02_lai_build_verification import file_receipt, reference_contract

SCHEMA = "r02_marker_reference_audit_job_v1"
RESULT_SCHEMA = "r02_marker_reference_audit_result_v1"
KINDS = {"source", "rare"}
AUTOSOMES = {str(x) for x in range(1, 23)}
LIMIT_KEYS = {"max_records", "max_line_bytes", "max_stderr_bytes", "max_output_bytes",
              "min_free_bytes", "timeout_seconds"}
MISMATCH = re.compile(r"^REF_MISMATCH\t([^\t]+)\t([0-9]+)\t([^\t]+)\t([^\t]+)$")
SUMMARY = re.compile(r"^Lines\s+total/split/realigned/skipped:\s*([0-9]+)/([0-9]+)/([0-9]+)/([0-9]+)$")


def exact_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            base.require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")))


def validate_manifest(value):
    base.require(isinstance(value, dict) and set(value) ==
                 {"schema", "kind", "chromosome", "alias", "inventory", "reference", "limits"},
                 "Invalid marker-reference job fields")
    base.require(value["schema"] == SCHEMA and value["kind"] in KINDS,
                 "Unknown marker-reference schema or kind")
    chrom = str(value["chromosome"])
    base.require(chrom in AUTOSOMES and value["chromosome"] == chrom,
                 "Chromosome must be a string autosome 1-22")
    base.require(isinstance(value["alias"], str) and value["alias"].strip() and
                 not any(c.isspace() for c in value["alias"]), "Invalid inventory contig alias")
    expected_aliases = {chrom, "chr" + chrom}
    base.require(value["alias"] in expected_aliases,
                 "Inventory alias must be N or chrN for the declared chromosome")
    inv = value["inventory"]
    base.require(isinstance(inv, dict) and set(inv) == {"path", "sha256", "rows"} and
                 isinstance(inv["path"], str) and inv["path"] and
                 re.fullmatch(r"[0-9a-f]{64}", inv["sha256"] or "") and
                 type(inv["rows"]) is int and inv["rows"] > 0, "Invalid inventory descriptor")
    ref = value["reference"]
    base.require(isinstance(ref, dict) and set(ref) ==
                 {"fasta_path", "fai_path", "identity_path", "identity_sha256", "reference_contig"},
                 "Invalid reference descriptor")
    base.require(all(isinstance(ref[k], str) and ref[k] for k in
                     ("fasta_path", "fai_path", "identity_path", "reference_contig")) and
                 re.fullmatch(r"[0-9a-f]{64}", ref["identity_sha256"] or ""),
                 "Incomplete reference descriptor")
    base.require(ref["reference_contig"] in expected_aliases,
                 "Reference contig must be N or chrN for the same declared chromosome")
    limits = value["limits"]
    base.require(isinstance(limits, dict) and set(limits) == LIMIT_KEYS and
                 all(type(limits[k]) is int and limits[k] > 0 for k in LIMIT_KEYS),
                 "All resource limits must be explicit positive integers")
    base.require(inv["rows"] <= limits["max_records"], "Declared rows exceed max-records")
    base.require(limits["max_stderr_bytes"] >= inv["rows"] * 64 + 1024,
                 "max-stderr-bytes must cover the declared all-mismatch case")
    base.require(limits["max_output_bytes"] >= inv["rows"] * 64 + 4096,
                 "max-output-bytes must cover the declared all-mismatch result")
    return value


def bounded_lines(path, max_line_bytes):
    path = Path(path)
    with path.open("rb") as raw:
        base.require(raw.read(2) == b"\x1f\x8b", "Inventory must be gzip compressed")
    with gzip.open(path, "rb") as handle:
        while True:
            line = handle.readline(max_line_bytes + 1)
            if not line:
                return
            base.require(len(line) <= max_line_bytes, "Inventory line exceeds max-line-bytes")
            try:
                yield line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise ValueError("Inventory is not UTF-8") from error


def read_fai(path):
    lengths = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split("\t")
        base.require(len(fields) >= 5 and fields[0] not in lengths and fields[1].isdigit() and
                     int(fields[1]) > 0, "Malformed or duplicate FASTA index contig")
        lengths[fields[0]] = int(fields[1])
    base.require(lengths, "Empty FASTA index")
    return lengths


def materialize_vcf(inventory, manifest, output, contig_length):
    rows = bounded_lines(inventory, manifest["limits"]["max_line_bytes"])
    try:
        header = next(rows).split("\t")
    except StopIteration as error:
        raise ValueError("Empty marker inventory") from error
    base.require(len(header) >= 4 and header[:4] == ["CHROM", "POS", "REF", "ALT"] and
                 len(header) == len(set(header)), "Inventory header must begin exactly CHROM POS REF ALT")
    extras = header[4:]
    base.require(manifest["kind"] == "rare" or not extras,
                 "Only rare inventories may contain explicitly ignored extra columns")
    count, previous = 0, 0
    with Path(output).open("x") as target:
        target.write("##fileformat=VCFv4.2\n")
        target.write(f"##contig=<ID={manifest['reference']['reference_contig']},length={contig_length}>\n")
        target.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for line in rows:
            fields = line.split("\t")
            base.require(len(fields) == len(header), "Inventory row width differs from header")
            alias, position, ref, alt = fields[:4]
            base.require(alias == manifest["alias"], "Inventory contig differs from manifest alias")
            base.require(position.isdigit() and 0 < int(position) <= contig_length,
                         "Inventory position is outside the authenticated contig")
            pos = int(position)
            base.require(pos >= previous, "Inventory positions decrease")
            base.require(len(ref) == 1 and ref in "ACGT", "REF must be an uppercase canonical SNV allele")
            base.require(len(alt) == 1 and alt in "ACGT" and alt != ref,
                         "ALT must be valid SNV syntax; this does not certify ALT against FASTA")
            count += 1
            base.require(count <= manifest["limits"]["max_records"], "Inventory exceeds max-records")
            target.write(f"{manifest['reference']['reference_contig']}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")
            previous = pos
    base.require(count > 0 and count == manifest["inventory"]["rows"],
                 "Inventory row count differs from manifest or is zero")
    return count, extras


def parse_stderr(path, expected_rows, reference_contig, mismatch_path, max_line_bytes):
    mismatch_count, summary = 0, None
    with Path(path).open("rb") as source, gzip.open(mismatch_path, "wt") as target:
        target.write("CHROM\tPOS\tVCF_REF\tFASTA_REF\n")
        while True:
            raw = source.readline(max_line_bytes + 1)
            if not raw:
                break
            base.require(len(raw) <= max_line_bytes, "bcftools stderr line exceeds bound")
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise ValueError("bcftools stderr is not UTF-8") from error
            if not line:
                continue
            match = MISMATCH.fullmatch(line)
            if match:
                base.require(match[1] == reference_contig, "bcftools reported a mismatch on another contig")
                # vcfnorm.c emits VCF REF first and fetched FASTA REF second.
                target.write("\t".join(match.groups()) + "\n")
                mismatch_count += 1
                base.require(mismatch_count <= expected_rows, "More mismatches than input records")
                continue
            match = SUMMARY.fullmatch(line)
            base.require(match is not None and summary is None, "Unknown or duplicate bcftools stderr line")
            summary = tuple(int(x) for x in match.groups())
    base.require(summary is not None and summary[0] == expected_rows and summary[1:] == (0, 0, 0),
                 "bcftools processed-count summary differs from the exact input")
    return mismatch_count, summary


def run(args):
    started = time.monotonic()
    manifest_receipt = file_receipt(args.manifest, args.manifest_sha256)
    manifest = validate_manifest(exact_json(args.manifest))
    limits, reference = manifest["limits"], manifest["reference"]
    output = Path(args.out)
    mismatch_output = Path(args.mismatches)
    base.require(output.parent.is_dir() and mismatch_output.parent == output.parent and
                 not output.exists() and not mismatch_output.exists(),
                 "Output parent missing, outputs differ in directory, or result exists")
    base.require(shutil.disk_usage(output.parent).free >= limits["min_free_bytes"], "Free disk below declared reserve")
    receipts = {
        "manifest": manifest_receipt,
        "inventory": file_receipt(args.inventory, manifest["inventory"]["sha256"]),
        "identity": file_receipt(args.identity, reference["identity_sha256"]),
    }
    identity = reference_contract(exact_json(args.identity))
    receipts["fasta"] = file_receipt(args.fasta, identity["fasta_sha256"])
    receipts["fai"] = file_receipt(args.fai, identity["fai_sha256"])
    base.require(reference["reference_contig"] in identity["contigs"], "Reference contig absent from identity contract")
    length = identity["contigs"][reference["reference_contig"]]
    base.require(read_fai(args.fai).get(reference["reference_contig"]) == length,
                 "FAI length differs from identity contract")
    version_result = subprocess.run([args.bcftools, "--version"], capture_output=True, text=True, timeout=30)
    base.require(version_result.returncode == 0 and version_result.stdout.startswith("bcftools "),
                 "bcftools version query failed")
    with tempfile.TemporaryDirectory(prefix="r02-marker-ref-", dir=output.parent) as tmp:
        vcf = Path(tmp) / "inventory.vcf"
        stderr_path = Path(tmp) / "bcftools.stderr"
        mismatch_tmp = Path(tmp) / "mismatches.tsv.gz"
        n_rows, extras = materialize_vcf(args.inventory, manifest, vcf, length)
        command = [args.bcftools, "norm", "-c", "w", "-f", str(Path(args.fasta).resolve()), "-Ou", str(vcf)]
        def bound_stderr():
            resource.setrlimit(resource.RLIMIT_FSIZE,
                               (limits["max_stderr_bytes"], limits["max_stderr_bytes"]))
        with stderr_path.open("wb") as stderr_handle:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stderr_handle,
                                    timeout=limits["timeout_seconds"], preexec_fn=bound_stderr)
        base.require(stderr_path.stat().st_size <= limits["max_stderr_bytes"],
                     "bcftools stderr exceeds declared bound")
        base.require(result.returncode == 0, "bcftools failed; no compatibility status produced")
        mismatch_count, summary = parse_stderr(stderr_path, n_rows, reference["reference_contig"],
                                                mismatch_tmp, limits["max_line_bytes"])
        base.require(mismatch_tmp.stat().st_size <= limits["max_output_bytes"],
                     "Compressed mismatch table exceeds max-output-bytes")
        shutil.move(mismatch_tmp, mismatch_output)
    base.verify_inputs_unchanged(receipts)
    base.require(shutil.disk_usage(output.parent).free >= limits["min_free_bytes"], "Free disk fell below reserve")
    mismatch_receipt = {"path": str(mismatch_output.resolve()), "sha256": base.sha256(mismatch_output),
                        "size_bytes": mismatch_output.stat().st_size, "rows": mismatch_count,
                        "columns": ["CHROM", "POS", "VCF_REF", "FASTA_REF"]}
    status = "COMPLETED_INCOMPATIBLE" if mismatch_count else "VERIFIED"
    report = {
        "schema": RESULT_SCHEMA, "status": status, "kind": manifest["kind"],
        "chromosome": manifest["chromosome"], "alias": manifest["alias"],
        "reference_contig": reference["reference_contig"], "assembly": identity["assembly"],
        "records_checked": n_rows, "mismatch_count": mismatch_count, "mismatch_file": mismatch_receipt,
        "ignored_extra_columns": extras, "bcftools_summary": {
            "total": summary[0], "split": summary[1], "realigned": summary[2], "skipped": summary[3]},
        "input_files": receipts, "bcftools_version": version_result.stdout.splitlines()[0],
        "command_template": "bcftools norm -c w -f <authenticated_fasta> -Ou <temporary_sample_free_vcf> > /dev/null",
        "interpretation": "REF/FASTA compatibility for every inventory row only; ALT, GT, samples, phase, rarity and unique assembly identity are not certified",
        "contains_genotypes": False, "contains_sample_identifiers": False,
        "resource_bounds": {**limits, "stderr_bound_formula": "rows*64+1024",
                            "output_bound_formula": "rows*64+4096"},
        "source_sha256": base.sha256(__file__), "elapsed_seconds": time.monotonic() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    base.require(len(encoded) + mismatch_output.stat().st_size <= limits["max_output_bytes"],
                 "Combined result exceeds max-output-bytes")
    output.write_bytes(encoded)
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha256", "inventory", "fasta", "fai", "identity", "out", "mismatches"):
        value.add_argument("--" + name, required=True)
    value.add_argument("--bcftools", default="bcftools")
    return value


if __name__ == "__main__":
    base.os.umask(0o077)
    run(parser().parse_args())
