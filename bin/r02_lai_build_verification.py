#!/usr/bin/env python3
"""Produce an authenticated, all-record REF/FASTA compatibility receipt.

This does not discover an assembly from genotypes or certify allele splitting.
The reference's assembly identity must come from a separately reviewed source
contract (r02_lai_reference_identity_v1). Its exact keys are assembly,
assembly_accession, source_uri, identity_evidence, fasta_sha256, fai_sha256,
contigs {reference_contig: length}, and schema. Both that contract and the VCF
must have independently supplied expected SHA256. A matching REF audit alone
does not prove unique assembly identity; the supplied provenance does that job.

Run separately for source and panel. All VCF rows are checked; contig aliases
are explicitly renamed in a pipe only, never in the input or persistent VCF.
bcftools norm --check-ref e checks REF without fixing/swapping incorrect REF,
splitting sites or selecting only passing sites. Its normalized output is
discarded, never persisted. Do not add -N: it bypassed REF checking in the
pinned tool, as demonstrated by the negative mismatch regression test.

Example (engineering caps belong to the container/Nextflow caller):
  python3 r02_lai_build_verification.py --vcf panel.vcf.gz \
    --expected-vcf-sha256 <sha> --fasta reference.fa \
    --reference-contract reference_identity.json --expected-reference-contract-sha256 <sha> \
    --contig-map contigs.json --expected-contig-map-sha256 <sha> \
    --chromosome 22 --reference-contig chr22 --max-records 100000000 \
    --max-line-bytes 8388608 --timeout-seconds 3600 --min-free-gib 1 --out receipt.json

Requires an existing, authenticated plain FASTA and adjacent .fai. No FASTA
copy/index is made. Preflight verifies availability/reserve before hashing.
Hashing before and after reads the FASTA twice; account for that I/O explicitly.
Synthetic tests use 1 CPU/2 GB; production runtime/RSS must still be measured.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import r02_lai_allele_support as base

METHOD = "bcftools_norm_check_ref_e_all_records_v1"


def reference_contract(data):
    base.require(isinstance(data, dict) and set(data) == {"schema", "assembly", "assembly_accession", "source_uri",
                 "identity_evidence", "fasta_sha256", "fai_sha256", "contigs"}, "Invalid reference identity contract fields")
    base.require(data["schema"] == "r02_lai_reference_identity_v1", "Unknown reference identity schema")
    for name in ("assembly", "assembly_accession", "source_uri", "identity_evidence"):
        base.require(isinstance(data[name], str) and data[name].strip(), "Reference identity provenance is missing")
    for name in ("fasta_sha256", "fai_sha256"):
        base.require(isinstance(data[name], str) and re.fullmatch(r"[0-9a-f]{64}", data[name]), "Invalid reference digest")
    base.require(isinstance(data["contigs"], dict) and data["contigs"] and
                 all(isinstance(key, str) and key and isinstance(value, int) and not isinstance(value, bool) and value > 0
                     for key, value in data["contigs"].items()), "Explicit reference contig lengths required")
    return data


def file_receipt(path, expected):
    path = Path(path)
    base.require(path.is_file() and re.fullmatch(r"[0-9a-f]{64}", expected or ""), "Missing file or expected SHA256")
    result = dict(path=str(path.resolve()), size_bytes=path.stat().st_size, sha256=base.sha256(path))
    base.require(result["sha256"] == expected, "Build audit input SHA256 mismatch")
    return result


def run(args):
    started = time.monotonic()
    out = Path(args.out)
    base.require(not out.exists() and out.parent.is_dir(), "Receipt exists or output parent is absent")
    base.require(args.chromosome in {str(i) for i in range(1, 23)}, "Autosome 1-22 required")
    base.require(args.max_records > 0 and args.max_line_bytes > 0 and args.timeout_seconds > 0 and args.min_free_gib >= 0,
                 "Invalid build-audit resource bounds")
    fasta, fai = Path(args.fasta), Path(str(args.fasta) + ".fai")
    base.require(fasta.is_file() and fai.is_file(), "Existing FASTA and adjacent .fai required; no index will be created")
    with fasta.open("rb") as handle:
        base.require(handle.read(1) == b">", "Audit requires plain FASTA, not compressed input")
    base.free_disk(out.parent, args.min_free_gib)
    receipts = dict(reference_contract=file_receipt(args.reference_contract, args.expected_reference_contract_sha256),
                    contig_map=file_receipt(args.contig_map, args.expected_contig_map_sha256))
    identity = reference_contract(base.read_json(args.reference_contract))
    base.require(args.reference_contig in identity["contigs"], "Reference contig absent from identity contract")
    length = identity["contigs"][args.reference_contig]
    receipts["fai"] = file_receipt(fai, identity["fai_sha256"])
    lengths = {}
    for line in fai.read_text().splitlines():
        fields = line.split("\t")
        base.require(len(fields) >= 5 and fields[0] not in lengths and fields[1].isdigit(), "Malformed or duplicate FASTA index contig")
        lengths[fields[0]] = int(fields[1])
    base.require(lengths.get(args.reference_contig) == length, "FASTA index length differs from identity contract")
    mapping = base.contig_mapping(args.contig_map, args.chromosome)
    receipts["vcf"] = file_receipt(args.vcf, args.expected_vcf_sha256)
    n_records, previous, observed_contigs = 0, 0, set()
    with base.MetadataVCF(args.vcf, args.max_line_bytes) as reader:
        for line in reader.header.lines:
            match = re.match(r"##contig=<ID=([^,>]+),length=([0-9]+)[,>]", line)
            if match and match[1] in mapping:
                base.require(int(match[2]) == length, "VCF header contig length differs from authenticated FASTA")
        for fields in reader:
            key = base.record_key(fields, mapping)
            base.require(key[1] >= previous, "Decreasing VCF position in build audit")
            base.require(key[1] + len(key[2]) - 1 <= length and all(c in "ACGTN" for c in key[2]),
                         "REF lies outside the authenticated contig or contains unsupported symbols")
            previous = key[1]
            observed_contigs.add(fields[0])
            n_records += 1
            base.require(n_records <= args.max_records, "Build audit exceeds max-records")
    base.require(n_records > 0 and len(observed_contigs) == 1, "Build audit requires one nonempty VCF contig")
    receipts["fasta"] = file_receipt(fasta, identity["fasta_sha256"])
    version = subprocess.run([args.bcftools, "--version"], check=True, capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory(prefix="r02-ref-audit-", dir=out.parent) as directory:
        alias_path = Path(directory) / "aliases.tsv"
        aliases = {name: args.reference_contig for name in observed_contigs}
        alias_path.write_text("".join(f"{name}\t{target}\n" for name, target in sorted(aliases.items())))
        annotate = [args.bcftools, "annotate", "--rename-chrs", str(alias_path), "--no-version", "-Ou", str(args.vcf)]
        norm = [args.bcftools, "norm", "--check-ref", "e", "--fasta-ref", str(fasta), "-Ou", "-"]
        with (Path(directory) / "stderr.log").open("w+b") as stderr:
            producer = subprocess.Popen(annotate, stdout=subprocess.PIPE, stderr=stderr)
            try:
                result = subprocess.run(norm, stdin=producer.stdout, stdout=subprocess.DEVNULL, stderr=stderr,
                                        timeout=args.timeout_seconds)
                producer.stdout.close()
                code = producer.wait(timeout=args.timeout_seconds)
                base.require(code == 0 and result.returncode == 0,
                             "bcftools REF/FASTA audit failed; no VERIFIED receipt produced (no allele repair attempted)")
            finally:
                if producer.stdout and not producer.stdout.closed:
                    producer.stdout.close()
                if producer.poll() is None:
                    producer.kill()
                    producer.wait()
    base.verify_inputs_unchanged(receipts)
    base.free_disk(out.parent, args.min_free_gib)
    proof = dict(schema="r02_lai_build_verification_v1", status="VERIFIED", build=identity["assembly"],
                 vcf_sha256=receipts["vcf"]["sha256"], fasta_sha256=receipts["fasta"]["sha256"], method=METHOD,
                 reference_provenance=identity, input_files=receipts,
                 reference_audit=dict(chromosome=args.chromosome, records_checked=n_records, mismatches=0,
                                      bcftools_exit_code=0, scope="ALL_VCF_RECORDS", reference_contig=args.reference_contig,
                                      vcf_contigs=sorted(observed_contigs), contig_rename=aliases),
                 command_templates=["bcftools annotate --rename-chrs <explicit_aliases.tsv> --no-version -Ou <vcf>",
                                    "bcftools norm --check-ref e --fasta-ref <authenticated_fasta> -Ou - > /dev/null"],
                 package_versions=dict(bcftools=version), source_sha256=base.sha256(__file__),
                 interpretation="REF compatibility across this entire VCF under the supplied authenticated assembly provenance; not genotype quality, phase, normalization, unique build identification or original allele-representation proof",
                 contains_sample_identifiers=False, contains_individual_genotypes=False,
                 elapsed_seconds=time.monotonic() - started, created_at=datetime.now(timezone.utc).isoformat())
    with out.open("x") as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return proof


def parser():
    value = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("vcf", "expected-vcf-sha256", "fasta", "reference-contract", "expected-reference-contract-sha256",
                 "contig-map", "expected-contig-map-sha256", "chromosome", "reference-contig", "out"):
        value.add_argument("--" + name, required=True)
    value.add_argument("--bcftools", default="bcftools")
    value.add_argument("--max-records", type=int, required=True)
    value.add_argument("--max-line-bytes", type=int, required=True)
    value.add_argument("--timeout-seconds", type=int, required=True)
    value.add_argument("--min-free-gib", type=float, required=True)
    return value


if __name__ == "__main__":
    base.os.umask(0o077)
    run(parser().parse_args())
