#!/usr/bin/env python3
"""Stream a pinned REF_TRAIN VCF into phase-aware neural reference chunks.

This adapter preserves every ALT allele, missing genotype and unphased call.
It does not select COMMON loci or read simulated truth.  Chunk size is a
required technical bound supplied by the execution request.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_people(roles_path):
    with Path(roles_path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == ["sample_id", "role", "ancestry", "split", "family_component"],
                "roles columns differ")
        rows = [row for row in reader if row["role"] == "REF_TRAIN"]
    require(rows and all(row["split"] == "REFERENCE" for row in rows), "REF_TRAIN roles differ")
    ids = [row["sample_id"] for row in rows]
    require(len(ids) == len(set(ids)), "duplicate REF_TRAIN person")
    return ids, rows


def parse_gt(value, gt_index, alt_index, n_alts=None):
    token = value.split(":")[gt_index]
    if token in (".", "./.", ".|."):
        return -1, False, (-1, -1), False
    require(not ("|" in token and "/" in token), "mixed GT separators")
    phased = "|" in token
    alleles = token.replace("|", "/").split("/")
    require(len(alleles) == 2, "diploid GT required")
    require(all(a == "." or re.fullmatch(r"[0-9]+", a) for a in alleles), "invalid GT allele")
    require(n_alts is None or all(a == "." or int(a) <= n_alts for a in alleles),
            "GT allele outside REF/ALT")
    if any(allele == "." for allele in alleles):
        return -1, False, (-1, -1), False
    indexes = tuple(int(allele) for allele in alleles)
    dosage = sum(allele == alt_index for allele in indexes)
    haplotypes = tuple(int(allele == alt_index) for allele in indexes) if phased else (-1, -1)
    return dosage, True, haplotypes, phased


def flush(out, index, records, people, compressed=False):
    n = len(records); m = len(people)
    dosage = np.empty((n, m), dtype=np.int8)
    observed = np.empty((n, m), dtype=np.bool_)
    haplotype_alt = np.empty((n, m, 2), dtype=np.int8)
    phase_observed = np.empty((n, m), dtype=np.bool_)
    for row_index, record in enumerate(records):
        dosage[row_index] = record.pop("dosage")
        observed[row_index] = record.pop("observed")
        haplotype_alt[row_index] = record.pop("haplotype_alt")
        phase_observed[row_index] = record.pop("phase_observed")
    target = out / f"reference-detail.{index:06d}.npz"
    arrays = dict(
             chromosome=np.asarray([r["chromosome"] for r in records]),
             position_bp=np.asarray([r["position_bp"] for r in records], dtype=np.int64),
             ref=np.asarray([r["ref"] for r in records]), alt=np.asarray([r["alt"] for r in records]),
             source_alt_index=np.asarray([r["source_alt_index"] for r in records], dtype=np.int16),
             dosage=dosage, observed=observed, haplotype_alt=haplotype_alt,
             phase_observed=phase_observed,
             phase_set=np.asarray([r["phase_set"] for r in records]),
             source_record_index=np.asarray([r["source_record_index"] for r in records], dtype=np.int64))
    partial = target.with_suffix('.partial.npz')
    (np.savez_compressed if compressed else np.savez)(partial, **arrays)
    with np.load(partial, allow_pickle=False) as stored:
        require(set(stored.files) == set(arrays) and all(stored[k].dtype == v.dtype and
                np.array_equal(stored[k], v) for k, v in arrays.items()), "NPZ roundtrip differs")
    os.replace(partial, target)
    return {"path": target.name, "bytes": target.stat().st_size, "sha256": sha(target),
            "loci": n, "observed_genotypes": int(observed.sum()),
            "phased_genotypes": int(phase_observed.sum())}


def run(request_path, request_sha256, out):
    require(sha(request_path) == request_sha256, "request SHA256 differs")
    request = json.loads(Path(request_path).read_text())
    fields = {"schema", "chromosome", "vcf_path", "vcf_sha256", "roles_path",
              "roles_sha256", "max_loci_per_chunk"}
    version = request.get("schema")
    require(version in {"r02_lai_reference_detail_request_v1", "r02_lai_reference_detail_request_v2"},
            "request schema differs")
    if version.endswith('_v2'):
        fields |= {"compressed", "min_free_bytes"}
        require(type(request.get('compressed')) is bool and type(request.get('min_free_bytes')) is int
                and request['min_free_bytes'] > 0, "explicit storage bounds required")
    require(set(request) == fields, "request fields differ")
    compressed = request.get('compressed', False)
    chrom = str(request["chromosome"])
    require(chrom in map(str, range(1, 23)), "invalid autosome")
    vcf = Path(request["vcf_path"]); roles = Path(request["roles_path"])
    require(vcf.is_file() and sha(vcf) == request["vcf_sha256"], "reference VCF differs")
    require(roles.is_file() and sha(roles) == request["roles_sha256"], "roles differ")
    limit = request["max_loci_per_chunk"]
    require(type(limit) is int and limit > 0, "explicit positive locus chunk bound required")
    people, role_rows = reference_people(roles)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    chunks, records, source_rows = [], [], 0
    header_seen, previous_position = False, 0
    def emit():
        require(shutil.disk_usage(out).free >= request.get('min_free_bytes', 0), 'Disk reserve reached')
        return flush(out, len(chunks), records, people, compressed)
    with gzip.open(vcf, "rt") as stream:
        for line in stream:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                require(not header_seen and source_rows == 0, "duplicate or late VCF header")
                header = line.rstrip("\n").split("\t")
                require(header[9:] == people, "VCF and REF_TRAIN person order differ")
                header_seen = True
                continue
            require(header_seen, "VCF data before sample header")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(people), "VCF row width differs")
            row_chrom = fields[0].removeprefix("chr")
            require(row_chrom == chrom, "VCF chromosome differs")
            position = int(fields[1])
            require(position > 0 and position >= previous_position, "unordered VCF coordinate")
            previous_position = position
            alts = fields[4].split(",")
            require(fields[3] not in ("", ".") and all(alt not in ("", ".") for alt in alts),
                    "REF/ALT missing")
            formats = fields[8].split(":")
            require("GT" in formats and len(formats) == len(set(formats)), "GT missing or repeated FORMAT")
            gt_index = formats.index("GT"); source_rows += 1
            ps_index = formats.index('PS') if 'PS' in formats else None
            phase_sets = []
            for value in fields[9:]:
                cell = value.split(':')
                require(gt_index < len(cell), "GT value missing")
                phase_sets.append(cell[ps_index] if ps_index is not None and ps_index < len(cell) else '.')
            for alt_index, alt in enumerate(alts, 1):
                parsed = [parse_gt(value, gt_index, alt_index, len(alts)) for value in fields[9:]]
                records.append({"chromosome": row_chrom, "position_bp": position,
                                "ref": fields[3], "alt": alt, "source_alt_index": alt_index,
                                "phase_set": phase_sets, "source_record_index": source_rows,
                                "dosage": [x[0] for x in parsed], "observed": [x[1] for x in parsed],
                                "haplotype_alt": [x[2] for x in parsed],
                                "phase_observed": [x[3] for x in parsed]})
                if len(records) == limit:
                    chunks.append(emit()); records = []
    require(header_seen and source_rows > 0, "empty reference VCF")
    if records:
        chunks.append(emit())
    people_path = out / "reference-people.tsv"
    with people_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(role_rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(role_rows)
    result = {"schema": "r02_lai_reference_detail_v2",
              "status": "REFERENCE_DETAIL_COMPLETE_NO_COMMON_SELECTION_NO_TRAINING",
              "chromosome": chrom, "people": len(people), "role": "REF_TRAIN",
              "source_vcf_sha256": request["vcf_sha256"], "source_roles_sha256": request["roles_sha256"],
              "source_rows": source_rows, "expanded_alt_loci": sum(x["loci"] for x in chunks),
              "chunks": chunks, "people_file": {"path": people_path.name, "sha256": sha(people_path)},
              "phase_policy": "haplotype_alt=-1 for unphased/missing; dosage remains available when observed",
              "phase_set_policy": "Literal PS preserved; missing PS remains '.', no cross-block or parental correspondence certified",
              "storage_compressed": compressed, "roundtrip_content_verified": True,
              "truth_read": False, "common_loci_selected": False, "training_allowed": False}
    (out / "manifest.partial.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(out / 'manifest.partial.json', out / 'manifest.json')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.request, args.request_sha256, args.out)
