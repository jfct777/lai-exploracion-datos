#!/usr/bin/env python3
"""Preserve small evidence from successful checkpointed M01 tasks, not genotypes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re


TASK_FILES = {
    "ANNOTATE_ORIGINAL_ALLELES": ("annotation.log", "annotation.storage.json"),
    "NORMALIZE_ANNOTATED_ALLELES": ("norm.log", "normalization.storage.json"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_small_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Audit evidence must be a regular, nonsymlink file: " + str(path))
    before = path.stat()
    if before.st_size > 16 * 1024**2:
        raise ValueError("Unexpectedly large preprocessing audit file: " + str(path))
    data = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Audit evidence changed while being read: " + str(path))
    return data


def write_exclusive(path: Path, data: bytes):
    if path.is_symlink():
        raise ValueError("Audit destination must not be a symlink: " + str(path))
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        if read_small_file(path) != data:
            raise ValueError("Refuse to replace different existing audit evidence: " + str(path))


def collect_preprocess_audit(folder: Path, chrom: int) -> dict:
    """Resolve only successful trace task hashes; copying arbitrary work files is forbidden."""
    if type(chrom) is not int or not 1 <= chrom <= 22:
        raise ValueError("Expected an autosome number, 1..22")
    folder = Path(folder).resolve(strict=True)
    trace_path = folder / "trace.tsv"
    trace_data = read_small_file(trace_path)
    rows = list(csv.DictReader(io.StringIO(trace_data.decode()), delimiter="\t"))
    work = folder / "work"
    if work.is_symlink() or not work.is_dir():
        raise ValueError("Expected the chromosome's original local Nextflow work directory")
    tasks, files, copies = [], [], []
    for process, suffixes in TASK_FILES.items():
        expression = re.compile(r"(?:^|:)" + process + rf" \(chr{chrom}\)$")
        matches = [row for row in rows if expression.search(row.get("name", ""))
                   and row.get("status") in {"COMPLETED", "CACHED"}]
        if len(matches) != 1:
            raise ValueError("Expected exactly one successful trace task for " + process)
        row = matches[0]
        task_hash = row.get("hash", "")
        match = re.fullmatch(r"([0-9a-f]{2})/([0-9a-f]{6,30})", task_hash)
        if not match:
            raise ValueError("Invalid Nextflow task hash: " + task_hash)
        parent = work / match[1]
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("Missing or symlinked task-hash directory")
        candidates = [p for p in parent.iterdir() if p.name.startswith(match[2])
                      and re.fullmatch(r"[0-9a-f]{30}", p.name)]
        if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_dir():
            raise ValueError("Missing or ambiguous Nextflow work directory for " + task_hash)
        task = candidates[0]
        if read_small_file(task / ".exitcode").strip() != b"0":
            raise ValueError("Trace task lacks a successful exit code: " + task_hash)
        full_hash = match[1] + task.name
        tasks.append({"process": process, "trace_status": row["status"],
                      "trace_hash": task_hash, "work_hash": full_hash, "work_directory": str(task)})
        for suffix in suffixes:
            name = f"dnabr.hg38.2723.chr{chrom}.{suffix}"
            source = task / name
            data = read_small_file(source)
            relative = f"{full_hash}/{name}"
            files.append({"source": str(source), "relative_path": relative,
                          "bytes": len(data), "sha256": digest(data), "process": process,
                          "work_hash": full_hash})
            copies.append((relative, data))
    # Validate all inputs before creating any output. No genotype, cache, shell
    # command, or incidental work-directory file is eligible for publication.
    audit = folder / "preprocess" / "audit"
    for directory in (audit.parent, audit):
        if directory.is_symlink():
            raise ValueError("Audit directory must not be a symlink")
        directory.mkdir(exist_ok=True)
    for relative, data in copies:
        target = audit / relative
        if target.parent.is_symlink():
            raise ValueError("Audit task directory must not be a symlink")
        target.parent.mkdir(exist_ok=True)
        write_exclusive(target, data)
    manifest = {"schema_version": 1, "chromosome": chrom, "trace_source": str(trace_path),
                "trace_sha256": digest(trace_data), "tasks": tasks, "files": files,
                "scope": "Successful task provenance and storage checks; not biological validation"}
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    # A resumed trace can say CACHED instead of COMPLETED. Preserve both
    # observations in distinct content-addressed manifests, never overwrite.
    manifest_path = audit / ("manifest-" + digest(encoded) + ".json")
    write_exclusive(manifest_path, encoded)
    return {"status": "PREPROCESS_AUDIT_COLLECTED", "manifest": str(manifest_path),
            "manifest_sha256": digest(encoded), "files": len(files), "chromosome": chrom}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--chrom", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(collect_preprocess_audit(args.folder, args.chrom), sort_keys=True))


if __name__ == "__main__":
    main()
