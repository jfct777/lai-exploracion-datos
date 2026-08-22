#!/usr/bin/env python3
"""Verify that each pinned Tabix index reproduces its complete BGZF VCF."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def line_digest(lines) -> tuple[int, str]:
    count, digest = 0, hashlib.sha256()
    for line in lines:
        if isinstance(line, str):
            line = line.encode()
        digest.update(line.rstrip(b"\r\n"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def audit_pair(vcf: Path, tbi: Path) -> dict:
    with vcf.open("rb") as handle:
        handle.seek(-len(BGZF_EOF), os.SEEK_END)
        require(handle.read() == BGZF_EOF, "VCF lacks canonical BGZF EOF block")
    require(subprocess.run(["tabix", "-l", str(vcf)], check=True, capture_output=True, text=True).stdout.split() == ["22"],
            "Tabix contig inventory drifted")
    header = subprocess.run(["tabix", "-H", str(vcf)], check=True, capture_output=True).stdout
    require(b"##fileformat=VCFv4.2\n" in header and b"#CHROM\t" in header,
            "Tabix header fetch failed")
    indexed = subprocess.run(["tabix", str(vcf), "22"], check=True, capture_output=True).stdout.splitlines()
    with gzip.open(vcf, "rb") as handle:
        sequential = [line for line in handle if not line.startswith(b"#")]
    indexed_count, indexed_sha = line_digest(indexed)
    sequential_count, sequential_sha = line_digest(sequential)
    require(indexed_count == sequential_count and indexed_sha == sequential_sha,
            "Tabix index does not reproduce the complete VCF record stream")
    with gzip.open(tbi, "rb") as handle:
        require(handle.read(4) == b"TBI\x01", "invalid TBI magic")
    return {
        "vcf_sha256": sha256_file(vcf), "tbi_sha256": sha256_file(tbi),
        "record_count": indexed_count, "record_sha256": indexed_sha,
    }


def write_exclusive(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); temporary.unlink()
    finally:
        if temporary.exists(): temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ref_vcf", "ref_tbi", "target_vcf", "target_tbi"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True, type=Path)
    parser.add_argument("--source-auth", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.source_auth.read_text(encoding="utf-8"))
    require(source.get("stage") == "M33_A0_SOURCE_AUTH" and source.get("git_commit") == args.git_commit,
            "source-auth identity drifted")
    version = subprocess.run(["tabix", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    require(version == "tabix (htslib) 1.16", "Tabix/htslib version drifted")
    write_exclusive(args.output, {
        "stage": "M33_A0_TABIX_AUDIT",
        "status": "PASS_BGZF_AND_INDEX_FULL_STREAM_PARITY",
        "git_commit": args.git_commit,
        "expected_container_image_id": args.expected_image_id,
        "tabix_version": version,
        "source_auth_sha256": sha256_file(args.source_auth),
        "ref": audit_pair(args.ref_vcf, args.ref_tbi),
        "target": audit_pair(args.target_vcf, args.target_tbi),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
