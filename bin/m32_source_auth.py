#!/usr/bin/env python3
"""Authenticate the complete M32 packed implementation against one Git commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


STAGE = "M32_PACKED_SOURCE_AUTH"
REQUIRED_SOURCES = {
    "bin/m32_source_auth.py",
    "bin/m32_packed_benchmark.py",
    "bin/m31_ordered_linear.py",
    "bin/m32_locus_contract.py",
    "bin/m32_locus_smoke.py",
    "conf/m32_packed_benchmark_preregistration.json",
    "conf/m32_packed_benchmark.config",
    "modules/32_PACKED_BENCHMARK.nf",
    "workflows/m32_packed_benchmark.nf",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        relative, separator, staged = value.partition("=")
        require(bool(separator) and bool(relative), "invalid source specification")
        require(not relative.startswith("/") and ".." not in Path(relative).parts, "unsafe source path")
        require(relative not in sources, "duplicate source specification")
        sources[relative] = Path(staged)
    require(set(sources) == REQUIRED_SOURCES, "source set does not cover the complete M32 implementation")
    return sources


def write_json_exclusive(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[0-9a-f]{40}", args.git_commit)), "git commit must be exact")
    sources = parse_sources(args.source)
    head = subprocess.run(
        ["git", "-C", str(args.repository_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(head == args.git_commit, "Git HEAD differs from requested commit")
    status = subprocess.run(
        ["git", "-C", str(args.repository_root), "status", "--porcelain", "--", *sorted(sources)],
        check=True, capture_output=True, text=True,
    ).stdout
    require(not status.strip(), "authenticated sources are dirty or untracked")
    hashes: dict[str, str] = {}
    for relative, staged in sorted(sources.items()):
        require(staged.is_file(), f"missing staged source: {relative}")
        committed = subprocess.run(
            ["git", "-C", str(args.repository_root), "show", f"{args.git_commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        staged_hash = sha256_file(staged)
        require(staged_hash == sha256_bytes(committed), f"staged source differs from commit: {relative}")
        hashes[relative] = staged_hash
    write_json_exclusive(args.output, {
        "stage": STAGE,
        "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
        "git_commit": args.git_commit,
        "source_sha256": hashes,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
