#!/usr/bin/env python3
"""Authenticate the exact committed sources for M33 factorized/lazy M0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


REQUIRED_SOURCES = {
    "bin/m33_materialize.py",
    "bin/m33_m0_contract.py",
    "bin/m31_ordered_linear.py",
    "bin/m33_m0_factorized_lazy_amendment.py",
    "bin/m33_m0_factorized_lazy_source_auth.py",
    "bin/m33_m0_factorized_lazy_technical_kat.py",
    "conf/m33_m0_factorized_lazy_amendment_contract.json",
    "conf/m33_m0_factorized_lazy_contract.config",
    "modules/33_M0_FACTORIZED_LAZY_CONTRACT.nf",
    "modules/33_M0_FACTORIZED_LAZY_TECHNICAL_KAT.nf",
    "workflows/m33_m0_factorized_lazy_contract.nf",
    "workflows/m33_m0_factorized_lazy_technical_kat.nf",
    "tests/test_m33_materialize.py",
    "tests/test_m33_m0_factorized_lazy_amendment.py",
    "tests/test_m33_m0_factorized_lazy_nextflow.py",
    "tests/test_m33_m0_factorized_lazy_source_auth.py",
    "tests/test_m33_m0_factorized_lazy_technical_kat.py",
    "conf/m33_m0_factorized_lazy_technical_kat.config",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is not None,
            "git commit must be exact")
    sources: dict[str, Path] = {}
    for item in args.source:
        relative, separator, staged = item.partition("=")
        require(bool(separator) and relative and not relative.startswith("/"),
                "invalid source specification")
        require(".." not in Path(relative).parts and relative not in sources,
                "unsafe or duplicate source")
        sources[relative] = Path(staged)
    require(set(sources) == REQUIRED_SOURCES, "factorized/lazy source inventory is incomplete")
    head = subprocess.run(
        ["git", "-C", str(args.repository_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(head == args.git_commit, "Git HEAD differs from requested commit")
    dirty = subprocess.run(
        ["git", "-C", str(args.repository_root), "status", "--porcelain", "--", *sorted(sources)],
        check=True, capture_output=True, text=True,
    ).stdout
    require(not dirty.strip(), "factorized/lazy sources are dirty or untracked")
    hashes: dict[str, str] = {}
    for relative, staged in sorted(sources.items()):
        committed = subprocess.run(
            ["git", "-C", str(args.repository_root), "show", f"{args.git_commit}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        observed = sha256_file(staged)
        require(observed == hashlib.sha256(committed).hexdigest(),
                f"source differs from commit: {relative}")
        hashes[relative] = observed
    write_exclusive(args.output, {
        "stage": "M33_M0_FACTORIZED_LAZY_SOURCE_AUTH",
        "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
        "git_commit": args.git_commit,
        "source_sha256": hashes,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
