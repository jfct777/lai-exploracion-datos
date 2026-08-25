#!/usr/bin/env python3
"""Authenticate the exact committed sources used by the M33 REF-label sham KAT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping


REQUIRED_SOURCES = {
    "bin/m33_safe_bridge_core.py",
    "bin/m33_ref_label_sham_kat.py",
    "bin/m33_ref_label_sham_source_auth.py",
    "conf/m33_ref_label_sham_contract.json",
    "conf/m33_ref_label_sham.config",
    "conf/m33_pre4_preregistration.json",
    "conf/m33_m0_materializer_contract.json",
    "modules/33_REF_LABEL_SHAM_KAT.nf",
    "workflows/m33_ref_label_sham.nf",
    "tests/test_m33_ref_label_sham.py",
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


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "source-auth JSON must be an object")
    return value


def validate_source_auth(path: Path, commit: str, source_root: Path) -> str:
    payload = load_json(path)
    hashes = payload.get("source_sha256", {})
    require(payload.get("stage") == "M33_REF_LABEL_SHAM_SOURCE_AUTH" and
            payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            payload.get("git_commit") == commit and set(hashes) == REQUIRED_SOURCES and
            all(re.fullmatch(r"[0-9a-f]{64}", value or "") for value in hashes.values()),
            "REF-label sham source authentication differs")
    for relative in REQUIRED_SOURCES:
        candidate = source_root / relative
        require(candidate.is_file() and not candidate.is_symlink() and
                sha256_file(candidate) == hashes[relative],
                f"authenticated REF-label sham source differs: {relative}")
    return sha256_file(path)


def write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    require(not path.exists() and path.parent.is_dir(), "source-auth output must be new")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
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
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.git_commit or "") is not None,
            "git commit must be exact")
    sources: dict[str, Path] = {}
    for item in args.source:
        relative, separator, staged = item.partition("=")
        require(bool(separator) and relative and not relative.startswith("/") and
                ".." not in Path(relative).parts and relative not in sources,
                "invalid source specification")
        sources[relative] = Path(staged)
    require(set(sources) == REQUIRED_SOURCES, "REF-label sham source inventory is incomplete")
    head = subprocess.run(
        ["git", "-C", str(args.repository_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(head == args.git_commit, "Git HEAD differs from requested commit")
    dirty = subprocess.run(
        ["git", "-C", str(args.repository_root), "status", "--porcelain", "--",
         *sorted(sources)], check=True, capture_output=True, text=True,
    ).stdout
    require(not dirty.strip(), "REF-label sham sources are dirty or untracked")
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
        "stage": "M33_REF_LABEL_SHAM_SOURCE_AUTH",
        "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
        "git_commit": args.git_commit,
        "source_sha256": hashes,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
