#!/usr/bin/env python3
"""Authenticate the complete M33 execution-contract source set at one clean commit."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

from m33_asset_manifest_contract import require, sha256_file, write_exclusive


REQUIRED_SOURCES = {
    "bin/m33_asset_execution_contract.py",
    "bin/m33_asset_execution_source_auth.py",
    "bin/m33_asset_manifest_contract.py",
    "conf/m33_asset_execution_amendment.json",
    "conf/m33_asset_execution_contract.config",
    "conf/m33_asset_manifest_contract.json",
    "modules/33_ASSET_EXECUTION_CONTRACT.nf",
    "workflows/m33_asset_execution_contract.nf",
    "tests/test_m33_asset_execution_contract.py",
    "tests/test_m33_asset_execution_nextflow.py",
}


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
    require(set(sources) == REQUIRED_SOURCES, "execution source inventory is incomplete")
    head = subprocess.run(
        ["git", "-C", str(args.repository_root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    require(head == args.git_commit, "Git HEAD differs from requested orchestrator commit")
    dirty = subprocess.run(
        ["git", "-C", str(args.repository_root), "status", "--porcelain", "--", *sorted(sources)],
        check=True, capture_output=True, text=True,
    ).stdout
    require(not dirty.strip(), "authenticated execution sources are dirty or untracked")
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
        "stage": "M33_ASSET_EXECUTION_SOURCE_AUTH",
        "status": "PASS_EXACT_COMMIT_AND_SOURCE_HASHES",
        "git_commit": args.git_commit,
        "source_sha256": hashes,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
