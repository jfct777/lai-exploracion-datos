#!/usr/bin/env python3
"""Authenticate the small frozen code tree used by FULL22 store restoration."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root, manifest_path, manifest_sha256):
    root = Path(root).resolve(); manifest_path = Path(manifest_path).resolve()
    if sha256(manifest_path) != manifest_sha256:
        raise ValueError("frozen restore code manifest differs")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (value.get("schema") != "r02_structure_restore_code_manifest_v1" or
            value.get("status") != "FROZEN_EXACT_V3_AND_HELPERS_NO_ASSEMBLY_NO_TRAINING"):
        raise ValueError("frozen restore code manifest state differs")
    for row in value.get("files", []):
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError("frozen source missing or unsafe: " + row["path"])
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("frozen source differs: " + row["path"])
    if len(value.get("files", [])) != 7:
        raise ValueError("exact seven-file frozen restore source set required")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    result = verify(args.root, args.manifest, args.manifest_sha256)
    print(json.dumps({"status": result["status"], "files": len(result["files"])}, sort_keys=True))
