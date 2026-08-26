#!/usr/bin/env python3
"""Recompress one VCF as BGZF, build Tabix index, and emit an audit receipt."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists() or Path(str(args.output) + ".tbi").exists():
        raise ValueError("refusing to overwrite BGZF or index")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.input, "rb") as source, args.output.open("xb") as target:
        process = subprocess.Popen([args.bgzip, "-c"], stdin=subprocess.PIPE, stdout=target)
        assert process.stdin is not None
        try:
            shutil.copyfileobj(source, process.stdin, length=1 << 20)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("bgzip failed")
    subprocess.run([args.tabix, "-p", "vcf", str(args.output)], check=True)
    index = Path(str(args.output) + ".tbi")
    sequential = subprocess.run(
        [args.tabix, str(args.output), "22"], check=True, text=True, capture_output=True
    ).stdout
    count = sum(1 for line in sequential.splitlines() if line and not line.startswith("#"))
    if count != args.expected_records:
        raise ValueError(f"indexed record count differs: {count} != {args.expected_records}")
    receipt = {
        "schema_version": "1.0.0",
        "stage": "M33_BGZIP_TABIX",
        "status": "PASS",
        "source_sha256": sha256(args.input),
        "bgzf_sha256": sha256(args.output),
        "tbi_sha256": sha256(index),
        "records_chr22": count,
        "tabix_version": subprocess.run(
            [args.tabix, "--version"], check=True, text=True, capture_output=True
        ).stdout.splitlines()[0],
    }
    receipt_path = args.output.with_name(args.output.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--bgzip", default="bgzip")
    parser.add_argument("--tabix", default="tabix")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "records": result["records_chr22"]}, sort_keys=True))
