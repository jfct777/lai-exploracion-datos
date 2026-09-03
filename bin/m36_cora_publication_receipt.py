#!/usr/bin/env python3
"""Bind a hash-checked M36 materialization to immutable published objects.

This is intentionally post-publication: it cannot write cloud objects.  A
publisher supplies the observed URI/generation/SHA-256 descriptors after the
factorized artifacts were copied to the project-owned bucket.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {"loci", "carriers", "missing", "covariates", "components", "targets"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--published-descriptors", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.materialization_receipt.read_text())
    published = json.loads(args.published_descriptors.read_text())
    if receipt.get("stage") != "M36_CORA_MATERIALIZE" or receipt.get("status") != "MATERIALIZED_PASS":
        raise SystemExit("M36 publication receipt error: materialization receipt is not chainable")
    if set(published) != EXPECTED:
        raise SystemExit("M36 publication receipt error: descriptors must bind exactly factorized artifacts")
    local = receipt.get("input_descriptors", {})
    for name in EXPECTED:
        descriptor = published[name]
        if not isinstance(descriptor, dict) or not all(isinstance(descriptor.get(field), str) and descriptor[field]
                                                       for field in ("uri", "generation", "sha256")):
            raise SystemExit(f"M36 publication receipt error: incomplete descriptor {name}")
        if descriptor["sha256"] != local.get(name, {}).get("sha256"):
            raise SystemExit(f"M36 publication receipt error: published hash differs for {name}")
    receipt["status"] = "PUBLISHED_PASS"
    receipt["input_descriptors"] = published
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
