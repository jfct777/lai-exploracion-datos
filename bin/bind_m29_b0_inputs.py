#!/usr/bin/env python3
"""Create an immutable M29 binding for authenticated root-specific B0 output."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-seed", type=int, required=True, choices=[20260817, 20260818])
    parser.add_argument("--fb", type=Path, required=True)
    parser.add_argument("--msp", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.fb, args.msp, args.provenance_manifest):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    provenance = json.loads(args.provenance_manifest.read_text(encoding="utf-8"))
    provenance_text = json.dumps(provenance, sort_keys=True)
    if str(args.root_seed) not in provenance_text or "M28" not in provenance_text.upper():
        raise SystemExit("provenance manifest does not identify the requested M28-v2 root")
    payload = {
        "stage": "M29_AUTHENTICATED_B0_BINDING",
        "root_seed": args.root_seed,
        "sha256": {"fb": sha256(args.fb), "msp": sha256(args.msp)},
        "provenance_manifest_sha256": sha256(args.provenance_manifest),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
