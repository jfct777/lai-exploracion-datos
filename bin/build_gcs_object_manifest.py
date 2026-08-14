#!/usr/bin/env python3
"""Build a deterministic Cloud Storage object manifest without downloading data."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--suffix", action="append", required=True)
    parser.add_argument("--expected-per-suffix", type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    completed = subprocess.run(
        ["gcloud", "storage", "ls", "--json", args.pattern],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    objects = json.loads(completed.stdout)
    rows = []
    counts = {suffix: 0 for suffix in args.suffix}
    for item in objects:
        metadata = item.get("metadata", {})
        name = metadata.get("name", "")
        matched = [suffix for suffix in args.suffix if name.endswith(suffix)]
        if not matched:
            continue
        if len(matched) != 1:
            raise SystemExit(f"Ambiguous suffix match: {name}")
        suffix = matched[0]
        counts[suffix] += 1
        rows.append(
            {
                "uri": f"gs://{metadata['bucket']}/{name}",
                "generation": metadata.get("generation", ""),
                "size_bytes": metadata.get("size", ""),
                "crc32c": metadata.get("crc32c", ""),
                "md5_hash": metadata.get("md5Hash", ""),
                "storage_class": metadata.get("storageClass", ""),
                "updated": metadata.get("updated", ""),
            }
        )
    if args.expected_per_suffix is not None:
        unexpected = {suffix: count for suffix, count in counts.items() if count != args.expected_per_suffix}
        if unexpected:
            raise SystemExit(f"Unexpected object counts: {unexpected}")
    rows.sort(key=lambda row: row["uri"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"counts": counts, "n_rows": len(rows), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
