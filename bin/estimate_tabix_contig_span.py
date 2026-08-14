#!/usr/bin/env python3
"""Estimate compressed byte spans for one contig from standard .tbi indexes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
from pathlib import Path


def read_i32(handle) -> int:
    value = handle.read(4)
    if len(value) != 4:
        raise ValueError("Unexpected end of TBI index")
    return struct.unpack("<i", value)[0]


def read_u32(handle) -> int:
    value = handle.read(4)
    if len(value) != 4:
        raise ValueError("Unexpected end of TBI index")
    return struct.unpack("<I", value)[0]


def read_u64(handle) -> int:
    value = handle.read(8)
    if len(value) != 8:
        raise ValueError("Unexpected end of TBI index")
    return struct.unpack("<Q", value)[0]


def contig_span(path: Path, requested: str) -> int:
    with gzip.open(path, "rb") as handle:
        if handle.read(4) != b"TBI\x01":
            raise ValueError(f"Not a TBI index: {path}")
        n_ref = read_i32(handle)
        for _ in range(6):
            read_i32(handle)
        names_length = read_i32(handle)
        names = handle.read(names_length).rstrip(b"\x00").decode("utf-8").split("\x00")
        if len(names) != n_ref:
            raise ValueError(f"Reference-name count mismatch: {path}")
        target_index = names.index(requested)
        target_offsets: list[int] = []
        for ref_index in range(n_ref):
            n_bin = read_i32(handle)
            for _ in range(n_bin):
                bin_id = read_u32(handle)
                n_chunk = read_i32(handle)
                for _ in range(n_chunk):
                    begin = read_u64(handle)
                    end = read_u64(handle)
                    if ref_index == target_index and bin_id < 37450:
                        target_offsets.extend((begin >> 16, end >> 16))
            n_interval = read_i32(handle)
            for _ in range(n_interval):
                read_u64(handle)
        if not target_offsets:
            raise ValueError(f"No chunks for {requested}: {path}")
        return max(target_offsets) - min(target_offsets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexes", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mount-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contig", default="chr22")
    args = parser.parse_args()
    indexes = list(args.indexes or [])
    if args.manifest:
        if not args.mount_root:
            raise SystemExit("--mount-root is required with --manifest")
        with args.manifest.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                uri = row["uri"]
                if not uri.endswith(".tbi"):
                    continue
                relative = uri.split("/haplotypecaller/", 1)[1]
                indexes.append(args.mount_root / relative)
                if args.limit is not None and len(indexes) >= args.limit:
                    break
    if not indexes:
        raise SystemExit("Provide --indexes or --manifest with --mount-root")
    spans = [contig_span(path, args.contig) for path in indexes]
    result = {
        "contig": args.contig,
        "n_indexes": len(spans),
        "compressed_span_bytes_sum": sum(spans),
        "compressed_span_gib_sum": sum(spans) / (1024 ** 3),
        "compressed_span_bytes_min": min(spans),
        "compressed_span_bytes_median": sorted(spans)[len(spans) // 2],
        "compressed_span_bytes_max": max(spans),
        "interpretation": "Upper span from first to last indexed BGZF chunk; actual range reads can be lower and request charges are separate.",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
