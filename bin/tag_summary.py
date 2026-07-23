#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chr", required=True)
    p.add_argument("--prune_in", required=True)
    p.add_argument("--prune_in_strict", required=True)
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def _count_lines(path: str) -> int:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    n = 0
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def main():
    args = parse_args()

    summary = {
        "chr": args.chr,
        "n_tag_snps_r2": _count_lines(args.prune_in),
        "n_tag_snps_r2_strict": _count_lines(args.prune_in_strict),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
