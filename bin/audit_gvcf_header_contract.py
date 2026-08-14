#!/usr/bin/env python3
"""Audit aggregate gVCF header requirements without reading genotype records."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path.cwd()))

from m27c_gvcf_core import parse_header_contract  # noqa: E402


REQUIRED_FORMAT_FIELDS = {"GT", "DP", "GQ", "MIN_DP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvcfs", nargs="+", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--expected-samples", required=True, type=int)
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_header(bcftools: str, path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [bcftools, "view", "-h", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_header_contract(completed.stdout)


def summarize(headers: list[dict[str, object]], expected_samples: int) -> dict[str, object]:
    missing_patterns = Counter(
        ",".join(sorted(REQUIRED_FORMAT_FIELDS - set(header["format_ids"]))) or "none"
        for header in headers
    )
    source_counts = Counter(str(header["source"]) for header in headers)
    length_counts = Counter(str(header["chr22_length"]) for header in headers)
    checks = {
        "sample_count": sum(len(header["samples"]) == 1 for header in headers),
        "chr22_length": sum(header["chr22_length"] == 50818468 for header in headers),
        "required_format_fields": sum(bool(header["has_required_fields"]) for header in headers),
        "haplotypecaller_source": sum(
            "HaplotypeCaller" in str(header["source"]) for header in headers
        ),
    }
    passed = len(headers) == expected_samples and all(
        count == expected_samples for count in checks.values()
    )
    return {
        "stage": "M27C_GVCF_HEADER_CONTRACT",
        "expected_samples": expected_samples,
        "n_headers": len(headers),
        "pass_counts": checks,
        "source_counts": dict(sorted(source_counts.items())),
        "chr22_length_counts": dict(sorted(length_counts.items())),
        "missing_format_patterns": dict(sorted(missing_patterns.items())),
        "header_contract_pass": passed,
        "sample_ids_emitted": False,
    }


def main() -> int:
    args = parse_args()
    if args.expected_samples < 1 or args.readers < 1:
        raise SystemExit("expected-samples and readers must be positive")
    with ThreadPoolExecutor(max_workers=args.readers) as pool:
        headers = list(pool.map(lambda path: read_header(args.bcftools, path), args.gvcfs))
    result = summarize(headers, args.expected_samples)
    result["input_manifest_sha256"] = hashlib.sha256(args.input_manifest.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
