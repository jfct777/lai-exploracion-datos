#!/usr/bin/env python3
"""Map already staged K04 sources; downstream admission rechecks every SHA256."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--supplement-archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    if sha256(request_path) != args.request_sha256:
        raise ValueError("Request SHA256 differs")
    request = json.loads(request_path.read_text())
    if request.get("schema") != "r02_k04_common_pair_repair_request_v2":
        raise ValueError("Wrong request schema")
    root = Path(args.work_root).resolve()
    allowed = Path("/opt/r02-integrated-20261008ah/k04-pairs").resolve()
    if root != allowed / "work":
        raise ValueError("Work root differs from the frozen K04 recovery root")
    supplement = Path(args.supplement_archive).resolve()
    if allowed not in supplement.parents or not supplement.is_file() or \
            supplement.stat().st_size != request["supplement_archive"]["bytes"]:
        raise ValueError("Supplement cache path/size differs")
    chromosomes = {}
    names = ("event.tar.gz", "partial.npz", "partial-manifest.json", "common_markers.tsv")
    keys = ("event_archive", "partial", "partial_manifest", "common_markers")
    for value in range(1, 23):
        chrom = str(value); label = f"chr{value:02d}"; desc = request["chromosomes"][chrom]
        candidates = []
        for event in root.glob(f"*/*/{label}/event.tar.gz"):
            directory = event.parent
            paths = [directory / name for name in names]
            if all(path.is_file() and path.stat().st_size == desc[key]["bytes"]
                   for path, key in zip(paths, keys)):
                candidates.append(paths)
        if len(candidates) != 1:
            raise ValueError(f"Expected one size-matched staged source tuple for {label}; got {len(candidates)}")
        chromosomes[chrom] = {key: str(path.resolve()) for key, path in zip(keys, candidates[0])}
    output = Path(args.output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Fresh cache-manifest output in an existing directory required")
    result = {"schema": "r02_k04_staged_source_cache_v1",
              "request_sha256": args.request_sha256,
              "supplement_archive": str(supplement), "chromosomes": chromosomes,
              "content_authentication": "ADMIT_PROCESSES_AND_PRODUCER_RECHECK_REQUEST_SHA256"}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("K04_STAGED_SOURCE_CACHE_MAPPED_22_CHROMOSOMES_PENDING_SHA_ADMISSION")


if __name__ == "__main__":
    main()
