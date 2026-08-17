#!/usr/bin/env python3
"""Verify byte reproducibility of the scientific M28B outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCIENTIFIC_FILES = (
    "m28b_capacity.public.json",
    "m28b_capacity_screens.tsv",
    "m28b_B0.tsv.gz",
    "m28b_BR_additions.tsv.gz",
    "m28b_BS_additions.tsv.gz",
    "m28b_B0_mapping.tsv.gz",
    "m28b_BR_BS_pairs.tsv.gz",
)

SCIENTIFIC_FILES_V2 = (
    "m28b_v2_capacity.public.json",
    "m28b_v2_capacity_screens.tsv",
    "m28b_v2_B0.tsv.gz",
    "m28b_v2_BR_additions.tsv.gz",
    "m28b_v2_BS_additions.tsv.gz",
    "m28b_v2_B0_mapping.tsv.gz",
    "m28b_v2_BR_BS_pairs.tsv.gz",
)

SCIENTIFIC_FILES_V3 = (
    "m28b_v3_capacity.public.json",
    "m28b_v3_capacity_screens.tsv",
    "m28b_v3_B0.tsv.gz",
    "m28b_v3_BR_additions.tsv.gz",
    "m28b_v3_BS_additions.tsv.gz",
    "m28b_v3_BR_BS_pairs.tsv.gz",
    "m28b_v3_common_common_null.tsv",
)

PROFILE_FILES = {
    "v1": SCIENTIFIC_FILES,
    "v2": SCIENTIFIC_FILES_V2,
    "v3": SCIENTIFIC_FILES_V3,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate(root: Path, name: str) -> Path:
    candidates = [root / name] + [root / stage / name for stage in ("m28b", "m28b_v2", "m28b_v3")]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def verify(run1: Path, run2: Path, filenames=SCIENTIFIC_FILES) -> dict:
    files = {}
    for name in filenames:
        left = locate(run1, name)
        right = locate(run2, name)
        left_hash = sha256(left)
        right_hash = sha256(right)
        files[name] = {
            "run1_sha256": left_hash,
            "run2_sha256": right_hash,
            "byte_identical": left_hash == right_hash,
        }
    passed = all(row["byte_identical"] for row in files.values())
    return {
        "stage": "M28B_REPRODUCIBILITY_AUDIT",
        "files": files,
        "gate": "PASS" if passed else "FAIL",
        "decision": "GO_PREREGISTER_LAI_COMPARATOR" if passed else "STOP_REPRODUCIBILITY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run1", required=True, type=Path)
    parser.add_argument("--run2", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILE_FILES), default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filenames = PROFILE_FILES[args.profile]
    report = verify(args.run1, args.run2, filenames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "gate": report["gate"]}, sort_keys=True))
    if report["gate"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
