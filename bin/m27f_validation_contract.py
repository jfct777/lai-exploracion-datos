#!/usr/bin/env python3
"""Build the immutable code-and-environment plan for one M27F VALID analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_plan(
    scripts: dict[str, Path],
    container_image: str,
    container_digest: str,
    preregistration: dict[str, object],
) -> tuple[dict[str, object], str]:
    support = preregistration["support_contract"]
    diagnostics = preregistration["diagnostics"]
    plan = {
        "scripts_sha256": {
            role: sha256_file(path) for role, path in sorted(scripts.items())
        },
        "container_image": container_image,
        "container_digest": container_digest,
        "primary_ref_min_atomic_units": int(
            support["primary_ref_min_atomic_units"]
        ),
        "required_valid_atomic_units": int(support["required_valid_atomic_units"]),
        "diagnostic_ref_thresholds": [
            int(value) for value in diagnostics["ref_support_thresholds"]
        ],
        "missingness_bound": "same_site_ref_and_valid_atomic_unit_bounds",
        "baseline_comparison": "exact_CHROM_POS_REF_ALT_key",
    }
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return plan, hashlib.sha256(payload).hexdigest()
