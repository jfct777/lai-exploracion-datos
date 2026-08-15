#!/usr/bin/env python3
"""Verifica de forma fail-closed los recursos persistentes de M27D."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_STAGE = "M27D_DONOR_KINSHIP_MARKER_PREPARATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prepared_inputs(
    manifest_path: Path,
    prepared_paths: list[Path],
    expected_manifest_sha256: str,
) -> dict:
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Preparation manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {observed_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != EXPECTED_STAGE:
        raise ValueError(
            f"Unexpected preparation stage: {manifest.get('stage')!r}"
        )
    params = manifest.get("params")
    if not isinstance(params, dict):
        raise ValueError("Preparation manifest is missing params")
    if params.get("scope") != "m27d_marker_preparation":
        raise ValueError("Preparation manifest has an unexpected scope")
    if params.get("scientific_result") is not False:
        raise ValueError("Preparation manifest does not declare itself non-scientific")
    # The legacy run-level flag was removed from stage manifests: authorization belongs to
    # run_provenance.json, and a per-stage copy is a second place for it to drift. Old
    # manifests still carry it, so it is rejected only when it claims an authorized run.
    if params.get("full_run_authorized") not in (None, False):
        raise ValueError("Preparation manifest claims an authorized full run")

    expected_hashes = manifest.get("sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Preparation manifest is missing sha256 checksums")

    observed = {}
    for path in prepared_paths:
        if not path.is_file():
            raise ValueError(f"Prepared input is missing: {path}")
        name = path.name
        if name not in expected_hashes:
            raise ValueError(f"Prepared input {name!r} is absent from the manifest")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hashes[name]:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected_hashes[name]}, "
                f"observed {observed_hash}"
            )
        observed[name] = {
            "bytes": path.stat().st_size,
            "sha256": observed_hash,
        }

    return {
        "stage": "M27D_PREPARED_INPUT_VERIFICATION",
        "verified": True,
        "scientific_result": False,
        "king_executed": False,
        "preparation_stage": manifest["stage"],
        "preparation_manifest_sha256": observed_manifest_sha256,
        "verified_files": observed,
        "sample_ids_emitted": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--gds", required=True, type=Path)
    parser.add_argument("--anchor-rds", required=True, type=Path)
    parser.add_argument("--strict-rds", required=True, type=Path)
    # Optional because the donor audit resolves its own strata table instead of reusing
    # the one frozen with the preparation: the resolution policy was corrected after that
    # run, so the audit must not verify itself against the superseded file.  The genotype
    # resources it does consume are still checked.
    parser.add_argument("--metadata-strata", type=Path, default=None)
    # A phase that reuses a training set produced by an earlier run must prove it is the
    # reviewed one. Naming the file is not proof: the path can be repointed at another
    # run's output and every downstream number would still look self-consistent.
    parser.add_argument("--training-set", type=Path, default=None)
    parser.add_argument("--expected-training-set-sha256", default=None)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = [args.gds, args.anchor_rds, args.strict_rds]
    if args.metadata_strata is not None:
        prepared.append(args.metadata_strata)
    result = verify_prepared_inputs(
        args.manifest,
        prepared,
        args.expected_manifest_sha256,
    )
    if (args.training_set is None) != (args.expected_training_set_sha256 is None):
        raise SystemExit("--training-set and --expected-training-set-sha256 go together")
    if args.training_set is not None:
        observed = sha256_file(args.training_set)
        if observed != args.expected_training_set_sha256:
            raise ValueError(
                f"Training set SHA-256 mismatch: expected "
                f"{args.expected_training_set_sha256}, observed {observed}"
            )
        result["reused_training_set"] = {
            "bytes": args.training_set.stat().st_size,
            "sha256": observed,
            "n_samples": sum(
                1
                for line in args.training_set.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
        }
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
