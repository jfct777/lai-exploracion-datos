#!/usr/bin/env python3
"""Materialize ordered context for authenticated development roles, without truth.

TRAIN and SELECT are separated before retrieval. SCORE preparation is deliberately
absent from this entry point. All REF people and selected loci are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from m33_safe_bridge_core import write_exclusive_json
from m39_carrier_context import extract_inputs, load_config, materialize_radius, require, sha256_file
from m39_ordered_context import OrderedContextStore, build_factorized_store, canonicalize_reference_homologs
from m39_profile_ordered import CODE_FILES, load_profile, subset_queries, training_indices


def role_indices(path: Path, keys: np.ndarray, expected_sha256: str, fold: int,
                 counts: dict, role: str) -> np.ndarray:
    require(role in ("TRAIN", "SELECT"), "only development roles can be materialized")
    train = training_indices(path, keys, expected_sha256, fold, counts)
    if role == "TRAIN":
        return train
    with np.load(path, allow_pickle=False) as z:
        lookup = {bytes(key): i for i, key in enumerate(z["sample_key_sha256"])}
        joined = z["roles"][fold, [lookup[bytes(key)] for key in keys]]
    selected = np.flatnonzero(joined == role)
    return selected[np.argsort(keys[selected], kind="stable")]


def materialize(bridge_path: Path, profile_path: Path, input_dir: Path,
                roles: list[str], outdir: Path) -> dict:
    started = time.monotonic()
    require(roles and len(set(roles)) == len(roles)
            and set(roles) <= {"TRAIN", "SELECT"}, "invalid development role inventory")
    require(not outdir.exists() and not outdir.is_symlink(), "output already exists")
    profile = load_profile(profile_path)
    bridge, paths = load_config(bridge_path, input_dir)
    folds = input_dir / profile["folds"]["path"]
    source = {name: spec["sha256"] for name, spec in bridge["inputs"].items()}
    source.update(folds=profile["folds"]["sha256"], bridge_config=sha256_file(bridge_path),
                  profile_config=sha256_file(profile_path))
    code = {name: sha256_file(Path(__file__).with_name(name))
            for name in (*CODE_FILES, Path(__file__).name)}
    data = extract_inputs(paths, bridge)
    indices = {role: role_indices(folds, data["sample_key_sha256"], source["folds"],
                                profile["fold"], profile["role_counts"], role) for role in roles}
    # The same reference lane convention is used in the original TRAIN stores.
    data = canonicalize_reference_homologs(data)
    anchors = np.arange(len(data["selected"]["locus_id"]), dtype=np.int64)
    outdir.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = []
    for role in roles:
        query = subset_queries(data, indices[role])
        for radius in profile["radii_cm"]:
            case_started = time.monotonic()
            candidates, stats = materialize_radius(query, anchors, profile["K"], float(radius))
            store = build_factorized_store(query, candidates)
            path = outdir / role.lower() / f"radius_{radius:g}cm"
            saved = store.save(path, source_hashes=source)
            reopened = OrderedContextStore.open(path, expected_manifest_sha256=saved["manifest_sha256"])
            require(np.array_equal(reopened.arrays["sample_key_sha256"],
                                   data["sample_key_sha256"][indices[role]]), "role keys changed")
            rows.append({"role": role, "radius_cm": radius, "people": len(indices[role]),
                         "anchors": len(anchors), "relative_path": str(path.relative_to(outdir)),
                         **saved, "retrieval_stats": stats,
                         "elapsed_seconds": time.monotonic() - case_started})
            print(json.dumps({"materialized": role, "radius_cm": radius}), flush=True)
    require(all(sha256_file(paths[k]) == spec["sha256"] for k, spec in bridge["inputs"].items())
            and sha256_file(folds) == source["folds"], "original input changed")
    require(sha256_file(bridge_path) == source["bridge_config"]
            and sha256_file(profile_path) == source["profile_config"]
            and all(sha256_file(Path(__file__).with_name(k)) == v for k, v in code.items()),
            "configuration or source changed during materialization")
    result = {"schema_version": "m39-ordered-development-store-v1",
              "decision": "PASS_DEVELOPMENT_FEATURES_ONLY", "profiles": rows,
              "source_sha256": source, "code_sha256": code,
              "elapsed_seconds": time.monotonic() - started,
              "boundaries": {"truth_opened": False, "predictions_opened": False,
                             "feature_computation_roles": roles, "SCORE_features_created": False,
                             "source_TEST_opened": False, "model_training": False,
                             "rare_phase_assigned": False}}
    write_exclusive_json(outdir / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=("TRAIN", "SELECT"), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.bridge_config, args.profile_config, args.input_dir, args.roles, args.outdir)


if __name__ == "__main__":
    main()
