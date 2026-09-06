#!/usr/bin/env python3
"""Profile ordered common context on authenticated inner TRAIN mosaics.

This is an input-engineering experiment, not training or an ancestry benchmark.
The shared historical VCF/NPZ contains 96 mosaics and is parsed for reconciliation;
only keyed TRAIN individuals enter retrieval and output. No truth is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_exclusive_json
from m39_carrier_context import extract_inputs, load_config, materialize_radius, require, sha256_file
from m39_ordered_context import (OrderedContextStore, build_factorized_store,
                                 canonicalize_reference_homologs)

SCHEMA = "m39-ordered-input-profile-v1"
CODE_FILES = ("m39_profile_ordered.py", "m39_ordered_context.py", "m39_carrier_context.py",
              "m34_prepare_panel_factors.py", "m34_generate_mosaics.py", "m33_safe_bridge_core.py")


def training_indices(path: Path, keys: np.ndarray, expected_sha256: str,
                     fold: int, counts: dict) -> np.ndarray:
    """Join all inner roles by sample key, then sort TRAIN by key, never label."""
    require(sha256_file(path) == expected_sha256, "fold hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {"sample_key_sha256", "roles", "outer_fold",
                                 "inner_split_seed", "outer_seed"}, "fold inventory differs")
        source, roles = z["sample_key_sha256"], z["roles"]
        require(source.dtype == keys.dtype == np.dtype("S64") and source.shape == keys.shape
                and keys.ndim == 1, "sample-key schema differs")
        require(len(set(source.tolist())) == len(source) and len(set(keys.tolist())) == len(keys)
                and set(source.tolist()) == set(keys.tolist()), "sample universes differ")
        require(all(len(bytes(k)) == 64 and set(bytes(k)) <= set(b"0123456789abcdef")
                    for k in keys), "sample-key encoding differs")
        require(np.array_equal(z["outer_fold"], np.arange(3)) and roles.shape == (3, len(keys))
                and type(fold) is int and 0 <= fold < 3, "fold dimensions differ")
        require(set(counts) == {"TRAIN", "SELECT", "SCORE"}
                and set(np.unique(roles)) == set(counts), "role labels differ")
        require(all(type(n) is int and n > 0 for n in counts.values())
                and all(all(int(np.sum(row == role)) == n for role, n in counts.items())
                        for row in roles), "role counts differ")
        require(np.all((roles == "SCORE").sum(0) == 1), "outer SCORE partition differs")
        lookup = {key: i for i, key in enumerate(source.tolist())}
        joined = roles[fold, [lookup[key] for key in keys.tolist()]]
    fit = np.flatnonzero(joined == "TRAIN")
    return fit[np.argsort(keys[fit], kind="stable")]


def subset_queries(data: dict, indices: np.ndarray) -> dict:
    """Retain every REF and locus; subset only the TARGET axes."""
    require(indices.ndim == 1 and indices.dtype.kind in "iu"
            and len(set(indices.tolist())) == len(indices)
            and np.all((indices >= 0) & (indices < len(data["sample_key_sha256"]))),
            "invalid query subset")
    out = dict(data)
    for key in ("sample_key_sha256", "query_dosage", "query_observed"):
        out[key] = np.ascontiguousarray(data[key][indices])
    out["common_target"] = np.ascontiguousarray(data["common_target"][:, indices])
    return out


def load_profile(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf8"))
    require(isinstance(p, dict) and set(p) == {"schema_version", "scope", "folds", "fold", "role_counts",
                       "people", "radii_cm", "K", "site_chunk_size", "max_window_bytes"},
            "profile inventory differs")
    require(p["schema_version"] == SCHEMA and p["scope"] == "technical_inner_TRAIN_only",
            "profile scope differs")
    require(type(p["fold"]) is int and p["fold"] == 0, "only historical fold0 TRAIN allowed")
    counts = p["role_counts"]
    require(isinstance(counts, dict) and set(counts) == {"TRAIN", "SELECT", "SCORE"}
            and all(type(n) is int for n in counts.values())
            and counts == {"TRAIN": 48, "SELECT": 16, "SCORE": 32}, "historical role counts differ")
    sizes = p["people"]
    require(isinstance(sizes, list) and sizes and all(type(n) is int and n > 0 for n in sizes)
            and sizes == sorted(set(sizes)) and max(sizes) <= p["role_counts"]["TRAIN"],
            "profile sizes must be increasing nested TRAIN subsets")
    require(type(p["K"]) is int and 1 <= p["K"] <= 8, "technical K outside envelope")
    radii = p["radii_cm"]
    require(isinstance(radii, list) and radii and all(type(r) in (float, int)
            and np.isfinite(r) and 0 < r <= 1 for r in radii)
            and len(set(radii)) == len(radii), "invalid technical radii")
    require(type(p["site_chunk_size"]) is int and 1 <= p["site_chunk_size"] <= 4096
            and type(p["max_window_bytes"]) is int and 0 < p["max_window_bytes"] <= 64 * 1024**2,
            "invalid streaming limits")
    spec = p["folds"]
    require(isinstance(spec, dict) and set(spec) == {"path", "uri", "sha256"}
            and isinstance(spec["path"], str) and bool(spec["path"])
            and Path(spec["path"]).name == spec["path"] and spec["path"] not in (".", "..")
            and isinstance(spec["uri"], str) and bool(spec["uri"])
            and isinstance(spec["sha256"], str) and len(spec["sha256"]) == 64
            and set(spec["sha256"]) <= set("0123456789abcdef"),
            "invalid fold descriptor")
    return p


def profile(bridge_config: Path, profile_config: Path, input_dir: Path, outdir: Path) -> dict:
    started = time.monotonic()
    code_hashes = {name: sha256_file(Path(__file__).with_name(name)) for name in CODE_FILES}
    config_hashes = {"bridge_config": sha256_file(bridge_config),
                     "profile_config": sha256_file(profile_config)}
    require(not outdir.exists() and not outdir.is_symlink(), "output already exists")
    parameters = load_profile(profile_config)
    bridge, paths = load_config(bridge_config, input_dir)
    folds = input_dir / parameters["folds"]["path"]
    require(folds.is_file() and not folds.is_symlink(), "fold file missing or symlinked")
    require(sha256_file(folds) == parameters["folds"]["sha256"], "fold hash mismatch")
    data = extract_inputs(paths, bridge)
    fit = training_indices(folds, data["sample_key_sha256"], parameters["folds"]["sha256"],
                           parameters["fold"], parameters["role_counts"])
    # Drop SELECT/SCORE from feature computation before canonicalization/retrieval.
    data = canonicalize_reference_homologs(subset_queries(data, fit))
    extracted_seconds = time.monotonic() - started
    anchors = np.arange(len(data["selected"]["locus_id"]), dtype=np.int64)
    source_hashes = {k: v["sha256"] for k, v in bridge["inputs"].items()}
    source_hashes.update(folds=parameters["folds"]["sha256"],
                         **config_hashes)
    outdir.mkdir(parents=True, exist_ok=False, mode=0o700)
    rows = []
    for count in parameters["people"]:
        subset = subset_queries(data, np.arange(count, dtype=np.int64))
        for radius in parameters["radii_cm"]:
            case_start = time.monotonic()
            case = f"people_{count}/radius_{radius:g}cm"
            arrays, stats = materialize_radius(subset, anchors, parameters["K"], float(radius))
            retrieval_seconds = time.monotonic() - case_start
            store = build_factorized_store(subset, arrays)
            destination = outdir / case
            destination.parent.mkdir(mode=0o700, exist_ok=True)
            store.save(destination, source_hashes=source_hashes)
            manifest_sha = sha256_file(destination / "manifest.json")
            opened = OrderedContextStore.open(destination, expected_manifest_sha256=manifest_sha)
            scan_start = time.monotonic()
            chunks, max_bytes, streamed_sites = 0, 0, 0
            checksum = hashlib.sha256()
            # Read every ordered window, including densest regions; no downsampling of sites.
            for window in opened.iter_windows(site_chunk_size=parameters["site_chunk_size"],
                                              max_window_bytes=parameters["max_window_bytes"]):
                chunk = window.channels
                require(np.isfinite(chunk).all(), "nonfinite stream channel")
                checksum.update(np.ascontiguousarray(chunk).tobytes())
                max_bytes = max(max_bytes, chunk.nbytes)
                chunks += 1
                streamed_sites += chunk.shape[-2]
            expected_sites = count * sum(int(np.sum(np.abs(subset["common_cm"] - float(cm)) <= radius))
                                         for cm in subset["selected"]["cM"])
            require(streamed_sites == expected_sites, "stream omitted or duplicated common sites")
            row = {"case": case, "people": count, "anchors": len(anchors),
                   "radius_cm": radius, "K_per_homolog_ancestry": parameters["K"],
                   "retrieval_seconds": retrieval_seconds,
                   "scan_seconds": time.monotonic() - scan_start,
                   "elapsed_seconds": time.monotonic() - case_start,
                   "stream_chunks": chunks, "largest_channel_chunk_bytes": max_bytes,
                   "streamed_query_anchor_sites": streamed_sites,
                   "expected_query_anchor_sites": expected_sites,
                   "stream_channels_sha256": checksum.hexdigest(),
                   "store_manifest_sha256": manifest_sha,
                   "store_bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
                   "peak_process_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "retrieval_stats": stats}
            write_exclusive_json(destination.parent / f"radius_{radius:g}cm.profile.json", row)
            rows.append(row)
            print(json.dumps({"completed_case": case, "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
            del opened, store, arrays
    require(all(sha256_file(paths[k]) == spec["sha256"] for k, spec in bridge["inputs"].items())
            and sha256_file(folds) == parameters["folds"]["sha256"], "original input changed")
    require(all(sha256_file(Path(__file__).with_name(name)) == digest for name, digest in code_hashes.items())
            and sha256_file(bridge_config) == config_hashes["bridge_config"]
            and sha256_file(profile_config) == config_hashes["profile_config"], "source code or config changed")
    receipt = {"schema_version": SCHEMA, "decision": "PASS_ORDERED_INPUT_TECHNICAL_ONLY",
               "parameters": parameters, "input_hashes": source_hashes, "profiles": rows,
               "input_counts": data["counts"], "extract_reconcile_seconds": extracted_seconds,
               "elapsed_seconds": time.monotonic() - started,
               "resources": {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                             "system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime},
               "boundaries": {"truth_opened": False, "predictions_opened": False,
                              "model_training": False, "training_memory_profiled": False,
                              "source_shared_target_genotypes_parsed": bridge["parameters"]["expected_target_people"],
                              "feature_computation_roles": ["TRAIN"],
                              "query_subset_policy": "nested_sample_key_order_in_inner_TRAIN",
                              "rare_phase_assigned": False, "new_cloud_instances": 0},
               "source_code_sha256": code_hashes,
               "input_dir_files_unchanged": True,
               "input_dir_semantics": "staged_copies_under_Nextflow_original_source_recheck_is_external"}
    write_exclusive_json(outdir / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.bridge_config, args.profile_config, args.input_dir, args.outdir)
    print(json.dumps({"decision": result["decision"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
