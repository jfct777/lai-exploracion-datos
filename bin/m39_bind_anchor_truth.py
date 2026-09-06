#!/usr/bin/env python3
"""Bind fixed M39 anchors to probabilities and exact M34 segment truth.

This downstream binder never creates features or fits a model. Development and
SCORE truth are written to separate files. Probability interpolation acts on the
six unordered diploid states; truth is evaluated at exact base-pair coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from m33_safe_bridge_core import write_deterministic_npz, write_exclusive_json, reopen_npz
from m34_generate_mosaics import read_genetic_map
from m34_parse_flare_truth import load_truth_segments, align_truth
from m37_trace_core import baseline_to_states, m34_labels_to_states
from m38_build_f_minus_s660 import load_selected_axis
from m38b_project_baselines import load_f0, load_marker_cm, f0_axis, verify_hash

SCHEMA = "m39-anchor-truth-v1"
STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")
ANCESTRIES = ("AFR", "EUR", "NAM")
ANCHOR_FIELDS = ("sample_key_sha256", "chrom", "pos", "ref", "alt", "cM", "locus_id", "anchor_indices")
REQUIRED_INPUTS = {"features", "bridge_receipt", "selected_loci", "genetic_map",
                   "fminus", "fminus_cm", "folds", "truth_segments"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_order(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """An exact hash join; no positional assumption or partial intersection."""
    for values in (source, target):
        require(values.ndim == 1 and values.dtype == np.dtype("|S64")
                and len(set(values.tolist())) == len(values), "sample-key axis is invalid or duplicated")
        require(all(len(bytes(value)) == 64 and set(bytes(value)).issubset(b"0123456789abcdef")
                    for value in values), "sample-key encoding differs")
    require(set(source.tolist()) == set(target.tolist()), "sample-key universes differ")
    lookup = {key: i for i, key in enumerate(source.tolist())}
    return np.asarray([lookup[key] for key in target.tolist()], dtype=np.int64)


def project_probabilities(states: np.ndarray, source_cm: np.ndarray,
                          anchor_cm: np.ndarray, exact_indices: np.ndarray | None = None
                          ) -> tuple[np.ndarray, dict]:
    """Exact-key override; otherwise mean cM ties, interpolate, clamp extremes.

    Interpolation is AFTER converting source haplotype marginals to six states.
    This avoids cross-products between different flanking markers. It does not
    reconstruct a joint posterior absent from the source haplotype marginals.
    """
    p = np.asarray(states)
    cm, anchors = np.asarray(source_cm), np.asarray(anchor_cm)
    require(p.ndim == 3 and p.shape[-1] == 6 and p.shape[1] == len(cm)
            and len(cm) > 0, "probability/coordinate dimensions differ")
    require(cm.ndim == anchors.ndim == 1 and len(anchors) > 0
            and np.isfinite(cm).all() and np.isfinite(anchors).all()
            and np.all(np.diff(cm) >= 0) and np.all(np.diff(anchors) >= 0), "cM axes must be finite and ordered")
    require(np.isfinite(p).all() and np.all(p >= 0)
            and np.allclose(p.sum(-1), 1., atol=5e-6, rtol=0), "source state simplex differs")
    exact = np.full(len(anchors), -1, dtype=np.int64) if exact_indices is None else np.asarray(exact_indices)
    require(exact.shape == anchors.shape and np.issubdtype(exact.dtype, np.integer)
            and np.all((exact >= -1) & (exact < len(cm))), "exact-key indices differ")
    used = exact >= 0
    require(np.allclose(cm[exact[used]], anchors[used], atol=1e-9, rtol=0),
            "exact-key coordinates disagree")
    unique_cm, starts, counts = np.unique(cm, return_index=True, return_counts=True)
    grouped = np.add.reduceat(p.astype(np.float64), starts, axis=1) / counts[None, :, None]
    output = np.empty((p.shape[0], len(anchors), 6), dtype=np.float64)
    audit = {"exact_locus": int(used.sum()), "exact_cm_group": 0, "interpolated": 0,
             "clamped_left": 0, "clamped_right": 0,
             "source_tied_cm_groups": int(np.sum(counts > 1))}
    for j, cm_value in enumerate(anchors):
        if used[j]:
            output[:, j] = p[:, exact[j]]
            continue
        right = int(np.searchsorted(unique_cm, cm_value, side="left"))
        if right < len(unique_cm) and unique_cm[right] == cm_value:
            output[:, j] = grouped[:, right]
            audit["exact_cm_group"] += 1
        elif right == 0:
            output[:, j] = grouped[:, 0]
            audit["clamped_left"] += 1
        elif right == len(unique_cm):
            output[:, j] = grouped[:, -1]
            audit["clamped_right"] += 1
        else:
            fraction = (cm_value - unique_cm[right - 1]) / (unique_cm[right] - unique_cm[right - 1])
            output[:, j] = (1 - fraction) * grouped[:, right - 1] + fraction * grouped[:, right]
            audit["interpolated"] += 1
    output /= output.sum(axis=-1, keepdims=True)
    return np.ascontiguousarray(output, dtype=np.float32), audit


def fold_roles(path: Path, target_keys: np.ndarray, fold: int, expected: dict) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"sample_key_sha256", "roles", "outer_fold", "inner_split_seed", "outer_seed"},
                "fold NPZ inventory differs")
        order = sample_order(archive["sample_key_sha256"], target_keys)
        roles, folds = archive["roles"], archive["outer_fold"]
        require(roles.shape == (3, len(target_keys)) and np.array_equal(folds, np.arange(3)),
                "M38 three-fold axes differ")
        require(fold in (0, 1, 2) and set(np.unique(roles).tolist()) == {"TRAIN", "SELECT", "SCORE"},
                "fold index or role labels differ")
        require(np.all((roles == "SCORE").sum(axis=0) == 1), "SCORE membership is not one per person")
        for row in roles:
            require(all(int(np.sum(row == role)) == expected[role] for role in expected),
                    "M38 TRAIN/SELECT/SCORE counts differ")
        return np.ascontiguousarray(roles[fold, order]), {
            "outer_fold": fold, "outer_seed": int(archive["outer_seed"][0]),
            "inner_split_seed": int(archive["inner_split_seed"][fold]),
        }


def partition_payload(payload: dict[str, np.ndarray], roles: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray]:
    person_fields = {"sample_key_sha256", "baseline", "full_baseline", "truth_state"}
    result = {key: np.ascontiguousarray(value[indices] if key in person_fields else value)
              for key, value in payload.items()}
    selected_roles = roles[indices]
    result["source_indices"] = np.asarray(indices, dtype=np.int64)
    for role in ("TRAIN", "SELECT", "SCORE"):
        local_indices = np.flatnonzero(selected_roles == role).astype(np.int64)
        if len(local_indices):
            result[f"{role.lower()}_indices"] = local_indices
    return result


def bind(manifest_path: Path, input_root: Path, outdir: Path) -> dict:
    started = time.monotonic()
    require(not outdir.exists(), "refusing to overwrite an existing binding directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(set(manifest) == {"schema_version", "scope", "inputs", "parameters"}
            and manifest["schema_version"] == SCHEMA
            and manifest["scope"] == "exploratory_chr22_R0_FIT_no_training", "binder manifest scope/schema differs")
    descriptors, params = manifest["inputs"], manifest["parameters"]
    require(set(descriptors) in (REQUIRED_INPUTS, REQUIRED_INPUTS | {"full", "full_cm"}),
            "binder input inventory differs")
    require(set(params) == {"people", "anchors", "fminus_markers", "full_markers", "fold", "role_counts", "write_all"},
            "binder parameters differ")
    require(all(type(params[key]) is int and params[key] > 0
                for key in ("people", "anchors", "fminus_markers", "full_markers"))
            and type(params["fold"]) is int and type(params["write_all"]) is bool,
            "invalid binder sizes, fold or write_all")
    require(set(params["role_counts"]) == {"TRAIN", "SELECT", "SCORE"}
            and all(type(v) is int and v > 0 for v in params["role_counts"].values())
            and sum(params["role_counts"].values()) == params["people"], "role-count contract differs")
    paths, inputs = {}, {}
    for name, descriptor in descriptors.items():
        require(set(descriptor) == {"path", "sha256", "uri"}, "input descriptor fields differ")
        path = (input_root / descriptor["path"]).resolve()
        observed = verify_hash(path, descriptor["sha256"], name)
        paths[name] = path
        inputs[name] = {"sha256": observed, "bytes": path.stat().st_size, "uri": descriptor["uri"]}
    with np.load(paths["features"], allow_pickle=False) as archive:
        require(set(ANCHOR_FIELDS).issubset(archive.files), "feature anchor metadata incomplete")
        anchors = {name: np.ascontiguousarray(archive[name]) for name in ANCHOR_FIELDS}
        require(tuple(archive["ancestry"].astype(str)) == ANCESTRIES, "feature ancestry order differs")
        require(archive["rare_semantics"].astype(str).tolist() == ["diploid_reference_feature_not_phased_allele"],
                "feature rare semantics differ")
    n, j = params["people"], params["anchors"]
    require(anchors["sample_key_sha256"].shape == (n,), "feature people count differs")
    sample_order(anchors["sample_key_sha256"], anchors["sample_key_sha256"])
    require(all(anchors[name].shape == (j,) for name in ANCHOR_FIELDS if name != "sample_key_sha256")
            and np.array_equal(anchors["anchor_indices"], np.arange(j)), "complete selected-anchor axes differ")
    require(anchors["pos"].dtype == np.int64 and np.all(anchors["chrom"] == 22)
            and np.all(anchors["pos"] > 0) and np.all(np.diff(anchors["pos"]) > 0)
            and len(np.unique(anchors["locus_id"])) == j, "anchor positions or locus identities differ")
    bridge = json.loads(paths["bridge_receipt"].read_text(encoding="utf-8"))
    matching_profiles = [row for row in bridge.get("profiles", []) if row.get("sha256") == inputs["features"]["sha256"]]
    require(bridge.get("decision") == "PASS_TECHNICAL_BRIDGE_ONLY"
            and bridge.get("boundaries", {}).get("truth_opened") is False
            and bridge.get("boundaries", {}).get("target_partition") == "R0_FIT"
            and len(matching_profiles) == 1
            and matching_profiles[0]["tensor_shape"][:4] == [n, j, 2, 3], "bridge authentication or scope differs")
    selected = load_selected_axis(paths["selected_loci"], expected_chromosome="22", expected_count=j)
    anchor_keys = tuple(("22", int(p), bytes(r).decode("ascii"), bytes(a).decode("ascii"))
                        for p, r, a in zip(anchors["pos"], anchors["ref"], anchors["alt"]))
    require(anchor_keys == selected, "anchor allele axis differs from exact selected loci")
    with np.load(paths["selected_loci"], allow_pickle=False) as archive:
        require(np.array_equal(archive["locus_id"], anchors["locus_id"])
                and np.array_equal(archive["cM"], anchors["cM"]), "selected locus IDs/cM differ")
    genetic_map = read_genetic_map(paths["genetic_map"], "22")
    expected_cm = np.asarray([genetic_map.bp_to_cm(int(pos)) for pos in anchors["pos"]])
    require(np.allclose(expected_cm, anchors["cM"], atol=1e-9, rtol=0), "anchor coordinates differ from genetic map")
    roles, fold_audit = fold_roles(paths["folds"], anchors["sample_key_sha256"], params["fold"], params["role_counts"])
    payload = {name: value for name, value in anchors.items() if name != "cM"}
    payload["coords"] = anchors["cM"].astype(np.float64)
    payload["state_names"] = np.asarray(STATE_NAMES, dtype="|S2")
    projections = {}
    minus_keys = None
    for source, output in (("fminus", "baseline"), ("full", "full_baseline")):
        if source not in paths:
            continue
        arrays = load_f0(paths[source], source)
        keys = f0_axis(arrays, source)
        require(len(keys) == params[f"{source}_markers"], "baseline marker count differs")
        cm = load_marker_cm(paths[source + "_cm"], len(keys), source)
        check_cm = np.asarray([genetic_map.bp_to_cm(key[1]) for key in keys])
        require(np.allclose(cm, check_cm, atol=1e-9, rtol=0), "baseline cM does not match its marker axis/map")
        order = sample_order(arrays["sample_key_sha256"], anchors["sample_key_sha256"])
        if source == "fminus":
            require(not set(keys).intersection(anchor_keys), "incremental anchors overlap F-minus-S")
            minus_keys = set(keys)
        else:
            require(set(keys) == minus_keys | set(anchor_keys), "full/minus/anchor partition is not exact")
        lookup = {key: index for index, key in enumerate(keys)}
        exact = np.asarray([lookup.get(key, -1) for key in anchor_keys], dtype=np.int64)
        states = baseline_to_states(arrays["F0"])[order]
        payload[output], projections[source] = project_probabilities(states, cm, anchors["cM"], exact)
        del states, arrays
    # Truth is opened only after feature, baseline, map and role authentication.
    segments = load_truth_segments(paths["truth_segments"], ANCESTRIES)
    haplotype_truth, truth_audit = align_truth(segments, anchors["sample_key_sha256"], anchors["pos"], ANCESTRIES)
    payload["truth_state"] = m34_labels_to_states(haplotype_truth)
    outdir.mkdir(parents=True, mode=0o700, exist_ok=False)
    outputs = {}
    partitions = {"development.npz": np.flatnonzero(roles != "SCORE"),
                  "score.npz": np.flatnonzero(roles == "SCORE")}
    if params["write_all"]:
        partitions["anchor-data.npz"] = np.arange(n)
    for filename, indices in partitions.items():
        result = partition_payload(payload, roles, indices)
        path = outdir / filename
        write_deterministic_npz(path, result)
        reopen_npz(path, result)
        path.chmod(0o400)
        outputs[filename] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                             "people": len(indices), "roles": {r: int(np.sum(roles[indices] == r)) for r in params["role_counts"]}}
    receipt = {"schema_version": SCHEMA, "decision": "PASS_EXPLORATORY_EXACT_ANCHOR_BINDING",
               "scope": manifest["scope"], "people": n, "anchors": j, "fold": fold_audit,
               "inputs": inputs, "outputs": outputs, "projection_audit": projections,
               "truth_audit": truth_audit, "state_names": list(STATE_NAMES),
               "probability_policy": "six_states_then_exact_key_else_mean_cm_ties_linear_cm_clamped_extremes_renormalize_no_floor",
               "full_baseline_kind": "original_full_marker_posterior" if "full" in paths else "not_available",
               "truth_policy": "exact_POS_in_recorded_half_open_start_bp_end_bp_exclusive_segments",
               "boundaries": {"truth_opened": True, "SCORE_physically_separated": True,
                              "training_performed": False, "features_created_or_selected": False,
                              "VALID_TEST_opened": False, "source_TEST_opened": False,
                              "F1_evaluated": False, "contains_raw_sample_ids": False},
               "manifest_sha256": sha256(manifest_path),
               "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in (
                   "m39_bind_anchor_truth.py", "m33_safe_bridge_core.py", "m34_parse_flare_truth.py",
                   "m34_generate_mosaics.py", "m37_trace_core.py", "m38b_project_baselines.py", "m38_build_f_minus_s660.py")},
               "elapsed_seconds": time.monotonic() - started}
    write_exclusive_json(outdir / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    receipt = bind(args.manifest, args.input_root, args.outdir)
    print(json.dumps({key: receipt[key] for key in ("decision", "people", "anchors", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
