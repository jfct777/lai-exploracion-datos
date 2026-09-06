#!/usr/bin/env python3
"""Audit stored M39 zero probabilities without fitting or changing source files.

The earliest observed stage is normalized, exported haplotype probabilities in
F0 NPZs, not FLARE's internal posterior. A zero already present there cannot be
attributed to internal support, decimal export, or parser rounding by this audit.
The six-state and anchor-projection replay below is independent of the historical
generators; those generators and the original probability vectors stay intact.

Input manifest: schema_version=m39-zero-origin-audit-v1,
scope=historical_R0_FIT_artifact_audit, inputs={role: {path, sha256}} with the seven
roles in INPUT_ROLES. Paths are relative to --input-root. Only aggregate counts,
hashes and numeric metadata are emitted; neither sample keys nor loci are output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "m39-zero-origin-audit-v1"
SCOPE = "historical_R0_FIT_artifact_audit"
INPUT_ROLES = {"binding_receipt", "development", "score", "fminus", "fminus_cm", "full", "full_cm"}
STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")
STATE_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
ORIGINS = ("absent_support_in_normalized_exported_F0", "six_state_float32_underflow",
           "projection_float64_underflow", "projection_float32_underflow")
SUPPORT_CATEGORIES = ("zero_inherited_absent_F0_support", "zero_numerical",
                      "positive_mixed_F0_support", "positive_all_F0_support")
F0_FIELDS = {"sample_key_sha256", "marker_chrom", "marker_pos", "marker_ref", "marker_alt", "F0"}
AXIS_FIELDS = ("chrom", "pos", "ref", "alt", "coords", "locus_id", "anchor_indices", "state_names")


class ZeroOriginAuditError(ValueError):
    """The authenticated scope, source axes, or replay differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ZeroOriginAuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_keys(keys: np.ndarray) -> None:
    require(keys.ndim == 1 and keys.dtype == np.dtype("S64") and len(keys) > 0
            and len(set(keys.tolist())) == len(keys), "sample-key axis invalid or duplicated")
    require(all(len(bytes(key)) == 64 and set(bytes(key)).issubset(b"0123456789abcdef")
                for key in keys), "sample-key encoding differs")


def simplex(values: np.ndarray) -> None:
    require(np.isfinite(values).all() and np.all(values >= 0)
            and np.allclose(values.sum(-1), 1, atol=5e-6, rtol=0), "probability simplex differs")


def variant_axis(data: dict, prefix: str = "") -> tuple[tuple, ...]:
    names = tuple(prefix + name for name in ("chrom", "pos", "ref", "alt"))
    require(all(name in data for name in names), "variant axis missing")
    n = len(data[names[1]])
    require(all(data[name].shape == (n,) for name in names), "variant-axis dimensions differ")
    require(np.issubdtype(data[names[1]].dtype, np.integer), "variant positions must be integers")
    rows = tuple((int(c), int(p), bytes(r), bytes(a)) for c, p, r, a in zip(*(data[name] for name in names)))
    require(n > 0 and all(c == 22 and p > 0 and r in (b"A", b"C", b"G", b"T")
                         and a in (b"A", b"C", b"G", b"T") and r != a for c, p, r, a in rows),
            "variant axis is not chr22 SNVs")
    require(all(a[1] < b[1] for a, b in zip(rows, rows[1:])), "variant positions not strictly ordered")
    return rows


def six_states(haplotypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use float64 products/sums, then the historical float32 state storage."""
    p = np.asarray(haplotypes)
    require(p.ndim == 4 and p.shape[1] == 2 and p.shape[-1] == 3, "haplotype dimensions differ")
    require(p.dtype == np.dtype("float32"), "F0 must retain the historical float32 dtype")
    simplex(p)
    left, right = p[:, 0].astype(np.float64), p[:, 1].astype(np.float64)
    states = np.stack([left[..., a] * right[..., b] if a == b else
                       left[..., a] * right[..., b] + left[..., b] * right[..., a]
                       for a, b in STATE_PAIRS], axis=-1)
    return states, states.astype(np.float32)


def projection_plan(source_cm: np.ndarray, anchor_cm: np.ndarray, exact: np.ndarray) -> list[dict]:
    """Freeze nonnegative contributor weights, preserving exact-key precedence."""
    cm, anchors = np.asarray(source_cm), np.asarray(anchor_cm)
    require(cm.ndim == anchors.ndim == 1 and len(cm) > 0 and len(anchors) > 0
            and np.isfinite(cm).all() and np.isfinite(anchors).all()
            and np.all(np.diff(cm) >= 0) and np.all(np.diff(anchors) >= 0), "cM axes invalid")
    require(exact.shape == anchors.shape and np.issubdtype(exact.dtype, np.integer)
            and np.all((exact >= -1) & (exact < len(cm))), "exact-key indices invalid")
    used = exact >= 0
    require(np.allclose(cm[exact[used]], anchors[used], atol=1e-9, rtol=0), "exact-key cM differs")
    unique, starts, counts = np.unique(cm, return_index=True, return_counts=True)
    groups = [np.arange(start, start + count) for start, count in zip(starts, counts)]
    result = []
    for value, index in zip(anchors, exact):
        if index >= 0:
            result.append({"mode": "exact_locus", "groups": [np.asarray([index])], "weights": [1.]})
            continue
        right = int(np.searchsorted(unique, value))
        if right < len(unique) and unique[right] == value:
            group_indices, weights, mode = [right], [1.], "exact_cm_group"
        elif right == 0:
            group_indices, weights, mode = [0], [1.], "clamped_left"
        elif right == len(unique):
            group_indices, weights, mode = [len(unique) - 1], [1.], "clamped_right"
        else:
            fraction = (value - unique[right - 1]) / (unique[right] - unique[right - 1])
            group_indices, weights, mode = [right - 1, right], [1. - fraction, fraction], "interpolated"
        result.append({"mode": mode, "groups": [groups[i] for i in group_indices], "weights": weights})
    return result


def project(states: np.ndarray, plan: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Average each cM group before interpolation, normalize, and store float32."""
    out = np.zeros((len(states), len(plan), 6), dtype=np.float64)
    support = np.zeros(out.shape, dtype=bool)
    for j, item in enumerate(plan):
        for group, weight in zip(item["groups"], item["weights"]):
            if weight > 0:
                out[:, j] += weight * (np.sum(states[:, group].astype(np.float64), axis=1) / len(group))
                support[:, j] |= np.any(states[:, group] > 0, axis=1)
    total = out.sum(-1, keepdims=True)
    require(np.isfinite(out).all() and np.all(total > 0), "invalid projected mass")
    out /= total
    return out, support


def classify_zeros(states64: np.ndarray, states32: np.ndarray, plan: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Partition output zeros by observed support and numerical conversion stage.

    The source tensors are float32: even two minimum subnormals have a product
    representable in float64. Thus positive float64 products diagnose state-cast
    underflow without confusing it with absent stored haplotype support.
    """
    projected, support32 = project(states32, plan)
    _, support64 = project(states64, plan)
    stored = projected.astype(np.float32)
    zero = stored == 0
    labels = np.full(stored.shape, -1, dtype=np.int8)
    labels[zero & ~support64] = 0
    labels[zero & support64 & ~support32] = 1
    labels[zero & support32 & (projected == 0)] = 2
    labels[zero & (projected > 0)] = 3
    require(np.array_equal(labels >= 0, zero), "zero-origin partition incomplete")
    return stored, labels


def origin_counts(labels: np.ndarray, truth: np.ndarray | None = None) -> dict[str, int]:
    if truth is not None:
        labels = np.take_along_axis(labels, truth[..., None], axis=-1)[..., 0]
    return {name: int(np.sum(labels == index)) for index, name in enumerate(ORIGINS)}


def support_categories(states64: np.ndarray, plan: list[dict], labels: np.ndarray) -> np.ndarray:
    """Classify zero and positive cells using every positive-weight F0 contributor."""
    all_positive = np.ones(labels.shape, dtype=bool)
    for j, item in enumerate(plan):
        for group, weight in zip(item["groups"], item["weights"]):
            if weight > 0:
                all_positive[:, j] &= np.all(states64[:, group] > 0, axis=1)
    classes = np.full(labels.shape, 2, dtype=np.int8)
    classes[(labels < 0) & all_positive] = 3
    classes[labels == 0] = 0
    classes[labels > 0] = 1
    return classes


def update_positive_minimum(result: dict, name: str, values: np.ndarray) -> None:
    positive = values[values > 0]
    if positive.size:
        minimum = float(positive.min())
        result[name] = min(result[name] if result[name] is not None else minimum, minimum)


def validate_bound(data: dict, role: str) -> None:
    required = set(AXIS_FIELDS) | {"sample_key_sha256", "source_indices", "baseline", "full_baseline", "truth_state"}
    indices = {"train_indices", "select_indices"} if role == "development" else {"score_indices"}
    require(set(data) == required | indices, "bound NPZ inventory differs")
    validate_keys(data["sample_key_sha256"])
    n, j = len(data["sample_key_sha256"]), len(data["pos"])
    variant_axis(data)
    require(data["coords"].dtype == np.dtype("float64") and data["coords"].shape == (j,)
            and np.isfinite(data["coords"]).all() and np.all(np.diff(data["coords"]) >= 0), "anchor cM invalid")
    require(data["state_names"].tolist() == [name.encode() for name in STATE_NAMES], "state order differs")
    require(np.array_equal(data["anchor_indices"], np.arange(j)) and data["locus_id"].shape == (j,)
            and len(set(data["locus_id"].tolist())) == j, "anchor identities differ")
    truth = data["truth_state"]
    require(truth.shape == (n, j) and np.issubdtype(truth.dtype, np.integer)
            and np.all((truth >= 0) & (truth < 6)), "truth labels invalid")
    require(data["source_indices"].shape == (n,) and np.issubdtype(data["source_indices"].dtype, np.integer),
            "source indices invalid")
    for name in ("baseline", "full_baseline"):
        require(data[name].shape == (n, j, 6) and data[name].dtype == np.dtype("float32"), "bound baseline dtype/shape differs")
        simplex(data[name])
    combined = []
    for name in sorted(indices):
        value = data[name]
        require(value.ndim == 1 and len(value) > 0 and np.issubdtype(value.dtype, np.integer)
                and np.all((value >= 0) & (value < n)), "role indices invalid")
        combined.extend(value.tolist())
    require(sorted(combined) == list(range(n)), "role indices overlap or leave gaps")


def audit(manifest_path: Path, input_root: Path, output: Path, chunk_people: int = 8) -> dict[str, Any]:
    """Authenticate, replay in person chunks, verify unchanged inputs, write once."""
    require(not output.exists(), "refusing to overwrite audit output")
    require(type(chunk_people) is int and chunk_people > 0, "chunk_people must be positive")
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(set(manifest) == {"schema_version", "scope", "inputs"}
            and manifest["schema_version"] == SCHEMA and manifest["scope"] == SCOPE, "manifest scope/schema differs")
    require(set(manifest["inputs"]) == INPUT_ROLES, "input inventory differs")
    root = input_root.resolve()
    paths, authenticated = {}, {}
    for role, descriptor in manifest["inputs"].items():
        require(set(descriptor) == {"path", "sha256"}, "input descriptor differs")
        relative = Path(descriptor["path"])
        path = (root / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(root) and path.is_file(), "input escapes root or is absent")
        expected = descriptor["sha256"]
        require(isinstance(expected, str) and len(expected) == 64 and set(expected).issubset("0123456789abcdef"),
                "input SHA-256 malformed")
        require(sha256(path) == expected, f"input SHA-256 mismatch: {role}")
        paths[role] = path
        authenticated[role] = {"sha256": expected, "bytes": path.stat().st_size}
    require(output.resolve() not in set(paths.values()) | {manifest_path.resolve()}, "output aliases input")
    binding = json.loads(paths["binding_receipt"].read_text())
    require(binding.get("schema_version") == "m39-anchor-truth-v1"
            and binding.get("scope") == "exploratory_chr22_R0_FIT_no_training"
            and binding.get("decision") == "PASS_EXPLORATORY_EXACT_ANCHOR_BINDING"
            and binding.get("boundaries", {}).get("VALID_TEST_opened") is False
            and binding.get("boundaries", {}).get("source_TEST_opened") is False, "binding receipt scope differs")
    for role in ("development", "score"):
        require(binding.get("outputs", {}).get(role + ".npz", {}).get("sha256") == authenticated[role]["sha256"],
                "bound artifact not authenticated by binder")
    for role in ("fminus", "fminus_cm", "full", "full_cm"):
        require(binding.get("inputs", {}).get(role, {}).get("sha256") == authenticated[role]["sha256"],
                "source artifact not authenticated by binder")
    bound = {role: load_npz(paths[role]) for role in ("development", "score")}
    for role, data in bound.items():
        validate_bound(data, role)
        descriptor = binding["outputs"][role + ".npz"]
        observed_roles = {name: int(len(data.get(name.lower() + "_indices", [])))
                          for name in ("TRAIN", "SELECT", "SCORE")}
        require(descriptor.get("people") == len(data["sample_key_sha256"])
                and descriptor.get("roles") == observed_roles, "bound role counts differ from binder")
    dev, score = bound["development"], bound["score"]
    require(all(np.array_equal(dev[name], score[name]) for name in AXIS_FIELDS), "development/SCORE anchor axes differ")
    all_keys = np.concatenate([dev["sample_key_sha256"], score["sample_key_sha256"]])
    validate_keys(all_keys)
    source_indices = np.concatenate([dev["source_indices"], score["source_indices"]])
    require(sorted(source_indices.tolist()) == list(range(len(all_keys))), "source indices overlap or leave gaps")
    require(binding.get("people") == len(all_keys) and binding.get("anchors") == len(dev["pos"]), "binder sizes differ")
    anchors = variant_axis(dev)
    results, minus_axis = {}, None
    for source, target in (("fminus", "baseline"), ("full", "full_baseline")):
        data = load_npz(paths[source])
        require(set(data) == F0_FIELDS, "F0 inventory differs")
        validate_keys(data["sample_key_sha256"])
        require(set(data["sample_key_sha256"].tolist()) == set(all_keys.tolist()), "F0 sample universe differs")
        axis = variant_axis(data, "marker_")
        require(data["F0"].shape == (len(all_keys), 2, len(axis), 3), "F0 dimensions differ")
        cm_data = load_npz(paths[source + "_cm"])
        require(set(cm_data) == {"marker_cM"} and cm_data["marker_cM"].shape == (len(axis),)
                and cm_data["marker_cM"].dtype == np.dtype("float64"), "source cM inventory differs")
        if source == "fminus":
            require(not set(axis) & set(anchors), "Fminus overlaps selected anchors")
            minus_axis = set(axis)
        else:
            require(set(axis) == minus_axis | set(anchors), "full/minus/selected exact partition differs")
        lookup = {key: index for index, key in enumerate(axis)}
        exact = np.asarray([lookup.get(key, -1) for key in anchors], dtype=np.int64)
        plan = projection_plan(cm_data["marker_cM"], dev["coords"], exact)
        mode_counts = Counter(item["mode"] for item in plan)
        require(all(binding.get("projection_audit", {}).get(source, {}).get(mode, 0) == mode_counts.get(mode, 0)
                    for mode in ("exact_locus", "exact_cm_group", "interpolated", "clamped_left", "clamped_right")),
                "binder projection modes differ")
        result = {"source_people": len(all_keys), "source_markers": len(axis), "projection_modes": dict(mode_counts),
                  "source_haplotype_entries": int(data["F0"].size), "source_haplotype_zeros": 0,
                  "source_haplotype_min_positive": None, "six_state_entries": len(all_keys) * len(axis) * 6,
                  "six_state_float64_min_positive": None, "six_state_float32_min_positive": None,
                  "anchor_float32_min_positive": None,
                  "six_state_float64_zeros": 0, "six_state_new_float32_zeros": 0,
                  "max_absolute_replay_error": 0., "roles": {}}
        sample_lookup = {key: i for i, key in enumerate(data["sample_key_sha256"].tolist())}
        for part, package in bound.items():
            for role in (("TRAIN", "SELECT") if part == "development" else ("SCORE",)):
                indices = package[role.lower() + "_indices"]
                role_result = {"people": len(indices), "anchors": len(anchors), "person_anchor_observations": len(indices) * len(anchors),
                               "all_state_entries": len(indices) * len(anchors) * 6,
                               "all_state_zero_origins": dict.fromkeys(ORIGINS, 0),
                               "true_state_zero_origins": dict.fromkeys(ORIGINS, 0),
                               "true_state_zeros_by_state": dict.fromkeys(STATE_NAMES, 0),
                               "true_state_support_categories": dict.fromkeys(SUPPORT_CATEGORIES, 0)}
                for start in range(0, len(indices), chunk_people):
                    rows = indices[start:start + chunk_people]
                    order = [sample_lookup[key] for key in package["sample_key_sha256"][rows].tolist()]
                    f0 = data["F0"][order]
                    states64, states32 = six_states(f0)
                    replay, labels = classify_zeros(states64, states32, plan)
                    expected = package[target][rows]
                    result["max_absolute_replay_error"] = max(result["max_absolute_replay_error"],
                                                               float(np.max(np.abs(replay.astype(float) - expected))))
                    require(np.array_equal(replay, expected), "anchor probability replay is not exact")
                    result["source_haplotype_zeros"] += int(np.sum(f0 == 0))
                    for name, values in (("source_haplotype_min_positive", f0),
                                         ("six_state_float64_min_positive", states64),
                                         ("six_state_float32_min_positive", states32),
                                         ("anchor_float32_min_positive", replay)):
                        update_positive_minimum(result, name, values)
                    result["six_state_float64_zeros"] += int(np.sum(states64 == 0))
                    result["six_state_new_float32_zeros"] += int(np.sum((states64 > 0) & (states32 == 0)))
                    truth = package["truth_state"][rows]
                    classes = support_categories(states64, plan, labels)
                    true_classes = np.take_along_axis(classes, truth[..., None], axis=-1)[..., 0]
                    for index, name in enumerate(SUPPORT_CATEGORIES):
                        role_result["true_state_support_categories"][name] += int(np.sum(true_classes == index))
                    for field, counts in (("all_state_zero_origins", origin_counts(labels)),
                                          ("true_state_zero_origins", origin_counts(labels, truth))):
                        for name, count in counts.items():
                            role_result[field][name] += count
                    true_zero = np.take_along_axis(labels, truth[..., None], axis=-1)[..., 0] >= 0
                    for state, name in enumerate(STATE_NAMES):
                        role_result["true_state_zeros_by_state"][name] += int(np.sum(true_zero & (truth == state)))
                role_result["true_state_zeros"] = sum(role_result["true_state_zero_origins"].values())
                require(sum(role_result["true_state_support_categories"].values()) == role_result["person_anchor_observations"],
                        "true-state support categories do not partition observations")
                result["roles"][role] = role_result
        results[source] = result
        del data
    require(sha256(manifest_path) == manifest_hash, "manifest changed during audit")
    require(all(sha256(paths[role]) == item["sha256"] for role, item in authenticated.items()), "original changed during audit")
    report = {"schema_version": SCHEMA, "scope": SCOPE, "decision": "PASS_EXACT_STORED_BASELINE_REPLAY",
              "earliest_verified_stage": "normalized_exported_haplotype_probabilities",
              "raw_internal_probability": "unavailable", "raw_ancestry_vcf_checked": False,
              "parser_zero_creation_assessed": False, "internal_vs_export_zero_origin": "not_identified",
              "training_performed": False, "genotypes_opened": False, "VALID_TEST_opened": False,
              "individual_identifiers_emitted": False, "original_inputs_unchanged": True,
              "manifest_sha256": manifest_hash, "inputs": authenticated,
              "historical_generator_sha256": binding.get("code_sha256", {}),
              "audit_code_sha256": sha256(Path(__file__)), "numpy_version": np.__version__, "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    output.chmod(0o400)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-people", type=int, default=8, help="Memory-only person chunk size; does not change the estimand")
    args = parser.parse_args()
    report = audit(args.manifest, args.input_root, args.output, args.chunk_people)
    print(json.dumps({"decision": report["decision"], "original_inputs_unchanged": True}))


if __name__ == "__main__":
    main()
