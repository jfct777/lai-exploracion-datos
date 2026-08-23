#!/usr/bin/env python3
"""Run the non-consumable root17/root18 equivalence KAT for lazy M33 contexts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import m33_materialize as materialize
from m31_ordered_linear import load_genetic_map


ROOTS = {"root17": 20260817, "root18": 20260818}
FILES = {
    "selected": "technical_kat_selected_loci_incremental.npz",
    "target": "technical_kat_target_rare_diploid_incremental.npz",
    "reference": "technical_kat_reference_rare_summary_incremental.npz",
    "f0": "technical_kat_flare_f0_sanitized.npz",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON root differs")
    return value


def validate_source_auth(path: Path, implementation_commit: str, source_root: Path) -> str:
    payload = load_json(path)
    require(payload.get("stage") == "M33_M0_FACTORIZED_LAZY_SOURCE_AUTH" and
            payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES" and
            payload.get("git_commit") == implementation_commit,
            "factorized/lazy source authentication differs")
    hashes = payload.get("source_sha256", {})
    for relative in ("bin/m33_materialize.py", "bin/m33_m0_factorized_lazy_technical_kat.py",
                     "bin/m33_m0_contract.py", "bin/m31_ordered_linear.py"):
        require(hashes.get(relative) == sha256_file(source_root / relative),
                f"authenticated source differs: {relative}")
    return sha256_file(path)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def memory_limit_bytes() -> int:
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if path.is_file():
            raw = path.read_text(encoding="ascii").strip()
            if raw != "max" and raw.isdigit() and int(raw) < 1 << 60:
                return int(raw)
    return int(Path("/proc/meminfo").read_text().split("MemTotal:", 1)[1].split()[0]) * 1024


def enforce_memory_gate(limit_bytes: int) -> float:
    fraction = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024) / limit_bytes
    require(fraction < 0.8, f"peak RSS reached the 0.8 stop fraction: {fraction:.6f}")
    return fraction


def load_technical(root_dir: Path, root_label: str, root_seed: int,
                   verify_receipt_path: Path, genetic_map_path: Path):
    receipt_path = root_dir / "safe_bridge_technical_kat.receipt.json"
    receipt = load_json(receipt_path)
    verify = load_json(verify_receipt_path)
    for payload in (receipt, verify):
        require(payload["root_label"] == root_label and payload["root_seed"] == root_seed,
                "technical root identity differs")
        require(payload["status"] == "PASS_SAFE_BRIDGE_TECHNICAL_ROOT_KAT_ONLY_NON_CONSUMABLE" and
                payload["consumable"] is False and payload["truth_read"] is False and
                payload["ready_emitted"] is False, "technical firewall differs")
    require(receipt["materialize_authorized"] is False and receipt["training_authorized"] is False and
            verify["materialize"] is False and verify["training"] is False and
            verify["reopen_verified"] is True, "technical authorization differs")
    map_sha = sha256_file(genetic_map_path)
    require(receipt["input_sha256_pre"]["genetic_map"] ==
            receipt["input_sha256_post"]["genetic_map"] == map_sha,
            "authenticated genetic map differs")
    require(verify["bridge_receipt_sha256"] == sha256_file(receipt_path),
            "bridge receipt binding differs")
    paths = {kind: root_dir / name for kind, name in FILES.items()}
    observed_hashes = {path.name: sha256_file(path) for path in paths.values()}
    require(observed_hashes == receipt["artifact_raw_sha256"] == verify["artifact_raw_sha256"],
            "technical artifact raw hashes differ")

    archives: dict[str, dict[str, np.ndarray]] = {}
    for kind, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            archives[kind] = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    technical_selected = archives["selected"]
    expected_selected = {"locus_key_sha256", "chrom", "pos", "ref", "alt", "cM", "minor_code"}
    require(set(technical_selected) == expected_selected, "technical selected schema differs")
    keys = technical_selected["locus_key_sha256"]
    key_text = [value.decode("ascii") for value in keys]
    require(all(len(value) == 64 and set(value) <= set("0123456789abcdef") for value in key_text),
            "technical locus hash differs")
    locus_ids = np.asarray([int(value[:16], 16) for value in key_text], dtype="<u8")
    require(np.unique(locus_ids).size == locus_ids.size, "technical uint64 locus projection collided")
    require(np.all(np.isin(technical_selected["minor_code"], [0, 1])) and
            int(np.sum(technical_selected["minor_code"] == 0)) ==
            receipt["incremental_minor_code_0_locus_count"], "technical minor orientation audit differs")
    selected = {
        "locus_id": locus_ids, "chrom": technical_selected["chrom"],
        "pos": technical_selected["pos"], "ref": technical_selected["ref"],
        "alt": technical_selected["alt"], "cM": technical_selected["cM"],
    }
    technical_target = archives["target"]
    technical_reference = archives["reference"]
    require(np.array_equal(technical_target.pop("locus_key_sha256"), keys) and
            np.array_equal(technical_reference.pop("locus_key_sha256"), keys),
            "technical locus axes differ")
    target = {**technical_target, "locus_id": locus_ids}
    reference = {**technical_reference, "locus_id": locus_ids}
    technical_f0 = archives["f0"]
    require(set(technical_f0) == {"sample_key_sha256", "chrom", "pos", "ref", "alt", "probabilities"},
            "technical F0 schema differs")
    f0 = {
        "sample_key_sha256": technical_f0["sample_key_sha256"],
        "marker_chrom": technical_f0["chrom"], "marker_pos": technical_f0["pos"],
        "marker_ref": technical_f0["ref"], "marker_alt": technical_f0["alt"],
        "F0": technical_f0["probabilities"],
    }
    genetic_map = load_genetic_map(genetic_map_path, "22")
    marker_cm = np.ascontiguousarray(genetic_map.cm_at(f0["marker_pos"]), dtype="<f8")
    materialize.validate_inputs(selected, target, reference, f0, marker_cm)
    return (selected, target, reference, f0, marker_cm, sha256_file(receipt_path),
            array_sha256(keys))


def stratified_markers(rare_cm: np.ndarray, marker_cm: np.ndarray, count: int) -> np.ndarray:
    require(0 < count <= marker_cm.size, "marker KAT count differs")
    intervals = materialize.build_interval_table(rare_cm, marker_cm)
    chosen = set(np.linspace(0, marker_cm.size - 1, min(320, count), dtype=int).tolist())
    for radius_index in range(len(materialize.RADII)):
        lengths = intervals["context_stop"][radius_index] - intervals["context_start"][radius_index]
        order = np.argsort(lengths, kind="stable")
        chosen.update(order[:16].tolist())
        chosen.update(order[-16:].tolist())
        empty = np.flatnonzero(lengths == 0)
        chosen.update(empty[:8].tolist())
    if len(chosen) < count:
        for index in np.linspace(0, marker_cm.size - 1, count * 4, dtype=int):
            chosen.add(int(index))
            if len(chosen) == count:
                break
    ordered = np.asarray(sorted(chosen), dtype="<u8")
    if ordered.size > count:
        keep = np.linspace(0, ordered.size - 1, count, dtype=int)
        ordered = ordered[keep]
    require(ordered.size == count and np.unique(ordered).size == count, "marker stratum differs")
    return ordered


def validate_marker_count(value: int) -> None:
    require(type(value) is int and value == 512,
            "technical KAT requires exactly 512 markers")


def subset_f0(f0: Mapping[str, np.ndarray], marker_indexes: np.ndarray) -> dict[str, np.ndarray]:
    result = {name: np.ascontiguousarray(value[marker_indexes]) for name, value in f0.items()
              if name != "sample_key_sha256" and name != "F0"}
    result["sample_key_sha256"] = np.ascontiguousarray(f0["sample_key_sha256"])
    result["F0"] = np.ascontiguousarray(f0["F0"][:, :, marker_indexes, :], dtype="<f4")
    return result


def equivalence_digest(selected, target, reference, f0, marker_cm, root_seed,
                       bridge_receipt_sha256, marker_count: int,
                       limit_bytes: int) -> tuple[str, int, int, int, str, float]:
    marker_indexes = stratified_markers(selected["cM"], marker_cm, marker_count)
    marker_cm_kat = np.ascontiguousarray(marker_cm[marker_indexes], dtype="<f8")
    f0_kat = subset_f0(f0, marker_indexes)
    intervals = materialize.build_interval_table(selected["cM"], marker_cm_kat)
    maxima = {name: int(reference["callable_an"][index].max())
              for index, name in enumerate(("AFR", "EUR", "ASIA"))}
    digest = hashlib.sha256()
    total_tokens = total_shards = max_valid_tokens = 0
    sample_count = target["sample_key_sha256"].size
    for person_start in range(0, sample_count, materialize.PERSON_BATCH):
        person_end = min(person_start + materialize.PERSON_BATCH, sample_count)
        prepared = materialize.prepare_person_batch_channels(
            target, reference, maxima, person_start, person_end, root_seed=root_seed,
            rotation_id="TECHNICAL_KAT",
            fit_normalization_manifest_sha256=bridge_receipt_sha256,
        )
        for radius in materialize.RADII:
            plan = materialize.plan_lazy_marker_chunks(
                intervals, selected["cM"], marker_cm_kat, radius,
                person_end - person_start,
            )
            for chunk in plan:
                start, end = int(chunk["marker_start"]), int(chunk["marker_end_exclusive"])
                common = dict(
                    prepared_channels=prepared, expected_root_seed=root_seed,
                    expected_rotation_id="TECHNICAL_KAT",
                    expected_fit_normalization_manifest_sha256=bridge_receipt_sha256,
                )
                eager = materialize.build_packed_shard(
                    selected, target, reference, f0_kat, marker_cm_kat, maxima, radius,
                    person_start, person_end, start, end,
                )
                lazy = materialize.build_lazy_packed_shard(
                    selected, target, reference, f0_kat, marker_cm_kat, intervals, maxima,
                    radius, person_start, person_end, start, end, **common,
                )
                require(set(eager) == set(lazy), "eager/lazy member inventory differs")
                for name in sorted(eager):
                    require(eager[name].dtype == lazy[name].dtype and
                            eager[name].shape == lazy[name].shape and
                            eager[name].tobytes(order="C") == lazy[name].tobytes(order="C"),
                            f"eager/lazy array differs: {name}")
                    digest.update(name.encode())
                    digest.update(array_sha256(lazy[name]).encode())
                total_tokens += int(lazy["rare_tokens"].shape[0])
                total_shards += 1
                max_valid_tokens = max(max_valid_tokens, int(lazy["rare_tokens"].shape[0]))
                enforce_memory_gate(limit_bytes)
    return (digest.hexdigest(), total_tokens, total_shards, max_valid_tokens,
            array_sha256(marker_indexes), enforce_memory_gate(limit_bytes))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-label", choices=sorted(ROOTS), required=True)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--technical-dir", type=Path, required=True)
    parser.add_argument("--independent-verify-receipt", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--marker-count", type=int, default=512)
    parser.add_argument("--source-auth", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(ROOTS[args.root_label] == args.root_seed, "root label/seed differs")
    validate_marker_count(args.marker_count)
    source_auth_sha = validate_source_auth(
        args.source_auth, args.implementation_commit, args.source_root,
    )
    started = time.monotonic()
    limit_bytes = memory_limit_bytes()
    passes = []
    locus_axis_sha = bridge_sha = None
    selected = target = None
    for _pass_index in range(2):
        loaded = load_technical(args.technical_dir, args.root_label, args.root_seed,
                                args.independent_verify_receipt, args.genetic_map)
        selected, target, reference, f0, marker_cm, bridge_sha, locus_axis_sha = loaded
        passes.append(equivalence_digest(
            selected, target, reference, f0, marker_cm, args.root_seed, bridge_sha,
            args.marker_count, limit_bytes,
        ))
        del reference, f0, marker_cm, loaded
        gc.collect()
    require(passes[0][:5] == passes[1][:5], "two independently reloaded passes differ")
    first = passes[0]
    receipt = {
        "stage": "M33_M0_FACTORIZED_LAZY_TECHNICAL_KAT",
        "status": "PASS_TECHNICAL_KAT_NON_CONSUMABLE",
        "root_label": args.root_label, "root_seed": args.root_seed,
        "marker_count": args.marker_count, "radii_cM": list(materialize.RADII),
        "target_count": int(target["sample_key_sha256"].size),
        "rare_locus_count": int(selected["locus_id"].size),
        "equivalence_semantic_sha256": first[0], "valid_tokens": first[1],
        "shard_count": first[2], "maximum_valid_tokens_per_shard": first[3],
        "marker_index_semantic_sha256": first[4],
        "two_independently_reloaded_passes_equal": True,
        "truth_read": False, "scientific_selection": False, "consumable": False,
        "expanded_arrays_persisted": False, "training": False,
        "bridge_receipt_sha256": bridge_sha,
        "independent_verify_receipt_sha256": sha256_file(args.independent_verify_receipt),
        "genetic_map_sha256": sha256_file(args.genetic_map),
        "technical_locus_key_axis_semantic_sha256": locus_axis_sha,
        "technical_locus_id_projection": "first_16_hex_to_uint64_collision_checked",
        "implementation_commit": args.implementation_commit,
        "source_auth_sha256": source_auth_sha,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2),
        "memory_limit_gib": limit_bytes / (1024.0 ** 3),
        "peak_rss_fraction": max(passes[0][5], passes[1][5]),
        "memory_warning": max(passes[0][5], passes[1][5]) >= 0.7,
        "memory_stop_fraction": 0.8,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
