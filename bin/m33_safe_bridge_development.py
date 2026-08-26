#!/usr/bin/env python3
"""Build productive, truth-free M33 factors for one DEVELOPMENT root.

The bridge resolves the minor allele inside its boundary, exports diploid TARGET
dosage, aggregated REF summaries, three same-locus TARGET shams, three REF-label
shams and sanitized FLARE probabilities.  Raw phase, identities, FREQ metrics,
donor genotypes and local-ancestry truth are not exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import m33_safe_bridge_core as core
from m31_ordered_linear import load_genetic_map, load_ordered_rare, load_ref_minor_dosage
from m33_materialize import validate_inputs
from m33_safe_bridge_core import reopen_npz, semantic_arrays_sha256, write_deterministic_npz
from m33_safe_bridge_technical_kat import load_f0_projection


TARGET_SHAM_SEEDS = (1277457345, 943666774, 1858042568)
REF_SHAM_SEEDS = core.REF_LABEL_SHAM_SEEDS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def locus_id(position: int) -> int:
    return int.from_bytes(hashlib.sha256(f"22:{position}:A:C".encode()).digest()[:8], "little")


def ref_nodes_by_person(pool_manifest: Path) -> dict[str, tuple[int, int]]:
    grouped: dict[str, list[int]] = {}
    with pool_manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["role"] == "REF_LAI":
                grouped.setdefault(row["individual_id"], []).append(int(row["node_id"]))
    result = {person: tuple(sorted(nodes)) for person, nodes in grouped.items()}
    require(len(result) == 90 and all(len(nodes) == 2 for nodes in result.values()),
            "REF person/node mapping differs")
    return result


def target_shams(dosage: np.ndarray, observed: np.ndarray, root_seed: int) -> dict[int, dict[str, np.ndarray]]:
    require(dosage.shape == observed.shape and dosage.ndim == 2, "TARGET axes differ")
    sample_count, locus_count = dosage.shape
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for seed in TARGET_SHAM_SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, root_seed]))
        sham_dose = np.empty_like(dosage)
        sham_mask = np.empty_like(observed)
        nonidentity = 0
        for locus in range(locus_count):
            permutation = rng.permutation(sample_count)
            if np.array_equal(permutation, np.arange(sample_count)):
                permutation = np.roll(permutation, 1)
            nonidentity += int(not np.array_equal(permutation, np.arange(sample_count)))
            sham_dose[:, locus] = dosage[permutation, locus]
            sham_mask[:, locus] = observed[permutation, locus]
            before = Counter(zip(dosage[:, locus].tolist(), observed[:, locus].tolist()))
            after = Counter(zip(sham_dose[:, locus].tolist(), sham_mask[:, locus].tolist()))
            require(before == after, "TARGET sham changed the within-locus joint multiset")
        require(nonidentity == locus_count, "TARGET sham contains an identity locus permutation")
        outputs[seed] = {
            "minor_dosage": np.ascontiguousarray(sham_dose),
            "observed_mask": np.ascontiguousarray(sham_mask),
        }
    return outputs


def run(args: argparse.Namespace) -> dict:
    pre4 = json.loads(args.pre4.read_text(encoding="utf-8"))
    require(args.root_seed in pre4["root_registry"]["DEVELOPMENT"],
            "root is not registered for DEVELOPMENT")
    require(tuple(pre4["controls"]["target_same_locus_sham"]["seeds"]) == TARGET_SHAM_SEEDS,
            "TARGET-sham seeds drifted")
    require(tuple(pre4["controls"]["REF_label_sham"]["seeds"]) == REF_SHAM_SEEDS,
            "REF-sham seeds drifted")
    require(not args.outdir.exists(), f"refusing to overwrite {args.outdir}")

    rare = load_ordered_rare(args.m31_sites, args.m31_target, args.root_seed)
    f0_loci, f0_values = load_f0_projection(args.flare_anc, rare.samples)
    flare_positions = {position for position, _ref, _alt in f0_loci}
    keep = np.asarray([int(position) not in flare_positions for position in rare.positions], dtype=bool)
    require(keep.any(), "incremental rare universe is empty")
    positions = np.asarray(rare.positions[keep], dtype="<i8")
    ids = np.asarray([locus_id(int(position)) for position in positions], dtype="<u8")
    require(np.unique(ids).size == ids.size, "locus-id collision")
    genetic_map = load_genetic_map(args.genetic_map, "22")
    cms = np.asarray(genetic_map.cm_at(positions), dtype="<f8")
    order = np.lexsort((ids, positions, cms))
    require(np.array_equal(order, np.arange(len(ids))), "rare locus order differs")

    observed_site_major = np.all(np.isfinite(rare.hap_presence[keep]), axis=2)
    target_site_major = np.where(
        observed_site_major, np.nansum(rare.hap_presence[keep], axis=2), 0,
    ).astype("|i1")
    target_dosage = np.ascontiguousarray(target_site_major.T)
    target_observed = np.ascontiguousarray(observed_site_major.astype("|u1").T)
    sample_keys = np.asarray([core.sample_key(sample) for sample in rare.samples], dtype="|S64")

    ref_dosage, ref_people, ref_labels = load_ref_minor_dosage(
        args.tree_sequence, args.pool_manifest, rare, genetic_map,
    )
    ref_dosage = np.ascontiguousarray(ref_dosage[keep])
    labels = np.asarray(ref_labels, dtype=object)
    require(Counter(ref_labels) == Counter({"AFR": 30, "EUR": 30, "ASIA": 30}),
            "REF ancestry counts differ")
    ref_ac_locus = np.column_stack([
        ref_dosage[:, labels == ancestry].sum(axis=1) for ancestry in ("AFR", "EUR", "ASIA")
    ]).astype("<u2")
    ref_an_locus = np.full(ref_ac_locus.shape, 60, dtype="<u2")
    ref_ac = np.ascontiguousarray(ref_ac_locus.T)
    ref_an = np.ascontiguousarray(ref_an_locus.T)
    ref_af = np.divide(ref_ac, ref_an, out=np.zeros_like(ref_ac, dtype="<f8"), where=ref_an > 0)
    ref_observed = (ref_an > 0).astype("|u1")
    ref_no_support = ((ref_an > 0) & (ref_ac == 0)).astype("|u1")

    selected = {
        "locus_id": ids, "chrom": np.full(len(ids), 22, dtype="|u1"), "pos": positions,
        "ref": np.full(len(ids), b"A", dtype="|S1"), "alt": np.full(len(ids), b"C", dtype="|S1"),
        "cM": cms,
    }
    target = {
        "sample_key_sha256": sample_keys, "locus_id": ids,
        "minor_dosage": target_dosage, "observed_mask": target_observed,
    }
    reference = {
        "ancestry": np.asarray(("AFR", "EUR", "ASIA"), dtype="|S4"), "locus_id": ids,
        "minor_ac": ref_ac, "callable_an": ref_an, "minor_af": ref_af,
        "observed_mask": ref_observed, "no_support": ref_no_support,
    }
    f0 = {
        "sample_key_sha256": sample_keys,
        "marker_chrom": np.full(len(f0_loci), 22, dtype="|u1"),
        "marker_pos": np.asarray([row[0] for row in f0_loci], dtype="<i8"),
        "marker_ref": np.asarray([row[1].encode() for row in f0_loci], dtype="|S1"),
        "marker_alt": np.asarray([row[2].encode() for row in f0_loci], dtype="|S1"),
        "F0": np.ascontiguousarray(f0_values, dtype="<f4"),
    }
    marker_cm = np.asarray(genetic_map.cm_at(f0["marker_pos"]), dtype="<f8")
    validate_inputs(selected, target, reference, f0, marker_cm)

    target_controls = target_shams(target_dosage, target_observed, args.root_seed)
    nodes = ref_nodes_by_person(args.pool_manifest)
    ref_controls, ref_diagnostics = core.summarize_diploid_dosage_reference_label_shams(
        ref_dosage, ref_people, ref_labels, nodes, seeds=REF_SHAM_SEEDS,
        expected_people_by_ancestry={"AFR": 30, "EUR": 30, "ASIA": 30},
    )
    args.outdir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, np.ndarray]] = {
        "selected_loci_incremental.npz": selected,
        "target_rare_diploid_incremental.npz": target,
        "reference_rare_summary_incremental.npz": reference,
        "flare_f0_sanitized.npz": f0,
        "marker_cM.npz": {"marker_cM": marker_cm},
    }
    for seed, arrays in target_controls.items():
        artifacts[f"target_same_locus_sham_{seed}.npz"] = {
            "sample_key_sha256": sample_keys, "locus_id": ids, **arrays,
        }
    for seed, summary in ref_controls.items():
        artifacts[f"reference_label_sham_{seed}.npz"] = {
            "ancestry": np.asarray(("AFR", "EUR", "ASIA"), dtype="|S4"),
            "locus_id": ids, **summary,
        }
    raw_hashes: dict[str, str] = {}
    semantic_hashes: dict[str, str] = {}
    for name, arrays in artifacts.items():
        path = args.outdir / name
        write_deterministic_npz(path, arrays)
        reopen_npz(path, arrays)
        raw_hashes[name] = sha256_file(path)
        semantic_hashes[name] = semantic_arrays_sha256(f"m33_development_{name}_v1", arrays)
    receipt = {
        "schema_version": "1.0.0", "stage": "M33_SAFE_BRIDGE_DEVELOPMENT",
        "status": "PASS_PRODUCTIVE_TRUTH_FREE_FACTORS", "root_seed": args.root_seed,
        "pre4_sha256": sha256_file(args.pre4),
        "counts": {
            "selected_all": int(len(rare.positions)), "selected_incremental": int(keep.sum()),
            "selected_overlap_flare": int((~keep).sum()), "target_people": len(rare.samples),
            "flare_markers": len(f0_loci), "reference_people": len(ref_people),
            "reference_no_support_loci": int(np.all(ref_no_support == 1, axis=0).sum()),
        },
        "input_sha256": {
            "tree_sequence": sha256_file(args.tree_sequence),
            "pool_manifest": sha256_file(args.pool_manifest),
            "m31_sites": sha256_file(args.m31_sites), "m31_target": sha256_file(args.m31_target),
            "flare_anc": sha256_file(args.flare_anc), "genetic_map": sha256_file(args.genetic_map),
        },
        "artifact_raw_sha256": raw_hashes, "artifact_semantic_sha256": semantic_hashes,
        "target_sham_seeds": list(TARGET_SHAM_SEEDS), "ref_sham_seeds": list(REF_SHAM_SEEDS),
        "ref_sham_diagnostics": ref_diagnostics,
        "truth_argument_available": False, "truth_accessed": False,
        "raw_identifiers_exported": False, "rare_phase_exported": False,
        "reopen_verified": True,
    }
    (args.outdir / "safe_bridge.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-seed", type=int, required=True)
    parser.add_argument("--pre4", type=Path, required=True)
    parser.add_argument("--tree-sequence", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--m31-sites", type=Path, required=True)
    parser.add_argument("--m31-target", type=Path, required=True)
    parser.add_argument("--flare-anc", type=Path, required=True)
    parser.add_argument("--genetic-map", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "root_seed": result["root_seed"]}, sort_keys=True))
