#!/usr/bin/env python3
"""Truth-free M39 chr22 FIT technical bridge; invoked by Nextflow.

Common haplotype neighbours are selected independently of rare genotypes.
Attached rare values are *diploid reference features*, never phased alleles.
The authenticated M34 parser and deterministic writers are reused verbatim.
No truth, predictions, training, cloud discovery or biological TEST loader exists.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np

from m33_safe_bridge_core import reopen_npz, sample_key, write_deterministic_npz, write_exclusive_json
from m34_generate_mosaics import read_genetic_map
from m34_prepare_panel_factors import (
    ANCESTRIES, _canonical_key, _gt_from_sample_field, _locus_id,
    iter_vcf_records, load_split_contract, open_hashed_text, read_vcf_header,
    require, sha256_file,
)

SCHEMA = "m39-carrier-context-bridge-v1"
INPUT_NAMES = frozenset(("reference_vcf", "target_vcf", "selected_loci",
                         "target_rare", "reference_summary", "genetic_map",
                         "split_manifest"))
COMMON_FEATURES = ("mismatch_rate", "joint_called_fraction",
                   "log1p_joint_called", "log1p_window_markers")
RARE_SEMANTICS = "diploid_reference_feature_not_phased_allele"
# A safety ceiling for this technical pilot, not a scientific hyperparameter.
ARRAY_BUDGET_BYTES = 750_000_000


def load_config(path: Path, input_dir: Path) -> tuple[dict, dict[str, Path]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    require(set(config) == {"schema_version", "scope", "inputs", "parameters", "claims"},
            "configuration fields differ from the contract")
    require(config["schema_version"] == SCHEMA and
            config["scope"] == "technical_only_chr22_R0_FIT", "wrong bridge scope")
    require(set(config["inputs"]) == INPUT_NAMES, "only the seven bridge inputs are allowed")
    p = config["parameters"]
    required = {"chromosome", "expected_reference_people", "expected_target_people",
                "expected_full_loci", "expected_excluded_loci", "loci", "K", "radii_cm",
                "common_maf_min", "anchor_selection", "frequency_role", "minor_orientation",
                "rare_phase", "target_partition"}
    require(set(p) == required, "parameter fields differ from the contract")
    fixed = {"chromosome": "22", "anchor_selection": "evenly_spaced_selected_locus_indices",
             "frequency_role": "REF_TRAIN",
             "minor_orientation": "REF_TRAIN_ALT_count_le_REF_count_tie_ALT",
             "rare_phase": "discarded_diploid_dosage_only", "target_partition": "FIT",
             "common_maf_min": 0.01}
    require(all(p[k] == v for k, v in fixed.items()), "frozen biological policy differs")
    for name in ("expected_reference_people", "expected_target_people", "expected_full_loci",
                 "expected_excluded_loci", "loci", "K"):
        require(type(p[name]) is int and p[name] > 0, "counts must be positive integers")
    require(p["loci"] <= p["expected_excluded_loci"], "anchor count exceeds selected loci")
    radii = p["radii_cm"]
    require(isinstance(radii, list) and radii and
            all(type(r) in (int, float) and np.isfinite(r) and 0 < r <= 1 for r in radii)
            and len(set(radii)) == len(radii), "invalid or repeated cM radii")
    paths = {}
    for name in sorted(INPUT_NAMES):
        descriptor = config["inputs"][name]
        require(set(descriptor) == {"path", "uri", "sha256"}, "invalid input descriptor")
        basename = descriptor["path"]
        require(isinstance(basename, str) and Path(basename).name == basename and basename,
                "input paths must be staged basenames")
        source = input_dir / basename
        require(source.is_file() and not source.is_symlink(), "input missing or symlinked")
        expected = descriptor["sha256"]
        require(isinstance(expected, str) and len(expected) == 64 and
                all(c in "0123456789abcdef" for c in expected), "invalid SHA256")
        require(sha256_file(source) == expected, f"input hash mismatch: {name}")
        paths[name] = source
    require(len({p.resolve() for p in paths.values()}) == len(INPUT_NAMES), "inputs are aliased")
    return config, paths


def read_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == required, "NPZ fields differ from canonical contract")
        arrays = {k: archive[k].copy() for k in archive.files}
    require(all(a.dtype.kind != "O" for a in arrays.values()), "object arrays forbidden")
    return arrays


def load_factors(paths: dict[str, Path], p: dict, genetic_map) -> tuple[dict, dict, dict]:
    selected = read_npz(paths["selected_loci"], {"locus_id", "chrom", "pos", "ref", "alt", "cM"})
    target = read_npz(paths["target_rare"], {"sample_key_sha256", "locus_id", "minor_dosage", "observed_mask"})
    ref = read_npz(paths["reference_summary"], {"ancestry", "locus_id", "minor_ac", "callable_an",
                                               "minor_af", "observed_mask", "no_support"})
    size = p["expected_excluded_loci"]
    require(all(a.shape == (size,) for a in selected.values()), "selected locus shape differs")
    require(selected["locus_id"].dtype.kind == "u" and selected["pos"].dtype.kind in "iu",
            "selected integer axes have wrong dtype")
    require(len(set(map(int, selected["locus_id"]))) == size and
            np.all(selected["chrom"] == 22), "selected IDs duplicate or chromosome differs")
    require(np.all(np.isfinite(selected["cM"])), "nonfinite selected cM")
    order = np.lexsort((selected["locus_id"], selected["pos"], selected["cM"]))
    require(np.array_equal(order, np.arange(size)), "selected axis not canonical cM/pos/ID order")
    keys = []
    for index in range(size):
        bp = int(selected["pos"][index])
        r, a = (selected[k][index].decode("ascii") for k in ("ref", "alt"))
        require(len(r) == len(a) == 1 and r in "ACGT" and a in "ACGT" and r != a,
                "selected allele is not a biallelic ACGT SNV")
        require(int(selected["locus_id"][index]) == _locus_id("22", bp, r, a),
                "selected locus ID differs from canonical allele key")
        require(abs(float(selected["cM"][index]) - genetic_map.bp_to_cm(bp)) <= 1e-12,
                "selected cM differs from genetic map")
        keys.append((bp, r, a))
    require(len(set(keys)) == size, "selected allele keys duplicated")
    for factor in (target, ref):
        require(np.array_equal(factor["locus_id"], selected["locus_id"]), "canonical NPZ locus axes differ")
    require(np.array_equal(ref["ancestry"], np.asarray(ANCESTRIES, dtype="S4")), "ancestry order differs")
    require(all(ref[k].shape == (3, size) for k in ref if k not in {"ancestry", "locus_id"}),
            "reference summary shapes differ")
    require(target["minor_dosage"].shape == target["observed_mask"].shape ==
            (p["expected_target_people"], size), "target dosage shapes differ")
    require(target["sample_key_sha256"].shape == (p["expected_target_people"],), "target sample axis differs")
    return selected, target, ref


def parse_states(record, header, label: str) -> np.ndarray:
    # Exactly the previously validated phased, biallelic, diploid M34 GT parser.
    states = [_gt_from_sample_field(record.fields[8], value, label=label,
                                    sample=sample, line_number=record.line_number)[1]
              for sample, value in zip(header.samples, record.fields[9:])]
    return np.asarray([[(-1 if x is None else x) for x in pair] for pair in states], dtype=np.int8)


def validate_headers(ref_header, target_header, split, p: dict) -> np.ndarray:
    require(len(ref_header.samples) == p["expected_reference_people"] and
            set(ref_header.samples) == set(split.reference_ancestry), "REF_TRAIN header membership differs")
    require(len(target_header.samples) == p["expected_target_people"], "TARGET sample count differs")
    require(not (set(target_header.samples) & split.all_biological_samples),
            "TARGET must not contain biological panel samples")
    require(all(s.startswith("M34_R0_FIT_") and s.removeprefix("M34_R0_FIT_").isdigit()
                for s in target_header.samples), "only R0 FIT synthetic target IDs are allowed")
    for header, role in ((ref_header, "REFERENCE_REF_TRAIN"),
                         (target_header, "TARGET_SOURCE_VALID_MOSAICS")):
        required = {"##m34_bridge_scope=exploratory_only", f"##m34_bridge_vcf_role={role}",
                    "##m34_reference_and_frequency_role=REF_TRAIN",
                    "##m34_mosaic_donor_role_upstream=SOURCE_VALID",
                    "##m34_non_reference_panel_genotypes_opened=false",
                    "##m34_source_test_mosaic_donors_upstream=false"}
        require(required <= set(header.metadata), "M34 VCF role metadata missing or wrong")
        for line in required:
            prefix = line.split("=", 1)[0] + "="
            require(sum(x.startswith(prefix) for x in header.metadata) == 1,
                    "duplicated or contradictory M34 role metadata")
    return np.asarray([ANCESTRIES.index(split.reference_ancestry[s]) for s in ref_header.samples], dtype=np.int8)


def extract_inputs(paths: dict[str, Path], config: dict) -> dict:
    p = config["parameters"]
    split = load_split_contract(paths["split_manifest"])
    genetic_map = read_genetic_map(paths["genetic_map"], "22")
    require(np.all(np.isfinite(genetic_map.positions_cm)), "nonfinite map cM")
    selected, target_factor, ref_factor = load_factors(paths, p, genetic_map)
    locus_index = {(int(bp), r.decode(), a.decode()): i for i, (bp, r, a) in
                   enumerate(zip(selected["pos"], selected["ref"], selected["alt"]))}
    nr, nq, ns = p["expected_reference_people"], p["expected_target_people"], p["expected_excluded_loci"]
    common_ref, common_target, common_cm = [], [], []
    rare_ref = np.full((ns, nr, 2), -1, dtype=np.int8)
    rare_target = np.full((ns, nq, 2), -1, dtype=np.int8)
    seen_rare = set()
    counts = {"full_loci": 0, "excluded_selected_loci": 0, "common_loci": 0,
              "non_common_loci": 0, "reference_missing_common_alleles": 0,
              "target_missing_common_alleles": 0}
    with open_hashed_text(paths["reference_vcf"]) as (rh, rm), \
            open_hashed_text(paths["target_vcf"]) as (qh, qm):
        ref_header = read_vcf_header(rh, "REF")
        query_header = read_vcf_header(qh, "TARGET")
        ancestry = validate_headers(ref_header, query_header, split, p)
        query_keys = np.asarray([sample_key(s) for s in query_header.samples], dtype="S64")
        require(np.array_equal(query_keys, target_factor["sample_key_sha256"]), "TARGET hashed sample axis differs")
        previous = None
        for rr, qr in itertools.zip_longest(iter_vcf_records(rh, ref_header, "REF"),
                                            iter_vcf_records(qh, query_header, "TARGET")):
            require(rr is not None and qr is not None, "VCF row counts differ")
            rkey = _canonical_key(rr.fields, "22", label="REF", line_number=rr.line_number)
            qkey = _canonical_key(qr.fields, "22", label="TARGET", line_number=qr.line_number)
            require(rkey == qkey and rr.fields[0:2] == qr.fields[0:2] and
                    rr.fields[3:5] == qr.fields[3:5], "REF/TARGET allele axes differ")
            key = rkey[1:]
            require(previous is None or key > previous, "VCF allele axis duplicated or unsorted")
            previous = key
            require(len(rkey[2]) == len(rkey[3]) == 1 and rkey[2] in "ACGT" and
                    rkey[3] in "ACGT" and rkey[2] != rkey[3], "full M34 VCF must contain biallelic SNVs")
            counts["full_loci"] += 1
            require(counts["full_loci"] <= p["expected_full_loci"], "VCF exceeds expected locus count")
            ref_states = parse_states(rr, ref_header, "REF")
            if key in locus_index:
                index = locus_index[key]
                require(index not in seen_rare, "selected locus repeats in VCF")
                rare_ref[index] = ref_states
                rare_target[index] = parse_states(qr, query_header, "TARGET")
                seen_rare.add(index)
                counts["excluded_selected_loci"] += 1
                continue
            an = int((ref_states >= 0).sum())
            alt_ac = int((ref_states == 1).sum())
            if not an or not p["common_maf_min"] <= alt_ac / an <= 1 - p["common_maf_min"]:
                counts["non_common_loci"] += 1
                continue
            query_states = parse_states(qr, query_header, "TARGET")
            common_ref.append(ref_states)
            common_target.append(query_states)
            common_cm.append(genetic_map.bp_to_cm(rkey[1]))
            counts["common_loci"] += 1
            counts["reference_missing_common_alleles"] += int((ref_states < 0).sum())
            counts["target_missing_common_alleles"] += int((query_states < 0).sum())
            # Include list-to-contiguous copies and a conservative scoring workspace.
            require(4 * len(common_ref) * (nr + nq) + 200_000_000 < ARRAY_BUDGET_BYTES,
                    "scaffold exceeds pilot array memory ceiling")
        require(rm.digest.hexdigest() == config["inputs"]["reference_vcf"]["sha256"] and
                qm.digest.hexdigest() == config["inputs"]["target_vcf"]["sha256"],
                "VCF changed during authenticated parsing")
    require(counts["full_loci"] == p["expected_full_loci"] and len(seen_rare) == ns,
            "full VCF or selected locus count differs")
    ref_ac_alt = (rare_ref == 1).sum(axis=(1, 2))
    ref_ac_ref = (rare_ref == 0).sum(axis=(1, 2))
    require(np.all(ref_ac_alt + ref_ac_ref > 0), "rare locus has no REF alleles")
    minor_is_alt = ref_ac_alt <= ref_ac_ref
    minor = minor_is_alt.astype(np.int8)
    ref_observed = np.all(rare_ref >= 0, axis=2)
    query_observed = np.all(rare_target >= 0, axis=2)
    ref_dosage = np.where(ref_observed, (rare_ref == minor[:, None, None]).sum(axis=2), 0).astype(np.int8)
    query_dosage = np.where(query_observed, (rare_target == minor[:, None, None]).sum(axis=2), 0).astype(np.int8)
    ac = np.stack([(rare_ref[:, ancestry == a, :] == minor[:, None, None]).sum(axis=(1, 2))
                   for a in range(3)])
    an = np.stack([(rare_ref[:, ancestry == a, :] >= 0).sum(axis=(1, 2)) for a in range(3)])
    af = np.divide(ac, an, out=np.zeros_like(ac, dtype=np.float64), where=an > 0)
    for key, actual in {"minor_ac": ac, "callable_an": an, "minor_af": af,
                        "observed_mask": an > 0, "no_support": (an > 0) & (ac == 0)}.items():
        require(np.array_equal(actual, ref_factor[key]), f"canonical REF reconciliation failed: {key}")
    require(np.array_equal(query_dosage.T, target_factor["minor_dosage"]) and
            np.array_equal(query_observed.T, target_factor["observed_mask"]),
            "canonical TARGET dosage/mask reconciliation failed")
    counts["reference_partial_rare_genotypes"] = int(((rare_ref >= 0).sum(axis=2) == 1).sum())
    counts["target_partial_rare_genotypes"] = int(((rare_target >= 0).sum(axis=2) == 1).sum())
    counts["minor_is_ref_selected_loci"] = int((~minor_is_alt).sum())
    return {"selected": selected, "reference_ancestry": ancestry,
            "reference_sample_key_sha256": np.asarray([sample_key(s) for s in ref_header.samples], dtype="S64"),
            "sample_key_sha256": query_keys, "minor_is_alt": minor_is_alt,
            "common_ref": np.stack(common_ref) if common_ref else np.empty((0, nr, 2), dtype=np.int8),
            "common_target": np.stack(common_target) if common_target else np.empty((0, nq, 2), dtype=np.int8),
            "common_cm": np.asarray(common_cm, dtype=np.float64),
            "ref_dosage": ref_dosage, "ref_observed": ref_observed,
            "query_dosage": query_dosage.T, "query_observed": query_observed.T,
            "counts": counts}


def anchor_indices(size: int, count: int) -> np.ndarray:
    require(0 < count <= size, "invalid anchor count")
    return np.linspace(0, size - 1, count, dtype=np.int64)


def pooled_summary(data: dict, anchors: np.ndarray) -> np.ndarray:
    nr = len(data["reference_ancestry"])
    require(data["ref_dosage"].shape[1] == nr, "reference axis differs")
    summary = np.zeros((len(anchors), 3, 4), dtype=np.float32)
    for a in range(3):
        mask = data["reference_ancestry"] == a
        observed = data["ref_observed"][anchors][:, mask]
        dosage = data["ref_dosage"][anchors][:, mask]
        n = observed.sum(axis=1)
        for g in range(3):
            summary[:, a, g] = np.divide(((dosage == g) & observed).sum(axis=1), n,
                                        out=np.zeros(len(anchors), dtype=np.float64), where=n > 0)
        summary[:, a, 3] = n / int(mask.sum())
    return np.broadcast_to(summary, (len(data["sample_key_sha256"]), *summary.shape)).copy()


def materialize_radius(data: dict, anchors: np.ndarray, k: int, radius: float) -> tuple[dict, dict]:
    """Select K haplotypes per ancestry solely by called common mismatch rate.

    Equal mismatch rates use SHA256(sample-key|haplotype), never row order or
    rare-carrier status. Unsupported pairs are padded, not arbitrarily selected.
    Integer counts use float32 BLAS (exact for the bounded <2**24 marker axis).
    """
    nq, nr = len(data["sample_key_sha256"]), len(data["reference_sample_key_sha256"])
    shape = (nq, len(anchors), 2, 3, k)
    require(np.prod(shape, dtype=np.int64) * 40 < ARRAY_BUDGET_BYTES, "output tensor exceeds memory ceiling")
    common = np.zeros((*shape, len(COMMON_FEATURES)), dtype=np.float32)
    candidate = np.full(shape, -1, dtype=np.int32)
    hap = np.full(shape, -1, dtype=np.int8)
    valid = np.zeros(shape, dtype=np.uint8)
    ref_dosage = np.zeros(shape, dtype=np.int8)
    observed = np.zeros(shape, dtype=np.uint8)
    ref_hap_ancestry = np.repeat(data["reference_ancestry"], 2)
    tie_hashes = [hashlib.sha256(bytes(key) + b"|" + str(h).encode()).hexdigest()
                  for key in data["reference_sample_key_sha256"] for h in range(2)]
    tie_rank = np.empty(2 * nr, dtype=np.int64)
    tie_rank[np.argsort(tie_hashes)] = np.arange(2 * nr)
    window_counts = []
    for j, locus in enumerate(anchors):
        window = np.abs(data["common_cm"] - data["selected"]["cM"][locus]) <= radius
        count = int(window.sum())
        window_counts.append(count)
        if count == 0:
            continue
        require(count < 2**24, "window exceeds exact float32 count range")
        r = data["common_ref"][window].reshape(count, 2 * nr).T
        q = data["common_target"][window].reshape(count, 2 * nq).T
        r0, r1 = (r == 0).astype(np.float32), (r == 1).astype(np.float32)
        q0, q1 = (q == 0).astype(np.float32), (q == 1).astype(np.float32)
        joint = (q0 + q1) @ (r0 + r1).T
        mismatch = q0 @ r1.T + q1 @ r0.T
        rates = np.divide(mismatch, joint, out=np.full(joint.shape, np.inf, dtype=np.float64), where=joint > 0)
        for a in range(3):
            pool = np.flatnonzero(ref_hap_ancestry == a)
            for qh in range(2 * nq):
                supported = pool[joint[qh, pool] > 0]
                chosen = supported[np.lexsort((tie_rank[supported], rates[qh, supported]))[:k]]
                if len(chosen) == 0:
                    continue
                index = (qh // 2, j, qh % 2, a, slice(0, len(chosen)))
                persons = chosen // 2
                candidate[index], hap[index], valid[index] = persons, chosen % 2, 1
                common[index] = np.stack((rates[qh, chosen], joint[qh, chosen] / count,
                                          np.log1p(joint[qh, chosen]),
                                          np.full(len(chosen), np.log1p(count))), axis=-1)
                ref_dosage[index] = data["ref_dosage"][locus, persons]
                observed[index] = data["ref_observed"][locus, persons]
    arrays = {"common_context": common, "candidate_ref_index": candidate, "candidate_hap": hap,
              "candidate_mask": valid, "ref_dosage": ref_dosage, "rare_ref_observed": observed,
              "query_dosage": data["query_dosage"][:, anchors],
              "query_observed": data["query_observed"][:, anchors].astype(np.uint8),
              "pooled_summary": pooled_summary(data, anchors),
              "reference_sample_key_sha256": data["reference_sample_key_sha256"],
              "sample_key_sha256": data["sample_key_sha256"],
              "reference_ancestry": data["reference_ancestry"],
              "ancestry": np.asarray(ANCESTRIES, dtype="S4"), "anchor_indices": anchors,
              "minor_is_alt": data["minor_is_alt"][anchors].astype(np.uint8),
              "common_feature_names": np.asarray(COMMON_FEATURES, dtype="S32"),
              "rare_semantics": np.asarray([RARE_SEMANTICS], dtype="S64"),
              "radius_cm": np.asarray([radius], dtype=np.float64)}
    arrays.update({key: value[anchors] for key, value in data["selected"].items()})
    unique_counts, selected_carriers = [], []
    for q, j, h, a in np.ndindex(nq, len(anchors), 2, 3):
        persons = np.unique(candidate[q, j, h, a][valid[q, j, h, a].astype(bool)])
        unique_counts.append(len(persons))
        selected_carriers.append(int(((data["ref_dosage"][anchors[j], persons] > 0) &
                                       data["ref_observed"][anchors[j], persons]).sum()))
    global_carriers = [int(((data["ref_dosage"][locus] > 0) & data["ref_observed"][locus] &
                           (data["reference_ancestry"] == a)).sum())
                       for locus in anchors for a in range(3)]

    def count_summary(values):
        return {"groups": len(values), "min": min(values), "max": max(values),
                "mean": float(np.mean(values)), "sum": sum(values)}

    stats = {"radius_cm": radius, "tensor_shape": list(common.shape),
             "array_bytes": sum(a.nbytes for a in arrays.values()),
             "candidate_slots": int(valid.size), "supported_candidate_slots": int(valid.sum()),
             "zero_common_window_anchors": sum(x == 0 for x in window_counts),
             "window_markers_min": min(window_counts), "window_markers_max": max(window_counts),
             "window_markers_median": float(np.median(window_counts)),
             "supported_rare_observed_slots": int((valid & observed).sum()),
             "unique_ref_people_per_query_locus_homolog_ancestry": count_summary(unique_counts),
             "selected_rare_carriers_per_query_locus_homolog_ancestry": count_summary(selected_carriers),
             "global_rare_carriers_per_locus_ancestry": count_summary(global_carriers),
             "carrier_counter_semantics": "observed_diploid_dosage_gt_zero_unique_person_per_named_group"}
    return arrays, stats


def run_bridge(config_path: Path, input_dir: Path, outdir: Path) -> dict:
    start = time.monotonic()
    require(not outdir.exists() and not outdir.is_symlink(), "output directory already exists")
    config, paths = load_config(config_path, input_dir)
    authenticated = time.monotonic()
    data = extract_inputs(paths, config)
    extracted = time.monotonic()
    anchors = anchor_indices(config["parameters"]["expected_excluded_loci"], config["parameters"]["loci"])
    outdir.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation + sibling staging: an observer only sees a complete directory.
    lock_path = outdir.parent / ("." + outdir.name + ".m39.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    staging = Path(tempfile.mkdtemp(prefix=".m39-stage-", dir=outdir.parent))
    try:
        profiles = []
        for radius in config["parameters"]["radii_cm"]:
            t = time.monotonic()
            arrays, stats = materialize_radius(data, anchors, config["parameters"]["K"], float(radius))
            filename = f"radius_{radius:g}cm/features.npz"
            output = staging / filename
            write_deterministic_npz(output, arrays)
            reopen_npz(output, arrays)
            stats.update({"path": filename, "sha256": sha256_file(output), "bytes": output.stat().st_size,
                          "elapsed_seconds": time.monotonic() - t})
            profiles.append(stats)
            del arrays
        # Detect replacement of small authenticated inputs during processing, too.
        require(all(sha256_file(paths[k]) == config["inputs"][k]["sha256"] for k in INPUT_NAMES),
                "authenticated input changed during bridge")
        receipt = {"schema_version": SCHEMA, "decision": "PASS_TECHNICAL_BRIDGE_ONLY",
                   "scope": config["scope"], "parameters": config["parameters"],
                   "claims": config["claims"], "inputs": {
                       k: {**config["inputs"][k], "bytes": paths[k].stat().st_size}
                       for k in sorted(INPUT_NAMES)},
                   "config_sha256": sha256_file(config_path),
                   "code_sha256": {Path(__file__).name: sha256_file(Path(__file__)),
                                    "m34_prepare_panel_factors.py": sha256_file(Path(__file__).with_name("m34_prepare_panel_factors.py")),
                                    "m34_generate_mosaics.py": sha256_file(Path(__file__).with_name("m34_generate_mosaics.py")),
                                    "m33_safe_bridge_core.py": sha256_file(Path(__file__).with_name("m33_safe_bridge_core.py"))},
                   "privacy": {"sample_identifiers_or_keys_in_json": False,
                               "private_npz_contains_hashed_sample_keys": True},
                   "boundaries": {"truth_opened": False, "predictions_opened": False,
                                  "validation_targets_opened": False, "source_test_genotypes_opened": False,
                                  "reference_genotype_role": "REF_TRAIN", "target_partition": "R0_FIT",
                                  "upstream_mosaic_donor_role": "SOURCE_VALID",
                                  "rare_semantics": RARE_SEMANTICS,
                                  "rare_partial_gt_policy": "diploid_dosage_zero_observed_false",
                                  "common_partial_gt_policy": "use_observed_phased_alleles_per_homolog",
                                  "mask_storage": "uint8_0_or_1_consumer_must_cast_to_bool",
                                  "pooled_summary_unit": "reference_diploid_person_not_topK_haplotype",
                                  "retrieval_inputs": "common_genotypes_and_REF_ancestry_only"},
                   "reconciliation": {"all_selected_loci": True, "reference_ac_an_af_masks": True,
                                      "target_diploid_dosage_mask": True},
                   "counts": data["counts"], "profiles": profiles,
                   "timings_seconds": {"authenticate": authenticated - start,
                                       "extract_and_reconcile": extracted - authenticated,
                                       "total": time.monotonic() - start},
                   "resources": {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 "user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                                 "system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                                 "numpy_version": np.__version__}}
        write_exclusive_json(staging / "receipt.json", receipt)
        require(not outdir.exists() and not outdir.is_symlink(), "output appeared during materialization")
        os.rename(staging, outdir)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only the task-created exact temporary directory.
        lock_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path.cwd())
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_bridge(args.config, args.input_dir, args.outdir)
    print(json.dumps({"decision": receipt["decision"], "counts": receipt["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
