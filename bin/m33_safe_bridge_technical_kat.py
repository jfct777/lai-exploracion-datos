#!/usr/bin/env python3
"""Materialize a truth-free, non-consumable SAFE_BRIDGE KAT from technical roots."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import socket
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import m33_safe_bridge_core as bridge_core
from m31_ordered_linear import load_genetic_map, load_ordered_rare, load_ref_minor_dosage
from m31_ordered_rare_preflight import derive_freq_sites
from m33_a0_real_adapter import audit_pools, audit_ref_mapping, audit_tbi, audit_tree_vcf, audit_vcf
from m33_safe_bridge_core import (
    reopen_npz, semantic_arrays_sha256, write_deterministic_npz, write_exclusive_json,
)


STAGE = "M33_SAFE_BRIDGE_TECHNICAL_ROOT_KAT"
STATUS = "PASS_SAFE_BRIDGE_TECHNICAL_ROOT_KAT_ONLY_NON_CONSUMABLE"
ROOTS = {"root17": 20260817, "root18": 20260818}
ANCESTRIES = ("AFR", "EUR", "ASIA")
INPUT_NAMES = {
    "tree_sequence", "pools", "rare_catalog", "rare_haplotypes", "m31_sites",
    "m31_target", "ref_vcf", "ref_tbi", "ref_pairs", "panel_map",
    "genetic_map", "flare_anc", "flare_tbi",
}
SCHEMAS = {
    "technical_kat_selected_loci_incremental.npz":
        "tests_m33_safe_bridge_technical_kat_selected_loci_incremental_v1",
    "technical_kat_target_rare_diploid_incremental.npz":
        "tests_m33_safe_bridge_technical_kat_target_rare_diploid_incremental_v1",
    "technical_kat_reference_rare_summary_incremental.npz":
        "tests_m33_safe_bridge_technical_kat_reference_rare_summary_incremental_v1",
    "technical_kat_flare_f0_sanitized.npz":
        "tests_m33_safe_bridge_technical_kat_flare_f0_sanitized_v1",
}
REQUIRED_SOURCE_PATHS = {
    "bin/m33_safe_bridge_technical_kat.py", "bin/m33_safe_bridge_core.py",
    "bin/m33_a0_real_adapter.py", "bin/m31_ordered_linear.py",
    "bin/m31_ordered_rare_preflight.py",
    "conf/m33_safe_bridge_technical_kat_contract.json",
    "conf/m33_safe_bridge_technical_kat_authorization.json",
    "conf/m33_safe_bridge_technical_kat.config",
    "modules/33_SAFE_BRIDGE_TECHNICAL_KAT.nf",
    "workflows/m33_safe_bridge_technical_kat.nf",
    "tests/test_m33_safe_bridge_technical_kat.py",
    "tests/test_m33_safe_bridge_technical_kat_nextflow.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")),
        object_pairs_hook=unique,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parse_sources(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        relative, separator, staged = value.partition("=")
        require(separator and relative in REQUIRED_SOURCE_PATHS and relative not in result,
                "source mapping is invalid or duplicated")
        result[relative] = Path(staged)
    require(set(result) == REQUIRED_SOURCE_PATHS, "source mapping is incomplete")
    return result


def validate_source_auth(path: Path, sources: dict[str, Path], commit: str) -> str:
    payload = load_json(path)
    require(payload.get("stage") == "M33_SAFE_BRIDGE_TECHNICAL_KAT_SOURCE_AUTH" and
            payload.get("status") == "AUTHORIZED_EXACT_TECHNICAL_KAT_SOURCES",
            "source-auth identity drifted")
    require(payload.get("implementation_commit") == commit, "source-auth commit drifted")
    hashes = payload.get("files", {})
    require(set(hashes) == REQUIRED_SOURCE_PATHS, "source-auth inventory drifted")
    observed = {name: sha256_file(staged) for name, staged in sorted(sources.items())}
    require(observed == hashes, "staged sources differ from source-auth")
    return sha256_file(path)


def validate_contract(contract: dict[str, Any]) -> None:
    require(contract.get("stage") == "M33_SAFE_BRIDGE_TECHNICAL_KAT_CONTRACT" and
            contract.get("status") == "AUTHORIZED_TECHNICAL_ROOT_KAT_NO_MATERIALIZE",
            "technical KAT contract identity drifted")
    require(contract.get("roots") == ROOTS and contract.get("chromosome") == 22,
            "technical root/chromosome scope drifted")
    require(contract.get("expected_ref_people_by_ancestry") == {name: 30 for name in ANCESTRIES},
            "REF firewall counts drifted")
    gates = contract.get("gates", {})
    for key in ("consumable", "scientific_evidence", "truth_read", "materialize_authorized",
                "ready_emitted", "training_authorized", "gcs_write"):
        require(gates.get(key) is False, f"technical contract enables {key}")
    require(contract.get("f0_projection") == {
        "read_only": ["CHROM", "POS", "REF", "ALT", "ANP1", "ANP2"],
        "ignored_source_fields": ["GT", "AN1", "AN2"],
        "forbidden_outputs": ["GT", "AN1", "AN2", "sample_id", "target_rare_phase", "truth"],
    }, "F0 projection drifted")


def validate_authorization(payload: dict[str, Any], root_label: str, root_seed: int,
                           inputs: dict[str, Path]) -> dict[str, str]:
    require(payload.get("stage") == "M33_SAFE_BRIDGE_TECHNICAL_KAT_AUTHORIZATION" and
            payload.get("status") == "AUTHORIZED_EXACT_READ_ONLY_TECHNICAL_ROOTS",
            "authorization identity drifted")
    require(payload.get("execution", {}) == {
        "both_roots_one_invocation": True, "cache": False, "retries": 0,
        "network": False, "credentials": False, "truth": False,
        "materialize": False, "ready": False, "training": False,
    }, "execution authorization drifted")
    root = payload.get("roots", {}).get(root_label)
    require(isinstance(root, dict) and root.get("root_seed") == root_seed,
            "root label/seed is not authorized")
    assets = root.get("assets", {})
    require(set(assets) == INPUT_NAMES and set(inputs) == INPUT_NAMES,
            "authorized input inventory drifted")
    observed: dict[str, str] = {}
    for name, path in sorted(inputs.items()):
        spec = assets[name]
        require(set(spec) == {"uri", "generation", "size_bytes", "sha256", "crc32c_base64", "md5_base64"},
                f"descriptor fields drifted: {name}")
        require(isinstance(spec["uri"], str) and spec["uri"].startswith("gs://") and
                re.fullmatch(r"[0-9]+", spec["generation"]) and
                type(spec["size_bytes"]) is int and spec["size_bytes"] >= 0 and
                re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]), f"descriptor invalid: {name}")
        require(path.is_file() and not path.is_symlink(), f"staged input invalid: {name}")
        require(path.stat().st_size == spec["size_bytes"], f"staged size differs: {name}")
        observed[name] = sha256_file(path)
        require(observed[name] == spec["sha256"], f"staged sha256 differs: {name}")
    return observed


def prove_runner_read_only(paths: dict[str, Path], source_paths: dict[str, Path], sidecars: list[Path]) -> None:
    candidates = list(paths.values()) + list(source_paths.values()) + sidecars
    for path in candidates:
        require(path.is_file() and not path.is_symlink(), f"read-only candidate invalid: {path.name}")
        try:
            with path.open("ab"):
                pass
        except PermissionError:
            pass
        else:
            raise ValueError(f"write probe unexpectedly succeeded: {path.name}")
        try:
            os.chmod(path, 0o600)
        except PermissionError:
            pass
        else:
            raise ValueError(f"chmod probe unexpectedly succeeded: {path.name}")
        try:
            path.rename(path.with_name(f".{path.name}.rename_probe"))
        except PermissionError:
            pass
        else:
            raise ValueError(f"rename probe unexpectedly succeeded: {path.name}")


def network_is_disabled() -> bool:
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    connection.settimeout(0.1)
    try:
        connection.connect(("8.8.8.8", 53))
        return False
    except OSError:
        return True
    finally:
        connection.close()


def locus_key(position: int) -> bytes:
    return hashlib.sha256(f"22:{position}:A:C".encode()).hexdigest().encode("ascii")


def load_rare_crosschecked(inputs: dict[str, Path], root_seed: int):
    selection_contract = {
        "chromosome_domain": {"chrom": "22", "start_bp": 15287922},
        "rare_universe": {
            "minimum_mac": 2, "maximum_maf_exclusive": 0.01,
            "minimum_carrier_individuals": 2,
            "prohibited_selectors": ["TARGET", "truth", "Gnomix_prediction", "FLARE_prediction"],
        },
    }
    selected, _frequency_facts = derive_freq_sites(
        inputs["tree_sequence"], inputs["pools"], inputs["rare_catalog"], selection_contract,
    )
    rare = load_ordered_rare(inputs["m31_sites"], inputs["m31_target"], root_seed)
    require([row.position for row in selected] == rare.positions.tolist(),
            "M31 loci differ from an exhaustive FREQ-only recomputation")
    require([row.minor_code for row in selected] == rare.minor_codes.tolist(),
            "M31 minor orientation differs from the authenticated rare catalog")

    selected_index = {int(position): index for index, position in enumerate(rare.positions)}
    expected_haps = [f"{sample}_h{hap}" for sample in rare.samples for hap in (0, 1)]
    seen: set[int] = set()
    with gzip.open(inputs["rare_haplotypes"], "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and reader.fieldnames[3:] == expected_haps,
                "raw TARGET haplotype axis differs from M31")
        for row in reader:
            position = int(row["position"])
            index = selected_index.get(position)
            if index is None:
                continue
            require(position not in seen and int(row["minor_code"]) == int(rare.minor_codes[index]),
                    "raw TARGET locus is duplicated or differently oriented")
            states = np.asarray([int(row[field]) for field in expected_haps], dtype=np.int8).reshape(len(rare.samples), 2)
            expected = (states == int(rare.minor_codes[index])).astype(np.float64)
            require(np.array_equal(expected, rare.hap_presence[index]),
                    "M31 TARGET presence differs from I(state == minor_code)")
            seen.add(position)
    require(seen == set(selected_index), "raw TARGET haplotypes omit a selected locus")
    return rare


def load_f0_projection(path: Path, expected_samples: tuple[str, ...]) -> tuple[list[tuple[int, str, str]], np.ndarray]:
    ancestry_codes: dict[str, str] = {}
    samples: tuple[str, ...] | None = None
    loci: list[tuple[int, str, str]] = []
    rows: list[np.ndarray] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("##ANCESTRY=<"):
                ancestry_codes = {token.split("=", 1)[1]: token.split("=", 1)[0]
                                  for token in line.strip()[len("##ANCESTRY=<"):-1].split(",")}
                continue
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                samples = tuple(line.rstrip("\n").split("\t")[9:])
                require(samples == expected_samples, "FLARE sample order differs from TARGET")
                continue
            require(samples is not None and ancestry_codes == {"0": "AFR", "1": "EUR", "2": "ASIA"},
                    "FLARE ancestry/sample header drifted")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples) and fields[0].removeprefix("chr") == "22" and
                    fields[3:5] == ["A", "C"], f"FLARE locus malformed at row {line_number}")
            position = int(fields[1])
            require(not loci or position > loci[-1][0], "FLARE loci are not strictly ordered")
            fmt = fields[8].split(":")
            require("ANP1" in fmt and "ANP2" in fmt, "FLARE lacks ANP1/ANP2")
            probability_indexes = (fmt.index("ANP1"), fmt.index("ANP2"))
            matrix = np.empty((len(samples), 2, 3), dtype=np.float64)
            for sample_index, raw in enumerate(fields[9:]):
                values = raw.split(":")
                for haplotype, index in enumerate(probability_indexes):
                    probability = np.asarray([float(item) for item in values[index].split(",")], dtype=np.float64)
                    require(probability.shape == (3,) and np.all(np.isfinite(probability)) and
                            np.all(probability >= 0.0) and 0.98 <= probability.sum() <= 1.02,
                            "FLARE ANP probability is invalid")
                    matrix[sample_index, haplotype] = probability / probability.sum()
            loci.append((position, fields[3], fields[4]))
            rows.append(matrix)
    require(samples is not None and rows, "FLARE projection is empty")
    marker_major = np.asarray(rows, dtype=np.float64)
    return loci, bridge_core.sanitize_f0(marker_major.transpose(1, 2, 0, 3))


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(args.root_label in ROOTS and ROOTS[args.root_label] == args.root_seed,
            "root label/seed pair is invalid")
    require(re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is not None, "git commit must be exact")
    require(os.geteuid() == 65534, "bridge runner must execute as the unprivileged nobody user")
    require(all(name not in os.environ for name in ("HOME", "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_CONFIG")),
            "credential-bearing environment is visible to bridge")
    inputs = {name: getattr(args, name) for name in INPUT_NAMES}
    sources = parse_sources(args.source)
    sidecars = [args.contract, args.authorization, args.source_auth]
    prove_runner_read_only(inputs, sources, sidecars)
    require(network_is_disabled(), "network namespace is not disabled")

    contract = load_json(args.contract)
    authorization = load_json(args.authorization)
    validate_contract(contract)
    input_hashes_pre = validate_authorization(authorization, args.root_label, args.root_seed, inputs)
    source_auth_sha = validate_source_auth(args.source_auth, sources, args.git_commit)

    rare = load_rare_crosschecked(inputs, args.root_seed)
    pools = audit_pools(inputs["pools"], {"freq_people": 300, "ref_people": 90, "donor_people": 768})
    ref_samples, ref_counts, ref_nodes = audit_ref_mapping(
        inputs["ref_pairs"], inputs["panel_map"], {"ref_people": 90}, pools["ref_pool_pairs"],
    )
    audit_tbi(inputs["ref_tbi"])
    audit_tbi(inputs["flare_tbi"])
    ref_vcf = audit_vcf(inputs["ref_vcf"], ref_samples)
    f0_loci, f0 = load_f0_projection(inputs["flare_anc"], rare.samples)
    require(tuple((position, ref, alt) for position, _identifier, ref, alt, _tsid in ref_vcf["loci"])
            == tuple(f0_loci), "REF and FLARE marker grids differ")
    audit_tree_vcf(inputs["tree_sequence"], ref_vcf["loci"], inputs["ref_vcf"],
                   ref_samples, ref_nodes, pools["all_nodes"])

    flare_positions = {position for position, _ref, _alt in f0_loci}
    keep = np.asarray([position not in flare_positions for position in rare.positions], dtype=bool)
    require(np.all(keep), "these technical roots unexpectedly contain rare/FLARE overlap")
    positions = rare.positions[keep]
    minor_codes = rare.minor_codes[keep].astype("|i1")
    genetic_map = load_genetic_map(inputs["genetic_map"], "22")
    cms = np.asarray(genetic_map.cm_at(positions), dtype="<f8")
    keys = np.asarray([locus_key(int(position)) for position in positions], dtype="|S64")

    observed = np.all(np.isfinite(rare.hap_presence[keep]), axis=2)
    target_site_major = np.where(
        observed, np.nansum(rare.hap_presence[keep], axis=2), 0,
    ).astype("|i1")
    swapped = np.where(
        observed, np.nansum(rare.hap_presence[keep, :, ::-1], axis=2), 0,
    ).astype("|i1")
    require(np.array_equal(target_site_major, swapped), "TARGET diploid dose changes after haplotype swap")
    target_dosage = np.ascontiguousarray(target_site_major.T)
    target_observed = np.ascontiguousarray(observed.astype("|u1").T)
    target_keys = np.asarray([bridge_core.sample_key(sample) for sample in rare.samples], dtype="|S64")

    ref_dosage, ref_people, ref_labels = load_ref_minor_dosage(
        inputs["tree_sequence"], inputs["pools"], rare, genetic_map,
    )
    ref_dosage = ref_dosage[keep]
    label_array = np.asarray(ref_labels, dtype=object)
    require(Counter(ref_labels) == Counter({name: 30 for name in ANCESTRIES}) and len(ref_people) == 90,
            "REF role firewall ancestry/person counts differ")
    ref_ac_locus_major = np.column_stack([
        ref_dosage[:, label_array == ancestry].sum(axis=1) for ancestry in ANCESTRIES
    ]).astype("<u2")
    ref_an_locus_major = np.full(ref_ac_locus_major.shape, 60, dtype="<u2")
    ref_ac = np.ascontiguousarray(ref_ac_locus_major.T)
    ref_an = np.ascontiguousarray(ref_an_locus_major.T)
    ref_observed = (ref_an > 0).astype("|u1")
    ref_no_support = ((ref_an > 0) & (ref_ac == 0)).astype("|u1")
    ref_af = np.divide(ref_ac, ref_an, out=np.zeros_like(ref_ac, dtype="<f8"), where=ref_an > 0)

    selected_arrays = {
        "locus_key_sha256": keys, "chrom": np.full(len(keys), 22, dtype="|u1"),
        "pos": positions.astype("<i8"), "ref": np.full(len(keys), b"A", dtype="|S1"),
        "alt": np.full(len(keys), b"C", dtype="|S1"), "cM": cms,
        "minor_code": minor_codes,
    }
    target_arrays = {
        "sample_key_sha256": target_keys, "locus_key_sha256": keys,
        "minor_dosage": target_dosage, "observed_mask": target_observed,
    }
    reference_arrays = {
        "ancestry": np.asarray(ANCESTRIES, dtype="|S4"), "locus_key_sha256": keys,
        "minor_ac": ref_ac, "callable_an": ref_an, "minor_af": ref_af,
        "observed_mask": ref_observed, "no_support": ref_no_support,
    }
    f0_arrays = {
        "sample_key_sha256": target_keys,
        "chrom": np.full(len(f0_loci), 22, dtype="|u1"),
        "pos": np.asarray([row[0] for row in f0_loci], dtype="<i8"),
        "ref": np.asarray([row[1].encode() for row in f0_loci], dtype="|S1"),
        "alt": np.asarray([row[2].encode() for row in f0_loci], dtype="|S1"),
        "probabilities": f0,
    }
    outputs = {
        "technical_kat_selected_loci_incremental.npz": selected_arrays,
        "technical_kat_target_rare_diploid_incremental.npz": target_arrays,
        "technical_kat_reference_rare_summary_incremental.npz": reference_arrays,
        "technical_kat_flare_f0_sanitized.npz": f0_arrays,
    }
    require(not args.output_dir.exists() and args.output_dir.parent.is_dir(), "output directory must be new")
    args.output_dir.mkdir(mode=0o700)
    raw_hashes: dict[str, str] = {}
    semantic_hashes: dict[str, str] = {}
    for name, arrays in outputs.items():
        path = args.output_dir / name
        write_deterministic_npz(path, arrays)
        reopen_npz(path, arrays)
        raw_hashes[name] = sha256_file(path)
        semantic_hashes[name] = semantic_arrays_sha256(SCHEMAS[name], arrays)

    input_hashes_post = {name: sha256_file(path) for name, path in sorted(inputs.items())}
    require(input_hashes_post == input_hashes_pre, "an input changed during SAFE_BRIDGE")
    legacy_target = np.where(observed, np.nansum(rare.hap_presence[keep], axis=2), -1).astype(np.int8)
    receipt = {
        "stage": STAGE, "schema_id": "tests_m33_safe_bridge_technical_kat_receipt_v1",
        "status": STATUS, "root_label": args.root_label, "root_seed": args.root_seed,
        "consumable": False, "scientific_evidence": False, "truth_read": False,
        "materialize_authorized": False, "ready_emitted": False, "training_authorized": False,
        "gcs_write": False, "selected_all_count": int(len(rare.positions)),
        "selected_incremental_count": int(keep.sum()), "selected_overlap_count": int((~keep).sum()),
        "incremental_minor_code_0_locus_count": int((minor_codes == 0).sum()),
        "target_missing_cells": int((target_observed == 0).sum()),
        "reference_no_support_loci": int(np.all(ref_no_support == 1, axis=0).sum()),
        "flare_marker_count": len(f0_loci), "target_count": len(rare.samples),
        "ref_people": len(ref_people), "ref_people_by_ancestry": ref_counts,
        "f0_probability_vectors": int(np.prod(f0.shape[:-1])),
        "target_diploid_dosage_legacy_sha256": array_sha256(legacy_target),
        "reference_ac_an_legacy_sha256": array_sha256(np.column_stack((
            ref_ac_locus_major.astype(np.int16), ref_an_locus_major.astype(np.int16),
        ))),
        "phase_swap_invariant": True, "f0_anp_only_projection": True,
        "f0_gt_an1_an2_ignored": True, "raw_identifiers_exported": False,
        "real_overlap_exercised": bool((~keep).sum()),
        "real_target_missingness_exercised": bool((target_observed == 0).sum()),
        "real_reference_missingness_exercised": False,
        "write_chmod_rename_probes_failed": True, "runner_uid": os.getuid(),
        "runner_euid": os.geteuid(), "network_disabled": True,
        "credential_environment_absent": True,
        "artifact_schema": SCHEMAS, "artifact_raw_sha256": raw_hashes,
        "artifact_semantic_sha256": semantic_hashes,
        "input_sha256_pre": input_hashes_pre, "input_sha256_post": input_hashes_post,
        "contract_sha256": sha256_file(args.contract),
        "authorization_sha256": sha256_file(args.authorization),
        "source_auth_sha256": source_auth_sha, "git_commit": args.git_commit,
        "oci_digest": args.oci_digest, "nextflow_version": args.nextflow_version,
        "append_only": True, "reopen_verified": True,
    }
    write_exclusive_json(args.output_dir / "safe_bridge_technical_kat.receipt.json", receipt)
    os.chmod(args.output_dir, 0o500)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-auth", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--root-label", required=True)
    parser.add_argument("--root-seed", required=True, type=int)
    parser.add_argument("--nextflow-version", required=True)
    parser.add_argument("--oci-digest", required=True)
    for name in sorted(INPUT_NAMES):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
