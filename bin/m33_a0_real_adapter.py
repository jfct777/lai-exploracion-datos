#!/usr/bin/env python3
"""Audit consumed M28--M31 chr22 assets before M33 tensor materialization.

This adapter is deliberately read-only and truth-free.  It proves technical
compatibility of legacy simulation roots; it cannot emit READY or scientific
evidence and it never trains or runs a model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from m31_ordered_linear import ancestry_support, load_genetic_map, load_ordered_rare, load_ref_minor_dosage
from m31_ordered_rare_preflight import derive_freq_sites


STAGE = "M33_A0_REAL_ADAPTER"
STATUS = "PASS_TECHNICAL_COMPATIBILITY_ONLY"
INPUT_NAMES = {
    "tree_sequence", "pools", "rare_catalog", "rare_haplotypes", "m31_sites",
    "m31_target", "ref_vcf", "ref_tbi", "target_vcf",
    "target_tbi", "ref_pairs", "panel_map", "flare_anc", "genetic_map",
}
ANCESTRIES = ("AFR", "EUR", "ASIA")
EXPECTED_PREREGISTRATION_SHA256 = "c99f890e00c383df25bbdfbb94e9fba3bb181adbfa4db1e1301e4648cfa3d70d"
EXPECTED_ASSET_REGISTRY_SHA256 = "649993d6e098b3cf92260a95d5bfcf8a89a529f8438f753dce368598958773de"
REQUIRED_SOURCE_PATHS = {
    "bin/m33_a0_real_adapter.py", "bin/m33_a0_source_auth.py", "bin/m33_a0_tabix_audit.py",
    "bin/m31_ordered_linear.py", "bin/m31_ordered_rare_preflight.py",
    "conf/m33_a0_legacy_assets.json", "conf/m33_a0_real_adapter.config",
    "conf/m33_a0_real_adapter_preregistration.json", "modules/33_A0_REAL_ADAPTER.nf",
    "workflows/m33_a0_real_adapter.nf", "tests/test_m33_a0_real_adapter.py",
    "tests/test_m33_a0_real_adapter_nextflow.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json_strict(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                      object_pairs_hook=unique_object)


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


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def load_contract(preregistration: Path, registry: Path, root_label: str,
                  root_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    require(sha256_file(preregistration) == EXPECTED_PREREGISTRATION_SHA256,
            "immutable A0 preregistration hash drifted")
    require(sha256_file(registry) == EXPECTED_ASSET_REGISTRY_SHA256,
            "immutable A0 asset registry hash drifted")
    contract = load_json_strict(preregistration)
    require(contract.get("stage") == STAGE, "unsupported A0 stage")
    require(contract.get("status") == "TECHNICAL_COMPATIBILITY_ONLY",
            "A0 must remain technical compatibility only")
    require(contract.get("phase_separation", {}).get("M0_materializer", "").startswith("BLOCKED"),
            "M0 must remain blocked during A0")
    for key in ("F0_forward", "B1_backward"):
        require(contract.get("phase_separation", {}).get(key, "").startswith("BLOCKED"),
                f"{key} must remain blocked during A0")
    authorization = contract.get("execution_authorization", {})
    for forbidden in ("write_READY", "materialize_tensor", "forward", "backward", "training", "truth_scoring"):
        require(authorization.get(forbidden) is False, f"A0 authorization enables {forbidden}")
    allowed = contract["root_registry"]["allowed_technical_roots"]
    require(allowed.get(root_label) == root_seed, "root label/seed is not allowed")

    assets = load_json_strict(registry)
    require(assets.get("stage") == "M33_A0_LEGACY_ASSET_REGISTRY", "unsupported asset registry")
    require(assets.get("status") == "TECHNICAL_COMPATIBILITY_ONLY", "asset registry status drifted")
    require(root_label in assets.get("roots", {}), "root is not yet frozen in the asset registry")
    root = assets["roots"][root_label]
    require(root.get("root_seed") == root_seed, "asset registry root seed drifted")
    require(set(root.get("sha256", {})) == INPUT_NAMES | {"flare_anc_tbi"},
            "asset hash inventory drifted")
    require(root["sha256"]["flare_anc_tbi"] is None,
            "legacy FLARE index absence must remain explicit")
    return contract, root


def authenticate_inputs(paths: dict[str, Path], root: dict[str, Any]) -> dict[str, str]:
    require(set(paths) == INPUT_NAMES, "A0 input inventory is incomplete")
    expected = root["sha256"]
    observed: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        require(str(path) and not str(path).startswith("gs://"), "A0 parser accepts staged local files only")
        require(path.is_file(), f"missing A0 input: {name}")
        observed[name] = sha256_file(path)
        require(observed[name] == expected[name], f"input sha256 mismatch: {name}")
    return observed


def audit_tbi(path: Path) -> None:
    with gzip.open(path, "rb") as handle:
        require(handle.read(4) == b"TBI\x01", "invalid Tabix index magic")


def audit_pools(path: Path, expected: dict[str, int]) -> dict[str, Any]:
    people: dict[tuple[str, str], list[int]] = defaultdict(list)
    ancestries: dict[tuple[str, str], str] = {}
    nodes: set[int] = set()
    individual_roles: dict[str, str] = {}
    fingerprints: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"role", "ancestry", "individual_id", "node_id", "node_identity_sha256"}
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                "pool header drifted")
        for row in reader:
            role, individual, ancestry = row["role"], row["individual_id"], row["ancestry"]
            require(role in {"FREQ", "REF_LAI", "DONOR"} and ancestry in ANCESTRIES,
                    "pool role/ancestry invalid")
            node = int(row["node_id"])
            require(node not in nodes, "tree node is reused across pool roles")
            fingerprint = row["node_identity_sha256"]
            require(fingerprint == hashlib.sha256(f"source-node:{node}".encode()).hexdigest(),
                    "node identity hash does not bind the node")
            require(fingerprint not in fingerprints, "node identity fingerprint is reused")
            require(individual not in individual_roles or individual_roles[individual] == role,
                    "individual crosses pool roles")
            individual_roles[individual] = role
            fingerprints.add(fingerprint)
            nodes.add(node)
            key = (role, individual)
            people[key].append(node)
            require(key not in ancestries or ancestries[key] == ancestry,
                    "person crosses ancestries")
            ancestries[key] = ancestry
    require(all(len(value) == 2 for value in people.values()), "each pool person needs two nodes")
    counts = Counter(role for role, _individual in people)
    for role, key in (("FREQ", "freq_people"), ("REF_LAI", "ref_people"), ("DONOR", "donor_people")):
        require(counts[role] == expected[key], f"unexpected {role} person count")
    return {
        "people_by_role": dict(sorted(counts.items())), "nodes": len(nodes), "all_nodes": nodes,
        "ref_pool_pairs": {
            (ancestries[(role, individual)], *tuple(sorted(pair)))
            for (role, individual), pair in people.items() if role == "REF_LAI"
        },
    }


def audit_rare(paths: dict[str, Path], contract: dict[str, Any], root_seed: int,
               expected: dict[str, int]) -> dict[str, Any]:
    # Recompute the FREQ-only selection from the actual tree before reading TARGET.
    m31_contract = {
        "chromosome_domain": {"chrom": "22", "start_bp": 15287922},
        "rare_universe": {
            "minimum_mac": 2, "maximum_maf_exclusive": 0.01,
            "minimum_carrier_individuals": 2,
            "prohibited_selectors": ["TARGET", "truth", "Gnomix_prediction", "FLARE_prediction"],
        },
    }
    selected, freq = derive_freq_sites(
        paths["tree_sequence"], paths["pools"], paths["rare_catalog"], m31_contract
    )
    with gzip.open(paths["m31_sites"], "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and "chrom" in reader.fieldnames, "M31 sites lack chromosome")
        require(all(row["chrom"].removeprefix("chr") == "22" for row in reader),
                "M31 sites contain a chromosome outside chr22")
    rare = load_ordered_rare(paths["m31_sites"], paths["m31_target"], root_seed)
    require(len(selected) == len(rare.positions) == expected["selected_rare_sites"],
            "selected rare-site count drifted")
    require(freq["catalog_sites"] == expected["catalog_sites"], "rare catalog count drifted")
    require(freq["excluded_freq_single_carrier_sites"] == expected["excluded_single_carrier_sites"],
            "single-carrier exclusion count drifted")
    require([site.position for site in selected] == rare.positions.tolist(),
            "M31 positions differ from exhaustive FREQ selection")
    require([site.minor_code for site in selected] == rare.minor_codes.tolist(),
            "M31 minor orientation differs from exhaustive FREQ selection")
    require(int((rare.minor_codes == 0).sum()) == expected["minor_code_zero_sites"],
            "minor-code-zero count drifted")

    selected_index = {int(position): index for index, position in enumerate(rare.positions)}
    seen: set[int] = set()
    with gzip.open(paths["rare_haplotypes"], "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None, "rare haplotypes lack a header")
        hap_fields = reader.fieldnames[3:]
        expected_haps = [f"{sample}_h{hap}" for sample in rare.samples for hap in (0, 1)]
        require(hap_fields == expected_haps, "rare-haplotype TARGET order differs from M31")
        for row in reader:
            position = int(row["position"])
            require(row["chrom"].removeprefix("chr") == "22", "rare haplotypes contain a chromosome outside chr22")
            index = selected_index.get(position)
            if index is None:
                continue
            require(position not in seen, "rare haplotypes duplicate a selected locus")
            seen.add(position)
            require(int(row["minor_code"]) == int(rare.minor_codes[index]),
                    "rare haplotype minor code differs from M31")
            observed = np.asarray([int(row[field]) for field in hap_fields], dtype=np.int8).reshape(len(rare.samples), 2)
            expected_presence = (observed == int(rare.minor_codes[index])).astype(np.int8)
            require(np.array_equal(expected_presence, rare.hap_presence[index]),
                    "M31 TARGET presence differs from M28 haplotypes")
    require(seen == set(selected_index), "rare haplotypes omit selected M31 loci")
    return {
        "catalog_sites": freq["catalog_sites"],
        "selected_rare_sites": len(selected),
        "excluded_single_carrier_sites": freq["excluded_freq_single_carrier_sites"],
        "minor_code_zero_sites": int((rare.minor_codes == 0).sum()),
        "target_people": len(rare.samples),
        "target_missing_haplotypes": int(np.isnan(rare.hap_presence).sum()),
        "target_samples": rare.samples,
        "rare": rare,
    }


def read_panel_map(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            require(len(row) == 2 and row[0] and row[1] in ANCESTRIES, "panel map row invalid")
            require(row[0] not in rows, "panel map sample duplicated")
            rows[row[0]] = row[1]
    return rows


def audit_ref_mapping(ref_pairs: Path, panel_map_path: Path, expected: dict[str, int],
                      ref_pool_pairs: set[tuple[str, int, int]]) -> tuple[tuple[str, ...], dict[str, int], dict[str, tuple[int, int]]]:
    panel = read_panel_map(panel_map_path)
    pairs: dict[str, tuple[str, int, int]] = {}
    nodes: set[int] = set()
    with ref_pairs.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "ancestry", "haplotype_0_node", "haplotype_1_node"}
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames), "REF pairs header drifted")
        for row in reader:
            sample, ancestry = row["sample_id"], row["ancestry"]
            pair = (int(row["haplotype_0_node"]), int(row["haplotype_1_node"]))
            require(sample not in pairs and pair[0] != pair[1] and not nodes.intersection(pair),
                    "REF pair sample/node reused")
            require(panel.get(sample) == ancestry, "REF pairs differ from panel map")
            pairs[sample] = (ancestry, *pair)
            nodes.update(pair)
    require(len(pairs) == expected["ref_people"] and set(pairs) == set(panel), "REF mapping count/set drifted")
    require({(value[0], value[1], value[2]) for value in pairs.values()} == ref_pool_pairs,
            "REF pairs are not the exact REF_LAI pool nodes")
    counts = Counter(value[0] for value in pairs.values())
    require(counts == Counter({ancestry: 30 for ancestry in ANCESTRIES}), "REF ancestry balance drifted")
    return tuple(pairs), dict(sorted(counts.items())), {
        sample: (value[1], value[2]) for sample, value in pairs.items()
    }


def audit_vcf(path: Path, expected_samples: Iterable[str], *, flare: bool = False) -> dict[str, Any]:
    expected_samples = tuple(expected_samples)
    samples: tuple[str, ...] | None = None
    loci: list[tuple[int, str, str, str, int]] = []
    probability_vectors = 0
    fileformat_seen = False
    genotype_digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.rstrip("\n") == "##fileformat=VCFv4.2":
                fileformat_seen = True
            if line.startswith("##") or not line.strip():
                continue
            if line.startswith("#CHROM"):
                samples = tuple(line.rstrip("\n").split("\t")[9:])
                require(samples == expected_samples, "VCF sample order differs from authenticated source")
                continue
            require(samples is not None, "VCF data precede #CHROM")
            fields = line.rstrip("\n").split("\t")
            require(len(fields) == 9 + len(samples), f"malformed VCF row {line_number}")
            require(fields[0].removeprefix("chr") == "22" and fields[3:5] == ["A", "C"],
                    "VCF chromosome/allele mapping drifted")
            position = int(fields[1])
            require(not loci or position > loci[-1][0], "VCF loci are not strictly ordered")
            require(re.fullmatch(r"m28s[0-9]+", fields[2]) is not None, "VCF site ID is not an M28 TSID")
            tsid = int(fields[2][4:])
            info = dict(token.split("=", 1) for token in fields[7].split(";") if "=" in token)
            require(int(info.get("TSID", "-1")) == tsid, "VCF ID/TSID mismatch")
            fmt = fields[8].split(":")
            require("GT" in fmt, "VCF lacks GT")
            gt_index = fmt.index("GT")
            for sample_value in fields[9:]:
                values = sample_value.split(":")
                require(re.fullmatch(r"[01]\|[01]", values[gt_index]) is not None,
                        "VCF genotype is not phased binary diploid")
                genotype_digest.update(values[gt_index].encode())
                genotype_digest.update(b"\0")
                if flare:
                    require(all(name in fmt for name in ("AN1", "AN2", "ANP1", "ANP2")),
                            "FLARE fields missing")
                    for hard_name, prob_name in (("AN1", "ANP1"), ("AN2", "ANP2")):
                        hard = int(values[fmt.index(hard_name)])
                        probs = [float(value) for value in values[fmt.index(prob_name)].split(",")]
                        require(hard in (0, 1, 2) and len(probs) == 3 and all(math.isfinite(v) and v >= 0 for v in probs),
                                "FLARE ancestry probability invalid")
                        require(0.98 - 1e-12 <= sum(probs) <= 1.02 + 1e-12,
                                "FLARE probability vector outside rounding tolerance")
                        require(probs[hard] >= max(probs) - 1e-12, "FLARE hard call is not a probability maximum")
                        probability_vectors += 1
            loci.append((position, fields[2], fields[3], fields[4], tsid))
    require(fileformat_seen, "VCFv4.2 fileformat header is absent")
    require(samples is not None and loci, "VCF is empty")
    return {"samples": samples, "loci": tuple(loci), "probability_vectors": probability_vectors,
            "gt_sha256": genotype_digest.hexdigest()}


def audit_tree_vcf(tree_path: Path, loci: tuple[tuple[int, str, str, str, int], ...],
                   ref_vcf: Path, ref_samples: tuple[str, ...],
                   ref_nodes: dict[str, tuple[int, int]], all_pool_nodes: set[int]) -> dict[str, int]:
    import tskit

    ts = tskit.load(str(tree_path))
    require(ts.sequence_length == 35503456.0, "tree sequence length drifted")
    sample_nodes = {int(node) for node in ts.samples()}
    require(all_pool_nodes.issubset(sample_nodes), "pool contains a non-sample tree node")
    for position, identifier, ref, alt, tsid in loci:
        require(0 <= tsid < ts.num_sites, "VCF TSID is outside the tree")
        site = ts.site(tsid)
        require(float(site.position).is_integer(), "tree site position is fractional")
        require(identifier == f"m28s{tsid}" and 15287922 + int(site.position) == position,
                "VCF TSID/position differs from tree")
        require(site.ancestral_state == "0" and
                all(mutation.derived_state in {"0", "1"} for mutation in site.mutations),
                "tree binary allele coding drifted")
        require((ref, alt) == ("A", "C"), "VCF code mapping drifted")

    requested_nodes = [node for sample in ref_samples for node in ref_nodes[sample]]
    vcf_records = []
    with gzip.open(ref_vcf, "rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            fmt = fields[8].split(":")
            gt_index = fmt.index("GT")
            states = tuple(int(allele) for value in fields[9:] for allele in value.split(":")[gt_index].split("|"))
            vcf_records.append((int(fields[2][4:]), states))
    record_index = 0
    for variant in ts.variants(samples=requested_nodes):
        if record_index == len(vcf_records):
            break
        wanted_tsid, expected_states = vcf_records[record_index]
        if variant.site.id < wanted_tsid:
            continue
        require(variant.site.id == wanted_tsid, "REF VCF TSID absent from tree variants")
        require(tuple(int(value) for value in variant.genotypes) == expected_states,
                "REF VCF genotype differs from mapped REF_LAI tree nodes")
        record_index += 1
    require(record_index == len(vcf_records), "REF VCF/tree genotype cross-check truncated")
    return {"tree_sites": ts.num_sites, "tree_samples": ts.num_samples,
            "vcf_sites_crosschecked": len(loci), "ref_genotype_sites_crosschecked": record_index}


def audit_m0_bridge(paths: dict[str, Path], rare: Any,
                    flare_loci: tuple[tuple[int, str, str, str, int], ...],
                    expected: dict[str, int]) -> dict[str, Any]:
    genetic_map = load_genetic_map(paths["genetic_map"], "22")
    require(int(genetic_map.positions[0]) <= 15287922 and int(genetic_map.positions[-1]) >= 50791377,
            "genetic map does not cover the simulated chr22 domain")
    rare_positions = {int(position) for position in rare.positions}
    flare_positions = {position for position, _identifier, _ref, _alt, _tsid in flare_loci}
    overlap = rare_positions & flare_positions
    incremental = rare_positions - flare_positions
    require(len(overlap) == expected["rare_overlap_flare_sites"], "rare/FLARE overlap count drifted")
    require(len(incremental) == expected["incremental_rare_sites"], "incremental rare count drifted")
    require(overlap.isdisjoint(incremental) and overlap | incremental == rare_positions,
            "rare all/incremental/overlap partition is not exact")

    ref_dosage, ref_people, ref_labels = load_ref_minor_dosage(
        paths["tree_sequence"], paths["pools"], rare, genetic_map
    )
    require(ref_dosage.shape == (len(rare.positions), expected["ref_people"]),
            "REF rare dosage dimensions drifted")
    require(np.all((ref_dosage >= 0) & (ref_dosage <= 2)), "REF rare dosage is outside 0/1/2")
    label_array = np.asarray(ref_labels, dtype=object)
    require(Counter(ref_labels) == Counter({ancestry: 30 for ancestry in ANCESTRIES}),
            "REF rare ancestry panel sizes drifted")
    ref_ac = np.column_stack([
        ref_dosage[:, label_array == ancestry].sum(axis=1) for ancestry in ANCESTRIES
    ]).astype(np.int16)
    ref_an = np.full(ref_ac.shape, 60, dtype=np.int16)
    require(np.all((ref_ac >= 0) & (ref_ac <= ref_an)), "REF AC/AN invalid")
    support, no_support = ancestry_support(ref_dosage, ref_labels)
    require(np.allclose(support.sum(axis=1), (~no_support).astype(float), atol=1e-12, rtol=0.0),
            "REF support denominator semantics drifted")

    observed = np.isfinite(rare.hap_presence)
    diploid_observed = observed.all(axis=2)
    target_dosage = np.where(diploid_observed, np.nansum(rare.hap_presence, axis=2), -1).astype(np.int8)
    require(np.all((target_dosage == -1) | ((target_dosage >= 0) & (target_dosage <= 2))),
            "TARGET diploid dosage is outside missing/0/1/2")
    if np.all(observed):
        require(np.array_equal(target_dosage, rare.hap_presence.sum(axis=2).astype(np.int8)),
                "TARGET dosage does not equal h0+h1")
    return {
        "genetic_map_points": len(genetic_map.positions),
        "genetic_map_start_bp": int(genetic_map.positions[0]),
        "genetic_map_end_bp": int(genetic_map.positions[-1]),
        "rare_overlap_flare_sites": len(overlap),
        "incremental_rare_sites": len(incremental),
        "ref_callable_AN_per_ancestry": 60,
        "ref_no_support_sites": int(no_support.sum()),
        "ref_ac_an_sha256": array_sha256(np.column_stack((ref_ac, ref_an))),
        "target_diploid_dosage_sha256": array_sha256(target_dosage),
        "target_missing_diploid_cells": int((target_dosage == -1).sum()),
        "phase_exported_to_M0": False,
        "ref_people": len(ref_people),
    }


def parse_sources(values: Iterable[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        relative, separator, staged = value.partition("=")
        require(bool(separator) and relative and not relative.startswith("/"), "invalid source specification")
        require(".." not in Path(relative).parts and relative not in sources, "unsafe or duplicate source")
        sources[relative] = Path(staged)
    require(set(sources) == REQUIRED_SOURCE_PATHS, "staged source inventory is incomplete")
    return sources


def load_source_auth(path: Path, git_commit: str, sources: dict[str, Path]) -> str:
    payload = load_json_strict(path)
    require(set(payload) == {"stage", "status", "git_commit", "source_sha256"},
            "source-auth keys drifted")
    require(payload.get("stage") == "M33_A0_SOURCE_AUTH", "source-auth stage drifted")
    require(payload.get("status") == "PASS_EXACT_COMMIT_AND_SOURCE_HASHES", "source-auth did not pass")
    require(payload.get("git_commit") == git_commit, "source-auth commit drifted")
    hashes = payload.get("source_sha256", {})
    require(set(hashes) == REQUIRED_SOURCE_PATHS and
            all(re.fullmatch(r"[0-9a-f]{64}", value) is not None for value in hashes.values()),
            "source-auth inventory or hashes drifted")
    observed = {relative: sha256_file(staged) for relative, staged in sorted(sources.items())}
    require(observed == hashes, "staged A0 sources differ from source-auth")
    return sha256_file(path)


def validate_index_audit(path: Path, input_hashes: dict[str, str], source_auth_sha: str,
                         git_commit: str) -> str:
    payload = load_json_strict(path)
    require(set(payload) == {"stage", "status", "git_commit", "expected_container_image_id",
                             "tabix_version", "source_auth_sha256", "ref", "target"},
            "BGZF/Tabix audit keys drifted")
    require(payload.get("stage") == "M33_A0_TABIX_AUDIT" and
            payload.get("status") == "PASS_BGZF_AND_INDEX_FULL_STREAM_PARITY",
            "BGZF/Tabix audit did not pass")
    require(payload.get("git_commit") == git_commit and payload.get("source_auth_sha256") == source_auth_sha,
            "BGZF/Tabix audit provenance drifted")
    require(payload.get("expected_container_image_id") ==
            "sha256:b89353efc9a4a5953519fa9f066728e2f63d0e9125fc9fc771ef3ea9bb9c961c",
            "BGZF/Tabix image identity drifted")
    require(payload.get("tabix_version") == "tabix (htslib) 1.16", "Tabix/htslib version drifted")
    for section, prefix in (("ref", "ref"), ("target", "target")):
        require(set(payload.get(section, {})) == {"vcf_sha256", "tbi_sha256", "record_count", "record_sha256"},
                f"BGZF/Tabix audit section keys drifted for {section}")
        require(payload.get(section, {}).get("vcf_sha256") == input_hashes[f"{prefix}_vcf"] and
                payload.get(section, {}).get("tbi_sha256") == input_hashes[f"{prefix}_tbi"],
                f"BGZF/Tabix audit hashes drifted for {section}")
        require(payload[section].get("record_count") == 79791 and
                re.fullmatch(r"[0-9a-f]{64}", payload[section].get("record_sha256", "")) is not None,
                f"BGZF/Tabix audit record inventory drifted for {section}")
    return sha256_file(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is not None, "git commit must be exact")
    contract, root = load_contract(args.preregistration, args.asset_registry, args.root_label, args.root_seed)
    import tskit
    require(args.nextflow_version == "26.04.6", "Nextflow version drifted")
    require(args.adapter_image_id == contract["runtime"]["adapter_image_id"], "adapter image identity drifted")
    require(platform.python_version_tuple()[:2] == ("3", "11"), "Python major/minor version drifted")
    require(np.__version__ == "2.4.6" and tskit.__version__ == "1.0.3", "NumPy/tskit version drifted")
    paths = {name: getattr(args, name) for name in INPUT_NAMES}
    input_hashes = authenticate_inputs(paths, root)
    source_auth_sha = load_source_auth(args.source_auth, args.git_commit, parse_sources(args.source))
    index_audit_sha = validate_index_audit(args.index_audit, input_hashes, source_auth_sha, args.git_commit)
    expected = root["expected_counts"]

    pools = audit_pools(paths["pools"], expected)
    rare = audit_rare(paths, contract, args.root_seed, expected)
    ref_samples, ref_counts, ref_nodes = audit_ref_mapping(
        paths["ref_pairs"], paths["panel_map"], expected, pools["ref_pool_pairs"]
    )
    audit_tbi(paths["ref_tbi"])
    audit_tbi(paths["target_tbi"])
    ref_vcf = audit_vcf(paths["ref_vcf"], ref_samples)
    target_vcf = audit_vcf(paths["target_vcf"], rare["target_samples"])
    require(ref_vcf["loci"] == target_vcf["loci"], "REF and TARGET VCF grids differ")
    require(len(ref_vcf["loci"]) == expected["flare_loci"], "common grid locus count drifted")
    flare = audit_vcf(paths["flare_anc"], rare["target_samples"], flare=True)
    require(flare["loci"] == target_vcf["loci"], "FLARE output grid differs from TARGET VCF")
    require(flare["gt_sha256"] == target_vcf["gt_sha256"], "FLARE GT differs from TARGET GT")
    tree = audit_tree_vcf(paths["tree_sequence"], target_vcf["loci"], paths["ref_vcf"],
                          ref_samples, ref_nodes, pools["all_nodes"])
    bridge = audit_m0_bridge(paths, rare["rare"], flare["loci"], expected)

    receipt = {
        "stage": STAGE,
        "status": STATUS,
        "scope": contract["scope"],
        "root_label": args.root_label,
        "root_seed": args.root_seed,
        "scientific_evidence": False,
        "ready_emitted": False,
        "counts": {
            "people_by_role": pools["people_by_role"],
            "ref_people_by_ancestry": ref_counts,
            "target_people": rare["target_people"],
            "catalog_sites": rare["catalog_sites"],
            "selected_rare_sites": rare["selected_rare_sites"],
            "excluded_single_carrier_sites": rare["excluded_single_carrier_sites"],
            "minor_code_zero_sites": rare["minor_code_zero_sites"],
            "flare_loci": len(flare["loci"]),
            "flare_probability_vectors": flare["probability_vectors"],
            **tree,
            **bridge,
        },
        "checks": {
            "freq_selection_recomputed_before_target": True,
            "target_rare_crosschecked_against_m28_and_m31": True,
            "ref_tree_vcf_mapping_exact": True,
            "ref_target_flare_grid_exact": True,
            "phased_binary_gt_exact": True,
            "tabix_magic_and_pinned_hash_exact": True,
            "bgzf_tabix_complete_stream_parity": True,
            "flare_gt_equals_target_gt": True,
            "ref_gt_equals_tree_nodes": True,
            "rare_partition_and_ref_support_ready_for_M0": True,
            "flare_probability_simplex_with_rounding_tolerance": True,
            "truth_not_read": True,
        },
        "legacy_exceptions": {
            "flare_anc_tbi": "ABSENT_SEQUENTIAL_BGZF_A0_ONLY",
            "assembly": "GRCh38_COORDINATE_DOMAIN_ONLY_NO_FASTA_NORMALIZATION_CLAIM",
            "root_role": "CONSUMED_TECHNICAL_ONLY_NOT_PROSPECTIVE",
        },
        "downstream_gates": {"M0": "BLOCKED_PENDING_A0_POST", "F0": "BLOCKED", "B1": "BLOCKED", "TRAIN": "BLOCKED"},
        "input_sha256": input_hashes,
        "preregistration_sha256": sha256_file(args.preregistration),
        "asset_registry_sha256": sha256_file(args.asset_registry),
        "source_auth_sha256": source_auth_sha,
        "index_audit_sha256": index_audit_sha,
        "git_commit": args.git_commit,
        "runtime": {
            "nextflow": args.nextflow_version,
            "adapter_image_id": args.adapter_image_id,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tskit": tskit.__version__,
        },
    }
    write_exclusive_json(args.output, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--asset-registry", required=True, type=Path)
    parser.add_argument("--source-auth", required=True, type=Path)
    parser.add_argument("--index-audit", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--nextflow-version", required=True)
    parser.add_argument("--adapter-image-id", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--root-label", required=True)
    parser.add_argument("--root-seed", required=True, type=int)
    for name in sorted(INPUT_NAMES):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    receipt = run(parse_args())
    print(json.dumps({"status": receipt["status"], "root_label": receipt["root_label"],
                      "selected_rare_sites": receipt["counts"]["selected_rare_sites"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
