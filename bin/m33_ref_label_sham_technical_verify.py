#!/usr/bin/env python3
"""Independent oracle for M33 technical REF-label-sham artifacts.

This verifier deliberately imports none of the SAFE_BRIDGE, M31 or A0 code.
It reconstructs diploid minor-allele dosage from the authenticated tree sequence
and REF_LAI pool, reapplies the preregistered label permutations, and compares
the resulting aggregate arrays byte-for-byte.  It never reads LAI truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ANCESTRIES = ("AFR", "EUR", "ASIA")
SEEDS = (79351217, 202307732, 1737132171)
DOMAIN = b"DNABR_M33_PRE4_REF_LABEL_SHAM_V1|"
SCHEMA = "tests_m33_safe_bridge_technical_kat_reference_rare_summary_ref_label_sham_v1"
MEMBERS = {"ancestry", "locus_key_sha256", "minor_ac", "callable_an", "minor_af",
           "observed_mask", "no_support"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(SCHEMA.encode() + b"\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b"\0" + value.dtype.str.encode() + b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_npz(path: Path, members: set[str]) -> dict[str, np.ndarray]:
    require(path.is_file() and not path.is_symlink(), f"invalid NPZ: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == members, f"NPZ members differ: {path.name}")
        arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    require(all(value.dtype.kind not in "OUV" for value in arrays.values()),
            f"unsafe NPZ dtype: {path.name}")
    return arrays


def load_ref_pool(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[int, int]]]:
    labels: dict[str, str] = {}
    nodes: dict[str, list[int]] = {}
    seen: set[int] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and
                {"node_id", "role", "ancestry", "individual_id"}.issubset(reader.fieldnames),
                "pool header differs")
        for row in reader:
            node = int(row["node_id"])
            require(node not in seen, "pool node duplicated")
            seen.add(node)
            if row["role"] != "REF_LAI":
                continue
            person, ancestry = row["individual_id"], row["ancestry"]
            require(person and ancestry in ANCESTRIES, "invalid REF_LAI record")
            require(person not in labels or labels[person] == ancestry,
                    "REF person crosses ancestries")
            labels[person] = ancestry
            nodes.setdefault(person, []).append(node)
    people = tuple(sorted(nodes))
    pairs = {person: tuple(sorted(nodes[person])) for person in people}
    require(len(people) == 90 and Counter(labels[p] for p in people) ==
            Counter({ancestry: 30 for ancestry in ANCESTRIES}), "REF 30/30/30 firewall differs")
    require(all(len(pair) == 2 and pair[0] != pair[1] for pair in pairs.values()) and
            len({node for pair in pairs.values() for node in pair}) == 180,
            "REF person-to-two-node mapping differs")
    return people, tuple(labels[p] for p in people), pairs


def map_ref_pseudonyms(ref_pairs_path: Path, panel_map_path: Path,
                       private_people: Sequence[str], labels: Sequence[str],
                       private_pairs: Mapping[str, Sequence[int]]) -> tuple[tuple[str, ...], dict[str, tuple[int, int]]]:
    panel: dict[str, str] = {}
    with panel_map_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            require(len(row) == 2 and row[0] not in panel and row[1] in ANCESTRIES,
                    "invalid panel map")
            panel[row[0]] = row[1]
    node_pair_to_sample: dict[tuple[int, int], str] = {}
    with ref_pairs_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and
                {"sample_id", "ancestry", "haplotype_0_node", "haplotype_1_node"}.issubset(reader.fieldnames),
                "invalid REF pairs header")
        for row in reader:
            sample, ancestry = row["sample_id"], row["ancestry"]
            pair = tuple(sorted((int(row["haplotype_0_node"]), int(row["haplotype_1_node"]))))
            require(sample in panel and panel[sample] == ancestry and pair not in node_pair_to_sample,
                    "REF pairs and panel map differ")
            node_pair_to_sample[pair] = sample
    mapped: list[str] = []
    mapped_pairs: dict[str, tuple[int, int]] = {}
    for person, label in zip(private_people, labels):
        pair = tuple(sorted(private_pairs[person]))
        sample = node_pair_to_sample.get(pair)
        require(sample is not None and panel.get(sample) == label,
                "private REF nodes do not map exactly to the pseudonymous panel")
        mapped.append(sample)
        mapped_pairs[sample] = pair
    require(len(set(mapped)) == len(mapped) == 90 and set(mapped) == set(panel),
            "REF pseudonym mapping is not a 90-person bijection")
    return tuple(mapped), mapped_pairs


def permute_labels(people: Sequence[str], labels: Sequence[str],
                   pairs: Mapping[str, Sequence[int]], seed: int) -> tuple[str, ...]:
    require(seed in SEEDS, "unregistered sham seed")
    order = sorted(range(len(people)), key=lambda i: (
        hashlib.sha256(DOMAIN + str(seed).encode() + b"|" + people[i].encode() + b"|" +
                       ",".join(str(x) for x in sorted(pairs[people[i]])).encode()).digest(), i))
    result = tuple(labels[i] for i in order)
    if result == tuple(labels):
        for shift in range(1, len(labels)):
            candidate = tuple(labels[shift:]) + tuple(labels[:shift])
            if candidate != tuple(labels):
                result = candidate
                break
    require(result != tuple(labels) and Counter(result) == Counter(labels),
            "sham is identity or changes group sizes")
    require(all(any(a == source and b != source for a, b in zip(labels, result))
                for source in ANCESTRIES), "an ancestry has no outgoing reassignment")
    return result


def aggregate(dosage: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    require(dosage.ndim == 2 and dosage.shape[1] == len(labels) and
            dosage.dtype.kind in "iu" and np.all(np.isin(dosage, (0, 1, 2))),
            "invalid diploid minor dosage")
    axis = np.asarray(labels, dtype=object)
    ac = np.vstack([dosage[:, axis == ancestry].sum(axis=1) for ancestry in ANCESTRIES]).astype("<u2")
    an = np.vstack([np.full(dosage.shape[0], 2 * int(np.sum(axis == ancestry)), dtype="<u2")
                    for ancestry in ANCESTRIES])
    af = np.divide(ac, an, out=np.zeros_like(ac, dtype="<f8"), where=an > 0)
    return {"minor_ac": ac, "callable_an": an, "minor_af": af,
            "observed_mask": (an > 0).astype("|u1"),
            "no_support": ((an > 0) & (ac == 0)).astype("|u1")}


def genetic_map_start(path: Path) -> int:
    first: int | None = None
    previous = -1
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            fields = raw.split()
            require(len(fields) >= 3 and fields[0].removeprefix("chr") == "22",
                    f"invalid chr22 genetic map row {line_number}")
            position = int(fields[1])
            require(position > previous, "genetic-map positions are not strictly increasing")
            if first is None:
                first = position
            previous = position
    require(first is not None, "genetic map is empty")
    return first


def reconstruct_dosage(tree_path: Path, genetic_map_path: Path,
                        positions: np.ndarray, minor_codes: np.ndarray,
                        people: Sequence[str], pairs: Mapping[str, Sequence[int]]) -> np.ndarray:
    import tskit
    ts = tskit.load(str(tree_path))
    sample_index = {int(node): i for i, node in enumerate(ts.samples())}
    indexes = np.asarray([sample_index[node] for person in people for node in pairs[person]], dtype=np.int64)
    tree_pos = np.fromiter((int(site.position) for site in ts.sites()), dtype=np.int64,
                           count=ts.num_sites)
    candidates = {0, genetic_map_start(genetic_map_path), int(positions[0] - tree_pos[0])}
    offsets = [offset for offset in candidates
               if set(positions.tolist()).issubset(set((tree_pos + offset).tolist()))]
    require(len(offsets) == 1, f"cannot identify one tree coordinate offset: {offsets}")
    by_pos = {int(pos): i for i, pos in enumerate(positions)}
    dosage = np.empty((len(positions), len(people)), dtype="|i1")
    seen: set[int] = set()
    for variant, raw_pos in zip(ts.variants(), tree_pos):
        absolute = int(raw_pos + offsets[0])
        index = by_pos.get(absolute)
        if index is None:
            continue
        require(len(variant.alleles) == 2, "selected tree variant is not biallelic")
        states = np.asarray(variant.genotypes, dtype=np.int8)[indexes]
        require(np.all(np.isin(states, (0, 1))), "REF contains missing/nonbinary state")
        dosage[index] = (states == minor_codes[index]).reshape(len(people), 2).sum(axis=1)
        seen.add(absolute)
    require(seen == set(positions.tolist()), "tree omits selected loci")
    return dosage


def verify(args: argparse.Namespace) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is not None,
            "git commit must be an exact lowercase SHA-1")
    selected = load_npz(args.selected_loci,
                        {"locus_key_sha256", "chrom", "pos", "ref", "alt", "cM", "minor_code"})
    require(np.all(selected["chrom"] == 22) and np.all(np.isin(selected["minor_code"], (0, 1))),
            "selected locus scope/orientation differs")
    private_people, labels, private_pairs = load_ref_pool(args.pools)
    people, pairs = map_ref_pseudonyms(
        args.ref_pairs, args.panel_map, private_people, labels, private_pairs,
    )
    dosage = reconstruct_dosage(args.tree_sequence, args.genetic_map,
                                selected["pos"], selected["minor_code"],
                                private_people, private_pairs)
    receipt = json.loads(args.bridge_receipt.read_text(encoding="utf-8"))
    real = load_npz(args.real_reference, MEMBERS)
    real_oracle = {"ancestry": np.asarray(ANCESTRIES, dtype="|S4"),
                   "locus_key_sha256": selected["locus_key_sha256"], **aggregate(dosage, labels)}
    require(all(np.array_equal(real[name], real_oracle[name]) for name in MEMBERS),
            "real REF summary differs from independent oracle")
    observed_hashes: dict[str, str] = {}
    semantic_hashes: set[str] = set()
    for value in args.sham:
        seed_text, sep, raw_path = value.partition("=")
        require(sep and seed_text.isdigit() and int(seed_text) in SEEDS, "invalid sham argument")
        seed, path = int(seed_text), Path(raw_path)
        arrays = load_npz(path, MEMBERS)
        oracle = {"ancestry": np.asarray(ANCESTRIES, dtype="|S4"),
                  "locus_key_sha256": selected["locus_key_sha256"],
                  **aggregate(dosage, permute_labels(people, labels, pairs, seed))}
        require(all(np.array_equal(arrays[name], oracle[name]) for name in MEMBERS),
                f"sham {seed} differs from independent oracle")
        require(receipt["artifact_raw_sha256"].get(path.name) == sha256_file(path),
                f"sham {seed} raw hash differs from bridge receipt")
        digest = semantic_sha256(arrays)
        require(receipt["artifact_semantic_sha256"].get(path.name) == digest,
                f"sham {seed} semantic hash differs from bridge receipt")
        require(digest != semantic_sha256(real) and digest not in semantic_hashes,
                "sham summary is real-identical or duplicated")
        semantic_hashes.add(digest)
        observed_hashes[str(seed)] = digest
    require(set(map(int, observed_hashes)) == set(SEEDS), "not exactly three shams")
    return {"stage": "M33_REF_LABEL_SHAM_TECHNICAL_INDEPENDENT_VERIFY",
            "status": "PASS_REF_LABEL_SHAM_TECHNICAL_ONLY_NON_CONSUMABLE",
            "scientific_evidence": False, "truth_read": False, "materialize": False,
            "ready": False, "training": False, "root_label": receipt["root_label"],
            "root_seed": receipt["root_seed"], "ref_people": 90, "ref_nodes": 180,
            "ref_people_by_ancestry": {a: 30 for a in ANCESTRIES},
            "selected_loci": int(len(selected["pos"])), "sham_semantic_sha256": observed_hashes,
            "tree_sequence_sha256": sha256_file(args.tree_sequence),
            "genetic_map_sha256": sha256_file(args.genetic_map),
            "pools_sha256": sha256_file(args.pools),
            "ref_pairs_sha256": sha256_file(args.ref_pairs),
            "panel_map_sha256": sha256_file(args.panel_map),
            "bridge_receipt_sha256": sha256_file(args.bridge_receipt),
            "verifier_git_commit": args.git_commit,
            "verifier_source_sha256": sha256_file(Path(__file__).resolve())}


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    require(path.parent.is_dir() and not path.exists(), "output must be new")
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.link(raw, path); os.unlink(raw); os.chmod(path, 0o400)
    finally:
        if os.path.exists(raw): os.unlink(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-loci", required=True, type=Path)
    parser.add_argument("--real-reference", required=True, type=Path)
    parser.add_argument("--sham", action="append", required=True)
    parser.add_argument("--tree-sequence", required=True, type=Path)
    parser.add_argument("--genetic-map", required=True, type=Path)
    parser.add_argument("--pools", required=True, type=Path)
    parser.add_argument("--ref-pairs", required=True, type=Path)
    parser.add_argument("--panel-map", required=True, type=Path)
    parser.add_argument("--bridge-receipt", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args(); result = verify(arguments); write_exclusive(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
