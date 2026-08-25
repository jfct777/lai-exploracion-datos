#!/usr/bin/env python3
"""Truth-free primitives for the M33 SAFE_BRIDGE boundary.

This module only implements deterministic synthetic known-answer transforms.
It has no filesystem discovery, cloud access, truth loader, model or training
dependency.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANCESTRIES = ("AFR", "EUR", "ASIA")
REF_LABEL_SHAM_SEEDS = (79351217, 202307732, 1737132171)
REF_LABEL_SHAM_DOMAIN = b"DNABR_M33_PRE4_REF_LABEL_SHAM_V1|"
SAMPLE_DOMAIN = b"DNABR_M33_M0_SAMPLE_V1|"
LOCUS_FIELDS = ("chrom", "pos", "ref", "alt", "locus_id", "cM")
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sample_key(sample_id: str) -> bytes:
    require(isinstance(sample_id, str) and sample_id and sample_id == sample_id.strip(),
            "sample identifier is invalid")
    return hashlib.sha256(SAMPLE_DOMAIN + sample_id.encode("utf-8")).hexdigest().encode("ascii")


def strict_integer_array(value: Any, *, name: str, allowed: set[int]) -> np.ndarray:
    def validate(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        require(type(item) is int and item in allowed, f"{name} contains a non-integer or invalid state")

    validate(value)
    return np.asarray(value, dtype=np.int8)


def _locus(row: Mapping[str, Any]) -> tuple[int, int, str, str, int, float]:
    require(set(row) == set(LOCUS_FIELDS), "locus fields differ from the contract")
    chrom, pos, ref, alt = row["chrom"], row["pos"], row["ref"], row["alt"]
    locus_id, cm = row["locus_id"], row["cM"]
    require(type(chrom) is int and chrom == 22, "only canonical chromosome 22 is accepted")
    require(type(pos) is int and pos > 0, "locus position is invalid")
    require(isinstance(ref, str) and isinstance(alt, str) and len(ref) == len(alt) == 1,
            "only biallelic SNVs are accepted")
    require(ref in "ACGT" and alt in "ACGT" and ref != alt, "locus alleles are invalid")
    require(type(locus_id) is int and 0 <= locus_id < 2**64, "locus_id is invalid")
    require(type(cm) in (int, float) and np.isfinite(cm), "genetic position is invalid")
    return chrom, pos, ref, alt, locus_id, float(cm)


def canonical_loci(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, str, str, int, float]]:
    parsed = [_locus(row) for row in rows]
    require(parsed, "locus table is empty")
    keys = [(r[0], r[1], r[2], r[3]) for r in parsed]
    ids = [r[4] for r in parsed]
    require(len(keys) == len(set(keys)) and len(ids) == len(set(ids)), "duplicate locus key or locus_id")
    positions: dict[tuple[int, int], tuple[str, str]] = {}
    for chrom, pos, ref, alt, _locus_id, _cm in parsed:
        previous = positions.setdefault((chrom, pos), (ref, alt))
        require(previous == (ref, alt), "REF/ALT mismatch at the same chromosome and position")
    return sorted(parsed, key=lambda r: (r[5], r[1], r[4]))


def bind_genetic_map(
    locus_rows: Sequence[Mapping[str, Any]], map_records: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, str, str, int, float]]:
    loci = canonical_loci(locus_rows)
    mapping: dict[tuple[int, int], float] = {}
    for record in map_records:
        require(set(record) == {"chrom", "pos", "cM"}, "genetic-map record fields differ")
        require(type(record["chrom"]) is int and record["chrom"] == 22 and
                type(record["pos"]) is int and record["pos"] > 0 and
                type(record["cM"]) in (int, float) and not isinstance(record["cM"], bool) and
                np.isfinite(record["cM"]), "genetic-map record type or value is invalid")
        key = (record["chrom"], record["pos"])
        require(key not in mapping, "genetic-map position is duplicated")
        mapping[key] = float(record["cM"])
    for chrom, pos, _ref, _alt, _locus_id, cm in loci:
        require((chrom, pos) in mapping and abs(mapping[(chrom, pos)] - cm) <= 1e-12,
                "locus cM differs from the authenticated genetic map")
    return loci


def partition_incremental(
    selected_rows: Sequence[Mapping[str, Any]],
    catalog_rows: Sequence[Mapping[str, Any]],
    flare_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int, str, str, int, float]], list[tuple[int, int, str, str, int, float]]]:
    selected = canonical_loci(selected_rows)
    catalog = canonical_loci(catalog_rows)
    flare = canonical_loci(flare_rows)
    catalog_by_key = {(r[0], r[1], r[2], r[3]): r for r in catalog}
    flare_by_key = {(r[0], r[1], r[2], r[3]): r for r in flare}
    flare_by_position = {(r[0], r[1]): (r[2], r[3]) for r in flare}
    incremental, overlap = [], []
    for row in selected:
        key = (row[0], row[1], row[2], row[3])
        require(key in catalog_by_key and catalog_by_key[key][4] == row[4],
                "selected locus is absent or ambiguous in the authenticated rare catalog")
        positional = flare_by_position.get((row[0], row[1]))
        require(positional is None or positional == (row[2], row[3]),
                "rare/common REF/ALT mismatch at the same position")
        (overlap if key in flare_by_key else incremental).append(row)
    require(len(selected) == len(incremental) + len(overlap), "locus partition is incomplete")
    return incremental, overlap


def orient_target(raw_haplotypes: np.ndarray, minor_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(raw_haplotypes)
    codes = np.asarray(minor_codes)
    require(states.ndim == 3 and states.shape[2] == 2, "TARGET states must be sample x locus x haplotype")
    require(codes.shape == (states.shape[1],), "minor-code axis differs from TARGET loci")
    require(np.all(np.isin(codes, [0, 1])) and codes.dtype.kind in "iu", "minor code is outside 0/1")
    require(np.all(np.isin(states, [-1, 0, 1])) and states.dtype.kind in "iu",
            "TARGET raw state is outside missing/0/1")
    observed = np.all(states >= 0, axis=2)
    dosage = np.sum(states == codes[None, :, None], axis=2).astype(np.int8)
    dosage[~observed] = 0
    return np.ascontiguousarray(dosage), np.ascontiguousarray(observed.astype(np.uint8))


def parse_target_state_records(
    records: Sequence[Mapping[str, Any]], sample_ids: Sequence[str],
    loci: Sequence[tuple[int, int, str, str, int, float]],
) -> np.ndarray:
    fields = {"sample_id", "locus_id", "haplotype", "state"}
    require(sample_ids and len(sample_ids) == len(set(sample_ids)), "TARGET sample axis is invalid")
    sample_index = {sample: index for index, sample in enumerate(sample_ids)}
    locus_index = {row[4]: index for index, row in enumerate(loci)}
    values = np.empty((len(sample_ids), len(loci), 2), dtype=np.int8)
    seen: set[tuple[str, int, int]] = set()
    for record in records:
        require(set(record) == fields and isinstance(record["sample_id"], str) and
                type(record["locus_id"]) is int and type(record["haplotype"]) is int,
                "TARGET state record fields or types differ")
        sample, locus_id, haplotype = record["sample_id"], record["locus_id"], record["haplotype"]
        require(sample in sample_index and locus_id in locus_index and haplotype in (0, 1),
                "TARGET state record is outside authenticated axes")
        state = strict_integer_array([record["state"]], name="TARGET state", allowed={-1, 0, 1})[0]
        key = (sample, locus_id, haplotype)
        require(key not in seen, "TARGET state record is duplicated")
        seen.add(key)
        values[sample_index[sample], locus_index[locus_id], haplotype] = state
    require(len(seen) == len(sample_ids) * len(loci) * 2, "TARGET state Cartesian axis is incomplete")
    return values


def parse_reference_state_records(
    records: Sequence[Mapping[str, Any]], expected_ref_records: Sequence[Mapping[str, Any]],
    loci: Sequence[tuple[int, int, str, str, int, float]],
) -> tuple[np.ndarray, list[int], list[str], list[str]]:
    node_metadata: dict[int, tuple[str, str]] = {}
    for record in expected_ref_records:
        require(set(record) == {"node_id", "person_id", "ancestry"} and
                type(record["node_id"]) is int and isinstance(record["person_id"], str) and
                isinstance(record["ancestry"], str), "REF expected-record fields or types differ")
        require(record["node_id"] not in node_metadata, "REF expected node is duplicated")
        node_metadata[record["node_id"]] = (record["person_id"], record["ancestry"])
    node_ids = sorted(node_metadata)
    node_index = {node: index for index, node in enumerate(node_ids)}
    locus_index = {row[4]: index for index, row in enumerate(loci)}
    values = np.empty((len(node_ids), len(loci)), dtype=np.int8)
    seen: set[tuple[int, int]] = set()
    for record in records:
        require(set(record) == {"node_id", "locus_id", "state"} and
                type(record["node_id"]) is int and type(record["locus_id"]) is int,
                "REF state record fields or types differ")
        node, locus_id = record["node_id"], record["locus_id"]
        require(node in node_index and locus_id in locus_index, "REF state record is outside authenticated axes")
        state = strict_integer_array([record["state"]], name="REF state", allowed={-1, 0, 1})[0]
        key = (node, locus_id)
        require(key not in seen, "REF state record is duplicated")
        seen.add(key)
        values[node_index[node], locus_index[locus_id]] = state
    require(len(seen) == len(node_ids) * len(loci), "REF state Cartesian axis is incomplete")
    return (values, node_ids, [node_metadata[node][0] for node in node_ids],
            [node_metadata[node][1] for node in node_ids])


def summarize_reference(
    raw_states: np.ndarray,
    minor_codes: np.ndarray,
    node_ids: Sequence[int],
    node_person_ids: Sequence[str],
    node_ancestries: Sequence[str],
    expected_ref_records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    states = np.asarray(raw_states)
    codes = np.asarray(minor_codes)
    require(states.ndim == 2, "REF states must be node x locus")
    require(codes.shape == (states.shape[1],), "minor-code axis differs from REF loci")
    require(np.all(np.isin(codes, [0, 1])) and codes.dtype.kind in "iu", "minor code is outside 0/1")
    require(np.all(np.isin(states, [-1, 0, 1])) and states.dtype.kind in "iu",
            "REF raw state is outside missing/0/1")
    require(len(node_ids) == len(node_person_ids) == len(node_ancestries) == states.shape[0],
            "REF node metadata axis differs from genotype rows")
    require(all(set(row) == {"node_id", "person_id", "ancestry"} for row in expected_ref_records),
            "REF expected-record fields differ")
    require(all(type(node) is int for node in node_ids) and
            all(type(row["node_id"]) is int for row in expected_ref_records),
            "REF node ID must be an exact integer")
    require(all(isinstance(person, str) and person for person in node_person_ids) and
            all(isinstance(ancestry, str) for ancestry in node_ancestries) and
            all(isinstance(row["person_id"], str) and row["person_id"] and
                isinstance(row["ancestry"], str) for row in expected_ref_records),
            "REF person/ancestry metadata type is invalid")
    actual = tuple((node, person, ancestry)
                   for node, person, ancestry in zip(node_ids, node_person_ids, node_ancestries))
    expected = tuple((row["node_id"], row["person_id"], row["ancestry"])
                     for row in expected_ref_records)
    require(len({row[0] for row in actual}) == len(actual) and
            len({row[0] for row in expected}) == len(expected), "REF node is duplicated")
    require(set(actual) == set(expected),
            "contributing node/person/ancestry records are not exactly the authenticated REF records")
    person_to_nodes: dict[str, list[tuple[int, str]]] = {}
    for node, person, ancestry in actual:
        require(ancestry in ANCESTRIES and person, "REF person or ancestry is invalid")
        person_to_nodes.setdefault(person, []).append((node, ancestry))
    require(all(len(rows) == 2 and rows[0][1] == rows[1][1] for rows in person_to_nodes.values()),
            "each REF person must map to exactly two nodes in one ancestry")
    callable_mask = states >= 0
    callable_an = np.vstack([
        callable_mask[np.asarray(node_ancestries) == ancestry].sum(axis=0) for ancestry in ANCESTRIES
    ]).astype("<u2")
    minor_ac = np.vstack([
        ((states[np.asarray(node_ancestries) == ancestry] == codes[None, :]) &
         callable_mask[np.asarray(node_ancestries) == ancestry]).sum(axis=0)
        for ancestry in ANCESTRIES
    ]).astype("<u2")
    observed = (callable_an > 0).astype("|u1")
    no_support = ((callable_an > 0) & (minor_ac == 0)).astype("|u1")
    minor_af = np.divide(minor_ac, callable_an, out=np.zeros_like(minor_ac, dtype="<f8"),
                         where=callable_an > 0)
    return {
        "minor_ac": np.ascontiguousarray(minor_ac),
        "callable_an": np.ascontiguousarray(callable_an),
        "minor_af": np.ascontiguousarray(minor_af),
        "observed_mask": np.ascontiguousarray(observed),
        "no_support": np.ascontiguousarray(no_support),
    }


def permute_diploid_reference_labels(
    node_ids: Sequence[int],
    node_person_ids: Sequence[str],
    node_ancestries: Sequence[str],
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Permute complete-person REF labels with a version-stable hash ordering.

    The two haploid nodes of one person always receive the same permuted label.
    Only ancestry labels move; node order and genotypes are not touched.
    """
    require(type(seed) is int and 0 <= seed < 2**31, "REF-label sham seed is invalid")
    require(len(node_ids) == len(node_person_ids) == len(node_ancestries) and node_ids,
            "REF-label sham metadata axes differ")
    require(len(set(node_ids)) == len(node_ids), "REF-label sham node IDs are duplicated")
    people: dict[str, list[int]] = {}
    for index, (person, ancestry) in enumerate(zip(node_person_ids, node_ancestries)):
        require(isinstance(person, str) and person and ancestry in ANCESTRIES,
                "REF-label sham person or ancestry is invalid")
        people.setdefault(person, []).append(index)
    require(len(people) >= 2, "REF-label sham needs at least two diploid people")
    for indices in people.values():
        require(len(indices) == 2 and
                node_ancestries[indices[0]] == node_ancestries[indices[1]],
                "REF-label sham requires two same-ancestry nodes per person")

    ordered_people = sorted(
        people,
        key=lambda person: (person, tuple(sorted(node_ids[index] for index in people[person]))),
    )
    original = [node_ancestries[people[person][0]] for person in ordered_people]

    def rank_key(index: int) -> bytes:
        person = ordered_people[index]
        nodes = ",".join(str(value) for value in sorted(node_ids[i] for i in people[person]))
        payload = (REF_LABEL_SHAM_DOMAIN + str(seed).encode("ascii") + b"|" +
                   person.encode("utf-8") + b"|" + nodes.encode("ascii"))
        return hashlib.sha256(payload).digest()

    source_order = sorted(range(len(ordered_people)), key=lambda index: (rank_key(index), index))
    permuted = [original[index] for index in source_order]
    if permuted == original:
        for shift in range(1, len(original)):
            candidate = original[shift:] + original[:shift]
            if candidate != original:
                permuted = candidate
                break
    require(permuted != original and sorted(permuted) == sorted(original),
            "REF-label sham is identity or changed ancestry group sizes")
    require(any(before != after for before, after in zip(original, permuted)),
            "REF-label sham did not reassign any person across ancestry")

    permuted_nodes = list(node_ancestries)
    transition = {source: {target: 0 for target in ANCESTRIES} for source in ANCESTRIES}
    moved = 0
    for person, before, after in zip(ordered_people, original, permuted):
        transition[before][after] += 1
        moved += int(before != after)
        for index in people[person]:
            permuted_nodes[index] = after
    require(all(
        sum(transition[source][target] for target in ANCESTRIES if target != source) > 0
        for source in ANCESTRIES
    ), "REF-label sham left at least one ancestry without cross-ancestry reassignment")
    require(all(permuted_nodes[indices[0]] == permuted_nodes[indices[1]]
                for indices in people.values()),
            "REF-label sham split the two nodes of one person")
    assignment_payload = json.dumps(
        {"seed": seed, "original": original, "permuted": permuted},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return permuted_nodes, {
        "seed": seed,
        "person_count": len(ordered_people),
        "moved_person_count": moved,
        "ancestry_transition_counts": transition,
        "assignment_sha256": hashlib.sha256(assignment_payload).hexdigest(),
    }


def summarize_reference_label_shams(
    raw_states: np.ndarray,
    minor_codes: np.ndarray,
    node_ids: Sequence[int],
    node_person_ids: Sequence[str],
    node_ancestries: Sequence[str],
    expected_ref_records: Sequence[Mapping[str, Any]],
    seeds: Sequence[int] = REF_LABEL_SHAM_SEEDS,
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Recompute three aggregated REF summaries after complete-person label shams."""
    require(tuple(seeds) == REF_LABEL_SHAM_SEEDS,
            "REF-label sham seeds differ from PRE4")
    original = summarize_reference(
        raw_states, minor_codes, node_ids, node_person_ids, node_ancestries,
        expected_ref_records,
    )
    genotype_before = semantic_arrays_sha256(
        "m33_ref_label_sham_raw_ref_v1",
        {"raw_states": np.ascontiguousarray(raw_states),
         "minor_codes": np.ascontiguousarray(minor_codes)},
    )
    summaries: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    assignment_hashes: set[str] = set()
    for seed in seeds:
        permuted_nodes, diagnostic = permute_diploid_reference_labels(
            node_ids, node_person_ids, node_ancestries, seed,
        )
        permuted_expected = [
            {"node_id": node, "person_id": person, "ancestry": ancestry}
            for node, person, ancestry in zip(node_ids, node_person_ids, permuted_nodes)
        ]
        summary = summarize_reference(
            raw_states, minor_codes, node_ids, node_person_ids, permuted_nodes,
            permuted_expected,
        )
        require(np.array_equal(summary["minor_ac"].sum(axis=0),
                               original["minor_ac"].sum(axis=0)) and
                np.array_equal(summary["callable_an"].sum(axis=0),
                               original["callable_an"].sum(axis=0)),
                "REF-label sham changed pooled AC or AN")
        genotype_after = semantic_arrays_sha256(
            "m33_ref_label_sham_raw_ref_v1",
            {"raw_states": np.ascontiguousarray(raw_states),
             "minor_codes": np.ascontiguousarray(minor_codes)},
        )
        require(genotype_after == genotype_before,
                "REF-label sham changed raw genotypes or minor codes")
        require(diagnostic["assignment_sha256"] not in assignment_hashes,
                "REF-label sham assignments are duplicated")
        assignment_hashes.add(diagnostic["assignment_sha256"])
        diagnostic["raw_genotype_minor_code_sha256"] = genotype_before
        summaries[seed] = summary
        diagnostics.append(diagnostic)
    return summaries, diagnostics


def summarize_diploid_dosage_reference_label_shams(
    minor_dosage: np.ndarray,
    people: Sequence[str],
    labels: Sequence[str],
    person_to_node_ids: Mapping[str, Sequence[int]],
    *,
    seeds: Sequence[int] = REF_LABEL_SHAM_SEEDS,
    expected_people_by_ancestry: Mapping[str, int] | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Aggregate complete-person REF-label shams from diploid minor dosage.

    ``minor_dosage`` is locus x person and must contain exact integer states
    0/1/2 with no missing values. The authenticated person-to-node mapping is
    used only to bind both homologous nodes to the same permuted label; no
    person or node identifier is returned.
    """
    dosage = np.asarray(minor_dosage)
    person_axis = tuple(people)
    label_axis = tuple(labels)
    require(tuple(seeds) == REF_LABEL_SHAM_SEEDS,
            "REF-label sham seeds differ from PRE4")
    require(dosage.ndim == 2 and dosage.shape[1] == len(person_axis),
            "diploid REF dosage must be locus x person")
    require(dosage.dtype.kind in "iu" and np.all(np.isin(dosage, [0, 1, 2])),
            "diploid REF dosage contains missing, non-integer or invalid state")
    require(person_axis and len(person_axis) == len(label_axis) and
            all(isinstance(person, str) and person and person == person.strip()
                for person in person_axis) and len(set(person_axis)) == len(person_axis),
            "diploid REF person axis is empty, duplicated or invalid")
    require(all(isinstance(label, str) and label in ANCESTRIES for label in label_axis),
            "diploid REF ancestry label is invalid")
    require(isinstance(person_to_node_ids, Mapping) and
            set(person_to_node_ids) == set(person_axis),
            "authenticated REF person-to-node mapping differs from the dosage axis")

    flat_nodes: list[int] = []
    flat_people: list[str] = []
    flat_labels: list[str] = []
    for person, label in zip(person_axis, label_axis):
        pair = person_to_node_ids[person]
        require(isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and
                len(pair) == 2 and all(type(node) is int for node in pair) and
                pair[0] != pair[1],
                "each authenticated REF person must map to two distinct integer nodes")
        flat_nodes.extend((pair[0], pair[1]))
        flat_people.extend((person, person))
        flat_labels.extend((label, label))
    require(len(set(flat_nodes)) == len(flat_nodes),
            "authenticated REF node is duplicated across people")

    original_counts = {ancestry: label_axis.count(ancestry) for ancestry in ANCESTRIES}
    if expected_people_by_ancestry is not None:
        require(set(expected_people_by_ancestry) == set(ANCESTRIES) and
                all(type(expected_people_by_ancestry[ancestry]) is int and
                    expected_people_by_ancestry[ancestry] > 0 for ancestry in ANCESTRIES),
                "expected REF ancestry counts are invalid")
        require(original_counts == dict(expected_people_by_ancestry),
                "REF ancestry/person counts differ from the required firewall")
    require(2 * len(person_axis) <= np.iinfo(np.uint16).max,
            "diploid REF cohort exceeds the uint16 summary schema")

    def aggregate(person_labels: Sequence[str]) -> dict[str, np.ndarray]:
        label_values = np.asarray(person_labels, dtype=object)
        minor_ac_u64 = np.vstack([
            dosage[:, label_values == ancestry].sum(axis=1, dtype=np.uint64)
            for ancestry in ANCESTRIES
        ])
        require(np.all(minor_ac_u64 <= np.iinfo(np.uint16).max),
                "diploid REF minor AC exceeds the uint16 summary schema")
        minor_ac = minor_ac_u64.astype("<u2")
        callable_an = np.vstack([
            np.full(dosage.shape[0], 2 * int(np.sum(label_values == ancestry)), dtype="<u2")
            for ancestry in ANCESTRIES
        ])
        require(np.all(minor_ac <= callable_an), "diploid REF minor AC exceeds callable AN")
        minor_af = np.divide(
            minor_ac, callable_an, out=np.zeros_like(minor_ac, dtype="<f8"),
            where=callable_an > 0,
        )
        observed = (callable_an > 0).astype("|u1")
        no_support = ((callable_an > 0) & (minor_ac == 0)).astype("|u1")
        return {
            "minor_ac": np.ascontiguousarray(minor_ac),
            "callable_an": np.ascontiguousarray(callable_an),
            "minor_af": np.ascontiguousarray(minor_af),
            "observed_mask": np.ascontiguousarray(observed),
            "no_support": np.ascontiguousarray(no_support),
        }

    original = aggregate(label_axis)
    original_summary_hash = semantic_arrays_sha256(
        "m33_ref_label_sham_diploid_reference_summary_v1", original,
    )
    genotype_hash = semantic_arrays_sha256(
        "m33_ref_label_sham_diploid_minor_dosage_v1",
        {"minor_dosage": np.ascontiguousarray(dosage)},
    )
    summaries: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    assignment_hashes: set[str] = set()
    summary_hashes: set[str] = set()
    for seed in seeds:
        permuted_nodes, diagnostic = permute_diploid_reference_labels(
            flat_nodes, flat_people, flat_labels, seed,
        )
        permuted_people: list[str] = []
        for index in range(len(person_axis)):
            first, second = permuted_nodes[2 * index:2 * index + 2]
            require(first == second,
                    "REF-label sham split the two authenticated nodes of one person")
            permuted_people.append(first)
        require({ancestry: permuted_people.count(ancestry) for ancestry in ANCESTRIES}
                == original_counts, "REF-label sham changed ancestry person counts")
        summary = aggregate(permuted_people)
        require(np.array_equal(summary["minor_ac"].sum(axis=0),
                               original["minor_ac"].sum(axis=0)) and
                np.array_equal(summary["callable_an"].sum(axis=0),
                               original["callable_an"].sum(axis=0)),
                "REF-label sham changed pooled AC or AN")
        require(diagnostic["assignment_sha256"] not in assignment_hashes,
                "REF-label sham assignments are duplicated")
        assignment_hashes.add(diagnostic["assignment_sha256"])
        summary_hash = semantic_arrays_sha256(
            "m33_ref_label_sham_diploid_reference_summary_v1", summary,
        )
        require(summary_hash != original_summary_hash,
                "REF-label sham summary is identical to the real REF summary")
        require(summary_hash not in summary_hashes,
                "REF-label sham summaries are duplicated")
        summary_hashes.add(summary_hash)
        diagnostic["raw_diploid_minor_dosage_sha256"] = genotype_hash
        diagnostic["aggregated_reference_summary_sha256"] = summary_hash
        diagnostic["ancestry_person_counts"] = original_counts
        summaries[seed] = summary
        diagnostics.append(diagnostic)
    return summaries, diagnostics


def sanitize_f0(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype="<f8")
    require(values.ndim == 4 and values.shape[1] == 2 and values.shape[3] == 3,
            "F0 must be sample x haplotype x marker x ancestry")
    require(np.all(np.isfinite(values)) and np.all(values >= 0), "F0 contains invalid probability")
    sums = values.sum(axis=3, keepdims=True)
    require(np.all((sums >= 0.98) & (sums <= 1.02)), "F0 probability sum is outside tolerance")
    normalized = (values / sums).astype("<f4")
    require(np.allclose(normalized.sum(axis=3), 1.0, atol=5e-6, rtol=0),
            "F0 float32 simplex check failed")
    return np.ascontiguousarray(normalized)


def parse_f0_records(
    records: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
    flare_loci: Sequence[tuple[int, int, str, str, int, float]],
    root_seed: int,
) -> np.ndarray:
    require(type(root_seed) is int and root_seed >= 0, "F0 root seed is invalid")
    fields = {"root_seed", "sample_id", "chrom", "pos", "ref", "alt", "ANP1", "ANP2"}
    require(records and len(sample_ids) == len(set(sample_ids)), "F0 sample axis is empty or duplicated")
    sample_index = {sample: index for index, sample in enumerate(sample_ids)}
    locus_index = {(row[0], row[1], row[2], row[3]): index for index, row in enumerate(flare_loci)}
    values = np.empty((len(sample_ids), 2, len(flare_loci), 3), dtype="<f8")
    seen: set[tuple[str, tuple[int, int, str, str]]] = set()
    for record in records:
        require(set(record) == fields, "F0 record contains missing, extra or forbidden source fields")
        require(type(record["root_seed"]) is int and record["root_seed"] == root_seed,
                "F0 root seed differs from the authenticated root")
        require(isinstance(record["sample_id"], str) and type(record["chrom"]) is int and
                type(record["pos"]) is int and isinstance(record["ref"], str) and
                isinstance(record["alt"], str), "F0 identity field type is invalid")
        sample = record["sample_id"]
        key = (record["chrom"], record["pos"], record["ref"], record["alt"])
        require(sample in sample_index and key in locus_index, "F0 sample or marker is outside authenticated axes")
        pair = (sample, key)
        require(pair not in seen, "F0 sample/marker record is duplicated")
        seen.add(pair)
        for haplotype, field in enumerate(("ANP1", "ANP2")):
            probability = record[field]
            require(isinstance(probability, list) and len(probability) == 3,
                    f"{field} must contain three ancestry probabilities")
            require(all(type(item) in (int, float) and not isinstance(item, bool) and np.isfinite(item)
                        for item in probability), f"{field} contains a non-numeric probability")
            values[sample_index[sample], haplotype, locus_index[key], :] = probability
    require(len(seen) == len(sample_ids) * len(flare_loci), "F0 Cartesian sample/marker axis is incomplete")
    return sanitize_f0(values)


def semantic_arrays_sha256(schema_id: str, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(schema_id.encode("utf-8") + b"\0")
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        require(array.dtype.kind != "O" and not np.issubdtype(array.dtype, np.flexible) or array.dtype.kind == "S",
                "object or unsupported flexible array")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    require(not path.exists(), "output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
            for name in sorted(arrays):
                array = np.ascontiguousarray(arrays[name])
                require(array.dtype.kind != "O", "object arrays are forbidden")
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, array, allow_pickle=False)
                member = zipfile.ZipInfo(f"{name}.npy", FIXED_ZIP_TIME)
                member.compress_type = zipfile.ZIP_STORED
                member.external_attr = 0o100444 << 16
                archive.writestr(member, buffer.getvalue())
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        os.chmod(path, 0o400)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), "output already exists")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
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
        os.chmod(path, 0o400)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        require(path.read_bytes() == encoded, "JSON reopen differs")
    finally:
        if temporary.exists():
            temporary.unlink()


def reopen_npz(path: Path, expected: Mapping[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as reopened:
        require(set(reopened.files) == set(expected), "NPZ members differ")
        for name, wanted in expected.items():
            got = reopened[name]
            require(got.dtype == wanted.dtype and got.shape == wanted.shape and np.array_equal(got, wanted),
                    f"NPZ member differs: {name}")
