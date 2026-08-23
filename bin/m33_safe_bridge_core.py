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
