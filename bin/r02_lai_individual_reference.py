#!/usr/bin/env python3
"""Read exact REF_TRAIN ALT dosages from a pinned reference-detail tar.

The reader never extracts archive members and never reads haplotype or FLARE
arrays.  Returned arrays exist only in memory and follow the requested locus
order and the authenticated reference-people order.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Iterable, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA = "r02_lai_reference_detail_v2"
INDEX_SCHEMA = "r02_lai_reference_chunk_interval_index_v1"
ANCESTRIES = ("African", "European", "Native_American")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _chromosome(value: object) -> str:
    text = str(value)
    if text.lower().startswith("chr"):
        text = text[3:]
    _require(text.isdigit() and 1 <= int(text) <= 22, "Expected an autosome from 1 through 22")
    return str(int(text))


def _key(value: Sequence[object]) -> tuple[str, int, str, str]:
    _require(len(value) == 4, "Each requested key must contain chromosome, bp, REF and ALT")
    chrom = _chromosome(value[0])
    _require(type(value[1]) is int and value[1] > 0, "Locus bp must be a positive integer")
    ref, alt = value[2], value[3]
    _require(isinstance(ref, str) and isinstance(alt, str) and ref and alt and ref != alt,
             "REF and ALT must be distinct non-empty strings")
    return chrom, value[1], ref, alt


@dataclass(frozen=True)
class ReferenceDosageBatch:
    """In-memory ALT dosage with axes ``keys x reference_ids``."""

    keys: tuple[tuple[str, int, str, str], ...]
    reference_ids: tuple[str, ...]
    reference_panels: tuple[str, ...]
    ancestries: tuple[str, ...]
    dosage: np.ndarray
    observed: np.ndarray


class IndividualReferenceArchive:
    """Authenticate a retained tar once and read explicit exact locus keys."""

    def __init__(
        self,
        archive_path: str | Path,
        archive_sha256: str,
        max_output_genotypes: int,
        chunk_indexes: Mapping[str, str | Path] | None = None,
    ) -> None:
        self.archive_path = Path(archive_path)
        _require(self.archive_path.is_file(), "Reference archive is absent")
        _require(len(archive_sha256) == 64 and all(c in "0123456789abcdef" for c in archive_sha256),
                 "Expected a lowercase archive SHA-256")
        _require(type(max_output_genotypes) is int and max_output_genotypes > 0,
                 "max_output_genotypes must be an explicit positive integer")
        _require(_sha256_file(self.archive_path) == archive_sha256, "Reference archive SHA-256 differs")
        self.archive_sha256 = archive_sha256
        self.max_output_genotypes = max_output_genotypes
        self._tar = tarfile.open(self.archive_path, "r:*")
        members: dict[str, tarfile.TarInfo] = {}
        for member in self._tar.getmembers():
            if member.name in members:
                self.close()
                raise ValueError("Duplicate tar member: " + member.name)
            members[member.name] = member
        self._members = members
        inventory = json.loads(self._member_bytes("INVENTORY.json"))
        _require(isinstance(inventory, list), "Reference inventory must be a list")
        self._inventory = {}
        for row in inventory:
            chrom = _chromosome(row.get("chromosome"))
            _require(chrom not in self._inventory, "Duplicate chromosome in reference inventory")
            expected = row.get("manifest_sha256")
            _require(isinstance(expected, str) and len(expected) == 64,
                     "Reference inventory manifest SHA-256 is absent")
            self._inventory[chrom] = row
        self._index_paths = {_chromosome(chrom): Path(path)
                             for chrom, path in (chunk_indexes or {}).items()}
        self._manifests: dict[str, dict] = {}
        self._indexes: dict[str, list[dict]] = {}
        self._discovered_intervals: dict[str, dict[str, tuple[int, int]]] = {}
        self._reference_ids: tuple[str, ...] | None = None
        self._reference_panels: tuple[str, ...] | None = None

    def close(self) -> None:
        tar = getattr(self, "_tar", None)
        if tar is not None:
            tar.close()
            self._tar = None

    def __enter__(self) -> "IndividualReferenceArchive":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _member_bytes(self, name: str) -> bytes:
        _require(self._tar is not None, "Reference archive reader is closed")
        member = self._members.get(name)
        _require(member is not None and member.isfile(), "Required regular tar member is absent: " + name)
        stream = self._tar.extractfile(member)
        _require(stream is not None, "Tar member stream is unavailable: " + name)
        value = stream.read()
        _require(len(value) == member.size, "Tar member size differs: " + name)
        return value

    def _manifest(self, chrom: str) -> dict:
        if chrom in self._manifests:
            return self._manifests[chrom]
        record = self._inventory.get(chrom)
        _require(record is not None, "Autosome is absent from reference inventory: " + chrom)
        name = f"published/chr{chrom}/manifest.json"
        raw = self._member_bytes(name)
        _require(_sha256_bytes(raw) == record["manifest_sha256"],
                 "Reference manifest SHA-256 differs for chromosome " + chrom)
        manifest = json.loads(raw)
        _require(manifest.get("schema") == MANIFEST_SCHEMA
                 and _chromosome(manifest.get("chromosome")) == chrom
                 and manifest.get("role") == "REF_TRAIN"
                 and manifest.get("training_allowed") is False,
                 "Reference manifest identity/status differs for chromosome " + chrom)
        chunks = manifest.get("chunks")
        _require(isinstance(chunks, list) and chunks, "Reference manifest has no chunks")
        paths = [row.get("path") for row in chunks]
        _require(len(paths) == len(set(paths)) and all(isinstance(path, str) for path in paths),
                 "Reference manifest contains duplicate or invalid chunk paths")
        people_name = f"published/chr{chrom}/{manifest['people_file']['path']}"
        people_raw = self._member_bytes(people_name)
        _require(_sha256_bytes(people_raw) == manifest["people_file"]["sha256"],
                 "Reference people SHA-256 differs for chromosome " + chrom)
        reader = csv.DictReader(io.StringIO(people_raw.decode("utf-8")), delimiter="\t")
        rows = list(reader)
        _require(reader.fieldnames is not None
                 and {"sample_id", "role", "ancestry"} <= set(reader.fieldnames),
                 "Reference people columns differ")
        ids = tuple(row["sample_id"] for row in rows)
        panels = tuple(row["ancestry"] for row in rows)
        _require(len(ids) == manifest.get("people") and len(ids) == len(set(ids))
                 and all(row["role"] == "REF_TRAIN" for row in rows)
                 and set(panels) == set(ANCESTRIES),
                 "Reference people count, identity, role or panel differs")
        if self._reference_ids is None:
            self._reference_ids = ids
            self._reference_panels = panels
        else:
            _require(ids == self._reference_ids and panels == self._reference_panels,
                     "Reference people order or panel labels differ between chromosomes")
        self._manifests[chrom] = manifest
        return manifest

    def _index(self, chrom: str, manifest: dict) -> list[dict] | None:
        if chrom not in self._index_paths:
            return None
        if chrom in self._indexes:
            return self._indexes[chrom]
        path = self._index_paths[chrom]
        _require(path.is_file(), "Reference chunk index is absent: " + str(path))
        index = json.loads(path.read_text())
        _require(index.get("schema") == INDEX_SCHEMA
                 and _chromosome(index.get("chromosome")) == chrom
                 and index.get("manifest_sha256") == self._inventory[chrom]["manifest_sha256"],
                 "Reference chunk index identity differs")
        manifest_rows = {row["path"]: row for row in manifest["chunks"]}
        rows = index.get("chunks")
        _require(isinstance(rows, list) and len(rows) == len(manifest_rows),
                 "Reference chunk index coverage differs")
        seen: set[str] = set()
        for row in rows:
            path_value = row.get("path")
            source = manifest_rows.get(path_value)
            _require(source is not None and path_value not in seen,
                     "Reference chunk index path is absent or duplicated")
            seen.add(path_value)
            _require(all(row.get(field) == source.get(field)
                         for field in ("path", "sha256", "bytes", "loci")),
                     "Reference chunk index metadata differs")
            _require(type(row.get("first_bp")) is int and type(row.get("last_bp")) is int
                     and 0 < row["first_bp"] <= row["last_bp"],
                     "Reference chunk interval is invalid")
        self._indexes[chrom] = rows
        return rows

    def _chunk_arrays(self, chrom: str, row: dict, wanted_positions: set[int]):
        name = f"published/chr{chrom}/{row['path']}"
        raw = self._member_bytes(name)
        _require(len(raw) == row.get("bytes") and _sha256_bytes(raw) == row.get("sha256"),
                 "Reference chunk differs: " + name)
        with np.load(io.BytesIO(raw), allow_pickle=False) as stored:
            required = {"chromosome", "position_bp", "ref", "alt", "dosage", "observed"}
            _require(required <= set(stored.files), "Reference chunk arrays differ: " + name)
            position = np.asarray(stored["position_bp"])
            chromosome = np.asarray(stored["chromosome"]).astype(str)
            ref = np.asarray(stored["ref"]).astype(str)
            alt = np.asarray(stored["alt"]).astype(str)
            loci = row.get("loci")
            _require(position.ndim == chromosome.ndim == ref.ndim == alt.ndim == 1
                     and len(position) == len(chromosome) == len(ref) == len(alt) == loci
                     and np.issubdtype(position.dtype, np.integer)
                     and np.all(position > 0)
                     and np.all(chromosome == chrom),
                     "Reference locus axes differ: " + name)
            intervals = self._discovered_intervals.setdefault(chrom, {})
            intervals[row["path"]] = (int(position.min()), int(position.max()))
            selected = np.flatnonzero(np.isin(position, list(wanted_positions)))
            if not len(selected):
                return []
            dosage = np.asarray(stored["dosage"])
            observed = np.asarray(stored["observed"])
            people = len(self._reference_ids or ())
            _require(dosage.shape == observed.shape == (loci, people)
                     and dosage.dtype == np.int8 and observed.dtype == np.bool_
                     and np.isin(dosage[observed], [0, 1, 2]).all(),
                     "Reference dosage or observed axes differ: " + name)
            return [((chrom, int(position[i]), ref[i], alt[i]),
                     dosage[i].copy(), observed[i].copy()) for i in selected]

    def _read_chromosome(self, chrom: str, keys: Sequence[tuple[str, int, str, str]]):
        manifest = self._manifest(chrom)
        positions = {key[1] for key in keys}
        index = self._index(chrom, manifest)
        if index is not None:
            rows = [row for row in index
                    if any(row["first_bp"] <= bp <= row["last_bp"] for bp in positions)]
        else:
            known = self._discovered_intervals.get(chrom, {})
            if len(known) == len(manifest["chunks"]):
                rows = [row for row in manifest["chunks"]
                        if any(known[row["path"]][0] <= bp <= known[row["path"]][1]
                               for bp in positions)]
            else:
                # First unindexed query reads only locus arrays for this autosome;
                # genotype arrays are decompressed only for matching positions.
                rows = manifest["chunks"]
        found: dict[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]] = {}
        alleles: dict[int, set[tuple[str, str]]] = {bp: set() for bp in positions}
        for row in rows:
            for source_key, dosage, observed in self._chunk_arrays(chrom, row, positions):
                alleles[source_key[1]].add((source_key[2], source_key[3]))
                _require(source_key not in found, "Duplicate reference locus: " + repr(source_key))
                found[source_key] = dosage, observed
        for requested in keys:
            if requested not in found:
                if alleles[requested[1]]:
                    raise ValueError("Reference allele discrepancy for locus: " + repr(requested))
                raise ValueError("Reference locus is absent; cannot impute REF: " + repr(requested))
        return found

    def read(self, keys: Iterable[Sequence[object]]) -> ReferenceDosageBatch:
        requested = tuple(_key(value) for value in keys)
        _require(requested, "At least one explicit reference locus key is required")
        _require(len(requested) == len(set(requested)), "Duplicate requested reference locus")
        by_chromosome: dict[str, list[tuple[str, int, str, str]]] = {}
        for value in requested:
            by_chromosome.setdefault(value[0], []).append(value)
        # Authenticate every requested chromosome's people axis before any
        # genotype array is loaded, then enforce the technical in-memory bound.
        for chrom in by_chromosome:
            self._manifest(chrom)
        people = len(self._reference_ids or ())
        _require(len(requested) * people <= self.max_output_genotypes,
                 "Requested locus-by-reference output exceeds max_output_genotypes")
        found = {}
        for chrom, chrom_keys in by_chromosome.items():
            found.update(self._read_chromosome(chrom, chrom_keys))
        dosage = np.stack([found[value][0] for value in requested], axis=0)
        observed = np.stack([found[value][1] for value in requested], axis=0)
        dosage.setflags(write=False)
        observed.setflags(write=False)
        return ReferenceDosageBatch(requested, self._reference_ids or (),
                                    self._reference_panels or (), ANCESTRIES,
                                    dosage, observed)


def read_reference_keys(
    archive_path: str | Path,
    archive_sha256: str,
    keys: Iterable[Sequence[object]],
    max_output_genotypes: int,
    chunk_indexes: Mapping[str, str | Path] | None = None,
) -> ReferenceDosageBatch:
    """Convenience wrapper for one authenticated, in-memory read."""
    with IndividualReferenceArchive(archive_path, archive_sha256,
                                    max_output_genotypes, chunk_indexes) as archive:
        return archive.read(keys)
