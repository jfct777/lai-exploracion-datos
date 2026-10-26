#!/usr/bin/env python3
"""Lazy chromosome dispatch for the 22 authenticated masked-genotype stores.

The module binds existing store, cell-axis and FIT-table identities.  It opens
genotype stores only when a caller requests a chromosome; it does not define a
training schedule, read all chromosomes, or grant scientific training.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

from r02_masked_genotypes import require
from r02_masked_store import MaskedStoreReader


SCHEMA = "r02_autosomal_masked_store_bindings_v1"
AUTOSOMES = tuple(range(1, 23))
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            label + " must be a lowercase SHA-256")
    return value


def _absolute_path(value: object, label: str) -> Path:
    require(isinstance(value, str) and value, label + " path is required")
    path = Path(value)
    require(path.is_absolute(), label + " path must be absolute")
    return path


@dataclass(frozen=True)
class AutosomalMaskedInput:
    chromosome: int
    store: Path
    manifest_sha256: str
    axis_directory: Path
    axis_receipt_sha256: str
    cells_sha256: str
    role: str
    fit_persons_sha256: str
    common_counted_allele: str
    rare_counted_allele: str
    common_fit_table: Path
    common_fit_table_sha256: str
    rare_fit_table: Path
    rare_fit_table_sha256: str

    @classmethod
    def from_json(cls, raw: object) -> "AutosomalMaskedInput":
        required = {
            "chromosome", "store", "manifest_sha256", "axis_directory",
            "axis_receipt_sha256", "cells_sha256", "role",
            "fit_persons_sha256", "common_counted_allele",
            "rare_counted_allele", "common_fit_table",
            "common_fit_table_sha256", "rare_fit_table",
            "rare_fit_table_sha256",
        }
        require(isinstance(raw, dict) and set(raw) == required,
                "autosomal input field inventory differs")
        chromosome = raw["chromosome"]
        require(type(chromosome) is int and chromosome in AUTOSOMES,
                "chromosome must be an integer from 1 through 22")
        require(raw["role"] == "FIT", "autosomal masked training input must be FIT")
        require(raw["common_counted_allele"] == "ALT",
                "common counted allele must remain ALT")
        require(raw["rare_counted_allele"] == "source_minor_DNABR2723",
                "rare counted allele must remain source_minor_DNABR2723")
        return cls(
            chromosome=chromosome,
            store=_absolute_path(raw["store"], "store"),
            manifest_sha256=_digest(raw["manifest_sha256"], "store manifest"),
            axis_directory=_absolute_path(raw["axis_directory"], "axis directory"),
            axis_receipt_sha256=_digest(raw["axis_receipt_sha256"], "axis receipt"),
            cells_sha256=_digest(raw["cells_sha256"], "cells table"),
            role=raw["role"],
            fit_persons_sha256=_digest(raw["fit_persons_sha256"], "FIT roster"),
            common_counted_allele=raw["common_counted_allele"],
            rare_counted_allele=raw["rare_counted_allele"],
            common_fit_table=_absolute_path(raw["common_fit_table"], "common FIT table"),
            common_fit_table_sha256=_digest(
                raw["common_fit_table_sha256"], "common FIT table"),
            rare_fit_table=_absolute_path(raw["rare_fit_table"], "rare FIT table"),
            rare_fit_table_sha256=_digest(
                raw["rare_fit_table_sha256"], "rare FIT table"),
        )


def load_autosomal_inputs(path: str | Path, *, verify_source: bool = True
                           ) -> dict[int, AutosomalMaskedInput]:
    """Load metadata identities without opening a genotype chunk or FIT table."""
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text())
    require(isinstance(raw, dict) and set(raw) == {
        "schema", "status", "source_inputs", "chromosomes", "scope"},
        "autosomal binding manifest field inventory differs")
    require(raw["schema"] == SCHEMA and
            raw["status"] == "METADATA_BOUND_NOT_TRAINING_ADMISSION",
            "autosomal binding manifest schema or status differs")
    source = raw["source_inputs"]
    require(isinstance(source, dict) and set(source) == {"path", "sha256"},
            "source input identity differs")
    require(isinstance(source["path"], str) and source["path"],
            "source input path is required")
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = (manifest_path.parent / source_path).resolve()
    source_sha256 = _digest(source["sha256"], "source input")
    if verify_source:
        require(source_path.is_file() and _sha256(source_path) == source_sha256,
                "source AUTOSOME_INPUTS.json changed")
    entries = [AutosomalMaskedInput.from_json(value)
               for value in raw["chromosomes"]]
    by_chromosome = {entry.chromosome: entry for entry in entries}
    require(tuple(sorted(by_chromosome)) == AUTOSOMES and len(entries) == len(AUTOSOMES),
            "all 22 autosomes must occur exactly once")
    roster_hashes = {entry.fit_persons_sha256 for entry in entries}
    require(len(roster_hashes) == 1,
            "FIT roster/order identity must agree across all autosomes")
    if verify_source:
        source_raw = json.loads(source_path.read_text())
        require(source_raw.get("schema") == "r02_existing_autosomal_inputs_binding_v1"
                and source_raw.get("status") ==
                "METADATA_BOUND_NOT_TRAINING_ADMISSION",
                "source AUTOSOME_INPUTS.json schema or status differs")
        source_rows = source_raw.get("structure")
        require(isinstance(source_rows, list) and len(source_rows) == len(AUTOSOMES),
                "source AUTOSOME_INPUTS.json does not bind 22 stores")
        source_by_chromosome = {row.get("chromosome"): row for row in source_rows}
        require(tuple(sorted(source_by_chromosome)) == AUTOSOMES,
                "source chromosome inventory differs")
        for chromosome, entry in by_chromosome.items():
            original = source_by_chromosome[chromosome]
            common = original.get("fit_tables", {}).get("common", {})
            rare = original.get("fit_tables", {}).get("rare", {})
            require(
                Path(original.get("store_manifest", {}).get("path", "")).resolve().parent
                == entry.store
                and original.get("store_manifest", {}).get("sha256")
                == entry.manifest_sha256
                and Path(original.get("cell_axis_receipt", {}).get("path", "")).resolve().parent
                == entry.axis_directory
                and original.get("cell_axis_receipt", {}).get("sha256")
                == entry.axis_receipt_sha256
                and common.get("fit_persons_sha256") == entry.fit_persons_sha256
                and rare.get("fit_persons_sha256") == entry.fit_persons_sha256
                and common.get("allele_definition") == entry.common_counted_allele
                and rare.get("allele_definition") == entry.rare_counted_allele
                and Path(common.get("table", "")).resolve() == entry.common_fit_table
                and common.get("recorded_table_sha256")
                == entry.common_fit_table_sha256
                and Path(rare.get("table", "")).resolve() == entry.rare_fit_table
                and rare.get("recorded_table_sha256") == entry.rare_fit_table_sha256,
                f"chromosome {chromosome} differs from AUTOSOME_INPUTS.json")
    require(raw["scope"] == {
        "genotypes_read": False,
        "fit_tables_read": False,
        "training_started": False,
        "evaluate_opened": False,
    }, "autosomal binding scope differs")
    return by_chromosome


class AutosomalMaskedStoreReaders(Mapping[int, MaskedStoreReader]):
    """Mapping used by ``fit_cell_loader`` with an LRU of open chromosomes."""

    def __init__(self, inputs: Mapping[int, AutosomalMaskedInput], *,
                 max_people: int, max_window_sites: int, max_cache_entries: int,
                 max_open_readers: int = 1,
                 reader_factory: Callable[..., MaskedStoreReader] = MaskedStoreReader):
        self._inputs = dict(inputs)
        require(tuple(sorted(self._inputs)) == AUTOSOMES,
                "reader dispatch requires all 22 autosomal bindings")
        require(type(max_open_readers) is int and
                0 < max_open_readers < len(AUTOSOMES),
                "max_open_readers must bound simultaneous stores below 22")
        require(type(max_people) is int and max_people > 0 and
                type(max_window_sites) is int and max_window_sites > 0 and
                type(max_cache_entries) is int and max_cache_entries > 0,
                "positive explicit reader resource bounds required")
        self.max_people = max_people
        self.max_window_sites = max_window_sites
        self.max_cache_entries = max_cache_entries
        self.max_open_readers = max_open_readers
        self._reader_factory = reader_factory
        self._open: OrderedDict[int, MaskedStoreReader] = OrderedDict()

    def __len__(self) -> int:
        return len(self._inputs)

    def __iter__(self) -> Iterator[int]:
        return iter(AUTOSOMES)

    def __getitem__(self, chromosome: int) -> MaskedStoreReader:
        return self.reader_for(chromosome)

    @property
    def open_chromosomes(self) -> tuple[int, ...]:
        return tuple(self._open)

    def clear(self) -> None:
        self._open.clear()

    def reader_for(self, chromosome: int) -> MaskedStoreReader:
        require(type(chromosome) is int and chromosome in self._inputs,
                "requested chromosome is absent from autosomal bindings")
        cached = self._open.pop(chromosome, None)
        if cached is not None:
            self._open[chromosome] = cached
            return cached
        binding = self._inputs[chromosome]
        while len(self._open) >= self.max_open_readers:
            self._open.popitem(last=False)
        reader = self._reader_factory(
            binding.store, binding.manifest_sha256, binding.axis_directory,
            binding.axis_receipt_sha256, role=binding.role,
            max_people=self.max_people, max_window_sites=self.max_window_sites,
            max_cache_entries=self.max_cache_entries)
        require(reader.chromosome == chromosome and reader.role == binding.role,
                "opened reader chromosome or role differs from binding")
        require(reader.reader.manifest["outputs_sha256"].get("source/persons.tsv") ==
                binding.fit_persons_sha256,
                "opened reader roster/order identity differs from FIT tables")
        require(reader.axis_receipt_sha256 == binding.axis_receipt_sha256 and
                reader.cells_sha256 == binding.cells_sha256,
                "opened reader cell-axis identity differs from binding")
        self._open[chromosome] = reader
        return reader


def autosomal_fit_cell_loader(readers: AutosomalMaskedStoreReaders, model_config, *,
                               context_exclusion_cm, device,
                               fragment_transform=None):
    """Delegate each example to the existing FIT-only cell loader."""
    from r02_masked_training_campaign import fit_cell_loader
    return fit_cell_loader(
        readers, model_config, context_exclusion_cm=context_exclusion_cm,
        device=device, fragment_transform=fragment_transform)
