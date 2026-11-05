#!/usr/bin/env python3
"""Authenticate historical NONE metadata and produce duration fragments.

The inventory path reads only the small query/identity arrays needed to declare
the complete chromosome axis.  The fragment iterator uses the existing full
factorized-link loader, slices people before the neural forward pass, and keeps
label availability separate from query geometry.

This is an input adapter.  ``max_cache_bytes`` bounds retained NumPy array bytes,
not temporary loader workspace, the model, or neural activations.  It does not
choose duration stages, rates, ancestry switches or a start prior, and it does
not implement an optimizer/checkpoint
loop.  A limit on returned logits does not limit the model's internal
activations.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Iterator, Mapping, Sequence
import zipfile

import numpy as np
import torch

from r02_lai_duration_model import (
    DeclaredSequenceAxis,
    DurationFragment,
    SequenceDeclaration,
)


SHA256 = re.compile(r"^[0-9a-f]{64}$")
INVENTORY_SCHEMA = "r02_lai_duration_none_metadata_inventory_v1"
METADATA_ARRAYS = (
    "person_id", "role", "component_id", "query_global_index", "query_cm",
    "query_chrom", "query_core", "query_valid",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strings(value: np.ndarray) -> tuple[str, ...]:
    array = np.asarray(value)
    _require(array.ndim == 1, "identity arrays must be one-dimensional")
    return tuple(array.astype(str).tolist())


@dataclass(frozen=True)
class DurationRegionSource:
    region_id: str
    entry: Mapping[str, object]
    source_receipt_path: str
    source_receipt_sha256: str
    core_global_indices: tuple[int, ...]
    core_positions_cm: tuple[float, ...]
    observed_core_labels_by_person: tuple[int, ...]
    metadata_bytes_loaded: int
    metadata_archive_uncompressed_bytes_read: int


@dataclass(frozen=True)
class DurationCaseInventory:
    contract_path: str
    contract_sha256: str
    case_id: str
    chromosome: int
    condition: str
    role: str
    person_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    query_global_indices: tuple[int, ...]
    query_positions_cm: tuple[float, ...]
    observed_labels_by_person: tuple[int, ...]
    regions: tuple[DurationRegionSource, ...]
    metadata_bytes_loaded: int
    metadata_archive_uncompressed_bytes_read: int
    inventory_sha256: str


@dataclass(frozen=True)
class DurationProducedChunk:
    declared_axis: DeclaredSequenceAxis
    fragments: tuple[DurationFragment, ...]
    person_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    global_label_denominator: int
    chunk_observed_labels: int
    inventory_sha256: str
    estimated_cache_bytes: int
    cache_bytes: int
    loader_calls: int


def _read_json(path: str | Path, expected_sha256: str) -> dict:
    from r02_lai_training_campaign import read_json
    _require(isinstance(expected_sha256, str) and SHA256.fullmatch(expected_sha256) is not None,
             "a lowercase SHA-256 is required")
    value = read_json(path, expected_sha256)
    _require(isinstance(value, dict), "JSON document must be an object")
    return value


def _verify_array(array: np.ndarray, expected: Mapping[str, str], name: str) -> None:
    from r02_lai_regional_channel_bridge import array_sha256
    _require(name in expected and array_sha256(array, np) == expected[name],
             f"authenticated reconstructed array differs: {name}")


def _npz_uncompressed_bytes_before_decode(path: Path, names: Sequence[str],
                                          max_bytes: int) -> int:
    """Read ZIP headers and reject selected members before NumPy decodes them."""
    _require(type(max_bytes) is int and max_bytes > 0,
             "metadata decode byte limit must be a positive integer")
    with zipfile.ZipFile(path) as archive:
        members: dict[str, list[zipfile.ZipInfo]] = {}
        for info in archive.infolist():
            members.setdefault(info.filename, []).append(info)
        selected = []
        for name in names:
            key = name + ".npy"
            _require(len(members.get(key, [])) == 1,
                     f"NPZ member is absent or duplicated: {key}")
            selected.append(members[key][0])
        total = sum(info.file_size for info in selected)
    _require(total <= max_bytes,
             "selected metadata arrays exceed contract max_batch_bytes before decode")
    return total


def _load_region_metadata(request: dict, expected: Mapping[str, str],
                          max_decode_bytes: int) -> tuple[dict[str, np.ndarray], int]:
    descriptor = request.get("regional_batch")
    _require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"}
             and SHA256.fullmatch(descriptor.get("sha256", "")) is not None,
             "regional batch descriptor differs")
    path = Path(descriptor["path"])
    _require(path.is_file(), "regional batch is absent")
    archive_bytes = _npz_uncompressed_bytes_before_decode(
        path, METADATA_ARRAYS, max_decode_bytes)
    # Its multi-gigabyte neighbours and the NPZ container itself are deliberately
    # not hashed here.  Each loaded array is checked against the authenticated
    # reconstructed-array hash; the full loader performs container/source checks.
    with np.load(path, allow_pickle=False) as archive:
        _require(set(METADATA_ARRAYS) <= set(archive.files),
                 "regional batch lacks duration metadata")
        arrays = {name: archive[name] for name in METADATA_ARRAYS}
    for name, value in arrays.items():
        _verify_array(value, expected, name)
    return arrays, archive_bytes


def _load_truth_metadata(request: dict, regional: Mapping[str, np.ndarray],
                         expected: Mapping[str, str], max_decode_bytes: int
                         ) -> tuple[np.ndarray, np.ndarray, int]:
    descriptor = request.get("overlay")
    _require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"}
             and SHA256.fullmatch(descriptor.get("sha256", "")) is not None,
             "truth overlay descriptor differs")
    path = Path(descriptor["path"])
    _require(path.is_file(), "truth overlay is absent")
    required = ("person_id", "role", "component_id", "truth_state", "truth_observed")
    archive_bytes = _npz_uncompressed_bytes_before_decode(
        path, required, max_decode_bytes)
    with np.load(path, allow_pickle=False) as archive:
        _require(set(required) <= set(archive.files), "truth overlay metadata is incomplete")
        overlay_ids = _strings(archive["person_id"])
        _require(len(set(overlay_ids)) == len(overlay_ids),
                 "truth overlay person identifiers are not unique")
        lookup = {person_id: index for index, person_id in enumerate(overlay_ids)}
        regional_ids = _strings(regional["person_id"])
        _require(all(person_id in lookup for person_id in regional_ids),
                 "regional person is absent from truth overlay")
        people = np.asarray([lookup[person_id] for person_id in regional_ids])
        for name in ("role", "component_id"):
            _require(np.array_equal(np.asarray(regional[name]).astype(str),
                                    archive[name][people].astype(str)),
                     f"regional and truth-overlay {name} differ")
        indices = np.asarray(regional["query_global_index"])
        _require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
                 and bool((indices >= 0).all()), "query_global_index differs")
        truth_state = archive["truth_state"][people[:, None], indices[None, :]]
        truth_observed = archive["truth_observed"][people[:, None], indices[None, :]]
    _verify_array(truth_state, expected, "truth_state")
    _verify_array(truth_observed, expected, "truth_observed")
    return truth_state, truth_observed, archive_bytes


def load_duration_case_inventory(contract_path: str | Path, contract_sha256: str,
                                 case_id: str) -> DurationCaseInventory:
    """Read one real FIT case without reconstructing genotype/model arrays."""
    contract_path = Path(contract_path).resolve()
    contract = _read_json(contract_path, contract_sha256)
    chromosome_text = str(contract.get("lai_chromosome", ""))
    _require(chromosome_text.isdigit() and 1 <= int(chromosome_text) <= 22,
             "contract must declare one autosomal chromosome")
    chromosome = int(chromosome_text)
    _require(contract.get("objective") == "lai"
             and contract.get("information_condition") == "NONE"
             and contract.get("scope") == "DEVELOPMENT",
             "duration inventory admits the authenticated historical NONE LAI development contract only")
    metadata_decode_limit = contract.get("limits", {}).get("max_batch_bytes")
    _require(type(metadata_decode_limit) is int and metadata_decode_limit > 0,
             "contract lacks its metadata decode byte limit")
    scope_spec = contract.get("lai_development_scope")
    _require(isinstance(scope_spec, dict) and set(scope_spec) == {"path", "sha256"},
             "contract lacks an authenticated LAI development declaration")
    development_scope = _read_json(scope_spec["path"], scope_spec["sha256"])
    _require(development_scope.get("schema") == "r02_lai_factorized_development_scope_v1"
             and str(development_scope.get("chromosome")) == chromosome_text
             and case_id in development_scope.get("cases", {}).get("FIT", [])
             and development_scope.get("evaluate_opened") is False,
             "case/chromosome differs from the authenticated development declaration")
    batches = contract.get("batches")
    _require(isinstance(batches, dict) and "FIT" in batches,
             "contract has no FIT batches")
    # SELECT is neither opened nor inspected by this operation.
    entries = [entry for entry in batches["FIT"]
               if entry.get("lai_case_id") == case_id]
    _require(len(entries) > 0, "FIT case has no declared regional entries")

    sources: list[DurationRegionSource] = []
    person_axis = component_axis = None
    observed_total = None
    full_axis: list[tuple[int, float]] = []
    seen_regions: list[str] = []
    for entry in entries:
        _require(entry.get("role") == "FIT"
                 and entry.get("format") == "LAI_FACTORIZED_TRAINING_LINK_V1",
                 "duration inventory accepts authenticated FIT links only")
        request = _read_json(entry["path"], entry["sha256"])
        wrapper = _read_json(entry["receipt_path"], entry["receipt_sha256"])
        _require(request.get("schema") == "r02_lai_regional_truth_join_request_v2"
                 and request.get("case_id") == case_id
                 and request.get("condition") == "NONE"
                 and request.get("role") == "FIT"
                 and str(request.get("chromosome")) == chromosome_text
                 and request.get("training_authorized") is False
                 and request.get("selection_authorized") is False
                 and request.get("evaluate_opened") is False,
                 "factorized NONE request identity differs")
        required_wrapper = {"schema", "status", "request_sha256", "source_prepare_receipt",
                            "direction", "chromosome", "role", "condition", "case_id",
                            "region_id", "arrays", "reconstructed_arrays_sha256",
                            "evaluate_opened"}
        _require(required_wrapper <= set(wrapper)
                 and wrapper.get("schema") == "r02_lai_factorized_training_link_receipt_v1"
                 and wrapper.get("status") ==
                     "FACTORIZED_LINK_BOUND_TO_AUTHENTICATED_TRAINING_PROTOCOL"
                 and wrapper.get("request_sha256") == entry["sha256"]
                 and wrapper.get("case_id") == case_id
                 and wrapper.get("condition") == "NONE"
                 and wrapper.get("role") == "FIT"
                 and str(wrapper.get("chromosome")) == chromosome_text
                 and wrapper.get("evaluate_opened") is False,
                 "training wrapper identity differs")
        source_spec = wrapper["source_prepare_receipt"]
        source = _read_json(source_spec["path"], source_spec["sha256"])
        _require(source.get("schema") == "r02_lai_regional_truth_factorized_link_receipt_v1"
                 and source.get("status") ==
                     "REGIONAL_PREDICTOR_LINKED_TO_ONE_CONDITION_WITHOUT_ARRAY_DUPLICATION"
                 and source.get("factorized_link") is True
                 and source.get("truth_joined") is True
                 and source.get("request_sha256") == entry["sha256"]
                 and source.get("case_id") == case_id
                 and source.get("condition") == "NONE"
                 and source.get("role") == "FIT"
                 and str(source.get("chromosome")) == chromosome_text
                 and source.get("training_authorized") is False
                 and source.get("selection_authorized") is False
                 and source.get("evaluate_opened") is False
                 and wrapper.get("arrays") == source.get("arrays")
                 and source.get("reconstructed_arrays_sha256") ==
                     wrapper.get("reconstructed_arrays_sha256"),
                 "source PREPARE receipt identity differs")
        region_id = wrapper["region_id"]
        _require(region_id == source.get("region_id"), "region identity differs")
        seen_regions.append(region_id)
        expected = wrapper["reconstructed_arrays_sha256"]
        regional, regional_archive_bytes = _load_region_metadata(
            request, expected, metadata_decode_limit)
        truth_state, truth_observed, truth_archive_bytes = _load_truth_metadata(
            request, regional, expected, metadata_decode_limit)

        people = _strings(regional["person_id"])
        roles = _strings(regional["role"])
        components = _strings(regional["component_id"])
        _require(len(set(people)) == len(people) and set(roles) == {"FIT"},
                 "FIT person/role axis differs")
        declared_people = contract.get("persons", {})
        _require(all(declared_people.get(person) == {"component": component, "role": "FIT"}
                     for person, component in zip(people, components)),
                 "person component/role differs from the contract")
        if person_axis is None:
            person_axis, component_axis = people, components
            observed_total = np.zeros(len(people), dtype=np.int64)
        else:
            _require(people == person_axis and components == component_axis,
                     "person/component order changed between regions")

        qidx = np.asarray(regional["query_global_index"])
        core = np.asarray(regional["query_core"], dtype=bool)
        valid = np.asarray(regional["query_valid"], dtype=bool)
        qcm = np.asarray(regional["query_cm"])
        qchrom = np.asarray(regional["query_chrom"])
        _require(core.shape == valid.shape == qcm.shape == qchrom.shape == truth_state.shape
                 == truth_observed.shape and core.shape[0] == len(people)
                 and core.shape[1] == len(qidx), "regional query axes differ")
        _require(bool((core == core[0]).all()), "query_core differs between people")
        owned = core[0]
        _require(bool(owned.any()), "region has no owned query positions")
        _require(bool((qcm[:, owned] == qcm[0, owned]).all())
                 and bool((qchrom[:, owned] == chromosome).all()),
                 "owned coordinates differ between people or chromosome")
        owned_idx = tuple(int(value) for value in qidx[owned].tolist())
        owned_cm = tuple(float(value) for value in qcm[0, owned].tolist())
        _require(len(set(owned_idx)) == len(owned_idx)
                 and all(right > left for left, right in zip(owned_idx, owned_idx[1:]))
                 and all(right > left for left, right in zip(owned_cm, owned_cm[1:])),
                 "owned query axis is not strictly increasing")
        # Truth and emission availability affect masks/counts only.  They never
        # decide which owned positions form the chromosome axis.
        observed = core & valid & np.asarray(truth_observed, dtype=bool)
        _require(bool(np.isin(truth_state[observed], np.arange(6)).all()),
                 "observed truth state is outside the six unordered states")
        counts = observed.sum(axis=1, dtype=np.int64)
        observed_total += counts
        full_axis.extend(zip(owned_idx, owned_cm))
        sources.append(DurationRegionSource(
            region_id=region_id, entry=dict(entry),
            source_receipt_path=str(Path(source_spec["path"]).resolve()),
            source_receipt_sha256=source_spec["sha256"],
            core_global_indices=owned_idx, core_positions_cm=owned_cm,
            observed_core_labels_by_person=tuple(int(value) for value in counts),
            metadata_bytes_loaded=(sum(value.nbytes for value in regional.values())
                                   + truth_state.nbytes + truth_observed.nbytes),
            metadata_archive_uncompressed_bytes_read=(regional_archive_bytes
                                                      + truth_archive_bytes),
        ))

    _require(len(set(seen_regions)) == len(entries),
             "contract-declared FIT regional identities are duplicated")
    full_axis.sort()
    indices = tuple(value[0] for value in full_axis)
    positions = tuple(value[1] for value in full_axis)
    _require(len(indices) == len(set(indices))
             and all(right > left for left, right in zip(indices, indices[1:]))
             and all(right > left for left, right in zip(positions, positions[1:])),
             "contract-declared regional cores do not form one complete ordered axis")
    identity = {
        "contract_sha256": contract_sha256, "case_id": case_id, "chromosome": chromosome,
        "condition": "NONE", "role": "FIT", "persons_sha256": _digest(person_axis),
        "components_sha256": _digest(component_axis),
        "query_global_indices": indices, "query_positions_cm": positions,
        "observed_labels_by_person": observed_total.tolist(),
        "regions": [{"region_id": source.region_id,
                     "request_sha256": source.entry["sha256"],
                     "wrapper_sha256": source.entry["receipt_sha256"],
                     "source_receipt_sha256": source.source_receipt_sha256,
                     "metadata_archive_uncompressed_bytes_read":
                         source.metadata_archive_uncompressed_bytes_read}
                    for source in sources],
    }
    return DurationCaseInventory(
        contract_path=str(contract_path), contract_sha256=contract_sha256,
        case_id=case_id, chromosome=chromosome, condition="NONE", role="FIT",
        person_ids=person_axis, component_ids=component_axis,
        query_global_indices=indices, query_positions_cm=positions,
        observed_labels_by_person=tuple(int(value) for value in observed_total),
        regions=tuple(sorted(sources, key=lambda source: source.region_id)),
        metadata_bytes_loaded=sum(source.metadata_bytes_loaded for source in sources),
        metadata_archive_uncompressed_bytes_read=sum(
            source.metadata_archive_uncompressed_bytes_read for source in sources),
        inventory_sha256=_digest(identity),
    )


def build_declared_axis(inventory: DurationCaseInventory,
                        declaration: SequenceDeclaration) -> DeclaredSequenceAxis:
    _require(isinstance(inventory, DurationCaseInventory), "duration inventory is required")
    _require(isinstance(declaration, SequenceDeclaration),
             "caller must supply a sequence declaration and prior provenance")
    _require(declaration.unit_kind == "chromosome"
             and declaration.unit_id == inventory.case_id
             and declaration.chromosome == inventory.chromosome
             and declaration.expected_query_positions_cm == inventory.query_positions_cm,
             "sequence declaration differs from the authenticated complete axis")
    return DeclaredSequenceAxis(declaration, inventory.query_global_indices)


def _validate_loaded_geometry(data: Mapping[str, np.ndarray], source: DurationRegionSource,
                              people: tuple[str, ...], components: tuple[str, ...]) -> None:
    _require(_strings(data["person_id"]) == people
             and _strings(data["component_id"]) == components
             and set(_strings(data["role"])) == {"FIT"},
             "loaded FIT person/component/role axis differs from inventory")
    qidx = np.asarray(data["query_global_index"])
    core = np.asarray(data["query_core"], dtype=bool)
    valid = np.asarray(data["query_valid"], dtype=bool)
    _require(bool((core == core[0]).all())
             and tuple(int(value) for value in qidx[core[0]]) == source.core_global_indices,
             "loaded query core differs from inventory")
    _require(not bool((core & ~valid).any()),
             "owned query lacks a neural emission under missing_emission_policy=REJECT")


def iter_duration_person_chunks(
        model, inventory: DurationCaseInventory, declaration: SequenceDeclaration, *,
        contract_path: str | Path, contract_sha256: str,
        resolved_path: str | Path, resolved_sha256: str,
        people_per_chunk: int, max_output_numel: int,
        max_reconstruction_bytes: Mapping[str, int],
        max_cache_bytes: int) -> Iterator[DurationProducedChunk]:
    """Cache each authenticated region once; yield one person-block graph at a time.

    ``max_cache_bytes`` covers the retained ``array.nbytes`` sum.  Reconstruction
    workspace, model parameters, returned logits and internal activations remain
    separate resource quantities.
    """
    from r02_lai_person_microbatch import slice_people, validate_person_separation
    from r02_lai_training_campaign import load_training_link

    axis = build_declared_axis(inventory, declaration)
    _require(str(Path(contract_path).resolve()) == inventory.contract_path
             and contract_sha256 == inventory.contract_sha256,
             "producer contract identity differs from inventory")
    contract = _read_json(contract_path, contract_sha256)
    resolved = _read_json(resolved_path, resolved_sha256)
    validate_person_separation(model, "lai", people_per_chunk)
    _require(type(max_output_numel) is int and max_output_numel > 0,
             "max_output_numel must be a positive integer")
    per_region_bound = contract.get("limits", {}).get("max_batch_bytes")
    _require(type(max_cache_bytes) is int and max_cache_bytes > 0,
             "max_cache_bytes must be an explicit positive integer")
    _require(type(per_region_bound) is int and per_region_bound > 0,
             "contract lacks its authenticated per-region byte bound")
    estimated_cache_bytes = per_region_bound * len(inventory.regions)
    _require(estimated_cache_bytes <= max_cache_bytes,
             "declared regional cache cannot fit max_cache_bytes before loading")

    cached: list[tuple[DurationRegionSource, dict[str, np.ndarray], tuple[int, ...]]] = []
    cache_bytes = 0
    loader_calls = 0
    cached_observed_total = np.zeros(len(inventory.person_ids), dtype=np.int64)
    # No neural forward or autograd graph is constructed until every region has
    # passed the authenticated loader and the complete cache fits the caller's
    # explicit bound.
    for source in inventory.regions:
        data = load_training_link(
            source.entry, contract, resolved, "FIT",
            max_reconstruction_bytes=max_reconstruction_bytes)
        loader_calls += 1
        _require(isinstance(data, dict),
                 "authenticated training-link loader must return one array mapping")
        _validate_loaded_geometry(
            data, source, inventory.person_ids, inventory.component_ids)
        observed = (np.asarray(data["query_core"], dtype=bool)
                    & np.asarray(data["query_valid"], dtype=bool)
                    & np.asarray(data["truth_observed"], dtype=bool))
        region_counts = tuple(int(value) for value in
                              observed.sum(axis=1, dtype=np.int64))
        _require(region_counts == source.observed_core_labels_by_person,
                 "cached observed-label counts differ from metadata inventory")
        cached_observed_total += np.asarray(region_counts, dtype=np.int64)
        region_bytes = sum(np.asarray(value).nbytes for value in data.values())
        cache_bytes += region_bytes
        _require(cache_bytes <= max_cache_bytes,
                 "authenticated regional arrays exceed max_cache_bytes")
        cached.append((source, data, region_counts))
    _require(loader_calls == len(inventory.regions),
             "each declared region must be loaded exactly once")
    _require(tuple(int(value) for value in cached_observed_total)
             == inventory.observed_labels_by_person,
             "cached global observed-label counts differ from metadata inventory")
    global_denominator = int(cached_observed_total.sum())
    _require(global_denominator > 0, "global observed-label denominator is zero")

    people_count = len(inventory.person_ids)
    for start in range(0, people_count, people_per_chunk):
        end = min(start + people_per_chunk, people_count)
        wanted_people = inventory.person_ids[start:end]
        wanted_components = inventory.component_ids[start:end]
        fragments: list[DurationFragment] = []
        for source, data, _region_counts in cached:
            sliced = slice_people(data, start, end)
            # slice_people returns NumPy views for person-axis arrays.  Copy the
            # admitted block without constructing an autograd graph over every
            # cached person.  Shared site axes remain complete.
            micro = {name: np.array(value, copy=True) for name, value in sliced.items()}
            del sliced
            _validate_loaded_geometry(micro, source, wanted_people, wanted_components)
            logits = model.duration_fragment_logits(
                micro, "NONE", max_output_numel=max_output_numel)
            _require(isinstance(logits, torch.Tensor) and logits.ndim == 3
                     and logits.shape[:2] == np.asarray(micro["query_valid"]).shape
                     and logits.shape[-1] == 6,
                     "duration_fragment_logits must return [people,queries,6]")
            core = torch.as_tensor(np.asarray(micro["query_core"], dtype=bool),
                                   device=logits.device)
            valid = torch.as_tensor(np.asarray(micro["query_valid"], dtype=bool),
                                    device=logits.device)
            observed = torch.as_tensor(np.asarray(micro["truth_observed"], dtype=bool),
                                       device=logits.device)
            fragments.append(DurationFragment(
                neural_logits=logits,
                query_global_index=torch.as_tensor(np.asarray(micro["query_global_index"]),
                                                   device=logits.device),
                query_core=core, emission_valid=valid,
                query_cm=torch.as_tensor(np.asarray(micro["query_cm"]), device=logits.device),
                query_chromosome=torch.as_tensor(np.asarray(micro["query_chrom"]),
                                                 device=logits.device),
                labels=torch.as_tensor(np.asarray(micro["truth_state"]), dtype=torch.long,
                                       device=logits.device),
                label_mask=core & valid & observed,
                roles=_strings(micro["role"]), person_ids=wanted_people,
                component_ids=wanted_components,
            ))
            del micro
        chunk_count = sum(region_counts[index]
                          for _source, _data, region_counts in cached
                          for index in range(start, end))
        yield DurationProducedChunk(
            declared_axis=axis, fragments=tuple(fragments), person_ids=wanted_people,
            component_ids=wanted_components, global_label_denominator=global_denominator,
            chunk_observed_labels=chunk_count, inventory_sha256=inventory.inventory_sha256,
            estimated_cache_bytes=estimated_cache_bytes,
            cache_bytes=cache_bytes, loader_calls=loader_calls,
        )


def globally_normalized_duration_loss(row_label_loss_sums: torch.Tensor,
                                      global_label_denominator: int) -> torch.Tensor:
    """Normalize unaveraged per-person sums by the global observed-label count."""
    _require(isinstance(row_label_loss_sums, torch.Tensor)
             and row_label_loss_sums.ndim == 1
             and bool(torch.isfinite(row_label_loss_sums).all()),
             "duration loss input must contain one finite sum per person")
    _require(type(global_label_denominator) is int and global_label_denominator > 0,
             "global observed-label denominator must be a positive integer")
    return row_label_loss_sums.sum() / global_label_denominator


def inventory_receipt(inventory: DurationCaseInventory, elapsed_seconds: float) -> dict:
    return {
        "schema": INVENTORY_SCHEMA, "status": "METADATA_AUTHENTICATED_NOT_TRAINED",
        "contract_path": inventory.contract_path,
        "contract_sha256": inventory.contract_sha256,
        "case_id": inventory.case_id, "chromosome": inventory.chromosome,
        "condition": inventory.condition, "role": inventory.role,
        "persons": len(inventory.person_ids), "regions": len(inventory.regions),
        "owned_query_positions": len(inventory.query_global_indices),
        "observed_labels": sum(inventory.observed_labels_by_person),
        "metadata_logical_bytes_loaded": inventory.metadata_bytes_loaded,
        "metadata_archive_uncompressed_bytes_read":
            inventory.metadata_archive_uncompressed_bytes_read,
        "inventory_sha256": inventory.inventory_sha256,
        "elapsed_seconds": elapsed_seconds,
        "opened_roles": ["FIT"], "evaluate_opened": False,
        "payload_scope": "IDENTITY_QUERY_GEOMETRY_AND_TRUTH_MASK_ONLY",
        "training_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    receipts = []
    for case_id in args.case_id:
        case_started = time.monotonic()
        inventory = load_duration_case_inventory(args.contract, args.contract_sha256, case_id)
        receipts.append(inventory_receipt(inventory, time.monotonic() - case_started))
    output = Path(args.output)
    _require(not output.exists() and output.parent.is_dir(), "fresh inventory output path required")
    document = {"schema": INVENTORY_SCHEMA, "cases": receipts,
                "total_elapsed_seconds": time.monotonic() - started}
    output.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
