#!/usr/bin/env python3
"""Bind authenticated LAI neural potentials to complete duration sequences.

This module keeps the historical LAI network as the emission function but uses
its raw ``state_head`` outputs.  FLARE probabilities enter the network once as
features; the historical probability mixture and ``gate_head`` are not used as
independent genetic likelihoods.  The caller must freeze that unused head
before constructing the optimizer.

Two explicit input strategies are supported.  ``BOUNDED_ONE_CASE_MEMORY_CACHE``
loads and authenticates every region for one case once per process, then slices
person blocks from that bounded cache.  ``STREAM_ONE_REGION_PER_PERSON_CHUNK``
keeps one reconstructed region at a time but repeats all regional reads for
each person block.  Neither strategy splits a person's chromosome sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Lock
import time
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import Tensor, nn

from r02_lai_duration_input import (
    DurationCaseInventory, DurationRegionSource, _read_json, _strings,
    _validate_loaded_geometry, build_declared_axis,
)
from r02_lai_duration_model import (
    DurationFragment, MISSING_EMISSION_REJECT, SequenceDeclaration,
    assemble_duration_sequences,
)
from r02_lai_duration_training import (
    DurationParameterSpecification, DurationParameters, DurationTrainingChunk,
    DurationTrainingState, DurationUpdatePlan, ParameterProvenance,
)
from r02_lai_duration_fit import (
    fit_chunk_identity, validate_fit_batch, validate_fit_plan, validate_fit_rows,
)


SHA256 = re.compile(r"^[0-9a-f]{64}$")
RAW_EMISSION_SEMANTICS = (
    "RAW_STATE_HEAD_SIX_STATE_POTENTIALS_FLARE_INPUT_ONCE_NO_LEGACY_GATE_MIXTURE"
)
BOUNDED_ONE_CASE_MEMORY_CACHE = "BOUNDED_ONE_CASE_MEMORY_CACHE"
STREAM_ONE_REGION_PER_PERSON_CHUNK = "STREAM_ONE_REGION_PER_PERSON_CHUNK"
CACHE_STRATEGIES = frozenset((
    BOUNDED_ONE_CASE_MEMORY_CACHE, STREAM_ONE_REGION_PER_PERSON_CHUNK,
))
CHUNK_MEANING = (
    "ONE_COMPLETE_PERSON_BLOCK_FOR_ONE_AUTHENTICATED_CASE_AND_CHROMOSOME;"
    "TECHNICAL_REGIONAL_CORES_ARE_JOINED_BEFORE_THE_DURATION_LOSS"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authenticate_descriptor(value: object, label: str) -> tuple[Path, str]:
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"}
             and isinstance(value.get("path"), str)
             and SHA256.fullmatch(value.get("sha256", "")) is not None,
             f"{label} descriptor differs")
    path = Path(value["path"]).resolve()
    _require(path.is_file() and _file_sha256(path) == value["sha256"],
             f"{label} source SHA256 differs")
    return path, value["sha256"]


def _d1_q(log_rates: Sequence[Sequence[float]],
           switch_logits: Sequence[Sequence[float]]) -> list[list[float]]:
    result = [[0.0] * 3 for _ in range(3)]
    for ancestry in range(3):
        destinations = [j for j in range(3) if j != ancestry]
        maximum = max(float(switch_logits[ancestry][j]) for j in destinations)
        weights = [math.exp(float(switch_logits[ancestry][j]) - maximum)
                   for j in destinations]
        total = math.fsum(weights)
        exit_rate = math.exp(float(log_rates[ancestry][0]))
        for destination, weight in zip(destinations, weights):
            result[ancestry][destination] = exit_rate * weight / total
    return result


def build_fixed_d1_parameters(path: str | Path, expected_sha256: str, *,
                              floating_dtype: torch.dtype) -> DurationParameters:
    """Authenticate FP's D=1 values and construct fixed duration parameters.

    ``floating_dtype`` is mandatory because the duration tensors must match the
    neural network output dtype.  This constructor does not make any parameter
    learnable and does not choose optimizer values.
    """
    _require(floating_dtype in (torch.float32, torch.float64),
             "D1 floating dtype must be explicitly float32 or float64")
    _require(isinstance(expected_sha256, str)
             and SHA256.fullmatch(expected_sha256) is not None,
             "D1 source requires a lowercase SHA256")
    source = Path(path).resolve()
    _require(source.is_file() and _file_sha256(source) == expected_sha256,
             "D1 initial-values SHA256 differs")
    value = json.loads(source.read_text())
    required = {
        "status", "duration_stages", "ancestry_order", "log_stage_rates",
        "switch_logits", "joint_start_prior", "source_calibration",
        "source_prior", "offdiagonal_q", "limitations",
    }
    _require(isinstance(value, dict) and set(value) == required
             and value["status"] == "PREPARED_INITIAL_VALUES_NOT_TRAINING_ADMISSION"
             and value["duration_stages"] == 1
             and value["ancestry_order"] == ["AFR", "EUR", "NAM"],
             "D1 declaration status, stage count, or ancestry order differs")
    calibration_path, calibration_sha = _authenticate_descriptor(
        value["source_calibration"], "D1 calibration")
    prior_path, prior_sha = _authenticate_descriptor(value["source_prior"], "D1 prior")
    log_rates = value["log_stage_rates"]
    switch_logits = value["switch_logits"]
    prior = value["joint_start_prior"]
    declared_q = value["offdiagonal_q"]
    _require(len(log_rates) == 3 and all(isinstance(row, list) and len(row) == 1
                                        for row in log_rates)
             and len(switch_logits) == 3 and all(isinstance(row, list) and len(row) == 3
                                                 for row in switch_logits)
             and len(prior) == 3 and all(isinstance(row, list) and len(row) == 3
                                         for row in prior)
             and len(declared_q) == 3 and all(isinstance(row, list) and len(row) == 3
                                              for row in declared_q),
             "D1 numerical array dimensions differ")
    flat = [float(x) for table in (log_rates, switch_logits, prior, declared_q)
            for row in table for x in row]
    _require(all(math.isfinite(x) for x in flat), "D1 values must be finite")
    reconstructed_q = _d1_q(log_rates, switch_logits)
    for i in range(3):
        for j in range(3):
            _require(math.isclose(reconstructed_q[i][j], float(declared_q[i][j]),
                                  rel_tol=2e-12, abs_tol=2e-12),
                     "D1 log rates/switch logits do not reproduce offdiagonal_q")
    _require(math.isclose(math.fsum(float(x) for row in prior for x in row), 1.0,
                          rel_tol=0, abs_tol=2e-15),
             "D1 joint start prior must sum to one")
    _require(all(math.isclose(float(prior[i][j]), float(prior[j][i]),
                              rel_tol=0, abs_tol=2e-15)
                 for i in range(3) for j in range(3)),
             "D1 joint start prior must be symmetric")
    _require(isinstance(value["limitations"], list) and value["limitations"]
             and all(isinstance(item, str) and item for item in value["limitations"]),
             "D1 limitations must remain explicit")

    common = dict(source_sha256=expected_sha256, learning_status="FIXED",
                  permitted_claim="FIXED_INPUT")
    stages = ParameterProvenance(
        declaration_id="FP_D1_DURATION_STAGES_FIXED",
        initial_value_source=str(source),
        interpretation_limit="D=1 is a fixed candidate/control, not a selected biological optimum",
        **common)
    rates = ParameterProvenance(
        declaration_id="FP_D1_EFFECTIVE_FIT_RATES_FIXED",
        initial_value_source=f"{source};calibration={calibration_path};sha256={calibration_sha}",
        interpretation_limit="effective FIT initialization only; not demographic rates or dates",
        **common)
    switches = ParameterProvenance(
        declaration_id="FP_D1_ANCESTRY_SWITCHES_FIXED",
        initial_value_source=f"{source};calibration={calibration_path};sha256={calibration_sha}",
        interpretation_limit="effective FIT initialization only; shared donors limit inference",
        **common)
    start = ParameterProvenance(
        declaration_id="FO_JOINT_START_PRIOR_FIXED",
        initial_value_source=f"{source};prior={prior_path};sha256={prior_sha}",
        interpretation_limit="fixed unordered diploid start prior; not HWE or phased haplotypes",
        **common)
    specification = DurationParameterSpecification(
        duration_stages=1,
        log_stage_rates=torch.tensor(log_rates, dtype=floating_dtype),
        switch_logits=torch.tensor(switch_logits, dtype=floating_dtype),
        joint_start_prior=torch.tensor(prior, dtype=floating_dtype),
        stages_provenance=stages, rates_provenance=rates,
        switches_provenance=switches, prior_provenance=start)
    return DurationParameters(specification)


def freeze_legacy_probability_mixture_head(model: nn.Module) -> dict[str, object]:
    """Freeze the historical gate that is absent from raw duration emissions."""
    encoder = getattr(model, "encoder", None)
    gate = getattr(encoder, "gate_head", None)
    state = getattr(encoder, "state_head", None)
    _require(isinstance(gate, nn.Module) and isinstance(state, nn.Module),
             "existing LAI model must expose gate_head and state_head")
    gate_names = []
    for name, parameter in gate.named_parameters():
        parameter.requires_grad_(False)
        gate_names.append("encoder.gate_head." + name)
    _require(gate_names and any(parameter.requires_grad for parameter in state.parameters()),
             "raw duration emission head has no trainable state_head parameters")
    return {"emission_semantics": RAW_EMISSION_SEMANTICS,
            "frozen_parameter_names": sorted(gate_names),
            "reason": "gate_head belongs to historical post-softmax FLARE mixture and is unused"}


@dataclass(frozen=True)
class DurationNeuralCase:
    inventory: DurationCaseInventory
    declaration: SequenceDeclaration
    contract_path: str
    contract_sha256: str
    resolved_path: str
    resolved_sha256: str
    people_per_chunk: int
    max_output_numel: int
    max_retained_logit_numel: int
    max_reconstruction_bytes: Mapping[str, int]
    cache_strategy: str
    max_case_cache_bytes: int
    emission_semantics: str
    cache_directory: str | None = None
    cache_load_workers: int = 1

    def __post_init__(self) -> None:
        _require(isinstance(self.inventory, DurationCaseInventory)
                 and isinstance(self.declaration, SequenceDeclaration),
                 "duration inventory and sequence declaration are required")
        _require(type(self.people_per_chunk) is int and self.people_per_chunk > 0,
                 "people_per_chunk must be explicit and positive")
        _require(type(self.max_output_numel) is int and self.max_output_numel > 0
                 and type(self.max_retained_logit_numel) is int
                 and self.max_retained_logit_numel > 0,
                 "neural output and retained-logit bounds must be positive")
        _require(self.cache_strategy in CACHE_STRATEGIES,
                 "cache strategy must be selected explicitly")
        _require(type(self.max_case_cache_bytes) is int and self.max_case_cache_bytes > 0,
                 "case cache byte bound must be explicit and positive")
        _require(self.emission_semantics == RAW_EMISSION_SEMANTICS,
                 "posterior probabilities are not admitted as duration emissions")
        _require(isinstance(self.max_reconstruction_bytes, Mapping)
                 and all(isinstance(k, str) and type(v) is int and v > 0
                         for k, v in self.max_reconstruction_bytes.items()),
                 "authenticated reconstruction byte limits are required")
        _require(SHA256.fullmatch(self.contract_sha256 or "") is not None
                 and SHA256.fullmatch(self.resolved_sha256 or "") is not None,
                 "contract and resolved SHA256 values are required")
        _require(self.cache_directory is None
                 or (isinstance(self.cache_directory, str)
                     and bool(self.cache_directory.strip())),
                 "cache_directory must be absent or a nonempty path")
        _require(type(self.cache_load_workers) is int and self.cache_load_workers in (1, 2, 4)
                 and (self.cache_load_workers == 1
                      or self.cache_strategy == BOUNDED_ONE_CASE_MEMORY_CACHE),
                 "one, two or four bounded regional cache workers are required")


def _zip_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        _require(infos and len({info.filename for info in infos}) == len(infos),
                 "NPZ members must be present and unique")
        return sum(info.file_size for info in infos)


def estimate_case_input_bytes_from_headers(case: DurationNeuralCase) -> dict[str, object]:
    """Read authenticated request JSON and ZIP headers, never NumPy array bodies."""
    by_path: dict[str, dict[str, object]] = {}
    for source in case.inventory.regions:
        request = _read_json(source.entry["path"], source.entry["sha256"])
        for field in ("regional_batch", "overlay", "event_burden_prefix"):
            descriptor = request.get(field)
            _require(isinstance(descriptor, dict) and set(descriptor) == {"path", "sha256"}
                     and SHA256.fullmatch(descriptor.get("sha256", "")) is not None,
                     f"{field} descriptor differs")
            path = Path(descriptor["path"]).resolve()
            _require(path.is_file(), f"{field} file is absent")
            key = str(path)
            if key not in by_path:
                by_path[key] = {"kind": field, "path": key,
                                "declared_sha256": descriptor["sha256"],
                                "stored_bytes": path.stat().st_size,
                                "npz_uncompressed_bytes": _zip_uncompressed_bytes(path)}
            else:
                _require(by_path[key]["declared_sha256"] == descriptor["sha256"],
                         "one input path has conflicting declared hashes")
    per_region_bound = case.max_reconstruction_bytes.get(
        "max_reconstructed_join_bytes")
    _require(type(per_region_bound) is int and per_region_bound > 0,
             "audited max_reconstructed_join_bytes is required")
    return {
        "status": "NPZ_ZIP_HEADERS_ONLY_NO_ARRAY_BODY_READ",
        "case_id": case.inventory.case_id,
        "unique_files": tuple(by_path.values()),
        "stored_bytes": sum(int(row["stored_bytes"]) for row in by_path.values()),
        "unique_input_uncompressed_bytes": sum(
            int(row["npz_uncompressed_bytes"]) for row in by_path.values()),
        "audited_reconstructed_case_bound_bytes":
            per_region_bound * len(case.inventory.regions),
        "cache_strategy": case.cache_strategy,
        "max_case_cache_bytes": case.max_case_cache_bytes,
        "limitation": (
            "ZIP sizes describe unique source arrays; the live cache is checked again "
            "from reconstructed array.nbytes before use"),
    }


@dataclass
class _CaseRuntime:
    specification: DurationNeuralCase
    contract: dict
    resolved: dict
    axis: object
    cache: list[tuple[DurationRegionSource, dict[str, np.ndarray]]] | None = None
    cache_bytes: int = 0
    loader_calls: int = 0
    disk_cache_hits: int = 0
    disk_cache_writes: int = 0
    completed_region_loads: int = 0
    progress_lock: object = field(default_factory=Lock, repr=False)


class DurationNeuralChunkProvider:
    """Produce complete chromosome chunks at the exact durable cursor index."""

    def __init__(self, cases: Sequence[DurationNeuralCase], fit_rows: dict):
        _require(isinstance(cases, Sequence) and len(cases) > 0,
                 "at least one duration neural case is required")
        validate_fit_rows(fit_rows)
        self.fit_rows = fit_rows
        self._cases: list[_CaseRuntime] = []
        self._chunk_lookup: list[tuple[int, int, int]] = []
        rows = []
        by_id = {case.inventory.case_id: (index, case)
                 for index, case in enumerate(cases)}
        _require(len(by_id) == len(cases) and set(by_id) == set(fit_rows["cases"]),
                 "provider cases differ from the authenticated FIT row declaration")
        for case_index, specification in enumerate(cases):
            _require(isinstance(specification, DurationNeuralCase),
                     "duration neural cases must use DurationNeuralCase")
            inventory = specification.inventory
            _require(str(Path(specification.contract_path).resolve()) == inventory.contract_path
                     and specification.contract_sha256 == inventory.contract_sha256,
                     "provider contract differs from inventory")
            contract = _read_json(specification.contract_path,
                                  specification.contract_sha256)
            resolved = _read_json(specification.resolved_path,
                                  specification.resolved_sha256)
            axis = build_declared_axis(inventory, specification.declaration)
            runtime = _CaseRuntime(specification, contract, resolved, axis)
            self._cases.append(runtime)
            fit_case = fit_rows["cases"][inventory.case_id]
            _require(fit_case["inventory_sha256"] == inventory.inventory_sha256
                     and fit_case["contract"] == {
                         "path": inventory.contract_path,
                         "sha256": inventory.contract_sha256}
                     and tuple(fit_case["person_ids"]) == inventory.person_ids
                     and tuple(fit_case["component_ids"]) == inventory.component_ids
                     and tuple(fit_case["query_positions_cm"])
                        == inventory.query_positions_cm
                     and tuple(fit_case["query_global_indices"])
                        == inventory.query_global_indices
                     and tuple(fit_case["observed_labels_by_person"])
                        == inventory.observed_labels_by_person,
                     "provider inventory differs from authenticated FIT rows")
        for index, chunk in enumerate(fit_rows["chunks"]):
            case_index, specification = by_id[chunk["case_id"]]
            inventory = specification.inventory
            start, end = chunk["person_start"], chunk["person_end"]
            _require(end - start <= specification.people_per_chunk,
                     "FIT chunk exceeds the provider people_per_chunk bound")
            observed = sum(inventory.observed_labels_by_person[start:end])
            payload = {
                "case_id": inventory.case_id,
                "inventory_sha256": inventory.inventory_sha256,
                "contract": {"path": inventory.contract_path,
                             "sha256": inventory.contract_sha256},
                "resolved": {"path": str(Path(specification.resolved_path).resolve()),
                             "sha256": specification.resolved_sha256},
                "person_ids": inventory.person_ids[start:end],
                "component_ids": inventory.component_ids[start:end],
                "query_positions_cm": inventory.query_positions_cm,
                "chromosome": inventory.chromosome,
                "observed_labels": observed,
                "meaning": CHUNK_MEANING,
                "emission_semantics": RAW_EMISSION_SEMANTICS,
                "cache_strategy": specification.cache_strategy,
                "identity_sha256": fit_chunk_identity(fit_rows, index),
            }
            rows.append(payload)
            self._chunk_lookup.append((case_index, start, end))
        self.chunk_rows = tuple(rows)
        self.plan = DurationUpdatePlan(
            tuple(row["identity_sha256"] for row in rows),
            global_observed_labels=sum(
                sum(case["observed_labels_by_person"])
                for case in fit_rows["cases"].values()),
            global_rows=sum(len(case["person_ids"])
                            for case in fit_rows["cases"].values()))
        validate_fit_plan(fit_rows, self.plan)
        self._active_cache_case: int | None = None

    @staticmethod
    def _validate_model(model: nn.Module) -> None:
        encoder = getattr(model, "encoder", None)
        gate = getattr(encoder, "gate_head", None)
        state = getattr(encoder, "state_head", None)
        _require(isinstance(model, nn.Module) and getattr(model, "objective", None) == "lai"
                 and callable(getattr(model, "duration_fragment_logits", None))
                 and isinstance(gate, nn.Module) and isinstance(state, nn.Module),
                 "provider requires the existing LAI TaskModel raw-potential API")
        _require(all(not parameter.requires_grad for parameter in gate.parameters()),
                 "legacy gate_head must be frozen before optimizer construction")
        _require(any(parameter.requires_grad for parameter in state.parameters()),
                 "state_head must remain trainable for duration emissions")

    def _load_region(self, runtime: _CaseRuntime,
                     source: DurationRegionSource) -> dict[str, np.ndarray]:
        from r02_lai_training_campaign import load_training_link
        spec = runtime.specification
        started = time.monotonic()
        source_loaded = cache_hit = cache_written = 0
        disk_cache = None
        data = None
        if spec.cache_directory is not None:
            from r02_lai_duration_cache import (
                DurationRegionCache, file_sha256, region_cache_identity,
            )
            wrapper = _read_json(source.entry["receipt_path"],
                                 source.entry["receipt_sha256"])
            expected = wrapper.get("reconstructed_arrays_sha256")
            _require(isinstance(expected, dict) and expected,
                     "training wrapper lacks reconstructed-array SHA256 values")
            cache_module = Path(__file__).with_name("r02_lai_duration_cache.py")
            identity = region_cache_identity(
                region_id=source.region_id, entry=source.entry,
                contract_sha256=spec.contract_sha256,
                resolved_sha256=spec.resolved_sha256,
                emission_semantics=spec.emission_semantics,
                code_sha256={
                    "r02_lai_duration_neural.py": file_sha256(Path(__file__)),
                    "r02_lai_duration_cache.py": file_sha256(cache_module),
                })
            disk_cache = DurationRegionCache(
                Path(spec.cache_directory) / spec.inventory.inventory_sha256,
                identity, expected)
            data = disk_cache.load()
            if data is not None:
                cache_hit = 1
                result = "CACHE_VERIFIED_AND_REUSED"
            else:
                data = None
        else:
            result = "SOURCE_AUTHENTICATED_AND_CACHED"
        if data is None:
            data = load_training_link(
                source.entry, runtime.contract, runtime.resolved, "FIT",
                max_reconstruction_bytes=spec.max_reconstruction_bytes)
            source_loaded = 1
            _require(isinstance(data, dict),
                     "authenticated training-link loader must return one array mapping")
            if disk_cache is not None:
                data = disk_cache.save(data)
                cache_written = 1
                result = "SOURCE_AUTHENTICATED_AND_CACHED"
        _validate_loaded_geometry(data, source, spec.inventory.person_ids,
                                  spec.inventory.component_ids)
        observed = (np.asarray(data["query_core"], dtype=bool)
                    & np.asarray(data["query_valid"], dtype=bool)
                    & np.asarray(data["truth_observed"], dtype=bool))
        counts = tuple(int(value) for value in observed.sum(axis=1, dtype=np.int64))
        _require(counts == source.observed_core_labels_by_person,
                 "loaded observed-label counts differ from inventory")
        with runtime.progress_lock:
            runtime.loader_calls += source_loaded
            runtime.disk_cache_hits += cache_hit
            runtime.disk_cache_writes += cache_written
            runtime.completed_region_loads += 1
            if disk_cache is not None:
                from r02_lai_duration_cache import write_region_progress
                write_region_progress(
                    Path(spec.cache_directory) / spec.inventory.inventory_sha256 /
                        "PROGRESS.json",
                    case_id=spec.inventory.case_id, region_id=source.region_id,
                    cache_identity_sha256=disk_cache.identity_sha256, result=result,
                    completed_regions=runtime.completed_region_loads,
                    source_loader_calls=runtime.loader_calls,
                    disk_cache_hits=runtime.disk_cache_hits,
                    disk_cache_writes=runtime.disk_cache_writes,
                    elapsed_seconds=time.monotonic() - started)
        return data

    def _ordered_region_loads(self, runtime):
        """Bound in-flight reads by declared workers; preserve order and join."""
        sources = runtime.specification.inventory.regions
        workers = runtime.specification.cache_load_workers
        if workers == 1:
            for source in sources:
                yield source, self._load_region(runtime, source)
            return
        import r02_lai_neural_input
        _require(r02_lai_neural_input.STAGED_INPUTS is None,
                 "parallel source reads require immutable unstaged input configuration")
        pool = ThreadPoolExecutor(max_workers=workers)
        pending = [pool.submit(self._load_region, runtime, s) for s in sources[:workers]]
        try:
            for index, source in enumerate(sources):
                data = pending.pop(0).result()
                yield source, data
                if index + workers < len(sources):
                    pending.append(pool.submit(self._load_region, runtime, sources[index + workers]))
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    def _ensure_cache(self, runtime: _CaseRuntime) -> None:
        if runtime.cache is not None:
            return
        spec = runtime.specification
        bound = spec.max_reconstruction_bytes.get("max_reconstructed_join_bytes")
        _require(type(bound) is int and bound > 0
                 and bound * len(spec.inventory.regions) <= spec.max_case_cache_bytes,
                 "audited reconstructed case bound exceeds max_case_cache_bytes")
        cache = []
        total = 0
        with closing(self._ordered_region_loads(runtime)) as loaded:
            for source, data in loaded:
                total += sum(np.asarray(value).nbytes for value in data.values())
                _require(total <= spec.max_case_cache_bytes,
                         "live reconstructed arrays exceed max_case_cache_bytes")
                cache.append((source, data))
        runtime.cache = cache
        runtime.cache_bytes = total

    def _activate_case_cache(self, case_index: int) -> None:
        if self._active_cache_case == case_index:
            return
        if self._active_cache_case is not None:
            previous = self._cases[self._active_cache_case]
            previous.cache = None
            previous.cache_bytes = 0
        self._active_cache_case = case_index
        self._ensure_cache(self._cases[case_index])

    def _region_data(self, runtime: _CaseRuntime, case_index: int):
        spec = runtime.specification
        if spec.cache_strategy == BOUNDED_ONE_CASE_MEMORY_CACHE:
            self._activate_case_cache(case_index)
            assert runtime.cache is not None
            yield from runtime.cache
        else:
            for source in spec.inventory.regions:
                yield source, self._load_region(runtime, source)

    def __call__(self, state: DurationTrainingState,
                 index: int) -> DurationTrainingChunk:
        _require(isinstance(state, DurationTrainingState),
                 "DurationTrainingState is required")
        _require(type(index) is int and 0 <= index < len(self._chunk_lookup),
                 "duration neural chunk index is out of range")
        self._validate_model(state.emission_model)
        from r02_lai_person_microbatch import slice_people, validate_person_separation
        case_index, start, end = self._chunk_lookup[index]
        runtime = self._cases[case_index]
        spec = runtime.specification
        validate_person_separation(state.emission_model, "lai", spec.people_per_chunk)
        wanted_people = spec.inventory.person_ids[start:end]
        wanted_components = spec.inventory.component_ids[start:end]
        fragments = []
        for source, data in self._region_data(runtime, case_index):
            sliced = slice_people(data, start, end)
            micro = {name: np.array(value, copy=True) for name, value in sliced.items()}
            del sliced
            _validate_loaded_geometry(micro, source, wanted_people, wanted_components)
            logits = state.emission_model.duration_fragment_logits(
                micro, "NONE", max_output_numel=spec.max_output_numel)
            _require(isinstance(logits, Tensor) and logits.ndim == 3
                     and tuple(logits.shape[:2]) == tuple(np.asarray(
                         micro["query_valid"]).shape)
                     and logits.shape[-1] == 6,
                     "raw duration emission API must return [people,queries,6]")
            # The existing network is float32, while authenticated cM coordinates
            # and the exact duration recursion use float64.  Casting the raw
            # potentials preserves autograd into state_head without interpreting
            # them as normalized probabilities.
            logits = logits.to(dtype=state.duration.log_stage_rates.dtype)
            core = torch.as_tensor(np.asarray(micro["query_core"], dtype=bool),
                                   device=logits.device)
            valid = torch.as_tensor(np.asarray(micro["query_valid"], dtype=bool),
                                    device=logits.device)
            observed = torch.as_tensor(np.asarray(micro["truth_observed"], dtype=bool),
                                       device=logits.device)
            fragments.append(DurationFragment(
                neural_logits=logits,
                query_global_index=torch.as_tensor(
                    np.asarray(micro["query_global_index"]), device=logits.device),
                query_core=core, emission_valid=valid,
                query_cm=torch.as_tensor(np.asarray(micro["query_cm"]),
                                         device=logits.device),
                query_chromosome=torch.as_tensor(np.asarray(micro["query_chrom"]),
                                                 device=logits.device),
                labels=torch.as_tensor(np.asarray(micro["truth_state"]),
                                       dtype=torch.long, device=logits.device),
                label_mask=core & valid & observed,
                roles=_strings(micro["role"]), person_ids=wanted_people,
                component_ids=wanted_components))
            del micro
            if spec.cache_strategy == STREAM_ONE_REGION_PER_PERSON_CHUNK:
                del data
        batch = assemble_duration_sequences(
            tuple(fragments), runtime.axis,
            missing_emission_policy=MISSING_EMISSION_REJECT,
            rows_per_chunk=spec.people_per_chunk,
            max_retained_logit_numel=spec.max_retained_logit_numel)
        row = self.chunk_rows[index]
        _require(batch.person_ids == row["person_ids"]
                 and batch.component_ids == row["component_ids"]
                 and int(batch.label_mask.sum().item()) == row["observed_labels"],
                 "assembled neural chunk differs from its authenticated row")
        validate_fit_batch(self.fit_rows, index, batch)
        return DurationTrainingChunk(row["identity_sha256"], batch)

    def runtime_receipt(self) -> dict[str, object]:
        return {
            "fit_rows_sha256": _digest(self.fit_rows),
            "chunk_count": len(self.chunk_rows),
            "global_rows": self.plan.global_rows,
            "global_observed_labels": self.plan.global_observed_labels,
            "plan_sha256": _digest(self.plan.receipt()),
            "first_chunk_identity_sha256": self.plan.chunk_identity_sha256[0],
            "last_chunk_identity_sha256": self.plan.chunk_identity_sha256[-1],
            "cases": tuple({
                "case_id": runtime.specification.inventory.case_id,
                "cache_strategy": runtime.specification.cache_strategy,
                "live_cache_bytes": runtime.cache_bytes,
                "loader_calls": runtime.loader_calls,
                "disk_cache_hits": runtime.disk_cache_hits,
                "disk_cache_writes": runtime.disk_cache_writes,
                "completed_region_loads": runtime.completed_region_loads,
            } for runtime in self._cases),
            "warning": (
                "streaming reloads and authenticates every region for every person chunk; "
                "bounded case caching reloads once per process and still requires measured "
                "neural activation memory before human admission"),
        }

    def statistics(self) -> dict[str, object]:
        """Return exact provider work counters for a measurement receipt."""
        return self.runtime_receipt()
