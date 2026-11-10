#!/usr/bin/env python3
"""One recoverable Adam update per declared autonomous-donor LAI batch.

The scientific loss is fixed here: stable six-state negative log likelihood,
averaged over declared observed/core queries within each person and then over
the declared people in one logical batch.  Person and query blocks are memory
controls only.  Truth tensors are kept outside :class:`DonorContextBatch` and
are never passed to the model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from r02_context_checkpoint import (
    CheckpointStore, _canonical, _sha_bytes, _sha_file, identity_digest,
)
from r02_context_update_checkpoint import (
    PartialUpdateStore, capture_parameter_gradients, optimizer_start_digest,
    restore_parameter_gradients,
)
from r02_lai_donor_context import (
    AutonomousDonorContextLAI, DonorContextBatch, JOINT_ANCESTRY_STATES,
    STATE_NAMES,
)


TRAINING_SCHEMA = "r02_lai_donor_training_v1"
PARTIAL_SCHEMA = "r02_lai_donor_partial_update_v1"
WEIGHTED_TRAINING_SCHEMA = "r02_lai_donor_training_v2_occurrence_weighted"
WEIGHTED_PARTIAL_SCHEMA = "r02_lai_donor_partial_update_v2_occurrence_weighted"
FROZEN_READOUT_TRAINING_SCHEMA = "r02_lai_frozen_readout_training_v1_occurrence_weighted"
FROZEN_READOUT_PARTIAL_SCHEMA = "r02_lai_frozen_readout_partial_v1_occurrence_weighted"
FROZEN_READOUT_MODEL_KIND = "R02_LAI_FROZEN_READOUT_V1"
SYNTHETIC_SCOPE = "SYNTHETIC_SOFTWARE_FIXTURE"
FIT_SCOPE = "FIT_DEVELOPMENT_SHARED_DONORS_NO_INDEPENDENT_VALIDATION"
SHA256 = re.compile(r"[0-9a-f]{64}")

# A caller must provide this complete declaration.  Any different loss,
# weighting, support rule, or state order is a new scientific specification.
DONOR_OBJECTIVE = {
    "schema": "r02_lai_donor_objective_v1",
    "state_order": list(STATE_NAMES),
    "loss": "STABLE_LOG_SOFTMAX_SIX_STATE_NEGATIVE_LOG_LIKELIHOOD",
    "scientific_query_mask": "TRUTH_OBSERVED_AND_QUERY_CORE_AND_QUERY_VALID",
    "aggregation": "MEAN_SCORED_QUERIES_PER_PERSON_THEN_MEAN_DECLARED_PEOPLE",
    "technical_partition_rule": "GLOBAL_PERSON_AND_QUERY_DENOMINATORS_FIXED_BEFORE_MICROBATCHES",
    "unsupported_scored_query": "LEARNED_HEAD_NO_LOCAL_SUPPORT",
    "unsupported_scored_query_interpretation":
        "LEARNED_DISTRIBUTION_WITHOUT_LOCAL_LOCUS_EVIDENCE_NOT_EMPIRICAL_FREQUENCY",
    "query_weighting": "EQUAL_WITHIN_PERSON_NO_CM_WEIGHT",
}

# Explicit opt-in objective for the stratified FIT occurrence plan.  Coefficients
# already contain the natural state mass, anchor quota and block width.  Dividing
# again by people, unique queries, microbatches or their realized weight would
# change the reviewed target.
WEIGHTED_DONOR_OBJECTIVE = {
    "schema": "r02_lai_donor_objective_v2_occurrence_weighted",
    "state_order": list(STATE_NAMES),
    "loss": "STABLE_LOG_SOFTMAX_SIX_STATE_NEGATIVE_LOG_LIKELIHOOD",
    "scientific_query_mask": "TRUTH_OBSERVED_AND_QUERY_CORE_AND_QUERY_VALID",
    "aggregation": "SUM_OCCURRENCE_COEFFICIENT_TIMES_NLL_NO_RENORMALIZATION",
    "technical_partition_rule":
        "SUM_DUPLICATE_COEFFICIENTS_PER_UPDATE_PERSON_QUERY_BEFORE_MICROBATCHING",
    "coefficient_source": "ANCHOR_STATE_NOT_QUERY_STATE",
    "coefficient_sum_per_update": 1.0,
    "optimizer_update": "ONE_ADAM_STEP_AFTER_ALL_WEIGHTED_MICROBATCHES",
    "executed_context_geometry":
        "CONTIGUOUS_SEGMENTS_OVER_COMPLETE_FROZEN_PLAN_WITH_EXACT_START_STOP_IN_IDENTITY",
    "unsupported_scored_query": "LEARNED_HEAD_NO_LOCAL_SUPPORT",
    "unsupported_scored_query_interpretation":
        "LEARNED_DISTRIBUTION_WITHOUT_LOCAL_LOCUS_EVIDENCE_NOT_EMPIRICAL_FREQUENCY",
    "query_weighting": "NATURAL_EQUAL_PERSON_THEN_EQUAL_QUERY_IN_EXPECTATION",
}

FROZEN_READOUT_OBJECTIVE = {
    **WEIGHTED_DONOR_OBJECTIVE,
    "schema": "r02_lai_frozen_readout_objective_v1_occurrence_weighted",
    "model_input": "AUTHENTICATED_FROZEN_PRE_HEAD_VECTOR_OR_FROZEN_GQ_LOGITS",
    "upstream_parameters": "ABSENT_FROM_OPTIMIZER_AND_UNCHANGED",
    "cache_partition_rule":
        "ONE_EXACT_PHYSICAL_SEGMENT_PAYLOAD_REUSED_WITHOUT_CHANGING_OCCURRENCE_WEIGHTS",
    "scientific_scope": "CONDITIONAL_FIT_DEVELOPMENT_NO_SELECT_OR_EVALUATE",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _model_kind(model: torch.nn.Module) -> str:
    if isinstance(model, AutonomousDonorContextLAI):
        return "AUTONOMOUS_DONOR_CONTEXT_V1"
    if (isinstance(model, torch.nn.Module)
            and getattr(model, "training_model_kind", None) == FROZEN_READOUT_MODEL_KIND
            and isinstance(getattr(model, "configuration", None), Mapping)):
        return FROZEN_READOUT_MODEL_KIND
    raise ValueError("model must be AutonomousDonorContextLAI or the versioned frozen readout")


def hash_sources(source_paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    """Hash exact source files now; callers do not transcribe their digests."""
    _require(isinstance(source_paths, Mapping) and bool(source_paths),
             "at least one named source path is required")
    result: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted(source_paths.items()):
        _require(isinstance(name, str) and name.strip(), "source names are required")
        path = Path(raw_path)
        _require(path.is_file(), f"source file does not exist: {name}")
        result[name] = {"path": str(path.resolve()), "sha256": _sha_file(path)}
    return result


def backend_determinism_receipt() -> dict[str, object]:
    """Record the CPU numerical backend that must match on resumption."""
    return {
        "device_policy": "CPU_ONLY_BECAUSE_SHARED_CHECKPOINT_STORES_CAPTURE_CPU_RNG",
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "torch_build_sha256": hashlib.sha256(
            torch.__config__.show().encode()).hexdigest(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cross_cpu_gpu_bit_identity_claimed": False,
    }


@dataclass(frozen=True)
class AdamSpecification:
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    gradient_clip_norm: float | None
    amsgrad: bool
    maximize: bool
    foreach: bool
    capturable: bool
    differentiable: bool
    fused: bool

    def __post_init__(self) -> None:
        _require(math.isfinite(self.learning_rate) and self.learning_rate > 0,
                 "positive finite Adam learning rate required")
        _require(isinstance(self.betas, tuple) and len(self.betas) == 2 and
                 all(math.isfinite(value) and 0 <= value < 1 for value in self.betas),
                 "two explicit Adam beta values in [0,1) are required")
        _require(math.isfinite(self.epsilon) and self.epsilon > 0,
                 "positive finite Adam epsilon required")
        _require(math.isfinite(self.weight_decay) and self.weight_decay >= 0,
                 "nonnegative finite Adam weight decay required")
        _require(self.gradient_clip_norm is None or
                 (math.isfinite(self.gradient_clip_norm) and
                  self.gradient_clip_norm > 0),
                 "gradient clip norm must be positive or explicitly None")
        values = (self.amsgrad, self.maximize, self.foreach, self.capturable,
                  self.differentiable, self.fused)
        _require(all(type(value) is bool for value in values),
                 "every Adam implementation option must be explicit boolean")
        _require(not any((self.capturable, self.differentiable, self.fused)),
                 "the data-only CPU checkpoint supports conventional unfused Adam only")

    def receipt(self) -> dict[str, object]:
        return {
            "optimizer": "Adam", "learning_rate": self.learning_rate,
            "betas": list(self.betas), "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "amsgrad": self.amsgrad, "maximize": self.maximize,
            "foreach": self.foreach, "capturable": self.capturable,
            "differentiable": self.differentiable, "fused": self.fused,
        }


def build_adam(model: torch.nn.Module,
               specification: AdamSpecification) -> torch.optim.Adam:
    _model_kind(model)
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    _require(bool(parameters), "donor model has no learnable parameters")
    return torch.optim.Adam(
        parameters, lr=specification.learning_rate, betas=specification.betas,
        eps=specification.epsilon, weight_decay=specification.weight_decay,
        amsgrad=specification.amsgrad, maximize=specification.maximize,
        foreach=specification.foreach, capturable=specification.capturable,
        differentiable=specification.differentiable, fused=specification.fused)


def _validate_adam(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   specification: AdamSpecification) -> None:
    _require(type(optimizer) is torch.optim.Adam and len(optimizer.param_groups) == 1,
             "one Adam parameter group is required")
    group = optimizer.param_groups[0]
    expected = {
        "lr": specification.learning_rate, "betas": specification.betas,
        "eps": specification.epsilon, "weight_decay": specification.weight_decay,
        "amsgrad": specification.amsgrad, "maximize": specification.maximize,
        "foreach": specification.foreach, "capturable": specification.capturable,
        "differentiable": specification.differentiable,
        "fused": specification.fused,
    }
    _require(all(group.get(name) == value for name, value in expected.items()),
             "live Adam options differ from the declared specification")
    optimized = [parameter for parameter in group["params"]]
    learnable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    _require(len(optimized) == len(learnable) and
             {id(value) for value in optimized} == {id(value) for value in learnable},
             "Adam must optimize every and only learnable donor-model parameter")


@dataclass(frozen=True)
class MicrobatchPlan:
    identity_sha256: str
    person_ids: tuple[str, ...]
    scored_query_global_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _require(bool(SHA256.fullmatch(self.identity_sha256)),
                 "microbatch SHA256 identity required")
        _require(bool(self.person_ids) and len(set(self.person_ids)) == len(self.person_ids)
                 and len(self.person_ids) == len(self.scored_query_global_indices),
                 "microbatch requires unique person rows and their query indices")
        _require(all(isinstance(value, str) and value.strip() for value in self.person_ids)
                 and all(bool(indices) and len(set(indices)) == len(indices) and
                         all(type(index) is int and index >= 0 for index in indices)
                         for indices in self.scored_query_global_indices),
                 "microbatch scored-query indices must be nonempty unique integers")

    def receipt(self) -> dict[str, object]:
        return {"identity_sha256": self.identity_sha256,
                "person_ids": list(self.person_ids),
                "scored_query_global_indices":
                    [list(value) for value in self.scored_query_global_indices]}


@dataclass(frozen=True)
class LogicalBatchPlan:
    """Scientific batch fixed before any technical person/query block is read."""
    step_index: int
    logical_batch_id: str
    person_ids: tuple[str, ...]
    person_query_denominators: tuple[int, ...]
    microbatches: tuple[MicrobatchPlan, ...]

    def __post_init__(self) -> None:
        _require(type(self.step_index) is int and self.step_index >= 0,
                 "nonnegative logical-batch step required")
        _require(isinstance(self.logical_batch_id, str) and
                 self.logical_batch_id.strip(), "logical batch id required")
        _require(bool(self.person_ids) and len(set(self.person_ids)) == len(self.person_ids)
                 and all(isinstance(value, str) and value.strip()
                         for value in self.person_ids),
                 "ordered unique person ids required")
        _require(len(self.person_query_denominators) == len(self.person_ids) and
                 all(type(value) is int and value > 0
                     for value in self.person_query_denominators),
                 "positive scored-query denominator required for every person")
        _require(bool(self.microbatches) and
                 all(isinstance(value, MicrobatchPlan) for value in self.microbatches),
                 "ordered person-by-query microbatch plan required")
        declared = dict(zip(self.person_ids, self.person_query_denominators))
        observed = {person: set() for person in self.person_ids}
        for microbatch in self.microbatches:
            for person, indices in zip(microbatch.person_ids,
                                       microbatch.scored_query_global_indices):
                _require(person in declared, "microbatch contains undeclared person")
                _require(observed[person].isdisjoint(indices),
                         "a scored person-query appears in multiple microbatches")
                observed[person].update(indices)
        _require(all(len(observed[person]) == denominator
                     for person, denominator in declared.items()),
                 "microbatch query coverage differs from global person denominators")

    @property
    def person_count(self) -> int:
        return len(self.person_ids)

    def receipt(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "logical_batch_id": self.logical_batch_id,
            "person_ids": list(self.person_ids),
            "person_query_denominators": list(self.person_query_denominators),
            "microbatches": [value.receipt() for value in self.microbatches],
            "scientific_adam_updates": 1,
        }


@dataclass(frozen=True)
class WeightedMicrobatchPlan:
    """One physical query segment with aggregated occurrence coefficients."""

    identity_sha256: str
    region_id: str
    query_start: int
    query_stop: int
    person_ids: tuple[str, ...]
    scored_query_global_indices: tuple[tuple[int, ...], ...]
    loss_coefficients: tuple[tuple[float, ...], ...]
    occurrence_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _require(bool(SHA256.fullmatch(self.identity_sha256)),
                 "weighted microbatch SHA256 identity required")
        _require(isinstance(self.region_id, str) and self.region_id.strip(),
                 "weighted microbatch region id required")
        _require(type(self.query_start) is int and type(self.query_stop) is int
                 and 0 <= self.query_start < self.query_stop,
                 "weighted microbatch requires explicit nonempty query bounds")
        _require(bool(self.person_ids) and len(set(self.person_ids)) == len(self.person_ids)
                 and len(self.person_ids) == len(self.scored_query_global_indices)
                 == len(self.loss_coefficients) == len(self.occurrence_counts),
                 "weighted microbatch person/query/weight axes differ")
        for person, indices, coefficients, counts in zip(
                self.person_ids, self.scored_query_global_indices,
                self.loss_coefficients, self.occurrence_counts):
            _require(isinstance(person, str) and person.strip()
                     and bool(indices) and len(indices) == len(set(indices))
                     and all(type(index) is int and
                             self.query_start <= index < self.query_stop
                             for index in indices),
                     "weighted scored queries must be unique and inside explicit bounds")
            _require(len(coefficients) == len(indices)
                     and all(type(value) is float and math.isfinite(value) and value > 0
                             for value in coefficients),
                     "positive finite coefficient required for every weighted query")
            _require(len(counts) == len(indices)
                     and all(type(value) is int and value > 0 for value in counts),
                     "positive occurrence count required for every weighted query")

    @property
    def unique_scored_queries(self) -> int:
        return sum(len(value) for value in self.scored_query_global_indices)

    @property
    def occurrence_count(self) -> int:
        return sum(sum(value) for value in self.occurrence_counts)

    @property
    def coefficient_sum(self) -> float:
        return math.fsum(value for row in self.loss_coefficients for value in row)

    def receipt(self) -> dict[str, object]:
        return {
            "identity_sha256": self.identity_sha256,
            "region_id": self.region_id,
            "query_start": self.query_start, "query_stop": self.query_stop,
            "person_ids": list(self.person_ids),
            "scored_query_global_indices":
                [list(value) for value in self.scored_query_global_indices],
            "loss_coefficients": [list(value) for value in self.loss_coefficients],
            "occurrence_counts": [list(value) for value in self.occurrence_counts],
        }


@dataclass(frozen=True)
class WeightedLogicalBatchPlan:
    """One atomic Adam update for the versioned occurrence-weighted objective."""

    step_index: int
    logical_batch_id: str
    microbatches: tuple[WeightedMicrobatchPlan, ...]
    declared_occurrences: int

    def __post_init__(self) -> None:
        _require(type(self.step_index) is int and self.step_index >= 0,
                 "nonnegative weighted logical-batch step required")
        _require(isinstance(self.logical_batch_id, str) and self.logical_batch_id.strip(),
                 "weighted logical batch id required")
        _require(bool(self.microbatches) and
                 all(isinstance(value, WeightedMicrobatchPlan)
                     for value in self.microbatches),
                 "ordered weighted microbatch plan required")
        _require(type(self.declared_occurrences) is int and self.declared_occurrences > 0,
                 "positive declared occurrence count required")
        observed: set[tuple[str, int]] = set()
        for microbatch in self.microbatches:
            for person, indices in zip(microbatch.person_ids,
                                       microbatch.scored_query_global_indices):
                cells = {(person, index) for index in indices}
                _require(observed.isdisjoint(cells),
                         "weighted person-query must be aggregated within one update")
                observed.update(cells)
        _require(self.occurrence_count == self.declared_occurrences,
                 "weighted occurrence counts differ from declared plan")
        _require(math.isclose(self.coefficient_sum, 1.0, rel_tol=0.0, abs_tol=1e-12),
                 "weighted update coefficients must sum to one")

    @property
    def unique_scored_queries(self) -> int:
        return sum(value.unique_scored_queries for value in self.microbatches)

    @property
    def occurrence_count(self) -> int:
        return sum(value.occurrence_count for value in self.microbatches)

    @property
    def coefficient_sum(self) -> float:
        return math.fsum(value.coefficient_sum for value in self.microbatches)

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "r02_lai_weighted_logical_batch_v1",
            "step_index": self.step_index,
            "logical_batch_id": self.logical_batch_id,
            "microbatches": [value.receipt() for value in self.microbatches],
            "declared_occurrences": self.declared_occurrences,
            "unique_scored_queries": self.unique_scored_queries,
            "coefficient_sum": self.coefficient_sum,
            "scientific_adam_updates": 1,
        }


TrainingPlan = LogicalBatchPlan | WeightedLogicalBatchPlan
TrainingMicrobatchPlan = MicrobatchPlan | WeightedMicrobatchPlan


@dataclass(frozen=True)
class DonorTrainingBatch:
    identity_sha256: str
    person_ids: tuple[str, ...]
    context: DonorContextBatch
    truth_state: torch.Tensor
    truth_observed: torch.Tensor
    query_core: torch.Tensor
    query_global_index: torch.Tensor

    def __post_init__(self) -> None:
        _require(bool(SHA256.fullmatch(self.identity_sha256)),
                 "training microbatch SHA256 identity required")
        _require(bool(self.person_ids) and all(isinstance(value, str) and value
                                               for value in self.person_ids),
                 "training microbatch person ids required")
        _require(isinstance(self.context, DonorContextBatch),
                 "training input must contain DonorContextBatch")
        shape = self.context.query_valid.shape
        _require(self.truth_state.shape == self.truth_observed.shape ==
                 self.query_core.shape == self.query_global_index.shape == shape and
                 shape[0] == len(self.person_ids),
                 "truth/core axes must match DonorContextBatch queries and people")
        _require(self.truth_state.dtype in (torch.int32, torch.int64) and
                 self.query_global_index.dtype in (torch.int32, torch.int64) and
                 self.truth_observed.dtype == self.query_core.dtype == torch.bool,
                 "truth/query indices must be integer and truth/core masks boolean")
        _require(all(value.device == self.context.query_valid.device for value in
                     (self.truth_state, self.truth_observed, self.query_core,
                      self.query_global_index)),
                 "truth/core and DonorContextBatch must share one device")
        _require(not bool((self.truth_observed & ~self.context.query_valid).any()),
                 "truth_observed cannot advertise an invalid query")


@dataclass(frozen=True)
class FrozenReadoutTrainingBatch:
    """Authenticated cached input for one weighted physical query segment."""
    identity_sha256: str
    person_ids: tuple[str, ...]
    head_input: torch.Tensor
    frozen_source_logits: torch.Tensor
    support: torch.Tensor
    truth_state: torch.Tensor
    truth_observed: torch.Tensor
    query_core: torch.Tensor
    query_valid: torch.Tensor
    query_global_index: torch.Tensor

    def __post_init__(self) -> None:
        _require(bool(SHA256.fullmatch(self.identity_sha256)),
                 "frozen-readout microbatch SHA256 identity required")
        _require(bool(self.person_ids) and len(set(self.person_ids)) == len(self.person_ids)
                 and all(isinstance(value, str) and value for value in self.person_ids),
                 "frozen-readout person ids required")
        shape = self.query_valid.shape
        _require(self.head_input.ndim == 3 and self.head_input.shape[:2] == shape
                 and self.frozen_source_logits.shape == (*shape, JOINT_ANCESTRY_STATES)
                 and self.support.shape == self.truth_state.shape ==
                    self.truth_observed.shape == self.query_core.shape ==
                    self.query_global_index.shape == shape
                 and shape[0] == len(self.person_ids),
                 "frozen-readout cached axes differ")
        _require(self.head_input.is_floating_point()
                 and self.frozen_source_logits.is_floating_point()
                 and self.support.dtype == self.truth_observed.dtype ==
                    self.query_core.dtype == self.query_valid.dtype == torch.bool
                 and self.truth_state.dtype in (torch.int32, torch.int64)
                 and self.query_global_index.dtype in (torch.int32, torch.int64),
                 "frozen-readout dtypes differ")
        tensors = tuple(value for value in self.__dict__.values() if torch.is_tensor(value))
        _require(all(value.device == self.head_input.device for value in tensors),
                 "frozen-readout tensors must share one device")
        _require(bool(torch.isfinite(self.head_input).all())
                 and bool(torch.isfinite(self.frozen_source_logits).all())
                 and not bool((self.truth_observed & ~self.query_valid).any()),
                 "frozen-readout cache contains nonfinite values or invalid truth")


def donor_training_identity(
        *, task_id: str, data_scope: str,
        model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        optimizer_specification: AdamSpecification,
        source_paths: Mapping[str, Path], policy_path: Path,
        objective: Mapping[str, object], seed: int,
        logical_batches: Sequence[TrainingPlan]) -> dict[str, object]:
    """Build identity from files read now, model configuration and exact plans."""
    _require(isinstance(task_id, str) and task_id.strip(), "training task id required")
    _require(data_scope in (SYNTHETIC_SCOPE, FIT_SCOPE), "unknown donor-training data scope")
    model_kind = _model_kind(model)
    _validate_adam(model, optimizer, optimizer_specification)
    _require(type(seed) is int and seed >= 0, "explicit nonnegative seed required")
    plans = tuple(logical_batches)
    legacy = bool(plans) and all(isinstance(value, LogicalBatchPlan) for value in plans)
    weighted = bool(plans) and all(isinstance(value, WeightedLogicalBatchPlan)
                                   for value in plans)
    _require((legacy or weighted)
             and tuple(value.step_index for value in plans) == tuple(range(len(plans))),
             "logical batches must cover consecutive checkpoint steps from zero")
    readout = model_kind == FROZEN_READOUT_MODEL_KIND
    _require(not readout or weighted,
             "frozen readout requires the occurrence-weighted logical plan")
    expected_objective = (FROZEN_READOUT_OBJECTIVE if readout else
                          DONOR_OBJECTIVE if legacy else WEIGHTED_DONOR_OBJECTIVE)
    _require(dict(objective) == expected_objective,
             "donor objective differs from the explicit logical-batch schema")
    policy = Path(policy_path)
    _require(policy.is_file(), "training policy file does not exist")
    configuration = _json_copy(model.configuration)
    return {
        "schema": (FROZEN_READOUT_TRAINING_SCHEMA if readout else
                   TRAINING_SCHEMA if legacy else WEIGHTED_TRAINING_SCHEMA),
        "task_id": task_id,
        "data_scope": data_scope,
        "scientific_scope": ("SOFTWARE_FIXTURE_ONLY" if data_scope == SYNTHETIC_SCOPE
                             else "FIT_DEVELOPMENT_SHARED_DONORS_NO_INDEPENDENT_VALIDATION"),
        **({"model_kind": model_kind} if readout else {}),
        "model_configuration": configuration,
        "model_configuration_sha256": identity_digest(configuration),
        "initial_model_optimizer_sha256": optimizer_start_digest(model, optimizer),
        "sources": hash_sources(source_paths),
        "policy": {"path": str(policy.resolve()), "sha256": _sha_file(policy)},
        "objective": _json_copy(dict(objective)), "seed": seed,
        "optimizer": optimizer_specification.receipt(),
        "backend_determinism": backend_determinism_receipt(),
        "logical_batches": [value.receipt() for value in plans],
        "checkpoint_step_semantics":
            "MANIFEST_NEXT_STEP_IS_NUMBER_OF_COMPLETED_LOGICAL_BATCH_ADAM_UPDATES",
    }


def _partial_identity(training_identity: dict[str, object], plan: TrainingPlan,
                      history: list[float], model: torch.nn.Module,
                      optimizer: torch.optim.Optimizer) -> dict[str, object]:
    return {
        "schema": (FROZEN_READOUT_PARTIAL_SCHEMA
                   if training_identity["schema"] == FROZEN_READOUT_TRAINING_SCHEMA
                   else PARTIAL_SCHEMA if isinstance(plan, LogicalBatchPlan)
                   else WEIGHTED_PARTIAL_SCHEMA),
        "training_identity_sha256": identity_digest(training_identity),
        "step_index": plan.step_index,
        "logical_batch": plan.receipt(),
        "seed": training_identity["seed"],
        "optimizer": training_identity["optimizer"],
        "model_configuration_sha256": training_identity["model_configuration_sha256"],
        "completed_history_sha256": _sha_bytes(_canonical(history)),
        "model_optimizer_start_sha256": optimizer_start_digest(model, optimizer),
    }


def _validate_progress(progress: dict[str, object], plan: TrainingPlan,
                       *, frozen_readout: bool = False) -> None:
    weighted = isinstance(plan, WeightedLogicalBatchPlan)
    integer_fields = ["next_person_batch_index", "next_tile_index",
                      "total_microbatches", "scored_queries_processed",
                      "unsupported_scored_queries"]
    if weighted:
        integer_fields.extend(("occurrences_processed", "unsupported_occurrences"))
    common = (isinstance(progress, dict) and
             progress.get("phase") == ("LAI_WEIGHTED_OCCURRENCE_MICROBATCH"
                                        if weighted else "LAI_PERSON_MICROBATCH") and
             all(type(progress.get(name)) is int and progress[name] >= 0
                 for name in integer_fields) and
             progress["next_tile_index"] == 0 and
             progress["total_microbatches"] == len(plan.microbatches) and
             progress["next_person_batch_index"] <= progress["total_microbatches"] and
             progress["scored_queries_processed"] <=
                (plan.unique_scored_queries if weighted
                 else sum(plan.person_query_denominators)) and
             (not weighted or progress["occurrences_processed"] <=
                plan.declared_occurrences) and
             isinstance(progress.get("unsupported_query_nll_sum"), (int, float)) and
             math.isfinite(progress["unsupported_query_nll_sum"]))
    if weighted:
        objective = (isinstance(progress.get("objective_sum"), (int, float)) and
                     math.isfinite(progress["objective_sum"]) and
                     isinstance(progress.get("coefficient_sum_processed"), (int, float)) and
                     math.isfinite(progress["coefficient_sum_processed"]) and
                     isinstance(progress.get("unsupported_weight_sum"), (int, float)) and
                     math.isfinite(progress["unsupported_weight_sum"]))
    else:
        objective = (isinstance(progress.get("person_loss_sum"), (int, float)) and
                     math.isfinite(progress["person_loss_sum"]))
    _require(common and objective,
             "donor partial progress differs from the logical batch")


@dataclass(frozen=True)
class DonorBatchLoss:
    person_contributions: torch.Tensor
    scored_queries: int
    unsupported_scored_queries: int
    unsupported_query_nll_sum: float
    occurrences: int = 0
    unsupported_occurrences: int = 0
    coefficient_sum: float = 0.0
    unsupported_weight_sum: float = 0.0


def donor_person_losses(model: AutonomousDonorContextLAI,
                        batch: DonorTrainingBatch,
                        microbatch_plan: MicrobatchPlan,
                        global_query_denominators: Mapping[str, int]) -> DonorBatchLoss:
    """Return each fragment's contribution to its person's global mean NLL."""
    _require(batch.person_ids == microbatch_plan.person_ids,
             "training microbatch person order differs from the logical batch")
    output = model(batch.context)  # truth/core are intentionally absent here
    _require(output.logits.shape == (*batch.truth_state.shape, JOINT_ANCESTRY_STATES),
             "donor model logits do not match truth axes or six-state order")
    scored = batch.truth_observed & batch.query_core & batch.context.query_valid
    observed_indices = tuple(tuple(int(index) for index in
                                   batch.query_global_index[row][scored[row]].tolist())
                             for row in range(len(batch.person_ids)))
    _require(observed_indices == microbatch_plan.scored_query_global_indices,
             "scored-query ownership/indices differ from the fixed microbatch plan")
    _require(bool(torch.isfinite(output.logits[scored]).all()),
             "scientifically scored donor logits must be finite")
    scored_truth = batch.truth_state[scored]
    _require(bool(((scored_truth >= 0) &
                   (scored_truth < JOINT_ANCESTRY_STATES)).all()),
             "scientifically scored truth state is outside 0..5")
    # Invalid values behind truth_observed=False or query_core=False cannot
    # enter gather.  This is also safe for invalid query padding.
    safe_truth = torch.where(scored, batch.truth_state,
                             torch.zeros_like(batch.truth_state)).long()
    log_probability = F.log_softmax(output.logits, dim=-1)
    selected = log_probability.gather(-1, safe_truth[..., None]).squeeze(-1)
    denominators = torch.as_tensor(
        [global_query_denominators[person] for person in batch.person_ids],
                                   dtype=selected.dtype, device=selected.device)
    contributions = -(selected * scored.to(selected.dtype)).sum(1) / denominators
    _require(contributions.shape == (len(batch.person_ids),) and
             bool(torch.isfinite(contributions).all()) and
             contributions.requires_grad,
             "donor person-query contribution must be finite and differentiable")
    unsupported = scored & ~output.support
    unsupported_sum = float((-(selected.detach()) * unsupported.to(selected.dtype)).sum())
    return DonorBatchLoss(contributions, int(scored.sum()),
                          int(unsupported.sum()), unsupported_sum)


def donor_weighted_occurrence_loss(
        model: torch.nn.Module,
        batch: DonorTrainingBatch | FrozenReadoutTrainingBatch,
        microbatch_plan: WeightedMicrobatchPlan) -> DonorBatchLoss:
    """Return ``sum(coefficient * NLL)`` for one aggregated query fragment."""
    _require(batch.person_ids == microbatch_plan.person_ids,
             "weighted training microbatch person order differs")
    readout = isinstance(batch, FrozenReadoutTrainingBatch)
    # The cached batch is a new closed interface and therefore requires its
    # versioned model marker.  Preserve the older standalone loss helper's use
    # with small fixed-logit test modules; the recoverable trainer itself still
    # accepts only AutonomousDonorContextLAI for DonorTrainingBatch.
    _require(not readout or _model_kind(model) == FROZEN_READOUT_MODEL_KIND,
             "weighted model and cached batch kinds differ")
    output = model(batch if readout else batch.context)
    _require(output.logits.shape == (*batch.truth_state.shape, JOINT_ANCESTRY_STATES),
             "weighted donor logits do not match truth axes or six-state order")
    query_valid = batch.query_valid if readout else batch.context.query_valid
    scored = batch.truth_observed & batch.query_core & query_valid
    observed_indices = tuple(tuple(int(index) for index in
                                   batch.query_global_index[row][scored[row]].tolist())
                             for row in range(len(batch.person_ids)))
    _require(observed_indices == microbatch_plan.scored_query_global_indices,
             "weighted scored-query ownership/indices differ from fixed plan")
    _require(bool(torch.isfinite(output.logits[scored]).all()),
             "weighted scored donor logits must be finite")
    scored_truth = batch.truth_state[scored]
    _require(bool(((scored_truth >= 0) &
                   (scored_truth < JOINT_ANCESTRY_STATES)).all()),
             "weighted scored truth state is outside 0..5")
    safe_truth = torch.where(scored, batch.truth_state,
                             torch.zeros_like(batch.truth_state)).long()
    selected = F.log_softmax(output.logits, dim=-1).gather(
        -1, safe_truth[..., None]).squeeze(-1)
    coefficient_rows = [torch.as_tensor(values, dtype=selected.dtype,
                                        device=selected.device)
                        for values in microbatch_plan.loss_coefficients]
    coefficients = torch.cat(coefficient_rows)
    nll = -selected[scored]
    _require(coefficients.shape == nll.shape and bool(torch.isfinite(nll).all()),
             "weighted NLL/coefficient axes differ or contain nonfinite values")
    contribution = (coefficients * nll).sum().reshape(1)
    _require(contribution.requires_grad and bool(torch.isfinite(contribution).all()),
             "weighted contribution must be finite and differentiable")
    unsupported = (~output.support)[scored]
    unsupported_weight = float(coefficients.detach()[unsupported].double().sum())
    unsupported_sum = float(
        (coefficients.detach()[unsupported] * nll.detach()[unsupported]).double().sum())
    unsupported_occurrences = sum(
        sum(count for count, is_unsupported in zip(counts, mask.tolist())
            if is_unsupported)
        for counts, mask in zip(microbatch_plan.occurrence_counts,
                                (~output.support)[scored].split(
                                    [len(row) for row in
                                     microbatch_plan.scored_query_global_indices])))
    return DonorBatchLoss(
        contribution, microbatch_plan.unique_scored_queries, int(unsupported.sum()),
        unsupported_sum, microbatch_plan.occurrence_count, unsupported_occurrences,
        microbatch_plan.coefficient_sum, unsupported_weight)


@dataclass(frozen=True)
class DonorTrainingCallResult:
    status: str
    completed_steps: int
    history: tuple[float, ...]
    logical_batch_id: str | None
    next_microbatch: int
    unsupported_scored_queries: int
    unsupported_occurrences: int
    unsupported_scored_query_mean_nll: float | None


TrainingBatch = DonorTrainingBatch | FrozenReadoutTrainingBatch
BatchCallback = Callable[[torch.nn.Module, int, int], TrainingBatch]
BatchSource = Sequence[TrainingBatch] | BatchCallback


def _get_batch(source: BatchSource, model: torch.nn.Module,
               plan: TrainingPlan, index: int) -> TrainingBatch:
    if callable(source):
        value = source(model, plan.step_index, index)
    else:
        _require(not isinstance(source, (str, bytes)) and
                 len(source) == len(plan.microbatches),
                 "batch sequence must contain the complete logical batch")
        value = source[index]
    expected_type = (FrozenReadoutTrainingBatch
                     if _model_kind(model) == FROZEN_READOUT_MODEL_KIND
                     else DonorTrainingBatch)
    _require(type(value) is expected_type,
             "batch source returned a different versioned training batch kind")
    _require(value.identity_sha256 == plan.microbatches[index].identity_sha256,
             "batch source returned a different authenticated microbatch")
    return value


def run_donor_training_call(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        optimizer_specification: AdamSpecification, checkpoint_root: Path,
        training_identity: dict[str, object],
        logical_batches: Sequence[TrainingPlan], batch_source: BatchSource,
        *, max_microbatches_this_call: int | None) -> DonorTrainingCallResult:
    """Resume and complete at most one scientific logical-batch Adam update."""
    model_kind = _model_kind(model)
    frozen_readout = model_kind == FROZEN_READOUT_MODEL_KIND
    _require(all(parameter.device.type == "cpu" for parameter in model.parameters()),
             "exact RNG recovery currently requires CPU donor training")
    _validate_adam(model, optimizer, optimizer_specification)
    plans = tuple(logical_batches)
    declared_schema = training_identity.get("schema")
    declared_weighted = declared_schema in (
        WEIGHTED_TRAINING_SCHEMA, FROZEN_READOUT_TRAINING_SCHEMA)
    expected_schema = (FROZEN_READOUT_TRAINING_SCHEMA if frozen_readout else
                       WEIGHTED_TRAINING_SCHEMA if declared_weighted else TRAINING_SCHEMA)
    expected_objective = (FROZEN_READOUT_OBJECTIVE if frozen_readout else
                          WEIGHTED_DONOR_OBJECTIVE if declared_weighted else DONOR_OBJECTIVE)
    _require(training_identity.get("schema") == expected_schema and
             ((frozen_readout and training_identity.get("model_kind") == model_kind)
              or (not frozen_readout and "model_kind" not in training_identity)) and
             training_identity.get("model_configuration") == model.configuration and
             training_identity.get("optimizer") == optimizer_specification.receipt() and
             training_identity.get("objective") == expected_objective and
             training_identity.get("backend_determinism") ==
                backend_determinism_receipt() and
             training_identity.get("logical_batches") ==
                [value.receipt() for value in plans] and
             training_identity.get("data_scope") in (SYNTHETIC_SCOPE, FIT_SCOPE),
             "donor training identity differs from model, optimizer, objective, or plans")
    legacy = bool(plans) and all(isinstance(value, LogicalBatchPlan) for value in plans)
    weighted = bool(plans) and all(isinstance(value, WeightedLogicalBatchPlan)
                                   for value in plans)
    _require((legacy and not declared_weighted and not frozen_readout)
             or (weighted and declared_weighted),
             "one versioned donor logical-batch schema required")
    _require(max_microbatches_this_call is None or
             (type(max_microbatches_this_call) is int and
              max_microbatches_this_call > 0),
             "max microbatches this call must be positive or explicitly None")

    completed_store = CheckpointStore(Path(checkpoint_root) / "completed",
                                      training_identity)
    partial_store = None
    try:
        completed, history = completed_store.restore(model, optimizer)
        _require(0 <= completed <= len(plans),
                 "checkpoint step exceeds declared logical batches")
        if completed == len(plans):
            return DonorTrainingCallResult(
                "ALREADY_COMPLETE", completed, tuple(history), None, 0, 0, 0, None)
        plan = plans[completed]
        _require(plan.step_index == completed,
                 "next logical batch does not match checkpoint step")
        partial_store = PartialUpdateStore(
            Path(checkpoint_root) / f"update-{completed:012d}",
            _partial_identity(training_identity, plan, history, model, optimizer))
        restored = partial_store.restore(model, optimizer)
        if restored is None:
            optimizer.zero_grad(set_to_none=True)
            progress = {"phase": ("LAI_PERSON_MICROBATCH"
                                   if legacy else "LAI_WEIGHTED_OCCURRENCE_MICROBATCH"),
                        "next_person_batch_index": 0, "next_tile_index": 0,
                        "total_microbatches": len(plan.microbatches),
                        "scored_queries_processed": 0,
                        "unsupported_scored_queries": 0,
                        "unsupported_query_nll_sum": 0.0}
            if legacy:
                progress["person_loss_sum"] = 0.0
            else:
                progress.update(occurrences_processed=0, objective_sum=0.0,
                                coefficient_sum_processed=0.0,
                                unsupported_weight_sum=0.0,
                                unsupported_occurrences=0)
        else:
            progress, partial_state = restored
            _require(isinstance(partial_state, dict) and
                     set(partial_state) == {"parameter_gradients"},
                     "donor partial checkpoint fields differ")
            restore_parameter_gradients(model,
                                        partial_state["parameter_gradients"])
        _validate_progress(progress, plan, frozen_readout=frozen_readout)
        progress = dict(progress)
        processed_this_call = 0
        while progress["next_person_batch_index"] < progress["total_microbatches"]:
            if (max_microbatches_this_call is not None and
                    processed_this_call == max_microbatches_this_call):
                unsupported_denominator = (progress["unsupported_scored_queries"]
                    if legacy else progress["unsupported_weight_sum"])
                return DonorTrainingCallResult(
                    "PARTIAL", completed, tuple(history), plan.logical_batch_id,
                    progress["next_person_batch_index"],
                    progress["unsupported_scored_queries"],
                    progress.get("unsupported_occurrences", 0),
                    (progress["unsupported_query_nll_sum"] /
                     unsupported_denominator
                     if unsupported_denominator else None))
            index = progress["next_person_batch_index"]
            batch = _get_batch(batch_source, model, plan, index)
            if legacy:
                global_denominators = dict(zip(plan.person_ids,
                                               plan.person_query_denominators))
                loss_result = donor_person_losses(
                    model, batch, plan.microbatches[index], global_denominators)
            else:
                loss_result = donor_weighted_occurrence_loss(
                    model, batch, plan.microbatches[index])
            contributions = loss_result.person_contributions
            (contributions.sum() / plan.person_count
             if legacy else contributions.sum()).backward()
            _require(all(parameter.grad is None or
                         bool(torch.isfinite(parameter.grad).all())
                         for parameter in model.parameters()),
                     "nonfinite donor-model gradient")
            common_progress = dict(
                next_person_batch_index=index + 1,
                scored_queries_processed=(progress["scored_queries_processed"] +
                                          loss_result.scored_queries),
                unsupported_scored_queries=(
                    progress["unsupported_scored_queries"] +
                    loss_result.unsupported_scored_queries),
                unsupported_query_nll_sum=(
                    progress["unsupported_query_nll_sum"] +
                    loss_result.unsupported_query_nll_sum))
            if legacy:
                common_progress["person_loss_sum"] = (
                    progress["person_loss_sum"] +
                    float(contributions.detach().double().sum()))
            else:
                common_progress.update(
                    occurrences_processed=(progress["occurrences_processed"] +
                                           loss_result.occurrences),
                    objective_sum=(progress["objective_sum"] +
                                   float(contributions.detach().double().sum())),
                    coefficient_sum_processed=(
                        progress["coefficient_sum_processed"] +
                        loss_result.coefficient_sum),
                    unsupported_weight_sum=(progress["unsupported_weight_sum"] +
                                            loss_result.unsupported_weight_sum),
                    unsupported_occurrences=(progress["unsupported_occurrences"] +
                                             loss_result.unsupported_occurrences))
            progress.update(common_progress)
            partial_store.save(
                model, optimizer, progress,
                {"parameter_gradients": capture_parameter_gradients(model)})
            processed_this_call += 1
            del contributions, loss_result, batch

        if legacy:
            _require(progress["scored_queries_processed"] ==
                     sum(plan.person_query_denominators),
                     "logical batch did not process every declared person-query once")
        else:
            _require(progress["scored_queries_processed"] == plan.unique_scored_queries
                     and progress["occurrences_processed"] == plan.declared_occurrences
                     and math.isclose(progress["coefficient_sum_processed"], 1.0,
                                      rel_tol=0.0, abs_tol=1e-12),
                     "weighted logical batch did not process its exact occurrences/weights")
        if optimizer_specification.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimizer_specification.gradient_clip_norm,
                error_if_nonfinite=True)
        objective = (progress["person_loss_sum"] / plan.person_count
                     if legacy else progress["objective_sum"])
        optimizer.step()
        history.append(float(objective))
        # The completed generation is durable before the in-flight gradients
        # are removed.  CheckpointStore retains its existing two generations.
        completed_store.save(model, optimizer, completed + 1, history)
        partial_store.clear()
        unsupported_denominator = (progress["unsupported_scored_queries"]
                                   if legacy else progress["unsupported_weight_sum"])
        return DonorTrainingCallResult(
            "COMPLETE", completed + 1, tuple(history), plan.logical_batch_id,
            len(plan.microbatches),
            progress["unsupported_scored_queries"],
            progress.get("unsupported_occurrences", 0),
            (progress["unsupported_query_nll_sum"] /
             unsupported_denominator if unsupported_denominator else None))
    finally:
        if partial_store is not None:
            partial_store.close()
        completed_store.close()
