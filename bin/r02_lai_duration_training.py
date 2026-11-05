#!/usr/bin/env python3
"""Generic, resumable optimization for complete-person LAI duration rows.

This module does not select duration stages, rates, ancestry switches, a start
prior, or optimizer values.  The caller supplies each value and its scientific
provenance.  It processes one declared person chunk at a time, saves detached
accumulated gradients after every chunk, and resumes at the next chunk without
reconstructing earlier neural emissions.

Inputs retain centiMorgan coordinates.  Conversion to Morgans remains inside
``duration_unordered_label_loss``.  ``label_mask=False`` marginalizes only the
label at a query; the neural potential and query remain in the sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from r02_context_checkpoint import (
    CheckpointStore, _atomic_bytes, _canonical, _sha_bytes, identity_digest,
)
from r02_context_update_checkpoint import (
    PartialUpdateStore, capture_parameter_gradients, optimizer_start_digest,
    restore_parameter_gradients,
)
from r02_lai_diploid_duration import ANCESTRIES
from r02_lai_duration_model import (
    DurationParameterDeclaration, DurationSequenceBatch,
    duration_unordered_label_loss,
)


TRAINING_SCHEMA = "r02_lai_duration_training_v1"
PARTIAL_SCHEMA = "r02_lai_duration_partial_update_v1"
BUDGET_SCHEMA = "r02_lai_duration_budget_v1"
BUDGET_FILE = "DURATION_GRADIENT_BUDGET.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
LEARNING_STATUSES = frozenset(("FIXED", "FIT_OPTIMIZED_DIAGNOSTIC"))
PERMITTED_CLAIMS = frozenset(("FIXED_INPUT", "FIT_OPTIMIZED_DIAGNOSTIC"))
SYNTHETIC_SCOPE = "SYNTHETIC_SOFTWARE_FIXTURE"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor_receipt(value: Tensor) -> dict[str, object]:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return {"sha256": digest, "dtype": str(array.dtype), "shape": list(array.shape)}


def _module_state_receipt(module: nn.Module) -> dict[str, dict[str, object]]:
    return {name: _tensor_receipt(value)
            for name, value in sorted(module.state_dict().items())}


@dataclass(frozen=True)
class ParameterProvenance:
    """Required source and allowed interpretation for one supplied parameter."""

    declaration_id: str
    initial_value_source: str
    source_sha256: str
    learning_status: str
    permitted_claim: str
    interpretation_limit: str

    def __post_init__(self) -> None:
        _require(all(isinstance(value, str) and value.strip() for value in (
            self.declaration_id, self.initial_value_source,
            self.interpretation_limit)), "parameter provenance strings are required")
        _require(bool(SHA256.fullmatch(self.source_sha256)),
                 "parameter provenance requires a source SHA256")
        _require(self.learning_status in LEARNING_STATUSES,
                 "parameter learning status must be explicit")
        _require(self.permitted_claim in PERMITTED_CLAIMS,
                 "duration parameters cannot be labelled as demographic estimates")
        _require((self.learning_status == "FIXED") ==
                 (self.permitted_claim == "FIXED_INPUT"),
                 "parameter learning status and permitted claim differ")

    def receipt(self) -> dict[str, str]:
        return {
            "declaration_id": self.declaration_id,
            "initial_value_source": self.initial_value_source,
            "source_sha256": self.source_sha256,
            "learning_status": self.learning_status,
            "permitted_claim": self.permitted_claim,
            "interpretation_limit": self.interpretation_limit,
        }

    def core_declaration(self) -> str:
        return json.dumps(self.receipt(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class DurationParameterSpecification:
    """Numerical initial values supplied by a separate scientific declaration."""

    duration_stages: int
    log_stage_rates: Tensor
    switch_logits: Tensor
    joint_start_prior: Tensor
    stages_provenance: ParameterProvenance
    rates_provenance: ParameterProvenance
    switches_provenance: ParameterProvenance
    prior_provenance: ParameterProvenance

    def __post_init__(self) -> None:
        _require(type(self.duration_stages) is int and self.duration_stages > 0,
                 "duration stages must be supplied explicitly")
        d = self.duration_stages
        _require(isinstance(self.log_stage_rates, Tensor) and
                 tuple(self.log_stage_rates.shape) == (len(ANCESTRIES), d) and
                 self.log_stage_rates.dtype in (torch.float32, torch.float64) and
                 bool(torch.isfinite(self.log_stage_rates).all()),
                 "supplied log stage rates must be finite [3,D]")
        _require(isinstance(self.switch_logits, Tensor) and
                 tuple(self.switch_logits.shape) == (len(ANCESTRIES), len(ANCESTRIES)) and
                 self.switch_logits.dtype == self.log_stage_rates.dtype and
                 bool(torch.isfinite(self.switch_logits).all()),
                 "supplied switch logits must be finite [3,3]")
        n = len(ANCESTRIES) * d
        prior = self.joint_start_prior
        _require(isinstance(prior, Tensor) and tuple(prior.shape) == (n, n) and
                 prior.dtype == self.log_stage_rates.dtype and
                 bool(torch.isfinite(prior).all()) and bool((prior >= 0).all()) and
                 bool(prior.sum() > 0),
                 "supplied joint start prior must be finite nonnegative [3D,3D]")
        tolerance = 128 * torch.finfo(prior.dtype).eps
        _require(bool(torch.allclose(prior, prior.T, atol=tolerance, rtol=0)),
                 "supplied joint start prior must be symmetric under homologue exchange")
        _require(self.stages_provenance.learning_status == "FIXED",
                 "the discrete stage count cannot be optimized inside one fit")
        pairs = ((self.rates_provenance, "rates"),
                 (self.switches_provenance, "switches"),
                 (self.prior_provenance, "prior"))
        for provenance, name in pairs:
            if provenance.learning_status == "FIT_OPTIMIZED_DIAGNOSTIC":
                _require(name != "prior" or bool((prior > 0).all()),
                         "a learnable start prior must be strictly positive")


class DurationParameters(nn.Module):
    """Own supplied fixed buffers and optionally FIT-optimized tensors."""

    def __init__(self, specification: DurationParameterSpecification):
        super().__init__()
        _require(isinstance(specification, DurationParameterSpecification),
                 "duration parameter specification required")
        self.specification = specification
        self.duration_stages = specification.duration_stages
        self._install("log_stage_rates", specification.log_stage_rates,
                      specification.rates_provenance)
        self._install("switch_logits", specification.switch_logits,
                      specification.switches_provenance)
        prior = specification.joint_start_prior / specification.joint_start_prior.sum()
        if specification.prior_provenance.learning_status == "FIT_OPTIMIZED_DIAGNOSTIC":
            self.joint_start_prior_logits = nn.Parameter(prior.log().detach().clone())
        else:
            self.register_buffer("fixed_joint_start_prior", prior.detach().clone())

    def _install(self, name: str, value: Tensor, provenance: ParameterProvenance) -> None:
        copied = value.detach().clone()
        if provenance.learning_status == "FIT_OPTIMIZED_DIAGNOSTIC":
            setattr(self, name, nn.Parameter(copied))
        else:
            self.register_buffer(name, copied)

    def joint_start_prior(self, rows: int) -> Tensor:
        _require(type(rows) is int and rows > 0, "positive duration row count required")
        if hasattr(self, "joint_start_prior_logits"):
            symmetric_logits = (self.joint_start_prior_logits +
                                self.joint_start_prior_logits.T) / 2
            prior = symmetric_logits.flatten().softmax(0).reshape(
                self.joint_start_prior_logits.shape)
        else:
            prior = self.fixed_joint_start_prior
        return prior.unsqueeze(0).expand(rows, -1, -1)

    def core_declaration(self) -> DurationParameterDeclaration:
        spec = self.specification
        return DurationParameterDeclaration(
            duration_stages_provenance=spec.stages_provenance.core_declaration(),
            stage_rates_provenance=spec.rates_provenance.core_declaration(),
            ancestry_switch_provenance=spec.switches_provenance.core_declaration())

    def receipt(self) -> dict[str, object]:
        spec = self.specification
        return {
            "duration_stages": self.duration_stages,
            "initial_values": {
                "log_stage_rates": _tensor_receipt(spec.log_stage_rates),
                "switch_logits": _tensor_receipt(spec.switch_logits),
                "joint_start_prior": _tensor_receipt(spec.joint_start_prior),
            },
            "provenance": {
                "stages": spec.stages_provenance.receipt(),
                "rates": spec.rates_provenance.receipt(),
                "switches": spec.switches_provenance.receipt(),
                "joint_start_prior": spec.prior_provenance.receipt(),
            },
        }


class DurationTrainingState(nn.Module):
    """Checkpoint the neural emission model and duration parameters together."""

    def __init__(self, emission_model: nn.Module, duration: DurationParameters):
        super().__init__()
        _require(isinstance(emission_model, nn.Module) and
                 isinstance(duration, DurationParameters),
                 "emission model and duration parameters are required")
        self.emission_model = emission_model
        self.duration = duration


@dataclass(frozen=True)
class OptimizerSpecification:
    optimizer: str
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    gradient_clip_norm: float | None

    def __post_init__(self) -> None:
        _require(self.optimizer == "Adam", "only explicitly requested Adam is implemented")
        _require(math.isfinite(self.learning_rate) and self.learning_rate > 0,
                 "positive finite learning rate required")
        _require(isinstance(self.betas, tuple) and len(self.betas) == 2 and
                 all(math.isfinite(value) and 0 <= value < 1 for value in self.betas),
                 "two explicit Adam beta values in [0,1) are required")
        _require(math.isfinite(self.epsilon) and self.epsilon > 0,
                 "positive finite Adam epsilon required")
        _require(math.isfinite(self.weight_decay) and self.weight_decay >= 0,
                 "nonnegative finite weight decay required")
        _require(self.gradient_clip_norm is None or
                 (math.isfinite(self.gradient_clip_norm) and self.gradient_clip_norm > 0),
                 "gradient clip norm must be positive or explicitly None")

    def receipt(self) -> dict[str, object]:
        return {"optimizer": self.optimizer, "learning_rate": self.learning_rate,
                "betas": list(self.betas), "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "gradient_clip_norm": self.gradient_clip_norm}


def build_duration_optimizer(state: DurationTrainingState,
                             specification: OptimizerSpecification) -> torch.optim.Optimizer:
    parameters = [value for value in state.parameters() if value.requires_grad]
    _require(bool(parameters), "duration training has no learnable parameters")
    return torch.optim.Adam(parameters, lr=specification.learning_rate,
                            betas=specification.betas, eps=specification.epsilon,
                            weight_decay=specification.weight_decay)


def _validate_duration_optimizer(optimizer: torch.optim.Optimizer,
                                 specification: OptimizerSpecification) -> None:
    _require(type(optimizer) is torch.optim.Adam and bool(optimizer.param_groups),
             "duration optimizer must be Adam from the explicit specification")
    for group in optimizer.param_groups:
        _require(group.get("lr") == specification.learning_rate and
                 tuple(group.get("betas", ())) == specification.betas and
                 group.get("eps") == specification.epsilon and
                 group.get("weight_decay") == specification.weight_decay,
                 "live Adam values differ from the duration optimizer specification")


@dataclass(frozen=True)
class DurationUpdatePlan:
    chunk_identity_sha256: tuple[str, ...]
    global_observed_labels: int
    global_rows: int

    def __post_init__(self) -> None:
        _require(isinstance(self.chunk_identity_sha256, tuple) and
                 len(self.chunk_identity_sha256) > 0 and
                 all(bool(SHA256.fullmatch(value)) for value in self.chunk_identity_sha256),
                 "ordered duration chunk SHA256 identities are required")
        _require(type(self.global_observed_labels) is int and
                 self.global_observed_labels > 0,
                 "positive global observed-label denominator required")
        _require(type(self.global_rows) is int and self.global_rows > 0,
                 "positive complete-row count required")

    def receipt(self) -> dict[str, object]:
        return {"chunk_identity_sha256": list(self.chunk_identity_sha256),
                "global_observed_labels": self.global_observed_labels,
                "global_rows": self.global_rows}


@dataclass(frozen=True)
class DurationTrainingChunk:
    identity_sha256: str
    batch: DurationSequenceBatch


def duration_training_identity(*, task_id: str, data_declaration_sha256: str,
                               data_scope: str, data_source: str,
                               randomness_declaration_sha256: str,
                               code_sha256: dict[str, str],
                               state: DurationTrainingState,
                               optimizer: OptimizerSpecification,
                               plan: DurationUpdatePlan,
                               fit_rows: dict | None = None) -> dict[str, object]:
    _require(isinstance(task_id, str) and task_id.strip(), "duration task id required")
    _require(bool(SHA256.fullmatch(data_declaration_sha256)),
             "duration data declaration SHA256 required")
    from r02_lai_duration_fit import FIT_SCOPE, validate_fit_plan
    _require(data_scope in (SYNTHETIC_SCOPE, FIT_SCOPE), "unknown duration data scope")
    if data_scope == FIT_SCOPE:
        validate_fit_plan(fit_rows, plan)
        _require(data_declaration_sha256 == identity_digest(fit_rows),
                 "FIT declaration hash differs from authenticated row plan")
    else:
        _require(fit_rows is None, "synthetic fixtures cannot carry a FIT declaration")
    _require(isinstance(data_source, str) and data_source.strip(),
             "explicit data source required")
    _require(bool(SHA256.fullmatch(randomness_declaration_sha256)),
             "randomness and seed declaration SHA256 required")
    _require(isinstance(code_sha256, dict) and code_sha256 and
             all(isinstance(name, str) and name and bool(SHA256.fullmatch(digest))
                 for name, digest in code_sha256.items()),
             "exact duration code SHA256 mapping required")
    identity = {"schema": TRAINING_SCHEMA, "task_id": task_id,
            "data_declaration": {"sha256": data_declaration_sha256,
                                 "scope": data_scope,
                                 "source": data_source},
            "randomness_declaration_sha256": randomness_declaration_sha256,
            "code_sha256": dict(sorted(code_sha256.items())),
            "initial_training_state": _module_state_receipt(state),
            "duration_parameters": state.duration.receipt(),
            "optimizer": optimizer.receipt(), "update_plan": plan.receipt(),
            "scientific_scope": ("SYNTHETIC_FIXTURE_ONLY_NOT_HUMAN_ADMISSION"
                if data_scope == SYNTHETIC_SCOPE else
                "FIT_DEVELOPMENT_SHARED_DONORS_NO_INDEPENDENT_VALIDATION"),
            "phase_and_haplotype_claim": "UNORDERED_DIPLOID_LABELS_DO_NOT_IDENTIFY_HAPLOTYPES"}
    if fit_rows is not None:
        identity["fit_rows"] = json.loads(json.dumps(fit_rows))
    return identity


def _partial_identity(training_identity: dict[str, object], update_index: int,
                      history: list[float], state: DurationTrainingState,
                      optimizer: torch.optim.Optimizer) -> dict[str, object]:
    return {"schema": PARTIAL_SCHEMA,
            "training_identity_sha256": identity_digest(training_identity),
            "update_index": update_index,
            "completed_history_sha256": _sha_bytes(_canonical(history)),
            "model_optimizer_start_sha256": optimizer_start_digest(state, optimizer)}


def _validate_duration_progress(progress: dict[str, object]) -> None:
    """Validate duration fields layered on the shared partial-update cursor."""
    keys = ("next_person_chunk", "total_person_chunks", "rows_processed",
            "observed_labels_processed", "row_loss_sum",
            "attempts_durable_lower_bound")
    _require(isinstance(progress, dict) and
             progress.get("phase") == "DURATION_PERSON_CHUNKS" and
             progress.get("next_person_batch_index") ==
                progress.get("next_person_chunk") and
             progress.get("next_tile_index") == 0 and
             all(key in progress for key in keys),
             "duration partial progress fields differ")
    _require(all(type(progress[key]) is int and progress[key] >= 0 for key in
                 keys if key != "row_loss_sum") and
             isinstance(progress["row_loss_sum"], (int, float)) and
             math.isfinite(progress["row_loss_sum"]),
             "duration partial progress values differ")
    _require(progress["next_person_chunk"] <= progress["total_person_chunks"],
             "duration partial cursor exceeds chunk count")


def _save_duration_budget(store: PartialUpdateStore,
                          progress: dict[str, object]) -> dict[str, object]:
    """Write only the duration-specific durable-work receipt."""
    _validate_duration_progress(progress)
    _require(progress["next_person_chunk"] == progress["total_person_chunks"],
             "duration budget requires all person chunks durable")
    value = {"schema": BUDGET_SCHEMA,
             "training_identity_sha256": store.identity["training_identity_sha256"],
             "update_identity_sha256": store.identity_sha256,
             "update_index": store.identity["update_index"],
             "progress": progress}
    _atomic_bytes(store.root / BUDGET_FILE, _canonical(value))
    return value


def load_duration_budgets(root: Path, training_identity_sha256: str,
                          completed_updates: int) -> dict[str, int]:
    _require(type(completed_updates) is int and completed_updates >= 0,
             "nonnegative completed duration updates required")
    rows: dict[int, dict[str, object]] = {}
    for path in Path(root).glob(f"update-*/{BUDGET_FILE}"):
        value = json.loads(path.read_bytes())
        index = value.get("update_index"); progress = value.get("progress")
        _require(value.get("schema") == BUDGET_SCHEMA and
                 value.get("training_identity_sha256") == training_identity_sha256 and
                 type(index) is int and 0 <= index < completed_updates and
                 index not in rows and isinstance(progress, dict),
                 "duration budget identity or index differs")
        _validate_duration_progress(progress)
        _require(progress["next_person_chunk"] == progress["total_person_chunks"],
                 "incomplete duration budget advertised as completed")
        rows[index] = progress
    _require(set(rows) == set(range(completed_updates)),
             "duration budget ledger omits a completed optimizer update")
    return {"optimizer_updates_completed": completed_updates,
            "person_chunks_durable": sum(row["total_person_chunks"] for row in rows.values()),
            "rows_durable": sum(row["rows_processed"] for row in rows.values()),
            "observed_labels_durable": sum(row["observed_labels_processed"]
                                            for row in rows.values()),
            "attempts_durable_lower_bound": sum(row["attempts_durable_lower_bound"]
                                                 for row in rows.values())}


@dataclass(frozen=True)
class DurationTrainingCallResult:
    status: str
    completed_updates: int
    history: tuple[float, ...]
    next_person_chunk: int
    budgets: dict[str, int]


ChunkProvider = Callable[[DurationTrainingState, int], DurationTrainingChunk]


def run_duration_training_call(
        state: DurationTrainingState, optimizer: torch.optim.Optimizer,
        optimizer_specification: OptimizerSpecification, checkpoint_root: Path,
        training_identity: dict[str, object], plan: DurationUpdatePlan,
        chunk_provider: ChunkProvider, *, max_chunks_this_call: int | None
        ) -> DurationTrainingCallResult:
    """Run or resume one optimizer update over complete person/chromosome rows.

    ``max_chunks_this_call`` is a technical interruption budget.  ``None`` must
    be passed explicitly for no artificial interruption.  The provider is
    called only for the durable cursor and later chunks; prior neural emissions
    are neither downloaded nor reconstructed after resume.
    """
    _require(isinstance(state, DurationTrainingState) and
             isinstance(optimizer, torch.optim.Optimizer),
             "duration state and optimizer required")
    _validate_duration_optimizer(optimizer, optimizer_specification)
    from r02_lai_duration_fit import FIT_SCOPE, validate_fit_plan, validate_fit_batch
    data_declaration = training_identity.get("data_declaration")
    _require(training_identity.get("schema") == TRAINING_SCHEMA and
             isinstance(data_declaration, dict) and
             data_declaration.get("scope") in (SYNTHETIC_SCOPE, FIT_SCOPE) and
             training_identity.get("update_plan") == plan.receipt() and
             training_identity.get("optimizer") == optimizer_specification.receipt() and
             training_identity.get("duration_parameters") == state.duration.receipt(),
             "duration training identity differs from state, optimizer, or plan")
    fit_rows = training_identity.get("fit_rows")
    if data_declaration["scope"] == FIT_SCOPE:
        validate_fit_plan(fit_rows, plan)
        _require(data_declaration["sha256"] == identity_digest(fit_rows),
                 "FIT declaration changed after identity creation")
    else:
        _require(fit_rows is None, "fixture identity cannot carry FIT rows")
    _require(max_chunks_this_call is None or
             (type(max_chunks_this_call) is int and max_chunks_this_call > 0),
             "max chunks this call must be positive or explicitly None")
    root = Path(checkpoint_root)
    completed_store = CheckpointStore(root / "completed", training_identity)
    partial_store = None
    try:
        completed, history = completed_store.restore(state, optimizer)
        if fit_rows is not None:
            _require(completed < fit_rows["max_optimizer_updates"],
                     "cumulative FIT optimizer-update budget is exhausted")
        identity_sha = identity_digest(training_identity)
        previous_budgets = load_duration_budgets(root, identity_sha, completed)
        partial_identity = _partial_identity(training_identity, completed, history,
                                             state, optimizer)
        partial_store = PartialUpdateStore(
            root / f"update-{completed:012d}", partial_identity)
        restored = partial_store.restore(state, optimizer)
        if restored is None:
            optimizer.zero_grad(set_to_none=True)
            progress = {"phase": "DURATION_PERSON_CHUNKS",
                        "next_person_batch_index": 0,
                        "next_tile_index": 0,
                        "next_person_chunk": 0,
                        "total_person_chunks": len(plan.chunk_identity_sha256),
                        "rows_processed": 0, "observed_labels_processed": 0,
                        "row_loss_sum": 0.0, "attempts_durable_lower_bound": 0}
        else:
            progress, partial_state = restored
            _require(isinstance(partial_state, dict) and
                     set(partial_state) == {"parameter_gradients"},
                     "duration partial state fields differ")
            restore_parameter_gradients(state, partial_state["parameter_gradients"])
        _validate_duration_progress(progress)
        _require(progress["total_person_chunks"] == len(plan.chunk_identity_sha256),
                 "duration partial chunk count differs from plan")
        progress = dict(progress)
        progress["attempts_durable_lower_bound"] += 1
        processed_this_call = 0
        while progress["next_person_chunk"] < progress["total_person_chunks"]:
            if (max_chunks_this_call is not None and
                    processed_this_call == max_chunks_this_call):
                budgets = dict(previous_budgets)
                budgets.update({"current_update_person_chunks_durable":
                                progress["next_person_chunk"],
                                "current_update_attempts_durable_lower_bound":
                                progress["attempts_durable_lower_bound"]})
                return DurationTrainingCallResult(
                    "PARTIAL", completed, tuple(history),
                    progress["next_person_chunk"], budgets)
            index = progress["next_person_chunk"]
            chunk = chunk_provider(state, index)
            _require(isinstance(chunk, DurationTrainingChunk) and
                     chunk.identity_sha256 == plan.chunk_identity_sha256[index] and
                     isinstance(chunk.batch, DurationSequenceBatch),
                     "duration provider returned a different authenticated chunk")
            batch = chunk.batch
            if fit_rows is not None:
                validate_fit_batch(fit_rows, index, batch)
            rows = len(batch.roles)
            _require(rows > 0 and set(batch.roles) == {"FIT"},
                     "duration training admits complete FIT rows only")
            prior = state.duration.joint_start_prior(rows)
            raw_loss = duration_unordered_label_loss(
                batch.neural_logits, batch.labels, batch.label_mask,
                batch.query_cm, batch.query_chromosome, batch.query_valid,
                state.duration.log_stage_rates, state.duration.switch_logits,
                prior, batch.sequence_declarations,
                state.duration.core_declaration(),
                duration_stages=state.duration.duration_stages, reduction="sum")
            _require(raw_loss.ndim == 0 and bool(torch.isfinite(raw_loss)) and
                     raw_loss.requires_grad, "duration chunk loss must be finite and differentiable")
            (raw_loss / plan.global_observed_labels).backward()
            _require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                         for parameter in state.parameters()),
                     "nonfinite duration gradient")
            labels = int(batch.label_mask.sum().item())
            progress.update(
                next_person_batch_index=index + 1,
                next_person_chunk=index + 1,
                rows_processed=progress["rows_processed"] + rows,
                observed_labels_processed=progress["observed_labels_processed"] + labels,
                row_loss_sum=progress["row_loss_sum"] + float(raw_loss.detach()))
            partial_store.save(
                state, optimizer, progress,
                {"parameter_gradients": capture_parameter_gradients(state)})
            processed_this_call += 1
            del raw_loss, prior, batch, chunk

        _require(progress["rows_processed"] == plan.global_rows and
                 progress["observed_labels_processed"] == plan.global_observed_labels,
                 "duration global loss denominator or row count differs from observed chunks")
        if optimizer_specification.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(state.parameters(),
                                           optimizer_specification.gradient_clip_norm,
                                           error_if_nonfinite=True)
        objective = progress["row_loss_sum"] / plan.global_observed_labels
        _save_duration_budget(partial_store, progress)
        optimizer.step()
        history.append(float(objective))
        completed_store.save(state, optimizer, completed + 1, history)
        partial_store.clear()
        completed += 1
        budgets = load_duration_budgets(root, identity_sha, completed)
        return DurationTrainingCallResult(
            "COMPLETE", completed, tuple(history), len(plan.chunk_identity_sha256), budgets)
    finally:
        if partial_store is not None:
            partial_store.close()
        completed_store.close()
