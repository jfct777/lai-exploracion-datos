"""Explicit adapter from neural LAI potentials to the duration CRF reference.

This module does not select duration stages, rates, ancestry switches or a
population prior.  Each row is one declared scientific sequence unit: either a
chromosome or a predefined interval.  Technical fragments are rejected because
exact smoothed marginals would require both forward and backward boundary
messages, which this first adapter does not expose.

The six neural values are conditional-random-field potentials.  FLARE may have
conditioned the network that produced them, but this adapter does not accept or
multiply a second FLARE probability.  Marginalization covers ordered ancestry
orientation and finite duration stages only; it does not cover genotype phase
or reference-haplotype assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor

from r02_lai_diploid_duration import (
    ANCESTRIES,
    DEFAULT_ADJOINT_BLOCK_POSITIONS,
    UNORDERED_STATES,
    forward_partition,
    phase_type_generator,
    unordered_label_loss,
)


COMPLETE_SCIENTIFIC_UNIT = "complete_scientific_unit"
UNIT_KINDS = frozenset(("chromosome", "predefined_interval"))
MISSING_EMISSION_REJECT = "REJECT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class SequenceDeclaration:
    """Caller declaration of one complete scientific sequence unit.

    The strings record provenance for later authentication; this module only
    checks that they are present and that tensor geometry matches the declared
    positions.  It does not establish that the source or prior is biologically
    valid.
    """

    unit_kind: str
    unit_id: str
    chromosome: int
    expected_query_positions_cm: tuple[float, ...]
    coverage: str
    start_prior_provenance: str

    def __post_init__(self) -> None:
        _require(self.unit_kind in UNIT_KINDS, "unknown scientific sequence unit kind")
        _require(isinstance(self.unit_id, str) and bool(self.unit_id.strip()),
                 "sequence unit_id must be declared")
        _require(type(self.chromosome) is int and 1 <= self.chromosome <= 22,
                 "sequence chromosome must be an autosome")
        positions = self.expected_query_positions_cm
        _require(isinstance(positions, tuple) and len(positions) > 0,
                 "expected query positions must be a nonempty tuple")
        _require(all(type(value) in (int, float) and math.isfinite(value) for value in positions),
                 "expected query positions must be finite cM values")
        _require(all(right > left for left, right in zip(positions, positions[1:])),
                 "expected query positions must be strictly increasing")
        _require(isinstance(self.coverage, str) and bool(self.coverage.strip()),
                 "sequence coverage must be declared")
        _require(isinstance(self.start_prior_provenance, str)
                 and bool(self.start_prior_provenance.strip()),
                 "start-prior provenance must be declared")


@dataclass(frozen=True)
class DurationParameterDeclaration:
    """Traceability declarations, not scientific validation of parameter values."""

    duration_stages_provenance: str
    stage_rates_provenance: str
    ancestry_switch_provenance: str

    def __post_init__(self) -> None:
        for name, value in (("duration stages", self.duration_stages_provenance),
                            ("stage rates", self.stage_rates_provenance),
                            ("ancestry switches", self.ancestry_switch_provenance)):
            _require(isinstance(value, str) and bool(value.strip()),
                     f"{name} provenance must be declared")


@dataclass(frozen=True)
class DurationBatchResult:
    unordered_marginals: Tensor
    log_partitions: Tensor


@dataclass(frozen=True)
class DeclaredSequenceAxis:
    """Authenticated query identities for one complete declared unit."""

    sequence: SequenceDeclaration
    query_global_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(isinstance(self.sequence, SequenceDeclaration),
                 "declared sequence axis requires a sequence declaration")
        indices = self.query_global_indices
        _require(isinstance(indices, tuple) and len(indices) > 0
                 and all(type(value) is int and value >= 0 for value in indices),
                 "declared query indices must be a nonempty tuple of nonnegative integers")
        _require(all(right > left for left, right in zip(indices, indices[1:])),
                 "declared query indices must be strictly increasing")
        _require(len(indices) == len(self.sequence.expected_query_positions_cm),
                 "declared query indices and cM axis differ")


@dataclass(frozen=True)
class DurationFragment:
    """Neural output plus narrow geometry/label metadata for one technical cut.

    ``emission_valid`` describes availability of neural potentials.  It is not
    label availability.  Labels outside ``label_mask`` and outside ``query_core``
    are ignored and cannot alter sequence geometry or loss.
    """

    neural_logits: Tensor
    query_global_index: Tensor
    query_core: Tensor
    emission_valid: Tensor
    query_cm: Tensor
    query_chromosome: Tensor
    labels: Tensor
    label_mask: Tensor
    roles: tuple[str, ...]
    person_ids: tuple[str, ...]
    component_ids: tuple[str, ...]


@dataclass(frozen=True)
class DurationSequenceBatch:
    """Complete rows assembled without severing their autograd graphs."""

    neural_logits: Tensor
    labels: Tensor
    label_mask: Tensor
    query_cm: Tensor
    query_chromosome: Tensor
    query_valid: Tensor
    query_global_index: Tensor
    roles: tuple[str, ...]
    person_ids: tuple[str, ...]
    component_ids: tuple[str, ...]
    sequence_declarations: tuple[SequenceDeclaration, ...]
    retained_fragment_logit_numel: int


def _validate_fragment(fragment: DurationFragment) -> tuple[int, int]:
    _require(isinstance(fragment, DurationFragment),
             "duration fragments must use the narrow DurationFragment API")
    logits = fragment.neural_logits
    _require(isinstance(logits, Tensor) and logits.dtype in (torch.float32, torch.float64)
             and logits.ndim == 3 and logits.shape[-1] == len(UNORDERED_STATES),
             "fragment neural_logits must be float32/float64 [rows,queries,6]")
    rows, queries, _ = logits.shape
    _require(rows > 0 and queries > 0, "duration fragment axes must be nonempty")
    _require(isinstance(fragment.query_global_index, Tensor)
             and fragment.query_global_index.dtype in (torch.int32, torch.int64)
             and tuple(fragment.query_global_index.shape) == (queries,),
             "fragment query_global_index must be integer [queries]")
    for name in ("query_core", "emission_valid", "label_mask"):
        value = getattr(fragment, name)
        _require(isinstance(value, Tensor) and value.dtype == torch.bool
                 and tuple(value.shape) == (rows, queries),
                 f"fragment {name} must be boolean [rows,queries]")
    _require(isinstance(fragment.query_cm, Tensor)
             and fragment.query_cm.dtype in (torch.float32, torch.float64)
             and tuple(fragment.query_cm.shape) == (rows, queries),
             "fragment query_cm must be floating [rows,queries]")
    _require(isinstance(fragment.query_chromosome, Tensor)
             and fragment.query_chromosome.dtype in (
                 torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
             and tuple(fragment.query_chromosome.shape) == (rows, queries),
             "fragment query_chromosome must be integer [rows,queries]")
    _require(isinstance(fragment.labels, Tensor) and fragment.labels.dtype == torch.long
             and tuple(fragment.labels.shape) == (rows, queries),
             "fragment labels must be int64 [rows,queries]")
    axes = (fragment.roles, fragment.person_ids, fragment.component_ids)
    _require(all(isinstance(axis, tuple) and len(axis) == rows
                 and all(isinstance(value, str) and bool(value.strip()) for value in axis)
                 for axis in axes), "fragment role/person/component axes differ")
    _require(len(set(fragment.person_ids)) == rows,
             "fragment person identifiers must be unique")
    return rows, queries


def assemble_duration_sequences(
        fragments: Sequence[DurationFragment], declared_axis: DeclaredSequenceAxis, *,
        missing_emission_policy: str, rows_per_chunk: int,
        max_retained_logit_numel: int) -> DurationSequenceBatch:
    """Join technical cores into complete sequence rows while preserving gradients.

    The caller must construct and backpropagate one row chunk at a time.  This
    function rejects more than ``rows_per_chunk`` rows; it does not split them.
    ``max_retained_logit_numel`` validates the total size of the fragment logits
    presented for assembly.  Because ``fragments`` is a ``Sequence``, its caller
    may already retain every graph before this function begins: this check does
    not bound producer retention or all neural activations.  A real memory bound
    still requires a caller that constructs, assembles, backpropagates and frees
    one admitted row chunk at a time.
    """
    _require(isinstance(declared_axis, DeclaredSequenceAxis),
             "complete declared sequence axis required")
    _require(missing_emission_policy == MISSING_EMISSION_REJECT,
             "FIT duration assembly only admits missing_emission_policy=REJECT")
    _require(type(rows_per_chunk) is int and rows_per_chunk > 0,
             "rows_per_chunk must be an explicit positive integer")
    _require(type(max_retained_logit_numel) is int and max_retained_logit_numel > 0,
             "max_retained_logit_numel must be an explicit positive integer")
    _require(isinstance(fragments, Sequence) and len(fragments) > 0,
             "at least one duration fragment is required")

    row_axis = None
    dtype = device = None
    retained_numel = 0
    pieces: list[list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]] = []
    for fragment in fragments:
        rows, _ = _validate_fragment(fragment)
        current_axis = (fragment.roles, fragment.person_ids, fragment.component_ids)
        if row_axis is None:
            row_axis = current_axis
            _require(rows <= rows_per_chunk,
                     "duration row chunk exceeds rows_per_chunk before assembly")
            _require(set(fragment.roles) == {"FIT"},
                     "duration assembly admits one FIT role only")
            dtype, device = fragment.neural_logits.dtype, fragment.neural_logits.device
            pieces = [[] for _ in range(rows)]
        else:
            _require(fragment.roles == row_axis[0],
                     "role axis changed between duration fragments")
            _require(fragment.person_ids == row_axis[1],
                     "person axis changed between duration fragments")
            _require(fragment.component_ids == row_axis[2],
                     "component axis changed between duration fragments")
            _require(fragment.neural_logits.dtype == dtype
                     and fragment.neural_logits.device == device,
                     "duration fragment logits changed dtype or device")
        retained_numel += fragment.neural_logits.numel()
        _require(retained_numel <= max_retained_logit_numel,
                 "duration fragment logits exceed max_retained_logit_numel")

        for row in range(rows):
            owned_cpu = fragment.query_core[row].detach().to(device="cpu")
            if not bool(owned_cpu.any()):
                continue
            owned = owned_cpu.to(device=device)
            emission_valid = fragment.emission_valid[row].detach().to(device="cpu")[owned_cpu]
            _require(bool(emission_valid.all()),
                     "missing neural potential under missing_emission_policy=REJECT")
            logits = fragment.neural_logits[row][owned]
            _require(bool(torch.isfinite(logits).all()),
                     "missing or nonfinite neural potential under missing_emission_policy=REJECT")
            indices = fragment.query_global_index.detach().to(device=device)[owned]
            cm = fragment.query_cm[row].detach().to(device=device, dtype=dtype)[owned]
            # Source storage uses int16 for autosome labels. Widen integers
            # exactly here; floating and boolean inputs were rejected above.
            chromosome = fragment.query_chromosome[row].detach().to(
                device=device, dtype=torch.long)[owned]
            mask = fragment.label_mask[row].detach().to(device=device)[owned]
            labels = fragment.labels[row].detach().to(device=device)[owned]
            _require(bool(((labels[mask] >= 0) &
                           (labels[mask] < len(UNORDERED_STATES))).all()),
                     "observed duration labels must be unordered states 0..5")
            labels = torch.where(mask, labels, torch.zeros_like(labels))
            pieces[row].append((indices, logits, cm, chromosome, labels, mask))

    assert row_axis is not None and dtype is not None and device is not None
    expected_indices = torch.tensor(declared_axis.query_global_indices,
                                    device=device, dtype=torch.long)
    expected_cm = torch.tensor(declared_axis.sequence.expected_query_positions_cm,
                               device=device, dtype=dtype)
    row_logits, row_cm, row_chromosome, row_labels, row_label_mask = [], [], [], [], []
    for row, row_pieces in enumerate(pieces):
        _require(bool(row_pieces), "duration row has no owned core queries")
        indices = torch.cat([value[0].to(torch.long) for value in row_pieces])
        _require(torch.unique(indices).numel() == indices.numel(),
                 "duplicate query core in duration fragments")
        order = torch.argsort(indices, stable=True)
        indices = indices[order]
        _require(torch.equal(indices, expected_indices),
                 "duration query core has a missing query or differs from declared axis")
        logits = torch.cat([value[1] for value in row_pieces])[order]
        cm = torch.cat([value[2] for value in row_pieces])[order]
        chromosome = torch.cat([value[3] for value in row_pieces])[order]
        labels = torch.cat([value[4] for value in row_pieces])[order]
        label_mask = torch.cat([value[5] for value in row_pieces])[order]
        _require(bool(torch.isfinite(cm).all())
                 and bool((cm[1:] > cm[:-1]).all()),
                 "assembled duration cM must be finite and strictly increasing")
        scale = torch.maximum(cm.abs(), expected_cm.abs()).clamp_min(1.)
        tolerance = 4 * torch.finfo(dtype).eps * scale
        _require(bool(((cm - expected_cm).abs() <= tolerance).all()),
                 "query_global_index maps to a different declared cM coordinate")
        _require(bool((chromosome == declared_axis.sequence.chromosome).all()),
                 "assembled duration row mixes chromosomes")
        row_logits.append(logits); row_cm.append(cm); row_chromosome.append(chromosome)
        row_labels.append(labels); row_label_mask.append(label_mask)

    logits = torch.stack(row_logits)
    cm = torch.stack(row_cm)
    chromosome = torch.stack(row_chromosome)
    labels = torch.stack(row_labels)
    label_mask = torch.stack(row_label_mask)
    valid = torch.ones(labels.shape, dtype=torch.bool, device=device)
    roles, person_ids, component_ids = row_axis
    return DurationSequenceBatch(
        neural_logits=logits, labels=labels, label_mask=label_mask,
        query_cm=cm, query_chromosome=chromosome, query_valid=valid,
        query_global_index=expected_indices, roles=roles, person_ids=person_ids,
        component_ids=component_ids,
        sequence_declarations=tuple(declared_axis.sequence for _ in range(len(roles))),
        retained_fragment_logit_numel=retained_numel)


def _validate_batch(neural_logits: Tensor, query_cm: Tensor, query_chromosome: Tensor,
                    query_valid: Tensor, joint_start_prior: Tensor,
                    sequence_declarations: Sequence[SequenceDeclaration],
                    parameter_declaration: DurationParameterDeclaration, *,
                    duration_stages: int) -> list[Tensor]:
    _require(type(duration_stages) is int and duration_stages >= 1,
             "duration_stages must be an explicit positive integer")
    _require(isinstance(neural_logits, Tensor) and neural_logits.dtype in (torch.float32, torch.float64)
             and neural_logits.ndim == 3 and neural_logits.shape[-1] == len(UNORDERED_STATES),
             "neural_logits must be float32/float64 [batch,queries,6]")
    batch, queries, _ = neural_logits.shape
    _require(batch > 0 and queries > 0, "nonempty batch and query axes required")
    _require(isinstance(query_cm, Tensor) and query_cm.dtype == neural_logits.dtype
             and tuple(query_cm.shape) == (batch, queries),
             "query_cm must match logits and use their floating dtype")
    _require(isinstance(query_chromosome, Tensor)
             and query_chromosome.dtype in (torch.int32, torch.int64)
             and tuple(query_chromosome.shape) == (batch, queries),
             "query_chromosome must be integer [batch,queries]")
    _require(isinstance(query_valid, Tensor) and query_valid.dtype == torch.bool
             and tuple(query_valid.shape) == (batch, queries),
             "query_valid must be boolean [batch,queries]")
    _require(all(value.device == neural_logits.device
                 for value in (query_cm, query_chromosome, query_valid, joint_start_prior)),
             "LAI duration tensors must share one device")
    n = len(ANCESTRIES) * duration_stages
    _require(isinstance(joint_start_prior, Tensor)
             and joint_start_prior.dtype == neural_logits.dtype
             and tuple(joint_start_prior.shape) == (batch, n, n),
             "joint_start_prior must have shape [batch,3D,3D] and match logits dtype")
    _require(isinstance(parameter_declaration, DurationParameterDeclaration),
             "duration parameter provenance declaration required")
    _require(len(sequence_declarations) == batch
             and all(isinstance(value, SequenceDeclaration) for value in sequence_declarations),
             "one sequence declaration per batch row is required")
    _require(bool(torch.isfinite(neural_logits[query_valid]).all()),
             "valid neural logits must be finite")
    invalid = ~query_valid
    _require(bool((~torch.isfinite(query_cm[invalid])).all()),
             "invalid entries must be unlocated padding with nonfinite cM")

    orders: list[Tensor] = []
    for row, declaration in enumerate(sequence_declarations):
        _require(declaration.coverage == COMPLETE_SCIENTIFIC_UNIT,
                 "technical fragments require unimplemented forward and backward boundary messages")
        ids = torch.where(query_valid[row])[0]
        _require(len(ids) > 0, "each sequence unit needs at least one valid query")
        chromosomes = query_chromosome[row, ids]
        _require(bool((chromosomes == declaration.chromosome).all()),
                 "query chromosomes differ from the sequence declaration")
        order = ids[torch.argsort(query_cm[row, ids], stable=True)]
        positions = query_cm[row, order]
        expected = positions.new_tensor(declaration.expected_query_positions_cm)
        _require(tuple(positions.shape) == tuple(expected.shape),
                 "query count differs from declared scientific-unit positions")
        scale = torch.maximum(positions.abs(), expected.abs()).clamp_min(1.)
        tolerance = 4 * torch.finfo(positions.dtype).eps * scale
        _require(bool(((positions - expected).abs() <= tolerance).all()),
                 "query positions differ from the declared complete scientific unit")
        orders.append(order)
    return orders


def duration_marginals(neural_logits: Tensor, query_cm: Tensor, query_chromosome: Tensor,
                       query_valid: Tensor, log_stage_rates: Tensor, switch_logits: Tensor,
                       joint_start_prior: Tensor,
                       sequence_declarations: Sequence[SequenceDeclaration],
                       parameter_declaration: DurationParameterDeclaration, *,
                       duration_stages: int) -> DurationBatchResult:
    """Return normalized six-state smoothed marginals for declared sequence units."""
    orders = _validate_batch(
        neural_logits, query_cm, query_chromosome, query_valid, joint_start_prior,
        sequence_declarations, parameter_declaration, duration_stages=duration_stages)
    _require(log_stage_rates.device == neural_logits.device
             and log_stage_rates.dtype == neural_logits.dtype
             and switch_logits.device == neural_logits.device
             and switch_logits.dtype == neural_logits.dtype,
             "duration parameters must share logits device and dtype")
    generator = phase_type_generator(
        log_stage_rates, switch_logits, duration_stages=duration_stages)
    output = torch.full_like(neural_logits, float("nan"))
    log_partitions = []
    rows = []
    for row, order in enumerate(orders):
        positions = query_cm[row, order]
        deltas_morgan = (positions[1:] - positions[:-1]) / 100.0
        result = forward_partition(
            neural_logits[row, order], deltas_morgan, generator, joint_start_prior[row],
            duration_stages=duration_stages, return_marginals=True)
        row_output = output[row].index_copy(0, order, result.unordered_posterior)
        rows.append(row_output)
        log_partitions.append(result.log_partition)
    marginals = torch.stack(rows)
    _require(bool(torch.isfinite(marginals[query_valid]).all()),
             "duration marginals must be finite at valid queries")
    tolerance = 128 * torch.finfo(marginals.dtype).eps
    _require(bool(torch.allclose(marginals.sum(-1)[query_valid],
                                 torch.ones_like(marginals.sum(-1)[query_valid]),
                                 atol=tolerance, rtol=0)),
             "duration marginals must sum to one")
    return DurationBatchResult(marginals, torch.stack(log_partitions))


def duration_unordered_label_loss(
        neural_logits: Tensor, labels: Tensor, label_mask: Tensor,
        query_cm: Tensor, query_chromosome: Tensor,
        query_valid: Tensor, log_stage_rates: Tensor, switch_logits: Tensor,
        joint_start_prior: Tensor, sequence_declarations: Sequence[SequenceDeclaration],
        parameter_declaration: DurationParameterDeclaration, *, duration_stages: int,
        reduction: str,
        adjoint_block_positions: int = DEFAULT_ADJOINT_BLOCK_POSITIONS) -> Tensor:
    """Compute an explicitly selected sequential CRF loss, separate from legacy LAI."""
    orders = _validate_batch(
        neural_logits, query_cm, query_chromosome, query_valid, joint_start_prior,
        sequence_declarations, parameter_declaration, duration_stages=duration_stages)
    _require(isinstance(labels, Tensor) and labels.dtype == torch.long
             and labels.shape == query_valid.shape and labels.device == neural_logits.device,
             "labels must be int64 [batch,queries] on the logits device")
    _require(isinstance(label_mask, Tensor) and label_mask.dtype == torch.bool
             and label_mask.shape == query_valid.shape and label_mask.device == neural_logits.device
             and not bool((label_mask & ~query_valid).any()),
             "label_mask must be boolean [batch,queries] within the complete query axis")
    _require(bool(label_mask.any(1).all()),
             "each duration row requires at least one observed label")
    _require(bool(((labels[label_mask] >= 0)
                   & (labels[label_mask] < len(UNORDERED_STATES))).all()),
             "observed labels must be unordered-state indices 0..5")
    _require(reduction in ("none", "sum", "mean"), "loss reduction must be none, sum or mean")
    _require(log_stage_rates.device == neural_logits.device
             and log_stage_rates.dtype == neural_logits.dtype
             and switch_logits.device == neural_logits.device
             and switch_logits.dtype == neural_logits.dtype,
             "duration parameters must share logits device and dtype")
    generator = phase_type_generator(
        log_stage_rates, switch_logits, duration_stages=duration_stages)
    masked_labels = torch.where(label_mask, labels, torch.full_like(labels, -1))
    losses = []
    for row, order in enumerate(orders):
        positions = query_cm[row, order]
        losses.append(unordered_label_loss(
            neural_logits[row, order], masked_labels[row, order],
            (positions[1:] - positions[:-1]) / 100.0,
            generator, joint_start_prior[row], duration_stages=duration_stages,
            adjoint_block_positions=adjoint_block_positions))
    stacked = torch.stack(losses)
    if reduction == "sum":
        return stacked.sum()
    if reduction == "mean":
        return stacked.mean()
    return stacked
