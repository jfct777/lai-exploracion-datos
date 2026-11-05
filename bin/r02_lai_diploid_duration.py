"""Exact small-state reference for diploid local-ancestry duration.

This module implements a finite phase-type adaptation of the duration proposal
in ``.claude/runs/r02-theory-architectures-20261031ff/PLAN.md``.  It is a
mathematical reference and is not a training entry point.  In particular, it
does not assign reference haplotypes, marginalize reference phase, read FLARE
probabilities, or choose biological duration parameters.

The two latent homologues have conditionally independent transitions, while
their initial distribution and their emission potential are joint.  Thus the
implementation does not assert that homologues are marginally independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


ANCESTRIES = ("AFR", "EUR", "NAM")
UNORDERED_STATES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
DEFAULT_ADJOINT_BLOCK_POSITIONS = 256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _floating_tensor(value: Tensor, name: str) -> None:
    _require(isinstance(value, Tensor) and value.dtype in (torch.float32, torch.float64),
             f"{name} must be a float32/float64 tensor")


def phase_type_generator(log_stage_rates: Tensor, switch_logits: Tensor,
                         *, duration_stages: int) -> Tensor:
    """Build the haploid generator over ``3 * duration_stages`` states.

    ``log_stage_rates[a, r]`` is the unconstrained log rate per Morgan for
    ancestry ``a`` and latent stage ``r``.  A state advances to the next stage;
    the last stage changes ancestry and enters stage zero.  ``switch_logits``
    parameterizes the destination among the other two ancestries.  Its diagonal
    is excluded rather than interpreted as an invisible ancestry change.

    The stages describe a phase-type duration distribution.  They are neither
    tract ages nor generations since admixture.  ``duration_stages`` is
    required deliberately: no biological value is selected here.
    """
    _floating_tensor(log_stage_rates, "log_stage_rates")
    _floating_tensor(switch_logits, "switch_logits")
    _require(type(duration_stages) is int and duration_stages >= 1,
             "duration_stages must be an explicit positive integer")
    _require(tuple(log_stage_rates.shape) == (len(ANCESTRIES), duration_stages),
             "log_stage_rates must have shape [3,duration_stages]")
    _require(tuple(switch_logits.shape) == (len(ANCESTRIES), len(ANCESTRIES)),
             "switch_logits must have shape [3,3]")
    _require(log_stage_rates.device == switch_logits.device
             and log_stage_rates.dtype == switch_logits.dtype,
             "rate and switch tensors must share device and dtype")
    _require(bool(torch.isfinite(log_stage_rates).all()), "log_stage_rates must be finite")
    _require(bool(torch.isfinite(switch_logits).all()), "switch_logits must be finite")

    rates = log_stage_rates.exp()
    _require(bool(torch.isfinite(rates).all()), "exponentiated stage rates must be finite")
    diagonal = torch.eye(len(ANCESTRIES), device=switch_logits.device, dtype=torch.bool)
    destinations = switch_logits.masked_fill(diagonal, -torch.inf).softmax(dim=-1)
    n = len(ANCESTRIES) * duration_stages
    off_diagonal = rates.new_zeros((n, n))
    for ancestry in range(len(ANCESTRIES)):
        base = ancestry * duration_stages
        for stage in range(duration_stages - 1):
            off_diagonal[base + stage, base + stage + 1] = rates[ancestry, stage]
        last = base + duration_stages - 1
        for destination in range(len(ANCESTRIES)):
            if destination != ancestry:
                off_diagonal[last, destination * duration_stages] = (
                    rates[ancestry, -1] * destinations[ancestry, destination]
                )
    return off_diagonal - torch.diag_embed(off_diagonal.sum(dim=-1))


def transition_matrices(generator: Tensor, deltas_morgan: Tensor,
                        *, duration_stages: int) -> Tensor:
    """Return ``exp(generator * delta)`` for piecewise-constant intervals."""
    _floating_tensor(generator, "generator")
    _floating_tensor(deltas_morgan, "deltas_morgan")
    _require(type(duration_stages) is int and duration_stages >= 1,
             "duration_stages must be an explicit positive integer")
    n = len(ANCESTRIES) * duration_stages
    _require(tuple(generator.shape) == (n, n), "generator shape differs from duration_stages")
    _require(deltas_morgan.ndim == 1, "deltas_morgan must be one-dimensional")
    _require(generator.device == deltas_morgan.device and generator.dtype == deltas_morgan.dtype,
             "generator and distances must share device and dtype")
    _require(bool(torch.isfinite(generator).all()), "generator must be finite")
    _require(bool(torch.isfinite(deltas_morgan).all()) and bool((deltas_morgan >= 0).all()),
             "genetic distances must be finite and nonnegative")
    tolerance = 128 * torch.finfo(generator.dtype).eps
    _require(bool(torch.allclose(generator.sum(-1), generator.new_zeros(n),
                                 atol=tolerance, rtol=0)),
             "generator rows must sum to zero")
    off_diagonal = generator - torch.diag_embed(generator.diagonal())
    _require(bool((off_diagonal >= -tolerance).all()),
             "generator off-diagonal entries must be nonnegative")
    if len(deltas_morgan) == 0:
        return generator.new_empty((0, n, n))
    # matrix_exp(0 * generator) is already exactly the identity.  Do not select
    # a constant identity with torch.where: that has the same value but erases
    # the mathematically valid one-sided derivative d exp(delta Q)/d delta = Q
    # at delta=0.
    return torch.matrix_exp(deltas_morgan[:, None, None] * generator[None])


def lift_unordered_potentials(log_potentials: Tensor, *, duration_stages: int,
                              labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Lift six unordered CRF potentials to symmetric ordered state matrices.

    Mixed-ancestry potentials are repeated in both orientations without a
    ``log(2)`` correction.  They are conditional random-field potentials, not
    normalized six-class probabilities.  When every input potential is zero,
    every ordered latent state therefore receives multiplicative potential one.

    ``labels`` contains unordered-state indices 0..5, with -1 for an unlabelled
    position.  A label constrains ancestry while marginalizing orientation and
    latent duration stage.
    """
    _floating_tensor(log_potentials, "log_potentials")
    _require(type(duration_stages) is int and duration_stages >= 1,
             "duration_stages must be an explicit positive integer")
    _require(log_potentials.ndim == 2 and log_potentials.shape[1] == len(UNORDERED_STATES),
             "log_potentials must have shape [positions,6]")
    _require(len(log_potentials) > 0 and bool(torch.isfinite(log_potentials).all()),
             "log_potentials must be nonempty and finite")
    if labels is not None:
        _require(isinstance(labels, Tensor) and labels.dtype == torch.long
                 and tuple(labels.shape) == (len(log_potentials),),
                 "labels must be int64 [positions]")
        _require(labels.device == log_potentials.device,
                 "labels and potentials must share device")
        _require(bool(((labels >= -1) & (labels < len(UNORDERED_STATES))).all()),
                 "labels must be -1 or an unordered-state index 0..5")

    n = len(ANCESTRIES) * duration_stages
    ancestry = torch.arange(n, device=log_potentials.device) // duration_stages
    class_index = torch.empty((len(ANCESTRIES), len(ANCESTRIES)),
                              device=log_potentials.device, dtype=torch.long)
    for index, (left, right) in enumerate(UNORDERED_STATES):
        class_index[left, right] = index
        class_index[right, left] = index
    state_classes = class_index[ancestry[:, None], ancestry[None, :]]
    # Removing one permitted maximum per position prevents overflow and is
    # restored in the log partition.  For a labelled position the permitted
    # class itself supplies the shift.  A larger forbidden potential must not
    # underflow the finite constrained partition before it is masked.
    unconstrained_shifts = log_potentials.max(dim=-1).values
    if labels is not None:
        labelled_shifts = log_potentials.gather(1, labels.clamp_min(0)[:, None]).squeeze(1)
        shifts = torch.where(labels == -1, unconstrained_shifts, labelled_shifts)
        allowed = ((labels[:, None, None] == -1)
                   | (state_classes[None] == labels[:, None, None]))
        state_potentials = log_potentials[:, state_classes]
        reduced = (state_potentials - shifts[:, None, None]).masked_fill(~allowed, -torch.inf)
        emissions = reduced.exp()
    else:
        shifts = unconstrained_shifts
        emissions = (log_potentials[:, state_classes] - shifts[:, None, None]).exp()
    return emissions, shifts


def _validate_start_prior(start_prior: Tensor, n: int, reference: Tensor) -> None:
    _floating_tensor(start_prior, "start_prior")
    _require(tuple(start_prior.shape) == (n, n), "start_prior must have shape [states,states]")
    _require(start_prior.device == reference.device and start_prior.dtype == reference.dtype,
             "start_prior and potentials must share device and dtype")
    _require(bool(torch.isfinite(start_prior).all()) and bool((start_prior >= 0).all()),
             "start_prior must be finite and nonnegative")
    tolerance = 128 * torch.finfo(start_prior.dtype).eps
    _require(bool(torch.allclose(start_prior, start_prior.T, atol=tolerance, rtol=0)),
             "start_prior must be symmetric under homologue exchange")
    _require(bool(torch.allclose(start_prior.sum(), start_prior.new_tensor(1.),
                                 atol=tolerance, rtol=0)),
             "start_prior must sum to one")


@dataclass(frozen=True)
class ForwardResult:
    log_partition: Tensor
    ordered_posterior: Tensor | None = None
    unordered_posterior: Tensor | None = None


def _scaled_forward(emissions: Tensor, shifts: Tensor, transitions: Tensor,
                    start_prior: Tensor, *, keep_states: bool) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    alpha = start_prior
    log_partition = shifts.new_zeros(())
    alphas: list[Tensor] = []
    scales: list[Tensor] = []
    for position in range(len(emissions)):
        if position:
            transition = transitions[position - 1]
            alpha = transition.T @ alpha @ transition
        alpha = alpha * emissions[position]
        scale = alpha.sum()
        _require(bool(torch.isfinite(scale)) and bool(scale > 0),
                 "no finite latent path supports these potentials/labels")
        alpha = alpha / scale
        log_partition = log_partition + shifts[position] + scale.log()
        if keep_states:
            alphas.append(alpha)
            scales.append(scale)
    return log_partition, alphas, scales


def _posterior(alphas: list[Tensor], scales: list[Tensor], emissions: Tensor,
               transitions: Tensor, *, duration_stages: int) -> tuple[Tensor, Tensor]:
    beta = torch.ones_like(alphas[-1])
    ordered = [torch.empty_like(beta) for _ in alphas]
    for position in range(len(alphas) - 1, -1, -1):
        marginal = alphas[position] * beta
        ordered[position] = marginal / marginal.sum()
        if position:
            transition = transitions[position - 1]
            beta = transition @ (emissions[position] * beta) @ transition.T
            beta = beta / scales[position]
    ordered_tensor = torch.stack(ordered)
    n = ordered_tensor.shape[-1]
    ancestry = torch.arange(n, device=ordered_tensor.device) // duration_stages
    unordered = []
    for left, right in UNORDERED_STATES:
        selected = ((ancestry[:, None] == left) & (ancestry[None, :] == right))
        if left != right:
            selected = selected | ((ancestry[:, None] == right) & (ancestry[None, :] == left))
        unordered.append(ordered_tensor[:, selected].sum(-1))
    return ordered_tensor, torch.stack(unordered, dim=-1)


def forward_partition(log_potentials: Tensor, deltas_morgan: Tensor, generator: Tensor,
                      start_prior: Tensor, *, duration_stages: int,
                      labels: Tensor | None = None, return_marginals: bool = False) -> ForwardResult:
    """Compute an exact discrete-observation forward partition for the CTMC.

    The first observation uses the supplied joint prior directly.  It may put
    mass on any latent stage, which represents left censoring but is not claimed
    to be the correct population prior automatically.  The final partition sums
    every state; no tract exit is forced at the right chromosome boundary.

    Dense propagation ``P.T @ alpha @ P`` costs O(T n^3) time and O(n^2)
    forward memory (or O(T n^2) when marginals/autograd intermediates are kept),
    with n=3D.  Matrix exponentials also have cubic cost.  This routine makes no
    O(T n^2) scalability claim.
    """
    emissions, shifts = lift_unordered_potentials(
        log_potentials, duration_stages=duration_stages, labels=labels)
    n = len(ANCESTRIES) * duration_stages
    _validate_start_prior(start_prior, n, log_potentials)
    _require(tuple(deltas_morgan.shape) == (len(log_potentials) - 1,),
             "deltas_morgan must contain one interval fewer than positions")
    transitions = transition_matrices(generator, deltas_morgan,
                                      duration_stages=duration_stages)
    return _partition_from_prepared(emissions, shifts, transitions, start_prior,
                                    duration_stages=duration_stages,
                                    return_marginals=return_marginals)


def _partition_from_prepared(emissions: Tensor, shifts: Tensor, transitions: Tensor,
                             start_prior: Tensor, *, duration_stages: int,
                             return_marginals: bool = False) -> ForwardResult:
    """Internal recurrence for validated emissions and same-call transitions."""
    n = len(ANCESTRIES) * duration_stages
    _require(tuple(transitions.shape) == (len(emissions) - 1, n, n)
             and tuple(emissions.shape[1:]) == (n, n)
             and tuple(shifts.shape) == (len(emissions),),
             "prepared transition/emission axes differ")
    log_partition, alphas, scales = _scaled_forward(
        emissions, shifts, transitions, start_prior, keep_states=return_marginals)
    if not return_marginals:
        return ForwardResult(log_partition)
    ordered, unordered = _posterior(alphas, scales, emissions, transitions,
                                    duration_stages=duration_stages)
    return ForwardResult(log_partition, ordered, unordered)


def unordered_label_loss_autograd_reference(
        log_potentials: Tensor, labels: Tensor, deltas_morgan: Tensor,
        generator: Tensor, start_prior: Tensor, *, duration_stages: int) -> Tensor:
    """Full-autograd reference negative conditional log likelihood.

    This marginalizes only ordered ancestry orientation and finite duration
    stages.  It does not demonstrate marginalization of genotype phase or of
    reference haplotype assignments, which must be supplied by a future emission
    model and independently verified.
    """
    emissions, shifts = lift_unordered_potentials(
        log_potentials, duration_stages=duration_stages)
    labelled_emissions, labelled_shifts = lift_unordered_potentials(
        log_potentials, duration_stages=duration_stages, labels=labels)
    _validate_start_prior(start_prior, len(ANCESTRIES) * duration_stages,
                          log_potentials)
    _require(tuple(deltas_morgan.shape) == (len(log_potentials) - 1,),
             "deltas_morgan must contain one interval fewer than positions")
    # Both partitions use exactly the same generator and observation axis.
    # Share this differentiable tensor only within this call, never across
    # optimizer updates. Autograd accumulates both partition contributions.
    transitions = transition_matrices(generator, deltas_morgan,
                                      duration_stages=duration_stages)
    denominator = _partition_from_prepared(
        emissions, shifts, transitions, start_prior,
        duration_stages=duration_stages).log_partition
    numerator = _partition_from_prepared(
        labelled_emissions, labelled_shifts, transitions, start_prior,
        duration_stages=duration_stages).log_partition
    return denominator - numerator


def _block_transition_slice(start: int, end: int) -> tuple[int, int]:
    """Return the half-open interval axis needed by positions [start,end)."""
    return (0 if start == 0 else start - 1), end - 1


def _forward_block(emissions: Tensor, shifts: Tensor, transitions: Tensor,
                   boundary_alpha: Tensor, *, global_start: int
                   ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    """Advance one computational block without resetting its left message."""
    alpha = boundary_alpha
    log_partition = shifts.new_zeros(())
    alphas: list[Tensor] = []
    scales: list[Tensor] = []
    for local_position in range(len(emissions)):
        global_position = global_start + local_position
        if global_position:
            transition_index = local_position if global_start else local_position - 1
            transition = transitions[transition_index]
            alpha = transition.T @ alpha @ transition
        alpha = alpha * emissions[local_position]
        scale = alpha.sum()
        _require(bool(torch.isfinite(scale)) and bool(scale > 0),
                 "no finite latent path supports these potentials/labels")
        alpha = alpha / scale
        log_partition = log_partition + shifts[local_position] + scale.log()
        alphas.append(alpha)
        scales.append(scale)
    return alpha, log_partition, alphas, scales


def _state_classes(reference: Tensor, duration_stages: int) -> Tensor:
    n = len(ANCESTRIES) * duration_stages
    ancestry = torch.arange(n, device=reference.device) // duration_stages
    class_index = torch.empty((len(ANCESTRIES), len(ANCESTRIES)),
                              device=reference.device, dtype=torch.long)
    for index, (left, right) in enumerate(UNORDERED_STATES):
        class_index[left, right] = index
        class_index[right, left] = index
    return class_index[ancestry[:, None], ancestry[None, :]]


def _class_posterior(ordered_posterior: Tensor, state_classes: Tensor) -> Tensor:
    return torch.stack(tuple(ordered_posterior[state_classes == index].sum()
                             for index in range(len(UNORDERED_STATES))))


def _boundary_invariant_tolerance(dtype: torch.dtype) -> float:
    """Fixed, nonvacuous tolerance for a scaled alpha-beta boundary."""
    _require(dtype in (torch.float32, torch.float64),
             "alpha-beta invariant requires float32 or float64")
    return 2e-4 if dtype == torch.float32 else 2e-10


def _normalize_beta(alpha: Tensor, beta: Tensor, *, tolerance: float,
                    name: str, check: bool) -> Tensor:
    """Restore the exact scaled-message invariant after checking a boundary."""
    invariant = (alpha * beta).sum()
    _require(bool(torch.isfinite(invariant)) and bool(invariant > 0),
             f"{name} alpha-beta invariant is nonfinite or nonpositive")
    if check:
        _require(bool(torch.isclose(invariant, invariant.new_tensor(1.),
                                   atol=tolerance, rtol=tolerance)),
                 f"{name} alpha-beta block-boundary invariant failed")
    return beta / invariant


class _ChunkedLabelLoss(torch.autograd.Function):
    """First-order exact adjoint with bounded latent-state workspace.

    Inputs and output gradients remain O(T).  Only the latent forward/backward
    workspace is checkpointed: O((B + ceil(T/B)) n^2).  Higher-order gradients
    are deliberately unsupported because this backward is a manual adjoint.
    """

    @staticmethod
    def forward(ctx, log_potentials: Tensor, labels: Tensor,
                deltas_morgan: Tensor, generator: Tensor, start_prior: Tensor,
                duration_stages: int, block_positions: int) -> Tensor:
        positions = len(log_potentials)
        unconstrained_alpha = start_prior
        constrained_alpha = start_prior
        unconstrained_log_partition = log_potentials.new_zeros(())
        constrained_log_partition = log_potentials.new_zeros(())
        unconstrained_boundaries = []
        constrained_boundaries = []
        with torch.no_grad():
            for start in range(0, positions, block_positions):
                end = min(start + block_positions, positions)
                interval_start, interval_end = _block_transition_slice(start, end)
                transitions = transition_matrices(
                    generator, deltas_morgan[interval_start:interval_end],
                    duration_stages=duration_stages)
                unconstrained_emissions, unconstrained_shifts = lift_unordered_potentials(
                    log_potentials[start:end], duration_stages=duration_stages)
                constrained_emissions, constrained_shifts = lift_unordered_potentials(
                    log_potentials[start:end], duration_stages=duration_stages,
                    labels=labels[start:end])
                unconstrained_boundaries.append(unconstrained_alpha)
                constrained_boundaries.append(constrained_alpha)
                unconstrained_alpha, value, _, _ = _forward_block(
                    unconstrained_emissions, unconstrained_shifts, transitions,
                    unconstrained_alpha, global_start=start)
                unconstrained_log_partition = unconstrained_log_partition + value
                constrained_alpha, value, _, _ = _forward_block(
                    constrained_emissions, constrained_shifts, transitions,
                    constrained_alpha, global_start=start)
                constrained_log_partition = constrained_log_partition + value
        ctx.duration_stages = duration_stages
        ctx.block_positions = block_positions
        ctx.save_for_backward(
            log_potentials, labels, deltas_morgan, generator, start_prior,
            torch.stack(unconstrained_boundaries),
            torch.stack(constrained_boundaries))
        return unconstrained_log_partition - constrained_log_partition

    @staticmethod
    def backward(ctx, output_gradient: Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "unordered_label_loss chunked adjoint supports first-order gradients only")
        (log_potentials, labels, deltas_morgan, generator, start_prior,
         unconstrained_boundaries, constrained_boundaries) = ctx.saved_tensors
        duration_stages = ctx.duration_stages
        block_positions = ctx.block_positions
        positions = len(log_potentials)
        n = len(ANCESTRIES) * duration_stages
        state_classes = _state_classes(log_potentials, duration_stages)
        potential_gradient = torch.zeros_like(log_potentials)
        generator_gradient = torch.zeros_like(generator)
        delta_gradient = torch.zeros_like(deltas_morgan)
        prior_gradient = torch.zeros_like(start_prior)
        unconstrained_beta = torch.ones((n, n), dtype=log_potentials.dtype,
                                         device=log_potentials.device)
        constrained_beta = torch.ones_like(unconstrained_beta)
        starts = tuple(range(0, positions, block_positions))
        tolerance = _boundary_invariant_tolerance(log_potentials.dtype)

        for block_index in range(len(starts) - 1, -1, -1):
            start = starts[block_index]
            end = min(start + block_positions, positions)
            interval_start, interval_end = _block_transition_slice(start, end)
            delta_values = deltas_morgan[interval_start:interval_end].detach()
            need_generator = ctx.needs_input_grad[3] and len(delta_values) > 0
            need_deltas = ctx.needs_input_grad[2] and len(delta_values) > 0
            if need_generator or need_deltas:
                with torch.enable_grad():
                    generator_leaf = generator.detach().requires_grad_(need_generator)
                    delta_leaf = delta_values.requires_grad_(need_deltas)
                    differentiable_transitions = transition_matrices(
                        generator_leaf, delta_leaf, duration_stages=duration_stages)
                transitions = differentiable_transitions.detach()
            else:
                with torch.no_grad():
                    transitions = transition_matrices(
                        generator, delta_values, duration_stages=duration_stages)
                differentiable_transitions = None

            with torch.no_grad():
                unconstrained_emissions, unconstrained_shifts = lift_unordered_potentials(
                    log_potentials[start:end], duration_stages=duration_stages)
                constrained_emissions, constrained_shifts = lift_unordered_potentials(
                    log_potentials[start:end], duration_stages=duration_stages,
                    labels=labels[start:end])
                _, _, unconstrained_alphas, unconstrained_scales = _forward_block(
                    unconstrained_emissions, unconstrained_shifts, transitions,
                    unconstrained_boundaries[block_index], global_start=start)
                _, _, constrained_alphas, constrained_scales = _forward_block(
                    constrained_emissions, constrained_shifts, transitions,
                    constrained_boundaries[block_index], global_start=start)
                transition_gradient = torch.zeros_like(transitions)

                for global_position in range(end - 1, start - 1, -1):
                    local_position = global_position - start
                    if global_position:
                        transition_index = (local_position if start
                                            else local_position - 1)
                        transition = transitions[transition_index]
                    else:
                        transition_index = -1

                    for sign, emissions, alphas, scales, beta_name in (
                            (1., unconstrained_emissions, unconstrained_alphas,
                             unconstrained_scales, "unconstrained"),
                            (-1., constrained_emissions, constrained_alphas,
                             constrained_scales, "constrained")):
                        beta = (unconstrained_beta if beta_name == "unconstrained"
                                else constrained_beta)
                        beta = _normalize_beta(
                            alphas[local_position], beta, tolerance=tolerance,
                            name=beta_name, check=(global_position == end - 1))
                        marginal = alphas[local_position] * beta
                        marginal_sum = marginal.sum()
                        _require(bool(torch.isfinite(marginal_sum)) and
                                 bool(marginal_sum > 0),
                                 "nonfinite forward-backward marginal")
                        ordered_posterior = marginal / marginal_sum
                        potential_gradient[global_position] += (
                            sign * _class_posterior(ordered_posterior, state_classes))
                        emission_beta = emissions[local_position] * beta
                        scale = scales[local_position]
                        if global_position:
                            previous_alpha = (alphas[local_position - 1]
                                              if local_position else
                                              (unconstrained_boundaries[block_index]
                                               if beta_name == "unconstrained" else
                                               constrained_boundaries[block_index]))
                            transition_gradient[transition_index] += sign * (
                                previous_alpha @ transition @ emission_beta.T
                                + previous_alpha.T @ transition @ emission_beta) / scale
                            beta = (transition @ emission_beta @ transition.T) / scale
                        else:
                            prior_gradient += sign * emission_beta / scale
                            # Message immediately to the left of the first
                            # emission.  It is needed only for the scaled
                            # alpha-beta invariant because no earlier block
                            # consumes it.
                            beta = emission_beta / scale
                        if beta_name == "unconstrained":
                            unconstrained_beta = beta
                        else:
                            constrained_beta = beta

                for boundary, beta, name in (
                        (unconstrained_boundaries[block_index], unconstrained_beta,
                         "unconstrained"),
                        (constrained_boundaries[block_index], constrained_beta,
                         "constrained")):
                    beta = _normalize_beta(
                        boundary, beta, tolerance=tolerance, name=name, check=True)
                    if name == "unconstrained":
                        unconstrained_beta = beta
                    else:
                        constrained_beta = beta

            if differentiable_transitions is not None:
                inputs = []
                if need_generator:
                    inputs.append(generator_leaf)
                if need_deltas:
                    inputs.append(delta_leaf)
                with torch.enable_grad():
                    gradients = torch.autograd.grad(
                        differentiable_transitions, tuple(inputs),
                        grad_outputs=transition_gradient, create_graph=False)
                gradient_index = 0
                if need_generator:
                    generator_gradient += gradients[gradient_index]
                    gradient_index += 1
                if need_deltas:
                    delta_gradient[interval_start:interval_end] = gradients[gradient_index]

        scale = output_gradient.to(dtype=log_potentials.dtype,
                                   device=log_potentials.device)
        return (
            potential_gradient * scale if ctx.needs_input_grad[0] else None,
            None,
            delta_gradient * scale if ctx.needs_input_grad[2] else None,
            generator_gradient * scale if ctx.needs_input_grad[3] else None,
            prior_gradient * scale if ctx.needs_input_grad[4] else None,
            None,
            None,
        )


def unordered_label_loss(log_potentials: Tensor, labels: Tensor, deltas_morgan: Tensor,
                         generator: Tensor, start_prior: Tensor, *, duration_stages: int,
                         adjoint_block_positions: int = DEFAULT_ADJOINT_BLOCK_POSITIONS
                         ) -> Tensor:
    """Exact first-order loss with blockwise forward-backward adjoints.

    Computational blocks carry alpha and beta messages across their boundaries;
    they never reset the model or remove a query position.  Use
    ``unordered_label_loss_autograd_reference`` for the small full-autograd
    oracle used by synthetic equivalence tests.
    """
    _floating_tensor(log_potentials, "log_potentials")
    _floating_tensor(deltas_morgan, "deltas_morgan")
    _floating_tensor(generator, "generator")
    _require(type(duration_stages) is int and duration_stages >= 1,
             "duration_stages must be an explicit positive integer")
    _require(type(adjoint_block_positions) is int and adjoint_block_positions > 0,
             "adjoint_block_positions must be a positive integer")
    _require(log_potentials.ndim == 2 and
             log_potentials.shape[1] == len(UNORDERED_STATES) and
             len(log_potentials) > 0 and bool(torch.isfinite(log_potentials).all()),
             "log_potentials must be finite [positions,6]")
    _require(isinstance(labels, Tensor) and labels.dtype == torch.long and
             tuple(labels.shape) == (len(log_potentials),) and
             labels.device == log_potentials.device and
             bool(((labels >= -1) & (labels < len(UNORDERED_STATES))).all()),
             "labels must be int64 -1 or 0..5 for each position")
    n = len(ANCESTRIES) * duration_stages
    _require(tuple(generator.shape) == (n, n),
             "generator shape differs from duration_stages")
    _require(tuple(deltas_morgan.shape) == (len(log_potentials) - 1,),
             "deltas_morgan must contain one interval fewer than positions")
    _require(generator.device == log_potentials.device == deltas_morgan.device and
             generator.dtype == log_potentials.dtype == deltas_morgan.dtype,
             "loss tensors must share device and floating dtype")
    _validate_start_prior(start_prior, n, log_potentials)
    _require(bool(torch.isfinite(deltas_morgan).all()) and
             bool((deltas_morgan >= 0).all()),
             "genetic distances must be finite and nonnegative")
    # A nonempty sequence with no interval still needs generator validation;
    # otherwise the custom forward validates it while building each block.
    if len(deltas_morgan) == 0:
        transition_matrices(generator, deltas_morgan,
                            duration_stages=duration_stages)
    return _ChunkedLabelLoss.apply(
        log_potentials, labels, deltas_morgan, generator, start_prior,
        duration_stages, adjoint_block_positions)
