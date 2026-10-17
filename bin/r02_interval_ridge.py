#!/usr/bin/env python3
"""Matrix-free robust interval Ridge for the reviewed K04 FIT objective.

The caller owns the FIT-only feature construction and supplies a scipy
LinearOperator for the already weighted-centered design Z.  This module does
not choose a penalty, feature schema, tolerance, or resource limit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tempfile
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, name, length=None):
    out = np.asarray(value, dtype=np.float64)
    _require(out.ndim == 1 and (length is None or len(out) == length) and
             len(out) > 0 and np.isfinite(out).all(), f"Invalid finite {name}")
    return out


def _array_sha256(value):
    array = np.ascontiguousarray(value, dtype='<f8')
    digest = hashlib.sha256(json.dumps(list(array.shape)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _soft_threshold(value, threshold):
    return np.sign(value)*np.maximum(np.abs(value)-threshold, 0.)


def interval_prox(value, step, normalized_weights, half_width):
    """Prox of step*sum a_i*(r_i^2+2*h_i*abs(r_i)+h_i^2)."""
    value = _vector(value, 'proximal value')
    a = _vector(normalized_weights, 'normalized weights', len(value))
    h = _vector(half_width, 'half widths', len(value))
    _require(math.isfinite(step) and step > 0 and np.all(a > 0) and
             np.all(h >= 0), 'Positive step/weights and nonnegative widths required')
    return _soft_threshold(value, 2.*step*a*h)/(1.+2.*step*a)


def _matvec(operator, value, expected, name):
    out = np.asarray(operator.matvec(value), dtype=np.float64)
    _require(out.shape == (expected,) and np.isfinite(out).all(),
             f"Invalid {name} matvec result")
    return out


def _rmatvec(operator, value, expected, name):
    out = np.asarray(operator.rmatvec(value), dtype=np.float64)
    _require(out.shape == (expected,) and np.isfinite(out).all(),
             f"Invalid {name} rmatvec result")
    return out


def _problem_fingerprint(*, lower, upper, weights, design_identity, shape,
                         penalty, augmented_operator_norm_bound, step_fraction):
    value = dict(schema='r02_interval_ridge_problem_v1',
        design_identity=design_identity, shape=list(shape), penalty=penalty,
        augmented_operator_norm_bound=augmented_operator_norm_bound,
        step_fraction=step_fraction, lower_sha256=_array_sha256(lower),
        upper_sha256=_array_sha256(upper),
        normalized_weights_sha256=_array_sha256(weights))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _save_checkpoint(path, state, metadata):
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = np.frombuffer(json.dumps(metadata, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode(), dtype=np.uint8)
    fd, temporary = tempfile.mkstemp(prefix=target.name+'.', suffix='.tmp',
                                     dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, metadata_utf8=payload, **state)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_checkpoint(path, fingerprint, n_rows, n_columns):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved['metadata_utf8'].tobytes().decode())
        _require(metadata.get('schema') == 'r02_interval_ridge_checkpoint_v1' and
                 metadata.get('problem_fingerprint') == fingerprint,
                 'Checkpoint does not belong to this exact interval-Ridge problem')
        state = {name: np.asarray(saved[name], dtype=np.float64).copy()
                 for name in ('beta', 'beta_previous', 'dual')}
        state['intercept'] = float(saved['intercept'])
        state['intercept_previous'] = float(saved['intercept_previous'])
        state['iteration'] = int(saved['iteration'])
        state['elapsed_seconds'] = float(saved['elapsed_seconds'])
    _require(state['beta'].shape == state['beta_previous'].shape == (n_columns,) and
             state['dual'].shape == (n_rows,) and state['iteration'] >= 0 and
             state['elapsed_seconds'] >= 0 and
             all(np.isfinite(x).all() for x in (state['beta'],
                                                state['beta_previous'], state['dual'])) and
             math.isfinite(state['intercept']) and
             math.isfinite(state['intercept_previous']), 'Invalid checkpoint state')
    return state


def _feasible_dual(dual):
    out = dual-np.sum(dual, dtype=np.float64)/len(dual)
    # Remove the residual of the floating-point projection from one coordinate.
    out[-1] -= np.sum(out, dtype=np.float64)
    return out


def _diagnostics(operator, lower, upper, a, penalty, intercept, beta, dual, curvature_lower):
    n, p = operator.shape
    midpoint = (lower+upper)/2.
    half = (upper-lower)/2.
    prediction = intercept+_matvec(operator, beta, n, 'design')
    residual = prediction-midpoint
    primal_terms = a*(residual*residual+2.*half*np.abs(residual)+half*half)
    primal = math.fsum(map(float, primal_terms))+penalty*float(beta @ beta)
    feasible = _feasible_dual(dual)
    zts = _rmatvec(operator, feasible, p, 'design adjoint')
    excess = np.maximum(np.abs(feasible)-2.*a*half, 0.)
    conjugate = excess*excess/(4.*a)-a*half*half
    dual_objective = (-math.fsum(map(float, conjugate))-float(midpoint @ feasible)
                      -float(zts @ zts)/(4.*penalty))
    gap = primal-dual_objective
    raw_feasibility = abs(float(np.sum(dual, dtype=np.float64)))
    feasible_residual = abs(float(np.sum(feasible, dtype=np.float64)))
    return dict(primal_objective=primal, feasible_dual_objective=dual_objective,
                primal_dual_gap=gap, raw_dual_sum_residual=raw_feasibility,
                feasible_dual_sum_residual=feasible_residual,
                dual_projection_l2=float(np.linalg.norm(dual-feasible)),
                coefficient_error_bound_exact_arithmetic=(math.sqrt(gap/curvature_lower)
                    if gap >= 0 and curvature_lower > 0 else None),
                strong_curvature_lower=curvature_lower,
                error_bound_scope="Conditional on FP64 gap and supplied design; unavailable for negative gap or nonpositive curvature",
                prediction_min=float(np.min(prediction)),
                prediction_max=float(np.max(prediction)))


@dataclass(frozen=True)
class IntervalRidgeResult:
    status: str
    intercept: float
    coefficients: np.ndarray
    feasible_dual: np.ndarray
    iterations: int
    elapsed_seconds: float
    diagnostics: dict
    provenance: dict


def solve_interval_ridge(operator, lower, upper, weights, *, penalty,
                          design_identity, augmented_operator_norm_bound,
                          step_fraction, max_iterations, max_seconds,
                          check_every, gap_absolute_tolerance,
                          gap_relative_tolerance, dual_feasibility_tolerance,
                          centering_tolerance, checkpoint_every, checkpoint_path,
                          resume):
    """Solve the reviewed K04 robust interval objective with matrix-free PDHG.

    ``operator`` is the FIT-learned, a-weighted-centered Z.  The explicit norm
    bound must safely upper-bound ``[ones, Z]`` in the ordinary Euclidean norm.
    A returned CONVERGED status requires a feasible-dual gap test; resource caps
    always return an INCOMPLETE status and retain an optional numeric checkpoint.
    """
    _require(isinstance(operator, LinearOperator) and len(operator.shape) == 2,
             'A scipy LinearOperator design is required')
    n, p = operator.shape
    _require(n > 0 and p >= 0, 'Nonempty interval rows and valid design required')
    lower = _vector(lower, 'lower endpoints', n)
    upper = _vector(upper, 'upper endpoints', n)
    raw_weights = _vector(weights, 'weights', n)
    _require(np.all(lower <= upper) and np.all(lower >= 0) and np.all(upper <= 1),
             'Intervals must obey 0 <= lower <= upper <= 1')
    _require(np.all(raw_weights > 0), 'Every explicit FIT weight must be positive')
    weight_sum = math.fsum(map(float, raw_weights))
    _require(math.isfinite(weight_sum) and weight_sum > 0,
             'Finite positive FIT weight sum required')
    a = raw_weights/weight_sum
    _require(np.isfinite(a).all() and np.all(a > 0) and
             abs(math.fsum(map(float, a))-1.) <= 32*np.finfo(float).eps,
             'Weights cannot be normalized safely in FP64')
    _require(math.isfinite(penalty) and penalty > 0, 'Positive finite penalty required')
    _require(isinstance(design_identity, str) and len(design_identity) == 64 and
             all(c in '0123456789abcdef' for c in design_identity),
             'Authenticated 64-hex design identity required')
    for value, name in ((augmented_operator_norm_bound, 'operator norm bound'),
                        (max_seconds, 'time cap'),
                        (gap_absolute_tolerance, 'absolute gap tolerance'),
                        (gap_relative_tolerance, 'relative gap tolerance'),
                        (dual_feasibility_tolerance, 'dual feasibility tolerance'),
                        (centering_tolerance, 'centering tolerance')):
        _require(math.isfinite(value) and value > 0, f'Explicit positive {name} required')
    _require(math.isfinite(step_fraction) and 0 < step_fraction < 1,
             'Explicit primal-dual step fraction must be strictly between zero and one')
    _require(type(max_iterations) is int and max_iterations > 0 and
             type(check_every) is int and check_every > 0 and
             type(checkpoint_every) is int and checkpoint_every > 0 and
             isinstance(resume, bool), 'Explicit positive iteration limits required')
    _require(check_every <= max_iterations and checkpoint_every <= max_iterations,
             'Diagnostic/checkpoint frequencies cannot exceed iteration cap')
    centered = _rmatvec(operator, a, p, 'weighted-centering check')
    center_scale = max(1., float(np.linalg.norm(a)))
    _require(float(np.linalg.norm(centered)) <= centering_tolerance*center_scale,
             'Design is not centered with the normalized FIT weights')
    # A nonzero accepted centering residual weakens the joint curvature bound.
    curvature_lower = min(1., penalty)-float(np.linalg.norm(centered))
    # tau*sigma*||[1,Z]||^2 = step_fraction < 1 is the PDHG condition.
    step = math.sqrt(step_fraction)/augmented_operator_norm_bound
    _require(math.isfinite(step) and step > 0,
             'Operator norm bound yields an invalid primal-dual step')
    fingerprint = _problem_fingerprint(lower=lower, upper=upper, weights=a,
        design_identity=design_identity, shape=operator.shape, penalty=float(penalty),
        augmented_operator_norm_bound=float(augmented_operator_norm_bound),
        step_fraction=float(step_fraction))
    checkpoint = None if checkpoint_path is None else pathlib.Path(checkpoint_path)
    _require(not resume or checkpoint is not None and checkpoint.is_file(),
             'Resume requires an existing numeric checkpoint')
    if resume:
        state = _load_checkpoint(checkpoint, fingerprint, n, p)
        beta, beta_previous, dual = state['beta'], state['beta_previous'], state['dual']
        intercept, intercept_previous = state['intercept'], state['intercept_previous']
        iteration, prior_elapsed = state['iteration'], state['elapsed_seconds']
        _require(iteration < max_iterations and prior_elapsed < max_seconds,
                 'Amended limits must exceed the saved incomplete state')
    else:
        beta = np.zeros(p, dtype=np.float64)
        beta_previous = beta.copy()
        midpoint = (lower+upper)/2.
        intercept = float(a @ midpoint)
        intercept_previous = intercept
        dual = np.zeros(n, dtype=np.float64)
        iteration, prior_elapsed = 0, 0.
    started = time.monotonic()
    diagnostics = _diagnostics(operator, lower, upper, a, penalty,
                               intercept, beta, dual, curvature_lower)
    status = 'INCOMPLETE_ITERATION_CAP'

    def elapsed():
        return prior_elapsed+time.monotonic()-started

    def save():
        if checkpoint is None:
            return
        _save_checkpoint(checkpoint, dict(beta=beta, beta_previous=beta_previous,
            dual=dual, intercept=np.asarray(intercept),
            intercept_previous=np.asarray(intercept_previous),
            iteration=np.asarray(iteration, dtype=np.int64),
            elapsed_seconds=np.asarray(elapsed())),
            dict(schema='r02_interval_ridge_checkpoint_v1',
                 problem_fingerprint=fingerprint, design_identity=design_identity,
                 status='NUMERIC_STATE_NOT_SCIENTIFIC_RESULT'))

    while iteration < max_iterations:
        if elapsed() >= max_seconds:
            status = 'INCOMPLETE_TIME_CAP'
            break
        extrapolated_beta = 2.*beta-beta_previous
        extrapolated_intercept = 2.*intercept-intercept_previous
        affine = extrapolated_intercept+_matvec(operator, extrapolated_beta, n, 'design')
        candidate = dual+step*(affine-(lower+upper)/2.)
        # Moreau: prox_{sigma F*}(y) = y-sigma*prox_{F/sigma}(y/sigma).
        prox = interval_prox(candidate/step, 1./step, a, (upper-lower)/2.)
        dual_new = candidate-step*prox
        adjoint = _rmatvec(operator, dual_new, p, 'design adjoint')
        old_beta, old_intercept = beta, intercept
        beta = (beta-step*adjoint)/(1.+2.*step*penalty)
        intercept = intercept-step*float(np.sum(dual_new, dtype=np.float64))
        beta_previous, intercept_previous, dual = old_beta, old_intercept, dual_new
        iteration += 1
        _require(np.isfinite(beta).all() and np.isfinite(dual).all() and
                 math.isfinite(intercept), 'Nonfinite primal-dual iterate')
        if iteration % checkpoint_every == 0:
            save()
        if iteration % check_every == 0 or iteration == max_iterations:
            diagnostics = _diagnostics(operator, lower, upper, a, penalty,
                                       intercept, beta, dual, curvature_lower)
            scale = max(abs(diagnostics['primal_objective']),
                        abs(diagnostics['feasible_dual_objective']), 1.)
            gap_limit = gap_absolute_tolerance+gap_relative_tolerance*scale
            gap = diagnostics['primal_dual_gap']
            if (gap >= -gap_absolute_tolerance and gap <= gap_limit and
                    diagnostics['feasible_dual_sum_residual'] <= dual_feasibility_tolerance):
                status = 'CONVERGED_DUALITY_GAP'
                break
    # Recompute for the exact returned state.  A time cap may be observed at an
    # iteration boundary between scheduled diagnostics.
    diagnostics = _diagnostics(operator, lower, upper, a, penalty,
                               intercept, beta, dual, curvature_lower)
    save()
    feasible = _feasible_dual(dual)
    provenance = dict(schema='r02_interval_ridge_result_v1',
        objective='sum a*max((prediction-lower)^2,(prediction-upper)^2) + penalty*||beta||^2',
        role='FIT', design_identity=design_identity, problem_fingerprint=fingerprint,
        n_rows=n, n_columns=p, raw_weight_sum=weight_sum,
        normalized_weights_sha256=_array_sha256(a), penalty=float(penalty),
        augmented_operator_norm_bound=float(augmented_operator_norm_bound),
        operator_norm_bound_scope='caller-supplied safe bound; not certified by this solver',
        step_fraction=float(step_fraction), primal_dual_step=step,
        numerical_scope='FP64 diagnostics; dual is feasible to reported residual, not outward-rounded',
        matrix_scope='LinearOperator products plus O(n_rows+n_columns) numeric vectors; no dense Gram',
        prediction_scope='Unclipped FIT optimizer; caller applies reviewed [0,1] clip in selection/evaluation')
    return IntervalRidgeResult(status=status, intercept=float(intercept),
        coefficients=beta.copy(), feasible_dual=feasible, iterations=iteration,
        elapsed_seconds=elapsed(), diagnostics=diagnostics, provenance=provenance)
