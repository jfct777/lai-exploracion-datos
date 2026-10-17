#!/usr/bin/env python3
"""FIT-only Ridge scale statistics and a declared finite-family planner.

This module neither fits nor selects a model. It does not authorize a human
analysis. The caller authenticates the FIT design and supplies its identity.
No scientific choices have defaults: epsilon, spacing, guard width and resource
limits must be present in the reviewed specification.
"""
import hashlib
import json
import math
import time

import numpy as np

from r02_pair_operator import check_weights, require


def content_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def array_sha256(value):
    array = np.ascontiguousarray(value, dtype='<f8')
    digest = hashlib.sha256(json.dumps(list(array.shape)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def fit_gram_statistics(design, weights, *, role, input_identity,
                        max_seconds, preprocessing=None):
    """Accumulate diag(H), H=Z_centered' diag(w) Z_centered / sum(w).

    Uses the same FIT preprocessing and intercept centering as the solver. One
    extra pass over pair chunks computes squares directly; no p-by-p Gram or
    n_pairs-by-p matrix is stored. The design owns the chunk memory admission.
    FP64 statistics are diagnostics, not outward-rounded spectral certificates.
    No response, SELECT or EVALUATE values enter this function.
    """
    require(role == 'FIT', 'Statistics require FIT role')
    require(isinstance(input_identity, str) and len(input_identity) == 64 and
            all(c in '0123456789abcdef' for c in input_identity),
            'Authenticated FIT design identity required')
    require(math.isfinite(max_seconds) and max_seconds > 0,
            'Explicit positive statistics time cap required')
    started = time.monotonic()

    def check_time():
        require(time.monotonic()-started <= max_seconds,
                'FIT statistics time cap exceeded; no completed statistics')

    weights = check_weights(weights, design.shape[0])
    weight_sum = float(weights.sum())
    require(weight_sum > 0, 'Positive total FIT weight required')
    if preprocessing is None:
        preprocessing = design.fit_preprocessing(weights)
    check_time()
    center, scale = map(lambda a: np.asarray(a, dtype=np.float64), preprocessing)
    _, offset = design.ridge_operator(weights, center, scale)
    check_time()
    diagonal = np.zeros(design.shape[1], dtype=np.float64)
    compensation = np.zeros_like(diagonal)
    normalized_weight = weights/weight_sum
    rows_seen = 0
    all_entries_exact_zero = True
    for sl, values in design.chunks():
        check_time()
        require(sl.start == rows_seen, 'FIT chunks are not contiguous')
        values -= center
        values /= scale
        values -= offset
        all_entries_exact_zero = all_entries_exact_zero and not np.any(values)
        piece = np.einsum('ij,i,ij->j', values, normalized_weight[sl], values,
                          optimize=False)
        # Compensated accumulation across chunks reduces ordering sensitivity.
        corrected = piece-compensation
        total = diagonal+corrected
        compensation = (total-diagonal)-corrected
        diagonal = total
        rows_seen = sl.stop
    check_time()
    require(rows_seen == design.shape[0] and np.isfinite(diagonal).all() and
            np.all(diagonal >= 0), 'Invalid/incomplete FIT Gram diagonal')
    upper = math.fsum(map(float, diagonal))
    lower = float(np.max(diagonal))
    require(math.isfinite(upper), 'Nonfinite FIT trace')
    return dict(schema='r02_ridge_fit_scale_v1', role='FIT',
        status='FIT_SCALE_DIAGNOSTIC_NOT_MODEL_SELECTION',
        input_identity=input_identity, n_rows=int(design.shape[0]),
        n_columns=int(design.shape[1]), weight_sum=weight_sum,
        diagonal=diagonal.tolist(), spectral_lower=lower, spectral_upper=upper,
        all_standardized_entries_exact_zero=bool(all_entries_exact_zero),
        spectral_bounds='max(diag(H)) <= max_eigenvalue(H) <= trace(H)',
        numerical_scope='FP64 accumulation; not an outward-rounded certificate',
        feature_schema_preserved=True,
        preprocessing_sha256=dict(center=array_sha256(center), scale=array_sha256(scale),
                                  standardized_offset=array_sha256(offset)),
        weights_sha256=array_sha256(weights), backend=design.backend,
        rows_per_chunk=design.rows_per_chunk,
        resources=dict(max_seconds=float(max_seconds), elapsed_seconds=time.monotonic()-started,
                       extra_pass='one full pair-feature pass after frozen preprocessing',
                       additional_persistent_bytes=int(normalized_weight.nbytes+diagonal.nbytes+
                                                       compensation.nbytes+offset.nbytes)))


def _validated_statistics(stats):
    require(stats.get('schema') == 'r02_ridge_fit_scale_v1' and
            stats.get('role') == 'FIT' and
            stats.get('status') == 'FIT_SCALE_DIAGNOSTIC_NOT_MODEL_SELECTION',
            'Complete FIT scale statistics required')
    diagonal = np.asarray(stats.get('diagonal'), dtype=np.float64)
    require(type(stats.get('n_columns')) is int and stats['n_columns'] > 0 and
            diagonal.shape == (stats['n_columns'],) and np.isfinite(diagonal).all() and
            np.all(diagonal >= 0), 'Invalid FIT diagonal')
    require(type(stats.get('all_standardized_entries_exact_zero')) is bool,
            'Exact-zero observation required; zero variance is not constancy')
    weight = stats.get('weight_sum')
    require(isinstance(weight, (float, int)) and math.isfinite(weight) and weight > 0,
            'Invalid FIT total weight')
    lower, upper = float(diagonal.max()), math.fsum(map(float, diagonal))
    require(stats.get('spectral_lower') == lower and stats.get('spectral_upper') == upper and
            math.isfinite(upper), 'FIT spectral statistics disagree with diagonal')
    return float(weight), lower, upper


def _make_values(start_tick, end_tick, subdivisions, weight):
    values = []
    for tick in range(start_tick, end_tick+1):
        exponent = tick/subdivisions
        try:
            lam = 10.**exponent
        except OverflowError as exc:
            raise ValueError('Penalty domain exceeds FP64 range') from exc
        alpha = weight*lam
        require(math.isfinite(lam) and lam > 0 and math.isfinite(alpha) and alpha > 0,
                'Penalty domain exceeds FP64 range')
        values.append(dict(key=f'log10_tick_{tick}_per_{subdivisions}',
                           log10_lambda=exponent, normalized_lambda=lam, alpha=alpha,
                           coarse_decade_sensitivity=(tick % subdivisions == 0)))
    require(all(a['alpha'] < b['alpha'] for a, b in zip(values, values[1:])),
            'Distinct penalties collapse at FP64 precision')
    return values


def finite_family(stats, *, expected_statistics_sha256, epsilon,
                  subdivisions_per_decade, guard_decades, max_penalties):
    """Plan a finite, reproducible family from FIT spectral scale bounds.

    For an eigenvalue h the shrinkage is h/(h+lambda). The low anchor retains
    >=1-epsilon of the leading direction; the high anchor retains <=epsilon of
    every direction. Weak eigendirections need not be near OLS at the low anchor.
    Spacing/guards are design choices, not a continuous-optimum certificate.
    """
    require(content_sha256(stats) == expected_statistics_sha256,
            'FIT statistics content hash mismatch')
    weight, lower, upper = _validated_statistics(stats)
    require(math.isfinite(epsilon) and 0 < epsilon < .5,
            'Explicit epsilon must be between 0 and one half')
    require(type(subdivisions_per_decade) is int and subdivisions_per_decade > 0 and
            type(guard_decades) is int and guard_decades >= 1 and
            type(max_penalties) is int and max_penalties >= 1,
            'Invalid explicit spacing, guards or penalty cap')
    policy = dict(epsilon=float(epsilon), subdivisions_per_decade=subdivisions_per_decade,
                  guard_decades=guard_decades, max_penalties=max_penalties)
    base = dict(schema='r02_ridge_finite_family_v1', role='FIT',
        statistics_sha256=expected_statistics_sha256, input_identity=stats['input_identity'],
        policy=policy, weight_sum=weight, spectral_lower=lower, spectral_upper=upper,
        penalty_convention='weighted SSE + alpha * squared coefficient norm; alpha=W*lambda',
        scientific_admission=False, continuous_optimum_claim=False,
        numerical_scope=stats['numerical_scope'])
    if upper == 0:
        # Underflow of squares must not turn a nonzero column into a claim of
        # structural constancy or silently discard it from the logical design.
        status = ('FIT_CONSTANT_INTERCEPT_ONLY' if stats['all_standardized_entries_exact_zero']
                  else 'INCOMPLETE_FP64_SCALE_UNDERFLOW')
        return base | dict(status=status, penalties=[],
                           n_penalties=0, extensions=[])
    low = epsilon*lower/(1-epsilon)
    high = (1-epsilon)*upper/epsilon
    require(math.isfinite(low) and low > 0 and math.isfinite(high) and high > 0,
            'Derived anchors exceed FP64 range')
    start = (math.floor(math.log10(low))-guard_decades)*subdivisions_per_decade
    end = (math.ceil(math.log10(high))+guard_decades)*subdivisions_per_decade
    count = end-start+1
    return base | dict(status=('PLANNED_NOT_FITTED' if count <= max_penalties
                               else 'INCOMPLETE_PENALTY_CAP'),
        anchor_low=low, anchor_high=high, start_tick=start, end_tick=end,
        n_penalties=count,
        penalties=(_make_values(start, end, subdivisions_per_decade, weight)
                   if count <= max_penalties else []),
        leading_retention_lower_at_low=lower/(lower+low),
        all_direction_retention_upper_at_high=upper/(upper+high),
        maximum_adjacent_shrinkage_change=(10.**(1/(2*subdivisions_per_decade))-1)/
                                           (10.**(1/(2*subdivisions_per_decade))+1),
        low_end_limitation='No near-OLS guarantee for weak directions', extensions=[])


def propose_extension(plan, *, expected_plan_sha256, completed_penalty_keys,
                      plausible_penalty_keys):
    """Only propose new requests after every declared penalty completed.

    The caller supplies numerically plausible keys from its reviewed selector.
    This does not compute loss, choose a model, omit a failed penalty or dispatch.
    Every iteration keeps the same outer guard width and expands by one decade.
    """
    require(content_sha256(plan) == expected_plan_sha256,
            'Finite family content hash mismatch')
    require(plan.get('schema') == 'r02_ridge_finite_family_v1' and
            plan.get('status') == 'PLANNED_NOT_FITTED', 'Extendable complete plan required')
    wanted = [v['key'] for v in plan['penalties']]
    require(len(completed_penalty_keys) == len(set(completed_penalty_keys)) and
            set(completed_penalty_keys) == set(wanted),
            'Every planned penalty must complete before extension or selection')
    require(plausible_penalty_keys and len(plausible_penalty_keys) == len(set(plausible_penalty_keys)) and
            set(plausible_penalty_keys) <= set(wanted), 'Invalid plausible penalty keys')
    sub = plan['policy']['subdivisions_per_decade']
    guard = plan['policy']['guard_decades']*sub
    ticks = {v['key']: plan['start_tick']+i for i, v in enumerate(plan['penalties'])}
    selected_ticks = [ticks[k] for k in plausible_penalty_keys]
    low = min(selected_ticks) <= plan['start_tick']+guard
    high = max(selected_ticks) >= plan['end_tick']-guard
    if not (low or high):
        return dict(status='DECLARED_FINITE_SEARCH_INTERIOR', continuous_optimum_claim=False,
                    prior_plan_sha256=expected_plan_sha256, required_new_penalties=[])
    start, end = plan['start_tick']-sub*low, plan['end_tick']+sub*high
    count = end-start+1
    if count > plan['policy']['max_penalties']:
        return dict(status='INCOMPLETE_PENALTY_CAP', prior_plan_sha256=expected_plan_sha256,
                    required_penalty_count=count, low_extension=low, high_extension=high,
                    selection_allowed=False)
    values = _make_values(start, end, sub, plan['weight_sum'])
    amended = plan | dict(start_tick=start, end_tick=end, n_penalties=count, penalties=values,
        extensions=plan['extensions']+[dict(prior_plan_sha256=expected_plan_sha256,
                                           low=low, high=high)])
    return dict(status='NEW_REQUESTS_REQUIRE_FAMILY_BARRIER', amended_plan=amended,
                required_new_penalties=[v for v in values if v['key'] not in wanted],
                selection_allowed=False, scientific_admission=False)
