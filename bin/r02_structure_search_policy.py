#!/usr/bin/env python3
"""Explicit stochastic-search accounting; no genotype reads or model selection.

The caller freezes each episode BEFORE running its seeds, authenticates results,
and persists that episode and every completed candidate. Probability statements
are conditional on a uniformly sampled optimizer seed / SHAM permutation model;
they are not population-genetic uncertainty or evidence of global optimality.
No numerical policy parameter has a default.
"""
import bisect
import hashlib
import json
import math
import secrets


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and 0 < value < 1,
            name + ' must lie strictly between zero and one')


def episode_error(*, family_error, views, query_index, episode_index):
    probability(family_error, 'family_error')
    require(all(type(x) is int and x > 0 for x in (views, query_index, episode_index)),
            'Positive, one-based declared family/query/episode indices required')
    # Sum 1/[j(j+1)] = 1. No finite query count need be invented for this bound.
    value = family_error / views / query_index / (query_index+1) / episode_index / (episode_index+1)
    require(value > 0, 'Error allocation underflowed')
    return value


def confirmation_size(*, missing_mass, error):
    probability(missing_mass, 'missing_mass')
    probability(error, 'error')
    count = math.ceil(math.log(error) / math.log1p(-missing_mass))
    # Do not accidentally round down the declared probability bound.
    while count * math.log1p(-missing_mass) > math.log(error):
        count += 1
    return count


def freeze_episode(*, source_sha256, view_id, gamma, known_memberships,
                   prior_results, family_error, views, query_index, episode_index,
                   missing_mass, draw_below=None):
    """Sample unique fresh 31-bit seeds, returned for write-before-execute use.

    OS randomness draws uniform seeds without replacement. Reproducibility is
    provided by the immutable returned seed ledger, not by choosing seeds after
    observing results. Tests may inject a deterministic draw function. Production
    must persist/fsync this result before dispatch and reuse it on resume.
    Sampling without replacement retains the (1-epsilon)^r bound: previously
    observed seeds all return a known member, so removing them cannot lower the
    missing-membership proportion of the remaining seed population.
    """
    require(isinstance(source_sha256, str) and len(source_sha256) == 64 and
            all(c in '0123456789abcdef' for c in source_sha256), 'Exact source SHA256 required')
    require(isinstance(view_id, str) and view_id and type(gamma) in (int, float)
            and math.isfinite(gamma) and gamma > 0, 'Exact view and positive gamma required')
    require(isinstance(known_memberships, (list, tuple, set)) and
            all(isinstance(k, str) and k for k in known_memberships), 'Frozen membership universe required')
    require(isinstance(prior_results, list), 'Authenticated prior results at this same gamma required')
    prior_seeds = [r['seed'] for r in prior_results]
    require(len(set(prior_seeds)) == len(prior_seeds) and
            all(type(s) is int and 0 <= s < 2**31 for s in prior_seeds), 'Unique prior seed ledger required')
    for r in prior_results:
        require(r['source_sha256'] == source_sha256 and r['gamma_hex'] == float(gamma).hex()
                and r['status'] == 'FIT_ZERO_IMPROVEMENT_ITERATION_REACHED'
                and r['membership_key'] in known_memberships,
                'Prior seed must belong to same source/gamma and a retained known member')
    error = episode_error(family_error=family_error, views=views, query_index=query_index,
                          episode_index=episode_index)
    count = confirmation_size(missing_mass=missing_mass, error=error)
    require(count + len(prior_seeds) <= 2**31, 'Seed population exhausted')
    draw = draw_below if draw_below is not None else secrets.randbelow
    used, seeds = sorted(prior_seeds), []
    while len(seeds) < count:
        available = 2**31-len(used)
        seed = draw(available)
        require(type(seed) is int and 0 <= seed < available,
                'Seed provider must draw a uniform index among remaining seeds')
        # Map a uniform rank to the corresponding unused seed, without retries.
        for previous in used:
            if previous > seed:
                break
            seed += 1
        seeds.append(seed)
        bisect.insort(used, seed)
    result = dict(schema='r02_optimizer_confirmation_episode_v1', source_sha256=source_sha256,
        view_id=view_id, gamma_hex=float(gamma).hex(), known_memberships=sorted(set(known_memberships)),
        prior_seeds=sorted(prior_seeds), seeds=seeds, family_error=family_error, views=views,
        prior_results_sha256=hashlib.sha256(json.dumps(prior_results, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        query_index=query_index, episode_index=episode_index, missing_mass=missing_mass,
        allocated_error=error, required_attempts=count,
        sampling='UNIFORM_31BIT_WITHOUT_REPLACEMENT_CONDITIONAL_ON_RECORDED_HISTORY',
        biological_validation=False, global_optimum_proven=False)
    result['episode_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def verify_episode(episode):
    payload = {k: v for k, v in episode.items() if k != 'episode_sha256'}
    require(episode.get('episode_sha256') == hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'Frozen episode changed')
    error = episode_error(family_error=episode['family_error'], views=episode['views'],
        query_index=episode['query_index'], episode_index=episode['episode_index'])
    require(error == episode['allocated_error'] and episode['required_attempts'] ==
        confirmation_size(missing_mass=episode['missing_mass'], error=error), 'Episode error accounting differs')
    require(len(episode['seeds']) == len(set(episode['seeds'])) == episode['required_attempts'] and
            not set(episode['seeds']) & set(episode['prior_seeds']), 'Episode seeds are not fresh')


def assess_episode(episode, results):
    """No majority vote, objective filtering, or partial-family acceptance.

    The caller authenticates each task's source/code/result hashes. This function
    checks the small seed/source/gamma/status records and accounts for novelty.
    All returned memberships remain predictive candidates, including RB-dominated
    candidates. A failed optimization makes coverage incomplete.
    """
    verify_episode(episode)
    require(isinstance(results, list), 'Complete result records required')
    seeds = [r['seed'] for r in results]
    require(len(seeds) == len(set(seeds)) and set(seeds) == set(episode['seeds']),
            'Incomplete or duplicate confirmation results')
    for r in results:
        require(r['source_sha256'] == episode['source_sha256'] and r['gamma_hex'] == episode['gamma_hex'],
                'Result source or gamma differs from frozen episode')
        require(r['status'] == 'FIT_ZERO_IMPROVEMENT_ITERATION_REACHED',
                'Unfinished optimizer cannot provide coverage')
        require(isinstance(r['membership_key'], str) and r['membership_key'], 'Membership identity missing')
    found = {r['membership_key'] for r in results}
    new = sorted(found-set(episode['known_memberships']))
    return dict(status='NEW_MEMBERSHIPS_REQUIRE_NEW_EPISODE' if new else 'DECLARED_SEED_MASS_COVERED',
        episode_sha256=episode['episode_sha256'], new_memberships=new,
        retained_memberships=sorted(found | set(episode['known_memberships'])),
        missing_mass_upper=episode['missing_mass'] if not new else None,
        allocated_error=episode['allocated_error'], biological_validation=False,
        global_optimum_proven=False)


def sham_probability_interval(*, lower_successes, upper_successes, replicas,
                              family_error, comparisons, desired_width):
    """Exact binomial intervals at n=1,2,4,... with summable error spending.

    Success means SHAM gain >= REAL gain. Unresolved numerical comparisons use
    lower_successes <= true successes <= upper_successes, never dropped rows.
    Clopper-Pearson monotonicity supplies the wider interval for this bracket.
    Union over looks and comparisons makes coverage valid at a data-dependent
    stopping look. This describes the conditional randomized representation
    procedure, NOT a demographic null or a biological significance test.
    """
    from scipy.stats import binomtest
    probability(family_error, 'family_error')
    probability(desired_width, 'desired_width')
    require(type(comparisons) is int and comparisons > 0, 'Declared comparison family required')
    require(type(replicas) is int and replicas > 0 and replicas & (replicas-1) == 0,
            'Only predeclared doubling looks are supported')
    require(type(lower_successes) is int and type(upper_successes) is int and
            0 <= lower_successes <= upper_successes <= replicas, 'Invalid numerical success bracket')
    look = replicas.bit_length()
    error = family_error/comparisons/look/(look+1)
    require(0 < error < 1 and 1-error < 1, 'Confidence allocation underflowed')
    lower = binomtest(lower_successes, replicas).proportion_ci(confidence_level=1-error, method='exact').low
    upper = binomtest(upper_successes, replicas).proportion_ci(confidence_level=1-error, method='exact').high
    return dict(schema='r02_conditional_sham_mc_interval_v1', replicas=replicas, look=look,
        lower_successes=lower_successes, upper_successes=upper_successes,
        interval=[float(lower), float(upper)], width=float(upper-lower),
        desired_width=desired_width, precision_reached=bool(upper-lower <= desired_width),
        allocated_error=error, family_error=family_error, comparisons=comparisons,
        event='SHAM_GAIN_GREATER_THAN_OR_EQUAL_TO_REAL_GAIN',
        inference='CONDITIONAL_RANDOMIZATION_MONTE_CARLO_PROBABILITY_ONLY',
        biological_p_value=False, biological_validation=False)
