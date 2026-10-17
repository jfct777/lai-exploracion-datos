#!/usr/bin/env python3
"""Select complete K03 procedures by view, retaining numerical ambiguity.

Only SELECT losses enter this step. Its interval arithmetic describes numerical
uncertainty conditional on residual accuracy and fixed preprocessing, never
population confidence. Scientific search adequacy needs separate admission.
"""
import json
import math
from pathlib import Path

import r02_structure_k03_family as barrier

require = barrier.require
ARMS = ('CONTINUOUS', 'COMMUNITY_A', 'COMMUNITY_B')


def minimum(rows):
    """Bound the exact minimum; do not choose an arbitrary overlapping model."""
    require(rows, 'A declared comparison arm cannot be empty')
    bounds = {}
    for row in rows:
        loss = row['weighted_select_mse']
        n = row['numerical_loss']
        low, high = n['exact_ridge_mse_lower'], n['exact_ridge_mse_upper']
        require(all(type(v) in (float, int) and math.isfinite(v) for v in (loss, low, high))
                and 0 <= low <= loss <= high, 'Point loss inconsistent with numerical bounds')
        bounds[row['model_id']] = (low, high)
    upper = min(v[1] for v in bounds.values())
    plausible = sorted(k for k, v in bounds.items() if v[0] <= upper)
    # Equality is deliberately unresolved, including zero-width exact ties.
    winners = [key for key, (_, high) in bounds.items()
               if all(key == other or high < low for other, (low, _) in bounds.items())]
    require(len(winners) <= 1, 'Contradictory numerical ordering')
    exact_tie = (len(plausible) > 1 and
                 all(bounds[key][0] == bounds[key][1] == upper for key in plausible))
    tie_winner = None
    if exact_tie:
        tied = [r for r in rows if r['model_id'] in plausible]
        designs = {(r['arm'], r['candidate_id'], r['baseline_identity']) for r in tied}
        # Preserve the recorded R02 alpha tie rule only within the same design.
        # It makes no claim that alpha orders complexity across different graphs.
        if len(designs) == 1:
            strongest = max(r['alpha'] for r in tied)
            finalists = [r['model_id'] for r in tied if r['alpha'] == strongest]
            if len(finalists) == 1:
                tie_winner = finalists[0]
    return dict(minimum_point_loss=min(r['weighted_select_mse'] for r in rows),
                minimum_exact_loss_lower=min(v[0] for v in bounds.values()),
                minimum_exact_loss_upper=upper, plausible_model_ids=plausible,
                selected_model_id=winners[0] if winners else tie_winner,
                model_count=len(rows),
                status='UNIQUE_NUMERICAL_MINIMUM' if winners else
                       'EXACT_LOSS_TIE_STRONGEST_ALPHA_WITHIN_SAME_DESIGN' if tie_winner else
                       'EXACT_TIE_REQUIRES_PREDECLARED_RULE' if exact_tie else 'NUMERICAL_SELECTION_UNRESOLVED')


def summarize(family, complete, *, family_sha256):
    """Use authoritative group membership, never representative baseline labels."""
    require(barrier.valid_digest(family_sha256) and complete.get('family_sha256') == family_sha256,
            'Selection must consume the same authenticated family')
    require(complete.get('stage') == 'DECLARED_FAMILY_AUTHENTICATED_SELECTION_NOT_PERFORMED'
            and complete.get('selection_performed') is False
            and complete.get('evaluate_targets_computed') is False
            and complete.get('scientific_spec_sha256') == family['scientific_spec_sha256'],
            'Complete unselected FIT/SELECT family required')
    declared = {r['model_id']: r for r in family['models']}
    observed = {r['model_id']: r for r in complete['models']}
    require(len(declared) == len(family['models']) == len(observed) == len(complete['models'])
            and declared.keys() == observed.keys(), 'Complete model inventory differs')
    for name, row in observed.items():
        require(all(row[k] == declared[name][k] for k in barrier.IDENTITY),
                'Authenticated model identity changed before selection')
    views = {(r['block_id'], r['exclusion_scope']) for r in family['required_views']}
    require(len(views) == len(family['required_views']), 'Duplicate target view')
    seen, used, used_community, results, view_support = set(), set(), set(), [], {}
    for group in family['selection_groups']:
        view = (group['block_id'], group['exclusion_scope'])
        control, seed = group['control'], group['sham_seed']
        require(view in views and ((control == 'REAL' and seed is None) or
                (control == 'SHAM_WITHIN_FIT_COMPONENT' and type(seed) is int and seed >= 0)),
                'Invalid view or randomization identity')
        key = (*view, control, seed)
        require(key not in seen, 'Duplicate comparison group'); seen.add(key)
        by_arm = group['models_by_arm']
        require(set(by_arm) == set(ARMS), 'All three comparison arms required')
        summaries, target_support = {}, set()
        for arm in ARMS:
            names = by_arm[arm]
            require(names and len(names) == len(set(names)) and set(names) <= observed.keys(),
                    'Missing or duplicate group model')
            rows = [observed[n] for n in names]
            require(all((r['block_id'], r['exclusion_scope']) == view and r['arm'] == arm for r in rows),
                    'Group uses a different view or arm')
            if arm != 'CONTINUOUS':
                require(not (set(names) & used_community),
                        'Community model reused across different randomization groups')
                used_community.update(names)
                require(all(r['baseline_model_id'] in by_arm['CONTINUOUS'] for r in rows),
                        'Group omitted a candidate availability-matched baseline')
            target_support.update((r['select_target_identity'], r['fit_pairs'], r['select_pairs']) for r in rows)
            summaries[arm] = minimum(rows); used.update(names)
        require(len(target_support) == 1, 'Comparison group target support differs')
        support = next(iter(target_support))
        require(view not in view_support or view_support[view] == support,
                'Randomization changed target support within the same view')
        view_support[view] = support
        contrasts = {}
        baseline = summaries['CONTINUOUS']
        for arm in ARMS[1:]:
            extension = summaries[arm]
            contrasts[arm] = dict(
                point_gain=baseline['minimum_point_loss']-extension['minimum_point_loss'],
                exact_gain_lower=baseline['minimum_exact_loss_lower']-extension['minimum_exact_loss_upper'],
                exact_gain_upper=baseline['minimum_exact_loss_upper']-extension['minimum_exact_loss_lower'])
        results.append(dict(block_id=view[0], exclusion_scope=view[1], control=control,
                            sham_seed=seed, arms=summaries, contrasts=contrasts))
    require(used == set(observed), 'Declared models omitted from all selection groups')
    require({(r['block_id'], r['exclusion_scope']) for r in results if r['control'] == 'REAL'} == views,
            'Every declared view needs its REAL comparison')
    real_means = []
    for scope in sorted({v[1] for v in views}):
        rows = [r for r in results if r['control'] == 'REAL' and r['exclusion_scope'] == scope]
        for arm in ARMS[1:]:
            real_means.append(dict(exclusion_scope=scope, arm=arm, block_count=len(rows),
                **{k:math.fsum(r['contrasts'][arm][k] for r in rows)/len(rows)
                   for k in ('point_gain','exact_gain_lower','exact_gain_upper')}))
    resolved = all(a['selected_model_id'] is not None for r in results for a in r['arms'].values())
    exact_tie = any(a['status'] == 'EXACT_TIE_REQUIRES_PREDECLARED_RULE'
                    for r in results for a in r['arms'].values())
    return dict(schema='r02_k03_select_procedure_v1', family_sha256=family_sha256,
                scientific_spec_sha256=family['scientific_spec_sha256'], groups=results,
                real_equal_block_means=real_means, model_count=len(observed),
                all_numerical_selections_resolved=resolved,
                stage='SELECT_MODELS_RESOLVED_UNDER_DECLARED_RULES' if resolved else
                      'SELECT_TIE_RULE_REQUIRED' if exact_tie else 'SELECT_NUMERICAL_REFINEMENT_REQUIRED',
                tie_rule='LARGER_ALPHA_ONLY_WITHIN_IDENTICAL_DESIGN_EXACT_LOSS_TIE',
                selection_performed=True, evaluate_allowed=False, evaluate_targets_computed=False,
                global_partition_alignment_assumed=False, biological_validation=False,
                statistical_confidence_interval=False, scientific_search_adequacy_certified=False,
                randomization_precision_assessed=False,
                bound_scope='CONDITIONAL_ON_RESIDUAL_ACCURACY_AND_FIXED_PREPROCESSING')


def collect_and_select(family_path, family_sha256, descriptors, **limits):
    """Authenticate original model products before this numerical selection step."""
    complete = barrier.collect(family_path, family_sha256, descriptors, **limits)
    family = json.loads(Path(family_path).read_text())
    require(barrier.sha256(family_path) == family_sha256, 'Family changed before selection')
    result = summarize(family, complete, family_sha256=family_sha256)
    require(barrier.sha256(family_path) == family_sha256, 'Family changed during selection')
    return result
