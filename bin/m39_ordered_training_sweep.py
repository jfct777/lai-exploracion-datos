#!/usr/bin/env python3
"""Freeze paired recipes and audit development comparisons without reading SCORE.

GPU execution belongs to the existing Nextflow worker. This module separates
recipe preparation from primary-result verification and exploratory selection.
It never infers dense ancestry transitions from the sparse anchor predictions.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from m33_safe_bridge_core import write_exclusive_json
from m39_anchor_screen import metrics, sha256
from m39_ordered_gpu_manifest import ARMS, SCHEMA, SCOPE, STAGE_ARMS, load_plan, require
from m39_ordered_training import load_config
from m39_ordered_training_data import stratified_metrics


# This is a cross-runtime serialization check, not a model-selection margin.
# Frozen NPZ/checkpoint/config hashes, discrete metrics and all metadata remain
# exact. The floor is below any reported scientific effect and matches the
# pre-existing independent primary reader's absolute float64 check.
METRIC_RECOMPUTATION_ATOL = 1e-12
RECOMPUTED_FLOAT_FIELDS = frozenset({
    'brier', 'log_loss', 'dosage_mae', 'brier_per_person', 'log_loss_per_person'})


@dataclass(frozen=True)
class SweepProtocol:
    """Names and config validation for a paired experiment; auditing stays shared."""

    config_loader: Callable[[Path], dict] | None = None
    focal_arm: str = 'real'
    common_arm: str = 'common'
    screen_stage: str = 'exploratory_screen'
    followup_stage: str = 'controlled_followup'
    technical_stage: str = 'technical_e2e'
    comparison_schema: str = 'm39-ordered-development-comparison-v1'
    initial_checkpoint_eligible: bool = True
    preserve_followup_geometry: bool = False

    def read_config(self, path: Path) -> dict:
        return (self.config_loader or load_config)(path)


def verify_recomputed_metrics(actual, expected, path: str = 'metrics') -> list[dict]:
    """Check finite same-schema metrics, recording only bounded float roundoff.

    No receipt value is replaced or rounded. Only named derived error measures
    admit an absolute 1e-12 difference (relative tolerance is zero). Integers,
    booleans, nulls, lengths, keys, accuracy, floor and weighting remain exact.
    """
    differences = []

    def visit(left, right, where, floating_error=False):
        require(type(left) is type(right), f'{where}: metric value types differ')
        if isinstance(left, dict):
            require(left.keys() == right.keys(), f'{where}: metric fields differ')
            for key in left:
                visit(left[key], right[key], f'{where}.{key}', key in RECOMPUTED_FLOAT_FIELDS)
        elif isinstance(left, list):
            require(len(left) == len(right), f'{where}: metric array length differs')
            for i, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f'{where}[{i}]', floating_error)
        elif isinstance(left, float):
            require(math.isfinite(left) and math.isfinite(right), f'{where}: non-finite metric')
            delta = abs(left - right)
            require(delta <= (METRIC_RECOMPUTATION_ATOL if floating_error else 0.),
                    f'{where}: recomputed metric differs')
            if delta:
                differences.append({'field': where, 'receipt': right,
                                    'recomputed': left, 'absolute_difference': delta})
        else:
            require(left == right, f'{where}: exact metric metadata differs')

    visit(actual, expected, path)
    return differences


def freeze_plan(base_path: Path, resources: dict, recipes: list[dict], stage: str,
                outdir: Path, *, protocol: SweepProtocol | None = None) -> Path:
    """Freeze explicitly supplied values; no automatic parameter or anchor search."""
    protocol = protocol or SweepProtocol()
    base = protocol.read_config(base_path)
    require(base['device'] == 'cuda:0', 'scientific plan requires the measured GPU runtime')
    require(stage in STAGE_ARMS, 'unknown stage')
    require(not outdir.exists() and not outdir.is_symlink(), 'plan output already exists')
    require(isinstance(recipes, list) and len(recipes) > 0, 'empty recipe set')
    for recipe in recipes:
        require(set(recipe) == {'id', 'family', 'learning_rate', 'seed', 'pair_seed'},
                'recipe fields differ')
        require(recipe['family'] in ('cnn', 'attention'), 'unknown model family')
    arms = STAGE_ARMS[stage]
    outdir.mkdir(parents=True, mode=0o700, exist_ok=False)
    groups = []
    for recipe in recipes:
        specs = []
        for arm in arms:
            cfg = copy.deepcopy(base)
            cfg.update(case_id=f"{recipe['id']}-{arm}", paired_budget_id=recipe['id'], arm=arm,
                       learning_rate=recipe['learning_rate'], seed=recipe['seed'],
                       pair_seed=recipe['pair_seed'])
            cfg['model']['family'] = recipe['family']
            path = outdir / f"{recipe['id']}-{arm}.json"
            write_exclusive_json(path, cfg)
            protocol.read_config(path)
            specs.append({'file': path.name, 'sha256': sha256(path)})
        groups.append({'id': recipe['id'], 'configs': specs})
    plan = {'schema_version': SCHEMA, 'scope': SCOPE, 'stage': stage,
            'resources': resources, 'inputs': {key: base[key] for key in
             ('train_manifest_sha256', 'select_manifest_sha256', 'development_sha256')},
            'groups': groups}
    path = outdir / 'plan.json'
    write_exclusive_json(path, plan)
    load_plan(path)
    write_exclusive_json(outdir / 'preparation.receipt.json', {
        'stage': stage, 'base_config_sha256': sha256(base_path),
        'plan_sha256': sha256(path), 'generator_sha256': sha256(Path(__file__)),
        'recipes': recipes, 'SCORE_opened': False, 'training_launched': False})
    return path


def _verified_case(path: Path, config_path: Path, receipt_sha: str, *,
                   protocol: SweepProtocol | None = None) -> tuple[dict, dict, list]:
    protocol = protocol or SweepProtocol()
    receipt_path = path / 'training.receipt.json'
    require(sha256(receipt_path) == receipt_sha, 'worker/primary receipt hash differs')
    receipt = json.loads(receipt_path.read_text())
    cfg = protocol.read_config(config_path)
    require(receipt['config'] == cfg and receipt['sources']['config_sha256'] == sha256(config_path)
            and receipt['decision'] == 'COMPLETED_EXPLORATORY_DEVELOPMENT_CASE'
            and receipt['scope']['SCORE_opened'] is False, 'case receipt scope differs')
    require(sha256(path / 'checkpoint.pt') == receipt['checkpoint_sha256'], 'checkpoint hash differs')
    if cfg['arm'] == 'sham':
        require(sha256(path / 'sham.reference-map.npz') == receipt['sham_reference_map_sha256'],
                'SHAM map hash differs')
        require(json.loads((path / 'sham.design-diagnostic.json').read_text()) ==
                receipt['sham_TRAIN_dose_changes'], 'SHAM change diagnostic differs')
    predictions = path / 'select.predictions.npz'
    require(sha256(predictions) == receipt['predictions_sha256'], 'prediction hash differs')
    with np.load(predictions, allow_pickle=False) as z:
        require(set(z.files) == {'probabilities', 'truth_state', 'anchor_indices',
                                'locus_id', 'sample_key_sha256', 'query_carrier', 'query_observed'},
                'prediction axes differ')
        arrays = {name: z[name].copy() for name in z.files}
    differences = verify_recomputed_metrics(
        metrics(arrays['probabilities'], arrays['truth_state']), receipt['selected_SELECT_metrics'])
    differences.extend(verify_recomputed_metrics(stratified_metrics(
        arrays['probabilities'], arrays['truth_state'], arrays['query_carrier'], arrays['query_observed']),
        receipt['selected_SELECT_stratified_descriptive_only']['model'], 'descriptive_strata'))
    require(arrays['anchor_indices'].tolist() == receipt['anchor_indices'], 'receipt/prediction anchors differ')
    keys = [(row['SELECT']['brier'], row['SELECT']['log_loss'], row['step'])
            for row in receipt['curve'] if protocol.initial_checkpoint_eligible or row['step'] > 0]
    require(bool(keys), 'no eligible checkpoint in the declared curve')
    require(min(keys)[2] == receipt['selected_step'], 'checkpoint selection is not declared SELECT minimum')
    selected_rows = [row for row in receipt['curve'] if row['step'] == receipt['selected_step']]
    require(len(selected_rows) == 1 and selected_rows[0]['SELECT'] == receipt['selected_SELECT_metrics'],
            'selected curve metrics differ from selected receipt metrics')
    for row in receipt['curve']:
        if not protocol.initial_checkpoint_eligible:
            require(row.get('checkpoint_eligible') is (row['step'] > 0),
                    'checkpoint eligibility differs from the declared protocol')
        primary = json.loads((path / f"curve-step-{row['step']:07d}.json").read_text())
        require(primary == row, 'primary curve checkpoint differs')
    return receipt, arrays, differences


def verify_complete_pass_schedule(receipt: dict, cfg: dict) -> None:
    """Check actual pair exposure and completed-pass evaluation, including remainders."""
    available = receipt['TRAIN_exposure']['available_pairs']
    anchors = len(receipt['anchor_indices'])
    require(type(available) is int and available > 0 and anchors > 0 and available % anchors == 0,
            'invalid declared training-pair universe')
    people = available // anchors
    steps_per_pass = ((people + cfg['batch_size'] - 1) // cfg['batch_size']) * anchors
    require(cfg['evaluate_every_steps'] == steps_per_pass
            and cfg['steps'] % steps_per_pass == 0, 'scientific checkpoints must close complete passes')
    expected_steps = ([0] if cfg.get('evaluate_initial', False) else []) + list(
        range(steps_per_pass, cfg['steps'] + 1, steps_per_pass))
    require([row['step'] for row in receipt['curve']] == expected_steps,
            'scientific checkpoint schedule differs')
    for row in receipt['curve']:
        passes = row['step'] // steps_per_pass
        exposure = row['TRAIN_exposure']
        require(exposure['available_pairs'] == available
                and exposure['minimum_visits_per_pair'] == exposure['maximum_visits_per_pair'] == passes
                and exposure['complete_passes_over_declared_pairs'] == passes
                and exposure['observations'] == available * passes
                and row['train_observations_cumulative'] == available * passes,
                'scientific checkpoint did not complete its declared pair passes')
    require(receipt['TRAIN_exposure'] == receipt['curve'][-1]['TRAIN_exposure']
            and receipt['training_observations'] == available * (cfg['steps'] // steps_per_pass),
            'final scientific exposure differs')


def audit_results(plan_path: Path, outputs: Path, outdir: Path | None = None, *,
                  protocol: SweepProtocol | None = None) -> dict:
    """Reopen all declared cases and same-axis controls before producing a comparison."""
    plan = load_plan(plan_path)
    # Keep legacy call signatures for downstream diagnostic hooks and tests.
    options = {'protocol': protocol} if protocol is not None else {}
    protocol = protocol or SweepProtocol()
    if outdir is not None:
        require(not outdir.exists() and not outdir.is_symlink(), 'comparison output already exists')
    groups, rows, primary_hashes, numeric_differences = [], [], {}, []
    for group in plan['groups']:
        # Match the Nextflow publishDir contract exactly; do not infer alternate
        # folders from whichever stale run happens to exist beside this one.
        group_path = outputs / f"training-{group['id']}"
        done_path = group_path / 'group.completion.json'
        done = json.loads(done_path.read_text())
        require(done['status'] == 'COMPLETED_DECLARED_PAIRED_ARMS_NEEDS_SCIENTIFIC_POST'
                and done['group_id'] == group['id'] and done['plan_sha256'] == sha256(plan_path)
                and done['stage'] == plan['stage'], 'incomplete or differently bound worker group')
        primary_hashes[group['id']] = sha256(done_path)
        cases, reference_arrays, paired_reference = {}, None, None
        for spec in group['configs']:
            cfg_path = plan_path.parent / spec['file']
            cfg = protocol.read_config(cfg_path)
            receipt, arrays, differences = _verified_case(group_path / cfg['arm'], cfg_path,
                                                          done['case_receipt_sha256'][cfg['arm']], **options)
            if protocol.preserve_followup_geometry and plan['stage'] != protocol.technical_stage:
                verify_complete_pass_schedule(receipt, cfg)
            numeric_differences.extend({'case_id': cfg['case_id'], **row} for row in differences)
            paired = {key: receipt[key] for key in ('initial_state_sha256', 'training_pair_stream_sha256',
                'anchor_indices', 'training_observations', 'TRAIN_exposure', 'batch_policy')}
            require(paired_reference is None or paired_reference == paired, 'paired training exposure differs')
            if reference_arrays is not None:
                for key in ('truth_state', 'anchor_indices', 'locus_id', 'sample_key_sha256',
                            'query_carrier', 'query_observed'):
                    require(np.array_equal(reference_arrays[key], arrays[key]), 'paired prediction axes differ')
                for key in ('Fminus_SELECT', 'Ffull_SELECT'):
                    require(cases[next(iter(cases))][key] == receipt[key], 'paired FLARE comparator differs')
            paired_reference, reference_arrays = paired, arrays
            cases[cfg['arm']] = receipt
            selected = receipt['selected_SELECT_metrics']
            rows.append({'group': group['id'], 'family': cfg['model']['family'], 'arm': cfg['arm'],
                'seed': cfg['seed'], 'learning_rate': cfg['learning_rate'],
                'selected_step': receipt['selected_step'], 'brier': selected['brier'],
                'log_loss': selected['log_loss'],
                **{f'dosage_mae_{ancestry}': selected['dosage_mae'][i]
                   for i, ancestry in enumerate(('AFR', 'EUR', 'NAM'))},
                'complete_passes': receipt['TRAIN_exposure']['complete_passes_over_declared_pairs'],
                'budget_end_selected': receipt['budget_diagnostics']['best_checkpoint_at_budget_end']})
        require(done['completed_arms'] == list(cases), 'worker arm inventory differs')
        focal = cases[protocol.focal_arm]['selected_SELECT_metrics']
        focal_label, common_label = protocol.focal_arm.upper(), protocol.common_arm.upper()
        improvements = {arm: {metric: (np.asarray(receipt['selected_SELECT_metrics'][metric])
                                      - np.asarray(focal[metric])).tolist()
                              for metric in ('brier', 'log_loss', 'dosage_mae')}
                        for arm, receipt in cases.items() if arm != protocol.focal_arm}
        for name in ('Fminus_SELECT', 'Ffull_SELECT'):
            improvements[name] = {metric: (np.asarray(cases[protocol.focal_arm][name][metric])
                                           - np.asarray(focal[metric])).tolist()
                                  for metric in ('brier', 'log_loss', 'dosage_mae')}
        groups.append({'id': group['id'], 'family': cfg['model']['family'],
                       'learning_rate': cfg['learning_rate'], 'seed': cfg['seed'],
                       f'{focal_label}_metrics': focal,
                       f'{common_label}_metrics': cases[protocol.common_arm]['selected_SELECT_metrics'],
                       'descriptive_strata': {arm: receipt['selected_SELECT_stratified_descriptive_only']
                                              for arm, receipt in cases.items()},
                       'sham_TRAIN_dose_changes': cases.get('sham', {}).get('sham_TRAIN_dose_changes'),
                       f'control_minus_{focal_label}': improvements,
                       'positive_improvement_means': f'lower_error_for_{focal_label}',
                       'paired_exposure': paired_reference})
    result = {'schema_version': protocol.comparison_schema,
              'stage': plan['stage'], 'plan_sha256': sha256(plan_path),
              'primary_group_hashes': primary_hashes, 'groups': groups, 'cases': rows,
              'metric_recomputation': {'policy': 'finite_float64_absolute_error_only_v1',
                  'atol': METRIC_RECOMPUTATION_ATOL, 'rtol': 0.,
                  'derived_float_fields': sorted(RECOMPUTED_FLOAT_FIELDS),
                  'metadata_axes_hashes_selection_and_discrete_values': 'exact',
                  'receipt_values_modified': False, 'nonzero_differences': numeric_differences},
              'scope': {'exploratory_SELECT_reused_for_selection': plan['stage'] != protocol.technical_stage,
                        'technical_e2e_only': plan['stage'] == protocol.technical_stage,
                        'SCORE_opened': False, 'dense_LAI_or_border_F1': False,
                        'negative_family_conclusion_allowed': False,
                        'SHAM_exact_exchangeability_test': False}}
    if outdir is not None:
        outdir.mkdir(mode=0o700, parents=True, exist_ok=False)
        write_exclusive_json(outdir / 'comparison.json', result)
        with (outdir / 'cases.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return result


def selected_learning_rates(summary: dict, *, protocol: SweepProtocol | None = None) -> dict:
    """Keep both families; choose LR by mean COMMON/REAL error, never rare improvement."""
    protocol = protocol or SweepProtocol()
    require(summary['stage'] == protocol.screen_stage, 'LR selection requires screening outputs')
    focal_label, common_label = protocol.focal_arm.upper(), protocol.common_arm.upper()
    chosen = {}
    for family in ('cnn', 'attention'):
        groups = [group for group in summary['groups'] if group['family'] == family]
        require(len(groups) >= 2 and len({group['learning_rate'] for group in groups}) == len(groups),
                'incomplete or duplicated learning-rate comparison')
        def objective(group, metric):
            values = [group[f'{label}_metrics'][metric] for label in (focal_label, common_label)]
            require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        for value in values), 'invalid learning-rate selection metric')
            return sum(values) / 2
        require(all(type(group['learning_rate']) in (int, float)
                    and math.isfinite(group['learning_rate']) and group['learning_rate'] > 0
                    for group in groups), 'invalid learning rate')
        winner = min(groups, key=lambda group: (objective(group, 'brier'),
            objective(group, 'log_loss'), group['learning_rate']))
        chosen[family] = {'learning_rate': winner['learning_rate'], 'source_group': winner['id'],
                          'selection_metric': f'minimum_mean_{common_label}_{focal_label}_SELECT_Brier_then_mean_log_loss_then_lower_LR',
                          f'mean_{common_label}_{focal_label}_brier': objective(winner, 'brier'),
                          'does_not_establish_incremental_rare_value': True}
    return chosen


def freeze_followup(base_path: Path, resources: dict, screen_plan: Path,
                    screen_outputs: Path, seeds: list[int], outdir: Path, *,
                    protocol: SweepProtocol | None = None) -> Path:
    """Audit screening primaries, retain both families and freeze fresh paired replicas."""
    require(len(seeds) >= 2 and len(set(seeds)) == len(seeds)
            and all(type(seed) is int and 0 <= seed < 2**63 - 100 for seed in seeds),
            'at least two distinct valid followup seeds are required')
    options = {'protocol': protocol} if protocol is not None else {}
    protocol = protocol or SweepProtocol()
    summary = audit_results(screen_plan, screen_outputs, **options)
    require(not set(seeds) & {row['seed'] for row in summary['groups']},
            'followup seeds must differ from screening')
    chosen = selected_learning_rates(summary, **options)
    if protocol.preserve_followup_geometry:
        base = protocol.read_config(base_path)
        plan = load_plan(screen_plan)
        for group in plan['groups']:
            screen = protocol.read_config(screen_plan.parent / group['configs'][0]['file'])
            for key in ('train_manifest_sha256', 'select_manifest_sha256', 'development_sha256',
                        'anchor_count', 'anchor_seed', 'batch_size', 'multichannel'):
                require(base[key] == screen[key], f'followup changes paired geometry/input: {key}')
            model = {key: value for key, value in base['model'].items() if key != 'family'}
            previous = {key: value for key, value in screen['model'].items() if key != 'family'}
            require(model == previous, 'followup changes encoder geometry/capacity')
    recipes = [dict(id=f'{family}-seed{seed}', family=family,
                    learning_rate=chosen[family]['learning_rate'], seed=seed, pair_seed=seed + 100)
               for family in ('cnn', 'attention') for seed in seeds]
    path = freeze_plan(base_path, resources, recipes, protocol.followup_stage, outdir, **options)
    write_exclusive_json(outdir / 'screen-primary-audit.json', summary)
    write_exclusive_json(outdir / 'selection.receipt.json', {
        'selected_learning_rates': chosen, 'screen_plan_sha256': sha256(screen_plan),
        'screen_primary_audit_sha256': sha256(outdir / 'screen-primary-audit.json'),
        'followup_plan_sha256': sha256(path), 'SCORE_opened': False,
        'selection_generalizes_across_radii': ('not_claimed_same_geometry_required'
            if protocol.preserve_followup_geometry else 'unproven_fixed_budget_transfer'),
        'no_family_eliminated': True, 'followup_is_confirmatory': False})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prepare = modes.add_parser('prepare')
    for name in ('base-config', 'recipe-file', 'outdir'):
        prepare.add_argument('--' + name, type=Path, required=True)
    post = modes.add_parser('audit')
    for name in ('plan', 'outputs', 'outdir'):
        post.add_argument('--' + name, type=Path, required=True)
    followup = modes.add_parser('prepare-followup')
    for name in ('base-config', 'resources-file', 'screen-plan', 'screen-outputs', 'outdir'):
        followup.add_argument('--' + name, type=Path, required=True)
    followup.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        spec = json.loads(args.recipe_file.read_text())
        require(set(spec) == {'stage', 'resources', 'recipes'}, 'recipe file fields differ')
        freeze_plan(args.base_config, spec['resources'], spec['recipes'], spec['stage'], args.outdir)
    elif args.mode == 'audit':
        audit_results(args.plan, args.outputs, args.outdir)
    else:
        resources = json.loads(args.resources_file.read_text())
        freeze_followup(args.base_config, resources, args.screen_plan,
                        args.screen_outputs, args.seeds, args.outdir)


if __name__ == '__main__':
    main()
