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
import json
from pathlib import Path

import numpy as np

from m33_safe_bridge_core import write_exclusive_json
from m39_anchor_screen import metrics, sha256
from m39_ordered_gpu_manifest import ARMS, SCHEMA, SCOPE, STAGE_ARMS, load_plan, require
from m39_ordered_training import load_config
from m39_ordered_training_data import stratified_metrics


def freeze_plan(base_path: Path, resources: dict, recipes: list[dict], stage: str,
                outdir: Path) -> Path:
    """Freeze explicitly supplied values; no automatic parameter or anchor search."""
    base = load_config(base_path)
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
            load_config(path)
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


def _verified_case(path: Path, config_path: Path, receipt_sha: str) -> tuple[dict, dict]:
    receipt_path = path / 'training.receipt.json'
    require(sha256(receipt_path) == receipt_sha, 'worker/primary receipt hash differs')
    receipt = json.loads(receipt_path.read_text())
    cfg = load_config(config_path)
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
    require(metrics(arrays['probabilities'], arrays['truth_state']) == receipt['selected_SELECT_metrics'],
            'primary predictions do not reproduce selected metrics')
    require(stratified_metrics(arrays['probabilities'], arrays['truth_state'],
            arrays['query_carrier'], arrays['query_observed']) ==
            receipt['selected_SELECT_stratified_descriptive_only']['model'],
            'primary predictions do not reproduce descriptive strata')
    require(arrays['anchor_indices'].tolist() == receipt['anchor_indices'], 'receipt/prediction anchors differ')
    keys = [(row['SELECT']['brier'], row['SELECT']['log_loss'], row['step']) for row in receipt['curve']]
    require(min(keys)[2] == receipt['selected_step'], 'checkpoint selection is not declared SELECT minimum')
    for row in receipt['curve']:
        primary = json.loads((path / f"curve-step-{row['step']:07d}.json").read_text())
        require(primary == row, 'primary curve checkpoint differs')
    return receipt, arrays


def audit_results(plan_path: Path, outputs: Path, outdir: Path | None = None) -> dict:
    """Reopen all declared cases and same-axis controls before producing a comparison."""
    plan = load_plan(plan_path)
    if outdir is not None:
        require(not outdir.exists() and not outdir.is_symlink(), 'comparison output already exists')
    groups, rows, primary_hashes = [], [], {}
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
            cfg = load_config(cfg_path)
            receipt, arrays = _verified_case(group_path / cfg['arm'], cfg_path,
                                             done['case_receipt_sha256'][cfg['arm']])
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
        real = cases['real']['selected_SELECT_metrics']
        improvements = {arm: {metric: (np.asarray(receipt['selected_SELECT_metrics'][metric])
                                      - np.asarray(real[metric])).tolist()
                              for metric in ('brier', 'log_loss', 'dosage_mae')}
                        for arm, receipt in cases.items() if arm != 'real'}
        for name in ('Fminus_SELECT', 'Ffull_SELECT'):
            improvements[name] = {metric: (np.asarray(cases['real'][name][metric])
                                           - np.asarray(real[metric])).tolist()
                                  for metric in ('brier', 'log_loss', 'dosage_mae')}
        groups.append({'id': group['id'], 'family': cfg['model']['family'],
                       'learning_rate': cfg['learning_rate'], 'seed': cfg['seed'],
                       'REAL_metrics': real, 'COMMON_metrics': cases['common']['selected_SELECT_metrics'],
                       'descriptive_strata': {arm: receipt['selected_SELECT_stratified_descriptive_only']
                                              for arm, receipt in cases.items()},
                       'sham_TRAIN_dose_changes': cases.get('sham', {}).get('sham_TRAIN_dose_changes'),
                       'control_minus_REAL': improvements,
                       'positive_improvement_means': 'lower_error_for_REAL',
                       'paired_exposure': paired_reference})
    result = {'schema_version': 'm39-ordered-development-comparison-v1',
              'stage': plan['stage'], 'plan_sha256': sha256(plan_path),
              'primary_group_hashes': primary_hashes, 'groups': groups, 'cases': rows,
              'scope': {'exploratory_SELECT_reused_for_selection': plan['stage'] != 'technical_e2e',
                        'technical_e2e_only': plan['stage'] == 'technical_e2e',
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


def selected_learning_rates(summary: dict) -> dict:
    """Keep both families; choose LR by mean COMMON/REAL error, never rare improvement."""
    require(summary['stage'] == 'exploratory_screen', 'LR selection requires screening outputs')
    chosen = {}
    for family in ('cnn', 'attention'):
        groups = [group for group in summary['groups'] if group['family'] == family]
        require(len(groups) >= 2 and len({group['learning_rate'] for group in groups}) == len(groups),
                'incomplete or duplicated learning-rate comparison')
        def objective(group, metric):
            return (group['REAL_metrics'][metric] + group['COMMON_metrics'][metric]) / 2
        winner = min(groups, key=lambda group: (objective(group, 'brier'),
            objective(group, 'log_loss'), group['learning_rate']))
        chosen[family] = {'learning_rate': winner['learning_rate'], 'source_group': winner['id'],
                          'selection_metric': 'minimum_mean_COMMON_REAL_SELECT_Brier_then_mean_log_loss_then_lower_LR',
                          'mean_COMMON_REAL_brier': objective(winner, 'brier'),
                          'does_not_establish_incremental_rare_value': True}
    return chosen


def freeze_followup(base_path: Path, resources: dict, screen_plan: Path,
                    screen_outputs: Path, seeds: list[int], outdir: Path) -> Path:
    """Audit screening primaries, retain both families and freeze fresh paired replicas."""
    require(len(seeds) >= 2 and len(set(seeds)) == len(seeds)
            and all(type(seed) is int and 0 <= seed < 2**63 - 100 for seed in seeds),
            'at least two distinct valid followup seeds are required')
    summary = audit_results(screen_plan, screen_outputs)
    require(not set(seeds) & {row['seed'] for row in summary['groups']},
            'followup seeds must differ from screening')
    chosen = selected_learning_rates(summary)
    recipes = [dict(id=f'{family}-seed{seed}', family=family,
                    learning_rate=chosen[family]['learning_rate'], seed=seed, pair_seed=seed + 100)
               for family in ('cnn', 'attention') for seed in seeds]
    path = freeze_plan(base_path, resources, recipes, 'controlled_followup', outdir)
    write_exclusive_json(outdir / 'screen-primary-audit.json', summary)
    write_exclusive_json(outdir / 'selection.receipt.json', {
        'selected_learning_rates': chosen, 'screen_plan_sha256': sha256(screen_plan),
        'screen_primary_audit_sha256': sha256(outdir / 'screen-primary-audit.json'),
        'followup_plan_sha256': sha256(path), 'SCORE_opened': False,
        'selection_generalizes_across_radii': 'unproven_fixed_budget_transfer',
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
