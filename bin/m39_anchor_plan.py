#!/usr/bin/env python3
"""Bounded SELECT-only adaptation and immutable pre-scoring model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m39_anchor_screen import ARMS, VARIANTS, require, sha256, validate_config, write_json


def initial_plan(*, profile: bool = False) -> dict:
    cases = []
    for variant in VARIANTS:
        for width, radius in ((32, .05), (32, .2), (32, .5), (64, .2)):
            item = {'variant': variant, 'width': width, 'radius_cm': radius,
                    'learning_rate': .001, 'initial_epochs': 8, 'max_epochs': 16,
                    'person_batch': 4, 'locus_batch': 64, 'seed': 39062026}
            item['id'] = f'{variant}-w{width}-r{radius}-lr0.001'
            cases.append(item)
    if profile:
        cases = [cases[1]]
        cases[0] = {**cases[0], 'initial_epochs': 1, 'max_epochs': 1}
    return {'schema_version': 'm39-anchor-plan-v1', 'scope': 'exploratory_chr22_R0_fold0',
            'cases': cases, 'adaptive_max_cases': 0 if profile else 6,
            'selection_metric': 'SELECT_equal_person_equal_anchor_log_loss',
            'score_for_adaptation': False, 'prior_pseudocount': 1,
            'head': 'probability_mixture', 'arms': list(ARMS),
            'extension': '8_to_16_epochs_if_best_SELECT_epoch_in_last_three;patience4_after_epoch8',
            'adaptation': 'per_family_best_carrier_SELECT:LR0.0003_and0.003;'
                          'if_width64_beats32_at_radius0.2_add_width128_instead_of_LR0.003',
            'tie_break': 'lexicographic_case_id;earliest_checkpoint_on_equal_SELECT',
            'max_total_cases': 1 if profile else 18,
            'claims_excluded': ['chromosome_boundary_F1', 'new_NAM_donor_replication',
                                'confirmatory_TEST', 'all_rare_variants']}


def read_results(paths: list[Path]) -> list[tuple[Path, dict]]:
    results = []
    for path in paths:
        result = json.loads((path / 'training.receipt.json').read_text())
        require(result['score_read'] is False, 'training used score')
        require(tuple(r['arm'] for r in result['results']) == ARMS, 'incomplete arm set')
        validate_config(result['config'])
        results.append((path, result))
    require(len({r['config']['id'] for _, r in results}) == len(results), 'duplicate cases')
    require(len({r['input_sha256']['development'] for _, r in results}) == 1,
            'cases used different development partitions')
    require(len({json.dumps(r['source_sha256'], sort_keys=True) for _, r in results}) == 1,
            'cases used different training/model source code')
    return results


def loss(result: dict, arm: str) -> float:
    return next(item['select']['log_loss'] for item in result['results'] if item['arm'] == arm)


def adaptive_value(results: list[tuple[Path, dict]]) -> dict:
    require(len(results) == 12, 'adaptation requires all twelve initial configurations')
    expected = {c['id']: c for c in initial_plan()['cases']}
    require({r['config']['id']: r['config'] for _, r in results} == expected,
            'initial configurations differ from the frozen screen plan')
    extra, reasons = [], []
    for variant in VARIANTS:
        family = sorted([r for _, r in results if r['config']['variant'] == variant],
                        key=lambda r: (loss(r, 'carrier'), r['config']['id']))
        require(len(family) == 4, 'incomplete initial family search')
        best = dict(family[0]['config'])
        probe32 = next(r for r in family if r['config']['width'] == 32 and r['config']['radius_cm'] == .2)
        probe64 = next(r for r in family if r['config']['width'] == 64 and r['config']['radius_cm'] == .2)
        grow = loss(probe64, 'carrier') < loss(probe32, 'carrier')
        alternatives = [{**best, 'learning_rate': .0003}]
        alternatives.append({**best, 'width': 128, 'learning_rate': .001} if grow
                            else {**best, 'learning_rate': .003})
        for config in alternatives:
            config['id'] = (f"{variant}-w{config['width']}-r{config['radius_cm']}"
                            f"-lr{config['learning_rate']}")
            validate_config(config)
            extra.append(config)
        reasons.append({'variant': variant, 'best_initial': best['id'],
                        'w64_select_minus_w32': loss(probe64, 'carrier') - loss(probe32, 'carrier'),
                        'expanded_width_to128': grow})
    require(len({r['id'] for r in extra}) == 6, 'nonunique adaptive cases')
    return {'schema_version': 'm39-select-adaptation-v1', 'cases': extra,
            'reasons': reasons, 'score_read': False,
            'training_receipts': {p.name: sha256(p/'training.receipt.json') for p, _ in results}}


def adapt(paths: list[Path], output: Path) -> dict:
    value = adaptive_value(read_results(paths))
    write_json(output, value)
    return value


def lock(paths: list[Path], output: Path, binding_receipt: Path) -> dict:
    results = read_results(paths)
    require(len(results) == 18, 'selection requires all eighteen planned configurations')
    initial = {c['id']: c for c in initial_plan()['cases']}
    first = [(p, r) for p, r in results if r['config']['id'] in initial]
    extra = {c['id']: c for c in adaptive_value(first)['cases']}
    require({r['config']['id']: r['config'] for _, r in results} == initial | extra,
            'configuration inventory differs from prescribed SELECT adaptation')
    binding = json.loads(binding_receipt.read_text())
    require(binding['decision'] == 'PASS_EXPLORATORY_EXACT_ANCHOR_BINDING',
            'exact truth binding receipt required')
    require(all(r['input_sha256']['development'] == binding['outputs']['development.npz']['sha256']
                for _, r in results), 'training did not use authenticated binding development')
    winners = {arm: min(results, key=lambda item: (loss(item[1], arm), item[1]['config']['id']))[1]['config']['id']
               for arm in ARMS}
    families = {v: min([r for _, r in results if r['config']['variant'] == v],
                      key=lambda r: (loss(r, 'carrier'), r['config']['id']))['config']['id'] for v in VARIANTS}
    records = []
    for path, result in results:
        hashes = {arm: sha256(path/f'{arm}.pt') for arm in ARMS}
        for r in result['results']:
            require(hashes[r['arm']] == r['checkpoint_sha256'], 'checkpoint changed before selection')
        records.append({'id': result['config']['id'], 'config': result['config'],
                        'receipt_sha256': sha256(path/'training.receipt.json'),
                        'checkpoint_sha256': hashes,
                        'select_log_loss': {arm: loss(result, arm) for arm in ARMS}})
    value = {'schema_version': 'm39-anchor-selection-lock-v1', 'score_read': False,
             'scope': 'exploratory_selected_anchor_only', 'best_by_arm': winners,
             'binding_receipt_sha256': sha256(binding_receipt),
             'expected_score_sha256': binding['outputs']['score.npz']['sha256'],
             'best_carrier_by_family': families, 'cases': records,
             'rule': 'lowest_SELECT_loss_then_lexicographic_id',
             'claim_rule': 'carrier_must_improve_matched_pooled_and_best_pooled_and_Fminus;'
                           'trained_carrier_sham_and_replicates_needed_before_mechanistic_promotion'}
    write_json(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('initial', 'profile', 'adapt', 'lock'), required=True)
    parser.add_argument('--results', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binding-receipt', type=Path)
    args = parser.parse_args()
    if args.mode in ('initial', 'profile'):
        write_json(args.output, initial_plan(profile=args.mode == 'profile'))
    elif args.mode == 'adapt':
        adapt(args.results, args.output)
    else:
        require(args.binding_receipt is not None, 'binding receipt required before locking')
        lock(args.results, args.output, args.binding_receipt)


if __name__ == '__main__':
    main()
