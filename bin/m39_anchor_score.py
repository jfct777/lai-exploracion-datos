#!/usr/bin/env python3
"""Score frozen M39 checkpoints once on a physically separate anchor partition."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from m39_anchor_screen import (ARMS, CarrierContextModel, load_inputs, metrics,
                               predict, require, sha256, write_json)


def paired_summary(candidate: dict, comparator: dict, seed: int = 39062027) -> dict:
    delta = (np.asarray(candidate['log_loss_per_person']) -
             np.asarray(comparator['log_loss_per_person']))
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    return {'delta_log_loss': float(delta.mean()),
            'conditional_person_bootstrap_95': np.quantile(boot, [.025, .975]).tolist(),
            'people_improved': int((delta < 0).sum()), 'people': len(delta),
            'delta_per_person': delta.tolist(), 'bootstrap_draws': len(boot),
            'scope': 'conditional_fixed_donor_library_not_NAM_population_interval'}


def score_locked(lock_path: Path, cases: list[Path], features: list[Path],
                 score_path: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=False)
    lock = json.loads(lock_path.read_text())
    require(lock['schema_version'] == 'm39-anchor-selection-lock-v1' and not lock['score_read'],
            'pre-SCORE selection lock required')
    paths = {p.name: p for p in cases}
    require(set(paths) == {r['id'] for r in lock['cases']}, 'lock/case set mismatch')
    radii = {}
    for p in features:
        with np.load(p, allow_pickle=False) as archive:
            radius = float(archive['radius_cm'][0])
        require(radius not in radii, 'duplicate feature radius')
        radii[radius] = p
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    results, records = {}, []
    baseline_metrics = full_metrics = None
    for case in lock['cases']:
        folder = paths[case['id']]
        require(sha256(folder/'training.receipt.json') == case['receipt_sha256'],
                'receipt changed after selection lock')
        training = json.loads((folder/'training.receipt.json').read_text())
        for name, expected in training['source_sha256'].items():
            require(sha256(Path(__file__).with_name(name)) == expected,
                    'scoring implementation differs from frozen training source')
        feature_path = radii[case['config']['radius_cm']]
        require(sha256(feature_path) == training['input_sha256']['features'], 'feature bytes changed')
        for arm in ARMS:
            checkpoint = folder / f'{arm}.pt'
            require(sha256(checkpoint) == case['checkpoint_sha256'][arm], 'checkpoint changed after lock')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(saved['config'] == case['config'] and saved['arm'] == arm, 'checkpoint identity differs')
            batch, labels, data, _, _, _, _ = load_inputs(
                feature_path, score_path, fit=False,
                mean=np.asarray(saved['normalization_mean']), scale=np.asarray(saved['normalization_scale']))
            require('train_indices' not in data and 'select_indices' not in data,
                    'SCORE payload must not include training roles')
            model = CarrierContextModel(**saved['model_kwargs'])
            model.load_state_dict(saved['state_dict'])
            probabilities = predict(model, batch, np.arange(len(labels)), arm,
                                    case['config']['person_batch'], case['config']['locus_batch'])
            evaluated = metrics(probabilities, labels.numpy())
            if baseline_metrics is None:
                baseline_metrics = metrics(data['baseline'], labels.numpy())
                full_metrics = metrics(data['full_baseline'], labels.numpy())
            key = f"{case['id']}/{arm}"
            results[key] = evaluated
            output = outdir / f"{case['id']}__{arm}.npz"
            np.savez_compressed(output, probabilities=probabilities, truth_state=labels.numpy(),
                                sample_key_sha256=data['sample_key_sha256'], coords=data['coords'])
            records.append({'case': case['id'], 'arm': arm, 'log_loss': evaluated['log_loss'],
                            'brier': evaluated['brier'], 'accuracy': evaluated['accuracy'],
                            'selected_by_arm': case['id'] == lock['best_by_arm'][arm],
                            'predictions_sha256': sha256(output),
                            'checkpoint_sha256': sha256(checkpoint)})
    chosen = lock['best_by_arm']['carrier']
    carrier = results[f'{chosen}/carrier']
    contrasts = {
        'carrier_minus_matched_pooled': paired_summary(carrier, results[f'{chosen}/pooled']),
        'carrier_minus_best_pooled': paired_summary(carrier, results[f"{lock['best_by_arm']['pooled']}/pooled"]),
        'carrier_minus_best_common': paired_summary(carrier, results[f"{lock['best_by_arm']['common']}/common"]),
        'carrier_minus_Fminus': paired_summary(carrier, baseline_metrics),
        'carrier_minus_Ffull': paired_summary(carrier, full_metrics),
    }
    promising = all(contrasts[k]['delta_log_loss'] < 0 for k in
                   ('carrier_minus_matched_pooled', 'carrier_minus_best_pooled', 'carrier_minus_Fminus'))
    report = {'schema_version': 'm39-anchor-score-v1', 'selection_lock_sha256': sha256(lock_path),
              'score_sha256': sha256(score_path), 'scope': 'exploratory_chr22_R0_S660_historical_fold0',
              'selected': lock['best_by_arm'], 'best_carrier_by_family': lock['best_carrier_by_family'],
              'Fminus': baseline_metrics, 'Ffull': full_metrics, 'contrasts': contrasts,
              'all_results': results, 'records': records,
              'status': 'EXPLORATORY_CANDIDATE_NEEDS_TRAINED_SHAM_AND_REPLICATION' if promising
                        else 'NO_INCREMENT_IN_THIS_SELECTED_ANCHOR_SCREEN',
              'global_boundary_F1_evaluated': False, 'trained_carrier_sham_evaluated': False,
              'source_test_or_valid_opened': False, 'independent_NAM_donor_units': 2,
              'neither_status_closes_all_models_or_all_rare_variants': True}
    write_json(outdir/'score.receipt.json', report)
    with (outdir/'metrics.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', type=Path, required=True)
    parser.add_argument('--features', nargs='+', type=Path, required=True)
    parser.add_argument('--score', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    report = score_locked(args.lock, args.cases, args.features, args.score, args.outdir)
    print(json.dumps({'status': report['status'], 'selected': report['selected']}))


if __name__ == '__main__':
    main()
