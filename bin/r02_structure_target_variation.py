#!/usr/bin/env python3
"""FIT-only descriptive target variation; no noise ceiling or training admission."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

FIELDS = {'person_id', 'component_id', 'role', 'pairs', 'target',
          'pair_weight', 'joint_marker_count'}


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def authenticate(entry):
    path = Path(entry['path'])
    require(path.is_file() and sha(path) == entry['sha256'], 'Changed source: ' + str(path))
    return path


def mean(x, w):
    return float(np.dot(x, w) / w.sum())


def describe_prediction(y, x, w):
    """An identity prediction and an in-sample weighted affine description."""
    my, mx = mean(y, w), mean(x, w)
    yc, xc = y - my, x - mx
    vy, vx = mean(yc * yc, w), mean(xc * xc, w)
    cov = mean(yc * xc, w)
    slope = cov / vx if vx > 0 else 0.0
    affine_error = mean((yc - slope * xc) ** 2, w)
    direct_error = mean((y - x) ** 2, w)
    return {'target_mean': my, 'target_variance': vy, 'target_sd': float(np.sqrt(vy)),
            'loco_mean': mx, 'loco_variance': vx,
            'correlation': cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else None,
            'identity_mse': direct_error,
            'identity_r2': 1 - direct_error / vy if vy > 0 else None,
            'affine_in_sample_intercept': my - slope * mx,
            'affine_in_sample_slope': slope,
            'affine_in_sample_mse': affine_error,
            'affine_in_sample_r2': 1 - affine_error / vy if vy > 0 else None,
            'affine_residual_variance_fraction': affine_error / vy if vy > 0 else None}


def loco(y, counts, chromosomes, chromosome):
    keep = np.asarray(chromosomes) != chromosome
    require(keep.any(), 'LOCO has no remaining chromosome')
    # Counts reconstruct a marker-pooled response. They NEVER multiply pair loss.
    n = counts[keep].sum(axis=0, dtype=np.float64)
    require(np.all(n > 0), 'LOCO response lacks observed support')
    return np.einsum('ij,ij->j', y[keep], counts[keep], dtype=np.float64) / n


def weighted_covariance(y, w, chunk):
    mu = y @ w / w.sum()
    cov = np.zeros((len(y), len(y)), dtype=np.float64)
    for start in range(0, len(w), chunk):
        stop = start + chunk
        z = y[:, start:stop] - mu[:, None]
        cov += (z * w[start:stop]) @ z.T
    return mu, cov / w.sum()


def run(plan_path, output):
    started = time.monotonic()
    plan = json.loads(Path(plan_path).read_text())
    require(plan['stage'] == 'FIT' and plan['read_select'] is False and
            plan['read_evaluate'] is False, 'Diagnostic must remain FIT-only')
    index = json.loads(authenticate(plan['index']).read_text())
    blocks_path = authenticate(plan['blocks'])
    with blocks_path.open(newline='') as stream:
        blocks = {r['block_id']: r for r in csv.DictReader(stream, delimiter='\t')}
    require({int(r['chromosome']) for r in blocks.values()} == set(range(1, 23)),
            'Expected exactly 22 autosomes')
    for chromosome in range(1, 23):
        intervals = sorted((int(r['start_bp']), int(r['end_bp'])) for r in blocks.values()
                           if int(r['chromosome']) == chromosome)
        require(len(intervals) == 2 and all(a > 0 and a <= b for a, b in intervals)
                and intervals[0][1] < intervals[1][0], 'Target chromosome halves overlap or differ')
    views = {}
    for task in index['tasks']:
        view = task['view_id']
        entry = task['targets']['FIT']
        if view in views:
            require(views[view]['targets']['FIT'] == entry, 'Contexts disagree on FIT target')
        else:
            views[view] = task
    require(len(views) == 88 and len(blocks) == 44, 'Expected fixed 88 views/44 regions')
    groups = {}
    sources = []
    for view, task in sorted(views.items()):
        entry = task['targets']['FIT']
        path = authenticate(entry)
        receipt_path = authenticate({'path': entry['receipt_path'], 'sha256': entry['receipt_sha256']})
        receipt = json.loads(receipt_path.read_text())
        require(receipt['role'] == 'FIT' and receipt['arrays_sha256'] == entry['sha256'] and
                receipt['block_id'] == task['block_id'] and
                receipt['exclusion_scope'] == task['exclusion_scope'] and
                receipt['evaluate_targets_computed'] is False,
                'Receipt identity/role differs')
        groups.setdefault(task['block_id'], []).append(task)
        sources.append({'view_id': view, 'path': str(path), 'sha256': entry['sha256'],
                        'receipt_sha256': entry['receipt_sha256']})
    require(set(groups) == set(blocks), 'Target and region identities differ')
    ids = sorted(groups)
    first = None
    y = counts = None
    duplicates = []
    for b, block_id in enumerate(ids):
        tasks = groups[block_id]
        require(len(tasks) == 2 and {t['exclusion_scope'] for t in tasks} ==
                {'BLOCK_AND_LD_MARGIN', 'WHOLE_CHROMOSOME'}, 'Two exclusion views required')
        entries = [t['targets']['FIT'] for t in tasks]
        require(entries[0]['sha256'] == entries[1]['sha256'],
                'Scope targets differ: do not deduplicate or conceal the discrepancy')
        with np.load(entries[0]['path'], allow_pickle=False) as z:
            require(set(z.files) == FIELDS, 'Target field inventory differs')
            data = {name: z[name] for name in z.files}
        require(set(data['role'].astype(str)) == {'FIT'} and
                len(data['person_id']) == plan['fit_people'], 'FIT person roster differs')
        n = len(data['target'])
        require(n <= plan['max_pairs'] and data['pairs'].shape == (n, 2) and
                data['pair_weight'].shape == data['joint_marker_count'].shape == (n,) and
                np.isfinite(data['target']).all() and np.isfinite(data['pair_weight']).all() and
                np.all(data['pair_weight'] > 0) and np.all(data['joint_marker_count'] > 0),
                'Target dimensions/finite values/support differ')
        if first is None:
            first = {k: data[k] for k in ['person_id', 'component_id', 'role', 'pairs', 'pair_weight']}
            y = np.empty((44, n), dtype=np.float64)
            counts = np.empty((44, n), dtype=np.int64)
            require(y.nbytes + counts.nbytes < plan['max_array_bytes'], 'Array memory bound exceeded')
            pairs = first['pairs']
            require(np.all(pairs[:, 0] < pairs[:, 1]) and np.all(pairs >= 0) and
                    np.all(pairs < plan['fit_people']), 'Invalid pair indices')
            _, ci = np.unique(first['component_id'], return_inverse=True)
            a, c = ci[pairs[:, 0]], ci[pairs[:, 1]]
            require(np.all(a != c), 'Within-component pair present')
            labels = np.minimum(a, c) * len(first['person_id']) + np.maximum(a, c)
            frequency = np.bincount(labels)
            expected = 1.0 / frequency[labels]; expected /= expected.mean()
            require(np.allclose(first['pair_weight'], expected, rtol=1e-13, atol=0),
                    'Component-pair loss weights differ')
            del a, c, ci, labels, frequency, expected
        else:
            for name, value in first.items():
                require(value.dtype == data[name].dtype and np.array_equal(value, data[name]),
                        'Nonidentical pair/person/weight axes: stop; no silent intersection')
        y[b], counts[b] = data['target'], data['joint_marker_count']
        duplicates.append({'block_id': block_id, 'view_ids': [t['view_id'] for t in tasks],
                           'exact_same_npz_sha256': entries[0]['sha256']})
        del data
    w = first['pair_weight']
    chromosomes = [int(blocks[b]['chromosome']) for b in ids]
    means, covariance = weighted_covariance(y, w, plan['pair_chunk'])
    sds = np.sqrt(np.diag(covariance))
    divisor = sds[:, None] * sds[None, :]
    correlations = np.divide(covariance, divisor, out=np.zeros_like(covariance), where=divisor > 0)
    rows = []
    for chrom in sorted(set(chromosomes)):
        x = loco(y, counts, chromosomes, chrom)
        require(sum(c != chrom for c in chromosomes) == 42, 'LOCO must use other 21 chromosomes')
        for b in [i for i, c in enumerate(chromosomes) if c == chrom]:
            row = {'block_id': ids[b], 'chromosome': chrom, 'pairs': len(w),
                   'width_bp': int(blocks[ids[b]]['end_bp']) - int(blocks[ids[b]]['start_bp']) + 1,
                   'joint_marker_count_min': int(counts[b].min()),
                   'joint_marker_count_max': int(counts[b].max()),
                   'target_min': float(y[b].min()), 'target_max': float(y[b].max()),
                   **describe_prediction(y[b], x, w)}
            rows.append(row)
    rows.sort(key=lambda r: r['block_id'])
    # Exact variation decomposition after removing each block's weighted mean.
    centered_pair_average = (y - means[:, None]).mean(axis=0)
    common_variance = mean(centered_pair_average ** 2, w)
    average_variance = float(np.trace(covariance) / len(ids))
    specific_variance = sum(mean((y[i] - means[i] - centered_pair_average) ** 2, w)
                            for i in range(len(ids))) / len(ids)
    require(np.isclose(average_variance, common_variance + specific_variance, rtol=1e-11, atol=1e-14),
            'Variance decomposition failed')
    result = {'schema': 'r02_structure_target_variation_v1', 'status': 'FIT_DESCRIPTIVE_AUDIT_COMPLETE',
              'plan_sha256': sha(plan_path), 'implementation_sha256': sha(__file__),
              'stage': 'FIT', 'select_read': False, 'evaluate_read': False, 'training_started': False,
              'fit_people': len(first['person_id']), 'fit_components': len(set(first['component_id'].tolist())),
              'pairs': len(w), 'component_pair_weight_sum': float(w.sum()),
              'sources': sources, 'scope_duplicates': duplicates,
              'block_order': ids, 'block_results': rows,
              'weighted_block_correlations': [[float(correlations[i, j]) if divisor[i, j] > 0 else None
                                               for j in range(len(ids))] for i in range(len(ids))],
              'width_bp': {'mean': float(np.mean([r['width_bp'] for r in rows])),
                           'min': min(r['width_bp'] for r in rows), 'max': max(r['width_bp'] for r in rows)},
              'variation_decomposition': {'mean_block_variance': average_variance,
                  'shared_pair_mean_variance': common_variance, 'block_specific_variance': specific_variance,
                  'raw_within_pair_variance': specific_variance + float(np.var(means)),
                  'shared_fraction': common_variance / average_variance if average_variance else None},
              'runtime': {'elapsed_seconds': time.monotonic() - started,
                          'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
              'interpretation_limits': ['No Bayes/noise ceiling is identified by these files.',
                  'Affine coefficients and their performance use the same FIT pairs: descriptive, not generalization.',
                  'Pairs sharing people/components and blocks are dependent; no iid standard errors or p-values.',
                  'LOCO pools observed-marker counts of other chromosomes, not the pair-loss weights.',
                  '44 regions and 88 input exclusion views are not 88 independent outcomes.',
                  'No rescaling, target replacement, locus selection or biological winner is authorized by this audit.']}
    out = Path(output)
    require(not out.exists(), 'Refusing to overwrite an audit result')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'blocks': len(rows), 'runtime': result['runtime']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.plan, a.output)
