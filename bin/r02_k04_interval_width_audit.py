#!/usr/bin/env python3
"""Describe authenticated saved FIT intervals; no fit, phase or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def describe(lower, upper, primal):
    if (lower.ndim != 1 or not len(lower) or upper.shape != lower.shape or
            not np.isfinite(lower).all() or not np.isfinite(upper).all() or
            not np.all((0 <= lower) & (lower <= upper) & (upper <= 1)) or
            not np.isfinite(primal)):
        raise ValueError('Finite nonempty ordered rate intervals required')
    width = upper - lower
    floor = float(np.mean((width / 2)**2))
    guard = 128 * np.finfo(float).eps * max(1., abs(primal))
    if primal + guard < floor:
        raise ValueError('Primal below interval loss floor; source/objective mismatch')
    levels = [0., .25, .5, .75, 1.]
    return dict(n_pairs=len(lower), exactly_identified_pairs=int(np.count_nonzero(width == 0)),
        ambiguous_pairs=int(np.count_nonzero(width > 0)), quantile_levels=levels,
        quantile_method='linear', width_quantiles=np.quantile(width, levels).tolist(),
        midpoint_quantiles=np.quantile((lower+upper)/2, levels).tolist(),
        mean_width=float(np.mean(width)), minimum_possible_mean_interval_loss=floor,
        retained_primal=primal, primal_above_interval_floor=primal-floor,
        loss_floor_fraction_of_primal=(floor/primal if primal > 0 else None),
        weighting='EQUAL_PAIRS_AS_SAVED_OBJECTIVE',
        floor_attainability='NOT_CERTIFIED',
        midpoint_is_observed_truth=False, confidence_interval=False,
        optimization_convergence_measured=False, biological_validation=False)


def run(request, output, request_sha256):
    started = time.monotonic()
    if sha(request) != request_sha256:
        raise ValueError('Frozen request differs')
    spec = json.loads(Path(request).read_text())
    if spec['role'] != 'FIT' or spec['operation'] != 'SAVED_INTERVAL_WIDTH_DIAGNOSTIC':
        raise ValueError('Explicit FIT-only descriptive scope required')
    if sha(__file__) != spec['code_sha256']:
        raise ValueError('Frozen implementation differs')
    audit = spec['pre_check']
    if sha(audit['path']) != audit['sha256']:
        raise ValueError('PRE receipt changed')
    review = json.loads(Path(audit['path']).read_text())
    if review['phase'] != 'PRE' or not review['pre_review_checks_passed']:
        raise ValueError('Prospective documented review required')
    previous = spec['source_evidence']
    if sha(previous['path']) != previous['sha256']:
        raise ValueError('Previous objective evidence differs')
    previous_result = json.loads(Path(previous['path']).read_text())
    if previous_result['primal_objective'] != spec['retained_primal']:
        raise ValueError('Retained primal differs from authenticated result')
    arrays = {}
    for name in ('lower', 'upper'):
        source = spec[name]
        p = Path(source['path'])
        prior = next(e for e in previous_result['evidence'] if e['role'] == name)
        if p.parent.name != 'FIT' or prior['sha256'] != source['sha256']:
            raise ValueError('FIT interval identity differs from prior result')
        if p.stat().st_size != source['bytes'] or sha(p) != source['sha256']:
            raise ValueError('Saved FIT interval source differs: '+name)
        arrays[name] = np.load(p, mmap_mode='r', allow_pickle=False)
    result = describe(arrays['lower'], arrays['upper'], spec['retained_primal'])
    result.update(scope=spec['scope'], source_evidence=spec['source_evidence'],
        request_sha256=sha(request), code_sha256=sha(__file__),
        elapsed_seconds=time.monotonic()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        optimizer_iterations=0, matrix_products=0, select_opened=False, evaluate_opened=False)
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    p.add_argument('--request-sha256', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    signal.alarm(120)
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    print(json.dumps(run(a.request, a.output, a.request_sha256), allow_nan=False))
