#!/usr/bin/env python3
"""Bounded probability-only calibration of the historical M39 anchor screen.

Temperature scaling follows Guo et al., ICML 2017, section 4; positive mixtures
are a separate adaptation for baseline zeros. Only TRAIN fits parameters/prior;
SELECT chooses among five frozen families. SCORE is an already-observed R0
partition, not a new holdout. No features, genotypes or checkpoints are loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

SCOPE = 'historical_R0_exploratory_post_hoc'
STATE_NAMES = ('AA', 'AE', 'AN', 'EE', 'EN', 'NN')
DOSAGES = np.asarray(((2, 0, 0), (1, 1, 0), (1, 0, 1),
                      (0, 2, 0), (0, 1, 1), (0, 0, 2)))
FAMILIES = ('identity', 'temperature', 'uniform_mix', 'prior_mix', 'temperature_prior_mix')
DIMENSIONS = dict(zip(FAMILIES, (0, 1, 1, 1, 2)))
AXES = ('chrom', 'pos', 'ref', 'alt', 'coords', 'locus_id', 'anchor_indices', 'state_names')
PAYLOAD = set(AXES) | {'sample_key_sha256', 'source_indices', 'baseline', 'full_baseline', 'truth_state'}
PLAN_FIELDS = {'schema_version', 'scope', 'development_sha256', 'score_sha256',
               'comparators_manifest_sha256', 'temperature_grid', 'alpha_grid',
               'prior_pseudocount', 'floor', 'bootstrap_replicates', 'bootstrap_seed',
               'reliability_bins', 'temperature_endpoints', 'tie_tolerance'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), 'empty CSV output')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_hash(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def authenticated(path: Path, expected: str) -> Path:
    path = Path(path)
    require(valid_hash(expected) and sha256(path) == expected, f'input SHA-256 differs: {path.name}')
    return path


def read_npz(path: Path, allowed: set[str]) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == allowed and len(archive.files) == len(allowed),
                'NPZ inventory differs; genetic features/extra roles are not permitted')
        return {key: archive[key].copy() for key in archive.files}


def validate_probabilities(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values)
    require(p.ndim >= 2 and p.shape[-1] == 6 and p.size > 0 and p.dtype.kind in 'fi',
            'six-state numeric probability array required')
    require(np.isfinite(p).all() and np.all(p >= 0) and np.all(p <= 1), 'invalid probabilities')
    require(np.allclose(p.astype(np.float64).sum(-1), 1, atol=5e-5, rtol=0),
            'probability simplex differs')
    return p


def validate_labels(labels: np.ndarray, shape: tuple) -> np.ndarray:
    y = np.asarray(labels)
    require(y.shape == shape and y.dtype.kind in 'iu' and np.all((y >= 0) & (y < 6)),
            'truth must be integer six-state labels on the probability axes')
    return y.astype(np.int64)


def array_digest(values: np.ndarray) -> str:
    x = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((x.dtype.str, x.shape)).encode())
    digest.update(x.tobytes())
    return digest.hexdigest()


def validate_data(data: dict, development: bool) -> tuple[np.ndarray, np.ndarray]:
    p = validate_probabilities(data['baseline'])
    require(p.ndim == 3, 'baseline must be person by anchor by state')
    n, anchors, _ = p.shape
    require(n > 0 and anchors > 0, 'empty person or anchor axis')
    validate_labels(data['truth_state'], (n, anchors))
    require(validate_probabilities(data['full_baseline']).shape == p.shape, 'Ffull dimensions differ')
    require(tuple(data['state_names'].astype(str)) == STATE_NAMES, 'state order differs')
    keys = data['sample_key_sha256']
    require(keys.shape == (n,) and keys.dtype == np.dtype('|S64') and
            len(np.unique(keys)) == n and all(valid_hash(k.decode()) for k in keys),
            'sample hashes are invalid or duplicated')
    source = data['source_indices']
    require(source.shape == (n,) and source.dtype.kind in 'iu' and np.all(source >= 0) and
            len(np.unique(source)) == n, 'source person indices are invalid or duplicated')
    for name in AXES[:-1]:
        require(data[name].shape == (anchors,), f'{name} anchor dimensions differ')
    for name in ('chrom', 'pos', 'anchor_indices'):
        require(data[name].dtype.kind in 'iu', f'{name} must be integer')
    require(np.all(data['chrom'] == 22) and np.all(data['pos'] > 0) and
            np.all(np.diff(data['pos'].astype(np.int64)) >= 0), 'physical anchor axis differs')
    require(np.array_equal(data['anchor_indices'], np.arange(anchors)), 'anchor indices differ')
    require(data['coords'].dtype.kind in 'fi' and np.isfinite(data['coords']).all() and
            np.all(data['coords'] >= 0) and np.all(np.diff(data['coords']) >= 0), 'genetic axis differs')
    for name in ('ref', 'alt'):
        require(data[name].dtype.kind in 'SU' and np.all(data[name].astype(str) != ''),
                f'{name} identities are invalid')
    require(data['locus_id'].dtype == np.dtype('uint64'), 'locus identities must be uint64 binder hashes')
    require(len(np.unique(data['locus_id'])) == anchors, 'duplicate locus identities')
    names = ('train_indices', 'select_indices') if development else ('score_indices',)
    roles = [data[name] for name in names]
    require(all(x.ndim == 1 and x.size > 0 and x.dtype.kind in 'iu' for x in roles),
            'role indices must be nonempty integer vectors')
    require(np.array_equal(np.sort(np.concatenate(roles)), np.arange(n)),
            'roles must disjointly exhaust partition people')
    if not development:
        require(np.array_equal(roles[0], np.arange(n)), 'SCORE indices must be in exact person order')
        return roles[0].astype(np.int64), np.empty(0, dtype=np.int64)
    return tuple(x.astype(np.int64) for x in roles)


def validate_plan(plan: dict) -> None:
    require(set(plan) == PLAN_FIELDS and plan['schema_version'] == 'm39-calibration-plan-v1'
            and plan['scope'] == SCOPE, 'plan schema/scope differs')
    require(all(valid_hash(plan[k]) for k in ('development_sha256', 'score_sha256',
                                            'comparators_manifest_sha256')), 'invalid plan input hash')
    require(plan['floor'] == 1e-12 and plan['prior_pseudocount'] == 1 and
            plan['tie_tolerance'] == 1e-12, 'M39 loss/prior/tie contract differs')
    for key in ('temperature_grid', 'alpha_grid', 'temperature_endpoints'):
        values = plan[key]
        require(isinstance(values, list) and len(values) >= 2 and
                all(type(v) in (float, int) and math.isfinite(v) for v in values) and
                values == sorted(set(values)), f'{key} must be finite, unique and increasing')
    ts, alphas, ends = plan['temperature_grid'], plan['alpha_grid'], plan['temperature_endpoints']
    require(0 < ends[0] < ts[0] and ts[-1] < ends[-1] and len(ends) == 2 and
            1 in ts and ends[0] >= .125 and ends[-1] <= 32 and len(ts) <= 15,
            'temperature grid/endpoints exceed bounded contract')
    require(alphas[0] == 0 and alphas[-1] == 1 and len(alphas) <= 25, 'mixture grid differs')
    require(type(plan['bootstrap_replicates']) is int and plan['bootstrap_replicates'] == 10000 and
            type(plan['bootstrap_seed']) is int and 0 <= plan['bootstrap_seed'] < 2**32,
            'bootstrap contract differs')
    require(plan['reliability_bins'] == [5, 10, 20], 'fixed reliability bins differ')


def train_prior(labels: np.ndarray, pseudocount: float = 1.) -> np.ndarray:
    y = validate_labels(labels, np.asarray(labels).shape)
    require(y.size > 0 and math.isfinite(pseudocount) and pseudocount > 0, 'invalid TRAIN prior')
    counts = np.bincount(y.ravel(), minlength=6).astype(np.float64) + pseudocount
    return counts / counts.sum()


def transform(probabilities: np.ndarray, family: str, temperature: float = 1.,
              alpha: float = 0., prior: np.ndarray | None = None) -> np.ndarray:
    """Apply a global map; identity/T=1,a=0 are exact copies, never an epsilon floor.

    Temperature retains exact input zeros. Positive outputs smaller than float64
    can represent are saturated at its smallest subnormal, not the scoring floor.
    """
    p = validate_probabilities(probabilities)
    require(family in FAMILIES and math.isfinite(temperature) and temperature > 0 and
            math.isfinite(alpha) and 0 <= alpha <= 1, 'invalid calibration parameters')
    require(family != 'identity' or (temperature == 1 and alpha == 0), 'identity parameters differ')
    require(family in ('temperature', 'temperature_prior_mix') or temperature == 1,
            'temperature is not a parameter of this family')
    require(family not in ('identity', 'temperature') or alpha == 0, 'mixture not allowed in family')
    if temperature == 1 and alpha == 0:
        return p.copy()
    q = p.astype(np.float64)
    q /= q.sum(-1, keepdims=True)
    if temperature != 1:
        positive = p > 0
        logs = np.full(q.shape, -np.inf)
        np.log(p.astype(np.float64), out=logs, where=positive)
        logs -= logs.max(-1, keepdims=True)
        with np.errstate(over='ignore'):
            logs /= temperature
        tiny = np.nextafter(0., 1.)
        q = np.where(positive, np.exp(np.maximum(logs, np.log(tiny))), 0.)
        q /= q.sum(-1, keepdims=True)
        q = np.where(positive, np.maximum(q, tiny), 0.)
    if alpha:
        target = np.full(6, 1 / 6) if family == 'uniform_mix' else np.asarray(prior, dtype=np.float64)
        require(target.shape == (6,) and np.isfinite(target).all() and np.all(target > 0)
                and np.isclose(target.sum(), 1., atol=1e-14, rtol=0), 'positive TRAIN prior required')
        q = (1 - alpha) * q + alpha * target
        require(np.all(q > 0), 'positive mixture underflowed')
    validate_probabilities(q)
    return q


def true_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = validate_probabilities(probabilities).astype(np.float64)
    y = validate_labels(labels, p.shape[:-1])
    require(y.ndim == 2, 'metrics require person by anchor labels')
    p /= p.sum(-1, keepdims=True)
    return p, np.take_along_axis(p, y[..., None], -1)[..., 0]


def evaluate(probabilities: np.ndarray, labels: np.ndarray, floor: float = 1e-12) -> dict:
    """Same equal-person/equal-anchor M39 metrics, with raw-loss diagnostics.

    Deliberately NumPy-only: importing m39_anchor_screen also imports Torch and
    its trainable model. Regression tests compare the formulas independently.
    """
    require(math.isfinite(floor) and 0 < floor < 1, 'invalid evaluation floor')
    p, true = true_probabilities(probabilities, labels)
    labels = np.asarray(labels, dtype=np.int64)
    ll = -np.log(np.maximum(true, floor))
    brier = ((p - np.eye(6)[labels]) ** 2).sum(-1)
    dosage = np.abs(p @ DOSAGES - DOSAGES[labels])
    correct = p.argmax(-1) == labels
    raw_infinite = bool((true == 0).any())
    return {'log_loss': float(ll.mean()), 'raw_log_loss': None if raw_infinite else float(-np.log(true).mean()),
            'raw_infinite': raw_infinite, 'brier': float(brier.mean()),
            'accuracy': float(correct.mean()), 'dosage_mae': dosage.mean((0, 1)).tolist(),
            'log_loss_per_person': ll.mean(1).tolist(), 'brier_per_person': brier.mean(1).tolist(),
            'accuracy_per_person': correct.mean(1).tolist(),
            'dosage_mae_per_person': dosage.mean(1).tolist(),
            'true_zero_count': int((true == 0).sum()), 'true_below_floor_count': int((true < floor).sum()),
            'fraction_true_probability_below_floor': float((true < floor).mean()),
            'probability_zero_count': int((p == 0).sum()), 'people': p.shape[0], 'anchors': p.shape[1],
            'observations': true.size, 'floor': floor, 'weighting': 'equal_people_equal_selected_anchors'}


def reliability(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> tuple[list[dict], float]:
    p, _ = true_probabilities(probabilities, labels)
    require(type(bins) is int and bins > 0, 'positive integer bin count required')
    confidence, correct = p.max(-1).ravel(), (p.argmax(-1) == labels).ravel()
    membership = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    rows, ece = [], 0.
    for b in range(bins):
        used = membership == b
        count = int(used.sum())
        mean_confidence = float(confidence[used].mean()) if count else None
        accuracy = float(correct[used].mean()) if count else None
        if count:
            ece += count / confidence.size * abs(mean_confidence - accuracy)
        rows.append({'bins': bins, 'bin': b, 'lower': b / bins, 'upper': (b + 1) / bins,
                     'right_inclusive': b == bins - 1, 'count': count, 'total': confidence.size,
                     'people_with_observations': int(used.reshape(p.shape[:2]).any(1).sum()),
                     'mean_confidence': mean_confidence, 'accuracy': accuracy})
    return rows, float(ece)


def strata_metrics(probabilities: np.ndarray, original: np.ndarray, labels: np.ndarray, floor: float) -> list[dict]:
    _, baseline_true = true_probabilities(original, labels)
    _, true = true_probabilities(probabilities, labels)
    loss = -np.log(np.maximum(true, floor))
    strata = {'Fminus_true_zero': baseline_true == 0,
              'Fminus_true_positive_below_floor': (baseline_true > 0) & (baseline_true < floor),
              'Fminus_true_at_or_above_floor': baseline_true >= floor}
    rows = []
    for name, used in strata.items():
        count = int(used.sum())
        infinite = bool((true[used] == 0).any())
        rows.append({'stratum': name, 'count': count, 'total': true.size,
                     'people_with_observations': int(used.any(1).sum()),
                     'log_loss': float(loss[used].mean()) if count else None,
                     'weighted_log_loss_contribution': float(loss[used].sum() / true.size),
                     'raw_log_loss': float(-np.log(true[used]).mean()) if count and not infinite else None,
                     'raw_infinite': infinite})
    return rows


def paired_summary(candidate: dict, comparator: dict, replicates: int, seed: int) -> dict:
    delta = np.asarray(candidate['log_loss_per_person']) - np.asarray(comparator['log_loss_per_person'])
    require(delta.ndim == 1 and delta.size > 0 and np.isfinite(delta).all(), 'invalid paired people')
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(delta.size, size=(replicates, delta.size))].mean(1)
    return {'delta_log_loss': float(delta.mean()), 'delta_per_person': delta.tolist(),
            'conditional_person_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'people_improved': int((delta < 0).sum()), 'people': delta.size,
            'bootstrap_draws': replicates, 'bootstrap_seed': seed,
            'scope': 'conditional_fixed_historical_donor_library_not_NAM_population_interval'}


def choose(records: list[dict], objective: str, tolerance: float) -> dict:
    minimum = min(r[objective] for r in records)
    near = [r for r in records if r[objective] <= minimum + tolerance]
    return min(near, key=lambda r: (DIMENSIONS[r['family']], FAMILIES.index(r['family']),
                                  abs(math.log(r['temperature'])) + r['alpha'],
                                  r['temperature'], r['alpha']))


def neighbors(grid: list[float], best: float, geometric: bool, endpoints: list[float] | None = None) -> list[float]:
    index = grid.index(best)
    values = [best]
    for adjacent in (index - 1, index + 1):
        if 0 <= adjacent < len(grid):
            values.append(math.sqrt(best * grid[adjacent]) if geometric else (best + grid[adjacent]) / 2)
    if endpoints is not None:
        if index == 0:
            values.append(endpoints[0])
        if index == len(grid) - 1:
            values.append(endpoints[1])
    return sorted(set(values))


def fit_families(probabilities: np.ndarray, labels: np.ndarray, plan: dict) -> tuple[np.ndarray, list[dict], list[dict]]:
    """TRAIN-only search; this function cannot accept SELECT or SCORE tensors."""
    prior = train_prior(labels, plan['prior_pseudocount'])
    candidates, finalists = [], []
    for family in FAMILIES:
        ts = plan['temperature_grid'] if family in ('temperature', 'temperature_prior_mix') else [1.]
        alphas = plan['alpha_grid'] if family in ('uniform_mix', 'prior_mix', 'temperature_prior_mix') else [0.]
        family_records, seen = [], set()

        def search(temperatures, mixtures, stage):
            for temperature, alpha in itertools.product(temperatures, mixtures):
                if (temperature, alpha) in seen:
                    continue
                seen.add((temperature, alpha))
                report = evaluate(transform(probabilities, family, temperature, alpha, prior), labels, plan['floor'])
                record = {'family': family, 'temperature': temperature, 'alpha': alpha, 'stage': stage,
                          'train_log_loss': report['log_loss'], 'train_raw_log_loss': report['raw_log_loss'],
                          'train_raw_infinite': report['raw_infinite'],
                          'train_true_zero_count': report['true_zero_count'],
                          'train_true_below_floor_count': report['true_below_floor_count']}
                family_records.append(record)
        search(ts, alphas, 'initial')
        best_initial = choose(family_records, 'train_log_loss', plan['tie_tolerance'])
        refined_ts = neighbors(ts, best_initial['temperature'], True, plan['temperature_endpoints']) if len(ts) > 1 else ts
        refined_as = neighbors(alphas, best_initial['alpha'], False) if len(alphas) > 1 else alphas
        search(refined_ts, refined_as, 'train_neighbor_refinement')
        best = choose(family_records, 'train_log_loss', plan['tie_tolerance']).copy()
        for record in family_records:
            record['train_finalist'] = (record['temperature'], record['alpha']) == (best['temperature'], best['alpha'])
        finalists.append(best)
        candidates.extend(family_records)
    return prior, candidates, finalists


def fit_select(development_path: Path, plan_path: Path, outdir: Path) -> dict:
    plan_path, outdir = Path(plan_path), Path(outdir)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    data = read_npz(authenticated(development_path, plan['development_sha256']),
                    PAYLOAD | {'train_indices', 'select_indices'})
    train, select = validate_data(data, True)
    prior, candidates, finalists = fit_families(data['baseline'][train], data['truth_state'][train], plan)
    for finalist in finalists:
        selected = evaluate(transform(data['baseline'][select], finalist['family'], finalist['temperature'],
                                      finalist['alpha'], prior), data['truth_state'][select], plan['floor'])
        finalist['select_log_loss'] = selected['log_loss']
        finalist['select_metrics'] = selected
    chosen = choose(finalists, 'select_log_loss', plan['tie_tolerance'])
    model = {'schema_version': 'm39-calibration-model-v1', 'scope': SCOPE,
             'family': chosen['family'], 'temperature': chosen['temperature'], 'alpha': chosen['alpha'],
             'prior': prior.tolist(), 'state_names': list(STATE_NAMES),
             'prior_fitted_on': 'TRAIN', 'parameters_fitted_on': 'TRAIN', 'family_selected_on': 'SELECT',
             'refit_train_select': False}
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir / 'model.json', model)
    write_csv(outdir / 'candidates.csv', candidates)
    lock = {'schema_version': 'm39-calibration-lock-v1', 'scope': SCOPE, 'score_read': False,
            'plan': plan, 'plan_sha256': sha256(plan_path), 'model': model, 'model_sha256': json_hash(model),
            'model_file_sha256': sha256(outdir / 'model.json'), 'finalists': finalists,
            'candidate_csv_sha256': sha256(outdir / 'candidates.csv'),
            'source_sha256': sha256(Path(__file__)), 'numpy_version': np.__version__,
            'axis_sha256': {key: array_digest(data[key]) for key in AXES},
            'development_sample_keys': [k.decode() for k in data['sample_key_sha256']],
            'development_source_indices': data['source_indices'].tolist(),
            'role_counts': {'TRAIN': len(train), 'SELECT': len(select)}, 'anchors': data['baseline'].shape[1]}
    lock['payload_sha256'] = json_hash(lock)
    write_json(outdir / 'lock.json', lock)
    receipt = {'schema_version': 'm39-calibration-fit-v1', 'scope': SCOPE, 'score_read': False,
               'lock_sha256': sha256(outdir / 'lock.json'), 'model': model,
               'candidates': len(candidates), 'finalists': finalists, 'role_counts': lock['role_counts'],
               'input_sha256': {'development': plan['development_sha256'], 'plan': lock['plan_sha256']},
               'primary': 'clipped_log_loss_floor_1e-12_not_strictly_proper',
               'score_hash_received_opaque': True, 'genetic_features_read': False}
    write_json(outdir / 'fit.receipt.json', receipt)
    return receipt


def load_comparators(manifest_path: Path, expected_hash: str, score_hash: str, data: dict) -> dict:
    path = authenticated(manifest_path, expected_hash)
    manifest = json.loads(path.read_text())
    require(set(manifest) == {'schema_version', 'selection', 'score_receipt', 'comparators'} and
            manifest['schema_version'] == 'm39-calibration-comparators-v1', 'comparator manifest differs')

    def resolve(descriptor):
        filename = descriptor['path']
        require(isinstance(filename, str) and filename not in ('', '.', '..') and
                Path(filename).name == filename and '/' not in filename and '\\' not in filename,
                'comparator path must be a relative basename')
        resolved = (path.parent / filename).resolve()
        require(resolved.parent == path.parent.resolve() and resolved.is_file(),
                'comparator path escapes its frozen directory')
        return authenticated(resolved, descriptor['sha256'])

    selection = json.loads(resolve(manifest['selection']).read_text())
    receipt = json.loads(resolve(manifest['score_receipt']).read_text())
    require(selection['schema_version'] == 'm39-anchor-selection-lock-v1' and selection['score_read'] is False,
            'historical selection is not a pre-SCORE lock')
    require(receipt['schema_version'] == 'm39-anchor-score-v1' and receipt['score_sha256'] == score_hash
            and selection['expected_score_sha256'] == score_hash
            and receipt['selection_lock_sha256'] == manifest['selection']['sha256']
            and receipt['selected'] == selection['best_by_arm'], 'historical selection/score provenance differs')
    descriptors = manifest['comparators']
    require(len(descriptors) == 3 and {d['arm'] for d in descriptors} == {'common', 'pooled', 'carrier'},
            'exactly the three historical selected comparator arms required')
    result = {}
    for descriptor in descriptors:
        require(set(descriptor) == {'arm', 'path', 'sha256', 'case_id'}, 'comparator descriptor differs')
        arm, case = descriptor['arm'], descriptor['case_id']
        require(case == selection['best_by_arm'][arm], 'comparator was not historically selected')
        records = [r for r in receipt['records'] if r['case'] == case and r['arm'] == arm]
        require(len(records) == 1 and records[0]['selected_by_arm'] is True and
                records[0]['predictions_sha256'] == descriptor['sha256'], 'comparator record/hash differs')
        prediction = read_npz(resolve(descriptor), {'probabilities', 'truth_state', 'sample_key_sha256', 'coords'})
        for key in ('truth_state', 'sample_key_sha256', 'coords'):
            require(np.array_equal(prediction[key], data[key]), f'comparator {key} alignment differs')
        require(validate_probabilities(prediction['probabilities']).shape == data['baseline'].shape,
                'comparator probability shape differs')
        validate_labels(prediction['truth_state'], data['truth_state'].shape)
        result[arm.upper()] = prediction['probabilities']
    return result


def score_locked(lock_path: Path, score_path: Path, comparators_manifest_path: Path, outdir: Path) -> dict:
    lock_path, outdir = Path(lock_path), Path(outdir)
    lock = json.loads(lock_path.read_text())
    payload_hash = lock.pop('payload_sha256')
    require(valid_hash(payload_hash) and json_hash(lock) == payload_hash, 'lock payload changed')
    require(lock['schema_version'] == 'm39-calibration-lock-v1' and lock['scope'] == SCOPE and
            lock['score_read'] is False, 'pre-SCORE calibration lock required')
    plan, model = lock['plan'], lock['model']
    validate_plan(plan)
    require(sha256(Path(__file__)) == lock['source_sha256'], 'calibration code changed after fit')
    require(lock['numpy_version'] == np.__version__, 'NumPy version changed after fit')
    require(json_hash(model) == lock['model_sha256'] and model['scope'] == SCOPE and
            model['state_names'] == list(STATE_NAMES) and model['refit_train_select'] is False,
            'frozen model differs')
    data = read_npz(authenticated(score_path, plan['score_sha256']), PAYLOAD | {'score_indices'})
    validate_data(data, False)
    require(all(array_digest(data[k]) == lock['axis_sha256'][k] for k in AXES), 'SCORE anchor/state axis differs')
    require(set(k.decode() for k in data['sample_key_sha256']).isdisjoint(lock['development_sample_keys']) and
            set(data['source_indices'].tolist()).isdisjoint(lock['development_source_indices']),
            'SCORE people overlap development')
    probabilities = {'Fminus': data['baseline'], 'Ffull': data['full_baseline']}
    probabilities.update(load_comparators(Path(comparators_manifest_path), plan['comparators_manifest_sha256'],
                                         plan['score_sha256'], data))
    probabilities['CALIBRATED'] = transform(data['baseline'], model['family'], model['temperature'],
                                             model['alpha'], np.asarray(model['prior']))
    results, rows, reliability_rows, strata = {}, [], [], {}
    for name, p in probabilities.items():
        report = evaluate(p, data['truth_state'], plan['floor'])
        original = data['baseline']
        report['support_relative_to_original_Fminus'] = {
            'original_positive_to_zero': int(((original > 0) & (p == 0)).sum()),
            'original_zero_to_positive': int(((original == 0) & (p > 0)).sum()),
            'smallest_float64_positive_count': int((p == np.nextafter(0., 1.)).sum()),
            'probability_entries': int(p.size)}
        report['ece_descriptive'] = {}
        for bins in plan['reliability_bins']:
            entries, ece = reliability(p, data['truth_state'], bins)
            report['ece_descriptive'][str(bins)] = ece
            reliability_rows.extend({'model': name, **r} for r in entries)
        results[name] = report
        strata[name] = strata_metrics(p, data['baseline'], data['truth_state'], plan['floor'])
        rows.append({'model': name, 'log_loss': report['log_loss'], 'raw_log_loss': report['raw_log_loss'],
                     'raw_infinite': report['raw_infinite'], 'brier': report['brier'], 'accuracy': report['accuracy'],
                     **{f'dosage_mae_{ancestry}': value for ancestry, value in zip(('AFR', 'EUR', 'NAM'), report['dosage_mae'])},
                     'ece_10_descriptive': report['ece_descriptive']['10'], 'people': report['people'],
                     'anchors': report['anchors'], 'true_zero_count': report['true_zero_count'],
                     'true_below_floor_count': report['true_below_floor_count']})
    contrasts = {f'CALIBRATED_minus_{name}': paired_summary(results['CALIBRATED'], report,
                 plan['bootstrap_replicates'], plan['bootstrap_seed']) for name, report in results.items()
                 if name != 'CALIBRATED'}
    outdir.mkdir(parents=True, exist_ok=False)
    with (outdir / 'prediction.npz').open('xb') as stream:
        np.savez_compressed(stream, probabilities=probabilities['CALIBRATED'], truth_state=data['truth_state'],
                            sample_key_sha256=data['sample_key_sha256'], **{k: data[k] for k in AXES})
    write_csv(outdir / 'metrics.csv', rows)
    write_csv(outdir / 'reliability.csv', reliability_rows)
    receipt = {'schema_version': 'm39-calibration-score-v1', 'scope': SCOPE,
               'status': 'EXPLORATORY_CALIBRATION_DIAGNOSTIC_NOT_VALIDATION', 'model': model,
               'input_sha256': {'lock': sha256(lock_path), 'score': plan['score_sha256'],
                                'comparators_manifest': plan['comparators_manifest_sha256']},
               'outputs_sha256': {name: sha256(outdir / name) for name in
                                  ('prediction.npz', 'metrics.csv', 'reliability.csv')},
               'results': results, 'contrasts': contrasts, 'strata_by_original_Fminus': strata,
               'primary': 'clipped_log_loss_floor_1e-12_not_strictly_proper',
               'score_previously_observed': True, 'refit_after_score': False, 'genetic_features_read': False,
               'temperature_numerics': 'log_domain; positive underflow saturated to smallest float64 subnormal; not evaluation floor',
               'causal_fraction_explained_estimated': False, 'global_boundary_F1_evaluated': False,
               'source_test_or_valid_opened': False, 'independent_NAM_donor_units': 2,
               'does_not_close_deep_learning_or_rare_information': True}
    write_json(outdir / 'score.receipt.json', receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit-select')
    fit.add_argument('--development', type=Path, required=True)
    fit.add_argument('--plan', type=Path, required=True)
    score = commands.add_parser('score')
    score.add_argument('--lock', type=Path, required=True)
    score.add_argument('--score', type=Path, required=True)
    score.add_argument('--comparators-manifest', type=Path, required=True)
    for command in (fit, score):
        command.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit-select':
        result = fit_select(args.development, args.plan, args.outdir)
    else:
        result = score_locked(args.lock, args.score, args.comparators_manifest, args.outdir)
    print(json.dumps({'schema_version': result['schema_version'], 'scope': result['scope'],
                      'family': result['model']['family']}))


if __name__ == '__main__':
    main()
