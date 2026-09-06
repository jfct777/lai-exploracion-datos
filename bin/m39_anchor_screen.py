#!/usr/bin/env python3
"""Train and select carrier-context models without reading the scoring partition.

The unit-weighted endpoint concerns the selected rare loci, not chromosome-wide
boundary accuracy. This exploratory screen reuses a historical donor library.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

from m39_carrier_models import CarrierContextModel, STATE_NAMES, VARIANTS

# CARRIER-POOLED is the mechanism contrast. COMMON lacks an active rare branch;
# it is a useful baseline, not a claim of equal effective model capacity.
ARMS = ('common', 'pooled', 'carrier')
MASKS = ('candidate_mask', 'rare_ref_observed', 'query_observed')
FEATURES = ('common_context', 'candidate_mask', 'ref_dosage', 'rare_ref_observed',
            'query_dosage', 'query_observed', 'pooled_summary')
FLOOR = 1e-12
STATE_DOSAGES = np.asarray(((2, 0, 0), (1, 1, 0), (1, 0, 1),
                            (0, 2, 0), (0, 1, 1), (0, 0, 2)))


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def check_memory() -> None:
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 2.4 * 1024**3:
        raise MemoryError('M39 training crossed the 80% of 3GiB memory stop')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def match_samples(source: np.ndarray, requested: np.ndarray) -> np.ndarray:
    require(len(np.unique(source)) == len(source), 'duplicate feature sample keys')
    require(len(np.unique(requested)) == len(requested), 'duplicate requested samples')
    lookup = {bytes(key): i for i, key in enumerate(source)}
    require(all(bytes(key) in lookup for key in requested), 'unknown bound sample')
    return np.asarray([lookup[bytes(key)] for key in requested], dtype=np.int64)


def normalizer(context: np.ndarray, mask: np.ndarray, train: np.ndarray) -> tuple:
    values = context[train][mask[train].astype(bool)].astype(np.float64)
    require(len(values) > 0 and np.isfinite(values).all(), 'invalid TRAIN common features')
    return values.mean(0), np.maximum(values.std(0), 1e-6)


def ancestry_prior(labels: np.ndarray, train: np.ndarray, alpha: float = 1.) -> np.ndarray:
    require(alpha > 0 and np.isfinite(alpha), 'positive prior pseudocount required')
    counts = np.bincount(labels[train].reshape(-1), minlength=6).astype(np.float64) + alpha
    return (counts / counts.sum()).astype(np.float32)


def validate_partition(data: dict) -> tuple[np.ndarray, np.ndarray]:
    require('score_indices' not in data, 'development payload must not contain SCORE roles')
    train, select = data['train_indices'], data['select_indices']
    n, j = data['truth_state'].shape
    require(data['truth_state'].dtype.kind in 'iu', 'truth states must be integer codes')
    require(train.ndim == select.ndim == 1 and train.size > 0 and select.size > 0,
            'nonempty vector TRAIN/SELECT indices required')
    require(train.dtype.kind in 'iu' and select.dtype.kind in 'iu', 'integer role indices required')
    require(np.array_equal(np.sort(np.concatenate((train, select))), np.arange(n)),
            'TRAIN/SELECT must disjointly exhaust development people')
    require(data['baseline'].shape == (n, j, 6), 'baseline dimensions differ')
    require(np.all((data['truth_state'] >= 0) & (data['truth_state'] < 6)), 'invalid truth state')
    return train.astype(np.int64), select.astype(np.int64)


def load_inputs(features: Path, data_path: Path, *, fit: bool = True,
                mean: np.ndarray | None = None, scale: np.ndarray | None = None) -> tuple:
    data, arrays = read_npz(data_path), read_npz(features)
    require(data['truth_state'].dtype.kind in 'iu' and
            np.all((data['truth_state'] >= 0) & (data['truth_state'] < 6)),
            'truth states must be integer codes zero through five')
    indices = match_samples(arrays['sample_key_sha256'], data['sample_key_sha256'])
    require(np.array_equal(arrays['pos'], data['pos']), 'anchor physical positions differ')
    require(np.array_equal(arrays['anchor_indices'], data['anchor_indices']), 'anchor selection differs')
    if 'locus_id' in arrays:
        require(np.array_equal(arrays['locus_id'], data['locus_id']), 'anchor allele identities differ')
    selected = {}
    n, j = data['truth_state'].shape
    for key in FEATURES:
        value = arrays[key]
        if key == 'pooled_summary' and value.shape == (j, 3, 4):
            value = np.broadcast_to(value, (n, j, 3, 4)).copy()
        else:
            value = value[indices]
        selected[key] = value.astype(bool if key in MASKS else np.float32)
    del arrays
    if fit:
        train, select = validate_partition(data)
        mean, scale = normalizer(selected['common_context'], selected['candidate_mask'], train)
    else:
        require(mean is not None and scale is not None, 'scoring requires frozen TRAIN normalization')
        train = select = np.empty(0, dtype=np.int64)
    selected['common_context'] = np.where(selected['candidate_mask'][..., None],
        (selected['common_context'] - mean) / scale, 0).astype(np.float32)
    selected['baseline'] = data['baseline'].astype(np.float32)
    selected['coords'] = data['coords'].astype(np.float64)
    batch = {key: torch.from_numpy(value.copy()) for key, value in selected.items()}
    check_memory()
    labels = torch.from_numpy(data['truth_state'].astype(np.int64))
    return batch, labels, data, train, select, mean, scale


def subset(batch: dict, people: np.ndarray, loci: np.ndarray) -> dict:
    rows, columns = torch.as_tensor(people), torch.as_tensor(loci)
    return {key: value[columns] if key == 'coords' else value[rows[:, None], columns[None, :]]
            for key, value in batch.items()}


def arm_input(batch: dict, arm: str) -> tuple[dict, str]:
    require(arm in ARMS, 'unsupported screening arm')
    return batch, arm


def batches(people: np.ndarray, loci: int, person_batch: int, locus_batch: int,
            rng: np.random.Generator | None = None):
    persons = people.copy()
    columns = np.arange(loci)
    if rng is not None:
        rng.shuffle(persons)
        rng.shuffle(columns)
    for p in range(0, len(persons), person_batch):
        for j in range(0, loci, locus_batch):
            # Coordinates remain ordered within each forward call.
            yield persons[p:p+person_batch], np.sort(columns[j:j+locus_batch])


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    p = probabilities.astype(np.float64)
    require(p.shape == (*labels.shape, 6) and np.isfinite(p).all() and (p >= 0).all(),
            'nonfinite or incompatible probabilities')
    require(np.allclose(p.sum(-1), 1, atol=5e-5, rtol=0), 'prediction simplex differs')
    p /= p.sum(-1, keepdims=True)
    true = np.take_along_axis(p, labels[..., None], -1)[..., 0]
    ll = -np.log(np.maximum(true, FLOOR))
    brier = ((p - np.eye(6)[labels]) ** 2).sum(-1)
    dosage_error = np.abs(p @ STATE_DOSAGES - STATE_DOSAGES[labels])
    return {'log_loss': float(ll.mean()), 'brier': float(brier.mean()),
            'dosage_mae': dosage_error.mean((0, 1)).tolist(),
            'accuracy': float((p.argmax(-1) == labels).mean()),
            'log_loss_per_person': ll.mean(1).tolist(),
            'brier_per_person': brier.mean(1).tolist(),
            'fraction_true_probability_below_floor': float((true < FLOOR).mean()),
            'floor': FLOOR, 'weighting': 'equal_people_equal_selected_anchors'}


def predict(model, batch: dict, people: np.ndarray, arm: str,
            person_batch: int = 4, locus_batch: int = 64) -> np.ndarray:
    result = np.empty((len(people), batch['baseline'].shape[1], 6), dtype=np.float32)
    mapping = {int(p): i for i, p in enumerate(people)}
    model.eval()
    with torch.inference_mode():
        for rows, cols in batches(people, result.shape[1], person_batch, locus_batch):
            inputs, forward_arm = arm_input(subset(batch, rows, cols), arm)
            probabilities = model(inputs, arm=forward_arm).numpy()
            check_memory()
            result[np.asarray([mapping[int(p)] for p in rows])[:, None], cols[None, :]] = probabilities
    return result


def validate_config(config: dict) -> None:
    require(config['variant'] in VARIANTS and config['width'] in (16, 32, 64, 128), 'invalid model')
    require(config['radius_cm'] in (.05, .2, .5), 'unmaterialized radius')
    require(config['learning_rate'] in (.0003, .001, .003), 'learning rate outside screen')
    require(1 <= config['initial_epochs'] <= config['max_epochs'] <= 32, 'invalid epoch budget')
    require(1 <= config['person_batch'] <= 8 and 1 <= config['locus_batch'] <= 128,
            'minibatch outside resource budget')


def train_case(features: Path, development: Path, config: dict, outdir: Path) -> dict:
    validate_config(config)
    outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    batch, labels, data, train, select, mean, scale = load_inputs(features, development)
    require(float(read_npz(features)['radius_cm'][0]) == config['radius_cm'], 'radius/config mismatch')
    prior = ancestry_prior(labels.numpy(), train)
    model_kwargs = dict(common_features=4, variant=config['variant'], width=config['width'],
                        correction_head='probability_mixture', mixture_init=.01,
                        mixture_prior=prior.tolist())
    results = []
    for arm in ARMS:
        torch.manual_seed(config['seed'])
        model = CarrierContextModel(**model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        rng = np.random.default_rng(config['seed'] + 11)
        best_loss, best_epoch, best_state = math.inf, 0, None
        curve, exposure, step, wall = [], 0, 0, time.monotonic()
        limit = config['initial_epochs']
        for epoch in range(1, config['max_epochs'] + 1):
            model.train()
            total_loss, count = 0., 0
            for rows, cols in batches(train, labels.shape[1], config['person_batch'],
                                      config['locus_batch'], rng):
                inputs, forward_arm = arm_input(subset(batch, rows, cols), arm)
                auxiliary = model(inputs, arm=forward_arm, return_aux=True)
                target = labels[torch.as_tensor(rows)[:, None], torch.as_tensor(cols)[None, :]]
                loss = -auxiliary['log_probabilities'].gather(-1, target[..., None]).mean()
                require(bool(torch.isfinite(loss)), 'nonfinite exact training likelihood')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                check_memory()
                total_loss += loss.item() * target.numel()
                count += target.numel()
                step += 1
            exposure += count
            require(count == len(train) * labels.shape[1], 'incomplete epoch exposure')
            probabilities = predict(model, batch, select, arm,
                                    config['person_batch'], config['locus_batch'])
            evaluation = metrics(probabilities, labels.numpy()[select])
            if evaluation['log_loss'] < best_loss:
                best_loss, best_epoch = evaluation['log_loss'], epoch
                best_state = copy.deepcopy(model.state_dict())
            curve.append({'epoch': epoch, 'training_online_log_loss': total_loss / count,
                          'select_log_loss': evaluation['log_loss'], 'best_epoch': best_epoch,
                          'steps': step, 'exposures': exposure, 'seconds': time.monotonic()-wall})
            write_json(outdir / f'{arm}.epoch{epoch:02}.json', curve[-1])
            check_memory()
            if epoch == limit:
                if limit < config['max_epochs'] and best_epoch >= epoch - 2:
                    limit = config['max_epochs']
                else:
                    break
            if epoch >= config['initial_epochs'] and epoch - best_epoch >= 4:
                break
        require(best_state is not None, 'no finite SELECT checkpoint')
        model.load_state_dict(best_state)
        probabilities = predict(model, batch, select, arm, config['person_batch'], config['locus_batch'])
        train_pred = predict(model, batch, train, arm, config['person_batch'], config['locus_batch'])
        evaluation = metrics(probabilities, labels.numpy()[select])
        checkpoint = outdir / f'{arm}.pt'
        torch.save({'state_dict': best_state, 'model_kwargs': model_kwargs,
                    'normalization_mean': mean.tolist(), 'normalization_scale': scale.tolist(),
                    'arm': arm, 'config': config, 'best_epoch': best_epoch}, checkpoint)
        np.savez_compressed(outdir / f'{arm}.select.npz', probabilities=probabilities,
                            truth_state=data['truth_state'][select],
                            sample_key_sha256=data['sample_key_sha256'][select])
        results.append({'arm': arm, 'select': evaluation,
                        'train': metrics(train_pred, labels.numpy()[train]),
                        'best_epoch': best_epoch, 'epochs_run': epoch, 'curve': curve,
                        'steps': step, 'exposures': exposure,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'parameters_with_last_gradient': sum(p.numel() for p in model.parameters() if p.grad is not None),
                        'checkpoint_sha256': sha256(checkpoint),
                        'seconds': time.monotonic() - wall})
    receipt = {'schema_version': 'm39-anchor-training-v1', 'scope': 'exploratory_chr22_R0_development_only',
               'score_read': False, 'config': config, 'results': results,
               'train_people': len(train), 'select_people': len(select), 'anchors': labels.shape[1],
               'source_sha256': {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('m39_carrier_models.py'))},
               'input_sha256': {'features': sha256(features), 'development': sha256(development)},
               'baseline_select': metrics(data['baseline'][select], data['truth_state'][select]),
               'prior': prior.tolist(), 'prior_pseudocount_each_class': 1.,
               'normalization_mean': mean.tolist(), 'normalization_scale': scale.tolist(),
               'seconds': time.monotonic()-started,
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'torch_version': torch.__version__, 'numpy_version': np.__version__}
    write_json(outdir / 'training.receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=Path)
    group.add_argument('--config-json')
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text() if args.config else args.config_json)
    result = train_case(args.features, args.development, config, args.outdir)
    print(json.dumps({key: result[key] for key in ('schema_version', 'seconds', 'score_read')}))


if __name__ == '__main__':
    main()
