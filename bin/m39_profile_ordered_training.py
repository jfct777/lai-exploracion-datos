#!/usr/bin/env python3
"""Measure complete ordered-model optimization on authenticated TRAIN inputs.

Targets are artificial row/step ordinals, never ancestral truth. These weights
are discarded: this profiles implementation and resources, not LAI accuracy.
Each case runs in a fresh process to make peak RSS independently interpretable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import resource
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from m33_safe_bridge_core import require, write_exclusive_json
from m34_prepare_panel_factors import sha256_file
from m39_ordered_context import OrderedContextStore
from m39_ordered_batches import pack_batch, estimate_batch_bytes
from m39_ordered_models import OrderedLAIModel, OrderedModelConfig

SCHEMA = 'm39-ordered-training-profile-v1'
SOURCE_FILES = ('m39_profile_ordered_training.py', 'm39_ordered_models.py',
                'm39_ordered_batches.py', 'm39_ordered_context.py',
                'm33_safe_bridge_core.py', 'm34_prepare_panel_factors.py',
                'm34_generate_mosaics.py', 'm39_carrier_context.py')
PROFILE_FIELDS = {'schema_version', 'scope', 'parent_receipt_sha256', 'store_case',
                  'store_manifest_sha256', 'folds_sha256', 'fold', 'seed', 'steps',
                  'learning_rate', 'weight_decay', 'torch_threads', 'max_input_bytes',
                  'max_rss_kib', 'max_seconds', 'equivalence_atol', 'equivalence_rtol',
                  'recipes', 'cases'}


def load_profile(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    require(isinstance(cfg, dict) and set(cfg) == PROFILE_FIELDS, 'profile inventory differs')
    require(cfg['schema_version'] == SCHEMA and cfg['scope'] ==
            'technical_optimization_synthetic_labels_inner_TRAIN_only', 'scope differs')
    for name in ('parent_receipt_sha256', 'store_manifest_sha256', 'folds_sha256'):
        require(isinstance(cfg[name], str) and re.fullmatch('[a-f0-9]{64}', cfg[name]),
                f'{name}: invalid hash')
    require(cfg['store_case'] == 'people_48/radius_1cm' and type(cfg['fold']) is int
            and cfg['fold'] == 0, 'only frozen TRAIN48 radius1 fold0 resource profile')
    for name, lo, hi in (('steps', 2, 5), ('torch_threads', 1, 2),
                         ('max_input_bytes', 1, 64 * 1024**2),
                         ('max_rss_kib', 1, 6710886), ('max_seconds', 1, 900),
                         ('seed', 0, 2**32-1)):
        require(type(cfg[name]) is int and lo <= cfg[name] <= hi, f'{name}: outside envelope')
    require(cfg['learning_rate'] == 0.001 and cfg['weight_decay'] == 0.0,
            'technical optimizer recipe differs')
    require(cfg['equivalence_atol'] == 3e-5 and cfg['equivalence_rtol'] == 1e-4,
            'numerical tolerances differ')
    require(set(cfg['recipes']) == {'small', 'medium'}, 'recipe sizes differ')
    recipe_fields = {'width', 'depth', 'kernels', 'dilations', 'heads', 'attention_radius_tokens'}
    for size, recipe in cfg['recipes'].items():
        require(isinstance(recipe, dict) and set(recipe) == recipe_fields, 'recipe fields differ')
        require(all(type(recipe[k]) is int for k in
                    ('width', 'depth', 'heads', 'attention_radius_tokens')) and
                all(type(n) is int for k in ('kernels', 'dilations') for n in recipe[k]),
                'recipe counts must be integers')
        require(recipe['width'] == (32 if size == 'small' else 64) and
                recipe['depth'] == (2 if size == 'small' else 4), 'technical capacity differs')
        require(recipe['heads'] == 4 and recipe['attention_radius_tokens'] ==
                (4 if size == 'small' else 8), 'attention envelope differs')
        require(recipe['kernels'] == ([3, 7] if size == 'small' else [3, 7, 15]) and
                recipe['dilations'] == ([1, 2] if size == 'small' else [1, 2, 4, 8]),
                'CNN envelope differs')
    cases = cfg['cases']
    require(isinstance(cases, list) and len(cases) == 16, 'sixteen technical cases required')
    seen = set()
    for case in cases:
        require(isinstance(case, dict) and set(case) ==
                {'id', 'family', 'size', 'window', 'batch_size', 'arm', 'core_sites'}, 'case fields differ')
        require(case['family'] in ('cnn', 'attention') and case['size'] in cfg['recipes']
                and case['window'] in ('median', 'maximum') and case['arm'] in ('common', 'real'),
                'unsupported case')
        require(type(case['batch_size']) is int and case['batch_size'] ==
                (1 if case['window'] == 'median' else 2) and type(case['core_sites']) is int
                and case['core_sites'] == 256,
                'batch/core envelope differs')
        expected = '{family}-{size}-{window}-b{batch_size}-{arm}'.format(**case)
        require(case['id'] == expected and expected not in seen, 'case identity differs')
        seen.add(expected)
    return cfg


def authenticate(store_dir: Path, parent_path: Path, folds_path: Path, cfg: dict):
    require(sha256_file(parent_path) == cfg['parent_receipt_sha256'], 'parent receipt hash differs')
    parent = json.loads(parent_path.read_text())
    require(parent['decision'] == 'PASS_ORDERED_INPUT_TECHNICAL_ONLY' and
            not parent['boundaries']['truth_opened'], 'parent scope differs')
    parents = [p for p in parent['profiles'] if p['case'] == cfg['store_case']]
    require(len(parents) == 1 and parents[0]['store_manifest_sha256'] == cfg['store_manifest_sha256'],
            'store not bound to parent receipt')
    store = OrderedContextStore.open(store_dir, expected_manifest_sha256=cfg['store_manifest_sha256'])
    manifest = json.loads((store_dir / 'manifest.json').read_text())
    require(manifest['source_sha256'] == parent['input_hashes'], 'source hashes differ from parent')
    require(sha256_file(folds_path) == cfg['folds_sha256'] == parent['input_hashes']['folds'],
            'fold hash differs')
    with np.load(folds_path, allow_pickle=False) as z:
        require(set(z.files) == {'sample_key_sha256', 'roles', 'outer_fold',
                                 'inner_split_seed', 'outer_seed'}, 'fold fields differ')
        keys, roles = z['sample_key_sha256'], z['roles']
        require(keys.dtype == np.dtype('S64') and keys.shape == (96,) and roles.shape == (3,96)
                and len(set(keys.tolist())) == 96 and np.array_equal(z['outer_fold'], np.arange(3)),
                'fold axes differ')
        train_keys = keys[roles[cfg['fold']] == 'TRAIN']
        require(len(train_keys) == 48 and
                np.array_equal(np.sort(train_keys), store.arrays['sample_key_sha256']),
                'store is not exactly the frozen TRAIN set')
    require(store.shape == (48,660,8) and len(store.arrays['reference_ancestry']) == 753,
            'resource regime differs')
    return store, manifest


def choose_pairs(store, case: dict):
    lengths = np.array([store.window_bounds(j)[1] - store.window_bounds(j)[0]
                        for j in range(store.shape[1])])
    require(np.all(lengths > 0), 'profile requires nonempty real windows')
    if case['window'] == 'maximum':
        anchor = int(np.argmax(lengths))
    else:
        anchor = int(np.argmin(np.abs(lengths - np.median(lengths))))
    return [(q, anchor) for q in range(case['batch_size'])], lengths


def tensor_digest(batch: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def gradients(model):
    norms = {}
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                f'missing/nonfinite gradient: {name}')
        group = name.split('.')[0]
        norms[group] = norms.get(group, 0.0) + float(parameter.grad.double().square().sum())
    require(norms and all(value > 0 for value in norms.values()), 'inactive model component')
    return {k: float(v**0.5) for k,v in norms.items()}


def recheck_inputs(store_dir: Path, manifest: dict, paths_and_hashes: dict) -> None:
    for path, digest in paths_and_hashes.items():
        require(sha256_file(path) == digest, f'input/source changed: {path.name}')
    for spec in manifest['arrays'].values():
        require(sha256_file(store_dir / spec['file']) == spec['sha256'], 'store array changed')


def run_case(store_dir: Path, parent_path: Path, folds_path: Path, profile_path: Path,
             case_id: str, output_dir: Path, source_commit: str) -> dict:
    start = time.monotonic()
    cfg = load_profile(profile_path)
    require(re.fullmatch('[a-f0-9]{40}', source_commit), 'full source commit required')
    require(not output_dir.exists(), 'output already exists')
    matches = [c for c in cfg['cases'] if c['id'] == case_id]
    require(len(matches) == 1, 'unknown case')
    case = matches[0]
    torch.set_num_threads(cfg['torch_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    paths = {profile_path: sha256_file(profile_path), parent_path: cfg['parent_receipt_sha256'],
             folds_path: cfg['folds_sha256'], store_dir / 'manifest.json': cfg['store_manifest_sha256']}
    code_hashes = {}
    for name in SOURCE_FILES:
        path = Path(__file__).with_name(name)
        code_hashes[name] = sha256_file(path)
        paths[path] = code_hashes[name]
    store, manifest = authenticate(store_dir, parent_path, folds_path, cfg)
    pairs, lengths = choose_pairs(store, case)
    allocation_plan = estimate_batch_bytes(store, pairs)
    batch = pack_batch(store, pairs, max_input_bytes=cfg['max_input_bytes'], device='cpu')
    data_hash = tensor_digest(batch)
    recipe = dict(cfg['recipes'][case['size']])
    recipe['kernels'], recipe['dilations'] = tuple(recipe['kernels']), tuple(recipe['dilations'])
    model_config = OrderedModelConfig(family=case['family'], **recipe, dropout=0.0,
                                     core_sites=case['core_sites'], checkpoint_chunks=True)
    model = OrderedLAIModel(model_config)
    # Real full windows, not cropped cores. Backward equivalence is separately
    # tested on controlled fixtures; this checks the actual production geometry.
    parity_start = time.monotonic()
    model.eval()
    with torch.no_grad():
        full = model(batch, arm=case['arm'], chunked=False)
        blocked = model(batch, arm=case['arm'], chunked=True)
    torch.testing.assert_close(full, blocked, atol=cfg['equivalence_atol'],
                               rtol=cfg['equivalence_rtol'])
    max_logit_error = float((full - blocked).abs().max())
    parity_seconds = time.monotonic() - parity_start
    del full, blocked
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    prepared_seconds = time.monotonic() - start
    rows = []
    for step in range(cfg['steps']):
        optimizer.zero_grad(set_to_none=True)
        before = {name: value.detach().clone() for name,value in model.named_parameters()}
        # These labels carry no biological or input-dependent information.
        labels = (torch.arange(case['batch_size']) + step) % 6
        tick = time.monotonic()
        logits = model(batch, arm=case['arm'], chunked=True)
        loss = F.cross_entropy(logits, labels)
        require(torch.isfinite(loss).item(), 'nonfinite artificial loss')
        forward_seconds = time.monotonic() - tick
        tick = time.monotonic()
        loss.backward()
        backward_seconds = time.monotonic() - tick
        norms = gradients(model)
        tick = time.monotonic()
        optimizer.step()
        optimizer_seconds = time.monotonic() - tick
        updated = {}
        for name, parameter in model.named_parameters():
            require(torch.isfinite(parameter).all().item(), f'nonfinite parameter: {name}')
            group = name.split('.')[0]
            updated[group] = updated.get(group, 0) + int(not torch.equal(before[name], parameter))
        require(all(value > 0 for value in updated.values()), 'component weights unchanged')
        optimizer_bytes = sum(t.numel() * t.element_size() for state in optimizer.state.values()
                              for t in state.values() if isinstance(t, torch.Tensor))
        require(optimizer_bytes > 0, 'optimizer state not allocated')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        require(rss <= cfg['max_rss_kib'], 'RSS exceeds declared technical stop rule')
        rows.append({'step': step, 'warmup': step == 0, 'forward_seconds': forward_seconds,
                     'backward_seconds': backward_seconds, 'optimizer_seconds': optimizer_seconds,
                     'optimizer_state_bytes': optimizer_bytes, 'gradient_norm_by_component': norms,
                     'updated_tensors_by_component': updated, 'peak_rss_kib': rss})
        del loss, logits, before
    require(tensor_digest(batch) == data_hash, 'batch was modified')
    recheck_inputs(store_dir, manifest, paths)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {'schema_version': SCHEMA, 'decision': 'PASS_ORDERED_MODEL_RESOURCE_TECHNICAL_ONLY',
              'case': case, 'architecture': asdict(model_config),
              'receptive_radius_tokens': model_config.receptive_radius_tokens,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'input': {'store_case': cfg['store_case'], 'store_shape': list(store.shape),
                        'allocation_plan': allocation_plan,
                        'query_anchor_pairs': pairs, 'full_window_sites': [int(lengths[j]) for _,j in pairs],
                        'padded_sites': int(batch['channels'].shape[-2]),
                        'tensor_bytes': sum(t.numel()*t.element_size() for t in batch.values()),
                        'tensor_sha256': data_hash},
              'equivalence': {'real_full_vs_chunk_logits_max_abs_error': max_logit_error,
                              'atol': cfg['equivalence_atol'], 'rtol': cfg['equivalence_rtol'],
                              'seconds': parity_seconds,
                              'gradient_equivalence': 'separate_controlled_fixture_tests'},
              'steps': rows, 'prepared_seconds': prepared_seconds,
              'elapsed_seconds': time.monotonic() - start,
              'resources': {'peak_rss_kib_including_full_forward_check': usage.ru_maxrss,
                            'user_cpu_seconds': usage.ru_utime, 'system_cpu_seconds': usage.ru_stime},
              'runtime': {'torch': torch.__version__, 'numpy': np.__version__,
                          'device': 'cpu', 'threads': torch.get_num_threads(), 'dtype': 'float32'},
              'provenance': {'source_commit': source_commit, 'source_sha256': code_hashes,
                             'profile_sha256': paths[profile_path],
                             'parent_sha256': cfg['parent_receipt_sha256'],
                             'manifest_sha256': cfg['store_manifest_sha256'], 'folds_sha256': cfg['folds_sha256']},
              'scope': {'optimization_executed': True, 'biological_training': False,
                        'label_source': '(step + batch_row) mod 6; artificial',
                        'truth_opened': False, 'predictions_opened': False,
                        'feature_roles': ['TRAIN'], 'accuracy_evaluated': False,
                        'weights_consumable': False, 'new_cloud_instances': 0}}
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_exclusive_json(output_dir / 'profile.json', report)
    print(json.dumps({'case': case_id, 'decision': report['decision'],
                      'elapsed_seconds': report['elapsed_seconds'], 'rss_kib': usage.ru_maxrss}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('store-dir', 'parent-receipt', 'folds', 'profile-config', 'output-dir'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    cfg = load_profile(args.profile_config)
    def timeout(_signum, _frame):
        raise TimeoutError('technical case time limit reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cfg['max_seconds'])
    run_case(args.store_dir.resolve(), args.parent_receipt.resolve(), args.folds.resolve(),
             args.profile_config.resolve(), args.case_id, args.output_dir, args.source_commit)


if __name__ == '__main__':
    main()
