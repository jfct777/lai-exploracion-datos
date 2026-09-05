#!/usr/bin/env python3
"""Exercise carrier-context interaction learning without opening genomic inputs.

The balanced construction deliberately removes ancestry information from the
common-only and pooled arms. This is a capacity diagnostic, not a LAI benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

from m39_carrier_models import CarrierContextModel, VARIANTS


def interaction_fixture(blocks: int, seed: int, baseline_mode: str = 'balanced') -> tuple[dict, torch.Tensor]:
    """Balanced XOR with identical common inputs under both possible labels.

    Common similarity identifies a context, while rare genotype identifies
    which ancestry's reference carries an allele in that context. Each ancestry
    has the same pooled frequencies in every example. Noise is shared between
    both labels within each common input, so controls cannot memorize it.
    """
    generator = np.random.default_rng(seed)
    size = blocks * 4
    context = np.zeros((size, 1, 2, 3, 2, 4), dtype=np.float32)
    dosage = np.zeros(context.shape[:-1], dtype=np.int64)
    labels = np.empty((size, 1), dtype=np.int64)
    for block in range(blocks):
        noise = generator.uniform(0, .1, (2, 3, 2))
        for x in range(2):
            for y in range(2):
                i = block * 4 + x * 2 + y
                context[i, ..., 1] = 1.0
                context[i, ..., 2] = np.log1p(32)
                context[i, ..., 3] = noise
                context[i, 0, 0, :, :, 0] = (np.arange(2) != x)
                context[i, 0, 1, :, :, 0] = .5
                # One diploid heterozygote per ancestry, not a phased allele.
                dosage[i, 0, :, 0, y] = 1
                dosage[i, 0, :, 2, 1-y] = 1
                labels[i, 0] = 1 if x == y else 4  # AE or EN
    mask = np.ones(dosage.shape, dtype=bool)
    pooled = np.zeros((size, 1, 3, 4), dtype=np.float32)
    pooled[:, :, (0, 2), 0:2] = .5
    pooled[:, :, 1, 0] = 1.0
    pooled[..., 3] = 1.0
    baseline = np.full((size, 1, 6), 1e-6, dtype=np.float32)
    baseline[..., (1, 4)] = (1-4e-6)/2
    if baseline_mode in ('zero_wrong', 'floor_wrong'):
        baseline.fill(0 if baseline_mode == 'zero_wrong' else 1e-12)
        baseline[..., 0] = 1 if baseline_mode == 'zero_wrong' else 1-5e-12
    elif baseline_mode != 'balanced':
        raise ValueError('Unknown synthetic baseline mode')
    batch = {
        'common_context': torch.from_numpy(context),
        'candidate_mask': torch.from_numpy(mask),
        'ref_dosage': torch.from_numpy(dosage),
        'rare_ref_observed': torch.from_numpy(mask.copy()),
        'query_dosage': torch.ones((size, 1), dtype=torch.long),
        'query_observed': torch.ones((size, 1), dtype=torch.bool),
        'pooled_summary': torch.from_numpy(pooled),
        'baseline': torch.from_numpy(baseline),
        'coords': torch.zeros(1),
    }
    return batch, torch.from_numpy(labels)


def select_people(batch: dict, indices: torch.Tensor) -> dict:
    return {k: v if k == 'coords' else v[indices] for k, v in batch.items()}


def score(probabilities: torch.Tensor, labels: torch.Tensor) -> dict:
    ll = -probabilities.gather(-1, labels[..., None]).clamp_min(1e-12).log().mean()
    # Each complete input is duplicated with opposite labels in this null.
    null_ll = -.5 * (probabilities[..., 1].clamp_min(1e-12).log()
                     + probabilities[..., 4].clamp_min(1e-12).log()).mean()
    if null_ll.item() < math.log(2) - 1e-6:
        raise ValueError('Identical-input null violates its analytic lower bound')
    return {'log_loss': ll.item(),
            'log_loss_probability_floor': 1e-12,
            'zero_true_state_probability_fraction': (
                probabilities.gather(-1, labels[..., None]) == 0).float().mean().item(),
            'accuracy': (probabilities.argmax(-1) == labels).float().mean().item(),
            'identical_input_null_log_loss': null_ll.item()}


def run_case(variant: str, width: int, learning_rate: float, steps: int,
             seed: int, outdir: Path, *, correction_head: str = 'multiplicative',
             baseline_mode: str = 'balanced', mixture_init: float = .01) -> dict:
    outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train, y_train = interaction_fixture(16, seed + 1, baseline_mode)
    evaluation, y_evaluation = interaction_fixture(32, seed + 2, baseline_mode)
    start = time.monotonic()
    results = []
    for arm in ('common', 'pooled', 'carrier'):
        torch.manual_seed(seed)
        model = CarrierContextModel(4, variant, width=width, correction_head=correction_head,
                                    mixture_init=mixture_init,
                                    mixture_prior=[1/6]*6 if correction_head == 'probability_mixture' else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed + 3)
        curve = []
        arm_start = time.monotonic()
        for step in range(steps):
            indices = torch.randint(len(y_train), (32,), generator=generator)
            batch = select_people(train, indices)
            probabilities = model(batch, arm=arm)
            loss = -probabilities.gather(-1, y_train[indices, ..., None]).clamp_min(1e-12).log().mean()
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite capacity loss for {variant}/{arm}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 50 == 0 or step == steps - 1:
                curve.append({'step': step+1, 'training_batch_loss': loss.item()})
        model.eval()
        with torch.no_grad():
            probabilities = model(evaluation, arm=arm)
            metrics = score(probabilities, y_evaluation)
        checkpoint = outdir / f'{arm}.pt'
        torch.save(model.state_dict(), checkpoint)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        np.savez_compressed(outdir / f'{arm}.predictions.npz',
                            probabilities=probabilities.numpy(), labels=y_evaluation.numpy())
        results.append({'arm': arm, **metrics, 'training_curve': curve,
                        'trainable_parameters': sum(p.numel() for p in model.parameters()),
                        'parameters_with_allocated_gradient_at_last_step': sum(
                            p.numel() for p in model.parameters() if p.grad is not None),
                        'nonzero_gradient_elements_at_last_step': sum(
                            int(torch.count_nonzero(p.grad)) for p in model.parameters() if p.grad is not None),
                        'wall_seconds': time.monotonic()-arm_start,
                        'checkpoint_sha256': checkpoint_hash})
    real = next(r for r in results if r['arm'] == 'carrier')
    controls = [r for r in results if r['arm'] != 'carrier']
    capacity_pass = real['accuracy'] >= .95 and real['log_loss'] <= .15
    control_pass = all(r['log_loss'] >= math.log(2)-1e-5 and r['accuracy'] <= .50001
                       for r in controls)
    report = {
        'schema_version': 'm39-capacity-v2',
        'scope': 'synthetic_interaction_capacity_not_biological_result',
        'variant': variant, 'width': width, 'learning_rate': learning_rate,
        'correction_head': correction_head, 'baseline_mode': baseline_mode,
        'mixture_init': mixture_init,
        'mixture_prior': [1/6]*6 if correction_head == 'probability_mixture' else None,
        'baseline_metrics': score(evaluation['baseline'], y_evaluation),
        'steps': steps, 'seed': seed, 'train_examples': len(y_train),
        'evaluation_examples': len(y_evaluation),
        'capacity_pass': capacity_pass, 'control_pass': control_pass,
        'results': results, 'wall_seconds': time.monotonic()-start,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_source_sha256': hashlib.sha256(Path(__file__).with_name('m39_carrier_models.py').read_bytes()).hexdigest(),
    }
    with (outdir/'capacity.receipt.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=39052026)
    parser.add_argument('--correction-head', choices=('multiplicative', 'probability_mixture'),
                        default='multiplicative')
    parser.add_argument('--baseline-mode', choices=('balanced', 'zero_wrong', 'floor_wrong'),
                        default='balanced')
    parser.add_argument('--mixture-init', type=float, default=.01)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000 or not 0 < args.learning_rate < 1:
        parser.error('steps must be 1..2000 and learning rate between zero and one')
    result = run_case(args.variant, args.width, args.learning_rate, args.steps, args.seed, args.outdir,
                      correction_head=args.correction_head, baseline_mode=args.baseline_mode,
                      mixture_init=args.mixture_init)
    print(json.dumps({key: result[key] for key in
                      ('variant', 'width', 'capacity_pass', 'control_pass', 'wall_seconds')}))


if __name__ == '__main__':
    main()
