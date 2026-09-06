"""Exact log-mixture and backward checks on synthetic tensors, not real fits."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m39_carrier_models import CarrierContextModel, VARIANTS
from test_m39_carrier_models import interaction_fixture, clone_batch, nonzero_heads


def mixture_model(variant):
    return CarrierContextModel(2, variant, width=16, correction_head='probability_mixture',
                               mixture_prior=[1/6]*6)


class LogMixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(29)
        self.batch, self.labels = interaction_fixture(1)

    def assert_finite_gradients(self, model):
        present = [(name, parameter.grad) for name, parameter in model.named_parameters()
                   if parameter.grad is not None]
        self.assertTrue(present)
        for name, grad in present:
            self.assertTrue(bool(torch.isfinite(grad).all()), f'nonfinite gradient: {name}')

    def test_log_exp_matches_raw_probability_all_arms_and_variants(self):
        for variant in VARIANTS:
            model = mixture_model(variant)
            nonzero_heads(model)
            for arm in ('common', 'pooled', 'carrier'):
                with self.subTest(variant=variant, arm=arm):
                    auxiliary = model(self.batch, arm, return_aux=True)
                    torch.testing.assert_close(auxiliary['log_probabilities'].exp(),
                                               auxiliary['probabilities'], atol=1e-7, rtol=2e-6)
                    torch.testing.assert_close(torch.logsumexp(auxiliary['log_probabilities'], -1),
                                               torch.zeros_like(self.labels).float(), atol=2e-7, rtol=0)

    def test_zero_baseline_rescuable_true_state_has_finite_nonzero_gradient(self):
        for variant in VARIANTS:
            for arm in ('common', 'pooled', 'carrier'):
                with self.subTest(variant=variant, arm=arm):
                    model = mixture_model(variant)
                    auxiliary = model(self.batch, arm, return_aux=True)
                    self.assertTrue(bool((self.batch['baseline'][..., 0] == 0).all()))
                    loss = -auxiliary['log_probabilities'][..., 0].mean()
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    self.assert_finite_gradients(model)
                    self.assertGreater(float(model.common_head[-1].bias.grad.abs().sum()), 0)

    def test_no_common_or_rare_support_preserves_zeros_without_nan_backward(self):
        batch = clone_batch(self.batch)
        batch['candidate_mask'].zero_()
        batch['rare_ref_observed'].zero_()
        batch['baseline'].zero_()
        batch['baseline'][..., 0] = 1
        for variant in VARIANTS:
            for arm in ('common', 'pooled', 'carrier'):
                with self.subTest(variant=variant, arm=arm):
                    model = mixture_model(variant)
                    auxiliary = model(batch, arm, return_aux=True)
                    torch.testing.assert_close(auxiliary['probabilities'], batch['baseline'], rtol=0, atol=0)
                    torch.testing.assert_close(auxiliary['log_probabilities'].exp(), batch['baseline'], rtol=0, atol=0)
                    self.assertTrue(bool(torch.isneginf(auxiliary['log_probabilities'][..., 1:]).all()))
                    loss = -auxiliary['log_probabilities'][..., 0].mean()
                    self.assertEqual(float(loss.detach()), 0.)
                    loss.backward()
                    self.assert_finite_gradients(model)

    def test_missing_rare_query_is_exact_common_fallback_in_log_space(self):
        batch = clone_batch(self.batch)
        batch['query_observed'].zero_()
        batch['query_dosage'].fill_(float('nan'))
        for variant in VARIANTS:
            model = mixture_model(variant)
            nonzero_heads(model)
            expected = model(batch, 'common', return_aux=True)['log_probabilities'].detach()
            for arm in ('pooled', 'carrier'):
                with self.subTest(variant=variant, arm=arm):
                    model.zero_grad(set_to_none=True)
                    auxiliary = model(batch, arm, return_aux=True)
                    torch.testing.assert_close(auxiliary['log_probabilities'], expected, rtol=0, atol=0)
                    (-auxiliary['log_probabilities'][..., 0].mean()).backward()
                    self.assert_finite_gradients(model)

    def test_mixed_supported_and_unsupported_rows_have_finite_backward(self):
        batch = clone_batch(self.batch)
        batch['candidate_mask'][0].zero_()
        batch['rare_ref_observed'][0].zero_()
        batch['baseline'].zero_()
        batch['baseline'][..., 0] = 1
        for variant in VARIANTS:
            for arm in ('common', 'pooled', 'carrier'):
                with self.subTest(variant=variant, arm=arm):
                    model = mixture_model(variant)
                    auxiliary = model(batch, arm, return_aux=True)
                    self.assertEqual(float(auxiliary['log_probabilities'][0, 0, 0].detach()), 0.)
                    loss = -auxiliary['log_probabilities'][..., 0].mean()
                    loss.backward()
                    self.assert_finite_gradients(model)
                    self.assertGreater(float(model.common_head[-1].bias.grad.abs().sum()), 0)

    def test_log_likelihood_stays_finite_and_trainable_after_expert_softmax_underflow(self):
        batch = clone_batch(self.batch)
        batch['baseline'].zero_()
        batch['baseline'][..., 0] = 1
        for variant in VARIANTS:
            for arm in ('common', 'pooled', 'carrier'):
                with self.subTest(variant=variant, arm=arm):
                    model = mixture_model(variant)
                    with torch.no_grad():
                        model.common_head[-1].weight.zero_()
                        model.common_head[-1].bias.fill_(-1000)
                        model.common_head[-1].bias[0] = 1000
                        model.rare_head[-1].weight.zero_()
                        model.rare_head[-1].bias.zero_()
                        for gate in (model.common_gate, model.rare_gate):
                            gate.weight.zero_()
                            gate.bias.zero_()
                    auxiliary = model(batch, arm, return_aux=True)
                    self.assertTrue(bool((auxiliary['probabilities'][..., 1] == 0).all()))
                    loss = -auxiliary['log_probabilities'][..., 1].mean()
                    expected = 2000 - math.log(.5 if arm == 'common' else .75)
                    self.assertAlmostEqual(float(loss.detach()), expected, delta=3e-4)
                    loss.backward()
                    self.assert_finite_gradients(model)
                    self.assertGreater(float(model.common_head[-1].bias.grad.abs().sum()), 0)

    def test_positive_subnormal_baseline_not_replaced_by_normal_float_floor(self):
        batch = clone_batch(self.batch)
        batch['candidate_mask'].zero_()
        batch['rare_ref_observed'].zero_()
        batch['baseline'].zero_()
        batch['baseline'][..., 0] = 1
        batch['baseline'][..., 1] = 1e-40
        expected = math.log(float(batch['baseline'][0, 0, 1]))
        for arm in ('common', 'pooled', 'carrier'):
            with self.subTest(arm=arm):
                auxiliary = mixture_model(VARIANTS[0])(batch, arm, return_aux=True)
                self.assertAlmostEqual(float(auxiliary['log_probabilities'][0, 0, 1].detach()), expected, delta=1e-5)

    def test_saturated_gate_logits_do_not_break_finite_loss_or_backward(self):
        for raw_gate in (-1000., 1000.):
            for arm in ('common', 'pooled', 'carrier'):
                with self.subTest(raw_gate=raw_gate, arm=arm):
                    model = mixture_model(VARIANTS[0])
                    with torch.no_grad():
                        model.common_gate.bias.fill_(raw_gate)
                        model.rare_gate.bias.fill_(raw_gate)
                    auxiliary = model(self.batch, arm, return_aux=True)
                    loss = -auxiliary['log_probabilities'][..., 0].mean()
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    self.assert_finite_gradients(model)


if __name__ == '__main__':
    unittest.main()
