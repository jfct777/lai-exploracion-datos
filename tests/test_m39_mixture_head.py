from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))
from m39_carrier_models import CarrierContextModel, VARIANTS  # noqa: E402
from test_m39_carrier_models import (  # noqa: E402
    clone_batch, interaction_fixture, log_loss, nonzero_heads,
)


def mixture_model(variant: str) -> CarrierContextModel:
    return CarrierContextModel(2, variant, width=16,
                               correction_head="probability_mixture",
                               mixture_prior=torch.full((6,), 1 / 6))


def force_state(model: CarrierContextModel, state: int) -> None:
    """Set a deterministic head-capacity fixture without passing labels to forward."""
    with torch.no_grad():
        for head in (model.common_head, model.rare_head):
            head[-1].weight.zero_()
            head[-1].bias.fill_(-100.)
            head[-1].bias[state] = 100.
        for gate in (model.common_gate, model.rare_gate):
            gate.weight.zero_()
            gate.bias.fill_(100.)


class MixtureHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(29)
        self.batch, self.labels = interaction_fixture(repeats=1)

    def test_prior_and_initial_gate_are_explicit_validated_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit mixture_prior"):
            CarrierContextModel(2, correction_head="probability_mixture")
        for prior in (torch.zeros(6), torch.ones(6), torch.ones(5) / 5,
                      torch.tensor([.2, .2, .2, .2, .2, 0.]), torch.full((6,), float("nan"))):
            with self.assertRaisesRegex(ValueError, "six-state simplex"):
                CarrierContextModel(2, correction_head="probability_mixture", mixture_prior=prior)
        for initial in (0., 1., float("nan")):
            with self.assertRaisesRegex(ValueError, "mixture_init"):
                CarrierContextModel(2, correction_head="probability_mixture",
                                    mixture_prior=torch.full((6,), 1 / 6), mixture_init=initial)

    def test_default_multiplicative_parameters_and_outputs_are_unchanged(self) -> None:
        for variant in VARIANTS:
            torch.manual_seed(7)
            implicit = CarrierContextModel(2, variant, width=16)
            torch.manual_seed(7)
            explicit = CarrierContextModel(2, variant, width=16, correction_head="multiplicative")
            self.assertEqual(implicit.state_dict().keys(), explicit.state_dict().keys())
            self.assertNotIn("mixture_log_prior", implicit.state_dict())
            nonzero_heads(implicit)
            explicit.load_state_dict(implicit.state_dict(), strict=True)
            for arm in ("common", "pooled", "carrier", "perturbed"):
                torch.testing.assert_close(implicit(self.batch, arm), explicit(self.batch, arm), rtol=0, atol=0)

    def test_initialization_and_component_equations(self) -> None:
        model = mixture_model("gated_deepset")
        result = model(self.batch, return_aux=True)
        prior = torch.full_like(self.batch["baseline"], 1 / 6)
        expected_common = .99 * self.batch["baseline"] + .01 * prior
        expected = .99 * expected_common + .01 * prior
        torch.testing.assert_close(result["common_gate"], torch.full_like(result["common_gate"], .01))
        torch.testing.assert_close(result["common_expert_probabilities"], prior)
        torch.testing.assert_close(result["rare_expert_probabilities"], prior)
        torch.testing.assert_close(result["common_probabilities"], expected_common)
        torch.testing.assert_close(result["probabilities"], expected)
        self.assertFalse(torch.equal(result["probabilities"], self.batch["baseline"]))

    def test_old_zero_lock_and_floor_odds_bound(self) -> None:
        zero = clone_batch(self.batch)
        zero["baseline"].zero_()
        zero["baseline"][..., 0] = 1.
        floored = clone_batch(zero)
        epsilon = 1e-12
        floored["baseline"][..., 0] = 1 - 2 * epsilon
        floored["baseline"][..., 1] = floored["baseline"][..., 4] = epsilon
        bound = epsilon * math.exp(16) / (1 - epsilon + epsilon * math.exp(16))
        for variant in VARIANTS:
            old = CarrierContextModel(2, variant, width=16)
            force_state(old, 1)
            self.assertTrue(torch.equal(old(zero), zero["baseline"]))
            self.assertLess(float(old(floored)[..., 1].max().detach()), 1e-5)
            torch.testing.assert_close(old(floored)[..., 1], torch.full_like(self.labels.float(), bound),
                                       atol=1e-10, rtol=1e-5)
            new = mixture_model(variant)
            force_state(new, 1)
            self.assertGreater(float(new(zero)[..., 1].min().detach()), .99)
            self.assertGreater(float(new(floored)[..., 1].min().detach()), .99)

    def test_all_six_outputs_can_recover_an_exact_zero(self) -> None:
        for variant in VARIANTS:
            model = mixture_model(variant)
            for state in range(6):
                batch = clone_batch(self.batch)
                batch["baseline"].zero_()
                batch["baseline"][..., (state + 1) % 6] = 1.
                force_state(model, state)
                probability = model(batch)
                self.assertTrue((probability.argmax(-1) == state).all())
                self.assertGreater(float(probability[..., state].min().detach()), .99)

    def test_finite_initial_gradients_with_zero_baseline(self) -> None:
        batch = clone_batch(self.batch)
        batch["baseline"].zero_()
        batch["baseline"][..., 0] = 1.
        for variant in VARIANTS:
            model = mixture_model(variant)
            probability = model(batch)
            self.assertTrue((probability > 0).all())
            torch.testing.assert_close(probability.sum(-1), torch.ones_like(self.labels).float())
            loss = log_loss(probability, self.labels)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(float(model.common_gate.bias.grad.abs().sum()), 0.)
            self.assertGreater(float(model.rare_gate.bias.grad.abs().sum()), 0.)
            self.assertGreater(float(model.rare_head[-1].weight.grad.abs().sum()), 0.)

    def test_fallbacks_and_common_rare_isolation(self) -> None:
        missing = clone_batch(self.batch)
        missing["query_observed"].fill_(False)
        missing["query_dosage"].fill_(float("nan"))
        no_reference = clone_batch(self.batch)
        no_reference["rare_ref_observed"].fill_(False)
        no_reference["ref_dosage"].fill_(float("nan"))
        empty = clone_batch(self.batch)
        empty["candidate_mask"].fill_(False)
        empty["common_context"].fill_(float("nan"))
        empty["ref_dosage"].fill_(float("nan"))
        only_common = {key: self.batch[key] for key in ("common_context", "candidate_mask", "baseline", "coords")}
        for variant in VARIANTS:
            model = mixture_model(variant)
            nonzero_heads(model)
            common = model(only_common, "common")
            torch.testing.assert_close(model(no_reference), common, rtol=0, atol=0)
            for arm in ("carrier", "pooled", "perturbed"):
                torch.testing.assert_close(model(missing, arm), common, rtol=0, atol=0)
                torch.testing.assert_close(model(empty, arm), empty["baseline"], rtol=0, atol=0)

    def test_joint_permutations_and_pooled_isolation(self) -> None:
        permuted = clone_batch(self.batch)
        for key in ("common_context", "candidate_mask", "ref_dosage", "rare_ref_observed"):
            permuted[key] = permuted[key].flip(2).flip(4)
        corrupted = clone_batch(self.batch)
        corrupted["ref_dosage"] = torch.full((1,), float("nan"))
        corrupted["rare_ref_observed"] = torch.full((1,), float("nan"))
        for variant in VARIANTS:
            model = mixture_model(variant)
            nonzero_heads(model)
            for arm in ("common", "pooled", "carrier", "perturbed"):
                torch.testing.assert_close(model(self.batch, arm), model(permuted, arm), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(model(self.batch, "pooled"), model(corrupted, "pooled"), rtol=0, atol=0)

    def test_identical_input_balanced_negative_respects_entropy_bound(self) -> None:
        negative = {key: (value.repeat_interleave(2, dim=0) if key != "coords" else value)
                    for key, value in self.batch.items()}
        labels = self.labels.repeat_interleave(2, dim=0)
        labels[1::2] = torch.where(labels[1::2] == 1, 4, 1)
        for variant in VARIANTS:
            model = mixture_model(variant)
            nonzero_heads(model)
            for arm in ("common", "pooled", "carrier"):
                self.assertGreaterEqual(float(log_loss(model(negative, arm), labels).detach()), math.log(2) - 1e-6)


if __name__ == "__main__":
    unittest.main()
