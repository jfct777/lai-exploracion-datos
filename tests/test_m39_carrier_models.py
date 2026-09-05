from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from m39_carrier_models import (  # noqa: E402
    VARIANTS, CarrierContextModel, paired_reference_dosage_distribution,
    pooled_reference_summary,
)


def interaction_fixture(repeats: int = 3) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Balanced pure carrier-context interaction with identical pooled margins.

    Query G=1 is fixed. AFR/NAM each have one heterozygous carrier among two
    references, but opposite carrier contexts. A second EUR homologue has G=0.
    This is an engineered capacity fixture, not a population simulation.
    """
    combinations = [(-1, -1), (-1, 1), (1, -1), (1, 1)] * repeats
    people = len(combinations)
    context = torch.zeros(people, 1, 2, 3, 2, 2)
    mask = torch.zeros(context.shape[:-1], dtype=torch.bool)
    dosage = torch.zeros(mask.shape)
    labels = torch.zeros(people, 1, dtype=torch.long)
    for i, (query_context, carrier_context) in enumerate(combinations):
        for ancestry in (0, 2):
            mask[i, 0, 0, ancestry] = True
            context[i, 0, 0, ancestry, :, 0] = torch.tensor([-query_context, query_context])
        mask[i, 0, 1, 1] = True
        dosage[i, 0, 0, 0, int(carrier_context == 1)] = 1
        dosage[i, 0, 0, 2, int(carrier_context == -1)] = 1
        labels[i, 0] = 1 if query_context == carrier_context else 4
    baseline = torch.zeros(people, 1, 6)
    baseline[..., 1] = baseline[..., 4] = 0.5
    summary = torch.tensor([[0.5, 0.5, 0., 1.], [1., 0., 0., 1.], [0.5, 0.5, 0., 1.]])
    return {
        "common_context": context, "candidate_mask": mask,
        "ref_dosage": dosage, "rare_ref_observed": mask.clone(),
        "query_dosage": torch.ones(people, 1), "query_observed": torch.ones(people, 1, dtype=torch.bool),
        "baseline": baseline, "coords": torch.zeros(1),
        "pooled_summary": summary[None, None].expand(people, 1, 3, 4).clone(),
    }, labels


def clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in batch.items()}


def nonzero_heads(model: CarrierContextModel) -> None:
    with torch.no_grad():
        for head in (model.common_head, model.rare_head):
            head[-1].weight.normal_(std=0.2)
            head[-1].bias.normal_(std=0.1)


def log_loss(probability: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -probability.gather(-1, labels[..., None]).squeeze(-1).clamp_min(1e-12).log().mean()


class CarrierModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(17)
        self.batch, self.labels = interaction_fixture()

    def test_exact_shared_reference_phase_latent(self) -> None:
        self.assertEqual(paired_reference_dosage_distribution(1, 1, same_individual=True), (0., 1., 0.))
        self.assertEqual(paired_reference_dosage_distribution(
            1, 1, same_individual=True, second_homolog=0), (0.5, 0., 0.5))
        self.assertEqual(paired_reference_dosage_distribution(1, 1, same_individual=False), (0.25, 0.5, 0.25))
        self.assertEqual(paired_reference_dosage_distribution(0, 2, same_individual=False), (0., 1., 0.))
        with self.assertRaises(ValueError):
            paired_reference_dosage_distribution(1, 2, same_individual=True)
        with self.assertRaises(ValueError):
            paired_reference_dosage_distribution(-1, 0, same_individual=False)

    def test_global_pooled_summary_counts_people_and_missing(self) -> None:
        dosage = torch.tensor([[0., 2.], [1., float("nan")], [2., 0.]])
        observed = torch.tensor([[True, True], [True, False], [True, True]])
        summary = pooled_reference_summary(dosage, observed, torch.tensor([0, 0, 1]))
        torch.testing.assert_close(summary[0, 0], torch.tensor([0.5, 0.5, 0., 1.]))
        torch.testing.assert_close(summary[1, 0], torch.tensor([0., 0., 1., 0.5]))
        self.assertEqual(float(summary[:, 2].abs().sum()), 0.)

    def test_initial_baseline_and_bounded_gates(self) -> None:
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = CarrierContextModel(2, variant, width=16)
                torch.testing.assert_close(model(self.batch), self.batch["baseline"], rtol=0, atol=0)
                nonzero_heads(model)
                for arm in ("common", "pooled", "carrier", "perturbed"):
                    result = model(self.batch, arm, return_aux=True)
                    probabilities = result["probabilities"]
                    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(self.labels).float())
                    self.assertTrue(torch.isfinite(probabilities).all())
                    self.assertTrue(((result["rare_gate"] >= 0) & (result["rare_gate"] <= 1)).all())
                    self.assertTrue((result["rare_delta"].abs() <= model.residual_bound).all())

    def test_common_does_not_even_require_rare_inputs(self) -> None:
        only_common = {key: self.batch[key] for key in ("common_context", "candidate_mask", "baseline", "coords")}
        corrupted = clone_batch(self.batch)
        for key in ("ref_dosage", "rare_ref_observed", "query_dosage", "query_observed", "pooled_summary"):
            corrupted[key] = torch.full((1,), float("nan"))
        for variant in VARIANTS:
            model = CarrierContextModel(2, variant, width=16)
            nonzero_heads(model)
            torch.testing.assert_close(model(only_common, "common"), model(corrupted, "common"), rtol=0, atol=0)

    def test_pooled_never_uses_top_k_rare_data(self) -> None:
        corrupted = clone_batch(self.batch)
        corrupted["ref_dosage"] = torch.full((1,), float("nan"))
        corrupted["rare_ref_observed"] = torch.full((1,), float("nan"))
        for variant in VARIANTS:
            model = CarrierContextModel(2, variant, width=16)
            nonzero_heads(model)
            torch.testing.assert_close(model(self.batch, "pooled"), model(corrupted, "pooled"), rtol=0, atol=0)
            missing_summary = {key: value for key, value in self.batch.items() if key != "pooled_summary"}
            with self.assertRaisesRegex(ValueError, "GLOBAL"):
                model(missing_summary, "pooled")

    def test_candidate_permutation_and_homolog_swap(self) -> None:
        permuted = clone_batch(self.batch)
        swapped = clone_batch(self.batch)
        for key in ("common_context", "candidate_mask", "ref_dosage", "rare_ref_observed"):
            permuted[key] = self.batch[key].flip(4)
            swapped[key] = self.batch[key].flip(2)
        for variant in VARIANTS:
            model = CarrierContextModel(2, variant, width=16)
            nonzero_heads(model)
            for arm in ("common", "pooled", "carrier"):
                torch.testing.assert_close(model(self.batch, arm), model(permuted, arm), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(model(self.batch, arm), model(swapped, arm), atol=1e-6, rtol=1e-6)

    def test_carrier_shuffle_changes_representation_not_pooled(self) -> None:
        shuffled = clone_batch(self.batch)
        shuffled["ref_dosage"] = shuffled["ref_dosage"].flip(4)
        torch.testing.assert_close(shuffled["ref_dosage"].sum(4), self.batch["ref_dosage"].sum(4))
        for variant in VARIANTS:
            model = CarrierContextModel(2, variant, width=16)
            nonzero_heads(model)
            first = model(self.batch, return_aux=True)["rare_representation"]
            second = model(shuffled, "perturbed", return_aux=True)["rare_representation"]
            self.assertGreater(float((first - second).detach().abs().max()), 1e-5)
            for arm in ("common", "pooled"):
                torch.testing.assert_close(model(self.batch, arm), model(shuffled, arm), rtol=0, atol=0)

    def test_missing_and_empty_panel_return_common_or_baseline(self) -> None:
        missing = clone_batch(self.batch)
        missing["query_observed"].fill_(False)
        missing["query_dosage"].fill_(float("nan"))
        empty = clone_batch(self.batch)
        empty["candidate_mask"].fill_(False)
        empty["common_context"].fill_(float("nan"))
        empty["ref_dosage"].fill_(float("nan"))
        for variant in VARIANTS:
            model = CarrierContextModel(2, variant, width=16)
            nonzero_heads(model)
            for arm in ("carrier", "pooled"):
                torch.testing.assert_close(model(missing, arm), model(self.batch, "common"), rtol=0, atol=0)
                torch.testing.assert_close(model(empty, arm), empty["baseline"], rtol=0, atol=0)

    def test_invalid_observed_genotypes_are_rejected(self) -> None:
        invalid = clone_batch(self.batch)
        invalid["query_dosage"].fill_(0.5)
        with self.assertRaisesRegex(ValueError, "diploid"):
            CarrierContextModel(2)(invalid)

    def test_exact_balanced_negative_cannot_beat_entropy(self) -> None:
        # Each exact input occurs with both labels: no predictor can beat log(2).
        negative = {key: (value.repeat_interleave(2, dim=0) if key != "coords" else value)
                    for key, value in self.batch.items()}
        labels = self.labels.repeat_interleave(2, dim=0)
        labels[1::2] = torch.where(labels[1::2] == 1, 4, 1)
        for variant in VARIANTS:
            model = CarrierContextModel(2, variant, width=16)
            nonzero_heads(model)
            self.assertGreaterEqual(float(log_loss(model(negative), labels).detach()), math.log(2) - 1e-6)

    def test_small_positive_capacity_for_each_variant(self) -> None:
        # Fixed synthetic optimization budget; no real-data fitting or selection.
        for variant in VARIANTS:
            for signal in ("interaction", "additive"):
                with self.subTest(variant=variant, signal=signal):
                    torch.manual_seed(7)
                    batch, labels = interaction_fixture(repeats=1)
                    if signal == "additive":
                        batch["common_context"].zero_()
                        batch["ref_dosage"].zero_()
                        batch["query_dosage"] = (labels == 1).float()
                    model = CarrierContextModel(2, variant, width=16)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
                    for _ in range(100):
                        optimizer.zero_grad(set_to_none=True)
                        loss = log_loss(model(batch), labels)
                        loss.backward()
                        optimizer.step()
                    # A permutation checks the learned finite truth table, not
                    # generalization to independent biological founders.
                    evaluation = clone_batch(batch)
                    for key in ("common_context", "candidate_mask", "ref_dosage", "rare_ref_observed"):
                        evaluation[key] = evaluation[key].flip(4).flip(2)
                    model.eval()
                    with torch.no_grad():
                        probability = model(evaluation)
                        self.assertLess(float(log_loss(probability, labels)), 0.15)
                        self.assertGreaterEqual(float((probability.argmax(-1) == labels).float().mean()), 0.95)
                    if signal == "interaction":
                        # All pooled/common inputs remain label-uninformative.
                        self.assertGreaterEqual(float(log_loss(model(batch, "pooled"), labels).detach()), math.log(2) - 1e-6)
                        self.assertGreaterEqual(float(log_loss(model(batch, "common"), labels).detach()), math.log(2) - 1e-6)


if __name__ == "__main__":
    unittest.main()
