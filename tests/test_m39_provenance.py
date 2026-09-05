"""Independent provenance and identifiability checks using artificial inputs only."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from m39_capacity_screen import interaction_fixture, score  # noqa: E402
from m39_carrier_models import (  # noqa: E402
    CarrierContextModel, VARIANTS, pooled_reference_summary,
)


class CapacityProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.batch, self.labels = interaction_fixture(5, 701)

    def test_every_control_visible_input_has_opposite_balanced_labels(self) -> None:
        # Adjacent examples differ only in carrier assignment, never common noise.
        for key in ("common_context", "candidate_mask", "baseline", "query_dosage",
                    "query_observed", "pooled_summary"):
            torch.testing.assert_close(self.batch[key][0::2], self.batch[key][1::2],
                                       rtol=0, atol=0)
        self.assertTrue(bool((self.labels[0::2] != self.labels[1::2]).all()))
        self.assertEqual(int((self.labels == 1).sum()), 10)
        self.assertEqual(int((self.labels == 4).sum()), 10)

    def test_pooled_fixture_matches_unique_diploid_people_not_haplotypes(self) -> None:
        ancestry = torch.tensor([0, 0, 1, 1, 2, 2])
        for index in range(len(self.labels)):
            # The six artificial references are 2 people in each ancestry.
            dosage = self.batch["ref_dosage"][index, 0, 0].reshape(6, 1)
            observed = torch.ones_like(dosage, dtype=torch.bool)
            expected = pooled_reference_summary(dosage, observed, ancestry)
            torch.testing.assert_close(expected, self.batch["pooled_summary"][index],
                                       rtol=0, atol=0)

    def test_carrier_assignment_changes_without_changing_margins(self) -> None:
        first = self.batch["ref_dosage"][0::2]
        second = self.batch["ref_dosage"][1::2]
        self.assertTrue(bool((first != second).flatten(1).any(1).all()))
        torch.testing.assert_close(first.sum(-1), second.sum(-1), rtol=0, atol=0)

    def test_identical_input_null_floor_is_an_analytic_metric_check(self) -> None:
        probabilities = self.batch["baseline"].clone()
        result = score(probabilities, self.labels)
        self.assertGreaterEqual(result["identical_input_null_log_loss"], math.log(2)-1e-6)
        probabilities.zero_()
        probabilities[..., 1] = .9
        probabilities[..., 4] = .1
        result = score(probabilities, self.labels)
        self.assertGreater(result["identical_input_null_log_loss"], math.log(2))

    def test_all_reference_missing_is_neutral_even_after_heads_change(self) -> None:
        self.batch["rare_ref_observed"].fill_(False)
        # A sentinel must not become a called reference genotype.
        self.batch["ref_dosage"].fill_(-1)
        for variant in VARIANTS:
            model = CarrierContextModel(4, variant, width=16)
            with torch.no_grad():
                model.common_head[-1].bias.normal_()
                model.rare_head[-1].bias.normal_()
                model.rare_head[-1].weight.normal_()
            common = model(self.batch, "common")
            carrier = model(self.batch, "carrier", return_aux=True)
            torch.testing.assert_close(common, carrier["probabilities"], rtol=0, atol=0)
        self.assertEqual(float(carrier["rare_delta"].detach().abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
