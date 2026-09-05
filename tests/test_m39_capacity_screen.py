from pathlib import Path
import math
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from m39_capacity_screen import interaction_fixture, score, select_people


class CapacityFixtureTest(unittest.TestCase):
    def test_common_and_pooled_identical_with_opposite_labels(self):
        batch, labels = interaction_fixture(8, 17)
        for i in range(0, len(labels), 2):
            self.assertNotEqual(labels[i].item(), labels[i+1].item())
            for name in ('common_context', 'candidate_mask', 'baseline', 'pooled_summary',
                         'query_dosage', 'query_observed', 'rare_ref_observed'):
                torch.testing.assert_close(batch[name][i], batch[name][i+1], rtol=0, atol=0)
            self.assertFalse(torch.equal(batch['ref_dosage'][i], batch['ref_dosage'][i+1]))

    def test_baseline_and_exact_null_bound(self):
        batch, labels = interaction_fixture(4, 19)
        result = score(batch['baseline'], labels)
        self.assertAlmostEqual(result['accuracy'], .5)
        self.assertGreaterEqual(result['log_loss'], math.log(2)-1e-6)
        torch.manual_seed(1)
        random_probs = torch.randn(*batch['baseline'].shape).softmax(-1)
        self.assertGreaterEqual(score(random_probs, labels)['identical_input_null_log_loss'],
                                math.log(2)-1e-6)

    def test_minibatch_preserves_coordinate_axis(self):
        batch, labels = interaction_fixture(2, 1)
        subset = select_people(batch, torch.tensor([3, 1]))
        self.assertEqual(subset['common_context'].shape[:2], (2, 1))
        self.assertEqual(subset['coords'].shape, (1,))

    def test_wrong_baselines_do_not_change_labels_or_visible_context(self):
        original, labels = interaction_fixture(3, 51)
        for mode in ('zero_wrong', 'floor_wrong'):
            batch, other = interaction_fixture(3, 51, mode)
            torch.testing.assert_close(labels, other, rtol=0, atol=0)
            for key in set(batch)-{'baseline'}:
                torch.testing.assert_close(original[key], batch[key], rtol=0, atol=0)
            self.assertEqual(score(batch['baseline'], labels)['accuracy'], 0.)
            self.assertEqual(score(batch['baseline'], labels)['zero_true_state_probability_fraction'],
                             1. if mode == 'zero_wrong' else 0.)
            self.assertEqual(bool((batch['baseline'][..., 1] == 0).all()), mode == 'zero_wrong')


if __name__ == '__main__':
    unittest.main()
