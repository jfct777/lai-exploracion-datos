"""Adversarial regional-loss checks without human arrays."""
import copy
import math
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import torch
import r02_context_train as trainer
from r02_lai_query_score import QueryScore, RULE


def batch(coordinates, core=None):
    n = len(coordinates)
    return dict(person_id=np.asarray(["p"]), component_id=np.asarray(["g"]),
                query_valid=np.ones((1, n), bool), truth_observed=np.ones((1, n), bool),
                query_core=np.asarray([core if core is not None else [True] * n]),
                query_chrom=np.full((1, n), 22), query_pos_bp=np.asarray([coordinates]),
                truth_state=np.zeros((1, n), np.int64))


def probability(p):
    return np.asarray([[[value] + [(1-value)/5]*5 for value in p]], np.float64)


class RegionalScoringTest(unittest.TestCase):
    def test_halo_truth_cannot_change_loss_or_gradient(self):
        data = batch([1, 2, 3], [False, True, False])
        logits = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6).requires_grad_()
        first = trainer.training_loss(logits.softmax(-1), data, "lai")
        first.backward(); gradient = logits.grad.clone(); logits.grad.zero_()
        changed = copy.deepcopy(data); changed["truth_state"][0, [0, 2]] = [4, 5]
        second = trainer.training_loss(logits.softmax(-1), changed, "lai")
        second.backward()
        self.assertEqual(first.item(), second.item())
        self.assertTrue(torch.equal(gradient, logits.grad))
        self.assertEqual(float(gradient[0, [0, 2]].abs().sum()), 0.0)
        self.assertTrue(data["query_valid"].all())

    def test_empty_owned_truth_rejected_and_legacy_kept(self):
        data = batch([1], [False])
        with self.assertRaises(ValueError): trainer.lai_loss_mask(data)
        del data["query_core"]
        self.assertTrue(trainer.lai_loss_mask(data).all())

    def test_unequal_regions_equal_continuous_score(self):
        separated, continuous = QueryScore(), QueryScore()
        for coords, values in [([1], [.2]), ([2, 3, 4], [.8, .8, .8])]:
            data = batch(coords)
            separated.add("case", data, probability(values), trainer.lai_loss_mask(data))
        data = batch([1, 2, 3, 4])
        continuous.add("case", data, probability([.2, .8, .8, .8]), trainer.lai_loss_mask(data))
        self.assertAlmostEqual(separated.result()[0], continuous.result()[0])
        self.assertNotAlmostEqual(separated.result()[0], (-math.log(.2)-math.log(.8))/2)

    def test_duplicate_core_or_changed_component_rejected(self):
        score = QueryScore(); data = batch([1, 2])
        score.add("case", data, probability([.5, .5]), trainer.lai_loss_mask(data))
        with self.assertRaisesRegex(ValueError, "repeated"):
            score.add("case", data, probability([.5, .5]), trainer.lai_loss_mask(data))
        changed = batch([3, 4]); changed["component_id"] = np.asarray(["other"])
        with self.assertRaisesRegex(ValueError, "axes changed"):
            score.add("case", changed, probability([.5, .5]), trainer.lai_loss_mask(changed))

    def test_same_person_name_in_distinct_cases_does_not_merge(self):
        score = QueryScore()
        for case, values in [("a", [.2]), ("b", [.8, .8])]:
            data = batch(list(range(1, len(values)+1)))
            score.add(case, data, probability(values), trainer.lai_loss_mask(data))
        self.assertAlmostEqual(score.result()[0], (-math.log(.2)-math.log(.8))/2)
        self.assertEqual(len(score.result()[1]), 2)

    def test_trainer_select_uses_authenticated_query_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); entries, source = [], {}
            for i, values in enumerate(([.2], [.8, .8, .8])):
                path = root / f"receipt-{i}.json"
                trainer.write_json(path, {"case_id": "case"})
                entry = {"receipt_path": str(path), "receipt_sha256": trainer.sha(path), "index": i}
                entries.append(entry)
                data = batch([1] if i == 0 else [2, 3, 4])
                source[i] = (torch.from_numpy(probability(values)), data, None)
            contract = {"objective": "lai", "lai_training_protocol": {"path": "fixture", "sha256": "fixture"},
                        "batches": {"SELECT": entries}}
            protocol = {"training_unit": {"selection_loss_aggregation": RULE}}
            with patch("r02_lai_training_campaign.validate_protocol", return_value=(protocol, "fixture")), \
                 patch.object(trainer, "predict_training_entry", side_effect=lambda m, e, *a, **k: source[e["index"]]):
                value, _, _ = trainer.score_batches(torch.nn.Identity(), contract, "SELECT", {}, "NONE", root)
            self.assertAlmostEqual(value, (-math.log(.2)-3*math.log(.8))/4)


if __name__ == "__main__":
    unittest.main()
