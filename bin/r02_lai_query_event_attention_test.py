"""Focused operational tests for the query-to-EVENT readout.

Synthetic equivalence and gradient checks do not establish biological benefit.
"""
import sys
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from genomic_context_runtime import ContextConfig, Sites, build_model, local_mean
from r02_lai_common_context import LOCAL_MEAN_V1
from r02_lai_query_event_attention import (
    QUERY_EVENT_CONTENT_V1,
    QUERY_EVENT_RELATIVE_V1,
    QueryEventAttention,
    initialize_from_mean_model,
)
from r02_local_attention_blocks import ContextExecution


def execution(block=2, checkpoint=True):
    return ContextExecution(query_block_size=block, max_mask_elements=100_000,
                            checkpoint_blocks=checkpoint)


def sites(features, cm, chrom, valid, *, requires_grad=False):
    values = torch.tensor(features, dtype=torch.float64, requires_grad=requires_grad)
    return Sites(values, torch.tensor(cm, dtype=torch.float64),
                 torch.tensor(chrom, dtype=torch.int64), torch.tensor(valid))


def recipe(mode=None, family="lai_local_attention"):
    result = {
        "family": family,
        "common_radius_cm": .25,
        "rare_radius_cm": .5,
        "width": 4,
        "depth": 1,
        "dropout": 0.,
        "heads": 2,
        "learning_rate": .001,
    }
    if mode is not None:
        result["lai_query_event_mode"] = mode
    return result


class QueryEventAttentionTest(unittest.TestCase):
    def test_initial_readout_is_uniform_mean_with_identical_counts(self):
        layer = QueryEventAttention(3, 4, 2, execution(),
                                    QUERY_EVENT_RELATIVE_V1).double()
        query = sites([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]],
                      [[0., .5, 2.]], [[1, 1, 1]], [[True, True, True]])
        source = sites([[[0.] * 4] * 5],
                       [[-.5, 0., .5, 1., 2.]], [[1] * 5], [[True] * 5])
        values = torch.tensor([[[1., 0., 0., 2.], [0., 1., 0., 4.],
                                [0., 0., 1., 6.], [1., 1., 0., 8.],
                                [2., 0., 1., 10.]]], dtype=torch.float64)

        expected, expected_count = local_mean(
            values, query, source, .5, execution=execution())
        actual, actual_count = layer(
            query.features, query.cm, query.chrom, query.valid,
            values, source.cm, source.chrom, source.valid, .5)

        self.assertTrue(torch.equal(actual_count, expected_count))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_relative_position_distinguishes_reversal_but_content_control_does_not(self):
        relative = QueryEventAttention(1, 2, 1, execution(),
                                       QUERY_EVENT_RELATIVE_V1).double()
        with torch.no_grad():
            relative.relative_position.weight[0, 0] = 4.
        content = QueryEventAttention(1, 2, 1, execution(),
                                      QUERY_EVENT_CONTENT_V1).double()
        content.load_state_dict(relative.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in relative.parameters()),
                         sum(p.numel() for p in content.parameters()))

        query = torch.ones((1, 1, 1), dtype=torch.float64)
        query_cm = torch.zeros((1, 1), dtype=torch.float64)
        query_chrom = torch.ones((1, 1), dtype=torch.int64)
        query_valid = torch.ones((1, 1), dtype=torch.bool)
        event_cm = torch.tensor([[-.5, .5]], dtype=torch.float64)
        event_chrom = torch.ones((1, 2), dtype=torch.int64)
        event_valid = torch.ones((1, 2), dtype=torch.bool)
        left_a = torch.tensor([[[1., 0.], [0., 1.]]], dtype=torch.float64)
        left_b = left_a.flip(1)

        def read(layer, values):
            return layer(query, query_cm, query_chrom, query_valid, values,
                         event_cm, event_chrom, event_valid, .5)[0]

        relative_a, relative_b = read(relative, left_a), read(relative, left_b)
        content_a, content_b = read(content, left_a), read(content, left_b)
        self.assertFalse(torch.allclose(relative_a, relative_b))
        torch.testing.assert_close(content_a, content_b, atol=0, rtol=0)
        torch.testing.assert_close(content_a,
                                   torch.tensor([[[.5, .5]]], dtype=torch.float64),
                                   atol=0, rtol=0)

    def test_masks_exact_boundary_and_no_neighbor_return_zero(self):
        layer = QueryEventAttention(1, 2, 1, execution(),
                                    QUERY_EVENT_RELATIVE_V1).double()
        query = torch.tensor([[[1.], [2.], [float("nan")]]], dtype=torch.float64)
        query_cm = torch.tensor([[0., 2., float("nan")]], dtype=torch.float64)
        query_chrom = torch.tensor([[1, 1, 1]], dtype=torch.int64)
        query_valid = torch.tensor([[True, True, False]])
        values = torch.tensor([[[2., 0.], [0., 4.], [float("nan"), float("nan")],
                                [50., 50.]]], dtype=torch.float64)
        event_cm = torch.tensor([[-.5, .5, float("nan"), 0.]], dtype=torch.float64)
        event_chrom = torch.tensor([[1, 1, 1, 2]], dtype=torch.int64)
        event_valid = torch.tensor([[True, True, False, True]])

        output, count = layer(query, query_cm, query_chrom, query_valid, values,
                              event_cm, event_chrom, event_valid, .5)
        self.assertEqual(count.tolist(), [[2, 0, 0]])
        torch.testing.assert_close(output[0, 0], torch.tensor([1., 2.], dtype=torch.float64),
                                   atol=0, rtol=0)
        self.assertEqual(output[0, 1:].abs().sum().item(), 0.)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_blocked_forward_and_gradients_match_and_are_finite(self):
        torch.manual_seed(1719)
        whole = QueryEventAttention(3, 4, 2, execution(20, False),
                                    QUERY_EVENT_RELATIVE_V1).double()
        blocked = QueryEventAttention(3, 4, 2, execution(2, True),
                                      QUERY_EVENT_RELATIVE_V1).double()
        with torch.no_grad():
            whole.common_key.weight.copy_(torch.eye(4, dtype=torch.float64))
            whole.common_key.bias.fill_(.1)
            whole.relative_position.weight.copy_(
                torch.tensor([[.3, -.2], [-.1, .4]], dtype=torch.float64))
        blocked.load_state_dict(whole.state_dict(), strict=True)

        query_features = torch.tensor(
            [[[1., 0., 2.], [0., 2., 1.], [3., 1., 0.]],
             [[2., 1., 0.], [1., 3., 2.], [float("nan")] * 3]],
            dtype=torch.float64)
        source_values = torch.tensor(
            [[[1., 0., 2., 1.], [0., 1., 1., 2.], [2., 1., 0., 3.],
              [1., 2., 3., 0.], [float("nan")] * 4],
             [[2., 0., 1., 1.], [0., 2., 1., 3.], [1., 1., 2., 0.],
              [3., 1., 0., 2.], [2., 2., 2., 2.]]], dtype=torch.float64)
        qcm = torch.tensor([[0., .4, .8], [0., .5, float("nan")]], dtype=torch.float64)
        qchrom = torch.tensor([[1, 1, 1], [1, 2, 1]], dtype=torch.int64)
        qvalid = torch.tensor([[True, True, True], [True, True, False]])
        ecm = torch.tensor([[-.2, .2, .5, .9, float("nan")],
                            [-.1, .2, .4, .7, 1.]], dtype=torch.float64)
        echrom = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 2, 2, 2]], dtype=torch.int64)
        evalid = torch.tensor([[True, True, True, True, False], [True] * 5])

        left_q = query_features.clone().requires_grad_()
        right_q = query_features.clone().requires_grad_()
        left_v = source_values.clone().requires_grad_()
        right_v = source_values.clone().requires_grad_()
        left, left_count = whole(left_q, qcm, qchrom, qvalid, left_v,
                                 ecm, echrom, evalid, .5)
        right, right_count = blocked(right_q, qcm, qchrom, qvalid, right_v,
                                     ecm, echrom, evalid, .5)
        self.assertTrue(torch.equal(left_count, right_count))
        torch.testing.assert_close(left, right, atol=2e-12, rtol=2e-12)
        weight = torch.arange(left.numel(), dtype=torch.float64).reshape_as(left) + 1
        (left * weight).sum().backward()
        (right * weight).sum().backward()
        for a, b in ((left_q.grad, right_q.grad), (left_v.grad, right_v.grad)):
            self.assertTrue(bool(torch.isfinite(a).all()))
            self.assertTrue(bool(torch.isfinite(b).all()))
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)
        self.assertEqual(left_q.grad[1, 2].abs().sum().item(), 0.)
        self.assertEqual(left_v.grad[0, 4].abs().sum().item(), 0.)
        for a, b in zip(whole.parameters(), blocked.parameters()):
            self.assertIsNotNone(a.grad)
            self.assertIsNotNone(b.grad)
            self.assertTrue(bool(torch.isfinite(a.grad).all()))
            self.assertTrue(bool(torch.isfinite(b.grad).all()))
            torch.testing.assert_close(a.grad, b.grad, atol=2e-11, rtol=2e-11)
        self.assertGreater(whole.event_query.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(whole.common_key.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(whole.relative_position.weight.grad.abs().sum().item(), 0.)


class RuntimeIntegrationTest(unittest.TestCase):
    def model_inputs(self):
        common = sites([[[1.], [2.], [3.], [float("nan")]]],
                       [[0., .2, .4, float("nan")]], [[1, 1, 1, 1]],
                       [[True, True, True, False]])
        events = sites([[[2.], [4.], [float("nan")]]],
                       [[.1, .5, float("nan")]], [[1, 1, 1]],
                       [[True, True, False]])
        queries = sites([[[0.], [0.]]], [[.1, .5]], [[1, 1]], [[True, True]])
        probabilities = torch.tensor(
            [[[.1, .2, .2, .1, .2, .2], [.2, .1, .2, .2, .1, .2]]],
            dtype=torch.float64)
        burden = torch.tensor([[[1., 1., 2.], [0., 0., 2.]]], dtype=torch.float64)
        return common, events, queries, probabilities, burden

    def build(self, mode):
        return build_model(recipe(mode), 1, 1, max_common_sites=10,
                           max_events=10, max_queries=10,
                           context_execution={
                               "query_block_size": 2,
                               "max_mask_elements": 10_000,
                               "checkpoint_blocks": True,
                           }, lai_query_burden_features=3).double()

    def test_mean_checkpoint_import_preserves_complete_model_output(self):
        torch.manual_seed(2207)
        mean_model = self.build(LOCAL_MEAN_V1)
        torch.manual_seed(9999)
        attention_model = self.build(QUERY_EVENT_RELATIVE_V1)
        receipt = initialize_from_mean_model(attention_model, mean_model.state_dict())
        self.assertFalse(receipt["optimizer_state_imported"])
        self.assertEqual(receipt["semantics"], "WEIGHT_INITIALIZATION_NOT_TRAINING_RESUME")
        args = self.model_inputs()
        expected = mean_model.predict_lai(args[0], args[2], args[3], args[1],
                                          query_burden=args[4])
        actual = attention_model.predict_lai(args[0], args[2], args[3], args[1],
                                             query_burden=args[4])
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)

    def test_strict_resume_and_mean_import_reject_partial_or_new_state(self):
        model = self.build(QUERY_EVENT_RELATIVE_V1)
        state = model.state_dict()
        resumed = self.build(QUERY_EVENT_RELATIVE_V1)
        resumed.load_state_dict(state, strict=True)
        for key in state:
            torch.testing.assert_close(resumed.state_dict()[key], state[key], atol=0, rtol=0)

        partial_new = dict(state)
        partial_new.pop(next(iter(partial_new)))
        with self.assertRaises(RuntimeError):
            resumed.load_state_dict(partial_new, strict=True)
        with self.assertRaisesRegex(ValueError, "historical mean model"):
            initialize_from_mean_model(resumed, partial_new)
        with self.assertRaisesRegex(ValueError, "historical mean model"):
            initialize_from_mean_model(resumed, state)

        mean = self.build(LOCAL_MEAN_V1).state_dict()
        partial_mean = dict(mean)
        partial_mean.pop(next(iter(partial_mean)))
        with self.assertRaisesRegex(ValueError, "historical mean model"):
            initialize_from_mean_model(resumed, partial_mean)

    def test_query_event_mode_is_rejected_for_structure(self):
        for mode in (QUERY_EVENT_RELATIVE_V1, QUERY_EVENT_CONTENT_V1):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "applies only to LAI"):
                    ContextConfig.from_recipe(recipe(mode, "structure_local_attention"))


if __name__ == "__main__":
    unittest.main()
