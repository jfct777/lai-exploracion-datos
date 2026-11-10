import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from genomic_context_runtime import ContextConfig, Sites, build_model
from r02_lai_common_context import (EVENT_QUERY_COMMON_KV_V1, LOCAL_MEAN_V1,
    EventQueryCommonAttention, event_common_attention_parameter_count)
from r02_local_attention_blocks import ContextExecution


def recipe(mode=None):
    result = {
        "family": "lai_local_attention", "common_radius_cm": .2,
        "rare_radius_cm": .2, "width": 8, "depth": 2, "dropout": 0.,
        "heads": 2, "learning_rate": .001,
    }
    if mode is not None:
        result["lai_common_context_mode"] = mode
    return result


def attention_inputs():
    event_features = torch.tensor([
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [float("nan")] * 3],
        [[2., 1., 0.], [3., 2., 1.], [4., 3., 2.], [5., 4., 3.]],
    ], dtype=torch.float64)
    event_cm = torch.tensor([[.10, .30, 1.00, float("nan")],
                             [.05, .45, .90, 1.20]], dtype=torch.float64)
    event_chrom = torch.tensor([[1, 1, 2, 1], [1, 1, 2, 2]])
    event_valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    common_values = torch.arange(2 * 6 * 8, dtype=torch.float64).reshape(2, 6, 8) / 50
    common_values[0, 5] = float("nan")
    common_cm = torch.tensor([[.00, .20, .50, .90, 1.10, float("nan")],
                              [.00, .20, .50, .85, 1.10, 1.50]], dtype=torch.float64)
    common_chrom = torch.tensor([[1, 1, 1, 2, 2, 1], [1, 1, 1, 2, 2, 2]])
    common_valid = torch.tensor([[True, True, True, True, True, False],
                                 [True, True, True, True, True, True]])
    return (event_features, event_cm, event_chrom, event_valid,
            common_values, common_cm, common_chrom, common_valid)


def differentiable_copy(values):
    result = []
    for index, value in enumerate(values):
        clone = value.detach().clone()
        if index in (0, 4):
            clone.requires_grad_()
        result.append(clone)
    return result


class EventQueryCommonAttentionTest(unittest.TestCase):
    def test_dense_and_blocked_forward_and_gradients_match_with_invalid_nan_padding(self):
        torch.manual_seed(1701)
        dense = EventQueryCommonAttention(3, 8, 2).double()
        blocked = EventQueryCommonAttention(3, 8, 2, execution=ContextExecution(
            query_block_size=2, max_mask_elements=10_000, checkpoint_blocks=True)).double()
        blocked.load_state_dict(dense.state_dict(), strict=True)
        left, right = differentiable_copy(attention_inputs()), differentiable_copy(attention_inputs())
        a, na = dense(*left, .2)
        b, nb = blocked(*right, .2)
        self.assertTrue(torch.equal(na, nb))
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
        weight = torch.arange(a.numel(), dtype=a.dtype).reshape_as(a) / a.numel()
        (a * weight).sum().backward()
        (b * weight).sum().backward()
        for index in (0, 4):
            self.assertTrue(bool(torch.isfinite(left[index].grad).all()))
            self.assertTrue(bool(torch.isfinite(right[index].grad).all()))
            torch.testing.assert_close(left[index].grad, right[index].grad,
                                       atol=2e-11, rtol=2e-11)
        self.assertEqual(left[0].grad[0, 3].abs().sum().item(), 0.)
        self.assertEqual(left[4].grad[0, 5].abs().sum().item(), 0.)
        for a_parameter, b_parameter in zip(dense.parameters(), blocked.parameters()):
            self.assertIsNotNone(a_parameter.grad)
            self.assertIsNotNone(b_parameter.grad)
            torch.testing.assert_close(a_parameter.grad, b_parameter.grad,
                                       atol=2e-11, rtol=2e-11)

    def test_checkpoint_many_windows_shares_full_key_value_storage_and_matches_gradients(self):
        """Checkpoint must recreate, rather than retain, per-window KV copies."""
        batch, queries, common, features, width, heads = 2, 33, 80, 3, 8, 2
        execution = dict(query_block_size=2, max_mask_elements=100_000)
        torch.manual_seed(20261108)
        plain = EventQueryCommonAttention(
            features, width, heads, execution=ContextExecution(
                checkpoint_blocks=False, **execution)).double()
        checked = EventQueryCommonAttention(
            features, width, heads, execution=ContextExecution(
                checkpoint_blocks=True, **execution)).double()
        checked.load_state_dict(plain.state_dict(), strict=True)

        event = torch.randn(batch, queries, features, dtype=torch.float64)
        source = torch.randn(batch, common, width, dtype=torch.float64)
        event_cm = torch.linspace(.03, .97, queries, dtype=torch.float64).repeat(batch, 1)
        common_cm = torch.linspace(0., 1., common, dtype=torch.float64).repeat(batch, 1)
        event_chrom = torch.ones((batch, queries), dtype=torch.int64)
        common_chrom = torch.ones((batch, common), dtype=torch.int64)
        event_valid = torch.ones((batch, queries), dtype=torch.bool)
        common_valid = torch.ones((batch, common), dtype=torch.bool)

        left_event = event.detach().clone().requires_grad_()
        left_source = source.detach().clone().requires_grad_()
        expected, expected_count = plain(
            left_event, event_cm, event_chrom, event_valid,
            left_source, common_cm, common_chrom, common_valid, .08)

        checkpoint_calls, saved = [], []

        def storage(tensor):
            return tensor.untyped_storage().data_ptr()

        def observe_checkpoint(function, *args, **kwargs):
            checkpoint_calls.append({
                "name": function.__name__,
                "key_shape": tuple(args[1].shape),
                "value_shape": tuple(args[2].shape),
                "key_storage": storage(args[1]),
                "value_storage": storage(args[2]),
                "columns": int(args[9].numel()),
            })
            return torch_checkpoint(function, *args, **kwargs)

        def pack(tensor):
            saved.append({
                "shape": tuple(tensor.shape),
                "storage": storage(tensor),
                "bytes": tensor.untyped_storage().nbytes(),
            })
            return tensor

        right_event = event.detach().clone().requires_grad_()
        right_source = source.detach().clone().requires_grad_()
        with patch("r02_lai_common_context.checkpoint", side_effect=observe_checkpoint):
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                actual, actual_count = checked(
                    right_event, event_cm, event_chrom, event_valid,
                    right_source, common_cm, common_chrom, common_valid, .08)

        self.assertEqual(len(checkpoint_calls), 17)
        self.assertEqual({call["name"] for call in checkpoint_calls}, {"_attend_columns"})
        self.assertEqual({call["key_shape"] for call in checkpoint_calls},
                         {(batch, heads, common, width // heads)})
        self.assertEqual({call["value_shape"] for call in checkpoint_calls},
                         {(batch, heads, common, width // heads)})
        self.assertEqual(len({call["key_storage"] for call in checkpoint_calls}), 1)
        self.assertEqual(len({call["value_storage"] for call in checkpoint_calls}), 1)
        self.assertTrue(all(2 < call["columns"] < common for call in checkpoint_calls))

        # Non-reentrant checkpoint retains repeated references to the two full
        # projection storages.  It must not retain distinct [B,H,K,D] advanced-
        # indexing copies for the intermediate K selected columns of each block.
        full_storages = ({call["key_storage"] for call in checkpoint_calls}
                         | {call["value_storage"] for call in checkpoint_calls})
        self.assertTrue(full_storages.issubset({item["storage"] for item in saved}))
        shared_kv_bytes = sum(dict(
            (item["storage"], item["bytes"]) for item in saved
            if item["storage"] in full_storages).values())
        selected_kv_bytes_before_repair = sum(
            2 * batch * heads * call["columns"] * (width // heads) * 8
            for call in checkpoint_calls)
        self.assertEqual(sum(call["columns"] for call in checkpoint_calls), 260)
        self.assertEqual(shared_kv_bytes, 20_480)
        self.assertEqual(selected_kv_bytes_before_repair, 66_560)
        self.assertLess(shared_kv_bytes, selected_kv_bytes_before_repair)
        selected_copies = [
            item for item in saved
            if len(item["shape"]) == 4
            and item["shape"][0] == batch and item["shape"][1] == heads
            and 2 < item["shape"][2] < common
            and item["shape"][3] == width // heads
        ]
        self.assertEqual(selected_copies, [])

        self.assertTrue(torch.equal(actual_count, expected_count))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        weight = torch.linspace(.1, 1., expected.numel(), dtype=torch.float64).reshape_as(expected)
        (expected * weight).sum().backward()
        (actual * weight).sum().backward()
        torch.testing.assert_close(right_event.grad, left_event.grad, atol=2e-11, rtol=2e-11)
        torch.testing.assert_close(right_source.grad, left_source.grad, atol=2e-11, rtol=2e-11)
        for left, right in zip(plain.parameters(), checked.parameters()):
            torch.testing.assert_close(right.grad, left.grad, atol=2e-11, rtol=2e-11)

    def test_exact_observation_chromosome_and_radius_mask_and_position(self):
        layer = EventQueryCommonAttention(1, 2, 1).double()
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
            layer.common_value.weight.copy_(torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64))
            layer.output.weight.copy_(torch.eye(2, dtype=torch.float64))
            layer.relative_position.weight[0, 0] = 3.
        event = torch.ones(1, 2, 1, dtype=torch.float64)
        common = torch.tensor([[[1., 0.], [0., 2.], [50., 50.], [80., 80.]]], dtype=torch.float64)
        ecm = torch.tensor([[.2, 1.0]], dtype=torch.float64)
        ccm = torch.tensor([[.1, .3, .2, 1.2]], dtype=torch.float64)
        echrom = torch.tensor([[1, 2]])
        cchrom = torch.tensor([[1, 1, 2, 2]])
        evalid = torch.tensor([[True, True]])
        cvalid = torch.tensor([[True, True, True, False]])
        output, count = layer(event, ecm, echrom, evalid,
                              common, ccm, cchrom, cvalid, .2)
        self.assertEqual(count.tolist(), [[2, 0]])
        self.assertNotEqual(output[0, 0, 0].item(), output[0, 0, 1].item())
        self.assertEqual(output[0, 1].abs().sum().item(), 0.)

    def test_no_common_neighbor_is_finite_zero_with_zero_gradients(self):
        layer = EventQueryCommonAttention(2, 4, 2).double()
        event = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
        common = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        output, count = layer(event, torch.tensor([[0., 1.]], dtype=torch.float64),
            torch.tensor([[1, 1]]), torch.tensor([[True, True]]), common,
            torch.tensor([[10., 11., 12.]], dtype=torch.float64), torch.tensor([[2, 2, 2]]),
            torch.tensor([[True, True, True]]), .1)
        self.assertEqual(count.sum().item(), 0)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(output.abs().sum().item(), 0.)
        output.sum().backward()
        self.assertEqual(event.grad.abs().sum().item(), 0.)
        self.assertEqual(common.grad.abs().sum().item(), 0.)
        for parameter in layer.parameters():
            self.assertTrue(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()))
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.abs().sum().item(), 0.)

    def test_nonfinite_observed_value_rejected_but_invalid_nan_is_masked(self):
        values = list(attention_inputs())
        values[0] = values[0].clone()
        values[0][0, 0, 0] = float("nan")
        layer = EventQueryCommonAttention(3, 8, 2).double()
        with self.assertRaisesRegex(ValueError, "nonfinite observed"):
            layer(*values, .2)
        clean = list(attention_inputs())
        output, count = layer(*clean, .2)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(torch.isfinite(count).all()))

    def test_block_budget_fails_without_neighbor_truncation(self):
        layer = EventQueryCommonAttention(3, 8, 2, execution=ContextExecution(
            query_block_size=2, max_mask_elements=1, checkpoint_blocks=False)).double()
        with self.assertRaisesRegex(ValueError, "budget exceeded; no neighbors were truncated"):
            layer(*attention_inputs(), .2)

    def test_parameter_count_is_explicit_and_capacity_is_not_matched(self):
        layer = EventQueryCommonAttention(31, 64, 4)
        self.assertEqual(event_common_attention_parameter_count(31, 64, 4), 14_540)
        self.assertEqual(layer.trainable_parameter_count, 14_540)


class RuntimeIntegrationTest(unittest.TestCase):
    def sites(self, features, cm, chrom, valid):
        return Sites(torch.tensor(features, dtype=torch.float64),
                     torch.tensor(cm, dtype=torch.float64),
                     torch.tensor(chrom), torch.tensor(valid))

    def test_default_mean_preserves_legacy_state_dict_and_exact_forward(self):
        torch.manual_seed(1432575816)
        implicit = build_model(recipe(), 1, 1, max_common_sites=10,
                               max_events=10, max_queries=10).double()
        torch.manual_seed(1432575816)
        explicit = build_model(recipe(LOCAL_MEAN_V1), 1, 1, max_common_sites=10,
                               max_events=10, max_queries=10).double()
        self.assertEqual(list(implicit.state_dict()), list(explicit.state_dict()))
        self.assertFalse(any("event_common_attention" in key for key in implicit.state_dict()))
        explicit.load_state_dict(implicit.state_dict(), strict=True)
        common = self.sites([[[1.], [2.], [3.]]], [[0., .1, .2]], [[1, 1, 1]],
                            [[True, True, True]])
        events = self.sites([[[2.], [4.]]], [[.05, .15]], [[1, 1]], [[True, True]])
        queries = self.sites([[[0.], [0.]]], [[.05, .15]], [[1, 1]], [[True, True]])
        probabilities = torch.full((1, 2, 6), 1 / 6, dtype=torch.float64)
        torch.testing.assert_close(
            implicit.predict_lai(common, queries, probabilities, events),
            explicit.predict_lai(common, queries, probabilities, events), atol=0, rtol=0)

    def test_explicit_mode_propagates_and_predict_lai_has_finite_gradients(self):
        resolved = recipe(EVENT_QUERY_COMMON_KV_V1)
        config = ContextConfig.from_recipe(resolved)
        self.assertEqual(config.lai_common_context_mode, EVENT_QUERY_COMMON_KV_V1)
        model = build_model(resolved, 1, 1, max_common_sites=10,
            max_events=10, max_queries=10, context_execution={
                "query_block_size": 2, "max_mask_elements": 1_000,
                "checkpoint_blocks": True}).double()
        self.assertIsInstance(model.event_common_attention, EventQueryCommonAttention)
        self.assertTrue(any("event_common_attention" in key for key in model.state_dict()))
        common = self.sites([[[1.], [2.], [float("nan")]]], [[0., .1, float("nan")]],
                            [[1, 1, 1]], [[True, True, False]])
        common.features.requires_grad_()
        events = self.sites([[[2.], [float("nan")]]], [[.05, float("nan")]],
                            [[1, 1]], [[True, False]])
        events.features.requires_grad_()
        queries = self.sites([[[0.], [0.]]], [[.05, .15]], [[1, 1]], [[True, True]])
        probabilities = torch.full((1, 2, 6), 1 / 6, dtype=torch.float64, requires_grad=True)
        output = model.predict_lai(common, queries, probabilities, events)
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.square().sum().backward()
        for tensor in (common.features, events.features, probabilities):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
        self.assertEqual(common.features.grad[0, 2].abs().sum().item(), 0.)
        self.assertEqual(events.features.grad[0, 1].abs().sum().item(), 0.)

    def test_cross_attention_mode_is_rejected_for_structure(self):
        values = recipe(EVENT_QUERY_COMMON_KV_V1)
        values.update(family="structure_local_attention")
        with self.assertRaisesRegex(ValueError, "applies only to LAI"):
            ContextConfig.from_recipe(values)

    def test_cross_attention_requires_bounded_execution_configuration(self):
        with self.assertRaisesRegex(ValueError, "requires bounded context execution"):
            build_model(recipe(EVENT_QUERY_COMMON_KV_V1), 1, 1,
                        max_common_sites=10, max_events=10, max_queries=10)


if __name__ == "__main__":
    unittest.main()
