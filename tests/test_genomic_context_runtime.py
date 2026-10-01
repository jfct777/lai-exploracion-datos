"""Operational tensor tests only: no genotypes, cohort fits or GPU required."""
import dataclasses
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import torch
from genomic_context_runtime import (ContextConfig, Sites, build_model, build_optimizer,
                                      genetic_neighbors, local_mean)

torch.set_num_threads(1)


def recipe(family="structure_local_attention", **overrides):
    result = dict(family=family, common_radius_cm=.05,
                  rare_radius_cm=None if family == "structure_deep_sets" else .05,
                  width=8, depth=2, dropout=0.,
                  heads=2 if family.endswith("local_attention") else None, learning_rate=.0003)
    if family == "lai_cnn":
        # Synthetic query spacing is .05 cM; not a proposed DNABR setting.
        result["query_max_gap_cm"] = .051
    result.update(overrides)
    return result


def sites(cm, values, chrom=None, valid=None):
    n = len(cm)
    return Sites(torch.tensor([values], dtype=torch.float32).reshape(1, n, -1),
                 torch.tensor([cm], dtype=torch.float32),
                 torch.tensor([chrom or [22] * n], dtype=torch.int64),
                 torch.tensor([valid or [True] * n], dtype=torch.bool))


def permute(x, order):
    return Sites(x.features[:, order], x.cm[:, order], x.chrom[:, order], x.valid[:, order])


class ContextRuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(901)
        self.common = sites([1., 1.03, 1.12, 1.21], [0., 1., 2., .5])
        self.events = sites([1.01, 1.11, 1.2], [1., 2., 1.])
        self.queries = sites([1., 1.05, 1.1, 1.15, 1.2], [0.] * 5)
        self.base = torch.full((1, 5, 6), 1 / 6)

    def test_conditional_fields_reject_silent_inactive_parameters(self):
        for kwargs in (recipe("structure_deep_sets", rare_radius_cm=.2),
                       recipe("lai_cnn", heads=2), recipe(width=10, heads=4),
                       recipe(common_radius_cm=0), recipe(dropout=1)):
            with self.assertRaises(ValueError):
                ContextConfig.from_recipe(kwargs)
        with self.assertRaises(ValueError):
            ContextConfig.from_recipe({**recipe(), "radius_cm": .2})

    def test_common_radius_changes_actual_neighbors_and_context(self):
        a, na = local_mean(self.common.features, self.events, self.common, .05)
        b, nb = local_mean(self.common.features, self.events, self.common, .3)
        self.assertEqual(na.tolist(), [[2, 1, 1]])
        self.assertEqual(nb.tolist(), [[4, 4, 4]])
        self.assertFalse(torch.allclose(a, b))

    def test_rare_radius_distinct_from_common_radius(self):
        small = build_model(recipe(), 1, 1).eval()
        large = build_model(recipe(rare_radius_cm=.3), 1, 1).eval()
        large.load_state_dict(small.state_dict())
        self.assertEqual(genetic_neighbors(self.events, self.events, .05).sum().item(), 3)
        self.assertEqual(genetic_neighbors(self.events, self.events, .3).sum().item(), 9)
        self.assertFalse(torch.allclose(small.encode_person(self.common, self.events),
                                        large.encode_person(self.common, self.events)))

    def test_same_cm_different_chromosomes_not_neighbors(self):
        source = sites([1., 1.], [2., 100.], chrom=[22, 21])
        query = sites([1.], [0.], chrom=[22])
        pooled, count = local_mean(source.features, query, source, .2)
        self.assertEqual(count.item(), 1)
        self.assertEqual(pooled.item(), 2.)

    def test_person_representation_permutation_invariant(self):
        for family in ("structure_deep_sets", "structure_local_attention"):
            model = build_model(recipe(family), 1, 1).eval()
            a = model.encode_person(self.common, self.events)
            b = model.encode_person(permute(self.common, [3, 0, 2, 1]), permute(self.events, [2, 0, 1]))
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_no_rare_arm_never_consumes_personal_event_geometry(self):
        model = build_model(recipe(), 1, 1).eval()
        a = model.encode_person(self.common, self.events, include_rare=False)
        b = model.encode_person(self.common, None, include_rare=False)
        torch.testing.assert_close(a, b)

    def test_empty_and_nan_padding_safe(self):
        model = build_model(recipe(), 1, 1).eval()
        empty = sites([float("nan")], [float("nan")], valid=[False])
        a = model.encode_person(self.common, empty)
        b = model.encode_person(self.common, include_rare=False)
        torch.testing.assert_close(a, b)
        with self.assertRaises(ValueError):
            model.encode_person(self.common, sites([1.], [float("nan")]))

    def test_depth_heads_dropout_and_learning_rate_reach_computation(self):
        model = build_model(recipe(depth=3, heads=4, dropout=.2, learning_rate=.0006), 1, 1)
        self.assertEqual(len(model.event_attention), 3)
        self.assertEqual(model.event_attention[0].attention.num_heads, 4)
        self.assertEqual(model.event_attention[0].attention.dropout, .2)
        self.assertEqual(build_optimizer(model).param_groups[0]["lr"], .0006)
        shallow = build_model(recipe(depth=1), 1, 1)
        self.assertGreater(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in shallow.parameters()))
        model.eval()
        torch.testing.assert_close(model.encode_person(self.common, self.events),
                                   model.encode_person(self.common, self.events))
        model.train()
        self.assertFalse(torch.allclose(model.encode_person(self.common, self.events),
                                        model.encode_person(self.common, self.events)))

    def test_deep_sets_depth_materializes(self):
        one = build_model(recipe("structure_deep_sets", depth=1), 1, 1)
        three = build_model(recipe("structure_deep_sets", depth=3), 1, 1)
        count = lambda m: sum(isinstance(x, torch.nn.Linear) for x in m.event_encoder)
        self.assertEqual(count(one), 1)
        self.assertEqual(count(three), 3)

    def test_lai_queries_are_not_rare_event_sites(self):
        for family in ("lai_cnn", "lai_local_attention"):
            model = build_model(recipe(family), 1, 1).eval()
            out = model.predict_lai(self.common, self.queries, self.base, self.events)
            self.assertEqual(tuple(out.shape), (1, 5, 6))
            torch.testing.assert_close(out.sum(-1), torch.ones((1, 5)))
            self.assertTrue(bool((out >= 0).all()))
            order = [4, 2, 0, 3, 1]
            shuffled = model.predict_lai(self.common, permute(self.queries, order), self.base[:, order], self.events)
            torch.testing.assert_close(out[:, order], shuffled, atol=2e-6, rtol=2e-6)

    def test_lai_rare_radius_changes_actual_query_inputs(self):
        for family in ("lai_cnn", "lai_local_attention"):
            a = build_model(recipe(family), 1, 1).eval()
            b = build_model(recipe(family, rare_radius_cm=.3), 1, 1).eval()
            b.load_state_dict(a.state_dict())
            self.assertFalse(torch.allclose(a.predict_lai(self.common, self.queries, self.base, self.events),
                                            b.predict_lai(self.common, self.queries, self.base, self.events)))

    def test_cnn_chromosomes_not_convolved_together(self):
        model = build_model(recipe("lai_cnn"), 1, 1).eval()
        query = sites([1., 1.05, 1.1, 1.15, 1.2], [0.] * 5, chrom=[22, 22, 21, 21, 21])
        x = torch.randn(1, 5, 8)
        out = model._convolve_queries(x, query)
        changed = x.clone()
        changed[:, 2:] += 100
        torch.testing.assert_close(out[:, :2], model._convolve_queries(changed, query)[:, :2])

    def test_cnn_requires_fixed_query_gap_and_rejects_conflict(self):
        missing = recipe("lai_cnn")
        del missing["query_max_gap_cm"]
        with self.assertRaisesRegex(ValueError, "explicit fixed"):
            build_model(missing, 1, 1)
        model = build_model(missing, 1, 1, query_max_gap_cm=.051)
        self.assertEqual(model.query_max_gap_cm, .051)
        self.assertEqual(build_model(recipe("lai_cnn"), 1, 1).query_max_gap_cm, .051)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            build_model(recipe("lai_cnn"), 1, 1, query_max_gap_cm=.02)
        for invalid in (0, -.1, True, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "query_max_gap_cm"):
                build_model(recipe("lai_cnn", query_max_gap_cm=invalid), 1, 1)

    def test_cnn_does_not_bridge_unrepresented_genomic_gaps(self):
        model = build_model(recipe("lai_cnn"), 1, 1).eval()
        query = sites([1., 1.05, 1.2, 1.25], [0.] * 4)
        x = torch.randn(1, 4, 8)
        changed = x.clone()
        changed[:, 2:] += 100
        out = model._convolve_queries(x, query)
        altered = model._convolve_queries(changed, query)
        torch.testing.assert_close(out[:, :2], altered[:, :2])
        self.assertFalse(torch.allclose(out[:, 2:], altered[:, 2:]))
        order = [3, 0, 2, 1]
        shuffled = model._convolve_queries(x[:, order], permute(query, order))
        torch.testing.assert_close(out[:, order], shuffled)

    def test_cnn_invalid_located_query_splits_even_below_max_gap(self):
        model = build_model(recipe("lai_cnn"), 1, 1).eval()
        query = sites([1., 1.025, 1.05, 1.1], [0.] * 4, valid=[True, False, True, True])
        x = torch.randn(1, 4, 8)
        x[:, 1] = float("nan")
        changed = x.clone()
        changed[:, 2:] += 100
        out = model._convolve_queries(x, query)
        altered = model._convolve_queries(changed, query)
        self.assertTrue(bool(torch.isfinite(out).all()))
        torch.testing.assert_close(out[:, :1], altered[:, :1])
        self.assertEqual(out[:, 1].abs().sum().item(), 0.)
        order = [2, 1, 3, 0]
        shuffled = model._convolve_queries(x[:, order], permute(query, order))
        torch.testing.assert_close(out[:, order], shuffled)

    def test_cnn_exact_grid_step_tolerates_float32_coordinate_rounding(self):
        model = build_model(recipe("lai_cnn", query_max_gap_cm=.01), 1, 1).eval()
        query = sites([100., 100.01, 100.02], [0.] * 3)
        x = torch.randn(1, 3, 8)
        expected = x[0].T[None]
        for conv in model.query_convs:
            expected = expected + torch.nn.functional.gelu(conv(expected))
        torch.testing.assert_close(model._convolve_queries(x, query), expected.transpose(1, 2))

    def test_duplicate_queries_rejected_in_both_lai_families(self):
        for family in ("lai_cnn", "lai_local_attention"):
            model = build_model(recipe(family), 1, 1).eval()
            duplicate = sites([1., 1., 1.1, 1.15, 1.2], [0.] * 5)
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "duplicate query"):
                model.predict_lai(self.common, duplicate, self.base, self.events)
            separate_chrom = sites([1., 1., 1.1, 1.15, 1.2], [0.] * 5, chrom=[21, 22, 22, 22, 22])
            output = model.predict_lai(self.common, separate_chrom, self.base, self.events)
            self.assertTrue(bool(torch.isfinite(output).all()))

    @staticmethod
    def mixed_padding(cm, values, valid):
        """A partly measured person and an entirely padded person in one batch."""
        n = len(cm)
        return Sites(torch.tensor([values, [float("nan")] * n], dtype=torch.float32)[..., None].requires_grad_(),
                     torch.tensor([cm, [float("nan")] * n], dtype=torch.float32),
                     torch.tensor([[22] * n, [0] * n], dtype=torch.int64),
                     torch.tensor([valid, [False] * n], dtype=torch.bool))

    def test_mixed_nan_padding_forward_backward_all_four_families(self):
        for family in ("structure_deep_sets", "structure_local_attention", "lai_cnn", "lai_local_attention"):
            with self.subTest(family=family):
                model = build_model(recipe(family, dropout=.2), 1, 1, max_queries=8).train()
                common = self.mixed_padding([1., 1.02, float("nan")], [0., 1., float("nan")], [True, True, False])
                events = self.mixed_padding([1.01, float("nan"), 1.03], [1., float("nan"), 2.], [True, False, True])
                if family.startswith("structure_"):
                    output = model.encode_person(common, events)
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    self.assertEqual(output[1].abs().sum().item(), 0.)
                    loss = output.square().sum()
                else:
                    queries = self.mixed_padding([1., 1.01, 1.02, float("nan")],
                                                 [0., 0., 0., float("nan")], [True, False, True, False])
                    base = torch.full((2, 4, 6), 1 / 6)
                    base[~queries.valid] = float("nan")
                    base.requires_grad_()
                    output = model.predict_lai(common, queries, base, events)
                    self.assertTrue(bool(torch.isfinite(output[queries.valid]).all()))
                    self.assertTrue(bool(torch.isnan(output[~queries.valid]).all()))
                    # Boolean selection, never 0 * NaN for missing predictions.
                    loss = output[queries.valid].square().sum()
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                for source in (common, events):
                    self.assertIsNotNone(source.features.grad)
                    self.assertTrue(bool(torch.isfinite(source.features.grad).all()))
                    self.assertEqual(source.features.grad[~source.valid].abs().sum().item(), 0.)
                if family.startswith("lai_"):
                    self.assertTrue(bool(torch.isfinite(base.grad).all()))
                    self.assertEqual(base.grad[~queries.valid].abs().sum().item(), 0.)

    def test_entirely_empty_event_batch_backward_is_finite(self):
        for family in ("structure_deep_sets", "structure_local_attention", "lai_cnn", "lai_local_attention"):
            with self.subTest(family=family):
                model = build_model(recipe(family), 1, 1, max_queries=8).eval()
                empty = sites([float("nan")], [float("nan")], valid=[False])
                empty.features.requires_grad_()
                if family.startswith("structure_"):
                    output = model.encode_person(self.common, empty)
                    baseline = model.encode_person(self.common, include_rare=False)
                else:
                    output = model.predict_lai(self.common, self.queries, self.base, empty)
                    baseline = model.predict_lai(self.common, self.queries, self.base, include_rare=False)
                torch.testing.assert_close(output, baseline)
                output.square().sum().backward()
                self.assertTrue(bool(torch.isfinite(empty.features.grad).all()))
                self.assertEqual(empty.features.grad.abs().sum().item(), 0.)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_all_queries_invalid_remain_nan_without_nan_gradients(self):
        for family in ("lai_cnn", "lai_local_attention"):
            with self.subTest(family=family):
                model = build_model(recipe(family), 1, 1, max_queries=8).train()
                queries = sites([float("nan")], [float("nan")], valid=[False])
                base = torch.full((1, 1, 6), float("nan"), requires_grad=True)
                output = model.predict_lai(self.common, queries, base, self.events)
                self.assertTrue(bool(torch.isnan(output).all()))
                # Empty valid selection has zero loss for this arithmetic test;
                # a scientific evaluator must instead record zero evaluability.
                output[queries.valid].sum().backward()
                self.assertTrue(bool(torch.isfinite(base.grad).all()))
                self.assertEqual(base.grad.abs().sum().item(), 0.)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_limits_fail_without_silent_truncation(self):
        model = build_model(recipe(), 1, 1, max_events=2)
        with self.assertRaisesRegex(ValueError, "never truncate"):
            model.encode_person(self.common, self.events)

    def test_backward_reaches_actual_attention_parameters(self):
        model = build_model(recipe(), 1, 1)
        model.encode_person(self.common, self.events).square().sum().backward()
        grad = model.event_attention[0].attention.in_proj_weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
