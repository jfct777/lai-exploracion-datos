"""Synthetic contract/numerical tests; no real data, fitting or cloud access."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m39_ordered_multichannel import MODES, MultichannelConfig, OrderedMultichannelAdapter
from test_m39_ordered_models import fixture, model_for


def inputs(dtype=torch.float64):
    original = fixture(dtype=dtype, length=7)
    common = {key: original[key] for key in ("channels", "delta_cm", "radius_cm",
                                             "site_mask", "candidate_mask")}
    rare = {key: original[key] for key in ("query_dosage", "query_observed",
                                           "reference_dosage", "reference_observed")}
    rare["ref_dosage_counts"] = torch.tensor([[[5, 1, 0], [3, 2, 1], [2, 1, 1]],
                                               [[2, 3, 1], [4, 1, 1], [1, 1, 2]]])
    rare["ref_eligible"] = torch.tensor([[8, 8, 5], [8, 8, 5]])
    base = torch.tensor([[0., .2, .3, .1, .2, .2], [0., .1, .2, .3, .2, .2]], dtype=dtype)
    return common, base, rare


def adapter(family="cnn", dtype=torch.float64):
    encoder = model_for(family, dtype=dtype, depth=1, dilations=(1,))
    torch.manual_seed(61)
    return OrderedMultichannelAdapter(encoder, MultichannelConfig(hidden_width=8, initial_gate=.1))


class Bomb(dict):
    def __contains__(self, key):
        raise AssertionError("disabled mapping was consulted")

    def __getitem__(self, key):
        raise AssertionError("disabled mapping was read")


class OrderedMultichannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_config_roundtrip_and_rejections(self):
        cfg = MultichannelConfig(8, .1)
        self.assertEqual(MultichannelConfig(**asdict(cfg)), cfg)
        for width, gate in ((0, .1), (True, .1), (8, 0), (8, 1), (8, float("nan")), (8, True)):
            with self.subTest(width=width, gate=gate), self.assertRaises(ValueError):
                MultichannelConfig(width, gate)

    def test_off_exact_identity_does_not_read_inputs_and_detaches_base(self):
        _, base, _ = inputs()
        base.requires_grad_(True)
        result = adapter()(Bomb(), base, Bomb(), mode="OFF")
        self.assertTrue(torch.equal(result["probabilities"], base))
        self.assertFalse(result["probabilities"].requires_grad)
        self.assertNotEqual(result["probabilities"].data_ptr(), base.data_ptr())
        self.assertTrue(torch.isneginf(result["log_probabilities"][:, 0]).all())
        self.assertTrue(torch.equal(result["gate"], torch.zeros_like(result["gate"])))

    def test_all_modes_simplex_log_parity_zero_rescue_and_input_immutability(self):
        for family in ("cnn", "attention"):
            common, base, rare = inputs()
            originals = copy.deepcopy((common, base, rare))
            model = adapter(family)
            for mode in MODES:
                result = model(common, base, rare, mode=mode)
                torch.testing.assert_close(result["probabilities"].sum(-1), torch.ones(2).double())
                torch.testing.assert_close(result["log_probabilities"].exp(), result["probabilities"])
                if mode != "OFF":
                    self.assertTrue((result["probabilities"][:, 0] > 0).all())
            for actual, expected in zip((common, base, rare), originals):
                if isinstance(actual, dict):
                    for key in actual:
                        self.assertTrue(torch.equal(actual[key], expected[key]), key)
                else:
                    self.assertTrue(torch.equal(actual, expected))

    def test_disabled_branches_are_not_read(self):
        common, base, rare = inputs()
        model = adapter()
        model(common, base, Bomb(), mode="NONE")
        summary = {key: value for key, value in rare.items() if not key.startswith("reference_")}
        detail = {key: value for key, value in rare.items() if not key.startswith("ref_")}
        for mode, selected in (("SUMMARY", summary), ("DETAIL", detail)):
            torch.testing.assert_close(model(common, base, selected, mode=mode)["probabilities"],
                                       model(common, base, rare, mode=mode)["probabilities"], atol=0, rtol=0)

    def test_missing_query_exact_none_fallback_but_observed_zero_is_supported(self):
        common, base, rare = inputs()
        model = adapter()
        zero = copy.deepcopy(rare)
        zero["query_dosage"].zero_()
        missing = copy.deepcopy(zero)
        missing["query_observed"].zero_()
        missing["query_dosage"].fill_(float("nan"))
        expected = model(common, base, mode="NONE")
        for mode in ("SUMMARY", "DETAIL", "BOTH"):
            result = model(common, base, missing, mode=mode)
            for key in ("probabilities", "log_probabilities"):
                torch.testing.assert_close(result[key], expected[key], atol=0, rtol=0)
            observed = model(common, base, zero, mode=mode)
            self.assertFalse(torch.equal(observed["probabilities"], result["probabilities"]))
            self.assertTrue((observed["summary_support"] | observed["detail_support"]).all())

    def test_empty_reference_support_exact_none_fallback(self):
        common, base, rare = inputs()
        rare["reference_observed"].zero_()
        rare["reference_dosage"].fill_(float("nan"))
        rare["ref_dosage_counts"].zero_()
        model = adapter()
        expected = model(common, base, mode="NONE")
        for mode in ("SUMMARY", "DETAIL", "BOTH"):
            result = model(common, base, rare, mode=mode)
            torch.testing.assert_close(result["probabilities"], expected["probabilities"], atol=0, rtol=0)

    def test_candidate_permutation_and_homolog_symmetry(self):
        for family in ("cnn", "attention"):
            common, base, rare = inputs()
            model = adapter(family)
            for mode in ("NONE", "SUMMARY", "DETAIL", "BOTH"):
                expected = model(common, base, rare, mode=mode)["probabilities"]
                for axis in (1, 3):
                    changed_common, changed_rare = copy.deepcopy((common, rare))
                    for key in ("channels", "candidate_mask"):
                        changed_common[key] = changed_common[key].flip(axis)
                    for key in ("reference_dosage", "reference_observed"):
                        changed_rare[key] = changed_rare[key].flip(axis)
                    result = model(changed_common, base, changed_rare, mode=mode)["probabilities"]
                    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)

    def test_padding_nan_is_ignored(self):
        common, base, rare = inputs()
        model = adapter()
        expected = model(common, base, rare, mode="BOTH")["probabilities"]
        valid = common["candidate_mask"][..., None] & common["site_mask"][:, None, None, None, :]
        common["channels"][~valid] = float("nan")
        rare["reference_dosage"][~common["candidate_mask"]] = float("nan")
        result = model(common, base, rare, mode="BOTH")["probabilities"]
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_branch_and_encoder_gradients_with_zero_base_target(self):
        for family in ("cnn", "attention"):
            common, base, rare = inputs()
            base.requires_grad_(True)
            model = adapter(family)
            result = model(common, base, rare, mode="BOTH")
            loss = -result["log_probabilities"][:, 0].mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNone(base.grad)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for module in (model.encoder.input_projection, model.summary_projection[0],
                           model.detail_candidate[0], model.expert_head, model.gate_head):
                self.assertGreater(float(module.weight.grad.abs().sum()), 0)

    def test_appending_masked_candidates_preserves_predictions(self):
        for family in ("cnn", "attention"):
            common, base, rare = inputs()
            model = adapter(family)
            padded_common, padded_rare = copy.deepcopy((common, rare))
            for key in ("channels", "candidate_mask"):
                extra = common[key][:, :, :, :1].clone()
                extra.fill_(False if extra.dtype == torch.bool else float("nan"))
                padded_common[key] = torch.cat((common[key], extra), dim=3)
            for key in ("reference_dosage", "reference_observed"):
                extra = rare[key][:, :, :, :1].clone()
                extra.fill_(False if extra.dtype == torch.bool else float("nan"))
                padded_rare[key] = torch.cat((rare[key], extra), dim=3)
            for mode in MODES:
                expected = model(common, base, rare, mode=mode)
                actual = model(padded_common, base, padded_rare, mode=mode)
                torch.testing.assert_close(actual["probabilities"], expected["probabilities"], atol=1e-12, rtol=1e-12)

    def test_no_common_support_and_mixed_rows_finite_backward(self):
        for missing_rows in ((0,), (0, 1)):
            common, base, rare = inputs()
            common["candidate_mask"][list(missing_rows)] = False
            base.zero_()
            base[:, 1] = 1
            model = adapter()
            result = model(common, base, rare, mode="BOTH")
            torch.testing.assert_close(result["probabilities"][list(missing_rows)],
                                       base[list(missing_rows)], atol=0, rtol=0)
            (-result["log_probabilities"][:, 1].mean()).backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_extreme_logits_gates_and_positive_subnormal(self):
        common, base, rare = inputs(torch.float32)
        for raw_gate in (-1000., 1000.):
            model = adapter(dtype=torch.float32)
            with torch.no_grad():
                model.gate_head.bias.fill_(raw_gate)
                model.expert_head.weight.zero_()
                model.expert_head.bias.fill_(-1000.)
                model.expert_head.bias[1] = 1000.
            result = model(common, base, rare, mode="BOTH")
            loss = -result["log_probabilities"][:, 0].mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(model.expert_head.bias.grad.abs().sum()), 0)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        base[:, 0] = 1e-40
        common["candidate_mask"].zero_()
        result = model(common, base, rare, mode="NONE")
        torch.testing.assert_close(result["probabilities"], base, atol=0, rtol=0)
        torch.testing.assert_close(result["log_probabilities"][:, 0], base[:, 0].log(), atol=0, rtol=0)

    def test_full_chunked_output_and_gradients_match(self):
        for family in ("cnn", "attention"):
            common, base, rare = inputs()
            full = adapter(family)
            chunks = copy.deepcopy(full)
            outputs = []
            for model, chunked in ((full, False), (chunks, True)):
                result = model(common, base, rare, mode="BOTH", chunked=chunked)
                outputs.append(result["probabilities"])
                (-result["log_probabilities"][:, 0].mean()).backward()
            torch.testing.assert_close(*outputs, atol=1e-11, rtol=1e-10)
            for (name, left), (_, right) in zip(full.named_parameters(), chunks.named_parameters()):
                if left.grad is not None:
                    torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-8, msg=name)

    def test_nearly_saturated_gate_preserves_representable_baseline_mass(self):
        common, base, rare = inputs(torch.float32)
        base[:, 0] = .1
        base[:, 1] -= .1
        model = adapter(dtype=torch.float32)
        with torch.no_grad():
            model.gate_head.weight.zero_()
            model.gate_head.bias.fill_(20.)
            model.expert_head.weight.zero_()
            model.expert_head.bias.fill_(-1000.)
            model.expert_head.bias[1] = 1000.
        result = model(common, base, rare, mode="BOTH")
        expected = (base.double()[:, 0] * torch.sigmoid(torch.tensor(-20., dtype=torch.float64))).float()
        self.assertTrue((result["probabilities"][:, 0] > 0).all())
        torch.testing.assert_close(result["probabilities"][:, 0], expected, atol=0, rtol=2e-6)
        torch.testing.assert_close(result["log_probabilities"].exp(), result["probabilities"], atol=0, rtol=0)

    def test_invalid_contracts_fail_closed(self):
        common, base, rare = inputs()
        model = adapter()
        mutations = (("query_dosage", lambda x: x.fill_(.5)),
                     ("reference_dosage", lambda x: x.fill_(3)),
                     ("query_observed", lambda x: x.to(torch.uint8)),
                     ("ref_dosage_counts", lambda x: x.float()),
                     ("ref_dosage_counts", lambda x: x.fill_(-1)),
                     ("ref_eligible", lambda x: x.zero_()))
        for key, change in mutations:
            changed = copy.deepcopy(rare)
            changed[key] = change(changed[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                model(common, base, changed, mode="BOTH")
        with self.assertRaises(ValueError):
            model(common, base * 2, rare, mode="BOTH")
        with self.assertRaises(ValueError):
            model(common, base, rare, mode="OTHER")
        with self.assertRaises(ValueError):
            model(common, base, None, mode="BOTH")


if __name__ == "__main__":
    unittest.main()
