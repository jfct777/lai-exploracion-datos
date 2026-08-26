#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


MODELS = load_module("m34_models", "bin/m34_models.py")


class M34ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def fixture(self, length: int = 15):
        generator = torch.Generator().manual_seed(71)
        tokens = torch.randn(5, length, 7, generator=generator)
        mask = torch.ones(5, length, dtype=torch.bool)
        mask[0] = False
        mask[1, 9:] = False
        mask[2, 3] = False
        mask[2, 10:] = False
        mask[3, -2:] = False
        tokens = tokens * mask.unsqueeze(-1)
        baseline = torch.softmax(torch.randn(5, 2, 4, generator=generator), dim=-1)
        return tokens, mask, baseline

    def spec(self, family: str, **changes):
        values = dict(
            family=family,
            channels=7,
            ancestries=4,
            hidden_dim=16,
            depth=2,
            kernel_size=5,
            dilations=(1, 3),
            dropout=0.0,
            lstm_layers=1,
            transformer_heads=4,
            transformer_ff_dim=32,
            transformer_max_tokens=32,
            seed=17,
        )
        values.update(changes)
        return MODELS.ModelSpec(**values)

    def test_registry_shapes_simplex_and_zero_head_baseline(self):
        tokens, mask, baseline = self.fixture()
        for family in MODELS.FAMILIES:
            with self.subTest(family=family):
                model = MODELS.build_model(self.spec(family, zero_init_head=True)).eval()
                with torch.inference_mode():
                    probabilities, delta = model.forward_with_delta(tokens, mask, baseline)
                self.assertEqual(probabilities.shape, baseline.shape)
                self.assertEqual(delta.shape, baseline.shape)
                self.assertEqual(float(delta.abs().max()), 0.0)
                self.assertLessEqual(float((probabilities - baseline).abs().max()), 2e-7)
                MODELS.assert_simplex(probabilities)
                self.assertGreater(MODELS.parameter_count(model), 0)

    def test_padding_and_masked_value_invariance_with_active_heads(self):
        tokens, mask, baseline = self.fixture()
        padded_tokens = torch.cat((tokens, torch.randn(5, 7, 7)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(5, 7, dtype=torch.bool)), dim=1)
        altered = tokens.clone()
        altered[~mask] = 1000.0
        for family in MODELS.FAMILIES:
            with self.subTest(family=family):
                model = MODELS.build_model(self.spec(family, zero_init_head=False)).eval()
                with torch.inference_mode():
                    reference = model(tokens, mask, baseline)
                    padded = model(padded_tokens, padded_mask, baseline)
                    masked_values = model(altered, mask, baseline)
                self.assertTrue(torch.allclose(reference, padded, atol=2e-6, rtol=2e-6),
                                family)
                self.assertTrue(torch.allclose(reference, masked_values, atol=2e-6, rtol=2e-6),
                                family)

    def test_active_models_have_finite_gradients_and_predictions(self):
        tokens, mask, baseline = self.fixture()
        labels = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
        for family in MODELS.FAMILIES:
            with self.subTest(family=family):
                model = MODELS.build_model(self.spec(family, zero_init_head=False)).train()
                probabilities = model(tokens, mask, baseline)
                selected = probabilities.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                loss = -torch.log(selected.clamp_min(1e-8)).mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item(), family)
                self.assertTrue(torch.isfinite(probabilities).all().item(), family)
                gradients = [parameter.grad for parameter in model.parameters()
                             if parameter.requires_grad]
                self.assertTrue(all(gradient is not None for gradient in gradients), family)
                self.assertTrue(all(torch.isfinite(gradient).all().item()
                                    for gradient in gradients if gradient is not None), family)

    def test_residual_can_reopen_a_zero_baseline_class(self):
        tokens, mask, baseline = self.fixture()
        baseline[:, :, 0] = 0.0
        baseline[:, :, 1:] /= baseline[:, :, 1:].sum(dim=-1, keepdim=True)
        for family in MODELS.FAMILIES:
            with self.subTest(family=family):
                zero = MODELS.build_model(self.spec(family, zero_init_head=True)).eval()
                active = MODELS.build_model(self.spec(family, zero_init_head=False)).eval()
                with torch.inference_mode():
                    unchanged = zero(tokens, mask, baseline)
                    corrected = active(tokens, mask, baseline)
                self.assertLessEqual(float((unchanged - baseline).abs().max()), 3e-7)
                self.assertTrue(torch.all(corrected[:, :, 0] > 0).item())
                MODELS.assert_simplex(corrected)

    def test_tcn_has_a_wider_receptive_field_than_local_residual_cnn(self):
        residual = MODELS.build_model(self.spec("residual_cnn_1d", depth=3))
        tcn = MODELS.build_model(self.spec("tcn", depth=3, dilations=(1, 2, 4)))
        residual_dilations = [block.depthwise.dilation[0] for block in residual.blocks]
        tcn_dilations = [block.filter_conv.dilation[0] for block in tcn.blocks]
        self.assertEqual(residual_dilations, [1, 1, 1])
        self.assertEqual(tcn_dilations, [1, 2, 4])
        self.assertNotEqual(MODELS.parameter_count(residual), MODELS.parameter_count(tcn))
        self.assertNotEqual(set(residual.state_dict()), set(tcn.state_dict()))
        tokens, mask, baseline = self.fixture()
        residual = MODELS.build_model(
            self.spec("residual_cnn_1d", depth=3, zero_init_head=False)).eval()
        tcn = MODELS.build_model(
            self.spec("tcn", depth=3, dilations=(1, 2, 4),
                      zero_init_head=False)).eval()
        with torch.inference_mode():
            residual_output = residual(tokens, mask, baseline)
            tcn_output = tcn(tokens, mask, baseline)
        self.assertFalse(torch.allclose(residual_output, tcn_output, atol=1e-8, rtol=1e-8))

    def test_transformer_caps_attention_for_full_chr22_scale_context(self):
        tokens = torch.randn(3, 7503, 7)
        mask = torch.ones(3, 7503, dtype=torch.bool)
        mask[1, 7000:] = False
        mask[2] = False
        compressed, compressed_mask = MODELS._compress_masked_context(tokens, mask, 64)
        self.assertEqual(compressed.shape, (3, 64, 7))
        self.assertEqual(compressed_mask.sum(dim=1).tolist(), [64, 64, 0])
        model = MODELS.build_model(
            self.spec("transformer_small", transformer_max_tokens=64,
                      zero_init_head=False)).eval()
        baseline = torch.softmax(torch.randn(3, 2, 4), dim=-1)
        with torch.inference_mode():
            probabilities = model(tokens, mask, baseline)
        MODELS.assert_simplex(probabilities)

    def test_bilstm_and_unet_keep_odd_ragged_contexts_stable(self):
        tokens, mask, baseline = self.fixture(length=17)
        extension = torch.randn(5, 11, 7)
        extended_tokens = torch.cat((tokens, extension), dim=1)
        extended_mask = torch.cat((mask, torch.zeros(5, 11, dtype=torch.bool)), dim=1)
        for family in ("bilstm", "unet_1d"):
            with self.subTest(family=family):
                model = MODELS.build_model(self.spec(family, zero_init_head=False)).eval()
                with torch.inference_mode():
                    first = model(tokens, mask, baseline)
                    second = model(extended_tokens, extended_mask, baseline)
                self.assertTrue(torch.allclose(first, second, atol=2e-6, rtol=2e-6), family)

    def test_configurable_ancestry_and_haplotype_axes(self):
        tokens = torch.randn(2, 9, 5)
        mask = torch.ones(2, 9, dtype=torch.bool)
        baseline = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        for family in MODELS.FAMILIES:
            spec = MODELS.ModelSpec(
                family=family, channels=5, ancestries=5, hidden_dim=20,
                depth=1, dropout=0.0, transformer_heads=4,
                transformer_ff_dim=40, transformer_max_tokens=16,
            )
            with self.subTest(family=family), torch.inference_mode():
                result = MODELS.build_model(spec)(tokens, mask, baseline)
                self.assertEqual(result.shape, (2, 3, 5))
                MODELS.assert_simplex(result)

    def test_invalid_specs_and_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown model family"):
            MODELS.ModelSpec(family="unknown", channels=3, ancestries=3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.spec("transformer_small", hidden_dim=18, transformer_heads=4)
        # Transformer-only constraints must not leak into convolutional families.
        self.spec("residual_cnn_1d", hidden_dim=18, transformer_heads=4)
        tokens, mask, baseline = self.fixture()
        model = MODELS.build_model(self.spec("local_linear"))
        with self.assertRaisesRegex(ValueError, "ancestry count"):
            model(tokens, mask, baseline[..., :3])


if __name__ == "__main__":
    unittest.main()
