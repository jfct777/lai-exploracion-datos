"""Synthetic controls for ordered encoders; no ancestry-truth files are used."""

import copy
from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from m39_ordered_models import (LocalAttentionBlock, OrderedLAIModel,
                                OrderedModelConfig, STATE_NAMES)


def fixture(*, count=2, candidates=3, length=17, dtype=torch.float64):
    generator = torch.Generator().manual_seed(714)
    shape = (count, 2, 3, candidates, length)
    query_called = torch.rand(shape, generator=generator) > 0.15
    ref_called = torch.rand(shape, generator=generator) > 0.2
    same = torch.rand(shape, generator=generator) > 0.5
    joint = query_called & ref_called
    channels = torch.stack((same & joint, ~same & joint, ~query_called,
                            ~ref_called, joint), dim=-1).to(dtype)
    site_mask = torch.ones((count, length), dtype=torch.bool)
    if count > 1 and length > 3:
        site_mask[-1, -3:] = False
    candidate_mask = torch.ones((count, 2, 3, candidates), dtype=torch.bool)
    if candidates > 1:
        candidate_mask[0, 1, 2, -1] = False
    return {
        "channels": channels,
        "delta_cm": torch.linspace(-0.5, 0.5, length, dtype=dtype).expand(count, -1).clone(),
        "radius_cm": torch.full((count,), 0.5, dtype=dtype),
        "site_mask": site_mask,
        "candidate_mask": candidate_mask,
        "reference_dosage": torch.randint(0, 3, (count, 2, 3, candidates), generator=generator).to(dtype),
        "reference_observed": torch.ones((count, 2, 3, candidates), dtype=torch.bool),
        "query_dosage": (torch.arange(count) % 3).to(dtype),
        "query_observed": torch.ones(count, dtype=torch.bool),
        "pooled_af": torch.rand((count, 3), generator=generator, dtype=dtype),
        "pooled_observed": torch.ones((count, 3), dtype=torch.bool),
    }


def model_for(family, *, dtype=torch.float64, **kwargs):
    torch.manual_seed(802)
    defaults = dict(family=family, width=8, depth=2, kernels=(3, 5),
                    dilations=(1, 2), heads=2, attention_radius_tokens=2,
                    dropout=0, core_sites=4, checkpoint_chunks=True)
    defaults.update(kwargs)
    return OrderedLAIModel(OrderedModelConfig(**defaults)).to(dtype)


def cloned_batch(batch):
    return {key: value.detach().clone() for key, value in batch.items()}


class OrderedModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_tensors_close(self, left, right, *, atol=3e-10, rtol=3e-9):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)

    def test_configuration_roundtrip_and_receptive_fields(self):
        config = OrderedModelConfig(depth=3, kernels=(3, 7), dilations=(1, 2, 4))
        self.assertEqual(config.receptive_radius_tokens, 21)
        mapping = asdict(config)
        mapping["kernels"] = list(mapping["kernels"])
        mapping["dilations"] = list(mapping["dilations"])
        self.assertEqual(OrderedModelConfig(**mapping), config)
        self.assertEqual(replace(config, family="attention", attention_radius_tokens=5).receptive_radius_tokens, 15)
        self.assertEqual(STATE_NAMES, ("AA", "AE", "AN", "EE", "EN", "NN"))

    def test_invalid_configurations_rejected(self):
        cases = ({"family": "global"}, {"width": 0}, {"depth": 0},
                 {"kernels": ()}, {"kernels": (2, 3)}, {"kernels": (3, 3)},
                 {"dilations": (1,)}, {"dilations": (0, 2)},
                 {"family": "attention", "width": 7, "heads": 2},
                 {"attention_radius_tokens": -1}, {"dropout": 1},
                 {"dropout": float("nan")}, {"core_sites": 0},
                 {"core_sites": True}, {"checkpoint_chunks": "yes"})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OrderedModelConfig(**kwargs)

    def test_two_families_have_identical_head_architectures(self):
        cnn, attention = (model_for(family) for family in ("cnn", "attention"))
        self.assertEqual([(key, tuple(value.shape)) for key, value in cnn.head.state_dict().items()],
                         [(key, tuple(value.shape)) for key, value in attention.head.state_dict().items()])

    def test_full_chunked_logits_losses_all_parameter_and_input_gradients(self):
        for family in ("cnn", "attention"):
            for core in (1, 4, 7, 50):
                with self.subTest(family=family, core=core):
                    full = model_for(family, core_sites=core)
                    chunks = copy.deepcopy(full)
                    batch_full, batch_chunks = fixture(length=19), fixture(length=19)
                    for batch in (batch_full, batch_chunks):
                        batch["channels"].requires_grad_(True)
                        batch["delta_cm"].requires_grad_(True)
                    logits_full = full(batch_full, arm="real", chunked=False)
                    logits_chunks = chunks(batch_chunks, arm="real", chunked=True)
                    self.assert_tensors_close(logits_full, logits_chunks)
                    labels = torch.tensor([0, 4])
                    loss_full = F.cross_entropy(logits_full, labels)
                    loss_chunks = F.cross_entropy(logits_chunks, labels)
                    self.assert_tensors_close(loss_full, loss_chunks)
                    loss_full.backward()
                    loss_chunks.backward()
                    for (name, left), (other, right) in zip(full.named_parameters(), chunks.named_parameters()):
                        self.assertEqual(name, other)
                        self.assertIsNotNone(left.grad, name)
                        self.assertIsNotNone(right.grad, name)
                        self.assert_tensors_close(left.grad, right.grad, atol=5e-10, rtol=5e-8)
                    for key in ("channels", "delta_cm"):
                        self.assert_tensors_close(batch_full[key].grad, batch_chunks[key].grad,
                                                  atol=5e-10, rtol=5e-8)
                    optimizer_full = torch.optim.AdamW(full.parameters(), lr=0.001, weight_decay=0)
                    optimizer_chunks = torch.optim.AdamW(chunks.parameters(), lr=0.001, weight_decay=0)
                    optimizer_full.step()
                    optimizer_chunks.step()
                    for left, right in zip(full.parameters(), chunks.parameters()):
                        self.assert_tensors_close(left, right, atol=3e-9, rtol=3e-8)

    def test_float32_checkpoint_gradients_without_trainable_inputs(self):
        for family in ("cnn", "attention"):
            with self.subTest(family=family):
                full = model_for(family, dtype=torch.float32)
                chunked = copy.deepcopy(full)
                batch = fixture(dtype=torch.float32)
                full(batch, "common", chunked=False).square().sum().backward()
                chunked(batch, "common", chunked=True).square().sum().backward()
                for left, right in zip(full.parameters(), chunked.parameters()):
                    self.assertIsNotNone(right.grad)
                    self.assert_tensors_close(left.grad, right.grad, atol=2e-6, rtol=2e-4)

    def test_checkpointing_does_not_change_logits_or_gradients(self):
        for family in ("cnn", "attention"):
            first = model_for(family)
            second = model_for(family, checkpoint_chunks=False)
            batch = fixture()
            left, right = first(batch, "real"), second(batch, "real")
            self.assert_tensors_close(left, right)
            left.square().sum().backward()
            right.square().sum().backward()
            for a, b in zip(first.parameters(), second.parameters()):
                self.assert_tensors_close(a.grad, b.grad)

    def test_candidate_permutation_equivariance_and_head_invariance(self):
        batch = fixture(candidates=4)
        permuted = cloned_batch(batch)
        permutation = torch.tensor([2, 0, 3, 1])
        for key in ("channels", "candidate_mask", "reference_dosage", "reference_observed"):
            permuted[key] = batch[key].index_select(3, permutation)
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            encoded = model.encode_candidates(batch)
            changed = model.encode_candidates(permuted)
            self.assert_tensors_close(encoded.index_select(3, permutation), changed)
            for arm in ("common", "real", "pooled"):
                self.assert_tensors_close(model(batch, arm), model(permuted, arm))

    def test_query_homolog_swap_invariance(self):
        batch = fixture()
        swapped = cloned_batch(batch)
        for key in ("channels", "candidate_mask", "reference_dosage", "reference_observed"):
            swapped[key] = batch[key].flip(1)
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            self.assert_tensors_close(model.encode_candidates(batch).flip(1), model.encode_candidates(swapped))
            for arm in ("common", "real", "pooled"):
                self.assert_tensors_close(model(batch, arm), model(swapped, arm))

    def test_common_ignores_all_rare_values_and_masks(self):
        batch = fixture()
        changed = cloned_batch(batch)
        for key in ("reference_dosage", "query_dosage", "pooled_af"):
            changed[key].fill_(float("nan"))
        for key in ("reference_observed", "query_observed", "pooled_observed"):
            changed[key] = ~changed[key]
        minimal = {key: value for key, value in batch.items()
                   if key in ("channels", "delta_cm", "radius_cm", "site_mask", "candidate_mask")}
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            expected = model(batch, "common")
            self.assertTrue(torch.equal(expected, model(changed, "common")))
            self.assertTrue(torch.equal(expected, model(minimal, "common")))
            self.assertTrue(torch.equal(model.encode_candidates(batch), model.encode_candidates(changed)))

    def test_common_parameter_gradients_are_rare_independent(self):
        batch = fixture()
        changed = cloned_batch(batch)
        changed["reference_dosage"] = 2 - changed["reference_dosage"]
        changed["query_dosage"] = 2 - changed["query_dosage"]
        for family in ("cnn", "attention"):
            first, second = model_for(family), model_for(family)
            first(batch, "common").square().sum().backward()
            second(changed, "common").square().sum().backward()
            for a, b in zip(first.parameters(), second.parameters()):
                self.assertTrue(torch.equal(a.grad, b.grad))

    def test_missing_rare_is_not_observed_reference_and_ignored_payload_is_safe(self):
        batch = fixture()
        batch["reference_dosage"].zero_()
        batch["query_dosage"].zero_()
        missing = cloned_batch(batch)
        missing["reference_observed"].zero_()
        missing["query_observed"].zero_()
        ignored = cloned_batch(missing)
        ignored["reference_dosage"].fill_(float("nan"))
        ignored["query_dosage"].fill_(float("nan"))
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            self.assertGreater(float((model(batch, "real") - model(missing, "real")).abs().max()), 1e-8)
            self.assertTrue(torch.equal(model(missing, "real"), model(ignored, "real")))

    def test_rare_and_pooled_positive_sensitivity(self):
        batch = fixture()
        real_change = cloned_batch(batch)
        real_change["reference_dosage"] = 2 - real_change["reference_dosage"]
        real_change["query_dosage"] = 2 - real_change["query_dosage"]
        pooled_change = cloned_batch(batch)
        pooled_change["pooled_af"] = 1 - pooled_change["pooled_af"]
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            self.assertGreater(float((model(batch, "real") - model(real_change, "real")).abs().max()), 1e-8)
            self.assertGreater(float((model(batch, "pooled") - model(pooled_change, "pooled")).abs().max()), 1e-8)
            self.assertTrue(torch.equal(model(batch, "real"), model(pooled_change, "real")))
            ignored = cloned_batch(batch)
            ignored["reference_dosage"].fill_(float("nan"))
            self.assertTrue(torch.equal(model(batch, "pooled"), model(ignored, "pooled")))

    def test_equal_counts_different_order_positive_control(self):
        first = fixture(count=1, candidates=1, length=12)
        first["delta_cm"].zero_()  # Genetic-map ties leave sequence order intact.
        first["channels"].zero_()
        second = cloned_batch(first)
        patterns = (torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
                    torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
        for batch, pattern in zip((first, second), patterns):
            batch["channels"][..., 0] = pattern
            batch["channels"][..., 1] = 1 - pattern
            batch["channels"][..., 4] = 1
        self.assertTrue(torch.equal(first["channels"].sum(dim=-2), second["channels"].sum(dim=-2)))
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            left, right = model.encode_candidates(first), model.encode_candidates(second)
            self.assertGreater(float((left - right).abs().max()), 1e-6)
            self.assertGreater(float((model(first) - model(second)).abs().max()), 1e-8)

    def test_missing_common_is_not_observed_reference(self):
        observed = fixture(count=1, candidates=1, length=9)
        observed["channels"].zero_()
        observed["channels"][..., 0] = 1
        observed["channels"][..., 4] = 1
        missing = cloned_batch(observed)
        missing["channels"].zero_()
        missing["channels"][..., 2] = 1
        missing["channels"][..., 3] = 1
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            self.assertGreater(float((model(observed) - model(missing)).abs().max()), 1e-8)

    def test_padding_and_absent_candidates_are_ignored(self):
        original = fixture(count=1, candidates=2, length=9)
        padded = cloned_batch(original)
        padded["channels"] = F.pad(padded["channels"], (0, 0, 0, 8), value=float("nan"))
        padded["delta_cm"] = F.pad(padded["delta_cm"], (0, 8), value=float("nan"))
        padded["site_mask"] = F.pad(padded["site_mask"], (0, 8), value=False)
        # Appended invalid lanes cannot dilute the masked K mean.
        for key in ("channels", "candidate_mask", "reference_dosage", "reference_observed"):
            value = padded[key]
            extra_shape = list(value.shape)
            extra_shape[3] = 2
            fill = False if value.dtype == torch.bool else float("nan")
            extra = torch.full(extra_shape, fill, dtype=value.dtype)
            padded[key] = torch.cat((value, extra), dim=3)
        for family in ("cnn", "attention"):
            model = model_for(family).eval()
            for arm in ("common", "real", "pooled"):
                self.assert_tensors_close(model(original, arm), model(padded, arm))

    def test_empty_short_and_all_masked_windows_are_finite(self):
        for family in ("cnn", "attention"):
            model = model_for(family)
            for length in (0, 1, 2):
                batch = fixture(count=1, candidates=1, length=length)
                for empty in (False, True):
                    with self.subTest(family=family, length=length, empty=empty):
                        if empty:
                            batch["site_mask"].zero_()
                            batch["candidate_mask"].zero_()
                            batch["channels"].fill_(float("nan"))
                        for arm in ("common", "real", "pooled"):
                            logits = model(batch, arm)
                            self.assertEqual(tuple(logits.shape), (1, 6))
                            self.assertTrue(bool(torch.isfinite(logits).all()))
                            self.assert_tensors_close(logits, model(batch, arm, chunked=False))
                            self.assert_tensors_close(logits.softmax(-1).sum(-1), torch.ones(1, dtype=logits.dtype))

    def test_local_attention_score_storage_is_a_band(self):
        model = model_for("attention", core_sites=7, attention_radius_tokens=2)
        shapes = []
        hooks = []
        for block in model.blocks:
            self.assertIsInstance(block, LocalAttentionBlock)
            hooks.append(block.softmax.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape))))
        try:
            model(fixture(length=101), "real").square().sum().backward()
        finally:
            for hook in hooks:
                hook.remove()
        self.assertTrue(shapes)
        for shape in shapes:
            self.assertEqual(shape[-1], 5)
            self.assertLessEqual(shape[-2], 7 + 2 * model.receptive_radius_tokens)
            self.assertEqual(len(shape), 4)

    def test_encoder_calls_never_exceed_core_plus_full_halo(self):
        for family in ("cnn", "attention"):
            model = model_for(family, core_sites=3)
            lengths = []
            hook = model.input_projection.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
            try:
                model(fixture(length=57), "real").square().sum().backward()
            finally:
                hook.remove()
            self.assertGreater(len(lengths), 1)
            self.assertLessEqual(max(lengths), 3 + 2 * model.receptive_radius_tokens)

    def test_radius_zero_attention_and_dilation_sensitivities(self):
        for kwargs in ({"family": "attention", "attention_radius_tokens": 0},
                       {"family": "cnn", "kernels": (1, 3, 7), "dilations": (2, 3)}):
            family = kwargs.pop("family")
            model = model_for(family, **kwargs)
            batch = fixture(length=23)
            self.assert_tensors_close(model(batch, "real"), model(batch, "real", chunked=False))

    def test_chunked_training_rejects_nonzero_dropout(self):
        for family in ("cnn", "attention"):
            model = model_for(family, dropout=0.1)
            batch = fixture()
            with self.assertRaisesRegex(ValueError, "dropout=0"):
                model(batch)
            self.assertTrue(bool(torch.isfinite(model(batch, chunked=False)).all()))
            model.eval()
            self.assert_tensors_close(model(batch), model(batch, chunked=False))

    def test_adamw_step_with_cyclic_synthetic_labels_updates_weights(self):
        for family in ("cnn", "attention"):
            for arm in ("common", "real"):
                with self.subTest(family=family, arm=arm):
                    model = model_for(family, dtype=torch.float32)
                    before = {key: value.detach().clone() for key, value in model.named_parameters()}
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
                    batch = fixture(count=3, dtype=torch.float32)
                    labels = torch.arange(3) % 6
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch, arm)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    self.assertTrue(bool(torch.isfinite(loss)))
                    self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
                    self.assertGreater(sum(float(p.grad.square().sum()) for p in model.parameters()), 0)
                    optimizer.step()
                    self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.named_parameters()))
                    self.assertTrue(any(not torch.equal(before[key], value)
                                        for key, value in model.named_parameters() if key.startswith("blocks.")))
                    self.assertTrue(any(not torch.equal(before[key], value)
                                        for key, value in model.named_parameters() if key.startswith("head.")))
                    self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
                    self.assertGreater(len(optimizer.state), 0)

    def test_invalid_inputs_fail_and_batch_is_not_mutated(self):
        model = model_for("cnn")
        batch = fixture()
        original = cloned_batch(batch)
        model(batch, "real")
        for key in batch:
            self.assertTrue(torch.equal(batch[key], original[key]))
        invalid = []
        wrong = cloned_batch(batch)
        wrong["site_mask"] = wrong["site_mask"].float()
        invalid.append(wrong)
        wrong = cloned_batch(batch)
        wrong["delta_cm"][0, 0] = 0.25
        invalid.append(wrong)
        wrong = cloned_batch(batch)
        wrong["channels"][0, 0, 0, 0, 0, 0] = float("nan")
        invalid.append(wrong)
        wrong = cloned_batch(batch)
        wrong["radius_cm"][0] = 0
        invalid.append(wrong)
        wrong = cloned_batch(batch)
        wrong["reference_dosage"][0, 0, 0, 0] = 3
        invalid.append(wrong)
        for wrong in invalid:
            with self.assertRaises(ValueError):
                model(wrong, "real")
        with self.assertRaises(ValueError):
            model(batch, "unknown")
        wrong = cloned_batch(batch)
        wrong["pooled_af"][0, 0] = 1.1
        with self.assertRaises(ValueError):
            model(wrong, "pooled")


if __name__ == "__main__":
    unittest.main()
