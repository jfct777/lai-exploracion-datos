"""Synthetic paired refits and regression tests for the shared training backend."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
sys.path.insert(0, str(ROOT / "tests"))

import m39_ordered_training as T
import m39_ordered_multichannel_training as M
from m39_ordered_training_data import bind_development, reference_link_permutation
from m39_profile_device import DeviceRuntime
from m39_profile_device import LOGIT_ATOL, LOGIT_RTOL, GRADIENT_ATOL, GRADIENT_RTOL
import test_m39_ordered_training as legacy_fixture


class MultichannelTrainingTests(unittest.TestCase):
    # Reuse the authenticated two-role fixture without inheriting its tests.
    setUpClass = classmethod(lambda cls: torch.set_num_threads(1))
    setUp = legacy_fixture.OrderedTrainingTests.setUp
    tearDown = legacy_fixture.OrderedTrainingTests.tearDown
    store = legacy_fixture.OrderedTrainingTests.store
    setup_training = legacy_fixture.OrderedTrainingTests.setup_training

    def setup_multichannel(self):
        values = self.setup_training()
        cfg, path = values[-1], values[-2]
        cfg.update(schema_version=M.SCHEMA, arm="both",
                   multichannel={"hidden_width": 4, "initial_gate": .01})
        path.write_text(json.dumps(cfg))
        return values

    def test_configuration_is_explicit_and_rejects_OFF_as_a_fit(self):
        *_, path, cfg = self.setup_multichannel()
        self.assertEqual(M.load_config(path), json.loads(json.dumps(cfg)))
        bad_values = [{"arm": "off"}, {"arm": "BOTH"}, {"schema_version": T.SCHEMA},
                      {"multichannel": {"hidden_width": 4}},
                      {"multichannel": {"hidden_width": True, "initial_gate": .01}},
                      {"multichannel": {"hidden_width": 4, "initial_gate": 0}},
                      {"multichannel": {"hidden_width": 4, "initial_gate": .01, "extra": 1}}]
        for changed in bad_values:
            invalid = dict(cfg, **changed)
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                M.load_config(path)
        path.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            T.load_config(path)

    def _fit_all_arms(self, family):
        train, select, train_path, select_path, binding, path, cfg = self.setup_multichannel()
        cfg["model"]["family"] = family
        receipts, final_states = [], []
        for arm in M.ARMS:
            cfg["arm"] = arm
            path.write_text(json.dumps(cfg))
            output = self.root / arm
            receipt = M.run_case(train_path, select_path, binding, path, output)
            receipts.append(receipt)
            self.assertEqual(receipt["schema_version"], M.SCHEMA)
            self.assertEqual(receipt["decision"], "COMPLETED_EXPLORATORY_DEVELOPMENT_CASE")
            self.assertEqual(receipt["training_observations"], 2)
            self.assertFalse(receipt["scope"]["SCORE_opened"])
            self.assertEqual(receipt["sources"]["representation"]["mode"], M.model_mode(arm))
            self.assertEqual(set(receipt["sources"]["code_sha256"]), set(M.SOURCE_FILES))
            self.assertEqual(receipt["selected_step"], min(receipt["curve"],
                key=lambda row: (row["SELECT"]["brier"], row["SELECT"]["log_loss"], row["step"]))["step"])
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
            backend = M.MultichannelTrainingBackend()
            data = bind_development(binding, cfg["development_sha256"], train, select)
            backend.prepare(train, select, data, cfg)
            model = backend.make_model(cfg)
            model.load_state_dict(checkpoint["state_dict"])
            final_states.append(checkpoint["state_dict"]["expert_head.weight"])
            permutation = reference_link_permutation(train, cfg["sham_seed"]) if arm == "sham" else None
            replay = T.predict(model, select, np.asarray(receipt["anchor_indices"]), cfg,
                               DeviceRuntime(), permutation, started=T.time.monotonic(), backend=backend)
            with np.load(output / "select.predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(replay, saved["probabilities"])
            best_step = receipt["selected_step"]
            partial = torch.load(output / f"checkpoint-step-{best_step:07d}.pt", weights_only=True)
            for name, value in checkpoint["state_dict"].items():
                torch.testing.assert_close(value, partial["state_dict"][name], atol=0, rtol=0)
        self.assertEqual(len({r["initial_state_sha256"] for r in receipts}), 1)
        self.assertEqual(len({r["training_pair_stream_sha256"] for r in receipts}), 1)
        self.assertTrue(any(not torch.equal(final_states[0], other) for other in final_states[1:]))
        self.assertIsNotNone(receipts[-1]["sham_reference_map_sha256"])
        self.assertEqual(receipts[-1]["sources"]["representation"]["mode"], "BOTH")

    def test_CNN_all_five_arms_are_separately_fitted_paired_and_replay_exactly(self):
        self._fit_all_arms("cnn")

    def test_attention_all_five_arms_are_separately_fitted_paired_and_replay_exactly(self):
        self._fit_all_arms("attention")

    def test_initial_evaluation_is_diagnostic_never_selectable(self):
        _, _, train, select, binding, path, cfg = self.setup_multichannel()
        cfg["evaluate_initial"] = True
        path.write_text(json.dumps(cfg))
        original = T.metrics
        calls = 0

        def initial_is_best(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 1:
                result["brier"] = -1.0  # Adversarial sentinel: not a scientific metric.
            return result

        output = self.root / "initial"
        with patch.object(T, "metrics", side_effect=initial_is_best):
            receipt = M.run_case(train, select, binding, path, output)
        self.assertEqual(receipt["curve"][0]["SELECT"]["brier"], -1)
        self.assertFalse(receipt["curve"][0]["checkpoint_eligible"])
        self.assertGreater(receipt["selected_step"], 0)
        self.assertFalse((output / "checkpoint-step-0000000.pt").exists())

    def test_budget_failure_keeps_last_best_but_never_completion_receipt(self):
        _, _, train, select, binding, path, _ = self.setup_multichannel()
        calls = 0

        def stop_after_first_evaluation(*args):
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise ValueError("training time ceiling exceeded")

        output = self.root / "interrupted"
        with patch.object(T, "_limits", side_effect=stop_after_first_evaluation):
            with self.assertRaisesRegex(ValueError, "time ceiling"):
                M.run_case(train, select, binding, path, output)
        self.assertTrue((output / "checkpoint-step-0000001.pt").exists())
        self.assertTrue((output / "select-step-0000001.npz").exists())
        self.assertFalse((output / "training.receipt.json").exists())

    def test_legacy_backend_preserves_logits_loss_and_optimizer_update_exactly(self):
        store = self.store()
        *_, cfg = self.setup_training()
        backend = T.OrderedTrainingBackend()
        torch.manual_seed(31)
        model = backend.make_model(cfg)
        reference = copy.deepcopy(model)
        batch = T._batch(store, [(0, 0), (1, 1)], cfg, DeviceRuntime(), None)
        truth = torch.tensor([0, 1])
        actual_loss = backend.loss(model, batch, truth, cfg)
        expected_loss = F.cross_entropy(reference(batch, arm="real", chunked=True), truth)
        torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
        for candidate, loss in ((model, actual_loss), (reference, expected_loss)):
            optimizer = torch.optim.AdamW(candidate.parameters(), lr=cfg["learning_rate"],
                                           weight_decay=cfg["weight_decay"], foreach=False, fused=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), cfg["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
        model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(backend.probabilities(model, batch, cfg),
                                       model(batch, arm="real", chunked=True).softmax(-1), atol=0, rtol=0)

    def test_float32_zero_base_target_and_masked_rares_have_finite_native_loss_gradients(self):
        from test_m39_ordered_multichannel import inputs, adapter

        common, baseline, rare = inputs(torch.float32)
        baseline.requires_grad_(True)
        rare["query_observed"][0] = False
        rare["query_dosage"][0] = float("nan")
        rare["reference_dosage"][~common["candidate_mask"]] = float("nan")
        cfg = {"arm": "both"}
        backend = M.MultichannelTrainingBackend()
        for family in ("cnn", "attention"):
            model = adapter(family, torch.float32)
            batch = {**common, **rare, "baseline": baseline}
            output = backend.forward(model, batch, cfg)
            loss = backend.loss(model, batch, torch.tensor([0, 0]), cfg)
            torch.testing.assert_close(loss, -output["log_probabilities"][:, 0].mean(), atol=0, rtol=0)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(bool((output["probabilities"][:, 0] > 0).all()))
            loss.backward()
            self.assertIsNone(baseline.grad)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(model.gate_head.bias.grad.abs().sum()), 0)
            self.assertGreater(float(model.expert_head.weight.grad.abs().sum()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "optional L4 CUDA parity; no GPU available")
    def test_optional_L4_outputs_losses_active_gradients_and_detached_baseline_match_CPU(self):
        # This test is opt-in by device availability, not a CPU fallback or a
        # claim that a skipped check established GPU parity. It uses the exact
        # same pinned IEEE runtime and tolerances as the ordered GPU protocol.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        memory = min(20 * 1024**3, torch.cuda.get_device_properties(0).total_memory)
        runtime = DeviceRuntime("cuda:0", memory)
        runtime.configure()
        train, select, _, _, binding, _, cfg = self.setup_multichannel()
        data = bind_development(binding, cfg["development_sha256"], train, select)
        baseline = data["train"]["baseline"]
        baseline[:] = np.asarray([0, .2, .3, .1, .2, .2], dtype=np.float32)
        backend = M.MultichannelTrainingBackend()
        backend.prepare(train, select, data, cfg)
        for family in ("cnn", "attention"):
            cfg["model"]["family"] = family
            for arm in M.ARMS:
                cfg["arm"] = arm
                torch.manual_seed(cfg["seed"])
                cpu = backend.make_model(cfg)
                cuda = copy.deepcopy(cpu).to(runtime.device)
                permutation = reference_link_permutation(train, cfg["sham_seed"]) if arm == "sham" else None
                batch = backend.make_batch(train, [(0, 0), (0, 1)], cfg, DeviceRuntime(), permutation)
                snapshots = []
                for model, device in ((cpu, torch.device("cpu")), (cuda, runtime.device)):
                    tensors = {name: value.detach().clone().to(device) for name, value in batch.items()}
                    for value in tensors.values():
                        if value.is_floating_point():
                            value.requires_grad_(True)
                    output = backend.forward(model, tensors, cfg)
                    loss = F.nll_loss(output["log_probabilities"], torch.tensor([0, 0], device=device))
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    self.assertIsNone(tensors["baseline"].grad)
                    snapshots.append({
                        "probabilities": output["probabilities"].detach().cpu(),
                        "log_probabilities": output["log_probabilities"].detach().cpu(),
                        "loss": loss.detach().cpu(),
                        "parameters": {name: None if p.grad is None else p.grad.detach().cpu()
                                       for name, p in model.named_parameters()},
                        "inputs": {name: None if x.grad is None else x.grad.detach().cpu()
                                   for name, x in tensors.items() if x.is_floating_point()},
                    })
                left, right = snapshots
                for name in ("probabilities", "log_probabilities", "loss"):
                    torch.testing.assert_close(right[name], left[name], atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
                for group in ("parameters", "inputs"):
                    self.assertEqual(left[group].keys(), right[group].keys())
                    for name, value in left[group].items():
                        other = right[group][name]
                        self.assertEqual(value is None, other is None, name)
                        if value is not None:
                            torch.testing.assert_close(other, value, atol=GRADIENT_ATOL,
                                                       rtol=GRADIENT_RTOL, msg=f"{family}/{arm}/{name}")
                runtime.memory()


if __name__ == "__main__":
    unittest.main()
