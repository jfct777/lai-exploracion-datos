#!/usr/bin/env python3
import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import r02_lai_fit_monitor as monitor
import r02_context_train as trainer


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, value):
        return torch.softmax(self.linear(self.dropout(value)), dim=-1)


class FitMonitorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = ToyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        sum(parameter.sum() for parameter in self.model.parameters()).backward()
        self.optimizer.step()  # Materialize Adam state and its step counters.
        self.model.train()
        self.model.dropout.eval()  # Verify mixed per-module modes are restored.
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        self.fit = {"stage": "FIT", "path": "fixed-fit.npz", "sha256": "a" * 64}
        self.select = {"stage": "SELECT", "path": "closed-select.npz", "sha256": "b" * 64}
        self.contract = {"objective": "lai", "batches": {"FIT": [self.fit], "SELECT": [self.select]}}
        self.resolved = {"family": "lai_cnn", "seed": 1701}

    @staticmethod
    def validate(entry, contract, resolved, people_per_batch):
        random.random()
        np.random.random()
        torch.rand(1)
        return {"x": torch.tensor([[1.0, -1.0], [0.5, 0.25]]),
                "truth": torch.tensor([0, 1])}

    @staticmethod
    def predict(model, validated, entry, contract, resolved, arm, people_per_batch):
        return model(validated["x"])

    @staticmethod
    def loss(prediction, validated, entry):
        truth = validated["truth"]
        selected = prediction[torch.arange(len(truth)), truth]
        return {"log_loss_sum": float((-torch.log(selected)).sum()),
                "observed_count": len(truth),
                "brier_sum": float(((prediction - torch.nn.functional.one_hot(truth, 2)) ** 2).sum()),
                "state_error_sum": float((prediction.argmax(1) != truth).sum()),
                "dosage_abs_sum": [1.0, 2.0, 3.0]}

    def test_preserves_training_state_and_is_idempotent(self):
        random.seed(23)
        np.random.seed(29)
        torch.manual_seed(31)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        modes = [module.training for module in self.model.modules()]
        state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        gradients = [parameter.grad.detach().clone() for parameter in self.model.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "step-000012.json"
            receipt = monitor.score_fixed_fit(
                model=self.model, optimizer=self.optimizer,
                contract=self.contract, resolved=self.resolved,
                fit_entries=[copy.deepcopy(self.fit)], arm="NONE", people_per_batch=128,
                step=12, output=output, validate_entry=self.validate,
                predict_entry=self.predict, loss_statistics=self.loss)
            first = output.read_bytes()
            repeated = monitor.score_fixed_fit(
                model=self.model, optimizer=self.optimizer,
                contract=self.contract, resolved=self.resolved,
                fit_entries=[copy.deepcopy(self.fit)], arm="NONE", people_per_batch=128,
                step=12, output=output, validate_entry=self.validate,
                predict_entry=self.predict, loss_statistics=self.loss)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(receipt, repeated)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(all(torch.equal(self.model.state_dict()[name], value)
                            for name, value in state.items()))
        self.assertTrue(all(torch.equal(parameter.grad, gradient)
                            for parameter, gradient in zip(self.model.parameters(), gradients)))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(all(np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
                            for left, right in zip(numpy_state, np.random.get_state())))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertFalse(receipt["select_accessed"])
        self.assertEqual(receipt["metrics"]["observed_count"], 2)
        self.assertEqual(receipt["model_state_sha256_before"],
                         receipt["model_state_sha256_after"])
        self.assertEqual(receipt["training_state_sha256_before"],
                         receipt["training_state_sha256_after"])
        self.assertEqual(receipt["optimizer_state_sha256_before"],
                         receipt["optimizer_state_sha256_after"])
        self.assertEqual(receipt["optimizer_step_counters_before"], [1])

    def test_rejects_select_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not an exact contract FIT entry"):
                monitor.score_fixed_fit(
                    model=self.model, optimizer=self.optimizer,
                    contract=self.contract, resolved=self.resolved,
                    fit_entries=[self.select], arm="NONE", people_per_batch=128, step=0,
                    output=Path(directory) / "score.json", validate_entry=self.validate,
                    predict_entry=self.predict, loss_statistics=self.loss)

    def test_adversarial_optimizer_mutation_fails_after_restoration(self):
        before = monitor.optimizer_state_sha256(self.optimizer)

        def mutate_optimizer(model, validated, entry, contract, resolved, arm, people_per_batch):
            self.optimizer.param_groups[0]["lr"] = 123.0
            return self.predict(model, validated, entry, contract, resolved, arm, people_per_batch)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "attempted to mutate"):
                monitor.score_fixed_fit(
                    model=self.model, optimizer=self.optimizer,
                    contract=self.contract, resolved=self.resolved,
                    fit_entries=[self.fit], arm="NONE", people_per_batch=128, step=12,
                    output=Path(directory) / "score.json", validate_entry=self.validate,
                    predict_entry=mutate_optimizer, loss_statistics=self.loss)
            self.assertFalse((Path(directory) / "score.json").exists())
        self.assertEqual(before, monitor.optimizer_state_sha256(self.optimizer))

    def test_context_integration_fills_missing_then_reuses_after_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit = []
            selected = []
            for case_index in range(3):
                case = f"case-{case_index}"
                case_entries = []
                for region in range(3):
                    receipt_path = root / f"{case}.region-{region:02d}.receipt.json"
                    receipt_path.write_text(__import__("json").dumps({
                        "role": "FIT", "case_id": case, "region_id": f"region-{region:02d}"
                    }, sort_keys=True) + "\n")
                    entry = {"stage": "FIT", "role": "FIT", "lai_case_id": case,
                             "path": str(root / f"{case}.region-{region:02d}.json"),
                             "sha256": f"{case_index + 1}{region}".ljust(64, "0"),
                             "receipt_path": str(receipt_path),
                             "receipt_sha256": trainer.sha(receipt_path)}
                    fit.append(entry)
                    case_entries.append(entry)
                selected.append(case_entries[1])
            manifest_path = root / "fit-monitor-manifest.json"
            manifest_path.write_text(__import__("json").dumps({
                "schema": trainer.LAI_FIT_MONITOR_MANIFEST_SCHEMA,
                "source_stage": "FIT",
                "selection_rule": trainer.LAI_FIT_MONITOR_SELECTION,
                "case_count": 3,
                "persons_per_entry": 128,
                "metrics": list(trainer.LAI_FIT_MONITOR_METRICS),
                "entries": selected,
                "interpretation_limit":
                    "FIXED_FIT_NUMERICAL_TRAJECTORY_NOT_SELECTION_OR_BIOLOGICAL_EVALUATION",
            }, sort_keys=True) + "\n")
            contract = {"objective": "lai", "batches": {"FIT": fit, "SELECT": [self.select]},
                        "lai_fit_monitor": {
                            "schema": trainer.LAI_FIT_MONITOR_CONFIG_SCHEMA,
                            "manifest_path": str(manifest_path),
                            "manifest_sha256": trainer.sha(manifest_path),
                            "steps": list(trainer.LAI_FIT_MONITOR_STEPS),
                            "people_per_batch": 8}}
            calls = {"validate": 0}
            for state in self.optimizer.state.values():
                state["step"].fill_(24)

            def validate(entry, contract, resolved, people_per_batch):
                calls["validate"] += 1
                return self.validate(entry, contract, resolved, people_per_batch)

            receipt = trainer.run_lai_fit_monitor(
                self.model, self.optimizer, contract, self.resolved, "NONE", 24, root,
                validate_entry_callback=validate, predict_entry_callback=self.predict,
                loss_statistics_callback=self.loss)
            self.assertEqual(calls["validate"], 3)
            self.assertEqual(receipt["step"], 24)
            output = root / "fit-monitor" / "step-000000000024.json"
            self.assertTrue(output.is_file())

            def must_not_repeat(*args, **kwargs):
                raise AssertionError("durable FIT monitor result was recomputed")

            repeated = trainer.run_lai_fit_monitor(
                self.model, self.optimizer, contract, self.resolved, "NONE", 24, root,
                validate_entry_callback=must_not_repeat, predict_entry_callback=must_not_repeat,
                loss_statistics_callback=must_not_repeat)
            self.assertEqual(receipt, repeated)
            self.assertEqual(calls["validate"], 3)
            with self.assertRaisesRegex(ValueError, "missing earlier FIT monitor"):
                trainer.require_lai_fit_monitor_history(
                    contract, self.resolved, "NONE", 49, root)
            self.assertIsNone(trainer.run_lai_fit_monitor(
                self.model, self.optimizer, contract, self.resolved, "NONE", 23, root,
                validate_entry_callback=must_not_repeat, predict_entry_callback=must_not_repeat,
                loss_statistics_callback=must_not_repeat))


if __name__ == "__main__":
    unittest.main()
