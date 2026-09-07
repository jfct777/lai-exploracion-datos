"""CPU controls and CUDA API mocks; these tests do not certify real GPU execution."""
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import m39_profile_device as D
from test_m39_ordered_models import fixture, model_for


class ProfileDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_device_and_memory_envelope_rejected_without_cuda_initialization(self):
        for device, ceiling in (("cuda", 100), ("cuda:1", 100), ("cpu", 100),
                                ("cuda:0", 0), ("cuda:0", True), ("cpu", -1)):
            with self.subTest(device=device, ceiling=ceiling), self.assertRaises(ValueError):
                D.DeviceRuntime(device, ceiling)

    def test_cpu_path_never_calls_cuda_and_keeps_transfer_identity(self):
        runtime = D.DeviceRuntime()
        batch = {"value": torch.ones(2)}
        with patch.object(D.torch.cuda, "is_available", side_effect=AssertionError("unexpected CUDA")), \
                patch.object(D.torch.cuda, "synchronize", side_effect=AssertionError("unexpected CUDA")):
            self.assertEqual(runtime.configure(), {"device": "cpu"})
            self.assertIs(runtime.transfer(batch), batch)
            self.assertIsNone(runtime.memory())
            runtime.reset_measurement_peaks()
            self.assertGreaterEqual(runtime.elapsed(runtime.tick()), 0)

    def test_cuda_timers_wait_before_start_and_before_stop(self):
        events = []
        runtime = D.DeviceRuntime("cuda:0", 1000)
        def clock():
            events.append("clock")
            return float(len(events))
        with patch.object(D.torch.cuda, "synchronize", side_effect=lambda dev: events.append("sync")) as sync, \
                patch.object(D.time, "perf_counter", side_effect=clock):
            start = runtime.tick()
            self.assertEqual(runtime.elapsed(start), 2.0)
            self.assertEqual(events, ["sync", "clock", "sync", "clock"])
            sync.assert_called_with(torch.device("cuda:0"))

    def test_explicit_h2d_is_blocking_and_keeps_all_fields(self):
        runtime = D.DeviceRuntime("cuda:0", 1000)
        sources = {name: SimpleNamespace(to=Mock(return_value=name + "_device"))
                   for name in ("channels", "mask", "delta")}
        result = runtime.transfer(sources)
        self.assertEqual(result, {key: key + "_device" for key in sources})
        for tensor in sources.values():
            tensor.to.assert_called_once_with(torch.device("cuda:0"), non_blocking=False)

    def test_gpu_memory_separates_allocated_reserved_and_guards_both_peaks(self):
        runtime = D.DeviceRuntime("cuda:0", 1000)
        with ExitStack() as stack:
            stack.enter_context(patch.object(D.torch.cuda, "synchronize"))
            for name, value in (("memory_allocated", 100), ("memory_reserved", 400),
                                ("max_memory_allocated", 300), ("max_memory_reserved", 700)):
                stack.enter_context(patch.object(D.torch.cuda, name, return_value=value))
            self.assertEqual(runtime.memory(), {"allocated_bytes": 100, "reserved_bytes": 400,
                                               "peak_allocated_bytes": 300, "peak_reserved_bytes": 700})
            for name in ("max_memory_allocated", "max_memory_reserved"):
                with patch.object(D.torch.cuda, name, return_value=1001), self.assertRaisesRegex(ValueError, "ceiling"):
                    runtime.memory()

    def test_peak_reset_is_an_explicit_separate_setup_operation(self):
        events = []
        with patch.object(D.torch.cuda, "synchronize", side_effect=lambda dev: events.append("sync")), \
                patch.object(D.torch.cuda, "empty_cache", side_effect=lambda: events.append("empty")), \
                patch.object(D.torch.cuda, "reset_peak_memory_stats", side_effect=lambda dev: events.append("reset")):
            D.DeviceRuntime("cuda:0", 1000).reset_measurement_peaks()
        self.assertEqual(events, ["sync", "empty", "reset"])

    def configured_mock(self, **overrides):
        options = {"version": "2.12.1+cu126", "cuda": "12.6", "available": True,
                   "count": 1, "name": "NVIDIA L4", "workspace": ":4096:8"}
        options.update(overrides)
        backends = SimpleNamespace(fp32_precision="tf32",
                                   cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision="tf32")),
                                   cudnn=SimpleNamespace(fp32_precision="tf32", benchmark=True,
                                                        conv=SimpleNamespace(fp32_precision="tf32"),
                                                        version=lambda: 90000))
        properties = SimpleNamespace(name=options["name"], total_memory=24000, major=8, minor=9)
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": options["workspace"]}), \
                patch.object(D.torch, "__version__", options["version"]), \
                patch.object(D.torch.version, "cuda", options["cuda"]), \
                patch.object(D.torch.cuda, "is_available", return_value=options["available"]), \
                patch.object(D.torch.cuda, "device_count", return_value=options["count"]), \
                patch.object(D.torch.cuda, "get_device_properties", return_value=properties), \
                patch.object(D.torch, "backends", backends), \
                patch.object(D.torch, "use_deterministic_algorithms") as deterministic:
            result = D.DeviceRuntime("cuda:0", 8000).configure()
            deterministic.assert_called_once_with(True)
        return result, backends

    def test_gpu_configuration_sets_ieee_without_mixing_legacy_tf32_flags(self):
        result, backends = self.configured_mock()
        self.assertEqual(result["float32_precision"], "ieee")
        self.assertFalse(result["amp"])
        self.assertFalse(result["compile"])
        self.assertEqual(backends.cuda.matmul.fp32_precision, "ieee")
        self.assertEqual(backends.cudnn.conv.fp32_precision, "ieee")
        self.assertFalse(backends.cudnn.benchmark)
        self.assertFalse(hasattr(backends.cuda.matmul, "allow_tf32"))

    def test_runtime_version_availability_device_count_and_workspace_fail_closed(self):
        for change in ({"version": "2.12.1+cpu"}, {"version": "2.13.0+cu126"},
                       {"cuda": "13.0"}, {"available": False}, {"count": 0}, {"count": 2},
                       {"name": "NVIDIA A100"}, {"workspace": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.configured_mock(**change)

    def test_cpu_reference_control_compares_both_families_and_preserves_originals(self):
        for family in ("cnn", "attention"):
            model = model_for(family, dtype=torch.float32, depth=1, dilations=(1,), core_sites=4)
            batch = fixture(count=1, candidates=2, length=9, dtype=torch.float32)
            before_model = {key: value.clone() for key, value in model.state_dict().items()}
            before_batch = {key: value.clone() for key, value in batch.items()}
            result = D.check_warmup_parity(model, batch, torch.tensor([2]), torch.device("cpu"))
            self.assertEqual(result["candidate_device"], "cpu")
            self.assertEqual(result["logits_max_abs_error"], 0)
            self.assertEqual(set(result["parameter_max_abs_errors"]), dict(model.named_parameters()).keys())
            self.assertEqual(result["unused_float_inputs"], ["pooled_af"])
            self.assertIn("channels", result["input_max_abs_errors"])
            self.assertIn("delta_cm", result["input_max_abs_errors"])
            self.assertFalse(result["weights_saved"])
            self.assertFalse(result["predictions_saved"])
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before_model[name]))
            for name, value in batch.items():
                self.assertTrue(torch.equal(value, before_batch[name]))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_parity_failures_are_not_relaxed_or_silently_skipped(self):
        model = model_for("cnn", dtype=torch.float32, depth=1, dilations=(1,))
        batch = fixture(count=1, length=3, dtype=torch.float32)
        template = {"logits": torch.ones(1, 6), "loss": torch.tensor(1.),
                    "parameters": {"weight": torch.ones(2)},
                    "inputs": {"channels": torch.ones(2), "pooled_af": None}}
        for mode in ("logits", "gradient", "nan", "one_missing", "both_missing"):
            left, right = copy.deepcopy(template), copy.deepcopy(template)
            if mode == "logits":
                right["logits"][0, 0] += 0.01
            elif mode == "gradient":
                right["parameters"]["weight"][0] += 0.01
            elif mode == "nan":
                right["parameters"]["weight"][0] = float("nan")
            elif mode == "one_missing":
                right["inputs"]["channels"] = None
            else:
                left["inputs"]["channels"] = right["inputs"]["channels"] = None
            with patch.object(D, "_gradient_snapshot", side_effect=[left, right]), \
                    self.subTest(mode=mode), self.assertRaises(ValueError):
                D.check_warmup_parity(model, batch, torch.tensor([0]), "cpu")

    def test_cpu_parity_rejects_precision_change(self):
        model = model_for("cnn", dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "float32"):
            D.check_warmup_parity(model, fixture(count=1), torch.tensor([0]), "cpu")

    def test_relative_tolerance_uses_cpu_not_candidate_magnitude(self):
        with self.assertRaises(ValueError):
            D._compare(torch.tensor([1.]), torch.tensor([2.]), atol=0.1, rtol=0.5, name="direction")
        ratios = {}
        D._compare(torch.tensor([1.]), torch.tensor([1.3]), atol=0.1, rtol=0.5,
                   name="direction", ratios=ratios)
        self.assertAlmostEqual(ratios["direction"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
