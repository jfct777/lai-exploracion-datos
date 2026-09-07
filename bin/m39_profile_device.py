"""Synchronized device instrumentation and an unsaved CPU/device gradient check."""
from __future__ import annotations

import copy
import os
import time

import torch
from torch.nn import functional as F


LOGIT_ATOL, LOGIT_RTOL = 3e-5, 1e-4
GRADIENT_ATOL, GRADIENT_RTOL = 2e-6, 2e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DeviceRuntime:
    """CPU is a no-op backend; CUDA is explicit, synchronized and fail-closed."""

    def __init__(self, device="cpu", max_device_bytes=0):
        require(device in ("cpu", "cuda:0"), "only CPU or the single visible CUDA device is supported")
        require(type(max_device_bytes) is int and max_device_bytes >= 0, "invalid device byte ceiling")
        require((device == "cpu") == (max_device_bytes == 0), "device byte ceiling differs from backend")
        self.device = torch.device(device)
        self.max_device_bytes = max_device_bytes

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def configure(self):
        if not self.is_cuda:
            return {"device": "cpu"}
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
                "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization")
        require(str(torch.__version__) == "2.12.1+cu126" and torch.version.cuda == "12.6",
                "exact PyTorch 2.12.1+cu126 runtime required")
        require(torch.cuda.is_available(), "CUDA requested but unavailable; no CPU fallback")
        require(torch.cuda.device_count() == 1, "exactly one visible GPU required")
        properties = torch.cuda.get_device_properties(self.device)
        require(properties.name == "NVIDIA L4", "only the declared NVIDIA L4 device is in scope")
        require(self.max_device_bytes <= properties.total_memory, "device byte ceiling exceeds GPU capacity")
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        return {"device": str(self.device), "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [properties.major, properties.minor],
                "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "float32_precision": "ieee", "amp": False, "compile": False,
                "cudnn_benchmark": False, "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}

    def synchronize(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def tick(self):
        self.synchronize()
        return time.perf_counter()

    def elapsed(self, started):
        self.synchronize()
        return time.perf_counter() - started

    def transfer(self, batch):
        if not self.is_cuda:
            return batch
        return {key: value.to(self.device, non_blocking=False) for key, value in batch.items()}

    def memory(self):
        if not self.is_cuda:
            return None
        self.synchronize()
        memory = {"allocated_bytes": torch.cuda.memory_allocated(self.device),
                  "reserved_bytes": torch.cuda.memory_reserved(self.device),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
        require(memory["peak_allocated_bytes"] <= self.max_device_bytes and
                memory["peak_reserved_bytes"] <= self.max_device_bytes, "CUDA memory ceiling exceeded")
        return memory

    def reset_measurement_peaks(self):
        """Only before timed warmup, after the separate parity check is released."""
        if self.is_cuda:
            self.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)


def _gradient_snapshot(model, batch, labels, device):
    candidate = copy.deepcopy(model).to(device).train()
    tensors = {key: value.detach().to(device).clone() for key, value in batch.items()}
    for value in tensors.values():
        if value.is_floating_point():
            require(value.dtype == torch.float32, "parity inputs must remain float32")
            value.requires_grad_(True)
    candidate.zero_grad(set_to_none=True)
    logits = candidate(tensors, arm="real", chunked=True)
    loss = F.cross_entropy(logits, labels.to(device))
    loss.backward()
    parameters = {}
    for name, parameter in candidate.named_parameters():
        require(parameter.grad is not None, f"missing parity parameter gradient: {name}")
        parameters[name] = parameter.grad.detach().cpu().clone()
    inputs = {key: None if value.grad is None else value.grad.detach().cpu().clone()
              for key, value in tensors.items() if value.is_floating_point()}
    return {"logits": logits.detach().cpu().clone(), "loss": loss.detach().cpu().clone(),
            "parameters": parameters, "inputs": inputs}


def _compare(left, right, *, atol, rtol, name, ratios=None):
    require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()),
            f"nonfinite parity value: {name}")
    try:
        # The CPU value, not the candidate GPU value, sets the relative allowance.
        torch.testing.assert_close(right, left, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise ValueError(f"CPU/device parity failed: {name}") from error
    difference = (left - right).abs()
    if ratios is not None:
        ratios[name] = float((difference / (atol + rtol * left.abs())).max()) if left.numel() else 0.0
    return float(difference.max()) if left.numel() else 0.0


def check_warmup_parity(model, batch_cpu, labels_cpu, device):
    """Compare identical unsaved weights/inputs/labels; never optimize or return them.

    Both paths retain all sites and the same checkpointed encoder. This checks
    cross-device numerics, not a new proof of full/chunk equivalence. Unused
    floating input gradients must be absent on both sides, not silently skipped.
    """
    require(all(value.device.type == "cpu" for value in batch_cpu.values()), "CPU parity batch required")
    require(labels_cpu.device.type == "cpu", "CPU parity labels required")
    # Initialization happens only on CPU. Device copies cannot draw different weights.
    reference_model = copy.deepcopy(model).cpu()
    require(all(value.dtype == torch.float32 for value in reference_model.parameters()),
            "parity model must remain float32")
    left = _gradient_snapshot(reference_model, batch_cpu, labels_cpu, "cpu")
    right = _gradient_snapshot(reference_model, batch_cpu, labels_cpu, device)
    ratios = {}
    logits_error = _compare(left["logits"], right["logits"], atol=LOGIT_ATOL, rtol=LOGIT_RTOL,
                            name="logits", ratios=ratios)
    loss_error = _compare(left["loss"], right["loss"], atol=LOGIT_ATOL, rtol=LOGIT_RTOL,
                          name="loss", ratios=ratios)
    parameter_errors, input_errors, unused = {}, {}, []
    require(left["parameters"].keys() == right["parameters"].keys(), "parity parameter inventory differs")
    for name, value in left["parameters"].items():
        parameter_errors[name] = _compare(value, right["parameters"][name], atol=GRADIENT_ATOL,
                                           rtol=GRADIENT_RTOL, name="parameter " + name, ratios=ratios)
    require(left["inputs"].keys() == right["inputs"].keys(), "parity input inventory differs")
    for name, value in left["inputs"].items():
        other = right["inputs"][name]
        require((value is None) == (other is None), f"parity input gradient support differs: {name}")
        if value is None:
            require(name == "pooled_af", f"required REAL input is disconnected on both devices: {name}")
            unused.append(name)
        else:
            input_errors[name] = _compare(value, other, atol=GRADIENT_ATOL, rtol=GRADIENT_RTOL,
                                          name="input " + name, ratios=ratios)
    return {"decision": "PASS_CPU_DEVICE_WARMUP_GRADIENT_PARITY", "reference_device": "cpu",
            "candidate_device": str(device), "logits_max_abs_error": logits_error,
            "loss_abs_error": loss_error, "parameter_max_abs_errors": parameter_errors,
            "input_max_abs_errors": input_errors, "unused_float_inputs": unused,
            "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL,
            "gradient_atol": GRADIENT_ATOL, "gradient_rtol": GRADIENT_RTOL,
            "relative_tolerance_reference": "cpu", "max_tolerance_ratio_by_tensor": ratios,
            "full_vs_chunk_retested": False, "weights_saved": False, "predictions_saved": False}
