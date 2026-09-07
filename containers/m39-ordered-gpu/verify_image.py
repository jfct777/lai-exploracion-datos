#!/usr/bin/env python3
"""Check the pinned CUDA runtime without claiming a GPU was exercised at build time."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys


def verify_runtime(*, require_cuda=False):
    import numpy as np
    import torch
    from torch.utils.checkpoint import checkpoint

    expected = {"torch": "2.12.1+cu126", "numpy": "2.4.6"}
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 required, found {platform.python_version()}")
    for package, version in expected.items():
        observed = importlib.metadata.version(package)
        if observed != version:
            raise RuntimeError(f"{package}: expected {version}, found {observed}")
    if torch.__version__ != expected["torch"] or torch.version.cuda != "12.6":
        raise RuntimeError("Expected the explicit PyTorch CUDA 12.6 build, not CPU torch")
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR") != "/tmp/m39-torch-cache":
        raise RuntimeError("The cache must be explicit for execution under arbitrary numeric UIDs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("The deterministic cuBLAS workspace must be configured before import")

    # Small CPU-only integration test: NumPy bridge, checkpoint and AdamW secondary
    # imports must work even when no GPU or passwd entry for the task UID exists.
    torch.set_num_threads(2)
    parameter = torch.nn.Parameter(torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    before = parameter.detach().clone()
    loss = checkpoint(lambda value: value.square().mean(), parameter, use_reentrant=False)
    loss.backward()
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError("Checkpoint/AdamW CPU integration produced invalid gradients")
    optimizer.step()
    if torch.equal(parameter.detach(), before) or not torch.isfinite(parameter).all():
        raise RuntimeError("AdamW failed to update finite parameters")

    available = bool(torch.cuda.is_available())
    if require_cuda and not available:
        raise RuntimeError("CUDA execution was explicitly required but no GPU is available")
    result = {
        "schema": "m39-cuda-image-verification-v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "compiled_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": available,
        "cuda_exercised": False,
        "cpu_checkpoint_adamw_smoke": "PASS",
        "uid": os.getuid(),
        "scientific_model_or_data_loaded": False,
    }
    if require_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        value = torch.ones((2, 2), dtype=torch.float32, device="cuda")
        product = value @ value
        torch.cuda.synchronize()
        if not torch.equal(product.cpu(), torch.full((2, 2), 2.0)):
            raise RuntimeError("CUDA float32 matrix multiplication smoke failed")
        result.update(cuda_exercised=True, device_count=torch.cuda.device_count(),
                      device_name=torch.cuda.get_device_name(0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_runtime(require_cuda=args.require_cuda), sort_keys=True))


if __name__ == "__main__":
    main()
