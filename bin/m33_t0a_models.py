#!/usr/bin/env python3
"""Frozen PRE-4 model families used only by the truth-free M33 T0a gate."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


FAMILIES = ("local_linear", "small_residual_cnn_1d")
CHANNELS = 13
ANCESTRIES = 3
HAPLOTYPES = 2
MODEL_SEED = 20260823
SIMPLEX_ATOL = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def configure_deterministic_cpu() -> None:
    """Select deterministic single-thread CPU execution for cross-process checks."""
    # Docker preserves the host UID, which may not exist in the image's passwd file.
    # PyTorch consults the user name while configuring its deterministic CPU runtime.
    os.environ.setdefault("USER", "m33_t0a")
    os.environ.setdefault("LOGNAME", "m33_t0a")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(MODEL_SEED)


def _validate_inputs(tokens: torch.Tensor, mask: torch.Tensor,
                     f0: torch.Tensor) -> None:
    require(tokens.ndim == 3 and tokens.shape[2] == CHANNELS,
            "tokens must be rows x context x 13")
    require(mask.shape == tokens.shape[:2], "mask axis differs")
    require(f0.shape == (tokens.shape[0], HAPLOTYPES, ANCESTRIES),
            "F0 axis differs")
    require(tokens.dtype == mask.dtype == f0.dtype == torch.float32,
            "T0a tensors must be float32")
    require(torch.isfinite(tokens).all().item() and torch.isfinite(f0).all().item(),
            "non-finite model input")
    require(torch.all(f0 >= 0).item() and
            torch.max(torch.abs(f0.sum(dim=2) - 1.0)).item() <= 5e-6,
            "F0 does not satisfy the simplex")
    require(torch.all((mask == 0) | (mask == 1)).item(), "mask domain differs")


def _masked_mean_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(2)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (tokens * weights).sum(dim=1) / count
    masked = tokens.masked_fill(weights == 0, float("-inf"))
    maximum = masked.amax(dim=1)
    empty = mask.sum(dim=1) == 0
    maximum = torch.where(empty.unsqueeze(1), torch.zeros_like(maximum), maximum)
    mean = torch.where(empty.unsqueeze(1), torch.zeros_like(mean), mean)
    return torch.cat((mean, maximum), dim=1)


def _correct_f0(f0: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    logits = torch.log(f0.clamp_min(1e-7)) + delta
    return torch.softmax(logits, dim=2)


class LocalLinear(nn.Module):
    """PRE-4 masked mean/max summary followed by one shared 29x3 head."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(29, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_with_features(self, tokens: torch.Tensor, mask: torch.Tensor,
                              f0: torch.Tensor):
        _validate_inputs(tokens, mask, f0)
        pooled = _masked_mean_max(tokens, mask)
        features = torch.cat((pooled[:, None, :].expand(-1, 2, -1), f0), dim=2)
        delta = self.head(features)
        return _correct_f0(f0, delta), delta, features

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                f0: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(tokens, mask, f0)[0]


class MaskedGroupNorm(nn.Module):
    """Four-group normalization that excludes padded positions."""

    def __init__(self, channels: int = 16, groups: int = 4, eps: float = 1e-5) -> None:
        super().__init__()
        require(channels % groups == 0, "group count does not divide channels")
        self.channels, self.groups, self.eps = channels, groups, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, channels, length = values.shape
        require(channels == self.channels and mask.shape == (n, 1, length),
                "masked group normalization axes differ")
        grouped = values.reshape(n, self.groups, channels // self.groups, length)
        weights = mask[:, None, :, :]
        denominator = (weights.sum(dim=(2, 3), keepdim=True) *
                       (channels // self.groups)).clamp_min(1.0)
        mean = (grouped * weights).sum(dim=(2, 3), keepdim=True) / denominator
        variance = ((grouped - mean).square() * weights).sum(
            dim=(2, 3), keepdim=True) / denominator
        normalized = ((grouped - mean) * torch.rsqrt(variance + self.eps)).reshape(
            n, channels, length)
        normalized = normalized * self.weight[None, :, None] + self.bias[None, :, None]
        return normalized * mask


class ResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(16, 16, kernel_size=5, dilation=dilation,
                                   padding=2 * dilation, groups=16, bias=True)
        self.pointwise = nn.Conv1d(16, 16, kernel_size=1, bias=True)
        self.norm = MaskedGroupNorm(16, 4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(values) * mask
        update = self.pointwise(update) * mask
        update = self.norm(update, mask)
        update = self.dropout(F.gelu(update)) * mask
        return (values + update) * mask


class SmallResidualCNN1D(nn.Module):
    """Exact 1,651-parameter residual CNN frozen by the PRE-4 contract."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(13, 16, kernel_size=1, bias=True)
        self.block1 = ResidualBlock(dilation=1)
        self.block2 = ResidualBlock(dilation=4)
        self.head1 = nn.Linear(35, 16)
        self.dropout = nn.Dropout(0.1)
        self.head2 = nn.Linear(16, 3)
        nn.init.zeros_(self.head2.weight)
        nn.init.zeros_(self.head2.bias)

    def forward_with_features(self, tokens: torch.Tensor, mask: torch.Tensor,
                              f0: torch.Tensor):
        _validate_inputs(tokens, mask, f0)
        token_mask = mask[:, None, :]
        values = F.gelu(self.stem(tokens.transpose(1, 2))) * token_mask
        values = self.block1(values, token_mask)
        values = self.block2(values, token_mask)
        pooled = _masked_mean_max(values.transpose(1, 2), mask)
        features = torch.cat((pooled[:, None, :].expand(-1, 2, -1), f0), dim=2)
        hidden = self.dropout(F.gelu(self.head1(features)))
        delta = self.head2(hidden)
        return _correct_f0(f0, delta), delta, features

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                f0: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(tokens, mask, f0)[0]


def build_model(family: str) -> nn.Module:
    require(family in FAMILIES, "unknown PRE-4 model family")
    torch.manual_seed(MODEL_SEED)
    model: nn.Module = LocalLinear() if family == "local_linear" else SmallResidualCNN1D()
    model.eval()
    require(parameter_count(model) == (90 if family == "local_linear" else 1651),
            "PRE-4 parameter count differs")
    return model


def activate_deterministic_probe_head(model: nn.Module) -> None:
    """Make a private nonzero control that exercises the complete residual path."""
    final = model.head if isinstance(model, LocalLinear) else model.head2
    with torch.no_grad():
        values = torch.arange(final.weight.numel(), dtype=torch.float32).reshape_as(final.weight)
        final.weight.copy_((values.remainder(11) - 5) * 1e-3)
        final.bias.copy_(torch.tensor([-0.01, 0.0, 0.01], dtype=torch.float32))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_shape_sha256(model: nn.Module) -> str:
    payload = [(name, list(parameter.shape), str(parameter.dtype))
               for name, parameter in model.named_parameters()]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def parameter_value_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def assert_probabilities(probabilities: torch.Tensor) -> None:
    require(torch.isfinite(probabilities).all().item(), "non-finite prediction")
    require(torch.all(probabilities >= 0).item(), "negative prediction")
    error = torch.max(torch.abs(probabilities.sum(dim=2) - 1.0)).item()
    require(error <= SIMPLEX_ATOL, "prediction does not satisfy the simplex")
