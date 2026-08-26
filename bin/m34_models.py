#!/usr/bin/env python3
"""Compact, configurable residual model families for the M34 LAI screen."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


FAMILIES = (
    "local_linear",
    "residual_cnn_1d",
    "lainns_cnn_1d",
    "unet_1d",
    "bilstm",
    "tcn",
    "transformer_small",
)


@dataclass(frozen=True)
class ModelSpec:
    """Architecture choices shared by every model in the exploratory screen."""

    family: str
    channels: int
    ancestries: int
    hidden_dim: int = 32
    depth: int = 3
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    lstm_layers: int = 1
    transformer_heads: int = 4
    transformer_ff_dim: int = 64
    transformer_max_tokens: int = 256
    zero_init_head: bool = True
    seed: int = 20260826

    def __post_init__(self) -> None:
        _require(self.family in FAMILIES, f"unknown model family: {self.family}")
        _require(self.channels > 0, "channels must be positive")
        _require(self.ancestries >= 2, "at least two ancestries are required")
        _require(self.hidden_dim >= 4, "hidden_dim must be at least four")
        _require(self.depth >= 1, "depth must be positive")
        _require(self.kernel_size >= 3 and self.kernel_size % 2 == 1,
                 "kernel_size must be odd and at least three")
        _require(self.dilations and all(value >= 1 for value in self.dilations),
                 "dilations must contain positive integers")
        _require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        _require(self.lstm_layers >= 1, "lstm_layers must be positive")
        _require(self.transformer_heads >= 1, "transformer_heads must be positive")
        if self.family == "transformer_small":
            _require(self.hidden_dim % self.transformer_heads == 0,
                     "hidden_dim must be divisible by transformer_heads")
            _require(self.transformer_ff_dim >= self.hidden_dim,
                     "transformer_ff_dim cannot be smaller than hidden_dim")
            _require(self.transformer_max_tokens >= 8,
                     "transformer_max_tokens must be at least eight")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    _require(mask.dtype == torch.bool or not torch.is_floating_point(mask) or
             torch.isfinite(mask).all().item(), "mask contains non-finite values")
    _require(torch.all((mask == 0) | (mask == 1)).item(), "mask must be binary")
    return mask.to(dtype=torch.bool)


def _validate_inputs(spec: ModelSpec, tokens: torch.Tensor, mask: torch.Tensor,
                     baseline: torch.Tensor) -> torch.Tensor:
    _require(tokens.ndim == 3, "tokens must have shape [batch, context, channels]")
    _require(tokens.shape[2] == spec.channels, "token channel count differs from ModelSpec")
    _require(tokens.shape[1] >= 1, "context length must be positive")
    _require(mask.shape == tokens.shape[:2], "mask axes differ from tokens")
    _require(baseline.ndim == 3 and baseline.shape[0] == tokens.shape[0],
             "baseline must have shape [batch, haplotypes, ancestries]")
    _require(baseline.shape[2] == spec.ancestries,
             "baseline ancestry count differs from ModelSpec")
    _require(torch.is_floating_point(tokens) and torch.is_floating_point(baseline),
             "tokens and baseline must be floating point")
    _require(tokens.dtype == baseline.dtype, "tokens and baseline must share dtype")
    _require(torch.isfinite(tokens).all().item() and torch.isfinite(baseline).all().item(),
             "model inputs contain non-finite values")
    _require(torch.all(baseline >= 0).item(), "baseline contains negative probabilities")
    tolerance = 2e-3 if baseline.dtype in (torch.float16, torch.bfloat16) else 1e-5
    _require(torch.max(torch.abs(baseline.sum(dim=-1) - 1.0)).item() <= tolerance,
             "baseline does not satisfy the simplex")
    return _as_bool_mask(mask)


def _masked_mean_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool a ragged sequence without letting padding affect the summary."""
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    counts = weights.sum(dim=1).clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / counts
    lower = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~mask.unsqueeze(-1), lower).amax(dim=1)
    empty = ~mask.any(dim=1)
    mean = torch.where(empty.unsqueeze(1), torch.zeros_like(mean), mean)
    maximum = torch.where(empty.unsqueeze(1), torch.zeros_like(maximum), maximum)
    return torch.cat((mean, maximum), dim=-1)


def _apply_residual(baseline: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply a residual while allowing evidence to reopen a zero baseline class."""
    logits = torch.log(baseline.clamp_min(1e-7)) + delta
    return torch.softmax(logits, dim=-1)


def _initialize_output_head(head: nn.Linear, zero: bool) -> None:
    if zero:
        nn.init.zeros_(head.weight)
    else:
        # A small residual exposes gradients upstream without overwhelming the baseline.
        nn.init.normal_(head.weight, mean=0.0, std=1e-3)
    nn.init.zeros_(head.bias)


class ResidualLAIModel(nn.Module):
    """Common input validation, pooling and residual ancestry head."""

    def __init__(self, spec: ModelSpec, encoded_dim: int, head_hidden: int | None) -> None:
        super().__init__()
        self.spec = spec
        feature_dim = 2 * encoded_dim + spec.ancestries
        if head_hidden is None:
            self.pre_head = nn.Identity()
            output_dim = feature_dim
        else:
            self.pre_head = nn.Sequential(
                nn.Linear(feature_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(spec.dropout),
            )
            output_dim = head_hidden
        self.output_head = nn.Linear(output_dim, spec.ancestries)
        _initialize_output_head(self.output_head, spec.zero_init_head)

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_with_delta(self, tokens: torch.Tensor, mask: torch.Tensor,
                           baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = _validate_inputs(self.spec, tokens, mask, baseline)
        encoded, encoded_mask = self.encode(tokens * valid.unsqueeze(-1), valid)
        pooled = _masked_mean_max(encoded, encoded_mask)
        per_haplotype = pooled[:, None, :].expand(-1, baseline.shape[1], -1)
        features = torch.cat((per_haplotype, baseline), dim=-1)
        delta = self.output_head(self.pre_head(features))
        return _apply_residual(baseline, delta), delta

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                baseline: torch.Tensor) -> torch.Tensor:
        return self.forward_with_delta(tokens, mask, baseline)[0]


class LocalLinear(ResidualLAIModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, encoded_dim=spec.channels, head_hidden=None)

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tokens, mask


class PositionwiseLayerNorm(nn.Module):
    """Channel normalization at each locus, independent of context padding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(values.transpose(1, 2)).transpose(1, 2)
        return normalized * mask[:, None, :]


class MaskedResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                   padding=padding, dilation=dilation,
                                   groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = PositionwiseLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(values) * mask[:, None, :]
        update = self.pointwise(update) * mask[:, None, :]
        update = self.norm(update, mask)
        update = self.dropout(F.gelu(update)) * mask[:, None, :]
        return (values + update) * mask[:, None, :]


class ResidualCNN1D(ResidualLAIModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, encoded_dim=spec.hidden_dim, head_hidden=spec.hidden_dim)
        self.stem = nn.Conv1d(spec.channels, spec.hidden_dim, 1)
        self.blocks = nn.ModuleList(
            MaskedResidualConvBlock(
                spec.hidden_dim,
                spec.kernel_size,
                1,
                spec.dropout,
            )
            for index in range(spec.depth)
        )

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = F.gelu(self.stem(tokens.transpose(1, 2))) * mask[:, None, :]
        for block in self.blocks:
            values = block(values, mask)
        return values.transpose(1, 2), mask


class LAiNNSCNN1D(ResidualLAIModel):
    """Multi-scale CNN for short, intermediate and broad locus patterns."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, encoded_dim=spec.hidden_dim, head_hidden=spec.hidden_dim)
        broad = 2 * spec.kernel_size + 1
        self.branches = nn.ModuleList(
            nn.Conv1d(spec.channels, spec.hidden_dim, kernel,
                      padding=(kernel - 1) // 2)
            for kernel in (3, spec.kernel_size, broad)
        )
        self.fuse = nn.Conv1d(3 * spec.hidden_dim, spec.hidden_dim, 1)
        self.norm = PositionwiseLayerNorm(spec.hidden_dim)
        self.blocks = nn.ModuleList(
            MaskedResidualConvBlock(
                spec.hidden_dim,
                spec.kernel_size,
                spec.dilations[index % len(spec.dilations)],
                spec.dropout,
            )
            for index in range(max(1, spec.depth - 1))
        )

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = tokens.transpose(1, 2)
        branches = [F.gelu(branch(values)) * mask[:, None, :] for branch in self.branches]
        fused = self.fuse(torch.cat(branches, dim=1)) * mask[:, None, :]
        fused = F.gelu(self.norm(fused, mask)) * mask[:, None, :]
        for block in self.blocks:
            fused = block(fused, mask)
        return fused.transpose(1, 2), mask


class MaskedConvUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = PositionwiseLayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = PositionwiseLayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.conv1(values) * mask[:, None, :]
        values = self.dropout(F.gelu(self.norm1(values, mask))) * mask[:, None, :]
        values = self.conv2(values) * mask[:, None, :]
        return F.gelu(self.norm2(values, mask)) * mask[:, None, :]


def _masked_pool(values: torch.Tensor,
                 mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average pairs using only valid loci; ceil mode keeps odd final positions."""
    weights = mask[:, None, :].to(values.dtype)
    numerator = F.avg_pool1d(values * weights, 2, stride=2, ceil_mode=True,
                             count_include_pad=False)
    denominator = F.avg_pool1d(weights, 2, stride=2, ceil_mode=True,
                               count_include_pad=False)
    pooled_mask = denominator[:, 0, :] > 0
    pooled = numerator / denominator.clamp_min(1e-8)
    return pooled * pooled_mask[:, None, :], pooled_mask


def _upsample_prefix(values: torch.Tensor, length: int) -> torch.Tensor:
    """Repeat each coarse cell twice, then crop/pad without length-dependent interpolation."""
    expanded = values.repeat_interleave(2, dim=-1)
    if expanded.shape[-1] < length:
        expanded = F.pad(expanded, (0, length - expanded.shape[-1]))
    return expanded[..., :length]


class UNet1D(ResidualLAIModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, encoded_dim=spec.hidden_dim, head_hidden=spec.hidden_dim)
        levels = max(2, spec.depth)
        widths = [spec.hidden_dim * (2 ** level) for level in range(levels)]
        self.down = nn.ModuleList()
        input_width = spec.channels
        for width in widths:
            self.down.append(MaskedConvUnit(input_width, width, spec.kernel_size, spec.dropout))
            input_width = width
        self.up = nn.ModuleList()
        for level in range(levels - 2, -1, -1):
            self.up.append(MaskedConvUnit(
                widths[level + 1] + widths[level], widths[level],
                spec.kernel_size, spec.dropout,
            ))

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = tokens.transpose(1, 2)
        current_mask = mask
        skips: list[tuple[torch.Tensor, torch.Tensor]] = []
        for level, unit in enumerate(self.down):
            values = unit(values, current_mask)
            if level < len(self.down) - 1:
                skips.append((values, current_mask))
                values, current_mask = _masked_pool(values, current_mask)
        for unit, (skip, skip_mask) in zip(self.up, reversed(skips)):
            values = _upsample_prefix(values, skip.shape[-1]) * skip_mask[:, None, :]
            values = unit(torch.cat((values, skip), dim=1), skip_mask)
            current_mask = skip_mask
        return values.transpose(1, 2), current_mask


def _compact_masked_sequences(values: torch.Tensor, mask: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Remove masked holes before a recurrent model and retain scatter indices."""
    indices = [torch.nonzero(row, as_tuple=False).squeeze(1) for row in mask]
    lengths = [max(1, int(index.numel())) for index in indices]
    compact = values.new_zeros((values.shape[0], max(lengths), values.shape[2]))
    compact_mask = torch.zeros((values.shape[0], max(lengths)), dtype=torch.bool,
                               device=values.device)
    for row, index in enumerate(indices):
        if index.numel():
            compact[row, :index.numel()] = values[row, index]
            compact_mask[row, :index.numel()] = True
    return compact, compact_mask, indices


class BiLSTM(ResidualLAIModel):
    def __init__(self, spec: ModelSpec) -> None:
        recurrent_dim = spec.hidden_dim if spec.hidden_dim % 2 == 0 else spec.hidden_dim + 1
        super().__init__(spec, encoded_dim=recurrent_dim, head_hidden=spec.hidden_dim)
        self.recurrent_dim = recurrent_dim
        self.input_projection = nn.Linear(spec.channels, recurrent_dim)
        self.lstm = nn.LSTM(
            input_size=recurrent_dim,
            hidden_size=recurrent_dim // 2,
            num_layers=spec.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=spec.dropout if spec.lstm_layers > 1 else 0.0,
        )

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = F.gelu(self.input_projection(tokens)) * mask.unsqueeze(-1)
        compact, compact_mask, _indices = _compact_masked_sequences(projected, mask)
        lengths = compact_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(compact, lengths, batch_first=True,
                                      enforce_sorted=False)
        packed_output, _state = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=compact.shape[1])
        output = output * compact_mask.unsqueeze(-1)
        return output, compact_mask


class MaskedGatedTCNBlock(nn.Module):
    """Gated two-branch TCN update with a symmetric dilated receptive field."""

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.filter_conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding,
            dilation=dilation, groups=channels)
        self.gate_conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding,
            dilation=dilation, groups=channels)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = PositionwiseLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_mask = mask[:, None, :]
        filtered = torch.tanh(self.filter_conv(values))
        gated = torch.sigmoid(self.gate_conv(values))
        update = self.mix(filtered * gated) * token_mask
        update = self.dropout(self.norm(update, mask)) * token_mask
        return (values + update) * token_mask


class TCN(ResidualLAIModel):
    """Symmetric dilated TCN that sees loci on both sides of the central marker."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, encoded_dim=spec.hidden_dim, head_hidden=spec.hidden_dim)
        self.stem = nn.Conv1d(spec.channels, spec.hidden_dim, 1)
        self.blocks = nn.ModuleList(
            MaskedGatedTCNBlock(
                spec.hidden_dim,
                spec.kernel_size,
                spec.dilations[index % len(spec.dilations)] *
                (2 ** (index // len(spec.dilations))),
                spec.dropout,
            )
            for index in range(spec.depth)
        )

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = F.gelu(self.stem(tokens.transpose(1, 2))) * mask[:, None, :]
        for block in self.blocks:
            values = block(values, mask)
        return values.transpose(1, 2), mask


def _compress_one_sequence(values: torch.Tensor, maximum_tokens: int) -> torch.Tensor:
    if values.shape[0] <= maximum_tokens:
        return values
    boundaries = torch.linspace(0, values.shape[0], maximum_tokens + 1,
                                device=values.device).floor().to(torch.int64)
    chunks = [values[boundaries[index]:boundaries[index + 1]].mean(dim=0)
              for index in range(maximum_tokens)]
    return torch.stack(chunks, dim=0)


def _compress_masked_context(values: torch.Tensor, mask: torch.Tensor,
                             maximum_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress ordered valid loci so attention never exceeds a fixed quadratic budget."""
    rows = [_compress_one_sequence(values[row, mask[row]], maximum_tokens)
            for row in range(values.shape[0])]
    lengths = [max(1, row.shape[0]) for row in rows]
    compressed = values.new_zeros((values.shape[0], max(lengths), values.shape[2]))
    compressed_mask = torch.zeros((values.shape[0], max(lengths)), dtype=torch.bool,
                                  device=values.device)
    for index, row in enumerate(rows):
        if row.shape[0]:
            compressed[index, :row.shape[0]] = row
            compressed_mask[index, :row.shape[0]] = True
    return compressed, compressed_mask


def _sinusoidal_positions(mask: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    ordinal = (mask.to(torch.int64).cumsum(dim=1) - 1).clamp_min(0).to(dtype)
    even_width = (width + 1) // 2
    scale = torch.exp(
        torch.arange(even_width, device=mask.device, dtype=dtype) *
        (-math.log(10000.0) / max(1, even_width - 1))
    )
    angles = ordinal.unsqueeze(-1) * scale
    encoding = torch.zeros((*mask.shape, width), device=mask.device, dtype=dtype)
    encoding[..., 0::2] = torch.sin(angles[..., :encoding[..., 0::2].shape[-1]])
    encoding[..., 1::2] = torch.cos(angles[..., :encoding[..., 1::2].shape[-1]])
    return encoding * mask.unsqueeze(-1)


class TransformerSmall(ResidualLAIModel):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec, encoded_dim=spec.hidden_dim, head_hidden=spec.hidden_dim)
        self.input_projection = nn.Linear(spec.channels, spec.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.hidden_dim,
            nhead=spec.transformer_heads,
            dim_feedforward=spec.transformer_ff_dim,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=spec.depth, enable_nested_tensor=False)

    def encode(self, tokens: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        compressed, compressed_mask = _compress_masked_context(
            tokens, mask, self.spec.transformer_max_tokens)
        values = self.input_projection(compressed)
        values = values + _sinusoidal_positions(compressed_mask, values.shape[-1], values.dtype)
        # Multi-head attention requires at least one unmasked key. Empty rows use a private
        # zero token and are restored to an empty representation immediately afterwards.
        attention_mask = compressed_mask.clone()
        empty = ~attention_mask.any(dim=1)
        attention_mask[empty, 0] = True
        values = values * attention_mask.unsqueeze(-1)
        values = self.transformer(values, src_key_padding_mask=~attention_mask)
        values = values * compressed_mask.unsqueeze(-1)
        return values, compressed_mask


def build_model(spec: ModelSpec) -> ResidualLAIModel:
    """Build one deterministic architecture without altering the caller's RNG state."""
    builders = {
        "local_linear": LocalLinear,
        "residual_cnn_1d": ResidualCNN1D,
        "lainns_cnn_1d": LAiNNSCNN1D,
        "unet_1d": UNet1D,
        "bilstm": BiLSTM,
        "tcn": TCN,
        "transformer_small": TransformerSmall,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.seed)
        return builders[spec.family](spec)


def parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    parameters: Sequence[torch.nn.Parameter] = tuple(model.parameters())
    if trainable_only:
        parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def assert_simplex(probabilities: torch.Tensor, tolerance: float = 1e-5) -> None:
    _require(torch.isfinite(probabilities).all().item(), "prediction contains non-finite values")
    _require(torch.all(probabilities >= 0).item(), "prediction contains negative values")
    error = torch.max(torch.abs(probabilities.sum(dim=-1) - 1.0)).item()
    _require(error <= tolerance, "prediction does not satisfy the simplex")
