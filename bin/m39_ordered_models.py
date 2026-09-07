"""Local common-sequence encoders with a symmetric diploid six-state head.

Candidate lanes are exchangeable within each ancestry and query homolog. Their
indices are not positional identities across anchors. Rare dosages are diploid.
The input is one bounded batch of anchors, never a person-by-anchor corpus.
"""

from dataclasses import dataclass
from functools import partial
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")
ARMS = ("common", "real", "pooled")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class OrderedModelConfig:
    family: str = "cnn"
    width: int = 32
    depth: int = 2
    kernels: tuple[int, ...] = (3, 7)
    dilations: tuple[int, ...] = (1, 2)
    heads: int = 4
    attention_radius_tokens: int = 4
    dropout: float = 0.0
    core_sites: int = 256
    checkpoint_chunks: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernels", tuple(self.kernels))
        object.__setattr__(self, "dilations", tuple(self.dilations))
        _require(self.family in ("cnn", "attention"), "unknown encoder family")
        for key in ("width", "depth", "heads", "core_sites"):
            value = getattr(self, key)
            _require(type(value) is int and value > 0, f"{key} must be positive integer")
        _require(bool(self.kernels), "kernels cannot be empty")
        _require(all(type(k) is int and k > 0 and k % 2 for k in self.kernels),
                 "kernels must be positive odd integers")
        _require(len(set(self.kernels)) == len(self.kernels), "duplicate kernels")
        _require(len(self.dilations) == self.depth, "one dilation is required per block")
        _require(all(type(d) is int and d > 0 for d in self.dilations),
                 "dilations must be positive integers")
        _require(type(self.attention_radius_tokens) is int and self.attention_radius_tokens >= 0,
                 "attention radius must be a nonnegative integer")
        _require(self.family != "attention" or self.width % self.heads == 0,
                 "attention width must be divisible by heads")
        _require(0 <= self.dropout < 1, "dropout must be in [0,1)")
        _require(type(self.checkpoint_chunks) is bool, "checkpoint_chunks must be boolean")

    @property
    def receptive_radius_tokens(self) -> int:
        if self.family == "attention":
            return self.depth * self.attention_radius_tokens
        return (max(self.kernels) // 2) * sum(self.dilations)


def _masked_tokens(value: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))


class _TokenFeedForward(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2 * width, width),
                                    nn.Dropout(dropout))

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        return _masked_tokens(value + self.layers(self.norm(value)), valid)


class MultiscaleConvBlock(nn.Module):
    """Parallel stride-one convolutions followed by a tokenwise residual MLP."""

    def __init__(self, width: int, kernels: tuple[int, ...], dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.branches = nn.ModuleList([
            nn.Conv1d(width, width, kernel, padding=dilation * (kernel // 2),
                      dilation=dilation) for kernel in kernels
        ])
        self.dropout = nn.Dropout(dropout)
        self.feedforward = _TokenFeedForward(width, dropout)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        normalized = _masked_tokens(self.norm(value), valid).transpose(1, 2)
        mixed = None
        for branch in self.branches:
            current = F.gelu(branch(normalized).transpose(1, 2))
            mixed = current if mixed is None else mixed + current
        value = _masked_tokens(value + self.dropout(mixed / len(self.branches)), valid)
        return self.feedforward(value, valid)


class LocalAttentionBlock(nn.Module):
    """Sliding-band attention; no global L-by-L score tensor is constructed."""

    def __init__(self, width: int, heads: int, radius: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.radius = radius
        self.head_width = width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(width, width)
        self.output_dropout = nn.Dropout(dropout)
        self.feedforward = _TokenFeedForward(width, dropout)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        count, length, width = value.shape
        normalized = _masked_tokens(self.norm(value), valid)
        query, key, val = self.qkv(normalized).reshape(
            count, length, 3, self.heads, self.head_width
        ).unbind(dim=2)
        query, key, val = (part.transpose(1, 2) for part in (query, key, val))
        band_width = 2 * self.radius + 1
        key_band = F.pad(key, (0, 0, self.radius, self.radius)).unfold(
            2, band_width, 1
        ).movedim(-1, -2)
        val_band = F.pad(val, (0, 0, self.radius, self.radius)).unfold(
            2, band_width, 1
        ).movedim(-1, -2)
        key_mask = F.pad(valid, (self.radius, self.radius), value=False).unfold(
            1, band_width, 1
        ).unsqueeze(1)
        scores = torch.einsum("nhtd,nhtwd->nhtw", query, key_band) / self.head_width ** 0.5
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = self.softmax(scores)
        # Fully padded lanes get zero weights without an all-minus-infinity softmax.
        weights = self.attention_dropout(weights.masked_fill(~key_mask, 0))
        attended = torch.einsum("nhtw,nhtwd->nhtd", weights, val_band)
        attended = attended.transpose(1, 2).reshape(count, length, width)
        value = _masked_tokens(value + self.output_dropout(self.projection(attended)), valid)
        return self.feedforward(value, valid)


class SymmetricDiploidHead(nn.Module):
    """Shared candidate MLP, masked K mean, homolog-symmetric six-state logits."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(),
                                       nn.Linear(width, width), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(6 * width + 2, 2 * width), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(2 * width, 6))

    def forward(self, encoded: Tensor, candidate_mask: Tensor,
                reference_rare: Tensor, query_rare: Tensor) -> Tensor:
        candidate = self.candidate(torch.cat((encoded, reference_rare), dim=-1))
        candidate = _masked_tokens(candidate, candidate_mask)
        denominator = candidate_mask.sum(dim=3, keepdim=True).clamp_min(1)
        grouped = candidate.sum(dim=3) / denominator.to(candidate.dtype)
        left, right = grouped.flatten(start_dim=2).unbind(dim=1)
        symmetric = torch.cat((left + right, (left - right).abs(), query_rare), dim=-1)
        return self.classifier(symmetric)


class OrderedLAIModel(nn.Module):
    """Predict an unordered ancestry state independently for each supplied anchor.

    Chunking partitions all sites, includes the complete stacked receptive-field
    halo, and pools only encoded core outputs. Checkpoints limit retained internal
    activations, not the dense input batch or the number of chunk graph nodes.
    """

    def __init__(self, config: OrderedModelConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(7, config.width)
        if config.family == "cnn":
            self.blocks = nn.ModuleList([
                MultiscaleConvBlock(config.width, config.kernels, dilation, config.dropout)
                for dilation in config.dilations
            ])
        else:
            self.blocks = nn.ModuleList([
                LocalAttentionBlock(config.width, config.heads,
                                    config.attention_radius_tokens, config.dropout)
                for _ in range(config.depth)
            ])
        self.output_norm = nn.LayerNorm(config.width)
        self.head = SymmetricDiploidHead(config.width, config.dropout)

    @property
    def receptive_radius_tokens(self) -> int:
        return self.config.receptive_radius_tokens

    def _validate_common(self, batch: Mapping[str, Tensor]) -> tuple[int, int, int]:
        required = ("channels", "delta_cm", "site_mask", "candidate_mask", "radius_cm")
        for key in required:
            _require(key in batch and isinstance(batch[key], Tensor), f"missing tensor {key}")
        channels = batch["channels"]
        _require(channels.ndim == 6, "channels must have shape [B,2,3,K,L,5]")
        count, homologs, ancestries, candidates, length, features = channels.shape
        _require(count > 0 and candidates > 0, "batch and candidate counts must be positive")
        _require((homologs, ancestries, features) == (2, 3, 5), "invalid common axes")
        _require(channels.is_floating_point(), "channels must have floating dtype")
        _require(channels.device == self.input_projection.weight.device and
                 channels.dtype == self.input_projection.weight.dtype, "model/input dtype or device mismatch")
        shapes = {"delta_cm": (count, length), "site_mask": (count, length),
                  "candidate_mask": (count, 2, 3, candidates), "radius_cm": (count,)}
        for key, shape in shapes.items():
            _require(tuple(batch[key].shape) == shape, f"invalid shape for {key}")
            _require(batch[key].device == channels.device, f"device mismatch for {key}")
        for key in ("site_mask", "candidate_mask"):
            _require(batch[key].dtype == torch.bool, f"{key} must be boolean")
        _require(bool(torch.isfinite(batch["radius_cm"]).all()) and
                 bool((batch["radius_cm"] > 0).all()), "radius_cm must be finite and positive")
        site_mask = batch["site_mask"]
        valid = site_mask[:, None, None, None, :, None] & batch["candidate_mask"][..., None, None]
        safe = torch.where(valid, channels, torch.zeros_like(channels))
        _require(bool(torch.isfinite(safe).all()), "nonfinite common channel at a valid token")
        _require(bool(((safe >= 0) & (safe <= 1)).all()), "common channels must be in [0,1]")
        delta = torch.where(site_mask, batch["delta_cm"], torch.zeros_like(batch["delta_cm"]))
        _require(bool(torch.isfinite(delta).all()), "nonfinite delta_cm at a valid site")
        tolerance = 8 * torch.finfo(channels.dtype).eps
        _require(bool((delta.abs() <= batch["radius_cm"][:, None] * (1 + tolerance)).all()),
                 "common position outside retrieval radius")
        for row, mask in zip(delta, site_mask):
            observed = row[mask]
            _require(bool((observed[1:] >= observed[:-1]).all()), "common positions are not ordered")
        return count, candidates, length

    def _encode_core(self, channels: Tensor, delta_cm: Tensor, site_mask: Tensor,
                     candidate_mask: Tensor, radius_cm: Tensor, *,
                     core_start: int, core_stop: int) -> Tensor:
        count, _, _, candidates, length, _ = channels.shape
        valid = site_mask[:, None, None, None, :] & candidate_mask[..., None]
        common = _masked_tokens(channels, valid)
        delta = torch.where(site_mask, delta_cm, torch.zeros_like(delta_cm))
        delta = delta.to(common.dtype) / radius_cm.to(common.dtype)[:, None]
        position = torch.stack((delta, delta.abs()), dim=-1)
        position = position[:, None, None, None, :, :].expand(count, 2, 3, candidates, length, 2)
        features = _masked_tokens(torch.cat((common, position), dim=-1), valid)
        valid = valid.reshape(-1, length)
        value = self.input_projection(features.reshape(-1, length, 7))
        value = _masked_tokens(value, valid)
        for block in self.blocks:
            value = block(value, valid)
        value = _masked_tokens(self.output_norm(value), valid)
        return value[:, core_start:core_stop].sum(dim=1).reshape(count, 2, 3, candidates, -1)

    def encode_candidates(self, batch: Mapping[str, Tensor], *, chunked: bool = True) -> Tensor:
        """Return common-only embeddings [B,2,3,K,width]; rare fields are not read."""
        count, candidates, length = self._validate_common(batch)
        _require(not (chunked and self.training and self.config.dropout > 0),
                 "chunked training requires dropout=0 for full-context gradient equivalence")
        if length == 0:
            return batch["channels"].new_zeros((count, 2, 3, candidates, self.config.width))
        core_size = self.config.core_sites if chunked else length
        halo = self.receptive_radius_tokens
        total = None
        for start in range(0, length, core_size):
            stop = min(length, start + core_size)
            low, high = max(0, start - halo), min(length, stop + halo)
            encode = partial(self._encode_core, core_start=start - low, core_stop=stop - low)
            arguments = (batch["channels"][..., low:high, :], batch["delta_cm"][:, low:high],
                         batch["site_mask"][:, low:high], batch["candidate_mask"], batch["radius_cm"])
            if chunked and self.config.checkpoint_chunks and torch.is_grad_enabled():
                current = checkpoint(encode, *arguments, use_reentrant=False, preserve_rng_state=True)
            else:
                current = encode(*arguments)
            total = current if total is None else total + current
        denominator = batch["site_mask"].sum(dim=1).clamp_min(1).to(total.dtype)
        return total / denominator[:, None, None, None, None]

    @staticmethod
    def _rare_tensor(batch: Mapping[str, Tensor], key: str, shape: tuple[int, ...],
                     exemplar: Tensor, *, boolean: bool = False) -> Tensor:
        _require(key in batch and isinstance(batch[key], Tensor), f"missing tensor {key}")
        value = batch[key]
        _require(tuple(value.shape) == shape, f"invalid shape for {key}")
        _require(value.device == exemplar.device, f"device mismatch for {key}")
        if boolean:
            _require(value.dtype == torch.bool, f"{key} must be boolean")
        return value

    @staticmethod
    def _observed_value(value: Tensor, observed: Tensor, *, maximum: float,
                        discrete: bool, name: str, dtype: torch.dtype) -> Tensor:
        clean = torch.where(observed, value, torch.zeros_like(value))
        _require(bool(torch.isfinite(clean).all()), f"nonfinite observed {name}")
        _require(bool(((clean >= 0) & (clean <= maximum)).all()), f"invalid observed {name}")
        if discrete:
            _require(bool((clean == clean.round()).all()), f"{name} must be diploid dosage 0,1,2")
        clean = clean.to(dtype) / maximum
        return torch.stack((clean, observed.to(dtype)), dim=-1)

    def _rare_features(self, batch: Mapping[str, Tensor], encoded: Tensor,
                       arm: str) -> tuple[Tensor, Tensor]:
        count, _, _, candidates, _ = encoded.shape
        shape = (count, 2, 3, candidates)
        if arm == "common":
            # Do not even inspect individual or pooled rare values/masks in this arm.
            return encoded.new_zeros((*shape, 2)), encoded.new_zeros((count, 2))
        query = self._rare_tensor(batch, "query_dosage", (count,), encoded)
        query_observed = self._rare_tensor(batch, "query_observed", (count,), encoded, boolean=True)
        query_features = self._observed_value(query, query_observed, maximum=2,
                                              discrete=True, name="query dosage", dtype=encoded.dtype)
        if arm == "real":
            reference = self._rare_tensor(batch, "reference_dosage", shape, encoded)
            observed = self._rare_tensor(batch, "reference_observed", shape, encoded, boolean=True)
            observed = observed & batch["candidate_mask"]
            reference_features = self._observed_value(reference, observed, maximum=2,
                                                       discrete=True, name="reference dosage", dtype=encoded.dtype)
        else:
            pooled = self._rare_tensor(batch, "pooled_af", (count, 3), encoded)
            observed = self._rare_tensor(batch, "pooled_observed", (count, 3), encoded, boolean=True)
            pooled_features = self._observed_value(pooled, observed, maximum=1,
                                                   discrete=False, name="pooled AF", dtype=encoded.dtype)
            reference_features = pooled_features[:, None, :, None, :].expand(*shape, 2)
            reference_features = _masked_tokens(reference_features, batch["candidate_mask"])
        return reference_features, query_features

    def forward(self, batch: Mapping[str, Tensor], arm: str = "common", *,
                chunked: bool = True) -> Tensor:
        _require(arm in ARMS, "unknown rare arm")
        encoded = self.encode_candidates(batch, chunked=chunked)
        reference, query = self._rare_features(batch, encoded, arm)
        return self.head(encoded, batch["candidate_mask"], reference, query)
