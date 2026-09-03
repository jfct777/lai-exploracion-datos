#!/usr/bin/env python3
"""Model registry for the isolated M36 CORA-Set exploratory lane.

The registry records the two permitted permutation-invariant encoders and
imports PyTorch only in the explicit training runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CoraModelSpec:
    """Finite, matched-capacity specification for a CORA encoder."""

    family: str
    token_dim: int
    hidden_dim: int
    depth: int
    attention_heads: int = 0
    inducing_points: int = 0
    context_dim: int = 8


def available_specs(families: Iterable[str]) -> list[CoraModelSpec]:
    """Return the finite candidate registry, rejecting unreviewed families."""
    requested = tuple(families)
    unknown = set(requested) - {"deep_sets", "set_transformer"}
    if unknown:
        raise ValueError(f"Unsupported M36 model family: {sorted(unknown)}")
    specs: list[CoraModelSpec] = []
    if "deep_sets" in requested:
        specs.extend(
            [
                CoraModelSpec("deep_sets", 10, 32, 2),
                CoraModelSpec("deep_sets", 10, 64, 2),
            ]
        )
    if "set_transformer" in requested:
        specs.append(CoraModelSpec("set_transformer", 10, 32, 2, 4, 8))
    return specs


def build_encoder(spec: CoraModelSpec):
    """Build an encoder only in an authorised training environment.

    PyTorch is intentionally optional for local contract smokes.  The two
    encoders operate on numeric ``[batch, events, token_dim]``, FIT-vocabulary
    context IDs and a boolean event mask; they return one embedding per person.
    """
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:  # pragma: no cover - container concern
        raise RuntimeError("M36 training requires the pinned PyTorch container") from error

    class DeepSetsEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_embedding = nn.Embedding(1, spec.context_dim)
            layers: list[nn.Module] = [
                nn.Linear(spec.token_dim + spec.context_dim, spec.hidden_dim), nn.GELU()
            ]
            for _ in range(spec.depth - 1):
                layers.extend([nn.Linear(spec.hidden_dim, spec.hidden_dim), nn.GELU()])
            self.phi = nn.Sequential(*layers)
            self.rho = nn.Sequential(nn.Linear(spec.hidden_dim, spec.hidden_dim), nn.GELU())

        def set_context_size(self, n_contexts: int) -> None:
            self.context_embedding = nn.Embedding(n_contexts, spec.context_dim)

        def forward(self, tokens, contexts, mask):
            values = torch.cat((tokens, self.context_embedding(contexts)), dim=-1)
            encoded = self.phi(values) * mask.unsqueeze(-1)
            # Sum pooling keeps rare-event burden identifiable.  Mean pooling
            # erased the count/dosage signal in the synthetic additive control.
            # Padding is zeroed above, so this remains permutation invariant.
            return self.rho(encoded.sum(dim=1))

    class SetTransformerEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_embedding = nn.Embedding(1, spec.context_dim)
            self.input = nn.Linear(spec.token_dim + spec.context_dim, spec.hidden_dim)
            self.inducing = nn.Parameter(torch.empty(spec.inducing_points, spec.hidden_dim))
            nn.init.xavier_uniform_(self.inducing)
            self.attend_inducing = nn.MultiheadAttention(
                spec.hidden_dim, spec.attention_heads, batch_first=True
            )
            self.attend_tokens = nn.MultiheadAttention(
                spec.hidden_dim, spec.attention_heads, batch_first=True
            )
            self.norm = nn.LayerNorm(spec.hidden_dim)

        def set_context_size(self, n_contexts: int) -> None:
            self.context_embedding = nn.Embedding(n_contexts, spec.context_dim)

        def forward(self, tokens, contexts, mask):
            values = self.input(torch.cat((tokens, self.context_embedding(contexts)), dim=-1))
            batch = values.shape[0]
            inducing = self.inducing.unsqueeze(0).expand(batch, -1, -1)
            # MultiheadAttention cannot receive an all-padding row.  A zero
            # sentinel is attended internally then removed from the output.
            safe_mask = mask.clone()
            empty = ~safe_mask.any(dim=1)
            safe_mask[empty, 0] = True
            pooled, _ = self.attend_inducing(
                inducing, values, values, key_padding_mask=~safe_mask
            )
            contextual, _ = self.attend_tokens(
                values, pooled, pooled, need_weights=False
            )
            contextual = self.norm(contextual + values) * mask.unsqueeze(-1)
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
            return contextual.sum(dim=1) / denominator

    if spec.family == "deep_sets":
        return DeepSetsEncoder()
    if spec.family == "set_transformer":
        return SetTransformerEncoder()
    raise ValueError(f"Unsupported M36 model family: {spec.family}")


def build_pair_regressor(spec: CoraModelSpec, covariate_dim: int, n_contexts: int):
    """Return a symmetric continuous-IBD regressor with a covariate baseline.

    The residual sees only ``|z_i-z_j|`` and ``z_i*z_j``.  Thus pair order
    cannot leak into the prediction and the rare channel is explicitly an
    increment over the covariate-only baseline.
    """
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:  # pragma: no cover - container concern
        raise RuntimeError("M36 training requires the pinned PyTorch container") from error

    encoder = build_encoder(spec)
    encoder.set_context_size(n_contexts)

    class PairRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.baseline = nn.Sequential(
                nn.Linear(2 * covariate_dim, spec.hidden_dim), nn.GELU(), nn.Linear(spec.hidden_dim, 1)
            )
            self.residual = nn.Sequential(
                nn.Linear(2 * spec.hidden_dim, spec.hidden_dim), nn.GELU(), nn.Linear(spec.hidden_dim, 1)
            )

        @staticmethod
        def pair_covariates(left, right):
            return torch.cat((torch.abs(left - right), left * right), dim=-1)

        def forward(self, left_tokens, left_contexts, left_mask, right_tokens, right_contexts, right_mask,
                    left_covariates, right_covariates, include_rare=True):
            baseline = self.baseline(self.pair_covariates(left_covariates, right_covariates)).squeeze(-1)
            if not include_rare:
                return baseline, baseline
            left = self.encoder(left_tokens, left_contexts, left_mask)
            right = self.encoder(right_tokens, right_contexts, right_mask)
            pair = torch.cat((torch.abs(left - right), left * right), dim=-1)
            return baseline + self.residual(pair).squeeze(-1), baseline

    return PairRegressor()
