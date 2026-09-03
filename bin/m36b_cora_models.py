#!/usr/bin/env python3
"""Cached person-level encoders for the isolated M36B experiment.

Each person is encoded once per optimisation step.  Pair predictions then
reuse those embeddings in bounded pair batches, avoiding the repeated
person-by-pair encoding used by the original M36 screen.
"""

from __future__ import annotations

from m36_cora_models import CoraModelSpec, build_encoder


def build_cached_pair_regressor(spec: CoraModelSpec, covariate_dim: int, n_contexts: int):
    """Build a symmetric residual regressor with reusable person embeddings."""
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:  # pragma: no cover - container concern
        raise RuntimeError("M36B training requires the pinned PyTorch container") from error

    class CachedPairRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.baseline = nn.Sequential(
                nn.Linear(2 * covariate_dim, spec.hidden_dim),
                nn.GELU(),
                nn.Linear(spec.hidden_dim, 1),
            )
            self.residual = nn.Sequential(
                nn.Linear(2 * spec.hidden_dim, spec.hidden_dim),
                nn.GELU(),
                nn.Linear(spec.hidden_dim, 1),
            )
            # Construct the encoder last so both prediction heads receive
            # identical initial weights across matched control arms.
            self.encoder = build_encoder(spec)
            self.encoder.set_context_size(n_contexts)

        @staticmethod
        def symmetric_pair(left, right):
            return torch.cat((torch.abs(left - right), left * right), dim=-1)

        def encode_population(self, tokens, contexts, mask):
            return self.encoder(tokens, contexts, mask)

        def encode_shared_geometry(self, tokens, contexts, mask, n_people: int):
            if tokens.shape[0] != 1:
                raise ValueError("shared geometry must be encoded as one common event set")
            return self.encoder(tokens, contexts, mask).expand(n_people, -1)

        def predict_from_embeddings(self, embeddings, covariates, left, right, include_rare=True):
            baseline = self.baseline(
                self.symmetric_pair(covariates[left], covariates[right])
            ).squeeze(-1)
            if not include_rare:
                return baseline, baseline
            pair = self.symmetric_pair(embeddings[left], embeddings[right])
            return baseline + self.residual(pair).squeeze(-1), baseline

    return CachedPairRegressor()
