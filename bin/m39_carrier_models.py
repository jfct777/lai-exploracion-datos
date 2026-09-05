"""Compact diploid carrier-context models on a fixed, common-derived locus grid.

Rare genotypes remain diploid dosages. These discriminative models do not infer
or certify rare haplotype phase. The separate phase-pair helper integrates one
explicit reference-pair latent variable; it is not used as a network likelihood.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")
VARIANTS = ("gated_deepset", "carrier_cross_attention", "bilinear_context")
ARMS = ("common", "pooled", "carrier", "perturbed")
CORRECTION_HEADS = ("multiplicative", "probability_mixture")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def paired_reference_dosage_distribution(
    first_dosage: int,
    second_dosage: int,
    *,
    same_individual: bool,
    first_homolog: int = 0,
    second_homolog: int = 1,
) -> tuple[float, float, float]:
    """Return P(B_first + B_second = 0, 1, 2), not a posterior ancestry prior.

    A heterozygote has orientations (0,1)/(1,0) with equal probability. For
    candidates from the same individual, both alleles use the SAME orientation.
    Distinct individuals assume independent orientation; missing is rejected.
    """
    _require(first_dosage in (0, 1, 2) and second_dosage in (0, 1, 2),
             "reference dosages must be observed diploid 0/1/2")
    _require(first_homolog in (0, 1) and second_homolog in (0, 1),
             "common homolog index must be 0 or 1")
    _require(not same_individual or first_dosage == second_dosage,
             "the same individual cannot have two dosages at one locus")

    def orientations(dosage: int) -> tuple[tuple[int, int], ...]:
        return (((0, 1), (1, 0)) if dosage == 1 else ((dosage // 2,) * 2,))

    first = orientations(first_dosage)
    second = orientations(second_dosage)
    pairs = [(state, state) for state in first] if same_individual else [
        (left, right) for left in first for right in second
    ]
    probabilities = [0.0, 0.0, 0.0]
    for left, right in pairs:
        probabilities[left[first_homolog] + right[second_homolog]] += 1.0 / len(pairs)
    return tuple(probabilities)


def pooled_reference_summary(dosage: Tensor, observed: Tensor, ancestry: Tensor) -> Tensor:
    """Summarize a GLOBAL unique-person reference panel, never top-K candidates.

    Inputs are dosage/observed [reference_person, locus] and ancestry [person]
    coded AFR=0, EUR=1, NAM=2. The caller must deduplicate biological individuals
    upstream. Output [locus, ancestry, 4] contains observed genotype frequencies
    (0,1,2) and callable fraction. Empty ancestry has all-zero summary.
    """
    _require(dosage.ndim == 2 and observed.shape == dosage.shape,
             "global reference dosage and observed must be [person,locus]")
    _require(observed.dtype == torch.bool, "reference observed must be boolean")
    _require(ancestry.shape == (dosage.shape[0],), "reference ancestry axis differs")
    _require(bool(((ancestry >= 0) & (ancestry < 3) & (ancestry == ancestry.round())).all()),
             "reference ancestry must be integer 0/1/2")
    _check_dosage(dosage, observed, "global reference")
    safe = torch.where(observed, dosage, torch.zeros_like(dosage)).long()
    one_hot = F.one_hot(safe, 3).to(torch.get_default_dtype())
    summaries = []
    for state in range(3):
        selected = (ancestry == state)
        called = observed & selected[:, None]
        count = called.sum(dim=0)
        frequencies = (one_hot * called[..., None]).sum(dim=0) / count.clamp_min(1)[..., None]
        fraction = count / selected.sum().clamp_min(1)
        summaries.append(torch.cat((frequencies, fraction[:, None]), dim=-1))
    return torch.stack(summaries, dim=1)


def _check_dosage(dosage: Tensor, observed: Tensor, name: str) -> None:
    values = dosage[observed]
    _require(bool((torch.isfinite(values) & (values >= 0) & (values <= 2)
                   & (values == values.round())).all()),
             f"{name} observed dosage must be diploid 0/1/2")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    numerator = torch.where(mask[..., None], values, torch.zeros_like(values)).sum(dim=-2)
    return numerator / mask.sum(dim=-1, keepdim=True).clamp_min(1)


def _symmetrize(values: Tensor) -> Tensor:
    """[B,J,2,ancestry,features] -> symmetric [B,J,ancestry*features*3]."""
    first, second = values.unbind(dim=2)
    return torch.cat(((first + second) / 2, (first - second).abs(), first * second),
                     dim=-1).flatten(start_dim=2)


def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1) * mask
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)


class CarrierContextModel(nn.Module):
    """Three encoders with one genotype-symmetric six-state residual interface.

    Required common keys:
      common_context [B,J,2,3,K,F], candidate_mask [B,J,2,3,K],
      baseline [B,J,6], coords [J].
    Carrier/perturbed keys:
      ref_dosage and rare_ref_observed [B,J,2,3,K],
      query_dosage and query_observed [B,J].
    Pooled additionally requires pooled_summary [B,J,3,4] derived from the
    entire eligible unique-person reference panel. It never reads top-K dosages.

    coords are validated but not used to fit a positional label shortcut. Locus
    smoothing is intentionally outside this local evidence module. Candidate
    selection, masks and the locus grid must be common-derived upstream.

    The optional probability_mixture head requires an explicit positive six-state
    mixture_prior. Its supported predictions start near, not exactly at, baseline:
    each of two mixture gates starts at mixture_init. Missing rare information
    falls back to the common prediction; absent common support returns baseline.
    """

    def __init__(self, common_features: int, variant: str = "gated_deepset", *,
                 width: int = 32, heads: int = 2, rank: int = 8,
                 residual_bound: float = 4.0, correction_head: str = "multiplicative",
                 mixture_init: float = 0.01, mixture_prior: Tensor | None = None) -> None:
        super().__init__()
        _require(common_features > 0 and width > 0 and heads > 0 and rank > 0,
                 "feature count, width, heads and rank must be positive")
        _require(variant in VARIANTS, f"variant must be one of {VARIANTS}")
        _require(variant != "carrier_cross_attention" or width % heads == 0,
                 "cross-attention width must be divisible by heads")
        _require(residual_bound > 0, "residual_bound must be positive")
        _require(correction_head in CORRECTION_HEADS,
                 f"correction_head must be one of {CORRECTION_HEADS}")
        self.correction_head = correction_head
        if correction_head == "probability_mixture":
            _require(mixture_prior is not None, "probability_mixture requires explicit mixture_prior")
            _require(math.isfinite(mixture_init) and 0 < mixture_init < 1,
                     "mixture_init must be finite and strictly between zero and one")
            prior = torch.as_tensor(mixture_prior, dtype=torch.get_default_dtype(),
                                    device="cpu").detach().clone()
            _require(prior.shape == (6,) and bool(torch.isfinite(prior).all())
                     and bool((prior > 0).all())
                     and bool(torch.allclose(prior.sum(), prior.new_tensor(1.), atol=5e-6, rtol=0)),
                     "mixture_prior must be a strictly positive six-state simplex")
            self.register_buffer("mixture_log_prior", (prior / prior.sum()).log())
        self.common_features, self.variant = common_features, variant
        self.width, self.heads, self.residual_bound = width, heads, residual_bound
        self.common_encoder = nn.Sequential(nn.Linear(common_features, width), nn.GELU(),
                                            nn.Linear(width, width), nn.GELU())
        symmetric_size = 3 * 3 * (width + 1)
        self.common_head = nn.Sequential(nn.Linear(symmetric_size, width), nn.GELU(),
                                         nn.Linear(width, 6))
        self.rare_head = nn.Sequential(nn.Linear(symmetric_size, width), nn.GELU(),
                                       nn.Linear(width, 6))
        self.common_gate = nn.Linear(symmetric_size, 1)
        self.rare_gate = nn.Linear(symmetric_size, 1)
        # Pooled and carrier arms share every trainable layer; only values change.
        if variant == "gated_deepset":
            self.context_projection = nn.Linear(common_features, width)
            self.value_projection = nn.Linear(15, width, bias=False)
            self.token_gate = nn.Linear(common_features, width)
        elif variant == "carrier_cross_attention":
            self.key_projection = nn.Linear(common_features, width)
            self.query_projection = nn.Linear(width, width)
            self.value_projection = nn.Linear(15, width, bias=False)
            self.cross_output = nn.Linear(width, width, bias=False)
        else:
            self.context_projection = nn.Linear(common_features, rank)
            self.value_projection = nn.Linear(15, rank, bias=False)
            self.bilinear_output = nn.Linear(rank, width, bias=False)
        # Initialization adds zero residual; no ancestral prior is replaced.
        nn.init.zeros_(self.common_head[-1].weight)
        nn.init.zeros_(self.common_head[-1].bias)
        nn.init.zeros_(self.rare_head[-1].weight)
        nn.init.zeros_(self.rare_head[-1].bias)
        if correction_head == "probability_mixture":
            for gate in (self.common_gate, self.rare_gate):
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, math.log(mixture_init / (1 - mixture_init)))

    def _common(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        context, mask, baseline, coords = (
            batch[key] for key in ("common_context", "candidate_mask", "baseline", "coords")
        )
        _require(context.ndim == 6 and context.shape[2:4] == (2, 3)
                 and context.shape[-1] == self.common_features and context.shape[-2] > 0,
                 "common_context must be [B,J,2,3,K,F] with K>0")
        _require(mask.shape == context.shape[:-1] and mask.dtype == torch.bool,
                 "candidate_mask shape/dtype differs")
        _require(baseline.shape == (*context.shape[:2], 6), "baseline must be [B,J,6]")
        _require(coords.shape == (context.shape[1],) and bool(torch.isfinite(coords).all())
                 and bool((coords[1:] >= coords[:-1]).all()), "coords must be finite ordered [J]")
        _require(bool(torch.isfinite(context[mask]).all()), "observed common features must be finite")
        _require(bool(torch.isfinite(baseline).all()) and bool((baseline >= 0).all())
                 and bool(torch.allclose(baseline.sum(-1), torch.ones_like(baseline[..., 0]),
                                         atol=5e-5, rtol=0)), "baseline must be a six-state simplex")
        clean = torch.where(mask[..., None], context, torch.zeros_like(context))
        common = _masked_mean(self.common_encoder(clean), mask)
        common = torch.cat((common, mask.to(context.dtype).mean(-1, keepdim=True)), dim=-1)
        return clean, mask, baseline, common

    def _query(self, batch: Mapping[str, Tensor], shape: torch.Size) -> tuple[Tensor, Tensor]:
        dosage, observed = batch["query_dosage"], batch["query_observed"]
        _require(dosage.shape == shape and observed.shape == shape and observed.dtype == torch.bool,
                 "query dosage/observed must be [B,J], with boolean observed")
        _check_dosage(dosage, observed, "query")
        safe = torch.where(observed, dosage, torch.zeros_like(dosage)).long()
        return F.one_hot(safe, 3), observed

    def _carrier(self, context: Tensor, mask: Tensor, common: Tensor,
                 batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        dosage, observed = batch["ref_dosage"], batch["rare_ref_observed"]
        _require(dosage.shape == mask.shape and observed.shape == mask.shape
                 and observed.dtype == torch.bool, "reference dosage/observed axes differ")
        active = observed & mask
        _check_dosage(dosage, active, "reference")
        query, query_observed = self._query(batch, context.shape[:2])
        safe = torch.where(active, dosage, torch.zeros_like(dosage)).long()
        reference = F.one_hot(safe, 3).to(context.dtype)
        query = query.to(context.dtype)[:, :, None, None, None, :].expand_as(reference)
        interaction = (reference[..., :, None] * query[..., None, :]).flatten(start_dim=-2)
        values = torch.cat((reference, query, interaction), dim=-1)
        active = active & query_observed[:, :, None, None, None]
        values = torch.where(active[..., None], values, torch.zeros_like(values))
        return self._encode_values(context, mask, common, values, active.to(context.dtype))

    def _encode_values(self, context: Tensor, mask: Tensor, common: Tensor,
                       values: Tensor, call_fraction: Tensor) -> tuple[Tensor, Tensor]:
        if self.variant == "gated_deepset":
            projected = self.context_projection(context)
            tokens = (F.gelu(projected + self.value_projection(values)) - F.gelu(projected))
            tokens = tokens * torch.sigmoid(self.token_gate(context))
            encoded = _masked_mean(tokens, mask)
        elif self.variant == "bilinear_context":
            tokens = self.context_projection(context) * self.value_projection(values)
            encoded = self.bilinear_output(_masked_mean(tokens, mask))
        else:
            dim = self.width // self.heads
            keys = self.key_projection(context).reshape(*mask.shape, self.heads, dim)
            query_keys = self.query_projection(common[..., :-1]).reshape(
                *mask.shape[:-1], self.heads, dim)
            scores = (keys * query_keys.unsqueeze(-3)).sum(-1) / dim ** 0.5
            weights = _masked_softmax(scores.transpose(-1, -2), mask.unsqueeze(-2))
            projected = self.value_projection(values).reshape(*mask.shape, self.heads, dim)
            encoded = (weights.transpose(-1, -2)[..., None] * projected).sum(-3)
            encoded = self.cross_output(encoded.flatten(start_dim=-2))
        fraction = (call_fraction * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
        return torch.cat((encoded, fraction.to(context.dtype)), dim=-1), fraction.flatten(2).gt(0).any(-1)

    def _pooled(self, context: Tensor, mask: Tensor, common: Tensor,
                batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        _require("pooled_summary" in batch, "pooled requires a GLOBAL pooled_summary, not top-K means")
        summary = batch["pooled_summary"]
        _require(summary.shape == (*context.shape[:2], 3, 4), "pooled_summary must be [B,J,3,4]")
        _require(bool(torch.isfinite(summary).all()) and bool(((summary >= 0) & (summary <= 1)).all()),
                 "pooled frequencies/callability must be finite probabilities")
        called = summary[..., 3] > 0
        _require(bool(torch.allclose(summary[..., :3].sum(-1), called.to(summary.dtype),
                                    atol=5e-5, rtol=0)), "pooled genotype frequencies do not sum to call status")
        query, observed = self._query(batch, context.shape[:2])
        reference = summary[..., :3].to(context.dtype)[:, :, None, :, None, :].expand(*mask.shape, 3)
        query = query.to(context.dtype)[:, :, None, None, None, :].expand_as(reference)
        interaction = (reference[..., :, None] * query[..., None, :]).flatten(start_dim=-2)
        values = torch.cat((reference, query, interaction), dim=-1)
        call_fraction = (summary[..., 3].to(context.dtype) * observed[..., None])[
            :, :, None, :, None].expand_as(mask)
        # The GLOBAL summary is broadcast to candidates before the SAME encoder.
        # It contains no information about which common context carries the allele.
        values = values * call_fraction[..., None] * mask[..., None]
        return self._encode_values(context, mask, common, values, call_fraction)

    def _forward_mixture(self, batch: Mapping[str, Tensor], arm: str,
                         return_aux: bool) -> Tensor | dict[str, Tensor]:
        """Two convex corrections; no baseline log/floor or multiplicative bound.

        qC=softmax(log prior+hC); pC=(1-lambdaC)*p0+lambdaC*qC.
        qR=softmax(log prior+hC+hR); p=(1-lambdaR)*pC+lambdaR*qR.
        Gates depend on common features and observed support, never on truth.
        """
        context, mask, baseline, common = self._common(batch)
        common_features = _symmetrize(common)
        support = mask.flatten(2).any(-1, keepdim=True).to(context.dtype)
        common_gate = torch.sigmoid(self.common_gate(common_features)) * support
        common_logits = self.common_head(common_features)
        common_expert = torch.softmax(self.mixture_log_prior + common_logits, dim=-1)
        common_probability = (1 - common_gate) * baseline + common_gate * common_expert
        rare_gate = torch.zeros_like(common_gate)
        rare_expert = common_expert
        representation = torch.zeros_like(common_features)
        if arm != "common":
            if arm == "pooled":
                rare, available = self._pooled(context, mask, common, batch)
            else:
                rare, available = self._carrier(context, mask, common, batch)
            representation = _symmetrize(rare)
            rare_gate = torch.sigmoid(self.rare_gate(common_features)) * available[..., None]
            rare_expert = torch.softmax(
                self.mixture_log_prior + common_logits + self.rare_head(representation), dim=-1)
        probabilities = (1 - rare_gate) * common_probability + rare_gate * rare_expert
        if return_aux:
            return {"probabilities": probabilities, "common_probabilities": common_probability,
                    "common_expert_probabilities": common_expert,
                    "rare_expert_probabilities": rare_expert,
                    "common_gate": common_gate, "rare_gate": rare_gate,
                    "rare_representation": representation}
        return probabilities

    def forward(self, batch: Mapping[str, Tensor], arm: str = "carrier", *,
                return_aux: bool = False) -> Tensor | dict[str, Tensor]:
        _require(arm in ARMS, f"arm must be one of {ARMS}")
        if self.correction_head == "probability_mixture":
            return self._forward_mixture(batch, arm, return_aux)
        context, mask, baseline, common = self._common(batch)
        common_features = _symmetrize(common)
        support = mask.flatten(2).any(-1, keepdim=True).to(context.dtype)
        common_gate = torch.sigmoid(self.common_gate(common_features)) * support
        common_delta = self.residual_bound * torch.tanh(self.common_head(common_features)) * common_gate
        rare_delta = torch.zeros_like(common_delta)
        rare_gate = torch.zeros_like(common_gate)
        representation = torch.zeros_like(common_features)
        if arm != "common":
            if arm == "pooled":
                rare, available = self._pooled(context, mask, common, batch)
            else:
                # A perturbed arm supplies externally permuted values; no hidden RNG or label access.
                rare, available = self._carrier(context, mask, common, batch)
            representation = _symmetrize(rare)
            rare_gate = torch.sigmoid(self.rare_gate(common_features)) * available[..., None]
            rare_delta = self.residual_bound * torch.tanh(self.rare_head(representation)) * rare_gate
        delta = common_delta + rare_delta
        # A multiplicative residual preserves genuine zero-support baseline states.
        unnormalized = baseline * torch.exp(delta - delta.amax(-1, keepdim=True))
        probabilities = unnormalized / unnormalized.sum(-1, keepdim=True)
        if return_aux:
            return {"probabilities": probabilities, "common_delta": common_delta,
                    "rare_delta": rare_delta, "common_gate": common_gate, "rare_gate": rare_gate,
                    "rare_representation": representation}
        return probabilities
