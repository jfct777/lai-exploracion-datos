"""Experimental multichannel FLARE adapter at supplied rare anchors, not dense LAI.

The common-only OrderedLAIModel encoder is reused; its historical classification
head is not called. SUMMARY and DETAIL share one diploid query dosage, and BOTH
learns their joint fusion. Refit ablations with matched inputs/splits/budgets;
switching modes on one fitted model is only a diagnostic, not that experiment.

No truth, source IDs, retrieval, smoothing, training or HPO is implemented here.
Neighboring events at arbitrary output positions require a new materializer and
event-axis contract. Reference dosages attached to common haplotype candidates
remain diploid, even when the same reference person occurs in several slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from m39_ordered_models import OrderedLAIModel


MODES = ("NONE", "SUMMARY", "DETAIL", "BOTH", "OFF")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class MultichannelConfig:
    """Explicit architectural settings; no scientific default or search policy."""

    hidden_width: int
    initial_gate: float

    def __post_init__(self) -> None:
        _require(type(self.hidden_width) is int and self.hidden_width > 0,
                 "hidden_width must be a positive integer")
        _require(type(self.initial_gate) in (int, float)
                 and math.isfinite(self.initial_gate) and 0 < self.initial_gate < 1,
                 "initial_gate must be finite and strictly between zero and one")


def _symmetric(value: Tensor) -> Tensor:
    left, right = value.flatten(start_dim=2).unbind(dim=1)
    return torch.cat((left + right, (left - right).abs()), dim=-1)


def _pool(value: Tensor, mask: Tensor) -> Tensor:
    safe = torch.where(mask[..., None], value, torch.zeros_like(value))
    return safe.sum(dim=3) / mask.sum(dim=3).clamp_min(1)[..., None].to(value.dtype)


def _log_probability(probability: Tensor) -> Tensor:
    # Preserve positive subnormals and genuine zeros without log(0) backward.
    safe = torch.where(probability > 0, probability, torch.ones_like(probability))
    return safe.log().masked_fill(probability == 0, -torch.inf)


class OrderedMultichannelAdapter(nn.Module):
    """Common context + optional summary/detail, mixed with a fixed FLARE base.

    ``common`` has the existing [B,2,3,K,L,5] encoder contract. ``baseline`` is
    floating [B,6] in STATE_NAMES order and is detached and copied, never changed.

    ``rare`` keys, read only when their branch is enabled:
      shared: query_dosage [B], query_observed bool [B];
      SUMMARY: ref_dosage_counts int64 [B,3,3] (REF persons by dosage 0/1/2),
               ref_eligible int64 [B,3] (all eligible unique REF persons);
      DETAIL: reference_dosage [B,2,3,K], reference_observed bool [B,2,3,K].

    Counts include the full eligible REF panel, not candidate occurrences.
    Observed counts, AF, callability and summary masks are derived, not supplied
    redundantly. Counts are support descriptors, not independent sample sizes.
    Missing dosage values are ignored; observed zeros are real information.

    NONE is a common-context correction; OFF is exact detached baseline without
    consulting common/rare. Absent rare support equals NONE for the same weights.
    Absent common support equals baseline. Positive initial gate permits learning
    and is deliberately NOT claimed to initialize an exact identity map.
    """

    def __init__(self, encoder: OrderedLAIModel, config: MultichannelConfig):
        super().__init__()
        _require(isinstance(encoder, OrderedLAIModel), "encoder must be OrderedLAIModel")
        self.encoder, self.config = encoder, config
        width, hidden = encoder.config.width, config.hidden_width
        self.common_projection = nn.Sequential(nn.Linear(6 * width + 6, hidden), nn.GELU())
        self.summary_projection = nn.Sequential(nn.Linear(17, hidden), nn.GELU())
        self.detail_candidate = nn.Sequential(nn.Linear(width + 4, hidden), nn.GELU(),
                                              nn.Linear(hidden, hidden), nn.GELU())
        self.detail_projection = nn.Sequential(nn.Linear(6 * hidden + 6, hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(3 * hidden + 6, hidden), nn.GELU())
        self.expert_head = nn.Linear(hidden, 6)
        self.gate_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, math.log(config.initial_gate / (1 - config.initial_gate)))
        self.to(device=encoder.input_projection.weight.device,
                dtype=encoder.input_projection.weight.dtype)

    def _field(self, rare: Mapping[str, Tensor], name: str, shape: tuple[int, ...],
               exemplar: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        _require(name in rare and isinstance(rare[name], Tensor), f"missing tensor {name}")
        value = rare[name]
        _require(tuple(value.shape) == shape, f"invalid shape for {name}")
        _require(value.device == exemplar.device, f"device mismatch for {name}")
        if dtype is not None:
            _require(value.dtype == dtype, f"invalid dtype for {name}")
        else:
            _require(value.is_floating_point() or value.dtype in (torch.int8, torch.int16,
                     torch.int32, torch.int64, torch.uint8), f"invalid dosage dtype for {name}")
        return value

    @staticmethod
    def _dosage(value: Tensor, observed: Tensor, exemplar: Tensor, name: str) -> Tensor:
        safe = torch.where(observed, value, torch.zeros_like(value))
        _require(bool(torch.isfinite(safe).all()) and bool(((safe >= 0) & (safe <= 2)).all())
                 and bool((safe == safe.round()).all()), f"invalid observed diploid {name}")
        return torch.stack((safe.to(exemplar.dtype) / 2, observed.to(exemplar.dtype)), dim=-1)

    def _summary(self, rare: Mapping[str, Tensor], query: Tensor, query_observed: Tensor,
                 exemplar: Tensor) -> tuple[Tensor, Tensor]:
        count = len(exemplar)
        counts = self._field(rare, "ref_dosage_counts", (count, 3, 3), exemplar, dtype=torch.int64)
        eligible = self._field(rare, "ref_eligible", (count, 3), exemplar, dtype=torch.int64)
        _require(bool((counts >= 0).all()) and bool((eligible >= 0).all()), "negative reference count")
        # Subtract from a nonnegative budget: no integer overflow or rounded
        # float comparison can conceal that the counts exceed eligible persons.
        remaining = eligible.clone()
        for dosage_count in counts.unbind(dim=-1):
            _require(bool((dosage_count <= remaining).all()), "observed count exceeds eligible panel")
            remaining = remaining - dosage_count
        observed = (eligible - remaining).double()
        available = observed > 0
        distribution = counts.double() / observed.clamp_min(1)[..., None]
        callability = observed / eligible.double().clamp_min(1)
        features = torch.cat((distribution, callability[..., None], observed.log1p()[..., None]), -1)
        features = torch.where(available[..., None], features, torch.zeros_like(features)).to(exemplar.dtype)
        summary = self.summary_projection(torch.cat((features.flatten(1), query), -1))
        supported = available.any(dim=-1) & query_observed
        return torch.where(supported[:, None], summary, torch.zeros_like(summary)), supported

    def _detail(self, rare: Mapping[str, Tensor], query: Tensor, query_observed: Tensor,
                encoded: Tensor, candidates: Tensor) -> tuple[Tensor, Tensor]:
        shape = tuple(candidates.shape)
        dose = self._field(rare, "reference_dosage", shape, encoded)
        observed = self._field(rare, "reference_observed", shape, encoded, dtype=torch.bool)
        active = candidates & observed & query_observed[:, None, None, None]
        reference = self._dosage(dose, active, encoded, "reference dosage")
        query_values = query[:, None, None, None, :].expand(*shape, 2)
        values = self.detail_candidate(torch.cat((encoded, reference, query_values), -1))
        grouped = _pool(values, active)
        fraction = active.sum(dim=3).to(encoded.dtype) / candidates.sum(dim=3).clamp_min(1)
        detail = self.detail_projection(torch.cat((_symmetric(grouped),
                                                   _symmetric(fraction[..., None])), -1))
        supported = active.flatten(1).any(dim=-1)
        return torch.where(supported[:, None], detail, torch.zeros_like(detail)), supported

    def forward(self, common: Mapping[str, Tensor], baseline: Tensor,
                rare: Mapping[str, Tensor] | None = None, *, mode: str,
                chunked: bool = True) -> dict[str, Tensor]:
        _require(mode in MODES, "unknown multichannel mode")
        _require(isinstance(baseline, Tensor) and baseline.ndim == 2 and baseline.shape[1] == 6
                 and len(baseline) > 0 and baseline.dtype in (torch.float32, torch.float64),
                 "baseline must be float32/float64 [B,6]")
        _require(bool(torch.isfinite(baseline).all()) and bool((baseline >= 0).all())
                 and bool((baseline <= 1).all()), "baseline must contain finite probabilities")
        _require(bool(torch.allclose(baseline.sum(-1), torch.ones_like(baseline[:, 0]),
                                    atol=8 * torch.finfo(baseline.dtype).eps, rtol=0)),
                 "baseline rows must sum to one")
        base = baseline.detach().clone()
        log_base = _log_probability(base)
        empty = torch.zeros(len(base), device=base.device, dtype=torch.bool)
        if mode == "OFF":
            # No encoder call, rare access, dtype conversion, normalization or floor.
            return {"probabilities": base, "log_probabilities": log_base,
                    "gate": base.new_zeros((len(base), 1)), "expert_probabilities": base.clone(),
                    "common_support": empty, "summary_support": empty.clone(),
                    "detail_support": empty.clone()}
        weight = self.encoder.input_projection.weight
        _require(base.device == weight.device and base.dtype == weight.dtype,
                 "baseline/model dtype or device mismatch")
        encoded = self.encoder.encode_candidates(common, chunked=chunked)
        _require(len(encoded) == len(base), "common/baseline batch mismatch")
        candidates = common["candidate_mask"] & common["site_mask"].any(-1)[:, None, None, None]
        support = candidates.flatten(1).any(-1)
        # Describe usable candidate occurrences, not the variable padded width K.
        # This count is not a number of independent reference individuals.
        candidate_count = candidates.sum(dim=3).to(encoded.dtype).log1p()
        common_value = self.common_projection(torch.cat((_symmetric(_pool(encoded, candidates)),
                                                         _symmetric(candidate_count[..., None])), -1))
        summary = torch.zeros_like(common_value)
        detail = torch.zeros_like(common_value)
        summary_support, detail_support = empty, empty.clone()
        if mode != "NONE":
            _require(isinstance(rare, Mapping), "enabled rare mode requires rare mapping")
            query_observed = self._field(rare, "query_observed", (len(base),), base, dtype=torch.bool)
            query_dose = self._field(rare, "query_dosage", (len(base),), base)
            query = self._dosage(query_dose, query_observed, base, "query dosage")
            if mode in ("SUMMARY", "BOTH"):
                summary, summary_support = self._summary(rare, query, query_observed, base)
                summary_support = summary_support & support
                summary = torch.where(support[:, None], summary, torch.zeros_like(summary))
            if mode in ("DETAIL", "BOTH"):
                detail, detail_support = self._detail(rare, query, query_observed, encoded, candidates)
        fused = self.fusion(torch.cat((common_value, summary, detail, base), -1))
        logits, raw_gate = self.expert_head(fused), self.gate_head(fused)
        log_expert = F.log_softmax(logits, dim=-1)
        expert = log_expert.exp()
        gate = raw_gate.sigmoid() * support[:, None].to(base.dtype)
        # Both mixture operands are evaluated safely even for unsupported rows.
        mixed_log = torch.logaddexp(F.logsigmoid(-raw_gate) + log_base,
                                   F.logsigmoid(raw_gate) + log_expert)
        log_probability = torch.where(support[:, None], mixed_log, log_base)
        # Derive both public views from one calculation. In float32 sigmoid(20)
        # rounds to one, but the remaining baseline mass is still representable;
        # computing 1 - sigmoid separately would incorrectly discard that mass.
        probability = torch.where(support[:, None], mixed_log.exp(), base)
        return {"probabilities": probability, "log_probabilities": log_probability,
                "gate": gate, "expert_probabilities": expert, "common_support": support,
                "summary_support": summary_support, "detail_support": detail_support}
