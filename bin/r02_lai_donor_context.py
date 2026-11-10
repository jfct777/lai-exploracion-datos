"""Autonomous donor-by-locus context for six-state diploid LAI.

Inputs are unphased ALT dosages and explicit masks.  The module neither reads
FLARE values nor assigns heterozygotes to haplotypes.  Donor transformations
share every parameter; reference-panel pooling occurs only after each donor's
loci have interacted and been read at the requested genetic-map positions.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from r02_lai_common_context import LocalQuerySourceAttention
from r02_local_attention_blocks import ContextExecution, candidate_keys


ANCESTRY_NAMES = ("AFR", "EUR", "NAM")
# Existing M39/evaluation order: A=AFR, E=EUR and N=NAM.  States are
# unordered diploid pairs, not phased homologues.
STATE_NAMES = ("AA", "AE", "AN", "EE", "EN", "NN")
PANEL_COUNT = len(ANCESTRY_NAMES)
JOINT_ANCESTRY_STATES = len(STATE_NAMES)
PAIR_FEATURES = 3 + 3 + 4 + 4 + 1 + PANEL_COUNT + 1
GROUP_SCALARS = 7
FINAL_HEAD_LINEAR = "linear"
FINAL_HEAD_FACTORIZED_LINEAR = "factorized_linear"
FINAL_HEAD_MLP = "mlp"
FINAL_HEAD_MODES = frozenset((
    FINAL_HEAD_LINEAR, FINAL_HEAD_FACTORIZED_LINEAR, FINAL_HEAD_MLP))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class DonorContextBatch:
    target_alt_dosage: torch.Tensor       # B,L
    reference_alt_dosage: torch.Tensor    # D,L
    target_observed: torch.Tensor         # B,L bool
    reference_observed: torch.Tensor      # D,L bool
    ref_alt_alleles: torch.Tensor         # L,2 integers A/C/G/T -> 0..3
    rare_catalog: torch.Tensor            # L bool, fixed before fitting
    locus_cm: torch.Tensor                # L
    locus_chrom: torch.Tensor             # L integer
    query_cm: torch.Tensor                # B,Q
    query_chrom: torch.Tensor             # B,Q integer
    query_valid: torch.Tensor             # B,Q bool
    reference_panel: torch.Tensor         # D integer 0..2
    donor_allowed: torch.Tensor           # B,D bool, resolved outside model


@dataclass(frozen=True)
class DonorContextOutput:
    logits: torch.Tensor
    support: torch.Tensor
    group_presence: torch.Tensor
    eligible_donor_count: torch.Tensor
    supported_donor_count: torch.Tensor
    observed_pair_locus_count: torch.Tensor
    genotype_match_count: torch.Tensor
    genotype_discrepancy_count: torch.Tensor
    learned_score_logmeanexp: torch.Tensor
    donor_query: torch.Tensor
    donor_local_support_count: torch.Tensor


@dataclass(frozen=True)
class DonorReadoutRepresentation:
    """Exact values immediately before the six-state final transformation.

    ``head_input`` has axes person, query and ``3 * (width + 7)``.  The
    remaining fields are the same support and audit quantities returned by
    :class:`DonorContextOutput`; keeping them beside the vector prevents a
    cache from silently replacing unsupported-but-valid queries with zeros.
    """
    head_input: torch.Tensor
    support: torch.Tensor
    group_presence: torch.Tensor
    eligible_donor_count: torch.Tensor
    supported_donor_count: torch.Tensor
    observed_pair_locus_count: torch.Tensor
    genotype_match_count: torch.Tensor
    genotype_discrepancy_count: torch.Tensor
    learned_score_logmeanexp: torch.Tensor
    donor_query: torch.Tensor
    donor_local_support_count: torch.Tensor


class _DonorLocusLayer(nn.Module):
    def __init__(self, width, heads, feed_forward_multiplier, execution):
        super().__init__()
        self.attention = LocalQuerySourceAttention(
            width, width, heads, dropout=0., execution=execution)
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feed_forward_multiplier * width), nn.GELU(),
            nn.Linear(feed_forward_multiplier * width, width))
        self.feed_forward_norm = nn.LayerNorm(width)

    def forward(self, values, cm, chrom, valid, radius_cm):
        attended, _ = self.attention(
            values, cm, chrom, valid, values, cm, chrom, valid, radius_cm)
        values = self.attention_norm(values + attended)
        values = torch.where(valid[..., None], values, torch.zeros_like(values))
        values = self.feed_forward_norm(values + self.feed_forward(values))
        return torch.where(valid[..., None], values, torch.zeros_like(values))


class _TwoLayerJointStateHead(nn.Module):
    """Two affine maps whose optional GELU is the sole mode difference."""
    def __init__(self, input_width, hidden_width, output_width, *, nonlinear):
        super().__init__()
        self.input_layer = nn.Linear(input_width, hidden_width)
        self.activation = nn.GELU() if nonlinear else nn.Identity()
        self.output_layer = nn.Linear(hidden_width, output_width)

    def forward(self, values):
        return self.output_layer(self.activation(self.input_layer(values)))


class AutonomousDonorContextLAI(nn.Module):
    """Relate loci within each donor, then pool donors by named reference panel.

    Width, depth, radii and all storage limits are required constructor values;
    synthetic-test dimensions are therefore not implicit scientific defaults.
    ``max_pair_state_elements`` bounds B*D*L*W and
    ``max_query_state_elements`` bounds B*D*Q*W before either state is built.
    ``execution.max_mask_elements`` separately bounds exact attention masks.
    """
    ancestry_names = ANCESTRY_NAMES
    state_names = STATE_NAMES

    def __init__(self, *, width: int, heads: int, depth: int,
                 feed_forward_multiplier: int,
                 locus_radius_cm: float, query_radius_cm: float,
                 execution: ContextExecution, max_pair_state_elements: int,
                 max_query_state_elements: int, donor_block_size: int | None = None,
                 final_head_mode: str = FINAL_HEAD_LINEAR,
                 final_head_hidden_width: int | None = None):
        super().__init__()
        _require(all(type(value) is int and value > 0
                     for value in (width, heads, depth, feed_forward_multiplier)),
                 "positive integer width, heads, depth and feed-forward multiplier required")
        _require(width % heads == 0, "heads must divide width")
        _require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                     for value in (locus_radius_cm, query_radius_cm)),
                 "positive finite cM radii required")
        _require(isinstance(execution, ContextExecution),
                 "validated ContextExecution required")
        _require(all(type(value) is int and value > 0 for value in
                     (max_pair_state_elements, max_query_state_elements)),
                 "positive integer state limits required")
        _require(donor_block_size is None or (type(donor_block_size) is int and donor_block_size > 0),
                 "donor block size must be a positive integer or None")
        _require(type(final_head_mode) is str and final_head_mode in FINAL_HEAD_MODES,
                 "final head mode must be linear, factorized_linear or mlp")
        if final_head_mode == FINAL_HEAD_LINEAR:
            _require(final_head_hidden_width is None,
                     "linear final head has no hidden width")
        else:
            _require(type(final_head_hidden_width) is int and final_head_hidden_width >=
                     JOINT_ANCESTRY_STATES,
                     "two-layer final head hidden width must be an integer at least 6")
        self.donor_block_size = donor_block_size
        self.width = width
        self.heads = heads
        self.depth = depth
        self.feed_forward_multiplier = feed_forward_multiplier
        self.locus_radius_cm = float(locus_radius_cm)
        self.query_radius_cm = float(query_radius_cm)
        self.execution = execution
        self.max_pair_state_elements = max_pair_state_elements
        self.max_query_state_elements = max_query_state_elements
        self.final_head_mode = final_head_mode
        self.final_head_hidden_width = final_head_hidden_width

        self.pair_projection = nn.Linear(PAIR_FEATURES, width)
        self.locus_layers = nn.ModuleList(
            [_DonorLocusLayer(width, heads, feed_forward_multiplier, execution)
             for _ in range(depth)])
        self.query_token = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query_token, std=width ** -.5)
        self.query_attention = LocalQuerySourceAttention(
            width, width, heads, dropout=0., execution=execution)
        self.donor_score = nn.Linear(width, 1)
        head_input_width = PANEL_COUNT * (width + GROUP_SCALARS)
        if final_head_mode == FINAL_HEAD_LINEAR:
            # Preserve historical state_dict names and RNG initialization exactly.
            self.joint_state_head = nn.Linear(head_input_width, JOINT_ANCESTRY_STATES)
        else:
            self.joint_state_head = _TwoLayerJointStateHead(
                head_input_width, final_head_hidden_width, JOINT_ANCESTRY_STATES,
                nonlinear=final_head_mode == FINAL_HEAD_MLP)

    @property
    def effective_radius_cm(self):
        """Largest genetic-map distance that can reach a prediction query."""
        return len(self.locus_layers) * self.locus_radius_cm + self.query_radius_cm

    @property
    def configuration(self):
        """JSON-serializable metadata required alongside a future state_dict."""
        return {
            "schema": ("r02_lai_donor_context_config_v1"
                       if self.final_head_mode == FINAL_HEAD_LINEAR
                       else "r02_lai_donor_context_config_v2"),
            "dimensions": {"pair_features": PAIR_FEATURES, "width": self.width,
                           "group_scalars": GROUP_SCALARS, "panels": PANEL_COUNT,
                           "joint_states": JOINT_ANCESTRY_STATES},
            "heads": self.heads, "depth": self.depth,
            "feed_forward_multiplier": self.feed_forward_multiplier,
            "activation": "GELU", "attention_dropout": 0.0,
            "locus_radius_cm": self.locus_radius_cm,
            "query_radius_cm": self.query_radius_cm,
            "effective_radius_cm": self.effective_radius_cm,
            **({"donor_block_size": self.donor_block_size}
               if self.donor_block_size is not None else {}),
            "context_execution": {
                "query_block_size": self.execution.query_block_size,
                "max_mask_elements": self.execution.max_mask_elements,
                "checkpoint_blocks": self.execution.checkpoint_blocks,
                "local_mean_mode": self.execution.local_mean_mode},
            "limits": {"max_pair_state_elements": self.max_pair_state_elements,
                       "max_query_state_elements": self.max_query_state_elements},
            "ancestry_order": list(self.ancestry_names),
            "state_order": list(self.state_names),
            **({"final_head": {
                "mode": self.final_head_mode,
                "input_width": PANEL_COUNT * (self.width + GROUP_SCALARS),
                "hidden_width": self.final_head_hidden_width,
                "output_width": JOINT_ANCESTRY_STATES,
                "activation": ("GELU" if self.final_head_mode == FINAL_HEAD_MLP
                               else "identity"),
            }} if self.final_head_mode != FINAL_HEAD_LINEAR else {}),
        }

    @staticmethod
    def _validate(batch):
        _require(isinstance(batch, DonorContextBatch), "DonorContextBatch required")
        t, r = batch.target_alt_dosage, batch.reference_alt_dosage
        _require(t.ndim == r.ndim == 2, "dosages require person/donor and locus axes")
        batch_size, loci = t.shape
        donors, reference_loci = r.shape
        _require(batch_size > 0 and donors > 0 and loci > 0 and reference_loci == loci,
                 "nonempty aligned target/reference loci required")
        _require(batch.target_observed.shape == t.shape
                 and batch.reference_observed.shape == r.shape,
                 "dosage and observation axes differ")
        _require(batch.ref_alt_alleles.shape == (loci, 2)
                 and batch.rare_catalog.shape == batch.locus_cm.shape ==
                 batch.locus_chrom.shape == (loci,), "locus metadata axes differ")
        _require(batch.query_cm.ndim == 2 and batch.query_cm.shape[0] == batch_size
                 and batch.query_chrom.shape == batch.query_valid.shape == batch.query_cm.shape,
                 "query axes differ")
        _require(batch.query_cm.shape[1] > 0 and batch.reference_panel.shape == (donors,)
                 and batch.donor_allowed.shape == (batch_size, donors),
                 "donor/query metadata axes differ")
        bools = (batch.target_observed, batch.reference_observed, batch.rare_catalog,
                 batch.query_valid, batch.donor_allowed)
        _require(all(value.dtype == torch.bool for value in bools), "masks must be boolean")
        integers = (batch.ref_alt_alleles, batch.locus_chrom,
                    batch.query_chrom, batch.reference_panel)
        _require(all(value.dtype in (torch.int32, torch.int64) for value in integers),
                 "alleles, chromosomes and panels must be integer")
        tensors = tuple(batch.__dict__.values())
        _require(all(value.device == t.device for value in tensors),
                 "all donor-context inputs must share one device")
        _require(t.is_floating_point() and r.is_floating_point()
                 and batch.locus_cm.is_floating_point() and batch.query_cm.is_floating_point(),
                 "dosages and genetic-map coordinates must be floating point")
        _require(bool(((batch.ref_alt_alleles >= 0) & (batch.ref_alt_alleles <= 3)).all()),
                 "REF/ALT alleles must use integers 0..3")
        _require(bool((batch.ref_alt_alleles[:, 0] != batch.ref_alt_alleles[:, 1]).all()),
                 "REF and ALT alleles must differ")
        _require(bool(((batch.reference_panel >= 0) &
                       (batch.reference_panel < PANEL_COUNT)).all()),
                 "reference panels must use integers 0..2")
        _require(bool(torch.isfinite(batch.query_cm[batch.query_valid]).all()),
                 "valid query coordinates must be finite")
        _require(bool((batch.query_chrom[batch.query_valid] > 0).all()),
                 "valid query chromosomes must be positive")
        return batch_size, donors, loci, batch.query_cm.shape[1]

    @staticmethod
    def _valid_dosages(values, name):
        _require(bool(torch.isfinite(values).all()), f"effective {name} dosage must be finite")
        _require(bool(((values >= 0) & (values <= 2) & (values == values.round())).all()),
                 f"effective {name} dosage must be an integer in 0..2")

    def _pair_values(self, batch, batch_size, donors, loci):
        pair_valid = (batch.target_observed[:, None, :]
                      & batch.reference_observed[None, :, :]
                      & batch.donor_allowed[:, :, None])
        effective_locus = pair_valid.any((0, 1))
        _require(bool(torch.isfinite(batch.locus_cm[effective_locus]).all()),
                 "effective locus coordinates must be finite")
        _require(bool((batch.locus_chrom[effective_locus] > 0).all()),
                 "effective locus chromosomes must be positive")
        target = batch.target_alt_dosage[:, None, :].expand(batch_size, donors, loci)
        reference = batch.reference_alt_dosage[None, :, :].expand(batch_size, donors, loci)
        self._valid_dosages(target[pair_valid], "target")
        self._valid_dosages(reference[pair_valid], "reference")
        clean_target = torch.where(pair_valid, target, torch.zeros_like(target)).long()
        clean_reference = torch.where(pair_valid, reference, torch.zeros_like(reference)).long()
        feature_dtype = self.pair_projection.weight.dtype
        panel = F.one_hot(batch.reference_panel.long(), PANEL_COUNT).to(feature_dtype)
        allele = F.one_hot(batch.ref_alt_alleles.long(), 4).to(feature_dtype).flatten(-2)
        features = torch.cat((
            F.one_hot(clean_target, 3).to(feature_dtype),
            F.one_hot(clean_reference, 3).to(feature_dtype),
            allele[None, None].expand(batch_size, donors, loci, 8),
            pair_valid[..., None].to(feature_dtype),
            panel[None, :, None].expand(batch_size, donors, loci, PANEL_COUNT),
            batch.rare_catalog[None, None, :, None].expand(batch_size, donors, loci, 1).to(feature_dtype),
        ), -1)
        features = torch.where(pair_valid[..., None], features, torch.zeros_like(features))
        projected = self.pair_projection(features)
        return (torch.where(pair_valid[..., None], projected, torch.zeros_like(projected)),
                pair_valid, clean_target, clean_reference)

    def _local_boolean_counts(self, values, query_cm, query_chrom, query_valid,
                              source_cm, source_chrom, source_valid):
        """Exact local sums with the same neighborhoods and mask budget as attention."""
        pieces = []
        for start in range(0, query_cm.shape[1], self.execution.query_block_size):
            end = min(start + self.execution.query_block_size, query_cm.shape[1])
            keys = candidate_keys(query_cm[:, start:end], query_chrom[:, start:end],
                                  query_valid[:, start:end], source_cm, source_chrom,
                                  source_valid, self.query_radius_cm)
            elements = query_cm.shape[0] * (end - start) * len(keys)
            _require(elements <= self.execution.max_mask_elements,
                     "donor-query count mask budget exceeded; no neighbors were truncated")
            allowed = (query_valid[:, start:end, None] & source_valid[:, None, keys]
                       & (query_chrom[:, start:end, None] == source_chrom[:, None, keys])
                       & ((query_cm[:, start:end, None] - source_cm[:, None, keys]).abs()
                          <= self.query_radius_cm))
            pieces.append((allowed & values[:, None, keys]).sum(-1))
        return torch.cat(pieces, 1)

    def _donor_queries(self, *tensors):
        """Compute independent donor representations before any panel reduction."""
        batch = DonorContextBatch(*tensors)
        batch_size, loci = batch.target_alt_dosage.shape
        donors = batch.reference_alt_dosage.shape[0]
        queries = batch.query_cm.shape[1]
        values, pair_valid, target, reference = self._pair_values(
            batch, batch_size, donors, loci)
        flat = batch_size * donors
        values = values.reshape(flat, loci, self.width)
        valid = pair_valid.reshape(flat, loci)
        locus_cm = batch.locus_cm[None].expand(flat, loci)
        locus_chrom = batch.locus_chrom[None].expand(flat, loci)
        for layer in self.locus_layers:
            values = layer(values, locus_cm, locus_chrom, valid, self.locus_radius_cm)

        query_cm = batch.query_cm[:, None].expand(batch_size, donors, queries).reshape(flat, queries)
        query_chrom = batch.query_chrom[:, None].expand(batch_size, donors, queries).reshape(flat, queries)
        query_valid = (batch.query_valid[:, None, :]
                       & batch.donor_allowed[:, :, None]).reshape(flat, queries)
        tokens = self.query_token.view(1, 1, -1).expand(flat, queries, self.width)
        donor_query, local_support = self.query_attention(
            tokens, query_cm, query_chrom, query_valid, values, locus_cm,
            locus_chrom, valid, self.query_radius_cm)
        donor_query = donor_query.reshape(batch_size, donors, queries, self.width)
        local_support = local_support.reshape(batch_size, donors, queries)
        donor_supported = query_valid.reshape(batch_size, donors, queries) & (local_support > 0)

        matched = valid & (target.reshape(flat, loci) == reference.reshape(flat, loci))
        discrepant = valid & ~matched
        local_pairs = self._local_boolean_counts(valid, query_cm, query_chrom, query_valid,
                                                 locus_cm, locus_chrom, valid)
        local_matches = self._local_boolean_counts(matched, query_cm, query_chrom, query_valid,
                                                   locus_cm, locus_chrom, valid)
        local_discrepancies = self._local_boolean_counts(
            discrepant, query_cm, query_chrom, query_valid, locus_cm, locus_chrom, valid)
        local_pairs = local_pairs.reshape(batch_size, donors, queries)
        local_matches = local_matches.reshape(batch_size, donors, queries)
        local_discrepancies = local_discrepancies.reshape(batch_size, donors, queries)
        return (donor_query, local_support, donor_supported,
                local_pairs, local_matches, local_discrepancies)

    def _checkpointed_donor_queries(self, _dummy, *tensors):
        """Recompute one donor block; the grad-requiring scalar is only a trigger."""
        return self._donor_queries(*tensors)

    def readout_representation(self, batch: DonorContextBatch):
        """Return the exact frozen vector consumed by ``joint_state_head``."""
        batch_size, donors, loci, queries = self._validate(batch)
        _require(self.query_token.device == batch.target_alt_dosage.device,
                 "model and donor-context inputs must share one device")
        _require(batch_size * donors * loci * self.width <= self.max_pair_state_elements,
                 "B*D*L*W pair-state budget exceeded before allocation")
        _require(batch_size * donors * queries * self.width <= self.max_query_state_elements,
                 "B*D*Q*W query-state budget exceeded before allocation")
        block_size = self.donor_block_size or donors
        outputs = []
        for start in range(0, donors, block_size):
            stop = min(start + block_size, donors)
            part = replace(batch,
                reference_alt_dosage=batch.reference_alt_dosage[start:stop],
                reference_observed=batch.reference_observed[start:stop],
                reference_panel=batch.reference_panel[start:stop],
                donor_allowed=batch.donor_allowed[:, start:stop])
            arguments = tuple(getattr(part, field.name) for field in fields(part))
            # Each call receives its own immutable tuple of tensor views. Retain
            # original dosages/masks, not expanded pair features or layer states.
            if self.donor_block_size is not None and torch.is_grad_enabled():
                dummy = batch.target_alt_dosage.new_zeros((), requires_grad=True)
                result = checkpoint(self._checkpointed_donor_queries, dummy, *arguments,
                                    use_reentrant=True, preserve_rng_state=True)
            else:
                result = self._donor_queries(*arguments)
            outputs.append(result)
        (donor_query, local_support, donor_supported, local_pairs,
         local_matches, local_discrepancies) = (
            torch.cat([part[index] for part in outputs], dim=1)
            for index in range(6))
        # Scores, softmax and panel reductions see ALL donors together; chunking
        # changes temporary storage, never donor weights or normalization.
        scores = self.donor_score(donor_query).squeeze(-1)

        group_vectors, metrics = [], [[] for _ in range(7)]
        for panel_index in range(PANEL_COUNT):
            in_panel = batch.reference_panel == panel_index
            eligible = batch.donor_allowed & in_panel[None]
            member = donor_supported & in_panel[None, :, None]
            presence = member.any(1)
            safe = member.clone()
            safe[:, 0, :] |= ~presence
            weights = torch.softmax(scores.masked_fill(~safe, float("-inf")), dim=1)
            weights = torch.where(member, weights, torch.zeros_like(weights))
            pooled = torch.einsum("bdq,bdqw->bqw", weights, donor_query)
            count = member.sum(1)
            log_mean_exp = (torch.logsumexp(scores.masked_fill(~safe, float("-inf")), 1)
                            - count.clamp_min(1).to(scores.dtype).log())
            log_mean_exp = torch.where(presence, log_mean_exp, torch.zeros_like(log_mean_exp))
            query_mask = batch.query_valid
            numbers = (
                presence,
                eligible.sum(1)[:, None].expand(batch_size, queries) * query_mask,
                count,
                (local_pairs * member).sum(1),
                (local_matches * member).sum(1),
                (local_discrepancies * member).sum(1),
                log_mean_exp,
            )
            # Keep exact raw counts in DonorContextOutput.  log1p is a fixed,
            # monotone and invertible (via expm1) scale change for the copies
            # entering the head; it has no fitted parameter or reliability meaning.
            head_numbers = (numbers[0], *(torch.log1p(value.to(scores.dtype))
                                           for value in numbers[1:6]), numbers[6])
            scalars = torch.stack([value.to(scores.dtype) for value in head_numbers], -1)
            group_vectors.append(torch.cat((pooled, scalars), -1))
            for store, value in zip(metrics, numbers):
                store.append(value)

        combined = torch.cat(group_vectors, -1)
        support = donor_supported.any(1) & batch.query_valid
        stacked = [torch.stack(values, -1) for values in metrics]
        return DonorReadoutRepresentation(
            combined, support, *stacked,
            donor_query, local_support)

    def forward(self, batch: DonorContextBatch):
        represented = self.readout_representation(batch)
        logits = self.joint_state_head(represented.head_input)
        zero = logits.sum(-1, keepdim=True) * 0
        logits = torch.where(batch.query_valid[..., None], logits, zero.expand_as(logits))
        return DonorContextOutput(logits, *tuple(represented.__dict__.values())[1:])
