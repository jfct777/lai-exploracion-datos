"""Authenticated tensor boundary for masked-genotype carrier integration.

This module joins already-masked EVENT loci to a fixed FIT reference panel.  It
does not read human stores, choose people, pairs, segments or scientific
settings.  The only admitted source definition is unphased DNABR2723
``source_minor``.  A real human-data loader is intentionally still required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from r02_context_update_checkpoint import _state_digest
from r02_carrier_interaction_model import PairCommonContext
from r02_masked_genotype_update import MaskedFragment


SOURCE_MINOR = "source_minor_DNABR2723"
SOURCE_KINDS = frozenset(("SYNTHETIC_TEST", "AUTHENTICATED_MAPPED_EVENT_PREFLIGHT"))
_BASES = "ACGT"


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(value) -> str:
    """Reuse the shared digest, encoding only previously unsupported empties.

    ``_state_digest`` is unchanged because it also identifies frozen LAI
    checkpoints.  For every state without an empty tensor this function passes
    the original object through exactly, preserving its historical hash.
    """
    return _state_digest(_digestable_empty_tensors(value))


def _digestable_empty_tensors(value):
    if isinstance(value, Tensor):
        if value.numel() != 0:
            return value
        return {"schema": "r02_empty_tensor_digest_v1",
                "dtype": str(value.detach().cpu().numpy().dtype),
                "shape": tuple(value.shape)}
    if isinstance(value, dict):
        return {key: _digestable_empty_tensors(item)
                for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_digestable_empty_tensors(item) for item in value)
    if isinstance(value, list):
        return [_digestable_empty_tensors(item) for item in value]
    return value


def _sha256_text(name: str, value: str) -> None:
    _require(isinstance(value, str) and len(value) == 64
             and all(letter in "0123456789abcdef" for letter in value),
             f"{name} must be a lowercase SHA-256")


def channel_locus_ids(channel) -> Tuple[str, ...]:
    """Canonical unphased locus identities; counted-allele orientation is separate."""
    chrom = channel.chromosome.detach().cpu().tolist()
    bp = channel.position_bp.detach().cpu().tolist()
    ref = channel.ref.detach().cpu().tolist()
    alt = channel.alt.detach().cpu().tolist()
    return tuple(f"{int(c)}:{int(p)}:{_BASES[int(r)]}:{_BASES[int(a)]}"
                 for c, p, r, a in zip(chrom, bp, ref, alt))


@dataclass(frozen=True)
class CarrierFragmentBatch:
    """One masked fragment plus fixed reference/pair/context tensors.

    ``structural_hidden`` contains masks selected independently of genotype,
    including any declared context-exclusion territory. ``natural_missing`` is
    the raw callability complement. Neither is inferred from ``score_mask``.
    """

    target_dosage: Tensor
    target_called: Tensor
    rare_structural_hidden: Tensor
    rare_natural_missing: Tensor
    common_structural_hidden: Tensor
    common_natural_missing: Tensor
    reference_dosage: Tensor
    reference_called: Tensor
    pairs: Tensor
    pair_segment: Tensor
    fit_reference_eligible: Tensor
    reference_roles: Tuple[str, ...]
    target_ids: Tuple[str, ...]
    target_roles: Tuple[str, ...]
    target_components: Tuple[str, ...]
    reference_ids: Tuple[str, ...]
    reference_components: Tuple[str, ...]
    pair_features: Tensor | None
    locus_ids: Tuple[str, ...]
    segment_ids: Tuple[str, ...]
    common_query_segment: Tensor
    rare_query_segment: Tensor
    common_context: PairCommonContext | None
    source_minor_definition: str
    role_manifest_sha256: str
    source_receipt_sha256: str
    source_kind: str
    human_training_authorized: bool = False

    def check(self, fragment: MaskedFragment, *, common_context_width: int) -> None:
        fragment.check(max(fragment.common.dosage.shape[1],
                           fragment.rare.dosage.shape[1], 1))
        _require(self.source_minor_definition == SOURCE_MINOR,
                 "DNABR2723 source_minor definition must remain fixed")
        _require(self.source_kind in SOURCE_KINDS,
                 "unknown or human-admitting carrier source kind")
        _require(self.human_training_authorized is False,
                 "this preflight tensor API cannot authorize human training")
        _sha256_text("role_manifest_sha256", self.role_manifest_sha256)
        _sha256_text("source_receipt_sha256", self.source_receipt_sha256)

        device = fragment.rare.dosage.device
        _require(isinstance(self.target_dosage, Tensor)
                 and isinstance(self.reference_dosage, Tensor)
                 and self.target_dosage.ndim == self.reference_dosage.ndim == 2,
                 "EVENT target/reference dosage must be matrices")
        batch, loci = self.target_dosage.shape
        references = self.reference_dosage.shape[0]
        dtype = next((value.dtype for value in (self.target_dosage,
                                                self.reference_dosage)), None)
        _require(self.reference_dosage.shape[1] == loci and batch > 0
                 and references > 0 and loci > 0
                 and self.target_dosage.is_floating_point()
                 and self.reference_dosage.dtype == dtype
                 and self.target_dosage.device == self.reference_dosage.device == device,
                 "EVENT target/reference matrices must share nonempty axes, dtype and device")
        for name, value, shape in (
                ("target_called", self.target_called, (batch, loci)),
                ("rare_structural_hidden", self.rare_structural_hidden, (batch, loci)),
                ("rare_natural_missing", self.rare_natural_missing, (batch, loci)),
                ("reference_called", self.reference_called, (references, loci))):
            _require(isinstance(value, Tensor) and value.dtype == torch.bool
                     and tuple(value.shape) == shape and value.device == device,
                     f"{name} axis/dtype/device differs")

        _require(tuple(self.target_ids) == fragment.rare.person_ids
                 == fragment.common.person_ids,
                 "carrier targets differ from masked-fragment people/order")
        _require(tuple(self.target_roles) == ("FIT",) * batch,
                 "carrier integration is FIT-only before separate admission")
        _require(len(self.target_components) == batch and all(self.target_components),
                 "one authenticated component per target required")
        _require(len(self.reference_ids) == references
                 and len(set(self.reference_ids)) == references
                 and len(self.reference_components) == references
                 and all(self.reference_ids) and all(self.reference_components)
                 and tuple(self.reference_roles) == ("FIT",) * references,
                 "fixed unique FIT reference/component axes required")
        _require(tuple(self.locus_ids) == channel_locus_ids(fragment.rare),
                 "EVENT locus identities/order differ from masked rare channel")
        _require(bool((fragment.rare.counted_allele[fragment.rare.locus_valid]
                       >= 0).all()), "rare counted-allele metadata missing")

        raw_visible = self.target_called & ~self.rare_structural_hidden
        _require(torch.equal(self.rare_natural_missing, ~self.target_called),
                 "rare natural-missing mask must be the raw callability complement")
        _require(torch.equal(raw_visible, fragment.rare.genotype_visible),
                 "rare structural mask/callability does not reconstruct visible inputs")
        _require(bool(torch.equal(
            self.target_dosage[raw_visible],
            fragment.rare.dosage[raw_visible].to(dtype))),
            "visible EVENT dosage differs from masked rare input")
        _require(bool((fragment.rare_targets.target_scored
                       & ~(self.target_called & self.rare_structural_hidden)).any()) is False,
                 "scored rare label is not raw-called and structurally hidden")

        common_shape = tuple(fragment.common.dosage.shape)
        for name, value in (("common_structural_hidden", self.common_structural_hidden),
                            ("common_natural_missing", self.common_natural_missing)):
            _require(isinstance(value, Tensor) and value.dtype == torch.bool
                     and tuple(value.shape) == common_shape and value.device == device,
                     f"{name} axis/dtype/device differs")
        common_called = ~self.common_natural_missing
        _require(torch.equal(common_called & ~self.common_structural_hidden,
                             fragment.common.genotype_visible),
                 "common structural mask/callability does not reconstruct visible inputs")
        _require(bool((fragment.common_targets.target_scored
                       & ~(common_called & self.common_structural_hidden)).any()) is False,
                 "scored common label is not raw-called and structurally hidden")

        segments = len(self.segment_ids)
        _require(segments > 0 and len(set(self.segment_ids)) == segments
                 and all(isinstance(value, str) and value for value in self.segment_ids),
                 "unique nonempty segment identities required")
        _require(self.fit_reference_eligible.dtype == torch.bool
                 and tuple(self.fit_reference_eligible.shape)
                 == (batch, segments, references)
                 and self.fit_reference_eligible.device == device,
                 "per-target B x S x R FIT eligibility required")
        for i, (target, component) in enumerate(zip(self.target_ids,
                                                    self.target_components)):
            for j, (reference, ref_component) in enumerate(zip(
                    self.reference_ids, self.reference_components)):
                if target == reference or component == ref_component:
                    _require(not bool(self.fit_reference_eligible[i, :, j].any()),
                             "self and complete family component must be excluded per target")
        _require(bool(self.fit_reference_eligible.any(2).all()),
                 "every target/segment needs a fixed admissible FIT reference")

        _require(self.pairs.dtype == self.pair_segment.dtype == torch.int64
                 and self.pairs.ndim == 2 and self.pairs.shape[1] == 2
                 and tuple(self.pair_segment.shape) == (self.pairs.shape[0],)
                 and self.pairs.device == self.pair_segment.device == device,
                 "pair and segment axes differ")
        for name, query, count in (("common", self.common_query_segment,
                                    fragment.common.dosage.shape[1]),
                                   ("rare", self.rare_query_segment,
                                    fragment.rare.dosage.shape[1])):
            _require(query.dtype == torch.int64 and tuple(query.shape) == (count,)
                     and query.device == device
                     and (query.numel() == 0 or bool(
                         ((query >= 0) & (query < segments)).all())),
                     f"{name} query-to-segment map differs")

        if common_context_width == 0:
            _require(self.common_context is None,
                     "COMMON context supplied to a width-zero interaction model")
        else:
            _require(isinstance(self.common_context, PairCommonContext),
                     "PairCommonContext required")
            declaration = self.common_context.declaration
            _require(declaration.target_ids == tuple(self.target_ids)
                     and declaration.reference_ids == tuple(self.reference_ids)
                     and declaration.segment_ids == tuple(self.segment_ids),
                     "COMMON declaration axes differ from carrier batch")

    def identities(self) -> dict:
        """Hashes used by update/checkpoint identity; hidden label values are absent."""
        visible = self.target_called & ~self.rare_structural_hidden
        clean_target = torch.where(visible, self.target_dosage,
                                   torch.zeros((), dtype=self.target_dosage.dtype,
                                               device=self.target_dosage.device))
        context = None if self.common_context is None else {
            "values": self.common_context.values,
            "valid": self.common_context.valid,
            "declaration": {
                "target_ids": self.common_context.declaration.target_ids,
                "reference_ids": self.common_context.declaration.reference_ids,
                "segment_ids": self.common_context.declaration.segment_ids,
                "excluded_locus_ids_by_target_segment":
                    self.common_context.declaration.excluded_locus_ids_by_target_segment,
            },
        }
        panel = {"target_ids": self.target_ids, "target_roles": self.target_roles,
                 "target_components": self.target_components,
                 "reference_ids": self.reference_ids,
                 "reference_roles": self.reference_roles,
                 "reference_components": self.reference_components,
                 "eligible": self.fit_reference_eligible,
                 "role_manifest": self.role_manifest_sha256,
                 "source_receipt": self.source_receipt_sha256}
        pair_map = {"pairs": self.pairs, "pair_segment": self.pair_segment,
                    "pair_features": self.pair_features,
                    "locus_ids": self.locus_ids, "segment_ids": self.segment_ids,
                    "common_query_segment": self.common_query_segment,
                    "rare_query_segment": self.rare_query_segment,
                    "source_minor": self.source_minor_definition}
        masks = {"clean_visible_target": clean_target,
                 "target_called": self.target_called,
                 "rare_structural_hidden": self.rare_structural_hidden,
                 "rare_natural_missing": self.rare_natural_missing,
                 "common_structural_hidden": self.common_structural_hidden,
                 "common_natural_missing": self.common_natural_missing,
                 "reference_dosage": self.reference_dosage,
                 "reference_called": self.reference_called}
        per_target = []
        for index, target in enumerate(self.target_ids):
            per_target.append(_sha({"target": target,
                "component": self.target_components[index],
                "references": self.reference_ids,
                "reference_components": self.reference_components,
                "eligibility": self.fit_reference_eligible[index],
                "pairs": pair_map,
                "context": None if context is None else {
                    "values": self.common_context.values[index],
                    "valid": self.common_context.valid[index],
                    "exclusions": self.common_context.declaration.
                        excluded_locus_ids_by_target_segment[index]}}))
        return {"schema": "r02_masked_carrier_batch_v1",
                "panel_sha256": _sha(panel), "pairs_sha256": _sha(pair_map),
                "context_sha256": _sha(context), "feature_inputs_sha256": _sha(masks),
                "per_target_policy_sha256": per_target,
                "source_kind": self.source_kind,
                "human_training_authorized": self.human_training_authorized}
