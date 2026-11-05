"""Produce directed COMMON background features for carrier interactions.

The four channels are a target-directed standardized genotype cross-product,
joint callability, target callability and reference callability.  Statistics
use an explicit fixed FIT panel for each target and empirical observed mean and
variance.  This is a supplementary background summary: it discards locus
order and local LD and never replaces ordered COMMON inputs or EVENT
interactions.  It does not infer phase, IBD, ancestry or relatedness.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import NamedTuple, Sequence, Tuple

import numpy as np
import torch

from r02_carrier_interaction_model import (
    PairCommonContext, PairCommonContextDeclaration)


EXCLUDE_SAME_COMPONENT = "EXCLUDE_SAME_COMPONENT"
INCLUDE_SAME_COMPONENT_SENSITIVITY = "INCLUDE_SAME_COMPONENT_SENSITIVITY"
FAMILY_RULES = frozenset((EXCLUDE_SAME_COMPONENT,
                          INCLUDE_SAME_COMPONENT_SENSITIVITY))
NO_PHASE_INTERPRETATION = "NONE_DIPLOID_DOSAGE_ONLY"
CHANNELS = ("empirical_standardized_cross_product", "joint_called_fraction",
            "target_called_fraction", "reference_called_fraction")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _axis(name, values, length):
    result = tuple(map(str, values))
    _require(len(result) == length and len(set(result)) == length
             and all(result), f"{name} axis differs")
    return result


def _array_hash(value):
    value = np.ascontiguousarray(value)
    header = json.dumps({"dtype": value.dtype.str, "shape": value.shape},
                        sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _source_identity(module_name, path):
    path = Path(path).resolve()
    _require(path.is_file() and path.suffix == ".py",
             f"Python source required for {module_name}")
    return {"module": module_name, "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


PRODUCER_SOURCE = _source_identity(__name__, __file__)


@dataclass(frozen=True)
class CommonContextPolicy:
    """Explicit biological scope; no numerical or pair-selection tuning."""

    family_rule: str
    ordered_common_path_retained: bool
    event_interaction_path_retained: bool
    phase_interpretation: str = NO_PHASE_INTERPRETATION

    def __post_init__(self):
        _require(self.family_rule in FAMILY_RULES, "explicit family rule required")
        _require(self.ordered_common_path_retained is True,
                 "COMMON summary cannot replace ordered COMMON inputs")
        _require(self.event_interaction_path_retained is True,
                 "COMMON summary cannot replace EVENT interactions")
        _require(self.phase_interpretation == NO_PHASE_INTERPRETATION,
                 "COMMON context cannot reinterpret phase")


@dataclass(frozen=True)
class CommonContextTechnicalLimits:
    """Engineering bounds; none selects loci, references or model capacity."""

    max_chunk_sites: int
    max_source_sites: int
    max_chunk_bytes: int
    max_working_bytes: int
    max_output_numel: int

    def __post_init__(self):
        for name in ("max_chunk_sites", "max_source_sites", "max_chunk_bytes",
                     "max_working_bytes", "max_output_numel"):
            _require(type(getattr(self, name)) is int and getattr(self, name) > 0,
                     f"positive integer {name} required")


@dataclass(frozen=True)
class CommonContextArrays:
    target_dosage: np.ndarray
    target_called: np.ndarray
    reference_dosage: np.ndarray
    reference_called: np.ndarray
    target_ids: Tuple[str, ...]
    target_components: Tuple[str, ...]
    reference_ids: Tuple[str, ...]
    reference_components: Tuple[str, ...]
    reference_roles: Tuple[str, ...]
    common_locus_ids: Tuple[str, ...]
    segment_ids: Tuple[str, ...]
    segment_locus_mask: np.ndarray
    excluded_locus_ids_by_target_segment: Tuple[
        Tuple[Tuple[str, ...], ...], ...]


class CommonContextProduct(NamedTuple):
    context: PairCommonContext
    target_reference_eligible: torch.Tensor  # B x S x R
    receipt: dict


def _validate(values: CommonContextArrays):
    _require(isinstance(values, CommonContextArrays), "CommonContextArrays required")
    td, rd = values.target_dosage, values.reference_dosage
    tc, rc = values.target_called, values.reference_called
    _require(isinstance(td, np.ndarray) and isinstance(rd, np.ndarray)
             and td.ndim == rd.ndim == 2 and td.shape[1] == rd.shape[1]
             and np.issubdtype(td.dtype, np.number)
             and np.issubdtype(rd.dtype, np.number), "dosage matrices differ")
    b, loci = td.shape; references = rd.shape[0]
    _require(tc.shape == td.shape and rc.shape == rd.shape
             and tc.dtype == rc.dtype == np.bool_, "called masks differ")
    target_ids = _axis("target", values.target_ids, b)
    reference_ids = _axis("reference", values.reference_ids, references)
    target_components = tuple(map(str, values.target_components))
    reference_components = tuple(map(str, values.reference_components))
    _require(len(target_components) == b and all(target_components)
             and len(reference_components) == references
             and all(reference_components), "component axes differ")
    _require(tuple(values.reference_roles) == ("FIT",) * references,
             "all common-context references must be FIT")
    locus_ids = _axis("COMMON locus", values.common_locus_ids, loci)
    segment_ids = _axis("segment", values.segment_ids,
                        values.segment_locus_mask.shape[0])
    segments = len(segment_ids)
    _require(values.segment_locus_mask.shape == (segments, loci)
             and values.segment_locus_mask.dtype == np.bool_,
             "segment COMMON mask differs")
    exclusions = values.excluded_locus_ids_by_target_segment
    _require(type(exclusions) is tuple and len(exclusions) == b,
             "one exclusion row required per target")
    for row in exclusions:
        _require(type(row) is tuple and len(row) == segments,
                 "one exclusion cell required per target and segment")
        for cell in row:
            _require(type(cell) is tuple and len(cell) == len(set(cell))
                     and all(type(x) is str and x for x in cell),
                     "excluded locus identities differ")
    _require(np.isfinite(rd[rc]).all()
             and np.isin(rd[rc], (0, 1, 2)).all(),
             "observed FIT dosage must be 0, 1 or 2")
    # Hidden target values may be arbitrary. Validate only a cell visible in at
    # least one target-specific segment after its own exclusions.
    locus_lookup = {name: index for index, name in enumerate(locus_ids)}
    visible_any = np.zeros_like(tc)
    for i in range(b):
        for s in range(segments):
            mask = values.segment_locus_mask[s].copy()
            for name in exclusions[i][s]:
                if name in locus_lookup:
                    mask[locus_lookup[name]] = False
            visible_any[i] |= mask
    used_target = tc & visible_any
    _require(np.isfinite(td[used_target]).all()
             and np.isin(td[used_target], (0, 1, 2)).all(),
             "visible target dosage must be 0, 1 or 2")
    return (target_ids, target_components, reference_ids,
            reference_components, locus_ids, segment_ids, locus_lookup)


def _validate_metadata(target_ids, target_components, reference_ids,
                       reference_components, reference_roles, segment_ids,
                       exclusions):
    target_ids = _axis("target", target_ids, len(target_ids))
    reference_ids = _axis("reference", reference_ids, len(reference_ids))
    segment_ids = _axis("segment", segment_ids, len(segment_ids))
    target_components = tuple(map(str, target_components))
    reference_components = tuple(map(str, reference_components))
    _require(len(target_components) == len(target_ids)
             and all(target_components)
             and len(reference_components) == len(reference_ids)
             and all(reference_components), "component axes differ")
    _require(tuple(reference_roles) == ("FIT",) * len(reference_ids),
             "all common-context references must be FIT")
    _require(type(exclusions) is tuple and len(exclusions) == len(target_ids),
             "one exclusion row required per target")
    for row in exclusions:
        _require(type(row) is tuple and len(row) == len(segment_ids),
                 "one exclusion cell required per target and segment")
        for cell in row:
            _require(type(cell) is tuple and len(cell) == len(set(cell))
                     and all(type(x) is str and x for x in cell),
                     "excluded locus identities differ")
    return (target_ids, target_components, reference_ids,
            reference_components, segment_ids)


class _CommonContextAccumulator:
    """Accumulate independent-locus sufficient sums without retaining loci."""

    def __init__(self, target_ids, target_components, reference_ids,
                 reference_components, reference_roles, segment_ids, exclusions,
                 policy, dtype, device, limits=None):
        _require(isinstance(policy, CommonContextPolicy),
                 "CommonContextPolicy required")
        _require(dtype in (torch.float32, torch.float64),
                 "float32 or float64 output required")
        (self.target_ids, self.target_components, self.reference_ids,
         self.reference_components, self.segment_ids) = _validate_metadata(
            target_ids, target_components, reference_ids, reference_components,
            reference_roles, segment_ids, exclusions)
        self.exclusions = exclusions
        self.policy = policy
        self.dtype = dtype
        self.device = device
        self.limits = limits
        b, r, s = (len(self.target_ids), len(self.reference_ids),
                   len(self.segment_ids))
        numel = b * r * s * len(CHANNELS)
        if limits is not None:
            _require(isinstance(limits, CommonContextTechnicalLimits),
                     "CommonContextTechnicalLimits required")
            _require(numel <= limits.max_output_numel,
                     "COMMON output element limit exceeded before source read")
            tensor_itemsize = 4 if dtype == torch.float32 else 8
            brs, bs = b * r * s, b * s
            # Includes accumulators, final NumPy/tensor values and both masks.
            self.fixed_bytes = (brs * (8 * 3 + 1) + bs * 8 * 2
                                + numel * (8 + tensor_itemsize) + brs * 2)
            _require(self.fixed_bytes <= limits.max_working_bytes,
                     "COMMON fixed working-memory limit exceeded before source read")
        else:
            self.fixed_bytes = 0
        self.eligible = np.asarray([[[
            self.target_ids[i] != self.reference_ids[j]
            and (policy.family_rule != EXCLUDE_SAME_COMPONENT
                 or self.target_components[i] != self.reference_components[j])
            for j in range(r)] for _ in range(s)] for i in range(b)], np.bool_)
        _require(np.all(self.eligible == self.eligible[:, :1, :]),
                 "fixed FIT eligibility must be identical across segments")
        self.dot_sum = np.zeros((b, r, s), np.float64)
        self.d_counts = np.zeros((b, r, s), np.int64)
        self.reference_called_counts = np.zeros((b, r, s), np.int64)
        self.target_called_counts = np.zeros((b, s), np.int64)
        self.l_counts = np.zeros((b, s), np.int64)
        self.seen_exclusions = [[set() for _ in range(s)] for _ in range(b)]

    def check_chunk_budget(self, sites, source_bytes):
        """Reject a source descriptor before mapping or allocating its arrays."""
        if self.limits is None:
            return
        _require(type(sites) is int and 0 < sites <= self.limits.max_chunk_sites,
                 "COMMON source chunk site limit exceeded before source read")
        _require(type(source_bytes) is int
                 and 0 < source_bytes <= self.limits.max_chunk_bytes,
                 "COMMON source chunk byte limit exceeded before source read")
        b, r = len(self.target_ids), len(self.reference_ids)
        # Conservative upper bound for selected dosage/masks, per-target panel
        # centering and locus statistics. The mapped source remains read-only.
        transient = sites * ((b + r) * 18 + r * 24 + 64)
        _require(self.fixed_bytes + transient <= self.limits.max_working_bytes,
                 "COMMON chunk working-memory limit exceeded before source read")

    def add_chunk(self, target_dosage, target_called, reference_dosage,
                  reference_called, locus_ids, segment_locus_mask):
        td, rd = np.asarray(target_dosage), np.asarray(reference_dosage)
        tc, rc = np.asarray(target_called), np.asarray(reference_called)
        b, r, s = (len(self.target_ids), len(self.reference_ids),
                   len(self.segment_ids))
        _require(td.ndim == rd.ndim == 2 and td.shape[0] == b
                 and rd.shape[0] == r and td.shape[1] == rd.shape[1]
                 and np.issubdtype(td.dtype, np.number)
                 and np.issubdtype(rd.dtype, np.number), "chunk dosage matrices differ")
        loci = td.shape[1]
        _require(tc.shape == td.shape and rc.shape == rd.shape
                 and tc.dtype == rc.dtype == np.bool_, "chunk called masks differ")
        locus_ids = _axis("COMMON chunk locus", locus_ids, loci)
        segment_locus_mask = np.asarray(segment_locus_mask)
        _require(segment_locus_mask.shape == (s, loci)
                 and segment_locus_mask.dtype == np.bool_,
                 "chunk segment COMMON mask differs")
        _require(np.isfinite(rd[rc]).all() and np.isin(rd[rc], (0, 1, 2)).all(),
                 "observed FIT dosage must be 0, 1 or 2")
        lookup = {name: index for index, name in enumerate(locus_ids)}
        for i in range(b):
            panel = self.eligible[i, 0]
            panel_dose, panel_called = rd[panel], rc[panel]
            for k in range(s):
                mask = segment_locus_mask[k].copy()
                for name in self.exclusions[i][k]:
                    index = lookup.get(name)
                    if index is not None:
                        self.seen_exclusions[i][k].add(name)
                        if segment_locus_mask[k, index]:
                            mask[index] = False
                selected = np.flatnonzero(mask)
                if not len(selected) or not panel.any():
                    continue
                visible = tc[i, selected]
                _require(np.isfinite(td[i, selected][visible]).all()
                         and np.isin(td[i, selected][visible], (0, 1, 2)).all(),
                         "visible target dosage must be 0, 1 or 2")
                called = panel_called[:, selected]
                dose = panel_dose[:, selected]
                counts = called.sum(0)
                means = np.zeros(len(selected), np.float64)
                positive = counts > 0
                means[positive] = np.where(called, dose, 0).sum(0)[positive] / counts[positive]
                centered = np.where(called, dose - means[None, :], 0.)
                variances = np.zeros(len(selected), np.float64)
                variances[positive] = (centered ** 2).sum(0)[positive] / counts[positive]
                keep_local = positive & np.isfinite(variances) & (variances > 0)
                kept = selected[keep_local]
                l_count = len(kept)
                self.l_counts[i, k] += l_count
                if not l_count:
                    continue
                mean = means[keep_local]
                scale = np.sqrt(variances[keep_local])
                target_observed = tc[i, kept]
                self.target_called_counts[i, k] += int(target_observed.sum())
                target_z = (td[i, kept].astype(np.float64) - mean) / scale
                for j in np.flatnonzero(panel):
                    reference_observed = rc[j, kept]
                    self.reference_called_counts[i, j, k] += int(reference_observed.sum())
                    joint = target_observed & reference_observed
                    d = int(joint.sum())
                    self.d_counts[i, j, k] += d
                    if d:
                        reference_z = (rd[j, kept].astype(np.float64) - mean) / scale
                        self.dot_sum[i, j, k] += float(np.sum(
                            target_z[joint] * reference_z[joint], dtype=np.float64))

    def finish(self, source_receipt=None):
        b, r, s = self.dot_sum.shape
        unmatched = tuple(tuple(tuple(sorted(
            set(self.exclusions[i][k]) - self.seen_exclusions[i][k]))
            for k in range(s)) for i in range(b))
        _require(not any(cell for row in unmatched for cell in row),
                 "excluded COMMON locus identity absent after all source chunks")
        output = np.zeros((b, r, s, len(CHANNELS)), np.float64)
        valid = self.eligible.transpose(0, 2, 1) & (self.d_counts > 0)
        for i, j, k in np.argwhere(valid):
            d, loci = self.d_counts[i, j, k], self.l_counts[i, k]
            _require(loci > 0, "joint support requires positive locus support")
            output[i, j, k] = (
                self.dot_sum[i, j, k] / d,
                d / loci,
                self.target_called_counts[i, k] / loci,
                self.reference_called_counts[i, j, k] / loci)
        declaration = PairCommonContextDeclaration(
            self.target_ids, self.reference_ids, self.segment_ids,
            self.exclusions)
        tensor_values = torch.as_tensor(output, dtype=self.dtype, device=self.device)
        tensor_valid = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        tensor_eligible = torch.as_tensor(
            self.eligible, dtype=torch.bool, device=self.device)
        tensor_values_numpy = tensor_values.detach().cpu().numpy()
        receipt = {
            "schema": "r02_pair_common_context_receipt_v1",
            "status": "PRODUCED_NOT_TRAINING_ADMISSION",
            "channels": list(CHANNELS),
            "definition": "TARGET_DIRECTED_FIXED_FIT_PANEL_EMPIRICAL_MEAN_POPULATION_VARIANCE",
            "family_rule": self.policy.family_rule,
            "self_reference": "EXCLUDED_PER_TARGET",
            "batch_dependence": "NONE_REFERENCES_AND_STATISTICS_FIXED_PER_TARGET",
            "phase_interpretation": self.policy.phase_interpretation,
            "common_role": "SUPPLEMENT_RETAINS_ORDERED_COMMON_AND_EVENT_INTERACTIONS",
            "information_lost": ["locus_order", "local_LD", "EVENT_locus_identity"],
            "event_source_minor_redefined": False,
            "loci_post_exclusion_positive_empirical_variance": self.l_counts.tolist(),
            "cells_without_joint_support": int((self.d_counts == 0).sum()),
            "unmatched_exclusion_names": [[list(cell) for cell in row]
                                            for row in unmatched],
            "joint_called_counts_sha256": _array_hash(self.d_counts),
            "values_sha256": _array_hash(output),
            "values_sha256_alias": "values_float64_sha256",
            "values_float64_sha256": _array_hash(output),
            "values_tensor_sha256": _array_hash(tensor_values_numpy),
            "values_tensor_dtype": str(tensor_values_numpy.dtype),
            "valid_sha256": _array_hash(valid),
            "eligible_sha256": _array_hash(self.eligible),
            "target_row_sha256": {
                self.target_ids[i]: {
                    "eligible_sha256": _array_hash(self.eligible[i]),
                    "values_float64_sha256": _array_hash(output[i]),
                    "valid_sha256": _array_hash(valid[i]),
                } for i in range(b)},
            "exclusions_sha256": _json_hash(self.exclusions),
            "source": source_receipt,
            "role_admission": ("AUTHENTICATED_BY_MAPPED_SOURCE_RECEIPT"
                               if source_receipt is not None else
                               "NONE_ARRAY_API_REQUIRES_SEPARATE_ADMISSION"),
            "technical_limits": (None if self.limits is None else {
                name: getattr(self.limits, name) for name in (
                    "max_chunk_sites", "max_source_sites", "max_chunk_bytes",
                    "max_working_bytes", "max_output_numel")}),
            "training_authorized": False,
        }
        return CommonContextProduct(PairCommonContext(
            tensor_values, tensor_valid, declaration), tensor_eligible, receipt)


def produce_common_context(values: CommonContextArrays,
                           policy: CommonContextPolicy, *,
                           dtype=torch.float32, device="cpu",
                           source_receipt=None) -> CommonContextProduct:
    """Compute b in float64, then return the existing PairCommonContext API."""
    (target_ids, target_components, reference_ids, reference_components,
     locus_ids, segment_ids, _) = _validate(values)
    accumulator = _CommonContextAccumulator(
        target_ids, target_components, reference_ids, reference_components,
        values.reference_roles, segment_ids,
        values.excluded_locus_ids_by_target_segment, policy, dtype, device)
    accumulator.add_chunk(
        values.target_dosage, values.target_called, values.reference_dosage,
        values.reference_called, locus_ids, values.segment_locus_mask)
    return accumulator.finish(source_receipt)


@dataclass(frozen=True)
class CommonSegment:
    segment_id: str
    start_bp: int
    end_bp: int


def _authenticated_reader_sources(reader, kind):
    """Authenticate one exact reader implementation; never accept a proxy."""
    reader_type = type(reader)
    reader_module = sys.modules.get(reader_type.__module__)
    expected = ("r02_event_mmap_reader", "MappedEventChromosome") if kind == "mapped" else (
        "r02_masked_store", "RawMaskedChromosome")
    _require(reader_type.__module__ == expected[0] and reader_module is not None
             and getattr(reader_module, expected[1], None) is reader_type
             and callable(getattr(reader, "_assert_metadata_unchanged", None))
             and callable(getattr(reader, "safe_path", None))
             and callable(getattr(getattr(reader, "cache", None), "open", None))
             and isinstance(getattr(reader, "r", None), dict)
             and type(reader.r.get("max_array_mb")) is int,
             f"exact {expected[1]} implementation required")
    reader._assert_metadata_unchanged()
    code_sources = {
        "producer": dict(PRODUCER_SOURCE),
    }
    if kind == "mapped":
        code_sources.update(
            mapped_reader=_source_identity(reader_module.__name__, reader_module.__file__),
            base_reader=_source_identity(reader_module.original.__name__,
                                         reader_module.original.__file__))
    else:
        import r02_event_mmap_reader as mapped_module
        code_sources.update(
            raw_masked_reader=_source_identity(reader_module.__name__, reader_module.__file__),
            mapped_reader=_source_identity(mapped_module.__name__, mapped_module.__file__),
            base_reader=_source_identity(mapped_module.original.__name__,
                                         mapped_module.original.__file__))
    return code_sources


def _produce_from_authenticated_reader(reader, target_ids, reference_ids,
                                        segments, exclusions, policy, *, limits,
                                        dtype, device, reader_kind):
    """Shared streaming core after an exact reader-specific authentication."""
    code_sources = _authenticated_reader_sources(reader, reader_kind)
    manifest = reader.manifest
    _require(manifest.get("schema") == "r02_genomic_event_chromosome_v1"
             and manifest.get("stage") == "COMPLETE_RAW_INDIVIDUAL_GENOTYPES_NO_TRAINING"
             and manifest.get("training_allowed") is False
             and "common" in manifest.get("channels", {}),
             "authenticated non-training COMMON chromosome manifest required")
    segments = tuple(segments)
    _require(segments and all(isinstance(x, CommonSegment) and x.segment_id
             and type(x.start_bp) is int and type(x.end_bp) is int
             and x.start_bp <= x.end_bp for x in segments),
             "explicit nonempty COMMON segments required")
    targets, references = tuple(map(str, target_ids)), tuple(map(str, reference_ids))
    _axis("target", targets, len(targets)); _axis("reference", references, len(references))
    _require(set(targets) | set(references) <= set(reader.people),
             "unknown target or FIT reference")
    combined = tuple(dict.fromkeys((*targets, *references)))
    bases = np.asarray(tuple("ACGT"))
    person = {p["sample_id"]: p for p in reader.persons}
    _require(set(combined) <= set(person), "person metadata axis differs")
    _require(tuple(person[x]["role"] for x in targets) == ("FIT",) * len(targets),
             "all common-context targets must be FIT before source read")
    _require(isinstance(limits, CommonContextTechnicalLimits),
             "CommonContextTechnicalLimits required before source read")
    source_chunks = manifest["channels"]["common"]
    _require(isinstance(source_chunks, list) and source_chunks,
             "nonempty authenticated COMMON chunk inventory required")
    previous_end = None
    for chunk in source_chunks:
        _require(isinstance(chunk, dict)
                 and type(chunk.get("n_sites")) is int
                 and chunk["n_sites"] > 0
                 and type(chunk.get("bytes")) is int and chunk["bytes"] > 0
                 and type(chunk.get("start_bp")) is int
                 and type(chunk.get("end_bp")) is int
                 and chunk["start_bp"] <= chunk["end_bp"]
                 and isinstance(chunk.get("path"), str) and chunk["path"]
                 and isinstance(chunk.get("sha256"), str)
                 and len(chunk["sha256"]) == 64,
                 "authenticated COMMON chunk descriptor differs")
        _require(previous_end is None or chunk["start_bp"] > previous_end,
                 "COMMON chunk inventory must be ordered and nonoverlapping")
        previous_end = chunk["end_bp"]

    relevant_chunks = [chunk for chunk in source_chunks if not all(
        chunk["end_bp"] < x.start_bp or chunk["start_bp"] > x.end_bp
        for x in segments)]
    _require(sum(chunk["n_sites"] for chunk in relevant_chunks)
             <= limits.max_source_sites,
             "COMMON source site limit exceeded before source read")

    accumulator = _CommonContextAccumulator(
        targets, tuple(person[x]["component"] for x in targets), references,
        tuple(person[x]["component"] for x in references),
        tuple(person[x]["role"] for x in references),
        tuple(x.segment_id for x in segments), exclusions, policy, dtype, device,
        limits=limits)
    target_rows = np.asarray([reader.people[x] for x in targets], np.int64)
    reference_rows = np.asarray([reader.people[x] for x in references], np.int64)

    selected_chunk_receipts = []
    for chunk in relevant_chunks:
        # These checks precede safe_path(), cache.open() and every per-source
        # NumPy allocation performed by this producer.
        accumulator.check_chunk_budget(chunk["n_sites"], chunk["bytes"])
        with reader.cache.open(reader.safe_path(chunk["path"]), chunk,
                               len(reader.persons),
                               reader.r["max_array_mb"]) as data:
            positions = data["position_bp"]
            selected_mask = np.zeros(len(positions), np.bool_)
            for segment in segments:
                selected_mask |= ((positions >= segment.start_bp)
                                  & (positions <= segment.end_bp))
            selected = np.flatnonzero(selected_mask)
            if not len(selected):
                continue
            selected_positions = positions[selected]
            locus_ids = tuple(
                f"{reader.chrom}:{int(bp)}:{bases[int(ref)]}:{bases[int(alt)]}"
                for bp, ref, alt in zip(selected_positions, data["ref"][selected],
                                        data["alt"][selected]))
            segment_mask = np.asarray([
                (selected_positions >= x.start_bp)
                & (selected_positions <= x.end_bp) for x in segments], np.bool_)
            accumulator.add_chunk(
                data["dosage"][np.ix_(target_rows, selected)],
                data["observed"][np.ix_(target_rows, selected)],
                data["dosage"][np.ix_(reference_rows, selected)],
                data["observed"][np.ix_(reference_rows, selected)],
                locus_ids, segment_mask)
            selected_chunk_receipts.append({
                "path": chunk["path"], "sha256": chunk["sha256"],
                "bytes": chunk["bytes"], "n_sites": chunk["n_sites"],
                "selected_sites": len(selected),
                "selected_locus_ids_sha256": _json_hash(locus_ids)})
        reader._assert_metadata_unchanged()
    reader._assert_metadata_unchanged()
    source = {"schema": reader.manifest["schema"],
        "chromosome_manifest_sha256": reader.manifest_sha256,
        "event_contract_sha256": reader.manifest["event_contract_sha256"],
        "source_contract_sha256": reader.manifest["source_contract_sha256"],
        "chromosome": reader.chrom, "channel_read": "common",
        "target_axis_sha256": _json_hash(targets),
        "reference_axis_sha256": _json_hash(references),
        "role_component_sha256": _json_hash([(x, person[x]["role"], person[x]["component"])
                                              for x in combined]),
        "selected_source_chunks": selected_chunk_receipts,
        "selected_source_chunks_sha256": _json_hash(selected_chunk_receipts),
        "code_sources": code_sources,
        "code_sources_sha256": _json_hash(code_sources),
        "streaming": "AUTHENTICATED_CHUNK_ORDER_NO_CHROMOSOME_MATRIX",
        "target_role_requirement": "FIT_ONLY_VERIFIED_BEFORE_SOURCE_READ",
        "exclusions_sha256": _json_hash(exclusions)}
    return accumulator.finish(source)


def produce_from_mapped_reader(reader, target_ids: Sequence[str],
                               reference_ids: Sequence[str],
                               segments: Sequence[CommonSegment],
                               exclusions, policy: CommonContextPolicy, *,
                               limits: CommonContextTechnicalLimits,
                               dtype=torch.float32, device="cpu"):
    """Stream COMMON chunks from the exact historical mapped reader."""
    return _produce_from_authenticated_reader(
        reader, target_ids, reference_ids, segments, exclusions, policy,
        limits=limits, dtype=dtype, device=device, reader_kind="mapped")


def produce_from_raw_masked_reader(reader, target_ids: Sequence[str],
                                   reference_ids: Sequence[str],
                                   segments: Sequence[CommonSegment],
                                   exclusions, policy: CommonContextPolicy, *,
                                   limits: CommonContextTechnicalLimits,
                                   dtype=torch.float32, device="cpu"):
    """Stream COMMON chunks from the exact authenticated masked-store reader.

    This explicit entry point keeps the historical exact-type check intact. It
    does not wrap, subclass or impersonate ``MappedEventChromosome``.
    """
    return _produce_from_authenticated_reader(
        reader, target_ids, reference_ids, segments, exclusions, policy,
        limits=limits, dtype=dtype, device=device, reader_kind="raw_masked")
