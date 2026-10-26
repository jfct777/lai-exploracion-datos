"""Development controls for the person-by-cell masked-genotype objective.

The flanking-marker predictor represents one deliberately small source of local
linkage disequilibrium: the nearest structurally valid COMMON marker on each
side of the hidden cell, restricted to an explicitly supplied distance ring.
It fits genotype counts only in declared FIT people and retains unsupported
rows in its output so model comparisons can use the same support explicitly.

The rare-identity transform applies an already frozen bijection over the full
scientific territory for one example (person, hidden cell and effective model
reach).  It never generates a permutation from a technical fragment.  A
reader that wants to stream fragments must first implement and verify stable
locus-key projection of this same frozen plan; this module intentionally fails
when only a different or incomplete territory is supplied.

Neither control chooses a radius, exclusion margin, pseudocount, mask schedule,
split, model, seed, learning budget or stopping rule.  It does not authorize
training or access EVALUATE.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np
import torch
from torch import Tensor

from r02_masked_genotype_model import MaskedTensorChannel
from r02_masked_genotype_update import MaskedFragment, MaskedTargets


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _checked_channel(channel: MaskedTensorChannel, expected: str) -> None:
    _require(isinstance(channel, MaskedTensorChannel), "masked tensor channel required")
    _require(channel.channel == expected, f"{expected.upper()} channel required")
    channel.check(max(1, channel.dosage.shape[1]))


def locus_keys(channel: MaskedTensorChannel) -> tuple[str, ...]:
    """Return stable chromosome/bp/REF/ALT/counted-allele identities."""
    fields = (channel.chromosome, channel.position_bp, channel.ref, channel.alt,
              channel.counted_allele)
    rows = zip(*(value.detach().cpu().tolist() for value in fields))
    keys = tuple(":".join(map(str, row)) for row in rows)
    _require(len(keys) == len(set(keys)), "locus identity is not unique in territory")
    return keys


@dataclass(frozen=True)
class FlankWindow:
    """Allowed genetic-distance ring outside one hidden cell, in centimorgans."""

    target_cell_index: int
    cell_left_cm: float
    cell_right_cm: float
    min_distance_cm: float
    max_distance_cm: float

    def __post_init__(self) -> None:
        _require(type(self.target_cell_index) is int and self.target_cell_index >= 0,
                 "nonnegative target cell index required")
        values = (self.cell_left_cm, self.cell_right_cm,
                  self.min_distance_cm, self.max_distance_cm)
        _require(all(type(value) in (int, float) and math.isfinite(value)
                     for value in values), "finite flank coordinates/distances required")
        _require(self.cell_right_cm > self.cell_left_cm,
                 "hidden cell must have positive genetic width")
        _require(0 <= self.min_distance_cm < self.max_distance_cm,
                 "flank ring requires 0 <= minimum < maximum distance")


@dataclass(frozen=True)
class FlankMarkers:
    window: FlankWindow
    left_locus_key: str
    right_locus_key: str


def select_common_flanks(common: MaskedTensorChannel, window: FlankWindow) -> FlankMarkers:
    """Choose fixed COMMON loci without inspecting genotypes or missingness."""
    _checked_channel(common, "common")
    _require(isinstance(window, FlankWindow), "FlankWindow required")
    valid = common.locus_valid.detach().cpu().numpy().astype(bool, copy=False)
    cm = common.position_cm.detach().cpu().numpy()
    bp = common.position_bp.detach().cpu().numpy()
    cells = common.cell_index.detach().cpu().numpy()
    left_distance = window.cell_left_cm - cm
    right_distance = cm - window.cell_right_cm
    left = np.flatnonzero(valid & (cells != window.target_cell_index)
        & (left_distance > 0) & (left_distance >= window.min_distance_cm)
        & (left_distance <= window.max_distance_cm))
    right = np.flatnonzero(valid & (cells != window.target_cell_index)
        & (right_distance > 0) & (right_distance >= window.min_distance_cm)
        & (right_distance <= window.max_distance_cm))
    _require(len(left) > 0 and len(right) > 0,
             "no structural COMMON marker on both sides inside admitted flank ring")
    # Genetic-map plateaus are legitimate.  Physical position breaks equal-cM
    # ties without using dosage, carrier status or person-specific missingness.
    left_index = int(max(left, key=lambda i: (float(cm[i]), int(bp[i]))))
    right_index = int(min(right, key=lambda i: (float(cm[i]), int(bp[i]))))
    keys = locus_keys(common)
    return FlankMarkers(window, keys[left_index], keys[right_index])


def _key_index(channel: MaskedTensorChannel, key: str) -> int:
    keys = locus_keys(channel)
    _require(key in keys, "frozen flank locus is absent from supplied territory")
    return keys.index(key)


def _person_mask(mask, batch: int, name: str, device) -> Tensor:
    value = torch.as_tensor(mask, dtype=torch.bool, device=device)
    _require(value.shape == (batch,), f"{name} must be one boolean per person")
    return value


def _components(values, batch: int) -> tuple[str, ...]:
    result = tuple(values)
    _require(len(result) == batch and all(isinstance(value, str) and value for value in result),
             "one nonempty component identifier per person required")
    return result


@dataclass(frozen=True)
class FlankFitCounts:
    """FIT counts for target dose conditional on two COMMON flank doses."""

    counts: Tensor  # target locus x 9 ordered flank states x target dose 0/1/2
    target_locus_keys: tuple[str, ...]
    target_channel: str
    target_position_cm: tuple[float, ...]
    target_cell_index: tuple[int, ...]
    flanks: FlankMarkers
    reference_person_ids: tuple[str, ...]
    reference_components: tuple[str, ...]
    reference_role: str
    role_manifest_identity: str
    fit_common_lineage: str
    fit_target_lineage: str
    fit_mask_plan_identity: str

    def __post_init__(self) -> None:
        _require(self.target_channel in {"common", "rare"},
                 "common or rare fitted target channel required")
        loci = len(self.target_locus_keys)
        _require(self.counts.shape == (loci, 9, 3)
                 and self.counts.dtype == torch.int64
                 and bool((self.counts >= 0).all()),
                 "flank counts must be nonnegative int64 locus by 9 by 3")
        _require(len(self.target_position_cm) == loci
                 and all(math.isfinite(value) for value in self.target_position_cm)
                 and len(self.target_cell_index) == loci,
                 "fitted target genetic coordinates/cells differ from locus axis")
        _require(self.reference_role == "FIT" and self.reference_person_ids
                 and self.reference_components,
                 "nonempty FIT reference roster/components required")
        _require(isinstance(self.role_manifest_identity, str)
                 and len(self.role_manifest_identity) == 64
                 and all(value in "0123456789abcdef" for value in self.role_manifest_identity),
                 "role manifest SHA-256 identity required")
        _require(all(isinstance(value, str) and value for value in
                     (self.fit_common_lineage, self.fit_target_lineage)),
                 "authenticated FIT input lineages required")
        _require(len(self.fit_mask_plan_identity) == 64,
                 "FIT mask-plan SHA-256 identity required")

    @property
    def identity(self) -> str:
        payload = {
            "schema": "r02_flank_fit_counts_v1",
            "target_locus_keys": self.target_locus_keys,
            "target_channel": self.target_channel,
            "target_position_cm": self.target_position_cm,
            "target_cell_index": self.target_cell_index,
            "flanks": {
                "window": self.flanks.window.__dict__,
                "left": self.flanks.left_locus_key,
                "right": self.flanks.right_locus_key,
            },
            "reference_person_ids": self.reference_person_ids,
            "reference_components": self.reference_components,
            "reference_role": self.reference_role,
            "role_manifest_identity": self.role_manifest_identity,
            "fit_common_lineage": self.fit_common_lineage,
            "fit_target_lineage": self.fit_target_lineage,
            "fit_mask_plan_identity": self.fit_mask_plan_identity,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                          separators=(",", ":")).encode())
        digest.update(self.counts.detach().cpu().to(torch.int64).numpy().astype("<i8").tobytes())
        return digest.hexdigest()


def fit_flanking_genotype_counts(
    common: MaskedTensorChannel,
    target: MaskedTensorChannel,
    targets: MaskedTargets,
    *,
    window: FlankWindow,
    reference_people,
    person_components,
    reference_role: str,
    role_manifest_identity: str,
) -> FlankFitCounts:
    """Count FIT labels by fixed flank pair; target labels never enter features."""
    _checked_channel(common, "common")
    _checked_channel(target, target.channel)
    _require(target.channel in {"common", "rare"}, "common or rare target channel required")
    targets.check(target)
    _require(common.person_ids == target.person_ids
             and common.plan_identity == target.plan_identity,
             "flanks and targets must share people and exact mask plan")
    _require(reference_role == "FIT", "flanking predictor may be fitted only from FIT")
    batch = common.dosage.shape[0]
    reference = _person_mask(reference_people, batch, "reference_people", common.dosage.device)
    _require(bool(reference.any()), "at least one FIT reference person required")
    components = _components(person_components, batch)
    flanks = select_common_flanks(common, window)
    left = _key_index(common, flanks.left_locus_key)
    right = _key_index(common, flanks.right_locus_key)
    flank_visible = common.genotype_visible[:, left] & common.genotype_visible[:, right]
    pair = common.dosage[:, left].to(torch.int64) * 3 + common.dosage[:, right].to(torch.int64)
    counts = torch.zeros((target.dosage.shape[1], 9, 3), dtype=torch.int64,
                         device=target.dosage.device)
    score = targets.score_mask
    for locus in range(target.dosage.shape[1]):
        use = reference & flank_visible & score[:, locus]
        if bool(use.any()):
            encoded = pair[use] * 3 + targets.dosage[use, locus]
            counts[locus] = torch.bincount(encoded, minlength=27).reshape(9, 3)
    selected = reference.detach().cpu().numpy().astype(bool, copy=False)
    return FlankFitCounts(
        counts=counts,
        target_locus_keys=locus_keys(target),
        target_channel=target.channel,
        target_position_cm=tuple(float(value) for value in
                                 target.position_cm.detach().cpu().tolist()),
        target_cell_index=tuple(int(value) for value in
                                target.cell_index.detach().cpu().tolist()),
        flanks=flanks,
        reference_person_ids=tuple(person for person, keep in zip(common.person_ids, selected) if keep),
        reference_components=tuple(sorted({component for component, keep
                                           in zip(components, selected) if keep})),
        reference_role=reference_role,
        role_manifest_identity=role_manifest_identity,
        fit_common_lineage=common.lineage,
        fit_target_lineage=target.lineage,
        fit_mask_plan_identity=common.plan_identity,
    )


def merge_flanking_genotype_counts(parts) -> FlankFitCounts:
    """Sum disjoint FIT-person batches with identical loci and flank geometry."""
    values = tuple(parts)
    _require(values and all(isinstance(value, FlankFitCounts) for value in values),
             "nonempty FlankFitCounts batches required")
    first = values[0]
    people = []
    for value in values:
        _require(value.target_locus_keys == first.target_locus_keys
                 and value.target_channel == first.target_channel
                 and value.target_position_cm == first.target_position_cm
                 and value.target_cell_index == first.target_cell_index
                 and value.flanks == first.flanks
                 and value.reference_role == "FIT"
                 and value.role_manifest_identity == first.role_manifest_identity,
                 "flank-count batches use different loci, geometry or FIT manifest")
        _require(set(people).isdisjoint(value.reference_person_ids),
                 "FIT person occurs in more than one flank-count batch")
        people.extend(value.reference_person_ids)
    identity = hashlib.sha256(json.dumps(
        [value.identity for value in values], separators=(",", ":")).encode()).hexdigest()
    return FlankFitCounts(
        counts=sum((value.counts for value in values),
                   torch.zeros_like(first.counts)),
        target_locus_keys=first.target_locus_keys,
        target_channel=first.target_channel,
        target_position_cm=first.target_position_cm,
        target_cell_index=first.target_cell_index,
        flanks=first.flanks,
        reference_person_ids=tuple(people),
        reference_components=tuple(sorted({component for value in values
                                           for component in value.reference_components})),
        reference_role="FIT", role_manifest_identity=first.role_manifest_identity,
        fit_common_lineage="merged:" + identity,
        fit_target_lineage="merged:" + identity,
        fit_mask_plan_identity=identity)


def leave_out_flanking_component_counts(
    total: FlankFitCounts,
    component: FlankFitCounts,
    *,
    component_id: str,
    expected_component_person_ids,
) -> FlankFitCounts:
    """Subtract one complete FIT family component from authenticated totals.

    ``total`` and ``component`` must have been counted on the same locus and
    flank axes.  This function never chooses the component from target dosage;
    the caller supplies the authenticated component identifier of the query
    person.  Keeping subtraction here makes the leave-one-family-component-out
    rule identical for COMMON and RARE targets and for every scheduled cell.
    """
    _require(isinstance(total, FlankFitCounts)
             and isinstance(component, FlankFitCounts),
             "total and component FlankFitCounts required")
    _require(isinstance(component_id, str) and component_id,
             "nonempty query component identifier required")
    _require(total.target_locus_keys == component.target_locus_keys
             and total.target_channel == component.target_channel
             and total.target_position_cm == component.target_position_cm
             and total.target_cell_index == component.target_cell_index
             and total.flanks == component.flanks
             and total.reference_role == component.reference_role == "FIT"
             and total.role_manifest_identity == component.role_manifest_identity,
             "component counts use different loci, geometry or FIT manifest")
    _require(component.reference_components == (component_id,),
             "subtracted counts must contain exactly the query component")
    total_people = set(total.reference_person_ids)
    component_people = set(component.reference_person_ids)
    expected_people = set(expected_component_person_ids)
    _require(expected_people and component_people == expected_people,
             "component counts do not contain the complete authenticated family")
    _require(component_people and component_people <= total_people,
             "query component people are absent from complete FIT totals")
    counts = total.counts - component.counts.to(total.counts.device)
    _require(bool((counts >= 0).all()),
             "component counts exceed complete FIT totals")
    remaining_people = tuple(person for person in total.reference_person_ids
                             if person not in component_people)
    remaining_components = tuple(value for value in total.reference_components
                                 if value != component_id)
    _require(remaining_people and remaining_components
             and component_id not in remaining_components,
             "leave-one-component-out reference set is empty or still contains query family")
    identity = hashlib.sha256(json.dumps({
        "schema": "r02_flank_leave_one_component_out_v1",
        "total": total.identity, "component": component.identity,
        "component_id": component_id,
        "expected_component_person_ids": sorted(expected_people),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return FlankFitCounts(
        counts=counts,
        target_locus_keys=total.target_locus_keys,
        target_channel=total.target_channel,
        target_position_cm=total.target_position_cm,
        target_cell_index=total.target_cell_index,
        flanks=total.flanks,
        reference_person_ids=remaining_people,
        reference_components=remaining_components,
        reference_role="FIT",
        role_manifest_identity=total.role_manifest_identity,
        fit_common_lineage="loco:" + identity,
        fit_target_lineage="loco:" + identity,
        fit_mask_plan_identity=identity)


@dataclass(frozen=True)
class FlankPredictions:
    probabilities: Tensor  # NaN outside rows with visible flanks and scored targets
    raw_reference_support: Tensor
    prediction_defined: Tensor
    comparison_mask: Tensor  # defined and the observed flank pair occurred in FIT
    fit_identity: str
    prediction_role: str
    alpha: float


def predict_from_flanking_counts(
    fitted: FlankFitCounts,
    common: MaskedTensorChannel,
    target: MaskedTensorChannel,
    targets: MaskedTargets,
    *,
    query_people,
    person_components,
    prediction_role: str,
    role_manifest_identity: str,
    alpha: float,
) -> FlankPredictions:
    """Predict all requested rows and expose the exact common-support mask."""
    _require(isinstance(fitted, FlankFitCounts), "FlankFitCounts required")
    _checked_channel(common, "common")
    _checked_channel(target, fitted.target_channel)
    targets.check(target)
    _require(prediction_role in {"FIT", "SELECT"},
             "only FIT development or SELECT scoring is open; EVALUATE remains closed")
    _require(type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0,
             "explicit positive finite pseudocount alpha required")
    _require(role_manifest_identity == fitted.role_manifest_identity,
             "role/component manifest identity differs from FIT reference")
    _require(common.person_ids == target.person_ids
             and common.plan_identity == target.plan_identity,
             "flanks and targets must share people and exact mask plan")
    _require(locus_keys(target) == fitted.target_locus_keys,
             "target locus identity/order differs from fitted flank counts")
    _require(tuple(float(value) for value in target.position_cm.detach().cpu().tolist())
             == fitted.target_position_cm
             and tuple(int(value) for value in target.cell_index.detach().cpu().tolist())
             == fitted.target_cell_index,
             "target genetic coordinates or cell assignments differ from fitted flank counts")
    _require(select_common_flanks(common, fitted.flanks.window) == fitted.flanks,
             "COMMON flank geometry differs from fitted predictor")
    batch, loci = target.dosage.shape
    query = _person_mask(query_people, batch, "query_people", target.dosage.device)
    _require(bool(query.any()), "at least one query person required")
    components = _components(person_components, batch)
    query_ids = {person for person, keep in zip(common.person_ids,
                                                query.detach().cpu().tolist()) if keep}
    query_components = {component for component, keep in zip(components,
        query.detach().cpu().tolist()) if keep}
    _require(query_ids.isdisjoint(fitted.reference_person_ids),
             "query person occurs in FIT flank reference")
    _require(query_components.isdisjoint(fitted.reference_components),
             "query component occurs in FIT flank reference")
    left = _key_index(common, fitted.flanks.left_locus_key)
    right = _key_index(common, fitted.flanks.right_locus_key)
    flank_visible = common.genotype_visible[:, left] & common.genotype_visible[:, right]
    pair = common.dosage[:, left].to(torch.int64) * 3 + common.dosage[:, right].to(torch.int64)
    probabilities = torch.full((batch, loci, 3), float("nan"), dtype=torch.float64,
                               device=target.dosage.device)
    support = torch.zeros((batch, loci), dtype=torch.int64, device=target.dosage.device)
    defined = query[:, None] & flank_visible[:, None] & targets.score_mask
    counts = fitted.counts.to(target.dosage.device)
    for locus in range(loci):
        use = defined[:, locus]
        if bool(use.any()):
            selected_counts = counts[locus, pair[use]]
            selected_support = selected_counts.sum(1)
            support[use, locus] = selected_support
            probabilities[use, locus] = ((selected_counts.to(torch.float64) + alpha)
                / (selected_support.to(torch.float64)[:, None] + 3 * alpha))
    comparison = defined & (support > 0)
    return FlankPredictions(probabilities, support, defined, comparison,
                            fitted.identity, prediction_role, float(alpha))


@dataclass(frozen=True)
class RareIdentityPlan:
    """Frozen full-territory dosage-to-locus bijection for each person."""

    territory_id: str
    person_ids: tuple[str, ...]
    mask_plan_identity: str
    locus_keys: tuple[str, ...]
    source_index: np.ndarray
    genotype_visible: np.ndarray

    def __post_init__(self) -> None:
        _require(isinstance(self.territory_id, str) and self.territory_id,
                 "nonempty scientific territory identity required")
        _require(self.person_ids and len(set(self.person_ids)) == len(self.person_ids),
                 "unique person identities required")
        _require(isinstance(self.mask_plan_identity, str)
                 and len(self.mask_plan_identity) == 64,
                 "mask-plan SHA-256 identity required")
        _require(len(self.locus_keys) == len(set(self.locus_keys)),
                 "unique stable locus identities required")
        source = np.array(self.source_index, dtype=np.int64, copy=True)
        _require(source.shape == (len(self.person_ids), len(self.locus_keys)),
                 "one source locus per person and destination locus required")
        _require(np.all((source >= 0) & (source < len(self.locus_keys))) or source.size == 0,
                 "rare identity source index outside territory")
        visible = np.array(self.genotype_visible, dtype=np.bool_, copy=True)
        _require(visible.shape == source.shape,
                 "frozen rare visibility must match person by locus territory")
        _validate_rare_identity_mapping(source, visible)
        source.setflags(write=False)
        visible.setflags(write=False)
        object.__setattr__(self, "source_index", source)
        object.__setattr__(self, "genotype_visible", visible)

    @property
    def visibility_sha256(self) -> str:
        return hashlib.sha256(self.genotype_visible.astype(np.uint8).tobytes()).hexdigest()

    @property
    def identity(self) -> str:
        payload = {"schema": "r02_rare_identity_plan_v1", "territory_id": self.territory_id,
                   "person_ids": self.person_ids, "mask_plan_identity": self.mask_plan_identity,
                   "locus_keys": self.locus_keys, "visibility_sha256": self.visibility_sha256}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                          separators=(",", ":")).encode())
        digest.update(self.source_index.astype("<i8").tobytes())
        return digest.hexdigest()


def _validate_rare_identity_mapping(source: np.ndarray, visible: np.ndarray) -> None:
    """Reject duplicate donors, hidden donors and movement of hidden destinations."""
    expected = np.arange(source.shape[1])
    for person in range(source.shape[0]):
        shown = expected[visible[person]]
        hidden = expected[~visible[person]]
        _require(np.array_equal(np.sort(source[person, visible[person]]), shown),
                 "visible rare loci must map bijectively within the frozen territory")
        _require(np.array_equal(source[person, hidden], hidden),
                 "hidden or missing rare loci must remain fixed and cannot be donors")


def freeze_rare_identity_plan(
    rare: MaskedTensorChannel,
    *,
    territory_id: str,
    source_index,
) -> RareIdentityPlan:
    """Validate a supplied, dose-independent bijection; never generate one here."""
    _checked_channel(rare, "rare")
    source = np.asarray(source_index)
    batch, loci = rare.dosage.shape
    _require(source.dtype.kind in "iu" and source.shape == (batch, loci),
             "integer source_index must match person by locus territory")
    visible = rare.genotype_visible.detach().cpu().numpy()
    expected = np.arange(loci)
    for person in range(batch):
        shown = expected[visible[person]]
        hidden = expected[~visible[person]]
        _require(np.array_equal(np.sort(source[person, visible[person]]), shown),
                 "visible rare loci must map bijectively within the frozen territory")
        _require(np.array_equal(source[person, hidden], hidden),
                 "hidden or missing rare loci must remain fixed and cannot be donors")
    return RareIdentityPlan(territory_id, rare.person_ids, rare.plan_identity,
                            locus_keys(rare), source, visible)


def freeze_hashed_rare_identity_plan(
    rare: MaskedTensorChannel,
    *,
    territory_id: str,
    seed: int,
) -> RareIdentityPlan:
    """Freeze a dose-independent permutation over each person's visible loci.

    Donor loci are ordered by SHA-256 of the fixed seed, complete scientific
    territory, person and stable locus key.  Dosage is deliberately absent.
    Hidden or missing loci remain fixed and never donate.
    """
    _checked_channel(rare, "rare")
    _require(type(seed) is int and seed >= 0, "nonnegative integer seed required")
    _require(isinstance(territory_id, str) and territory_id,
             "nonempty scientific territory identity required")
    keys = locus_keys(rare)
    visible = rare.genotype_visible.detach().cpu().numpy().astype(bool, copy=False)
    source = np.tile(np.arange(len(keys), dtype=np.int64),
                     (len(rare.person_ids), 1))
    for person_index, person_id in enumerate(rare.person_ids):
        shown = np.flatnonzero(visible[person_index])
        donors = sorted(shown.tolist(), key=lambda locus: hashlib.sha256(
            json.dumps(["r02_rare_identity_hash_order_v1", seed, territory_id,
                        person_id, keys[locus]], separators=(",", ":")).encode()
        ).hexdigest())
        source[person_index, shown] = donors
    return freeze_rare_identity_plan(rare, territory_id=territory_id,
                                     source_index=source)


def apply_rare_identity_plan(
    rare: MaskedTensorChannel,
    plan: RareIdentityPlan,
    *,
    territory_id: str,
) -> MaskedTensorChannel:
    """Apply one full-territory plan; incomplete technical fragments are rejected."""
    _checked_channel(rare, "rare")
    _require(isinstance(plan, RareIdentityPlan), "RareIdentityPlan required")
    _require(territory_id == plan.territory_id, "scientific territory identity differs")
    _require(rare.person_ids == plan.person_ids
             and rare.plan_identity == plan.mask_plan_identity,
             "people or mask plan differ from frozen rare identity control")
    _require(locus_keys(rare) == plan.locus_keys,
             "rare identity plan requires the exact complete scientific territory")
    _require(np.array_equal(rare.genotype_visible.detach().cpu().numpy(),
                            plan.genotype_visible),
             "visible/hidden/missing pattern differs from frozen rare identity control")
    # Defensive revalidation makes application safe even if a caller bypassed
    # ``freeze_rare_identity_plan`` or deliberately mutated a frozen array.
    _validate_rare_identity_mapping(np.asarray(plan.source_index),
                                    np.asarray(plan.genotype_visible))
    source = torch.as_tensor(np.array(plan.source_index, copy=True), dtype=torch.int64,
                             device=rare.dosage.device)
    dosage = torch.gather(rare.dosage, 1, source)
    _require(not bool((dosage[~rare.genotype_visible] != 0).any()),
             "rare identity control made hidden or missing dosage visible")
    controlled = replace(rare, dosage=dosage,
                         lineage=rare.lineage + ":rare_identity:" + plan.identity)
    controlled.check(max(1, controlled.dosage.shape[1]))
    return controlled


def rare_identity_fragment_transform(plan_by_territory, territory_key):
    """Return a campaign-loader hook for reviewed, frozen rare permutations.

    ``territory_key`` receives the authenticated region lineage and scheduled
    example and must return the scientific-territory identifier used when the
    plan was frozen.  The hook never constructs, refits or selects a
    permutation.  A missing plan is an error rather than a request to generate
    one from the current technical fragment.
    """
    _require(isinstance(plan_by_territory, dict) and plan_by_territory,
             "nonempty frozen rare-identity plan mapping required")
    _require(callable(territory_key), "territory_key callable required")

    def transform(*, fragment, lineage, example):
        _require(isinstance(fragment, MaskedFragment), "MaskedFragment required")
        territory = territory_key(lineage, example)
        _require(isinstance(territory, str) and territory in plan_by_territory,
                 "scheduled scientific territory lacks a frozen rare-identity plan")
        plan = plan_by_territory[territory]
        controlled = apply_rare_identity_plan(fragment.rare, plan,
                                              territory_id=territory)
        return replace(fragment, rare=controlled,
                       lineage=fragment.lineage + ":rare_identity:" + plan.identity)

    return transform
