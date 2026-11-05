"""Prepare authenticated human tensors for the pair-free separable model.

The adapter reuses the masked-cell reader and the streamed COMMON producer.  It
does not choose an EVENT radius, model capacity, decay length or training
policy.  A caller must supply an externally adopted, internally authenticated
territory receipt.  The returned tensors remain non-admitting: they can support
a technical forward call but cannot authorize human training.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import NamedTuple, Sequence, Tuple

import numpy as np
import torch

from r02_carrier_common_context import (
    CHANNELS, EXCLUDE_SAME_COMPONENT, CommonContextPolicy,
    CommonContextTechnicalLimits, CommonSegment,
    produce_from_raw_masked_reader,
)
from r02_masked_carrier_batch import (
    SOURCE_MINOR, CarrierFragmentBatch, channel_locus_ids,
)
from r02_masked_genotype_model import MaskedModelConfig
from r02_masked_genotype_update import MaskedFragment
from r02_masked_genotypes import CellMaskPlan
from r02_masked_separable_integration import SeparableCarrierGeometry
from r02_masked_training_batch import (
    hidden_locus_mask, load_masked_cell, plan_masked_cell,
)


TERRITORY_SCHEMA = "r02_masked_separable_event_territory_v1"
TERRITORY_STATUS = "ADOPTED_EXTERNALLY_NOT_BY_ADAPTER"
SOURCE_KIND = "AUTHENTICATED_MAPPED_EVENT_PREFLIGHT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(name: str, value: str) -> None:
    _require(type(value) is str and len(value) == 64
             and all(letter in "0123456789abcdef" for letter in value),
             f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True)
class FitReferencePanel:
    """Complete ordered FIT panel and identities authenticated elsewhere."""

    reference_ids: Tuple[str, ...]
    reference_components: Tuple[str, ...]
    roster_sha256: str
    role_manifest_sha256: str
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.reference_ids) is tuple and self.reference_ids
                 and len(set(self.reference_ids)) == len(self.reference_ids)
                 and all(type(value) is str and value
                         for value in self.reference_ids),
                 "complete unique ordered FIT reference IDs required")
        _require(type(self.reference_components) is tuple
                 and len(self.reference_components) == len(self.reference_ids)
                 and all(type(value) is str and value
                         for value in self.reference_components),
                 "one nonempty component per FIT reference required")
        for name in ("roster_sha256", "role_manifest_sha256",
                     "source_receipt_sha256"):
            _sha256(name, getattr(self, name))


class HumanSeparableProduct(NamedTuple):
    fragment: MaskedFragment
    carrier_batch: CarrierFragmentBatch
    geometry: SeparableCarrierGeometry
    receipt: dict


def territory_receipt(*, chromosome: int, cell_index: int,
                      cell_left_cm: float, cell_right_cm: float,
                      radius_cm: float, cells_sha256: str,
                      panel: FitReferencePanel, segment_id: str) -> dict:
    """Encode an already adopted radius; this helper does not select one."""
    value = {
        "schema": TERRITORY_SCHEMA,
        "status": TERRITORY_STATUS,
        "chromosome": chromosome,
        "cell_index": cell_index,
        "cell_left_cm": cell_left_cm,
        "cell_right_cm": cell_right_cm,
        "radius_cm": radius_cm,
        "cells_sha256": cells_sha256,
        "roster_sha256": panel.roster_sha256,
        "role_manifest_sha256": panel.role_manifest_sha256,
        "source_receipt_sha256": panel.source_receipt_sha256,
        "segment_id": segment_id,
        "source_minor": SOURCE_MINOR,
        "human_training_admitted": False,
    }
    value["territory_sha256"] = _digest(value)
    return value


def _validate_territory(value, reader, panel: FitReferencePanel,
                        cell_index: int) -> dict:
    _require(isinstance(value, dict),
             "explicit authenticated EVENT territory receipt required")
    required = {
        "schema", "status", "chromosome", "cell_index", "cell_left_cm",
        "cell_right_cm", "radius_cm", "cells_sha256", "roster_sha256",
        "role_manifest_sha256", "source_receipt_sha256", "segment_id",
        "source_minor", "human_training_admitted", "territory_sha256",
    }
    _require(set(value) == required, "exact EVENT territory receipt required")
    identity = value["territory_sha256"]
    _sha256("territory_sha256", identity)
    _require(_digest({key: item for key, item in value.items()
                      if key != "territory_sha256"}) == identity,
             "EVENT territory receipt identity differs")
    _require(value["schema"] == TERRITORY_SCHEMA
             and value["status"] == TERRITORY_STATUS
             and value["source_minor"] == SOURCE_MINOR
             and value["human_training_admitted"] is False,
             "externally adopted non-admitting EVENT territory required")
    _require(value["chromosome"] == reader.chromosome
             and value["cell_index"] == cell_index,
             "EVENT territory chromosome/cell differs")
    _require(value["cells_sha256"] == reader.cells_sha256
             and value["roster_sha256"] == panel.roster_sha256
             and value["role_manifest_sha256"] == panel.role_manifest_sha256
             and value["source_receipt_sha256"] == panel.source_receipt_sha256,
             "EVENT territory source, cell or FIT roster identity differs")
    left, right = reader.edges[cell_index]
    _require(np.float64(value["cell_left_cm"]) == np.float64(left)
             and np.float64(value["cell_right_cm"]) == np.float64(right),
             "EVENT territory cell boundaries differ")
    radius = value["radius_cm"]
    _require(type(radius) in (int, float) and math.isfinite(radius)
             and radius > 0, "explicit finite positive EVENT radius required")
    _require(type(value["segment_id"]) is str and value["segment_id"],
             "nonempty EVENT segment identity required")
    return value


def _validate_panel(reader, panel: FitReferencePanel,
                    target_ids: Tuple[str, ...], block_size: int) -> None:
    _require(isinstance(panel, FitReferencePanel),
             "FitReferencePanel required")
    people = tuple(person for person in reader.reader.persons
                   if person["role"] == "FIT")
    _require(tuple(person["sample_id"] for person in people)
             == panel.reference_ids
             and tuple(person["component"] for person in people)
             == panel.reference_components,
             "complete ordered FIT panel differs from authenticated source")
    _require(set(target_ids) <= set(panel.reference_ids),
             "target lies outside complete FIT panel")
    _require(panel.source_receipt_sha256 == reader.reader.manifest_sha256,
             "FIT panel source receipt differs from authenticated store")
    _require(type(block_size) is int and block_size > 0
             and block_size <= reader.max_people
             and block_size <= reader.reader.r["max_batch_people"],
             "reference block size exceeds authenticated reader limits")


def _same_locus_axis(inputs: dict, channel, name: str) -> None:
    expected = {
        "position_bp": channel.position_bp.detach().cpu().numpy(),
        "position_cm": channel.position_cm.detach().cpu().numpy(),
        "ref": channel.ref.detach().cpu().numpy(),
        "alt": channel.alt.detach().cpu().numpy(),
        "counted_allele": channel.counted_allele.detach().cpu().numpy(),
    }
    _require(all(key in inputs and np.array_equal(np.asarray(inputs[key]), value)
                 for key, value in expected.items()),
             f"{name} locus identity/order differs from masked fragment")


def _read_fit_panel(reader, fragment: MaskedFragment,
                    plan, panel: FitReferencePanel, *, block_size: int,
                    target_ids: Tuple[str, ...]):
    """Read every FIT row in bounded blocks, with no genotype-selected mask."""
    rare_dosage, rare_called = [], []
    target_common = {}
    lineage_receipts = []
    target_set = set(target_ids)
    for start in range(0, len(panel.reference_ids), block_size):
        stop = min(start + block_size, len(panel.reference_ids))
        ids = panel.reference_ids[start:stop]
        empty = CellMaskPlan(ids, reader.chromosome, reader.edges,
                             np.zeros((len(ids), len(reader.edges)), bool))
        result = reader.read(empty, core_start_cell=plan.cell_index,
                             core_stop_cell=plan.cell_index + 1,
                             halo_cm=plan.halo_cm)
        common = result["channels"]["common"].inputs
        rare = result["channels"]["rare"].inputs
        _same_locus_axis(common, fragment.common, "COMMON reference")
        _same_locus_axis(rare, fragment.rare, "EVENT reference")
        rd = np.asarray(rare["dosage"])
        rc = np.asarray(rare["genotype_visible"])
        _require(rd.shape == rc.shape == (len(ids), fragment.rare.dosage.shape[1])
                 and rc.dtype == np.bool_, "EVENT reference block axes differ")
        rare_dosage.append(rd)
        rare_called.append(rc)
        common_dose = np.asarray(common["dosage"])
        common_called = np.asarray(common["genotype_visible"])
        for row, person in enumerate(ids):
            if person in target_set:
                _require(person not in target_common,
                         "target occurred twice in FIT reference panel")
                target_common[person] = (common_dose[row].copy(),
                                         common_called[row].copy())
        lineage_receipts.append(_digest(result["lineage"]))
    _require(set(target_common) == target_set,
             "not every target was recovered from the FIT panel")
    return (np.concatenate(rare_dosage), np.concatenate(rare_called),
            target_common, tuple(lineage_receipts))


def prepare_human_separable_inputs(
        reader, model_config: MaskedModelConfig, *,
        target_ids: Sequence[str], cell_index: int,
        context_exclusion_cm: float, territory: dict,
        panel: FitReferencePanel, reference_block_size: int,
        common_limits: CommonContextTechnicalLimits,
        common_policy: CommonContextPolicy,
        dtype: torch.dtype, device: torch.device | str,
        ) -> HumanSeparableProduct:
    """Prepare one human FIT cell without admitting a forward or training run."""
    targets = tuple(target_ids)
    _require(isinstance(model_config, MaskedModelConfig),
             "MaskedModelConfig required")
    _require(targets and len(set(targets)) == len(targets)
             and all(type(value) is str and value for value in targets),
             "unique nonempty target IDs required")
    _require(dtype in (torch.float32, torch.float64),
             "explicit float32 or float64 dtype required")
    device = torch.device(device)
    _require(isinstance(common_limits, CommonContextTechnicalLimits),
             "CommonContextTechnicalLimits required")
    _require(isinstance(common_policy, CommonContextPolicy)
             and common_policy.family_rule == EXCLUDE_SAME_COMPONENT,
             "COMMON must exclude the target and complete family component")
    adopted = _validate_territory(territory, reader, panel, cell_index)
    _validate_panel(reader, panel, targets, reference_block_size)

    plan = plan_masked_cell(
        reader, model_config, person_ids=targets, cell_index=cell_index,
        context_exclusion_cm=context_exclusion_cm)
    fragment, target_lineage = load_masked_cell(
        reader, model_config, person_ids=targets, cell_index=cell_index,
        context_exclusion_cm=context_exclusion_cm, device=device)
    _require(fragment.common.dosage.shape[1] > 0
             and fragment.rare.dosage.shape[1] > 0,
             "nonempty COMMON and EVENT fragment axes required")

    (reference_dosage_np, reference_called_np, target_common,
     block_lineages) = _read_fit_panel(
        reader, fragment, plan, panel, block_size=reference_block_size,
        target_ids=targets)
    target_rows = np.asarray([panel.reference_ids.index(value)
                              for value in targets], dtype=np.int64)
    target_dosage_np = reference_dosage_np[target_rows]
    target_called_np = reference_called_np[target_rows]
    common_target_dosage = np.stack([target_common[x][0] for x in targets])
    common_target_called = np.stack([target_common[x][1] for x in targets])

    rare_hidden_axis = hidden_locus_mask(
        fragment.rare.position_cm.detach().cpu().numpy(), plan)
    common_hidden_axis = hidden_locus_mask(
        fragment.common.position_cm.detach().cpu().numpy(), plan)
    rare_hidden = np.broadcast_to(rare_hidden_axis,
                                  target_dosage_np.shape).copy()
    common_hidden = np.broadcast_to(common_hidden_axis,
                                    common_target_dosage.shape).copy()

    common_ids = channel_locus_ids(fragment.common)
    excluded = tuple((tuple(common_ids[index]
                            for index in np.flatnonzero(common_hidden[row])),)
                     for row in range(len(targets)))
    common_segment = CommonSegment(
        adopted["segment_id"],
        int(fragment.common.position_bp.min().item()),
        int(fragment.common.position_bp.max().item()))
    common_product = produce_from_raw_masked_reader(
        reader.reader, targets, panel.reference_ids, (common_segment,),
        excluded, common_policy, limits=common_limits,
        dtype=dtype, device=device)

    target_dosage = torch.as_tensor(target_dosage_np, dtype=dtype,
                                    device=device)
    target_called = torch.as_tensor(target_called_np, dtype=torch.bool,
                                    device=device)
    reference_dosage = torch.as_tensor(reference_dosage_np, dtype=dtype,
                                       device=device)
    reference_called = torch.as_tensor(reference_called_np, dtype=torch.bool,
                                       device=device)
    rare_structural_hidden = torch.as_tensor(
        rare_hidden, dtype=torch.bool, device=device)
    common_structural_hidden = torch.as_tensor(
        common_hidden, dtype=torch.bool, device=device)
    segment_id = adopted["segment_id"]
    batch = CarrierFragmentBatch(
        target_dosage=target_dosage, target_called=target_called,
        rare_structural_hidden=rare_structural_hidden,
        rare_natural_missing=~target_called,
        common_structural_hidden=common_structural_hidden,
        common_natural_missing=~torch.as_tensor(
            common_target_called, dtype=torch.bool, device=device),
        reference_dosage=reference_dosage,
        reference_called=reference_called,
        pairs=torch.empty((0, 2), dtype=torch.int64, device=device),
        pair_segment=torch.empty(0, dtype=torch.int64, device=device),
        fit_reference_eligible=common_product.target_reference_eligible,
        reference_roles=("FIT",) * len(panel.reference_ids),
        target_ids=targets, target_roles=("FIT",) * len(targets),
        target_components=tuple(panel.reference_components[index]
                                for index in target_rows),
        reference_ids=panel.reference_ids,
        reference_components=panel.reference_components,
        pair_features=None, locus_ids=channel_locus_ids(fragment.rare),
        segment_ids=(segment_id,),
        common_query_segment=torch.zeros(
            fragment.common.dosage.shape[1], dtype=torch.int64, device=device),
        rare_query_segment=torch.zeros(
            fragment.rare.dosage.shape[1], dtype=torch.int64, device=device),
        common_context=common_product.context,
        source_minor_definition=SOURCE_MINOR,
        role_manifest_sha256=panel.role_manifest_sha256,
        source_receipt_sha256=panel.source_receipt_sha256,
        source_kind=SOURCE_KIND, human_training_authorized=False)

    positions = fragment.rare.position_cm.to(dtype=dtype)
    valid = fragment.rare.locus_valid
    hidden_axis = rare_structural_hidden[0]
    _require(bool((rare_structural_hidden == hidden_axis).all()),
             "one genotype-independent structural EVENT territory required")
    left_edge, right_edge = (adopted["cell_left_cm"],
                             adopted["cell_right_cm"])
    radius = adopted["radius_cm"]
    left_flank = (valid & ~hidden_axis
                  & (positions >= left_edge - radius)
                  & (positions < left_edge))
    right_flank = (valid & ~hidden_axis
                   & (positions > right_edge)
                   & (positions <= right_edge + radius))
    axis_id = _digest({"locus_ids": batch.locus_ids,
                       "cells_sha256": adopted["cells_sha256"],
                       "segment_id": segment_id})
    geometry = SeparableCarrierGeometry(
        positions_cm=positions,
        left_flank=left_flank.unsqueeze(0),
        right_flank=right_flank.unsqueeze(0),
        context_id=adopted["territory_sha256"], axis_id=axis_id)
    batch.check(fragment, common_context_width=len(CHANNELS))
    geometry.check(fragment.rare, batch)

    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    reference_cells = len(panel.reference_ids) * len(batch.locus_ids)
    target_common_cells = len(targets) * fragment.common.dosage.shape[1]
    # The reader bounds each source block, while the model API necessarily
    # retains the complete R x L EVENT panel.  This estimate includes the two
    # block-list arrays plus their concatenated arrays, final float dosage and
    # boolean callability, and the recovered target COMMON arrays.  It excludes
    # temporary arrays already bounded inside MaskedStoreReader.read.
    adapter_peak_bytes_estimate = (
        reference_cells * (4 + dtype_bytes + 1)
        + target_common_cells * 4
        + len(targets) * len(batch.locus_ids) * (dtype_bytes + 1))

    receipt = {
        "schema": "r02_masked_separable_human_product_v1",
        "status": "PREPARED_NOT_FORWARD_NOT_TRAINING_ADMISSION",
        "territory_sha256": adopted["territory_sha256"],
        "roster_sha256": panel.roster_sha256,
        "source_receipt_sha256": panel.source_receipt_sha256,
        "target_lineage_sha256": _digest(target_lineage),
        "reference_block_lineage_sha256": list(block_lineages),
        "reference_block_size": reference_block_size,
        "reference_count": len(panel.reference_ids),
        "target_count": len(targets),
        "event_locus_count": len(batch.locus_ids),
        "left_flank_locus_count": int(left_flank.sum()),
        "right_flank_locus_count": int(right_flank.sum()),
        "common_context_receipt_sha256": _digest(common_product.receipt),
        "event_locus_axis_sha256": _digest(batch.locus_ids),
        "reference_axis_sha256": _digest(batch.reference_ids),
        "carrier_batch_identities": batch.identities(),
        "memory_estimate": {
            "complete_reference_panel_tensor_bytes":
                reference_cells * (dtype_bytes + 1),
            "adapter_peak_bytes_excluding_reader_temporaries":
                adapter_peak_bytes_estimate,
            "reference_blocks_bound_source_reads_only": True,
            "complete_reference_panel_accumulated_for_model": True,
            "complete_panel_memory_limit_enforced": False,
        },
        "dtype": str(dtype), "device": str(device),
        "human_training_admitted": False,
        "neural_forward_performed": False,
    }
    return HumanSeparableProduct(fragment, batch, geometry, receipt)
