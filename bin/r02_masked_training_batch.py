"""Join authenticated chromosome data to one masked-cell training example.

One cell is the prediction unit, independently of genotype carriage. Callers
enumerate people/cells prospectively and group examples into optimizer updates;
this function does not choose that schedule or grant scientific admission.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np

from r02_masked_genotypes import CellMaskPlan, require
from r02_masked_genotype_model import MaskedTensorChannel
from r02_masked_genotype_update import MaskedFragment, MaskedTargets


@dataclass(frozen=True)
class MaskedCellPlan:
    """One prospective cell mask and its exact locus-level hidden territory."""

    mask: CellMaskPlan
    cell_index: int
    halo_cm: float
    hidden_left_cm: float
    hidden_right_cm: float
    context_exclusion_cm: float


def plan_masked_cell(reader, model_config, *, person_ids, cell_index,
                     context_exclusion_cm):
    """Build the geometry shared by the loader and carrier-context adapter."""
    require(type(cell_index) is int and 0 <= cell_index < len(reader.edges),
            'cell index outside declared axis')
    require(type(context_exclusion_cm) in (int, float)
            and math.isfinite(context_exclusion_cm) and context_exclusion_cm >= 0,
            'explicit finite nonnegative context exclusion in cM required')
    require(context_exclusion_cm < min(model_config.common_radius_cm,
                                       model_config.rare_radius_cm),
            'context exclusion must leave a visible outer neighborhood')
    selected = np.zeros((len(person_ids), len(reader.edges)), bool)
    selected[:, cell_index] = True
    mask = CellMaskPlan(tuple(person_ids), reader.chromosome, reader.edges, selected)
    halo = max(model_config.required_halo_cm(name) for name in ('common', 'rare'))
    left, right = reader.edges[cell_index]
    return MaskedCellPlan(mask, cell_index, float(halo),
                          float(left-context_exclusion_cm),
                          float(right+context_exclusion_cm),
                          float(context_exclusion_cm))


def hidden_locus_mask(position_cm, plan):
    """Return the same structural territory hidden by ``load_masked_cell``."""
    require(isinstance(plan, MaskedCellPlan), 'MaskedCellPlan required')
    position_cm = np.asarray(position_cm)
    require(position_cm.ndim == 1 and np.issubdtype(position_cm.dtype, np.number)
            and np.isfinite(position_cm).all(),
            'one finite genetic coordinate per locus required')
    if plan.context_exclusion_cm == 0:
        # CellMaskPlan is canonical here: internal cells are half-open and the
        # last cell owns the chromosome endpoint. Repeated cM positions must
        # receive the same searchsorted(side="right") assignment.
        indices = plan.mask.indices(position_cm)
        require(np.all(plan.mask.selected == plan.mask.selected[:1]),
                'one shared cell selection required for locus exclusions')
        return plan.mask.selected[0, indices].copy()
    # Preserve the historical explicit-margin rule, which closed both outer
    # boundaries after the base cell mask had already been applied.
    return ((position_cm >= plan.hidden_left_cm)
            & (position_cm <= plan.hidden_right_cm))


def hidden_locus_ids(channel_inputs, chromosome, plan):
    """Encode hidden loci exactly as the streamed COMMON producer does."""
    require(type(chromosome) is int and 1 <= chromosome <= 22,
            'autosomal chromosome required')
    required = {'position_cm', 'position_bp', 'ref', 'alt'}
    require(required <= set(channel_inputs), 'locus identity metadata missing')
    hidden = hidden_locus_mask(channel_inputs['position_cm'], plan)
    bases = np.asarray(tuple('ACGT'))
    bp = np.asarray(channel_inputs['position_bp'])
    ref = np.asarray(channel_inputs['ref'])
    alt = np.asarray(channel_inputs['alt'])
    require(bp.shape == ref.shape == alt.shape == hidden.shape
            and np.all((ref >= 0) & (ref < 4))
            and np.all((alt >= 0) & (alt < 4)), 'locus identity axes differ')
    return tuple(f'{chromosome}:{int(bp[i])}:{bases[int(ref[i])]}:{bases[int(alt[i])]}'
                 for i in np.flatnonzero(hidden))


def load_masked_cell(reader, model_config, *, person_ids, cell_index,
                     context_exclusion_cm, device):
    planned = plan_masked_cell(reader, model_config, person_ids=person_ids,
                               cell_index=cell_index,
                               context_exclusion_cm=context_exclusion_cm)
    result = reader.read(planned.mask, core_start_cell=cell_index,
                         core_stop_cell=cell_index+1, halo_cm=planned.halo_cm)
    lineage = {**result['lineage'], 'context_exclusion_cm': context_exclusion_cm,
               'hidden_territory_cm': [planned.hidden_left_cm,
                                       planned.hidden_right_cm],
               'hidden_territory_boundary_rule': (
                   'HALF_OPEN_CELLMASKPLAN_CLOSED_LAST_CELL'
                   if context_exclusion_cm == 0 else 'CLOSED_EXPLICIT_MARGIN'),
               'mask_schedule': 'ONE_CELL_PER_EXAMPLE_INDEPENDENT_OF_GENOTYPES',
               'halo_equation': 'max(required_halo_cm(common),required_halo_cm(rare))'}
    identity = hashlib.sha256(json.dumps(lineage, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    channels, targets = {}, {}
    for name, masked in result['channels'].items():
        if context_exclusion_cm:
            remove = hidden_locus_mask(masked.inputs['position_cm'], planned)
            inputs = dict(masked.inputs)
            inputs['genotype_visible'] = masked.inputs['genotype_visible'].copy()
            inputs['dosage'] = masked.inputs['dosage'].copy()
            inputs['genotype_visible'][:, remove] = False
            inputs['dosage'][:, remove] = 0
            masked = replace(masked, inputs=inputs)
        channels[name] = MaskedTensorChannel.from_masked_channel(masked,
            person_ids=tuple(person_ids), lineage=identity+':'+name, device=device)
        targets[name] = MaskedTargets.from_masked_channel(masked,
            owned=result['owned'][name], device=device)
    return MaskedFragment(channels['common'], channels['rare'], targets['common'],
                          targets['rare'], identity), lineage
