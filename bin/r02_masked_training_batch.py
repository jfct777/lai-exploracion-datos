"""Join authenticated chromosome data to one masked-cell training example.

One cell is the prediction unit, independently of genotype carriage. Callers
enumerate people/cells prospectively and group examples into optimizer updates;
this function does not choose that schedule or grant scientific admission.
"""
from dataclasses import replace
import hashlib
import json
import math

import numpy as np

from r02_masked_genotypes import CellMaskPlan, require
from r02_masked_genotype_model import MaskedTensorChannel
from r02_masked_genotype_update import MaskedFragment, MaskedTargets


def load_masked_cell(reader, model_config, *, person_ids, cell_index,
                     context_exclusion_cm, device):
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
    plan = CellMaskPlan(tuple(person_ids), reader.chromosome, reader.edges, selected)
    halo = max(model_config.required_halo_cm(name) for name in ('common', 'rare'))
    result = reader.read(plan, core_start_cell=cell_index, core_stop_cell=cell_index+1,
                         halo_cm=halo)
    lineage = {**result['lineage'], 'context_exclusion_cm': context_exclusion_cm,
               'mask_schedule': 'ONE_CELL_PER_EXAMPLE_INDEPENDENT_OF_GENOTYPES',
               'halo_equation': 'max(required_halo_cm(common),required_halo_cm(rare))'}
    identity = hashlib.sha256(json.dumps(lineage, sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()
    channels, targets = {}, {}
    for name, masked in result['channels'].items():
        if context_exclusion_cm:
            left, right = reader.edges[cell_index]
            remove = ((masked.inputs['position_cm'] >= left-context_exclusion_cm)
                      & (masked.inputs['position_cm'] <= right+context_exclusion_cm))
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
