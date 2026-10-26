#!/usr/bin/env python3
"""Validate FLARE reference and simulated-target roles by original unit."""
from __future__ import annotations

import csv
from collections import defaultdict


def require(condition, message):
    if not condition:
        raise ValueError(message)


def role_units(path, references, targets, ancestry_names, synthetic):
    units = defaultdict(set); observed = defaultdict(set); ref_ancestry = {}
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        require(set(reader.fieldnames or []) == {'sample_id','role','atomic_unit_id','ancestry'}, 'Explicit original unit table required')
        for row in reader:
            sid, role, unit = row['sample_id'], row['role'], row['atomic_unit_id']
            require(sid and unit and role in {'REF_TRAIN','SIMULATED_TARGET'}, 'Unknown role or absent original unit')
            if synthetic: require(sid.startswith('SYNTH_'), 'Synthetic scope requires synthetic samples')
            require(unit not in observed[sid], 'Duplicate sample/unit record')
            observed[sid].add(unit); units[role].add(unit)
            require(sid in (references if role == 'REF_TRAIN' else targets), 'Role differs from VCF sample sets')
            require(row['ancestry'] in ancestry_names, 'Unknown donor/reference ancestry')
            if role == 'REF_TRAIN':
                require(sid not in ref_ancestry, 'Each reference needs exactly one original unit')
                ref_ancestry[sid] = row['ancestry']
    require(set(observed) == set(references)|set(targets), 'Missing original units for a sample')
    require(units['REF_TRAIN'].isdisjoint(units['SIMULATED_TARGET']), 'Original donor unit crosses reference/target boundary')
    return ref_ancestry, {role:len(value) for role,value in units.items()}
