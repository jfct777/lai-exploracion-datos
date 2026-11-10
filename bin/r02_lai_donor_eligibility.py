"""Compute a partition-fixed reference mask from explicit original units.

This operation uses no mosaic genotypes, realized ancestry or model outputs.
It removes identities and documented relationships, not all possible kinship.
The caller must authenticate units/relationships and justify the upstream policy;
the returned mask does not authorize training or certify donor independence.
"""
from collections import Counter

from r02_lai_role_units import require


def partition_reference_mask(reference_ids, reference_units, reference_panels,
                             source_pool_units, related_unit_pairs):
    """Return a 1-D mask shared by every mosaic from the declared source pool.

    Relationships are unordered original-unit pairs, NOT sample IDs or inferred
    ancestry labels. A relation involving a unit outside both supplied axes is
    rejected so a mismatched provenance table cannot silently yield no exclusions.
    No relationship threshold or transitive biological kinship is inferred here.
    """
    ids, units, panels = map(list, (reference_ids, reference_units, reference_panels))
    source = list(source_pool_units)
    require(ids and len(ids) == len(units) == len(panels), 'reference axes differ or are empty')
    require(all(isinstance(x, str) and x for x in ids + units + panels + source),
            'nonempty explicit identities, units and panels required')
    require(len(set(ids)) == len(ids), 'duplicate reference sample identity')
    require(source and len(set(source)) == len(source), 'source pool empty or duplicated')
    require(set(panels) <= {'African', 'European', 'Native_American'}, 'unknown reference panel')
    source = set(source)
    known = source | set(units)
    linked = set()
    relationships = set()
    for pair in related_unit_pairs:
        require(isinstance(pair, (tuple, list)) and len(pair) == 2,
                'each relationship needs two explicit original units')
        a, b = pair
        require(isinstance(a, str) and isinstance(b, str) and a != b
                and a in known and b in known, 'relationship unit absent or self-paired')
        edge = tuple(sorted((a, b)))
        require(edge not in relationships, 'duplicate relationship, including reversed pair')
        relationships.add(edge)
        if a in source:
            linked.add(b)
        if b in source:
            linked.add(a)
    allowed = [unit not in source and unit not in linked for unit in units]
    reasons = ['IDENTITY' if unit in source else 'DOCUMENTED_RELATIONSHIP'
               if unit in linked else 'NOT_EXCLUDED_BY_SUPPLIED_RELATIONSHIPS' for unit in units]
    before = Counter(panels)
    after = Counter(panel for panel, keep in zip(panels, allowed) if keep)
    return {'allowed': allowed, 'reasons': reasons,
            'reference_samples_before_by_panel': dict(before),
            'reference_samples_after_by_panel': {p: after[p] for p in before},
            'eligible_source_units': len(source), 'relationship_pairs_supplied': len(relationships),
            'reference_original_units_before': len(set(units)),
            'reference_original_units_after': len({u for u, keep in zip(units, allowed) if keep}),
            'partition_fixed': True, 'training_authorized': False,
            'family_independence_certified': False}
