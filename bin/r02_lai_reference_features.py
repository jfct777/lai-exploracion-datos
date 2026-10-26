#!/usr/bin/env python3
"""Validate fixed LAI reference counts and make their 15 explicit features."""
import hashlib
import json

ANCESTRIES = ("African", "European", "Native_American")
FEATURES_PER_ANCESTRY = ("AC_0", "AN_0", "AF_0", "KNOWN", "FREQUENCY_DEFINED")
FEATURE_COUNT = len(ANCESTRIES) * len(FEATURES_PER_ANCESTRY)
SCHEMA = "r02_lai_event_reference_features_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def _rows_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _axis_sha256(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def reference_axes_identity(data, np):
    """Hash exact person/role and locus/allele axes after validating their join."""
    people = np.asarray(data["person_id"]).astype(str)
    roles = np.asarray(data["role"]).astype(str)
    components = np.asarray(data["component_id"]).astype(str)
    require(people.ndim == roles.ndim == components.ndim == 1
            and len(people) == len(roles) == len(components),
            "LAI person/role/component axes differ")
    require(all(people) and all(roles) and all(components), "empty LAI person/role/component identity")

    padding = np.asarray(data["event_padding"])
    chrom = np.asarray(data["event_chrom"])
    position = np.asarray(data["event_pos_bp"])
    require(padding.dtype == np.dtype(bool) and chrom.shape == position.shape == padding.shape
            and padding.ndim == 2 and padding.shape[0] == len(people)
            and np.array_equal(padding, np.broadcast_to(padding[:1], padding.shape))
            and np.array_equal(chrom, np.broadcast_to(chrom[:1], chrom.shape))
            and np.array_equal(position, np.broadcast_to(position[:1], position.shape)),
            "LAI person/locus coordinate axes differ")
    keep = ~padding[0]
    ref = np.asarray(data["event_ref"]).astype(str)
    alt = np.asarray(data["event_alt"]).astype(str)
    counted = np.asarray(data["event_counted_allele"]).astype(str)
    is_alt = np.asarray(data["event_is_alt"])
    empty_padding = padding.shape[1] == 1 and bool(padding[0, 0]) and all(
        value.shape == (0,) for value in (ref, alt, counted, is_alt))
    require(empty_padding or ref.shape == alt.shape == counted.shape == is_alt.shape == (padding.shape[1],),
            "LAI REF/ALT/orientation locus axes differ")
    rows = [] if empty_padding else [
        [int(c), int(p), r, a, allele, bool(orientation)]
        for c, p, r, a, allele, orientation in
        zip(chrom[0, keep], position[0, keep], ref[keep], alt[keep], counted[keep], is_alt[keep])]
    join_keys = [tuple(row[:4]) for row in rows]
    require(len(join_keys) == len(set(join_keys)), "duplicate LAI chromosome/position/REF/ALT key")
    return {
        "person_axis_sha256": _axis_sha256(people.tolist()),
        "person_role_component_sha256": _rows_sha256(
            [[person, role, component] for person, role, component in zip(people, roles, components)]),
        "event_reference_join_sha256": _rows_sha256(rows),
        "reference_duplicate_keys": 0,
    }


def reference_features(data, np):
    """Return [people,event,15] without treating unknown counts as frequency zero."""
    required = {"event_reference_ac", "event_reference_an", "event_reference_known",
                "event_reference_ancestry_order", "event_counted_allele", "event_ref",
                "event_alt", "event_is_alt", "event_global_index", "event_padding"}
    require(required <= set(data), "LAI EVENT reference arrays incomplete")
    ac = np.asarray(data["event_reference_ac"])
    an = np.asarray(data["event_reference_an"])
    known = np.asarray(data["event_reference_known"])
    require(ac.dtype == np.dtype(np.int64) and an.dtype == np.dtype(np.int64)
            and known.dtype == np.dtype(bool), "LAI reference count dtypes differ")
    require(ac.ndim == 2 and ac.shape == an.shape == known.shape
            and ac.shape[1] == len(ANCESTRIES), "LAI reference count axes differ")
    require(np.asarray(data["event_reference_ancestry_order"]).astype(str).tolist() == list(ANCESTRIES),
            "LAI reference ancestry order differs")
    events = ac.shape[0]
    locus_lengths = {name: np.asarray(data[name]).shape for name in
                     ("event_counted_allele", "event_ref", "event_alt", "event_is_alt",
                      "event_global_index")}
    ref = np.asarray(data["event_ref"]).astype(str)
    alt = np.asarray(data["event_alt"]).astype(str)
    counted = np.asarray(data["event_counted_allele"]).astype(str)
    is_alt = np.asarray(data["event_is_alt"])
    padding_grid = np.asarray(data["event_padding"])
    require(is_alt.dtype == np.dtype(bool) and padding_grid.dtype == np.dtype(bool)
            and padding_grid.shape == np.asarray(data["event_valid"]).shape
            and np.array_equal(padding_grid, np.broadcast_to(padding_grid[:1], padding_grid.shape)),
            "LAI EVENT orientation/padding dtypes differ")
    padding = padding_grid[0]
    empty_padding = events == 1 and bool(padding[0]) and all(shape == (0,) for shape in locus_lengths.values())
    if empty_padding:
        ref = alt = counted = np.asarray([""])
        is_alt = np.asarray([False])
    else:
        require(all(shape == (events,) for shape in locus_lengths.values()),
                "LAI EVENT locus axis differs")
    from r02_lai_event_annotations import counted_is_alt
    orientation = np.zeros(events, dtype=bool)
    for index in np.flatnonzero(~padding):
        orientation[index] = counted_is_alt(str(ref[index]), str(alt[index]), str(counted[index]))
    require(np.array_equal(orientation[~padding], is_alt[~padding]),
            "LAI EVENT counted allele differs from source_minor orientation")
    require(bool(((ac[known] >= 0) & (an[known] >= 0) & (ac[known] <= an[known])).all())
            and bool((ac[~known] == -1).all()) and bool((an[~known] == -1).all()),
            "LAI reference count mask differs")
    require(not known[padding].any(), "padding EVENT cannot contain reference evidence")
    ac0 = np.where(known, ac, 0)
    an0 = np.where(known, an, 0)
    defined = known & (an > 0)
    af0 = np.divide(ac0, an0, out=np.zeros(ac.shape, np.float64), where=defined)
    locus = np.stack((ac0, an0, af0, known, defined), axis=-1).reshape(events, FEATURE_COUNT)
    people = np.asarray(data["event_valid"]).shape[0]
    require(np.asarray(data["event_valid"]).shape[1] == events, "EVENT mask/reference locus axes differ")
    return np.broadcast_to(locus.astype(np.float32), (people, events, FEATURE_COUNT)).copy()
