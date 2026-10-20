"""Validate and encode versioned LAI EVENT locus annotations.

These arrays remain separate from the twelve learned EVENT metadata features.
Missing fields use a sentinel plus an explicit mask; a known AC=0, AN=0 pair
remains known and has no frequency computed here.
"""
from __future__ import annotations

import re


SCHEMA = "r02_lai_event_annotations_v1"
LEGACY_SCHEMA = "LEGACY_EVENT_ANNOTATIONS_NOT_AVAILABLE"
ANCESTRIES = ("African", "European", "Native_American")
PROVENANCE_ROLES = ("ref_train", "source_valid")
PROVENANCE_COUNTS = ("called_copies", "original_observed_copies", "completed_copies")
REVISED_ONLY_FIELDS = (
    "event_origin",
    "ref_train_called_copies", "ref_train_original_observed_copies", "ref_train_completed_copies",
    "source_valid_called_copies", "source_valid_original_observed_copies", "source_valid_completed_copies",
    "phase_status", "quality_status",
)
REQUIRED_FIELDS = {
    "ref", "alt", "counted_allele", "copyable_in_case_fold", "source_catalogue_sha256",
    *REVISED_ONLY_FIELDS,
    *(f"reference_{ancestry}_{suffix}" for ancestry in ANCESTRIES for suffix in ("ac", "an")),
}
TYPED_ARRAY_FIELDS = (
    "event_reference_ac", "event_reference_an", "event_reference_known",
    "event_call_provenance_counts", "event_call_provenance_known",
    "event_origin", "event_phase_status", "event_quality_status",
    "event_source_catalogue_sha256", "event_counted_allele",
    "event_copyable_in_case_fold", "event_copyable_known",
    "event_reference_ancestry_order", "event_call_provenance_role_order",
    "event_call_provenance_count_order", "event_annotation_schema",
)
NON_LOCUS_ARRAY_FIELDS = (
    "event_reference_ancestry_order", "event_call_provenance_role_order",
    "event_call_provenance_count_order", "event_annotation_schema",
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(value, message):
    if not value:
        raise ValueError(message)


def counted_is_alt(ref, alt, counted):
    """Return orientation after accepting only explicit labels or literal alleles."""
    require(ref in "ACGT" and alt in "ACGT" and len(ref) == len(alt) == 1 and ref != alt,
            "Expected literal biallelic SNV alleles")
    if counted == "ALT" or counted == alt:
        return True
    if counted == "REF" or counted == ref:
        return False
    raise ValueError("Counted allele is neither REF/ALT nor a matching literal allele")


def catalogue_annotation_schema(fields):
    """Distinguish a complete revised extension from an unextended legacy table."""
    fields = set(fields)
    present = fields.intersection(REVISED_ONLY_FIELDS)
    if not present:
        return LEGACY_SCHEMA
    require(set(REVISED_ONLY_FIELDS) <= fields, "Partial revised EVENT annotation extension")
    require(REQUIRED_FIELDS <= fields, "Revised EVENT annotation columns incomplete")
    return SCHEMA


def _nullable_nonnegative(value, label):
    if value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid integer {label}") from None
    require(parsed >= 0, f"Negative {label}")
    return parsed


def count_pair(ac_value, an_value, label):
    """Parse AC/AN without conflating absent values and known zero coverage."""
    ac = _nullable_nonnegative(ac_value, label + " AC")
    an = _nullable_nonnegative(an_value, label + " AN")
    require((ac is None) == (an is None), f"Partial {label} AC/AN pair")
    if ac is None:
        return -1, -1, False
    require(ac <= an, f"{label} AC exceeds AN")
    return ac, an, True


def provenance_triplet(values, label):
    parsed = [_nullable_nonnegative(value, f"{label} {name}")
              for value, name in zip(values, PROVENANCE_COUNTS)]
    missing = [value is None for value in parsed]
    require(all(missing) or not any(missing), f"Partial {label} provenance triplet")
    if all(missing):
        return (-1, -1, -1), False
    called, original, completed = parsed
    require(called == original + completed, f"{label} provenance copies do not add up")
    return (called, original, completed), True


def nullable_bool(value, label):
    if value == "":
        return -1, False
    require(value in {"True", "False"}, f"Invalid {label} boolean")
    return int(value == "True"), True


def encode_rows(fields, rows):
    """Validate rows and return plain-Python values before array allocation."""
    require(catalogue_annotation_schema(fields) == SCHEMA,
            "Typed annotations require the complete revised EVENT schema")
    reference_ac, reference_an, reference_known = [], [], []
    provenance, provenance_known = [], []
    copyable, copyable_known = [], []
    origins, phases, qualities, catalogues, counted_values = [], [], [], [], []
    for row in rows:
        counted_is_alt(row["ref"], row["alt"], row["counted_allele"])
        row_ac, row_an, row_known = [], [], []
        for ancestry in ANCESTRIES:
            ac, an, known = count_pair(row[f"reference_{ancestry}_ac"],
                                       row[f"reference_{ancestry}_an"],
                                       "reference " + ancestry)
            row_ac.append(ac); row_an.append(an); row_known.append(known)
        reference_ac.append(row_ac); reference_an.append(row_an); reference_known.append(row_known)
        row_provenance, row_provenance_known = [], []
        for role in PROVENANCE_ROLES:
            values = tuple(row[f"{role}_{name}"] for name in PROVENANCE_COUNTS)
            parsed, known = provenance_triplet(values, role)
            row_provenance.append(parsed); row_provenance_known.append(known)
        provenance.append(row_provenance); provenance_known.append(row_provenance_known)
        value, known = nullable_bool(row["copyable_in_case_fold"], "copyable_in_case_fold")
        copyable.append(value); copyable_known.append(known)
        require(row["event_origin"] in {"ORIGINAL_ALT_EVENT", "REVIEWED_REF_EXTENSION"},
                "Unknown EVENT origin")
        require(bool(row["phase_status"]) and bool(row["quality_status"]),
                "Empty phase or quality status")
        require(bool(SHA256.fullmatch(row["source_catalogue_sha256"])),
                "Invalid source catalogue SHA256")
        origins.append(row["event_origin"]); phases.append(row["phase_status"])
        qualities.append(row["quality_status"]); catalogues.append(row["source_catalogue_sha256"])
        counted_values.append(row["counted_allele"])
    return {
        "event_reference_ac": reference_ac, "event_reference_an": reference_an,
        "event_reference_known": reference_known,
        "event_call_provenance_counts": provenance,
        "event_call_provenance_known": provenance_known,
        "event_origin": origins, "event_phase_status": phases, "event_quality_status": qualities,
        "event_source_catalogue_sha256": catalogues, "event_counted_allele": counted_values,
        "event_copyable_in_case_fold": copyable, "event_copyable_known": copyable_known,
    }


def build_index_arrays(fields, rows, np):
    """Build typed locus arrays for a complete revised EVENT catalogue."""
    values = encode_rows(fields, rows)
    return {
        "event_reference_ac": np.asarray(values["event_reference_ac"], dtype=np.int64),
        "event_reference_an": np.asarray(values["event_reference_an"], dtype=np.int64),
        "event_reference_known": np.asarray(values["event_reference_known"], dtype=bool),
        "event_call_provenance_counts": np.asarray(values["event_call_provenance_counts"], dtype=np.int64),
        "event_call_provenance_known": np.asarray(values["event_call_provenance_known"], dtype=bool),
        "event_origin": np.asarray(values["event_origin"], dtype=str),
        "event_phase_status": np.asarray(values["event_phase_status"], dtype=str),
        "event_quality_status": np.asarray(values["event_quality_status"], dtype=str),
        "event_source_catalogue_sha256": np.asarray(values["event_source_catalogue_sha256"], dtype=str),
        "event_counted_allele": np.asarray(values["event_counted_allele"], dtype=str),
        "event_copyable_in_case_fold": np.asarray(values["event_copyable_in_case_fold"], dtype=np.int8),
        "event_copyable_known": np.asarray(values["event_copyable_known"], dtype=bool),
        "event_reference_ancestry_order": np.asarray(ANCESTRIES, dtype=str),
        "event_call_provenance_role_order": np.asarray(PROVENANCE_ROLES, dtype=str),
        "event_call_provenance_count_order": np.asarray(PROVENANCE_COUNTS, dtype=str),
        "event_annotation_schema": np.asarray(SCHEMA),
    }


def validate_index_arrays(index, np):
    """Validate shapes, axis labels, masks and sentinels in a stored index."""
    require(set(TYPED_ARRAY_FIELDS) <= set(index), "Typed EVENT annotation arrays incomplete")
    integer64 = ("event_reference_ac", "event_reference_an",
                 "event_call_provenance_counts")
    boolean = ("event_reference_known", "event_call_provenance_known",
               "event_copyable_known")
    text = ("event_origin", "event_phase_status", "event_quality_status",
            "event_source_catalogue_sha256", "event_counted_allele",
            "event_reference_ancestry_order", "event_call_provenance_role_order",
            "event_call_provenance_count_order", "event_annotation_schema")
    require(all(index[name].dtype == np.dtype(np.int64) for name in integer64),
            "Stored EVENT count dtypes differ")
    require(all(index[name].dtype == np.dtype(bool) for name in boolean),
            "Stored EVENT known-mask dtypes differ")
    require(index["event_copyable_in_case_fold"].dtype == np.dtype(np.int8),
            "Stored EVENT copyability dtype differs")
    require(all(index[name].dtype.kind in {"U", "S"} for name in text),
            "Stored EVENT text dtypes differ")
    require(index["event_annotation_schema"].shape == (),
            "Stored EVENT annotation schema must be scalar")
    require(str(index["event_annotation_schema"].item()) == SCHEMA,
            "Stored EVENT annotation schema differs")
    require(index["event_reference_ancestry_order"].astype(str).tolist() == list(ANCESTRIES)
            and index["event_call_provenance_role_order"].astype(str).tolist() == list(PROVENANCE_ROLES)
            and index["event_call_provenance_count_order"].astype(str).tolist() == list(PROVENANCE_COUNTS),
            "Stored EVENT annotation axis labels differ")
    n = len(index["event_origin"])
    ac, an = index["event_reference_ac"], index["event_reference_an"]
    known = index["event_reference_known"]
    provenance, provenance_known = (index["event_call_provenance_counts"],
                                    index["event_call_provenance_known"])
    require(ac.shape == an.shape == known.shape == (n, 3)
            and provenance.shape == (n, 2, 3) and provenance_known.shape == (n, 2),
            "Stored EVENT annotation shapes differ")
    require(all(index[name].shape == (n,) for name in (
        "event_phase_status", "event_quality_status", "event_source_catalogue_sha256",
        "event_counted_allele", "event_copyable_in_case_fold", "event_copyable_known")),
        "Stored EVENT annotation locus axes differ")
    require(bool(((ac[known] >= 0) & (an[known] >= 0) & (ac[known] <= an[known])).all())
            and bool((ac[~known] == -1).all()) and bool((an[~known] == -1).all()),
            "Stored EVENT reference count masks differ")
    expanded_known = np.broadcast_to(provenance_known[..., None], provenance.shape)
    require(bool((provenance[~expanded_known] == -1).all())
            and bool((provenance[expanded_known] >= 0).all()),
            "Stored EVENT provenance masks differ")
    if bool(provenance_known.any()):
        values = provenance[provenance_known]
        require(bool((values[:, 0] == values[:, 1] + values[:, 2]).all()),
                "Stored EVENT provenance copies do not add up")
    copyable, copyable_known = index["event_copyable_in_case_fold"], index["event_copyable_known"]
    require(bool(np.isin(copyable[copyable_known], (0, 1)).all())
            and bool((copyable[~copyable_known] == -1).all()),
            "Stored EVENT copyability masks differ")
    require(set(index["event_origin"].astype(str).tolist()) <=
            {"ORIGINAL_ALT_EVENT", "REVIEWED_REF_EXTENSION"},
            "Stored EVENT origins differ")
    require(all(bool(value) for value in index["event_phase_status"].astype(str))
            and all(bool(value) for value in index["event_quality_status"].astype(str)),
            "Stored EVENT phase or quality status is empty")
    require(all(SHA256.fullmatch(value) for value in
                index["event_source_catalogue_sha256"].astype(str)),
            "Stored EVENT source catalogue SHA256 differs")
    if {"event_ref", "event_alt", "event_is_alt"} <= set(index):
        orientation = np.asarray([
            counted_is_alt(str(ref), str(alt), str(counted))
            for ref, alt, counted in zip(index["event_ref"], index["event_alt"],
                                         index["event_counted_allele"])
        ], dtype=bool)
        require(index["event_is_alt"].dtype == np.dtype(bool)
                and np.array_equal(orientation, index["event_is_alt"]),
                "Stored EVENT counted-allele orientation differs")
    return n


def region_arrays(index, event_ids, np, pad_empty=True, validated=False):
    """Slice typed annotations on a regional EVENT axis, padding only explicitly."""
    schema = str(index["event_annotation_schema"].item())
    require(schema == SCHEMA, "Regional annotations require the revised schema")
    if not validated:
        validate_index_arrays(index, np)
    ids = np.asarray(event_ids, dtype=np.int64)
    require(ids.ndim == 1 and (not len(ids) or (ids.min() >= 0 and ids.max() < len(index["event_origin"]))),
            "Invalid regional EVENT annotation indices")
    locus_fields = set(TYPED_ARRAY_FIELDS) - set(NON_LOCUS_ARRAY_FIELDS)
    result = {name: np.asarray(index[name][ids]) for name in locus_fields}
    if not len(ids) and pad_empty:
        result = {
            "event_reference_ac": np.full((1, 3), -1, np.int64),
            "event_reference_an": np.full((1, 3), -1, np.int64),
            "event_reference_known": np.zeros((1, 3), bool),
            "event_call_provenance_counts": np.full((1, 2, 3), -1, np.int64),
            "event_call_provenance_known": np.zeros((1, 2), bool),
            "event_origin": np.asarray([""], dtype=str),
            "event_phase_status": np.asarray([""], dtype=str),
            "event_quality_status": np.asarray([""], dtype=str),
            "event_source_catalogue_sha256": np.asarray([""], dtype=str),
            "event_counted_allele": np.asarray([""], dtype=str),
            "event_copyable_in_case_fold": np.asarray([-1], dtype=np.int8),
            "event_copyable_known": np.asarray([False], dtype=bool),
        }
    result.update({
        "event_reference_ancestry_order": np.asarray(index["event_reference_ancestry_order"]),
        "event_call_provenance_role_order": np.asarray(index["event_call_provenance_role_order"]),
        "event_call_provenance_count_order": np.asarray(index["event_call_provenance_count_order"]),
        "event_annotation_schema": np.asarray(schema),
        "event_annotation_is_padding": np.asarray([True] if not len(ids) and pad_empty
                                                   else [False] * len(ids), dtype=bool),
    })
    return result
