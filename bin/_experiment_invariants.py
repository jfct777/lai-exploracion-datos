#!/usr/bin/env python3
"""Reusable fail-closed invariants for genomic model comparisons.

The helpers in this module are deliberately independent of file formats and
workflow engines.  Callers must first authenticate and decode their inputs,
then pass ordinary Python mappings and sequences here.  A failed invariant
raises ``ValueError`` before model fitting or score inspection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


LOCUS_FIELDS = ("CHROM", "POS", "REF", "ALT")
ROLE_FIELDS = (
    "person_id", "haplotype_id", "atomic_unit_id", "donor_lineage_id", "role",
)
ARTIFACT_FIELDS = (
    "artifact_id", "purpose", "sha256", "roles", "data_kinds", "depends_on",
)
SIGNATURE_FIELDS = (
    "feature_names", "event_rate", "sparsity", "missingness",
    "class_proportions", "component_counts",
)
SIGNATURE_TOLERANCE_FIELDS = (
    "event_rate_abs", "sparsity_abs", "missingness_abs",
    "class_proportion_abs", "class_sum_abs", "component_count_abs",
)
NULL_REQUIRED_FIELDS = (
    "locus_axis", "positions", "masks", "dosage", "burden",
    "unit_ids", "ancestry_mapping",
)
ALLELE_SEMANTIC_FIELDS = (
    "mode", "effect_alleles", "frequency_estimation_roles",
    "frequency_source_sha256", "pooled_alt_frequencies",
    "within_ancestry_alt_frequencies", "rare_threshold",
    "ancestral_alleles", "ancestral_source_sha256", "novelty_catalogs",
    "tie_policy",
)
NOVELTY_CATALOG_FIELDS = (
    "catalog_id", "sha256", "effect_allele_states",
)
ALLELE_SEMANTIC_MODES = (
    "ALT", "MINOR", "WITHIN_ANCESTRY_RARE", "DERIVED", "NOVEL",
)
PHASE_CONTRACT_FIELDS = (
    "state", "ploidy", "encoding", "phase_method",
    "phase_artifact_sha256", "haplotype_axis_sha256", "phase_qc_sha256",
    "heterozygote_policy", "haplotype_specific_claims",
)
PHASE_STATES = ("GENOTYPE", "PHASED", "AMBIGUOUS")
PARAMETER_ACTIVITY_TRIAL_FIELDS = (
    "parameter", "parameters", "output", "output_replay",
)
PARAMETER_ACTIVITY_OUTPUT_FIELDS = ("axis_sha256", "values")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DNA_ALLELE_RE = re.compile(r"^[ACGT]+$")


STANDARD_CLAIM_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "TECHNICAL_ONLY",
        (
            "inputs_authenticated",
            "schemas_valid",
            "locus_partition_valid",
            "allele_semantics_valid",
            "phase_contract_valid",
        ),
    ),
    (
        "EXPLORATORY",
        (
            "inputs_authenticated",
            "schemas_valid",
            "locus_partition_valid",
            "allele_semantics_valid",
            "phase_contract_valid",
            "roles_separated",
            "selection_isolated",
            "fixture_matches_production",
            "null_invariants_valid",
            "parameter_activity_valid",
        ),
    ),
    (
        "CONFIRMATORY",
        (
            "inputs_authenticated",
            "schemas_valid",
            "locus_partition_valid",
            "allele_semantics_valid",
            "phase_contract_valid",
            "roles_separated",
            "selection_isolated",
            "fixture_matches_production",
            "null_invariants_valid",
            "parameter_activity_valid",
            "analysis_preregistered",
            "score_independent",
            "power_adequate",
            "independent_truth",
            "effect_replicated",
        ),
    ),
)


def require(condition: bool, message: str) -> None:
    """Raise a uniform fail-closed error when an invariant is not met."""
    if not condition:
        raise ValueError(message)


def _canonical_value(value: Any, *, path: str = "root") -> Any:
    """Convert a JSON-like value to a deterministic, strictly finite form."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        require(math.isfinite(value), f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        require(all(isinstance(key, str) for key in value),
                f"{path} contains a non-string mapping key")
        return {
            key: _canonical_value(value[key], path=f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} contains a non-canonical value of type {type(value).__name__}")


def canonical_sha256(value: Any, *, domain: str = "DNABR_EXPERIMENT_INVARIANTS_V1") -> str:
    """Hash a JSON-like value with a domain separator and canonical encoding."""
    require(isinstance(domain, str) and domain and "\x00" not in domain,
            "hash domain must be a non-empty string without NUL")
    payload = json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + payload).hexdigest()


def _text(value: Any, *, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError(f"{label} is not ASCII") from None
    require(isinstance(value, str), f"{label} must be text")
    result = value.strip()
    require(result, f"{label} is empty")
    return result


def _normalize_chrom(value: Any, *, label: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    if isinstance(value, int):
        require(value > 0, f"{label} is invalid")
        return str(value)
    chrom = _text(value, label=label)
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    chrom = chrom.upper()
    if chrom == "M":
        chrom = "MT"
    require(re.fullmatch(r"(?:[1-9][0-9]*|X|Y|MT)", chrom) is not None,
            f"{label} is invalid")
    return chrom


def _normalize_position(value: Any, *, label: str) -> int:
    require(not isinstance(value, (bool, float)), f"{label} is not an integer")
    if isinstance(value, str):
        require(re.fullmatch(r"[0-9]+", value.strip()) is not None,
                f"{label} is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not an integer") from None
    require(0 < result < 2**31, f"{label} is outside the supported coordinate range")
    return result


def _normalize_allele(value: Any, *, label: str) -> str:
    allele = _text(value, label=label).upper()
    require(DNA_ALLELE_RE.fullmatch(allele) is not None,
            f"{label} is not an explicit DNA allele")
    return allele


def normalize_locus_axis(
    rows: Sequence[Any], *, label: str = "locus_axis",
) -> tuple[tuple[str, int, str, str], ...]:
    """Normalize an ordered CHROM/POS/REF/ALT axis and reject duplicates.

    Each row must be either a four-element sequence or a mapping containing
    exactly the uppercase or lowercase locus fields.  Chromosome prefixes and
    allele case are normalized, but no surrogate locus identifier is accepted.
    """
    require(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)),
            f"{label} must be a sequence")
    normalized: list[tuple[str, int, str, str]] = []
    for index, row in enumerate(rows):
        row_label = f"{label}[{index}]"
        if isinstance(row, Mapping):
            keys = set(row)
            if keys == set(LOCUS_FIELDS):
                raw = tuple(row[field] for field in LOCUS_FIELDS)
            elif keys == {field.lower() for field in LOCUS_FIELDS}:
                raw = tuple(row[field.lower()] for field in LOCUS_FIELDS)
            else:
                raise ValueError(
                    f"{row_label} must contain exactly CHROM/POS/REF/ALT"
                )
        else:
            require(
                isinstance(row, Sequence) and
                not isinstance(row, (str, bytes, bytearray)) and len(row) == 4,
                f"{row_label} is not CHROM/POS/REF/ALT",
            )
            raw = tuple(row)
        chrom = _normalize_chrom(raw[0], label=f"{row_label}.CHROM")
        pos = _normalize_position(raw[1], label=f"{row_label}.POS")
        ref = _normalize_allele(raw[2], label=f"{row_label}.REF")
        alt = _normalize_allele(raw[3], label=f"{row_label}.ALT")
        require(ref != alt, f"{row_label} has identical REF and ALT")
        normalized.append((chrom, pos, ref, alt))
    require(len(normalized) == len(set(normalized)), f"{label} contains duplicate loci")
    chromosome_order = {
        **{str(chrom): chrom for chrom in range(1, 100)},
        "X": 100, "Y": 101, "MT": 102,
    }
    coordinates = [(chromosome_order[chrom], pos) for chrom, pos, _ref, _alt in normalized]
    require(all(left <= right for left, right in zip(coordinates, coordinates[1:])),
            f"{label} is not ordered by chromosome and position")
    return tuple(normalized)


def validate_exact_locus_partition(
    full_loci: Sequence[Any],
    minus_selected_loci: Sequence[Any],
    selected_loci: Sequence[Any],
    *,
    require_nonempty_selected: bool = True,
) -> dict[str, Any]:
    """Require ``F_full = F_minus_selected`` disjoint-union ``selected``.

    Besides set equality, this checks that both child axes preserve the order
    induced by ``F_full``.  This catches duplicate, overlap, omission and
    reordering errors before a baseline comparison is run.
    """
    full = normalize_locus_axis(full_loci, label="F_full")
    minus = normalize_locus_axis(minus_selected_loci, label="F_minus_selected")
    selected = normalize_locus_axis(selected_loci, label="selected")
    require(full, "F_full is empty")
    if require_nonempty_selected:
        require(selected, "selected is empty")

    full_set = set(full)
    minus_set = set(minus)
    selected_set = set(selected)
    require(not (minus_set & selected_set),
            "F_minus_selected intersects selected")
    require(selected_set <= full_set, "selected contains loci absent from F_full")
    require(minus_set | selected_set == full_set,
            "F_full is not the exact union of F_minus_selected and selected")

    expected_minus = tuple(locus for locus in full if locus not in selected_set)
    expected_selected = tuple(locus for locus in full if locus in selected_set)
    require(minus == expected_minus,
            "F_minus_selected does not preserve the F_full order")
    require(selected == expected_selected,
            "selected does not preserve the F_full order")

    return {
        "status": "PASS_EXACT_LOCUS_PARTITION",
        "counts": {
            "F_full": len(full),
            "F_minus_selected": len(minus),
            "selected": len(selected),
            "overlap": 0,
        },
        "axis_sha256": {
            "F_full": canonical_sha256(full, domain="DNABR_LOCUS_AXIS_V1"),
            "F_minus_selected": canonical_sha256(
                minus, domain="DNABR_LOCUS_AXIS_V1"
            ),
            "selected": canonical_sha256(selected, domain="DNABR_LOCUS_AXIS_V1"),
        },
        "order_preserved": True,
    }


def _strict_id(value: Any, *, label: str) -> str:
    result = _text(value, label=label)
    require(result == value if isinstance(value, str) else True,
            f"{label} has surrounding whitespace")
    return result


def validate_role_separation(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_roles: Sequence[str] = ("TRAIN", "SELECT", "SCORE"),
    no_cross_fields: Sequence[str] = (
        "person_id", "atomic_unit_id", "donor_lineage_id",
    ),
    expected_haplotypes_per_person: int = 2,
) -> dict[str, Any]:
    """Validate complete haplotypes and dependency-safe biological roles."""
    require(isinstance(rows, Sequence) and rows, "role rows are empty")
    roles = tuple(required_roles)
    require(roles and len(roles) == len(set(roles)) and
            all(isinstance(role, str) and role for role in roles),
            "required_roles must be unique non-empty strings")
    require(isinstance(expected_haplotypes_per_person, int) and
            not isinstance(expected_haplotypes_per_person, bool) and
            expected_haplotypes_per_person > 0,
            "expected_haplotypes_per_person must be a positive integer")
    cross_fields = tuple(no_cross_fields)
    require(cross_fields and len(cross_fields) == len(set(cross_fields)) and
            set(cross_fields) <= set(ROLE_FIELDS),
            "no_cross_fields are invalid or duplicated")

    role_by_value: dict[str, dict[str, set[str]]] = {
        field: {} for field in cross_fields
    }
    person_haplotypes: dict[str, set[str]] = {}
    person_metadata: dict[str, dict[str, set[str]]] = {}
    haplotype_owner: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    normalized_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        require(isinstance(row, Mapping) and set(row) == set(ROLE_FIELDS),
                f"role row {index} fields differ")
        normalized = {
            field: _strict_id(row[field], label=f"role row {index}.{field}")
            for field in ROLE_FIELDS
        }
        role = normalized["role"]
        require(role in roles, f"role row {index} has unsupported role {role}")
        person = normalized["person_id"]
        haplotype = normalized["haplotype_id"]
        pair = (person, haplotype)
        require(pair not in seen_pairs, f"duplicate person/haplotype row: {pair}")
        seen_pairs.add(pair)
        prior_owner = haplotype_owner.setdefault(haplotype, person)
        require(prior_owner == person,
                f"haplotype_id belongs to multiple people: {haplotype}")
        person_haplotypes.setdefault(person, set()).add(haplotype)
        metadata = person_metadata.setdefault(
            person,
            {"role": set(), "atomic_unit_id": set(), "donor_lineage_id": set()},
        )
        for field in metadata:
            metadata[field].add(normalized[field])
        for field in cross_fields:
            role_by_value[field].setdefault(normalized[field], set()).add(role)
        normalized_rows.append(normalized)

    require(set(row["role"] for row in normalized_rows) == set(roles),
            "one or more required roles have no rows")
    for person, haplotypes in person_haplotypes.items():
        for field, values in person_metadata[person].items():
            require(len(values) == 1,
                    f"person {person} has inconsistent {field}")
        require(len(haplotypes) == expected_haplotypes_per_person,
                f"person {person} does not have complete haplotypes")
    for field, values in role_by_value.items():
        for value, value_roles in values.items():
            require(len(value_roles) == 1,
                    f"{field} crosses roles: {value}")

    role_counts: dict[str, dict[str, int]] = {}
    role_hashes: dict[str, str] = {}
    for role in roles:
        role_rows = [row for row in normalized_rows if row["role"] == role]
        role_counts[role] = {
            "people": len({row["person_id"] for row in role_rows}),
            "haplotypes": len(role_rows),
            "atomic_units": len({row["atomic_unit_id"] for row in role_rows}),
            "donor_lineages": len({row["donor_lineage_id"] for row in role_rows}),
        }
        role_hashes[role] = canonical_sha256(
            sorted(role_rows, key=lambda row: (row["person_id"], row["haplotype_id"])),
            domain="DNABR_ROLE_PARTITION_V1",
        )
    return {
        "status": "PASS_ROLE_SEPARATION",
        "required_roles": list(roles),
        "no_cross_fields": list(cross_fields),
        "expected_haplotypes_per_person": expected_haplotypes_per_person,
        "counts": role_counts,
        "role_sha256": role_hashes,
    }


def _require_sha256(value: Any, *, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def _optional_sha256(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, label=label)


def _frequency_vector(
    values: Any, *, expected_length: int, label: str,
) -> tuple[float, ...]:
    require(isinstance(values, Sequence) and
            not isinstance(values, (str, bytes, bytearray)),
            f"{label} must be a sequence")
    require(len(values) == expected_length,
            f"{label} length differs from the locus axis")
    return tuple(
        _unit_interval(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )


def validate_allele_semantics(
    locus_axis: Sequence[Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate which biological allele every encoded value represents.

    The five supported modes are deliberately distinct.  ``ALT`` is a VCF
    encoding choice; ``MINOR`` and ``WITHIN_ANCESTRY_RARE`` require authenticated
    frequency estimates; ``DERIVED`` requires an authenticated ancestral-state
    source; and ``NOVEL`` requires callable absence of the effect allele in every
    declared external catalog.  Evidence roles used to define frequency-based
    alleles may not include validation, scoring, evaluation or holdout data.
    """
    loci = normalize_locus_axis(locus_axis, label="allele_semantics.locus_axis")
    require(loci, "allele_semantics.locus_axis is empty")
    require(isinstance(contract, Mapping) and
            set(contract) == set(ALLELE_SEMANTIC_FIELDS),
            "allele semantic contract fields differ")

    mode = _strict_id(contract["mode"], label="allele_semantics.mode")
    require(mode in ALLELE_SEMANTIC_MODES,
            f"unsupported allele semantic mode: {mode}")

    raw_effect = contract["effect_alleles"]
    require(isinstance(raw_effect, Sequence) and
            not isinstance(raw_effect, (str, bytes, bytearray)) and
            len(raw_effect) == len(loci),
            "effect_alleles length differs from the locus axis")
    effect_alleles: tuple[str, ...] = tuple(
        _normalize_allele(value, label=f"effect_alleles[{index}]")
        for index, value in enumerate(raw_effect)
    )
    for index, (effect, locus) in enumerate(zip(effect_alleles, loci)):
        require(effect in locus[2:],
                f"effect_alleles[{index}] is neither REF nor ALT")

    raw_roles = contract["frequency_estimation_roles"]
    require(isinstance(raw_roles, Sequence) and
            not isinstance(raw_roles, (str, bytes, bytearray)),
            "frequency_estimation_roles must be a sequence")
    roles = tuple(
        _strict_id(value, label="frequency_estimation_roles")
        for value in raw_roles
    )
    require(len(roles) == len(set(roles)),
            "frequency_estimation_roles contains duplicates")
    for role in roles:
        require(role == role.upper() and
                re.fullmatch(r"[A-Z][A-Z0-9_:-]*", role) is not None,
                f"frequency estimation role is not explicit uppercase text: {role}")
        tokens = set(re.split(r"[_:-]+", role))
        require(not (tokens & {
            "SCORE", "TEST", "EVAL", "VALID", "VALIDATION", "HOLDOUT",
        }),
                f"frequency estimation uses an evaluation role: {role}")

    frequency_source = _optional_sha256(
        contract["frequency_source_sha256"],
        label="frequency_source_sha256",
    )
    pooled = None
    if contract["pooled_alt_frequencies"] is not None:
        pooled = _frequency_vector(
            contract["pooled_alt_frequencies"], expected_length=len(loci),
            label="pooled_alt_frequencies",
        )

    raw_within = contract["within_ancestry_alt_frequencies"]
    require(isinstance(raw_within, Mapping),
            "within_ancestry_alt_frequencies must be a mapping")
    require(all(isinstance(key, str) for key in raw_within),
            "within_ancestry_alt_frequencies has a non-text ancestry label")
    within: dict[str, tuple[float, ...]] = {}
    for ancestry, values in sorted(raw_within.items()):
        label = _strict_id(ancestry, label="ancestry label")
        require(label == label.upper(),
                f"ancestry label is not uppercase: {label}")
        within[label] = _frequency_vector(
            values, expected_length=len(loci),
            label=f"within_ancestry_alt_frequencies.{label}",
        )

    threshold = contract["rare_threshold"]
    if threshold is not None:
        require(isinstance(threshold, (int, float)) and
                not isinstance(threshold, bool) and math.isfinite(float(threshold)) and
                0.0 < float(threshold) < 0.5,
                "rare_threshold must be finite, positive and below 0.5")
        threshold = float(threshold)

    ancestral = None
    if contract["ancestral_alleles"] is not None:
        raw_ancestral = contract["ancestral_alleles"]
        require(isinstance(raw_ancestral, Sequence) and
                not isinstance(raw_ancestral, (str, bytes, bytearray)) and
                len(raw_ancestral) == len(loci),
                "ancestral_alleles length differs from the locus axis")
        ancestral = tuple(
            _normalize_allele(value, label=f"ancestral_alleles[{index}]")
            for index, value in enumerate(raw_ancestral)
        )
        for index, (allele, locus) in enumerate(zip(ancestral, loci)):
            require(allele in locus[2:],
                    f"ancestral_alleles[{index}] is neither REF nor ALT")
    ancestral_source = _optional_sha256(
        contract["ancestral_source_sha256"],
        label="ancestral_source_sha256",
    )

    raw_catalogs = contract["novelty_catalogs"]
    require(isinstance(raw_catalogs, Sequence) and
            not isinstance(raw_catalogs, (str, bytes, bytearray)),
            "novelty_catalogs must be a sequence")
    catalogs: list[dict[str, Any]] = []
    catalog_ids: set[str] = set()
    allowed_catalog_states = {"PRESENT", "ABSENT_CALLABLE", "UNKNOWN"}
    for index, catalog in enumerate(raw_catalogs):
        require(isinstance(catalog, Mapping) and
                set(catalog) == set(NOVELTY_CATALOG_FIELDS),
                f"novelty catalog {index} fields differ")
        catalog_id = _strict_id(
            catalog["catalog_id"], label=f"novelty catalog {index}.catalog_id"
        )
        require(catalog_id not in catalog_ids,
                f"duplicate novelty catalog: {catalog_id}")
        catalog_ids.add(catalog_id)
        states = catalog["effect_allele_states"]
        require(isinstance(states, Sequence) and
                not isinstance(states, (str, bytes, bytearray)) and
                len(states) == len(loci),
                f"novelty catalog {catalog_id} state length differs")
        normalized_states = tuple(
            _strict_id(state, label=f"novelty catalog {catalog_id}.state[{state_index}]")
            for state_index, state in enumerate(states)
        )
        require(set(normalized_states) <= allowed_catalog_states,
                f"novelty catalog {catalog_id} has an unsupported state")
        catalogs.append({
            "catalog_id": catalog_id,
            "sha256": _require_sha256(
                catalog["sha256"], label=f"novelty catalog {catalog_id}.sha256"
            ),
            "effect_allele_states": normalized_states,
        })

    tie_policy = _strict_id(contract["tie_policy"], label="tie_policy")
    require(tie_policy in {"REJECT", "NOT_APPLICABLE"},
            "tie_policy must be REJECT or NOT_APPLICABLE")

    frequency_modes = {"MINOR", "WITHIN_ANCESTRY_RARE"}
    if mode in frequency_modes:
        require(roles, f"{mode} requires frequency_estimation_roles")
        require(frequency_source is not None,
                f"{mode} requires frequency_source_sha256")
    else:
        require(not roles and frequency_source is None,
                f"{mode} must not carry frequency-estimation evidence")

    if mode == "ALT":
        require(all(effect == locus[3] for effect, locus in zip(effect_alleles, loci)),
                "ALT mode has a non-ALT effect allele")
        require(pooled is None and not within and threshold is None,
                "ALT mode must not carry frequency evidence")
        require(ancestral is None and ancestral_source is None and not catalogs,
                "ALT mode must not carry derived or novelty evidence")
        require(tie_policy == "NOT_APPLICABLE",
                "ALT mode requires NOT_APPLICABLE tie_policy")
    elif mode == "MINOR":
        require(pooled is not None and not within and threshold is None,
                "MINOR requires only pooled ALT frequencies")
        require(ancestral is None and ancestral_source is None and not catalogs,
                "MINOR must not carry derived or novelty evidence")
        require(tie_policy == "REJECT", "MINOR requires REJECT tie_policy")
        for index, (alt_frequency, effect, locus) in enumerate(
            zip(pooled, effect_alleles, loci)
        ):
            require(alt_frequency != 0.5,
                    f"MINOR locus {index} has no unique minor allele")
            expected = locus[3] if alt_frequency < 0.5 else locus[2]
            require(effect == expected,
                    f"MINOR effect allele disagrees with frequency at locus {index}")
    elif mode == "WITHIN_ANCESTRY_RARE":
        require(pooled is None and len(within) >= 2 and threshold is not None,
                "WITHIN_ANCESTRY_RARE requires at least two ancestry frequency axes")
        require(ancestral is None and ancestral_source is None and not catalogs,
                "WITHIN_ANCESTRY_RARE must not carry derived or novelty evidence")
        require(tie_policy == "NOT_APPLICABLE",
                "WITHIN_ANCESTRY_RARE requires NOT_APPLICABLE tie_policy")
        for ancestry, alt_frequencies in within.items():
            for index, (alt_frequency, effect, locus) in enumerate(
                zip(alt_frequencies, effect_alleles, loci)
            ):
                effect_frequency = (
                    alt_frequency if effect == locus[3] else 1.0 - alt_frequency
                )
                require(effect_frequency < threshold,
                        f"effect allele is not rare in {ancestry} at locus {index}")
    elif mode == "DERIVED":
        require(pooled is None and not within and threshold is None,
                "DERIVED must not carry frequency evidence")
        require(ancestral is not None and ancestral_source is not None,
                "DERIVED requires ancestral alleles and an authenticated source")
        require(not catalogs, "DERIVED must not carry novelty evidence")
        require(tie_policy == "NOT_APPLICABLE",
                "DERIVED requires NOT_APPLICABLE tie_policy")
        for index, (effect, ancestor) in enumerate(zip(effect_alleles, ancestral)):
            require(effect != ancestor,
                    f"DERIVED effect allele equals the ancestral allele at locus {index}")
    else:
        require(pooled is None and not within and threshold is None,
                "NOVEL must not carry frequency evidence")
        require(ancestral is None and ancestral_source is None,
                "NOVEL must not carry ancestral-state evidence")
        require(catalogs, "NOVEL requires at least one authenticated catalog")
        require(tie_policy == "NOT_APPLICABLE",
                "NOVEL requires NOT_APPLICABLE tie_policy")
        for catalog in catalogs:
            require(set(catalog["effect_allele_states"]) == {"ABSENT_CALLABLE"},
                    f"NOVEL is not callable-absent in catalog {catalog['catalog_id']}")

    normalized = {
        "mode": mode,
        "locus_axis": loci,
        "effect_alleles": effect_alleles,
        "frequency_estimation_roles": roles,
        "frequency_source_sha256": frequency_source,
        "pooled_alt_frequencies": pooled,
        "within_ancestry_alt_frequencies": within,
        "rare_threshold": threshold,
        "ancestral_alleles": ancestral,
        "ancestral_source_sha256": ancestral_source,
        "novelty_catalogs": catalogs,
        "tie_policy": tie_policy,
    }
    return {
        "status": "PASS_ALLELE_SEMANTICS",
        "mode": mode,
        "locus_count": len(loci),
        "ref_effect_count": sum(
            effect == locus[2] for effect, locus in zip(effect_alleles, loci)
        ),
        "alt_effect_count": sum(
            effect == locus[3] for effect, locus in zip(effect_alleles, loci)
        ),
        "frequency_estimation_roles": list(roles),
        "ancestries": sorted(within),
        "novelty_catalogs": sorted(catalog_ids),
        "contract_sha256": canonical_sha256(
            normalized, domain="DNABR_ALLELE_SEMANTICS_V1"
        ),
    }


def validate_phase_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Distinguish diploid genotypes, certified phase and ambiguous phase.

    Unphased or ambiguous heterozygotes cannot support haplotype-specific
    assignments.  A phased representation must authenticate both its phase
    artifact and ordered haplotype axis.  Haplotype-specific scientific claims
    additionally require a phase-QC artifact.
    """
    require(isinstance(contract, Mapping) and
            set(contract) == set(PHASE_CONTRACT_FIELDS),
            "phase contract fields differ")
    state = _strict_id(contract["state"], label="phase.state")
    require(state in PHASE_STATES, f"unsupported phase state: {state}")
    ploidy = contract["ploidy"]
    require(isinstance(ploidy, int) and not isinstance(ploidy, bool) and ploidy == 2,
            "phase contract currently requires diploid data")
    encoding = _strict_id(contract["encoding"], label="phase.encoding")
    method = contract["phase_method"]
    if method is not None:
        method = _strict_id(method, label="phase.phase_method")
    phase_artifact = _optional_sha256(
        contract["phase_artifact_sha256"], label="phase.phase_artifact_sha256"
    )
    haplotype_axis = _optional_sha256(
        contract["haplotype_axis_sha256"], label="phase.haplotype_axis_sha256"
    )
    phase_qc = _optional_sha256(
        contract["phase_qc_sha256"], label="phase.phase_qc_sha256"
    )
    heterozygote_policy = _strict_id(
        contract["heterozygote_policy"], label="phase.heterozygote_policy"
    )
    claims = contract["haplotype_specific_claims"]
    require(isinstance(claims, bool),
            "haplotype_specific_claims must be boolean")

    if state == "GENOTYPE":
        require(encoding == "DIPLOID_DOSAGE",
                "GENOTYPE requires DIPLOID_DOSAGE encoding")
        require(method is None and phase_artifact is None and
                haplotype_axis is None and phase_qc is None,
                "GENOTYPE must not carry phase artifacts")
        require(heterozygote_policy == "UNASSIGNED",
                "GENOTYPE heterozygotes must remain UNASSIGNED")
        require(not claims, "GENOTYPE cannot support haplotype-specific claims")
    elif state == "PHASED":
        require(encoding == "ORDERED_HAPLOTYPES",
                "PHASED requires ORDERED_HAPLOTYPES encoding")
        require(method is not None and phase_artifact is not None and
                haplotype_axis is not None,
                "PHASED requires method, artifact and ordered haplotype axis")
        require(heterozygote_policy == "ASSIGNED_BY_PHASE",
                "PHASED requires ASSIGNED_BY_PHASE heterozygote policy")
        if claims:
            require(phase_qc is not None,
                    "haplotype-specific claims require phase_qc_sha256")
    else:
        require(encoding == "PHASE_UNCERTAIN",
                "AMBIGUOUS requires PHASE_UNCERTAIN encoding")
        require((method is None) == (phase_artifact is None),
                "AMBIGUOUS phase method and artifact must be declared together")
        require(haplotype_axis is None and phase_qc is None,
                "AMBIGUOUS cannot certify a haplotype axis or phase QC")
        require(heterozygote_policy in {"UNASSIGNED", "MARGINALIZE"},
                "AMBIGUOUS heterozygotes must be UNASSIGNED or MARGINALIZE")
        require(not claims, "AMBIGUOUS cannot support haplotype-specific claims")

    normalized = {
        "state": state,
        "ploidy": ploidy,
        "encoding": encoding,
        "phase_method": method,
        "phase_artifact_sha256": phase_artifact,
        "haplotype_axis_sha256": haplotype_axis,
        "phase_qc_sha256": phase_qc,
        "heterozygote_policy": heterozygote_policy,
        "haplotype_specific_claims": claims,
    }
    return {
        "status": "PASS_PHASE_CONTRACT",
        "state": state,
        "haplotype_specific_claims": claims,
        "phase_qc_authenticated": phase_qc is not None,
        "contract_sha256": canonical_sha256(
            normalized, domain="DNABR_PHASE_CONTRACT_V1"
        ),
    }


def _reject_parameter_metadata(value: Any, *, path: str) -> None:
    """Keep activity fingerprints restricted to decision-relevant outputs."""
    forbidden = {
        "parameter", "parameters", "params", "config", "configuration",
        "command", "command_line", "run_id", "timestamp", "provenance",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            require(normalized_key not in forbidden,
                    f"{path} contains parameter or run metadata: {key}")
            _reject_parameter_metadata(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _reject_parameter_metadata(item, path=f"{path}[{index}]")


def _validate_activity_output(value: Any, *, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping) and
            set(value) == set(PARAMETER_ACTIVITY_OUTPUT_FIELDS),
            f"{label} fields differ")
    axis = _require_sha256(value["axis_sha256"], label=f"{label}.axis_sha256")
    values = value["values"]
    require((isinstance(values, Mapping) and values) or
            (isinstance(values, Sequence) and
             not isinstance(values, (str, bytes, bytearray)) and values),
            f"{label}.values must be a non-empty mapping or sequence")
    normalized_values = _canonical_value(values, path=f"{label}.values")
    _reject_parameter_metadata(normalized_values, path=f"{label}.values")
    return {
        "axis_sha256": axis,
        "values": normalized_values,
        "values_sha256": canonical_sha256(
            normalized_values, domain="DNABR_PARAMETER_ACTIVITY_OUTPUT_V1"
        ),
    }


def validate_parameter_activity(
    baseline_parameters: Mapping[str, Any],
    baseline_output: Mapping[str, Any],
    baseline_output_replay: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
    *,
    required_parameters: Sequence[str],
) -> dict[str, Any]:
    """Prove that declared parameters are wired to decision-relevant output.

    Each trial must change exactly one required parameter, hold the complete
    remaining configuration fixed and reproduce its output byte-for-byte at the
    canonical-value level.  Output snapshots contain only a fixed biological
    axis hash and decision-relevant values; embedding configuration or run
    metadata in those values is rejected because it can make an inert parameter
    appear active.
    """
    require(isinstance(baseline_parameters, Mapping) and baseline_parameters,
            "baseline_parameters is empty")
    require(all(isinstance(key, str) and key for key in baseline_parameters),
            "baseline_parameters has an invalid key")
    baseline = _canonical_value(
        baseline_parameters, path="baseline_parameters"
    )
    required = tuple(required_parameters)
    require(required and len(required) == len(set(required)) and
            all(isinstance(name, str) and name for name in required),
            "required_parameters must be unique non-empty strings")
    require(set(required) <= set(baseline),
            "required_parameters references an absent baseline parameter")
    require(isinstance(trials, Sequence) and
            not isinstance(trials, (str, bytes, bytearray)) and trials,
            "parameter activity trials are empty")

    baseline_snapshot = _validate_activity_output(
        baseline_output, label="baseline_output"
    )
    baseline_replay = _validate_activity_output(
        baseline_output_replay, label="baseline_output_replay"
    )
    require(baseline_snapshot == baseline_replay,
            "baseline output is not reproducible")

    seen: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    baseline_config_sha = canonical_sha256(
        baseline, domain="DNABR_PARAMETER_CONFIGURATION_V1"
    )
    for index, trial in enumerate(trials):
        require(isinstance(trial, Mapping) and
                set(trial) == set(PARAMETER_ACTIVITY_TRIAL_FIELDS),
                f"parameter activity trial {index} fields differ")
        parameter = _strict_id(
            trial["parameter"], label=f"parameter activity trial {index}.parameter"
        )
        require(parameter in required,
                f"undeclared parameter activity trial: {parameter}")
        require(parameter not in seen,
                f"duplicate parameter activity trial: {parameter}")
        seen.add(parameter)
        candidate_raw = trial["parameters"]
        require(isinstance(candidate_raw, Mapping) and
                set(candidate_raw) == set(baseline),
                f"parameter activity trial {parameter} configuration fields differ")
        candidate = _canonical_value(
            candidate_raw, path=f"parameter_activity.{parameter}.parameters"
        )
        changed = [
            name for name in baseline if baseline[name] != candidate[name]
        ]
        require(changed == [parameter],
                f"parameter activity trial {parameter} is not one-factor-at-a-time")

        output = _validate_activity_output(
            trial["output"], label=f"parameter_activity.{parameter}.output"
        )
        replay = _validate_activity_output(
            trial["output_replay"],
            label=f"parameter_activity.{parameter}.output_replay",
        )
        require(output == replay,
                f"parameter activity output is not reproducible: {parameter}")
        require(output["axis_sha256"] == baseline_snapshot["axis_sha256"],
                f"parameter activity changes the comparison axis: {parameter}")
        require(output["values_sha256"] != baseline_snapshot["values_sha256"],
                f"parameter is inactive on decision-relevant output: {parameter}")
        candidate_config_sha = canonical_sha256(
            candidate, domain="DNABR_PARAMETER_CONFIGURATION_V1"
        )
        require(candidate_config_sha != baseline_config_sha,
                f"parameter activity configuration is unchanged: {parameter}")
        results[parameter] = {
            "baseline_value_sha256": canonical_sha256(
                baseline[parameter], domain="DNABR_PARAMETER_VALUE_V1"
            ),
            "perturbed_value_sha256": canonical_sha256(
                candidate[parameter], domain="DNABR_PARAMETER_VALUE_V1"
            ),
            "perturbed_config_sha256": candidate_config_sha,
            "output_sha256": output["values_sha256"],
            "replay_exact": True,
            "active": True,
        }

    require(seen == set(required),
            "parameter activity trials do not cover required_parameters exactly")
    core = {
        "status": "PASS_PARAMETER_ACTIVITY",
        "baseline_config_sha256": baseline_config_sha,
        "baseline_output_sha256": baseline_snapshot["values_sha256"],
        "axis_sha256": baseline_snapshot["axis_sha256"],
        "parameters": {name: results[name] for name in sorted(results)},
    }
    return {
        **core,
        "contract_sha256": canonical_sha256(
            core, domain="DNABR_PARAMETER_ACTIVITY_CONTRACT_V1"
        ),
    }


def validate_selection_isolation(
    artifacts: Sequence[Mapping[str, Any]],
    selector_replay_hashes: Mapping[str, Mapping[str, str]],
    *,
    selector_purposes: Sequence[str] = ("selection", "checkpoint"),
) -> dict[str, Any]:
    """Reject SCORE/truth dependencies in selection and checkpoint artifacts.

    ``selector_replay_hashes`` must contain the hash produced with SCORE absent
    and with SCORE present for every selector/checkpoint artifact.  Equality of
    those hashes is an executable check that SCORE availability did not change
    the selector output.
    """
    require(isinstance(artifacts, Sequence) and artifacts, "artifact inventory is empty")
    purposes = tuple(selector_purposes)
    require(purposes and len(purposes) == len(set(purposes)),
            "selector_purposes are empty or duplicated")
    inventory: dict[str, dict[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        require(isinstance(artifact, Mapping) and set(artifact) == set(ARTIFACT_FIELDS),
                f"artifact {index} fields differ")
        artifact_id = _strict_id(artifact["artifact_id"], label=f"artifact {index}.artifact_id")
        require(artifact_id not in inventory, f"duplicate artifact_id: {artifact_id}")
        purpose = _strict_id(artifact["purpose"], label=f"artifact {index}.purpose")
        digest = _require_sha256(artifact["sha256"], label=f"artifact {artifact_id}.sha256")
        role_values = artifact["roles"]
        kind_values = artifact["data_kinds"]
        dependency_values = artifact["depends_on"]
        for name, values in (
            ("roles", role_values), ("data_kinds", kind_values),
            ("depends_on", dependency_values),
        ):
            require(isinstance(values, Sequence) and
                    not isinstance(values, (str, bytes, bytearray)),
                    f"artifact {artifact_id}.{name} must be a sequence")
            require(len(values) == len(set(values)),
                    f"artifact {artifact_id}.{name} contains duplicates")
        roles = tuple(_strict_id(value, label=f"artifact {artifact_id}.roles")
                      for value in role_values)
        require(set(roles) <= {"TRAIN", "SELECT", "SCORE"},
                f"artifact {artifact_id} has an unsupported role")
        kinds = tuple(_strict_id(value, label=f"artifact {artifact_id}.data_kinds").lower()
                      for value in kind_values)
        dependencies = tuple(
            _strict_id(value, label=f"artifact {artifact_id}.depends_on")
            for value in dependency_values
        )
        inventory[artifact_id] = {
            "artifact_id": artifact_id,
            "purpose": purpose,
            "sha256": digest,
            "roles": roles,
            "data_kinds": kinds,
            "depends_on": dependencies,
        }

    for artifact_id, artifact in inventory.items():
        missing = set(artifact["depends_on"]) - set(inventory)
        require(not missing,
                f"artifact {artifact_id} has missing dependencies: {sorted(missing)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def dependencies_of(artifact_id: str) -> set[str]:
        require(artifact_id not in visiting, "artifact dependency graph contains a cycle")
        if artifact_id in visited:
            return set(inventory[artifact_id]["_transitive_dependencies"])
        visiting.add(artifact_id)
        result: set[str] = set()
        for dependency in inventory[artifact_id]["depends_on"]:
            result.add(dependency)
            result.update(dependencies_of(dependency))
        visiting.remove(artifact_id)
        visited.add(artifact_id)
        inventory[artifact_id]["_transitive_dependencies"] = tuple(sorted(result))
        return result

    selectors = sorted(
        artifact_id for artifact_id, artifact in inventory.items()
        if artifact["purpose"] in purposes
    )
    require(selectors, "artifact inventory has no selector or checkpoint artifacts")
    require(isinstance(selector_replay_hashes, Mapping) and
            set(selector_replay_hashes) == set(selectors),
            "selector replay inventory differs")

    forbidden_kinds = {
        "score_truth", "test_truth", "eval_truth", "score_labels", "score_metrics",
    }
    replay_hashes: dict[str, str] = {}
    for artifact_id in selectors:
        dependency_ids = dependencies_of(artifact_id)
        chain = {artifact_id, *dependency_ids}
        chain_roles = {
            role for item_id in chain for role in inventory[item_id]["roles"]
        }
        chain_kinds = {
            kind for item_id in chain for kind in inventory[item_id]["data_kinds"]
        }
        require("SCORE" not in chain_roles,
                f"selector {artifact_id} depends on SCORE")
        require(not any(kind in forbidden_kinds for kind in chain_kinds),
                f"selector {artifact_id} depends on SCORE truth or metrics")

        replay = selector_replay_hashes[artifact_id]
        require(isinstance(replay, Mapping) and
                set(replay) == {"without_score", "with_score"},
                f"selector replay fields differ: {artifact_id}")
        without_score = _require_sha256(
            replay["without_score"], label=f"selector {artifact_id}.without_score"
        )
        with_score = _require_sha256(
            replay["with_score"], label=f"selector {artifact_id}.with_score"
        )
        require(without_score == with_score,
                f"SCORE availability changes selector hash: {artifact_id}")
        require(without_score == inventory[artifact_id]["sha256"],
                f"selector replay does not reproduce artifact hash: {artifact_id}")
        replay_hashes[artifact_id] = without_score

    return {
        "status": "PASS_SELECTION_ISOLATION",
        "artifact_count": len(inventory),
        "selector_artifacts": selectors,
        "selector_sha256": replay_hashes,
        "score_or_truth_dependency_count": 0,
        "score_invariant_replay": True,
    }


def _unit_interval(value: Any, *, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result) and 0.0 <= result <= 1.0,
            f"{label} must be finite and between zero and one")
    return result


def _validate_signature(signature: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    require(isinstance(signature, Mapping) and set(signature) == set(SIGNATURE_FIELDS),
            f"{label} signature fields differ")
    features = signature["feature_names"]
    require(isinstance(features, Sequence) and
            not isinstance(features, (str, bytes, bytearray)) and features,
            f"{label}.feature_names must be a non-empty sequence")
    feature_names = tuple(_strict_id(value, label=f"{label}.feature_names")
                          for value in features)
    require(len(feature_names) == len(set(feature_names)),
            f"{label}.feature_names contains duplicates")

    proportions = signature["class_proportions"]
    components = signature["component_counts"]
    require(isinstance(proportions, Mapping) and proportions,
            f"{label}.class_proportions must be a non-empty mapping")
    require(isinstance(components, Mapping) and components,
            f"{label}.component_counts must be a non-empty mapping")
    require(all(isinstance(key, str) and key for key in proportions),
            f"{label}.class_proportions has an invalid key")
    require(all(isinstance(key, str) and key for key in components),
            f"{label}.component_counts has an invalid key")
    normalized_proportions = {
        key: _unit_interval(value, label=f"{label}.class_proportions.{key}")
        for key, value in sorted(proportions.items())
    }
    normalized_components: dict[str, int] = {}
    for key, value in sorted(components.items()):
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{label}.component_counts.{key} must be a non-negative integer")
        normalized_components[key] = value
    return {
        "feature_names": feature_names,
        "event_rate": _unit_interval(signature["event_rate"], label=f"{label}.event_rate"),
        "sparsity": _unit_interval(signature["sparsity"], label=f"{label}.sparsity"),
        "missingness": _unit_interval(signature["missingness"], label=f"{label}.missingness"),
        "class_proportions": normalized_proportions,
        "component_counts": normalized_components,
    }


def validate_fixture_production_signature(
    fixture: Mapping[str, Any],
    production: Mapping[str, Any],
    tolerances: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a technical fixture to exercise the production data regime."""
    fixture_value = _validate_signature(fixture, label="fixture")
    production_value = _validate_signature(production, label="production")
    require(isinstance(tolerances, Mapping) and
            set(tolerances) == set(SIGNATURE_TOLERANCE_FIELDS),
            "signature tolerance fields differ")
    tolerance_values: dict[str, float] = {}
    for name in SIGNATURE_TOLERANCE_FIELDS:
        value = tolerances[name]
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                math.isfinite(float(value)) and float(value) >= 0,
                f"tolerance {name} must be finite and non-negative")
        tolerance_values[name] = float(value)

    require(fixture_value["feature_names"] == production_value["feature_names"],
            "fixture and production feature axes differ")
    require(set(fixture_value["class_proportions"]) ==
            set(production_value["class_proportions"]),
            "fixture and production class axes differ")
    require(set(fixture_value["component_counts"]) ==
            set(production_value["component_counts"]),
            "fixture and production component axes differ")
    for label, value in (("fixture", fixture_value), ("production", production_value)):
        total = sum(value["class_proportions"].values())
        require(abs(total - 1.0) <= tolerance_values["class_sum_abs"],
                f"{label} class proportions do not sum to one")

    scalar_deltas = {
        name: abs(fixture_value[name] - production_value[name])
        for name in ("event_rate", "sparsity", "missingness")
    }
    for name, delta in scalar_deltas.items():
        require(delta <= tolerance_values[f"{name}_abs"],
                f"fixture/production {name} delta exceeds tolerance")
    class_deltas = {
        name: abs(fixture_value["class_proportions"][name] -
                  production_value["class_proportions"][name])
        for name in fixture_value["class_proportions"]
    }
    require(all(delta <= tolerance_values["class_proportion_abs"]
                for delta in class_deltas.values()),
            "fixture/production class proportion delta exceeds tolerance")
    component_deltas = {
        name: abs(fixture_value["component_counts"][name] -
                  production_value["component_counts"][name])
        for name in fixture_value["component_counts"]
    }
    require(all(delta <= tolerance_values["component_count_abs"]
                for delta in component_deltas.values()),
            "fixture/production component count delta exceeds tolerance")

    return {
        "status": "PASS_FIXTURE_PRODUCTION_SIGNATURE",
        "fixture_sha256": canonical_sha256(
            fixture_value, domain="DNABR_EXPERIMENT_SIGNATURE_V1"
        ),
        "production_sha256": canonical_sha256(
            production_value, domain="DNABR_EXPERIMENT_SIGNATURE_V1"
        ),
        "tolerances": tolerance_values,
        "absolute_deltas": {
            **scalar_deltas,
            "class_proportions": class_deltas,
            "component_counts": component_deltas,
        },
    }


def _mapping_values(value: Any, *, label: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(value, Mapping):
        require(value, f"{label} is empty")
        require(all(isinstance(key, str) and key for key in value),
                f"{label} has an invalid key")
        keys = tuple(sorted(value))
        values = tuple(
            json.dumps(_canonical_value(value[key], path=f"{label}.{key}"),
                       sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for key in keys
        )
        return keys, values
    require(isinstance(value, Sequence) and
            not isinstance(value, (str, bytes, bytearray)) and value,
            f"{label} must be a non-empty mapping or sequence")
    keys = tuple(str(index) for index in range(len(value)))
    values = tuple(
        json.dumps(_canonical_value(item, path=f"{label}[{index}]"),
                   sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for index, item in enumerate(value)
    )
    return keys, values


def validate_null_invariants(
    observed: Mapping[str, Any],
    null: Mapping[str, Any],
    *,
    require_ancestry_change: bool = True,
) -> dict[str, Any]:
    """Require a null to alter only the ancestry assignment.

    The locus axis, positions, masks, dosage, total burden and biological units
    are exact invariants.  Any additional field supplied by the caller is also
    held fixed.  The ancestry mapping must retain the same keys and multiset of
    values, so it is a permutation rather than a change in class balance.
    """
    require(isinstance(observed, Mapping) and isinstance(null, Mapping),
            "observed and null payloads must be mappings")
    require(set(observed) == set(null), "observed and null fields differ")
    require(set(NULL_REQUIRED_FIELDS) <= set(observed),
            "observed/null payload lacks required invariant fields")

    observed_loci = normalize_locus_axis(observed["locus_axis"], label="observed.locus_axis")
    null_loci = normalize_locus_axis(null["locus_axis"], label="null.locus_axis")
    require(observed_loci == null_loci, "null changes the ordered locus axis")

    positions = observed["positions"]
    null_positions = null["positions"]
    require(isinstance(positions, Sequence) and
            not isinstance(positions, (str, bytes, bytearray)) and
            len(positions) == len(observed_loci),
            "observed positions do not match the locus axis")
    require(isinstance(null_positions, Sequence) and
            not isinstance(null_positions, (str, bytes, bytearray)) and
            len(null_positions) == len(null_loci),
            "null positions do not match the locus axis")
    for label, values in (("observed", positions), ("null", null_positions)):
        numeric: list[float] = []
        for index, value in enumerate(values):
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                    math.isfinite(float(value)),
                    f"{label} position {index} is not finite numeric data")
            numeric.append(float(value))
        require(all(left <= right for left, right in zip(numeric, numeric[1:])),
                f"{label} positions are not ordered")

    invariant_fields = sorted(set(observed) - {"ancestry_mapping"})
    invariant_hashes: dict[str, str] = {}
    for field in invariant_fields:
        left = observed_loci if field == "locus_axis" else observed[field]
        right = null_loci if field == "locus_axis" else null[field]
        left_hash = canonical_sha256(left, domain=f"DNABR_NULL_{field}_V1")
        right_hash = canonical_sha256(right, domain=f"DNABR_NULL_{field}_V1")
        require(left_hash == right_hash, f"null changes invariant field: {field}")
        invariant_hashes[field] = left_hash

    observed_keys, observed_values = _mapping_values(
        observed["ancestry_mapping"], label="observed.ancestry_mapping"
    )
    null_keys, null_values = _mapping_values(
        null["ancestry_mapping"], label="null.ancestry_mapping"
    )
    require(observed_keys == null_keys, "null changes ancestry mapping keys")
    require(Counter(observed_values) == Counter(null_values),
            "null changes ancestry class counts")
    mapping_changed = observed_values != null_values
    if require_ancestry_change:
        require(mapping_changed, "null leaves ancestry mapping unchanged")

    return {
        "status": "PASS_NULL_INVARIANTS",
        "invariant_sha256": invariant_hashes,
        "allowed_changed_fields": ["ancestry_mapping"],
        "ancestry_mapping_changed": mapping_changed,
        "ancestry_class_counts_preserved": True,
    }


def build_claim_level_contract(
    gates: Mapping[str, bool],
    *,
    requirements: Sequence[tuple[str, Sequence[str]]] = STANDARD_CLAIM_REQUIREMENTS,
    evidence_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic claim level from monotone, explicit gate sets."""
    require(isinstance(gates, Mapping) and gates, "claim gates are empty")
    require(all(isinstance(name, str) and name for name in gates),
            "claim gate names must be non-empty strings")
    require(all(isinstance(value, bool) for value in gates.values()),
            "claim gate values must be booleans")
    require(isinstance(requirements, Sequence) and requirements,
            "claim requirements are empty")

    normalized_requirements: list[tuple[str, tuple[str, ...]]] = []
    previous: set[str] = set()
    seen_levels: set[str] = set()
    used_gates: set[str] = set()
    for index, item in enumerate(requirements):
        require(isinstance(item, Sequence) and len(item) == 2,
                f"claim requirement {index} is malformed")
        level, names = item
        require(isinstance(level, str) and level and level not in seen_levels,
                f"claim level {index} is empty or duplicated")
        require(isinstance(names, Sequence) and
                not isinstance(names, (str, bytes, bytearray)),
                f"claim level {level} gates must be a sequence")
        gate_names = tuple(names)
        require(gate_names and len(gate_names) == len(set(gate_names)) and
                all(isinstance(name, str) and name for name in gate_names),
                f"claim level {level} gates are empty, invalid or duplicated")
        gate_set = set(gate_names)
        require(previous <= gate_set,
                f"claim level {level} is not cumulative")
        require(gate_set <= set(gates),
                f"claim level {level} references missing gates")
        previous = gate_set
        used_gates.update(gate_set)
        seen_levels.add(level)
        normalized_requirements.append((level, gate_names))
    require(used_gates == set(gates),
            "claim gates are not represented exactly in the requirements")

    satisfied: list[str] = []
    for level, gate_names in normalized_requirements:
        if all(gates[name] for name in gate_names):
            satisfied.append(level)
        else:
            break
    if satisfied:
        claim_level = satisfied[-1]
        status = f"PASS_{claim_level}"
    else:
        claim_level = "NO_DEFENSIBLE_CLAIM"
        status = "STOP_FAILED_INTEGRITY_GATES"

    next_requirement: tuple[str, tuple[str, ...]] | None = None
    if len(satisfied) < len(normalized_requirements):
        next_requirement = normalized_requirements[len(satisfied)]
    blocking = [] if next_requirement is None else [
        name for name in next_requirement[1] if not gates[name]
    ]
    scope = {} if evidence_scope is None else _canonical_value(
        evidence_scope, path="evidence_scope"
    )
    core = {
        "schema_version": "1.0.0",
        "status": status,
        "claim_level": claim_level,
        "gate_results": {name: gates[name] for name in sorted(gates)},
        "requirements": {
            level: list(gate_names) for level, gate_names in normalized_requirements
        },
        "satisfied_levels": satisfied,
        "next_level": None if next_requirement is None else next_requirement[0],
        "blocking_gates": blocking,
        "evidence_scope": scope,
    }
    return {
        **core,
        "contract_sha256": canonical_sha256(
            core, domain="DNABR_CLAIM_LEVEL_CONTRACT_V1"
        ),
    }
