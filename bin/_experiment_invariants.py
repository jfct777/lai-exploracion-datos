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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DNA_ALLELE_RE = re.compile(r"^[ACGT]+$")


STANDARD_CLAIM_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "TECHNICAL_ONLY",
        (
            "inputs_authenticated",
            "schemas_valid",
            "locus_partition_valid",
        ),
    ),
    (
        "EXPLORATORY",
        (
            "inputs_authenticated",
            "schemas_valid",
            "locus_partition_valid",
            "roles_separated",
            "selection_isolated",
            "fixture_matches_production",
            "null_invariants_valid",
        ),
    ),
    (
        "CONFIRMATORY",
        (
            "inputs_authenticated",
            "schemas_valid",
            "locus_partition_valid",
            "roles_separated",
            "selection_isolated",
            "fixture_matches_production",
            "null_invariants_valid",
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
