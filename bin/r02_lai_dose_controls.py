"""Personal-dose interventions retaining EVENT geometry and observation masks.

These are input interventions, not exchangeable genetic nulls. Population rare
reference information remains. They do not authorize training or evaluation.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from r02_lai_control_views import array_sha256, require
from r02_lai_query_burden import build_prefix, project

ALGORITHM = "OBSERVED_PERSON_DOSE_SHA256_PCG64_V1"
REMOVED = "PERSONAL_DOSE_REMOVED_FIXED_SUPPORT"
PERMUTED = "PERSONAL_DOSE_PERMUTED_FIXED_SUPPORT"
RETAINED_BATCH_KEYS = frozenset("""
baseline_probabilities common_alt common_chrom common_cm common_core common_dosage
common_features common_global_index common_is_alt common_padding common_pos_bp common_ref common_valid
component_id person_id role event_alt event_annotation_is_padding event_annotation_schema
event_call_provenance_count_order event_call_provenance_counts event_call_provenance_known
event_call_provenance_role_order event_chrom event_cm event_copyable_in_case_fold event_copyable_known
event_core event_counted_allele event_dosage event_global_index event_is_alt event_metadata event_origin
event_padding event_phase_status event_pos_bp event_quality_status event_ref event_reference_ac
event_reference_an event_reference_ancestry_order event_reference_known event_source_catalogue_sha256
event_valid event_reference_features query_chrom query_cm query_core query_features query_global_index
query_pos_bp query_valid query_burden query_score_weight_cm truth_state truth_observed
""".split())


def _ids(values, name):
    values = np.asarray(values)
    require(values.ndim == 1 and values.dtype.kind in "US", name + " must be text axis")
    result = values.astype(str).tolist()
    require(all(result) and len(set(result)) == len(result), name + " empty or duplicated")
    return result


def transform(dose, observed, *, person_ids, expected_person_ids, locus_keys,
              role, expected_role, stratum_id, seed):
    """Whole-person-axis transform; independent locus RNG permits locus chunking.

    Membership must be authenticated by the caller, not inferred from this batch.
    Related mosaics need not be exchangeable: no p-value or conditional-null claim.
    Zero in REMOVED is an unavailable-channel sentinel, not an observed genotype.
    """
    dose, observed = np.asarray(dose), np.asarray(observed)
    people = _ids(person_ids, "person IDs")
    expected = _ids(expected_person_ids, "expected person IDs")
    keys = _ids(locus_keys, "locus keys")
    require(people and expected and set(people) == set(expected), "complete declared people required")
    require(role == expected_role == "FIT", "only explicitly declared FIT intervention supported")
    require(isinstance(stratum_id, str) and stratum_id, "declared case/cohort stratum required")
    require(type(seed) is int and 0 <= seed < 2**63, "invalid seed")
    require(dose.shape == observed.shape == (len(people), len(keys))
            and observed.dtype == np.bool_, "dose/observation axes differ")
    require(dose.dtype.kind in "iuf" and np.isin(dose[observed], (0, 1, 2)).all(),
            "observed doses must be diploid source-minor counts")
    canonical = np.argsort(np.asarray(people))
    universe = hashlib.sha256(json.dumps(sorted(expected), separators=(",", ":")).encode()).hexdigest()
    shuffled = dose.copy()
    changed = moved = permutable = observed_count = 0
    for locus, key in enumerate(keys):
        ids = canonical[observed[canonical, locus]]
        observed_count += len(ids)
        permutable += int(len(np.unique(dose[ids, locus])) > 1)
        material = json.dumps([ALGORITHM, seed, stratum_id, role, universe, key],
                              separators=(",", ":")).encode()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(material).digest()[:16], "little")))
        sources = rng.permutation(ids)
        shuffled[ids, locus] = dose[sources, locus]
        moved += int(np.count_nonzero(sources != ids))
        changed += int(np.count_nonzero(dose[ids, locus] != shuffled[ids, locus]))
    shuffled[~observed] = 0
    removed = np.zeros_like(dose)
    receipt = dict(algorithm=ALGORITHM, role=role, stratum_id=stratum_id, seed=seed,
        people=len(people), loci=len(keys), universe_sha256=universe,
        original_dose_sha256=array_sha256(dose, np), observation_sha256=array_sha256(observed, np),
        observed_cells=int(observed_count), shuffled_indices_moved=moved,
        shuffled_doses_changed=changed, loci_with_distinct_observed_doses=permutable,
        permutation_effect="ZERO_CHANGED_DOSES" if changed == 0 else "NONZERO_CHANGED_DOSES_NOT_POWER_CERTIFICATION",
        missing_dose_storage="ZERO_MASKED_SENTINEL_OBSERVATION_MASK_UNCHANGED",
        removed_channel_semantics="DOSAGE_UNAVAILABLE_NOT_HOMOZYGOUS_REFERENCE",
        retained_information="EVENT_CATALOGUE_GEOMETRY_MASK_AND_POPULATION_REFERENCE",
        statistical_null_certified=False, training_authorized=False, evaluate_opened=False)
    return {REMOVED: removed, PERMUTED: shuffled}, receipt


def chromosome_prefixes(views, observed, event_cm, event_chrom, *,
                        person_ids, event_global_index, expected_event_count):
    """Counts from the entire authenticated chromosome, never a regional slice."""
    index = np.asarray(event_global_index)
    require(type(expected_event_count) is int and expected_event_count >= 0
            and index.dtype.kind in "iu" and index.ndim == 1
            and np.array_equal(np.sort(index), np.arange(expected_event_count)),
            "complete declared chromosome EVENT axis required for counts")
    require(set(views) == {REMOVED, PERMUTED}, "both explicit personal-dose views required")
    require(np.asarray(observed).ndim == 2 and observed.shape[1] == expected_event_count,
            "observation axis differs from chromosome")
    require(len(_ids(person_ids, "prefix people")) == observed.shape[0], "prefix people shape differs")
    require(len(np.unique(np.asarray(event_chrom))) == 1, "single declared chromosome required")
    return {name: dict(build_prefix(dose, observed, event_cm, event_chrom, "NONE", np), person_id=np.asarray(person_ids).copy())
            for name, dose in views.items()}


def regional_view(original, *, condition, views, prefixes, whole_person_ids, whole_observed,
                  whole_event_global_index, regional_event_global_index, whole_event_cm,
                  whole_event_chrom, radius_cm, expected_radius_cm):
    """Use full-chromosome products before microbatching; replace only dose/counts.

    Original must be the real-dose view, with source hashes checked by its caller.
    This function does not change its receipt to claim an admitted training batch.
    """
    require(condition in (REMOVED, PERMUTED), "unknown intervention")
    require(set(original) <= RETAINED_BATCH_KEYS, "unreviewed array in regional view")
    require(radius_cm == expected_radius_cm, "radius differs from authenticated original request")
    prefix = prefixes[condition]
    require(np.array_equal(prefix["person_id"], whole_person_ids), "prefix person axis differs")
    order = prefix["event_order"]
    require(np.array_equal(prefix["event_cm"], np.asarray(whole_event_cm)[order])
            and np.array_equal(prefix["event_chrom"], np.asarray(whole_event_chrom)[order]),
            "prefix coordinate identity differs")
    people, region_people = _ids(whole_person_ids, "whole people"), _ids(original["person_id"], "region people")
    lookup = {value: i for i, value in enumerate(people)}
    require(all(value in lookup for value in region_people), "region people outside declared universe")
    global_ids = np.asarray(whole_event_global_index)
    local_ids = np.asarray(regional_event_global_index)
    require(global_ids.ndim == local_ids.ndim == 1 and global_ids.dtype.kind in "iu"
            and local_ids.dtype.kind in "iu" and len(set(global_ids.tolist())) == len(global_ids),
            "invalid EVENT identity axes")
    positions = {int(value): i for i, value in enumerate(global_ids)}
    require(all(int(value) in positions for value in local_ids), "region locus outside whole axis")
    rows = np.asarray([lookup[value] for value in region_people], dtype=np.int64)
    cols = np.asarray([positions[int(value)] for value in local_ids], dtype=np.int64)
    for key, full in (("event_cm", whole_event_cm), ("event_chrom", whole_event_chrom)):
        expected_axis = np.broadcast_to(np.asarray(full)[cols], original["event_dosage"].shape)
        require(np.array_equal(original[key], expected_axis), "regional coordinate identity differs: " + key)
    selected = views[condition][rows[:, None], cols[None, :]]
    require(selected.shape == original["event_dosage"].shape, "regional dose shape differs")
    require(np.asarray(whole_observed).dtype == np.bool_
            and np.array_equal(whole_observed[rows[:, None], cols[None, :]], original["event_valid"]),
            "regional observation mask differs from fixed-support source")
    result = dict(original)
    result["event_dosage"] = selected.copy()
    # Prefix arrays span all people; query coordinates are shared in these inputs.
    prefix = dict(prefix); prefix["prefix_counts"] = prefix["prefix_counts"][rows]
    result["query_burden"] = project(prefix, original["query_cm"], original["query_chrom"], radius_cm, np)
    result["personal_dose_intervention"] = condition
    result["personal_dose_available"] = condition != REMOVED
    return result
