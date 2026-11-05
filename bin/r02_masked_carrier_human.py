"""Authenticate the human masked-carrier metadata path without reading genotypes.

This preflight binds the frozen FIT roster, calendar, source metadata, exact cell
mask geometry and per-target family exclusions.  It deliberately cannot admit
human training; EVENT pair recipes come from ``r02_event_pair_manifest.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from r02_masked_genotype_model import MaskedModelConfig
from r02_masked_store import MaskedStoreReader
from r02_masked_training_batch import plan_masked_cell


REQUEST_SCHEMA = "r02_masked_carrier_human_preflight_request_v1"
PAIR_SPEC_SCHEMA = "r02_masked_carrier_event_pair_producer_spec_v1"
FAMILY_RULE = "COMPLETE_FIT_ROSTER_EXCLUDE_SELF_AND_SAME_PCRELATE_COMPONENT"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _document(descriptor, name):
    _require(isinstance(descriptor, dict)
             and set(descriptor) == {"path", "sha256"},
             f"exact {name} descriptor required")
    path = Path(descriptor["path"]).resolve()
    _require(path.is_file() and sha256(path) == descriptor["sha256"],
             f"{name} missing or changed")
    return json.loads(path.read_text()), path


def _validate_roster(roster):
    _require(roster.get("schema") == "r02_masked_chr22_fit_roster_v1"
             and roster.get("role") == "FIT", "frozen FIT roster required")
    people = tuple(roster.get("person_ids", ()))
    components = roster.get("person_components")
    _require(roster.get("count") == len(people) == 1460
             and len(set(people)) == len(people)
             and isinstance(components, dict) and set(components) == set(people)
             and all(isinstance(components[x], str) and components[x] for x in people),
             "complete ordered 1,460-person FIT roster/components required")
    _require(isinstance(roster.get("role_manifest_identity_sha256"), str)
             and len(roster["role_manifest_identity_sha256"]) == 64,
             "role manifest identity required")
    return people, components


def _calendar_fragments(calendar):
    for expected_update, update in enumerate(calendar.get("updates", ()), 1):
        _require(update.get("update") == expected_update
                 and isinstance(update.get("fragments"), list),
                 "calendar update order differs")
        for fragment in update["fragments"]:
            yield fragment


def _calendar_ordered(values, seed, visit, kind):
    """Reproduce the frozen calendar builder's newline-free JSON hash."""
    def key(value):
        encoded = json.dumps([seed, visit, kind, value], sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest(), value
    return sorted(values, key=key)


def _validate_calendar(calendar, roster_sha, people):
    _require(calendar.get("schema") == "r02_masked_chr22_calendar_v1"
             and calendar.get("status") == "FROZEN_METADATA_ONLY_NO_DOSAGE_READ"
             and calendar.get("chromosome") == 22
             and calendar.get("roster_sha256") == roster_sha
             and calendar.get("fit_people") == len(people)
             and calendar.get("cell_selection") ==
             "HASH_ORDER_OF_COMPLETE_FROZEN_CELL_AXIS_BY_SEED_AND_VISIT_NO_DOSAGE_OR_LOSS"
             and calendar.get("coverage", {}).get("no_cell_selected_from_dosage_or_loss") is True,
             "frozen genotype-independent chr22 calendar required")
    rows = tuple(_calendar_fragments(calendar))
    person_groups = [tuple(_calendar_ordered(
        people, calendar["seed"], 0, "coverage-people")[start:
            start+calendar["people_per_fragment"]])
        for start in range(0, len(people), calendar["people_per_fragment"])]
    flattened = []
    for visit in range(calendar["person_coverage_visits"]):
        cells = _calendar_ordered(range(calendar["cell_count"]),
                                  calendar["seed"], visit, "coverage-cells")
        flattened.extend((visit, 22, cells[index], group)
                         for index, group in enumerate(person_groups))
    _require(len(rows) == len(flattened) == calendar["person_cell_visits"] //
             calendar["people_per_fragment"], "calendar fragment count differs")
    for row, example in zip(rows, flattened):
        _require(row["visit"] == example[0]
                 and row["chromosome"] == example[1]
                 and row["cell"] == example[2]
                 and tuple(row["person_ids"]) == example[3],
                 "calendar differs from frozen genotype-independent traversal")
    return rows


def reference_policy(roster, target_ids):
    """Return hashes of full-axis eligibility rows, never a batch-level hash."""
    people, components = _validate_roster(roster)
    _require(set(target_ids) <= set(people), "target outside frozen FIT roster")
    axis_hash = digest(people)
    policy = {"schema": "r02_masked_carrier_reference_policy_v1",
              "family_rule": FAMILY_RULE, "reference_ids": people,
              "reference_axis_sha256": axis_hash,
              "role_manifest_identity_sha256":
                  roster["role_manifest_identity_sha256"]}
    policy_hash = digest(policy)
    rows = {}
    for target in target_ids:
        eligible = tuple(target != reference
                         and components[target] != components[reference]
                         for reference in people)
        _require(not eligible[people.index(target)], "self reference remained eligible")
        _require(all(not keep or components[target] != components[reference]
                     for keep, reference in zip(eligible, people)),
                 "same-component reference remained eligible")
        rows[target] = {
            "target_component": components[target],
            "eligible_reference_count": int(sum(eligible)),
            "eligible_row_sha256": digest(eligible),
            "reference_policy_sha256": policy_hash,
        }
    return policy, rows


def event_pair_producer_spec(calendar, campaign, full_request):
    """Describe the implemented geometry producer without admitting a radius."""
    return {
        "schema": PAIR_SPEC_SCHEMA,
        "status": "IMPLEMENTED_GEOMETRY_ONLY_NOT_EXECUTED_NOT_ADMITTED",
        "producer": "bin/r02_event_pair_manifest.py",
        "locus_source": "AUTHENTICATED_EVENT_SOURCE_MINOR_DNABR2723_AXIS",
        "segment_axis": "FROZEN_CHR22_CELL_AXIS_FROM_CALENDAR_AND_CELLS_SHA256",
        "segment_identity_fields": ["chromosome", "cell_index", "cells_sha256"],
        "pair_identity_fields": ["left_locus_id", "right_locus_id"],
        "territory": "[a-r,a)_UNION_[b,b+r]_VISIBLE_FLANKS_ONLY",
        "radius_meaning": "MAXIMUM_LOCUS_TO_HIDDEN_CELL_DISTANCE_NOT_PAIR_DISTANCE",
        "ordering": "LEFT_LEFT_THEN_LEFT_RIGHT_THEN_RIGHT_RIGHT_LEXICOGRAPHIC",
        "selection_forbidden": ["target_dosage", "carrier_status", "loss",
                                "frequency", "family", "score", "microbatch",
                                "SELECT", "EVALUATE"],
        "required_receipt_fields": [
            "event_locus_axis_sha256", "segment_axis_sha256",
            "candidate_recipes", "manifest_sha256", "source_minor"],
        "compact_indices": ["left_start", "left_stop", "hidden_start",
                            "hidden_stop", "right_start", "right_stop"],
        "cost_formula":
            "P_s=C(L_left,2)+L_left*L_right+C(L_right,2)",
        "overflow_rule": "STOP_FOR_REVIEW_NEVER_TRUNCATE_OR_SUBSAMPLE",
        "zero_pair_rule": "ZERO_CARRIER_CORRECTION_AND_ZERO_SUPPORT_NO_FAKE_PAIR",
        "candidate_radius_cm": [0.05, 0.2, 1.0],
        "historical_full_radius_cm": full_request["model"]["rare_radius_cm"],
        "historical_radius_role":
            "MATCHED_INFORMATION_PRIMARY_SAME_LOCAL_TERRITORY_NOT_OPTIMAL_GENOMIC_SCALE",
        "radius_adopted": None,
        "cells_sha256": campaign["source"]["cells_sha256"],
        "calendar_axis_cells_sha256": calendar["axis_cells_sha256"],
        "source_minor": campaign["source"]["rare_counted_allele"],
        "uses_genotype_dosage": False,
    }


def _validate_pair_manifest(value, pair_spec):
    _require(value.get("schema") == "r02_masked_carrier_event_pair_manifest_v1"
             and value.get("status") ==
                "DESCRIPTIVE_GEOMETRY_ONLY_NOT_TRAINING_ADMISSION"
             and value.get("source_minor") == pair_spec["source_minor"]
             and value.get("candidate_radius_cm") ==
                pair_spec["candidate_radius_cm"]
             and value.get("matched_information_primary_radius_cm") == 0.2
             and value.get("radius_adopted") is None
             and value.get("human_training_admitted") is False
             and value.get("training_performed") is False
             and value.get("global_pair_tensor_materialized") is False,
             "closed geometry-only EVENT pair inventory required")
    _require(value.get("source_metadata", {}).get("cells_tsv_sha256") ==
             pair_spec["cells_sha256"]
             and value.get("source_metadata", {}).get(
                 "dosage_or_observed_opened") is False
             and value.get("source_metadata", {}).get(
                 "frequency_or_counted_allele_opened") is False,
             "EVENT pair source/cell identity differs")
    identity = value.get("manifest_sha256")
    _require(isinstance(identity, str) and len(identity) == 64
             and digest({key: item for key, item in value.items()
                         if key not in ("manifest_sha256", "elapsed_seconds",
                                       "peak_rss_bytes")}) == identity,
             "EVENT pair manifest internal identity differs")


def preflight(request, *, reader_factory=MaskedStoreReader):
    """Authenticate metadata and return a non-admitting, inspectable plan."""
    _require(request.get("schema") == REQUEST_SCHEMA
             and request.get("human_training_admitted") is False
             and request.get("select_opened") is False
             and request.get("evaluate_opened") is False,
             "closed metadata-only human preflight required")
    campaign, _ = _document(request.get("campaign"), "campaign")
    full, _ = _document(request.get("historical_full_request"),
                        "historical FULL request")
    roster, roster_path = _document(campaign.get("fit_roster"), "FIT roster")
    calendar, _ = _document(campaign.get("calendar"), "calendar")
    people, components = _validate_roster(roster)
    rows = _validate_calendar(calendar, sha256(roster_path), people)
    _require(campaign.get("schema") == "r02_masked_chr22_campaign_v1"
             and campaign.get("role") == "FIT"
             and campaign.get("chromosome") == 22
             and campaign.get("select_opened") is False
             and campaign.get("evaluate_opened") is False
             and campaign.get("source", {}).get("rare_counted_allele") ==
                 "source_minor_DNABR2723",
             "closed chr22 FIT campaign required")
    _require(full.get("schema") == "r02_masked_chr22_fit_request_v1"
             and full.get("condition") == "full"
             and full.get("select_opened") is False
             and full.get("evaluate_opened") is False,
             "closed historical FULL request required")
    resources = campaign["reader_resources"]
    source = campaign["source"]
    reader = reader_factory(
        source["store"], source["manifest_sha256"], source["axis_directory"],
        source["axis_receipt_sha256"], role="FIT",
        max_people=resources["max_people_per_fragment"],
        max_window_sites=resources["max_window_sites"],
        max_cache_entries=resources["max_cache_entries"])
    source_fit = tuple(p["sample_id"] for p in reader.reader.persons
                       if p["role"] == "FIT")
    _require(source_fit == people and all(
        reader.reader.persons[reader.reader.people[x]]["component"] == components[x]
        for x in people), "roster order/components differ from authenticated source")
    _require(reader.cells_sha256 == calendar["axis_cells_sha256"]
             == source["cells_sha256"], "calendar/source cell axis differs")
    model = MaskedModelConfig(**full["model"])
    exclusion = request.get("context_exclusion_cm")
    _require(type(exclusion) in (int, float) and math.isfinite(exclusion)
             and 0 <= exclusion < min(model.common_radius_cm,
                                      model.rare_radius_cm),
             "finite context exclusion below both model radii required")
    boundary_rule = ("HALF_OPEN_CELLMASKPLAN_CLOSED_LAST_CELL"
                     if exclusion == 0 else "CLOSED_EXPLICIT_MARGIN")
    mask_rows, target_visits = [], {}
    for row in rows:
        plan = plan_masked_cell(reader, model, person_ids=tuple(row["person_ids"]),
                                cell_index=row["cell"],
                                context_exclusion_cm=exclusion)
        _require(np.float64(plan.hidden_left_cm) ==
                 np.float64(row["cell_left_cm"]-exclusion)
                 and np.float64(plan.hidden_right_cm) ==
                 np.float64(row["cell_right_cm"]+exclusion),
                 "calendar cell differs from shared masking geometry")
        mask_rows.append({"visit": row["visit"], "cell": row["cell"],
            "person_ids": row["person_ids"], "mask_identity": plan.mask.identity,
            "hidden_territory_cm": [plan.hidden_left_cm, plan.hidden_right_cm],
            "boundary_rule": boundary_rule, "halo_cm": plan.halo_cm})
        for target in row["person_ids"]:
            target_visits.setdefault(target, 0); target_visits[target] += 1
    policy, target_rows = reference_policy(roster, people)
    _require(set(target_rows) == set(target_visits)
             and set(target_visits.values()) == {calendar["person_coverage_visits"]},
             "calendar target coverage differs")
    pair_spec = event_pair_producer_spec(calendar, campaign, full)
    pair_descriptor = request.get("event_pair_manifest")
    dependencies = []
    if pair_descriptor is None:
        dependencies.append({"name": "event_pair_manifest",
            "status": "MISSING", "required_schema":
                "r02_masked_carrier_event_pair_manifest_v1"})
    else:
        pair_manifest, _ = _document(pair_descriptor, "EVENT pair manifest")
        _validate_pair_manifest(pair_manifest, pair_spec)
    return {
        "schema": "r02_masked_carrier_human_preflight_v1",
        "status": ("BLOCKED_DEPENDENCY_EVENT_PAIR_MANIFEST" if dependencies
                   else "METADATA_READY_TRAINING_STILL_NOT_ADMITTED"),
        "campaign_sha256": request["campaign"]["sha256"],
        "roster_sha256": campaign["fit_roster"]["sha256"],
        "calendar_sha256": campaign["calendar"]["sha256"],
        "historical_full_request_sha256": request["historical_full_request"]["sha256"],
        "source_metadata_authenticated": True,
        "genotype_matrices_read": False,
        "target_count": len(people), "fragment_count": len(rows),
        "reference_policy": policy,
        "target_reference_rows": target_rows,
        "mask_plan_sha256": digest(mask_rows),
        "mask_plan": mask_rows,
        "context_exclusion_cm": exclusion,
        "boundary_rule": boundary_rule,
        "event_pair_producer_spec": pair_spec,
        "dependencies": dependencies,
        "technical_limits_select_loci_or_references": False,
        "human_training_admitted": False,
        "training_performed": False, "neural_forward_performed": False,
        "select_opened": False, "evaluate_opened": False,
        "code_sha256": sha256(__file__),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request_path, output = Path(args.request).resolve(), Path(args.output).resolve()
    _require(request_path.is_file(), "request missing")
    _require(not output.exists(), "fresh preflight output required")
    result = preflight(json.loads(request_path.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical(result) + b"\n")


if __name__ == "__main__":
    main()
