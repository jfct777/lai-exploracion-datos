#!/usr/bin/env python3
"""Audit and run the fixed chr22 FIT masked-genotype campaign.

The campaign contains six principal fits plus two rare-identity diagnostics.
Deep Sets and local attention each use ``no_rare``, ``count_only`` or ``full``;
the two diagnostics refit ``full`` after a frozen visible-locus permutation.
All eight reuse one frozen calendar. The calendar gives every one of 1,460 FIT people exactly
two cells (2,920 person/cell visits); it is not two passes over 7,267 cells.

``audit-denominators`` reads the scheduled cells without constructing a model,
records observed COMMON and RARE target counts for every Adam update, and never
retries or replaces a cell. ``audit-rare-identity`` freezes and measures both
family-specific permutations. ``run`` requires the applicable audits and an
exact PRE acceptance. SELECT and EVALUATE are unavailable here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from r02_masked_training_campaign import (COVERAGE_SCHEDULE_VERSION, CellExample,
    fit_cell_loader, run_masked_campaign)


CAMPAIGN_SCHEMA = "r02_masked_chr22_campaign_v1"
CAMPAIGN_STATUS = "FROZEN_FOR_EF_REVIEW_NOT_LAUNCHED"
CALENDAR_SCHEMA = "r02_masked_chr22_calendar_v1"
REQUEST_SCHEMA = "r02_masked_chr22_fit_request_v1"
REQUEST_STATUS = "PREPARED_FOR_EF_REVIEW_NOT_LAUNCHED"
PRE_SCHEMA = "r02_masked_chr22_pre_v1"
PRE_STATUS = "ACCEPTED_FOR_FIT_AUDITS_AND_EIGHT_PAIRED_FITS"
AUDIT_PRE_STATUS = "ACCEPTED_FOR_FIT_DENOMINATOR_AUDIT_ONLY"
AUDIT_SCHEMA = "r02_masked_chr22_denominator_audit_v1"
AUDIT_STATUS = "365_READ_GROUPS_FROZEN_INTO_353_POSITIVE_DENOMINATOR_UPDATES"
AUDIT_SOURCE_SCHEMA = "r02_masked_chr22_denominator_audit_source_v1"
RARE_AUDIT_SCHEMA = "r02_masked_chr22_rare_identity_audit_v1"
RARE_AUDIT_STATUS = "TWO_FAMILIES_FROZEN_WITH_NONZERO_ALTERATION"
RECEIPT_SCHEMA = "r02_masked_chr22_fit_receipt_v1"
FAMILIES = ("local_deepsets", "local_attention")
CONDITIONS = ("no_rare", "count_only", "full")
PRIMARY_FIT_IDS = tuple(family + "__" + condition
                        for family in FAMILIES for condition in CONDITIONS)
RARE_IDENTITY_FIT_IDS = tuple(family + "__full__rare_identity"
                              for family in FAMILIES)
ALL_FIT_IDS = PRIMARY_FIT_IDS + RARE_IDENTITY_FIT_IDS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2,
                                     allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)


def is_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def checked_descriptor(value, name):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and value["path"]
            and is_hash(value["sha256"]), "invalid " + name + " descriptor")
    path = Path(value["path"])
    require(path.is_file() and sha(path) == value["sha256"], name + " differs")
    return path


def code_hashes():
    names = ("r02_masked_training_request.py", "r02_masked_training_campaign.py",
             "r02_masked_training_batch.py", "r02_masked_genotype_controls.py",
             "r02_masked_flank_diagnostic.py",
             "r02_masked_genotype_update.py", "r02_masked_genotype_loss.py",
             "r02_masked_genotype_model.py", "r02_masked_genotypes.py",
             "r02_masked_store.py", "r02_masked_fit_budget.py")
    root = Path(__file__).resolve().parent
    return {name: sha(root / name) for name in names}


def validate_calendar(calendar, campaign):
    require(calendar.get("schema") == CALENDAR_SCHEMA
            and calendar.get("status") == "FROZEN_METADATA_ONLY_NO_DOSAGE_READ",
            "calendar schema/status differs")
    required = {"schema", "status", "schedule_version", "seed", "chromosome",
                "cell_count", "fit_people", "people_per_fragment",
                "fragments_per_update", "person_coverage_visits",
                "calendar_read_groups", "person_cell_visits", "cell_selection",
                "roster_sha256", "axis_cells_sha256", "updates", "coverage"}
    require(set(calendar) == required
            and calendar["schedule_version"] == COVERAGE_SCHEDULE_VERSION
            and calendar["seed"] == 1701 and calendar["chromosome"] == 22
            and calendar["cell_count"] == 7267 and calendar["fit_people"] == 1460
            and calendar["people_per_fragment"] == 4
            and calendar["fragments_per_update"] == 2
            and calendar["person_coverage_visits"] == 2
            and calendar["calendar_read_groups"] == 365
            and calendar["person_cell_visits"] == 2920,
            "calendar units differ from the reviewed 365-update design")
    require(calendar["roster_sha256"] == campaign["fit_roster"]["sha256"]
            and calendar["axis_cells_sha256"] == campaign["source"]["cells_sha256"],
            "calendar source identities differ")
    updates = calendar["updates"]
    require(isinstance(updates, list) and len(updates) == 365,
            "calendar must contain exactly 365 updates")
    visits = {}
    pairs = set()
    for update_index, update in enumerate(updates, 1):
        require(update.get("update") == update_index
                and isinstance(update.get("fragments"), list)
                and len(update["fragments"]) == 2,
                "calendar update/fragments differ")
        for fragment in update["fragments"]:
            require(set(fragment) == {"visit", "chromosome", "cell", "cell_left_cm",
                                      "cell_right_cm", "person_ids"}
                    and fragment["visit"] in (0, 1)
                    and fragment["chromosome"] == 22
                    and type(fragment["cell"]) is int
                    and 0 <= fragment["cell"] < 7267
                    and len(fragment["person_ids"]) == 4
                    and fragment["cell_right_cm"] > fragment["cell_left_cm"],
                    "calendar fragment differs")
            for person in fragment["person_ids"]:
                visits[person] = visits.get(person, 0) + 1
                require((person, fragment["cell"]) not in pairs,
                        "calendar repeats a person/cell pair")
                pairs.add((person, fragment["cell"]))
    require(len(visits) == 1460 and set(visits.values()) == {2}
            and len(pairs) == 2920,
            "calendar does not give exactly two distinct cell visits per FIT person")
    require(calendar["coverage"] == {
        "persons_with_exactly_two_visits": 1460,
        "minimum_person_visits": 2, "maximum_person_visits": 2,
        "distinct_person_cell_pairs": 2920,
        "no_cell_selected_from_dosage_or_loss": True},
        "calendar coverage receipt differs")
    return calendar


def validate_campaign(campaign):
    required = {"schema", "status", "role", "chromosome", "source", "fit_roster",
                "calendar", "accumulation_plan", "protocol", "code_sha256", "objective", "shared_model",
                "denominator_audit_model", "reader_resources", "control_status",
                "select_opened", "evaluate_opened"}
    require(isinstance(campaign, dict) and set(campaign) == required
            and campaign["schema"] == CAMPAIGN_SCHEMA
            and campaign["status"] == CAMPAIGN_STATUS
            and campaign["role"] == "FIT" and campaign["chromosome"] == 22
            and campaign["select_opened"] is False
            and campaign["evaluate_opened"] is False,
            "campaign fields, scope or role differ")
    require(campaign["code_sha256"] == code_hashes(), "campaign code hashes differ")
    source = campaign["source"]
    require(set(source) == {"store", "manifest_sha256", "axis_directory",
                            "axis_receipt_sha256", "cells_sha256",
                            "fit_common_frequency", "fit_rare_frequency",
                            "common_counted_allele", "rare_counted_allele"}
            and Path(source["store"]).is_absolute()
            and Path(source["axis_directory"]).is_absolute()
            and all(is_hash(source[key]) for key in
                    ("manifest_sha256", "axis_receipt_sha256", "cells_sha256"))
            and source["common_counted_allele"] == "ALT"
            and source["rare_counted_allele"] == "source_minor_DNABR2723",
            "source paths, hashes or counted-allele definitions differ")
    checked_descriptor(source["fit_common_frequency"], "FIT COMMON frequency")
    checked_descriptor(source["fit_rare_frequency"], "FIT RARE frequency")
    roster_path = checked_descriptor(campaign["fit_roster"], "FIT roster")
    roster = read_json(roster_path)
    require(roster.get("schema") == "r02_masked_chr22_fit_roster_v1"
            and roster.get("role") == "FIT" and roster.get("count") == 1460
            and len(roster.get("person_ids", [])) == 1460
            and len(set(roster["person_ids"])) == 1460,
            "FIT roster differs")
    checked_descriptor(campaign["protocol"], "campaign protocol")
    calendar_path = checked_descriptor(campaign["calendar"], "calendar")
    calendar = validate_calendar(read_json(calendar_path), campaign)
    plan_path = checked_descriptor(campaign["accumulation_plan"], "accumulation plan")
    plan = read_json(plan_path)
    require(plan.get("schema") == "r02_masked_chr22_accumulation_plan_v1"
            and plan.get("status") == "AUDITED_ACCUMULATION_PLAN"
            and plan.get("calendar_sha256") == campaign["calendar"]["sha256"]
            and plan.get("calendar_groups") == 365
            and plan.get("optimizer_updates") == 353
            and plan.get("fragments") == 730
            and plan.get("person_cell_visits") == 2920
            and plan.get("cells_reordered_retried_or_replaced") is False,
            "audited accumulation plan differs")
    objective = campaign["objective"]
    require(objective == {"name": "EQUAL_CHANNEL_OBSERVED_CROSS_ENTROPY",
        "equation": "0.5*common_cross_entropy_sum/common_observed_count+0.5*rare_cross_entropy_sum/rare_observed_count",
        "common_weight": 0.5, "rare_weight": 0.5,
        "target_dosages": [0, 1, 2],
        "empty_fragment_rule": "KEEP_FRAGMENT_AND_ADD_ZERO_TO_ITS_CHANNEL_SUM_AND_COUNT",
        "empty_update_rule": "STOP_AS_INCOMPATIBLE_WITH_EQUAL_CHANNEL_MEANS_DO_NOT_RESAMPLE"},
        "masked-genotype objective differs")
    shared = campaign["shared_model"]
    require(shared == {"width": 32, "projection_depth": 2, "dropout": 0,
        "common_radius_cm": 0.2, "rare_radius_cm": 0.2,
        "position_bp_scale": 50818468, "position_cm_scale": 74.109562,
        "max_loci_per_channel": 100000, "query_block_size": 16,
        "max_mask_elements": 1000000, "checkpoint_blocks": False},
        "shared model configuration differs")
    require(campaign["denominator_audit_model"] == {
        **shared, "family": "local_attention", "attention_depth": 2, "heads": 4},
        "denominator audit must use the largest reviewed 0.6 cM model halo")
    resources = campaign["reader_resources"]
    require(resources == {"max_people_per_fragment": 4, "max_window_sites": 100000,
                          "max_cache_entries": 2000, "torch_threads": 1},
            "reader resources differ")
    return roster, calendar


def validate_request(request, campaign, campaign_path):
    required = {"schema", "status", "fit_id", "campaign", "family", "condition",
                "seed", "model", "optimizer", "updates", "calendar_read_groups", "training_input_control",
                "resources", "select_opened", "evaluate_opened"}
    require(isinstance(request, dict) and set(request) == required
            and request["schema"] == REQUEST_SCHEMA and request["status"] == REQUEST_STATUS,
            "fit request fields/schema/status differ")
    require(Path(request["campaign"]["path"]).resolve() == Path(campaign_path).resolve()
            and request["campaign"]["sha256"] == sha(campaign_path),
            "fit request belongs to another campaign")
    primary_id = request["family"] + "__" + request["condition"]
    control_id = request["family"] + "__full__rare_identity"
    is_primary = request["fit_id"] == primary_id
    is_control = request["fit_id"] == control_id and request["condition"] == "full"
    require(request["family"] in FAMILIES and request["condition"] in CONDITIONS
            and (is_primary or is_control)
            and request["seed"] == 1701 and request["updates"] == 353
            and request["calendar_read_groups"] == 365
            and request["training_input_control"] == (
                "RARE_IDENTITY_SHA256_V1" if is_control else "NONE")
            and request["select_opened"] is False
            and request["evaluate_opened"] is False,
            "fit identity, condition, budget, or role boundary differs")
    expected = dict(campaign["shared_model"])
    expected.update({"family": request["family"],
                     "attention_depth": 0 if request["family"] == "local_deepsets" else 2,
                     "heads": None if request["family"] == "local_deepsets" else 4})
    require(request["model"] == expected, "fit model differs")
    require(request["optimizer"] == {"learning_rate": math.sqrt(1e-7),
                                      "adam_state_precision": "fp64"},
            "Adam configuration differs")
    require(request["resources"] == {
        "cpus": 1, "memory_bytes": 8589934592, "max_seconds": 28800},
        "per-fit resource ceiling differs")
    return request


def validate_pre(pre, campaign_path, request_paths, *, audit_only_allowed=False):
    expected = {Path(path).name: sha(path) for path in request_paths}
    common = (pre.get("schema") == PRE_SCHEMA
            and pre.get("campaign_sha256") == sha(campaign_path)
            and pre.get("request_sha256") == expected
            and pre.get("authorized_role") == "FIT"
            and pre.get("denominator_audit_authorized") is True
            and pre.get("select_opened") is False
            and pre.get("evaluate_opened") is False)
    full = (pre.get("status") == PRE_STATUS
            and pre.get("authorized_updates_per_fit") == 353
            and pre.get("authorized_fit_ids") == sorted(ALL_FIT_IDS))
    audit_only = (audit_only_allowed and pre.get("status") == AUDIT_PRE_STATUS
                  and pre.get("authorized_updates_per_fit") == 0
                  and pre.get("authorized_fit_ids") == [])
    require(common and (full or audit_only),
            "accepted EF PRE does not bind the exact campaign and eight requests")


@dataclass(frozen=True)
class FrozenCalendarSchedule:
    calendar: dict
    calendar_sha256: str
    person_ids: tuple[str, ...]

    @property
    def update_count(self):
        return 365

    @property
    def minimum_updates_for_all_people(self):
        return 183

    def identity_payload(self):
        return {"schema": "r02_masked_frozen_calendar_schedule_v1",
                "calendar_sha256": self.calendar_sha256,
                "updates": 365, "fit_people": 1460,
                "person_coverage_visits": 2}

    def updates(self):
        for update in self.calendar["updates"]:
            yield tuple(CellExample(fragment["visit"], fragment["chromosome"],
                                    fragment["cell"], tuple(fragment["person_ids"]))
                        for fragment in update["fragments"])


@dataclass(frozen=True)
class AccumulatedCalendarSchedule:
    calendar: dict
    calendar_sha256: str
    plan: dict
    plan_sha256: str

    @property
    def update_count(self):
        return 353

    @property
    def minimum_updates_for_all_people(self):
        return 353

    def identity_payload(self):
        return {"schema": "r02_masked_accumulated_calendar_schedule_v1",
                "calendar_sha256": self.calendar_sha256,
                "accumulation_plan_sha256": self.plan_sha256,
                "calendar_read_groups": 365, "optimizer_updates": 353,
                "fit_people": 1460, "person_coverage_visits": 2}

    def updates(self):
        for row in self.plan["updates"]:
            examples = []
            for group in row["calendar_groups"]:
                for fragment in self.calendar["updates"][group - 1]["fragments"]:
                    examples.append(CellExample(fragment["visit"],
                        fragment["chromosome"], fragment["cell"],
                        tuple(fragment["person_ids"])))
            require(len(examples) == row["fragment_count"],
                    "accumulation plan fragment count differs")
            yield tuple(examples)


def open_reader(campaign):
    from r02_masked_store import MaskedStoreReader, role_person_ids
    source = campaign["source"]
    resources = campaign["reader_resources"]
    reader = MaskedStoreReader(source["store"], source["manifest_sha256"],
        source["axis_directory"], source["axis_receipt_sha256"], role="FIT",
        max_people=resources["max_people_per_fragment"],
        max_window_sites=resources["max_window_sites"],
        max_cache_entries=resources["max_cache_entries"])
    require(reader.chromosome == 22 and reader.cells_sha256 == source["cells_sha256"],
            "authenticated chr22 reader/cell axis differs")
    return reader, role_person_ids(reader.reader, "FIT")


def schedule_for(campaign, roster, calendar):
    plan = read_json(campaign["accumulation_plan"]["path"])
    return AccumulatedCalendarSchedule(calendar, campaign["calendar"]["sha256"],
        plan, campaign["accumulation_plan"]["sha256"])


def example_value(example):
    return {"visit": example.epoch, "chromosome": example.chromosome,
            "cell": example.cell, "person_ids": list(example.person_ids)}


def rare_territory_id(family, lineage, example):
    return digest({"schema": "r02_rare_identity_territory_v1", "family": family,
                   "example": example_value(example), "region_lineage": lineage})


def rare_identity_transform(family, expected=None, observed=None):
    from r02_masked_genotype_controls import (
        apply_rare_identity_plan, freeze_hashed_rare_identity_plan)

    def transform(*, fragment, lineage, example):
        territory = rare_territory_id(family, lineage, example)
        plan = freeze_hashed_rare_identity_plan(
            fragment.rare, territory_id=territory, seed=1701)
        key = digest(example_value(example))
        if expected is not None:
            require(key in expected and expected[key]["territory_id"] == territory
                    and expected[key]["plan_identity"] == plan.identity,
                    "rare-identity plan differs from reviewed audit")
        controlled = apply_rare_identity_plan(fragment.rare, plan,
                                              territory_id=territory)
        if observed is not None:
            visible = fragment.rare.genotype_visible
            source = np.asarray(plan.source_index)
            identity = np.tile(np.arange(source.shape[1]), (source.shape[0], 1))
            moved = int(((source != identity) & np.asarray(plan.genotype_visible)).sum())
            changed = int(((controlled.dosage != fragment.rare.dosage) & visible).sum().item())
            observed.append({"example": example_value(example), "example_sha256": key,
                "territory_id": territory, "plan_identity": plan.identity,
                "visible_loci": int(visible.sum().item()), "moved_loci": moved,
                "changed_doses": changed})
        return replace(fragment, rare=controlled,
                       lineage=fragment.lineage + ":rare_identity:" + plan.identity)
    return transform


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def validate_denominator_update(value, *, index, campaign_sha, calendar_sha,
                                examples):
    required = {"schema", "campaign_sha256", "calendar_sha256", "update",
                "examples", "fragments", "common_observed_count",
                "rare_observed_count", "cells_retried_or_replaced"}
    require(isinstance(value, dict) and set(value) == required
            and value["schema"] == "r02_masked_denominator_update_v1"
            and value["campaign_sha256"] == campaign_sha
            and value["calendar_sha256"] == calendar_sha
            and value["update"] == index and value["examples"] == examples
            and value["cells_retried_or_replaced"] is False,
            "recoverable denominator update belongs to another schedule")
    fragments = value["fragments"]
    require(isinstance(fragments, list) and len(fragments) == 2
            and all(isinstance(row, dict)
                    and set(row) == {"common_observed_count",
                                     "rare_observed_count", "lineage"}
                    and type(row["common_observed_count"]) is int
                    and row["common_observed_count"] >= 0
                    and type(row["rare_observed_count"]) is int
                    and row["rare_observed_count"] >= 0
                    and is_hash(row["lineage"]) for row in fragments),
            "denominator fragment counts or lineage differ")
    common = sum(row["common_observed_count"] for row in fragments)
    rare = sum(row["rare_observed_count"] for row in fragments)
    require(value["common_observed_count"] == common
            and value["rare_observed_count"] == rare,
            "denominator update totals differ from its fragments")
    return value


def audit_denominators(campaign_path, pre_path, request_paths, output):
    import torch
    from r02_masked_genotype_model import MaskedModelConfig
    campaign = read_json(campaign_path)
    roster, calendar = validate_campaign(campaign)
    requests = [validate_request(read_json(path), campaign, campaign_path)
                for path in request_paths]
    require({request["fit_id"] for request in requests} == set(ALL_FIT_IDS),
            "all eight paired fit requests are required")
    validate_pre(read_json(pre_path), campaign_path, request_paths,
                 audit_only_allowed=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reader, fit_ids = open_reader(campaign)
    require(set(roster["person_ids"]) == fit_ids,
            "frozen roster is not the complete authenticated FIT role")
    config = MaskedModelConfig(**campaign["denominator_audit_model"])
    torch.set_num_threads(campaign["reader_resources"]["torch_threads"])
    loader = fit_cell_loader({22: reader}, config, context_exclusion_cm=0, device="cpu")
    schedule = FrozenCalendarSchedule(calendar, campaign["calendar"]["sha256"],
                                      tuple(roster["person_ids"]))
    campaign_sha = sha(campaign_path)
    totals = {"common": 0, "rare": 0}
    zero_updates = []
    updates = []
    for index, examples in enumerate(schedule.updates(), 1):
        receipt_path = output / "updates" / f"update-{index:08d}.json"
        example_value = [{"visit": e.epoch, "chromosome": e.chromosome,
                          "cell": e.cell, "person_ids": list(e.person_ids)} for e in examples]
        if receipt_path.exists():
            value = read_json(receipt_path)
            validate_denominator_update(value, index=index,
                campaign_sha=campaign_sha,
                calendar_sha=campaign["calendar"]["sha256"],
                examples=example_value)
        else:
            common = rare = 0
            fragments = []
            for example in examples:
                fragment = loader(example)
                fragment_common = int(fragment.common_targets.score_mask.sum().item())
                fragment_rare = int(fragment.rare_targets.score_mask.sum().item())
                common += fragment_common; rare += fragment_rare
                fragments.append({"common_observed_count": fragment_common,
                                  "rare_observed_count": fragment_rare,
                                  "lineage": fragment.lineage})
            value = {"schema": "r02_masked_denominator_update_v1",
                     "campaign_sha256": campaign_sha,
                     "calendar_sha256": campaign["calendar"]["sha256"],
                     "update": index, "examples": example_value,
                     "fragments": fragments,
                     "common_observed_count": common,
                     "rare_observed_count": rare,
                     "cells_retried_or_replaced": False}
            validate_denominator_update(value, index=index,
                campaign_sha=campaign_sha,
                calendar_sha=campaign["calendar"]["sha256"],
                examples=example_value)
            atomic_json(receipt_path, value)
        totals["common"] += value["common_observed_count"]
        totals["rare"] += value["rare_observed_count"]
        if value["common_observed_count"] == 0 or value["rare_observed_count"] == 0:
            zero_updates.append(index)
        updates.append({"update": index,
                        "common_observed_count": value["common_observed_count"],
                        "rare_observed_count": value["rare_observed_count"],
                        "receipt_sha256": sha(receipt_path)})
    plan = read_json(campaign["accumulation_plan"]["path"])
    grouped_positive = True
    for planned in plan["updates"]:
        rows = [updates[group - 1] for group in planned["calendar_groups"]]
        grouped_positive &= (sum(row["common_observed_count"] for row in rows) > 0
                             and sum(row["rare_observed_count"] for row in rows) > 0)
    status = AUDIT_STATUS if grouped_positive else \
        "INCOMPATIBLE_ACCUMULATED_UPDATE_TRAINING_BLOCKED"
    result = {"schema": AUDIT_SCHEMA, "status": status,
        "campaign_sha256": campaign_sha,
        "calendar_sha256": campaign["calendar"]["sha256"],
        "pre_sha256": sha(pre_path),
        "request_sha256": {Path(path).name: sha(path) for path in request_paths},
        "updates_checked": 365, "fragments_checked": 730,
        "person_cell_visits_checked": 2920,
        "common_observed_count_total": totals["common"],
        "rare_observed_count_total": totals["rare"],
        "zero_denominator_updates": zero_updates,
        "updates": updates,
        "cell_selection_used_dosage_or_loss": False,
        "cells_retried_or_replaced": False,
        "model_constructed": False, "adam_updates": 0,
        "select_opened": False, "evaluate_opened": False}
    atomic_json(output / "DENOMINATOR_AUDIT.json", result)
    return result


def validate_audit_source(path, audit, campaign, campaign_path, request_paths):
    """Authenticate reuse of the completed audit after packaging-only changes.

    The snapshot proves that the cell calendar, source, masking/reader code and
    accumulated denominators are unchanged.  The current campaign may differ
    in control/orchestration code and therefore in its whole-file SHA-256; the
    completed 365 genotype reads are never relabelled as if run by new code.
    """
    source = read_json(path)
    required = {"schema", "status", "audit", "audit_pre", "campaign",
                "calendar", "accumulation_plan", "requests",
                "genotypes_reread", "select_opened", "evaluate_opened"}
    require(isinstance(source, dict) and set(source) == required
            and source["schema"] == AUDIT_SOURCE_SCHEMA
            and source["status"] == "FROZEN_COMPLETED_AUDIT_SOURCE"
            and source["genotypes_reread"] is False
            and source["select_opened"] is False
            and source["evaluate_opened"] is False,
            "denominator audit source binding differs")
    descriptors = {}
    for name in ("audit", "audit_pre", "campaign", "calendar",
                 "accumulation_plan"):
        descriptors[name] = checked_descriptor(source[name], name)
    require("__path__" in audit
            and descriptors["audit"].resolve() == Path(audit["__path__"]).resolve()
            and source["audit"]["sha256"] == sha(Path(audit["__path__"])),
            "completed denominator audit differs from source binding")
    old_campaign = read_json(descriptors["campaign"])
    old_calendar = read_json(descriptors["calendar"])
    old_plan = read_json(descriptors["accumulation_plan"])
    old_pre = read_json(descriptors["audit_pre"])
    request_descriptors = source["requests"]
    require(isinstance(request_descriptors, dict)
            and set(request_descriptors) == {Path(item).name for item in request_paths},
            "denominator audit source request inventory differs")
    old_requests = {}
    for name, descriptor in request_descriptors.items():
        old_requests[name] = read_json(checked_descriptor(descriptor,
                                                          "source request " + name))
    old_request_hashes = {name: descriptor["sha256"]
                          for name, descriptor in request_descriptors.items()}
    require(sha(descriptors["campaign"]) == audit["campaign_sha256"]
            and sha(descriptors["calendar"]) == audit["calendar_sha256"]
            and sha(descriptors["audit_pre"]) == audit["pre_sha256"]
            and old_request_hashes == audit["request_sha256"],
            "completed audit does not belong to its frozen source snapshot")
    require(old_pre.get("schema") == PRE_SCHEMA
            and old_pre.get("status") == AUDIT_PRE_STATUS
            and old_pre.get("campaign_sha256") == audit["campaign_sha256"]
            and old_pre.get("request_sha256") == old_request_hashes
            and old_pre.get("authorized_role") == "FIT"
            and old_pre.get("authorized_updates_per_fit") == 0
            and old_pre.get("authorized_fit_ids") == []
            and old_pre.get("denominator_audit_authorized") is True
            and old_pre.get("select_opened") is False
            and old_pre.get("evaluate_opened") is False,
            "source PRE did not authorize only the completed denominator audit")
    require(old_campaign["calendar"]["sha256"] == sha(descriptors["calendar"])
            and old_campaign["accumulation_plan"]["sha256"] ==
                sha(descriptors["accumulation_plan"])
            and campaign["calendar"]["sha256"] == sha(descriptors["calendar"])
            and campaign["accumulation_plan"]["sha256"] ==
                sha(descriptors["accumulation_plan"]),
            "current campaign changed the audited calendar or accumulation plan")
    # Fields that determine which genotype rows are read and how observed
    # denominators are formed must be byte-for-byte identical.  Control status
    # and the request runner may change without changing these reads.
    invariant = ("role", "chromosome", "source", "fit_roster",
                 "denominator_audit_model", "reader_resources")
    require(all(old_campaign[name] == campaign[name] for name in invariant),
            "current campaign changed an audited source, role or reader parameter")
    audit_code = ("r02_masked_training_campaign.py", "r02_masked_training_batch.py",
                  "r02_masked_genotype_model.py", "r02_masked_genotypes.py",
                  "r02_masked_store.py")
    require(all(old_campaign["code_sha256"][name] ==
                campaign["code_sha256"][name] for name in audit_code),
            "code that reads or masks denominator targets changed")
    current = {Path(item).name: read_json(item) for item in request_paths}
    for name, old in old_requests.items():
        new = current[name]
        require({key: value for key, value in old.items() if key != "campaign"} ==
                {key: value for key, value in new.items() if key != "campaign"},
                "training request changed beyond its campaign descriptor")
    require(old_calendar == read_json(campaign["calendar"]["path"])
            and old_plan == read_json(campaign["accumulation_plan"]["path"]),
            "current calendar or accumulated denominators differ from snapshot")
    return source


def validate_audit(path, campaign_path, request_paths, pre_path,
                   denominator_audit_pre=None):
    audit = read_json(path)
    audit["__path__"] = str(Path(path).resolve())
    required = {"schema", "status", "campaign_sha256", "calendar_sha256",
                "pre_sha256", "request_sha256", "updates_checked",
                "fragments_checked", "person_cell_visits_checked",
                "common_observed_count_total", "rare_observed_count_total",
                "zero_denominator_updates", "updates",
                "cell_selection_used_dosage_or_loss",
                "cells_retried_or_replaced", "model_constructed",
                "adam_updates", "select_opened", "evaluate_opened"}
    campaign = read_json(campaign_path)
    _, calendar = validate_campaign(campaign)
    exact_current = (audit.get("campaign_sha256") == sha(campaign_path)
                     and audit.get("pre_sha256") == sha(pre_path)
                     and audit.get("request_sha256") == {
                         Path(item).name: sha(item) for item in request_paths})
    source = None
    if not exact_current:
        require(denominator_audit_pre is not None,
                "audit from an earlier packaging identity needs its frozen source binding")
        source = validate_audit_source(denominator_audit_pre, audit, campaign,
                                       campaign_path, request_paths)
    audit.pop("__path__")
    require(isinstance(audit, dict) and set(audit) == required
            and audit.get("schema") == AUDIT_SCHEMA and audit.get("status") == AUDIT_STATUS
            and audit.get("calendar_sha256") == campaign["calendar"]["sha256"]
            and (exact_current or source is not None)
            and audit.get("updates_checked") == 365
            and audit.get("fragments_checked") == 730
            and audit.get("person_cell_visits_checked") == 2920
            and audit.get("common_observed_count_total", 0) > 0
            and audit.get("rare_observed_count_total", 0) > 0
            and audit.get("zero_denominator_updates") ==
                [6, 102, 179, 196, 215, 216, 226, 227, 236, 288, 322, 332]
            and audit.get("cell_selection_used_dosage_or_loss") is False
            and audit.get("cells_retried_or_replaced") is False
            and audit.get("model_constructed") is False
            and audit.get("adam_updates") == 0
            and audit.get("select_opened") is False
            and audit.get("evaluate_opened") is False,
            "completed positive-denominator audit for the exact campaign is required")
    require(isinstance(audit.get("updates"), list) and len(audit["updates"]) == 365,
            "denominator audit update inventory differs")
    common = rare = 0
    for index, row in enumerate(audit["updates"], 1):
        require(isinstance(row, dict)
                and set(row) == {"update", "common_observed_count",
                                 "rare_observed_count", "receipt_sha256"}
                and row["update"] == index
                and type(row["common_observed_count"]) is int
                and row["common_observed_count"] >= 0
                and type(row["rare_observed_count"]) is int
                and row["rare_observed_count"] > 0
                and is_hash(row["receipt_sha256"]),
                "denominator audit has a missing, nonpositive or malformed update")
        receipt = Path(path).parent / "updates" / f"update-{index:08d}.json"
        require(receipt.is_file() and sha(receipt) == row["receipt_sha256"],
                "denominator update receipt is missing or changed")
        fragments = calendar["updates"][index - 1]["fragments"]
        examples = [{"visit": item["visit"], "chromosome": item["chromosome"],
                     "cell": item["cell"], "person_ids": item["person_ids"]}
                    for item in fragments]
        receipt_value = validate_denominator_update(read_json(receipt), index=index,
            campaign_sha=audit["campaign_sha256"],
            calendar_sha=audit["calendar_sha256"], examples=examples)
        require(row["common_observed_count"] ==
                receipt_value["common_observed_count"]
                and row["rare_observed_count"] ==
                receipt_value["rare_observed_count"],
                "denominator audit update differs from its receipt")
        common += row["common_observed_count"]; rare += row["rare_observed_count"]
    require(common == audit["common_observed_count_total"]
            and rare == audit["rare_observed_count_total"],
            "denominator audit totals differ from its updates")
    plan = read_json(campaign["accumulation_plan"]["path"])
    for planned in plan["updates"]:
        rows = [audit["updates"][group - 1] for group in planned["calendar_groups"]]
        require(sum(row["common_observed_count"] for row in rows) ==
                planned["common_observed_count"] > 0
                and sum(row["rare_observed_count"] for row in rows) ==
                planned["rare_observed_count"] > 0,
                "accumulated update denominators differ or are empty")
    return audit


def audit_rare_identity(campaign_path, pre_path, request_paths, output):
    import torch
    from r02_masked_genotype_model import MaskedModelConfig
    campaign = read_json(campaign_path)
    roster, calendar = validate_campaign(campaign)
    requests = [validate_request(read_json(path), campaign, campaign_path)
                for path in request_paths]
    require({item["fit_id"] for item in requests} == set(ALL_FIT_IDS),
            "all eight requests are required for the rare-identity audit")
    validate_pre(read_json(pre_path), campaign_path, request_paths)
    reader, fit_ids = open_reader(campaign)
    require(set(roster["person_ids"]) == fit_ids,
            "frozen roster is not the complete authenticated FIT role")
    torch.set_num_threads(campaign["reader_resources"]["torch_threads"])
    families = {}
    for family in FAMILIES:
        control = next(item for item in requests
                       if item["fit_id"] == family + "__full__rare_identity")
        observed = []
        loader = fit_cell_loader({22: reader}, MaskedModelConfig(**control["model"]),
            context_exclusion_cm=0, device="cpu",
            fragment_transform=rare_identity_transform(family, observed=observed))
        for examples in schedule_for(campaign, roster, calendar).updates():
            for example in examples:
                loader(example)
        visible = sum(row["visible_loci"] for row in observed)
        moved = sum(row["moved_loci"] for row in observed)
        changed = sum(row["changed_doses"] for row in observed)
        require(len(observed) == 730 and visible > 0 and moved > 0 and changed > 0,
                "rare-identity audit did not alter a nonzero visible dose set")
        families[family] = {"effective_reach_cm": control["model"]["rare_radius_cm"] *
                (1 if family == "local_deepsets" else
                 control["model"]["attention_depth"] + 1),
            "fragments": 730, "visible_loci": visible, "moved_loci": moved,
            "changed_doses": changed, "moved_locus_fraction": moved / visible,
            "changed_dose_fraction": changed / visible, "plans": observed}
    value = {"schema": RARE_AUDIT_SCHEMA, "status": RARE_AUDIT_STATUS,
        "campaign_sha256": sha(campaign_path),
        "calendar_sha256": campaign["calendar"]["sha256"],
        "pre_sha256": sha(pre_path),
        "request_sha256": {Path(path).name: sha(path) for path in request_paths},
        "seed": 1701, "role": "FIT", "families": families,
        "dosage_used_to_construct_permutation": False,
        "cells_retried_or_replaced": False, "model_constructed": False,
        "adam_updates": 0, "select_opened": False, "evaluate_opened": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "RARE_IDENTITY_AUDIT.json", value)
    return value


def validate_rare_audit(path, campaign_path, request_paths, pre_path, family):
    value = read_json(path)
    require(value.get("schema") == RARE_AUDIT_SCHEMA
            and value.get("status") == RARE_AUDIT_STATUS
            and value.get("campaign_sha256") == sha(campaign_path)
            and value.get("pre_sha256") == sha(pre_path)
            and value.get("request_sha256") == {
                Path(item).name: sha(item) for item in request_paths}
            and value.get("seed") == 1701 and value.get("role") == "FIT"
            and value.get("dosage_used_to_construct_permutation") is False
            and value.get("cells_retried_or_replaced") is False
            and value.get("model_constructed") is False
            and value.get("adam_updates") == 0
            and value.get("select_opened") is False
            and value.get("evaluate_opened") is False,
            "reviewed rare-identity audit for the exact campaign is required")
    family_value = value.get("families", {}).get(family)
    expected_reach = 0.2 if family == "local_deepsets" else 0.6
    require(isinstance(family_value, dict)
            and math.isclose(family_value.get("effective_reach_cm", -1),
                             expected_reach, rel_tol=0, abs_tol=1e-12)
            and family_value.get("fragments") == 730
            and family_value.get("visible_loci", 0) > 0
            and family_value.get("moved_loci", 0) > 0
            and family_value.get("changed_doses", 0) > 0
            and isinstance(family_value.get("plans"), list)
            and len(family_value["plans"]) == 730,
            "rare-identity family audit is incomplete")
    mapping = {}
    for row in family_value["plans"]:
        require(set(row) == {"example", "example_sha256", "territory_id",
                            "plan_identity", "visible_loci", "moved_loci",
                            "changed_doses"}
                and row["example_sha256"] == digest(row["example"])
                and is_hash(row["territory_id"]) and is_hash(row["plan_identity"]),
                "rare-identity plan inventory differs")
        require(row["example_sha256"] not in mapping,
                "rare-identity audit repeats a scheduled example")
        mapping[row["example_sha256"]] = row
    return mapping


def run_fit(campaign_path, request_path, all_request_paths, pre_path, audit_path,
            denominator_audit_pre, rare_audit_path, output):
    import torch
    from r02_masked_genotype_model import (MaskedGenotypeModel, MaskedModelConfig,
                                           build_masked_adam)
    campaign = read_json(campaign_path)
    roster, calendar = validate_campaign(campaign)
    request = validate_request(read_json(request_path), campaign, campaign_path)
    all_requests = [validate_request(read_json(path), campaign, campaign_path)
                    for path in all_request_paths]
    require({item["fit_id"] for item in all_requests} == set(ALL_FIT_IDS),
            "all eight requests required to authenticate paired campaign")
    pre = read_json(pre_path)
    validate_pre(pre, campaign_path, all_request_paths)
    audit = validate_audit(audit_path, campaign_path, all_request_paths, pre_path,
                           denominator_audit_pre)
    reader, fit_ids = open_reader(campaign)
    require(set(roster["person_ids"]) == fit_ids,
            "frozen roster is not the complete authenticated FIT role")
    torch.set_num_threads(campaign["reader_resources"]["torch_threads"])
    seed_all(request["seed"])
    config = MaskedModelConfig(**request["model"])
    model = MaskedGenotypeModel(config)
    optimizer = build_masked_adam(model, **request["optimizer"])
    schedule = schedule_for(campaign, roster, calendar)
    transform = None
    if request["training_input_control"] == "RARE_IDENTITY_SHA256_V1":
        require(rare_audit_path, "rare-identity fit requires its completed audit")
        expected = validate_rare_audit(rare_audit_path, campaign_path,
                                       all_request_paths, pre_path,
                                       request["family"])
        transform = rare_identity_transform(request["family"], expected=expected)
    loader = fit_cell_loader({22: reader}, config, context_exclusion_cm=0,
                             device="cpu", fragment_transform=transform)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = run_masked_campaign(model, optimizer, schedule, loader,
        output / "checkpoint",
        source_identity_sha256=digest({"campaign_sha256": sha(campaign_path),
                                       "audit_sha256": sha(audit_path),
                                       "denominator_audit_source_sha256":
                                           sha(denominator_audit_pre),
                                       "rare_identity_audit_sha256":
                                           (sha(rare_audit_path) if transform else None),
                                       "pre_sha256": sha(pre_path),
                                       "paired_request_sha256": audit["request_sha256"]}),
        protocol_sha256=campaign["protocol"]["sha256"],
        information_condition=request["condition"], max_updates=353,
        max_seconds=request["resources"]["max_seconds"])
    receipt = {"schema": RECEIPT_SCHEMA, "status": result["status"],
        "fit_id": request["fit_id"], "campaign_sha256": sha(campaign_path),
        "request_sha256": sha(request_path), "pre_sha256": sha(pre_path),
        "denominator_audit_sha256": sha(audit_path), "campaign_result": result,
        "denominator_audit_source_sha256": sha(denominator_audit_pre),
        "rare_identity_audit_sha256": (sha(rare_audit_path) if transform else None),
        "elapsed_this_invocation_seconds": time.monotonic() - started,
        "checkpoint_root": str((output / "checkpoint").resolve()),
        "checkpoint_reusable_only_for_exact_bound_identity": True,
        "select_opened": False, "evaluate_opened": False,
        "biological_result_claimed": False}
    atomic_json(output / "TRAINING_RECEIPT.json", receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-denominators")
    rare_audit = commands.add_parser("audit-rare-identity")
    run = commands.add_parser("run")
    for command in (audit, rare_audit, run):
        command.add_argument("--campaign", required=True)
        command.add_argument("--pre-receipt", required=True)
        command.add_argument("--all-request", action="append", required=True)
        command.add_argument("--output", required=True)
    run.add_argument("--request", required=True)
    run.add_argument("--denominator-audit", required=True)
    run.add_argument("--denominator-audit-pre", required=True,
                     help="frozen source binding for the completed denominator audit")
    run.add_argument("--rare-identity-audit")
    check = commands.add_parser("check")
    check.add_argument("--campaign", required=True)
    check.add_argument("--all-request", action="append", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "check":
        campaign = read_json(args.campaign)
        validate_campaign(campaign)
        requests = [validate_request(read_json(path), campaign, args.campaign)
                    for path in args.all_request]
        require({item["fit_id"] for item in requests} == set(ALL_FIT_IDS),
            "check requires the eight exact requests")
        value = {"status": "EIGHT_REQUESTS_AND_CALENDAR_VALID",
                 "fit_ids": sorted(item["fit_id"] for item in requests)}
    elif args.command == "audit-denominators":
        value = audit_denominators(args.campaign, args.pre_receipt,
                                   args.all_request, args.output)
    elif args.command == "audit-rare-identity":
        value = audit_rare_identity(args.campaign, args.pre_receipt,
                                    args.all_request, args.output)
    else:
        value = run_fit(args.campaign, args.request, args.all_request,
                        args.pre_receipt, args.denominator_audit,
                        args.denominator_audit_pre,
                        args.rare_identity_audit, args.output)
    print(json.dumps(value, sort_keys=True))
    if args.command == "audit-denominators" and value["status"] != AUDIT_STATUS:
        return 3
    if args.command == "run" and value["status"] != "DECLARED_UPDATE_BUDGET_COMPLETE":
        # The checkpoint and receipt remain recoverable, but Nextflow must not
        # cache a time-limited partial fit as a completed process output.
        return 4
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        raise SystemExit(2)
