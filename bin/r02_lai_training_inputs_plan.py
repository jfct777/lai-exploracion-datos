#!/usr/bin/env python3
"""Plan and bind LAI FIT/SELECT inputs without reading simulated ancestry.

``plan`` closes the deterministic L1/L2 rules and enumerates the twelve cases
of one reciprocal development direction.  It does not create genotype matrices
or label a case ready when an exact marker catalogue or receipt is absent.
``adapter-request`` and ``input-request`` create the two small authenticated
requests after their upstream files exist.  The latter requires a separate,
admitted control decision because this program does not implement controls.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import zipfile


DIRECTION = "FOLD_A_TO_FOLD_B"
PRIMARY_FLARE = {
    "initial_gen": "10",
    "policy": "ALL",
    "marker_condition": "BASELINE_EXCLUDING_EVALUATED_DNABR_RARE",
}
SENSITIVITY_GENS = ("5", "16")
REFERENCE_HASHES = ("event_catalogue_sha256", "ref_train_receipt_sha256",
                    "ref_support_by_provenance_sha256", "reference_vcf_sha256",
                    "reference_roles_sha256")
PREPARE_CONTROLS = (
    ("NONE", "ALL_ADMITTED_CONTEXT"),
    ("NO_RARE", "COMMON_AND_FLARE_ONLY"),
    ("COUNT_MATCHED", "RARE_COUNT_FIXED_IDENTITY_REPLACED"),
    ("DENSITY_MATCHED", "RARE_DENSITY_FIXED_IDENTITY_REPLACED"),
    ("IDENTITY_SHAM", "RARE_LOCUS_AND_COUNT_FIXED_PERSON_RELATION_BROKEN"),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_file(), f"file absent: {path}")
    return {"path": str(path), "sha256": sha(path)}


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists() and path.parent.is_dir(), "fresh output file required: " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def recorded_file(path, expected_sha):
    path = Path(path).resolve()
    require(path.is_file(), f"file absent: {path}")
    require(sha(path) == expected_sha, f"recorded SHA-256 differs: {path}")
    return {"path": str(path), "sha256": expected_sha, "bytes": path.stat().st_size}


def recorded_descriptor(item, local_path=None):
    result = {
        "uri": item["probability_uri"],
        "generation": item["probability_generation"],
        "sha256": item["probability_sha256"],
        "bytes": int(item["probability_bytes"]),
        "receipt_uri": item["receipt_uri"],
        "receipt_generation": item["receipt_generation"],
        "receipt_sha256": item["receipt_sha256"],
    }
    if local_path is not None:
        result["local_path"] = str(Path(local_path).resolve())
    return result


def choose_flare(alternatives, initial_gen):
    wanted = {**PRIMARY_FLARE, "initial_gen": str(initial_gen)}
    matches = [row for row in alternatives
               if all(str(row.get(key)) == value for key, value in wanted.items())]
    require(len(matches) == 1, f"expected exactly one FLARE alternative: {wanted}")
    return matches[0]


def derive_case_files(task):
    available = task["available"]
    dosage = Path(available["simulation_predictors"]["alt_dosage.npy"]["path"])
    material = dosage.parent / "manifest.json"
    case_id, direction = task["case_id"], task["direction"]
    integration = Path(available["query_overlay"]["path"]).parents[2]
    truthblind = integration / "requests" / f"{case_id}.{direction}.request.json"
    genotype_root = dosage.parents[3]
    genotype = genotype_root / "requests" / f"{case_id}.json"
    return {"material_manifest": str(material.resolve()),
            "genotype_request": str(genotype.resolve()),
            "truthblind_request": str(truthblind.resolve())}


def build_plan(args):
    inventory, fit_plan, reader = load(args.inventory), load(args.fit_plan), load(args.reader_plan)
    registry = load(args.flare_registry)
    flare_decision = load(args.flare_decision)
    catalogue_receipt = load(args.catalogue_receipt) if args.catalogue_receipt else None
    flare_receipt_staging = load(args.flare_receipt_staging) if args.flare_receipt_staging else None
    require(inventory.get("schema") == "r02_lai_channel_input_inventory_v1", "inventory schema differs")
    require(fit_plan.get("schema") == "r02_lai_fit_statistics_plan_v1", "FIT statistics plan schema differs")
    require(reader.get("schema") == "r02_lai_regional_reader_plan_v1", "regional reader plan schema differs")
    require(registry.get("status") == "COMPLETE_108_FITTED_MODELS_RANGE_VERIFIED"
            and len(registry.get("records", [])) == 108, "FLARE fitted-model registry differs")
    require(flare_decision.get("schema") == "r02_lai_flare_initialization_decision_cx_v1"
            and flare_decision.get("status") == "PROSPECTIVE_DESIGN_DECISION_REQUIRES_EXACT_EXISTING_INPUT_BINDING",
            "FLARE initialization decision differs")
    catalogues = {}
    if catalogue_receipt is not None:
        require(catalogue_receipt.get("schema") == "r02_lai_common_event_catalogue_receipt_v1"
                and catalogue_receipt.get("status") == "COMPLETE_AUTHENTICATED_COMMON_EVENT_CORE_CATALOGUES"
                and catalogue_receipt.get("direction") == DIRECTION
                and catalogue_receipt.get("tasks_complete") == 12
                and catalogue_receipt.get("truth_read") is False
                and catalogue_receipt.get("selection_performed") is False
                and catalogue_receipt.get("evaluate_opened") is False,
                "COMMON/EVENT/core catalogue receipt differs")
        catalogues = {row["task_id"]: row for row in catalogue_receipt["tasks"]}
        require(len(catalogues) == 12, "COMMON/EVENT/core receipt lacks tasks")
    staged_flare_receipts = {}
    if flare_receipt_staging is not None:
        require(flare_receipt_staging.get("schema") == "r02_lai_flare_projection_receipt_staging_v1"
                and flare_receipt_staging.get("status") == "COMPLETE_12_PRIMARY_FLARE_RECEIPTS_AUTHENTICATED"
                and flare_receipt_staging.get("direction") == DIRECTION
                and flare_receipt_staging.get("tasks_complete") == 12
                and flare_receipt_staging.get("truth_read") is False
                and flare_receipt_staging.get("selection_performed") is False,
                "staged primary FLARE receipt inventory differs")
        staged_flare_receipts = {row["task_id"]: row for row in flare_receipt_staging["tasks"]}
    fit_units = {row["unit_id"]: row for row in fit_plan["logical_units"]
                 if row["direction"] == DIRECTION}
    reader_tasks = {row["task_id"]: row for row in reader["tasks"]
                    if row["direction"] == DIRECTION
                    and {key: str(value) for key, value in row["flare_identity"].items()} == PRIMARY_FLARE}
    tasks = []
    for item in inventory["tasks"]:
        if item["direction"] != DIRECTION:
            continue
        task_id = item["task_id"]
        require(task_id in fit_units and task_id in reader_tasks, "case is absent from FIT or reader plan")
        require(item["role"] == fit_units[task_id]["target_role"], "FIT/SELECT role differs")
        primary = choose_flare(item["available"]["flare_projection_alternatives"], "10")
        model_records = [row for row in registry["records"]
                         if row["case_id"] == item["case_id"] and row["initial_gen"] == 10
                         and row["policy"] == "ALL"
                         and row["marker_condition"] == "BASELINE_EXCLUDING_EVALUATED_DNABR_RARE"]
        require(len(model_records) == 1, "primary FLARE model archive record differs")
        model_record = model_records[0]; member = model_record["members"]["manifest"]
        payload_start = int(member["tar_header_offset"]) + 512
        model_manifest_range = {
            "archive_uri": model_record["archive"]["uri"],
            "archive_generation": model_record["archive"]["generation"],
            "archive_sha256": model_record["archive"]["sha256"],
            "archive_bytes": int(model_record["archive"]["bytes"]),
            "member_path": member["path"], "member_sha256": member["sha256"],
            "member_bytes": int(member["bytes"]), "tar_header_offset": int(member["tar_header_offset"]),
            "gcs_payload_byte_range_inclusive": [payload_start, payload_start + int(member["bytes"]) - 1],
            "archive_inventory": model_record["archive_inventory"],
        }
        reader_flare = reader_tasks[task_id]["inputs"]["flare_probabilities"]
        reader_identity = {key: str(value) for key, value in reader_tasks[task_id]["flare_identity"].items()}
        require(reader_identity == PRIMARY_FLARE, "reader primary FLARE identity differs")
        require(reader_flare["sha256"] == primary["probability_sha256"], "reader primary FLARE hash differs")
        local_root = Path(reader_flare["path"]).parents[1]
        sensitivities = []
        for initial_gen in SENSITIVITY_GENS:
            alt = choose_flare(item["available"]["flare_projection_alternatives"], initial_gen)
            local = local_root / f"gen{initial_gen}.ALL.BASELINE_EXCLUDING_EVALUATED_DNABR_RARE" / "probabilities.npy"
            sensitivities.append({"initial_gen": initial_gen,
                                  "use": "FIXED_GEN10_NETWORK_WEIGHTS_NO_RETRAIN_NO_RESELECTION",
                                  **recorded_descriptor(alt, local)})
        event_table = item["available"]["event_candidate_table"]
        files = derive_case_files(item)
        missing = []
        if not all(Path(path).is_file() for path in files.values()):
            missing.append("EXISTING_MATERIAL_OR_REQUEST_PATH_ABSENT")
        # These are deliberate product requests, not inferred absences of data.
        if catalogue_receipt is None:
            missing += [
                "EXACT_ORDERED_FLARE_INPUT_MARKER_CATALOGUE_WITH_REF_ALT_AND_HASH_NOT_MATERIALIZED",
                "DNABR_SOURCE_MINOR_AC_AN_AND_PER_ANCESTRY_REFERENCE_AVAILABILITY_NOT_JOINED",
            ]
        missing += [
            "REGIONAL_BATCH_WITH_QUERY_OWNERSHIP_AND_SELECTED_GEN10_FLARE_NOT_MATERIALIZED",
            "EXPLICIT_CHANNEL_BATCH_AND_RECEIPT_NOT_MATERIALIZED",
            "L3_CONTROL_MATERIALS_NOT_PART_OF_THIS_PLAN",
        ]
        if flare_receipt_staging is None:
            missing.append("LOCAL_AUTHENTICATED_FLARE_PROJECTION_RECEIPT_NOT_STAGED")
        task_inputs = {
                "event_candidate_table": {"path": event_table["path"], "sha256": event_table["observed_sha256"]},
                "query_overlay": {"path": item["available"]["query_overlay"]["path"],
                                  "sha256": item["available"]["query_overlay"]["recorded_sha256"]},
                "query_overlay_receipt": {"path": item["available"]["query_overlay_receipt"]["path"],
                                          "sha256": item["available"]["query_overlay_receipt"]["observed_sha256"]},
                "material_manifest": files["material_manifest"],
                "genotype_request": files["genotype_request"],
                "truthblind_request": files["truthblind_request"],
                "reference_manifest": reader_tasks[task_id]["inputs"]["reference_manifest"],
                "reference_index": reader_tasks[task_id]["inputs"]["reference_index"],
                "genetic_map": item["available"]["genetic_map"],
            }
        if catalogue_receipt is not None:
            bound = catalogues[task_id]
            require(bound["case_id"] == item["case_id"] and bound["role"] == item["role"]
                    and bound["chromosome"] == str(item["chromosome"]), "catalogue task identity differs")
            task_inputs["common_event_core_catalogues"] = {
                row["kind"]: {key: row[key] for key in ("path", "sha256", "bytes", "rows")}
                for row in bound["catalogues"]}
            task_inputs["query_core"] = bound["query_core"]
        if flare_receipt_staging is not None:
            staged = staged_flare_receipts[task_id]
            require(staged["sha256"] == primary["receipt_sha256"], "staged FLARE receipt binding differs")
            task_inputs["flare_projection_receipt"] = {
                key: staged[key] for key in ("path", "sha256", "bytes")}
        tasks.append({
            "task_id": task_id,
            "case_id": item["case_id"],
            "chromosome": str(item["chromosome"]),
            "direction": DIRECTION,
            "role": item["role"],
            "people": item["people"],
            "fit_statistics_job": fit_units[task_id]["simulated_fit_statistics_job"],
            "inputs": task_inputs,
            "flare": {
                "primary": {"initial_gen": "10", "use": "COMMON_INPUT_FOR_FIT_AND_SELECT",
                            **recorded_descriptor(primary, reader_flare["path"]),
                            "model_manifest_range": model_manifest_range},
                "sensitivity": sensitivities,
                "truth_used_for_choice": False,
                "evaluate_used_for_choice": False,
            },
            "product_requests": missing,
            "status": "CATALOGUES_COMPLETE_REGIONAL_BATCH_PENDING" if catalogue_receipt else "RULES_FIXED_PRODUCTS_NOT_MATERIALIZED",
        })
    require(len(tasks) == 12 and sum(t["role"] == "FIT" for t in tasks) == 6
            and sum(t["role"] == "SELECT" for t in tasks) == 6,
            "one A-to-B direction must contain six FIT and six SELECT cases")
    result = {
        "schema": "r02_lai_training_inputs_plan_v1",
        "status": "CATALOGUES_COMPLETE_REGIONAL_BATCH_PENDING" if catalogue_receipt else "RULES_FIXED_PRODUCTS_NOT_MATERIALIZED",
        "direction": DIRECTION,
        "evaluate_opened": False,
        "new_nam_reserve_claimed": False,
        "training_authorized": False,
        "selection_authorized": False,
        "scientific_rules": {
            "common": {
                "catalogue": "EXACT_ORDERED_MARKER_IDENTITIES_CONSUMED_BY_EACH_PRIMARY_FLARE_RUN",
                "primary_flare_identity": PRIMARY_FLARE,
                "counted_allele": "ALT",
                "fit_frequency": "ALT_AC_DIVIDED_BY_CALLED_DIPLOID_ALT_AN_IN_THE_SIX_A_FIT_CASES",
                "extra_frequency_filter": None,
                "reason": "The network and FLARE receive the same marker information; marker availability does not redefine the DNABR rare catalogue.",
            },
            "event": {
                "catalogue": "FIXED_DNABR_SOURCE_MINOR_CATALOGUE_INTERSECT_EXACT_SIMULATION_LOCUS_AND_MAP_SUPPORT",
                "inclusion": "ALL_EVENT_CANDIDATE_ROWS; COPYABLE_IN_SOURCE_FOLD_IS_RECORDED_NOT_AN_INCLUSION_FILTER",
                "counted_allele": "SOURCE_MINOR_REF_OR_ALT",
                "target_frequency_filter": None,
                "truth_filter": None,
                "required_metadata": ["source_minor_ac", "source_an", "per_ancestry_reference_ac_an_or_explicit_absence"],
            },
            "core": {
                "current_case_unit": "ONE_FULL_CHROMOSOME_QUERY_AXIS",
                "rule": "CORE_IFF_VARIANT_BP_IS_WITHIN_FIRST_AND_LAST_AUTHENTICATED_QUERY_GRID_BP_ON_MAP_SUPPORT_INCLUSIVE",
                "outside": "MAP_SUPPORTED_VARIANTS_OUTSIDE_CORE_MAY_ONLY_BE_CONTEXT",
                "overlap": "EACH_QUERY_IS_OWNED_AND_SCORED_ONCE",
                "radius_selected_here": False,
            },
            "flare": flare_decision,
        },
        "losses_recorded_for_later_training": {
            "fit": "MEAN_NEGATIVE_LOG_PROBABILITY_OF_TRUE_UNORDERED_DIPLOID_STATE",
            "fit_weighting": "EQUAL_PERSON_WEIGHT_AND_GENETIC_LENGTH_WITHIN_PERSON",
            "select": "ONE_MINUS_DIPLOID_BOUNDARY_F1_AT_0.2_CM",
            "select_weighting": "EQUAL_PERSON_WEIGHT_AND_GENETIC_LENGTH_WITHIN_PERSON",
            "descriptive_boundary_tolerances_cm": [0.1, 0.5],
            "implemented_by_this_program": False,
        },
        "tasks": sorted(tasks, key=lambda row: row["task_id"]),
        "sources": {
            "inventory": descriptor(args.inventory),
            "fit_plan": descriptor(args.fit_plan),
            "reader_plan": descriptor(args.reader_plan),
            "flare_decision": descriptor(args.flare_decision),
            "flare_registry": descriptor(args.flare_registry),
            **({"common_event_core_catalogue_receipt": descriptor(args.catalogue_receipt)}
               if args.catalogue_receipt else {}),
            **({"flare_projection_receipt_staging": descriptor(args.flare_receipt_staging)}
               if args.flare_receipt_staging else {}),
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.case_manifest_output:
        case_manifest = {
            "schema": "r02_lai_training_case_manifest_v1",
            "status": result["status"], "direction": DIRECTION,
            "evaluate_opened": False, "cases": [{key: row[key] for key in
                ("task_id", "case_id", "chromosome", "direction", "role", "people", "fit_statistics_job")}
                | {"primary_flare": row["flare"]["primary"]} for row in result["tasks"]],
            "plan_sha256": sha(args.output),
        }
        Path(args.case_manifest_output).write_text(json.dumps(case_manifest, indent=2, sort_keys=True) + "\n")
    if args.product_request_output:
        request_manifest = {
            "schema": "r02_lai_training_product_request_manifest_v1",
            "status": "REQUESTS_RECORDED_NOT_MATERIALIZED",
            "direction": DIRECTION, "training_authorized": False, "evaluate_opened": False,
            "requests": [{"task_id": row["task_id"], "case_id": row["case_id"],
                          "role": row["role"], "requested_products": row["product_requests"]}
                         for row in result["tasks"]],
            "plan_sha256": sha(args.output),
        }
        Path(args.product_request_output).write_text(json.dumps(request_manifest, indent=2, sort_keys=True) + "\n")
    return result


def case_from_plan(path, task_id):
    plan = load(path)
    require(plan.get("schema") == "r02_lai_training_inputs_plan_v1", "case plan schema differs")
    rows = [row for row in plan["tasks"] if row["task_id"] == task_id]
    require(len(rows) == 1, "task is absent or duplicated in plan")
    return plan, rows[0]


def reference_case(path, task_id, role):
    receipt = load(path)
    require(receipt.get("schema") == "r02_lai_reference_linkage_receipt_v1"
            and receipt.get("status") == "COUNTS_AND_ORIGINAL_UNITS_MATCH"
            and receipt.get("training_authorized") is False
            and receipt.get("controls_materialized") is False
            and receipt.get("truth_loaded") is False
            and receipt.get("evaluate_opened") is False,
            "DM reference receipt is not PREPARE-only")
    rows = [row for row in receipt.get("cases", []) if row.get("task_id") == task_id]
    require(len(rows) == 1, "DM reference case absent or duplicated")
    row = rows[0]
    require(row.get("role") == role and row.get("problems") == []
            and row.get("reference_join_key") == ["chromosome", "position_bp", "REF", "ALT"]
            and row.get("reference_duplicate_keys") == 0
            and row.get("reference_target_person_overlap") == 0,
            "DM reference case checks differ")
    identity = {name: row.get(name) for name in REFERENCE_HASHES}
    require(all(isinstance(value, str) and len(value) == 64 for value in identity.values()),
            "DM reference identity is incomplete")
    return receipt, row, identity


def adapter_request(args):
    plan, case = case_from_plan(args.plan, args.task_id)
    probability = Path(case["flare"]["primary"]["local_path"])
    require(probability.is_file() and sha(probability) == case["flare"]["primary"]["sha256"],
            "recovered primary FLARE probability file differs")
    material = Path(case["inputs"]["material_manifest"])
    reference_receipt, reference_row, reference_identity = reference_case(
        args.reference_receipt, args.task_id, case["role"])
    explicit_receipt = load(args.explicit_receipt)
    require(explicit_receipt.get("schema") == "r02_lai_predictor_channels_receipt_v2"
            and explicit_receipt.get("status") == "PREPARED_EXACT_REVIEWED_AXES_NO_TRAINING"
            and explicit_receipt.get("task_id") == args.task_id
            and explicit_receipt.get("case_id") == case["case_id"]
            and explicit_receipt.get("direction") == plan["direction"]
            and explicit_receipt.get("role") == case["role"]
            and explicit_receipt.get("reference_identity") == reference_identity
            and explicit_receipt.get("reference_linkage_receipt_sha256") == sha(args.reference_receipt)
            and explicit_receipt.get("training_allowed") is False
            and explicit_receipt.get("controls_materialized") is False,
            "explicit channel receipt is not the exact PREPARE product for this case")
    require(explicit_receipt.get("n_event") == reference_row.get("event_loci"),
            "explicit channel EVENT axis differs from reviewed catalogue")
    value = {
        "schema": "r02_lai_neural_channel_adapter_request_v1",
        "operation": "BIND_EXISTING_CHANNELS_NO_SELECTION",
        "case_id": case["case_id"], "direction": plan["direction"], "expected_role": case["role"],
        "source_genotype_material_sha256": sha(material),
        "reference_identity": reference_identity,
        "inputs": {
            "source_batch": descriptor(args.explicit_batch),
            "source_receipt": descriptor(args.explicit_receipt),
            "overlay": case["inputs"]["query_overlay"],
            "overlay_receipt": case["inputs"]["query_overlay_receipt"],
            "flare_probabilities": {"path": str(probability.resolve()), "sha256": sha(probability)},
            "flare_receipt": descriptor(args.flare_receipt),
        },
    }
    return write_new_json(args.output, value)


def input_request(args):
    plan, case = case_from_plan(args.plan, args.task_id)
    controls = load(args.controls)
    require(controls.get("schema") == "r02_lai_materialized_controls_v1"
            and controls.get("status") == "ADMITTED_AND_MATERIALIZED"
            and controls.get("case_id") == case["case_id"],
            "separate controls decision is absent, differs, or is not materialized")
    paths = case["inputs"]
    value = {
        "schema": "r02_lai_neural_input_request_v1", "operation": "ASSEMBLE_ONLY_NO_TRAINING",
        "protocol_id": plan["direction"], "direction": plan["direction"],
        "case_id": case["case_id"], "expected_role": case["role"],
        "controls": controls["controls"],
        "inputs": {
            "material_manifest": descriptor(paths["material_manifest"]),
            "genotype_request": descriptor(paths["genotype_request"]),
            "truthblind_request": descriptor(paths["truthblind_request"]),
            "overlay": paths["query_overlay"], "overlay_receipt": paths["query_overlay_receipt"],
            "channel_batch": descriptor(args.channel_batch),
            "channel_receipt": descriptor(args.channel_receipt),
        },
        "resources": {"cpus": 1, "memory_bytes": 3 * 1024**3, "timeout_seconds": 2700,
                      "max_input_bytes": 3 * 1024**3, "max_output_bytes": 2 * 1024**3},
    }
    return write_new_json(args.output, value)


def prepare_input_request(args):
    """Write an assembly-only request; definitions are registered but not materialized."""
    plan, case = case_from_plan(args.plan, args.task_id)
    channel = load(args.channel_receipt)
    require(channel.get("schema") == "r02_lai_neural_channel_receipt_v1"
            and channel.get("status") == "CHANNELS_AND_FLARE_BOUND_NO_TRAINING_NO_SELECTION"
            and channel.get("case_id") == case["case_id"]
            and channel.get("direction") == plan["direction"]
            and channel.get("expected_role") == case["role"]
            and channel.get("batch_sha256") == sha(args.channel_batch)
            and channel.get("controls_materialized") is False
            and channel.get("truth_arrays_loaded") is False
            and channel.get("selection_performed") is False,
            "channel batch is not an exact PREPARE-only product")
    paths = case["inputs"]
    controls = [{"name": name, "operation": operation, "truth_used": False,
                 "preserves_people": True, "materialized": False}
                for name, operation in PREPARE_CONTROLS]
    value = {
        "schema": "r02_lai_neural_input_request_v1", "operation": "ASSEMBLE_ONLY_NO_TRAINING",
        "protocol_id": plan["direction"], "direction": plan["direction"],
        "case_id": case["case_id"], "expected_role": case["role"],
        "controls": controls, "controls_materialized": False,
        "training_authorized": False, "model_selection_authorized": False, "evaluate_opened": False,
        "inputs": {
            "material_manifest": descriptor(paths["material_manifest"]),
            "genotype_request": descriptor(paths["genotype_request"]),
            "truthblind_request": descriptor(paths["truthblind_request"]),
            "overlay": paths["query_overlay"], "overlay_receipt": paths["query_overlay_receipt"],
            "channel_batch": descriptor(args.channel_batch),
            "channel_receipt": descriptor(args.channel_receipt),
        },
        "resources": {"cpus": 1, "memory_bytes": 3 * 1024**3, "timeout_seconds": 2700,
                      "max_input_bytes": 3 * 1024**3, "max_output_bytes": 2 * 1024**3},
    }
    return write_new_json(args.output, value)


def gcs_cat(uri, generation, byte_range, account, project):
    command = ["gcloud", "storage", "cat"]
    if byte_range is not None:
        command.append("--range=" + byte_range)
    command.append(uri + "#" + str(generation))
    env = dict(os.environ, CLOUDSDK_CORE_ACCOUNT=account, CLOUDSDK_CORE_PROJECT=project)
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return completed.stdout


def tar_offsets(inventory, archive_bytes):
    # The retained archives were made with tarfile.PAX_FORMAT.  The original
    # producer wrote a 1024-byte PAX record before each first unique payload.
    offset = 0; seen = set(); result = {}
    for row in inventory["entries"]:
        member = row.get("archive_member")
        if row.get("disposition") != "PRESERVE" or member in seen:
            continue
        seen.add(member); offset += 1024; result[member] = (offset, row)
        offset += 512 + math.ceil(int(row["bytes"]) / 512) * 512
    computed = math.ceil((offset + 1024) / 10240) * 10240
    require(computed == int(archive_bytes), "computed TAR length differs from authenticated archive")
    return result


def extract_common(args):
    plan = load(args.plan)
    require(plan.get("schema") == "r02_lai_training_inputs_plan_v1"
            and plan.get("direction") == DIRECTION and len(plan.get("tasks", [])) == 12,
            "one-direction LAI input plan differs")
    output = Path(args.output_dir)
    require(not output.exists() and output.parent.is_dir(), "fresh output directory required")
    output.mkdir(mode=0o700); objects = output / "objects"; logical = output / "logical"
    objects.mkdir(); logical.mkdir()
    receipts = []; downloaded = 0
    marker_member = "models/ALL/material/material/BASELINE_EXCLUDING_EVALUATED_DNABR_RARE/marker_inventory.tsv"
    for task in plan["tasks"]:
        source = task["flare"]["primary"]["model_manifest_range"]
        inv_desc = source["archive_inventory"]
        raw_inventory = gcs_cat(inv_desc["uri"], inv_desc["generation"], None,
                                args.gcloud_account, args.gcloud_project)
        downloaded += len(raw_inventory)
        require(len(raw_inventory) == int(inv_desc["bytes"]) and hashlib.sha256(raw_inventory).hexdigest() == inv_desc["sha256"],
                "FLARE archive inventory size or SHA-256 differs")
        inventory = json.loads(raw_inventory)
        indexed = tar_offsets(inventory, source["archive_bytes"])
        # Older archive-inventory descriptors do not repeat archive bytes.
        if marker_member not in indexed:
            raise ValueError("primary FLARE marker inventory member is absent")
        offset, entry = indexed[marker_member]
        size = int(entry["bytes"]); expected = 512 + size
        object_path = objects / (entry["sha256"] + ".tsv")
        if object_path.exists():
            body = object_path.read_bytes()
            require(len(body) == size and hashlib.sha256(body).hexdigest() == entry["sha256"],
                    "deduplicated local marker inventory differs")
        else:
            raw = gcs_cat(source["archive_uri"], source["archive_generation"],
                          f"{offset}-{offset + expected - 1}", args.gcloud_account, args.gcloud_project)
            downloaded += len(raw)
            require(len(raw) == expected, "short marker inventory range read")
            header = raw[:512]; body = raw[512:]
            name = header[:100].split(b"\0")[0].decode(); header_size = int(header[124:136].strip(b"\0 ") or b"0", 8)
            require(name == marker_member and header_size == size
                    and hashlib.sha256(body).hexdigest() == entry["sha256"],
                    "marker inventory TAR header, size, or SHA-256 differs")
        manifest_range = source["gcs_payload_byte_range_inclusive"]
        model_manifest = gcs_cat(source["archive_uri"], source["archive_generation"],
                                 f"{manifest_range[0]}-{manifest_range[1]}",
                                 args.gcloud_account, args.gcloud_project)
        downloaded += len(model_manifest)
        require(len(model_manifest) == source["member_bytes"]
                and hashlib.sha256(model_manifest).hexdigest() == source["member_sha256"],
                "FLARE run manifest range differs")
        run_manifest = json.loads(model_manifest)
        require(run_manifest.get("truth_accessed") is False and run_manifest.get("scoring_performed") is False
                and run_manifest.get("marker_condition") == "BASELINE_EXCLUDING_EVALUATED_DNABR_RARE",
                "FLARE run manifest truth/selection/marker condition differs")
        first = body.splitlines()[0].decode().split("\t")
        lower = {name.lower() for name in first}
        require({"chromosome", "ref", "alt"} <= lower and bool({"position", "position_bp"} & lower),
                "marker inventory lacks exact chromosome/position/REF/ALT identity")
        require(not any("truth" in name.lower() or "ancestry_state" in name.lower() for name in first),
                "marker inventory exposes truth columns")
        rows = body.count(b"\n") - 1
        require(rows == int(run_manifest["audit"]["marker_count"]), "marker row count differs from FLARE run")
        if not object_path.exists():
            partial = objects / (entry["sha256"] + ".partial")
            partial.write_bytes(body); require(sha(partial) == entry["sha256"], "written marker object differs")
            partial.rename(object_path)
        case_path = logical / (task["task_id"] + ".common-markers.tsv")
        os.link(object_path, case_path)
        receipts.append({"task_id": task["task_id"], "case_id": task["case_id"], "role": task["role"],
                         "marker_count": rows, "columns": first, "path": str(case_path.resolve()),
                         "bytes": size, "sha256": entry["sha256"], "archive_uri": source["archive_uri"],
                         "archive_generation": source["archive_generation"], "tar_header_offset": offset,
                         "truth_read": False, "selection_performed": False})
    result = {"schema": "r02_lai_exact_flare_marker_inventory_receipt_v1",
              "status": "COMPLETE_AUTHENTICATED_EXISTING_FLARE_MARKER_LISTS",
              "direction": DIRECTION, "tasks": receipts, "tasks_complete": len(receipts),
              "unique_objects": len(list(objects.glob("*.tsv"))), "gcs_bytes_read": downloaded,
              "truth_read": False, "selection_performed": False, "evaluate_opened": False,
              "plan_sha256": sha(args.plan)}
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def stage_flare_receipts(args):
    plan = load(args.plan)
    require(plan.get("schema") == "r02_lai_training_inputs_plan_v1"
            and plan.get("direction") == DIRECTION and len(plan.get("tasks", [])) == 12,
            "one-direction LAI plan differs")
    output = Path(args.output_dir)
    require(not output.exists() and output.parent.is_dir(), "fresh output directory required")
    output.mkdir(mode=0o700); logical = output / "logical"; logical.mkdir()
    tasks = []; downloaded = 0
    for task in plan["tasks"]:
        flare = task["flare"]["primary"]
        body = gcs_cat(flare["receipt_uri"], flare["receipt_generation"], None,
                       args.gcloud_account, args.gcloud_project)
        downloaded += len(body)
        require(hashlib.sha256(body).hexdigest() == flare["receipt_sha256"],
                "FLARE projection receipt SHA-256 differs")
        receipt = json.loads(body)
        require(receipt.get("schema") == "r02_lai_flare_query_projection_receipt_v1"
                and receipt.get("probabilities_sha256") == flare["sha256"]
                and receipt.get("truth_used_for_projection") is False
                and receipt.get("selection_performed") is False,
                "FLARE projection receipt semantics differ")
        path = logical / (task["task_id"] + ".flare-receipt.json")
        path.write_bytes(body)
        tasks.append({"task_id": task["task_id"], "path": str(path.resolve()),
                      "sha256": flare["receipt_sha256"], "bytes": len(body),
                      "uri": flare["receipt_uri"], "generation": flare["receipt_generation"]})
    result = {"schema": "r02_lai_flare_projection_receipt_staging_v1",
              "status": "COMPLETE_12_PRIMARY_FLARE_RECEIPTS_AUTHENTICATED",
              "direction": DIRECTION, "tasks_complete": len(tasks), "tasks": tasks,
              "gcs_bytes_read": downloaded, "truth_read": False,
              "selection_performed": False, "evaluate_opened": False,
              "plan_sha256": sha(args.plan)}
    (output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def query_bounds(overlay_path):
    """Read only the first predictor grid from an authenticated NPZ.

    The producer receipt proves every query is valid.  The grid is repeated for
    each person, so the first stored row is sufficient to define the inclusive
    chromosome core without reading ancestry truth arrays.
    """
    with zipfile.ZipFile(overlay_path) as archive, archive.open("query_pos_bp.npy") as handle:
        require(handle.read(6) == b"\x93NUMPY", "query position array is not NPY")
        version = handle.read(2); require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "unsupported NPY version")
        width = 2 if version[0] == 1 else 4
        length = struct.unpack("<H" if width == 2 else "<I", handle.read(width))[0]
        header = ast.literal_eval(handle.read(length).decode("latin1"))
        require(header["descr"] == "<i8" and header["fortran_order"] is False
                and len(header["shape"]) == 2 and header["shape"][0] > 0 and header["shape"][1] > 1,
                "query position array shape or dtype differs")
        people, queries = header["shape"]
        row = handle.read(queries * 8)
        require(len(row) == queries * 8, "short first query grid")
        positions = struct.unpack("<" + "q" * queries, row)
        require(all(a < b for a, b in zip(positions, positions[1:])), "query grid is not strictly increasing")
        return {"first_bp": positions[0], "last_bp": positions[-1],
                "people": people, "queries_per_person": queries}


def open_deterministic_gzip(path):
    raw = Path(path).open("xb")
    return raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0)


def source_event_rows(path, chromosome):
    rows = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chromosome", "position_bp", "REF", "ALT", "counted_allele",
                    "source_rare_ac", "source_rare_an", "panel_ancestry", "decision"}
        require(required <= set(reader.fieldnames or ()), "source-minor table columns differ")
        for row in reader:
            if row["chromosome"] != chromosome or row["panel_ancestry"] != "ALL":
                continue
            if row["decision"] != "CONDITIONAL_INFERRED_HAPLOTYPE_CANDIDATE":
                continue
            key = (int(row["position_bp"]), row["REF"], row["ALT"])
            require(key not in rows, "duplicate source-minor event identity")
            counted = row["counted_allele"]
            require(counted in (row["REF"], row["ALT"]), "source-minor allele is neither REF nor ALT")
            ac, an = int(row["source_rare_ac"]), int(row["source_rare_an"])
            require(0 < ac <= an // 2, "source-minor AC/AN is invalid")
            rows[key] = {"counted_allele": "REF" if counted == row["REF"] else "ALT",
                         "source_minor_ac": ac, "source_an": an,
                         "source_catalogue_sha256": row["source_catalogue_sha256"]}
    require(rows, "source-minor candidate table has no admitted events")
    return rows


def materialize_catalogs(args):
    plan = load(args.plan); common_receipt = load(args.common_receipt)
    require(plan.get("schema") == "r02_lai_training_inputs_plan_v1"
            and plan.get("direction") == DIRECTION and len(plan.get("tasks", [])) == 12,
            "one-direction LAI plan differs")
    require(common_receipt.get("schema") == "r02_lai_exact_flare_marker_inventory_receipt_v1"
            and common_receipt.get("status") == "COMPLETE_AUTHENTICATED_EXISTING_FLARE_MARKER_LISTS"
            and common_receipt.get("truth_read") is False
            and common_receipt.get("selection_performed") is False,
            "authenticated primary FLARE marker receipt differs")
    common_by_task = {row["task_id"]: row for row in common_receipt["tasks"]}
    require(len(common_by_task) == 12, "primary FLARE marker receipt lacks tasks")
    source_root = Path(args.source_event_root)
    source_paths = {chrom: source_root / f"chr{chrom}" / "decision" / "candidate_alleles.tsv.gz"
                    for chrom in ("1", "22")}
    source_manifests = {chrom: source_paths[chrom].with_name("manifest.json") for chrom in source_paths}
    source_descriptors, source_rows = {}, {}
    for chrom, path in source_paths.items():
        manifest = load(source_manifests[chrom])
        expected = manifest["outputs_sha256"]["candidate_alleles.tsv.gz"]
        source_descriptors[chrom] = recorded_file(path, expected)
        source_descriptors[chrom]["manifest"] = descriptor(source_manifests[chrom])
        source_rows[chrom] = source_event_rows(path, chrom)
    output = Path(args.output_dir)
    require(not output.exists() and output.parent.is_dir(), "fresh catalogue output directory required")
    objects = output / "objects"; logical = output / "logical"
    output.mkdir(mode=0o700); objects.mkdir(); logical.mkdir()
    products = []
    common_fields = ("chromosome", "position_bp", "ref", "alt", "position_cm", "core",
                     "fit_alt_ac", "fit_an", "counted_allele", "marker_channel", "frequency_definition")
    event_fields = ("chromosome", "position_bp", "ref", "alt", "position_cm", "core",
                    "counted_allele", "source_minor_ac", "source_an", "copyable_in_case_fold",
                    "reference_African_ac", "reference_African_an",
                    "reference_European_ac", "reference_European_an",
                    "reference_Native_American_ac", "reference_Native_American_an",
                    "source_catalogue_sha256")
    for task in plan["tasks"]:
        task_id = task["task_id"]; chrom = task["chromosome"]
        marker = common_by_task.get(task_id); require(marker is not None, "task lacks exact FLARE markers")
        marker_path = Path(marker["path"]); recorded_file(marker_path, marker["sha256"])
        event_desc = task["inputs"]["event_candidate_table"]
        event_path = Path(event_desc["path"]); recorded_file(event_path, event_desc["sha256"])
        stats_dir = Path(args.fit_statistics_root) / task["fit_statistics_job"]
        stats_path = stats_dir / "fit-source-statistics.tsv.gz"
        summary_path = stats_dir / "fit-source-statistics.summary.json"
        summary = load(summary_path)
        require(summary.get("schema") == "r02_lai_fit_source_join_summary_v1"
                and summary.get("job_id") == task["fit_statistics_job"]
                and summary.get("truth_used") is False
                and summary.get("select_evaluate_read") is False,
                "FIT/source statistics summary differs")
        stats_desc = recorded_file(stats_path, summary["output"]["sha256"])
        bounds = query_bounds(task["inputs"]["query_overlay"]["path"])
        require(bounds["people"] == task["people"], "query-grid people differ")
        with marker_path.open(newline="") as handle:
            mr = csv.DictReader(handle, delimiter="\t"); markers = list(mr)
        require(len(markers) == marker["marker_count"], "primary marker row count differs")
        with event_path.open(newline="") as handle:
            er = csv.DictReader(handle, delimiter="\t"); events = list(er)
        require(events, "case event catalogue is empty")
        marker_index = {(int(row["position"]), row["ref"], row["alt"]): row for row in markers}
        event_index = {(int(row["position_bp"]), row["REF"], row["ALT"]): row for row in events}
        require(len(marker_index) == len(markers) and len(event_index) == len(events), "duplicate catalogue identity")
        require(set(event_index) <= set(source_rows[chrom]), "case event is absent from fixed DNABR source-minor catalogue")
        common_partial = output / (task_id + ".common.tsv.gz")
        event_partial = output / (task_id + ".event.tsv.gz")
        raw_c, gz_c = open_deterministic_gzip(common_partial)
        raw_e, gz_e = open_deterministic_gzip(event_partial)
        ctext = __import__("io").TextIOWrapper(gz_c, encoding="utf-8", newline="")
        etext = __import__("io").TextIOWrapper(gz_e, encoding="utf-8", newline="")
        cw = csv.DictWriter(ctext, fieldnames=common_fields, delimiter="\t", lineterminator="\n"); cw.writeheader()
        ew = csv.DictWriter(etext, fieldnames=event_fields, delimiter="\t", lineterminator="\n"); ew.writeheader()
        common_seen = []; event_seen = []
        with gzip.open(stats_path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (int(row["position_bp"]), row["REF"], row["ALT"])
                require(row["map_supported"] == "TRUE", "FIT/source row outside genetic map support")
                core = bounds["first_bp"] <= key[0] <= bounds["last_bp"]
                if key in marker_index:
                    m = marker_index[key]
                    cw.writerow({"chromosome": chrom, "position_bp": key[0], "ref": key[1], "alt": key[2],
                                 "position_cm": row["position_cm"], "core": str(core).lower(),
                                 "fit_alt_ac": row["fit_AC_ALT"], "fit_an": row["fit_AN_fully_called_diploid"],
                                 "counted_allele": "ALT", "marker_channel": m["channel"],
                                 "frequency_definition": m["frequency_definition"]})
                    common_seen.append(key)
                if key in event_index:
                    event = event_index[key]; source = source_rows[chrom][key]
                    require(event["counted_allele"] == source["counted_allele"],
                            "case and DNABR source-minor counted alleles differ")
                    out = {"chromosome": chrom, "position_bp": key[0], "ref": key[1], "alt": key[2],
                           "position_cm": row["position_cm"], "core": str(core).lower(),
                           **source, "copyable_in_case_fold": event["copyable_in_source_fold"]}
                    for ancestry in ("African", "European", "Native_American"):
                        an = int(row[f"reference_{ancestry}_AN_fully_called_diploid"])
                        alt_ac = int(row[f"reference_{ancestry}_AC_ALT"])
                        out[f"reference_{ancestry}_ac"] = alt_ac if source["counted_allele"] == "ALT" else an - alt_ac
                        out[f"reference_{ancestry}_an"] = an
                    ew.writerow(out); event_seen.append(key)
        ctext.flush(); ctext.detach(); gz_c.close(); raw_c.close()
        etext.flush(); etext.detach(); gz_e.close(); raw_e.close()
        require(common_seen == list(marker_index), "FLARE marker order/intersection differs from FIT/source statistics")
        require(event_seen == list(event_index), "event order/intersection differs from FIT/source statistics")
        task_dir = logical / task_id; task_dir.mkdir()
        task_products = []
        for kind, partial, rows in (("common", common_partial, len(common_seen)), ("event", event_partial, len(event_seen))):
            digest = sha(partial); object_path = objects / (digest + ".tsv.gz")
            if object_path.exists():
                require(sha(object_path) == digest, "deduplicated catalogue object differs"); partial.unlink()
            else:
                partial.rename(object_path)
            target = task_dir / (kind + ".tsv.gz"); os.link(object_path, target)
            task_products.append({"kind": kind, "path": str(target.resolve()), "sha256": digest,
                                  "bytes": target.stat().st_size, "rows": rows})
        products.append({"task_id": task_id, "case_id": task["case_id"], "role": task["role"],
                         "chromosome": chrom, "query_core": bounds, "catalogues": task_products,
                         "sources": {"markers": {k: marker[k] for k in ("path", "sha256", "bytes", "marker_count")},
                                     "case_events": recorded_file(event_path, event_desc["sha256"]),
                                     "fit_statistics": stats_desc,
                                     "fit_statistics_summary": descriptor(summary_path),
                                     "source_minor": source_descriptors[chrom]},
                         "truth_read": False, "selection_performed": False})
    receipt = {"schema": "r02_lai_common_event_catalogue_receipt_v1",
               "status": "COMPLETE_AUTHENTICATED_COMMON_EVENT_CORE_CATALOGUES",
               "direction": DIRECTION, "tasks_complete": len(products), "tasks": products,
               "unique_objects": len(list(objects.glob("*.tsv.gz"))),
               "common_rule": "EXACT_ORDERED_PRIMARY_FLARE_MARKERS_WITH_FOLD_A_FIT_ALT_AC_AN",
               "event_rule": "FIXED_DNABR_SOURCE_MINOR_EVENTS_WITH_SOURCE_AC_AN_AND_REFERENCE_ANCESTRY_AVAILABILITY",
               "core_rule": "INCLUSIVE_FIRST_TO_LAST_AUTHENTICATED_QUERY_GRID_BP",
               "truth_read": False, "selection_performed": False, "evaluate_opened": False,
               "plan_sha256": sha(args.plan), "common_receipt_sha256": sha(args.common_receipt)}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--inventory", required=True); p.add_argument("--fit-plan", required=True)
    p.add_argument("--reader-plan", required=True); p.add_argument("--flare-decision", required=True)
    p.add_argument("--flare-registry", required=True)
    p.add_argument("--catalogue-receipt")
    p.add_argument("--flare-receipt-staging")
    p.add_argument("--output", required=True); p.add_argument("--case-manifest-output")
    p.add_argument("--product-request-output"); p.set_defaults(func=build_plan)
    p = sub.add_parser("adapter-request")
    p.add_argument("--plan", required=True); p.add_argument("--task-id", required=True)
    p.add_argument("--explicit-batch", required=True); p.add_argument("--explicit-receipt", required=True)
    p.add_argument("--reference-receipt", required=True)
    p.add_argument("--flare-receipt", required=True); p.add_argument("--output", required=True)
    p.set_defaults(func=adapter_request)
    p = sub.add_parser("input-request")
    p.add_argument("--plan", required=True); p.add_argument("--task-id", required=True)
    p.add_argument("--channel-batch", required=True); p.add_argument("--channel-receipt", required=True)
    p.add_argument("--controls", required=True); p.add_argument("--output", required=True)
    p.set_defaults(func=input_request)
    p = sub.add_parser("prepare-input-request")
    p.add_argument("--plan", required=True); p.add_argument("--task-id", required=True)
    p.add_argument("--channel-batch", required=True); p.add_argument("--channel-receipt", required=True)
    p.add_argument("--output", required=True); p.set_defaults(func=prepare_input_request)
    p = sub.add_parser("extract-common")
    p.add_argument("--plan", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--gcloud-account", default="653458115080-compute@developer.gserviceaccount.com")
    p.add_argument("--gcloud-project", default="uspbr-242713")
    p.set_defaults(func=extract_common)
    p = sub.add_parser("stage-flare-receipts")
    p.add_argument("--plan", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--gcloud-account", default="653458115080-compute@developer.gserviceaccount.com")
    p.add_argument("--gcloud-project", default="uspbr-242713")
    p.set_defaults(func=stage_flare_receipts)
    p = sub.add_parser("materialize-catalogs")
    p.add_argument("--plan", required=True); p.add_argument("--common-receipt", required=True)
    p.add_argument("--fit-statistics-root", required=True); p.add_argument("--source-event-root", required=True)
    p.add_argument("--output-dir", required=True); p.set_defaults(func=materialize_catalogs)
    args = parser.parse_args(); result = args.func(args)
    print(json.dumps({"schema": result["schema"], "status": result.get("status", "WRITTEN")}, sort_keys=True))


if __name__ == "__main__":
    main()
